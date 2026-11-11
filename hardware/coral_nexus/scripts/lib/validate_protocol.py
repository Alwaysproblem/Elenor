"""Validate the C009 boundary-protocol drafts (stdlib + PyYAML).

Loads protocol/{cn_protocol_v1,channel_budgets,lifetime_bounds,csr_name_map}.yaml
and upstream_csr_occupied.json and returns every violation as a precise error.
Each class below maps to a C009 acceptance clause:

- structure: every message/field names an owner and a source; refs resolve.
- identity: TaskKey carries launch_generation == root_launch_generation;
  a Context's identity includes ctx_generation, so one slot reused with a new
  generation is distinguishable; PC never appears as a completion key.
- dedup: the semantic key is (txn_id, message_type), so a txn's HOLD and FINAL
  are different legal stages; the same (txn_id, type, stage) twice is a
  duplicate; an ACK must reference the request's (txn_id, message_type).
- stale generation: a message whose ctx_generation lags the live occupant is
  rejected (ABA guard).
- budgets/lifetimes: widths come from `2**w > max_in_flight + rate*lifetime`
  with every term in message COUNTS; adding a tick count to a count is
  rejected as a unit error; a domain with no sourced bound is not guessed.
- CSR: names are registered without addresses; an address supplied here, or a
  declared name colliding with an upstream-occupied one, is rejected.
- single Context: a domain whose window is 1 must still derive a width >= 1;
  a zero-width field is rejected.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROTOCOL_DIR = Path(__file__).resolve().parents[2] / "protocol"
_REQUIRED_MESSAGE_KEYS = ("name", "channel", "dir", "owner", "status", "source", "key")
_COUNT_UNITS = {"count", "msg_per_tick"}


def _load(path: Path) -> Any:
  return yaml.safe_load(path.read_text())


def load_all(directory: Path = PROTOCOL_DIR) -> dict[str, Any]:
  return {
    "protocol": _load(directory / "cn_protocol_v1.yaml"),
    "budgets": _load(directory / "channel_budgets.yaml"),
    "lifetimes": _load(directory / "lifetime_bounds.yaml"),
    "csr": _load(directory / "csr_name_map.yaml"),
    "upstream_csr": json.loads((directory / "upstream_csr_occupied.json").read_text()),
  }


def _resolve_ref(ref: str, budgets: dict) -> Any:
  """Resolve `channel_budgets.<ledger>.<name>` or `channel_budgets.rates.<name>`."""
  parts = ref.split(".")
  if parts[0] != "channel_budgets":
    raise KeyError(ref)
  if parts[1] == "rates":
    for item in budgets["rates"]:
      if item["name"] == parts[2]:
        return item
    raise KeyError(ref)
  for item in budgets["ledgers"][parts[1]].get("entries", []):
    if item["name"] == parts[2]:
      return item
  raise KeyError(ref)


def check_structure(protocol: dict, errors: list[str]) -> None:
  names: set[str] = set()
  for msg in protocol.get("messages", []):
    for key in _REQUIRED_MESSAGE_KEYS:
      if key not in msg:
        errors.append(f"message {msg.get('name', '?')}: missing required field {key!r}")
    if msg.get("name") in names:
      errors.append(f"message {msg['name']}: defined twice")
    names.add(msg.get("name", ""))
  channels = {c["name"] for c in protocol.get("channels", [])}
  for msg in protocol.get("messages", []):
    if msg.get("channel") not in channels:
      errors.append(f"message {msg.get('name')}: unknown channel {msg.get('channel')!r}")
    if "subqueue" in msg:
      allowed: list[str] = next(
        (c.get("subqueues", []) for c in protocol["channels"] if c["name"] == msg["channel"]), []
      )
      if msg["subqueue"] not in allowed:
        errors.append(f"message {msg['name']}: subqueue {msg['subqueue']!r} not on {msg['channel']}")
  for domain, body in protocol.get("identity_domains", {}).items():
    if "source" not in body:
      errors.append(f"identity domain {domain}: missing source")
    for field in body.get("fields", []):
      if "owner" not in field:
        errors.append(f"identity field {domain}.{field.get('name')}: missing owner")


def check_identity(protocol: dict, errors: list[str]) -> None:
  domains = protocol["identity_domains"]
  task = {f["name"]: f for f in domains["task_key"]["fields"]}
  if task.get("root_launch_generation", {}).get("equals") != "launch_generation":
    errors.append("task_key: root_launch_generation must be declared equal to launch_generation")
  ctx = {f["name"] for f in domains["context_key"]["fields"]}
  for required in ("tile_id", "ctx_slot", "ctx_generation"):
    if required not in ctx:
      errors.append(f"context_key: missing {required}")
  all_fields = {f["name"] for body in domains.values() for f in body["fields"]}
  if "pc" in all_fields or "program_counter" in all_fields:
    errors.append("identity: a PC field is present; completions may not be matched by PC")
  if "op_seq" in {f["name"] for f in domains["transport"]["fields"]}:
    errors.append("transport: op_seq must not be a transport counter")


def check_dedup_rules(protocol: dict, errors: list[str]) -> None:
  if protocol["dedup"]["semantic_key"] != ["txn_id", "message_type"]:
    errors.append("dedup: semantic key must be exactly [txn_id, message_type]")
  stages = {m["name"]: m.get("stage") for m in protocol["messages"]}
  if not (stages.get("TASK_COMMIT") == "HOLD" and stages.get("TASK_COMMIT_FINAL") == "FINAL"):
    errors.append("dedup: TASK_COMMIT/TASK_COMMIT_FINAL must be the distinct HOLD/FINAL stages")
  stage_fields = protocol["dedup"].get("stage_fields")
  if not isinstance(stage_fields, dict) or stage_fields.get("ACK") != "ack_for_message_type":
    errors.append("dedup: an ACK must be keyed by the request type it answers (stage_fields.ACK)")
  elif not all(isinstance(v, str) for v in stage_fields.values()):
    errors.append("dedup: stage_fields must name exactly one field per message")
  if protocol["dedup"].get("transport_replay_outcome") != "INCOMPLETE":
    errors.append("dedup: transport replay must resolve to INCOMPLETE")
  run_begin = next((m for m in protocol["messages"] if m["name"] == "RUN_BEGIN"), None)
  if run_begin is None or "program_epoch" in run_begin.get("key", []):
    errors.append("RUN_BEGIN must exist and must not key on program_epoch")


def check_csr(csr: dict, upstream: dict, errors: list[str]) -> None:
  occupied_names = {e["name"].lower() for e in upstream["occupied"]}
  occupied_addrs = {e["address"]: e["name"] for e in upstream["occupied"]}
  seen: set[str] = set()
  for item in csr["declared"]:
    name = item["name"]
    if name in seen:
      errors.append(f"csr {name}: declared twice")
    seen.add(name)
    if "address" in item:
      holder = occupied_addrs.get(item["address"])
      detail = f" (collides with upstream {holder})" if holder else ""
      errors.append(f"csr {name}: an address is assigned here{detail}; addresses come from csr_allocator")
    if name.lower().split(".")[-1] in occupied_names:
      errors.append(f"csr {name}: logical name collides with an upstream-occupied CSR")
    for key in ("owner", "scope", "source"):
      if key not in item:
        errors.append(f"csr {name}: missing {key}")


@dataclass(frozen=True)
class Width:
  name: str
  window: int
  bits: int


def derive_width(
  name: str, max_in_flight: int, rate: int, lifetime_ticks: int, units: dict[str, str]
) -> Width:
  """2**w > max_in_flight + rate * lifetime, all terms in message counts."""
  if units.get("max_in_flight") not in _COUNT_UNITS:
    raise ValueError(f"{name}: max_in_flight must be a count, got {units.get('max_in_flight')!r}")
  if units.get("rate") != "msg_per_tick":
    raise ValueError(f"{name}: rate must be msg_per_tick, got {units.get('rate')!r}")
  if units.get("lifetime") != "tick":
    raise ValueError(
      f"{name}: lifetime must be ticks multiplied by the rate, got {units.get('lifetime')!r}"
    )
  for label, value in (("max_in_flight", max_in_flight), ("rate", rate), ("lifetime", lifetime_ticks)):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
      raise ValueError(f"{name}: {label} must be a non-negative integer, got {value!r}")
  window = max_in_flight + rate * lifetime_ticks
  bits = max(1, math.ceil(math.log2(window + 1)))
  if 2**bits <= window:
    raise ValueError(f"{name}: derived {bits} bits cannot cover window {window}")
  if bits < 1:
    raise ValueError(f"{name}: zero-width field")
  return Width(name=name, window=window, bits=bits)


def _evidence_bounds(item: dict) -> str | None:
  """What a rate/lifetime term declares it bounds (None when undeclared)."""
  return item.get("bounds") or item.get("source", {}).get("bounds")


def derive_all(budgets: dict, lifetimes: dict) -> tuple[list[Width], list[str]]:
  """Derive widths ONLY for domains whose rate and lifetime are evidence for them.

  A term derives a domain's width only if it declares `bounds: <that domain>`.
  A constant that bounds something else (reset-drain threshold, task-commit
  rate, host issue width) is refused. A domain with no such evidence must say
  `derivable: false` with a reason and `needs`; it emits no width (fail closed).
  """
  widths: list[Width] = []
  errors: list[str] = []
  for dom in lifetimes["domains"]:
    name = dom["name"]
    if dom.get("derivable") is False:
      if "reason" not in dom:
        errors.append(f"lifetime {name}: not derivable but gives no reason")
      if not dom.get("needs"):
        errors.append(f"lifetime {name}: not derivable but lists no `needs` evidence")
      continue
    # A domain that claims to be derivable must supply domain-specific evidence.
    try:
      mif = _resolve_ref(dom["max_in_flight"]["ref"], budgets)
    except (KeyError, TypeError) as exc:
      errors.append(f"lifetime {name}: unresolved max_in_flight ref {exc}")
      continue
    rate = dom.get("rate")
    life = dom.get("lifetime_ticks")
    if not isinstance(rate, dict) or "value" not in rate:
      errors.append(f"lifetime {name}: no domain rate evidence; mark derivable: false")
      continue
    if not isinstance(life, dict) or "value" not in life:
      errors.append(f"lifetime {name}: no domain lifetime evidence; mark derivable: false")
      continue
    for label, item in (("rate", rate), ("lifetime", life), ("max_in_flight", mif)):
      source = item.get("source", {})
      if "file" not in source:
        errors.append(f"lifetime {name}: {label} has no sourced file")
    for label, item in (("rate", rate), ("lifetime", life)):
      declared = _evidence_bounds(item)
      if declared != name:
        errors.append(
          f"lifetime {name}: {label} bounds {declared!r}, not {name!r}; "
          "a constant for another quantity is not evidence for this domain"
        )
    if any(e.startswith(f"lifetime {name}:") for e in errors):
      continue
    try:
      widths.append(
        derive_width(
          name,
          mif["value"],
          rate["value"],
          life["value"],
          {"max_in_flight": mif["unit"], "rate": rate["unit"], "lifetime": life["unit"]},
        )
      )
    except ValueError as exc:
      errors.append(str(exc))
  return widths, errors


def check_budgets(budgets: dict, errors: list[str]) -> None:
  for ledger, body in budgets["ledgers"].items():
    for entry in body.get("entries", []):
      if "source" not in entry or "file" not in entry["source"] or "line" not in entry["source"]:
        errors.append(f"budget {ledger}.{entry.get('name')}: missing source file:line")
      if not isinstance(entry.get("value"), int) or entry["value"] < 1:
        errors.append(f"budget {ledger}.{entry.get('name')}: capacity must be a positive integer")
    if ledger == "channel_credit" and body.get("capacity") not in (None, "derived"):
      if "source" not in body:
        errors.append("budget channel_credit: capacity given without a source")
  for item in budgets["rates"]:
    if "source" not in item:
      errors.append(f"rate {item['name']}: missing source")


class ProtocolError(Exception):
  """Carries the explicit error code the contract names."""

  def __init__(self, code: str, detail: str) -> None:
    super().__init__(f"{code}: {detail}")
    self.code = code


class DedupTracker:
  """Reference semantics of the dedup contract, built from cn_protocol_v1.yaml.

  Semantic key = (txn_id, message_type). message_seq is transport-only: a gap
  or replay raises a transport error and is NEVER counted as a semantic effect.
  """

  def __init__(self, protocol: dict) -> None:
    self._types = {m["name"]: m for m in protocol["messages"]}
    self._semantic_key = tuple(protocol["dedup"]["semantic_key"])
    self._stage_fields: dict[str, str] = protocol["dedup"].get("stage_fields", {})
    self._seen: set[tuple] = set()
    self._requests: set[tuple] = set()
    self._next_seq: dict[tuple, int] = {}
    self.effects = 0

  def deliver(self, msg: dict) -> None:
    link = (msg["tile"], msg["channel"], msg["dir"])
    expected = self._next_seq.get(link, 0)
    seq = msg["message_seq"]
    if seq < expected:
      raise ProtocolError("TRANSPORT_REPLAY", f"message_seq {seq} < expected {expected} on {link}")
    if seq > expected:
      raise ProtocolError("INCOMPLETE", f"message_seq gap: got {seq}, expected {expected} on {link}")
    self._next_seq[link] = expected + 1
    mtype = msg["message_type"]
    if mtype not in self._types:
      raise ProtocolError("UNKNOWN_MESSAGE", mtype)
    if mtype == "ACK":
      ref = (msg["txn_id"], msg.get("ack_for_message_type"))
      if ref not in self._requests:
        raise ProtocolError("ACK_MISMATCH", f"ACK references unseen request {ref}")
    key = tuple(msg[k] if k != "message_type" else mtype for k in self._semantic_key)
    stage_field = self._stage_fields.get(mtype)
    if stage_field is not None:
      key += (msg[stage_field],)
    if key in self._seen:
      raise ProtocolError("DUPLICATE_STAGE", f"{key} already applied")
    self._seen.add(key)
    if "txn_id" in msg:
      self._requests.add((msg["txn_id"], mtype))
    self.effects += 1


def check_stale_generation(live_generation: int, message_generation: int) -> None:
  """ABA guard: a message from an older slot occupant is rejected with a code."""
  if message_generation < live_generation:
    raise ProtocolError("STALE_GENERATION", f"ctx_generation {message_generation} < live {live_generation}")


def make_identity(protocol: dict, domain: str, values: dict) -> tuple:
  """Build a canonical identity from the contract's own field list.

  Field set and the `equals` constraint come from cn_protocol_v1.yaml, so the
  YAML (not a hand-written tuple) decides what distinguishes two identities.
  Returns an ordered tuple of (field, value); missing/unknown fields and a
  violated `equals` constraint raise a coded ProtocolError.
  """
  spec = protocol["identity_domains"][domain]["fields"]
  declared = [f["name"] for f in spec]
  missing = [n for n in declared if n not in values]
  if missing:
    raise ProtocolError("IDENTITY_FIELD_MISSING", f"{domain}: {missing}")
  unknown = sorted(set(values) - set(declared))
  if unknown:
    raise ProtocolError("IDENTITY_FIELD_UNKNOWN", f"{domain}: {unknown}")
  for field in spec:
    other = field.get("equals")
    if other is not None and values[field["name"]] != values[other]:
      raise ProtocolError(
        "IDENTITY_EQUALITY", f"{domain}: {field['name']}={values[field['name']]} != {other}={values[other]}"
      )
  return tuple((name, values[name]) for name in declared)


def derive_index_width(name: str, count: int) -> Width:
  """Bits to index `count` slots, never zero: a single Context still gets 1 bit."""
  if not isinstance(count, int) or isinstance(count, bool) or count < 1:
    raise ValueError(f"{name}: slot count must be a positive integer, got {count!r}")
  bits = max(1, math.ceil(math.log2(count))) if count > 1 else 1
  return Width(name=name, window=count, bits=bits)


def check_phases_and_channels(protocol: dict, errors: list[str]) -> None:
  phases = protocol["phases"]
  if phases["order"] != ["P0", "P1", "P2", "P3", "P4"]:
    errors.append(f"phases: order must be P0..P4, got {phases['order']}")
  covered = {rule["phase"] for rule in phases["rules"]}
  for required in ("P1", "P3", "P4"):
    if required not in covered:
      errors.append(f"phases: no rule recorded for {required}")
  names = [c["name"] for c in protocol["channels"]]
  if sorted(names) != sorted(["CTRL", "EVENT", "PROGRAM", "MEMORY", "DEBUG"]):
    errors.append(f"channels: exactly CTRL/EVENT/PROGRAM/MEMORY/DEBUG expected, got {names}")
  if "STREAM" in names:
    errors.append("channels: STREAM must be a MEMORY sub-queue, not a sixth channel")


def validate(directory: Path = PROTOCOL_DIR) -> dict:
  data = load_all(directory)
  errors: list[str] = []
  check_structure(data["protocol"], errors)
  check_identity(data["protocol"], errors)
  check_dedup_rules(data["protocol"], errors)
  check_phases_and_channels(data["protocol"], errors)
  check_budgets(data["budgets"], errors)
  check_csr(data["csr"], data["upstream_csr"], errors)
  widths, width_errors = derive_all(data["budgets"], data["lifetimes"])
  errors.extend(width_errors)
  return {
    "schema": "cn_protocol_report/v1",
    "valid": not errors,
    "errors": errors,
    "widths": {w.name: {"window": w.window, "bits": w.bits} for w in widths},
    "messages": len(data["protocol"]["messages"]),
  }


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dir", default=str(PROTOCOL_DIR))
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  report = validate(Path(args.dir))
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / "protocol_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  if not report["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
