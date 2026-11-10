"""Validate the C010 native owner / ROB tag / retire contracts.

Two kinds of check, neither of which is "grep for OwnerTag":

1. Structured consumer behaviour. `RobPartition`/`RobBank` decode a completion
   by its FULL key and refuse everything else, so a result for a reused slot
   cannot be read as the new op. Tests drive legal and mismatched samples
   through it.
2. Source anchors. Every upstream line the YAML cites is read back from the
   pinned tree and must contain what the YAML claims there, so a drifted line
   number or a wrong quotation is a failure rather than a silent citation.

Widths with no sourced bound stay `unresolved`; an integer literal on one of
those fields is rejected as an invented width.
"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB))

from validate_protocol import ProtocolError, derive_index_width, load_all  # noqa: E402

H = LIB.parents[1]
REPO = LIB.parents[3]
UPSTREAM = H / "upstream" / "coralnpu"
DOC = H / "rtl" / "scalar" / "doc"
PROTOCOL = H / "protocol"

_DOCS = ("rob_tag", "ctx_ctrl", "retire_record", "oracle_requirements", "perctx_issue")
# Fields whose width has no sourced bound anywhere; a numeric rule is invention.
_UNSOURCED_WIDTH_FIELDS = {"ctx_generation", "owner_op_seq", "subop_id", "slot_generation"}
_FIELD_KEYS = ("name", "width_rule", "owner", "consumers", "required")


class OwnerError(ProtocolError):
  """Coded error from the reference ROB (inherits the protocol error shape)."""


def log2_ceil(n: int) -> int:
  """Chisel log2Ceil: 1 -> 0, 2 -> 1, 16 -> 4, 66 -> 7."""
  if n < 1:
    raise ValueError(f"log2_ceil needs n >= 1, got {n}")
  return (n - 1).bit_length()


def tag_layout(context_count: int, rob_capacity: int) -> dict[str, Any]:
  """Layout implied by rob_tag.yaml for one configuration.

  One Context still gets a real ctx field (a constant), never a zero-width one.
  """
  ctx = derive_index_width("ctx_slot", context_count)
  return {
    "context_count": context_count,
    "ctx_bits": ctx.bits,
    "ctx_slot_constant": context_count == 1,
    "rob_slot_bits": log2_ceil(rob_capacity),
    "partitions": context_count,
  }


@dataclass(frozen=True)
class Tag:
  ctx_slot: int
  ctx_generation: int
  rob_slot: int
  slot_generation: int
  owner_op_seq: int


@dataclass
class _Entry:
  owner_op_seq: int
  generation: int
  completed: bool = False


@dataclass
class RobPartition:
  """One Context's private ROB: completion needs the complete key."""

  ctx_slot: int
  ctx_generation: int
  capacity: int
  _entries: list[_Entry | None] = field(init=False)
  _slot_gen: list[int] = field(init=False)
  _head: int = 0
  _tail: int = 0
  _count: int = 0

  def __post_init__(self) -> None:
    self._entries = [None] * self.capacity
    self._slot_gen = [0] * self.capacity

  @property
  def live(self) -> int:
    return self._count

  def allocate(self, owner_op_seq: int) -> Tag:
    if self._count == self.capacity:
      raise OwnerError("ROB_FULL", f"ctx {self.ctx_slot} partition of {self.capacity} is full")
    slot = self._tail
    self._entries[slot] = _Entry(owner_op_seq, self._slot_gen[slot])
    self._tail = (self._tail + 1) % self.capacity
    self._count += 1
    return Tag(self.ctx_slot, self.ctx_generation, slot, self._slot_gen[slot], owner_op_seq)

  def complete(self, tag: Tag) -> None:
    if tag.ctx_slot != self.ctx_slot:
      raise OwnerError("WRONG_CONTEXT", f"tag ctx {tag.ctx_slot} != partition {self.ctx_slot}")
    if tag.ctx_generation != self.ctx_generation:
      raise OwnerError(
        "STALE_CTX_GENERATION", f"tag gen {tag.ctx_generation} != live {self.ctx_generation}"
      )
    if not 0 <= tag.rob_slot < self.capacity:
      raise OwnerError("SLOT_OUT_OF_RANGE", f"rob_slot {tag.rob_slot}")
    entry = self._entries[tag.rob_slot]
    if entry is None:
      raise OwnerError("SLOT_NOT_LIVE", f"slot {tag.rob_slot} holds no op")
    if tag.slot_generation != entry.generation:
      raise OwnerError(
        "STALE_SLOT_GENERATION",
        f"slot {tag.rob_slot}: tag gen {tag.slot_generation} != live {entry.generation}",
      )
    if tag.owner_op_seq != entry.owner_op_seq:
      raise OwnerError("OP_SEQ_MISMATCH", f"{tag.owner_op_seq} != {entry.owner_op_seq}")
    if entry.completed:
      raise OwnerError("DUPLICATE_COMPLETION", f"slot {tag.rob_slot} already completed")
    entry.completed = True

  def retire(self) -> int | None:
    """Retire the head in order; returns its owner_op_seq or None if not ready."""
    if self._count == 0:
      return None
    entry = self._entries[self._head]
    if entry is None or not entry.completed:
      return None
    self._entries[self._head] = None
    self._slot_gen[self._head] += 1  # the NEXT occupant of this slot is a new generation
    self._head = (self._head + 1) % self.capacity
    self._count -= 1
    return entry.owner_op_seq


class RobBank:
  """Static private partitions, one per Context slot (W2-D02)."""

  def __init__(self, context_count: int, capacity: int) -> None:
    derive_index_width("ctx_slot", context_count)  # rejects count < 1
    self.context_count = context_count
    self.parts = [RobPartition(i, 1, capacity) for i in range(context_count)]

  def allocate(self, ctx_slot: int, owner_op_seq: int) -> Tag:
    return self._part(ctx_slot).allocate(owner_op_seq)

  def complete(self, tag: Tag) -> None:
    self._part(tag.ctx_slot).complete(tag)

  def reuse_context(
    self, ctx_slot: int, *, completion_consumed: bool, obligations_closed: bool | None = None
  ) -> int:
    """Hand the slot to a new Task; every precondition is named when it fails."""
    part = self._part(ctx_slot)
    closed = (part.live == 0) if obligations_closed is None else obligations_closed
    missing = []
    if not closed:
      missing.append("old obligations still open")
    if not completion_consumed:
      missing.append("TASK_COMPLETION not consumed")
    if missing:
      raise OwnerError("REUSE_BLOCKED", "; ".join(missing))
    part.ctx_generation += 1
    return part.ctx_generation

  def _part(self, ctx_slot: int) -> RobPartition:
    if not 0 <= ctx_slot < self.context_count:
      raise OwnerError("CTX_OUT_OF_RANGE", f"ctx_slot {ctx_slot} not in 0..{self.context_count - 1}")
    return self.parts[ctx_slot]


# --------------------------------------------------------------------------
# Source anchors
# --------------------------------------------------------------------------
def _source_line(file: str, line: int) -> str | None:
  for root in (UPSTREAM, REPO):
    path = root / file
    if path.is_file():
      lines = path.read_text().splitlines()
      return lines[line - 1] if 0 < line <= len(lines) else None
  return None


def check_anchor(errors: list[str], where: str, source: dict, needle: str) -> None:
  text = _source_line(source["file"], source["line"])
  if text is None:
    errors.append(f"{where}: cannot read {source['file']}:{source['line']}")
  elif needle not in text:
    errors.append(f"{where}: {source['file']}:{source['line']} does not contain {needle!r}")


def _stall_reason_names() -> set[str]:
  tree = ast.parse((REPO / "pipeline_validator" / "pmu.py").read_text())
  for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "StallReason":
      return {
        t.id
        for stmt in node.body
        if isinstance(stmt, ast.Assign)
        for t in stmt.targets
        if isinstance(t, ast.Name)
      }
  return set()


# --------------------------------------------------------------------------
# File checks
# --------------------------------------------------------------------------
def load_docs(directory: Path = DOC) -> dict[str, Any]:
  return {name: yaml.safe_load((directory / f"{name}.yaml").read_text()) for name in _DOCS}


def _all_fields(docs: dict[str, Any]) -> list[tuple[str, dict]]:
  rob, ctrl, retire = docs["rob_tag"], docs["ctx_ctrl"], docs["retire_record"]
  groups = [
    ("rob_tag.owner_tag", rob["owner_tag"]["fields"]),
    ("rob_tag.rob_tag", rob["rob_tag"]["fields"]),
    ("ctx_ctrl.common", ctrl["fields"]["common"]),
    ("retire_record", retire["fields"]),
  ]
  return [(where, f) for where, fields in groups for f in fields]


def check_field_schema(docs: dict[str, Any], errors: list[str]) -> None:
  for where, f in _all_fields(docs):
    for key in _FIELD_KEYS:
      if key == "consumers" and f.get("consumers"):
        continue
      if key in f and f[key] not in (None, "", []):
        continue
      errors.append(f"{where}.{f.get('name', '?')}: missing or empty {key!r}")


def check_width_rules(docs: dict[str, Any], errors: list[str]) -> None:
  seen: dict[str, str] = {}
  for where, f in _all_fields(docs):
    name, rule = f.get("name"), str(f.get("width_rule"))
    if name in _UNSOURCED_WIDTH_FIELDS and rule.strip().isdigit():
      errors.append(f"{where}.{name}: width {rule} is a literal but no source bounds it (invented width)")
    if name in seen and seen[name] != rule:
      errors.append(f"field {name}: width_rule {seen[name]!r} conflicts with {rule!r} in {where}")
    seen.setdefault(name, rule)


def check_replaced_keys(docs: dict[str, Any], errors: list[str]) -> None:
  for item in docs["rob_tag"]["upstream_keys_replaced"]:
    check_anchor(errors, f"upstream_keys_replaced.{item['id']}", item["source"], item["upstream"])


def check_parameters(docs: dict[str, Any], errors: list[str]) -> dict[str, int]:
  values: dict[str, int] = {}
  for item in docs["rob_tag"]["parameters"]:
    anchor = item.get("anchor", item["name"])
    check_anchor(errors, f"parameters.{item['name']}", item["source"], anchor)
    text = _source_line(item["source"]["file"], item["source"]["line"]) or ""
    if str(item["value"]) not in text:
      errors.append(f"parameters.{item['name']}: value {item['value']} not on its cited line")
    values[item["name"]] = int(item["value"])
  return values


def check_conflicts(docs: dict[str, Any], params: dict[str, int], errors: list[str]) -> None:
  float_regs = params["floatRegCount"]
  rob_bits = log2_ceil(params["retirementBufferSize"])
  idx_bits_no_float = log2_ceil(params["scalarRegCount"] + params["rvvRegCount"] + 2)
  idx_bits_float = log2_ceil(params["scalarRegCount"] + float_regs + params["rvvRegCount"] + 2)
  for item in docs["rob_tag"]["conflicts"]:
    if item["id"] == "rob_tag_vs_regidx":
      right = item["right"]["source"]
      check_anchor(errors, "conflicts.rob_tag_vs_regidx", right, "retirementBufferIdxWidth")
      if rob_bits in (idx_bits_no_float, idx_bits_float):
        errors.append(
          f"rob_tag width {rob_bits} equals retirementBufferIdxWidth; they are different quantities"
        )
    elif item["id"] == "chisel_rob_tag_vs_rvv_define":
      check_anchor(errors, "conflicts.chisel_rob_tag", item["left"]["source"], "rob_tag")
      check_anchor(errors, "conflicts.rvv_define", item["right"]["source"], "RTBUF_DEPTH")
      sv_width = params["rvv_rob_tag_width_fallback"]
      if rob_bits != sv_width:
        errors.append(f"ROB tag width mismatch: Chisel {rob_bits} bits vs RVV ROB_TAG_WIDTH {sv_width}")
    else:
      errors.append(f"conflicts.{item['id']}: unknown conflict rule")


def check_csr_commit(docs: dict[str, Any], upstream_csr: dict, errors: list[str]) -> None:
  known = {e["name"] for e in upstream_csr["occupied"]}
  commit = docs["ctx_ctrl"]["csr_commit"]
  groups = {
    "dispatch_beat": commit["dispatch_beat"]["csrs"],
    "retire_commit": commit["retire_commit"]["csrs"],
    "flags_delta": commit["flags_delta"]["csrs"],
  }
  unclassified = commit["dispatch_beat"].get("unclassified", {})
  if unclassified:
    groups["unclassified"] = unclassified["csrs"]
    if unclassified.get("status") != "open" or not unclassified.get("reason"):
      errors.append("csr_commit.dispatch_beat.unclassified must stay status: open with a reason")
  owner_of: dict[str, str] = {}
  for group, names in groups.items():
    for name in names:
      if name not in known:
        errors.append(f"csr_commit.{group}: {name} is not an upstream CSR")
      if name in owner_of:
        errors.append(f"csr_commit: {name} listed in both {owner_of[name]} and {group}")
      owner_of[name] = group
  gpr_time = commit["gpr_writeback"]["csr_commit_time"]
  if gpr_time != "not applicable; GPR completion never implies CSR commit":
    errors.append("csr_commit.gpr_writeback must keep GPR completion separate from CSR commit time")


def check_ops_and_control(docs: dict[str, Any], protocol: dict, errors: list[str]) -> None:
  ctrl = docs["ctx_ctrl"]
  owners = {op["name"]: op["record_owner"] for op in ctrl["ops"]}
  expect = {"WAIT": "W6", "WAITALL": "W6", "SIGNAL_PHASE": "W6", "REGION_START": "W3", "REGION_END": "W3"}
  for name, owner in expect.items():
    if owners.get(name) != owner:
      errors.append(f"ctx_ctrl op {name}: record owner {owners.get(name)!r}, expected {owner}")
  wire = {m["name"] for m in protocol["messages"]}
  if "EXEC_DRAINED" in wire:
    errors.append("EXEC_DRAINED is an internal W2->W6 notice and must not be a boundary message")
  sep = ctrl["control_separation"]
  for kind in ("park", "cancel", "fail_stop"):
    if kind not in sep:
      errors.append(f"ctx_ctrl.control_separation: {kind} missing")
  if sep.get("park", {}).get("uses_core_clock_gate") is not False:
    errors.append("park must not use the core clock gate")
  if not ctrl["reuse_rule"]["context_slot_reuse_requires"]:
    errors.append("ctx_ctrl.reuse_rule must list its preconditions")


def check_retire_record(docs: dict[str, Any], params: dict[str, int], errors: list[str]) -> None:
  rec = docs["retire_record"]
  if rec["lanes"]["value"] != params["retirementLanes"]:
    errors.append("retire_record.lanes disagrees with retirementLanes")
  check_anchor(errors, "retire_record.lanes", rec["lanes"]["source"], "retirementLanes")
  for f in rec["fields"]:
    if f.get("origin") == "upstream":
      check_anchor(errors, f"retire_record.{f['name']}", f["source"], f"{f['name']}_i")
  if rec["mode_requirement"]["retirement_buffer_mode"] != "full":
    errors.append("retire_record must require a full RetirementBuffer")
  check_anchor(errors, "retire_record.mode", rec["mode_requirement"]["source"], "mini")
  check_anchor(errors, "retire_record.mode.also", rec["mode_requirement"]["also"], "dataWidth")
  if "W7" not in rec["read_only_for"]:
    errors.append("retire_record must be read-only for W7")


def check_issue(docs: dict[str, Any], params: dict[str, int], errors: list[str]) -> None:
  issue = docs["perctx_issue"]
  if issue["bundle"]["lanes"] != params["instructionLanes"]:
    errors.append("perctx_issue.bundle.lanes disagrees with instructionLanes")
  check_anchor(errors, "perctx_issue.bundle", issue["bundle"]["source"], "instructionLanes")
  check_anchor(errors, "perctx_issue.selection", issue["selection"]["source"], "_select_context")
  rules = {r["id"] for r in issue["bundle"]["rules"]}
  for needed in ("same_context", "prefix_only", "partial_accept", "owner_op_seq_advance"):
    if needed not in rules:
      errors.append(f"perctx_issue.bundle: rule {needed} missing")
  declared = set(issue["signals"]["stall_reasons"]["existing_validator"])
  actual = _stall_reason_names() - {"NONE"}
  if declared != actual:
    errors.append(
      f"perctx_issue stall reasons differ from pmu.StallReason: "
      f"missing {sorted(actual - declared)}, extra {sorted(declared - actual)}"
    )
  if set(issue["epochs"]) < {"fetch_epoch", "ctx_generation", "separation"}:
    errors.append("perctx_issue.epochs must keep fetch_epoch and ctx_generation separate")


def check_oracle(docs: dict[str, Any], errors: list[str]) -> None:
  oracle = docs["oracle_requirements"]
  if oracle["input_contract"].get("per_context") is not True:
    errors.append("oracle input must be per Context")
  rule_ids = {r["id"] for r in oracle["comparison_rules"]}
  for needed in ("key_match", "no_pc_only_pairing", "flags_at_retire", "mode"):
    if needed not in rule_ids:
      errors.append(f"oracle comparison rule {needed} missing")
  if oracle["output_contract"]["retire_stream"]["schema_ref"] != "retire_record.yaml":
    errors.append("oracle retire stream must reference retire_record.yaml")


def check_field_request(
  request: dict, protocol: dict, docs: dict[str, Any], errors: list[str]
) -> None:
  catalog = {m["name"]: m for m in protocol["messages"]}
  for item in request["requests"]:
    msg = catalog.get(item["message"])
    if msg is None:
      errors.append(f"field_request {item['message']}: not in the protocol catalog")
      continue
    if msg["owner"] != "W2":
      errors.append(f"field_request {item['message']}: owner is {msg['owner']}, not W2")
    if msg["dir"] != item["direction"]:
      errors.append(f"field_request {item['message']}: direction {item['direction']} != {msg['dir']}")
    for needed in item.get("fields_needed", []):
      stem = Path(needed["from"]).stem
      doc = docs.get(stem)
      if doc is None:
        errors.append(f"field_request {item['message']}.{needed['name']}: unknown doc {needed['from']}")
        continue
      node: Any = doc
      for part in needed["key"].split("."):
        node = node.get(part) if isinstance(node, dict) else None
      names = {f["name"] for f in node} if isinstance(node, list) else set()
      if needed["name"] not in names:
        errors.append(
          f"field_request {item['message']}.{needed['name']}: not found at {needed['from']}:{needed['key']}"
        )
  ctx_fields = {f["name"] for f in protocol["identity_domains"]["context_key"]["fields"]}
  for name in ("ctx_slot", "ctx_generation"):
    if name not in ctx_fields:
      errors.append(f"protocol context_key lacks {name} that W2 relies on")


def validate(docs_dir: Path = DOC, protocol_dir: Path = PROTOCOL) -> dict:
  docs = load_docs(docs_dir)
  proto = load_all(protocol_dir)
  request = yaml.safe_load((protocol_dir / "field_requests" / "w2.yaml").read_text())
  errors: list[str] = []
  check_field_schema(docs, errors)
  check_width_rules(docs, errors)
  check_replaced_keys(docs, errors)
  params = check_parameters(docs, errors)
  if {"floatRegCount", "retirementBufferSize", "scalarRegCount", "rvvRegCount"} <= params.keys():
    check_conflicts(docs, params, errors)
  check_csr_commit(docs, proto["upstream_csr"], errors)
  check_ops_and_control(docs, proto["protocol"], errors)
  check_retire_record(docs, params, errors)
  check_issue(docs, params, errors)
  check_oracle(docs, errors)
  check_field_request(request, proto["protocol"], docs, errors)
  open_widths = sorted({f["name"] for _, f in _all_fields(docs) if f.get("width_rule") == "unresolved"})
  return {
    "schema": "cn_native_owner_report/v1",
    "valid": not errors,
    "errors": errors,
    "open_widths": open_widths,
    "layouts": {n: tag_layout(n, params.get("retirementBufferSize", 16)) for n in (1, 2)},
  }


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  report = validate()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / "native_owner_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  if not report["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
