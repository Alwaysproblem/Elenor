"""Freeze checker for the boundary protocol (C017 collect/verify; C020 seals).

* ``collect`` records the sha256 of every input and generated artifact with state ``collected-not-frozen``.
* ``verify_generated`` re-renders every generated artifact in memory and compares bytes with disk, so a
  hand-edited or stale generated file fails even when no record exists.
* ``verify`` re-hashes a record against the tree.
* ``seal`` (C020) writes the ONE joint ``PROTOCOL_FREEZE.json``: state ``frozen``, every required input and
  generated artifact hashed, its own digest nowhere inside itself, and no commit sha of itself required.
  A sealed file is never appended to: sealing again with identical content is idempotent, different content
  requires ``--bump`` (``protocol_seq + 1``; the previous seal is archived, never reused), and sealing with
  a required input missing or a half-covered set is refused.

The seal record is byte-pinned (prettier must not rewrite it); its correctness is decided by re-hashing and
re-rendering, never by trusting its own fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

H = Path(__file__).resolve().parents[1]
REPO = H.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_protocol as gp  # noqa: E402

STATES = ("collected-not-frozen", "frozen")
SEAL_PATH = H / "protocol/PROTOCOL_FREEZE.json"
W4_CONSTRAINTS = H / "configs/w4_memory_constraints.yaml"
W4_DECISIONS = H / "configs/w4_capacity_decisions.yaml"
DECISIONS_SCHEMA = "cn_w4_capacity_decisions/v1"
DECISION_TEXT_FIELDS = ("decided_by", "rationale", "evidence")

# Capacity budgets (NOT widths) the seal requires CLOSED with a sourced value. A width may legally stay
# declared-underivable (C009 fail-closed rule; it rides a 64-bit transport container); a capacity budget
# left on the unresolved list is an OPEN budget, and the W1 freeze rule forbids sealing over it
# ("a missing source fails the freeze; an empty TBD placeholder is forbidden"). Closing one means
# recording the decision in configs/w4_capacity_decisions.yaml -- never a number invented here.
REQUIRED_CLOSED_BUDGETS = ("per_ctx_outstanding_credit_cap", "response_record_budget")

# The required joint-freeze input set (C009-C019 producer fragments) lives HERE, in the checker, so
# deleting an entry from any YAML cannot silently narrow the seal. Relative to the repo root.
REQUIRED_INPUTS = (
  # C009 boundary identities / budgets / CSR names
  "hardware/coral_nexus/protocol/cn_protocol_v1.yaml",
  "hardware/coral_nexus/protocol/wire_schema_v1.yaml",
  "hardware/coral_nexus/protocol/channel_budgets.yaml",
  "hardware/coral_nexus/protocol/lifetime_bounds.yaml",
  "hardware/coral_nexus/protocol/csr_name_map.yaml",
  "hardware/coral_nexus/protocol/upstream_csr_occupied.json",
  # C010 native owner / retirement contracts
  "hardware/coral_nexus/rtl/scalar/doc/rob_tag.yaml",
  "hardware/coral_nexus/rtl/scalar/doc/retire_record.yaml",
  "hardware/coral_nexus/rtl/scalar/doc/oracle_requirements.yaml",
  "hardware/coral_nexus/rtl/scalar/doc/ctx_ctrl.yaml",
  "hardware/coral_nexus/rtl/scalar/doc/perctx_issue.yaml",
  # C011/C012 numeric + VME declaration ABI
  "hardware/coral_nexus/protocol/w3_numeric_profile_v1.yaml",
  "hardware/coral_nexus/protocol/w3_vme_decl_abi_v1.json",
  "hardware/coral_nexus/protocol/w3_vme_mode_tables_v1.json",
  # C013 memory contract + envelope
  "hardware/coral_nexus/protocol/w4_memory.contract.yaml",
  "hardware/coral_nexus/configs/w4_memory_constraints.yaml",
  "hardware/coral_nexus/configs/w4_capacity_decisions.yaml",
  # C014 package contract
  "hardware/coral_nexus/protocol/w5_package.contract.yaml",
  # C015/C016 event, terminal-reset and trace contracts
  "hardware/coral_nexus/protocol/w6_event_facility.contract.yaml",
  "hardware/coral_nexus/protocol/w6_terminal_reset.contract.yaml",
  "hardware/coral_nexus/verification/contracts/checker_verdict_contract_v1.yaml",
  "hardware/coral_nexus/verification/contracts/log_record_contract_v1.yaml",
  "hardware/coral_nexus/verification/contracts/perf_gates_contract_v1.yaml",
  "hardware/coral_nexus/verification/contracts/replay_bundle_contract_v1.yaml",
  "hardware/coral_nexus/verification/contracts/trace_window_contract_v1.yaml",
  # field requests of every package
  *(f"hardware/coral_nexus/protocol/field_requests/w{i}.yaml" for i in range(2, 8)),
  # C018 measured resources (committed evidence record)
  "hardware/coral_nexus/verification/kernel_bootstrap/kernel_measurement.json",
)

# Everything generated the seal covers. Relative to the repo root.
REQUIRED_GENERATED = (
  "hardware/coral_nexus/protocol/generated/cn_layout.json",
  "hardware/coral_nexus/protocol/generated/cn_messages.h",
  "hardware/coral_nexus/protocol/generated/cn_offsets.svh",
  "pipeline_validator/cosim/wire.py",
  "hardware/coral_nexus/configs/cs1_c1.yaml",
  "hardware/coral_nexus/configs/cs1_c2.yaml",
  "hardware/coral_nexus/configs/params.d/coral_cs1.json",
  "hardware/coral_nexus/rtl/targets.bzl",
)


def open_budgets() -> list[str]:
  """Required capacity budgets whose DECISION is not recorded in ``configs/w4_capacity_decisions.yaml``.

  A capacity with no derivable bound is a design decision, so it is closed by an external, reviewed,
  hash-sealed config and never by a number invented in code. Fail closed in every direction: a missing
  file, a missing or extra decision, ``null``, zero, a bool or non-integer value, or an empty
  decided_by / rationale / evidence all leave the budget open. The w4 constraints file keeps listing the
  two as underivable; the decision does not claim they were derived.
  """
  import yaml

  try:
    doc = yaml.safe_load(W4_DECISIONS.read_text())
  except (OSError, yaml.YAMLError):
    return list(REQUIRED_CLOSED_BUDGETS)
  decisions = doc.get("decisions") if isinstance(doc, dict) else None
  if not isinstance(decisions, dict) or doc.get("schema") != DECISIONS_SCHEMA:
    return list(REQUIRED_CLOSED_BUDGETS)
  state: list[str] = []
  for name in REQUIRED_CLOSED_BUDGETS:
    entry = decisions.get(name)
    if not isinstance(entry, dict) or not _decision_closed(entry):
      state.append(name)
  extra = sorted(set(decisions) - set(REQUIRED_CLOSED_BUDGETS))
  return state + [f"{n} (not a required budget)" for n in extra]


def _decision_closed(entry: dict[str, Any]) -> bool:
  value = entry.get("value")
  if isinstance(value, bool) or not isinstance(value, int) or value < 1:
    return False
  if entry.get("unit") != "entries":
    return False
  return all(isinstance(entry.get(k), str) and entry[k].strip() for k in DECISION_TEXT_FIELDS)


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
  return str(path.resolve().relative_to(REPO))


def collect(directory: Path = gp.PROTOCOL_DIR) -> dict[str, Any]:
  """Hash inputs and generated outputs. Generation must already be consistent (verify_generated)."""
  errors = verify_generated(directory)
  if errors:
    raise gp.GenError(errors)
  inputs = {f"hardware/coral_nexus/protocol/{n}": _sha(directory / n) for n in gp._INPUTS}
  outputs = {_rel(p): _sha(p) for p in sorted(gp.render_all(gp.build_model(directory)))}
  return {
    "schema": "cn_protocol_freeze_record/v1",
    "state": "collected-not-frozen",
    "inputs": inputs,
    "generated": outputs,
  }


def verify_generated(
  directory: Path = gp.PROTOCOL_DIR, on_disk: dict[Path, Path] | None = None
) -> list[str]:
  """Re-render in memory and compare with disk: catches hand edits and stale outputs.

  `on_disk` maps a default output path to the file to compare instead (tests use a private copy).
  """
  try:
    rendered = gp.render_all(gp.build_model(directory))
  except gp.GenError as exc:
    return [f"inputs do not generate: {e}" for e in exc.errors]
  errors: list[str] = []
  for path, text in rendered.items():
    actual = (on_disk or {}).get(path, path)
    if not actual.is_file():
      errors.append(f"{_rel(path)}: generated file is missing")
    elif actual.read_text() != text:
      errors.append(f"{_rel(path)}: differs from a fresh generation (hand-edited or stale)")
  return errors


def verify(record: dict[str, Any], root: Path = REPO) -> list[str]:
  errors: list[str] = []
  if record.get("schema") != "cn_protocol_freeze_record/v1":
    errors.append("record: unknown schema")
  if record.get("state") not in STATES:
    errors.append(f"record: state {record.get('state')!r} is not one of {STATES}")
  for section in ("inputs", "generated"):
    entries = record.get(section)
    if not isinstance(entries, dict) or not entries:
      errors.append(f"record.{section}: missing or empty")
      continue
    for rel, digest in sorted(entries.items()):
      path = root / rel
      if not path.is_file():
        errors.append(f"{rel}: listed in the record but missing")
      elif _sha(path) != digest:
        errors.append(f"{rel}: sha256 differs from the record")
  return errors


def _seal_body(inputs: dict[str, str], generated: dict[str, str], protocol_seq: int) -> dict[str, Any]:
  return {
    "schema": "cn_protocol_freeze/v1",
    "state": "frozen",
    "protocol_seq": protocol_seq,
    "sealed_by": "hardware/coral_nexus/protocol/freeze_check.py seal",
    "note": "one-shot joint seal of CN-G1; new content needs a new seal (protocol_seq+1, old archived)",
    "inputs": dict(sorted(inputs.items())),
    "generated": dict(sorted(generated.items())),
  }


def missing_required(rel_paths: tuple[str, ...]) -> list[str]:
  return [rel for rel in rel_paths if not (REPO / rel).is_file()]


def seal(out: Path | None = None, *, bump: bool = False) -> dict[str, Any]:
  """Write the joint seal. Refuses: a missing required input, an inconsistent generation, and a content
  change without --bump. The digest of the seal file itself appears nowhere inside it."""
  out = out or SEAL_PATH
  absent = missing_required(REQUIRED_INPUTS) + missing_required(REQUIRED_GENERATED)
  if absent:
    raise gp.GenError([f"required input missing, refusing to seal: {rel}" for rel in absent])
  budgets = open_budgets()
  if budgets:
    raise gp.GenError(
      [
        f"open budget {name}: no recorded decision; sealing over it is forbidden (fill it in "
        "configs/w4_capacity_decisions.yaml: value, decided_by, rationale, evidence)"
        for name in budgets
      ]
    )
  if verify_generated():
    raise gp.GenError(verify_generated())
  inputs = {rel: _sha(REPO / rel) for rel in REQUIRED_INPUTS}
  generated = {rel: _sha(REPO / rel) for rel in REQUIRED_GENERATED}
  protocol_seq = 1
  if out.is_file():
    previous = json.loads(out.read_text())
    if previous.get("inputs") == inputs and previous.get("generated") == generated:
      return previous  # idempotent: identical content re-seals to the same record
    if not bump:
      raise gp.GenError(
        ["content changed since the last seal; a new seal needs --bump (protocol_seq+1, old archived)"]
      )
    protocol_seq = int(previous.get("protocol_seq", 1)) + 1
    archive = out.with_name(f"PROTOCOL_FREEZE.v{protocol_seq - 1}.json")
    shutil.copy2(out, archive)
  body = _seal_body(inputs, generated, protocol_seq)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
  return body


def verify_seal(path: Path | None = None) -> list[str]:
  """Verify the committed seal: schema/state/seq plus full hash coverage, no half-seal."""
  path = path or SEAL_PATH
  if not path.is_file():
    return [f"{_rel(path)}: the joint seal is missing"]
  try:
    record = json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    return [f"{_rel(path)}: not valid JSON ({exc})"]
  errors: list[str] = []
  if record.get("schema") != "cn_protocol_freeze/v1":
    errors.append("seal: unknown schema")
  if record.get("state") != "frozen":
    errors.append(f"seal: state {record.get('state')!r} is not frozen (half-seal)")
  if not isinstance(record.get("protocol_seq"), int) or record["protocol_seq"] < 1:
    errors.append("seal: protocol_seq must be a positive integer")
  for section, required in (("inputs", REQUIRED_INPUTS), ("generated", REQUIRED_GENERATED)):
    listed = record.get(section)
    if not isinstance(listed, dict):
      errors.append(f"seal.{section}: missing")
      continue
    for rel in required:
      if rel not in listed:
        errors.append(f"seal.{section}: required entry {rel} is not covered (half-seal)")
    for rel in sorted(listed):
      file_path = REPO / rel
      if not file_path.is_file():
        errors.append(f"{rel}: sealed but missing")
      elif _sha(file_path) != listed[rel]:
        errors.append(f"{rel}: sha256 differs from the seal (stale input or tampered artifact)")
  errors.extend(verify_generated())
  budgets = open_budgets()
  if budgets:
    errors.append(
      "open required budget, seal invalid until its decision is recorded: " + ", ".join(budgets)
    )
  return errors


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="cmd", required=True)
  c = sub.add_parser("collect")
  c.add_argument("--out", required=True)
  v = sub.add_parser("verify")
  v.add_argument("--record")
  s = sub.add_parser("seal")
  s.add_argument("--out", default=str(SEAL_PATH))
  s.add_argument("--bump", action="store_true")
  args = parser.parse_args()
  if args.cmd == "collect":
    try:
      record = collect()
    except gp.GenError as exc:
      print("\n".join(exc.errors), file=sys.stderr)
      raise SystemExit(1) from exc
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} (state={record['state']})")
    return
  if args.cmd == "seal":
    try:
      body = seal(Path(args.out), bump=args.bump)
    except gp.GenError as exc:
      print("\n".join(exc.errors), file=sys.stderr)
      raise SystemExit(1) from exc
    print(f"sealed {args.out} (state={body['state']}, protocol_seq={body['protocol_seq']})")
    return
  errors = verify_generated()
  if args.record:
    errors += verify(json.loads(Path(args.record).read_text()))
  errors += verify_seal()
  if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
  print("freeze_check: ok")


if __name__ == "__main__":
  main()
