"""Validate the C010 native owner / ROB tag / retire contracts.

Two kinds of check, neither of which is "grep for OwnerTag":

1. Structured consumer behaviour. `contract_codec.RobConsumer` is built FROM the
   YAML (field list, width rules, required flags, reuse rule); it decodes a
   completion by every required field and refuses everything else, so a result
   for a reused slot cannot be read as the new op. Tests drive legal, missing,
   mismatched and out-of-range samples through it, and mutate the YAML to prove
   the consumer follows it.
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
from pathlib import Path
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB))

import contract_codec as cc  # noqa: E402
from validate_protocol import load_all  # noqa: E402

H = LIB.parents[1]
REPO = LIB.parents[3]
UPSTREAM = H / "upstream" / "coralnpu"
DOC = H / "rtl" / "scalar" / "doc"
PROTOCOL = H / "protocol"

_DOCS = ("rob_tag", "ctx_ctrl", "retire_record", "oracle_requirements", "perctx_issue")
# Fields whose width has no sourced bound anywhere; a numeric rule is invention.
_UNSOURCED_WIDTH_FIELDS = {"ctx_generation", "owner_op_seq", "subop_id", "slot_generation"}
_FIELD_KEYS = ("name", "width_rule", "owner", "consumers", "required")


def layout(docs: dict[str, Any], values: dict[str, int], context_count: int) -> dict[str, Any]:
  """Layout implied by the YAML for one configuration, via the contract codec.

  One Context still gets a real ctx field (a constant), never a zero-width one.
  Fields whose width is `unresolved` are listed as open, not given a number.
  """
  config = cc.ConfigParams(context_count, values)
  env = config.env()
  fields = docs["rob_tag"]["rob_tag"]["fields"] + docs["rob_tag"]["owner_tag"]["fields"]
  resolvable, open_ = cc.split_resolvable(fields, env)
  widths: dict[str, int] = {}
  for f in resolvable:
    widths.setdefault(f["name"], cc.eval_width(f["width_rule"], env))
  return {
    "context_count": context_count,
    "ctx_bits": env["ctx_bits"],
    "ctx_slot_constant": context_count == 1,
    "rob_slot_bits": widths["rob_slot"],
    "partitions": context_count,
    "resolved_widths": widths,
    "open_widths": sorted(set(open_)),
  }


def rob_consumer(docs: dict[str, Any], values: dict[str, int], context_count: int, **fixture_widths: int):
  """Build the YAML-driven ROB. Unresolved widths need an explicit fixture width."""
  config = cc.ConfigParams(context_count, values, fixture_widths or None)
  return cc.RobConsumer(docs["rob_tag"], config)


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
    name, rule = str(f.get("name")), str(f.get("width_rule"))
    if name in _UNSOURCED_WIDTH_FIELDS and rule.strip() != cc.UNRESOLVED:
      # Any other rule is invention: a literal, an expression over literals (8+8,
      # max(8,8), log2Ceil(16)), or a borrowed parameter. No source bounds these.
      errors.append(
        f"{where}.{name}: width rule {rule!r} is not 'unresolved'; "
        "no source bounds this field (invented width)"
      )
    if rule.strip() == cc.UNRESOLVED and name not in _UNSOURCED_WIDTH_FIELDS:
      errors.append(f"{where}.{name}: declared unresolved but its width derives from sourced parameters")
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
  rob_bits = cc.log2_ceil(params["retirementBufferSize"])
  idx_bits_no_float = cc.log2_ceil(params["scalarRegCount"] + params["rvvRegCount"] + 2)
  idx_bits_float = cc.log2_ceil(params["scalarRegCount"] + float_regs + params["rvvRegCount"] + 2)
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
  for item in rec["width_parameters"]:
    anchor = item.get("anchor", item["name"])
    check_anchor(errors, f"retire_record.width_parameters.{item['name']}", item["source"], anchor)
    text = _source_line(item["source"]["file"], item["source"]["line"]) or ""
    if str(item["value"]) not in text and "def " not in text:
      errors.append(f"retire_record.width_parameters.{item['name']}: value not on its cited line")
  trace = UPSTREAM / "hdl/chisel/src/coralnpu/RvviTrace.scala"
  errors.extend(f"retire_record: {e}" for e in cc.check_trace_coverage(rec, trace))
  if rec["mode_requirement"]["retirement_buffer_mode"] != "full":
    errors.append("retire_record must require a full RetirementBuffer")
  check_anchor(errors, "retire_record.mode", rec["mode_requirement"]["source"], "mini")
  check_anchor(errors, "retire_record.mode.also", rec["mode_requirement"]["also"], "dataWidth")
  if "W7" not in rec["read_only_for"]:
    errors.append("retire_record must be read-only for W7")


# The completion match floor: removing any of these from `required` removes the
# protection it gives (ABA, cross-Context, stale Task, sub-operation), so the
# contract may add required fields but never drop these.
_REQUIRED_FLOOR = {
  "rob_tag": {"ctx_slot", "rob_slot", "slot_generation"},
  "owner_tag": {"ctx_slot", "ctx_generation", "owner_op_seq", "subop_id"},
}


def check_required_floor(docs: dict[str, Any], errors: list[str]) -> None:
  for group, floor in _REQUIRED_FLOOR.items():
    fields = {f["name"]: f for f in docs["rob_tag"][group]["fields"]}
    for name in sorted(floor):
      if name not in fields:
        errors.append(f"rob_tag.{group}: required match field {name!r} is missing")
      elif fields[name].get("required") is not True:
        errors.append(f"rob_tag.{group}.{name}: must be required: true (it is part of the match floor)")


def check_oracle_retire_fields(docs: dict[str, Any], errors: list[str]) -> None:
  carried = {f["name"]: f for f in docs["retire_record"]["fields"]}
  for name in docs["oracle_requirements"]["retire_fields_compared"]["fields"]:
    if name not in carried:
      errors.append(f"oracle compares {name!r} but retire_record does not carry it")
    elif carried[name].get("required") is not True:
      errors.append(f"oracle compares {name!r} but retire_record marks it optional")


def check_width_evaluation(docs: dict[str, Any], params: dict[str, int], errors: list[str]) -> None:
  """Every resolvable width rule must evaluate for both C1 and C2; none may be zero."""
  values = dict(params)
  values.update({p["name"]: int(p["value"]) for p in docs["retire_record"]["width_parameters"]})
  for count in (1, 2):
    env = cc.ConfigParams(count, values).env()
    for where, f in _all_fields(docs):
      if str(f["width_rule"]).strip() == cc.UNRESOLVED:
        continue
      try:
        cc.eval_width(f["width_rule"], env)
      except cc.ContractError as exc:
        errors.append(f"{where}.{f['name']} (context_count={count}): {exc}")


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


def check_field_request(request: dict, protocol: dict, docs: dict[str, Any], errors: list[str]) -> None:
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
  return validate_with(load_docs(docs_dir), protocol_dir)


def validate_with(docs: dict[str, Any], protocol_dir: Path = PROTOCOL) -> dict:
  """Run every check on already-loaded docs (so tests can validate a mutated copy)."""
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
  check_width_evaluation(docs, params, errors)
  check_required_floor(docs, errors)
  check_oracle_retire_fields(docs, errors)
  check_issue(docs, params, errors)
  check_oracle(docs, errors)
  check_field_request(request, proto["protocol"], docs, errors)
  open_widths = sorted({f["name"] for _, f in _all_fields(docs) if f.get("width_rule") == "unresolved"})
  layouts: dict[int, Any] = {}
  for count in (1, 2):
    try:
      layouts[count] = layout(docs, params, count)
    except cc.ContractError as exc:
      # A rejected rule must surface as a failed report, never as a crash.
      errors.append(f"layout(context_count={count}): {exc}")
  return {
    "schema": "cn_native_owner_report/v1",
    "valid": not errors,
    "errors": errors,
    "open_widths": open_widths,
    "layouts": layouts,
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
