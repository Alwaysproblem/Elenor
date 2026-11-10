"""Validate the C016 W7 contracts, the T matrix and the W7 field request.

What it checks:

1. Every `source: {file, line, needle}` in the five contracts is READ BACK from the pinned tree.
2. The required contract is held HERE, not in the YAML: window/header fields, window kinds, the completeness
   conditions, record fields, R1/R2 rules, verdict values, outcome classes, evidence kinds, the C7 gate
   rule, perf baselines and metrics. A synchronized edit that weakens the YAML and its guard together
   cannot pass.
3. The T matrix is checked against the REQUIREMENTS document itself: exactly T-01..T-68, each row's text
   hash equals the hash of the live requirement cell, every row starts NOT_RUN (a preset PASS is rejected),
   evidence kinds fit the applicability, and a deferral is allowed only for the three reasons the design
   names, each with a start condition.
4. Unresolved quantities (capture quota, perf thresholds) stay `unresolved` with an owner.
5. The field request names only catalog identity fields and keys that exist in the contracts.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB))

import w7_trace as w7  # noqa: E402

H = LIB.parents[1]
REPO = LIB.parents[3]
UPSTREAM = H / "upstream" / "coralnpu"
REQUEST = H / "protocol" / "field_requests" / "w7.yaml"
REQ_DOC = REPO / "Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md"
CATALOG = H / "protocol" / "cn_protocol_v1.yaml"
PLAN_DOC = REPO / "design" / "coral_nexus_tile" / "implementation" / "W7_Verification_Trace_Plan.md"

# ---------------------------------------------------------------------------------------------------
# The required contract. Held here so one edit cannot weaken the YAML and its guard together.
# ---------------------------------------------------------------------------------------------------
REQ_HEADER = (
  "schema_ver",
  "run_id",
  "window_id",
  "scenario_id",
  "t_id",
  "window_kind",
  "startup_epoch",
  "first_seq",
  "last_seq",
  "record_count",
  "gap_status",
  "end_boundary",
  "blob_manifest",
  "config_refs",
  "seal",
)
REQ_WINDOW_KINDS = ("full", "partial", "fault_segment")
REQ_SEAL = ("first_seq", "last_seq", "record_count", "end_boundary", "gap_status", "records_sha256")
REQ_BLOB_KINDS = ("payload", "initial_memory", "image_ref", "prefetched_state")
REQ_GAP = ("none", "quota", "drop", "truncate")
REQ_COMPLETE = (
  "seq_contiguous",
  "every_blob_ref_retrievable",
  "every_hash_matches",
  "initial_snapshot_and_image_listed",
  "seal_present_and_valid",
  "end_boundary_reached",
  "gap_status_none",
  "seed_and_backpressure_config_present",
)
REQ_END_TYPES = ("run_terminal", "declared_capture_point")
REQ_RECORD_FIELDS = (
  "seq",
  "cycle",
  "phase",
  "protocol",
  "dir",
  "port",
  "valid",
  "ready",
  "accepted",
  "len",
  "payload_hash",
  "reliability",
  "kind",
  "payload_bytes",
  "blob_ref",
  "message_seq",
  "r1_seq",
  "identity",
  "t_id",
)
REQ_REQUIRED_RECORD = (
  "seq",
  "cycle",
  "phase",
  "protocol",
  "dir",
  "port",
  "valid",
  "ready",
  "accepted",
  "len",
  "payload_hash",
  "reliability",
  "kind",
)
REQ_PROTOCOLS = ("CTRL", "EVENT", "PROGRAM", "MEMORY", "DEBUG")
REQ_UNIQUE_KEY = ("window_id", "protocol", "port", "dir", "cycle", "phase")
REQ_RAW_FORBIDDEN = ("ts_us", "ts_ns", "ts", "quantized_cycle", "rounded_cycle", "wall_time")
REQ_R1_CHANNELS = ("CTRL", "EVENT", "PROGRAM", "MEMORY")
REQ_GEN_DOMAINS = ("ctx_generation", "arena_alloc_gen", "program_epoch", "profile_l1_gen", "run_generation")
REQ_T_VERDICTS = ("PASS", "FAIL", "BLOCKED", "DEFERRED_NOT_APPLICABLE", "NOT_RUN", "REJECTED")
REQ_VERDICT_FIELDS = (
  "schema",
  "run_id",
  "scenario_id",
  "t_id",
  "t_verdict",
  "phase",
  "checker",
  "evidence",
  "t_scope",
  "notes",
)
REQ_CHECKERS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
REQ_OUTCOMES = {
  "CASE_PASS": "PASS",
  "DUT_ERROR": "FAIL",
  "NUMERIC_FAIL": "FAIL",
  "CHECKER_FAIL": "FAIL",
  "PENDING_NOT_COMPLETE": "FAIL",
  "NOT_FULLY_REPLAYABLE": "REJECTED",
}
REQ_EVIDENCE_KINDS = ("CS1_LOG", "CS1_PY", "BUILD", "PERF", "REPLAY", "BIT_ORACLE", "PMU", "ORACLE", "CS2")
REQ_PY_ALONE = (
  ("T-49", "python cold/warm epoch baseline"),
  ("T-35", "P0 comparison row"),
  ("T-37", "build-time matrix"),
  ("T-33", "build-time rejection boundary"),
  ("T-53", "loader-level part"),
)
REQ_DEFER_REASONS = ("level_not_started", "review_not_started", "n2_subitem_before_n_ge_2_claim")
REQ_PERF_BASELINES = ("P0", "P1", "PN", "PS")
REQ_PERF_METRICS = ("PERF-01", "PERF-02", "PERF-04", "PERF-05", "PERF-07")
REQ_REPORT_KEYS = (
  "cycles",
  "completed",
  "utilization",
  "credit_invariant_ok",
  "stall_breakdown",
  "stall_categories",
  "traffic",
  "events",
  "memory",
  "scheduler",
  "resources",
  "profile",
  "arenas",
  "task_leases",
  "checks",
)
REQ_STALL = (
  "NONE",
  "WAIT_EVENT",
  "WAIT_OPERAND",
  "STREAM_CREDIT",
  "WAIT_L1_BANK",
  "WAIT_L2_BANK",
  "WAIT_NOC_CREDIT",
  "WAIT_DMA_QUEUE",
  "WAIT_HBM_OUTSTANDING",
  "UCE_PROGRAM_DESC",
  "UNKNOWN",
  "WAIT_MSHR",
)
REQ_UNRESOLVED = {
  ("window", ("gap_status", "quota", "limit")): "C019",
  ("perf", ("pre_registration", "threshold_value")): "C019",
}
REQ_CS2_ROWS = ("T-15", "T-19", "T-31", "T-42", "T-43", "T-46", "T-47", "T-48", "T-52", "T-57")
REQ_N2_ROWS = ("T-28", "T-68")
REQ_DEFERRED_ROWS = ("T-59", "T-60")
REQ_MATRIX_STATUS = ("NOT_RUN", "DEFERRED_NOT_APPLICABLE")
# Evidence kinds a row must carry, held here and not only in the matrix. Rows not listed need CS1_LOG alone
# (build/validator/perf rows are listed explicitly). Source: the "证据类型" column of the W7 plan T table.
REQ_ROW_KINDS: dict[str, tuple[str, ...]] = {
  **dict.fromkeys(("T-01", "T-16", "T-28", "T-29", "T-39", "T-50", "T-61"), ("CS1_LOG", "BIT_ORACLE")),
  **dict.fromkeys(("T-34", "T-36"), ("CS1_LOG", "PMU")),
  **dict.fromkeys(("T-33", "T-53"), ("BUILD", "CS1_LOG")),
  "T-35": ("PERF",),
  "T-37": ("BUILD",),
  "T-49": ("CS1_PY",),
  "T-65": ("CS1_LOG", "ORACLE"),
  "T-67": ("CS1_LOG", "REPLAY", "PERF"),
  "T-59": (),
  "T-60": (),
}
BANNED = ("address", "bits", "opcode", "width_bits", "encoding")


# ---------------------------------------------------------------------------------------------------
def _read_line(file: str, line: int) -> str | None:
  for root in (REPO, UPSTREAM):
    path = root / file
    if path.is_file():
      lines = path.read_text().splitlines()
      return lines[line - 1] if 0 < line <= len(lines) else None
  return None


def iter_sources(node: Any, where: str = "") -> list[tuple[str, dict]]:
  found: list[tuple[str, dict]] = []
  if isinstance(node, dict):
    if isinstance(node.get("file"), str) and isinstance(node.get("line"), int):
      found.append((where, node))
    for key, value in node.items():
      found.extend(iter_sources(value, f"{where}.{key}"))
  elif isinstance(node, list):
    for index, value in enumerate(node):
      found.extend(iter_sources(value, f"{where}[{index}]"))
  return found


def check_sources(label: str, doc: Any, errors: list[str]) -> int:
  count = 0
  for where, src in iter_sources(doc):
    count += 1
    if "needle" not in src:
      errors.append(f"{label}{where}: MISSING_SOURCE (need file/line/needle)")
      continue
    text = _read_line(src["file"], src["line"])
    if text is None:
      errors.append(f"{label}{where}: SOURCE_FILE_MISSING {src['file']}:{src['line']}")
    elif str(src["needle"]) not in text:
      errors.append(
        f"{label}{where}: SOURCE_LINE_MISMATCH {src['file']}:{src['line']} lacks {src['needle']!r}"
      )
  return count


def _eq(errors: list[str], where: str, got: Any, want: Any) -> None:
  if got != want:
    errors.append(f"{where}: {got!r} != required {want!r}")


def _tup(v: Any) -> Any:
  return tuple(v) if isinstance(v, list) else v


def _dig(doc: dict[str, Any], path: tuple[str, ...]) -> Any:
  node: Any = doc
  for key in path:
    node = node[key]
  return node


# ---------------------------------------------------------------------------------------------------
def check_window(w: dict[str, Any], errors: list[str]) -> None:
  _eq(errors, "window.header.required_fields", _tup(w["header"]["required_fields"]), REQ_HEADER)
  _eq(errors, "window.header.window_kinds", _tup(w["header"]["window_kinds"]), REQ_WINDOW_KINDS)
  if w["header"].get("unknown_fields") != "reject":
    errors.append("window.header.unknown_fields must be reject")
  t = w["time"]
  if t["cycle"].get("quantized") is not False:
    errors.append("window.time.cycle: the raw log is never quantized")
  if t["seq"].get("wraps") is not False or t["seq"].get("monotonic") != "strict":
    errors.append("window.time.seq: strictly monotonic and never wraps")
  if t.get("float_or_bool_or_str_in_integer_field") != "reject":
    errors.append("window.time: a non-integer in an integer field must be rejected")
  d = t["display_view"]
  if d.get("is_evidence") is not False or d.get("is_sort_key") is not False:
    errors.append("window.time.display_view: a display view is neither evidence nor a sort key")
  sd = w["sequence_domains"]
  if sd.get("never_substitute") is not True:
    errors.append("window.sequence_domains: seq and message_seq must never substitute for each other")
  if not (
    sd.get("continuous_message_seq_does_not_repair_a_seq_gap")
    and sd.get("continuous_seq_does_not_repair_a_message_seq_gap")
  ):
    errors.append("window.sequence_domains: neither sequence may repair a gap in the other")
  if (
    sd["seq"].get("scope") != "window" or sd["message_seq"].get("scope") != "per (tile, channel, direction)"
  ):
    errors.append("window.sequence_domains: scopes changed")
  se = w["startup_epoch"]
  if tuple(se["required_fields"]) != ("includes_startup", "clock_ref"):
    errors.append("window.startup_epoch.required_fields")
  if se.get("raw_log_applies_profile_init_discard") is not False:
    errors.append("window.startup_epoch: the raw log never drops the Profile initialization")
  _eq(errors, "window.end_boundary.types", _tup(w["end_boundary"]["types"]), REQ_END_TYPES)
  if w["end_boundary"].get("declared_end_is_held_outside_the_window") is not True:
    errors.append("window.end_boundary: the declared end must be held outside the window")
  s = w["seal"]
  _eq(errors, "window.seal.required_fields", _tup(s["required_fields"]), REQ_SEAL)
  if s["records_sha256"].get("algorithm") != "sha256":
    errors.append("window.seal.records_sha256.algorithm must be sha256")
  for name in ("missing_seal", "seal_mismatch"):
    if s.get(name) != "not_replayable":
      errors.append(f"window.seal.{name} must be not_replayable")
  if (
    s.get("seal_is_itself_in_the_window") is not True
    or s.get("records_before_seal_are_immutable") is not True
  ):
    errors.append("window.seal: the seal is part of the window and the records before it are immutable")
  _eq(errors, "window.blob_manifest.blob_kinds", _tup(w["blob_manifest"]["blob_kinds"]), REQ_BLOB_KINDS)
  if w["blob_manifest"].get("hash_only_is_not_a_blob") is not True:
    errors.append("window.blob_manifest: a hash alone is not a blob")
  _eq(errors, "window.gap_status.values", _tup(w["gap_status"]["values"]), REQ_GAP)
  _eq(errors, "window.gap_status.replayable_only_when", w["gap_status"]["replayable_only_when"], "none")
  if w["gap_status"]["quota"].get("never_unbounded") is not True:
    errors.append("window.gap_status.quota: capture storage is never unbounded")
  c = w["completeness"]
  _eq(errors, "window.completeness.replayable_requires", _tup(c["replayable_requires"]), REQ_COMPLETE)
  if (
    c.get("prefix_still_contiguous_is_not_enough") is not True
    or c.get("replayable_is_not_task_settled") is not True
  ):
    errors.append("window.completeness: a contiguous prefix is not enough, and replayable is not settled")
  if c.get("any_missing") != "not_fully_replayable":
    errors.append("window.completeness.any_missing must be not_fully_replayable")
  sr = w["single_side_replay"]
  if (
    sr.get("both_sides") != "not_promised"
    or sr.get("response_returned_only_when") != "issued_request_matches_logged_identity"
  ):
    errors.append("window.single_side_replay: one side only, response only for a matching request")
  hp = w["host_passive"]
  if hp.get("drives_dut_ports") is not False or hp.get("writes_dut_state") is not False:
    errors.append("window.host_passive: the Host never drives a port or writes DUT state")


def check_record(r: dict[str, Any], errors: list[str]) -> None:
  names = tuple(f["name"] for f in r["fields"])
  _eq(errors, "record.fields", names, REQ_RECORD_FIELDS)
  req = tuple(f["name"] for f in r["fields"] if f["required"])
  _eq(errors, "record.fields.required", req, REQ_REQUIRED_RECORD)
  _eq(errors, "record.protocol.names", _tup(r["protocol"]["names"]), REQ_PROTOCOLS)
  _eq(errors, "record.uniqueness.key", _tup(r["uniqueness"]["key"]), REQ_UNIQUE_KEY)
  if (
    r["uniqueness"].get("at_most_one_record_per_key") is not True
    or r["uniqueness"].get("req_and_resp_are_different_keys") is not True
  ):
    errors.append("record.uniqueness: one record per key; req and resp are different keys")
  _eq(errors, "record.raw_log_forbidden_fields", _tup(r["raw_log_forbidden_fields"]), REQ_RAW_FORBIDDEN)
  if r.get("unknown_fields") != "reject":
    errors.append("record.unknown_fields must be reject")
  h = r["handshake"]
  if h.get("accepted_must_equal_valid_and_ready") is not True or h.get("accepted_is") != "valid and ready":
    errors.append("record.handshake: accepted is valid and ready")
  if h["valid_without_ready"].get("payload_must_stay_stable_until_accepted") is not True:
    errors.append("record.handshake: payload must stay stable until accepted")
  _eq(
    errors, "record.identity.generation_domains", _tup(r["identity"]["generation_domains"]), REQ_GEN_DOMAINS
  )
  if r["identity"].get("generation_domains_distinct") is not True:
    errors.append("record.identity: generation domains are distinct")
  if r["identity"].get("vme_instance_visible_to_software") is not False:
    errors.append("record.identity: the VME instance number is invisible to software (T-68)")
  if r["identity"].get("physical_tile_is_routing_only") is not True:
    errors.append("record.identity: the physical Tile is routing only")
  rel = r["reliability"]
  _eq(errors, "record.reliability.R1.channels", _tup(rel["R1"]["channels"]), REQ_R1_CHANNELS)
  for k in ("reliable", "ordered", "lossless"):
    if rel["R1"].get(k) is not True:
      errors.append(f"record.reliability.R1.{k} must be true")
  if rel["R1"].get("absence") != "FAIL":
    errors.append("record.reliability.R1.absence must be FAIL")
  if rel["R2"].get("loss_changes_functional_verdict") is not False or rel["R2"].get("channels") != [
    "DEBUG"
  ]:
    errors.append("record.reliability.R2: lossy, DEBUG only, never changes a functional verdict")
  if (
    rel.get("r1_on_debug_channel") != "forbidden"
    or rel.get("functional_message_on_lossy_path") != "forbidden"
  ):
    errors.append("record.reliability: no R1 and no functional message on the lossy path")
  d = r["data"]
  if d.get("hash_only_record_replayable") is not False or d["hash"].get("algorithm") != "sha256":
    errors.append("record.data: sha256, and a hash-only record is not replayable")
  nm = r["normal_mode"]
  if nm.get("keeps_history_on_chip") is not False or nm.get("hbm_log_writes") != 0:
    errors.append("record.normal_mode: no on-chip history and zero HBM log writes")


def check_verdict(v: dict[str, Any], errors: list[str]) -> None:
  vd = v["verdict"]
  _eq(errors, "verdict.t_verdict_values", _tup(vd["t_verdict_values"]), REQ_T_VERDICTS)
  _eq(errors, "verdict.required_fields", _tup(vd["required_fields"]), REQ_VERDICT_FIELDS)
  if vd.get("not_run_is_never_pass") is not True:
    errors.append("verdict: NOT_RUN is never a pass")
  if vd.get("harness_only_is_cs1_evidence") is not False:
    errors.append("verdict: harness_only is not CS1 evidence")
  _eq(errors, "verdict.checkers", tuple(k for k in v["checkers"] if re.fullmatch(r"C\d", k)), REQ_CHECKERS)
  if v["checkers"].get("all_seven_required_for_a_cs1_log_pass") is not True:
    errors.append("verdict.checkers: all seven are required for a CS1_LOG pass")
  if v["checkers"].get("checker_may_write_functional_state") is not False:
    errors.append("verdict.checkers: a checker is read-only")
  oc = v["outcome_classes"]
  _eq(errors, "verdict.outcome_classes.maps_to_t_verdict", oc["maps_to_t_verdict"], REQ_OUTCOMES)
  if oc.get("legal_injected_fault_is_a_case_pass") is not True:
    errors.append("verdict.outcome_classes: a legal injected fault ending as expected is a case pass")
  if (
    oc.get("replay_gap_is_a_numeric_fail") is not False
    or oc.get("replayable_window_is_task_success") is not False
  ):
    errors.append("verdict.outcome_classes: a replay gap is not a numeric fail; replayable is not success")
  c4 = v["c4"]
  if (
    c4.get("bitwise_only") is not True
    or c4.get("tolerance_allowed") is not False
    or c4.get("ulp_relaxation") != "forbidden"
  ):
    errors.append("verdict.c4: bitwise only, no tolerance, no ULP relaxation")
  if c4.get("upstream_spike_or_mpact_as_dut_pass") != "forbidden":
    errors.append("verdict.c4: an upstream Spike/MPACT result is never a DUT pass")
  ev = v["evidence"]
  _eq(errors, "verdict.evidence.kinds", _tup(ev["kinds"]), REQ_EVIDENCE_KINDS)
  if ev.get("cs1_py_overrides_a_missing_cs1_log") is not False:
    errors.append("verdict.evidence: CS1_PY never covers a missing CS1_LOG")
  allowed = tuple((a["t_id"], a["part"]) for a in ev["cs1_py_alone_allowed_for"])
  _eq(errors, "verdict.evidence.cs1_py_alone_allowed_for", allowed, REQ_PY_ALONE)
  if ev.get("python_cycles_as_rtl_cycles") != "forbidden":
    errors.append("verdict.evidence: Python cycles are never RTL cycles")
  g = v["c7_gate"]
  if (
    g.get("all_applicable_mandatory_rows") != "PASS"
    or g.get("deadline_counts_as_pass") is not False
    or g.get("pending_counts_as_complete") is not False
  ):
    errors.append(
      "verdict.c7_gate: every applicable mandatory row must PASS; a deadline or a pending is not a pass"
    )
  _eq(
    errors,
    "verdict.c7_gate.deferred.allowed_only_for",
    _tup(g["deferred"]["allowed_only_for"]),
    REQ_DEFER_REASONS,
  )
  if (
    g.get("evidence_missing_or_hash_mismatch") != "manifest_invalid"
    or g.get("evidence_refs_closed") is not True
  ):
    errors.append("verdict.c7_gate: missing or mismatching evidence makes the manifest invalid")
  if v["not_empty_pass"].get("pass_from_a_file_that_merely_exists") != "forbidden":
    errors.append("verdict.not_empty_pass: a file that merely exists is not evidence")


def check_bundle(b: dict[str, Any], errors: list[str]) -> None:
  rv = b["replay_gate"]
  _eq(errors, "bundle.replay_gate.verdict_when_any", rv["verdict_when_any"], "NOT_FULLY_REPLAYABLE")
  _eq(errors, "bundle.replay_gate.exit_code_when_any", rv["exit_code_when_any"], 3)
  need = {
    "missing_blob",
    "blob_hash_mismatch",
    "missing_initial_snapshot",
    "missing_image",
    "missing_seal",
    "seal_mismatch",
    "seq_gap",
    "quota_gap",
    "truncated_end",
    "missing_seed_or_backpressure_config",
  }
  if set(rv["parts_that_make_it_not_replayable"]) != need:
    errors.append("bundle.replay_gate.parts_that_make_it_not_replayable: the full set is required")
  if rv.get("gap_is_a_numeric_fail") is not False or rv.get("reject_before_driving_any_port") is not True:
    errors.append("bundle.replay_gate: a gap is not a numeric fail and rejection precedes driving any port")
  if b["oracle_ref"].get("validator_as_numeric_oracle") != "forbidden":
    errors.append("bundle.oracle_ref: the validator is never a numeric oracle")
  if b["bundle"].get("unknown_fields") != "reject":
    errors.append("bundle.bundle.unknown_fields must be reject")
  if b["config"].get("seed_list_empty") != "reject":
    errors.append("bundle.config: an empty seed list is rejected")
  if b["single_side"].get("cycle_equality_required") is not False:
    errors.append("bundle.single_side: cycle equality is not required")


def check_perf(p: dict[str, Any], errors: list[str]) -> None:
  _eq(errors, "perf.rows.baselines", tuple(x["id"] for x in p["rows"]["baselines"]), REQ_PERF_BASELINES)
  ps = next(x for x in p["rows"]["baselines"] if x["id"] == "PS")
  if ps.get("run") != "not_run" or p["rows"].get("ps_row_status") != "not_run":
    errors.append("perf.rows: PS is registered not_run")
  if (
    p["rows"].get("ps_is_not_a_deferral_of_a_mandatory_row") is not True
    or p["rows"].get("p0_p1_pn_are_mandatory") is not True
  ):
    errors.append("perf.rows: P0/P1/PN stay mandatory; PS is not a deferral of a mandatory row")
  if p["rows"].get("pn_differs_from_p1_only_in") != "active_context_count":
    errors.append("perf.rows: PN differs from P1 only in the active Context count")
  _eq(errors, "perf.metrics", tuple(m["id"] for m in p["metrics"]["required"]), REQ_PERF_METRICS)
  if (
    p["metrics"].get("from_source_text_or_file_existence") != "forbidden"
    or p["metrics"].get("needs_reverse_control") is not True
  ):
    errors.append(
      "perf.metrics: no metric from source text or file existence; each needs a reverse control"
    )
  if (
    p["timing"].get("authority") != "python_simulator_cycle"
    or p["timing"].get("debug_in_normal_performance") != "forbidden"
  ):
    errors.append("perf.timing: simulator cycle authority; no debug perturbation in normal numbers")
  _eq(errors, "perf.report_keys.required", _tup(p["report_keys"]["required"]), REQ_REPORT_KEYS)
  if p["report_keys"].get("never_add_stall_categories_to_stall_breakdown") is not True:
    errors.append(
      "perf.report_keys: stall_categories is an aggregate and is never added to stall_breakdown"
    )
  _eq(errors, "perf.stall_reasons.names", _tup(p["stall_reasons"]["names"]), REQ_STALL)
  _eq(errors, "perf.stall_reasons.count", p["stall_reasons"]["count"], 12)
  pr = p["pre_registration"]
  _eq(errors, "perf.pre_registration.order", pr["order"], "register_then_measure")
  if pr.get("zero_placeholder") != "forbidden" or pr.get("threshold_chosen_after_the_data") != "forbidden":
    errors.append("perf.pre_registration: no zero placeholder and no threshold chosen after the data")
  need = {
    "truncated_window",
    "gap_window",
    "missing_blob",
    "missing_seal",
    "quota_window",
    "debug_perturbed_run_in_normal_numbers",
    "unregistered_threshold",
    "python_cycles_reported_as_rtl_cycles",
  }
  if set(p["gate_rejects"]) != need:
    errors.append("perf.gate_rejects: the full rejection set is required")
  if set(p["forbidden_claims"]) != {
    "fmax",
    "area",
    "power",
    "multi_vme_utilization",
    "ppa",
    "python_cycles_as_rtl_truth",
  }:
    errors.append("perf.forbidden_claims: the full set is required")


def check_unresolved(docs: dict[str, Any], errors: list[str]) -> None:
  for (which, path), owner in REQ_UNRESOLVED.items():
    where = f"{which}.{'.'.join(path)}"
    try:
      node = _dig(docs[which], path)
    except (KeyError, TypeError):
      errors.append(f"{where}: required unresolved quantity is missing")
      continue
    if not isinstance(node, dict):
      errors.append(f"{where}: must be an unresolved record with owner and reason")
      continue
    if node.get("value") != "unresolved":
      errors.append(
        f"{where}: must stay 'unresolved' (a number here would be invented), got {node.get('value')!r}"
      )
    _eq(errors, f"{where}.owner", node.get("owner"), owner)
    if not str(node.get("reason", "")).strip():
      errors.append(f"{where}: unresolved needs a reason")

  def walk(node: Any, where: str) -> None:
    if isinstance(node, dict):
      for k, val in node.items():
        if k in ("width_bits", "bit_position", "csr_address", "opcode"):
          errors.append(f"{where}.{k}: invented encoding")
        walk(val, f"{where}.{k}")
    elif isinstance(node, list):
      for i, val in enumerate(node):
        walk(val, f"{where}[{i}]")

  for name, doc in docs.items():
    walk(doc, name)


# ---------------------------------------------------------------------------------------------------
def requirement_rows(path: Path = REQ_DOC) -> dict[str, tuple[int, str]]:
  out: dict[str, tuple[int, str]] = {}
  for i, line in enumerate(path.read_text().splitlines(), 1):
    m = re.match(r"^\| (T-\d\d) +\|", line)
    if m:
      cells = [c.strip() for c in line.strip().strip("|").split("|")]
      out[m.group(1)] = (i, re.sub(r"\s+", " ", cells[1]).strip())
  return out


_PLAN_KIND = {
  "CS1_LOG": "CS1_LOG",
  "bit_oracle": "BIT_ORACLE",
  "PMU": "PMU",
  "oracle": "ORACLE",
  "PERF": "PERF",
  "BUILD": "BUILD",
  "CS1_PY": "CS1_PY",
}


def plan_evidence_kinds(path: Path = PLAN_DOC) -> dict[str, set[str]]:
  """Evidence kinds the W7 plan's T table names per row (the last column), parsed independently."""
  out: dict[str, set[str]] = {}
  for line in path.read_text().splitlines():
    m = re.match(r"^\| (T-\d\d) +\|", line)
    if not m:
      continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) < 6:
      continue
    head = re.sub(r"[\uFF08(][^\uFF09)]*[\uFF09)]", "", cells[5])
    out[m.group(1)] = {
      _PLAN_KIND[x.strip().strip("`")] for x in head.split("+") if x.strip().strip("`") in _PLAN_KIND
    }
  return out


def check_matrix(
  m: Any,
  verdict: dict[str, Any],
  errors: list[str],
  req: dict[str, tuple[int, str]] | None = None,
  plan: dict[str, set[str]] | None = None,
) -> None:
  req = req if req is not None else requirement_rows()
  plan = plan if plan is not None else plan_evidence_kinds()
  if m.get("schema") != "cn_t_matrix/v1" or m.get("status") != "draft":
    errors.append("matrix: schema cn_t_matrix/v1 and status draft")
  rows = m["rows"]
  ids = [r.get("t_id") for r in rows]
  want = [f"T-{i:02d}" for i in range(1, 69)]
  if ids != want:
    dup = sorted({x for x in ids if ids.count(x) > 1})
    miss = sorted(set(want) - set(ids))
    extra = sorted(set(ids) - set(want))
    errors.append(
      f"matrix.rows: must be exactly T-01..T-68 in order (duplicates {dup}, missing {miss}, extra {extra})"
    )
  if sorted(req) != want:
    errors.append("matrix: the requirements document does not have exactly T-01..T-68")
  kinds_ok = set(verdict["evidence"]["kinds"])
  legal_scope = {
    "mandatory",
    "cs1_subset",
    "n2_conditional",
    "build_time",
    "validator_layer",
    "cs2_only",
    "cs3_only",
  }
  legal_app = {"rtl_cs1", "build_and_rtl", "deferred", "perf", "build_time", "validator_layer"}
  for r in rows:
    t = r.get("t_id", "?")
    for name in (
      "title",
      "owner_packages",
      "impl_tasks",
      "scenario_source",
      "cs1_scope",
      "applicability",
      "evidence_kinds",
      "oracle",
      "status",
      "req_text_sha256",
    ):
      if name not in r or (r[name] in (None, "", []) and name != "evidence_kinds"):
        errors.append(f"matrix.{t}.{name}: MISSING_FIELD")
    if t in req:
      line, text = req[t]
      if r.get("req_line") != line:
        errors.append(f"matrix.{t}.req_line: {r.get('req_line')} != {line}")
      if r.get("req_text_sha256") != hashlib.sha256(text.encode()).hexdigest():
        errors.append(f"matrix.{t}.req_text_sha256: the requirement text changed under the matrix")
    ss = r.get("scenario_source") or {}
    if ss.get("kind") != "plan_id" or not ss.get("ref"):
      errors.append(f"matrix.{t}.scenario_source: needs a plan id")
    if r.get("cs1_scope") not in legal_scope:
      errors.append(f"matrix.{t}.cs1_scope: {r.get('cs1_scope')!r} is not legal")
    if r.get("applicability") not in legal_app:
      errors.append(f"matrix.{t}.applicability: {r.get('applicability')!r} is not legal")
    if r.get("status") not in REQ_MATRIX_STATUS:
      errors.append(f"matrix.{t}.status: a row starts NOT_RUN (or is deferred), never {r.get('status')!r}")
    for k in r.get("evidence_kinds") or []:
      if k not in kinds_ok:
        errors.append(f"matrix.{t}.evidence_kinds: {k!r} is not an evidence kind")
    ks = r.get("evidence_kinds") or []
    if r.get("status") == "PASS":
      errors.append(f"matrix.{t}.status: a matrix never carries PASS; results live in verdict files")
    want_kinds = REQ_ROW_KINDS.get(t, ("CS1_LOG",))
    if t in req and tuple(ks) != want_kinds:
      errors.append(f"matrix.{t}.evidence_kinds: {tuple(ks)} != required {want_kinds}")
    # the plan's own table is a third witness: a row cannot lose a kind there and here together
    in_plan = plan.get(t)
    if in_plan is not None and t not in REQ_DEFERRED_ROWS and not in_plan <= set(ks):
      errors.append(f"matrix.{t}.evidence_kinds: the W7 plan names {sorted(in_plan)} for this row")
    if r.get("status") == "NOT_RUN" and not ks and not r.get("subitems"):
      errors.append(f"matrix.{t}.evidence_kinds: an executable row needs at least one evidence kind")
    app = r.get("applicability")
    if app == "rtl_cs1" and not (ks and "CS1_LOG" in ks):
      errors.append(f"matrix.{t}: an RTL CS1 row needs CS1_LOG")
    if app == "rtl_cs1" and "CS1_PY" in ks:
      errors.append(f"matrix.{t}: an RTL CS1 row cannot be satisfied by CS1_PY")
    if app == "validator_layer" and ks != ["CS1_PY"]:
      errors.append(f"matrix.{t}: a validator-layer row uses CS1_PY only")
    if app == "validator_layer" and not str(r.get("notes", "")).strip():
      errors.append(f"matrix.{t}.notes: a validator-layer row must say why no RTL evidence is needed")
    if app == "build_time" and ks != ["BUILD"]:
      errors.append(f"matrix.{t}: a build-time row uses BUILD only")
    if app == "build_and_rtl" and not ({"BUILD", "CS1_LOG"} <= set(ks)):
      errors.append(f"matrix.{t}: needs BUILD and CS1_LOG")
    if app == "perf" and "PERF" not in ks:
      errors.append(f"matrix.{t}: a perf row needs PERF")
    for s in r.get("subitems") or []:
      for k in s.get("evidence_kinds") or []:
        if k not in kinds_ok:
          errors.append(f"matrix.{t}.subitem {s.get('id')}: {k!r} is not an evidence kind")
      if s.get("status") == "PASS":
        errors.append(f"matrix.{t}.subitem {s.get('id')}: preset PASS")
      if s.get("scope") == "deferred" and not (s.get("defer_reason") and s.get("start_condition")):
        errors.append(f"matrix.{t}.subitem {s.get('id')}: a deferral needs a reason and a start condition")
      if s.get("scope") == "mandatory" and s.get("status") != "NOT_RUN":
        errors.append(f"matrix.{t}.subitem {s.get('id')}: a mandatory subitem starts NOT_RUN")
    if r.get("status") == "DEFERRED_NOT_APPLICABLE":
      if not (r.get("defer_reason") and r.get("start_condition")):
        errors.append(f"matrix.{t}: a deferral needs defer_reason and start_condition")
      if t not in REQ_DEFERRED_ROWS:
        errors.append(f"matrix.{t}: only {list(REQ_DEFERRED_ROWS)} may be deferred as a whole row")
    elif t in REQ_DEFERRED_ROWS:
      errors.append(f"matrix.{t}: must be deferred (CS2/CS3 exit not started)")
    if t in REQ_CS2_ROWS:
      sub = {s.get("id"): s for s in r.get("subitems") or []}
      if r.get("cs1_scope") != "cs1_subset" or "cs1_subset" not in sub or "cs2_completion" not in sub:
        errors.append(f"matrix.{t}: needs a mandatory CS1 subset and a deferred CS2 completion")
      elif (
        sub["cs1_subset"].get("scope") != "mandatory" or sub["cs2_completion"].get("scope") != "deferred"
      ):
        errors.append(f"matrix.{t}: the CS1 subset is mandatory and only the CS2 completion is deferred")
    if t in REQ_N2_ROWS:
      sub = {s.get("id"): s for s in r.get("subitems") or []}
      if sub.get("n1", {}).get("scope") != "mandatory" or sub.get("n2", {}).get("scope") != "deferred":
        errors.append(f"matrix.{t}: the N=1 part is mandatory and only the N>=2 part is deferred")
    if t == "T-35":
      sub = {s.get("id"): s for s in r.get("subitems") or []}
      for k in ("p0", "p1", "pn"):
        if sub.get(k, {}).get("scope") != "mandatory":
          errors.append(f"matrix.T-35.{k}: P0/P1/PN are mandatory")
      if sub.get("ps", {}).get("status") != "NOT_RUN" or sub.get("ps", {}).get("scope") == "deferred":
        errors.append("matrix.T-35.ps: PS is registered not_run, not turned into a deferral")
  declared = m.get("deferral_conditions") or {}
  for key in ("cs2", "cs3", "n2"):
    if not str(declared.get(key, "")).strip():
      errors.append(f"matrix.deferral_conditions.{key}: missing start condition")
  if (m.get("n05_override") or {}).get("t_id") != "T-20":
    errors.append("matrix.n05_override: N05 follows T-20")
  if "NOT_RUN" not in str(m.get("initial_status_rule", "")):
    errors.append("matrix.initial_status_rule: must state that every row starts NOT_RUN")


def check_field_request(req: Any, docs: dict[str, Any], errors: list[str]) -> None:
  if req.get("schema") != "cn_field_request/v1" or req.get("package") != "W7":
    errors.append("field_request: schema/package must be cn_field_request/v1 + W7")
  catalog = yaml.safe_load(CATALOG.read_text())
  ids = {f["name"] for d in catalog["identity_domains"].values() for f in d["fields"]}
  ids |= {"message_seq"}
  record_fields = {f["name"] for f in docs["record"]["fields"]}
  for item in req.get("log_fields") or []:
    name = item.get("name")
    if name not in record_fields:
      errors.append(f"field_request.log_fields.{name}: not a field of log_record_contract_v1.yaml")
    if item.get("from_identity") and name not in ids and name != "identity":
      errors.append(f"field_request.log_fields.{name}: not an identity field of the C009 catalog")
    if not str(item.get("why", "")).strip():
      errors.append(f"field_request.log_fields.{name}: needs a reason")
    for banned in BANNED:
      if banned in item:
        errors.append(f"field_request.log_fields.{name}.{banned}: invented value (INVENTED_VALUE)")
  asked = {i.get("name") for i in req.get("log_fields") or []}
  need = {
    "seq",
    "cycle",
    "phase",
    "protocol",
    "dir",
    "port",
    "len",
    "payload_hash",
    "payload_bytes",
    "blob_ref",
    "message_seq",
    "r1_seq",
    "identity",
  }
  if not need <= asked:
    errors.append(f"field_request.log_fields: missing {sorted(need - asked)}")
  phases = req.get("phase_enum") or {}
  if phases.get("from") != "protocol/cn_protocol_v1.yaml#phases.order":
    errors.append(
      "field_request.phase_enum: the phase enum comes from the C009 catalog, W7 defines no second one"
    )
  for item in req.get("contract_refs") or []:
    which = item.get("contract")
    if which not in docs:
      errors.append(f"field_request.contract_refs: unknown contract {which!r}")
      continue
    try:
      _dig(docs[which], tuple(str(item["key"]).split(".")))
    except (KeyError, TypeError):
      errors.append(f"field_request.contract_refs: key {item.get('key')!r} does not exist in {which}")
  if (req.get("consumes") or {}).get("w6_contracts") != [
    "protocol/w6_event_facility.contract.yaml",
    "protocol/w6_terminal_reset.contract.yaml",
  ]:
    errors.append("field_request.consumes: names both W6 contracts")


# ---------------------------------------------------------------------------------------------------
def validate_docs(
  docs: Any, matrix: Any, request: Any, req_rows: Any = None, plan_kinds: Any = None
) -> dict[str, Any]:
  errors: list[str] = []
  for name, doc in (("contracts", docs), ("matrix", matrix), ("field request", request)):
    if not isinstance(doc, dict):
      errors.append(f"{name}: root must be a mapping, got {type(doc).__name__}")
  if errors:
    return {"schema": "cn_w7_contract_report/v1", "valid": False, "errors": errors, "anchors_read_back": 0}

  def guarded(label: str, fn: Any, *args: Any) -> Any:
    try:
      return fn(*args)
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
      errors.append(f"{label}: document is malformed ({type(exc).__name__}: {exc})")
      return None

  anchors = 0
  for name, doc in docs.items():
    anchors += guarded("sources", check_sources, name, doc, errors) or 0
    if not isinstance(doc, dict) or doc.get("status") != "draft":
      errors.append(f"{name}.status: must stay draft (C020 seals, not this card)")
  guarded("window", check_window, docs["window"], errors)
  guarded("record", check_record, docs["record"], errors)
  guarded("verdict", check_verdict, docs["verdict"], errors)
  guarded("bundle", check_bundle, docs["bundle"], errors)
  guarded("perf", check_perf, docs["perf"], errors)
  guarded("unresolved", check_unresolved, docs, errors)
  guarded("matrix", check_matrix, matrix, docs["verdict"], errors, req_rows, plan_kinds)
  guarded("field request", check_field_request, request, docs, errors)
  return {
    "schema": "cn_w7_contract_report/v1",
    "valid": not errors,
    "errors": errors,
    "anchors_read_back": anchors,
    "rows": len(matrix.get("rows", [])) if isinstance(matrix, dict) else 0,
    "open": [
      "C2/C3/C4/C5 checker implementations (C040+)",
      "capture quota and perf thresholds (C019)",
      "binary log encoding (C017)",
      "every T row is NOT_RUN: no row has executed",
    ],
  }


def load_docs() -> dict[str, Any]:
  c = w7.Contracts()
  return {"window": c.window, "record": c.record, "verdict": c.verdict, "bundle": c.bundle, "perf": c.perf}


def validate() -> dict[str, Any]:
  return validate_docs(load_docs(), w7.load(w7.MATRIX), yaml.safe_load(REQUEST.read_text()))


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  report = validate()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / "w7_contract_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  if not report["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
