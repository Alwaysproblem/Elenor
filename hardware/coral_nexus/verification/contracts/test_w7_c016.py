"""C016 W7 trace-window / T-matrix / evidence-applicability contract tests.

The reference consumer (`w7_trace`) is built FROM the five W7 contracts and the T matrix, and the
required set is held in `validate_w7_contract`, so these tests drive real structured windows and
real verdict sets.
A pass here is a CONTRACT result: C2/C3/C4/C5 are specified, not implemented, and no T row has run.

Card pass-condition clauses -> tests:
  T-01..T-68 unique and complete, bound to the live requirement .. test_matrix_*
  every mandatory subitem has a type-correct requirement ......... test_evidence_kinds_*, test_row_kinds_*
  an unrun row can never parse as PASS ............................ test_unrun_*, test_preset_pass_*
  prefix contiguous but no correct seal -> rejected ................ test_truncated_*, test_seal_*
  missing blob -> rejected ......................................... test_missing_blob_*, test_hash_only_*
  unauthorized deferral -> rejected ................................ test_*deferral*
  legal-fault case-PASS vs DUT-ERROR, replay gap vs numeric FAIL ... test_outcome_*
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

H = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(H / "scripts/lib"))

import validate_w7_contract as vw  # noqa: E402
import w7_trace as w7  # noqa: E402


@pytest.fixture(scope="module")
def c() -> w7.Contracts:
  return w7.Contracts()


@pytest.fixture
def docs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
  return (
    copy.deepcopy(vw.load_docs()),
    copy.deepcopy(w7.load(w7.MATRIX)),
    yaml.safe_load(vw.REQUEST.read_text()),
  )


def row(m: dict[str, Any], t: str) -> dict[str, Any]:
  return next(x for x in m["rows"] if x["t_id"] == t)


def caught(
  d: Any, *, dm: Any = None, mm: Any = None, rm: Any = None, req: Any = None, needle: str = ""
) -> None:
  docs_, m, r = copy.deepcopy(d[0]), copy.deepcopy(d[1]), copy.deepcopy(d[2])
  rq = vw.requirement_rows()
  for fn, doc in ((dm, docs_), (mm, m), (rm, r)):
    if fn:
      fn(doc)
  if req:
    req(rq)
  rep = vw.validate_docs(docs_, m, r, rq)
  assert not rep["valid"], "the mutation was not detected"
  assert needle in " | ".join(rep["errors"]), rep["errors"]


def expect(code: str, fn: Any, *a: Any, **k: Any) -> None:
  with pytest.raises(w7.TraceError) as err:
    fn(*a, **k)
  assert err.value.code == code


# ---------------------------------------------------------------------------------------------------
# a real window
# ---------------------------------------------------------------------------------------------------
def make_blob(store: dict[str, bytes], data: bytes, kind: str) -> dict[str, Any]:
  h = hashlib.sha256(data).hexdigest()
  store[h] = data
  return {"sha256": h, "len": len(data), "blob_kind": kind}


def make_record(seq: int, cycle: int, phase: str = "P1", **kw: Any) -> dict[str, Any]:
  payload = kw.pop("payload", b"\x01\x02\x03\x04")
  rec: dict[str, Any] = {
    "seq": seq,
    "cycle": cycle,
    "phase": phase,
    "protocol": "CTRL",
    "dir": "req",
    "port": f"p{seq}",
    "valid": True,
    "ready": True,
    "accepted": True,
    "len": len(payload),
    "payload_hash": hashlib.sha256(payload).hexdigest(),
    "payload_bytes": payload,
    "reliability": "R1",
    "kind": "task",
    "r1_seq": seq,
  }
  rec.update(kw)
  return rec


def make_window(
  n: int = 4, *, end_type: str = "run_terminal"
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
  store: dict[str, bytes] = {}
  recs = [make_record(i, 10 + i) for i in range(n)]
  last_cycle = recs[-1]["cycle"]
  recs[-1]["kind"] = "RUN_END" if end_type == "run_terminal" else "task"
  end = {"type": end_type, "cycle": last_cycle}
  window: dict[str, Any] = {
    "schema_ver": "cn_trace/v1",
    "run_id": "r1",
    "window_id": "w1",
    "scenario_id": "s1",
    "t_id": "T-13",
    "window_kind": "full" if end_type == "run_terminal" else "fault_segment",
    "startup_epoch": {"includes_startup": True, "clock_ref": "cycle0"},
    "first_seq": 0,
    "last_seq": n - 1,
    "record_count": n,
    "gap_status": "none",
    "end_boundary": end,
    "blob_manifest": [make_blob(store, b"mem0", "initial_memory"), make_blob(store, b"elf0", "image_ref")],
    "config_refs": {
      "seed_list": [42],
      "backpressure_config_ref": "bp0",
      "inject_plan_ref": "ip0",
      "delay_config_ref": "dl0",
    },
    "records": recs,
  }
  window["seal"] = w7.seal_window(window)
  return window, store, end


def reseal(window: dict[str, Any]) -> None:
  window["seal"] = w7.seal_window(window)


def gate(window: Any, store: Any, end: Any, c: w7.Contracts) -> dict[str, Any]:
  return w7.replay_gate(window, store, end, c)


# ---------------------------------------------------------------------------------------------------
# committed files, CLI
# ---------------------------------------------------------------------------------------------------
def test_committed_contracts_validate() -> None:
  rep = vw.validate()
  assert rep["valid"], rep["errors"]
  assert rep["rows"] == 68 and rep["anchors_read_back"] >= 6


def test_cli_writes_report_and_exits_zero(tmp_path: Path) -> None:
  proc = subprocess.run(
    [sys.executable, str(H / "scripts/lib/validate_w7_contract.py"), "--out", str(tmp_path)],
    capture_output=True,
    text=True,
  )
  assert proc.returncode == 0, proc.stderr
  assert json.loads((tmp_path / "w7_contract_report.json").read_text())["valid"] is True


def test_non_mapping_documents_are_reported_not_raised() -> None:
  assert vw.validate_docs([], {}, None)["valid"] is False


def test_status_stays_draft(docs: Any) -> None:
  caught(docs, dm=lambda d: d["window"].__setitem__("status", "frozen"), needle="draft")
  caught(docs, mm=lambda m: m.__setitem__("status", "frozen"), needle="draft")


def test_every_citation_is_read_back(docs: Any) -> None:
  caught(
    docs,
    dm=lambda d: d["perf"]["report_keys"]["source"].__setitem__("line", 1),
    needle="SOURCE_LINE_MISMATCH",
  )
  caught(
    docs,
    dm=lambda d: d["window"]["time"]["display_conversion_source"].pop("needle"),
    needle="MISSING_SOURCE",
  )


# ---------------------------------------------------------------------------------------------------
# T matrix
# ---------------------------------------------------------------------------------------------------
def test_matrix_is_exactly_t01_to_t68_in_order(docs: Any) -> None:
  caught(
    docs, mm=lambda m: m["rows"].insert(5, copy.deepcopy(row(m, "T-05"))), needle="duplicates ['T-05']"
  )
  caught(docs, mm=lambda m: m["rows"].remove(row(m, "T-30")), needle="missing ['T-30']")
  caught(
    docs,
    mm=lambda m: m["rows"].append({**copy.deepcopy(row(m, "T-68")), "t_id": "T-69"}),
    needle="extra ['T-69']",
  )
  caught(docs, mm=lambda m: m["rows"].reverse(), needle="in order")


def test_matrix_is_bound_to_the_live_requirement_text(docs: Any) -> None:
  caught(
    docs,
    mm=lambda m: row(m, "T-14").__setitem__("req_text_sha256", "0" * 64),
    needle="changed under the matrix",
  )
  caught(docs, mm=lambda m: row(m, "T-14").__setitem__("req_line", 1), needle="req_line")
  caught(
    docs,
    req=lambda rq: rq.__setitem__("T-14", (rq["T-14"][0], rq["T-14"][1] + " edited")),
    needle="changed under the matrix",
  )
  caught(docs, req=lambda rq: rq.pop("T-30"), needle="does not have exactly")


@pytest.mark.parametrize(
  "field", ["title", "owner_packages", "impl_tasks", "scenario_source", "oracle", "status"]
)
def test_matrix_rows_need_every_field(docs: Any, field: str) -> None:
  caught(docs, mm=lambda m: row(m, "T-12").pop(field), needle=f"T-12.{field}")


def test_matrix_scenario_source_needs_a_plan_id(docs: Any) -> None:
  caught(
    docs, mm=lambda m: row(m, "T-12")["scenario_source"].__setitem__("ref", ""), needle="needs a plan id"
  )


def test_matrix_scope_and_applicability_are_closed_sets(docs: Any) -> None:
  caught(docs, mm=lambda m: row(m, "T-12").__setitem__("cs1_scope", "optional"), needle="not legal")
  caught(docs, mm=lambda m: row(m, "T-12").__setitem__("applicability", "anywhere"), needle="not legal")


def test_unrun_rows_cannot_be_pass_and_the_matrix_never_carries_pass(docs: Any) -> None:
  caught(docs, mm=lambda m: row(m, "T-12").__setitem__("status", "PASS"), needle="never 'PASS'")
  caught(docs, mm=lambda m: row(m, "T-12").__setitem__("status", ""), needle="T-12.status")
  caught(docs, mm=lambda m: row(m, "T-12").__setitem__("status", "FAIL"), needle="never 'FAIL'")
  caught(
    docs,
    mm=lambda m: m.__setitem__("initial_status_rule", "rows may start as PASS"),
    needle="initial_status_rule",
  )


def test_preset_pass_stays_rejected_even_when_the_guard_is_edited_too(
  docs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(vw, "REQ_MATRIX_STATUS", (*vw.REQ_MATRIX_STATUS, "PASS"))
  caught(docs, mm=lambda m: row(m, "T-12").__setitem__("status", "PASS"), needle="never carries PASS")


def test_every_executable_row_starts_not_run() -> None:
  m = w7.load(w7.MATRIX)
  for r in m["rows"]:
    assert r["status"] in ("NOT_RUN", "DEFERRED_NOT_APPLICABLE"), r["t_id"]
    for s in r["subitems"]:
      assert s["status"] == "NOT_RUN" or s["scope"] == "deferred", (r["t_id"], s["id"])


@pytest.mark.parametrize(
  ("t", "wrong"),
  [
    ("T-01", ["CS1_LOG"]),
    ("T-29", ["CS1_LOG"]),
    ("T-65", ["CS1_LOG"]),
    ("T-34", ["CS1_LOG"]),
    ("T-67", ["CS1_LOG", "PERF"]),
    ("T-12", ["CS1_LOG", "PMU"]),
  ],
)
def test_evidence_kinds_cannot_be_dropped_or_added(docs: Any, t: str, wrong: list[str]) -> None:
  assert row(docs[1], t)["evidence_kinds"] != wrong
  caught(
    docs, mm=lambda m: row(m, t).__setitem__("evidence_kinds", wrong), needle=f"matrix.{t}.evidence_kinds"
  )


def test_evidence_kind_edits_need_the_guard_the_matrix_and_the_plan_together(
  docs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setitem(vw.REQ_ROW_KINDS, "T-01", ("CS1_LOG",))
  caught(
    docs, mm=lambda m: row(m, "T-01").__setitem__("evidence_kinds", ["CS1_LOG"]), needle="the W7 plan names"
  )


def test_row_kinds_match_what_the_applicability_allows(docs: Any) -> None:
  caught(docs, mm=lambda m: row(m, "T-12").__setitem__("evidence_kinds", ["BUILD"]), needle="needs CS1_LOG")
  caught(
    docs, mm=lambda m: row(m, "T-12").__setitem__("evidence_kinds", ["CS1_LOG", "CS1_PY"]), needle="CS1_PY"
  )
  caught(
    docs,
    mm=lambda m: row(m, "T-12").__setitem__("evidence_kinds", ["CS1_LOG", "VIBES"]),
    needle="not an evidence kind",
  )
  caught(
    docs, mm=lambda m: row(m, "T-12").__setitem__("evidence_kinds", []), needle="at least one evidence kind"
  )
  caught(docs, mm=lambda m: row(m, "T-49").__setitem__("evidence_kinds", ["CS1_LOG"]), needle="CS1_PY only")
  caught(docs, mm=lambda m: row(m, "T-37").__setitem__("evidence_kinds", ["CS1_LOG"]), needle="BUILD only")
  caught(
    docs, mm=lambda m: row(m, "T-33").__setitem__("evidence_kinds", ["CS1_LOG"]), needle="BUILD and CS1_LOG"
  )
  caught(docs, mm=lambda m: row(m, "T-35").__setitem__("evidence_kinds", ["CS1_LOG"]), needle="needs PERF")


def test_a_validator_layer_row_must_say_why_no_rtl_evidence_is_needed(docs: Any) -> None:
  caught(docs, mm=lambda m: row(m, "T-49").__setitem__("notes", ""), needle="why no RTL evidence")


def test_not_every_row_demands_cs1_log() -> None:
  m = w7.load(w7.MATRIX)
  kinds = {r["t_id"]: r["evidence_kinds"] for r in m["rows"]}
  assert (
    "CS1_LOG" not in kinds["T-37"] and "CS1_LOG" not in kinds["T-49"] and "CS1_LOG" not in kinds["T-35"]
  )
  assert kinds["T-59"] == [] and kinds["T-60"] == []
  assert sum("CS1_LOG" in k for k in kinds.values()) == 63


def test_deferral_only_for_t59_t60_and_it_needs_reason_and_start_condition(docs: Any) -> None:
  caught(docs, mm=lambda m: row(m, "T-59").__setitem__("status", "NOT_RUN"), needle="must be deferred")
  caught(docs, mm=lambda m: row(m, "T-59").pop("defer_reason"), needle="defer_reason and start_condition")
  caught(
    docs, mm=lambda m: row(m, "T-60").pop("start_condition"), needle="defer_reason and start_condition"
  )
  caught(
    docs,
    mm=lambda m: row(m, "T-61").update(
      {"status": "DEFERRED_NOT_APPLICABLE", "defer_reason": "hard", "start_condition": "later"}
    ),
    needle="may be deferred as a whole row",
  )
  caught(
    docs,
    mm=lambda m: row(m, "T-20").update(
      {"status": "DEFERRED_NOT_APPLICABLE", "defer_reason": "CS2", "start_condition": "D-24"}
    ),
    needle="may be deferred as a whole row",
  )


@pytest.mark.parametrize(
  "t", ["T-15", "T-19", "T-31", "T-42", "T-43", "T-46", "T-47", "T-48", "T-52", "T-57"]
)
def test_cs2_rows_keep_a_mandatory_cs1_subset_and_defer_only_the_completion(docs: Any, t: str) -> None:
  def defer_subset(m: dict[str, Any]) -> None:
    for s in row(m, t)["subitems"]:
      if s["id"] == "cs1_subset":
        s["scope"] = "deferred"

  def mandatory_completion(m: dict[str, Any]) -> None:
    for s in row(m, t)["subitems"]:
      if s["id"] == "cs2_completion":
        s["scope"] = "mandatory"

  caught(docs, mm=defer_subset, needle=t)
  caught(docs, mm=mandatory_completion, needle=t)
  caught(docs, mm=lambda m: row(m, t).update({"subitems": [], "cs1_scope": "mandatory"}), needle=t)
  caught(
    docs,
    mm=lambda m: row(m, t).update(
      {"status": "DEFERRED_NOT_APPLICABLE", "defer_reason": "CS2", "start_condition": "D-24"}
    ),
    needle="may be deferred as a whole row",
  )


@pytest.mark.parametrize("t", ["T-28", "T-68"])
def test_n2_rows_keep_the_n1_part_mandatory(docs: Any, t: str) -> None:
  def defer_n1(m: dict[str, Any]) -> None:
    for s in row(m, t)["subitems"]:
      if s["id"] == "n1":
        s["scope"] = "deferred"

  def mandatory_n2(m: dict[str, Any]) -> None:
    for s in row(m, t)["subitems"]:
      if s["id"] == "n2":
        s["scope"] = "mandatory"

  caught(docs, mm=defer_n1, needle=t)
  caught(docs, mm=mandatory_n2, needle=t)


def test_a_deferred_subitem_needs_its_reason_and_start_condition(docs: Any) -> None:
  def drop(key: str) -> Any:
    def f(m: dict[str, Any]) -> None:
      for s in row(m, "T-15")["subitems"]:
        if s["id"] == "cs2_completion":
          s.pop(key)

    return f

  caught(docs, mm=drop("defer_reason"), needle="a deferral needs a reason and a start condition")
  caught(docs, mm=drop("start_condition"), needle="a deferral needs a reason and a start condition")
  caught(docs, mm=lambda m: m["deferral_conditions"].pop("cs2"), needle="deferral_conditions.cs2")
  caught(docs, mm=lambda m: m["deferral_conditions"].__setitem__("n2", ""), needle="deferral_conditions.n2")


def test_ps_is_registered_not_run_and_never_turned_into_a_deferral(docs: Any) -> None:
  def ps(f: Any) -> Any:
    def g(m: dict[str, Any]) -> None:
      for s in row(m, "T-35")["subitems"]:
        if s["id"] == "ps":
          f(s)

    return g

  caught(docs, mm=ps(lambda s: s.__setitem__("scope", "deferred")), needle="T-35")
  caught(docs, mm=ps(lambda s: s.__setitem__("status", "PASS")), needle="preset PASS")
  for k in ("p0", "p1", "pn"):

    def defer(m: dict[str, Any], k: str = k) -> None:
      for s in row(m, "T-35")["subitems"]:
        if s["id"] == k:
          s["scope"] = "deferred"

    caught(docs, mm=defer, needle="T-35")
  caught(docs, mm=lambda m: row(m, "T-35")["subitems"].pop(2), needle="T-35.pn")


def test_n05_follows_t20(docs: Any) -> None:
  caught(docs, mm=lambda m: m["n05_override"].__setitem__("t_id", "T-21"), needle="N05 follows T-20")


# ---------------------------------------------------------------------------------------------------
# C1 and window sealing
# ---------------------------------------------------------------------------------------------------
def test_a_well_formed_window_has_no_c1_findings(c: w7.Contracts) -> None:
  window, _, _ = make_window()
  assert w7.check_window(window, c) == []


def codes(window: dict[str, Any], c: w7.Contracts) -> set[str]:
  return {f["code"] for f in w7.check_window(window, c)}


def test_c1_flags_a_seq_gap_and_names_the_seq(c: w7.Contracts) -> None:
  window, _, _ = make_window(5)
  del window["records"][2]
  found = [f for f in w7.check_window(window, c) if f["code"] == "SEQ_GAP"]
  assert found and found[0]["seq"] == 3


def test_c1_flags_a_duplicate_cycle_phase_port_dir_key(c: w7.Contracts) -> None:
  window, _, _ = make_window()
  window["records"][1].update(cycle=window["records"][0]["cycle"], port=window["records"][0]["port"])
  assert "DUPLICATE_KEY" in codes(window, c)


def test_req_and_resp_in_the_same_cycle_are_two_keys(c: w7.Contracts) -> None:
  window, _, _ = make_window()
  r = window["records"]
  r[1].update(cycle=r[0]["cycle"], port=r[0]["port"], dir="resp")
  assert "DUPLICATE_KEY" not in codes(window, c)


@pytest.mark.parametrize(
  ("mutate", "code"),
  [
    (lambda r: r.__setitem__("cycle", 1.5), "BAD_TYPE"),
    (lambda r: r.__setitem__("seq", True), "BAD_TYPE"),
    (lambda r: r.__setitem__("len", "4"), "BAD_TYPE"),
    (lambda r: r.__setitem__("valid", 1), "BAD_TYPE"),
    (lambda r: r.__setitem__("phase", "P9"), "BAD_PHASE"),
    (lambda r: r.__setitem__("protocol", "SIDEBAND"), "BAD_PROTOCOL"),
    (lambda r: r.__setitem__("dir", "both"), "BAD_DIR"),
    (lambda r: r.__setitem__("reliability", "R3"), "BAD_RELIABILITY"),
    (lambda r: r.__setitem__("ts_us", 1.234), "QUANTIZED_FIELD_IN_RAW_LOG"),
    (lambda r: r.__setitem__("wall_time", 5), "QUANTIZED_FIELD_IN_RAW_LOG"),
    (lambda r: r.__setitem__("bogus", 1), "UNKNOWN_FIELD"),
    (lambda r: r.pop("payload_hash"), "MISSING_FIELD"),
    (lambda r: r.__setitem__("accepted", False), "HANDSHAKE_MISMATCH"),
    (lambda r: r.update(valid=False, ready=True, accepted=True), "HANDSHAKE_MISMATCH"),
    (lambda r: r.__setitem__("payload_hash", "0" * 64), "HASH_MISMATCH"),
    (lambda r: r.__setitem__("len", 99), "LEN_MISMATCH"),
    (lambda r: r.__setitem__("r1_seq", 9), "R1_MISSING"),
    (lambda r: r.update(protocol="DEBUG"), "R1_ON_DEBUG"),
  ],
)
def test_c1_rejects_each_kind_of_damage_with_its_code(c: w7.Contracts, mutate: Any, code: str) -> None:
  window, _, _ = make_window()
  mutate(window["records"][1])
  assert code in codes(window, c)


def test_c1_flags_a_cycle_that_goes_backwards(c: w7.Contracts) -> None:
  window, _, _ = make_window()
  window["records"][2]["cycle"] = 1
  assert "CYCLE_REGRESSION" in codes(window, c)


def test_c1_flags_a_phase_that_goes_backwards_within_a_cycle(c: w7.Contracts) -> None:
  window, _, _ = make_window()
  for rec, phase in zip(window["records"], ["P3", "P1", "P4", "P4"], strict=True):
    rec.update(cycle=20, phase=phase)
  found = [f for f in w7.check_window(window, c) if f["code"] == "CYCLE_REGRESSION"]
  assert [f["seq"] for f in found] == [1], "only the P3 -> P1 step goes backwards"


def test_c1_accepts_phases_that_advance_within_a_cycle(c: w7.Contracts) -> None:
  window, _, _ = make_window()
  for rec, phase in zip(window["records"], ["P0", "P1", "P3", "P4"], strict=True):
    rec.update(cycle=20, phase=phase)
  assert "CYCLE_REGRESSION" not in codes(window, c)


def test_c1_flags_a_payload_that_changes_while_valid_and_not_ready(c: w7.Contracts) -> None:
  window, _, _ = make_window()
  r = window["records"]
  for i in (0, 1):
    r[i].update(port="same", dir="req", valid=True, ready=False, accepted=False)
  r[1]["payload_hash"] = hashlib.sha256(b"other").hexdigest()
  r[1]["payload_bytes"] = b"other"
  r[1]["len"] = 5
  assert "UNSTABLE_PAYLOAD" in codes(window, c)


def test_c1_accepts_a_stable_stalled_payload(c: w7.Contracts) -> None:
  window, _, _ = make_window()
  r = window["records"]
  for i in (0, 1):
    r[i].update(port="same", valid=True, ready=False, accepted=False)
  assert "UNSTABLE_PAYLOAD" not in codes(window, c)


def test_a_header_with_an_unknown_or_missing_field_is_flagged(c: w7.Contracts) -> None:
  window, _, _ = make_window()
  window["surprise"] = 1
  window.pop("run_id")
  found = codes(window, c)
  assert {"UNKNOWN_FIELD", "MISSING_HEADER_FIELD"} <= found
  window2, _, _ = make_window()
  window2["window_kind"] = "summary"
  assert "BAD_WINDOW_KIND" in codes(window2, c)


def test_message_seq_and_seq_are_separate_gap_domains(c: w7.Contracts) -> None:
  window, _, _ = make_window()
  r = window["records"]
  for i, rec in enumerate(r):
    rec["message_seq"] = i
    rec["identity"] = {"tile_id": 0}
  assert codes(window, c) == set()
  r[2]["message_seq"] = 9  # transport gap while seq stays contiguous
  found = codes(window, c)
  assert "MESSAGE_SEQ_GAP" in found and "SEQ_GAP" not in found
  window2, _, _ = make_window(5)
  for i, rec in enumerate(window2["records"]):
    rec["message_seq"] = i
    rec["identity"] = {"tile_id": 0}
  del window2["records"][2]  # seq gap while message_seq... also has a hole here
  window3, _, _ = make_window()
  for i, rec in enumerate(window3["records"]):
    rec["identity"] = {"tile_id": 0}
    rec["message_seq"] = i
  window3["records"][1]["seq"] = 7  # seq gap, message_seq contiguous
  found3 = codes(window3, c)
  assert "SEQ_GAP" in found3 and "MESSAGE_SEQ_GAP" not in found3


def test_message_seq_is_counted_per_tile_channel_and_direction(c: w7.Contracts) -> None:
  window, _, _ = make_window(6)
  r = window["records"]
  plan = [(0, "CTRL", "req", 0), (1, "CTRL", "req", 0), (0, "CTRL", "req", 1), (1, "CTRL", "req", 1)]
  plan += [(0, "EVENT", "req", 0), (0, "CTRL", "resp", 0)]
  for rec, (tile, protocol, direction, mseq) in zip(r, plan, strict=True):
    rec.update(protocol=protocol, dir=direction, message_seq=mseq, identity={"tile_id": tile})
  assert "MESSAGE_SEQ_GAP" not in codes(window, c), (
    "independent streams must not be compared with each other"
  )
  r[2]["message_seq"] = 5  # tile 0's CTRL/req stream skips ahead
  found = [f for f in w7.check_window(window, c) if f["code"] == "MESSAGE_SEQ_GAP"]
  assert [f["seq"] for f in found] == [2]


def test_a_seq_gap_is_not_repaired_by_a_contiguous_message_seq_in_the_replay_gate(c: w7.Contracts) -> None:
  window, store, end = make_window(5)
  for i, rec in enumerate(window["records"]):
    rec["message_seq"] = i
    rec["identity"] = {"tile_id": 0}
  del window["records"][2]
  reseal(window)
  out = gate(window, store, end, c)
  assert out["verdict"] == "NOT_FULLY_REPLAYABLE" and "seq_gap" in out["reasons"]


def test_seal_is_computed_from_the_records(c: w7.Contracts) -> None:
  window, _, end = make_window()
  s = window["seal"]
  assert (s["first_seq"], s["last_seq"], s["record_count"]) == (0, 3, 4)
  assert s["end_boundary"] == end and s["gap_status"] == "none" and len(s["records_sha256"]) == 64


# ---------------------------------------------------------------------------------------------------
# C6 replay gate
# ---------------------------------------------------------------------------------------------------
def test_a_complete_window_is_replayable_with_exit_zero(c: w7.Contracts) -> None:
  window, store, end = make_window()
  out = gate(window, store, end, c)
  assert out == {
    "verdict": "REPLAYABLE",
    "reasons": [],
    "exit_code": 0,
    "summary": {"records": 4, "first_seq": 0, "findings": 0},
  }


def test_a_known_fault_segment_window_is_replayable_and_may_leave_work_pending(c: w7.Contracts) -> None:
  window, store, end = make_window(end_type="declared_capture_point")
  assert window["records"][-1]["kind"] != "RUN_END"
  out = gate(window, store, end, c)
  assert out["verdict"] == "REPLAYABLE", out["reasons"]


def test_a_truncated_window_with_a_contiguous_prefix_and_a_recomputed_seal_is_rejected(
  c: w7.Contracts,
) -> None:
  """The card's named negative: the prefix is contiguous and the seal is internally consistent, but the
  records stop before the declared end, which the scenario holds outside the window."""
  window, store, end = make_window(6)
  del window["records"][-2:]
  window["end_boundary"] = {"type": "run_terminal", "cycle": window["records"][-1]["cycle"]}
  reseal(window)
  assert w7.check_window(window, c) == []  # C1 alone cannot see it
  out = gate(window, store, end, c)
  assert out["verdict"] == "NOT_FULLY_REPLAYABLE" and out["exit_code"] == 3
  assert "end_boundary_differs_from_declaration" in out["reasons"]


def test_truncation_that_keeps_the_declared_boundary_is_caught_by_the_last_record(c: w7.Contracts) -> None:
  window, store, end = make_window(6)
  del window["records"][-2:]
  reseal(window)
  out = gate(window, store, end, c)
  assert "truncated_end" in out["reasons"] or any(r.startswith("seal_mismatch") for r in out["reasons"])
  assert out["verdict"] == "NOT_FULLY_REPLAYABLE"


def test_a_stale_seal_after_truncation_is_a_seal_mismatch(c: w7.Contracts) -> None:
  window, store, end = make_window(6)
  del window["records"][-1]
  out = gate(window, store, end, c)
  assert any(r.startswith("seal_mismatch") for r in out["reasons"]) and out["exit_code"] == 3


@pytest.mark.parametrize("name", ["first_seq", "last_seq", "record_count", "records_sha256", "gap_status"])
def test_every_seal_field_is_checked_against_the_records(c: w7.Contracts, name: str) -> None:
  window, store, end = make_window()
  window["seal"][name] = (
    "none-of-these"
    if name == "gap_status"
    else (window["seal"][name] + 1 if name != "records_sha256" else "0" * 64)
  )
  out = gate(window, store, end, c)
  assert out["verdict"] == "NOT_FULLY_REPLAYABLE"


def test_a_missing_or_incomplete_seal_is_not_replayable(c: w7.Contracts) -> None:
  window, store, end = make_window()
  window.pop("seal")
  assert "missing_seal" in gate(window, store, end, c)["reasons"]
  window2, store2, end2 = make_window()
  window2["seal"].pop("records_sha256")
  assert "incomplete_seal" in gate(window2, store2, end2, c)["reasons"]


def test_a_payload_edit_with_the_hash_left_alone_is_caught_twice(c: w7.Contracts) -> None:
  window, store, end = make_window()
  window["records"][1]["payload_bytes"] = b"\xff\xff\xff\xff"
  out = gate(window, store, end, c)
  assert "hash_mismatch" in out["reasons"] and any(r.startswith("seal_mismatch") for r in out["reasons"])


@pytest.mark.parametrize("status", ["quota", "drop", "truncate"])
def test_a_gap_status_makes_the_window_not_replayable_even_if_resealed(
  c: w7.Contracts, status: str
) -> None:
  window, store, end = make_window()
  window["gap_status"] = status
  reseal(window)
  out = gate(window, store, end, c)
  assert out["verdict"] == "NOT_FULLY_REPLAYABLE" and f"gap:{status}" in out["reasons"]


def test_the_headers_gap_status_is_judged_on_its_own_not_only_through_the_seal(c: w7.Contracts) -> None:
  window, store, end = make_window()
  window["gap_status"] = "quota"  # the seal still says none: stale seal AND a header gap
  out = gate(window, store, end, c)
  assert "gap:quota" in out["reasons"] and "seal_mismatch:gap_status" in out["reasons"]
  window2, store2, end2 = make_window()
  window2["seal"]["gap_status"] = "drop"  # the seal admits a gap the header does not
  assert "gap:drop" in gate(window2, store2, end2, c)["reasons"]


def test_a_missing_blob_is_not_replayable(c: w7.Contracts) -> None:
  window, store, end = make_window()
  store.pop(window["blob_manifest"][0]["sha256"])
  out = gate(window, store, end, c)
  assert out["verdict"] == "NOT_FULLY_REPLAYABLE" and "missing_blob" in out["reasons"]


def test_a_tampered_blob_is_a_hash_mismatch_and_exits_three(c: w7.Contracts) -> None:
  window, store, end = make_window()
  store[window["blob_manifest"][1]["sha256"]] = b"tampered"
  out = gate(window, store, end, c)
  assert "blob_hash_mismatch" in out["reasons"] and out["exit_code"] == 3


def test_a_hash_alone_cannot_replay_a_record(c: w7.Contracts) -> None:
  window, store, end = make_window()
  window["records"][1].pop("payload_bytes")
  reseal(window)
  out = gate(window, store, end, c)
  assert "hash_only_record" in out["reasons"]


def test_a_blob_ref_must_be_listed_and_retrievable(c: w7.Contracts) -> None:
  window, store, end = make_window()
  data = b"payload-in-a-blob"
  entry = make_blob(store, data, "payload")
  window["records"][1].pop("payload_bytes")
  window["records"][1].update(blob_ref=entry["sha256"], len=len(data), payload_hash=entry["sha256"])
  window["blob_manifest"].append(entry)
  reseal(window)
  assert gate(window, store, end, c)["verdict"] == "REPLAYABLE"
  window["blob_manifest"].pop()
  reseal(window)
  assert "unlisted_blob_ref" in gate(window, store, end, c)["reasons"]


@pytest.mark.parametrize(
  ("kind", "reason"), [("initial_memory", "missing_initial_snapshot"), ("image_ref", "missing_image")]
)
def test_the_initial_snapshot_and_the_image_must_be_listed(c: w7.Contracts, kind: str, reason: str) -> None:
  window, store, end = make_window()
  window["blob_manifest"] = [b for b in window["blob_manifest"] if b["blob_kind"] != kind]
  reseal(window)
  assert reason in gate(window, store, end, c)["reasons"]


@pytest.mark.parametrize(
  "name", ["seed_list", "backpressure_config_ref", "inject_plan_ref", "delay_config_ref"]
)
def test_seed_and_backpressure_configuration_must_be_present(c: w7.Contracts, name: str) -> None:
  window, store, end = make_window()
  window["config_refs"].pop(name)
  reseal(window)
  assert f"missing_config:{name}" in gate(window, store, end, c)["reasons"]


def test_an_empty_seed_list_is_missing(c: w7.Contracts) -> None:
  window, store, end = make_window()
  window["config_refs"]["seed_list"] = []
  reseal(window)
  assert "missing_config:seed_list" in gate(window, store, end, c)["reasons"]


def test_the_startup_epoch_must_be_declared(c: w7.Contracts) -> None:
  window, store, end = make_window()
  window["startup_epoch"] = {"includes_startup": False}
  reseal(window)
  assert "missing_startup_epoch" in gate(window, store, end, c)["reasons"]
  window2, store2, end2 = make_window()
  window2["startup_epoch"] = {"includes_startup": False, "clock_ref": "cycle0"}
  reseal(window2)
  assert gate(window2, store2, end2, c)["verdict"] == "REPLAYABLE"


def test_the_end_boundary_is_checked_against_the_scenarios_declaration(c: w7.Contracts) -> None:
  window, store, end = make_window()
  window["end_boundary"] = {"type": "best_effort", "cycle": end["cycle"]}
  reseal(window)
  assert "bad_end_boundary" in gate(window, store, end, c)["reasons"]
  window2, store2, end2 = make_window()
  out = gate(window2, store2, {**end2, "cycle": end2["cycle"] + 5}, c)
  assert "end_boundary_differs_from_declaration" in out["reasons"]


def test_a_run_terminal_window_must_end_with_run_end(c: w7.Contracts) -> None:
  window, store, end = make_window()
  window["records"][-1]["kind"] = "task"
  reseal(window)
  assert "truncated_end" in gate(window, store, end, c)["reasons"]


def test_a_malformed_window_is_not_replayable_and_does_not_crash(c: w7.Contracts) -> None:
  window, store, end = make_window()
  window["records"][1]["cycle"] = 1.5
  reseal(window)
  out = gate(window, store, end, c)
  assert out["verdict"] == "NOT_FULLY_REPLAYABLE" and "malformed_window" in out["reasons"]


def test_an_empty_window_is_not_replayable(c: w7.Contracts) -> None:
  window, store, end = make_window()
  window["records"] = []
  reseal(window)
  assert gate(window, store, end, c)["verdict"] == "NOT_FULLY_REPLAYABLE"


def test_replayable_is_not_a_task_success_claim(c: w7.Contracts) -> None:
  window, store, end = make_window(end_type="declared_capture_point")
  assert gate(window, store, end, c)["verdict"] == "REPLAYABLE"
  out = w7.classify_outcome(c, expected_dut_error=False, dut_ended_in_error=False, pending_at_end=True)
  assert out == {"class": "PENDING_NOT_COMPLETE", "t_verdict": "FAIL"}


# ---------------------------------------------------------------------------------------------------
# outcome classes
# ---------------------------------------------------------------------------------------------------
def test_outcome_a_legal_injected_fault_that_ends_as_expected_is_a_case_pass(c: w7.Contracts) -> None:
  out = w7.classify_outcome(c, expected_dut_error=True, dut_ended_in_error=True)
  assert out == {"class": "CASE_PASS", "t_verdict": "PASS"}


def test_outcome_an_unexpected_dut_error_fails(c: w7.Contracts) -> None:
  assert w7.classify_outcome(c, expected_dut_error=False, dut_ended_in_error=True) == {
    "class": "DUT_ERROR",
    "t_verdict": "FAIL",
  }


def test_outcome_an_injected_fault_that_did_not_happen_is_not_a_pass(c: w7.Contracts) -> None:
  assert w7.classify_outcome(c, expected_dut_error=True, dut_ended_in_error=False)["t_verdict"] == "FAIL"


def test_outcome_a_replay_gap_is_rejected_not_a_numeric_fail(c: w7.Contracts) -> None:
  out = w7.classify_outcome(c, expected_dut_error=False, dut_ended_in_error=False, replayable=False)
  assert out == {"class": "NOT_FULLY_REPLAYABLE", "t_verdict": "REJECTED"}
  numeric = w7.classify_outcome(
    c, expected_dut_error=False, dut_ended_in_error=False, numeric_mismatch=True
  )
  assert numeric == {"class": "NUMERIC_FAIL", "t_verdict": "FAIL"} and numeric != out


def test_outcome_a_gap_outranks_a_numeric_mismatch(c: w7.Contracts) -> None:
  out = w7.classify_outcome(
    c, expected_dut_error=False, dut_ended_in_error=False, numeric_mismatch=True, replayable=False
  )
  assert out["class"] == "NOT_FULLY_REPLAYABLE"


def test_outcome_checker_failure_fails(c: w7.Contracts) -> None:
  assert (
    w7.classify_outcome(c, expected_dut_error=False, dut_ended_in_error=False, checker_failed=True)[
      "t_verdict"
    ]
    == "FAIL"
  )


def test_outcome_mapping_is_held_in_the_contract(docs: Any) -> None:
  caught(
    docs,
    dm=lambda d: d["verdict"]["outcome_classes"]["maps_to_t_verdict"].__setitem__("DUT_ERROR", "PASS"),
    needle="maps_to_t_verdict",
  )
  caught(
    docs,
    dm=lambda d: d["verdict"]["outcome_classes"]["maps_to_t_verdict"].__setitem__(
      "PENDING_NOT_COMPLETE", "PASS"
    ),
    needle="maps_to_t_verdict",
  )
  caught(
    docs,
    dm=lambda d: d["verdict"]["outcome_classes"]["maps_to_t_verdict"].__setitem__(
      "NOT_FULLY_REPLAYABLE", "FAIL"
    ),
    needle="maps_to_t_verdict",
  )
  caught(
    docs,
    dm=lambda d: d["verdict"]["outcome_classes"].__setitem__("legal_injected_fault_is_a_case_pass", False),
    needle="case pass",
  )


# ---------------------------------------------------------------------------------------------------
# C7 manifest gate
# ---------------------------------------------------------------------------------------------------
CHECKERS_PASS = [{"name": f"C{i}", "status": "PASS", "reason": "ok", "evidence": []} for i in range(1, 8)]


def put(root: Path, rel: str, data: bytes) -> dict[str, str]:
  p = root / rel
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_bytes(data)
  return {"path": rel, "sha256": hashlib.sha256(data).hexdigest()}


def ev(root: Path, t: str, kind: str, tag: str = "") -> dict[str, str]:
  return {"kind": kind, **put(root, f"{t}/{kind}{tag}.bin", f"{t}:{kind}:{tag}".encode()), "t_id": t}


def verdict_for(m: dict[str, Any], root: Path, t: str, status: str = "PASS") -> dict[str, Any]:
  r = row(m, t)
  v: dict[str, Any] = {
    "schema": "cn_verdict/1",
    "run_id": "r1",
    "scenario_id": f"s-{t}",
    "t_id": t,
    "t_verdict": status,
    "phase": "G5",
    "checker": copy.deepcopy(CHECKERS_PASS),
    "evidence": [],
    "t_scope": "mandatory",
    "notes": "",
  }
  if status != "PASS":
    return v
  if r["subitems"] and any(s["scope"] == "mandatory" for s in r["subitems"]):
    v["subitems"] = {}
    for s in r["subitems"]:
      if s["scope"] == "mandatory":
        v["subitems"][s["id"]] = {
          "status": "PASS",
          "evidence": [ev(root, t, k, s["id"]) for k in s["evidence_kinds"]],
        }
      elif s["scope"] == "deferred":
        v["subitems"][s["id"]] = {
          "status": "DEFERRED_NOT_APPLICABLE",
          "defer_reason": s["defer_reason"],
          "start_condition": s["start_condition"],
        }
      else:
        v["subitems"][s["id"]] = {"status": "NOT_RUN"}
  else:
    v["evidence"] = [ev(root, t, k) for k in r["evidence_kinds"]]
  return v


def full_set(m: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
  out: dict[str, dict[str, Any]] = {}
  for r in m["rows"]:
    t = r["t_id"]
    if r["status"] == "DEFERRED_NOT_APPLICABLE":
      out[t] = {
        **verdict_for(m, root, t, "DEFERRED_NOT_APPLICABLE"),
        "defer_reason": r["defer_reason"],
        "start_condition": r["start_condition"],
        "t_scope": "deferred",
      }
    else:
      out[t] = verdict_for(m, root, t)
  return out


@pytest.fixture
def matrix() -> dict[str, Any]:
  return copy.deepcopy(w7.load(w7.MATRIX))


def test_a_complete_honest_set_passes_the_gate(matrix: Any, c: w7.Contracts, tmp_path: Path) -> None:
  out = w7.evaluate_manifest(matrix, full_set(matrix, tmp_path), tmp_path, c)
  assert out["manifest_valid"] and out["gate_pass"], out["problems"][:3]
  assert out["rows_checked"] == 68 and out["accepted"]["T-59"] == "DEFERRED_NOT_APPLICABLE"


def test_an_all_not_run_set_is_a_valid_manifest_but_never_a_gate_pass(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = {r["t_id"]: {**verdict_for(matrix, tmp_path, r["t_id"], "NOT_RUN")} for r in matrix["rows"]}
  for t in ("T-59", "T-60"):
    vs[t] = {
      **verdict_for(matrix, tmp_path, t, "DEFERRED_NOT_APPLICABLE"),
      "defer_reason": row(matrix, t)["defer_reason"],
      "start_condition": row(matrix, t)["start_condition"],
      "t_scope": "deferred",
    }
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["manifest_valid"] is True and out["gate_pass"] is False
  assert out["accepted"]["T-12"] == "NOT_RUN"


def test_one_unrun_mandatory_row_blocks_the_gate(matrix: Any, c: w7.Contracts, tmp_path: Path) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-29"] = verdict_for(matrix, tmp_path, "T-29", "NOT_RUN")
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["gate_pass"] is False and out["manifest_valid"] is True


def test_a_missing_row_invalidates_the_manifest(matrix: Any, c: w7.Contracts, tmp_path: Path) -> None:
  vs = full_set(matrix, tmp_path)
  vs.pop("T-40")
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["manifest_valid"] is False and out["gate_pass"] is False


def test_a_pass_with_no_evidence_is_rejected(matrix: Any, c: w7.Contracts, tmp_path: Path) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-12"]["evidence"] = []
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["accepted"]["T-12"] == "REJECTED" and out["gate_pass"] is False


def test_a_pass_citing_a_file_that_does_not_exist_dangles(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  (tmp_path / vs["T-12"]["evidence"][0]["path"]).unlink()
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["manifest_valid"] is False
  assert any(p["code"] == "DANGLING_REF" and p["t_id"] == "T-12" for p in out["problems"])


def test_a_pass_citing_a_file_that_merely_exists_but_was_changed_is_a_hash_mismatch(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  (tmp_path / vs["T-12"]["evidence"][0]["path"]).write_bytes(b"forged")
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["manifest_valid"] is False
  assert any(p["code"] == "HASH_MISMATCH" and p["t_id"] == "T-12" for p in out["problems"])


def test_evidence_for_another_row_does_not_count(matrix: Any, c: w7.Contracts, tmp_path: Path) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-12"]["evidence"] = [ev(tmp_path, "T-13", "CS1_LOG")]
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["accepted"]["T-12"] == "REJECTED"
  assert any(p["code"] == "DANGLING_REF" for p in out["problems"])


def test_a_required_kind_that_is_missing_rejects_the_row(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-29"]["evidence"] = [e for e in vs["T-29"]["evidence"] if e["kind"] != "BIT_ORACLE"]
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["accepted"]["T-29"] == "REJECTED"
  assert any("needs BIT_ORACLE" in p["detail"] for p in out["problems"])


def test_an_evidence_kind_outside_the_contract_is_refused_even_when_the_required_ones_are_present(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-12"]["evidence"].append(ev(tmp_path, "T-12", "MANUAL"))
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["accepted"]["T-12"] == "REJECTED"
  assert any(p["code"] == "KIND_NOT_ALLOWED" and "MANUAL" in p["detail"] for p in out["problems"])


def test_a_python_cs1_py_pass_cannot_stand_in_for_a_missing_cs1_log(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-12"]["evidence"] = [ev(tmp_path, "T-12", "CS1_PY")]
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["accepted"]["T-12"] == "REJECTED"
  assert any(p["code"] == "KIND_NOT_ALLOWED" for p in out["problems"])


def test_cs1_py_alone_is_enough_for_the_validator_layer_row(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  assert [e["kind"] for e in vs["T-49"]["evidence"]] == ["CS1_PY"]
  assert w7.evaluate_manifest(matrix, vs, tmp_path, c)["accepted"]["T-49"] == "PASS"
  vs["T-49"]["evidence"] = [ev(tmp_path, "T-49", "CS1_LOG")]
  assert w7.evaluate_manifest(matrix, vs, tmp_path, c)["accepted"]["T-49"] == "REJECTED"


@pytest.mark.parametrize("bad", ["C1", "C4", "C7"])
def test_a_cs1_log_pass_needs_every_checker_to_pass(
  matrix: Any, c: w7.Contracts, tmp_path: Path, bad: str
) -> None:
  vs = full_set(matrix, tmp_path)
  for x in vs["T-12"]["checker"]:
    if x["name"] == bad:
      x["status"] = "FAIL"
  assert w7.evaluate_manifest(matrix, vs, tmp_path, c)["accepted"]["T-12"] == "REJECTED"


def test_a_missing_checker_rejects_the_row(matrix: Any, c: w7.Contracts, tmp_path: Path) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-12"]["checker"] = vs["T-12"]["checker"][:-1]
  assert w7.evaluate_manifest(matrix, vs, tmp_path, c)["accepted"]["T-12"] == "REJECTED"


def test_a_harness_only_pass_is_not_cs1_evidence(matrix: Any, c: w7.Contracts, tmp_path: Path) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-12"]["t_scope"] = "harness_only"
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["accepted"]["T-12"] == "REJECTED" and out["gate_pass"] is False


def test_an_unknown_verdict_value_invalidates_the_manifest(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-12"]["t_verdict"] = "MAYBE"
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["manifest_valid"] is False


def test_a_verdict_missing_a_required_field_is_rejected(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-12"].pop("checker")
  assert w7.evaluate_manifest(matrix, vs, tmp_path, c)["accepted"]["T-12"] == "REJECTED"


def test_deferral_is_accepted_only_where_the_matrix_defers(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-12"] = {
    **verdict_for(matrix, tmp_path, "T-12", "DEFERRED_NOT_APPLICABLE"),
    "defer_reason": "hard",
    "start_condition": "later",
    "t_scope": "deferred",
  }
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["manifest_valid"] is False and any(
    p["code"] == "BAD_DEFERRAL" and p["t_id"] == "T-12" for p in out["problems"]
  )


def test_a_deferral_with_another_reason_than_the_matrix_is_rejected(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-59"]["defer_reason"] = "ran out of time"
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["manifest_valid"] is False and out["accepted"]["T-59"] == "REJECTED"


def test_a_deferral_without_a_start_condition_is_rejected(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-60"].pop("start_condition")
  assert w7.evaluate_manifest(matrix, vs, tmp_path, c)["manifest_valid"] is False


def test_a_deferred_row_cannot_be_reported_as_pass(matrix: Any, c: w7.Contracts, tmp_path: Path) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-59"] = {
    **verdict_for(matrix, tmp_path, "T-59", "PASS"),
    "evidence": [ev(tmp_path, "T-59", "CS1_LOG")],
  }
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["manifest_valid"] is False and out["accepted"]["T-59"] == "REJECTED"


def test_a_deferred_subitem_must_carry_the_matrix_reason(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-15"]["subitems"]["cs2_completion"]["defer_reason"] = "skipped"
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["accepted"]["T-15"] == "REJECTED" and out["manifest_valid"] is False


def test_a_deferred_subitem_start_condition_is_judged_on_its_own(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-15"]["subitems"]["cs2_completion"]["start_condition"] = "whenever"  # the reason still matches
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["accepted"]["T-15"] == "REJECTED" and out["manifest_valid"] is False
  vs2 = full_set(matrix, tmp_path)
  vs2["T-59"]["start_condition"] = "whenever"  # a whole-row deferral: reason matches, condition differs
  assert w7.evaluate_manifest(matrix, vs2, tmp_path, c)["accepted"]["T-59"] == "REJECTED"


def test_a_mandatory_subitem_that_is_not_pass_rejects_the_row(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-15"]["subitems"]["cs1_subset"] = {"status": "NOT_RUN"}
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["accepted"]["T-15"] == "REJECTED"
  assert any("subitem cs1_subset is not PASS" in p["detail"] for p in out["problems"])


def test_the_n1_part_cannot_be_skipped_by_deferring_everything(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-68"]["subitems"]["n1"] = {
    "status": "DEFERRED_NOT_APPLICABLE",
    "defer_reason": "x",
    "start_condition": "y",
  }
  assert w7.evaluate_manifest(matrix, vs, tmp_path, c)["accepted"]["T-68"] == "REJECTED"


def test_ps_is_outside_the_approved_run_scope_and_cannot_be_pass(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  assert vs["T-35"]["subitems"]["ps"] == {"status": "NOT_RUN"}
  assert w7.evaluate_manifest(matrix, vs, tmp_path, c)["accepted"]["T-35"] == "PASS"
  vs["T-35"]["subitems"]["ps"] = {"status": "PASS"}
  assert w7.evaluate_manifest(matrix, vs, tmp_path, c)["accepted"]["T-35"] == "REJECTED"


def test_p0_p1_pn_must_all_pass_for_t35(matrix: Any, c: w7.Contracts, tmp_path: Path) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-35"]["subitems"]["pn"] = {"status": "NOT_RUN"}
  assert w7.evaluate_manifest(matrix, vs, tmp_path, c)["accepted"]["T-35"] == "REJECTED"


def test_blocked_must_cite_a_missing_assumption_or_an_unfrozen_spec(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-29"] = {**verdict_for(matrix, tmp_path, "T-29", "BLOCKED")}
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["manifest_valid"] is False and any(p["code"] == "BAD_BLOCKED" for p in out["problems"])
  vs["T-29"]["blocked_by"] = {"kind": "unfrozen_spec", "ref": "C047 independent oracle"}
  out2 = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out2["accepted"]["T-29"] == "BLOCKED" and out2["gate_pass"] is False


def test_a_fail_must_carry_a_minimal_repro(matrix: Any, c: w7.Contracts, tmp_path: Path) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-12"] = verdict_for(matrix, tmp_path, "T-12", "FAIL")
  assert w7.evaluate_manifest(matrix, vs, tmp_path, c)["accepted"]["T-12"] == "REJECTED"
  vs["T-12"]["repro"] = {"window_id": "w1", "seed": 42, "inject_plan_ref": "ip0"}
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["accepted"]["T-12"] == "FAIL" and out["gate_pass"] is False


def test_a_deadline_does_not_turn_a_pending_row_into_a_pass(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-27"] = verdict_for(matrix, tmp_path, "T-27", "NOT_RUN")
  vs["T-27"]["notes"] = "deadline reached"
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["accepted"]["T-27"] == "NOT_RUN" and out["gate_pass"] is False


def test_rows_not_in_the_matrix_have_no_effect_and_each_t_is_judged_alone(
  matrix: Any, c: w7.Contracts, tmp_path: Path
) -> None:
  vs = full_set(matrix, tmp_path)
  vs["T-99"] = verdict_for(matrix, tmp_path, "T-12")
  out = w7.evaluate_manifest(matrix, vs, tmp_path, c)
  assert out["rows_checked"] == 68 and "T-99" not in out["accepted"]


# ---------------------------------------------------------------------------------------------------
# held contract rules
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
  ("which", "mutate"),
  [
    ("window", lambda d: d["header"]["window_kinds"].append("summary")),
    ("window", lambda d: d["header"]["required_fields"].remove("seal")),
    ("window", lambda d: d["time"]["cycle"].__setitem__("quantized", True)),
    ("window", lambda d: d["time"]["seq"].__setitem__("wraps", True)),
    ("window", lambda d: d["time"]["display_view"].__setitem__("is_sort_key", True)),
    ("window", lambda d: d["sequence_domains"].__setitem__("never_substitute", False)),
    ("window", lambda d: d["startup_epoch"].__setitem__("raw_log_applies_profile_init_discard", True)),
    ("window", lambda d: d["end_boundary"].__setitem__("declared_end_is_held_outside_the_window", False)),
    ("window", lambda d: d["seal"]["required_fields"].remove("record_count")),
    ("window", lambda d: d["seal"].__setitem__("missing_seal", "replayable")),
    ("window", lambda d: d["blob_manifest"].__setitem__("hash_only_is_not_a_blob", False)),
    ("window", lambda d: d["gap_status"].__setitem__("replayable_only_when", "quota")),
    ("window", lambda d: d["gap_status"]["quota"].__setitem__("never_unbounded", False)),
    ("window", lambda d: d["completeness"]["replayable_requires"].remove("end_boundary_reached")),
    ("window", lambda d: d["completeness"].__setitem__("prefix_still_contiguous_is_not_enough", False)),
    ("window", lambda d: d["single_side_replay"].__setitem__("both_sides", "promised")),
    ("window", lambda d: d["host_passive"].__setitem__("drives_dut_ports", True)),
    ("record", lambda d: d["fields"][0].__setitem__("required", False)),
    ("record", lambda d: d["uniqueness"]["key"].remove("phase")),
    ("record", lambda d: d["raw_log_forbidden_fields"].remove("ts_us")),
    ("record", lambda d: d["handshake"].__setitem__("accepted_must_equal_valid_and_ready", False)),
    ("record", lambda d: d["identity"].__setitem__("vme_instance_visible_to_software", True)),
    ("record", lambda d: d["reliability"]["R1"].__setitem__("lossless", False)),
    ("record", lambda d: d["reliability"]["R1"]["channels"].append("DEBUG")),
    ("record", lambda d: d["reliability"]["R2"].__setitem__("loss_changes_functional_verdict", True)),
    ("record", lambda d: d["data"].__setitem__("hash_only_record_replayable", True)),
    ("record", lambda d: d["normal_mode"].__setitem__("hbm_log_writes", 4)),
    ("verdict", lambda d: d["verdict"]["t_verdict_values"].append("MAYBE")),
    ("verdict", lambda d: d["verdict"].__setitem__("not_run_is_never_pass", False)),
    ("verdict", lambda d: d["verdict"].__setitem__("harness_only_is_cs1_evidence", True)),
    ("verdict", lambda d: d["checkers"].pop("C4")),
    ("verdict", lambda d: d["c4"].__setitem__("tolerance_allowed", True)),
    ("verdict", lambda d: d["c4"].__setitem__("ulp_relaxation", "allowed")),
    ("verdict", lambda d: d["c4"].__setitem__("upstream_spike_or_mpact_as_dut_pass", "allowed")),
    ("verdict", lambda d: d["evidence"].__setitem__("cs1_py_overrides_a_missing_cs1_log", True)),
    (
      "verdict",
      lambda d: d["evidence"]["cs1_py_alone_allowed_for"].append({"t_id": "T-01", "part": "all"}),
    ),
    ("verdict", lambda d: d["c7_gate"].__setitem__("deadline_counts_as_pass", True)),
    ("verdict", lambda d: d["c7_gate"].__setitem__("pending_counts_as_complete", True)),
    ("verdict", lambda d: d["c7_gate"]["deferred"]["allowed_only_for"].append("too_hard")),
    ("bundle", lambda d: d["replay_gate"].__setitem__("exit_code_when_any", 1)),
    ("bundle", lambda d: d["replay_gate"]["parts_that_make_it_not_replayable"].remove("truncated_end")),
    ("bundle", lambda d: d["replay_gate"].__setitem__("gap_is_a_numeric_fail", True)),
    ("bundle", lambda d: d["oracle_ref"].__setitem__("validator_as_numeric_oracle", "allowed")),
    ("bundle", lambda d: d["config"].__setitem__("seed_list_empty", "allow")),
    ("perf", lambda d: d["rows"].__setitem__("ps_is_not_a_deferral_of_a_mandatory_row", False)),
    ("perf", lambda d: d["rows"].__setitem__("pn_differs_from_p1_only_in", "rtl_hash")),
    ("perf", lambda d: d["metrics"]["required"].pop(3)),
    ("perf", lambda d: d["metrics"].__setitem__("needs_reverse_control", False)),
    ("perf", lambda d: d["timing"].__setitem__("debug_in_normal_performance", "allowed")),
    ("perf", lambda d: d["report_keys"]["required"].remove("arenas")),
    (
      "perf",
      lambda d: d["report_keys"].__setitem__("never_add_stall_categories_to_stall_breakdown", False),
    ),
    ("perf", lambda d: d["stall_reasons"]["names"].remove("WAIT_MSHR")),
    ("perf", lambda d: d["pre_registration"].__setitem__("order", "measure_then_register")),
    ("perf", lambda d: d["pre_registration"].__setitem__("zero_placeholder", "allowed")),
    ("perf", lambda d: d["gate_rejects"].remove("gap_window")),
    ("perf", lambda d: d["forbidden_claims"].remove("fmax")),
  ],
)
def test_contract_rules_are_held(docs: Any, which: str, mutate: Any) -> None:
  caught(docs, dm=lambda d: mutate(d[which]), needle=which)


@pytest.mark.parametrize(
  ("which", "path"),
  [("window", ("gap_status", "quota", "limit")), ("perf", ("pre_registration", "threshold_value"))],
)
def test_unsourced_quantities_cannot_be_filled_in(docs: Any, which: str, path: tuple) -> None:
  def fill(d: dict[str, Any]) -> None:
    node = d[which]
    for k in path:
      node = node[k]
    node["value"] = 5

  caught(docs, dm=fill, needle="unresolved")
  caught(docs, dm=lambda d: d[which].__setitem__(path[0], {}), needle="unresolved")


def test_an_invented_encoding_is_rejected_anywhere(docs: Any) -> None:
  caught(
    docs,
    dm=lambda d: d["window"]["time"]["cycle"].__setitem__("width_bits", 48),
    needle="invented encoding",
  )


# ---------------------------------------------------------------------------------------------------
# field request
# ---------------------------------------------------------------------------------------------------
def test_field_request_fields_must_exist_in_the_record_contract(docs: Any) -> None:
  caught(
    docs,
    rm=lambda r: r["log_fields"][0].__setitem__("name", "nonsense"),
    needle="not a field of log_record",
  )
  caught(docs, rm=lambda r: r["log_fields"].pop(0), needle="missing")


def test_field_request_invents_no_value_and_gives_reasons(docs: Any) -> None:
  caught(docs, rm=lambda r: r["log_fields"][0].__setitem__("opcode", 3), needle="INVENTED_VALUE")
  caught(docs, rm=lambda r: r["log_fields"][0].__setitem__("why", ""), needle="needs a reason")


def test_field_request_takes_the_phase_enum_from_the_catalog(docs: Any) -> None:
  caught(docs, rm=lambda r: r["phase_enum"].__setitem__("from", "w7"), needle="no second one")


def test_field_request_contract_refs_must_resolve(docs: Any) -> None:
  caught(
    docs, rm=lambda r: r["contract_refs"][0].__setitem__("key", "seal.nonexistent"), needle="does not exist"
  )
  caught(
    docs, rm=lambda r: r["contract_refs"][0].__setitem__("contract", "other"), needle="unknown contract"
  )


def test_field_request_names_both_w6_contracts(docs: Any) -> None:
  caught(docs, rm=lambda r: r["consumes"]["w6_contracts"].pop(), needle="both W6 contracts")
  caught(docs, rm=lambda r: r.__setitem__("package", "W6"), needle="W7")
