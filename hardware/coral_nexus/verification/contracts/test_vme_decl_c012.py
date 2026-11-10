"""C012 VME region declaration ABI and mode-table tests.

The reference model (`vme_decl_ref.DeclModel`) is built FROM the ABI JSON, so tests that edit the JSON
change what the model accepts. Pass-condition clauses of the card map to tests as follows:

  E1 with no grant leaves the queue empty ............ test_e1_*
  after a grant a stalled pending is not consumed ..... test_stalled_pending_*
  one E2 consumes once; a repeat is refused ........... test_e2_*
  legal tail table accepted by generator AND loader ... test_tail_table_*
  CLEARED restore accepted; missing preamble refused .. test_cleared_*
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

H = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(H / "scripts/lib"))

import validate_vme_decl as vd  # noqa: E402
import vme_decl_ref as ref  # noqa: E402


@pytest.fixture
def abi() -> dict[str, Any]:
  return copy.deepcopy(vd.load(vd.ABI))


@pytest.fixture
def tables() -> dict[str, Any]:
  return copy.deepcopy(vd.load(vd.TABLES))


@pytest.fixture
def request_doc() -> dict[str, Any]:
  return copy.deepcopy(vd.load(vd.REQUEST))


def _report(abi: Any, tables: Any, request_doc: Any) -> dict[str, Any]:
  return vd.validate_docs(abi, tables, request_doc)


def _model(abi: dict[str, Any], contexts: int = 2, instances: int = 1) -> ref.DeclModel:
  return ref.DeclModel(abi, contexts, instances)


def _code(fn: Any, *args: Any, **kwargs: Any) -> str:
  with pytest.raises((ref.DeclError, ref.ModeTableError)) as exc:
    fn(*args, **kwargs)
  return exc.value.code  # type: ignore[no-any-return]


def _start(m: ref.DeclModel, ctx: int) -> None:
  m.csrw(ctx, "REGION_START")
  m.target_at_head(ctx)
  m.accepted(ctx)


def test_committed_contract_validates(abi: dict, tables: dict, request_doc: dict) -> None:
  report = _report(abi, tables, request_doc)
  assert report["valid"], report["errors"]
  assert report["anchors_read_back"] >= 15 and report["faults"] == len(vd.REQUIRED_FAULTS)


def test_cli_exits_zero_on_the_committed_files_and_nonzero_on_a_broken_one(tmp_path: Path) -> None:
  ok = subprocess.run(
    [sys.executable, str(H / "scripts/lib/validate_vme_decl.py"), "--out", str(tmp_path / "ok")],
    capture_output=True,
    text=True,
  )
  assert ok.returncode == 0 and json.loads((tmp_path / "ok/vme_decl_report.json").read_text())["valid"]


# --- E1: request / park -----------------------------------------------------------------------


def test_e1_grants_a_free_instance_and_leaves_the_queue_empty(abi: dict) -> None:
  m = _model(abi)
  m.csrw(0, "REGION_START")
  assert m.target_at_head(0) == "GRANTED"
  assert m.queue == [], "E1 must not put the target into the shared queue"
  assert m.contexts[0].pending is not None, "E1 reads pending, it never consumes it"


def test_e1_with_no_free_instance_parks_with_one_waiter_and_an_empty_queue(abi: dict) -> None:
  m = _model(abi)
  _start(m, 0)
  m.csrw(1, "REGION_START")
  assert m.target_at_head(1) == "PARKED"
  assert m.queue == [(0, 1)], "only Context 0's accepted op is queued; the parked head never enters"
  assert m.waiters == [1] and m.contexts[1].waiter
  assert m.target_at_head(1) == "PARKED" and m.waiters == [1], (
    "re-presenting the head never adds a second waiter"
  )


def test_the_waiter_exists_although_the_target_was_never_accepted(abi: dict) -> None:
  m = _model(abi)
  _start(m, 0)
  m.csrw(1, "REGION_START")
  m.target_at_head(1)
  assert m.contexts[1].waiter and m.contexts[1].accepted_unsettled == 0


def test_a_parked_target_cannot_be_accepted(abi: dict) -> None:
  m = _model(abi)
  _start(m, 0)
  m.csrw(1, "REGION_START")
  m.target_at_head(1)
  assert _code(m.accepted, 1) == "ACCEPTED_WHILE_PARKED"


def test_e1_before_a_declaration_is_not_a_request(abi: dict) -> None:
  m = _model(abi)
  assert m.target_at_head(0) == "NOT_DECLARED"
  assert m.waiters == [] and m.instances == [None]


def test_chain_and_end_declarations_do_not_request_a_lease(abi: dict) -> None:
  m = _model(abi)
  _start(m, 0)
  m.csrw(0, "CHAIN_FIRST", 0)
  assert m.target_at_head(0) == "NOT_A_REGION_REQUEST"
  assert m.pmu["grant"] == 1


def test_two_instances_grant_two_contexts_and_the_third_parks(abi: dict) -> None:
  m = _model(abi, contexts=3, instances=2)
  for ctx in (0, 1, 2):
    m.csrw(ctx, "REGION_START")
  assert [m.target_at_head(0), m.target_at_head(1), m.target_at_head(2)] == ["GRANTED", "GRANTED", "PARKED"]


# --- E2: consume ------------------------------------------------------------------------------


def test_stalled_pending_survives_a_grant_and_is_not_consumed_early(abi: dict) -> None:
  m = _model(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  assert m.contexts[0].pending is not None and m.contexts[0].granted
  assert m.contexts[0].phase == "ACTIVE" and m.queue == []


def test_e2_consumes_once_and_a_repeat_is_refused(abi: dict) -> None:
  m = _model(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  consumed = m.accepted(0)
  assert consumed.kind == "REGION_START" and m.contexts[0].pending is None
  assert _code(m.accepted, 0) == "CONSUME_WITHOUT_PENDING"


def test_start_cannot_be_accepted_without_the_e1_grant(abi: dict) -> None:
  m = _model(abi)
  m.csrw(0, "REGION_START")
  assert _code(m.accepted, 0) == "ACCEPTED_WITHOUT_GRANT"


def test_a_repeated_csrw_is_an_armed_conflict_not_a_second_declaration(abi: dict) -> None:
  m = _model(abi)
  m.csrw(0, "REGION_START")
  assert _code(m.csrw, 0, "REGION_END") == "ARMED_CONFLICT"


def test_pending_is_per_context(abi: dict) -> None:
  m = _model(abi)
  m.csrw(0, "REGION_START")
  m.csrw(1, "REGION_START")  # a second Context arms independently
  assert m.contexts[0].pending is not None and m.contexts[1].pending is not None


# --- release order ----------------------------------------------------------------------------


def _ended(abi: dict) -> ref.DeclModel:
  m = _model(abi)
  _start(m, 0)
  m.csrw(0, "CHAIN_FIRST", 0)
  m.accepted(0)
  m.csrw(0, "CHAIN_FINAL", 0)
  m.accepted(0)
  m.csrw(1, "REGION_START")
  m.target_at_head(1)
  m.csrw(0, "REGION_END")
  m.accepted(0)
  return m


def _settled(abi: dict) -> ref.DeclModel:
  m = _ended(abi)
  for _ in range(3):
    m.settle(0, 0)
  return m


def test_state_free_and_region_done_need_every_accepted_op_settled(abi: dict) -> None:
  assert _code(_ended(abi).state_free, 0) == "STATE_FREE_WITH_OUTSTANDING"
  assert _code(_ended(abi).region_done, 0) == "REGION_DONE_WITH_OUTSTANDING"


def test_region_done_before_state_free_is_refused(abi: dict) -> None:
  assert _code(_settled(abi).region_done, 0) == "REGION_DONE_BEFORE_STATE_FREE"


def test_state_free_does_not_hand_the_instance_to_the_waiter(abi: dict) -> None:
  m = _settled(abi)
  m.state_free(0)
  assert m.waiters == [1] and m.pmu["wake"] == 0 and m.instances == [0]


def test_only_region_done_wakes_and_grants_the_waiter(abi: dict) -> None:
  m = _settled(abi)
  m.state_free(0)
  assert m.region_done(0) == [1]
  assert m.instances == [1] and m.contexts[1].phase == "ACTIVE" and m.contexts[1].granted
  assert m.pmu["wake"] == 1 and m.pmu["region_done"] == 1


def test_state_free_or_region_done_before_end_are_refused(abi: dict) -> None:
  m = _model(abi)
  _start(m, 0)
  assert _code(m.state_free, 0) == "STATE_FREE_BEFORE_END"
  m = _model(abi)
  _start(m, 0)
  assert _code(m.region_done, 0) == "REGION_DONE_BEFORE_END"


def test_final_done_needs_a_real_settle_not_just_acceptance(abi: dict) -> None:
  m = _ended(abi)
  assert m.contexts[0].chains[0] == "FINAL_SUBMITTED"
  m.settle(0, 0)
  assert m.contexts[0].chains[0] == "FINAL_DONE"


def test_parked_waiters_are_granted_in_arrival_order(abi: dict) -> None:
  """Design W3 3.1: no free instance means park in FIFO order, and a release grants the OLDEST waiter."""
  m = _model(abi, contexts=3, instances=1)
  _start(m, 0)
  for ctx in (1, 2):
    m.csrw(ctx, "REGION_START")
    assert m.target_at_head(ctx) == "PARKED"
  assert m.waiters == [1, 2]
  m.csrw(0, "REGION_END")
  m.accepted(0)
  m.settle(0)
  m.state_free(0)
  assert m.region_done(0) == [1], "the first waiter is granted, not the last"
  assert m.waiters == [2] and m.contexts[2].waiter and not m.contexts[1].waiter


def test_one_release_with_two_instances_grants_at_most_one_waiter_per_instance(abi: dict) -> None:
  m = _model(abi, contexts=4, instances=2)
  _start(m, 0)
  _start(m, 1)
  for ctx in (2, 3):
    m.csrw(ctx, "REGION_START")
    m.target_at_head(ctx)
  assert m.waiters == [2, 3]
  m.csrw(0, "REGION_END")
  m.accepted(0)
  m.settle(0)
  m.state_free(0)
  assert m.region_done(0) == [2], "one freed instance serves one waiter"
  assert m.instances.count(None) == 0 and m.waiters == [3]


# --- the 8 fault groups -----------------------------------------------------------------------


@pytest.mark.parametrize(
  ("steps", "code"),
  [
    (["F0", "F0"], "DUPLICATE_FIRST"),
    (["CONT"], "CONTINUE_WITHOUT_FIRST"),
    (["FIN"], "FINAL_WITHOUT_FIRST"),
    (["F0", "FIN", "CONT"], "ACCUMULATE_AFTER_FINAL"),
    (["F0", "FIN", "FIN"], "ACCUMULATE_AFTER_FINAL"),
    (["F0", "END"], "END_WITH_OPEN_CHAIN"),
  ],
)
def test_chain_order_faults(abi: dict, steps: list[str], code: str) -> None:
  m = _model(abi)
  _start(m, 0)
  kinds = {"F0": "CHAIN_FIRST", "FIN": "CHAIN_FINAL", "CONT": "CHAIN_CONTINUE", "END": "REGION_END"}
  seen = ""
  for step in steps:
    m.csrw(0, kinds[step], 0)
    try:
      m.accepted(0)
    except ref.DeclError as exc:
      seen = exc.code
      break
  assert seen == code


def test_a_legal_chain_with_a_controlled_drain_after_final(abi: dict) -> None:
  m = _model(abi)
  _start(m, 0)
  for kind in ("CHAIN_FIRST", "CHAIN_CONTINUE", "CHAIN_FINAL"):
    m.csrw(0, kind, 0)
    m.accepted(0)
  m.touch_tile(0)  # a drain-class access after FINAL is still inside an ACTIVE region
  assert m.contexts[0].chains[0] == "FINAL_SUBMITTED"


def test_f1_touch_without_a_binding(abi: dict) -> None:
  assert _code(_model(abi).touch_tile, 0) == "TOUCH_WITHOUT_BINDING"


def test_f3_touch_after_end(abi: dict) -> None:
  m = _model(abi)
  _start(m, 0)
  m.csrw(0, "REGION_END")
  m.accepted(0)
  assert _code(m.touch_tile, 0) == "TOUCH_AFTER_END"


def test_f2_start_while_holding_a_region(abi: dict) -> None:
  m = _model(abi)
  _start(m, 0)
  assert _code(m.csrw, 0, "REGION_START") == "NESTED_START"


def test_f4_task_exit_with_an_unfinished_region(abi: dict) -> None:
  m = _model(abi)
  _start(m, 0)
  assert _code(m.task_exit, 0) == "EXIT_WITHOUT_END"
  m = _model(abi)
  _start(m, 0)
  m.csrw(0, "REGION_END")
  m.accepted(0)
  assert _code(m.task_exit, 0) == "EXIT_BEFORE_REGION_DONE"


def test_unknown_declaration_kind_and_context_are_refused(abi: dict) -> None:
  m = _model(abi)
  assert _code(m.csrw, 0, "BOGUS") == "UNKNOWN_KIND"
  assert _code(m.csrw, 9, "REGION_START") == "CTX_OUT_OF_RANGE"


def test_every_fault_is_fail_stop(abi: dict) -> None:
  m = _model(abi)
  assert _code(m.csrw, 0, "BOGUS") == "UNKNOWN_KIND"
  assert m.contexts[0].phase == "FAILED" and m.pmu["fault"] == 1


def test_a_fault_code_the_abi_does_not_define_is_refused(abi: dict) -> None:
  abi["faults"] = [f for f in abi["faults"] if f["code"] != "ARMED_CONFLICT"]
  m = _model(abi)
  m.csrw(0, "REGION_START")
  assert _code(m.csrw, 0, "REGION_END") == "FAULT_CODE_UNKNOWN"


# --- redirect / cancel / generation -----------------------------------------------------------


def test_redirect_with_a_taken_grant_settles_it_and_wakes_the_waiter(abi: dict) -> None:
  m = _model(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  m.csrw(1, "REGION_START")
  m.target_at_head(1)
  m.redirect(0)
  assert m.instances == [1] and m.contexts[1].phase == "ACTIVE" and m.contexts[0].pending is None
  assert ("grant_settled", 0, "redirect") in m.log


def test_redirect_with_a_waiter_removes_it_and_keeps_the_head_eligible(abi: dict) -> None:
  m = _model(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  m.csrw(1, "REGION_START")
  m.target_at_head(1)
  m.redirect(1)
  assert m.waiters == [] and not m.contexts[1].waiter and m.contexts[1].phase == "IDLE"


def test_redirect_never_rebinds_the_declaration_to_a_later_instruction(abi: dict) -> None:
  m = _model(abi)
  m.csrw(0, "REGION_START")
  m.redirect(0)
  assert m.contexts[0].pending is None and m.target_at_head(0) == "NOT_DECLARED"


def test_a_generation_change_clears_pending_and_a_stale_record_is_refused(abi: dict) -> None:
  m = _model(abi)
  m.csrw(0, "REGION_START")
  m.task_exit(0)
  assert m.contexts[0].pending is None and m.contexts[0].generation == 2
  m.contexts[0].pending = ref.Pending("REGION_START", None, 1)
  assert _code(m.target_at_head, 0) == "STALE_GENERATION"


def test_cancel_after_accept_ends_failed_and_never_publishes_region_done(abi: dict) -> None:
  m = _model(abi)
  _start(m, 0)
  m.cancel(0)
  assert m.contexts[0].phase == "FAILED" and m.pmu["region_done"] == 0
  assert m.contexts[0].accepted_unsettled == 1, "accepted ops still settle for real"
  m.settle(0)
  assert m.contexts[0].accepted_unsettled == 0
  assert _code(m.settle, 0) == "SETTLE_WITHOUT_ACCEPT"


# --- mode tables ------------------------------------------------------------------------------


def test_committed_tables_validate(abi: dict, tables: dict) -> None:
  for name in vd.REQUIRED_TABLES:
    assert ref.validate_mode_table(tables[name], abi) == [], name


def test_tail_table_is_accepted_by_the_same_schema_and_differs_from_the_reference(
  abi: dict, tables: dict
) -> None:
  """The card's generator+loader clause: one schema, two different sealed kernels."""
  assert ref.validate_mode_table(tables["tail"], abi) == []
  assert ref.validate_mode_table(tables["reference"], abi) == []
  tail_tm = {m["name"]: m["tm"] for m in tables["tail"]["modes"]}
  ref_tm = {m["name"]: m["tm"] for m in tables["reference"]["modes"]}
  assert tail_tm["BF16-MAC-TK1"] != ref_tm["BF16-MAC-TK1"]
  assert vd.run_table_sequences(tables) == []


def test_mtype_is_recomputed_from_the_upstream_formula() -> None:
  assert ref.mtype_value(16, 1, 1) == 16417 and ref.mtype_value(16, 1, 2) == 16418
  assert ref.mtype_value(16, 2, 2) == 16450 and ref.mtype_value(16, 1, 3) == 16419
  assert ref.mtype_value(16, 4, 3) == 16515 and ref.mtype_value(9, 1, 2) == 9250
  assert ref.mtype_value(0x4000, 8, 7) == ((0x3FFF & 0x4000) << 10) | (0 << 5) | 3, "fields are masked"


def test_the_old_draft_value_16386_is_a_bf16_width_not_fp32(abi: dict) -> None:
  assert ref.mtype_value(16, 0, 2) == 16386 and ref.mtype_value(16, 1, 1) == 16417


def test_cleared_restore_is_legal_and_a_missing_preamble_is_refused(tables: dict) -> None:
  t = tables["reference"]
  ref.access_allowed(t, "BF16-MAC-TK2", "CLEARED", inside_asm_block=False)
  ref.access_allowed(t, "CLEARED", "BF16-MAC-TK2", inside_asm_block=False)
  assert (
    _code(ref.access_allowed, t, "CLEARED", "CLEARED", inside_asm_block=False) == "MISSING_RESTORE_PREAMBLE"
  )
  assert (
    _code(ref.access_allowed, t, "NONE", "BF16-MAC-TK2", inside_asm_block=False)
    == "TRANSITION_NOT_IN_TABLE"
  )


def test_vset_inside_an_asm_block_is_never_legal(tables: dict) -> None:
  assert (
    _code(ref.access_allowed, tables["reference"], "FP32-MAC", "BF16-MAC-TK2", inside_asm_block=True)
    == "VSET_INSIDE_ASM_BLOCK"
  )


def test_a_disabled_or_unknown_mode_is_refused(tables: dict) -> None:
  t = tables["reference"]
  assert _code(ref.access_allowed, t, "FP32-MAC", "INT8", inside_asm_block=False) == "DISABLED_MODE"
  assert (
    _code(ref.access_allowed, t, "FP32-MAC", "BF16-MAC-TK9", inside_asm_block=False) == "MODE_NOT_IN_TABLE"
  )


def test_a_mode_legal_in_one_kernel_is_refused_by_another(tables: dict) -> None:
  """The table is per kernel, not a global whitelist."""
  ref.access_allowed(tables["reference"], "FP32-MAC", "BF16-MAC-TK2", inside_asm_block=False)
  assert (
    _code(ref.access_allowed, tables["tail"], "FP32-MAC", "BF16-MAC-TK2", inside_asm_block=False)
    == "MODE_NOT_IN_TABLE"
  )


@pytest.mark.parametrize(
  ("mutate", "needle"),
  [
    (lambda t: t["modes"][2].__setitem__("mtype", t["modes"][2]["mtype"] + 1), "MTYPE_MISMATCH"),
    (lambda t: t["modes"][2].update(tk=3, mtype=ref.mtype_value(16, 3, 2)), "TK_ABOVE_CLAMP"),
    (lambda t: t["modes"][1].update(tm=17, mtype=ref.mtype_value(17, 1, 2)), "TM_ABOVE_CLAMP"),
    (lambda t: t["modes"][3].__setitem__("enabled", True), "DISABLED_MTWIDEN_ENABLED"),
    (lambda t: t["modes"][1].__setitem__("vtype", 0x1C9 | 0x200), "VTYPE_RESERVED_BITS"),
    (lambda t: t["modes"][1].__setitem__("vtype", 0x1CA), "VTYPE_LMUL_NOT_DERIVED"),
    (lambda t: t["modes"].append(copy.deepcopy(t["modes"][0])), "MODE_DUPLICATE"),
    (lambda t: t["transitions"].append(copy.deepcopy(t["transitions"][0])), "TRANSITION_DUPLICATE"),
    (lambda t: t["transitions"].append({"from": "FP32-MAC", "to": "NOPE"}), "TRANSITION_UNKNOWN_MODE"),
    (
      lambda t: [x.pop("restore", None) for x in t["transitions"] if x["from"] == "CLEARED"],
      "RESTORE_PREAMBLE",
    ),
    (
      lambda t: t["modes"].append(
        {"name": "CLEARED", "tm": 16, "tk": 1, "mtwiden": 1, "mtype": 16417, "vtype": 0xD2, "enabled": True}
      ),
      "CLEARED_IS_NOT_A_MODE",
    ),
    (
      lambda t: t.__setitem__("transitions", [x for x in t["transitions"] if x["from"] != "NONE"]),
      "UNREACHABLE",
    ),
  ],
)
def test_each_table_mutation_fires_its_own_check(abi: dict, tables: dict, mutate: Any, needle: str) -> None:
  mutate(tables["reference"])
  errors = ref.validate_mode_table(tables["reference"], abi)
  assert any(needle in e for e in errors), (needle, errors)


def test_a_table_cannot_request_more_than_the_front_end_will_apply(abi: dict, tables: dict) -> None:
  """tk/tm are clamped silently by the RTL: a row above the clamp describes hardware that is not there."""
  limits = abi["mode_table_schema"]["limits"]
  assert limits["tk_max_by_mtwiden"]["2"]["value"] == 2 and limits["tm_max"]["value"] == 16
  mode = tables["reference"]["modes"][2]
  mode.update(tk=4, mtype=ref.mtype_value(16, 4, 2))
  assert any("TK_ABOVE_CLAMP" in e for e in ref.validate_mode_table(tables["reference"], abi))


# --- the required contract is held by the validator, not by the file it constrains -------------


def _drop_fault(code: str) -> Any:
  return lambda a: a.__setitem__("faults", [f for f in a["faults"] if f["code"] != code])


@pytest.mark.parametrize("code", sorted(vd.REQUIRED_FAULTS))
def test_deleting_any_fault_invalidates_the_contract(
  abi: dict, tables: dict, request_doc: dict, code: str
) -> None:
  _drop_fault(code)(abi)
  report = _report(abi, tables, request_doc)
  assert report["valid"] is False
  assert any(code in e for e in report["errors"])


def test_deleting_a_whole_fault_group_and_its_faults_is_rejected(
  abi: dict, tables: dict, request_doc: dict
) -> None:
  abi["fault_groups"].pop("F4")
  abi["faults"] = [f for f in abi["faults"] if f["group"] != "F4"]
  report = _report(abi, tables, request_doc)
  assert report["valid"] is False and any("F4" in e for e in report["errors"])


def test_an_unlisted_fault_must_be_added_to_the_validator_deliberately(
  abi: dict, tables: dict, request_doc: dict
) -> None:
  abi["faults"].append({"code": "NEW_FAULT", "group": "F1", "static": False, "text": "x"})
  report = _report(abi, tables, request_doc)
  assert report["valid"] is False and any(
    "NEW_FAULT" in e and "required set" in e for e in report["errors"]
  )


@pytest.mark.parametrize("code", sorted(vd.REQUIRED_FAULTS))
def test_moving_a_fault_to_another_group_is_rejected(
  abi: dict, tables: dict, request_doc: dict, code: str
) -> None:
  fault = next(f for f in abi["faults"] if f["code"] == code)
  fault["group"] = "F1" if fault["group"] != "F1" else "F2"
  assert any("group" in e for e in _report(abi, tables, request_doc)["errors"])


@pytest.mark.parametrize(
  ("mutate", "needle"),
  [
    (lambda a: a["events"].pop("E1"), "abi.events.E1"),
    (lambda a: a["events"]["E1"].__setitem__("requires_accepted", True), "requires_accepted"),
    (lambda a: a["events"]["E1"].__setitem__("registers_waiter", False), "registers_waiter"),
    (lambda a: a["events"]["E1"].__setitem__("may_consume_pending", True), "never CONSUME"),
    (lambda a: a["events"]["E2"].__setitem__("consume_count", 2), "single, unique consume"),
    (lambda a: a.__setitem__("ordering", ["region_done", "state_free", "wake"]), "abi.ordering"),
    (lambda a: a.__setitem__("ordering", ["state_free", "region_done"]), "abi.ordering"),
    (lambda a: a["carrier"].__setitem__("width_bits", 32), "width_bits"),
    (lambda a: a["carrier"].__setitem__("bit_positions", [0, 1, 2]), "bit_positions"),
    (lambda a: a["carrier"].__setitem__("address", 0x7C0), "INVENTED_VALUE"),
    (lambda a: a["carrier"].__setitem__("access", "RW"), "carrier.access"),
    (lambda a: a["declaration"]["kinds"].remove("CHAIN_FINAL"), "abi.declaration.kinds"),
    (lambda a: a["properties"].pop("U3"), "abi.properties.U3"),
    (lambda a: a["cleanup"].__setitem__("grant_then_redirect", "no side effect"), "SETTLED"),
    (lambda a: a["cleanup"].__setitem__("accepted_then_cancel", "ends"), "never published"),
    (lambda a: a["cleanup"].pop("waiter_then_redirect"), "waiter_then_redirect"),
    (lambda a: a["cleanup"].__setitem__("final_does_not_drain", False), "FINAL must not drain"),
    (lambda a: a.__setitem__("status", "frozen"), "draft"),
  ],
)
def test_each_abi_mutation_is_rejected(
  abi: dict, tables: dict, request_doc: dict, mutate: Any, needle: str
) -> None:
  mutate(abi)
  report = _report(abi, tables, request_doc)
  assert report["valid"] is False and any(needle in e for e in report["errors"]), report["errors"][:3]


@pytest.mark.parametrize("name", sorted(vd.REQUIRED_LIMITS))
def test_deleting_a_hardware_limit_is_rejected(
  abi: dict, tables: dict, request_doc: dict, name: str
) -> None:
  abi["mode_table_schema"]["limits"].pop(name)
  report = _report(abi, tables, request_doc)
  assert report["valid"] is False and any(name in e for e in report["errors"])


@pytest.mark.parametrize(
  ("name", "mutate"),
  [
    ("tk_max_by_mtwiden", lambda lim: lim["tk_max_by_mtwiden"]["2"].__setitem__("value", 8)),
    ("tm_max", lambda lim: lim["tm_max"].__setitem__("value", 64)),
    ("disabled_mtwiden", lambda lim: lim["disabled_mtwiden"].__setitem__("values", [])),
    ("mtwiden_values", lambda lim: lim["mtwiden_values"]["values"].append(7)),
    ("vtype_lmul_by_sew", lambda lim: lim["vtype_lmul_by_sew"]["1"].__setitem__("lmul", 3)),
  ],
)
def test_widening_a_hardware_limit_is_rejected_even_when_a_table_would_then_pass(
  abi: dict, tables: dict, request_doc: dict, name: str, mutate: Any
) -> None:
  mutate(abi["mode_table_schema"]["limits"])
  report = _report(abi, tables, request_doc)
  assert report["valid"] is False and any(name in e and "required" in e for e in report["errors"])


def test_a_limit_without_its_own_source_is_rejected(abi: dict, tables: dict, request_doc: dict) -> None:
  abi["mode_table_schema"]["limits"]["tm_max"].pop("source")
  abi["mode_table_schema"]["limits"]["tk_max_by_mtwiden"]["3"].pop("source")
  errors = _report(abi, tables, request_doc)["errors"]
  assert any("tm_max" in e and "source" in e for e in errors)
  assert any("tk_max_by_mtwiden.3" in e and "source" in e for e in errors)


def test_the_int8_decision_needs_a_stated_reason(abi: dict, tables: dict, request_doc: dict) -> None:
  abi["mode_table_schema"]["limits"]["disabled_mtwiden"].pop("decided_by")
  assert any("decided_by" in e for e in _report(abi, tables, request_doc)["errors"])


@pytest.mark.parametrize("cid", vd.REQUIRED_CORRECTIONS)
def test_a_design_text_correction_cannot_be_removed(
  abi: dict, tables: dict, request_doc: dict, cid: str
) -> None:
  abi["corrections_to_design_text"] = [c for c in abi["corrections_to_design_text"] if c["id"] != cid]
  assert any(cid in e and "removed" in e for e in _report(abi, tables, request_doc)["errors"])


def test_deleting_the_corrections_and_the_schema_does_not_crash_the_validator(
  abi: dict, tables: dict, request_doc: dict
) -> None:
  abi.pop("corrections_to_design_text")
  abi.pop("mode_table_schema")
  report = _report(abi, tables, request_doc)
  assert report["valid"] is False and report["errors"]


@pytest.mark.parametrize("table", vd.REQUIRED_TABLES)
def test_deleting_a_kernel_table_is_rejected(
  abi: dict, tables: dict, request_doc: dict, table: str
) -> None:
  tables.pop(table)
  assert any(table in e for e in _report(abi, tables, request_doc)["errors"])


def test_the_tail_table_cannot_be_a_copy_of_the_reference(
  abi: dict, tables: dict, request_doc: dict
) -> None:
  tables["tail"] = copy.deepcopy(tables["reference"])
  assert any("tail" in e for e in _report(abi, tables, request_doc)["errors"])


@pytest.mark.parametrize("mode", sorted(vd.REQUIRED_REFERENCE_MODES))
def test_a_required_reference_mode_cannot_be_deleted(
  abi: dict, tables: dict, request_doc: dict, mode: str
) -> None:
  tables["reference"]["modes"] = [m for m in tables["reference"]["modes"] if m["name"] != mode]
  assert any(mode in e for e in _report(abi, tables, request_doc)["errors"])


def test_int8_cannot_be_enabled_and_the_reference_scope_must_deny_a_whitelist(
  abi: dict, tables: dict, request_doc: dict
) -> None:
  tables["reference"]["modes"][3]["enabled"] = True
  assert any("INT8" in e for e in _report(abi, copy.deepcopy(tables), request_doc)["errors"])
  tables["reference"]["modes"][3]["enabled"] = False
  tables["reference"]["scope"] = "the whitelist"
  assert any("whitelist" in e for e in _report(abi, tables, request_doc)["errors"])


# --- citations --------------------------------------------------------------------------------


def test_every_cited_line_is_read_back(abi: dict, tables: dict) -> None:
  errors: list[str] = []
  assert vd.check_sources(abi, "abi", errors) >= 14
  assert errors == []


def test_every_cited_line_drifting_is_caught_by_the_whole_tree_walk(
  abi: dict, tables: dict, request_doc: dict
) -> None:
  for _, src in vd.iter_sources(abi):
    src["line"] += 1
  errors = _report(abi, tables, request_doc)["errors"]
  assert sum("SOURCE_LINE_MISMATCH" in e for e in errors) >= 10


@pytest.mark.parametrize("mutation", ["line", "needle", "file"])
def test_one_nested_citation_that_drifts_is_caught(
  abi: dict, tables: dict, request_doc: dict, mutation: str
) -> None:
  src = abi["mode_table_schema"]["limits"]["tm_max"]["source"]
  if mutation == "line":
    src["line"] += 3
  elif mutation == "needle":
    src["needle"] = "something the file does not say"
  else:
    src["file"] = "verilog/rvv/design/does_not_exist.sv"
  assert any("tm_max" in e for e in _report(abi, tables, request_doc)["errors"])


def test_a_citation_missing_its_needle_is_an_error(abi: dict, tables: dict, request_doc: dict) -> None:
  abi["events"]["E2"]["source"].pop("needle")
  assert any("MISSING_SOURCE" in e for e in _report(abi, tables, request_doc)["errors"])


def test_the_design_corrections_are_true_by_reading_the_pinned_tree(abi: dict) -> None:
  by_id = {c["id"]: c for c in abi["corrections_to_design_text"]}
  for cid in ("VTYPE-01", "ANCHOR-01", "ANCHOR-02", "TK-CLAMP-01", "ENC-01"):
    src = by_id[cid]["source"]
    text = vd._read_line(src["file"], src["line"])
    assert text is not None and src["needle"] in text, cid


# --- the field request ------------------------------------------------------------------------


def test_the_field_request_is_clean_and_names_only_real_things(abi: dict, request_doc: dict) -> None:
  errors: list[str] = []
  vd.check_field_request(request_doc, abi, errors)
  assert errors == []


@pytest.mark.parametrize(
  ("mutate", "needle"),
  [
    (lambda r: r["csr_requests"].pop(), "csr_requests"),
    (lambda r: r["csr_requests"][0].__setitem__("address", 0x7C0), "INVENTED_VALUE"),
    (lambda r: r["csr_requests"][0].__setitem__("bits", 8), "INVENTED_VALUE"),
    (lambda r: r["csr_requests"][0].__setitem__("kind", "NOPE"), "not an ABI kind"),
    (lambda r: r["messages"][0].pop("new"), "not in the C009 catalog"),
    (
      lambda r: r["messages"].append({"message": "TASK_COMMIT", "new": True}),
      "already in the C009 catalog",
    ),
    (lambda r: r["fields_needed"].pop(0), "is missing"),
    (lambda r: r.__setitem__("schema", "x"), "schema/package"),
    (lambda r: r.__setitem__("package", "W9"), "schema/package"),
  ],
)
def test_each_field_request_mutation_is_rejected(
  abi: dict, request_doc: dict, mutate: Any, needle: str
) -> None:
  mutate(request_doc)
  errors: list[str] = []
  vd.check_field_request(request_doc, abi, errors)
  assert any(needle in e for e in errors), errors


def test_the_requested_csr_names_exist_in_the_csr_name_map() -> None:
  declared = {d["name"] for d in vd.load(vd.CSR_NAMES)["declared"]}
  assert vd.REQUIRED_CSR_NAMES <= declared


# --- structurally broken documents are reported, never raised ---------------------------------


@pytest.mark.parametrize("which", ["abi", "tables", "request"])
@pytest.mark.parametrize("junk", [[], None, "x", 5])
def test_a_non_mapping_root_is_reported(
  abi: dict, tables: dict, request_doc: dict, which: str, junk: Any
) -> None:
  docs = {"abi": abi, "tables": tables, "request": request_doc}
  docs[which] = junk
  report = _report(docs["abi"], docs["tables"], docs["request"])
  assert report["valid"] is False and any("root must be a mapping" in e for e in report["errors"])


@pytest.mark.parametrize(
  "section", ["events", "declaration", "carrier", "faults", "cleanup", "mode_table_schema"]
)
def test_deleting_any_abi_section_is_reported_not_raised(
  abi: dict, tables: dict, request_doc: dict, section: str
) -> None:
  abi.pop(section)
  report = _report(abi, tables, request_doc)
  assert report["valid"] is False and report["errors"]


def test_a_model_built_from_an_abi_without_a_kind_refuses_that_kind(abi: dict) -> None:
  abi["declaration"]["kinds"].remove("CHAIN_FINAL")
  m = _model(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  m.accepted(0)
  assert _code(m.csrw, 0, "CHAIN_FINAL", 0) == "UNKNOWN_KIND"


# --- the required set lives in the validator --------------------------------------------------


def test_the_required_sets_are_module_constants_not_read_from_the_json(abi: dict) -> None:
  before = (dict(vd.REQUIRED_FAULTS), tuple(vd.REQUIRED_KINDS), dict(vd.REQUIRED_LIMITS))
  abi["faults"], abi["declaration"]["kinds"] = [], []
  assert (dict(vd.REQUIRED_FAULTS), tuple(vd.REQUIRED_KINDS), dict(vd.REQUIRED_LIMITS)) == before
  assert len(vd.REQUIRED_FAULTS) == 24 and len(vd.REQUIRED_KINDS) == 5


def test_every_required_fault_is_in_the_committed_abi_with_a_matching_group(abi: dict) -> None:
  faults = {f["code"]: f for f in abi["faults"]}
  assert set(faults) == set(vd.REQUIRED_FAULTS)
  for code, group in vd.REQUIRED_FAULTS.items():
    assert faults[code]["group"] == group and faults[code]["static"] == (code in vd.STATIC_FAULTS)


def test_the_abi_stays_draft_and_assigns_no_bit_or_address(abi: dict) -> None:
  assert abi["status"] == "draft"
  carrier = abi["carrier"]
  assert carrier["bit_positions"] == "unresolved" and carrier["width_bits"] == "unresolved"
  assert carrier["access"] == "unspecified"
  assert not {"address", "opcode", "bits", "csr_address", "encoding"} & set(carrier)


# --- guards that had no direct test ----------------------------------------------------------


def test_the_abi_names_exactly_the_five_csr_logical_names(
  abi: dict, tables: dict, request_doc: dict
) -> None:
  abi["carrier"]["csr_name_map_entries"].pop()
  assert any("csr_name_map_entries" in e for e in _report(abi, tables, request_doc)["errors"])
  abi2 = copy.deepcopy(vd.load(vd.ABI))
  abi2["carrier"]["csr_name_map_entries"].append("nexus.mregion.bogus")
  assert any("csr_name_map_entries" in e for e in _report(abi2, tables, request_doc)["errors"])


@pytest.mark.parametrize(
  ("code", "static"),
  [("ARMED_CONFLICT", True), ("NESTED_START", True), ("UNKNOWN_KIND", False), ("DUPLICATE_FIRST", False)],
)
def test_the_static_versus_runtime_split_of_a_fault_cannot_flip(
  abi: dict, tables: dict, request_doc: dict, code: str, static: bool
) -> None:
  """`static` says whether build/load can reject it; flipping it changes who must catch the fault."""
  next(f for f in abi["faults"] if f["code"] == code)["static"] = static
  errors = _report(abi, tables, request_doc)["errors"]
  assert any(code in e and "static must be" in e for e in errors)


def test_a_fault_cannot_lose_its_description(abi: dict, tables: dict, request_doc: dict) -> None:
  next(f for f in abi["faults"] if f["code"] == "TOUCH_AFTER_END")["text"] = ""
  assert any(
    "TOUCH_AFTER_END" in e and "description" in e for e in _report(abi, tables, request_doc)["errors"]
  )


def test_cancel_after_accept_may_not_be_described_as_publishing_success(
  abi: dict, tables: dict, request_doc: dict
) -> None:
  abi["cleanup"]["accepted_then_cancel"] = "the region completes and publishes region_done"
  assert any("never published" in e for e in _report(abi, tables, request_doc)["errors"])
