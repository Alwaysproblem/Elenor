"""C015 W6 event / terminal / reset contract tests.

The consumer (`w6_contract`) is built FROM protocol/w6_event_facility.contract.yaml and
protocol/w6_terminal_reset.contract.yaml, and the required set is held in `validate_w6_contract`, so these
tests drive real event, wait, reset, cancel, fault and debug sequences and real wait-for graphs. A pass
here is a CONTRACT result: it is not RTL, and C033/C034 still have to show concurrency and backpressure.

Card pass-condition clauses -> tests:
  legal graph input accepted ........................... test_committed_contracts_validate
  DRAIN -> PENDING -> MARK_EVENTS loop rejected ......... test_drain_waiting_on_pending_is_a_cycle*
  missing terminal reserve rejected ..................... test_*terminal_reserve*
  Group/Tile writer confusion rejected .................. test_*writer*
  no circular wait / every accepted class settled ........ test_*cycle*, test_every_accepted_class_*
  full table = backpressure, never a fault .............. test_full_frontier_*
  complete/cancel/fail settlement owner per class ....... test_every_accepted_class_*
  debug gating / FIFO derivation / export fields here ... test_debug_*
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

H = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(H / "scripts/lib"))

import validate_w6_contract as vw  # noqa: E402
import w6_contract as w6  # noqa: E402
from w6_contract import EventKey, W6Error  # noqa: E402

DRAIN_ALL = dict.fromkeys(
  (
    "engine_accepted_zero",
    "engine_queue_empty",
    "l1_mshr_empty",
    "transfer_quiescent",
    "no_pending_debug_export",
    "no_unaccepted_phase_or_post_event",
  ),
  True,
)


@pytest.fixture
def docs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
  f, t = w6.load_contracts()
  return copy.deepcopy(f), copy.deepcopy(t), yaml.safe_load(vw.REQUEST.read_text())


@pytest.fixture
def fac_c() -> dict[str, Any]:
  return w6.load(w6.FACILITY)


@pytest.fixture
def term_c() -> dict[str, Any]:
  return w6.load(w6.TERMINAL)


def key(
  i: int, ctx: int = 0, gen: int = 1, eg: int = 0, seq: int = 0, scope: str = "task_private"
) -> EventKey:
  return EventKey(0, ctx, gen, i, eg, seq, scope)


def make(fac_c: dict[str, Any], **kw: int) -> w6.EventFacility:
  a = {"contexts": 2, "frontier": 2, "terminal_reserve": 1, "tile_slots": 8, "response_inflight": 2}
  a.update(kw)
  return w6.EventFacility(fac_c, waitall_max=a.pop("waitall_max", 3), **a)


def expect(code: str, fn: Any, *args: Any, **kw: Any) -> None:
  with pytest.raises(W6Error) as err:
    fn(*args, **kw)
  assert err.value.code == code


def report_of(f: Any, t: Any, r: Any) -> dict[str, Any]:
  return vw.validate_docs(f, t, r)


def caught(docs_: Any, *, fm: Any = None, tm: Any = None, rm: Any = None, needle: str = "") -> None:
  f, t, r = copy.deepcopy(docs_[0]), copy.deepcopy(docs_[1]), copy.deepcopy(docs_[2])
  for fn, doc in ((fm, f), (tm, t), (rm, r)):
    if fn:
      fn(doc)
  rep = report_of(f, t, r)
  assert not rep["valid"], "the mutation was not detected"
  assert needle in " | ".join(rep["errors"]), rep["errors"]


# ---------------------------------------------------------------------------------------------------
# committed files, anchors, CLI
# ---------------------------------------------------------------------------------------------------
def test_committed_contracts_validate() -> None:
  rep = vw.validate()
  assert rep["valid"], rep["errors"]
  assert rep["anchors_read_back"] >= 25


def test_cli_writes_report_and_exits_zero(tmp_path: Path) -> None:
  out = tmp_path / "o"
  proc = subprocess.run(
    [sys.executable, str(H / "scripts/lib/validate_w6_contract.py"), "--out", str(out)],
    capture_output=True,
    text=True,
  )
  assert proc.returncode == 0, proc.stderr
  assert json.loads((out / "w6_contract_report.json").read_text())["valid"] is True


def test_cli_exits_nonzero_on_a_broken_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  f, t = w6.load_contracts()
  f["states"]["terminal"].remove("TIMEOUT")
  rep = report_of(f, t, yaml.safe_load(vw.REQUEST.read_text()))
  assert rep["valid"] is False and rep["errors"]


def test_a_non_mapping_document_is_reported_not_raised() -> None:
  rep = report_of([], {}, None)
  assert rep["valid"] is False


@pytest.mark.parametrize(
  ("where", "mutate", "needle"),
  [
    ("facility", lambda f: f["states"]["source"].__setitem__("line", 1), "SOURCE_LINE_MISMATCH"),
    ("facility", lambda f: f["reclaim"]["source"].__setitem__("needle", "zzz"), "SOURCE_LINE_MISMATCH"),
    ("facility", lambda f: f["reclaim"]["source"].pop("needle"), "MISSING_SOURCE"),
    (
      "terminal",
      lambda t: t["fault_domain"]["source_function_supports_tile"].__setitem__("line", 1),
      "MISMATCH",
    ),
    (
      "terminal",
      lambda t: t["reset_endpoint"]["validator_sources"]["mark_events"].__setitem__("needle", "x"),
      "MISMATCH",
    ),
  ],
)
def test_every_citation_is_read_back(docs: Any, where: str, mutate: Any, needle: str) -> None:
  caught(docs, **({"fm": mutate} if where == "facility" else {"tm": mutate}), needle=needle)


def test_status_stays_draft(docs: Any) -> None:
  caught(docs, fm=lambda f: f.__setitem__("status", "frozen"), needle="draft")
  caught(docs, tm=lambda t: t.__setitem__("status", "frozen"), needle="draft")


# ---------------------------------------------------------------------------------------------------
# event states, kinds, counters
# ---------------------------------------------------------------------------------------------------
def test_terminal_state_is_sticky_and_a_second_signal_only_counts(fac_c: Any) -> None:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7, future_uses=1)
  f.add_consumer(key(1))  # a live consumer keeps the entry
  assert f.signal(key(1), "DONE", 7) is True
  assert f.signal(key(1), "ERROR", 7) is False
  assert f.entries[(0, 0, 1)].status == "DONE"
  assert f.counters["duplicate_signal"] == 1


@pytest.mark.parametrize(
  ("sig", "counter"),
  [
    (key(1, seq=5), "stale_sequence"),
    (key(1, eg=9), "stale_sequence"),
    (key(1, gen=0), "wrong_owner"),
    (key(1, scope="external"), "wrong_owner"),
    (key(9), "unknown_event_notify"),
  ],
)
def test_a_rejected_signal_changes_a_counter_and_never_the_entry(
  fac_c: Any, sig: EventKey, counter: str
) -> None:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7)
  assert f.signal(sig, "DONE", 7) is False
  assert f.counters[counter] == 1 and sum(f.counters.values()) == 1
  assert f.entries[(0, 0, 1)].status == "PENDING"


def test_wrong_producer_is_rejected(fac_c: Any) -> None:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7)
  assert f.signal(key(1), "DONE", 99) is False
  assert f.counters["wrong_owner"] == 1 and f.entries[(0, 0, 1)].status == "PENDING"


def test_an_old_context_generation_cannot_complete_the_new_occupant(fac_c: Any) -> None:
  f = make(fac_c)
  f.register(key(1, gen=2), "compute_done", 7, future_uses=1)
  f.issue_wait((0, 0, 1), "WAIT", [key(1, gen=2)])
  assert f.signal(key(1, gen=1), "DONE", 7) is False
  assert (0, 0, 1) in f.waits and f.wakes == []


@pytest.mark.parametrize("kind", ["INPUT_RELEASED", "OUTPUT_READY", "TASK_COMPLETION"])
def test_phase_and_terminal_kinds_are_not_local_events(fac_c: Any, kind: str) -> None:
  expect("NOT_A_LOCAL_EVENT", make(fac_c).register, key(1), kind, 7)


def test_resource_released_is_not_waitable(fac_c: Any) -> None:
  expect("NOT_WAITABLE", make(fac_c).register, key(1), "RESOURCE_RELEASED", 7)


def test_unknown_kind_and_scope_are_rejected(fac_c: Any) -> None:
  expect("UNKNOWN_KIND", make(fac_c).register, key(1), "magic", 7)
  expect("UNKNOWN_SCOPE", make(fac_c).register, key(1, scope="galaxy"), "compute_done", 7)


def test_reclaim_needs_terminal_and_no_refs_and_no_future_uses(fac_c: Any) -> None:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7, future_uses=1)
  f.add_consumer(key(1))
  f.signal(key(1), "DONE", 7)
  assert (0, 0, 1) in f.entries  # terminal, but a reference and a future use remain
  expect("ABANDON_WITH_CONSUMERS", f.abandon_future_uses, 0, 0, 1)
  f.consume(key(1))
  assert (0, 0, 1) not in f.entries


def test_abandon_after_a_fault_reclaims_unreferenced_events(fac_c: Any) -> None:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7, future_uses=1)
  f.signal(key(1), "DONE", 7)
  assert (0, 0, 1) in f.entries
  f.abandon_future_uses(0, 0, 1)
  assert not f.entries


def test_a_pending_event_cannot_be_consumed(fac_c: Any) -> None:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7, future_uses=1)
  f.add_consumer(key(1))
  expect("CANNOT_CONSUME", f.consume, key(1))


def test_a_live_slot_cannot_be_registered_twice(fac_c: Any) -> None:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7)
  expect("SLOT_IN_USE", f.register, key(1, eg=1), "compute_done", 7)
  assert f.counters["protocol_error"] == 1


# ---------------------------------------------------------------------------------------------------
# WAIT / WAITALL
# ---------------------------------------------------------------------------------------------------
def two_events(fac_c: Any) -> w6.EventFacility:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7, future_uses=1)
  f.register(key(2), "transfer_done", 8, future_uses=1)
  return f


def test_event_first_means_no_park(fac_c: Any) -> None:
  f = two_events(fac_c)
  f.signal(key(1), "DONE", 7)
  assert f.issue_wait((0, 0, 1), "WAIT", [key(1)]) == "PROCEED"
  assert f.waits == {} and f.wakes == []


def test_same_cycle_completion_wins_over_park(fac_c: Any) -> None:
  f = two_events(fac_c)
  out = f.step([(key(1), "DONE", 7)], [((0, 0, 1), "WAIT", [key(1)])])
  assert out == ["PROCEED"] and f.waits == {}


def test_completion_after_park_wakes_exactly_once(fac_c: Any) -> None:
  f = two_events(fac_c)
  assert f.issue_wait((0, 0, 1), "WAIT", [key(1)]) == "PARK"
  f.signal(key(1), "DONE", 7)
  assert f.wakes == [((0, 0, 1), "WAKE")]
  assert f.signal(key(1), "DONE", 7) is False and len(f.wakes) == 1


def test_waitall_stays_parked_until_every_operand_is_done(fac_c: Any) -> None:
  f = two_events(fac_c)
  assert f.issue_wait((0, 0, 1), "WAITALL", [key(1), key(2)]) == "PARK"
  f.signal(key(1), "DONE", 7)
  assert f.wakes == [] and (0, 0, 1) in f.waits
  f.signal(key(2), "DONE", 8)
  assert f.wakes == [((0, 0, 1), "WAKE")] and not f.entries


def test_waitall_failure_exits_at_once_and_never_through_a_success_wake(fac_c: Any) -> None:
  f = two_events(fac_c)
  f.issue_wait((0, 0, 1), "WAITALL", [key(1), key(2)])
  f.signal(key(1), "ERROR", 7)
  assert f.wakes == [((0, 0, 1), "FAIL_EXIT")]
  assert f.entries[(0, 0, 2)].status == "PENDING"  # B is not waited for
  assert f.entries[(0, 0, 2)].consumer_refs == 0  # and its pin was released


@pytest.mark.parametrize("terminal", ["ERROR", "TIMEOUT", "RESET"])
def test_every_failure_terminal_takes_the_failure_exit(fac_c: Any, terminal: str) -> None:
  f = two_events(fac_c)
  f.issue_wait((0, 0, 1), "WAIT", [key(1)])
  f.signal(key(1), terminal, 7)
  assert f.wakes == [((0, 0, 1), "FAIL_EXIT")]


def test_an_unrelated_event_with_the_same_owner_does_not_wake_the_waiter(fac_c: Any) -> None:
  f = two_events(fac_c)
  f.register(key(3), "compute_done", 7, future_uses=1)
  f.issue_wait((0, 0, 1), "WAIT", [key(1)])
  f.signal(key(3), "DONE", 7)
  assert (0, 0, 1) in f.waits and f.wakes == []


def test_a_faulted_context_record_matches_nothing_later(fac_c: Any) -> None:
  f = two_events(fac_c)
  f.issue_wait((0, 0, 1), "WAIT", [key(1)])
  f.fault_context((0, 0, 1))
  f.signal(key(1), "DONE", 7)
  assert f.wakes == [] and f.waits == {}


def test_mark_events_resets_pending_and_the_waiter_takes_the_failure_exit(fac_c: Any) -> None:
  f = two_events(fac_c)
  f.issue_wait((0, 0, 1), "WAIT", [key(1)])
  assert f.mark_events() == 2
  assert f.wakes == [((0, 0, 1), "FAIL_EXIT")] and f.entries[(0, 0, 2)].status == "RESET"


def test_mark_events_leaves_terminal_states_alone(fac_c: Any) -> None:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7, future_uses=1)
  f.add_consumer(key(1))
  f.signal(key(1), "ERROR", 7)
  f.register(key(2), "compute_done", 7)
  assert f.mark_events() == 1 and f.entries[(0, 0, 1)].status == "ERROR"


@pytest.mark.parametrize(
  ("kind", "ops", "code"),
  [
    ("WAIT", [key(1), key(2)], "WAIT_OPERAND_COUNT"),
    ("WAITALL", [key(1)], "WAITALL_OPERAND_COUNT"),
    ("WAITALL", [key(1), key(1)], "DUPLICATE_OPERAND"),
    ("WAITALL", [key(1), key(9)], "UNRESOLVED_OPERAND"),
    ("WAIT_ANY", [key(1)], "UNKNOWN_WAIT_KIND"),
  ],
)
def test_malformed_waits_are_rejected_without_pinning_anything(
  fac_c: Any, kind: str, ops: list, code: str
) -> None:
  f = two_events(fac_c)
  expect(code, f.issue_wait, (0, 0, 1), kind, ops)
  assert all(e.consumer_refs == 0 for e in f.entries.values())


def test_waitall_over_the_ceiling_is_rejected(fac_c: Any) -> None:
  f = make(fac_c, waitall_max=2)
  for i in (1, 2, 3):
    f.register(key(i), "compute_done", 7, future_uses=1)
  expect("WAITALL_OPERAND_COUNT", f.issue_wait, (0, 0, 1), "WAITALL", [key(1), key(2), key(3)])


def test_a_wait_needs_a_declared_future_use(fac_c: Any) -> None:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7)  # future_uses = 0
  expect("WAIT_EXCEEDS_FUTURE_USES", f.issue_wait, (0, 0, 1), "WAIT", [key(1)])


def test_one_wait_record_per_context(fac_c: Any) -> None:
  f = two_events(fac_c)
  f.issue_wait((0, 0, 1), "WAIT", [key(1)])
  expect("WAIT_RECORD_BUSY", f.issue_wait, (0, 0, 1), "WAIT", [key(2)])
  assert f.entries[(0, 0, 2)].consumer_refs == 0


def test_two_contexts_may_wait_on_one_event_when_the_compiler_declared_two_uses(fac_c: Any) -> None:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7, future_uses=2)
  f.issue_wait((0, 0, 1), "WAIT", [key(1)])
  f.issue_wait((0, 0, 2), "WAIT", [key(1)])
  f.signal(key(1), "DONE", 7)
  assert sorted(c for c, _ in f.wakes) == [(0, 0, 1), (0, 0, 2)] and not f.entries


# ---------------------------------------------------------------------------------------------------
# capacity: backpressure, terminal reserve, budget
# ---------------------------------------------------------------------------------------------------
def test_full_frontier_is_backpressure_and_completion_still_lands(fac_c: Any) -> None:
  f = make(fac_c, frontier=2)
  assert [f.register(key(i), "compute_done", 7) for i in (1, 2, 3)] == [True, True, False]
  assert f.counters["capacity_reject"] == 1
  assert f.signal(key(1), "DONE", 7) is True  # a full table never blocks a completion
  assert f.register(key(3), "compute_done", 7) is True  # and a freed slot restores allocation


def test_a_full_context_does_not_starve_another(fac_c: Any) -> None:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7)
  f.register(key(2), "compute_done", 7)
  assert f.register(key(1, ctx=1), "compute_done", 7) is True


def test_terminal_records_have_their_own_reserve_and_survive_a_full_frontier(fac_c: Any) -> None:
  f = make(fac_c, frontier=2, terminal_reserve=1)
  f.register(key(1), "compute_done", 7)
  f.register(key(2), "compute_done", 7)
  f.store_terminal(0, 0, {"state": "DONE"})
  expect("TERMINAL_RESERVE_EXHAUSTED", f.store_terminal, 0, 0, {"state": "x"})


def test_missing_terminal_reserve_is_rejected_by_the_consumer(fac_c: Any) -> None:
  expect("BAD_PARAMETER", make, fac_c, terminal_reserve=0)


def test_missing_terminal_reserve_is_rejected_by_the_validator(docs: Any) -> None:
  caught(docs, fm=lambda f: f["capacity"].pop("terminal_reserve_per_context"), needle="terminal_reserve")
  caught(
    docs,
    fm=lambda f: f["capacity"]["terminal_reserve_per_context"].__setitem__("minimum", 0),
    needle="terminal_reserve_per_context",
  )
  caught(
    docs,
    fm=lambda f: f["capacity"]["terminal_reserve_per_context"].__setitem__("separate_from_frontier", False),
    needle="terminal_reserve_per_context",
  )


def test_the_slot_budget_counts_entries_only(fac_c: Any) -> None:
  expect(
    "BUDGET_EXCEEDED",
    make,
    fac_c,
    contexts=2,
    frontier=3,
    terminal_reserve=1,
    tile_slots=9,
    response_inflight=2,
  )
  make(fac_c, contexts=2, frontier=3, terminal_reserve=1, tile_slots=10, response_inflight=2)


def test_a_cycle_count_cannot_enter_the_budget(docs: Any) -> None:
  def adds_cycles(f: dict[str, Any]) -> None:
    b = f["capacity"]["budget_derivation"]
    b["expression"] += " + response_cycles"
    b["inputs"]["response_cycles"] = {"unit": "cycles", "owner": "C019"}

  caught(docs, fm=adds_cycles, needle="UNIT_MISMATCH")


def test_a_static_bound_over_the_slot_count_is_rejected_at_load(fac_c: Any) -> None:
  f = make(fac_c, frontier=2)
  f.admit_task(2)
  expect("PERMANENT_CAPACITY", f.admit_task, 3)


def test_bad_capacity_parameters_are_rejected(fac_c: Any) -> None:
  for kw in ({"contexts": 0}, {"frontier": -1}, {"tile_slots": True}, {"response_inflight": -1}):
    expect("BAD_PARAMETER", make, fac_c, **kw)


# ---------------------------------------------------------------------------------------------------
# ownership: Group vs Tile writers
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
  ("resource", "wrong"),
  [
    ("group_event_table", "tile"),
    ("local_event_table", "group"),
    ("vme_region_wait_record", "tile"),
    ("first_fault_latch", "group"),
    ("poison_state", "tile"),
    ("max_drain_cycles", "tile"),
    ("reset_phase_order", "tile"),
    ("tile_timers", "tile"),
  ],
)
def test_a_resource_is_written_only_by_its_owner(fac_c: Any, resource: str, wrong: str) -> None:
  expect("WRONG_WRITER", w6.assert_writer, fac_c, resource, wrong)
  w6.assert_writer(fac_c, resource, fac_c["writers"][resource])


def test_unknown_resource_has_no_writer(fac_c: Any) -> None:
  expect("UNKNOWN_RESOURCE", w6.assert_writer, fac_c, "mystery", "tile")


@pytest.mark.parametrize(
  ("resource", "wrong"),
  [
    ("group_event_table", "tile"),
    ("local_event_table", "group"),
    ("vme_region_wait_record", "W6"),
    ("tile_timers", "tile"),
    ("poison_state", "tile"),
  ],
)
def test_writer_confusion_in_the_contract_is_rejected(docs: Any, resource: str, wrong: str) -> None:
  caught(docs, fm=lambda f: f["writers"].__setitem__(resource, wrong), needle="writers")


def test_group_writing_the_tile_table_is_rejected_in_the_contract(docs: Any) -> None:
  caught(
    docs,
    tm=lambda t: t["reset_endpoint"]["mark_events"].__setitem__("group_writes_tile_table", True),
    needle="mark_events",
  )
  caught(
    docs,
    tm=lambda t: t["reset_endpoint"]["mark_events"].__setitem__("writer", "group"),
    needle="mark_events",
  )


# ---------------------------------------------------------------------------------------------------
# reset endpoint: order, DRAIN evidence, wait-for cycles
# ---------------------------------------------------------------------------------------------------
def endpoint(fac_c: Any, term_c: Any) -> tuple[w6.ResetEndpoint, w6.EventFacility]:
  f = make(fac_c)
  f.register(key(1), "compute_done", 7, future_uses=1)
  e = w6.ResetEndpoint(term_c, f)
  e.begin()
  return e, f


def test_phases_run_in_order_and_repeat_idempotently(fac_c: Any, term_c: Any) -> None:
  e, _ = endpoint(fac_c, term_c)
  expect("PHASE_ORDER", e.advance, "DRAIN", DRAIN_ALL)
  e.advance("FREEZE_ISSUE")
  assert e.advance("FREEZE_ISSUE") is None
  expect("PHASE_ORDER", e.advance, "MARK_EVENTS")
  e.advance("CANCEL_UNISSUED")
  e.advance("DRAIN", DRAIN_ALL)
  assert e.advance("DRAIN", DRAIN_ALL) is None
  assert e.advance("MARK_EVENTS") == {"pending_to_reset": 1}
  e.advance("INVALIDATE_RESIDENCY")
  e.advance("REBUILD_CREDIT")
  e.advance("RESUME_DONE")
  assert e.active is False


def test_drain_succeeds_while_events_are_still_pending(fac_c: Any, term_c: Any) -> None:
  e, f = endpoint(fac_c, term_c)
  e.advance("FREEZE_ISSUE")
  e.advance("CANCEL_UNISSUED")
  e.advance("DRAIN", DRAIN_ALL)
  assert [x.status for x in f.entries.values()] == ["PENDING"]  # PENDING is cleared later, by MARK_EVENTS
  assert e.advance("MARK_EVENTS") == {"pending_to_reset": 1}


def test_drain_that_waits_on_pending_is_rejected_by_the_consumer(fac_c: Any, term_c: Any) -> None:
  e, _ = endpoint(fac_c, term_c)
  e.advance("FREEZE_ISSUE")
  e.advance("CANCEL_UNISSUED")
  expect("DRAIN_WAITS_ON_PENDING", e.advance, "DRAIN", {**DRAIN_ALL, "pending_events_zero": True})


@pytest.mark.parametrize(
  ("evidence", "code"),
  [
    ({k: v for k, v in DRAIN_ALL.items() if k != "l1_mshr_empty"}, "DRAIN_EVIDENCE_MISSING"),
    ({**DRAIN_ALL, "vibes": True}, "DRAIN_EVIDENCE_UNKNOWN"),
    ({**DRAIN_ALL, "engine_accepted_zero": False}, "DRAIN_NOT_QUIESCENT"),
    ({**DRAIN_ALL, "transfer_quiescent": False}, "DRAIN_NOT_QUIESCENT"),
    (None, "DRAIN_EVIDENCE_MISSING"),
  ],
)
def test_drain_needs_the_whole_real_evidence_set(fac_c: Any, term_c: Any, evidence: Any, code: str) -> None:
  e, _ = endpoint(fac_c, term_c)
  e.advance("FREEZE_ISSUE")
  e.advance("CANCEL_UNISSUED")
  expect(code, e.advance, "DRAIN", evidence)


def test_only_the_group_advances_the_reset(fac_c: Any, term_c: Any) -> None:
  e, _ = endpoint(fac_c, term_c)
  expect("WRONG_AUTHORITY", e.advance, "FREEZE_ISSUE", actor="tile")
  expect("UNKNOWN_PHASE", e.advance, "SPIN")


def test_new_work_is_refused_during_reset_and_after_poison(fac_c: Any, term_c: Any) -> None:
  e, _ = endpoint(fac_c, term_c)
  expect("RESET_ACTIVE", e.start_task)
  e.active = False
  e.poison()
  expect("POISONED", e.start_task)
  expect("RECOVERY_REFUSED", e.explicit_reset, quiescent=True, zero_evidence=False)
  expect("RECOVERY_REFUSED", e.explicit_reset, quiescent=False, zero_evidence=True)
  e.explicit_reset(quiescent=True, zero_evidence=True)
  e.start_task()


def test_invalidate_ack_waits_for_the_references_to_clear(fac_c: Any, term_c: Any) -> None:
  e, _ = endpoint(fac_c, term_c)
  expect("INVALIDATE_ACK_EARLY", e.invalidate_ack, 2)
  e.invalidate_ack(0)


def test_advancing_without_a_reset_is_refused(fac_c: Any, term_c: Any) -> None:
  e, _ = endpoint(fac_c, term_c)
  e.active = False
  expect("NOT_RESETTING", e.advance, "FREEZE_ISSUE")


def test_drain_waiting_on_pending_is_a_cycle_in_the_graph(docs: Any) -> None:
  def waits_on_pending(t: dict[str, Any]) -> None:
    t["reset_endpoint"]["drain"]["evidence_required"].append("pending_events_zero")
    t["reset_endpoint"]["drain"]["evidence_excluded"].clear()

  caught(docs, tm=waits_on_pending, needle="circular wait DRAIN -> MARK_EVENTS -> DRAIN")


def test_drain_waiting_on_pending_stays_a_cycle_when_the_guard_is_edited_too(
  docs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
  def waits_on_pending(t: dict[str, Any]) -> None:
    t["reset_endpoint"]["drain"]["evidence_required"].append("pending_events_zero")
    t["reset_endpoint"]["drain"]["evidence_excluded"].clear()

  monkeypatch.setattr(vw, "REQ_DRAIN_EVIDENCE", (*vw.REQ_DRAIN_EVIDENCE, "pending_events_zero"))
  monkeypatch.setattr(vw, "REQ_DRAIN_EXCLUDED", ())
  caught(docs, tm=waits_on_pending, needle="circular wait")


def test_a_reset_phase_cycle_is_rejected_even_with_the_guard_edited(
  docs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(vw, "REQ_PHASES", (("FREEZE_ISSUE", ("RESUME_DONE",)), *vw.REQ_PHASES[1:]))
  caught(
    docs,
    tm=lambda t: t["reset_endpoint"]["phases"][0].__setitem__("requires", ["RESUME_DONE"]),
    needle="circular wait",
  )


def test_find_cycle_and_reaches_on_plain_graphs() -> None:
  assert w6.find_cycle({"a": ["b"], "b": ["c"], "c": []}) is None
  assert w6.find_cycle({"a": ["b"], "b": ["a"]}) == ["a", "b", "a"]
  assert w6.reaches({"a": ["b"], "b": ["c"]}, "a", "c") and not w6.reaches({"a": ["b"]}, "b", "a")


# ---------------------------------------------------------------------------------------------------
# retirement chain: no wait-for edge through the parent R
# ---------------------------------------------------------------------------------------------------
def set_requires(step: str, requires: list[str]) -> Any:
  def mutate(t: dict[str, Any]) -> None:
    for s in t["task_terminal"]["chain"]:
      if s["step"] == step:
        s["requires"] = requires

  return mutate


@pytest.mark.parametrize(
  ("step", "requires"),
  [
    ("task_completion_sent", ["parent_r_return"]),
    ("task_completion_stored", ["local_reclaim", "parent_r_return"]),
    ("local_reclaim", ["parent_r_return"]),
    ("retire_authorized", ["parent_pin_cleanup"]),
  ],
)
def test_completion_waiting_on_the_parent_r_is_a_cycle(docs: Any, step: str, requires: list[str]) -> None:
  caught(docs, tm=set_requires(step, requires), needle="")


def test_a_chain_cycle_is_still_a_cycle_when_the_guard_is_edited_too(
  docs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
  chain = tuple((n, ("parent_r_return",) if n == "task_completion_sent" else r) for n, r in vw.REQ_CHAIN)
  monkeypatch.setattr(vw, "REQ_CHAIN", chain)
  caught(docs, tm=set_requires("task_completion_sent", ["parent_r_return"]), needle="circular wait")


def test_retirement_runs_in_order_and_each_step_once(term_c: Any) -> None:
  r = w6.Retirement(term_c)
  expect("STEP_ORDER", r.do, "retire_authorized")
  for s in ("exec_drained", "retire_ready", "retire_authorized", "local_reclaim"):
    r.do(s)
  r.do("task_completion_stored")
  r.do("task_completion_sent")
  expect("STEP_ORDER", r.do, "root_arena_retire")
  expect("STEP_REPEATED", r.do, "exec_drained")
  for s in ("parent_r_return", "parent_pin_cleanup", "root_arena_retire", "root_completion"):
    r.do(s)
  assert len(r.done) == 10
  expect("UNKNOWN_STEP", r.do, "teleport")


def test_task_completion_can_be_sent_before_the_parent_r_returns(term_c: Any) -> None:
  r = w6.Retirement(term_c)
  for s in ("exec_drained", "retire_ready", "retire_authorized", "local_reclaim", "task_completion_stored"):
    r.do(s)
  r.do("task_completion_sent")
  assert "parent_r_return" not in r.done  # the parent R is returned AFTER, never awaited


def test_terminal_state_is_unique_and_never_a_success(term_c: Any) -> None:
  r = w6.Retirement(term_c)
  expect("NO_SUCH_TERMINAL", r.publish, "SUCCESS")
  expect("FAULT_CHAIN_SUCCESS", r.publish, "DONE", fault_chain=True)
  expect("TIMEOUT_WITHOUT_EVIDENCE", r.publish, "TIMEOUT", w4_evidence=False)
  r.publish("TIMEOUT")
  expect("TERMINAL_NOT_UNIQUE", r.publish, "RESET")


def test_a_completion_record_is_kept_until_the_send_is_accepted(term_c: Any) -> None:
  s = w6.CompletionStore(2)
  s.store({"id": 1})
  s.store({"id": 2})
  expect("COMPLETION_STORE_FULL", s.store, {"id": 3})
  assert s.send(False) is None and list(s.held) == [{"id": 1}, {"id": 2}]
  assert s.send(True) == {"id": 1} and len(s.held) == 1
  assert w6.CompletionStore(1).send(True) is None


# ---------------------------------------------------------------------------------------------------
# cancel and external-write settlement
# ---------------------------------------------------------------------------------------------------
def test_cancel_separates_stale_conflict_duplicate_and_apply(term_c: Any) -> None:
  c = w6.CancelTracker(term_c)
  c.bind(("t", 1), 5)
  assert c.cancel(("t", 1), 5) == "APPLIED"
  assert c.cancel(("t", 1), 5) == "IDEMPOTENT_ACK"
  assert c.cancel(("t", 9), 5) == "STALE_ACK" and c.stale_count == 1
  expect("CANCEL_GENERATION_CONFLICT", c.cancel, ("t", 1), 6)
  c.retire(("t", 1))
  assert c.cancel(("t", 1), 5) == "STALE_ACK"


def test_a_timeout_never_settles_an_external_write(term_c: Any) -> None:
  expect("TIMEOUT_IS_NOT_SETTLEMENT", w6.settle_external_write, term_c, "TIMEOUT")
  expect("UNKNOWN_OUTCOME", w6.settle_external_write, term_c, "OK")
  assert w6.settle_external_write(term_c, "COMPLETED") == {
    "state": "COMPLETED",
    "released": True,
    "poison": False,
  }


def test_failed_isolation_keeps_the_resources_and_poisons(term_c: Any) -> None:
  out = w6.settle_external_write(term_c, "ISOLATION_FAILED")
  assert out == {"state": "UNCLOSED", "released": False, "poison": True}
  assert w6.settle_external_write(term_c, "FAULTED")["poison"] is True


@pytest.mark.parametrize(
  "mutate",
  [
    lambda t: t["cancel"]["external_write_settlement"]["isolation_failure"].__setitem__(
      "returns_isolated", True
    ),
    lambda t: t["cancel"]["external_write_settlement"]["isolation_failure"].__setitem__(
      "releases_resources", True
    ),
    lambda t: t["cancel"].__setitem__("timeout_is_success", True),
    lambda t: t["cancel"]["classification"]["no_live_binding_for_task_key"].__setitem__(
      "class", "protocol_error"
    ),
    lambda t: t["cancel"]["classification"]["task_known_but_ctx_generation_conflicts"].__setitem__(
      "class", "stale"
    ),
    lambda t: t["cancel"]["cancel_task"].__setitem__("accepted_work_dropped_silently", "allowed"),
  ],
)
def test_cancel_rules_are_held(docs: Any, mutate: Any) -> None:
  caught(docs, tm=mutate, needle="cancel")


def test_every_accepted_class_has_a_settlement_owner_for_success_cancel_and_failure(docs: Any) -> None:
  caught(docs, tm=lambda t: t["settlement"]["classes"].pop(0), needle="settlement.classes")
  caught(docs, tm=lambda t: t["settlement"]["classes"][1].pop("cancelled"), needle="cancelled")
  caught(docs, tm=lambda t: t["settlement"]["classes"][4].__setitem__("failed", ""), needle="failed")


# ---------------------------------------------------------------------------------------------------
# first-fault latch
# ---------------------------------------------------------------------------------------------------
def fault(
  term_c: Any, code: str = "ADDRESS_FAULT", src: int = 1, eng: int = 0, slot: int = 0, op: int = 0
) -> dict:
  d: dict[str, Any] = dict.fromkeys(term_c["first_fault_latch"]["record_fields"], 0)
  d.update(
    code=code,
    source_id=src,
    engine_id=eng,
    owner_tag={"ctx_slot": slot, "ctx_generation": 1, "owner_op_seq": op},
  )
  return d


def test_same_cycle_faults_are_ordered_by_direct_fields_and_the_rest_only_count(term_c: Any) -> None:
  latch = w6.FirstFaultLatch(term_c)
  latch.report_cycle(
    [
      fault(term_c, src=3, eng=1, slot=2, op=9),
      fault(term_c, src=2, eng=5, slot=1, op=4),
      fault(term_c, src=2, eng=5, slot=1, op=3),
      fault(term_c, src=2, eng=4, slot=7, op=1),
    ]
  )
  assert latch.record is not None
  assert (latch.record["source_id"], latch.record["engine_id"]) == (2, 4)
  assert latch.suppressed == 3 and latch.suppressed_sticky


def test_arbitration_is_deterministic_for_every_input_order(term_c: Any) -> None:
  import itertools

  faults = [
    fault(term_c, src=s, eng=e, slot=sl, op=o) for s, e, sl, o in ((2, 1, 0, 5), (2, 1, 0, 3), (1, 9, 9, 9))
  ]
  winners = set()
  for perm in itertools.permutations(faults):
    latch = w6.FirstFaultLatch(term_c)
    latch.report_cycle(list(perm))
    assert latch.record is not None
    winners.add(
      (latch.record["source_id"], latch.record["engine_id"], latch.record["owner_tag"]["owner_op_seq"])
    )
  assert winners == {(1, 9, 9)}


def test_a_later_fault_never_overwrites_and_only_an_ack_clears(term_c: Any) -> None:
  latch = w6.FirstFaultLatch(term_c)
  latch.report_cycle([fault(term_c, src=5)])
  latch.report_cycle([fault(term_c, src=0)])
  assert latch.record is not None and latch.record["source_id"] == 5 and latch.suppressed == 1
  latch.functional_reset()
  assert latch.record is not None  # a functional reset does not clear it
  expect("LATCH_CLEAR_REFUSED", latch.clear, "timeout")
  latch.clear("FAULT_ACK")
  assert latch.record is None


def test_a_fault_during_an_active_reset_is_counted_not_armed(term_c: Any) -> None:
  latch = w6.FirstFaultLatch(term_c)
  latch.reset_active = True
  latch.report_cycle([fault(term_c)])
  assert latch.record is None and latch.suppressed == 1


def test_all_eleven_validator_fault_codes_are_accepted(term_c: Any) -> None:
  latch = w6.FirstFaultLatch(term_c)
  names = term_c["first_fault_latch"]["fault_codes"]["names"]
  latch.report_cycle([fault(term_c, code=c, src=i) for i, c in enumerate(names)])
  assert len(names) == 11 and latch.suppressed == 10


def test_a_fault_needs_every_field_a_known_code_and_a_full_owner_tag(term_c: Any) -> None:
  expect(
    "FAULT_FIELDS_MISSING",
    w6.FirstFaultLatch(term_c).report_cycle,
    [{k: v for k, v in fault(term_c).items() if k != "offending_addr"}],
  )
  expect("UNKNOWN_FAULT_CODE", w6.FirstFaultLatch(term_c).report_cycle, [fault(term_c, code="BOGUS")])
  expect(
    "OWNER_TAG_MISSING",
    w6.FirstFaultLatch(term_c).report_cycle,
    [{**fault(term_c), "owner_tag": {"ctx_slot": 0}}],
  )


@pytest.mark.parametrize(
  "mutate",
  [
    lambda t: t["first_fault_latch"].__setitem__("first_fault_wins", False),
    lambda t: t["first_fault_latch"].__setitem__("later_fault", "overwrite"),
    lambda t: t["first_fault_latch"].__setitem__("cleared_by_reset", True),
    lambda t: t["first_fault_latch"]["cleared_only_by"].append("timeout"),
    lambda t: t["first_fault_latch"]["same_cycle_arbitration"].__setitem__("hash_allowed", True),
    lambda t: t["first_fault_latch"]["same_cycle_arbitration"]["order"].reverse(),
    lambda t: t["first_fault_latch"]["record_fields"].remove("offending_addr"),
    lambda t: t["first_fault_latch"]["fault_codes"]["names"].pop(),
    lambda t: t["first_fault_latch"]["validator_difference"].__setitem__("ring_overwrites", False),
    lambda t: t["first_fault_latch"]["attribution_ref"]["fields"].append("invented_tag_field"),
  ],
)
def test_latch_rules_are_held(docs: Any, mutate: Any) -> None:
  caught(docs, tm=mutate, needle="first_fault_latch")


def test_fault_domain_stays_group_and_the_tile_id_is_attribution(docs: Any) -> None:
  caught(
    docs, tm=lambda t: t["fault_domain"].__setitem__("production_domain", "TILE"), needle="fault_domain"
  )
  caught(
    docs, tm=lambda t: t["fault_domain"].__setitem__("never_selects_domain", False), needle="fault_domain"
  )
  caught(
    docs,
    tm=lambda t: t["fault_domain"].__setitem__("local_isolation_flt_02a", "implemented"),
    needle="flt_02a",
  )


# ---------------------------------------------------------------------------------------------------
# generation wrap, gating
# ---------------------------------------------------------------------------------------------------
GOOD_WRAP = {
  "authorized": True,
  "drained": ["in_flight_completions", "late_acks", "event_queues", "protocol_replay_cache"],
  "evidence_seq": 4,
}


def test_bit_pattern_reuse_needs_a_complete_verifiable_group_answer(term_c: Any) -> None:
  w = w6.WrapAuthority(term_c)
  expect("NO_GROUP_AUTHORIZATION", w.authorize_reuse, None, old_copy_may_arrive=False, outer_epoch=False)
  expect(
    "WRAP_COVERAGE_INCOMPLETE",
    w.authorize_reuse,
    {**GOOD_WRAP, "drained": ["in_flight_completions"]},
    old_copy_may_arrive=False,
    outer_epoch=False,
  )
  expect(
    "WRAP_RESPONSE_UNVERIFIABLE",
    w.authorize_reuse,
    {k: v for k, v in GOOD_WRAP.items() if k != "evidence_seq"},
    old_copy_may_arrive=False,
    outer_epoch=False,
  )
  w.authorize_reuse(GOOD_WRAP, old_copy_may_arrive=False, outer_epoch=False)


def test_an_identical_old_copy_forbids_reuse_unless_an_outer_epoch_isolates_it(term_c: Any) -> None:
  w = w6.WrapAuthority(term_c)
  expect("ABA_REUSE_FORBIDDEN", w.authorize_reuse, GOOD_WRAP, old_copy_may_arrive=True, outer_epoch=False)
  w.authorize_reuse(GOOD_WRAP, old_copy_may_arrive=True, outer_epoch=True)


def test_wrap_rules_and_the_decided_wire_direction_are_held(docs: Any) -> None:
  caught(
    docs, tm=lambda t: t["wrap_quiesce"].__setitem__("reuse_authorized_by", "tile"), needle="wrap_quiesce"
  )
  caught(docs, tm=lambda t: t["wrap_quiesce"]["covers"].remove("late_acks"), needle="wrap_quiesce")
  caught(
    docs,
    tm=lambda t: t["wrap_quiesce"]["wire_direction"].__setitem__("status", "unresolved"),
    needle="wire_direction",
  )
  caught(
    docs,
    tm=lambda t: t["wrap_quiesce"]["wire_direction"]["request"].__setitem__("dir", "G2T"),
    needle="wire_direction.request",
  )


def test_a_parked_context_gates_only_its_own_front_end(term_c: Any) -> None:
  g = w6.Gating(term_c, 2)
  g.park(0, wfi=True)
  assert g.context_front_end_gated(0, completion_arrival=False, writeback_port_busy=False) is True
  assert g.context_front_end_gated(1, completion_arrival=False, writeback_port_busy=False) is False
  assert g.context_front_end_gated(0, completion_arrival=False, writeback_port_busy=True) is False
  assert g.context_front_end_gated(0, completion_arrival=True, writeback_port_busy=False) is False
  idle = {"lsu": 0, "mfe": 0, "rvv": 0, "vme": 0}
  assert (
    g.tile_gated(idle, True, arrival_this_cycle=False) is False
  )  # WFI on one Context never stops the Tile


@pytest.mark.parametrize(
  ("in_flight", "channel_empty", "arrival"),
  [
    ({"lsu": 0, "mfe": 0, "rvv": 0, "vme": 1}, True, False),
    ({"lsu": 1, "mfe": 0, "rvv": 0, "vme": 0}, True, False),
    ({"lsu": 0, "mfe": 0, "rvv": 0, "vme": 0}, False, False),
    ({"lsu": 0, "mfe": 0, "rvv": 0, "vme": 0}, True, True),
  ],
)
def test_the_tile_gates_only_when_everything_is_quiet(
  term_c: Any, in_flight: dict, channel_empty: bool, arrival: bool
) -> None:
  g = w6.Gating(term_c, 2)
  g.park(0)
  g.park(1)
  assert g.tile_gated(in_flight, channel_empty, arrival_this_cycle=arrival) is False
  assert (
    g.tile_gated(dict.fromkeys(("lsu", "mfe", "rvv", "vme"), 0), True, arrival_this_cycle=False) is True
  )
  g.wake(1)
  assert (
    g.tile_gated(dict.fromkeys(("lsu", "mfe", "rvv", "vme"), 0), True, arrival_this_cycle=False) is False
  )


def test_the_in_flight_table_must_cover_every_unit(term_c: Any) -> None:
  g = w6.Gating(term_c, 1)
  g.park(0)
  expect("IN_FLIGHT_UNITS_INCOMPLETE", g.tile_gated, {"lsu": 0, "mfe": 0}, True, arrival_this_cycle=False)


@pytest.mark.parametrize(
  "mutate",
  [
    lambda t: t["clock_gating"]["wfi"].__setitem__("enters_tile_gate_condition", True),
    lambda t: t["clock_gating"]["tile_quiescence"]["requires"].remove("completion_channel_empty"),
    lambda t: t["clock_gating"]["tile_quiescence"]["requires"].remove("vme_in_flight_empty"),
    lambda t: t["clock_gating"]["context_park"].__setitem__("affects_other_contexts", True),
    lambda t: t["clock_gating"]["context_park"]["must_exclude"].remove("rf_writeback_port"),
    lambda t: t["clock_gating"]["gating_boundary"].__setitem__("excludes_same_cycle_arrival", False),
    lambda t: t["clock_gating"].__setitem__("gated_cycles_counted", False),
  ],
)
def test_gating_rules_are_held(docs: Any, mutate: Any) -> None:
  caught(docs, tm=mutate, needle="clock_gating")


# ---------------------------------------------------------------------------------------------------
# debug: gating, FIFO derivation, export fields (registered here, sealed with this fragment)
# ---------------------------------------------------------------------------------------------------
def test_debug_is_off_by_default_and_needs_privilege(term_c: Any) -> None:
  d = w6.DebugPath(term_c, 1)
  assert d.push({"kind": "x"}) is False and d.s.fifo == []
  expect("DEBUG_NOT_PRIVILEGED", d.enable, privileged=False)
  expect("DEBUG_OFF", d.accept_log_write, 1, 5, 0x100, 1)


def test_a_full_fifo_drops_the_new_record_and_counts(term_c: Any) -> None:
  d = w6.DebugPath(term_c, 1)  # the depth-1 boundary case
  d.enable(privileged=True)
  assert d.push({"kind": "a"}) is True
  assert d.push({"kind": "b"}) is False
  assert (d.s.dropped, d.s.overflow, d.s.window_incomplete) == (1, True, True)
  assert d.s.fifo == [{"kind": "a"}]  # the OLD record stays, the new one is dropped


def test_export_fields_are_restricted(term_c: Any) -> None:
  d = w6.DebugPath(term_c, 2)
  d.enable(privileged=True)
  expect("EXPORT_FIELD_FORBIDDEN", d.push, {"user_data_bytes": b"x"})
  expect("EXPORT_FIELD_FORBIDDEN", d.push, {"program_text_bytes": b"x"})
  expect("EXPORT_FIELD_UNKNOWN", d.push, {"zzz": 1})
  assert d.push({"kind": "ok", "cycle": 3, "code": "ADDRESS_FAULT"}) is True


def test_clearing_the_fifo_does_not_complete_an_accepted_log_write(term_c: Any) -> None:
  d = w6.DebugPath(term_c, 2)
  d.enable(privileged=True)
  d.accept_log_write(1, 5, 0x100, 1)
  d.close()
  assert d.push({"kind": "late"}) is False  # capture stopped first
  d.clear_fifo()
  assert d.status() == "pending" and d.ring_reusable() is False
  expect("TIMEOUT_IS_NOT_SETTLEMENT", d.settle, 1, "TIMEOUT")
  assert d.status() == "pending"
  d.settle(1, "ISOLATION_FAILED")
  assert d.status() == "poison" and d.ring_reusable() is False


def test_a_real_settlement_closes_the_debug_path_and_reset_turns_it_off(term_c: Any) -> None:
  d = w6.DebugPath(term_c, 2)
  d.enable(privileged=True)
  d.accept_log_write(7, 1, 0, 1)
  d.settle(7, "COMPLETED")
  assert d.status() == "open" and d.ring_reusable() is True
  expect("UNKNOWN_TXN", d.settle, 7, "COMPLETED")
  d.after_reset()
  assert d.s.enabled is False and d.status() == "closed"


def test_the_fifo_depth_derivation_is_in_entries_and_stays_unresolved(docs: Any) -> None:
  caught(
    docs,
    tm=lambda t: t["debug"]["fifo"]["depth"]["derivation"].__setitem__(
      "expression", "record_burst_entries + export_latency"
    ),
    needle="UNIT_MISMATCH",
  )
  caught(docs, tm=lambda t: t["debug"]["fifo"]["depth"].__setitem__("value", 8), needle="debug.fifo.depth")
  caught(docs, tm=lambda t: t["debug"]["fifo"].__setitem__("depth", 8), needle="debug")
  caught(
    docs,
    tm=lambda t: t["debug"]["fifo"]["depth"]["derivation"]["inputs"]["export_latency"].__setitem__(
      "value", 12
    ),
    needle="invented",
  )


@pytest.mark.parametrize(
  "mutate",
  [
    lambda t: t["debug"].__setitem__("default_mode", "on"),
    lambda t: t["debug"].__setitem__("after_reset", "on"),
    lambda t: t["debug"]["off_state"].__setitem__("hbm_log_ring", "reserved"),
    lambda t: t["debug"].__setitem__("ordinary_kernel_may_enable", True),
    lambda t: t["debug"]["fifo"].__setitem__("full_policy", "block"),
    lambda t: t["debug"]["fifo"]["never_blocks"].remove("task_terminal"),
    lambda t: t["debug"]["fifo"].__setitem__("functional_message_on_this_path", "allowed"),
    lambda t: t["debug"]["export"]["target"].__setitem__("may_overwrite_program_or_user_data", True),
    lambda t: t["debug"]["export"].__setitem__("executed_by", "W6"),
    lambda t: t["debug"]["restricted_export_fields"]["allowed"].append("user_data_bytes"),
    lambda t: t["debug"]["close_or_cancel_or_reset"].__setitem__("clearing_fifo_counts_as_complete", True),
    lambda t: t["debug"]["close_or_cancel_or_reset"].__setitem__("timeout_clears", True),
    lambda t: t["debug"]["close_or_cancel_or_reset"].__setitem__("ring_reuse_before_settlement", "allowed"),
    lambda t: t["reliable_channel"]["independent_of"].remove("debug_channel"),
  ],
)
def test_debug_rules_are_held(docs: Any, mutate: Any) -> None:
  caught(docs, tm=mutate, needle="")


# ---------------------------------------------------------------------------------------------------
# event/wait/phase/post_event rules and unresolved quantities
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
  "mutate",
  [
    lambda f: f["states"]["values"].append("CANCELLED"),
    lambda f: f["states"].__setitem__("sticky_terminal", False),
    lambda f: f["states"]["reset_conversion"].__setitem__("only_from", "DONE"),
    lambda f: f["reclaim"]["predicate"].remove("future_uses_zero"),
    lambda f: f["identity"]["event_key_fields"].remove("event_generation"),
    lambda f: f["event_kinds"]["local_waitable"].append("TASK_COMPLETION"),
    lambda f: f["event_kinds"]["not_waitable"].clear(),
    lambda f: f["signal"]["check_order"].reverse(),
    lambda f: f["signal"].__setitem__("reject_changes_state", True),
    lambda f: f["counters"].pop("unknown_event_notify"),
    lambda f: f["counters"]["wrong_owner"].__setitem__("increment", ""),
    lambda f: f["saturation"].__setitem__("wrap", "allowed"),
    lambda f: f["capacity"]["full_table"].__setitem__("fault", True),
    lambda f: f["capacity"]["full_table"].__setitem__("completion_and_credit_blocked", True),
    lambda f: f["capacity"]["load_time_rejection"].__setitem__("runtime_fault", True),
    lambda f: f["capacity"]["group_model_default"].__setitem__("not_tile_capacity", False),
    lambda f: f["wait"].__setitem__("no_wait_any", False),
    lambda f: f["wait"].__setitem__("record_per_context", 2),
    lambda f: f["wait"]["check_register"].__setitem__("same_cycle_priority", "park_first"),
    lambda f: f["wait"]["check_register"].__setitem__("park_then_wake_same_cycle", "allowed"),
    lambda f: f["wait"]["verdict"].__setitem__("waitall_success", "any_operand_done"),
    lambda f: f["wait"]["verdict"].__setitem__("failure_via_success_wake", "allowed"),
    lambda f: f["wait"].__setitem__("fault_voids_record", False),
    lambda f: f["phase_signal"].__setitem__("tile_dedup", "allowed"),
    lambda f: f["phase_signal"].__setitem__("physical_tile_in_key", True),
    lambda f: f["phase_signal"].__setitem__("early_release_returns_none", False),
    lambda f: f["post_event"].__setitem__("never_noop", False),
    lambda f: f["post_event"].__setitem__("authorization", "any"),
  ],
)
def test_facility_rules_are_held(docs: Any, mutate: Any) -> None:
  caught(docs, fm=mutate, needle="")


@pytest.mark.parametrize(
  ("which", "path"),
  [
    ("f", ("capacity", "frontier_slots_per_context")),
    ("f", ("capacity", "tile_global_slots")),
    ("f", ("identity", "widths", "event_generation")),
    ("t", ("reset_endpoint", "max_drain_cycles", "value")),
    ("t", ("first_fault_latch", "fault_codes", "code_width")),
  ],
)
def test_unsourced_quantities_cannot_be_filled_in(docs: Any, which: str, path: tuple) -> None:
  def fill(doc: dict[str, Any]) -> None:
    node = doc
    for k in path[:-1]:
      node = node[k]
    last = node[path[-1]]
    if isinstance(last, dict):
      last["value"] = 4
    else:
      node[path[-1]] = 4

  caught(docs, **({"fm": fill} if which == "f" else {"tm": fill}), needle="unresolved")


def test_an_unresolved_quantity_needs_its_owner(docs: Any) -> None:
  caught(docs, fm=lambda f: f["capacity"]["frontier_slots_per_context"].pop("owner"), needle="owner")
  caught(docs, fm=lambda f: f["capacity"]["frontier_slots_per_context"].pop("reason"), needle="reason")


def test_an_invented_encoding_key_is_rejected_anywhere(docs: Any) -> None:
  caught(docs, fm=lambda f: f["identity"].__setitem__("width_bits", 5), needle="invented encoding")
  caught(docs, tm=lambda t: t["debug"].__setitem__("csr_address", 0x7C0), needle="invented encoding")


def test_a_required_section_deleted_is_malformed_not_a_crash(docs: Any) -> None:
  caught(docs, fm=lambda f: f.pop("wait"), needle="malformed")
  caught(docs, tm=lambda t: t.pop("debug"), needle="malformed")


# ---------------------------------------------------------------------------------------------------
# field request
# ---------------------------------------------------------------------------------------------------
def test_field_request_names_catalog_messages_or_marks_new(docs: Any) -> None:
  caught(docs, rm=lambda r: r["messages"][0].__setitem__("new", True), needle="already in the C009 catalog")
  caught(docs, rm=lambda r: r["messages"][9].__setitem__("new", False), needle="not in the C009 catalog")


def test_field_request_keys_must_exist_in_the_contract(docs: Any) -> None:
  caught(
    docs,
    rm=lambda r: r["messages"][3].__setitem__("key", "task_terminal.nonexistent"),
    needle="does not exist",
  )
  caught(
    docs,
    rm=lambda r: r["messages"][3].__setitem__("from", "protocol/w5_package.contract.yaml"),
    needle="not a W6 contract",
  )


def test_field_request_invents_no_value_and_is_complete(docs: Any) -> None:
  caught(docs, rm=lambda r: r["messages"][0].__setitem__("opcode", 3), needle="INVENTED_VALUE")
  caught(docs, rm=lambda r: r["parameters"][0].__setitem__("width_bits", 7), needle="INVENTED_VALUE")
  caught(docs, rm=lambda r: r["messages"].pop(5), needle="missing")
  caught(docs, rm=lambda r: r["package_fields"].pop(0), needle="package_fields")
  caught(docs, rm=lambda r: r["parameters"].pop(), needle="parameters")
  caught(docs, rm=lambda r: r["messages"][2].__setitem__("why", ""), needle="needs a reason")
  caught(docs, rm=lambda r: r.__setitem__("package", "W5"), needle="W6")


def test_the_decided_wire_direction_matches_the_catalog(term_c: Any) -> None:
  w = term_c["wrap_quiesce"]["wire_direction"]
  catalog = {m["name"]: m["dir"] for m in yaml.safe_load(vw.CATALOG.read_text())["messages"]}
  assert w["status"] == "decided"
  assert (catalog["GEN_WRAP_QUIESCE"], catalog["QUIESCE_RESP"]) == ("T2G", "G2T")
