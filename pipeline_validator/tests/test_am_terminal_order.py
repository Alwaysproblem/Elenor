"""P07 (SW-A4) terminal ordering and safe release.

Successful FINAL: ``compute_done <= step_committed <= output_visible <=
slot_free <= chain_done <= UCE event``; failures quiesce in-flight AM
accesses, then free, then publish the failure terminal.  The terminal ring is
a bounded diagnostic that never gates grants.

pytest:
  conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_am_terminal_order.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from pipeline_validator.compiler import compile_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.engines import EngineJob
from pipeline_validator.execution_ir import ExecBoaMatmulDesc, GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.runtime.boa_result import BoaPreparedResult, BoaResultService
from pipeline_validator.runtime.shared_am import AmOwner, SharedAmManager
from pipeline_validator.simulator import Simulator
from pipeline_validator.workload_ir import parse_workload_ir

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "examples" / "workloads" / "boa_am_matmul.mlir"
INPUTS = ROOT / "examples" / "workloads" / "boa_am_inputs" / "boa_am_matmul"
C_BASE, C_BYTES = 0x300000, 65536
BLOCKED_UNTIL = 20000
PAYLOAD = b"\x01" * (64 * 256 * 4)


# --- full parser/compiler/loader/Simulator scenario ---------------------------


def _run_fixture(*, blocked_until: int | None = None, on_cycle=None, seed: int = 0):
  a = (INPUTS / "A.bin").read_bytes()
  b = (INPUTS / "B.bin").read_bytes()
  bindings = {
    "A": GlobalBinding("A", 0x100000, len(a), "r"),
    "B": GlobalBinding("B", 0x200000, len(b), "r"),
    "C": GlobalBinding("C", C_BASE, C_BYTES, "w"),
  }
  hw = HardwareConfig()
  sim_cfg = SimConfig(fidelity="full_memory", context_count=1, seed=seed)
  artifact = compile_program(
    parse_workload_ir(FIXTURE.read_text()), hw, sim_cfg,
    binding_assumptions=bindings, source_name=str(FIXTURE),
  )
  store = ByteStore()
  store.seed_hbm(0x100000, a)
  store.seed_hbm(0x200000, b)
  simulator = Simulator(hw, sim_cfg, enable_tracer=True, byte_store=store)
  if blocked_until is not None:
    simulator.group.tiles[0].boa_results.set_output_blocked(True, until_cycle=blocked_until)
  loaded = load_program(artifact, hw, sim_cfg, actual_bindings=bindings)
  if on_cycle is None:
    result = simulator.run(loaded)
  else:
    original_step = simulator.group.step

    def step(cycle: int) -> bool:
      on_cycle(simulator, cycle)
      return original_step(cycle)

    with pytest.MonkeyPatch.context() as patch:
      patch.setattr(simulator.group, "step", step)
      result = simulator.run(loaded)
  assert simulator.tracer is not None
  simulator.tracer.assert_well_formed()
  events = json.loads(simulator.tracer.to_chrome_json())["traceEvents"]
  unit = simulator.tracer.cycle_to_us(1)
  # Emission order is the tracer's list order: it orders same-cycle events.
  marks = [
    dict(event["args"], name=event["name"], cycle=round(event["ts"] / unit), seq=index)
    for index, event in enumerate(events)
    if event.get("ph") == "i" and event.get("cat") in ("SharedAM", "BoaResult")
  ]
  return simulator, store, result, marks


def _one(marks: list[dict], name: str, **match) -> dict:
  hits = [m for m in marks if m["name"] == name and all(m.get(k) == v for k, v in match.items())]
  assert len(hits) == 1, (name, match, len(hits))
  return hits[0]


@pytest.mark.parametrize("blocked_until", [None, BLOCKED_UNTIL])
def test_success_final_order_is_joined_by_identity_in_trace(blocked_until):
  simulator, store, result, marks = _run_fixture(blocked_until=blocked_until)
  assert result.completed
  tile = simulator.group.tiles[0]
  service = tile.boa_results.snapshot()
  (record,) = [r for r in service["records"] if r["final"]]
  lease_id = record["lease_id"]
  (terminal,) = tile.shared_am.snapshot()["terminals"]
  assert (terminal["lease_id"], terminal["kind"], terminal["success"]) == (lease_id, "chain_done", True)

  commit = _one(marks, "am_commit", lease_id=lease_id, final=True)
  output = _one(marks, "boa_result_output_written", event_id=record["event_id"])
  release = _one(marks, "am_release", lease_id=lease_id)
  chain_terminal = _one(marks, "am_chain_terminal", lease_id=lease_id)
  ready = _one(marks, "boa_result_ready", event_id=record["event_id"])
  # Canonical non-strict cycle order, each stage from its own trace event.
  cycles = [
    record["compute_done_cycle"], commit["cycle"], output["cycle"],
    release["cycle"], chain_terminal["cycle"], ready["cycle"],
  ]
  assert cycles == sorted(cycles), cycles
  # Same-cycle stages are ordered by emission: commit -> output -> free ->
  # terminal -> event.  No artificial bubble cycles are required.
  seqs = [commit["seq"], output["seq"], release["seq"], chain_terminal["seq"], ready["seq"]]
  assert seqs == sorted(seqs), seqs
  assert release["cycle"] == chain_terminal["cycle"] == chain_terminal["slot_free_cycle"]
  assert ready["slot_free_cycle"] == release["cycle"] and ready["lease_id"] == lease_id
  assert record["event_cycle"] == ready["cycle"] >= terminal["cycle"]
  assert chain_terminal["generation"] == release["generation"]
  assert chain_terminal["bound_engine"] == release["bound_engine"]
  if blocked_until is not None:
    # Backpressure separates compute from commit/output by the blocked window,
    # and recovery is autonomous: no UCE action between unblock and event.
    assert service["blocked_cycles"] > 0
    assert commit["cycle"] >= blocked_until > record["compute_done_cycle"]
  else:
    assert service["blocked_cycles"] == 0
  order_check = next(c for c in _report_checks(simulator, result) if c["check"] == "am_terminal_order")
  assert order_check["pass"] is True and order_check["joined_finals"] == 1
  _ = store


def _report_checks(simulator, result) -> list[dict]:
  from pipeline_validator.compiled_program import WorkloadInfo
  from pipeline_validator.report import build_report, report_to_json

  report = build_report(
    WorkloadInfo(name="p07", description="p07", expected={}), result, num_tiles=simulator.hw.num_tiles
  )
  return json.loads(report_to_json(report))["checks"]


def test_backpressure_changes_timing_but_not_output_bytes():
  _sim0, free_store, free_result, _ = _run_fixture()
  _sim1, held_store, held_result, _ = _run_fixture(blocked_until=BLOCKED_UNTIL)
  assert free_result.completed and held_result.completed
  assert held_result.cycles > free_result.cycles
  assert held_store.read_hbm(C_BASE, C_BYTES) == free_store.read_hbm(C_BASE, C_BYTES)


def test_nonfinal_step_commits_without_terminal_or_output():
  seen: list[dict] = []

  def observe(simulator, cycle):
    manager = simulator.group.tiles[0].shared_am
    if manager.commit_count == 1 and not seen:
      seen.append({
        "terminals": manager.snapshot()["terminals"],
        "output_writes": simulator.group.tiles[0].boa_results.output_write_count,
        "live": manager.live_lease_count,
      })

  simulator, _store, result, marks = _run_fixture(on_cycle=observe)
  assert result.completed
  assert seen == [{"terminals": [], "output_writes": 0, "live": 1}]
  commits = [m for m in marks if m["name"] == "am_commit"]
  assert [m["final"] for m in commits] == [False, True]
  # Only the FINAL publishes a terminal; the non-FINAL event is step_committed.
  assert len([m for m in marks if m["name"] == "am_chain_terminal"]) == 1
  statuses = [m["status"] for m in marks if m["name"] == "boa_result_ready"]
  assert statuses == ["step_committed", "chain_done"]
  _ = simulator


def test_context_fault_during_output_backpressure_quiesces_then_frees():
  injected: list[int] = []

  def inject(simulator, cycle):
    tile = simulator.group.tiles[0]
    if not injected and tile.boa_results.snapshot()["states"]["wait_output"]:
      injected.append(cycle)
      tile.uce._fault_context(tile.uce.contexts[0], "injected output fault", cycle)

  simulator, store, result, marks = _run_fixture(blocked_until=BLOCKED_UNTIL, on_cycle=inject)
  assert injected and not result.completed
  tile = simulator.group.tiles[0]
  manager = tile.shared_am.snapshot()
  assert manager["live_leases"] == 0 and manager["free_slots"] == manager["total_slots"]
  assert manager["fault_count"] == 1
  (terminal,) = manager["terminals"]
  assert (terminal["kind"], terminal["success"]) == ("fault", False)
  assert terminal["detail"] == "injected output fault"
  release = _one(marks, "am_release", lease_id=terminal["lease_id"])
  chain_terminal = _one(marks, "am_chain_terminal", lease_id=terminal["lease_id"])
  # The FINAL result still occupied the lease when the fault hit: the slot is
  # held (FAULT_QUIESCING) until the result service retires that access, the
  # free happens, and only then the failure terminal is published.
  assert release["reason"] == "fault"
  assert chain_terminal["cycle"] >= release["cycle"] >= injected[0]
  assert chain_terminal["seq"] > release["seq"]
  # No partial output became visible and no FINAL commit happened.
  assert tile.boa_results.output_write_count == 0
  assert not [m for m in marks if m["name"] == "am_commit" and m["final"]]
  assert not [m for m in marks if m["name"] == "boa_result_ready" and m["status"] == "chain_done"]
  _data, validity = store._read_relaxed("hbm", C_BASE, C_BYTES)
  assert validity == bytes(C_BYTES), "no byte of C may become visible after the fault"

def test_group_fault_reset_with_pending_final_records_actual_cycle():
  """Production reset-domain path with a FINAL stuck in WAIT_OUTPUT.

  The reset must reclaim the lease and publish its failure terminal at the
  real reset cycle, never at a hard-coded cycle 0.
  """
  from pipeline_validator.runtime.fault_ring import FaultCode

  injected: list[int] = []

  def inject(simulator, cycle):
    tile = simulator.group.tiles[0]
    if not injected and tile.boa_results.snapshot()["states"]["wait_output"]:
      injected.append(cycle)
      simulator.group.trigger_fault(FaultCode.ENGINE_INTERNAL_FAULT, tile_id=0, cycle=cycle)

  simulator, store, result, marks = _run_fixture(blocked_until=BLOCKED_UNTIL, on_cycle=inject)
  assert injected and not result.completed
  tile = simulator.group.tiles[0]
  manager = tile.shared_am.snapshot()
  assert manager["live_leases"] == 0 and manager["free_slots"] == manager["total_slots"]
  assert manager["fault_count"] == 1
  (terminal,) = manager["terminals"]
  assert (terminal["kind"], terminal["success"]) == ("fault", False)
  release = _one(marks, "am_release", lease_id=terminal["lease_id"])
  chain_terminal = _one(marks, "am_chain_terminal", lease_id=terminal["lease_id"])
  # Every reclamation timestamp is the actual post-fault cycle.
  assert terminal["cycle"] == release["cycle"] == chain_terminal["cycle"] >= injected[0] > 0
  assert chain_terminal["seq"] > release["seq"]
  assert tile.boa_results.output_write_count == 0
  _data, validity = store._read_relaxed("hbm", C_BASE, C_BYTES)
  assert validity == bytes(C_BYTES)


def test_group_fault_reset_with_unissued_boa_records_actual_cycle(monkeypatch):
  """A queued step must retire at FREEZE_DISPATCH, not the default cycle 0."""
  from pipeline_validator.runtime.fault_ring import FaultCode
  from pipeline_validator.runtime.reset_domain import ResetState

  injected: list[int] = []
  cancelled: list[int] = []

  def inject(simulator, cycle):
    tile = simulator.group.tiles[0]
    queued = any(
      entry.am_lease_id is not None
      for fifo in tile.uce._engine_queues.values()
      for entry in fifo
    )
    if queued and not injected:
      injected.append(cycle)
      simulator.group.trigger_fault(FaultCode.ENGINE_INTERNAL_FAULT, tile_id=0, cycle=cycle)
    if simulator.group.reset_domain.state is ResetState.FREEZE_DISPATCH:
      cancelled.append(cycle)

  # Keep the request UCE-accepted but unissued, as with a full result FIFO.
  monkeypatch.setattr(BoaResultService, "has_capacity", lambda self: False)
  simulator, store, result, marks = _run_fixture(on_cycle=inject)
  assert len(injected) == len(cancelled) == 1 and not result.completed
  tile = simulator.group.tiles[0]
  manager = tile.shared_am.snapshot()
  assert manager["live_leases"] == manager["pending_requests"] == 0
  assert manager["free_slots"] == manager["total_slots"]
  (terminal,) = manager["terminals"]
  assert (terminal["kind"], terminal["success"]) == ("fault", False)
  release = _one(marks, "am_release", lease_id=terminal["lease_id"])
  chain_terminal = _one(marks, "am_chain_terminal", lease_id=terminal["lease_id"])
  assert terminal["cycle"] == release["cycle"] == chain_terminal["cycle"] == cancelled[0]
  assert cancelled[0] >= injected[0] > 0 and chain_terminal["seq"] > release["seq"]
  service = tile.boa_results.snapshot()
  assert service["pending"] == service["reservations"] == service["output_write_count"] == 0
  assert service["credits"] == service["credit_capacity"]
  _data, validity = store._read_relaxed("hbm", C_BASE, C_BYTES)
  assert validity == bytes(C_BYTES)



# --- manager / result-service units -------------------------------------------


def _request(manager, owner, name, cycle, wake=None):
  return manager.request(
    owner, buffer_name=name, logical_slot=0, dims=(64, 256), byte_count=65536,
    output_base=f"l1:{name}", bundle_slots=1, cycle=cycle,
    wake=wake or (lambda lease, grant: pytest.fail(f"unexpected wake {lease.lease_id}")),
  )


def _prepared(manager, lease_id, events):
  matmul = cast(ExecBoaMatmulDesc, SimpleNamespace(am="am:0", final=True, seq=0, init=True))

  def commit_final(cycle: int) -> None:
    manager.commit_step(lease_id, seq=0, data=PAYLOAD, final=True, cycle=cycle)
    events.append(("commit_final", cycle))

  def write_output(cycle: int) -> None:
    events.append(("write", cycle))

  def release(cycle: int) -> int:
    lease = manager.release_final(lease_id, cycle)
    events.append(("release", cycle))
    assert lease.free_cycle is not None
    return lease.free_cycle

  def abort(cycle: int, reason: str) -> None:
    manager.abort_lease(lease_id, cycle, reason, retire_access=True)
    events.append(("abort", cycle))

  return BoaPreparedResult(
    matmul, PAYLOAD, True, lambda cycle: None, commit_final, write_output, release, abort,
    lambda: manager.lease(lease_id).state.value == "fault_quiescing",
  )


def _job(lease_id: int) -> EngineJob:
  job = EngineJob(
    desc=SimpleNamespace(params={"lease_id": lease_id}),  # type: ignore[arg-type]
    start_cycle=0, finish_cycle=0, event_id="e",
  )
  job.compute_done_cycle = 0
  return job


def test_fault_with_inflight_access_holds_slot_until_quiesced():
  manager = SharedAmManager(
    slot_capacity_bytes=65536, group_count=1, slots_per_group=1, slots_per_context=1
  )
  owner, waiter = AmOwner("ctx", 1, 0, 0), AmOwner("waiter", 1, 1, 0)
  lease = _request(manager, owner, "a", 0)
  manager.begin_step(lease.lease_id, seq=0, init=True, final=True, cycle=1)
  woken: list[tuple[int, int]] = []
  waiter_wake = lambda new, cycle: woken.append((new.generation, cycle))
  assert _request(manager, waiter, "w", 2, wake=waiter_wake) is None
  service = BoaResultService(HardwareConfig())
  service.set_output_blocked(True)
  events: list[tuple[str, int]] = []
  job = _job(lease.lease_id)
  service.submit(job, _prepared(manager, lease.lease_id, events), 0)
  assert service.advance(5) == []  # parked in WAIT_OUTPUT
  manager.abort_owner(owner, 6, "owner fault")
  held = manager.snapshot()
  # Quiescing: no free, no terminal, no wake while the result access is live.
  assert held["leases"][0]["state"] == "fault_quiescing" and held["leases"][0]["accesses"] == 1
  assert held["terminals"] == [] and held["free_slots"] == 0 and woken == []
  assert service.advance(7) == []
  assert service.pop_failures() == (("e", "AM lease faulted before commit"),)
  assert events == [("abort", 7)]
  after = manager.snapshot()
  (terminal,) = after["terminals"]
  assert (terminal["kind"], terminal["detail"], terminal["cycle"]) == ("fault", "owner fault", 7)
  assert after["fault_count"] == 1 and after["slot_free_count"] == 1
  # The waiter is granted only by the free, on a NEW generation; the old
  # chain never committed into it.
  assert woken == [(lease.generation + 1, 7)]
  assert service.snapshot()["records"][-1]["status"] == "fault"
  assert service.snapshot()["credits"] == service.snapshot()["credit_capacity"]


def test_double_abort_counts_one_fault_and_keeps_first_reason():
  manager = SharedAmManager(
    slot_capacity_bytes=65536, group_count=1, slots_per_group=1, slots_per_context=1
  )
  lease = _request(manager, AmOwner("ctx", 1, 0, 0), "a", 0)
  manager.begin_step(lease.lease_id, seq=0, init=True, final=False, cycle=1)
  manager.abort_lease(lease.lease_id, 2, "first")
  manager.abort_lease(lease.lease_id, 3, "second")
  assert manager.snapshot()["terminals"] == []
  manager.abort_lease(lease.lease_id, 4, "access retired", retire_access=True)
  snapshot = manager.snapshot()
  assert snapshot["fault_count"] == 1
  assert [(t["detail"], t["cycle"]) for t in snapshot["terminals"]] == [("first", 4)]


def test_release_final_refuses_pending_step():
  manager = SharedAmManager(
    slot_capacity_bytes=65536, group_count=1, slots_per_group=1, slots_per_context=1
  )
  lease = _request(manager, AmOwner("ctx", 1, 0, 0), "a", 0)
  manager.begin_step(lease.lease_id, seq=0, init=True, final=True, cycle=1)
  manager.commit_step(lease.lease_id, seq=0, data=PAYLOAD, final=True, cycle=2)
  lease.accesses += 1  # a stray second in-flight access on the same chain
  with pytest.raises(ValueError, match="pending steps or accesses"):
    manager.release_final(lease.lease_id, 3)
  assert manager.snapshot()["terminals"] == [] and manager.snapshot()["free_slots"] == 0


@pytest.mark.parametrize("capacity", [1, 3])
def test_terminal_ring_is_bounded_and_never_blocks_grants(capacity):
  manager = SharedAmManager(
    slot_capacity_bytes=65536, group_count=1, slots_per_group=1, slots_per_context=1,
    terminal_capacity=capacity,
  )
  owners = [AmOwner(f"ctx{i}", 1, i, 0) for i in range(capacity + 3)]
  lease = _request(manager, owners[0], "c0", 0)
  granted = [lease.lease_id]
  for index, owner in enumerate(owners[1:], start=1):
    woken: list = []
    wake = lambda new, _c, w=woken: w.append(new)
    assert _request(manager, owner, f"c{index}", 10 * index, wake=wake) is None
    manager.begin_step(lease.lease_id, seq=0, init=True, final=True, cycle=10 * index + 1)
    manager.commit_step(lease.lease_id, seq=0, data=PAYLOAD, final=True, cycle=10 * index + 2)
    manager.release_final(lease.lease_id, 10 * index + 3)
    # Ring full or not, the free grants the parked request immediately.
    assert len(woken) == 1
    lease = woken[0]
    granted.append(lease.lease_id)
  snapshot = manager.snapshot()
  finished = len(owners) - 1
  assert snapshot["terminal_capacity"] == capacity
  assert snapshot["terminal_overflow_count"] == finished - capacity
  assert [t["lease_id"] for t in snapshot["terminals"]] == granted[finished - capacity:finished]
  assert snapshot["live_leases"] == 1 and snapshot["acquire_count"] == len(owners)


# ---------------------------------------------------------------------------
# Module CLI (container target shared-am-cycle)
# ---------------------------------------------------------------------------


CLI_CASES = ["terminal-order-backpressure", "manager-terminal-units"]


def _unissued_reset_scenario() -> None:
  """The queued-unissued reset scenario under a real MonkeyPatch context."""
  with pytest.MonkeyPatch.context() as patch:
    test_group_fault_reset_with_unissued_boa_records_actual_cycle(patch)


def _capture_trace(blocked_until: int | None, seed: int, captured: dict, label: str) -> None:
  """Fresh real run whose full trace is kept for the independent SQL audit."""
  simulator, _store, result, _marks = _run_fixture(blocked_until=blocked_until, seed=seed)
  assert result.completed, result.reason
  order_check = next(c for c in _report_checks(simulator, result) if c["check"] == "am_terminal_order")
  assert order_check["pass"] is True and order_check["joined_finals"] == 1
  assert simulator.tracer is not None
  simulator.tracer.assert_well_formed()
  captured[f"{label}_trace"] = simulator.tracer.to_chrome_json()
  captured[f"{label}_cycles"] = result.cycles


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
  """Run the real P07 regressions and write report + trace artifacts."""
  from pathlib import Path

  from pipeline_validator.tests import case_cli

  args = case_cli.build_parser(
    "python -m pipeline_validator.tests.test_am_terminal_order",
    "P07 terminal order: compute_done <= step_committed <= output_visible <= "
    "slot_free <= chain_done <= UCE event; faults quiesce, free, then publish.",
    CLI_CASES,
  ).parse_args(argv)
  captured: dict = {}

  if args.case == "terminal-order-backpressure":
    spec: list = [
      ("success-final-order-joined-by-identity-free",
       lambda: test_success_final_order_is_joined_by_identity_in_trace(None)),
      ("success-final-order-joined-by-identity-backpressure",
       lambda: test_success_final_order_is_joined_by_identity_in_trace(BLOCKED_UNTIL)),
      ("backpressure-changes-timing-but-not-output-bytes",
       test_backpressure_changes_timing_but_not_output_bytes),
      ("nonfinal-step-commits-without-terminal-or-output",
       test_nonfinal_step_commits_without_terminal_or_output),
      ("context-fault-during-output-backpressure-quiesces-then-frees",
       test_context_fault_during_output_backpressure_quiesces_then_frees),
      ("group-fault-reset-with-pending-final-records-actual-cycle",
       test_group_fault_reset_with_pending_final_records_actual_cycle),
      ("group-fault-reset-with-unissued-boa-records-actual-cycle", _unissued_reset_scenario),
      ("trace-capture-free-run", lambda: _capture_trace(None, args.seed, captured, "free")),
      ("trace-capture-backpressure-run",
       lambda: _capture_trace(BLOCKED_UNTIL, args.seed, captured, "blocked")),
    ]
  else:
    spec = [
      ("fault-with-inflight-access-holds-slot-until-quiesced",
       test_fault_with_inflight_access_holds_slot_until_quiesced),
      ("double-abort-counts-one-fault-and-keeps-first-reason",
       test_double_abort_counts_one_fault_and_keeps_first_reason),
      ("release-final-refuses-pending-step", test_release_final_refuses_pending_step),
      ("terminal-ring-bounded-capacity-1",
       lambda: test_terminal_ring_is_bounded_and_never_blocks_grants(1)),
      ("terminal-ring-bounded-capacity-3",
       lambda: test_terminal_ring_is_bounded_and_never_blocks_grants(3)),
    ]
  checks = case_cli.run_checks(spec)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  for label in ("free", "blocked"):
    if f"{label}_trace" in captured:
      (out / f"{args.case}.{label}.trace.json").write_text(
        captured[f"{label}_trace"], encoding="utf-8"
      )
  report = case_cli.build_report(
    args.case,
    "test_am_terminal_order",
    args.seed,
    checks,
    metrics={key: value for key, value in captured.items() if key.endswith("_cycles")},
  )
  case_cli.write_report(out, report)
  return case_cli.finish(report)


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
