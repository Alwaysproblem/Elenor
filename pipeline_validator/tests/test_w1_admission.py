"""C035: failed holds compensate, authorized work drains, and retirement never waits for parent R."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.cosim import abi
from pipeline_validator.cosim.tile_port import BoundaryMessage
from pipeline_validator.execution_ir import ExecTileGroupTask
from pipeline_validator.memory import MemoryInvariantError
from pipeline_validator.memory.transfer import TransferStatus
from pipeline_validator.runtime.event_table import EventProtocolError, EventStatus, EventTable
from pipeline_validator.runtime.fault_ring import FaultCode
from pipeline_validator.runtime.reset_domain import ResetState
from pipeline_validator.tests.test_runtime import POW_BINDINGS, prepare_group_source
from pipeline_validator.tests.w1_lifecycle import ObservedNative, native_cases, start
from pipeline_validator.tile_group import TileGroup
from pipeline_validator.workloads import PowWorkload


def test_dispatch_quota_cannot_be_recycled_while_a_participant_owes_its_receipt() -> None:
  events = EventTable(capacity=1)
  assert events.reserve_quota(1, {"grid": 0}, owner="a", generation=7)
  events.reserve_outputs(("grid",), owner="a", generation=7)
  events.bind_producer("grid", owner="a", generation=7, producer_id=3)
  for _ in range(2):
    events.acquire_participant("grid", owner="a", generation=7)
  events.reset()  # MARK_EVENTS is not a participant's release.
  events.release_participant("grid", owner="a", generation=7, cycle=1)
  entry = events.get("grid")
  assert entry is not None and entry.status is EventStatus.RESET and entry.participant_refs == 1
  with pytest.raises(EventProtocolError):
    events.release_owner(owner="a", generation=7, cycle=2)
  assert not events.reserve_quota(1, {}, owner="b", generation=8)
  events.release_participant("grid", owner="a", generation=7, cycle=3)
  events.release_owner(owner="a", generation=7, cycle=3)
  assert events.reserve_quota(1, {}, owner="b", generation=8)


def test_group_uses_configured_drain_limit_without_claiming_transfer_closure() -> None:
  hw = HardwareConfig().with_overrides(max_drain_cycles=3, hbm_fixed_latency_cycles=300)
  sim = SimConfig(fidelity="full_memory")
  g = TileGroup(hw, fidelity=sim.fidelity)
  artifact = prepare_group_source(g, PowWorkload(num_group_chunks=1, hw=hw).module, POW_BINDINGS, sim=sim)
  g.load_task(artifact.entry, input_bindings=POW_BINDINGS)
  for cycle in range(500):
    g.step(cycle)
    if g.transfer_manager.snapshot()["hbm_outstanding"]:
      break
  else:
    pytest.fail("prefetch did not reach HBM")
  (txn,) = g._txn_sequencer
  g.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=cycle)
  observed = {}
  for tick in range(cycle + 1, cycle + 40):
    g.step(tick)
    n = g.reset_domain.drain_cycles
    if n:
      observed[n] = g.transfer_manager.status(txn)
      assert g.reset_domain.state is ResetState.DRAIN_SAFE
      if n == 4:
        break
  assert [observed[i] for i in (1, 2, 3)] == [TransferStatus.RUNNING] * 3
  assert observed[4] is TransferStatus.CANCEL_REQUESTED and g.transfer_manager.inflight_count == 1


def python_outcome(cancel: str | None) -> dict:
  """Independent single-process baseline. Block the same slot with real Arena/Frame capacity."""
  run = start()
  g = run.group
  accepted: set[tuple[int, int, int]] = set()
  if cancel == "before":
    from pipeline_validator.execution_ir import GridInstanceId
    from pipeline_validator.tile import TileAdmission

    root = g.sequencer.task
    assert root is not None
    (binding,) = root.role_bindings.values()
    tile = g.python_tiles_by_id[0]
    peer = tile.plan_admission(binding.tile_program, GridInstanceId("holder", 0, 0, 0), "peer", 0, 0)
    assert isinstance(peer, TileAdmission)
    tile.commit_admission(peer, binding.tile_program, run.cycle)
    # Occupy the other physical slot too; a real capacity rejection precedes the cancellation.
    peer2 = tile.plan_admission(binding.tile_program, GridInstanceId("holder", 0, 0, 1), "peer2", 0, 1)
    assert isinstance(peer2, TileAdmission)
    tile.commit_admission(peer2, binding.tile_program, run.cycle)
    run.until(lambda: any(r.wait_reasons for r in g._grid_routes.values()))
    assert not run.leases and not run.pins
    tile.abort_admission(peer, run.cycle)
    (route,) = g._grid_routes.values()
    task = route.expected[0]
    own = tile.plan_admission(binding.tile_program, route.grid, route.event_id, task.task_id, 0)
    assert isinstance(own, TileAdmission)
    lease = (route.request.binding_id, route.grid.launch_generation, 0)
    g._pin_grid_l2(route.grid, binding, task, run.cycle)
    g._task_leases[lease] = {task}
    tile.commit_admission(own, binding.tile_program, run.cycle)
    # Stall the local response at the same cut as the native CTRL link. The
    # prepared Frame and backing pin are real; the UCE is not runnable.
    for _ in range(8):
      run.tick()
      assert not tile.uce.has_active_contexts() and run.leases == run.pins == 1
    g.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=run.cycle)
    tile.abort_admission(own, run.cycle)
    g._unpin_task_l2(task, run.cycle)
    del g._task_leases[lease]
    tile.abort_admission(peer2, run.cycle)
  else:
    run.until(lambda: bool(run.leases))
    (route,) = g._grid_routes.values()
    accepted.update(
      (tid, route.expected[tid].grid.dispatch_ordinal, route.expected[tid].task_id)
      for tid in route.admissions
    )
    assert run.pins == 1
    if cancel == "after":
      g.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=run.cycle)
  run.finish(fault=cancel is not None)
  return {"accepted": sorted(accepted), "faulted": g.sequencer.faulted, "unclosed": g.unclosed_l2_objects()}


def admission_case(cancel: str | None, fail_hold: bool = False) -> dict:
  flags = abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_INPUT_RELEASED
  if fail_hold:
    flags |= abi.CN_STUB_FLAG_FAIL_HOLD
  p = ObservedNative(0, stub_flags=flags, ready_seed=5, channel_capacities={"CTRL": 2})
  try:
    r = start(ports={0: p})
    g = r.group
    r.until(lambda: bool(p.messages("TASK_COMMIT", sent=True)))
    hold = p.messages("TASK_COMMIT", sent=True)[0]
    (ticket,) = g.admission.tickets.values()
    assert r.leases == r.pins == 1 and not ticket.accepted
    event = g.event_table.get(ticket.route.event_id)
    assert event is not None and event.participant_refs == 1
    assert p.g2t_accepted_seq("CTRL") < hold.message_seq
    bad_update = BoundaryMessage(
      "RESOURCE_UPDATE",
      "EVENT",
      "T2G",
      {
        "tile_id": 0,
        "ctx_slot": ticket.ctx_slot,
        "ctx_generation": ticket.ctx_generation,
        "version": 500,
        "arena_alloc_gen": 37,
      },
    )
    with pytest.raises(MemoryInvariantError):
      g.admission.handle(0, bad_update, r.cycle)
    assert ticket.arena_alloc_gen is None
    for stage, ack_status in (("TASK_COMMIT_FINAL", "ACCEPTED"), ("TASK_COMMIT", "HOLD_READY")):
      premature = BoundaryMessage(
        "ACK",
        "CTRL",
        "T2G",
        {
          "txn_id": hold.fields["txn_id"],
          "ack_for_message_type": stage,
          "outcome": ack_status,
          "wait_reason": None,
          "op_seq": 0,
        },
      )
      with pytest.raises(MemoryInvariantError):
        g.admission.handle(0, BoundaryMessage.from_wire(premature.pack_wire()), r.cycle)
    assert not g.python_tiles_by_id.get(0) and not g._grid_routes[ticket.task.grid].admissions
    assert not p.messages("TASK_COMMIT_FINAL", sent=True)
    if cancel == "before":
      # ACK channel stalls independently of the genuine local allocation indication.
      p.set_t2g_ready("CTRL", False)
      r.until(lambda: bool(p.messages("RESOURCE_UPDATE")))
      assert p.messages("RESOURCE_UPDATE")[0].fields["arena_alloc_gen"] > 0
      assert ticket.arena_alloc_gen is None  # no HOLD_READY has crossed the stalled CTRL link
      g.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=r.cycle)
      for _ in range(8):
        r.tick()
      assert ticket in g.admission.tickets.values() and not p.messages("TASK_COMMIT_FINAL", sent=True)
      p.set_t2g_ready("CTRL", True)
      r.until(lambda: bool(p.messages("TICKET_ROLLBACK", sent=True)))
      assert ticket in g.admission.tickets.values(), "withdrawal is not a compensation receipt"
      assert p.messages("TICKET_ROLLBACK", sent=True)[0].fields["txn_id"] == hold.fields["txn_id"]
      r.finish(fault=True)
      assert event.participant_refs == 0
      compensated = [m for m in p.messages("ACK") if m.fields["ack_for_message_type"] == "TICKET_ROLLBACK"]
      assert len(compensated) == 1 and compensated[0].fields["outcome"] == "COMPENSATED"
      assert not p.messages("TASK_COMPLETION") and not p.messages("CANCEL_TASK", sent=True)
    elif cancel == "after":
      r.until(lambda: bool(p.messages("TASK_COMMIT_FINAL", sent=True)))
      (final,) = p.messages("TASK_COMMIT_FINAL", sent=True)
      assert final.fields["txn_id"] == hold.fields["txn_id"] and not ticket.accepted
      p.set_t2g_ready("CTRL", False)
      g.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=r.cycle)
      for _ in range(8):
        r.tick()
      assert r.leases == r.pins == 1 and ticket in g.admission.tickets.values()
      p.set_t2g_ready("CTRL", True)
      r.finish(fault=True)
      assert not p.messages("TICKET_ROLLBACK", sent=True) and len(p.messages("CANCEL_TASK", sent=True)) == 1
      (completion,) = p.messages("TASK_COMPLETION")
      assert completion.fields["status"] == "RESET"
    elif fail_hold:
      r.finish(fault=True)
      assert not p.messages("TASK_COMMIT_FINAL", sent=True) and not p.messages("TASK_COMPLETION")
      assert event.participant_refs == 0
      assert [(m.fields["ack_for_message_type"], m.fields["outcome"]) for m in p.messages("ACK")] == [
        ("TASK_PLAN", "ACCEPTED"),
        ("TASK_COMMIT", "FAULT"),
        ("TICKET_ROLLBACK", "COMPENSATED"),
      ]
    else:
      r.until(lambda: bool(p.messages("RETIRE", sent=True)))
      (retire,) = p.messages("RETIRE", sent=True)
      update = p.messages("RESOURCE_UPDATE")[-1]
      assert retire.fields["arena_alloc_gen"] == update.fields["arena_alloc_gen"]
      assert r.leases == 1 and g._l2_arenas and not g.sequencer.done
      p.set_t2g_ready("CTRL", False)
      for _ in range(12):
        r.tick()
      assert not p.messages("TASK_COMPLETION") and r.leases == 1
      live = g.event_table.get(ticket.route.event_id)
      assert live is not None and live.participant_refs == 1
      p.set_t2g_ready("CTRL", True)
      r.finish()
      (commit,) = p.messages("TASK_COMMIT_FINAL", sent=True)
      assert commit.fields["txn_id"] in {m.fields["txn_id"] for m in p.messages("TASK_COMMIT", sent=True)}
      assert sum(m.fields["outcome"] == "HOLD_READY" for m in p.messages("ACK")) == 1
      assert p.messages("TASK_COMPLETION")[0].fields["status"] == "DONE"
    finals = {m.fields["txn_id"]: m.fields for m in p.messages("TASK_COMMIT_FINAL", sent=True)}
    accepted = [
      (p.tile_id, finals[m.fields["txn_id"]]["dispatch_ordinal"], finals[m.fields["txn_id"]]["task_id"])
      for m in p.messages("ACK")
      if m.fields["ack_for_message_type"] == "TASK_COMMIT_FINAL" and m.fields["outcome"] == "ACCEPTED"
    ]
    outcome = {
      "accepted": sorted(accepted),
      "faulted": g.sequencer.faulted,
      "unclosed": g.unclosed_l2_objects(),
    }
    baseline = python_outcome("before" if fail_hold else cancel)
    assert outcome == baseline
    result = {"outcome": outcome, "reference": baseline, "wire": p.summary()}
    if fail_hold:
      # Reuse the same physical endpoint after the observed drain, not a freshly reset test double.
      artifact = g.loaded_program
      assert artifact is not None and isinstance(artifact.entry, ExecTileGroupTask)
      g.reset()
      r.cycle = g._last_step_cycle + 1
      g.begin_launch(artifact, {})
      g.load_task(artifact.entry, cycle=r.cycle)
      r.finish()
      (done,) = p.messages("TASK_COMPLETION")
      assert done.fields["status"] == "DONE"
      assert done.fields["launch_generation"] > ticket.key.launch_generation
      result["recovery"] = p.summary()
    return result
  finally:
    p.close()


def concurrency_case(leases: int) -> dict:
  p = ObservedNative(
    0, stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_INPUT_RELEASED, ready_seed=19
  )
  try:
    r = start(grids=2, leases=leases, ports={0: p}, pin=1 if leases == 1 else None)
    g = r.group
    r.until(lambda: bool(p.messages("TASK_COMMIT", sent=True)))
    p.set_t2g_ready("CTRL", False)
    for _ in range(30):
      r.tick()
    assert 1 <= r.leases <= leases and all(not route.retired for route in g._grid_routes.values())
    p.set_t2g_ready("CTRL", True)
    peak_r = r.leases
    wait_reasons: set[str] = set()
    saw_residual = False
    for _ in range(2000):
      r.tick()
      peak_r = max(peak_r, r.leases)
      wait_reasons.update(w for route in g._grid_routes.values() for w in route.wait_reasons.values())
      if len(p.messages("TASK_COMPLETION")) == 1 and g._grid_routes:
        saw_residual = True
        assert not g.sequencer.done and g._l2_arenas
      if g.sequencer.done:
        break
    r.finish()
    assert peak_r == leases and saw_residual
    finals = p.messages("TASK_COMMIT_FINAL", sent=True)
    if leases == 1:
      assert "WAIT_CONTEXT_LIMIT" in wait_reasons
      assert all(m.fields["ctx_slot"] == 1 for m in finals), "authored Context pin was ignored"
    completions = p.messages("TASK_COMPLETION")
    assert len(finals) == len(completions) == 2
    assert len({f.fields["dispatch_ordinal"] for f in finals}) == 2
    assert all(x.fields["status"] == "DONE" for x in completions)
    return {"max_reserved_r": peak_r, "wire": p.summary()}
  finally:
    p.close()


def failed_final_case(cancel_first: bool) -> dict:
  p = ObservedNative(0, stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_FAIL_FINAL)
  try:
    run = start(ports={0: p})
    g = run.group
    if cancel_first:
      run.until(lambda: bool(p.messages("TASK_COMMIT_FINAL", sent=True)))
      (pending,) = g.admission.tickets.values()
      g.admission.cancel(0, pending.key, run.cycle)
    run.until(
      lambda: any(m.fields["ack_for_message_type"] == "TASK_COMMIT_FINAL" for m in p.messages("ACK"))
    )
    (ticket,) = g.admission.tickets.values()
    assert not ticket.accepted and run.pins == run.leases == 1
    assert p.messages("ACK")[-1].fields["outcome"] == "FAULT"
    run.until(lambda: bool(p.messages("RETIRE", sent=True)))
    (grant,) = p.messages("RETIRE", sent=True)
    run.until(lambda: p.g2t_accepted_seq("CTRL") >= grant.message_seq)
    p.set_t2g_ready("CTRL", False)
    for _ in range(8):
      run.tick()
    assert not p.messages("PHASE_SIGNAL") and run.pins == run.leases == 1
    forged = BoundaryMessage(
      "TASK_COMPLETION",
      "CTRL",
      "T2G",
      {**ticket.key.to_wire_fields(), "status": "DONE", "first_fault_ref": 0},
    )
    assert g.retirement.handle(0, forged, run.cycle) == "invalid"
    assert run.leases == run.pins == 1 and not ticket.accepted
    p.set_t2g_ready("CTRL", True)
    run.finish(fault=True)
    (terminal,) = p.messages("TASK_COMPLETION")
    assert terminal.fields["status"] == "ERROR" and terminal.fields["first_fault_ref"] != 0
    assert not p.messages("TICKET_ROLLBACK", sent=True) and len(p.messages("CANCEL_TASK", sent=True)) == 1
    return p.summary()
  finally:
    p.close()


def native_owner_case() -> dict:
  """Old transaction/slot receipts may not alter an actual held/authorized slot."""
  p = ObservedNative(
    0, stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_FAIL_HOLD | abi.CN_STUB_FLAG_INPUT_RELEASED
  )
  key = {
    "launch_generation": 5,
    "root_launch_generation": 5,
    "context_handle": 0,
    "task_id": 7,
    "dispatch_ordinal": 2,
  }
  owner = {**key, "txn_id": 11, "ctx_slot": 1, "ctx_generation": 9}
  cycle = 0

  def tick() -> None:
    nonlocal cycle
    p.step(cycle)
    cycle += 1
    while p.pop_inbound() is not None:
      pass

  def request(kind: str, fields: dict, replies: str = "ACK") -> BoundaryMessage:
    before = len(p.messages(replies))
    p.send(BoundaryMessage(kind, "CTRL", "G2T", fields))
    for _ in range(80):
      tick()
      if len(p.messages(replies)) != before:
        return p.messages(replies)[before]
    raise AssertionError(f"no {replies} for {kind}")

  try:
    p.send(BoundaryMessage("RUN_BEGIN", "CTRL", "G2T", {"run_generation": 5}))
    plan = {**key, "txn_id": owner["txn_id"], "pool_epoch": 0, "pool_version": 0}
    assert request("TASK_PLAN", plan).fields["outcome"] == "ACCEPTED"
    assert request("TASK_COMMIT", {**owner, "pool_version": 0}).fields["outcome"] == "FAULT"
    # Replaying a partially failed HOLD, even with a fresh transaction number, cannot recycle it.
    for txn in (11, 12):
      assert request("TASK_COMMIT", {**owner, "txn_id": txn, "pool_version": 0}).fields["outcome"] in (
        "FAULT",
        "STALE",
      )
    assert request("TICKET_ROLLBACK", owner).fields["outcome"] == "COMPENSATED"
    owner = {**owner, "txn_id": 13, "ctx_generation": 10}
    assert request("TASK_PLAN", {**plan, "txn_id": 13}).fields["outcome"] == "ACCEPTED"
    assert request("TASK_COMMIT", {**owner, "pool_version": 0}).fields["outcome"] == "STALE"
    for _ in range(12):
      tick()
    (stale_report,) = p.messages("RESOURCE_UPDATE")
    version = stale_report.fields["version"]
    # The failed reservation and its real release both changed the pool. The
    # stale report retains the OLD allocation generation; no second hold exists yet.
    assert version == 2 and stale_report.fields["arena_alloc_gen"] == 1
    assert request("TICKET_ROLLBACK", owner).fields["outcome"] == "COMPENSATED"
    owner = {**owner, "txn_id": 14}
    assert (
      request("TASK_PLAN", {**plan, "txn_id": 14, "pool_version": version}).fields["outcome"] == "ACCEPTED"
    )
    assert request("TASK_COMMIT", {**owner, "pool_version": version}).fields["outcome"] == "HOLD_READY"
    for kind in ("TICKET_ROLLBACK", "TASK_COMMIT_FINAL"):
      for changes in ({"ctx_slot": 0}, {"ctx_slot": 65}, {"ctx_generation": 8}, {"txn_id": 12}):
        assert request(kind, {**owner, **changes}).fields["outcome"] in ("FAULT", "STALE")
    assert not p.messages("RETIRE_READY") and not p.messages("TASK_COMPLETION")
    for _ in range(12):
      tick()
    stale, update = p.messages("RESOURCE_UPDATE")
    assert stale == stale_report and update.fields["arena_alloc_gen"] != stale.fields["arena_alloc_gen"]
    assert update.fields["ctx_slot"] == 1 and update.fields["ctx_generation"] == 10
    grant = {
      **key,
      "tile_id": 0,
      "ctx_slot": 1,
      "ctx_generation": 10,
      "arena_alloc_gen": update.fields["arena_alloc_gen"],
    }
    p.set_t2g_ready("EVENT", False)
    assert request("TASK_COMMIT_FINAL", owner).fields["outcome"] == "ACCEPTED"
    assert request("TICKET_ROLLBACK", owner).fields["outcome"] in ("FAULT", "STALE")
    p.send(BoundaryMessage("RETIRE", "CTRL", "G2T", grant))
    for _ in range(16):
      tick()
    assert not p.messages("RETIRE_READY") and not p.messages("TASK_COMPLETION")
    p.set_t2g_ready("EVENT", True)
    for _ in range(40):
      tick()
    assert len(p.messages("RETIRE_READY")) == 1 and not p.messages("TASK_COMPLETION")
    p.send(BoundaryMessage("CANCEL_TASK", "CTRL", "G2T", {**key, "ctx_slot": 65, "ctx_generation": 10}))
    p.send(BoundaryMessage("RETIRE", "CTRL", "G2T", {**grant, "ctx_slot": 65}))
    p.send(
      BoundaryMessage("RETIRE", "CTRL", "G2T", {**grant, "arena_alloc_gen": grant["arena_alloc_gen"] + 1})
    )
    for _ in range(16):
      tick()
    assert not p.messages("TASK_COMPLETION")
    assert request("RETIRE", grant, "TASK_COMPLETION").fields["status"] == "DONE"
    return p.summary()
  finally:
    p.close()


def withdrawn_hold_stalled_report() -> dict:
  p = ObservedNative(0, stub_flags=abi.CN_STUB_FLAG_LIFECYCLE)
  try:
    run = start(ports={0: p})
    g = run.group
    run.until(lambda: bool(p.messages("TASK_COMMIT", sent=True)))
    p.set_t2g_ready("EVENT", False)
    g.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=run.cycle)
    run.until(
      lambda: (
        g.reset_domain.state is ResetState.DRAIN_SAFE
        and any(m.fields["outcome"] == "COMPENSATED" for m in p.messages("ACK"))
      )
    )
    assert not g.admission.tickets and not run.leases and not run.pins
    for _ in range(20):
      run.tick()
    assert g.reset_domain.state is ResetState.DRAIN_SAFE and not p.messages("RESOURCE_UPDATE")
    assert not any(m.fields["phase"] == "DRAIN" for m in p.messages("RESET_PHASE_ACK"))
    p.set_t2g_ready("EVENT", True)
    run.finish(fault=True)
    (update,) = p.messages("RESOURCE_UPDATE")
    (drained,) = [m for m in p.messages("RESET_PHASE_ACK") if m.fields["phase"] == "DRAIN"]
    assert p.received.index(update) < p.received.index(drained) and drained.fields["evidence_mask"] == 0x3F
    return p.summary()
  finally:
    p.close()


def full_ctrl_free_slot_case() -> dict:
  # RUN_BEGIN occupies the link's single credit. The seed keeps the CTRL
  # receiver closed until after the Group registers this Grid; no Task owns a slot.
  p = ObservedNative(
    0,
    ready_seed=0x10,
    channel_capacities={"CTRL": 1},
    stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_INPUT_RELEASED,
  )
  try:
    run = start(ports={0: p}, pin=1)
    g = run.group
    run.until(lambda: bool(g._grid_routes))
    (route,) = g._grid_routes.values()
    assert route.binding.context_id == 1
    assert not g.admission.tickets and not p.outbox_ready("CTRL") and p.g2t_accepted_seq("CTRL") == -1
    assert not run.leases and not run.pins and not p.messages("TASK_PLAN", sent=True)
    reason = route.wait_reasons[0]
    assert reason == "WAIT_CONTROL_RESOURCE", f"free Context 1, full CTRL: {reason}"
    run.finish()
    (hold,) = p.messages("TASK_COMMIT", sent=True)
    assert hold.fields["ctx_slot"] == 1
    assert [m.fields["status"] for m in p.messages("TASK_COMPLETION")] == ["DONE"]
    return {"blocked_reason": reason, "wire": p.summary()}
  finally:
    p.close()


def pool_version_drift(event_first: bool) -> dict:
  """Two actual admissions plan the same pool. The first hold changes the second plan's version."""
  p = ObservedNative(
    0, ready_seed=5, stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_INPUT_RELEASED
  )
  try:
    p.set_t2g_ready("EVENT", False)
    r = start(grids=2, leases=2, pin=(0, 1), ports={0: p})
    g = r.group
    r.until(lambda: len(p.messages("TASK_COMMIT", sent=True)) == 2)
    first, second = p.messages("TASK_COMMIT", sent=True)
    assert first.fields["pool_version"] == second.fields["pool_version"]
    old = next(t for t in g.admission.tickets.values() if t.txn_id == second.fields["txn_id"])
    event = g.event_table.get(old.route.event_id)
    assert event is not None and event.participant_refs == 1 and r.pins == r.leases == 2
    r.until(lambda: p.g2t_accepted_seq("CTRL") >= second.message_seq)

    def replies(kind: str) -> list[BoundaryMessage]:
      return [
        m
        for m in p.messages("ACK")
        if m.fields["txn_id"] == old.txn_id and m.fields["ack_for_message_type"] == kind
      ]

    def version_reports() -> list[BoundaryMessage]:
      return [
        m
        for m in p.messages("RESOURCE_UPDATE")
        if m.fields["ctx_slot"] == old.ctx_slot and m.fields["ctx_generation"] == old.ctx_generation
      ]

    if event_first:
      p.set_t2g_ready("CTRL", False)
      p.set_t2g_ready("EVENT", True)
      r.until(lambda: bool(version_reports()))
      assert not replies("TASK_COMMIT") and old.arena_alloc_gen is None
      p.set_t2g_ready("CTRL", True)
    r.until(lambda: bool(replies("TASK_COMMIT")))
    outcome = replies("TASK_COMMIT")[0].fields["outcome"]
    assert outcome == "STALE", f"competing real HOLD did not invalidate the old pool version: {outcome}"
    assert not old.accepted and old.arena_alloc_gen is None and not g.sequencer.faulted
    assert old.task not in g._task_leases.get(
      (old.route.request.binding_id, old.key.launch_generation, 0), ()
    )
    assert old.task.task_id not in g._grid_l2_pins.get(old.task.grid, {})
    assert event.participant_refs == 0
    r.until(lambda: any(m.fields["outcome"] == "COMPENSATED" for m in replies("TICKET_ROLLBACK")))
    if not event_first:
      for _ in range(12):
        r.tick()
      assert not version_reports() and len(p.messages("TASK_PLAN", sent=True)) == 2
      assert g.admission.tickets[0, old.key] is old and r.pins == r.leases == 1
      p.set_t2g_ready("EVENT", True)
    r.until(lambda: bool(version_reports()))
    reported = version_reports()[0].fields["version"]
    assert reported > old.pool_version
    r.until(lambda: len(p.messages("TASK_COMMIT", sent=True)) >= 3)
    retry_hold = p.messages("TASK_COMMIT", sent=True)[2]
    r.until(lambda: p.g2t_accepted_seq("CTRL") >= retry_hold.message_seq)
    current = g.admission.tickets[0, old.key]
    assert current.txn_id == retry_hold.fields["txn_id"] and current.pool_report is None
    # The old, valid report has the SAME slot, generation and version as the new PLAN.
    # Replaying it must not install the stale Arena or consume the new ticket's report.
    g.admission.handle(0, version_reports()[0], r.cycle)
    assert current.pool_report is None and current.arena_alloc_gen is None
    r.finish()
    attempts = [
      m
      for m in p.messages("TASK_PLAN", sent=True)
      if m.fields["dispatch_ordinal"] == old.key.dispatch_ordinal
    ]
    assert len(attempts) >= 2 and attempts[1].fields["pool_version"] == reported
    assert attempts[1].fields["txn_id"] != old.txn_id
    retry = next(
      m for m in p.messages("TASK_COMMIT", sent=True) if m.fields["txn_id"] == attempts[1].fields["txn_id"]
    )
    assert retry.fields["ctx_slot"] == old.ctx_slot and retry.fields["ctx_generation"] == old.ctx_generation
    assert not p.messages("CANCEL_TASK", sent=True) and not p.messages("RESET_PHASE_REQ", sent=True)
    finals = p.messages("TASK_COMMIT_FINAL", sent=True)
    assert len(finals) == 2 and all(m.fields["txn_id"] != old.txn_id for m in finals)
    assert [(m.fields["status"]) for m in p.messages("TASK_COMPLETION")] == ["DONE", "DONE"]
    return {"rejected_version": old.pool_version, "reported_version": reported, "wire": p.summary()}
  finally:
    p.close()


def test_python_and_native_admission_under_backpressure_and_cancellation(tmp_path: Path) -> None:
  results = native_cases("test_w1_admission", tmp_path)
  assert results["before"]["outcome"]["accepted"] == []
  assert results["after"]["outcome"]["accepted"] == results["normal"]["outcome"]["accepted"] == [[0, 0, 0]]
  assert results["failed_hold"]["outcome"] == results["before"]["outcome"]
  assert [results[k]["max_reserved_r"] for k in ("r1", "r2")] == [1, 2]


if __name__ == "__main__":
  out = Path(sys.argv[1])
  results = {
    k: admission_case(c, fail)
    for k, c, fail in (
      ("normal", None, False),
      ("before", "before", False),
      ("after", "after", False),
      ("failed_hold", None, True),
    )
  }
  results["failed_final"] = failed_final_case(False)
  results["cancel_before_final_fault_ack"] = failed_final_case(True)
  results["owner_identity"] = native_owner_case()
  results["withdrawn_hold_stalled_report"] = withdrawn_hold_stalled_report()
  results["full_ctrl_free_slot"] = full_ctrl_free_slot_case()
  results["pool_event_first"] = pool_version_drift(True)
  results["pool_ack_first"] = pool_version_drift(False)
  results.update({f"r{n}": concurrency_case(n) for n in (1, 2)})
  (out / "c035.json").write_text(json.dumps(results, indent=2) + "\n")
