"""C036 / W1-04: event-driven L1/L2 gate wake for Route registration, Task admission and GATE_NOTIFY.

Verifies:
1. Positive Route wake: L1 gate closed -> ``dispatch_role`` returns BACKPRESSURE and parks the
   action out of the scheduler scan queue with ``wait_gate_epoch``; ``OPEN_ISSUE`` wakes the
   parked action in-step and registers the Grid Route on the next issue step with zero polling
   in between.
2. Negative zero-poll (Route + Task): L1 gate held closed with no ``OPEN_ISSUE`` across repeated
   ticks -> zero retry probes on ``dispatch_role``, zero candidate scans on ``AdmissionCoordinator``,
   and zero ``TASK_PLAN`` / ``TASK_COMMIT`` on an attached boundary port.
3. Positive Task admission wake: an already-registered Route blocked while L1 gate is closed
   marks ``WAIT_CONTROL_RESOURCE`` once, suppresses scans while closed, and wakes on ``OPEN_ISSUE``
   to admit and complete the Task (both Python and boundary/native paths).
4. Coalescing: multiple gate state transitions before the next scheduler/admission step coalesce
   into a single retry round (not 1:1 with gate flips) and coalesce pending ``GATE_NOTIFY`` under
   CTRL backpressure.
5. Unified ``GroupPortAdapter`` wake token: ``(admission_version, _slot_version, gate_epoch)``
   has a single uniform tuple shape for pure-Python and boundary runs, includes L1 gate closure
   in ``admission_version`` and advances on gate transitions without polling while unchanged.
6. Non-gate backpressure invariance (C022/C035 parity): ``dispatch_capacity`` backpressure and
   ``s0`` ordering semantics remain untouched and never park on a gate epoch.
7. Real Verilated C ABI boundary ``GATE_NOTIFY`` transmission: G2T ``GATE_NOTIFY`` frames carry
   exact C017 wire fields (``level``, ``state``, ``profile_generation``) over CTRL without
   violating link credit invariants or leaking into lifecycle admission state.
"""

from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from pathlib import Path
import pytest

from dataclasses import replace

from pipeline_validator.config import GroupSchedulerConfig, HardwareConfig, SimConfig
from pipeline_validator.cosim import BoundaryMessage, ChannelCreditLedger, TileProtocolStub, abi, wire
from pipeline_validator.device import DeviceLaunchRequest
from pipeline_validator.execution_ir import ExecTileGroupTask
from pipeline_validator.profiles import ProfileReconfigDesc, SourceRef
from pipeline_validator.runtime.group_port import GroupPortAdapter
from pipeline_validator.tests.test_runtime import POW_BINDINGS, prepare_group_source
from pipeline_validator.tests.w1_lifecycle import ObservedNative, native_cases, reader_module, start
from pipeline_validator.tile_group import TileGroup
from pipeline_validator.workloads import PowWorkload


def _profile_desc(group: TileGroup, command_id: str, level: str, target_mode: int) -> ProfileReconfigDesc:
  current = group.profile_controller.active_modes[level]
  profile = group.registry.profile(level, target_mode)
  return ProfileReconfigDesc(
    command_id=command_id,
    level=level,
    expected_mode=current,
    target_mode=target_mode,
    registry_hash=group.registry.registry_hash,
    wait_instruction_ids=(),
    frontier=(),
    affected_domains=(level,),
    member_ids=profile.member_ids,
    exclusive_binding_id="",
    source_ref=SourceRef("<test>", "c036", 0, "profile.reconfig"),
  )


def test_route_parks_when_l1_gate_closed_and_wakes_exactly_once_on_open_issue() -> None:
  """Positive + negative zero-poll proof on the real ProfileController CLOSE_ISSUE -> OPEN_ISSUE path."""
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  # Start from reset_mode=1 so a single real 12-step L1 ProfileReconfigDesc (1 -> 0)
  # drives CLOSE_ISSUE -> publish(mode=0) -> OPEN_ISSUE for the compiled l1_mode=0 task.
  hw = replace(hw, memory_target=replace(hw.memory_target, l1=replace(hw.memory_target.l1, reset_mode=1)))
  sim = SimConfig(fidelity="runtime", context_count=2)
  # Compile against reset_mode=0 (matches l1_mode=0 after the 1 -> 0 profile switch),
  # while running the Group from reset_mode=1 so real CLOSE_ISSUE -> publish -> OPEN_ISSUE runs.
  compile_hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  group = TileGroup(hw, fidelity=sim.fidelity, context_count=sim.context_count)
  from pipeline_validator.tests.test_runtime import begin_group_program, compile_source

  artifact = compile_source(reader_module(1, 1, 1, False, None), compile_hw, sim)
  begin_group_program(group, artifact)
  assert isinstance(artifact.entry, ExecTileGroupTask)
  assert group.profile_controller.active_modes["l1"] == 1
  cycle = group._last_step_cycle + 1
  cmd_to_0 = _profile_desc(group, "l1-to-0", "l1", 0)
  group.profile_controller.queue_member_response(
    "l1-to-0", cmd_to_0.member_ids[-1], "COMMIT", delay_cycles=20
  )
  assert group.profile_controller.submit(cmd_to_0, f"device@{group.run_generation}", cycle)
  for _ in range(2000):
    group.step(cycle)
    cycle += 1
    status = group.profile_controller.status("l1-to-0")
    assert status not in ("faulted", "cancelled"), group.profile_controller.command_reason("l1-to-0")
    commands = group.profile_controller.snapshot()["commands"]
    assert isinstance(commands, dict)
    if commands["l1-to-0"]["step"] == "OPEN_ISSUE":
      break
  else:
    pytest.fail("l1-to-0 did not reach OPEN_ISSUE")
  # Right after _publish_profile(), active_modes["l1"] == 0, generation == 1, gate STILL closed.
  assert group.profile_controller.active_modes["l1"] == 0
  assert group.profile_controller.generations["l1"] == 1
  assert group.profile_controller.issue_gate_closed("l1")
  # Load the root Context now while L1 gate is closed.
  group.load_task(artifact.entry, cycle=cycle)

  dispatch_probes = 0
  orig_dispatch = group.dispatch_role

  def spy_dispatch(*args, **kwargs):
    nonlocal dispatch_probes
    dispatch_probes += 1
    return orig_dispatch(*args, **kwargs)

  group.dispatch_role = spy_dispatch  # type: ignore[method-assign]

  # Step scheduler + admission WITHOUT stepping profile_controller so OPEN_ISSUE does not run yet:
  # 1st tick registers DISPATCH_ROLE, 2nd tick issues -> BACKPRESSURE -> parks with wait_gate_epoch.
  # Following 30 ticks with gate permanently closed must produce ZERO extra dispatch_role probes
  # and ZERO admission candidate scans.
  scans_before = group.admission.stats["candidate_scans"]
  suppressed_before = group.admission.stats["gate_suppressed_ticks"]
  for _ in range(32):
    group.scheduler.step(group._active_sequencers, cycle)
    group.admission.step(cycle)
    cycle += 1

  assert dispatch_probes == 1, f"expected exactly 1 probe before gate park, got {dispatch_probes}"
  assert group.scheduler.gate_parked_total == 1
  assert group.scheduler.gate_woken_total == 0
  assert len(group.scheduler._gate_parked) == 1
  assert group.scheduler._gate_parked[0].wait_gate_level == "l1"
  assert group.scheduler._gate_parked[0].wait_gate_epoch == group.gate_wake.epoch("l1")
  assert not group._grid_routes
  assert group.admission.stats["candidate_scans"] == scans_before
  assert group.admission.stats["gate_park_events"] == 1
  assert group.admission.stats["gate_suppressed_ticks"] == suppressed_before + 32

  # Now advance ONE real TileGroup.step(): ProfileController executes OPEN_ISSUE -> gate_wake.register
  # pushes the parked record back into _queued -> scheduler issues dispatch_role ONCE and succeeds.
  group.step(cycle)
  cycle += 1
  assert not group.profile_controller.issue_gate_closed("l1")
  assert dispatch_probes == 2, f"OPEN_ISSUE must trigger exactly one retry probe, got {dispatch_probes}"
  assert group.scheduler.gate_woken_total == 1
  assert len(group.scheduler._gate_parked) == 0
  assert len(group._grid_routes) == 1
  assert group.gate_wake.stats["scheduler_wakes"] == 1
  assert group.gate_wake.stats["admission_wakes"] == 1

  # Drain to natural completion and verify zero residuals.
  for _ in range(400):
    if group.step(cycle):
      break
    cycle += 1
  assert group.sequencer.done and not group.sequencer.faulted
  assert not group._grid_routes and not group._task_leases
  assert group.unclosed_l2_objects() == []


def test_task_admission_parks_zero_poll_when_l1_gate_closed_and_wakes_on_open() -> None:
  """An already-registered Grid Route waits with WAIT_CONTROL_RESOURCE, zero scans while closed,
  and admits on the exact tick OPEN_ISSUE flips the L1 gate."""
  run = start(participants=1, grids=1, leases=1, reads=False)
  g = run.group

  # Advance until the Grid Route is registered, BEFORE admission commits the Task:
  # register on tick 0, issue dispatch_role on tick 1; close L1 gate immediately after Route exists.
  g.scheduler.step(g._active_sequencers, 0)
  g.scheduler.step(g._active_sequencers, 1)
  assert len(g._grid_routes) == 1
  (route,) = g._grid_routes.values()
  assert not route.admissions

  # Authoritatively close L1 gate through ProfileController.
  g.profile_controller._set_profile_gate("l1", True, cycle=2)
  assert g.profile_controller.issue_gate_closed("l1")

  # Repeated ticks with L1 gate held closed: zero candidate scans, zero Python plan_admission calls.
  plan_probes = 0
  tile0 = g.python_tiles_by_id[0]
  orig_plan = tile0.plan_admission

  def spy_plan(*args, **kwargs):
    nonlocal plan_probes
    plan_probes += 1
    return orig_plan(*args, **kwargs)

  tile0.plan_admission = spy_plan  # type: ignore[method-assign]

  # Count every controller read from the moment the latch parks: step() must make none.
  controller_reads = 0
  orig_closed = g.profile_controller.issue_gate_closed

  def spy_closed(level):
    nonlocal controller_reads
    controller_reads += 1
    return orig_closed(level)

  g.profile_controller.issue_gate_closed = spy_closed  # type: ignore[method-assign]
  for cycle in range(2, 27):
    g.admission.step(cycle)

  assert controller_reads == 0, f"step() polled the controller {controller_reads} times while parked"
  assert plan_probes == 0
  assert g.admission.stats["candidate_scans"] == 0
  assert g.admission.stats["gate_park_events"] == 1
  assert g.admission.stats["gate_suppressed_ticks"] == 24
  assert route.wait_reasons[0] == "WAIT_CONTROL_RESOURCE"
  assert not route.admissions and run.leases == 0

  # Open the gate -> coalesced wake -> next admission.step scans once and commits the Task.
  g.profile_controller._set_profile_gate("l1", False, cycle=27)
  assert g.admission.stats["gate_wake_events"] == 1
  g.admission.step(27)
  assert plan_probes == 1
  assert g.admission.stats["candidate_scans"] == 1
  assert 0 in route.admissions and 0 not in route.wait_reasons

  run.cycle = 28
  run.finish()


def test_l1_transition_latch_alone_drives_admission_step() -> None:
  """Reopening resumes admission from the transition latch even with the wake callback removed,
  and the controller is never read by step()."""
  run = start(participants=1, grids=1, leases=1, reads=False)
  g = run.group
  g.scheduler.step(g._active_sequencers, 0)
  g.scheduler.step(g._active_sequencers, 1)
  (route,) = g._grid_routes.values()
  g.admission.wake_on_l1_gate = lambda *a, **k: False  # type: ignore[method-assign]
  g.profile_controller._set_profile_gate("l1", True, cycle=2)
  g.admission.step(2)
  assert g.admission._l1_closed and not route.admissions

  reads = 0
  orig = g.profile_controller.issue_gate_closed

  def spy(level):
    nonlocal reads
    reads += 1
    return orig(level)

  g.profile_controller.issue_gate_closed = spy  # type: ignore[method-assign]
  g.profile_controller._set_profile_gate("l1", False, cycle=3)
  reads_after_register = reads
  g.admission.step(3)
  assert reads == reads_after_register, "step() read the controller after the latch flipped"
  assert not g.admission._l1_closed
  assert 0 in route.admissions
  run.cycle = 4
  run.finish()


def test_rapid_gate_flips_coalesce_to_one_retry_and_coalesce_gate_notify_under_backpressure() -> None:
  """Multiple gate flips before the next issue step coalesce to one retry round and one CTRL notify."""
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  # Unit-only TileProtocolStub with CTRL capacity=1 so we can observe GATE_NOTIFY coalescing.
  port0 = TileProtocolStub(tile_id=0, context_count=2)
  port0.credits = ChannelCreditLedger({"CTRL": 1})
  sent_notifies: list[tuple[str, str, int]] = []
  orig_send = port0.send

  def spy_send(msg: BoundaryMessage) -> bytes:
    blob = orig_send(msg)
    if msg.message_type == "GATE_NOTIFY":
      decoded = BoundaryMessage.from_wire(blob)
      sent_notifies.append(
        (
          str(decoded.fields["level"]),
          str(decoded.fields["state"]),
          int(decoded.fields["profile_generation"]),
        )
      )
    return blob

  port0.send = spy_send  # type: ignore[method-assign]

  sim = SimConfig(fidelity="runtime", context_count=2)
  group = TileGroup(hw, fidelity=sim.fidelity, context_count=sim.context_count, boundary_ports={0: port0})
  artifact = prepare_group_source(group, reader_module(1, 1, 1, False, None), sim=sim)
  assert isinstance(artifact.entry, ExecTileGroupTask)
  # Drain RUN_BEGIN from begin_launch so port0 CTRL credit starts full.
  port0.step(0)
  assert port0.outbox_ready("CTRL")
  group.load_task(artifact.entry, cycle=1)

  # Close L1 gate and let DISPATCH_ROLE park once.
  group.profile_controller._set_profile_gate("l1", True, cycle=1)
  port0.step(1)
  assert sent_notifies[-1] == ("l1", "closed", 0)

  group.scheduler.step(group._active_sequencers, 2)
  group.scheduler.step(group._active_sequencers, 3)
  assert len(group.scheduler._gate_parked) == 1

  # Occupy CTRL credit with an unrelated QUERY_CAPS so GATE_NOTIFY queues and coalesces.
  port0.send(BoundaryMessage("QUERY_CAPS", "CTRL", "G2T", {"txn_id": 1}))
  assert not port0.outbox_ready("CTRL")

  dispatch_probes = 0
  orig_dispatch = group.dispatch_role

  def spy_dispatch(*args, **kwargs):
    nonlocal dispatch_probes
    dispatch_probes += 1
    return orig_dispatch(*args, **kwargs)

  group.dispatch_role = spy_dispatch  # type: ignore[method-assign]

  # Flip L1 gate open -> closed -> open BEFORE scheduler steps:
  # - Parked Route was woken once on first open and stays queued (no duplicate wake).
  # - Pending GATE_NOTIFY entries coalesce in place instead of emitting 3 CTRL frames.
  group.profile_controller._set_profile_gate("l1", False, cycle=4)
  group.profile_controller._set_profile_gate("l1", True, cycle=4)
  group.profile_controller._set_profile_gate("l1", False, cycle=4)

  assert group.scheduler.gate_woken_total == 1
  assert group.gate_wake.stats["gate_notifies_coalesced"] == 2
  assert group.gate_wake.snapshot()["pending_notifies"] == 1

  # Step scheduler once: only ONE retry probe runs (coalesced across the flips).
  group.scheduler.step(group._active_sequencers, 5)
  assert dispatch_probes == 1
  assert len(group._grid_routes) == 1

  # Free CTRL credit by stepping port0 (consumes QUERY_CAPS), then flush GATE_NOTIFY and step again.
  port0.step(5)
  assert port0.outbox_ready("CTRL")
  sent = group.gate_wake.flush(6)
  assert sent == 1
  port0.step(6)
  assert sent_notifies == [("l1", "closed", 0), ("l1", "open", 0)]
  assert port0.credit_invariant_holds()


def test_group_port_wake_token_has_unified_shape_and_zero_polls_when_gate_unchanged() -> None:
  """GroupPortAdapter token always has unified 3-tuple shape (admission_version, slot_version, gate_epoch)
  and admission_version includes L1 gate state without dual cosim/non-cosim branches."""
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  sim = SimConfig(fidelity="runtime", context_count=2)
  workload = PowWorkload(num_group_chunks=1, hw=hw)
  group = TileGroup(hw, fidelity=sim.fidelity, context_count=sim.context_count)
  artifact = prepare_group_source(group, workload.module, POW_BINDINGS, sim=sim)
  assert isinstance(artifact.entry, ExecTileGroupTask)
  port = GroupPortAdapter(group, active_context_capacity=1)

  # Close L2 gate so pending root blocks cleanly.
  group.profile_controller._set_profile_gate("l2", True, cycle=0)
  req = DeviceLaunchRequest(
    request_id=1,
    context_name=artifact.entry.name,
    task=artifact.entry,
    global_bindings=POW_BINDINGS,
    formal_bindings={item.name: item.name for item in artifact.entry.global_inputs},
  )
  assert port.try_submit(req, cycle=0)
  assert port.poll_completions(cycle=0) == ()

  token = port._last_admission_token
  assert token is not None and len(token) == 3
  adm_ver, slot_ver, gate_epoch = token
  assert isinstance(adm_ver, tuple) and len(adm_ver) == 7
  assert adm_ver[-2:] == (True, False)  # (l2_closed, l1_closed)
  assert slot_ver == 0
  assert gate_epoch == group.gate_wake.gate_epoch

  # Repeated poll_completions while nothing changed must NOT retry.
  record = port._pending[1]
  retries_before = record.retry_count
  for c in range(1, 20):
    assert port.poll_completions(cycle=c) == ()
  assert record.retry_count == retries_before

  # Flip L1 gate (even with L2 still closed): token advances via both admission_version and gate_epoch.
  group.profile_controller._set_profile_gate("l1", True, cycle=20)
  assert port.poll_completions(cycle=20) == ()
  assert record.retry_count == retries_before + 1
  updated = port._last_admission_token
  assert updated is not None and isinstance(updated[0], tuple)
  assert updated[2] == group.gate_wake.gate_epoch
  assert updated[0][-1] is True
  # Open L2 gate -> single wake admits the root cleanly.
  group.profile_controller._set_profile_gate("l1", False, cycle=21)
  group.profile_controller._set_profile_gate("l2", False, cycle=21)
  assert port.poll_completions(cycle=21) == ()
  assert port.active_count == 1 and port.pending_count == 0


def test_non_gate_dispatch_capacity_backpressure_never_parks_on_gate() -> None:
  """Non-gate BACKPRESSURE (dispatch_capacity full) preserves existing scheduler order and never parks."""
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  sched_cfg = GroupSchedulerConfig(policy="s0", dispatch_capacity=2)
  sim = SimConfig(fidelity="runtime", context_count=2, group=sched_cfg)
  group = TileGroup(hw, fidelity=sim.fidelity, context_count=sim.context_count, scheduler_config=sched_cfg)
  artifact = prepare_group_source(group, reader_module(1, 2, 2, False, None), sim=sim)
  assert isinstance(artifact.entry, ExecTileGroupTask)
  # Narrow runtime dispatch_capacity to 1 after compilation so grid1 legitimately hits
  # non-gate dispatch_capacity backpressure while grid0 is in flight.
  group.scheduler_config = replace(group.scheduler_config, dispatch_capacity=1)
  group.scheduler.config = group.scheduler_config
  group.load_task(artifact.entry, cycle=0)

  saw_backpressure = False
  for cycle in range(500):
    if group.step(cycle):
      break
    if group.scheduler.backpressure_total > 0:
      saw_backpressure = True
    assert group.scheduler.gate_parked_total == 0
    assert len(group.scheduler._gate_parked) == 0

  assert saw_backpressure
  assert group.sequencer.done and not group.sequencer.faulted
  assert group.unclosed_l2_objects() == []


def test_gate_wake_rejects_unknown_level_and_negative_generation() -> None:
  hw = HardwareConfig()
  group = TileGroup(hw, fidelity="runtime")
  with pytest.raises(ValueError, match="unknown gate level"):
    group.gate_wake.epoch("l3")
  with pytest.raises(ValueError, match="unknown gate level"):
    group.gate_wake.register("l3")
  with pytest.raises(ValueError, match="profile_generation"):
    group.gate_wake.register("l1", closed=True, profile_generation=-1)
  # No-op register (same closed state and generation) returns False and emits nothing.
  assert not group.gate_wake.register("l1", closed=True, profile_generation=0)


def test_gate_parked_route_cancels_cleanly_on_fault() -> None:
  """Cancelling/faulting a Context while its DISPATCH_ROLE is gate-parked cleans all scheduler & L2 state."""
  from pipeline_validator.runtime.fault_ring import FaultCode

  run = start(participants=1, grids=1, leases=1, reads=True)
  g = run.group
  g.profile_controller._set_profile_gate("l1", True, cycle=0)
  for _ in range(5):
    run.tick()
  assert len(g.scheduler._gate_parked) == 1
  assert g.scheduler.queued >= 1

  g.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=run.cycle, desc_id="c036-parked-fault")
  run.finish(fault=True)
  assert len(g.scheduler._gate_parked) == 0
  assert g.scheduler.queued == 0


def test_l2_gate_open_does_not_wake_l1_parked_route() -> None:
  """Gate wakeups are level-scoped: L2 OPEN_ISSUE must not wake an L1-parked Route."""
  run = start(participants=1, grids=1, leases=1, reads=False)
  g = run.group
  g.profile_controller._set_profile_gate("l1", True, cycle=0)
  for _ in range(4):
    run.tick()
  assert len(g.scheduler._gate_parked) == 1
  assert g.scheduler._gate_parked[0].wait_gate_level == "l1"

  # Close and re-open L2 gate; L1-parked Route must remain parked with zero retries.
  g.profile_controller._set_profile_gate("l2", True, cycle=run.cycle)
  g.profile_controller._set_profile_gate("l2", False, cycle=run.cycle)
  for _ in range(5):
    run.tick()
  assert len(g.scheduler._gate_parked) == 1
  assert g.scheduler.gate_woken_total == 0
  assert not g._grid_routes

  # Now open L1 gate -> Route wakes once and completes.
  g.profile_controller._set_profile_gate("l1", False, cycle=run.cycle)
  assert g.scheduler.gate_woken_total == 1
  run.finish()


# ---------------------------------------------------------------------------
# Native Verilated C ABI boundary GATE_NOTIFY + zero-poll verification
# ---------------------------------------------------------------------------


def native_gate_wake_case() -> dict:
  """Run a real Verilated C ABI endpoint under L1 gate close -> zero-poll -> OPEN_ISSUE wake."""
  with ExitStack() as stack:
    p = ObservedNative(
      0, ready_seed=7, stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_INPUT_RELEASED
    )
    stack.callback(p.close)
    run = start(participants=1, grids=1, leases=1, reads=True, ports={0: p})
    g = run.group
    assert 0 not in g.python_tiles_by_id

    # Close L1 gate BEFORE stepping the Group so DISPATCH_ROLE has not yet issued.
    g.profile_controller._set_profile_gate("l1", True, cycle=run.cycle)
    dispatch_probes = 0
    orig_dispatch = g.dispatch_role

    def spy_dispatch(*args, **kwargs):
      nonlocal dispatch_probes
      dispatch_probes += 1
      return orig_dispatch(*args, **kwargs)

    g.dispatch_role = spy_dispatch  # type: ignore[method-assign]

    run.until(lambda: len(p.messages("GATE_NOTIFY", sent=True)) == 1)
    closed_msg = p.messages("GATE_NOTIFY", sent=True)[0]
    assert closed_msg.fields == {"level": "l1", "state": "closed", "profile_generation": 0}
    run.until(lambda: p.g2t_accepted_seq("CTRL") >= closed_msg.message_seq)

    # Hold L1 gate closed across 25 additional full Group ticks: zero retries, zero scans, zero TASK_PLAN.
    for _ in range(25):
      run.tick()

    assert dispatch_probes == 1, f"expected 1 park probe, got {dispatch_probes}"
    assert g.scheduler.gate_parked_total == 1
    assert not g._grid_routes and not g.admission.tickets
    assert not p.messages("TASK_PLAN", sent=True)
    assert g.admission.stats["candidate_scans"] == 0
    assert g.admission.stats["gate_suppressed_ticks"] >= 20
    # Simulate ProfileController OPEN_ISSUE with generation=1: wakes Route, sends GATE_NOTIFY(open, gen=1),
    # Route registers on next tick, AdmissionCoordinator submits TASK_PLAN -> HOLD -> FINAL -> DONE.
    g.profile_controller._generations_mut["l1"] = 1
    g.profile_controller._set_profile_gate("l1", False, cycle=run.cycle)
    assert g.scheduler.gate_woken_total == 1
    assert g.admission.stats["gate_wake_events"] == 1

    run.finish()
    notifies = p.messages("GATE_NOTIFY", sent=True)
    assert [m.fields for m in notifies] == [
      {"level": "l1", "state": "closed", "profile_generation": 0},
      {"level": "l1", "state": "open", "profile_generation": 1},
    ]
    assert dispatch_probes == 2
    assert len(p.messages("TASK_PLAN", sent=True)) == 1
    assert len(p.messages("TASK_COMMIT_FINAL", sent=True)) == 1
    assert [m.fields["status"] for m in p.messages("TASK_COMPLETION")] == ["DONE"]
    return {
      "dispatch_probes": dispatch_probes,
      "gate_wake": g.gate_wake.snapshot(),
      "admission_stats": dict(g.admission.stats),
      "gate_notifies": [m.fields for m in notifies],
      "wire": p.summary(),
    }


def native_task_gate_park_case() -> dict:
  """Route already registered; L1 gate closes before TASK_PLAN -> zero CTRL traffic -> opens -> DONE."""
  with ExitStack() as stack:
    p = ObservedNative(
      0, ready_seed=11, stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_INPUT_RELEASED
    )
    stack.callback(p.close)
    run = start(participants=1, grids=1, leases=1, reads=False, ports={0: p})
    g = run.group
    # Register DISPATCH_ROLE and issue it without calling admission.step yet.
    g.scheduler.step(g._active_sequencers, 0)
    g.scheduler.step(g._active_sequencers, 1)
    assert len(g._grid_routes) == 1
    (route,) = g._grid_routes.values()
    run.cycle = 2

    g.profile_controller._set_profile_gate("l1", True, cycle=run.cycle)
    for _ in range(20):
      run.tick()

    assert not p.messages("TASK_PLAN", sent=True)
    assert not g.admission.tickets
    assert route.wait_reasons[0] == "WAIT_CONTROL_RESOURCE"
    assert g.admission.stats["candidate_scans"] == 0
    assert g.admission.stats["gate_park_events"] == 1
    assert g.admission.stats["gate_suppressed_ticks"] == 19

    # Open gate -> single wake -> TASK_PLAN crosses C ABI -> full lifecycle completes cleanly.
    g.profile_controller._set_profile_gate("l1", False, cycle=run.cycle)
    run.finish()
    assert g.admission.stats["gate_wake_events"] == 1
    assert len(p.messages("TASK_PLAN", sent=True)) == 1
    assert [m.fields["status"] for m in p.messages("TASK_COMPLETION")] == ["DONE"]
    assert [m.fields for m in p.messages("GATE_NOTIFY", sent=True)] == [
      {"level": "l1", "state": "closed", "profile_generation": 0},
      {"level": "l1", "state": "open", "profile_generation": 0},
    ]
    return {
      "gate_wake": g.gate_wake.snapshot(),
      "admission_stats": dict(g.admission.stats),
      "wire": p.summary(),
    }


def native_hold_after_plan_gate_wake_case() -> dict:
  """Close L1 gate after TASK_PLAN ACK (PARENT_PREP) before HOLD: zero parent/HOLD while closed,
  exactly one HOLD after OPEN_ISSUE; rollback and already-accepted retirement stay live."""
  # Phase 1: Single ticket closes L1 gate in the exact window after PLAN is wire-sent and before
  # PLAN ACK is delivered -> transitions to PARENT_PREP while L1 gate is closed -> zero parent
  # acquisition (pins=0, leases=0, participant_refs=0) and zero TASK_COMMIT while closed -> opens ->
  # exactly one HOLD and DONE.
  with ExitStack() as stack:
    p = ObservedNative(
      0, ready_seed=13, stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_INPUT_RELEASED
    )
    stack.callback(p.close)
    p.set_t2g_ready("CTRL", False)
    run = start(participants=1, grids=1, leases=1, reads=True, ports={0: p})
    g = run.group
    run.until(lambda: len(p.messages("TASK_PLAN", sent=True)) == 1)
    plan_msg = p.messages("TASK_PLAN", sent=True)[0]
    run.until(lambda: p.g2t_accepted_seq("CTRL") >= plan_msg.message_seq)
    assert not p.messages("TASK_COMMIT", sent=True) and run.pins == 0 and run.leases == 0

    # Close L1 gate BEFORE releasing CTRL T2G so PLAN ACK lands while L1 is already closed.
    g.profile_controller._set_profile_gate("l1", True, cycle=run.cycle)
    p.set_t2g_ready("CTRL", True)
    run.until(
      lambda: any(
        m.fields["ack_for_message_type"] == "TASK_PLAN" and m.fields["outcome"] == "ACCEPTED"
        for m in p.messages("ACK")
      )
    )
    (ticket,) = g.admission.tickets.values()
    (route,) = g._grid_routes.values()
    assert ticket.state.value == "parent_prep"

    acquire_probes = 0
    orig_acquire = g.admission._acquire_parent

    def spy_acquire(*args, **kwargs):
      nonlocal acquire_probes
      acquire_probes += 1
      return orig_acquire(*args, **kwargs)

    g.admission._acquire_parent = spy_acquire  # type: ignore[method-assign]

    # Step 20 ticks while L1 gate remains closed: zero parent acquisition, zero HOLD attempts.
    for _ in range(20):
      run.tick()

    assert acquire_probes == 0
    assert not ticket.parent_held
    assert run.pins == 0 and run.leases == 0
    event_entry = g.event_table.get(route.event_id)
    assert event_entry is not None and event_entry.participant_refs == 0
    assert not p.messages("TASK_COMMIT", sent=True)
    assert ticket.wait_gate_epoch == g.gate_wake.epoch("l1")
    assert ticket.wait_reason == "WAIT_CONTROL_RESOURCE"
    assert route.wait_reasons[0] == "WAIT_CONTROL_RESOURCE"
    assert g.admission.stats["hold_gate_park_events"] == 1
    assert g.admission.stats["hold_gate_suppressed_ticks"] >= 19

    # Open L1 gate -> wakes the parked HOLD ticket once -> acquires parent once -> sends ONE HOLD -> DONE.
    g.profile_controller._set_profile_gate("l1", False, cycle=run.cycle)
    assert ticket.wait_gate_epoch is None
    assert g.admission.stats["hold_gate_wake_events"] == 1
    run.finish()
    assert acquire_probes == 1
    assert len(p.messages("TASK_COMMIT", sent=True)) == 1
    assert len(p.messages("TASK_COMMIT_FINAL", sent=True)) == 1
    assert [m.fields["status"] for m in p.messages("TASK_COMPLETION")] == ["DONE"]
    hold_wake_summary = {
      "acquire_probes": acquire_probes,
      "admission_stats": dict(g.admission.stats),
      "gate_notifies": [m.fields for m in p.messages("GATE_NOTIFY", sent=True)],
      "wire": p.summary(),
    }

  # Phase 2: Rollback remains live while L1 gate is closed (ticket parked in PARENT_PREP rolls back
  # via TICKET_ROLLBACK -> ACK(COMPENSATED) without ever opening L1 gate).
  with ExitStack() as stack:
    from pipeline_validator.runtime.fault_ring import FaultCode

    p = ObservedNative(
      0, ready_seed=17, stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_INPUT_RELEASED
    )
    stack.callback(p.close)
    p.set_t2g_ready("CTRL", False)
    run = start(participants=1, grids=1, leases=1, reads=True, ports={0: p})
    g = run.group
    run.until(lambda: len(p.messages("TASK_PLAN", sent=True)) == 1)
    plan_msg = p.messages("TASK_PLAN", sent=True)[0]
    run.until(lambda: p.g2t_accepted_seq("CTRL") >= plan_msg.message_seq)
    g.profile_controller._set_profile_gate("l1", True, cycle=run.cycle)
    p.set_t2g_ready("CTRL", True)
    run.until(lambda: g.admission.stats["hold_gate_park_events"] == 1)
    assert not p.messages("TASK_COMMIT", sent=True) and run.pins == 0 and run.leases == 0

    # Trigger fault while L1 gate stays CLOSED: TICKET_ROLLBACK -> COMPENSATED and reset DRAIN complete.
    g.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=run.cycle, desc_id="c036-closed-gate-rollback")
    run.finish(fault=True)
    assert g.profile_controller.issue_gate_closed("l1")
    assert not p.messages("TASK_COMMIT", sent=True)
    assert len(p.messages("TICKET_ROLLBACK", sent=True)) == 1
    assert any(
      m.fields["ack_for_message_type"] == "TICKET_ROLLBACK" and m.fields["outcome"] == "COMPENSATED"
      for m in p.messages("ACK")
    )
    rollback_live_summary = {
      "rollbacks": len(p.messages("TICKET_ROLLBACK", sent=True)),
      "holds": len(p.messages("TASK_COMMIT", sent=True)),
      "gate_closed_at_end": g.profile_controller.issue_gate_closed("l1"),
    }

  # Phase 3: Already-accepted Task retires live while L1 gate is closed (RETIRE_READY -> RETIRE ->
  # TASK_COMPLETION), while a second planned ticket stays parked in PARENT_PREP until L1 opens.
  with ExitStack() as stack:
    p = ObservedNative(
      0, ready_seed=19, stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_INPUT_RELEASED
    )
    stack.callback(p.close)
    # Allow grid0 to accept; hold CTRL T2G after grid0 is accepted and before grid1 PLAN ACK arrives.
    run = start(participants=1, grids=2, leases=2, reads=True, pin=(0, 1), ports={0: p})
    g = run.group
    # Wait until grid0 is ACCEPTED and grid1 has a PLAN ticket (initial or post-drift re-PLAN),
    # then record grid1 holds before closing the L1 gate.
    for _ in range(500):
      run.tick()
      t0 = next((t for t in g.admission.tickets.values() if t.key.dispatch_ordinal == 0), None)
      t1 = next((t for t in g.admission.tickets.values() if t.key.dispatch_ordinal == 1), None)
      if t0 is not None and t0.accepted and t1 is not None and t1.state.value == "plan":
        holds_before_close = len(
          [m for m in p.messages("TASK_COMMIT", sent=True) if m.fields["dispatch_ordinal"] == 1]
        )
        g.profile_controller._set_profile_gate("l1", True, cycle=run.cycle)
        break
    else:
      pytest.fail("did not reach grid0 ACCEPTED + grid1 PLAN window")

    # While L1 gate stays CLOSED: grid0 MUST finish RETIRE -> TASK_COMPLETION(DONE), while grid1
    # stays parked in PARENT_PREP with zero new TASK_COMMIT and zero parent resources held.
    run.until(
      lambda: any(
        m.fields["dispatch_ordinal"] == 0 and m.fields["status"] == "DONE"
        for m in p.messages("TASK_COMPLETION")
      )
    )
    assert g.profile_controller.issue_gate_closed("l1")
    t1 = next(t for t in g.admission.tickets.values() if t.key.dispatch_ordinal == 1)
    assert t1.state.value == "parent_prep" and not t1.parent_held
    assert t1.wait_gate_epoch == g.gate_wake.epoch("l1")
    holds_while_closed = (
      len([m for m in p.messages("TASK_COMMIT", sent=True) if m.fields["dispatch_ordinal"] == 1])
      - holds_before_close
    )
    assert holds_while_closed == 0
    assert run.leases == 0 and run.pins == 0

    # Now open L1 gate: grid1 wakes once and immediately issues HOLD for the parked txn.
    parked_txn = t1.txn_id
    g.profile_controller._set_profile_gate("l1", False, cycle=run.cycle)
    run.finish()
    holds_for_parked_txn = [
      m
      for m in p.messages("TASK_COMMIT", sent=True)
      if m.fields["dispatch_ordinal"] == 1 and m.fields["txn_id"] == parked_txn
    ]
    assert len(holds_for_parked_txn) == 1
    assert [m.fields["status"] for m in p.messages("TASK_COMPLETION")] == ["DONE", "DONE"]
    retire_live_summary = {
      "grid1_holds_while_closed": holds_while_closed,
      "grid1_holds_after_open": len(holds_for_parked_txn),
      "hold_gate_park_events": g.admission.stats["hold_gate_park_events"],
      "hold_gate_wake_events": g.admission.stats["hold_gate_wake_events"],
    }

  # Phase 4: Under CTRL capacity=1 backpressure, a pending GATE_NOTIFY(open) MUST cross the
  # CTRL FIFO strictly before the woken TASK_COMMIT (HOLD), while TICKET_ROLLBACK is never
  # blocked by a pending GATE_NOTIFY.
  with ExitStack() as stack:
    p = ObservedNative(
      0,
      ready_seed=0x10,
      channel_capacities={"CTRL": 1},
      stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_INPUT_RELEASED,
    )
    stack.callback(p.close)
    p.set_t2g_ready("CTRL", False)
    run = start(participants=1, grids=1, leases=1, reads=True, ports={0: p})
    g = run.group
    run.until(lambda: len(p.messages("TASK_PLAN", sent=True)) == 1)
    plan_msg = p.messages("TASK_PLAN", sent=True)[0]
    run.until(lambda: p.g2t_accepted_seq("CTRL") >= plan_msg.message_seq)

    # Close L1 gate and park the PARENT_PREP ticket once PLAN ACK lands.
    g.profile_controller._set_profile_gate("l1", True, cycle=run.cycle)
    p.set_t2g_ready("CTRL", True)
    run.until(lambda: g.admission.stats["hold_gate_park_events"] == 1)
    run.until(lambda: p.outbox_ready("CTRL"))
    # Fill the single CTRL credit with an unrelated L2 GATE_NOTIFY so GATE_NOTIFY(l1, open)
    # cannot send immediately and stays pending when L1 opens.
    g.profile_controller._set_profile_gate("l2", True, cycle=run.cycle)
    assert not p.outbox_ready("CTRL")
    # Open L1 gate while CTRL credit is 0: wake clears wait_gate_epoch, but pending GATE_NOTIFY(open)
    # MUST hold back TASK_COMMIT on the first returned credit so GATE_NOTIFY(open) enters CTRL first.
    g.profile_controller._generations_mut["l1"] = 1
    g.profile_controller._set_profile_gate("l1", False, cycle=run.cycle)
    assert g.gate_wake.has_pending_notify(0, "l1")
    assert not p.messages("TASK_COMMIT", sent=True)

    g.profile_controller._set_profile_gate("l2", False, cycle=run.cycle)
    run.finish()
    ctrl_sent = [m for m in p.sent if m.channel == "CTRL"]
    open_idx = next(
      i
      for i, m in enumerate(ctrl_sent)
      if m.message_type == "GATE_NOTIFY" and m.fields["level"] == "l1" and m.fields["state"] == "open"
    )
    hold_idx = next(i for i, m in enumerate(ctrl_sent) if m.message_type == "TASK_COMMIT")
    assert open_idx < hold_idx, (
      f"GATE_NOTIFY(l1, open) at CTRL seq {ctrl_sent[open_idx].message_seq} must precede "
      f"TASK_COMMIT at seq {ctrl_sent[hold_idx].message_seq}"
    )
    assert ctrl_sent[open_idx].message_seq < ctrl_sent[hold_idx].message_seq
    open_before_hold_summary = {
      "open_seq": ctrl_sent[open_idx].message_seq,
      "hold_seq": ctrl_sent[hold_idx].message_seq,
      "open_precedes_hold": ctrl_sent[open_idx].message_seq < ctrl_sent[hold_idx].message_seq,
    }

  # Phase 5: Pending GATE_NOTIFY never blocks TICKET_ROLLBACK on CTRL capacity=1.
  with ExitStack() as stack:
    from pipeline_validator.runtime.fault_ring import FaultCode

    p = ObservedNative(
      0,
      ready_seed=0x10,
      channel_capacities={"CTRL": 1},
      stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_INPUT_RELEASED,
    )
    stack.callback(p.close)
    p.set_t2g_ready("CTRL", False)
    run = start(participants=1, grids=1, leases=1, reads=True, ports={0: p})
    g = run.group
    run.until(lambda: len(p.messages("TASK_PLAN", sent=True)) == 1)
    plan_msg = p.messages("TASK_PLAN", sent=True)[0]
    run.until(lambda: p.g2t_accepted_seq("CTRL") >= plan_msg.message_seq)
    # Fill the single CTRL credit with an L2 GATE_NOTIFY so GATE_NOTIFY(l1, closed) stays pending
    # when fault triggers rollback.
    run.until(lambda: p.outbox_ready("CTRL"))
    g.profile_controller._set_profile_gate("l2", True, cycle=run.cycle)
    assert not p.outbox_ready("CTRL")
    g.profile_controller._set_profile_gate("l1", True, cycle=run.cycle)
    assert g.gate_wake.has_pending_notify(0, "l1")
    p.set_t2g_ready("CTRL", True)
    g.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=run.cycle, desc_id="c036-pending-notify-rollback")
    run.finish(fault=True)
    assert len(p.messages("TICKET_ROLLBACK", sent=True)) == 1
    assert not p.messages("TASK_COMMIT", sent=True)
    rollback_unblocked_summary = {
      "rollbacks": len(p.messages("TICKET_ROLLBACK", sent=True)),
      "holds": len(p.messages("TASK_COMMIT", sent=True)),
    }

  return {
    "hold_wake": hold_wake_summary,
    "rollback_live": rollback_live_summary,
    "retire_live": retire_live_summary,
    "open_before_hold": open_before_hold_summary,
    "rollback_unblocked": rollback_unblocked_summary,
  }


def test_native_gate_wake_and_gate_notify_wire_roundtrip(tmp_path: Path) -> None:
  results = native_cases("test_w1_gate_wake", tmp_path)
  assert results["route_wake"]["dispatch_probes"] == 2
  assert results["route_wake"]["gate_notifies"] == [
    {"level": "l1", "state": "closed", "profile_generation": 0},
    {"level": "l1", "state": "open", "profile_generation": 1},
  ]
  assert results["task_wake"]["admission_stats"]["gate_park_events"] == 1
  assert results["task_wake"]["admission_stats"]["gate_wake_events"] == 1
  assert results["task_wake"]["admission_stats"]["gate_suppressed_ticks"] == 19
  assert results["hold_after_plan"]["hold_wake"]["acquire_probes"] == 1
  assert results["hold_after_plan"]["hold_wake"]["admission_stats"]["hold_gate_park_events"] == 1
  assert results["hold_after_plan"]["hold_wake"]["admission_stats"]["hold_gate_wake_events"] == 1
  assert results["hold_after_plan"]["hold_wake"]["admission_stats"]["hold_gate_suppressed_ticks"] >= 19
  assert results["hold_after_plan"]["rollback_live"] == {
    "rollbacks": 1,
    "holds": 0,
    "gate_closed_at_end": True,
  }
  assert results["hold_after_plan"]["retire_live"] == {
    "grid1_holds_while_closed": 0,
    "grid1_holds_after_open": 1,
    "hold_gate_park_events": 1,
    "hold_gate_wake_events": 1,
  }
  assert results["hold_after_plan"]["open_before_hold"]["open_precedes_hold"] is True
  assert results["hold_after_plan"]["rollback_unblocked"] == {"rollbacks": 1, "holds": 0}
  # Wire-level sanity on encoded GATE_NOTIFY fields.
  msg = BoundaryMessage(
    "GATE_NOTIFY", "CTRL", "G2T", {"level": "l1", "state": "open", "profile_generation": 3}
  )
  decoded = BoundaryMessage.from_wire(
    msg.pack_wire(), expected_port=(wire.CHANNELS.index("CTRL"), wire.DIRS.index("G2T"))
  )
  assert decoded.fields == {"level": "l1", "state": "open", "profile_generation": 3}


if __name__ == "__main__":
  out = Path(sys.argv[1])
  payload = {
    "route_wake": native_gate_wake_case(),
    "task_wake": native_task_gate_park_case(),
    "hold_after_plan": native_hold_after_plan_gate_wake_case(),
  }
  (out / "c035.json").write_text(json.dumps(payload, indent=2) + "\n")
