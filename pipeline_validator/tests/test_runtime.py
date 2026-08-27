"""Tests for the runtime-level cycle-accurate simulator (V2 fidelity).

Run with:  python -m pytest pipeline_validator/tests/test_runtime.py -v

These tests exercise the runtime / full_memory fidelity modes:
  - cold vs warm launch (residency)
  - event_id + sequence (P0-4 stale rejection)
  - fault ring + reset/drain FSM
  - L2 capacity gate
  - L1 slot frame
  - NoC VC model
  - payload tracker
  - backward compat: timing_only unaffected
"""

from __future__ import annotations

import json

import pytest
from xdsl.dialects.builtin import ModuleOp

from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.dialects.elenor import (
  NestAllocOp,
  NestAwaitOp,
  NestBuffer,
  NestContextOp,
  NestDispatchOp,
  NestGlobalMemref,
  NestGlobalView,
  NestL2View,
  NestReleaseOp,
  NestReturnOp,
  NestSubviewOp,
  NestTask,
  NestTaskRangeOp,
  NexusAwaitOp,
  NexusProgramOp,
  NexusReturnOp,
  NexusSubmitContextOp,
  TileAllocOp,
  TileAwaitOp,
  TileEvuOp,
  TileLoadOp,
  TileProgramDefOp,
  TileReturnOp,
  TileSignalOp,
  TileSubviewOp,
)
from pipeline_validator.execution_ir import (
  ExecDispatchRequest,
  ExecGroupAction,
  ExecGroupActionOp,
  ExecSignalPolicy,
  ExecStreamDesc,
  ExecTileGroupTask,
  ExecTileInst,
  ExecTileOp,
  ExecTileProgram,
  ExecTileRoleBinding,
  GlobalBinding,
  GridInstanceId,
  TaskIdentity,
)
from pipeline_validator.ir_lowering import lower_workload_ir
from pipeline_validator.memory import L2SRAM, NoCRouter, PayloadTracker
from pipeline_validator.runtime import EventStatus, EventTable, FaultCode, FaultRing
from pipeline_validator.runtime.fault_ring import FaultDomain, FaultRecord
from pipeline_validator.runtime.reset_domain import ResetDomain, ResetRequest, ResetState
from pipeline_validator.simulator import Simulator
from pipeline_validator.tile import TileUCE
from pipeline_validator.tile_group import TileGroup
from pipeline_validator.workload_builders import make_pow_tile_program
from pipeline_validator.workload_ir import parse_workload_ir, print_workload_ir
from pipeline_validator.workloads import ALL_WORKLOADS, PowWorkload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sim(fidelity: str = "runtime") -> Simulator:
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  sim = SimConfig(fidelity=fidelity)
  return Simulator(hw, sim)


L2_WAIT_DIMS = [1, 256, 256]  # 128 KiB bf16 (for make_waiting_mfe_program)
L2_WAIT_BYTES = 256 * 256 * 2  # 131072
POW_BINDINGS = {"Y": GlobalBinding("Y", 0x100000, 524288, "rw")}


MODEL_BINDINGS = {
  "Y0": GlobalBinding("Y0", 0x100000, L2_WAIT_BYTES, "rw"),
  "Y1": GlobalBinding("Y1", 0x200000, L2_WAIT_BYTES, "rw"),
}


def make_waiting_mfe_program(name: str = "ctx_wait_mfe") -> TileProgramDefOp:
  prog = TileProgramDefOp(
    name, [], arg_types=[NestTask(), NestBuffer.of(L2_WAIT_DIMS, "bf16")], arg_names=["task", "l2_buf"]
  )
  _task_arg, l2_arg = prog.body.block.args
  view = TileSubviewOp(
    l2_arg, None, None, [0, 0, 0], L2_WAIT_DIMS, [1, 1, 1], NestL2View.of(L2_WAIT_DIMS, "bf16")
  )
  l1 = TileAllocOp(L2_WAIT_DIMS[1:], "bf16")
  load = TileLoadOp(view.result, l1.result, "e_load")
  prog.body.block.add_ops(
    [view, l1, load, TileAwaitOp([load.result]), TileSignalOp("input_released", _task_arg), TileReturnOp()]
  )
  return prog


def make_short_evu_program(name: str = "ctx_short_evu") -> TileProgramDefOp:
  prog = TileProgramDefOp(name, [], arg_types=[NestTask()], arg_names=["task"])
  evu = TileEvuOp(op_name="relu", evu_ops=16, tag="e_evu")
  prog.body.block.add_ops([evu, TileAwaitOp([evu.result]), TileReturnOp()])
  return prog


def make_same_tile_roles_task(role_count: int, pins: list[int | None] | None = None) -> ModuleOp:
  """Dispatch role_count programs to one tile to exercise context switching."""
  names = ["ctx_wait_mfe"] + [f"ctx_short_evu{i}" for i in range(role_count - 1)]
  progs = [make_waiting_mfe_program(names[0])] + [make_short_evu_program(n) for n in names[1:]]
  tasks = NestTaskRangeOp(0, 1)
  buffer = NestAllocOp("l2_buf", "in", L2_WAIT_DIMS, "bf16")
  dispatches = []
  for i, name in enumerate(names):
    ins = [buffer.result] if i == 0 else []
    outs = [buffer.result] if i == 0 else []
    if i == 0:
      dispatches.append(
        NestDispatchOp(
          name,
          tasks.result,
          ins,
          outs,
          f"ev_role{i}",
          f"ev_inrel{i}",
          "",
          signal_policy={"input_released": "all_tasks"},
          context_id=None if pins is None else pins[i],
        )
      )
    else:
      dispatches.append(
        NestDispatchOp(
          name,
          tasks.result,
          ins,
          outs,
          f"ev_role{i}",
          "",
          "",
          signal_policy={},
          context_id=None if pins is None else pins[i],
        )
      )
  context = NestContextOp(
    "same_tile_roles",
    [
      buffer,
      tasks,
      *dispatches,
      NestReleaseOp(buffer.result, depends_on=[dispatches[0].input_released]),
      NestAwaitOp([d.grid_done for d in dispatches]),
      NestReturnOp(),
    ],
    placement=1,
  )
  return ModuleOp([*progs, context])


def make_two_context_model(pins: tuple[int | None, ...] = (None, None)) -> ModuleOp:
  """Two single-dispatch nest.contexts + one nexus.program submitting both."""
  prog = make_waiting_mfe_program("model_wait_mfe")
  ctxs = []
  for i, pin in enumerate(pins):
    buffer = NestAllocOp(f"l2_buf_c{i}", "in", L2_WAIT_DIMS, "bf16")
    tasks = NestTaskRangeOp(0, 1)
    disp = NestDispatchOp(
      "model_wait_mfe",
      tasks.result,
      [buffer.result],
      [buffer.result],
      f"ev_grid_c{i}",
      f"ev_inrel_c{i}",
      "",
      signal_policy={"input_released": "all_tasks"},
    )
    ctxs.append(
      NestContextOp(
        f"ctx{i}",
        [
          buffer,
          tasks,
          disp,
          NestReleaseOp(buffer.result, depends_on=[disp.input_released]),
          NestAwaitOp([disp.grid_done]),
          NestReturnOp(),
        ],
        arg_types=[NestGlobalMemref.of(L2_WAIT_DIMS, "bf16")],
        arg_names=["Y"],
        placement=1,
        context_id=pin,
      )
    )
  program = NexusProgramOp(
    "run_model",
    [],
    arg_types=[NestGlobalMemref.of(L2_WAIT_DIMS, "bf16")] * len(pins),
    arg_names=[f"Y{i}" for i in range(len(pins))],
  )
  args = list(program.body.block.args)
  submits = [NexusSubmitContextOp(f"ctx{i}", f"done_c{i}", actuals=[args[i]]) for i in range(len(pins))]
  program.body.block.add_ops([*submits, NexusAwaitOp([s.result for s in submits]), NexusReturnOp()])
  return ModuleOp([prog, *ctxs, program])


# ---------------------------------------------------------------------------
# Cold / warm launch (residency)
# ---------------------------------------------------------------------------


class TestRuntimeColdWarm:
  def test_cold_launch_includes_program_load(self):
    """Cold launch's PMU records program_cold_load > 0."""
    s = make_sim("runtime")
    wl = PowWorkload()
    r = s.run(wl.module, input_bindings=POW_BINDINGS)
    assert r.completed
    cold = r.pmu.named_cycles.get("program_cold_load", 0)
    assert cold > 0, f"cold launch should record cold_load > 0, got {cold}"

  def test_warm_launch_no_program_reload(self):
    """Second launch of same program: 0 new cold-load cycles."""
    s = make_sim("runtime")
    wl = PowWorkload()
    _r1 = s.run(wl.module, input_bindings=POW_BINDINGS)
    c1 = s.group.program_table.cold_load_cycles
    r2 = s.run(wl.module, input_bindings=POW_BINDINGS)
    c2 = s.group.program_table.cold_load_cycles
    assert c2 == c1, f"warm should add 0 cold cycles, got delta {c2 - c1}"
    assert r2.completed

  def test_warm_faster_than_cold(self):
    """Warm launch completes in fewer cycles than cold."""
    s = make_sim("runtime")
    wl = PowWorkload()
    r1 = s.run(wl.module, input_bindings=POW_BINDINGS)
    r2 = s.run(wl.module, input_bindings=POW_BINDINGS)
    assert r2.cycles < r1.cycles, f"warm {r2.cycles} should be < cold {r1.cycles}"

  def test_program_epoch_invalidate_on_group_reset(self):
    """Group reset bumps epoch; next dispatch is cold again."""
    s = make_sim("runtime")
    wl = PowWorkload()
    s.run(wl.module, input_bindings=POW_BINDINGS)
    c1 = s.group.program_table.cold_load_cycles
    s.group.program_table.invalidate_group()
    _r2 = s.run(wl.module, input_bindings=POW_BINDINGS)
    c2 = s.group.program_table.cold_load_cycles
    assert c2 > c1, "reset should force cold re-install"

  def test_tile_reset_invalidates_residency(self):
    """Per-tile reset makes that tile cold again."""
    s = make_sim("runtime")
    wl = PowWorkload()
    s.run(wl.module, input_bindings=POW_BINDINGS)
    c1 = s.group.program_table.cold_load_cycles
    s.group.program_table.invalidate_tile(0)
    _r2 = s.run(wl.module, input_bindings=POW_BINDINGS)
    c2 = s.group.program_table.cold_load_cycles
    assert c2 > c1, "tile reset should force cold re-install on that tile"

  def test_program_id_hash_stable_and_ir_unchanged_across_warm_runs(self):
    """Repeated runs of the same module keep canonical IR unchanged and
    produce the same program_id/program_hash; changing a descriptor scalar
    changes the hash and triggers a fresh cold install."""
    s = make_sim("runtime")
    wl = PowWorkload()

    before = print_workload_ir(wl.module)
    lowered1 = lower_workload_ir(wl.module)
    s._assign_program_ids(lowered1)
    prog1 = lowered1.role_bindings[0].tile_program
    id1 = prog1.program_id
    hash1 = prog1.program_hash

    r1 = s.run(wl.module, input_bindings=POW_BINDINGS)
    assert r1.completed, r1.reason
    after1 = print_workload_ir(wl.module)
    assert after1 == before

    lowered2 = lower_workload_ir(wl.module)
    s._assign_program_ids(lowered2)
    prog2 = lowered2.role_bindings[0].tile_program
    assert prog2.program_id == id1
    assert prog2.program_hash == hash1

    r2 = s.run(wl.module, input_bindings=POW_BINDINGS)
    assert r2.completed, r2.reason
    after2 = print_workload_ir(wl.module)
    assert after2 == before
    assert r2.pmu.named_cycles.get("program_cold_load", 0) == 0

    # Mutate a pow descriptor scalar (exponent 2 -> 3) to force hash change
    mutated_text = before.replace("exponent = 2 pow_ops = 65536", "exponent = 3 pow_ops = 65536", 1)
    mutated_module = parse_workload_ir(mutated_text, source_name="<mutated>")
    lowered3 = lower_workload_ir(mutated_module)
    s._assign_program_ids(lowered3)
    prog3 = lowered3.role_bindings[0].tile_program
    assert prog3.program_id == id1
    assert prog3.program_hash != hash1

    cold_before = s.group.program_table.cold_load_cycles
    r3 = s.run(mutated_module, input_bindings=POW_BINDINGS)
    assert r3.completed, r3.reason
    cold_after = s.group.program_table.cold_load_cycles
    assert cold_after > cold_before, "changed descriptor scalar should force cold install"

  def test_group_reset_rebinds_same_name_in_new_hbm_epoch(self):
    """Fresh reset clears the name->handle cache before HBM reset; the
    second run must bind the same global name to a new region/epoch."""
    from pipeline_validator.memory import MemoryInvariantError

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = Simulator(hw, SimConfig(fidelity="runtime", max_cycles=200000))
    first = {"Y": GlobalBinding("Y", 0x100000, 524288, "rw")}
    r1 = sim.run(PowWorkload().module, input_bindings=first)
    assert r1.completed, r1.reason
    old = sim.group._global_handles["Y"]
    sim.group.reset()
    assert sim.group._global_handles == {}
    assert sim.group.hbm.snapshot()["external_bindings"] == 0
    with pytest.raises(MemoryInvariantError, match="stale allocation generation"):
      sim.group.hbm.assert_live(old)
    second = {"Y": GlobalBinding("Y", 0x400000, 524288, "rw")}
    r2 = sim.run(PowWorkload().module, input_bindings=second)
    assert r2.completed, r2.reason
    new = sim.group._global_handles["Y"]
    assert new.base_address == 0x400000
    assert new.generation > old.generation
    assert new != old


# ---------------------------------------------------------------------------
# Global DMA channel allocation
# ---------------------------------------------------------------------------


class TestDMAChannelScheduling:
  def test_two_channels_dma_stores_distribute(self):
    """Four DMA stores complete as transactions on the GroupDMA track.

    PR 2 replaces the round-robin channel selector with the
    TransferManager's lowest-free-channel allocation; the observable
    contract is that all four stores complete and are traced.
    """
    hw = HardwareConfig(num_dma_channels=2)
    sim = Simulator(hw, SimConfig(fidelity="runtime", max_cycles=200_000), enable_tracer=True)
    result = sim.run(PowWorkload().module, input_bindings=POW_BINDINGS)
    assert result.completed, result.reason
    assert result.tracer is not None
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    store_events = [event for event in events if event.get("name", "").startswith("dma.store:")]
    assert len(store_events) == 4
    for event in store_events:
      args = event["args"]
      expected_cycles = args["completion_cycle"] - args["start_cycle"]
      assert expected_cycles > 1
      assert event["dur"] == pytest.approx(expected_cycles * hw.cycle_ns() / 1000.0)


class TestFullMemorySnapshot:
  def test_pow_full_memory_snapshot_invariants(self):
    """API-level full-memory run: completed, peak allocations positive,
    context-owned L2/L1 released, HBM binding kept, no inflight transfers,
    NoC credits restored."""
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = Simulator(hw, SimConfig(fidelity="full_memory", max_cycles=200000))
    result = sim.run(PowWorkload().module, input_bindings=POW_BINDINGS)
    assert result.completed, result.reason
    mem = sim.group.snapshot()["memory"]
    assert mem["fidelity"] == "full_memory"
    assert mem["hbm"]["external_bindings"] == 1  # external binding kept
    assert mem["l2"]["peak_allocated_bytes"] > 0
    assert mem["l2"]["live_allocations"] == 0
    for tile_id, l1 in mem["l1"].items():
      assert l1["allocator"]["peak_allocated_bytes"] > 0, tile_id
      assert l1["allocator"]["live_allocations"] == 0, tile_id
    assert mem["transfers"]["inflight"] == 0
    for vc in mem["noc"].values():
      assert vc["credit"] == hw.noc_vc_depth  # credits restored


class TestL2DispatchPins:
  def test_inout_actual_pins_once_per_task_and_defers_release(self):
    """Dispatch deduplicates in/out actuals by allocation id; RELEASE_L2
    stays pending until the grid's output_ready pins are unpinned."""
    task = lower_workload_ir(PowWorkload().module)
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    group = TileGroup(hw, fidelity="runtime")
    group.load_task(task, input_bindings=POW_BINDINGS)
    seq = group.sequencer
    binding = task.role_bindings[0]
    role_event = "ev_pin_contract"
    request = ExecDispatchRequest(
      role_id=binding.role_id,
      dispatch_ordinal=0,
      signal_policy=ExecSignalPolicy(input_released="all_tasks", output_ready="all_tasks"),
      input_released_event="ev_inrel_pin",
      output_ready_event="ev_outready_pin",
    )
    assert group.dispatch_role(binding, cycle=0, request=request, event_id=role_event, sequencer=seq)
    slot = binding.actuals[0]
    key = (seq.context_launch_generation, slot)
    handle = group._l2_handles[key]
    grid = seq.grid_id(0)
    pins = group._grid_l2_pins[grid]
    # Four logical tasks (one per tile), despite the same inout appearing
    # in both ins and outs: no duplicate pin per task.
    for task_id in range(4):
      assert slot in pins[task_id]
    # Check actual consumer IDs in the allocator.
    record = group.l2_sram._allocator._live[handle.allocation_id]
    expected_pins = {f"pow_task:s0:g{seq.context_launch_generation}:d0:t{tid}:{slot}" for tid in range(4)}
    assert record.pins == expected_pins
    group.reset()


class TestAtomicDispatchAdmission:
  @staticmethod
  def _sequencer(group: TileGroup, task: ExecTileGroupTask):
    from pipeline_validator.tile_group_sequencer import TileGroupSequencer

    seq = TileGroupSequencer(group)
    seq.context_launch_generation = group.sequencer.context_launch_generation
    seq.context_name = group.sequencer.context_name
    seq.device_slot = 0
    seq.load(task)
    return seq

  @staticmethod
  def _make_request(binding):
    return ExecDispatchRequest(
      role_id=binding.role_id,
      dispatch_ordinal=0,
      signal_policy=ExecSignalPolicy(input_released="all_tasks", output_ready="all_tasks"),
      input_released_event="ev_inrel_atomic",
      output_ready_event="ev_outready_atomic",
    )

  def test_later_tile_capacity_failure_commits_nothing(self):
    """Tile0 plan succeeds, tile1 capacity fails; no earlier tile commits,
    pins, frames or contexts may become live, and the issuing seq faults."""
    from pipeline_validator.memory import AdmissionFailure, AllocationRequest, TaskBufferOwner

    task = lower_workload_ir(PowWorkload().module)
    group = TileGroup(HardwareConfig(), fidelity="runtime")
    group.load_task(task, input_bindings=POW_BINDINGS)
    blocker_owner = TaskBufferOwner("block", 0, "block", 0, 1, 0, "block")
    blocker_plan = group.tiles[1].l1_allocator.plan_bundle(
      [AllocationRequest("l1", "block", blocker_owner, group.cfg.tile_l1_bytes - 16 * 1024, 1)]
    )
    assert not isinstance(blocker_plan, AdmissionFailure)
    group.tiles[1].l1_allocator.commit(blocker_plan, cycle=0)
    seq = self._sequencer(group, task)
    binding = task.role_bindings[0]
    event_id = "ev_atomic_capacity"
    assert not group.dispatch_role(
      binding, cycle=1, request=self._make_request(binding), event_id=event_id, sequencer=seq
    )
    assert seq.faulted and seq.done
    assert "tile 1" in seq.fault_reason
    assert not group.sequencer.faulted
    assert group.tiles[0].l1_allocator.snapshot()["live_allocations"] == 0
    assert group.tiles[1].l1_allocator.snapshot()["live_allocations"] == 1
    for tile in group.tiles:
      assert all(ctx["state"] == "empty" for ctx in tile.uce.snapshot()["contexts"])
      assert all(frame.snapshot()["active_slots"] == 0 for frame in tile.l1_frames)
    assert event_id not in group._role_event_tile_mask
    assert event_id not in group._role_l1_handles
    assert not group._grid_l2_pins
    group.reset()

  def test_late_context_bind_failure_rolls_back_all_tiles(self, monkeypatch):
    """All plans/commits/prepares/pins succeed, then tile1 bind fails:
    tile0's earlier bind and every allocation/pin/frame are rolled back."""
    task = lower_workload_ir(PowWorkload().module)
    group = TileGroup(HardwareConfig(), fidelity="runtime")
    group.load_task(task, input_bindings=POW_BINDINGS)
    seq = self._sequencer(group, task)
    binding = task.role_bindings[0]
    event_id = "ev_atomic_bind"
    monkeypatch.setattr(group.tiles[1], "load_program", lambda *args, **kwargs: None)
    assert not group.dispatch_role(
      binding, cycle=1, request=self._make_request(binding), event_id=event_id, sequencer=seq
    )
    assert seq.faulted and seq.done
    assert "tile 1" in seq.fault_reason
    assert not group.sequencer.faulted
    for tile in group.tiles:
      assert tile.l1_allocator.snapshot()["live_allocations"] == 0
      assert all(ctx["state"] == "empty" for ctx in tile.uce.snapshot()["contexts"])
      assert all(frame.snapshot()["active_slots"] == 0 for frame in tile.l1_frames)
    assert event_id not in group._role_event_tile_mask
    assert event_id not in group._role_l1_handles
    assert not group._grid_l2_pins
    for handle in group._l2_handles.values():
      record = group.l2_sram._allocator._live[handle.allocation_id]
      assert record.pins == set()
    group.reset()


# ---------------------------------------------------------------------------
# Event sequence (P0-4)
# ---------------------------------------------------------------------------


class TestEventSequence:
  def test_stale_sequence_rejected(self):
    """signal() with a sequence the waiter doesn't expect is rejected."""
    et = EventTable()
    et.register("ev0")
    et.wait("ev0", expected_sequence=5)
    # signal with sequence 0 (stale) should fail
    ok = et.signal("ev0", EventStatus.DONE, producer_id=0, cycle=0)
    assert not ok, "stale sequence should be rejected"
    assert et.pmu_stale_sequence_count == 1

  def test_correct_sequence_accepted(self):
    """signal() with matching sequence succeeds."""
    et = EventTable()
    e = et.register("ev0")
    et.wait("ev0", expected_sequence=e.sequence)
    ok = et.signal("ev0", EventStatus.DONE, producer_id=0, cycle=0)
    assert ok

  def test_reset_marks_pending_reset(self):
    """Runtime ABI 3.2: reset marks pending events as RESET, not silent."""
    et = EventTable()
    et.register("ev0")
    et.register("ev1")
    et.signal("ev0", EventStatus.DONE, producer_id=0, cycle=0)
    # ev1 is still pending
    et.reset()
    e1 = et.get("ev1")
    assert e1.status == EventStatus.RESET
    e0 = et.get("ev0")
    assert e0.status == EventStatus.DONE  # already done, not overwritten

  def test_wait_returns_none_when_pending(self):
    """wait() on a pending event returns None (not truthy)."""
    et = EventTable()
    et.register("ev0")
    status = et.wait("ev0")
    assert status is None

  def test_error_status_not_treated_as_success(self):
    """wait() returns EventStatus.ERROR, which must not be truthy-success."""
    et = EventTable()
    et.register("ev0")
    et.signal("ev0", EventStatus.ERROR, producer_id=0, cycle=0)
    status = et.wait("ev0")
    assert status is EventStatus.ERROR
    assert status is not EventStatus.DONE


# ---------------------------------------------------------------------------
# Fault / reset
# ---------------------------------------------------------------------------


class TestFaultReset:
  def test_fault_ring_write_and_read(self):
    fr = FaultRing(slots=4)
    rec = FaultRecord(code=FaultCode.ENGINE_INTERNAL_FAULT, tile_id=2)
    idx = fr.write(rec)
    assert idx == 0
    assert len(fr) == 1
    latest = fr.latest()
    assert latest is not None
    assert latest.code == FaultCode.ENGINE_INTERNAL_FAULT
    assert latest.tile_id == 2

  def test_trigger_fault_writes_record_and_starts_drain(self):
    s = make_sim("runtime")
    wl = PowWorkload()
    s.run(wl.module, input_bindings=POW_BINDINGS)
    idx = s.group.trigger_fault(FaultCode.ENGINE_INTERNAL_FAULT, tile_id=1, cycle=100)
    assert idx >= 0
    assert len(s.group.fault_ring) == 1
    assert s.group.reset_domain.is_active

  def test_reset_drain_advances_to_done(self):
    """The reset/drain FSM steps through to DONE."""
    hw = HardwareConfig()
    rd = ResetDomain(hw)
    req = ResetRequest(domain=FaultDomain.TILE, tile_id=0)
    rd.begin(req, cycle=0)
    assert rd.is_active
    # step through all states (8 transitions: FAULT_DETECTED -> DONE)
    for _ in range(20):
      rd.step(cycle=100, group=None)
      if rd.is_done:
        break
    assert rd.is_done

  def test_fault_reset_cancels_inflight_and_returns_resources(self):
    """A fault with in-flight transfers: the reset domain drains or
    times out, ``cancel_all`` returns HBM outstanding credits, NoC
    credits, DMA channels and bank reservations, and context-owned
    L2/L1 allocations are released."""
    from pipeline_validator.ir_lowering import lower_workload_ir

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=1000)
    s = Simulator(hw, SimConfig(fidelity="full_memory", max_cycles=100000))
    wl = PowWorkload()
    task = lower_workload_ir(wl.module)
    s._assign_program_ids(task)
    s.group.load_task(task, input_bindings=POW_BINDINGS)
    # first steps: sequencer issues the first prefetch (cycle 0), the
    # HBM leg then issues on the manager step of cycle 1
    s.group.step(0)
    s.group.step(1)
    assert s.group.transfer_manager.inflight_count > 0
    assert s.group.transfer_manager._hbm_read._outstanding > 0
    assert s.group.l2_sram.snapshot()["live_allocations"] > 0
    s.group.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=2)
    # FAULT_DETECTED advances once; after STOP_QUEUE the sequencer index
    # must stay frozen while transfers/engines continue draining.
    s.group.step(2)
    assert s.group.reset_domain.state == ResetState.STOP_QUEUE
    frozen_action_index = s.group.sequencer.action_index
    # step until the reset domain completes (drain timeout cancels the
    # in-flight prefetches long before the 1000-cycle HBM leg finishes)
    for cycle in range(3, 500):
      s.group.step(cycle)
      if s.group.reset_domain.is_done:
        break
    assert s.group.reset_domain.is_done
    assert s.group.sequencer.action_index == frozen_action_index
    assert s.group.transfer_manager.inflight_count == 0
    assert s.group.transfer_manager._hbm_read._outstanding == 0
    assert s.group.transfer_manager._hbm_write._outstanding == 0
    # NoC credit / DMA / bank resources all returned; no flit pending
    for stage in (
      s.group.transfer_manager._global_dma,
      s.group.transfer_manager._l2_read,
      s.group.transfer_manager._l2_write,
    ):
      assert all(b == 0 for b in stage._busy_until), stage.name
      assert all(h is None for h in stage._holders), stage.name
    for vc in s.group.noc.vcs.values():
      assert vc.occupancy == 0  # no pending flits
    assert s.group.l2_sram.snapshot()["live_allocations"] == 0
    for tile in s.group.tiles:
      assert tile.l1_allocator.snapshot()["live_allocations"] == 0
    # NoC router credits restored to full depth
    for vc in s.group.noc.vcs.values():
      assert vc.credit_available == hw.noc_vc_depth

  def test_l2_capacity_fault_terminates_task(self):
    """A tiny L2 SRAM triggers a capacity fault on prefetch."""
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10, group_sram_bytes=1024)
    sim = SimConfig(fidelity="full_memory", max_cycles=10000)
    s = Simulator(hw, sim)
    wl = PowWorkload()
    r = s.run(wl.module, input_bindings=POW_BINDINGS)
    assert not r.completed
    assert "faulted" in r.reason
    assert "L2 capacity fault" in r.reason
    assert s.group.reset_domain.is_done
    assert s.group.transfer_manager.inflight_count == 0
    assert s.group.l2_sram.snapshot()["live_allocations"] == 0
    latest = s.group.fault_ring.latest()
    assert latest is not None
    assert latest.code == FaultCode.L2_CAPACITY_FAULT

  def test_l2_exact_capacity_completes_and_overshoot_faults(self):
    """4 pow chunks (4 x 128 KiB per chunk) fit exactly in a 512 KiB L2
    and complete; a 5th chunk overshoots capacity and faults.  This
    proves the alloc/store/release accounting stays balanced (no double
    accounting on DMA_STORE touching an existing slot)."""
    chunk_bytes = 128 * 128 * 2  # per-tile plane
    bytes_per_chunk = chunk_bytes * 4  # 4 tiles' input per group chunk
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10, group_sram_bytes=4 * bytes_per_chunk)
    sim = SimConfig(fidelity="full_memory", max_cycles=200000)
    s = Simulator(hw, sim)
    wl = PowWorkload()
    r = s.run(wl.module, input_bindings=POW_BINDINGS)
    assert r.completed, r.reason

    # overshoot: 5 chunks need 5 x bytes_per_chunk but only 4 x fit
    from pipeline_validator.workload_builders import make_pow_task

    module5 = make_pow_task(num_group_chunks=5)
    hw2 = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10, group_sram_bytes=4 * bytes_per_chunk)
    sim2 = SimConfig(fidelity="full_memory", max_cycles=200000)
    s2 = Simulator(hw2, sim2)
    r2 = s2.run(module5, input_bindings={"Y": GlobalBinding("Y", 0x100000, 655360, "rw")})
    assert not r2.completed
    assert "faulted" in r2.reason


# ---------------------------------------------------------------------------
# Memory models
# ---------------------------------------------------------------------------


class TestMemory:
  def test_l2_plan_commit_release(self):
    """L2 plan/commit/release round-trip with the new allocator."""
    from pipeline_validator.memory import AllocationRequest, ContextBufferOwner

    l2 = L2SRAM(capacity_bytes=4096, banks=4)
    o = ContextBufferOwner("ctx", 0, "A")
    plan = l2.plan_bundle([AllocationRequest("l2", "A", o, 2048, 1)])
    handles = l2.commit(plan, cycle=0)
    assert len(handles) == 1
    assert l2.snapshot()["allocated_bytes"] == 2048
    assert l2.request_release(handles[0], o, cycle=10) is True
    assert l2.snapshot()["allocated_bytes"] == 0

  def test_l2_capacity_fault(self):
    """Over-capacity L2 allocation returns AdmissionFailure."""
    from pipeline_validator.memory import AdmissionFailure, AllocationRequest, ContextBufferOwner

    l2 = L2SRAM(capacity_bytes=1024, banks=2)
    o = ContextBufferOwner("ctx", 0, "A")
    plan = l2.plan_bundle([AllocationRequest("l2", "A", o, 1025, 1)])
    assert isinstance(plan, AdmissionFailure)
    assert plan.reason == "allocation capacity exceeded"

  def test_noc_router_has_four_vcs(self):
    noc = NoCRouter()
    assert len(noc.vcs) == 4
    # VC0 (command/event) has highest priority (lowest int)
    assert noc.vcs[0].priority == 0

  def test_noc_vc0_not_starved_by_vc2(self):
    """VC0 with starvation boost should eventually send even if VC2 is full."""
    noc = NoCRouter(vc_depth=2)
    # fill VC2 with flits
    from pipeline_validator.memory.noc import Flit

    for i in range(4):
      noc.send(2, Flit(vc=2, src=0, dst=1, bytes_total=64, tag=f"f{i}"), cycle=0)
    # put one flit on VC0
    noc.send(0, Flit(vc=0, src=0, dst=1, bytes_total=32, tag="cmd"), cycle=0)
    sent_vc0 = False
    for cycle in range(20):
      sent = noc.step(cycle)
      for f in sent:
        if f.vc == 0:
          sent_vc0 = True
      if sent_vc0:
        break
    assert sent_vc0, "VC0 should not be starved by VC2"

  def test_payload_copy_creates_metadata(self):
    pt = PayloadTracker()
    from pipeline_validator.memory.payload import Payload

    pt.alloc(100, Payload(iova=100, bytes_total=1024, layout="row_major"))
    ok = pt.copy(100, 200, 1024)
    assert ok
    dst = pt.get(200)
    assert dst is not None
    assert dst.layout == "row_major"

  def test_payload_layout_compat_check(self):
    pt = PayloadTracker()
    from pipeline_validator.memory.payload import Payload

    pt.alloc(100, Payload(iova=100, bytes_total=1024, layout="paged_kv", head_dim=64, producer_kind="MFE"))
    # matching layout → ok
    assert pt.check_layout_compat(100, "BOA", expected_layout="paged_kv", expected_head_dim=64)
    # mismatched layout → fault
    assert not pt.check_layout_compat(100, "BOA", expected_layout="row_major")
    assert pt.layout_fault_count == 1


class TestLocalViewResolution:
  @staticmethod
  def _view(space: str, base: str, backing: int, size: int, offset: int = 0):
    from pipeline_validator.execution_ir import ExecMemoryView

    return ExecMemoryView(
      space=space,
      base=base,
      backing_dims=(backing,),
      dims=(size,),
      offsets=(offset,),
      strides=(1,),
      dtype="i8",
      element_bytes=1,
      bytes=size,
    )

  def test_l1_and_l2_views_keep_real_cross_bank_segments(self):
    from pipeline_validator.memory import (
      AdmissionFailure,
      AllocationRequest,
      ContextBufferOwner,
      TaskBufferOwner,
    )
    from pipeline_validator.tile import ComputeTile, _TileContextMemory

    cfg = HardwareConfig().with_overrides(tile_l1_bytes=1024, tile_l1_banks=2)
    tile = ComputeTile(0, cfg)
    l1_owner = TaskBufferOwner("ctx", 1, "ev", 0, 0, 0, "l1:0")
    l1_plan = tile.l1_allocator.plan_bundle([AllocationRequest("l1", "l1:0", l1_owner, 768, 1)])
    assert not isinstance(l1_plan, AdmissionFailure)
    l1_handle = tile.l1_allocator.commit(l1_plan, cycle=0)[0]

    l2 = L2SRAM(capacity_bytes=1024, banks=2)
    l2_owner = ContextBufferOwner("ctx", 1, "l2_buf")
    l2_plan = l2.plan_bundle([AllocationRequest("l2", "l2_buf", l2_owner, 768, 1)])
    assert not isinstance(l2_plan, AdmissionFailure)
    l2_handle = l2.commit(l2_plan, cycle=0)[0]
    memory = _TileContextMemory(
      owner=l1_owner,
      task_identity=TaskIdentity(grid=GridInstanceId("ctx", 0, 1, 0), task_id=0),
      l2_formal_handles=(l2_handle,),
      l1_handles={"l1:0": l1_handle},
      l2_resolver=l2,
    )

    l1_view = TileUCE._resolve_tile_view(self._view("l1", "l1:0", 768, 768), memory, tile)
    l2_view = TileUCE._resolve_tile_view(self._view("l2", "formal:1", 768, 768), memory, tile)
    assert l1_view is not None
    assert l2_view is not None
    assert [(s.bank_id, s.size_bytes) for s in l1_view.segments] == [(0, 512), (1, 256)]
    assert [(s.bank_id, s.size_bytes) for s in l2_view.segments] == [(0, 512), (1, 256)]

  def test_local_view_oob_raises_allocator_fault(self):
    from pipeline_validator.memory import (
      AdmissionFailure,
      AllocationRequest,
      MemoryInvariantError,
      TaskBufferOwner,
    )
    from pipeline_validator.tile import ComputeTile, _TileContextMemory

    cfg = HardwareConfig().with_overrides(tile_l1_bytes=1024, tile_l1_banks=2)
    tile = ComputeTile(0, cfg)
    owner = TaskBufferOwner("ctx", 1, "ev", 0, 0, 0, "l1:0")
    plan = tile.l1_allocator.plan_bundle([AllocationRequest("l1", "l1:0", owner, 512, 1)])
    assert not isinstance(plan, AdmissionFailure)
    handle = tile.l1_allocator.commit(plan, cycle=0)[0]
    memory = _TileContextMemory(
      owner=owner,
      task_identity=TaskIdentity(grid=GridInstanceId("ctx", 0, 1, 0), task_id=0),
      l1_handles={"l1:0": handle},
    )
    with pytest.raises(MemoryInvariantError, match="memory view out of bounds"):
      TileUCE._resolve_tile_view(self._view("l1", "l1:0", 512, 200, offset=400), memory, tile)

  def test_local_view_use_after_release_raises(self):
    from pipeline_validator.memory import (
      AdmissionFailure,
      AllocationRequest,
      MemoryInvariantError,
      TaskBufferOwner,
    )
    from pipeline_validator.tile import ComputeTile, _TileContextMemory

    cfg = HardwareConfig().with_overrides(tile_l1_bytes=1024, tile_l1_banks=2)
    tile = ComputeTile(0, cfg)
    owner = TaskBufferOwner("ctx", 1, "ev", 0, 0, 0, "l1:0")
    plan = tile.l1_allocator.plan_bundle([AllocationRequest("l1", "l1:0", owner, 512, 1)])
    assert not isinstance(plan, AdmissionFailure)
    handle = tile.l1_allocator.commit(plan, cycle=0)[0]
    memory = _TileContextMemory(
      owner=owner,
      task_identity=TaskIdentity(grid=GridInstanceId("ctx", 0, 1, 0), task_id=0),
      l1_handles={"l1:0": handle},
    )
    tile.l1_allocator.request_release(handle, owner, cycle=1)
    with pytest.raises(MemoryInvariantError, match="use-after-release"):
      TileUCE._resolve_tile_view(self._view("l1", "l1:0", 512, 64), memory, tile)


# ---------------------------------------------------------------------------
# Slot frame
# ---------------------------------------------------------------------------


class TestSlotFrame:
  def test_frame_prepare_bind_succeeds(self):
    """prepare() + bind() round-trip with real allocation handles."""
    from pipeline_validator.execution_ir import ExecL1Buffer
    from pipeline_validator.memory import AllocationHandle, BankSegment, SlotFrame, TaskBufferOwner

    f = SlotFrame(l1_bytes=1024 * 1024)
    owner = TaskBufferOwner("ctx", 0, "ev", 0, 0, 0, "l1:0")
    handle = AllocationHandle(
      allocation_id="l1:0:1",
      memory_space="l1",
      owner=owner,
      base_address=0,
      size_bytes=512,
      alignment=256,
      bank_segments=(BankSegment(0, 0, 512),),
      generation=0,
      allocate_cycle=0,
    )
    spec = ExecL1Buffer(name="l1:0", dims=(16, 16), dtype="bf16", element_bytes=2, alignment=256, bytes=512)
    assert f.prepare([handle], [spec]) is True
    ok, cycles = f.bind(cycle=0, bind_cycles=8)
    assert ok
    assert cycles == 8
    assert f.shadow is not None

  def test_frame_capacity_fault(self):
    """prepare() rejects an L1 spec that exceeds l1_bytes."""
    from pipeline_validator.execution_ir import ExecL1Buffer
    from pipeline_validator.memory import AllocationHandle, BankSegment, SlotFrame, TaskBufferOwner

    f = SlotFrame(l1_bytes=512)
    owner = TaskBufferOwner("ctx", 0, "ev", 0, 0, 0, "l1:0")
    handle = AllocationHandle(
      allocation_id="l1:0:1",
      memory_space="l1",
      owner=owner,
      base_address=0,
      size_bytes=512,
      alignment=1,
      bank_segments=(BankSegment(0, 0, 512),),
      generation=0,
      allocate_cycle=0,
    )
    spec = ExecL1Buffer(name="l1:0", dims=(16, 16), dtype="bf16", element_bytes=2, alignment=1, bytes=512)
    assert f.prepare([handle], [spec]) is True  # exactly fits
    # a second buffer exceeding capacity fails
    spec2 = ExecL1Buffer(name="l1:1", dims=(16, 16), dtype="bf16", element_bytes=2, alignment=1, bytes=512)
    handle2 = AllocationHandle(
      allocation_id="l1:0:2",
      memory_space="l1",
      owner=owner,
      base_address=512,
      size_bytes=512,
      alignment=1,
      bank_segments=(BankSegment(0, 512, 512),),
      generation=0,
      allocate_cycle=0,
    )
    f2 = SlotFrame(l1_bytes=512)
    assert f2.prepare([handle, handle2], [spec, spec2]) is False

  def test_frame_generation_gate(self):
    from pipeline_validator.memory import SlotFrame

    f = SlotFrame(generation=5)
    assert f.check_generation(5)
    assert not f.check_generation(4)


# ---------------------------------------------------------------------------
# Fidelity modes
# ---------------------------------------------------------------------------


class TestFidelityModes:
  def test_all_workloads_complete_in_all_fidelities(self):
    """Every workload completes in all three fidelity modes."""
    import signal

    def handler(signum, frame):
      raise TimeoutError("workload timed out")

    signal.signal(signal.SIGALRM, handler)
    for fidelity in ("timing_only", "runtime", "full_memory"):
      hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
      sim = SimConfig(fidelity=fidelity, context_count=1, max_cycles=200000)
      for wl_cls in ALL_WORKLOADS:
        wl = wl_cls()
        s = Simulator(hw, sim)
        signal.alarm(60)
        r = s.run(wl.module, input_bindings=POW_BINDINGS)
        signal.alarm(0)
        assert r.completed, f"{wl.name} failed in {fidelity}: {r.reason}"

  def test_runtime_context_count_two_runs_two_same_tile_roles(self):
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", context_count=2, max_cycles=10000))
    result = sim.run(make_same_tile_roles_task(2))
    assert result.completed, result.reason
    assert result.pmu.events.get("uce_context_switch", 0) > 0
    assert result.pmu.named_cycles.get("task_accept", 0) > 0

  def test_runtime_context_count_three_overlaps_three_roles(self):
    sim = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", context_count=3, max_cycles=10000), enable_tracer=True
    )
    result = sim.run(make_same_tile_roles_task(3))
    assert result.completed, result.reason
    assert result.tracer is not None
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    peak = max(e["args"]["active_context_count"] for e in events if e.get("name") == "active_context_count")
    assert peak == 3

  def test_context_count_bounds(self):
    with pytest.raises(ValueError, match="context_count must be between 1 and 8"):
      SimConfig(context_count=0)
    with pytest.raises(ValueError, match="context_count must be between 1 and 8"):
      SimConfig(context_count=9)
    with pytest.raises(ValueError, match="context_count must be between 1 and 8"):
      TileUCE(0, HardwareConfig(), context_count=9)

  def test_runtime_snapshot_exposes_real_allocators(self):
    """runtime uses real HBM/L2/L1 handles, so only timing_only may expose
    allocator fields as None."""
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = Simulator(hw, SimConfig(fidelity="runtime", max_cycles=200000))
    result = sim.run(PowWorkload().module, input_bindings=POW_BINDINGS)
    assert result.completed, result.reason
    mem = result.group_snapshot["memory"]
    assert mem["fidelity"] == "runtime"
    assert mem["hbm"] is not None
    assert mem["hbm"]["external_bindings"] == 1
    assert mem["l2"] is not None
    assert mem["l2"]["peak_allocated_bytes"] > 0
    assert mem["l2"]["live_allocations"] == 0
    assert mem["noc"] is None  # contention fabric only in full_memory
    for tile in mem["l1"].values():
      assert tile["allocator"] is not None
      assert tile["allocator"]["peak_allocated_bytes"] > 0
      assert tile["allocator"]["live_allocations"] == 0

  def test_model_second_run_resets_l2_generation(self):
    """runtime model fresh reset clears prior live extents and makes old L2
    handles stale before admitting the second run."""
    sim = Simulator(
      HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10),
      SimConfig(fidelity="runtime", device_context_count=2, max_cycles=200000),
    )
    module = make_two_context_model()
    first = sim.run(module, input_bindings=MODEL_BINDINGS)
    assert first.completed, first.reason
    assert sim.group.l2_sram.snapshot()["live_allocations"] == 0
    second = sim.run(module, input_bindings=MODEL_BINDINGS)
    assert second.completed, second.reason
    assert sim.group.l2_sram.snapshot()["live_allocations"] == 0

  def test_dispatch_pinned_same_context_serializes(self):
    """Two roles pinned to the same context serialize: zero context switches."""
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", context_count=2, max_cycles=10000))
    result = sim.run(make_same_tile_roles_task(2, pins=[0, 0]))
    assert result.completed, result.reason
    assert result.pmu.events.get("uce_context_switch", 0) == 0

  def test_dispatch_pinned_context_binds_requested_index(self):
    """Pinned dispatch lands on the requested tile-local context index."""
    sim = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", context_count=2, max_cycles=10000), enable_tracer=True
    )
    result = sim.run(make_same_tile_roles_task(2, pins=[1, 1]))
    assert result.completed, result.reason
    assert result.tracer is not None
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    dispatch_ctxs = [e["args"]["ctx_id"] for e in events if e.get("name") == "tile_role_dispatch"]
    assert dispatch_ctxs == [1, 1]

  def test_dispatch_pinned_context_out_of_range_fails_at_load(self):
    """Out-of-range context pin fails fast at task load, not silent deadlock."""
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", context_count=2, max_cycles=10000))
    with pytest.raises(ValueError, match="pins context 2 but context_count is 2"):
      sim.run(make_same_tile_roles_task(2, pins=[2, None]))

  def test_same_program_different_pins_make_distinct_roles(self):
    """Same program + mask but different context pins produce distinct roles."""
    prog = make_waiting_mfe_program()
    tasks = NestTaskRangeOp(0, 1)
    buffer = NestAllocOp("l2_buf", "in", L2_WAIT_DIMS, "bf16")
    disp0 = NestDispatchOp(
      "ctx_wait_mfe",
      tasks.result,
      [buffer.result],
      [buffer.result],
      "ev_a",
      "ev_inrel_a",
      "",
      signal_policy={"input_released": "all_tasks"},
      context_id=0,
    )
    disp1 = NestDispatchOp(
      "ctx_wait_mfe",
      tasks.result,
      [buffer.result],
      [buffer.result],
      "ev_b",
      "ev_inrel_b",
      "",
      signal_policy={"input_released": "all_tasks"},
      context_id=1,
    )
    module = ModuleOp(
      [
        prog,
        NestContextOp(
          "same_prog_two_pins",
          [
            buffer,
            tasks,
            disp0,
            disp1,
            NestReleaseOp(buffer.result, depends_on=[disp0.input_released, disp1.input_released]),
            NestAwaitOp([disp0.grid_done, disp1.grid_done]),
            NestReturnOp(),
          ],
          placement=1,
        ),
      ]
    )
    task = lower_workload_ir(module)
    assert sorted(b.context_id for b in task.role_bindings.values()) == [0, 1]
    result = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", context_count=2, max_cycles=10000)
    ).run(module)
    assert result.completed, result.reason


# ---------------------------------------------------------------------------
# Model mode (nexus.program + device slot scheduling)
# ---------------------------------------------------------------------------


class TestModelMode:
  @staticmethod
  def _submit_done_cycles(result):
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    submits = {
      e["args"]["context"]: e["args"]["cycle"] for e in events if e.get("name") == "context_submit"
    }
    dones = {e["args"]["context"]: e["args"]["cycle"] for e in events if e.get("name") == "context_done"}
    slots = {e["args"]["context"]: e["args"]["slot"] for e in events if e.get("name") == "context_submit"}
    return submits, dones, slots

  def test_two_contexts_run_concurrently_on_two_slots(self):
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=2, max_cycles=10000),
      enable_tracer=True,
    )
    result = sim.run(make_two_context_model(), input_bindings=MODEL_BINDINGS)
    assert result.completed, result.reason
    submits, dones, slots = self._submit_done_cycles(result)
    assert sorted(slots.values()) == [0, 1]
    assert submits["ctx1"] < dones["ctx0"], (submits, dones)

  def test_backpressure_serializes_on_one_slot(self):
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=1, max_cycles=10000),
      enable_tracer=True,
    )
    result = sim.run(make_two_context_model(), input_bindings=MODEL_BINDINGS)
    assert result.completed, result.reason
    submits, dones, _ = self._submit_done_cycles(result)
    assert submits["ctx1"] > dones["ctx0"], (submits, dones)
    assert result.pmu.named_cycles.get("device_submit_wait", 0) > 0

  def test_context_pin_selects_slot(self):
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=2, max_cycles=10000),
      enable_tracer=True,
    )
    result = sim.run(make_two_context_model(pins=(1, 0)), input_bindings=MODEL_BINDINGS)
    assert result.completed, result.reason
    _, _, slots = self._submit_done_cycles(result)
    assert slots == {"ctx0": 1, "ctx1": 0}

  def test_pin_out_of_range_fails_at_load(self):
    sim = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", device_context_count=2, max_cycles=10000)
    )
    with pytest.raises(ValueError, match="pins device context 2 but device_context_count is 2"):
      sim.run(make_two_context_model(pins=(2, None)), input_bindings=MODEL_BINDINGS)

  def test_model_fault_waits_for_reset_cleanup(self):
    """Model-mode admission fault freezes device submits and returns only
    after reset cleanup released all context-owned memory."""
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10, group_sram_bytes=1024)
    sim = Simulator(hw, SimConfig(fidelity="full_memory", device_context_count=2, max_cycles=10000))
    result = sim.run(make_two_context_model(), input_bindings=MODEL_BINDINGS)
    assert not result.completed
    assert "L2 capacity fault" in result.reason
    assert sim.group.reset_domain.is_done
    mem = result.group_snapshot["memory"]
    assert mem["transfers"]["inflight"] == 0
    assert mem["l2"]["live_allocations"] == 0
    for tile in mem["l1"].values():
      assert tile["allocator"]["live_allocations"] == 0

  def test_legacy_module_rejects_out_of_range_context_pin(self):
    prog = make_waiting_mfe_program()
    tasks = NestTaskRangeOp(0, 1)
    buffer = NestAllocOp("l2_buf", "in", L2_WAIT_DIMS, "bf16")
    disp = NestDispatchOp(
      "ctx_wait_mfe",
      tasks.result,
      [buffer.result],
      [buffer.result],
      "ev_a",
      "ev_inrel_a",
      "",
      signal_policy={"input_released": "all_tasks"},
    )
    module = ModuleOp(
      [
        prog,
        NestContextOp(
          "legacy_pinned",
          [
            buffer,
            tasks,
            disp,
            NestReleaseOp(buffer.result, depends_on=[disp.input_released]),
            NestAwaitOp([disp.grid_done]),
            NestReturnOp(),
          ],
          placement=1,
          context_id=1,
        ),
      ]
    )
    sim = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", device_context_count=1, max_cycles=10000)
    )
    with pytest.raises(ValueError, match="pins device context 1 but device_context_count is 1"):
      sim.run(module)

  def test_sequential_slot_reuse_gets_fresh_launch_namespace(self):
    """Submitting the same context twice on one slot must not alias
    stale completions from the first launch (launch-ID namespacing)."""
    prog = make_waiting_mfe_program("model_wait_mfe")
    buffer = NestAllocOp("l2_buf_c0", "in", L2_WAIT_DIMS, "bf16")
    tasks = NestTaskRangeOp(0, 1)
    disp = NestDispatchOp(
      "model_wait_mfe",
      tasks.result,
      [buffer.result],
      [buffer.result],
      "ev_grid_c0",
      "ev_inrel_c0",
      "",
      signal_policy={"input_released": "all_tasks"},
    )
    ctx = NestContextOp(
      "ctx0",
      [
        buffer,
        tasks,
        disp,
        NestReleaseOp(buffer.result, depends_on=[disp.input_released]),
        NestAwaitOp([disp.grid_done]),
        NestReturnOp(),
      ],
      placement=1,
    )
    sub0 = NexusSubmitContextOp("ctx0", "done_c0")
    sub1 = NexusSubmitContextOp("ctx0", "done_c0_1")
    program = NexusProgramOp(
      "run_model", [sub0, NexusAwaitOp([sub0.result]), sub1, NexusAwaitOp([sub1.result]), NexusReturnOp()]
    )
    module = ModuleOp([prog, ctx, program])
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=1, max_cycles=10000),
      enable_tracer=True,
    )
    result = sim.run(module)
    assert result.completed, result.reason
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    submit_cycles = [e["args"]["cycle"] for e in events if e.get("name") == "context_submit"]
    done_cycles = [e["args"]["cycle"] for e in events if e.get("name") == "context_done"]
    assert len(submit_cycles) == 2 and len(done_cycles) == 2, (submit_cycles, done_cycles)
    # both submissions on slot 0; the second starts after the first completes
    assert sorted(e["args"]["slot"] for e in events if e.get("name") == "context_submit") == [0, 0]
    assert submit_cycles[1] >= done_cycles[0], (submit_cycles, done_cycles)

  def test_concurrent_contexts_with_same_stream_ids_namespaced(self):
    """Two concurrent contexts using the same original stream queue IDs
    must get slot/launch-namespaced queues: both complete, credit
    invariants hold, and two distinct queues exist (no overwrite)."""

    def make_stream_task(name: str) -> ExecTileGroupTask:
      prog = ExecTileProgram(
        name=f"{name}_prog",
        insts=[
          ExecTileInst(ExecTileOp.STREAM_ACQUIRE, dst="tok0", args=(0,)),
          ExecTileInst(ExecTileOp.STREAM_PUSH, args=(0, "tok0", 0)),
          ExecTileInst(ExecTileOp.STREAM_POP, dst="tok1", args=(0,)),
          ExecTileInst(ExecTileOp.STREAM_RELEASE, args=(0, "tok1")),
          ExecTileInst(ExecTileOp.RET),
        ],
      )
      return ExecTileGroupTask(
        name=name,
        actions=[
          ExecGroupAction(ExecGroupActionOp.INIT_STREAM, args=(0, 1, 1, 1)),
          ExecGroupAction(
            ExecGroupActionOp.DISPATCH_ROLE,
            args=(
              ExecDispatchRequest(
                role_id=0,
                dispatch_ordinal=0,
                signal_policy=ExecSignalPolicy(None, None),
                input_released_event="",
                output_ready_event="",
              ),
            ),
            dst="ev_grid",
          ),
          ExecGroupAction(ExecGroupActionOp.WAIT_EVENT, args=("ev_grid",)),
        ],
        streams=[ExecStreamDesc(queue_id=0, depth=1, producer_mask=1, consumer_mask=1)],
        role_bindings={
          0: ExecTileRoleBinding(role_id=0, tile_mask=1, tile_program=prog, in_stream=0, out_stream=0)
        },
      )

    sim = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", device_context_count=2, max_cycles=10000)
    )
    seq0 = sim.group.load_context_task(make_stream_task("ctx0"), slot_index=0)
    seq1 = sim.group.load_context_task(make_stream_task("ctx1"), slot_index=1)
    seen_qids: set[int] = set()
    for cycle in range(10000):
      seen_qids |= set(sim.group.queues.keys())
      if sim.group.step(cycle):
        break
    assert seq0.done and seq1.done, f"seq0={seq0.done} seq1={seq1.done}"
    assert not seq0.faulted and not seq1.faulted
    # Two distinct namespaced queues existed while both launches ran:
    # launch 0/slot 0 keeps qid 0; launch 1/slot 1 offsets to 1_010_000.
    # Without the rewrite the second INIT_STREAM would overwrite the
    # first queue (only qid 0 would ever be seen).
    assert 0 in seen_qids and 1_010_000 in seen_qids, seen_qids
    # After both sequencers drain, their queues and tile bindings are
    # reclaimed (no unbounded growth across sequential submits).
    assert sim.group.queues == {}, sim.group.queues
    assert all(t.streams == {} for t in sim.group.tiles), [t.streams for t in sim.group.tiles]
    assert sim.group.credit_invariants_hold()

  def test_sequential_stream_reuse_reclaims_queues(self):
    """Repeatedly submitting a stream-bearing context on one slot must
    reclaim each launch's queues on drain (no queue/binding growth)."""

    def make_stream_task(name: str) -> ExecTileGroupTask:
      prog = ExecTileProgram(
        name=f"{name}_prog",
        insts=[
          ExecTileInst(ExecTileOp.STREAM_ACQUIRE, dst="tok0", args=(0,)),
          ExecTileInst(ExecTileOp.STREAM_PUSH, args=(0, "tok0", 0)),
          ExecTileInst(ExecTileOp.STREAM_POP, dst="tok1", args=(0,)),
          ExecTileInst(ExecTileOp.STREAM_RELEASE, args=(0, "tok1")),
          ExecTileInst(ExecTileOp.RET),
        ],
      )
      return ExecTileGroupTask(
        name=name,
        actions=[
          ExecGroupAction(ExecGroupActionOp.INIT_STREAM, args=(0, 1, 1, 1)),
          ExecGroupAction(
            ExecGroupActionOp.DISPATCH_ROLE,
            args=(
              ExecDispatchRequest(
                role_id=0,
                dispatch_ordinal=0,
                signal_policy=ExecSignalPolicy(None, None),
                input_released_event="",
                output_ready_event="",
              ),
            ),
            dst="ev_grid",
          ),
          ExecGroupAction(ExecGroupActionOp.WAIT_EVENT, args=("ev_grid",)),
        ],
        streams=[ExecStreamDesc(queue_id=0, depth=1, producer_mask=1, consumer_mask=1)],
        role_bindings={
          0: ExecTileRoleBinding(role_id=0, tile_mask=1, tile_program=prog, in_stream=0, out_stream=0)
        },
      )

    sim = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", device_context_count=1, max_cycles=10000)
    )
    for round_idx in range(3):
      seq = sim.group.load_context_task(make_stream_task(f"ctx{round_idx}"), slot_index=0)
      for cycle in range(10000):
        if sim.group.step(cycle):
          break
      assert seq.done and not seq.faulted
      # After each drain, queues and tile bindings are fully reclaimed.
      assert sim.group.queues == {}, sim.group.queues
      assert all(t.streams == {} for t in sim.group.tiles), [t.streams for t in sim.group.tiles]
      assert sim.group.credit_invariants_hold()

  def test_drain_gate_holds_until_unawaited_role_completes(self):
    """A context whose actions end without awaiting its dispatch must
    not finish (and must not reclaim its stream queues) until the
    launched role's tile program completes (IR_SPEC §3.10)."""
    prog = ExecTileProgram(
      name="stream_prog",
      insts=[
        ExecTileInst(ExecTileOp.STREAM_ACQUIRE, dst="tok0", args=(0,)),
        ExecTileInst(ExecTileOp.STREAM_PUSH, args=(0, "tok0", 0)),
        ExecTileInst(ExecTileOp.STREAM_POP, dst="tok1", args=(0,)),
        ExecTileInst(ExecTileOp.STREAM_RELEASE, args=(0, "tok1")),
        ExecTileInst(ExecTileOp.RET),
      ],
    )
    task = ExecTileGroupTask(
      name="ctx0",
      actions=[
        ExecGroupAction(ExecGroupActionOp.INIT_STREAM, args=(0, 1, 1, 1)),
        # no WAIT_EVENT after the dispatch: actions end while the role
        # is still running on the tile.
        ExecGroupAction(
          ExecGroupActionOp.DISPATCH_ROLE,
          args=(
            ExecDispatchRequest(
              role_id=0,
              dispatch_ordinal=0,
              signal_policy=ExecSignalPolicy(None, None),
              input_released_event="",
              output_ready_event="",
            ),
          ),
          dst="ev_grid",
        ),
      ],
      streams=[ExecStreamDesc(queue_id=0, depth=1, producer_mask=1, consumer_mask=1)],
      role_bindings={
        0: ExecTileRoleBinding(role_id=0, tile_mask=1, tile_program=prog, in_stream=0, out_stream=0)
      },
    )
    sim = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", device_context_count=1, max_cycles=10000)
    )
    seq = sim.group.load_context_task(task, slot_index=0)
    # Step until the sequencer exhausts its actions while the role is
    # still running: the drain gate must hold it not-done and keep its
    # queues alive.
    gate_held = False
    for cycle in range(10000):
      if sim.group.step(cycle):
        break
      if seq.action_index >= len(task.actions) and not seq.done:
        assert sim.group.queues != {}, "queues reclaimed before role drained"
        gate_held = True
        break
    assert gate_held, "sequencer never hit the end-of-actions drain gate"
    # Drain to completion: role finishes, queues reclaimed.
    for drain_cycle in range(cycle + 1, 10000):
      if sim.group.step(drain_cycle):
        break
    assert seq.done and not seq.faulted
    assert sim.group.queues == {}, sim.group.queues
    assert all(t.streams == {} for t in sim.group.tiles), [t.streams for t in sim.group.tiles]
    assert sim.group.credit_invariants_hold()

  # -----------------------------------------------------------------------
  # Input binding contract tests (PR 1, §2.5 / §3 Step 5)
  # -----------------------------------------------------------------------

  def test_missing_binding_fails(self):
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", max_cycles=10000))
    with pytest.raises(ValueError, match="missing input binding for global 'Y0'"):
      sim.run(make_two_context_model(), input_bindings={})

  def test_unused_binding_fails(self):
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", max_cycles=10000))
    bindings = {**MODEL_BINDINGS, "ZZ": GlobalBinding("ZZ", 0x300000, 1024, "rw")}
    with pytest.raises(ValueError, match="input binding 'ZZ' does not match any program input"):
      sim.run(make_two_context_model(), input_bindings=bindings)

  def test_binding_too_small_fails(self):
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", max_cycles=10000))
    bindings = {
      "Y0": GlobalBinding("Y0", 0x100000, 64, "rw"),
      "Y1": GlobalBinding("Y1", 0x200000, L2_WAIT_BYTES, "rw"),
    }
    with pytest.raises(ValueError, match="input binding 'Y0' size 64 is smaller than required"):
      sim.run(make_two_context_model(), input_bindings=bindings)

  def test_binding_overlap_fails(self):
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", max_cycles=10000))
    bindings = {
      "Y0": GlobalBinding("Y0", 0x100000, L2_WAIT_BYTES, "rw"),
      "Y1": GlobalBinding("Y1", 0x100000, L2_WAIT_BYTES, "rw"),
    }
    with pytest.raises(ValueError, match="input bindings 'Y0' and 'Y1' overlap"):
      sim.run(make_two_context_model(), input_bindings=bindings)

  def test_binding_exceeds_hbm_capacity_fails(self):
    hw = HardwareConfig()
    sim = Simulator(hw, SimConfig(fidelity="runtime", max_cycles=10000))
    cap = hw.hbm_capacity_bytes
    bindings = {
      "Y0": GlobalBinding("Y0", cap, L2_WAIT_BYTES, "rw"),
      "Y1": GlobalBinding("Y1", 0x100000, L2_WAIT_BYTES, "rw"),
    }
    with pytest.raises(ValueError, match="input binding 'Y0' exceeds HBM capacity"):
      sim.run(make_two_context_model(), input_bindings=bindings)

  def test_readonly_binding_rejects_store(self):
    """A read-only binding used as a store destination is rejected."""
    from pipeline_validator.dialects.elenor import (
      NestDispatchOp,
      NestDMAStoreOp,
      NestPrefetchOp,
      NestReleaseOp,
    )

    prog = make_pow_tile_program()
    ctx = NestContextOp(
      "pow_task", [], arg_types=[NestGlobalMemref.of([4, 128, 128], "bf16")], arg_names=["Y"], placement=15
    )
    y_arg = ctx.body.block.args[0]
    buf = NestAllocOp("l2_buf", "inout", [4, 128, 128], "bf16", alignment=256)
    src = NestSubviewOp(
      y_arg, [0, 0, 0], [4, 128, 128], [1, 1, 1], NestGlobalView.of([4, 128, 128], "bf16")
    )
    pref = NestPrefetchOp(src.result, buf.result, "ev_in")
    tasks = NestTaskRangeOp(0, 4)
    disp = NestDispatchOp(
      "pow_4k_tile",
      tasks.result,
      [buf.result],
      [buf.result],
      "ev_grid",
      "ev_inrel",
      "ev_outready",
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=[pref.result],
    )
    store = NestDMAStoreOp(buf.result, src.result, "ev_out", depends_on=[disp.output_ready])
    release = NestReleaseOp(buf.result, depends_on=[store.result])
    ctx.body.block.add_ops(
      [
        buf,
        src,
        pref,
        tasks,
        disp,
        store,
        release,
        NestAwaitOp([disp.grid_done, store.result]),
        NestReturnOp(),
      ]
    )
    program = NexusProgramOp(
      "run_pow", [], arg_types=[NestGlobalMemref.of([4, 128, 128], "bf16")], arg_names=["Y0"]
    )
    y0 = program.body.block.args[0]
    sub = NexusSubmitContextOp("pow_task", "done0", actuals=[y0])
    program.body.block.add_ops([sub, NexusAwaitOp([sub.result]), NexusReturnOp()])
    module = ModuleOp([prog, ctx, program])
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", max_cycles=10000))
    with pytest.raises(ValueError, match="is not writable but is used as store destination"):
      sim.run(module, input_bindings={"Y0": GlobalBinding("Y0", 0x100000, 131072, "r")})


class TestGridSignalAggregation:
  """PR 3: grid-scoped phase aggregation with logical task identity.

  These tests step the sim until the dispatch registers its
  ``_GridSignalState``, then inject PhaseSignals directly to prove the
  3/4 barrier, duplicate idempotency, stale-launch rejection and
  cross-context isolation.
  """

  @staticmethod
  def _dispatch_and_get_grid(fidelity="runtime"):
    """Step until one dispatch registers; return (group, seq, grid, state)."""
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = Simulator(hw, SimConfig(fidelity=fidelity, max_cycles=200000))
    group = sim.group
    task = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    group.load_task(task, input_bindings=POW_BINDINGS)
    seq = group.sequencer
    # Step until the dispatch registers a grid signal state.
    for c in range(5000):
      group.step(c)
      if group._grid_signals:
        break
    assert group._grid_signals, "dispatch did not register grid signal state"
    grid = next(iter(group._grid_signals))
    state = group._grid_signals[grid]
    return group, seq, grid, state

  def test_partial_signals_do_not_complete_phase(self):
    """3/4 signals: phase event does not fire; 4th completes exactly-once."""
    from pipeline_validator.execution_ir import PhaseSignal, TaskIdentity

    group, seq, grid, state = self._dispatch_and_get_grid()
    phase_ev = state.phase_event_ids["input_released"]
    # Fire 3 of 4 input_released signals
    for tid in range(3):
      group._on_phase_signal(PhaseSignal(TaskIdentity(grid, tid), "input_released"), 1)
    assert phase_ev not in seq._events_done
    assert "input_released" not in state.completed_phases
    # 4th signal completes exactly-once
    group._on_phase_signal(PhaseSignal(TaskIdentity(grid, 3), "input_released"), 2)
    assert phase_ev in seq._events_done
    assert "input_released" in state.completed_phases
    group.reset()

  def test_duplicate_signal_does_not_advance(self):
    """Same (grid, phase, task) signal is idempotent: duplicate +1."""
    from pipeline_validator.execution_ir import PhaseSignal, TaskIdentity

    group, _seq, grid, _state = self._dispatch_and_get_grid()
    sig = PhaseSignal(TaskIdentity(grid, 0), "input_released")
    group._on_phase_signal(sig, 0)
    group._on_phase_signal(sig, 1)
    assert group.pmu.events.get("tile_signal_duplicate", 0) == 1
    assert "input_released" not in _state.completed_phases
    group.reset()

  def test_stale_launch_signal_ignored(self):
    """Signal for a retired launch only increments tile_signal_stale."""
    from pipeline_validator.execution_ir import GridInstanceId, PhaseSignal, TaskIdentity

    group, _seq, grid, _state = self._dispatch_and_get_grid()
    old_gen = grid.launch_generation
    # Retire the current launch by running to completion, then reload
    for c in range(10000):
      group.step(c)
      if _seq.done:
        break
    task = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    group.load_task(task, input_bindings=POW_BINDINGS)
    # Old-generation signal is stale
    stale_grid = GridInstanceId(grid.context_name, grid.device_slot, old_gen, grid.dispatch_ordinal)
    group._on_phase_signal(PhaseSignal(TaskIdentity(stale_grid, 0), "input_released"), 0)
    assert group.pmu.events.get("tile_signal_stale", 0) == 1
    group.reset()


class TestSignalGatedRelease:
  """PR 3: role-aware release gating and pin lifecycle."""

  def test_exact_capacity_blocks_until_release(self):
    """Exact-capacity L2 (one pow chunk = 131072 bytes): first batch
    admits and holds all L2; second admit_l2_buffers must fail."""
    from pipeline_validator.tile_group import TileGroup

    hw = HardwareConfig().with_overrides(group_sram_bytes=4 * 128 * 128 * 2)
    group = TileGroup(hw, fidelity="full_memory")
    task1 = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    task2 = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    group.load_task(task1, input_bindings=POW_BINDINGS)
    # L2 is exactly full; a second admission must fail
    assert not group.admit_l2_buffers(task2, cycle=0)
    assert group.l2_sram.snapshot()["free_bytes"] == 0
    group.reset()

  def test_exact_capacity_retries_after_3_4_barrier_and_release(self):
    """Exact-capacity L2: after 3/4 signals, second admit returns False
    (pins hold L2). After 4th signal + legal release, retry returns True."""
    from pipeline_validator.execution_ir import PhaseSignal, TaskIdentity
    from pipeline_validator.tile_group import TileGroup

    chunk = 4 * 128 * 128 * 2  # one pow allocation = 131072 bytes
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10, group_sram_bytes=chunk)
    group = TileGroup(hw, fidelity="full_memory")
    task = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    group.load_task(task, input_bindings=POW_BINDINGS)
    seq = group.sequencer
    # Step until the dispatch registers a grid signal state.
    for c in range(5000):
      group.step(c)
      if group._grid_signals:
        break
    assert group._grid_signals, "dispatch did not register grid signal state"
    grid = next(iter(group._grid_signals))
    # Fire only 3 of 4 input_released signals — phase not complete,
    # pins still hold L2.
    for tid in range(3):
      group._on_phase_signal(PhaseSignal(TaskIdentity(grid, tid), "input_released"), c + 1)
    task2 = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    # Second admission must fail: L2 is pinned by the first batch.
    group._context_launch_generation += 1
    assert not group.admit_l2_buffers(task2, cycle=c + 2)
    group._context_launch_generation -= 1  # restore for retry
    # Complete the 4th signal + all output_ready, then step until
    # the RELEASE_L2 action unpins and releases the handle.
    group._on_phase_signal(PhaseSignal(TaskIdentity(grid, 3), "input_released"), c + 3)
    for tid in range(4):
      group._on_phase_signal(PhaseSignal(TaskIdentity(grid, tid), "output_ready"), c + 4)
    for c2 in range(c + 5, c + 20000):
      group.step(c2)
      if seq.done:
        break
    assert seq.done, "sequencer did not complete after release"
    assert group.l2_sram.snapshot()["live_allocations"] == 0
    # Retry: L2 is now free; second admission must succeed.
    group._context_launch_generation += 1
    assert group.admit_l2_buffers(task2, cycle=c2)
    assert group.l2_sram.snapshot()["live_allocations"] == 1
    group.reset()

  def test_trace_tile_signal_count_and_args(self):
    """Tracer-enabled dual-context run: tile_signal count ==
    context_count * task_count * phase_count; every event has 8 args."""
    sim = Simulator(
      HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10),
      SimConfig(fidelity="full_memory", device_context_count=2, max_cycles=200000),
      enable_tracer=True,
    )
    result = sim.run(parse_workload_ir(open("examples/example.mlir").read()), input_bindings=MODEL_BINDINGS)
    assert result.completed, result.reason
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    signals = [e for e in events if e.get("name") == "tile_signal"]
    assert len(signals) == 16
    required = {
      "context_name",
      "device_slot",
      "launch_generation",
      "dispatch_ordinal",
      "task_id",
      "phase",
      "tile_id",
      "hardware_context_id",
    }
    grids = {}
    for e in signals:
      assert required <= e["args"].keys()
      g = (
        e["args"]["context_name"],
        e["args"]["device_slot"],
        e["args"]["launch_generation"],
        e["args"]["dispatch_ordinal"],
      )
      grids.setdefault(g, {}).setdefault(e["args"]["phase"], set()).add(e["args"]["task_id"])
    assert len(grids) == 2
    for phases in grids.values():
      assert phases == {"input_released": {0, 1, 2, 3}, "output_ready": {0, 1, 2, 3}}
    ev = result.pmu.events
    for key in (
      "tile_signal_duplicate",
      "tile_signal_stale",
      "tile_signal_invalid",
      "release_invariant_fault",
    ):
      assert ev.get(key, 0) == 0, (key, ev)


class TestReleaseFaultPath:
  """PR 3 §5: wrong-owner / double RELEASE_L2 through the sequencer
  produces ADDRESS_FAULT + fault ring + ResetDomain zero-leak."""

  @staticmethod
  def _step_until_action(group, seq, target_op):
    """Step until the sequencer's next action is ``target_op``."""
    from pipeline_validator.execution_ir import ExecGroupActionOp

    for c in range(50000):
      group.step(c)
      if seq.done or seq.faulted:
        return c
      if seq.action_index < len(seq.task.actions):
        if seq.task.actions[seq.action_index].op == target_op:
          return c
    raise AssertionError(f"sequencer never reached {target_op}")

  @staticmethod
  def _fire_all_signals(group, grid):
    from pipeline_validator.execution_ir import PhaseSignal, TaskIdentity

    for tid in range(4):
      group._on_phase_signal(PhaseSignal(TaskIdentity(grid, tid), "input_released"), 0)
    for tid in range(4):
      group._on_phase_signal(PhaseSignal(TaskIdentity(grid, tid), "output_ready"), 0)

  @staticmethod
  def _assert_zero_leak(group):
    assert not group._grid_l2_pins
    assert not group._grid_signals
    assert group.l2_sram.snapshot()["live_allocations"] == 0
    assert group.l2_sram.snapshot()["pending_release"] == 0
    for tile in group.tiles:
      assert tile.l1_allocator.snapshot()["live_allocations"] == 0

  def test_unknown_buffer_release_faults_and_resets(self):
    """RELEASE_L2 for a non-existent buffer slot: sequencer catches
    MemoryInvariantError, writes ADDRESS_FAULT, starts reset/drain;
    after cleanup all state returns to zero-leak."""
    from pipeline_validator.execution_ir import ExecGroupActionOp, ExecReleaseRequest
    from pipeline_validator.memory.allocator import MemoryInvariantError
    from pipeline_validator.tile_group import TileGroup

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    group = TileGroup(hw, fidelity="runtime")
    task = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    group.load_task(task, input_bindings=POW_BINDINGS)
    seq = group.sequencer
    # Step until dispatch registers, then fire all signals.
    for c in range(5000):
      group.step(c)
      if group._grid_signals:
        break
    grid = next(iter(group._grid_signals))
    self._fire_all_signals(group, grid)
    # Find the RELEASE_L2 action and step until it is ready to issue
    # (deps satisfied, _pending cleared, action_index pointing at it).
    rel_idx = next(i for i, a in enumerate(seq.task.actions) if a.op == ExecGroupActionOp.RELEASE_L2)
    for c2 in range(c + 1, c + 5000):
      group.step(c2)
      if seq.faulted or seq.done:
        break
      if (
        seq.action_index == rel_idx
        and seq._pending is None
        and all(ev in seq._events_done for ev in seq.task.actions[rel_idx].args[0].dependency_events)
      ):
        break
    assert not seq.faulted, f"premature fault: {seq.fault_reason}"
    # Mutate the release request to reference a non-existent slot.
    original_req = seq.task.actions[rel_idx].args[0]
    bad_req = ExecReleaseRequest(
      buffer_slot="nonexistent",
      buffer_role=original_req.buffer_role,
      consumer_dispatch_ordinals=original_req.consumer_dispatch_ordinals,
      dependency_events=original_req.dependency_events,
    )
    seq.task.actions[rel_idx] = type(seq.task.actions[rel_idx])(
      ExecGroupActionOp.RELEASE_L2, args=(bad_req,)
    )
    # Step one more cycle — RELEASE_L2 issues, catches, faults.
    group.step(c2 + 1)
    assert seq.faulted
    assert "release invariant fault" in seq.fault_reason
    assert group.pmu.events.get("release_invariant_fault", 0) >= 1
    if group.runtime_enabled:
      assert group.fault_ring.snapshot()["count"] > 0
      assert group.fault_ring.snapshot()["latest_code"] == FaultCode.ADDRESS_FAULT.name
    # Step until reset cleanup completes.
    for c3 in range(c2 + 2, c2 + 5000):
      group.step(c3)
      if group.reset_domain.is_done:
        break
    assert group.reset_domain.is_done
    self._assert_zero_leak(group)
    group.reset()

  def test_double_release_faults_and_resets(self):
    """A second RELEASE_L2 for an already-released buffer hits the
    allocator's double-release check; sequencer catches, faults, and
    reset restores zero-leak."""
    from dataclasses import replace
    from pipeline_validator.execution_ir import ExecGroupAction, ExecGroupActionOp
    from pipeline_validator.tile_group import TileGroup

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    group = TileGroup(hw, fidelity="runtime")
    task = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    # Append a duplicate RELEASE_L2 action after the first one.
    first_release_idx = next(i for i, a in enumerate(task.actions) if a.op == ExecGroupActionOp.RELEASE_L2)
    dup_action = task.actions[first_release_idx]
    task.actions.insert(
      first_release_idx + 1, ExecGroupAction(ExecGroupActionOp.RELEASE_L2, args=dup_action.args)
    )
    group.load_task(task, input_bindings=POW_BINDINGS)
    seq = group.sequencer
    # Step until dispatch, fire signals, step until fault.
    for c in range(5000):
      group.step(c)
      if group._grid_signals:
        break
    grid = next(iter(group._grid_signals))
    self._fire_all_signals(group, grid)
    # Step until the second RELEASE_L2 faults.
    for c2 in range(c + 1, c + 5000):
      group.step(c2)
      if seq.faulted:
        break
    assert seq.faulted
    assert "release invariant fault" in seq.fault_reason
    assert group.pmu.events.get("release_invariant_fault", 0) >= 1
    # Step until reset cleanup completes.
    for c3 in range(c2 + 1, c2 + 5000):
      group.step(c3)
      if group.reset_domain.is_done:
        break
    assert group.reset_domain.is_done
    self._assert_zero_leak(group)
    group.reset()

  def test_wrong_owner_release_faults_and_resets(self):
    """RELEASE_L2 with a mismatched context_name: assert_live raises
    wrong-owner; sequencer catches, writes ADDRESS_FAULT, resets to
    zero-leak."""
    from pipeline_validator.execution_ir import ExecGroupActionOp
    from pipeline_validator.tile_group import TileGroup

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    group = TileGroup(hw, fidelity="runtime")
    task = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    group.load_task(task, input_bindings=POW_BINDINGS)
    seq = group.sequencer
    for c in range(5000):
      group.step(c)
      if group._grid_signals:
        break
    grid = next(iter(group._grid_signals))
    self._fire_all_signals(group, grid)
    rel_idx = next(i for i, a in enumerate(seq.task.actions) if a.op == ExecGroupActionOp.RELEASE_L2)
    for c2 in range(c + 1, c + 5000):
      group.step(c2)
      if seq.faulted or seq.done:
        break
      if (
        seq.action_index == rel_idx
        and seq._pending is None
        and all(ev in seq._events_done for ev in seq.task.actions[rel_idx].args[0].dependency_events)
      ):
        break
    assert not seq.faulted, f"premature fault: {seq.fault_reason}"
    # Corrupt the sequencer's context_name so the owner check fails.
    seq.context_name = "wrong_owner_ctx"
    group.step(c2 + 1)
    assert seq.faulted
    assert "release invariant fault" in seq.fault_reason
    assert group.pmu.events.get("release_invariant_fault", 0) >= 1
    if group.runtime_enabled:
      assert group.fault_ring.snapshot()["count"] > 0
      assert group.fault_ring.snapshot()["latest_code"] == FaultCode.ADDRESS_FAULT.name
    for c3 in range(c2 + 2, c2 + 5000):
      group.step(c3)
      if group.reset_domain.is_done:
        break
    assert group.reset_domain.is_done
    self._assert_zero_leak(group)
    group.reset()

  def test_stale_generation_release_faults_and_resets(self):
    """RELEASE_L2 with a stale context_launch_generation: the
    generation-keyed handle lookup returns None; sequencer catches,
    writes ADDRESS_FAULT, resets to zero-leak."""
    from pipeline_validator.execution_ir import ExecGroupActionOp
    from pipeline_validator.tile_group import TileGroup

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    group = TileGroup(hw, fidelity="runtime")
    task = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    group.load_task(task, input_bindings=POW_BINDINGS)
    seq = group.sequencer
    for c in range(5000):
      group.step(c)
      if group._grid_signals:
        break
    grid = next(iter(group._grid_signals))
    self._fire_all_signals(group, grid)
    rel_idx = next(i for i, a in enumerate(seq.task.actions) if a.op == ExecGroupActionOp.RELEASE_L2)
    for c2 in range(c + 1, c + 5000):
      group.step(c2)
      if seq.faulted or seq.done:
        break
      if (
        seq.action_index == rel_idx
        and seq._pending is None
        and all(ev in seq._events_done for ev in seq.task.actions[rel_idx].args[0].dependency_events)
      ):
        break
    assert not seq.faulted, f"premature fault: {seq.fault_reason}"
    # Corrupt the sequencer's launch generation so the handle
    # lookup misses the stored (gen, slot) key.
    seq.context_launch_generation = seq.context_launch_generation + 999
    group.step(c2 + 1)
    assert seq.faulted
    assert "release invariant fault" in seq.fault_reason
    assert group.pmu.events.get("release_invariant_fault", 0) >= 1
    if group.runtime_enabled:
      assert group.fault_ring.snapshot()["count"] > 0
      assert group.fault_ring.snapshot()["latest_code"] == FaultCode.ADDRESS_FAULT.name
    for c3 in range(c2 + 2, c2 + 5000):
      group.step(c3)
      if group.reset_domain.is_done:
        break
    assert group.reset_domain.is_done
    self._assert_zero_leak(group)
    group.reset()
