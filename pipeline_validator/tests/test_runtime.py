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
  NestContextOp,
  NestDispatchOp,
  NestReturnOp,
  NestTaskRangeOp,
  NexusAwaitOp,
  NexusProgramOp,
  NexusReturnOp,
  NexusSubmitContextOp,
  TileAwaitOp,
  TileEvuOp,
  TileLoadOp,
  TileProgramDefOp,
  TileReturnOp,
)
from pipeline_validator.execution_ir import (
  ExecGroupAction,
  ExecGroupActionOp,
  ExecStreamDesc,
  ExecTileGroupTask,
  ExecTileInst,
  ExecTileOp,
  ExecTileProgram,
  ExecTileRoleBinding,
)
from pipeline_validator.ir_lowering import lower_workload_ir
from pipeline_validator.memory import L2SRAM, NoCRouter, PayloadTracker
from pipeline_validator.runtime import EventStatus, EventTable, FaultCode, FaultRing
from pipeline_validator.runtime.fault_ring import FaultDomain, FaultRecord
from pipeline_validator.runtime.reset_domain import ResetDomain, ResetRequest
from pipeline_validator.simulator import Simulator
from pipeline_validator.tile import TileUCE
from pipeline_validator.workload_ir import parse_workload_ir, print_workload_ir
from pipeline_validator.workloads import ALL_WORKLOADS, PowWorkload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sim(fidelity: str = "runtime") -> Simulator:
  hw = HardwareConfig()
  sim = SimConfig(fidelity=fidelity)
  return Simulator(hw, sim)


def make_waiting_mfe_program(name: str = "ctx_wait_mfe") -> TileProgramDefOp:
  load = TileLoadOp(bytes_total=128 * 1024, tag="e_load")
  return TileProgramDefOp(name, [load, TileAwaitOp([load.result]), TileReturnOp()])


def make_short_evu_program(name: str = "ctx_short_evu") -> TileProgramDefOp:
  evu = TileEvuOp(op_name="relu", evu_ops=16, tag="e_evu")
  return TileProgramDefOp(name, [evu, TileAwaitOp([evu.result]), TileReturnOp()])


def make_same_tile_roles_task(
  role_count: int,
  pins: list[int | None] | None = None,
) -> ModuleOp:
  """Dispatch role_count programs to one tile to exercise context switching.

  When ``pins`` is given, dispatch ``i`` is pinned to ``pins[i]``
  (None = no pin).  ``pins`` must have at least ``role_count`` entries.
  """
  names = ["ctx_wait_mfe"] + [f"ctx_short_evu{i}" for i in range(role_count - 1)]
  progs = [make_waiting_mfe_program(names[0])] + [make_short_evu_program(n) for n in names[1:]]
  tasks = NestTaskRangeOp(0, 1)
  buffer = NestAllocOp("l2_buf", 4096)
  dispatches = [
    NestDispatchOp(
      name, tasks.result, buffer.result, buffer.result, f"ev_role{i}", "", "",
      context_id=None if pins is None else pins[i],
    )
    for i, name in enumerate(names)
  ]
  context = NestContextOp(
    "same_tile_roles",
    [buffer, tasks, *dispatches, NestAwaitOp([d.grid_done for d in dispatches]), NestReturnOp()],
    placement=1,
  )
  return ModuleOp([*progs, context])


def make_two_context_model(pins: tuple[int | None, ...] = (None, None)) -> ModuleOp:
  """Two single-dispatch nest.contexts + one nexus.program submitting both."""
  prog = make_waiting_mfe_program("model_wait_mfe")
  ctxs = []
  for i, pin in enumerate(pins):
    buffer = NestAllocOp(f"l2_buf_c{i}", 4096)
    tasks = NestTaskRangeOp(0, 1)
    disp = NestDispatchOp("model_wait_mfe", tasks.result, buffer.result,
                          buffer.result, f"ev_grid_c{i}", "", "")
    ctxs.append(NestContextOp(
      f"ctx{i}",
      [buffer, tasks, disp, NestAwaitOp([disp.grid_done]), NestReturnOp()],
      placement=1,
      context_id=pin,
    ))
  submits = [NexusSubmitContextOp(f"ctx{i}", f"done_c{i}") for i in range(len(pins))]
  program = NexusProgramOp(
    "run_model",
    [*submits, NexusAwaitOp([s.result for s in submits]), NexusReturnOp()],
  )
  return ModuleOp([prog, *ctxs, program])




# ---------------------------------------------------------------------------
# Cold / warm launch (residency)
# ---------------------------------------------------------------------------


class TestRuntimeColdWarm:
  def test_cold_launch_includes_program_load(self):
    """Cold launch's PMU records program_cold_load > 0."""
    s = make_sim("runtime")
    wl = PowWorkload()
    r = s.run(wl.module)
    assert r.completed
    cold = r.pmu.named_cycles.get("program_cold_load", 0)
    assert cold > 0, f"cold launch should record cold_load > 0, got {cold}"

  def test_warm_launch_no_program_reload(self):
    """Second launch of same program: 0 new cold-load cycles."""
    s = make_sim("runtime")
    wl = PowWorkload()
    _r1 = s.run(wl.module)
    c1 = s.group.program_table.cold_load_cycles
    r2 = s.run(wl.module)
    c2 = s.group.program_table.cold_load_cycles
    assert c2 == c1, f"warm should add 0 cold cycles, got delta {c2 - c1}"
    assert r2.completed

  def test_warm_faster_than_cold(self):
    """Warm launch completes in fewer cycles than cold."""
    s = make_sim("runtime")
    wl = PowWorkload()
    r1 = s.run(wl.module)
    r2 = s.run(wl.module)
    assert r2.cycles < r1.cycles, f"warm {r2.cycles} should be < cold {r1.cycles}"

  def test_program_epoch_invalidate_on_group_reset(self):
    """Group reset bumps epoch; next dispatch is cold again."""
    s = make_sim("runtime")
    wl = PowWorkload()
    s.run(wl.module)
    c1 = s.group.program_table.cold_load_cycles
    s.group.program_table.invalidate_group()
    _r2 = s.run(wl.module)
    c2 = s.group.program_table.cold_load_cycles
    assert c2 > c1, "reset should force cold re-install"

  def test_tile_reset_invalidates_residency(self):
    """Per-tile reset makes that tile cold again."""
    s = make_sim("runtime")
    wl = PowWorkload()
    s.run(wl.module)
    c1 = s.group.program_table.cold_load_cycles
    s.group.program_table.invalidate_tile(0)
    _r2 = s.run(wl.module)
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

    r1 = s.run(wl.module)
    assert r1.completed, r1.reason
    after1 = print_workload_ir(wl.module)
    assert after1 == before

    lowered2 = lower_workload_ir(wl.module)
    s._assign_program_ids(lowered2)
    prog2 = lowered2.role_bindings[0].tile_program
    assert prog2.program_id == id1
    assert prog2.program_hash == hash1

    r2 = s.run(wl.module)
    assert r2.completed, r2.reason
    after2 = print_workload_ir(wl.module)
    assert after2 == before
    assert r2.pmu.named_cycles.get("program_cold_load", 0) == 0

    # Mutate a pow descriptor scalar (exponent 2 -> 3) to force hash change
    mutated_text = before.replace("exponent = 2 ops = 65536", "exponent = 3 ops = 65536", 1)
    mutated_module = parse_workload_ir(mutated_text, source_name="<mutated>")
    lowered3 = lower_workload_ir(mutated_module)
    s._assign_program_ids(lowered3)
    prog3 = lowered3.role_bindings[0].tile_program
    assert prog3.program_id == id1
    assert prog3.program_hash != hash1

    cold_before = s.group.program_table.cold_load_cycles
    r3 = s.run(mutated_module)
    assert r3.completed, r3.reason
    cold_after = s.group.program_table.cold_load_cycles
    assert cold_after > cold_before, "changed descriptor scalar should force cold install"


# ---------------------------------------------------------------------------
# Global DMA channel allocation
# ---------------------------------------------------------------------------


class TestDMAChannelScheduling:
  def test_two_channels_round_robin_dma_stores(self):
    """DMA stores must advance the shared round-robin channel selector."""
    sim = Simulator(
      HardwareConfig(num_dma_channels=2),
      SimConfig(fidelity="runtime", max_cycles=200_000),
      enable_tracer=True,
    )
    result = sim.run(PowWorkload().module)
    assert result.completed, result.reason
    assert result.tracer is not None
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    store_channels = [
      event["args"]["channel"]
      for event in events
      if event.get("name", "").startswith("dma.store:")
    ]
    assert store_channels == [0, 1, 0, 1]


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
    s.run(wl.module)
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

  def test_l2_capacity_fault_terminates_task(self):
    """A tiny L2 SRAM triggers a capacity fault on prefetch."""
    hw = HardwareConfig()
    hw = hw.with_overrides(group_sram_bytes=1024)  # 1 KB — too small
    sim = SimConfig(fidelity="full_memory", max_cycles=10000)
    s = Simulator(hw, sim)
    wl = PowWorkload()
    r = s.run(wl.module)
    assert not r.completed
    assert "faulted" in r.reason

  def test_l2_exact_capacity_completes_and_overshoot_faults(self):
    """4 pow chunks (4 x 128 KiB per chunk) fit exactly in a 512 KiB L2
    and complete; a 5th chunk overshoots capacity and faults.  This
    proves the alloc/store/release accounting stays balanced (no double
    accounting on DMA_STORE touching an existing slot)."""
    chunk_bytes = 128 * 128 * 2  # per-tile plane
    bytes_per_chunk = chunk_bytes * 4  # 4 tiles' input per group chunk
    hw = HardwareConfig()
    hw = hw.with_overrides(group_sram_bytes=4 * bytes_per_chunk)
    sim = SimConfig(fidelity="full_memory", max_cycles=50000)
    s = Simulator(hw, sim)
    wl = PowWorkload()
    r = s.run(wl.module)
    assert r.completed, r.reason

    # overshoot: 5 chunks need 5 x bytes_per_chunk but only 4 x fit
    from pipeline_validator.workload_builders import make_pow_task

    module5 = make_pow_task(num_group_chunks=5)
    hw2 = HardwareConfig()
    hw2 = hw2.with_overrides(group_sram_bytes=4 * bytes_per_chunk)
    sim2 = SimConfig(fidelity="full_memory", max_cycles=50000)
    s2 = Simulator(hw2, sim2)
    r2 = s2.run(module5)
    assert not r2.completed
    assert "faulted" in r2.reason


# ---------------------------------------------------------------------------
# Memory models
# ---------------------------------------------------------------------------


class TestMemory:
  def test_l2_capacity_ok(self):
    l2 = L2SRAM(capacity_bytes=4096)
    assert l2.capacity_ok(2048)
    slot = l2.alloc_slot("A", 2048)
    assert slot is not None
    assert l2.capacity_ok(2048)

  def test_l2_capacity_fault(self):
    l2 = L2SRAM(capacity_bytes=1024)
    l2.alloc_slot("A", 1024)
    slot = l2.alloc_slot("B", 1)
    assert slot is None
    assert l2.pmu_capacity_fault_count == 1

  def test_l2_bank_conflict_serializes(self):
    """Same-bank accesses serialize; cross-bank may parallel."""
    l2 = L2SRAM(banks=2, bank_bandwidth_gbs=12.8)
    clock_hz = 1e9
    # first access on bank 0
    lat0 = l2.access_latency(128, bank=0, cycle=0, clock_hz=clock_hz)
    # second access on same bank at cycle 0 → serialized
    lat1 = l2.access_latency(128, bank=0, cycle=0, clock_hz=clock_hz)
    assert lat1 >= lat0  # serialized behind first
    assert l2.pmu_bank_conflict_cycles > 0

  def test_l2_stable_bank_assignment(self):
    """Bank assignment is deterministic across runs (crc32, not hash())."""
    l2_a = L2SRAM(banks=16)
    l2_b = L2SRAM(banks=16)
    assert l2_a._pick_bank("slot_A") == l2_b._pick_bank("slot_A")
    assert l2_a._pick_bank("slot_B") == l2_b._pick_bank("slot_B")

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
      noc.send(2, Flit(vc=2, src=0, dst=1, bytes_total=64, tag=i), cycle=0)
    # put one flit on VC0
    noc.send(0, Flit(vc=0, src=0, dst=1, bytes_total=32, tag=99), cycle=0)
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


# ---------------------------------------------------------------------------
# Slot frame
# ---------------------------------------------------------------------------


class TestSlotFrame:
  def test_frame_bind_succeeds(self):
    from pipeline_validator.memory import SlotFrame

    f = SlotFrame(l1_bytes=1024 * 1024)
    ok, cycles = f.bind(cycle=0, bind_cycles=8)
    assert ok
    assert cycles == 8
    assert f.shadow is not None

  def test_frame_capacity_fault(self):
    from pipeline_validator.memory import Slot, SlotFrame

    f = SlotFrame(l1_bytes=512)
    f.slots[0] = Slot(0, base=0, size=512)
    ok, _ = f.bind(cycle=0)
    assert ok  # exactly fits
    f.slots[1] = Slot(1, base=0, size=1)
    ok2, _ = f.bind(cycle=0)
    assert not ok2  # overlap fault

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
      hw = HardwareConfig()
      sim = SimConfig(fidelity=fidelity, context_count=1, max_cycles=200000)
      for wl_cls in ALL_WORKLOADS:
        wl = wl_cls()
        s = Simulator(hw, sim)
        signal.alarm(60)
        r = s.run(wl.module)
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
      HardwareConfig(),
      SimConfig(fidelity="runtime", context_count=3, max_cycles=10000),
      enable_tracer=True,
    )
    result = sim.run(make_same_tile_roles_task(3))
    assert result.completed, result.reason
    assert result.tracer is not None
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    peak = max(
      e["args"]["active_context_count"] for e in events if e.get("name") == "active_context_count"
    )
    assert peak == 3

  def test_context_count_bounds(self):
    with pytest.raises(ValueError, match="context_count must be between 1 and 8"):
      SimConfig(context_count=0)
    with pytest.raises(ValueError, match="context_count must be between 1 and 8"):
      SimConfig(context_count=9)
    with pytest.raises(ValueError, match="context_count must be between 1 and 8"):
      TileUCE(0, HardwareConfig(), context_count=9)

  def test_dispatch_pinned_same_context_serializes(self):
    """Two roles pinned to the same context serialize: zero context switches."""
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", context_count=2, max_cycles=10000))
    result = sim.run(make_same_tile_roles_task(2, pins=[0, 0]))
    assert result.completed, result.reason
    assert result.pmu.events.get("uce_context_switch", 0) == 0

  def test_dispatch_pinned_context_binds_requested_index(self):
    """Pinned dispatch lands on the requested tile-local context index."""
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", context_count=2, max_cycles=10000),
      enable_tracer=True,
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
    buffer = NestAllocOp("l2_buf", 4096)
    disp0 = NestDispatchOp("ctx_wait_mfe", tasks.result, buffer.result, buffer.result,
                           "ev_a", "", "", context_id=0)
    disp1 = NestDispatchOp("ctx_wait_mfe", tasks.result, buffer.result, buffer.result,
                           "ev_b", "", "", context_id=1)
    module = ModuleOp([
      prog,
      NestContextOp(
        "same_prog_two_pins",
        [buffer, tasks, disp0, disp1,
         NestAwaitOp([disp0.grid_done, disp1.grid_done]), NestReturnOp()],
        placement=1,
      ),
    ])
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
      e["args"]["context"]: e["args"]["cycle"]
      for e in events if e.get("name") == "context_submit"
    }
    dones = {
      e["args"]["context"]: e["args"]["cycle"]
      for e in events if e.get("name") == "context_done"
    }
    slots = {
      e["args"]["context"]: e["args"]["slot"]
      for e in events if e.get("name") == "context_submit"
    }
    return submits, dones, slots

  def test_two_contexts_run_concurrently_on_two_slots(self):
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=2, max_cycles=10000),
      enable_tracer=True,
    )
    result = sim.run(make_two_context_model())
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
    result = sim.run(make_two_context_model())
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
    result = sim.run(make_two_context_model(pins=(1, 0)))
    assert result.completed, result.reason
    _, _, slots = self._submit_done_cycles(result)
    assert slots == {"ctx0": 1, "ctx1": 0}

  def test_pin_out_of_range_fails_at_load(self):
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=2, max_cycles=10000),
    )
    with pytest.raises(ValueError, match="pins device context 2 but device_context_count is 2"):
      sim.run(make_two_context_model(pins=(2, None)))

  def test_legacy_module_rejects_out_of_range_context_pin(self):
    prog = make_waiting_mfe_program()
    tasks = NestTaskRangeOp(0, 1)
    buffer = NestAllocOp("l2_buf", 4096)
    disp = NestDispatchOp("ctx_wait_mfe", tasks.result, buffer.result, buffer.result,
                         "ev_a", "", "")
    module = ModuleOp([
      prog,
      NestContextOp(
        "legacy_pinned",
        [buffer, tasks, disp, NestAwaitOp([disp.grid_done]), NestReturnOp()],
        placement=1,
        context_id=1,
      ),
    ])
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=1, max_cycles=10000),
    )
    with pytest.raises(ValueError, match="pins device context 1 but device_context_count is 1"):
      sim.run(module)

  def test_sequential_slot_reuse_gets_fresh_launch_namespace(self):
    """Submitting the same context twice on one slot must not alias
    stale completions from the first launch (launch-ID namespacing)."""
    prog = make_waiting_mfe_program("model_wait_mfe")
    buffer = NestAllocOp("l2_buf_c0", 4096)
    tasks = NestTaskRangeOp(0, 1)
    disp = NestDispatchOp("model_wait_mfe", tasks.result, buffer.result,
                          buffer.result, "ev_grid_c0", "", "")
    ctx = NestContextOp(
      "ctx0",
      [buffer, tasks, disp, NestAwaitOp([disp.grid_done]), NestReturnOp()],
      placement=1,
    )
    sub0 = NexusSubmitContextOp("ctx0", "done_c0")
    sub1 = NexusSubmitContextOp("ctx0", "done_c0_1")
    program = NexusProgramOp(
      "run_model",
      [sub0, NexusAwaitOp([sub0.result]),
       sub1, NexusAwaitOp([sub1.result]), NexusReturnOp()],
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
    submit_cycles = [e["args"]["cycle"] for e in events
                     if e.get("name") == "context_submit"]
    done_cycles = [e["args"]["cycle"] for e in events
                   if e.get("name") == "context_done"]
    assert len(submit_cycles) == 2 and len(done_cycles) == 2, (submit_cycles, done_cycles)
    # both submissions on slot 0; the second starts after the first completes
    assert sorted(e["args"]["slot"] for e in events
                  if e.get("name") == "context_submit") == [0, 0]
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
          ExecGroupAction(ExecGroupActionOp.DISPATCH_ROLE,
                          args=(0, "", ""), dst="ev_grid"),
          ExecGroupAction(ExecGroupActionOp.WAIT_EVENT, args=("ev_grid",)),
        ],
        streams=[
          ExecStreamDesc(queue_id=0, depth=1, producer_mask=1, consumer_mask=1),
        ],
        role_bindings={
          0: ExecTileRoleBinding(
            role_id=0, tile_mask=1, tile_program=prog,
            in_stream=0, out_stream=0,
          ),
        },
      )

    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=2, max_cycles=10000),
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
    assert all(t.streams == {} for t in sim.group.tiles), [
      t.streams for t in sim.group.tiles]
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
          ExecGroupAction(ExecGroupActionOp.DISPATCH_ROLE,
                          args=(0, "", ""), dst="ev_grid"),
          ExecGroupAction(ExecGroupActionOp.WAIT_EVENT, args=("ev_grid",)),
        ],
        streams=[
          ExecStreamDesc(queue_id=0, depth=1, producer_mask=1, consumer_mask=1),
        ],
        role_bindings={
          0: ExecTileRoleBinding(
            role_id=0, tile_mask=1, tile_program=prog,
            in_stream=0, out_stream=0,
          ),
        },
      )

    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=1, max_cycles=10000),
    )
    for round_idx in range(3):
      seq = sim.group.load_context_task(make_stream_task(f"ctx{round_idx}"),
                                        slot_index=0)
      for cycle in range(10000):
        if sim.group.step(cycle):
          break
      assert seq.done and not seq.faulted
      # After each drain, queues and tile bindings are fully reclaimed.
      assert sim.group.queues == {}, sim.group.queues
      assert all(t.streams == {} for t in sim.group.tiles), [
        t.streams for t in sim.group.tiles]
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
        ExecGroupAction(ExecGroupActionOp.DISPATCH_ROLE,
                        args=(0, "", ""), dst="ev_grid"),
      ],
      streams=[
        ExecStreamDesc(queue_id=0, depth=1, producer_mask=1, consumer_mask=1),
      ],
      role_bindings={
        0: ExecTileRoleBinding(
          role_id=0, tile_mask=1, tile_program=prog,
          in_stream=0, out_stream=0,
        ),
      },
    )
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=1, max_cycles=10000),
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
    for cycle in range(cycle + 1, 10000):
      if sim.group.step(cycle):
        break
    assert seq.done and not seq.faulted
    assert sim.group.queues == {}, sim.group.queues
    assert all(t.streams == {} for t in sim.group.tiles), [
      t.streams for t in sim.group.tiles]
    assert sim.group.credit_invariants_hold()
