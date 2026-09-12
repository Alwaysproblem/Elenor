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
from collections import Counter
from pathlib import Path

import pytest
from xdsl.dialects.builtin import ModuleOp

from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.dialects.elenor import (
  NestAllocOp,
  NestAwaitOp,
  NestBuffer,
  NestContextOp,
  NestDispatchOp,
  NestDMAStoreOp,
  NestGlobalMemref,
  NestGlobalView,
  NestL2View,
  NestPrefetchOp,
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
  TileBoaOp,
  TileEvuOp,
  TileGatherOp,
  TileLoadOp,
  TileProgramDefOp,
  TileProfiledAccessOp,
  TileReturnOp,
  TileSignalOp,
  TileStoreOp,
  TileSubviewOp,
)
from pipeline_validator.execution_ir import (
  ContextAdmissionStatus,
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
  PhaseSignal,
  TaskIdentity,
)
from pipeline_validator.ir_lowering import lower_model_ir, lower_workload_ir
from pipeline_validator.memory import L2SRAM, NoCRouter, PayloadTracker
from pipeline_validator.runtime import EventStatus, EventTable, FaultCode, FaultRing
from pipeline_validator.runtime.fault_ring import FaultDomain, FaultRecord
from pipeline_validator.runtime.reset_domain import ResetDomain, ResetRequest, ResetState
from pipeline_validator.simulator import SimResult, Simulator
from pipeline_validator.tile import TileUCE
from pipeline_validator.tile_group import L2AdmissionStatus, TileGroup
from pipeline_validator.workload_builders import make_pow_tile_program
from pipeline_validator.workload_ir import load_workload_ir, parse_workload_ir, print_workload_ir
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

GATHER_BINDINGS = {
  "table": GlobalBinding("table", 0x400000, 4096, "r"),
}


def assert_uce_instructions_issue_once(result: SimResult) -> list[dict]:
  assert result.tracer is not None
  events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
  issues = [event for event in events if event.get("name") == "uce_issue"]
  issue_counts = Counter(
    (event["args"]["ctx_id"], event["args"]["pc"]) for event in issues
  )
  assert issue_counts
  assert set(issue_counts.values()) == {1}
  assert any(
    event.get("name", "").startswith("WAIT_ENGINE_QUEUE:") for event in events
  )
  return issues


def make_gather_module(
  accesses: list[tuple[str, str, str | None, str | None]],
  *,
  include_evu_context: bool = False,
  l1_mshr_hint: int = 16,
) -> ModuleOp:
  program = TileProgramDefOp(
    "gather_tile",
    [],
    arg_types=[NestTask(), NestGlobalView.of([4096], "i8")],
    arg_names=["task", "table"],
  )
  _task, table = program.body.block.args
  indices = TileAllocOp([16], "i32")
  destination = TileAllocOp([len(accesses) * 64], "i8")
  profile = [
    TileProfiledAccessOp(
      request_id,
      outcome,
      64,
      line_token=line_token,
      merge_group=merge_group,
    )
    for request_id, outcome, line_token, merge_group in accesses
  ]
  gather = TileGatherOp(
    table,
    indices.result,
    destination.result,
    len(accesses) * 64,
    16384,
    65536,
    l1_mshr_hint,
    profile,
    "gather_done",
  )
  program.body.block.add_ops(
    [
      indices,
      destination,
      gather,
      TileAwaitOp([gather.result]),
      TileReturnOp(),
    ]
  )

  context = NestContextOp(
    "gather_context",
    [],
    placement=1,
    arg_types=[NestGlobalMemref.of([4096], "i8")],
    arg_names=["table"],
  )
  table_arg = context.body.block.args[0]
  table_view = NestSubviewOp(
    table_arg,
    [0],
    [4096],
    [1],
    NestGlobalView.of([4096], "i8"),
  )
  tasks = NestTaskRangeOp(0, 1)
  dispatch = NestDispatchOp(
    "gather_tile",
    tasks.result,
    [table_view.result],
    [],
    [],
    "grid_done",
    "",
    "",
    bindings=[],
    signal_policy={},
  )
  programs = [program]
  dispatches = [dispatch]
  if include_evu_context:
    evu_program = make_short_evu_program("gather_parallel_evu")
    evu_dispatch = NestDispatchOp(
      "gather_parallel_evu",
      tasks.result,
      [],
      [],
      [],
      "evu_grid_done",
      "",
      "",
      bindings=[],
      signal_policy={},
    )
    programs.append(evu_program)
    dispatches.append(evu_dispatch)
  context.body.block.add_ops(
    [
      table_view,
      tasks,
      *dispatches,
      NestAwaitOp([item.grid_done for item in dispatches]),
      NestReturnOp(),
    ]
  )
  return ModuleOp([*programs, context])


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


def make_boa_program(name: str) -> TileProgramDefOp:
  prog = TileProgramDefOp(name, [], arg_types=[NestTask()], arg_names=["task"])
  boa = TileBoaOp(
    op_name="matmul",
    m=256,
    n=256,
    k=256,
    boa_ops=33554432,
    tag="e_boa",
  )
  prog.body.block.add_ops([boa, TileAwaitOp([boa.result]), TileReturnOp()])
  return prog


def make_held_mfe_launch_module() -> ModuleOp:
  dims = [1, 64, 64]
  prog = TileProgramDefOp(
    "held_mfe_launch",
    [],
    arg_types=[NestTask(), NestBuffer.of(dims, "bf16")],
    arg_names=["task", "l2_buf"],
  )
  task_arg, l2_arg = prog.body.block.args
  loads = []
  for i in range(6):
    view = TileSubviewOp(
      l2_arg,
      None,
      None,
      [0, 0, 0],
      dims,
      [1, 1, 1],
      NestL2View.of(dims, "bf16"),
    )
    l1 = TileAllocOp(dims[1:], "bf16")
    load = TileLoadOp(view.result, l1.result, f"e_load{i}")
    prog.body.block.add_ops([view, l1, load])
    loads.append(load)
  prog.body.block.add_ops(
    [
      TileAwaitOp([load.result for load in loads]),
      TileSignalOp("input_released", task_arg),
      TileReturnOp(),
    ]
  )

  tasks = NestTaskRangeOp(0, 1)
  buffer = NestAllocOp("held_l2_buf", "in", dims, "bf16")
  dispatch = NestDispatchOp(
    "held_mfe_launch",
    tasks.result,
    [],
    [buffer.result],
    [],
    "held_mfe_done",
    "held_mfe_input_released",
    "",
    bindings=[buffer.result],
    signal_policy={"input_released": "all_tasks"},
  )
  context = NestContextOp(
    "held_mfe_context",
    [
      buffer,
      tasks,
      dispatch,
      NestReleaseOp(buffer.result, depends_on=[dispatch.input_released]),
      NestAwaitOp([dispatch.grid_done]),
      NestReturnOp(),
    ],
    placement=1,
  )
  return ModuleOp([prog, context])


def make_same_tile_roles_task(role_count: int, pins: list[int | None] | None = None) -> ModuleOp:
  """Dispatch role_count programs to one tile to exercise context switching."""
  names = ["ctx_wait_mfe"] + [f"ctx_short_evu{i}" for i in range(role_count - 1)]
  progs = [make_waiting_mfe_program(names[0])] + [make_short_evu_program(n) for n in names[1:]]
  tasks = NestTaskRangeOp(0, 1)
  buffer = NestAllocOp("l2_buf", "in", L2_WAIT_DIMS, "bf16")
  dispatches = []
  for i, name in enumerate(names):
    bindings = [buffer.result] if i == 0 else []
    ins = [buffer.result] if i == 0 else []
    dispatches.append(
      NestDispatchOp(
        name,
        tasks.result,
        [],
        ins,
        [],
        f"ev_role{i}",
        f"ev_inrel{i}" if i == 0 else "",
        "",
        bindings=bindings,
        signal_policy={"input_released": "all_tasks"} if i == 0 else {},
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
      [],
      [buffer.result],
      [],
      f"ev_grid_c{i}",
      f"ev_inrel_c{i}",
      "",
      bindings=[buffer.result],
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
# Deterministic profiled Gather runtime
# ---------------------------------------------------------------------------


class TestGatherRuntime:
  def test_all_l1_profile_completes_without_hbm_refill(self):
    module = make_gather_module(
      [
        ("r0", "L1_HIT", "line0", None),
        ("r1", "L1_HIT", "line1", None),
      ]
    )
    simulator = Simulator(
      HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10),
      SimConfig(fidelity="full_memory", max_cycles=10000),
      enable_tracer=True,
    )
    result = simulator.run(module, input_bindings=GATHER_BINDINGS)
    assert result.completed, result.reason
    assert result.pmu.events["gather_requests"] == 2
    assert result.pmu.events["gather_l1_hits"] == 2
    assert result.pmu.events["gather_hbm_misses"] == 0
    assert {
      "gather_requests",
      "gather_l1_hits",
      "gather_l2_hits",
      "gather_hbm_misses",
      "gather_mshr_merges",
      "gather_mshr_stalls",
      "gather_reorder_wait_cycles",
      "gather_bytes",
    } <= result.pmu.events.keys()
    assert result.pmu.events["gather_l2_hits"] == 0
    assert result.pmu.events["gather_mshr_merges"] == 0
    assert result.pmu.events["gather_mshr_stalls"] == 0
    assert result.pmu.events["gather_reorder_wait_cycles"] == 0
    assert simulator.group.l2_cache.snapshot()["refills"] == 0
    events = simulator.tracer._events if simulator.tracer is not None else []
    writes = [event for event in events if event["name"] == "gather_destination_write"]
    from pipeline_validator.report import build_report
    from pipeline_validator.workloads import Workload

    report = build_report(
      Workload("gather", module, expected={}, description="profiled gather"),
      result,
    )
    checks = {check["check"]: check for check in report.checks}
    assert checks["gather_request_conservation"]["pass"]
    assert checks["gather_zero_leak"]["pass"]
    assert report.gather_fidelity == "deterministic_profiled_not_address_or_value_accurate"
    done = [event for event in events if event["name"] == "gather_done"]
    assert [event["args"]["ordinal"] for event in writes] == [0, 1]
    assert len(done) == 1
    assert done[0]["ts"] >= writes[-1]["ts"]
  def test_l2_hit_uses_cache_noc_and_local_fill_without_hbm(self):
    module = make_gather_module(
      [("r0", "L2_HIT", "line0", None)]
    )
    simulator = Simulator(
      HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10),
      SimConfig(fidelity="full_memory", max_cycles=10000),
    )
    result = simulator.run(module, input_bindings=GATHER_BINDINGS)
    assert result.completed, result.reason
    issued = simulator.group.transfer_manager.snapshot()["issued_by_op"]
    assert issued["gather_l2_hit"] == 1
    assert issued.get("gather_hbm_refill", 0) == 0
    assert simulator.group.tiles[0].l1_cache.snapshot()["refills"] == 1
    assert simulator.group.l2_cache.snapshot()["hits"] == 1

  def test_two_merged_hbm_misses_issue_one_leader_refill(self):
    module = make_gather_module(
      [
        ("r0", "HBM_MISS", "line42", "miss42"),
        ("r1", "HBM_MISS", "line42", "miss42"),
      ]
    )
    simulator = Simulator(
      HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10),
      SimConfig(fidelity="full_memory", max_cycles=10000),
      enable_tracer=True,
    )
    result = simulator.run(module, input_bindings=GATHER_BINDINGS)
    assert result.completed, result.reason
    assert result.pmu.events["gather_requests"] == 2
    assert result.pmu.events["gather_hbm_misses"] == 2
    assert result.pmu.events["gather_mshr_merges"] == 1
    issued = simulator.group.transfer_manager.snapshot()["issued_by_op"]
    assert issued["gather_hbm_refill"] == 1
    assert issued["gather_l2_refill"] == 1
    assert simulator.group.l2_cache.snapshot()["resident_lines"] == 1
    assert simulator.group.tiles[0].l1_cache.snapshot()["resident_lines"] == 1
    events = simulator.tracer._events if simulator.tracer is not None else []
    writes = [event for event in events if event["name"] == "gather_destination_write"]
    assert [event["args"]["ordinal"] for event in writes] == [0, 1]

  def test_out_of_order_responses_materialize_in_profile_order(self):
    module = make_gather_module(
      [
        ("slow", "HBM_MISS", "slow_line", None),
        ("fast", "L1_HIT", "fast_line", None),
      ]
    )
    simulator = Simulator(
      HardwareConfig().with_overrides(hbm_fixed_latency_cycles=20),
      SimConfig(fidelity="full_memory", max_cycles=10000),
      enable_tracer=True,
    )
    result = simulator.run(module, input_bindings=GATHER_BINDINGS)
    assert result.completed, result.reason
    events = simulator.tracer._events if simulator.tracer is not None else []
    responses = [event for event in events if event["name"] == "gather_response"]
    writes = [event for event in events if event["name"] == "gather_destination_write"]
    done = [event for event in events if event["name"] == "gather_done"]
    assert [event["args"]["ordinal"] for event in responses] == [1, 0]
    assert [event["args"]["ordinal"] for event in writes] == [0, 1]
    assert result.pmu.events["gather_reorder_wait_cycles"] > 0
    assert len(done) == 1
    assert done[0]["ts"] >= writes[-1]["ts"]
    assert done[0]["args"] == {
      "request_id": "fast",
      "ordinal": 1,
      "outcome": "L1_HIT",
      "event_id": writes[-1]["args"]["event_id"],
    }

  def test_l1_mshr_full_stalls_one_gather_while_other_context_progresses(self):
    module = make_gather_module(
      [
        ("r0", "HBM_MISS", "line0", None),
        ("r1", "HBM_MISS", "line1", None),
      ],
      include_evu_context=True,
      l1_mshr_hint=1,
    )
    simulator = Simulator(
      HardwareConfig().with_overrides(
        hbm_fixed_latency_cycles=30,
        l1_mshr_entries=1,
      ),
      SimConfig(
        fidelity="full_memory",
        context_count=2,
        device_context_count=2,
        max_cycles=10000,
      ),
      enable_tracer=True,
    )
    result = simulator.run(module, input_bindings=GATHER_BINDINGS)
    from pipeline_validator.pmu import StallReason

    assert result.pmu.stall_cycles[StallReason.WAIT_MSHR] > 0
    assert result.completed, result.reason
    assert result.pmu.events["gather_mshr_stalls"] == 1
    assert result.pmu.events["uce_context_switch"] > 0
    assert simulator.group.tiles[0].l1_mshr.snapshot()["active"] == 0
    assert simulator.group.l2_mshr.snapshot()["active"] == 0
    events = simulator.tracer._events if simulator.tracer is not None else []
    evu = [event for event in events if event["name"] == "EVU:relu"]
    gather_done = [event for event in events if event["name"] == "gather_done"]
    assert len(evu) == 1
    assert len(gather_done) == 1
    assert evu[0]["ts"] < gather_done[0]["ts"]




  def test_fault_reset_clears_gather_transactions_mshrs_and_allocations(self):
    module = make_gather_module(
      [("r0", "HBM_MISS", "line0", None)]
    )
    config = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=1000)
    group = TileGroup(
      config,
      fidelity="full_memory",
      context_count=1,
    )
    group.load_task(
      lower_workload_ir(module),
      input_bindings=GATHER_BINDINGS,
    )
    fault_cycle = None
    for cycle in range(200):
      group.step(cycle)
      if group.tiles[0].mfe._gather_jobs:
        fault_cycle = cycle
        break
    assert fault_cycle is not None
    group.trigger_fault(
      FaultCode.ADDRESS_FAULT,
      tile_id=0,
      cycle=fault_cycle,
      desc_id="injected gather fault",
    )
    for cycle in range(fault_cycle + 1, fault_cycle + 500):
      group.step(cycle)
      if group.reset_domain.is_done:
        break
    assert group.reset_domain.is_done
    snapshot = group.snapshot()
    memory = snapshot["memory"]
    assert memory["mshr"]["l2"]["active"] == 0
    assert memory["mshr"]["l2"]["callbacks"] == 0
    assert memory["cache"]["l2"]["resident_lines"] == 0
    assert all(
      item["resident_lines"] == 0
      for item in memory["cache"]["l1"].values()
    )
    assert all(item["active"] == 0 for item in memory["mshr"]["l1"].values())
    assert all(tile["gather_active_jobs"] == 0 for tile in snapshot["tiles"])
    assert memory["transfers"]["inflight"] == 0
    assert memory["l2"]["live_allocations"] == 0
    assert all(
      item["allocator"]["live_allocations"] == 0
      for item in memory["l1"].values()
    )
    assert all(
      stage["busy_resources"] == 0 and stage["outstanding"] == 0
      for stage in memory["transfers"]["stages"].values()
    )

  def test_gather_source_binding_bounds_fail_before_runtime(self):
    module = make_gather_module(
      [("r0", "L1_HIT", "line0", None)]
    )
    simulator = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="full_memory"),
    )
    with pytest.raises(ValueError, match="smaller than required 4096 bytes"):
      simulator.run(
        module,
        input_bindings={
          "table": GlobalBinding("table", 0x400000, 4095, "r"),
        },
      )
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
    """Four DMA stores complete on non-overlapping logical output lanes.

    PR 2 replaces the round-robin channel selector with the
    TransferManager's lowest-free-channel allocation; every store must
    complete and retain its end-to-end summary timing.
    """
    hw = HardwareConfig(num_dma_channels=2)
    sim = Simulator(hw, SimConfig(fidelity="runtime", max_cycles=200_000), enable_tracer=True)
    result = sim.run(PowWorkload().module, input_bindings=POW_BINDINGS)
    assert result.completed, result.reason
    assert result.tracer is not None
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    store_events = [
      event for event in events
      if event.get("args", {}).get("summary_kind") == "group_transfer"
      and event["args"].get("op") == "global_store"
    ]
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
  def test_readwrite_actual_pins_once_per_task(self):
    """One readwrite actual has one pin per task with merged access flags."""
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
    # The same actual is read and written, but each task receives one pin.
    for task_id in range(4):
      assert set(pins[task_id]) == {slot}
      assert pins[task_id][slot].reads
      assert pins[task_id][slot].writes
    record = group.l2_sram._allocator._live[handle.allocation_id]
    assert len(record.pins) == 4
    assert record.pins == {
      pins[task_id][slot].consumer_id for task_id in range(4)
    }
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
    s = Simulator(
      hw, SimConfig(fidelity="full_memory", max_cycles=100000),
      enable_tracer=True)
    wl = PowWorkload()
    task = lower_workload_ir(wl.module)
    s._assign_program_ids(task)
    s.group.load_task(task, input_bindings=POW_BINDINGS)
    # first steps: sequencer issues the first prefetch (cycle 0), the
    # HBM leg then issues on the manager step of cycle 1
    s.group.step(0)
    s.group.step(1)
    assert s.group.transfer_manager.inflight_count > 0
    assert s.group._group_transfer_trace_slots
    assert any(s.group._group_transfer_trace_busy_slots.values())
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
    assert s.group._group_transfer_trace_slots == {}
    assert not any(s.group._group_transfer_trace_busy_slots.values())
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
      task_identity=TaskIdentity(grid=GridInstanceId("ctx", 0, 1, 0), task_id=0),
      l2_formal_handles={1: l2_handle},
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

  def test_frame_accepts_disjoint_fragmented_allocation_segments(self):
    from pipeline_validator.execution_ir import ExecL1Buffer
    from pipeline_validator.memory import (
      AdmissionFailure,
      AllocationRequest,
      BankedFreeExtentAllocator,
      SlotFrame,
      TaskBufferOwner,
    )

    allocator = BankedFreeExtentAllocator("l1", 256, 2)
    owner_a = TaskBufferOwner("ctx", 0, "ev", 0, 0, 0, "a")
    owner_b = TaskBufferOwner("ctx", 0, "ev", 0, 0, 0, "b")
    initial = allocator.plan_bundle(
      [
        AllocationRequest("l1", "a", owner_a, 32, 1),
        AllocationRequest("l1", "b", owner_b, 32, 1),
      ]
    )
    assert not isinstance(initial, AdmissionFailure)
    handle_a, handle_b = allocator.commit(initial, cycle=0)
    allocator.request_release(handle_a, owner_a, cycle=1)

    owner_fragmented = TaskBufferOwner("ctx", 0, "ev", 0, 0, 0, "fragmented")
    fragmented = allocator.plan_bundle(
      [AllocationRequest("l1", "fragmented", owner_fragmented, 96, 1)]
    )
    assert not isinstance(fragmented, AdmissionFailure)
    fragmented_handle = allocator.commit(fragmented, cycle=2)[0]
    assert len(fragmented_handle.bank_segments) == 2

    frame = SlotFrame(l1_bytes=256)
    fragmented_spec = ExecL1Buffer(
      "fragmented", (96,), "i8", 1, 1, 96
    )
    blocker_spec = ExecL1Buffer("b", (32,), "i8", 1, 1, 32)
    assert frame.prepare(
      [fragmented_handle, handle_b],
      [fragmented_spec, blocker_spec],
    )

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

  def test_held_engine_launch_issues_once_and_parks(self):
    context_count = 8
    programs = [
      make_boa_program(f"ctx_held_boa{i}") for i in range(context_count)
    ]
    tasks = NestTaskRangeOp(0, 1)
    dispatches = [
      NestDispatchOp(
        program.sym_name.data,
        tasks.result,
        [],
        [],
        [],
        f"ev_held_boa{i}",
        "",
        "",
        bindings=[],
        signal_policy={},
        context_id=i,
      )
      for i, program in enumerate(programs)
    ]
    context = NestContextOp(
      "held_boa_context",
      [
        tasks,
        *dispatches,
        NestAwaitOp([dispatch.grid_done for dispatch in dispatches]),
        NestReturnOp(),
      ],
      placement=1,
    )
    result = Simulator(
      HardwareConfig(),
      SimConfig(
        fidelity="runtime",
        context_count=context_count,
        max_cycles=200000,
      ),
      enable_tracer=True,
    ).run(ModuleOp([*programs, context]))

    assert result.completed, result.reason
    queue_stalls = result.pmu.named_cycles.get("engine_queue_full", 0)
    assert 0 < queue_stalls <= result.cycles
    issues = assert_uce_instructions_issue_once(result)
    assert sum(
      event["args"]["op"] == ExecTileOp.LAUNCH_BOA.value for event in issues
    ) == context_count

  def test_held_mfe_launch_issues_once(self):
    result = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", max_cycles=200000),
      enable_tracer=True,
    ).run(make_held_mfe_launch_module())

    assert result.completed, result.reason
    assert result.pmu.named_cycles.get("engine_queue_full", 0) > 0
    issues = assert_uce_instructions_issue_once(result)
    assert sum(
      event["args"]["op"] == ExecTileOp.LAUNCH_MFE.value for event in issues
    ) == 6

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
      [],
      [buffer.result],
      [],
      "ev_a",
      "ev_inrel_a",
      "",
      bindings=[buffer.result],
      signal_policy={"input_released": "all_tasks"},
      context_id=0,
    )
    disp1 = NestDispatchOp(
      "ctx_wait_mfe",
      tasks.result,
      [],
      [buffer.result],
      [],
      "ev_b",
      "ev_inrel_b",
      "",
      bindings=[buffer.result],
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
      [],
      [buffer.result],
      [],
      "ev_a",
      "ev_inrel_a",
      "",
      bindings=[buffer.result],
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
      [],
      [buffer.result],
      [],
      "ev_grid_c0",
      "ev_inrel_c0",
      "",
      bindings=[buffer.result],
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
      [],
      [buf.result],
      [buf.result],
      "ev_grid",
      "ev_inrel",
      "ev_outready",
      bindings=[buf.result],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=[pref.result],
    )
    store = NestDMAStoreOp(buf.result, src.result, "ev_out", depends_on=[disp.output_ready])
    release = NestReleaseOp(
      buf.result,
      depends_on=[disp.input_released, pref.result, store.result],
    )
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
  """Access-aware release gating and pin lifecycle."""

  def test_exact_capacity_blocks_until_release(self):
    """Exact-capacity L2 (one pow chunk = 131072 bytes): first batch
    admits and holds all L2; the second bundle is WAIT_CAPACITY."""
    from pipeline_validator.tile_group import L2AdmissionStatus, TileGroup

    hw = HardwareConfig().with_overrides(group_sram_bytes=4 * 128 * 128 * 2)
    group = TileGroup(hw, fidelity="full_memory")
    task1 = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    task2 = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    group.load_task(task1, input_bindings=POW_BINDINGS)
    # L2 is exactly full; a second admission must wait, not fault
    outcome = group.try_admit_l2_buffers(
      task2, context_name="ctx2",
      launch_generation=group._context_launch_generation + 1, cycle=0)
    assert outcome.status is L2AdmissionStatus.WAIT_CAPACITY
    assert group.l2_sram.snapshot()["free_bytes"] == 0
    group.reset()

  def test_exact_capacity_retries_after_3_4_barrier_and_release(self):
    """Only real, completed tile phases drive the controlled 3/4 barrier."""
    from pipeline_validator.tile_group import L2AdmissionStatus, TileGroup

    chunk = 4 * 128 * 128 * 2
    hw = HardwareConfig().with_overrides(
      hbm_fixed_latency_cycles=10,
      group_sram_bytes=chunk,
    )
    group = TileGroup(hw, fidelity="full_memory")
    task = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    group.load_task(task, input_bindings=POW_BINDINGS)
    seq = group.sequencer
    for cycle in range(5000):
      group.step(cycle)
      if group._grid_signals:
        break
    assert group._grid_signals, "dispatch did not register grid signal state"
    grid = next(iter(group._grid_signals))
    captured: list[tuple[PhaseSignal, int]] = []

    def defer_target(signal, emitted_cycle):
      if signal.task.grid == grid:
        captured.append((signal, emitted_cycle))
      else:
        group._on_phase_signal(signal, emitted_cycle)

    for tile in group.tiles:
      tile.uce._phase_signal_callback = defer_target
    for actual_cycle in range(cycle + 1, cycle + 200000):
      group.step(actual_cycle)
      if len(captured) == 8:
        break
    assert len(captured) == 8
    assert not seq.faulted
    for tile in group.tiles:
      tile.uce._phase_signal_callback = group._on_phase_signal
    input_signals = sorted(
      (signal for signal, _ in captured if signal.phase == "input_released"),
      key=lambda signal: signal.task.task_id,
    )
    output_signals = sorted(
      (signal for signal, _ in captured if signal.phase == "output_ready"),
      key=lambda signal: signal.task.task_id,
    )
    assert len(input_signals) == len(output_signals) == 4

    delivery_cycle = actual_cycle + 1
    for signal in input_signals[:3]:
      group._on_phase_signal(signal, delivery_cycle)
      delivery_cycle += 1
    task2 = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    gen2 = group._context_launch_generation + 1
    outcome = group.try_admit_l2_buffers(
      task2,
      context_name="ctx2",
      launch_generation=gen2,
      cycle=delivery_cycle,
    )
    assert outcome.status is L2AdmissionStatus.WAIT_CAPACITY

    delivery_cycle += 1
    group._on_phase_signal(input_signals[3], delivery_cycle)
    for signal in output_signals:
      delivery_cycle += 1
      group._on_phase_signal(signal, delivery_cycle)
    for release_cycle in range(delivery_cycle + 1, delivery_cycle + 20000):
      group.step(release_cycle)
      if seq.done:
        break
    assert seq.done, "sequencer did not complete after real phase delivery"
    assert group.l2_sram.snapshot()["live_allocations"] == 0
    outcome = group.try_admit_l2_buffers(
      task2,
      context_name="ctx2",
      launch_generation=gen2,
      cycle=release_cycle,
    )
    assert outcome.status is L2AdmissionStatus.ADMITTED
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
    result = sim.run(
      parse_workload_ir(open("examples/workloads/pow_dual_context.mlir").read()),
      input_bindings=MODEL_BINDINGS,
    )
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


class TestL2AccessRelease:
  """Access-based L2 pinning and release regressions on real transfers."""

  _ARENA_DIMS = [4194304]
  _BUFFER_DIMS = [4, 64, 64]
  _TASK_VIEW_DIMS = [1, 64, 64]
  _TENSOR_ELEMENTS = 131072
  _TENSOR_BYTES = 32768
  _TASK_BYTES = 8192
  _BINDINGS = {
    "arena": GlobalBinding("arena", 0x1000000, 8388608, "rw"),
  }

  @classmethod
  def _task_view(cls, buffer, task):
    return TileSubviewOp(
      buffer,
      task,
      0,
      [0, 0, 0],
      cls._TASK_VIEW_DIMS,
      [1, 1, 1],
      NestL2View.of(cls._TASK_VIEW_DIMS, "bf16"),
    )

  @classmethod
  def _global_view(cls, arena, tensor_index: int, elements: int = 16384):
    return NestSubviewOp(
      arena,
      [tensor_index * cls._TENSOR_ELEMENTS],
      [elements],
      [1],
      NestGlobalView.of([elements], "bf16"),
    )

  @classmethod
  def _make_access_programs(cls):
    program_a = TileProgramDefOp(
      "access_A",
      [],
      arg_types=[
        NestTask(),
        NestBuffer.of(cls._BUFFER_DIMS, "bf16"),
        NestBuffer.of(cls._BUFFER_DIMS, "bf16"),
      ],
      arg_names=["task", "source_X", "shared_A"],
    )
    task_a, source_a, shared_a = program_a.body.block.args
    source_view_a = cls._task_view(source_a, task_a)
    shared_view_a = cls._task_view(shared_a, task_a)
    work_a = TileAllocOp([64, 64], "bf16", alignment=256)
    load_a = TileLoadOp(source_view_a.result, work_a.result, "a_load")
    compute_a = TileEvuOp("relu", 16448, "a_evu")
    store_a = TileStoreOp(work_a.result, shared_view_a.result, "a_store")
    program_a.body.block.add_ops(
      [
        source_view_a,
        shared_view_a,
        work_a,
        load_a,
        TileAwaitOp([load_a.result]),
        TileSignalOp("input_released", task_a),
        compute_a,
        TileAwaitOp([compute_a.result]),
        store_a,
        TileAwaitOp([store_a.result]),
        TileSignalOp("output_ready", task_a),
        TileReturnOp(),
      ]
    )

    program_b = TileProgramDefOp(
      "access_B",
      [],
      arg_types=[
        NestTask(),
        NestBuffer.of(cls._BUFFER_DIMS, "bf16"),
        NestBuffer.of(cls._BUFFER_DIMS, "bf16"),
      ],
      arg_names=["task", "shared_A", "B_out"],
    )
    task_b, shared_b, output_b = program_b.body.block.args
    shared_view_b = cls._task_view(shared_b, task_b)
    output_view_b = cls._task_view(output_b, task_b)
    work_b = TileAllocOp([64, 64], "bf16", alignment=256)
    load_b = TileLoadOp(shared_view_b.result, work_b.result, "b_load")
    compute_b = TileEvuOp("relu", 16448, "b_evu")
    store_b = TileStoreOp(work_b.result, output_view_b.result, "b_store")
    program_b.body.block.add_ops(
      [
        shared_view_b,
        output_view_b,
        work_b,
        load_b,
        TileAwaitOp([load_b.result]),
        TileSignalOp("input_released", task_b),
        compute_b,
        TileAwaitOp([compute_b.result]),
        store_b,
        TileAwaitOp([store_b.result]),
        TileSignalOp("output_ready", task_b),
        TileReturnOp(),
      ]
    )

    program_d = TileProgramDefOp(
      "access_D",
      [],
      arg_types=[
        NestTask(),
        NestBuffer.of(cls._BUFFER_DIMS, "bf16"),
        NestBuffer.of(cls._BUFFER_DIMS, "bf16"),
        NestBuffer.of(cls._BUFFER_DIMS, "bf16"),
      ],
      arg_names=["task", "gate", "shared_A", "D_out"],
    )
    task_d, gate_d, shared_d, output_d = program_d.body.block.args
    gate_view_d = cls._task_view(gate_d, task_d)
    shared_view_d = cls._task_view(shared_d, task_d)
    output_view_d = cls._task_view(output_d, task_d)
    gate_lhs = TileAllocOp([64, 64], "bf16", alignment=256)
    gate_rhs = TileAllocOp([64, 64], "bf16", alignment=256)
    accumulator = TileAllocOp([64, 64], "bf16", alignment=256)
    shared_work = TileAllocOp([64, 64], "bf16", alignment=256)
    gate_load_lhs = TileLoadOp(
      gate_view_d.result,
      gate_lhs.result,
      "d_gate_load_lhs",
    )
    gate_load_rhs = TileLoadOp(
      gate_view_d.result,
      gate_rhs.result,
      "d_gate_load_rhs",
    )
    d_ops = [
      gate_view_d,
      shared_view_d,
      output_view_d,
      gate_lhs,
      gate_rhs,
      accumulator,
      shared_work,
      gate_load_lhs,
      gate_load_rhs,
      TileAwaitOp([gate_load_lhs.result, gate_load_rhs.result]),
    ]
    for index in range(100):
      boa = TileBoaOp(
        "matmul",
        64,
        64,
        64,
        524288,
        f"d_boa_{index}",
        accumulate=index > 0,
      )
      d_ops.extend([boa, TileAwaitOp([boa.result])])
    shared_load_d = TileLoadOp(
      shared_view_d.result,
      shared_work.result,
      "d_shared_load",
    )
    d_ops.extend(
      [
        shared_load_d,
        TileAwaitOp([shared_load_d.result]),
        TileSignalOp("input_released", task_d),
      ]
    )
    for index in range(100):
      evu = TileEvuOp("relu", 16448, f"d_evu_{index}")
      d_ops.extend([evu, TileAwaitOp([evu.result])])
    store_d = TileStoreOp(
      shared_work.result,
      output_view_d.result,
      "d_store",
    )
    d_ops.extend(
      [
        store_d,
        TileAwaitOp([store_d.result]),
        TileSignalOp("output_ready", task_d),
        TileReturnOp(),
      ]
    )
    program_d.body.block.add_ops(d_ops)
    return program_a, program_b, program_d

  @classmethod
  def _make_early_store_model(cls) -> ModuleOp:
    program_a, program_b, program_d = cls._make_access_programs()
    context = NestContextOp(
      "ctx_access",
      [],
      placement=15,
      arg_types=[NestGlobalMemref.of(cls._ARENA_DIMS, "bf16")],
      arg_names=["arena"],
    )
    arena = context.body.block.args[0]
    source_x = NestAllocOp(
      "source_X",
      "in",
      cls._BUFFER_DIMS,
      "bf16",
      alignment=256,
    )
    shared_a = NestAllocOp(
      "shared_A",
      "inout",
      cls._BUFFER_DIMS,
      "bf16",
      alignment=256,
    )
    gate = NestAllocOp(
      "gate",
      "in",
      cls._BUFFER_DIMS,
      "bf16",
      alignment=256,
    )
    output_b = NestAllocOp(
      "B_out",
      "out",
      cls._BUFFER_DIMS,
      "bf16",
      alignment=256,
    )
    output_d = NestAllocOp(
      "D_out",
      "out",
      cls._BUFFER_DIMS,
      "bf16",
      alignment=256,
    )
    source_view = cls._global_view(arena, 0)
    shared_view = cls._global_view(arena, 1)
    gate_view = cls._global_view(arena, 2)
    output_b_view = cls._global_view(arena, 3)
    output_d_view = cls._global_view(arena, 4)
    prefetch_source = NestPrefetchOp(
      source_view.result,
      source_x.result,
      "pref_source_X",
    )
    prefetch_gate = NestPrefetchOp(
      gate_view.result,
      gate.result,
      "pref_gate",
    )
    tasks = NestTaskRangeOp(0, 4)
    dispatch_a = NestDispatchOp(
      "access_A",
      tasks.result,
      [],
      [source_x.result],
      [shared_a.result],
      "grid_A",
      "read_A",
      "ready_A",
      bindings=[source_x.result, shared_a.result],
      signal_policy={
        "input_released": "all_tasks",
        "output_ready": "all_tasks",
      },
      depends_on=[prefetch_source.result],
      context_id=0,
    )
    dispatch_b = NestDispatchOp(
      "access_B",
      tasks.result,
      [],
      [shared_a.result],
      [output_b.result],
      "grid_B",
      "read_B",
      "ready_B",
      bindings=[shared_a.result, output_b.result],
      signal_policy={
        "input_released": "all_tasks",
        "output_ready": "all_tasks",
      },
      depends_on=[dispatch_a.output_ready],
      context_id=1,
    )
    dispatch_d = NestDispatchOp(
      "access_D",
      tasks.result,
      [],
      [gate.result, shared_a.result],
      [output_d.result],
      "grid_D",
      "read_D",
      "ready_D",
      bindings=[gate.result, shared_a.result, output_d.result],
      signal_policy={
        "input_released": "all_tasks",
        "output_ready": "all_tasks",
      },
      depends_on=[dispatch_a.output_ready, prefetch_gate.result],
      context_id=2,
    )
    store_shared = NestDMAStoreOp(
      shared_a.result,
      shared_view.result,
      "store_shared_A",
      depends_on=[dispatch_a.output_ready],
    )
    store_b = NestDMAStoreOp(
      output_b.result,
      output_b_view.result,
      "store_B",
      depends_on=[dispatch_b.output_ready],
    )
    store_d = NestDMAStoreOp(
      output_d.result,
      output_d_view.result,
      "store_D",
      depends_on=[dispatch_d.output_ready],
    )
    context.body.block.add_ops(
      [
        source_x,
        shared_a,
        gate,
        output_b,
        output_d,
        source_view,
        shared_view,
        gate_view,
        output_b_view,
        output_d_view,
        prefetch_source,
        prefetch_gate,
        tasks,
        dispatch_a,
        dispatch_b,
        dispatch_d,
        store_shared,
        NestReleaseOp(
          source_x.result,
          depends_on=[dispatch_a.input_released, prefetch_source.result],
        ),
        store_b,
        NestReleaseOp(output_b.result, depends_on=[store_b.result]),
        NestReleaseOp(
          shared_a.result,
          depends_on=[
            dispatch_b.input_released,
            dispatch_d.input_released,
            store_shared.result,
          ],
        ),
        NestReleaseOp(
          gate.result,
          depends_on=[dispatch_d.input_released, prefetch_gate.result],
        ),
        store_d,
        NestReleaseOp(output_d.result, depends_on=[store_d.result]),
        NestAwaitOp(
          [
            dispatch_a.grid_done,
            dispatch_b.grid_done,
            dispatch_d.grid_done,
            store_shared.result,
            store_b.result,
            store_d.result,
          ]
        ),
        NestReturnOp(),
      ]
    )
    root = NexusProgramOp(
      "run_access",
      [],
      arg_types=[NestGlobalMemref.of(cls._ARENA_DIMS, "bf16")],
      arg_names=["arena"],
    )
    root_arena = root.body.block.args[0]
    submit = NexusSubmitContextOp(
      "ctx_access",
      "done_access",
      actuals=[root_arena],
    )
    root.body.block.add_ops(
      [submit, NexusAwaitOp([submit.result]), NexusReturnOp()]
    )
    return ModuleOp([program_a, program_b, program_d, context, root])

  @staticmethod
  def _trace_events(tracer) -> list[dict]:
    return json.loads(tracer.to_chrome_json())["traceEvents"]

  @staticmethod
  def _cycle_of(sim: Simulator, event: dict) -> int:
    return round(event["ts"] * 1000.0 / sim.hw.cycle_ns())

  @classmethod
  def _tile_transactions(
    cls,
    events: list[dict],
    role_event_suffix: str,
    op: str,
  ) -> dict[str, dict]:
    transactions: dict[str, dict] = {}
    for event in events:
      args = event.get("args", {})
      if (
        event.get("ph") != "X"
        or args.get("op") != op
        or not str(args.get("role_event_id", "")).endswith(role_event_suffix)
        or "accepted_cycle" not in args
      ):
        continue
      transaction_id = args["transaction_id"]
      record = transactions.setdefault(
        transaction_id,
        {
          "start": args["accepted_cycle"],
          "done": args["completion_cycle"],
          "source_address": args.get("source_address"),
          "destination_address": args.get("destination_address"),
          "bytes": args["bytes"],
          "task_id": args["task_id"],
        },
      )
      record["start"] = min(record["start"], args["accepted_cycle"])
      record["done"] = max(record["done"], args["completion_cycle"])
    return transactions

  @staticmethod
  def _pin_fingerprint(group: TileGroup, handle) -> set[tuple]:
    return {
      (
        grid,
        task_id,
        slot,
        pin.consumer_id,
        pin.reads,
        pin.writes,
      )
      for grid, task_pins in group._grid_l2_pins.items()
      for task_id, slot_pins in task_pins.items()
      for slot, pin in slot_pins.items()
      if pin.handle == handle
    }

  @staticmethod
  def _assert_runtime_zero_leak(group: TileGroup) -> None:
    memory = group.snapshot()["memory"]
    assert memory["l2"]["live_allocations"] == 0
    assert memory["l2"]["pending_release"] == 0
    assert memory["transfers"]["inflight"] == 0
    assert not group._grid_l2_pins
    for tile_id, l1 in memory["l1"].items():
      assert l1["allocator"]["live_allocations"] == 0, tile_id

  @pytest.mark.parametrize("fidelity", ["runtime", "full_memory"])
  def test_early_store_preserves_delayed_reader(self, fidelity):
    hw = HardwareConfig().with_overrides(
      num_dma_channels=2,
      hbm_fixed_latency_cycles=10,
    )
    sim = Simulator(
      hw,
      SimConfig(
        fidelity=fidelity,
        context_count=4,
        device_context_count=1,
        memory_trace=True,
        max_cycles=2000000,
      ),
      enable_tracer=True,
    )
    result = sim.run(
      self._make_early_store_model(),
      input_bindings=self._BINDINGS,
    )
    assert result.completed, result.reason
    assert result.tracer is not None
    events = self._trace_events(result.tracer)
    shared_alloc = next(
      event
      for event in events
      if event.get("name") == "l2_alloc"
      and event["args"].get("buffer_id") == "shared_A"
    )
    shared_id = shared_alloc["args"]["allocation_id"]
    shared_base = shared_alloc["args"]["base_address"]
    shared_release = next(
      event
      for event in events
      if event.get("name") == "l2_release"
      and event["args"].get("allocation_id") == shared_id
    )
    release_cycle = self._cycle_of(sim, shared_release)
    shared_store = next(
      event["args"]
      for event in events
      if event.get("args", {}).get("summary_kind") == "group_transfer"
      and event["args"].get("op") == "global_store"
      and event["args"].get("buffer_id") == "shared_A"
    )
    reads_b = self._tile_transactions(events, "grid_B", "tile_load")
    reads_d = {
      transaction_id: transaction
      for transaction_id, transaction in self._tile_transactions(
        events,
        "grid_D",
        "tile_load",
      ).items()
      if shared_base
      <= transaction["source_address"]
      < shared_base + self._TENSOR_BYTES
    }
    assert len(reads_b) == len(reads_d) == 4
    for transaction in reads_d.values():
      assert transaction["source_address"] == (
        shared_base + transaction["task_id"] * self._TASK_BYTES
      )
      assert transaction["bytes"] == self._TASK_BYTES
    assert shared_store["source_address"] == shared_base
    assert shared_store["bytes"] == self._TENSOR_BYTES
    first_d_read = min(item["start"] for item in reads_d.values())
    last_b_read = max(item["done"] for item in reads_b.values())
    last_d_read = max(item["done"] for item in reads_d.values())
    assert shared_store["completion_cycle"] < first_d_read
    d_evu_events = [
      event
      for event in events
      if event.get("name") == "EVU:relu"
      and event.get("args", {}).get("program") == "access_D"
      and event["args"].get("local_event_id") == "d_evu_99"
    ]
    assert len(d_evu_events) == 4
    d_compute_end = max(
      self._cycle_of(
        sim,
        {
          "ts": event["ts"] + event["dur"],
        },
      )
      for event in d_evu_events
    )
    assert max(
      last_b_read,
      last_d_read,
      shared_store["completion_cycle"],
    ) <= release_cycle < d_compute_end
    assert shared_store["transaction_id"]
    assert set(reads_b).isdisjoint(reads_d)
    context_done = next(
      event["args"]
      for event in events
      if event.get("name") == "context_done"
      and event["args"].get("context") == "ctx_access"
    )
    assert context_done["cycle"] >= release_cycle
    self._assert_runtime_zero_leak(sim.group)
    assert result.credit_invariant_ok
    if fidelity == "full_memory":
      for vc in result.group_snapshot["memory"]["noc"].values():
        assert vc["credit"] == hw.noc_vc_depth
    result.tracer.assert_well_formed()

  def test_release_preflight_rejects_late_reader(self):
    from dataclasses import replace

    from pipeline_validator.memory.allocator import MemoryInvariantError

    sim = Simulator(
      HardwareConfig().with_overrides(
        num_dma_channels=2,
        hbm_fixed_latency_cycles=10,
      ),
      SimConfig(
        fidelity="full_memory",
        context_count=4,
        device_context_count=1,
        memory_trace=True,
        max_cycles=2000000,
      ),
      enable_tracer=True,
    )
    model = lower_model_ir(self._make_early_store_model())
    for task in model.tasks.values():
      sim._assign_program_ids(task)
    group = sim.group
    seq = group.load_context_task(
      model.tasks["ctx_access"],
      slot_index=0,
      context_name="ctx_access",
      input_bindings=self._BINDINGS,
      formal_bindings={"arena": "arena"},
      cycle=0,
    )
    release_request = next(
      action.args[0]
      for action in seq.task.actions
      if action.op == ExecGroupActionOp.RELEASE_L2
      and action.args[0].buffer_slot == "shared_A"
    )
    late_reader = max(release_request.reader_dispatch_ordinals)
    late_event = next(
      event
      for event in release_request.dependency_events
      if event.endswith("read_D")
    )
    bad_request = replace(
      release_request,
      reader_dispatch_ordinals=tuple(
        ordinal
        for ordinal in release_request.reader_dispatch_ordinals
        if ordinal != late_reader
      ),
      dependency_events=tuple(
        event
        for event in release_request.dependency_events
        if event != late_event
      ),
    )
    for cycle in range(1000000):
      group.step(cycle)
      if (
        set(bad_request.dependency_events) <= seq._events_done
        and late_event not in seq._events_done
      ):
        break
    assert late_event not in seq._events_done
    handle = group._l2_handles[
      (seq.context_launch_generation, "shared_A")
    ]
    pins_before = self._pin_fingerprint(group, handle)
    assert pins_before
    assert any(item[-2] and not item[-1] for item in pins_before)
    assert any(item[-1] for item in pins_before)
    allocator_pins_before = set(
      group.l2_sram._allocator._live[handle.allocation_id].pins
    )
    snapshot_before = group.l2_sram.snapshot()
    with pytest.raises(MemoryInvariantError):
      group.release_l2(bad_request, sequencer=seq, cycle=cycle + 1)
    assert not group.l2_sram.is_released(handle)
    assert self._pin_fingerprint(group, handle) == pins_before
    assert (
      group.l2_sram._allocator._live[handle.allocation_id].pins
      == allocator_pins_before
    )
    assert (
      group.l2_sram.snapshot()["pending_release"]
      == snapshot_before["pending_release"]
      == 0
    )
    group.reset()
    self._assert_runtime_zero_leak(group)
    assert group.credit_invariants_hold()

  @classmethod
  def _make_alias_module(cls, reverse_phases: bool) -> ModuleOp:
    dims = [1, 64, 64]
    task_view_dims = [1, 64, 64]
    program = TileProgramDefOp(
      "alias_reverse" if reverse_phases else "alias_forward",
      [],
      arg_types=[
        NestTask(),
        NestBuffer.of(dims, "bf16"),
        NestBuffer.of(dims, "bf16"),
      ],
      arg_names=["task", "read_formal", "write_formal"],
    )
    task, read_formal, write_formal = program.body.block.args
    read_view = TileSubviewOp(
      read_formal,
      task,
      0,
      [0, 0, 0],
      task_view_dims,
      [1, 1, 1],
      NestL2View.of(task_view_dims, "bf16"),
    )
    write_view = TileSubviewOp(
      write_formal,
      task,
      0,
      [0, 0, 0],
      task_view_dims,
      [1, 1, 1],
      NestL2View.of(task_view_dims, "bf16"),
    )
    work = TileAllocOp([64, 64], "bf16", alignment=256)
    scratch = TileAllocOp([64, 64], "bf16", alignment=256)
    initial_load = TileLoadOp(read_view.result, work.result, "alias_load_0")
    ops = [
      read_view,
      write_view,
      work,
      scratch,
      initial_load,
      TileAwaitOp([initial_load.result]),
    ]
    if not reverse_phases:
      ops.append(TileSignalOp("input_released", task))
    if reverse_phases:
      tile_store = TileStoreOp(
        work.result,
        write_view.result,
        "alias_tile_store",
      )
      ops.extend(
        [
          tile_store,
          TileAwaitOp([tile_store.result]),
          TileSignalOp("output_ready", task),
        ]
      )
    for index in range(100):
      evu = TileEvuOp("relu", 16448, f"alias_evu_{index}")
      ops.extend([evu, TileAwaitOp([evu.result])])
    if reverse_phases:
      late_load = TileLoadOp(
        read_view.result,
        scratch.result,
        "alias_load_1",
      )
      ops.extend(
        [
          late_load,
          TileAwaitOp([late_load.result]),
          TileSignalOp("input_released", task),
        ]
      )
    else:
      tile_store = TileStoreOp(
        work.result,
        write_view.result,
        "alias_tile_store",
      )
      ops.extend(
        [
          tile_store,
          TileAwaitOp([tile_store.result]),
          TileSignalOp("output_ready", task),
        ]
      )
    ops.append(TileReturnOp())
    program.body.block.add_ops(ops)

    context = NestContextOp(
      "ctx_alias",
      [],
      placement=1,
      arg_types=[NestGlobalMemref.of(cls._ARENA_DIMS, "bf16")],
      arg_names=["arena"],
    )
    arena = context.body.block.args[0]
    buffer = NestAllocOp(
      "alias",
      "inout",
      dims,
      "bf16",
      alignment=256,
    )
    global_view = cls._global_view(arena, 0, elements=4096)
    prefetch = NestPrefetchOp(
      global_view.result,
      buffer.result,
      "alias_prefetch",
    )
    tasks = NestTaskRangeOp(0, 1)
    dispatch = NestDispatchOp(
      program.sym_name.data,
      tasks.result,
      [],
      [buffer.result],
      [buffer.result],
      "alias_grid",
      "alias_read",
      "alias_ready",
      bindings=[buffer.result, buffer.result],
      signal_policy={
        "input_released": "all_tasks",
        "output_ready": "all_tasks",
      },
      depends_on=[prefetch.result],
      context_id=0,
    )
    store = NestDMAStoreOp(
      buffer.result,
      global_view.result,
      "alias_global_store",
      depends_on=[dispatch.output_ready],
    )
    context.body.block.add_ops(
      [
        buffer,
        global_view,
        prefetch,
        tasks,
        dispatch,
        store,
        NestReleaseOp(
          buffer.result,
          depends_on=[
            dispatch.input_released,
            prefetch.result,
            store.result,
          ],
        ),
        NestAwaitOp([dispatch.grid_done, store.result]),
        NestReturnOp(),
      ]
    )
    return ModuleOp([program, context])

  @pytest.mark.parametrize("fidelity", ["runtime", "full_memory"])
  @pytest.mark.parametrize("reverse_phases", [False, True])
  def test_readwrite_alias_lifetime(self, fidelity, reverse_phases):
    hw = HardwareConfig().with_overrides(
      num_dma_channels=2,
      hbm_fixed_latency_cycles=10,
    )
    sim = Simulator(
      hw,
      SimConfig(
        fidelity=fidelity,
        context_count=4,
        device_context_count=1,
        memory_trace=True,
        max_cycles=2000000,
      ),
      enable_tracer=True,
    )
    task = lower_workload_ir(self._make_alias_module(reverse_phases))
    sim._assign_program_ids(task)
    group = sim.group
    group.load_task(task, input_bindings=self._BINDINGS)
    seq = group.sequencer
    for dispatch_cycle in range(10000):
      group.step(dispatch_cycle)
      if group._grid_l2_pins:
        break
    assert group._grid_l2_pins
    grid = next(iter(group._grid_l2_pins))
    pin = group._grid_l2_pins[grid][0]["alias"]
    assert pin.reads and pin.writes
    handle = group._l2_handles[
      (seq.context_launch_generation, "alias")
    ]
    record = group.l2_sram._allocator._live[handle.allocation_id]
    assert record.pins == {pin.consumer_id}
    for cycle in range(dispatch_cycle + 1, 2000000):
      group.step(cycle)
      if seq.done:
        break
    assert seq.done and not seq.faulted, seq.fault_reason
    assert sim.tracer is not None
    events = self._trace_events(sim.tracer)
    loads = self._tile_transactions(events, "alias_grid", "tile_load")
    assert len(loads) == (2 if reverse_phases else 1)
    release = next(
      event
      for event in events
      if event.get("name") == "l2_release"
      and event["args"].get("allocation_id") == handle.allocation_id
    )
    release_cycle = self._cycle_of(sim, release)
    store = next(
      event["args"]
      for event in events
      if event.get("args", {}).get("summary_kind") == "group_transfer"
      and event["args"].get("buffer_id") == "alias"
      and event["args"].get("op") == "global_store"
    )
    assert release_cycle >= max(
      store["completion_cycle"],
      max(item["done"] for item in loads.values()),
    )
    if reverse_phases:
      ordered_loads = sorted(loads.values(), key=lambda item: item["start"])
      assert store["completion_cycle"] < ordered_loads[1]["start"]
    self._assert_runtime_zero_leak(group)
    assert group.credit_invariants_hold()
    sim.tracer.assert_well_formed()

  @classmethod
  def _make_inflight_store_module(cls) -> ModuleOp:
    dims = [1, 64, 64]
    program = TileProgramDefOp(
      "inflight_copy",
      [],
      arg_types=[
        NestTask(),
        NestBuffer.of(dims, "bf16"),
        NestBuffer.of(dims, "bf16"),
      ],
      arg_names=["task", "source", "output"],
    )
    task, source, output = program.body.block.args
    source_view = TileSubviewOp(
      source,
      task,
      0,
      [0, 0, 0],
      dims,
      [1, 1, 1],
      NestL2View.of(dims, "bf16"),
    )
    output_view = TileSubviewOp(
      output,
      task,
      0,
      [0, 0, 0],
      dims,
      [1, 1, 1],
      NestL2View.of(dims, "bf16"),
    )
    work = TileAllocOp([64, 64], "bf16", alignment=256)
    load = TileLoadOp(source_view.result, work.result, "copy_load")
    tile_store = TileStoreOp(work.result, output_view.result, "copy_store")
    program.body.block.add_ops(
      [
        source_view,
        output_view,
        work,
        load,
        TileAwaitOp([load.result]),
        TileSignalOp("input_released", task),
        tile_store,
        TileAwaitOp([tile_store.result]),
        TileSignalOp("output_ready", task),
        TileReturnOp(),
      ]
    )
    context = NestContextOp(
      "ctx_inflight_store",
      [],
      placement=1,
      arg_types=[NestGlobalMemref.of(cls._ARENA_DIMS, "bf16")],
      arg_names=["arena"],
    )
    arena = context.body.block.args[0]
    source_buffer = NestAllocOp(
      "store_source",
      "in",
      dims,
      "bf16",
      alignment=256,
    )
    output_buffer = NestAllocOp(
      "store_output",
      "out",
      dims,
      "bf16",
      alignment=256,
    )
    source_global = cls._global_view(arena, 0, elements=4096)
    output_global_1 = cls._global_view(arena, 1, elements=4096)
    output_global_2 = cls._global_view(arena, 2, elements=4096)
    prefetch = NestPrefetchOp(
      source_global.result,
      source_buffer.result,
      "store_prefetch",
    )
    tasks = NestTaskRangeOp(0, 1)
    dispatch = NestDispatchOp(
      "inflight_copy",
      tasks.result,
      [],
      [source_buffer.result],
      [output_buffer.result],
      "copy_grid",
      "copy_read",
      "copy_ready",
      bindings=[source_buffer.result, output_buffer.result],
      signal_policy={
        "input_released": "all_tasks",
        "output_ready": "all_tasks",
      },
      depends_on=[prefetch.result],
      context_id=0,
    )
    store_1 = NestDMAStoreOp(
      output_buffer.result,
      output_global_1.result,
      "output_store_1",
      depends_on=[dispatch.output_ready],
    )
    store_2 = NestDMAStoreOp(
      output_buffer.result,
      output_global_2.result,
      "output_store_2",
      depends_on=[dispatch.output_ready],
    )
    context.body.block.add_ops(
      [
        source_buffer,
        output_buffer,
        source_global,
        output_global_1,
        output_global_2,
        prefetch,
        tasks,
        dispatch,
        store_1,
        store_2,
        NestReleaseOp(
          source_buffer.result,
          depends_on=[dispatch.input_released, prefetch.result],
        ),
        NestReleaseOp(
          output_buffer.result,
          depends_on=[store_1.result, store_2.result],
        ),
        NestAwaitOp(
          [dispatch.grid_done, store_1.result, store_2.result]
        ),
        NestReturnOp(),
      ]
    )
    return ModuleOp([program, context])

  @classmethod
  def _make_inflight_prefetch_module(cls) -> ModuleOp:
    dims = [1, 64, 64]
    program = TileProgramDefOp(
      "inflight_reader",
      [],
      arg_types=[NestTask(), NestBuffer.of(dims, "bf16")],
      arg_names=["task", "input"],
    )
    task, input_buffer = program.body.block.args
    input_view = TileSubviewOp(
      input_buffer,
      task,
      0,
      [0, 0, 0],
      dims,
      [1, 1, 1],
      NestL2View.of(dims, "bf16"),
    )
    work = TileAllocOp([64, 64], "bf16", alignment=256)
    load = TileLoadOp(input_view.result, work.result, "reader_load")
    program.body.block.add_ops(
      [
        input_view,
        work,
        load,
        TileAwaitOp([load.result]),
        TileSignalOp("input_released", task),
        TileReturnOp(),
      ]
    )
    context = NestContextOp(
      "ctx_inflight_prefetch",
      [],
      placement=1,
      arg_types=[NestGlobalMemref.of(cls._ARENA_DIMS, "bf16")],
      arg_names=["arena"],
    )
    arena = context.body.block.args[0]
    buffer = NestAllocOp(
      "prefetch_input",
      "in",
      dims,
      "bf16",
      alignment=256,
    )
    global_1 = cls._global_view(arena, 0, elements=4096)
    global_2 = cls._global_view(arena, 1, elements=4096)
    prefetch_1 = NestPrefetchOp(
      global_1.result,
      buffer.result,
      "input_prefetch_1",
    )
    tasks = NestTaskRangeOp(0, 1)
    dispatch = NestDispatchOp(
      "inflight_reader",
      tasks.result,
      [],
      [buffer.result],
      [],
      "reader_grid",
      "reader_done",
      "",
      bindings=[buffer.result],
      signal_policy={"input_released": "all_tasks"},
      depends_on=[prefetch_1.result],
      context_id=0,
    )
    prefetch_2 = NestPrefetchOp(
      global_2.result,
      buffer.result,
      "input_prefetch_2",
    )
    context.body.block.add_ops(
      [
        buffer,
        global_1,
        global_2,
        prefetch_1,
        tasks,
        dispatch,
        NestAwaitOp([dispatch.input_released]),
        prefetch_2,
        NestReleaseOp(
          buffer.result,
          depends_on=[
            dispatch.input_released,
            prefetch_1.result,
            prefetch_2.result,
          ],
        ),
        NestAwaitOp([dispatch.grid_done, prefetch_2.result]),
        NestReturnOp(),
      ]
    )
    return ModuleOp([program, context])

  def test_release_rejects_inflight_store(self):
    from dataclasses import replace

    from pipeline_validator.memory.allocator import MemoryInvariantError
    from pipeline_validator.memory.transfer import TransferStatus

    hw = HardwareConfig().with_overrides(
      num_dma_channels=2,
      hbm_fixed_latency_cycles=10,
    )
    sim = Simulator(
      hw,
      SimConfig(
        fidelity="full_memory",
        context_count=4,
        device_context_count=1,
        memory_trace=True,
        max_cycles=200000,
      ),
      enable_tracer=True,
    )
    task = lower_workload_ir(self._make_inflight_store_module())
    sim._assign_program_ids(task)
    group = sim.group
    group.load_task(task, input_bindings=self._BINDINGS)
    seq = group.sequencer
    request = next(
      action.args[0]
      for action in seq.task.actions
      if action.op == ExecGroupActionOp.RELEASE_L2
      and action.args[0].buffer_slot == "store_output"
    )
    assert len(request.dependency_events) == 2
    first_store, second_store = request.dependency_events
    bad_request = replace(
      request,
      dependency_events=(first_store,),
    )
    handle = group._l2_handles[
      (seq.context_launch_generation, "store_output")
    ]
    second_transaction = None
    for cycle in range(200000):
      group.step(cycle)
      second_transaction = next(
        (
          transaction
          for transaction in group.transfer_manager._transactions.values()
          if transaction.completion_event == second_store
        ),
        None,
      )
      if (
        first_store in seq._events_done
        and second_store not in seq._events_done
        and second_transaction is not None
        and group.transfer_manager.has_inflight_access(handle)
      ):
        break
    assert first_store in seq._events_done
    assert second_store not in seq._events_done
    assert second_transaction is not None
    assert second_transaction.status is TransferStatus.RUNNING
    pins_before = self._pin_fingerprint(group, handle)
    allocator_pins_before = set(
      group.l2_sram._allocator._live[handle.allocation_id].pins
    )
    pending_before = group.l2_sram.snapshot()["pending_release"]
    with pytest.raises(MemoryInvariantError):
      group.release_l2(bad_request, sequencer=seq, cycle=cycle + 1)
    assert not group.l2_sram.is_released(handle)
    assert self._pin_fingerprint(group, handle) == pins_before
    assert (
      group.l2_sram._allocator._live[handle.allocation_id].pins
      == allocator_pins_before
    )
    assert group.l2_sram.snapshot()["pending_release"] == pending_before
    for finish_cycle in range(cycle + 2, cycle + 200000):
      group.step(finish_cycle)
      if seq.done:
        break
    assert seq.done and not seq.faulted, seq.fault_reason
    self._assert_runtime_zero_leak(group)
    assert group.credit_invariants_hold()
    assert sim.tracer is not None
    sim.tracer.assert_well_formed()
    group.reset()
    self._assert_runtime_zero_leak(group)

  def test_release_rejects_inflight_prefetch(self):
    from dataclasses import replace

    from pipeline_validator.memory.allocator import MemoryInvariantError
    from pipeline_validator.memory.transfer import TransferStatus

    hw = HardwareConfig().with_overrides(
      num_dma_channels=2,
      hbm_fixed_latency_cycles=10,
    )
    sim = Simulator(
      hw,
      SimConfig(
        fidelity="full_memory",
        context_count=4,
        device_context_count=1,
        memory_trace=True,
        max_cycles=200000,
      ),
      enable_tracer=True,
    )
    task = lower_workload_ir(self._make_inflight_prefetch_module())
    sim._assign_program_ids(task)
    group = sim.group
    group.load_task(task, input_bindings=self._BINDINGS)
    seq = group.sequencer
    request = next(
      action.args[0]
      for action in seq.task.actions
      if action.op == ExecGroupActionOp.RELEASE_L2
      and action.args[0].buffer_slot == "prefetch_input"
    )
    second_prefetch = next(
      event
      for event in request.dependency_events
      if event.endswith("input_prefetch_2")
    )
    bad_request = replace(
      request,
      dependency_events=tuple(
        event
        for event in request.dependency_events
        if event != second_prefetch
      ),
    )
    handle = group._l2_handles[
      (seq.context_launch_generation, "prefetch_input")
    ]
    transaction = None
    for cycle in range(200000):
      group.step(cycle)
      transaction = next(
        (
          candidate
          for candidate in group.transfer_manager._transactions.values()
          if candidate.completion_event == second_prefetch
        ),
        None,
      )
      if (
        set(bad_request.dependency_events) <= seq._events_done
        and second_prefetch not in seq._events_done
        and transaction is not None
        and group.transfer_manager.has_inflight_access(handle)
      ):
        break
    assert transaction is not None
    assert transaction.status is TransferStatus.RUNNING
    pins_before = self._pin_fingerprint(group, handle)
    allocator_pins_before = set(
      group.l2_sram._allocator._live[handle.allocation_id].pins
    )
    pending_before = group.l2_sram.snapshot()["pending_release"]
    with pytest.raises(MemoryInvariantError):
      group.release_l2(bad_request, sequencer=seq, cycle=cycle + 1)
    assert not group.l2_sram.is_released(handle)
    assert self._pin_fingerprint(group, handle) == pins_before
    assert (
      group.l2_sram._allocator._live[handle.allocation_id].pins
      == allocator_pins_before
    )
    assert group.l2_sram.snapshot()["pending_release"] == pending_before
    group.release_context_memory(cycle + 1)
    assert transaction.status is TransferStatus.CANCELLED
    assert not group.transfer_manager.has_inflight_access(handle)
    group.reset()
    self._assert_runtime_zero_leak(group)
    assert group.credit_invariants_hold()
    assert sim.tracer is not None
    sim.tracer.assert_well_formed()

  def test_inflight_query_uses_status_and_handle_generation(self):
    from dataclasses import replace

    from pipeline_validator.memory import (
      AdmissionFailure,
      AllocationRequest,
      ContextBufferOwner,
      TaskBufferOwner,
    )
    from pipeline_validator.memory.transfer import (
      MemoryTransaction,
      ResolvedMemoryView,
      TransferManager,
      TransferOp,
      TransferStatus,
    )

    owner = ContextBufferOwner("status_ctx", 0, "status_buffer")
    l2 = L2SRAM(capacity_bytes=8192, banks=1)
    plan = l2.plan_bundle(
      [AllocationRequest("l2", "status_buffer", owner, 4096, 256)]
    )
    assert not isinstance(plan, AdmissionFailure)
    original = l2.commit(plan, cycle=0)[0]
    source = ResolvedMemoryView(
      handle=original,
      offset_bytes=0,
      size_bytes=4096,
      address=original.base_address,
      segments=original.bank_segments,
    )
    destination_handle = replace(
      original,
      allocation_id="l1:status-access",
      memory_space="l1",
    )
    destination = ResolvedMemoryView(
      handle=destination_handle,
      offset_bytes=0,
      size_bytes=4096,
      address=destination_handle.base_address,
      segments=destination_handle.bank_segments,
    )
    transaction = MemoryTransaction(
      transaction_id="status-access",
      op=TransferOp.TILE_LOAD,
      issuer=TaskBufferOwner(
        "status_ctx",
        0,
        "status-grid",
        0,
        0,
        0,
        "status-l1",
      ),
      src=source,
      dst=destination,
      bytes_total=4096,
      completion_event="status-done",
      tile_id=0,
    )
    manager = TransferManager(HardwareConfig(), full_memory=False)
    manager._transactions[transaction.transaction_id] = transaction
    for status in (
      TransferStatus.PENDING,
      TransferStatus.RUNNING,
      TransferStatus.FAULTED,
    ):
      transaction.status = status
      assert manager.has_inflight_access(original)
    for status in (TransferStatus.DONE, TransferStatus.CANCELLED):
      transaction.status = status
      assert not manager.has_inflight_access(original)
    same_id_new_generation = replace(
      original,
      generation=original.generation + 1,
    )
    transaction.src = destination
    transaction.dst = source
    transaction.status = TransferStatus.RUNNING
    assert manager.has_inflight_access(original)
    assert not manager.has_inflight_access(same_id_new_generation)

    assert l2.request_release(original, owner, cycle=1)
    l2.reset()
    next_owner = ContextBufferOwner("status_ctx", 1, "status_buffer")
    next_plan = l2.plan_bundle(
      [AllocationRequest("l2", "status_buffer", next_owner, 4096, 256)]
    )
    assert not isinstance(next_plan, AdmissionFailure)
    replacement = l2.commit(next_plan, cycle=2)[0]
    assert replacement.base_address == original.base_address
    assert replacement.generation != original.generation
    transaction.status = TransferStatus.RUNNING
    assert manager.has_inflight_access(original)
    assert not manager.has_inflight_access(replacement)
    assert l2.request_release(replacement, next_owner, cycle=3)
    manager.cancel_all(cycle=3)
    assert l2.snapshot()["live_allocations"] == 0


class TestReleaseFaultPath:
  """PR 3 §5: wrong-owner / double RELEASE_L2 through the sequencer
  produces ADDRESS_FAULT + fault ring + ResetDomain zero-leak."""


  @staticmethod
  def _assert_zero_leak(group):
    assert not group._grid_l2_pins
    assert not group._grid_signals
    assert group.l2_sram.snapshot()["live_allocations"] == 0
    assert group.l2_sram.snapshot()["pending_release"] == 0
    for tile in group.tiles:
      assert tile.l1_allocator.snapshot()["live_allocations"] == 0

  def test_unknown_buffer_release_faults_and_resets(self):
    """An unknown RELEASE_L2 slot faults and reset restores zero leaks."""
    from pipeline_validator.execution_ir import ExecGroupActionOp, ExecReleaseRequest

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    group = TileGroup(hw, fidelity="runtime")
    task = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    group.load_task(task, input_bindings=POW_BINDINGS)
    seq = group.sequencer
    rel_idx = next(
      i for i, action in enumerate(seq.task.actions)
      if action.op == ExecGroupActionOp.RELEASE_L2
    )
    for cycle in range(50000):
      group.step(cycle)
      if (
        not seq.faulted
        and not seq.done
        and seq.action_index == rel_idx
        and seq._pending is None
        and all(
          event in seq._events_done
          for event in seq.task.actions[rel_idx].args[0].dependency_events
        )
      ):
        break
    assert not seq.faulted, f"premature fault: {seq.fault_reason}"
    original_req = seq.task.actions[rel_idx].args[0]
    bad_req = ExecReleaseRequest(
      buffer_slot="nonexistent",
      buffer_role=original_req.buffer_role,
      reader_dispatch_ordinals=original_req.reader_dispatch_ordinals,
      writer_dispatch_ordinals=original_req.writer_dispatch_ordinals,
      dependency_events=original_req.dependency_events,
    )
    seq.task.actions[rel_idx] = type(seq.task.actions[rel_idx])(
      ExecGroupActionOp.RELEASE_L2,
      args=(bad_req,),
    )
    group.step(cycle + 1)
    assert seq.faulted
    assert "release invariant fault" in seq.fault_reason
    assert group.pmu.events.get("release_invariant_fault", 0) >= 1
    if group.runtime_enabled:
      assert group.fault_ring.snapshot()["count"] > 0
      assert group.fault_ring.snapshot()["latest_code"] == FaultCode.ADDRESS_FAULT.name
    for reset_cycle in range(cycle + 2, cycle + 5000):
      group.step(reset_cycle)
      if group.reset_domain.is_done:
        break
    assert group.reset_domain.is_done
    self._assert_zero_leak(group)
    group.reset()

  def test_double_release_faults_and_resets(self):
    """A second RELEASE_L2 for an already-released buffer hits the
    allocator's double-release check; sequencer catches, faults, and
    reset restores zero-leak."""
    from pipeline_validator.execution_ir import ExecGroupAction, ExecGroupActionOp

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
    for cycle in range(50000):
      group.step(cycle)
      if seq.faulted:
        break
    assert seq.faulted
    assert "release invariant fault" in seq.fault_reason
    assert group.pmu.events.get("release_invariant_fault", 0) >= 1
    # Step until reset cleanup completes.
    for reset_cycle in range(cycle + 1, cycle + 5000):
      group.step(reset_cycle)
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
    rel_idx = next(
      i for i, action in enumerate(seq.task.actions)
      if action.op == ExecGroupActionOp.RELEASE_L2
    )
    for cycle in range(50000):
      group.step(cycle)
      if (
        not seq.faulted
        and not seq.done
        and seq.action_index == rel_idx
        and seq._pending is None
        and all(
          event in seq._events_done
          for event in seq.task.actions[rel_idx].args[0].dependency_events
        )
      ):
        break
    assert not seq.faulted, f"premature fault: {seq.fault_reason}"
    # Corrupt the sequencer's context_name so the owner check fails.
    seq.context_name = "wrong_owner_ctx"
    group.step(cycle + 1)
    assert seq.faulted
    assert "release invariant fault" in seq.fault_reason
    assert group.pmu.events.get("release_invariant_fault", 0) >= 1
    if group.runtime_enabled:
      assert group.fault_ring.snapshot()["count"] > 0
      assert group.fault_ring.snapshot()["latest_code"] == FaultCode.ADDRESS_FAULT.name
    for reset_cycle in range(cycle + 2, cycle + 5000):
      group.step(reset_cycle)
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
    rel_idx = next(
      i for i, action in enumerate(seq.task.actions)
      if action.op == ExecGroupActionOp.RELEASE_L2
    )
    for cycle in range(50000):
      group.step(cycle)
      if (
        not seq.faulted
        and not seq.done
        and seq.action_index == rel_idx
        and seq._pending is None
        and all(
          event in seq._events_done
          for event in seq.task.actions[rel_idx].args[0].dependency_events
        )
      ):
        break
    assert not seq.faulted, f"premature fault: {seq.fault_reason}"
    # Corrupt the sequencer's launch generation so the handle
    # lookup misses the stored (gen, slot) key.
    seq.context_launch_generation = seq.context_launch_generation + 999
    group.step(cycle + 1)
    assert seq.faulted
    assert "release invariant fault" in seq.fault_reason
    assert group.pmu.events.get("release_invariant_fault", 0) >= 1
    if group.runtime_enabled:
      assert group.fault_ring.snapshot()["count"] > 0
      assert group.fault_ring.snapshot()["latest_code"] == FaultCode.ADDRESS_FAULT.name
    for reset_cycle in range(cycle + 2, cycle + 5000):
      group.step(reset_cycle)
      if group.reset_domain.is_done:
        break
    assert group.reset_domain.is_done
    self._assert_zero_leak(group)
    group.reset()


# ---------------------------------------------------------------------------
# PR 3.5: L2 admission wait queue + release-driven cross-context wakeup
# ---------------------------------------------------------------------------


ADMISSION_WAIT_BINDINGS = {
  "A_IN": GlobalBinding("A_IN", 0x100000, L2_WAIT_BYTES, "rw"),
  "A_OUT": GlobalBinding("A_OUT", 0x200000, L2_WAIT_BYTES, "rw"),
  "B_IN": GlobalBinding("B_IN", 0x300000, L2_WAIT_BYTES, "rw"),
}


def _admission_wait_sim() -> tuple[Simulator, ModuleOp]:
  hw = HardwareConfig().with_overrides(
    hbm_fixed_latency_cycles=10, group_sram_bytes=2 * L2_WAIT_BYTES)
  sim = Simulator(
    hw,
    SimConfig(fidelity="full_memory", device_context_count=2,
              max_cycles=500000),
    enable_tracer=True)
  module = load_workload_ir("examples/scenarios/l2_admission_wait.mlir")
  return sim, module


def _admission_model_names(sim: Simulator, module: ModuleOp) -> dict[str, dict[str, str]]:
  """Lower ``module`` into the fresh group and return per-context
  formal→actual binding maps (same computation as ``_run_model``)."""
  model = lower_model_ir(module)
  for task in model.tasks.values():
    sim._assign_program_ids(task)
  sim.group.reset()
  name_maps: dict[str, dict[str, str]] = {}
  for dop in model.body:
    if dop.op != "submit":
      continue
    task = model.tasks[dop.ctx_name]
    name_maps[dop.ctx_name] = {
      formal.name: model.inputs[actual_index].name
      for formal, actual_index in zip(task.global_inputs, dop.actual_inputs)
    }
  return name_maps


class TestL2AdmissionWait:
  """PR 3.5: A fills L2 exactly, B submits in the same device-PC cycle
  and waits; A's legal input release final-free admits B in the same
  cycle while A still holds its output buffer."""

  def test_full_run_ab_overlap_release_wakes_b(self):
    sim, module = _admission_wait_sim()
    result = sim.run(module, input_bindings=ADMISSION_WAIT_BINDINGS)
    assert result.completed, result.reason
    ev = result.pmu.events
    assert ev.get("l2_admission_wait") == 1
    assert ev.get("l2_admission_wakeup") == 1
    assert ev.get("l2_admission_permanent_fault", 0) == 0
    assert ev.get("release_invariant_fault", 0) == 0
    snap = result.group_snapshot
    assert snap["pending_context_admissions"] == []
    assert snap["memory"]["l2"]["live_allocations"] == 0
    for tile_id, l1 in snap["memory"]["l1"].items():
      assert l1["allocator"]["live_allocations"] == 0, tile_id
    assert snap["memory"]["transfers"]["inflight"] == 0
    assert result.credit_invariant_ok
    for vc in snap["memory"]["noc"].values():
      assert vc["credit"] == sim.hw.noc_vc_depth
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]

    def to_cycle(us: float) -> int:
      return round(us * 1000.0 / sim.hw.cycle_ns())

    submits = [e["args"] for e in events if e.get("name") == "context_submit"]
    assert len(submits) == 2
    assert submits[0]["cycle"] == submits[1]["cycle"]  # same device-PC cycle
    b_wait = next(e["args"] for e in events
                  if e.get("name") == "context_admission_wait"
                  and e["args"].get("context") == "ctx_b")
    b_retry = next(e["args"] for e in events
                   if e.get("name") == "context_admission_retry"
                   and e["args"].get("context") == "ctx_b")
    b_admit = next(e["args"] for e in events
                   if e.get("name") == "context_admitted"
                   and e["args"].get("context") == "ctx_b")
    b_first = next(e["args"] for e in events
                   if e.get("name") == "context_first_action"
                   and e["args"].get("context") == "ctx_b")
    a_done = next(e["args"] for e in events
                  if e.get("name") == "context_done"
                  and e["args"].get("context") == "ctx_a")
    release_cycle = b_retry["capacity_change_cycle"]
    assert b_wait["cycle"] < release_cycle
    assert b_admit["cycle"] == release_cycle
    assert b_first["cycle"] == b_admit["cycle"] + 1
    assert b_admit["l2_live_allocations"] == 2  # A output still live
    b_dispatch = next(
      to_cycle(e["ts"]) for e in events
      if e.get("name") == "tile_role_dispatch" and e["args"]["ctx_id"] == 1)
    store_done = next(
      e["args"]["completion_cycle"] for e in events
      if e.get("args", {}).get("summary_kind") == "group_transfer"
      and e["args"].get("op") == "global_store")
    assert b_dispatch < store_done
    assert b_dispatch < a_done["cycle"]

  def test_wait_gating_at_signal_and_release_boundaries(self):
    """Deferred real input phases preserve the 3/4 and final-free gates."""
    sim, module = _admission_wait_sim()
    name_maps = _admission_model_names(sim, module)
    group = sim.group
    model = lower_model_ir(module)
    seq_a = group.load_context_task(
      model.tasks["ctx_a"],
      slot_index=0,
      context_name="ctx_a",
      input_bindings=ADMISSION_WAIT_BINDINGS,
      formal_bindings=name_maps["ctx_a"],
      cycle=0,
    )
    seq_b = group.load_context_task(
      model.tasks["ctx_b"],
      slot_index=1,
      context_name="ctx_b",
      input_bindings=ADMISSION_WAIT_BINDINGS,
      formal_bindings=name_maps["ctx_b"],
      cycle=0,
    )
    assert seq_a.admission_status is ContextAdmissionStatus.ACTIVE
    assert seq_b.admission_status is ContextAdmissionStatus.WAIT_CAPACITY
    assert not seq_b.faulted and not seq_b.done
    assert group.l2_sram.snapshot()["live_allocations"] == 2
    assert group.l2_sram.snapshot()["free_bytes"] == 0
    assert seq_b not in group._active_sequencers
    assert (
      seq_b.context_name,
      1,
      seq_b.context_launch_generation,
    ) not in group._live_launches
    assert group._role_l1_handles == {}
    assert all(not tile.uce.has_active_contexts() for tile in group.tiles)
    assert len(group.queues) == 0
    gen_b = seq_b.context_launch_generation
    assert not [key for key in group._l2_handles if key[0] == gen_b]

    grid_a = None
    for cycle in range(5000):
      group.step(cycle)
      grid_a = next(
        (grid for grid in group._grid_signals if grid.context_name == "ctx_a"),
        None,
      )
      if grid_a is not None:
        break
    assert grid_a is not None
    captured: list[tuple[PhaseSignal, int]] = []

    def defer_target_input(signal, emitted_cycle):
      if signal.task.grid == grid_a and signal.phase == "input_released":
        captured.append((signal, emitted_cycle))
      else:
        group._on_phase_signal(signal, emitted_cycle)

    for tile in group.tiles:
      tile.uce._phase_signal_callback = defer_target_input
    for actual_cycle in range(cycle + 1, cycle + 200000):
      group.step(actual_cycle)
      if len(captured) == 4:
        break
    assert len(captured) == 4
    assert not seq_a.faulted
    for tile in group.tiles:
      tile.uce._phase_signal_callback = group._on_phase_signal
    inputs = sorted(
      (signal for signal, _ in captured),
      key=lambda signal: signal.task.task_id,
    )

    delivery_cycle = actual_cycle + 1
    for signal in inputs[:3]:
      group._on_phase_signal(signal, delivery_cycle)
      delivery_cycle += 1
    group.step(delivery_cycle)
    assert seq_b.admission_status is ContextAdmissionStatus.WAIT_CAPACITY
    assert group._pending_context_admissions[0].sequencer is seq_b

    delivery_cycle += 1
    group._on_phase_signal(inputs[3], delivery_cycle)
    assert seq_b.admission_status is ContextAdmissionStatus.WAIT_CAPACITY
    admit_cycle = None
    for release_cycle in range(delivery_cycle + 1, delivery_cycle + 1000):
      group.step(release_cycle)
      if seq_b.admission_status is ContextAdmissionStatus.ACTIVE:
        admit_cycle = release_cycle
        break
    assert admit_cycle is not None
    assert group._l2_capacity_change_cycle == admit_cycle
    out_handle = group._l2_handles[
      (seq_a.context_launch_generation, "a_output")
    ]
    assert not group.l2_sram.is_released(out_handle)
    assert group._grid_l2_pins
    assert seq_b.action_index == 0
    group.step(admit_cycle + 1)
    assert seq_b.action_index == 1
    for finish_cycle in range(admit_cycle + 2, admit_cycle + 500000):
      group.step(finish_cycle)
      if seq_a.done and seq_b.done:
        break
    assert seq_a.done and seq_b.done, (seq_a.fault_reason, seq_b.fault_reason)
    assert not group._pending_context_admissions
    assert group.l2_sram.snapshot()["live_allocations"] == 0
    for tile in group.tiles:
      assert tile.l1_allocator.snapshot()["live_allocations"] == 0
    assert not group._grid_l2_pins
    assert group.transfer_manager.inflight_count == 0
    group.reset()


class TestAdmissionFaultAndQueue:
  """PR 3.5: permanent oversized faults never queue; transient waits
  are strict FIFO; reset cancels pending tickets."""

  def test_standalone_load_sets_active_admission_status(self):
    group = TileGroup(HardwareConfig(), fidelity="runtime")
    task = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    group.load_task(task, input_bindings=POW_BINDINGS)
    assert group.sequencer.admission_status is ContextAdmissionStatus.ACTIVE
    group.reset()


  def test_oversized_bundle_faults_without_queueing(self):
    sim = Simulator(
      HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10,
                                      group_sram_bytes=1024),
      SimConfig(fidelity="full_memory", device_context_count=2,
                max_cycles=10000))
    result = sim.run(make_two_context_model(), input_bindings=MODEL_BINDINGS)
    assert not result.completed
    assert "L2 capacity fault" in result.reason
    assert not sim.group._pending_context_admissions
    assert sim.group.pmu.events.get("l2_admission_permanent_fault", 0) >= 1
    assert sim.group.pmu.events.get("l2_admission_wait", 0) == 0

  def test_three_waiters_strict_fifo_order(self):
    """Three same-size waiters behind a full L2: a final-free admits
    strictly the head; head-of-line waiters never bypass."""
    chunk = L2_WAIT_BYTES
    hw = HardwareConfig().with_overrides(
      hbm_fixed_latency_cycles=10, group_sram_bytes=2 * chunk)
    group = TileGroup(hw, fidelity="full_memory", context_count=2)
    blocker = lower_workload_ir(PowWorkload(num_group_chunks=2).module)
    group.load_task(blocker, input_bindings=POW_BINDINGS)
    assert group.l2_sram.snapshot()["free_bytes"] == 0
    group._next_launch_id = 100  # keep waiter generations distinct
    waiters = []
    for i in range(3):
      task = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
      seq = group.load_context_task(
        task, slot_index=1, context_name=f"w{i}",
        input_bindings=POW_BINDINGS, cycle=0)
      waiters.append(seq)
      assert seq.admission_status is ContextAdmissionStatus.WAIT_CAPACITY
    assert len(group._pending_context_admissions) == 3
    # release one blocker buffer, then run the retry barrier once
    handles = [h for (gen, _slot), h in group._l2_handles.items()
               if gen == group.sequencer.context_launch_generation]
    assert len(handles) == 2
    assert group.l2_sram.request_release(handles[0], handles[0].owner, 1)
    group._l2_capacity_change_cycle = 1
    group._retry_pending_context_admissions(1)
    for ticket in group._pending_activations:
      group._activate_admitted_context(ticket, 1)
    group._pending_activations.clear()
    assert waiters[0].admission_status is ContextAdmissionStatus.ACTIVE
    assert waiters[1].admission_status is ContextAdmissionStatus.WAIT_CAPACITY
    assert waiters[2].admission_status is ContextAdmissionStatus.WAIT_CAPACITY
    assert len(group._pending_context_admissions) == 2
    # duplicate notification for the same release: no new pass, no
    # double commit, no extra retry event
    retries_before = group.pmu.events.get("l2_admission_retry", 0)
    group._retry_pending_context_admissions(1)
    assert waiters[1].admission_status is ContextAdmissionStatus.WAIT_CAPACITY
    assert len(group._pending_context_admissions) == 2
    assert group.pmu.events.get("l2_admission_retry", 0) == retries_before
    assert group.l2_sram.snapshot()["live_allocations"] == 2
    group.reset()

  def test_fragmentation_waiter_wakes_after_release_merges_extent(self):
    """A final-free merges an extent and wakes the fragmentation waiter."""
    from pipeline_validator.execution_ir import ExecL2Buffer, ExecReleaseRequest

    hw = HardwareConfig().with_overrides(
      group_sram_bytes=64, group_sram_banks=2)
    group = TileGroup(hw, fidelity="full_memory", context_count=2)
    blocker = ExecTileGroupTask(
      name="fragmentation_blocker",
      l2_buffers=(
        ExecL2Buffer("a", (16,), "i8", "in", 1, 32, 16),
        ExecL2Buffer("b", (16,), "i8", "in", 1, 32, 16),
      ),
    )
    waiter = ExecTileGroupTask(
      name="fragmentation_waiter",
      l2_buffers=(
        ExecL2Buffer("merged", (32,), "i8", "in", 1, 32, 32),
      ),
    )
    seq_a = group.load_context_task(
      blocker, slot_index=0, context_name="fragmentation_blocker", cycle=0)
    seq_b = group.load_context_task(
      waiter, slot_index=1, context_name="fragmentation_waiter", cycle=0)
    assert seq_a.admission_status is ContextAdmissionStatus.ACTIVE
    assert seq_b.admission_status is ContextAdmissionStatus.WAIT_CAPACITY
    assert group.l2_sram.snapshot()["free_bytes"] == 32

    group.release_l2(
      ExecReleaseRequest(
        buffer_slot="a",
        buffer_role="in",
        reader_dispatch_ordinals=(),
        writer_dispatch_ordinals=(),
        dependency_events=(),
      ),
      sequencer=seq_a,
      cycle=5,
    )
    group._retry_pending_context_admissions(5)
    assert group._pending_activations
    for ticket in group._pending_activations:
      group._activate_admitted_context(ticket, 5)
    group._pending_activations.clear()
    assert seq_b.admission_status is ContextAdmissionStatus.ACTIVE
    assert not group._pending_context_admissions
    assert (seq_b.context_launch_generation, "merged") in group._l2_handles
    group.reset()

  def test_reset_unwinds_committed_staged_activation(self):
    """A ticket between retry and activation already owns L2; reset must
    release those committed handles instead of treating it as a waiter."""
    from pipeline_validator.execution_ir import ExecL2Buffer, ExecReleaseRequest

    hw = HardwareConfig().with_overrides(
      group_sram_bytes=64, group_sram_banks=2)
    group = TileGroup(hw, fidelity="full_memory", context_count=2)
    blocker = ExecTileGroupTask(
      name="staged_blocker",
      l2_buffers=(
        ExecL2Buffer("a", (16,), "i8", "in", 1, 32, 16),
        ExecL2Buffer("b", (16,), "i8", "in", 1, 32, 16),
      ),
    )
    waiter = ExecTileGroupTask(
      name="staged_waiter",
      l2_buffers=(
        ExecL2Buffer("merged", (32,), "i8", "in", 1, 32, 32),
      ),
    )
    seq_a = group.load_context_task(
      blocker, slot_index=0, context_name="staged_blocker", cycle=0)
    seq_b = group.load_context_task(
      waiter, slot_index=1, context_name="staged_waiter", cycle=0)
    group.release_l2(
      ExecReleaseRequest(
        buffer_slot="a",
        buffer_role="in",
        reader_dispatch_ordinals=(),
        writer_dispatch_ordinals=(),
        dependency_events=(),
      ),
      sequencer=seq_a,
      cycle=5,
    )
    group._retry_pending_context_admissions(5)
    assert group._pending_activations
    assert (seq_b.context_launch_generation, "merged") in group._l2_handles
    assert group.l2_sram.snapshot()["live_allocations"] == 2

    group.reset()
    assert seq_b.admission_status is ContextAdmissionStatus.CANCELLED
    assert not group._pending_context_admissions
    assert not group._pending_activations
    assert not group._l2_handles
    assert group.l2_sram.snapshot()["live_allocations"] == 0

  def test_reset_cancels_pending_without_release(self):
    from pipeline_validator.trace import Tracer

    hw = HardwareConfig().with_overrides(
      hbm_fixed_latency_cycles=10, group_sram_bytes=L2_WAIT_BYTES)
    tracer = Tracer(hw)
    group = TileGroup(
      hw, tracer=tracer, fidelity="full_memory", context_count=2)
    blocker = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    group.load_task(blocker, input_bindings=POW_BINDINGS)
    waiter_task = lower_workload_ir(PowWorkload(num_group_chunks=1).module)
    seq_b = group.load_context_task(
      waiter_task, slot_index=1, context_name="waiter",
      input_bindings=POW_BINDINGS, cycle=7)
    assert seq_b.admission_status is ContextAdmissionStatus.WAIT_CAPACITY
    gen_b = seq_b.context_launch_generation
    assert len(group._pending_context_admissions) == 1
    # the waiting ticket owns no allocation — reset must not release it
    assert not [k for k in group._l2_handles if k[0] == gen_b]
    for cycle in range(8, 11):
      group.step(cycle)
      assert seq_b.admission_status is ContextAdmissionStatus.WAIT_CAPACITY
    group.reset()
    assert seq_b.admission_status is ContextAdmissionStatus.CANCELLED
    assert group.pmu.named_cycles["l2_admission_wait_cycles"] == 3
    events = json.loads(tracer.to_chrome_json())["traceEvents"]
    cancelled = next(
      e["args"] for e in events
      if e.get("name") == "context_admission_cancelled")
    assert cancelled["cycle"] == 10
    assert not group._pending_context_admissions
    assert not group._pending_activations
    assert not group._live_launches
    assert group.l2_sram.snapshot()["live_allocations"] == 0
    for t in group.tiles:
      assert t.l1_allocator.snapshot()["live_allocations"] == 0
    assert not group._grid_l2_pins
    assert group.transfer_manager.inflight_count == 0

  def test_active_fault_cancels_pending_model_context(self, monkeypatch):
    """An active A fault preserves the original fault result while the
    ResetDomain cancels pending B and drains every resource."""
    from pipeline_validator.tile_group_sequencer import TileGroupSequencer

    sim, module = _admission_wait_sim()
    original_step = sim.group.step
    pending: list[TileGroupSequencer] = []
    fault_injected = False

    def faulting_step(cycle: int) -> bool:
      nonlocal fault_injected
      done = original_step(cycle)
      if not fault_injected and sim.group._pending_context_admissions:
        active_a = next(
          seq for seq in sim.group._active_sequencers
          if seq.context_name == "ctx_a")
        pending.append(
          sim.group._pending_context_admissions[0].sequencer)
        active_a.faulted = True
        active_a.fault_reason = "injected active context fault"
        active_a.done = True
        fault_injected = True
      return done

    monkeypatch.setattr(sim.group, "step", faulting_step)
    result = sim.run(module, input_bindings=ADMISSION_WAIT_BINDINGS)
    assert not result.completed
    assert "injected active context fault" in result.reason
    assert pending
    assert pending[0].admission_status is ContextAdmissionStatus.CANCELLED
    assert sim.group.reset_domain.is_done
    snap = result.group_snapshot
    assert not snap["pending_context_admissions"]
    assert snap["memory"]["l2"]["live_allocations"] == 0
    assert snap["memory"]["transfers"]["inflight"] == 0
    for l1 in snap["memory"]["l1"].values():
      assert l1["allocator"]["live_allocations"] == 0
    for vc in snap["memory"]["noc"].values():
      assert vc["credit"] == sim.hw.noc_vc_depth

  def test_same_formal_name_bindings_stay_launch_scoped(self):
    """Both contexts use the formal name 'Y' (A's store destination and
    B's prefetch source) with different actuals.  B's submit must not
    overwrite A's mapping: A's final store still lands in A's IOVA
    range and B's prefetch in B's."""
    text = Path("examples/scenarios/l2_admission_wait.mlir").read_text()
    # rename A's output formal and B's input formal to the shared name Y
    start_a = text.index("nest.context @ctx_a")
    start_b = text.index("nest.context @ctx_b")
    prog_idx = text.index("nexus.program")
    ctx_a = text[start_a:start_b].replace("%A_OUT", "%Y")
    ctx_b = text[start_b:prog_idx].replace("%B_IN", "%Y")
    module = parse_workload_ir(
      text[:start_a] + ctx_a + ctx_b + text[prog_idx:])
    sim, _ = _admission_wait_sim()
    name_maps = _admission_model_names(sim, module)
    group = sim.group
    model = lower_model_ir(module)
    assert name_maps["ctx_a"]["Y"] == "A_OUT"
    assert name_maps["ctx_b"]["Y"] == "B_IN"
    seq_a = group.load_context_task(
      model.tasks["ctx_a"], slot_index=0, context_name="ctx_a",
      input_bindings=ADMISSION_WAIT_BINDINGS,
      formal_bindings=name_maps["ctx_a"], cycle=0)
    seq_b = group.load_context_task(
      model.tasks["ctx_b"], slot_index=1, context_name="ctx_b",
      input_bindings=ADMISSION_WAIT_BINDINGS,
      formal_bindings=name_maps["ctx_b"], cycle=0)
    assert seq_b.admission_status is ContextAdmissionStatus.WAIT_CAPACITY
    for c in range(500000):
      group.step(c)
      if seq_a.done and seq_b.done:
        break
    assert seq_a.done and seq_b.done, (seq_a.fault_reason, seq_b.fault_reason)
    events = json.loads(sim.tracer.to_chrome_json())["traceEvents"]
    a_out = ADMISSION_WAIT_BINDINGS["A_OUT"]
    b_in = ADMISSION_WAIT_BINDINGS["B_IN"]
    a_in = ADMISSION_WAIT_BINDINGS["A_IN"]
    store_addrs = [
      e["args"]["destination_address"] for e in events
      if e.get("args", {}).get("summary_kind") == "group_transfer"
      and e["args"].get("op") == "global_store"
    ]
    pref_srcs = [
      e["args"]["source_address"] for e in events
      if e.get("args", {}).get("summary_kind") == "group_transfer"
      and e["args"].get("op") == "prefetch"
    ]
    assert store_addrs, "no store transactions traced"
    assert all(a_out.base_iova <= a < a_out.base_iova + a_out.size_bytes
               for a in store_addrs), store_addrs
    assert any(b_in.base_iova <= a < b_in.base_iova + b_in.size_bytes
               for a in pref_srcs), pref_srcs
    assert any(a_in.base_iova <= a < a_in.base_iova + a_in.size_bytes
               for a in pref_srcs), pref_srcs
    group.reset()


class TestTileFreeRuntime:
  BINDINGS = {"input": GlobalBinding("input", 0x100000, 8192, "r")}

  @staticmethod
  def make_module(explicit_free=True):
    from pipeline_validator.dialects.elenor import TileFreeOp

    programs = []
    for name, iterations in (("free_holder", 20), ("free_peer", 40)):
      program = TileProgramDefOp(
        name, [], arg_types=[NestTask(), NestBuffer.of([1, 64, 64], "bf16")],
        arg_names=["task", "input"],
      )
      task_arg, buffer_arg = program.body.block.args
      view = TileSubviewOp(
        buffer_arg, task_arg, 0, [0, 0, 0], [1, 64, 64], [1, 1, 1],
        NestL2View.of([1, 64, 64], "bf16"),
      )
      scratch = TileAllocOp([64, 64], "bf16", alignment=256)
      work = TileAllocOp([64, 64], "bf16", alignment=256)
      load = TileLoadOp(view.result, scratch.result, "scratch_loaded")
      ops = [view, scratch]
      if name == "free_holder":
        ops.append(work)
      ops.extend([load, TileAwaitOp([load.result])])
      if name == "free_holder":
        work_load = TileLoadOp(view.result, work.result, "work_loaded")
        ops.extend([work_load, TileAwaitOp([work_load.result])])
        if explicit_free:
          ops.append(TileFreeOp(scratch.result))
      ops.append(TileSignalOp("input_released", task_arg))
      for index in range(iterations):
        compute = TileEvuOp("relu", 16448, f"compute_{index}")
        ops.extend([compute, TileAwaitOp([compute.result])])
      # Peer intentionally keeps its allocation until automatic terminal cleanup.
      if name == "free_holder" and explicit_free:
        ops.append(TileFreeOp(work.result))
      ops.append(TileReturnOp())
      program.body.block.add_ops(ops)
      programs.append(program)
    context = NestContextOp(
      "free_context", [], placement=1,
      arg_types=[NestGlobalMemref.of([1, 64, 64], "bf16")], arg_names=["input"],
    )
    buf = NestAllocOp("input", "in", [1, 64, 64], "bf16", alignment=256)
    global_view = NestSubviewOp(
      context.body.block.args[0], [0, 0, 0], [1, 64, 64], [1, 1, 1],
      NestGlobalView.of([1, 64, 64], "bf16"),
    )
    pref = NestPrefetchOp(global_view.result, buf.result, "prefetched")
    tasks = NestTaskRangeOp(0, 1)
    holder = NestDispatchOp(
      "free_holder", tasks.result, [], [buf.result], [], "holder_grid", "holder_read", "",
      bindings=[buf.result], signal_policy={"input_released": "all_tasks"},
      depends_on=[pref.result], context_id=0,
    )
    peer = NestDispatchOp(
      "free_peer", tasks.result, [], [buf.result], [], "peer_grid", "peer_read", "",
      bindings=[buf.result], signal_policy={"input_released": "all_tasks"},
      depends_on=[holder.input_released], context_id=1,
    )
    context.body.block.add_ops([
      buf, global_view, pref, tasks, holder, peer,
      NestReleaseOp(buf.result, depends_on=[holder.input_released, peer.input_released, pref.result]),
      NestAwaitOp([holder.grid_done, peer.grid_done]), NestReturnOp(),
    ])
    return ModuleOp([*programs, context])

  @staticmethod
  def make_sim(fidelity):
    return Simulator(
      HardwareConfig().with_overrides(tile_l1_bytes=16384, hbm_fixed_latency_cycles=10),
      SimConfig(fidelity=fidelity, context_count=2, memory_trace=True, max_cycles=200000),
      enable_tracer=True,
    )

  @staticmethod
  def assert_empty(group):
    if group.runtime_enabled:
      assert group.l2_sram.snapshot()["live_allocations"] == 0
    assert group.transfer_manager.inflight_count == 0
    for tile in group.tiles:
      assert tile.l1_allocator.snapshot()["live_allocations"] == 0
    assert not group._grid_l2_pins
    assert not any(group._role_l1_handles.values())
    assert group.credit_invariants_hold()

  @pytest.mark.parametrize("fidelity", ["runtime", "full_memory"])
  def test_early_free_reuses_extent_before_holder_returns(self, fidelity):
    baseline = self.make_sim(fidelity)
    without_free = baseline.run(self.make_module(False), input_bindings=self.BINDINGS)
    assert not without_free.completed
    assert baseline.group.reset_domain.is_done
    self.assert_empty(baseline.group)

    sim = self.make_sim(fidelity)
    result = sim.run(self.make_module(), input_bindings=self.BINDINGS)
    assert result.completed, result.reason
    assert result.tracer is not None
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    allocations = [e for e in events if e["name"] == "l1_alloc" and e["ph"] == "i"]
    scratch_allocs = sorted(
      (e for e in allocations if e["args"]["buffer_id"] == "l1:0"),
      key=lambda e: e["args"]["allocate_cycle"],
    )
    assert len(scratch_allocs) == 2
    original, replacement = scratch_allocs
    assert original["args"]["allocation_id"] != replacement["args"]["allocation_id"]
    assert original["args"]["base_address"] == replacement["args"]["base_address"]
    releases = {
      e["args"]["allocation_id"]: e for e in events
      if e["name"] == "l1_release" and e["ph"] == "i"
    }
    holder_done = next(
      e for e in events if e["name"] == "tile_done"
      and e["args"]["event_id"].endswith("holder_grid")
    )
    assert (
      releases[original["args"]["allocation_id"]]["ts"]
      < replacement["ts"]
      < holder_done["ts"]
      < releases[replacement["args"]["allocation_id"]]["ts"]
    )
    original_loads = [
      e["args"] for e in events if e["ph"] == "X"
      and e.get("args", {}).get("op") == "tile_load"
      and e["args"].get("role_event_id", "").endswith("holder_grid")
      and e["args"]["destination_address"] == original["args"]["base_address"]
    ]
    assert original_loads
    freed_cycle = round(
      releases[original["args"]["allocation_id"]]["ts"] * 1000 / sim.hw.cycle_ns()
    )
    assert max(e["completion_cycle"] for e in original_loads) < freed_cycle
    assert set(releases) == {e["args"]["allocation_id"] for e in allocations}
    self.assert_empty(sim.group)
    assert result.credit_invariant_ok
    result.tracer.assert_well_formed()

  @pytest.mark.parametrize("fidelity", ["timing_only", "runtime", "full_memory"])
  @pytest.mark.parametrize("violation", ["pending-load", "double-free", "load-after-free"])
  def test_raw_execution_rejects_unsafe_free(self, fidelity, violation):
    sim = self.make_sim(fidelity)
    task = lower_workload_ir(self.make_module())
    program = next(b.tile_program for b in task.role_bindings.values() if b.tile_program.name == "free_holder")
    free_index = next(i for i, ins in enumerate(program.insts) if ins.op == ExecTileOp.FREE_L1)
    scratch_name = program.insts[free_index].args[0]
    if violation == "pending-load":
      wait_index = next(i for i, ins in enumerate(program.insts) if ins.op == ExecTileOp.WAIT)
      program.insts[wait_index] = ExecTileInst(ExecTileOp.FREE_L1, args=(scratch_name,))
    elif violation == "double-free":
      program.insts.insert(len(program.insts) - 1, ExecTileInst(ExecTileOp.FREE_L1, args=(scratch_name,)))
    else:
      load = next(ins for ins in program.insts if ins.op == ExecTileOp.LAUNCH_MFE)
      program.insts.insert(
        free_index + 1, ExecTileInst(ExecTileOp.LAUNCH_MFE, dst="illegal_reload", args=load.args)
      )
    sim._assign_program_ids(task)
    sim.group.load_task(task, input_bindings=self.BINDINGS)
    for cycle in range(200000):
      sim.group.step(cycle)
      if sim.group.sequencer.faulted:
        break
    assert sim.group.sequencer.faulted
    # A stale free from holder must not release peer's same-address allocation.
    if violation == "double-free" and fidelity != "timing_only":
      peer_memory = sim.group.tiles[0].uce.contexts[1].memory
      assert peer_memory is not None and peer_memory.l1_handles
      for handle in peer_memory.l1_handles.values():
        sim.group.tiles[0].l1_allocator.assert_live(handle, handle.owner)
    sim.group.release_context_memory(cycle + 1)
    self.assert_empty(sim.group)

  @pytest.mark.parametrize("fidelity", ["timing_only", "runtime", "full_memory"])
  def test_raw_free_rejects_pending_gather(self, fidelity):
    path = Path(__file__).resolve().parents[2] / "examples/workloads/gather_profiled.mlir"
    task = lower_model_ir(load_workload_ir(path)).tasks["gather_context"]
    program = next(iter(task.role_bindings.values())).tile_program
    free = next(ins for ins in program.insts if ins.op == ExecTileOp.FREE_L1)
    wait_index = next(
      i for i, ins in enumerate(program.insts)
      if ins.op == ExecTileOp.WAIT and ins.args == ("gather_done",)
    )
    program.insts[wait_index] = ExecTileInst(ExecTileOp.FREE_L1, args=free.args)
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity=fidelity, memory_trace=True, max_cycles=100000),
      enable_tracer=True,
    )
    sim._assign_program_ids(task)
    sim.group.load_task(task, input_bindings={
      "table": GlobalBinding("table", 0x200000, 8388608, "r"),
      "indices": GlobalBinding("indices", 0xA00000, 4096, "r"),
      "output": GlobalBinding("output", 0xB00000, 256, "w"),
    })
    for cycle in range(100000):
      sim.group.step(cycle)
      if sim.group.sequencer.faulted:
        break
    assert sim.group.sequencer.faulted
    memory = sim.group.tiles[0].uce.contexts[0].memory
    assert memory is not None
    if fidelity != "timing_only":
      assert len(memory.l1_handles) == 2
      for handle in memory.l1_handles.values():
        sim.group.tiles[0].l1_allocator.assert_live(handle, handle.owner)
    sim.group.release_context_memory(cycle + 1)
    self.assert_empty(sim.group)

  @pytest.mark.parametrize("violation", ["stale-generation", "wrong-owner", "pinned", "inflight"])
  def test_free_preflight_is_atomic(self, monkeypatch, violation):
    from dataclasses import replace

    from pipeline_validator.memory.transfer import MemoryTransaction, ResolvedMemoryView, TransferOp

    sim = self.make_sim("full_memory")
    task = lower_workload_ir(self.make_module())
    sim._assign_program_ids(task)
    group = sim.group
    group.load_task(task, input_bindings=self.BINDINGS)
    tile = group.tiles[0]
    original_issue = tile.uce._issue_context
    captured = []

    def stop_at_free(ctx, cycle, compute_tile):
      if ctx.program is not None and ctx.program.insts[ctx.pc].op == ExecTileOp.FREE_L1:
        captured.append(ctx)
        return
      original_issue(ctx, cycle, compute_tile)

    monkeypatch.setattr(tile.uce, "_issue_context", stop_at_free)
    for cycle in range(100000):
      group.step(cycle)
      if captured:
        break
    assert captured
    ctx = captured[0]
    assert ctx.memory is not None
    name = ctx.program.insts[ctx.pc].args[0]
    handle = ctx.memory.l1_handles[name]
    if violation == "stale-generation":
      ctx.memory.l1_handles[name] = replace(handle, generation=handle.generation + 1)
    elif violation == "wrong-owner":
      ctx.memory.l1_handles[name] = replace(
        handle, owner=replace(handle.owner, hardware_context_id=1)
      )
    elif violation == "pinned":
      tile.l1_allocator.pin(handle, "external-test-reader")
    else:
      source = ctx.memory.l2_formal_handles[1]
      transaction = MemoryTransaction(
        transaction_id="extra_l1_access", op=TransferOp.TILE_LOAD, issuer=handle.owner,
        src=ResolvedMemoryView(source, 0, source.size_bytes, source.base_address, source.bank_segments),
        dst=ResolvedMemoryView(handle, 0, handle.size_bytes, handle.base_address, handle.bank_segments),
        bytes_total=handle.size_bytes, completion_event="extra_l1_done", tile_id=0,
      )
      group.transfer_manager.submit(transaction, cycle)
      group.transfer_manager.step(cycle + 1)
      assert group.transfer_manager.has_inflight_access(handle)
    before = tile.l1_allocator.snapshot()
    slot_ids = [slot.allocation_id for slot in tile.l1_frames[0].slots]
    original_issue(ctx, cycle + 2, tile)
    assert ctx.state.name == "FAULT"
    tile.l1_allocator.assert_live(handle, handle.owner)
    assert tile.l1_allocator.snapshot()["allocated_bytes"] == before["allocated_bytes"]
    assert tile.l1_allocator.snapshot()["pending_release"] == before["pending_release"] == 0
    assert [slot.allocation_id for slot in tile.l1_frames[0].slots] == slot_ids
    ctx.memory.l1_handles[name] = handle
    if violation == "pinned":
      tile.l1_allocator.unpin(handle, "external-test-reader", cycle + 3)
    group.release_context_memory(cycle + 3)
    self.assert_empty(group)
