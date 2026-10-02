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
from typing import ClassVar

import pytest
from xdsl.dialects.builtin import ModuleOp

from pipeline_validator.compiled_program import WorkloadInfo
from pipeline_validator.compiler import compile_program
from pipeline_validator.compiler.resources import conservative_arena_bytes, layout_buffers
from pipeline_validator.config import GroupSchedulerConfig, HardwareConfig, SimConfig
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
  TileIndexedMapAttr,
  TileLoadOp,
  TileProgramDefOp,
  TileReturnOp,
  TileSignalOp,
  TileStoreOp,
  TileSubviewOp,
)
from pipeline_validator.execution_ir import (
  ExecGroupActionOp,
  ExecTileGroupTask,
  ExecTileOp,
  GlobalBinding,
  GridInstanceId,
  PhaseSignal,
  TaskIdentity,
)
from pipeline_validator.loader import load_program
from pipeline_validator.memory import (
  L2SRAM,
  AdmissionFailure,
  MemoryInvariantError,
  NoCRouter,
  PayloadTracker,
)
from pipeline_validator.memory.arena import ArenaPool, RootInvocation
from pipeline_validator.profiles import CacheRequirement, ContextResources, TileResources, build_registry
from pipeline_validator.runtime import EventStatus, EventTable, FaultCode, FaultRing
from pipeline_validator.runtime.fault_ring import FaultDomain, FaultRecord
from pipeline_validator.runtime.reset_domain import ResetDomain, ResetRequest, ResetState
from pipeline_validator.simulator import SimResult, Simulator
from pipeline_validator.tile import TileUCE
from pipeline_validator.tile_group import TileGroup
from pipeline_validator.workload_builders import make_pow_tile_program
from pipeline_validator.workload_ir import load_workload_ir, parse_workload_ir, print_workload_ir
from pipeline_validator.workloads import ALL_WORKLOADS, PowWorkload


def tile_resources(
  bytes_per_context: int = 0,
  *,
  allowed: tuple[int, ...] = (0, 1, 2),
  l1_cache: CacheRequirement | None = None,
  l2_cache: CacheRequirement | None = None,
) -> TileResources:
  return TileResources(allowed, bytes_per_context, l1_cache=l1_cache, l2_cache=l2_cache)


def context_resources(
  logical_tasks: int = 0,
  l2_spm_bytes: int = 0,
  *,
  l2_mode: int = 0,
  allowed: tuple[int, ...] = (0, 1, 2),
  requested_contexts_per_tile: int = 1,
  l2_cache: CacheRequirement | None = None,
) -> ContextResources:
  return ContextResources(
    l2_mode, allowed, logical_tasks, l2_spm_bytes, requested_contexts_per_tile, l2_cache
  )


def authored_arena_bytes(
  hw: HardwareConfig, level: str, buffers: list[tuple[int, int]], *, mode: int = 0
) -> int:
  if not buffers:
    return 0
  return conservative_arena_bytes(buffers, build_registry(hw).profile(level, mode))


def compile_source(
  module: ModuleOp, hw: HardwareConfig, sim: SimConfig, *, workload_info: WorkloadInfo | None = None
):
  return compile_program(module, hw, sim, workload_info=workload_info)


def run_source(
  simulator: Simulator,
  module: ModuleOp,
  bindings: dict[str, GlobalBinding] | None = None,
  *,
  workload_info: WorkloadInfo | None = None,
):
  artifact = compile_source(module, simulator.hw, simulator.sim, workload_info=workload_info)
  loaded = load_program(artifact, simulator.hw, simulator.sim, actual_bindings=bindings)
  return simulator.run(loaded)


def assert_run_rejected_without_group_mutation(
  simulator: Simulator, module: ModuleOp, bindings: dict[str, GlobalBinding] | None = None
) -> None:
  """A rejected compile/load contract must not enter or mutate Runtime."""
  before = simulator.group.snapshot()
  with pytest.raises(ValueError):
    run_source(simulator, module, bindings)
  assert simulator.group.snapshot() == before


def compiled_entry(module: ModuleOp, hw: HardwareConfig, sim: SimConfig):
  return compile_source(module, hw, sim).entry


def begin_group_program(
  group: TileGroup, artifact, bindings: dict[str, GlobalBinding] | None = None
) -> None:
  cycle = 0
  while not group.profile_controller.initialized:
    group.profile_controller.step(cycle)
    cycle += 1
  group.begin_launch(artifact, bindings or {})


def task_named(entry, name: str) -> ExecTileGroupTask:
  tasks = entry.tasks.values() if hasattr(entry, "tasks") else (entry,)
  matches = [task for task in tasks if task.name == name]
  assert len(matches) == 1
  return matches[0]


def prepare_group_source(
  group: TileGroup,
  module: ModuleOp,
  bindings: dict[str, GlobalBinding] | None = None,
  *,
  sim: SimConfig | None = None,
):
  sim = sim or SimConfig(
    fidelity=group.fidelity, context_count=len(group.tiles[0].l1_frames), group=group.scheduler_config
  )
  artifact = compile_source(module, group.cfg, sim)
  begin_group_program(group, artifact, bindings)
  return artifact


def finish_group_and_reset(group: TileGroup, *, max_cycles: int = 2000000) -> None:
  """Reach natural quiescence before invoking the explicit reset entry."""
  start = group._last_step_cycle + 1
  for cycle in range(start, start + max_cycles):
    if group.step(cycle):
      break
  assert not group._active_sequencers
  assert not group._grid_routes
  assert not group._task_leases
  assert not group._l2_arenas
  assert group.transfer_manager.inflight_count == 0
  group.reset()


def fault_drain_and_reset(group: TileGroup, *, max_cycles: int = 2000000) -> None:
  """Isolate live work through the real fault drain, then recover explicitly."""
  start = group._last_step_cycle + 1
  if not (group.reset_domain.is_active or group.reset_domain.is_done):
    group.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=start, desc_id="test-requested isolation")
  for cycle in range(start, start + max_cycles):
    group.step(cycle)
    if group.reset_domain.is_done:
      break
  assert group.reset_domain.is_done
  assert not group._active_sequencers
  assert not group._grid_routes
  assert not group._task_leases
  assert not group._l2_arenas
  assert group.transfer_manager.inflight_count == 0
  group.reset()


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
  "indices": GlobalBinding("indices", 0x500000, 16, "r"),
}


def assert_uce_instructions_issue_once(result: SimResult) -> list[dict]:
  """Engine-launch instructions issue once; blocked probes are not issues."""
  assert result.tracer is not None
  events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
  launch_ops = {
    ExecTileOp.LAUNCH_BOA.value,
    ExecTileOp.LAUNCH_EVU.value,
    ExecTileOp.LAUNCH_MFE.value,
    ExecTileOp.LAUNCH_USE.value,
    ExecTileOp.LAUNCH_GATHER.value,
  }
  issues = [
    event
    for event in events
    if event.get("name") == "uce_issue" and event.get("args", {}).get("op") in launch_ops
  ]
  issue_counts = Counter(
    (
      event["pid"],
      event["args"]["program"],
      event["args"]["ctx_id"],
      event["args"]["pc"],
      event["args"]["op"],
    )
    for event in issues
  )
  assert issue_counts
  assert set(issue_counts.values()) == {1}
  return issues


def assert_model_launch_lifecycle(result: SimResult) -> tuple[list[dict], dict[int, dict]]:
  """Assert the CPU and hardware-port lifecycle records agree with the trace."""
  assert result.tracer is not None
  launch_records = result.device_snapshot["launch_records"]
  port_records = {
    record["request_id"]: record for record in result.device_snapshot["port"]["request_records"]
  }
  request_ids = {record["request_id"] for record in launch_records}
  assert request_ids == set(port_records)
  for record in launch_records:
    assert {
      "request_id",
      "context",
      "event",
      "submit_cycle",
      "admission_cycle",
      "completion_cycle",
      "status",
    } <= record.keys()
    port_record = port_records[record["request_id"]]
    assert {
      "slot_index",
      "request_id",
      "context",
      "submit_cycle",
      "active_cycle",
      "completion_cycle",
      "status",
    } <= port_record.keys()
    assert port_record["context"] == record["context"]
    assert port_record["status"] == record["status"] == "success"
    assert record["submit_cycle"] <= record["admission_cycle"]
    assert record["admission_cycle"] <= port_record["active_cycle"]
    assert port_record["active_cycle"] <= record["completion_cycle"]

  events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
  cpu_pid = next(
    event["pid"]
    for event in events
    if event.get("name") == "process_name" and event.get("args", {}).get("name") == "CPU Device"
  )
  for phase in ("launch_submit", "launch_admission", "launch_completion"):
    phase_events = [event for event in events if event.get("name") == phase]
    assert len(phase_events) == len(request_ids)
    assert {event["args"]["request_id"] for event in phase_events} == request_ids
    assert all(event["pid"] == cpu_pid for event in phase_events)
  return launch_records, port_records


def make_gather_module(
  index_rows: list[int],
  *,
  include_evu_context: bool = False,
  window_entries: int = 4,
  segment: int = 16,
) -> ModuleOp:
  """Address-resolved gather: one 16 B segment per index row."""
  cache = CacheRequirement(False, "read", "allowed", 65536)
  program = TileProgramDefOp(
    "gather_tile",
    TileResources((1, 2), 2048, l1_cache=cache, l2_cache=cache),
    arg_types=[NestTask(), NestGlobalView.of([4096], "i8"), NestBuffer.of([len(index_rows)], "i32")],
    arg_names=["task", "table", "indices_l2"],
  )
  _task, table, indices_l2 = program.body.block.args
  indices_view = TileSubviewOp(indices_l2, None, None, [0], [len(index_rows)], [1], NestL2View.of([len(index_rows)], "i32"))
  indices = TileAllocOp([len(index_rows)], "i32")
  destination = TileAllocOp([len(index_rows) * segment], "i8")
  load_indices = TileLoadOp(indices_view.result, indices.result, "indices_ready")
  gather = TileGatherOp(
    table,
    indices.result,
    destination.result,
    "gather_done",
    address_map=TileIndexedMapAttr.of(64, 0, 0, 1, 0, segment),
    window_entries=window_entries,
  )
  program.body.block.add_ops(
    [
      indices_view,
      indices,
      destination,
      load_indices,
      TileAwaitOp([load_indices.result]),
      TileSignalOp("input_released", _task),
      gather,
      TileAwaitOp([gather.result]),
      TileReturnOp(),
    ]
  )

  context = NestContextOp(
    "gather_context",
    context_resources(
      1 + int(include_evu_context),
      l2_mode=1,
      allowed=(1, 2),
      l2_cache=cache,
      requested_contexts_per_tile=1 + int(include_evu_context),
    ),
    placement=1,
    arg_types=[NestGlobalMemref.of([4096], "i8"), NestGlobalMemref.of([len(index_rows)], "i32")],
    arg_names=["table", "indices"],
  )
  table_arg, indices_arg = context.body.block.args
  table_view = NestSubviewOp(table_arg, [0], [4096], [1], NestGlobalView.of([4096], "i8"))
  indices_global = NestSubviewOp(indices_arg, [0], [len(index_rows)], [1], NestGlobalView.of([len(index_rows)], "i32"))
  indices_alloc = NestAllocOp("indices_l2", "in", [len(index_rows)], "i32")
  indices_prefetch = NestPrefetchOp(indices_global.result, indices_alloc.result, "indices_prefetched")
  tasks = NestTaskRangeOp(0, 1)
  dispatch = NestDispatchOp(
    "gather_tile",
    tasks.result,
    [table_view.result],
    [indices_alloc.result],
    [],
    "grid_done",
    "input_released",
    "",
    l1_mode=1,
    bindings=[indices_alloc.result],
    signal_policy={"input_released": "all_tasks"},
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
      l1_mode=1,
      bindings=[],
      signal_policy={},
    )
    programs.append(evu_program)
    dispatches.append(evu_dispatch)
  context.body.block.add_ops(
    [
      table_view,
      indices_global,
      indices_alloc,
      indices_prefetch,
      tasks,
      *dispatches,
      NestAwaitOp([item.grid_done for item in dispatches]),
      NestReleaseOp(indices_alloc.result, (indices_prefetch.result, dispatch.input_released)),
      NestReturnOp(),
    ]
  )
  return ModuleOp([*programs, context])


def make_waiting_mfe_program(name: str = "ctx_wait_mfe") -> TileProgramDefOp:
  prog = TileProgramDefOp(
    name,
    tile_resources(L2_WAIT_BYTES),
    arg_types=[NestTask(), NestBuffer.of(L2_WAIT_DIMS, "bf16")],
    arg_names=["task", "l2_buf"],
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
  prog = TileProgramDefOp(name, tile_resources(), arg_types=[NestTask()], arg_names=["task"])
  evu = TileEvuOp(op_name="relu", evu_ops=16, tag="e_evu")
  prog.body.block.add_ops([evu, TileAwaitOp([evu.result]), TileReturnOp()])
  return prog


def make_boa_program(name: str) -> TileProgramDefOp:
  prog = TileProgramDefOp(name, tile_resources(), arg_types=[NestTask()], arg_names=["task"])
  boa = TileBoaOp(op_name="matmul", m=256, n=256, k=256, boa_ops=33554432, tag="e_boa")
  prog.body.block.add_ops([boa, TileAwaitOp([boa.result]), TileReturnOp()])
  return prog


def make_held_mfe_launch_module() -> ModuleOp:
  dims = [1, 64, 64]
  prog = TileProgramDefOp(
    "held_mfe_launch",
    tile_resources(6 * 8192),
    arg_types=[NestTask(), NestBuffer.of(dims, "bf16")],
    arg_names=["task", "l2_buf"],
  )
  task_arg, l2_arg = prog.body.block.args
  loads = []
  for i in range(6):
    view = TileSubviewOp(l2_arg, None, None, [0, 0, 0], dims, [1, 1, 1], NestL2View.of(dims, "bf16"))
    l1 = TileAllocOp(dims[1:], "bf16")
    load = TileLoadOp(view.result, l1.result, f"e_load{i}")
    prog.body.block.add_ops([view, l1, load])
    loads.append(load)
  prog.body.block.add_ops(
    [TileAwaitOp([load.result for load in loads]), TileSignalOp("input_released", task_arg), TileReturnOp()]
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
    l1_mode=0,
    bindings=[buffer.result],
    signal_policy={"input_released": "all_tasks"},
  )
  context = NestContextOp(
    "held_mfe_context",
    context_resources(1, 8192),
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
  """Dispatch programs to one Tile with the authored lease concurrency."""
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
        l1_mode=0,
        bindings=bindings,
        signal_policy={"input_released": "all_tasks"} if i == 0 else {},
        context_id=None if pins is None else pins[i],
      )
    )
  same_fixed_pin = (
    pins is not None and bool(pins) and pins[0] is not None and all(pin == pins[0] for pin in pins)
  )
  context = NestContextOp(
    "same_tile_roles",
    context_resources(
      role_count, L2_WAIT_BYTES, requested_contexts_per_tile=1 if same_fixed_pin else role_count
    ),
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
      l1_mode=0,
      bindings=[buffer.result],
      signal_policy={"input_released": "all_tasks"},
    )
    ctxs.append(
      NestContextOp(
        f"ctx{i}",
        context_resources(1, L2_WAIT_BYTES),
        [
          buffer,
          tasks,
          disp,
          NestReleaseOp(buffer.result, depends_on=[disp.input_released]),
          NestAwaitOp([disp.grid_done]),
          NestReturnOp(),
        ],
        placement=1,
        context_id=pin,
        arg_types=[NestGlobalMemref.of(L2_WAIT_DIMS, "bf16")],
        arg_names=["Y"],
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


# ---------------------------------------------------------------------------
# Cold / warm launch (residency)
# ---------------------------------------------------------------------------


class TestRuntimeColdWarm:
  def test_cold_launch_includes_program_load(self):
    """Cold launch's PMU records program_cold_load > 0."""
    s = make_sim("runtime")
    wl = PowWorkload()
    r = run_source(s, wl.module, POW_BINDINGS)
    assert r.completed
    cold = r.pmu.named_cycles.get("program_cold_load", 0)
    assert cold > 0, f"cold launch should record cold_load > 0, got {cold}"

  def test_warm_launch_no_program_reload(self):
    """Second launch of same program: 0 new cold-load cycles."""
    s = make_sim("runtime")
    wl = PowWorkload()
    _r1 = run_source(s, wl.module, POW_BINDINGS)
    c1 = s.group.program_table.cold_load_cycles
    r2 = run_source(s, wl.module, POW_BINDINGS)
    c2 = s.group.program_table.cold_load_cycles
    assert c2 == c1, f"warm should add 0 cold cycles, got delta {c2 - c1}"
    assert r2.completed

  def test_warm_faster_than_cold(self):
    """Warm launch completes in fewer cycles than cold."""
    s = make_sim("runtime")
    wl = PowWorkload()
    r1 = run_source(s, wl.module, POW_BINDINGS)
    r2 = run_source(s, wl.module, POW_BINDINGS)
    assert r2.cycles < r1.cycles, f"warm {r2.cycles} should be < cold {r1.cycles}"

  def test_program_epoch_invalidate_on_group_reset(self):
    """Group reset bumps epoch; next dispatch is cold again."""
    s = make_sim("runtime")
    wl = PowWorkload()
    run_source(s, wl.module, POW_BINDINGS)
    c1 = s.group.program_table.cold_load_cycles
    s.group.program_table.invalidate_group()
    _r2 = run_source(s, wl.module, POW_BINDINGS)
    c2 = s.group.program_table.cold_load_cycles
    assert c2 > c1, "reset should force cold re-install"

  def test_tile_reset_invalidates_residency(self):
    """Per-tile reset makes that tile cold again."""
    s = make_sim("runtime")
    wl = PowWorkload()
    run_source(s, wl.module, POW_BINDINGS)
    c1 = s.group.program_table.cold_load_cycles
    s.group.program_table.invalidate_tile(0)
    _r2 = run_source(s, wl.module, POW_BINDINGS)
    c2 = s.group.program_table.cold_load_cycles
    assert c2 > c1, "tile reset should force cold re-install on that tile"

  def test_cb06_program_identity_and_artifact_are_stable_across_warm_runs(self):
    """One immutable artifact may launch cold then warm without template mutation."""
    s = make_sim("runtime")
    wl = PowWorkload(hw=s.hw)
    before = print_workload_ir(wl.module)
    artifact = compile_source(wl.module, s.hw, s.sim, workload_info=wl.info)
    prog1 = artifact.entry.role_bindings[0].tile_program
    identity = (prog1.program_id, prog1.program_hash, artifact.artifact_hash)
    loaded = load_program(artifact, s.hw, s.sim, actual_bindings=POW_BINDINGS)

    r1 = s.run(loaded)
    assert r1.completed, r1.reason
    assert print_workload_ir(wl.module) == before
    assert artifact.artifact_hash == identity[2]

    r2 = s.run(loaded)
    assert r2.completed, r2.reason
    assert print_workload_ir(wl.module) == before
    assert artifact.artifact_hash == identity[2]
    assert r2.cycles < r1.cycles

    compiled_again = compile_source(wl.module, s.hw, s.sim, workload_info=wl.info)
    prog2 = compiled_again.entry.role_bindings[0].tile_program
    assert (prog2.program_id, prog2.program_hash) == identity[:2]

    mutated_text = before.replace("exponent = 2 pow_ops = 65536", "exponent = 3 pow_ops = 65536", 1)
    mutated_module = parse_workload_ir(mutated_text, source_name="<mutated>")
    mutated = compile_source(mutated_module, s.hw, s.sim)
    prog3 = mutated.entry.role_bindings[0].tile_program
    assert prog3.program_id == prog1.program_id
    assert prog3.program_hash != prog1.program_hash

    cold_before = s.group.program_table.cold_load_cycles
    r3 = s.run(load_program(mutated, s.hw, s.sim, actual_bindings=POW_BINDINGS))
    assert r3.completed, r3.reason
    assert s.group.program_table.cold_load_cycles > cold_before

  def test_binding_change_rebinds_same_name_after_quiescent_reset(self):
    """A quiescent reset preserves the Host binding; changing that binding
    at the next launch explicitly rebinds it with a fresh HBM generation."""
    from pipeline_validator.memory import MemoryInvariantError

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = Simulator(hw, SimConfig(fidelity="runtime", max_cycles=200000))
    first = {"Y": GlobalBinding("Y", 0x100000, 524288, "rw")}
    r1 = run_source(sim, PowWorkload().module, first)
    assert r1.completed, r1.reason
    old = sim.group._global_handles["Y"]

    sim.group.reset()
    assert sim.group._global_handles == {"Y": old}
    assert sim.group.hbm.snapshot()["external_bindings"] == 1
    sim.group.hbm.assert_live(old)

    second = {"Y": GlobalBinding("Y", 0x400000, 524288, "rw")}
    r2 = run_source(sim, PowWorkload().module, second)
    assert r2.completed, r2.reason
    new = sim.group._global_handles["Y"]
    assert new.base_address == 0x400000
    assert new.generation > old.generation
    assert new != old
    with pytest.raises(MemoryInvariantError):
      sim.group.hbm.assert_live(old)


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
    result = run_source(sim, PowWorkload().module, POW_BINDINGS)
    assert result.completed, result.reason
    assert result.tracer is not None
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    store_events = [
      event
      for event in events
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
    result = run_source(sim, PowWorkload().module, POW_BINDINGS)
    assert result.completed, result.reason
    mem = sim.group.snapshot()["memory"]
    assert mem["fidelity"] == "full_memory"
    assert mem["hbm"]["external_bindings"] == 1  # external binding kept
    assert mem["l2"]["peak_arena_reserved_bytes"] > 0
    assert mem["l2"]["live_arenas"] == 0
    for tile_id, l1 in mem["l1"].items():
      assert l1["allocator"]["peak_arena_reserved_bytes"] > 0, tile_id
      assert l1["allocator"]["live_arenas"] == 0, tile_id
    assert mem["transfers"]["inflight"] == 0
    for name, vc in mem["noc"].items():
      if name != "summary":
        assert vc["credit"] == hw.noc_vc_depth  # credits restored


class TestIndependentTileAdmission:
  @staticmethod
  def _launched_group():
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = SimConfig(fidelity="runtime", context_count=1, max_cycles=200000)
    workload = PowWorkload(num_group_chunks=1, hw=hw, context_count=1)
    artifact = compile_source(workload.module, hw, sim, workload_info=workload.info)
    group = TileGroup(hw, fidelity="runtime", context_count=1)
    begin_group_program(group, artifact, POW_BINDINGS)
    assert isinstance(artifact.entry, ExecTileGroupTask)
    group.load_task(artifact.entry, input_bindings=POW_BINDINGS)
    return group, artifact.entry

  @staticmethod
  def _step_until(group, predicate, limit=2000):
    for cycle in range(limit):
      group.step(cycle)
      if predicate():
        return cycle
    raise AssertionError("condition did not become observable")

  def test_l2_pins_are_per_committed_task_until_independent_retirement(self):
    group, _task = self._launched_group()
    self._step_until(
      group,
      lambda: (
        bool(group._grid_routes)
        and len(next(iter(group._grid_routes.values())).admissions) == group.cfg.num_tiles
      ),
    )
    route = next(iter(group._grid_routes.values()))
    pins = group._grid_l2_pins[route.grid]
    assert set(pins) == {task.task_id for task in route.expected.values()}
    for task_pins in pins.values():
      assert len(task_pins) == 1
      pin = next(iter(task_pins.values()))
      assert pin.reads and pin.writes
      record = group.l2_sram._views[pin.handle.allocation_id]
      assert pin.consumer_id in record.pins

  def test_t03_t13_blocked_tile_does_not_rollback_peer_or_complete_grid_and_refills_independently(self):
    group, _task = self._launched_group()
    tile = group.tiles[1]
    profile = tile.l1_allocator.profile
    blocker_layout = layout_buffers(
      (), profile, profile.user_spm_bytes, slot_capacity=group.cfg.frame_slot_capacity
    )
    blocker_owner = TaskIdentity(GridInstanceId("blocker", 0, 0, 0), 0)
    plan = tile.l1_allocator.plan_arena(blocker_owner, blocker_layout)
    assert not isinstance(plan, AdmissionFailure)
    blocker = tile.l1_allocator.commit_arena(plan, 0)

    self._step_until(
      group,
      lambda: (
        bool(group._grid_routes)
        and len(next(iter(group._grid_routes.values())).admissions) >= group.cfg.num_tiles - 1
      ),
    )
    route = next(iter(group._grid_routes.values()))
    assert 1 not in route.admissions
    assert set(route.admissions) == {0, 2, 3}
    assert route.wait_reasons[1] in ("WAIT_CAPACITY", "WAIT_FRAGMENTATION")
    assert route.event_id not in route.sequencer._events_done
    assert tile.l1_allocator.snapshot()["live_arenas"] == 1

    assert tile.l1_allocator.retire_arena(blocker, 100)
    self._step_until(group, lambda: 1 in route.admissions)
    assert set(route.admissions) == {0, 1, 2, 3}

  def test_t08_late_tile_bind_failure_rolls_back_only_its_ticket_then_drains(self, monkeypatch):
    group, _task = self._launched_group()
    monkeypatch.setattr(group.tiles[1], "load_program", lambda *args, **kwargs: None)
    fault_cycle = self._step_until(group, lambda: any(seq.faulted for seq in group._active_sequencers))
    route = next(iter(group._grid_routes.values()))
    assert 0 in route.admissions
    assert 1 not in route.admissions
    assert group.tiles[1].l1_allocator.snapshot()["live_arenas"] == 0

    self._step_until(group, lambda: group.reset_domain.is_done, limit=fault_cycle + 5000)
    assert not group._grid_routes
    assert not group._task_leases
    assert group.l2_sram.snapshot()["live_arenas"] == 0
    assert all(tile.l1_allocator.snapshot()["live_arenas"] == 0 for tile in group.tiles)


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
    run_source(s, wl.module, POW_BINDINGS)
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

  def test_fault_reset_waits_for_cancel_confirmation_before_returning_resources(self):
    """Accepted transfer work remains owned while cancellation is requested;
    fault drain completes only after the accepted leg confirms isolation."""
    from pipeline_validator.memory.transfer import TransferStatus

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=1000)
    sim_config = SimConfig(fidelity="full_memory", max_cycles=100000)
    s = Simulator(hw, sim_config, enable_tracer=True)
    wl = PowWorkload(hw=hw)
    artifact = compile_source(wl.module, hw, sim_config, workload_info=wl.info)
    begin_group_program(s.group, artifact, POW_BINDINGS)
    assert isinstance(artifact.entry, ExecTileGroupTask)
    s.group.load_task(artifact.entry, input_bindings=POW_BINDINGS)

    transfer_cycle = None
    transaction = None
    for cycle in range(100):
      s.group.step(cycle)
      running = [
        item
        for item in s.group.transfer_manager._transactions.values()
        if item.status is TransferStatus.RUNNING
      ]
      if (
        running
        and s.group._group_transfer_trace_slots
        and any(s.group._group_transfer_trace_busy_slots.values())
        and s.group.transfer_manager._hbm_read._outstanding > 0
      ):
        transfer_cycle = cycle
        transaction = running[0]
        break
    assert transfer_cycle is not None and transaction is not None
    assert s.group.l2_sram.snapshot()["live_arenas"] > 0

    fault_cycle = transfer_cycle + 1
    s.group.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=fault_cycle)
    s.group.step(fault_cycle)
    assert s.group.reset_domain.state == ResetState.STOP_QUEUE
    frozen_submission_pc = s.group.sequencer.submission_pc

    cancel_requested_cycle = None
    for cycle in range(fault_cycle + 1, fault_cycle + 500):
      s.group.step(cycle)
      if transaction.status is TransferStatus.CANCEL_REQUESTED:
        cancel_requested_cycle = cycle
        break
    assert cancel_requested_cycle is not None
    assert not s.group.reset_domain.is_done
    assert s.group.transfer_manager.inflight_count > 0
    assert s.group.transfer_manager._hbm_read._outstanding > 0
    assert s.group.l2_sram.snapshot()["live_arenas"] > 0

    for cycle in range(cancel_requested_cycle + 1, fault_cycle + 5000):
      s.group.step(cycle)
      if s.group.reset_domain.is_done:
        break
    assert s.group.reset_domain.is_done
    assert transaction.status is TransferStatus.CANCELLED
    assert s.group.sequencer.submission_pc == frozen_submission_pc
    assert s.group.transfer_manager.inflight_count == 0
    assert s.group._group_transfer_trace_slots == {}
    assert not any(s.group._group_transfer_trace_busy_slots.values())
    assert s.group.transfer_manager._hbm_read._outstanding == 0
    assert s.group.transfer_manager._hbm_write._outstanding == 0
    for stage in (
      s.group.transfer_manager._global_dma,
      s.group.transfer_manager._l2_read,
      s.group.transfer_manager._l2_write,
    ):
      assert all(b == 0 for b in stage._busy_until), stage.name
      assert all(h is None for h in stage._holders), stage.name
    for vc in s.group.noc.vcs.values():
      assert vc.occupancy == 0
      assert vc.credit_available == hw.noc_vc_depth
    assert s.group.l2_sram.snapshot()["live_arenas"] == 0
    assert all(tile.l1_allocator.snapshot()["live_arenas"] == 0 for tile in s.group.tiles)
    s.group.reset()


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

  @staticmethod
  def _l1_view(logical_bytes=768):
    from pipeline_validator.execution_ir import ExecL1Buffer
    from pipeline_validator.memory import AdmissionFailure

    hw = HardwareConfig()
    profile = build_registry(hw).profile("l1", 0)
    pool = ArenaPool(profile, pool_id=0, tile_id=0)
    spec = ExecL1Buffer("l1:0", (logical_bytes,), "i8", 1, 64, logical_bytes)
    reserved = authored_arena_bytes(hw, "l1", [(logical_bytes, 64)])
    layout = layout_buffers((spec,), profile, reserved)
    identity = TaskIdentity(GridInstanceId("ctx", 0, 1, 0), 0)
    plan = pool.plan_arena(identity, layout)
    assert not isinstance(plan, AdmissionFailure)
    arena = pool.commit_arena(plan, 0)
    pool.bind_task_metadata(arena, "ev", 0)
    view = pool.bind_view(arena, "l1:0", 0)
    return hw, pool, arena, identity, view

  def test_l1_and_l2_views_keep_compiled_cross_bank_segments(self):
    from pipeline_validator.execution_ir import ExecL2Buffer
    from pipeline_validator.memory import AdmissionFailure
    from pipeline_validator.tile import ComputeTile, _TileContextMemory

    hw, l1_pool, l1_arena, identity, l1_handle = self._l1_view()
    tile = ComputeTile(0, hw)
    tile.l1_allocator = l1_pool

    l2_profile = build_registry(hw).profile("l2", 0)
    l2_pool = ArenaPool(l2_profile)
    l2_spec = ExecL2Buffer("l2_buf", (768,), "i8", "in", 1, 64, 768)
    l2_reserved = authored_arena_bytes(hw, "l2", [(768, 64)])
    l2_layout = layout_buffers((l2_spec,), l2_profile, l2_reserved)
    l2_plan = l2_pool.plan_arena(RootInvocation("ctx", 1), l2_layout)
    assert not isinstance(l2_plan, AdmissionFailure)
    l2_arena = l2_pool.commit_arena(l2_plan, 0)
    l2_handle = l2_pool.bind_view(l2_arena, "l2_buf", 0)
    memory = _TileContextMemory(
      task_identity=identity,
      l2_formal_handles={1: l2_handle},
      l1_handles={"l1:0": l1_handle},
      l2_resolver=l2_pool,
      arena=l1_arena,
      binding_id="fixture",
    )

    l1_view = TileUCE._resolve_tile_view(self._view("l1", "l1:0", 768, 768), memory, tile)
    l2_view = TileUCE._resolve_tile_view(self._view("l2", "formal:1", 768, 768), memory, tile)
    assert l1_view is not None and l2_view is not None
    assert sum(segment.size_bytes for segment in l1_view.segments) == 768
    assert sum(segment.size_bytes for segment in l2_view.segments) == 768
    assert len({segment.bank_id for segment in l1_view.segments}) > 1
    assert len({segment.bank_id for segment in l2_view.segments}) > 1

  def test_local_view_oob_and_use_after_invalidate_raise(self):
    from pipeline_validator.memory import MemoryInvariantError
    from pipeline_validator.tile import ComputeTile, _TileContextMemory

    hw, pool, arena, identity, handle = self._l1_view(512)
    tile = ComputeTile(0, hw)
    tile.l1_allocator = pool
    memory = _TileContextMemory(
      task_identity=identity, l1_handles={"l1:0": handle}, arena=arena, binding_id="fixture"
    )
    with pytest.raises(MemoryInvariantError, match="memory view out of bounds"):
      TileUCE._resolve_tile_view(self._view("l1", "l1:0", 512, 200, offset=400), memory, tile)
    assert pool.invalidate_view(handle, handle.owner, 1)
    with pytest.raises(MemoryInvariantError, match="use-after-release"):
      TileUCE._resolve_tile_view(self._view("l1", "l1:0", 512, 64), memory, tile)


# ---------------------------------------------------------------------------
# Slot frame
# ---------------------------------------------------------------------------


class TestSlotFrame:
  @staticmethod
  def _arena(buffers, reserved_bytes, *, lifetimes=None):
    from pipeline_validator.memory import AdmissionFailure

    hw = HardwareConfig()
    profile = build_registry(hw).profile("l1", 0)
    pool = ArenaPool(profile, pool_id=0, tile_id=0)
    layout = layout_buffers(
      buffers, profile, reserved_bytes, lifetimes=lifetimes, slot_capacity=hw.frame_slot_capacity
    )
    task = TaskIdentity(GridInstanceId("frame", 0, 0, 0), 0)
    plan = pool.plan_arena(task, layout)
    assert not isinstance(plan, AdmissionFailure)
    arena = pool.commit_arena(plan, 0)
    pool.bind_task_metadata(arena, "frame-role", 0)
    return hw, pool, arena, layout

  def test_frame_prepare_bind_and_runtime_slot_binding_succeeds(self):
    from pipeline_validator.execution_ir import ExecL1Buffer
    from pipeline_validator.memory import SlotFrame

    spec = ExecL1Buffer("l1:0", (16, 16), "bf16", 2, 256, 512)
    reserved = authored_arena_bytes(HardwareConfig(), "l1", [(512, 256)])
    hw, pool, arena, layout = self._arena((spec,), reserved)
    frame = SlotFrame(l1_bytes=hw.tile_l1_bytes, slot_count=hw.frame_slot_capacity)
    assert frame.prepare(arena, layout)
    ok, cycles = frame.bind(cycle=0, bind_cycles=hw.frame_bind_cycles)
    assert ok and cycles == hw.frame_bind_cycles
    view = pool.bind_view(arena, "l1:0", 1)
    slot = frame.bind_view("l1:0", view)
    assert slot == layout.buffer_layouts[0].slot_id
    frame.assert_slot_binding(slot, view)

  def test_frame_prepare_rejects_target_stride_mismatch(self):
    from pipeline_validator.execution_ir import ExecL1Buffer
    from pipeline_validator.memory import SlotFrame

    spec = ExecL1Buffer("l1:0", (16, 16), "bf16", 2, 256, 512)
    reserved = authored_arena_bytes(HardwareConfig(), "l1", [(512, 256)])
    _hw, _pool, arena, layout = self._arena((spec,), reserved)
    frame = SlotFrame(l1_bytes=512)
    assert not frame.prepare(arena, layout)
    assert frame.pmu_permission_fault_count == 1

  def test_local_free_reuses_compiled_slot_without_returning_arena_capacity(self):
    from pipeline_validator.execution_ir import ExecL1Buffer
    from pipeline_validator.memory import SlotFrame

    a = ExecL1Buffer("a", (64,), "i8", 1, 64, 64)
    b = ExecL1Buffer("b", (64,), "i8", 1, 64, 64)
    reserved = authored_arena_bytes(HardwareConfig(), "l1", [(64, 64)])
    hw, pool, arena, layout = self._arena((a, b), reserved, lifetimes={"a": (0, 1), "b": (1, 2)})
    assert layout.buffer_layouts[0].arena_offset == layout.buffer_layouts[1].arena_offset
    assert layout.buffer_layouts[0].slot_id == layout.buffer_layouts[1].slot_id
    frame = SlotFrame(l1_bytes=hw.tile_l1_bytes, slot_count=hw.frame_slot_capacity)
    assert frame.prepare(arena, layout)
    assert frame.bind(0, hw.frame_bind_cycles)[0]

    view_a = pool.bind_view(arena, "a", 1)
    slot = frame.bind_view("a", view_a)
    before = pool.snapshot()
    frame.release_slot(slot, view_a)
    assert pool.invalidate_view(view_a, view_a.owner, 2)
    after_free = pool.snapshot()
    assert after_free["arena_reserved_bytes"] == before["arena_reserved_bytes"]
    assert after_free["free_bytes"] == before["free_bytes"]

    view_b = pool.bind_view(arena, "b", 3)
    assert frame.bind_view("b", view_b) == slot
    frame.release_slot(slot, view_b)
    assert pool.invalidate_view(view_b, view_b.owner, 4)
    assert pool.retire_arena(arena, 5)
    assert pool.snapshot()["live_arenas"] == 0


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
        wl = wl_cls(hw=hw, context_count=sim.context_count)
        s = Simulator(hw, sim)
        signal.alarm(60)
        r = run_source(s, wl.module, POW_BINDINGS, workload_info=wl.info)
        signal.alarm(0)
        assert r.completed, f"{wl.name} failed in {fidelity}: {r.reason}"

  def test_runtime_context_count_two_runs_two_same_tile_roles(self):
    sim = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", context_count=2, max_cycles=10000), enable_tracer=True
    )
    result = run_source(sim, make_same_tile_roles_task(2))
    assert result.completed, result.reason
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    leases = [event for event in events if event.get("name") == "task_lease_acquire"]
    assert len(leases) == 2
    assert {(event["args"]["tile_id"], event["args"]["ctx_id"]) for event in leases} == {(0, 0), (0, 1)}
    assert len({event["args"]["grid_event"] for event in leases}) == 2
    assert result.pmu.named_cycles.get("task_accept", 0) > 0

  def test_runtime_context_count_three_overlaps_three_roles(self):
    sim = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", context_count=3, max_cycles=10000), enable_tracer=True
    )
    result = run_source(sim, make_same_tile_roles_task(3))
    assert result.completed, result.reason
    assert result.tracer is not None
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    peak = max(e["args"]["active_context_count"] for e in events if e.get("name") == "active_context_count")
    assert peak == 3
    leases = [event for event in events if event.get("name") == "task_lease_acquire"]
    assert len(leases) == 3
    assert {event["args"]["ctx_id"] for event in leases} == {0, 1, 2}

  def test_held_engine_launch_issues_once_and_parks(self):
    context_count = 8
    programs = [make_boa_program(f"ctx_held_boa{i}") for i in range(context_count)]
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
        l1_mode=0,
        bindings=[],
        signal_policy={},
        context_id=i,
      )
      for i, program in enumerate(programs)
    ]
    context = NestContextOp(
      "held_boa_context",
      context_resources(logical_tasks=context_count, requested_contexts_per_tile=context_count),
      [tasks, *dispatches, NestAwaitOp([dispatch.grid_done for dispatch in dispatches]), NestReturnOp()],
      placement=1,
    )
    simulator = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", context_count=context_count, max_cycles=200000),
      enable_tracer=True,
    )
    result = run_source(simulator, ModuleOp([*programs, context]))

    assert result.completed, result.reason
    issues = assert_uce_instructions_issue_once(result)
    assert sum(event["args"]["op"] == ExecTileOp.LAUNCH_BOA.value for event in issues) == context_count

  def test_held_mfe_launch_issues_once(self):
    simulator = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", max_cycles=200000), enable_tracer=True
    )
    result = run_source(simulator, make_held_mfe_launch_module())

    assert result.completed, result.reason
    issues = assert_uce_instructions_issue_once(result)
    assert sum(event["args"]["op"] == ExecTileOp.LAUNCH_MFE.value for event in issues) == 6

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
    result = run_source(sim, PowWorkload().module, POW_BINDINGS)
    assert result.completed, result.reason
    mem = result.group_snapshot["memory"]
    assert mem["fidelity"] == "runtime"
    assert mem["hbm"] is not None
    assert mem["hbm"]["external_bindings"] == 1
    assert mem["l2"] is not None
    assert mem["l2"]["peak_arena_reserved_bytes"] > 0
    assert mem["l2"]["live_arenas"] == 0
    assert mem["noc"] is None  # contention fabric only in full_memory
    for tile in mem["l1"].values():
      assert tile["allocator"] is not None
      assert tile["allocator"]["peak_arena_reserved_bytes"] > 0
      assert tile["allocator"]["live_arenas"] == 0

  def test_model_second_run_resets_l2_generation(self):
    """runtime model fresh reset clears prior live extents and makes old L2
    handles stale before admitting the second run."""
    sim = Simulator(
      HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10),
      SimConfig(fidelity="runtime", device_context_count=2, max_cycles=200000),
    )
    module = make_two_context_model()
    first = run_source(sim, module, MODEL_BINDINGS)
    assert first.completed, first.reason
    assert sim.group.l2_sram.snapshot()["live_arenas"] == 0
    second = run_source(sim, module, MODEL_BINDINGS)
    assert second.completed, second.reason
    assert sim.group.l2_sram.snapshot()["live_arenas"] == 0

  def test_dispatch_pinned_same_context_serializes(self):
    """A fixed UCE pin uses R=1 and returns its lease before reuse."""
    sim = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", context_count=2, max_cycles=10000), enable_tracer=True
    )
    result = run_source(sim, make_same_tile_roles_task(2, pins=[0, 0]))
    assert result.completed, result.reason
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    acquired = sorted(
      (event for event in events if event.get("name") == "task_lease_acquire"),
      key=lambda event: event["ts"],
    )
    released = {
      event["args"]["grid_event"]: event["ts"]
      for event in events
      if event.get("name") == "task_lease_release"
    }
    assert len(acquired) == 2
    assert all((event["args"]["tile_id"], event["args"]["ctx_id"]) == (0, 0) for event in acquired)
    assert released[acquired[0]["args"]["grid_event"]] <= acquired[1]["ts"]

  def test_dispatch_pinned_context_binds_requested_index(self):
    """Pinned dispatch lands on the requested tile-local context index."""
    sim = Simulator(
      HardwareConfig(), SimConfig(fidelity="runtime", context_count=2, max_cycles=10000), enable_tracer=True
    )
    result = run_source(sim, make_same_tile_roles_task(2, pins=[1, 1]))
    assert result.completed, result.reason
    assert result.tracer is not None
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    leases = [event for event in events if event.get("name") == "task_lease_acquire"]
    assert len(leases) == 2
    assert [(event["args"]["tile_id"], event["args"]["ctx_id"]) for event in leases] == [(0, 1), (0, 1)]
    assert len({event["args"]["grid_event"] for event in leases}) == 2

  def test_dispatch_pinned_context_out_of_range_fails_at_load(self):
    """Out-of-range context pin fails fast at task load, not silent deadlock."""
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", context_count=2, max_cycles=10000))
    with pytest.raises(ValueError):
      run_source(sim, make_same_tile_roles_task(2, pins=[2, None]))

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
      l1_mode=0,
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
      l1_mode=0,
      bindings=[buffer.result],
      signal_policy={"input_released": "all_tasks"},
      context_id=1,
    )
    module = ModuleOp(
      [
        prog,
        NestContextOp(
          "same_prog_two_pins",
          context_resources(2, L2_WAIT_BYTES, requested_contexts_per_tile=2),
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
    hw = HardwareConfig()
    sim_config = SimConfig(fidelity="runtime", context_count=2, max_cycles=10000)
    task = compiled_entry(module, hw, sim_config)
    assert sorted(b.context_id for b in task.role_bindings.values()) == [0, 1]
    result = run_source(Simulator(hw, sim_config), module)
    assert result.completed, result.reason


# ---------------------------------------------------------------------------
# Model mode (nexus.program + device slot scheduling)
# ---------------------------------------------------------------------------


class TestModelMode:
  def test_two_contexts_run_concurrently_on_two_slots(self):
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=2, max_cycles=10000),
      enable_tracer=True,
    )
    result = run_source(sim, make_two_context_model(), MODEL_BINDINGS)
    assert result.completed, result.reason
    records, port_records = assert_model_launch_lifecycle(result)
    by_context = {record["context"]: record for record in records}
    slots = {
      context: port_records[record["request_id"]]["slot_index"] for context, record in by_context.items()
    }
    assert sorted(slots.values()) == [0, 1]
    assert (
      port_records[by_context["ctx1"]["request_id"]]["active_cycle"]
      < by_context["ctx0"]["completion_cycle"]
    )

  def test_max_outstanding_one_serializes_hardware_admission(self):
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=1, max_cycles=10000),
      enable_tracer=True,
    )
    result = run_source(sim, make_two_context_model(), MODEL_BINDINGS)
    assert result.completed, result.reason
    records, port_records = assert_model_launch_lifecycle(result)
    by_context = {record["context"]: record for record in records}
    first = by_context["ctx0"]
    second = by_context["ctx1"]
    # CPU pending accepts ctx1 while ctx0 is still running, but the admitted
    # hardware-request limit holds ctx1 until ctx0 completes.
    assert second["submit_cycle"] < first["completion_cycle"]
    assert second["admission_cycle"] > first["completion_cycle"]
    assert result.device_snapshot["counters"]["outstanding_peak"] == 1
    assert {
      port_records[first["request_id"]]["slot_index"],
      port_records[second["request_id"]]["slot_index"],
    } == {0}

  def test_group_affinity_selects_hardware_slot(self):
    sim = Simulator(
      HardwareConfig(),
      SimConfig(fidelity="runtime", device_context_count=2, max_cycles=10000),
      enable_tracer=True,
    )
    result = run_source(sim, make_two_context_model(pins=(1, 0)), MODEL_BINDINGS)
    assert result.completed, result.reason
    records, port_records = assert_model_launch_lifecycle(result)
    slots = {record["context"]: port_records[record["request_id"]]["slot_index"] for record in records}
    assert slots == {"ctx0": 1, "ctx1": 0}

  def test_group_affinity_out_of_range_fails_at_load(self):
    sim = Simulator(
      HardwareConfig(),
      SimConfig(
        fidelity="runtime",
        device_context_count=2,
        group=GroupSchedulerConfig(active_context_capacity=2),
        max_cycles=10000,
      ),
    )
    with pytest.raises(ValueError):
      run_source(sim, make_two_context_model(pins=(2, None)), MODEL_BINDINGS)

  def test_standalone_module_rejects_nonzero_group_affinity(self):
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
      l1_mode=0,
      bindings=[buffer.result],
      signal_policy={"input_released": "all_tasks"},
    )
    module = ModuleOp(
      [
        prog,
        NestContextOp(
          "legacy_pinned",
          context_resources(1, L2_WAIT_BYTES),
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
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", max_cycles=10000))
    with pytest.raises(ValueError):
      run_source(sim, module)

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
      l1_mode=0,
      bindings=[buffer.result],
      signal_policy={"input_released": "all_tasks"},
    )
    ctx = NestContextOp(
      "ctx0",
      context_resources(1, L2_WAIT_BYTES),
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
    result = run_source(sim, module)
    assert result.completed, result.reason
    launch_records, port_records = assert_model_launch_lifecycle(result)
    assert len(launch_records) == 2
    first, second = launch_records
    assert first["request_id"] != second["request_id"]
    assert [first["event"], second["event"]] == ["done_c0", "done_c0_1"]
    assert second["submit_cycle"] > first["completion_cycle"]
    assert second["admission_cycle"] >= second["submit_cycle"]
    assert [
      port_records[first["request_id"]]["slot_index"],
      port_records[second["request_id"]]["slot_index"],
    ] == [0, 0]
    leases = [
      event
      for event in json.loads(result.tracer.to_chrome_json())["traceEvents"]
      if event.get("name") == "task_lease_acquire"
    ]
    assert len(leases) == 2
    assert {(event["args"]["tile_id"], event["args"]["ctx_id"]) for event in leases} == {(0, 0)}
    assert len({event["args"]["launch_generation"] for event in leases}) == 2
    assert len({event["args"]["grid_event"] for event in leases}) == 2

  # -----------------------------------------------------------------------
  # Input binding contract tests (PR 1, §2.5 / §3 Step 5)
  # -----------------------------------------------------------------------

  def test_missing_binding_fails_without_entering_runtime(self):
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", max_cycles=10000))
    assert_run_rejected_without_group_mutation(sim, make_two_context_model(), {})

  def test_unused_binding_fails_without_entering_runtime(self):
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", max_cycles=10000))
    bindings = {**MODEL_BINDINGS, "ZZ": GlobalBinding("ZZ", 0x300000, 1024, "rw")}
    assert_run_rejected_without_group_mutation(sim, make_two_context_model(), bindings)

  def test_binding_too_small_fails(self):
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", max_cycles=10000))
    bindings = {
      "Y0": GlobalBinding("Y0", 0x100000, 64, "rw"),
      "Y1": GlobalBinding("Y1", 0x200000, L2_WAIT_BYTES, "rw"),
    }
    assert_run_rejected_without_group_mutation(sim, make_two_context_model(), bindings)

  def test_binding_overlap_fails(self):
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="runtime", max_cycles=10000))
    bindings = {
      "Y0": GlobalBinding("Y0", 0x100000, L2_WAIT_BYTES, "rw"),
      "Y1": GlobalBinding("Y1", 0x100000, L2_WAIT_BYTES, "rw"),
    }
    assert_run_rejected_without_group_mutation(sim, make_two_context_model(), bindings)

  def test_binding_exceeds_hbm_capacity_fails(self):
    hw = HardwareConfig()
    sim = Simulator(hw, SimConfig(fidelity="runtime", max_cycles=10000))
    cap = hw.hbm_capacity_bytes
    bindings = {
      "Y0": GlobalBinding("Y0", cap, L2_WAIT_BYTES, "rw"),
      "Y1": GlobalBinding("Y1", 0x100000, L2_WAIT_BYTES, "rw"),
    }
    assert_run_rejected_without_group_mutation(sim, make_two_context_model(), bindings)

  def test_readonly_binding_rejects_store(self):
    """A read-only binding used as a store destination is rejected."""
    from pipeline_validator.dialects.elenor import (
      NestDispatchOp,
      NestDMAStoreOp,
      NestPrefetchOp,
      NestReleaseOp,
    )

    hw = HardwareConfig()
    prog = make_pow_tile_program(hw=hw)
    ctx = NestContextOp(
      "pow_task",
      context_resources(4, 131072),
      arg_types=[NestGlobalMemref.of([4, 128, 128], "bf16")],
      arg_names=["Y"],
      placement=15,
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
      l1_mode=0,
      bindings=[buf.result],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=[pref.result],
    )
    store = NestDMAStoreOp(buf.result, src.result, "ev_out", depends_on=[disp.output_ready])
    release = NestReleaseOp(buf.result, depends_on=[disp.input_released, pref.result, store.result])
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
    sim = Simulator(hw, SimConfig(fidelity="runtime", max_cycles=10000))
    assert_run_rejected_without_group_mutation(
      sim, module, {"Y0": GlobalBinding("Y0", 0x100000, 131072, "r")}
    )


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
    sim_config = SimConfig(fidelity=fidelity, max_cycles=200000)
    sim = Simulator(hw, sim_config)
    group = sim.group
    workload = PowWorkload(num_group_chunks=1, hw=hw)
    artifact = prepare_group_source(group, workload.module, POW_BINDINGS, sim=sim_config)
    assert isinstance(artifact.entry, ExecTileGroupTask)
    group.load_task(artifact.entry, input_bindings=POW_BINDINGS)
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
    from pipeline_validator.execution_ir import TaskIdentity

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
    fault_drain_and_reset(group)

  def test_duplicate_signal_faults_without_advancing_phase(self):
    """A repeated live milestone faults without completing the phase."""
    from pipeline_validator.execution_ir import TaskIdentity

    group, seq, grid, state = self._dispatch_and_get_grid()
    sig = PhaseSignal(TaskIdentity(grid, 0), "input_released")
    group._on_phase_signal(sig, 0)
    group._on_phase_signal(sig, 1)
    assert group.pmu.events.get("tile_signal_duplicate", 0) == 1
    assert seq.faulted
    assert "input_released" not in state.completed_phases
    fault_drain_and_reset(group)

  def test_stale_launch_signal_ignored(self):
    """Signal for a retired launch only increments tile_signal_stale."""
    from pipeline_validator.execution_ir import GridInstanceId, TaskIdentity

    group, _seq, grid, _state = self._dispatch_and_get_grid()
    old_gen = grid.launch_generation
    # Retire the current launch before creating the fresh launch namespace.
    start = group._last_step_cycle + 1
    for cycle in range(start, start + 10000):
      group.step(cycle)
      if _seq.done and not group._active_sequencers:
        break
    assert _seq.done and not group._active_sequencers
    assert isinstance(group.loaded_program.entry, ExecTileGroupTask)
    reload_cycle = group._last_step_cycle + 1
    group.load_task(group.loaded_program.entry, input_bindings=POW_BINDINGS, cycle=reload_cycle)
    stale_grid = GridInstanceId(grid.context_name, grid.device_slot, old_gen, grid.dispatch_ordinal)
    group._on_phase_signal(PhaseSignal(TaskIdentity(stale_grid, 0), "input_released"), reload_cycle)
    assert group.pmu.events.get("tile_signal_stale", 0) == 1
    finish_group_and_reset(group)


class TestSignalGatedRelease:
  """Access-aware release gating and pin lifecycle."""

  def test_trace_tile_signal_count_and_args(self):
    """Tracer-enabled dual-context run: tile_signal count ==
    context_count * task_count * phase_count; every event has 8 args."""
    sim = Simulator(
      HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10),
      SimConfig(fidelity="full_memory", context_count=2, device_context_count=2, max_cycles=200000),
      enable_tracer=True,
    )
    result = run_source(
      sim, parse_workload_ir(open("examples/workloads/pow_dual_context.mlir").read()), MODEL_BINDINGS
    )
    assert result.completed, result.reason
    assert_model_launch_lifecycle(result)
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

  _ARENA_DIMS = (4194304,)
  _BUFFER_DIMS = (4, 64, 64)
  _TASK_VIEW_DIMS = (1, 64, 64)
  _TENSOR_ELEMENTS = 131072
  _TENSOR_BYTES = 32768
  _TASK_BYTES = 8192
  _BINDINGS: ClassVar[dict[str, GlobalBinding]] = {
    "arena": GlobalBinding("arena", 0x1000000, 8388608, "rw")
  }

  @classmethod
  def _task_view(cls, buffer, task):
    return TileSubviewOp(
      buffer, task, 0, [0, 0, 0], cls._TASK_VIEW_DIMS, [1, 1, 1], NestL2View.of(cls._TASK_VIEW_DIMS, "bf16")
    )

  @classmethod
  def _global_view(cls, arena, tensor_index: int, elements: int = 16384):
    return NestSubviewOp(
      arena, [tensor_index * cls._TENSOR_ELEMENTS], [elements], [1], NestGlobalView.of([elements], "bf16")
    )

  @classmethod
  def _make_access_programs(cls):
    program_a = TileProgramDefOp(
      "access_A",
      tile_resources(8192),
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
      tile_resources(8192),
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
      tile_resources(32768),
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
    gate_load_lhs = TileLoadOp(gate_view_d.result, gate_lhs.result, "d_gate_load_lhs")
    gate_load_rhs = TileLoadOp(gate_view_d.result, gate_rhs.result, "d_gate_load_rhs")
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
      boa = TileBoaOp("matmul", 64, 64, 64, 524288, f"d_boa_{index}", accumulate=index > 0)
      d_ops.extend([boa, TileAwaitOp([boa.result])])
    shared_load_d = TileLoadOp(shared_view_d.result, shared_work.result, "d_shared_load")
    d_ops.extend(
      [shared_load_d, TileAwaitOp([shared_load_d.result]), TileSignalOp("input_released", task_d)]
    )
    for index in range(100):
      evu = TileEvuOp("relu", 16448, f"d_evu_{index}")
      d_ops.extend([evu, TileAwaitOp([evu.result])])
    store_d = TileStoreOp(shared_work.result, output_view_d.result, "d_store")
    d_ops.extend(
      [store_d, TileAwaitOp([store_d.result]), TileSignalOp("output_ready", task_d), TileReturnOp()]
    )
    program_d.body.block.add_ops(d_ops)
    return program_a, program_b, program_d

  @classmethod
  def _make_early_store_model(cls) -> ModuleOp:
    program_a, program_b, program_d = cls._make_access_programs()
    context = NestContextOp(
      "ctx_access",
      context_resources(12, 5 * 32768, requested_contexts_per_tile=3),
      placement=15,
      arg_types=[NestGlobalMemref.of(cls._ARENA_DIMS, "bf16")],
      arg_names=["arena"],
    )
    arena = context.body.block.args[0]
    source_x = NestAllocOp("source_X", "in", cls._BUFFER_DIMS, "bf16", alignment=256)
    shared_a = NestAllocOp("shared_A", "inout", cls._BUFFER_DIMS, "bf16", alignment=256)
    gate = NestAllocOp("gate", "in", cls._BUFFER_DIMS, "bf16", alignment=256)
    output_b = NestAllocOp("B_out", "out", cls._BUFFER_DIMS, "bf16", alignment=256)
    output_d = NestAllocOp("D_out", "out", cls._BUFFER_DIMS, "bf16", alignment=256)
    source_view = cls._global_view(arena, 0)
    shared_view = cls._global_view(arena, 1)
    gate_view = cls._global_view(arena, 2)
    output_b_view = cls._global_view(arena, 3)
    output_d_view = cls._global_view(arena, 4)
    prefetch_source = NestPrefetchOp(source_view.result, source_x.result, "pref_source_X")
    prefetch_gate = NestPrefetchOp(gate_view.result, gate.result, "pref_gate")
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
      l1_mode=0,
      bindings=[source_x.result, shared_a.result],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
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
      l1_mode=0,
      bindings=[shared_a.result, output_b.result],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
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
      l1_mode=0,
      bindings=[gate.result, shared_a.result, output_d.result],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=[dispatch_a.output_ready, prefetch_gate.result],
      context_id=2,
    )
    store_shared = NestDMAStoreOp(
      shared_a.result, shared_view.result, "store_shared_A", depends_on=[dispatch_a.output_ready]
    )
    store_b = NestDMAStoreOp(
      output_b.result, output_b_view.result, "store_B", depends_on=[dispatch_b.output_ready]
    )
    store_d = NestDMAStoreOp(
      output_d.result, output_d_view.result, "store_D", depends_on=[dispatch_d.output_ready]
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
        NestReleaseOp(source_x.result, depends_on=[dispatch_a.input_released, prefetch_source.result]),
        store_b,
        NestReleaseOp(output_b.result, depends_on=[store_b.result]),
        NestReleaseOp(
          shared_a.result,
          depends_on=[dispatch_b.input_released, dispatch_d.input_released, store_shared.result],
        ),
        NestReleaseOp(gate.result, depends_on=[dispatch_d.input_released, prefetch_gate.result]),
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
      "run_access", [], arg_types=[NestGlobalMemref.of(cls._ARENA_DIMS, "bf16")], arg_names=["arena"]
    )
    root_arena = root.body.block.args[0]
    submit = NexusSubmitContextOp("ctx_access", "done_access", actuals=[root_arena])
    root.body.block.add_ops([submit, NexusAwaitOp([submit.result]), NexusReturnOp()])
    return ModuleOp([program_a, program_b, program_d, context, root])

  @staticmethod
  def _trace_events(tracer) -> list[dict]:
    return json.loads(tracer.to_chrome_json())["traceEvents"]

  @staticmethod
  def _cycle_of(sim: Simulator, event: dict) -> int:
    return round(event["ts"] * 1000.0 / sim.hw.cycle_ns())

  @classmethod
  def _tile_transactions(cls, events: list[dict], role_event_suffix: str, op: str) -> dict[str, dict]:
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
      (grid, task_id, slot, pin.consumer_id, pin.reads, pin.writes)
      for grid, task_pins in group._grid_l2_pins.items()
      for task_id, slot_pins in task_pins.items()
      for slot, pin in slot_pins.items()
      if pin.handle == handle
    }

  @staticmethod
  def _assert_runtime_zero_leak(group: TileGroup) -> None:
    memory = group.snapshot()["memory"]
    assert memory["l2"]["live_arenas"] == 0
    assert memory["l2"]["pending_release"] == 0
    assert memory["transfers"]["inflight"] == 0
    assert not group._grid_l2_pins
    for tile_id, l1 in memory["l1"].items():
      assert l1["allocator"]["live_arenas"] == 0, tile_id

  @pytest.mark.parametrize("fidelity", ["runtime", "full_memory"])
  def test_early_store_preserves_delayed_reader(self, fidelity):
    hw = HardwareConfig().with_overrides(num_dma_channels=2, hbm_fixed_latency_cycles=10)
    sim = Simulator(
      hw,
      SimConfig(
        fidelity=fidelity, context_count=4, device_context_count=1, memory_trace=True, max_cycles=2000000
      ),
      enable_tracer=True,
    )
    result = run_source(sim, self._make_early_store_model(), self._BINDINGS)
    assert result.completed, result.reason
    assert result.tracer is not None
    events = self._trace_events(result.tracer)
    shared_invalidation = next(
      event
      for event in events
      if event.get("name") == "buffer_view_invalidate" and event["args"].get("buffer_id") == "shared_A"
    )
    shared_base = shared_invalidation["args"]["base_address"]
    release_cycle = self._cycle_of(sim, shared_invalidation)
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
      for transaction_id, transaction in self._tile_transactions(events, "grid_D", "tile_load").items()
      if str(transaction_id).endswith(":d_shared_load") and shared_base <= transaction["source_address"] < shared_base + self._TENSOR_BYTES
    }
    assert len(reads_b) == len(reads_d) == 4
    for transaction in reads_d.values():
      assert shared_base <= transaction["source_address"] < shared_base + self._TENSOR_BYTES
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
    d_compute_end = max(self._cycle_of(sim, {"ts": event["ts"] + event["dur"]}) for event in d_evu_events)
    assert max(last_b_read, last_d_read, shared_store["completion_cycle"]) <= release_cycle < d_compute_end
    assert shared_store["transaction_id"]
    assert set(reads_b).isdisjoint(reads_d)
    launch_records, _ = assert_model_launch_lifecycle(result)
    context_record = next(record for record in launch_records if record["context"] == "ctx_access")
    assert context_record["completion_cycle"] >= release_cycle
    self._assert_runtime_zero_leak(sim.group)
    assert result.credit_invariant_ok
    if fidelity == "full_memory":
      for name, vc in result.group_snapshot["memory"]["noc"].items():
        if name != "summary":
          assert vc["credit"] == hw.noc_vc_depth
    result.tracer.assert_well_formed()

  def test_release_preflight_rejects_late_reader(self):
    from dataclasses import replace

    from pipeline_validator.memory.allocator import MemoryInvariantError

    sim = Simulator(
      HardwareConfig().with_overrides(num_dma_channels=2, hbm_fixed_latency_cycles=10),
      SimConfig(
        fidelity="full_memory",
        context_count=4,
        device_context_count=1,
        memory_trace=True,
        max_cycles=2000000,
      ),
      enable_tracer=True,
    )
    artifact = prepare_group_source(sim.group, self._make_early_store_model(), self._BINDINGS, sim=sim.sim)
    group = sim.group
    task = task_named(artifact.entry, "ctx_access")
    seq = group.try_admit_context_task(
      task,
      slot_index=0,
      context_name="ctx_access",
      input_bindings=self._BINDINGS,
      formal_bindings={"arena": "arena"},
      cycle=0,
    )
    assert seq is not None and seq.task is not None
    release_request = next(
      action.args[0]
      for action in seq.task.actions
      if action.op == ExecGroupActionOp.RELEASE_L2 and action.args[0].buffer_slot == "shared_A"
    )
    late_reader = max(release_request.reader_dispatch_ordinals)
    late_event = next(event for event in release_request.dependency_events if event.endswith("read_D"))
    bad_request = replace(
      release_request,
      reader_dispatch_ordinals=tuple(
        ordinal for ordinal in release_request.reader_dispatch_ordinals if ordinal != late_reader
      ),
      dependency_events=tuple(event for event in release_request.dependency_events if event != late_event),
    )
    for cycle in range(1000000):
      group.step(cycle)
      if set(bad_request.dependency_events) <= seq._events_done and late_event not in seq._events_done:
        break
    assert late_event not in seq._events_done
    handle = group._l2_handles[(seq.context_launch_generation, "shared_A")]
    pins_before = self._pin_fingerprint(group, handle)
    assert pins_before
    assert any(item[-2] and not item[-1] for item in pins_before)
    allocator_pins_before = set(group.l2_sram._views[handle.allocation_id].pins)
    snapshot_before = group.l2_sram.snapshot()
    with pytest.raises(MemoryInvariantError):
      group.release_l2(bad_request, sequencer=seq, cycle=cycle + 1)
    assert not group.l2_sram.is_released(handle)
    assert self._pin_fingerprint(group, handle) == pins_before
    assert group.l2_sram._views[handle.allocation_id].pins == allocator_pins_before
    assert group.l2_sram.snapshot()["pending_release"] == snapshot_before["pending_release"] == 0
    fault_drain_and_reset(group)
    self._assert_runtime_zero_leak(group)
    assert group.credit_invariants_hold()

  @classmethod
  def _make_alias_module(cls, reverse_phases: bool) -> ModuleOp:
    dims = [1, 64, 64]
    task_view_dims = [1, 64, 64]
    program = TileProgramDefOp(
      "alias_reverse" if reverse_phases else "alias_forward",
      tile_resources(16384),
      arg_types=[NestTask(), NestBuffer.of(dims, "bf16"), NestBuffer.of(dims, "bf16")],
      arg_names=["task", "read_formal", "write_formal"],
    )
    task, read_formal, write_formal = program.body.block.args
    read_view = TileSubviewOp(
      read_formal, task, 0, [0, 0, 0], task_view_dims, [1, 1, 1], NestL2View.of(task_view_dims, "bf16")
    )
    write_view = TileSubviewOp(
      write_formal, task, 0, [0, 0, 0], task_view_dims, [1, 1, 1], NestL2View.of(task_view_dims, "bf16")
    )
    work = TileAllocOp([64, 64], "bf16", alignment=256)
    scratch = TileAllocOp([64, 64], "bf16", alignment=256)
    initial_load = TileLoadOp(read_view.result, work.result, "alias_load_0")
    ops = [read_view, write_view, work, scratch, initial_load, TileAwaitOp([initial_load.result])]
    if not reverse_phases:
      ops.append(TileSignalOp("input_released", task))
    if reverse_phases:
      tile_store = TileStoreOp(work.result, write_view.result, "alias_tile_store")
      ops.extend([tile_store, TileAwaitOp([tile_store.result]), TileSignalOp("output_ready", task)])
    for index in range(100):
      evu = TileEvuOp("relu", 16448, f"alias_evu_{index}")
      ops.extend([evu, TileAwaitOp([evu.result])])
    if reverse_phases:
      late_load = TileLoadOp(read_view.result, scratch.result, "alias_load_1")
      ops.extend([late_load, TileAwaitOp([late_load.result]), TileSignalOp("input_released", task)])
    else:
      tile_store = TileStoreOp(work.result, write_view.result, "alias_tile_store")
      ops.extend([tile_store, TileAwaitOp([tile_store.result]), TileSignalOp("output_ready", task)])
    ops.append(TileReturnOp())
    program.body.block.add_ops(ops)

    context = NestContextOp(
      "ctx_alias",
      context_resources(1, 8192),
      placement=1,
      arg_types=[NestGlobalMemref.of(cls._ARENA_DIMS, "bf16")],
      arg_names=["arena"],
    )
    arena = context.body.block.args[0]
    buffer = NestAllocOp("alias", "inout", dims, "bf16", alignment=256)
    global_view = cls._global_view(arena, 0, elements=4096)
    prefetch = NestPrefetchOp(global_view.result, buffer.result, "alias_prefetch")
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
      l1_mode=0,
      bindings=[buffer.result, buffer.result],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=[prefetch.result],
      context_id=0,
    )
    store = NestDMAStoreOp(
      buffer.result, global_view.result, "alias_global_store", depends_on=[dispatch.output_ready]
    )
    context.body.block.add_ops(
      [
        buffer,
        global_view,
        prefetch,
        tasks,
        dispatch,
        store,
        NestReleaseOp(buffer.result, depends_on=[dispatch.input_released, prefetch.result, store.result]),
        NestAwaitOp([dispatch.grid_done, store.result]),
        NestReturnOp(),
      ]
    )
    return ModuleOp([program, context])

  @pytest.mark.parametrize("fidelity", ["runtime", "full_memory"])
  @pytest.mark.parametrize("reverse_phases", [False, True])
  def test_readwrite_alias_lifetime(self, fidelity, reverse_phases):
    hw = HardwareConfig().with_overrides(num_dma_channels=2, hbm_fixed_latency_cycles=10)
    sim = Simulator(
      hw,
      SimConfig(
        fidelity=fidelity, context_count=4, device_context_count=1, memory_trace=True, max_cycles=2000000
      ),
      enable_tracer=True,
    )
    artifact = prepare_group_source(
      sim.group, self._make_alias_module(reverse_phases), self._BINDINGS, sim=sim.sim
    )
    assert isinstance(artifact.entry, ExecTileGroupTask)
    task = artifact.entry
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
    handle = group._l2_handles[(seq.context_launch_generation, "alias")]
    record = group.l2_sram._views[handle.allocation_id]
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
      if event.get("name") == "buffer_view_invalidate"
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
    assert release_cycle >= max(store["completion_cycle"], max(item["done"] for item in loads.values()))
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
      tile_resources(8192),
      arg_types=[NestTask(), NestBuffer.of(dims, "bf16"), NestBuffer.of(dims, "bf16")],
      arg_names=["task", "source", "output"],
    )
    task, source, output = program.body.block.args
    source_view = TileSubviewOp(source, task, 0, [0, 0, 0], dims, [1, 1, 1], NestL2View.of(dims, "bf16"))
    output_view = TileSubviewOp(output, task, 0, [0, 0, 0], dims, [1, 1, 1], NestL2View.of(dims, "bf16"))
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
      context_resources(1, 16384),
      placement=1,
      arg_types=[NestGlobalMemref.of(cls._ARENA_DIMS, "bf16")],
      arg_names=["arena"],
    )
    arena = context.body.block.args[0]
    source_buffer = NestAllocOp("store_source", "in", dims, "bf16", alignment=256)
    output_buffer = NestAllocOp("store_output", "out", dims, "bf16", alignment=256)
    source_global = cls._global_view(arena, 0, elements=4096)
    output_global_1 = cls._global_view(arena, 1, elements=4096)
    output_global_2 = cls._global_view(arena, 2, elements=4096)
    prefetch = NestPrefetchOp(source_global.result, source_buffer.result, "store_prefetch")
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
      l1_mode=0,
      bindings=[source_buffer.result, output_buffer.result],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=[prefetch.result],
      context_id=0,
    )
    store_1 = NestDMAStoreOp(
      output_buffer.result, output_global_1.result, "output_store_1", depends_on=[dispatch.output_ready]
    )
    store_2 = NestDMAStoreOp(
      output_buffer.result, output_global_2.result, "output_store_2", depends_on=[dispatch.output_ready]
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
        NestReleaseOp(source_buffer.result, depends_on=[dispatch.input_released, prefetch.result]),
        NestReleaseOp(output_buffer.result, depends_on=[store_1.result, store_2.result]),
        NestAwaitOp([dispatch.grid_done, store_1.result, store_2.result]),
        NestReturnOp(),
      ]
    )
    return ModuleOp([program, context])

  @classmethod
  def _make_inflight_prefetch_module(cls) -> ModuleOp:
    dims = [1, 64, 64]
    program = TileProgramDefOp(
      "inflight_reader",
      tile_resources(8192),
      arg_types=[NestTask(), NestBuffer.of(dims, "bf16")],
      arg_names=["task", "input"],
    )
    task, input_buffer = program.body.block.args
    input_view = TileSubviewOp(
      input_buffer, task, 0, [0, 0, 0], dims, [1, 1, 1], NestL2View.of(dims, "bf16")
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
      context_resources(1, 8192),
      placement=1,
      arg_types=[NestGlobalMemref.of(cls._ARENA_DIMS, "bf16")],
      arg_names=["arena"],
    )
    arena = context.body.block.args[0]
    buffer = NestAllocOp("prefetch_input", "in", dims, "bf16", alignment=256)
    global_1 = cls._global_view(arena, 0, elements=4096)
    global_2 = cls._global_view(arena, 1, elements=4096)
    prefetch_1 = NestPrefetchOp(global_1.result, buffer.result, "input_prefetch_1")
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
      l1_mode=0,
      bindings=[buffer.result],
      signal_policy={"input_released": "all_tasks"},
      depends_on=[prefetch_1.result],
      context_id=0,
    )
    prefetch_2 = NestPrefetchOp(global_2.result, buffer.result, "input_prefetch_2")
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
          buffer.result, depends_on=[dispatch.input_released, prefetch_1.result, prefetch_2.result]
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

    hw = HardwareConfig().with_overrides(num_dma_channels=2, hbm_fixed_latency_cycles=10)
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
    artifact = prepare_group_source(
      sim.group, self._make_inflight_store_module(), self._BINDINGS, sim=sim.sim
    )
    assert isinstance(artifact.entry, ExecTileGroupTask)
    task = artifact.entry
    group = sim.group
    group.load_task(task, input_bindings=self._BINDINGS)
    seq = group.sequencer
    request = next(
      action.args[0]
      for action in seq.task.actions
      if action.op == ExecGroupActionOp.RELEASE_L2 and action.args[0].buffer_slot == "store_output"
    )
    first_store = next(event for event in request.dependency_events if event.endswith("output_store_1"))
    second_store = next(event for event in request.dependency_events if event.endswith("output_store_2"))
    assert {first_store, second_store} <= set(request.dependency_events)
    bad_request = replace(
      request,
      dependency_events=tuple(event for event in request.dependency_events if event != second_store),
    )
    handle = None
    second_transaction = None
    for cycle in range(200000):
      group.step(cycle)
      handle = group._l2_handles.get((seq.context_launch_generation, "store_output"))
      second_transaction = next(
        (
          transaction
          for transaction in group.transfer_manager._transactions.values()
          if transaction.completion_event == second_store
        ),
        None,
      )
      if (
        handle is not None
        and set(bad_request.dependency_events) <= seq._events_done
        and second_store not in seq._events_done
        and second_transaction is not None
        and group.transfer_manager.has_inflight_access(handle)
      ):
        break
    assert handle is not None
    assert first_store in seq._events_done
    assert second_store not in seq._events_done
    assert second_transaction is not None
    assert second_transaction.status is TransferStatus.RUNNING
    pins_before = self._pin_fingerprint(group, handle)
    allocator_pins_before = set(group.l2_sram._views[handle.allocation_id].pins)
    pending_before = group.l2_sram.snapshot()["pending_release"]
    with pytest.raises(MemoryInvariantError):
      group.release_l2(bad_request, sequencer=seq, cycle=cycle + 1)
    assert not group.l2_sram.is_released(handle)
    assert self._pin_fingerprint(group, handle) == pins_before
    assert group.l2_sram._views[handle.allocation_id].pins == allocator_pins_before
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

    hw = HardwareConfig().with_overrides(num_dma_channels=2, hbm_fixed_latency_cycles=10)
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
    artifact = prepare_group_source(
      sim.group, self._make_inflight_prefetch_module(), self._BINDINGS, sim=sim.sim
    )
    assert isinstance(artifact.entry, ExecTileGroupTask)
    task = artifact.entry
    group = sim.group
    group.load_task(task, input_bindings=self._BINDINGS)
    seq = group.sequencer
    request = next(
      action.args[0]
      for action in seq.task.actions
      if action.op == ExecGroupActionOp.RELEASE_L2 and action.args[0].buffer_slot == "prefetch_input"
    )
    second_prefetch = next(
      event for event in request.dependency_events if event.endswith("input_prefetch_2")
    )
    bad_request = replace(
      request,
      dependency_events=tuple(event for event in request.dependency_events if event != second_prefetch),
    )
    handle = None
    transaction = None
    for cycle in range(200000):
      group.step(cycle)
      handle = group._l2_handles.get((seq.context_launch_generation, "prefetch_input"))
      transaction = next(
        (
          candidate
          for candidate in group.transfer_manager._transactions.values()
          if candidate.completion_event == second_prefetch
        ),
        None,
      )
      if (
        handle is not None
        and set(bad_request.dependency_events) <= seq._events_done
        and second_prefetch not in seq._events_done
        and transaction is not None
        and group.transfer_manager.has_inflight_access(handle)
      ):
        break
    assert handle is not None
    assert transaction is not None
    assert transaction.status is TransferStatus.RUNNING
    pins_before = self._pin_fingerprint(group, handle)
    allocator_pins_before = set(group.l2_sram._views[handle.allocation_id].pins)
    pending_before = group.l2_sram.snapshot()["pending_release"]
    with pytest.raises(MemoryInvariantError):
      group.release_l2(bad_request, sequencer=seq, cycle=cycle + 1)
    assert not group.l2_sram.is_released(handle)
    assert self._pin_fingerprint(group, handle) == pins_before
    assert group.l2_sram._views[handle.allocation_id].pins == allocator_pins_before
    assert group.l2_sram.snapshot()["pending_release"] == pending_before
    fault_drain_and_reset(group)
    assert transaction.status in (TransferStatus.DONE, TransferStatus.CANCELLED)
    assert not group.transfer_manager.has_inflight_access(handle)
    self._assert_runtime_zero_leak(group)
    assert group.credit_invariants_hold()
    assert sim.tracer is not None
    sim.tracer.assert_well_formed()
    group.reset()
    self._assert_runtime_zero_leak(group)

  def test_release_preflight_rejects_unacknowledged_reference_without_mutation(self, monkeypatch):
    """plan/01 §2.4/§3.4: a DONE-but-unacknowledged transfer keeps its ledger
    reference; the release preflight must consult the physical ledger and
    fail before any mutation (the manager query alone misses it)."""
    from pipeline_validator.memory.allocator import MemoryInvariantError
    from pipeline_validator.memory.transfer import TransferStatus

    hw = HardwareConfig().with_overrides(num_dma_channels=2, hbm_fixed_latency_cycles=10)
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
    artifact = prepare_group_source(
      sim.group, self._make_inflight_prefetch_module(), self._BINDINGS, sim=sim.sim
    )
    assert isinstance(artifact.entry, ExecTileGroupTask)
    task = artifact.entry
    group = sim.group
    group.load_task(task, input_bindings=self._BINDINGS)
    seq = group.sequencer
    request = next(
      action.args[0]
      for action in seq.task.actions
      if action.op == ExecGroupActionOp.RELEASE_L2 and action.args[0].buffer_slot == "prefetch_input"
    )
    second_prefetch = next(
      event for event in request.dependency_events if event.endswith("input_prefetch_2")
    )
    monkeypatch.setattr(group.transfer_manager, "acknowledge", lambda transaction_id, cycle: None)
    handle = None
    transaction = None
    for cycle in range(200000):
      group.step(cycle)
      handle = group._l2_handles.get((seq.context_launch_generation, "prefetch_input"))
      transaction = next(
        (
          candidate
          for candidate in group.transfer_manager._transactions.values()
          if candidate.completion_event == second_prefetch
        ),
        None,
      )
      if transaction is not None and transaction.status is TransferStatus.DONE:
        break
    assert handle is not None and transaction is not None
    assert transaction.status is TransferStatus.DONE
    # The manager query classifies DONE as finished; only the physical
    # ledger still sees the reference.
    assert not group.transfer_manager.has_inflight_access(handle)
    view = group.l2_sram._views[handle.allocation_id]
    inflight_before = set(view.inflight)
    assert inflight_before
    pins_before = self._pin_fingerprint(group, handle)
    snapshot_before = group.l2_sram.snapshot()
    with pytest.raises(MemoryInvariantError, match="unacknowledged transfer reference"):
      group.release_l2(request, sequencer=seq, cycle=cycle + 1)
    # Zero mutation: pins, ledger, view and free map all untouched.
    assert self._pin_fingerprint(group, handle) == pins_before
    assert set(group.l2_sram._views[handle.allocation_id].inflight) == inflight_before
    assert not group.l2_sram.is_released(handle)
    assert group.l2_sram.snapshot() == snapshot_before
    # Mirror divergence between the view and backing ledgers is an
    # invariant fault: blocking on either side independently.
    saved_ledger = set(view.inflight)
    view.inflight.clear()
    try:
      with pytest.raises(MemoryInvariantError, match="ledgers disagree"):
        group.l2_sram.has_inflight_references(handle)
    finally:
      view.inflight.update(saved_ledger)
    # Dropping the ledger references the way the acknowledgement hook would
    # (pool-level end_inflight) re-permits the release.  Owner jobs keep
    # ownership of their own acknowledgements; none are issued manually.
    monkeypatch.undo()
    for txn_id in sorted(inflight_before):
      group.l2_sram.end_inflight(handle, txn_id, cycle + 2)
    assert not group.l2_sram.has_inflight_references(handle)
    live_before = group.l2_sram.snapshot()["live_backings"]
    assert group.release_l2(request, sequencer=seq, cycle=cycle + 3)
    assert group.l2_sram.snapshot()["live_backings"] == live_before - 1
    # The sequencer's own release action now faults on the forfeited view;
    # drain that to a clean, well-formed stop.
    fault_drain_and_reset(group)
    self._assert_runtime_zero_leak(group)

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
    plan = l2.plan_bundle([AllocationRequest("l2", "status_buffer", owner, 4096, 256)])
    assert not isinstance(plan, AdmissionFailure)
    original = l2.commit(plan, cycle=0)[0]
    source = ResolvedMemoryView(
      handle=original,
      offset_bytes=0,
      size_bytes=4096,
      address=original.base_address,
      segments=original.bank_segments,
    )
    destination_handle = replace(original, allocation_id="l1:status-access", memory_space="l1")
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
      issuer=TaskBufferOwner("status_ctx", 0, "status-grid", 0, 0, 0, "status-l1"),
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
      TransferStatus.CANCEL_REQUESTED,
      TransferStatus.FAULTED,
    ):
      transaction.status = status
      assert manager.has_inflight_access(original)
    for status in (TransferStatus.DONE, TransferStatus.CANCELLED):
      transaction.status = status
      assert not manager.has_inflight_access(original)
    same_id_new_generation = replace(original, generation=original.generation + 1)
    transaction.src = destination
    transaction.dst = source
    transaction.status = TransferStatus.RUNNING
    assert manager.has_inflight_access(original)
    assert not manager.has_inflight_access(same_id_new_generation)

    assert l2.request_release(original, owner, cycle=1)
    l2.reset()
    next_owner = ContextBufferOwner("status_ctx", 1, "status_buffer")
    next_plan = l2.plan_bundle([AllocationRequest("l2", "status_buffer", next_owner, 4096, 256)])
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


# ---------------------------------------------------------------------------
# Root admission waits for whole-Arena retirement
# ---------------------------------------------------------------------------


class TestRootArenaAdmission:
  BINDINGS: ClassVar[dict[str, GlobalBinding]] = {
    "A_IN": GlobalBinding("A_IN", 0x100000, 131072, "rw"),
    "A_OUT": GlobalBinding("A_OUT", 0x200000, 131072, "rw"),
    "B_IN": GlobalBinding("B_IN", 0x300000, 131072, "rw"),
  }

  @pytest.mark.parametrize("fidelity", ["runtime", "full_memory"])
  def test_pending_root_admits_on_input_extent_release(self, fidelity):
    """R3-6: B's root admission waits for A's input backing physical release.

    plan/01 §6.4: the same capacity contract holds in both fidelities.  While
    pending, B holds no device slot, no L2 arena/backing and no partial
    event reserve; A's a_input ``l2_extent_release`` is the exact cycle of
    B's port admission and strictly precedes A's context completion.  B's
    real HBM→L2 prefetch transaction overlaps A's ``EVU:pow`` execution.
    """
    root = Path(__file__).resolve().parents[2]
    hw = HardwareConfig.from_yaml(root / "examples/configs/profile_l2_256k.yaml").with_overrides(
      num_dma_channels=2, hbm_fixed_latency_cycles=10
    )
    sim_config = SimConfig(
      fidelity=fidelity, device_context_count=2, memory_trace=True, max_cycles=200000
    )
    simulator = Simulator(hw, sim_config, enable_tracer=True)
    module = load_workload_ir(root / "examples/scenarios/l2_admission_wait.mlir")
    result = run_source(simulator, module, self.BINDINGS)
    assert result.completed, result.reason

    ports = {record["context"]: record for record in result.device_snapshot["port"]["request_records"]}
    records = {record["context"]: record for record in result.device_snapshot["launch_records"]}
    port_a, port_b = ports["ctx_a"], ports["ctx_b"]
    assert port_b["submit_cycle"] < port_a["completion_cycle"]
    assert result.device_snapshot["port"]["pending_peak"] >= 1
    assert result.device_snapshot["port"]["active_peak"] == 2

    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    to_cycle = lambda event: round(event["ts"] * 1000.0 / hw.cycle_ns())
    releases = [
      event
      for event in events
      if event.get("name") == "l2_extent_release"
      and event.get("args", {}).get("buffer_id") == "a_input"
    ]
    assert len(releases) == 1
    release_args = releases[0]["args"]
    release_cycle = release_args["release_cycle"]
    assert release_args["run_generation"] == simulator.group.run_generation == 1
    assert release_args["padded_bytes"] == 131072
    assert release_args["pool_version"] >= 1
    assert release_args["per_bank_segments"]
    assert all(segment["size_bytes"] > 0 for segment in release_args["per_bank_segments"])

    # B's port admission is exactly the input extent release cycle, strictly
    # before A's completion; the device launch record agrees.
    assert port_b["active_cycle"] == release_cycle
    assert release_cycle < port_a["completion_cycle"]
    assert records["ctx_b"]["active_cycle"] == release_cycle
    assert records["ctx_b"]["admission_cycle"] <= release_cycle

    # While pending, B never reserved an L2 arena: every ctx_a-reserve
    # precedes the release and every ctx_b-reserve follows it.
    reserves = [
      (event.get("args", {}).get("context_name"), to_cycle(event))
      for event in events
      if event.get("name") == "arena_reserve" and event.get("args", {}).get("space") == "l2"
    ]
    assert reserves
    assert all(cycle < release_cycle for context, cycle in reserves if context == "ctx_a")
    assert all(cycle >= release_cycle for context, cycle in reserves if context == "ctx_b")

    # A's whole-Arena retirement stays a distinct, later event: buffer
    # release must not be masked as an Arena capacity return.
    a_retires = [
      to_cycle(event)
      for event in events
      if event.get("name") == "arena_retire"
      and event.get("args", {}).get("space") == "l2"
      and event.get("args", {}).get("context_name") == "ctx_a"
    ]
    assert a_retires and min(a_retires) > release_cycle

    # Real transfer evidence: B's HBM→L2 prefetch transaction (first leg to
    # last leg) overlaps A's EVU:pow execution window by a positive number
    # of cycles, while A's output store and context are still unfinished.
    pow_spans = [
      (to_cycle(event), to_cycle(event) + max(round(event["dur"]), 1))
      for event in events
      if event.get("cat") == "EVU" and event.get("ph") == "X" and event.get("name") == "EVU:pow"
    ]
    assert pow_spans
    prefetch_legs = [
      (to_cycle(event), to_cycle(event) + max(round(event["dur"]), 1))
      for event in events
      if event.get("ph") == "X"
      and str(event.get("args", {}).get("transaction_id", "")).endswith("ev_pref_b")
    ]
    assert prefetch_legs
    first_leg = min(start for start, _ in prefetch_legs)
    last_leg = max(end for _, end in prefetch_legs)
    overlap = sum(
      max(0, min(pow_end, last_leg) - max(pow_start, first_leg)) for pow_start, pow_end in pow_spans
    )
    assert overlap > 0
    assert last_leg < port_a["completion_cycle"]
    assert result.group_snapshot["memory"]["l2"]["live_arenas"] == 0
    assert result.group_snapshot["task_leases"]["active"] == 0


class TestTileFreeRuntime:
  BINDINGS: ClassVar[dict[str, GlobalBinding]] = {"input": GlobalBinding("input", 0x100000, 8192, "r")}

  @staticmethod
  def make_module(explicit_free=True):
    from pipeline_validator.dialects.elenor import TileFreeOp

    programs = []
    for name, iterations in (("free_holder", 20), ("free_peer", 40)):
      l1_bytes = 16384 if name == "free_holder" else 8192
      program = TileProgramDefOp(
        name,
        tile_resources(l1_bytes),
        arg_types=[NestTask(), NestBuffer.of([1, 64, 64], "bf16")],
        arg_names=["task", "input"],
      )
      task_arg, buffer_arg = program.body.block.args
      view = TileSubviewOp(
        buffer_arg, task_arg, 0, [0, 0, 0], [1, 64, 64], [1, 1, 1], NestL2View.of([1, 64, 64], "bf16")
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
      "free_context",
      context_resources(2, 8192, requested_contexts_per_tile=2),
      placement=1,
      arg_types=[NestGlobalMemref.of([1, 64, 64], "bf16")],
      arg_names=["input"],
    )
    buf = NestAllocOp("input", "in", [1, 64, 64], "bf16", alignment=256)
    global_view = NestSubviewOp(
      context.body.block.args[0], [0, 0, 0], [1, 64, 64], [1, 1, 1], NestGlobalView.of([1, 64, 64], "bf16")
    )
    pref = NestPrefetchOp(global_view.result, buf.result, "prefetched")
    tasks = NestTaskRangeOp(0, 1)
    holder = NestDispatchOp(
      "free_holder",
      tasks.result,
      [],
      [buf.result],
      [],
      "holder_grid",
      "holder_read",
      "",
      l1_mode=0,
      bindings=[buf.result],
      signal_policy={"input_released": "all_tasks"},
      depends_on=[pref.result],
      context_id=0,
    )
    peer = NestDispatchOp(
      "free_peer",
      tasks.result,
      [],
      [buf.result],
      [],
      "peer_grid",
      "peer_read",
      "",
      l1_mode=0,
      bindings=[buf.result],
      signal_policy={"input_released": "all_tasks"},
      depends_on=[holder.input_released],
      context_id=1,
    )
    context.body.block.add_ops(
      [
        buf,
        global_view,
        pref,
        tasks,
        holder,
        peer,
        NestReleaseOp(buf.result, depends_on=[holder.input_released, peer.input_released, pref.result]),
        NestAwaitOp([holder.grid_done, peer.grid_done]),
        NestReturnOp(),
      ]
    )
    return ModuleOp([*programs, context])

  @staticmethod
  def make_sim(fidelity):
    return Simulator(
      HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10),
      SimConfig(fidelity=fidelity, context_count=2, memory_trace=True, max_cycles=200000),
      enable_tracer=True,
    )

  @staticmethod
  def assert_empty(group):
    if group.runtime_enabled:
      assert group.l2_sram.snapshot()["live_arenas"] == 0
    assert group.transfer_manager.inflight_count == 0
    for tile in group.tiles:
      assert tile.l1_allocator.snapshot()["live_arenas"] == 0
    assert not group._grid_l2_pins
    assert not any(group._role_l1_handles.values())
    assert group.credit_invariants_hold()

  @pytest.mark.parametrize("fidelity", ["runtime", "full_memory"])
  def test_explicit_free_invalidates_only_the_view_before_task_arena_retirement(self, fidelity):
    simulator = self.make_sim(fidelity)
    result = run_source(simulator, self.make_module(explicit_free=True), self.BINDINGS)
    assert result.completed, result.reason
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    invalidations = [
      event
      for event in events
      if event.get("name") == "buffer_view_invalidate" and event.get("args", {}).get("buffer_id") == "l1:0"
    ]
    retirements = [
      event
      for event in events
      if event.get("name") == "arena_retire" and event.get("args", {}).get("space") == "l1"
    ]
    assert invalidations and retirements
    assert min(event["ts"] for event in invalidations) < max(event["ts"] for event in retirements)
    assert all(event["args"]["pool_reserved_bytes"] > 0 for event in invalidations)
    self.assert_empty(simulator.group)
    assert result.credit_invariant_ok

  @pytest.mark.parametrize("fidelity", ["runtime", "full_memory"])
  def test_implicit_task_return_retires_unfreed_views_and_arena(self, fidelity):
    simulator = self.make_sim(fidelity)
    result = run_source(simulator, self.make_module(explicit_free=False), self.BINDINGS)
    assert result.completed, result.reason
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    implicit_invalidations = [
      event
      for event in events
      if event.get("name") == "buffer_view_invalidate"
      and event.get("args", {}).get("buffer_id") in {"l1:0", "l1:1"}
    ]
    assert implicit_invalidations
    assert any(
      event.get("name") == "arena_retire" and event.get("args", {}).get("space") == "l1" for event in events
    )
    self.assert_empty(simulator.group)
    assert result.credit_invariant_ok


class TestCycleCapPoison:
  """plan/01 §4.4/§6.6: the post-cap drain itself can fail and poison."""

  BINDINGS: ClassVar[dict[str, GlobalBinding]] = {
    "A_IN": GlobalBinding("A_IN", 0x100000, 131072, "rw"),
    "A_OUT": GlobalBinding("A_OUT", 0x200000, 131072, "rw"),
    "B_IN": GlobalBinding("B_IN", 0x300000, 131072, "rw"),
  }

  def test_unisolatable_state_after_cap_poisons_group_and_rejects_reuse(self):
    from dataclasses import replace as dc_replace

    from pipeline_validator.memory.allocator import MemoryInvariantError

    root = Path(__file__).resolve().parents[2]
    hw = HardwareConfig.from_yaml(root / "examples/configs/profile_l2_256k.yaml").with_overrides(
      num_dma_channels=2, hbm_fixed_latency_cycles=10
    )
    hw = dc_replace(hw, memory_target=dc_replace(hw.memory_target, profile_command_timeout_cycles=500))
    sim_config = SimConfig(
      fidelity="full_memory", device_context_count=2, memory_trace=True, max_cycles=60
    )
    simulator = Simulator(hw, sim_config, enable_tracer=True)
    module = load_workload_ir(root / "examples/scenarios/l2_admission_wait.mlir")
    group = simulator.group
    original_step = group.transfer_manager.step

    def stalled_step(cycle):
      # Run normally up to the cap so transfers are real, issued work; then
      # freeze every leg so the drain can neither confirm isolation nor
      # acknowledge the held references.
      if cycle < 60:
        return original_step(cycle)
      return []

    group.transfer_manager.step = stalled_step
    result = run_source(simulator, module, self.BINDINGS)
    assert not result.completed
    assert "poisoned" in result.reason, result.reason
    assert group.poisoned_reason is not None

    snapshot = group.l2_sram.snapshot()
    # Unsafe physical state is retained, never returned to the free map, and
    # the poison reason names the concrete backing IDs (plan/01 §4.4).
    assert snapshot["live_backings"] >= 1
    assert snapshot["arena_reserved_bytes"] > 0
    assert snapshot["free_bytes"] < (
      snapshot["user_spm_capacity_bytes"] - snapshot["system_reserved_bytes"]
    )
    backing_ids = group.l2_sram.live_backing_ids()
    assert backing_ids
    assert all(backing_id in result.reason for backing_id in backing_ids)

    # Poisoned groups reject a new launch outright...
    with pytest.raises(MemoryInvariantError, match="poisoned"):
      group.begin_launch(None, None)
    # ...and an explicit reset is refused while the drain never completed.
    assert not group.reset_domain.is_done
    with pytest.raises(MemoryInvariantError, match="explicit reset requires"):
      group.reset()


class TestCycleCapDrain:
  """plan/01 §6.6: a low cap still drains to DONE without poison."""

  BINDINGS: ClassVar[dict[str, GlobalBinding]] = {
    "A_IN": GlobalBinding("A_IN", 0x100000, 131072, "rw"),
    "A_OUT": GlobalBinding("A_OUT", 0x200000, 131072, "rw"),
    "B_IN": GlobalBinding("B_IN", 0x300000, 131072, "rw"),
  }

  def test_low_cycle_cap_drains_to_done_and_stays_clean(self):
    root = Path(__file__).resolve().parents[2]
    hw = HardwareConfig.from_yaml(root / "examples/configs/profile_l2_256k.yaml").with_overrides(
      num_dma_channels=2, hbm_fixed_latency_cycles=10
    )
    sim_config = SimConfig(
      fidelity="full_memory", device_context_count=2, memory_trace=True, max_cycles=60
    )
    simulator = Simulator(hw, sim_config, enable_tracer=True)
    module = load_workload_ir(root / "examples/scenarios/l2_admission_wait.mlir")
    result = run_source(simulator, module, self.BINDINGS)
    group = simulator.group
    # The cap is a failure, never a fabricated success; the extra drain ran
    # the reset domain to DONE and the private L2 closed zero-leak.
    assert not result.completed
    assert "cycle cap 60 reached" in result.reason
    assert "poisoned" not in result.reason
    assert group.poisoned_reason is None
    assert group.reset_domain.is_done
    group.assert_l2_closed()
    snapshot = group.l2_sram.snapshot()
    assert snapshot["live_backings"] == 0
    assert snapshot["arena_reserved_bytes"] == 0

  def test_explicit_reset_recovers_a_marked_group_after_safe_drain(self):
    """plan/01 §4.4/§6.6: only an explicit quiescent reset clears poison.

    The drain in this scenario reaches DONE and the physical state is clean,
    so recovery must be possible; the marker stands in for a poison flag the
    un-isolatable path would have set (that path itself is covered by
    TestCycleCapPoison).
    """
    root = Path(__file__).resolve().parents[2]
    hw = HardwareConfig.from_yaml(root / "examples/configs/profile_l2_256k.yaml").with_overrides(
      num_dma_channels=2, hbm_fixed_latency_cycles=10
    )
    sim_config = SimConfig(
      fidelity="full_memory", device_context_count=2, memory_trace=True, max_cycles=60
    )
    simulator = Simulator(hw, sim_config, enable_tracer=True)
    module = load_workload_ir(root / "examples/scenarios/l2_admission_wait.mlir")
    result = run_source(simulator, module, self.BINDINGS)
    group = simulator.group
    assert not result.completed and group.reset_domain.is_done
    assert group.poisoned_reason is None
    group.poison("synthetic marker for recovery path")
    group.reset()
    assert group.poisoned_reason is None


class TestSuccessExitClosure:
  """plan/01 §4.3: a leak at the success exit becomes a controller-visible
  fault that drains before the run reports failure."""

  BINDINGS: ClassVar[dict[str, GlobalBinding]] = {
    "A_IN": GlobalBinding("A_IN", 0x100000, 131072, "rw"),
    "A_OUT": GlobalBinding("A_OUT", 0x200000, 131072, "rw"),
    "B_IN": GlobalBinding("B_IN", 0x300000, 131072, "rw"),
  }

  def test_closure_violation_faults_drains_and_reports_failure(self, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    hw = HardwareConfig.from_yaml(root / "examples/configs/profile_l2_256k.yaml").with_overrides(
      num_dma_channels=2, hbm_fixed_latency_cycles=10
    )
    sim_config = SimConfig(
      fidelity="full_memory", device_context_count=2, memory_trace=True, max_cycles=200000
    )
    simulator = Simulator(hw, sim_config, enable_tracer=True)
    module = load_workload_ir(root / "examples/scenarios/l2_admission_wait.mlir")
    group = simulator.group
    first = {"pending": True}
    original = group.assert_l2_closed

    def injecting_closed():
      if first["pending"]:
        first["pending"] = False
        raise MemoryInvariantError("injected leak: backing l2:p0:arena:1:backing:1:a_input")
      original()

    monkeypatch.setattr(group, "assert_l2_closed", injecting_closed)
    result = run_source(simulator, module, self.BINDINGS)
    assert not result.completed
    assert "L2 closure violation" in result.reason
    assert "injected leak" in result.reason
    # The drain ran to DONE and the real state is clean afterwards.
    assert group.reset_domain.is_done
    assert group.poisoned_reason is None
    original()


class TestL2ProfileSwitchOrdering:
  """plan TODO / batch-III seed: a cross-profile successor's first load is
  NOT pulled forward to the predecessor's early input release; it waits for
  the completed L2 profile switch whose frontier is the predecessor's HBM
  store and completion (demonstrated with NO source-level await)."""

  BINDINGS: ClassVar[dict[str, GlobalBinding]] = {
    "A_IN": GlobalBinding("A_IN", 0x100000, 131072, "r"),
    "A_OUT": GlobalBinding("A_OUT", 0x200000, 131072, "rw"),
    "B_IN": GlobalBinding("B_IN", 0x300000, 131072, "r"),
  }

  @pytest.mark.parametrize("fidelity", ["runtime", "full_memory"])
  def test_cross_profile_load_waits_for_switch_not_release(self, fidelity):
    root = Path(__file__).resolve().parents[2]
    hw = HardwareConfig.from_yaml(root / "examples/configs/profile_l2_switch.yaml").with_overrides(
      num_dma_channels=2, hbm_fixed_latency_cycles=10
    )
    sim_config = SimConfig(
      fidelity=fidelity, device_context_count=2, memory_trace=True, max_cycles=300000
    )
    simulator = Simulator(hw, sim_config, enable_tracer=True)
    module = load_workload_ir(root / "examples/scenarios/l2_profile_switch_load_ordering.mlir")
    result = run_source(simulator, module, self.BINDINGS)
    assert result.completed, result.reason

    ports = {record["context"]: record for record in result.device_snapshot["port"]["request_records"]}
    trace = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    to_cycle = lambda event: round(event["ts"] * 1000.0 / hw.cycle_ns())

    releases = [
      event
      for event in trace
      if event.get("name") == "l2_extent_release"
      and event.get("args", {}).get("buffer_id") == "a_input"
    ]
    assert len(releases) == 1
    release_cycle = releases[0]["args"]["release_cycle"]

    # A's HBM store completion: last leg of the global_store transaction.
    store_legs = [
      (to_cycle(event), to_cycle(event) + max(round(event["dur"] * 1000.0), 1))
      for event in trace
      if event.get("ph") == "X" and event.get("args", {}).get("op") == "global_store"
    ]
    assert store_legs
    store_done = max(end for _, end in store_legs)

    # The compiler-generated L2 0->1 switch: args carry real cycles.
    switches = [
      event
      for event in trace
      if event.get("name") == "profile_command"
      and event.get("args", {}).get("level") == "l2"
      and event.get("args", {}).get("status") == "completed"
      and event.get("ph") == "X"
    ]
    assert switches, "no completed L2 profile command in trace"
    switch = max(switches, key=lambda event: event["args"]["completed_cycle"])
    switch_start = switch["args"]["accepted_cycle"]
    switch_end = switch["args"]["completed_cycle"]
    assert switch_end > switch_start

    prefetch_legs = [
      to_cycle(event)
      for event in trace
      if event.get("ph") == "X"
      and str(event.get("args", {}).get("transaction_id", "")).endswith("ev_pref_b")
    ]
    assert prefetch_legs
    first_prefetch = min(prefetch_legs)
    # B's first L2->L1 tile load after its prefetch.
    tile_loads = [
      to_cycle(event)
      for event in trace
      if event.get("ph") == "X"
      and event.get("args", {}).get("op") == "tile_load"
      and to_cycle(event) >= first_prefetch
    ]
    assert tile_loads
    tile_load = min(tile_loads)

    completion = ports["ctx_a"]["completion_cycle"]
    admit = ports["ctx_b"]["active_cycle"]
    # The full requested chain, in cycles:
    #   release < store_done <= A_done <= switch_start < switch_end
    #   <= B_admit < prefetch <= B tile_load
    assert release_cycle < store_done
    assert store_done <= completion
    assert completion <= switch_start
    assert switch_start < switch_end
    assert switch_end <= admit
    assert admit < first_prefetch
    assert first_prefetch <= tile_load

    # ctx_b held no L2 arena before the switch completed.
    b_reserves = [
      to_cycle(event)
      for event in trace
      if event.get("name") == "arena_reserve"
      and event.get("args", {}).get("space") == "l2"
      and event.get("args", {}).get("context_name") == "ctx_b"
    ]
    assert b_reserves and min(b_reserves) >= switch_end

    # The pool actually ended on the new profile.
    assert result.group_snapshot["arenas"]["l2"]["profile_mode"] == 1
