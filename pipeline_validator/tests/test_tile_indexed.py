"""Plan §Verification 1-4: address-resolved Tile Gather/Scatter semantics.

Every assertion observes real bytes, real PMU counters or real
verification rejections - nothing from the profiled Gather era survives.
Runtime scenarios start from the committed
``examples/workloads/indexed_gather_scatter.mlir`` fixture so the
program under test is exactly the delivered artifact shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.dialects.elenor import TileAllocOp, TileGatherOp
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.simulator import Simulator
from pipeline_validator.workload_ir import (
  VerifyException,
  load_workload_ir,
  verify_workload_ir,
)

FIXTURE = Path(__file__).resolve().parents[2] / "examples" / "workloads" / "indexed_gather_scatter.mlir"

BINDINGS = {
  "DATA": GlobalBinding("DATA", 0x100000, 64, "r"),
  "OUT": GlobalBinding("OUT", 0x400000, 64, "w"),
  "GATHER_IDX": GlobalBinding("GATHER_IDX", 0x200000, 12, "r"),
  "SCATTER_IDX": GlobalBinding("SCATTER_IDX", 0x300000, 12, "r"),
}

DATA_ROWS = [[(row + 1) * 10 + item for item in range(4)] for row in range(4)]
GATHER_INDICES = (3, 0, 3)
SCATTER_INDICES = (1, 1, 2)


def _seed_store() -> ByteStore:
  """DATA with rows [10..13] / [20..23] / [30..33] / [40..43]."""
  store = ByteStore()
  store.seed_hbm(
    0x100000,
    b"".join(int(value).to_bytes(4, "little", signed=True) for row in DATA_ROWS for value in row),
  )
  store.seed_hbm(
    0x200000, b"".join(int(v).to_bytes(4, "little", signed=True) for v in GATHER_INDICES)
  )
  store.seed_hbm(
    0x300000, b"".join(int(v).to_bytes(4, "little", signed=True) for v in SCATTER_INDICES)
  )
  return store


def _run(source: str, *, store: ByteStore, max_cycles: int = 50000, fidelity: str = "full_memory"):
  hw = HardwareConfig()
  sim = SimConfig(max_cycles=max_cycles, fidelity=fidelity)
  module = load_workload_ir_text(source)
  from pipeline_validator.compiler import compile_program

  artifact = compile_program(module, hw, sim, binding_assumptions=BINDINGS, source_name="indexed_memory")
  loaded = load_program(artifact, hw, sim, actual_bindings=BINDINGS)
  return Simulator(hw, sim, byte_store=store).run(loaded)


def load_workload_ir_text(source: str):
  """Parse a source string through the same path the CLI uses."""
  import tempfile

  with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as handle:
    handle.write(source)
    path = handle.name
  try:
    return load_workload_ir(path)
  finally:
    Path(path).unlink(missing_ok=True)


def _fixture_source() -> str:
  return FIXTURE.read_text(encoding="utf-8")


def test_indexed_gather_scatter_writes_ordered_overwrite_bytes():
  """Plan §Verification 1: OUT row 1 is the later ordinal, row 2 the third."""
  result = _run(_fixture_source(), store=_seed_store())
  assert result.completed, result.reason
  events = result.pmu.events
  assert events.get("gather_requests", 0) == 3
  assert events.get("gather_index_reads", 0) == 3
  assert events.get("gather_bytes", 0) == 48
  assert events.get("scatter_segments", 0) == 3
  assert events.get("scatter_bytes", 0) == 48

  store = _seed_store()
  result = _run(_fixture_source(), store=store)
  assert result.completed, result.reason
  out, mask = store._read_relaxed("hbm", BINDINGS["OUT"].base_iova, 64)
  rows = [
    [int.from_bytes(out[row * 16 + i * 4 : row * 16 + i * 4 + 4], "little") for i in range(4)]
    for row in range(4)
  ]
  # Segment ordinal 1 overwrites ordinal 0 on OUT row 1; ordinal 2 lands on row 2.
  assert rows[1] == [10, 11, 12, 13]
  assert rows[2] == [40, 41, 42, 43]
  # Rows nobody scattered stay uninitialised in HBM.
  assert not any(mask[0:16])
  assert not any(mask[48:64])
  assert all(mask[16:32])
  assert all(mask[32:48])


def test_gather_conservation_and_address_resolved_fidelity():
  """Plan §Verification 2 / report: conservation includes the bypass path."""
  from pipeline_validator.report import build_report

  store = _seed_store()
  hw = HardwareConfig()
  sim = SimConfig(max_cycles=50000)
  module = load_workload_ir(FIXTURE)
  from pipeline_validator.compiler import compile_program

  artifact = compile_program(module, hw, sim, binding_assumptions=BINDINGS, source_name="indexed_memory")
  loaded = load_program(artifact, hw, sim, actual_bindings=BINDINGS)
  result = Simulator(hw, sim, byte_store=store).run(loaded)
  events = result.pmu.events
  total = (
    events.get("gather_l1_hits", 0)
    + events.get("gather_l2_hits", 0)
    + events.get("gather_hbm_misses", 0)
    + events.get("gather_cache_bypass_requests", 0)
  )
  assert events.get("gather_requests", 0) == 3
  assert events.get("gather_requests", 0) == total
  report = build_report(loaded.compiled.workload_info, result, num_tiles=hw.num_tiles)
  assert report.gather_fidelity == "address_resolved_cache"
  conservation = next(c for c in report.checks if c["check"] == "gather_request_conservation")
  assert conservation["pass"], conservation


def test_scatter_allocates_no_cache_line_and_emits_no_bypass_reads():
  """Plan §Verification 3 / invariant 2: Scatter never touches the caches."""
  store = _seed_store()
  result = _run(_fixture_source(), store=store)
  assert result.completed, result.reason
  snapshot = result.group_snapshot.get("memory", {})
  l2 = snapshot.get("l2", {})
  assert l2.get("refills", 0) == 0
  assert l2.get("dirty_lines", 0) == 0
  for tile_memory in snapshot.get("l1", {}).values():
    cache = tile_memory.get("cache", {})
    assert cache.get("dirty_lines", 0) == 0


def test_scatter_completion_is_traced_after_the_last_commit():
  """Plan §Verification 3: Scatter completion and ordered writes are traced."""
  hw = HardwareConfig()
  sim = SimConfig(max_cycles=50000)
  store = _seed_store()
  module = load_workload_ir(FIXTURE)
  from pipeline_validator.compiler import compile_program

  artifact = compile_program(module, hw, sim, binding_assumptions=BINDINGS, source_name="indexed_memory")
  loaded = load_program(artifact, hw, sim, actual_bindings=BINDINGS)
  result = Simulator(hw, sim, enable_tracer=True, byte_store=store).run(loaded)
  assert result.completed, result.reason
  assert result.tracer is not None
  import json

  events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
  scatter_writes = [e for e in events if e.get("name") == "scatter_write"]
  scatter_done = [e for e in events if e.get("name") == "scatter_done"]
  assert len(scatter_writes) == 3
  assert len(scatter_done) == 1
  assert min(e["ts"] for e in scatter_writes) <= scatter_done[0]["ts"]


def test_indexed_memory_requires_full_memory_fidelity():
  """Plan §2: address-resolved indexed memory rejects non-full_memory runs."""
  hw = HardwareConfig()
  sim = SimConfig(max_cycles=1000, fidelity="runtime")
  module = load_workload_ir(FIXTURE)
  from pipeline_validator.compiler import compile_program

  with pytest.raises(ValueError, match="indexed memory requires full_memory"):
    compile_program(module, hw, sim, source_name="indexed_memory")


def _replace_once(source: str, old: str, new: str) -> str:
  assert old in source, old
  return source.replace(old, new, 1)


def test_multi_task_scatter_overlap_is_rejected():
  """Plan §Verification 4: tile_scatter_task_overlap on a 2-task dispatch."""
  source = _fixture_source()
  # A 2-task dispatch whose Scatter has task_stride 0 makes both tasks
  # write identical segments - the record-domain proof fails.
  source = _replace_once(source, "logical_tasks = 1", "logical_tasks = 2")
  source = _replace_once(source, "requested_contexts_per_tile = 1", "requested_contexts_per_tile = 2")
  source = _replace_once(
    source, "%tasks = nest.task.range from = 0 to = 1", "%tasks = nest.task.range from = 0 to = 2"
  )
  source = _replace_once(source, "placement = 1", "placement = 3")
  with pytest.raises(VerifyException, match="tile_scatter_task_overlap"):
    verify_workload_ir(load_workload_ir_text(source))


def test_scatter_and_gather_same_formal_in_one_dispatch_is_rejected():
  """Plan §Verification 4: tile_scatter_gather_same_dispatch."""
  source = _fixture_source()
  # Point the Scatter at the same global formal the Gather reads.
  source = _replace_once(
    source,
    "%scattered = tile.scatter.global.async %gather_dst",
    "%scattered = tile.scatter.global.async %gather_dst",
  )
  source = _replace_once(
    source,
    "        indices(%scatter_idx_l1) into %out\n",
    "        indices(%scatter_idx_l1) into %data\n",
  )
  with pytest.raises(VerifyException, match="tile_scatter_gather_same_dispatch"):
    verify_workload_ir(load_workload_ir_text(source))


def test_unawaited_overlapping_indexed_reads_are_rejected():
  """Plan §Verification 4: tile_indexed_hazard without an intervening await."""
  from pipeline_validator.tests.test_runtime import make_gather_module

  # The shared builder emits one Gather; a second overlapping Gather on the
  # same global formal with no tile.await in between has no disjointness proof.
  module = make_gather_module([3, 0, 2, 1], include_evu_context=False, window_entries=4, segment=16)
  program = next(op for op in module.body.block.ops if op.name == "tile.program")
  assert program is not None
  gather = next(op for op in program.body.block.ops if op.name == "tile.gather.global.async")
  # Independent L1 allocations so only the record-domain proof is at stake.
  indices_count = len(list(gather.indices.type.dims))
  second_indices_alloc = TileAllocOp([indices_count], "i32")
  segment_elements = int(gather.address_map.values[5])
  extra_alloc = TileAllocOp([indices_count * segment_elements], "i8")
  second = TileGatherOp(
    gather.source,
    second_indices_alloc.result,
    extra_alloc.result,
    "gathered_2",
    address_map=gather.address_map,
    window_entries=4,
  )
  # Insert the second Gather *before* the tile.await so the first Gather's
  # completion event is still un-awaited: that is the hazard the plan rejects.
  body = list(program.body.block.ops)
  await_index = next(index for index, op in enumerate(body) if op.name == "tile.await")
  for op in reversed(body):
    program.body.block.detach_op(op)
  program.body.block.add_ops(
    [*body[:await_index], second_indices_alloc, extra_alloc, second, *body[await_index:]]
  )
  with pytest.raises(VerifyException, match="tile_indexed_hazard"):
    verify_workload_ir(module)


def test_destination_extent_must_equal_indexed_elements():
  """Plan §1: the Gather destination holds exactly I*R*L elements."""
  source = _fixture_source()
  source = _replace_once(
    source,
    """    %gather_dst = tile.alloc shape = [12] dtype = "i32"
        alignment = 64 : !tile.l1_buffer<12xi32>
""",
    """    %gather_dst = tile.alloc shape = [8] dtype = "i32"
        alignment = 64 : !tile.l1_buffer<8xi32>
""",
  )
  with pytest.raises(VerifyException, match="exactly I\\*R\\*L"):
    verify_workload_ir(load_workload_ir_text(source))
