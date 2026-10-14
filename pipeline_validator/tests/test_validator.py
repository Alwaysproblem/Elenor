"""Tests for the ELENOR pipeline validator.

Run with:  python -m pytest pipeline_validator/tests/  (or: pytest)
"""

from __future__ import annotations

import json
import subprocess
import sys
from math import prod
from pathlib import Path

import pytest
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.exceptions import ParseError, VerifyException

from pipeline_validator.compiled_program import WorkloadInfo
from pipeline_validator.compiler import compile_program
from pipeline_validator.compiler.resources import conservative_arena_bytes
from pipeline_validator.config import HardwareConfig, SimConfig, _load_hw_yaml
from pipeline_validator.dialects.elenor import (
  NestAllocOp,
  NestAwaitOp,
  NestBarrierOp,
  NestBuffer,
  NestCollectiveOp,
  NestContextOp,
  NestDispatchOp,
  NestDMAStoreOp,
  NestEvent,
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
  TaskRange,
  TileAllocOp,
  TileAwaitOp,
  TileBoaOp,
  TileEvent,
  TileEvuOp,
  TileGatherOp,
  TileIndexedMapAttr,
  TileL1Buffer,
  TileLoadOp,
  TilePowOp,
  TileProgramDefOp,
  TileReturnOp,
  TileSignalOp,
  TileStoreOp,
  TileSubviewOp,
)
from pipeline_validator.engines import EngineState, MFEEngine
from pipeline_validator.execution_ir import (
  ExecEngineDesc,
  ExecGroupActionOp,
  ExecTileGatherDesc,
  ExecTileOp,
  GlobalBinding,
)
from pipeline_validator.loader import load_program
from pipeline_validator.profiles import CacheRequirement, ContextResources, TileResources, build_registry
from pipeline_validator.report import build_report, report_to_text
from pipeline_validator.simulator import Simulator
from pipeline_validator.stream_queue import EOSPolicy, StreamQueue, StreamToken
from pipeline_validator.workload_builders import (
  make_identity_tile_program,
  make_pow_task,
  make_pow_tile_program,
)
from pipeline_validator.workload_ir import parse_workload_ir, print_workload_ir, verify_workload_ir
from pipeline_validator.workloads import ALL_WORKLOADS, PowWorkload

# Fast config for tests that don't validate the unfrozen 200-cycle HBM
# latency: lowers it to keep full_memory simulations under seconds.
FAST_HW = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
# ---------------------------------------------------------------------------
# xDSL workload IR tests
# ---------------------------------------------------------------------------

POW_BINDINGS = {"Y": GlobalBinding("Y", 0x100000, 524288, "rw")}


def compiled_entry(module: ModuleOp, *, hw: HardwareConfig = FAST_HW, sim: SimConfig | None = None):
  return compile_program(module, hw, sim or SimConfig()).entry


def run_source(
  simulator: Simulator,
  module: ModuleOp,
  bindings: dict[str, GlobalBinding] | None = None,
  *,
  workload_info: WorkloadInfo | None = None,
):
  artifact = compile_program(module, simulator.hw, simulator.sim, workload_info=workload_info)
  loaded = load_program(artifact, simulator.hw, simulator.sim, actual_bindings=bindings)
  return simulator.run(loaded)


def tile_resources(bytes_per_context: int = 0, *, allowed: tuple[int, ...] = (0, 1, 2)) -> TileResources:
  return TileResources(allowed_profiles=allowed, tile_l1_spm_bytes_per_context=bytes_per_context)


def context_resources(
  logical_tasks: int = 0,
  l2_spm_bytes: int = 0,
  *,
  l2_mode: int = 0,
  allowed: tuple[int, ...] = (0, 1, 2),
  requested_contexts_per_tile: int = 1,
) -> ContextResources:
  return ContextResources(
    l2_mode=l2_mode,
    allowed_profiles=allowed,
    logical_tasks=logical_tasks,
    l2_spm_bytes=l2_spm_bytes,
    requested_contexts_per_tile=requested_contexts_per_tile,
  )


def authored_arena_bytes(
  level: str, buffers: list[tuple[int, int]], *, hw: HardwareConfig = FAST_HW
) -> int:
  """Use the compiler's authoring helper; tests do not duplicate layout math."""
  return conservative_arena_bytes(buffers, build_registry(hw).profile(level, 0)) if buffers else 0


MODEL_CHAIN_IR = """builtin.module {
  tile.program @pow_4k_tile(
      %task : !nest.task,
      %l2_buf : !nest.l2_buffer<4x128x128xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 32768> {
    %l2_tile = tile.subview %l2_buf task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 128, 128] strides = [1, 1, 1]
        : !nest.l2_view<1x128x128xbf16>
    %l1 = tile.alloc shape = [128, 128] dtype = "bf16" alignment = 256
        : !tile.l1_buffer<128x128xbf16>
    %e_load = tile.load.async %l2_tile into %l1
        : !tile.event<"e_load">
    tile.await %e_load
    %e_pow = tile.pow.async bytes = 32768 exponent = 2 pow_ops = 65536
        : !tile.event<"e_pow">
    tile.await %e_pow
    %e_store = tile.store.async %l1 into %l2_tile
        : !tile.event<"e_store">
    tile.await %e_store
    tile.signal input_released(%task)
    tile.signal output_ready(%task)
    tile.return
  }

  nest.context @pow_task(
      %Y : !nest.global_memref<4x128x128xbf16>)
      placement = 15 context = 0
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
          logical_tasks = 4, l2_spm_bytes = 131072, requested_contexts_per_tile = 1> {
    %l2_buf = nest.alloc slot = "l2_buf_pow0" role = "inout"
        shape = [4, 128, 128] dtype = "bf16" alignment = 256
        : !nest.l2_buffer<4x128x128xbf16>
    %src = nest.subview %Y offsets = [0, 0, 0] sizes = [4, 128, 128]
        strides = [1, 1, 1]
        : !nest.global_view<4x128x128xbf16>
    %ev_in = nest.dma.prefetch.async %src into %l2_buf
        : !nest.event<"ev_dma_pow_in0">
    %0 = nest.task.range from = 0 to = 4 : !nest.task_range
    %ev_role, %ev_inrel, %ev_outready = nest.dispatch.tasks.async
        @pow_4k_tile l1_mode = 0 context = 0
        tasks(%0) globals() bindings(%l2_buf) ins(%l2_buf) outs(%l2_buf)
        signal_policy { input_released = #nest.aggregate<all_tasks>,
                        output_ready = #nest.aggregate<all_tasks> }
        depends_on(%ev_in)
        : (!nest.event<"ev_role_pow0">, !nest.event<"ev_inrel_pow0">,
           !nest.event<"ev_outready_pow0">)
    %ev_out = nest.dma.store.async %l2_buf into %src
        depends_on(%ev_outready) : !nest.event<"ev_dma_pow_out0">
    nest.release %l2_buf depends_on(%ev_inrel, %ev_in, %ev_out)
    nest.await %ev_role, %ev_out
    nest.return
  }

  nexus.program @run_pow(
      %Y0 : !nest.global_memref<4x128x128xbf16>,
      %Y1 : !nest.global_memref<4x128x128xbf16>) {
    %done0 = nexus.submit_context.async @pow_task(%Y0)
        : !nexus.event<"context_done">
    %done1 = nexus.submit_context.async @pow_task(%Y1)
        : !nexus.event<"context_done_1">
    nexus.await %done0
    nexus.await %done1
    nexus.return
  }
}
"""


TRANSFER_BYTE_MISMATCH_IR = """builtin.module {
  tile.program @p(%task : !nest.task)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 0> {
    tile.return
  }
  nest.context @c(%Y : !nest.global_memref<4x128x128xbf16>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
          logical_tasks = 0, l2_spm_bytes = 65536, requested_contexts_per_tile = 1> {
    %buf = nest.alloc slot = "buf" role = "in" shape = [2, 128, 128]
        dtype = "bf16" : !nest.l2_buffer<2x128x128xbf16>
    %src = nest.subview %Y offsets = [0, 0, 0] sizes = [4, 128, 128]
        strides = [1, 1, 1] : !nest.global_view<4x128x128xbf16>
    %ev = nest.dma.prefetch.async %src into %buf : !nest.event<"ev_in">
    nest.await %ev
    nest.return
  }
}
"""

TILE_PROGRAM_NO_TASK_IR = """builtin.module {
  tile.program @p(%l2 : !nest.l2_buffer<4x128x128xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 0> {
    tile.return
  }
  nest.context @c placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
          logical_tasks = 1, l2_spm_bytes = 131072, requested_contexts_per_tile = 1> {
    %buf = nest.alloc slot = "buf" role = "in" shape = [4, 128, 128]
        dtype = "bf16" : !nest.l2_buffer<4x128x128xbf16>
    %0 = nest.task.range from = 0 to = 1 : !nest.task_range
    %g, %i, %o = nest.dispatch.tasks.async @p l1_mode = 0
        tasks(%0) globals() bindings(%buf) ins() outs() signal_policy {}
        : (!nest.event<"g">, !nest.event<"i">, !nest.event<"o">)
    nest.await %g
    nest.return
  }
}
"""


GATHER_MAP = (
  "        map = #tile.indexed_map<index_scale = 64 offset = 0 task_stride = 0"
  " repeat = 1 stride = 0 segment = 16>"
)

GATHER_IR = f"""builtin.module {{
  tile.program @gather_tile(
      %task : !nest.task,
      %table : !nest.global_view<4096xi8>)
      resource_contract = #tile.resources<allowed_profiles = [1, 2],
          tile_l1_spm_bytes_per_context = 2048,
          l1_cache = {{required = false, access = "read", bypass = "allowed", target_bytes = 65536}},
          l2_cache = {{required = false, access = "read", bypass = "allowed", target_bytes = 65536}}> {{
    %indices = tile.alloc shape = [16] dtype = "i32"
        : !tile.l1_buffer<16xi32>
    %destination = tile.alloc shape = [256] dtype = "i8"
        : !tile.l1_buffer<256xi8>
    %done = tile.gather.global.async %table
        indices(%indices) into %destination
{GATHER_MAP}
        window_entries = 16 : !tile.event<"gather_done">
    tile.await %done
    tile.return
  }}

  nest.context @gather_context(
      %table : !nest.global_memref<4096xi8>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [1, 2],
          logical_tasks = 1, l2_spm_bytes = 0, requested_contexts_per_tile = 1,
          l2_cache = {{required = false, access = "read", bypass = "allowed", target_bytes = 65536}}> {{
    %table_view = nest.subview %table offsets = [0] sizes = [4096]
        strides = [1] : !nest.global_view<4096xi8>
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %inrel, %outready = nest.dispatch.tasks.async @gather_tile l1_mode = 1
        tasks(%tasks) globals(%table_view) bindings() ins() outs() signal_policy {{}}
        : (!nest.event<"grid_done">, !nest.event<"">, !nest.event<"">)
    nest.await %grid
    nest.return
  }}
}}
"""




class TestXDSLIR:
  def _assert_verify_failure(self, module: ModuleOp) -> None:
    with pytest.raises(VerifyException):
      verify_workload_ir(module)

  def _assert_parse_failure(self, text: str) -> None:
    with pytest.raises(ParseError):
      parse_workload_ir(text, source_name="<negative>")

  def test_pow_workload_round_trip_uses_function_calls(self):
    """PowWorkload IR round-trips in the function-call dialect."""
    text = print_workload_ir(PowWorkload().module)
    for fragment in (
      "nest.alloc",
      "nest.subview",
      "nest.dispatch.tasks.async",
      "depends_on",
      "tile.signal",
      "into",
    ):
      assert fragment in text

    reparsed = parse_workload_ir(text, source_name="<pow>")
    verify_workload_ir(reparsed)
    assert print_workload_ir(reparsed) == text

  def test_model_input_chain_round_trip_byte_stable(self):
    """The full global-input chain parses, verifies, and prints byte-stable."""
    module = parse_workload_ir(MODEL_CHAIN_IR, source_name="<chain>")
    text1 = print_workload_ir(module)
    reparsed = parse_workload_ir(text1, source_name="<chain-rt>")
    text2 = print_workload_ir(reparsed)
    assert text1 == text2
    for fragment in (
      "nest.subview",
      "tile.subview",
      "tile.alloc",
      "into",
      "!nest.global_view",
      "!nest.l2_view",
      "!tile.l1_buffer",
      "!nest.task",
      "@pow_task(%Y0)",
    ):
      assert fragment in text1

  def test_gather_round_trip_is_byte_stable(self):
    module = parse_workload_ir(GATHER_IR, source_name="<gather>")
    text = print_workload_ir(module)
    reparsed = parse_workload_ir(text, source_name="<gather-rt>")
    assert print_workload_ir(reparsed) == text
    assert "tile.gather.global.async" in text
    assert "#tile.indexed_map<" in text

  def test_gather_lowering_produces_plain_execution_dtos(self):
    task = compiled_entry(parse_workload_ir(GATHER_IR))
    binding = next(iter(task.role_bindings.values()))
    assert [formal.space for formal in binding.tile_program.formals] == ["task", "global"]
    assert len(binding.global_actuals) == 1
    assert binding.global_actuals[0].base == "global:table"
    launch = next(inst for inst in binding.tile_program.insts if inst.op == ExecTileOp.LAUNCH_GATHER)
    descriptor = binding.tile_program.descriptors[launch.args[0]]
    gather = descriptor.params["gather"]
    assert descriptor.kind == "MFE"
    assert descriptor.op == "gather"
    assert isinstance(gather, ExecTileGatherDesc)
    assert gather.source.base == "formal:1"
    assert gather.indices.base == "l1:0"
    assert gather.destination.base == "l1:1"
    assert gather.window_entries == 16
    assert gather.scope is None
    assert (
      gather.address_map.index_scale,
      gather.address_map.offset,
      gather.address_map.task_stride,
      gather.address_map.repeat,
      gather.address_map.stride,
      gather.address_map.segment,
    ) == (64, 0, 0, 1, 0, 16)

  @pytest.mark.parametrize(
    "path",
    [
      "examples/workloads/gather_indexed.mlir",
      "examples/workloads/gather_matmul.mlir",
      "examples/workloads/matmul_gather_add.mlir",
      "examples/workloads/gather_matmul_4tiles_2contexts.mlir",
      "examples/workloads/matmul_gather_add_4tiles_2contexts.mlir",
    ],
  )
  def test_gather_examples_lower_complete_output_store_path(self, path):
    # The *_2contexts fixtures pin both Device and Tile contexts at index 1.
    # Compile every case against that declared target rather than relying on
    # the single-context default.
    sim = SimConfig(context_count=2, device_context_count=2)
    model = compiled_entry(parse_workload_ir(Path(path).read_text()), sim=sim)
    assert any("output" in input_.name.lower() for input_ in model.inputs)
    for task in model.tasks.values():
      assert any(buffer.role == "out" for buffer in task.l2_buffers)
      assert any(action.op == ExecGroupActionOp.DMA_STORE for action in task.actions)
      for binding in task.role_bindings.values():
        store_descriptors = {
          name for name, descriptor in binding.tile_program.descriptors.items() if descriptor.op == "store"
        }
        assert store_descriptors
        assert any(
          inst.op == ExecTileOp.LAUNCH_MFE and inst.args and inst.args[0] in store_descriptors
          for inst in binding.tile_program.insts
        )

  def test_gather_source_requires_readable_binding(self):
    artifact = compile_program(parse_workload_ir(GATHER_IR), FAST_HW, SimConfig())
    with pytest.raises(ValueError):
      load_program(
        artifact,
        FAST_HW,
        SimConfig(),
        actual_bindings={"table": GlobalBinding("table", 0x100000, 4096, "w")},
      )

  @pytest.mark.parametrize("group", ["globals(%table_view)", "bindings()", "ins()", "outs()"])
  def test_dispatch_rejects_missing_required_group(self, group):
    self._assert_parse_failure(GATHER_IR.replace(f" {group}", "", 1))

  @pytest.mark.parametrize(
    ("old", "new"),
    [
      ("globals(%table_view)", "globals()"),
      ("%table : !nest.global_view<4096xi8>", "%table : !nest.global_view<2048xi8>"),
      (
        "%table : !nest.global_view<4096xi8>",
        "%l2 : !nest.l2_buffer<1xi8>, %table : !nest.global_view<4096xi8>",
      ),
    ],
  )
  def test_global_formal_dispatch_contract_rejects_mismatch(self, old, new):
    with pytest.raises(VerifyException):
      parse_workload_ir(GATHER_IR.replace(old, new, 1), source_name="<global-contract>")

  @pytest.mark.parametrize(
    "text",
    [
      # destination must hold exactly I * R * L elements (the buffer type
      # follows so the failure is the indexed-extent rule, not the allocator)
      GATHER_IR.replace("shape = [256] dtype", "shape = [255] dtype", 1).replace(
        "!tile.l1_buffer<256xi8>", "!tile.l1_buffer<255xi8>", 1
      ),
      GATHER_IR.replace("shape = [256] dtype", "shape = [257] dtype", 1).replace(
        "!tile.l1_buffer<256xi8>", "!tile.l1_buffer<257xi8>", 1
      ),
      # indices and destination must be different allocations
      GATHER_IR.replace("into %destination", "into %indices", 1),
      # index_scale / repeat / segment must be > 0
      GATHER_IR.replace("index_scale = 64", "index_scale = 0", 1),
      GATHER_IR.replace("repeat = 1", "repeat = 0", 1),
      GATHER_IR.replace("segment = 16", "segment = 0", 1),
      # offset / task_stride / stride must be >= 0
      GATHER_IR.replace("offset = 0", "offset = -1", 1),
      GATHER_IR.replace("task_stride = 0", "task_stride = -1", 1),
      GATHER_IR.replace("stride = 0 segment", "stride = -1 segment", 1),
      # window_entries must be > 0
      GATHER_IR.replace("window_entries = 16", "window_entries = 0", 1),
      # destination dtype must match the remote formal
      GATHER_IR.replace('shape = [256] dtype = "i8"', 'shape = [256] dtype = "i32"', 1).replace(
        "!tile.l1_buffer<256xi8>", "!tile.l1_buffer<256xi32>", 1
      ),
      # indices must be i32
      GATHER_IR.replace('shape = [16] dtype = "i32"', 'shape = [16] dtype = "i8"', 1).replace(
        "!tile.l1_buffer<16xi32>", "!tile.l1_buffer<16xi8>", 1
      ),
      # an empty scope string is not a scope
      GATHER_IR.replace(
        'window_entries = 16 : !tile.event<"gather_done">',
        'window_entries = 16 scope = "" : !tile.event<"gather_done">',
        1,
      ),
    ],
  )
  def test_gather_static_contract_rejects_invalid_ir(self, text):
    with pytest.raises(VerifyException):
      parse_workload_ir(text, source_name="<invalid-gather>")

  def test_gather_builders_are_public_operations(self):
    cache = CacheRequirement(False, "read", "allowed", 64)
    program = TileProgramDefOp(
      "public_gather",
      TileResources((1, 2), 2048, l1_cache=cache, l2_cache=cache),
      arg_types=[NestTask(), NestGlobalView.of([64], "i8")],
      arg_names=["task", "source"],
    )
    _task, source = program.body.block.args
    indices = TileAllocOp([1], "i32")
    destination = TileAllocOp([16], "i8")
    gather = TileGatherOp(
      source,
      indices.result,
      destination.result,
      "done",
      address_map=TileIndexedMapAttr.of(64, 0, 0, 1, 0, 16),
      window_entries=1,
    )
    program.body.block.add_ops([indices, destination, gather, TileAwaitOp([gather.result]), TileReturnOp()])
    verify_workload_ir(
      ModuleOp(
        [program, NestContextOp("unused_context", context_resources(), [NestReturnOp()], placement=1)]
      )
    )

  def test_function_call_op_builders_verify(self):
    """Every function-call operation can participate in a verified module."""
    l2_dims = [1, 4, 32]
    prog = TileProgramDefOp(
      "all_ops",
      tile_resources(4096),
      arg_types=[NestTask(), NestBuffer.of(l2_dims, "bf16")],
      arg_names=["task", "l2_buf"],
    )
    _task_arg, l2_arg = prog.body.block.args
    view = TileSubviewOp(l2_arg, None, None, [0, 0, 0], l2_dims, [1, 1, 1], NestL2View.of(l2_dims, "bf16"))
    l1 = TileAllocOp([4, 32], "bf16", alignment=256)
    load = TileLoadOp(view.result, l1.result, "load")
    pow_op = TilePowOp(bytes_total=256, exponent=2, pow_ops=32, tag="pow")
    evu = TileEvuOp(op_name="relu", evu_ops=16, tag="evu")
    boa = TileBoaOp(op_name="matmul", m=1, n=1, k=1, boa_ops=2, tag="boa")
    store = TileStoreOp(l1.result, view.result, "store")
    prog.body.block.add_ops(
      [
        view,
        l1,
        load,
        TileAwaitOp([load.result]),
        TileSignalOp("input_released", _task_arg),
        pow_op,
        TileAwaitOp([pow_op.result]),
        evu,
        TileAwaitOp([evu.result]),
        boa,
        TileAwaitOp([boa.result]),
        store,
        TileAwaitOp([store.result]),
        TileSignalOp("output_ready", _task_arg),
        TileReturnOp(),
      ]
    )

    ctx = NestContextOp(
      "all_ops_context",
      context_resources(1, 1024),
      placement=1,
      arg_types=[NestGlobalMemref.of(l2_dims, "bf16")],
      arg_names=["Y"],
    )
    y_arg = ctx.body.block.args[0]
    buffer = NestAllocOp(slot="l2_buf", role="inout", shape=l2_dims, dtype="bf16")
    src = NestSubviewOp(y_arg, [0, 0, 0], l2_dims, [1, 1, 1], NestGlobalView.of(l2_dims, "bf16"))
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    prefetch = NestPrefetchOp(src.result, buffer.result, "prefetch")
    dispatch = NestDispatchOp(
      "all_ops",
      tasks.result,
      [],
      [buffer.result],
      [buffer.result],
      "grid_done",
      "input_released",
      "output_ready",
      l1_mode=0,
      bindings=[buffer.result],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=[prefetch.result],
    )
    collective = NestCollectiveOp("reduce", bytes_total=256, participant_mask=1, tag="collective")
    dma_store = NestDMAStoreOp(
      src=buffer.result, dst=src.result, tag="store_done", depends_on=[dispatch.output_ready]
    )
    release = NestReleaseOp(
      buffer.result, depends_on=[dispatch.input_released, prefetch.result, dma_store.result]
    )
    ctx.body.block.add_ops(
      [
        buffer,
        src,
        prefetch,
        tasks,
        dispatch,
        collective,
        dma_store,
        release,
        NestAwaitOp([dispatch.grid_done, collective.result, dma_store.result]),
        NestBarrierOp(),
        NestReturnOp(),
      ]
    )
    module = ModuleOp([prog, ctx])

    verify_workload_ir(module)
    assert isinstance(buffer.result.type, NestBuffer)
    assert isinstance(view.result.type, NestL2View)
    assert isinstance(l1.result.type, TileL1Buffer)
    assert isinstance(src.result.type, NestGlobalView)
    assert isinstance(tasks.result.type, TaskRange)
    assert isinstance(prefetch.result.type, NestEvent)
    assert isinstance(load.result.type, TileEvent)
    assert isinstance(make_pow_task(), ModuleOp)
    assert isinstance(make_pow_tile_program(), TileProgramDefOp)
    assert isinstance(make_identity_tile_program(), TileProgramDefOp)
    assert ALL_WORKLOADS == [PowWorkload]

  def _make_identity_context(self, ctx_name="c", placement=1, context_id=None):
    """Minimal identity program with one dispatch context."""
    prog = make_identity_tile_program()
    tasks = NestTaskRangeOp(0, 1)
    dispatch = NestDispatchOp(
      prog.sym_name.data,
      tasks.result,
      [],
      [],
      [],
      "grid_done",
      "",
      "",
      l1_mode=0,
      bindings=[],
      signal_policy={},
    )
    ctx = NestContextOp(
      ctx_name,
      context_resources(logical_tasks=1),
      [tasks, dispatch, NestAwaitOp([dispatch.grid_done]), NestReturnOp()],
      placement=placement,
      context_id=context_id,
    )
    return [prog, ctx], dispatch

  def test_verifier_rejects_unknown_program_symbol(self):
    tasks = NestTaskRangeOp(0, 1)
    dispatch = NestDispatchOp(
      "missing_program",
      tasks.result,
      [],
      [],
      [],
      "grid_done",
      "",
      "",
      l1_mode=0,
      bindings=[],
      signal_policy={},
    )
    module = ModuleOp(
      [
        NestContextOp(
          "unknown_program",
          context_resources(logical_tasks=1),
          [tasks, dispatch, NestReturnOp()],
          placement=1,
        )
      ]
    )
    self._assert_verify_failure(module)

  def test_verifier_rejects_undefined_event_in_await(self):
    ctx = NestContextOp(
      "undefined_await",
      context_resources(l2_spm_bytes=32768),
      arg_types=[NestGlobalMemref.of([1, 128, 128], "bf16")],
      arg_names=["Y"],
      placement=1,
    )
    y = ctx.body.block.args[0]
    buf = NestAllocOp("buf", "in", [1, 128, 128], "bf16")
    src = NestSubviewOp(y, [0, 0, 0], [1, 128, 128], [1, 1, 1], NestGlobalView.of([1, 128, 128], "bf16"))
    later = NestPrefetchOp(src.result, buf.result, "defined_later")
    ctx.body.block.add_ops([buf, src, NestAwaitOp([later.result]), later, NestReturnOp()])
    module = ModuleOp([ctx])
    self._assert_verify_failure(module)

  def test_verifier_rejects_duplicate_event_tag(self):
    ctx = NestContextOp(
      "duplicate_event",
      context_resources(l2_spm_bytes=32768),
      arg_types=[NestGlobalMemref.of([1, 128, 128], "bf16")],
      arg_names=["Y"],
      placement=1,
    )
    y = ctx.body.block.args[0]
    buf = NestAllocOp("buf", "in", [1, 128, 128], "bf16")
    src = NestSubviewOp(y, [0, 0, 0], [1, 128, 128], [1, 1, 1], NestGlobalView.of([1, 128, 128], "bf16"))
    first = NestPrefetchOp(src.result, buf.result, "duplicate")
    second = NestPrefetchOp(src.result, buf.result, "duplicate")
    ctx.body.block.add_ops([buf, src, first, second, NestReturnOp()])
    module = ModuleOp([ctx])
    self._assert_verify_failure(module)

  def test_dispatch_context_attribute_round_trip(self):
    """A dispatch pin survives print/parse while an unpinned dispatch stays absent."""
    prog = make_identity_tile_program()
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    dispatch = NestDispatchOp(
      prog.sym_name.data,
      tasks.result,
      [],
      [],
      [],
      "grid_done",
      "",
      "",
      bindings=[],
      signal_policy={},
      context_id=1,
      l1_mode=0,
    )
    module = ModuleOp(
      [
        prog,
        NestContextOp(
          "pinned_ctx",
          context_resources(logical_tasks=1),
          [tasks, dispatch, NestAwaitOp([dispatch.grid_done]), NestReturnOp()],
          placement=1,
        ),
      ]
    )
    verify_workload_ir(module)
    reparsed = parse_workload_ir(print_workload_ir(module), source_name="<pinned>")
    verify_workload_ir(reparsed)
    reparsed_dispatches = [
      body_op
      for op in reparsed.ops
      if isinstance(op, NestContextOp)
      for body_op in op.body.blocks[0].ops
      if isinstance(body_op, NestDispatchOp)
    ]
    assert len(reparsed_dispatches) == 1
    assert reparsed_dispatches[0].context_id is not None
    assert int(reparsed_dispatches[0].context_id.value.data) == 1

    unpinned = parse_workload_ir(print_workload_ir(PowWorkload().module), source_name="<unpinned-dispatch>")
    unpinned_dispatches = [
      body_op
      for op in unpinned.ops
      if isinstance(op, NestContextOp)
      for body_op in op.body.blocks[0].ops
      if isinstance(body_op, NestDispatchOp)
    ]
    assert unpinned_dispatches
    assert all(op.context_id is None for op in unpinned_dispatches)

  def test_verifier_rejects_negative_dispatch_context(self):
    prog = make_identity_tile_program()
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    dispatch = NestDispatchOp(
      prog.sym_name.data,
      tasks.result,
      [],
      [],
      [],
      "grid_done",
      "",
      "",
      bindings=[],
      signal_policy={},
      context_id=-1,
      l1_mode=0,
    )
    module = ModuleOp(
      [
        prog,
        NestContextOp(
          "neg_ctx", context_resources(logical_tasks=1), [tasks, dispatch, NestReturnOp()], placement=1
        ),
      ]
    )
    self._assert_verify_failure(module)

  def test_context_level_context_attribute_round_trip(self):
    """A Device slot pin survives print/parse while unpinned roots stay absent."""
    module = ModuleOp(self._make_identity_context(ctx_name="ctx_default_pin", context_id=1)[0])
    verify_workload_ir(module)
    reparsed = parse_workload_ir(print_workload_ir(module), source_name="<ctx-pin>")
    verify_workload_ir(reparsed)
    ctx_op = next(op for op in reparsed.ops if isinstance(op, NestContextOp))
    assert ctx_op.context_id is not None
    assert int(ctx_op.context_id.value.data) == 1

    unpinned = parse_workload_ir(print_workload_ir(PowWorkload().module), source_name="<unpinned-context>")
    unpinned_contexts = [op for op in unpinned.ops if isinstance(op, NestContextOp)]
    assert unpinned_contexts
    assert all(op.context_id is None for op in unpinned_contexts)

  def test_verifier_rejects_negative_context_level_context(self):
    module = ModuleOp(self._make_identity_context(ctx_name="neg_ctx_pin", context_id=-1)[0])
    self._assert_verify_failure(module)

  def test_nexus_program_round_trip(self):
    """nexus.program with submit/await/return round-trips in custom assembly."""
    prog = make_identity_tile_program()
    ctxs = []
    for i in range(2):
      tasks = NestTaskRangeOp(0, 1)
      disp = NestDispatchOp(
        prog.sym_name.data,
        tasks.result,
        [],
        [],
        [],
        f"ev_grid_c{i}",
        "",
        "",
        l1_mode=0,
        bindings=[],
        signal_policy={},
      )
      ctxs.append(
        NestContextOp(
          f"ctx{i}",
          context_resources(logical_tasks=1),
          [tasks, disp, NestAwaitOp([disp.grid_done]), NestReturnOp()],
          arg_types=[NestGlobalMemref.of([4, 128, 128], "bf16")],
          arg_names=["Y"],
          placement=1,
          context_id=i,
        )
      )
    program = NexusProgramOp(
      "run_model", [], arg_types=[NestGlobalMemref.of([4, 128, 128], "bf16")] * 2, arg_names=["Y0", "Y1"]
    )
    y0, y1 = program.body.block.args
    sub0 = NexusSubmitContextOp("ctx0", "done0", actuals=[y0])
    sub1 = NexusSubmitContextOp("ctx1", "done1", actuals=[y1])
    program.body.block.add_ops([sub0, sub1, NexusAwaitOp([sub0.result, sub1.result]), NexusReturnOp()])
    module = ModuleOp([prog, *ctxs, program])
    verify_workload_ir(module)
    text = print_workload_ir(module)
    assert "nexus.program" in text
    assert "nexus.submit_context.async" in text
    assert '@nexus.event<"' not in text
    assert "@ctx0(%Y0)" in text
    assert "!nest.global_memref<4x128x128xbf16>" in text
    reparsed = parse_workload_ir(text)
    verify_workload_ir(reparsed)
    assert print_workload_ir(reparsed) == text

  def test_verifier_rejects_unknown_submit_context(self):
    """submit_context referencing an undefined nest.context is rejected."""
    ops, _ = self._make_identity_context(ctx_name="ctx0", context_id=0)
    program = NexusProgramOp("bad", [NexusSubmitContextOp("missing", "done0"), NexusReturnOp()])
    module = ModuleOp([*ops, program])
    self._assert_verify_failure(module)

  def test_verifier_rejects_undefined_nexus_await(self):
    """nexus.await referencing an event not yet submitted is rejected."""
    ops, _ = self._make_identity_context(ctx_name="ctx0", context_id=0)
    sub = NexusSubmitContextOp("ctx0", "done0")
    program = NexusProgramOp("bad", [NexusAwaitOp([sub.result]), sub, NexusReturnOp()])
    module = ModuleOp([*ops, program])
    self._assert_verify_failure(module)

  def test_submit_context_arity_mismatch_fails(self):
    text = MODEL_CHAIN_IR.replace("@pow_task(%Y0)", "@pow_task(%Y0, %Y1)", 1)
    with pytest.raises(VerifyException):
      parse_workload_ir(text, source_name="<arity>")

  def test_submit_context_type_mismatch_fails(self):
    text = MODEL_CHAIN_IR.replace(
      "%Y1 : !nest.global_memref<4x128x128xbf16>", "%Y1 : !nest.global_memref<8x128x128xbf16>", 1
    ).replace("@pow_task(%Y0)", "@pow_task(%Y1)", 1)
    with pytest.raises(VerifyException):
      parse_workload_ir(text, source_name="<type>")

  def test_nest_subview_out_of_bounds_fails(self):
    text = MODEL_CHAIN_IR.replace(
      "nest.subview %Y offsets = [0, 0, 0] sizes = [4, 128, 128]",
      "nest.subview %Y offsets = [5, 0, 0] sizes = [4, 128, 128]",
      1,
    )
    with pytest.raises(VerifyException):
      parse_workload_ir(text, source_name="<oob>")

  def test_tile_subview_task_range_overflow_fails(self):
    text = (
      MODEL_CHAIN_IR.replace("sizes = [1, 128, 128]", "sizes = [2, 128, 128]", 1)
      .replace(": !nest.l2_view<1x128x128xbf16>", ": !nest.l2_view<2x128x128xbf16>", 1)
      .replace(
        'shape = [128, 128] dtype = "bf16" alignment = 256',
        'shape = [256, 128] dtype = "bf16" alignment = 256',
        1,
      )
      .replace(": !tile.l1_buffer<128x128xbf16>", ": !tile.l1_buffer<256x128xbf16>", 1)
    )
    with pytest.raises(VerifyException):
      parse_workload_ir(text, source_name="<task-overflow>")

  def test_non_unit_stride_rejected(self):
    text = MODEL_CHAIN_IR.replace("strides = [1, 1, 1]", "strides = [1, 2, 1]", 1)
    with pytest.raises(VerifyException):
      parse_workload_ir(text, source_name="<stride>")

  def test_transfer_byte_mismatch_fails(self):
    with pytest.raises(VerifyException):
      parse_workload_ir(TRANSFER_BYTE_MISMATCH_IR, source_name="<bytes>")

  def test_tile_program_requires_task_formal(self):
    with pytest.raises(VerifyException):
      parse_workload_ir(TILE_PROGRAM_NO_TASK_IR, source_name="<no-task>")

  @pytest.mark.parametrize(
    "old_text",
    [
      '%e = tile.load.async bytes = 32768 : !tile.event<"e">',
      '%e = tile.store.async bytes = 32768 : !tile.event<"e">',
      '%ev = nest.dma.prefetch.async %buf bytes = 131072 : !nest.event<"ev">',
      '%ev = nest.dma.store.async %buf bytes = 131072 : !nest.event<"ev">',
    ],
  )
  def test_legacy_addressless_syntax_rejected(self, old_text):
    if old_text.startswith("%e"):
      text = (
        "builtin.module {\n"
        "  tile.program @p(%task : !nest.task)\n"
        "      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2], "
        "tile_l1_spm_bytes_per_context = 0> {\n"
        f"    {old_text}\n"
        "    tile.await %e\n"
        "    tile.return\n"
        "  }\n"
        "}\n"
      )
    else:
      text = (
        "builtin.module {\n"
        "  nest.context @c placement = 1\n"
        "      resource_contract = #nest.context_resources<l2_mode = 0, "
        "allowed_profiles = [0, 1, 2], logical_tasks = 0, l2_spm_bytes = 32768, "
        "requested_contexts_per_tile = 1> {\n"
        '    %buf = nest.alloc slot = "buf" role = "in" shape = [1, 128, 128]'
        ' dtype = "bf16" : !nest.l2_buffer<1x128x128xbf16>\n'
        f"    {old_text}\n"
        "    nest.await %ev\n"
        "    nest.return\n"
        "  }\n"
        "}\n"
      )
    self._assert_parse_failure(text)


class TestLoweringDTOFields:
  """PR 2 §1.4: lowering preserves backing_dims, strides, element_bytes,
  alignment and task_dim; non-contiguous subviews are rejected."""

  def test_pow_lowering_preserves_dto_fields(self):
    """Lowered PowWorkload DTOs carry backing_dims, strides, element_bytes,
    alignment and task_dim exactly as declared in the source IR."""
    task = compiled_entry(PowWorkload(hw=FAST_HW).module)
    # L2 buffer: element_bytes from dtype, alignment from nest.alloc
    l2 = task.l2_buffers[0]
    assert l2.element_bytes == 2  # bf16
    assert l2.alignment == 256
    assert l2.bytes == 4 * 128 * 128 * 2
    # prefetch transfer: src is the global subview
    prog = task.role_bindings[0].tile_program
    prefetch_action = next(action for action in task.actions if action.op is ExecGroupActionOp.DMA_PREFETCH)
    pref_src = prefetch_action.args[1].src
    assert pref_src.space == "global"
    assert pref_src.backing_dims == (16, 128, 128)  # 4 chunks * 4
    assert pref_src.dims == (4, 128, 128)
    assert pref_src.strides == (1, 1, 1)
    assert pref_src.element_bytes == 2
    assert pref_src.task_dim is None
    # tile load transfer: src is the tile.subview (l2), dst is l1
    load_desc = next(iter(prog.descriptors.values()))
    assert load_desc.op == "load"
    tile_src = load_desc.transfer.src
    assert tile_src.space == "l2"
    assert tile_src.backing_dims == (4, 128, 128)
    assert tile_src.dims == (1, 128, 128)
    assert tile_src.strides == (1, 1, 1)
    assert tile_src.element_bytes == 2
    assert tile_src.task_dim == 0
    # L1 buffer: element_bytes + alignment from tile.alloc
    l1 = prog.l1_buffers[0]
    assert l1.element_bytes == 2
    assert l1.alignment == 256

  @staticmethod
  def _make_subview_module(
    g_sv_sizes: list[int], l2_formal_sizes: list[int], tile_sv_sizes: list[int], l1_sizes: list[int]
  ) -> ModuleOp:
    """Build a module with explicit global + tile subviews.

    ``g_sv_sizes``: nest.subview sizes on a [4,128,128] global formal; the
    L2 buffer matches these (prefetch byte equality).
    ``l2_formal_sizes``: the tile.program L2 formal + dispatch actual shape.
    ``tile_sv_sizes``: the tile.subview slice of the L2 formal.
    ``l1_sizes``: the tile.alloc shape (must equal tile_sv_sizes bytes).
    All bf16.  The L2 formal and dispatch actuals use ``l2_formal_sizes``;
    the prefetch copies ``g_sv_sizes`` bytes into an L2 buffer of the same
    shape, so ``g_sv_sizes`` must equal ``l2_formal_sizes`` for the
    prefetch to verify.
    """
    g_dims = [4, 128, 128]
    l1_bytes = authored_arena_bytes("l1", [(2 * prod(l1_sizes), 256)])
    prog = TileProgramDefOp(
      "sv_prog",
      tile_resources(l1_bytes),
      arg_types=[NestTask(), NestBuffer.of(l2_formal_sizes, "bf16")],
      arg_names=["task", "l2_buf"],
    )
    _task_arg, l2_arg = prog.body.block.args
    l2_view = TileSubviewOp(
      l2_arg,
      _task_arg,
      0,
      [0] * len(tile_sv_sizes),
      tile_sv_sizes,
      [1] * len(tile_sv_sizes),
      NestL2View.of(tile_sv_sizes, "bf16"),
    )
    l1 = TileAllocOp(l1_sizes, "bf16", alignment=256)
    load = TileLoadOp(l2_view.result, l1.result, "e_load")
    prog.body.block.add_ops(
      [
        l2_view,
        l1,
        load,
        TileAwaitOp([load.result]),
        TileSignalOp("input_released", _task_arg),
        TileReturnOp(),
      ]
    )
    l2_bytes = authored_arena_bytes("l2", [(2 * prod(l2_formal_sizes), 256)])
    ctx = NestContextOp(
      "sv_ctx",
      context_resources(1, l2_bytes),
      arg_types=[NestGlobalMemref.of(g_dims, "bf16")],
      arg_names=["Y"],
      placement=1,
    )
    y_arg = ctx.body.block.args[0]
    buf = NestAllocOp("l2_buf", "in", l2_formal_sizes, "bf16", alignment=256)
    src = NestSubviewOp(
      y_arg, [0] * len(g_sv_sizes), g_sv_sizes, [1] * len(g_sv_sizes), NestGlobalView.of(g_sv_sizes, "bf16")
    )
    pref = NestPrefetchOp(src.result, buf.result, "ev_in")
    tasks = NestTaskRangeOp(0, 1)
    disp = NestDispatchOp(
      "sv_prog",
      tasks.result,
      [],
      [buf.result],
      [],
      "ev_grid",
      "ev_inrel",
      "",
      l1_mode=0,
      bindings=[buf.result],
      signal_policy={"input_released": "all_tasks"},
      depends_on=[pref.result],
    )
    ctx.body.block.add_ops(
      [
        buf,
        src,
        pref,
        tasks,
        disp,
        NestReleaseOp(buf.result, depends_on=[disp.input_released, pref.result]),
        NestAwaitOp([disp.grid_done]),
        NestReturnOp(),
      ]
    )
    return ModuleOp([prog, ctx])

  def test_nest_subview_non_contiguous_rejected(self):
    """A non-contiguous row-major nest.subview is rejected.

    sizes = [2, 64, 128] on a [4, 128, 128] backing: dim 0 size 2 > 1 but
    dim 1 size 64 != backing 128 → non-contiguous.
    """
    module = self._make_subview_module(
      g_sv_sizes=[2, 64, 128], l2_formal_sizes=[2, 64, 128], tile_sv_sizes=[1, 64, 128], l1_sizes=[64, 128]
    )
    with pytest.raises(VerifyException):
      verify_workload_ir(module)

  def test_tile_subview_non_contiguous_rejected(self):
    """A non-contiguous tile.subview is rejected.

    tile.subview sizes = [2, 64, 128] on a [4, 128, 128] L2 formal: dim 0
    size 2 > 1 but dim 1 size 64 != 128 → non-contiguous.
    """
    module = self._make_subview_module(
      g_sv_sizes=[4, 128, 128],
      l2_formal_sizes=[4, 128, 128],
      tile_sv_sizes=[2, 64, 128],
      l1_sizes=[64, 128],
    )
    with pytest.raises(VerifyException):
      verify_workload_ir(module)

  def test_contiguous_trailing_full_subview_accepted(self):
    """A contiguous subview where the leading dim is sliced but trailing
    dims are full is accepted (sizes = [2, 128, 128] on [4,128,128])."""
    module = self._make_subview_module(
      g_sv_sizes=[2, 128, 128],
      l2_formal_sizes=[2, 128, 128],
      tile_sv_sizes=[1, 128, 128],
      l1_sizes=[128, 128],
    )
    verify_workload_ir(module)
    # lowering must also succeed and preserve the backing shape
    task = compiled_entry(module)
    pref = next(action for action in task.actions if action.op is ExecGroupActionOp.DMA_PREFETCH)
    assert pref.args[1].src.backing_dims == (4, 128, 128)
    assert pref.args[1].src.dims == (2, 128, 128)


class TestExternalIRCLI:
  def _run_cli(self, *args: str):
    return subprocess.run(
      [sys.executable, "-m", "pipeline_validator", *args],
      cwd=Path(__file__).resolve().parents[2],
      capture_output=True,
      text=True,
    )

  def test_ir_file_success_and_print_only_mode(self, tmp_path: Path):
    input_path = tmp_path / "pow.mlir"
    trace_json = tmp_path / "trace.json"
    trace_html = tmp_path / "trace.html"
    report_path = tmp_path / "report.txt"
    input_path.write_text(print_workload_ir(PowWorkload().module), encoding="utf-8")

    run = self._run_cli(
      "--ir-file",
      str(input_path),
      "--input-binding",
      "Y=0x100000:524288:rw",
      "--hw-override",
      "hbm_fixed_latency_cycles=10",
      "--trace-json",
      str(trace_json),
      "--trace-html",
      str(trace_html),
      "--report",
      str(report_path),
    )
    assert run.returncode == 0, run.stderr
    data = json.loads(trace_json.read_text(encoding="utf-8"))
    assert data["traceEvents"]
    assert "traceEvents" in trace_html.read_text(encoding="utf-8")
    report = report_path.read_text(encoding="utf-8")
    assert "[PASS] task_completed" in report
    assert "[PASS] credit_invariant" in report

    trace_json.unlink()
    print_only = self._run_cli("--ir-file", str(input_path), "--print-ir", "--trace-json", str(trace_json))
    assert print_only.returncode == 0, print_only.stderr
    assert print_only.stdout == input_path.read_text(encoding="utf-8")
    assert not trace_json.exists()

  def test_memory_trace_flag_gates_trace_and_report(self, tmp_path: Path):
    input_path = tmp_path / "pow.mlir"
    input_path.write_text(print_workload_ir(PowWorkload().module), encoding="utf-8")
    off_trace = tmp_path / "off_trace.json"
    off_report = tmp_path / "off_report.json"
    on_trace = tmp_path / "on_trace.json"
    on_report = tmp_path / "on_report.json"
    base_args = (
      "--ir-file",
      str(input_path),
      "--input-binding",
      "Y=0x100000:524288:rw",
      "--hw-override",
      "hbm_fixed_latency_cycles=10",
      "--trace-json",
      str(off_trace),
      "--json",
      "--report",
      str(off_report),
    )
    off = self._run_cli(*base_args)
    assert off.returncode == 0, off.stderr
    off_events = json.loads(off_trace.read_text(encoding="utf-8"))["traceEvents"]
    off_names = {e.get("name") for e in off_events}
    mem_names = {
      "l2_allocated_bytes",
      "hbm_outstanding",
      "hbm_bind",
      "l1_alloc",
      "l2_alloc",
      "noc_occupancy",
    }
    assert not (off_names & mem_names), sorted(off_names & mem_names)
    assert not [e for e in off_events if e.get("ph") in ("s", "t", "f")]
    off_report_data = json.loads(off_report.read_text(encoding="utf-8"))
    assert off_report_data[0]["memory"] == {}

    on = self._run_cli(
      "--ir-file",
      str(input_path),
      "--input-binding",
      "Y=0x100000:524288:rw",
      "--hw-override",
      "hbm_fixed_latency_cycles=10",
      "--memory-trace",
      "--trace-json",
      str(on_trace),
      "--json",
      "--report",
      str(on_report),
    )
    assert on.returncode == 0, on.stderr
    on_events = json.loads(on_trace.read_text(encoding="utf-8"))["traceEvents"]
    on_names = {e.get("name") for e in on_events}
    assert on_names & mem_names
    assert any(e.get("ph") in ("s", "t", "f") for e in on_events)
    on_report_data = json.loads(on_report.read_text(encoding="utf-8"))
    assert on_report_data[0]["memory"]["l2_peak_allocated_bytes"] > 0

  def test_ir_file_conflicts_and_load_errors(self, tmp_path: Path):
    input_path = tmp_path / "pow.mlir"
    input_path.write_text(print_workload_ir(PowWorkload().module), encoding="utf-8")

    conflict_artifact = tmp_path / "conflict.json"
    conflict = self._run_cli(
      "--ir-file", str(input_path), "-w", "pow", "--compiled-output", str(conflict_artifact)
    )
    assert conflict.returncode == 2
    assert conflict.stdout == ""
    assert not conflict_artifact.exists()

    def assert_source_rejected(path: Path, stem: str) -> None:
      artifact = tmp_path / f"{stem}.compiled.json"
      trace = tmp_path / f"{stem}.trace.json"
      report = tmp_path / f"{stem}.report.txt"
      result = self._run_cli(
        "--ir-file",
        str(path),
        "--compiled-output",
        str(artifact),
        "--trace-json",
        str(trace),
        "--report",
        str(report),
      )
      assert result.returncode == 2
      assert result.stdout == ""
      assert not artifact.exists()
      assert not trace.exists()
      assert not report.exists()

    assert_source_rejected(tmp_path / "missing.mlir", "missing")

    bad_utf8 = tmp_path / "bad_utf8.mlir"
    bad_utf8.write_bytes(b"\xff\xfe")
    assert_source_rejected(bad_utf8, "bad_utf8")

    unknown_op = tmp_path / "unknown_op.mlir"
    unknown_op.write_text(
      input_path.read_text(encoding="utf-8").replace("nest.dma.prefetch", "nest.bad_op", 1),
      encoding="utf-8",
    )
    assert_source_rejected(unknown_op, "unknown_op")

    malformed = tmp_path / "malformed.mlir"
    malformed.write_text("not mlir\n", encoding="utf-8")
    assert_source_rejected(malformed, "malformed")

  def test_context_mode_cli_bounds(self):
    ok = self._run_cli(
      "-w",
      "pow",
      "--context-mode",
      "3",
      "--max-cycles",
      "200000",
      "--hw-override",
      "hbm_fixed_latency_cycles=10",
      "--input-binding",
      "Y=0x100000:524288:rw",
    )
    assert ok.returncode == 0, ok.stderr
    for bad in ("0", "9"):
      res = self._run_cli("-w", "pow", "--context-mode", bad)
      assert res.returncode == 2
      assert res.stdout == ""

  def test_example_mlir_runs_on_two_device_contexts(self):
    res = self._run_cli(
      "--ir-file",
      "examples/workloads/pow_dual_context.mlir",
      "--device-context-mode",
      "2",
      "--context-mode",
      "2",
      "--input-binding",
      "Y0=0x100000:131072:rw",
      "--input-binding",
      "Y1=0x200000:131072:rw",
      "--hw-override",
      "hbm_fixed_latency_cycles=10",
      "--max-cycles",
      "200000",
    )
    assert res.returncode == 0, res.stderr
    assert "Completed:" in res.stdout and "True" in res.stdout

  def test_missing_input_binding_exits_2_without_execution(self, tmp_path: Path):
    artifact = tmp_path / "pow_dual_context.json"
    compile_result = self._run_cli(
      "--ir-file",
      "examples/workloads/pow_dual_context.mlir",
      "--context-mode",
      "2",
      "--device-context-mode",
      "2",
      "--compile-only",
      "--compiled-output",
      str(artifact),
    )
    assert compile_result.returncode == 0, compile_result.stderr
    assert artifact.exists()
    artifact_before = artifact.read_bytes()

    trace = tmp_path / "missing_binding.trace.json"
    report = tmp_path / "missing_binding.report.txt"
    res = self._run_cli(
      "--compiled-file",
      str(artifact),
      "--context-mode",
      "2",
      "--device-context-mode",
      "2",
      "--trace-json",
      str(trace),
      "--report",
      str(report),
    )
    assert res.returncode == 2
    assert res.stdout == ""
    assert artifact.read_bytes() == artifact_before
    assert not trace.exists()
    assert not report.exists()

  def test_device_context_mode_cli_bounds(self):
    for bad in ("0", "9"):
      res = self._run_cli("-w", "pow", "--device-context-mode", bad)
      assert res.returncode == 2
      assert res.stdout == ""


# ---------------------------------------------------------------------------
# Stream Queue unit tests
# ---------------------------------------------------------------------------


def make_queue(depth=3, producers=(0,), consumers=(1,), **kw) -> StreamQueue:
  q = StreamQueue(
    queue_id=0, depth=depth, producers=frozenset(producers), consumers=frozenset(consumers), **kw
  )
  q.init()
  return q


class TestStreamQueue:
  def test_credit_invariant_initial(self):
    q = make_queue()
    assert q.credit_invariant_holds()
    assert q._credit_available == 3

  def test_acquire_and_push(self):
    q = make_queue()
    assert q.acquire(0) is True
    tok = StreamToken(token_id=0, producer_id=0)
    assert q.push(tok, 1) is True
    assert q.occupancy == 1
    assert q.credit_invariant_holds()

  def test_full_backpressure(self):
    q = make_queue(depth=2)
    # fill both credits
    assert q.acquire(0)
    q.push(StreamToken(token_id=0, producer_id=0), 1)
    assert q.acquire(2)
    q.push(StreamToken(token_id=1, producer_id=0), 3)
    # third acquire must fail (backpressure)
    assert q.acquire(4) is False
    assert q.is_full

  def test_pop_release(self):
    q = make_queue()
    q.acquire(0)
    q.push(StreamToken(token_id=0, producer_id=0), 1)
    tok = q.pop(2)
    assert tok is not None
    assert tok.token_id == 0
    q.release(tok, 3)
    # credit returned
    assert q._credit_available == q.depth
    assert q.credit_invariant_holds()

  def test_empty_consumer_stall(self):
    q = make_queue()
    assert q.is_empty
    tok = q.pop(0)
    assert tok is None
    # PMU recorded stall
    assert q.pmu.stall_cycles.get(0, 0) > 0 or q.pmu.named_cycles.get("queue_empty", 0) > 0

  def test_eos_single_producer(self):
    q = make_queue(depth=2, producers=(0,), consumers=(1,), eos_policy=EOSPolicy.SINGLE_PRODUCER)
    q.push_eos(0, 0)
    assert q.all_eos_seen

  def test_eos_all_producers(self):
    q = make_queue(depth=4, producers=(0, 1), consumers=(2, 3), eos_policy=EOSPolicy.ALL_PRODUCERS)
    q.push_eos(0, 0)
    assert not q.all_eos_seen  # only one of two producers
    q.push_eos(1, 1)
    assert q.all_eos_seen

  def test_sequence_id_monotonic(self):
    q = make_queue(depth=4)
    q.acquire(0)
    q.push(StreamToken(token_id=0, producer_id=0), 1)
    q.acquire(2)
    q.push(StreamToken(token_id=1, producer_id=0), 3)
    t0 = q.pop(4)
    t1 = q.pop(5)
    assert t1.sequence_id > t0.sequence_id

  def test_reset_reconciles_credit(self):
    q = make_queue()
    q.acquire(0)
    q.push(StreamToken(token_id=0, producer_id=0), 1)
    q.pop(2)
    # popped but not released -> credit invariant still holds (popped_unreleased counts)
    assert q.credit_invariant_holds()
    q.reset()
    assert q._credit_available == q.depth
    assert q.occupancy == 0
    assert q.credit_invariant_holds()

  def test_push_eos_enqueues_single_token_and_drains(self):
    # A single push_eos() must create exactly one FIFO token that drains
    # after pop+release, leaving occupancy 0 and the credit invariant intact.
    q = make_queue(depth=1, producers=(0,), consumers=(1,))
    q.push_eos(0, cycle=0)
    assert q.occupancy == 1
    tok = q.pop(cycle=1)
    assert tok is not None

    q.release(tok, cycle=2)
    assert q.occupancy == 0
    assert q.credit_invariant_holds()


# ---------------------------------------------------------------------------
# Pow simulation tests
# ---------------------------------------------------------------------------


class TestPowSimulation:
  def test_pow_completes(self):
    hw = FAST_HW
    sim = Simulator(hw, SimConfig(max_cycles=200_000))
    wl = PowWorkload(hw=hw)
    result = run_source(sim, wl.module, POW_BINDINGS, workload_info=wl.info)
    assert result.completed, f"pow did not complete: {result.reason}"
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_pow_report_has_passing_checks(self):
    wl = PowWorkload(hw=FAST_HW)
    sim = Simulator(FAST_HW, SimConfig(max_cycles=200_000))
    result = run_source(sim, wl.module, POW_BINDINGS, workload_info=wl.info)
    assert result.completed, result.reason
    rep = build_report(wl.info, result)
    failed = [c for c in rep.checks if not c["pass"]]
    assert not failed, f"pow failed checks: {failed}"

  def test_pow_report_text_renderable(self):
    wl = PowWorkload(hw=FAST_HW)
    sim = Simulator(FAST_HW, SimConfig(max_cycles=200_000))
    result = run_source(sim, wl.module, POW_BINDINGS, workload_info=wl.info)
    rep = build_report(wl.info, result)
    text = report_to_text(rep)
    assert "Workload: pow" in text
    assert "Checks:" in text


# ---------------------------------------------------------------------------
# HardwareConfig grouped YAML tests
# ---------------------------------------------------------------------------


class TestHardwareConfigYaml:
  """Grouped YAML maps losslessly to the flat HardwareConfig API."""

  def _write_yaml(self, tmp_path, text: str) -> Path:
    path = tmp_path / "hw.yaml"
    path.write_text(text, encoding="utf-8")
    return path

  @pytest.mark.parametrize(
    "override",
    [
      {"cache_line_bytes": 0},
      {"cache_line_bytes": 48},
      {"l2_cache_lookup_latency_cycles": 0},
      {"l1_cache_lookup_latency_cycles": 0},
      {"l2_mshr_entries": 0},
      {"l1_mshr_entries": 0},
    ],
  )
  def test_cache_config_rejects_invalid_values(self, override):
    with pytest.raises(ValueError):
      HardwareConfig(**override)

  def test_from_yaml_partial_nested_override(self, tmp_path):
    path = self._write_yaml(
      tmp_path,
      "system:\n"
      "  clock:\n"
      "    core_mhz: 2000.0\n"
      "  topology:\n"
      "    tiles_per_group: 8\n"
      "engines:\n"
      "  boa:\n"
      "    opa:\n"
      "      count: 8\n",
    )
    cfg = HardwareConfig.from_yaml(path)
    assert cfg.clock_mhz == 2000.0
    assert cfg.num_tiles == 8
    assert cfg.boa_num_opa == 8
    assert cfg.group_sram_bytes == 8 * 1024 * 1024
    assert cfg.with_overrides(clock_mhz=1500.0).clock_mhz == 1500.0

  def test_from_yaml_rejects_unknown_nested_path(self, tmp_path):
    path = self._write_yaml(tmp_path, "engines:\n  boa:\n    unknown_lanes: 4\n")
    with pytest.raises(ValueError):
      HardwareConfig.from_yaml(path)

  def test_from_yaml_rejects_scalar_group(self, tmp_path):
    path = self._write_yaml(tmp_path, "engines:\n  boa: 4\n")
    with pytest.raises(ValueError):
      HardwareConfig.from_yaml(path)

  def test_from_yaml_rejects_mapping_leaf(self, tmp_path):
    path = self._write_yaml(tmp_path, "engines:\n  boa:\n    launch_cycles:\n      value: 4\n")
    with pytest.raises(ValueError):
      HardwareConfig.from_yaml(path)

  def test_from_yaml_rejects_sequence(self, tmp_path):
    path = self._write_yaml(tmp_path, "engines:\n  boa:\n    launch_cycles: [4]\n")
    with pytest.raises(ValueError):
      HardwareConfig.from_yaml(path)

  def test_from_yaml_rejects_unsupported_schema_version(self, tmp_path):
    path = self._write_yaml(tmp_path, "schema_version: 1\n")
    with pytest.raises(ValueError):
      HardwareConfig.from_yaml(path)

  def test_from_yaml_rejects_duplicate_key(self, tmp_path):
    path = self._write_yaml(tmp_path, "engines:\n  boa:\n    launch_cycles: 4\n    launch_cycles: 8\n")
    with pytest.raises(ValueError):
      HardwareConfig.from_yaml(path)

  def test_required_yaml_rejects_missing_leaf_before_class_defaults(self, tmp_path):
    path = self._write_yaml(tmp_path, "schema_version: 2\nsystem:\n  profile: balanced-small\n")
    with pytest.raises(ValueError):
      _load_hw_yaml(path, required=True)


# ---------------------------------------------------------------------------
# MFE channelization (engines.py)
# ---------------------------------------------------------------------------


class TestMFEChannels:
  """MFE = N load lanes + M store lanes (design/elenor_mfe 3.1.4).

  PR 2: tile load/store go through the shared ``TransferManager`` as
  ``MemoryTransaction``s.  These tests submit explicit timing
  transactions (src/dst=None) and verify lane count, queue depth and
  parallelism by stepping the manager + engine tick.
  """

  @staticmethod
  def _desc(op: str, name: str) -> ExecEngineDesc:
    return ExecEngineDesc(name=name, kind="MFE", op=op, params={"bytes": 4096})

  @staticmethod
  def _timing_txn(op: str, txn_id: str, tile_id: int = 0):
    from pipeline_validator.memory.allocator import TaskBufferOwner
    from pipeline_validator.memory.transfer import MemoryTransaction, TransferOp

    return MemoryTransaction(
      transaction_id=txn_id,
      op=TransferOp.TILE_LOAD if op == "load" else TransferOp.TILE_STORE,
      issuer=TaskBufferOwner("ctx", 0, "ev", 0, tile_id, 0, "task"),
      src=None,
      dst=None,
      bytes_total=4096,
      completion_event=txn_id,
      tile_id=tile_id,
    )

  @staticmethod
  def _make_eng(cfg=None, tile_id=0):
    from pipeline_validator.memory.transfer import TransferManager

    cfg = cfg or HardwareConfig()
    tm = TransferManager(cfg)
    return MFEEngine(cfg, tile_id, transfer_manager=tm), tm

  @staticmethod
  def _drain(eng, tm, start_cycle=10, max_cycles=2000):
    """Step tm+eng until all lanes drain; return completed EngineJobs."""
    completed = []
    for c in range(start_cycle, start_cycle + max_cycles):
      tm.step(c)
      for job in eng.tick(c):
        completed.append(job)
      if eng.state == EngineState.IDLE:
        break
    return completed

  def test_config_rejects_zero_load_channels(self):
    with pytest.raises(ValueError):
      HardwareConfig(mfe_load_channels=0)

  def test_config_rejects_zero_store_channels(self):
    with pytest.raises(ValueError):
      HardwareConfig(mfe_store_channels=0)

  def test_two_load_channels_run_in_parallel(self):
    """Two load channels: both jobs submit and complete."""
    eng, tm = self._make_eng(HardwareConfig(mfe_load_channels=2))
    assert (
      eng.launch(self._desc("load", "ld0"), 10, "e0", transaction=self._timing_txn("load", "t0"))
      is not None
    )
    assert (
      eng.launch(self._desc("load", "ld1"), 10, "e1", transaction=self._timing_txn("load", "t1"))
      is not None
    )
    completed = self._drain(eng, tm)
    assert len(completed) == 2

  def test_single_load_channel_chains_serially(self):
    """One load channel: two jobs chain serially (second starts after
    first completes)."""
    eng, tm = self._make_eng()  # V1 baseline: 1 load channel
    assert (
      eng.launch(self._desc("load", "ld0"), 10, "e0", transaction=self._timing_txn("load", "t0"))
      is not None
    )
    assert (
      eng.launch(self._desc("load", "ld1"), 10, "e1", transaction=self._timing_txn("load", "t1"))
      is not None
    )
    completed = self._drain(eng, tm)
    assert len(completed) == 2
    # serial: second completes after first
    assert completed[1].finish_cycle >= completed[0].finish_cycle

  def test_load_and_store_are_independent_lanes(self):
    """Default 1/1: load and store run in parallel on separate lanes."""
    eng, tm = self._make_eng()
    assert (
      eng.launch(self._desc("load", "ld"), 10, "e_ld", transaction=self._timing_txn("load", "t_ld"))
      is not None
    )
    assert (
      eng.launch(self._desc("store", "st"), 10, "e_st", transaction=self._timing_txn("store", "t_st"))
      is not None
    )
    completed = self._drain(eng, tm)
    assert len(completed) == 2

  def test_full_lane_returns_none_for_backpressure(self):
    eng, _ = self._make_eng(HardwareConfig(mfe_load_channels=1, mfe_pipeline_depth=1))
    assert (
      eng.launch(self._desc("load", "ld0"), 10, "e0", transaction=self._timing_txn("load", "t0"))
      is not None
    )
    assert (
      eng.launch(self._desc("load", "ld1"), 10, "e1", transaction=self._timing_txn("load", "t1")) is None
    )

  def test_reset_freeze_does_not_start_queued_lane_job(self):
    """A reset drain retires the running transfer without submitting its queue."""
    from pipeline_validator.memory.allocator import MemoryInvariantError

    cfg = HardwareConfig(mfe_load_channels=1, mfe_pipeline_depth=2)
    eng, tm = self._make_eng(cfg)
    first_txn = self._timing_txn("load", "t0")
    queued_txn = self._timing_txn("load", "t1")
    assert eng.launch(self._desc("load", "ld0"), 10, "e0", transaction=first_txn) is not None
    assert eng.launch(self._desc("load", "ld1"), 10, "e1", transaction=queued_txn) is not None

    # Drive exactly the accepted transfer's completion window.  The reset
    # freeze prevents the queued lane entry from becoming a TransferManager
    # transaction when its predecessor retires.
    tm.step(10)
    assert eng.tick(10, start_queued=False) == []
    completion_cycle = first_txn.leg_completion_cycle
    assert completion_cycle > 10
    tm.step(completion_cycle)
    completed = eng.tick(completion_cycle, start_queued=False)
    assert len(completed) == 1

    lane = eng._load_lanes[0]
    assert lane.running is None
    assert len(lane.queue) == 1
    assert not lane.queue[0].transaction_submitted
    with pytest.raises(MemoryInvariantError):
      tm.status("t1")
    assert tm.pmu_issued_count == 1
    assert tm.cancel_all(cycle=completion_cycle) is True
    eng.reset()
    assert tm.inflight_count == 0
    assert lane.running is None
    assert len(lane.queue) == 0


class TestHardwareConfigCLI:
  """--hw-config 端到端 (复用 TestExternalIRCLI 的 subprocess 模式)。"""

  def _run_cli(self, *args: str):
    return subprocess.run(
      [sys.executable, "-m", "pipeline_validator", *args],
      cwd=Path(__file__).resolve().parents[2],
      capture_output=True,
      text=True,
    )

  def test_hw_config_file_runs(self, tmp_path):
    path = tmp_path / "hw.yaml"
    path.write_text(
      "schema_version: 2\n"
      "memory:\n"
      "  target:\n"
      "    l2:\n"
      "      system_reserved_spm_per_bank: 4096\n"
      "      reset_mode: 0\n"
      "      spm_mapping_id: striped_arena_v0\n"
      "      cache_org_id: profiled_lru_v0\n"
      "      cache_write_policy: read_only\n"
      "      maintenance_caps: [invalidate_range, clean_invalidate_all, bypass]\n"
      "  profile_source:\n"
      "    l2:\n"
      "      modes:\n"
      "        0: {spm_bytes_per_bank: 1048576, cache_bytes_per_bank: 0}\n"
      "        1: {spm_bytes_per_bank: 917504, cache_bytes_per_bank: 131072}\n"
      "        2: {spm_bytes_per_bank: 786432, cache_bytes_per_bank: 262144}\n"
      "  group_sram:\n"
      "    capacity_bytes: 16777216\n"
      "    banks: 16\n",
      encoding="utf-8",
    )
    result = self._run_cli(
      "-w",
      "pow",
      "--hw-config",
      str(path),
      "--input-binding",
      "Y=0x100000:524288:rw",
      "--max-cycles",
      "200000",
    )
    assert result.returncode == 0, result.stderr

  def test_hw_config_missing_file(self, tmp_path):
    artifact = tmp_path / "missing-hw.json"
    result = self._run_cli(
      "-w", "pow", "--hw-config", str(tmp_path / "nope.yaml"), "--compiled-output", str(artifact)
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert not artifact.exists()


class TestPR3SignalPolicy:
  """Signal policies follow the Tile Program's real L2 accesses."""

  @staticmethod
  def _make_signal_prog(phases: tuple[str, ...]) -> TileProgramDefOp:
    prog = TileProgramDefOp(
      "sig_prog",
      tile_resources(1024),
      arg_types=[NestTask(), NestBuffer.of([1, 4, 32], "bf16")],
      arg_names=["task", "l2_buf"],
    )
    task_arg, l2_arg = prog.body.block.args
    view = TileSubviewOp(
      l2_arg, task_arg, 0, [0, 0, 0], [1, 4, 32], [1, 1, 1], NestL2View.of([1, 4, 32], "bf16")
    )
    l1 = TileAllocOp([4, 32], "bf16")
    ops = [view, l1]
    for phase in phases:
      if phase == "input_released":
        load = TileLoadOp(view.result, l1.result, "e_load")
        ops.extend([load, TileAwaitOp([load.result])])
      else:
        store = TileStoreOp(l1.result, view.result, "e_store")
        ops.extend([store, TileAwaitOp([store.result])])
      ops.append(TileSignalOp(phase, task_arg))
    ops.append(TileReturnOp())
    prog.body.block.add_ops(ops)
    return prog

  @staticmethod
  def _make_context(prog, role, inrel_tag, outready_tag, policy):
    reads = any(isinstance(op, TileLoadOp) for op in prog.body.block.ops)
    writes = any(isinstance(op, TileStoreOp) for op in prog.body.block.ops)
    ctx = NestContextOp(
      "sig_ctx",
      context_resources(1, 4096),
      arg_types=[NestGlobalMemref.of([1, 4, 32], "bf16")],
      arg_names=["Y"],
      placement=1,
    )
    y_arg = ctx.body.block.args[0]
    buf = NestAllocOp("l2_buf", role, [1, 4, 32], "bf16", alignment=256)
    src = NestSubviewOp(y_arg, [0, 0, 0], [1, 4, 32], [1, 1, 1], NestGlobalView.of([1, 4, 32], "bf16"))
    ops = [buf, src]
    pref = NestPrefetchOp(src.result, buf.result, "ev_in") if reads else None
    if pref is not None:
      ops.append(pref)
    tasks = NestTaskRangeOp(0, 1)
    disp = NestDispatchOp(
      "sig_prog",
      tasks.result,
      [],
      [buf.result] if reads else [],
      [buf.result] if writes else [],
      "ev_grid",
      inrel_tag,
      outready_tag,
      l1_mode=0,
      bindings=[buf.result],
      signal_policy=policy,
      depends_on=[pref.result] if pref is not None else [],
    )
    ops.extend([tasks, disp])
    store = None
    if writes:
      store = NestDMAStoreOp(buf.result, src.result, "ev_out", depends_on=[disp.output_ready])
      ops.append(store)
    release_deps = []
    if reads:
      release_deps.append(disp.input_released)
    if pref is not None:
      release_deps.append(pref.result)
    if store is not None:
      release_deps.append(store.result)
    ops.extend(
      [
        NestReleaseOp(buf.result, depends_on=release_deps),
        NestAwaitOp([disp.grid_done] + ([store.result] if store is not None else [])),
        NestReturnOp(),
      ]
    )
    ctx.body.block.add_ops(ops)
    return ctx

  def test_signal_policy_round_trip(self):
    """Empty, read-only, write-only, and read-write policies round-trip."""
    cases = [
      ((), None),
      (("input_released",), "in"),
      (("output_ready",), "out"),
      (("input_released", "output_ready"), "inout"),
    ]
    for phases, role in cases:
      if not phases:
        prog = make_identity_tile_program()
        tasks = NestTaskRangeOp(0, 1)
        disp = NestDispatchOp(
          prog.sym_name.data,
          tasks.result,
          [],
          [],
          [],
          "ev_grid",
          "",
          "",
          l1_mode=0,
          bindings=[],
          signal_policy={},
        )
        ctx = NestContextOp(
          "sig_ctx",
          context_resources(logical_tasks=1),
          [tasks, disp, NestAwaitOp([disp.grid_done]), NestReturnOp()],
          placement=1,
        )
      else:
        prog = self._make_signal_prog(phases)
        policy = dict.fromkeys(phases, "all_tasks")
        ctx = self._make_context(
          prog,
          role,
          "ev_i" if "input_released" in phases else "",
          "ev_o" if "output_ready" in phases else "",
          policy,
        )
      module = ModuleOp([prog, ctx])
      verify_workload_ir(module)
      text = print_workload_ir(module)
      reparsed = parse_workload_ir(text, source_name="<rt>")
      assert print_workload_ir(reparsed) == text

  def test_legacy_signal_syntax_rejected(self):
    text = """builtin.module {
  tile.program @p (%task : !nest.task, %l2 : !nest.l2_buffer<1x4x32xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 0> {
    tile.signal input_released
    tile.return
  }
}
"""
    with pytest.raises(ParseError):
      parse_workload_ir(text, source_name="<legacy>")

  def test_signal_requires_program_task_formal(self):
    prog = TileProgramDefOp(
      "bad_sig",
      tile_resources(),
      arg_types=[NestTask(), NestBuffer.of([1, 4, 32], "bf16")],
      arg_names=["task", "l2_buf"],
    )
    _task_arg, l2_arg = prog.body.block.args
    prog.body.block.add_ops([TileSignalOp("input_released", l2_arg), TileReturnOp()])
    with pytest.raises(VerifyException):
      verify_workload_ir(ModuleOp([prog]))

  def test_signal_policy_matches_program_phases(self):
    prog = self._make_signal_prog(("input_released",))
    ctx = self._make_context(
      prog, "in", "ev_i", "ev_o", {"input_released": "all_tasks", "output_ready": "all_tasks"}
    )
    with pytest.raises(VerifyException):
      verify_workload_ir(ModuleOp([prog, ctx]))

  @pytest.mark.parametrize(
    ("role", "phases"),
    [
      ("in", ("input_released",)),
      ("out", ("output_ready",)),
      ("inout", ("input_released", "output_ready")),
    ],
  )
  def test_access_based_release_chains_verify(self, role, phases):
    prog = self._make_signal_prog(phases)
    ctx = self._make_context(
      prog,
      role,
      "ev_i" if "input_released" in phases else "",
      "ev_o" if "output_ready" in phases else "",
      dict.fromkeys(phases, "all_tasks"),
    )
    verify_workload_ir(ModuleOp([prog, ctx]))


class TestL2AccessContract:
  BINDING_IR = """builtin.module {
  tile.program @access(
      %task : !nest.task,
      %read_src : !nest.l2_buffer<1x4x32xbf16>,
      %write_dst : !nest.l2_buffer<1x4x32xbf16>,
      %unused : !nest.l2_buffer<1x4x32xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 1024> {
    %read_view = tile.subview %read_src task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 4, 32] strides = [1, 1, 1]
        : !nest.l2_view<1x4x32xbf16>
    %write_view = tile.subview %write_dst task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 4, 32] strides = [1, 1, 1]
        : !nest.l2_view<1x4x32xbf16>
    %work = tile.alloc shape = [4, 32] dtype = "bf16"
        : !tile.l1_buffer<4x32xbf16>
    %load = tile.load.async %read_view into %work : !tile.event<"load">
    tile.await %load
    tile.signal input_released(%task)
    %write = tile.store.async %work into %write_view : !tile.event<"write">
    tile.await %write
    tile.signal output_ready(%task)
    tile.return
  }

  nest.context @access_ctx(
      %dst : !nest.global_memref<1x4x32xbf16>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
          logical_tasks = 1, l2_spm_bytes = 3072, requested_contexts_per_tile = 1> {
    %X = nest.alloc slot = "X" role = "in" shape = [1, 4, 32]
        dtype = "bf16" : !nest.l2_buffer<1x4x32xbf16>
    %Y = nest.alloc slot = "Y" role = "out" shape = [1, 4, 32]
        dtype = "bf16" : !nest.l2_buffer<1x4x32xbf16>
    %Z = nest.alloc slot = "Z" role = "in" shape = [1, 2, 32]
        dtype = "bf16" : !nest.l2_buffer<1x2x32xbf16>
    %dst_view = nest.subview %dst offsets = [0, 0, 0] sizes = [1, 4, 32]
        strides = [1, 1, 1] : !nest.global_view<1x4x32xbf16>
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %read, %ready = nest.dispatch.tasks.async @access l1_mode = 0
        tasks(%tasks) globals() bindings(%X, %Y, %X) ins(%X) outs(%Y)
        signal_policy { input_released = #nest.aggregate<all_tasks>,
                        output_ready = #nest.aggregate<all_tasks> }
        : (!nest.event<"grid">, !nest.event<"read">, !nest.event<"ready">)
    %stored = nest.dma.store.async %Y into %dst_view depends_on(%ready)
        : !nest.event<"stored">
    nest.release %X depends_on(%read)
    nest.release %Y depends_on(%stored)
    nest.release %Z
    nest.await %grid, %stored
    nest.return
  }
}
"""

  PHASE_IR = """builtin.module {
  tile.program @phase(
      %task : !nest.task,
      %buffer : !nest.l2_buffer<1x4x32xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 1024> {
    %view = tile.subview %buffer task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 4, 32] strides = [1, 1, 1]
        : !nest.l2_view<1x4x32xbf16>
    %work = tile.alloc shape = [4, 32] dtype = "bf16"
        : !tile.l1_buffer<4x32xbf16>
    %load = tile.load.async %view into %work : !tile.event<"load">
    tile.await %load
    tile.signal input_released(%task)
    %store = tile.store.async %work into %view : !tile.event<"store">
    tile.await %store
    tile.signal output_ready(%task)
    tile.return
  }

  nest.context @phase_ctx(
      %dst : !nest.global_memref<1x4x32xbf16>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
          logical_tasks = 1, l2_spm_bytes = 1024, requested_contexts_per_tile = 1> {
    %buffer = nest.alloc slot = "buffer" role = "inout" shape = [1, 4, 32]
        dtype = "bf16" : !nest.l2_buffer<1x4x32xbf16>
    %dst_view = nest.subview %dst offsets = [0, 0, 0] sizes = [1, 4, 32]
        strides = [1, 1, 1] : !nest.global_view<1x4x32xbf16>
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %read, %ready = nest.dispatch.tasks.async @phase l1_mode = 0
        tasks(%tasks) globals() bindings(%buffer) ins(%buffer) outs(%buffer)
        signal_policy { input_released = #nest.aggregate<all_tasks>,
                        output_ready = #nest.aggregate<all_tasks> }
        : (!nest.event<"grid">, !nest.event<"read">, !nest.event<"ready">)
    %stored = nest.dma.store.async %buffer into %dst_view depends_on(%ready)
        : !nest.event<"stored">
    nest.release %buffer depends_on(%read, %stored)
    nest.await %grid, %stored
    nest.return
  }
}
"""

  RELEASE_IR = """builtin.module {
  tile.program @writer(
      %task : !nest.task,
      %buffer : !nest.l2_buffer<1x4x32xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 1024> {
    %view = tile.subview %buffer task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 4, 32] strides = [1, 1, 1]
        : !nest.l2_view<1x4x32xbf16>
    %work = tile.alloc shape = [4, 32] dtype = "bf16"
        : !tile.l1_buffer<4x32xbf16>
    %write = tile.store.async %work into %view : !tile.event<"write">
    tile.await %write
    tile.signal output_ready(%task)
    tile.return
  }

  tile.program @reader(
      %task : !nest.task,
      %buffer : !nest.l2_buffer<1x4x32xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 1024> {
    %view = tile.subview %buffer task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 4, 32] strides = [1, 1, 1]
        : !nest.l2_view<1x4x32xbf16>
    %work = tile.alloc shape = [4, 32] dtype = "bf16"
        : !tile.l1_buffer<4x32xbf16>
    %read = tile.load.async %view into %work : !tile.event<"read">
    tile.await %read
    tile.signal input_released(%task)
    tile.return
  }

  nest.context @release_ctx(
      %arena : !nest.global_memref<1x4x32xbf16>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
          logical_tasks = 3, l2_spm_bytes = 1024, requested_contexts_per_tile = 1> {
    %shared = nest.alloc slot = "shared" role = "inout" shape = [1, 4, 32]
        dtype = "bf16" : !nest.l2_buffer<1x4x32xbf16>
    %view = nest.subview %arena offsets = [0, 0, 0] sizes = [1, 4, 32]
        strides = [1, 1, 1] : !nest.global_view<1x4x32xbf16>
    %pref = nest.dma.prefetch.async %view into %shared : !nest.event<"pref">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %writer_grid, %writer_read, %writer_ready = nest.dispatch.tasks.async @writer l1_mode = 0
        tasks(%tasks) globals() bindings(%shared) ins() outs(%shared)
        signal_policy { output_ready = #nest.aggregate<all_tasks> }
        depends_on(%pref)
        : (!nest.event<"writer_grid">, !nest.event<"">,
           !nest.event<"writer_ready">)
    %early_store = nest.dma.store.async %shared into %view
        depends_on(%writer_ready) : !nest.event<"early_store">
    %reader_a_grid, %reader_a_read, %reader_a_ready = nest.dispatch.tasks.async @reader l1_mode = 0
        tasks(%tasks) globals() bindings(%shared) ins(%shared) outs()
        signal_policy { input_released = #nest.aggregate<all_tasks> }
        depends_on(%writer_ready)
        : (!nest.event<"reader_a_grid">, !nest.event<"reader_a_read">,
           !nest.event<"">)
    %reader_b_grid, %reader_b_read, %reader_b_ready = nest.dispatch.tasks.async @reader l1_mode = 0
        tasks(%tasks) globals() bindings(%shared) ins(%shared) outs()
        signal_policy { input_released = #nest.aggregate<all_tasks> }
        depends_on(%writer_ready)
        : (!nest.event<"reader_b_grid">, !nest.event<"reader_b_read">,
           !nest.event<"">)
    %late_store = nest.dma.store.async %shared into %view
        depends_on(%writer_ready) : !nest.event<"late_store">
    nest.release %shared depends_on(%reader_a_read, %reader_b_read, %pref,
                                    %early_store, %late_store)
    nest.await %writer_grid, %reader_a_grid, %reader_b_grid,
               %early_store, %late_store
    nest.return
  }
}
"""

  INPUT_WITH_STORE_IR = """builtin.module {
  tile.program @reader(
      %task : !nest.task,
      %buffer : !nest.l2_buffer<1x4x32xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 1024> {
    %view = tile.subview %buffer task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 4, 32] strides = [1, 1, 1]
        : !nest.l2_view<1x4x32xbf16>
    %work = tile.alloc shape = [4, 32] dtype = "bf16"
        : !tile.l1_buffer<4x32xbf16>
    %read = tile.load.async %view into %work : !tile.event<"read">
    tile.await %read
    tile.signal input_released(%task)
    tile.return
  }
  nest.context @input_ctx(
      %arena : !nest.global_memref<1x4x32xbf16>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
          logical_tasks = 1, l2_spm_bytes = 1024, requested_contexts_per_tile = 1> {
    %buffer = nest.alloc slot = "buffer" role = "in" shape = [1, 4, 32]
        dtype = "bf16" : !nest.l2_buffer<1x4x32xbf16>
    %view = nest.subview %arena offsets = [0, 0, 0] sizes = [1, 4, 32]
        strides = [1, 1, 1] : !nest.global_view<1x4x32xbf16>
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %read, %ready = nest.dispatch.tasks.async @reader l1_mode = 0
        tasks(%tasks) globals() bindings(%buffer) ins(%buffer) outs()
        signal_policy { input_released = #nest.aggregate<all_tasks> }
        : (!nest.event<"grid">, !nest.event<"read">, !nest.event<"">)
    %stored = nest.dma.store.async %buffer into %view
        : !nest.event<"stored">
    nest.release %buffer depends_on(%read, %stored)
    nest.await %grid, %stored
    nest.return
  }
}
"""

  OUTPUT_IR = """builtin.module {
  tile.program @writer(
      %task : !nest.task,
      %buffer : !nest.l2_buffer<1x4x32xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 1024> {
    %view = tile.subview %buffer task = %task task_dim = 0
        offsets = [0, 0, 0] sizes = [1, 4, 32] strides = [1, 1, 1]
        : !nest.l2_view<1x4x32xbf16>
    %work = tile.alloc shape = [4, 32] dtype = "bf16"
        : !tile.l1_buffer<4x32xbf16>
    %write = tile.store.async %work into %view : !tile.event<"write">
    tile.await %write
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @output_ctx(
      %arena : !nest.global_memref<1x4x32xbf16>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
          logical_tasks = 1, l2_spm_bytes = 1024, requested_contexts_per_tile = 1> {
    %buffer = nest.alloc slot = "buffer" role = "out" shape = [1, 4, 32]
        dtype = "bf16" : !nest.l2_buffer<1x4x32xbf16>
    %view = nest.subview %arena offsets = [0, 0, 0] sizes = [1, 4, 32]
        strides = [1, 1, 1] : !nest.global_view<1x4x32xbf16>
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %read, %ready = nest.dispatch.tasks.async @writer l1_mode = 0
        tasks(%tasks) globals() bindings(%buffer) ins() outs(%buffer)
        signal_policy { output_ready = #nest.aggregate<all_tasks> }
        : (!nest.event<"grid">, !nest.event<"">, !nest.event<"ready">)
    %stored = nest.dma.store.async %buffer into %view depends_on(%ready)
        : !nest.event<"stored">
    nest.release %buffer depends_on(%stored)
    nest.await %grid, %stored
    nest.return
  }
}
"""

  @staticmethod
  def _assert_rejected(text: str) -> None:
    with pytest.raises(VerifyException):
      parse_workload_ir(text, source_name="<l2-access-negative>")

  def test_bindings_and_effects_are_independent(self):
    module = parse_workload_ir(self.BINDING_IR, source_name="<bindings>")
    text = print_workload_ir(module)
    assert print_workload_ir(parse_workload_ir(text, source_name="<bindings-round-trip>")) == text

    mutations = [
      ("bindings(%X, %Y, %X)", "bindings(%X, %Y)"),
      ("bindings(%X, %Y, %X)", "bindings(%X, %Y, %Z)"),
      ("ins(%X)", "ins()"),
      ("outs(%Y)", "outs(%Y, %X)"),
      ("ins(%X)", "ins(%X, %X)"),
      ("ins(%X)", "ins(%Z)"),
    ]
    for old, new in mutations:
      self._assert_rejected(self.BINDING_IR.replace(old, new, 1))

  def test_phase_signals_seal_completed_l2_accesses(self):
    parse_workload_ir(self.PHASE_IR, source_name="<phase>")

    mutations = [
      ("    tile.await %load\n", ""),
      (
        "    tile.signal input_released(%task)\n",
        (
          "    tile.signal input_released(%task)\n"
          "    %late_load = tile.load.async %view into %work"
          ' : !tile.event<"late_load">\n'
          "    tile.await %late_load\n"
        ),
      ),
      ("    tile.await %store\n", ""),
      (
        "    tile.signal output_ready(%task)\n",
        (
          "    tile.signal output_ready(%task)\n"
          "    %late_store = tile.store.async %work into %view"
          ' : !tile.event<"late_store">\n'
          "    tile.await %late_store\n"
        ),
      ),
      (
        "    tile.signal input_released(%task)\n",
        ("    tile.signal input_released(%task)\n    tile.signal input_released(%task)\n"),
      ),
    ]
    for old, new in mutations:
      self._assert_rejected(self.PHASE_IR.replace(old, new, 1))

    early_return = self.PHASE_IR.replace(
      "    tile.signal input_released(%task)\n",
      "    tile.signal input_released(%task)\n    tile.return\n",
      1,
    )
    self._assert_rejected(early_return)

    ordered = """    %load = tile.load.async %view into %work : !tile.event<"load">
    tile.await %load
    tile.signal input_released(%task)
    %store = tile.store.async %work into %view : !tile.event<"store">
    tile.await %store
    tile.signal output_ready(%task)
"""
    reversed_phases = """    %store = tile.store.async %work into %view : !tile.event<"store">
    tile.await %store
    tile.signal output_ready(%task)
    %load = tile.load.async %view into %work : !tile.event<"load">
    tile.await %load
    tile.signal input_released(%task)
"""
    parse_workload_ir(self.PHASE_IR.replace(ordered, reversed_phases, 1), source_name="<reversed-phases>")

  def test_release_requires_readers_and_all_transfers(self):
    parse_workload_ir(self.RELEASE_IR, source_name="<release>")
    self._assert_rejected(
      self.RELEASE_IR.replace(
        (
          "    %early_store = nest.dma.store.async %shared into %view\n"
          '        depends_on(%writer_ready) : !nest.event<"early_store">'
        ),
        (
          '    %early_store = nest.dma.store.async %shared into %view\n        : !nest.event<"early_store">'
        ),
        1,
      )
    )

    release_line = (
      "    nest.release %shared depends_on(%reader_a_read, %reader_b_read, %pref,\n"
      "                                    %early_store, %late_store)"
    )
    invalid_releases = [
      (
        "    nest.release %shared depends_on(%reader_b_read, %pref,\n"
        "                                    %early_store, %late_store)"
      ),
      (
        "    nest.release %shared depends_on(%reader_a_read, %reader_b_read, %pref,\n"
        "                                    %late_store)"
      ),
      (
        "    nest.release %shared depends_on(%reader_a_read, %reader_b_read,\n"
        "                                    %early_store, %late_store)"
      ),
      (
        "    nest.release %shared depends_on(%reader_a_read, %reader_b_grid, %pref,\n"
        "                                    %early_store, %late_store)"
      ),
      (
        "    nest.release %shared depends_on(%reader_a_read, %reader_b_read,"
        " %pref, %pref,\n"
        "                                    %early_store, %late_store)"
      ),
    ]
    for replacement in invalid_releases:
      self._assert_rejected(self.RELEASE_IR.replace(release_line, replacement, 1))

    release_after_return = self.RELEASE_IR.replace(
      (
        f"{release_line}\n"
        "    nest.await %writer_grid, %reader_a_grid, %reader_b_grid,\n"
        "               %early_store, %late_store\n"
        "    nest.return"
      ),
      (
        "    nest.await %writer_grid, %reader_a_grid, %reader_b_grid,\n"
        "               %early_store, %late_store\n"
        f"    nest.return\n{release_line}"
      ),
      1,
    )
    self._assert_rejected(release_after_return)

    self._assert_rejected(self.RELEASE_IR.replace('role = "inout"', 'role = "in"', 1))

    parse_workload_ir(self.INPUT_WITH_STORE_IR, source_name="<input-with-store>")
    for role in ("out", "inout"):
      self._assert_rejected(self.INPUT_WITH_STORE_IR.replace('role = "in"', f'role = "{role}"', 1))

    for role in ("out", "inout"):
      valid_output = self.OUTPUT_IR.replace('role = "out"', f'role = "{role}"', 1)
      parse_workload_ir(valid_output, source_name=f"<{role}-writer>")
      self._assert_rejected(
        valid_output.replace(
          (
            "    %stored = nest.dma.store.async %buffer into %view"
            " depends_on(%ready)\n"
            '        : !nest.event<"stored">\n'
            "    nest.release %buffer depends_on(%stored)\n"
            "    nest.await %grid, %stored"
          ),
          "    nest.release %buffer\n    nest.await %grid",
          1,
        )
      )


class TestTileFree:
  IR = """builtin.module {
  tile.program @free_work(%task: !nest.task, %buf: !nest.l2_buffer<1x64xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 2048> {
    %view = tile.subview %buf offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.l2_view<1x64xbf16>
    %a = tile.alloc shape = [64] dtype = "bf16" : !tile.l1_buffer<64xbf16>
    %b = tile.alloc shape = [64] dtype = "bf16" : !tile.l1_buffer<64xbf16>
    %ra = tile.load.async %view into %a : !tile.event<"ra">
    tile.await %ra
    %rb = tile.load.async %view into %b : !tile.event<"rb">
    tile.free %a
    tile.await %rb
    tile.signal input_released(%task)
    %write = tile.store.async %b into %view : !tile.event<"write">
    tile.await %write
    tile.free %b
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @ctx(%input: !nest.global_memref<1x64xbf16>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
          logical_tasks = 1, l2_spm_bytes = 1024, requested_contexts_per_tile = 1> {
    %buf = nest.alloc slot = "buf" role = "inout" shape = [1, 64] dtype = "bf16"
        : !nest.l2_buffer<1x64xbf16>
    %view = nest.subview %input offsets = [0, 0] sizes = [1, 64] strides = [1, 1]
        : !nest.global_view<1x64xbf16>
    %pref = nest.dma.prefetch.async %view into %buf : !nest.event<"pref">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %read, %ready = nest.dispatch.tasks.async @free_work l1_mode = 0
        tasks(%tasks) globals() bindings(%buf) ins(%buf) outs(%buf)
        signal_policy { input_released = #nest.aggregate<all_tasks>,
                        output_ready = #nest.aggregate<all_tasks> }
        depends_on(%pref) : (!nest.event<"grid">, !nest.event<"read">, !nest.event<"ready">)
    %stored = nest.dma.store.async %buf into %view depends_on(%ready) : !nest.event<"stored">
    nest.release %buf depends_on(%read, %pref, %stored)
    nest.await %grid, %stored
    nest.return
  }
}"""

  def test_roundtrip_allows_unrelated_pending_transfer(self):
    module = parse_workload_ir(self.IR)
    assert module.is_structurally_equivalent(parse_workload_ir(print_workload_ir(module)))
    # b's load is intentionally not awaited until after freeing a.
    simulator = Simulator(HardwareConfig(), SimConfig(fidelity="full_memory", max_cycles=100000))
    result = run_source(simulator, module, {"input": GlobalBinding("input", 0x100000, 128, "rw")})
    assert result.completed, result.reason
    assert result.credit_invariant_ok

  @pytest.mark.parametrize(
    ("old", "new"),
    [
      ("    tile.await %ra\n", ""),
      ("    tile.await %write\n", ""),
      ("tile.free %a", "tile.free %a\n    tile.free %a"),
      ("tile.free %a", 'tile.free %a\n    %again = tile.load.async %view into %a : !tile.event<"again">'),
      ("tile.store.async %b into %view", "tile.store.async %a into %view"),
      (
        "tile.free %a",
        '%compute = tile.evu.async "relu" ops = 16448 : !tile.event<"compute">\n    tile.free %a',
      ),
    ],
    ids=[
      "pending-load",
      "pending-store",
      "double-free",
      "load-after-free",
      "store-after-free",
      "pending-opaque-compute",
    ],
  )
  def test_rejects_unsafe_lifetime(self, old, new):
    with pytest.raises(VerifyException):
      parse_workload_ir(self.IR.replace(old, new, 1))

  def test_free_requires_current_program_local_allocation(self):
    from pipeline_validator.dialects.elenor import TileFreeOp

    foreign = TileAllocOp([64], "bf16")
    owner = TileProgramDefOp(
      "owner", tile_resources(1024), [foreign, TileReturnOp()], arg_types=[NestTask()], arg_names=["task"]
    )
    consumer = TileProgramDefOp(
      "consumer",
      tile_resources(),
      [TileFreeOp(foreign.result), TileReturnOp()],
      arg_types=[NestTask()],
      arg_names=["task"],
    )
    with pytest.raises(VerifyException):
      verify_workload_ir(
        ModuleOp(
          [owner, consumer, NestContextOp("ctx", context_resources(), [NestReturnOp()], placement=1)]
        )
      )
    with pytest.raises(VerifyException):
      parse_workload_ir(self.IR.replace("tile.free %a", "tile.free %buf", 1))

  @pytest.mark.parametrize("buffer", ["indices_l1", "gather_dst"])
  def test_gather_buffers_live_until_gather_completion(self, buffer):
    source = (Path(__file__).resolve().parents[2] / "examples/workloads/gather_indexed.mlir").read_text()
    with pytest.raises(VerifyException):
      parse_workload_ir(
        source.replace(
          "    tile.await %gather_done", f"    tile.free %{buffer}\n    tile.await %gather_done", 1
        )
      )


class TestIndependentL2NoRebindVerifier:
  """plan/01 §1/§6.1: the independent executable verifier rejects reuse-era
  or tampered L2 layouts from the frozen DTO alone."""

  @staticmethod
  def _compile_two_buffer_task():
    from pathlib import Path

    from pipeline_validator.execution_ir import ExecModel

    root = Path(__file__).resolve().parents[2]
    hw = HardwareConfig().with_overrides(num_dma_channels=2, hbm_fixed_latency_cycles=10)
    sim = SimConfig(max_cycles=200000, context_count=4)
    module = parse_workload_ir((root / "examples/workloads/matmul_pow_data_dep.mlir").read_text())
    artifact = compile_program(module, hw, sim)
    assert isinstance(artifact.entry, ExecModel)
    binding_id, task = next(
      (key, task) for key, task in artifact.entry.tasks.items() if len(task.layout.buffer_layouts) >= 2
    )
    return artifact, hw, sim, binding_id, task

  @staticmethod
  def _reseal_with_layout(artifact, task, mutate):
    from dataclasses import replace

    from pipeline_validator.compiled_program import seal_program
    from pipeline_validator.immutable import digest

    base = replace(task.layout, layout_hash="")
    tampered = mutate(base)
    relaid = replace(tampered, layout_hash=digest(tampered))
    model = replace(
      artifact.entry, tasks={**artifact.entry.tasks, task.binding_id: replace(task, layout=relaid)}
    )
    return seal_program(replace(artifact, entry=model)), relaid

  def test_overlap_tampered_layout_rejected(self):
    from dataclasses import replace

    artifact, hw, sim, _binding_id, task = self._compile_two_buffer_task()

    def mutate(layout):

      items = list(layout.buffer_layouts)
      items[1] = replace(items[1], arena_offset=items[0].arena_offset)
      return replace(layout, buffer_layouts=tuple(items))

    corrupted, _ = self._reseal_with_layout(artifact, task, mutate)
    with pytest.raises(ValueError, match="overlapping padded L2 spans"):
      load_program(corrupted, hw, sim)

  def test_padded_span_escape_and_overlap_unit(self):
    """Unit: padded spans that escape their bank or overlap are rejected."""
    from dataclasses import replace

    from pipeline_validator.execution_verifier import _verify_l2_no_rebind_layout
    from pipeline_validator.immutable import digest
    from pipeline_validator.profiles import ArenaLayout, BufferLayout

    # banks=2, stripe=64, per_bank=256, reserved=512 (whole stripe rounds).
    a = BufferLayout("a", 100, 0, 0, 64, 2, "in")  # padded span [0, 64) per bank
    base = ArenaLayout(64, 64, 512, (256, 256), (a,), "")

    def sealed(**changes):
      candidate = replace(base, **changes)
      return replace(candidate, layout_hash=digest(candidate))

    _verify_l2_no_rebind_layout(sealed(), "x")
    fitting = BufferLayout("b", 100, 1, 384, 64, 2, "inout")  # span [192, 256)
    # Leading gap [64,192) is legal arena slack between disjoint spans.
    _verify_l2_no_rebind_layout(sealed(buffer_layouts=(a, fitting)), "x")
    # A padded span that escapes one bank would force bank 0's real segments
    # to escape too, so ArenaLayout construction rejects it first; the
    # verifier's bound check stays as defense-in-depth for mutated objects.
    overlapped = BufferLayout("d", 100, 1, 0, 64, 2, "inout")
    with pytest.raises(ValueError, match="overlapping padded L2 spans"):
      _verify_l2_no_rebind_layout(sealed(buffer_layouts=(a, overlapped)), "x")

  def test_double_release_rejected(self):
    from dataclasses import replace

    from pipeline_validator.compiled_program import seal_program
    from pipeline_validator.execution_ir import ExecGroupActionOp

    artifact, hw, sim, _binding_id, task = self._compile_two_buffer_task()
    last_release_index = max(
      index for index, action in enumerate(task.actions) if action.op is ExecGroupActionOp.RELEASE_L2
    )
    release = task.actions[last_release_index]
    forged = replace(
      release,
      dst=f"{release.dst}:forged",
      instruction_id=f"{task.binding_id}:group:forged_release",
    )
    actions = (*task.actions[:last_release_index + 1], forged, *task.actions[last_release_index + 1 :])
    rebound = replace(task, actions=actions)
    model = replace(artifact.entry, tasks={**artifact.entry.tasks, task.binding_id: rebound})
    corrupted = seal_program(replace(artifact, entry=model))
    with pytest.raises(ValueError, match="already released L2 view"):
      load_program(corrupted, hw, sim)

  def test_duplicate_bind_rejected(self):
    from dataclasses import replace

    from pipeline_validator.compiled_program import seal_program
    from pipeline_validator.execution_ir import ExecGroupActionOp

    artifact, hw, sim, _binding_id, task = self._compile_two_buffer_task()
    first_bind = next(action for action in task.actions if action.op is ExecGroupActionOp.BIND_L2_VIEW)
    forged = replace(
      first_bind,
      dst=f"{first_bind.dst}:forged",
      instruction_id=f"{task.binding_id}:group:forged_bind",
    )
    actions = (*task.actions, forged)
    rebound = replace(task, actions=actions)
    model = replace(artifact.entry, tasks={**artifact.entry.tasks, task.binding_id: rebound})
    corrupted = seal_program(replace(artifact, entry=model))
    with pytest.raises(ValueError):
      load_program(corrupted, hw, sim)


class TestIndependentL2LifetimeTamper:
  """plan/01 §6.1: wrong boundary and missing bind/release artifact cases."""

  @staticmethod
  def _compile_two_buffer_task():
    from pathlib import Path

    from pipeline_validator.execution_ir import ExecModel

    root = Path(__file__).resolve().parents[2]
    hw = HardwareConfig().with_overrides(num_dma_channels=2, hbm_fixed_latency_cycles=10)
    sim = SimConfig(max_cycles=200000, context_count=4)
    module = parse_workload_ir((root / "examples/workloads/matmul_pow_data_dep.mlir").read_text())
    artifact = compile_program(module, hw, sim)
    assert isinstance(artifact.entry, ExecModel)
    binding_id, task = next(
      (key, task) for key, task in artifact.entry.tasks.items() if len(task.layout.buffer_layouts) >= 2
    )
    return artifact, hw, sim, binding_id, task

  @staticmethod
  def _reseal(artifact, task):
    from dataclasses import replace

    from pipeline_validator.compiled_program import seal_program

    model = replace(artifact.entry, tasks={**artifact.entry.tasks, task.binding_id: task})
    return seal_program(replace(artifact, entry=model))

  def test_wrong_boundary_layout_rejected_at_artifact_boundary(self):
    """A padded span crossing its bank reservation cannot construct a valid
    ArenaLayout; the independent boundary rejection fires before the
    verifier's own span checks."""
    from dataclasses import replace

    from pipeline_validator.immutable import digest

    artifact, _hw, _sim, _binding_id, task = self._compile_two_buffer_task()
    base = replace(task.layout, layout_hash="")
    round_bytes = base.stripe_bytes * len(base.per_bank_bytes)
    escaped_offset = (base.reserved_bytes // round_bytes - 1) * round_bytes
    items = list(base.buffer_layouts)
    items[-1] = replace(items[-1], arena_offset=escaped_offset)
    with pytest.raises(ValueError, match="escapes its reservation"):
      tampered = replace(base, buffer_layouts=tuple(items))
      replace(tampered, layout_hash=digest(tampered))
      self._reseal(artifact, replace(task, layout=tampered))

  @staticmethod
  def _prune_action(task, index):
    """Drop one action and its produced event from all downstream
    dependencies so the lifetime checks (not dependency wiring) fire."""
    from dataclasses import replace


    dropped = task.actions[index]
    actions = []
    for position, action in enumerate(task.actions):
      if position == index:
        continue
      if dropped.dst and dropped.dst in action.dependencies:
        action = replace(
          action,
          dependencies=tuple(event for event in action.dependencies if event != dropped.dst),
        )
      actions.append(action)
    return replace(task, actions=tuple(actions))

  def test_missing_bind_rejected(self):
    from pipeline_validator.execution_ir import ExecGroupActionOp

    artifact, hw, sim, _binding_id, task = self._compile_two_buffer_task()
    first_bind_index = next(
      index for index, action in enumerate(task.actions) if action.op is ExecGroupActionOp.BIND_L2_VIEW
    )
    pruned = self._prune_action(task, first_bind_index)
    corrupted = self._reseal(artifact, pruned)
    with pytest.raises(ValueError, match="unbound or released L2 view"):
      load_program(corrupted, hw, sim)

  def test_missing_release_rejected(self):
    from pipeline_validator.execution_ir import ExecGroupActionOp

    artifact, hw, sim, _binding_id, task = self._compile_two_buffer_task()
    last_release_index = max(
      index for index, action in enumerate(task.actions) if action.op is ExecGroupActionOp.RELEASE_L2
    )
    pruned = self._prune_action(task, last_release_index)
    corrupted = self._reseal(artifact, pruned)
    with pytest.raises(ValueError, match="L2 view lifetime is incomplete"):
      load_program(corrupted, hw, sim)
