"""Tests for the ELENOR pipeline validator.

Run with:  python -m pytest pipeline_validator/tests/  (or: pytest)
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, fields
from pathlib import Path

import pytest
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.exceptions import ParseError, VerifyException

from pipeline_validator.config import _HW_YAML_PATH_TO_FIELD, HardwareConfig, SimConfig, _load_hw_yaml
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
from pipeline_validator.execution_ir import ExecEngineDesc, ExecGroupActionOp, GlobalBinding
from pipeline_validator.ir_lowering import lower_workload_ir
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

MODEL_CHAIN_IR = """builtin.module {
  tile.program @pow_4k_tile(
      %task : !nest.task,
      %l2_buf : !nest.l2_buffer<4x128x128xbf16>) {
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
      placement = 15 context = 0 {
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
        @pow_4k_tile context = 0
        tasks(%0) ins(%l2_buf) outs(%l2_buf)
        signal_policy { input_released = #nest.aggregate<all_tasks>,
                        output_ready = #nest.aggregate<all_tasks> }
        depends_on(%ev_in)
        : (!nest.event<"ev_role_pow0">, !nest.event<"ev_inrel_pow0">,
           !nest.event<"ev_outready_pow0">)
    %ev_out = nest.dma.store.async %l2_buf into %src
        depends_on(%ev_outready) : !nest.event<"ev_dma_pow_out0">
    nest.release %l2_buf depends_on(%ev_out)
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

DISPATCH_TYPE_MISMATCH_IR = """builtin.module {
  tile.program @p(%task : !nest.task, %l2 : !nest.l2_buffer<4x128x128xbf16>) {
    tile.return
  }
  nest.context @c placement = 1 {
    %bad = nest.alloc slot = "bad" role = "in" shape = [2, 128, 128]
        dtype = "bf16" : !nest.l2_buffer<2x128x128xbf16>
    %0 = nest.task.range from = 0 to = 1 : !nest.task_range
    %g, %i, %o = nest.dispatch.tasks.async @p
        tasks(%0) ins(%bad) outs(%bad) signal_policy {}
        : (!nest.event<"g">, !nest.event<"i">, !nest.event<"o">)
    nest.await %g
    nest.return
  }
}
"""

TRANSFER_BYTE_MISMATCH_IR = """builtin.module {
  tile.program @p(%task : !nest.task) {
    tile.return
  }
  nest.context @c(%Y : !nest.global_memref<4x128x128xbf16>) placement = 1 {
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
  tile.program @p(%l2 : !nest.l2_buffer<4x128x128xbf16>) {
    tile.return
  }
  nest.context @c placement = 1 {
    %buf = nest.alloc slot = "buf" role = "in" shape = [4, 128, 128]
        dtype = "bf16" : !nest.l2_buffer<4x128x128xbf16>
    %0 = nest.task.range from = 0 to = 1 : !nest.task_range
    %g, %i, %o = nest.dispatch.tasks.async @p
        tasks(%0) ins(%buf) outs() signal_policy {}
        : (!nest.event<"g">, !nest.event<"i">, !nest.event<"o">)
    nest.await %g
    nest.return
  }
}
"""


class TestXDSLIR:
  def _assert_verify_failure(self, module: ModuleOp, message: str) -> None:
    with pytest.raises(VerifyException, match=message):
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

  def test_function_call_op_builders_verify(self):
    """Every function-call operation can participate in a verified module."""
    l2_dims = [1, 4, 32]
    prog = TileProgramDefOp(
      "all_ops", [], arg_types=[NestTask(), NestBuffer.of(l2_dims, "bf16")], arg_names=["task", "l2_buf"]
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
      "all_ops_context", [], placement=1, arg_types=[NestGlobalMemref.of(l2_dims, "bf16")], arg_names=["Y"]
    )
    y_arg = ctx.body.block.args[0]
    buffer = NestAllocOp(slot="l2_buf", role="inout", shape=l2_dims, dtype="bf16")
    src = NestSubviewOp(y_arg, [0, 0, 0], l2_dims, [1, 1, 1], NestGlobalView.of(l2_dims, "bf16"))
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    prefetch = NestPrefetchOp(src.result, buffer.result, "prefetch")
    dispatch = NestDispatchOp(
      "all_ops",
      tasks.result,
      [buffer.result],
      [buffer.result],
      "grid_done",
      "input_released",
      "output_ready",
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=[prefetch.result],
    )
    collective = NestCollectiveOp("reduce", bytes_total=256, participant_mask=1, tag="collective")
    dma_store = NestDMAStoreOp(
      src=buffer.result, dst=src.result, tag="store_done", depends_on=[dispatch.output_ready]
    )
    release = NestReleaseOp(buffer.result, depends_on=[dma_store.result])
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
    """Minimal legacy module: identity program + one dispatch context."""
    prog = make_identity_tile_program()
    tasks = NestTaskRangeOp(0, 1)
    dispatch = NestDispatchOp(
      prog.sym_name.data, tasks.result, [], [], "grid_done", "", "", signal_policy={}
    )
    ctx = NestContextOp(
      ctx_name,
      [tasks, dispatch, NestAwaitOp([dispatch.grid_done]), NestReturnOp()],
      placement=placement,
      context_id=context_id,
    )
    return [prog, ctx], dispatch

  def test_verifier_rejects_unknown_program_symbol(self):
    tasks = NestTaskRangeOp(0, 1)
    dispatch = NestDispatchOp(
      "missing_program", tasks.result, [], [], "grid_done", "", "", signal_policy={}
    )
    module = ModuleOp([NestContextOp("unknown_program", [tasks, dispatch, NestReturnOp()], placement=1)])
    self._assert_verify_failure(module, "dispatch references unknown tile program '@missing_program'")

  def test_verifier_rejects_undefined_event_in_await(self):
    ctx = NestContextOp(
      "undefined_await",
      [],
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
    self._assert_verify_failure(module, "nest.await references undefined event 'defined_later'")

  def test_verifier_rejects_duplicate_event_tag(self):
    ctx = NestContextOp(
      "duplicate_event",
      [],
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
    self._assert_verify_failure(module, "duplicate event tag 'duplicate'")

  def test_dispatch_context_attribute_round_trip(self):
    """``context = N`` prints, parses, and round-trips; absent when not pinned."""
    prog = make_identity_tile_program()
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    dispatch = NestDispatchOp(
      prog.sym_name.data, tasks.result, [], [], "grid_done", "", "", signal_policy={}, context_id=1
    )
    module = ModuleOp(
      [
        prog,
        NestContextOp(
          "pinned_ctx", [tasks, dispatch, NestAwaitOp([dispatch.grid_done]), NestReturnOp()], placement=1
        ),
      ]
    )
    verify_workload_ir(module)
    text = print_workload_ir(module)
    assert "context = 1" in text
    reparsed = parse_workload_ir(text, source_name="<pinned>")
    verify_workload_ir(reparsed)
    found = False
    for op in reparsed.ops:
      if isinstance(op, NestContextOp):
        for body_op in op.body.blocks[0].ops:
          if isinstance(body_op, NestDispatchOp):
            assert body_op.context_id is not None
            assert int(body_op.context_id.value.data) == 1
            found = True
    assert found, "no NestDispatchOp found in reparsed context body"
    # Unpinned dispatch must NOT print the attribute.
    assert "context = " not in print_workload_ir(PowWorkload().module)

  def test_verifier_rejects_negative_dispatch_context(self):
    prog = make_identity_tile_program()
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    dispatch = NestDispatchOp(
      prog.sym_name.data, tasks.result, [], [], "grid_done", "", "", signal_policy={}, context_id=-1
    )
    module = ModuleOp([prog, NestContextOp("neg_ctx", [tasks, dispatch, NestReturnOp()], placement=1)])
    self._assert_verify_failure(module, "dispatch context must be >= 0")

  def test_context_level_context_attribute_round_trip(self):
    """``context = N`` on ``nest.context`` (device slot pin) prints, parses, and round-trips."""
    module = ModuleOp(self._make_identity_context(ctx_name="ctx_default_pin", context_id=1)[0])
    verify_workload_ir(module)
    text = print_workload_ir(module)
    assert "context = 1" in text
    reparsed = parse_workload_ir(text, source_name="<ctx-pin>")
    verify_workload_ir(reparsed)
    ctx_op = next(op for op in reparsed.ops if isinstance(op, NestContextOp))
    assert ctx_op.context_id is not None
    assert int(ctx_op.context_id.value.data) == 1
    # Unpinned context must NOT print the attribute.
    assert "context = " not in print_workload_ir(PowWorkload().module)

  def test_verifier_rejects_negative_context_level_context(self):
    module = ModuleOp(self._make_identity_context(ctx_name="neg_ctx_pin", context_id=-1)[0])
    self._assert_verify_failure(module, "nest.context context must be >= 0")

  def test_nexus_program_round_trip(self):
    """nexus.program with submit/await/return round-trips in custom assembly."""
    prog = make_identity_tile_program()
    ctxs = []
    for i in range(2):
      tasks = NestTaskRangeOp(0, 1)
      disp = NestDispatchOp(
        prog.sym_name.data, tasks.result, [], [], f"ev_grid_c{i}", "", "", signal_policy={}
      )
      ctxs.append(
        NestContextOp(
          f"ctx{i}",
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
    self._assert_verify_failure(module, "submit_context references unknown nest.context '@missing'")

  def test_verifier_rejects_undefined_nexus_await(self):
    """nexus.await referencing an event not yet submitted is rejected."""
    ops, _ = self._make_identity_context(ctx_name="ctx0", context_id=0)
    sub = NexusSubmitContextOp("ctx0", "done0")
    program = NexusProgramOp("bad", [NexusAwaitOp([sub.result]), sub, NexusReturnOp()])
    module = ModuleOp([*ops, program])
    self._assert_verify_failure(module, "nexus.await references undefined event 'done0'")

  def test_submit_context_arity_mismatch_fails(self):
    text = MODEL_CHAIN_IR.replace("@pow_task(%Y0)", "@pow_task(%Y0, %Y1)", 1)
    with pytest.raises(
      VerifyException,
      match=(
        r"submit_context '@pow_task' passes 2 actuals"
        r" but nest.context '@pow_task' declares 1 formals"
      ),
    ):
      parse_workload_ir(text, source_name="<arity>")

  def test_submit_context_type_mismatch_fails(self):
    text = MODEL_CHAIN_IR.replace(
      "%Y1 : !nest.global_memref<4x128x128xbf16>", "%Y1 : !nest.global_memref<8x128x128xbf16>", 1
    ).replace("@pow_task(%Y0)", "@pow_task(%Y1)", 1)
    with pytest.raises(
      VerifyException,
      match=(
        r"submit_context actual 0 type does not match"
        r" nest.context '@pow_task' formal 0"
      ),
    ):
      parse_workload_ir(text, source_name="<type>")

  def test_dispatch_actual_arity_mismatch_fails(self):
    text = MODEL_CHAIN_IR.replace(
      "tasks(%0) ins(%l2_buf) outs(%l2_buf)", "tasks(%0) ins(%l2_buf, %l2_buf) outs(%l2_buf)", 1
    )
    with pytest.raises(
      VerifyException,
      match=(
        r"dispatch '@pow_4k_tile' passes 3 actuals"
        r" but tile.program declares 1 l2 formals"
      ),
    ):
      parse_workload_ir(text, source_name="<dispatch-arity>")

  def test_dispatch_actual_type_mismatch_fails(self):
    with pytest.raises(
      VerifyException,
      match=(
        r"dispatch actual 0 type does not match"
        r" tile.program '@p' formal 0"
      ),
    ):
      parse_workload_ir(DISPATCH_TYPE_MISMATCH_IR, source_name="<dispatch-type>")

  def test_nest_subview_out_of_bounds_fails(self):
    text = MODEL_CHAIN_IR.replace(
      "nest.subview %Y offsets = [0, 0, 0] sizes = [4, 128, 128]",
      "nest.subview %Y offsets = [5, 0, 0] sizes = [4, 128, 128]",
      1,
    )
    with pytest.raises(
      VerifyException,
      match=(
        r"nest.subview exceeds bounds of 'Y' dim 0:"
        r" offset 5 \+ size 4 > 4"
      ),
    ):
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
    with pytest.raises(
      VerifyException,
      match=(
        r"tile.subview on formal 0 dim 0:"
        r" offset 0 \+ max task 3 \+ size 2 exceeds 4"
      ),
    ):
      parse_workload_ir(text, source_name="<task-overflow>")

  def test_non_unit_stride_rejected(self):
    text = MODEL_CHAIN_IR.replace("strides = [1, 1, 1]", "strides = [1, 2, 1]", 1)
    with pytest.raises(VerifyException, match="non-unit strides are not supported in V1"):
      parse_workload_ir(text, source_name="<stride>")

  def test_transfer_byte_mismatch_fails(self):
    with pytest.raises(
      VerifyException,
      match=(
        r"transfer 'nest.dma.prefetch.async' src bytes \(131072\)"
        r" != dst bytes \(65536\)"
      ),
    ):
      parse_workload_ir(TRANSFER_BYTE_MISMATCH_IR, source_name="<bytes>")

  def test_tile_program_requires_task_formal(self):
    with pytest.raises(VerifyException, match=(r"tile.program '@p' first formal must be !nest.task")):
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
        "  tile.program @p(%task : !nest.task) {\n"
        f"    {old_text}\n"
        "    tile.await %e\n"
        "    tile.return\n"
        "  }\n"
        "}\n"
      )
    else:
      text = (
        "builtin.module {\n"
        "  nest.context @c placement = 1 {\n"
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
    task = lower_workload_ir(PowWorkload().module)
    # L2 buffer: element_bytes from dtype, alignment from nest.alloc
    l2 = task.l2_buffers[0]
    assert l2.element_bytes == 2  # bf16
    assert l2.alignment == 256
    assert l2.bytes == 4 * 128 * 128 * 2
    # prefetch transfer: src is the global subview
    prog = task.role_bindings[0].tile_program
    prefetch_action = task.actions[0]
    assert prefetch_action.op == ExecGroupActionOp.DMA_PREFETCH
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
    prog = TileProgramDefOp(
      "sv_prog",
      [],
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
    ctx = NestContextOp(
      "sv_ctx", [], arg_types=[NestGlobalMemref.of(g_dims, "bf16")], arg_names=["Y"], placement=1
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
      [buf.result],
      [buf.result],
      "ev_grid",
      "ev_inrel",
      "",
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
        NestReleaseOp(buf.result, depends_on=[disp.input_released]),
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
    with pytest.raises(
      VerifyException,
      match=(
        r"non-contiguous subviews are not supported by the physical"
        r" transfer model"
      ),
    ):
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
    with pytest.raises(
      VerifyException,
      match=(
        r"non-contiguous subviews are not supported by the physical"
        r" transfer model"
      ),
    ):
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
    task = lower_workload_ir(module)
    pref = task.actions[0]
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

  def test_ir_file_conflicts_and_load_errors(self, tmp_path: Path):
    input_path = tmp_path / "pow.mlir"
    input_path.write_text(print_workload_ir(PowWorkload().module), encoding="utf-8")

    conflict = self._run_cli("--ir-file", str(input_path), "-w", "pow")
    assert conflict.returncode == 2

    missing_trace = tmp_path / "missing.json"
    missing_report = tmp_path / "missing.txt"
    missing_path = tmp_path / "missing.mlir"
    missing = self._run_cli(
      "--ir-file", str(missing_path), "--trace-json", str(missing_trace), "--report", str(missing_report)
    )
    assert missing.returncode == 2
    assert "failed to load IR" in missing.stderr
    assert str(missing_path) in missing.stderr
    assert not missing_trace.exists()
    assert not missing_report.exists()

    bad_utf8 = tmp_path / "bad_utf8.mlir"
    bad_utf8.write_bytes(b"\xff\xfe")
    bad_utf8_res = self._run_cli("--ir-file", str(bad_utf8))
    assert bad_utf8_res.returncode == 2
    assert "failed to load IR" in bad_utf8_res.stderr
    assert str(bad_utf8) in bad_utf8_res.stderr

    unknown_op = tmp_path / "unknown_op.mlir"
    unknown_op.write_text(
      input_path.read_text(encoding="utf-8").replace("nest.dma.prefetch", "nest.bad_op", 1),
      encoding="utf-8",
    )
    unknown_res = self._run_cli("--ir-file", str(unknown_op))
    assert unknown_res.returncode == 2
    assert "failed to load IR" in unknown_res.stderr
    assert str(unknown_op) in unknown_res.stderr

    malformed = tmp_path / "malformed.mlir"
    malformed.write_text("not mlir\n", encoding="utf-8")
    malformed_res = self._run_cli("--ir-file", str(malformed))
    assert malformed_res.returncode == 2
    assert "failed to load IR" in malformed_res.stderr
    assert str(malformed) in malformed_res.stderr

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
      assert "--context-mode must be between 1 and 8" in res.stderr

  def test_example_mlir_runs_on_two_device_contexts(self):
    res = self._run_cli(
      "--ir-file",
      "examples/example.mlir",
      "--device-context-mode",
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

  def test_missing_input_binding_exits_2(self):
    res = self._run_cli(
      "--ir-file", "examples/example.mlir", "--device-context-mode", "2", "--max-cycles", "200000"
    )
    assert res.returncode == 2
    assert "missing input binding for global 'Y0'" in res.stderr

  def test_device_context_mode_cli_bounds(self):
    for bad in ("0", "9"):
      res = self._run_cli("-w", "pow", "--device-context-mode", bad)
      assert res.returncode == 2
      assert "--device-context-mode must be between 1 and 8" in res.stderr


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
    result = sim.run(PowWorkload().module, input_bindings=POW_BINDINGS)
    assert result.completed, f"pow did not complete: {result.reason}"
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_pow_report_has_passing_checks(self):
    wl = PowWorkload()
    sim = Simulator(FAST_HW, SimConfig(max_cycles=200_000))
    result = sim.run(wl.module, input_bindings=POW_BINDINGS)
    assert result.completed, result.reason
    rep = build_report(wl, result)
    failed = [c for c in rep.checks if not c["pass"]]
    assert not failed, f"pow failed checks: {failed}"

  def test_pow_report_text_renderable(self):
    wl = PowWorkload()
    sim = Simulator(FAST_HW, SimConfig(max_cycles=200_000))
    result = sim.run(wl.module, input_bindings=POW_BINDINGS)
    rep = build_report(wl, result)
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

  def test_defaults_frozen(self):
    # 55 个字段逐一列出, 值等于迁移前 config.py 的字面量。
    expected = {
      "profile": "balanced-small",
      "num_tiles": 4,
      "clock_mhz": 1000.0,
      "group_sram_bytes": 8388608,
      "group_sram_banks": 16,
      "hbm_bandwidth_gbs": 819.2,
      "group_dma_bandwidth_gbs": 204.8,
      "num_dma_channels": 2,
      "tile_l1_bytes": 1048576,
      "tile_l1_banks": 16,
      "tile_l1_bandwidth_gbs": 512.0,
      "boa_num_opa": 4,
      "boa_opa_rows": 16,
      "boa_opa_cols": 16,
      "boa_clock_multiplier": 1.0,
      "boa_dtype_bytes": 2,
      "boa_acc_bytes": 4,
      "evu_lanes": 32,
      "evu_clock_multiplier": 1.0,
      "evu_dtype_bytes": 2,
      "mfe_bandwidth_gbs": 256.0,
      "mfe_clock_multiplier": 1.0,
      "use_clock_mhz": 500.0,
      "use_state_cache_bytes": 131072,
      "uce_clock_mhz": 1000.0,
      "uce_dispatch_per_cycle": 1,
      "stream_depth_default": 3,
      "stream_token_overhead_cycles": 1,
      "stream_fence_cycles": 1,
      "boa_launch_cycles": 4,
      "evu_launch_cycles": 3,
      "mfe_launch_cycles": 3,
      "mfe_pipeline_depth": 4,
      "mfe_load_channels": 1,
      "mfe_store_channels": 1,
      "mfe_load_queue_depth": 1,
      "mfe_store_queue_depth": 1,
      "mfe_stream_buffer_bytes": 0,
      "use_launch_cycles": 2,
      "dma_launch_cycles": 2,
      "hbm_capacity_bytes": 17179869184,
      "hbm_outstanding_limit": 32,
      "hbm_channels": 8,
      "hbm_fixed_latency_cycles": 200,
      "hbm_burst_bytes": 64,
      "l2_access_latency_cycles": 4,
      "l1_access_latency_cycles": 1,
      "l2_bank_bandwidth_gbs": 12.8,
      "tile_program_sram_bytes": 65536,
      "noc_vc_depth": 8,
      "noc_router_latency_cycles": 4,
      "dma_desc_cycles": 2,
      "dma_issue_cycles": 1,
      "dma_completion_cycles": 1,
      "host_validate_cycles": 50,
      "host_patch_cycles": 10,
      "doorbell_latency_cycles": 5,
      "firmware_fetch_cycles": 3,
      "firmware_validate_cycles": 5,
      "frame_bind_cycles": 8,
    }
    assert asdict(HardwareConfig()) == expected

  def test_schema_maps_every_field_once(self):
    mapped = list(_HW_YAML_PATH_TO_FIELD.values())
    field_names = {field.name for field in fields(HardwareConfig)}
    assert len(_HW_YAML_PATH_TO_FIELD) == 60
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == field_names

  def test_bundled_yaml_roundtrip(self):
    default_path = Path(__file__).resolve().parents[1] / "hardware_config.yaml"
    assert asdict(HardwareConfig.from_yaml(default_path)) == asdict(HardwareConfig())

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

  def test_from_yaml_empty_file_uses_defaults(self, tmp_path):
    path = self._write_yaml(tmp_path, "")
    assert HardwareConfig.from_yaml(path) == HardwareConfig()

  def test_from_yaml_rejects_unknown_nested_path(self, tmp_path):
    path = self._write_yaml(tmp_path, "engines:\n  boa:\n    unknown_lanes: 4\n")
    with pytest.raises(ValueError, match=r"engines\.boa\.unknown_lanes"):
      HardwareConfig.from_yaml(path)

  def test_from_yaml_rejects_scalar_group(self, tmp_path):
    path = self._write_yaml(tmp_path, "engines:\n  boa: 4\n")
    with pytest.raises(ValueError, match=r"engines\.boa.*mapping"):
      HardwareConfig.from_yaml(path)

  def test_from_yaml_rejects_mapping_leaf(self, tmp_path):
    path = self._write_yaml(tmp_path, "engines:\n  boa:\n    launch_cycles:\n      value: 4\n")
    with pytest.raises(ValueError, match=r"engines\.boa\.launch_cycles.*scalar"):
      HardwareConfig.from_yaml(path)

  def test_from_yaml_rejects_sequence(self, tmp_path):
    path = self._write_yaml(tmp_path, "engines:\n  boa:\n    launch_cycles: [4]\n")
    with pytest.raises(ValueError, match="scalar"):
      HardwareConfig.from_yaml(path)

  def test_from_yaml_rejects_unsupported_schema_version(self, tmp_path):
    path = self._write_yaml(tmp_path, "schema_version: 2\n")
    with pytest.raises(ValueError, match="schema_version"):
      HardwareConfig.from_yaml(path)

  def test_from_yaml_rejects_duplicate_key(self, tmp_path):
    path = self._write_yaml(tmp_path, "engines:\n  boa:\n    launch_cycles: 4\n    launch_cycles: 8\n")
    with pytest.raises(ValueError, match="duplicate YAML key"):
      HardwareConfig.from_yaml(path)

  def test_required_yaml_rejects_missing_leaf_before_class_defaults(self, tmp_path):
    path = self._write_yaml(tmp_path, "schema_version: 1\nsystem:\n  profile: balanced-small\n")
    with pytest.raises(ValueError, match="missing required HardwareConfig fields"):
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
    with pytest.raises(ValueError, match="mfe_load_channels must be >= 1"):
      HardwareConfig(mfe_load_channels=0)

  def test_config_rejects_zero_store_channels(self):
    with pytest.raises(ValueError, match="mfe_store_channels must be >= 1"):
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
    """A running lane may drain under reset freeze, but its queued job
    must remain unsubmitted and is cleared by reset cleanup."""
    cfg = HardwareConfig(mfe_load_channels=1, mfe_pipeline_depth=2)
    eng, tm = self._make_eng(cfg)
    assert (
      eng.launch(self._desc("load", "ld0"), 10, "e0", transaction=self._timing_txn("load", "t0"))
      is not None
    )
    assert (
      eng.launch(self._desc("load", "ld1"), 10, "e1", transaction=self._timing_txn("load", "t1"))
      is not None
    )
    completed = []
    for cycle in range(10, 100):
      tm.step(cycle)
      completed.extend(eng.tick(cycle, start_queued=False))
      if completed:
        break
    assert len(completed) == 1
    lane = eng._load_lanes[0]
    assert lane.running is None
    assert len(lane.queue) == 1
    assert tm.status("t1").value == "pending"  # never submitted
    assert tm.pmu_issued_count == 1
    tm.cancel_all(cycle=cycle)
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
    path.write_text("memory:\n  group_sram:\n    capacity_bytes: 16777216\n", encoding="utf-8")
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
    result = self._run_cli("-w", "pow", "--hw-config", str(tmp_path / "nope.yaml"))
    assert result.returncode == 2
    assert "failed to load hardware config" in result.stderr


class TestPR3SignalPolicy:
  """PR 3: signal_policy, task-bound signals, and role-aware release."""

  @staticmethod
  def _make_signal_prog(phases: tuple[str, ...] = ("input_released", "output_ready")) -> TileProgramDefOp:
    prog = TileProgramDefOp(
      "sig_prog",
      [],
      arg_types=[NestTask(), NestBuffer.of([1, 4, 32], "bf16")],
      arg_names=["task", "l2_buf"],
    )
    task_arg, l2_arg = prog.body.block.args
    view = TileSubviewOp(
      l2_arg, task_arg, 0, [0, 0, 0], [1, 4, 32], [1, 1, 1], NestL2View.of([1, 4, 32], "bf16")
    )
    l1 = TileAllocOp([4, 32], "bf16")
    load = TileLoadOp(view.result, l1.result, "e_load")
    ops = [view, l1, load, TileAwaitOp([load.result])]
    for phase in phases:
      ops.append(TileSignalOp(phase, task_arg))
    ops.append(TileReturnOp())
    prog.body.block.add_ops(ops)
    return prog

  @staticmethod
  def _make_context(prog, role, inrel_tag, outready_tag, policy):
    ctx = NestContextOp(
      "sig_ctx", [], arg_types=[NestGlobalMemref.of([1, 4, 32], "bf16")], arg_names=["Y"], placement=1
    )
    y_arg = ctx.body.block.args[0]
    buf = NestAllocOp("l2_buf", role, [1, 4, 32], "bf16", alignment=256)
    src = NestSubviewOp(y_arg, [0, 0, 0], [1, 4, 32], [1, 1, 1], NestGlobalView.of([1, 4, 32], "bf16"))
    pref = NestPrefetchOp(src.result, buf.result, "ev_in")
    tasks = NestTaskRangeOp(0, 1)
    disp = NestDispatchOp(
      "sig_prog",
      tasks.result,
      [buf.result],
      [buf.result],
      "ev_grid",
      inrel_tag,
      outready_tag,
      signal_policy=policy,
      depends_on=[pref.result],
    )
    ops = [buf, src, pref, tasks, disp]
    if role == "in":
      if inrel_tag:
        release = NestReleaseOp(buf.result, depends_on=[disp.input_released])
      else:
        release = NestReleaseOp(buf.result, depends_on=[disp.grid_done])
    else:
      store = NestDMAStoreOp(
        buf.result, src.result, "ev_out", depends_on=([disp.output_ready] if outready_tag else [])
      )
      ops.append(store)
      release = NestReleaseOp(buf.result, depends_on=[store.result])
    ops.extend([release, NestAwaitOp([disp.grid_done]), NestReturnOp()])
    ctx.body.block.add_ops(ops)
    return ctx, disp

  def test_signal_policy_round_trip(self):
    """0/1/2 phase policy custom assembly byte-stable round-trip."""
    for phases, policy, inrel, outready in [
      ((), {}, "", ""),
      (("input_released",), {"input_released": "all_tasks"}, "ev_i", ""),
      (
        ("input_released", "output_ready"),
        {"input_released": "all_tasks", "output_ready": "all_tasks"},
        "ev_i",
        "ev_o",
      ),
    ]:
      prog = self._make_signal_prog(phases)
      ctx, disp = self._make_context(
        prog, "in" if phases == ("input_released",) else "inout", inrel, outready, policy
      )
      if not inrel and not outready:
        # no-L2 identity dispatch: no alloc, no release, empty policy
        prog = make_identity_tile_program()
        tasks = NestTaskRangeOp(0, 1)
        disp = NestDispatchOp(prog.sym_name.data, tasks.result, [], [], "ev_grid", "", "", signal_policy={})
        ctx = NestContextOp(
          "sig_ctx", [tasks, disp, NestAwaitOp([disp.grid_done]), NestReturnOp()], placement=1
        )
      module = ModuleOp([prog, ctx])
      text = print_workload_ir(module)
      reparsed = parse_workload_ir(text, source_name="<rt>")
      assert print_workload_ir(reparsed) == text

  def test_legacy_signal_syntax_rejected(self):
    """Old operand-less tile.signal is a parse error."""
    text = """builtin.module {
  tile.program @p (%task : !nest.task, %l2 : !nest.l2_buffer<1x4x32xbf16>) {
    tile.signal input_released
    tile.return
  }
}
"""
    with pytest.raises(ParseError):
      parse_workload_ir(text, source_name="<legacy>")

  def test_signal_requires_program_task_formal(self):
    """tile.signal operand must be block arg 0."""
    prog = TileProgramDefOp(
      "bad_sig", [], arg_types=[NestTask(), NestBuffer.of([1, 4, 32], "bf16")], arg_names=["task", "l2_buf"]
    )
    _task_arg, l2_arg = prog.body.block.args
    # Second formal is NOT a task; using it as signal operand must fail.
    prog.body.block.add_ops([TileSignalOp("input_released", l2_arg), TileReturnOp()])
    with pytest.raises(VerifyException, match=r"nest.task"):
      verify_workload_ir(
        ModuleOp(
          [
            prog,
            NestContextOp(
              "c",
              [
                NestTaskRangeOp(0, 1),
                NestDispatchOp(
                  "bad_sig",
                  NestTaskRangeOp(0, 1).result,
                  [],
                  [],
                  "g",
                  "i",
                  "",
                  signal_policy={"input_released": "all_tasks"},
                ),
                NestReturnOp(),
              ],
              placement=1,
            ),
          ]
        )
      )

  def test_signal_policy_matches_program_phases(self):
    """Missing declared phase, extra phase, and unsupported mode all reject."""
    prog = self._make_signal_prog(("input_released",))
    # extra undeclared phase in policy
    ctx, _ = self._make_context(
      prog, "inout", "ev_i", "ev_o", {"input_released": "all_tasks", "output_ready": "all_tasks"}
    )
    with pytest.raises(VerifyException, match="signal_policy phases"):
      verify_workload_ir(ModuleOp([prog, ctx]))

  def test_release_dependency_matches_buffer_role(self):
    """Legal in/out/inout release chains pass verification."""
    for role, phases, policy_kwargs, _dep_kind in [
      ("in", ("input_released",), {"input_released": "all_tasks"}, "input"),
      (
        "inout",
        ("input_released", "output_ready"),
        {"input_released": "all_tasks", "output_ready": "all_tasks"},
        "output",
      ),
    ]:
      prog = self._make_signal_prog(phases)
      ctx, _disp = self._make_context(
        prog,
        role,
        "ev_i" if "input_released" in phases else "",
        "ev_o" if "output_ready" in phases else "",
        policy_kwargs,
      )
      verify_workload_ir(ModuleOp([prog, ctx]))

  def test_release_rejects_wrong_phase_or_grid(self):
    """grid_done dependency, missing release, wrong dependency all reject."""
    prog = self._make_signal_prog(("input_released", "output_ready"))
    ctx = NestContextOp(
      "bad", [], placement=1, arg_types=[NestGlobalMemref.of([1, 4, 32], "bf16")], arg_names=["Y"]
    )
    y = ctx.body.block.args[0]
    buf = NestAllocOp("l2_buf", "inout", [1, 4, 32], "bf16", alignment=256)
    src = NestSubviewOp(y, [0, 0, 0], [1, 4, 32], [1, 1, 1], NestGlobalView.of([1, 4, 32], "bf16"))
    pref = NestPrefetchOp(src.result, buf.result, "ev_in")
    tasks = NestTaskRangeOp(0, 1)
    disp = NestDispatchOp(
      "sig_prog",
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
    # release depends on grid_done instead of store
    bad_release = NestReleaseOp(buf.result, depends_on=[disp.grid_done])
    ctx.body.block.add_ops(
      [
        buf,
        src,
        pref,
        tasks,
        disp,
        store,
        bad_release,
        NestAwaitOp([disp.grid_done, store.result]),
        NestReturnOp(),
      ]
    )
    with pytest.raises(VerifyException, match="must depend only on"):
      verify_workload_ir(ModuleOp([prog, ctx]))

  def test_lowering_preserves_dispatch_and_release_dtos(self):
    """Lowered ExecDispatchRequest/ExecReleaseRequest carry exact
    ordinals, policies, consumer ordinals, and dependency events."""
    from pipeline_validator.execution_ir import ExecDispatchRequest, ExecReleaseRequest

    task = lower_workload_ir(PowWorkload().module)
    dispatch_actions = [a for a in task.actions if a.op == ExecGroupActionOp.DISPATCH_ROLE]
    release_actions = [a for a in task.actions if a.op == ExecGroupActionOp.RELEASE_L2]
    assert len(dispatch_actions) == 4
    assert len(release_actions) == 4
    for i, a in enumerate(dispatch_actions):
      req = a.args[0]
      assert isinstance(req, ExecDispatchRequest)
      assert req.dispatch_ordinal == i
      assert req.signal_policy.input_released == "all_tasks"
      assert req.signal_policy.output_ready == "all_tasks"
    for a in release_actions:
      req = a.args[0]
      assert isinstance(req, ExecReleaseRequest)
      assert req.buffer_role == "inout"
      assert len(req.consumer_dispatch_ordinals) == 1

  def test_release_after_return_rejected(self):
    """A nest.release after nest.return is rejected (plan §1)."""
    prog = self._make_signal_prog(("input_released", "output_ready"))
    ctx = NestContextOp(
      "post_return", [], placement=1, arg_types=[NestGlobalMemref.of([1, 4, 32], "bf16")], arg_names=["Y"]
    )
    y = ctx.body.block.args[0]
    buf = NestAllocOp("l2_buf", "inout", [1, 4, 32], "bf16", alignment=256)
    src = NestSubviewOp(y, [0, 0, 0], [1, 4, 32], [1, 1, 1], NestGlobalView.of([1, 4, 32], "bf16"))
    pref = NestPrefetchOp(src.result, buf.result, "ev_in")
    tasks = NestTaskRangeOp(0, 1)
    disp = NestDispatchOp(
      "sig_prog",
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
    # release appears AFTER nest.return
    ctx.body.block.add_ops(
      [
        buf,
        src,
        pref,
        tasks,
        disp,
        store,
        NestAwaitOp([disp.grid_done, store.result]),
        NestReturnOp(),
        NestReleaseOp(buf.result, depends_on=[store.result]),
      ]
    )
    with pytest.raises(VerifyException, match="before nest.return"):
      verify_workload_ir(ModuleOp([prog, ctx]))

  def test_release_rejects_out_inout_without_outs_producer(self):
    """out/inout buffer with no dispatch in outs lacks a matching
    producer; release must be rejected (plan §1)."""
    prog = self._make_signal_prog(("input_released", "output_ready"))
    ctx = NestContextOp(
      "no_producer", [], placement=1, arg_types=[NestGlobalMemref.of([1, 4, 32], "bf16")], arg_names=["Y"]
    )
    y = ctx.body.block.args[0]
    buf = NestAllocOp("l2_buf", "inout", [1, 4, 32], "bf16", alignment=256)
    src = NestSubviewOp(y, [0, 0, 0], [1, 4, 32], [1, 1, 1], NestGlobalView.of([1, 4, 32], "bf16"))
    pref = NestPrefetchOp(src.result, buf.result, "ev_in")
    store = NestDMAStoreOp(buf.result, src.result, "ev_out")
    ctx.body.block.add_ops(
      [
        buf,
        src,
        pref,
        store,
        NestReleaseOp(buf.result, depends_on=[store.result]),
        NestAwaitOp([store.result]),
        NestReturnOp(),
      ]
    )
    with pytest.raises(VerifyException, match="lacks a matching"):
      verify_workload_ir(ModuleOp([prog, ctx]))
