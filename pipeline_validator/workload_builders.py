"""Workload builders: direct xDSL construction of function-call style IR.

Every builder constructs the reference.mlir-style IR directly from
``pipeline_validator.dialects.elenor`` xDSL operations.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp

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
  NestSubviewOp,
  NestTask,
  NestTaskRangeOp,
  TileAllocOp,
  TileAwaitOp,
  TileLoadOp,
  TilePowOp,
  TileProgramDefOp,
  TileReturnOp,
  TileSignalOp,
  TileStoreOp,
  TileSubviewOp,
)


def make_pow_tile_program(
  name: str = "pow_4k_tile",
  chunk_bytes: int = 32768,
  exponent: int = 2,
  pow_ops: int = 65536,
) -> TileProgramDefOp:
  """Build a per-task L2-to-L1 pow tile program."""
  rows = chunk_bytes // (2 * 128)
  prog = TileProgramDefOp(
    name,
    arg_types=[NestTask(), NestBuffer.of([4, rows, 128], "bf16")],
    arg_names=["task", "l2_buf"],
  )
  task_arg, l2_buf = prog.body.block.args
  l2_view = TileSubviewOp(
    l2_buf,
    task_arg,
    0,
    [0, 0, 0],
    [1, rows, 128],
    [1, 1, 1],
    NestL2View.of([1, rows, 128], "bf16"),
  )
  l1 = TileAllocOp([rows, 128], "bf16", alignment=256)
  load = TileLoadOp(l2_view.result, l1.result, "e_load")
  pow_op = TilePowOp(chunk_bytes, exponent, pow_ops, "e_pow")
  store = TileStoreOp(l1.result, l2_view.result, "e_store")
  prog.body.block.add_ops(
    [
      l2_view,
      l1,
      load,
      TileAwaitOp([load.result]),
      TileSignalOp("input_released", task_arg),
      pow_op,
      TileAwaitOp([pow_op.result]),
      store,
      TileAwaitOp([store.result]),
      TileSignalOp("output_ready", task_arg),
      TileReturnOp(),
    ]
  )
  return prog


def make_identity_tile_program() -> TileProgramDefOp:
  """Build a tile program that only participates in role dispatch."""
  return TileProgramDefOp(
    "identity_tile",
    [TileReturnOp()],
    arg_types=[NestTask()],
    arg_names=["task"],
  )


def make_pow_task(num_group_chunks: int = 4) -> ModuleOp:
  """Build a standalone EVU pow workload with addressable global input."""
  chunk_bytes = 128 * 128 * 2
  buffer_shape = [4, 128, 128]
  global_shape = [4 * num_group_chunks, 128, 128]
  prog = make_pow_tile_program(name="pow_4k_tile", chunk_bytes=chunk_bytes)
  ctx = NestContextOp(
    "pow_task",
    placement=0x0F,
    arg_types=[NestGlobalMemref.of(global_shape, "bf16")],
    arg_names=["Y"],
  )
  global_input = ctx.body.block.args[0]
  tasks = NestTaskRangeOp(from_task=0, to_task=4)
  body: list = []

  # Buffers + subviews + prefetches for all chunks (up-front, as before).
  bufs = []
  sources = []
  prefetches = []
  for group in range(num_group_chunks):
    buffer = NestAllocOp(
      slot=f"l2_buf_pow{group}",
      role="inout",
      shape=buffer_shape,
      dtype="bf16",
      alignment=256,
    )
    source = NestSubviewOp(
      global_input,
      [4 * group, 0, 0],
      buffer_shape,
      [1, 1, 1],
      NestGlobalView.of(buffer_shape, "bf16"),
    )
    prefetch = NestPrefetchOp(source.result, buffer.result, f"ev_dma_pow_in{group}")
    bufs.append(buffer)
    sources.append(source)
    prefetches.append(prefetch)
    body.extend([buffer, source, prefetch])

  # Task range: each chunk dispatch covers 4 logical tasks (one per tile).
  body.append(tasks)

  # Dispatch per chunk (depends_on prefetch).
  dispatches = []
  for group in range(num_group_chunks):
    dispatch = NestDispatchOp(
      "pow_4k_tile",
      tasks.result,
      [bufs[group].result],
      [bufs[group].result],
      f"ev_role_pow{group}",
      f"ev_inrel_pow{group}",
      f"ev_outready_pow{group}",
      signal_policy={
        "input_released": "all_tasks",
        "output_ready": "all_tasks",
      },
      depends_on=[prefetches[group].result],
    )
    dispatches.append(dispatch)
    body.append(dispatch)

  # Store per chunk (depends_on output_ready).
  stores = []
  for group in range(num_group_chunks):
    store = NestDMAStoreOp(
      bufs[group].result,
      sources[group].result,
      f"ev_dma_pow_out{group}",
      depends_on=[dispatches[group].output_ready],
    )
    stores.append(store)
    body.append(store)

  # Await grid_done + store for each chunk, then release the buffer.
  for group in range(num_group_chunks):
    body.append(NestAwaitOp([dispatches[group].grid_done, stores[group].result]))
    body.append(NestReleaseOp(bufs[group].result, depends_on=[stores[group].result]))
  ctx.body.block.add_ops(body)
  return ModuleOp([prog, ctx])
