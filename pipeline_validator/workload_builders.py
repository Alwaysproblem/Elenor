"""Workload builders: direct xDSL construction of function-call style IR.

Every builder constructs the reference.mlir-style IR directly from
``pipeline_validator.dialects.elenor`` xDSL operations.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp

from pipeline_validator.dialects.elenor import (
  NestAllocOp,
  NestAwaitOp,
  NestContextOp,
  NestDispatchOp,
  NestDMAStoreOp,
  NestPrefetchOp,
  NestReleaseOp,
  NestReturnOp,
  NestTaskRangeOp,
  TileAwaitOp,
  TileLoadOp,
  TilePowOp,
  TileProgramDefOp,
  TileReturnOp,
  TileSignalOp,
  TileStoreOp,
)


def make_pow_tile_program(
  name: str = "pow_4k_tile",
  chunk_bytes: int = 32768,
  exponent: int = 2,
  pow_ops: int = 65536,
) -> TileProgramDefOp:
  """EVU elementwise pow on one tile chunk (reference.mlir tile program).

  load input L2->L1 (MFE) -> signal input_released -> EVU pow -> store
  output L1->L2 (MFE) -> signal output_ready -> return.
  """
  load = TileLoadOp(bytes_total=chunk_bytes, tag="e_load")
  pow_op = TilePowOp(bytes_total=chunk_bytes, exponent=exponent, pow_ops=pow_ops, tag="e_pow")
  store = TileStoreOp(bytes_total=chunk_bytes, tag="e_store")
  body: list = [
    load,
    TileAwaitOp([load.result]),
    TileSignalOp("input_released"),
    pow_op,
    TileAwaitOp([pow_op.result]),
    store,
    TileAwaitOp([store.result]),
    TileSignalOp("output_ready"),
    TileReturnOp(),
  ]
  return TileProgramDefOp(name, body)


def make_identity_tile_program() -> TileProgramDefOp:
  """A tile program that does nothing (for pure role dispatch testing)."""
  return TileProgramDefOp("identity_tile", [TileReturnOp()])


def make_pow_task(num_group_chunks: int = 4) -> ModuleOp:
  """Standalone EVU pow workload with pipelined group DMA.

  Mirrors reference.mlir: one tile program definition + one context.
  Each chunk: alloc L2 buffer -> prefetch HBM->L2 -> dispatch (depends_on
  prefetch) -> store L2->HBM (depends_on output_ready) -> await grid+store.
  """
  chunk_bytes = 128 * 128 * 2
  bytes_pow = chunk_bytes * 4

  prog = make_pow_tile_program(name="pow_4k_tile", chunk_bytes=chunk_bytes)

  # Build context body
  body: list = []

  # Buffers + prefetches for all chunks
  bufs = []
  prefetches = []
  for g in range(num_group_chunks):
    buf = NestAllocOp(slot=f"l2_buf_pow{g}", bytes_total=bytes_pow)
    pref = NestPrefetchOp(
      buffer=buf.result,
      bytes_total=bytes_pow,
      tag=f"ev_dma_pow_in{g}",
    )
    bufs.append(buf)
    prefetches.append(pref)
    body.append(buf)
    body.append(pref)

  # Task range: each chunk dispatch covers 4 logical tasks (one per tile)
  tasks = NestTaskRangeOp(from_task=0, to_task=4)
  body.append(tasks)

  # Dispatch per chunk (depends_on prefetch)
  dispatches = []
  for g in range(num_group_chunks):
    disp = NestDispatchOp(
      program="pow_4k_tile",
      tasks=tasks.result,
      ins=bufs[g].result,
      outs=bufs[g].result,
      grid_tag=f"ev_role_pow{g}",
      inrel_tag=f"ev_inrel_pow{g}",
      outready_tag=f"ev_outready_pow{g}",
      depends_on=[prefetches[g].result],
    )
    dispatches.append(disp)
    body.append(disp)

  # Store per chunk (depends_on output_ready)
  stores = []
  for g in range(num_group_chunks):
    store = NestDMAStoreOp(
      buffer=bufs[g].result,
      bytes_total=bytes_pow,
      tag=f"ev_dma_pow_out{g}",
      depends_on=[dispatches[g].output_ready],
    )
    stores.append(store)
    body.append(store)

  # Await grid_done + store for each chunk, then release the buffer
  for g in range(num_group_chunks):
    body.append(NestAwaitOp([dispatches[g].grid_done, stores[g].result]))
    body.append(NestReleaseOp(buffer=bufs[g].result, depends_on=[stores[g].result]))
  body.append(NestReturnOp())
  context = NestContextOp("pow_task", body, placement=0x0F)
  return ModuleOp([prog, context])
