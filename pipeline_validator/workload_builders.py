"""Workload builders: direct xDSL construction of function-call style IR.

Every builder constructs author source IR directly from
``pipeline_validator.dialects.elenor`` xDSL operations.  Resource contracts
are computed at authoring time against the explicitly supplied target; the
runtime never synthesizes them.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp

from pipeline_validator.config import HardwareConfig
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
from pipeline_validator.profiles import ContextResources, TileResources, build_registry


def _contract_bytes(
  hw: HardwareConfig, level: str, byte_alignments: list[tuple[int, int]]
) -> tuple[int, tuple[int, ...]]:
  # Kept local so importing workload builders does not import the compiler.
  from pipeline_validator.compiler.resources import conservative_arena_bytes

  registry = build_registry(hw)
  profiles = getattr(registry, level)
  baseline = registry.profile(level, 0)
  reserved = conservative_arena_bytes(byte_alignments, baseline)
  allowed = tuple(
    mode for mode, profile in profiles.items() if reserved // profile.banks <= profile.user_spm_per_bank
  )
  if 0 not in allowed:
    raise ValueError(f"{level}.mode0 cannot satisfy authored Arena reservation {reserved}")
  return reserved, allowed


def make_pow_tile_program(
  name: str = "pow_4k_tile",
  chunk_bytes: int = 32768,
  exponent: int = 2,
  pow_ops: int = 65536,
  *,
  hw: HardwareConfig | None = None,
) -> TileProgramDefOp:
  """Build a per-task L2-to-L1 pow tile program with an explicit L1 contract."""
  hw = hw or HardwareConfig()
  l1_bytes, allowed_l1 = _contract_bytes(hw, "l1", [(chunk_bytes, 256)])
  rows = chunk_bytes // (2 * 128)
  prog = TileProgramDefOp(
    name,
    TileResources(allowed_profiles=allowed_l1, tile_l1_spm_bytes_per_context=l1_bytes),
    arg_types=[NestTask(), NestBuffer.of([4, rows, 128], "bf16")],
    arg_names=["task", "l2_buf"],
  )
  task_arg, l2_buf = prog.body.block.args
  l2_view = TileSubviewOp(
    l2_buf, task_arg, 0, [0, 0, 0], [1, rows, 128], [1, 1, 1], NestL2View.of([1, rows, 128], "bf16")
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


def make_identity_tile_program(*, hw: HardwareConfig | None = None) -> TileProgramDefOp:
  """Build a tile program with a zero-byte, explicit L1 resource contract."""
  hw = hw or HardwareConfig()
  l1_bytes, allowed_l1 = _contract_bytes(hw, "l1", [])
  return TileProgramDefOp(
    "identity_tile",
    TileResources(allowed_profiles=allowed_l1, tile_l1_spm_bytes_per_context=l1_bytes),
    [TileReturnOp()],
    arg_types=[NestTask()],
    arg_names=["task"],
  )


def make_pow_task(
  num_group_chunks: int = 4, *, hw: HardwareConfig | None = None, context_count: int = 1
) -> ModuleOp:
  """Build a standalone EVU pow workload with addressable global input."""
  if type(num_group_chunks) is not int or num_group_chunks < 1:
    raise ValueError("num_group_chunks must be a positive integer")
  if type(context_count) is not int or context_count < 1:
    raise ValueError("context_count must be a positive integer")
  hw = hw or HardwareConfig()
  chunk_bytes = 128 * 128 * 2
  buffer_shape = [4, 128, 128]
  global_shape = [4 * num_group_chunks, 128, 128]
  l2_buffer_bytes = 4 * chunk_bytes
  l2_bytes, allowed_l2 = _contract_bytes(
    hw, "l2", [(l2_buffer_bytes, 256) for _ in range(num_group_chunks)]
  )
  prog = make_pow_tile_program(name="pow_4k_tile", chunk_bytes=chunk_bytes, hw=hw)
  ctx = NestContextOp(
    "pow_task",
    ContextResources(
      l2_mode=0,
      allowed_profiles=allowed_l2,
      logical_tasks=4 * num_group_chunks,
      l2_spm_bytes=l2_bytes,
      requested_contexts_per_tile=min(num_group_chunks, context_count),
    ),
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
      slot=f"l2_buf_pow{group}", role="inout", shape=buffer_shape, dtype="bf16", alignment=256
    )
    source = NestSubviewOp(
      global_input, [4 * group, 0, 0], buffer_shape, [1, 1, 1], NestGlobalView.of(buffer_shape, "bf16")
    )
    prefetch = NestPrefetchOp(source.result, buffer.result, f"ev_dma_pow_in{group}")
    bufs.append(buffer)
    sources.append(source)
    prefetches.append(prefetch)
    body.extend([buffer, source, prefetch])

  body.append(tasks)

  dispatches = []
  for group in range(num_group_chunks):
    dispatch = NestDispatchOp(
      "pow_4k_tile",
      tasks.result,
      [],
      [bufs[group].result],
      [bufs[group].result],
      f"ev_role_pow{group}",
      f"ev_inrel_pow{group}",
      f"ev_outready_pow{group}",
      l1_mode=0,
      bindings=[bufs[group].result],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=[prefetches[group].result],
    )
    dispatches.append(dispatch)
    body.append(dispatch)

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

  for group in range(num_group_chunks):
    body.append(NestAwaitOp([dispatches[group].grid_done, stores[group].result]))
    body.append(
      NestReleaseOp(
        bufs[group].result,
        depends_on=[dispatches[group].input_released, prefetches[group].result, stores[group].result],
      )
    )
  body.append(NestReturnOp())
  ctx.body.block.add_ops(body)
  return ModuleOp([prog, ctx])
