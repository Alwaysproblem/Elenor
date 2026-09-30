"""Generate the single-request, four-partition Transformer decode workload.

The 2048-token history is split into four independent two-block partitions.
Each partition has its own per-KV-head online-softmax state and unnormalized
output, feeding a stable four-way merge. Its first block pins UCE context
``p % R`` and its continuation rotates to ``(p + 1) % R``; output_ready
orders the recurrence across that handoff. Each partition owns one bounded
K/V staging slot, reused after input_released. Append K and V are independent
dispatches, allowing K to run before the WV prefetch completes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformer_common import contract_bytes, l2_view, write_workload
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import OpResult

from pipeline_validator.compiler.resources import conservative_arena_bytes
from pipeline_validator.config import HardwareConfig
from pipeline_validator.dialects.elenor import (
  NestAllocOp,
  NestAwaitOp,
  NestBuffer,
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
  TileAllocOp,
  TileAwaitOp,
  TileBoaOp,
  TileEvuOp,
  TileFreeOp,
  TileLoadOp,
  TileProgramDefOp,
  TileReturnOp,
  TileSignalOp,
  TileStoreOp,
  TileSubviewOp,
)
from pipeline_validator.profiles import ContextResources, TileResources, build_registry

PARTITIONS = 4
SEQ_LEN = 2048
KV_BLOCK = 256
KV_HEADS = 4
Q_HEADS = 16
HEADS_PER_KV = Q_HEADS // KV_HEADS
HEAD_DIM = 64
HIDDEN = 1024
BLOCKS = SEQ_LEN // KV_BLOCK
PLACEMENT = 0x0F
PARTITION_SUFFIXES = ("pa", "pb", "pc", "pd")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "examples/workloads/transformer_decode_kv_multicontext.mlir"


def _l1_contract(
  hw: HardwareConfig, buffers: list[tuple[int, int]], contexts_per_tile: int
) -> tuple[int, tuple[int, ...]]:
  """Derive an L1 contract whose every allowed mode covers the R envelope."""
  registry = build_registry(hw)
  reservations = {mode: conservative_arena_bytes(buffers, profile) for mode, profile in registry.l1.items()}
  reserved = max(reservations.values())
  allowed = tuple(
    mode
    for mode, profile in registry.l1.items()
    if contexts_per_tile * ((reserved + profile.banks - 1) // profile.banks) <= profile.user_spm_per_bank
  )
  if registry.target.l1.reset_mode not in allowed:
    raise ValueError(
      f"L1 mode {registry.target.l1.reset_mode} cannot satisfy R={contexts_per_tile} "
      f"with authored Arena reservation {reserved}"
    )
  return reserved, allowed


def _decode_program(hw: HardwareConfig, contexts_per_tile: int) -> TileProgramDefOp:
  """One KV block update of a partition's (m,l) and unnormalized output."""
  l1_bytes, allowed = _l1_contract(
    hw,
    [
      (KV_BLOCK * HEAD_DIM * 2, 256),  # K block
      (KV_BLOCK * HEAD_DIM * 2, 256),  # V block
      (HEADS_PER_KV * HEAD_DIM * 2, 256),  # Q
      (HEADS_PER_KV * 2 * 4, 64),  # per-head FP32 (m,l)
      (HEADS_PER_KV * HEAD_DIM * 4, 64),  # unnormalized FP32 output
      (HEADS_PER_KV * KV_BLOCK * 4, 64),  # FP32 score scratch
      (HEADS_PER_KV * HEAD_DIM * 4, 64),  # FP32 PV scratch
    ],
    contexts_per_tile,
  )
  program = TileProgramDefOp(
    "decode_partition_block",
    TileResources(allowed_profiles=allowed, tile_l1_spm_bytes_per_context=l1_bytes),
    arg_types=[
      NestTask(),
      NestBuffer.of([KV_HEADS, HEADS_PER_KV, HEAD_DIM], "bf16"),
      NestBuffer.of([KV_HEADS, KV_BLOCK, HEAD_DIM], "bf16"),
      NestBuffer.of([KV_HEADS, KV_BLOCK, HEAD_DIM], "bf16"),
      NestBuffer.of([KV_HEADS, HEADS_PER_KV, 2], "f32"),
      NestBuffer.of([KV_HEADS, HEADS_PER_KV, HEAD_DIM], "f32"),
    ],
    arg_names=["task", "q_l2", "k_l2", "v_l2", "state_l2", "partial_out_l2"],
  )
  body = program.body.block
  task = body.args[0]
  q_l2, k_l2, v_l2, state_l2, partial_out_l2 = body.args[1:]

  def task_view(src, shape, dtype):
    full_shape = [1, *shape]
    return TileSubviewOp(src, task, 0, [0, 0, 0], full_shape, [1, 1, 1], NestL2View.of(full_shape, dtype))

  q_view = task_view(q_l2, [HEADS_PER_KV, HEAD_DIM], "bf16")
  k_view = task_view(k_l2, [KV_BLOCK, HEAD_DIM], "bf16")
  v_view = task_view(v_l2, [KV_BLOCK, HEAD_DIM], "bf16")
  state_view = task_view(state_l2, [HEADS_PER_KV, 2], "f32")
  out_view = task_view(partial_out_l2, [HEADS_PER_KV, HEAD_DIM], "f32")
  body.add_ops([q_view, k_view, v_view, state_view, out_view])

  q_l1 = TileAllocOp([HEADS_PER_KV, HEAD_DIM], "bf16", alignment=256)
  k_l1 = TileAllocOp([KV_BLOCK, HEAD_DIM], "bf16", alignment=256)
  v_l1 = TileAllocOp([KV_BLOCK, HEAD_DIM], "bf16", alignment=256)
  state_l1 = TileAllocOp([HEADS_PER_KV, 2], "f32", alignment=64)
  out_l1 = TileAllocOp([HEADS_PER_KV, HEAD_DIM], "f32", alignment=64)
  score_l1 = TileAllocOp([HEADS_PER_KV, KV_BLOCK], "f32", alignment=64)
  pv_l1 = TileAllocOp([HEADS_PER_KV, HEAD_DIM], "f32", alignment=64)
  buffers = [q_l1, k_l1, v_l1, state_l1, out_l1, score_l1, pv_l1]
  body.add_ops(buffers)

  loads = [
    TileLoadOp(q_view.result, q_l1.result, "q_loaded"),
    TileLoadOp(k_view.result, k_l1.result, "k_loaded"),
    TileLoadOp(v_view.result, v_l1.result, "v_loaded"),
    TileLoadOp(state_view.result, state_l1.result, "state_loaded"),
    TileLoadOp(out_view.result, out_l1.result, "partial_out_loaded"),
  ]
  body.add_ops(loads)
  body.add_op(TileAwaitOp(loads))
  body.add_op(TileSignalOp("input_released", task))
  qk = TileBoaOp(
    "matmul", HEADS_PER_KV, KV_BLOCK, HEAD_DIM, 2 * HEADS_PER_KV * KV_BLOCK * HEAD_DIM, "qk_boa"
  )
  body.add_ops([qk, TileAwaitOp([qk])])
  score_update = TileEvuOp("online_softmax_score_update", HEADS_PER_KV * (KV_BLOCK + 2), "score_update")
  body.add_ops([score_update, TileAwaitOp([score_update])])
  pv = TileBoaOp(
    "matmul", HEADS_PER_KV, HEAD_DIM, KV_BLOCK, 2 * HEADS_PER_KV * HEAD_DIM * KV_BLOCK, "pv_boa"
  )
  body.add_op(pv)
  body.add_op(TileAwaitOp([pv]))
  accumulate = TileEvuOp(
    "online_softmax_output_accumulate", 2 * HEADS_PER_KV * HEAD_DIM, "output_accumulate"
  )
  body.add_ops([accumulate, TileAwaitOp([accumulate])])

  state_store = TileStoreOp(state_l1.result, state_view.result, "state_stored")
  out_store = TileStoreOp(out_l1.result, out_view.result, "partial_out_stored")
  body.add_ops([state_store, out_store, TileAwaitOp([state_store, out_store])])
  body.add_op(TileSignalOp("output_ready", task))
  body.add_ops([TileFreeOp(buffer) for buffer in buffers])
  body.add_op(TileReturnOp())
  return program


def _merge_program(hw: HardwareConfig, contexts_per_tile: int) -> TileProgramDefOp:
  """Stable max/exp merge of four partition states, including final o/l."""
  l1_specs = [(HEADS_PER_KV * 2 * 4, 64), (HEADS_PER_KV * HEAD_DIM * 4, 64)] * PARTITIONS + [
    (HEADS_PER_KV * PARTITIONS * 4, 64),  # alpha scratch
    (HEADS_PER_KV * 2 * 4, 64),  # merged (m,l) scratch
    (HEADS_PER_KV * HEAD_DIM * 4, 64),  # merged output
  ]
  l1_bytes, allowed = _l1_contract(hw, l1_specs, contexts_per_tile)
  arg_types: list[NestTask | NestBuffer] = [NestTask()]
  arg_names = ["task"]
  for suffix in PARTITION_SUFFIXES:
    arg_types.extend(
      [
        NestBuffer.of([KV_HEADS, HEADS_PER_KV, 2], "f32"),
        NestBuffer.of([KV_HEADS, HEADS_PER_KV, HEAD_DIM], "f32"),
      ]
    )
    arg_names.extend([f"state_{suffix}", f"partial_out_{suffix}"])
  arg_types.append(NestBuffer.of([KV_HEADS, HEADS_PER_KV, HEAD_DIM], "f32"))
  arg_names.append("merged_out_l2")
  program = TileProgramDefOp(
    "decode_stable_partition_merge",
    TileResources(allowed_profiles=allowed, tile_l1_spm_bytes_per_context=l1_bytes),
    arg_types=arg_types,
    arg_names=arg_names,
  )
  body = program.body.block
  task = body.args[0]
  state_buffers = [body.args[1 + 2 * p] for p in range(PARTITIONS)]
  partial_buffers = [body.args[2 + 2 * p] for p in range(PARTITIONS)]
  merged_out_l2 = body.args[-1]

  def task_view(src, trailing_shape, dtype):
    full_shape = [1, *trailing_shape]
    return TileSubviewOp(src, task, 0, [0, 0, 0], full_shape, [1, 1, 1], NestL2View.of(full_shape, dtype))

  state_views = [task_view(buffer, [HEADS_PER_KV, 2], "f32") for buffer in state_buffers]
  partial_views = [task_view(buffer, [HEADS_PER_KV, HEAD_DIM], "f32") for buffer in partial_buffers]
  merged_view = task_view(merged_out_l2, [HEADS_PER_KV, HEAD_DIM], "f32")
  body.add_ops([*state_views, *partial_views, merged_view])

  state_l1 = [TileAllocOp([HEADS_PER_KV, 2], "f32", alignment=64) for _ in range(PARTITIONS)]
  partial_l1 = [TileAllocOp([HEADS_PER_KV, HEAD_DIM], "f32", alignment=64) for _ in range(PARTITIONS)]
  alpha_l1 = TileAllocOp([HEADS_PER_KV, PARTITIONS], "f32", alignment=64)
  merged_state_l1 = TileAllocOp([HEADS_PER_KV, 2], "f32", alignment=64)
  merged_out_l1 = TileAllocOp([HEADS_PER_KV, HEAD_DIM], "f32", alignment=64)
  buffers = [*state_l1, *partial_l1, alpha_l1, merged_state_l1, merged_out_l1]
  body.add_ops(buffers)

  loads = []
  for p in range(PARTITIONS):
    suffix = PARTITION_SUFFIXES[p]
    loads.extend(
      [
        TileLoadOp(state_views[p].result, state_l1[p].result, f"state_{suffix}_loaded"),
        TileLoadOp(partial_views[p].result, partial_l1[p].result, f"out_{suffix}_loaded"),
      ]
    )
  body.add_ops(loads)
  body.add_op(TileAwaitOp(loads))
  body.add_op(TileSignalOp("input_released", task))

  # Per Q head: 3 max comparisons, 4 exponentials, 4 weighted multiplies +
  # 3 additions for l, and the same 7 operations per output lane plus final
  # normalization. The EVU op is timing-only; it carries no tensor values.
  merge_ops_per_head = 18 + 8 * HEAD_DIM
  merge = TileEvuOp("stable_online_softmax_merge", HEADS_PER_KV * merge_ops_per_head, "stable_merge")
  body.add_ops([merge, TileAwaitOp([merge])])
  store = TileStoreOp(merged_out_l1.result, merged_view.result, "merged_out_stored")
  body.add_ops([store, TileAwaitOp([store])])
  body.add_op(TileSignalOp("output_ready", task))
  body.add_ops([TileFreeOp(buffer) for buffer in buffers])
  body.add_op(TileReturnOp())
  return program


def _append_program(hw: HardwareConfig, contexts_per_tile: int, projection: str) -> TileProgramDefOp:
  """Project one append weight independently so K and V can overlap."""
  l1_bytes, allowed = _l1_contract(
    hw,
    [
      (HIDDEN * 2, 256),  # current-token hidden input
      (HIDDEN * HEAD_DIM * 2, 256),  # one projection's weight slice
      (HEAD_DIM * 2, 256),  # projection result scratch
    ],
    contexts_per_tile,
  )
  program = TileProgramDefOp(
    f"decode_kv_append_{projection}",
    TileResources(allowed_profiles=allowed, tile_l1_spm_bytes_per_context=l1_bytes),
    arg_types=[
      NestTask(),
      NestBuffer.of([HIDDEN], "bf16"),
      NestBuffer.of([KV_HEADS, HIDDEN, HEAD_DIM], "bf16"),
      NestBuffer.of([KV_HEADS, 1, HEAD_DIM], "bf16"),
    ],
    arg_names=["task", "h_l2", "weight_l2", "result_l2"],
  )
  body = program.body.block
  task, h_l2, weight_l2, result_l2 = body.args

  h_view = l2_view(h_l2, None, None, [0], [HIDDEN], [HIDDEN], "bf16")
  weight_view = l2_view(weight_l2, task, 0, [0, 0, 0], [1, HIDDEN, HEAD_DIM], [1, HIDDEN, HEAD_DIM], "bf16")
  result_view = l2_view(result_l2, task, 0, [0, 0, 0], [1, 1, HEAD_DIM], [1, 1, HEAD_DIM], "bf16")
  body.add_ops([h_view, weight_view, result_view])

  h_l1 = TileAllocOp([HIDDEN], "bf16", alignment=256)
  weight_l1 = TileAllocOp([HIDDEN, HEAD_DIM], "bf16", alignment=256)
  result_l1 = TileAllocOp([1, HEAD_DIM], "bf16", alignment=256)
  buffers = [h_l1, weight_l1, result_l1]
  body.add_ops(buffers)

  loads = [
    TileLoadOp(h_view.result, h_l1.result, "h_loaded"),
    TileLoadOp(weight_view.result, weight_l1.result, "weight_loaded"),
  ]
  body.add_ops(loads)
  body.add_op(TileAwaitOp(loads))
  boa = TileBoaOp("matmul", 1, HEAD_DIM, HIDDEN, 2 * HEAD_DIM * HIDDEN, f"{projection}_new_boa")
  body.add_ops([boa, TileAwaitOp([boa])])
  store = TileStoreOp(result_l1.result, result_view.result, f"{projection}_append_stored")
  body.add_ops([store, TileAwaitOp([store])])
  body.add_op(TileSignalOp("input_released", task))
  body.add_op(TileSignalOp("output_ready", task))
  body.add_ops([TileFreeOp(buffer) for buffer in buffers])
  body.add_op(TileReturnOp())
  return program


def make_decode_multicontext(
  *, hw: HardwareConfig | None = None, contexts_per_tile: int = 4, kv_padding_bytes: int = 64
) -> ModuleOp:
  """Build the same single-request decode with independent KV partitions.

  ``contexts_per_tile`` selects R; each partition continuation rotates one UCE
  context from its first block, with output_ready preserving the recurrence.
  ``kv_padding_bytes`` is a physical HBM packet gap, not transferred data:
  rotating packet start channels avoids camping all independent partitions
  on channel zero in the current start-address-selected HBM model.
  """
  if type(contexts_per_tile) is not int or contexts_per_tile not in (1, 2, 4):
    raise ValueError("contexts_per_tile must be one of 1, 2, or 4")
  if type(kv_padding_bytes) is not int or kv_padding_bytes < 0 or kv_padding_bytes % 2:
    raise ValueError("kv_padding_bytes must be a nonnegative multiple of the BF16 element size")
  packet_elements = KV_HEADS * KV_BLOCK * HEAD_DIM
  packet_stride = packet_elements + kv_padding_bytes // 2
  hw = hw or HardwareConfig()
  attention_program = _decode_program(hw, contexts_per_tile)
  merge_program = _merge_program(hw, contexts_per_tile)
  append_k_program = _append_program(hw, contexts_per_tile, "k")
  append_v_program = _append_program(hw, contexts_per_tile, "v")

  l2_specs = (
    [(KV_HEADS * KV_BLOCK * HEAD_DIM * 2, 256)] * (2 * PARTITIONS)  # one K/V slot per partition
    + [(KV_HEADS * HEADS_PER_KV * HEAD_DIM * 2, 256)]  # Q
    + [(KV_HEADS * HEADS_PER_KV * 2 * 4, 64)] * PARTITIONS  # per-partition (m,l)
    + [(KV_HEADS * HEADS_PER_KV * HEAD_DIM * 4, 64)] * PARTITIONS  # partial outputs
    + [(KV_HEADS * HEADS_PER_KV * HEAD_DIM * 4, 64)]  # final OUT staging
    + [(HIDDEN * 2, 256)]  # current-token hidden staging
    + [(KV_HEADS * HIDDEN * HEAD_DIM * 2, 256)] * 2  # WK/WV append staging
    + [(KV_HEADS * HEAD_DIM * 2, 256)] * 2  # K/V append staging
  )
  l2_bytes, allowed_l2 = contract_bytes(hw, "l2", l2_specs)
  logical_tasks = BLOCKS * KV_HEADS + 2 * KV_HEADS + KV_HEADS
  context = NestContextOp(
    "decode_multicontext",
    ContextResources(
      l2_mode=0,
      allowed_profiles=allowed_l2,
      logical_tasks=logical_tasks,
      l2_spm_bytes=l2_bytes,
      requested_contexts_per_tile=contexts_per_tile,
    ),
    placement=PLACEMENT,
    arg_types=[
      NestGlobalMemref.of([BLOCKS, packet_stride], "bf16"),  # K_CACHE packets, including gap
      NestGlobalMemref.of([BLOCKS, packet_stride], "bf16"),  # V_CACHE packets, including gap
      NestGlobalMemref.of([KV_HEADS, HEADS_PER_KV, HEAD_DIM], "bf16"),  # Q_IN
      NestGlobalMemref.of([PARTITIONS, KV_HEADS, HEADS_PER_KV, 2], "f32"),  # S_INIT_STATE
      NestGlobalMemref.of([PARTITIONS, KV_HEADS, HEADS_PER_KV, HEAD_DIM], "f32"),  # S_INIT_OUT
      NestGlobalMemref.of([KV_HEADS, HEADS_PER_KV, HEAD_DIM], "f32"),  # OUT
      NestGlobalMemref.of([HIDDEN], "bf16"),  # H_T
      NestGlobalMemref.of([KV_HEADS, HIDDEN, HEAD_DIM], "bf16"),  # WK_A
      NestGlobalMemref.of([KV_HEADS, HIDDEN, HEAD_DIM], "bf16"),  # WV_A
      NestGlobalMemref.of([KV_HEADS, 1, HEAD_DIM], "bf16"),  # K_APPEND
      NestGlobalMemref.of([KV_HEADS, 1, HEAD_DIM], "bf16"),  # V_APPEND
    ],
    arg_names=[
      "K_CACHE_ctx",
      "V_CACHE_ctx",
      "Q_IN_ctx",
      "S_INIT_STATE_ctx",
      "S_INIT_OUT_ctx",
      "OUT_ctx",
      "H_T_ctx",
      "WK_A_ctx",
      "WV_A_ctx",
      "K_APPEND_ctx",
      "V_APPEND_ctx",
    ],
  )
  (K_CACHE, V_CACHE, Q_IN, S_INIT_STATE, S_INIT_OUT, OUT, H_T, WK_A, WV_A, K_APPEND, V_APPEND) = (
    context.body.block.args
  )
  block = context.body.block

  k_buffers = [
    NestAllocOp(f"k_{suffix}", "in", [KV_HEADS, KV_BLOCK, HEAD_DIM], "bf16", alignment=256)
    for suffix in PARTITION_SUFFIXES
  ]
  v_buffers = [
    NestAllocOp(f"v_{suffix}", "in", [KV_HEADS, KV_BLOCK, HEAD_DIM], "bf16", alignment=256)
    for suffix in PARTITION_SUFFIXES
  ]
  q_buffer = NestAllocOp("q_l2", "in", [KV_HEADS, HEADS_PER_KV, HEAD_DIM], "bf16", alignment=256)
  states = [
    NestAllocOp(
      f"state_{suffix}", "inout", [KV_HEADS, HEADS_PER_KV, 2], "f32", alignment=64, sharing="context-local"
    )
    for suffix in PARTITION_SUFFIXES
  ]
  partial_outputs = [
    NestAllocOp(
      f"partial_out_{suffix}",
      "inout",
      [KV_HEADS, HEADS_PER_KV, HEAD_DIM],
      "f32",
      alignment=64,
      sharing="context-local",
    )
    for suffix in PARTITION_SUFFIXES
  ]
  final_output = NestAllocOp(
    "merged_out_l2", "out", [KV_HEADS, HEADS_PER_KV, HEAD_DIM], "f32", alignment=64
  )
  h_l2 = NestAllocOp("h_l2", "in", [HIDDEN], "bf16", alignment=256)
  wk_l2 = NestAllocOp("wk_l2", "in", [KV_HEADS, HIDDEN, HEAD_DIM], "bf16", alignment=256)
  wv_l2 = NestAllocOp("wv_l2", "in", [KV_HEADS, HIDDEN, HEAD_DIM], "bf16", alignment=256)
  kn_l2 = NestAllocOp("kn_l2", "out", [KV_HEADS, 1, HEAD_DIM], "bf16", alignment=256)
  vn_l2 = NestAllocOp("vn_l2", "out", [KV_HEADS, 1, HEAD_DIM], "bf16", alignment=256)
  allocations = [
    *k_buffers,
    *v_buffers,
    q_buffer,
    *states,
    *partial_outputs,
    final_output,
    h_l2,
    wk_l2,
    wv_l2,
    kn_l2,
    vn_l2,
  ]
  block.add_ops(allocations)

  q_view = NestSubviewOp(
    Q_IN,
    [0, 0, 0],
    [KV_HEADS, HEADS_PER_KV, HEAD_DIM],
    [1, 1, 1],
    NestGlobalView.of([KV_HEADS, HEADS_PER_KV, HEAD_DIM], "bf16"),
  )
  init_state_views = []
  init_out_views = []
  for p in range(PARTITIONS):
    init_state_views.append(
      NestSubviewOp(
        S_INIT_STATE,
        [p, 0, 0, 0],
        [1, KV_HEADS, HEADS_PER_KV, 2],
        [1, 1, 1, 1],
        NestGlobalView.of([1, KV_HEADS, HEADS_PER_KV, 2], "f32"),
      )
    )
    init_out_views.append(
      NestSubviewOp(
        S_INIT_OUT,
        [p, 0, 0, 0],
        [1, KV_HEADS, HEADS_PER_KV, HEAD_DIM],
        [1, 1, 1, 1],
        NestGlobalView.of([1, KV_HEADS, HEADS_PER_KV, HEAD_DIM], "f32"),
      )
    )
  merged_out_view = NestSubviewOp(
    OUT,
    [0, 0, 0],
    [KV_HEADS, HEADS_PER_KV, HEAD_DIM],
    [1, 1, 1],
    NestGlobalView.of([KV_HEADS, HEADS_PER_KV, HEAD_DIM], "f32"),
  )
  h_view = NestSubviewOp(H_T, [0], [HIDDEN], [1], NestGlobalView.of([HIDDEN], "bf16"))
  wk_view = NestSubviewOp(
    WK_A,
    [0, 0, 0],
    [KV_HEADS, HIDDEN, HEAD_DIM],
    [1, 1, 1],
    NestGlobalView.of([KV_HEADS, HIDDEN, HEAD_DIM], "bf16"),
  )
  wv_view = NestSubviewOp(
    WV_A,
    [0, 0, 0],
    [KV_HEADS, HIDDEN, HEAD_DIM],
    [1, 1, 1],
    NestGlobalView.of([KV_HEADS, HIDDEN, HEAD_DIM], "bf16"),
  )
  k_append_view = NestSubviewOp(
    K_APPEND,
    [0, 0, 0],
    [KV_HEADS, 1, HEAD_DIM],
    [1, 1, 1],
    NestGlobalView.of([KV_HEADS, 1, HEAD_DIM], "bf16"),
  )
  v_append_view = NestSubviewOp(
    V_APPEND,
    [0, 0, 0],
    [KV_HEADS, 1, HEAD_DIM],
    [1, 1, 1],
    NestGlobalView.of([KV_HEADS, 1, HEAD_DIM], "bf16"),
  )
  block.add_ops(
    [
      q_view,
      *init_state_views,
      *init_out_views,
      merged_out_view,
      h_view,
      wk_view,
      wv_view,
      k_append_view,
      v_append_view,
    ]
  )
  tasks = NestTaskRangeOp(0, KV_HEADS)
  block.add_op(tasks)

  pre_q = NestPrefetchOp(q_view, q_buffer, "pre_q")
  block.add_op(pre_q)
  pre_state = []
  pre_out = []
  for p, suffix in enumerate(PARTITION_SUFFIXES):
    state_prefetch = NestPrefetchOp(init_state_views[p], states[p], f"pre_state_{suffix}")
    out_prefetch = NestPrefetchOp(init_out_views[p], partial_outputs[p], f"pre_out_{suffix}")
    block.add_ops([state_prefetch, out_prefetch])
    pre_state.append(state_prefetch)
    pre_out.append(out_prefetch)

  pre_k: dict[int, NestPrefetchOp] = {}
  pre_v: dict[int, NestPrefetchOp] = {}
  grids: dict[int, OpResult[NestEvent]] = {}
  inrels: dict[int, OpResult[NestEvent]] = {}
  outs: dict[int, OpResult[NestEvent]] = {}
  # Register one ready block from every partition before any recurrence
  # continuation; a blocked continuation must not fill the ready-action
  # window ahead of another partition's independent first block.
  block_order = [2 * p + step for step in range(2) for p in range(PARTITIONS)]
  for block_index in block_order:
    partition = block_index // 2
    reuse_gate = [inrels[block_index - 1]] if block_index % 2 else []
    k_view = NestSubviewOp(
      K_CACHE,
      [block_index, 0],
      [1, packet_elements],
      [1, 1],
      NestGlobalView.of([1, packet_elements], "bf16"),
    )
    v_view = NestSubviewOp(
      V_CACHE,
      [block_index, 0],
      [1, packet_elements],
      [1, 1],
      NestGlobalView.of([1, packet_elements], "bf16"),
    )
    block.add_ops([k_view, v_view])
    pre_k[block_index] = NestPrefetchOp(
      k_view, k_buffers[partition], f"pre_k_block_{block_index}_event", depends_on=reuse_gate
    )
    pre_v[block_index] = NestPrefetchOp(
      v_view, v_buffers[partition], f"pre_v_block_{block_index}_event", depends_on=reuse_gate
    )
    block.add_ops([pre_k[block_index], pre_v[block_index]])

    dependencies: list[NestPrefetchOp | OpResult[NestEvent]] = [
      pre_k[block_index],
      pre_v[block_index],
      pre_q,
      pre_state[partition],
      pre_out[partition],
    ]
    if block_index % 2:
      dependencies.append(outs[block_index - 1])
    dispatch = NestDispatchOp(
      "decode_partition_block",
      tasks,
      [],
      [q_buffer, k_buffers[partition], v_buffers[partition], states[partition], partial_outputs[partition]],
      [states[partition], partial_outputs[partition]],
      f"attn_block_{block_index}_grid",
      f"attn_block_{block_index}_inrel",
      f"attn_block_{block_index}_out",
      l1_mode=0,
      bindings=[
        q_buffer,
        k_buffers[partition],
        v_buffers[partition],
        states[partition],
        partial_outputs[partition],
      ],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=dependencies,
      context_id=(partition + block_index % 2) % contexts_per_tile,
    )
    block.add_op(dispatch)
    grids[block_index] = dispatch.grid_done
    inrels[block_index] = dispatch.input_released
    outs[block_index] = dispatch.output_ready

  pre_h = NestPrefetchOp(h_view, h_l2, "pre_h")
  pre_wk = NestPrefetchOp(wk_view, wk_l2, "pre_wk")
  pre_wv = NestPrefetchOp(wv_view, wv_l2, "pre_wv")
  block.add_ops([pre_h, pre_wk, pre_wv])

  append_k = NestDispatchOp(
    "decode_kv_append_k",
    tasks,
    [],
    [h_l2, wk_l2],
    [kn_l2],
    "append_k_grid",
    "append_k_inrel",
    "append_k_out",
    l1_mode=0,
    bindings=[h_l2, wk_l2, kn_l2],
    signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
    depends_on=[pre_h, pre_wk],
    context_id=0,
  )
  append_v = NestDispatchOp(
    "decode_kv_append_v",
    tasks,
    [],
    [h_l2, wv_l2],
    [vn_l2],
    "append_v_grid",
    "append_v_inrel",
    "append_v_out",
    l1_mode=0,
    bindings=[h_l2, wv_l2, vn_l2],
    signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
    depends_on=[pre_h, pre_wv],
    context_id=1 % contexts_per_tile,
  )
  block.add_ops([append_k, append_v])
  k_append_store = NestDMAStoreOp(
    kn_l2, k_append_view, "k_append_store", depends_on=[append_k.output_ready]
  )
  v_append_store = NestDMAStoreOp(
    vn_l2, v_append_view, "v_append_store", depends_on=[append_v.output_ready]
  )
  block.add_ops([k_append_store, v_append_store])

  merge_dependencies = [outs[2 * p + 1] for p in range(PARTITIONS)]
  merge_bindings = []
  for p in range(PARTITIONS):
    merge_bindings.extend([states[p], partial_outputs[p]])
  merge_bindings.append(final_output)
  merge_ins = [*states, *partial_outputs]
  merge = NestDispatchOp(
    "decode_stable_partition_merge",
    tasks,
    [],
    merge_ins,
    [final_output],
    "merge_grid",
    "merge_inrel",
    "merge_out",
    l1_mode=0,
    bindings=merge_bindings,
    signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
    depends_on=merge_dependencies,
  )
  block.add_op(merge)
  output_store = NestDMAStoreOp(
    final_output, merged_out_view, "merged_output_store", depends_on=[merge.output_ready]
  )
  block.add_op(output_store)

  def release(buffer, dependencies) -> None:
    block.add_op(NestReleaseOp(buffer, depends_on=dependencies))

  for p in range(PARTITIONS):
    block_indices = [2 * p, 2 * p + 1]
    release(k_buffers[p], [pre_k[b] for b in block_indices] + [inrels[b] for b in block_indices])
    release(v_buffers[p], [pre_v[b] for b in block_indices] + [inrels[b] for b in block_indices])
  all_attention_inrels = [inrels[b] for b in range(BLOCKS)]
  release(q_buffer, [pre_q, *all_attention_inrels])
  for p in range(PARTITIONS):
    partition_blocks = [2 * p, 2 * p + 1]
    partition_inrels = [inrels[b] for b in partition_blocks]
    partition_outs = [outs[b] for b in partition_blocks]
    shared_state_readers_writers = [*partition_inrels, *partition_outs, merge.input_released]
    release(states[p], [pre_state[p], *shared_state_readers_writers])
    release(partial_outputs[p], [pre_out[p], *shared_state_readers_writers])
  release(final_output, [output_store])
  release(h_l2, [pre_h, append_k.input_released, append_v.input_released])
  release(wk_l2, [pre_wk, append_k.input_released])
  release(wv_l2, [pre_wv, append_v.input_released])
  release(kn_l2, [k_append_store])
  release(vn_l2, [v_append_store])

  block.add_op(
    NestAwaitOp(
      [
        *grids.values(),
        merge.grid_done,
        output_store,
        append_k.grid_done,
        append_v.grid_done,
        k_append_store,
        v_append_store,
      ]
    )
  )
  block.add_op(NestReturnOp())

  public_names = [
    "K_CACHE_mc",
    "V_CACHE_mc",
    "Q_IN_mc",
    "S_INIT_STATE_mc",
    "S_INIT_OUT_mc",
    "OUT_mc",
    "H_T_mc",
    "WK_A_mc",
    "WV_A_mc",
    "K_APPEND_mc",
    "V_APPEND_mc",
  ]
  dev = NexusProgramOp(
    "transformer_decode_kv_multicontext",
    [],
    arg_types=[arg.type for arg in context.body.block.args],
    arg_names=public_names,
  )
  submit = NexusSubmitContextOp(
    "decode_multicontext", "decode_multicontext_done", actuals=list(dev.body.block.args)
  )
  dev.body.block.add_ops([submit, NexusAwaitOp([submit.result]), NexusReturnOp()])
  return ModuleOp([attention_program, merge_program, append_k_program, append_v_program, context, dev])


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Generate the single-request multicontext Transformer decode workload"
  )
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument(
    "--context-mode",
    "--contexts-per-tile",
    dest="contexts_per_tile",
    type=int,
    choices=(1, 2, 4),
    default=4,
    help="requested concurrent Tile Tasks per tile; continuations rotate modulo R",
  )
  parser.add_argument(
    "--kv-padding-bytes",
    type=int,
    default=64,
    help="untransferred gap between HBM KV packets; 0 is the channel-camping control",
  )
  args = parser.parse_args()
  module = make_decode_multicontext(
    contexts_per_tile=args.contexts_per_tile, kv_padding_bytes=args.kv_padding_bytes
  )
  output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
  header = [
    "Single-request Transformer decode multicontext workload (GQA 16:4, BF16 history).",
    "",
    "Shapes: history=2048, KV_BLOCK=256 -> 8 blocks; hidden=1024; head_dim=64.",
    "K/V packets flatten [kv_head,token,dim] into a contiguous BF16 payload.",
    f"Each packet has {args.kv_padding_bytes} untransferred padding bytes after its 128 KiB payload.",
    "Default globals [8,65568]xbf16 occupy 1,049,088 B each but transfer exactly",
    "1 MiB each. With a 64-byte gap, first-block K channels are 0,2,4,6;",
    "binding V_CACHE_mc at base+64 puts its first blocks on channels 1,3,5,7.",
    "This channel-placement optimization is distinct from UCE multicontext:",
    "compare R1/R2/R4 with the same padding/bindings to isolate UCE gains.",
    "Q_IN_mc [4,4,64] is bf16, and OUT_mc [4,4,64] is f32.",
    "S_INIT_STATE_mc [4,4,4,2] and S_INIT_OUT_mc [4,4,4,64] are f32.",
    "Append inputs H_T_mc [1024], WK_A_mc/WV_A_mc [4,1024,64] are bf16;",
    "K_APPEND_mc/V_APPEND_mc [4,1,64] are written at static position 2048.",
    "Each of the 8 attention-block dispatches fans out to all four tiles exactly once.",
    "Tile t owns KV head t and query heads 4t..4t+3. Partition p owns blocks",
    "2p and 2p+1; its first block pins context p%R and its continuation uses",
    "context (p+1)%R. output_ready orders each recurrence across that handoff.",
    "K/V append are two four-task dispatches; K pins context 0 and V pins 1%R.",
    "Partition state and partial-output L2 allocations are distinct.",
    "Context resources count 44 logical tasks: 32 attention, 8 append-phase, and 4 merge tasks.",
    "",
    f"R={args.contexts_per_tile} requested Tile Tasks per tile; continuations rotate modulo R.",
    "Default R=4 explores four UCE contexts; R=1/2 remain available as ablations.",
    f"This source needs sim context_count >= {args.contexts_per_tile}; one Device root is submitted.",
    "The compiler-derived L1 contracts use conservative_arena_bytes and admit",
    "only profiles where R times every per-program per-bank envelope fits.",
    "All task contexts share one context-local L2 Arena; R does not multiply L2.",
    "L2 mode 0 is the baseline; allowed L2 profiles are derived from the full",
    "conservative allocation list and must cover the one shared root Arena.",
    "",
    "Each partition has one block-sized K/V staging slot, so all four first-block",
    "prefetch/dispatch pairs are independent. Its second-block prefetch waits",
    "for that partition's first input_released before overwriting its slot; the",
    "second dispatch waits for its own prior output_ready. State/output buffers",
    "are distinct across partitions; there is no whole-KV prefetch barrier.",
    "K append depends only on H_T/WK and may run while WV is still transferring;",
    "V append depends on H_T/WV. R>=2 pins these projection grids to contexts 0/1.",
    "The merge dispatch depends on all four final partition output_ready events.",
    "The 8 attention, 2 append, and 1 merge dispatches need at most 11 live",
    "Grid routes, below the default group.dispatch_capacity=16.",
    "",
    "Each partition has one S_INIT_STATE row (m=-inf, l=0) and one S_INIT_OUT",
    "row (unnormalized o=0). They are split into contiguous global arrays to",
    "preserve legal row-major views. Stable merge uses m=max(m_i),",
    "alpha_i=exp(m_i-m), l=sum(alpha_i*l_i), o=sum(alpha_i*o_i), then y=o/l.",
    "The merge EVU count is 8,480 ops (16 query heads x [18+8*64]); local",
    "attention EVU work is 49,408 ops across all blocks/tiles, including the",
    "per-block output recurrence. Attention BOA remains 8,388,608 FLOPs, the",
    "same useful QK+PV work as the one-request software-pipeline reference.",
    "The fixed-position-2048 H_T[1024] -> K/V append remains 1,048,576 BOA FLOPs.",
    "Splitting projections rereads H_T once per phase, adding 8,192 L2/local-DMA bytes.",
    "HBM/global payload bytes, output bytes, and all BOA/EVU work remain unchanged.",
    "S_INIT adds 16,896 input bytes; OUT plus K_APPEND/V_APPEND write 5,120 bytes.",
    "",
    "TIMING/RESOURCE MODEL ONLY: BOA and EVU ops carry no tensor values and do",
    "not establish numerical correctness. S_INIT bindings must contain the",
    "documented identity states; runtime data values are not executed here.",
    "No token-generation loop or dynamic append address is modeled.",
    "",
    "Regenerate from the repository root:",
    "  PYTHONPATH=. conda run -n elenor-validator python \\",
    "    examples/generators/generate_transformer_decode_multicontext.py \\",
    f"    --context-mode {args.contexts_per_tile}",
    "Run after run.sh integration: bash examples/run.sh transformer-decode-kv-multicontext",
  ]
  write_workload(output, header, module)
  print(f"wrote {output}")


if __name__ == "__main__":
  main()
