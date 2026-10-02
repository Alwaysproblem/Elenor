"""Generate the PagedAttention decode workloads and scenario (plan §6).

Two variants of the same decode program:

- ``pipeline``: attention blocks are round-robin across 4 partitions
  (block ``b`` -> partition ``b % 4``); a block dispatch waits only on
  ``output_ready`` of block ``b - 4``; one final merge dispatch folds the
  four partition states/accumulators into OUT.
- ``baseline``: one partition; block ``b`` waits on ``grid_done`` of block
  ``b - 1``; the last block runs the normalize EVU inside the attention
  program and the context stores the accumulator to OUT.

Each (request, step) context prefetches K_NEW/V_NEW, the APPEND_IDS slot,
one BLOCK_TABLE entry per attention block (the last block reuses the
APPEND_IDS slot), Q and the partition state/accumulator into L2.  The
append dispatch (4 tasks, one per KV head) scatters the new K/V token row
into the pool page with ``tile.scatter.global.async``; every attention
block dispatch gathers its page's K/V rows with ``tile.gather.global.async``
directly into L1 and then runs the online-softmax decode timing model.

Host routines ``prepare_r{r}_s{s}`` / ``commit_r{r}_s{s}`` / ``release_r{r}``
manage the page pool and the block table (plan §6); the runner provides the
handlers.  Cross-request ordering is only through the per-request host
chain: prepare(s) -> step(s) -> commit(s) -> prepare(s+1) -> ... ->
release(r); no cross-request barriers.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paged_attention_common as common
from paged_attention_common import (
  APPEND_IDS,
  BLOCK_TABLE,
  GLOBAL_ORDER,
  LENGTHS,
  PARTITIONS,
  Scenario,
)
from transformer_common import contract_bytes, write_workload
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
  NestReturnOp,
  NestSubviewOp,
  NestTask,
  NestTaskRangeOp,
  NexusAwaitOp,
  NexusHostCallOp,
  NexusProgramOp,
  NexusReturnOp,
  NexusSubmitContextOp,
  TileAllocOp,
  TileAwaitOp,
  TileBoaOp,
  TileEvuOp,
  TileFreeOp,
  TileGatherOp,
  TileIndexedMapAttr,
  TileLoadOp,
  TileProgramDefOp,
  TileReturnOp,
  TileScatterOp,
  TileSignalOp,
  TileStoreOp,
  TileSubviewOp,
)
from pipeline_validator.profiles import CacheRequirement, ContextResources, TileResources

# The last attention block reads its page id from the APPEND_IDS slot (the
# token appended this step lives on that page), so the block-table prefetch
# only covers blocks 0..n-2.  Two L2 slots alternate to overlap the index
# prefetch of block b+2 with block b's dispatch.
BLOCK_IDX_SLOTS = 2


def _cache_requirement() -> CacheRequirement:
  """L1/L2 cache capability mirroring the indexed-memory reference scenario."""
  return CacheRequirement(required=False, access="read", bypass="allowed", target_bytes=65536)


def _pool_types(scn: Scenario):
  shape = (scn.physical_pages, scn.page_stride_elements)
  return NestGlobalMemref.of(shape, scn.kv_dtype), NestGlobalView.of(shape, scn.kv_dtype)


def _tile_resources(scn: Scenario, hw, buffers, level: str):
  bytes_reserved, allowed = contract_bytes(hw, level, buffers)
  return TileResources(
    allowed_profiles=allowed,
    tile_l1_spm_bytes_per_context=bytes_reserved,
    l1_cache=_cache_requirement(),
    l2_cache=_cache_requirement(),
  )


# ---------------------------------------------------------------------------
# tile programs
# ---------------------------------------------------------------------------


def build_append_program(scn: Scenario, request: int, tip: int, hw) -> TileProgramDefOp:
  """4-task append dispatch: task t = KV head t scatters its new K and V
  token row into the current pool page (plan §6 map: S=page stride,
  O=token_in_page*D, P=B*D, R=1, T=0, L=D; V's O adds kvh*B*D)."""
  kvh, d = scn.kv_heads, scn.head_dim
  elem = common.DTYPE_BYTES[scn.kv_dtype]
  _, pool_view = _pool_types(scn)
  prog = TileProgramDefOp(
    scn.append_program(request, tip),
    _tile_resources(
      scn, hw, [(1 * d * elem, 256), (1 * d * elem, 256), (4, 64), (4, 64)], "l1"
    ),
    arg_types=[
      NestTask(),
      pool_view,
      NestBuffer.of([kvh, 1, d], scn.kv_dtype),
      NestBuffer.of([kvh, 1, d], scn.kv_dtype),
      NestBuffer.of([1], "i32"),
    ],
    arg_names=["task", "pool", "k_new_l2", "v_new_l2", "append_idx_l2"],
  )
  task, pool, k_new_l2, v_new_l2, append_idx_l2 = prog.body.block.args
  suffix = f"r{request}_tip{tip}"

  k_view = TileSubviewOp(
    k_new_l2, task, 0, [0, 0, 0], [1, 1, d], [1, 1, 1], NestL2View.of([1, 1, d], scn.kv_dtype)
  )
  v_view = TileSubviewOp(
    v_new_l2, task, 0, [0, 0, 0], [1, 1, d], [1, 1, 1], NestL2View.of([1, 1, d], scn.kv_dtype)
  )
  idx_view = TileSubviewOp(append_idx_l2, None, None, [0], [1], [1], NestL2View.of([1], "i32"))
  k_l1 = TileAllocOp([1, d], scn.kv_dtype, alignment=256)
  v_l1 = TileAllocOp([1, d], scn.kv_dtype, alignment=256)
  # Each indexed op owns its index allocation: the verifier treats the
  # index buffer as touched by every indexed op, so the K and V scatters
  # read the same read-only L2 slot through separate L1 indices and stay
  # independent.
  idx_l1_k = TileAllocOp([1], "i32", alignment=64)
  idx_l1_v = TileAllocOp([1], "i32", alignment=64)
  idx_event_k = TileLoadOp(idx_view.result, idx_l1_k.result, f"aidx_k_{suffix}")
  idx_event_v = TileLoadOp(idx_view.result, idx_l1_v.result, f"aidx_v_{suffix}")
  k_event = TileLoadOp(k_view.result, k_l1.result, f"k_row_{suffix}")
  v_event = TileLoadOp(v_view.result, v_l1.result, f"v_row_{suffix}")
  k_map = TileIndexedMapAttr.of(scn.page_stride_elements, tip * d, scn.page_tokens * d, 1, 0, d)
  v_map = TileIndexedMapAttr.of(
    scn.page_stride_elements,
    (kvh * scn.page_tokens + tip) * d,
    scn.page_tokens * d,
    1,
    0,
    d,
  )
  k_scatter = TileScatterOp(
    k_l1.result,
    idx_l1_k.result,
    pool,
    f"scatter_k_{suffix}",
    address_map=k_map,
    window_entries=1,
    scope=scn.scope(request),
  )
  v_scatter = TileScatterOp(
    v_l1.result,
    idx_l1_v.result,
    pool,
    f"scatter_v_{suffix}",
    address_map=v_map,
    window_entries=1,
    scope=scn.scope(request),
  )
  prog.body.block.add_ops(
    [
      k_view,
      v_view,
      idx_view,
      k_l1,
      v_l1,
      idx_l1_k,
      idx_l1_v,
      idx_event_k,
      idx_event_v,
      k_event,
      v_event,
      TileAwaitOp([idx_event_k, idx_event_v, k_event, v_event]),
      TileSignalOp("input_released", task),
      k_scatter,
      v_scatter,
      TileAwaitOp([k_scatter, v_scatter]),
      TileFreeOp(k_l1),
      TileFreeOp(v_l1),
      TileFreeOp(idx_l1_k),
      TileFreeOp(idx_l1_v),
      TileReturnOp(),
    ]
  )
  return prog


def build_attention_program(
  scn: Scenario, request: int, tokens: int, final: bool, hw
) -> TileProgramDefOp:
  """4-task attention dispatch over one block of ``tokens`` valid rows
  (task t = KV head t): load the block page id into L1, gather K/V rows
  into L1 with task_stride=B*D, then the decode online-softmax timing
  model; final blocks append the normalize EVU (plan §6)."""
  kvh, hpk, d = scn.kv_heads, scn.heads_per_kv, scn.head_dim
  elem = common.DTYPE_BYTES[scn.kv_dtype]
  _, pool_view = _pool_types(scn)
  prog = TileProgramDefOp(
    scn.attention_program(request, tokens, final),
    _tile_resources(
      scn,
      hw,
      [
        (4, 64),
        (4, 64),
        (tokens * d * elem, 256),
        (tokens * d * elem, 256),
        (hpk * d * elem, 256),
        (hpk * 2 * 4, 64),
        (hpk * d * 4, 64),
      ],
      "l1",
    ),
    arg_types=[
      NestTask(),
      pool_view,
      NestBuffer.of([1], "i32"),
      NestBuffer.of([kvh, hpk, d], scn.kv_dtype),
      NestBuffer.of([kvh, hpk, 2], scn.state_dtype),
      NestBuffer.of([kvh, hpk, d], scn.state_dtype),
    ],
    arg_names=["task", "pool", "block_idx_l2", "q_l2", "state_l2", "acc_l2"],
  )
  task, pool, block_idx_l2, q_l2, state_l2, acc_l2 = prog.body.block.args
  suffix = f"r{request}_t{tokens}{'_final' if final else ''}"

  idx_view = TileSubviewOp(block_idx_l2, None, None, [0], [1], [1], NestL2View.of([1], "i32"))
  q_view = TileSubviewOp(
    q_l2, task, 0, [0, 0, 0], [1, hpk, d], [1, 1, 1], NestL2View.of([1, hpk, d], scn.kv_dtype)
  )
  state_view = TileSubviewOp(
    state_l2, task, 0, [0, 0, 0], [1, hpk, 2], [1, 1, 1], NestL2View.of([1, hpk, 2], scn.state_dtype)
  )
  acc_view = TileSubviewOp(
    acc_l2, task, 0, [0, 0, 0], [1, hpk, d], [1, 1, 1], NestL2View.of([1, hpk, d], scn.state_dtype)
  )
  # One index allocation per gather: the verifier treats the index buffer as
  # touched by each indexed op, so K and V gather through separate L1
  # indices loaded from the same read-only L2 slot.
  idx_l1_k = TileAllocOp([1], "i32", alignment=64)
  idx_l1_v = TileAllocOp([1], "i32", alignment=64)
  k_l1 = TileAllocOp([tokens, d], scn.kv_dtype, alignment=256)
  v_l1 = TileAllocOp([tokens, d], scn.kv_dtype, alignment=256)
  q_l1 = TileAllocOp([hpk, d], scn.kv_dtype, alignment=256)
  state_l1 = TileAllocOp([hpk, 2], scn.state_dtype, alignment=64)
  acc_l1 = TileAllocOp([hpk, d], scn.state_dtype, alignment=64)
  idx_event_k = TileLoadOp(idx_view.result, idx_l1_k.result, f"bidx_k_{suffix}")
  idx_event_v = TileLoadOp(idx_view.result, idx_l1_v.result, f"bidx_v_{suffix}")
  q_event = TileLoadOp(q_view.result, q_l1.result, f"q_{suffix}")
  state_event = TileLoadOp(state_view.result, state_l1.result, f"state_{suffix}")
  acc_event = TileLoadOp(acc_view.result, acc_l1.result, f"acc_{suffix}")
  # The gather consumes the L1 index, so the index load completes first
  # (workload_ir L1 order rule for gather indices).
  k_map = TileIndexedMapAttr.of(
    scn.page_stride_elements, 0, scn.page_tokens * d, 1, 0, tokens * d
  )
  v_map = TileIndexedMapAttr.of(
    scn.page_stride_elements,
    kvh * scn.page_tokens * d,
    scn.page_tokens * d,
    1,
    0,
    tokens * d,
  )
  k_gather = TileGatherOp(
    pool,
    idx_l1_k.result,
    k_l1.result,
    f"gather_k_{suffix}",
    address_map=k_map,
    window_entries=1,
    scope=scn.scope(request),
  )
  v_gather = TileGatherOp(
    pool,
    idx_l1_v.result,
    v_l1.result,
    f"gather_v_{suffix}",
    address_map=v_map,
    window_entries=1,
    scope=scn.scope(request),
  )
  qk = TileBoaOp("matmul", hpk, tokens, d, 2 * hpk * tokens * d, f"qk_{suffix}")
  sm = TileEvuOp("online_softmax_update", 4 * (tokens + 2), f"sm_{suffix}")
  pv = TileBoaOp("matmul", hpk, d, tokens, 2 * hpk * d * tokens, f"pv_{suffix}")
  acc_evu = TileEvuOp(
    "online_softmax_output_accumulate", 2 * hpk * d, f"acc_evu_{suffix}"
  )
  body: list = [
    idx_view,
    q_view,
    state_view,
    acc_view,
    idx_l1_k,
    idx_l1_v,
    k_l1,
    v_l1,
    q_l1,
    state_l1,
    acc_l1,
    idx_event_k,
    idx_event_v,
    q_event,
    state_event,
    acc_event,
    TileAwaitOp([idx_event_k, idx_event_v]),
    k_gather,
    v_gather,
    TileAwaitOp([k_gather, v_gather, q_event, state_event, acc_event]),
    # Q/state/acc loads and the two gathers complete before the first
    # compute op.
    TileSignalOp("input_released", task),
    qk,
    TileAwaitOp([qk]),
    sm,
    TileAwaitOp([sm]),
    pv,
    TileAwaitOp([pv]),
    acc_evu,
    TileAwaitOp([acc_evu]),
  ]
  if final:
    norm = TileEvuOp("normalize_attention_output", hpk * d, f"norm_{suffix}")
    body += [norm, TileAwaitOp([norm])]
  state_store = TileStoreOp(state_l1.result, state_view.result, f"state_store_{suffix}")
  acc_store = TileStoreOp(acc_l1.result, acc_view.result, f"acc_store_{suffix}")
  body += [
    state_store,
    acc_store,
    TileAwaitOp([state_store, acc_store]),
    TileSignalOp("output_ready", task),
  ]
  body += [
    TileFreeOp(buf)
    for buf in (idx_l1_k, idx_l1_v, k_l1, v_l1, q_l1, state_l1, acc_l1)
  ]
  body.append(TileReturnOp())
  prog.body.block.add_ops(body)
  return prog


def build_merge_program(scn: Scenario, hw, partitions: int = PARTITIONS) -> TileProgramDefOp:
  """Pipeline merge dispatch: fold the partition states and accumulators
  into OUT with one ``(18+8*D)``-op EVU per partition including the final
  normalize (plan §6).  Short sequences use fewer than four partitions."""
  kvh, hpk, d = scn.kv_heads, scn.heads_per_kv, scn.head_dim
  buffers = [(hpk * 2 * 4, 64)] * partitions + [(hpk * d * 4, 64)] * (partitions + 1)
  prog = TileProgramDefOp(
    scn.merge_program(partitions),
    _tile_resources(scn, hw, buffers, "l1"),
    arg_types=[NestTask()]
    + [NestBuffer.of([kvh, hpk, 2], scn.state_dtype) for _ in range(partitions)]
    + [NestBuffer.of([kvh, hpk, d], scn.state_dtype) for _ in range(partitions + 1)],
    arg_names=["task"]
    + [f"state_p{p}" for p in range(partitions)]
    + [f"acc_p{p}" for p in range(partitions)]
    + ["out_l2"],
  )
  args = prog.body.block.args
  task = args[0]
  state_l2 = list(args[1 : 1 + partitions])
  acc_l2 = list(args[1 + partitions : 1 + 2 * partitions])
  out_l2 = args[1 + 2 * partitions]
  suffix = "merge"

  state_views = [
    TileSubviewOp(
      buf, task, 0, [0, 0, 0], [1, hpk, 2], [1, 1, 1], NestL2View.of([1, hpk, 2], scn.state_dtype)
    )
    for buf in state_l2
  ]
  acc_views = [
    TileSubviewOp(
      buf, task, 0, [0, 0, 0], [1, hpk, d], [1, 1, 1], NestL2View.of([1, hpk, d], scn.state_dtype)
    )
    for buf in acc_l2
  ]
  out_view = TileSubviewOp(
    out_l2, task, 0, [0, 0, 0], [1, hpk, d], [1, 1, 1], NestL2View.of([1, hpk, d], scn.state_dtype)
  )
  state_l1 = [TileAllocOp([hpk, 2], scn.state_dtype, alignment=64) for _ in range(partitions)]
  acc_l1 = [TileAllocOp([hpk, d], scn.state_dtype, alignment=64) for _ in range(partitions)]
  events = []
  for p in range(partitions):
    events.append(TileLoadOp(state_views[p].result, state_l1[p].result, f"mstate_p{p}_{suffix}"))
    events.append(TileLoadOp(acc_views[p].result, acc_l1[p].result, f"macc_p{p}_{suffix}"))
  merge = TileEvuOp("online_softmax_merge", partitions * (18 + 8 * d), f"merge_{suffix}")
  # BOA/EVU only time the work (no Attention numerics).  The merge therefore
  # folds the partitions through a loaded, initialised L1 view (the last
  # partition accumulator) and hands that view to OUT, so no store ever reads
  # uninitialised L1 bytes.
  merge_out_l1 = TileAllocOp([hpk, d], scn.state_dtype, alignment=64)
  merge_load = TileLoadOp(acc_views[-1].result, merge_out_l1.result, f"mout_{suffix}")
  copy = TileEvuOp("normalize", partitions * d, f"norm_{suffix}")
  store = TileStoreOp(merge_out_l1.result, out_view.result, f"out_{suffix}")
  prog.body.block.add_ops(
    [
      *state_views,
      *acc_views,
      out_view,
      *state_l1,
      *acc_l1,
      merge_out_l1,
      *events,
      merge_load,
      TileAwaitOp([*events, merge_load.result]),
      TileSignalOp("input_released", task),
      merge,
      TileAwaitOp([merge]),
      copy,
      TileAwaitOp([copy]),
      store,
      TileAwaitOp([store]),
      TileSignalOp("output_ready", task),
      *(TileFreeOp(buf) for buf in (*state_l1, *acc_l1, merge_out_l1)),
      TileReturnOp(),
    ]
  )
  return prog


# ---------------------------------------------------------------------------
# step context
# ---------------------------------------------------------------------------


def build_step_context(scn: Scenario, request: int, step: int, variant: str, hw) -> NestContextOp:
  """One decode step: prefetches, the append dispatch, the per-block
  attention dispatches, the pipeline merge, the OUT store and the L2
  release schedule (plan §6)."""
  kvh, hpk, d = scn.kv_heads, scn.heads_per_kv, scn.head_dim
  elem = common.DTYPE_BYTES[scn.kv_dtype]
  pipeline = variant == "pipeline"
  blocks = scn.block_count(request, step)
  last = blocks - 1
  tip = scn.token_in_page(request, step)
  # Only partitions that receive a block get state/accumulator slots.
  partitions = min(PARTITIONS, blocks) if pipeline else 1

  l2_buffers = [
    (kvh * 1 * d * elem, 256),
    (kvh * 1 * d * elem, 256),
    (4, 64),
    (kvh * hpk * d * elem, 256),
  ]
  l2_buffers += [(kvh * hpk * 2 * 4, 64), (kvh * hpk * d * 4, 64)] * partitions
  if pipeline:
    l2_buffers.append((kvh * hpk * d * 4, 64))
  l2_buffers += [(4, 64)] * BLOCK_IDX_SLOTS
  l2_bytes, allowed_l2 = contract_bytes(hw, "l2", l2_buffers)
  # One dispatch per partition block plus append (+ merge); each dispatch runs
  # `bin(placement).count('1')` logical tasks.
  dispatches = 1 + blocks + (1 if pipeline else 0)
  logical_tasks = bin(scn.placement).count("1") * dispatches

  shapes = scn.global_shapes()
  dtypes = scn.global_dtypes()
  arg_names = [name for name in GLOBAL_ORDER if name != LENGTHS]
  ctx = NestContextOp(
    scn.context_name(request, step),
    ContextResources(
      l2_mode=scn.l2_mode,
      allowed_profiles=allowed_l2,
      logical_tasks=logical_tasks,
      l2_spm_bytes=l2_bytes,
      requested_contexts_per_tile=scn.contexts_per_tile,
      l2_cache=_cache_requirement(),
    ),
    placement=scn.placement,
    arg_types=[NestGlobalMemref.of(shapes[name], dtypes[name]) for name in arg_names],
    arg_names=arg_names,
  )
  pool_arg, table_arg, append_arg, q_arg, k_arg, v_arg, s_arg, o_arg, out_arg = (
    ctx.body.block.args
  )
  block = ctx.body.block

  k_new_l2 = NestAllocOp("k_new", "in", [kvh, 1, d], scn.kv_dtype, alignment=256)
  v_new_l2 = NestAllocOp("v_new", "in", [kvh, 1, d], scn.kv_dtype, alignment=256)
  append_idx_l2 = NestAllocOp("append_idx", "in", [1], "i32", alignment=64)
  q_l2 = NestAllocOp("q_l2", "in", [kvh, hpk, d], scn.kv_dtype, alignment=256)
  state_l2 = [
    NestAllocOp(
      f"state_p{p}",
      "inout",
      [kvh, hpk, 2],
      scn.state_dtype,
      sharing="context-local",
      alignment=64,
    )
    for p in range(partitions)
  ]
  acc_l2 = [
    NestAllocOp(
      f"acc_p{p}",
      "inout",
      [kvh, hpk, d],
      scn.state_dtype,
      sharing="context-local",
      alignment=64,
    )
    for p in range(partitions)
  ]
  out_l2 = (
    NestAllocOp(
      "out_l2",
      "inout",
      [kvh, hpk, d],
      scn.state_dtype,
      sharing="context-local",
      alignment=64,
    )
    if pipeline
    else None
  )
  block_idx_l2 = [
    NestAllocOp(f"block_idx_p{p}", "in", [1], "i32", alignment=64) for p in range(BLOCK_IDX_SLOTS)
  ]
  allocs: list = [k_new_l2, v_new_l2, append_idx_l2, q_l2, *state_l2, *acc_l2, *block_idx_l2]
  if out_l2 is not None:
    allocs.append(out_l2)
  block.add_ops(allocs)

  def global_view(arg, offsets, sizes, dtype):
    return NestSubviewOp(
      arg, list(offsets), list(sizes), [1] * len(sizes), NestGlobalView.of(sizes, dtype)
    )

  pool_v = global_view(pool_arg, [0, 0], [scn.physical_pages, scn.page_stride_elements], scn.kv_dtype)
  k_new_v = global_view(k_arg, [request, step, 0, 0, 0], [1, 1, kvh, 1, d], scn.kv_dtype)
  v_new_v = global_view(v_arg, [request, step, 0, 0, 0], [1, 1, kvh, 1, d], scn.kv_dtype)
  append_v = global_view(append_arg, [request * scn.steps + step], [1], "i32")
  q_v = global_view(q_arg, [request, step, 0, 0, 0], [1, 1, kvh, hpk, d], scn.kv_dtype)
  state_v = [
    global_view(s_arg, [request, step, p, 0, 0, 0], [1, 1, 1, kvh, hpk, 2], scn.state_dtype)
    for p in range(partitions)
  ]
  acc_v = [
    global_view(o_arg, [request, step, p, 0, 0, 0], [1, 1, 1, kvh, hpk, d], scn.state_dtype)
    for p in range(partitions)
  ]
  out_v = global_view(out_arg, [request, step, 0, 0, 0], [1, 1, kvh, hpk, d], scn.state_dtype)
  table_v = {b: global_view(table_arg, [b], [1], "i32") for b in range(last)}
  block.add_ops([pool_v, k_new_v, v_new_v, append_v, q_v, *state_v, *acc_v, out_v])
  block.add_ops(list(table_v.values()))

  tag = f"r{request}_s{step}"
  pf_k = NestPrefetchOp(k_new_v, k_new_l2, f"pf_k_{tag}")
  pf_v = NestPrefetchOp(v_new_v, v_new_l2, f"pf_v_{tag}")
  pf_append = NestPrefetchOp(append_v, append_idx_l2, f"pf_aidx_{tag}")
  pf_q = NestPrefetchOp(q_v, q_l2, f"pf_q_{tag}")
  pf_state = [
    NestPrefetchOp(state_v[p], state_l2[p], f"pf_state_p{p}_{tag}") for p in range(partitions)
  ]
  pf_acc = [NestPrefetchOp(acc_v[p], acc_l2[p], f"pf_acc_p{p}_{tag}") for p in range(partitions)]
  block.add_ops([pf_k, pf_v, pf_append, pf_q, *pf_state, *pf_acc])

  pf_bidx: dict[int, NestPrefetchOp] = {}


  tasks = NestTaskRangeOp(0, kvh)
  block.add_op(tasks)
  append_disp = NestDispatchOp(
    scn.append_program(request, tip),
    tasks,
    [pool_v],
    [k_new_l2, v_new_l2, append_idx_l2],
    [],
    f"append_grid_{tag}",
    f"append_inrel_{tag}",
    "",
    l1_mode=scn.l1_mode,
    bindings=[k_new_l2, v_new_l2, append_idx_l2],
    signal_policy={"input_released": "all_tasks"},
    depends_on=[pf_k, pf_v, pf_append],
  )
  block.add_op(append_disp)
  grids: dict[int, object] = {}
  inrels: dict[int, object] = {}
  outreadies: dict[int, object] = {}
  for b in range(blocks):
    tokens = scn.block_tokens(request, step, b)
    final = b == last
    p = b % 4 if pipeline else 0
    if final:
      idx_buffer = append_idx_l2
      idx_prefetch = pf_append
    else:
      idx_buffer = block_idx_l2[b % BLOCK_IDX_SLOTS]
      gate = (inrels[b - BLOCK_IDX_SLOTS],) if b >= BLOCK_IDX_SLOTS else ()
      pf_bidx[b] = NestPrefetchOp(
        table_v[b], idx_buffer, f"pf_bidx{b}_{tag}", depends_on=gate
      )
      block.add_op(pf_bidx[b])
      idx_prefetch = pf_bidx[b]
    depends: list = [pf_q, pf_state[p], pf_acc[p], idx_prefetch]
    if pipeline and b >= 4:
      depends.append(outreadies[b - 4])
    if not pipeline and b > 0:
      depends.append(grids[b - 1])
    disp = NestDispatchOp(
      scn.attention_program(request, tokens, final),
      tasks,
      [pool_v],
      [idx_buffer, q_l2, state_l2[p], acc_l2[p]],
      [state_l2[p], acc_l2[p]],
      f"att{b}_grid_{tag}",
      f"att{b}_inrel_{tag}",
      f"att{b}_out_{tag}",
      l1_mode=scn.l1_mode,
      bindings=[idx_buffer, q_l2, state_l2[p], acc_l2[p]],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=depends,
    )
    grids[b] = disp.grid_done
    inrels[b] = disp.input_released
    outreadies[b] = disp.output_ready
    block.add_op(disp)

  if pipeline:
    # Only partitions that actually received a block contribute a merge input
    # (a short sequence can have fewer blocks than partitions).
    merge_deps = [
      outreadies[max(b for b in range(blocks) if b % PARTITIONS == p)]
      for p in range(PARTITIONS)
      if any(b % PARTITIONS == p for b in range(blocks))
    ]
    merge_disp = NestDispatchOp(
      scn.merge_program(partitions),
      tasks,
      [],
      [*state_l2, *acc_l2],
      [out_l2],
      f"merge_grid_{tag}",
      f"merge_inrel_{tag}",
      f"merge_out_{tag}",
      l1_mode=scn.l1_mode,
      bindings=[*state_l2, *acc_l2, out_l2],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=merge_deps,
    )
    block.add_op(merge_disp)
    store = NestDMAStoreOp(out_l2, out_v, f"out_store_{tag}", depends_on=[merge_disp.output_ready])
    block.add_op(store)
    block.add_op(NestAwaitOp([append_disp.grid_done, grids[last], merge_disp.grid_done, store]))
    releases = [(out_l2, [merge_disp.output_ready, store])]
    merge_inrel = [merge_disp.input_released]
  else:
    # Every block dispatch writes the context-local accumulator, so the
    # HBM store must wait for all of their output_ready results.
    store = NestDMAStoreOp(
      acc_l2[0], out_v, f"out_store_{tag}", depends_on=[outreadies[b] for b in range(blocks)]
    )
    block.add_op(store)
    block.add_op(NestAwaitOp([append_disp.grid_done, grids[last], store]))
    releases = []
    merge_inrel = []

  def release(buf, deps) -> None:
    block.add_op(NestReleaseOp(buf, depends_on=list(dict.fromkeys(deps))))

  for slot in range(BLOCK_IDX_SLOTS):
    members = [b for b in range(last) if b % BLOCK_IDX_SLOTS == slot]
    release(
      block_idx_l2[slot],
      [pf_bidx[b] for b in members] + [inrels[b] for b in members],
    )
  release(k_new_l2, [pf_k, append_disp.input_released])
  release(v_new_l2, [pf_v, append_disp.input_released])
  release(append_idx_l2, [pf_append, append_disp.input_released, inrels[last]])
  release(q_l2, [pf_q, *(inrels[b] for b in range(blocks))])
  for p in range(partitions):
    members = [b for b in range(blocks) if (b % PARTITIONS == p if pipeline else True)]
    if pipeline and not members:
      # No block landed in this partition, so its state/accumulator L2 slots
      # were only prefetched, never written; they are not released.
      continue
    release(
      state_l2[p],
      [
        pf_state[p],
        *(inrels[b] for b in members),
        *(outreadies[b] for b in members),
        *merge_inrel,
      ],
    )
    acc_deps = [
      pf_acc[p],
      *(inrels[b] for b in members),
      *(outreadies[b] for b in members),
      *merge_inrel,
    ]
    if not pipeline:
      # The baseline's single accumulator slot is the store source.
      acc_deps.append(store)
    release(acc_l2[p], acc_deps)
  for buf, deps in releases:
    release(buf, deps)
  block.add_op(NestReturnOp())
  return ctx


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------


def build_module(scn: Scenario, variant: str, hw) -> ModuleOp:
  """Full module for one variant: tile programs, step contexts and the
  nexus.program with the host-call DAG (plan §6)."""
  programs: list = []
  for request in range(scn.num_requests):
    for tip in sorted({scn.token_in_page(request, s) for s in range(scn.steps)}):
      programs.append(build_append_program(scn, request, tip, hw))
    seen: dict[tuple[int, bool], None] = {}
    for step in range(scn.steps):
      for b in range(scn.block_count(request, step)):
        tokens = scn.block_tokens(request, step, b)
        final = b == scn.block_count(request, step) - 1
        seen.setdefault((tokens, final), None)
    for (tokens, final) in seen:
      programs.append(build_attention_program(scn, request, tokens, final, hw))
  if variant == "pipeline":
    # One merge program per distinct partition count actually used.
    for used in sorted(
      {
        min(PARTITIONS, scn.block_count(request, step))
        for request in range(scn.num_requests)
        for step in range(scn.steps)
      }
    ):
      programs.append(build_merge_program(scn, hw, used))
  contexts = [
    build_step_context(scn, request, step, variant, hw)
    for request in range(scn.num_requests)
    for step in range(scn.steps)
  ]

  shapes = scn.global_shapes()
  dtypes = scn.global_dtypes()
  program = NexusProgramOp(
    f"paged_attention_decode_{variant}",
    [],
    arg_types=[NestGlobalMemref.of(shapes[name], dtypes[name]) for name in GLOBAL_ORDER],
    arg_names=list(GLOBAL_ORDER),
  )
  args = dict(zip(GLOBAL_ORDER, program.body.block.args))
  context_arg_names = [name for name in GLOBAL_ORDER if name != LENGTHS]
  submits = []
  body: list = []
  for request in range(scn.num_requests):
    depends = None
    for step in range(scn.steps):
      tag = f"r{request}_s{step}"
      prepare_deps = [depends] if depends is not None else []
      prepare = NexusHostCallOp(
        scn.prepare_routine(request, step),
        f"prepared_{tag}",
        bindings=[args[APPEND_IDS]],
        accesses=[((request * scn.steps + step) * 4, 4, "write")],
        scopes=[scn.scope(request)],
        depends_on=prepare_deps,
      )
      submit = NexusSubmitContextOp(
        scn.context_name(request, step),
        f"step_done_{tag}",
        actuals=[args[name] for name in context_arg_names],
        depends_on=[prepare.result],
      )
      token = scn.token(request, step)
      commit_accesses: list[tuple[int, int, str]] = [(request * 4, 4, "write")]
      commit_bindings = [args[LENGTHS]]
      if scn.opens_page(request, step):
        block = token // scn.page_tokens
        commit_bindings = [args[BLOCK_TABLE], args[LENGTHS]]
        commit_accesses = [(block * 4, 4, "write"), (request * 4, 4, "write")]
      commit = NexusHostCallOp(
        scn.commit_routine(request, step),
        f"committed_{tag}",
        bindings=commit_bindings,
        accesses=commit_accesses,
        scopes=[],
        depends_on=[submit.result],
      )
      body += [prepare, submit, commit]
      # Every submitted root must be awaited before nexus.return.
      submits.append(submit.result)
      depends = commit.result
    used = scn.used_table_entries(request)
    release = NexusHostCallOp(
      scn.release_routine(request),
      f"released_r{request}",
      bindings=[args[BLOCK_TABLE], args[LENGTHS]],
      accesses=[(0, used * 4, "write"), (request * 4, 4, "write")],
      scopes=[],
      depends_on=[depends],
    )
    body.append(release)
    submits.append(release.result)
  program.body.block.add_ops(body)
  program.body.block.add_op(NexusAwaitOp(submits))
  program.body.block.add_op(NexusReturnOp())
  return ModuleOp([*programs, *contexts, program])


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", default="examples/workloads")
  parser.add_argument("--variant", choices=("pipeline", "baseline", "both"), default="both")
  parser.add_argument("--num-requests", type=int, default=3)
  parser.add_argument("--initial-lengths", default="255,511,767")
  parser.add_argument("--steps", type=int, default=4)
  parser.add_argument("--page-tokens", type=int, default=16)
  parser.add_argument("--kv-heads", type=int, default=4)
  parser.add_argument("--heads-per-kv", type=int, default=4)
  parser.add_argument("--head-dim", type=int, default=64)
  parser.add_argument("--physical-pages", type=int, default=128)
  parser.add_argument("--page-padding-bytes", type=int, default=64)
  parser.add_argument("--placement", type=int, default=None,
                      help="tile placement mask; default 2**kv_heads-1 for the group")
  parser.add_argument("--initial-owner-pages", default=None,
                      help="comma-separated first pages per request; default seeded shuffle")
  args = parser.parse_args()

  lengths = tuple(int(v) for v in args.initial_lengths.split(","))
  if args.initial_owner_pages is not None:
    mapping = tuple((int(v),) for v in args.initial_owner_pages.split(","))
  else:
    mapping = None
  scenario = Scenario(
    placement=args.placement if args.placement is not None else (1 << args.kv_heads) - 1,
    num_requests=args.num_requests,
    initial_lengths=lengths,
    steps=args.steps,
    page_tokens=args.page_tokens,
    kv_heads=args.kv_heads,
    heads_per_kv=args.heads_per_kv,
    head_dim=args.head_dim,
    physical_pages=args.physical_pages,
    page_padding_bytes=args.page_padding_bytes,
    initial_mapping=mapping,
  )

  from pipeline_validator.config import HardwareConfig

  hw = HardwareConfig()
  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  variants = ["pipeline", "baseline"] if args.variant == "both" else [args.variant]
  hashes: dict[str, str] = {}
  for variant in variants:
    module = build_module(scenario, variant, hw)
    path = out_dir / f"paged_attention_decode_{variant}.mlir"
    header = [
      f"PagedAttention decode ({variant} variant, plan §6) -- timing model, no numerics.",
      f"R={scenario.num_requests} initial={list(scenario.initial_lengths)} steps={scenario.steps}"
      f" page_tokens={scenario.page_tokens} kv_heads={scenario.kv_heads}"
      f" heads_per_kv={scenario.heads_per_kv} head_dim={scenario.head_dim}"
      f" pages={scenario.physical_pages} page_stride={scenario.page_stride_bytes}B",
      "Pool pages are host-managed (HostAllocPages/HostFreePages); the block table is",
      "host-written between steps; attention gathers K/V rows straight into L1 with",
      "#tile.indexed_map<index_scale=page_stride, task_stride=B*D, segment=valid*D>.",
    ]
    write_workload(path, header, module)
    hashes[variant] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"wrote {path}")
  scenario_path = out_dir / "paged_attention_decode_scenario.json"
  scenario_path.write_text(
    __import__("json").dumps(common.scenario_to_dict(scenario, hashes), indent=2) + "\n",
    encoding="utf-8",
  )
  print(f"wrote {scenario_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
