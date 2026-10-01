"""Generate a Prefill whose producer programs sink every tile.await.

The baseline multicontext prefill pipelines inside the tile programs: Q/K/V
stores co-issue with the following BOAs and ``input_released`` fires before the
last BOA so the root can refill staging early.  This variant keeps the same
logical work and the same attention tail but replaces that hoisting in the
three producer programs (``qkv_chunk_{init,accum}`` -> split Q vs K/V,
``prefill_outproj_tile`` -> split per output row half) with sunk awaits:

* no load is hoisted across the BOA that reads its buffer (one X buffer and
  one accumulator per program keep the true RAW/WAR chain), so
  ``input_released`` only fires after the last L2 load;
* every ``tile.await`` is sunk to the last dependency-safe point: the weight
  fill co-issues with the first X fill (plus the first partials when
  accumulating) behind one merged await before the first BOA, and each store
  issues as soon as its own BOA has been awaited, overlapping the next
  independent fills — no store ever co-issues with an outstanding BOA;
* the parallelism that the pipeline used to provide comes from dispatching
  more, independent Grids instead: the Q chain (writes ``q_l2``) and the K/V
  chain (writes ``k_l2``/``v_l2``) are separate allocations, so 16 QKV
  dispatches form two independent 8-step chains, and each query block's output
  projection is issued twice (low/high 64-row halves with their own output
  buffers), giving 8 independent outproj Grids;
* the attention programs are emitted identically to the baseline.

This is a timing/resource model: BOA and EVU descriptors do not carry
numerical tensors, so the workload does not establish numerical correctness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from transformer_common import (
  boa_matmul,
  contract_bytes,
  evu,
  l1_alloc,
  l1_load,
  l1_store,
  l2_view,
  write_workload,
)
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
  TileAwaitOp,
  TileFreeOp,
  TileProgramDefOp,
  TileReturnOp,
  TileSignalOp,
)
from pipeline_validator.profiles import ContextResources, TileResources, build_registry

PLACEMENT = 0x0F
# Grid routes stay resident until a later action depends on their grid_done, so
# every dispatch waits out this many Grids before requiring a retirement edge.
GRID_WINDOW = 8
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
  REPO_ROOT / "examples/workloads/transformer_prefill_attention_dispatchparallel_multicontext.mlir"
)


@dataclass(frozen=True)
class DispatchParallelConfig:
  seq_len: int = 512
  hidden_dim: int = 1024
  q_heads: int = 16
  kv_heads: int = 4
  head_dim: int = 64
  query_block: int = 128
  kv_block: int = 128
  proj_k_chunk: int = 128
  projection_row_block: int = 64
  outproj_k_chunk: int = 64

  def __post_init__(self) -> None:
    names = (
      "seq_len",
      "hidden_dim",
      "q_heads",
      "kv_heads",
      "head_dim",
      "query_block",
      "kv_block",
      "proj_k_chunk",
      "projection_row_block",
      "outproj_k_chunk",
    )
    for name in names:
      value = getattr(self, name)
      if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    if self.q_heads % self.kv_heads:
      raise ValueError("q_heads must be divisible by kv_heads")
    if (
      self.seq_len % self.query_block
      or self.seq_len % self.projection_row_block
      or self.seq_len % self.kv_block
      or self.query_block % self.projection_row_block
    ):
      raise ValueError("query, projection-row and KV blocks must divide their parent extents")
    if self.hidden_dim % self.proj_k_chunk:
      raise ValueError("proj_k_chunk must divide hidden_dim evenly")
    if self.q_slice % self.outproj_k_chunk:
      raise ValueError("outproj_k_chunk must divide the contiguous Q-head width evenly")

  @property
  def q_slice(self) -> int:
    return (self.q_heads // self.kv_heads) * self.head_dim

  @property
  def query_blocks(self) -> int:
    return self.seq_len // self.query_block

  @property
  def projection_row_blocks(self) -> int:
    return self.seq_len // self.projection_row_block

  @property
  def kv_blocks(self) -> int:
    return self.seq_len // self.kv_block

  @property
  def projection_chunks(self) -> int:
    return self.hidden_dim // self.proj_k_chunk


def _tile_contract(
  hw: HardwareConfig, buffers: list[tuple[int, int]], contexts_per_tile: int
) -> TileResources:
  """Prove one child Arena against every advertised L1 mode and the R envelope."""
  registry = build_registry(hw)
  l1_profiles = registry.l1
  base = l1_profiles[0]
  reserved = conservative_arena_bytes(buffers, base)
  allowed = tuple(
    mode
    for mode, profile in l1_profiles.items()
    if all(
      contexts_per_tile * (reserved // profile.banks) <= profile.user_spm_per_bank
      for _bank in range(profile.banks)
    )
  )
  if 0 not in allowed:
    raise ValueError(
      f"L1 mode 0 cannot satisfy {contexts_per_tile} contexts per tile "
      f"for child Arena reservation {reserved}"
    )
  return TileResources(allowed_profiles=allowed, tile_l1_spm_bytes_per_context=reserved)


def _qkv_program(
  cfg: DispatchParallelConfig, hw: HardwareConfig, contexts_per_tile: int, *, accumulate: bool, q_only: bool
) -> TileProgramDefOp:
  """Project one input-K chunk, Q only or K/V only, with sunk awaits.

  No load is hoisted across the BOA that reads its buffer, so
  ``input_released`` still fires only after the final L2 load.  Every
  ``tile.await`` sits at the last dependency-safe point: the weight fill
  co-issues with the first X fill (plus the first partials when accumulating)
  behind one merged await, and each store issues right after its own BOA has
  been awaited, overlapping the next independent fills.
  """
  seq, rows, chunk = cfg.seq_len, cfg.projection_row_block, cfg.proj_k_chunk
  q_slice, hd = cfg.q_slice, cfg.head_dim
  name = ("qkv_q_" if q_only else "qkv_kv_") + ("accum" if accumulate else "init")
  if q_only:
    buffers = [(rows * chunk * 2, 256), (rows * q_slice * 2, 256), (chunk * q_slice * 2, 256)]
    arg_types = [
      NestTask(),
      NestBuffer.of([seq, chunk], "bf16"),
      NestBuffer.of([cfg.kv_heads, chunk, q_slice], "bf16"),
      NestBuffer.of([cfg.kv_heads, seq, q_slice], "bf16"),
    ]
    arg_names = ["task", "x_chunk", "wq_chunk", "q_l2"]
  else:
    buffers = [
      (rows * chunk * 2, 256),
      (rows * hd * 2, 256),
      (chunk * hd * 2, 256),
      (rows * hd * 2, 256),
      (chunk * hd * 2, 256),
    ]
    arg_types = [
      NestTask(),
      NestBuffer.of([seq, chunk], "bf16"),
      NestBuffer.of([cfg.kv_heads, chunk, hd], "bf16"),
      NestBuffer.of([cfg.kv_heads, chunk, hd], "bf16"),
      NestBuffer.of([cfg.kv_heads, seq, hd], "bf16"),
      NestBuffer.of([cfg.kv_heads, seq, hd], "bf16"),
    ]
    arg_names = ["task", "x_chunk", "wk_chunk", "wv_chunk", "k_l2", "v_l2"]
  prog = TileProgramDefOp(
    name, _tile_contract(hw, buffers, contexts_per_tile), arg_types=arg_types, arg_names=arg_names
  )
  task = prog.body.block.args[0]
  x_chunk = prog.body.block.args[1]

  x_buf = l1_alloc([rows, chunk], "bf16", "x_buf")
  body: list = [x_buf]
  allocs = [x_buf]
  weight_loads: list = []
  if q_only:
    q_acc = l1_alloc([rows, q_slice], "bf16", "q_acc")
    wq_buf = l1_alloc([chunk, q_slice], "bf16", "wq_buf")
    allocs += [q_acc, wq_buf]
    q_l2 = prog.body.block.args[3]
    body += [q_acc, wq_buf]
    wq_v = l2_view(
      prog.body.block.args[2], task, 0, [0, 0, 0], [1, chunk, q_slice], [1, chunk, q_slice], "bf16"
    )
    wq_loaded = l1_load(wq_v, wq_buf, "wq_loaded")
    body += [wq_v, wq_loaded]
    weight_loads.append(wq_loaded)
    dests = [("q", q_slice, q_acc, q_l2)]
  else:
    wk_chunk, wv_chunk, k_l2, v_l2 = prog.body.block.args[2:6]
    k_acc = l1_alloc([rows, hd], "bf16", "k_acc")
    wk_buf = l1_alloc([chunk, hd], "bf16", "wk_buf")
    v_acc = l1_alloc([rows, hd], "bf16", "v_acc")
    wv_buf = l1_alloc([chunk, hd], "bf16", "wv_buf")
    allocs += [k_acc, wk_buf, v_acc, wv_buf]
    body += [k_acc, wk_buf, v_acc, wv_buf]
    wk_v = l2_view(wk_chunk, task, 0, [0, 0, 0], [1, chunk, hd], [1, chunk, hd], "bf16")
    wv_v = l2_view(wv_chunk, task, 0, [0, 0, 0], [1, chunk, hd], [1, chunk, hd], "bf16")
    wk_loaded = l1_load(wk_v, wk_buf, "wk_loaded")
    wv_loaded = l1_load(wv_v, wv_buf, "wv_loaded")
    body += [wk_v, wv_v, wk_loaded, wv_loaded]
    weight_loads += [wk_loaded, wv_loaded]
    dests = [("k", hd, k_acc, k_l2), ("v", hd, v_acc, v_l2)]

  # Sunk awaits: every async op issues as soon as its L1 buffers are free and
  # each await sits at the last legal point before a conflicting op (RAW on an
  # accumulator, WAR on a reused buffer).  Stores are never awaited inline;
  # they drain in the gate that protects the next conflicting op.
  pending_stores: list = []
  for row_block in range(cfg.projection_row_blocks):
    row = row_block * rows
    x_v = l2_view(x_chunk, None, None, [row, 0], [rows, chunk], [rows, chunk], "bf16")
    x_loaded = l1_load(x_v, x_buf, f"x_loaded_{row_block}")
    body += [x_v, x_loaded]
    partial_loads = []
    if accumulate:
      if row_block:
        # The partial fills overwrite the accumulators the previous stores
        # still read, so they are the first ops gated by those stores.
        body.append(TileAwaitOp(pending_stores))
        pending_stores = []
      for tag, width, acc_buf, dest in dests:
        dv = l2_view(dest, task, 0, [0, row, 0], [1, rows, width], [1, rows, width], "bf16")
        loaded = l1_load(dv, acc_buf, f"{tag}_partial_loaded_{row_block}")
        body += [dv, loaded]
        partial_loads.append(loaded)
    # One merged gate before the first BOA: RAW on X plus the partials, plus
    # the WAR of the BOAs over accumulators still being stored when this is
    # not the first chunk (the accumulate path already drained them above).
    gate = [x_loaded, *partial_loads]
    if row_block == 0:
      gate += weight_loads
    else:
      gate += pending_stores
      pending_stores = []
    body.append(TileAwaitOp(gate))
    if row_block == cfg.projection_row_blocks - 1:
      # The root may only recycle staging after the final L2 read of this
      # program, not before the last BOA.
      body.append(TileSignalOp("input_released", task))
    for tag, width, acc_buf, dest in dests:
      dv = l2_view(dest, task, 0, [0, row, 0], [1, rows, width], [1, rows, width], "bf16")
      boa = boa_matmul(rows, width, chunk, f"{tag}_boa_{row_block}", accumulate=accumulate)
      # RAW accumulator: the store issues only after its own BOA has been
      # awaited, then overlaps the next chunk's independent fills.
      body += [dv, boa, TileAwaitOp([boa])]
      stored = l1_store(acc_buf, dv, f"{tag}_stored_{row_block}")
      body.append(stored)
      pending_stores.append(stored)

  body.append(TileAwaitOp(pending_stores))
  body.append(TileSignalOp("output_ready", task))
  body += [TileFreeOp(buf) for buf in allocs]
  body.append(TileReturnOp())
  prog.body.block.add_ops(body)
  return prog


def _attention_program(
  cfg: DispatchParallelConfig, hw: HardwareConfig, contexts_per_tile: int, query_block: int
) -> TileProgramDefOp:
  """Attend one 128-row query block using two reusable 64-row engine tiles."""
  qb, kb, q_slice, hd = cfg.query_block, cfg.kv_block, cfg.q_slice, cfg.head_dim
  rows = cfg.projection_row_block
  microblocks = qb // rows
  q_offset = query_block * qb
  sm_ops = rows * kb + rows * 66
  buffers = [
    *[(rows * q_slice * 2, 256)] * (2 * microblocks),
    *[(4 * rows * 4, 64)] * (2 * microblocks),
    (rows * kb * 4, 64),
    *[(kb * hd * 2, 256)] * 4,
  ]
  name = f"prefill_attention_q{query_block}"
  prog = TileProgramDefOp(
    name,
    _tile_contract(hw, buffers, contexts_per_tile),
    arg_types=[
      NestTask(),
      NestBuffer.of([cfg.kv_heads, cfg.seq_len, q_slice], "bf16"),
      NestBuffer.of([cfg.kv_heads, cfg.seq_len, hd], "bf16"),
      NestBuffer.of([cfg.kv_heads, cfg.seq_len, hd], "bf16"),
      NestBuffer.of([cfg.kv_heads, qb, q_slice], "bf16"),
    ],
    arg_names=["task", "q_l2", "k_l2", "v_l2", "o_l2"],
  )
  task = prog.body.block.args[0]
  q_l2, k_l2, v_l2, o_l2 = prog.body.block.args[1:]
  q_bufs = [l1_alloc([rows, q_slice], "bf16", f"q{part}") for part in range(microblocks)]
  acc_bufs = [l1_alloc([rows, q_slice], "bf16", f"acc{part}") for part in range(microblocks)]
  m_bufs = [l1_alloc([4, rows], "f32", f"m{part}") for part in range(microblocks)]
  l_bufs = [l1_alloc([4, rows], "f32", f"l{part}") for part in range(microblocks)]
  score_buf = l1_alloc([rows, kb], "f32", "score")
  k_bufs = [l1_alloc([kb, hd], "bf16", f"k{p}") for p in range(2)]
  v_bufs = [l1_alloc([kb, hd], "bf16", f"v{p}") for p in range(2)]
  body: list = [*q_bufs, *acc_bufs, *m_bufs, *l_bufs, score_buf, *k_bufs, *v_bufs]

  q_loads = []
  for part in range(microblocks):
    row_offset = q_offset + part * rows
    qv = l2_view(q_l2, task, 0, [0, row_offset, 0], [1, rows, q_slice], [1, rows, q_slice], "bf16")
    ql = l1_load(qv, q_bufs[part], f"q{part}_loaded")
    body += [qv, ql]
    q_loads.append(ql)
  loads: dict[int, list] = {}
  for step in range(min(2, cfg.kv_blocks)):
    kv = l2_view(k_l2, task, 0, [0, step * kb, 0], [1, kb, hd], [1, kb, hd], "bf16")
    vv = l2_view(v_l2, task, 0, [0, step * kb, 0], [1, kb, hd], [1, kb, hd], "bf16")
    kl = l1_load(kv, k_bufs[step % 2], f"k{step}_loaded")
    vl = l1_load(vv, v_bufs[step % 2], f"v{step}_loaded")
    body += [kv, vv, kl, vl]
    loads[step] = [kl, vl]
  body.append(TileAwaitOp(q_loads))

  def kv_views(step: int):
    kv = l2_view(k_l2, task, 0, [0, step * kb, 0], [1, kb, hd], [1, kb, hd], "bf16")
    vv = l2_view(v_l2, task, 0, [0, step * kb, 0], [1, kb, hd], [1, kb, hd], "bf16")
    return kv, vv

  for step in range(cfg.kv_blocks):
    body.append(TileAwaitOp(loads[step]))
    if step == cfg.kv_blocks - 1:
      body.append(TileSignalOp("input_released", task))
    for part in range(microblocks):
      for head in range(4):
        qk = boa_matmul(rows, kb, hd, f"qk{step}_p{part}_h{head}")
        body += [qk, TileAwaitOp([qk])]
        sm = evu("online_softmax_update", sm_ops, f"sm{step}_p{part}_h{head}")
        body += [sm, TileAwaitOp([sm])]
        pv = boa_matmul(rows, hd, kb, f"pv{step}_p{part}_h{head}", accumulate=step > 0)
        body += [pv, TileAwaitOp([pv])]
    if step + 2 < cfg.kv_blocks:
      next_step = step + 2
      kv, vv = kv_views(next_step)
      kl = l1_load(kv, k_bufs[next_step % 2], f"k{next_step}_loaded")
      vl = l1_load(vv, v_bufs[next_step % 2], f"v{next_step}_loaded")
      body += [kv, vv, kl, vl]
      loads[next_step] = [kl, vl]

  output_stores = []
  for part in range(microblocks):
    ov = l2_view(o_l2, task, 0, [0, part * rows, 0], [1, rows, q_slice], [1, rows, q_slice], "bf16")
    stored = l1_store(acc_bufs[part], ov, f"o{part}_stored")
    body += [ov, stored]
    output_stores.append(stored)
  body.append(TileAwaitOp(output_stores))
  body.append(TileSignalOp("output_ready", task))
  body += [TileFreeOp(buf) for buf in (*q_bufs, *acc_bufs, *m_bufs, *l_bufs, score_buf, *k_bufs, *v_bufs)]
  body.append(TileReturnOp())
  prog.body.block.add_ops(body)
  return prog


def _outproj_program(
  cfg: DispatchParallelConfig, hw: HardwareConfig, contexts_per_tile: int, row_half: int
) -> TileProgramDefOp:
  """Project one 64-row half of one query block with sunk awaits.

  The O fill co-issues with the first weight fill behind one merged await;
  later weight fills stay directly awaited before their BOA because they
  reuse the one weight buffer the previous BOA is still reading.

  Each half owns an independent output buffer, so the low and high halves of a
  query block are two Grids that never contend.
  """
  qb, q_slice = cfg.query_block, cfg.q_slice
  rows, k_chunk = cfg.projection_row_block, cfg.outproj_k_chunk
  chunks = q_slice // k_chunk
  row_offset = row_half * rows
  buffers = [(rows * q_slice * 2, 256), (rows * q_slice * 2, 256), (k_chunk * q_slice * 2, 256)]
  name = f"prefill_outproj_{'lo' if row_half == 0 else 'hi'}"
  prog = TileProgramDefOp(
    name,
    _tile_contract(hw, buffers, contexts_per_tile),
    arg_types=[
      NestTask(),
      NestBuffer.of([cfg.kv_heads, qb, q_slice], "bf16"),
      NestBuffer.of([cfg.kv_heads, cfg.hidden_dim, q_slice], "bf16"),
      NestBuffer.of([cfg.kv_heads, rows, q_slice], "bf16"),
    ],
    arg_names=["task", "o_l2", "wo_l2", "out_l2"],
  )
  task = prog.body.block.args[0]
  o_l2, wo_l2, out_l2 = prog.body.block.args[1:]
  acc_buf = l1_alloc([rows, q_slice], "bf16", "acc")
  o_buf = l1_alloc([rows, q_slice], "bf16", "o")
  w_buf = l1_alloc([k_chunk, q_slice], "bf16", "w_buf")
  body: list = [acc_buf, o_buf, w_buf]

  for head_block in range(cfg.kv_heads):
    ov = l2_view(
      o_l2, None, None, [head_block, row_offset, 0], [1, rows, q_slice], [1, rows, q_slice], "bf16"
    )
    ol = l1_load(ov, o_buf, f"o{head_block}_loaded")
    body += [ov, ol]
    for chunk_index in range(chunks):
      row = head_block * q_slice + chunk_index * k_chunk
      wv = l2_view(wo_l2, task, 0, [0, row, 0], [1, k_chunk, q_slice], [1, k_chunk, q_slice], "bf16")
      wl = l1_load(wv, w_buf, f"w{head_block}_{chunk_index}_loaded")
      body += [wv, wl]
      # The O fill is consumed by the first BOA of the head block, so its
      # await sinks into that BOA's gate; the remaining weight fills are
      # gated only by themselves (the previous BOA already drained).
      gate: list = [wl]
      if chunk_index == 0:
        gate.append(ol)
      body.append(TileAwaitOp(gate))
      if head_block == cfg.kv_heads - 1 and chunk_index == chunks - 1:
        body.append(TileSignalOp("input_released", task))
      boa = boa_matmul(
        rows,
        q_slice,
        k_chunk,
        f"outproj{head_block}_{chunk_index}",
        accumulate=head_block > 0 or chunk_index > 0,
      )
      body += [boa, TileAwaitOp([boa])]

  out_v = l2_view(out_l2, task, 0, [0, 0, 0], [1, rows, q_slice], [1, rows, q_slice], "bf16")
  stored = l1_store(acc_buf, out_v, "out_stored")
  body += [out_v, stored, TileAwaitOp([stored])]
  body.append(TileSignalOp("output_ready", task))
  body += [TileFreeOp(buf) for buf in (acc_buf, o_buf, w_buf)]
  body.append(TileReturnOp())
  prog.body.block.add_ops(body)
  return prog


def make_prefill_dispatchparallel(
  *, hw: HardwareConfig | None = None, contexts_per_tile: int = 4
) -> ModuleOp:
  """Build prefill whose producers sink every await and dispatch more Grids."""
  if type(contexts_per_tile) is not int or contexts_per_tile not in (1, 2, 4):
    raise ValueError("contexts_per_tile must be one of 1, 2, or 4")
  cfg = DispatchParallelConfig()
  hw = hw or HardwareConfig()
  seq, chunk, q_slice, hd = cfg.seq_len, cfg.proj_k_chunk, cfg.q_slice, cfg.head_dim
  kh, nc = cfg.kv_heads, cfg.projection_chunks
  query_blocks, out_halves = cfg.query_blocks, 2

  l2_buffers = [
    (seq * chunk * 2, 256),
    (seq * chunk * 2, 256),
    (kh * chunk * q_slice * 2, 256),
    (kh * chunk * q_slice * 2, 256),
    (kh * chunk * hd * 2, 256),
    (kh * chunk * hd * 2, 256),
    (kh * chunk * hd * 2, 256),
    (kh * chunk * hd * 2, 256),
    (kh * seq * q_slice * 2, 256),
    (kh * seq * hd * 2, 256),
    (kh * seq * hd * 2, 256),
    (kh * cfg.hidden_dim * q_slice * 2, 256),
  ]
  l2_buffers += [(kh * cfg.query_block * q_slice * 2, 256)] * query_blocks
  l2_buffers += [(kh * cfg.projection_row_block * q_slice * 2, 256)] * (query_blocks * out_halves)
  l2_bytes, allowed_l2 = contract_bytes(hw, "l2", l2_buffers)
  logical_tasks = kh * (2 * nc + 2 * query_blocks + query_blocks)

  programs = [
    _qkv_program(cfg, hw, contexts_per_tile, accumulate=False, q_only=True),
    _qkv_program(cfg, hw, contexts_per_tile, accumulate=True, q_only=True),
    _qkv_program(cfg, hw, contexts_per_tile, accumulate=False, q_only=False),
    _qkv_program(cfg, hw, contexts_per_tile, accumulate=True, q_only=False),
    *[_attention_program(cfg, hw, contexts_per_tile, block) for block in range(query_blocks)],
    *[_outproj_program(cfg, hw, contexts_per_tile, half) for half in range(out_halves)],
  ]
  context = NestContextOp(
    "prefill_dispatchparallel_ctx",
    ContextResources(
      l2_mode=0,
      allowed_profiles=allowed_l2,
      logical_tasks=logical_tasks,
      l2_spm_bytes=l2_bytes,
      requested_contexts_per_tile=contexts_per_tile,
    ),
    placement=PLACEMENT,
    arg_types=[
      NestGlobalMemref.of([nc, seq, chunk], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, q_slice], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, hd], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, hd], "bf16"),
      NestGlobalMemref.of([kh, cfg.hidden_dim, q_slice], "bf16"),
      NestGlobalMemref.of([query_blocks, out_halves, kh, cfg.projection_row_block, q_slice], "bf16"),
    ],
    arg_names=["X", "WQ", "WK", "WV", "WO", "OUT"],
  )
  X, WQ, WK, WV, WO, OUT = context.body.block.args
  block = context.body.block

  x_p = [NestAllocOp(f"x_p{p}", "in", [seq, chunk], "bf16", alignment=256) for p in range(2)]
  wq_p = [NestAllocOp(f"wq_p{p}", "in", [kh, chunk, q_slice], "bf16", alignment=256) for p in range(2)]
  wk_p = [NestAllocOp(f"wk_p{p}", "in", [kh, chunk, hd], "bf16", alignment=256) for p in range(2)]
  wv_p = [NestAllocOp(f"wv_p{p}", "in", [kh, chunk, hd], "bf16", alignment=256) for p in range(2)]
  q_l2 = NestAllocOp("q_l2", "inout", [kh, seq, q_slice], "bf16", sharing="context-local", alignment=256)
  k_l2 = NestAllocOp("k_l2", "inout", [kh, seq, hd], "bf16", sharing="context-local", alignment=256)
  v_l2 = NestAllocOp("v_l2", "inout", [kh, seq, hd], "bf16", sharing="context-local", alignment=256)
  wo_l2 = NestAllocOp("wo_l2", "in", [kh, cfg.hidden_dim, q_slice], "bf16", alignment=256)
  o_l2 = [
    NestAllocOp(
      f"o_q{query_block}",
      "inout",
      [kh, cfg.query_block, q_slice],
      "bf16",
      sharing="context-local",
      alignment=256,
    )
    for query_block in range(query_blocks)
  ]
  out_l2 = [
    NestAllocOp(
      f"out_q{query_block}_{'lo' if half == 0 else 'hi'}",
      "out",
      [kh, cfg.projection_row_block, q_slice],
      "bf16",
      alignment=256,
    )
    for query_block in range(query_blocks)
    for half in range(out_halves)
  ]
  block.add_ops([*x_p, *wq_p, *wk_p, *wv_p, q_l2, k_l2, v_l2, wo_l2, *o_l2, *out_l2])
  tasks = NestTaskRangeOp(0, kh)
  block.add_op(tasks)

  grid_dones: list[OpResult[NestEvent]] = []

  def retire(deps: list[OpResult[NestEvent]]) -> list[OpResult[NestEvent]]:
    if len(grid_dones) >= GRID_WINDOW:
      deps.append(grid_dones[len(grid_dones) - GRID_WINDOW])
    return deps

  def input_view(global_arg, offsets, sizes):
    return NestSubviewOp(global_arg, offsets, sizes, [1] * len(offsets), NestGlobalView.of(sizes, "bf16"))

  # ---- projection phase: one Q chain and one K/V chain, independent buffers ----
  q_inrel: list[OpResult[NestEvent]] = []
  q_out: list[OpResult[NestEvent]] = []
  kv_inrel: list[OpResult[NestEvent]] = []
  kv_out: list[OpResult[NestEvent]] = []
  prefetches: dict[tuple[str, int], NestPrefetchOp] = {}
  for c in range(nc):
    x_view = input_view(X, [c, 0, 0], [1, seq, chunk])
    wq_view = input_view(WQ, [c, 0, 0, 0], [1, kh, chunk, q_slice])
    wk_view = input_view(WK, [c, 0, 0, 0], [1, kh, chunk, hd])
    wv_view = input_view(WV, [c, 0, 0, 0], [1, kh, chunk, hd])
    block.add_ops([x_view, wq_view, wk_view, wv_view])
    # Staging is a ping-pong: chunk c refills slot c%2 once every consumer of
    # that slot's previous fill released its inputs.  X is read by both chains,
    # the Q weights only by the Q chain, the K/V weights only by the K/V chain.
    x_prev: tuple[OpResult[NestEvent], ...] = ()
    wq_prev: tuple[OpResult[NestEvent], ...] = ()
    kv_prev: tuple[OpResult[NestEvent], ...] = ()
    if c >= 2:
      x_prev = (q_inrel[c - 2], kv_inrel[c - 2])
      wq_prev = (q_inrel[c - 2],)
      kv_prev = (kv_inrel[c - 2],)
    for kind, view, destination, previous in (
      ("x", x_view, x_p[c % 2], x_prev),
      ("wq", wq_view, wq_p[c % 2], wq_prev),
      ("wk", wk_view, wk_p[c % 2], kv_prev),
      ("wv", wv_view, wv_p[c % 2], kv_prev),
    ):
      event = NestPrefetchOp(view, destination, f"pre_{kind}_{c}", depends_on=previous)
      prefetches[(kind, c)] = event
      block.add_op(event)

    accum = c > 0
    # An accumulate step also reads the partial it folds into.
    q_reads = [x_p[c % 2], wq_p[c % 2], *([q_l2] if accum else [])]
    q_writes = [q_l2]
    q_deps: list[OpResult[NestEvent]] = [prefetches[("x", c)].result, prefetches[("wq", c)].result]
    if c:
      # Whole-buffer WAW on q_l2 orders the Q chain.
      q_deps.append(q_out[c - 1])
    q_dispatch = NestDispatchOp(
      f"qkv_q_{'init' if c == 0 else 'accum'}",
      tasks,
      [],
      q_reads,
      q_writes,
      f"qkv_q_grid_{c}",
      f"qkv_q_inrel_{c}",
      f"qkv_q_out_{c}",
      l1_mode=0,
      bindings=[x_p[c % 2], wq_p[c % 2], q_l2],
      # Bindings always enumerate the program formals, deduplicated.
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=retire(q_deps),
    )
    block.add_op(q_dispatch)
    q_inrel.append(q_dispatch.input_released)
    q_out.append(q_dispatch.output_ready)
    grid_dones.append(q_dispatch.grid_done)

    kv_reads = [x_p[c % 2], wk_p[c % 2], wv_p[c % 2], *([k_l2, v_l2] if accum else [])]
    kv_writes = [k_l2, v_l2]
    kv_deps: list[OpResult[NestEvent]] = [
      prefetches[("x", c)].result,
      prefetches[("wk", c)].result,
      prefetches[("wv", c)].result,
    ]
    if c:
      # Whole-buffer WAW on k_l2/v_l2 orders the K/V chain.
      kv_deps.append(kv_out[c - 1])
    kv_dispatch = NestDispatchOp(
      f"qkv_kv_{'init' if c == 0 else 'accum'}",
      tasks,
      [],
      kv_reads,
      kv_writes,
      f"qkv_kv_grid_{c}",
      f"qkv_kv_inrel_{c}",
      f"qkv_kv_out_{c}",
      l1_mode=0,
      bindings=[x_p[c % 2], wk_p[c % 2], wv_p[c % 2], k_l2, v_l2],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=retire(kv_deps),
    )
    block.add_op(kv_dispatch)
    kv_inrel.append(kv_dispatch.input_released)
    kv_out.append(kv_dispatch.output_ready)
    grid_dones.append(kv_dispatch.grid_done)

  wo_view = input_view(WO, [0, 0, 0], [kh, cfg.hidden_dim, q_slice])
  block.add_op(wo_view)
  wo_prefetch = NestPrefetchOp(wo_view, wo_l2, "pre_wo", depends_on=[q_inrel[-1], kv_inrel[-1]])
  block.add_op(wo_prefetch)

  attention_dispatches: list[NestDispatchOp] = []
  for query_block in range(query_blocks):
    att = NestDispatchOp(
      f"prefill_attention_q{query_block}",
      tasks,
      [],
      [q_l2, k_l2, v_l2],
      [o_l2[query_block]],
      f"att_grid_{query_block}",
      f"att_inrel_{query_block}",
      f"att_out_{query_block}",
      l1_mode=0,
      bindings=[q_l2, k_l2, v_l2, o_l2[query_block]],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=retire([q_out[-1], kv_out[-1]]),
    )
    attention_dispatches.append(att)
    grid_dones.append(att.grid_done)
    block.add_op(att)

  outproj_dispatches: dict[tuple[int, int], NestDispatchOp] = {}
  for query_block in range(query_blocks):
    for half in range(out_halves):
      outproj = NestDispatchOp(
        f"prefill_outproj_{'lo' if half == 0 else 'hi'}",
        tasks,
        [],
        [o_l2[query_block], wo_l2],
        [out_l2[query_block * out_halves + half]],
        f"pj_grid_{query_block}_{'lo' if half == 0 else 'hi'}",
        f"pj_inrel_{query_block}_{'lo' if half == 0 else 'hi'}",
        f"pj_out_{query_block}_{'lo' if half == 0 else 'hi'}",
        l1_mode=0,
        bindings=[o_l2[query_block], wo_l2, out_l2[query_block * out_halves + half]],
        signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
        depends_on=retire([attention_dispatches[query_block].output_ready, wo_prefetch.result]),
      )
      outproj_dispatches[(query_block, half)] = outproj
      grid_dones.append(outproj.grid_done)
      block.add_op(outproj)

  store_events: list[OpResult[NestEvent]] = []
  for query_block in range(query_blocks):
    for half in range(out_halves):
      half_name = "lo" if half == 0 else "hi"
      out_view = input_view(
        OUT, [query_block, half, 0, 0, 0], [1, 1, kh, cfg.projection_row_block, q_slice]
      )
      block.add_op(out_view)
      store = NestDMAStoreOp(
        out_l2[query_block * out_halves + half],
        out_view,
        f"out_store_{query_block}_{half_name}",
        depends_on=[outproj_dispatches[(query_block, half)].output_ready],
      )
      store_events.append(store.result)
      block.add_op(store)

  # ---- releases ----
  for p in range(2):
    chunk_ids = list(range(p, nc, 2))
    q_readers = [q_inrel[c] for c in chunk_ids]
    kv_readers = [kv_inrel[c] for c in chunk_ids]
    for kind, buffer, readers in (
      ("x", x_p[p], q_readers + kv_readers),
      ("wq", wq_p[p], q_readers),
      ("wk", wk_p[p], kv_readers),
      ("wv", wv_p[p], kv_readers),
    ):
      block.add_op(
        NestReleaseOp(buffer, depends_on=[*[prefetches[(kind, c)].result for c in chunk_ids], *readers])
      )
  q_readers = [q_inrel[c] for c in range(1, nc)] + [att.input_released for att in attention_dispatches]
  block.add_op(NestReleaseOp(q_l2, depends_on=[*q_readers, *q_out]))
  kv_readers = [kv_inrel[c] for c in range(1, nc)] + [att.input_released for att in attention_dispatches]
  for buffer in (k_l2, v_l2):
    block.add_op(NestReleaseOp(buffer, depends_on=[*kv_readers, *kv_out]))
  block.add_op(
    NestReleaseOp(
      wo_l2,
      depends_on=[
        wo_prefetch.result,
        *[dispatch.input_released for dispatch in outproj_dispatches.values()],
      ],
    )
  )
  for query_block in range(query_blocks):
    half_readers = [outproj_dispatches[(query_block, half)].input_released for half in range(out_halves)]
    block.add_op(
      NestReleaseOp(
        o_l2[query_block], depends_on=[attention_dispatches[query_block].output_ready, *half_readers]
      )
    )
  for query_block, half in outproj_dispatches:
    half_name = "lo" if half == 0 else "hi"
    store_result = next(
      event
      for event in store_events
      if event.name_hint == f"out_store_{query_block}_{half_name}"
    )
    block.add_op(NestReleaseOp(out_l2[query_block * out_halves + half], depends_on=[store_result]))

  block.add_op(NestAwaitOp([*grid_dones, *store_events]))
  block.add_op(NestReturnOp())

  dev = NexusProgramOp(
    "transformer_prefill_dispatchparallel",
    [],
    arg_types=[
      NestGlobalMemref.of([nc, seq, chunk], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, q_slice], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, hd], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, hd], "bf16"),
      NestGlobalMemref.of([kh, cfg.hidden_dim, q_slice], "bf16"),
      NestGlobalMemref.of([query_blocks, out_halves, kh, cfg.projection_row_block, q_slice], "bf16"),
    ],
    arg_names=["X", "WQ", "WK", "WV", "WO", "OUT"],
  )
  submit = NexusSubmitContextOp(
    "prefill_dispatchparallel_ctx", "prefill_dispatchparallel_done", actuals=list(dev.body.block.args)
  )
  dev.body.block.add_ops([submit, NexusAwaitOp([submit.result]), NexusReturnOp()])
  return ModuleOp([*programs, context, dev])


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Generate the dispatch-parallel (no software pipeline) Prefill workload"
  )
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--contexts-per-tile", type=int, choices=(1, 2, 4), default=4)
  args = parser.parse_args()
  output = args.output if args.output.is_absolute() else REPO_ROOT / args.output

  header = [
    "Transformer Prefill Attention, dispatch-parallel producers (one device root, BF16).",
    "",
    "Same logical shapes, packing and attention tail as",
    "transformer_prefill_attention_multicontext.mlir; the three producer programs",
    "drop load hoisting and the schedule compensates with more Grids:",
    "",
    "No load hoisting inside the producers; every tile.await is sunk to the",
    "last dependency-safe point:",
    "- the weight fill co-issues with the first X fill (plus the first",
    "  accumulated partials), behind one merged await before the first BOA;",
    "- each store issues as soon as its own BOA has been awaited and overlaps",
    "  the next independent fill (the K store co-issues with the V BOA);",
    "- no store ever co-issues with an outstanding BOA and no load is hoisted",
    "  across the BOA that reads its buffer, so input_released still fires only",
    "  after the final L2 load and staging recycling cannot start early.",
    "",
    "Parallelism comes from dispatch count instead:",
    "- the Q projection (writes q_l2) and the K/V projection (writes k_l2/v_l2)",
    "  are separate allocations, so each input-K chunk becomes two Grids that",
    "  never contend: 16 QKV dispatches form two independent 8-step chains;",
    "- each query block's output projection is issued twice, over the low and",
    "  high 64-row halves, each with its own output buffer: 8 independent",
    "  outproj Grids (32 partial OUT stores cover the head-major packing);",
    "- attention programs and dispatch structure are unchanged from the baseline.",
    "",
    "Traffic: HBM and L2 payload bytes match the baseline; only dispatch count",
    "grows (20 -> 28 Grids).  A GRID_WINDOW=8 retirement throttle keeps the live",
    "Grid routes inside the 16-entry Group table without any --sim-override.",
    "",
    "TIMING MODEL ONLY: tile.boa.async / tile.evu.async carry no tensor operands and",
    "execute no numerics.  This source validates scheduling, resources and traffic, not",
    "numerical correctness or hardware performance guarantees.",
    "Generate: PYTHONPATH=. python",
    "    examples/generators/generate_transformer_prefill_dispatchparallel.py",
    f"  --contexts-per-tile {args.contexts_per_tile}",
    "Run: bash examples/run.sh transformer-prefill-attention-dispatchparallel",
  ]
  module = make_prefill_dispatchparallel(contexts_per_tile=args.contexts_per_tile)
  write_workload(output, header, module)
  print(f"wrote {output}")


if __name__ == "__main__":
  main()
