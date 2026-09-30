"""Generate a query-row-partitioned Transformer prefill workload.

The single root projects Q/K/V once, then launches independent attention
and output-projection grids for each query-row block.  Each query block has
its own writable L2 buffers, allowing up to four contexts per tile to work
on separate blocks without duplicating the logical matrix operations.

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
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "examples/workloads/transformer_prefill_attention_multicontext.mlir"


@dataclass(frozen=True)
class PrefillMultiContextConfig:
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


def _qkv_chunk_program(
  cfg: PrefillMultiContextConfig, hw: HardwareConfig, contexts_per_tile: int, accumulate: bool
) -> TileProgramDefOp:
  """Project one input-K chunk over disjoint 64-row tiles, per KV-head tile."""
  seq, rows, chunk = cfg.seq_len, cfg.projection_row_block, cfg.proj_k_chunk
  q_slice, hd = cfg.q_slice, cfg.head_dim
  name = "qkv_chunk_accum" if accumulate else "qkv_chunk_init"
  buffers = [
    (rows * chunk * 2, 256),
    (rows * q_slice * 2, 256),
    (chunk * q_slice * 2, 256),
    (rows * hd * 2, 256),
    (chunk * hd * 2, 256),
    (rows * hd * 2, 256),
    (chunk * hd * 2, 256),
  ]
  prog = TileProgramDefOp(
    name,
    _tile_contract(hw, buffers, contexts_per_tile),
    arg_types=[
      NestTask(),
      NestBuffer.of([seq, chunk], "bf16"),
      NestBuffer.of([cfg.kv_heads, chunk, q_slice], "bf16"),
      NestBuffer.of([cfg.kv_heads, chunk, hd], "bf16"),
      NestBuffer.of([cfg.kv_heads, chunk, hd], "bf16"),
      NestBuffer.of([cfg.kv_heads, seq, q_slice], "bf16"),
      NestBuffer.of([cfg.kv_heads, seq, hd], "bf16"),
      NestBuffer.of([cfg.kv_heads, seq, hd], "bf16"),
    ],
    arg_names=["task", "x_chunk", "wq_chunk", "wk_chunk", "wv_chunk", "q_l2", "k_l2", "v_l2"],
  )
  task = prog.body.block.args[0]
  x_chunk, wq_chunk, wk_chunk, wv_chunk, q_l2, k_l2, v_l2 = prog.body.block.args[1:]

  x_buf = l1_alloc([rows, chunk], "bf16", "x_buf")
  q_acc = l1_alloc([rows, q_slice], "bf16", "q_acc")
  wq_buf = l1_alloc([chunk, q_slice], "bf16", "wq_buf")
  k_acc = l1_alloc([rows, hd], "bf16", "k_acc")
  wk_buf = l1_alloc([chunk, hd], "bf16", "wk_buf")
  v_acc = l1_alloc([rows, hd], "bf16", "v_acc")
  wv_buf = l1_alloc([chunk, hd], "bf16", "wv_buf")
  body: list = [x_buf, q_acc, wq_buf, k_acc, wk_buf, v_acc, wv_buf]

  # The K-chunk weights are loaded once and reused for every row block.
  wq_v = l2_view(wq_chunk, task, 0, [0, 0, 0], [1, chunk, q_slice], [1, chunk, q_slice], "bf16")
  wk_v = l2_view(wk_chunk, task, 0, [0, 0, 0], [1, chunk, hd], [1, chunk, hd], "bf16")
  wv_v = l2_view(wv_chunk, task, 0, [0, 0, 0], [1, chunk, hd], [1, chunk, hd], "bf16")
  wq_loaded = l1_load(wq_v, wq_buf, "wq_loaded")
  wk_loaded = l1_load(wk_v, wk_buf, "wk_loaded")
  wv_loaded = l1_load(wv_v, wv_buf, "wv_loaded")
  body += [wq_v, wk_v, wv_v, wq_loaded, wk_loaded, wv_loaded]
  body.append(TileAwaitOp([wq_loaded, wk_loaded, wv_loaded]))

  for row_block in range(cfg.projection_row_blocks):
    row = row_block * rows
    x_v = l2_view(x_chunk, None, None, [row, 0], [rows, chunk], [rows, chunk], "bf16")
    q_v = l2_view(q_l2, task, 0, [0, row, 0], [1, rows, q_slice], [1, rows, q_slice], "bf16")
    k_v = l2_view(k_l2, task, 0, [0, row, 0], [1, rows, hd], [1, rows, hd], "bf16")
    v_v = l2_view(v_l2, task, 0, [0, row, 0], [1, rows, hd], [1, rows, hd], "bf16")
    body += [x_v, q_v, k_v, v_v]
    loads = [l1_load(x_v, x_buf, f"x_loaded_{row_block}")]
    if accumulate:
      loads += [
        l1_load(q_v, q_acc, f"q_partial_loaded_{row_block}"),
        l1_load(k_v, k_acc, f"k_partial_loaded_{row_block}"),
        l1_load(v_v, v_acc, f"v_partial_loaded_{row_block}"),
      ]
    body += loads
    body.append(TileAwaitOp(loads))
    if row_block == cfg.projection_row_blocks - 1:
      # No later L2 reads occur in this program; L1 BOAs/stores may continue.
      body.append(TileSignalOp("input_released", task))

    stores = []
    for tag, n_dim, acc_buf, view in (
      ("q", q_slice, q_acc, q_v),
      ("k", hd, k_acc, k_v),
      ("v", hd, v_acc, v_v),
    ):
      boa = boa_matmul(rows, n_dim, chunk, f"{tag}_boa_{row_block}", accumulate=accumulate)
      body += [boa, TileAwaitOp([boa])]
      stored = l1_store(acc_buf, view, f"{tag}_stored_{row_block}")
      body.append(stored)
      stores.append(stored)
    # Each result buffer is distinct: Q/K stores may overlap the following BOA.
    # Drain all stores before the next row reuses any accumulator.
    body.append(TileAwaitOp(stores))

  body.append(TileSignalOp("output_ready", task))
  body += [TileFreeOp(buf) for buf in (x_buf, q_acc, wq_buf, k_acc, wk_buf, v_acc, wv_buf)]
  body.append(TileReturnOp())
  prog.body.block.add_ops(body)
  return prog


def _attention_program(
  cfg: PrefillMultiContextConfig, hw: HardwareConfig, contexts_per_tile: int, query_block: int
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
  cfg: PrefillMultiContextConfig, hw: HardwareConfig, contexts_per_tile: int
) -> TileProgramDefOp:
  """Project one query block while reusing each weight chunk across row halves."""
  qb, q_slice = cfg.query_block, cfg.q_slice
  rows, k_chunk = cfg.projection_row_block, cfg.outproj_k_chunk
  microblocks = qb // rows
  chunks = q_slice // k_chunk
  buffers = [*[(rows * q_slice * 2, 256)] * (2 * microblocks), (k_chunk * q_slice * 2, 256)]
  prog = TileProgramDefOp(
    "prefill_outproj_tile",
    _tile_contract(hw, buffers, contexts_per_tile),
    arg_types=[
      NestTask(),
      NestBuffer.of([cfg.kv_heads, qb, q_slice], "bf16"),
      NestBuffer.of([cfg.kv_heads, cfg.hidden_dim, q_slice], "bf16"),
      NestBuffer.of([cfg.kv_heads, qb, q_slice], "bf16"),
    ],
    arg_names=["task", "o_l2", "wo_l2", "out_l2"],
  )
  task = prog.body.block.args[0]
  o_l2, wo_l2, out_l2 = prog.body.block.args[1:]
  acc_bufs = [l1_alloc([rows, q_slice], "bf16", f"acc{part}") for part in range(microblocks)]
  o_bufs = [l1_alloc([rows, q_slice], "bf16", f"o{part}") for part in range(microblocks)]
  w_buf = l1_alloc([k_chunk, q_slice], "bf16", "w_buf")
  body: list = [*acc_bufs, *o_bufs, w_buf]

  for head_block in range(cfg.kv_heads):
    o_loads = []
    for part in range(microblocks):
      ov = l2_view(
        o_l2, None, None, [head_block, part * rows, 0], [1, rows, q_slice], [1, rows, q_slice], "bf16"
      )
      ol = l1_load(ov, o_bufs[part], f"o{head_block}_p{part}_loaded")
      body += [ov, ol]
      o_loads.append(ol)
    body.append(TileAwaitOp(o_loads))

    for chunk_index in range(chunks):
      row = head_block * q_slice + chunk_index * k_chunk
      wv = l2_view(wo_l2, task, 0, [0, row, 0], [1, k_chunk, q_slice], [1, k_chunk, q_slice], "bf16")
      wl = l1_load(wv, w_buf, f"w{head_block}_{chunk_index}_loaded")
      body += [wv, wl, TileAwaitOp([wl])]
      if head_block == cfg.kv_heads - 1 and chunk_index == chunks - 1:
        body.append(TileSignalOp("input_released", task))
      for part in range(microblocks):
        boa = boa_matmul(
          rows,
          q_slice,
          k_chunk,
          f"outproj{head_block}_{chunk_index}_p{part}",
          accumulate=head_block > 0 or chunk_index > 0,
        )
        body += [boa, TileAwaitOp([boa])]

  output_stores = []
  for part in range(microblocks):
    out_v = l2_view(out_l2, task, 0, [0, part * rows, 0], [1, rows, q_slice], [1, rows, q_slice], "bf16")
    stored = l1_store(acc_bufs[part], out_v, f"out{part}_stored")
    body += [out_v, stored]
    output_stores.append(stored)
  body.append(TileAwaitOp(output_stores))
  body.append(TileSignalOp("output_ready", task))
  body += [TileFreeOp(buf) for buf in (*acc_bufs, *o_bufs, w_buf)]
  body.append(TileReturnOp())
  prog.body.block.add_ops(body)
  return prog


def make_prefill_multicontext(*, hw: HardwareConfig | None = None, contexts_per_tile: int = 4) -> ModuleOp:
  """Build prefill with independent row-block grids and an explicit UCE R lease."""
  if type(contexts_per_tile) is not int or contexts_per_tile not in (1, 2, 4):
    raise ValueError("contexts_per_tile must be one of 1, 2, or 4")
  cfg = PrefillMultiContextConfig()
  hw = hw or HardwareConfig()
  seq, chunk, q_slice, hd = cfg.seq_len, cfg.proj_k_chunk, cfg.q_slice, cfg.head_dim
  kh, nc = cfg.kv_heads, cfg.projection_chunks
  query_blocks = cfg.query_blocks

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
  l2_buffers += [(kh * cfg.query_block * q_slice * 2, 256)] * (2 * query_blocks)
  l2_bytes, allowed_l2 = contract_bytes(hw, "l2", l2_buffers)
  logical_tasks = kh * (nc + 2 * query_blocks)

  programs = [
    _qkv_chunk_program(cfg, hw, contexts_per_tile, accumulate=False),
    _qkv_chunk_program(cfg, hw, contexts_per_tile, accumulate=True),
    *[_attention_program(cfg, hw, contexts_per_tile, block) for block in range(query_blocks)],
    _outproj_program(cfg, hw, contexts_per_tile),
  ]
  context = NestContextOp(
    "prefill_multicontext_ctx",
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
      NestGlobalMemref.of([query_blocks, kh, cfg.query_block, q_slice], "bf16"),
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
    NestAllocOp(f"out_q{query_block}", "out", [kh, cfg.query_block, q_slice], "bf16", alignment=256)
    for query_block in range(query_blocks)
  ]
  block.add_ops(x_p + wq_p + wk_p + wv_p + [q_l2, k_l2, v_l2, wo_l2, *o_l2, *out_l2])
  tasks = NestTaskRangeOp(0, kh)
  block.add_op(tasks)
  qkv_input_released: list[OpResult[NestEvent]] = []
  qkv_output_ready: list[OpResult[NestEvent]] = []
  qkv_grid_done: list[OpResult[NestEvent]] = []

  def input_view(global_arg, offsets, sizes):
    return NestSubviewOp(global_arg, offsets, sizes, [1] * len(offsets), NestGlobalView.of(sizes, "bf16"))

  prefetches: dict[tuple[str, int], NestPrefetchOp] = {}
  for c in range(nc):
    x_view = input_view(X, [c, 0, 0], [1, seq, chunk])
    wq_view = input_view(WQ, [c, 0, 0, 0], [1, kh, chunk, q_slice])
    wk_view = input_view(WK, [c, 0, 0, 0], [1, kh, chunk, hd])
    wv_view = input_view(WV, [c, 0, 0, 0], [1, kh, chunk, hd])
    block.add_ops([x_view, wq_view, wk_view, wv_view])
    previous: tuple[OpResult[NestEvent], ...] = ()
    if c >= 2:
      previous = (qkv_input_released[c - 2],)
    p = c % 2
    for kind, view, destination in (
      ("x", x_view, x_p[p]),
      ("wq", wq_view, wq_p[p]),
      ("wk", wk_view, wk_p[p]),
      ("wv", wv_view, wv_p[p]),
    ):
      event = NestPrefetchOp(view, destination, f"pre_{kind}_{c}", depends_on=previous)
      prefetches[(kind, c)] = event
      block.add_op(event)

    # The tuple is completed after the prior loop iteration has registered its dispatch.
    deps: list[NestPrefetchOp | OpResult[NestEvent]] = [
      prefetches[(kind, c)] for kind in ("x", "wq", "wk", "wv")
    ]
    if c:
      deps.append(qkv_output_ready[c - 1])
    weights = [x_p[p], wq_p[p], wk_p[p], wv_p[p]]
    reads = [*weights, *([q_l2, k_l2, v_l2] if c else [])]
    dispatch = NestDispatchOp(
      "qkv_chunk_init" if c == 0 else "qkv_chunk_accum",
      tasks,
      [],
      reads,
      [q_l2, k_l2, v_l2],
      f"qkv_grid_{c}",
      f"qkv_inrel_{c}",
      f"qkv_out_{c}",
      l1_mode=0,
      bindings=[*weights, q_l2, k_l2, v_l2],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=deps,
    )
    qkv_input_released.append(dispatch.input_released)
    qkv_output_ready.append(dispatch.output_ready)
    qkv_grid_done.append(dispatch.grid_done)
    block.add_op(dispatch)

  wo_view = input_view(WO, [0, 0, 0], [kh, cfg.hidden_dim, q_slice])
  block.add_op(wo_view)
  wo_prefetch = NestPrefetchOp(wo_view, wo_l2, "pre_wo", depends_on=[qkv_input_released[-1]])
  block.add_op(wo_prefetch)
  # Every attention Grid already depends on the final QKV output-ready frontier.
  # Do not wait for QKV Grid retirement here: it needlessly delays registration
  # of independent attention work after its data is ready.
  attention_dispatches: list[NestDispatchOp] = []
  for query_block in range(query_blocks):
    program = f"prefill_attention_q{query_block}"
    att = NestDispatchOp(
      program,
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
      depends_on=[qkv_output_ready[-1]],
    )
    attention_dispatches.append(att)
    block.add_op(att)

  # Register all independent attention Grids before any downstream dispatches
  # whose dependencies are not ready yet. This preserves the attention issue
  # window instead of interleaving it with blocked projection/store actions.
  outproj_dispatches: list[NestDispatchOp] = []
  for query_block in range(query_blocks):
    outproj = NestDispatchOp(
      "prefill_outproj_tile",
      tasks,
      [],
      [o_l2[query_block], wo_l2],
      [out_l2[query_block]],
      f"pj_grid_{query_block}",
      f"pj_inrel_{query_block}",
      f"pj_out_{query_block}",
      l1_mode=0,
      bindings=[o_l2[query_block], wo_l2, out_l2[query_block]],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=[attention_dispatches[query_block].output_ready, wo_prefetch],
    )
    outproj_dispatches.append(outproj)
    block.add_op(outproj)

  store_events: list[OpResult[NestEvent]] = []
  for query_block in range(query_blocks):
    out_view = input_view(OUT, [query_block, 0, 0, 0], [1, kh, cfg.query_block, q_slice])
    block.add_op(out_view)
    store = NestDMAStoreOp(
      out_l2[query_block],
      out_view,
      f"out_store_{query_block}",
      depends_on=[outproj_dispatches[query_block].output_ready],
    )
    store_events.append(store.result)
    block.add_op(store)

  # Release the K/V input ping-pong allocations only after their last readers.
  for p in range(2):
    chunk_ids = list(range(p, nc, 2))
    readers = [qkv_input_released[c] for c in chunk_ids]
    for kind, buffers in (("x", x_p), ("wq", wq_p), ("wk", wk_p), ("wv", wv_p)):
      block.add_op(
        NestReleaseOp(buffers[p], depends_on=[*[prefetches[(kind, c)] for c in chunk_ids], *readers])
      )

  qkv_readers = [qkv_input_released[c] for c in range(1, nc)] + [
    att.input_released for att in attention_dispatches
  ]
  qkv_writers = list(qkv_output_ready)
  for buffer in (q_l2, k_l2, v_l2):
    block.add_op(NestReleaseOp(buffer, depends_on=[*qkv_readers, *qkv_writers]))

  block.add_op(
    NestReleaseOp(
      wo_l2, depends_on=[wo_prefetch, *[dispatch.input_released for dispatch in outproj_dispatches]]
    )
  )
  for query_block in range(query_blocks):
    block.add_op(
      NestReleaseOp(
        o_l2[query_block],
        depends_on=[
          attention_dispatches[query_block].output_ready,
          outproj_dispatches[query_block].input_released,
        ],
      )
    )
    block.add_op(NestReleaseOp(out_l2[query_block], depends_on=[store_events[query_block]]))

  block.add_op(
    NestAwaitOp(
      [
        *qkv_grid_done,
        *[dispatch.grid_done for dispatch in attention_dispatches],
        *[dispatch.grid_done for dispatch in outproj_dispatches],
        *store_events,
      ]
    )
  )
  block.add_op(NestReturnOp())

  dev = NexusProgramOp(
    "transformer_prefill_multicontext",
    [],
    arg_types=[
      NestGlobalMemref.of([nc, seq, chunk], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, q_slice], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, hd], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, hd], "bf16"),
      NestGlobalMemref.of([kh, cfg.hidden_dim, q_slice], "bf16"),
      NestGlobalMemref.of([query_blocks, kh, cfg.query_block, q_slice], "bf16"),
    ],
    arg_names=["X", "WQ", "WK", "WV", "WO", "OUT"],
  )
  submit = NexusSubmitContextOp(
    "prefill_multicontext_ctx", "prefill_multicontext_done", actuals=list(dev.body.block.args)
  )
  dev.body.block.add_ops([submit, NexusAwaitOp([submit.result]), NexusReturnOp()])
  return ModuleOp([*programs, context, dev])


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate the Transformer prefill multicontext workload")
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--contexts-per-tile", type=int, choices=(1, 2, 4), default=4)
  args = parser.parse_args()
  output = args.output if args.output.is_absolute() else REPO_ROOT / args.output

  header = [
    "Transformer Prefill Attention, query-row multicontext schedule (one device root, BF16).",
    "",
    "Logical shapes: seq=512, hidden=1024, q_heads=16, kv_heads=4, head_dim=64;",
    "tile t owns KV head t and Q heads 4t..4t+3 (placement 15).",
    "Tiles: query block=128, internal/query and QKV row tile=64, KV block=128, QKV input-K=128.",
    "Four independent query blocks partition rows 0..511; each attends all four KV blocks.",
    "QKV is projected once across all rows; QKV weights are fetched once per K chunk.",
    "Each query block has distinct writable L2 O/OUT allocations, avoiding",
    "whole-buffer write hazards between concurrently submitted blocks.",
    f"R={args.contexts_per_tile} unpinned UCE Tasks per tile; every dispatch uses task range [0,4).",
    "The R x child per-bank L1 envelope is proved for each advertised L1 mode.",
    "Default root Arena is 6.5 MiB (L2 modes 0/1); all concurrent roots share the Group L2 pool.",
    "At R=4, attention child=240 KiB (15 KiB/bank): 60 KiB/bank fits only mode 0's 62 KiB.",
    "QKV/outproj children are 160 KiB each; all L1 allowed-mode sets are recomputed for R.",
    "",
    "Packing and traffic: HBM inputs retain baseline extents (X [8,512,128],",
    "WQ [8,4,128,256], WK/WV [8,4,128,64], WO [4,1024,256]).",
    "OUT [4,4,128,256] is query-block-major physical packing of logical OUT [4,512,256]:",
    "OUT[qblock,head,row,:] maps to logical OUT[head,qblock*128+row,:]. O/OUT are",
    "split into four 128-row L2 allocations with the same aggregate bytes as baseline.",
    "Q/K/V and input ping-pong retain baseline bytes; WO is still prefetched from HBM once.",
    "Each of four query-block output Tasks loads its tile's 512 KiB WO shard from shared L2,",
    "adding exactly 6 MiB aggregate L2-to-L1 read payload vs baseline (three extra copies).",
    "All other interface payloads, HBM bytes and logical BOA FLOPs match baseline.",
    "",
    "Compute: QKV = 512x1024x1536 useful MACs; attention covers 16 Q heads x 512 queries",
    "x all 512 keys x 64 features for both QK and PV; output projection = 512x1024x1024",
    "Online-softmax L1 state is m/l [4,64]xf32 per 64-row microtile plus one reusable",
    "[64,128]xf32 score tile. Q/K stores overlap the independent following QKV BOAs;",
    "independent final row stores launch together, then share a completion frontier.",
    "Every buffer reuse is gated by the preceding consumers/stores; Arena sizes are unchanged.",
    "",
    "TIMING MODEL ONLY: tile.boa.async / tile.evu.async carry no tensor operands and",
    "execute no numerics. This source validates scheduling, resources and traffic, not",
    "numerical correctness or hardware performance guarantees.",
    "Generate: PYTHONPATH=. python examples/generators/generate_transformer_prefill_multicontext.py",
    f"  --contexts-per-tile {args.contexts_per_tile}",
    "Run: bash examples/run.sh transformer-prefill-attention-multicontext",
  ]
  module = make_prefill_multicontext(contexts_per_tile=args.contexts_per_tile)
  write_workload(output, header, module)
  print(f"wrote {output}")


if __name__ == "__main__":
  main()
