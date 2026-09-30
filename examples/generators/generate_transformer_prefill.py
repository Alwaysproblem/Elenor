"""Generate the Transformer prefill attention pipeline workload (.mlir).

One device root (``nest.context``) owns the whole prefill block:

    X, WQ, WK, WV -> K-chunk ping/pong prefetch -> QKV projection (chunk
    dispatches, BOA accumulate over the K axis) -> blocked attention (4
    query blocks x 4 KV blocks, L1 KV ping/pong, per-head BOA QK/PV + EVU
    online softmax) -> output projection (N-split over N-packed Wo) ->
    HBM store.

Everything stays inside one root: cross-root readers are only admitted
after the producer root retires (IR_SPEC 7.2), while the guide requires
attention and the Wo prefetch to start from ``output_ready`` /
``input_released`` of the projection dispatches, i.e. before ``grid_done``.

Timing model only: ``tile.boa.async`` / ``tile.evu.async`` carry no tensor
operands and execute no numerics.  The workload validates scheduling,
lifetimes and traffic -- not Transformer numerical correctness.
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
from pipeline_validator.profiles import ContextResources, TileResources

PLACEMENT = 0x0F  # 4 tiles: tile t owns KV head t and Q heads 4t..4t+3


@dataclass(frozen=True)
class PrefillConfig:
  seq_len: int = 512
  hidden_dim: int = 1024
  q_heads: int = 16
  kv_heads: int = 4
  head_dim: int = 64
  q_block: int = 128
  kv_block: int = 128
  proj_k_chunk: int = 128
  pipelined: bool = True

  def __post_init__(self) -> None:
    ints = (
      "seq_len", "hidden_dim", "q_heads", "kv_heads", "head_dim", "q_block", "kv_block", "proj_k_chunk"
    )
    for name in ints:
      value = getattr(self, name)
      if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    if self.q_heads % self.kv_heads or self.seq_len % self.q_block or self.seq_len % self.kv_block:
      raise ValueError("head/sequence tiling must divide evenly")
    if self.hidden_dim % self.proj_k_chunk:
      raise ValueError("hidden_dim must divide by proj_k_chunk")

  @property
  def q_slice(self) -> int:
    return (self.q_heads // self.kv_heads) * self.head_dim

  @property
  def num_q_blocks(self) -> int:
    return self.seq_len // self.q_block

  @property
  def num_kv_blocks(self) -> int:
    return self.seq_len // self.kv_block

  @property
  def proj_chunks(self) -> int:
    return self.hidden_dim // self.proj_k_chunk


# ---------------------------------------------------------------------------
# Tile programs
# ---------------------------------------------------------------------------


def _qkv_chunk_program(cfg: PrefillConfig, hw: HardwareConfig, accumulate: bool) -> TileProgramDefOp:
  """One K-chunk of the fused QKV projection, per task t (KV head t).

  ``qkv_chunk_init`` overwrites the Q/K/V L1 accumulators (chunk 0).
  ``qkv_chunk_accum`` first loads the partial Q/K/V sums from L2, runs
  BOA with ``accumulate`` and stores the updated sums back (chunks 1..n).
  Two static programs exist because one program cannot vary its loads or
  accumulate flag per dispatch ordinal.
  """
  seq, chunk, q_slice, hd = cfg.seq_len, cfg.proj_k_chunk, cfg.q_slice, cfg.head_dim
  buffers = [
    (seq * chunk * 2, 256),  # x_buf
    (seq * q_slice * 2, 256),  # q_acc
    (chunk * q_slice * 2, 256),  # wq_buf
    (chunk * hd * 2, 256),  # wk_buf
    (seq * hd * 2, 256),  # k_acc
    (chunk * hd * 2, 256),  # wv_buf
    (seq * hd * 2, 256),  # v_acc
  ]
  l1_bytes, allowed = contract_bytes(hw, "l1", buffers)
  name = "qkv_chunk_accum" if accumulate else "qkv_chunk_init"
  prog = TileProgramDefOp(
    name,
    TileResources(allowed_profiles=allowed, tile_l1_spm_bytes_per_context=l1_bytes),
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

  body: list = []
  body.append(l2_view(x_chunk, None, None, [0, 0], [seq, chunk], [seq, chunk], "bf16"))
  body.append(l2_view(wq_chunk, task, 0, [0, 0, 0], [1, chunk, q_slice], [1, chunk, q_slice], "bf16"))
  body.append(l2_view(wk_chunk, task, 0, [0, 0, 0], [1, chunk, hd], [1, chunk, hd], "bf16"))
  body.append(l2_view(wv_chunk, task, 0, [0, 0, 0], [1, chunk, hd], [1, chunk, hd], "bf16"))
  body.append(l2_view(q_l2, task, 0, [0, 0, 0], [1, seq, q_slice], [1, seq, q_slice], "bf16"))
  body.append(l2_view(k_l2, task, 0, [0, 0, 0], [1, seq, hd], [1, seq, hd], "bf16"))
  body.append(l2_view(v_l2, task, 0, [0, 0, 0], [1, seq, hd], [1, seq, hd], "bf16"))
  x_v, wq_v, wk_v, wv_v, q_v, k_v, v_v = body

  x_buf = l1_alloc([seq, chunk], "bf16", "x_buf")
  q_acc = l1_alloc([seq, q_slice], "bf16", "q_acc")
  wq_buf = l1_alloc([chunk, q_slice], "bf16", "wq_buf")
  wk_buf = l1_alloc([chunk, hd], "bf16", "wk_buf")
  k_acc = l1_alloc([seq, hd], "bf16", "k_acc")
  wv_buf = l1_alloc([chunk, hd], "bf16", "wv_buf")
  v_acc = l1_alloc([seq, hd], "bf16", "v_acc")
  body += [x_buf, q_acc, wq_buf, wk_buf, k_acc, wv_buf, v_acc]

  # All loads are issued before input_released (no L2 load may follow it).
  loads = [
    l1_load(x_v, x_buf, "x_loaded"),
    l1_load(wq_v, wq_buf, "wq_loaded"),
    l1_load(wk_v, wk_buf, "wk_loaded"),
    l1_load(wv_v, wv_buf, "wv_loaded"),
  ]
  if accumulate:
    loads += [
      l1_load(q_v, q_acc, "q_partial_loaded"),
      l1_load(k_v, k_acc, "k_partial_loaded"),
      l1_load(v_v, v_acc, "v_partial_loaded"),
    ]
  body += loads
  body.append(TileAwaitOp(loads))
  body.append(TileSignalOp("input_released", task))

  for tag, n_dim, weight, acc_buf, view in (
    ("q", q_slice, wq_buf, q_acc, q_v),
    ("k", hd, wk_buf, k_acc, k_v),
    ("v", hd, wv_buf, v_acc, v_v),
  ):
    del weight
    boa = boa_matmul(seq, n_dim, chunk, f"{tag}_boa", accumulate=accumulate)
    body.append(boa)
    body.append(TileAwaitOp([boa]))
    stored = l1_store(acc_buf, view, f"{tag}_stored")
    body += [stored, TileAwaitOp([stored])]

  body.append(TileSignalOp("output_ready", task))
  body += [TileFreeOp(b) for b in (x_buf, q_acc, wq_buf, wk_buf, k_acc, wv_buf, v_acc)]
  body.append(TileReturnOp())
  prog.body.block.add_ops(body)
  return prog


def _attention_program(cfg: PrefillConfig, hw: HardwareConfig, pipelined: bool) -> TileProgramDefOp:
  """Blocked online-softmax attention for one tile (one KV head, 4 Q heads).

  pipelined: KV ping/pong L1 buffers; load KV(i+1) overlaps QK/softmax/PV(i).
  serial:    all loads of a query block complete before any of its compute.
  """
  qb, kb, q_slice, hd = cfg.q_block, cfg.kv_block, cfg.q_slice, cfg.head_dim
  nq, nkv = cfg.num_q_blocks, cfg.num_kv_blocks
  sm_ops = 4 * qb * kb + 4 * qb * 66  # S elements plus running m/l/out state
  if pipelined:
    buffers = (
      [(qb * q_slice * 2, 256), (qb * q_slice * 2, 256), (qb * 4, 64), (qb * 4, 64)]
      + [(kb * hd * 2, 256)] * 4
    )
    par = 2
  else:
    buffers = (
      [(qb * q_slice * 2, 256), (qb * q_slice * 2, 256), (qb * 4, 64), (qb * 4, 64)]
      + [(kb * hd * 2, 256)] * 8
    )
    par = 4
  l1_bytes, allowed = contract_bytes(hw, "l1", buffers)
  prog = TileProgramDefOp(
    "prefill_attention_tile",
    TileResources(allowed_profiles=allowed, tile_l1_spm_bytes_per_context=l1_bytes),
    arg_types=[
      NestTask(),
      NestBuffer.of([cfg.kv_heads, cfg.seq_len, q_slice], "bf16"),
      NestBuffer.of([cfg.kv_heads, cfg.seq_len, hd], "bf16"),
      NestBuffer.of([cfg.kv_heads, cfg.seq_len, hd], "bf16"),
      NestBuffer.of([cfg.kv_heads, cfg.seq_len, q_slice], "bf16"),
    ],
    arg_names=["task", "q_l2", "k_l2", "v_l2", "o_l2"],
  )
  task = prog.body.block.args[0]
  q_l2, k_l2, v_l2, o_l2 = prog.body.block.args[1:]

  body: list = []
  q_buf = l1_alloc([qb, q_slice], "bf16", "q_buf")
  acc = l1_alloc([qb, q_slice], "bf16", "acc")
  m_buf = l1_alloc([qb], "f32", "m")
  l_buf = l1_alloc([qb], "f32", "l")
  k_bufs = [l1_alloc([kb, hd], "bf16", f"k{p}") for p in range(par)]
  v_bufs = [l1_alloc([kb, hd], "bf16", f"v{p}") for p in range(par)]
  body += [q_buf, acc, m_buf, l_buf, *k_bufs, *v_bufs]

  def q_view(b):
    return l2_view(q_l2, task, 0, [0, b * qb, 0], [1, qb, q_slice], [1, qb, q_slice], "bf16")

  def kv_view(l2, step):
    # Every query block re-scans the same nkv KV blocks of its KV head; the
    # token offset wraps per query block while tags/buffers use the global step.
    return l2_view(
      l2, task, 0, [0, (step % nkv) * kb, 0], [1, kb, hd], [1, kb, hd], "bf16"
    )

  def o_view(b):
    return l2_view(o_l2, task, 0, [0, b * qb, 0], [1, qb, q_slice], [1, qb, q_slice], "bf16")

  def qk_heads(step):
    return [boa_matmul(qb, kb, hd, f"qk{step}_h{h}") for h in range(4)]

  def pv_heads(step):
    return [boa_matmul(qb, hd, kb, f"pv{step}_h{h}", accumulate=step > 0) for h in range(4)]

  def softmax(step):
    return evu("online_softmax_update", sm_ops, f"sm{step}")

  if pipelined:
    for b in range(nq):
      qv = q_view(b)
      body.append(qv)
      q_load = l1_load(qv, q_buf, f"q{b}_loaded")
      body.append(q_load)
      # Distance-2 KV ping/pong (guide section 6): startup issues steps 0,1;
      # while compute(i) is in flight, load(i+2) refills step i's buffer.
      loads: dict[int, list] = {}
      for i in range(min(2, nkv)):
        step = b * nkv + i
        kvv = kv_view(k_l2, step)
        vvv = kv_view(v_l2, step)
        kl = l1_load(kvv, k_bufs[step % 2], f"k{step}_loaded")
        vl = l1_load(vvv, v_bufs[step % 2], f"v{step}_loaded")
        body += [kvv, vvv, kl, vl]
        loads[step] = [kl, vl]
      body.append(TileAwaitOp([q_load]))
      for i in range(nkv):
        step = b * nkv + i
        qks = qk_heads(step)
        body.append(TileAwaitOp(loads[step]))
        body += qks
        body.append(TileAwaitOp(qks))
        sm = softmax(step)
        body.append(sm)
        body.append(TileAwaitOp([sm]))
        pvs = pv_heads(step)
        body += pvs
        if i + 2 < nkv:
          nxt = step + 2
          kvv = kv_view(k_l2, nxt)
          vvv = kv_view(v_l2, nxt)
          kl = l1_load(kvv, k_bufs[nxt % 2], f"k{nxt}_loaded")
          vl = l1_load(vvv, v_bufs[nxt % 2], f"v{nxt}_loaded")
          body += [kvv, vvv, kl, vl]
          loads[nxt] = [kl, vl]
        body.append(TileAwaitOp(pvs))
      ov = o_view(b)
      body.append(ov)
      stored = l1_store(acc, ov, f"o{b}_stored")
      body += [stored, TileAwaitOp([stored])]
    # input_released follows the last awaited KV load by construction.
    body.append(TileSignalOp("input_released", task))
    body.append(TileSignalOp("output_ready", task))
  else:
    # Serial baseline: per query block, every load of that block completes
    # before its compute; the next block's loads start after the previous
    # block's compute and stores.  input_released still follows the last
    # awaited load (no L2 load may follow it, IR_SPEC 4.10).
    for b in range(nq):
      qv = q_view(b)
      body.append(qv)
      ql = l1_load(qv, q_buf, f"q{b}_loaded")
      body.append(ql)
      stage: list = [ql]
      for i in range(nkv):
        step = b * nkv + i
        kvv = kv_view(k_l2, step)
        body.append(kvv)
        kl = l1_load(kvv, k_bufs[i], f"k{step}_loaded")
        vvv = kv_view(v_l2, step)
        body.append(vvv)
        vl = l1_load(vvv, v_bufs[i], f"v{step}_loaded")
        body += [kl, vl]
        stage += [kl, vl]
      body.append(TileAwaitOp(stage))
      for i in range(nkv):
        step = b * nkv + i
        qks = qk_heads(step)
        body += qks
        body.append(TileAwaitOp(qks))
        sm = softmax(step)
        body.append(sm)
        body.append(TileAwaitOp([sm]))
        pvs = pv_heads(step)
        body += pvs
        body.append(TileAwaitOp(pvs))
      ov = o_view(b)
      body.append(ov)
      stored = l1_store(acc, ov, f"o{b}_stored")
      body += [stored, TileAwaitOp([stored])]
    body.append(TileSignalOp("input_released", task))
    body.append(TileSignalOp("output_ready", task))
  body.append(TileReturnOp())
  prog.body.block.add_ops(body)
  return prog


def _outproj_program(cfg: PrefillConfig, hw: HardwareConfig, pipelined: bool) -> TileProgramDefOp:
  """Output projection, N-split across tasks over N-packed Wo.

  Task t computes its q_slice-wide OUT column block as sum_hb O_hb x
  Wo[t, hb] via BOA accumulate over the head blocks.  V1 contiguity allows
  only full-width O rows per head block ([1, seq, q_slice] slices), so the
  K axis is not chunked further; the pipelined variant overlaps the O load
  of head block hb+1 with the BOA of head block hb (Wo itself is only
  q_slice rows per head block and shares the single weight buffer).
  """
  seq, q_slice = cfg.seq_len, cfg.q_slice
  steps = cfg.kv_heads  # one BOA per head block, K = q_slice per step
  if pipelined:
    buffers = [
      (seq * q_slice * 2, 256),  # acc
      (seq * q_slice * 2, 256),  # o0/o1 double buffer
      (seq * q_slice * 2, 256),
      (q_slice * q_slice * 2, 256),  # w (q_slice x q_slice Wo slice)
    ]
  else:
    buffers = [
      (seq * q_slice * 2, 256),  # acc
      (seq * q_slice * 2, 256),  # o
      (q_slice * q_slice * 2, 256),  # w
    ]
  l1_bytes, allowed = contract_bytes(hw, "l1", buffers)
  prog = TileProgramDefOp(
    "prefill_outproj_tile",
    TileResources(allowed_profiles=allowed, tile_l1_spm_bytes_per_context=l1_bytes),
    arg_types=[
      NestTask(),
      NestBuffer.of([cfg.kv_heads, seq, q_slice], "bf16"),
      NestBuffer.of([cfg.kv_heads, cfg.hidden_dim, q_slice], "bf16"),
      NestBuffer.of([cfg.kv_heads, seq, q_slice], "bf16"),
    ],
    arg_names=["task", "o_l2", "wo_l2", "out_l2"],
  )
  task = prog.body.block.args[0]
  o_l2, wo_l2, out_l2 = prog.body.block.args[1:]

  body: list = []
  acc = l1_alloc([seq, q_slice], "bf16", "acc")
  par = 2 if pipelined else 1
  o_bufs = [l1_alloc([seq, q_slice], "bf16", f"o{p}") for p in range(par)]
  w_bufs = [l1_alloc([q_slice, q_slice], "bf16", "w")]
  body += [acc, *o_bufs, *w_bufs]

  def o_view(hb):
    return l2_view(o_l2, None, None, [hb, 0, 0], [1, seq, q_slice], [1, seq, q_slice], "bf16")

  def wo_view(hb):
    # Wo is N-packed per task: Wo_packed[t] rows hb*q_slice..(hb+1)*q_slice.
    return l2_view(
      wo_l2,
      task,
      0,
      [0, hb * q_slice, 0],
      [1, q_slice, q_slice],
      [1, q_slice, q_slice],
      "bf16",
    )

  out_v = l2_view(out_l2, task, 0, [0, 0, 0], [1, seq, q_slice], [1, seq, q_slice], "bf16")
  body.append(out_v)

  if pipelined:
    # Single shared Wo buffer: W(s+1) may only be loaded after BOA(s) has
    # been awaited (W(s) consumed).  The O double buffer lets O(s+1) load
    # overlap BOA(s).
    loads: dict[int, list] = {}
    ov = o_view(0)
    wv = wo_view(0)
    ol = l1_load(ov, o_bufs[0], "o0_loaded")
    wl = l1_load(wv, w_bufs[0], "w0_loaded")
    body += [ov, wv, ol, wl]
    loads[0] = [ol, wl]
    for s in range(steps):
      boa = boa_matmul(seq, q_slice, q_slice, f"pj{s}", accumulate=s > 0)
      body.append(TileAwaitOp(loads[s]))
      body.append(boa)
      pending_o = None
      if s + 1 < steps:
        ov = o_view(s + 1)
        pending_o = l1_load(ov, o_bufs[(s + 1) % 2], f"o{s + 1}_loaded")
        body.append(ov)
        body.append(pending_o)
      body.append(TileAwaitOp([boa]))
      if s + 1 < steps:
        wv = wo_view(s + 1)
        wl = l1_load(wv, w_bufs[0], f"w{s + 1}_loaded")
        body.append(wv)
        body.append(wl)
        loads[s + 1] = [pending_o, wl]
  else:
    for s in range(steps):
      ov = o_view(s)
      body.append(ov)
      ol = l1_load(ov, o_bufs[0], f"o{s}_loaded")
      wv = wo_view(s)
      body.append(wv)
      wl = l1_load(wv, w_bufs[0], f"w{s}_loaded")
      body += [ol, wl]
      body.append(TileAwaitOp([ol, wl]))
      boa = boa_matmul(seq, q_slice, q_slice, f"pj{s}", accumulate=s > 0)
      body.append(boa)
      body.append(TileAwaitOp([boa]))
  body.append(TileSignalOp("input_released", task))
  stored = l1_store(acc, out_v, "out_stored")
  body += [stored, TileAwaitOp([stored])]
  body.append(TileSignalOp("output_ready", task))
  body.append(TileReturnOp())
  prog.body.block.add_ops(body)
  return prog


# ---------------------------------------------------------------------------
# Context + device program
# ---------------------------------------------------------------------------


def make_prefill_attention(cfg: PrefillConfig, *, hw: HardwareConfig | None = None) -> ModuleOp:
  hw = hw or HardwareConfig()
  seq, chunk, q_slice, hd = cfg.seq_len, cfg.proj_k_chunk, cfg.q_slice, cfg.head_dim
  nc = cfg.proj_chunks
  pipelined = cfg.pipelined
  kh = cfg.kv_heads

  l2_buffers = [
    (seq * chunk * 2, 256),  # x ping/pong
    (seq * chunk * 2, 256),
    (kh * chunk * q_slice * 2, 256),  # wq ping/pong
    (kh * chunk * q_slice * 2, 256),
    (kh * chunk * hd * 2, 256),  # wk ping/pong
    (kh * chunk * hd * 2, 256),
    (kh * chunk * hd * 2, 256),  # wv ping/pong
    (kh * chunk * hd * 2, 256),
    (kh * seq * q_slice * 2, 256),  # q_l2
    (kh * seq * hd * 2, 256),  # k_l2
    (kh * seq * hd * 2, 256),  # v_l2
    (kh * cfg.hidden_dim * q_slice * 2, 256),  # wo_l2
    (kh * seq * q_slice * 2, 256),  # o_l2
    (kh * seq * q_slice * 2, 256),  # out_l2
  ]
  l2_bytes, allowed_l2 = contract_bytes(hw, "l2", l2_buffers)
  logical_tasks = (nc + 2) * kh

  progs = [
    _qkv_chunk_program(cfg, hw, accumulate=False),
    _qkv_chunk_program(cfg, hw, accumulate=True),
    _attention_program(cfg, hw, pipelined=pipelined),
    _outproj_program(cfg, hw, pipelined=pipelined),
  ]

  ctx = NestContextOp(
    "prefill_ctx",
    ContextResources(
      l2_mode=0,
      allowed_profiles=allowed_l2,
      logical_tasks=logical_tasks,
      l2_spm_bytes=l2_bytes,
      requested_contexts_per_tile=1,
    ),
    placement=PLACEMENT,
    arg_types=[
      NestGlobalMemref.of([nc, seq, chunk], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, q_slice], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, hd], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, hd], "bf16"),
      NestGlobalMemref.of([kh, cfg.hidden_dim, q_slice], "bf16"),
      NestGlobalMemref.of([kh, seq, q_slice], "bf16"),
    ],
    arg_names=["X", "WQ", "WK", "WV", "WO", "OUT"],
  )
  X, WQ, WK, WV, WO, OUT = ctx.body.block.args
  block = ctx.body.block

  x_p = [NestAllocOp(f"x_p{p}", "in", [seq, chunk], "bf16", alignment=256) for p in range(2)]
  wq_p = [
    NestAllocOp(f"wq_p{p}", "in", [kh, chunk, q_slice], "bf16", alignment=256) for p in range(2)
  ]
  wk_p = [NestAllocOp(f"wk_p{p}", "in", [kh, chunk, hd], "bf16", alignment=256) for p in range(2)]
  wv_p = [NestAllocOp(f"wv_p{p}", "in", [kh, chunk, hd], "bf16", alignment=256) for p in range(2)]
  q_l2 = NestAllocOp(
    "q_l2", "inout", [kh, seq, q_slice], "bf16", sharing="context-local", alignment=256
  )
  k_l2 = NestAllocOp(
    "k_l2", "inout", [kh, seq, hd], "bf16", sharing="context-local", alignment=256
  )
  v_l2 = NestAllocOp(
    "v_l2", "inout", [kh, seq, hd], "bf16", sharing="context-local", alignment=256
  )
  wo_l2 = NestAllocOp("wo_l2", "in", [kh, cfg.hidden_dim, q_slice], "bf16", alignment=256)
  o_l2 = NestAllocOp(
    "o_l2", "inout", [kh, seq, q_slice], "bf16", sharing="context-local", alignment=256
  )
  out_l2 = NestAllocOp("out_l2", "out", [kh, seq, q_slice], "bf16", alignment=256)
  block.add_ops(x_p + wq_p + wk_p + wv_p + [q_l2, k_l2, v_l2, wo_l2, o_l2, out_l2])

  def x_view(c):
    return NestSubviewOp(
      X, [c, 0, 0], [1, seq, chunk], [1, 1, 1], NestGlobalView.of([1, seq, chunk], "bf16")
    )

  def wq_view(c):
    return NestSubviewOp(
      WQ,
      [c, 0, 0, 0],
      [1, kh, chunk, q_slice],
      [1, 1, 1, 1],
      NestGlobalView.of([1, kh, chunk, q_slice], "bf16"),
    )

  def wk_view(c):
    return NestSubviewOp(
      WK,
      [c, 0, 0, 0],
      [1, kh, chunk, hd],
      [1, 1, 1, 1],
      NestGlobalView.of([1, kh, chunk, hd], "bf16"),
    )

  def wv_view(c):
    return NestSubviewOp(
      WV,
      [c, 0, 0, 0],
      [1, kh, chunk, hd],
      [1, 1, 1, 1],
      NestGlobalView.of([1, kh, chunk, hd], "bf16"),
    )

  wo_view = NestSubviewOp(
    WO,
    [0, 0, 0],
    [kh, cfg.hidden_dim, q_slice],
    [1, 1, 1],
    NestGlobalView.of([kh, cfg.hidden_dim, q_slice], "bf16"),
  )
  out_view = NestSubviewOp(
    OUT,
    [0, 0, 0],
    [kh, seq, q_slice],
    [1, 1, 1],
    NestGlobalView.of([kh, seq, q_slice], "bf16"),
  )

  tasks = NestTaskRangeOp(0, kh)
  block.add_op(tasks)

  # Chunk dispatches interleave with their prefetches: prefetch(c) for c>=2
  # waits for the inrel of dispatch c-2 (L2 ping/pong distance two).
  pre: dict[tuple[str, int], NestPrefetchOp] = {}
  grid_events: dict[int, OpResult[NestEvent]] = {}
  inrel_events: dict[int, OpResult[NestEvent]] = {}
  out_events: dict[int, OpResult[NestEvent]] = {}
  for c in range(nc):
    p = c % 2
    gate = (inrel_events[c - 2],) if pipelined and c >= 2 else ()
    xv, wqv, wkv, wvv = x_view(c), wq_view(c), wk_view(c), wv_view(c)
    block.add_ops([xv, wqv, wkv, wvv])
    pre[("x", c)] = NestPrefetchOp(xv, x_p[p], f"pre_x_{c}", depends_on=gate)
    pre[("wq", c)] = NestPrefetchOp(wqv, wq_p[p], f"pre_wq_{c}", depends_on=gate)
    pre[("wk", c)] = NestPrefetchOp(wkv, wk_p[p], f"pre_wk_{c}", depends_on=gate)
    pre[("wv", c)] = NestPrefetchOp(wvv, wv_p[p], f"pre_wv_{c}", depends_on=gate)
    block.add_ops([pre[("x", c)], pre[("wq", c)], pre[("wk", c)], pre[("wv", c)]])
    if not pipelined:
      # Baseline: frontend fence -- no prefetch may overlap a dispatch.
      block.add_op(NestAwaitOp([pre[("x", c)], pre[("wq", c)], pre[("wk", c)], pre[("wv", c)]]))

    prog_name = "qkv_chunk_init" if c == 0 else "qkv_chunk_accum"
    weight_bufs = [x_p[p], wq_p[p], wk_p[p], wv_p[p]]
    ins = [*weight_bufs] if c == 0 else [*weight_bufs, q_l2, k_l2, v_l2]
    depends: list = [pre[("x", c)], pre[("wq", c)], pre[("wk", c)], pre[("wv", c)]]
    if c > 0:
      depends.append(out_events[c - 1])
    d = NestDispatchOp(
      prog_name,
      tasks,
      [],
      ins,
      [q_l2, k_l2, v_l2],
      f"d{c}_grid",
      f"d{c}_inrel",
      f"d{c}_out",
      l1_mode=0,
      bindings=[*weight_bufs, q_l2, k_l2, v_l2],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=depends,
    )
    grid_events[c] = d.grid_done
    inrel_events[c] = d.input_released
    out_events[c] = d.output_ready
    block.add_op(d)
    if not pipelined:
      block.add_op(NestAwaitOp([grid_events[c]]))

  block.add_op(wo_view)
  wo_pre = NestPrefetchOp(
    wo_view, wo_l2, "pre_wo", depends_on=(inrel_events[nc - 1],) if pipelined else ()
  )
  block.add_op(wo_pre)
  if not pipelined:
    block.add_op(NestAwaitOp([wo_pre]))

  att = NestDispatchOp(
    "prefill_attention_tile",
    tasks,
    [],
    [q_l2, k_l2, v_l2],
    [o_l2],
    "att_grid",
    "att_inrel",
    "att_out",
    l1_mode=0,
    bindings=[q_l2, k_l2, v_l2, o_l2],
    signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
    depends_on=[out_events[nc - 1]],
  )
  block.add_op(att)
  if not pipelined:
    block.add_op(NestAwaitOp([att.grid_done]))

  pj = NestDispatchOp(
    "prefill_outproj_tile",
    tasks,
    [],
    [o_l2, wo_l2],
    [out_l2],
    "pj_grid",
    "pj_inrel",
    "pj_out",
    l1_mode=0,
    bindings=[o_l2, wo_l2, out_l2],
    signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
    depends_on=[att.output_ready, wo_pre],
  )
  block.add_op(pj)
  if not pipelined:
    block.add_op(NestAwaitOp([pj.grid_done]))

  block.add_op(out_view)
  store = NestDMAStoreOp(out_l2, out_view, "out_store_done", depends_on=(pj.output_ready,))
  block.add_op(store)
  if not pipelined:
    block.add_op(NestAwaitOp([store]))

  # One release per physical buffer: deps = every prefetch into it plus the
  # input_released of every dispatch that reads it (R union P, IR_SPEC 3.8).
  def release(buf, deps) -> None:
    block.add_op(NestReleaseOp(buf, depends_on=list(dict.fromkeys(deps))))

  for p in range(2):
    parity_chunks = list(range(p, nc, 2))
    pre_events = [pre[(kind, c)] for c in parity_chunks for kind in ("x", "wq", "wk", "wv")]
    readers = [inrel_events[c] for c in parity_chunks]
    release(x_p[p], [*pre_events[0::4], *readers])
    release(wq_p[p], [*pre_events[1::4], *readers])
    release(wk_p[p], [*pre_events[2::4], *readers])
    release(wv_p[p], [*pre_events[3::4], *readers])

  qkv_deps = (
    [inrel_events[c] for c in range(1, nc)]
    + [out_events[c] for c in range(nc)]
    + [att.input_released]
  )
  release(q_l2, qkv_deps)
  release(k_l2, qkv_deps)
  release(v_l2, qkv_deps)
  release(o_l2, [att.output_ready, pj.input_released])
  release(wo_l2, [wo_pre, pj.input_released])
  release(out_l2, [store])

  block.add_op(NestAwaitOp([att.grid_done, pj.grid_done, store]))
  block.add_op(NestReturnOp())

  dev = NexusProgramOp(
    "transformer_prefill_attention",
    [],
    arg_types=[
      NestGlobalMemref.of([nc, seq, chunk], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, q_slice], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, hd], "bf16"),
      NestGlobalMemref.of([nc, kh, chunk, hd], "bf16"),
      NestGlobalMemref.of([kh, cfg.hidden_dim, q_slice], "bf16"),
      NestGlobalMemref.of([kh, seq, q_slice], "bf16"),
    ],
    arg_names=["X", "WQ", "WK", "WV", "WO", "OUT"],
  )
  submit = NexusSubmitContextOp("prefill_ctx", "prefill_done", actuals=list(dev.body.block.args))
  dev.body.block.add_ops([submit, NexusAwaitOp([submit.result]), NexusReturnOp()])

  return ModuleOp([*progs, ctx, dev])


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate the Transformer prefill attention workload")
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("examples/workloads/transformer_prefill_attention_pipeline.mlir"),
  )
  parser.add_argument(
    "--baseline-output",
    type=Path,
    default=Path("examples/workloads/transformer_prefill_attention_baseline.mlir"),
  )
  parser.add_argument("--seq-len", type=int, default=512)
  parser.add_argument("--kv-block", type=int, default=128)
  parser.add_argument("--proj-k-chunk", type=int, default=128)
  args = parser.parse_args()

  header = [
    "Transformer Prefill Attention block (one device root, GQA 16:4, BF16).",
    "",
    "Shapes: seq=512, hidden=1024, q_heads=16, kv_heads=4, head_dim=64;",
    "QUERY_BLOCK=128, KV_BLOCK=128, QKV projection K-chunk=128.",
    "Tile t (placement 15) owns KV head t and Q heads 4t..4t+3.",
    "",
    "TIMING MODEL ONLY: tile.boa.async / tile.evu.async carry no tensor",
    "operands and execute no numerics.  This workload validates Group",
    "ready-action scheduling, L2 lifetimes (input_released / output_ready),",
    "HBM->L2->L1 streaming and BOA/EVU/MFE overlap -- NOT numerical",
    "correctness of Transformer attention.",
    "",
    "Pipelined variant: per K-chunk c the X/WQ/WK/WV prefetch (L2",
    "ping/pong, gated on chunk c-2 input_released) feeds qkv_chunk_init",
    "(chunk 0) / qkv_chunk_accum (chunks 1..7, read-modify-write of the",
    "L2 partial sums); attention runs 4 query blocks x 4 KV blocks with",
    "L1 KV ping/pong (per-head BOA QK + EVU online softmax + BOA PV); the",
    "Wo prefetch starts at the last chunk's input_released and overlaps",
    "attention; the N-split output projection consumes O and Wo, then one",
    "HBM store.  Baseline variant: identical buffers and traffic, but",
    "every stage is fenced with per-chunk nest.await and tile programs",
    "issue all loads of a stage before any compute.",
    "",
    "Useful MACs: QKV 512x1024x1536; attention 4 tiles x 16 x 4 heads x",
    "(128x128x64 + 128x64x128); output projection 512x1024x1024.",
    "",
    "Run: bash examples/run.sh transformer-prefill-attention",
  ]

  pipelined = PrefillConfig(
    seq_len=args.seq_len, kv_block=args.kv_block, proj_k_chunk=args.proj_k_chunk, pipelined=True
  )
  write_workload(args.output, header, make_prefill_attention(pipelined))
  baseline = PrefillConfig(
    seq_len=args.seq_len, kv_block=args.kv_block, proj_k_chunk=args.proj_k_chunk, pipelined=False
  )
  write_workload(args.baseline_output, header, make_prefill_attention(baseline))
  print(f"wrote {args.output}")
  print(f"wrote {args.baseline_output}")


if __name__ == "__main__":
  main()
