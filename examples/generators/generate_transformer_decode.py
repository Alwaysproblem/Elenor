"""Generate the Transformer decode + KV-cache pipeline workload (.mlir).

One decode step at valid_sequence_length = 2048 (KV_BLOCK=256 -> 8 KV
blocks, GQA 16:4, tile t owns KV head t and Q heads 4t..4t+3).  The KV
cache uses the block-packed layout K_CACHE[kv_block][tile][token][dim]
so every HBM->L2 block fetch is one contiguous transfer.

Pipelined variant: L2 ping/pong buffer sets; prefetch of block b+2 waits
for block b's ``input_released`` (the tile has copied the block into L1,
so the L2 buffer may be overwritten), while dispatch b+1 additionally
waits block b's ``output_ready`` (online-softmax state is loop-carried
through the context-local L2 state buffer).  Baseline variant: strictly
prefetch -> await -> dispatch -> await per block with a single buffer set.

One dispatch processes exactly one KV block; running softmax state
(m, l, out[64] per query head) lives in a small context-local L2 buffer
that every block reads and writes back.

This example represents one decoder iteration at
valid_sequence_length = 2048.

Future device-side loop support may turn the append offset into a runtime
loop-carried value; the current IR has no dynamic addresses, so the
position is static and no token-generation loop is modeled.

Timing model only: tile.boa.async / tile.evu.async carry no tensor
operands and execute no numerics; the workload measures KV-block
pipelining, not numerical correctness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from transformer_common import contract_bytes, l2_view, write_workload
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
from pipeline_validator.profiles import ContextResources, TileResources

PLACEMENT = 0x0F


@dataclass(frozen=True)
class DecodeConfig:
  seq_len: int = 2048
  q_heads: int = 16
  kv_heads: int = 4
  head_dim: int = 64
  kv_block: int = 256
  num_requests: int = 1
  pipelined: bool = True

  def __post_init__(self) -> None:
    for name in ("seq_len", "q_heads", "kv_heads", "head_dim", "kv_block", "num_requests"):
      value = getattr(self, name)
      if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    if self.q_heads % self.kv_heads or self.seq_len % self.kv_block:
      raise ValueError("head/sequence tiling must divide evenly")

  @property
  def blocks(self) -> int:
    return self.seq_len // self.kv_block


def _decode_block_program(cfg: DecodeConfig, hw: HardwareConfig) -> TileProgramDefOp:
  """One KV-block attention update per task (tile t == KV head t)."""
  kb, hd = cfg.kv_block, cfg.head_dim
  heads = cfg.q_heads // cfg.kv_heads
  buffers = [
    (kb * hd * 2, 256),  # k_l1
    (kb * hd * 2, 256),  # v_l1
    (heads * hd * 2, 256),  # q_l1
    (heads * 66 * 4, 64),  # state_l1 f32: m, l, out[hd] per head
    (heads * hd * 4, 64),  # out_l1 f32
  ]
  l1_bytes, allowed = contract_bytes(hw, "l1", buffers)
  sm_ops = heads * kb + heads * 66  # score elements + running state elements
  prog = TileProgramDefOp(
    "decode_attention_block",
    TileResources(allowed_profiles=allowed, tile_l1_spm_bytes_per_context=l1_bytes),
    arg_types=[
      NestTask(),
      NestBuffer.of([cfg.kv_heads, heads, hd], "bf16"),  # q_l2
      NestBuffer.of([cfg.kv_heads, kb, hd], "bf16"),  # k block (parity set)
      NestBuffer.of([cfg.kv_heads, kb, hd], "bf16"),  # v block (parity set)
      NestBuffer.of([cfg.kv_heads, heads, 66], "f32"),  # state (ctx-local)
      NestBuffer.of([cfg.kv_heads, heads, hd], "f32"),  # out
    ],
    arg_names=["task", "q_l2", "k_l2", "v_l2", "state", "out_l2"],
  )
  task = prog.body.block.args[0]
  q_l2, k_l2, v_l2, state, out_l2 = prog.body.block.args[1:]

  def view(src, shape, dtype, offsets=(0, 0, 0)):
    return TileSubviewOp(
      src, task, 0, list(offsets), [1, *shape], [1, 1, 1], NestL2View.of([1, *shape], dtype)
    )

  body: list = []
  q_v = view(q_l2, [heads, hd], "bf16")
  k_v = view(k_l2, [kb, hd], "bf16")
  v_v = view(v_l2, [kb, hd], "bf16")
  state_v = view(state, [heads, 66], "f32")
  out_v = view(out_l2, [heads, hd], "f32")
  body += [q_v, k_v, v_v, state_v, out_v]

  k_l1 = TileAllocOp([kb, hd], "bf16", alignment=256)
  v_l1 = TileAllocOp([kb, hd], "bf16", alignment=256)
  q_l1 = TileAllocOp([heads, hd], "bf16", alignment=256)
  state_l1 = TileAllocOp([heads, 66], "f32", alignment=64)
  out_l1 = TileAllocOp([heads, hd], "f32", alignment=64)
  body += [k_l1, v_l1, q_l1, state_l1, out_l1]

  loads = [
    TileLoadOp(q_v.result, q_l1.result, "q_loaded"),
    TileLoadOp(k_v.result, k_l1.result, "k_loaded"),
    TileLoadOp(v_v.result, v_l1.result, "v_loaded"),
    TileLoadOp(state_v.result, state_l1.result, "state_loaded"),
  ]
  body += loads
  body.append(TileAwaitOp(loads))
  # All inputs are in tile-local storage: the L2 block buffers are free.
  body.append(TileSignalOp("input_released", task))

  qk = TileBoaOp("matmul", heads, kb, hd, 2 * heads * kb * hd, "qk_boa")
  body += [qk, TileAwaitOp([qk])]
  sm = TileEvuOp("online_softmax_update", sm_ops, "sm_update")
  body += [sm, TileAwaitOp([sm])]
  pv = TileBoaOp("matmul", heads, hd, kb, 2 * heads * hd * kb, "pv_boa")
  body += [pv, TileAwaitOp([pv])]

  st = TileStoreOp(state_l1.result, state_v.result, "state_stored")
  body += [st, TileAwaitOp([st])]
  ot = TileStoreOp(out_l1.result, out_v.result, "out_stored")
  body += [ot, TileAwaitOp([ot])]
  body.append(TileSignalOp("output_ready", task))
  body += [TileFreeOp(b) for b in (k_l1, v_l1, q_l1, state_l1, out_l1)]
  body.append(TileReturnOp())
  prog.body.block.add_ops(body)
  return prog


def _append_program(cfg: DecodeConfig, hw: HardwareConfig) -> TileProgramDefOp:
  """Timing-only K/V append for the fixed position valid_len = seq_len.

  QKV projection of the current token, per task t (KV head t): K_new[t] =
  H_T x WK_A[t] and V_new[t] = H_T x WV_A[t], each a single m=1, n=hd,
  k=hidden_dim BOA reading the shared current-token hidden state
  [hidden_dim] and the per-tile N-packed projection weight slice
  [hidden_dim, hd]; results go to L2 staging buffers and then to the
  block-packed K_APPEND/V_APPEND globals (the static append slot for
  position seq_len).  BOA carries no operands -- timing only.
  """
  hd = cfg.head_dim
  hidden = 1024  # decoder hidden_dim (guide section 13)
  kh = cfg.kv_heads
  buffers = [
    (hidden * 2, 256),  # h_t
    (hidden * hd * 2, 256),  # wk
    (hidden * hd * 2, 256),  # wv
    (1 * hd * 2, 256),  # k_new
    (1 * hd * 2, 256),  # v_new
  ]
  l1_bytes, allowed = contract_bytes(hw, "l1", buffers)
  prog = TileProgramDefOp(
    "kv_append_tile",
    TileResources(allowed_profiles=allowed, tile_l1_spm_bytes_per_context=l1_bytes),
    arg_types=[
      NestTask(),
      NestBuffer.of([hidden], "bf16"),
      NestBuffer.of([kh, hidden, hd], "bf16"),
      NestBuffer.of([kh, hidden, hd], "bf16"),
      NestBuffer.of([kh, 1, hd], "bf16"),
      NestBuffer.of([kh, 1, hd], "bf16"),
    ],
    arg_names=["task", "h_l2", "wk_l2", "wv_l2", "kn_l2", "vn_l2"],
  )
  task = prog.body.block.args[0]
  h_l2, wk_l2, wv_l2, kn_l2, vn_l2 = prog.body.block.args[1:]

  body: list = []
  h_v = l2_view(h_l2, None, None, [0], [hidden], [hidden], "bf16")
  wk_v = l2_view(wk_l2, task, 0, [0, 0, 0], [1, hidden, hd], [1, hidden, hd], "bf16")
  wv_v = l2_view(wv_l2, task, 0, [0, 0, 0], [1, hidden, hd], [1, hidden, hd], "bf16")
  kn_v = l2_view(kn_l2, task, 0, [0, 0, 0], [1, 1, hd], [1, 1, hd], "bf16")
  vn_v = l2_view(vn_l2, task, 0, [0, 0, 0], [1, 1, hd], [1, 1, hd], "bf16")
  body += [h_v, wk_v, wv_v, kn_v, vn_v]

  h_t = TileAllocOp([hidden], "bf16", alignment=256)
  wk = TileAllocOp([hidden, hd], "bf16", alignment=256)
  wv = TileAllocOp([hidden, hd], "bf16", alignment=256)
  kn = TileAllocOp([1, hd], "bf16", alignment=256)
  vn = TileAllocOp([1, hd], "bf16", alignment=256)
  body += [h_t, wk, wv, kn, vn]

  loads = [
    TileLoadOp(h_v.result, h_t.result, "h_loaded"),
    TileLoadOp(wk_v.result, wk.result, "wk_loaded"),
    TileLoadOp(wv_v.result, wv.result, "wv_loaded"),
  ]
  body += loads
  body.append(TileAwaitOp(loads))
  body.append(TileSignalOp("input_released", task))
  bk = TileBoaOp("matmul", 1, hd, hidden, 2 * hd * hidden, "k_new_boa")
  body += [bk, TileAwaitOp([bk])]
  ks = TileStoreOp(kn.result, kn_v.result, "k_new_stored")
  bv = TileBoaOp("matmul", 1, hd, hidden, 2 * hd * hidden, "v_new_boa")
  body += [bv, TileAwaitOp([bv])]
  vs = TileStoreOp(vn.result, vn_v.result, "v_new_stored")
  body += [ks, TileAwaitOp([ks]), vs, TileAwaitOp([vs])]
  body.append(TileSignalOp("output_ready", task))
  body += [TileFreeOp(b) for b in (h_t, wk, wv, kn, vn)]
  body.append(TileReturnOp())
  prog.body.block.add_ops(body)
  return prog


def make_decode_kv(cfg: DecodeConfig, *, hw: HardwareConfig | None = None) -> ModuleOp:
  hw = hw or HardwareConfig()
  kb, hd = cfg.kv_block, cfg.head_dim
  heads = cfg.q_heads // cfg.kv_heads
  kh, blocks = cfg.kv_heads, cfg.blocks
  pipelined = cfg.pipelined

  prog = _decode_block_program(cfg, hw)

  l1_shape_state = [kh, heads, 66]
  l2_buffers = (
    [(kh * kb * hd * 2, 256)] * (4 if pipelined else 2)
    + [(kh * heads * hd * 2, 256), (kh * heads * 66 * 4, 64), (kh * heads * hd * 4, 64)]
    + [(1024 * 2, 256)]  # h_l2 staging (current-token hidden state, 2 KiB)
    + [(kh * 1024 * hd * 2, 256)] * 2  # wk/wv staging (4x1024x64 = 512 KiB each)
    + [(kh * 1 * hd * 2, 256)] * 2  # k_new/v_new staging
  )
  l2_bytes, allowed_l2 = contract_bytes(hw, "l2", l2_buffers)
  logical_tasks = blocks * kh + kh

  tile_prog = prog
  append_prog = _append_program(cfg, hw)
  contexts = []
  submit_actuals = []
  for r in range(cfg.num_requests):
    # num_requests > 1: every request (including 0) carries a letter suffix.
    # xDSL's printer strips trailing-numeric SSA suffixes (K_CACHE_0 prints
    # as %K_CACHE and later duplicates get renumbered), which would make the
    # parsed entry-input names unstable; letters avoid that entirely.
    suffix = f"_r{chr(ord('a') + r)}" if cfg.num_requests > 1 else ""
    ctx = NestContextOp(
      f"decode_req{suffix}",
      ContextResources(
        l2_mode=0,
        allowed_profiles=allowed_l2,
        logical_tasks=logical_tasks,
        l2_spm_bytes=l2_bytes,
        requested_contexts_per_tile=1,
      ),
      placement=PLACEMENT,
      arg_types=[
        NestGlobalMemref.of([blocks, kh, kb, hd], "bf16"),  # K_CACHE
        NestGlobalMemref.of([blocks, kh, kb, hd], "bf16"),  # V_CACHE
        NestGlobalMemref.of([kh, heads, hd], "bf16"),  # Q_IN
        NestGlobalMemref.of([kh, heads, 66], "f32"),  # S_INIT
        NestGlobalMemref.of([kh, heads, hd], "f32"),  # OUT
        NestGlobalMemref.of([1024], "bf16"),  # H_T (current token hidden state)
        NestGlobalMemref.of([kh, 1024, hd], "bf16"),  # WK_A (per-tile K projection)
        NestGlobalMemref.of([kh, 1024, hd], "bf16"),  # WV_A (per-tile V projection)
        NestGlobalMemref.of([kh, 1, hd], "bf16"),  # K_APPEND (fixed position)
        NestGlobalMemref.of([kh, 1, hd], "bf16"),  # V_APPEND (fixed position)
      ],
      arg_names=[
        f"K_CACHE{suffix}",
        f"V_CACHE{suffix}",
        f"Q_IN{suffix}",
        f"S_INIT{suffix}",
        f"OUT{suffix}",
        f"H_T{suffix}",
        f"WK_A{suffix}",
        f"WV_A{suffix}",
        f"K_APPEND{suffix}",
        f"V_APPEND{suffix}",
      ],
    )
    K_CACHE, V_CACHE, Q_IN, S_INIT, OUT, H_T, WK_A, WV_A, K_APPEND, V_APPEND = ctx.body.block.args
    block = ctx.body.block

    if pipelined:
      k_buf = [NestAllocOp(f"k_p{p}{suffix}", "in", [kh, kb, hd], "bf16", alignment=256) for p in range(2)]
      v_buf = [NestAllocOp(f"v_p{p}{suffix}", "in", [kh, kb, hd], "bf16", alignment=256) for p in range(2)]
    else:
      k_buf = [NestAllocOp(f"k_l2{suffix}", "in", [kh, kb, hd], "bf16", alignment=256)]
      v_buf = [NestAllocOp(f"v_l2{suffix}", "in", [kh, kb, hd], "bf16", alignment=256)]
    q_l2 = NestAllocOp(f"q_l2{suffix}", "in", [kh, heads, hd], "bf16", alignment=256)
    state_l2 = NestAllocOp(
      f"state_l2{suffix}", "inout", l1_shape_state, "f32", sharing="context-local", alignment=64
    )
    out_l2 = NestAllocOp(f"out_l2{suffix}", "out", [kh, heads, hd], "f32", alignment=64)
    block.add_ops([*k_buf, *v_buf, q_l2, state_l2, out_l2])

    def k_view(b, _cache=K_CACHE):
      return NestSubviewOp(
        _cache,
        [b, 0, 0, 0],
        [1, kh, kb, hd],
        [1, 1, 1, 1],
        NestGlobalView.of([1, kh, kb, hd], "bf16"),
      )

    def v_view(b, _cache=V_CACHE):
      return NestSubviewOp(
        _cache,
        [b, 0, 0, 0],
        [1, kh, kb, hd],
        [1, 1, 1, 1],
        NestGlobalView.of([1, kh, kb, hd], "bf16"),
      )

    q_view = NestSubviewOp(
      Q_IN, [0, 0, 0], [kh, heads, hd], [1, 1, 1], NestGlobalView.of([kh, heads, hd], "bf16")
    )
    s_view = NestSubviewOp(
      S_INIT, [0, 0, 0], [kh, heads, 66], [1, 1, 1], NestGlobalView.of([kh, heads, 66], "f32")
    )
    out_view = NestSubviewOp(
      OUT, [0, 0, 0], [kh, heads, hd], [1, 1, 1], NestGlobalView.of([kh, heads, hd], "f32")
    )
    block.add_ops([q_view, s_view, out_view])
    tasks = NestTaskRangeOp(0, kh)
    block.add_op(tasks)

    pre_q = NestPrefetchOp(q_view, q_l2, f"pre_q{suffix}")
    pre_s = NestPrefetchOp(s_view, state_l2, f"pre_state{suffix}")
    block.add_ops([pre_q, pre_s])

    pre_k: dict[int, NestPrefetchOp] = {}
    pre_v: dict[int, NestPrefetchOp] = {}
    grids: dict[int, OpResult[NestEvent]] = {}
    inrels: dict[int, OpResult[NestEvent]] = {}
    outs: dict[int, OpResult[NestEvent]] = {}
    for b in range(blocks):
      p = b % 2 if pipelined else 0
      gate = (inrels[b - 2],) if pipelined and b >= 2 else ()
      kv = k_view(b)
      vv = v_view(b)
      block.add_ops([kv, vv])
      pre_k[b] = NestPrefetchOp(kv, k_buf[p], f"pre_k_{b}{suffix}", depends_on=gate)
      pre_v[b] = NestPrefetchOp(vv, v_buf[p], f"pre_v_{b}{suffix}", depends_on=gate)
      block.add_ops([pre_k[b], pre_v[b]])
      if not pipelined:
        block.add_op(NestAwaitOp([pre_k[b], pre_v[b]]))

      depends: list = [pre_k[b], pre_v[b], pre_q, pre_s]
      if b > 0:
        depends.append(outs[b - 1])
      d = NestDispatchOp(
        "decode_attention_block",
        tasks,
        [],
        [q_l2, k_buf[p], v_buf[p], state_l2],
        [state_l2, out_l2],
        f"d{b}_grid{suffix}",
        f"d{b}_inrel{suffix}",
        f"d{b}_out{suffix}",
        l1_mode=0,
        bindings=[q_l2, k_buf[p], v_buf[p], state_l2, out_l2],
        signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
        depends_on=depends,
      )
      grids[b] = d.grid_done
      inrels[b] = d.input_released
      outs[b] = d.output_ready
      block.add_op(d)
      if not pipelined:
        block.add_op(NestAwaitOp([grids[b]]))

    # out_l2 is written by every block dispatch, so the single HBM store
    # must depend on every writer's output_ready (IR_SPEC 3.5).
    store = NestDMAStoreOp(
      out_l2, out_view, f"out_store{suffix}", depends_on=[outs[b] for b in range(blocks)]
    )
    block.add_op(store)
    if not pipelined:
      block.add_op(NestAwaitOp([store]))

    # ---- fixed-position KV append (position = seq_len, static) ----------
    h_l2 = NestAllocOp(f"h_l2{suffix}", "in", [1024], "bf16", alignment=256)
    wk_l2 = NestAllocOp(f"wk_l2{suffix}", "in", [kh, 1024, hd], "bf16", alignment=256)
    wv_l2 = NestAllocOp(f"wv_l2{suffix}", "in", [kh, 1024, hd], "bf16", alignment=256)
    kn_l2 = NestAllocOp(f"kn_l2{suffix}", "out", [kh, 1, hd], "bf16", alignment=256)
    vn_l2 = NestAllocOp(f"vn_l2{suffix}", "out", [kh, 1, hd], "bf16", alignment=256)
    block.add_ops([h_l2, wk_l2, wv_l2, kn_l2, vn_l2])

    def small_view(g, shape, dtype):
      return NestSubviewOp(g, [0] * len(shape), shape, [1] * len(shape), NestGlobalView.of(shape, dtype))

    qt_g = small_view(H_T, [1024], "bf16")
    wk_g = small_view(WK_A, [kh, 1024, hd], "bf16")
    wv_g = small_view(WV_A, [kh, 1024, hd], "bf16")
    ka_g = small_view(K_APPEND, [kh, 1, hd], "bf16")
    va_g = small_view(V_APPEND, [kh, 1, hd], "bf16")
    block.add_ops([qt_g, wk_g, wv_g, ka_g, va_g])

    pre_qt = NestPrefetchOp(qt_g, h_l2, f"pre_qt{suffix}")
    pre_wk = NestPrefetchOp(wk_g, wk_l2, f"pre_wk{suffix}")
    pre_wv = NestPrefetchOp(wv_g, wv_l2, f"pre_wv{suffix}")
    block.add_ops([pre_qt, pre_wk, pre_wv])
    if not pipelined:
      block.add_op(NestAwaitOp([pre_qt, pre_wk, pre_wv]))

    app = NestDispatchOp(
      "kv_append_tile",
      tasks,
      [],
      [h_l2, wk_l2, wv_l2],
      [kn_l2, vn_l2],
      f"app_grid{suffix}",
      f"app_inrel{suffix}",
      f"app_out{suffix}",
      l1_mode=0,
      bindings=[h_l2, wk_l2, wv_l2, kn_l2, vn_l2],
      signal_policy={"input_released": "all_tasks", "output_ready": "all_tasks"},
      depends_on=[pre_qt, pre_wk, pre_wv],
    )
    block.add_op(app)
    if not pipelined:
      block.add_op(NestAwaitOp([app.grid_done]))

    k_store = NestDMAStoreOp(kn_l2, ka_g, f"k_append_store{suffix}", depends_on=(app.output_ready,))
    v_store = NestDMAStoreOp(vn_l2, va_g, f"v_append_store{suffix}", depends_on=(app.output_ready,))
    block.add_ops([k_store, v_store])
    if not pipelined:
      block.add_op(NestAwaitOp([k_store, v_store]))

    def release(buf, deps, _block=block) -> None:
      _block.add_op(NestReleaseOp(buf, depends_on=list(dict.fromkeys(deps))))

    all_inrels = [inrels[b] for b in range(blocks)]
    all_outs = [outs[b] for b in range(blocks)]
    if pipelined:
      for p in range(2):
        parity = list(range(p, blocks, 2))
        release(k_buf[p], [pre_k[b] for b in parity] + [inrels[b] for b in parity])
        release(v_buf[p], [pre_v[b] for b in parity] + [inrels[b] for b in parity])
    else:
      release(k_buf[0], [pre_k[b] for b in range(blocks)] + all_inrels)
      release(v_buf[0], [pre_v[b] for b in range(blocks)] + all_inrels)
    release(q_l2, [pre_q, *all_inrels])
    # context-local state: R (every reader inrel) + W (every writer out_ready) + P
    release(state_l2, [*all_inrels, *all_outs, pre_s])
    release(out_l2, [store])
    release(h_l2, [pre_qt, app.input_released])
    release(wk_l2, [pre_wk, app.input_released])
    release(wv_l2, [pre_wv, app.input_released])
    release(kn_l2, [k_store])
    release(vn_l2, [v_store])

    block.add_op(NestAwaitOp([grids[blocks - 1], store, app.grid_done, k_store, v_store]))
    block.add_op(NestReturnOp())
    contexts.append(ctx)
    submit_actuals.append(list(ctx.body.block.args))

  dev = NexusProgramOp(
    "transformer_decode_kv",
    [],
    arg_types=[a.type for args in submit_actuals for a in args],
    arg_names=[str(a.name_hint) for args in submit_actuals for a in args],
  )
  dev_args = list(dev.body.block.args)
  per = len(submit_actuals[0])
  submits = []
  for r in range(cfg.num_requests):
    # num_requests > 1: every request (including 0) carries a letter suffix.
    # xDSL's printer strips trailing-numeric SSA suffixes (K_CACHE_0 prints
    # as %K_CACHE and later duplicates get renumbered), which would make the
    # parsed entry-input names unstable; letters avoid that entirely.
    suffix = f"_r{chr(ord('a') + r)}" if cfg.num_requests > 1 else ""
    submits.append(
      NexusSubmitContextOp(
        f"decode_req{suffix}",
        f"decode_done{suffix}",
        actuals=dev_args[r * per : (r + 1) * per],
      )
    )
  dev.body.block.add_ops(submits)
  dev.body.block.add_op(NexusAwaitOp([s.result for s in submits]))
  dev.body.block.add_op(NexusReturnOp())

  return ModuleOp([tile_prog, append_prog, *contexts, dev])


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate the Transformer decode KV workload")
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("examples/workloads/transformer_decode_kv_pipeline.mlir"),
  )
  parser.add_argument(
    "--baseline-output",
    type=Path,
    default=Path("examples/workloads/transformer_decode_kv_baseline.mlir"),
  )
  parser.add_argument("--seq-len", type=int, default=2048)
  parser.add_argument("--kv-block", type=int, default=256)
  parser.add_argument("--num-requests", type=int, default=1)
  args = parser.parse_args()

  header = [
    "Transformer decode step + KV cache block pipeline (GQA 16:4, BF16).",
    "",
    "Shapes: valid_sequence=2048, KV_BLOCK=256 -> 8 KV blocks; head_dim=64;",
    "K_CACHE/V_CACHE [kv_block][tile][token][dim] block-packed so each",
    "HBM->L2 block fetch is one contiguous transfer; minimum useful KV",
    "traffic per scan = 2 MiB (K 1 MiB + V 1 MiB).",
    "Tile t (placement 15) owns KV head t and Q heads 4t..4t+3; one",
    "dispatch processes exactly one KV block (task -> KV head).",
    "",
    "This example represents one decoder iteration at",
    "valid_sequence_length = 2048.  The K/V append for the current token",
    "is written to the static block-packed append location",
    "K_APPEND/V_APPEND[tile][1][dim] (position 2048) by a timing-only",
    "kv_append_tile dispatch (QKV projection BOA, no numerics).",
    "Future device-side loop support may turn the append offset into a",
    "runtime loop-carried value; the current IR has no dynamic addresses",
    "and no token-generation loop.",
    "",
    "TIMING MODEL ONLY: tile.boa.async / tile.evu.async carry no tensor",
    "operands and execute no numerics.  The workload measures KV-block",
    "initiation interval, buffer lifetimes and dependency stalls -- NOT",
    "numerical correctness.",
    "",
    "Pipelined variant: L2 ping/pong buffer sets; prefetch block b+2 gates",
    "on block b input_released (tile has copied the block into L1), while",
    "dispatch b+1 also waits block b output_ready (online softmax state is",
    "loop-carried through the context-local L2 state buffer).  Baseline",
    "variant: prefetch -> await -> dispatch -> await per block, single",
    "buffer set, no overlap.",
    "",
    "Run: bash examples/run.sh transformer-decode-kv",
  ]

  def cfg_for(pipelined: bool) -> DecodeConfig:
    return DecodeConfig(
      seq_len=args.seq_len,
      kv_block=args.kv_block,
      num_requests=args.num_requests,
      pipelined=pipelined,
    )

  write_workload(args.output, header, make_decode_kv(cfg_for(True)))
  write_workload(args.baseline_output, header, make_decode_kv(cfg_for(False)))
  print(f"wrote {args.output}")
  print(f"wrote {args.baseline_output}")


if __name__ == "__main__":
  main()
