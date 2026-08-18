"""Tile Program builders: direct xDSL construction (no legacy ir import).

This fragment contains the helper section plus the 8 public tile-program
builders for ``pipeline_validator/workload_builders.py``.  Every builder
constructs ``TileProgramOp`` directly from ``pipeline_validator.dialects.
elenor`` xDSL operations; there is no import of ``pipeline_validator.ir``
and no legacy conversion path.

Semantics (names, comments, descriptor params, action/instruction order,
stream masks, event names, default args, unrolled control flow) are
preserved exactly from the legacy ``ir.py`` builders.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from pipeline_validator.dialects.elenor import (
  BOADescriptorOp,
  BranchEosOp,
  BranchOp,
  EngineDescriptorLike,
  EVUDescriptorOp,
  LabelOp,
  LaunchBOAOp,
  LaunchEVUOp,
  LaunchMFEOp,
  LaunchUSEOp,
  MFEDescriptorOp,
  ReturnOp,
  StreamAcquireOp,
  StreamEosOp,
  StreamPopOp,
  StreamPushOp,
  StreamReleaseOp,
  TileInstructionLike,
  TileProgramOp,
  USEDescriptorOp,
  WaitAllOp,
  WaitOp,
  _decode_params,
)

# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _label(name: str) -> LabelOp:
  """Construct a zero-cycle ``LabelOp`` (inserted before the labeled
  real instruction, exactly as the legacy ``_l`` helper attached a
  label to the following ``TileInst``)."""
  return LabelOp(name=name)


def _clone_descriptor(desc: EngineDescriptorLike) -> EngineDescriptorLike:
  """Return a shallow clone of a descriptor op preserving name, op_name,
  params and comment.  Used so that ``make_stream_pipeline_tile_program``
  can re-register caller-supplied descriptors without aliasing the
  caller's op objects."""
  name = desc.descriptor_name.data
  op_name = desc.op_name.data
  params = _decode_params(desc.params)  # type: ignore[attr-defined]
  comment = None
  if desc.comment is not None:
    comment = desc.comment.data  # type: ignore[attr-defined]
  if isinstance(desc, BOADescriptorOp):
    return BOADescriptorOp(name=name, op_name=op_name, params=params, comment=comment)
  if isinstance(desc, EVUDescriptorOp):
    return EVUDescriptorOp(name=name, op_name=op_name, params=params, comment=comment)
  if isinstance(desc, MFEDescriptorOp):
    return MFEDescriptorOp(name=name, op_name=op_name, params=params, comment=comment)
  if isinstance(desc, USEDescriptorOp):
    return USEDescriptorOp(name=name, op_name=op_name, params=params, comment=comment)
  raise TypeError(f"unknown descriptor type: {type(desc).__name__}")


def _dedup_descriptors(descs: Sequence[EngineDescriptorLike]) -> list[EngineDescriptorLike]:
  """Register descriptors by name preserving first-seen order, matching
  the legacy ``p.descriptors[name] = d`` dict-insert semantics (later
  duplicates overwrite the value but keep the original key position).
  Each descriptor is cloned so the resulting ops are not aliased to the
  caller's objects and can be safely placed into a fresh block."""
  out: OrderedDict[str, EngineDescriptorLike] = OrderedDict()
  for d in descs:
    out[d.descriptor_name.data] = _clone_descriptor(d)
  return list(out.values())


def _launch_for_descriptor(
  desc: EngineDescriptorLike, event: str, comment: str | None = None
) -> TileInstructionLike:
  """Dispatch to the correct ``Launch*Op`` based on descriptor engine
  class, mirroring the legacy ``{"BOA": TileOp.LAUNCH_BOA, ...}[kind]``
  lookup."""
  if isinstance(desc, BOADescriptorOp):
    return LaunchBOAOp(descriptor=desc.descriptor_name.data, event=event, comment=comment)
  if isinstance(desc, EVUDescriptorOp):
    return LaunchEVUOp(descriptor=desc.descriptor_name.data, event=event, comment=comment)
  if isinstance(desc, MFEDescriptorOp):
    return LaunchMFEOp(descriptor=desc.descriptor_name.data, event=event, comment=comment)
  if isinstance(desc, USEDescriptorOp):
    return LaunchUSEOp(descriptor=desc.descriptor_name.data, event=event, comment=comment)
  raise TypeError(f"unknown descriptor type: {type(desc).__name__}")


# ---------------------------------------------------------------------------
# Tile Program builders
# ---------------------------------------------------------------------------


def make_matmul_tile_program() -> TileProgramOp:
  """Single-tile matmul: load A/B, BOA matmul, store C.

  Mirrors the Tile-SPMD IR example (Architecture 17.4).
  """
  descriptors: list[EngineDescriptorLike] = [
    MFEDescriptorOp(name="load_A", op_name="load", params={"bytes": 128 * 256 * 2, "ops": 0}),
    MFEDescriptorOp(name="load_B", op_name="load", params={"bytes": 256 * 128 * 2, "ops": 0}),
    BOADescriptorOp(
      name="matmul", op_name="matmul", params={"m": 128, "n": 128, "k": 256, "ops": 2 * 128 * 128 * 256}
    ),
    MFEDescriptorOp(name="store_C", op_name="store", params={"bytes": 128 * 128 * 2, "ops": 0}),
  ]
  instructions: list[TileInstructionLike] = [
    LaunchMFEOp(descriptor="load_A", event="e0", comment="load A L2->L1"),
    LaunchMFEOp(descriptor="load_B", event="e1", comment="load B L2->L1"),
    WaitAllOp(events=["e0", "e1"]),
    LaunchBOAOp(descriptor="matmul", event="e2"),
    WaitOp(event="e2"),
    LaunchMFEOp(descriptor="store_C", event="e3"),
    WaitOp(event="e3"),
    ReturnOp(),
  ]
  return TileProgramOp(name="matmul_tile", descriptors=descriptors, instructions=instructions)


def make_tiled_matmul_tile_program(
  num_k_chunks: int = 4, tile_m: int = 128, tile_n: int = 128, tile_k: int = 64
) -> TileProgramOp:
  """Multi-level tiled matmul with K-dimension chunking and double-buffer.

  Models the classic two-level tiling pattern:
    - Outer tile (MxN) is fixed; K dimension is split into `num_k_chunks`
      chunks of size `tile_k`.
    - Each K chunk does: MFE load A_k + B_k  ->  BOA accumulate  ->  MFE store C_k.

  Both input and output are double-buffered and pipelined:
    - *Input* double-buffer: the MFE load for chunk i+1 is launched *before*
      the BOA wait for chunk i, so MFE prefetch overlaps BOA compute.
    - *Output* double-buffer: the MFE store for chunk i is launched right
      after BOA_i finishes and is *not* waited on immediately.  The wait for
      store(i-1) is placed after ``launch BOA_i`` so it overlaps with BOA_i
      compute.  The last store is drained in an epilogue before ``ret``.

  Unrolled (no loop register in the UCE ISA yet); the UCE issues one
  instruction per cycle and the MFE/BOA engines run concurrently while
  the UCE waits.

  Instruction sequence (per K chunk i):
      launch.mfe  load_A_k_i  -> e_a_i        (prefetch, or from prologue)
      launch.mfe  load_B_k_i  -> e_b_i
      [if i < n-1: also launch load_A_k_(i+1), load_B_k_(i+1)]
      waitall     (e_a_i, e_b_i)               # operands ready for chunk i
      launch.boa  matmul_k_i  -> e_mm_i        # accumulate partial sum
      [if i >= 1: wait e_store(i-1)]           # drain prev store, overlaps BOA_i
      wait        e_mm_i                        # BOA chunk i done
      launch.mfe  store_C_k_i -> e_store_i      # fire-and-forget store
  Epilogue:
      wait        e_store(n-1)                  # drain last store
      ret

  The overlap windows are:
    - Input:  T_mfe(chunk_i+1) hidden behind T_boa(chunk_i)
    - Output: T_store(chunk_i-1) hidden behind T_boa(chunk_i)
  With enough chunks both MFE load and store latencies are fully hidden
  behind BOA compute, validating the Architecture 21.2 roofline:
  BOA_perf bound by compute, not memory.
  """
  name = f"tiled_matmul_{num_k_chunks}k_tile"
  k_chunk_bytes_a = tile_m * tile_k * 2  # BF16
  k_chunk_bytes_b = tile_k * tile_n * 2
  # per-chunk BOA ops: 2*M*N*K_chunk (accumulate across chunks)
  k_chunk_ops = 2 * tile_m * tile_n * tile_k
  descriptors: OrderedDict[str, EngineDescriptorLike] = OrderedDict()
  insts: list[TileInstructionLike] = []

  # ---- prologue: prefetch chunk 0 inputs ----
  insts.append(LaunchMFEOp(descriptor="load_A_k0", event="e_a0", comment="prefetch A chunk 0"))
  insts.append(LaunchMFEOp(descriptor="load_B_k0", event="e_b0", comment="prefetch B chunk 0"))
  descriptors["load_A_k0"] = MFEDescriptorOp(
    name="load_A_k0", op_name="load", params={"bytes": k_chunk_bytes_a, "ops": 0}
  )
  descriptors["load_B_k0"] = MFEDescriptorOp(
    name="load_B_k0", op_name="load", params={"bytes": k_chunk_bytes_b, "ops": 0}
  )

  # ---- per-chunk loop body (unrolled) ----
  for i in range(num_k_chunks):
    # prefetch chunk i+1 inputs (input double-buffer: overlaps BOA_i)
    if i < num_k_chunks - 1:
      ni = i + 1
      insts.append(
        LaunchMFEOp(
          descriptor=f"load_A_k{ni}", event=f"e_a{ni}", comment=f"prefetch A chunk {ni} (overlap)"
        )
      )
      insts.append(
        LaunchMFEOp(
          descriptor=f"load_B_k{ni}", event=f"e_b{ni}", comment=f"prefetch B chunk {ni} (overlap)"
        )
      )
      descriptors[f"load_A_k{ni}"] = MFEDescriptorOp(
        name=f"load_A_k{ni}", op_name="load", params={"bytes": k_chunk_bytes_a, "ops": 0}
      )
      descriptors[f"load_B_k{ni}"] = MFEDescriptorOp(
        name=f"load_B_k{ni}", op_name="load", params={"bytes": k_chunk_bytes_b, "ops": 0}
      )

    # wait for chunk i operands
    insts.append(WaitAllOp(events=[f"e_a{i}", f"e_b{i}"], comment=f"operands for chunk {i} ready"))

    # BOA accumulate chunk i
    mm_name = f"matmul_k{i}"
    descriptors[mm_name] = BOADescriptorOp(
      name=mm_name,
      op_name="matmul",
      params={"m": tile_m, "n": tile_n, "k": tile_k, "ops": k_chunk_ops, "chunk": i, "accumulate": i > 0},
    )
    insts.append(LaunchBOAOp(descriptor=mm_name, event=f"e_mm{i}", comment=f"BOA accumulate chunk {i}"))

    # drain previous store while BOA_i runs (output double-buffer)
    if i >= 1:
      insts.append(WaitOp(event=f"e_store{i - 1}", comment=f"drain store {i - 1} (overlap BOA{i})"))

    # wait for BOA_i result
    insts.append(WaitOp(event=f"e_mm{i}", comment=f"BOA chunk {i} done"))

    # store chunk i output (fire-and-forget: overlaps next BOA)
    store_name = f"store_C_k{i}"
    descriptors[store_name] = MFEDescriptorOp(
      name=store_name, op_name="store", params={"bytes": tile_m * tile_n * 2, "ops": 0, "chunk": i}
    )
    insts.append(
      LaunchMFEOp(
        descriptor=store_name, event=f"e_store{i}", comment=f"MFE store result chunk {i} (deferred wait)"
      )
    )

  # ---- epilogue: drain last store ----
  insts.append(WaitOp(event=f"e_store{num_k_chunks - 1}", comment="drain last store"))
  insts.append(ReturnOp())
  return TileProgramOp(name=name, descriptors=list(descriptors.values()), instructions=insts)


def make_tiled_matmul_persistent_tile_program(
  num_group_chunks: int = 4, num_k_chunks: int = 4, tile_m: int = 128, tile_n: int = 128, tile_k: int = 64
) -> TileProgramOp:
  """Persistent single-dispatch tiled matmul with cross-chunk L2->L1 overlap.

  Unlike ``make_tiled_matmul_tile_program`` (which is re-loaded per group
  chunk via ``DISPATCH_ROLE``), this program runs **all group chunks** in
  one Tile Program instance.  The key benefit: chunk g+1's prologue L2->L1
  MFE load is issued *before* chunk g's last BOA ``WAIT``, so the load
  overlaps with chunk g's BOA compute -- eliminating the inter-program
  bubble that exists when each chunk is a separate ``DISPATCH_ROLE``.

  The program ``WAIT``s on sequencer-issued DMA events (``ev_dma_A{g}`` /
  ``ev_dma_B{g}``) via the cross-level event bridge: the TileGroup
  forwards group DMA completions to active tiles, and seeds already-
  completed events at ``dispatch_role`` time so chunk-0 waits resolve
  immediately.

  Per group chunk g, the K-chunk inner loop is identical to
  ``make_tiled_matmul_tile_program``: input double-buffer (MFE load
  k+1 before BOA k) + output double-buffer (store k deferred, drained
  during BOA k+1).

  Instruction sequence (simplified, per group chunk g):
      [if g < N-1: WAIT ev_dma_A{g+1}, ev_dma_B{g+1}   # cross-chunk overlap
                   launch.mfe load_A_k0_g{g+1}, load_B_k0_g{g+1}]
      WAIT ev_dma_A{g}, ev_dma_B{g}                     # L2 data ready (chunk g)
      launch.mfe load_A_k0_g{g}, load_B_k0_g{g}         # prologue load chunk g
      [K-chunk inner loop: same as make_tiled_matmul_tile_program]
      [epilogue: drain last store of chunk g]
  Final epilogue: drain last store, RET.
  """
  name = f"tiled_matmul_persistent_{num_group_chunks}g_{num_k_chunks}k"
  k_chunk_bytes_a = tile_m * tile_k * 2  # BF16
  k_chunk_bytes_b = tile_k * tile_n * 2
  k_chunk_ops = 2 * tile_m * tile_n * tile_k
  descriptors: OrderedDict[str, EngineDescriptorLike] = OrderedDict()
  insts: list[TileInstructionLike] = []

  # Helper: register a load descriptor for chunk (g, k)
  def _load_desc(dname: str, nbytes: int) -> None:
    if dname not in descriptors:
      descriptors[dname] = MFEDescriptorOp(name=dname, op_name="load", params={"bytes": nbytes, "ops": 0})

  def _store_desc(dname: str, nbytes: int, chunk: int) -> None:
    if dname not in descriptors:
      descriptors[dname] = MFEDescriptorOp(
        name=dname, op_name="store", params={"bytes": nbytes, "ops": 0, "chunk": chunk}
      )

  def _boa_desc(dname: str, chunk: int, accumulate: bool) -> None:
    if dname not in descriptors:
      descriptors[dname] = BOADescriptorOp(
        name=dname,
        op_name="matmul",
        params={
          "m": tile_m,
          "n": tile_n,
          "k": tile_k,
          "ops": k_chunk_ops,
          "chunk": chunk,
          "accumulate": accumulate,
        },
      )

  for g in range(num_group_chunks):
    # ---- wait for current chunk's L2 data (bridged DMA event) ----
    insts.append(WaitOp(event=f"ev_dma_A{g}", comment=f"L2 A{g} ready (group DMA)"))
    insts.append(WaitOp(event=f"ev_dma_B{g}", comment=f"L2 B{g} ready (group DMA)"))

    # ---- prologue: load chunk g's K-chunk 0 ----
    # For g == 0 the prologue is launched here.  For g > 0 it was
    # already launched by the cross-chunk overlap block of chunk
    # g-1 (hidden behind g-1's last BOA compute).
    if g == 0:
      _load_desc(f"load_A_k0_g{g}", k_chunk_bytes_a)
      _load_desc(f"load_B_k0_g{g}", k_chunk_bytes_b)
      insts.append(
        LaunchMFEOp(descriptor=f"load_A_k0_g{g}", event=f"e_a0_g{g}", comment=f"prefetch A k0 g{g}")
      )
      insts.append(
        LaunchMFEOp(descriptor=f"load_B_k0_g{g}", event=f"e_b0_g{g}", comment=f"prefetch B k0 g{g}")
      )

    # ---- K-chunk inner loop ----
    for i in range(num_k_chunks):
      # prefetch K-chunk i+1 inputs (input double-buffer)
      if i < num_k_chunks - 1:
        ni = i + 1
        _load_desc(f"load_A_k{ni}_g{g}", k_chunk_bytes_a)
        _load_desc(f"load_B_k{ni}_g{g}", k_chunk_bytes_b)
        insts.append(
          LaunchMFEOp(
            descriptor=f"load_A_k{ni}_g{g}",
            event=f"e_a{ni}_g{g}",
            comment=f"prefetch A k{ni} g{g} (overlap)",
          )
        )
        insts.append(
          LaunchMFEOp(
            descriptor=f"load_B_k{ni}_g{g}",
            event=f"e_b{ni}_g{g}",
            comment=f"prefetch B k{ni} g{g} (overlap)",
          )
        )

      # wait for chunk i operands
      insts.append(WaitAllOp(events=[f"e_a{i}_g{g}", f"e_b{i}_g{g}"], comment=f"operands k{i} g{g} ready"))

      # BOA accumulate chunk i
      mm_name = f"matmul_k{i}_g{g}"
      _boa_desc(mm_name, i, accumulate=(i > 0))
      insts.append(
        LaunchBOAOp(descriptor=mm_name, event=f"e_mm{i}_g{g}", comment=f"BOA accumulate k{i} g{g}")
      )

      # ---- cross-chunk overlap: on the LAST K-chunk, issue the
      #      next group chunk's prologue load *before* waiting for
      #      this BOA, so the L2->L1 load overlaps with BOA compute.
      #      The WAIT on the bridged DMA event is also deferred to
      #      here -- if the DMA isn't done yet, the WAIT stalls
      #      behind BOA (best case: DMA already finished = no
      #      stall).  This is the key overlap that eliminates the
      #      inter-program bubble.
      if i == num_k_chunks - 1 and g < num_group_chunks - 1:
        ng = g + 1
        insts.append(
          WaitOp(event=f"ev_dma_A{ng}", comment=(f"cross-chunk: L2 A{ng} ready (overlap last BOA g={g})"))
        )
        insts.append(
          WaitOp(event=f"ev_dma_B{ng}", comment=(f"cross-chunk: L2 B{ng} ready (overlap last BOA g={g})"))
        )
        _load_desc(f"load_A_k0_g{ng}", k_chunk_bytes_a)
        _load_desc(f"load_B_k0_g{ng}", k_chunk_bytes_b)
        insts.append(
          LaunchMFEOp(
            descriptor=f"load_A_k0_g{ng}",
            event=f"e_a0_g{ng}",
            comment=(f"cross-chunk: prefetch A k0 g{ng} (overlap last BOA g={g})"),
          )
        )
        insts.append(
          LaunchMFEOp(
            descriptor=f"load_B_k0_g{ng}",
            event=f"e_b0_g{ng}",
            comment=(f"cross-chunk: prefetch B k0 g{ng} (overlap last BOA g={g})"),
          )
        )

      # drain previous store while BOA_i runs (output double-buffer)
      if i >= 1:
        insts.append(
          WaitOp(event=f"e_store{i - 1}_g{g}", comment=f"drain store k{i - 1} g{g} (overlap BOA{i})")
        )

      # wait for BOA_i result
      insts.append(WaitOp(event=f"e_mm{i}_g{g}", comment=f"BOA k{i} g{g} done"))

      # store chunk i output (fire-and-forget)
      store_name = f"store_C_k{i}_g{g}"
      _store_desc(store_name, tile_m * tile_n * 2, i)
      insts.append(
        LaunchMFEOp(
          descriptor=store_name, event=f"e_store{i}_g{g}", comment=f"MFE store k{i} g{g} (deferred wait)"
        )
      )

    # ---- drain last store of this group chunk ----
    insts.append(WaitOp(event=f"e_store{num_k_chunks - 1}_g{g}", comment=f"drain last store g{g}"))

  # ---- final epilogue ----
  insts.append(ReturnOp())
  return TileProgramOp(name=name, descriptors=list(descriptors.values()), instructions=insts)


def make_conv_relu_tile_program() -> TileProgramOp:
  """Fused regular 3x3 Conv + ReLU tile.

  The raw input tile and lowered weight matrix are first loaded into L1.
  MFE then generates the im2col/window stream consumed as BOA operand A,
  BOA executes the lowered matmul, and EVU applies the relu epilogue.
  BOA owns only the dense GEMM; window generation stays in MFE.
  """
  gemm_m = 128  # output positions in this validator tile
  gemm_n = 128  # output channels
  in_channels = 128
  kernel_h = 3
  kernel_w = 3
  dtype_bytes = 2  # BF16
  gemm_k = in_channels * kernel_h * kernel_w
  input_bytes = gemm_m * in_channels * dtype_bytes
  im2col_bytes = gemm_m * gemm_k * dtype_bytes
  weight_bytes = gemm_k * gemm_n * dtype_bytes
  output_bytes = gemm_m * gemm_n * dtype_bytes
  descriptors: list[EngineDescriptorLike] = [
    MFEDescriptorOp(name="load_input", op_name="load", params={"bytes": input_bytes, "ops": 0}),
    MFEDescriptorOp(name="load_weight", op_name="load", params={"bytes": weight_bytes, "ops": 0}),
    MFEDescriptorOp(
      name="im2col_window",
      op_name="im2col",
      params={
        "m": gemm_m,
        "k": gemm_k,
        "ic": in_channels,
        "kh": kernel_h,
        "kw": kernel_w,
        "bytes": im2col_bytes,
        "ops": 0,
      },
    ),
    BOADescriptorOp(
      name="conv_gemm",
      op_name="matmul",
      params={"m": gemm_m, "n": gemm_n, "k": gemm_k, "ops": 2 * gemm_m * gemm_n * gemm_k},
    ),
    EVUDescriptorOp(name="relu", op_name="relu", params={"bytes": output_bytes, "ops": gemm_m * gemm_n}),
    MFEDescriptorOp(name="store_output", op_name="store", params={"bytes": output_bytes, "ops": 0}),
  ]
  instructions: list[TileInstructionLike] = [
    LaunchMFEOp(descriptor="load_input", event="e0", comment="load input source tile L2->L1"),
    LaunchMFEOp(descriptor="load_weight", event="e1", comment="load lowered conv weights L2->L1"),
    WaitOp(event="e0"),
    LaunchMFEOp(descriptor="im2col_window", event="e2", comment="MFE im2col/window stream for BOA A"),
    WaitAllOp(events=["e1", "e2"]),
    LaunchBOAOp(descriptor="conv_gemm", event="e3", comment="BOA GEMM over MFE-generated im2col A"),
    WaitOp(event="e3"),
    LaunchEVUOp(descriptor="relu", event="e4", comment="EVU relu epilogue"),
    WaitOp(event="e4"),
    LaunchMFEOp(descriptor="store_output", event="e5", comment="store output L1->L2"),
    WaitOp(event="e5"),
    ReturnOp(),
  ]
  return TileProgramOp(name="conv_relu_tile", descriptors=descriptors, instructions=instructions)


def make_pow_tile_program(
  name: str = "pow_tile", chunk_bytes: int = 32768, exponent: int = 2, pow_ops: int = 65536
) -> TileProgramOp:
  """EVU elementwise pow on one tile chunk.

  Mirrors the ``pow_4k_tile`` program from the tiled-matmul-pipelined-pow
  IR example: load input L2->L1 (MFE), EVU pow, store output L1->L2 (MFE).
  ``pow_ops`` defaults to the value used in the IR example (65536 for
  exponent=2 on a 128x128 BF16 chunk) and feeds the EVU latency model
  ``launch + ceil(ops / (lanes*2))``.
  """
  descriptors: list[EngineDescriptorLike] = [
    MFEDescriptorOp(name="load_pow", op_name="load", params={"bytes": chunk_bytes, "ops": 0}),
    EVUDescriptorOp(
      name="pow_chunk", op_name="pow", params={"bytes": chunk_bytes, "exponent": exponent, "ops": pow_ops}
    ),
    MFEDescriptorOp(name="store_pow", op_name="store", params={"bytes": chunk_bytes, "ops": 0}),
  ]
  instructions: list[TileInstructionLike] = [
    LaunchMFEOp(descriptor="load_pow", event="e_load", comment="load pow input L2->L1"),
    WaitOp(event="e_load", comment="pow input ready"),
    LaunchEVUOp(descriptor="pow_chunk", event="e_pow", comment="EVU pow on one tile chunk"),
    WaitOp(event="e_pow", comment="pow chunk done"),
    LaunchMFEOp(descriptor="store_pow", event="e_store", comment="store pow output L1->L2"),
    WaitOp(event="e_store", comment="drain pow output"),
    ReturnOp(),
  ]
  return TileProgramOp(name=name, descriptors=descriptors, instructions=instructions)


def make_paged_attention_tile_program() -> TileProgramOp:
  """Full paged-attention tile program (Architecture 20.2 example).

  Mirrors the exact Tile Program from the spec:
      launch.mfe desc_gather_K_pages -> e0
      launch.mfe desc_gather_V_pages -> e1
      waitall e0 | e1
      launch.boa desc_qk_matmul -> e2
      wait e2
      launch.evu desc_scale_mask -> e3
      wait e3
      launch.evu desc_softmax -> e4
      wait e4
      launch.boa desc_pv_matmul -> e5
      wait e5
      launch.mfe desc_store_output -> e6
      wait e6
      ret

  MFE does page-table walk + KV page prefetch/reorder (Page Stream, MFE
  design 3.3).  BOA does QK and PV.  EVU does scale/mask then softmax.
  This is a single-tile fused pipeline -- the UCE serializes the engines,
  so no Stream Queue is needed; MFE prefetch overlap with BOA compute is
  governed by the T_prefetch <= T_qk condition (Architecture 21.3).
  """
  # Canonical paged-attention block: q_len=128, head_dim=64, page_size=16,
  # num_pages=8 (seq_len=128).  K/V pages are gathered by MFE Page Stream.
  kv_page_bytes = 16 * 64 * 2  # one KV page: 16 tokens x 64 head_dim
  kv_total_bytes = 8 * kv_page_bytes  # 8 pages gathered
  score_bytes = 128 * 128 * 2  # QK score: 128 x 128 (q x pages*page_size)
  out_bytes = 128 * 64 * 2  # AV output: 128 x 64
  descriptors: list[EngineDescriptorLike] = [
    MFEDescriptorOp(
      name="gather_K_pages",
      op_name="page_stream",
      params={"bytes": kv_total_bytes, "ops": 0, "mode": "page_stream", "num_pages": 8, "page_size": 16},
    ),
    MFEDescriptorOp(
      name="gather_V_pages",
      op_name="page_stream",
      params={"bytes": kv_total_bytes, "ops": 0, "mode": "page_stream", "num_pages": 8, "page_size": 16},
    ),
    BOADescriptorOp(
      name="qk_matmul", op_name="matmul", params={"m": 128, "n": 128, "k": 64, "ops": 2 * 128 * 128 * 64}
    ),
    EVUDescriptorOp(
      name="scale_mask", op_name="scale_mask", params={"bytes": score_bytes, "ops": 128 * 128 * 2}
    ),
    EVUDescriptorOp(name="softmax", op_name="softmax", params={"bytes": score_bytes, "ops": 128 * 128 * 8}),
    BOADescriptorOp(
      name="pv_matmul", op_name="matmul", params={"m": 128, "n": 64, "k": 128, "ops": 2 * 128 * 64 * 128}
    ),
    MFEDescriptorOp(name="store_output", op_name="store", params={"bytes": out_bytes, "ops": 0}),
  ]
  instructions: list[TileInstructionLike] = [
    LaunchMFEOp(descriptor="gather_K_pages", event="e0", comment="MFE page-stream gather K pages"),
    LaunchMFEOp(descriptor="gather_V_pages", event="e1", comment="MFE page-stream gather V pages"),
    WaitAllOp(events=["e0", "e1"]),
    LaunchBOAOp(descriptor="qk_matmul", event="e2", comment="BOA QK^T matmul"),
    WaitOp(event="e2"),
    LaunchEVUOp(descriptor="scale_mask", event="e3", comment="EVU scale + causal mask"),
    WaitOp(event="e3"),
    LaunchEVUOp(descriptor="softmax", event="e4", comment="EVU softmax over scores"),
    WaitOp(event="e4"),
    LaunchBOAOp(descriptor="pv_matmul", event="e5", comment="BOA PV matmul"),
    WaitOp(event="e5"),
    LaunchMFEOp(descriptor="store_output", event="e6", comment="MFE store attention output"),
    WaitOp(event="e6"),
    ReturnOp(),
  ]
  return TileProgramOp(name="paged_attention_tile", descriptors=descriptors, instructions=instructions)


def make_stream_pipeline_tile_program(
  in_q: int | None,
  out_q: int,
  body_descs: Sequence[EngineDescriptorLike],
  producer_id: int = 0,
  block_count: int = 1,
) -> TileProgramOp:
  """A streaming tile program (Architecture 16.4 Tile Program example).

  Two variants:
    * source tile  (in_q is None): loads its own input from HBM via MFE,
      runs body, pushes to out_q.  Loops `block_count` times then pushes EOS.
    * consumer tile (in_q is not None): pops from in_q, runs body, pushes
      to out_q (if out_q >= 0), releases in token.  Exits on EOS.

  loop:
      [pop in_token from in_q | check block counter]   -> done if EOS / count reached
      [acquire out_token on out_q]
      [DMA_LOAD / MFE load]  -> wait
      BOA_RUN / EVU_RUN      -> wait
      [DMA_STORE / MFE store] -> wait
      [push out_token, release in_token]
      br loop
  done:
      push EOS on out_q
      ret
  """
  is_source = in_q is None
  name = ("src_tile_" if is_source else "cons_tile_") + "_".join(d.descriptor_name.data for d in body_descs)
  # Register descriptors by name (dict semantics, first-seen order).
  descriptors = _dedup_descriptors(body_descs)
  insts: list[TileInstructionLike] = []

  # ---- loop head ----
  if is_source:
    # source: use a pseudo block counter via a named register.
    # We model the loop with BR after a fixed number of iterations using
    # a CMP+BR pattern simulated by a counter register the UCE tracks.
    # For simplicity: emit `block_count` unrolled iterations (no loop).
    for blk in range(block_count):
      for i, d in enumerate(body_descs):
        insts.append(
          _launch_for_descriptor(d, event=f"e{blk}_{i}", comment=f"block {blk} {d.descriptor_name.data}")
        )
        insts.append(WaitOp(event=f"e{blk}_{i}"))
      # push output token
      insts.append(StreamAcquireOp(queue_id=out_q, destination_token="out_tok", comment="acquire credit"))
      insts.append(StreamPushOp(queue_id=out_q, token_register="out_tok", producer_id=producer_id))
    # after all blocks, push EOS to signal downstream
    insts.append(StreamEosOp(queue_id=out_q, producer_id=producer_id, comment="source EOS"))
  else:
    # consumer: loop pop -> body -> push -> release, exit on EOS.
    assert in_q is not None
    insts.append(_label("loop"))
    insts.append(
      StreamPopOp(
        queue_id=in_q,
        destination_token="in_tok",
        comment="pop input",
      )
    )
    insts.append(BranchEosOp(token_register="in_tok", target="done"))
    if out_q >= 0:
      insts.append(StreamAcquireOp(queue_id=out_q, destination_token="out_tok", comment="acquire credit"))
    for i, d in enumerate(body_descs):
      insts.append(_launch_for_descriptor(d, event=f"e_body{i}"))
      insts.append(WaitOp(event=f"e_body{i}"))
    if out_q >= 0:
      insts.append(StreamPushOp(queue_id=out_q, token_register="out_tok", producer_id=producer_id))
    insts.append(StreamReleaseOp(queue_id=in_q, token_register="in_tok"))
    insts.append(BranchOp(target="loop"))
    insts.append(_label("done"))
    insts.append(StreamEosOp(queue_id=out_q, producer_id=producer_id))

  insts.append(ReturnOp())
  return TileProgramOp(name=name, descriptors=descriptors, instructions=insts)


def make_identity_tile_program() -> TileProgramOp:
  """A tile program that does nothing (for pure role dispatch testing)."""
  return TileProgramOp(name="identity_tile", descriptors=[], instructions=[ReturnOp()])


# ---------------------------------------------------------------------------
# Task builders (12): each returns a ModuleOp containing exactly one
# TileGroupTaskOp.  Tile-program builders (8) are assumed to already exist
# in this module and are called by public name.
# ---------------------------------------------------------------------------

from xdsl.dialects.builtin import ModuleOp

from pipeline_validator.dialects.elenor import (
  DispatchRoleOp,
  GroupActionLike,
  GroupDMAPrefetchOp,
  GroupDMAStoreOp,
  GroupWaitEventOp,
  InitStreamOp,
  SignalEventOp,
  StreamDescOp,
  TileGroupTaskOp,
  TileRoleBindingOp,
)


def make_pow_task(num_group_chunks: int = 4) -> ModuleOp:
  """Standalone EVU pow workload with pipelined group DMA.

  Mirrors the ``pow.ir`` task shape: one role (role 1) across all four
  tiles, four HBM->L2 prefetches issued up front, per-chunk dispatch once
  the input is visible in L2, per-chunk L2->HBM store once the tiles finish,
  then a final drain of all output DMAs.
  """
  chunk_bytes = 128 * 128 * 2
  bytes_pow = chunk_bytes * 4
  roles = [
    TileRoleBindingOp(
      role_id=1, tile_mask=0x0F, program=make_pow_tile_program(name="pow_4k_tile", chunk_bytes=chunk_bytes)
    )
  ]
  actions: list[GroupActionLike] = []

  for g in range(num_group_chunks):
    actions.append(
      GroupDMAPrefetchOp(
        descriptor=f"gdma_prefetch_pow{g}",
        l2_slot=f"l2_buf_pow{g}",
        event=f"ev_dma_pow_in{g}",
        bytes_total=bytes_pow,
        comment=f"Pow input chunk {g} HBM->L2",
      )
    )

  for g in range(num_group_chunks):
    actions.append(
      GroupWaitEventOp(event=f"ev_dma_pow_in{g}", comment=f"Pow chunk {g} input visible in L2")
    )
    actions.append(
      DispatchRoleOp(role_id=1, event=f"ev_role_pow{g}", comment=f"Launch pow tile program for chunk {g}")
    )

  for g in range(num_group_chunks):
    actions.append(GroupWaitEventOp(event=f"ev_role_pow{g}", comment=f"Pow tiles finished chunk {g}"))
    actions.append(
      GroupDMAStoreOp(
        descriptor=f"gdma_store_pow{g}",
        l2_slot=f"l2_buf_pow{g}",
        event=f"ev_dma_pow_out{g}",
        bytes_total=bytes_pow,
        comment=f"Pow output chunk {g} L2->HBM",
      )
    )

  for g in range(num_group_chunks):
    actions.append(GroupWaitEventOp(event=f"ev_dma_pow_out{g}", comment=f"Drain pow output chunk {g}"))

  actions.append(SignalEventOp(event="group_task_done"))
  task = TileGroupTaskOp(name="pow_task", streams=[], roles=roles, actions=actions)
  return ModuleOp([task])


def make_matmul_task(block_count: int = 4) -> ModuleOp:
  """Task that dispatches a single matmul role across 4 tiles.

  Role 0 (tiles 0-3) runs the matmul tile program.  No inter-tile stream;
  the task prefetches A/B weights HBM->L2 via Group DMA, dispatches the
  role, then stores C L2->HBM.  This validates Global DMA + role dispatch
  + storeback trace coverage on the TileGroup timeline.
  """
  del block_count
  # Per-tile A = 128*256*2, B = 256*128*2, C = 128*128*2 (BF16).
  # 4-tile M-split: A and C are per-tile (x4), B is shared (x1).
  bytes_a = 128 * 256 * 2 * 4
  bytes_b = 256 * 128 * 2
  bytes_c = 128 * 128 * 2 * 4
  roles = [TileRoleBindingOp(role_id=0, tile_mask=0x0F, program=make_matmul_tile_program())]
  actions: list[GroupActionLike] = [
    # Group DMA HBM -> L2 prefetch (both prefetches issued before wait
    # so they overlap, per Architecture 16.5).
    GroupDMAPrefetchOp(
      descriptor="gdma_prefetch_A",
      l2_slot="l2_buf_A",
      event="ev_dma_A",
      bytes_total=bytes_a,
      comment="Group DMA prefetch A HBM->L2",
    ),
    GroupDMAPrefetchOp(
      descriptor="gdma_prefetch_B",
      l2_slot="l2_buf_B",
      event="ev_dma_B",
      bytes_total=bytes_b,
      comment="Group DMA prefetch B HBM->L2",
    ),
    GroupWaitEventOp(event="ev_dma_A"),
    GroupWaitEventOp(event="ev_dma_B"),
    DispatchRoleOp(role_id=0, event="ev_role0"),
    GroupWaitEventOp(event="ev_role0"),
    # Group DMA L2 -> HBM storeback
    GroupDMAStoreOp(
      descriptor="gdma_store_C",
      l2_slot="l2_buf_C",
      event="ev_dma_C",
      bytes_total=bytes_c,
      comment="Group DMA store C L2->HBM",
    ),
    GroupWaitEventOp(event="ev_dma_C"),
    SignalEventOp(event="group_task_done"),
  ]
  task = TileGroupTaskOp(name="matmul_task", streams=[], roles=roles, actions=actions)
  return ModuleOp([task])


def make_tiled_matmul_task(num_k_chunks: int = 4) -> ModuleOp:
  """Tiled matmul task with K-dimension chunking across 4 tiles.

  Each tile runs the tiled matmul program that splits K into
  `num_k_chunks` chunks of size tile_k and uses double-buffered MFE
  prefetch to overlap memory and compute.  This validates the
  multi-level tiling + pipeline overlap that the single-chunk matmul
  workload cannot expose.

  The task prefetches A/B HBM->L2 via Group DMA before dispatching the
  single role, then stores C L2->HBM after.
  """
  # K = tile_k * num_k_chunks.  A = M*K*2, B = K*N*2, C = M*N*2 (BF16).
  # 4-tile M-split: A and C are per-tile (x4), B is shared (x1).
  total_k = 64 * num_k_chunks
  bytes_a = 128 * total_k * 2 * 4
  bytes_b = total_k * 128 * 2
  bytes_c = 128 * 128 * 2 * 4
  roles = [
    TileRoleBindingOp(
      role_id=0, tile_mask=0x0F, program=make_tiled_matmul_tile_program(num_k_chunks=num_k_chunks)
    )
  ]
  actions: list[GroupActionLike] = [
    GroupDMAPrefetchOp(
      descriptor="gdma_prefetch_A",
      l2_slot="l2_buf_A",
      event="ev_dma_A",
      bytes_total=bytes_a,
      comment="Group DMA prefetch A HBM->L2",
    ),
    GroupDMAPrefetchOp(
      descriptor="gdma_prefetch_B",
      l2_slot="l2_buf_B",
      event="ev_dma_B",
      bytes_total=bytes_b,
      comment="Group DMA prefetch B HBM->L2",
    ),
    GroupWaitEventOp(event="ev_dma_A"),
    GroupWaitEventOp(event="ev_dma_B"),
    DispatchRoleOp(role_id=0, event="ev_role0"),
    GroupWaitEventOp(event="ev_role0"),
    GroupDMAStoreOp(
      descriptor="gdma_store_C",
      l2_slot="l2_buf_C",
      event="ev_dma_C",
      bytes_total=bytes_c,
      comment="Group DMA store C L2->HBM",
    ),
    GroupWaitEventOp(event="ev_dma_C"),
    SignalEventOp(event="group_task_done"),
  ]
  task = TileGroupTaskOp(name="tiled_matmul_task", streams=[], roles=roles, actions=actions)
  return ModuleOp([task])


def make_tiled_matmul_pipelined_task(num_group_chunks: int = 4, num_k_chunks: int = 4) -> ModuleOp:
  """Tiled matmul task with group-level async IO pipeline.

  Unlike `make_tiled_matmul_task` (which does ONE big DMA_PREFETCH for A+B
  and ONE DMA_STORE for C at the group level), this task issues multiple
  DMA_PREFETCH -> DISPATCH_ROLE -> DMA_STORE cycles, pipelined so
  HBM<->L2 DMA for stage i+1 overlaps tile compute for stage i.

  This is a **timing proxy** for a multi-stage tiled matmul: each stage
  dispatches the same K-chunked tile program and independently stores its
  result.  Cross-stage accumulation semantics are not modelled (the
  validator exercises pipeline timing, not numerical correctness).  What
  this validates:
    - Group-level IO pipeline: DMA for g+1 hidden behind compute for g.
    - Tile-level K-chunk pipeline: MFE load for k+1 hidden behind BOA for k.
    - Both levels working simultaneously (two-level overlap).

  Group action sequence (per group chunk g):
      [prologue: DMA_PREFETCH A_0, B_0]
      WAIT A_g, B_g
      DISPATCH_ROLE 0  -> ev_role_cg
      [if g < N-1: DMA_PREFETCH A_{g+1}, B_{g+1}  (overlaps compute)]
      WAIT ev_role_cg
      DMA_STORE C_g    -> ev_dma_Cg
      [if g >= 1: WAIT ev_dma_C{g-1}  (drain prev store)]
      [epilogue: WAIT last store, SIGNAL_EVENT]
  """
  total_k = 64 * num_k_chunks
  bytes_a = 128 * total_k * 2 * 4  # tile_m * total_k * BF16 * 4 tiles
  bytes_b = total_k * 128 * 2  # total_k * tile_n * BF16 (shared)
  bytes_c = 128 * 128 * 2 * 4  # tile_m * tile_n * BF16 * 4 tiles
  roles = [
    TileRoleBindingOp(
      role_id=0, tile_mask=0x0F, program=make_tiled_matmul_tile_program(num_k_chunks=num_k_chunks)
    )
  ]
  actions: list[GroupActionLike] = []

  # ---- prologue: prefetch first chunk ----
  actions.append(
    GroupDMAPrefetchOp(
      descriptor="gdma_prefetch_A0",
      l2_slot="l2_buf_A0",
      event="ev_dma_A0",
      bytes_total=bytes_a,
      comment="Group DMA prefetch A chunk 0 HBM->L2",
    )
  )
  actions.append(
    GroupDMAPrefetchOp(
      descriptor="gdma_prefetch_B0",
      l2_slot="l2_buf_B0",
      event="ev_dma_B0",
      bytes_total=bytes_b,
      comment="Group DMA prefetch B chunk 0 HBM->L2",
    )
  )

  # ---- per-chunk loop body (unrolled) ----
  for g in range(num_group_chunks):
    # wait for A_g, B_g DMA
    actions.append(GroupWaitEventOp(event=f"ev_dma_A{g}", comment=f"Wait for A chunk {g} DMA"))
    actions.append(GroupWaitEventOp(event=f"ev_dma_B{g}", comment=f"Wait for B chunk {g} DMA"))

    # dispatch tile role
    actions.append(
      DispatchRoleOp(role_id=0, event=f"ev_role_c{g}", comment=f"Dispatch tiles for group chunk {g}")
    )

    # prefetch next chunk while tiles compute (input double-buffer)
    if g < num_group_chunks - 1:
      ng = g + 1
      actions.append(
        GroupDMAPrefetchOp(
          descriptor=f"gdma_prefetch_A{ng}",
          l2_slot=f"l2_buf_A{ng}",
          event=f"ev_dma_A{ng}",
          bytes_total=bytes_a,
          comment=f"Group DMA prefetch A chunk {ng} (overlap compute g={g})",
        )
      )
      actions.append(
        GroupDMAPrefetchOp(
          descriptor=f"gdma_prefetch_B{ng}",
          l2_slot=f"l2_buf_B{ng}",
          event=f"ev_dma_B{ng}",
          bytes_total=bytes_b,
          comment=f"Group DMA prefetch B chunk {ng} (overlap compute g={g})",
        )
      )

    # wait for tile role completion
    actions.append(GroupWaitEventOp(event=f"ev_role_c{g}", comment=f"Wait for chunk {g} tiles"))

    # store results (non-blocking)
    actions.append(
      GroupDMAStoreOp(
        descriptor=f"gdma_store_C{g}",
        l2_slot=f"l2_buf_C{g}",
        event=f"ev_dma_C{g}",
        bytes_total=bytes_c,
        comment=f"Group DMA store C chunk {g} L2->HBM",
      )
    )

    # drain previous store (output double-buffer, overlaps next compute)
    if g >= 1:
      actions.append(
        GroupWaitEventOp(event=f"ev_dma_C{g - 1}", comment=f"Drain store chunk {g - 1} (overlap)")
      )

  # ---- epilogue: drain last store ----
  actions.append(GroupWaitEventOp(event=f"ev_dma_C{num_group_chunks - 1}", comment="Drain last store"))
  actions.append(SignalEventOp(event="group_task_done"))

  task = TileGroupTaskOp(name="tiled_matmul_pipelined_task", streams=[], roles=roles, actions=actions)
  return ModuleOp([task])


def make_tiled_matmul_top_task(num_group_chunks: int = 4, num_k_chunks: int = 4) -> ModuleOp:
  """Tiled matmul task matching ``tiled_matmul_top.ir`` exactly."""
  total_k = 64 * num_k_chunks
  bytes_a = 128 * total_k * 2 * 4
  bytes_b = total_k * 128 * 2
  bytes_c = 128 * 128 * 2 * 4
  roles = [
    TileRoleBindingOp(
      role_id=0, tile_mask=0x0F, program=make_tiled_matmul_tile_program(num_k_chunks=num_k_chunks)
    )
  ]
  actions: list[GroupActionLike] = []

  for g in range(num_group_chunks):
    if g == 0:
      a_comment = "Group DMA prefetch A chunk 0 HBM->L2"
      b_comment = "Group DMA prefetch B chunk 0 HBM->L2"
    else:
      a_comment = f"Group DMA prefetch A chunk {g} (overlap compute g={g - 1})"
      b_comment = f"Group DMA prefetch B chunk {g} (overlap compute g={g - 1})"
    actions.append(
      GroupDMAPrefetchOp(
        descriptor=f"gdma_prefetch_A{g}",
        l2_slot=f"l2_buf_A{g}",
        event=f"ev_dma_A{g}",
        bytes_total=bytes_a,
        comment=a_comment,
      )
    )
    actions.append(
      GroupDMAPrefetchOp(
        descriptor=f"gdma_prefetch_B{g}",
        l2_slot=f"l2_buf_B{g}",
        event=f"ev_dma_B{g}",
        bytes_total=bytes_b,
        comment=b_comment,
      )
    )

  for g in range(num_group_chunks):
    actions.append(GroupWaitEventOp(event=f"ev_dma_A{g}", comment=f"Wait for A chunk {g} DMA"))
    actions.append(GroupWaitEventOp(event=f"ev_dma_B{g}", comment=f"Wait for B chunk {g} DMA"))
    actions.append(
      DispatchRoleOp(role_id=0, event=f"ev_role_c{g}", comment=f"Dispatch tiles for group chunk {g}")
    )

  for g in range(num_group_chunks):
    actions.append(GroupWaitEventOp(event=f"ev_role_c{g}", comment=f"Wait for chunk {g} tiles"))
    actions.append(
      GroupDMAStoreOp(
        descriptor=f"gdma_store_C{g}",
        l2_slot=f"l2_buf_C{g}",
        event=f"ev_dma_C{g}",
        bytes_total=bytes_c,
        comment=f"Group DMA store C chunk {g} L2->HBM",
      )
    )

  for g in range(num_group_chunks):
    actions.append(
      GroupWaitEventOp(
        event=f"ev_dma_C{g}",
        comment=(f"Drain store chunk {g} (overlap)" if g < num_group_chunks - 1 else "Drain last store"),
      )
    )

  actions.append(SignalEventOp(event="group_task_done"))

  task = TileGroupTaskOp(name="tiled_matmul_pipelined_task", streams=[], roles=roles, actions=actions)
  return ModuleOp([task])


def make_tiled_matmul_pipelined_pow_task(num_group_chunks: int = 4, num_k_chunks: int = 4) -> ModuleOp:
  """Tiled matmul + elementwise pow two-stage task with pipelined group IO.

  Mirrors the ``tiled_matmul_pipelined_pow_task`` IR example.  It is the
  ``make_tiled_matmul_pipelined_task`` matmul phase followed by a second
  EVU-pow phase, sharing the same Tile Group.

  Two roles:
    - Role 0 (tiles 0-3): K-chunked tiled matmul with the same group-level
      async IO pipeline as ``tiled_matmul_pipelined`` (DMA prefetch for
      stage g+1 overlaps tile compute for stage g).
    - Role 1 (tiles 0-3): elementwise ``pow`` (EVU) applied to each stage's
      output.  The pow phase starts only after ALL matmul stores (C0..C3)
      are drained to HBM -- a strict phase fence.  Inside the pow phase
      the four HBM->L2 prefetches are issued up front, then each chunk is
      dispatched as soon as its own input is visible in L2.

  Group action sequence:
      [matmul phase: identical to make_tiled_matmul_pipelined_task
       minus the trailing SIGNAL_EVENT, keeping the final
       WAIT ev_dma_C{N-1} as the phase fence]
      DMA_PREFETCH pow_in{0..N-1}            # issue all prefetches up front
      for g in 0..N-1:
          WAIT ev_dma_pow_in{g}
          DISPATCH_ROLE 1 -> ev_role_pow{g}
      for g in 0..N-1:
          WAIT ev_role_pow{g}
          DMA_STORE pow_out{g} -> ev_dma_pow_out{g}
      WAIT ev_dma_pow_out{0..N-1}
      SIGNAL_EVENT group_task_done
  """
  total_k = 64 * num_k_chunks
  bytes_a = 128 * total_k * 2 * 4  # tile_m * total_k * BF16 * 4 tiles
  bytes_b = total_k * 128 * 2  # total_k * tile_n * BF16 (shared)
  bytes_c = 128 * 128 * 2 * 4  # tile_m * tile_n * BF16 * 4 tiles
  # pow operates on one stage's C (128x128 per tile, BF16) across 4 tiles.
  bytes_pow = 128 * 128 * 2 * 4
  roles = [
    TileRoleBindingOp(
      role_id=0, tile_mask=0x0F, program=make_tiled_matmul_tile_program(num_k_chunks=num_k_chunks)
    ),
    TileRoleBindingOp(
      role_id=1,
      tile_mask=0x0F,
      program=make_pow_tile_program(name="pow_4k_tile", chunk_bytes=128 * 128 * 2),
    ),
  ]
  actions: list[GroupActionLike] = []

  # ---- matmul phase: same pipeline as make_tiled_matmul_pipelined_task ----
  # prologue: prefetch first chunk
  actions.append(
    GroupDMAPrefetchOp(
      descriptor="gdma_prefetch_A0",
      l2_slot="l2_buf_A0",
      event="ev_dma_A0",
      bytes_total=bytes_a,
      comment="Group DMA prefetch A chunk 0 HBM->L2",
    )
  )
  actions.append(
    GroupDMAPrefetchOp(
      descriptor="gdma_prefetch_B0",
      l2_slot="l2_buf_B0",
      event="ev_dma_B0",
      bytes_total=bytes_b,
      comment="Group DMA prefetch B chunk 0 HBM->L2",
    )
  )

  for g in range(num_group_chunks):
    actions.append(GroupWaitEventOp(event=f"ev_dma_A{g}", comment=f"Wait for A chunk {g} DMA"))
    actions.append(GroupWaitEventOp(event=f"ev_dma_B{g}", comment=f"Wait for B chunk {g} DMA"))
    actions.append(
      DispatchRoleOp(role_id=0, event=f"ev_role_c{g}", comment=f"Dispatch tiles for group chunk {g}")
    )
    if g < num_group_chunks - 1:
      ng = g + 1
      actions.append(
        GroupDMAPrefetchOp(
          descriptor=f"gdma_prefetch_A{ng}",
          l2_slot=f"l2_buf_A{ng}",
          event=f"ev_dma_A{ng}",
          bytes_total=bytes_a,
          comment=f"Group DMA prefetch A chunk {ng} (overlap compute g={g})",
        )
      )
      actions.append(
        GroupDMAPrefetchOp(
          descriptor=f"gdma_prefetch_B{ng}",
          l2_slot=f"l2_buf_B{ng}",
          event=f"ev_dma_B{ng}",
          bytes_total=bytes_b,
          comment=f"Group DMA prefetch B chunk {ng} (overlap compute g={g})",
        )
      )
    actions.append(GroupWaitEventOp(event=f"ev_role_c{g}", comment=f"Wait for chunk {g} tiles"))
    actions.append(
      GroupDMAStoreOp(
        descriptor=f"gdma_store_C{g}",
        l2_slot=f"l2_buf_C{g}",
        event=f"ev_dma_C{g}",
        bytes_total=bytes_c,
        comment=f"Group DMA store C chunk {g} L2->HBM",
      )
    )
    if g >= 1:
      actions.append(
        GroupWaitEventOp(event=f"ev_dma_C{g - 1}", comment=f"Drain store chunk {g - 1} (overlap)")
      )

  # phase fence: drain last matmul store before pow stage starts
  actions.append(
    GroupWaitEventOp(event=f"ev_dma_C{num_group_chunks - 1}", comment="Drain last store before pow stage")
  )

  # ---- pow phase: issue all prefetches, then dispatch per-chunk ----
  for g in range(num_group_chunks):
    actions.append(
      GroupDMAPrefetchOp(
        descriptor=f"gdma_prefetch_pow{g}",
        l2_slot=f"l2_buf_pow{g}",
        event=f"ev_dma_pow_in{g}",
        bytes_total=bytes_pow,
        comment=f"Pow input chunk {g} HBM->L2",
      )
    )

  for g in range(num_group_chunks):
    actions.append(
      GroupWaitEventOp(event=f"ev_dma_pow_in{g}", comment=f"Pow chunk {g} input visible in L2")
    )
    actions.append(
      DispatchRoleOp(role_id=1, event=f"ev_role_pow{g}", comment=f"Launch pow tile program for chunk {g}")
    )

  for g in range(num_group_chunks):
    actions.append(GroupWaitEventOp(event=f"ev_role_pow{g}", comment=f"Pow tiles finished chunk {g}"))
    actions.append(
      GroupDMAStoreOp(
        descriptor=f"gdma_store_pow{g}",
        l2_slot=f"l2_buf_pow{g}",
        event=f"ev_dma_pow_out{g}",
        bytes_total=bytes_pow,
        comment=f"Pow output chunk {g} L2->HBM",
      )
    )

  # drain all pow outputs
  for g in range(num_group_chunks):
    actions.append(GroupWaitEventOp(event=f"ev_dma_pow_out{g}", comment=f"Drain pow output chunk {g}"))

  actions.append(SignalEventOp(event="group_task_done"))

  task = TileGroupTaskOp(name="tiled_matmul_pipelined_pow_task", streams=[], roles=roles, actions=actions)
  return ModuleOp([task])


def make_tiled_matmul_pow_nodep_task() -> ModuleOp:
  """Tiled matmul + pow trace fixture with no inter-role dependency fence.

  Matches ``tiled_matmul_pow_nodep.ir`` exactly after whitespace
  normalization.  This is a fixed-shape trace fixture, not a generic
  workload builder: pow inputs are prefetched and dispatched before any
  matmul A/B prefetch or matmul role dispatch, and the output stores drain
  in the fixture order.
  """
  num_group_chunks = 4
  num_k_chunks = 4
  total_k = 64 * num_k_chunks
  bytes_a = 128 * total_k * 2 * 4
  bytes_b = total_k * 128 * 2
  bytes_c = 128 * 128 * 2 * 4
  chunk_bytes = 128 * 128 * 2
  bytes_pow = chunk_bytes * 4

  roles = [
    TileRoleBindingOp(
      role_id=0, tile_mask=0x0F, program=make_tiled_matmul_tile_program(num_k_chunks=num_k_chunks)
    ),
    TileRoleBindingOp(
      role_id=1, tile_mask=0x0F, program=make_pow_tile_program(name="pow_4k_tile", chunk_bytes=chunk_bytes)
    ),
  ]

  actions: list[GroupActionLike] = []

  for g in range(num_group_chunks):
    actions.append(
      GroupDMAPrefetchOp(
        descriptor=f"gdma_prefetch_pow{g}",
        l2_slot=f"l2_buf_pow{g}",
        event=f"ev_dma_pow_in{g}",
        bytes_total=bytes_pow,
        comment=f"Pow input chunk {g} HBM->L2",
      )
    )

  for g in range(num_group_chunks):
    actions.append(
      GroupWaitEventOp(event=f"ev_dma_pow_in{g}", comment=f"Pow chunk {g} input visible in L2")
    )
    actions.append(
      DispatchRoleOp(role_id=1, event=f"ev_role_pow{g}", comment=f"Launch pow tile program for chunk {g}")
    )

  for g in range(num_group_chunks):
    a_comment = (
      "Group DMA prefetch A chunk 0 HBM->L2"
      if g == 0
      else f"Group DMA prefetch A chunk {g} (overlap compute g={g - 1})"
    )
    b_comment = (
      "Group DMA prefetch B chunk 0 HBM->L2"
      if g == 0
      else f"Group DMA prefetch B chunk {g} (overlap compute g={g - 1})"
    )
    actions.append(
      GroupDMAPrefetchOp(
        descriptor=f"gdma_prefetch_A{g}",
        l2_slot=f"l2_buf_A{g}",
        event=f"ev_dma_A{g}",
        bytes_total=bytes_a,
        comment=a_comment,
      )
    )
    actions.append(
      GroupDMAPrefetchOp(
        descriptor=f"gdma_prefetch_B{g}",
        l2_slot=f"l2_buf_B{g}",
        event=f"ev_dma_B{g}",
        bytes_total=bytes_b,
        comment=b_comment,
      )
    )
  for g in range(num_group_chunks):
    a_wait_comment = "Wait for A chunk 2 DMAd" if g == 2 else f"Wait for A chunk {g} DMA"
    actions.append(GroupWaitEventOp(event=f"ev_dma_A{g}", comment=a_wait_comment))
    actions.append(GroupWaitEventOp(event=f"ev_dma_B{g}", comment=f"Wait for B chunk {g} DMA"))
    actions.append(
      DispatchRoleOp(role_id=0, event=f"ev_role_c{g}", comment=f"Dispatch tiles for group chunk {g}")
    )

  for g in range(num_group_chunks):
    actions.append(GroupWaitEventOp(event=f"ev_role_c{g}", comment=f"Wait for chunk {g} tiles"))
    actions.append(
      GroupDMAStoreOp(
        descriptor=f"gdma_store_C{g}",
        l2_slot=f"l2_buf_C{g}",
        event=f"ev_dma_C{g}",
        bytes_total=bytes_c,
        comment=f"Group DMA store C chunk {g} L2->HBM",
      )
    )

  actions.append(GroupWaitEventOp(event="ev_role_pow0", comment="Pow tiles finished chunk 0"))
  actions.append(
    GroupDMAStoreOp(
      descriptor="gdma_store_pow0",
      l2_slot="l2_buf_pow0",
      event="ev_dma_pow_out0",
      bytes_total=bytes_pow,
      comment="Pow output chunk 0 L2->HBM",
    )
  )

  for g in range(1, num_group_chunks):
    actions.append(GroupWaitEventOp(event=f"ev_role_pow{g}", comment=f"Pow tiles finished chunk {g}"))
    actions.append(
      GroupDMAStoreOp(
        descriptor=f"gdma_store_pow{g}",
        l2_slot=f"l2_buf_pow{g}",
        event=f"ev_dma_pow_out{g}",
        bytes_total=bytes_pow,
        comment=f"Pow output chunk {g} L2->HBM",
      )
    )

  actions.append(GroupWaitEventOp(event="ev_dma_C0", comment="Drain store chunk 0 (overlap)"))
  actions.append(GroupWaitEventOp(event="ev_dma_C1", comment="Drain store chunk 1 (overlap)"))
  actions.append(GroupWaitEventOp(event="ev_dma_C2", comment="Drain store chunk 2 (overlap)"))
  actions.append(GroupWaitEventOp(event="ev_dma_C3", comment="Drain last store"))

  for g in range(num_group_chunks):
    actions.append(GroupWaitEventOp(event=f"ev_dma_pow_out{g}", comment=f"Drain pow output chunk {g}"))

  actions.append(SignalEventOp(event="group_task_done"))

  task = TileGroupTaskOp(name="tiled_matmul_pipelined_task", streams=[], roles=roles, actions=actions)
  return ModuleOp([task])


def make_tiled_matmul_persistent_task(num_group_chunks: int = 4, num_k_chunks: int = 4) -> ModuleOp:
  """Persistent single-dispatch tiled matmul with cross-chunk L2->L1 overlap.

  Unlike ``make_tiled_matmul_pipelined_task`` (which re-dispatches the tile
  program per group chunk, creating an inter-program bubble), this task
  dispatches the tile program **once**.  The persistent tile program
  iterates over all group chunks internally, using the cross-level event
  bridge to ``WAIT`` on sequencer-issued ``ev_dma_A{g}`` / ``ev_dma_B{g}``
  events.  This lets chunk g+1's prologue L2->L1 load overlap with chunk g's
  BOA compute -- eliminating the inter-program bubble.

  Group action sequence:
      prologue: DMA_PREFETCH A0, B0
      WAIT A0, B0
      [for g in 0..N-2: DMA_PREFETCH A{g+1}, B{g+1}]  # all prefetches
      DISPATCH_ROLE 0 -> ev_role0                      # single dispatch
      WAIT ev_role0                                     # persistent program finishes
      [for g in 0..N-1: DMA_STORE C{g}, WAIT ev_dma_C{g-1}]
      WAIT ev_dma_C{N-1}
      SIGNAL_EVENT group_task_done

  Note: this is a **different workload** from ``tiled_matmul_pipelined``,
  not a drop-in replacement.  The group-level IO pipeline differs:
  all HBM->L2 prefetches are issued before dispatch (they are async DMA
  jobs that complete independently).  The cross-chunk L2->L1 load
  overlap -- the key benefit -- happens *inside* the persistent tile
  program: chunk g+1's prologue load is issued behind chunk g's last
  BOA compute, gated by ``WAIT ev_dma_A{g+1}`` (bridged event).  Group
  DMA_STORE is issued after the tile program completes because the
  tile program fills ``l2_buf_C{g}`` via MFE stores; store overlap is
  not modelled at the group level in this variant.
  """
  total_k = 64 * num_k_chunks
  bytes_a = 128 * total_k * 2 * 4
  bytes_b = total_k * 128 * 2
  bytes_c = 128 * 128 * 2 * 4
  roles = [
    TileRoleBindingOp(
      role_id=0,
      tile_mask=0x0F,
      program=make_tiled_matmul_persistent_tile_program(
        num_group_chunks=num_group_chunks, num_k_chunks=num_k_chunks
      ),
    )
  ]
  actions: list[GroupActionLike] = []

  # prologue: prefetch first chunk
  actions.append(
    GroupDMAPrefetchOp(
      descriptor="gdma_prefetch_A0",
      l2_slot="l2_buf_A0",
      event="ev_dma_A0",
      bytes_total=bytes_a,
      comment="Group DMA prefetch A chunk 0 HBM->L2",
    )
  )
  actions.append(
    GroupDMAPrefetchOp(
      descriptor="gdma_prefetch_B0",
      l2_slot="l2_buf_B0",
      event="ev_dma_B0",
      bytes_total=bytes_b,
      comment="Group DMA prefetch B chunk 0 HBM->L2",
    )
  )
  actions.append(GroupWaitEventOp(event="ev_dma_A0", comment="Wait for A chunk 0 DMA"))
  actions.append(GroupWaitEventOp(event="ev_dma_B0", comment="Wait for B chunk 0 DMA"))

  # issue all remaining prefetches before dispatch (async DMA jobs;
  # the tile program gates L2->L1 loads on ev_dma_A{g+1} bridged events,
  # so the overlap happens inside the persistent program)
  for g in range(num_group_chunks - 1):
    ng = g + 1
    actions.append(
      GroupDMAPrefetchOp(
        descriptor=f"gdma_prefetch_A{ng}",
        l2_slot=f"l2_buf_A{ng}",
        event=f"ev_dma_A{ng}",
        bytes_total=bytes_a,
        comment=f"Group DMA prefetch A chunk {ng}",
      )
    )
    actions.append(
      GroupDMAPrefetchOp(
        descriptor=f"gdma_prefetch_B{ng}",
        l2_slot=f"l2_buf_B{ng}",
        event=f"ev_dma_B{ng}",
        bytes_total=bytes_b,
        comment=f"Group DMA prefetch B chunk {ng}",
      )
    )

  # single dispatch: persistent tile program handles all chunks
  actions.append(
    DispatchRoleOp(role_id=0, event="ev_role0", comment="Dispatch persistent tile program (all chunks)")
  )
  actions.append(GroupWaitEventOp(event="ev_role0", comment="Wait for persistent tile program"))

  # store all chunks after tile program completes (L2->HBM)
  for g in range(num_group_chunks):
    actions.append(
      GroupDMAStoreOp(
        descriptor=f"gdma_store_C{g}",
        l2_slot=f"l2_buf_C{g}",
        event=f"ev_dma_C{g}",
        bytes_total=bytes_c,
        comment=f"Group DMA store C chunk {g} L2->HBM",
      )
    )
    if g >= 1:
      actions.append(GroupWaitEventOp(event=f"ev_dma_C{g - 1}", comment=f"Drain store chunk {g - 1}"))
  actions.append(GroupWaitEventOp(event=f"ev_dma_C{num_group_chunks - 1}", comment="Drain last store"))
  actions.append(SignalEventOp(event="group_task_done"))

  task = TileGroupTaskOp(name="tiled_matmul_persistent_task", streams=[], roles=roles, actions=actions)
  return ModuleOp([task])


def make_conv_relu_task() -> ModuleOp:
  """Task that dispatches a fused Conv+ReLU role across 4 tiles.

  Role 0 (tiles 0-3) runs the conv_relu tile program.  No inter-tile
  stream; single role validates BOA conv compute + EVU relu epilogue
  fusion + MFE load overlap.  This exercises the BOA->EVU producer-
  consumer path within a single tile (no Stream Queue needed -- the UCE
  serializes them).
  """
  roles = [TileRoleBindingOp(role_id=0, tile_mask=0x0F, program=make_conv_relu_tile_program())]
  actions: list[GroupActionLike] = [
    DispatchRoleOp(role_id=0, event="ev_role0"),
    GroupWaitEventOp(event="ev_role0"),
    SignalEventOp(event="group_task_done"),
  ]
  task = TileGroupTaskOp(name="conv_relu_task", streams=[], roles=roles, actions=actions)
  return ModuleOp([task])


def make_paged_attention_task() -> ModuleOp:
  """Single-role paged-attention task across 4 tiles.

  Each tile runs the full paged-attention pipeline (Architecture 20.2):
  MFE page-walk gathers K/V pages, BOA does QK and PV, EVU does
  scale/mask + softmax.  No inter-tile stream -- every tile independently
  processes its own query block against the KV cache.  This validates
  the MFE Page Stream + multi-step EVU + dual-BOA (QK then PV) path and
  the T_prefetch <= T_qk overlap condition (Architecture 21.3).
  """
  roles = [TileRoleBindingOp(role_id=0, tile_mask=0x0F, program=make_paged_attention_tile_program())]
  actions: list[GroupActionLike] = [
    DispatchRoleOp(role_id=0, event="ev_role0"),
    GroupWaitEventOp(event="ev_role0"),
    SignalEventOp(event="group_task_done"),
  ]
  task = TileGroupTaskOp(name="paged_attention_task", streams=[], roles=roles, actions=actions)
  return ModuleOp([task])


def make_attention_task(block_count: int = 4) -> ModuleOp:
  """Two-role paged-attention-style task with a Stream Queue.

  Role 0 (tiles 0-1): source tiles -- QK matmul, push score tiles into S0.
  Role 1 (tiles 2-3): consumer tiles -- softmax (EVU) + AV matmul, consume S0.

  This exercises the Stream Queue producer-consumer pipeline, credit
  backpressure, and the BOA/EVU cross-engine overlap the specs predict
  (Architecture 21.3: T_prefetch <= T_qk enables overlap).
  """
  streams = [StreamDescOp(queue_id=0, depth=3, producer_mask=0x03, consumer_mask=0x0C)]

  qk_tile = make_stream_pipeline_tile_program(
    in_q=None,
    out_q=0,  # source: no input stream, produces S0
    body_descs=[
      MFEDescriptorOp("qk_load", "load", {"bytes": 128 * 64 * 2 * 2, "ops": 0}),
      BOADescriptorOp("qk", "matmul", {"m": 128, "n": 64, "k": 64, "ops": 2 * 128 * 64 * 64}),
      MFEDescriptorOp("qk_store", "store", {"bytes": 128 * 64 * 2, "ops": 0}),
    ],
    producer_id=0,
    block_count=block_count,
  )
  av_tile = make_stream_pipeline_tile_program(
    in_q=0,
    out_q=-1,  # sink: consume S0, no output stream
    body_descs=[
      EVUDescriptorOp("softmax", "softmax", {"bytes": 128 * 64 * 2, "ops": 128 * 64 * 3}),
      MFEDescriptorOp("av_load", "load", {"bytes": 64 * 128 * 2, "ops": 0}),
      BOADescriptorOp("av", "matmul", {"m": 128, "n": 128, "k": 64, "ops": 2 * 128 * 128 * 64}),
      MFEDescriptorOp("av_store", "store", {"bytes": 128 * 128 * 2, "ops": 0}),
    ],
    producer_id=1,
  )
  roles = [
    TileRoleBindingOp(role_id=0, tile_mask=0x03, program=qk_tile, out_stream=0),
    TileRoleBindingOp(role_id=1, tile_mask=0x0C, program=av_tile, in_stream=0),
  ]
  actions: list[GroupActionLike] = [
    InitStreamOp(queue_id=0, depth=3, producer_mask=0x03, consumer_mask=0x0C, comment="init S0"),
    DispatchRoleOp(role_id=0, event="ev_role0"),
    DispatchRoleOp(role_id=1, event="ev_role1"),
    GroupWaitEventOp(event="ev_role1"),
    SignalEventOp(event="group_task_done"),
  ]
  task = TileGroupTaskOp(name="attention_task", streams=streams, roles=roles, actions=actions)
  return ModuleOp([task])


def make_moe_task(num_experts: int = 8, tokens_per_batch: int = 1024, block_count: int = 4) -> ModuleOp:
  """MoE task: token-grouped expert matmul.

  Role 0 (tiles 0-1): source -- MFE segment-stream groups tokens per expert.
  Role 1 (tiles 2-3): consumer -- BOA runs expert MLP matmul per group.

  Models the MoE imbalance effect (BOA design 6.2: U_boa = 1/imbalance).
  """
  del tokens_per_batch
  streams = [StreamDescOp(queue_id=0, depth=3, producer_mask=0x03, consumer_mask=0x0C)]

  group_tile = make_stream_pipeline_tile_program(
    in_q=None,
    out_q=0,
    body_descs=[
      MFEDescriptorOp(
        "seg_load", "segment_stream", {"bytes": 256 * 64 * 2, "ops": 0, "groups": num_experts}
      ),
      MFEDescriptorOp("seg_push", "store", {"bytes": 256 * 64 * 2, "ops": 0}),
    ],
    producer_id=0,
    block_count=block_count,
  )
  expert_tile = make_stream_pipeline_tile_program(
    in_q=0,
    out_q=-1,
    body_descs=[
      MFEDescriptorOp("expert_load", "load", {"bytes": 256 * 256 * 2, "ops": 0}),
      BOADescriptorOp("expert_mm", "matmul", {"m": 256, "n": 256, "k": 256, "ops": 2 * 256 * 256 * 256}),
      MFEDescriptorOp("expert_store", "store", {"bytes": 256 * 256 * 2, "ops": 0}),
    ],
    producer_id=1,
  )
  roles = [
    TileRoleBindingOp(role_id=0, tile_mask=0x03, program=group_tile, out_stream=0),
    TileRoleBindingOp(role_id=1, tile_mask=0x0C, program=expert_tile, in_stream=0),
  ]
  actions: list[GroupActionLike] = [
    InitStreamOp(queue_id=0, depth=3, producer_mask=0x03, consumer_mask=0x0C),
    DispatchRoleOp(role_id=0, event="ev_role0"),
    DispatchRoleOp(role_id=1, event="ev_role1"),
    GroupWaitEventOp(event="ev_role1"),
    SignalEventOp(event="group_task_done"),
  ]
  task = TileGroupTaskOp(name="moe_task", streams=streams, roles=roles, actions=actions)
  return ModuleOp([task])
