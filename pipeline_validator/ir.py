"""Tile/Region Program IR for the ELENOR pipeline validator.

Mirrors the Tile-SPMD programming model and the Region/Tile Program
contracts in design/ELENOR_Architecture_Design_v1.md sections 16-17.

A *Tile Program* is a list of `TileInst` executed by a Tile UCE on each
Compute Tile.  A *Region Program* is a list of `RegionInst` executed by
the Region Sequencer on the Tile Group.  Both are pure data; the
controllers in `tile.py` / `region.py` interpret them cycle by cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Tile UCE ISA  (Architecture doc 16.6 / 17.6)
# ---------------------------------------------------------------------------


class TileOp(Enum):
    NOP = "nop"
    # Control
    MOV = "mov"
    ADD = "add"
    CMP = "cmp"
    BR = "br"  # unconditional branch to label
    BRP = "brp"  # branch on predicate
    BR_EOS = "br_eos"  # branch if input token is EOS
    RET = "ret"  # return / signal_tile_done
    # Engine launch  (returns an event id the UCE can wait on)
    LAUNCH_BOA = "launch.boa"
    LAUNCH_EVU = "launch.evu"
    LAUNCH_MFE = "launch.mfe"
    LAUNCH_USE = "launch.use"
    LAUNCH_DMA_LOAD = "dma_load"  # L2 -> L1
    LAUNCH_DMA_STORE = "dma_store"  # L1 -> L2
    # Sync
    WAIT = "wait"  # wait for a named event
    WAITALL = "waitall"  # wait for a set of events
    FENCE = "fence"
    # Stream  (Stream Queue design 4.2)
    STREAM_POP = "stream.pop"
    STREAM_PUSH = "stream.push"
    STREAM_ACQUIRE = "stream.acquire"
    STREAM_RELEASE = "stream.release"
    STREAM_PUSH_EOS = "stream.eos"
    # Descriptor
    PATCH_DESC = "patch.desc"
    LOAD_DESC = "load.desc"
    STORE_DESC = "store.desc"
    # Profiling / error
    PROF_BEGIN = "prof.begin"
    PROF_END = "prof.end"
    TRAP = "trap"


class RegionOp(Enum):
    """Region Sequencer opcodes (Architecture doc 16.5 + Region Sequencer 4.1)."""
    REGION_BEGIN = "region.begin"
    REGION_END = "region.end"
    INIT_STREAM = "init.stream"
    DMA_PREFETCH = "dma.prefetch"
    DMA_STORE = "dma.store"
    DISPATCH_STAGE = "dispatch.stage"
    WAIT_EVENT = "wait.event"
    WAIT_STREAM = "wait.stream"
    WAIT_CREDIT = "wait.credit"
    BARRIER_GROUP = "barrier.group"
    COLLECTIVE_RUN = "collective.run"
    PUSH_EOS = "push.eos"
    ADVANCE_BLOCK = "advance.block"
    BRANCH_LT = "branch.lt"
    SIGNAL_EVENT = "signal.event"


# ---------------------------------------------------------------------------
# Instruction records
# ---------------------------------------------------------------------------


@dataclass
class TileInst:
    """One Tile UCE instruction."""
    op: TileOp
    # generic operands resolved by the controller; meaning depends on op.
    # For launch.* : a descriptor name (str).
    # For wait/waitall : an event id (int) or set of ids.
    # For stream.* : a queue id (int) / token register name (str).
    # For br/brp : a target label (str).
    dst: str | None = None  # event/register produced
    args: tuple = ()
    label: str | None = None  # label this instruction carries (branch target)
    comment: str = ""


@dataclass
class RegionInst:
    """One Region Sequencer instruction."""
    op: RegionOp
    args: tuple = ()
    label: str | None = None
    dst: str | None = None  # event id produced
    comment: str = ""


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


@dataclass
class StreamDesc:
    """Stream Queue descriptor (Stream Queue design 4.1 elenor_stream_queue_desc_v0_t)."""
    queue_id: int
    depth: int
    producer_mask: int  # which stages produce
    consumer_mask: int  # which stages consume
    payload_slot_id: int = 0
    token_stride: int = 32
    pmu_stream_id: int = 0


@dataclass
class EngineDesc:
    """An engine descriptor template (Architecture doc 17.5).

    `kind` is one of BOA/EVU/MFE/USE/DMA.  `params` carries op-specific
    fields.  The validator uses `params['bytes']` (payload size) and
    `params['ops']` (compute work) to derive latency from the hardware model.
    """
    name: str
    kind: str  # "BOA" | "EVU" | "MFE" | "USE" | "DMA"
    op: str  # matmul | relu | page_stream | ...
    params: dict = field(default_factory=dict)


# A canonical DMA descriptor (Architecture doc 12.6 elenor_dma_desc_t).
@dataclass
class DMA_DESC:
    name: str
    bytes_total: int
    src_stride: int = 0
    dst_stride: int = 0
    rows: int = 1


# ---------------------------------------------------------------------------
# Program objects
# ---------------------------------------------------------------------------


@dataclass
class TileProgram:
    """A Tile Program executed by a Compute Tile UCE (Architecture 16.4)."""
    name: str
    insts: list[TileInst] = field(default_factory=list)
    # named descriptors this program may launch.
    descriptors: dict = field(default_factory=dict)  # name -> EngineDesc
    # map label -> instruction index, built lazily.
    _labels: dict = field(default_factory=dict, repr=False)

    def resolve_labels(self) -> None:
        self._labels = {}
        for i, ins in enumerate(self.insts):
            if ins.label is not None:
                self._labels[ins.label] = i

    def label_index(self, label: str) -> int:
        if not self._labels:
            self.resolve_labels()
        return self._labels[label]

    # ---- IR pretty-print ----

    def _fmt_inst(self, ins: TileInst) -> str:
        parts: list[str] = []
        if ins.label is not None:
            parts.append(f"{ins.label}:")
        parts.append(ins.op.value)
        if ins.dst is not None:
            parts.append(f"-> {ins.dst}")
        if ins.args:
            parts.append(", ".join(str(a) for a in ins.args))
        line = " ".join(parts)
        if ins.comment:
            line = f"{line:<40s}  ; {ins.comment}"
        return line

    def _fmt_desc(self, name: str, d: EngineDesc) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in sorted(d.params.items()))
        return f"  {name:<20s} kind={d.kind:<4s} op={d.op:<12s} {params_str}"

    def pretty_print(self) -> str:
        """Return an assembly-style listing of this Tile Program."""
        lines: list[str] = []
        lines.append(f"tile_program {self.name} {{")
        if self.descriptors:
            lines.append("  // descriptors")
            for name, d in self.descriptors.items():
                lines.append(self._fmt_desc(name, d))
            lines.append("")
        lines.append("  // instructions")
        for ins in self.insts:
            lines.append(f"  {self._fmt_inst(ins)}")
        lines.append("}")
        return "\n".join(lines)

@dataclass
class RegionProgram:
    """A Region Program executed by the Region Sequencer (Architecture 16.3)."""
    name: str
    insts: list[RegionInst] = field(default_factory=list)
    streams: list[StreamDesc] = field(default_factory=list)
    tile_programs: dict = field(
        default_factory=dict)  # stage_id -> TileProgram
    _labels: dict = field(default_factory=dict, repr=False)

    def resolve_labels(self) -> None:
        self._labels = {}
        for i, ins in enumerate(self.insts):
            if ins.label is not None:
                self._labels[ins.label] = i

    def label_index(self, label: str) -> int:
        if not self._labels:
            self.resolve_labels()
        return self._labels[label]

    # ---- IR pretty-print ----

    def _fmt_inst(self, ins: RegionInst) -> str:
        parts: list[str] = []
        if ins.label is not None:
            parts.append(f"{ins.label}:")
        parts.append(ins.op.value)
        if ins.dst is not None:
            parts.append(f"-> {ins.dst}")
        if ins.args:
            parts.append(", ".join(str(a) for a in ins.args))
        line = " ".join(parts)
        if ins.comment:
            line = f"{line:<40s}  ; {ins.comment}"
        return line

    def _fmt_stream(self, s: StreamDesc) -> str:
        return (f"  stream q{s.queue_id}: depth={s.depth} "
                f"prod=0x{s.producer_mask:X} cons=0x{s.consumer_mask:X}")

    def pretty_print(self) -> str:
        """Return an assembly-style listing of this Region Program and all
        its Tile Programs."""
        lines: list[str] = []
        lines.append(f"region_program {self.name} {{")
        if self.streams:
            lines.append("  // stream descriptors")
            for s in self.streams:
                lines.append(self._fmt_stream(s))
            lines.append("")
        lines.append("  // region instructions")
        for ins in self.insts:
            lines.append(f"  {self._fmt_inst(ins)}")
        lines.append("}")
        # tile programs
        for stage_id, tp in sorted(self.tile_programs.items()):
            lines.append("")
            lines.append(f"// --- tile program for stage {stage_id} ---")
            lines.append(tp.pretty_print())
        return "\n".join(lines)


# ===========================================================================
# Tile Program builders
# ===========================================================================


def _l(label: str, ins: TileInst) -> TileInst:
    """Attach a label to an instruction."""
    ins.label = label
    return ins


def make_matmul_tile_program() -> TileProgram:
    """Single-tile matmul: load A/B, BOA matmul, store C.

    Mirrors the Tile-SPMD IR example (Architecture 17.4).
    """
    p = TileProgram(name="matmul_tile")
    p.descriptors = {
        "load_A":
        EngineDesc("load_A", "MFE", "load", {
            "bytes": 128 * 256 * 2,
            "ops": 0
        }),
        "load_B":
        EngineDesc("load_B", "MFE", "load", {
            "bytes": 256 * 128 * 2,
            "ops": 0
        }),
        "matmul":
        EngineDesc("matmul", "BOA", "matmul", {
            "m": 128,
            "n": 128,
            "k": 256,
            "ops": 2 * 128 * 128 * 256
        }),
        "store_C":
        EngineDesc("store_C", "MFE", "store", {
            "bytes": 128 * 128 * 2,
            "ops": 0
        }),
    }
    p.insts = [
        TileInst(TileOp.LAUNCH_MFE,
                 dst="e0",
                 args=("load_A", ),
                 comment="load A L2->L1"),
        TileInst(TileOp.LAUNCH_MFE,
                 dst="e1",
                 args=("load_B", ),
                 comment="load B L2->L1"),
        TileInst(TileOp.WAITALL, args=("e0", "e1")),
        TileInst(TileOp.LAUNCH_BOA, dst="e2", args=("matmul", )),
        TileInst(TileOp.WAIT, args=("e2", )),
        TileInst(TileOp.LAUNCH_MFE, dst="e3", args=("store_C", )),
        TileInst(TileOp.WAIT, args=("e3", )),
        TileInst(TileOp.RET),
    ]
    p.resolve_labels()
    return p


def make_tiled_matmul_tile_program(num_k_chunks: int = 4,
                                   tile_m: int = 128,
                                   tile_n: int = 128,
                                   tile_k: int = 64) -> TileProgram:
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
    p = TileProgram(name=f"tiled_matmul_{num_k_chunks}k_tile")
    k_chunk_bytes_a = tile_m * tile_k * 2  # BF16
    k_chunk_bytes_b = tile_k * tile_n * 2
    # per-chunk BOA ops: 2*M*N*K_chunk (accumulate across chunks)
    k_chunk_ops = 2 * tile_m * tile_n * tile_k
    insts: list[TileInst] = []

    # ---- prologue: prefetch chunk 0 inputs ----
    insts.append(
        TileInst(TileOp.LAUNCH_MFE,
                 dst="e_a0",
                 args=("load_A_k0", ),
                 comment="prefetch A chunk 0"))
    insts.append(
        TileInst(TileOp.LAUNCH_MFE,
                 dst="e_b0",
                 args=("load_B_k0", ),
                 comment="prefetch B chunk 0"))
    p.descriptors["load_A_k0"] = EngineDesc("load_A_k0", "MFE", "load", {
        "bytes": k_chunk_bytes_a,
        "ops": 0
    })
    p.descriptors["load_B_k0"] = EngineDesc("load_B_k0", "MFE", "load", {
        "bytes": k_chunk_bytes_b,
        "ops": 0
    })

    # ---- per-chunk loop body (unrolled) ----
    for i in range(num_k_chunks):
        # prefetch chunk i+1 inputs (input double-buffer: overlaps BOA_i)
        if i < num_k_chunks - 1:
            ni = i + 1
            insts.append(
                TileInst(TileOp.LAUNCH_MFE,
                         dst=f"e_a{ni}",
                         args=(f"load_A_k{ni}", ),
                         comment=f"prefetch A chunk {ni} (overlap)"))
            insts.append(
                TileInst(TileOp.LAUNCH_MFE,
                         dst=f"e_b{ni}",
                         args=(f"load_B_k{ni}", ),
                         comment=f"prefetch B chunk {ni} (overlap)"))
            p.descriptors[f"load_A_k{ni}"] = EngineDesc(
                f"load_A_k{ni}", "MFE", "load", {
                    "bytes": k_chunk_bytes_a,
                    "ops": 0
                })
            p.descriptors[f"load_B_k{ni}"] = EngineDesc(
                f"load_B_k{ni}", "MFE", "load", {
                    "bytes": k_chunk_bytes_b,
                    "ops": 0
                })

        # wait for chunk i operands
        insts.append(
            TileInst(TileOp.WAITALL,
                     args=(f"e_a{i}", f"e_b{i}"),
                     comment=f"operands for chunk {i} ready"))

        # BOA accumulate chunk i
        mm_name = f"matmul_k{i}"
        p.descriptors[mm_name] = EngineDesc(
            mm_name, "BOA", "matmul", {
                "m": tile_m,
                "n": tile_n,
                "k": tile_k,
                "ops": k_chunk_ops,
                "chunk": i,
                "accumulate": i > 0
            })
        insts.append(
            TileInst(TileOp.LAUNCH_BOA,
                     dst=f"e_mm{i}",
                     args=(mm_name, ),
                     comment=f"BOA accumulate chunk {i}"))

        # drain previous store while BOA_i runs (output double-buffer)
        if i >= 1:
            insts.append(
                TileInst(TileOp.WAIT,
                         args=(f"e_store{i - 1}", ),
                         comment=f"drain store {i - 1} (overlap BOA{i})"))

        # wait for BOA_i result
        insts.append(
            TileInst(TileOp.WAIT,
                     args=(f"e_mm{i}", ),
                     comment=f"BOA chunk {i} done"))

        # store chunk i output (fire-and-forget: overlaps next BOA)
        store_name = f"store_C_k{i}"
        p.descriptors[store_name] = EngineDesc(
            store_name, "MFE", "store", {
                "bytes": tile_m * tile_n * 2,
                "ops": 0,
                "chunk": i,
            })
        insts.append(
            TileInst(TileOp.LAUNCH_MFE,
                     dst=f"e_store{i}",
                     args=(store_name, ),
                     comment=f"MFE store result chunk {i} (deferred wait)"))

    # ---- epilogue: drain last store ----
    insts.append(
        TileInst(TileOp.WAIT,
                 args=(f"e_store{num_k_chunks - 1}", ),
                 comment="drain last store"))
    insts.append(TileInst(TileOp.RET))
    p.insts = insts
    p.resolve_labels()
    return p


def make_relu_tile_program() -> TileProgram:
    """EVU elementwise relu on a tile."""
    p = TileProgram(name="relu_tile")
    p.descriptors = {
        "load":
        EngineDesc("load", "MFE", "load", {
            "bytes": 128 * 128 * 2,
            "ops": 0
        }),
        "relu":
        EngineDesc("relu", "EVU", "relu", {
            "bytes": 128 * 128 * 2,
            "ops": 128 * 128
        }),
        "store":
        EngineDesc("store", "MFE", "store", {
            "bytes": 128 * 128 * 2,
            "ops": 0
        }),
    }
    p.insts = [
        TileInst(TileOp.LAUNCH_MFE, dst="e0", args=("load", )),
        TileInst(TileOp.WAIT, args=("e0", )),
        TileInst(TileOp.LAUNCH_EVU, dst="e1", args=("relu", )),
        TileInst(TileOp.WAIT, args=("e1", )),
        TileInst(TileOp.LAUNCH_MFE, dst="e2", args=("store", )),
        TileInst(TileOp.WAIT, args=("e2", )),
        TileInst(TileOp.RET),
    ]
    p.resolve_labels()
    return p


def make_conv_relu_tile_program() -> TileProgram:
    """Fused Conv + ReLU tile: load input+weight, BOA conv (im2col->matmul),
    EVU relu epilogue, store output.

    Mirrors the Conv lowering mapping in BOA design 5.4:
    'im2col 或 implicit tile 后进入 OPA MUL; MFE 可选做 layout stream,
    EVU 处理尾部'.  Here the im2col transform is assumed done by the
    compiler/MFE layout stream, so BOA sees a matmul-shaped descriptor.
    The EVU relu is fused as the epilogue after BOA writeback.
    """
    p = TileProgram(name="conv_relu_tile")
    # Conv parameters: input tile 128x128, weight kernel 3x3 over 128 channels,
    # im2col expands to K=128*9=1152 effective K dim, output 128x128.
    p.descriptors = {
        "load_input":
        EngineDesc(
            "load_input",
            "MFE",
            "load",
            {
                "bytes": 128 * 128 * 2,
                "ops": 0
            },
        ),
        "load_weight":
        EngineDesc(
            "load_weight",
            "MFE",
            "load",
            {
                "bytes": 128 * 9 * 2,
                "ops": 0
            },
        ),
        "conv":
        EngineDesc(
            "conv",
            "BOA",
            "conv",
            {
                "m": 128,
                "n": 128,
                "k": 1152,
                "ops": 2 * 128 * 128 * 1152
            },
        ),
        "relu":
        EngineDesc(
            "relu",
            "EVU",
            "relu",
            {
                "bytes": 128 * 128 * 2,
                "ops": 128 * 128
            },
        ),
        "store_output":
        EngineDesc(
            "store_output",
            "MFE",
            "store",
            {
                "bytes": 128 * 128 * 2,
                "ops": 0
            },
        ),
    }
    p.insts = [
        TileInst(TileOp.LAUNCH_MFE,
                 dst="e0",
                 args=("load_input", ),
                 comment="load input patch L2->L1"),
        TileInst(TileOp.LAUNCH_MFE,
                 dst="e1",
                 args=("load_weight", ),
                 comment="load conv weight L2->L1"),
        TileInst(TileOp.WAITALL, args=("e0", "e1")),
        TileInst(TileOp.LAUNCH_BOA,
                 dst="e2",
                 args=("conv", ),
                 comment="BOA conv (im2col matmul)"),
        TileInst(TileOp.WAIT, args=("e2", )),
        TileInst(TileOp.LAUNCH_EVU,
                 dst="e3",
                 args=("relu", ),
                 comment="EVU relu epilogue"),
        TileInst(TileOp.WAIT, args=("e3", )),
        TileInst(TileOp.LAUNCH_MFE,
                 dst="e4",
                 args=("store_output", ),
                 comment="store output L1->L2"),
        TileInst(TileOp.WAIT, args=("e4", )),
        TileInst(TileOp.RET),
    ]
    p.resolve_labels()
    return p


def make_paged_attention_tile_program() -> TileProgram:
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
    This is a single-tile fused pipeline — the UCE serializes the engines,
    so no Stream Queue is needed; MFE prefetch overlap with BOA compute is
    governed by the T_prefetch <= T_qk condition (Architecture 21.3).
    """
    p = TileProgram(name="paged_attention_tile")
    # Canonical paged-attention block: q_len=128, head_dim=64, page_size=16,
    # num_pages=8 (seq_len=128).  K/V pages are gathered by MFE Page Stream.
    kv_page_bytes = 16 * 64 * 2  # one KV page: 16 tokens x 64 head_dim
    kv_total_bytes = 8 * kv_page_bytes  # 8 pages gathered
    score_bytes = 128 * 128 * 2  # QK score: 128 x 128 (q x pages*page_size)
    out_bytes = 128 * 64 * 2  # AV output: 128 x 64
    p.descriptors = {
        "gather_K_pages":
        EngineDesc(
            "gather_K_pages", "MFE", "page_stream", {
                "bytes": kv_total_bytes,
                "ops": 0,
                "mode": "page_stream",
                "num_pages": 8,
                "page_size": 16
            }),
        "gather_V_pages":
        EngineDesc(
            "gather_V_pages", "MFE", "page_stream", {
                "bytes": kv_total_bytes,
                "ops": 0,
                "mode": "page_stream",
                "num_pages": 8,
                "page_size": 16
            }),
        "qk_matmul":
        EngineDesc("qk_matmul", "BOA", "matmul", {
            "m": 128,
            "n": 128,
            "k": 64,
            "ops": 2 * 128 * 128 * 64
        }),
        "scale_mask":
        EngineDesc("scale_mask", "EVU", "scale_mask", {
            "bytes": score_bytes,
            "ops": 128 * 128 * 2
        }),
        "softmax":
        EngineDesc("softmax", "EVU", "softmax", {
            "bytes": score_bytes,
            "ops": 128 * 128 * 8
        }),
        "pv_matmul":
        EngineDesc("pv_matmul", "BOA", "matmul", {
            "m": 128,
            "n": 64,
            "k": 128,
            "ops": 2 * 128 * 64 * 128
        }),
        "store_output":
        EngineDesc("store_output", "MFE", "store", {
            "bytes": out_bytes,
            "ops": 0
        }),
    }
    p.insts = [
        TileInst(TileOp.LAUNCH_MFE,
                 dst="e0",
                 args=("gather_K_pages", ),
                 comment="MFE page-stream gather K pages"),
        TileInst(TileOp.LAUNCH_MFE,
                 dst="e1",
                 args=("gather_V_pages", ),
                 comment="MFE page-stream gather V pages"),
        TileInst(TileOp.WAITALL, args=("e0", "e1")),
        TileInst(TileOp.LAUNCH_BOA,
                 dst="e2",
                 args=("qk_matmul", ),
                 comment="BOA QK^T matmul"),
        TileInst(TileOp.WAIT, args=("e2", )),
        TileInst(TileOp.LAUNCH_EVU,
                 dst="e3",
                 args=("scale_mask", ),
                 comment="EVU scale + causal mask"),
        TileInst(TileOp.WAIT, args=("e3", )),
        TileInst(TileOp.LAUNCH_EVU,
                 dst="e4",
                 args=("softmax", ),
                 comment="EVU softmax over scores"),
        TileInst(TileOp.WAIT, args=("e4", )),
        TileInst(TileOp.LAUNCH_BOA,
                 dst="e5",
                 args=("pv_matmul", ),
                 comment="BOA PV matmul"),
        TileInst(TileOp.WAIT, args=("e5", )),
        TileInst(TileOp.LAUNCH_MFE,
                 dst="e6",
                 args=("store_output", ),
                 comment="MFE store attention output"),
        TileInst(TileOp.WAIT, args=("e6", )),
        TileInst(TileOp.RET),
    ]
    p.resolve_labels()
    return p


def make_stream_pipeline_tile_program(in_q: int | None,
                                      out_q: int,
                                      body_descs: list[EngineDesc],
                                      producer_id: int = 0,
                                      block_count: int = 1) -> TileProgram:
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
    name = ("src_tile_" if is_source else "cons_tile_") + "_".join(
        d.name for d in body_descs)
    p = TileProgram(name=name)
    for d in body_descs:
        p.descriptors[d.name] = d
    launch_descs = [(d.kind, d.name) for d in body_descs]

    insts: list[TileInst] = []
    # ---- loop head ----
    if is_source:
        # source: use a pseudo block counter via a named register.
        # We model the loop with BR after a fixed number of iterations using
        # a CMP+BR pattern simulated by a counter register the UCE tracks.
        # For simplicity: emit `block_count` unrolled iterations (no loop).
        for blk in range(block_count):
            for i, (kind, dname) in enumerate(launch_descs):
                op = {
                    "BOA": TileOp.LAUNCH_BOA,
                    "EVU": TileOp.LAUNCH_EVU,
                    "MFE": TileOp.LAUNCH_MFE,
                    "USE": TileOp.LAUNCH_USE
                }[kind]
                insts.append(
                    TileInst(op,
                             dst=f"e{blk}_{i}",
                             args=(dname, ),
                             comment=f"block {blk} {dname}"))
                insts.append(TileInst(TileOp.WAIT, args=(f"e{blk}_{i}", )))
            # push output token
            insts.append(
                TileInst(TileOp.STREAM_ACQUIRE,
                         dst="out_tok",
                         args=(out_q, ),
                         comment="acquire credit"))
            insts.append(
                TileInst(TileOp.STREAM_PUSH,
                         args=(out_q, "out_tok", producer_id)))
        # after all blocks, push EOS to signal downstream
        insts.append(
            TileInst(TileOp.STREAM_PUSH_EOS,
                     args=(out_q, producer_id),
                     comment="source EOS"))
    else:
        # consumer: loop pop -> body -> push -> release, exit on EOS.
        insts.append(
            _l(
                "loop",
                TileInst(TileOp.STREAM_POP,
                         dst="in_tok",
                         args=(in_q, ),
                         comment="pop input")))
        insts.append(TileInst(TileOp.BR_EOS, args=("in_tok", "done")))
        if out_q >= 0:
            insts.append(
                TileInst(TileOp.STREAM_ACQUIRE,
                         dst="out_tok",
                         args=(out_q, ),
                         comment="acquire credit"))
        for i, (kind, dname) in enumerate(launch_descs):
            op = {
                "BOA": TileOp.LAUNCH_BOA,
                "EVU": TileOp.LAUNCH_EVU,
                "MFE": TileOp.LAUNCH_MFE,
                "USE": TileOp.LAUNCH_USE
            }[kind]
            insts.append(TileInst(op, dst=f"e_body{i}", args=(dname, )))
            insts.append(TileInst(TileOp.WAIT, args=(f"e_body{i}", )))
        if out_q >= 0:
            insts.append(
                TileInst(TileOp.STREAM_PUSH,
                         args=(out_q, "out_tok", producer_id)))
        insts.append(TileInst(TileOp.STREAM_RELEASE, args=(in_q, "in_tok")))
        insts.append(TileInst(TileOp.BR, args=("loop", )))
        insts.append(
            _l("done",
               TileInst(TileOp.STREAM_PUSH_EOS, args=(out_q, producer_id))))

    insts.append(TileInst(TileOp.RET))
    p.insts = insts
    p.resolve_labels()
    return p


def make_identity_tile_program() -> TileProgram:
    """A tile program that does nothing (for pure stage dispatch testing)."""
    p = TileProgram(name="identity_tile")
    p.insts = [TileInst(TileOp.RET)]
    p.resolve_labels()
    return p


# ===========================================================================
# Region Program builders
# ===========================================================================


def make_matmul_region(block_count: int = 4) -> RegionProgram:
    """Region that dispatches a single matmul stage across 4 tiles.

    Stage0 (tiles 0-3) run the matmul tile program.  No inter-tile stream;
    the region prefetches A/B weights HBM->L2 via Group DMA, dispatches
    the stage, then stores C L2->HBM.  This validates Global DMA + stage
    + storeback trace coverage on the TileGroup timeline.
    """
    # Per-tile A = 128*256*2, B = 256*128*2, C = 128*128*2 (BF16).
    # 4-tile M-split: A and C are per-tile (x4), B is shared (x1).
    bytes_a = 128 * 256 * 2 * 4
    bytes_b = 256 * 128 * 2
    bytes_c = 128 * 128 * 2 * 4
    r = RegionProgram(name="matmul_region")
    r.streams = []
    r.tile_programs = {0: make_matmul_tile_program()}
    r.insts = [
        RegionInst(RegionOp.REGION_BEGIN, args=(0, ), comment="region 0"),
        # Group DMA HBM -> L2 prefetch (both prefetches issued before wait
        # so they overlap, per Architecture 16.3).
        RegionInst(RegionOp.DMA_PREFETCH,
                   args=("gdma_prefetch_A", "l2_buf_A", bytes_a),
                   dst="ev_dma_A",
                   comment="Group DMA prefetch A HBM->L2"),
        RegionInst(RegionOp.DMA_PREFETCH,
                   args=("gdma_prefetch_B", "l2_buf_B", bytes_b),
                   dst="ev_dma_B",
                   comment="Group DMA prefetch B HBM->L2"),
        RegionInst(RegionOp.WAIT_EVENT, args=("ev_dma_A", )),
        RegionInst(RegionOp.WAIT_EVENT, args=("ev_dma_B", )),
        RegionInst(
            RegionOp.DISPATCH_STAGE,
            args=(0, 0x0F, 0),  # stage=0, tile_mask=0x0F, program=matmul
            dst="ev_stage0"),
        RegionInst(RegionOp.WAIT_EVENT, args=("ev_stage0", )),
        # Group DMA L2 -> HBM storeback
        RegionInst(RegionOp.DMA_STORE,
                   args=("gdma_store_C", "l2_buf_C", bytes_c),
                   dst="ev_dma_C",
                   comment="Group DMA store C L2->HBM"),
        RegionInst(RegionOp.WAIT_EVENT, args=("ev_dma_C", )),
        RegionInst(RegionOp.REGION_END, dst="ev_region_done"),
    ]
    r.resolve_labels()
    return r


def make_tiled_matmul_region(num_k_chunks: int = 4) -> RegionProgram:
    """Tiled matmul region with K-dimension chunking across 4 tiles.

    Each tile runs the tiled matmul program that splits K into
    `num_k_chunks` chunks of size tile_k and uses double-buffered MFE
    prefetch to overlap memory and compute.  This validates the
    multi-level tiling + pipeline overlap that the single-chunk matmul
    workload cannot expose.

    The region prefetches A/B HBM->L2 via Group DMA before dispatching
    the single stage, then stores C L2->HBM after.
    """
    # K = tile_k * num_k_chunks.  A = M*K*2, B = K*N*2, C = M*N*2 (BF16).
    # 4-tile M-split: A and C are per-tile (x4), B is shared (x1).
    total_k = 64 * num_k_chunks
    bytes_a = 128 * total_k * 2 * 4
    bytes_b = total_k * 128 * 2
    bytes_c = 128 * 128 * 2 * 4
    r = RegionProgram(name="tiled_matmul_region")
    r.streams = []
    r.tile_programs = {
        0: make_tiled_matmul_tile_program(num_k_chunks=num_k_chunks)
    }
    r.insts = [
        RegionInst(RegionOp.REGION_BEGIN,
                   args=(5, ),
                   comment="tiled matmul region"),
        RegionInst(RegionOp.DMA_PREFETCH,
                   args=("gdma_prefetch_A", "l2_buf_A", bytes_a),
                   dst="ev_dma_A",
                   comment="Group DMA prefetch A HBM->L2"),
        RegionInst(RegionOp.DMA_PREFETCH,
                   args=("gdma_prefetch_B", "l2_buf_B", bytes_b),
                   dst="ev_dma_B",
                   comment="Group DMA prefetch B HBM->L2"),
        RegionInst(RegionOp.WAIT_EVENT, args=("ev_dma_A", )),
        RegionInst(RegionOp.WAIT_EVENT, args=("ev_dma_B", )),
        RegionInst(RegionOp.DISPATCH_STAGE, args=(0, 0x0F, 0),
                   dst="ev_stage0"),
        RegionInst(RegionOp.WAIT_EVENT, args=("ev_stage0", )),
        RegionInst(RegionOp.DMA_STORE,
                   args=("gdma_store_C", "l2_buf_C", bytes_c),
                   dst="ev_dma_C",
                   comment="Group DMA store C L2->HBM"),
        RegionInst(RegionOp.WAIT_EVENT, args=("ev_dma_C", )),
        RegionInst(RegionOp.REGION_END, dst="ev_region_done"),
    ]
    r.resolve_labels()
    return r


def make_conv_relu_region() -> RegionProgram:
    """Region that dispatches a fused Conv+ReLU stage across 4 tiles.

    Stage0 (tiles 0-3) run the conv_relu tile program.  No inter-tile stream;
    single stage validates BOA conv compute + EVU relu epilogue fusion +
    MFE load overlap.  This exercises the BOA->EVU producer-consumer path
    within a single tile (no Stream Queue needed — the UCE serializes them).
    """
    r = RegionProgram(name="conv_relu_region")
    r.streams = []
    r.tile_programs = {0: make_conv_relu_tile_program()}
    r.insts = [
        RegionInst(RegionOp.REGION_BEGIN,
                   args=(3, ),
                   comment="conv-relu region"),
        RegionInst(
            RegionOp.DISPATCH_STAGE,
            args=(0, 0x0F, 0),  # stage=0, tile_mask=0x0F, program=conv_relu
            dst="ev_stage0"),
        RegionInst(RegionOp.WAIT_EVENT, args=("ev_stage0", )),
        RegionInst(RegionOp.REGION_END, dst="ev_region_done"),
    ]
    r.resolve_labels()
    return r


def make_paged_attention_region() -> RegionProgram:
    """Single-stage paged-attention region across 4 tiles.

    Each tile runs the full paged-attention pipeline (Architecture 20.2):
    MFE page-walk gathers K/V pages, BOA does QK and PV, EVU does
    scale/mask + softmax.  No inter-tile stream — every tile independently
    processes its own query block against the KV cache.  This validates
    the MFE Page Stream + multi-step EVU + dual-BOA (QK then PV) path and
    the T_prefetch <= T_qk overlap condition (Architecture 21.3).
    """
    r = RegionProgram(name="paged_attention_region")
    r.streams = []
    r.tile_programs = {0: make_paged_attention_tile_program()}
    r.insts = [
        RegionInst(RegionOp.REGION_BEGIN,
                   args=(4, ),
                   comment="paged attention region"),
        RegionInst(
            RegionOp.DISPATCH_STAGE,
            args=(0, 0x0F, 0),  # stage=0, all 4 tiles, program=0
            dst="ev_stage0"),
        RegionInst(RegionOp.WAIT_EVENT, args=("ev_stage0", )),
        RegionInst(RegionOp.REGION_END, dst="ev_region_done"),
    ]
    r.resolve_labels()
    return r


def make_attention_region(block_count: int = 4) -> RegionProgram:
    """Two-stage paged-attention-style region with a Stream Queue.

    Stage0 (tiles 0-1): source tiles — QK matmul, push score tiles into S0.
    Stage1 (tiles 2-3): consumer tiles — softmax (EVU) + AV matmul, consume S0.

    This exercises the Stream Queue producer-consumer pipeline, credit
    backpressure, and the BOA/EVU cross-engine overlap the specs predict
    (Architecture 21.3: T_prefetch <= T_qk enables overlap).
    """
    r = RegionProgram(name="attention_region")
    s0 = StreamDesc(queue_id=0,
                    depth=3,
                    producer_mask=0x03,
                    consumer_mask=0x0C)

    qk_tile = make_stream_pipeline_tile_program(
        in_q=None,
        out_q=0,  # source: no input stream, produces S0
        body_descs=[
            EngineDesc("qk_load", "MFE", "load", {
                "bytes": 128 * 64 * 2 * 2,
                "ops": 0
            }),
            EngineDesc("qk", "BOA", "matmul", {
                "m": 128,
                "n": 64,
                "k": 64,
                "ops": 2 * 128 * 64 * 64
            }),
            EngineDesc("qk_store", "MFE", "store", {
                "bytes": 128 * 64 * 2,
                "ops": 0
            }),
        ],
        producer_id=0,
        block_count=block_count,
    )
    av_tile = make_stream_pipeline_tile_program(
        in_q=0,
        out_q=-1,  # sink: consume S0, no output stream
        body_descs=[
            EngineDesc("softmax", "EVU", "softmax", {
                "bytes": 128 * 64 * 2,
                "ops": 128 * 64 * 3
            }),
            EngineDesc("av_load", "MFE", "load", {
                "bytes": 64 * 128 * 2,
                "ops": 0
            }),
            EngineDesc("av", "BOA", "matmul", {
                "m": 128,
                "n": 128,
                "k": 64,
                "ops": 2 * 128 * 128 * 64
            }),
            EngineDesc("av_store", "MFE", "store", {
                "bytes": 128 * 128 * 2,
                "ops": 0
            }),
        ],
        producer_id=1,
    )
    r.streams = [s0]
    r.tile_programs = {0: qk_tile, 1: av_tile}
    r.insts = [
        RegionInst(RegionOp.REGION_BEGIN,
                   args=(1, ),
                   comment="attention region"),
        RegionInst(RegionOp.INIT_STREAM,
                   args=(0, 3, 0x03, 0x0C),
                   comment="init S0"),
        RegionInst(RegionOp.DISPATCH_STAGE,
                   args=(0, 0x03, 0, 0),
                   dst="ev_stage0"),
        RegionInst(RegionOp.DISPATCH_STAGE,
                   args=(1, 0x0C, 1, 0),
                   dst="ev_stage1"),
        RegionInst(RegionOp.WAIT_EVENT, args=("ev_stage1", )),
        RegionInst(RegionOp.REGION_END, dst="ev_region_done"),
    ]
    r.resolve_labels()
    return r


def make_moe_region(num_experts: int = 8,
                    tokens_per_batch: int = 1024,
                    block_count: int = 4) -> RegionProgram:
    """MoE region: token-grouped expert matmul.

    Stage0 (tiles 0-1): source — MFE segment-stream groups tokens per expert.
    Stage1 (tiles 2-3): consumer — BOA runs expert MLP matmul per group.

    Models the MoE imbalance effect (BOA design 6.2: U_boa = 1/imbalance).
    """
    r = RegionProgram(name="moe_region")
    s0 = StreamDesc(queue_id=0,
                    depth=3,
                    producer_mask=0x03,
                    consumer_mask=0x0C)

    group_tile = make_stream_pipeline_tile_program(
        in_q=None,
        out_q=0,
        body_descs=[
            EngineDesc("seg_load", "MFE", "segment_stream", {
                "bytes": 256 * 64 * 2,
                "ops": 0,
                "groups": num_experts
            }),
            EngineDesc("seg_push", "MFE", "store", {
                "bytes": 256 * 64 * 2,
                "ops": 0
            }),
        ],
        producer_id=0,
        block_count=block_count,
    )
    expert_tile = make_stream_pipeline_tile_program(
        in_q=0,
        out_q=-1,
        body_descs=[
            EngineDesc("expert_load", "MFE", "load", {
                "bytes": 256 * 256 * 2,
                "ops": 0
            }),
            EngineDesc("expert_mm", "BOA", "matmul", {
                "m": 256,
                "n": 256,
                "k": 256,
                "ops": 2 * 256 * 256 * 256
            }),
            EngineDesc("expert_store", "MFE", "store", {
                "bytes": 256 * 256 * 2,
                "ops": 0
            }),
        ],
        producer_id=1,
    )
    r.streams = [s0]
    r.tile_programs = {0: group_tile, 1: expert_tile}
    r.insts = [
        RegionInst(RegionOp.REGION_BEGIN, args=(2, ), comment="moe region"),
        RegionInst(RegionOp.INIT_STREAM, args=(0, 3, 0x03, 0x0C)),
        RegionInst(RegionOp.DISPATCH_STAGE,
                   args=(0, 0x03, 0, 0),
                   dst="ev_stage0"),
        RegionInst(RegionOp.DISPATCH_STAGE,
                   args=(1, 0x0C, 1, 0),
                   dst="ev_stage1"),
        RegionInst(RegionOp.WAIT_EVENT, args=("ev_stage1", )),
        RegionInst(RegionOp.REGION_END, dst="ev_region_done"),
    ]
    r.resolve_labels()
    return r
