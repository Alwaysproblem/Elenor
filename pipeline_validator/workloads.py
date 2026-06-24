"""Workload definitions.

Each workload builds a RegionProgram (and its TilePrograms) plus a
human-readable description.  The validator runs the region and compares
the measured PMU fingerprint against the architecture's predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import WorkloadConfig
from .ir import (
    RegionProgram,
    make_attention_region,
    make_conv_relu_region,
    make_matmul_region,
    make_moe_region,
    make_paged_attention_region,
    make_tiled_matmul_region,
)


@dataclass
class Workload:
    """Base workload: a name + a RegionProgram + expected PMU observations."""
    name: str
    description: str
    region: RegionProgram
    expected: dict = field(default_factory=dict)  # expected PMU fingerprint
    config: WorkloadConfig | None = None


class MatmulWorkload(Workload):
    """Dense GEMM across 4 tiles.

    Expected (Architecture 21.2): BOA-bound; high BOA active, low stream
    stall, MFE loads overlap with BOA compute.
    """

    def __init__(self, cfg: WorkloadConfig | None = None):
        cfg = cfg or WorkloadConfig(name="matmul", m=512, n=512, k=512)
        region = make_matmul_region()
        super().__init__(
            name="matmul",
            description=("Dense GEMM (512x512x512, BF16) across 4 tiles. "
                         "Single stage, no inter-tile stream. "
                         "Validates BOA peak compute + MFE/DMA load overlap."),
            region=region,
            expected={
                "primary_bottleneck": "BOA",
                "boa_active_ratio_min": 0.40,
                "stream_stall_ratio_max": 0.05,
                "mfe_active_ratio_min": 0.10,
            },
            config=cfg,
        )


class TiledMatmulWorkload(Workload):
    """Multi-level tiled GEMM across 4 tiles with K-dimension chunking.

    Unlike `MatmulWorkload` (which runs one monolithic matmul per tile),
    this workload splits K into `num_k_chunks` chunks and uses
    double-buffered MFE prefetch: the MFE load for chunk i+1 is launched
    *before* the BOA wait for chunk i, so MFE memory latency overlaps
    BOA compute.

    Expected (Architecture 21.2 roofline): with good tiling the MFE load
    is fully hidden behind BOA compute, so:
      - BOA active ratio stays high (compute-bound).
      - MFE active ratio stays positive (loads are issued) but the total
        cycle count is dominated by BOA, not by serial load->compute.
      - The tiled workload should complete in *fewer* cycles than a naive
        serial equivalent (load all -> wait -> compute all -> wait), proving
        the pipeline overlap.
    """

    def __init__(self,
                 cfg: WorkloadConfig | None = None,
                 num_k_chunks: int = 4):
        cfg = cfg or WorkloadConfig(name="tiled_matmul")
        region = make_tiled_matmul_region(num_k_chunks=num_k_chunks)
        super().__init__(
            name="tiled_matmul",
            description=(
                f"Multi-level tiled GEMM (128x128 output, K split into "
                f"{num_k_chunks} chunks of 64, BF16) across 4 tiles. "
                "Double-buffered MFE prefetch overlaps BOA compute. "
                "Validates K-dimension tiling + pipeline overlap vs the "
                "single-chunk matmul workload."),
            region=region,
            expected={
                "primary_bottleneck": "BOA",
                "boa_active_ratio_min": 0.40,
                "mfe_active_ratio_min": 0.10,
                "stream_stall_ratio_max": 0.05,
                "tiled_overlap": True,
            },
            config=cfg,
        )


class AttentionWorkload(Workload):
    """Paged-attention-style two-stage pipeline (QK -> softmax+AV).

    Expected (Architecture 21.3): with T_prefetch <= T_qk, MFE prefetch
    overlaps BOA QK; stream S0 carries score tiles stage0->stage1.
    """

    def __init__(self, cfg: WorkloadConfig | None = None):
        cfg = cfg or WorkloadConfig(
            name="attention", seq_len=2048, head_dim=64)
        region = make_attention_region()
        super().__init__(
            name="attention",
            description=(
                "Two-stage attention: stage0 QK matmul (tiles 0-1) -> "
                "stage1 softmax+AV (tiles 2-3), connected by Stream Queue S0. "
                "Validates stream pipeline, credit backpressure, and "
                "BOA/EVU cross-engine overlap."),
            region=region,
            expected={
                "primary_bottleneck": "BOA",
                "stream_s0_occupancy_seen": True,
                "producer_consumer_overlap": True,
                "evu_active_ratio_min": 0.10,
            },
            config=cfg,
        )


class MoEWorkload(Workload):
    """MoE: MFE segment-stream groups tokens, BOA runs expert MLP.

    Expected (BOA design 6.2): BOA utilization bounded by token imbalance;
    MFE segment stream active.
    """

    def __init__(self, cfg: WorkloadConfig | None = None):
        cfg = cfg or WorkloadConfig(
            name="moe", num_experts=8, tokens_per_batch=1024)
        region = make_moe_region(num_experts=cfg.num_experts)
        super().__init__(
            name="moe",
            description=(
                "MoE expert MLP: stage0 MFE segment-stream groups tokens "
                "(tiles 0-1) -> stage1 BOA expert matmul (tiles 2-3). "
                "Validates MFE segment stream + BOA expert batch utilization."
            ),
            region=region,
            expected={
                "primary_bottleneck": "BOA",
                "mfe_segment_active": True,
                "boa_imbalance_effect": True,
            },
            config=cfg,
        )


class ConvReLuWorkload(Workload):
    """Fused Conv + ReLU across 4 tiles.

    Maps the Conv lowering path (BOA design 5.4): im2col transforms the conv
    into a matmul on the OPA array, then EVU applies the relu epilogue.
    Single stage, no inter-tile stream — validates BOA->EVU fusion within
    a tile and that EVU is active (unlike pure matmul where EVU sits idle).

    Expected: BOA-bound (conv is a large matmul), EVU active on the relu
    epilogue, MFE loads overlap with BOA.
    """

    def __init__(self, cfg: WorkloadConfig | None = None):
        cfg = cfg or WorkloadConfig(name="conv_relu")
        region = make_conv_relu_region()
        super().__init__(
            name="conv_relu",
            description=(
                "Fused Conv (128x128, 3x3 kernel, im2col K=1152, BF16) + "
                "ReLU epilogue across 4 tiles.  Single stage, no inter-tile "
                "stream.  Validates BOA conv compute + EVU relu fusion + "
                "MFE load overlap."),
            region=region,
            expected={
                "primary_bottleneck": "BOA",
                "boa_active_ratio_min": 0.40,
                "evu_active_ratio_min": 0.01,
                "mfe_active_ratio_min": 0.01,
                "stream_stall_ratio_max": 0.05,
            },
            config=cfg,
        )


class PagedAttentionWorkload(Workload):
    """Full paged-attention pipeline across 4 tiles.

    Implements the Architecture 20.2 paged-attention Tile Program:
    MFE Page Stream gathers K/V pages -> BOA QK^T -> EVU scale/mask ->
    EVU softmax -> BOA PV -> MFE store.

    Each tile runs the full pipeline independently (single stage, no
    inter-tile stream).  This validates:
      - MFE Page Stream (page-table walk, KV prefetch, reorder) is active.
      - BOA runs two matmuls (QK then PV).
      - EVU runs two steps (scale/mask then softmax).
      - The T_prefetch <= T_qk overlap condition (Architecture 21.3):
        MFE KV prefetch should overlap with BOA QK compute.

    Expected: BOA-bound (two matmuls dominate), EVU active (softmax +
    scale/mask), MFE active (page-stream gather + store).
    """

    def __init__(self, cfg: WorkloadConfig | None = None):
        cfg = cfg or WorkloadConfig(
            name="paged_attention", seq_len=128, head_dim=64)
        region = make_paged_attention_region()
        super().__init__(
            name="paged_attention",
            description=(
                "Full paged-attention pipeline (Architecture 20.2): "
                "MFE page-stream gathers K/V pages (8 pages x 16 tokens, "
                "head_dim=64, BF16) -> BOA QK^T -> EVU scale/mask -> "
                "EVU softmax -> BOA PV -> MFE store.  Single stage across "
                "4 tiles, no inter-tile stream.  Validates MFE Page Stream "
                "+ dual-BOA (QK+PV) + multi-step EVU + the T_prefetch <= "
                "T_qk overlap condition."),
            region=region,
            expected={
                "primary_bottleneck": "BOA",
                "boa_active_ratio_min": 0.40,
                "evu_active_ratio_min": 0.05,
                "mfe_active_ratio_min": 0.01,
                "stream_stall_ratio_max": 0.05,
                "mfe_page_stream_active": True,
                "dual_boa_qk_pv": True,
            },
            config=cfg,
        )


ALL_WORKLOADS: list = [
    MatmulWorkload, TiledMatmulWorkload, ConvReLuWorkload,
    PagedAttentionWorkload, AttentionWorkload, MoEWorkload
]
