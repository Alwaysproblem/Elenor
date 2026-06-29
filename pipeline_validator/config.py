"""Configuration dataclasses for the ELENOR pipeline validator.

All hardware parameters derive from the design/ specs and are tagged with the
exact doc section they come from.  Unfrozen values (the specs mark them with
`由后续规格冻结` / `由 SRAM profile 冻结` / `由 PPA exploration 冻结`) take the
First Silicon V1 recommended value so a baseline simulation is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class HardwareConfig:
    """Static hardware parameters for one Tile Group + 4 Compute Tiles.

    Defaults follow the **Balanced-small** profile
    (design/ELENOR_Architecture_Design_v1.md section 12.3) which the specs call
    the realistic first-silicon configuration.
    """

    # --- Top-level -------------------------------------------------------
    profile: str = "balanced-small"  # 12.3 config table
    num_tiles: int = 4  # assignment scope (1 group, 4 tiles)

    # --- Clocking ---------------------------------------------------------
    # A single clock drives every cycle-accurate counter.  The specs do not
    # freeze a frequency; 1 GHz is the conventional modelling baseline.
    clock_mhz: float = 1000.0

    # --- Group / L2 ------------------------------------------------------
    group_sram_bytes: int = 8 * 1024 * 1024  # 8 MB Balanced-small (12.3)
    group_sram_banks: int = 16
    hbm_bandwidth_gbs: float = 819.2  # 8 HBM stacks * 102.4 GB/s
    group_dma_bandwidth_gbs: float = 204.8  # per-channel DMA peak (frozen later)
    num_dma_channels: int = 2  # concurrent Global DMA channels

    # --- Tile / L1 -------------------------------------------------------
    tile_l1_bytes: int = 1 * 1024 * 1024  # 1 MB Balanced-small (12.3)
    tile_l1_banks: int = 16  # 12.4 banking
    tile_l1_bandwidth_gbs: float = 512.0  # per-tile SRAM read peak (frozen later)

    # --- BOA --------------------------------------------------------------
    # 6.1 / 8.x: 4 OPA per tile, 16x16 outer-product tile.
    boa_num_opa: int = 4
    boa_opa_rows: int = 16
    boa_opa_cols: int = 16
    boa_clock_multiplier: float = 1.0  # BOA runs at core clock
    # Peak TOPS computed in boa_peak tops() from OPA geometry + dtype.
    boa_dtype_bytes: int = 2  # BF16 input
    boa_acc_bytes: int = 4  # INT32 / BF16 accumulate

    # --- EVU --------------------------------------------------------------
    # 9.x: 32-lane predicated vector unit.
    evu_lanes: int = 32
    evu_clock_multiplier: float = 1.0
    evu_dtype_bytes: int = 2  # BF16

    # --- MFE --------------------------------------------------------------
    # 10.x: stream shaping engine.  Bandwidth limited by L1 write port.
    mfe_bandwidth_gbs: float = 256.0  # page/segment stream into L1
    mfe_clock_multiplier: float = 1.0

    # --- USE --------------------------------------------------------------
    # 11.x: state engine, modelled as a small RISC-V co-controller.
    use_clock_mhz: float = 500.0  # slower control core
    use_state_cache_bytes: int = 128 * 1024  # 12.2 partition

    # --- Tile UCE ---------------------------------------------------------
    # 6.3: UCE + USE may share one tile-local RISC-V; UCE issues 1 inst/cycle.
    uce_clock_mhz: float = 1000.0
    uce_dispatch_per_cycle: int = 1

    # --- Stream queue -----------------------------------------------------
    # Stream Queue design 6.1: depth=3 canonical trace from TileGroupTask.
    stream_depth_default: int = 3
    stream_token_overhead_cycles: int = 1  # T_acquire + T_push + T_pop + T_release
    stream_fence_cycles: int = 1  # payload visibility fence

    # --- Engine launch overhead -----------------------------------------
    # Cycles to decode a descriptor and enter an engine (not compute).
    boa_launch_cycles: int = 4
    evu_launch_cycles: int = 3
    mfe_launch_cycles: int = 3
    use_launch_cycles: int = 2
    dma_launch_cycles: int = 2
    # --- Runtime / memory (V2, runtime-level simulator) -----------------
    # All values below are unfrozen and tagged per the spec conventions:
    #   由后续规格冻结 / 由 SRAM profile 冻结 / 由 PPA exploration 冻结
    # Defaults follow the First Silicon V1 recommended values so a baseline
    # simulation is reproducible.
    hbm_capacity_bytes: int = 16 * 1024 * 1024 * 1024  # 16 GB, 由后续规格冻结
    hbm_outstanding_limit: int = 32  # tag CAM depth, 由 PPA exploration 冻结
    l2_bank_bandwidth_gbs: float = 12.8  # per-bank, 由 SRAM profile 冻结
    tile_program_sram_bytes: int = 64 * 1024  # hot tile kernel, 由 SRAM profile 冻结
    noc_vc_depth: int = 8  # 由 PPA exploration 冻结
    noc_router_latency_cycles: int = 4  # 8-stage pipeline (NoC design 3.2)
    dma_desc_cycles: int = 2  # T_desc (Global DMA 6.2)
    dma_issue_cycles: int = 1  # T_issue
    dma_completion_cycles: int = 1  # T_completion
    host_validate_cycles: int = 50  # package validate, 由后续规格冻结
    host_patch_cycles: int = 10  # descriptor patch
    doorbell_latency_cycles: int = 5
    firmware_fetch_cycles: int = 3
    firmware_validate_cycles: int = 5
    frame_bind_cycles: int = 8  # slot frame 3.2 FSM (8 states)

    def cycle_ns(self) -> float:
        """Length of one simulator cycle in nanoseconds."""
        return 1000.0 / self.clock_mhz

    def boa_peak_tops(self) -> float:
        """Peak BOA TOPS for INT8/BF16 GEMM.

        2 FLOP per MAC element.  4 OPA * (16*16) MACs/cycle * 2 * clock.
        """
        macs_per_cycle = self.boa_num_opa * self.boa_opa_rows * self.boa_opa_cols
        flops_per_cycle = macs_per_cycle * 2 * self.boa_clock_multiplier
        return flops_per_cycle * (self.clock_mhz * 1e6) / 1e12

    def evu_peak_gflops(self) -> float:
        """Peak EVU GFLOP/s (32-lane vector FMA = 2 ops/lane/cycle)."""
        flops_per_cycle = self.evu_lanes * 2 * self.evu_clock_multiplier
        return flops_per_cycle * (self.clock_mhz * 1e6) / 1e9

    def with_overrides(self, **kw) -> HardwareConfig:
        return replace(self, **kw)


@dataclass(frozen=True)
class WorkloadConfig:
    """Per-workload shape parameters consumed by the workload builders."""

    name: str = "matmul"
    m: int = 512
    n: int = 512
    k: int = 512
    batch: int = 1
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 256
    dtype_bytes: int = 2  # BF16
    head_dim: int = 64  # attention
    num_heads: int = 8  # attention
    seq_len: int = 2048  # attention
    num_experts: int = 8  # MoE
    tokens_per_batch: int = 1024
    expert_ffn_dim: int = 512
    block_count: int = 4  # pipeline blocks (task loop iterations)

    def with_overrides(self, **kw) -> WorkloadConfig:
        return replace(self, **kw)


@dataclass(frozen=True)
class SimConfig:
    """Simulator run controls."""

    max_cycles: int = 2_000_000
    trace: bool = False
    trace_tile: int | None = None
    trace_json: str | None = None  # write Perfetto/Chrome trace.json
    trace_html: str | None = None  # write standalone trace.html
    report_path: str | None = None
    seed: int = 0
    fidelity: str = "timing_only"  # "timing_only" | "runtime" | "full_memory"

    def with_overrides(self, **kw) -> SimConfig:
        return replace(self, **kw)
