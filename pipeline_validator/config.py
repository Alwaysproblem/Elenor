"""Configuration dataclasses for the ELENOR pipeline validator.

All hardware parameters derive from the design/ specs and are tagged with the
exact doc section they come from.  Unfrozen values (the specs mark them with
`由后续规格冻结` / `由 SRAM profile 冻结` / `由 PPA exploration 冻结`) take the
First Silicon V1 recommended value so a baseline simulation is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

# Tile UCE execution contexts per tile.  Hardware V1.x caps at 2
# (design/elenor_tile_uce §3.2.1); the validator allows up to 8 for
# what-if exploration.
MAX_CONTEXT_COUNT = 8


# Explicit YAML path -> HardwareConfig field mapping for the grouped
# hardware_config.yaml schema.  Paths follow architecture ownership groups;
# leaf names drop prefixes already expressed by the parent path (e.g.
# engines.boa.launch_cycles, not engines.boa.boa_launch_cycles).  This table
# is the only adapter between the grouped YAML and the flat dataclass API;
# there is no string-concatenation fallback.
_HW_YAML_PATH_TO_FIELD = {
    "system.profile": "profile",
    "system.topology.tiles_per_group": "num_tiles",
    "system.clock.core_mhz": "clock_mhz",
    "memory.cache.line_bytes": "cache_line_bytes",
    "memory.hbm.capacity_bytes": "hbm_capacity_bytes",
    "memory.hbm.bandwidth_gbs": "hbm_bandwidth_gbs",
    "memory.hbm.outstanding_limit": "hbm_outstanding_limit",
    "memory.hbm.channels": "hbm_channels",
    "memory.hbm.fixed_latency_cycles": "hbm_fixed_latency_cycles",
    "memory.hbm.burst_bytes": "hbm_burst_bytes",
    "memory.group_sram.capacity_bytes": "group_sram_bytes",
    "memory.group_sram.banks": "group_sram_banks",
    "memory.group_sram.access_latency_cycles": "l2_access_latency_cycles",
    "memory.group_sram.bank_bandwidth_gbs": "l2_bank_bandwidth_gbs",
    "memory.group_sram.cache.capacity_bytes": "l2_cache_capacity_bytes",
    "memory.group_sram.cache.lookup_latency_cycles": "l2_cache_lookup_latency_cycles",
    "memory.group_sram.cache.mshr_entries": "l2_mshr_entries",
    "memory.tile_l1.capacity_bytes": "tile_l1_bytes",
    "memory.tile_l1.banks": "tile_l1_banks",
    "memory.tile_l1.access_latency_cycles": "l1_access_latency_cycles",
    "memory.tile_l1.bandwidth_gbs": "tile_l1_bandwidth_gbs",
    "memory.tile_l1.cache.capacity_bytes": "l1_cache_capacity_bytes",
    "memory.tile_l1.cache.lookup_latency_cycles": "l1_cache_lookup_latency_cycles",
    "memory.tile_l1.cache.mshr_entries": "l1_mshr_entries",
    "memory.tile_program_sram.capacity_bytes": "tile_program_sram_bytes",
    "engines.boa.opa.count": "boa_num_opa",
    "engines.boa.opa.rows": "boa_opa_rows",
    "engines.boa.opa.cols": "boa_opa_cols",
    "engines.boa.clock_multiplier": "boa_clock_multiplier",
    "engines.boa.dtype.input_bytes": "boa_dtype_bytes",
    "engines.boa.dtype.accumulator_bytes": "boa_acc_bytes",
    "engines.boa.launch_cycles": "boa_launch_cycles",
    "engines.evu.lanes": "evu_lanes",
    "engines.evu.clock_multiplier": "evu_clock_multiplier",
    "engines.evu.dtype_bytes": "evu_dtype_bytes",
    "engines.evu.launch_cycles": "evu_launch_cycles",
    "engines.mfe.bandwidth_gbs": "mfe_bandwidth_gbs",
    "engines.mfe.clock_multiplier": "mfe_clock_multiplier",
    "engines.mfe.launch_cycles": "mfe_launch_cycles",
    "engines.mfe.pipeline_depth": "mfe_pipeline_depth",
    "engines.mfe.channels.load": "mfe_load_channels",
    "engines.mfe.channels.store": "mfe_store_channels",
    "engines.mfe.command_queues.load_depth": "mfe_load_queue_depth",
    "engines.mfe.command_queues.store_depth": "mfe_store_queue_depth",
    "engines.mfe.stream_buffer_bytes": "mfe_stream_buffer_bytes",
    "engines.use.clock_mhz": "use_clock_mhz",
    "engines.use.state_cache_bytes": "use_state_cache_bytes",
    "engines.use.launch_cycles": "use_launch_cycles",
    "control.tile_uce.clock_mhz": "uce_clock_mhz",
    "control.tile_uce.dispatch_per_cycle": "uce_dispatch_per_cycle",
    "control.slot_frame.bind_cycles": "frame_bind_cycles",
    "fabric.dma.bandwidth_gbs": "group_dma_bandwidth_gbs",
    "fabric.dma.channels": "num_dma_channels",
    "fabric.dma.launch_cycles": "dma_launch_cycles",
    "fabric.dma.descriptor_cycles": "dma_desc_cycles",
    "fabric.dma.issue_cycles": "dma_issue_cycles",
    "fabric.dma.completion_cycles": "dma_completion_cycles",
    "fabric.noc.vc_depth": "noc_vc_depth",
    "fabric.noc.router_latency_cycles": "noc_router_latency_cycles",
    "fabric.stream_queue.default_depth": "stream_depth_default",
    "fabric.stream_queue.token_overhead_cycles": "stream_token_overhead_cycles",
    "fabric.stream_queue.fence_cycles": "stream_fence_cycles",
    "runtime.host.validate_cycles": "host_validate_cycles",
    "runtime.host.patch_cycles": "host_patch_cycles",
    "runtime.kernel_driver.doorbell_latency_cycles": "doorbell_latency_cycles",
    "runtime.firmware.fetch_cycles": "firmware_fetch_cycles",
    "runtime.firmware.validate_cycles": "firmware_validate_cycles",
}

# Every strict prefix of a mapped path is a group path and must map to a
# mapping in the YAML document.
_HW_YAML_GROUP_PATHS = {
    ".".join(parts[:index])
    for dotted_path in _HW_YAML_PATH_TO_FIELD
    for parts in [dotted_path.split(".")]
    for index in range(1, len(parts))
}
_HW_MAPPED_FIELDS = tuple(_HW_YAML_PATH_TO_FIELD.values())
if len(_HW_MAPPED_FIELDS) != len(set(_HW_MAPPED_FIELDS)):
    raise ValueError("multiple YAML paths map to one HardwareConfig field")
_HW_MAPPED_FIELD_NAMES = set(_HW_MAPPED_FIELDS)


class _StrictHwYamlLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate mapping keys.

    PyYAML's default silently lets a later duplicate key override the earlier
    value; a hardware configuration must fail loudly instead.
    """

    def construct_mapping(self, node: Any, deep: bool = False) -> Any:
        seen: set[Any] = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise ValueError(f"unhashable YAML key {key!r}") from exc
            if key in seen:
                raise ValueError(f"duplicate YAML key {key!r}")
            seen.add(key)
        return super().construct_mapping(node, deep)


def _flatten_hw_yaml(data: dict[Any, Any], path: Path) -> dict[str, Any]:
    """Recursively flatten the grouped YAML into flat field values.

    Group paths must be mappings, leaf paths must be scalars, and any unknown
    path is rejected with its full dotted name.
    """
    flat: dict[str, Any] = {}

    def walk(node: dict[Any, Any], dotted: str) -> None:
        for key, value in node.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"non-string key {key!r} under '{dotted or '<root>'}' in '{path}'"
                )
            child = f"{dotted}.{key}" if dotted else key
            if child in _HW_YAML_GROUP_PATHS:
                if not isinstance(value, dict):
                    raise ValueError(
                        f"group path '{child}' in '{path}' must be a mapping,"
                        f" got {type(value).__name__}"
                    )
                walk(value, child)
            elif child in _HW_YAML_PATH_TO_FIELD:
                if isinstance(value, (dict, list)):
                    raise ValueError(
                        f"leaf path '{child}' in '{path}' must be a scalar,"
                        f" got {type(value).__name__}"
                    )
                flat[_HW_YAML_PATH_TO_FIELD[child]] = value
            else:
                raise ValueError(f"unknown hardware config path '{child}' in '{path}'")

    walk(data, "")
    return flat


def _load_hw_yaml(path: Path, *, required: bool) -> dict[str, Any]:
    """Load grouped YAML and return flat HardwareConfig field values.

    ``required=True`` is the bundled defaults file: it must declare
    ``schema_version: 1`` and every mapped field.  ``required=False`` is a
    user override file: ``schema_version`` may be omitted and any legal
    subtree may be left out; missing fields fall back to bundled defaults.
    """
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictHwYamlLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in '{path}': {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"hardware config '{path}' must be a mapping, got {type(raw).__name__}"
        )

    if "schema_version" in raw:
        version = raw.pop("schema_version")
        if version != 1:
            raise ValueError(
                f"unsupported schema_version {version!r} in '{path}' (expected 1)"
            )
    elif required:
        raise ValueError(f"missing schema_version in '{path}' (expected 1)")

    flat = _flatten_hw_yaml(raw, path)
    if required:
        missing = sorted(_HW_MAPPED_FIELD_NAMES - flat.keys())
        if missing:
            raise ValueError(
                f"missing required HardwareConfig fields in '{path}': {missing}"
            )
    return flat


_DEFAULT_HW_CONFIG_PATH = Path(__file__).with_name("hardware_config.yaml")
_HW_DEFAULTS = _load_hw_yaml(_DEFAULT_HW_CONFIG_PATH, required=True)


@dataclass(frozen=True)
class HardwareConfig:
    """Static hardware parameters for one Tile Group + 4 Compute Tiles.

    Defaults follow the **Balanced-small** profile
    (design/ELENOR_Architecture_Design_v1.md section 12.3) which the specs call
    the realistic first-silicon configuration.
    """

    # --- Top-level -------------------------------------------------------
    profile: str = _HW_DEFAULTS["profile"]  # 12.3 config table
    num_tiles: int = _HW_DEFAULTS["num_tiles"]  # assignment scope (1 group, 4 tiles)

    # --- Clocking ---------------------------------------------------------
    # A single clock drives every cycle-accurate counter.  The specs do not
    # freeze a frequency; 1 GHz is the conventional modelling baseline.
    clock_mhz: float = _HW_DEFAULTS["clock_mhz"]

    # --- Group / L2 ------------------------------------------------------
    group_sram_bytes: int = _HW_DEFAULTS["group_sram_bytes"]  # 8 MB Balanced-small (12.3)
    group_sram_banks: int = _HW_DEFAULTS["group_sram_banks"]
    hbm_bandwidth_gbs: float = _HW_DEFAULTS["hbm_bandwidth_gbs"]  # 8 HBM stacks * 102.4 GB/s
    group_dma_bandwidth_gbs: float = _HW_DEFAULTS["group_dma_bandwidth_gbs"]  # per-channel peak
    num_dma_channels: int = _HW_DEFAULTS["num_dma_channels"]  # concurrent Global DMA channels

    # --- Tile / L1 -------------------------------------------------------
    tile_l1_bytes: int = _HW_DEFAULTS["tile_l1_bytes"]  # 1 MB Balanced-small (12.3)
    tile_l1_banks: int = _HW_DEFAULTS["tile_l1_banks"]  # 12.4 banking
    tile_l1_bandwidth_gbs: float = _HW_DEFAULTS["tile_l1_bandwidth_gbs"]  # per-tile read peak

    # --- BOA --------------------------------------------------------------
    # 6.1 / 8.x: 4 OPA per tile, 16x16 outer-product tile.
    boa_num_opa: int = _HW_DEFAULTS["boa_num_opa"]
    boa_opa_rows: int = _HW_DEFAULTS["boa_opa_rows"]
    boa_opa_cols: int = _HW_DEFAULTS["boa_opa_cols"]
    boa_clock_multiplier: float = _HW_DEFAULTS["boa_clock_multiplier"]  # BOA at core clock
    # Peak TOPS computed in boa_peak tops() from OPA geometry + dtype.
    boa_dtype_bytes: int = _HW_DEFAULTS["boa_dtype_bytes"]  # BF16 input
    boa_acc_bytes: int = _HW_DEFAULTS["boa_acc_bytes"]  # INT32 / BF16 accumulate

    # --- EVU --------------------------------------------------------------
    # 9.x: 32-lane predicated vector unit.
    evu_lanes: int = _HW_DEFAULTS["evu_lanes"]
    evu_clock_multiplier: float = _HW_DEFAULTS["evu_clock_multiplier"]
    evu_dtype_bytes: int = _HW_DEFAULTS["evu_dtype_bytes"]  # BF16

    # --- MFE --------------------------------------------------------------
    # 10.x: stream shaping engine.  Bandwidth limited by L1 write port.
    mfe_bandwidth_gbs: float = _HW_DEFAULTS["mfe_bandwidth_gbs"]  # page/segment stream into L1
    mfe_clock_multiplier: float = _HW_DEFAULTS["mfe_clock_multiplier"]

    # --- USE --------------------------------------------------------------
    # 11.x: state engine, modelled as a small RISC-V co-controller.
    use_clock_mhz: float = _HW_DEFAULTS["use_clock_mhz"]  # slower control core
    use_state_cache_bytes: int = _HW_DEFAULTS["use_state_cache_bytes"]  # 12.2 partition

    # --- Tile UCE ---------------------------------------------------------
    # 6.3: UCE + USE may share one tile-local RISC-V; UCE issues 1 inst/cycle.
    uce_clock_mhz: float = _HW_DEFAULTS["uce_clock_mhz"]
    uce_dispatch_per_cycle: int = _HW_DEFAULTS["uce_dispatch_per_cycle"]

    # --- Stream queue -----------------------------------------------------
    # Stream Queue design 6.1: depth=3 canonical trace from TileGroupTask.
    stream_depth_default: int = _HW_DEFAULTS["stream_depth_default"]
    stream_token_overhead_cycles: int = _HW_DEFAULTS["stream_token_overhead_cycles"]
    stream_fence_cycles: int = _HW_DEFAULTS["stream_fence_cycles"]  # payload visibility fence

    # --- Engine launch overhead -----------------------------------------
    # Cycles to decode a descriptor and enter an engine (not compute).
    boa_launch_cycles: int = _HW_DEFAULTS["boa_launch_cycles"]
    evu_launch_cycles: int = _HW_DEFAULTS["evu_launch_cycles"]
    mfe_launch_cycles: int = _HW_DEFAULTS["mfe_launch_cycles"]
    mfe_pipeline_depth: int = _HW_DEFAULTS["mfe_pipeline_depth"]  # per-channel accept queue
    mfe_load_channels: int = _HW_DEFAULTS["mfe_load_channels"]  # parallel load channels
    mfe_store_channels: int = _HW_DEFAULTS["mfe_store_channels"]  # parallel store channels
    mfe_load_queue_depth: int = _HW_DEFAULTS["mfe_load_queue_depth"]  # UCE→MFE load FIFO
    mfe_store_queue_depth: int = _HW_DEFAULTS["mfe_store_queue_depth"]  # UCE→MFE store FIFO
    # 0 = unfrozen / non-enforcing baseline (由 SRAM profile 冻结);
    # finite value enables page-stream prefetch-capacity validation.
    mfe_stream_buffer_bytes: int = _HW_DEFAULTS["mfe_stream_buffer_bytes"]
    use_launch_cycles: int = _HW_DEFAULTS["use_launch_cycles"]
    dma_launch_cycles: int = _HW_DEFAULTS["dma_launch_cycles"]
    # --- Runtime / memory (V2, runtime-level simulator) -----------------
    # All values below are unfrozen and tagged per the spec conventions:
    #   由后续规格冻结 / 由 SRAM profile 冻结 / 由 PPA exploration 冻结
    # Defaults follow the First Silicon V1 recommended values so a baseline
    # simulation is reproducible.
    hbm_capacity_bytes: int = _HW_DEFAULTS["hbm_capacity_bytes"]  # 16 GB, 由后续规格冻结
    hbm_outstanding_limit: int = _HW_DEFAULTS["hbm_outstanding_limit"]  # tag CAM depth
    hbm_channels: int = _HW_DEFAULTS["hbm_channels"]  # 8 HBM stacks, 由后续规格冻结
    hbm_fixed_latency_cycles: int = _HW_DEFAULTS["hbm_fixed_latency_cycles"]  # 由后续规格冻结
    hbm_burst_bytes: int = _HW_DEFAULTS["hbm_burst_bytes"]  # 2^N, 由后续规格冻结
    cache_line_bytes: int = _HW_DEFAULTS["cache_line_bytes"]
    l2_cache_capacity_bytes: int = _HW_DEFAULTS["l2_cache_capacity_bytes"]
    l2_cache_lookup_latency_cycles: int = _HW_DEFAULTS["l2_cache_lookup_latency_cycles"]
    l2_mshr_entries: int = _HW_DEFAULTS["l2_mshr_entries"]
    l1_cache_capacity_bytes: int = _HW_DEFAULTS["l1_cache_capacity_bytes"]
    l1_cache_lookup_latency_cycles: int = _HW_DEFAULTS["l1_cache_lookup_latency_cycles"]
    l1_mshr_entries: int = _HW_DEFAULTS["l1_mshr_entries"]
    l2_access_latency_cycles: int = _HW_DEFAULTS["l2_access_latency_cycles"]  # 由 SRAM profile 冻结
    l1_access_latency_cycles: int = _HW_DEFAULTS["l1_access_latency_cycles"]  # 由 SRAM profile 冻结
    l2_bank_bandwidth_gbs: float = _HW_DEFAULTS["l2_bank_bandwidth_gbs"]  # per-bank
    tile_program_sram_bytes: int = _HW_DEFAULTS["tile_program_sram_bytes"]  # hot tile kernel
    noc_vc_depth: int = _HW_DEFAULTS["noc_vc_depth"]  # 由 PPA exploration 冻结
    noc_router_latency_cycles: int = _HW_DEFAULTS["noc_router_latency_cycles"]  # NoC 3.2
    dma_desc_cycles: int = _HW_DEFAULTS["dma_desc_cycles"]  # T_desc (Global DMA 6.2)
    dma_issue_cycles: int = _HW_DEFAULTS["dma_issue_cycles"]  # T_issue
    dma_completion_cycles: int = _HW_DEFAULTS["dma_completion_cycles"]  # T_completion
    host_validate_cycles: int = _HW_DEFAULTS["host_validate_cycles"]  # package validate
    host_patch_cycles: int = _HW_DEFAULTS["host_patch_cycles"]  # descriptor patch
    doorbell_latency_cycles: int = _HW_DEFAULTS["doorbell_latency_cycles"]
    firmware_fetch_cycles: int = _HW_DEFAULTS["firmware_fetch_cycles"]
    firmware_validate_cycles: int = _HW_DEFAULTS["firmware_validate_cycles"]
    frame_bind_cycles: int = _HW_DEFAULTS["frame_bind_cycles"]  # slot frame 3.2 FSM (8 states)

    def __post_init__(self) -> None:
        if self.mfe_pipeline_depth < 1:
            raise ValueError("mfe_pipeline_depth must be >= 1")
        if self.mfe_load_channels < 1:
            raise ValueError("mfe_load_channels must be >= 1")
        if self.mfe_store_channels < 1:
            raise ValueError("mfe_store_channels must be >= 1")
        if self.mfe_load_queue_depth < 1:
            raise ValueError("mfe_load_queue_depth must be >= 1")
        if self.mfe_store_queue_depth < 1:
            raise ValueError("mfe_store_queue_depth must be >= 1")
        if self.mfe_stream_buffer_bytes < 0:
            raise ValueError("mfe_stream_buffer_bytes must be >= 0")
        if self.hbm_channels < 1:
            raise ValueError("hbm_channels must be >= 1")
        if self.hbm_fixed_latency_cycles < 0:
            raise ValueError("hbm_fixed_latency_cycles must be >= 0")
        if self.hbm_burst_bytes <= 0 or (self.hbm_burst_bytes &
                                         (self.hbm_burst_bytes - 1)) != 0:
            raise ValueError("hbm_burst_bytes must be a positive power of 2")
        if self.l2_access_latency_cycles < 1:
            raise ValueError("l2_access_latency_cycles must be >= 1")
        if self.l1_access_latency_cycles < 1:
            raise ValueError("l1_access_latency_cycles must be >= 1")
        if self.cache_line_bytes <= 0 or (
            self.cache_line_bytes & (self.cache_line_bytes - 1)
        ) != 0:
            raise ValueError("cache_line_bytes must be a positive power of 2")
        for field_name, capacity in (
            ("l2_cache_capacity_bytes", self.l2_cache_capacity_bytes),
            ("l1_cache_capacity_bytes", self.l1_cache_capacity_bytes),
        ):
            if capacity < self.cache_line_bytes or capacity % self.cache_line_bytes:
                raise ValueError(
                    f"{field_name} must be line-aligned and at least one cache line"
                )
        if self.l2_cache_lookup_latency_cycles <= 0:
            raise ValueError("l2_cache_lookup_latency_cycles must be > 0")
        if self.l1_cache_lookup_latency_cycles <= 0:
            raise ValueError("l1_cache_lookup_latency_cycles must be > 0")
        if self.l2_mshr_entries <= 0:
            raise ValueError("l2_mshr_entries must be > 0")
        if self.l1_mshr_entries <= 0:
            raise ValueError("l1_mshr_entries must be > 0")

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

    @classmethod
    def from_yaml(cls, path: str | Path) -> HardwareConfig:
        """Load a partial grouped hardware configuration.

        Omitted groups and fields keep the bundled defaults; the usual
        ``__post_init__`` validation applies to YAML values as well.
        """
        values = _load_hw_yaml(Path(path), required=False)
        return cls(**{**_HW_DEFAULTS, **values})


# Schema bijection guard: every HardwareConfig field is mapped from exactly
# one YAML path and vice versa (the many-paths-to-one-field direction is
# checked above via _HW_MAPPED_FIELDS).
_HW_FIELD_NAMES = {field.name for field in fields(HardwareConfig)}
if _HW_MAPPED_FIELD_NAMES != _HW_FIELD_NAMES:
    missing = sorted(_HW_FIELD_NAMES - _HW_MAPPED_FIELD_NAMES)
    extra = sorted(_HW_MAPPED_FIELD_NAMES - _HW_FIELD_NAMES)
    raise ValueError(f"grouped YAML schema mismatch: missing={missing}, extra={extra}")


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
    memory_trace: bool = False  # PR 5: emit memory lanes/counters/flows + report peaks
    trace_tile: int | None = None
    trace_json: str | None = None  # write Perfetto/Chrome trace.json
    trace_html: str | None = None  # write standalone trace.html
    report_path: str | None = None
    seed: int = 0
    fidelity: str = "full_memory"  # "timing_only" | "runtime" | "full_memory"
    context_count: int = 1
    device_context_count: int = 1

    def __post_init__(self) -> None:
        if self.fidelity not in ("timing_only", "runtime", "full_memory"):
            raise ValueError(
                "fidelity must be one of: timing_only, runtime, full_memory")
        if self.context_count < 1 or self.context_count > MAX_CONTEXT_COUNT:
            raise ValueError("context_count must be between 1 and 8")
        if self.device_context_count < 1 or self.device_context_count > MAX_CONTEXT_COUNT:
            raise ValueError("device_context_count must be between 1 and 8")

    def with_overrides(self, **kw) -> SimConfig:
        return replace(self, **kw)
