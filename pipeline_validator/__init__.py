"""ELENOR runtime pipeline efficiency validator.

A cycle-accurate functional simulator of one ELENOR Tile Group
(1 Region Sequencer + 4 Compute Tiles), grounded in the design/ specs.
Validates that the Graph -> Region -> Tile -> Engine control flow,
Stream Queue producer-consumer pipeline, and BOA/EVU/MFE/USE engine
partition produce the performance/PMU fingerprints the architecture predicts.
"""

from .cli import main  # noqa: F401  # exported via __all__
from .config import HardwareConfig, SimConfig, WorkloadConfig
from .engines import BOAEngine, EngineState, EVUEngine, MFEEngine, USEEngine
from .ir import (
    DMA_DESC,
    EngineDesc,
    RegionInst,
    RegionProgram,
    StreamDesc,
    TileInst,
    TileProgram,
    make_attention_region,
    make_conv_relu_region,
    make_conv_relu_tile_program,
    make_identity_tile_program,
    make_matmul_region,
    make_matmul_tile_program,
    make_moe_region,
    make_paged_attention_region,
    make_paged_attention_tile_program,
    make_relu_tile_program,
    make_stream_pipeline_tile_program,
    make_tiled_matmul_region,
    make_tiled_matmul_tile_program,
)
from .pmu import PMUCounter, StallReason
from .region import RegionSequencer
from .report import WorkloadReport  # noqa: F401  # exported via __all__
from .simulator import Simulator
from .stream_queue import EOSPolicy, QueueKind, StreamQueue, StreamToken, TokenFlags
from .tile import ComputeTile, TileUCE
from .tile_group import TileGroup
from .trace import Tracer, trace_to_html
from .workloads import (
    AttentionWorkload,
    ConvReLuWorkload,
    MatmulWorkload,
    MoEWorkload,
    PagedAttentionWorkload,
    Workload,
)

__all__ = [
    "DMA_DESC",
    "AttentionWorkload",
    "BOAEngine",
    "ComputeTile",
    "ConvReLuWorkload",
    "EOSPolicy",
    "EVUEngine",
    "EngineDesc",
    "EngineState",
    "HardwareConfig",
    "MFEEngine",
    "MatmulWorkload",
    "MoEWorkload",
    "PMUCounter",
    "PagedAttentionWorkload",
    "QueueKind",
    "RegionInst",
    "RegionProgram",
    "RegionSequencer",
    "SimConfig",
    "Simulator",
    "StallReason",
    "StreamDesc",
    "StreamQueue",
    "StreamToken",
    "TileGroup",
    "TileInst",
    "TileProgram",
    "TileUCE",
    "TiledMatmulWorkload",
    "TokenFlags",
    "Tracer",
    "USEEngine",
    "Workload",
    "WorkloadConfig",
    "make_attention_region",
    "make_conv_relu_region",
    "make_conv_relu_tile_program",
    "make_identity_tile_program",
    "make_matmul_region",
    "make_matmul_tile_program",
    "make_moe_region",
    "make_paged_attention_region",
    "make_paged_attention_tile_program",
    "make_relu_tile_program",
    "make_stream_pipeline_tile_program",
    "make_tiled_matmul_region",
    "make_tiled_matmul_tile_program",
    "trace_to_html",
]
