"""ELENOR runtime pipeline efficiency validator.

A cycle-accurate functional simulator of one ELENOR Tile Group
(1 Tile Group Sequencer + 4 Compute Tiles), grounded in the design/ specs.
Validates that the Graph -> Group Task -> Tile-SPMD role -> Engine control
flow, Stream Queue producer-consumer pipeline, and BOA/EVU/MFE/USE engine
partition produce the performance/PMU fingerprints the architecture predicts.
"""

from __future__ import annotations

from .cli import main  # noqa: F401  # exported via __all__
from .config import HardwareConfig, SimConfig, WorkloadConfig
from .engines import BOAEngine, EngineState, EVUEngine, MFEEngine, USEEngine
from .ir import (
    DMA_DESC,
    EngineDesc,
    GroupAction,
    GroupActionOp,
    StreamDesc,
    TileGroupTask,
    TileInst,
    TileProgram,
    TileRoleBinding,
    make_attention_task,
    make_conv_relu_task,
    make_conv_relu_tile_program,
    make_identity_tile_program,
    make_matmul_task,
    make_matmul_tile_program,
    make_moe_task,
    make_paged_attention_task,
    make_paged_attention_tile_program,
    make_pow_tile_program,
    make_relu_tile_program,
    make_stream_pipeline_tile_program,
    make_tiled_matmul_persistent_task,
    make_tiled_matmul_persistent_tile_program,
    make_tiled_matmul_pipelined_pow_task,
    make_tiled_matmul_pipelined_task,
    make_tiled_matmul_task,
    make_tiled_matmul_tile_program,
)
from .memory import L2SRAM, NoCRouter, PayloadTracker, SlotFrame
from .package import ElenorPackage
from .pmu import PMUCounter, StallReason
from .report import WorkloadReport  # noqa: F401  # exported via __all__
from .runtime import (
    DeviceRuntime,
    EventStatus,
    EventTable,
    FaultCode,
    FaultRecord,
    FaultRing,
    FirmwareRuntime,
    HostRuntime,
    KernelDriver,
    ProgramResidencyManager,
    ResetDomain,
)
from .simulator import Simulator
from .stream_queue import EOSPolicy, QueueKind, StreamQueue, StreamToken, TokenFlags
from .tile import ComputeTile, TileTaskState, TileUCE
from .tile_group import TileGroup
from .tile_group_sequencer import TileGroupSequencer
from .trace import Tracer, trace_to_html
from .workloads import (
    AttentionWorkload,
    ConvReLuWorkload,
    MatmulWorkload,
    MoEWorkload,
    PagedAttentionWorkload,
    TiledMatmulPersistentWorkload,
    TiledMatmulPipelinedPowWorkload,
    TiledMatmulPipelinedWorkload,
    TiledMatmulWorkload,
    Workload,
)

__all__ = [
    "DMA_DESC",
    "L2SRAM",
    "AttentionWorkload",
    "BOAEngine",
    "ComputeTile",
    "ConvReLuWorkload",
    "DeviceRuntime",
    "EOSPolicy",
    "EVUEngine",
    "ElenorPackage",
    "EngineDesc",
    "EngineState",
    "EventStatus",
    "EventTable",
    "FaultCode",
    "FaultRecord",
    "FaultRing",
    "FirmwareRuntime",
    "GroupAction",
    "GroupActionOp",
    "HardwareConfig",
    "HostRuntime",
    "KernelDriver",
    "MFEEngine",
    "MatmulWorkload",
    "MoEWorkload",
    "NoCRouter",
    "PMUCounter",
    "PagedAttentionWorkload",
    "PayloadTracker",
    "ProgramResidencyManager",
    "QueueKind",
    "ResetDomain",
    "SimConfig",
    "Simulator",
    "SlotFrame",
    "StallReason",
    "StreamDesc",
    "StreamQueue",
    "StreamToken",
    "TileGroup",
    "TileGroupSequencer",
    "TileGroupTask",
    "TileInst",
    "TileProgram",
    "TileRoleBinding",
    "TileTaskState",
    "TileUCE",
    "TiledMatmulPersistentWorkload",
    "TiledMatmulPipelinedPowWorkload",
    "TiledMatmulPipelinedWorkload",
    "TiledMatmulWorkload",
    "TokenFlags",
    "Tracer",
    "USEEngine",
    "Workload",
    "WorkloadConfig",
    "make_attention_task",
    "make_conv_relu_task",
    "make_conv_relu_tile_program",
    "make_identity_tile_program",
    "make_matmul_task",
    "make_matmul_tile_program",
    "make_moe_task",
    "make_paged_attention_task",
    "make_paged_attention_tile_program",
    "make_pow_tile_program",
    "make_relu_tile_program",
    "make_stream_pipeline_tile_program",
    "make_tiled_matmul_persistent_task",
    "make_tiled_matmul_persistent_tile_program",
    "make_tiled_matmul_pipelined_pow_task",
    "make_tiled_matmul_pipelined_task",
    "make_tiled_matmul_task",
    "make_tiled_matmul_tile_program",
    "trace_to_html",
]
