"""ELENOR runtime pipeline efficiency validator."""

from __future__ import annotations

from .cli import main
from .config import HardwareConfig, SimConfig, WorkloadConfig
from .dialects import Elenor
from .dialects.elenor import (
  NestAllocOp,
  NestAwaitOp,
  NestBarrierOp,
  NestBuffer,
  NestCollectiveOp,
  NestContextOp,
  NestDispatchOp,
  NestDMAStoreOp,
  NestEvent,
  NestPrefetchOp,
  NestReleaseOp,
  NestReturnOp,
  NestTaskRangeOp,
  TaskRange,
  TileAwaitOp,
  TileBoaOp,
  TileEvent,
  TileEvuOp,
  TileLoadOp,
  TilePowOp,
  TileProgramDefOp,
  TileReturnOp,
  TileSignalOp,
  TileStoreOp,
)
from .engines import BOAEngine, EngineState, EVUEngine, MFEEngine, USEEngine
from .memory import L2SRAM, NoCRouter, PayloadTracker, SlotFrame
from .package import ElenorPackage
from .pmu import PMUCounter, StallReason
from .report import WorkloadReport  # noqa: F401
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
from .tile import ComputeTile, TileUCE
from .tile_group import TileGroup
from .tile_group_sequencer import TileGroupSequencer
from .trace import Tracer, trace_to_html
from .workload_builders import (
  make_identity_tile_program,
  make_pow_task,
  make_pow_tile_program,
)
from .workload_ir import (
  load_workload_ir,
  make_elenor_context,
  parse_workload_ir,
  print_workload_ir,
  verify_workload_ir,
)
from .workloads import (
  PowWorkload,
  Workload,
)

__all__ = [
  "L2SRAM",
  "BOAEngine",
  "ComputeTile",
  "DeviceRuntime",
  "EOSPolicy",
  "EVUEngine",
  "Elenor",
  "ElenorPackage",
  "EngineState",
  "EventStatus",
  "EventTable",
  "FaultCode",
  "FaultRecord",
  "FaultRing",
  "FirmwareRuntime",
  "HardwareConfig",
  "HostRuntime",
  "KernelDriver",
  "MFEEngine",
  "NestAllocOp",
  "NestAwaitOp",
  "NestBarrierOp",
  "NestBuffer",
  "NestCollectiveOp",
  "NestContextOp",
  "NestDMAStoreOp",
  "NestDispatchOp",
  "NestEvent",
  "NestPrefetchOp",
  "NestReleaseOp",
  "NestReturnOp",
  "NestTaskRangeOp",
  "NoCRouter",
  "PMUCounter",
  "PayloadTracker",
  "PowWorkload",
  "ProgramResidencyManager",
  "QueueKind",
  "ResetDomain",
  "SimConfig",
  "Simulator",
  "SlotFrame",
  "StallReason",
  "StreamQueue",
  "StreamToken",
  "TaskRange",
  "TileAwaitOp",
  "TileBoaOp",
  "TileEvent",
  "TileEvuOp",
  "TileGroup",
  "TileGroupSequencer",
  "TileLoadOp",
  "TilePowOp",
  "TileProgramDefOp",
  "TileReturnOp",
  "TileSignalOp",
  "TileStoreOp",
  "TileUCE",
  "TokenFlags",
  "Tracer",
  "USEEngine",
  "Workload",
  "WorkloadConfig",
  "load_workload_ir",
  "main",
  "make_elenor_context",
  "make_identity_tile_program",
  "make_pow_task",
  "make_pow_tile_program",
  "parse_workload_ir",
  "print_workload_ir",
  "trace_to_html",
  "verify_workload_ir",
]
