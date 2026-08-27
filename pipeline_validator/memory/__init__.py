"""Memory hierarchy models for the runtime-level simulator."""

from __future__ import annotations

from .allocator import (
  AdmissionFailure,
  AllocationHandle,
  AllocationPlan,
  AllocationRequest,
  AllocationState,
  BankedFreeExtentAllocator,
  BankSegment,
  ContextBufferOwner,
  ExternalOwner,
  MemoryInvariantError,
  MemoryOwner,
  TaskBufferOwner,
)
from .hbm_region import HBMRegion
from .l1_slot_frame import FrameState, Slot, SlotFrame, SlotLifetime, SlotRole
from .l2_sram import L2SRAM
from .noc import NoCRouter, VirtualChannel
from .payload import Payload, PayloadTracker
from .transfer import (
  MemoryTransaction,
  ResolvedMemoryView,
  StageRequest,
  StageResult,
  StageWait,
  StageWaitReason,
  TransferLeg,
  TransferLegKind,
  TransferManager,
  TransferOp,
  TransferStage,
  TransferStatus,
)

__all__ = [
  "L2SRAM",
  "AdmissionFailure",
  "AllocationHandle",
  "AllocationPlan",
  "AllocationRequest",
  "AllocationState",
  "BankSegment",
  "BankedFreeExtentAllocator",
  "ContextBufferOwner",
  "ExternalOwner",
  "FrameState",
  "HBMRegion",
  "MemoryInvariantError",
  "MemoryOwner",
  "MemoryTransaction",
  "NoCRouter",
  "Payload",
  "PayloadTracker",
  "ResolvedMemoryView",
  "Slot",
  "SlotFrame",
  "SlotLifetime",
  "SlotRole",
  "StageRequest",
  "StageResult",
  "StageWait",
  "StageWaitReason",
  "TaskBufferOwner",
  "TransferLeg",
  "TransferLegKind",
  "TransferManager",
  "TransferOp",
  "TransferStage",
  "TransferStatus",
  "VirtualChannel",
]
