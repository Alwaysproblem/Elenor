"""Group L2 SRAM — banked free-extent allocation (NoC design 3.4).

Composes a ``BankedFreeExtentAllocator`` for the group SRAM.  The old
slot-name bump pointer and ``zlib.crc32`` bank picker are removed: every
allocation is an immutable handle with owner, generation and real bank
segments.  Capacity faults surface as ``AdmissionFailure`` from
``plan_bundle``, never as a successful transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .allocator import (
  AdmissionFailure,
  AllocationHandle,
  AllocationPlan,
  AllocationRequest,
  BankedFreeExtentAllocator,
  BankSegment,
  MemoryOwner,
)

if TYPE_CHECKING:
  from ..trace import MemoryTrace

@dataclass
class L2SRAM:
  """Group L2 SRAM backed by a ``BankedFreeExtentAllocator``."""
  capacity_bytes: int = 8 * 1024 * 1024
  banks: int = 16
  bank_bandwidth_gbs: float = 12.8
  trace: MemoryTrace | None = None

  def __post_init__(self) -> None:
    self._allocator = BankedFreeExtentAllocator(
      memory_space="l2",
      capacity_bytes=self.capacity_bytes,
      banks=self.banks,
      trace=self.trace,
    )

  @property
  def pool_version(self) -> int:
    """Live free-map version of the backing allocator (PR 3.5)."""
    return self._allocator.pool_version

  def plan_bundle(
    self, requests: list[AllocationRequest],
  ) -> AllocationPlan | AdmissionFailure:
    return self._allocator.plan_bundle(requests)

  def commit(self, plan: AllocationPlan, cycle: int) -> tuple[AllocationHandle, ...]:
    return self._allocator.commit(plan, cycle)

  def rollback(self, plan: AllocationPlan) -> None:
    self._allocator.rollback(plan)

  def assert_live(self, handle: AllocationHandle,
                  owner: MemoryOwner | None = None) -> None:
    self._allocator.assert_live(handle, owner)

  def pin(self, handle: AllocationHandle, consumer_id: str) -> None:
    self._allocator.pin(handle, consumer_id)

  def unpin(self, handle: AllocationHandle, consumer_id: str,
            cycle: int) -> bool:
    return self._allocator.unpin(handle, consumer_id, cycle)

  def request_release(self, handle: AllocationHandle, owner: MemoryOwner,
                      cycle: int) -> bool:
    return self._allocator.request_release(handle, owner, cycle)

  def resolve_segments(
    self, handle: AllocationHandle, offset_bytes: int, size_bytes: int,
  ) -> tuple[BankSegment, ...]:
    return self._allocator.resolve_segments(handle, offset_bytes, size_bytes)

  def is_released(self, handle: AllocationHandle) -> bool:
    return self._allocator.is_released(handle)

  def reset(self) -> None:
    self._allocator.reset()

  def snapshot(self) -> dict:
    return self._allocator.snapshot()
