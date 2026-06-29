"""HBM IOVA region (Global DMA 6.2, Executable Package 5.1).

Models HBM as a capacity + bandwidth + outstanding-limit resource.
Runtime allocates IOVA ranges for program/descriptor/weight/workspace
sections; Group DMA consumes bandwidth and outstanding slots.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HBMRegion:
  """HBM memory region with alloc/free + bandwidth + outstanding limit."""
  base_iova: int = 0
  size_bytes: int = 16 * 1024 * 1024 * 1024  # 16 GB default
  bandwidth_gbs: float = 819.2
  outstanding_limit: int = 32

  def __post_init__(self) -> None:
    self._allocated: dict[int, int] = {}
    self._next_iova: int = self.base_iova + 4096  # leave page 0
    self._outstanding: int = 0

  def alloc(self, size: int, alignment: int = 4096) -> int:
    """Allocate `size` bytes, return the base IOVA (simulates driver pin/map)."""
    if self._next_iova + size > self.base_iova + self.size_bytes:
      raise MemoryError("HBM region exhausted")
    # align
    iova = (self._next_iova + alignment - 1) & ~(alignment - 1)
    self._allocated[iova] = size
    self._next_iova = iova + size
    return iova

  def free(self, iova: int) -> None:
    self._allocated.pop(iova, None)

  def used_bytes(self) -> int:
    return sum(self._allocated.values())

  def bandwidth_bytes_per_cycle(self, clock_hz: float) -> float:
    return self.bandwidth_gbs * 1e9 / clock_hz

  def can_issue(self) -> bool:
    return self._outstanding < self.outstanding_limit

  def issue(self, bytes_total: int, clock_hz: float) -> int:
    """Issue a DMA transfer, return latency in cycles (T_read/T_write)."""
    if not self.can_issue():
      # stall until an outstanding request frees; model as extra cycle
      return -1
    self._outstanding += 1
    bw = self.bandwidth_bytes_per_cycle(clock_hz)
    return int(max((bytes_total + bw - 1) // bw, 1))

  def complete(self) -> None:
    if self._outstanding > 0:
      self._outstanding -= 1

  def reset(self) -> None:
    self._allocated.clear()
    self._next_iova = self.base_iova + 4096
    self._outstanding = 0

  def snapshot(self) -> dict:
    return {
      "used_bytes": self.used_bytes(),
      "capacity_bytes": self.size_bytes,
      "outstanding": self._outstanding,
      "limit": self.outstanding_limit,
    }
