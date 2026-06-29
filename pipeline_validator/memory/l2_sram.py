"""Group L2 SRAM — capacity + bank occupancy (NoC design 3.4).

Models the 8 MB Group SRAM (Balanced-small profile) with bank-level
occupancy tracking.  Group DMA prefetch must pass a capacity gate; bank
conflicts serialize accesses on the same bank.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass


@dataclass
class L2Slot:
  """One allocated slot in Group L2 SRAM."""
  name: str
  base: int
  size: int
  bank: int
  bank_policy: str = "DEFAULT"


@dataclass
class L2SRAM:
  """Group L2 SRAM with capacity gate + bank occupancy."""
  capacity_bytes: int = 8 * 1024 * 1024
  banks: int = 16
  bank_bandwidth_gbs: float = 12.8

  def __post_init__(self) -> None:
    self._slots: dict[str, L2Slot] = {}
    self._used: int = 0
    self._bank_busy_until: list[int] = [0] * self.banks
    self.pmu_bank_conflict_cycles: int = 0
    self.pmu_capacity_fault_count: int = 0

  def capacity_ok(self, size: int) -> bool:
    return self._used + size <= self.capacity_bytes

  def alloc_slot(self, name: str, size: int,
                 bank_policy: str = "DEFAULT") -> L2Slot | None:
    """Allocate an L2 slot.  Returns None if capacity exhausted (fault)."""
    if not self.capacity_ok(size):
      self.pmu_capacity_fault_count += 1
      return None
    bank = self._pick_bank(name)
    slot = L2Slot(name=name, base=self._used, size=size, bank=bank,
                  bank_policy=bank_policy)
    self._slots[name] = slot
    self._used += size
    return slot

  def free_slot(self, name: str) -> None:
    slot = self._slots.pop(name, None)
    if slot is not None:
      self._used -= slot.size

  def _pick_bank(self, name: str) -> int:
    """Stable bank assignment: crc32(name) % banks (PYTHONHASHSEED-safe)."""
    return zlib.crc32(name.encode()) % self.banks

  def bank_for_addr(self, addr: int, slot_size: int = 64) -> int:
    """Bank-interleave by address (default 64B cache line)."""
    return (addr // slot_size) % self.banks

  def access_latency(self, bytes_total: int, bank: int, cycle: int,
                     clock_hz: float) -> int:
    """Return latency for a bank access, serializing on bank contention."""
    bw = self.bank_bandwidth_gbs * 1e9 / clock_hz
    base_lat = int(max((bytes_total + bw - 1) // bw, 1))
    free = self._bank_busy_until[bank]
    if cycle < free:
      # bank conflict: serialized behind prior access on same bank
      self.pmu_bank_conflict_cycles += (free - cycle)
      start = free
    else:
      start = cycle
    finish = start + base_lat
    self._bank_busy_until[bank] = finish
    return finish - cycle

  def reset(self) -> None:
    self._slots.clear()
    self._used = 0
    self._bank_busy_until = [0] * self.banks
    self.pmu_bank_conflict_cycles = 0
    self.pmu_capacity_fault_count = 0

  def snapshot(self) -> dict:
    return {
      "used_bytes": self._used,
      "capacity_bytes": self.capacity_bytes,
      "occupancy_ratio": self._used / self.capacity_bytes if self.capacity_bytes else 0,
      "slots": len(self._slots),
      "bank_conflict_cycles": self.pmu_bank_conflict_cycles,
      "capacity_faults": self.pmu_capacity_fault_count,
    }
