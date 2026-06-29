"""Firmware Runtime — command loop + event/fault orchestration
(Driver-Firmware 3.3, Runtime ABI 3.1).

Models the firmware command loop: fetch command, validate ABI/descriptor,
dispatch group task, update event table, write fault record on error.
Each command consumes fetch + validate cycles before dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import HardwareConfig
from .event_table import EventStatus, EventTable
from .fault_ring import FaultRing


@dataclass
class FirmwareRuntime:
  """Firmware command loop (Driver-Firmware 3.3)."""
  cfg: HardwareConfig
  event_table: EventTable
  fault_ring: FaultRing

  def __post_init__(self) -> None:
    self.pmu_fetch_cycles: int = 0
    self.pmu_validate_cycles: int = 0
    self.pmu_dispatch_cycles: int = 0

  def fetch_and_validate(self, cycle: int) -> int:
    """Driver-Firmware 3.3: fetch command + validate header/descriptor.

    Returns cycles consumed (fetch + validate).  Firmware validate must
    complete before dispatch — engines never start on invalid descriptors.
    """
    lat = self.cfg.firmware_fetch_cycles + self.cfg.firmware_validate_cycles
    self.pmu_fetch_cycles += self.cfg.firmware_fetch_cycles
    self.pmu_validate_cycles += self.cfg.firmware_validate_cycles
    return lat

  def dispatch_group_task(self, cycle: int) -> int:
    """Dispatch a validated command to the Device Runtime."""
    self.pmu_dispatch_cycles += 1
    return 1

  def signal_event(self, name: str, status: EventStatus, producer_id: int,
                   cycle: int) -> bool:
    """Signal an event through the event table (with sequence check)."""
    return self.event_table.signal(name, status, producer_id, cycle)

  def write_fault(self, rec) -> int:
    """Write a fault record and signal an error event."""
    idx = self.fault_ring.write(rec)
    return idx

  def reset(self) -> None:
    self.pmu_fetch_cycles = 0
    self.pmu_validate_cycles = 0
    self.pmu_dispatch_cycles = 0

  def snapshot(self) -> dict:
    return {
      "fetch_cycles": self.pmu_fetch_cycles,
      "validate_cycles": self.pmu_validate_cycles,
      "dispatch_cycles": self.pmu_dispatch_cycles,
    }
