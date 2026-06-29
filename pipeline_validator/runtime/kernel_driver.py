"""Kernel Driver — context/queue/iova/interrupt (Driver-Firmware 2.2, 3.1).

Models the kernel driver's software overhead: context creation, memory
pin/map, command queue + doorbell, interrupt delivery.  These are the
control-plane latencies between host submit and firmware fetch.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import HardwareConfig


@dataclass
class KernelDriver:
  """Kernel driver: context lifecycle + doorbell + interrupt."""
  cfg: HardwareConfig

  def __post_init__(self) -> None:
    self.pmu_doorbell_cycles: int = 0
    self.pmu_interrupt_cycles: int = 0

  def create_context(self) -> int:
    """Driver-Firmware 3.1: Context lifecycle Uncreated -> Created."""
    return 2

  def submit(self, cycle: int) -> int:
    """Driver-Firmware 3.2: PublishRing -> RingDoorbell.

    Returns the doorbell latency (cycles before firmware fetches).
    """
    self.pmu_doorbell_cycles += self.cfg.doorbell_latency_cycles
    return self.cfg.doorbell_latency_cycles

  def deliver_interrupt(self, cycle: int) -> int:
    """Deliver a completion/fault interrupt to the host."""
    self.pmu_interrupt_cycles += self.cfg.doorbell_latency_cycles
    return self.cfg.doorbell_latency_cycles

  def reset(self) -> None:
    self.pmu_doorbell_cycles = 0
    self.pmu_interrupt_cycles = 0

  def snapshot(self) -> dict:
    return {
      "doorbell_cycles": self.pmu_doorbell_cycles,
      "interrupt_cycles": self.pmu_interrupt_cycles,
    }
