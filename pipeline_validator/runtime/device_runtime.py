"""Device Runtime — graph schedule entry to group task dispatch
(Architecture doc 15, 16).

The Device Runtime consumes the graph schedule from the executable package
and dispatches TileGroupTasks.  In the existing validator, the simulator
calls `TileGroup.load_task()` directly; in runtime fidelity, the Device
Runtime adds a graph-schedule lookup layer with its own latency.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import HardwareConfig


@dataclass
class DeviceRuntime:
  """Device Runtime: graph schedule -> group task dispatch."""
  cfg: HardwareConfig

  def __post_init__(self) -> None:
    self.pmu_schedule_lookup_cycles: int = 0

  def lookup_graph_entry(self, cycle: int) -> int:
    """Architecture 15: parse graph schedule entry.  Returns cycles."""
    self.pmu_schedule_lookup_cycles += 1
    return 1

  def dispatch_group_task(self, cycle: int) -> int:
    """Dispatch a TileGroupTask to the Tile Group Sequencer."""
    self.pmu_schedule_lookup_cycles += 1
    return 1

  def reset(self) -> None:
    self.pmu_schedule_lookup_cycles = 0

  def snapshot(self) -> dict:
    return {
      "schedule_lookup_cycles": self.pmu_schedule_lookup_cycles,
    }
