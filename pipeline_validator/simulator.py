"""Cycle-accurate ELENOR pipeline simulator.

Drives one Tile Group (1 Region Sequencer + 4 Compute Tiles) cycle by cycle
until the Region Program completes or the cycle cap is hit.  Collects PMU
fingerprints and validates credit invariants on every cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import HardwareConfig, SimConfig
from .ir import RegionProgram
from .pmu import PMUCounter
from .tile_group import TileGroup
from .trace import Tracer


@dataclass
class SimResult:
    """Outcome of one simulation run."""
    cycles: int = 0
    completed: bool = False
    reason: str = ""
    pmu: PMUCounter = field(default_factory=PMUCounter)
    group_snapshot: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    credit_invariant_ok: bool = True
    tracer: Tracer | None = None

    def utilization(self, num_tiles: int = 4) -> float:
        """Fraction of tile-cycles not stalled.  Stall cycles are aggregated
        across all tiles, so denominator is cycles * num_tiles."""
        return self.pmu.utilization(self.cycles * num_tiles)


class Simulator:
    """Top-level simulator wrapping a TileGroup."""

    def __init__(self,
                 hw: HardwareConfig,
                 sim: SimConfig,
                 enable_tracer: bool = False):
        self.hw = hw
        self.sim = sim
        self.tracer = Tracer(hw) if enable_tracer else None
        self.group = TileGroup(hw, self.tracer)
        self.cycle = 0
        self._trace: list = []

    def run(self, program: RegionProgram) -> SimResult:
        self.group.load_region(program)
        self.cycle = 0
        self._trace.clear()
        completed = False
        reason = ""
        trace_tile = self.sim.trace_tile

        while self.cycle < self.sim.max_cycles:
            done = self.group.step(self.cycle)
            if self.sim.trace and (trace_tile is None or trace_tile):
                snap = self.group.snapshot()
                self._trace.append({"cycle": self.cycle, **snap})

            # validate credit invariants every cycle
            if not self.group.credit_invariants_hold():
                completed = False
                reason = f"credit invariant violated at cycle {self.cycle}"
                break

            if done:
                completed = True
                reason = "region complete"
                break
            self.cycle += 1
        else:
            reason = f"cycle cap {self.sim.max_cycles} reached"

        result = SimResult(
            cycles=self.cycle,
            completed=completed,
            reason=reason,
            pmu=self.group.pmu,
            group_snapshot=self.group.snapshot(),
            trace=self._trace,
            credit_invariant_ok=self.group.credit_invariants_hold(),
            tracer=self.tracer,
        )
        return result

    def reset(self) -> None:
        self.group.reset()
        self.cycle = 0
        self._trace.clear()
