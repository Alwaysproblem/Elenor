"""Cycle-accurate ELENOR pipeline simulator.

Drives one Tile Group (1 Tile Group Sequencer + 4 Compute Tiles) cycle by
cycle until the TileGroupTask completes or the cycle cap is hit.  Collects
PMU fingerprints and validates credit invariants on every cycle.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

from .config import HardwareConfig, SimConfig
from .ir import TileGroupTask
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
        self.group = TileGroup(hw, self.tracer, fidelity=sim.fidelity)
        self.cycle = 0
        self._trace: list = []
        # persistent program identity registry: name -> stable program_id.
        # Survives across run() calls so warm launch works across tasks.
        self._program_name_registry: dict[str, int] = {}
        self._next_program_id: int = 1

    def run(self, task: TileGroupTask) -> SimResult:
        self._assign_program_ids(task)
        self.group.load_task(task)
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
                if self.group.sequencer.faulted:
                    completed = False
                    reason = (f"faulted: "
                              f"{self.group.sequencer.fault_reason}")
                else:
                    completed = True
                    reason = "group task complete"
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

    def _assign_program_ids(self, task: TileGroupTask) -> None:
        """Assign deterministic program_id + content-derived program_hash to
        every TileProgram in the task's role bindings.

        program_id: derived from program name via a stable registry so the
          same name maps to the same id across runs (PYTHONHASHSEED-safe).
        program_hash: crc32 of the canonical program content (instructions +
          descriptor templates), so a program that changes instructions or
          descriptors gets a different hash and triggers cold re-install.

        Only assigns if program_id == 0 (unassigned); existing builders that
        set their own ids are respected.
        """
        for binding in task.role_bindings.values():
            prog = binding.tile_program
            if prog.program_id == 0:
                if prog.name not in self._program_name_registry:
                    self._program_name_registry[prog.name] = (
                        self._next_program_id)
                    self._next_program_id += 1
                prog.program_id = self._program_name_registry[prog.name]
            if prog.program_hash == 0:
                # canonical content: instructions + descriptor templates,
                # sorted by descriptor name so dict insertion order doesn't
                # perturb the hash (warm reuse must be order-independent).
                parts = [f"{ins.op.value}|{ins.args}|{ins.dst}"
                         for ins in prog.insts]
                parts += [
                    f"{name}|{d.kind}|{d.op}|{sorted(d.params.items())}"
                    for name in sorted(prog.descriptors)
                    for d in [prog.descriptors[name]]]
                prog.program_hash = zlib.crc32(
                    "\n".join(parts).encode()) & 0xFFFFFFFF
