"""Cycle-accurate ELENOR pipeline simulator."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

from xdsl.dialects.builtin import ModuleOp

from .config import HardwareConfig, SimConfig
from .execution_ir import ExecTileGroupTask
from .ir_lowering import lower_workload_ir
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
    return self.pmu.utilization(self.cycles * num_tiles)


class Simulator:
  """Top-level simulator wrapping a TileGroup."""

  def __init__(self, hw: HardwareConfig, sim: SimConfig, enable_tracer: bool = False):
    self.hw = hw
    self.sim = sim
    self.tracer = Tracer(hw) if enable_tracer else None
    self.group = TileGroup(hw, self.tracer, fidelity=sim.fidelity, context_count=sim.context_count)
    self.cycle = 0
    self._trace: list = []
    self._program_name_registry: dict[str, int] = {}
    self._next_program_id: int = 1

  def run(self, module: ModuleOp) -> SimResult:
    task = lower_workload_ir(module)
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

      if not self.group.credit_invariants_hold():
        completed = False
        reason = f"credit invariant violated at cycle {self.cycle}"
        break

      if done:
        if self.group.sequencer.faulted:
          completed = False
          reason = f"faulted: {self.group.sequencer.fault_reason}"
        else:
          completed = True
          reason = "group task complete"
        break
      self.cycle += 1
    else:
      reason = f"cycle cap {self.sim.max_cycles} reached"

    return SimResult(
      cycles=self.cycle,
      completed=completed,
      reason=reason,
      pmu=self.group.pmu,
      group_snapshot=self.group.snapshot(),
      trace=self._trace,
      credit_invariant_ok=self.group.credit_invariants_hold(),
      tracer=self.tracer,
    )

  def reset(self) -> None:
    self.group.reset()
    self.cycle = 0
    self._trace.clear()

  def _assign_program_ids(self, task: ExecTileGroupTask) -> None:
    for binding in task.role_bindings.values():
      prog = binding.tile_program
      if prog.program_id == 0:
        if prog.name not in self._program_name_registry:
          self._program_name_registry[prog.name] = self._next_program_id
          self._next_program_id += 1
        prog.program_id = self._program_name_registry[prog.name]
      if prog.program_hash == 0:
        prog.program_hash = self._program_hash(prog)

  def _program_hash(self, prog) -> int:
    canonical = (
      prog.name,
      prog.version,
      tuple(
        (ins.op.value, ins.dst, tuple(self._tag_scalar(arg) for arg in ins.args)) for ins in prog.insts
      ),
      tuple(sorted(prog.labels.items())),
      tuple(
        (
          name,
          prog.descriptors[name].kind,
          prog.descriptors[name].op,
          tuple(
            (key, self._tag_scalar(value)) for key, value in sorted(prog.descriptors[name].params.items())
          ),
        )
        for name in sorted(prog.descriptors)
      ),
    )
    return zlib.crc32(repr(canonical).encode()) & 0xFFFFFFFF

  @staticmethod
  def _tag_scalar(value):
    if isinstance(value, bool):
      return ("bool", value)
    if isinstance(value, int):
      return ("int", value)
    if isinstance(value, float):
      return ("float", value)
    if isinstance(value, str):
      return ("str", value)
    return (type(value).__name__, value)
