"""Cycle-accurate ELENOR pipeline simulator."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

from xdsl.dialects.builtin import ModuleOp

from .config import HardwareConfig, SimConfig
from .dialects.elenor import NestContextOp, NexusProgramOp
from .execution_ir import ExecTileGroupTask
from .ir_lowering import lower_model_ir, lower_workload_ir
from .pmu import PMUCounter
from .tile_group import TileGroup
from .tile_group_sequencer import TileGroupSequencer
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
  slot_count: int = 1

  def utilization(self, num_tiles: int = 4) -> float:
    return self.pmu.utilization(self.cycles * num_tiles)


class Simulator:
  def __init__(self, hw: HardwareConfig, sim: SimConfig, enable_tracer: bool = False):
    self.hw = hw
    self.sim = sim
    self.tracer = Tracer(hw) if enable_tracer else None
    # Single TileGroup shared across all device execution slots.
    # context_count must be >= device_context_count so each slot gets
    # its own UCE context for true concurrency on the shared tiles.
    ctx_count = max(sim.context_count, sim.device_context_count)
    self.group = TileGroup(hw, self.tracer, fidelity=sim.fidelity,
                           context_count=ctx_count)
    self.groups = [self.group]  # legacy compat
    self.cycle = 0
    self._trace: list = []
    self._program_name_registry: dict[str, int] = {}
    self._next_program_id: int = 1

  def run(self, module: ModuleOp) -> SimResult:
    if any(isinstance(op, NexusProgramOp) for op in module.body.block.ops):
      return self._run_model(module)
    task = lower_workload_ir(module)
    self._assign_program_ids(task)
    # Legacy pin validation
    ctx_op = next(op for op in module.body.block.ops if isinstance(op, NestContextOp))
    pin = None if ctx_op.context_id is None else int(ctx_op.context_id.value.data)
    if pin is not None and pin >= self.sim.device_context_count:
      raise ValueError(
        f"context @{ctx_op.sym_name.data} pins device context {pin}"
        f" but device_context_count is {self.sim.device_context_count}")
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

  def _run_model(self, module: ModuleOp) -> SimResult:
    model = lower_model_ir(module)
    for task in model.tasks.values():
      self._assign_program_ids(task)
    count = self.sim.device_context_count
    for ctx_name, pin in model.context_pins.items():
      if pin is not None and pin >= count:
        raise ValueError(
          f"context @{ctx_name} pins device context {pin}"
          f" but device_context_count is {count}")
    # Reset the single shared TileGroup for a fresh model run
    self.group.reset()
    self.group._active_sequencers = []
    self.cycle = 0
    self._trace.clear()
    pmu = PMUCounter()
    slot_busy = [False] * count
    slot_seq: list[TileGroupSequencer | None] = [None] * count
    slot_tag: list[str | None] = [None] * count
    slot_ctx: list[str | None] = [None] * count
    slot_start = [0] * count
    done_events: set[str] = set()
    pc = 0
    returned = False
    completed = False
    reason = ""
    trace_tile = self.sim.trace_tile

    while self.cycle < self.sim.max_cycles:
      # 1. advance device PC: issue submits / pass awaits / return
      while pc < len(model.body):
        dop = model.body[pc]
        if dop.op == "submit":
          pin = model.context_pins.get(dop.ctx_name)
          if pin is not None:
            slot = pin if not slot_busy[pin] else None
          else:
            slot = next((i for i in range(count) if not slot_busy[i]), None)
          if slot is None:
            pmu.add_cycle("device_submit_wait", 1)
            break
          seq = self.group.load_context_task(
            model.tasks[dop.ctx_name], slot_index=slot)
          slot_busy[slot] = True
          slot_seq[slot] = seq
          slot_tag[slot] = dop.event_tag
          slot_ctx[slot] = dop.ctx_name
          slot_start[slot] = self.cycle
          if self.tracer is not None:
            self.tracer.instant("Device", f"Slot:{slot}", "context_submit",
                                self.cycle,
                                {"context": dop.ctx_name, "slot": slot,
                                 "pin": pin, "event": dop.event_tag,
                                 "cycle": self.cycle})
          pc += 1
        elif dop.op == "await":
          if dop.event_tag not in done_events:
            pmu.add_cycle("device_await_wait", 1)
            break
          pc += 1
        else:  # "return"
          returned = True
          pc += 1

      # 2. step the single shared TileGroup once
      self.group.step(self.cycle)
      if self.sim.trace and (trace_tile is None or trace_tile):
        self._trace.append({"cycle": self.cycle, **self.group.snapshot()})

      # 3. check which slots' sequencers completed this cycle
      fault = False
      for i in range(count):
        if not slot_busy[i]:
          continue
        done_seq = slot_seq[i]
        if done_seq is not None and not done_seq.done:
          continue
        if done_seq is not None and done_seq.faulted:
          reason = f"faulted: {done_seq.fault_reason}"
          fault = True
          break
        tag = slot_tag[i]
        assert tag is not None
        done_events.add(tag)
        if self.tracer is not None:
          self.tracer.complete("Device", f"Slot:{i}",
                               f"context:{slot_ctx[i]}:run",
                               slot_start[i], self.cycle,
                               args={"context": slot_ctx[i], "slot": i,
                                     "event": tag})
          self.tracer.instant("Device", f"Slot:{i}", "context_done",
                              self.cycle,
                              {"context": slot_ctx[i], "slot": i,
                               "event": tag, "cycle": self.cycle})
        slot_busy[i] = False
        slot_seq[i] = None
        slot_tag[i] = None
        slot_ctx[i] = None
      if fault:
        completed = False
        break
      if not self.group.credit_invariants_hold():
        completed = False
        reason = f"credit invariant violated at cycle {self.cycle}"
        break

      # 4. termination: return reached and every slot drained
      if returned and not any(slot_busy):
        completed = True
        reason = "model complete"
        break
      self.cycle += 1
    else:
      reason = f"cycle cap {self.sim.max_cycles} reached"

    # Merge the group PMU (accumulates all sequencer/tile/queue counters)
    pmu.merge(self.group.pmu)
    return SimResult(
      cycles=self.cycle, completed=completed, reason=reason, pmu=pmu,
      group_snapshot=self.group.snapshot(),
      trace=self._trace,
      credit_invariant_ok=self.group.credit_invariants_hold(),
      tracer=self.tracer, slot_count=count)
    for g in self.groups:
      g.reset()
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
