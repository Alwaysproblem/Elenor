"""Cycle-accurate ELENOR pipeline simulator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from .compiled_program import LoadedProgram
from .config import HardwareConfig, SimConfig
from .device import CpuDeviceController
from .execution_ir import ExecDeviceOp, ExecModel, ExecTileGroupTask, GlobalBinding
from .immutable import canonical_value
from .loader import load_program
from .pmu import PMUCounter
from .runtime.group_port import GroupPortAdapter
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
  device_snapshot: dict = field(default_factory=dict)
  trace: list = field(default_factory=list)
  credit_invariant_ok: bool = True
  tracer: Tracer | None = None
  memory_trace: bool = False
  slot_count: int = 1
  input_bindings: dict[str, GlobalBinding] = field(default_factory=dict)
  configuration: dict = field(default_factory=dict)

  def utilization(self, num_tiles: int = 4) -> float:
    return self.pmu.utilization(self.cycles * num_tiles)


class Simulator:
  def __init__(self, hw: HardwareConfig, sim: SimConfig, enable_tracer: bool = False, *, byte_store=None):
    self.hw = hw
    self.sim = sim
    self.tracer = Tracer(hw) if enable_tracer else None
    # CPU outstanding launches and physical Tile contexts are independent.
    self.group = TileGroup(
      hw,
      self.tracer,
      fidelity=sim.fidelity,
      context_count=sim.context_count,
      memory_trace=sim.memory_trace,
      scheduler_config=sim.group,
      byte_store=byte_store,
    )
    self.cycle = 0
    self._trace: list = []

  def _ensure_fault_drain(self, reason: str, cycle: int) -> None:
    """Begin reset/drain once for a sequencer/runtime fault."""
    rd = self.group.reset_domain
    if rd.is_active or rd.is_done:
      return
    self.group.trigger_fault(self.group._fault_code_for_reason(reason), cycle=cycle, desc_id=reason)

  def _observe_device_event(self, event: str, cycle: int, args: Mapping[str, object]) -> None:
    if self.tracer is None:
      return
    lane = {
      "launch_submit": "Controller",
      "launch_dependencies_ready": "Pending",
      "launch_admission": "Pending",
      "launch_completion": "Completion",
      "await_complete": "Controller",
      "return": "Controller",
      "fault": "Controller",
    }.get(event, "Controller")
    payload = dict(args)
    if event == "launch_completion":
      start_cycle = payload.get("submit_cycle")
      request_id = payload.get("request_id")
      context = payload.get("context")
      if isinstance(start_cycle, int):
        self.tracer.complete(
          "CPU Device", "Completion", f"launch:{request_id}:{context}", start_cycle, cycle, args=payload
        )
    self.tracer.instant("CPU Device", lane, event, cycle, payload)

  def run(self, program: LoadedProgram) -> SimResult:
    if not isinstance(program, LoadedProgram):
      raise ValueError("Simulator.run requires LoadedProgram; compile and load explicitly")
    checked = load_program(program.compiled, self.hw, self.sim, actual_bindings=program.actual_bindings)
    if checked.target_hash != program.target_hash:
      raise ValueError("LoadedProgram target fingerprint mismatch")
    compiled = program.compiled
    bindings = program.actual_bindings
    controller = self.group.profile_controller
    init_cycle = 0
    while not controller.initialized:
      if init_cycle >= self.hw.memory_target.profile_command_timeout_cycles:
        raise ValueError("profile initialization proof timed out")
      controller.step(init_cycle)
      init_cycle += 1
    expected = compiled.entry_profiles
    if (
      controller.active_modes["l1"] != expected.l1_mode or controller.active_modes["l2"] != expected.l2_mode
    ):
      raise ValueError("actual entry profile differs from compiled entry; explicit recovery required")
    self.group.begin_launch(compiled, bindings)
    self.cycle = 0
    self._trace.clear()
    if compiled.entry_kind == "model":
      if not isinstance(compiled.entry, ExecModel):
        raise ValueError("model artifact has wrong entry type")
      return self._run_model(
        replace(compiled.entry, body=compiled.entry_prefix + compiled.entry.body), bindings
      )
    task = compiled.entry
    if not isinstance(task, ExecTileGroupTask):
      raise ValueError("standalone artifact has wrong entry type")
    if compiled.entry_prefix:
      reason = self._execute_prefix(compiled.entry_prefix, bindings)
      if reason is not None:
        return SimResult(
          cycles=self.cycle,
          reason=reason,
          group_snapshot=self.group.snapshot(),
          tracer=self.tracer,
          configuration=self._configuration(bindings),
        )
    self.group.load_task(task, input_bindings=bindings, cycle=self.cycle)
    completed = False
    reason = ""
    fault_reason: str | None = None
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

      if self.group.sequencer.faulted and fault_reason is None:
        fault_reason = self.group.sequencer.fault_reason
        self._ensure_fault_drain(fault_reason, self.cycle)
      if fault_reason is not None:
        # Fault result is returned only after drain/reset cleanup completed
        # in runtime/full_memory. timing_only has no reset domain.
        if self.group.reset_domain.is_done:
          completed = False
          reason = f"faulted: {fault_reason}"
          break
        self.cycle += 1
        continue

      if done:
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
      memory_trace=self.sim.memory_trace,
      input_bindings=dict(bindings),
      configuration=self._configuration(bindings),
    )

  def _execute_prefix(
    self, instructions: tuple[ExecDeviceOp, ...], bindings: Mapping[str, GlobalBinding]
  ) -> str | None:
    model = ExecModel(
      name="entry_prefix", body=(*instructions, ExecDeviceOp("return", instruction_id="entry:return"))
    )
    port = GroupPortAdapter(self.group, self.sim.group.active_context_capacity)
    cpu = CpuDeviceController(model, self.sim.device, self.sim.device_context_count, port, bindings)
    while self.cycle < self.sim.max_cycles:
      cpu.step(self.cycle)
      self.group.step(self.cycle)
      cpu.harvest_completions(self.cycle)
      self.cycle += 1
      if cpu.faulted:
        return f"faulted: {cpu.fault_reason}"
      if cpu.succeeded:
        return None
    return f"cycle cap {self.sim.max_cycles} reached during entry configuration"

  def _run_model(self, model: ExecModel, bindings: Mapping[str, GlobalBinding]) -> SimResult:
    # A fresh adapter owns Group launch slots and sequencer identities.  The
    # CPU sees only the DevicePort protocol and stable request IDs.
    port = GroupPortAdapter(self.group, self.sim.group.active_context_capacity)
    controller = CpuDeviceController(
      model,
      self.sim.device,
      self.sim.device_context_count,
      port,
      bindings,
      observer=self._observe_device_event,
    )
    self.cycle = 0
    self._trace.clear()
    completed = False
    reason = ""
    credit_invariant_ok = True
    trace_tile = self.sim.trace_tile

    while self.cycle < self.sim.max_cycles:
      # Deterministic co-simulation order:
      #   CPU submit/dependency phase -> one Group cycle -> CPU completion
      #   harvest.  A completion from this Group step wakes deps next cycle.
      controller.step(self.cycle)
      self.group.step(self.cycle)
      if self.sim.trace and (trace_tile is None or trace_tile):
        self._trace.append({"cycle": self.cycle, **self.group.snapshot()})
      controller.harvest_completions(self.cycle)

      if controller.faulted:
        controller.note_fault_drain_started(self.cycle)
        self._ensure_fault_drain(controller.fault_reason or "device launch failed", self.cycle)

      if not self.group.credit_invariants_hold():
        credit_invariant_ok = False
        reason = f"credit invariant violated at cycle {self.cycle}"
        break

      if controller.faulted:
        if self.group.reset_domain.is_done:
          controller.note_fault_drain_completed(self.cycle)
          completed = False
          reason = f"faulted: {controller.fault_reason}"
          break
        self.cycle += 1
        continue

      if controller.succeeded:
        completed = True
        reason = "model complete"
        break
      self.cycle += 1
    else:
      reason = f"cycle cap {self.sim.max_cycles} reached"

    pmu = PMUCounter()
    pmu.merge(controller.pmu)
    pmu.merge(self.group.pmu)
    port_snapshot = port.snapshot()
    device_snapshot = controller.snapshot()
    port_records = {record["request_id"]: record for record in port_snapshot["request_records"]}
    for record in device_snapshot["launch_records"]:
      port_record = port_records.get(record["request_id"])
      if port_record is not None:
        record["active_cycle"] = port_record["active_cycle"]
      record["drain_cycle"] = controller.drain_start_cycle if record["status"] == "error" else None
    device_snapshot["port"] = port_snapshot
    credit_invariant_ok = credit_invariant_ok and self.group.credit_invariants_hold()
    return SimResult(
      cycles=self.cycle,
      completed=completed,
      reason=reason,
      pmu=pmu,
      group_snapshot=self.group.snapshot(),
      device_snapshot=device_snapshot,
      trace=self._trace,
      credit_invariant_ok=credit_invariant_ok,
      tracer=self.tracer,
      memory_trace=self.sim.memory_trace,
      slot_count=self.sim.device_context_count,
      input_bindings=dict(bindings),
      configuration=self._configuration(bindings),
    )

  def _configuration(self, bindings: Mapping[str, GlobalBinding]) -> dict:
    return {
      "hardware": canonical_value(self.hw),
      "simulation": canonical_value(self.sim),
      "bindings": canonical_value(bindings),
    }
