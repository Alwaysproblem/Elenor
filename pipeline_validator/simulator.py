"""Cycle-accurate ELENOR pipeline simulator."""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import pairwise

from xdsl.dialects.builtin import ModuleOp

from .config import HardwareConfig, SimConfig
from .device import CpuDeviceController
from .dialects.elenor import NestContextOp, NexusProgramOp
from .execution_ir import (
  ExecGatherDesc,
  ExecGlobalInput,
  ExecGroupActionOp,
  ExecTileGroupTask,
  GlobalBinding,
)
from .ir_lowering import lower_model_ir, lower_workload_ir
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


def _validate_input_bindings(
  inputs: list[ExecGlobalInput], bindings: Mapping[str, GlobalBinding], hw: HardwareConfig
) -> None:
  if bindings and not inputs:
    raise ValueError("input bindings provided but module declares no global inputs")

  inputs_by_name = {input_.name: input_ for input_ in inputs}
  for input_ in inputs:
    if input_.name not in bindings:
      raise ValueError(f"missing input binding for global '{input_.name}'")
  for name in bindings:
    if name not in inputs_by_name:
      raise ValueError(f"input binding '{name}' does not match any program input")
  for input_ in inputs:
    binding = bindings[input_.name]
    if binding.size_bytes < input_.size_bytes:
      raise ValueError(
        f"input binding '{binding.name}' size {binding.size_bytes} is smaller"
        f" than required {input_.size_bytes} bytes"
      )

  ordered = sorted(bindings.values(), key=lambda binding: binding.base_iova)
  for left, right in pairwise(ordered):
    if right.base_iova < left.base_iova + left.size_bytes:
      raise ValueError(f"input bindings '{left.name}' and '{right.name}' overlap")
  for binding in bindings.values():
    if binding.base_iova + binding.size_bytes > hw.hbm_capacity_bytes:
      raise ValueError(f"input binding '{binding.name}' exceeds HBM capacity")


def _validate_binding_permissions(
  tasks_with_maps: list[tuple[ExecTileGroupTask, Mapping[str, str]]], bindings: Mapping[str, GlobalBinding]
) -> None:
  for task, name_map in tasks_with_maps:
    for action in task.actions:
      if action.op not in (ExecGroupActionOp.DMA_PREFETCH, ExecGroupActionOp.DMA_STORE):
        continue
      transfer = action.args[1]
      if transfer.src.space == "global":
        formal_name = transfer.src.base.removeprefix("global:")
        input_name = name_map[formal_name]
        if "r" not in bindings[input_name].permissions:
          raise ValueError(f"input binding '{input_name}' is not readable but is used as prefetch source")
      if transfer.dst.space == "global":
        formal_name = transfer.dst.base.removeprefix("global:")
        input_name = name_map[formal_name]
        if "w" not in bindings[input_name].permissions:
          raise ValueError(f"input binding '{input_name}' is not writable but is used as store destination")

    for role_binding in task.role_bindings.values():
      for descriptor in role_binding.tile_program.descriptors.values():
        gather = descriptor.params.get("gather")
        if not isinstance(gather, ExecGatherDesc):
          continue
        if not gather.source.base.startswith("formal:"):
          raise ValueError("gather source must reference a global formal")
        try:
          formal_index = int(gather.source.base.removeprefix("formal:"))
        except ValueError as exc:
          raise ValueError("gather source formal index is invalid") from exc
        if (
          formal_index >= len(role_binding.tile_program.formals)
          or role_binding.tile_program.formals[formal_index].space != "global"
        ):
          raise ValueError("gather source formal index does not name a global formal")
        global_index = formal_index - 1
        if global_index < 0 or global_index >= len(role_binding.global_actuals):
          raise ValueError("gather source global actual index is out of range")
        global_actual = role_binding.global_actuals[global_index]
        if not global_actual.base.startswith("global:"):
          raise ValueError("gather global actual must reference a context formal")
        formal_name = global_actual.base.removeprefix("global:")
        gather_input_name = name_map.get(formal_name)
        if gather_input_name is None or gather_input_name not in bindings:
          raise ValueError(f"gather source mapping for global '{formal_name}' is missing")
        if "r" not in bindings[gather_input_name].permissions:
          raise ValueError(
            f"input binding '{gather_input_name}' is not readable but is used as gather source"
          )


class Simulator:
  def __init__(self, hw: HardwareConfig, sim: SimConfig, enable_tracer: bool = False):
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
    )
    self.cycle = 0
    self._trace: list = []
    self._program_name_registry: dict[str, int] = {}
    self._next_program_id: int = 1

  def _validate_tile_placement(self, task: ExecTileGroupTask) -> None:
    for binding in task.role_bindings.values():
      pin = binding.context_id
      if pin is not None and not 0 <= pin < self.sim.context_count:
        raise ValueError(
          f"dispatch '{binding.tile_program.name}' pins Tile context {pin}"
          f" outside context_count={self.sim.context_count}"
        )

  def _ensure_fault_drain(self, reason: str, cycle: int) -> None:
    """Begin reset/drain once for a sequencer/runtime fault."""
    if not self.group.runtime_enabled:
      return
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

  def run(self, module: ModuleOp, input_bindings: Mapping[str, GlobalBinding] | None = None) -> SimResult:
    bindings = {} if input_bindings is None else input_bindings
    if any(isinstance(op, NexusProgramOp) for op in module.body.block.ops):
      return self._run_model(module, bindings)
    task = lower_workload_ir(module)
    context = next(op for op in module.body.block.ops if isinstance(op, NestContextOp))
    if context.context_id is not None and int(context.context_id.value.data) != 0:
      raise ValueError("standalone nest.context must use Group context slot 0")
    self._validate_tile_placement(task)
    _validate_input_bindings(list(task.global_inputs), bindings, self.hw)
    _validate_binding_permissions(
      [(task, {input_.name: input_.name for input_ in task.global_inputs})], bindings
    )
    self._assign_program_ids(task)
    self.group.load_task(task, input_bindings=bindings)
    self.cycle = 0
    self._trace.clear()
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
        if not self.group.runtime_enabled or self.group.reset_domain.is_done:
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

  def _run_model(self, module: ModuleOp, bindings: Mapping[str, GlobalBinding]) -> SimResult:
    model = lower_model_ir(module)
    _validate_input_bindings(list(model.inputs), bindings, self.hw)
    tasks_with_maps: list[tuple[ExecTileGroupTask, Mapping[str, str]]] = []
    for device_op in model.body:
      if device_op.op != "submit":
        continue
      task = model.tasks[device_op.ctx_name]
      name_map = {
        formal.name: model.inputs[actual_index].name
        for formal, actual_index in zip(task.global_inputs, device_op.actual_inputs)
      }
      tasks_with_maps.append((task, name_map))
    _validate_binding_permissions(tasks_with_maps, bindings)
    for task in model.tasks.values():
      self._validate_tile_placement(task)
      self._assign_program_ids(task)
    for name, pin in model.context_pins.items():
      if pin is not None and not 0 <= pin < self.sim.group.active_context_capacity:
        raise ValueError(
          f"context '{name}' pins Group context slot {pin} outside"
          f" active_context_capacity={self.sim.group.active_context_capacity}"
        )

    # A fresh adapter owns Group launch slots and sequencer identities.  The
    # CPU sees only the DevicePort protocol and stable request IDs.
    self.group.reset()
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
        if not self.group.runtime_enabled or self.group.reset_domain.is_done:
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
      "hardware": asdict(self.hw),
      "simulation": asdict(self.sim),
      "bindings": {name: asdict(binding) for name, binding in bindings.items()},
    }

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
