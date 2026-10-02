"""CPU-side execution of lowered ``nexus.program`` models.

The controller intentionally depends only on execution DTOs, simulator
configuration, PMU accounting, and the message protocol declared here.  It
never observes TileGroup slots, sequencers, Tile contexts, or per-task PCs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from .config import DeviceConfig
from .execution_ir import ExecHostCall, ExecModel, ExecTileGroupTask, GlobalBinding
from .pmu import PMUCounter
from .profiles import MemoryMaintenanceDesc, ProfileReconfigDesc


class DeviceCompletionStatus(StrEnum):
  """Terminal status returned by a device port."""

  SUCCESS = "success"
  ERROR = "error"


@dataclass(frozen=True)
class DeviceLaunchRequest:
  """Immutable CPU-to-Group launch message.

  ``request_id`` identifies a launch instance, not a reusable hardware slot.
  The low-level task object is an already-loaded executable reference; the
  binding maps are immutable snapshots of the launch arguments.
  ``group_affinity`` optionally requests a Group context-table slot. It does
  not select a Group ID, Tile, or physical Tile execution context.
  """

  request_id: int
  task: ExecTileGroupTask
  context_name: str
  global_bindings: Mapping[str, GlobalBinding]
  formal_bindings: Mapping[str, str]
  group_affinity: int | None = None


@dataclass(frozen=True)
class DeviceCompletion:
  """Terminal Group-to-CPU completion message."""

  request_id: int
  status: DeviceCompletionStatus
  reason: str
  cycle: int


@dataclass(frozen=True)
class DeviceControlRequest:
  """Immutable CPU-to-Group profile or maintenance command."""

  command_id: str
  command: ProfileReconfigDesc | MemoryMaintenanceDesc


@dataclass(frozen=True)
class DeviceControlCompletion:
  """Terminal Group-to-CPU control completion."""

  command_id: str
  status: str
  reason: str
  cycle: int


@dataclass(frozen=True)
class DeviceHostRequest:
  """Immutable CPU-to-host routine invocation (plan §4).

  ``binding_names`` pairs one-to-one with ``command.accesses``; the CPU
  resolves entry-global formal indices to actual binding names so the
  host session never needs the ExecModel.
  """

  request_id: int
  name: str
  command: ExecHostCall
  binding_names: tuple[str, ...]


class HostPort(Protocol):
  """Hardware-facing host routine port visible to the CPU model."""

  def try_submit(self, request: DeviceHostRequest, cycle: int) -> bool:
    """Accept one host routine atomically, or return ``False`` when full."""

  def poll_completions(self, cycle: int) -> tuple[DeviceCompletion, ...]:
    """Return host completions made visible by the preceding harvest."""

  def step(self, cycle: int) -> None:
    """Issue up to the configured width of host commands for this cycle."""

  def abort(self, cycle: int) -> None:
    """Freeze new work, cancel unissued commands and in-flight transfers."""


class DevicePort(Protocol):
  """The complete hardware-facing interface visible to the CPU model."""

  def try_submit(self, request: DeviceLaunchRequest, cycle: int) -> bool:
    """Accept launch metadata atomically, or return ``False`` for backpressure."""

  def poll_completions(self, cycle: int) -> tuple[DeviceCompletion, ...]:
    """Return launch completions made visible by the preceding Group step."""

  def try_submit_control(self, request: DeviceControlRequest, cycle: int) -> bool:
    """Accept one explicit control command, or return ``False`` while its writer is busy."""

  def poll_control_completions(self, cycle: int) -> tuple[DeviceControlCompletion, ...]:
    """Return control completions made visible by the preceding Group step."""

  def note_await(self, instruction_id: str, events: tuple[str, ...], cycle: int) -> None:
    """Record one successfully consumed ordinary await for this run generation."""

  def note_dependencies(self, events: tuple[str, ...], cycle: int) -> None:
    """Forward successfully consumed explicit control dependencies."""


DeviceEventObserver = Callable[[str, int, Mapping[str, object]], None]


class DeviceLaunchState(StrEnum):
  """CPU-visible launch lifecycle."""

  WAIT_DEPS = "wait_deps"
  WAIT_ADMISSION = "wait_admission"
  ACTIVE = "active"
  COMPLETE = "complete"
  FAILED = "failed"


@dataclass
class _LaunchRecord:
  kind: str  # "group" | "host"
  request: DeviceLaunchRequest | DeviceHostRequest
  event_tag: str
  dependency_ids: tuple[int, ...]
  state: DeviceLaunchState
  submit_cycle: int
  dependencies_ready_cycle: int | None = None
  admission_cycle: int | None = None
  completion_cycle: int | None = None
  status: DeviceCompletionStatus | None = None
  reason: str = ""

  def snapshot(self) -> dict:
    return {
      "kind": self.kind,
      "request_id": self.request.request_id,
      "context": getattr(self.request, "context_name", "") or f"host:{self.event_tag}",
      "event": self.event_tag,
      "group_affinity": getattr(self.request, "group_affinity", None),
      "dependencies": list(self.dependency_ids),
      "state": self.state.value,
      "status": None if self.status is None else self.status.value,
      "reason": self.reason,
      "submit_cycle": self.submit_cycle,
      "dependencies_ready_cycle": self.dependencies_ready_cycle,
      "admission_cycle": self.admission_cycle,
      "active_cycle": None,
      "completion_cycle": self.completion_cycle,
      "pending_cycles": (
        None if self.admission_cycle is None else self.admission_cycle - self.submit_cycle
      ),
      "execution_cycles": (
        None
        if self.admission_cycle is None or self.completion_cycle is None
        else self.completion_cycle - self.admission_cycle
      ),
      "total_cycles": (
        None if self.completion_cycle is None else self.completion_cycle - self.submit_cycle
      ),
    }


@dataclass
class _PendingLaunch:
  record: _LaunchRecord
  unresolved_dependencies: list[int]


class CpuDeviceController:
  """Finite CPU interpreter and asynchronous launch controller.

  One call to :meth:`step` occurs before the Group step for a cycle.  One call
  to :meth:`harvest_completions` occurs after it.  Consequently a completion
  harvested in cycle ``N`` can wake a dependent launch no earlier than the CPU
  phase of cycle ``N + 1``.
  ``max_outstanding`` limits requests accepted by the hardware port. CPU
  pending descriptors have their own capacity; WAIT_DEPS never consumes a
  hardware-outstanding credit.
  """

  def __init__(
    self,
    model: ExecModel,
    config: DeviceConfig,
    max_outstanding: int,
    port: DevicePort,
    global_bindings: Mapping[str, GlobalBinding],
    observer: DeviceEventObserver | None = None,
    *,
    host_port: HostPort | None = None,
  ) -> None:
    if not isinstance(max_outstanding, int) or isinstance(max_outstanding, bool):
      raise ValueError("max_outstanding must be a positive integer")
    if max_outstanding < 1:
      raise ValueError("max_outstanding must be a positive integer")

    self.model = model
    self.config = config
    self.max_outstanding = max_outstanding
    self.port = port
    self.host_port = host_port
    self._active_host: dict[int, _LaunchRecord] = {}
    self.global_bindings = MappingProxyType(dict(global_bindings))
    self.observer = observer
    self.pmu = PMUCounter()

    self._future_references = self._validate_and_count_references(model)
    self._pc = 0
    self._returned = False
    self._next_request_id = 1
    self._event_handles: dict[str, int] = {}
    self._event_tags_by_request: dict[int, str] = {}
    self._remaining_references: dict[int, int] = {}
    self._completion_reserved: set[int] = set()
    self._completions: dict[int, DeviceCompletion] = {}
    self._pending: list[_PendingLaunch] = []
    self._active: dict[int, _LaunchRecord] = {}
    self._records: dict[int, _LaunchRecord] = {}
    self._active_control_id: str | None = None
    self._control_completion: DeviceControlCompletion | None = None
    self._control_receipts: set[str] = set()
    self._counters: Counter[str] = Counter()
    self._pending_peak = 0
    self._active_peak = 0
    self._outstanding_peak = 0
    self._completion_peak = 0
    self.fault_reason: str | None = None
    self.fault_cycle: int | None = None
    self.drain_start_cycle: int | None = None
    self.drain_complete_cycle: int | None = None

  @staticmethod
  def _validate_and_count_references(model: ExecModel) -> Counter[str]:
    """Validate direct DTO use and count event references before execution."""
    defined: set[str] = set()
    instruction_ids: set[str] = set()
    command_ids: set[str] = set()
    references: Counter[str] = Counter()
    returned = False
    for op in model.body:
      if returned:
        raise ValueError("device operation follows nexus.return")
      if not op.instruction_id or op.instruction_id in instruction_ids:
        raise ValueError("device operations require unique non-empty instruction_id values")
      instruction_ids.add(op.instruction_id)
      if op.op == "submit":
        if not op.event_tag:
          raise ValueError("device submit must define a non-empty event tag")
        if op.event_tag in defined:
          raise ValueError(f"duplicate device event tag '{op.event_tag}'")
        if not op.binding_id or op.binding_id not in model.tasks:
          raise ValueError(f"device submit references unknown binding '{op.binding_id}'")
        task = model.tasks[op.binding_id]
        if task.binding_id != op.binding_id or task.name != op.ctx_name:
          raise ValueError("device submit task identity disagrees with its binding")
        if op.binding_id not in model.context_pins:
          raise ValueError(f"device submit binding '{op.binding_id}' has no context pin entry")
        if len(op.actual_inputs) != len(task.global_inputs):
          raise ValueError(
            f"device submit for '{op.ctx_name}' has "
            f"{len(op.actual_inputs)} actuals for "
            f"{len(task.global_inputs)} global formals"
          )
        if any(
          not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(model.inputs)
          for index in op.actual_inputs
        ):
          raise ValueError(f"device submit for '{op.ctx_name}' has an invalid actual index")
        if len(set(op.dependencies)) != len(op.dependencies):
          raise ValueError(f"device submit for '{op.ctx_name}' repeats a dependency")
        for dependency in op.dependencies:
          if dependency not in defined:
            raise ValueError(f"device submit dependency '{dependency}' is not bound")
          references[dependency] += 1
        defined.add(op.event_tag)
      elif op.op == "await":
        if op.event_tag not in defined or op.dependencies or op.command is not None:
          raise ValueError(f"device await event '{op.event_tag}' is invalid")
        references[op.event_tag] += 1
      elif op.op == "host_call":
        if not isinstance(op.command, ExecHostCall):
          raise ValueError(f"device host_call '{op.instruction_id}' carries no host routine")
        if not op.command.name.strip():
          raise ValueError("device host_call requires a non-empty routine name")
        if not op.event_tag:
          raise ValueError("device host_call must define a non-empty event tag")
        if op.event_tag in defined:
          raise ValueError(f"duplicate device event tag '{op.event_tag}'")
        if op.actual_inputs or op.ctx_name or op.binding_id:
          raise ValueError("device host_call carries unexpected submit fields")
        if len(op.dependencies) != len(set(op.dependencies)) or not set(op.dependencies) <= defined:
          raise ValueError(f"device host_call '{op.instruction_id}' has invalid dependencies")
        for dependency in op.dependencies:
          references[dependency] += 1
        defined.add(op.event_tag)
      elif op.op in ("profile_reconfig", "memory_maintenance"):
        expected_type = ProfileReconfigDesc if op.op == "profile_reconfig" else MemoryMaintenanceDesc
        if (
          not isinstance(op.command, expected_type)
          or op.command.command_id != op.instruction_id
          or op.command.command_id in command_ids
          or op.event_tag
          or op.actual_inputs
        ):
          raise ValueError(f"invalid device control operation '{op.op}'")
        if op.op == "profile_reconfig" and op.dependencies:
          raise ValueError("profile reconfiguration cannot carry device dependencies")
        command_ids.add(op.command.command_id)
        if op.op == "memory_maintenance" and (
          len(op.dependencies) != len(set(op.dependencies)) or not set(op.dependencies) <= defined
        ):
          raise ValueError("memory maintenance has invalid device dependencies")
        if op.op == "memory_maintenance":
          assert isinstance(op.command, MemoryMaintenanceDesc)
          if not set(op.command.dependencies) <= set(op.dependencies):
            raise ValueError("maintenance command dependencies are absent from its Device operation")
          references.update(op.dependencies)
      elif op.op == "return":
        if op.event_tag or op.actual_inputs or op.dependencies or op.command is not None:
          raise ValueError("device return carries unexpected operands")
        returned = True
      else:
        raise ValueError(f"unknown device operation '{op.op}'")
    if not returned:
      raise ValueError("device program has no return")
    return references

  @property
  def returned(self) -> bool:
    return self._returned

  @property
  def faulted(self) -> bool:
    return self.fault_reason is not None

  @property
  def outstanding_count(self) -> int:
    return len(self._active)

  @property
  def done(self) -> bool:
    control_done = self._active_control_id is None and self._control_completion is None
    if self.faulted:
      return not self._pending and not self._active and not self._active_host and control_done
    return (
      self._returned
      and not self._pending
      and not self._active
      and not self._active_host
      and not self._completion_reserved
      and control_done
    )

  @property
  def succeeded(self) -> bool:
    return self.done and not self.faulted

  def step(self, cycle: int) -> None:
    """Run the CPU phase before the Group's cycle ``cycle`` step."""
    self.pmu.add_cycle("device_total_cycles")
    previous_pc = self._pc
    previous_capacity_stalls = self._counters["completion_backpressure_cycles"]

    if not self.faulted:
      self._resolve_pending_dependencies(cycle)
      self._advance_program(cycle)
      self._resolve_pending_dependencies(cycle)
      self._admit_ready_launches(cycle)
      if (
        not self._active
        and self._pc == previous_pc
        and self._counters["completion_backpressure_cycles"] > previous_capacity_stalls
      ):
        # No hardware completion can free this CPU-local budget, and the
        # program cannot reach another reference-consuming instruction.
        self._enter_fault("CPU completion capacity cannot satisfy the pending dependency frontier", cycle)

    self._sample_occupancy()

  def _normalize_completion(
    self, completion: DeviceCompletion, cycle: int
  ) -> DeviceCompletion:
    """Coerce a port completion to a valid terminal status at poll time."""
    try:
      status = DeviceCompletionStatus(completion.status)
    except (TypeError, ValueError):
      return DeviceCompletion(
        request_id=completion.request_id,
        status=DeviceCompletionStatus.ERROR,
        reason=f"request {completion.request_id} returned an invalid completion status",
        cycle=cycle,
      )
    if completion.cycle > cycle:
      return DeviceCompletion(
        request_id=completion.request_id,
        status=DeviceCompletionStatus.ERROR,
        reason=(
          f"request {completion.request_id} completion cycle {completion.cycle} "
          f"is later than poll cycle {cycle}"
        ),
        cycle=cycle,
      )
    return DeviceCompletion(
      request_id=completion.request_id,
      status=status,
      reason=completion.reason,
      cycle=completion.cycle,
    )

  def harvest_completions(self, cycle: int) -> tuple[DeviceCompletion, ...]:
    """Harvest launch and host completions after the Group's cycle step."""
    completions = self.port.poll_completions(cycle)
    first_error: DeviceCompletion | None = None
    for completion in completions:
      record = self._active.pop(completion.request_id, None)
      if record is None:
        self._protocol_fault(f"completion for unknown or retired request {completion.request_id}", cycle)
        continue
      completion = self._normalize_completion(completion, cycle)
      self._record_completion(record, completion)
      if completion.status is DeviceCompletionStatus.ERROR and first_error is None:
        first_error = completion

    for host_completion in (
      () if self.host_port is None else self.host_port.poll_completions(cycle)
    ):
      record = self._active_host.pop(host_completion.request_id, None)
      if record is None:
        self._protocol_fault(
          f"host completion for unknown or retired request {host_completion.request_id}", cycle
        )
        continue
      host_completion = self._normalize_completion(host_completion, cycle)
      self._record_completion(record, host_completion)
      if host_completion.status is DeviceCompletionStatus.ERROR and first_error is None:
        first_error = host_completion

    control_error: DeviceControlCompletion | None = None
    for control_completion in self.port.poll_control_completions(cycle):
      if control_completion.command_id != self._active_control_id or self._control_completion is not None:
        self._protocol_fault(
          f"control completion for unknown or retired command {control_completion.command_id!r}", cycle
        )
        continue
      if control_completion.cycle > cycle:
        control_completion = DeviceControlCompletion(
          control_completion.command_id,
          "faulted",
          f"control completion cycle {control_completion.cycle} is later than poll cycle {cycle}",
          cycle,
        )
      if control_completion.status not in ("completed", "faulted", "cancelled"):
        control_completion = DeviceControlCompletion(
          control_completion.command_id,
          "faulted",
          f"control command {control_completion.command_id!r} returned invalid status",
          cycle,
        )
      if control_completion.status == "completed" and not self.faulted:
        self._control_completion = control_completion
      else:
        self._active_control_id = None
        self._control_completion = None
        if control_completion.status != "completed":
          control_error = control_completion
          self._counters["controls_failed"] += 1
          self.pmu.add_event("device_control_failed")

    if first_error is not None:
      reason = first_error.reason or f"request {first_error.request_id} failed"
      self._enter_fault(reason, first_error.cycle)
      if self._control_completion is not None:
        self._active_control_id = None
        self._control_completion = None
    if control_error is not None:
      reason = control_error.reason or (
        f"control command {control_error.command_id} {control_error.status}"
      )
      self._enter_fault(reason, control_error.cycle)
    return completions

  def note_fault_drain_started(self, cycle: int) -> None:
    if self.drain_start_cycle is None:
      self.drain_start_cycle = cycle

  def note_fault_drain_completed(self, cycle: int) -> None:
    if self.drain_complete_cycle is None:
      self.drain_complete_cycle = cycle
    self._abandon_all_references()

  def _advance_program(self, cycle: int) -> None:
    issued = 0
    while issued < self.config.issue_width and self._pc < len(self.model.body):
      op = self.model.body[self._pc]
      if op.op == "submit":
        if not self._can_accept_submit():
          break
        dependency_ids = tuple(self._event_handles[tag] for tag in op.dependencies)
        request_id = self._next_request_id
        self._next_request_id += 1
        task = self.model.tasks[op.binding_id]
        formal_bindings = {
          formal.name: self.model.inputs[actual_index].name
          for formal, actual_index in zip(task.global_inputs, op.actual_inputs)
        }
        request = DeviceLaunchRequest(
          request_id=request_id,
          task=task,
          context_name=op.ctx_name,
          global_bindings=self.global_bindings,
          formal_bindings=MappingProxyType(formal_bindings),
          group_affinity=self.model.context_pins[op.binding_id],
        )
        state = DeviceLaunchState.WAIT_DEPS if dependency_ids else DeviceLaunchState.WAIT_ADMISSION
        record = _LaunchRecord(
          kind="group",
          request=request,
          event_tag=op.event_tag,
          dependency_ids=dependency_ids,
          state=state,
          submit_cycle=cycle,
          dependencies_ready_cycle=None if dependency_ids else cycle,
        )
        pending = _PendingLaunch(record, list(dependency_ids))
        self._records[request_id] = record
        self._pending.append(pending)
        self._event_handles[op.event_tag] = request_id
        self._event_tags_by_request[request_id] = op.event_tag
        remaining = self._future_references[op.event_tag]
        self._remaining_references[request_id] = remaining
        self._pc += 1
        issued += 1
        self._counters["submitted"] += 1
        self.pmu.add_event("device_launch_submit")
        self._observe("launch_submit", cycle, record)
        self._pending_peak = max(self._pending_peak, len(self._pending))
        self._outstanding_peak = max(self._outstanding_peak, self.outstanding_count)
        if not dependency_ids:
          self._mark_dependencies_ready(record, cycle)
        continue

      if op.op == "await":
        request_id = self._event_handles[op.event_tag]
        completion = self._completions.get(request_id)
        if completion is None:
          self._counters["await_wait_cycles"] += 1
          self.pmu.add_cycle("device_await_wait")
          break
        if completion.status is DeviceCompletionStatus.ERROR:
          self._enter_fault(completion.reason or f"awaited request {request_id} failed", cycle)
          break
        try:
          self.port.note_await(op.instruction_id, (op.event_tag,), cycle)
        except (RuntimeError, ValueError) as exc:
          self._protocol_fault(f"ordinary await proof was rejected: {exc}", cycle)
          break
        self._release_reference(request_id)
        self._pc += 1
        issued += 1
        self._counters["awaits_completed"] += 1
        self.pmu.add_event("device_await_complete")
        self._emit(
          "await_complete",
          cycle,
          {
            "request_id": request_id,
            "event": op.event_tag,
            "instruction_id": op.instruction_id,
            "events": (op.event_tag,),
          },
        )
        continue
      if op.op == "host_call":
        assert isinstance(op.command, ExecHostCall)
        if self.host_port is None:
          self._enter_fault("host_call issued without a host runtime port", cycle)
          break
        if not self._can_accept_submit():
          break
        dependency_ids = tuple(self._event_handles[tag] for tag in op.dependencies)
        host_request_id = self._next_request_id
        self._next_request_id += 1
        host_request = DeviceHostRequest(
          request_id=host_request_id,
          name=op.command.name,
          command=op.command,
          binding_names=tuple(
            self.model.inputs[access.input_index].name for access in op.command.accesses
          ),
        )
        state = DeviceLaunchState.WAIT_DEPS if dependency_ids else DeviceLaunchState.WAIT_ADMISSION
        record = _LaunchRecord(
          kind="host",
          request=host_request,
          event_tag=op.event_tag,
          dependency_ids=dependency_ids,
          state=state,
          submit_cycle=cycle,
          dependencies_ready_cycle=None if dependency_ids else cycle,
        )
        pending = _PendingLaunch(record, list(dependency_ids))
        self._records[host_request_id] = record
        self._pending.append(pending)
        self._event_handles[op.event_tag] = host_request_id
        self._event_tags_by_request[host_request_id] = op.event_tag
        self._remaining_references[host_request_id] = self._future_references[op.event_tag]
        self._pc += 1
        issued += 1
        self._counters["host_submitted"] += 1
        self.pmu.add_event("device_host_submit")
        self._observe("host_submit", cycle, record)
        self._pending_peak = max(self._pending_peak, len(self._pending))
        if not dependency_ids:
          self._mark_dependencies_ready(record, cycle)
        continue
      if op.op in ("profile_reconfig", "memory_maintenance"):
        assert isinstance(op.command, (ProfileReconfigDesc, MemoryMaintenanceDesc))
        command_id = op.command.command_id
        if self._active_control_id is None:
          if op.op == "memory_maintenance":
            assert isinstance(op.command, MemoryMaintenanceDesc)
            dependency_ids = tuple(self._event_handles[event] for event in op.dependencies)
            completed = [self._completions.get(request_id) for request_id in dependency_ids]
            if any(completion is None for completion in completed):
              self._counters["control_dependency_wait_cycles"] += 1
              self.pmu.add_cycle("device_control_dependency_wait")
              break
            failed = next(
              (
                item
                for item in completed
                if item is not None and item.status is DeviceCompletionStatus.ERROR
              ),
              None,
            )
            if failed is not None:
              self._enter_fault(failed.reason or "control dependency failed", cycle)
              break
            if command_id not in self._control_receipts:
              self.port.note_dependencies(op.command.dependencies, cycle)
              self._control_receipts.add(command_id)
          control_request = DeviceControlRequest(command_id, op.command)
          if not self.port.try_submit_control(control_request, cycle):
            self._counters["control_backpressure_cycles"] += 1
            self.pmu.add_cycle("device_control_wait")
            break
          self._active_control_id = command_id
          if op.op == "memory_maintenance":
            for request_id in dependency_ids:
              self._release_reference(request_id)
          self._counters["controls_submitted"] += 1
          self.pmu.add_event("device_control_submit")
          self._emit(
            "control_submit",
            cycle,
            {"command_id": command_id, "kind": op.op, "instruction_id": op.instruction_id},
          )
          break
        if self._active_control_id != command_id:
          self._protocol_fault(
            f"control fence for {command_id!r} found active command {self._active_control_id!r}", cycle
          )
          break
        control_result = self._control_completion
        if control_result is None:
          self._counters["control_wait_cycles"] += 1
          self.pmu.add_cycle("device_control_wait")
          break
        if control_result.status != "completed":
          self._enter_fault(
            control_result.reason or f"control command {command_id} {control_result.status}", cycle
          )
          break
        self._active_control_id = None
        self._control_completion = None
        self._pc += 1
        issued += 1
        self._counters["controls_completed"] += 1
        self.pmu.add_event("device_control_complete")
        self._emit(
          "control_complete",
          cycle,
          {
            "command_id": command_id,
            "kind": op.op,
            "instruction_id": op.instruction_id,
            "completion_cycle": control_result.cycle,
          },
        )
        continue

      self._returned = True
      self._pc += 1
      issued += 1
      self._counters["returns"] += 1
      self.pmu.add_event("device_return")
      self._emit("return", cycle, {"pc": self._pc})

    if issued == self.config.issue_width and self._pc < len(self.model.body):
      self._counters["issue_width_limited_cycles"] += 1
      self.pmu.add_cycle("device_issue_width_limited")

  def _can_accept_submit(self) -> bool:
    if len(self._pending) >= self.config.pending_capacity:
      self._counters["pending_backpressure_cycles"] += 1
      self.pmu.add_cycle("device_pending_full")
      return False
    return True

  def _mark_dependencies_ready(self, record: _LaunchRecord, cycle: int) -> None:
    record.dependencies_ready_cycle = cycle
    record.state = DeviceLaunchState.WAIT_ADMISSION
    self._counters["dependencies_ready"] += 1
    self.pmu.add_event("device_dependencies_ready")
    self._observe("launch_dependencies_ready", cycle, record)

  def _resolve_pending_dependencies(self, cycle: int) -> None:
    for pending in tuple(self._pending):
      if not pending.unresolved_dependencies:
        if pending.record.dependencies_ready_cycle is None:
          self._mark_dependencies_ready(pending.record, cycle)
        continue

      failed: DeviceCompletion | None = None
      for request_id in tuple(pending.unresolved_dependencies):
        completion = self._completions.get(request_id)
        if completion is None:
          continue
        pending.unresolved_dependencies.remove(request_id)
        self._release_reference(request_id)
        if completion.status is DeviceCompletionStatus.ERROR:
          failed = completion
          break
      if failed is not None:
        self._pending.remove(pending)
        self._release_pending_dependencies(pending)
        reason = f"dependency request {failed.request_id} failed" + (
          f": {failed.reason}" if failed.reason else ""
        )
        self._record_unadmitted_failure(pending.record, reason, cycle, dependency_failure=True)
        self._enter_fault(reason, cycle)
      elif not pending.unresolved_dependencies:
        self._mark_dependencies_ready(pending.record, cycle)

  def _admit_ready_launches(self, cycle: int) -> None:
    attempts = 0
    if len(self._active) >= self.max_outstanding:
      if self._pending:
        self._counters["outstanding_backpressure_cycles"] += 1
        self.pmu.add_cycle("device_outstanding_full")
      return
    for pending in tuple(self._pending):
      if pending.unresolved_dependencies:
        self._counters["dependency_wait_cycles"] += 1
        continue
      if attempts >= self.config.issue_width:
        break
      record = pending.record
      request_id = record.request.request_id
      if record.kind == "group" and len(self._active) >= self.max_outstanding:
        break
      reserve_completion = self._remaining_references[request_id] > 0
      if reserve_completion and len(self._completion_reserved) >= self.config.completion_capacity:
        self._counters["completion_backpressure_cycles"] += 1
        self.pmu.add_cycle("device_completion_full")
        break
      if reserve_completion:
        self._completion_reserved.add(request_id)
        self._completion_peak = max(self._completion_peak, len(self._completion_reserved))
      attempts += 1
      if record.kind == "host":
        assert self.host_port is not None
        assert isinstance(record.request, DeviceHostRequest)
        try:
          accepted = self.host_port.try_submit(record.request, cycle)
        except ValueError as exc:
          self._completion_reserved.discard(request_id)
          self._protocol_fault(f"host routine rejected at issue: {exc}", cycle)
          break
      else:
        assert isinstance(record.request, DeviceLaunchRequest)
        accepted = self.port.try_submit(record.request, cycle)
      if not accepted:
        self._completion_reserved.discard(request_id)
        self._counters["admission_backpressure_cycles"] += 1
        self.pmu.add_cycle("device_admission_wait")
        break
      self._pending.remove(pending)
      record.state = DeviceLaunchState.ACTIVE
      record.admission_cycle = cycle
      if record.kind == "host":
        self._active_host[request_id] = record
        self._counters["host_admitted"] += 1
        self.pmu.add_event("device_host_admit")
        self._observe("host_admission", cycle, record)
        continue
      self._active[request_id] = record
      self._active_peak = max(self._active_peak, len(self._active))
      self._counters["admitted"] += 1
      self.pmu.add_event("device_launch_admit")
      self._observe("launch_admission", cycle, record)

  def _record_completion(
    self, record: _LaunchRecord, completion: DeviceCompletion, *, retain: bool = True
  ) -> None:
    record.completion_cycle = completion.cycle
    record.status = completion.status
    record.reason = completion.reason
    record.state = (
      DeviceLaunchState.COMPLETE
      if completion.status is DeviceCompletionStatus.SUCCESS
      else DeviceLaunchState.FAILED
    )
    if retain and self._remaining_references[completion.request_id] > 0:
      if completion.request_id not in self._completion_reserved:
        raise RuntimeError(f"completion metadata was not reserved for request {completion.request_id}")
      self._completions[completion.request_id] = completion
    else:
      self._retire_event_handle(completion.request_id)
    if completion.status is DeviceCompletionStatus.SUCCESS:
      if record.kind == "host":
        self._counters["host_completed"] += 1
        self.pmu.add_event("device_host_complete")
      else:
        self._counters["completed"] += 1
        self.pmu.add_event("device_launch_complete")
    else:
      if record.kind == "host":
        self._counters["host_failed"] += 1
        self.pmu.add_event("device_host_failed")
      else:
        self._counters["failed"] += 1
        self.pmu.add_event("device_launch_failed")
    self._completion_peak = max(self._completion_peak, len(self._completions))
    self._observe("launch_completion" if record.kind == "group" else "host_completion",
                  completion.cycle, record)

  def _record_unadmitted_failure(
    self, record: _LaunchRecord, reason: str, cycle: int, *, dependency_failure: bool
  ) -> None:
    completion = DeviceCompletion(
      request_id=record.request.request_id, status=DeviceCompletionStatus.ERROR, reason=reason, cycle=cycle
    )
    self._record_completion(record, completion, retain=False)
    if dependency_failure:
      self._counters["dependency_failed"] += 1
      self.pmu.add_event("device_dependency_failed")
    else:
      self._counters["cancelled"] += 1
      self.pmu.add_event("device_launch_cancelled")

  def _enter_fault(self, reason: str, cycle: int) -> None:
    if self.fault_reason is None:
      self.fault_reason = reason
      self.fault_cycle = cycle
      self._counters["faults"] += 1
      self.pmu.add_event("device_fault")
      self._emit("fault", cycle, {"reason": reason})

    for pending in tuple(self._pending):
      dependency_error = next(
        (
          self._records[request_id]
          for request_id in pending.unresolved_dependencies
          if self._records[request_id].status is DeviceCompletionStatus.ERROR
        ),
        None,
      )
      self._pending.remove(pending)
      self._release_pending_dependencies(pending)
      if dependency_error is not None:
        pending_reason = f"dependency request {dependency_error.request.request_id} failed" + (
          f": {dependency_error.reason}" if dependency_error.reason else ""
        )
      else:
        pending_reason = f"device stopped after failure: {self.fault_reason}"
      self._record_unadmitted_failure(
        pending.record, pending_reason, cycle, dependency_failure=dependency_error is not None
      )

    # No future CPU operation can consume event handles after a device fault.
    self._abandon_all_references()

  def _protocol_fault(self, reason: str, cycle: int) -> None:
    self._counters["protocol_faults"] += 1
    self.pmu.add_event("device_protocol_fault")
    self._enter_fault(f"device port protocol error: {reason}", cycle)

  def _release_pending_dependencies(self, pending: _PendingLaunch) -> None:
    for request_id in tuple(pending.unresolved_dependencies):
      self._release_reference(request_id)
    pending.unresolved_dependencies.clear()

  def _retire_event_handle(self, request_id: int) -> None:
    tag = self._event_tags_by_request.get(request_id)
    if tag is not None and self._event_handles.get(tag) == request_id:
      self._event_handles.pop(tag, None)

  def _release_reference(self, request_id: int) -> None:
    remaining = self._remaining_references.get(request_id)
    if remaining is None or remaining < 1:
      raise RuntimeError(f"event reference underflow for request {request_id}")
    remaining -= 1
    self._remaining_references[request_id] = remaining
    if remaining:
      return
    self._completions.pop(request_id, None)
    self._completion_reserved.discard(request_id)
    self._retire_event_handle(request_id)

  def _abandon_all_references(self) -> None:
    for request_id in tuple(self._remaining_references):
      self._remaining_references[request_id] = 0
    self._completion_reserved.clear()
    self._completions.clear()
    self._event_handles.clear()

  def _sample_occupancy(self) -> None:
    pending = len(self._pending)
    active = len(self._active)
    outstanding = active
    retained = len(self._completions)
    self._pending_peak = max(self._pending_peak, pending)
    self._active_peak = max(self._active_peak, active)
    self._outstanding_peak = max(self._outstanding_peak, outstanding)
    self._completion_peak = max(self._completion_peak, retained)
    self.pmu.add_cycle("device_pending_occupancy", pending)
    self.pmu.add_cycle("device_active_occupancy", active)
    self.pmu.add_cycle("device_completion_occupancy", retained)
    self.pmu.add_cycle("device_control_occupancy", int(self._active_control_id is not None))

  def _observe(self, event: str, cycle: int, record: _LaunchRecord) -> None:
    self._emit(
      event,
      cycle,
      {
        "request_id": record.request.request_id,
        "context": getattr(record.request, "context_name", "") or f"host:{record.event_tag}",
        "event": record.event_tag,
        "state": record.state.value,
        "submit_cycle": record.submit_cycle,
        "dependencies_ready_cycle": record.dependencies_ready_cycle,
        "admission_cycle": record.admission_cycle,
        "completion_cycle": record.completion_cycle,
        "status": None if record.status is None else record.status.value,
        "reason": record.reason,
      },
    )

  def _emit(self, event: str, cycle: int, args: Mapping[str, object]) -> None:
    if self.observer is not None:
      self.observer(event, cycle, args)

  def snapshot(self) -> dict:
    """Return bounded live state plus the model's finite launch history."""
    counter_names = (
      "submitted",
      "host_submitted",
      "host_admitted",
      "host_completed",
      "host_failed",
      "dependencies_ready",
      "admitted",
      "completed",
      "failed",
      "dependency_failed",
      "cancelled",
      "faults",
      "protocol_faults",
      "awaits_completed",
      "controls_submitted",
      "controls_completed",
      "controls_failed",
      "returns",
      "await_wait_cycles",
      "dependency_wait_cycles",
      "control_dependency_wait_cycles",
      "pending_backpressure_cycles",
      "outstanding_backpressure_cycles",
      "completion_backpressure_cycles",
      "admission_backpressure_cycles",
      "control_backpressure_cycles",
      "control_wait_cycles",
      "issue_width_limited_cycles",
    )
    counters = {name: self._counters[name] for name in counter_names}
    counters.update(
      {
        "pending": len(self._pending),
        "active": len(self._active),
        "outstanding": self.outstanding_count,
        "live_launches": len(self._pending) + len(self._active),
        "active_host": len(self._active_host),
        "retained_completions": len(self._completions),
        "reserved_completions": len(self._completion_reserved),
        "pending_peak": self._pending_peak,
        "active_peak": self._active_peak,
        "outstanding_peak": self._outstanding_peak,
        "completion_peak": self._completion_peak,
      }
    )
    return {
      "configuration": {
        "pending_capacity": self.config.pending_capacity,
        "completion_capacity": self.config.completion_capacity,
        "issue_width": self.config.issue_width,
        "outstanding_launch_limit": self.max_outstanding,
      },
      "pc": self._pc,
      "body_length": len(self.model.body),
      "returned": self._returned,
      "completed": self.succeeded,
      "faulted": self.faulted,
      "reason": self.fault_reason or "",
      "fault_cycle": self.fault_cycle,
      "drain_start_cycle": self.drain_start_cycle,
      "drain_complete_cycle": self.drain_complete_cycle,
      "active_control_id": self._active_control_id,
      "control_completion": (
        None
        if self._control_completion is None
        else {
          "command_id": self._control_completion.command_id,
          "status": self._control_completion.status,
          "reason": self._control_completion.reason,
          "cycle": self._control_completion.cycle,
        }
      ),
      "counters": counters,
      "launch_records": [self._records[request_id].snapshot() for request_id in sorted(self._records)],
    }
