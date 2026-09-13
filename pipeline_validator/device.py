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
from .execution_ir import ExecModel, ExecTileGroupTask, GlobalBinding
from .pmu import PMUCounter


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


class DevicePort(Protocol):
  """The complete hardware-facing interface visible to the CPU model."""

  def try_submit(self, request: DeviceLaunchRequest, cycle: int) -> bool:
    """Accept ``request`` atomically, or return ``False`` for backpressure."""

  def poll_completions(self, cycle: int) -> tuple[DeviceCompletion, ...]:
    """Return terminal completions made visible by the preceding Group step."""


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
  request: DeviceLaunchRequest
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
      "request_id": self.request.request_id,
      "context": self.request.context_name,
      "event": self.event_tag,
      "group_affinity": self.request.group_affinity,
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
  ) -> None:
    if not isinstance(max_outstanding, int) or isinstance(max_outstanding, bool):
      raise ValueError("max_outstanding must be a positive integer")
    if max_outstanding < 1:
      raise ValueError("max_outstanding must be a positive integer")

    self.model = model
    self.config = config
    self.max_outstanding = max_outstanding
    self.port = port
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
    references: Counter[str] = Counter()
    returned = False
    for op in model.body:
      if returned:
        raise ValueError("device operation follows nexus.return")
      if op.op == "submit":
        if not op.event_tag:
          raise ValueError("device submit must define a non-empty event tag")
        if op.event_tag in defined:
          raise ValueError(f"duplicate device event tag '{op.event_tag}'")
        if op.ctx_name not in model.tasks:
          raise ValueError(f"device submit references unknown context '{op.ctx_name}'")
        task = model.tasks[op.ctx_name]
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
        if op.event_tag not in defined:
          raise ValueError(f"device await event '{op.event_tag}' is not bound")
        references[op.event_tag] += 1
      elif op.op == "return":
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
    if self.faulted:
      return not self._pending and not self._active
    return self._returned and not self._pending and not self._active and not self._completion_reserved

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

  def harvest_completions(self, cycle: int) -> tuple[DeviceCompletion, ...]:
    """Harvest completions after the Group's cycle ``cycle`` step."""
    completions = self.port.poll_completions(cycle)
    first_error: DeviceCompletion | None = None
    for completion in completions:
      record = self._active.pop(completion.request_id, None)
      if record is None:
        self._protocol_fault(f"completion for unknown or retired request {completion.request_id}", cycle)
        continue
      try:
        status = DeviceCompletionStatus(completion.status)
      except (TypeError, ValueError):
        status = DeviceCompletionStatus.ERROR
        completion = DeviceCompletion(
          request_id=completion.request_id,
          status=status,
          reason=(f"request {completion.request_id} returned invalid completion status"),
          cycle=cycle,
        )
      else:
        completion = DeviceCompletion(
          request_id=completion.request_id, status=status, reason=completion.reason, cycle=completion.cycle
        )
      if completion.cycle > cycle:
        completion = DeviceCompletion(
          request_id=completion.request_id,
          status=DeviceCompletionStatus.ERROR,
          reason=(
            f"request {completion.request_id} completion cycle "
            f"{completion.cycle} is later than poll cycle {cycle}"
          ),
          cycle=cycle,
        )
      self._record_completion(record, completion)
      if completion.status is DeviceCompletionStatus.ERROR and first_error is None:
        first_error = completion

    if first_error is not None:
      reason = first_error.reason or f"request {first_error.request_id} failed"
      self._enter_fault(reason, first_error.cycle)
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
        task = self.model.tasks[op.ctx_name]
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
          group_affinity=self.model.context_pins.get(op.ctx_name),
        )
        state = DeviceLaunchState.WAIT_DEPS if dependency_ids else DeviceLaunchState.WAIT_ADMISSION
        record = _LaunchRecord(
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
          self._counters["dependencies_ready"] += 1
          self.pmu.add_event("device_dependencies_ready")
          self._observe("launch_dependencies_ready", cycle, record)
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
        self._release_reference(request_id)
        self._pc += 1
        issued += 1
        self._counters["awaits_completed"] += 1
        self.pmu.add_event("device_await_complete")
        self._emit("await_complete", cycle, {"request_id": request_id, "event": op.event_tag})
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

  def _resolve_pending_dependencies(self, cycle: int) -> None:
    for pending in tuple(self._pending):
      if not pending.unresolved_dependencies:
        if pending.record.dependencies_ready_cycle is None:
          pending.record.dependencies_ready_cycle = cycle
          pending.record.state = DeviceLaunchState.WAIT_ADMISSION
          self._counters["dependencies_ready"] += 1
          self.pmu.add_event("device_dependencies_ready")
          self._observe("launch_dependencies_ready", cycle, pending.record)
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
        pending.record.dependencies_ready_cycle = cycle
        pending.record.state = DeviceLaunchState.WAIT_ADMISSION
        self._counters["dependencies_ready"] += 1
        self.pmu.add_event("device_dependencies_ready")
        self._observe("launch_dependencies_ready", cycle, pending.record)

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
      if attempts >= self.config.issue_width or len(self._active) >= self.max_outstanding:
        break
      record = pending.record
      request_id = record.request.request_id
      reserve_completion = self._remaining_references[request_id] > 0
      if reserve_completion and len(self._completion_reserved) >= self.config.completion_capacity:
        self._counters["completion_backpressure_cycles"] += 1
        self.pmu.add_cycle("device_completion_full")
        break
      if reserve_completion:
        self._completion_reserved.add(request_id)
        self._completion_peak = max(self._completion_peak, len(self._completion_reserved))
      attempts += 1
      if not self.port.try_submit(record.request, cycle):
        self._completion_reserved.discard(request_id)
        self._counters["admission_backpressure_cycles"] += 1
        self.pmu.add_cycle("device_admission_wait")
        break
      self._pending.remove(pending)
      record.state = DeviceLaunchState.ACTIVE
      record.admission_cycle = cycle
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
      tag = self._event_tags_by_request.get(completion.request_id)
      if tag is not None and self._event_handles.get(tag) == completion.request_id:
        self._event_handles.pop(tag, None)
    if completion.status is DeviceCompletionStatus.SUCCESS:
      self._counters["completed"] += 1
      self.pmu.add_event("device_launch_complete")
    else:
      self._counters["failed"] += 1
      self.pmu.add_event("device_launch_failed")
    self._completion_peak = max(self._completion_peak, len(self._completions))
    self._observe("launch_completion", completion.cycle, record)

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
    tag = self._event_tags_by_request.get(request_id)
    if tag is not None and self._event_handles.get(tag) == request_id:
      self._event_handles.pop(tag, None)

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

  def _observe(self, event: str, cycle: int, record: _LaunchRecord) -> None:
    self._emit(
      event,
      cycle,
      {
        "request_id": record.request.request_id,
        "context": record.request.context_name,
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
      "dependencies_ready",
      "admitted",
      "completed",
      "failed",
      "dependency_failed",
      "cancelled",
      "faults",
      "protocol_faults",
      "awaits_completed",
      "returns",
      "await_wait_cycles",
      "dependency_wait_cycles",
      "pending_backpressure_cycles",
      "outstanding_backpressure_cycles",
      "completion_backpressure_cycles",
      "admission_backpressure_cycles",
      "issue_width_limited_cycles",
    )
    counters = {name: self._counters[name] for name in counter_names}
    counters.update(
      {
        "pending": len(self._pending),
        "active": len(self._active),
        "outstanding": self.outstanding_count,
        "live_launches": len(self._pending) + len(self._active),
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
      "counters": counters,
      "launch_records": [self._records[request_id].snapshot() for request_id in sorted(self._records)],
    }
