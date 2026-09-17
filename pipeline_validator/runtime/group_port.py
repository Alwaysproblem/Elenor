"""Concrete message adapter between the CPU model and one TileGroup."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..device import (
  DeviceCompletion,
  DeviceCompletionStatus,
  DeviceControlCompletion,
  DeviceControlRequest,
  DeviceLaunchRequest,
  DevicePort,
)
from ..execution_ir import ContextAdmissionStatus
from ..memory.allocator import AdmissionWaitReason, MemoryInvariantError
from ..tile_group import TileGroup
from ..tile_group_sequencer import TileGroupSequencer


@dataclass
class _RootRequestRecord:
  request: DeviceLaunchRequest
  ready_seq: int
  category: str
  submit_cycle: int
  lifecycle: str = "pending"
  slot_index: int | None = None
  sequencer: TileGroupSequencer | None = None
  active_cycle: int | None = None
  completion_cycle: int | None = None
  status: DeviceCompletionStatus | None = None
  reason: str = ""
  wait_reason: AdmissionWaitReason | None = None
  wait_start_cycle: int | None = None
  retry_count: int = 0

  def snapshot(self) -> dict:
    return {
      "request_id": self.request.request_id,
      "binding_id": self.request.task.binding_id,
      "context": self.request.context_name,
      "ready_seq": self.ready_seq,
      "category": self.category,
      "lifecycle": self.lifecycle,
      "slot_index": self.slot_index,
      "submit_cycle": self.submit_cycle,
      "active_cycle": self.active_cycle,
      "completion_cycle": self.completion_cycle,
      "status": None if self.status is None else self.status.value,
      "reason": self.reason,
      "wait_reason": None if self.wait_reason is None else self.wait_reason.value,
      "wait_start_cycle": self.wait_start_cycle,
      "retry_count": self.retry_count,
    }


class GroupPortAdapter(DevicePort):
  """Bounded launch metadata and explicit control adapter for one TileGroup.

  A successful :meth:`try_submit` consumes only a pending metadata entry.
  Hardware slot, event, and L2 ownership are acquired together later by
  ``TileGroup.try_admit_context_task``.  The CPU request remains hardware
  outstanding throughout that wait.
  """

  def __init__(self, group: TileGroup, active_context_capacity: int) -> None:
    if (
      not isinstance(active_context_capacity, int)
      or isinstance(active_context_capacity, bool)
      or active_context_capacity < 1
    ):
      raise ValueError("active_context_capacity must be a positive integer")
    pending_capacity = group.scheduler_config.context_pending_capacity
    if not isinstance(pending_capacity, int) or isinstance(pending_capacity, bool) or pending_capacity < 1:
      raise ValueError("context_pending_capacity must be a positive integer")
    if group._pending_root_requests:
      raise RuntimeError("Group has stale pending root requests at run start")

    self.group = group
    self.active_context_capacity = active_context_capacity
    self.context_pending_capacity = pending_capacity
    self._run_generation = group.run_generation
    self._slots: list[_RootRequestRecord | None] = [None for _ in range(active_context_capacity)]
    self._pending: dict[int, _RootRequestRecord] = {}
    self._category_queues: dict[str, deque[int]] = {"same": deque(), "compatible": deque()}
    self._ready_completions: list[DeviceCompletion] = []
    self._seen_request_ids: set[int] = set()
    self._records: dict[int, _RootRequestRecord] = {}
    self._next_ready_seq = 0
    self._pending_changed = False
    self._slot_version = 0
    self._last_admission_token: tuple[object, int] | None = None

    self._ready_control_completions: list[DeviceControlCompletion] = []
    self._control_inflight: set[str] = set()
    self._seen_control_ids: set[str] = set()
    self._controls_submitted = 0
    self._controls_completed = 0
    self._controls_failed = 0
    self._control_backpressure = 0

    self._slot_peak = 0
    self._pending_peak = 0
    self._submitted = 0
    self._completed = 0
    self._failed = 0
    self._backpressure = 0
    self._fault_reason: str | None = None

  @property
  def active_count(self) -> int:
    return sum(slot is not None for slot in self._slots)

  @property
  def pending_count(self) -> int:
    return len(self._pending)

  @property
  def _control_owner(self) -> str:
    if self.group.run_generation != self._run_generation:
      raise RuntimeError("GroupPortAdapter run generation is stale")
    return f"device@{self._run_generation}"

  def _group_fault(self) -> str | None:
    if self._fault_reason is not None:
      return self._fault_reason
    if self.group.reset_domain.is_active:
      return "Group reset is draining the current run"
    if self.group.reset_domain.is_done:
      return "Group reset cancelled the current run"
    return None

  def _profile_category(self, request: DeviceLaunchRequest) -> str:
    binding_id = request.task.binding_id
    if not binding_id:
      raise ValueError("root launch task has no binding_id")
    program = self.group.loaded_program
    if program is None:
      raise ValueError("root launch requires a loaded compiled program")
    try:
      binding = program.call_bindings[binding_id]
    except KeyError as exc:
      raise ValueError(f"unknown compiled call binding {binding_id!r}") from exc
    if binding.binding_id != binding_id or binding.context_name != request.context_name:
      raise ValueError("root launch identity disagrees with its compiled call binding")
    if request.task.name != request.context_name:
      raise ValueError("root launch report name disagrees with its executable task")
    try:
      active_l2 = self.group.profile_controller.active_modes["l2"]
    except KeyError as exc:
      raise ValueError("ProfileController has no active L2 mode") from exc
    if active_l2 != binding.resolved_l2_mode:
      raise ValueError(
        f"binding {binding_id!r} resolved L2 mode {binding.resolved_l2_mode}, "
        f"but the active mode is {active_l2}"
      )
    if not any(profile.l2_mode == active_l2 for profile in binding.permitted_profiles):
      raise ValueError(f"active L2 mode {active_l2} is outside binding {binding_id!r} permitted profiles")
    return "same" if active_l2 == binding.requested_l2_mode else "compatible"

  def _new_record(self, request: DeviceLaunchRequest, cycle: int, category: str) -> _RootRequestRecord:
    record = _RootRequestRecord(request, self._next_ready_seq, category, cycle)
    self._next_ready_seq += 1
    self._seen_request_ids.add(request.request_id)
    self._records[request.request_id] = record
    self._submitted += 1
    return record

  def _reject_submit(self, request: DeviceLaunchRequest, cycle: int, reason: str) -> bool:
    record = self._new_record(request, cycle, "invalid")
    record.lifecycle = "failed"
    record.completion_cycle = cycle
    record.status = DeviceCompletionStatus.ERROR
    record.reason = reason
    self._failed += 1
    self._fault_reason = self._fault_reason or reason
    self._ready_completions.append(
      DeviceCompletion(request.request_id, DeviceCompletionStatus.ERROR, reason, cycle)
    )
    return True

  def try_submit(self, request: DeviceLaunchRequest, cycle: int) -> bool:
    if request.request_id in self._seen_request_ids:
      raise ValueError(f"duplicate device request_id {request.request_id}")
    if request.group_affinity is not None and (
      type(request.group_affinity) is not int
      or not 0 <= request.group_affinity < self.active_context_capacity
    ):
      return self._reject_submit(
        request,
        cycle,
        f"request {request.request_id} targets unavailable Group context slot {request.group_affinity}",
      )
    fault = self._group_fault()
    if fault is not None:
      return self._reject_submit(request, cycle, fault)
    try:
      category = self._profile_category(request)
    except (RuntimeError, ValueError) as exc:
      return self._reject_submit(request, cycle, str(exc))
    if len(self._pending) >= self.context_pending_capacity:
      self._backpressure += 1
      return False

    record = self._new_record(request, cycle, category)
    self._pending[request.request_id] = record
    self.group._pending_root_requests[request.request_id] = record
    self._category_queues[category].append(request.request_id)
    self._pending_changed = True
    self._pending_peak = max(self._pending_peak, len(self._pending))
    return True

  def _vacant_slot(self, record: _RootRequestRecord) -> int | None:
    affinity = record.request.group_affinity
    if affinity is not None:
      return affinity if self._slots[affinity] is None else None
    return next((index for index, slot in enumerate(self._slots) if slot is None), None)

  def _remove_pending(self, record: _RootRequestRecord) -> None:
    request_id = record.request.request_id
    pending = self._pending.pop(request_id, None)
    mirrored = self.group._pending_root_requests.pop(request_id, None)
    if pending is not record or mirrored is not record:
      raise RuntimeError(f"pending root request {request_id} lost its mirrored identity")

  def _fail_pending(
    self, record: _RootRequestRecord, reason: str, cycle: int, completions: list[DeviceCompletion]
  ) -> None:
    self._remove_pending(record)
    record.lifecycle = "failed"
    record.completion_cycle = cycle
    record.status = DeviceCompletionStatus.ERROR
    record.reason = reason
    self._failed += 1
    self._fault_reason = self._fault_reason or reason
    completions.append(
      DeviceCompletion(record.request.request_id, DeviceCompletionStatus.ERROR, reason, cycle)
    )

  def _queue_head(self, category: str) -> _RootRequestRecord | None:
    queue = self._category_queues[category]
    while queue and queue[0] not in self._pending:
      queue.popleft()
    return None if not queue else self._pending[queue[0]]

  def _trace_pending(self, name: str, record: _RootRequestRecord, cycle: int) -> None:
    if self.group.tracer is None or self.group.memory_trace is None:
      return
    self.group.tracer.instant(
      "TileGroup",
      "Scheduler:L2",
      name,
      cycle,
      {
        "context": record.request.context_name,
        "binding_id": record.request.task.binding_id,
        "request_id": record.request.request_id,
        "ready_seq": record.ready_seq,
        "slot": None,
        "retry_count": record.retry_count,
        "wait_reason": None if record.wait_reason is None else record.wait_reason.value,
      },
    )

  def _note_wait(self, record: _RootRequestRecord, reason: AdmissionWaitReason, cycle: int) -> None:
    changed = record.wait_reason is not reason
    record.wait_reason = reason
    if record.wait_start_cycle is None:
      record.wait_start_cycle = cycle
    if changed:
      self._trace_pending("context_admission_wait", record, cycle)

  def _attempt_category(self, category: str, cycle: int, completions: list[DeviceCompletion]) -> str:
    record = self._queue_head(category)
    if record is None:
      return "empty"
    if record.wait_reason is not None:
      record.retry_count += 1
      self._trace_pending("context_admission_retry", record, cycle)
    try:
      actual_category = self._profile_category(record.request)
    except (RuntimeError, ValueError) as exc:
      self._fail_pending(record, str(exc), cycle, completions)
      return "fault"
    if actual_category != category:
      self._fail_pending(
        record,
        f"request {record.request.request_id} profile category changed while pending",
        cycle,
        completions,
      )
      return "fault"
    slot_index = self._vacant_slot(record)
    if slot_index is None:
      self._note_wait(record, AdmissionWaitReason.SLOT, cycle)
      return "blocked"
    try:
      sequencer = self.group.try_admit_context_task(
        record.request.task,
        slot_index,
        context_name=record.request.context_name,
        input_bindings=record.request.global_bindings,
        formal_bindings=dict(record.request.formal_bindings),
        cycle=cycle,
      )
    except (RuntimeError, ValueError, MemoryInvariantError) as exc:
      self._fail_pending(record, str(exc), cycle, completions)
      return "fault"
    if sequencer is None:
      assert self.group.last_admission_wait is not None
      self._note_wait(record, self.group.last_admission_wait, cycle)
      return "blocked"

    if (
      sequencer.admission_status is not ContextAdmissionStatus.ACTIVE or sequencer.faulted or sequencer.done
    ):
      reason = sequencer.fault_reason or "Group rejected root admission permanently"
      record.sequencer = sequencer
      self._fail_pending(record, reason, cycle, completions)
      return "fault"
    self._remove_pending(record)

    record.sequencer = sequencer
    record.slot_index = slot_index
    record.active_cycle = cycle
    record.lifecycle = "active"
    self._slots[slot_index] = record
    self._slot_peak = max(self._slot_peak, self.active_count)
    return "admitted"

  def _admit_pending(self, cycle: int, completions: list[DeviceCompletion]) -> None:
    if not self._pending:
      self._pending_changed = False
      self._last_admission_token = (self.group.admission_version, self._slot_version)
      return
    token = (self.group.admission_version, self._slot_version)
    if not self._pending_changed and token == self._last_admission_token:
      return

    while self._pending and self._group_fault() is None:
      same = self._attempt_category("same", cycle, completions)
      if same == "admitted":
        continue
      if same == "fault":
        break
      compatible = self._attempt_category("compatible", cycle, completions)
      if compatible == "admitted":
        continue
      if compatible == "fault":
        break
      break
    self._pending_changed = False
    self._last_admission_token = (self.group.admission_version, self._slot_version)

  def _cancel_pending(self, reason: str, cycle: int, completions: list[DeviceCompletion]) -> None:
    for record in tuple(self._pending.values()):
      self._fail_pending(record, reason, cycle, completions)
    self._pending_changed = False

  def _retire_slots(self, cycle: int, completions: list[DeviceCompletion]) -> None:
    reset_active = self.group.reset_domain.is_active
    reset_done = self.group.reset_domain.is_done
    for slot_index, record in enumerate(tuple(self._slots)):
      if record is None:
        continue
      sequencer = record.sequencer
      if sequencer is None:
        raise RuntimeError("occupied Group slot has no sequencer")
      if reset_active and not sequencer.faulted:
        continue
      if not sequencer.done and not reset_done:
        continue
      if sequencer.faulted:
        status = DeviceCompletionStatus.ERROR
        reason = sequencer.fault_reason or "Group context failed"
      elif reset_active or reset_done:
        status = DeviceCompletionStatus.ERROR
        reason = self._fault_reason or "Group reset cancelled the launch"
      else:
        status = DeviceCompletionStatus.SUCCESS
        reason = ""

      record.completion_cycle = cycle
      record.status = status
      record.reason = reason
      record.lifecycle = "complete" if status is DeviceCompletionStatus.SUCCESS else "failed"
      completions.append(DeviceCompletion(record.request.request_id, status, reason, cycle))
      self._slots[slot_index] = None
      self._slot_version += 1
      if status is DeviceCompletionStatus.SUCCESS:
        self._completed += 1
      else:
        self._failed += 1
        self._fault_reason = self._fault_reason or reason

  def poll_completions(self, cycle: int) -> tuple[DeviceCompletion, ...]:
    completions = self._ready_completions
    self._ready_completions = []
    self._retire_slots(cycle, completions)
    fault = self._group_fault()
    if fault is not None:
      self._cancel_pending(fault, cycle, completions)
    else:
      self._admit_pending(cycle, completions)
      fault = self._group_fault()
      if fault is not None:
        self._cancel_pending(fault, cycle, completions)
    return tuple(completions)

  def try_submit_control(self, request: DeviceControlRequest, cycle: int) -> bool:
    owner = self._control_owner
    if request.command_id != request.command.command_id:
      raise ValueError("control request ID disagrees with its descriptor")
    if request.command_id in self._seen_control_ids:
      raise ValueError(f"duplicate device control command {request.command_id!r}")
    try:
      accepted = self.group.profile_controller.submit(request.command, owner, cycle)
    except (RuntimeError, ValueError) as exc:
      reason = str(exc)
      self._seen_control_ids.add(request.command_id)
      self._controls_submitted += 1
      self._controls_failed += 1
      self._ready_control_completions.append(
        DeviceControlCompletion(request.command_id, "faulted", reason, cycle)
      )
      return True
    if not accepted:
      self._control_backpressure += 1
      return False
    self._seen_control_ids.add(request.command_id)
    self._control_inflight.add(request.command_id)
    self._controls_submitted += 1
    return True

  def poll_control_completions(self, cycle: int) -> tuple[DeviceControlCompletion, ...]:
    _owner = self._control_owner
    completions = self._ready_control_completions
    self._ready_control_completions = []
    for command_id in tuple(self._control_inflight):
      try:
        status = self.group.profile_controller.status(command_id)
        if status in ("pending", "running"):
          continue
        if status not in ("completed", "faulted", "cancelled"):
          status = "faulted"
          reason = f"ProfileController returned invalid status for {command_id!r}"
        else:
          reason = self.group.profile_controller.command_reason(command_id)
      except (KeyError, RuntimeError, ValueError) as exc:
        status = "faulted"
        reason = str(exc)
      self._control_inflight.remove(command_id)
      completions.append(DeviceControlCompletion(command_id, status, reason, cycle))
      if status == "completed":
        self._controls_completed += 1
      else:
        self._controls_failed += 1
        self._fault_reason = self._fault_reason or reason or f"control command {command_id} {status}"
    return tuple(completions)

  def note_dependencies(self, events: tuple[str, ...], cycle: int) -> None:
    self.group.profile_controller.note_dependencies(self._control_owner, events, cycle)

  def note_await(self, instruction_id: str, events: tuple[str, ...], cycle: int) -> None:
    if not instruction_id or not events or len(events) != len(set(events)):
      raise ValueError("ordinary await proof requires an instruction ID and unique events")
    self.group.profile_controller.note_await(self._control_owner, instruction_id, events, cycle)

  def snapshot(self) -> dict:
    return {
      "configuration": {
        "active_context_capacity": self.active_context_capacity,
        "context_pending_capacity": self.context_pending_capacity,
      },
      "run_generation": self._run_generation,
      "active": self.active_count,
      "pending": len(self._pending),
      "active_peak": self._slot_peak,
      "pending_peak": self._pending_peak,
      "submitted": self._submitted,
      "completed": self._completed,
      "failed": self._failed,
      "backpressure": self._backpressure,
      "controls": {
        "inflight": len(self._control_inflight),
        "submitted": self._controls_submitted,
        "completed": self._controls_completed,
        "failed": self._controls_failed,
        "backpressure": self._control_backpressure,
      },
      "request_records": [self._records[request_id].snapshot() for request_id in sorted(self._records)],
    }
