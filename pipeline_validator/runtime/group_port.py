"""Concrete message adapter between the CPU model and one TileGroup."""

from __future__ import annotations

from dataclasses import dataclass

from ..device import DeviceCompletion, DeviceCompletionStatus, DeviceLaunchRequest, DevicePort
from ..execution_ir import ContextAdmissionStatus
from ..tile_group import TileGroup
from ..tile_group_sequencer import TileGroupSequencer


@dataclass
class _HardwareSlot:
  slot_index: int
  request_id: int
  context_name: str
  sequencer: TileGroupSequencer
  submit_cycle: int
  active_cycle: int | None = None
  completion_cycle: int | None = None
  status: DeviceCompletionStatus | None = None
  reason: str = ""

  def snapshot(self) -> dict:
    return {
      "slot_index": self.slot_index,
      "request_id": self.request_id,
      "context": self.context_name,
      "submit_cycle": self.submit_cycle,
      "active_cycle": self.active_cycle,
      "completion_cycle": self.completion_cycle,
      "status": None if self.status is None else self.status.value,
      "reason": self.reason,
    }


class GroupPortAdapter(DevicePort):
  """Finite launch port for one concrete TileGroup.

  Hardware slot indices are private adapter identities.  They are passed to
  ``TileGroup.load_context_task`` only to namespace a launch; neither the CPU
  controller nor the Group may interpret them as physical Tile context IDs.
  A slot remains occupied while the Group is temporarily waiting for finite
  event metadata or L2 memory admission.
  """

  def __init__(self, group: TileGroup, active_context_capacity: int) -> None:
    if (
      not isinstance(active_context_capacity, int)
      or isinstance(active_context_capacity, bool)
      or active_context_capacity < 1
    ):
      raise ValueError("active_context_capacity must be a positive integer")
    self.group = group
    self.active_context_capacity = active_context_capacity
    self._slots: list[_HardwareSlot | None] = [None for _ in range(active_context_capacity)]
    self._ready_completions: list[DeviceCompletion] = []
    self._seen_request_ids: set[int] = set()
    self._records: dict[int, _HardwareSlot] = {}
    self._slot_peak = 0
    self._submitted = 0
    self._completed = 0
    self._failed = 0
    self._backpressure = 0
    self._fault_reason: str | None = None

  @property
  def active_count(self) -> int:
    return sum(slot is not None for slot in self._slots)

  def try_submit(self, request: DeviceLaunchRequest, cycle: int) -> bool:
    if request.request_id in self._seen_request_ids:
      raise ValueError(f"duplicate device request_id {request.request_id}")

    if (
      request.group_affinity is not None and not 0 <= request.group_affinity < self.active_context_capacity
    ):
      self._seen_request_ids.add(request.request_id)
      reason = (
        f"request {request.request_id} targets unavailable Group context slot {request.group_affinity}"
      )
      self._fault_reason = self._fault_reason or reason
      self._ready_completions.append(
        DeviceCompletion(
          request_id=request.request_id, status=DeviceCompletionStatus.ERROR, reason=reason, cycle=cycle
        )
      )
      self._submitted += 1
      self._failed += 1
      return True

    if (
      self._fault_reason is not None
      or any(slot is not None and slot.sequencer.faulted for slot in self._slots)
      or (
        self.group.runtime_enabled
        and (self.group.reset_domain.is_active or self.group.reset_domain.is_done)
      )
    ):
      self._backpressure += 1
      return False

    if not self.group.can_accept_context_launch():
      self._backpressure += 1
      return False

    slot_index = request.group_affinity
    if slot_index is None:
      slot_index = next((index for index, slot in enumerate(self._slots) if slot is None), None)
    elif self._slots[slot_index] is not None:
      self._backpressure += 1
      return False
    if slot_index is None:
      self._backpressure += 1
      return False

    self._seen_request_ids.add(request.request_id)
    try:
      sequencer = self.group.load_context_task(
        request.task,
        slot_index=slot_index,
        context_name=request.context_name,
        input_bindings=request.global_bindings,
        formal_bindings=dict(request.formal_bindings),
        cycle=cycle,
      )
    except (RuntimeError, ValueError) as exc:
      reason = str(exc)
      self._fault_reason = self._fault_reason or reason
      self._ready_completions.append(
        DeviceCompletion(
          request_id=request.request_id, status=DeviceCompletionStatus.ERROR, reason=reason, cycle=cycle
        )
      )
      self._submitted += 1
      self._failed += 1
      return True

    slot = _HardwareSlot(
      slot_index=slot_index,
      request_id=request.request_id,
      context_name=request.context_name,
      sequencer=sequencer,
      submit_cycle=cycle,
      active_cycle=(cycle if sequencer.admission_status is ContextAdmissionStatus.ACTIVE else None),
    )
    self._slots[slot_index] = slot
    self._records[request.request_id] = slot
    self._submitted += 1
    self._slot_peak = max(self._slot_peak, self.active_count)
    return True

  def poll_completions(self, cycle: int) -> tuple[DeviceCompletion, ...]:
    completions = self._ready_completions
    self._ready_completions = []

    reset_done = self.group.runtime_enabled and self.group.reset_domain.is_done
    reset_active = self.group.runtime_enabled and self.group.reset_domain.is_active
    for slot_index, slot in enumerate(tuple(self._slots)):
      if slot is None:
        continue
      sequencer = slot.sequencer
      if slot.active_cycle is None and sequencer.admission_status is ContextAdmissionStatus.ACTIVE:
        slot.active_cycle = cycle

      if reset_active and not sequencer.faulted:
        continue

      if not sequencer.done and not reset_done:
        continue

      if sequencer.faulted:
        status = DeviceCompletionStatus.ERROR
        reason = sequencer.fault_reason or "Group context failed"
        self._fault_reason = self._fault_reason or reason
      elif reset_done:
        status = DeviceCompletionStatus.ERROR
        reason = self._fault_reason or "Group reset after another launch failed"
      else:
        status = DeviceCompletionStatus.SUCCESS
        reason = ""

      slot.completion_cycle = cycle
      slot.status = status
      slot.reason = reason
      completions.append(
        DeviceCompletion(request_id=slot.request_id, status=status, reason=reason, cycle=cycle)
      )
      self._slots[slot_index] = None
      if status is DeviceCompletionStatus.SUCCESS:
        self._completed += 1
      else:
        self._failed += 1

    return tuple(completions)

  def snapshot(self) -> dict:
    return {
      "configuration": {"active_context_capacity": self.active_context_capacity},
      "active": self.active_count,
      "active_peak": self._slot_peak,
      "submitted": self._submitted,
      "completed": self._completed,
      "failed": self._failed,
      "backpressure": self._backpressure,
      "request_records": [self._records[request_id].snapshot() for request_id in sorted(self._records)],
    }
