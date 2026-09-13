"""Per-context state for the shared Tile Group ready-action scheduler."""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

from .execution_ir import (
  ContextAdmissionStatus,
  ExecDispatchRequest,
  ExecGroupAction,
  ExecGroupActionOp,
  ExecReleaseRequest,
  ExecStreamDesc,
  ExecTileGroupTask,
  GridInstanceId,
)
from .group_scheduler import IssueResult, IssueStatus, RegisteredAction
from .memory.allocator import MemoryInvariantError
from .pmu import PMUCounter
from .runtime import EventStatus

if TYPE_CHECKING:
  from .tile_group import TileGroup


class TileGroupSequencer:
  """Launch-scoped context state; issue bandwidth is owned by the Group."""

  def __init__(self, group: TileGroup):
    self.group = group
    self.cfg = group.cfg
    self.pmu = PMUCounter()
    self.submission_pc = 0
    self.task: ExecTileGroupTask | None = None
    self._events_done: set[str] = set()
    self._event_errors: dict[str, EventStatus] = {}
    self._role_events: dict[int, str] = {}
    self._issued_role_events: set[str] = set()
    self._issued_phase_events: set[str] = set()
    self._queued_actions: dict[int, RegisteredAction] = {}
    self._inflight_actions: dict[int, RegisteredAction] = {}
    self._registration_fence: int | None = None
    self.submission_closed = False
    self.done = False
    self.faulted = False
    self.fault_reason = ""
    self.owned_queue_ids: set[int] = set()
    self._outstanding_jobs = 0
    self.context_launch_generation = 0
    self.context_name = ""
    self.device_slot = 0
    self.admission_status = ContextAdmissionStatus.PREPARED
    self.admission_wait_start_cycle: int | None = None
    self.admission_retry_count = 0
    self.formal_bindings: dict[str, str] = {}
    self._first_action_emitted = False

  @property
  def queued_count(self) -> int:
    return len(self._queued_actions)

  @property
  def inflight_count(self) -> int:
    return len(self._inflight_actions)

  def note_job_started(self) -> None:
    self._outstanding_jobs += 1

  def note_job_done(self) -> None:
    if self._outstanding_jobs <= 0:
      raise RuntimeError("sequencer outstanding job underflow")
    self._outstanding_jobs -= 1

  def grid_id(self, dispatch_ordinal: int) -> GridInstanceId:
    return GridInstanceId(
      context_name=self.context_name,
      device_slot=self.device_slot,
      launch_generation=self.context_launch_generation,
      dispatch_ordinal=dispatch_ordinal,
    )

  def load(self, task: ExecTileGroupTask) -> None:
    self.task = task
    self.submission_pc = 0
    self._events_done.clear()
    self._event_errors.clear()
    self._role_events.clear()
    self._issued_role_events.clear()
    self._issued_phase_events.clear()
    self._queued_actions.clear()
    self._inflight_actions.clear()
    self._registration_fence = None
    self._outstanding_jobs = 0
    self.submission_closed = not task.actions
    self.done = False
    self.faulted = False
    self.fault_reason = ""
    self.owned_queue_ids.clear()
    self.admission_status = ContextAdmissionStatus.PREPARED
    self.admission_wait_start_cycle = None
    self.admission_retry_count = 0
    self._first_action_emitted = False

  def can_register_action(self) -> bool:
    return (
      not self.done
      and not self.faulted
      and self.task is not None
      and self.admission_status is ContextAdmissionStatus.ACTIVE
      and not self.submission_closed
      and self._registration_fence is None
    )

  def peek_registration(self) -> tuple[int, ExecGroupAction]:
    if not self.can_register_action() or self.task is None:
      raise RuntimeError("sequencer has no registerable action")
    return self.submission_pc, self.task.actions[self.submission_pc]

  def dependencies_for(self, action: ExecGroupAction) -> tuple[str, ...]:
    dependencies = list(action.dependencies)
    if action.op is ExecGroupActionOp.WAIT_EVENT:
      dependencies.extend(arg for arg in action.args if isinstance(arg, str))
    elif action.op is ExecGroupActionOp.RELEASE_L2:
      request = action.args[0]
      if isinstance(request, ExecReleaseRequest):
        dependencies.extend(request.dependency_events)
    return tuple(dict.fromkeys(dependencies))

  def note_registered(self, record: RegisteredAction) -> None:
    if record.ordinal != self.submission_pc:
      raise RuntimeError("non-sequential Group action registration")
    self._queued_actions[record.ordinal] = record
    if record.action.op in (ExecGroupActionOp.WAIT_EVENT, ExecGroupActionOp.BARRIER_GROUP):
      self._registration_fence = record.ordinal
    self.submission_pc += 1
    if self.task is not None and self.submission_pc >= len(self.task.actions):
      self.submission_closed = True

  def note_issued(self, record: RegisteredAction, *, asynchronous: bool) -> None:
    if self._queued_actions.pop(record.ordinal, None) is not record:
      raise RuntimeError("issued action is not queued by its owner")
    if asynchronous:
      self._inflight_actions[record.ordinal] = record
    if not self._first_action_emitted:
      self._first_action_emitted = True

  def note_action_completed(self, record: RegisteredAction) -> None:
    self._inflight_actions.pop(record.ordinal, None)
    if self._registration_fence == record.ordinal:
      self._registration_fence = None

  def note_action_cancelled(self, record: RegisteredAction) -> None:
    self._queued_actions.pop(record.ordinal, None)
    if self._registration_fence == record.ordinal:
      self._registration_fence = None

  def barrier_ready(self, ordinal: int) -> bool:
    return not any(index < ordinal for index in self._queued_actions if index != ordinal) and not any(
      index < ordinal for index in self._inflight_actions
    )

  def dispatch_program_key(self, action: ExecGroupAction) -> tuple[int, int, int]:
    request = action.args[0]
    if not isinstance(request, ExecDispatchRequest) or self.task is None:
      raise RuntimeError("dispatch action has invalid request")
    binding = self.task.role_bindings.get(request.role_id)
    if binding is None:
      raise RuntimeError(f"unknown role_id {request.role_id}")
    program = binding.tile_program
    fallback = zlib.crc32(program.name.encode()) & 0xFFFFFFFF
    return (program.program_id or fallback, program.version, program.program_hash or fallback)

  def issue_registered(self, record: RegisteredAction, cycle: int) -> IssueResult:
    """Commit one already-eligible action with no hidden retry queue."""

    action = record.action
    try:
      if action.op is ExecGroupActionOp.INIT_STREAM:
        qid, depth, producer_mask, consumer_mask = action.args
        self.group.init_stream(
          ExecStreamDesc(
            queue_id=qid, depth=depth, producer_mask=producer_mask, consumer_mask=consumer_mask
          )
        )
        self.pmu.add_event("tgs_init_stream")
        return IssueResult(IssueStatus.ACCEPTED)

      if action.op in (ExecGroupActionOp.DMA_PREFETCH, ExecGroupActionOp.DMA_STORE):
        if action.dst is None:
          return IssueResult(IssueStatus.FAULT, reason=f"{action.op.name} requires a completion event")
        desc_id, transfer = action.args[0], action.args[1]
        op = "dma.prefetch" if action.op is ExecGroupActionOp.DMA_PREFETCH else "dma.store"
        accepted = self.group.submit_group_transfer(
          op, action.dst, cycle, desc_id, transfer, sequencer=self
        )
        if not accepted:
          return IssueResult(IssueStatus.FAULT, reason=f"invalid or stale memory view for {op} {desc_id}")
        self.pmu.add_event(
          "tgs_dma_prefetch" if action.op is ExecGroupActionOp.DMA_PREFETCH else "tgs_dma_store"
        )
        return IssueResult(
          IssueStatus.ACCEPTED,
          asynchronous=True,
          completion_event=action.dst,
          adapter=("prefetch" if action.op is ExecGroupActionOp.DMA_PREFETCH else "store"),
        )

      if action.op is ExecGroupActionOp.DISPATCH_ROLE:
        request = action.args[0]
        if not isinstance(request, ExecDispatchRequest) or self.task is None:
          return IssueResult(IssueStatus.FAULT, reason="invalid dispatch request")
        binding = self.task.role_bindings.get(request.role_id)
        if binding is None:
          return IssueResult(IssueStatus.FAULT, reason=f"unknown role_id {request.role_id}")
        event = action.dst or f"ev_role{request.role_id}"
        result = self.group.dispatch_role(binding, cycle, request=request, event_id=event, sequencer=self)
        if not result:
          return result
        self._role_events[request.role_id] = event
        self._issued_role_events.add(event)
        if request.input_released_event:
          self._issued_phase_events.add(request.input_released_event)
        if request.output_ready_event:
          self._issued_phase_events.add(request.output_ready_event)
        self.pmu.add_event("tgs_dispatch_role")
        return IssueResult(
          IssueStatus.ACCEPTED, asynchronous=True, completion_event=event, adapter="dispatch"
        )

      if action.op is ExecGroupActionOp.WAIT_EVENT:
        self.pmu.add_event("tgs_wait_event")
        return IssueResult(IssueStatus.ACCEPTED)

      if action.op is ExecGroupActionOp.BARRIER_GROUP:
        self.pmu.add_event("tgs_barrier")
        return IssueResult(IssueStatus.ACCEPTED)

      if action.op is ExecGroupActionOp.COLLECTIVE_RUN:
        if action.dst is None:
          return IssueResult(IssueStatus.FAULT, reason="COLLECTIVE_RUN requires a completion event")
        desc_id, op_name, bytes_total, participant_mask = action.args
        self.group.schedule_collective(
          desc_id, action.dst, op_name, bytes_total, participant_mask, cycle, sequencer=self
        )
        self.pmu.add_event("tgs_collective_run")
        return IssueResult(IssueStatus.ACCEPTED, asynchronous=True, completion_event=action.dst)

      if action.op is ExecGroupActionOp.SIGNAL_EVENT:
        event = action.args[0]
        if not isinstance(event, str):
          return IssueResult(IssueStatus.FAULT, reason="invalid signal event")
        if not self.notify_event(event, cycle):
          return IssueResult(IssueStatus.FAULT, reason=f"event {event!r} rejected signal")
        self.pmu.add_event("tgs_signal_event")
        return IssueResult(IssueStatus.ACCEPTED)

      if action.op is ExecGroupActionOp.RELEASE_L2:
        request = action.args[0]
        if not isinstance(request, ExecReleaseRequest):
          return IssueResult(IssueStatus.FAULT, reason="invalid L2 release request")
        self.group.release_l2(request, sequencer=self, cycle=cycle)
        self.pmu.add_event("tgs_release_l2")
        return IssueResult(IssueStatus.ACCEPTED)
    except (MemoryInvariantError, ValueError, RuntimeError) as exc:
      self.pmu.add_event("group_action_invariant_fault")
      return IssueResult(IssueStatus.FAULT, reason=str(exc))
    return IssueResult(IssueStatus.FAULT, reason=f"unsupported Group action {action.op.value}")

  def notify_event(
    self, event_id: str, cycle: int = 0, status: EventStatus = EventStatus.DONE, error_code: int = 0
  ) -> bool:
    """Publish one completion with exact owner/generation validation."""

    entry = self.group.event_table.get(event_id)
    if entry is None:
      reason = f"completion for unreserved event {event_id!r}"
      self.group.on_scheduler_fault(self, reason, cycle)
      self.group.scheduler.cancel_context(self, cycle)
      return False
    accepted = self.group.event_table.signal(
      event_id,
      status,
      producer_id=entry.producer_id,
      cycle=cycle,
      error_code=error_code,
      expected_owner=self.group.scheduler.event_owner(self),
      expected_generation=self.context_launch_generation,
      expected_sequence=entry.sequence,
    )
    if not accepted:
      reason = f"duplicate, stale, or foreign completion {event_id!r}"
      self.group.on_scheduler_fault(self, reason, cycle)
      self.group.scheduler.cancel_context(self, cycle)
      return False
    if status is EventStatus.DONE:
      self._events_done.add(event_id)
    else:
      self._event_errors[event_id] = status
    if status is EventStatus.DONE:
      self.group.note_l2_protocol_event(self, event_id, cycle)
    self.group.scheduler.note_completion(self, event_id, cycle, status)
    if status is not EventStatus.DONE:
      self.mark_fault(f"event {event_id!r} status={status.name}")
      self.group.scheduler.cancel_context(self, cycle)
    return True

  def mark_fault(self, reason: str) -> None:
    if not self.faulted:
      self.fault_reason = reason
    self.faulted = True
    self.submission_closed = True

  def maybe_finish(self) -> None:
    if self.done or self.task is None:
      return
    if not self.submission_closed or self.queued_count or self.inflight_count:
      return
    if self._outstanding_jobs != 0:
      return
    if not self.faulted:
      if not all(event in self._events_done for event in self._issued_role_events):
        return
      if not all(event in self._events_done for event in self._issued_phase_events):
        return
    if not self.group.context_cleanup_ready(self):
      return
    self.done = True

  def step(self, cycle: int) -> tuple[int, str] | None:
    """Compatibility hook; shared issue is exclusively driven by TileGroup."""

    self.pmu.add_cycle("total")
    self.maybe_finish()
    return None

  def abort_actions(self) -> None:
    """Drop scheduler metadata after reset has cancelled real work."""

    self._queued_actions.clear()
    self._inflight_actions.clear()
    self._registration_fence = None
    self._outstanding_jobs = 0
    self.submission_closed = True

  def reset(self) -> None:
    self.submission_pc = 0
    self.task = None
    self._events_done.clear()
    self._event_errors.clear()
    self._role_events.clear()
    self._issued_role_events.clear()
    self._issued_phase_events.clear()
    self._queued_actions.clear()
    self._inflight_actions.clear()
    self._registration_fence = None
    self.submission_closed = False
    self.done = False
    self.faulted = False
    self.fault_reason = ""
    self.owned_queue_ids.clear()
    self._outstanding_jobs = 0
    self.admission_status = ContextAdmissionStatus.PREPARED
    self.admission_wait_start_cycle = None
    self.admission_retry_count = 0
    self.formal_bindings.clear()
    self._first_action_emitted = False
    self.pmu.reset()
