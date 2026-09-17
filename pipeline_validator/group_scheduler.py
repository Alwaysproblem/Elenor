"""Finite shared ready-action scheduler for one Tile Group."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .config import GroupSchedulerConfig
from .execution_ir import ExecGroupAction, ExecGroupActionOp
from .pmu import PMUCounter
from .runtime.event_table import EventProtocolError, EventStatus

if TYPE_CHECKING:
  from .execution_ir import ExecTileGroupTask
  from .tile_group import TileGroup
  from .tile_group_sequencer import TileGroupSequencer


class IssueStatus(Enum):
  """Atomic result of probing and committing one Group action."""

  ACCEPTED = "accepted"
  BACKPRESSURE = "backpressure"
  FAULT = "fault"


@dataclass(frozen=True)
class IssueResult:
  """Action issue result shared by scheduler and adapters."""

  status: IssueStatus
  asynchronous: bool = False
  completion_event: str = ""
  adapter: str = ""
  reason: str = ""

  def __bool__(self) -> bool:
    return self.status is IssueStatus.ACCEPTED


@dataclass
class RegisteredAction:
  """One action instance resident in the finite Group action table."""

  sequencer: TileGroupSequencer
  ordinal: int
  action: ExecGroupAction
  registered_cycle: int
  issue_cycle: int = -1


class GroupScheduler:
  """One-register/one-issue shared Group scheduling pipeline.

  Completions enter through ``note_completion`` independently of the action
  table, so a full table cannot prevent credit or event reclamation.
  """

  def __init__(self, group: TileGroup, config: GroupSchedulerConfig) -> None:
    self.group = group
    self.config = config
    self.pmu = PMUCounter()
    self._queued: list[RegisteredAction] = []
    self._inflight_by_event: dict[str, tuple[RegisteredAction, str]] = {}
    self._control_inflight: dict[str, RegisteredAction] = {}
    self._register_cursor = 0
    self._scan_cursor = 0
    self._epoch_key: tuple[int, int, int] | None = None
    self._epoch_inflight = 0
    self._adapter_used = {"prefetch": 0, "store": 0, "dispatch": 0}
    self.queued_peak = 0
    self.inflight_peak = 0
    self.active_context_peak = 0
    self.registered_total = 0
    self.issued_total = 0
    self.completed_total = 0
    self.backpressure_total = 0
    self.dependency_wait_candidate_checks = 0
    self.dependency_error_total = 0
    self.protocol_fault_total = 0

  @property
  def queued(self) -> int:
    return len(self._queued)

  @property
  def inflight(self) -> int:
    return len(self._inflight_by_event)

  @property
  def event_version(self) -> int:
    return self.group.event_table.version

  def event_owner(self, sequencer: TileGroupSequencer) -> str:
    return f"{sequencer.context_name}@{sequencer.device_slot}"

  def reserve_context_events(self, sequencer: TileGroupSequencer, task: ExecTileGroupTask) -> IssueResult:
    """Atomically reserve the compiler-proved event frontier, without entries."""

    program = self.group.loaded_program
    if program is None:
      return IssueResult(IssueStatus.FAULT, reason="event admission requires a loaded program")
    budget = program.resource_budgets.get(task.binding_id)
    if budget is None:
      return IssueResult(
        IssueStatus.FAULT, reason=f"Context binding {task.binding_id!r} has no compiled resource budget"
      )
    if budget.event_frontier > self.config.event_capacity:
      return IssueResult(
        IssueStatus.FAULT,
        reason=(
          f"compiled event frontier {budget.event_frontier} exceeds "
          f"Group capacity {self.config.event_capacity}"
        ),
      )
    try:
      accepted = self.group.event_table.reserve_quota(
        budget.event_frontier,
        task.event_uses,
        owner=self.event_owner(sequencer),
        generation=sequencer.context_launch_generation,
      )
    except EventProtocolError as exc:
      return IssueResult(IssueStatus.FAULT, reason=str(exc))
    if not accepted:
      return IssueResult(IssueStatus.BACKPRESSURE, reason="event quota unavailable")
    return IssueResult(IssueStatus.ACCEPTED)

  def cancel_context_events(self, sequencer: TileGroupSequencer) -> None:
    try:
      self.group.event_table.cancel_reservation(
        owner=self.event_owner(sequencer), generation=sequencer.context_launch_generation
      )
    except EventProtocolError:
      self.protocol_fault_total += 1
      raise

  def retire_context_events(self, sequencer: TileGroupSequencer, cycle: int) -> None:
    self.group.event_table.release_owner(
      owner=self.event_owner(sequencer), generation=sequencer.context_launch_generation, cycle=cycle
    )

  def step(self, contexts: list[TileGroupSequencer], cycle: int) -> None:
    """Issue from old entries, then register at most one new entry."""

    self._poll_controls(cycle)

    self.pmu.add_cycle("group_scheduler_cycles")
    self.pmu.add_cycle("group_scheduler_queued", self.queued)
    self.pmu.add_cycle("group_scheduler_inflight", self.inflight)
    self.active_context_peak = max(self.active_context_peak, sum(not context.done for context in contexts))
    issued = self._issue_one(cycle)
    if not issued:
      self.pmu.add_cycle("group_issue_idle")
    registered = self._register_one(contexts, cycle)
    if not registered:
      self.pmu.add_cycle("group_register_idle")
    self._trace_occupancy(cycle)

  def _register_one(self, contexts: list[TileGroupSequencer], cycle: int) -> bool:
    if len(self._queued) >= self.config.action_capacity:
      self.pmu.add_cycle("group_action_table_full")
      return False
    live = [context for context in contexts if context.can_register_action()]
    if not live:
      return False
    start = self._register_cursor % len(live)
    for offset in range(len(live)):
      index = (start + offset) % len(live)
      sequencer = live[index]
      if sequencer.queued_count >= self.config.context_action_quota:
        self.pmu.add_cycle("group_context_quota_full")
        continue
      ordinal, action = sequencer.peek_registration()
      dependencies = sequencer.dependencies_for(action)
      outputs = action.output_events
      try:
        self._preflight_registration(sequencer, action, dependencies)
        self.group.event_table.reserve_outputs(
          outputs, owner=self.event_owner(sequencer), generation=sequencer.context_launch_generation
        )
        for event in outputs:
          self.group.event_table.bind_producer(
            event,
            owner=self.event_owner(sequencer),
            generation=sequencer.context_launch_generation,
            producer_id=ordinal,
          )
        for event in dependencies:
          self.group.event_table.add_consumer(
            event, owner=self.event_owner(sequencer), generation=sequencer.context_launch_generation
          )
      except EventProtocolError as exc:
        self._fault_context(sequencer, str(exc), cycle)
        self.protocol_fault_total += 1
        self._register_cursor = index + 1
        return False
      record = RegisteredAction(sequencer=sequencer, ordinal=ordinal, action=action, registered_cycle=cycle)
      self._queued.append(record)
      sequencer.note_registered(record)
      self._register_cursor = index + 1
      self.queued_peak = max(self.queued_peak, len(self._queued))
      self.registered_total += 1
      self.pmu.add_event("group_action_registered")
      self._trace_action("group_action_register", record, cycle)
      return True
    self._register_cursor = start + 1
    return False

  def _preflight_registration(
    self, sequencer: TileGroupSequencer, action: ExecGroupAction, dependencies: tuple[str, ...]
  ) -> None:
    self.group.event_table.preflight_action(
      action.output_events,
      dependencies,
      owner=self.event_owner(sequencer),
      generation=sequencer.context_launch_generation,
    )

  def _issue_one(self, cycle: int) -> bool:
    if not self._queued:
      return False
    scan_count = min(self.config.scan_width, len(self._queued))
    start = self._scan_cursor % len(self._queued)
    scanned = [self._queued[(start + offset) % len(self._queued)] for offset in range(scan_count)]
    self._scan_cursor = start + scan_count
    if self.config.policy == "s0":
      heads: dict[int, int] = {}
      for queued in self._queued:
        key = id(queued.sequencer)
        heads[key] = min(heads.get(key, queued.ordinal), queued.ordinal)
      excluded = [record for record in scanned if heads[id(record.sequencer)] != record.ordinal]
      scanned = [record for record in scanned if heads[id(record.sequencer)] == record.ordinal]
      if any(self._resource_eligible(record) for record in excluded) and not any(
        self._resource_eligible(record) for record in scanned
      ):
        self.pmu.add_cycle("group_ordering_stall")
    for record in scanned:
      status = self._dependency_status(record)
      if status is EventStatus.PENDING:
        self.dependency_wait_candidate_checks += 1
        self.pmu.add_event("group_dependency_wait_candidate_checks")
        continue
      if status is not EventStatus.DONE:
        self.dependency_error_total += 1
        self._fault_context(
          record.sequencer, f"dependency of action {record.ordinal} completed with {status.name}", cycle
        )
        return False
      if record.action.op is ExecGroupActionOp.BARRIER_GROUP:
        if not record.sequencer.barrier_ready(record.ordinal):
          self.pmu.add_cycle("group_barrier_wait")
          continue
      adapter = self._adapter_for(record.action)
      if not self._has_credit(record.action, adapter):
        self.backpressure_total += 1
        credit = adapter or "inflight"
        self.pmu.add_cycle(f"group_{credit}_credit_wait")
        continue
      if not self._epoch_allows(record):
        self.backpressure_total += 1
        self.pmu.add_cycle("group_epoch_wait")
        continue
      if self._is_async(record.action):
        expected_completion = record.action.dst
        if not expected_completion:
          self._fault_context(
            record.sequencer, f"async action {record.ordinal} has no reserved primary event", cycle
          )
          return False
        if expected_completion in self._inflight_by_event:
          self._fault_context(
            record.sequencer, f"duplicate inflight completion event {expected_completion!r}", cycle
          )
          return False
      result = record.sequencer.issue_registered(record, cycle)
      if result.status is IssueStatus.BACKPRESSURE:
        self.backpressure_total += 1
        self.pmu.add_cycle("group_action_backpressure")
        continue
      self._trace_action(
        "group_action_issue", record, cycle, status=result.status.value, reason=result.reason
      )
      if result.status is IssueStatus.FAULT:
        self._fault_context(record.sequencer, result.reason, cycle)
        return False
      dependency_fault = ""
      try:
        self.group.event_table.consume_dependencies(
          record.sequencer.dependencies_for(record.action),
          owner=self.event_owner(record.sequencer),
          generation=record.sequencer.context_launch_generation,
          cycle=cycle,
        )
      except EventProtocolError as exc:
        # The action has already been accepted.  Preserve its independent
        # inflight/control bookkeeping, then fault the Context below.
        dependency_fault = str(exc)
        self.protocol_fault_total += 1
      self._queued.remove(record)
      record.issue_cycle = cycle
      self.issued_total += 1
      self.pmu.add_event("group_action_issued")
      if result.adapter == "control":
        self._control_inflight[result.completion_event] = record
        record.sequencer.note_issued(record, asynchronous=True)
      elif result.asynchronous:
        completion_event = result.completion_event
        if not completion_event:
          self._fault_context(record.sequencer, "async action has no completion event", cycle)
          return False
        if completion_event in self._inflight_by_event:
          self._fault_context(
            record.sequencer, f"duplicate inflight completion event {completion_event!r}", cycle
          )
          return False
        if adapter:
          self._adapter_used[adapter] += 1
        self._inflight_by_event[completion_event] = (record, adapter)
        self.inflight_peak = max(self.inflight_peak, len(self._inflight_by_event))
        if record.action.op is ExecGroupActionOp.DISPATCH_ROLE:
          self._acquire_epoch(record)
        record.sequencer.note_issued(record, asynchronous=True)
      else:
        record.sequencer.note_issued(record, asynchronous=False)
        record.sequencer.note_action_completed(record)
        self.completed_total += 1
        self.pmu.add_event("group_action_completed")
        self._trace_action("group_action_complete", record, cycle, status="done")
      if dependency_fault:
        self._fault_context(record.sequencer, dependency_fault, cycle)
        return False
      return True
    return False

  def _poll_controls(self, cycle: int) -> None:
    for command_id, record in tuple(self._control_inflight.items()):
      status = self.group.profile_controller.status(command_id)
      if status in ("pending", "running"):
        continue
      del self._control_inflight[command_id]
      record.sequencer.note_action_completed(record)
      self.completed_total += 1
      self._trace_action("group_action_complete", record, cycle, status=status)
      if status != "completed":
        self._fault_context(record.sequencer, f"control {command_id} {status}", cycle)

  def _dependency_status(self, record: RegisteredAction) -> EventStatus:
    dependencies = record.sequencer.dependencies_for(record.action)
    for event in dependencies:
      entry = self.group.event_table.get(event)
      if entry is None or not entry.producer_bound:
        return EventStatus.ERROR
      if entry.status is EventStatus.PENDING:
        return EventStatus.PENDING
      if entry.status is not EventStatus.DONE:
        return entry.status
    return EventStatus.DONE

  def _adapter_for(self, action: ExecGroupAction) -> str:
    if action.op is ExecGroupActionOp.DMA_PREFETCH:
      return "prefetch"
    if action.op is ExecGroupActionOp.DMA_STORE:
      return "store"
    if action.op is ExecGroupActionOp.DISPATCH_ROLE:
      return "dispatch"
    return ""

  @staticmethod
  def _is_async(action: ExecGroupAction) -> bool:
    return action.op in (
      ExecGroupActionOp.DMA_PREFETCH,
      ExecGroupActionOp.DMA_STORE,
      ExecGroupActionOp.DISPATCH_ROLE,
      ExecGroupActionOp.COLLECTIVE_RUN,
    )

  def _has_credit(self, action: ExecGroupAction, adapter: str) -> bool:
    if self._is_async(action) and self.inflight >= self.config.inflight_capacity:
      return False
    if not adapter:
      return True
    capacity = getattr(self.config, f"{adapter}_capacity")
    return self._adapter_used[adapter] < capacity

  def _resource_eligible(self, record: RegisteredAction) -> bool:
    if self._dependency_status(record) is not EventStatus.DONE:
      return False
    if record.action.op is ExecGroupActionOp.BARRIER_GROUP and not record.sequencer.barrier_ready(
      record.ordinal
    ):
      return False
    adapter = self._adapter_for(record.action)
    return self._has_credit(record.action, adapter) and self._epoch_allows(record)

  def _epoch_allows(self, record: RegisteredAction) -> bool:
    if (
      self.config.epoch_policy != "same_program" or record.action.op is not ExecGroupActionOp.DISPATCH_ROLE
    ):
      return True
    key = record.sequencer.dispatch_program_key(record.action)
    return self._epoch_key is None or self._epoch_key == key

  def _acquire_epoch(self, record: RegisteredAction) -> None:
    if self.config.epoch_policy != "same_program":
      return
    key = record.sequencer.dispatch_program_key(record.action)
    if self._epoch_key is None:
      self._epoch_key = key
    elif self._epoch_key != key:
      raise RuntimeError("dispatch epoch changed after eligibility")
    self._epoch_inflight += 1

  def note_completion(
    self, sequencer: TileGroupSequencer, event: str, cycle: int, status: EventStatus
  ) -> None:
    """Return inflight/adapter credit on the real primary completion."""

    binding = self._inflight_by_event.get(event)
    if binding is None:
      return
    record, adapter = binding
    if record.sequencer is not sequencer:
      self._fault_context(sequencer, f"foreign completion for {event!r}", cycle)
      return
    if adapter and self._adapter_used[adapter] <= 0:
      self._fault_context(sequencer, f"{adapter} adapter credit underflow", cycle)
      return
    if (
      record.action.op is ExecGroupActionOp.DISPATCH_ROLE
      and self.config.epoch_policy == "same_program"
      and self._epoch_inflight <= 0
    ):
      self._fault_context(sequencer, "dispatch epoch reference underflow", cycle)
      return
    self._inflight_by_event.pop(event)
    if adapter:
      self._adapter_used[adapter] -= 1
    if record.action.op is ExecGroupActionOp.DISPATCH_ROLE:
      self._release_epoch()
    sequencer.note_action_completed(record)
    self.completed_total += 1
    self.pmu.add_event("group_action_completed")
    self._trace_action("group_action_complete", record, cycle, status=status.name.lower())
    if status is not EventStatus.DONE:
      self._fault_context(
        sequencer, f"action {record.ordinal} completion {event!r} status={status.name}", cycle
      )

  def _release_epoch(self) -> None:
    if self.config.epoch_policy != "same_program":
      return
    if self._epoch_inflight <= 0:
      raise RuntimeError("dispatch epoch reference underflow")
    self._epoch_inflight -= 1
    if self._epoch_inflight == 0:
      self._epoch_key = None

  def cancel_context(self, sequencer: TileGroupSequencer, cycle: int) -> None:
    """Remove unissued metadata; already accepted work drains independently."""

    for command_id, record in self._control_inflight.items():
      if record.sequencer is sequencer:
        self.group.profile_controller.request_cancel(command_id, cycle)
    cancelled = [record for record in self._queued if record.sequencer is sequencer]
    owner = self.event_owner(sequencer)
    generation = sequencer.context_launch_generation
    try:
      # First terminalize outputs that can no longer receive a completion.
      # Current consumer pins prevent premature reclamation.
      for record in cancelled:
        self.group.event_table.cancel_outputs(
          record.action.output_events,
          owner=owner,
          generation=generation,
          producer_id=record.ordinal,
          cycle=cycle,
        )
      for record in cancelled:
        self.group.event_table.cancel_dependencies(
          record.sequencer.dependencies_for(record.action), owner=owner, generation=generation, cycle=cycle
        )
      self.group.event_table.abandon_future_uses(owner=owner, generation=generation, cycle=cycle)
    except EventProtocolError:
      self.protocol_fault_total += 1
      raise
    self._queued = [record for record in self._queued if record.sequencer is not sequencer]
    for record in cancelled:
      sequencer.note_action_cancelled(record)
      self._trace_action("group_action_complete", record, cycle, status="cancelled")

  def abort_inflight(self, cycle: int) -> None:
    """Return scheduler metadata credit after reset cancelled real adapters."""

    for record, _adapter in self._inflight_by_event.values():
      self._trace_action("group_action_complete", record, cycle, status="reset")
    self._inflight_by_event.clear()
    self._adapter_used = {"prefetch": 0, "store": 0, "dispatch": 0}
    self._epoch_key = None
    self._epoch_inflight = 0

  def _fault_context(self, sequencer: TileGroupSequencer, reason: str, cycle: int) -> None:
    self.group.on_scheduler_fault(sequencer, reason, cycle)
    self.cancel_context(sequencer, cycle)

  def _trace_action(self, name: str, record: RegisteredAction, cycle: int, **extra) -> None:
    tracer = self.group.tracer
    if tracer is None:
      return
    tracer.instant(
      "TileGroup",
      "Scheduler:Control",
      name,
      cycle,
      {
        "context": record.sequencer.context_name,
        "launch_generation": record.sequencer.context_launch_generation,
        "ordinal": record.ordinal,
        "kind": record.action.op.value,
        "cycle": cycle,
        **extra,
      },
    )
    if name == "group_action_issue" and not record.sequencer._first_action_emitted:
      tracer.instant(
        "TileGroup",
        "Scheduler:L2",
        "context_first_action",
        cycle,
        {
          "context": record.sequencer.context_name,
          "slot": record.sequencer.device_slot,
          "launch_generation": record.sequencer.context_launch_generation,
          "cycle": cycle,
        },
      )

  def _trace_occupancy(self, cycle: int) -> None:
    tracer = self.group.tracer
    if tracer is None:
      return
    tracer.counter_if_changed(
      "TileGroup", "group_action_queued", cycle, self.queued, "actions", thread="Scheduler:Control"
    )
    tracer.counter_if_changed(
      "TileGroup", "group_action_inflight", cycle, self.inflight, "actions", thread="Scheduler:Control"
    )
    tracer.counter_if_changed(
      "TileGroup",
      "group_event_reserved",
      cycle,
      self.group.event_table.reserved,
      "events",
      thread="Scheduler:Control",
    )
    tracer.counter_if_changed(
      "TileGroup",
      "group_event_active",
      cycle,
      self.group.event_table.active,
      "events",
      thread="Scheduler:Control",
    )

  def snapshot(self) -> dict:
    return {
      "configuration": asdict(self.config),
      "queued": self.queued,
      "queued_peak": self.queued_peak,
      "inflight": self.inflight,
      "inflight_peak": self.inflight_peak,
      "event_reserved": self.group.event_table.reserved,
      "event_peak": self.group.event_table.peak_reserved,
      "event_active": self.group.event_table.active,
      "event_active_peak": self.group.event_table.peak_active,
      "active_context_peak": self.active_context_peak,
      "registered": self.registered_total,
      "issued": self.issued_total,
      "completed": self.completed_total,
      "backpressure": self.backpressure_total,
      "dependency_wait_candidate_checks": self.dependency_wait_candidate_checks,
      "dependency_errors": self.dependency_error_total,
      "protocol_faults": self.protocol_fault_total,
      "adapter_used": dict(self._adapter_used),
      "epoch_key": self._epoch_key,
      "epoch_inflight": self._epoch_inflight,
    }

  def reset(self) -> None:
    self._queued.clear()
    self._inflight_by_event.clear()
    if self._control_inflight:
      raise RuntimeError("scheduler reset before configuration control retirement")
    self._register_cursor = 0
    self._scan_cursor = 0
    self._epoch_key = None
    self._epoch_inflight = 0
    self._adapter_used = {"prefetch": 0, "store": 0, "dispatch": 0}
    self.queued_peak = 0
    self.inflight_peak = 0
    self.active_context_peak = 0
    self.registered_total = 0
    self.issued_total = 0
    self.completed_total = 0
    self.backpressure_total = 0
    self.dependency_wait_candidate_checks = 0
    self.dependency_error_total = 0
    self.protocol_fault_total = 0
    self.pmu.reset()
