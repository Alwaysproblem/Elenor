"""Tile Group Sequencer controller.

Executes an ExecTileGroupTask (design/elenor_tile_group_sequencer/ and
Architecture doc 16.5) on the Tile Group.  The sequencer:

  - inits stream queues,
  - prefetches blocks via Group DMA (modelled as latency),
  - dispatches role bindings (tile_mask -> which tiles run which Tile Program),
  - waits on role events,
  - runs collective/barrier/signal actions, and completes the task.

Like the Tile UCE it is a one-instruction-per-cycle controller with a
pending-wait mechanism.  There is no fetchable group-level program text
and no branch/label opcode: the action list is a flat sequence advanced
by an action index.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from .memory.allocator import MemoryInvariantError
from .pmu import PMUCounter, StallReason
from .runtime import EventStatus, FaultCode

if TYPE_CHECKING:
  from .tile_group import TileGroup


@dataclass
class _GroupTaskWait:
  events: tuple
  started_cycle: int


class TileGroupSequencer:
  """The Tile Group Sequencer: advances a ExecTileGroupTask action by action."""

  def __init__(self, group: TileGroup):
    self.group = group
    self.cfg = group.cfg
    self.pmu = PMUCounter()
    self.action_index = 0
    self.task: ExecTileGroupTask | None = None
    self._events_done: set[str] = set()
    self._pending: _GroupTaskWait | None = None
    # role_id -> completion event id (informational)
    self._role_events: dict[int, str] = {}
    # every grid event this sequencer issued; done is gated until all
    # fire (role re-dispatch overwrites _role_events, not this set).
    self._issued_role_events: set[str] = set()
    self.done = False
    self.faulted = False
    self.fault_reason: str = ""
    # round-robin DMA channel allocation
    self._next_dma_channel: int = 0
    # stream queue ids created by this sequencer's task (namespaced);
    # reclaimed by the TileGroup when this sequencer drains.
    self.owned_queue_ids: set[int] = set()
    # outstanding DMA/collective jobs issued by this sequencer; done is
    # gated until they drain (context_done covers the final store).
    self._outstanding_jobs: int = 0
    # PR 2: this sequencer's launch generation (set by TileGroup at load)
    self.context_launch_generation: int = 0
    # PR 3: phase events issued by this sequencer's dispatches; done is
    # gated until every one fires.
    self._issued_phase_events: set[str] = set()
    # PR 3: launch identity of the context this sequencer executes
    # (context symbol, device slot, launch generation).
    self.context_name: str = ""
    self.device_slot: int = 0
    # PR 3.5: context admission lifecycle + launch-scoped formal bindings
    self.admission_status: ContextAdmissionStatus = (
      ContextAdmissionStatus.PREPARED)
    self.admission_wait_start_cycle: int | None = None
    self.admission_retry_count: int = 0
    self.formal_bindings: dict[str, str] = {}
    # PR 3.5: first-action marker (trace evidence for admission timing)
    self._first_action_emitted: bool = False

  def note_job_started(self) -> None:
    self._outstanding_jobs += 1

  def note_job_done(self) -> None:
    self._outstanding_jobs -= 1

  def grid_id(self, dispatch_ordinal: int) -> GridInstanceId:
    """Identity of one dispatched grid in this launch (PR 3)."""
    return GridInstanceId(
      context_name=self.context_name,
      device_slot=self.device_slot,
      launch_generation=self.context_launch_generation,
      dispatch_ordinal=dispatch_ordinal,
    )

  def load(self, task: ExecTileGroupTask) -> None:
    self.task = task
    self.action_index = 0
    self._events_done.clear()
    self._pending = None
    self._role_events.clear()
    self._issued_role_events.clear()
    self._issued_phase_events.clear()
    self._next_dma_channel = 0
    self.owned_queue_ids.clear()
    self._outstanding_jobs = 0
    self.done = False
    self.faulted = False
    self.fault_reason = ""
    # PR 3.5: admission lifecycle restarts at PREPARED on a fresh load
    self.admission_status = ContextAdmissionStatus.PREPARED
    self.admission_wait_start_cycle = None
    self.admission_retry_count = 0
    self._first_action_emitted = False

  # ---- per-cycle step -------------------------------------------------

  def step(self, cycle: int) -> tuple[int, str] | None:
    """Execute one cycle.  Returns (role_id, event_id) if a role was
    just dispatched (so the TileGroup can start tiles), else None.
    """
    if self.done or self.task is None:
      self.pmu.add_cycle("idle", 1)
      self.pmu.add_cycle("total", 1)
      return None

    if self._pending is not None:
      if all(e in self._events_done for e in self._pending.events):
        # runtime fidelity: check if any event errored (P0-4/P0-5)
        if self.group.runtime_enabled:
          for ev in self._pending.events:
            status = self.group.event_table.wait(ev)
            if status is not None and status is not EventStatus.DONE:
              self.faulted = True
              self.fault_reason = f"event {ev} status={status.name}"
              self.done = True
              return None
        self._pending = None
        self.pmu.add_cycle("wait_resolved", 1)
        self.pmu.add_cycle("total", 1)
        return None
      else:
        self.pmu.add(StallReason.WAIT_EVENT, 1)
        self.pmu.add_cycle("wait_event", 1)
        self.pmu.add_cycle("total", 1)
        return None
    if self.action_index >= len(self.task.actions):
      # context_done covers grid_done and the final store (IR_SPEC
      # §3.10): don't mark done until every issued role completed
      roles_done = all(ev in self._events_done
                       for ev in self._issued_role_events)
      phases_done = all(ev in self._events_done
                        for ev in self._issued_phase_events)
      if (roles_done and phases_done and self._outstanding_jobs == 0):
        self.done = True
      else:
        self.pmu.add_cycle("drain_wait", 1)
      self.pmu.add_cycle("total", 1)
      return None

    ins = self.task.actions[self.action_index]
    self.pmu.add_cycle("total", 1)
    result = self._issue(ins, cycle)
    if (self.action_index > 0 and not self._first_action_emitted
        and self.group.tracer is not None):
      self._first_action_emitted = True
      self.group.tracer.instant("TileGroup", "Scheduler:L2",
                                "context_first_action", cycle, {
                                  "context": self.context_name,
                                  "slot": self.device_slot,
                                  "launch_generation":
                                    self.context_launch_generation,
                                  "cycle": cycle,
                                })
    return result
  # ---- action issue ----------------------------------------------

  def _issue(self, ins: ExecGroupAction, cycle: int) -> tuple[int, str] | None:
    op = ins.op
    if op == ExecGroupActionOp.INIT_STREAM:
      qid, depth, pmask, cmask = ins.args
      sdesc = ExecStreamDesc(queue_id=qid,
                         depth=depth,
                         producer_mask=pmask,
                         consumer_mask=cmask)
      self.group.init_stream(sdesc)
      self.pmu.add_event("tgs_init_stream")
      self.action_index += 1
    elif op == ExecGroupActionOp.DMA_PREFETCH:
      # Group DMA HBM->L2: submit as a MemoryTransaction.
      if ins.dst is None:
        raise ValueError("DMA_PREFETCH requires dst event id")
      desc_id, transfer = ins.args[0], ins.args[1]
      ok = self.group.submit_group_transfer(
          "dma.prefetch", ins.dst, cycle, desc_id, transfer, sequencer=self)
      if not ok:
        self.pmu.add_event("l2_capacity_fault")
        self.faulted = True
        self.fault_reason = f"L2 capacity fault during context admission on prefetch {desc_id}"
        self.done = True
        return None
      self.pmu.add_event("tgs_dma_prefetch")
      self.action_index += 1
    elif op == ExecGroupActionOp.DMA_STORE:
      if ins.dst is None:
        raise ValueError("DMA_STORE requires dst event id")
      desc_id, transfer = ins.args[0], ins.args[1]
      ok = self.group.submit_group_transfer(
          "dma.store", ins.dst, cycle, desc_id, transfer, sequencer=self)
      if not ok:
        self.pmu.add_event("l2_capacity_fault")
        self.faulted = True
        self.fault_reason = f"L2 capacity fault during context admission on store {desc_id}"
        self.done = True
        return None
      self.pmu.add_event("tgs_dma_store")
      self.action_index += 1
    elif op == ExecGroupActionOp.DISPATCH_ROLE:
      request = ins.args[0]
      assert isinstance(request, ExecDispatchRequest)
      role_id = request.role_id
      assert self.task is not None
      binding = self.task.role_bindings.get(role_id)
      if binding is None:
        raise ValueError(f"unknown role_id {role_id}")
      if not self.group.can_dispatch_role(binding):
        self.pmu.add(StallReason.WAIT_EVENT, 1)
        self.pmu.add_cycle("dispatch_wait", 1)
        return None
      ev = ins.dst or f"ev_role{role_id}"
      if not self.group.dispatch_role(
          binding, cycle, request=request, event_id=ev, sequencer=self):
        return None
      self._role_events[role_id] = ev
      self._issued_role_events.add(ev)
      if request.input_released_event:
        self._issued_phase_events.add(request.input_released_event)
      if request.output_ready_event:
        self._issued_phase_events.add(request.output_ready_event)
      self.pmu.add_event("tgs_dispatch_role")
      self.action_index += 1
      return (role_id, ev)
    elif op == ExecGroupActionOp.WAIT_EVENT:
      ev = ins.args[0]
      self._pending = _GroupTaskWait(events=(ev, ), started_cycle=cycle)
      self.action_index += 1
    elif op == ExecGroupActionOp.BARRIER_GROUP:
      self.pmu.add_event("tgs_barrier")
      self.action_index += 1
    elif op == ExecGroupActionOp.COLLECTIVE_RUN:
      if ins.dst is None:
        raise ValueError("COLLECTIVE_RUN requires dst event id")
      desc_id, op_name, bytes_total, participant_mask = ins.args
      self.group.schedule_collective(
          desc_id,
          ins.dst,
          op_name,
          bytes_total,
          participant_mask,
          cycle,
          sequencer=self,
      )
      self.pmu.add_event("tgs_collective_run")
      self.action_index += 1
    elif op == ExecGroupActionOp.SIGNAL_EVENT:
      self._events_done.add(ins.args[0])
      self.pmu.add_event("tgs_signal_event")
      self.action_index += 1
    elif op == ExecGroupActionOp.RELEASE_L2:
      request = ins.args[0]
      assert isinstance(request, ExecReleaseRequest)
      # Gate on the declared dependency events; keep the action pending
      # via the existing wait mechanism (no new busy-poll state).
      pending_events = tuple(
        ev for ev in request.dependency_events
        if ev not in self._events_done)
      if pending_events:
        self._pending = _GroupTaskWait(
          events=pending_events, started_cycle=cycle)
        self.pmu.add(StallReason.WAIT_EVENT, 1)
        self.pmu.add_cycle("wait_event", 1)
        return None
      if self.group.runtime_enabled or self.group.memory_enabled:
        try:
          self.group.release_l2(request, sequencer=self, cycle=cycle)
        except MemoryInvariantError as exc:
          self.pmu.add_event("release_invariant_fault")
          self.faulted = True
          self.fault_reason = f"release invariant fault: {exc}"
          self.done = True
          if (self.group.runtime_enabled
              and not (self.group.reset_domain.is_active
                       or self.group.reset_domain.is_done)):
            self.group.trigger_fault(
              FaultCode.ADDRESS_FAULT, cycle=cycle, desc_id=str(exc))
          return None
      self.pmu.add_event("tgs_release_l2")
      self.action_index += 1
    return None

  def notify_event(self, event_id: str) -> None:
    self._events_done.add(event_id)
    # runtime fidelity: also signal the EventTable so error/timeline/reset
    # status is observable (P0-4).  In timing_only this is a no-op.
    if self.group.runtime_enabled:
      self.group.event_table.signal(
          event_id, EventStatus.DONE, producer_id=-1, cycle=0)

  def reset(self) -> None:
    self.action_index = 0
    self._events_done.clear()
    self._pending = None
    self._role_events.clear()
    self._issued_role_events.clear()
    self._issued_phase_events.clear()
    self._next_dma_channel = 0
    self.owned_queue_ids.clear()
    self._outstanding_jobs = 0
    self.done = False
    self.faulted = False
    self.fault_reason = ""
    # PR 3.5: reset clears the admission lifecycle and launch bindings
    self.admission_status = ContextAdmissionStatus.PREPARED
    self.admission_wait_start_cycle = None
    self.admission_retry_count = 0
    self.formal_bindings.clear()
    self._first_action_emitted = False
    self.pmu.reset()
