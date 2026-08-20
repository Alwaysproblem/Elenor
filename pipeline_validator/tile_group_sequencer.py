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

from .execution_ir import ExecGroupAction, ExecGroupActionOp, ExecStreamDesc, ExecTileGroupTask
from .pmu import PMUCounter, StallReason
from .runtime import EventStatus

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

  def note_job_started(self) -> None:
    self._outstanding_jobs += 1

  def note_job_done(self) -> None:
    self._outstanding_jobs -= 1

  def load(self, task: ExecTileGroupTask) -> None:
    self.task = task
    self.action_index = 0
    self._events_done.clear()
    self._pending = None
    self._role_events.clear()
    self._issued_role_events.clear()
    self._next_dma_channel = 0
    self.owned_queue_ids.clear()
    self._outstanding_jobs = 0
    self.done = False
    self.faulted = False
    self.fault_reason = ""

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
      # and every DMA/collective job this sequencer issued drained.
      roles_done = all(ev in self._events_done
                       for ev in self._issued_role_events)
      if roles_done and self._outstanding_jobs == 0:
        self.done = True
      else:
        self.pmu.add_cycle("drain_wait", 1)
      self.pmu.add_cycle("total", 1)
      return None

    ins = self.task.actions[self.action_index]
    self.pmu.add_cycle("total", 1)
    return self._issue(ins, cycle)

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
      # Group DMA HBM->L2: model as latency, produces an event.
      if ins.dst is None:
        raise ValueError("DMA_PREFETCH requires dst event id")
      desc_id, dst_l2 = ins.args[0], ins.args[1]
      bytes_total = ins.args[2] if len(ins.args) > 2 else None
      resolved_bytes = bytes_total if bytes_total and bytes_total > 0 else 1024 * 1024
      lat = self._dma_latency(resolved_bytes, l2_slot=dst_l2)
      if lat < 0:
        # L2 capacity fault: terminate the task with a fault rather than
        # masking it as a successful DMA completion.  The sequencer marks
        # itself faulted+done so Simulator.run() reports the error; it does
        # NOT add ins.dst to _events_done (which would let waiters proceed
        # as if the DMA succeeded).
        self.pmu.add_event("l2_capacity_fault")
        self.faulted = True
        self.fault_reason = f"L2 capacity fault on prefetch {desc_id}"
        self.done = True
        return None
      # round-robin DMA channel allocation
      ch = self._next_dma_channel % self.cfg.num_dma_channels
      self._next_dma_channel += 1
      self.group.schedule_dma(
          ins.dst,
          lat,
          cycle,
          op="dma.prefetch",
          desc_id=desc_id,
          l2_slot=dst_l2,
          bytes_total=resolved_bytes,
          channel=ch,
          sequencer=self,
      )
      self.pmu.add_event("tgs_dma_prefetch")
      self.action_index += 1
    elif op == ExecGroupActionOp.DMA_STORE:
      if ins.dst is None:
        raise ValueError("DMA_STORE requires dst event id")
      desc_id, src_l2 = ins.args[0], ins.args[1]
      bytes_total = ins.args[2] if len(ins.args) > 2 else None
      resolved_bytes = bytes_total if bytes_total and bytes_total > 0 else 1024 * 1024
      lat = self._dma_latency(resolved_bytes, l2_slot=src_l2)
      # round-robin DMA channel allocation
      ch = self._next_dma_channel % self.cfg.num_dma_channels
      self.group.schedule_dma(
          ins.dst,
          lat,
          cycle,
          op="dma.store",
          desc_id=desc_id,
          l2_slot=src_l2,
          bytes_total=resolved_bytes,
          channel=ch,
          sequencer=self,
      )
      self.pmu.add_event("tgs_dma_store")
      self.action_index += 1
    elif op == ExecGroupActionOp.DISPATCH_ROLE:
      role_id = ins.args[0]
      inrel_tag = ins.args[1] if len(ins.args) > 1 else ""
      outready_tag = ins.args[2] if len(ins.args) > 2 else ""
      assert self.task is not None
      binding = self.task.role_bindings.get(role_id)
      if binding is None:
        raise ValueError(f"unknown role_id {role_id}")
      if not self.group.can_dispatch_role(binding):
        self.pmu.add(StallReason.WAIT_EVENT, 1)
        self.pmu.add_cycle("dispatch_wait", 1)
        return None
      ev = ins.dst or f"ev_role{role_id}"
      self._role_events[role_id] = ev
      self._issued_role_events.add(ev)
      phase_event_ids = {}
      if inrel_tag:
        phase_event_ids["input_released"] = inrel_tag
      if outready_tag:
        phase_event_ids["output_ready"] = outready_tag
      self.group.dispatch_role(
        binding, cycle, event_id=ev,
        phase_event_ids=phase_event_ids or None,
        sequencer=self)
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
      slot = ins.args[0]
      if self.group.memory_enabled:
        self.group.l2_sram.free_slot(slot)
      self.pmu.add_event("tgs_release_l2")
      self.action_index += 1
    else:
      self.action_index += 1
    return None

  def _dma_latency(self, bytes_total: int | None = None,
                   l2_slot: str = "") -> int:
    """Group DMA latency: bytes / group_dma_bandwidth (Global DMA 6.2).

    In full_memory fidelity, also checks L2 SRAM capacity.  If the L2
    slot cannot be allocated (capacity exhausted), returns -1 to signal
    a capacity fault — the caller records a fault and does not schedule.
    """
    nbytes = bytes_total if bytes_total and bytes_total > 0 else 1024 * 1024
    # full_memory: L2 capacity gate + T_desc + T_issue + T_completion
    if self.group.memory_enabled and l2_slot:
      slot = self.group.l2_sram.alloc_slot(l2_slot, nbytes)
      if slot is None:
        # L2 capacity fault: record and return -1
        self.group.pmu.add_cycle("l2_capacity_fault", 1)
        return -1
      # extended DMA latency model (Global DMA 6.2):
      # T_dma = max(T_read, T_write) + T_desc + T_issue + T_completion
      bw_bytes_per_cycle = self.cfg.group_dma_bandwidth_gbs * 1e9 / (
          self.cfg.clock_mhz * 1e6)
      t_xfer = int(max((nbytes + bw_bytes_per_cycle - 1)
                       // bw_bytes_per_cycle, 1))
      return (t_xfer + self.cfg.dma_desc_cycles
              + self.cfg.dma_issue_cycles + self.cfg.dma_completion_cycles)
    # timing_only / runtime: original latency model
    bw_bytes_per_cycle = self.cfg.group_dma_bandwidth_gbs * 1e9 / (
        self.cfg.clock_mhz * 1e6)
    return int(max((nbytes + bw_bytes_per_cycle - 1) // bw_bytes_per_cycle, 1))

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
    self._next_dma_channel = 0
    self.done = False
    self.faulted = False
    self.fault_reason = ""
    self.pmu.reset()
