"""Reset / Drain domain FSM (Driver-Firmware 3.4, Compute Tile 6.47-6.48).

Models the reset/drain state machine that fires after a fault is detected:

  FaultDetected -> StopAffectedQueue -> FreezeNewDispatch ->
  DrainSafeCommands -> MarkPendingEvents -> ResetTileOrGroupOrDevice ->
  ClearStreamCreditAndDescriptorCache -> ResumeOrDestroyContext

Reset must handle: stream token/credit, pending event (write RESET),
local descriptor cache, program residency metadata (epoch), PMU snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .fault_ring import FaultDomain, FaultRecord


class ResetState(IntEnum):
  IDLE = 0
  FAULT_DETECTED = 1
  STOP_QUEUE = 2
  FREEZE_DISPATCH = 3
  DRAIN_SAFE = 4
  MARK_EVENTS = 5
  RESET_DOMAIN = 6
  CLEAR_CREDIT_CACHE = 7
  RESUME = 8
  DONE = 9


@dataclass
class ResetRequest:
  domain: FaultDomain
  tile_id: int = -1
  queue_id: int = -1
  fault_record: FaultRecord | None = None


@dataclass
class ResetDomain:
  """Drain FSM for one reset request.

  Each state consumes one or more cycles; `step()` returns the current
  state.  When state == DONE the reset is complete and dispatch may resume.
  """

  cfg: object
  state: ResetState = ResetState.IDLE
  request: ResetRequest | None = None
  start_cycle: int = 0
  drain_cycles: int = 0  # PMU: total drain cycles

  def begin(self, req: ResetRequest, cycle: int) -> None:
    self.request = req
    self.state = ResetState.FAULT_DETECTED
    self.start_cycle = cycle

  def step(self, cycle: int, group: object | None = None) -> ResetState:
    """Advance the drain FSM one cycle.  `group` is the TileGroup, used
    to perform the actual credit/event/residency cleanup at RESET_DOMAIN."""
    if self.state == ResetState.IDLE or self.request is None:
      return self.state
    cfg = self.cfg
    # each state consumes 1 cycle except DRAIN_SAFE which waits for outstanding
    if self.state == ResetState.FAULT_DETECTED:
      self.state = ResetState.STOP_QUEUE
    elif self.state == ResetState.STOP_QUEUE:
      self.state = ResetState.FREEZE_DISPATCH
    elif self.state == ResetState.FREEZE_DISPATCH:
      self.state = ResetState.DRAIN_SAFE
    elif self.state == ResetState.DRAIN_SAFE:
      # wait for outstanding DMA/engine/token to settle (or timeout).
      # If group is None (standalone test), there is nothing to drain.
      if group is None or self._outstanding_zero(group):
        self.state = ResetState.MARK_EVENTS
      else:
        self.drain_cycles += 1
        # bounded drain to avoid deadlock
        if self.drain_cycles > getattr(cfg, "max_drain_cycles", 100):
          self.state = ResetState.MARK_EVENTS
    elif self.state == ResetState.MARK_EVENTS:
      # mark pending events as RESET (Runtime ABI 3.2)
      if group is not None and hasattr(group, "event_table"):
        group.event_table.reset()
      self.state = ResetState.RESET_DOMAIN
    elif self.state == ResetState.RESET_DOMAIN:
      # invalidate program residency + clear descriptor cache
      if group is not None and hasattr(group, "program_table"):
        if self.request.domain == FaultDomain.TILE:
          group.program_table.invalidate_tile(self.request.tile_id)
        elif self.request.domain == FaultDomain.GROUP:
          group.program_table.invalidate_group()
      self.state = ResetState.CLEAR_CREDIT_CACHE
    elif self.state == ResetState.CLEAR_CREDIT_CACHE:
      # reconcile stream credit, clear descriptor cache
      if group is not None:
        for q in getattr(group, "queues", {}).values():
          q.reset()
      self.state = ResetState.RESUME
    elif self.state == ResetState.RESUME:
      self.state = ResetState.DONE
    elif self.state == ResetState.DONE:
      pass
    return self.state

  def _outstanding_zero(self, group: object) -> bool:
    """Check that all outstanding DMA/engine jobs have completed."""
    dma_jobs = getattr(group, "_dma_jobs", [])
    coll_jobs = getattr(group, "_collective_jobs", [])
    if dma_jobs or coll_jobs:
      return False
    for t in getattr(group, "tiles", []):
      if not getattr(t, "done", True):
        return False
    return True

  @property
  def is_active(self) -> bool:
    return self.state not in (ResetState.IDLE, ResetState.DONE)

  @property
  def is_done(self) -> bool:
    return self.state == ResetState.DONE

  def reset(self) -> None:
    self.state = ResetState.IDLE
    self.request = None
    self.start_cycle = 0
    self.drain_cycles = 0
