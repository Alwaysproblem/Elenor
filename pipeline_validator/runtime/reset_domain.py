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

from ..memory.allocator import MemoryInvariantError
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
      if group is not None:
        self._cancel_unissued_engine_requests(group, cycle)
      self.state = ResetState.DRAIN_SAFE
    elif self.state == ResetState.DRAIN_SAFE:
      # Wait for accepted engines, transfers, fabric and control isolation.
      if group is None or self._outstanding_zero(group, cycle):
        tm = getattr(group, "transfer_manager", None)
        if tm is not None and tm.cancellation_pending:
          # Only an explicit isolation confirmation or leg completion may
          # retire the cancelled transaction; timeout is never success.
          self.state = ResetState.DRAIN_SAFE
        else:
          self.state = ResetState.MARK_EVENTS
      else:
        self.drain_cycles += 1
        if self.drain_cycles > getattr(cfg, "max_drain_cycles", 100):
          tm = getattr(group, "transfer_manager", None)
          if tm is not None:
            tm.cancel_all(cycle)
          for tile in getattr(group, "tiles", ()):
            request_mfe_cancel = getattr(getattr(tile, "mfe", None), "request_cancel_active", None)
            if not callable(request_mfe_cancel):
              raise MemoryInvariantError("reset integration lacks accepted MFE cancellation")
            request_mfe_cancel(cycle)
    elif self.state == ResetState.MARK_EVENTS:
      # mark pending events as RESET (Runtime ABI 3.2)
      if group is not None and hasattr(group, "event_table"):
        group.event_table.reset()
      self.state = ResetState.RESET_DOMAIN
    elif self.state == ResetState.RESET_DOMAIN:
      # reset cleanup: no cancellation here — the drain phase already issued
      # and confirmed cancellations.  A reset which reached this state may
      # still be rejected by the transfer manager if live work remains.
      tm = getattr(group, "transfer_manager", None)
      if tm is not None:
        snapshot = tm.snapshot()
        if snapshot["inflight"] or snapshot["cancel_requested"]:
          raise MemoryInvariantError("reset proceeded before transfer cancellation was confirmed")
      # release context-owned L2/L1 allocations, pins and frames
      releaser = getattr(group, "release_context_memory", None)
      if releaser is not None and not releaser(cycle):
        # Safe retirement is not complete; remain in RESET_DOMAIN and do not
        # invalidate program residency or advance to credit/cache cleanup.
        return self.state
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

  @staticmethod
  def _cancel_unissued_engine_requests(group: object, cycle: int) -> None:
    """Cancel UCE requests not yet accepted by an engine.

    Engine running/service queues are deliberately untouched and continue
    ticking during drain.  Only the UCE-to-engine request FIFOs are removed.
    """
    if not hasattr(group, "tiles"):
      raise MemoryInvariantError("reset integration is missing Tile list")
    for tile in group.tiles:
      if not hasattr(tile, "uce"):
        raise MemoryInvariantError("reset integration is missing Tile UCE")
      uce = tile.uce
      cancel = getattr(uce, "_clear_engine_queues", None)
      if not callable(cancel):
        raise MemoryInvariantError("Tile UCE lacks unissued engine-request cancellation")
      cancel(unregister=True)
      mfe = getattr(tile, "mfe", None)
      begin_mfe_drain = getattr(mfe, "begin_reset_drain", None)
      if not callable(begin_mfe_drain):
        raise MemoryInvariantError("Tile MFE lacks reset-drain cancellation")
      begin_mfe_drain(cycle)
    controller = getattr(group, "profile_controller", None)
    if controller is None:
      raise MemoryInvariantError("reset integration is missing ProfileController")
    request_cancel_all = getattr(controller, "request_cancel_all", None)
    if not callable(request_cancel_all):
      raise MemoryInvariantError("ProfileController lacks live-command cancellation")
    request_cancel_all(cycle)

  def _outstanding_zero(self, group: object, cycle: int) -> bool:
    """Check real accepted work/fabric/config isolation, not frozen UCE PCs."""
    transfer_manager = getattr(group, "transfer_manager", None)
    if transfer_manager is None:
      raise MemoryInvariantError("reset integration is missing TransferManager")
    transfer_closure = transfer_manager.closure_snapshot()
    if not transfer_closure["quiescent"] or transfer_manager.cancellation_pending:
      return False
    transfer_snapshot = transfer_manager.snapshot()
    if transfer_snapshot["hbm_outstanding"]:
      return False
    if any(
      stage["busy_resources"] or stage["outstanding"] for stage in transfer_snapshot["stages"].values()
    ):
      return False

    noc = getattr(group, "noc", None)
    if noc is not None and not noc.is_quiescent:
      return False
    if getattr(group, "_collective_jobs", ()):
      return False

    controller = getattr(group, "profile_controller", None)
    if controller is None:
      raise MemoryInvariantError("reset integration is missing ProfileController")
    if not hasattr(controller, "initialized") or not hasattr(controller, "cancellation_pending"):
      raise MemoryInvariantError("ProfileController lacks typed cancellation accounting")
    if not controller.initialized or controller.cancellation_pending:
      return False

    tiles = getattr(group, "tiles", None)
    if tiles is None:
      raise MemoryInvariantError("reset integration is missing Tile list")
    for tile in tiles:
      retire_isolated = getattr(getattr(tile, "mfe", None), "retire_isolated", None)
      if not callable(retire_isolated):
        raise MemoryInvariantError("reset integration lacks MFE isolation retirement")
      retire_isolated(cycle)
      for name in ("boa", "evu", "mfe", "use"):
        engine = getattr(tile, name, None)
        if engine is None or not hasattr(engine, "accepted_count"):
          raise MemoryInvariantError(f"reset integration is missing {name} accepted-job accounting")
        if engine.accepted_count:
          return False
      l1_mshr = getattr(tile, "l1_mshr", None)
      if l1_mshr is None:
        raise MemoryInvariantError("reset integration is missing L1 MSHR")
      if l1_mshr.stats.active:
        return False
    l2_mshr = getattr(group, "l2_mshr", None)
    if l2_mshr is None:
      raise MemoryInvariantError("reset integration is missing L2 MSHR")
    return not l2_mshr.stats.active

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
