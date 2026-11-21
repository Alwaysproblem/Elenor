"""Reset / Drain domain FSM (Driver-Firmware 3.4, Compute Tile 6.47-6.48).

Models the reset/drain state machine that fires after a fault is detected:
Reset must handle: stream token/credit, pending event (write RESET),
local descriptor cache, program residency metadata (epoch), PMU snapshot.

C034 adds the external Tile reset endpoints: attached links exchange real
C024 RESET_PHASE_REQ/ACK traffic at each mapped stage, and every attached
tile must ACK before the FSM leaves that stage.  With no attachments the
single-process timing is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from ..memory.allocator import MemoryInvariantError
from .fault_ring import FaultDomain, FaultRecord

if TYPE_CHECKING:
  from ..cosim.reset_phase import ResetPhaseService
  from ..cosim.tile_port import BoundaryMessage


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

# States that carry a wire reset stage (cosim/reset_phase.py STATE_TO_PHASE).
# DONE maps to the terminal RESUME_DONE already issued at RESUME and must not
# re-arm a stage.
_MAPPED_STAGE_STATES = (
  ResetState.STOP_QUEUE,
  ResetState.FREEZE_DISPATCH,
  ResetState.DRAIN_SAFE,
  ResetState.MARK_EVENTS,
  ResetState.RESET_DOMAIN,
  ResetState.CLEAR_CREDIT_CACHE,
  ResetState.RESUME,
)

# Wire epochs/ids are uint64 words (cosim/wire.py MASK64).
_UINT64_MAX = (1 << 64) - 1


@dataclass
class ResetRequest:
  domain: FaultDomain
  tile_id: int = -1
  queue_id: int = -1
  fault_record: FaultRecord | None = None


@dataclass
class _TileResetLink:
  """One attached external Tile reset endpoint (C034).

  `run_generation` is the authenticated epoch stamped on every request of
  this link and the only generation an ACK may carry.  Stage tracking is
  per reset run; observations record only what a real ACK carried.
  """

  tile_id: int
  run_generation: int
  service: ResetPhaseService
  issued_state: ResetState | None = None
  issued_phase: str | None = None
  request_taken: bool = False
  acked: bool = True
  acked_phases: list[str] = field(default_factory=list)
  drain_evidence: dict[str, bool] | None = None
  pending_to_reset: int | None = None
  isolation_reported: bool = False


@dataclass
class ResetDomain:
  """Drain FSM for one reset request.

  Each state consumes one or more cycles; `step()` returns the current
  state.  When state == DONE the reset is complete and dispatch may resume.

  With attached external endpoints (C034) every mapped stage additionally
  waits for all attached tiles' real RESET_PHASE_ACKs before advancing;
  local stage side effects still run exactly once per reset run.
  """

  cfg: object
  state: ResetState = ResetState.IDLE
  request: ResetRequest | None = None
  start_cycle: int = 0
  drain_cycles: int = 0  # PMU: total drain cycles
  tile_reset_links: dict[int, _TileResetLink] = field(default_factory=dict)
  # per-reset-run stage bookkeeping (C034)
  _issued_state: ResetState | None = None
  _local_done: set[ResetState] = field(default_factory=set)
  # per-tile retired-epoch watermark survives binding clears (C034)
  _retired_epochs: dict[int, int] = field(default_factory=dict)
  # an admitted isolation failure is owed this epoch's observed DONE drain
  _isolation_closure_pending: bool = False

  def begin(self, req: ResetRequest, cycle: int) -> None:
    if self.tile_reset_links and (self.is_active or self.is_done):
      # Each attached wire epoch is reset at most once (C034 contract): a
      # Tile replays cached ACKs for acknowledged phases without re-executing,
      # so a same-epoch reopen cannot distinguish fresh evidence from replay
      # and would let seven genuine replay ACKs clear a reported failure.
      # Recovery of a spent epoch needs the explicit quiescent reset() and a
      # new, greater authenticated epoch.  With no attachments the historical
      # restart-anytime injection behavior is preserved.
      raise MemoryInvariantError(
        "attached reset epoch cannot be reopened while a reset is active or done; "
        "explicit quiescent reset() and a new epoch first"
      )
    self.request = req
    self.state = ResetState.FAULT_DETECTED
    self.start_cycle = cycle
    self._begin_run()

  def step(self, cycle: int, group: object | None = None) -> ResetState:
    """Advance the drain FSM one cycle.  `group` is the TileGroup, used
    to perform the actual credit/event/residency cleanup at RESET_DOMAIN.

    With attached tile reset endpoints every mapped stage additionally
    holds until all attached tiles ACKed it (C034): no stage advances on a
    missing ACK, and local stage effects run once per reset run."""
    if self.state == ResetState.IDLE or self.request is None:
      return self.state
    cfg = self.cfg
    # each state consumes 1 cycle except DRAIN_SAFE which waits for outstanding
    if self.state == ResetState.FAULT_DETECTED:
      self.state = ResetState.STOP_QUEUE
    elif self.state == ResetState.STOP_QUEUE:
      if self._stage_acked():
        self.state = ResetState.FREEZE_DISPATCH
    elif self.state == ResetState.FREEZE_DISPATCH:
      if ResetState.FREEZE_DISPATCH not in self._local_done:
        if group is not None:
          self._cancel_unissued_engine_requests(group, cycle)
        self._local_done.add(ResetState.FREEZE_DISPATCH)
      if self._stage_acked():
        self.state = ResetState.DRAIN_SAFE
    elif self.state == ResetState.DRAIN_SAFE:
      # Wait for accepted engines, transfers, fabric and control isolation,
      # and for every attached tile's real DRAIN ACK.  One watchdog covers
      # local and external outstanding alike (C034): exceeding
      # max_drain_cycles escalates cancellation, never advances the FSM and
      # never treats timeout as drain success or failed isolation.
      local_quiet = group is None or self._outstanding_zero(group, cycle)
      tm = getattr(group, "transfer_manager", None)
      cancellation_pending = tm is not None and tm.cancellation_pending
      if local_quiet and not cancellation_pending and self._stage_acked():
        self.state = ResetState.MARK_EVENTS
      elif local_quiet and cancellation_pending:
        # Only an explicit isolation confirmation or leg completion may
        # retire the cancelled transaction; timeout is never success.
        pass
      else:
        self.drain_cycles += 1
        if self.drain_cycles > getattr(cfg, "max_drain_cycles", 100):
          if tm is not None:
            tm.cancel_all(cycle)
          for tile in getattr(group, "tiles", ()):
            request_mfe_cancel = getattr(getattr(tile, "mfe", None), "request_cancel_active", None)
            if not callable(request_mfe_cancel):
              raise MemoryInvariantError("reset integration lacks accepted MFE cancellation")
            request_mfe_cancel(cycle)
    elif self.state == ResetState.MARK_EVENTS:
      # mark pending events as RESET (Runtime ABI 3.2).  The Group writes
      # its own table exactly once; attached tiles mark their local tables
      # themselves and report the count in the MARK_EVENTS ACK (C034).
      if ResetState.MARK_EVENTS not in self._local_done:
        if group is not None and hasattr(group, "event_table"):
          group.event_table.reset()
        self._local_done.add(ResetState.MARK_EVENTS)
      if self._stage_acked():
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
      # release context-owned L2/L1 allocations, pins and frames, then
      # invalidate residency — once per run: the releaser is not re-invoked
      # on later holds while the stage waits for external ACKs (C034).
      if ResetState.RESET_DOMAIN not in self._local_done:
        releaser = getattr(group, "release_context_memory", None)
        if releaser is not None and not releaser(cycle):
          # Safe retirement is not complete; remain in RESET_DOMAIN and do not
          # invalidate program residency or advance to credit/cache cleanup.
          return self.state
        if group is not None and hasattr(group, "program_table"):
          if self.request.domain == FaultDomain.TILE:
            group.program_table.invalidate_tile(self.request.tile_id)
          elif self.request.domain == FaultDomain.GROUP:
            group.program_table.invalidate_group()
        self._local_done.add(ResetState.RESET_DOMAIN)
      if self._stage_acked():
        self.state = ResetState.CLEAR_CREDIT_CACHE
    elif self.state == ResetState.CLEAR_CREDIT_CACHE:
      # reconcile stream credit, clear descriptor cache (once per run)
      if ResetState.CLEAR_CREDIT_CACHE not in self._local_done:
        if group is not None:
          for q in getattr(group, "queues", {}).values():
            q.reset()
        self._local_done.add(ResetState.CLEAR_CREDIT_CACHE)
      if self._stage_acked():
        self.state = ResetState.RESUME
    elif self.state == ResetState.RESUME:
      if self._stage_acked():
        self.state = ResetState.DONE
    elif self.state == ResetState.DONE:
      pass
    self._arm_stage_requests()
    return self.state

  def _stage_acked(self) -> bool:
    """True when every attached tile ACKed the current stage (vacuous with none)."""
    return all(link.acked for link in self.tile_reset_links.values())

  def _arm_stage_requests(self) -> None:
    """Arm the entered stage's RESET_PHASE_REQ on every attached link (C034).

    The Group is the sole order authority: a stage arms when the FSM enters
    its mapped state, its request is taken once per link, and every attached
    tile must ACK it before the FSM may leave that state."""
    if not self.tile_reset_links or self.state not in _MAPPED_STAGE_STATES:
      return
    if self._issued_state is self.state:
      return
    from ..cosim.reset_phase import STATE_TO_PHASE  # lazy: cosim imports this module

    self._issued_state = self.state
    phase = STATE_TO_PHASE[self.state]
    for link in self.tile_reset_links.values():
      link.issued_state = self.state
      link.issued_phase = phase
      link.request_taken = False
      link.acked = False

  def _begin_run(self) -> None:
    """Fresh per-run link tracking; attachments and their epochs persist."""
    self._issued_state = None
    self._local_done = set()
    for link in self.tile_reset_links.values():
      link.service.begin_run()
      link.issued_state = None
      link.issued_phase = None
      link.request_taken = False
      link.acked = True
      link.acked_phases = []
      link.drain_evidence = None
      link.pending_to_reset = None

  def note_isolation_failure(self, tile_id: int, run_generation: int) -> bool:
    """Admit one externally reported Tile isolation failure (C034).

    Valid only while this epoch's drain is still ahead of the tile — before
    its DRAIN ACK was observed — and discharged only by this epoch's real
    observed drain reaching DONE plus the authorized quiescence path
    (reset()), which owns recovery separately from the Tile's receipts.
    A duplicate of an already reported failure is idempotent.  A NEW
    failure first reported after the tile certified DRAIN is a late
    contradictory input: observable fail-closed protocol error, never
    silently rehabilitated by prior or repeated receipts.  Returns True
    for the first valid report of this epoch, False for a duplicate."""
    link = self._require_link(tile_id)
    if link.run_generation != run_generation:
      raise MemoryInvariantError(
        f"isolation failure for tile {tile_id}@{run_generation} has no matching attached epoch"
      )
    if link.isolation_reported:
      return False
    if link.drain_evidence is not None:
      raise MemoryInvariantError(
        f"late contradictory isolation failure: tile {tile_id} certified DRAIN "
        f"(run_generation {run_generation}) before reporting it"
      )
    link.isolation_reported = True
    self._isolation_closure_pending = True
    return True

  @property
  def isolation_closure_satisfied(self) -> bool:
    """True when no isolation closure is owed, or this epoch's drain finished."""
    return not self._isolation_closure_pending or self.is_done

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

  # ---- attached external reset endpoints (C034) ------------------------

  def attach_tile_reset(self, tile_id: int, run_generation: int) -> None:
    """Register one external Tile reset endpoint, only at IDLE (C034).

    Exactly one registration per tile; a completed epoch's evidence never
    authorizes attaching (and later clearing) a fresh Tile.  `run_generation`
    is a uint64 epoch stamped on every RESET_PHASE_REQ of this link,
    strictly greater than any retired epoch of this tile, and the only
    generation an ACK of this link may carry."""
    from ..cosim.reset_phase import ResetPhaseService  # lazy: cosim imports this module

    self._check_tile_id(tile_id)
    if isinstance(run_generation, bool) or not isinstance(run_generation, int):
      raise ValueError(f"run_generation must be a uint64 int, got {run_generation!r}")
    if run_generation < 0 or run_generation > _UINT64_MAX:
      raise ValueError(f"run_generation must be a uint64 int, got {run_generation!r}")
    if tile_id in self.tile_reset_links:
      raise ValueError(f"tile {tile_id} already has an attached reset endpoint")
    if self.is_active or self.is_done:
      raise ValueError("cannot attach a reset endpoint while a reset is active or done")
    retired = self._retired_epochs.get(tile_id, -1)
    if run_generation <= retired:
      raise ValueError(
        f"run_generation {run_generation} for tile {tile_id} does not exceed "
        f"the retired epoch {retired}"
      )
    self.tile_reset_links[tile_id] = _TileResetLink(
      tile_id=tile_id,
      run_generation=run_generation,
      service=ResetPhaseService(),
    )

  def take_tile_reset_request(self, tile_id: int) -> BoundaryMessage | None:
    """Return the outstanding RESET_PHASE_REQ of this link, once per stage."""
    link = self._require_link(tile_id)
    if link.acked or link.issued_state is None or link.request_taken:
      return None
    msg = link.service.emit_request(link.issued_state, link.run_generation)
    link.request_taken = True  # only after the sealed REQ was actually built
    return msg

  def observe_tile_reset_ack(self, tile_id: int, msg: BoundaryMessage) -> None:
    """Observe one real RESET_PHASE_ACK fired by the attached tile.

    Erroneous, early and duplicate ACKs raise before anything is consumed:
    the outstanding request and every owed observation stay intact."""
    from ..cosim.reset_phase import ResetPhaseError  # lazy: cosim imports this module

    link = self._require_link(tile_id)
    if link.acked or link.issued_state is None:
      raise ResetPhaseError("UNSOLICITED_RESET_ACK", f"tile {tile_id} has no outstanding stage")
    if getattr(msg, "message_type", None) != "RESET_PHASE_ACK":
      raise ResetPhaseError(
        "BAD_RESET_ACK", f"{getattr(msg, 'message_type', type(msg).__name__)!r} is not a RESET_PHASE_ACK"
      )
    observation = link.service.handle_ack(msg)
    link.acked = True
    link.acked_phases.append(observation["phase"])
    if observation["phase"] == "DRAIN":
      link.drain_evidence = dict(observation["evidence"])
    elif observation["phase"] == "MARK_EVENTS":
      link.pending_to_reset = observation["pending_to_reset"]

  def tile_reset_observation(self, tile_id: int) -> dict:
    """Observed-only reset evidence for one attached tile.

    `is_quiescent_snapshot` holds the named drain-evidence bools of the
    OBSERVED DRAIN ACK (None until then), `pending_to_reset` the observed
    MARK_EVENTS count, `phases` the actually acknowledged stages only."""
    link = self._require_link(tile_id)
    return {
      "tile_id": link.tile_id,
      "run_generation": link.run_generation,
      "phases": list(link.acked_phases),
      "is_quiescent_snapshot": None if link.drain_evidence is None else dict(link.drain_evidence),
      "pending_to_reset": link.pending_to_reset,
      "outstanding_phase": None if link.acked else link.issued_phase,
      "request_available": link.issued_state is not None and not link.request_taken and not link.acked,
    }

  def _require_link(self, tile_id: int) -> _TileResetLink:
    self._check_tile_id(tile_id)  # reject bool/int aliasing before the lookup
    link = self.tile_reset_links.get(tile_id)
    if link is None:
      raise ValueError(f"tile {tile_id} has no attached reset endpoint")
    return link

  @staticmethod
  def _check_tile_id(tile_id: int) -> None:
    if isinstance(tile_id, bool) or not isinstance(tile_id, int) or tile_id < 0 or tile_id > _UINT64_MAX:
      raise ValueError(f"tile_id must be a uint64 int, got {tile_id!r}")

  def reset(self) -> None:
    if self.tile_reset_links and self.is_active:
      # An attached reset mid-handshake still owes real tile ACKs; its
      # bindings never drop before the staged reset finishes — the same
      # guard TileGroup.reset() applies, so a direct call cannot bypass it.
      # Unbound resets keep their historical semantics.
      raise MemoryInvariantError(
        "reset refused: an attached reset is still awaiting real tile ACKs"
      )
    if self.tile_reset_links and not self.isolation_closure_satisfied:
      # Direct calls are protected exactly like TileGroup.reset(): an
      # attached epoch's closure obligation never drops with its bindings.
      raise MemoryInvariantError(
        "reset refused: an attached tile isolation failure needs this epoch's "
        "observed DRAIN and a finished reset drain"
      )
    self.state = ResetState.IDLE
    self.request = None
    self.start_cycle = 0
    self.drain_cycles = 0
    # Retire every attached epoch behind a monotonic per-tile watermark that
    # survives this clear, then drop the bindings: a stale late ACK has no
    # attachment to apply to and a new run must carry a strictly greater
    # authenticated epoch (C034).  Recovery stays owned by this authorized
    # quiescence path, separate from the Tile's staged receipts.
    for tile_id, link in self.tile_reset_links.items():
      retired = self._retired_epochs.get(tile_id, -1)
      self._retired_epochs[tile_id] = max(retired, link.run_generation)
    self.tile_reset_links.clear()
    self._issued_state = None
    self._local_done = set()
    self._isolation_closure_pending = False
