"""EVENT-channel service: PHASE_SIGNAL aggregation, POST_EVENT gate, EVENT_RETIRE broadcast (C024).

Preserves `_on_phase_signal` semantics exactly (W1 plan §8.3):
- stale (retired launch) -> ignored, `tile_signal_stale` +1;
- invalid grid/task/phase or duplicate -> protocol fault counters (`tile_signal_invalid` /
  `tile_signal_duplicate`) each exactly +1 and the sequencer is faulted/cancelled once;
- first complete aggregate fires the phase event exactly once.

POST_EVENT is accepted ONLY from producers explicitly authorized by the sealed manifest
registration passed at construction; anything else is rejected and counted
(`post_event_unauthorized`), never silently dropped.

EVENT_RETIRE broadcast: the Group-side producer of the retire broadcast (previously a direct
`tile.retire_external_events` loop in `_finish_sequencer_retirement`) -- callers migrate to
`EventService.broadcast_event_retire`, which still performs the identical Tile-side retirement.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..execution_ir import PhaseSignal, TaskIdentity
from .tile_port import BoundaryMessage, ContextKey

if TYPE_CHECKING:  # tile_group imports this module at module scope (C035); stay import-cycle free
  from ..tile import ComputeTile
  from ..tile_group import TileGroup

PHASE_DIRECTIONS = ("input_released", "output_ready")


class EventServiceError(RuntimeError):
  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


class EventService:
  """Group-side EVENT channel service bound to one TileGroup."""

  def __init__(self, group: TileGroup, *, authorized_post_producers: frozenset[str] = frozenset()) -> None:
    self.group = group
    self.authorized_post_producers = authorized_post_producers
    self.stats = {
      "phase_signals": 0,
      "phase_stale": 0,
      "phase_invalid": 0,
      "phase_duplicate": 0,
      "phase_aggregates": 0,
      "post_event_accepted": 0,
      "post_event_unauthorized": 0,
      "event_retire_broadcasts": 0,
    }

  # -- PHASE_SIGNAL (T2G) ------------------------------------------------
  def handle_phase_signal(self, msg: BoundaryMessage, *, grid: TaskIdentity, cycle: int) -> str:
    """Feed one PHASE_SIGNAL through the preserved `_on_phase_signal` semantics.

    `grid` carries the logical task identity (context name/slot/launch generation and task id)
    that the emitting tile instruction belonged to; the caller derives it from its launch
    tables exactly as `TileUCE` did before the seam.
    """
    if msg.message_type != "PHASE_SIGNAL" or msg.direction != "T2G":
      raise EventServiceError("BAD_PHASE_SIGNAL", f"{msg.message_type}/{msg.direction}")
    direction = msg.fields.get("direction")
    if direction not in PHASE_DIRECTIONS:
      raise EventServiceError("BAD_PHASE_DIRECTION", repr(direction))
    self.stats["phase_signals"] += 1
    before = self._phase_counters()
    signal_phase = str(direction)
    state = self.group._grid_signals.get(grid.grid)
    completed_before = set(state.completed_phases) if state is not None else set()
    self.group._on_phase_signal(PhaseSignal(task=grid, phase=signal_phase), cycle)
    after = self._phase_counters()
    self._accumulate_phase_delta(before, after)
    if (after["tile_signal_stale"] - before["tile_signal_stale"]) > 0:
      return "stale"
    if (after["tile_signal_duplicate"] - before["tile_signal_duplicate"]) == 1:
      return "duplicate"
    if (after["tile_signal_invalid"] - before["tile_signal_invalid"]) == 1:
      return "invalid"
    if self._aggregate_fired(grid, signal_phase, completed_before):
      self.stats["phase_aggregates"] += 1
      return "aggregated"
    return "recorded"

  def _phase_counters(self) -> dict[str, int]:
    ev = self.group.pmu.events
    return {
      "tile_signal_stale": int(ev.get("tile_signal_stale", 0)),
      "tile_signal_invalid": int(ev.get("tile_signal_invalid", 0)),
      "tile_signal_duplicate": int(ev.get("tile_signal_duplicate", 0)),
    }

  def _accumulate_phase_delta(self, before: dict[str, int], after: dict[str, int]) -> None:
    self.stats["phase_stale"] += after["tile_signal_stale"] - before["tile_signal_stale"]
    self.stats["phase_invalid"] += after["tile_signal_invalid"] - before["tile_signal_invalid"]
    self.stats["phase_duplicate"] += after["tile_signal_duplicate"] - before["tile_signal_duplicate"]

  def _aggregate_fired(self, grid: TaskIdentity, phase: str, completed_before: set[str]) -> bool:
    # An aggregate completes the phase event exactly once: the completing call is the one
    # that moved `phase` into the grid signal state's `completed_phases`. Observed from
    # real Group state before/after the delegated `_on_phase_signal` call -- never from a
    # timer/freshness heuristic. A later duplicate is faulted before completion and can
    # never re-fire the aggregate.
    state = self.group._grid_signals.get(grid.grid)
    return state is not None and phase in state.completed_phases and phase not in completed_before

  # -- POST_EVENT (T2G) ---------------------------------------------------
  def handle_post_event(self, msg: BoundaryMessage, *, producer: str, cycle: int) -> str:
    """Accept one POST_EVENT only from an authorized producer; full-key terminal acceptance."""
    if msg.message_type != "POST_EVENT" or msg.direction != "T2G":
      raise EventServiceError("BAD_POST_EVENT", f"{msg.message_type}/{msg.direction}")
    if producer not in self.authorized_post_producers:
      self.stats["post_event_unauthorized"] += 1
      raise EventServiceError("POST_EVENT_UNAUTHORIZED", producer)
    scope = msg.fields.get("scope")
    status = msg.fields.get("status")
    if scope not in ("task_private", "tile_shared", "group_l2", "external"):
      self.stats["post_event_unauthorized"] += 1
      raise EventServiceError("BAD_EVENT_SCOPE", repr(scope))
    if status not in ("PENDING", "DONE", "ERROR", "TIMEOUT", "RESET"):
      self.stats["post_event_unauthorized"] += 1
      raise EventServiceError("BAD_EVENT_STATUS", repr(status))
    ctx = ContextKey(
      tile_id=int(msg.fields["tile_id"]),
      ctx_slot=int(msg.fields["ctx_slot"]),
      ctx_generation=int(msg.fields["ctx_generation"]),
    )
    _ = ctx, cycle
    self.stats["post_event_accepted"] += 1
    return "accepted"

  # -- EVENT_RETIRE (G2T) broadcast ----------------------------------------
  def broadcast_event_retire(self, events: Iterable[str], tiles: list[ComputeTile]) -> int:
    """Retire external events on every Tile -- the migrated producer of the retire broadcast."""
    retired = 0
    settled = tuple(events)
    for tile in tiles:
      tile.retire_external_events(settled)
      retired += 1
    self.stats["event_retire_broadcasts"] += 1
    return retired

  def snapshot(self) -> dict:
    return {"stats": dict(self.stats)}
