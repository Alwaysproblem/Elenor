"""Boundary retirement coordinator (C035 / W1-04): Python Tile step + external retire chain.

Python path (`step_python`): behavior-identical migration of the former
`TileGroup._step_python_tiles` -- per-Tile step, UCE terminal collection, real L1
view/Arena/Frame reclaim -- after which the participant leaves `_retiring_tasks` and
`TileGroup._finish_route_task` performs the parent close exactly once, like the native
path. Python admissions live in `route.admissions`; `admission.tickets` is never used
here. External Tiles (ids in `group.boundary_ports`) are skipped: no local L1 handling.

External path (`handle`/`advance`), per accepted admission ticket:

    RETIRE_READY (T2G CTRL) -> RETIRE (G2T CTRL, accepted P1 seq `sent_seq`) ->
    TASK_COMPLETION (T2G CTRL) -> `_finish_route_task` -> `admission.complete`
    (releases the ticket's EventTable participant ref, strictly after the close).

Rules:
- RETIRE is owed once READY is bound to the accepted ticket; `advance` sends it with
  CTRL credit + the immutable `arena_alloc_gen` evidence from the owner's
  RESOURCE_UPDATE (may arrive after READY). Never gated on `sequencer.faulted` or on
  parent R/pin settlement (forbidden wait cycle); exactly once per task.
- A terminal counts only after the RETIRE frame is actually accepted by the Tile:
  `g2t_accepted_seq('CTRL') >= sent_seq`. An enqueued-but-backpressured RETIRE
  justifies no release.
- Premature terminal, wrong live owner (forged key/ctx, wrong tile, unaccepted
  ticket), or a key naming an unlaunched/live-with-no-owner launch: reject with one
  real Group protocol fault (sequencer fault + cancel + GROUP trigger) and zero
  mutation of parent obligations; duplicates on a live chain keep the phase-signal
  duplicate root-fault semantics. Only truly retired (non-live) keys are counted
  stale. An authorized Task whose ticket vanished is an invariant violation (raise).
- The wired terminal outcome is authoritative: never rewritten, never invented.
- State is finite (READY -> AUTHORIZED, dropped at completion) and bounded: entries
  whose Route died without a terminal are swept in `advance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..execution_ir import TaskIdentity
from ..memory.allocator import MemoryInvariantError
from ..runtime.event_table import EventStatus
from ..runtime.fault_ring import FaultCode
from .tile_port import BoundaryMessage, TaskKey, wire_message_seq

if TYPE_CHECKING:  # tile_group imports this module at module scope; stay import-cycle free
  from ..tile_group import TileGroup, _GridRoute
  from ..tile_group_sequencer import TileGroupSequencer

_TERMINAL_EVENT_STATUS = {
  "DONE": EventStatus.DONE,
  "ERROR": EventStatus.ERROR,
  "TIMEOUT": EventStatus.TIMEOUT,
  "RESET": EventStatus.RESET,
}


class RetirementError(RuntimeError):
  """Malformed boundary frame (bad type/direction/fields); raised, never counted."""


class RetirementPhase(Enum):
  READY = "ready"  # RETIRE_READY bound; RETIRE authorization owed
  AUTHORIZED = "authorized"  # RETIRE sent; terminal owed once the Tile accepted it


@dataclass
class _ExternalRetirement:
  key: TaskKey
  tile_id: int
  ctx_slot: int
  ctx_generation: int
  route: _GridRoute
  task: TaskIdentity
  ready_cycle: int
  phase: RetirementPhase = RetirementPhase.READY
  sent_seq: int | None = None  # accepted-seq gate: no terminal counts before this
  arena_alloc_gen: int | None = None  # immutable, from the owner's RESOURCE_UPDATE


def _task_key_from(msg: BoundaryMessage) -> TaskKey:
  f = msg.fields
  try:
    return TaskKey(
      launch_generation=int(f["launch_generation"]),
      root_launch_generation=int(f["root_launch_generation"]),
      dispatch_ordinal=int(f["dispatch_ordinal"]),
      task_id=int(f["task_id"]),
      context_handle=int(f["context_handle"]),
    )
  except (KeyError, TypeError, ValueError) as exc:
    raise RetirementError(f"BAD_TASK_KEY: {exc}") from exc


def _terminal_status(msg: BoundaryMessage) -> str:
  raw = msg.fields.get("status")
  if isinstance(raw, str) and raw in _TERMINAL_EVENT_STATUS:
    return raw
  if not isinstance(raw, bool) and isinstance(raw, int) and 0 <= raw < 4:
    return ("DONE", "ERROR", "TIMEOUT", "RESET")[raw]
  raise RetirementError(f"BAD_TERMINAL_STATUS: {raw!r}")


class RetirementCoordinator:
  """Group-side retirement owner for one `TileGroup` (Python + external boundary paths)."""

  def __init__(self, group: TileGroup) -> None:
    self.group = group
    self._pending: dict[tuple[int, TaskKey], _ExternalRetirement] = {}
    self.stats = {
      "retire_ready": 0,
      "retire_authorized": 0,
      "task_completions": 0,
      "retire_stale": 0,
      "retire_invalid": 0,
      "retire_duplicate": 0,
      "phase_delegated": 0,
    }

  # ---- pure-Python path (migrated from TileGroup._step_python_tiles) ----
  def step_python(self, cycle: int, *, freeze_new_work: bool) -> None:
    group = self.group
    for tile in group.tiles:
      if tile.tile_id in group.boundary_ports:
        continue
      tile.step(cycle, freeze_new_work=freeze_new_work)
      for terminal in tile.drain_context_terminals():
        task = terminal.task_identity
        if task is None:
          raise MemoryInvariantError("terminal event lacks TaskIdentity")
        route = group._grid_routes.get(task.grid)
        if route is None or route.expected.get(tile.tile_id) != task:
          raise MemoryInvariantError("stale or foreign Task terminal")
        if terminal.status == "fault":
          group.on_scheduler_fault(route.sequencer, f"tile{tile.tile_id}: {terminal.reason}", cycle)
          continue
        group._retiring_tasks[task] = terminal
    for task, terminal in tuple(group._retiring_tasks.items()):
      route = group._grid_routes[task.grid]
      tile = group.python_tiles_by_id[terminal.tile_id]
      admission = route.admissions.get(tile.tile_id)
      if admission is None or admission.arena is None:
        raise MemoryInvariantError("terminal has no committed Task Arena")
      if any(group.transfer_manager.has_inflight_access(h) for h in admission.l1_handles.values()):
        continue
      pending = False
      for name, handle in tuple(admission.l1_handles.items()):
        if not tile.l1_allocator.invalidate_view(handle, handle.owner, cycle):
          pending = True
        else:
          del admission.l1_handles[name]
      if pending:
        continue
      if not tile.l1_allocator.retire_arena(admission.arena, cycle):
        continue
      tile.l1_frames[terminal.ctx_id].release()
      del group._retiring_tasks[task]
      group._finish_route_task(route, task, tile.tile_id, terminal.ctx_id, cycle, EventStatus.DONE)

  # ---- external boundary path ------------------------------------------
  def handle(self, tile_id: int, msg: BoundaryMessage, cycle: int) -> str:
    """Consume one T2G RETIRE_READY / TASK_COMPLETION / PHASE_SIGNAL from `tile_id`'s link."""
    if msg.direction != "T2G":
      raise RetirementError(f"BAD_DIRECTION: {msg.message_type}/{msg.direction}")
    if msg.message_type == "RETIRE_READY":
      return self._handle_retire_ready(tile_id, msg, cycle)
    if msg.message_type == "TASK_COMPLETION":
      return self._handle_task_completion(tile_id, msg, cycle)
    if msg.message_type == "PHASE_SIGNAL":
      return self._handle_phase_signal(tile_id, msg, cycle)
    raise RetirementError(f"BAD_MESSAGE: {msg.message_type}")

  def _handle_retire_ready(self, tile_id: int, msg: BoundaryMessage, cycle: int) -> str:
    group = self.group
    key = _task_key_from(msg)
    state = self._pending.get((tile_id, key))
    if state is not None:
      return self._reject_duplicate(state, "RETIRE_READY", cycle)
    ticket = group.admission.tickets.get((tile_id, key))
    if ticket is None:
      return self._reject_unbound(tile_id, key, "RETIRE_READY", cycle)
    try:
      ctx_slot = int(msg.fields["ctx_slot"])
      ctx_generation = int(msg.fields["ctx_generation"])
      drain_evidence_seq = int(msg.fields["drain_evidence_seq"])
    except (KeyError, TypeError, ValueError) as exc:
      raise RetirementError(f"BAD_RETIRE_READY: {exc}") from exc
    if (
      not ticket.can_retire
      or ticket.ctx_slot != ctx_slot
      or ticket.ctx_generation != ctx_generation
      or ticket.route.expected.get(tile_id) != ticket.task
    ):
      self._protocol_fault(
        ticket.route.sequencer,
        f"tile {tile_id} RETIRE_READY failed exact Task/Context binding for Task {ticket.task.task_id}",
        cycle,
        "retire_invalid",
        {**key.to_wire_fields(), "tile_id": tile_id,
         "ctx_slot": ctx_slot, "ctx_generation": ctx_generation},
      )
      return "invalid"
    self._pending[(tile_id, key)] = _ExternalRetirement(
      key=key,
      tile_id=tile_id,
      ctx_slot=ctx_slot,
      ctx_generation=ctx_generation,
      route=ticket.route,
      task=ticket.task,
      ready_cycle=cycle,
    )
    self.stats["retire_ready"] += 1
    self._trace(
      tile_id,
      "retire_ready",
      cycle,
      {
        "tile_id": tile_id,
        **key.to_wire_fields(),
        "ctx_slot": ctx_slot,
        "ctx_generation": ctx_generation,
        "drain_evidence_seq": drain_evidence_seq,
        "phase": RetirementPhase.READY.value,
      },
    )
    return "ready"

  def _handle_task_completion(self, tile_id: int, msg: BoundaryMessage, cycle: int) -> str:
    group = self.group
    key = _task_key_from(msg)
    status = _terminal_status(msg)
    try:
      first_fault_ref = int(msg.fields["first_fault_ref"])
    except (KeyError, TypeError, ValueError) as exc:
      raise RetirementError(f"BAD_TASK_COMPLETION: {exc}") from exc
    state = self._pending.get((tile_id, key))
    if state is None:
      ticket = group.admission.tickets.get((tile_id, key))
      if ticket is not None:
        return self._premature_terminal(ticket.route, tile_id, key, status, cycle)
      return self._reject_unbound(tile_id, key, "TASK_COMPLETION", cycle)
    if state.phase is RetirementPhase.READY:
      return self._premature_terminal(state.route, tile_id, key, status, cycle)
    # Authorization must be effective on the wire: a RETIRE still sitting in the
    # backpressured outbox (never accepted by the Tile) justifies no release.
    if state.sent_seq is None or group.boundary_ports[tile_id].g2t_accepted_seq("CTRL") < state.sent_seq:
      return self._premature_terminal(state.route, tile_id, key, status, cycle)
    ticket = group.admission.tickets.get((tile_id, key))
    if ticket is None:
      # Cancel/fail-stop never drops an external ticket: an authorized Task whose
      # ticket vanished is an invariant violation, not a release path.
      raise MemoryInvariantError(
        f"authorized Task (launch_generation {key.launch_generation}, task {key.task_id}) "
        "lost its admission ticket before TASK_COMPLETION"
      )
    if status == "DONE" and (not ticket.accepted or ticket.final_fault_acknowledged):
      # FAIL_FINAL path: a non-accepted or fault-bound owner may retire only on
      # the failure path -- never emit success under a negative FINAL (a late
      # ACCEPTED cannot overwrite a bound FAULT).
      self._protocol_fault(
        ticket.route.sequencer,
        f"tile {tile_id} TASK_COMPLETION(DONE) under a negative FINAL for Task {key.task_id}",
        cycle,
        "retire_invalid",
        {"tile_id": tile_id, **key.to_wire_fields(), "status": status},
      )
      return "invalid"
    self.stats["task_completions"] += 1
    self._trace(
      tile_id,
      "task_completion",
      cycle,
      {
        "tile_id": tile_id,
        **key.to_wire_fields(),
        "ctx_slot": state.ctx_slot,
        "ctx_generation": state.ctx_generation,
        "status": status,
        "first_fault_ref": first_fault_ref,
        "arena_alloc_gen": state.arena_alloc_gen,
      },
    )
    del self._pending[(tile_id, key)]
    group._finish_route_task(
      state.route, state.task, tile_id, state.ctx_slot, cycle, _TERMINAL_EVENT_STATUS[status]
    )
    # Strictly after the parent close: releases the ticket's EventTable participant ref.
    group.admission.complete(ticket, cycle)
    return "completed"

  def _premature_terminal(
    self, route: _GridRoute, tile_id: int, key: TaskKey, status: str, cycle: int
  ) -> str:
    self._protocol_fault(
      route.sequencer,
      f"tile {tile_id} TASK_COMPLETION before accepted RETIRE authorization for Task {key.task_id}",
      cycle,
      "retire_invalid",
      {"tile_id": tile_id, **key.to_wire_fields(), "status": status},
    )
    return "invalid"

  def _handle_phase_signal(self, tile_id: int, msg: BoundaryMessage, cycle: int) -> str:
    group = self.group
    key = _task_key_from(msg)
    ticket = group.admission.tickets.get((tile_id, key))
    if ticket is None:
      return self._reject_unbound(tile_id, key, "PHASE_SIGNAL", cycle)
    if not ticket.accepted or ticket.route.expected.get(tile_id) != ticket.task:
      self._protocol_fault(
        ticket.route.sequencer,
        f"tile {tile_id} PHASE_SIGNAL from a non-owner admission for Task {ticket.task.task_id}",
        cycle,
        "retire_invalid",
        {"tile_id": tile_id, **key.to_wire_fields()},
      )
      return "invalid"
    # Real aggregation through the persistent EventService; grid identity from the
    # accepted ticket binding, never guessed from the wire. Duplicate root-fault
    # semantics are preserved by the delegate.
    result = group.event_service.handle_phase_signal(msg, grid=ticket.task, cycle=cycle)
    self.stats["phase_delegated"] += 1
    return result

  def advance(self, cycle: int) -> None:
    """Authorize owed RETIREs, then sweep entries whose Route died without a terminal."""
    group = self.group
    for (tile_id, key), state in self._pending.items():
      if state.phase is not RetirementPhase.READY:
        continue
      port = group.boundary_ports.get(tile_id)
      if port is None:
        continue
      ticket = group.admission.tickets.get((tile_id, key))
      # Immutable arena generation from the owner's RESOURCE_UPDATE (recorded on the
      # admission ticket; may have arrived after READY). Never guessed and never a
      # stand-in for parent settlement: not gated on faulted sequencers or R/pins.
      if ticket is None or ticket.arena_alloc_gen is None:
        continue
      if not port.outbox_ready("CTRL"):
        continue
      frame = port.send(
        BoundaryMessage(
          message_type="RETIRE",
          channel="CTRL",
          direction="G2T",
          fields={
            **key.to_wire_fields(),
            "tile_id": tile_id,
            "ctx_slot": state.ctx_slot,
            "ctx_generation": state.ctx_generation,
            "arena_alloc_gen": int(ticket.arena_alloc_gen),
          },
        )
      )
      state.sent_seq = wire_message_seq(frame)
      state.arena_alloc_gen = int(ticket.arena_alloc_gen)
      state.phase = RetirementPhase.AUTHORIZED
      self.stats["retire_authorized"] += 1
      self._trace(
        tile_id,
        "retire_authorized",
        cycle,
        {
          "tile_id": tile_id,
          **key.to_wire_fields(),
          "ctx_slot": state.ctx_slot,
          "ctx_generation": state.ctx_generation,
          "arena_alloc_gen": state.arena_alloc_gen,
          "sent_seq": state.sent_seq,
        },
      )
    # Bounded state, no completion tombstones: entries whose Route was torn down
    # (fault/reset) without a terminal are reclaimed here.
    for bound in [k for k, s in self._pending.items() if s.task.grid not in group._grid_routes]:
      del self._pending[bound]

  # ---- classification + fault helpers -----------------------------------
  def _reject_unbound(self, tile_id: int, key: TaskKey, message_type: str, cycle: int) -> str:
    """Classify a frame attributing to no ticket and no chain state.

    Classification is by launch generation: a generation the Group never handed
    out is a future forgery (ownerless GROUP fault); a generation still present
    in `_live_launches` names a live launch, so ANY unbound key under it --
    including a forged dispatch ordinal -- is a real protocol fault against that
    launch's sequencer, never a silent stale. Only a retired generation counts
    stale with zero effect.
    """
    group = self.group
    args = {"tile_id": tile_id, **key.to_wire_fields(), "message": message_type}
    if key.launch_generation >= group._next_launch_id:
      self._fault(
        f"tile {tile_id} {message_type} names unlaunched generation {key.launch_generation}", cycle, args
      )
      return "invalid"
    seq = next(
      (s for (_name, _slot, gen), s in group._live_launches.items() if gen == key.launch_generation), None
    )
    if seq is not None:
      self._protocol_fault(
        seq,
        f"tile {tile_id} {message_type} names live launch but no accepted owner",
        cycle,
        "retire_invalid",
        args,
      )
      return "invalid"
    self.stats["retire_stale"] += 1
    group.pmu.add_event("retire_stale")
    self._trace(tile_id, "retire_stale", cycle, args)
    return "stale"

  def _reject_duplicate(self, state: _ExternalRetirement, message_type: str, cycle: int) -> str:
    if state.task.grid not in self.group._grid_routes:
      return self._reject_unbound(state.tile_id, state.key, message_type, cycle)
    self._protocol_fault(
      state.route.sequencer,
      f"tile {state.tile_id} Task {state.task.task_id}: duplicate {message_type}",
      cycle,
      "retire_duplicate",
      {"tile_id": state.tile_id, **state.key.to_wire_fields(), "message": message_type},
    )
    return "duplicate"

  def _protocol_fault(
    self, seq: TileGroupSequencer, reason: str, cycle: int, counter: str, args: dict
  ) -> None:
    """Reject against a live owner with the phase-signal root-fault semantics."""
    group = self.group
    self.stats[counter] += 1
    group.pmu.add_event(counter)
    if group.tracer is not None:
      group.tracer.instant("TileGroup", "Boundary:Retirement", counter, cycle, args)
    seq.mark_fault(reason)
    group.scheduler.cancel_context(seq, cycle)
    if not (group.reset_domain.is_active or group.reset_domain.is_done):
      group.trigger_fault(FaultCode.ADDRESS_FAULT, tile_id=-1, cycle=cycle, desc_id=reason)

  def _fault(self, reason: str, cycle: int, args: dict) -> None:
    """Ownerless protocol fault (no attributable Route): one GROUP fault, no sequencer."""
    group = self.group
    self.stats["retire_invalid"] += 1
    group.pmu.add_event("retire_invalid")
    if group.tracer is not None:
      group.tracer.instant("TileGroup", "Boundary:Retirement", "retire_invalid", cycle, args)
    if not (group.reset_domain.is_active or group.reset_domain.is_done):
      group.trigger_fault(FaultCode.ADDRESS_FAULT, tile_id=-1, cycle=cycle, desc_id=reason)

  def _trace(self, tile_id: int, name: str, cycle: int, args: dict) -> None:
    tr = self.group.tracer
    if tr is not None:
      tr.instant(f"Tile{tile_id}", "Boundary:Retirement", name, cycle, args)

  def snapshot(self) -> dict:
    phases = {phase.value: 0 for phase in RetirementPhase}
    for state in self._pending.values():
      phases[state.phase.value] += 1
    return {"stats": dict(self.stats), "pending": len(self._pending), "phases": phases}
