"""Cross-boundary task-admission coordinator (C035 / W1-04).

Python Tiles keep the verbatim migration of ``TileGroup._step_task_admission``
selection plus ``_commit_route_task``: one atomic commit decision per Tile per
Tick, original wait reasons/fault cutlines/trace timing, ``route.admissions``
authoritative, no ticket or extra per-task state on that path, zero boundary
frames by default.

External Tiles (``group.boundary_ports`` ids only) run a bounded multi-ticket
FSM over C017 CTRL: ``PLAN -> PARENT_PREP -> HOLD_REQ -> HOLD_ACKED ->
FINAL_COMMIT -> ACCEPTED``. Keyed ``TICKET_ROLLBACK`` compensates pre-FINAL
failures of a planned transaction. A stale pool snapshot grants no
hold. Re-plan only after compensation and a newer version from the Tile;
the Group never guesses the version or advances the Task generation. Use
``CANCEL_TASK`` + real drain after a FINAL was issued (CANCEL_TASK has no txn
id: no ACK required or trusted; only TASK_COMPLETION via retirement closes).
A pure PLAN WAIT that established no peer ticket closes locally; an
ACK-ambiguous cancel first disambiguates the in-flight PLAN ACK.  A negative
FINAL never rolls back: it drains with parents retained, and an ACK(FAULT) on
a P1-accepted FINAL is the irrevocable acknowledged-failure window
(``ticket.can_retire``; ERROR/TIMEOUT/RESET terminals only, never DONE).

Parent accounting is exact and reversible: PARENT_PREP takes L2 pins, one R
lease slot (cap-checked), and one ``EventTable.acquire_participant`` reference
on the root's already-bound dispatch event (per-root quota reused, never
re-billed); failures unwind in reverse and the ticket is kept until the exact
release proof.  ACKs match only a stage this coordinator issued on the SAME
link whose frame crossed ``port.g2t_accepted_seq('CTRL')``; illegal outcomes,
foreign links and forged stages raise ``AdmissionProtocolError`` with zero
effects.  Enqueued CTRL frames drain FIFO under freeze: ``advance`` keeps
sending owed rollbacks/cancels every Tick.
"""

from __future__ import annotations

import enum
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..execution_ir import TaskIdentity
from ..memory.allocator import (
  AdmissionFailure,
  AdmissionFailureKind,
  MemoryInvariantError,
)
from ..runtime.event_table import EventProtocolError
from . import wire
from .tile_port import BoundaryMessage, TaskKey, wire_message_seq

if TYPE_CHECKING:  # tile_group imports this module at module scope; stay import-cycle free
  from ..tile import TileAdmission
  from ..tile_group import TileGroup, _GridRoute

# Legal ACK outcomes per issued stage (native endpoint contract).  An outcome
# outside a stage's set is malformed and authorizes nothing.
_LEGAL_OUTCOMES: dict[str, frozenset[str]] = {
  "TASK_PLAN": frozenset({"ACCEPTED", "WAIT", "FAULT"}),
  "TASK_COMMIT": frozenset({"HOLD_READY", "WAIT", "FAULT", "STALE"}),
  "TASK_COMMIT_FINAL": frozenset({"ACCEPTED", "WAIT", "FAULT"}),
  "TICKET_ROLLBACK": frozenset({"COMPENSATED", "FAULT", "STALE"}),
}


class AdmissionProtocolError(MemoryInvariantError):
  """A boundary admission response that matches no issued stage; zero effects."""


class AdmissionState(enum.Enum):
  """One admission transaction's bounded FSM (external Tiles)."""

  PLAN = "plan"  # TASK_PLAN outstanding; zero side effects anywhere
  PARENT_PREP = "parent_prep"  # plan ACKed; parent pins/R/participant acquired, hold pending
  HOLD_REQ = "hold_req"  # TASK_COMMIT outstanding (Tile hold may be establishing)
  HOLD_ACKED = "hold_acked"  # Tile confirmed the local hold; FINAL pending
  FINAL_COMMIT = "final_commit"  # TASK_COMMIT_FINAL issued (irreversible cutline)
  ACCEPTED = "accepted"  # FINAL ACKed ACCEPTED; execution may begin
  COMPENSATE = "compensate"  # parent unwound; TICKET_ROLLBACK owed/outstanding, slot held for ACK
  COMPENSATED = "compensated"  # exact release proof collected; ticket closed
  CANCELLING = "cancelling"  # post-FINAL: CANCEL_TASK/drain; only a real terminal closes


@dataclass
class AdmissionTicket:
  """One external admission transaction's live record (bounded: slots x dispatch capacity)."""

  route: _GridRoute
  task: TaskIdentity
  key: TaskKey
  tile_id: int
  txn_id: int
  ctx_slot: int
  ctx_generation: int
  pool_epoch: int
  pool_version: int
  created_cycle: int
  state: AdmissionState = AdmissionState.PLAN
  accepted: bool = False
  arena_alloc_gen: int | None = None  # set only after both HOLD_READY and its resource report
  wait_reason: str | None = None
  cancel_requested: bool = False
  parent_held: bool = False
  stage_seq: dict[str, int] = field(default_factory=dict)  # issued stage -> stamped message_seq
  resolved: dict[str, str] = field(default_factory=dict)  # stage -> first latched outcome
  completed_cycle: int | None = None
  pool_report: tuple[int, int] | None = None  # Tile pool version and slot's observed Arena generation
  # Installation failed after irrevocable authorization: may drain a failure,
  # but never announce Task acceptance or DONE.
  final_fault_acknowledged: bool = False
  # C036: event-driven L1 gate park epoch when HOLD is held back in PARENT_PREP.
  wait_gate_epoch: int | None = None

  @property
  def can_retire(self) -> bool:
    """RETIRE_READY binds to accepted ownership or the acknowledged-failure
    window; PHASE_SIGNAL and a DONE terminal still require ``accepted``."""
    return self.accepted or self.final_fault_acknowledged


class AdmissionCoordinator:
  """Group-side admission owner for one ``TileGroup`` (Python + external paths)."""

  def __init__(
    self,
    group: TileGroup,
    context_count: int,
    *,
    pool_epoch: int = 0,
    pool_version: int = 0,
  ) -> None:
    if isinstance(context_count, bool) or not isinstance(context_count, int) or context_count < 1:
      raise ValueError("context_count must be a positive int")
    if isinstance(pool_epoch, bool) or not isinstance(pool_epoch, int) or pool_epoch < 0:
      raise ValueError("pool_epoch must be a non-negative int")
    if isinstance(pool_version, bool) or not isinstance(pool_version, int) or pool_version < 0:
      raise ValueError("pool_version must be a non-negative int")
    self.group = group
    self.context_count = context_count
    # Remote L1 owner's pool identity: fixture-declared initial values; the
    # per-link mirror advances only from a matched HOLD response and resource report.
    self.pool_epoch = pool_epoch
    self.pool_version = pool_version
    self.tickets: dict[tuple[int, TaskKey], AdmissionTicket] = {}
    self._txn_index: dict[int, AdmissionTicket] = {}
    self._slot_owner: dict[int, dict[int, AdmissionTicket]] = {}
    self._slot_generations: dict[int, list[int]] = {}
    self._link_pool_version: dict[int, int] = {}
    self._stale_reports: dict[tuple[int, int], tuple[int, int]] = {}  # one prior retry sample per slot
    # Closed transactions are retained only within the C017 dedup window.
    self._resolved: OrderedDict[int, tuple[int, frozenset[tuple[str, str]]]] = OrderedDict()
    self._next_txn_id = 1
    # C036: event-driven L1 gate wake bookkeeping (no per-tick gate polling). The latch is
    # written only by GateWakeCoordinator.register() on every L1 transition; step() never
    # reads ProfileController. The gate is closed until the controller's first open.
    self._l1_closed = True
    self._l1_gate_parked_epoch: int | None = None
    self._l1_gate_parked_route_seq: int = -1
    self.stats: dict[str, int] = {
      "tickets_accepted": 0,
      "tickets_wait_closed": 0,
      "tickets_faulted": 0,
      "tickets_compensated": 0,
      "tickets_cancelled": 0,
      "tickets_completed": 0,
      "acks_stale": 0,
      "resource_updates_matched": 0,
      "resource_updates_stale": 0,
      "resource_updates_unmatched": 0,
      "gate_park_events": 0,
      "gate_wake_events": 0,
      "gate_suppressed_ticks": 0,
      "hold_gate_park_events": 0,
      "hold_gate_wake_events": 0,
      "hold_gate_suppressed_ticks": 0,
      "candidate_scans": 0,
    }

  # ---- candidate selection (migrated from TileGroup._step_task_admission) ----

  def note_l1_gate(self, closed: bool) -> None:
    """L1 transition latch: the only input that opens or closes step()."""
    self._l1_closed = closed

  def wake_on_l1_gate(self, gate_epoch: int, cycle: int) -> bool:
    """Record one coalesced L1-gate open wake for step() and gate-parked HOLD tickets."""
    _ = cycle
    woke = False
    if self._l1_gate_parked_epoch is not None and gate_epoch > self._l1_gate_parked_epoch:
      self._l1_gate_parked_epoch = None
      self._l1_gate_parked_route_seq = -1
      self.stats["gate_wake_events"] += 1
      woke = True
    woke_holds = 0
    for ticket in self.tickets.values():
      if ticket.wait_gate_epoch is not None and gate_epoch > ticket.wait_gate_epoch:
        ticket.wait_gate_epoch = None
        if ticket.wait_reason == "WAIT_CONTROL_RESOURCE":
          ticket.wait_reason = None
        if ticket.route.wait_reasons.get(ticket.tile_id) == "WAIT_CONTROL_RESOURCE":
          ticket.route.wait_reasons.pop(ticket.tile_id, None)
        woke_holds += 1
    if woke_holds:
      self.stats["hold_gate_wake_events"] += woke_holds
      woke = True
    return woke

  def clear_gate_wait(self) -> None:
    self._l1_gate_parked_epoch = None
    self._l1_gate_parked_route_seq = -1
    for ticket in self.tickets.values():
      ticket.wait_gate_epoch = None

  def step(self, cycle: int) -> None:
    """Start candidates: one commit decision per Tile per Tick, deterministic."""
    group = self.group
    if self._l1_closed:
      gate_epoch = group.gate_wake.epoch("l1")
      route_seq = group._route_sequence
      if self._l1_gate_parked_epoch == gate_epoch and self._l1_gate_parked_route_seq == route_seq:
        self.stats["gate_suppressed_ticks"] += 1
        return
      self._l1_gate_parked_epoch = gate_epoch
      self._l1_gate_parked_route_seq = route_seq
      self.stats["gate_park_events"] += 1
      for queued_route in group._grid_routes.values():
        if queued_route.sequencer.faulted:
          continue
        for tile_id in queued_route.expected:
          if tile_id in queued_route.admissions or tile_id in queued_route.retired:
            continue
          tile = group.python_tiles_by_id.get(tile_id)
          if tile is None and (tile_id, self._task_key(queued_route.expected[tile_id])) in self.tickets:
            continue
          queued_route.wait_reasons[tile_id] = "WAIT_CONTROL_RESOURCE"
      return
    self._l1_gate_parked_epoch = None
    self._l1_gate_parked_route_seq = -1
    if not group._grid_routes:
      return
    self.stats["candidate_scans"] += 1
    current = group.profile_controller.active_modes["l1"]
    for tile_id in range(group.cfg.num_tiles):
      tile = group.python_tiles_by_id.get(tile_id)
      heads: dict[bool, _GridRoute] = {}
      for queued_route in group._grid_routes.values():
        if (
          queued_route.sequencer.faulted
          or tile_id not in queued_route.expected
          or tile_id in queued_route.admissions
          or tile_id in queued_route.retired
        ):
          continue
        if tile is None and (tile_id, self._task_key(queued_route.expected[tile_id])) in self.tickets:
          continue  # a live external ticket (incl. compensating) owns this task
        same = queued_route.request.requested_l1_mode == current
        heads.setdefault(same, queued_route)
      for same in (True, False):
        route = heads.get(same)
        if route is None:
          continue
        seq = route.sequencer
        assert seq.task is not None and seq.task.resource_contract is not None
        task = route.expected[tile_id]
        key = (seq.task.binding_id, seq.context_launch_generation, tile_id)
        leases = group._task_leases.get(key)
        if leases is not None and len(leases) >= seq.task.resource_contract.requested_contexts_per_tile:
          route.wait_reasons[tile_id] = "WAIT_CONTEXT_LIMIT"
          continue
        if tile is not None:
          candidate = tile.plan_admission(
            route.binding.tile_program, route.grid, route.event_id, task.task_id, route.binding.context_id
          )
          if candidate is None:
            route.wait_reasons[tile_id] = "WAIT_SLOT"
            continue
          if isinstance(candidate, AdmissionFailure):
            if candidate.kind is AdmissionFailureKind.TEMPORARY_CAPACITY:
              assert candidate.wait_reason is not None
              route.wait_reasons[tile_id] = candidate.wait_reason.value
              continue
            seq.mark_fault(candidate.reason)
            group.on_scheduler_fault(seq, candidate.reason, cycle)
            break
          try:
            self._commit_python(route, candidate, task, cycle)
          except (MemoryInvariantError, ValueError) as exc:
            seq.mark_fault(f"Task admission failed: {exc}")
            group.on_scheduler_fault(seq, seq.fault_reason, cycle)
            break
          group._task_leases.setdefault(key, set()).add(task)
          route.admissions[tile_id] = candidate
          route.wait_reasons.pop(tile_id, None)
          group._trace_task_lease("task_lease_acquire", route, task, tile_id, cycle, candidate.context_id)
          break
        if self._start_external(route, task, tile_id, cycle):
          break
        continue

  # ---- Python commit body (migrated verbatim from TileGroup._commit_route_task) ----

  def _commit_python(
    self, route: _GridRoute, admission: TileAdmission, task: TaskIdentity, cycle: int
  ) -> None:
    from ..tile import _TileContextMemory

    group = self.group
    binding, seq, tile = route.binding, route.sequencer, admission.tile
    program = binding.tile_program
    generation = seq.context_launch_generation
    l2_map = {
      index: group._l2_handles[(generation, slot)]
      for (index, _formal), slot in zip(
        ((i, f) for i, f in enumerate(program.formals) if f.space == "l2"), binding.actuals
      )
    }
    global_map = {}
    global_formals = [(i, formal) for i, formal in enumerate(program.formals) if formal.space == "global"]
    if len(global_formals) != len(binding.global_actuals):
      raise MemoryInvariantError("global actual/formal count mismatch")
    for (index, _formal), actual in zip(global_formals, binding.global_actuals):
      view = group._resolve_view(actual, generation, seq.formal_bindings)
      if view is None:
        raise MemoryInvariantError("missing global actual view")
      global_map[index] = view
    tile.commit_admission(admission, program, cycle)
    pinned = False
    try:
      group._pin_grid_l2(route.grid, binding, task, cycle)
      pinned = True
      group.program_table.register(
        program.program_id, program.version, program.program_hash, 0, program.text_bytes
      )
      admission.prepare_cycles = (
        group.program_table.ensure_resident(program.program_id, tile.tile_id, cycle)
        if group.runtime_enabled
        else 0
      )
      memory = _TileContextMemory(
        task_identity=task,
        l2_formal_handles=l2_map,
        global_formal_views=global_map,
        l1_handles=admission.l1_handles,
        l2_resolver=group.l2_sram,
        arena=admission.arena,
        binding_id=route.request.binding_id,
      )
      context = tile.load_program(
        program,
        binding.role_id,
        route.event_id,
        admission.prepare_cycles,
        admission.context_id,
        memory,
        task,
      )
      if context != admission.context_id:
        raise MemoryInvariantError("UCE binding failed after Arena commit")
      admission.bound = True
      group.profile_controller.note_user_issue(("l1",))
    except (MemoryInvariantError, ValueError):
      if pinned:
        group._unpin_task_l2(task, cycle)
      tile.abort_admission(admission, cycle)
      raise
    group._role_l1_handles[route.event_id][tile.tile_id] = admission.l1_handles
    tile.uce._phase_signal_callback = group._on_phase_signal
    for queue_id, queue in group.queues.items():
      tile.bind_stream(queue_id, queue)
    group.pmu.add_cycle("program_cold_load", admission.prepare_cycles)

  # ---- external boundary path ------------------------------------------

  def _start_external(
    self, route: _GridRoute, task: TaskIdentity, tile_id: int, cycle: int
  ) -> bool:
    """Take ONE new external commit decision; False means a pure finite wait."""
    group = self.group
    port = group.boundary_ports.get(tile_id)
    if port is None:
      raise MemoryInvariantError(f"tile {tile_id} has neither a local ComputeTile nor a boundary port")
    owners = self._slot_owner.setdefault(tile_id, {})
    # Bounded admission-transaction metadata is separate from local Context
    # occupancy: dispatch_capacity bounds live tickets per Tile, context slots
    # bound the Tile-local hold layout.  Nothing L1 is reserved at PLAN.
    if len(owners) >= group.scheduler_config.dispatch_capacity:
      route.wait_reasons[tile_id] = "WAIT_CONTROL_RESOURCE"
      return False
    preferred = route.binding.context_id
    if preferred is not None and not 0 <= preferred < self.context_count:
      self._fault(route, f"tile {tile_id}: authored context id {preferred} is out of range", cycle)
      self.stats["tickets_faulted"] += 1
      return False
    if preferred is not None:
      # Authored slot pinning is strict: an occupied authored slot waits and
      # never falls through to another free slot.
      slot: int | None = preferred
    else:
      slot = next((s for s in range(self.context_count) if s not in owners), None)
    if slot is None or slot in owners:
      route.wait_reasons[tile_id] = "WAIT_SLOT"
      return False
    if not port.outbox_ready("CTRL") or group.gate_wake.has_pending_notify(tile_id, "l1"):
      route.wait_reasons[tile_id] = "WAIT_CONTROL_RESOURCE"
      return False
    ticket = AdmissionTicket(
      route=route,
      task=task,
      key=self._task_key(task),
      tile_id=tile_id,
      txn_id=self._next_txn_id,
      ctx_slot=slot,
      ctx_generation=self._slot_generations.setdefault(tile_id, [0] * self.context_count)[slot],
      pool_epoch=self.pool_epoch,
      pool_version=self._link_pool_version.get(tile_id, self.pool_version),
      created_cycle=cycle,
    )
    self._next_txn_id += 1
    owners[slot] = ticket
    self.tickets[(tile_id, ticket.key)] = ticket
    self._txn_index[ticket.txn_id] = ticket
    self._send_ctrl(
      ticket,
      "TASK_PLAN",
      {"txn_id": ticket.txn_id, "pool_epoch": ticket.pool_epoch, "pool_version": ticket.pool_version},
      cycle,
    )
    route.wait_reasons.pop(tile_id, None)
    return True

  def advance(self, cycle: int) -> None:
    """Progress already-started transactions; runs even while the Group is frozen."""
    finals_authorized: set[int] = set()
    for ticket in tuple(self.tickets.values()):
      if ticket.state is AdmissionState.PARENT_PREP:
        if "TASK_COMMIT" not in ticket.stage_seq:
          if ticket.wait_gate_epoch is not None:
            self.stats["hold_gate_suppressed_ticks"] += 1
            continue
          if self.group.profile_controller.issue_gate_closed("l1"):
            ticket.wait_gate_epoch = self.group.gate_wake.epoch("l1")
            ticket.wait_reason = "WAIT_CONTROL_RESOURCE"
            ticket.route.wait_reasons[ticket.tile_id] = "WAIT_CONTROL_RESOURCE"
            self.stats["hold_gate_park_events"] += 1
            continue
          if self.group.gate_wake.has_pending_notify(ticket.tile_id, "l1"):
            continue
          port = self._port_or_fail(ticket)
          if not ticket.parent_held:
            if not port.outbox_ready("CTRL"):
              continue  # never take parent books while unable to proceed this Tick
            if not self._acquire_parent(ticket, cycle):
              continue  # wait-closed or faulted; the ticket is gone
          if port.outbox_ready("CTRL"):
            self._send_ctrl(
              ticket,
              "TASK_COMMIT",
              {
                "txn_id": ticket.txn_id,
                "ctx_slot": ticket.ctx_slot,
                "ctx_generation": ticket.ctx_generation,
                "pool_version": ticket.pool_version,
              },
              cycle,
            )
            ticket.state = AdmissionState.HOLD_REQ
      elif ticket.state is AdmissionState.HOLD_ACKED:
        # FINAL is the one irreversible authorization: at most one per Tile per
        # Tick, even when several holds pile up behind CTRL credit.
        if ticket.tile_id in finals_authorized:
          continue
        port = self._port_or_fail(ticket)
        if port.outbox_ready("CTRL"):
          self._send_ctrl(
            ticket,
            "TASK_COMMIT_FINAL",
            {"txn_id": ticket.txn_id, "ctx_slot": ticket.ctx_slot, "ctx_generation": ticket.ctx_generation},
            cycle,
          )
          ticket.state = AdmissionState.FINAL_COMMIT
          finals_authorized.add(ticket.tile_id)
      elif ticket.state is AdmissionState.COMPENSATE:
        self._send_rollback_if_owed(ticket, cycle)
      elif ticket.state is AdmissionState.CANCELLING:
        self._send_cancel_if_owed(ticket, cycle)

  def handle(self, tile_id: int, msg: BoundaryMessage, cycle: int) -> None:
    """Consume one T2G ACK / RESOURCE_UPDATE from ``tile_id``'s link."""
    if msg.direction != "T2G":
      raise AdmissionProtocolError(f"admission consumes T2G only, got {msg.message_type}/{msg.direction}")
    if msg.message_type == "ACK":
      self._handle_ack(tile_id, msg, cycle)
    elif msg.message_type == "RESOURCE_UPDATE":
      self._handle_resource_update(tile_id, msg, cycle)
    else:
      raise AdmissionProtocolError(f"admission cannot consume {msg.message_type} from tile {tile_id}")

  # ---- inbound ACK resolution -------------------------------------------

  def _handle_ack(self, tile_id: int, msg: BoundaryMessage, cycle: int) -> None:
    try:
      txn_id = int(msg.fields["txn_id"])
      ack_for = str(msg.fields["ack_for_message_type"])
      outcome = msg.fields["outcome"]
      wait_reason = msg.fields.get("wait_reason")
    except (KeyError, TypeError, ValueError) as exc:
      raise AdmissionProtocolError(f"tile {tile_id}: malformed ACK: {exc}") from exc
    # Link validation first, including closed history; nothing mutates before it.
    ticket = self._txn_index.get(txn_id)
    if ticket is None:
      resolved = self._resolved.get(txn_id)
      if resolved is not None and resolved[0] == tile_id and (ack_for, outcome) in resolved[1]:
        self.stats["acks_stale"] += 1  # bounded replay of an already-closed transaction
        return
      raise AdmissionProtocolError(f"tile {tile_id}: ACK for unissued transaction {txn_id} ({ack_for})")
    if ticket.tile_id != tile_id:
      raise AdmissionProtocolError(
        f"tile {tile_id}: ACK for txn {txn_id} owned by tile {ticket.tile_id}"
      )
    stage_seq = ticket.stage_seq.get(ack_for)
    if stage_seq is None:
      raise AdmissionProtocolError(
        f"tile {tile_id}: ACK for unissued stage {ack_for} of txn {txn_id} in {ticket.state.name}"
      )
    if outcome is None or outcome not in _LEGAL_OUTCOMES.get(ack_for, frozenset()):
      raise AdmissionProtocolError(
        f"tile {tile_id}: txn {txn_id} stage {ack_for} carries illegal outcome {outcome!r}"
      )
    # P1 receipt proof: the link accepted the request frame before its ACK may
    # consume or release anything (forged/early ACK fails closed).
    watermark = self._port_or_fail(ticket).g2t_accepted_seq("CTRL")
    if stage_seq > watermark:
      raise AdmissionProtocolError(
        f"tile {tile_id}: ACK for txn {txn_id} stage {ack_for} was not accepted by the link "
        f"(seq {stage_seq}, accepted watermark {watermark})"
      )

    # Exactly one resolved outcome per issued stage: an identical duplicate is
    # inert; a conflicting re-answer is a protocol error.
    prior = ticket.resolved.get(ack_for)
    if prior is not None:
      if prior == outcome:
        self.stats["acks_stale"] += 1
        return
      raise AdmissionProtocolError(
        f"tile {tile_id}: txn {txn_id} stage {ack_for} re-answered {outcome!r} after {prior!r}"
      )
    ticket.resolved[ack_for] = outcome

    if ack_for == "TASK_PLAN":
      if ticket.state is not AdmissionState.PLAN:
        self.stats["acks_stale"] += 1
        return
      if outcome == "ACCEPTED":
        # A wire-accepted PLAN leaves bounded PLANNED metadata on the Tile:
        # a cancel-requested ticket must compensate it out for real.
        if ticket.cancel_requested:
          self._begin_compensation(ticket, cycle)
        else:
          ticket.state = AdmissionState.PARENT_PREP
      elif outcome == "WAIT":
        # Pure PLAN WAIT established no peer ticket: nothing to roll back.
        self._close_wait(ticket, wait_reason or "WAIT_CONTROL_RESOURCE")
        if ticket.cancel_requested:
          self._fault_cancelled(ticket, cycle)
      else:  # FAULT: no record was created for this txn, but the pairing broke
        self._fault_and_close(ticket, cycle, f"tile {tile_id}: TASK_PLAN {outcome} for txn {txn_id}")
      return

    if ack_for == "TASK_COMMIT":
      self._confirm_pool_report(ticket)
      if ticket.state is not AdmissionState.HOLD_REQ:
        # Late response under compensation/cancel: the rollback (or the drain)
        # already owns this transaction's Tile-side release.
        self.stats["acks_stale"] += 1
        return
      if outcome == "HOLD_READY":
        ticket.state = AdmissionState.HOLD_ACKED
      elif outcome == "WAIT":
        # No hold was established, but the native PLANNED metadata persists:
        # compensate it out through the exact proof before any fresh retry.
        self._begin_compensation(ticket, cycle, wait_reason=wait_reason or "WAIT_CONTROL_RESOURCE")
      elif outcome == "STALE":
        # The same-generation version report may arrive after either CTRL ACK.
        self._begin_compensation(ticket, cycle, wait_reason="STALE_VERSION")
      else:
        self._begin_compensation(ticket, cycle, fault_reason=f"tile {tile_id}: HOLD fault for {txn_id}")
      return

    if ack_for == "TASK_COMMIT_FINAL":
      # The FIRST valid FINAL outcome latches regardless of a prior cancel: a
      # FAULT under CANCELLING still binds the acknowledged-failure window
      # (can_retire) and initiates the root fail-stop with parents retained,
      # and the stage latch above keeps any later ACCEPTED replay from
      # flipping a fault-bound owner into accepted (or vice versa).
      if ticket.state is AdmissionState.CANCELLING:
        if outcome == "ACCEPTED":
          ticket.accepted = True  # first resolution only; the drain owns the outcome
        elif outcome == "FAULT":
          ticket.final_fault_acknowledged = True
          self._fault(
            ticket.route, f"tile {tile_id}: TASK_COMMIT_FINAL FAULT for txn {txn_id}", cycle
          )
          self.stats["tickets_faulted"] += 1
        return  # WAIT (or a latched echo) leaves the drain waiting on the terminal
      if ticket.state is not AdmissionState.FINAL_COMMIT:
        self.stats["acks_stale"] += 1
        return
      if outcome == "ACCEPTED":
        ticket.state = AdmissionState.ACCEPTED
        ticket.accepted = True
        self.stats["tickets_accepted"] += 1
        self._trace(ticket, "admission_accepted", cycle)
      else:
        # No rollback after an issued FINAL: fail closed, parents retained,
        # the Task drains for real; without a genuine terminal this stays
        # INCOMPLETE.
        ticket.state = AdmissionState.CANCELLING
        if outcome == "FAULT":
          ticket.final_fault_acknowledged = True
          self._fault(
            ticket.route, f"tile {tile_id}: TASK_COMMIT_FINAL FAULT for txn {txn_id}", cycle
          )
          self.stats["tickets_faulted"] += 1
        self._send_cancel_if_owed(ticket, cycle)
      return

    # ack_for == "TICKET_ROLLBACK"; state must be COMPENSATE here.
    if ticket.state is not AdmissionState.COMPENSATE:
      self.stats["acks_stale"] += 1
      return
    if outcome == "COMPENSATED":
      self._finish_compensation(ticket, cycle)
    else:  # FAULT / STALE: refused rollback keeps every resource held
      self._fault(
        ticket.route,
        f"tile {ticket.tile_id}: TICKET_ROLLBACK {outcome} for txn {ticket.txn_id}; resources stay held",
        cycle,
      )

  def _handle_resource_update(self, tile_id: int, msg: BoundaryMessage, cycle: int) -> None:
    try:
      claimed_tile = int(msg.fields["tile_id"])
      slot = int(msg.fields["ctx_slot"])
      generation = int(msg.fields["ctx_generation"])
      version = int(msg.fields["version"])
      arena_alloc_gen = int(msg.fields["arena_alloc_gen"])
    except (KeyError, TypeError, ValueError) as exc:
      raise AdmissionProtocolError(f"tile {tile_id}: malformed RESOURCE_UPDATE: {exc}") from exc
    if claimed_tile != tile_id:
      raise AdmissionProtocolError(f"RESOURCE_UPDATE claims tile {claimed_tile} on tile {tile_id}'s link")
    ticket = self._slot_owner.get(tile_id, {}).get(slot)
    if ticket is None or ticket.ctx_generation != generation:
      self.stats["resource_updates_unmatched"] += 1
      return
    report = (version, arena_alloc_gen)
    if version < ticket.pool_version or report == self._stale_reports.get((tile_id, slot)):
      # Version-only retries retain their Context generation; a delayed old
      # sample may not install the slot's previous Arena on the new ticket.
      self.stats["resource_updates_stale"] += 1
      return
    # The hold must have crossed P1: the endpoint establishes (and reports)
    # only after the CTRL link accepted TASK_COMMIT, so an earlier update is
    # forged and grants nothing.
    hold_seq = ticket.stage_seq.get("TASK_COMMIT")
    if hold_seq is None or hold_seq > self._port_or_fail(ticket).g2t_accepted_seq("CTRL"):
      raise AdmissionProtocolError(f"tile {tile_id}: RESOURCE_UPDATE precedes its accepted TASK_COMMIT")
    if ticket.pool_report is not None:
      if report != ticket.pool_report:
        raise AdmissionProtocolError(f"tile {tile_id}: conflicting resource report for txn {ticket.txn_id}")
      self.stats["resource_updates_stale"] += 1
      return
    ticket.pool_report = report
    self._confirm_pool_report(ticket)
    if ticket.resolved.get("TICKET_ROLLBACK") == "COMPENSATED":
      self._finish_compensation(ticket, cycle)
    self.stats["resource_updates_matched"] += 1

  def _confirm_pool_report(self, ticket: AdmissionTicket) -> None:
    outcome = ticket.resolved.get("TASK_COMMIT")
    if ticket.pool_report is None or outcome not in ("HOLD_READY", "STALE"):
      return
    version, arena = ticket.pool_report
    if outcome == "HOLD_READY":
      ticket.arena_alloc_gen = arena
    elif version == ticket.pool_version:
      ticket.pool_report = None  # STALE alone is not evidence of a newer pool version
      return
    # With STALE, `arena` is the slot's historical allocation generation, not a grant.
    # An earlier valid pair still identifies its own Arena but cannot regress
    # the link watermark if another ticket's CTRL/EVENT pair completed first.
    self._link_pool_version[ticket.tile_id] = max(self._link_pool_version.get(ticket.tile_id, 0), version)

  # ---- parent acquisition / release (exact, reversible) ------------------

  def _acquire_parent(self, ticket: AdmissionTicket, cycle: int) -> bool:
    """PARENT_PREP: L2 pins -> R lease -> dispatch participant. False = closed."""
    group = self.group
    route, task = ticket.route, ticket.task
    seq = route.sequencer
    assert seq.task is not None and seq.task.resource_contract is not None
    lease_key = (seq.task.binding_id, seq.context_launch_generation, ticket.tile_id)
    leases = group._task_leases.get(lease_key)
    if leases is not None and len(leases) >= seq.task.resource_contract.requested_contexts_per_tile:
      # R competition: no hold exists yet, but the ACKed PLAN's native
      # metadata does -- compensate it out before a fresh retry may re-plan.
      self._begin_compensation(ticket, cycle, wait_reason="WAIT_CONTEXT_LIMIT")
      return False
    try:
      group._pin_grid_l2(route.grid, route.binding, task, cycle)
    except (MemoryInvariantError, ValueError) as exc:
      self._begin_compensation(ticket, cycle, fault_reason=f"Task admission failed: {exc}")
      return False
    group._task_leases.setdefault(lease_key, set()).add(task)
    try:
      group.event_table.acquire_participant(
        route.event_id,
        owner=group.scheduler.event_owner(seq),
        generation=seq.context_launch_generation,
      )
    except EventProtocolError as exc:
      held = group._task_leases.get(lease_key)
      if held is not None:
        held.discard(task)
        if not held:
          del group._task_leases[lease_key]
      group._unpin_task_l2(task, cycle)
      self._begin_compensation(
        ticket, cycle, fault_reason=f"Task admission failed: dispatch participant: {exc}"
      )
      return False
    ticket.parent_held = True
    # The R lease is the ledger truth; public acceptance is traced separately
    # at the FINAL ACCEPTED point.
    group._trace_task_lease("task_lease_acquire", route, task, ticket.tile_id, cycle, ticket.ctx_slot)
    return True

  def _release_parent(self, ticket: AdmissionTicket, cycle: int) -> None:
    """Unwind exactly what PARENT_PREP took, in reverse; idempotent."""
    if not ticket.parent_held:
      return
    group = self.group
    seq = ticket.route.sequencer
    try:
      group.event_table.release_participant(
        ticket.route.event_id,
        owner=group.scheduler.event_owner(seq),
        generation=seq.context_launch_generation,
        cycle=cycle,
      )
    except EventProtocolError as exc:
      raise AdmissionProtocolError(f"txn {ticket.txn_id}: participant release rejected: {exc}") from exc
    lease_key = (ticket.route.request.binding_id, ticket.task.grid.launch_generation, ticket.tile_id)
    leases = group._task_leases.get(lease_key)
    if leases is not None and ticket.task in leases:
      leases.remove(ticket.task)
      if not leases:
        del group._task_leases[lease_key]
    group._unpin_task_l2(ticket.task, cycle)
    group._trace_task_lease(
      "task_lease_release", ticket.route, ticket.task, ticket.tile_id, cycle, ticket.ctx_slot
    )
    ticket.parent_held = False

  # ---- cancellation ------------------------------------------------------

  def cancel(self, tile_id: int, key: TaskKey, cycle: int) -> None:
    """Cancel one live ticket; stale/double cancels never disturb a live txn."""
    ticket = self.tickets.get((tile_id, key))
    if ticket is not None:
      self._cancel_ticket(ticket, cycle)

  def cancel_all(self, cycle: int) -> None:
    """Cancel every live ticket in deterministic (tile, txn) order; idempotent."""
    for ticket in tuple(self.tickets.values()):
      self._cancel_ticket(ticket, cycle)

  def _cancel_ticket(self, ticket: AdmissionTicket, cycle: int) -> None:
    if ticket.state is AdmissionState.COMPENSATE:
      ticket.cancel_requested = True
      if ticket.resolved.get("TICKET_ROLLBACK") == "COMPENSATED":
        self._finish_compensation(ticket, cycle)
      return
    if ticket.state in (AdmissionState.CANCELLING, AdmissionState.COMPENSATED):
      return
    if ticket.completed_cycle is not None:
      return
    ticket.cancel_requested = True
    self.stats["tickets_cancelled"] += 1
    if ticket.state is AdmissionState.PLAN:
      # TASK_PLAN ACK still in flight: mark and disambiguate on its ACK (an
      # accepted PLAN leaves native metadata that must be rolled back; a pure
      # WAIT/FAULT leaves nothing and closes locally).
      return
    if ticket.state in (AdmissionState.PARENT_PREP, AdmissionState.HOLD_REQ, AdmissionState.HOLD_ACKED):
      # Pre-FINAL with the PLAN already wire-accepted: unwind the parent now,
      # send the keyed rollback and hold the control slot through the exact
      # COMPENSATED proof; the logical root fault follows the proof.
      self._begin_compensation(ticket, cycle)
      return
    # FINAL_COMMIT / ACCEPTED: the CTRL FIFO holds the FINAL, so the cancel
    # drains for real; success is only the TASK_COMPLETION terminal.
    ticket.state = AdmissionState.CANCELLING
    self._send_cancel_if_owed(ticket, cycle)

  # ---- completion --------------------------------------------------------

  def complete(self, ticket: AdmissionTicket, cycle: int) -> None:
    """Drop one external ticket after the shared parent close (retirement path)."""
    if self.tickets.get((ticket.tile_id, ticket.key)) is not ticket:
      if ticket.state is AdmissionState.COMPENSATED or ticket.completed_cycle is not None:
        return  # legitimately closed through compensation or a prior complete()
      raise AdmissionProtocolError("completion references a vanished non-completed admission ticket")
    if ticket.state not in (AdmissionState.ACCEPTED, AdmissionState.CANCELLING):
      raise AdmissionProtocolError(f"txn {ticket.txn_id} in state {ticket.state.name} cannot complete")
    if ticket.parent_held:
      # Retirement already released the pins/R lease (_finish_route_task);
      # the dispatch participant reference is ours and is released last.
      seq = ticket.route.sequencer
      try:
        self.group.event_table.release_participant(
          ticket.route.event_id,
          owner=self.group.scheduler.event_owner(seq),
          generation=seq.context_launch_generation,
            cycle=cycle,
        )
      except EventProtocolError as exc:
        raise AdmissionProtocolError(f"txn {ticket.txn_id}: participant release rejected: {exc}") from exc
      ticket.parent_held = False
    self._drop(ticket)
    ticket.completed_cycle = cycle
    self.stats["tickets_completed"] += 1

  # ---- shared helpers ----------------------------------------------------

  @staticmethod
  def _task_key(task: TaskIdentity) -> TaskKey:
    grid = task.grid
    return TaskKey(
      launch_generation=grid.launch_generation,
      root_launch_generation=grid.launch_generation,
      dispatch_ordinal=grid.dispatch_ordinal,
      task_id=task.task_id,
      context_handle=grid.device_slot,
    )

  def _port_or_fail(self, ticket: AdmissionTicket):
    port = self.group.boundary_ports.get(ticket.tile_id)
    if port is None:
      raise MemoryInvariantError(f"tile {ticket.tile_id} lost its boundary port mid-transaction")
    return port

  def _close_wait(self, ticket: AdmissionTicket, wait_reason: str) -> None:
    """Close a ticket on a true WAIT: zero hold anywhere, fresh retry allowed."""
    ticket.wait_reason = wait_reason
    ticket.route.wait_reasons[ticket.tile_id] = wait_reason
    self.stats["tickets_wait_closed"] += 1
    ticket.state = AdmissionState.COMPENSATED
    self._drop(ticket)

  def _begin_compensation(
    self,
    ticket: AdmissionTicket,
    cycle: int,
    *,
    wait_reason: str | None = None,
    fault_reason: str | None = None,
  ) -> None:
    """Pre-FINAL unwind: parent released now, keyed TICKET_ROLLBACK owed, the
    control slot held through the exact COMPENSATED proof before any retry."""
    self._release_parent(ticket, cycle)
    ticket.wait_gate_epoch = None
    ticket.state = AdmissionState.COMPENSATE
    if wait_reason is not None:
      ticket.wait_reason = wait_reason
      ticket.route.wait_reasons[ticket.tile_id] = wait_reason
    if fault_reason is not None:
      self._fault(ticket.route, fault_reason, cycle)
      self.stats["tickets_faulted"] += 1
    self.stats["tickets_compensated"] += 1
    self._send_rollback_if_owed(ticket, cycle)

  def _finish_compensation(self, ticket: AdmissionTicket, cycle: int) -> None:
    """ACK(COMPENSATED) collected: the exact Tile-side release proof."""
    if (
      ticket.resolved.get("TASK_COMMIT") == "STALE" and ticket.pool_report is None
      and not ticket.cancel_requested and not ticket.route.sequencer.faulted
    ):
      return  # parent refs are free; retain the control record until a fresh version is known
    ticket.state = AdmissionState.COMPENSATED
    self._drop(ticket)
    if ticket.cancel_requested:
      self._fault_cancelled(ticket, cycle)

  def _fault_and_close(self, ticket: AdmissionTicket, cycle: int, reason: str) -> None:
    """Close a ticket whose PLAN created nothing on the Tile (never accepted)."""
    self._fault(ticket.route, reason, cycle)
    self.stats["tickets_faulted"] += 1
    ticket.state = AdmissionState.COMPENSATED
    self._drop(ticket)

  def _fault_cancelled(self, ticket: AdmissionTicket, cycle: int) -> None:
    """After an explicit pre-FINAL cancellation no new acceptance may happen."""
    if not ticket.route.sequencer.faulted:
      self._fault(
        ticket.route,
        f"tile {ticket.tile_id} Task {ticket.task.task_id}: admission cancelled before FINAL",
        cycle,
      )

  def _fault(self, route: _GridRoute, reason: str, cycle: int) -> None:
    seq = route.sequencer
    seq.mark_fault(reason)
    self.group.on_scheduler_fault(seq, reason, cycle)

  def _drop(self, ticket: AdmissionTicket) -> None:
    self.tickets.pop((ticket.tile_id, ticket.key), None)
    self._txn_index.pop(ticket.txn_id, None)
    owners = self._slot_owner.get(ticket.tile_id)
    if owners is not None and owners.get(ticket.ctx_slot) is ticket:
      del owners[ticket.ctx_slot]
      slot_key = (ticket.tile_id, ticket.ctx_slot)
      if ticket.resolved.get("TASK_COMMIT") == "STALE" and ticket.pool_report is not None:
        self._stale_reports[slot_key] = ticket.pool_report  # version drift never occupied this Context
      else:
        self._stale_reports.pop(slot_key, None)
        self._slot_generations[ticket.tile_id][ticket.ctx_slot] += 1
    self._resolved[ticket.txn_id] = (ticket.tile_id, frozenset(ticket.resolved.items()))
    while len(self._resolved) > wire.DEDUP_ENTRIES:
      self._resolved.popitem(last=False)

  def _send_rollback_if_owed(self, ticket: AdmissionTicket, cycle: int) -> None:
    if "TICKET_ROLLBACK" in ticket.stage_seq:
      return
    port = self._port_or_fail(ticket)
    if not port.outbox_ready("CTRL"):
      return
    self._send_ctrl(
      ticket,
      "TICKET_ROLLBACK",
      {"txn_id": ticket.txn_id, "ctx_slot": ticket.ctx_slot, "ctx_generation": ticket.ctx_generation},
      cycle,
    )

  def _send_cancel_if_owed(self, ticket: AdmissionTicket, cycle: int) -> None:
    if "CANCEL_TASK" in ticket.stage_seq:
      return
    port = self._port_or_fail(ticket)
    if not port.outbox_ready("CTRL"):
      return
    # CANCEL_TASK carries the task key and context, no txn id (C017); no ACK is
    # required or trusted -- success arrives only via TASK_COMPLETION.
    self._send_ctrl(
      ticket,
      "CANCEL_TASK",
      {"ctx_slot": ticket.ctx_slot, "ctx_generation": ticket.ctx_generation},
      cycle,
    )

  def _send_ctrl(
    self, ticket: AdmissionTicket, message_type: str, fields: dict[str, int], cycle: int
  ) -> bytes:
    msg = BoundaryMessage(
      message_type=message_type,
      channel="CTRL",
      direction="G2T",
      fields={**ticket.key.to_wire_fields(), **fields},
    )
    blob = self._port_or_fail(ticket).send(msg)
    ticket.stage_seq[message_type] = wire_message_seq(blob)
    self._trace(ticket, f"admission_{message_type.lower()}", cycle)
    return blob

  def _trace(self, ticket: AdmissionTicket, event: str, cycle: int) -> None:
    tracer = self.group.tracer
    if tracer is not None:
      tracer.instant(
        f"Tile{ticket.tile_id}",
        "Cosim:Admission",
        event,
        cycle,
        {
          "txn_id": ticket.txn_id,
          "task_id": ticket.key.task_id,
          "ctx_slot": ticket.ctx_slot,
          "ctx_generation": ticket.ctx_generation,
          "state": ticket.state.value,
        },
      )
