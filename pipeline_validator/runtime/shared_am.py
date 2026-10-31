"""Software-managed shared accumulator memory (shared AM V1).

Each Compute Tile owns one manager containing its only mutable AM state:
physical slots, per-context lease quota, fixed engine affinity, generations and terminal records.
It never chooses software logical slots and never exposes AM as an L1
address: a caller must first obtain a lease, then commit typed Matmul steps
through that lease.

Physical layout is ``group_count x slots_per_group`` dynamic slots.  A
lease counts against its owner's quota from the successful grant until
``slot_free`` -- not until software awaits an event or reaps a record.

Engine affinity: physical groups are pure slot addressing and are
independent of the BOA engine pool (``engine_count`` engines, ids
``0..engine_count-1``).  At grant time the allocator binds the lease to the
engine with the fewest currently live handles, ties broken by the lowest
engine id.  The live count increments with the same grant and decrements
exactly once at the safe free, before parked requests are woken, so wake
grants observe the freed engine's updated count.  A finalized commit does
not decrement: the lease -- and its engine handle -- stay live through the
FINAL output and any output backpressure until ``slot_free``.  Affinity is
never re-evaluated and ``bound_engine`` is never mutated after grant; a busy
engine's queue therefore keeps its leases.  One engine may serve any number
of live handles (only the per-context quota and physical slots bound it);
the familiar "at most 2 per engine from an empty start" is a distribution
outcome of least-live balancing, not a limit.

Allocation is one request + blocking wait + event wake-up.  ``ALLOC_AM`` is
issued exactly once; a request that cannot be granted is parked here, not
retried by the issuer, and the issuer is woken by ``slot_free``.  To keep
blocking deadlock-free, every owner has a *bundle* (its Tile Program's
static peak of simultaneously live chains, at most ``slots_per_context``):

* a fresh owner (no live lease) is granted only when its whole bundle fits
  in unreserved free slots; the rest of the bundle stays reserved for it;
* a continuation request (owner already holds leases) is served from that
  reservation, or waits only for one of the owner's own finalized leases to
  drain -- never for another owner's slot;
* the bundle is returned to the pool once the owner has no live lease and
  no parked request.

So no owner ever waits on another owner while holding AM, and because
``bundle <= slots_per_context <= total slots`` every parked fresh request
eventually fits.  Fresh requests are granted in FIFO order.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from ..config import MAX_BOA_ENGINES
from ..trace import Tracer


class AmLeaseState(Enum):
  RESERVED = "reserved"
  RUNNING = "running"
  COMMITTING = "committing"
  FINAL_READY = "final_ready"
  DRAINING = "draining"
  FAULT_QUIESCING = "fault_quiescing"
  FREE = "free"


@dataclass(frozen=True)
class AmOwner:
  """One physical Tile UCE context incarnation identity."""

  context_name: str
  launch_generation: int
  ctx_id: int
  tile_id: int
  epoch: int = 0


@dataclass
class AmLease:
  lease_id: int
  owner: AmOwner
  chain_id: int
  buffer_name: str
  generation: int
  group: int
  slot: int
  logical_slot: int
  bound_engine: int
  dims: tuple[int, ...]
  byte_count: int
  output_base: str
  state: AmLeaseState
  next_seq: int = 0
  last_seq: int = -1
  partial: bytes | None = None
  final_issued: bool = False
  # Accepted engine steps whose AM read/commit/output drain has not retired.
  # slot_free waits for zero: a faulted lease stays FAULT_QUIESCING until its
  # last access drains, so no old response reaches a new generation.
  accesses: int = 0
  fault_reason: str = ""
  free_cycle: int | None = None


@dataclass(frozen=True)
class AmTerminalRecord:
  lease_id: int
  owner: AmOwner
  chain_id: int
  generation: int
  seq: int
  kind: str  # "chain_done" | "fault"; published only after slot_free
  success: bool
  detail: str
  cycle: int


@dataclass
class AmRequest:
  """One parked ALLOC_AM request; woken exactly once on grant."""

  request_id: int
  owner: AmOwner
  buffer_name: str
  logical_slot: int
  dims: tuple[int, ...]
  byte_count: int
  output_base: str
  bundle_slots: int
  wake: Callable[[AmLease, int], None]
  enqueue_cycle: int = 0


@dataclass
class _PhysicalSlot:
  group: int
  index: int
  capacity_bytes: int
  generation: int = 0
  lease_id: int | None = None

  @property
  def name(self) -> str:
    return f"am{self.group}.{self.index}"


class SharedAmManager:
  """Finite dynamic AM slots with per-context leases and safe reuse."""

  def __init__(
    self,
    *,
    slot_capacity_bytes: int,
    group_count: int = 4,
    slots_per_group: int = 2,
    slots_per_context: int = 2,
    engine_count: int = 1,
    tile_id: int = 0,
    tracer: Tracer | None = None,
    terminal_capacity: int = 256,
  ):
    if slot_capacity_bytes < 1:
      raise ValueError("AM slot capacity must be positive")
    if group_count < 1 or slots_per_group < 1:
      raise ValueError("AM groups and slots per group must be positive")
    if not 1 <= slots_per_context <= group_count * slots_per_group:
      raise ValueError("AM per-context quota must fit the physical slot count")
    if not isinstance(engine_count, int) or isinstance(engine_count, bool):
      raise ValueError("AM engine count must be an exact integer")
    if not 1 <= engine_count <= MAX_BOA_ENGINES:
      raise ValueError(f"AM engine count must be in 1..{MAX_BOA_ENGINES}")
    self.slot_capacity_bytes = slot_capacity_bytes
    self.group_count = group_count
    self.slots_per_group = slots_per_group
    self.slots_per_context = slots_per_context
    self.engine_count = engine_count
    self.tile_id = tile_id
    self.tracer = tracer
    self.terminal_capacity = terminal_capacity
    self._slots = [
      _PhysicalSlot(group, index, slot_capacity_bytes)
      for group in range(group_count)
      for index in range(slots_per_group)
    ]
    self._leases: dict[int, AmLease] = {}
    self._next_lease_id = 1
    self._next_chain_id = 0
    self._terminals: list[AmTerminalRecord] = []
    self._pending_requests: deque[AmRequest] = deque()
    self._bundles: dict[AmOwner, int] = {}
    self._granting_pending = False
    self._next_request_id = 1
    self.peak_pending = 0
    self.request_count = 0
    self.wake_count = 0
    self.cancelled_request_count = 0
    self.peak_live = 0
    self.parked_count = 0
    self.acquire_count = 0
    self.commit_count = 0
    self.final_count = 0
    self.fault_count = 0
    self.slot_free_count = 0
    self.terminal_overflow_count = 0
    self._engine_handles: dict[int, int] = {}

  # -- introspection -------------------------------------------------

  @property
  def total_slots(self) -> int:
    return len(self._slots)

  @property
  def live_lease_count(self) -> int:
    return len(self._leases)

  def live_for_owner(self, owner: AmOwner) -> int:
    return sum(lease.owner == owner for lease in self._leases.values())

  def lease(self, lease_id: int) -> AmLease:
    lease = self._leases.get(lease_id)
    if lease is None:
      raise ValueError(f"unknown or released AM lease {lease_id}")
    return lease

  def _trace(self, cycle: int, name: str, args: dict) -> None:
    if self.tracer is None:
      return
    payload = {"tile_id": self.tile_id, **args}
    self.tracer.instant(f"Tile{self.tile_id}", "SharedAM", name, cycle, payload)

  # -- acquisition ---------------------------------------------------

  def request(
    self,
    owner: AmOwner,
    *,
    buffer_name: str,
    logical_slot: int,
    dims: tuple[int, ...],
    byte_count: int,
    output_base: str,
    bundle_slots: int,
    cycle: int,
    wake: Callable[[AmLease, int], None],
  ) -> AmLease | None:
    """Issue ALLOC_AM exactly once.

    Returns the lease when granted now.  Otherwise the request is parked
    with no slot/quota/generation side effect and ``wake(lease, cycle)``
    runs exactly once when ``slot_free`` makes it grantable; the issuer
    must not re-issue it.  Permanent contract violations raise: waiting
    could never satisfy them and would block the context forever.
    """
    if logical_slot not in (0, 1):
      raise ValueError("AM logical slot must be 0 or 1")
    if byte_count < 0 or byte_count > self.slot_capacity_bytes:
      raise ValueError("AM byte request exceeds physical slot capacity")
    if not 1 <= bundle_slots <= self.slots_per_context:
      raise ValueError("AM bundle exceeds the per-context lease quota")
    if any(item.owner == owner for item in self._pending_requests):
      raise ValueError("AM owner already has a parked request")
    if any(
      lease.owner == owner and lease.logical_slot == logical_slot and not lease.final_issued
      for lease in self._leases.values()
    ):
      raise ValueError("AM logical slot is already live for an unfinished chain")
    bundle = self._bundles.get(owner)
    if bundle is not None and bundle != bundle_slots:
      raise ValueError("AM owner changed its bundle while holding leases")
    unfinished = sum(
      lease.owner == owner and not lease.final_issued for lease in self._leases.values()
    )
    if unfinished >= bundle_slots:
      # Every unfinished chain needs more UCE instructions to reach FINAL;
      # waiting here for one of them would be a self-deadlock.
      raise ValueError("AM allocation exceeds the context's declared bundle")

    request = AmRequest(
      request_id=self._next_request_id,
      owner=owner,
      buffer_name=buffer_name,
      logical_slot=logical_slot,
      dims=tuple(dims),
      byte_count=byte_count,
      output_base=output_base,
      bundle_slots=bundle_slots,
      wake=wake,
      enqueue_cycle=cycle,
    )
    self._next_request_id += 1
    self.request_count += 1
    fresh_waiting = any(item.owner not in self._bundles for item in self._pending_requests)
    if (bundle is not None or not fresh_waiting) and self._grantable(request):
      return self._grant(request, cycle)
    self._pending_requests.append(request)
    self.parked_count += 1
    self.peak_pending = max(self.peak_pending, len(self._pending_requests))
    self._trace(
      cycle,
      "am_request_parked",
      {
        "request_id": request.request_id,
        "owner": owner.context_name,
        "ctx_id": owner.ctx_id,
        "tile_id": owner.tile_id,
        "buffer": buffer_name,
        "bundle_slots": bundle_slots,
        "kind": "continuation" if bundle is not None else "fresh",
        "free_slots": self._free_slot_count(),
        "reserved_slots": self._reserved_slot_count(),
      },
    )
    return None

  def cancel_requests(self, owner: AmOwner, cycle: int, reason: str = "request cancelled") -> int:
    """Drop one owner's parked request without waking it."""
    cancelled = [item for item in self._pending_requests if item.owner == owner]
    if not cancelled:
      return 0
    self._pending_requests = deque(item for item in self._pending_requests if item.owner != owner)
    self.cancelled_request_count += len(cancelled)
    for item in cancelled:
      self._trace(
        cycle,
        "am_request_cancelled",
        {
          "request_id": item.request_id,
          "owner": owner.context_name,
          "ctx_id": owner.ctx_id,
          "tile_id": owner.tile_id,
          "reason": reason,
        },
      )
    self._drop_idle_bundle(owner)
    self._grant_pending(cycle)
    return len(cancelled)

  def _free_slot_count(self) -> int:
    return sum(slot.lease_id is None for slot in self._slots)

  def _reserved_for(self, owner: AmOwner) -> int:
    bundle = self._bundles.get(owner)
    if bundle is None:
      return 0
    return max(0, bundle - self.live_for_owner(owner))

  def _reserved_slot_count(self) -> int:
    return sum(self._reserved_for(owner) for owner in self._bundles)

  def _grantable(self, request: AmRequest) -> bool:
    """Fresh: whole bundle fits in unreserved free slots.  Continuation:
    the owner's own reservation still has a free slot."""
    # FINAL acceptance ends source liveness, not the physical lease.  A
    # finalized slot/quota conflict is transient until autonomous slot_free.
    if self.live_for_owner(request.owner) >= self.slots_per_context or any(
      lease.owner == request.owner and lease.logical_slot == request.logical_slot
      for lease in self._leases.values()
    ):
      return False
    if request.owner in self._bundles:
      return self._reserved_for(request.owner) >= 1
    return self._free_slot_count() - self._reserved_slot_count() >= request.bundle_slots

  def _grant(self, request: AmRequest, cycle: int) -> AmLease:
    self._bundles.setdefault(request.owner, request.bundle_slots)
    lease = self._allocate_lease(
      request.owner,
      buffer_name=request.buffer_name,
      logical_slot=request.logical_slot,
      dims=request.dims,
      byte_count=request.byte_count,
      output_base=request.output_base,
      cycle=cycle,
    )
    if lease is None:
      raise ValueError("AM reservation accounting lost a free slot")
    return lease

  def _drop_idle_bundle(self, owner: AmOwner) -> None:
    if owner in self._bundles and self.live_for_owner(owner) == 0 and not any(
      item.owner == owner for item in self._pending_requests
    ):
      del self._bundles[owner]

  def _grant_pending(self, cycle: int) -> None:
    if self._granting_pending:
      return
    self._granting_pending = True
    try:
      self._grant_pending_locked(cycle)
    finally:
      self._granting_pending = False

  def _grant_pending_locked(self, cycle: int) -> None:
    """Wake grantable parked requests: continuations whenever their own
    reservation frees, fresh requests strictly FIFO (no overtaking, so a
    large bundle cannot starve)."""
    progressed = True
    while progressed:
      progressed = False
      fresh_blocked = False
      for request in tuple(self._pending_requests):
        fresh = request.owner not in self._bundles
        if fresh and fresh_blocked:
          continue
        if not self._grantable(request):
          fresh_blocked = fresh_blocked or fresh
          continue
        self._pending_requests.remove(request)
        lease = self._grant(request, cycle)
        self.wake_count += 1
        self._trace(
          cycle,
          "am_request_wake",
          {
            "request_id": request.request_id,
            "lease_id": lease.lease_id,
            "owner": request.owner.context_name,
            "ctx_id": request.owner.ctx_id,
            "tile_id": request.owner.tile_id,
            "buffer": request.buffer_name,
            "bound_engine": lease.bound_engine,
            "waited_cycles": cycle - request.enqueue_cycle,
          },
        )
        request.wake(lease, cycle)
        progressed = True
        break

  def _allocate_lease(
    self,
    owner: AmOwner,
    *,
    buffer_name: str,
    logical_slot: int,
    dims: tuple[int, ...],
    byte_count: int,
    output_base: str,
    cycle: int,
  ) -> AmLease | None:
    slot = next((item for item in self._slots if item.lease_id is None), None)
    if slot is None:
      return None
    # Grant-time affinity: fewest currently live handles wins, ties go to
    # the lowest engine id.  ``min`` over the ascending id domain keeps the
    # tie rule; the count is read and incremented inside this same grant so
    # slot, quota and engine binding are one atomic result.
    handles = self._engine_handles
    bound_engine = min(range(self.engine_count), key=lambda engine: handles.get(engine, 0))
    slot.generation += 1
    self._next_chain_id += 1
    lease = AmLease(
      lease_id=self._next_lease_id,
      owner=owner,
      chain_id=self._next_chain_id,
      buffer_name=buffer_name,
      generation=slot.generation,
      group=slot.group,
      slot=slot.index,
      logical_slot=logical_slot,
      bound_engine=bound_engine,
      dims=tuple(dims),
      byte_count=byte_count,
      output_base=output_base,
      state=AmLeaseState.RESERVED,
    )
    self._next_lease_id += 1
    slot.lease_id = lease.lease_id
    self._leases[lease.lease_id] = lease
    handles[bound_engine] = handles.get(bound_engine, 0) + 1
    self.acquire_count += 1
    self.peak_live = max(self.peak_live, self.live_lease_count)
    self._trace(
      cycle,
      "am_allocate",
      {
        "lease_id": lease.lease_id,
        "chain_id": lease.chain_id,
        "owner": owner.context_name,
        "ctx_id": owner.ctx_id,
        "tile_id": owner.tile_id,
        "buffer": buffer_name,
        "slot": f"{lease.group}.{lease.slot}",
        "generation": lease.generation,
        "bound_engine": lease.bound_engine,
        "bytes": lease.byte_count,
        "logical_slot": logical_slot,
        "quota_slots": self.slots_per_context,
        "owner_live_handles": self.live_for_owner(owner),
        "engine_live_handles": handles[bound_engine],
        "live_leases": self.live_lease_count,
        "engine_count": self.engine_count,
      },
    )
    return lease

  # -- typed step lifecycle ------------------------------------------

  def begin_step(
    self, lease_id: int, *, seq: int, init: bool, final: bool, cycle: int
  ) -> AmLease:
    lease = self.lease(lease_id)
    if lease.state not in (AmLeaseState.RESERVED, AmLeaseState.RUNNING):
      raise ValueError(f"AM lease {lease_id} is not accepting Matmul steps")
    if seq != lease.next_seq:
      raise ValueError(
        f"AM lease {lease_id} expects seq {lease.next_seq}, got {seq}"
      )
    if init != (seq == 0):
      raise ValueError(f"AM lease {lease_id} INIT/ADD ordering is invalid at seq {seq}")
    if final and seq == 0 and not init:
      raise ValueError("AM FINAL without a prior INIT is invalid")
    lease.state = AmLeaseState.RUNNING
    # An accepted FINAL drains and frees the slot with no further UCE
    # instruction, so the owner may already wait on it for a new chain.
    lease.final_issued = final
    # The accepted step is an in-flight AM access until its commit (non-FINAL)
    # or its output drain + slot_free (FINAL), or until its owner retires it
    # through ``abort_lease(..., retire_access=True)``.
    lease.accesses += 1
    self._trace(
      cycle,
      "am_step",
      {
        "lease_id": lease_id,
        "chain_id": lease.chain_id,
        "owner": lease.owner.context_name,
        "ctx_id": lease.owner.ctx_id,
        "bound_engine": lease.bound_engine,
        "seq": seq,
        "init": init,
        "final": final,
      },
    )
    return lease

  def commit_step(
    self, lease_id: int, *, seq: int, data: bytes, final: bool, cycle: int
  ) -> AmLease:
    lease = self.lease(lease_id)
    if lease.state is not AmLeaseState.RUNNING:
      raise ValueError(f"AM lease {lease_id} is not RUNNING")
    if seq != lease.next_seq:
      raise ValueError(f"AM lease {lease_id} commit order is invalid")
    expected_bytes = 1
    for dim in lease.dims:
      expected_bytes *= dim
    expected_bytes *= 4
    if not isinstance(data, bytes) or len(data) != expected_bytes:
      raise ValueError("AM commit payload does not match its slot shape")
    lease.partial = data
    lease.last_seq = seq
    lease.next_seq = seq + 1
    self.commit_count += 1
    if final:
      # step_committed only: the FINAL access stays open through the output
      # write and drain; chain_done is published by release_final after
      # slot_free, never here.
      lease.state = AmLeaseState.FINAL_READY
      self.final_count += 1
    else:
      lease.state = AmLeaseState.RUNNING
      self._retire_access(lease)
    self._trace(
      cycle,
      "am_commit",
      {
        "lease_id": lease_id,
        "chain_id": lease.chain_id,
        "owner": lease.owner.context_name,
        "ctx_id": lease.owner.ctx_id,
        "bound_engine": lease.bound_engine,
        "seq": seq,
        "final": final,
        "slot": f"{lease.group}.{lease.slot}",
      },
    )
    return lease

  @staticmethod
  def _retire_access(lease: AmLease) -> None:
    if lease.accesses < 1:
      raise ValueError(f"AM lease {lease.lease_id} has no in-flight access to retire")
    lease.accesses -= 1

  def release_final(self, lease_id: int, cycle: int) -> AmLease:
    """slot_free after the FINAL output is visible and its drain ended.

    The caller (the BOA result service) has already written and drained the
    output.  The FINAL step is the only remaining access; no later step can
    exist.  The success terminal is published inside the free linearization
    point, after region/quota return and before parked requests are woken.
    """
    lease = self.lease(lease_id)
    if lease.state is not AmLeaseState.FINAL_READY:
      raise ValueError(f"AM lease {lease_id} is not ready for final release")
    if lease.accesses != 1 or lease.next_seq != lease.last_seq + 1:
      raise ValueError(f"AM lease {lease_id} has pending steps or accesses at slot_free")
    lease.state = AmLeaseState.DRAINING
    self._retire_access(lease)
    self._free_slot(lease, cycle, "chain_done", ("chain_done", True, ""))
    return lease

  def abort_lease(
    self, lease_id: int, cycle: int, reason: str, *, retire_access: bool = False
  ) -> AmLease:
    """Fault one lease: quiesce in-flight accesses, then free, then publish.

    ``retire_access`` is set by a caller that owned one of the in-flight
    accesses (a dropped result or failed engine job).  While other accesses
    remain the lease keeps its slot/quota in FAULT_QUIESCING; the failure
    terminal is recorded only after the slot is actually reclaimed.
    Repeated aborts keep the first reason and count one fault.
    """
    lease = self.lease(lease_id)
    if lease.state is not AmLeaseState.FAULT_QUIESCING:
      lease.state = AmLeaseState.FAULT_QUIESCING
      lease.fault_reason = reason
      self.fault_count += 1
    if retire_access:
      self._retire_access(lease)
    if lease.accesses == 0:
      self._free_slot(lease, cycle, "fault", ("fault", False, lease.fault_reason))
    return lease

  def abort_owner(self, owner: AmOwner, cycle: int, reason: str = "context abort") -> int:
    count = self.cancel_requests(owner, cycle, reason)
    for lease in tuple(self._leases.values()):
      if lease.owner == owner:
        self.abort_lease(lease.lease_id, cycle, reason)
        count += 1
    return count

  def reset(self, cycle: int = 0) -> None:
    """Controlled domain reset: the reset itself quiesces every access.

    Failure terminals are published like any fault; the bounded diagnostic
    ring is not cleared, so reset never erases the only failure record.
    """
    self._pending_requests.clear()
    for lease in tuple(self._leases.values()):
      lease.accesses = 0
      self.abort_lease(lease.lease_id, cycle, "reset")
    self._bundles.clear()
    for slot in self._slots:
      slot.generation += 1
    self._engine_handles.clear()

  def _record_terminal(
    self, lease: AmLease, kind: str, success: bool, detail: str, cycle: int
  ) -> None:
    record = AmTerminalRecord(
      lease.lease_id,
      lease.owner,
      lease.chain_id,
      lease.generation,
      lease.last_seq,
      kind,
      success,
      detail,
      cycle,
    )
    self._terminals.append(record)
    if len(self._terminals) > self.terminal_capacity:
      dropped = len(self._terminals) - self.terminal_capacity
      del self._terminals[:dropped]
      self.terminal_overflow_count += dropped
    self._trace(
      cycle,
      "am_chain_terminal",
      {
        "lease_id": lease.lease_id,
        "chain_id": lease.chain_id,
        "owner": lease.owner.context_name,
        "ctx_id": lease.owner.ctx_id,
        "tile_id": lease.owner.tile_id,
        "owner_epoch": lease.owner.epoch,
        "slot": f"{lease.group}.{lease.slot}",
        "generation": lease.generation,
        "bound_engine": lease.bound_engine,
        "seq": lease.last_seq,
        "kind": kind,
        "success": success,
        "detail": detail,
        "slot_free_cycle": lease.free_cycle,
        "terminal_overflow_count": self.terminal_overflow_count,
      },
    )

  def _free_slot(
    self, lease: AmLease, cycle: int, reason: str, terminal: tuple[str, bool, str]
  ) -> None:
    if lease.accesses:
      raise ValueError(f"AM lease {lease.lease_id} freed with in-flight accesses")
    slot = self._slots[lease.group * self.slots_per_group + lease.slot]
    if slot.lease_id != lease.lease_id:
      raise ValueError("AM physical slot ownership mismatch")
    slot.lease_id = None
    lease.state = AmLeaseState.FREE
    lease.free_cycle = cycle
    self._leases.pop(lease.lease_id, None)
    # The engine handle dies with the lease, exactly once, strictly before
    # any parked request is woken: the wake grant must weigh the freed
    # engine's updated live count.  FINAL commits never reach this path, so
    # handles stay live through output backpressure.
    remaining = self._engine_handles.get(lease.bound_engine, 0) - 1
    if remaining > 0:
      self._engine_handles[lease.bound_engine] = remaining
    else:
      self._engine_handles.pop(lease.bound_engine, None)
    self._drop_idle_bundle(lease.owner)
    self.slot_free_count += 1
    self._trace(
      cycle,
      "am_release",
      {
        "lease_id": lease.lease_id,
        "chain_id": lease.chain_id,
        "owner": lease.owner.context_name,
        "ctx_id": lease.owner.ctx_id,
        "tile_id": lease.owner.tile_id,
        "slot": f"{lease.group}.{lease.slot}",
        "generation": lease.generation,
        "bound_engine": lease.bound_engine,
        "engine_live_handles": self._engine_handles.get(lease.bound_engine, 0),
        "reason": reason,
      },
    )
    # Terminal linearization: region and quota are already returned; the
    # old chain's terminal is published before any parked owner is woken.
    self._record_terminal(lease, *terminal, cycle)
    self._grant_pending(cycle)

  # -- snapshots -----------------------------------------------------

  def snapshot(self) -> dict:
    return {
      "tile_id": self.tile_id,
      "slot_capacity_bytes": self.slot_capacity_bytes,
      "group_count": self.group_count,
      "slots_per_group": self.slots_per_group,
      "slots_per_context": self.slots_per_context,
      "engine_count": self.engine_count,
      "total_slots": self.total_slots,
      "free_slots": sum(slot.lease_id is None for slot in self._slots),
      "live_leases": self.live_lease_count,
      "peak_live": self.peak_live,
      "parked_count": self.parked_count,
      "pending_requests": len(self._pending_requests),
      "peak_pending": self.peak_pending,
      "request_count": self.request_count,
      "wake_count": self.wake_count,
      "cancelled_request_count": self.cancelled_request_count,
      "reserved_slots": self._reserved_slot_count(),
      "bundles": [
        {
          "owner": owner.context_name,
          "ctx_id": owner.ctx_id,
          "tile_id": owner.tile_id,
          "bundle_slots": bundle,
          "live": self.live_for_owner(owner),
        }
        for owner, bundle in self._bundles.items()
      ],
      "acquire_count": self.acquire_count,
      "commit_count": self.commit_count,
      "final_count": self.final_count,
      "fault_count": self.fault_count,
      "slot_free_count": self.slot_free_count,
      "terminal_capacity": self.terminal_capacity,
      "terminal_overflow_count": self.terminal_overflow_count,
      # Every engine id appears, zero counts included; the values always sum
      # to ``live_leases`` because each lease contributes exactly one handle
      # from grant until its single safe free.
      "engine_handle_counts": {
        engine: self._engine_handles.get(engine, 0) for engine in range(self.engine_count)
      },
      "slots": [
        {
          "slot": slot.name,
          "generation": slot.generation,
          "lease_id": slot.lease_id,
        }
        for slot in self._slots
      ],
      "requests": [
        {
          "request_id": item.request_id,
          "owner": item.owner.context_name,
          "ctx_id": item.owner.ctx_id,
          "tile_id": item.owner.tile_id,
          "buffer": item.buffer_name,
          "logical_slot": item.logical_slot,
          "bundle_slots": item.bundle_slots,
          "enqueue_cycle": item.enqueue_cycle,
        }
        for item in self._pending_requests
      ],
      "leases": [
        {
          "lease_id": lease.lease_id,
          "chain_id": lease.chain_id,
          "owner": lease.owner.context_name,
          "ctx_id": lease.owner.ctx_id,
          "tile_id": lease.owner.tile_id,
          "owner_epoch": lease.owner.epoch,
          "buffer": lease.buffer_name,
          "slot": f"{lease.group}.{lease.slot}",
          "generation": lease.generation,
          "bound_engine": lease.bound_engine,
          "state": lease.state.value,
          "next_seq": lease.next_seq,
          "last_seq": lease.last_seq,
          "accesses": lease.accesses,
          "logical_slot": lease.logical_slot,
          "bytes": lease.byte_count,
          "output": lease.output_base,
        }
        for lease in sorted(self._leases.values(), key=lambda item: item.lease_id)
      ],
      "terminals": [
        {
          "lease_id": record.lease_id,
          "chain_id": record.chain_id,
          "owner": record.owner.context_name,
          "ctx_id": record.owner.ctx_id,
          "tile_id": record.owner.tile_id,
          "owner_epoch": record.owner.epoch,
          "generation": record.generation,
          "seq": record.seq,
          "kind": record.kind,
          "success": record.success,
          "detail": record.detail,
          "cycle": record.cycle,
        }
        for record in self._terminals
      ],
    }
