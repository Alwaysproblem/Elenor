"""Software-managed shared accumulator memory (shared AM V1).

The manager owns the only mutable AM state: physical slots, per-context
lease quota, fixed engine affinity, slot generations and terminal records.
It never chooses software logical slots and never exposes AM as an L1
address: a caller must first obtain a lease, then commit typed Matmul steps
through that lease.  A failed request has no persistent side effect.

Physical layout is ``group_count x slots_per_group`` dynamic slots.  A
lease counts against its owner's quota from the successful grant until
``slot_free`` -- not until software awaits an event or reaps a record.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
  """One Tile UCE context incarnation identity."""

  context_name: str
  launch_generation: int
  ctx_id: int
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


@dataclass(frozen=True)
class AmTerminalRecord:
  lease_id: int
  owner: AmOwner
  chain_id: int
  generation: int
  seq: int
  kind: str  # "chain_done" | "fault"
  success: bool
  detail: str
  cycle: int


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
    self.slot_capacity_bytes = slot_capacity_bytes
    self.group_count = group_count
    self.slots_per_group = slots_per_group
    self.slots_per_context = slots_per_context
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
    self.peak_live = 0
    self.busy_count = 0
    self.acquire_count = 0
    self.commit_count = 0
    self.final_count = 0
    self.fault_count = 0
    self.slot_free_count = 0
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

  def try_acquire(
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
    """Atomically grant one lease or return ``None`` with no side effect."""
    if logical_slot not in (0, 1):
      raise ValueError("AM logical slot must be 0 or 1")
    if byte_count < 1 or byte_count > self.slot_capacity_bytes:
      raise ValueError("AM byte request exceeds physical slot capacity")
    if self.live_for_owner(owner) >= self.slots_per_context or any(
      lease.owner == owner and lease.logical_slot == logical_slot
      for lease in self._leases.values()
    ):
      self.busy_count += 1
      self._trace(
        cycle,
        "am_busy",
        {
          "owner": owner.context_name,
          "ctx_id": owner.ctx_id,
          "buffer": buffer_name,
          "reason": "quota",
          "live": self.live_for_owner(owner),
        },
      )
      return None
    # Deterministic first-free physical slot; software logical_slot never
    # selects a physical group or slot.
    slot = next((item for item in self._slots if item.lease_id is None), None)
    if slot is None:
      self.busy_count += 1
      self._trace(
        cycle,
        "am_busy",
        {
          "owner": owner.context_name,
          "ctx_id": owner.ctx_id,
          "buffer": buffer_name,
          "reason": "capacity",
        },
      )
      return None
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
      bound_engine=slot.group,
      dims=tuple(dims),
      byte_count=byte_count,
      output_base=output_base,
      state=AmLeaseState.RESERVED,
    )
    self._next_lease_id += 1
    slot.lease_id = lease.lease_id
    self._leases[lease.lease_id] = lease
    self._engine_handles[lease.bound_engine] = self._engine_handles.get(lease.bound_engine, 0) + 1
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
        "buffer": buffer_name,
        "slot": f"{lease.group}.{lease.slot}",
        "generation": lease.generation,
        "bound_engine": lease.bound_engine,
        "bytes": lease.byte_count,
        "logical_slot": logical_slot,
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
    self._trace(
      cycle,
      "am_step",
      {
        "lease_id": lease_id,
        "chain_id": lease.chain_id,
        "owner": lease.owner.context_name,
        "ctx_id": lease.owner.ctx_id,
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
      lease.state = AmLeaseState.FINAL_READY
      self.final_count += 1
      self._record_terminal(lease, seq, "chain_done", True, "", cycle)
    else:
      lease.state = AmLeaseState.RUNNING
    self._trace(
      cycle,
      "am_commit",
      {
        "lease_id": lease_id,
        "chain_id": lease.chain_id,
        "owner": lease.owner.context_name,
        "ctx_id": lease.owner.ctx_id,
        "seq": seq,
        "final": final,
        "slot": f"{lease.group}.{lease.slot}",
      },
    )
    return lease

  def release_final(self, lease_id: int, cycle: int) -> AmLease:
    """Free the slot after the FINAL output is visible and quiescent."""
    lease = self.lease(lease_id)
    if lease.state is not AmLeaseState.FINAL_READY:
      raise ValueError(f"AM lease {lease_id} is not ready for final release")
    lease.state = AmLeaseState.DRAINING
    self._free_slot(lease, cycle, "chain_done")
    return lease

  def abort_lease(self, lease_id: int, cycle: int, reason: str) -> AmLease:
    lease = self.lease(lease_id)
    lease.state = AmLeaseState.FAULT_QUIESCING
    self._record_terminal(lease, lease.last_seq, "fault", False, reason, cycle)
    self.fault_count += 1
    self._free_slot(lease, cycle, "fault")
    return lease

  def abort_owner(self, owner: AmOwner, cycle: int, reason: str = "context abort") -> int:
    count = 0
    for lease in tuple(self._leases.values()):
      if lease.owner == owner:
        self.abort_lease(lease.lease_id, cycle, reason)
        count += 1
    return count

  def reset(self, cycle: int = 0) -> None:
    for lease in tuple(self._leases.values()):
      self.abort_lease(lease.lease_id, cycle, "reset")
    self._terminals.clear()
    for slot in self._slots:
      slot.generation += 1
    self._engine_handles.clear()

  def _record_terminal(
    self, lease: AmLease, seq: int, kind: str, success: bool, detail: str, cycle: int
  ) -> None:
    self._terminals.append(
      AmTerminalRecord(
        lease.lease_id,
        lease.owner,
        lease.chain_id,
        lease.generation,
        seq,
        kind,
        success,
        detail,
        cycle,
      )
    )
    if len(self._terminals) > self.terminal_capacity:
      del self._terminals[: len(self._terminals) - self.terminal_capacity]

  def _free_slot(self, lease: AmLease, cycle: int, reason: str) -> None:
    slot = self._slots[lease.group * self.slots_per_group + lease.slot]
    if slot.lease_id != lease.lease_id:
      raise ValueError("AM physical slot ownership mismatch")
    slot.lease_id = None
    lease.state = AmLeaseState.FREE
    self._leases.pop(lease.lease_id, None)
    self.slot_free_count += 1
    self._trace(
      cycle,
      "am_release",
      {
        "lease_id": lease.lease_id,
        "chain_id": lease.chain_id,
        "owner": lease.owner.context_name,
        "ctx_id": lease.owner.ctx_id,
        "slot": f"{lease.group}.{lease.slot}",
        "generation": lease.generation,
        "reason": reason,
      },
    )

  # -- snapshots -----------------------------------------------------

  def snapshot(self) -> dict:
    return {
      "slot_capacity_bytes": self.slot_capacity_bytes,
      "group_count": self.group_count,
      "slots_per_group": self.slots_per_group,
      "slots_per_context": self.slots_per_context,
      "total_slots": self.total_slots,
      "free_slots": sum(slot.lease_id is None for slot in self._slots),
      "live_leases": self.live_lease_count,
      "peak_live": self.peak_live,
      "busy_count": self.busy_count,
      "acquire_count": self.acquire_count,
      "commit_count": self.commit_count,
      "final_count": self.final_count,
      "fault_count": self.fault_count,
      "slot_free_count": self.slot_free_count,
      "engine_handle_counts": dict(sorted(self._engine_handles.items())),
      "slots": [
        {
          "slot": slot.name,
          "generation": slot.generation,
          "lease_id": slot.lease_id,
        }
        for slot in self._slots
      ],
      "leases": [
        {
          "lease_id": lease.lease_id,
          "chain_id": lease.chain_id,
          "owner": lease.owner.context_name,
          "ctx_id": lease.owner.ctx_id,
          "owner_epoch": lease.owner.epoch,
          "buffer": lease.buffer_name,
          "slot": f"{lease.group}.{lease.slot}",
          "generation": lease.generation,
          "bound_engine": lease.bound_engine,
          "state": lease.state.value,
          "next_seq": lease.next_seq,
          "last_seq": lease.last_seq,
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
