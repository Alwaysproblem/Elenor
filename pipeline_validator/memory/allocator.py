"""Deterministic free-extent allocator with owner/generation/pin invariants.

Provides the single allocation contract for HBM (external bindings), L2
(group SRAM) and per-tile L1.  Every allocation is an immutable
``AllocationHandle`` carrying owner, generation, real physical address
and bank segments.  Capacity, permission, owner, generation and credit
errors never produce a successful event.

The allocator is a deterministic first-fit free-extent planner: plan on
a cloned free map (zero side effects), commit atomically, release by
owner/generation, pin/unpin for shared consumers.  Allocation ids are
reproducible (no UUID/hash); reset bumps a generation so prior handles
become stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class MemoryInvariantError(Exception):
  """Raised when an allocation invariant is violated.

  Diagnostics use fixed fragments so tests can assert on them:
  ``wrong-owner release``, ``double release``, ``stale allocation
  generation``, ``use-after-release``, ``duplicate allocation pin``,
  ``unknown allocation pin``, ``stale allocation plan``,
  ``memory view out of bounds``.
  """


# ---------------------------------------------------------------------------
# Owners — allocation lifetime identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalOwner:
  """HBM external binding owner (user-supplied global input)."""

  binding_name: str


@dataclass(frozen=True)
class ContextBufferOwner:
  """L2 context-buffer owner.

  Device slot does not enter lifetime identity: the same context
  re-submitted on a different slot is a new launch generation, not a
  different buffer.
  """

  context_name: str
  context_launch_generation: int
  buffer_id: str


@dataclass(frozen=True)
class TaskBufferOwner:
  """Per-tile L1/task buffer owner.

  Physical tile, hardware UCE context and logical task are kept
  independent so the same logical task on a different physical tile is a
  distinct owner.
  """

  context_name: str
  context_launch_generation: int
  role_event_id: str
  logical_task_id: int
  physical_tile_id: int
  hardware_context_id: int
  buffer_id: str


MemoryOwner = ExternalOwner | ContextBufferOwner | TaskBufferOwner


# ---------------------------------------------------------------------------
# Segments, requests, handles, plans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BankSegment:
  """One contiguous physical byte range within a bank.

  Tuple order is the allocation's logical byte order; ``address`` is the
  absolute physical address of this segment.
  """

  bank_id: int
  address: int
  size_bytes: int


@dataclass(frozen=True)
class AllocationRequest:
  """One request in an allocation bundle."""

  memory_space: str  # "hbm" | "l2" | "l1"
  buffer_id: str
  owner: MemoryOwner
  size_bytes: int
  alignment: int
  role: str = ""


@dataclass(frozen=True)
class AllocationHandle:
  """Immutable allocation result.  Never mutated in place."""

  allocation_id: str
  memory_space: str
  owner: MemoryOwner
  base_address: int
  size_bytes: int
  alignment: int
  bank_segments: tuple[BankSegment, ...]
  generation: int
  allocate_cycle: int

  @property
  def end_address(self) -> int:
    return self.base_address + self.size_bytes


@dataclass(frozen=True)
class AdmissionFailure:
  """Result of a failed ``plan_bundle``."""

  reason: str
  buffer_id: str = ""


@dataclass(frozen=True)
class AllocationPlan:
  """A staged, uncommitted allocation plan."""

  pool_version: int
  requests: tuple[AllocationRequest, ...]
  placements: dict[str, tuple[BankSegment, ...]] = field(default_factory=dict)


class AllocationState:
  """Live allocation lifecycle state."""

  LIVE = "live"
  RELEASE_PENDING = "release_pending"
  RELEASED = "released"


@dataclass
class _AllocationRecord:
  """Mutable release metadata; the frozen handle is never written back."""

  handle: AllocationHandle
  state: str = AllocationState.LIVE
  release_cycle: int = -1
  pins: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Banked free-extent allocator
# ---------------------------------------------------------------------------


class BankedFreeExtentAllocator:
  """Deterministic first-fit free-extent allocator over banked SRAM.

  Public API (frozen):
    - ``plan_bundle(requests)`` — zero-side-effect planning on a cloned
      free map; returns ``AllocationPlan`` or ``AdmissionFailure``.
    - ``commit(plan, cycle)`` — atomic all-or-nothing commit.
    - ``rollback(plan)`` — discard an uncommitted plan.
    - ``assert_live(handle, owner=None)`` — invariant check.
    - ``pin / unpin`` — shared-consumer reference counting.
    - ``request_release(handle, owner, cycle)`` — release or pending.
    - ``resolve_segments(handle, offset, size)`` — physical byte ranges.
    - ``reset()`` / ``snapshot()``.
  """

  def __init__(self, memory_space: str, capacity_bytes: int, banks: int,
               bytes_per_bank: int | None = None):
    if capacity_bytes <= 0:
      raise ValueError("invalid allocation capacity")
    if banks < 1:
      raise ValueError("banks must be >= 1")
    if capacity_bytes % banks != 0:
      raise ValueError(
        f"{memory_space} capacity {capacity_bytes} not divisible by"
        f" banks {banks}")
    self.memory_space = memory_space
    self.capacity_bytes = capacity_bytes
    self.banks = banks
    self.bytes_per_bank = capacity_bytes // banks
    # bank-local free extents: list of (local_start, local_size)
    self._free: list[list[tuple[int, int]]] = [
      [(0, self.bytes_per_bank)] for _ in range(banks)
    ]
    self._live: dict[str, _AllocationRecord] = {}
    self._pool_version: int = 0
    self._generation: int = 0
    self._counter: int = 0
    self._peak_allocated: int = 0
    self._allocated_bytes: int = 0

  # -- planning ---------------------------------------------------------

  def plan_bundle(
    self, requests: list[AllocationRequest],
  ) -> AllocationPlan | AdmissionFailure:
    """Plan a bundle on a cloned free map — zero side effects.

    Deterministic first-fit: request order, bank id ascending, extent
    start ascending.  The first segment's physical address must satisfy
    the request alignment.  On any failure the partial segments are
    discarded and an ``AdmissionFailure`` is returned.
    """
    # validate requests first
    for req in requests:
      if req.size_bytes <= 0:
        return AdmissionFailure("invalid allocation size", req.buffer_id)
      if req.alignment <= 0 or (req.alignment & (req.alignment - 1)) != 0:
        return AdmissionFailure("invalid allocation alignment", req.buffer_id)
    cloned_free = [list(extents) for extents in self._free]
    placements: dict[str, tuple[BankSegment, ...]] = {}
    for req in requests:
      segments = self._plan_one(cloned_free, req)
      if segments is None:
        return AdmissionFailure("allocation capacity exceeded", req.buffer_id)
      placements[req.buffer_id] = segments
    return AllocationPlan(
      pool_version=self._pool_version,
      requests=tuple(requests),
      placements=placements,
    )

  def _plan_one(
    self, free: list[list[tuple[int, int]]], req: AllocationRequest,
  ) -> tuple[BankSegment, ...] | None:
    """First-fit a single request across banks.  Returns segments or None."""
    remaining = req.size_bytes
    segments: list[BankSegment] = []
    for bank_id in range(self.banks):
      if remaining <= 0:
        break
      extents = free[bank_id]
      new_extents: list[tuple[int, int]] = []
      placed_here = 0
      for start, size in extents:
        if remaining <= 0:
          new_extents.append((start, size))
          continue
        # alignment applies to the very first segment of the allocation
        need_align = req.alignment if not segments else 1
        aligned_start = ((start + need_align - 1) // need_align) * need_align
        if aligned_start >= start + size:
          new_extents.append((start, size))
          continue
        take = min(remaining, (start + size) - aligned_start)
        addr = bank_id * self.bytes_per_bank + aligned_start
        segments.append(BankSegment(bank_id, addr, take))
        remaining -= take
        placed_here += take
        leftover_start = aligned_start + take
        leftover_size = (start + size) - leftover_start
        # preserve leading gap
        if aligned_start > start:
          new_extents.append((start, aligned_start - start))
        if leftover_size > 0:
          new_extents.append((leftover_start, leftover_size))
      free[bank_id] = new_extents
      if placed_here == 0 and remaining == req.size_bytes and bank_id == self.banks - 1:
        # no placement at all on the last bank
        pass
    if remaining > 0:
      return None
    return tuple(segments)

  # -- commit / rollback ------------------------------------------------

  def commit(
    self, plan: AllocationPlan, cycle: int,
  ) -> tuple[AllocationHandle, ...]:
    if plan.pool_version != self._pool_version:
      raise MemoryInvariantError("stale allocation plan")
    # apply placements to the live free map atomically
    new_free = [list(extents) for extents in self._free]
    handles: list[AllocationHandle] = []
    for req in plan.requests:
      segments = plan.placements[req.buffer_id]
      for seg in segments:
        local_start = seg.address - seg.bank_id * self.bytes_per_bank
        self._consume(new_free[seg.bank_id], local_start, seg.size_bytes)
    self._free = new_free
    results: list[AllocationHandle] = []
    for req in plan.requests:
      segments = plan.placements[req.buffer_id]
      self._counter += 1
      alloc_id = f"{self.memory_space}:{self._generation}:{self._counter}"
      handle = AllocationHandle(
        allocation_id=alloc_id,
        memory_space=self.memory_space,
        owner=req.owner,
        base_address=segments[0].address,
        size_bytes=req.size_bytes,
        alignment=req.alignment,
        bank_segments=segments,
        generation=self._generation,
        allocate_cycle=cycle,
      )
      self._live[alloc_id] = _AllocationRecord(handle=handle)
      results.append(handle)
      self._allocated_bytes += req.size_bytes
      if self._allocated_bytes > self._peak_allocated:
        self._peak_allocated = self._allocated_bytes
      handles.append(handle)
    self._pool_version += 1
    return tuple(results)

  @staticmethod
  def _consume(
    extents: list[tuple[int, int]], start: int, size: int,
  ) -> None:
    """Remove [start, start+size) from one bank's extent list."""
    new: list[tuple[int, int]] = []
    for estart, esize in extents:
      eend = estart + esize
      astart = start
      aend = start + size
      if aend <= estart or astart >= eend:
        new.append((estart, esize))
        continue
      if astart > estart:
        new.append((estart, astart - estart))
      if aend < eend:
        new.append((aend, eend - aend))
    extents.clear()
    extents.extend(new)

  def rollback(self, plan: AllocationPlan) -> None:
    if plan.pool_version != self._pool_version:
      raise MemoryInvariantError("stale allocation plan")
    # plans are pure clones; rollback is a no-op on live state.  We only
    # guard against rolling back an already-committed plan (stale).
    # Nothing to undo since plan_bundle had no side effects.

  # -- live checks ------------------------------------------------------

  def assert_live(self, handle: AllocationHandle,
                  owner: MemoryOwner | None = None) -> None:
    rec = self._live.get(handle.allocation_id)
    if rec is None or rec.handle.generation != handle.generation:
      raise MemoryInvariantError("stale allocation generation")
    if rec.handle != handle:
      raise MemoryInvariantError("stale allocation generation")
    if rec.state == AllocationState.RELEASED:
      raise MemoryInvariantError("use-after-release")
    if owner is not None and rec.handle.owner != owner:
      raise MemoryInvariantError("wrong-owner release")

  # -- pin / unpin ------------------------------------------------------

  def pin(self, handle: AllocationHandle, consumer_id: str) -> None:
    rec = self._live.get(handle.allocation_id)
    if rec is None or rec.handle.generation != handle.generation:
      raise MemoryInvariantError("stale allocation generation")
    if rec.state == AllocationState.RELEASED:
      raise MemoryInvariantError("use-after-release")
    if consumer_id in rec.pins:
      raise MemoryInvariantError("duplicate allocation pin")
    rec.pins.add(consumer_id)

  def unpin(self, handle: AllocationHandle, consumer_id: str,
            cycle: int) -> bool:
    rec = self._live.get(handle.allocation_id)
    if rec is None or rec.handle.generation != handle.generation:
      raise MemoryInvariantError("stale allocation generation")
    if consumer_id not in rec.pins:
      raise MemoryInvariantError("unknown allocation pin")
    rec.pins.discard(consumer_id)
    if rec.state == AllocationState.RELEASE_PENDING and not rec.pins:
      self._do_release(rec, cycle)
      return True
    return False

  # -- release ----------------------------------------------------------

  def request_release(self, handle: AllocationHandle, owner: MemoryOwner,
                      cycle: int) -> bool:
    rec = self._live.get(handle.allocation_id)
    if rec is None or rec.handle.generation != handle.generation:
      raise MemoryInvariantError("stale allocation generation")
    if rec.state == AllocationState.RELEASED:
      raise MemoryInvariantError("double release")
    if rec.handle.owner != owner:
      raise MemoryInvariantError("wrong-owner release")
    if rec.pins:
      rec.state = AllocationState.RELEASE_PENDING
      rec.release_cycle = cycle
      return False
    self._do_release(rec, cycle)
    return True

  def _do_release(self, rec: _AllocationRecord, cycle: int) -> None:
    for seg in rec.handle.bank_segments:
      local_start = seg.address - seg.bank_id * self.bytes_per_bank
      self._free[seg.bank_id].append((local_start, seg.size_bytes))
      self._free[seg.bank_id].sort()
    # merge adjacent extents in each touched bank
    for seg in rec.handle.bank_segments:
      self._merge_bank(seg.bank_id)
    self._allocated_bytes -= rec.handle.size_bytes
    rec.state = AllocationState.RELEASED
    rec.release_cycle = cycle
    self._pool_version += 1

  def _merge_bank(self, bank_id: int) -> None:
    extents = self._free[bank_id]
    if not extents:
      return
    extents.sort()
    merged: list[tuple[int, int]] = [extents[0]]
    for start, size in extents[1:]:
      last_start, last_size = merged[-1]
      if start == last_start + last_size:
        merged[-1] = (last_start, last_size + size)
      else:
        merged.append((start, size))
    self._free[bank_id] = merged

  # -- segment resolution ----------------------------------------------

  def resolve_segments(
    self, handle: AllocationHandle, offset_bytes: int, size_bytes: int,
  ) -> tuple[BankSegment, ...]:
    if offset_bytes < 0:
      raise MemoryInvariantError("memory view out of bounds")
    if offset_bytes + size_bytes > handle.size_bytes:
      raise MemoryInvariantError("memory view out of bounds")
    rec = self._live.get(handle.allocation_id)
    if rec is None or rec.handle.generation != handle.generation:
      raise MemoryInvariantError("stale allocation generation")
    if rec.state == AllocationState.RELEASED:
      raise MemoryInvariantError("use-after-release")
    # clip the allocation's segments to [offset, offset+size)
    result: list[BankSegment] = []
    cursor = 0
    for seg in handle.bank_segments:
      seg_end = cursor + seg.size_bytes
      if seg_end <= offset_bytes or cursor >= offset_bytes + size_bytes:
        cursor = seg_end
        continue
      clip_start = max(0, offset_bytes - cursor)
      clip_end = min(seg.size_bytes, offset_bytes + size_bytes - cursor)
      take = clip_end - clip_start
      if take > 0:
        result.append(BankSegment(
          bank_id=seg.bank_id,
          address=seg.address + clip_start,
          size_bytes=take,
        ))
      cursor = seg_end
    return tuple(result)

  # -- reset / snapshot -------------------------------------------------

  def is_released(self, handle: AllocationHandle) -> bool:
    """True if the allocation has been freed (RELEASED)."""
    rec = self._live.get(handle.allocation_id)
    return rec is None or rec.state == AllocationState.RELEASED

  def reset(self) -> None:
    self._generation += 1
    self._counter = 0
    self._free = [[(0, self.bytes_per_bank)] for _ in range(self.banks)]
    self._live.clear()
    self._pool_version = 0
    self._peak_allocated = 0
    self._allocated_bytes = 0

  def snapshot(self) -> dict:
    live = sum(1 for r in self._live.values()
               if r.state != AllocationState.RELEASED)
    pending = sum(1 for r in self._live.values()
                  if r.state == AllocationState.RELEASE_PENDING)
    per_bank: list[dict] = []
    free_total = 0
    largest = 0
    for bank_id in range(self.banks):
      bank_free = sum(s for _, s in self._free[bank_id])
      bank_largest = max((s for _, s in self._free[bank_id]), default=0)
      free_total += bank_free
      largest = max(largest, bank_largest)
      per_bank.append({
        "bank_id": bank_id,
        "free_bytes": bank_free,
        "largest_free_extent": bank_largest,
      })
    return {
      "memory_space": self.memory_space,
      "capacity_bytes": self.capacity_bytes,
      "allocated_bytes": self._allocated_bytes,
      "free_bytes": free_total,
      "largest_free_extent": largest,
      "peak_allocated_bytes": self._peak_allocated,
      "live_allocations": live,
      "pending_release": pending,
      "per_bank_occupancy": per_bank,
      "generation": self._generation,
    }
