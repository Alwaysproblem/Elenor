"""Arena-scoped SRAM reservation and buffer-view lifetimes.

An :class:`ArenaPool` owns one physical L1 or L2 SPM pool.  Admission reserves
compiled, striped per-bank extents atomically.  Binding or invalidating a
buffer view never changes the global free map: only retiring its parent arena
returns those extents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from ..execution_ir import TaskIdentity
from ..profiles import ArenaLayout, MemoryProfile
from .allocator import (
  AdmissionFailure,
  AdmissionFailureKind,
  AdmissionWaitReason,
  AllocationHandle,
  BankedFreeExtentAllocator,
  BankSegment,
  ContextBufferOwner,
  MemoryInvariantError,
  MemoryOwner,
  TaskBufferOwner,
)

if TYPE_CHECKING:
  from ..trace import MemoryTrace

__all__ = ["ArenaHandle", "ArenaOwner", "ArenaPlan", "ArenaPool", "RootInvocation"]


WAIT_CAPACITY = "WAIT_CAPACITY"
WAIT_FRAGMENTATION = "WAIT_FRAGMENTATION"
PERMANENT_CAPACITY = "PERMANENT_CAPACITY"


@dataclass(frozen=True)
class RootInvocation:
  """Lifetime identity of one root Context invocation."""

  context_name: str
  launch_generation: int

  def __post_init__(self) -> None:
    if not self.context_name:
      raise ValueError("root invocation context name must not be empty")
    if type(self.launch_generation) is not int or self.launch_generation < 0:
      raise ValueError("root invocation generation must be a non-negative integer")


ArenaOwner: TypeAlias = RootInvocation | TaskIdentity  # noqa: UP040  # Python 3.11 runtime


@dataclass(frozen=True)
class ArenaPlan:
  """Pure, uncommitted exact per-bank arena placement."""

  pool_version: int
  extent_pool_version: int
  pool_token: object = field(repr=False, compare=False)
  owner: ArenaOwner
  layout: ArenaLayout
  reserve: tuple[BankSegment, ...]
  profile_mode: int
  profile_generation: int


@dataclass(frozen=True)
class ArenaHandle:
  """Committed owner reservation.

  ``reserve`` includes all compiled padding.  It is deliberately distinct
  from the valid-byte segments of buffer-view ``AllocationHandle`` objects.
  """

  arena_id: str
  pool_token: object = field(repr=False, compare=False)
  memory_space: str
  owner: ArenaOwner
  reserve: tuple[BankSegment, ...]
  layout: ArenaLayout
  profile: MemoryProfile
  profile_generation: int
  allocation_generation: int
  allocate_cycle: int

  @property
  def reserved_bytes(self) -> int:
    return sum(segment.size_bytes for segment in self.reserve)


class _ViewState:
  LIVE = "live"
  INVALIDATE_PENDING = "invalidate_pending"
  INVALIDATED = "invalidated"


class _ArenaState:
  LIVE = "live"
  RETIRE_PENDING = "retire_pending"
  RETIRED = "retired"


@dataclass
class _ViewRecord:
  handle: AllocationHandle
  state: str = _ViewState.LIVE
  invalidate_cycle: int = -1
  pins: set[str] = field(default_factory=set)
  inflight: set[str] = field(default_factory=set)


@dataclass
class _ArenaRecord:
  handle: ArenaHandle
  state: str = _ArenaState.LIVE
  retire_cycle: int = -1
  task_metadata: tuple[str, int] | None = None
  views: dict[str, str] = field(default_factory=dict)


class ArenaPool:
  """One profiled, banked SRAM pool with arena-granularity admission.

  The target bank stride and bank count are fixed by the initial profile.
  Reconfiguration may move the SPM/cache boundary but cannot alter that
  geometry.  A pool instance represents one L1 Tile pool or one L2 Group
  pool, so L1-only reconfiguration has no path to mutate an L2 instance.
  """

  def __init__(
    self,
    profile: MemoryProfile,
    *,
    profile_generation: int = 0,
    pool_id: int = 0,
    tile_id: int | None = None,
    trace: MemoryTrace | None = None,
  ) -> None:
    if profile.level not in ("l1", "l2"):
      raise ValueError(f"unsupported arena memory level {profile.level!r}")
    if type(pool_id) is not int or not 0 <= pool_id < profile.pools:
      raise ValueError("arena pool id is out of range")
    self.memory_space = profile.level
    self.pool_id = pool_id
    self.tile_id = pool_id if profile.level == "l1" and tile_id is None else tile_id
    self.pools = profile.pools
    self.banks = profile.banks
    self.bytes_per_bank = profile.bank_bytes
    self.capacity_bytes = self.banks * self.bytes_per_bank
    self._trace = trace
    # The backing allocator is the sole implementation of extent consume and
    # coalescing.  ArenaPool does not use its allocation-record lifecycle.
    self._extents = BankedFreeExtentAllocator(
      memory_space=profile.level,
      capacity_bytes=self.capacity_bytes,
      banks=self.banks,
      trace=None,
      trace_tile_id=self.tile_id,
    )
    self._pool_token = object()
    self._pool_version = 0
    self._active_profile: MemoryProfile | None = None
    self._profile_generation: int | None = None
    self._allocation_generation = 0
    self._view_counter = 0
    self._arenas: dict[str, _ArenaRecord] = {}
    self._views: dict[str, _ViewRecord] = {}
    self._owner_arenas: dict[ArenaOwner, str] = {}
    self._arena_reserved_bytes = 0
    self._peak_arena_reserved_bytes = 0
    self._peak_live_view_bytes = 0
    self.reconfigure(profile, profile_generation, cycle=0)

  @property
  def pool_version(self) -> int:
    """Monotonic version changed by arena admission and retirement."""
    return self._pool_version

  @property
  def profile(self) -> MemoryProfile:
    assert self._active_profile is not None
    return self._active_profile

  @property
  def profile_generation(self) -> int:
    assert self._profile_generation is not None
    return self._profile_generation

  # -- admission --------------------------------------------------------

  def plan_arena(self, owner: ArenaOwner, layout: ArenaLayout) -> ArenaPlan | AdmissionFailure:
    """Plan an exact striped arena placement without mutating pool state."""
    owner_error = self._owner_error(owner)
    if owner_error is not None:
      return AdmissionFailure(AdmissionFailureKind.INVALID_REQUEST, owner_error)
    if owner in self._owner_arenas:
      return AdmissionFailure(
        AdmissionFailureKind.INVALID_REQUEST, "arena owner already has a live reservation"
      )
    layout_error = self._layout_error(layout)
    if layout_error is not None:
      reason, buffer_id = layout_error
      return AdmissionFailure(AdmissionFailureKind.INVALID_REQUEST, reason, buffer_id)
    reserve = self._plan_reserve(layout)
    if isinstance(reserve, AdmissionFailure):
      return reserve
    return ArenaPlan(
      pool_version=self.pool_version,
      extent_pool_version=self._extents.pool_version,
      pool_token=self._pool_token,
      owner=owner,
      layout=layout,
      reserve=reserve,
      profile_mode=self.profile.mode,
      profile_generation=self.profile_generation,
    )

  def rollback(self, plan: ArenaPlan) -> None:
    """Discard an uncommitted pure plan, rejecting an already-stale plan."""
    if plan.pool_token is not self._pool_token:
      raise MemoryInvariantError("arena plan belongs to another pool")
    if plan.pool_version != self.pool_version or plan.extent_pool_version != self._extents.pool_version:
      raise MemoryInvariantError("stale arena plan")

  def commit_arena(self, plan: ArenaPlan, cycle: int) -> ArenaHandle:
    """Atomically consume every planned bank extent and mint one arena."""
    if plan.pool_token is not self._pool_token:
      raise MemoryInvariantError("arena plan belongs to another pool")
    if plan.pool_version != self.pool_version or plan.extent_pool_version != self._extents.pool_version:
      raise MemoryInvariantError("stale arena plan")
    if plan.profile_mode != self.profile.mode or plan.profile_generation != self.profile_generation:
      raise MemoryInvariantError("stale arena profile generation")
    owner_error = self._owner_error(plan.owner)
    if owner_error is not None:
      raise MemoryInvariantError(owner_error)
    if plan.owner in self._owner_arenas:
      raise MemoryInvariantError("arena owner already has a live reservation")
    layout_error = self._layout_error(plan.layout)
    if layout_error is not None:
      raise MemoryInvariantError(layout_error[0])

    # Plans are public immutable records, so recompute deterministic first-fit
    # before the allocator atomically validates and consumes exact ranges.
    expected = self._plan_reserve(plan.layout)
    if isinstance(expected, AdmissionFailure):
      raise MemoryInvariantError("arena plan placement is no longer available")
    if plan.reserve != expected:
      raise MemoryInvariantError("arena plan placement does not match deterministic first-fit")
    allocation_generation = self._allocation_generation + 1
    scope = f"p{self.pool_id}:"
    arena_id = f"{self.memory_space}:{scope}arena:{allocation_generation}"
    handle = ArenaHandle(
      arena_id=arena_id,
      memory_space=self.memory_space,
      pool_token=self._pool_token,
      owner=plan.owner,
      reserve=plan.reserve,
      layout=plan.layout,
      profile=self.profile,
      profile_generation=self.profile_generation,
      allocation_generation=allocation_generation,
      allocate_cycle=cycle,
    )
    if plan.reserve:
      self._extents.commit_exact(plan.extent_pool_version, plan.reserve, cycle)
    self._allocation_generation = allocation_generation
    self._arenas[arena_id] = _ArenaRecord(handle=handle)
    self._owner_arenas[plan.owner] = arena_id
    self._arena_reserved_bytes += handle.reserved_bytes
    self._peak_arena_reserved_bytes = max(self._peak_arena_reserved_bytes, self._arena_reserved_bytes)
    self._pool_version += 1
    self._emit_trace(cycle)
    if self._trace is not None:
      self._trace.arena_reserve(self.memory_space, self.tile_id, handle, self.snapshot(), cycle)
    return handle

  # -- view binding and lifetime ---------------------------------------
  def bind_task_metadata(self, arena: ArenaHandle, role_event_id: str, hardware_context_id: int) -> None:
    """Attach committed physical Task metadata before its first view bind."""
    record = self._arena_record(arena)
    if not isinstance(arena.owner, TaskIdentity):
      raise MemoryInvariantError("task metadata requires a TaskIdentity arena")
    if record.state != _ArenaState.LIVE:
      raise MemoryInvariantError("cannot bind metadata to a retiring arena")
    if not isinstance(role_event_id, str) or not role_event_id:
      raise MemoryInvariantError("task role event id must not be empty")
    if type(hardware_context_id) is not int or hardware_context_id < 0:
      raise MemoryInvariantError("invalid hardware context id")
    if record.views:
      raise MemoryInvariantError("task metadata must precede all view bindings")
    if record.task_metadata is not None:
      raise MemoryInvariantError("task metadata is already bound")
    record.task_metadata = (role_event_id, hardware_context_id)

  def bind_view(self, arena: ArenaHandle, buffer_id: str, cycle: int) -> AllocationHandle:
    """Bind one compiled buffer layout to its arena's valid-byte segments."""
    arena_record = self._arena_record(arena)
    if arena_record.state != _ArenaState.LIVE:
      raise MemoryInvariantError("cannot bind a view after arena retirement was requested")
    if buffer_id in arena_record.views:
      raise MemoryInvariantError(f"arena buffer view {buffer_id!r} is already bound")
    layouts = tuple(item for item in arena.layout.buffer_layouts if item.buffer_id == buffer_id)
    if len(layouts) != 1:
      raise MemoryInvariantError(f"unknown arena buffer layout {buffer_id!r}")
    buffer_layout = layouts[0]
    reserve_by_bank = {segment.bank_id: segment for segment in arena.reserve}
    valid_segments: list[BankSegment] = []
    for bank_id, relative_start, size in buffer_layout.segments():
      reserve = reserve_by_bank.get(bank_id)
      if reserve is None or relative_start < 0:
        raise MemoryInvariantError("buffer view lies outside its arena reserve")
      if relative_start + size > reserve.size_bytes:
        raise MemoryInvariantError("buffer view lies outside its arena reserve")
      valid_segments.append(
        BankSegment(bank_id=bank_id, address=reserve.address + relative_start, size_bytes=size)
      )
    if sum(segment.size_bytes for segment in valid_segments) != buffer_layout.logical_bytes:
      raise MemoryInvariantError("buffer view valid-byte segments are incomplete")
    self._assert_no_live_view_overlap(valid_segments)
    owner = self._buffer_owner(arena_record, buffer_id)

    self._view_counter += 1
    allocation_id = f"{arena.arena_id}:view:{self._view_counter}:{buffer_id}"
    if valid_segments:
      base_address = valid_segments[0].address
    elif arena.reserve:
      base_address = arena.reserve[0].address
    else:
      base_address = 0
    handle = AllocationHandle(
      allocation_id=allocation_id,
      memory_space=self.memory_space,
      owner=owner,
      base_address=base_address,
      size_bytes=buffer_layout.logical_bytes,
      alignment=arena.layout.alignment,
      bank_segments=tuple(valid_segments),
      generation=arena.allocation_generation,
      allocate_cycle=cycle,
      arena_id=arena.arena_id,
      profile_generation=arena.profile_generation,
    )
    self._views[allocation_id] = _ViewRecord(handle=handle)
    arena_record.views[buffer_id] = allocation_id
    self._peak_live_view_bytes = max(self._peak_live_view_bytes, self._live_view_bytes())
    self._emit_trace(cycle)
    return handle

  def invalidate_view(self, view: AllocationHandle, owner: MemoryOwner, cycle: int) -> bool:
    """Invalidate a local view; never return parent arena extents.

    ``False`` records a pending invalidation while pins or in-flight users
    remain.  The final ``unpin``/``end_inflight`` completes only the local
    invalidation.  A separate successful :meth:`retire_arena` is the sole path
    that returns reserve to the global free map.
    """
    record = self._view_record(view)
    if record.state == _ViewState.INVALIDATED:
      raise MemoryInvariantError("double release")
    if record.handle.owner != owner:
      raise MemoryInvariantError("wrong-owner release")
    if record.pins or record.inflight:
      record.state = _ViewState.INVALIDATE_PENDING
      record.invalidate_cycle = cycle
      self._emit_trace(cycle)
      return False
    self._finalize_view(record, cycle)
    return True

  def assert_live(self, handle: AllocationHandle, owner: MemoryOwner | None = None) -> None:
    record = self._view_record(handle)
    if record.state == _ViewState.INVALIDATED:
      raise MemoryInvariantError("use-after-release")
    if owner is not None and record.handle.owner != owner:
      raise MemoryInvariantError("wrong-owner release")

  def pin(self, handle: AllocationHandle, consumer_id: str) -> None:
    record = self._view_record(handle)
    if record.state == _ViewState.INVALIDATED:
      raise MemoryInvariantError("use-after-release")
    if record.state == _ViewState.INVALIDATE_PENDING:
      raise MemoryInvariantError("cannot pin a release-pending allocation")
    if consumer_id in record.pins:
      raise MemoryInvariantError("duplicate allocation pin")
    record.pins.add(consumer_id)

  def unpin(self, handle: AllocationHandle, consumer_id: str, cycle: int) -> bool:
    record = self._view_record(handle)
    if record.state == _ViewState.INVALIDATED:
      raise MemoryInvariantError("use-after-release")
    if consumer_id not in record.pins:
      raise MemoryInvariantError("unknown allocation pin")
    record.pins.remove(consumer_id)
    if record.state == _ViewState.INVALIDATE_PENDING and not record.pins and not record.inflight:
      self._finalize_view(record, cycle)
      return True
    self._emit_trace(cycle)
    return False

  def begin_inflight(self, handle: AllocationHandle, transaction_id: str) -> None:
    """Register an accepted transaction that still references this view."""
    record = self._view_record(handle)
    if record.state == _ViewState.INVALIDATED:
      raise MemoryInvariantError("use-after-release")
    if record.state == _ViewState.INVALIDATE_PENDING:
      raise MemoryInvariantError("cannot issue from a release-pending allocation")
    if transaction_id in record.inflight:
      raise MemoryInvariantError("duplicate in-flight allocation reference")
    record.inflight.add(transaction_id)

  def end_inflight(self, handle: AllocationHandle, transaction_id: str, cycle: int) -> bool:
    """Finish one accepted transaction and possibly complete invalidation."""
    record = self._view_record(handle)
    if record.state == _ViewState.INVALIDATED:
      raise MemoryInvariantError("use-after-release")
    if transaction_id not in record.inflight:
      raise MemoryInvariantError("unknown in-flight allocation reference")
    record.inflight.remove(transaction_id)
    if record.state == _ViewState.INVALIDATE_PENDING and not record.pins and not record.inflight:
      self._finalize_view(record, cycle)
      return True
    self._emit_trace(cycle)
    return False

  def resolve_segments(
    self, handle: AllocationHandle, offset_bytes: int, size_bytes: int
  ) -> tuple[BankSegment, ...]:
    """Resolve a logical valid-byte subrange, excluding arena padding."""
    if (
      type(offset_bytes) is not int
      or type(size_bytes) is not int
      or offset_bytes < 0
      or size_bytes < 0
      or offset_bytes + size_bytes > handle.size_bytes
    ):
      raise MemoryInvariantError("memory view out of bounds")
    self.assert_live(handle)
    result: list[BankSegment] = []
    cursor = 0
    end = offset_bytes + size_bytes
    for segment in handle.bank_segments:
      segment_end = cursor + segment.size_bytes
      if segment_end <= offset_bytes or cursor >= end:
        cursor = segment_end
        continue
      clip_start = max(0, offset_bytes - cursor)
      clip_end = min(segment.size_bytes, end - cursor)
      if clip_end > clip_start:
        result.append(
          BankSegment(
            bank_id=segment.bank_id, address=segment.address + clip_start, size_bytes=clip_end - clip_start
          )
        )
      cursor = segment_end
    return tuple(result)

  def is_released(self, handle: AllocationHandle) -> bool:
    record = self._views.get(handle.allocation_id)
    return record is None or record.state == _ViewState.INVALIDATED

  # -- arena retirement and profile reconfiguration --------------------

  def retire_arena(self, arena: ArenaHandle, cycle: int) -> bool:
    """Return a whole safe arena reservation to the free map."""
    record = self._arena_record(arena)
    if record.state == _ArenaState.RETIRED:
      raise MemoryInvariantError("double arena release")
    for allocation_id in record.views.values():
      view_record = self._views[allocation_id]
      if view_record.state != _ViewState.INVALIDATED:
        record.state = _ArenaState.RETIRE_PENDING
        record.retire_cycle = cycle
        self._emit_trace(cycle)
        return False
      if view_record.pins or view_record.inflight:
        raise MemoryInvariantError("invalidated arena view retains references")
    if arena.reserved_bytes > self._arena_reserved_bytes:
      raise MemoryInvariantError("arena reserved-byte accounting underflow")
    for segment in arena.reserve:
      self._local_start(segment)
    if arena.reserve:
      self._extents.release_exact(self._extents.pool_version, arena.reserve, cycle)
    record.state = _ArenaState.RETIRED
    record.retire_cycle = cycle
    self._owner_arenas.pop(arena.owner, None)
    self._arena_reserved_bytes -= arena.reserved_bytes
    self._pool_version += 1
    self._emit_trace(cycle)
    if self._trace is not None:
      self._trace.arena_retire(self.memory_space, self.tile_id, arena, self.snapshot(), cycle)
    return True

  def reconfigure(self, profile: MemoryProfile, generation: int, cycle: int) -> None:
    """Install a quiescent profile's user-SPM intervals.

    The physical bank stride remains ``bank_bytes`` even when the SPM/cache
    boundary moves.  This method never calls allocator ``reset`` and refuses
    to discard live state.
    """
    if profile.level != self.memory_space:
      raise MemoryInvariantError("profile level does not match arena pool")
    if (
      profile.banks != self.banks
      or profile.bank_bytes != self.bytes_per_bank
      or profile.pools != self.pools
    ):
      raise MemoryInvariantError("profile changes fixed arena pool geometry")
    if not 0 <= self.pool_id < self.pools:
      raise MemoryInvariantError("profile omits this arena pool")
    if type(generation) is not int or generation < 0:
      raise MemoryInvariantError("profile generation must be a non-negative integer")
    if self._profile_generation is not None and generation <= self._profile_generation:
      raise MemoryInvariantError("profile generation must increase monotonically")
    if (
      profile.system_reserved_spm_per_bank < 0
      or profile.spm_bytes_per_bank < profile.system_reserved_spm_per_bank
      or profile.spm_bytes_per_bank > profile.bank_bytes
    ):
      raise MemoryInvariantError("invalid profile SPM interval")
    live_arenas = [record for record in self._arenas.values() if record.state != _ArenaState.RETIRED]
    live_views = [record for record in self._views.values() if record.state != _ViewState.INVALIDATED]
    retained_references = [record for record in self._views.values() if record.pins or record.inflight]
    if live_arenas or live_views or retained_references or self._owner_arenas:
      raise MemoryInvariantError("cannot reconfigure an arena pool with live state")
    if self._arena_reserved_bytes != 0:
      raise MemoryInvariantError("cannot reconfigure with reserved arena bytes")

    self._extents.reconfigure_free_intervals(
      profile.system_reserved_spm_per_bank, profile.spm_bytes_per_bank, cycle
    )
    self._active_profile = profile
    self._profile_generation = generation
    # Retired records from an old profile are no longer useful for identity
    # checks; allocation_generation remains monotonic, preventing ABA reuse.
    self._arenas.clear()
    self._views.clear()
    self._owner_arenas.clear()
    self._pool_version += 1
    self._emit_trace(cycle)

  # -- deterministic observability -------------------------------------

  def snapshot(self) -> dict:
    """Return deterministic pool, arena, view and per-bank counters."""
    profile = self.profile
    reserved_by_bank = [0] * self.banks
    padding_by_bank = [0] * self.banks
    live_by_bank = [0] * self.banks
    live_arena_records = [record for record in self._arenas.values() if record.state != _ArenaState.RETIRED]
    for arena_record in live_arena_records:
      reserve_by_bank = {segment.bank_id: segment for segment in arena_record.handle.reserve}
      for bank_id, reserve in reserve_by_bank.items():
        reserved_by_bank[bank_id] += reserve.size_bytes
        static_ranges: list[tuple[int, int]] = []
        for layout in arena_record.handle.layout.buffer_layouts:
          for layout_bank, offset, size in layout.segments():
            if layout_bank == bank_id and size:
              static_ranges.append((offset, offset + size))
        static_bytes = self._union_size(static_ranges)
        padding_by_bank[bank_id] += reserve.size_bytes - static_bytes
      for allocation_id in arena_record.views.values():
        view_record = self._views[allocation_id]
        if view_record.state == _ViewState.INVALIDATED:
          continue
        for segment in view_record.handle.bank_segments:
          live_by_bank[segment.bank_id] += segment.size_bytes

    per_bank: list[dict] = []
    free_total = 0
    largest = 0
    free_extents = self._extents.free_extents_snapshot()
    for bank_id, extents in enumerate(free_extents):
      free_bytes = sum(size for _, size in extents)
      largest_extent = max((size for _, size in extents), default=0)
      free_total += free_bytes
      largest = max(largest, largest_extent)
      per_bank.append(
        {
          "bank_id": bank_id,
          "bank_stride_bytes": self.bytes_per_bank,
          "system_reserved_bytes": profile.system_reserved_spm_per_bank,
          "cache_bytes": profile.cache_bytes_per_bank,
          "arena_reserved_bytes": reserved_by_bank[bank_id],
          "allocated_bytes": reserved_by_bank[bank_id],
          "live_view_bytes": live_by_bank[bank_id],
          "padding_bytes": padding_by_bank[bank_id],
          "free_bytes": free_bytes,
          "largest_free_extent": largest_extent,
        }
      )

    view_records = tuple(self._views.values())
    live_view_records = tuple(record for record in view_records if record.state != _ViewState.INVALIDATED)
    arena_rows = tuple(
      {
        "arena_id": record.handle.arena_id,
        "owner": repr(record.handle.owner),
        "state": record.state,
        "reserved_bytes": record.handle.reserved_bytes,
        "task_metadata": record.task_metadata,
        "live_views": sum(
          self._views[allocation_id].state != _ViewState.INVALIDATED
          for allocation_id in record.views.values()
        ),
      }
      for record in sorted(self._arenas.values(), key=lambda item: item.handle.arena_id)
      if record.state != _ArenaState.RETIRED
    )
    return {
      "memory_space": self.memory_space,
      "pool_id": self.pool_id,
      "tile_id": self.tile_id,
      "profile_mode": profile.mode,
      "profile_generation": self.profile_generation,
      "pool_version": self.pool_version,
      "extent_pool_version": self._extents.pool_version,
      "bank_stride_bytes": self.bytes_per_bank,
      "capacity_bytes": self.capacity_bytes,
      "user_spm_capacity_bytes": profile.user_spm_bytes,
      "system_reserved_bytes": self.banks * profile.system_reserved_spm_per_bank,
      "cache_bytes": profile.cache_bytes,
      "arena_reserved_bytes": self._arena_reserved_bytes,
      "allocated_bytes": self._arena_reserved_bytes,
      "free_bytes": free_total,
      "largest_free_extent": largest,
      "peak_arena_reserved_bytes": self._peak_arena_reserved_bytes,
      "peak_allocated_bytes": self._peak_arena_reserved_bytes,
      "live_view_bytes": self._live_view_bytes(),
      "peak_live_view_bytes": self._peak_live_view_bytes,
      "padding_bytes": sum(padding_by_bank),
      "live_arenas": len(live_arena_records),
      "live_allocations": len(live_arena_records),
      "zero_byte_arenas": sum(record.handle.reserved_bytes == 0 for record in live_arena_records),
      "live_views": len(live_view_records),
      "pending_view_invalidations": sum(
        record.state == _ViewState.INVALIDATE_PENDING for record in view_records
      ),
      "pending_arena_retirements": sum(
        record.state == _ArenaState.RETIRE_PENDING for record in live_arena_records
      ),
      "pending_release": sum(record.state == _ViewState.INVALIDATE_PENDING for record in view_records)
      + sum(record.state == _ArenaState.RETIRE_PENDING for record in live_arena_records),
      "pin_count": sum(len(record.pins) for record in live_view_records),
      "inflight_count": sum(len(record.inflight) for record in live_view_records),
      "per_bank_occupancy": per_bank,
      "arenas": arena_rows,
    }

  # -- validation and extent helpers -----------------------------------

  def _owner_error(self, owner: ArenaOwner) -> str | None:
    if self.memory_space == "l2" and not isinstance(owner, RootInvocation):
      return "L2 arena owner must be a RootInvocation"
    if self.memory_space == "l1" and not isinstance(owner, TaskIdentity):
      return "L1 arena owner must be a TaskIdentity"
    return None

  def _layout_error(self, layout: ArenaLayout) -> tuple[str, str] | None:
    if (
      type(layout.alignment) is not int
      or layout.alignment <= 0
      or layout.alignment & (layout.alignment - 1)
    ):
      return ("invalid arena alignment", "")
    if layout.alignment < self.profile.alignment:
      return ("arena alignment is weaker than the active profile", "")
    if (
      type(layout.stripe_bytes) is not int
      or layout.stripe_bytes <= 0
      or layout.stripe_bytes % layout.alignment != 0
    ):
      return ("invalid arena stripe size", "")
    if type(layout.reserved_bytes) is not int or layout.reserved_bytes < 0:
      return ("invalid arena reserved byte count", "")
    if len(layout.per_bank_bytes) != self.banks:
      return ("arena layout bank count does not match the pool", "")
    if any(type(size) is not int or size < 0 for size in layout.per_bank_bytes):
      return ("invalid arena per-bank byte count", "")
    if any(size % layout.stripe_bytes != 0 for size in layout.per_bank_bytes):
      return ("arena per-bank reserves must contain whole stripes", "")
    if sum(layout.per_bank_bytes) != layout.reserved_bytes:
      return ("arena reserved bytes do not match per-bank extents", "")
    if not isinstance(layout.layout_hash, str) or not layout.layout_hash:
      return ("arena layout hash must not be empty", "")

    seen_buffers: set[str] = set()
    for buffer_layout in layout.buffer_layouts:
      if not buffer_layout.buffer_id or buffer_layout.buffer_id in seen_buffers:
        return ("arena buffer ids must be non-empty and unique", buffer_layout.buffer_id)
      seen_buffers.add(buffer_layout.buffer_id)
      if type(buffer_layout.logical_bytes) is not int or buffer_layout.logical_bytes < 0:
        return ("invalid arena buffer byte count", buffer_layout.buffer_id)
      if type(buffer_layout.slot_id) is not int or buffer_layout.slot_id < 0:
        return ("invalid arena buffer slot id", buffer_layout.buffer_id)
      stripe_round = layout.stripe_bytes * self.banks
      if (
        type(buffer_layout.arena_offset) is not int
        or buffer_layout.arena_offset < 0
        or buffer_layout.arena_offset % stripe_round != 0
      ):
        return ("invalid arena buffer offset", buffer_layout.buffer_id)
      if buffer_layout.stripe_bytes != layout.stripe_bytes:
        return ("arena buffer stripe does not match its owner layout", buffer_layout.buffer_id)
      if buffer_layout.banks != self.banks:
        return ("arena buffer bank count does not match the pool", buffer_layout.buffer_id)
      segment_total = 0
      for bank_id, offset, size in buffer_layout.segments():
        if not 0 <= bank_id < self.banks or offset < 0 or size <= 0:
          return ("invalid arena buffer segment", buffer_layout.buffer_id)
        if offset + size > layout.per_bank_bytes[bank_id]:
          return ("arena buffer segment exceeds its bank reserve", buffer_layout.buffer_id)
        segment_total += size
      if segment_total != buffer_layout.logical_bytes:
        return ("arena buffer layout omits valid bytes", buffer_layout.buffer_id)
    return None

  def _plan_reserve(self, layout: ArenaLayout) -> tuple[BankSegment, ...] | AdmissionFailure:
    current_free = self._extents.free_extents_snapshot()
    reserve: list[BankSegment] = []
    for bank_id, size in enumerate(layout.per_bank_bytes):
      if size == 0:
        continue
      start = self._first_fit(current_free[bank_id], bank_id, size, layout.alignment)
      if start is None:
        pristine = (
          [(self.profile.system_reserved_spm_per_bank, self.profile.user_spm_per_bank)]
          if self.profile.user_spm_per_bank
          else []
        )
        if self._first_fit(pristine, bank_id, size, layout.alignment) is None:
          return AdmissionFailure(
            AdmissionFailureKind.PERMANENT_CAPACITY,
            f"{PERMANENT_CAPACITY}: arena cannot fit empty bank {bank_id}",
          )
        total_free = sum(extent_size for _, extent_size in current_free[bank_id])
        if total_free < size:
          reason = f"{WAIT_CAPACITY}: bank {bank_id} lacks free bytes"
          wait_reason = AdmissionWaitReason.CAPACITY
        else:
          reason = f"{WAIT_FRAGMENTATION}: bank {bank_id} lacks a contiguous aligned extent"
          wait_reason = AdmissionWaitReason.FRAGMENTATION
        return AdmissionFailure(AdmissionFailureKind.TEMPORARY_CAPACITY, reason, wait_reason=wait_reason)
      address = bank_id * self.bytes_per_bank + start
      reserve.append(BankSegment(bank_id, address, size))
    return tuple(reserve)

  def _first_fit(
    self,
    extents: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    bank_id: int,
    size: int,
    alignment: int,
  ) -> int | None:
    bank_base = bank_id * self.bytes_per_bank
    for start, extent_size in sorted(extents):
      absolute_start = bank_base + start
      aligned_absolute = ((absolute_start + alignment - 1) // alignment) * alignment
      aligned_start = aligned_absolute - bank_base
      if aligned_start + size <= start + extent_size:
        return aligned_start
    return None

  def _local_start(self, segment: BankSegment) -> int:
    if not 0 <= segment.bank_id < self.banks or segment.size_bytes <= 0:
      raise MemoryInvariantError("invalid arena reserve segment")
    local_start = segment.address - segment.bank_id * self.bytes_per_bank
    if (
      local_start < self.profile.system_reserved_spm_per_bank
      or local_start + segment.size_bytes > self.profile.spm_bytes_per_bank
    ):
      raise MemoryInvariantError("arena reserve lies outside active user SPM")
    return local_start

  def _arena_record(self, arena: ArenaHandle) -> _ArenaRecord:
    record = self._arenas.get(arena.arena_id)
    if (
      record is None
      or arena.pool_token is not self._pool_token
      or record.handle.allocation_generation != arena.allocation_generation
      or record.handle.profile_generation != arena.profile_generation
      or record.handle != arena
    ):
      raise MemoryInvariantError("stale arena generation")
    return record

  def _view_record(self, handle: AllocationHandle) -> _ViewRecord:
    record = self._views.get(handle.allocation_id)
    if (
      record is None
      or record.handle.generation != handle.generation
      or record.handle.arena_id != handle.arena_id
      or record.handle.profile_generation != handle.profile_generation
      or record.handle != handle
    ):
      raise MemoryInvariantError("stale allocation generation")
    return record

  def _buffer_owner(self, arena: _ArenaRecord, buffer_id: str) -> MemoryOwner:
    owner = arena.handle.owner
    if isinstance(owner, RootInvocation):
      return ContextBufferOwner(owner.context_name, owner.launch_generation, buffer_id)
    if self.tile_id is None:
      raise MemoryInvariantError("L1 arena pool has no physical tile id")
    if arena.task_metadata is None:
      raise MemoryInvariantError("Task arena metadata must be bound before its first view")
    role_event_id, hardware_context_id = arena.task_metadata
    return TaskBufferOwner(
      owner.grid.context_name,
      owner.grid.launch_generation,
      role_event_id,
      owner.task_id,
      self.tile_id,
      hardware_context_id,
      buffer_id,
    )

  def _assert_no_live_view_overlap(self, segments: list[BankSegment]) -> None:
    for record in self._views.values():
      if record.state == _ViewState.INVALIDATED:
        continue
      for candidate in segments:
        candidate_end = candidate.address + candidate.size_bytes
        for live in record.handle.bank_segments:
          if live.bank_id != candidate.bank_id:
            continue
          if candidate.address < live.address + live.size_bytes and live.address < candidate_end:
            raise MemoryInvariantError("arena view overlaps a live buffer view")

  def _finalize_view(self, record: _ViewRecord, cycle: int) -> None:
    if record.pins or record.inflight:
      raise MemoryInvariantError("cannot invalidate a referenced arena view")
    if record.state == _ViewState.INVALIDATED:
      raise MemoryInvariantError("double release")
    record.state = _ViewState.INVALIDATED
    record.invalidate_cycle = cycle
    # Deliberately no free-map or pool-version mutation here.
    self._emit_trace(cycle)
    if self._trace is not None:
      self._trace.buffer_view_invalidate(
        self.memory_space, self.tile_id, record.handle, self.snapshot(), cycle
      )

  def _live_view_bytes(self) -> int:
    return sum(
      record.handle.size_bytes for record in self._views.values() if record.state != _ViewState.INVALIDATED
    )

  @staticmethod
  def _union_size(ranges: list[tuple[int, int]]) -> int:
    if not ranges:
      return 0
    ordered = sorted(ranges)
    start, end = ordered[0]
    total = 0
    for next_start, next_end in ordered[1:]:
      if next_start <= end:
        end = max(end, next_end)
      else:
        total += end - start
        start, end = next_start, next_end
    return total + end - start

  def _emit_trace(self, cycle: int) -> None:
    if self._trace is None:
      return
    snapshot = self.snapshot()
    self._trace.capacity(self.memory_space, self.tile_id, snapshot, cycle)
    self._trace.banks(self.memory_space, self.tile_id, snapshot["per_bank_occupancy"], cycle)
