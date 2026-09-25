"""Arena-scoped SRAM reservation and buffer-view lifetimes.

An :class:`ArenaPool` owns one physical L1 or L2 SPM pool. Admission reserves
compiled, striped per-bank extents atomically. L2 capacity belongs to unique
backings; aliases and claims share that physical extent. L2 view invalidation
forfeits access, and the backing is freed after producer, claim, pin, and
transaction references close. L1 extents remain arena-scoped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, TypeAlias, cast

from ..execution_ir import TaskIdentity
from ..profiles import ArenaLayout, BufferLayout, MemoryProfile
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


class _BackingState:
  LIVE = "live"
  RELEASED = "released"


class _ClaimState:
  DECLARED = "DECLARED"
  BOUND = "BOUND"
  RELEASED = "RELEASED"
  CANCELLED = "CANCELLED"


L2ClaimId: TypeAlias = tuple[str, str]  # noqa: UP040  # Python 3.11 runtime


@dataclass
class _L2ClaimRecord:
  claim_id: L2ClaimId
  backing_id: str
  run_generation: int
  state: str = _ClaimState.DECLARED
  bound_allocation_id: str | None = None
  owner: ContextBufferOwner | None = None

  def snapshot(self) -> dict:
    return {
      "backing_id": self.backing_id,
      "claim_id": self.claim_id,
      "run_generation": self.run_generation,
      "state": self.state,
      "bound_allocation_id": self.bound_allocation_id,
      "owner": repr(self.owner) if self.owner is not None else None,
    }


@dataclass
class _L2BackingRecord:
  """One physical L2 padded backing and its run-scoped logical claims."""

  backing_id: str
  arena_id: str
  owner: RootInvocation
  buffer_id: str
  units: tuple[BankSegment, ...]
  valid_segments: tuple[BankSegment, ...]
  logical_bytes: int
  alignment: int
  padded_bytes: int
  profile_generation: int
  allocation_generation: int
  run_generation: int
  state: str = _BackingState.LIVE
  producer_live: bool = True
  origin_retired: bool = False
  published: bool = False
  producer_allocation_id: str | None = None
  view_allocation_ids: set[str] = field(default_factory=set)
  claims: dict[L2ClaimId, _L2ClaimRecord] = field(default_factory=dict)
  pins: set[tuple[str, str]] = field(default_factory=set)
  inflight: set[tuple[str, str]] = field(default_factory=set)
  commit_cycle: int = 0
  release_cycle: int | None = None

  def snapshot(self) -> dict:
    claims = tuple(
      claim.snapshot() for _claim_id, claim in sorted(self.claims.items())
    )
    return {
      "backing_id": self.backing_id,
      "arena_id": self.arena_id,
      "origin_arena_id": self.arena_id,
      "buffer_id": self.buffer_id,
      "logical_bytes": self.logical_bytes,
      "padded_bytes": self.padded_bytes,
      "state": self.state,
      "origin_retired": self.origin_retired,
      "producer_live": self.producer_live,
      "published": self.published,
      "profile_generation": self.profile_generation,
      "allocation_generation": self.allocation_generation,
      "run_generation": self.run_generation,
      "context_name": self.owner.context_name,
      "launch_generation": self.owner.launch_generation,
      "producer_allocation_id": self.producer_allocation_id,
      "registered_view_allocation_ids": tuple(sorted(self.view_allocation_ids)),
      "registered_view_count": len(self.view_allocation_ids),
      "pin_count": len(self.pins),
      "inflight_count": len(self.inflight),
      "pending_shared_claims": sum(
        claim.state == _ClaimState.DECLARED for claim in self.claims.values()
      ),
      "active_shared_references": sum(
        claim.state == _ClaimState.BOUND for claim in self.claims.values()
      ),
      "claims": claims,
      "commit_cycle": self.commit_cycle,
      "release_cycle": self.release_cycle,
    }


@dataclass
class _ViewRecord:
  handle: AllocationHandle
  state: str = _ViewState.LIVE
  invalidate_cycle: int = -1
  pins: set[str] = field(default_factory=set)
  inflight: set[str] = field(default_factory=set)
  claim_id: L2ClaimId | None = None


@dataclass
class _ArenaRecord:
  handle: ArenaHandle
  run_generation: int = 0
  state: str = _ArenaState.LIVE
  retire_cycle: int = -1
  task_metadata: tuple[str, int] | None = None
  views: dict[str, str] = field(default_factory=dict)
  # Logical producer slot -> independent pool-owned physical backing ID.
  backing_ids: dict[str, str] = field(default_factory=dict)
  slack_units: tuple[BankSegment, ...] = ()


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
    self._claim_records: dict[tuple[int, L2ClaimId], _L2ClaimRecord] = {}
    self._backings: dict[str, _L2BackingRecord] = {}
    self._claims_by_run: dict[tuple[int, L2ClaimId], str] = {}
    self._backing_counter = 0
    # Set by the owning TileGroup at begin_launch; stamped onto physical
    # final-free events for cross-invocation correlation.
    self.run_generation = 0
    # Physical final-free events drained by the TileGroup occupancy mirror.
    self._backing_release_events: list[dict] = []
    self._owner_arenas: dict[ArenaOwner, str] = {}
    self._arena_reserved_bytes = 0
    self._peak_arena_reserved_bytes = 0
    self._peak_live_view_bytes = 0
    self.reconfigure(profile, profile_generation, cycle=0)

  @property
  def pool_version(self) -> int:
    """Monotonic version changed only by physical capacity mutations."""
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

  def commit_arena(
    self,
    plan: ArenaPlan,
    cycle: int,
    *,
    claims_by_slot: Mapping[str, tuple[L2ClaimId, ...]] | None = None,
  ) -> ArenaHandle:
    """Atomically commit a root arena and its materialized shared claims."""
    if plan.pool_token is not self._pool_token:
      raise MemoryInvariantError("arena plan belongs to another pool")
    if plan.pool_version != self.pool_version or plan.extent_pool_version != self._extents.pool_version:
      raise MemoryInvariantError("stale arena plan")
    if plan.profile_mode != self.profile.mode or plan.profile_generation != self.profile_generation:
      raise MemoryInvariantError("stale arena profile generation")
    if self.memory_space == "l2" and any(
      backing.state == _BackingState.LIVE
      and backing.run_generation != self.run_generation
      for backing in self._backings.values()
    ):
      raise MemoryInvariantError("previous run still owns a live L2 backing")
    owner_error = self._owner_error(plan.owner)
    if owner_error is not None:
      raise MemoryInvariantError(owner_error)
    if plan.owner in self._owner_arenas:
      raise MemoryInvariantError("arena owner already has a live reservation")
    layout_error = self._layout_error(plan.layout)
    if layout_error is not None:
      raise MemoryInvariantError(layout_error[0])

    claims: dict[str, tuple[L2ClaimId, ...]] = {}
    if claims_by_slot is not None:
      if not isinstance(claims_by_slot, Mapping):
        raise MemoryInvariantError("claims_by_slot must be a mapping")
      if claims_by_slot and self.memory_space != "l2":
        raise MemoryInvariantError("shared claims require an L2 arena")
      known_layouts = {item.buffer_id: item for item in plan.layout.buffer_layouts}
      seen_claims: set[L2ClaimId] = set()
      for slot, claim_ids in claims_by_slot.items():
        if not isinstance(slot, str) or slot not in known_layouts:
          raise MemoryInvariantError(f"shared claims name unknown arena buffer {slot!r}")
        if not isinstance(claim_ids, tuple):
          raise MemoryInvariantError("shared claim lists must be tuples")
        if claim_ids and known_layouts[slot].logical_bytes <= 0:
          raise MemoryInvariantError("shared claims require a non-empty physical backing")
        for claim_id in claim_ids:
          if (
            not isinstance(claim_id, tuple)
            or len(claim_id) != 2
            or any(not isinstance(part, str) or not part for part in claim_id)
          ):
            raise MemoryInvariantError("shared claim IDs must be non-empty string pairs")
          if claim_id in seen_claims:
            raise MemoryInvariantError("duplicate shared claim ID")
          seen_claims.add(claim_id)
          if (self.run_generation, claim_id) in self._claims_by_run:
            raise MemoryInvariantError("duplicate shared claim ID in this run")
        claims[slot] = claim_ids

    # Recompute deterministic first-fit before any mutation.
    expected = self._plan_reserve(plan.layout)
    if isinstance(expected, AdmissionFailure):
      raise MemoryInvariantError("arena plan placement is no longer available")
    if plan.reserve != expected:
      raise MemoryInvariantError("arena plan placement does not match deterministic first-fit")
    if self.memory_space == "l2" and not isinstance(plan.owner, RootInvocation):
      raise MemoryInvariantError("L2 backing owner must be a RootInvocation")

    allocation_generation = self._allocation_generation + 1
    arena_id = f"{self.memory_space}:p{self.pool_id}:arena:{allocation_generation}"
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

    slack: tuple[BankSegment, ...] = ()
    backing_specs: list[
      tuple[BufferLayout, int, tuple[BankSegment, ...], tuple[BankSegment, ...]]
    ] = []
    units = plan.reserve
    if plan.reserve and self.memory_space == "l2":
      round_bytes = plan.layout.stripe_bytes * self.banks
      reserve_by_bank = {segment.bank_id: segment for segment in plan.reserve}
      buffer_padded: list[tuple[BufferLayout, int]] = []
      used_ranges: dict[int, list[tuple[int, int]]] = {bank: [] for bank in reserve_by_bank}
      for buffer_layout in plan.layout.buffer_layouts:
        padded = -(-buffer_layout.logical_bytes // round_bytes) * round_bytes
        buffer_padded.append((buffer_layout, padded))
        buffer_units = self._buffer_units(handle, buffer_layout, padded)
        valid_segments = self._buffer_valid_segments(handle, buffer_layout)
        backing_specs.append((buffer_layout, padded, buffer_units, valid_segments))
        for segment in buffer_units:
          local_start = self._local_start(segment)
          used_ranges[segment.bank_id].append((local_start, local_start + segment.size_bytes))
      slack_segments: list[BankSegment] = []
      for bank_id, segment in reserve_by_bank.items():
        local_base = self._local_start(segment)
        cursor = local_base
        for start, end in sorted(used_ranges[bank_id]):
          if start > cursor:
            slack_segments.append(
              BankSegment(
                bank_id=bank_id,
                address=segment.address + (cursor - local_base),
                size_bytes=start - cursor,
              )
            )
          cursor = max(cursor, end)
        if cursor < local_base + segment.size_bytes:
          slack_segments.append(
            BankSegment(
              bank_id=bank_id,
              address=segment.address + (cursor - local_base),
              size_bytes=local_base + segment.size_bytes - cursor,
            )
          )
      slack = tuple(slack_segments)
      units = tuple(unit for _layout, _padded, parts, _valid in backing_specs for unit in parts) + slack
      self._assert_units_tile_reserve(units, plan.reserve)
      for slot, claim_ids in claims.items():
        backing_spec = next((spec for spec in backing_specs if spec[0].buffer_id == slot), None)
        if claim_ids and (backing_spec is None or not backing_spec[2]):
          raise MemoryInvariantError("shared claims require a materialized physical backing")
    elif any(claim_ids for claim_ids in claims.values()):
      raise MemoryInvariantError("shared claims require a non-empty L2 producer arena")

    # Validation is complete; the allocator commit is the only physical mutation.
    if plan.reserve:
      self._extents.commit_exact(plan.extent_pool_version, units, cycle)
    record = self._arenas[arena_id] = _ArenaRecord(
      handle=handle, run_generation=self.run_generation
    )
    record.slack_units = slack
    next_backing_counter = self._backing_counter
    for buffer_layout, padded, buffer_units, valid_segments in backing_specs:
      next_backing_counter += 1
      backing_id = f"{arena_id}:backing:{next_backing_counter}:{buffer_layout.buffer_id}"
      backing = _L2BackingRecord(
        backing_id=backing_id,
        arena_id=arena_id,
        owner=cast(RootInvocation, plan.owner),
        buffer_id=buffer_layout.buffer_id,
        units=buffer_units,
        valid_segments=valid_segments,
        logical_bytes=buffer_layout.logical_bytes,
        alignment=plan.layout.alignment,
        padded_bytes=padded,
        profile_generation=self.profile_generation,
        allocation_generation=allocation_generation,
        run_generation=self.run_generation,
        commit_cycle=cycle,
      )
      for claim_id in claims.get(buffer_layout.buffer_id, ()):
        claim = _L2ClaimRecord(claim_id, backing_id, self.run_generation)
        backing.claims[claim_id] = claim
        key = (self.run_generation, claim_id)
        self._claims_by_run[key] = backing_id
        self._claim_records[key] = claim
      self._backings[backing_id] = backing
      record.backing_ids[buffer_layout.buffer_id] = backing_id
    self._backing_counter = next_backing_counter
    self._allocation_generation = allocation_generation
    self._owner_arenas[plan.owner] = arena_id
    self._arena_reserved_bytes += handle.reserved_bytes
    self._peak_arena_reserved_bytes = max(self._peak_arena_reserved_bytes, self._arena_reserved_bytes)
    if plan.reserve:
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
    """Bind one local allocation to its committed valid-byte backing."""
    arena_record = self._arena_record(arena)
    if arena_record.state != _ArenaState.LIVE:
      raise MemoryInvariantError("cannot bind a view after arena retirement was requested")
    if buffer_id in arena_record.views:
      raise MemoryInvariantError(f"arena buffer view {buffer_id!r} is already bound")
    layouts = tuple(item for item in arena.layout.buffer_layouts if item.buffer_id == buffer_id)
    if len(layouts) != 1:
      raise MemoryInvariantError(f"unknown arena buffer layout {buffer_id!r}")
    buffer_layout = layouts[0]
    valid_segments = self._buffer_valid_segments(arena, buffer_layout)

    backing: _L2BackingRecord | None = None
    backing_id = ""
    if self.memory_space == "l2":
      backing_id = arena_record.backing_ids.get(buffer_id) or ""
      backing = self._backings.get(backing_id)
      if not backing_id or backing is None:
        raise MemoryInvariantError(f"arena buffer {buffer_id!r} has no committed L2 backing")
      if backing.state != _BackingState.LIVE:
        raise MemoryInvariantError(f"arena buffer {buffer_id!r} backing is already released")
      if backing.producer_allocation_id is not None:
        raise MemoryInvariantError(f"arena buffer {buffer_id!r} backing is already bound")
    self._assert_no_live_view_overlap(list(valid_segments), backing_id=backing_id)
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
      bank_segments=valid_segments,
      generation=arena.allocation_generation,
      allocate_cycle=cycle,
      arena_id=arena.arena_id,
      profile_generation=arena.profile_generation,
      backing_id=backing_id,
    )
    self._views[allocation_id] = _ViewRecord(handle=handle)
    arena_record.views[buffer_id] = allocation_id
    if backing is not None:
      backing.producer_allocation_id = allocation_id
      backing.view_allocation_ids.add(allocation_id)
    self._peak_live_view_bytes = max(self._peak_live_view_bytes, self._live_view_bytes())
    self._emit_trace(cycle)
    return handle

  def l2_backing_id(self, arena: ArenaHandle, slot: str) -> str:
    """Return a committed local L2 slot's stable physical identity."""
    if self.memory_space != "l2":
      raise MemoryInvariantError("L2 backing lookup requires an L2 pool")
    record = self._arena_record(arena)
    if record.state != _ArenaState.LIVE:
      raise MemoryInvariantError("cannot look up a backing in a retired arena")
    backing_id = record.backing_ids.get(slot)
    backing = self._backings.get(backing_id or "")
    if backing_id is None or backing is None or backing.state != _BackingState.LIVE:
      raise MemoryInvariantError(f"unknown live L2 backing for slot {slot!r}")
    return backing_id

  def can_borrow_l2_view(
    self, backing_id: str, owner: ContextBufferOwner, claim_id: L2ClaimId
  ) -> bool:
    """Pure admission query for a published backing and one declared claim."""
    try:
      self.check_borrow_l2_view(backing_id, owner, claim_id)
    except MemoryInvariantError:
      return False
    return True

  def check_borrow_l2_view(
    self, backing_id: str, owner: ContextBufferOwner, claim_id: L2ClaimId
  ) -> None:
    """Validate a future logical alias without changing any pool state."""
    if self.memory_space != "l2":
      raise MemoryInvariantError("shared L2 views require an L2 pool")
    backing = self._backings.get(backing_id)
    if backing is None or backing.state != _BackingState.LIVE:
      raise MemoryInvariantError("unknown or released L2 backing")
    if not backing.published:
      raise MemoryInvariantError("cannot borrow an unpublished L2 backing")
    if backing.profile_generation != self.profile_generation:
      raise MemoryInvariantError("L2 backing belongs to a stale profile generation")
    if backing.run_generation != self.run_generation:
      raise MemoryInvariantError("L2 backing belongs to a stale run generation")
    if not isinstance(owner, ContextBufferOwner):
      raise MemoryInvariantError("shared L2 view requires a ContextBufferOwner")
    if (
      not isinstance(owner.context_name, str)
      or not owner.context_name
      or type(owner.context_launch_generation) is not int
      or owner.context_launch_generation < 0
      or not isinstance(owner.buffer_id, str)
      or not owner.buffer_id
    ):
      raise MemoryInvariantError("invalid shared L2 view owner")
    if (
      not isinstance(claim_id, tuple)
      or len(claim_id) != 2
      or any(not isinstance(part, str) or not part for part in claim_id)
      or owner.buffer_id != claim_id[1]
    ):
      raise MemoryInvariantError("shared L2 owner does not match its claim")
    claim = backing.claims.get(claim_id)
    if claim is None or claim.state != _ClaimState.DECLARED:
      raise MemoryInvariantError("missing, non-declared, or already-consumed L2 claim")
    if claim.run_generation != backing.run_generation:
      raise MemoryInvariantError("L2 claim belongs to a stale run generation")
    if owner.context_name == backing.owner.context_name and (
      owner.context_launch_generation == backing.owner.launch_generation
    ):
      raise MemoryInvariantError("producer cannot borrow its own L2 backing")
    logical_owner = RootInvocation(owner.context_name, owner.context_launch_generation)
    arena_id = self._owner_arenas.get(logical_owner)
    arena_record = self._arenas.get(arena_id) if arena_id is not None else None
    if arena_record is not None and (
      arena_record.state != _ArenaState.LIVE
      or arena_record.run_generation != backing.run_generation
      or arena_record.handle.profile_generation != backing.profile_generation
    ):
      raise MemoryInvariantError("borrower arena is not in the backing run/profile")
    if arena_record is not None and (
      owner.buffer_id in arena_record.views
      or any(
        item.buffer_id == owner.buffer_id
        for item in arena_record.handle.layout.buffer_layouts
      )
    ):
      raise MemoryInvariantError("borrower slot already has a local view or allocation")
    self._assert_no_live_view_overlap(list(backing.valid_segments), backing_id=backing_id)

  def borrow_l2_view(
    self, backing_id: str, owner: ContextBufferOwner, claim_id: L2ClaimId, cycle: int
  ) -> AllocationHandle:
    """Bind one predeclared reader claim as an alias; capacity is unchanged."""
    self.check_borrow_l2_view(backing_id, owner, claim_id)
    backing = self._backings[backing_id]
    logical_owner = RootInvocation(owner.context_name, owner.context_launch_generation)
    arena_id = self._owner_arenas.get(logical_owner)
    if arena_id is None:
      raise MemoryInvariantError("borrower logical arena must be committed before binding")
    arena_record = self._arenas[arena_id]
    self._view_counter += 1
    allocation_id = f"{arena_record.handle.arena_id}:view:{self._view_counter}:{owner.buffer_id}"
    base_address = backing.valid_segments[0].address if backing.valid_segments else 0
    handle = AllocationHandle(
      allocation_id=allocation_id,
      memory_space="l2",
      owner=owner,
      base_address=base_address,
      size_bytes=backing.logical_bytes,
      alignment=backing.alignment,
      bank_segments=backing.valid_segments,
      generation=backing.allocation_generation,
      allocate_cycle=cycle,
      arena_id=arena_record.handle.arena_id,
      profile_generation=backing.profile_generation,
      backing_id=backing.backing_id,
    )
    claim = backing.claims[claim_id]
    self._views[allocation_id] = _ViewRecord(handle=handle, claim_id=claim_id)
    arena_record.views[owner.buffer_id] = allocation_id
    backing.view_allocation_ids.add(allocation_id)
    claim.state = _ClaimState.BOUND
    claim.bound_allocation_id = allocation_id
    claim.owner = owner
    self._peak_live_view_bytes = max(self._peak_live_view_bytes, self._live_view_bytes())
    self._emit_trace(cycle)
    return handle

  def check_publish_l2(
    self, handle: AllocationHandle, *, allowed_pins: tuple[str, ...] = ()
  ) -> None:
    """Read-only publish preflight; allowed pins are completed writer pins."""
    if self.memory_space != "l2" or not handle.backing_id:
      raise MemoryInvariantError("publish requires an L2 producer view")
    record = self._view_record(handle)
    backing = self._backings.get(handle.backing_id)
    if backing is None or backing.state != _BackingState.LIVE:
      raise MemoryInvariantError("publish references an unknown L2 backing")
    if backing.run_generation != self.run_generation or (
      backing.profile_generation != self.profile_generation
    ):
      raise MemoryInvariantError("L2 backing belongs to a stale run or profile generation")
    if backing.producer_allocation_id != handle.allocation_id:
      raise MemoryInvariantError("only the producer view may publish an L2 backing")
    if not backing.producer_live or record.state != _ViewState.LIVE:
      raise MemoryInvariantError("cannot publish a released L2 producer view")
    if backing.published:
      raise MemoryInvariantError("L2 backing is already published")
    if not isinstance(handle.owner, ContextBufferOwner) or (
      handle.owner.context_name != backing.owner.context_name
      or handle.owner.context_launch_generation != backing.owner.launch_generation
    ):
      raise MemoryInvariantError("wrong producer owner for L2 publish")
    self._assert_backing_reference_mirrors(backing)
    expected_pins = set(allowed_pins)
    if len(allowed_pins) != len(expected_pins):
      raise MemoryInvariantError("allowed publish pins must be unique")
    producer_pins = {ref for allocation_id, ref in backing.pins if allocation_id == handle.allocation_id}
    if producer_pins != expected_pins or len(backing.pins) != len(producer_pins):
      raise MemoryInvariantError("L2 publish has unexpected active pins")
    if backing.inflight:
      raise MemoryInvariantError("L2 publish has active transfers")
    if any(claim.state != _ClaimState.DECLARED for claim in backing.claims.values()):
      raise MemoryInvariantError("L2 publish has a non-pending reader claim")
    if backing.view_allocation_ids != {handle.allocation_id}:
      raise MemoryInvariantError("L2 publish has unexpected alias views")

  def publish_l2(self, handle: AllocationHandle, cycle: int) -> None:
    """Make a producer backing immutable after every writer access completes."""
    self.check_publish_l2(handle)
    self._backings[handle.backing_id].published = True
    self._emit_trace(cycle)

  def permissions(self, handle: AllocationHandle) -> str:
    record = self._view_record(handle)
    if record.state == _ViewState.INVALIDATED:
      raise MemoryInvariantError("use-after-release")
    if self.memory_space != "l2":
      return "rw"
    backing = self._backings.get(handle.backing_id)
    if backing is None or backing.state != _BackingState.LIVE:
      raise MemoryInvariantError("use-after-release")
    if backing.producer_allocation_id == handle.allocation_id:
      return "r" if backing.published else "rw"
    return "r"

  def assert_access(self, handle: AllocationHandle, permission: str) -> None:
    if permission not in ("r", "w"):
      raise MemoryInvariantError("memory permission must be 'r' or 'w'")
    available = self.permissions(handle)
    if permission not in available:
      raise MemoryInvariantError("write access to read-only L2 backing")

  def invalidate_view(self, view: AllocationHandle, owner: MemoryOwner, cycle: int) -> bool:
    """Revoke one logical view; physical release waits for every claim and alias."""
    record = self._view_record(view)
    if record.state != _ViewState.LIVE:
      raise MemoryInvariantError("double release")
    if record.handle.owner != owner:
      raise MemoryInvariantError("wrong-owner release")
    if record.pins or record.inflight:
      record.state = _ViewState.INVALIDATE_PENDING
      record.invalidate_cycle = cycle
      self._emit_trace(cycle)
      return False
    self._check_finalize_view(record)
    self._finalize_view(record, cycle)
    self._release_backing_for(view, cycle)
    return True

  def cancel_l2_view(self, view: AllocationHandle, owner: MemoryOwner, cycle: int) -> bool:
    """Internally cancel an isolated L2 view and any bound claim."""
    if self.memory_space != "l2" or not view.backing_id:
      raise MemoryInvariantError("internal claim cancellation requires an L2 view")
    record = self._view_record(view)
    if record.handle.owner != owner:
      raise MemoryInvariantError("wrong-owner release")
    if record.state == _ViewState.INVALIDATED:
      return False
    if record.state not in (_ViewState.LIVE, _ViewState.INVALIDATE_PENDING):
      raise MemoryInvariantError("cannot cancel an L2 view in an unknown state")
    if record.pins or record.inflight:
      raise MemoryInvariantError("cannot cancel an L2 view with retained references")
    self._check_finalize_view(record)
    self._finalize_view(record, cycle, claim_state=_ClaimState.CANCELLED)
    self._release_backing_for(view, cycle)
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
    self._backing_reference(handle, consumer_id, pin=True, add=True)

  def unpin(self, handle: AllocationHandle, consumer_id: str, cycle: int) -> bool:
    record = self._view_record(handle)
    if record.state == _ViewState.INVALIDATED:
      raise MemoryInvariantError("use-after-release")
    if record.state == _ViewState.INVALIDATE_PENDING and len(record.pins) == 1 and not record.inflight:
      self._check_finalize_view(record, releasing=consumer_id, pin=True)
    self._backing_reference(handle, consumer_id, pin=True, add=False)
    if record.state == _ViewState.INVALIDATE_PENDING and not record.pins and not record.inflight:
      self._finalize_view(record, cycle)
      self._release_backing_for(handle, cycle)
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
    self._backing_reference(handle, transaction_id, pin=False, add=True)

  def end_inflight(self, handle: AllocationHandle, transaction_id: str, cycle: int) -> bool:
    """Finish one accepted transaction and possibly complete invalidation."""
    record = self._view_record(handle)
    if record.state == _ViewState.INVALIDATED:
      raise MemoryInvariantError("use-after-release")
    if (
      record.state == _ViewState.INVALIDATE_PENDING
      and len(record.inflight) == 1
      and not record.pins
    ):
      self._check_finalize_view(record, releasing=transaction_id, pin=False)
    self._backing_reference(handle, transaction_id, pin=False, add=False)
    if record.state == _ViewState.INVALIDATE_PENDING and not record.pins and not record.inflight:
      self._finalize_view(record, cycle)
      self._release_backing_for(handle, cycle)
      return True
    self._emit_trace(cycle)
    return False
  def _backing_reference(
    self, handle: AllocationHandle, reference: str, *, pin: bool, add: bool
  ) -> None:
    """Atomically update the view and its exact (allocation, reference) mirror."""
    record = self._view_record(handle)
    view_targets = record.pins if pin else record.inflight
    backing = self._backings.get(handle.backing_id) if handle.backing_id else None
    if handle.backing_id and (backing is None or backing.state != _BackingState.LIVE):
      raise MemoryInvariantError("view references an unknown physical backing")
    if backing is not None:
      self._assert_backing_reference_mirrors(backing)
    backing_targets = backing.pins if backing is not None and pin else (
      backing.inflight if backing is not None else None
    )
    pair = (handle.allocation_id, reference)
    if add:
      if reference in view_targets:
        raise MemoryInvariantError(
          "duplicate allocation pin" if pin else "duplicate in-flight allocation reference"
        )
      if backing_targets is not None and pair in backing_targets:
        raise MemoryInvariantError("view and backing reference ledgers disagree")
    else:
      if reference not in view_targets:
        message = "unknown allocation pin" if pin else "unknown in-flight allocation reference"
        raise MemoryInvariantError(message)
      if handle.backing_id and (
        backing_targets is None or pair not in backing_targets
      ):
        raise MemoryInvariantError("view and backing reference ledgers disagree")
    if add:
      view_targets.add(reference)
      if backing_targets is not None:
        backing_targets.add(pair)
    else:
      view_targets.remove(reference)
      if backing_targets is not None:
        backing_targets.remove(pair)
  def _release_backing_for(self, handle: AllocationHandle, cycle: int) -> bool:
    if not handle.backing_id:
      return False
    return self._try_release_l2_backing(handle.backing_id, cycle)

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
    """Retire logical arena metadata and release only its remaining physical units."""
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
    if self.memory_space == "l2":
      held_slack = sum(segment.size_bytes for segment in record.slack_units)
      for segment in record.slack_units:
        self._local_start(segment)
      origin_backings: list[_L2BackingRecord] = []
      for backing_id in record.backing_ids.values():
        backing = self._backings.get(backing_id)
        if backing is None:
          raise MemoryInvariantError("origin arena references an unknown L2 backing")
        origin_backings.append(backing)
        self._assert_backing_reference_mirrors(backing)
        if backing.producer_allocation_id is not None and backing.producer_live:
          raise MemoryInvariantError("cannot retire an arena with a live producer view")
        if (
          backing.state == _BackingState.LIVE
          and backing.producer_allocation_id is None
          and any(
            claim.state in (_ClaimState.DECLARED, _ClaimState.BOUND)
            for claim in backing.claims.values()
          )
        ):
          raise MemoryInvariantError("cannot retire an unbound producer with shared claims")
      for backing in origin_backings:
        if backing.state == _BackingState.LIVE and backing.producer_allocation_id is None:
          backing.producer_live = False
        if backing.state == _BackingState.LIVE:
          self._try_release_l2_backing(backing.backing_id, cycle)
      if record.slack_units:
        self._extents.release_exact(self._extents.pool_version, record.slack_units, cycle)
        self._arena_reserved_bytes -= held_slack
        record.slack_units = ()
        self._pool_version += 1
    elif arena.reserve:
      for segment in arena.reserve:
        self._local_start(segment)
      self._extents.release_exact(self._extents.pool_version, arena.reserve, cycle)
      self._arena_reserved_bytes -= arena.reserved_bytes
      self._pool_version += 1
    if self.memory_space == "l2":
      for backing in origin_backings:
        backing.origin_retired = True
    record.state = _ArenaState.RETIRED
    record.retire_cycle = cycle
    self._owner_arenas.pop(arena.owner, None)
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
    live_backings = [record for record in self._backings.values() if record.state == _BackingState.LIVE]
    materialized_claims = [
      claim
      for claim in self._claim_records.values()
      if claim.state in (_ClaimState.DECLARED, _ClaimState.BOUND)
    ]
    retained_references = [record for record in self._views.values() if record.pins or record.inflight]
    if (
      live_arenas
      or live_views
      or live_backings
      or materialized_claims
      or retained_references
      or self._owner_arenas
    ):
      raise MemoryInvariantError("cannot reconfigure an arena pool with live state")
    if self._arena_reserved_bytes != 0:
      raise MemoryInvariantError("cannot reconfigure with reserved arena bytes")

    self._extents.reconfigure_free_intervals(
      profile.system_reserved_spm_per_bank, profile.spm_bytes_per_bank, cycle
    )
    self._active_profile = profile
    self._profile_generation = generation
    # Retired records from an old profile are no longer useful for identity
    # checks; claim tombstones persist until run closure.
    self._arenas.clear()
    self._views.clear()
    self._backings.clear()
    self._owner_arenas.clear()
    self._pool_version += 1
    self._emit_trace(cycle)

  # -- L2 physical backing lifecycle ------------------------------------

  def _buffer_units(
    self, handle: ArenaHandle, buffer_layout: BufferLayout, padded_bytes: int
  ) -> tuple[BankSegment, ...]:
    """Exact per-bank padded units of one buffer inside its arena reserve."""
    span_per_bank = padded_bytes // self.banks
    if span_per_bank * self.banks != padded_bytes:
      raise MemoryInvariantError("padded backing span does not split evenly across banks")
    reserve_by_bank = {segment.bank_id: segment for segment in handle.reserve}
    result: list[BankSegment] = []
    offset = buffer_layout.arena_offset // self.banks
    for bank_id in range(self.banks):
      reserve = reserve_by_bank.get(bank_id)
      if reserve is None or span_per_bank == 0:
        if span_per_bank:
          raise MemoryInvariantError("backing unit has no reserve on its bank")
        continue
      self._local_start(reserve)
      if offset + span_per_bank > reserve.size_bytes:
        raise MemoryInvariantError("backing unit escapes its arena reserve")
      result.append(
        BankSegment(bank_id=bank_id, address=reserve.address + offset, size_bytes=span_per_bank)
      )
    return tuple(result)

  def _buffer_valid_segments(
    self, arena: ArenaHandle, buffer_layout: BufferLayout
  ) -> tuple[BankSegment, ...]:
    reserve_by_bank = {segment.bank_id: segment for segment in arena.reserve}
    valid_segments: list[BankSegment] = []
    for bank_id, relative_start, size in buffer_layout.segments():
      reserve = reserve_by_bank.get(bank_id)
      if reserve is None or relative_start < 0 or relative_start + size > reserve.size_bytes:
        raise MemoryInvariantError("buffer view lies outside its arena reserve")
      valid_segments.append(
        BankSegment(bank_id=bank_id, address=reserve.address + relative_start, size_bytes=size)
      )
    if sum(segment.size_bytes for segment in valid_segments) != buffer_layout.logical_bytes:
      raise MemoryInvariantError("buffer view valid-byte segments are incomplete")
    return tuple(valid_segments)

  def _assert_units_tile_reserve(
    self, units: tuple[BankSegment, ...], reserve: tuple[BankSegment, ...]
  ) -> None:
    """Units must exactly tile the reserve: aligned, bounded, no overlap/gap."""
    per_bank: dict[int, list[tuple[int, int]]] = {}
    for segment in units:
      local_start = self._local_start(segment)
      per_bank.setdefault(segment.bank_id, []).append((local_start, local_start + segment.size_bytes))
    reserve_by_bank = {segment.bank_id: segment for segment in reserve}
    for bank_id, reserve_segment in reserve_by_bank.items():
      base = self._local_start(reserve_segment)
      spans = sorted(per_bank.get(bank_id, []))
      for (_start, end), (next_start, _next_end) in pairwise(spans):
        if next_start < end:
          raise MemoryInvariantError("exact L2 units overlap inside one bank")
        if next_start != end:
          raise MemoryInvariantError("exact L2 units leave a gap inside one bank")
      if not spans or spans[0][0] != base or spans[-1][1] != base + reserve_segment.size_bytes:
        raise MemoryInvariantError("exact L2 units do not tile their bank reserve")
    if sum(segment.size_bytes for segment in units) != sum(
      segment.size_bytes for segment in reserve
    ):
      raise MemoryInvariantError("exact L2 units do not conserve the arena reservation")

  def has_inflight_references(self, handle: AllocationHandle) -> bool:
    """Check this view and prove its refs exactly mirror the backing ledger."""
    record = self._view_record(handle)
    if not handle.backing_id:
      return bool(record.inflight)
    backing = self._backings.get(handle.backing_id)
    if backing is None:
      raise MemoryInvariantError("view references an unknown physical backing")
    self._assert_backing_reference_mirrors(backing)
    return bool(record.inflight)

  def _assert_backing_reference_mirrors(self, backing: _L2BackingRecord) -> None:
    expected_pins: set[tuple[str, str]] = set()
    expected_inflight: set[tuple[str, str]] = set()
    actual_views = {
      allocation_id
      for allocation_id, record in self._views.items()
      if record.handle.backing_id == backing.backing_id
    }
    if actual_views != backing.view_allocation_ids:
      raise MemoryInvariantError("view and backing records disagree on aliases")
    for allocation_id in actual_views:
      record = self._views[allocation_id]
      expected_pins.update((allocation_id, ref) for ref in record.pins)
      expected_inflight.update((allocation_id, ref) for ref in record.inflight)
    if expected_pins != backing.pins:
      raise MemoryInvariantError("view and backing pin ledgers disagree")
    if expected_inflight != backing.inflight:
      raise MemoryInvariantError("view and backing inflight ledgers disagree")

  def live_backing_ids(self) -> list[str]:
    """Concrete IDs of every physically held backing (leak reporting)."""
    return sorted(
      backing_id
      for backing_id, backing in self._backings.items()
      if backing.state == _BackingState.LIVE
    )

  def backing_claims(self, backing_id: str) -> tuple[tuple[L2ClaimId, str], ...]:
    backing = self._backings.get(backing_id)
    if backing is not None:
      if backing.state != _BackingState.LIVE:
        return ()
      return tuple(
        (claim_id, backing.claims[claim_id].state) for claim_id in sorted(backing.claims)
      )
    terminal_claims = tuple(
      sorted(
        (claim.claim_id, claim.state)
        for claim in self._claim_records.values()
        if claim.backing_id == backing_id
      )
    )
    if not terminal_claims:
      raise MemoryInvariantError("unknown L2 backing")
    return terminal_claims

  def claim_snapshot(self) -> tuple[dict, ...]:
    return tuple(
      claim.snapshot()
      for _key, claim in sorted(self._claim_records.items(), key=lambda item: item[0])
    )

  def cancel_all_l2_claims(self, cycle: int) -> int:
    """Cancel declared readers; bound claims close through their view release."""
    changed = 0
    for claim in tuple(self._claim_records.values()):
      if claim.state == _ClaimState.DECLARED:
        changed += self.cancel_l2_claim(claim.backing_id, claim.claim_id, cycle)
    return changed

  def cancel_l2_claim(self, backing_id: str, claim_id: L2ClaimId, cycle: int) -> bool:
    backing = self._backings.get(backing_id)
    claim = backing.claims.get(claim_id) if backing is not None else None
    if claim is None:
      claim = next(
        (
          candidate
          for candidate in self._claim_records.values()
          if candidate.backing_id == backing_id and candidate.claim_id == claim_id
        ),
        None,
      )
    if claim is None:
      raise MemoryInvariantError("unknown L2 claim")
    if claim.state in (_ClaimState.CANCELLED, _ClaimState.RELEASED):
      return False
    if backing is None or backing.state != _BackingState.LIVE:
      raise MemoryInvariantError("cannot cancel an active claim without its live backing")
    if claim.state == _ClaimState.BOUND:
      view = self._views.get(claim.bound_allocation_id or "")
      if view is None or view.state != _ViewState.INVALIDATED:
        raise MemoryInvariantError("cannot cancel a bound claim before its view is released")
      if view.pins or view.inflight:
        raise MemoryInvariantError("cannot cancel a bound claim with retained references")
    claim.state = _ClaimState.CANCELLED
    self._try_release_l2_backing(backing_id, cycle)
    self._emit_trace(cycle)
    return True


  def close_l2_claims(self) -> None:
    """Forget only a fully terminal claim manifest after run closure."""
    if any(
      claim.state not in (_ClaimState.CANCELLED, _ClaimState.RELEASED)
      for claim in self._claim_records.values()
    ):
      raise MemoryInvariantError("cannot close a live L2 claim manifest")
    if self.live_backing_ids():
      raise MemoryInvariantError("cannot close L2 claims while physical backings remain")
    if (
      self._arena_reserved_bytes
      or self._owner_arenas
      or any(record.state != _ArenaState.RETIRED for record in self._arenas.values())
      or any(record.state != _ViewState.INVALIDATED for record in self._views.values())
      or any(record.pins or record.inflight for record in self._views.values())
    ):
      raise MemoryInvariantError("cannot close the L2 claim manifest before pool closure")
    for backing in self._backings.values():
      backing.claims.clear()
    self._claim_records.clear()
    self._claims_by_run.clear()

  def _try_release_l2_backing(self, backing_id: str, cycle: int) -> bool:
    """Return a padded span only after producer, claim, alias, and ref closure."""
    backing = self._backings.get(backing_id)
    if backing is None or backing.state != _BackingState.LIVE:
      return False
    self._assert_backing_reference_mirrors(backing)
    if backing.producer_live or backing.pins or backing.inflight:
      return False
    if any(claim.state in (_ClaimState.DECLARED, _ClaimState.BOUND) for claim in backing.claims.values()):
      return False
    if any(
      self._views[allocation_id].state != _ViewState.INVALIDATED
      for allocation_id in backing.view_allocation_ids
    ):
      return False
    if backing.units:
      self._extents.release_exact(self._extents.pool_version, backing.units, cycle)
    backing.state = _BackingState.RELEASED
    backing.release_cycle = cycle
    self._arena_reserved_bytes -= backing.padded_bytes
    if backing.units:
      self._pool_version += 1
    self._emit_trace(cycle)
    if not backing.units:
      return True
    event = {
      **backing.snapshot(),
      "run_generation": backing.run_generation,
      "pool_version": self._pool_version,
      "extent_pool_version": self._extents.pool_version,
      "per_bank_segments": [
        {"bank_id": segment.bank_id, "address": segment.address, "size_bytes": segment.size_bytes}
        for segment in backing.units
      ],
    }
    self._backing_release_events.append(event)
    if self._trace is not None:
      self._trace.l2_extent_release(self.tile_id, backing, self.snapshot(), event, cycle)
    return True

  def drain_l2_backing_release_events(self) -> list[dict]:
    """Return and clear physical final-free events for the occupancy mirror."""
    if not self._backing_release_events:
      return []
    events = self._backing_release_events
    self._backing_release_events = []
    return events

  def arena_held_bytes(self, arena: ArenaHandle) -> int:
    """Physical bytes still attributable to an active logical arena."""
    record = self._arena_record(arena)
    if record.state == _ArenaState.RETIRED:
      return 0
    if self.memory_space != "l2":
      return arena.reserved_bytes
    backing_bytes = 0
    for backing_id in record.backing_ids.values():
      backing = self._backings.get(backing_id)
      if backing is None:
        raise MemoryInvariantError("arena references an unknown L2 backing")
      if backing.state == _BackingState.LIVE:
        backing_bytes += backing.padded_bytes
    return sum(segment.size_bytes for segment in record.slack_units) + backing_bytes

  def can_fit_with_retained(self, layout: ArenaLayout, backing_ids: tuple[str, ...]) -> bool:
    """Purely test first-fit after retaining only the specified shared backings."""
    layout_error = self._layout_error(layout)
    if layout_error is not None:
      raise MemoryInvariantError(layout_error[0])
    if (
      not isinstance(backing_ids, tuple)
      or any(not isinstance(backing_id, str) or not backing_id for backing_id in backing_ids)
      or len(set(backing_ids)) != len(backing_ids)
    ):
      raise MemoryInvariantError("retained backing IDs must be a unique tuple of strings")
    free_by_bank: list[list[tuple[int, int]]] = [
      [(self.profile.system_reserved_spm_per_bank, self.profile.user_spm_per_bank)]
      if self.profile.user_spm_per_bank
      else []
      for _bank in range(self.banks)
    ]
    for backing_id in backing_ids:
      backing = self._backings.get(backing_id)
      if backing is None or backing.state != _BackingState.LIVE:
        raise MemoryInvariantError("retained set contains an unknown or released L2 backing")
      if backing.run_generation != self.run_generation:
        raise MemoryInvariantError("retained backing belongs to a stale run generation")
      if not any(
        claim.state in (_ClaimState.DECLARED, _ClaimState.BOUND)
        for claim in backing.claims.values()
      ):
        raise MemoryInvariantError("retained set contains a backing without live shared claims")
      for segment in backing.units:
        start = self._local_start(segment)
        end = start + segment.size_bytes
        replacement: list[tuple[int, int]] = []
        covered = False
        for free_start, free_size in free_by_bank[segment.bank_id]:
          free_end = free_start + free_size
          if free_start <= start and end <= free_end:
            replacement.extend(
              span
              for span in ((free_start, start - free_start), (end, free_end - end))
              if span[1] > 0
            )
            covered = True
          else:
            replacement.append((free_start, free_size))
        if not covered:
          raise MemoryInvariantError("retained backing units overlap in the pristine free map")
        free_by_bank[segment.bank_id] = replacement
    return all(
      size == 0 or self._first_fit(free_by_bank[bank_id], bank_id, size, layout.alignment) is not None
      for bank_id, size in enumerate(layout.per_bank_bytes)
    )

  # -- deterministic observability -------------------------------------

  def snapshot(self) -> dict:
    """Return physical-capacity and logical-alias state from the pool ledger."""
    profile = self.profile
    reserved_by_bank = [0] * self.banks
    live_by_bank = [0] * self.banks
    live_arena_records = tuple(
      record for record in self._arenas.values() if record.state != _ArenaState.RETIRED
    )
    live_backing_records = tuple(
      backing for backing in self._backings.values() if backing.state == _BackingState.LIVE
    )
    for backing in live_backing_records:
      self._assert_backing_reference_mirrors(backing)
      for segment in backing.units:
        reserved_by_bank[segment.bank_id] += segment.size_bytes
      for segment in backing.valid_segments:
        live_by_bank[segment.bank_id] += segment.size_bytes
    if self.memory_space == "l2":
      for arena_record in live_arena_records:
        for segment in arena_record.slack_units:
          reserved_by_bank[segment.bank_id] += segment.size_bytes
    else:
      for arena_record in live_arena_records:
        for segment in arena_record.handle.reserve:
          reserved_by_bank[segment.bank_id] += segment.size_bytes
        for allocation_id in arena_record.views.values():
          view_record = self._views[allocation_id]
          if view_record.state == _ViewState.INVALIDATED:
            continue
          for segment in view_record.handle.bank_segments:
            live_by_bank[segment.bank_id] += segment.size_bytes
    if sum(reserved_by_bank) != self._arena_reserved_bytes:
      raise MemoryInvariantError("physical reserved-byte ledger disagrees with per-bank extents")

    per_bank: list[dict] = []
    free_total = 0
    largest = 0
    free_extents = self._extents.free_extents_snapshot()
    for bank_id, extents in enumerate(free_extents):
      free_bytes = sum(size for _, size in extents)
      largest_extent = max((size for _, size in extents), default=0)
      if reserved_by_bank[bank_id] + free_bytes != profile.user_spm_per_bank:
        raise MemoryInvariantError("per-bank SPM capacity is not conserved")
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
          "padding_bytes": reserved_by_bank[bank_id] - live_by_bank[bank_id],
          "free_bytes": free_bytes,
          "largest_free_extent": largest_extent,
        }
      )

    view_records = tuple(self._views.values())
    live_view_records = tuple(
      record for record in view_records if record.state != _ViewState.INVALIDATED
    )
    arena_rows = tuple(
      {
        "arena_id": record.handle.arena_id,
        "owner": repr(record.handle.owner),
        "state": record.state,
        "run_generation": record.run_generation,
        "reserved_bytes": self.arena_held_bytes(record.handle),
        "initial_reserved_bytes": record.handle.reserved_bytes,
        "task_metadata": record.task_metadata,
        "live_views": sum(
          self._views[allocation_id].state != _ViewState.INVALIDATED
          for allocation_id in record.views.values()
        ),
        "live_backings": sum(
          backing.state == _BackingState.LIVE
          and backing.arena_id == record.handle.arena_id
          for backing in self._backings.values()
        ),
      }
      for record in sorted(self._arenas.values(), key=lambda item: item.handle.arena_id)
      if record.state != _ArenaState.RETIRED
    )

    def backing_row(backing: _L2BackingRecord) -> dict:
      live_view_ids = tuple(
        sorted(
          allocation_id
          for allocation_id in backing.view_allocation_ids
          if self._views[allocation_id].state != _ViewState.INVALIDATED
        )
      )
      return {
        **backing.snapshot(),
        "live_view_allocation_ids": live_view_ids,
        "live_view_count": len(live_view_ids),
      }

    backing_rows = tuple(
      backing_row(backing)
      for backing in sorted(live_backing_records, key=lambda item: item.backing_id)
    )
    origin_retired = tuple(
      backing_row(backing)
      for backing in sorted(live_backing_records, key=lambda item: item.backing_id)
      if backing.origin_retired
    )
    shared_claims = self.claim_snapshot()
    pending_claims = sum(
      claim.state == _ClaimState.DECLARED
      for backing in live_backing_records
      for claim in backing.claims.values()
    )
    active_claims = sum(
      claim.state == _ClaimState.BOUND
      for backing in live_backing_records
      for claim in backing.claims.values()
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
      "physical_live_backing_bytes": sum(backing.padded_bytes for backing in live_backing_records),
      "free_bytes": free_total,
      "largest_free_extent": largest,
      "peak_arena_reserved_bytes": self._peak_arena_reserved_bytes,
      "peak_allocated_bytes": self._peak_arena_reserved_bytes,
      "live_view_bytes": self._live_view_bytes(),
      "logical_live_view_bytes": sum(record.handle.size_bytes for record in live_view_records),
      "peak_live_view_bytes": self._peak_live_view_bytes,
      "padding_bytes": sum(
        reserved_by_bank[bank] - live_by_bank[bank] for bank in range(self.banks)
      ),
      "live_backings": len(live_backing_records),
      "pending_shared_claims": pending_claims,
      "active_shared_references": active_claims,
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
      "backings": backing_rows,
      "origin_retired_backings": origin_retired,
      "shared_claims": shared_claims,
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

  def _assert_no_live_view_overlap(
    self, segments: list[BankSegment], *, backing_id: str = ""
  ) -> None:
    for record in self._views.values():
      if record.state == _ViewState.INVALIDATED:
        continue
      if backing_id and record.handle.backing_id == backing_id:
        backing = self._backings.get(backing_id)
        if backing is not None and backing.published:
          continue
      for candidate in segments:
        candidate_end = candidate.address + candidate.size_bytes
        for live in record.handle.bank_segments:
          if live.bank_id != candidate.bank_id:
            continue
          if candidate.address < live.address + live.size_bytes and live.address < candidate_end:
            raise MemoryInvariantError("arena view overlaps a live buffer view")

  def _check_finalize_view(
    self, record: _ViewRecord, *, releasing: str | None = None, pin: bool = False
  ) -> None:
    pins = set(record.pins)
    inflight = set(record.inflight)
    if releasing is not None:
      targets = pins if pin else inflight
      if releasing not in targets:
        message = "unknown allocation pin" if pin else "unknown in-flight allocation reference"
        raise MemoryInvariantError(message)
      targets.remove(releasing)
    if pins or inflight:
      raise MemoryInvariantError("cannot invalidate a referenced arena view")
    handle = record.handle
    if not handle.backing_id:
      return
    backing = self._backings.get(handle.backing_id)
    if backing is None:
      raise MemoryInvariantError("view references an unknown physical backing")
    self._assert_backing_reference_mirrors(backing)
    if handle.allocation_id not in backing.view_allocation_ids:
      raise MemoryInvariantError("view and backing records disagree on binding")
    if backing.producer_allocation_id == handle.allocation_id:
      if not backing.producer_live:
        raise MemoryInvariantError("producer view ownership was already released")
      return
    claim = backing.claims.get(record.claim_id) if record.claim_id is not None else None
    if (
      claim is None
      or claim.state != _ClaimState.BOUND
      or claim.bound_allocation_id != handle.allocation_id
    ):
      raise MemoryInvariantError("borrower view and L2 claim ledger disagree")

  def _finalize_view(
    self, record: _ViewRecord, cycle: int, *, claim_state: str = _ClaimState.RELEASED
  ) -> None:
    if record.state == _ViewState.INVALIDATED:
      raise MemoryInvariantError("double release")
    self._check_finalize_view(record)
    backing = (
      self._backings.get(record.handle.backing_id) if record.handle.backing_id else None
    )
    if backing is not None:
      if backing.producer_allocation_id == record.handle.allocation_id:
        backing.producer_live = False
      else:
        assert record.claim_id is not None
        claim = backing.claims[record.claim_id]
        claim.state = claim_state
    record.state = _ViewState.INVALIDATED
    record.invalidate_cycle = cycle
    self._emit_trace(cycle)
    if self._trace is not None:
      self._trace.buffer_view_invalidate(
        self.memory_space, self.tile_id, record.handle, self.snapshot(), cycle, backing=backing
      )

  def _live_view_bytes(self) -> int:
    if self.memory_space == "l2":
      return sum(
        backing.logical_bytes
        for backing in self._backings.values()
        if backing.state == _BackingState.LIVE
      )
    return sum(
      record.handle.size_bytes
      for record in self._views.values()
      if record.state != _ViewState.INVALIDATED
    )

  def _emit_trace(self, cycle: int) -> None:
    if self._trace is None:
      return
    snapshot = self.snapshot()
    self._trace.capacity(self.memory_space, self.tile_id, snapshot, cycle)
    self._trace.banks(self.memory_space, self.tile_id, snapshot["per_bank_occupancy"], cycle)
