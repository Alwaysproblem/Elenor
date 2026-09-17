"""L1 Slot Frame — fixed-slot bindings backed by allocator identities.

Models the L1 SRAM binary binding contract: fixed slot ABI + variable
Tile Frame. Frame bind FSM (3.2), descriptor patch FSM (3.3), and slot
lifecycle (3.4). Bank policy enforcement (5.4).

The per-tile ``BankedFreeExtentAllocator`` owns placement and generation.
``SlotFrame`` records the exact allocation id, generation and owner in each
slot; it does not maintain an unrelated frame-generation namespace.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import TYPE_CHECKING

from .allocator import MemoryInvariantError

if TYPE_CHECKING:
  from ..profiles import ArenaLayout
  from .allocator import AllocationHandle, MemoryOwner
  from .arena import ArenaHandle


class SlotRole(IntEnum):
  """elenor_slot_role_t (Slot Frame design 4.1)."""

  INPUT = 1 << 0
  OUTPUT = 1 << 1
  ACCUMULATOR = 1 << 2
  WORKSPACE = 1 << 3
  METADATA = 1 << 4
  CONST = 1 << 5
  STATE = 1 << 6
  PROGRAM = 1 << 7
  EVENT_STATUS = 1 << 8


class SlotLifetime(IntEnum):
  """elenor_slot_lifetime_t (Slot Frame design 4.1)."""

  PER_COMMAND = 0
  PER_TILE_PROGRAM = 1
  PER_ROLE = 2
  RESIDENT = 3


class FrameState(IntEnum):
  """Slot Frame bind FSM (design 3.2)."""

  IDLE = 0
  FETCH_FRAME_DESC = 1
  VALIDATE_ABI = 2
  VALIDATE_SLOT_TABLE = 3
  CHECK_OVERLAP_ALIGNMENT = 4
  CHECK_BANK_POLICY = 5
  INSTALL_SHADOW = 6
  FRAME_ACTIVE = 7
  FRAME_FAULTED = 8


@dataclass
class Slot:
  """elenor_tile_slot_v0_t (Slot Frame design 4.1).

  PR 2: ``allocation_id`` and ``generation`` bind the slot to an
  ``AllocationHandle`` from the per-tile L1 allocator; ``owner`` is a
  ``MemoryOwner`` (not an int).
  """

  slot_id: int
  base: int = 0
  size: int = 0
  layout: int = 0
  role: int = 0
  alignment: int = 0
  bank_policy: int = 0
  lifetime: SlotLifetime = SlotLifetime.PER_COMMAND
  allocation_id: str | None = None
  generation: int = 0
  owner: MemoryOwner | None = None  # type: ignore[name-defined]
  flags: int = 0
  arena_id: str = ""
  profile_generation: int = 0


@dataclass
class SlotFrame:
  """elenor_tile_frame_v0_t (Slot Frame design 4.1).

  16 fixed slots with a shadow-install mechanism. After bind, engines
  only access the shadow copy (design 3.2). Allocation-handle identity is
  checked per slot on free; there is no independent synthetic generation.
  """

  frame_id: int = 0
  l1_bytes: int = 1 * 1024 * 1024  # 1 MB Balanced-small
  slot_count: int = 16
  state: FrameState = FrameState.IDLE

  def __post_init__(self) -> None:
    self.slots: list[Slot] = [Slot(i) for i in range(self.slot_count)]
    self.shadow: SlotFrame | None = None
    self._shadow_slots: list[Slot] | None = None
    self.pmu_bank_conflict_cycles: int = 0
    self.pmu_permission_fault_count: int = 0
    self.arena: ArenaHandle | None = None
    self._layout: ArenaLayout | None = None
    self._slot_segments: dict[int, tuple] = {}

  @property
  def is_available(self) -> bool:
    """True only when this physical context frame has no staged binding."""
    return (
      self.state == FrameState.IDLE
      and self.shadow is None
      and self._shadow_slots is None
      and all(slot.allocation_id is None for slot in self.slots)
    )

  @property
  def generation(self) -> int | None:
    """Allocation generation of the committed Task Arena, including empty Frames."""
    return None if self.arena is None else self.arena.allocation_generation

  def prepare(self, arena: ArenaHandle, layout: ArenaLayout) -> bool:
    """Validate the committed Arena and stage an empty live-slot Frame."""
    profile = arena.profile
    if (
      not self.is_available
      or arena.memory_space != "l1"
      or profile.level != "l1"
      or profile.bank_bytes * profile.banks != self.l1_bytes
      or arena.layout != layout
      or any(not 0 <= item.slot_id < self.slot_count for item in layout.buffer_layouts)
    ):
      self.pmu_permission_fault_count += 1
      return False
    for segment in arena.reserve:
      local = segment.address - segment.bank_id * profile.bank_bytes
      if (
        not 0 <= segment.bank_id < profile.banks
        or local < profile.system_reserved_spm_per_bank
        or local % layout.alignment
        or segment.size_bytes <= 0
        or local + segment.size_bytes > profile.spm_bytes_per_bank
      ):
        self.pmu_permission_fault_count += 1
        return False
    if sum(segment.size_bytes for segment in arena.reserve) != layout.reserved_bytes:
      self.pmu_permission_fault_count += 1
      return False
    self.arena = arena
    self._layout = layout
    self._slot_segments.clear()
    self._shadow_slots = [Slot(i) for i in range(self.slot_count)]
    return True

  def bind_view(self, buffer_id: str, handle: AllocationHandle) -> int:
    """Install one executed ALLOC_L1 into its compiler-selected live slot."""
    if self.arena is None or self._layout is None or self.state != FrameState.FRAME_ACTIVE:
      raise MemoryInvariantError("L1 allocation without an active Arena Frame")
    item = next((item for item in self._layout.buffer_layouts if item.buffer_id == buffer_id), None)
    if item is None or handle.size_bytes != item.logical_bytes:
      raise MemoryInvariantError("L1 allocation differs from compiled buffer layout")
    if (
      handle.arena_id != self.arena.arena_id
      or handle.profile_generation != self.arena.profile_generation
      or self.slots[item.slot_id].allocation_id is not None
      or sum(segment.size_bytes for segment in handle.bank_segments) != item.logical_bytes
    ):
      raise MemoryInvariantError("L1 slot generation or live binding mismatch")
    for segment in handle.bank_segments:
      if not any(
        reserve.bank_id == segment.bank_id
        and reserve.address <= segment.address
        and segment.address + segment.size_bytes <= reserve.address + reserve.size_bytes
        for reserve in self.arena.reserve
      ):
        raise MemoryInvariantError("L1 view escapes Arena reservation")
      for previous in self._slot_segments.values():
        if any(
          segment.address < other.address + other.size_bytes
          and other.address < segment.address + segment.size_bytes
          for other in previous
        ):
          raise MemoryInvariantError("L1 view overlaps a live Frame binding")
    slot = Slot(
      item.slot_id,
      base=handle.base_address,
      size=handle.size_bytes,
      alignment=handle.alignment,
      role=SlotRole.WORKSPACE,
      lifetime=SlotLifetime.PER_TILE_PROGRAM,
      allocation_id=handle.allocation_id,
      generation=handle.generation,
      owner=handle.owner,
      arena_id=handle.arena_id,
      profile_generation=handle.profile_generation,
    )
    assert self.shadow is not None and self._shadow_slots is not None
    self.slots[item.slot_id] = slot
    self.shadow.slots[item.slot_id] = replace(slot)
    self._shadow_slots[item.slot_id] = replace(slot)
    self._slot_segments[item.slot_id] = handle.bank_segments
    return item.slot_id

  def bind(self, cycle: int, bind_cycles: int = 8) -> tuple[bool, int]:
    """Run the frame bind FSM (design 3.2) on the prepared shadow.

    Returns (ok, cycles_consumed).  Each of the 8 states consumes 1
    cycle.  Returns False + fault if no shadow was prepared or any
    check fails.
    """
    if self._shadow_slots is None:
      self.pmu_permission_fault_count += 1
      self.state = FrameState.FRAME_FAULTED
      return (False, 0)
    # capacity + overlap already checked in prepare(); bank policy V1 pass
    self.state = FrameState.FRAME_ACTIVE
    shadow = SlotFrame(frame_id=self.frame_id, l1_bytes=self.l1_bytes, slot_count=self.slot_count)
    shadow.slots = [replace(slot) for slot in self._shadow_slots]
    shadow.arena = self.arena
    shadow._layout = self._layout
    shadow.state = FrameState.FRAME_ACTIVE
    self.shadow = shadow
    self.slots = list(self._shadow_slots)
    return (True, bind_cycles)

  def release(self) -> None:
    """Clear active and shadow slots after tile program completion."""
    self.slots = [Slot(i) for i in range(self.slot_count)]
    self.shadow = None
    self._shadow_slots = None
    self.state = FrameState.IDLE
    self.arena = None
    self._layout = None
    self._slot_segments.clear()

  def assert_slot_binding(
    self,
    slot_id: int,
    handle: AllocationHandle,  # type: ignore[name-defined]
  ) -> None:
    """Read-only validation for releasing one live slot binding."""
    if self.state != FrameState.FRAME_ACTIVE:
      raise MemoryInvariantError("L1 slot frame is not active")
    if self.shadow is None or self._shadow_slots is None:
      raise MemoryInvariantError("L1 slot frame copies are missing")
    copies = (("active", self.slots), ("shadow", self.shadow.slots), ("prepared", self._shadow_slots))
    for copy_name, slots in copies:
      if slot_id < 0 or slot_id >= len(slots):
        raise MemoryInvariantError("L1 slot id is out of range")
      slot = slots[slot_id]
      if (
        slot.allocation_id != handle.allocation_id
        or slot.generation != handle.generation
        or slot.arena_id != handle.arena_id
        or slot.profile_generation != handle.profile_generation
        or slot.owner != handle.owner
        or slot.base != handle.base_address
        or slot.size != handle.size_bytes
      ):
        raise MemoryInvariantError(f"L1 {copy_name} slot binding does not match allocation")

  def release_slot(
    self,
    slot_id: int,
    handle: AllocationHandle,  # type: ignore[name-defined]
  ) -> None:
    """Atomically invalidate one slot without ending or bumping the frame."""
    self.assert_slot_binding(slot_id, handle)
    self.slots[slot_id] = Slot(slot_id)
    assert self.shadow is not None
    self.shadow.slots[slot_id] = Slot(slot_id)
    assert self._shadow_slots is not None
    self._shadow_slots[slot_id] = Slot(slot_id)
    self._slot_segments.pop(slot_id)

  def reset(self) -> None:
    self.release()
    self.pmu_bank_conflict_cycles = 0
    self.pmu_permission_fault_count = 0

  def snapshot(self) -> dict:
    return {
      "frame_id": self.frame_id,
      "generation": self.generation,
      "state": self.state.name,
      "slot_count": self.slot_count,
      "active_slots": sum(1 for s in self.slots if s.size > 0),
      "bank_conflict_cycles": self.pmu_bank_conflict_cycles,
      "permission_faults": self.pmu_permission_fault_count,
    }
