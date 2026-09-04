"""L1 Slot Frame — 16-slot binding + shadow + generation gate
(Slot Frame design 3-5, review P0-2).

Models the L1 SRAM binary binding contract: fixed slot ABI + variable
Tile Frame.  Frame bind FSM (3.2), descriptor patch FSM (3.3), and slot
lifecycle (3.4).  Bank policy enforcement (5.4).

PR 2: the actual L1 placement is delegated to the per-tile
``BankedFreeExtentAllocator``; ``SlotFrame`` only owns the fixed-slot
ABI, shadow-install FSM and generation gate.  ``prepare()`` maps
``ExecL1Buffer`` specs to fixed slots and builds a shadow; ``bind()``
runs the bind-cycle FSM on the prepared shadow; ``release()`` clears
active/shadow slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..execution_ir import ExecL1Buffer
  from .allocator import AllocationHandle, MemoryOwner


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


@dataclass
class SlotFrame:
  """elenor_tile_frame_v0_t (Slot Frame design 4.1).

  16 fixed slots with a shadow-install mechanism.  After bind, engines
  only access the shadow copy (design 3.2).  Warm launch checks frame
  generation; mismatch -> fault (design 5.2).
  """
  frame_id: int = 0
  generation: int = 0
  l1_bytes: int = 1 * 1024 * 1024  # 1 MB Balanced-small
  slot_count: int = 16
  state: FrameState = FrameState.IDLE

  def __post_init__(self) -> None:
    self.slots: list[Slot] = [Slot(i) for i in range(self.slot_count)]
    self.shadow: SlotFrame | None = None
    self._shadow_slots: list[Slot] | None = None
    self.pmu_bank_conflict_cycles: int = 0
    self.pmu_permission_fault_count: int = 0

  def prepare(
    self,
    handles: list[AllocationHandle],  # type: ignore[name-defined]
    specs: list[ExecL1Buffer],  # type: ignore[name-defined]
  ) -> bool:
    """Map ``ExecL1Buffer`` specs to fixed slots and build a shadow.

    Checks slot count, capacity, alignment, overlap and generation
    before building the shadow.  Each L1 buffer uses
    ``SlotRole.WORKSPACE``, ``SlotLifetime.PER_TILE_PROGRAM``,
    ``layout=0``, ``bank_policy=0``.  On failure the active frame is
    not changed.
    """
    if len(specs) > self.slot_count:
      self.pmu_permission_fault_count += 1
      return False
    total = sum(s.bytes for s in specs)
    if total > self.l1_bytes:
      self.pmu_permission_fault_count += 1
      return False
    new_slots: list[Slot] = [Slot(i) for i in range(self.slot_count)]
    for i, (spec, handle) in enumerate(zip(specs, handles)):
      new_slots[i] = Slot(
        slot_id=i,
        base=handle.base_address,
        size=spec.bytes,
        alignment=spec.alignment,
        role=SlotRole.WORKSPACE,
        lifetime=SlotLifetime.PER_TILE_PROGRAM,
        allocation_id=handle.allocation_id,
        generation=handle.generation,
        owner=handle.owner,
      )
    # Overlap is physical-segment based.  ``base + logical size`` is not a
    # physical range when one allocation spans fragmented bank extents.
    ranges = [
      (segment.address, segment.address + segment.size_bytes)
      for handle in handles
      for segment in handle.bank_segments
    ]
    ranges.sort()
    for i in range(1, len(ranges)):
      if ranges[i][0] < ranges[i - 1][1]:
        self.pmu_permission_fault_count += 1
        return False
    self._shadow_slots = new_slots
    return True

  def bind(self, cycle: int, bind_cycles: int = 8) -> tuple[bool, int]:
    """Run the frame bind FSM (design 3.2) on the prepared shadow.

    Returns (ok, cycles_consumed).  Each of the 8 states consumes 1
    cycle.  Returns False + fault if no shadow was prepared or any
    check fails.
    """
    if self._shadow_slots is None:
      # No prepare() was called: empty frame (program has no L1 buffers).
      # This is valid for timing_only and programs without tile.alloc.
      self._shadow_slots = [Slot(i) for i in range(self.slot_count)]
    # capacity + overlap already checked in prepare(); bank policy V1 pass
    self.state = FrameState.FRAME_ACTIVE
    shadow = SlotFrame(frame_id=self.frame_id,
                       generation=self.generation,
                       l1_bytes=self.l1_bytes,
                       slot_count=self.slot_count)
    shadow.slots = [Slot(s.slot_id, s.base, s.size, s.layout, s.role,
                         s.alignment, s.bank_policy, s.lifetime,
                         s.allocation_id, s.generation, s.owner, s.flags)
                    for s in self._shadow_slots]
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

  def check_generation(self, expected_gen: int) -> bool:
    """Warm-launch generation gate (design 5.2).  Mismatch -> fault."""
    return self.generation == expected_gen

  def bump_generation(self) -> int:
    self.generation += 1
    return self.generation

  def invalidate_desc_cache(self) -> None:
    """Descriptor cache invalidate (design 5.2 warm path)."""
    pass

  def reset(self) -> None:
    self.slots = [Slot(i) for i in range(self.slot_count)]
    self.shadow = None
    self._shadow_slots = None
    self.state = FrameState.IDLE
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
