"""L1 Slot Frame — 16-slot binding + shadow + generation gate
(Slot Frame design 3-5, review P0-2).

Models the L1 SRAM binary binding contract: fixed slot ABI + variable
Tile Frame.  Frame bind FSM (3.2), descriptor patch FSM (3.3), and slot
lifecycle (3.4).  Bank policy enforcement (5.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


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
  """elenor_tile_slot_v0_t (Slot Frame design 4.1)."""
  slot_id: int
  base: int = 0
  size: int = 0
  layout: int = 0
  role: int = 0
  alignment: int = 0
  bank_policy: int = 0
  lifetime: SlotLifetime = SlotLifetime.PER_COMMAND
  owner: int = 0
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
    self.pmu_bank_conflict_cycles: int = 0
    self.pmu_permission_fault_count: int = 0

  def capacity_ok(self) -> bool:
    """16-slot total must fit in L1 (design 3.1)."""
    total = sum(s.size for s in self.slots if s.size > 0)
    return total <= self.l1_bytes

  def overlap_ok(self) -> bool:
    """Slot ranges must not overlap (design 3.2 CHECK_OVERLAP_ALIGNMENT)."""
    ranges = [(s.base, s.base + s.size) for s in self.slots if s.size > 0]
    ranges.sort()
    for i in range(1, len(ranges)):
      if ranges[i][0] < ranges[i - 1][1]:
        return False
    return True

  def bank_policy_ok(self) -> bool:
    """Check NO_HOT_CONFLICT policy (design 5.4).  V1: pass if no slot
    declares NO_HOT_CONFLICT on the same bank as program/accumulator."""
    # V1 simplification: accept all (bank_policy encoding unfrozen)
    return True

  def bind(self, cycle: int, bind_cycles: int = 8) -> tuple[bool, int]:
    """Run the frame bind FSM (design 3.2).  Returns (ok, cycles_consumed).

    Each of the 8 states consumes 1 cycle (FETCH -> VALIDATE_ABI ->
    VALIDATE_SLOT_TABLE -> CHECK_OVERLAP -> CHECK_BANK -> INSTALL_SHADOW ->
    FRAME_ACTIVE).  Returns False + fault if any check fails.
    """
    if not self.capacity_ok():
      self.state = FrameState.FRAME_FAULTED
      self.pmu_permission_fault_count += 1
      return (False, 1)
    if not self.overlap_ok():
      self.state = FrameState.FRAME_FAULTED
      self.pmu_permission_fault_count += 1
      return (False, 1)
    if not self.bank_policy_ok():
      self.state = FrameState.FRAME_FAULTED
      self.pmu_permission_fault_count += 1
      return (False, 1)
    # success: consume bind_cycles (8 FSM states), install shadow
    self.state = FrameState.FRAME_ACTIVE
    shadow = SlotFrame(frame_id=self.frame_id,
                       generation=self.generation,
                       l1_bytes=self.l1_bytes,
                       slot_count=self.slot_count)
    shadow.slots = [Slot(s.slot_id, s.base, s.size, s.layout, s.role,
                         s.alignment, s.bank_policy, s.lifetime, s.owner,
                         s.flags) for s in self.slots]
    shadow.state = FrameState.FRAME_ACTIVE
    self.shadow = shadow
    return (True, bind_cycles)

  def check_generation(self, expected_gen: int) -> bool:
    """Warm-launch generation gate (design 5.2).  Mismatch -> fault."""
    return self.generation == expected_gen

  def bump_generation(self) -> int:
    self.generation += 1
    return self.generation

  def invalidate_desc_cache(self) -> None:
    """Descriptor cache invalidate (design 5.2 warm path)."""
    # V1: no-op (descriptor cache coherence unfrozen)
    pass

  def reset(self) -> None:
    self.slots = [Slot(i) for i in range(self.slot_count)]
    self.shadow = None
    self.state = FrameState.IDLE
    self.pmu_bank_conflict_cycles = 0
    self.pmu_permission_fault_count = 0

  def snapshot(self) -> dict:
    return {
      "frame_id": self.frame_id,
      "generation": self.generation,
      "state": self.state.name,
      "capacity_ok": self.capacity_ok(),
      "overlap_ok": self.overlap_ok(),
      "bank_conflict_cycles": self.pmu_bank_conflict_cycles,
      "permission_faults": self.pmu_permission_fault_count,
    }
