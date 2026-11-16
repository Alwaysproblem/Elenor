"""C029 reference observations; RTL remains the only cosim L1 execution owner.

ArenaPool.commit_arena is used here to make the exclusive physical reservation
corresponding to LOCAL_HOLD. Frame.bind remains a separate action. These tests
neither implement the RTL authorization protocol nor add a Python L1 ledger.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline_validator.compiler.resources import conservative_arena_bytes, layout_buffers
from pipeline_validator.config import HardwareConfig
from pipeline_validator.execution_ir import ExecL1Buffer, GridInstanceId, TaskIdentity
from pipeline_validator.memory import (
  AdmissionFailure,
  AdmissionFailureKind,
  MemoryInvariantError,
  SlotFrame,
)
from pipeline_validator.memory.arena import ArenaHandle, ArenaPool
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.memory.transfer import ResolvedMemoryView
from pipeline_validator.profiles import ArenaLayout, build_registry


def _reserve(pool: ArenaPool, layout: ArenaLayout, ordinal: int) -> ArenaHandle:
  owner = TaskIdentity(GridInstanceId("c029", 0, 0, 0), ordinal)
  plan = pool.plan_arena(owner, layout)
  assert not isinstance(plan, AdmissionFailure)
  arena = pool.commit_arena(plan, ordinal)
  pool.bind_task_metadata(arena, "c029-role", 0)
  return arena


def _frame(hw: HardwareConfig) -> SlotFrame:
  return SlotFrame(l1_bytes=hw.tile_l1_bytes, slot_count=hw.frame_slot_capacity)


def test_held_physical_reservation_excludes_peer_before_frame_bind() -> None:
  hw = HardwareConfig()
  profile = build_registry(hw).profile("l1", 0)
  pool = ArenaPool(profile, pool_id=0, tile_id=0)
  capacity = profile.user_spm_per_bank * profile.banks
  layout = layout_buffers((), profile, capacity, slot_capacity=hw.frame_slot_capacity)
  arena = _reserve(pool, layout, 0)
  frame = _frame(hw)
  assert frame.prepare(arena, layout)
  assert not frame.is_available
  assert frame.shadow is None  # Hold is not an executable Frame.
  occupied = pool.snapshot()
  assert all(
    bank["allocated_bytes"] == profile.user_spm_per_bank and bank["free_bytes"] == 0
    for bank in occupied["per_bank_occupancy"]
  )
  peer = pool.plan_arena(TaskIdentity(GridInstanceId("c029", 0, 0, 0), 1), layout)
  assert isinstance(peer, AdmissionFailure)
  assert peer.kind is AdmissionFailureKind.TEMPORARY_CAPACITY
  assert pool.snapshot() == occupied

  # Preauthorization compensation returns the entire reservation and Frame.
  frame.release()
  assert pool.retire_arena(arena, 1)
  assert frame.is_available
  assert pool.snapshot()["free_bytes"] == capacity
  next_arena = _reserve(pool, layout, 1)
  assert frame.prepare(next_arena, layout)
  assert frame.bind(2, hw.frame_bind_cycles)[0]
  assert frame.generation == next_arena.allocation_generation
  assert pool.retire_arena(next_arena, 3)
  frame.release()
  assert pool.snapshot()["free_bytes"] == capacity


@pytest.mark.parametrize("domain", ("generation", "profile_generation"))
def test_reused_frame_rejects_old_view_and_independent_generation_domains(domain: str) -> None:
  hw = HardwareConfig()
  profile = build_registry(hw).profile("l1", 0)
  pool = ArenaPool(profile, pool_id=0, tile_id=0)
  spec = ExecL1Buffer("buffer", (64,), "i8", 1, 64, 64)
  reserved = conservative_arena_bytes([(spec.bytes, spec.alignment)], profile)
  layout = layout_buffers(
    (spec,), profile, reserved, lifetimes={"buffer": (0, 1)}, slot_capacity=hw.frame_slot_capacity
  )
  frame = _frame(hw)
  first = _reserve(pool, layout, 0)
  assert frame.prepare(first, layout)
  assert frame.bind(0, hw.frame_bind_cycles)[0]
  old_view = pool.bind_view(first, "buffer", 1)
  slot = frame.bind_view("buffer", old_view)
  store = ByteStore()
  old_bytes = ResolvedMemoryView(
    old_view, 0, old_view.size_bytes, old_view.base_address,
    pool.resolve_segments(old_view, 0, old_view.size_bytes),
  )
  payload = bytes(range(old_view.size_bytes))
  store.write_view(old_bytes, payload)
  assert store.read_view(old_bytes) == payload
  assert pool.invalidate_view(old_view, old_view.owner, 2)
  frame.release_slot(slot, old_view)
  assert pool.retire_arena(first, 3)
  frame.release()

  second = _reserve(pool, layout, 1)
  assert frame.prepare(second, layout)
  assert frame.bind(4, hw.frame_bind_cycles)[0]
  current = pool.bind_view(second, "buffer", 5)
  assert frame.bind_view("buffer", current) == slot
  assert current.base_address == old_view.base_address  # Physical-address ABA.
  assert second.allocation_generation != first.allocation_generation
  reused_bytes = ResolvedMemoryView(
    current, 0, current.size_bytes, current.base_address,
    pool.resolve_segments(current, 0, current.size_bytes),
  )
  _uninitialized, validity = store.read_view_relaxed(reused_bytes)
  assert not any(validity)
  replacement = payload[::-1]
  store.write_view(reused_bytes, replacement)
  assert store.read_view(reused_bytes) == replacement
  before_frame, before_pool = frame.snapshot(), pool.snapshot()
  forged = replace(current, **{domain: getattr(current, domain) + 1})
  for stale in (old_view, forged):
    with pytest.raises(MemoryInvariantError):
      frame.release_slot(slot, stale)
    assert frame.snapshot() == before_frame
    assert pool.snapshot() == before_pool
    assert frame.slots[slot].allocation_id == current.allocation_id
  assert frame.generation == second.allocation_generation
  assert pool.invalidate_view(current, current.owner, 6)
  frame.release_slot(slot, current)
  assert pool.snapshot()["arena_reserved_bytes"] == before_pool["arena_reserved_bytes"]
  assert pool.retire_arena(second, 7)
  frame.release()
  assert pool.snapshot()["arena_reserved_bytes"] == 0


def test_empty_frame_keeps_arena_identity_until_retirement() -> None:
  hw = HardwareConfig()
  profile = build_registry(hw).profile("l1", 0)
  pool = ArenaPool(profile, pool_id=0, tile_id=0)
  layout = layout_buffers((), profile, 0, slot_capacity=hw.frame_slot_capacity)
  arena = _reserve(pool, layout, 0)
  frame = _frame(hw)
  assert frame.prepare(arena, layout)
  assert frame.bind(0, hw.frame_bind_cycles)[0]
  assert frame.generation == arena.allocation_generation
  assert frame.snapshot()["active_slots"] == 0
  assert not frame.is_available
  assert not frame.prepare(arena, layout)
  assert frame.generation == arena.allocation_generation
  assert pool.retire_arena(arena, 1)
  frame.release()
  assert frame.is_available
  assert frame.generation is None
  assert pool.snapshot()["live_arenas"] == 0


def test_more_buffers_than_slots_reuse_one_slot_with_different_shapes() -> None:
  hw = HardwareConfig()
  profile = build_registry(hw).profile("l1", 0)
  pool = ArenaPool(profile, pool_id=0, tile_id=0)
  sizes = (profile.alignment, profile.alignment * profile.banks)
  buffers = tuple(
    ExecL1Buffer(f"buffer{i}", (sizes[i % 2],), "i8", 1, profile.alignment, sizes[i % 2])
    for i in range(hw.frame_slot_capacity + 3)
  )
  reserved = conservative_arena_bytes([(max(sizes), profile.alignment)], profile)
  layout = layout_buffers(
    buffers,
    profile,
    reserved,
    lifetimes={buffer.name: (2 * i, 2 * i + 1) for i, buffer in enumerate(buffers)},
    slot_capacity=hw.frame_slot_capacity,
  )
  arena = _reserve(pool, layout, 0)
  frame = _frame(hw)
  assert frame.prepare(arena, layout)
  assert frame.bind(0, hw.frame_bind_cycles)[0]
  slot = layout.buffer_layouts[0].slot_id
  previous = None
  allocation_ids = set()
  for cycle, buffer in enumerate(buffers, start=1):
    view = pool.bind_view(arena, buffer.name, cycle)
    assert view.allocation_id not in allocation_ids
    allocation_ids.add(view.allocation_id)
    assert frame.bind_view(buffer.name, view) == slot
    assert frame.slots[slot].size == buffer.bytes
    if previous is not None:
      with pytest.raises(MemoryInvariantError):
        frame.assert_slot_binding(slot, previous)
      assert frame.slots[slot].allocation_id == view.allocation_id
    assert pool.invalidate_view(view, view.owner, cycle)
    frame.release_slot(slot, view)
    assert pool.snapshot()["arena_reserved_bytes"] == arena.reserved_bytes
    previous = view
  assert frame.generation == arena.allocation_generation
  assert pool.retire_arena(arena, len(buffers) + 1)
  frame.release()
  assert pool.snapshot()["live_arenas"] == 0
