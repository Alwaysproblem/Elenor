from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline_validator.compiler.resources import (
  check_layout_capacity,
  conservative_arena_bytes,
  layout_buffers,
)
from pipeline_validator.config import HardwareConfig
from pipeline_validator.execution_ir import ExecL2Buffer, GridInstanceId, TaskIdentity
from pipeline_validator.immutable import digest
from pipeline_validator.memory import AdmissionFailure, AdmissionFailureKind, MemoryInvariantError
from pipeline_validator.memory.allocator import ContextBufferOwner
from pipeline_validator.memory.arena import WAIT_CAPACITY, WAIT_FRAGMENTATION, ArenaPool, RootInvocation
from pipeline_validator.profiles import (
  ArenaLayout,
  MemoryProfile,
  ProfileBytes,
  ProfileLevelSource,
  ProfileReconfigDesc,
  ProfileSourceConfig,
  SourceRef,
  build_registry,
  parse_memory_target,
  validate_allowed,
)
from pipeline_validator.tile_group import TileGroup


def _source_with(hw: HardwareConfig, *, level: str, mode: int, spm: int, cache: int) -> ProfileSourceConfig:
  source = hw.profile_source
  table = dict(getattr(source, level).modes)
  table[mode] = ProfileBytes(spm, cache)
  if level == "l1":
    return replace(source, l1=ProfileLevelSource(table))
  if level == "l2":
    return replace(source, l2=ProfileLevelSource(table))
  raise ValueError("unknown profile layer")


def _profile(*, level: str = "l2", banks: int = 2, bank_bytes: int = 512) -> MemoryProfile:
  return MemoryProfile(
    level=level,
    mode=0,
    bank_bytes=bank_bytes,
    banks=banks,
    pools=1,
    spm_bytes_per_bank=bank_bytes,
    cache_bytes_per_bank=0,
    system_reserved_spm_per_bank=0,
    alignment=64,
    spm_mapping_id="striped_arena_v0",
    cache_org_id="profiled_lru_v0",
    cache_write_policy="read_only",
    maintenance_caps=("invalidate_range", "bypass"),
  )


def _empty_layout(profile: MemoryProfile, per_bank: int) -> ArenaLayout:
  return layout_buffers((), profile, per_bank * profile.banks)


class TestProfileRegistry:
  def test_default_registry_conserves_each_bank_and_reserves_system_spm(self):
    hw = HardwareConfig()
    registry = build_registry(hw)
    for level in ("l1", "l2"):
      capacity = hw.tile_l1_bytes if level == "l1" else hw.group_sram_bytes
      banks = hw.tile_l1_banks if level == "l1" else hw.group_sram_banks
      for profile in getattr(registry, level).values():
        assert profile.bank_bytes == capacity // banks
        assert profile.spm_bytes_per_bank + profile.cache_bytes_per_bank == profile.bank_bytes
        assert profile.system_reserved_spm_per_bank <= profile.spm_bytes_per_bank
        assert profile.user_spm_per_bank == (
          profile.spm_bytes_per_bank - profile.system_reserved_spm_per_bank
        )
    assert registry.registry_hash == build_registry(hw).registry_hash

  @pytest.mark.parametrize(
    ("spm", "cache"),
    [
      (49152, 8192),  # does not conserve one 64-KiB L1 bank
      (65535, 1),  # violates the 64-byte partition/cache-line quantum
      (-64, 65600),
      (True, 65535),
    ],
    ids=["conservation", "alignment", "negative", "bool"],
  )
  def test_invalid_profile_bytes_are_rejected(self, spm, cache):
    hw = HardwareConfig()
    if type(spm) is not int or spm < 0:
      with pytest.raises(ValueError):
        ProfileBytes(spm, cache)
      return
    source = _source_with(hw, level="l1", mode=1, spm=spm, cache=cache)
    with pytest.raises(ValueError):
      build_registry(replace(hw, profile_source=source))

  def test_duplicate_effective_mode_and_missing_baseline_are_rejected(self):
    hw = HardwareConfig()
    mode0 = hw.profile_source.l1.modes[0]
    source = _source_with(
      hw, level="l1", mode=1, spm=mode0.spm_bytes_per_bank, cache=mode0.cache_bytes_per_bank
    )
    with pytest.raises(ValueError):
      build_registry(replace(hw, profile_source=source))
    with pytest.raises(ValueError):
      validate_allowed((1, 2), baseline=0)

  def test_explicit_target_layer_requires_system_reservation(self):
    hw = HardwareConfig()
    with pytest.raises(ValueError):
      parse_memory_target({"l1": {"reset_mode": 0}}, defaults=hw.memory_target)

  def test_registry_hash_changes_when_profile_bytes_change(self):
    hw = HardwareConfig()
    changed = _source_with(hw, level="l1", mode=1, spm=53248, cache=12288)
    assert (
      build_registry(replace(hw, profile_source=changed)).registry_hash != build_registry(hw).registry_hash
    )


class TestStaticResourceBudgets:
  def test_t04_r_times_per_bank_envelope_is_a_permanent_error(self):
    profile = _profile(banks=16, bank_bytes=22528)
    layout = _empty_layout(profile, 6 * 1024)
    with pytest.raises(ValueError):
      check_layout_capacity(layout, profile, copies=4)

  def test_t06_one_23kib_bank_is_rejected_even_when_total_is_small(self):
    profile = _profile(banks=16, bank_bytes=22 * 1024)
    per_bank = (23 * 1024,) + (1024,) * 15
    assert sum(per_bank) < profile.user_spm_bytes
    with pytest.raises(ValueError):
      layout = ArenaLayout(64, 64, sum(per_bank), per_bank, (), "")
      layout = replace(layout, layout_hash=digest(layout))
      check_layout_capacity(layout, profile)

  def test_t05_fragmentation_is_distinct_from_capacity_and_has_no_side_effects(self):
    profile = _profile()
    pool = ArenaPool(profile)
    unit = _empty_layout(profile, 128)
    handles = []
    for index in range(4):
      plan = pool.plan_arena(RootInvocation(f"root{index}", 0), unit)
      assert not isinstance(plan, AdmissionFailure)
      handles.append(pool.commit_arena(plan, index))
    assert pool.retire_arena(handles[0], 10)
    assert pool.retire_arena(handles[2], 11)
    before = pool.snapshot()
    fragmented = pool.plan_arena(RootInvocation("fragmented", 0), _empty_layout(profile, 192))
    assert isinstance(fragmented, AdmissionFailure)
    assert fragmented.kind is AdmissionFailureKind.TEMPORARY_CAPACITY
    assert fragmented.reason.startswith(WAIT_FRAGMENTATION)
    assert pool.snapshot() == before

  def test_t07_two_plans_for_last_arena_allow_only_one_atomic_commit(self):
    profile = _profile()
    pool = ArenaPool(profile)
    full = _empty_layout(profile, profile.user_spm_per_bank)
    left = pool.plan_arena(RootInvocation("left", 0), full)
    right = pool.plan_arena(RootInvocation("right", 0), full)
    assert not isinstance(left, AdmissionFailure)
    assert not isinstance(right, AdmissionFailure)
    handle = pool.commit_arena(left, 0)
    with pytest.raises(MemoryInvariantError, match="stale arena plan"):
      pool.commit_arena(right, 0)
    assert pool.snapshot()["live_arenas"] == 1
    assert pool.retire_arena(handle, 1)
    assert pool.snapshot()["live_arenas"] == 0

  def test_local_view_invalidation_never_changes_global_free_map(self):
    from pipeline_validator.execution_ir import ExecL1Buffer

    profile = _profile(level="l1", banks=2, bank_bytes=512)
    pool = ArenaPool(profile, pool_id=0, tile_id=0)
    spec = ExecL1Buffer("tmp", (64,), "i8", 1, 64, 64)
    layout = layout_buffers((spec,), profile, 128, lifetimes={"tmp": (0, 1)})
    owner = TaskIdentity(GridInstanceId("ctx", 0, 0, 0), 0)
    plan = pool.plan_arena(owner, layout)
    assert not isinstance(plan, AdmissionFailure)
    arena = pool.commit_arena(plan, 0)
    pool.bind_task_metadata(arena, "role", 0)
    view = pool.bind_view(arena, "tmp", 0)
    before = pool.snapshot()
    assert pool.invalidate_view(view, view.owner, 1)
    after = pool.snapshot()
    assert after["arena_reserved_bytes"] == before["arena_reserved_bytes"]
    assert after["free_bytes"] == before["free_bytes"]
    assert after["live_views"] == 0
    assert pool.retire_arena(arena, 2)


class TestL2BackingLifecycle:
  """plan/01 §2/§6.2: physical padded backings, exact units, early free."""

  def _pool(self, banks: int = 2, bank_bytes: int = 512) -> ArenaPool:
    return ArenaPool(_profile(banks=banks, bank_bytes=bank_bytes))

  @staticmethod
  def _buffer(slot: str, logical: int) -> ExecL2Buffer:
    return ExecL2Buffer(slot, (logical,), "i8", "inout", 1, 64, logical)

  def _committed(self, pool: ArenaPool, buffers: tuple[ExecL2Buffer, ...], cycle: int):
    reserved = conservative_arena_bytes(
      [(buffer.bytes, buffer.alignment) for buffer in buffers], pool.profile
    )
    layout = layout_buffers(buffers, pool.profile, reserved, lifetimes=None, slot_capacity=len(buffers))
    owner = RootInvocation("ctx", 0)
    plan = pool.plan_arena(owner, layout)
    assert not isinstance(plan, AdmissionFailure)
    return pool.commit_arena(plan, cycle), owner

  def test_tail_padding_returns_with_its_backing_not_its_valid_bytes(self):

    pool = self._pool()
    # 100 logical bytes pad to a full 128-byte-per-bank stripe round.
    empty_free = pool.snapshot()["free_bytes"]
    arena, _owner = self._committed(pool, (self._buffer("buf", 100),), 0)
    view = pool.bind_view(arena, "buf", 1)
    assert view.backing_id
    assert pool.invalidate_view(view, ContextBufferOwner("ctx", 0, "buf"), 2)
    after = pool.snapshot()
    # The complete padded span (including tail padding) returned to the map.
    assert after["free_bytes"] == empty_free
    assert after["live_backings"] == 0
    assert after["arena_reserved_bytes"] == 0
    assert after["pool_version"] > 1

  def test_slack_is_root_held_until_retirement_and_backing_free_never_touches_it(self):
    pool = self._pool()
    # Reserve one extra stripe round beyond the buffers: an internal gap plus
    # tail slack that only root retirement may return.
    buffers = (self._buffer("buf", 64),)
    padded_total = 128
    reserved = conservative_arena_bytes(
      [(buffer.bytes, buffer.alignment) for buffer in buffers], pool.profile
    ) + 128
    layout = layout_buffers(buffers, pool.profile, reserved, lifetimes=None, slot_capacity=1)
    owner = RootInvocation("ctx", 0)
    plan = pool.plan_arena(owner, layout)
    assert not isinstance(plan, AdmissionFailure)
    arena = pool.commit_arena(plan, 0)
    assert arena.reserved_bytes == reserved == padded_total + 128
    view = pool.bind_view(arena, "buf", 1)
    assert pool.invalidate_view(view, view.owner, 2)
    held = pool.snapshot()
    assert held["arena_reserved_bytes"] == 128  # slack only
    assert held["free_bytes"] == held["user_spm_capacity_bytes"] - held["system_reserved_bytes"] - 128
    assert pool.retire_arena(arena, 3)
    after = pool.snapshot()
    assert after["arena_reserved_bytes"] == 0
    assert after["free_bytes"] == after["user_spm_capacity_bytes"] - after["system_reserved_bytes"]

  def test_released_extent_is_reallocatable_at_original_capacity(self):
    pool = self._pool()
    full = _empty_layout(pool.profile, pool.profile.user_spm_per_bank)
    plan = pool.plan_arena(RootInvocation("a", 0), full)
    assert not isinstance(plan, AdmissionFailure)
    arena = pool.commit_arena(plan, 0)
    buffers = tuple(self._buffer(f"buf{index}", 128) for index in range(2))
    reserved = conservative_arena_bytes(
      [(buffer.bytes, buffer.alignment) for buffer in buffers], pool.profile
    )
    layout = layout_buffers(buffers, pool.profile, reserved, lifetimes=None, slot_capacity=2)
    owner = RootInvocation("b", 0)
    nested_plan = pool.plan_arena(owner, layout)
    assert isinstance(nested_plan, AdmissionFailure)
    assert nested_plan.kind is AdmissionFailureKind.TEMPORARY_CAPACITY
    assert nested_plan.reason.startswith(WAIT_CAPACITY)
    before = pool.snapshot()
    assert pool.retire_arena(arena, 1)
    after = pool.snapshot()
    assert after["free_bytes"] > before["free_bytes"]
    retried = pool.plan_arena(owner, layout)
    assert not isinstance(retried, AdmissionFailure)

  def test_double_release_rebind_and_wrong_owner_are_rejected_without_side_effects(self):
    pool = self._pool()
    arena, _owner = self._committed(pool, (self._buffer("buf", 64),), 0)
    view = pool.bind_view(arena, "buf", 1)
    with pytest.raises(MemoryInvariantError, match="wrong-owner release"):
      pool.invalidate_view(view, ContextBufferOwner("other", 0, "buf"), 2)
    assert pool.snapshot()["live_backings"] == 1
    assert pool.invalidate_view(view, view.owner, 2)
    before = pool.snapshot()
    with pytest.raises(MemoryInvariantError, match="double release"):
      pool.invalidate_view(view, view.owner, 3)
    with pytest.raises(MemoryInvariantError, match="already bound"):
      pool.bind_view(arena, "buf", 3)
    assert pool.snapshot() == before

  def test_pending_view_keeps_backing_until_last_pin_drains(self):
    pool = self._pool()
    arena, _owner = self._committed(pool, (self._buffer("buf", 64),), 0)
    committed_free = pool.snapshot()["free_bytes"]
    view = pool.bind_view(arena, "buf", 1)
    pool.pin(view, "consumer:0")
    assert not pool.invalidate_view(view, view.owner, 2)
    held = pool.snapshot()
    assert held["live_backings"] == 1
    assert held["free_bytes"] == committed_free
    assert pool.unpin(view, "consumer:0", 3)
    after = pool.snapshot()
    assert after["live_backings"] == 0
    assert after["free_bytes"] == held["user_spm_capacity_bytes"] - after["system_reserved_bytes"]

  def test_pending_inflight_reference_releases_only_at_end_inflight(self):
    pool = self._pool()
    arena, _owner = self._committed(pool, (self._buffer("buf", 64),), 0)
    view = pool.bind_view(arena, "buf", 1)
    # A transfer accepted before the release keeps the backing pending.
    pool.begin_inflight(view, "txn:1")
    assert not pool.invalidate_view(view, view.owner, 2)
    held = pool.snapshot()
    assert held["live_backings"] == 1
    assert pool.end_inflight(view, "txn:1", 3)
    after = pool.snapshot()
    assert after["live_backings"] == 0
    assert after["pool_version"] == held["pool_version"] + 1

  def test_exact_units_tile_reserve_and_conserve_bytes(self):
    pool = self._pool()
    buffers = (self._buffer("buf_a", 100), self._buffer("buf_b", 64))
    arena, _owner = self._committed(pool, buffers, 0)
    snapshot = pool.snapshot()
    for bank in snapshot["per_bank_occupancy"]:
      assert bank["allocated_bytes"] + bank["free_bytes"] == pool.profile.user_spm_per_bank
      assert bank["padding_bytes"] >= 0
    # The pool physically holds the whole reserve: padded spans + slack.
    round_bytes = arena.layout.stripe_bytes * pool.profile.banks
    expected_padded = sum(-(-buffer.bytes // round_bytes) * round_bytes for buffer in buffers)
    assert snapshot["arena_reserved_bytes"] == arena.reserved_bytes
    assert arena.reserved_bytes >= expected_padded
    for slot in ("buf_a", "buf_b"):
      view = pool.bind_view(arena, slot, 1)
      assert pool.invalidate_view(view, view.owner, 2)
    after = pool.snapshot()
    assert after["arena_reserved_bytes"] == arena.reserved_bytes - expected_padded
    for bank in after["per_bank_occupancy"]:
      assert bank["allocated_bytes"] + bank["free_bytes"] == pool.profile.user_spm_per_bank

  def test_release_backing_frees_capacity_for_wait_capacity_head_same_cycle(self):
    pool = self._pool()
    big = self._buffer("buf_a", 768)
    reserved = conservative_arena_bytes([(big.bytes, big.alignment)], pool.profile)
    layout = layout_buffers((big,), pool.profile, reserved, lifetimes=None, slot_capacity=1)
    first = pool.plan_arena(RootInvocation("a", 0), layout)
    assert not isinstance(first, AdmissionFailure)
    arena = pool.commit_arena(first, 0)
    view = pool.bind_view(arena, "buf_a", 1)
    # A second root of the same total size cannot fit while buf_a is held.
    second = pool.plan_arena(RootInvocation("b", 0), layout)
    assert isinstance(second, AdmissionFailure)
    assert second.reason.startswith(WAIT_CAPACITY)
    assert pool.invalidate_view(view, view.owner, 2)
    # The physical final-free is immediately visible to a fresh plan.
    retried = pool.plan_arena(RootInvocation("b", 0), layout)
    assert not isinstance(retried, AdmissionFailure)

class TestSharedL2ProfileClosure:
  def test_l1_profile_completes_but_l2_waits_for_shared_backing_release(self):
    hw = HardwareConfig()
    group = TileGroup(hw, fidelity="runtime")
    controller = group.profile_controller
    cycle = 0
    while not controller.initialized:
      controller.step(cycle)
      cycle += 1
      assert cycle < 10000
    group.run_generation = 1
    controller.begin_run(1)
    group.transfer_manager.begin_run(1)
    assert controller._level_quiescent("l1")
    assert controller._level_quiescent("l2")

    pool = group.l2_sram
    buffer = TestL2BackingLifecycle()._buffer("W", 64)
    reserved = conservative_arena_bytes([(buffer.bytes, buffer.alignment)], pool.profile)
    layout = layout_buffers((buffer,), pool.profile, reserved, lifetimes=None, slot_capacity=1)
    producer_owner = RootInvocation("producer", 0)
    plan = pool.plan_arena(producer_owner, layout)
    assert not isinstance(plan, AdmissionFailure)
    producer_arena = pool.commit_arena(
      plan,
      cycle,
      claims_by_slot={"W": (("reader-binding", "weight"),)},
    )
    producer = pool.bind_view(producer_arena, "W", cycle + 1)
    pool.publish_l2(producer, cycle + 2)
    assert pool.invalidate_view(producer, producer.owner, cycle + 3)
    assert pool.retire_arena(producer_arena, cycle + 4)
    cycle += 5

    held = pool.snapshot()
    assert held["live_arenas"] == 0
    assert held["live_backings"] == 1
    assert held["pending_shared_claims"] == 1
    assert held["active_shared_references"] == 0
    assert held["physical_live_backing_bytes"] > 0
    assert held["live_view_bytes"] == buffer.bytes

    reader_owner = RootInvocation("reader", 0)
    empty_layout = _empty_layout(pool.profile, 0)
    reader_plan = pool.plan_arena(reader_owner, empty_layout)
    assert not isinstance(reader_plan, AdmissionFailure)
    reader_arena = pool.commit_arena(reader_plan, cycle)
    borrower_owner = ContextBufferOwner("reader", 0, "weight")
    borrower = pool.borrow_l2_view(
      producer.backing_id,
      borrower_owner,
      ("reader-binding", "weight"),
      cycle + 1,
    )
    cycle += 2
    bound = pool.snapshot()
    assert bound["pending_shared_claims"] == 0
    assert bound["active_shared_references"] == 1
    assert controller._level_quiescent("l1")
    assert not controller._level_quiescent("l2")

    def command(command_id: str, level: str, target: int) -> ProfileReconfigDesc:
      target_profile = group.registry.profile(level, target)
      return ProfileReconfigDesc(
        command_id=command_id,
        level=level,
        expected_mode=controller.active_modes[level],
        target_mode=target,
        registry_hash=group.registry.registry_hash,
        wait_instruction_ids=(),
        frontier=(),
        affected_domains=(level,),
        member_ids=target_profile.member_ids,
        exclusive_binding_id="",
        source_ref=SourceRef("<test>", command_id, 0, "profile.reconfig"),
      )

    def run_to_terminal(command_id: str, start_cycle: int) -> int:
      for current in range(start_cycle, start_cycle + 10000):
        group.step(current)
        if controller.status(command_id) in ("completed", "faulted", "cancelled"):
          return current + 1
      raise AssertionError(f"profile command {command_id!r} did not terminate")

    l1_command = command("shared-l1-switch", "l1", 1)
    assert controller.submit(l1_command, "device@1", cycle)
    cycle = run_to_terminal(l1_command.command_id, cycle)
    assert controller.status(l1_command.command_id) == "completed"
    assert controller.active_modes["l1"] == 1
    assert pool.snapshot()["live_backings"] == 1
    assert pool.snapshot()["active_shared_references"] == 1

    l2_command = command("shared-l2-switch", "l2", 1)
    assert controller.submit(l2_command, "device@1", cycle)
    current = cycle
    for current in range(cycle, cycle + 1000):
      group.step(current)
      current += 1
      if controller.snapshot()["commands"][l2_command.command_id]["step"] == "DRAIN_REFERENCES":
        break
    assert controller.snapshot()["commands"][l2_command.command_id]["step"] == "DRAIN_REFERENCES"
    for _ in range(16):
      group.step(current)
      current += 1
      assert controller.snapshot()["commands"][l2_command.command_id]["step"] == "DRAIN_REFERENCES"
      assert controller.active_modes["l2"] == 0

    assert pool.invalidate_view(borrower, borrower_owner, current)
    assert pool.retire_arena(reader_arena, current + 1)
    pool.close_l2_claims()
    cycle = current + 2
    closed = pool.snapshot()
    assert closed["live_backings"] == 0
    assert closed["pending_shared_claims"] == 0
    assert closed["active_shared_references"] == 0
    assert closed["arena_reserved_bytes"] == 0

    cycle = run_to_terminal(l2_command.command_id, cycle)
    assert controller.status(l2_command.command_id) == "completed"
    assert controller.active_modes["l2"] == 1
    assert pool.profile.mode == 1


class TestCommitArenaAtomicity:
  """plan/01 §2.2: commit_arena validates before any pool mutation."""

  def test_overlapping_layout_commit_fails_without_pool_mutation(self):
    from dataclasses import replace

    from pipeline_validator.immutable import digest
    from pipeline_validator.profiles import BufferLayout

    pool = TestL2BackingLifecycle()._pool()
    buffer = TestL2BackingLifecycle()._buffer("buf", 64)
    reserved = conservative_arena_bytes([(buffer.bytes, buffer.alignment)], pool.profile)
    layout = layout_buffers((buffer,), pool.profile, reserved, lifetimes=None, slot_capacity=1)
    # Forge a second buffer whose padded span overlaps the first; both are
    # individually constructible and hash-consistent.
    empty = replace(layout, layout_hash="")
    overlapped = BufferLayout(
      "ghost", empty.buffer_layouts[0].logical_bytes, 1,
      empty.buffer_layouts[0].arena_offset, empty.stripe_bytes, len(empty.per_bank_bytes), "inout",
    )
    candidate = replace(empty, buffer_layouts=(*empty.buffer_layouts, overlapped))
    forged = replace(candidate, layout_hash=digest(candidate))
    before = pool.snapshot()
    plan = pool.plan_arena(RootInvocation("ctx", 0), forged)
    assert not isinstance(plan, AdmissionFailure)
    with pytest.raises(MemoryInvariantError, match="overlap inside one bank"):
      pool.commit_arena(plan, 0)
    after = pool.snapshot()
    assert after == before
    # The pool still admits a valid arena afterwards.
    valid = pool.plan_arena(RootInvocation("ctx", 0), layout)
    assert not isinstance(valid, AdmissionFailure)
    handle = pool.commit_arena(valid, 1)
    assert pool.snapshot()["live_arenas"] == 1
    assert pool.retire_arena(handle, 2)

  def test_l1_and_zero_reserve_commits_register_their_arena_record(self):
    from pipeline_validator.execution_ir import ExecL1Buffer

    l1_profile = _profile(level="l1", banks=2, bank_bytes=512)
    l1_pool = ArenaPool(l1_profile, pool_id=0, tile_id=0)
    spec = ExecL1Buffer("tmp", (64,), "i8", 1, 64, 64)
    l1_layout = layout_buffers((spec,), l1_profile, 128, lifetimes={"tmp": (0, 1)})
    owner = TaskIdentity(GridInstanceId("ctx", 0, 0, 0), 0)
    l1_plan = l1_pool.plan_arena(owner, l1_layout)
    assert not isinstance(l1_plan, AdmissionFailure)
    l1_handle = l1_pool.commit_arena(l1_plan, 0)
    assert l1_pool._arenas[l1_handle.arena_id].handle is l1_handle
    assert l1_pool._owner_arenas[owner] == l1_handle.arena_id

    l2_pool = TestL2BackingLifecycle()._pool()
    empty_layout = _empty_layout(l2_pool.profile, 0)
    plan = l2_pool.plan_arena(RootInvocation("empty", 0), empty_layout)
    assert not isinstance(plan, AdmissionFailure)
    handle = l2_pool.commit_arena(plan, 1)
    assert handle.reserved_bytes == 0
    assert l2_pool._arenas[handle.arena_id].handle is handle
    assert l2_pool._owner_arenas[RootInvocation("empty", 0)] == handle.arena_id
    assert l2_pool.snapshot()["zero_byte_arenas"] == 1
    assert l2_pool.retire_arena(handle, 2)
