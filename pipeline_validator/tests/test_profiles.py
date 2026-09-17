from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline_validator.compiler.resources import check_layout_capacity, layout_buffers
from pipeline_validator.config import HardwareConfig
from pipeline_validator.execution_ir import GridInstanceId, TaskIdentity
from pipeline_validator.immutable import digest
from pipeline_validator.memory import AdmissionFailure, AdmissionFailureKind, MemoryInvariantError
from pipeline_validator.memory.arena import WAIT_FRAGMENTATION, ArenaPool, RootInvocation
from pipeline_validator.profiles import (
  ArenaLayout,
  MemoryProfile,
  ProfileBytes,
  ProfileLevelSource,
  ProfileSourceConfig,
  build_registry,
  parse_memory_target,
  validate_allowed,
)


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
