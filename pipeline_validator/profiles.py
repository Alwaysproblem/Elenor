"""Layered SRAM profiles and compiler/runtime resource value contracts.

The bundled values are simulator experiments, 由后续规格冻结. A profile owns
one layer's data region; control/tag/ECC storage is not user SRAM capacity.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import lcm
from typing import TYPE_CHECKING

from .immutable import FrozenMap, digest

if TYPE_CHECKING:
  from .config import HardwareConfig

UINT64_MAX = (1 << 64) - 1
_HEX = re.compile(r"[0-9a-f]{64}\Z")
PROFILE_STEPS = (
  "ACQUIRE",
  "CHECK_FRONTIER",
  "CLOSE_ISSUE",
  "DRAIN_REFERENCES",
  "CLEAN_INVALIDATE",
  "DRAIN_DOWNSTREAM",
  "PREPARE",
  "WAIT_READY_ACK",
  "COMMIT",
  "WAIT_COMMIT_ACK",
  "OPEN_ISSUE",
  "RELEASE",
)
MAINTENANCE_STEPS = (
  "ACQUIRE",
  "WAIT_DEPENDENCIES",
  "BLOCK_RANGE_ISSUE",
  "DRAIN_RANGE_REFERENCES",
  "CLEAN_INVALIDATE",
  "DRAIN_DOWNSTREAM",
  "ACK",
  "UNBLOCK_RANGE_ISSUE",
  "RELEASE",
)


def uint64(value: int, name: str, *, positive: bool = False) -> int:
  if type(value) is not int or not (int(positive) <= value <= UINT64_MAX):
    raise ValueError(f"{name} must be a {'positive ' if positive else ''}uint64")
  return value


def _text(value: str, name: str, *, allow_empty: bool = False) -> str:
  if type(value) is not str or (not value and not allow_empty) or (bool(value) and not value.strip()):
    raise ValueError(f"{name} must be a {'string' if allow_empty else 'nonempty string'}")
  return value


def _sequence(value: object, name: str) -> tuple:
  if not isinstance(value, (tuple, list)):
    raise ValueError(f"{name} must be a sequence")
  return tuple(value)


def _strings(value: object, name: str, *, unique: bool = False) -> tuple[str, ...]:
  result = _sequence(value, name)
  if any(type(item) is not str or not item.strip() for item in result):
    raise ValueError(f"{name} must contain nonempty strings")
  if unique and len(result) != len(set(result)):
    raise ValueError(f"{name} must not contain duplicates")
  return result


def _hash(value: str, name: str, *, allow_empty: bool = False) -> str:
  if allow_empty and value == "":
    return value
  if type(value) is not str or not _HEX.fullmatch(value) or int(value, 16) == 0:
    raise ValueError(f"{name} must be a nonzero canonical SHA-256 digest")
  return value


def _cache_requirement(value: object, name: str) -> None:
  if value is not None and not isinstance(value, CacheRequirement):
    raise ValueError(f"{name} must be CacheRequirement or None")


@dataclass(frozen=True)
class MemoryLevelTarget:
  system_reserved_spm_per_bank: int
  reset_mode: int
  spm_mapping_id: str
  cache_org_id: str
  cache_write_policy: str
  maintenance_caps: tuple[str, ...]

  def __post_init__(self):
    uint64(self.system_reserved_spm_per_bank, "system_reserved_spm_per_bank")
    uint64(self.reset_mode, "reset_mode")
    _text(self.spm_mapping_id, "spm_mapping_id")
    _text(self.cache_org_id, "cache_org_id")
    _text(self.cache_write_policy, "cache_write_policy")
    caps = _strings(self.maintenance_caps, "maintenance_caps", unique=True)
    object.__setattr__(self, "maintenance_caps", caps)
    if self.spm_mapping_id != "striped_arena_v0" or self.cache_org_id != "profiled_lru_v0":
      raise ValueError("unsupported SRAM mapping or cache organization")
    if self.cache_write_policy not in ("read_only", "write_back"):
      raise ValueError("unsupported cache write policy")
    if not set(caps) <= {"invalidate_range", "clean_invalidate_all", "bypass"}:
      raise ValueError("invalid maintenance capabilities")


@dataclass(frozen=True)
class MemoryTargetConfig:
  l1: MemoryLevelTarget
  l2: MemoryLevelTarget
  profile_command_timeout_cycles: int

  def __post_init__(self):
    if not isinstance(self.l1, MemoryLevelTarget) or not isinstance(self.l2, MemoryLevelTarget):
      raise ValueError("memory target layers must be MemoryLevelTarget values")
    uint64(self.profile_command_timeout_cycles, "profile_command_timeout_cycles", positive=True)


@dataclass(frozen=True)
class ProfileBytes:
  spm_bytes_per_bank: int
  cache_bytes_per_bank: int

  def __post_init__(self):
    uint64(self.spm_bytes_per_bank, "spm_bytes_per_bank")
    uint64(self.cache_bytes_per_bank, "cache_bytes_per_bank")


@dataclass(frozen=True)
class ProfileLevelSource:
  modes: Mapping[int, ProfileBytes]

  def __post_init__(self):
    if not isinstance(self.modes, Mapping) or not self.modes:
      raise ValueError("profile modes cannot be empty")
    for mode, value in self.modes.items():
      uint64(mode, "mode")
      if not isinstance(value, ProfileBytes):
        raise ValueError("mode requires explicit SPM and Cache bytes")
    object.__setattr__(self, "modes", FrozenMap(self.modes))


@dataclass(frozen=True)
class ProfileSourceConfig:
  kind: str
  partition_granule_bytes: int
  l1: ProfileLevelSource
  l2: ProfileLevelSource

  def __post_init__(self):
    if self.kind != "simulator_experiment":
      raise ValueError("unsupported profile source kind")
    uint64(self.partition_granule_bytes, "partition_granule_bytes", positive=True)
    if not isinstance(self.l1, ProfileLevelSource) or not isinstance(self.l2, ProfileLevelSource):
      raise ValueError("profile source layers must be ProfileLevelSource values")


@dataclass(frozen=True)
class MemoryProfile:
  level: str
  mode: int
  bank_bytes: int
  banks: int
  pools: int
  spm_bytes_per_bank: int
  cache_bytes_per_bank: int
  system_reserved_spm_per_bank: int
  alignment: int
  spm_mapping_id: str
  cache_org_id: str
  cache_write_policy: str
  maintenance_caps: tuple[str, ...]

  def __post_init__(self):
    if self.level not in ("l1", "l2"):
      raise ValueError("profile level must be l1 or l2")
    uint64(self.mode, "profile mode")
    uint64(self.bank_bytes, "profile bank_bytes", positive=True)
    uint64(self.banks, "profile banks", positive=True)
    uint64(self.pools, "profile pools", positive=True)
    uint64(self.spm_bytes_per_bank, "profile spm_bytes_per_bank")
    uint64(self.cache_bytes_per_bank, "profile cache_bytes_per_bank")
    uint64(self.system_reserved_spm_per_bank, "profile system_reserved_spm_per_bank")
    uint64(self.alignment, "profile alignment", positive=True)
    _text(self.spm_mapping_id, "profile spm_mapping_id")
    _text(self.cache_org_id, "profile cache_org_id")
    _text(self.cache_write_policy, "profile cache_write_policy")
    caps = _strings(self.maintenance_caps, "profile maintenance_caps", unique=True)
    object.__setattr__(self, "maintenance_caps", caps)
    if self.spm_bytes_per_bank + self.cache_bytes_per_bank != self.bank_bytes:
      raise ValueError("profile SPM and Cache bytes do not conserve bank capacity")
    if self.system_reserved_spm_per_bank > self.spm_bytes_per_bank:
      raise ValueError("profile system reservation exceeds SPM")
    if any(
      value % self.alignment
      for value in (self.spm_bytes_per_bank, self.cache_bytes_per_bank, self.system_reserved_spm_per_bank)
    ):
      raise ValueError("profile bytes are not aligned")
    if (
      self.banks * self.pools > UINT64_MAX
      or self.banks * self.bank_bytes > UINT64_MAX
      or self.banks * self.pools * self.bank_bytes > UINT64_MAX
      or self.banks * self.user_spm_per_bank > UINT64_MAX
      or self.banks * self.cache_bytes_per_bank > UINT64_MAX
    ):
      raise ValueError("profile geometry or aggregate bytes overflow uint64")
    if self.spm_mapping_id != "striped_arena_v0" or self.cache_org_id != "profiled_lru_v0":
      raise ValueError("unsupported profile mapping or cache organization")
    if self.cache_write_policy not in ("read_only", "write_back"):
      raise ValueError("unsupported profile cache write policy")
    if not set(caps) <= {"invalidate_range", "clean_invalidate_all", "bypass"}:
      raise ValueError("invalid profile maintenance capabilities")

  @property
  def user_spm_per_bank(self) -> int:
    return self.spm_bytes_per_bank - self.system_reserved_spm_per_bank

  @property
  def user_spm_bytes(self) -> int:
    return self.banks * self.user_spm_per_bank

  @property
  def cache_bytes(self) -> int:
    return self.banks * self.cache_bytes_per_bank

  @property
  def member_ids(self) -> tuple[tuple[str, int, int], ...]:
    return tuple((self.level, pool, bank) for pool in range(self.pools) for bank in range(self.banks))


@dataclass(frozen=True)
class ProfileRegistry:
  l1: Mapping[int, MemoryProfile]
  l2: Mapping[int, MemoryProfile]
  target: MemoryTargetConfig
  registry_hash: str
  abi: str = "v0"

  def __post_init__(self):
    if not isinstance(self.l1, Mapping) or not isinstance(self.l2, Mapping):
      raise ValueError("Registry profile tables must be mappings")
    if not isinstance(self.target, MemoryTargetConfig) or self.abi != "v0":
      raise ValueError("invalid Registry target or ABI")
    tables = {}
    for level, values in (("l1", self.l1), ("l2", self.l2)):
      if not values:
        raise ValueError(f"Registry {level} profile table cannot be empty")
      target = getattr(self.target, level)
      table = FrozenMap(values)
      geometry = None
      effective = set()
      for mode, profile in table.items():
        uint64(mode, f"Registry {level} mode")
        if not isinstance(profile, MemoryProfile) or profile.level != level or profile.mode != mode:
          raise ValueError(f"Registry {level} profile identity mismatch")
        if (
          profile.system_reserved_spm_per_bank != target.system_reserved_spm_per_bank
          or profile.spm_mapping_id != target.spm_mapping_id
          or profile.cache_org_id != target.cache_org_id
          or profile.cache_write_policy != target.cache_write_policy
          or profile.maintenance_caps != target.maintenance_caps
        ):
          raise ValueError(f"Registry {level} profile disagrees with its target")
        current_geometry = (profile.bank_bytes, profile.banks, profile.pools, profile.alignment)
        if geometry is None:
          geometry = current_geometry
        elif current_geometry != geometry:
          raise ValueError(f"Registry {level} modes disagree on geometry")
        identity = (profile.spm_bytes_per_bank, profile.cache_bytes_per_bank)
        if identity in effective:
          raise ValueError(f"Registry {level} contains duplicate effective profiles")
        effective.add(identity)
      if target.reset_mode not in table:
        raise ValueError(f"Registry {level} reset mode is absent")
      tables[level] = table
    object.__setattr__(self, "l1", tables["l1"])
    object.__setattr__(self, "l2", tables["l2"])
    _hash(self.registry_hash, "registry_hash", allow_empty=True)
    if self.registry_hash and self.registry_hash != digest(replace(self, registry_hash="")):
      raise ValueError("registry_hash does not match Registry contents")

  def profile(self, level: str, mode: int) -> MemoryProfile:
    if level not in ("l1", "l2"):
      raise ValueError(f"unknown memory level {level!r}")
    try:
      return getattr(self, level)[mode]
    except KeyError as exc:
      raise ValueError(f"unknown {level} profile mode {mode}") from exc


@dataclass(frozen=True)
class CacheRequirement:
  required: bool
  access: str
  bypass: str
  target_bytes: int

  def __post_init__(self):
    if type(self.required) is not bool or self.access not in ("none", "read", "read_write"):
      raise ValueError("invalid cache requirement")
    if self.bypass not in ("allowed", "forbidden"):
      raise ValueError("invalid bypass requirement")
    uint64(self.target_bytes, "cache target_bytes")
    if self.required and self.access == "none":
      raise ValueError("required cache cannot declare access=none")


@dataclass(frozen=True)
class ContextResources:
  l2_mode: int
  allowed_profiles: tuple[int, ...]
  logical_tasks: int
  l2_spm_bytes: int
  requested_contexts_per_tile: int
  l2_cache: CacheRequirement | None = None

  def __post_init__(self):
    uint64(self.l2_mode, "l2_mode")
    object.__setattr__(self, "allowed_profiles", validate_allowed(self.allowed_profiles, self.l2_mode))
    uint64(self.logical_tasks, "logical_tasks")
    uint64(self.l2_spm_bytes, "l2_spm_bytes")
    uint64(self.requested_contexts_per_tile, "requested_contexts_per_tile", positive=True)
    _cache_requirement(self.l2_cache, "l2_cache")


@dataclass(frozen=True)
class TileResources:
  allowed_profiles: tuple[int, ...]
  tile_l1_spm_bytes_per_context: int
  l1_cache: CacheRequirement | None = None
  l2_cache: CacheRequirement | None = None

  def __post_init__(self):
    object.__setattr__(self, "allowed_profiles", validate_allowed(self.allowed_profiles))
    uint64(self.tile_l1_spm_bytes_per_context, "tile_l1_spm_bytes_per_context")
    _cache_requirement(self.l1_cache, "l1_cache")
    _cache_requirement(self.l2_cache, "l2_cache")


def validate_allowed(values: Sequence[int], baseline: int | None = None) -> tuple[int, ...]:
  result = _sequence(values, "allowed_profiles")
  if not result or len(result) != len(set(result)):
    raise ValueError("allowed_profiles must be nonempty and unique")
  for mode in result:
    uint64(mode, "allowed profile mode")
  if baseline is not None:
    uint64(baseline, "requested mode")
    if baseline not in result:
      raise ValueError("requested mode must belong to allowed_profiles")
  return result


@dataclass(frozen=True)
class BufferLayout:
  buffer_id: str
  logical_bytes: int
  slot_id: int
  arena_offset: int
  # A compact striped expression: logical chunk i -> bank i % banks,
  # bank-local offset arena_offset / banks + (i // banks) * stripe_bytes.
  stripe_bytes: int
  banks: int
  role: str = ""

  def __post_init__(self):
    _text(self.buffer_id, "buffer_id")
    uint64(self.logical_bytes, "buffer logical_bytes", positive=True)
    uint64(self.slot_id, "buffer slot_id")
    uint64(self.arena_offset, "buffer arena_offset")
    uint64(self.stripe_bytes, "buffer stripe_bytes", positive=True)
    uint64(self.banks, "buffer banks", positive=True)
    if self.role not in ("", "in", "out", "inout"):
      raise ValueError("invalid buffer role")
    round_bytes = self.stripe_bytes * self.banks
    if round_bytes > UINT64_MAX or self.arena_offset % round_bytes:
      raise ValueError("buffer arena offset is not a whole stripe round")

  def segments(self) -> tuple[tuple[int, int, int], ...]:
    remaining = self.logical_bytes
    chunk = 0
    result = []
    while remaining:
      size = min(remaining, self.stripe_bytes)
      result.append(
        (
          chunk % self.banks,
          self.arena_offset // self.banks + (chunk // self.banks) * self.stripe_bytes,
          size,
        )
      )
      chunk += 1
      remaining -= size
    return tuple(result)


@dataclass(frozen=True)
class ArenaLayout:
  alignment: int
  stripe_bytes: int
  reserved_bytes: int
  per_bank_bytes: tuple[int, ...]
  buffer_layouts: tuple[BufferLayout, ...]
  layout_hash: str

  def __post_init__(self):
    uint64(self.alignment, "Arena alignment", positive=True)
    uint64(self.stripe_bytes, "Arena stripe_bytes", positive=True)
    uint64(self.reserved_bytes, "Arena reserved_bytes")
    per_bank = _sequence(self.per_bank_bytes, "Arena per_bank_bytes")
    layouts = _sequence(self.buffer_layouts, "Arena buffer_layouts")
    object.__setattr__(self, "per_bank_bytes", per_bank)
    object.__setattr__(self, "buffer_layouts", layouts)
    if not per_bank:
      raise ValueError("Arena must describe at least one bank")
    for index, value in enumerate(per_bank):
      uint64(value, f"Arena per_bank_bytes[{index}]")
    if any(not isinstance(item, BufferLayout) for item in layouts):
      raise ValueError("Arena buffer_layouts must contain BufferLayout values")
    if len({item.buffer_id for item in layouts}) != len(layouts):
      raise ValueError("Arena buffer_layouts contain duplicate buffer IDs")
    if self.stripe_bytes % self.alignment:
      raise ValueError("Arena stripe_bytes must be aligned")
    total = sum(per_bank)
    round_bytes = self.stripe_bytes * len(per_bank)
    if total != self.reserved_bytes or total > UINT64_MAX:
      raise ValueError("Arena per-bank bytes do not conserve reserved_bytes")
    if round_bytes > UINT64_MAX:
      raise ValueError("Arena stripe round overflows uint64")
    if len(set(per_bank)) != 1 or any(value % self.stripe_bytes for value in per_bank):
      raise ValueError("Arena banks must have equal whole-stripe reservations")
    if self.reserved_bytes % round_bytes:
      raise ValueError("Arena reservation is not a whole stripe round")
    for item in layouts:
      if item.banks != len(per_bank) or item.stripe_bytes != self.stripe_bytes:
        raise ValueError("Arena buffer striping disagrees with its bank geometry")
      if item.arena_offset % self.alignment:
        raise ValueError("Arena buffer offset is misaligned")
      chunks = (item.logical_bytes + item.stripe_bytes - 1) // item.stripe_bytes
      base = item.arena_offset // item.banks
      for bank in range(min(item.banks, chunks)):
        bank_chunks = (chunks - 1 - bank) // item.banks + 1
        last_chunk = bank + (bank_chunks - 1) * item.banks
        last_size = (
          item.logical_bytes - (chunks - 1) * item.stripe_bytes
          if last_chunk == chunks - 1
          else item.stripe_bytes
        )
        if base + (bank_chunks - 1) * item.stripe_bytes + last_size > per_bank[bank]:
          raise ValueError("Arena buffer segment escapes its reservation")
    _hash(self.layout_hash, "layout_hash", allow_empty=True)
    if self.layout_hash and self.layout_hash != digest(replace(self, layout_hash="")):
      raise ValueError("layout_hash does not match Arena layout")


@dataclass(frozen=True)
class SourceRef:
  source_name: str
  symbol: str
  body_op_index: int
  op_name: str
  generated_by: str = ""
  reason: str = ""

  def __post_init__(self):
    _text(self.source_name, "source_ref.source_name")
    _text(self.symbol, "source_ref.symbol")
    uint64(self.body_op_index, "source_ref.body_op_index")
    _text(self.op_name, "source_ref.op_name")
    _text(self.generated_by, "source_ref.generated_by", allow_empty=True)
    _text(self.reason, "source_ref.reason", allow_empty=True)
    if self.generated_by and not self.reason:
      raise ValueError("generated source_ref requires a reason")


@dataclass(frozen=True)
class ProfileState:
  l1_mode: int
  l2_mode: int

  def __post_init__(self):
    uint64(self.l1_mode, "profile state l1_mode")
    uint64(self.l2_mode, "profile state l2_mode")


@dataclass(frozen=True)
class ProfileReconfigDesc:
  command_id: str
  level: str
  expected_mode: int
  target_mode: int
  registry_hash: str
  wait_instruction_ids: tuple[str, ...]
  frontier: tuple[str, ...]
  affected_domains: tuple[str, ...]
  member_ids: tuple[tuple[str, int, int], ...]
  exclusive_binding_id: str
  source_ref: SourceRef
  steps: tuple[str, ...] = PROFILE_STEPS

  def __post_init__(self):
    _text(self.command_id, "profile command_id")
    if self.level not in ("l1", "l2"):
      raise ValueError("profile command level must be l1 or l2")
    uint64(self.expected_mode, "profile expected_mode")
    uint64(self.target_mode, "profile target_mode")
    _hash(self.registry_hash, "profile registry_hash")
    waits = _strings(self.wait_instruction_ids, "profile wait_instruction_ids", unique=True)
    frontier = _strings(self.frontier, "profile frontier", unique=True)
    domains = _strings(self.affected_domains, "profile affected_domains", unique=True)
    members = _sequence(self.member_ids, "profile member_ids")
    steps = _strings(self.steps, "profile steps")
    object.__setattr__(self, "wait_instruction_ids", waits)
    object.__setattr__(self, "frontier", frontier)
    object.__setattr__(self, "affected_domains", domains)
    object.__setattr__(self, "member_ids", members)
    object.__setattr__(self, "steps", steps)
    if domains != (self.level,) or steps != PROFILE_STEPS:
      raise ValueError("profile command has invalid domains or steps")
    normalized_members = []
    for member in members:
      member = _sequence(member, "profile member_id")
      if len(member) != 3 or member[0] != self.level:
        raise ValueError("invalid profile member_id")
      uint64(member[1], "profile member pool")
      uint64(member[2], "profile member bank")
      normalized_members.append(member)
    normalized_members = tuple(normalized_members)
    if not normalized_members or len(normalized_members) != len(set(normalized_members)):
      raise ValueError("profile member_ids must be nonempty and unique")
    object.__setattr__(self, "member_ids", normalized_members)
    _text(self.exclusive_binding_id, "profile exclusive_binding_id", allow_empty=True)
    if not isinstance(self.source_ref, SourceRef):
      raise ValueError("profile command requires SourceRef")


@dataclass(frozen=True)
class MaintenanceRange:
  input_index: int
  offset: int
  bytes: int
  levels: tuple[str, ...]

  def __post_init__(self):
    uint64(self.input_index, "maintenance input_index")
    uint64(self.offset, "maintenance offset")
    uint64(self.bytes, "maintenance bytes", positive=True)
    if self.offset + self.bytes > UINT64_MAX:
      raise ValueError("maintenance range overflows uint64")
    levels = _strings(self.levels, "maintenance range levels", unique=True)
    if not levels or not set(levels) <= {"l1", "l2"}:
      raise ValueError("invalid maintenance range levels")
    object.__setattr__(self, "levels", levels)


@dataclass(frozen=True)
class MemoryMaintenanceDesc:
  command_id: str
  levels: tuple[str, ...]
  ranges: tuple[MaintenanceRange, ...]
  dependencies: tuple[str, ...]
  source_ref: SourceRef
  steps: tuple[str, ...] = MAINTENANCE_STEPS

  def __post_init__(self):
    _text(self.command_id, "maintenance command_id")
    levels = _strings(self.levels, "maintenance levels", unique=True)
    ranges = _sequence(self.ranges, "maintenance ranges")
    dependencies = _strings(self.dependencies, "maintenance dependencies", unique=True)
    steps = _strings(self.steps, "maintenance steps")
    object.__setattr__(self, "levels", levels)
    object.__setattr__(self, "ranges", ranges)
    object.__setattr__(self, "dependencies", dependencies)
    object.__setattr__(self, "steps", steps)
    if not levels or not set(levels) <= {"l1", "l2"}:
      raise ValueError("invalid maintenance levels")
    if not ranges or any(not isinstance(item, MaintenanceRange) for item in ranges):
      raise ValueError("maintenance ranges must contain MaintenanceRange values")
    if any(not set(item.levels) <= set(levels) for item in ranges):
      raise ValueError("maintenance range levels exceed command levels")
    if steps != MAINTENANCE_STEPS:
      raise ValueError("maintenance command has invalid steps")
    if not isinstance(self.source_ref, SourceRef):
      raise ValueError("maintenance command requires SourceRef")


@dataclass(frozen=True)
class CallBinding:
  binding_id: str
  context_name: str
  requested_l2_mode: int
  resolved_l2_mode: int
  entry_l1_mode: int
  exit_l1_mode: int
  requires_l1_exclusive: bool
  actual_inputs: tuple[int, ...] = ()
  permitted_profiles: tuple[ProfileState, ...] = ()

  def __post_init__(self):
    _text(self.binding_id, "call binding_id")
    _text(self.context_name, "call context_name")
    uint64(self.requested_l2_mode, "call requested_l2_mode")
    uint64(self.resolved_l2_mode, "call resolved_l2_mode")
    uint64(self.entry_l1_mode, "call entry_l1_mode")
    uint64(self.exit_l1_mode, "call exit_l1_mode")
    if type(self.requires_l1_exclusive) is not bool:
      raise ValueError("call requires_l1_exclusive must be bool")
    actuals = _sequence(self.actual_inputs, "call actual_inputs")
    profiles = _sequence(self.permitted_profiles, "call permitted_profiles")
    object.__setattr__(self, "actual_inputs", actuals)
    object.__setattr__(self, "permitted_profiles", profiles)
    for value in actuals:
      uint64(value, "call actual input")
    if not profiles or any(not isinstance(value, ProfileState) for value in profiles):
      raise ValueError("call permitted_profiles must contain ProfileState values")
    if len(profiles) != len(set(profiles)):
      raise ValueError("call permitted_profiles contain duplicates")


@dataclass(frozen=True)
class ResourceBudget:
  max_live_grids: int
  event_frontier: int
  frame_slots: int
  engine_requirements: Mapping[str, int] = field(default_factory=FrozenMap)

  def __post_init__(self):
    uint64(self.max_live_grids, "max_live_grids")
    uint64(self.event_frontier, "event_frontier")
    uint64(self.frame_slots, "frame_slots")
    if not isinstance(self.engine_requirements, Mapping):
      raise ValueError("engine_requirements must be a mapping")
    requirements = FrozenMap(self.engine_requirements)
    for name, value in requirements.items():
      _text(name, "engine requirement name")
      uint64(value, f"engine requirement {name}")
    object.__setattr__(self, "engine_requirements", requirements)


def build_registry(hw: HardwareConfig) -> ProfileRegistry:
  target = hw.memory_target
  source = hw.profile_source
  quantum = lcm(source.partition_granule_bytes, hw.cache_line_bytes)
  tables = {}
  for level, capacity, banks, pools in (
    ("l1", hw.tile_l1_bytes, hw.tile_l1_banks, hw.num_tiles),
    ("l2", hw.group_sram_bytes, hw.group_sram_banks, 1),
  ):
    uint64(capacity, f"{level} capacity", positive=True)
    uint64(banks, f"{level} banks", positive=True)
    uint64(pools, f"{level} pools", positive=True)
    if capacity % banks:
      raise ValueError(f"{level} capacity must be divisible by banks")
    bank_bytes = capacity // banks
    spec = getattr(target, level)
    table = {}
    seen = set()
    for mode, value in getattr(source, level).modes.items():
      s, k, r = value.spm_bytes_per_bank, value.cache_bytes_per_bank, spec.system_reserved_spm_per_bank
      if s + k != bank_bytes or any(n % quantum for n in (s, k, r)) or r > s:
        raise ValueError(f"{level}.mode{mode}: invalid conservation, alignment, or system reservation")
      if (s, k) in seen:
        raise ValueError(f"{level}.mode{mode}: duplicate effective profile")
      seen.add((s, k))
      table[mode] = MemoryProfile(
        level,
        mode,
        bank_bytes,
        banks,
        pools,
        s,
        k,
        r,
        quantum,
        spec.spm_mapping_id,
        spec.cache_org_id,
        spec.cache_write_policy,
        spec.maintenance_caps,
      )
    if spec.reset_mode not in table:
      raise ValueError(f"{level}: reset mode not in Registry")
    tables[level] = table
  registry = ProfileRegistry(tables["l1"], tables["l2"], target, "")
  return replace(registry, registry_hash=digest(registry))


def _mapping(value, allowed: set[str], name: str) -> Mapping:
  if not isinstance(value, Mapping) or not set(value) <= allowed:
    raise ValueError(f"invalid or unknown fields in {name}")
  return value


def parse_memory_target(data, defaults: MemoryTargetConfig | None = None) -> MemoryTargetConfig:
  data = _mapping(data, {"l1", "l2", "profile_command_timeout_cycles"}, "memory.target")
  layers = {}
  level_fields = {
    "system_reserved_spm_per_bank",
    "reset_mode",
    "spm_mapping_id",
    "cache_org_id",
    "cache_write_policy",
    "maintenance_caps",
  }
  for level in ("l1", "l2"):
    if level not in data:
      if defaults is None:
        raise ValueError(f"missing memory.target.{level}")
      layers[level] = getattr(defaults, level)
      continue
    raw = _mapping(data[level], level_fields, f"memory.target.{level}")
    if "system_reserved_spm_per_bank" not in raw:
      raise ValueError(f"explicit memory.target.{level} requires system_reserved_spm_per_bank")
    values = (
      {name: getattr(getattr(defaults, level), name) for name in level_fields}
      if defaults is not None
      else {}
    )
    values.update(raw)
    if set(values) != level_fields:
      raise ValueError(f"incomplete memory.target.{level}")
    caps = values["maintenance_caps"]
    if not isinstance(caps, (tuple, list)) or any(not isinstance(item, str) for item in caps):
      raise ValueError("maintenance_caps must be a string list")
    layers[level] = MemoryLevelTarget(**values)
  timeout = data.get("profile_command_timeout_cycles")
  if timeout is None and defaults is not None:
    timeout = defaults.profile_command_timeout_cycles
  if timeout is None:
    raise ValueError("missing profile_command_timeout_cycles")
  return MemoryTargetConfig(layers["l1"], layers["l2"], timeout)


def parse_profile_source(data, defaults: ProfileSourceConfig | None = None) -> ProfileSourceConfig:
  data = _mapping(data, {"kind", "partition_granule_bytes", "l1", "l2"}, "memory.profile_source")
  layers = {}
  for level in ("l1", "l2"):
    if level not in data:
      if defaults is None:
        raise ValueError(f"missing memory.profile_source.{level}")
      layers[level] = getattr(defaults, level)
      continue
    raw = _mapping(data[level], {"modes"}, f"memory.profile_source.{level}")
    if "modes" not in raw or not isinstance(raw["modes"], Mapping):
      raise ValueError(f"{level} requires an explicit modes mapping")
    modes = {}
    for mode, values in raw["modes"].items():
      uint64(mode, f"{level} mode")
      values = _mapping(values, {"spm_bytes_per_bank", "cache_bytes_per_bank"}, f"{level}.mode{mode}")
      if set(values) != {"spm_bytes_per_bank", "cache_bytes_per_bank"}:
        raise ValueError("both SPM and Cache bytes must be explicit")
      modes[mode] = ProfileBytes(**values)
    layers[level] = ProfileLevelSource(modes)
  kind = data.get("kind", defaults.kind if defaults is not None else None)
  granule = data.get("partition_granule_bytes", defaults.partition_granule_bytes if defaults else None)
  if kind is None or granule is None:
    raise ValueError("missing profile_source kind or partition_granule_bytes")
  return ProfileSourceConfig(kind, granule, layers["l1"], layers["l2"])
