"""Read-only semantic verification for immutable compiled executables.

This module deliberately does not import the compiler and never repairs an
executable graph.  Every ordering decision used below is reconstructed from
actual transfers, views, tile descriptors, and formal-to-actual bindings.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from itertools import pairwise
from typing import NoReturn

from .compiled_program import BindingGuard, CompiledProgram, WorkloadInfo, artifact_digest, program_digest
from .config import HardwareConfig, SimConfig
from .execution_ir import (
  ExecDeviceOp,
  ExecDispatchRequest,
  ExecEngineDesc,
  ExecGlobalInput,
  ExecGroupAction,
  ExecGroupActionOp,
  ExecHostCall,
  ExecIndexedMap,
  ExecL1Buffer,
  ExecL2Buffer,
  ExecMemoryView,
  ExecModel,
  ExecPublishRequest,
  ExecReleaseRequest,
  ExecSharedInput,
  ExecStreamDesc,
  ExecTileGatherDesc,
  ExecTileGroupTask,
  ExecTileOp,
  ExecTileProgram,
  ExecTileRoleBinding,
  ExecTileScatterDesc,
  ExecTransfer,
  GlobalBinding,
)
from .immutable import FrozenMap, canonical_value, digest
from .profiles import (
  MAINTENANCE_STEPS,
  PROFILE_STEPS,
  ArenaLayout,
  BufferLayout,
  CacheRequirement,
  CallBinding,
  ContextResources,
  MemoryMaintenanceDesc,
  MemoryProfile,
  ProfileReconfigDesc,
  ProfileRegistry,
  ProfileState,
  ResourceBudget,
  SourceRef,
  TileResources,
  build_registry,
)

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_DTYPE_BYTES = {"i8": 1, "bf16": 2, "f16": 2, "i32": 4, "f32": 4}
_UINT64_MAX = (1 << 64) - 1

# Only values which change topology, storage, queueing, or finite controller
# feasibility belong to the target.  Timing knobs and scheduler policy do not.
_HW_TARGET_FIELDS = (
  "num_tiles",
  "group_sram_bytes",
  "group_sram_banks",
  "tile_l1_bytes",
  "tile_l1_banks",
  "boa_num_opa",
  "boa_opa_rows",
  "boa_opa_cols",
  "boa_dtype_bytes",
  "boa_acc_bytes",
  "evu_lanes",
  "evu_dtype_bytes",
  "use_state_cache_bytes",
  "uce_dispatch_per_cycle",
  "stream_depth_default",
  "mfe_pipeline_depth",
  "mfe_load_channels",
  "mfe_store_channels",
  "mfe_load_queue_depth",
  "mfe_store_queue_depth",
  "mfe_stream_buffer_bytes",
  "hbm_capacity_bytes",
  "hbm_outstanding_limit",
  "hbm_channels",
  "hbm_burst_bytes",
  "cache_line_bytes",
  "l2_mshr_entries",
  "l1_mshr_entries",
  "tile_program_sram_bytes",
  "noc_vc_depth",
  "num_dma_channels",
  "frame_slot_capacity",
)
_SIM_TARGET_FIELDS = ("context_count", "device_context_count")
_DEVICE_TARGET_FIELDS = ("pending_capacity", "completion_capacity")
_GROUP_TARGET_FIELDS = (
  "active_context_capacity",
  "context_pending_capacity",
  "action_capacity",
  "context_action_quota",
  "scan_width",
  "event_capacity",
  "inflight_capacity",
  "prefetch_capacity",
  "store_capacity",
  "dispatch_capacity",
)


def target_fingerprint(hw: HardwareConfig, sim: SimConfig) -> str:
  """Return the canonical static-target SHA-256 used by compiler and Loader.

  Runtime-only choices (trace, fidelity, random seed, maximum cycles, timing
  latencies, and scheduling policy) are intentionally absent.
  """
  if not isinstance(hw, HardwareConfig) or not isinstance(sim, SimConfig):
    raise ValueError("target_fingerprint requires HardwareConfig and SimConfig")
  registry = build_registry(hw)
  payload = {
    "hardware": {name: getattr(hw, name) for name in _HW_TARGET_FIELDS},
    "simulation": {name: getattr(sim, name) for name in _SIM_TARGET_FIELDS},
    "device": {name: getattr(sim.device, name) for name in _DEVICE_TARGET_FIELDS},
    "group": {name: getattr(sim.group, name) for name in _GROUP_TARGET_FIELDS},
    "registry": registry,
  }
  return digest(payload)


def _fail(message: str) -> NoReturn:
  raise ValueError(f"invalid compiled program: {message}")


def _uint(value: object, name: str, *, positive: bool = False) -> int:
  if type(value) is not int or not (int(positive) <= value <= _UINT64_MAX):
    _fail(f"{name} must be a {'positive ' if positive else ''}uint64")
  return value


def _nonempty(value: object, name: str) -> str:
  if not isinstance(value, str) or not value.strip():
    _fail(f"{name} must be a non-empty string")
  return value


def _unique(values: Sequence[object], name: str) -> None:
  if len(values) != len(set(values)):
    _fail(f"{name} contains duplicates")


def _is_frozen(value: object) -> bool:
  if value is None or type(value) in (str, int, bool) or isinstance(value, Enum):
    return True
  if type(value) is float:
    return math.isfinite(value)
  if isinstance(value, tuple):
    return all(_is_frozen(item) for item in value)
  if isinstance(value, FrozenMap):
    return all(_is_frozen(key) and _is_frozen(item) for key, item in value.items())
  if is_dataclass(value) and not isinstance(value, type):
    params = getattr(type(value), "__dataclass_params__", None)
    return bool(params and params.frozen) and all(_is_frozen(getattr(value, f.name)) for f in fields(value))
  return False


def _verify_source_ref(ref: object, where: str) -> SourceRef:
  if not isinstance(ref, SourceRef):
    _fail(f"{where} is missing source_ref")
  _nonempty(ref.source_name, f"{where}.source_ref.source_name")
  _nonempty(ref.symbol, f"{where}.source_ref.symbol")
  _uint(ref.body_op_index, f"{where}.source_ref.body_op_index")
  _nonempty(ref.op_name, f"{where}.source_ref.op_name")
  if ref.generated_by and not ref.reason:
    _fail(f"{where} generated source_ref requires a reason")
  return ref


def _instruction_identity(item: object, where: str) -> tuple[str, SourceRef]:
  instruction_id = _nonempty(getattr(item, "instruction_id", ""), f"{where}.instruction_id")
  return instruction_id, _verify_source_ref(getattr(item, "source_ref", None), where)


def _verify_global_input(item: ExecGlobalInput, where: str) -> None:
  if not isinstance(item, ExecGlobalInput):
    _fail(f"{where} is not an ExecGlobalInput")
  _nonempty(item.name, f"{where}.name")
  if not item.dims or any(type(dim) is not int or dim <= 0 for dim in item.dims):
    _fail(f"{where}.dims must be positive integers")
  if item.dtype not in _DTYPE_BYTES:
    _fail(f"{where} has unsupported dtype {item.dtype!r}")
  expected = _product(item.dims) * _DTYPE_BYTES[item.dtype]
  if item.size_bytes != expected or not 0 < expected < (1 << 63):
    _fail(f"{where}.size_bytes does not match shape and dtype")


def _product(values: Sequence[int]) -> int:
  result = 1
  for value in values:
    result *= value
  return result


def _view_start(view: ExecMemoryView) -> int:
  element_offset = 0
  for index, offset in enumerate(view.offsets):
    element_offset += offset * _product(view.backing_dims[index + 1 :])
  return element_offset * view.element_bytes


def _verify_view(view: object, where: str, *, spaces: set[str] | None = None) -> ExecMemoryView:
  if not isinstance(view, ExecMemoryView):
    _fail(f"{where} is not an ExecMemoryView")
  if view.space not in (spaces or {"global", "l2", "l1"}):
    _fail(f"{where} has invalid memory space {view.space!r}")
  _nonempty(view.base, f"{where}.base")
  rank = len(view.backing_dims)
  if rank == 0 or any(len(part) != rank for part in (view.dims, view.offsets, view.strides)):
    _fail(f"{where} has inconsistent or zero rank")
  if any(type(dim) is not int or dim <= 0 for dim in (*view.backing_dims, *view.dims)):
    _fail(f"{where} dimensions must be positive integers")
  if any(type(offset) is not int or offset < 0 for offset in view.offsets):
    _fail(f"{where} offsets must be non-negative integers")
  if any(stride != 1 for stride in view.strides):
    _fail(f"{where} uses a non-unit stride")
  if any(
    offset + size > parent for offset, size, parent in zip(view.offsets, view.dims, view.backing_dims)
  ):
    _fail(f"{where} exceeds its backing extent")
  for index, size in enumerate(view.dims):
    if size > 1 and any(view.dims[j] != view.backing_dims[j] for j in range(index + 1, rank)):
      _fail(f"{where} is not a contiguous row-major view")
  if view.dtype not in _DTYPE_BYTES or view.element_bytes != _DTYPE_BYTES.get(view.dtype):
    _fail(f"{where} has invalid dtype width")
  expected_bytes = _product(view.dims) * view.element_bytes
  if view.bytes != expected_bytes or not 0 < expected_bytes < (1 << 63):
    _fail(f"{where}.bytes does not match its shape")
  if view.task_dim is not None and (
    type(view.task_dim) is not int or not 0 <= view.task_dim < rank or view.space != "l2"
  ):
    _fail(f"{where} has an invalid task dimension")
  return view


def _verify_transfer(transfer: object, where: str) -> ExecTransfer:
  if not isinstance(transfer, ExecTransfer):
    _fail(f"{where} is not an ExecTransfer")
  src = _verify_view(transfer.src, f"{where}.src")
  dst = _verify_view(transfer.dst, f"{where}.dst")
  if transfer.bytes != src.bytes or transfer.bytes != dst.bytes:
    _fail(f"{where} byte count differs from source or destination")
  return transfer


def _verify_indexed_map(address_map: object, where: str) -> ExecIndexedMap:
  if not isinstance(address_map, ExecIndexedMap):
    _fail(f"{where} is not an ExecIndexedMap")
  if address_map.index_scale <= 0 or address_map.repeat <= 0 or address_map.segment <= 0:
    _fail(f"{where} must have positive index_scale/repeat/segment")
  if address_map.offset < 0 or address_map.task_stride < 0 or address_map.stride < 0:
    _fail(f"{where} must have non-negative offset/task_stride/stride")
  return address_map


def _verify_tile_gather(gather: object, where: str) -> ExecTileGatherDesc:
  if not isinstance(gather, ExecTileGatherDesc):
    _fail(f"{where} is not an ExecTileGatherDesc")
  source = _verify_view(gather.source, f"{where}.source", spaces={"global"})
  indices = _verify_view(gather.indices, f"{where}.indices", spaces={"l1"})
  destination = _verify_view(gather.destination, f"{where}.destination", spaces={"l1"})
  if indices.base == destination.base:
    _fail(f"{where} indices and destination alias the same L1 allocation")
  _verify_indexed_map(gather.address_map, f"{where}.address_map")
  _uint(gather.window_entries, f"{where}.window_entries", positive=True)
  if indices.dtype != "i32" or indices.element_bytes != 4:
    _fail(f"{where} indices must be i32")
  if destination.dtype != source.dtype or destination.element_bytes != source.element_bytes:
    _fail(f"{where} destination dtype must match source dtype")
  index_count = indices.bytes // 4
  if index_count <= 0:
    _fail(f"{where} indices must hold at least one element")
  element_bytes = source.element_bytes
  required_bytes = index_count * gather.address_map.repeat * gather.address_map.segment * element_bytes
  if destination.bytes != required_bytes:
    _fail(f"{where} destination must hold exactly I*R*L elements")
  if gather.scope is not None:
    _nonempty(gather.scope, f"{where}.scope")
  return gather


def _verify_tile_scatter(scatter: object, where: str) -> ExecTileScatterDesc:
  if not isinstance(scatter, ExecTileScatterDesc):
    _fail(f"{where} is not an ExecTileScatterDesc")
  source = _verify_view(scatter.source, f"{where}.source", spaces={"l1"})
  indices = _verify_view(scatter.indices, f"{where}.indices", spaces={"l1"})
  destination = _verify_view(scatter.destination, f"{where}.destination", spaces={"global"})
  if indices.base == source.base:
    _fail(f"{where} indices and source alias the same L1 allocation")
  _verify_indexed_map(scatter.address_map, f"{where}.address_map")
  _uint(scatter.window_entries, f"{where}.window_entries", positive=True)
  if indices.dtype != "i32" or indices.element_bytes != 4:
    _fail(f"{where} indices must be i32")
  if source.dtype != destination.dtype or source.element_bytes != destination.element_bytes:
    _fail(f"{where} source dtype must match destination dtype")
  index_count = indices.bytes // 4
  if index_count <= 0:
    _fail(f"{where} indices must hold at least one element")
  element_bytes = destination.element_bytes
  required_bytes = index_count * scatter.address_map.repeat * scatter.address_map.segment * element_bytes
  if source.bytes != required_bytes:
    _fail(f"{where} source must hold exactly I*R*L elements")
  if scatter.scope is not None:
    _nonempty(scatter.scope, f"{where}.scope")
  return scatter


def _layout_digest(layout: ArenaLayout) -> str:
  return digest(replace(layout, layout_hash=""))


def _verify_layout(
  layout: object, where: str, profile: MemoryProfile, buffers: Mapping[str, ExecL1Buffer | ExecL2Buffer]
) -> ArenaLayout:
  if not isinstance(layout, ArenaLayout):
    _fail(f"{where} is missing ArenaLayout")
  _uint(layout.alignment, f"{where}.alignment", positive=True)
  _uint(layout.stripe_bytes, f"{where}.stripe_bytes", positive=True)
  _uint(layout.reserved_bytes, f"{where}.reserved_bytes")
  if layout.layout_hash != _layout_digest(layout):
    _fail(f"{where}.layout_hash mismatch")
  if layout.alignment % profile.alignment or layout.stripe_bytes % profile.alignment:
    _fail(f"{where} alignment is incompatible with the Registry profile")
  if len(layout.per_bank_bytes) != profile.banks:
    _fail(f"{where}.per_bank_bytes does not cover every bank")
  if any(
    type(value) is not int or value < 0 or value > profile.user_spm_per_bank
    for value in layout.per_bank_bytes
  ):
    _fail(f"{where}.per_bank_bytes exceeds the user SPM interval")
  if sum(layout.per_bank_bytes) != layout.reserved_bytes:
    _fail(f"{where}.reserved_bytes differs from its per-bank reservation")
  by_id: dict[str, BufferLayout] = {}
  for index, item in enumerate(layout.buffer_layouts):
    if not isinstance(item, BufferLayout):
      _fail(f"{where}.buffer_layouts[{index}] is invalid")
    _nonempty(item.buffer_id, f"{where}.buffer_layouts[{index}].buffer_id")
    if item.buffer_id in by_id:
      _fail(f"{where} has duplicate buffer layout {item.buffer_id!r}")
    by_id[item.buffer_id] = item
    if item.buffer_id not in buffers or item.logical_bytes != buffers[item.buffer_id].bytes:
      _fail(f"{where} buffer {item.buffer_id!r} does not match its executable allocation")
    if item.banks != profile.banks or item.stripe_bytes != layout.stripe_bytes:
      _fail(f"{where} buffer {item.buffer_id!r} uses inconsistent striping")
    _uint(item.slot_id, f"{where}.{item.buffer_id}.slot_id")
    _uint(item.arena_offset, f"{where}.{item.buffer_id}.arena_offset")
    if item.arena_offset % layout.alignment:
      _fail(f"{where} buffer {item.buffer_id!r} is misaligned")
    segments = item.segments()
    if sum(size for _, _, size in segments) != item.logical_bytes:
      _fail(f"{where} buffer {item.buffer_id!r} has an invalid segment expression")
    for bank, offset, size in segments:
      if not 0 <= bank < profile.banks or offset < 0 or size <= 0:
        _fail(f"{where} buffer {item.buffer_id!r} has an invalid bank segment")
      if offset + size > layout.per_bank_bytes[bank]:
        _fail(f"{where} buffer {item.buffer_id!r} escapes its per-bank reservation")
  if set(by_id) != set(buffers):
    _fail(f"{where} does not describe every executable buffer")
  return layout


def _verify_l2_no_rebind_layout(layout: ArenaLayout, where: str) -> None:
  """Independent no-rebind L2 contract, recomputed from the frozen DTO.

  Every buffer owns a whole-stripe-rounded padded span: per-bank spans stay
  inside the reservation, are aligned, pairwise disjoint, and never exceed
  the reservation.  Every unallocated byte is root-held arena slack; the
  exact spans-plus-slack tiling is committed at runtime (pool tests) and a
  static over-reservation is bounded by the resource contract check.
  Reuse-era layouts whose spans overlap are rejected here regardless of any
  source-side acceptance.
  """
  banks = len(layout.per_bank_bytes)
  round_bytes = layout.stripe_bytes * banks
  spans: list[tuple[int, int, str]] = []
  total_padded = 0
  for item in layout.buffer_layouts:
    padded = -(-item.logical_bytes // round_bytes) * round_bytes
    start = item.arena_offset // banks
    per_bank = padded // banks
    if start + per_bank > layout.per_bank_bytes[0]:
      _fail(f"{where} buffer {item.buffer_id!r} padded span escapes its per-bank reservation")
    spans.append((start, start + per_bank, item.buffer_id))
    total_padded += padded
  spans.sort()
  for (_left_start, left_end, left_id), (right_start, _right_end, right_id) in pairwise(spans):
    if right_start < left_end:
      _fail(f"{where} buffers {left_id!r} and {right_id!r} have overlapping padded L2 spans")
  # Spans are disjoint and bounded, so every unallocated byte between or
  # after them is root-held arena slack (plan/01 §2.2: buffer units plus
  # slack units exactly tile the reservation).
  if total_padded > layout.reserved_bytes:
    _fail(f"{where} padded L2 spans exceed the Arena reservation")


def _segments_overlap(left: BufferLayout, right: BufferLayout) -> bool:
  for left_bank, left_start, left_size in left.segments():
    for right_bank, right_start, right_size in right.segments():
      if (
        left_bank == right_bank
        and left_start < right_start + right_size
        and right_start < left_start + left_size
      ):
        return True
  return False


def _formal_index(view: ExecMemoryView, where: str) -> int:
  if not view.base.startswith("formal:"):
    _fail(f"{where} does not reference a program formal")
  try:
    index = int(view.base.removeprefix("formal:"))
  except ValueError:
    _fail(f"{where} has an invalid formal reference")
  if index < 0:
    _fail(f"{where} has an invalid formal reference")
  return index


@dataclass(frozen=True)
class _ProgramEffects:
  descriptor_views: Mapping[str, tuple[tuple[ExecMemoryView, bool], ...]]
  l2_reads: tuple[int, ...]
  l2_writes: tuple[int, ...]
  global_reads: tuple[int, ...]
  global_writes: tuple[int, ...]
  uses_l1: bool
  gather_levels: frozenset[str]
  scatter_formals: frozenset[int] = frozenset()
  gather_scopes: Mapping[int, str | None] = field(default_factory=dict)


def _verify_program(
  program: ExecTileProgram, where: str, registry: ProfileRegistry, hw: HardwareConfig
) -> _ProgramEffects:
  if not isinstance(program, ExecTileProgram):
    _fail(f"{where} is not an ExecTileProgram")
  _nonempty(program.name, f"{where}.name")
  _uint(program.program_id, f"{where}.program_id", positive=True)
  _uint(program.version, f"{where}.version", positive=True)
  if type(program.program_hash) is not int or not 0 < program.program_hash < (1 << 256):
    _fail(f"{where}.program_hash is not a nonzero SHA-256 integer")
  if program.program_hash != program_digest(program):
    _fail(f"{where}.program_hash does not match executable content")
  if not isinstance(program.resource_contract, TileResources):
    _fail(f"{where} is missing TileResources")
  _uint(program.text_bytes, f"{where}.text_bytes", positive=True)
  if program.text_bytes > hw.tile_program_sram_bytes:
    _fail(f"{where}.text_bytes exceeds Tile Program SRAM capacity")
  contract = program.resource_contract
  if (
    not program.formals
    or program.formals[0].space != "task"
    or program.formals[0].dims
    or program.formals[0].dtype
  ):
    _fail(f"{where} first formal must be the task formal")
  seen_l2 = False
  for index, formal in enumerate(program.formals[1:], start=1):
    if formal.space not in ("global", "l2"):
      _fail(f"{where}.formals[{index}] has invalid space")
    if formal.space == "global" and seen_l2:
      _fail(f"{where} global formal follows an L2 formal")
    seen_l2 |= formal.space == "l2"
    if not formal.dims or any(type(dim) is not int or dim <= 0 for dim in formal.dims):
      _fail(f"{where}.formals[{index}] has invalid shape")
    if formal.dtype not in _DTYPE_BYTES:
      _fail(f"{where}.formals[{index}] has invalid dtype")
  if program.labels:
    _fail(f"{where} contains control-flow labels not emitted by the formal compiler")
  l1_buffers: dict[str, ExecL1Buffer] = {}
  for index, buffer in enumerate(program.l1_buffers):
    if not isinstance(buffer, ExecL1Buffer):
      _fail(f"{where}.l1_buffers[{index}] is invalid")
    if buffer.name in l1_buffers:
      _fail(f"{where} has duplicate L1 buffer {buffer.name!r}")
    l1_buffers[buffer.name] = buffer
    _verify_buffer(buffer, f"{where}.l1_buffers[{index}]")
  for mode in contract.allowed_profiles:
    profile = registry.profile("l1", mode)
    if contract.tile_l1_spm_bytes_per_context > profile.user_spm_bytes:
      _fail(f"{where} L1 contract exceeds allowed profile {mode}")
  baseline_profile = registry.profile("l1", contract.allowed_profiles[0])
  layout = _verify_layout(program.layout, f"{where}.layout", baseline_profile, l1_buffers)
  if layout.reserved_bytes > contract.tile_l1_spm_bytes_per_context:
    _fail(f"{where} L1 layout exceeds its declared resource contract")

  descriptor_views: dict[str, tuple[tuple[ExecMemoryView, bool], ...]] = {}
  l2_reads: list[int] = []
  l2_writes: list[int] = []
  global_reads: list[int] = []
  global_writes: list[int] = []
  gather_levels: set[str] = set()
  scatter_formals: set[int] = set()
  gather_scopes: dict[int, str | None] = {}
  has_gather = any(
    isinstance(descriptor.params.get("gather"), ExecTileGatherDesc)
    for descriptor in program.descriptors.values()
    if isinstance(descriptor, ExecEngineDesc)
  )
  for name, descriptor in program.descriptors.items():
    if not isinstance(name, str) or not isinstance(descriptor, ExecEngineDesc) or descriptor.name != name:
      _fail(f"{where} has an invalid descriptor mapping")
    effects: list[tuple[ExecMemoryView, bool]] = []
    _nonempty(descriptor.name, f"{where}.descriptors[{name!r}].name")
    _nonempty(descriptor.kind, f"{where}.descriptors[{name!r}].kind")
    _nonempty(descriptor.op, f"{where}.descriptors[{name!r}].op")
    if descriptor.kind == "MFE" and descriptor.op in ("load", "store"):
      if descriptor.params or descriptor.transfer is None:
        _fail(f"{where}.descriptors[{name!r}] has invalid transfer fields")
      transfer = _verify_transfer(descriptor.transfer, f"{where}.descriptors[{name!r}].transfer")
      effects.extend(((transfer.src, False), (transfer.dst, True)))
    elif descriptor.kind == "MFE" and descriptor.op == "gather":
      if descriptor.transfer is not None or set(descriptor.params) != {"gather"}:
        _fail(f"{where}.descriptors[{name!r}] has invalid gather fields")
      gather = _verify_tile_gather(descriptor.params["gather"], f"{where}.descriptors[{name!r}].gather")
      effects.extend(((gather.source, False), (gather.indices, False), (gather.destination, True)))
      if has_gather:
        gather_levels.add("l1")
        gather_levels.add("l2")
      source_view = gather.source
      if source_view.base.startswith("formal:"):
        gather_scopes[_formal_index(source_view, f"{where}.descriptors[{name!r}]")] = gather.scope
    elif descriptor.kind == "MFE" and descriptor.op == "scatter":
      if descriptor.transfer is not None or set(descriptor.params) != {"scatter"}:
        _fail(f"{where}.descriptors[{name!r}] has invalid scatter fields")
      scatter = _verify_tile_scatter(
        descriptor.params["scatter"], f"{where}.descriptors[{name!r}].scatter"
      )
      effects.extend(((scatter.source, False), (scatter.indices, False), (scatter.destination, True)))
      destination_view = scatter.destination
      if destination_view.base.startswith("formal:"):
        gather_scopes[
          _formal_index(destination_view, f"{where}.descriptors[{name!r}]")
        ] = scatter.scope
    elif descriptor.kind == "EVU":
      if descriptor.transfer is not None or not isinstance(descriptor.params, Mapping):
        _fail(f"{where}.descriptors[{name!r}] has invalid EVU fields")
      required = {"ops"} | ({"bytes", "exponent"} if descriptor.op == "pow" else set())
      if set(descriptor.params) != required:
        _fail(f"{where}.descriptors[{name!r}] has invalid EVU parameters")
      for key in required:
        _uint(descriptor.params[key], f"{where}.descriptors[{name!r}].{key}", positive=key != "exponent")
    elif descriptor.kind == "BOA":
      if descriptor.transfer is not None or not {"m", "n", "k", "ops"} <= set(descriptor.params) <= {
        "m",
        "n",
        "k",
        "ops",
        "accumulate",
      }:
        _fail(f"{where}.descriptors[{name!r}] has invalid BOA fields")
      for key in ("m", "n", "k", "ops"):
        _uint(descriptor.params[key], f"{where}.descriptors[{name!r}].{key}", positive=True)
      if "accumulate" in descriptor.params and type(descriptor.params["accumulate"]) is not bool:
        _fail(f"{where}.descriptors[{name!r}].accumulate must be bool")
    else:
      _fail(f"{where}.descriptors[{name!r}] has unsupported engine opcode")
    for view, writing in effects:
      if view.base.startswith("formal:"):
        formal_index = _formal_index(view, f"{where}.descriptors[{name!r}]")
        if formal_index >= len(program.formals):
          _fail(f"{where}.descriptors[{name!r}] references an absent formal")
        formal = program.formals[formal_index]
        if view.space != formal.space or view.dtype != formal.dtype or view.backing_dims != formal.dims:
          _fail(f"{where}.descriptors[{name!r}] does not match its formal")
        if formal.space == "l2":
          (l2_writes if writing else l2_reads).append(formal_index)
        elif formal.space == "global":
          if writing:
            # Only the Scatter destination may target a global formal; the
            # Scatter path is validated in full by _verify_tile_scatter.
            if not isinstance(descriptor, ExecEngineDesc) or descriptor.op != "scatter":
              _fail(f"{where} tile descriptor writes a global formal")
            global_writes.append(formal_index)
            scatter_formals.add(formal_index)
          else:
            global_reads.append(formal_index)
      elif view.space == "l1":
        if view.base not in l1_buffers:
          _fail(f"{where}.descriptors[{name!r}] references an absent L1 buffer")
        buffer = l1_buffers[view.base]
        if view.dtype != buffer.dtype or view.backing_dims != buffer.dims:
          _fail(f"{where}.descriptors[{name!r}] has an incompatible L1 view")
      else:
        _fail(f"{where}.descriptors[{name!r}] has an unbound memory view")
    descriptor_views[name] = tuple(effects)

  _verify_cache_contract(program, registry, where)
  _verify_tile_instructions(program, descriptor_views, layout, where)
  return _ProgramEffects(
    FrozenMap(descriptor_views),
    tuple(dict.fromkeys(l2_reads)),
    tuple(dict.fromkeys(l2_writes)),
    tuple(dict.fromkeys(global_reads)),
    tuple(dict.fromkeys(global_writes)),
    bool(l1_buffers or descriptor_views),
    frozenset(gather_levels),
    frozenset(scatter_formals),
    FrozenMap(gather_scopes),
  )


def _verify_buffer(buffer: ExecL1Buffer | ExecL2Buffer, where: str) -> None:
  name = buffer.name if isinstance(buffer, ExecL1Buffer) else buffer.slot
  _nonempty(name, f"{where}.id")
  if not buffer.dims or any(type(dim) is not int or dim <= 0 for dim in buffer.dims):
    _fail(f"{where}.dims must be positive")
  if buffer.dtype not in _DTYPE_BYTES or buffer.element_bytes != _DTYPE_BYTES.get(buffer.dtype):
    _fail(f"{where} has invalid dtype width")
  if buffer.bytes != _product(buffer.dims) * buffer.element_bytes:
    _fail(f"{where}.bytes does not match shape")
  _uint(buffer.alignment, f"{where}.alignment", positive=True)
  if isinstance(buffer, ExecL2Buffer) and (
    buffer.role not in ("in", "out", "inout")
    or buffer.sharing not in ("private", "readonly", "context-local")
  ):
    _fail(f"{where} has invalid L2 role or sharing mode")


def _verify_cache_requirement(
  requirement: CacheRequirement | None,
  used: bool,
  bypass_needed: bool,
  required_needed: bool,
  profile: MemoryProfile,
  where: str,
) -> None:
  if requirement is not None:
    if used and requirement.access == "none":
      _fail(f"{where} declares cache access=none but executable code uses the cache")
    if required_needed and not requirement.required:
      _fail(f"{where} understates an outcome-required cache access")
    if requirement.required and profile.cache_bytes == 0:
      _fail(f"{where} requires a cache unavailable in profile {profile.mode}")
    if requirement.access == "read_write" and profile.cache_write_policy != "write_back":
      _fail(f"{where} requires write-back cache capability")
  if used and profile.cache_bytes == 0:
    _fail(f"{where} has an actual cache access under a zero-capacity profile")
  if bypass_needed and (
    "bypass" not in profile.maintenance_caps
    or (requirement is not None and requirement.bypass == "forbidden")
  ):
    _fail(f"{where} has no permitted bypass for its disabled-cache path")


def _cache_path_requirements(
  gathers: Sequence[ExecTileGatherDesc], profile: MemoryProfile, where: str
) -> tuple[bool, bool, bool]:
  """Return actual cache use, bypass, and required-cache facts.

  Plan §2: with an address-driven Gather, a program containing a Gather
  uses any level whose profile enables a cache; a disabled level is a
  bypass path.  Scatter never touches the cache and never needs a bypass.
  """

  del where
  used = bool(gathers) and profile.cache_bytes > 0
  bypass_needed = bool(gathers) and profile.cache_bytes == 0
  return used, bypass_needed, False


def _verify_cache_contract(program: ExecTileProgram, registry: ProfileRegistry, where: str) -> None:
  contract = program.resource_contract
  if not isinstance(contract, TileResources):
    _fail(f"{where} is missing TileResources")
  gathers = tuple(
    gather
    for descriptor in program.descriptors.values()
    if isinstance(gather := descriptor.params.get("gather"), ExecTileGatherDesc)
  )
  for mode in contract.allowed_profiles:
    profile = registry.profile("l1", mode)
    used, bypass_needed, required_needed = _cache_path_requirements(gathers, profile, where)
    _verify_cache_requirement(
      contract.l1_cache, used, bypass_needed, required_needed, profile, f"{where}.l1_cache"
    )


def _verify_parent_l2_capability(
  program: ExecTileProgram, context: ContextResources, profile: MemoryProfile, where: str
) -> None:
  tile_contract = program.resource_contract
  if not isinstance(tile_contract, TileResources):
    _fail(f"{where} Tile Program is missing TileResources")
  gathers = tuple(
    gather
    for descriptor in program.descriptors.values()
    if isinstance(gather := descriptor.params.get("gather"), ExecTileGatherDesc)
  )
  used, bypass_needed, required_needed = _cache_path_requirements(gathers, profile, where)
  _verify_cache_requirement(
    tile_contract.l2_cache, used, bypass_needed, required_needed, profile, f"{where}.tile_l2_cache"
  )
  _verify_cache_requirement(
    context.l2_cache, used, bypass_needed, required_needed, profile, f"{where}.context_l2_cache"
  )


def _verify_tile_instructions(
  program: ExecTileProgram,
  descriptors: Mapping[str, tuple[tuple[ExecMemoryView, bool], ...]],
  layout: ArenaLayout,
  where: str,
) -> None:
  layout_by_id = {item.buffer_id: item for item in layout.buffer_layouts}
  live: dict[str, BufferLayout] = {}
  allocated: set[str] = set()
  freed: set[str] = set()
  produced: set[str] = set()
  awaited: set[str] = set()
  pending_by_buffer: dict[str, set[str]] = {name: set() for name in layout_by_id}
  opaque_events: set[str] = set()
  input_signal = output_signal = 0
  instruction_ids: set[str] = set()
  ret_count = 0
  for index, inst in enumerate(program.insts):
    item_where = f"{where}.insts[{index}]"
    instruction_id, _ = _instruction_identity(inst, item_where)
    if instruction_id in instruction_ids:
      _fail(f"{where} has duplicate tile instruction_id {instruction_id!r}")
    instruction_ids.add(instruction_id)
    if not isinstance(inst.op, ExecTileOp):
      _fail(f"{item_where} has an invalid opcode")
    if inst.op is ExecTileOp.ALLOC_L1:
      if len(inst.args) != 2 or not isinstance(inst.args[0], str) or type(inst.args[1]) is not int:
        _fail(f"{item_where} ALLOC_L1 requires (buffer_id, layout_index)")
      buffer_id, layout_index = inst.args
      if buffer_id not in layout_by_id or not 0 <= layout_index < len(layout.buffer_layouts):
        _fail(f"{item_where} references an invalid L1 layout")
      item = layout.buffer_layouts[layout_index]
      if item.buffer_id != buffer_id or buffer_id in allocated:
        _fail(f"{item_where} has inconsistent or repeated L1 allocation")
      for other in live.values():
        if _segments_overlap(item, other):
          _fail(f"{item_where} overlaps a live L1 view")
      allocated.add(buffer_id)
      live[buffer_id] = item
    elif inst.op in (
      ExecTileOp.LAUNCH_MFE,
      ExecTileOp.LAUNCH_GATHER,
      ExecTileOp.LAUNCH_SCATTER,
      ExecTileOp.LAUNCH_EVU,
      ExecTileOp.LAUNCH_BOA,
    ):
      launch_event = inst.dst
      if len(inst.args) != 1 or inst.args[0] not in descriptors or not launch_event:
        _fail(f"{item_where} launch has an invalid descriptor or event")
      if launch_event in produced:
        _fail(f"{where} has duplicate tile event {launch_event!r}")
      descriptor = program.descriptors[inst.args[0]]
      expected_op = {
        "MFE": (
          ExecTileOp.LAUNCH_GATHER
          if descriptor.op == "gather"
          else ExecTileOp.LAUNCH_SCATTER
          if descriptor.op == "scatter"
          else ExecTileOp.LAUNCH_MFE
        ),
        "EVU": ExecTileOp.LAUNCH_EVU,
        "BOA": ExecTileOp.LAUNCH_BOA,
      }[descriptor.kind]
      if inst.op is not expected_op:
        _fail(f"{item_where} launch opcode does not match descriptor kind")
      for view, _ in descriptors[inst.args[0]]:
        if view.space == "l1":
          if view.base not in live:
            _fail(f"{item_where} accesses an L1 buffer outside its live interval")
          pending_by_buffer[view.base].add(launch_event)
      if descriptor.kind in ("EVU", "BOA"):
        opaque_events.add(launch_event)
      produced.add(launch_event)
    elif inst.op in (ExecTileOp.WAIT, ExecTileOp.WAITALL):
      if not inst.args or (inst.op is ExecTileOp.WAIT and len(inst.args) != 1):
        _fail(f"{item_where} has invalid wait arity")
      _unique(inst.args, f"{item_where}.args")
      if any(not isinstance(event, str) or event not in produced for event in inst.args):
        _fail(f"{item_where} waits for an undefined tile event")
      awaited.update(inst.args)
    elif inst.op is ExecTileOp.SIGNAL_PHASE:
      if len(inst.args) != 2 or inst.args[0] not in ("input_released", "output_ready") or inst.args[1] != 0:
        _fail(f"{item_where} has an invalid signal")
      if inst.args[0] == "input_released":
        input_signal += 1
        load_events = {
          event
          for name, views in descriptors.items()
          if program.descriptors[name].op == "load"
          for event in _descriptor_launch_events(program, name)
        }
        if not load_events <= awaited:
          _fail(f"{item_where} signals input release before all loads complete")
      else:
        output_signal += 1
        store_events = {
          event
          for name in descriptors
          if program.descriptors[name].op == "store"
          for event in _descriptor_launch_events(program, name)
        }
        if not store_events <= awaited:
          _fail(f"{item_where} signals output ready before all stores complete")
    elif inst.op is ExecTileOp.FREE_L1:
      if len(inst.args) != 1 or inst.args[0] not in live:
        _fail(f"{item_where} frees an absent or already freed L1 view")
      buffer_id = inst.args[0]
      if not pending_by_buffer[buffer_id] <= awaited or not opaque_events <= awaited:
        _fail(f"{item_where} precedes completion of an L1 access")
      live.pop(buffer_id)
      freed.add(buffer_id)
    elif inst.op is ExecTileOp.RET:
      if inst.args or index != len(program.insts) - 1:
        _fail(f"{item_where} must be the unique terminal RET")
      # RET retires all issued work and remaining views. Explicit FREE is
      # required only for an earlier buffer lifetime end/reuse.
      freed.update(live)
      live.clear()
      ret_count += 1
    else:
      _fail(f"{item_where} uses an opcode not emitted by the formal compiler")
    if inst.label is not None or not isinstance(inst.comment, str):
      _fail(f"{item_where} has invalid label/comment metadata")
  if ret_count != 1:
    _fail(f"{where} requires exactly one terminal RET")
  if allocated != set(layout_by_id) or freed != set(layout_by_id) or live:
    _fail(f"{where} L1 allocation lifetime is incomplete")
  has_l2_reads = any(
    view.space == "l2" and not writing for views in descriptors.values() for view, writing in views
  )
  has_l2_writes = any(
    view.space == "l2" and writing for views in descriptors.values() for view, writing in views
  )
  if input_signal != int(has_l2_reads) or output_signal != int(has_l2_writes):
    _fail(f"{where} has incomplete or duplicate L2 lifetime signals")


def _descriptor_launch_events(program: ExecTileProgram, descriptor_name: str) -> tuple[str, ...]:
  return tuple(
    inst.dst for inst in program.insts if inst.args == (descriptor_name,) and inst.dst is not None
  )


@dataclass(frozen=True)
class _Access:
  formal_index: int
  start: int
  end: int
  writing: bool
  event: str
  action_index: int
  cache_levels: frozenset[str] = frozenset()
  scope: str | None = None
  precise_writes: bool = False


@dataclass(frozen=True)
class _TaskSummary:
  task: ExecTileGroupTask
  accesses: tuple[_Access, ...]
  uses_l1: bool
  uses_l2: bool


def _binding_effects(
  binding: ExecTileRoleBinding, effects: _ProgramEffects, where: str
) -> tuple[
  tuple[str, ...],
  tuple[str, ...],
  tuple[tuple[ExecMemoryView, frozenset[str], str | None], ...],
  tuple[tuple[ExecMemoryView, str | None], ...],
]:
  program = binding.tile_program
  l2_formals = [index for index, formal in enumerate(program.formals) if formal.space == "l2"]
  global_formals = [index for index, formal in enumerate(program.formals) if formal.space == "global"]
  if len(binding.actuals) != len(l2_formals) or len(binding.global_actuals) != len(global_formals):
    _fail(f"{where} actual arity does not match tile program formals")
  l2_map = dict(zip(l2_formals, binding.actuals))
  global_map = dict(zip(global_formals, binding.global_actuals))
  reads = tuple(dict.fromkeys(l2_map[index] for index in effects.l2_reads))
  writes = tuple(dict.fromkeys(l2_map[index] for index in effects.l2_writes))
  if set(binding.read_actuals) != set(reads) or set(binding.write_actuals) != set(writes):
    _fail(f"{where} read/write actual metadata disagrees with executable descriptors")
  global_reads = []
  for index in effects.global_reads:
    view = _verify_view(global_map[index], f"{where}.global_actual", spaces={"global"})
    global_reads.append((view, effects.gather_levels, effects.gather_scopes.get(index)))
  global_writes = []
  for index in effects.global_writes:
    view = _verify_view(global_map[index], f"{where}.global_actual", spaces={"global"})
    scope = effects.gather_scopes.get(index)
    global_writes.append((view, scope))
  return reads, writes, tuple(global_reads), tuple(global_writes)


def _rectangles_cover_shape(
  rectangles: Sequence[tuple[tuple[int, int], ...]], dims: tuple[int, ...]
) -> bool:
  if not rectangles:
    return False
  unique = tuple(dict.fromkeys(rectangles))
  for rectangle in unique:
    if len(rectangle) != len(dims) or any(
      start < 0 or end <= start or end > dim
      for (start, end), dim in zip(rectangle, dims)
    ):
      return False

  def covered(boxes: tuple[tuple[tuple[int, int], ...], ...], axis: int) -> int:
    if axis == len(dims):
      return 1 if boxes else 0
    edges = {0, dims[axis]}
    for box in boxes:
      edges.update(box[axis])
    ordered_edges = sorted(edges)
    total = 0
    for start, end in pairwise(ordered_edges):
      if start == end:
        continue
      active = tuple(box for box in boxes if box[axis][0] <= start and box[axis][1] >= end)
      total += (end - start) * covered(active, axis + 1)
    return total

  return covered(unique, 0) == math.prod(dims)


def _writer_rectangles(
  binding: ExecTileRoleBinding, effects: _ProgramEffects, slot: str, where: str
) -> tuple[tuple[tuple[int, int], ...], ...]:
  program = binding.tile_program
  l2_slots = dict(
    zip(
      (i for i, formal in enumerate(program.formals) if formal.space == "l2"), binding.actuals
    )
  )
  domain = binding.task_domain
  if domain is None:
    _fail(f"{where} writer has no execution range")
  rectangles: list[tuple[tuple[int, int], ...]] = []
  launched = {
    inst.args[0]
    for inst in program.insts
    if inst.op
    in (
      ExecTileOp.LAUNCH_MFE,
      ExecTileOp.LAUNCH_GATHER,
      ExecTileOp.LAUNCH_SCATTER,
      ExecTileOp.LAUNCH_EVU,
      ExecTileOp.LAUNCH_BOA,
    )
  }
  for descriptor_name in launched:
    for view, writing in effects.descriptor_views[descriptor_name]:
      if not writing or view.space != "l2":
        continue
      formal_index = _formal_index(view, f"{where}.{descriptor_name}")
      if l2_slots.get(formal_index) != slot:
        continue
      rectangle = []
      for axis, (offset, size) in enumerate(zip(view.offsets, view.dims)):
        if view.task_dim == axis:
          start = offset + domain.from_task
          end = offset + domain.to_task - 1 + size
        else:
          start = offset
          end = offset + size
        rectangle.append((start, end))
      rectangles.append(tuple(rectangle))
  return tuple(rectangles)


def _padded_l2_bytes_per_bank(task: ExecTileGroupTask, slot: str, where: str) -> int:
  if task.layout is None:
    _fail(f"{where} has no local L2 layout")
  allocation = next((item for item in task.layout.buffer_layouts if item.buffer_id == slot), None)
  buffer = next((item for item in task.l2_buffers if item.slot == slot), None)
  if (
    allocation is None
    or buffer is None
    or allocation.logical_bytes != buffer.bytes
    or allocation.banks != len(task.layout.per_bank_bytes)
  ):
    _fail(f"{where} has no matching L2 layout for shared slot {slot!r}")
  round_bytes = allocation.stripe_bytes * allocation.banks
  padded_bytes = -(-allocation.logical_bytes // round_bytes) * round_bytes
  if round_bytes > _UINT64_MAX or padded_bytes > _UINT64_MAX:
    _fail(f"{where} shared L2 padded span exceeds uint64")
  return padded_bytes // allocation.banks


def _event_ancestors(
  dependencies: Sequence[str], ancestors: Mapping[str, frozenset[str]]
) -> frozenset[str]:
  result = set(dependencies)
  for event in dependencies:
    result.update(ancestors[event])
  return frozenset(result)


def _ordered(event: str, dependencies: Sequence[str], ancestors: Mapping[str, frozenset[str]]) -> bool:
  return event in _event_ancestors(dependencies, ancestors)


def _verify_profile_desc(
  desc: object,
  where: str,
  registry: ProfileRegistry,
  level: str,
  current_mode: int,
  waits: Mapping[str, frozenset[str]],
  ancestors: Mapping[str, frozenset[str]],
  expected_binding_id: str,
) -> int:
  if not isinstance(desc, ProfileReconfigDesc):
    _fail(f"{where} is missing ProfileReconfigDesc")
  _nonempty(desc.command_id, f"{where}.command_id")
  _uint(desc.expected_mode, f"{where}.expected_mode")
  _uint(desc.target_mode, f"{where}.target_mode")
  if (
    desc.level != level
    or desc.steps != PROFILE_STEPS
    or desc.registry_hash != registry.registry_hash
    or tuple(desc.affected_domains) != (level,)
  ):
    _fail(f"{where} has an invalid profile sequence, level, domain, or Registry")
  if desc.expected_mode != current_mode or desc.target_mode == current_mode:
    _fail(f"{where} profile transition does not match the current mode")
  target = registry.profile(level, desc.target_mode)
  if tuple(desc.member_ids) != target.member_ids:
    _fail(f"{where} does not name every affected profile member")
  _unique(desc.wait_instruction_ids, f"{where}.wait_instruction_ids")
  _unique(desc.frontier, f"{where}.frontier")
  if desc.exclusive_binding_id != expected_binding_id:
    _fail(f"{where} has the wrong exclusive binding")
  _verify_source_ref(desc.source_ref, f"{where}.command")
  if not desc.frontier and (
    desc.wait_instruction_ids or desc.source_ref.reason != "verified initialization proof"
  ):
    _fail(f"{where} empty frontier is not backed by the launch initialization proof")
  waited: set[str] = set()
  for wait_id in desc.wait_instruction_ids:
    if wait_id not in waits:
      _fail(f"{where} references a missing or subsequent await instruction")
    waited.update(waits[wait_id])
    for event in waits[wait_id]:
      waited.update(ancestors[event])
  if not set(desc.frontier) <= waited:
    _fail(f"{where} frontier is not covered by its ordinary awaits")
  return desc.target_mode


def _verify_maintenance_desc(
  desc: object, where: str, inputs: Sequence[ExecGlobalInput], produced: set[str]
) -> MemoryMaintenanceDesc:
  if not isinstance(desc, MemoryMaintenanceDesc) or desc.steps != MAINTENANCE_STEPS:
    _fail(f"{where} is missing the complete memory-maintenance sequence")
  _nonempty(desc.command_id, f"{where}.command_id")
  _verify_source_ref(desc.source_ref, f"{where}.command")
  if not desc.levels or not set(desc.levels) <= {"l1", "l2"} or len(desc.levels) != len(set(desc.levels)):
    _fail(f"{where} has invalid maintenance levels")
  if not desc.ranges:
    _fail(f"{where} has no maintenance ranges")
  if len(desc.dependencies) != len(set(desc.dependencies)) or not set(desc.dependencies) <= produced:
    _fail(f"{where} has invalid maintenance dependencies")
  for index, item in enumerate(desc.ranges):
    if type(item.input_index) is not int or not 0 <= item.input_index < len(inputs):
      _fail(f"{where}.ranges[{index}] has invalid input index")
    _uint(item.offset, f"{where}.ranges[{index}].offset")
    _uint(item.bytes, f"{where}.ranges[{index}].bytes", positive=True)
    if item.offset + item.bytes > inputs[item.input_index].size_bytes:
      _fail(f"{where}.ranges[{index}] exceeds its global input")
    if (
      not item.levels
      or len(item.levels) != len(set(item.levels))
      or not set(item.levels) <= set(desc.levels)
    ):
      _fail(f"{where}.ranges[{index}] has invalid levels")
  return desc


def _maintenance_covers(
  commands: Sequence[tuple[int, MemoryMaintenanceDesc]],
  producer: _Access,
  consumer: _Access,
  required_levels: frozenset[str],
  ancestors: Mapping[str, frozenset[str]],
) -> bool:
  for index, desc in commands:
    if not producer.action_index < index < consumer.action_index:
      continue
    if not _ordered(producer.event, desc.dependencies, ancestors):
      continue
    for item in desc.ranges:
      if (
        item.input_index == consumer.formal_index
        and item.offset <= consumer.start
        and item.offset + item.bytes >= consumer.end
        and required_levels <= set(item.levels)
      ):
        return True
  return False


def _maintenance_precedes(commands: Sequence[tuple[int, MemoryMaintenanceDesc]], access: _Access) -> bool:
  for index, desc in commands:
    if index >= access.action_index:
      continue
    for item in desc.ranges:
      if (
        item.input_index == access.formal_index
        and item.offset <= access.start
        and item.offset + item.bytes >= access.end
        and access.cache_levels <= set(item.levels)
      ):
        return True
  return False


def _verify_task(
  task: ExecTileGroupTask,
  call: CallBinding,
  registry: ProfileRegistry,
  hw: HardwareConfig,
  sim: SimConfig,
  where: str,
  identities: dict[int, tuple[int, int]],
) -> _TaskSummary:
  if not isinstance(task, ExecTileGroupTask):
    _fail(f"{where} is not an ExecTileGroupTask")
  _nonempty(task.name, f"{where}.name")
  if task.binding_id != call.binding_id:
    _fail(f"{where}.binding_id disagrees with call binding")
  if not isinstance(task.resource_contract, ContextResources):
    _fail(f"{where} is missing ContextResources")
  contract = task.resource_contract
  if contract.l2_mode != call.requested_l2_mode or call.resolved_l2_mode not in contract.allowed_profiles:
    _fail(f"{where} requested/resolved L2 profile violates its resource contract")
  if contract.requested_contexts_per_tile > sim.context_count:
    _fail(f"{where} requests more Tile contexts than the target provides")
  profile = registry.profile("l2", call.resolved_l2_mode)
  for mode in contract.allowed_profiles:
    allowed = registry.profile("l2", mode)
    if contract.l2_spm_bytes > allowed.user_spm_bytes:
      _fail(f"{where} L2 contract exceeds allowed profile {mode}")
    _verify_cache_requirement(contract.l2_cache, False, False, False, allowed, f"{where}.context_l2_cache")
  inputs = tuple(task.global_inputs)
  _unique(tuple(item.name for item in inputs), f"{where}.global_inputs")
  for index, item in enumerate(inputs):
    _verify_global_input(item, f"{where}.global_inputs[{index}]")
  input_index = {item.name: (index, item) for index, item in enumerate(inputs)}
  buffers: dict[str, ExecL2Buffer] = {}
  for index, buffer in enumerate(task.l2_buffers):
    if not isinstance(buffer, ExecL2Buffer) or buffer.slot in buffers:
      _fail(f"{where} has an invalid or duplicate L2 buffer")
    _verify_buffer(buffer, f"{where}.l2_buffers[{index}]")
    buffers[buffer.slot] = buffer
  imports: dict[str, ExecSharedInput] = {}
  imported_backings: set[tuple[str, str]] = set()
  for index, shared in enumerate(task.shared_inputs):
    if not isinstance(shared, ExecSharedInput) or shared.slot in imports or shared.slot in buffers:
      _fail(f"{where}.shared_inputs[{index}] is invalid or duplicates an L2 formal")
    _nonempty(shared.slot, f"{where}.shared_inputs[{index}].slot")
    _nonempty(shared.producer_binding_id, f"{where}.shared_inputs[{index}].producer_binding_id")
    _nonempty(shared.producer_slot, f"{where}.shared_inputs[{index}].producer_slot")
    if (
      not shared.dims
      or any(type(dim) is not int or dim <= 0 for dim in shared.dims)
      or shared.dtype not in _DTYPE_BYTES
      or shared.element_bytes != _DTYPE_BYTES[shared.dtype]
      or shared.bytes != _product(shared.dims) * shared.element_bytes
    ):
      _fail(f"{where}.shared_inputs[{index}] has inconsistent shape or dtype")
    backing = (shared.producer_binding_id, shared.producer_slot)
    if backing in imported_backings:
      _fail(f"{where} imports one producer backing more than once")
    imported_backings.add(backing)
    imports[shared.slot] = shared
  buffer_roles = {slot: buffer.role for slot, buffer in buffers.items()}
  buffer_roles.update(dict.fromkeys(imports, "in"))
  buffer_shapes = {
    slot: (buffer.dims, buffer.dtype, buffer.element_bytes, buffer.bytes)
    for slot, buffer in buffers.items()
  }
  buffer_shapes.update(
    {
      slot: (shared.dims, shared.dtype, shared.element_bytes, shared.bytes)
      for slot, shared in imports.items()
    }
  )
  all_slots = set(buffer_shapes)
  layout = _verify_layout(task.layout, f"{where}.layout", profile, buffers)
  if layout.reserved_bytes > contract.l2_spm_bytes:
    _fail(f"{where} L2 layout exceeds its declared resource contract")
  _verify_l2_no_rebind_layout(layout, f"{where}.layout")

  streams: dict[int, ExecStreamDesc] = {}
  for index, stream in enumerate(task.streams):
    if not isinstance(stream, ExecStreamDesc) or stream.queue_id in streams:
      _fail(f"{where}.streams[{index}] is invalid or duplicated")
    for name in ("queue_id", "payload_slot_id", "pmu_stream_id"):
      _uint(getattr(stream, name), f"{where}.streams[{index}].{name}")
    _uint(stream.depth, f"{where}.streams[{index}].depth", positive=True)
    _uint(stream.token_stride, f"{where}.streams[{index}].token_stride", positive=True)
    if stream.depth > sim.group.inflight_capacity:
      _fail(f"{where}.streams[{index}] exceeds finite stream capacity")
    max_mask = (1 << profile.pools) - 1
    if not 0 < stream.producer_mask <= max_mask or not 0 < stream.consumer_mask <= max_mask:
      _fail(f"{where}.streams[{index}] has invalid producer/consumer mask")
    streams[stream.queue_id] = stream
  known_queues = set(streams)
  known_queues.update(
    action.args[0]
    for action in task.actions
    if action.op is ExecGroupActionOp.INIT_STREAM and len(action.args) == 4 and type(action.args[0]) is int
  )

  programs: dict[int, _ProgramEffects] = {}
  role_bindings: dict[int, tuple[ExecTileRoleBinding, _ProgramEffects]] = {}
  for role_id, binding in task.role_bindings.items():
    if (
      type(role_id) is not int or not isinstance(binding, ExecTileRoleBinding) or binding.role_id != role_id
    ):
      _fail(f"{where} has an invalid role binding map")
    if (
      role_id < 0
      or binding.tile_mask <= 0
      or binding.tile_mask >> registry.profile("l1", call.entry_l1_mode).pools
    ):
      _fail(f"{where}.role_bindings[{role_id}] has invalid identity or placement")
    if binding.context_id is not None and (
      type(binding.context_id) is not int or not 0 <= binding.context_id < sim.context_count
    ):
      _fail(f"{where}.role_bindings[{role_id}] has invalid context pin")
    task_domain = binding.task_domain
    if task_domain is None or not 0 <= task_domain.from_task < task_domain.to_task:
      _fail(f"{where}.role_bindings[{role_id}] has an invalid task range")
    if task_domain.to_task - task_domain.from_task != binding.tile_mask.bit_count():
      _fail(f"{where}.role_bindings[{role_id}] task count differs from placement popcount")
    program = binding.tile_program
    l2_formals = [formal for formal in program.formals if formal.space == "l2"]
    global_formals = [formal for formal in program.formals if formal.space == "global"]
    if len(binding.actuals) != len(l2_formals) or len(binding.global_actuals) != len(global_formals):
      _fail(f"{where}.role_bindings[{role_id}] actual arity differs from program formals")
    for formal, slot in zip(l2_formals, binding.actuals):
      actual_shape = buffer_shapes.get(slot)
      if actual_shape is None or actual_shape[0] != formal.dims or actual_shape[1] != formal.dtype:
        _fail(f"{where}.role_bindings[{role_id}] has incompatible L2 actual")
    import_actuals = [slot for slot in binding.actuals if slot in imports]
    if len(import_actuals) != len(set(import_actuals)):
      _fail(f"{where}.role_bindings[{role_id}] aliases one shared backing through multiple formals")
    for formal, view in zip(global_formals, binding.global_actuals):
      _verify_view(view, f"{where}.role_bindings[{role_id}].global_actual", spaces={"global"})
      if view.dims != formal.dims or view.dtype != formal.dtype:
        _fail(f"{where}.role_bindings[{role_id}] has incompatible global actual")
    if binding.in_stream is not None and binding.in_stream not in known_queues:
      _fail(f"{where}.role_bindings[{role_id}] references an unknown input stream")
    if binding.out_stream is not None and binding.out_stream not in known_queues:
      _fail(f"{where}.role_bindings[{role_id}] references an unknown output stream")
    effects = programs.get(id(binding.tile_program))
    if effects is None:
      effects = _verify_program(
        binding.tile_program, f"{where}.program[{binding.tile_program.name}]", registry, hw
      )
      programs[id(binding.tile_program)] = effects
      identity = (binding.tile_program.program_id, binding.tile_program.version)
      previous = identities.setdefault(binding.tile_program.program_hash, identity)
      if previous != identity:
        _fail("identical tile program content has multiple identities")
    for l2_mode in contract.allowed_profiles:
      _verify_parent_l2_capability(
        binding.tile_program, contract, registry.profile("l2", l2_mode), f"{where}.role_bindings[{role_id}]"
      )
    role_reads, role_writes, _, _ = _binding_effects(binding, effects, f"{where}.role_bindings[{role_id}]")
    for views in effects.descriptor_views.values():
      for view, _writing in views:
        if view.space != "l2" or view.task_dim is None:
          continue
        domain = binding.task_domain
        if domain is None or view.offsets[view.task_dim] + domain.to_task - 1 + view.dims[view.task_dim] > (
          view.backing_dims[view.task_dim]
        ):
          _fail(f"{where}.role_bindings[{role_id}] task view exceeds its L2 formal")
    valid_slots = set(buffer_shapes)
    if any(slot not in valid_slots for slot in (*binding.actuals, *role_reads, *role_writes)):
      _fail(f"{where}.role_bindings[{role_id}] references an unknown L2 buffer")
    if set(role_writes) & set(imports):
      _fail(f"{where}.role_bindings[{role_id}] writes through a readonly shared import")
    if any(buffer_roles[slot] == "in" for slot in role_writes):
      _fail(f"{where}.role_bindings[{role_id}] writes a role=in L2 buffer")
    role_bindings[role_id] = (binding, effects)

  logical_tasks = 0
  dispatch_ordinals: set[int] = set()
  produced: set[str] = set()
  outstanding: set[str] = set()
  ancestors: dict[str, frozenset[str]] = {}
  waits: dict[str, frozenset[str]] = {}
  instruction_ids: set[str] = set()
  child_l1_modes: set[int] = set()
  for role_binding, _ in role_bindings.values():
    tile_contract = role_binding.tile_program.resource_contract
    if not isinstance(tile_contract, TileResources):
      _fail(f"{where} child Tile Program is missing its resource contract")
    child_l1_modes.update(tile_contract.allowed_profiles)
  if not child_l1_modes:
    child_l1_modes.update((call.entry_l1_mode, call.exit_l1_mode))
  if any(
    state.l2_mode not in contract.allowed_profiles or state.l1_mode not in child_l1_modes
    for state in call.permitted_profiles
  ):
    _fail(f"{where} call binding widens a source resource contract")
  l2_live: dict[str, BufferLayout | None] = {}
  bound: set[str] = set()
  released: set[str] = set()
  bind_events: dict[str, str] = {}
  accesses: list[_Access] = []
  maintenance: list[tuple[int, MemoryMaintenanceDesc]] = []
  semantic_effects: list[tuple[tuple[str, ...], tuple[str, ...], str, str]] = []
  dispatch_by_ordinal: dict[int, tuple[tuple[str, ...], tuple[str, ...], ExecDispatchRequest]] = {}
  prefetch_events: dict[str, list[str]] = {slot: [] for slot in buffer_shapes}
  store_events: dict[str, list[str]] = {slot: [] for slot in buffer_shapes}
  full_prefetch: set[str] = set()
  published: set[str] = set()
  publish_events: dict[str, str] = {}
  import_bind_count = 0
  profile_frontier: list[str] = []
  current_l1 = call.entry_l1_mode
  saw_l1_reconfig = False

  for index, action in enumerate(task.actions):
    action_where = f"{where}.actions[{index}]"
    instruction_id, _ = _instruction_identity(action, action_where)
    if instruction_id in instruction_ids:
      _fail(f"{where} has duplicate group instruction_id {instruction_id!r}")
    instruction_ids.add(instruction_id)
    if not isinstance(action.op, ExecGroupActionOp):
      _fail(f"{action_where} has invalid opcode")
    if not isinstance(action.comment, str):
      _fail(f"{action_where}.comment must be a string")
    if (
      len(action.dependencies) != len(set(action.dependencies)) or not set(action.dependencies) <= produced
    ):
      _fail(f"{action_where} has duplicate, unknown, or forward dependencies")
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    read_done = action.dst or ""
    write_done = action.dst or ""
    action_accesses: list[_Access] = []

    if action.op is ExecGroupActionOp.BIND_L2_IMPORT:
      if (
        index != import_bind_count
        or import_bind_count >= len(task.shared_inputs)
        or len(action.args) != 2
        or action.args[1] != import_bind_count
        or action.args[0] != task.shared_inputs[import_bind_count].slot
        or not action.dst
        or action.dependencies
        or action.reads
        or action.writes
      ):
        _fail(f"{action_where} has an invalid or misplaced shared import bind")
      slot = action.args[0]
      if slot in bound:
        _fail(f"{action_where} binds shared import {slot!r} more than once")
      bound.add(slot)
      bind_events[slot] = action.dst
      l2_live[slot] = None
      import_bind_count += 1
    elif action.op is ExecGroupActionOp.BIND_L2_VIEW:
      if len(action.args) != 2 or not isinstance(action.args[0], str) or type(action.args[1]) is not int:
        _fail(f"{action_where} BIND_L2_VIEW requires (buffer_id, layout_index)")
      buffer_id, layout_index = action.args
      bind_done = action.dst
      if (
        buffer_id not in buffers
        or buffer_id in bound
        or not bind_done
        or not 0 <= layout_index < len(layout.buffer_layouts)
      ):
        _fail(f"{action_where} references an invalid L2 layout or completion event")
      item = layout.buffer_layouts[layout_index]
      if item.buffer_id != buffer_id:
        _fail(f"{action_where} layout index does not match buffer")
      bound.add(buffer_id)
      bind_events[buffer_id] = bind_done
      l2_live[buffer_id] = item
    elif action.op in (ExecGroupActionOp.DMA_PREFETCH, ExecGroupActionOp.DMA_STORE):
      if len(action.args) != 2 or not isinstance(action.args[0], str):
        _fail(f"{action_where} has invalid DMA arguments")
      transfer = _verify_transfer(action.args[1], f"{action_where}.transfer")
      dma_done = action.dst
      if not dma_done:
        _fail(f"{action_where} DMA has no completion event")
      if action.op is ExecGroupActionOp.DMA_PREFETCH:
        if transfer.src.space != "global" or transfer.dst.space != "l2":
          _fail(f"{action_where} prefetch must copy global to L2")
        slot = transfer.dst.base
        if slot not in buffers or slot in published:
          _fail(f"{action_where} prefetch targets an absent, imported, or published L2 view")
        dims, dtype, _element_bytes, _buffer_bytes = buffer_shapes[slot]
        if transfer.dst.backing_dims != dims or transfer.dst.dtype != dtype:
          _fail(f"{action_where} prefetch destination disagrees with its L2 buffer")
        writes = (slot,)
        if slot not in l2_live:
          _fail(f"{action_where} writes an unbound or released L2 view")
        prefetch_events[slot].append(dma_done)
        if (
          transfer.dst.dims == dims
          and transfer.dst.offsets == (0,) * len(dims)
          and transfer.dst.task_dim is None
        ):
          full_prefetch.add(slot)
        action_accesses.append(_global_access(transfer.src, input_index, False, dma_done, index))
      else:
        if transfer.src.space != "l2" or transfer.dst.space != "global":
          _fail(f"{action_where} store must copy L2 to global")
        slot = transfer.src.base
        if slot not in buffer_shapes or slot in published:
          _fail(f"{action_where} store reads an absent or already published L2 view")
        dims, dtype, _element_bytes, _buffer_bytes = buffer_shapes[slot]
        if transfer.src.backing_dims != dims or transfer.src.dtype != dtype:
          _fail(f"{action_where} store source disagrees with its L2 buffer")
        reads = (slot,)
        if slot not in l2_live:
          _fail(f"{action_where} reads an unbound or released L2 view")
        store_events[slot].append(dma_done)
        writeback_levels = frozenset(
          level
          for level, mode in (("l1", current_l1), ("l2", call.resolved_l2_mode))
          if registry.profile(level, mode).cache_bytes
          and registry.profile(level, mode).cache_write_policy == "write_back"
        )
        action_accesses.append(
          _global_access(transfer.dst, input_index, True, dma_done, index, writeback_levels)
        )
    elif action.op is ExecGroupActionOp.DISPATCH_ROLE:
      if len(action.args) != 1 or not isinstance(action.args[0], ExecDispatchRequest):
        _fail(f"{action_where} has invalid dispatch request")
      request = action.args[0]
      if request.binding_id != task.binding_id or request.role_id not in role_bindings:
        _fail(f"{action_where} dispatch binding is invalid")
      _uint(request.role_id, f"{action_where}.role_id")
      _uint(request.dispatch_ordinal, f"{action_where}.dispatch_ordinal")
      _uint(request.requested_l1_mode, f"{action_where}.requested_l1_mode")
      _uint(request.resolved_l1_mode, f"{action_where}.resolved_l1_mode")
      if request.dispatch_ordinal in dispatch_ordinals or request.dispatch_ordinal < 0:
        _fail(f"{action_where} has duplicate dispatch ordinal")
      dispatch_ordinals.add(request.dispatch_ordinal)
      binding, effects = role_bindings[request.role_id]
      dispatch_domain = binding.task_domain
      if dispatch_domain is None:
        _fail(f"{action_where} dispatch role is missing its validated task range")
      logical_tasks += dispatch_domain.to_task - dispatch_domain.from_task
      reads, writes, global_reads, global_writes = _binding_effects(binding, effects, action_where)
      if any(slot not in l2_live for slot in (*reads, *writes)):
        _fail(f"{action_where} accesses an unbound or released L2 view")
      if {*reads, *writes} & published:
        _fail(f"{action_where} accesses an exported buffer after publish")
      grid_done = action.dst
      if (
        not grid_done
        or (reads and not request.input_released_event)
        or (writes and not request.output_ready_event)
      ):
        _fail(f"{action_where} dispatch lacks required lifecycle events")
      if bool(request.input_released_event) != bool(request.signal_policy.input_released) or bool(
        request.output_ready_event
      ) != bool(request.signal_policy.output_ready):
        _fail(f"{action_where} phase events differ from declared signal policy")
      tile_contract = binding.tile_program.resource_contract
      if not isinstance(tile_contract, TileResources):
        _fail(f"{action_where} Tile Program is missing its resource contract")
      if request.requested_l1_mode not in tile_contract.allowed_profiles:
        _fail(f"{action_where} requested L1 mode is outside the tile contract")
      if request.resolved_l1_mode not in tile_contract.allowed_profiles:
        _fail(f"{action_where} resolved L1 mode is outside the tile contract")
      if request.resolved_l1_mode != current_l1:
        _fail(f"{action_where} requires an L1 profile transition absent from executable control")
      if ProfileState(request.resolved_l1_mode, call.resolved_l2_mode) not in call.permitted_profiles:
        _fail(f"{action_where} uses an unpermitted cross-layer profile combination")
      read_done = request.input_released_event
      write_done = request.output_ready_event
      for view, levels, scope in global_reads:
        active_levels = frozenset(
          level
          for level in levels
          if registry.profile(level, current_l1 if level == "l1" else call.resolved_l2_mode).cache_bytes
        )
        action_accesses.append(
          _global_access(view, input_index, False, grid_done, index, active_levels, scope)
        )
      for view, scope in global_writes:
        action_accesses.append(
          _global_access(view, input_index, True, grid_done, index, frozenset(), scope, True)
        )
      dispatch_by_ordinal[request.dispatch_ordinal] = (reads, writes, request)
      profile_frontier.append(grid_done)
    elif action.op is ExecGroupActionOp.WAIT_EVENT:
      if len(action.args) != 1 or action.dependencies != tuple(action.args):
        _fail(f"{action_where} WAIT_EVENT must depend on exactly its source event")
      waits[instruction_id] = frozenset(action.args)
    elif action.op is ExecGroupActionOp.PROFILE_RECONFIG:
      if len(action.args) != 1:
        _fail(f"{action_where} has invalid profile command arguments")
      desc = action.args[0]
      if not isinstance(desc, ProfileReconfigDesc) or desc.command_id != instruction_id:
        _fail(f"{action_where} profile command identity is inconsistent")
      current_l1 = _verify_profile_desc(
        desc, action_where, registry, "l1", current_l1, waits, ancestors, task.binding_id
      )
      saw_l1_reconfig = True
      if tuple(desc.frontier) != tuple(profile_frontier):
        _fail(f"{action_where} omits a prior L1 Grid from its profile frontier")
      if tuple(desc.frontier) != action.dependencies or desc.source_ref != action.source_ref:
        _fail(f"{action_where} action and profile descriptor disagree")
      profile_frontier.clear()
    elif action.op is ExecGroupActionOp.MEMORY_MAINTENANCE:
      if len(action.args) != 1:
        _fail(f"{action_where} has invalid maintenance command arguments")
      desc = _verify_maintenance_desc(action.args[0], action_where, inputs, produced)
      if desc.command_id != instruction_id:
        _fail(f"{action_where} maintenance command identity is inconsistent")
      if tuple(desc.dependencies) != action.dependencies or desc.source_ref != action.source_ref:
        _fail(f"{action_where} action and maintenance descriptor disagree")
      maintenance.append((index, desc))
    elif action.op is ExecGroupActionOp.PUBLISH_L2:
      if len(action.args) != 1 or not isinstance(action.args[0], ExecPublishRequest):
        _fail(f"{action_where} has invalid publish request")
      publish_request = action.args[0]
      slot = publish_request.buffer_slot
      publish_buffer = buffers.get(slot)
      if (
        publish_buffer is None
        or slot not in l2_live
        or slot in published
        or publish_buffer.sharing != "readonly"
      ):
        _fail(f"{action_where} publishes an absent, private, released, or already published buffer")
      readers = tuple(
        sorted(
          ordinal
          for ordinal, (read_slots, _, _) in dispatch_by_ordinal.items()
          if slot in read_slots
        )
      )
      writers = tuple(
        sorted(
          ordinal
          for ordinal, (_, write_slots, _) in dispatch_by_ordinal.items()
          if slot in write_slots
        )
      )
      if (
        publish_request.reader_dispatch_ordinals != readers
        or publish_request.writer_dispatch_ordinals != writers
      ):
        _fail(f"{action_where} publish dispatch summary disagrees with descriptors")
      if publish_request.dependency_events != action.dependencies:
        _fail(f"{action_where} publish and action dependencies disagree")
      required = [
        bind_events[slot],
        *prefetch_events[slot],
        *store_events[slot],
      ]
      required.extend(dispatch_by_ordinal[item][2].input_released_event for item in readers)
      required.extend(dispatch_by_ordinal[item][2].output_ready_event for item in writers)
      if set(action.dependencies) != set(required):
        _fail(f"{action_where} publish dependencies do not exactly close initialization and accesses")
      if any(not event or not _ordered(event, action.dependencies, ancestors) for event in required):
        _fail(f"{action_where} does not retire every real buffer access before publish")
      rectangles = tuple(
        rectangle
        for ordinal in writers
        for binding, effects in (role_bindings[dispatch_by_ordinal[ordinal][2].role_id],)
        for rectangle in _writer_rectangles(binding, effects, slot, action_where)
      )
      if slot not in full_prefetch and not _rectangles_cover_shape(rectangles, publish_buffer.dims):
        _fail(f"{action_where} publishes a buffer without complete initialization")
      if not action.dst:
        _fail(f"{action_where} publish lacks its completion event")
      if action.reads or action.writes != (slot,):
        _fail(f"{action_where} publish must seal exactly its target L2 buffer")
      writes = (slot,)
      published.add(slot)
      publish_events[slot] = action.dst
    elif action.op is ExecGroupActionOp.RELEASE_L2:
      if len(action.args) != 1 or not isinstance(action.args[0], ExecReleaseRequest):
        _fail(f"{action_where} has invalid release request")
      release_request = action.args[0]
      slot = release_request.buffer_slot
      if slot not in buffer_shapes or slot not in l2_live:
        _fail(f"{action_where} releases an absent or already released L2 view")
      if release_request.buffer_role != buffer_roles[slot]:
        _fail(f"{action_where} release role disagrees with its L2 view")
      if release_request.dependency_events != action.dependencies:
        _fail(f"{action_where} release and action dependencies disagree")
      if slot in buffers:
        if buffers[slot].sharing == "readonly" and slot not in published:
          _fail(f"{action_where} releases a readonly export before publish")
        if buffers[slot].sharing == "private" and slot in published:
          _fail(f"{action_where} releases a private buffer after publish")
      readers = tuple(
        sorted(
          ordinal
          for ordinal, (read_slots, _, _) in dispatch_by_ordinal.items()
          if slot in read_slots
        )
      )
      writers = tuple(
        sorted(
          ordinal
          for ordinal, (_, write_slots, _) in dispatch_by_ordinal.items()
          if slot in write_slots
        )
      )
      if (
        release_request.reader_dispatch_ordinals != readers
        or release_request.writer_dispatch_ordinals != writers
      ):
        _fail(f"{action_where} release dispatch summary disagrees with descriptors")
      release_done = action.dst
      if not release_done:
        _fail(f"{action_where} release lacks its internal completion event")
      if store_events[slot] and any(
        not _ordered(
          dispatch_by_ordinal[item][2].output_ready_event,
          (store_events[slot][-1],),
          ancestors,
        )
        for item in writers
      ):
        _fail(f"{action_where} final store does not cover every actual writer")
      required = [bind_events[slot], *prefetch_events[slot], *store_events[slot]]
      required.extend(dispatch_by_ordinal[item][2].input_released_event for item in readers)
      if slot in buffers and buffers[slot].sharing == "context-local":
        required.extend(dispatch_by_ordinal[item][2].output_ready_event for item in writers)
      if slot in publish_events:
        required.append(publish_events[slot])
      if any(not event or not _ordered(event, action.dependencies, ancestors) for event in required):
        _fail(f"{action_where} does not retire every real buffer access")
      if slot in imports and writers:
        _fail(f"{action_where} releases a shared import that was written")
      if slot in buffers and buffer_roles[slot] in ("out", "inout") and not writers:
        _fail(f"{action_where} releases output without a writer")
      if (
        slot in buffers
        and buffers[slot].sharing == "private"
        and writers
        and not store_events[slot]
      ):
        _fail(f"{action_where} releases written private output without an HBM store")
      if buffer_roles[slot] == "in" and writers:
        _fail(f"{action_where} role=in buffer has a writer")
      l2_live.pop(slot)
      released.add(slot)
    elif action.op is ExecGroupActionOp.BARRIER_GROUP:
      if action.args:
        _fail(f"{action_where} barrier takes no arguments")
      if any(not _ordered(event, action.dependencies, ancestors) for event in outstanding):
        _fail(f"{action_where} barrier does not cover all outstanding work")
    elif action.op is ExecGroupActionOp.SIGNAL_EVENT:
      if len(action.args) != 1 or not isinstance(action.args[0], str):
        _fail(f"{action_where} has invalid signal event")
      if any(not _ordered(event, action.dependencies, ancestors) for event in outstanding):
        _fail(f"{action_where} completion signal does not cover all outstanding work")
    elif action.op is ExecGroupActionOp.INIT_STREAM:
      if len(action.args) != 4 or any(type(value) is not int for value in action.args):
        _fail(f"{action_where} INIT_STREAM requires integer queue/depth/masks")
      qid, depth, producer_mask, consumer_mask = action.args
      _uint(qid, f"{action_where}.queue_id")
      _uint(depth, f"{action_where}.depth", positive=True)
      if producer_mask <= 0 or consumer_mask <= 0:
        _fail(f"{action_where} stream masks must be positive")
      defined_stream = streams.get(qid)
      if defined_stream is not None and (
        defined_stream.depth != depth
        or defined_stream.producer_mask != producer_mask
        or defined_stream.consumer_mask != consumer_mask
      ):
        _fail(f"{action_where} disagrees with its stream descriptor")
    else:
      _fail(f"{action_where} uses an opcode not emitted by the formal compiler")
    for slot in (*reads, *writes):
      if slot not in bind_events or not _ordered(bind_events[slot], action.dependencies, ancestors):
        _fail(f"{action_where} can issue before L2 view {slot!r} is bound")
    semantic_effects.append((reads, writes, read_done, write_done))
    accesses.extend(action_accesses)
    outputs = list(action.output_events)
    if len(outputs) != len(set(outputs)) or any(
      not isinstance(event, str) or not event for event in outputs
    ):
      _fail(f"{action_where} has invalid output events")
    if set(outputs) & produced:
      _fail(f"{action_where} duplicates a produced event")
    base_ancestors = _event_ancestors(action.dependencies, ancestors)
    for event in outputs:
      event_ancestors = set(base_ancestors)
      if action.op is ExecGroupActionOp.DISPATCH_ROLE and event == action.dst:
        event_ancestors.update(
          phase
          for phase in (action.args[0].input_released_event, action.args[0].output_ready_event)
          if phase
        )
      ancestors[event] = frozenset(event_ancestors)
      produced.add(event)
    outstanding.update(outputs)
    outstanding.difference_update(
      {
        event
        for event in outstanding
        if any(event != later and event in ancestors[later] for later in outstanding)
      }
    )
    if action.op is ExecGroupActionOp.BARRIER_GROUP:
      outstanding.clear()
    elif action.op in (
      ExecGroupActionOp.WAIT_EVENT,
      ExecGroupActionOp.PROFILE_RECONFIG,
      ExecGroupActionOp.MEMORY_MAINTENANCE,
    ):
      outstanding.difference_update(_event_ancestors(action.dependencies, ancestors))

  if logical_tasks != contract.logical_tasks:
    _fail(f"{where} logical task budget disagrees with dispatches")
  if current_l1 != call.exit_l1_mode:
    _fail(f"{where} exits with an L1 mode different from its call binding")
  if call.requires_l1_exclusive != saw_l1_reconfig:
    _fail(f"{where} cross-root L1 exclusivity metadata is incomplete")
  if import_bind_count != len(imports) or bound != all_slots or released != all_slots or l2_live:
    _fail(f"{where} L2 view lifetime is incomplete")
  exports = {slot for slot, buffer in buffers.items() if buffer.sharing == "readonly"}
  if published != exports:
    _fail(f"{where} readonly allocations do not have exactly one publish")
  if not task.actions or task.actions[-1].op is not ExecGroupActionOp.SIGNAL_EVENT:
    _fail(f"{where} lacks a terminal completion signal")
  if task.actions[-1].args != (task.completion_event,):
    _fail(f"{where} terminal completion event is inconsistent")
  _verify_action_hazards(task.actions, semantic_effects, ancestors)
  alias_indices = (
    call.actual_inputs if len(call.actual_inputs) == len(inputs) else tuple(range(len(inputs)))
  )
  _verify_global_hazards(accesses, task.actions, ancestors, maintenance, alias_indices)
  _verify_task_budget(task, call, registry, sim, where)
  uses_l1 = bool(dispatch_by_ordinal)
  uses_l2 = bool(buffers or imports or any(reads or writes for reads, writes, _, _ in semantic_effects))
  return _TaskSummary(task, tuple(accesses), uses_l1, uses_l2)


def _global_access(
  view: ExecMemoryView,
  input_index: Mapping[str, tuple[int, ExecGlobalInput]],
  writing: bool,
  event: str,
  action_index: int,
  cache_levels: frozenset[str] = frozenset(),
  scope: str | None = None,
  precise_writes: bool = False,
) -> _Access:
  _verify_view(view, "global access", spaces={"global"})
  name = view.base.removeprefix("global:")
  if not view.base.startswith("global:") or name not in input_index:
    _fail("global view does not resolve to an executable input formal")
  formal_index, formal = input_index[name]
  if view.backing_dims != formal.dims or view.dtype != formal.dtype:
    _fail("global view shape or dtype differs from its executable input formal")
  start = _view_start(view)
  if start + view.bytes > formal.size_bytes:
    _fail("global view exceeds its executable input formal")
  return _Access(
    formal_index, start, start + view.bytes, writing, event, action_index, cache_levels, scope,
    precise_writes,
  )


def _verify_action_hazards(
  actions: Sequence[ExecGroupAction],
  effects: Sequence[tuple[tuple[str, ...], tuple[str, ...], str, str]],
  ancestors: Mapping[str, frozenset[str]],
) -> None:
  last_writer: dict[str, str] = {}
  readers: dict[str, set[str]] = {}
  for action, (reads, writes, read_done, write_done) in zip(actions, effects):
    required: set[str] = set()
    for slot in (*reads, *writes):
      if slot in last_writer:
        required.add(last_writer[slot])
    for slot in writes:
      required.update(readers.get(slot, ()))
    if any(not _ordered(event, action.dependencies, ancestors) for event in required):
      _fail(f"action {action.instruction_id!r} is missing an L2 RAW/WAR/WAW edge")
    for slot in writes:
      if not write_done:
        _fail(f"action {action.instruction_id!r} writes without a completion milestone")
      last_writer[slot] = write_done
      readers[slot] = set()
    for slot in reads:
      if not read_done:
        _fail(f"action {action.instruction_id!r} reads without a completion milestone")
      readers.setdefault(slot, set()).add(read_done)


def _verify_global_hazards(
  accesses: Sequence[_Access],
  actions: Sequence[ExecGroupAction],
  ancestors: Mapping[str, frozenset[str]],
  maintenance: Sequence[tuple[int, MemoryMaintenanceDesc]],
  alias_indices: Sequence[int],
) -> None:
  for current_index, current in enumerate(accesses):
    action = actions[current.action_index]
    for previous in accesses[:current_index]:
      if (
        alias_indices[current.formal_index] != alias_indices[previous.formal_index]
        or current.start >= previous.end
        or previous.start >= current.end
        or not (current.writing or previous.writing)
      ):
        continue
      if (
        current.scope is not None
        and previous.scope is not None
        and current.scope != previous.scope
      ):
        continue  # statically disjoint page scopes (plan §2 scope proof)
      if not _ordered(previous.event, action.dependencies, ancestors):
        _fail(f"action {action.instruction_id!r} lacks a required global RAW/WAR/WAW edge")
      if (
        previous.writing
        and not current.writing
        and current.cache_levels
        and not _maintenance_covers(maintenance, previous, current, current.cache_levels, ancestors)
      ):
        _fail(f"action {action.instruction_id!r} lacks required cache maintenance after HBM write")


def _verify_task_budget(
  task: ExecTileGroupTask, call: CallBinding, registry: ProfileRegistry, sim: SimConfig, where: str
) -> None:
  context_contract = task.resource_contract
  context_layout = task.layout
  if not isinstance(context_contract, ContextResources) or not isinstance(context_layout, ArenaLayout):
    _fail(f"{where} has incomplete Context resources")
  envelopes: dict[int, list[int]] = {}
  for binding in task.role_bindings.values():
    program = binding.tile_program
    tile_contract = program.resource_contract
    tile_layout = program.layout
    if not isinstance(tile_contract, TileResources) or not isinstance(tile_layout, ArenaLayout):
      _fail(f"{where} child Tile Program has incomplete resources")
    for mode in tile_contract.allowed_profiles:
      profile = registry.profile("l1", mode)
      envelope = envelopes.setdefault(mode, [0] * profile.banks)
      if len(tile_layout.per_bank_bytes) != profile.banks:
        _fail(f"{where} child L1 layout bank count differs from profile {mode}")
      for index, value in enumerate(tile_layout.per_bank_bytes):
        envelope[index] = max(envelope[index], value)
  for mode in context_contract.allowed_profiles:
    profile = registry.profile("l2", mode)
    if any(value > profile.user_spm_per_bank for value in context_layout.per_bank_bytes):
      _fail(f"{where} L2 per-bank layout exceeds allowed profile {mode}")
  for mode, envelope in envelopes.items():
    profile = registry.profile("l1", mode)
    if len(envelope) != profile.banks or any(
      context_contract.requested_contexts_per_tile * value > profile.user_spm_per_bank for value in envelope
    ):
      _fail(f"{where} R x child L1 envelope exceeds profile {mode}")


_EVENT_REGISTRATION_FENCES = frozenset(
  {
    ExecGroupActionOp.WAIT_EVENT,
    ExecGroupActionOp.BARRIER_GROUP,
    ExecGroupActionOp.PROFILE_RECONFIG,
    ExecGroupActionOp.MEMORY_MAINTENANCE,
  }
)


def _reconstruct_event_resources(task: ExecTileGroupTask, where: str) -> tuple[dict[str, int], int]:
  """Independently reconstruct event references and proven live occupancy."""

  producer: dict[str, int] = {}
  consumer_actions: dict[str, set[int]] = {}
  lineage: dict[str, frozenset[str]] = {}
  action_outputs: list[tuple[str, ...]] = []
  references: dict[str, int] = {}
  for action_index, action in enumerate(task.actions):
    if len(action.dependencies) != len(set(action.dependencies)):
      _fail(f"{where}.actions[{action_index}] has duplicate event dependencies")
    if any(event not in producer for event in action.dependencies):
      _fail(f"{where}.actions[{action_index}] has a forward or unknown event dependency")
    for event in action.dependencies:
      references[event] += 1
      consumer_actions[event].add(action_index)

    outputs = action.output_events
    if len(outputs) != len(set(outputs)):
      _fail(f"{where}.actions[{action_index}] produces duplicate events")
    inherited = set(action.dependencies)
    for event in action.dependencies:
      inherited.update(lineage[event])
    for event in outputs:
      if event in producer:
        _fail(f"{where}.actions[{action_index}] duplicates event {event!r}")
      event_lineage = set(inherited)
      if action.op is ExecGroupActionOp.DISPATCH_ROLE and event == action.dst:
        request = action.args[0]
        event_lineage.update(
          phase for phase in (request.input_released_event, request.output_ready_event) if phase
        )
      producer[event] = action_index
      consumer_actions[event] = set()
      lineage[event] = frozenset(event_lineage)
      references[event] = 0
    action_outputs.append(outputs)

  resident: set[str] = set()
  terminal: set[str] = set()
  issued: set[int] = set()
  maximum_resident = 0
  for action_index, action in enumerate(task.actions):
    # Registration allocates every output atomically while dependency entries
    # are still resident.  Retirement proof is applied only after this peak.
    resident.update(action_outputs[action_index])
    maximum_resident = max(maximum_resident, len(resident))
    if action.op not in _EVENT_REGISTRATION_FENCES:
      continue

    issued.add(action_index)
    if action.op is ExecGroupActionOp.BARRIER_GROUP:
      issued.update(range(action_index))
      terminal.update(event for outputs in action_outputs[:action_index] for event in outputs)
    else:
      completed = set(action.dependencies)
      for event in action.dependencies:
        completed.update(lineage[event])
      terminal.update(completed)
      issued.update(producer[event] for event in completed)
    resident.difference_update(
      {event for event in resident if event in terminal and consumer_actions[event] <= issued}
    )

  return references, maximum_resident


def _verify_resource_budget(budget: object, where: str, hw: HardwareConfig, sim: SimConfig) -> None:
  if not isinstance(budget, ResourceBudget):
    _fail(f"{where} is not a ResourceBudget")
  for name in ("max_live_grids", "event_frontier", "frame_slots"):
    _uint(getattr(budget, name), f"{where}.{name}")
  if budget.max_live_grids > sim.group.dispatch_capacity:
    _fail(f"{where}.max_live_grids exceeds dispatch capacity")
  if budget.event_frontier > sim.group.event_capacity:
    _fail(f"{where}.event_frontier exceeds event capacity")
  if budget.frame_slots > hw.frame_slot_capacity:
    _fail(f"{where}.frame_slots exceeds Frame Slot capacity")
  limits = {
    "mfe_load": sim.group.prefetch_capacity,
    "mfe_store": sim.group.store_capacity,
    "dispatch": sim.group.dispatch_capacity,
    "inflight": sim.group.inflight_capacity,
  }
  for name, value in budget.engine_requirements.items():
    _uint(value, f"{where}.engine_requirements[{name!r}]")
    if name in limits and value > limits[name]:
      _fail(f"{where} engine requirement {name!r} exceeds target capacity")


def _verify_budget_covers(task: ExecTileGroupTask, budget: ResourceBudget, where: str) -> None:
  ancestors: dict[str, frozenset[str]] = {}
  live_grids: set[str] = set()
  required_grids = 0
  for action in task.actions:
    completed = _event_ancestors(action.dependencies, ancestors)
    live_grids.difference_update(completed)
    if action.op is ExecGroupActionOp.DISPATCH_ROLE and action.dst:
      live_grids.add(action.dst)
      required_grids = max(required_grids, len(live_grids))
    base = _event_ancestors(action.dependencies, ancestors)
    for event in action.output_events:
      event_ancestors = set(base)
      if action.op is ExecGroupActionOp.DISPATCH_ROLE and event == action.dst:
        event_ancestors.update(
          event
          for event in (action.args[0].input_released_event, action.args[0].output_ready_event)
          if event
        )
      ancestors[event] = frozenset(event_ancestors)
  expected_uses, required_frontier = _reconstruct_event_resources(task, where)
  if not isinstance(task.event_uses, Mapping):
    _fail(f"{where}.event_uses is not a mapping")
  for event, count in task.event_uses.items():
    _nonempty(event, f"{where}.event_uses key")
    _uint(count, f"{where}.event_uses[{event!r}]")
  if dict(task.event_uses) != expected_uses:
    _fail(f"{where}.event_uses disagrees with executable dependency references")
  required_slots = 0
  for role in task.role_bindings.values():
    layout = role.tile_program.layout
    if not isinstance(layout, ArenaLayout):
      _fail(f"{where} Tile Program is missing its prepared layout")
    for item in layout.buffer_layouts:
      required_slots = max(required_slots, item.slot_id + 1)
  if budget.max_live_grids < required_grids:
    _fail(f"{where}.max_live_grids understates executable Grid liveness")
  if budget.event_frontier != required_frontier:
    _fail(f"{where}.event_frontier differs from proven Event Table occupancy")
  if budget.frame_slots < required_slots:
    _fail(f"{where}.frame_slots understates executable layout slots")


def _verify_device_control(
  program: CompiledProgram,
  tasks: Mapping[str, ExecTileGroupTask],
  summaries: Mapping[str, _TaskSummary],
  inputs: Sequence[ExecGlobalInput],
) -> None:
  registry = program.registry
  body = (*program.entry_prefix, *(program.entry.body if isinstance(program.entry, ExecModel) else ()))
  if any(op.op not in {"await", "profile_reconfig", "memory_maintenance"} for op in program.entry_prefix):
    _fail("entry_prefix may contain only explicit control instructions")
  current = program.entry_profiles
  produced: set[str] = set()
  ancestors: dict[str, frozenset[str]] = {}
  waited_events: set[str] = set()
  consumed_await_events: set[str] = set()
  waits: dict[str, frozenset[str]] = {}
  instruction_ids: set[str] = set()
  callsites: set[str] = set()
  submitted_bindings: set[str] = set()
  submitted: list[tuple[int, ExecDeviceOp, CallBinding, _TaskSummary, tuple[_Access, ...]]] = []
  active: dict[str, tuple[CallBinding, _TaskSummary]] = {}
  history: dict[str, list[str]] = {"l1": [], "l2": []}
  maintenance: list[tuple[int, MemoryMaintenanceDesc]] = []
  scope_records: list[tuple[str, frozenset[str], str]] = []
  returned = False

  for index, op in enumerate(body):
    where = f"device[{index}]"
    instruction_id, _ = _instruction_identity(op, where)
    if instruction_id in instruction_ids:
      _fail(f"duplicate device instruction_id {instruction_id!r}")
    instruction_ids.add(instruction_id)
    if op.op == "submit":
      if (
        not op.callsite_id
        or op.callsite_id in callsites
        or op.binding_id in submitted_bindings
        or op.binding_id not in program.call_bindings
        or op.binding_id not in tasks
        or op.command is not None
      ):
        _fail(f"{where} has invalid or duplicate callsite/binding identity")
      call = program.call_bindings[op.binding_id]
      task = tasks[op.binding_id]
      summary = summaries[op.binding_id]
      if op.ctx_name != call.context_name or task.name != call.context_name:
        _fail(f"{where} context name disagrees with its call binding")
      if tuple(op.actual_inputs) != call.actual_inputs or len(op.actual_inputs) != len(task.global_inputs):
        _fail(f"{where} actual input mapping disagrees with its specialized call")
      if any(type(item) is not int or not 0 <= item < len(inputs) for item in op.actual_inputs):
        _fail(f"{where} has an out-of-range actual input")
      for formal, actual_index in zip(task.global_inputs, op.actual_inputs):
        actual_input = inputs[actual_index]
        if actual_input.dims != formal.dims or actual_input.dtype != formal.dtype:
          _fail(f"{where} actual input shape or dtype disagrees with Context formal")
      if current.l1_mode != call.entry_l1_mode or current.l2_mode != call.resolved_l2_mode:
        _fail(f"{where} requires a profile transition absent from device control")
      if ProfileState(current.l1_mode, current.l2_mode) not in call.permitted_profiles:
        _fail(f"{where} uses an unpermitted profile combination")
      if not op.event_tag or op.event_tag in produced:
        _fail(f"{where} has missing or duplicate completion event")
      if len(op.dependencies) != len(set(op.dependencies)) or not set(op.dependencies) <= produced:
        _fail(f"{where} has invalid submit dependencies")
      mapped = tuple(
        replace(access, formal_index=op.actual_inputs[access.formal_index], action_index=index)
        for access in summary.accesses
      )
      _verify_cross_root_hazards(submitted, mapped, op, ancestors, waited_events)
      for old_event, (old_call, old_summary) in active.items():
        if (
          (call.requires_l1_exclusive or old_call.requires_l1_exclusive)
          and summary.uses_l1
          and old_summary.uses_l1
          and old_event not in waited_events
          and not _ordered(old_event, op.dependencies, ancestors)
        ):
          _fail(f"{where} violates cross-root L1 exclusivity")
      ancestors[op.event_tag] = _event_ancestors(op.dependencies, ancestors)
      produced.add(op.event_tag)
      callsites.add(op.callsite_id)
      submitted_bindings.add(op.binding_id)
      active[op.event_tag] = (call, summary)
      submitted.append((index, op, call, summary, mapped))
      history["l2"].append(op.event_tag)
      if summary.uses_l1:
        history["l1"].append(op.event_tag)
      ordered_events = waited_events | _event_ancestors(op.dependencies, ancestors)
      submit_scopes = _device_task_scopes(task)
      for prior_event, prior_scopes, prior_kind in scope_records:
        if prior_kind != "mutate":
          continue  # use/use pairs stay unordered; only host mutations gate submits
        sharing = prior_scopes & submit_scopes
        if sharing and prior_event not in ordered_events:
          _fail(
            f"{where} accesses scope '{sorted(sharing)[0]}' without a dependency on the prior"
            f" host {prior_kind}"
          )
      scope_records.append((op.event_tag, submit_scopes, "use"))
      current = ProfileState(call.exit_l1_mode, current.l2_mode)
    elif op.op == "await":
      if (
        not op.event_tag
        or op.event_tag in consumed_await_events
        or op.dependencies
        or op.command is not None
        or op.actual_inputs
        or op.ctx_name
        or op.callsite_id
        or op.binding_id
        or op.event_tag not in produced
      ):
        _fail(f"{where} has invalid ordinary await")
      waits[instruction_id] = frozenset((op.event_tag,))
      consumed_await_events.add(op.event_tag)
      closure = {op.event_tag, *ancestors[op.event_tag]}
      waited_events.update(closure)
      for event in closure:
        active.pop(event, None)
    elif op.op == "host_call":
      command = op.command
      if (
        op.actual_inputs
        or op.ctx_name
        or op.binding_id
        or not isinstance(command, ExecHostCall)
      ):
        _fail(f"{where} has invalid host call fields")
      if not command.name.strip():
        _fail(f"{where} host call routine name must be non-empty")
      if not op.event_tag or op.event_tag in produced:
        _fail(f"{where} has a missing or duplicate completion event")
      if len(op.dependencies) != len(set(op.dependencies)) or not set(op.dependencies) <= produced:
        _fail(f"{where} has invalid host call dependencies")
      declared: dict[int, list[tuple[int, int]]] = {}
      for access_index, access in enumerate(command.accesses):
        if type(access.input_index) is not int or not 0 <= access.input_index < len(inputs):
          _fail(f"{where}.accesses[{access_index}] names an out-of-range global input")
        formal = inputs[access.input_index]
        if access.mode not in ("read", "write", "readwrite"):
          _fail(f"{where}.accesses[{access_index}] has an invalid mode")
        if access.offset + access.bytes > formal.size_bytes:
          _fail(f"{where}.accesses[{access_index}] exceeds global input '{formal.name}'")
        declared.setdefault(access.input_index, []).append(
          (access.offset, access.offset + access.bytes)
        )
      for input_index, spans in declared.items():
        spans.sort()
        if any(
          start < previous_end
          for (_, previous_end), (start, _) in itertools.pairwise(spans)
        ):
          _fail(f"{where} has overlapping accesses on global input {input_index}")
      ordered_events = waited_events | _event_ancestors(op.dependencies, ancestors)
      for prior_event, prior_scopes, prior_kind in scope_records:
        sharing = prior_scopes & set(command.scopes)
        if sharing and prior_event not in ordered_events:
          _fail(
            f"{where} mutates scope '{sorted(sharing)[0]}' without a dependency on the prior"
            f" host {prior_kind}"
          )
      scope_records.append((op.event_tag, frozenset(command.scopes), "mutate"))
      ancestors[op.event_tag] = _event_ancestors(op.dependencies, ancestors)
      produced.add(op.event_tag)
    elif op.op == "profile_reconfig":
      profile_command = op.command
      if (
        not isinstance(profile_command, ProfileReconfigDesc)
        or op.event_tag
        or op.actual_inputs
        or op.dependencies
        or op.ctx_name
        or op.callsite_id
        or op.binding_id
      ):
        _fail(f"{where} has invalid profile control fields")
      level = profile_command.level
      if level not in ("l1", "l2"):
        _fail(f"{where} has invalid profile level")
      if tuple(profile_command.frontier) != tuple(history[level]):
        _fail(f"{where} profile frontier does not cover the complete affected root history")
      mode = current.l1_mode if level == "l1" else current.l2_mode
      target = _verify_profile_desc(profile_command, where, registry, level, mode, waits, ancestors, "")
      if profile_command.command_id != instruction_id:
        _fail(f"{where} profile command identity is inconsistent")
      if profile_command.source_ref != op.source_ref:
        _fail(f"{where} instruction and profile descriptor source disagree")
      history[level].clear()
      current = (
        ProfileState(target, current.l2_mode) if level == "l1" else ProfileState(current.l1_mode, target)
      )
    elif op.op == "memory_maintenance":
      maintenance_command = op.command
      if (
        not isinstance(maintenance_command, MemoryMaintenanceDesc)
        or op.event_tag
        or op.actual_inputs
        or op.ctx_name
        or op.callsite_id
        or op.binding_id
      ):
        _fail(f"{where} has invalid maintenance control fields")
      desc = _verify_maintenance_desc(maintenance_command, where, inputs, produced)
      if desc.command_id != instruction_id:
        _fail(f"{where} maintenance command identity is inconsistent")
      if tuple(desc.dependencies) != op.dependencies or desc.source_ref != op.source_ref:
        _fail(f"{where} instruction and maintenance descriptor disagree")
      maintenance.append((index, desc))
    elif op.op == "return":
      if (
        index != len(body) - 1
        or op.event_tag
        or op.actual_inputs
        or op.dependencies
        or op.command is not None
        or op.ctx_name
        or op.callsite_id
        or op.binding_id
      ):
        _fail(f"{where} must be the unique terminal return")
      if active:
        _fail(f"{where} returns before all submitted roots are awaited")
      returned = True
    else:
      _fail(f"{where} has unknown device opcode {op.op!r}")

  if isinstance(program.entry, ExecModel):
    if not returned or submitted_bindings != set(tasks):
      _fail("model must submit every specialized binding and end with return")
  else:
    call = program.call_bindings[program.entry.binding_id]
    if current != ProfileState(call.entry_l1_mode, call.resolved_l2_mode):
      _fail("standalone entry prefix does not establish its compiled entry profile")
    summary = summaries[program.entry.binding_id]
    for task_access in summary.accesses:
      standalone_access = replace(task_access, action_index=len(program.entry_prefix))
      if (
        standalone_access.writing
        and standalone_access.cache_levels
        and not _maintenance_precedes(maintenance, standalone_access)
      ):
        _fail("standalone entry lacks dirty-cache clean before HBM write")
    current = ProfileState(call.exit_l1_mode, call.resolved_l2_mode)
  if current != program.exit_profiles:
    _fail("device control exit profile does not match compiled exit_profiles")
  _verify_cross_root_maintenance(submitted, maintenance, ancestors)


def _host_static_effects(
  program: CompiledProgram, tasks: Mapping[str, ExecTileGroupTask]
) -> tuple[dict[str, object], dict[str, object]]:
  """Independently recompute the host_accesses and scope_effects maps.

  Mirrors the compiler's static effect bookkeeping from the frozen DTOs
  only; the executable body is the sole input.
  """
  entry = program.entry
  if not isinstance(entry, ExecModel):
    return {}, {}
  host_accesses: dict[str, object] = {}
  scope_effects: dict[str, object] = {}
  for op in entry.body:
    if op.op == "host_call" and isinstance(op.command, ExecHostCall):
      host_accesses[op.instruction_id] = {
        "name": op.command.name,
        "accesses": tuple(
          {
            "input_index": access.input_index,
            "offset": access.offset,
            "bytes": access.bytes,
            "mode": access.mode,
          }
          for access in op.command.accesses
        ),
        "scopes": op.command.scopes,
      }
      scope_effects[op.event_tag] = tuple(
        {"scope": scope, "kind": "mutate"} for scope in op.command.scopes
      )
    elif op.op == "submit" and op.binding_id in tasks:
      scopes = sorted(_device_task_scopes(tasks[op.binding_id]))
      if scopes:
        scope_effects[op.event_tag] = tuple(
          {"scope": scope, "kind": "use"} for scope in scopes
        )
  return host_accesses, scope_effects


def _verify_host_static_effects(
  program: CompiledProgram, tasks: Mapping[str, ExecTileGroupTask], inputs: Sequence[ExecGlobalInput]
) -> None:
  """Compare artifact static_effects host maps against an independent recompute."""
  if set(program.static_effects) - {
    "resources",
    "accesses",
    "host_accesses",
    "scope_effects",
    "program_text_bytes",
  }:
    _fail("static_effects contains unknown keys")
  expected_host, expected_scope = _host_static_effects(program, tasks)
  for key, expected in (("host_accesses", expected_host), ("scope_effects", expected_scope)):
    declared = program.static_effects.get(key, {})
    if not isinstance(declared, Mapping):
      _fail(f"static_effects[{key!r}] must be a mapping keyed by instruction/event identity")
    if set(declared) != set(expected):
      _fail(f"static_effects[{key!r}] does not match the executable device body")
    for identity, entry in expected.items():
      actual = declared[identity]
      if canonical_value(actual) != canonical_value(entry):
        _fail(f"static_effects[{key!r}][{identity!r}] disagrees with the executable body")
  if not isinstance(program.entry, ExecModel):
    return
  for op in program.entry.body:
    if op.op != "host_call" or not isinstance(op.command, ExecHostCall):
      continue
    for access in op.command.accesses:
      if not 0 <= access.input_index < len(inputs):
        _fail("host call access names an out-of-range global input")


def _device_task_scopes(task: ExecTileGroupTask) -> frozenset[str]:
  """Scopes a specialized submission uses: every tile gather/scatter scope."""
  scopes: set[str] = set()
  for role in task.role_bindings.values():
    for descriptor in role.tile_program.descriptors.values():
      if descriptor.kind == "MFE" and descriptor.op in ("gather", "scatter"):
        scope = descriptor.params[descriptor.op].scope
        if scope:
          scopes.add(scope)
  return frozenset(scopes)


def _verify_cross_root_hazards(
  prior_submits: Sequence[tuple[int, ExecDeviceOp, CallBinding, _TaskSummary, tuple[_Access, ...]]],
  current_accesses: Sequence[_Access],
  op: ExecDeviceOp,
  ancestors: Mapping[str, frozenset[str]],
  waited_events: set[str],
) -> None:
  for _, prior_op, _, _, prior_accesses in prior_submits:
    ordered = prior_op.event_tag in waited_events or _ordered(
      prior_op.event_tag, op.dependencies, ancestors
    )
    if ordered:
      continue
    if any(
      current.formal_index == previous.formal_index
      and current.start < previous.end
      and previous.start < current.end
      and (current.writing or previous.writing)
      # Different non-empty scopes own disjoint pages (plan §1).
      and not (
        current.scope is not None
        and previous.scope is not None
        and current.scope != previous.scope
      )
      for current in current_accesses
      for previous in prior_accesses
    ):
      _fail(f"device submit {op.instruction_id!r} is missing a global RAW/WAR/WAW edge")

def _verify_cross_root_maintenance(
  submits: Sequence[tuple[int, ExecDeviceOp, CallBinding, _TaskSummary, tuple[_Access, ...]]],
  maintenance: Sequence[tuple[int, MemoryMaintenanceDesc]],
  ancestors: Mapping[str, frozenset[str]],
) -> None:
  for index, _op, _call, _summary, accesses in submits:
    for access in accesses:
      actual = replace(access, action_index=index)
      if actual.writing and actual.cache_levels and not _maintenance_precedes(maintenance, actual):
        _fail("device program lacks dirty-cache clean before HBM write")
  for current_index, (_, current_op, _, _, current_accesses) in enumerate(submits):
    for _, prior_op, _, _, prior_accesses in submits[:current_index]:
      for previous in prior_accesses:
        if not previous.writing:
          continue
        for current in current_accesses:
          if (
            not current.cache_levels
            or current.formal_index != previous.formal_index
            or current.start >= previous.end
            or previous.start >= current.end
          ):
            continue
          producer = replace(
            previous, event=prior_op.event_tag, action_index=_device_position(submits, prior_op)
          )
          consumer = replace(
            current, event=current_op.event_tag, action_index=_device_position(submits, current_op)
          )
          if not _maintenance_covers(maintenance, producer, consumer, current.cache_levels, ancestors):
            _fail(f"device submit {current_op.instruction_id!r} lacks required cache maintenance")


def _device_position(
  submits: Sequence[tuple[int, ExecDeviceOp, CallBinding, _TaskSummary, tuple[_Access, ...]]],
  target: ExecDeviceOp,
) -> int:
  return next(index for index, op, _, _, _ in submits if op is target)


def _entry_tasks(
  program: CompiledProgram,
) -> tuple[Mapping[str, ExecTileGroupTask], Sequence[ExecGlobalInput]]:
  if program.entry_kind == "standalone":
    if not isinstance(program.entry, ExecTileGroupTask):
      _fail("standalone entry is not ExecTileGroupTask")
    return FrozenMap({program.entry.binding_id: program.entry}), program.entry.global_inputs
  if program.entry_kind == "model":
    if not isinstance(program.entry, ExecModel):
      _fail("model entry is not ExecModel")
    if set(program.entry.context_pins) != set(program.entry.tasks):
      _fail("model context_pins and specialized task bindings differ")
    for binding_id, pin in program.entry.context_pins.items():
      if pin is not None and (type(pin) is not int or pin < 0):
        _fail(f"model context pin for {binding_id!r} is invalid")
    return program.entry.tasks, program.entry.inputs
  _fail("entry_kind must be 'standalone' or 'model'")


def _verify_shared_relationships(
  program: CompiledProgram, tasks: Mapping[str, ExecTileGroupTask]
) -> None:
  if not isinstance(program.entry, ExecModel):
    if any(task.shared_inputs for task in tasks.values()):
      _fail("shared L2 imports require a specialized model executable")
    return

  submissions: dict[str, tuple[str, int, frozenset[str]]] = {}
  device_ancestors: dict[str, frozenset[str]] = {}
  l2_epoch = 0
  for op in (*program.entry_prefix, *program.entry.body):
    if op.op == "profile_reconfig":
      if isinstance(op.command, ProfileReconfigDesc) and op.command.level == "l2":
        l2_epoch += 1
    elif op.op == "host_call":
      device_ancestors[op.event_tag] = _event_ancestors(op.dependencies, device_ancestors)
    elif op.op == "submit":
      ancestry = _event_ancestors(op.dependencies, device_ancestors)
      device_ancestors[op.event_tag] = ancestry
      submissions[op.binding_id] = (op.event_tag, l2_epoch, ancestry)

  for binding_id, task in tasks.items():
    if not task.shared_inputs:
      continue
    consumer_submission = submissions.get(binding_id)
    if consumer_submission is None:
      _fail(f"shared consumer binding {binding_id!r} is not submitted")
    _consumer_event, consumer_epoch, consumer_ancestors = consumer_submission
    consumer_call = program.call_bindings[binding_id]
    backing_spans: dict[tuple[str, str], tuple[int, int]] = {}
    for shared in task.shared_inputs:
      if shared.producer_binding_id == binding_id:
        _fail(f"shared input {shared.slot!r} refers to its own consumer binding")
      producer_task = tasks.get(shared.producer_binding_id)
      producer_call = program.call_bindings.get(shared.producer_binding_id)
      producer_submission = submissions.get(shared.producer_binding_id)
      if producer_task is None or producer_call is None or producer_submission is None:
        _fail(f"shared input {shared.slot!r} names an unknown producer binding")
      producer_event, producer_epoch, _producer_ancestors = producer_submission
      exports = {
        buffer.slot: buffer for buffer in producer_task.l2_buffers if buffer.sharing == "readonly"
      }
      producer_buffer = exports.get(shared.producer_slot)
      if producer_buffer is None:
        _fail(f"shared input {shared.slot!r} names an unpublished producer slot")
      if (
        (shared.dims, shared.dtype, shared.element_bytes, shared.bytes)
        != (
          producer_buffer.dims,
          producer_buffer.dtype,
          producer_buffer.element_bytes,
          producer_buffer.bytes,
        )
      ):
        _fail(f"shared input {shared.slot!r} shape or dtype differs from its producer")
      if (
        producer_epoch != consumer_epoch
        or producer_call.resolved_l2_mode != consumer_call.resolved_l2_mode
      ):
        _fail(f"shared input {shared.slot!r} crosses an L2 profile epoch")
      if producer_event not in consumer_ancestors:
        _fail(f"shared input {shared.slot!r} lacks producer completion dependency ancestry")
      if producer_task.layout is None:
        _fail(f"shared producer {shared.producer_binding_id!r} has no L2 layout")
      key = (shared.producer_binding_id, shared.producer_slot)
      allocation = next(
        (
          item
          for item in producer_task.layout.buffer_layouts
          if item.buffer_id == shared.producer_slot
        ),
        None,
      )
      if allocation is None:
        _fail(f"shared producer layout omits slot {shared.producer_slot!r}")
      backing_spans.setdefault(
        key,
        (
          _padded_l2_bytes_per_bank(
            producer_task, shared.producer_slot, f"producer {shared.producer_binding_id!r}"
          ),
          allocation.banks,
        ),
      )

    if task.layout is None or not task.layout.per_bank_bytes:
      _fail(f"shared consumer {binding_id!r} has no local L2 reservation geometry")
    own_per_bank = task.layout.per_bank_bytes[0]
    if task.resource_contract is None:
      _fail(f"shared consumer {binding_id!r} has no resource contract")
    for mode in task.resource_contract.allowed_profiles:
      profile = program.registry.profile("l2", mode)
      if len(task.layout.per_bank_bytes) != profile.banks:
        _fail(f"shared consumer {binding_id!r} L2 bank geometry differs in profile {mode}")
      if any(banks != profile.banks for _, banks in backing_spans.values()):
        _fail(f"shared consumer {binding_id!r} shared backing bank geometry differs in profile {mode}")
      shared_per_bank = sum(span for span, _banks in backing_spans.values())
      if shared_per_bank > _UINT64_MAX:
        _fail(f"shared consumer {binding_id!r} shared L2 demand exceeds uint64")
      if shared_per_bank + own_per_bank > profile.user_spm_per_bank:
        _fail(
          f"shared consumer {binding_id!r} exceeds per-bank L2 capacity in profile {mode}"
        )


def _fail_indexed_memory_fidelity(program: CompiledProgram, sim: SimConfig) -> None:
  """Plan §2: address-resolved indexed memory requires full_memory ByteStore."""
  entry = program.entry
  tasks = entry.tasks.values() if isinstance(entry, ExecModel) else (entry,)
  for task in tasks:
    for binding in task.role_bindings.values():
      for descriptor in binding.tile_program.descriptors.values():
        if any(
          isinstance(descriptor.params.get(key), (ExecTileGatherDesc, ExecTileScatterDesc))
          for key in ("gather", "scatter")
        ):
          if sim.fidelity != "full_memory":
            _fail("indexed memory requires full_memory fidelity")
          return


def verify_compiled_program(program: CompiledProgram, hw: HardwareConfig, sim: SimConfig) -> None:
  """Verify package integrity and executable semantics without graph mutation."""
  if not isinstance(program, CompiledProgram):
    _fail("expected CompiledProgram")
  if type(program.schema_version) is not int or program.schema_version != 3 or program.compiler_abi != "v3":
    _fail("unsupported compiled schema or compiler ABI; recompile from source")
  _fail_indexed_memory_fidelity(program, sim)
  if not _is_frozen(program):
    _fail("compiled artifact is not deeply immutable")
  for name in ("source_hash", "registry_hash", "target_hash", "artifact_hash"):
    value = getattr(program, name)
    if not isinstance(value, str) or not _HEX.fullmatch(value):
      _fail(f"{name} is not a canonical SHA-256 digest")
  if digest(program.source_ir) != program.source_hash:
    _fail("source_hash does not match canonical source_ir")
  if artifact_digest(program) != program.artifact_hash:
    _fail("artifact_hash mismatch")
  if not isinstance(program.registry, ProfileRegistry):
    _fail("embedded Registry has invalid type")
  _nonempty(program.source_ir, "source_ir")
  if not isinstance(program.workload_info, WorkloadInfo):
    _fail("workload_info has invalid type")
  _nonempty(program.workload_info.name, "workload_info.name")
  _nonempty(program.workload_info.description, "workload_info.description")
  for map_name in ("dependency_proofs", "static_effects"):
    mapping = getattr(program, map_name)
    if any(not isinstance(key, str) or not key for key in mapping):
      _fail(f"{map_name} keys must be non-empty strings")

  expected_registry = build_registry(hw)
  if program.registry_hash != program.registry.registry_hash:
    _fail("package and embedded Registry hashes disagree")
  if digest(program.registry) != digest(expected_registry):
    _fail("embedded Registry does not match the load target")
  if program.registry_hash != expected_registry.registry_hash:
    _fail("Registry fingerprint mismatch")
  if program.target_hash != target_fingerprint(hw, sim):
    _fail("target fingerprint mismatch")
  registry = program.registry
  if not isinstance(program.entry_profiles, ProfileState) or not isinstance(
    program.exit_profiles, ProfileState
  ):
    _fail("entry_profiles and exit_profiles must be ProfileState values")
  for name, state in (("entry_profiles", program.entry_profiles), ("exit_profiles", program.exit_profiles)):
    _uint(state.l1_mode, f"{name}.l1_mode")
    _uint(state.l2_mode, f"{name}.l2_mode")
  registry.profile("l1", program.entry_profiles.l1_mode)
  registry.profile("l2", program.entry_profiles.l2_mode)
  registry.profile("l1", program.exit_profiles.l1_mode)
  registry.profile("l2", program.exit_profiles.l2_mode)

  tasks, inputs = _entry_tasks(program)
  for index, item in enumerate(inputs):
    _verify_global_input(item, f"entry.inputs[{index}]")
  _unique(tuple(item.name for item in inputs), "entry input names")
  if isinstance(program.entry, ExecModel):
    for binding_id, pin in program.entry.context_pins.items():
      if pin is not None and pin >= sim.group.active_context_capacity:
        _fail(f"model context pin for {binding_id!r} exceeds Group context capacity")
  if set(program.call_bindings) != set(tasks):
    _fail("call_bindings and specialized executable tasks differ")
  identities: dict[int, tuple[int, int]] = {}
  summaries: dict[str, _TaskSummary] = {}
  for binding_id, task in tasks.items():
    if (
      not isinstance(binding_id, str)
      or not binding_id
      or not isinstance(program.call_bindings[binding_id], CallBinding)
    ):
      _fail("invalid call binding map")
    call = program.call_bindings[binding_id]
    if call.binding_id != binding_id or not call.permitted_profiles:
      _fail(f"call binding {binding_id!r} is incomplete")
    _nonempty(call.context_name, f"call binding {binding_id!r}.context_name")
    for name in ("requested_l2_mode", "resolved_l2_mode", "entry_l1_mode", "exit_l1_mode"):
      _uint(getattr(call, name), f"call binding {binding_id!r}.{name}")
    if type(call.requires_l1_exclusive) is not bool:
      _fail(f"call binding {binding_id!r}.requires_l1_exclusive must be bool")
    if any(type(index) is not int or index < 0 for index in call.actual_inputs):
      _fail(f"call binding {binding_id!r} has invalid actual input indices")
    _unique(call.permitted_profiles, f"call binding {binding_id!r}.permitted_profiles")
    for state in call.permitted_profiles:
      if not isinstance(state, ProfileState):
        _fail(f"call binding {binding_id!r} has invalid permitted profile value")
      _uint(state.l1_mode, f"call binding {binding_id!r}.permitted.l1_mode")
      _uint(state.l2_mode, f"call binding {binding_id!r}.permitted.l2_mode")
      registry.profile("l1", state.l1_mode)
      registry.profile("l2", state.l2_mode)
    if (
      ProfileState(call.entry_l1_mode, call.resolved_l2_mode) not in call.permitted_profiles
      or ProfileState(call.exit_l1_mode, call.resolved_l2_mode) not in call.permitted_profiles
    ):
      _fail(f"call binding {binding_id!r} omits its entry or exit profile combination")
    summaries[binding_id] = _verify_task(
      task, call, registry, hw, sim, f"tasks[{binding_id!r}]", identities
    )

  used_identities: dict[tuple[int, int], int] = {}
  for program_hash, identity in identities.items():
    if identity in used_identities and used_identities[identity] != program_hash:
      _fail("one tile program identity names different executable contents")
    used_identities[identity] = program_hash
  program_ids = sorted({program_id for program_id, _ in used_identities})
  if program_ids != list(range(1, len(program_ids) + 1)):
    _fail("tile program IDs are not the canonical positive sequence")

  if set(program.resource_budgets) != set(program.call_bindings):
    _fail("resource_budgets must contain exactly one budget per call binding")
  for name, budget in program.resource_budgets.items():
    _verify_resource_budget(budget, f"resource_budgets[{name!r}]", hw, sim)
    _verify_budget_covers(tasks[name], budget, f"resource_budgets[{name!r}]")
  _verify_relocations(program, tasks)
  _verify_source_map(program, tasks)
  _verify_device_control(program, tasks, summaries, inputs)
  _verify_host_static_effects(program, tasks, inputs)
  _verify_shared_relationships(program, tasks)
  _verify_binding_guards(program, summaries, inputs)


def _relocation_key(
  binding_id: str,
  scope: str,
  ordinal: int,
  field: str,
  kind: str,
  argument_index: int = -1,
  role_id: int = -1,
) -> tuple[object, ...]:
  return (binding_id, scope, ordinal, field, kind, argument_index, role_id)


def _expected_relocations(task: ExecTileGroupTask) -> set[tuple[object, ...]]:
  binding_id = task.binding_id
  result = {_relocation_key(binding_id, "task", 0, "completion_event", "event")}
  result.add(_relocation_key(binding_id, "task", 0, "event_uses", "event_map"))
  input_indices = {item.name: index for index, item in enumerate(task.global_inputs)}
  for ordinal, action in enumerate(task.actions):
    if action.dst:
      result.add(_relocation_key(binding_id, "group", ordinal, "dst", "event"))
    if action.dependencies:
      result.add(_relocation_key(binding_id, "group", ordinal, "dependencies", "event_tuple"))
    action_kind = {
      ExecGroupActionOp.WAIT_EVENT: "event_tuple",
      ExecGroupActionOp.SIGNAL_EVENT: "event_tuple",
      ExecGroupActionOp.DISPATCH_ROLE: "dispatch_events",
      ExecGroupActionOp.RELEASE_L2: "release_events",
      ExecGroupActionOp.PUBLISH_L2: "publish_events",
      ExecGroupActionOp.PROFILE_RECONFIG: "profile_frontier",
      ExecGroupActionOp.MEMORY_MAINTENANCE: "maintenance_dependencies",
      ExecGroupActionOp.INIT_STREAM: "queue",
    }.get(action.op)
    if action_kind is not None:
      argument_index = 0 if action_kind == "queue" else -1
      result.add(_relocation_key(binding_id, "group", ordinal, "args", action_kind, argument_index))
    if action.op in (ExecGroupActionOp.DMA_PREFETCH, ExecGroupActionOp.DMA_STORE):
      transfer = action.args[1]
      view = transfer.src if transfer.src.space == "global" else transfer.dst
      if view.space == "global" and view.base.startswith("global:"):
        formal_name = view.base.removeprefix("global:")
        if formal_name in input_indices:
          result.add(
            _relocation_key(
              binding_id, "context", ordinal, "global_view", "binding", input_indices[formal_name]
            )
          )
    elif action.op is ExecGroupActionOp.DISPATCH_ROLE:
      role = task.role_bindings[action.args[0].role_id]
      for view in role.global_actuals:
        if view.base.startswith("global:"):
          formal_name = view.base.removeprefix("global:")
          if formal_name in input_indices:
            result.add(
              _relocation_key(
                binding_id,
                "tile_role",
                ordinal,
                "global_actuals",
                "binding",
                input_indices[formal_name],
                role.role_id,
              )
            )
  for ordinal, _stream in enumerate(task.streams):
    result.add(_relocation_key(binding_id, "stream", ordinal, "queue_id", "queue"))
  for role_id, role in task.role_bindings.items():
    if role.in_stream is not None:
      result.add(_relocation_key(binding_id, "role", 0, "in_stream", "queue", role_id=role_id))
    if role.out_stream is not None:
      result.add(_relocation_key(binding_id, "role", 0, "out_stream", "queue", role_id=role_id))
  return result


def _verify_relocations(program: CompiledProgram, tasks: Mapping[str, ExecTileGroupTask]) -> None:
  seen: set[tuple[object, ...]] = set()
  for index, relocation in enumerate(program.relocations):
    if relocation.binding_id not in tasks:
      _fail(f"relocations[{index}] references unknown binding")
    task = tasks[relocation.binding_id]
    key = tuple(getattr(relocation, item.name) for item in fields(relocation))
    if key in seen:
      _fail("duplicate relocation")
    seen.add(key)
    if relocation.ordinal < 0 or relocation.argument_index < -1 or relocation.role_id < -1:
      _fail(f"relocations[{index}] has an invalid index")
    if relocation.scope == "context" and relocation.field == "global_view" and relocation.kind == "binding":
      if (
        relocation.role_id != -1
        or relocation.ordinal >= len(task.actions)
        or relocation.argument_index < 0
        or relocation.argument_index >= len(task.global_inputs)
      ):
        _fail(f"relocations[{index}] has invalid context relocation coordinates")
    elif (
      relocation.scope == "tile_role"
      and relocation.field == "global_actuals"
      and relocation.kind == "binding"
    ):
      role = task.role_bindings.get(relocation.role_id)
      if (
        role is None
        or relocation.ordinal >= len(task.actions)
        or not 0 <= relocation.argument_index < len(task.global_inputs)
      ):
        _fail(f"relocations[{index}] has invalid tile-role relocation coordinates")
    elif (relocation.scope, relocation.field, relocation.kind) in {
      ("group", "dst", "event"),
      ("group", "dependencies", "event_tuple"),
      ("group", "args", "event_tuple"),
      ("group", "args", "dispatch_events"),
      ("group", "args", "release_events"),
      ("group", "args", "publish_events"),
      ("group", "args", "profile_frontier"),
      ("group", "args", "maintenance_dependencies"),
      ("group", "args", "queue"),
    }:
      if relocation.ordinal >= len(task.actions):
        _fail(f"relocations[{index}] has an invalid Group action ordinal")
      action = task.actions[relocation.ordinal]
      if relocation.kind == "queue" and (
        action.op is not ExecGroupActionOp.INIT_STREAM or relocation.argument_index != 0
      ):
        _fail(f"relocations[{index}] queue relocation does not name INIT_STREAM arg0")
      expected_ops = {
        "event_tuple": {ExecGroupActionOp.WAIT_EVENT, ExecGroupActionOp.SIGNAL_EVENT},
        "dispatch_events": {ExecGroupActionOp.DISPATCH_ROLE},
        "release_events": {ExecGroupActionOp.RELEASE_L2},
        "publish_events": {ExecGroupActionOp.PUBLISH_L2},
        "profile_frontier": {ExecGroupActionOp.PROFILE_RECONFIG},
        "maintenance_dependencies": {ExecGroupActionOp.MEMORY_MAINTENANCE},
      }
      if (
        relocation.kind in expected_ops
        and relocation.field == "args"
        and action.op not in expected_ops[relocation.kind]
      ):
        _fail(f"relocations[{index}] Group args relocation has the wrong opcode")
      if relocation.kind == "event" and not action.dst:
        _fail(f"relocations[{index}] event relocation names an action without dst")
    elif (relocation.scope, relocation.field, relocation.kind) == ("stream", "queue_id", "queue"):
      if relocation.ordinal >= len(task.streams):
        _fail(f"relocations[{index}] has an invalid stream ordinal")
    elif (
      relocation.scope == "role"
      and relocation.field in {"in_stream", "out_stream"}
      and relocation.kind == "queue"
    ):
      if relocation.role_id not in task.role_bindings:
        _fail(f"relocations[{index}] has an invalid role identity")
    elif (
      relocation.scope == "tile" and relocation.field == "args" and relocation.kind in {"queue", "event"}
    ):
      tile_role = task.role_bindings.get(relocation.role_id)
      if tile_role is None or relocation.ordinal >= len(tile_role.tile_program.insts):
        _fail(f"relocations[{index}] has invalid Tile instruction coordinates")
      instruction = tile_role.tile_program.insts[relocation.ordinal]
      if relocation.kind == "queue":
        if relocation.argument_index != 0:
          _fail(f"relocations[{index}] Tile queue relocation must name arg0")
      elif instruction.op not in (
        ExecTileOp.WAIT,
        ExecTileOp.WAITALL,
      ) or not 0 <= relocation.argument_index < len(instruction.args):
        _fail(f"relocations[{index}] Tile event relocation does not name an external WAIT operand")
    elif (relocation.scope, relocation.field, relocation.kind) == ("task", "completion_event", "event"):
      if relocation.ordinal != 0 or relocation.role_id != -1 or relocation.argument_index != -1:
        _fail(f"relocations[{index}] has invalid Task completion coordinates")
    elif (relocation.scope, relocation.field, relocation.kind) == ("task", "event_uses", "event_map"):
      if relocation.ordinal != 0 or relocation.role_id != -1 or relocation.argument_index != -1:
        _fail(f"relocations[{index}] has invalid Task event-map coordinates")
    else:
      _fail(f"relocations[{index}] has unsupported relocation form")
  expected = set().union(*(_expected_relocations(task) for task in tasks.values()))
  missing = expected - seen
  unexpected = seen - expected
  if missing or unexpected:
    _fail(
      "relocation table differs from mutable executable fields "
      f"({len(missing)} missing, {len(unexpected)} unexpected)"
    )


def _verify_source_map(program: CompiledProgram, tasks: Mapping[str, ExecTileGroupTask]) -> None:
  actual: dict[str, SourceRef] = {}

  def add(item: object, where: str) -> None:
    instruction_id, ref = _instruction_identity(item, where)
    if instruction_id in actual:
      if actual[instruction_id] != ref:
        _fail(f"executable instruction identity {instruction_id!r} has conflicting source refs")
      return
    actual[instruction_id] = ref

  for index, item in enumerate(program.entry_prefix):
    add(item, f"entry_prefix[{index}]")
  if isinstance(program.entry, ExecModel):
    for index, item in enumerate(program.entry.body):
      add(item, f"entry.body[{index}]")
  for binding_id, task in tasks.items():
    for index, action in enumerate(task.actions):
      add(action, f"tasks[{binding_id!r}].actions[{index}]")
    seen_programs: set[tuple[int, tuple[str, ...]]] = set()
    for role in task.role_bindings.values():
      key = (role.tile_program.program_hash, tuple(inst.instruction_id for inst in role.tile_program.insts))
      if key in seen_programs:
        continue
      seen_programs.add(key)
      for index, inst in enumerate(role.tile_program.insts):
        add(inst, f"tasks[{binding_id!r}].program.insts[{index}]")
  if set(program.source_map) != set(actual):
    _fail("source_map keys do not match executable instructions")
  if any(program.source_map[key] != ref for key, ref in actual.items()):
    _fail("source_map disagrees with instruction source_ref")


def _guard_permissions(required: set[str]) -> str:
  return "rw" if required == {"r", "w"} else next(iter(required), "r")


def _host_access_permissions(mode: str) -> frozenset[str]:
  return {
    "read": frozenset({"r"}),
    "write": frozenset({"w"}),
    "readwrite": frozenset({"r", "w"}),
  }[mode]


def _required_binding_contract(
  program: CompiledProgram, summaries: Mapping[str, _TaskSummary], inputs: Sequence[ExecGlobalInput]
) -> dict[str, tuple[int, str]]:
  required: list[set[str]] = [set() for _ in inputs]
  minimum = [item.size_bytes for item in inputs]
  if isinstance(program.entry, ExecModel):
    submits = [item for item in program.entry.body if item.op == "submit"]
    for op in submits:
      for access in summaries[op.binding_id].accesses:
        actual = op.actual_inputs[access.formal_index]
        required[actual].add("w" if access.writing else "r")
        minimum[actual] = max(minimum[actual], access.end)
    for op in program.entry.body:
      if op.op != "host_call" or not isinstance(op.command, ExecHostCall):
        continue
      for host_access in op.command.accesses:
        if not 0 <= host_access.input_index < len(inputs):
          _fail("host call access names an out-of-range global input")
        required[host_access.input_index].update(_host_access_permissions(host_access.mode))
        minimum[host_access.input_index] = max(
          minimum[host_access.input_index], host_access.offset + host_access.bytes
        )
  else:
    summary = summaries[program.entry.binding_id]
    for access in summary.accesses:
      required[access.formal_index].add("w" if access.writing else "r")
      minimum[access.formal_index] = max(minimum[access.formal_index], access.end)
  return {
    item.name: (minimum[index], _guard_permissions(required[index])) for index, item in enumerate(inputs)
  }


def _verify_binding_guards(
  program: CompiledProgram, summaries: Mapping[str, _TaskSummary], inputs: Sequence[ExecGlobalInput]
) -> None:
  required = _required_binding_contract(program, summaries, inputs)
  if any(not isinstance(guard, BindingGuard) for guard in program.binding_guards):
    _fail("binding_guards contains an invalid value")
  guards = {guard.name: guard for guard in program.binding_guards}
  if len(guards) != len(program.binding_guards) or set(guards) != set(required):
    _fail("binding_guards must name every global input exactly once")
  for name, (minimum, permissions) in required.items():
    guard = guards[name]
    _uint(guard.minimum_bytes, f"binding guard {name!r}.minimum_bytes")
    if guard.minimum_bytes < minimum:
      _fail(f"binding guard {name!r} is smaller than a real executable view")
    if guard.permissions not in ("r", "w", "rw") or not set(permissions) <= set(guard.permissions):
      _fail(f"binding guard {name!r} weakens required permissions")


def verify_actual_bindings(
  program: CompiledProgram, actual_bindings: Mapping[str, GlobalBinding], hw: HardwareConfig
) -> None:
  """Verify late-bound HBM ranges, permissions, and the compiled alias guard."""
  _tasks, inputs = _entry_tasks(program)
  # Semantic verification already established that guards cover all actual effects.
  guards = {guard.name: guard for guard in program.binding_guards}
  if set(actual_bindings) != {item.name for item in inputs}:
    _fail("actual bindings do not exactly match executable global inputs")
  ordered: list[GlobalBinding] = []
  for name, binding in actual_bindings.items():
    if not isinstance(binding, GlobalBinding) or binding.name != name:
      _fail(f"actual binding {name!r} has inconsistent identity")
    if binding.permissions not in ("r", "w", "rw"):
      _fail(f"actual binding {name!r} has invalid permissions")
    _uint(binding.base_iova, f"actual binding {name!r}.base_iova")
    _uint(binding.size_bytes, f"actual binding {name!r}.size_bytes", positive=True)
    guard = guards[name]
    if binding.size_bytes < guard.minimum_bytes:
      _fail(f"actual binding {name!r} is smaller than its compiled range guard")
    if not set(guard.permissions) <= set(binding.permissions):
      _fail(f"actual binding {name!r} violates its compiled permission guard")
    if binding.base_iova + binding.size_bytes > hw.hbm_capacity_bytes:
      _fail(f"actual binding {name!r} exceeds HBM capacity")
    ordered.append(binding)
  ordered.sort(key=lambda item: item.base_iova)
  for left, right in pairwise(ordered):
    if right.base_iova < left.base_iova + left.size_bytes:
      _fail(f"actual bindings {left.name!r} and {right.name!r} overlap")
