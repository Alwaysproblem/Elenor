"""Immutable executable package and strict persistent codec, without compiler imports."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import NoReturn, TypeGuard, TypeVar

from . import execution_ir as execution, profiles
from .execution_ir import ExecDeviceOp, ExecModel, ExecTileGroupTask, GlobalBinding
from .immutable import FrozenMap, digest, freeze
from .profiles import CallBinding, ProfileRegistry, ProfileState, ResourceBudget, SourceRef

_UINT64_MAX = (1 << 64) - 1
_SIGNED64_MIN = -(1 << 63)
_SIGNED64_MAX = (1 << 63) - 1
_DTYPE_BYTES = {"i8": 1, "bf16": 2, "f16": 2, "i32": 4, "f32": 4}

_T = TypeVar("_T")


# Runtime remains Python 3.11; PEP 695 function syntax requires Python 3.12.
def _is_exact_type(value: object, expected: type[_T]) -> TypeGuard[_T]:  # noqa: UP047
  """Narrow an allowlisted value without admitting subclasses."""
  return type(value) is expected


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
  if type(value) is not str or (not value and not allow_empty) or (bool(value) and not value.strip()):
    raise ValueError(f"{name} must be a {'string' if allow_empty else 'nonempty string'}")
  return value


def _uint(value: object, name: str, *, positive: bool = False) -> int:
  if type(value) is not int or not (int(positive) <= value <= _UINT64_MAX):
    raise ValueError(f"{name} must be a {'positive ' if positive else ''}uint64")
  return value


def _bounded_integer(value: object, name: str) -> int:
  if type(value) is not int or not _SIGNED64_MIN <= value <= _UINT64_MAX:
    raise ValueError(f"{name} must be an integer in the signed/uint64 codec range")
  return value


def _signed(value: object, name: str) -> int:
  if type(value) is not int or not _SIGNED64_MIN <= value <= _SIGNED64_MAX:
    raise ValueError(f"{name} must be a signed 64-bit integer")
  return value


def _tuple(value: object, name: str) -> tuple:
  if not isinstance(value, (tuple, list)):
    raise ValueError(f"{name} must be a sequence")
  return tuple(value)


def _strings(
  value: object, name: str, *, unique: bool = False, allow_empty_items: bool = False
) -> tuple[str, ...]:
  result = _tuple(value, name)
  if any(type(item) is not str or (not allow_empty_items and not item.strip()) for item in result):
    raise ValueError(f"{name} must contain strings")
  if unique and len(result) != len(set(result)):
    raise ValueError(f"{name} must not contain duplicates")
  return result


def _optional_text(value: object, name: str) -> None:
  if value is not None:
    _text(value, name)


def _plain_value(value: object, name: str) -> None:
  if value is None or type(value) in (str, bool):
    return
  if type(value) is int:
    _bounded_integer(value, name)
    return
  if type(value) is float:
    if not math.isfinite(value):
      raise ValueError(f"{name} must be finite")
    return
  if isinstance(value, tuple):
    for index, item in enumerate(value):
      _plain_value(item, f"{name}[{index}]")
    return
  if isinstance(value, FrozenMap):
    for key, item in value.items():
      if type(key) not in (str, int) or type(key) is bool:
        raise ValueError(f"{name} has an invalid mapping key")
      if type(key) is int:
        _bounded_integer(key, f"{name} key")
      _plain_value(item, f"{name}[{key!r}]")
    return
  raise ValueError(f"{name} contains unsupported {type(value).__name__}")


def _freeze_mapping(value: object, name: str) -> FrozenMap:
  if not isinstance(value, Mapping):
    raise ValueError(f"{name} must be a mapping")
  return FrozenMap(value)


@dataclass(frozen=True)
class WorkloadInfo:
  name: str
  description: str
  expected: Mapping[str, object]

  def __post_init__(self):
    _text(self.name, "workload name")
    _text(self.description, "workload description")
    expected = _freeze_mapping(self.expected, "workload expected")
    if any(type(key) is not str or not key.strip() for key in expected):
      raise ValueError("workload expected keys must be nonempty strings")
    _plain_value(expected, "workload expected")
    object.__setattr__(self, "expected", expected)


@dataclass(frozen=True)
class Relocation:
  binding_id: str
  scope: str
  ordinal: int
  field: str
  kind: str
  argument_index: int = -1
  role_id: int = -1

  def __post_init__(self):
    _text(self.binding_id, "relocation binding_id")
    _text(self.scope, "relocation scope")
    _uint(self.ordinal, "relocation ordinal")
    _text(self.field, "relocation field")
    _text(self.kind, "relocation kind")
    if type(self.argument_index) is not int or self.argument_index < -1:
      raise ValueError("relocation argument_index must be -1 or nonnegative")
    if type(self.role_id) is not int or self.role_id < -1:
      raise ValueError("relocation role_id must be -1 or nonnegative")
    if self.argument_index > _UINT64_MAX or self.role_id > _UINT64_MAX:
      raise ValueError("relocation index exceeds uint64")
    form = (self.scope, self.field, self.kind)
    simple = {
      ("group", "dst", "event"),
      ("group", "dependencies", "event_tuple"),
      ("group", "args", "dispatch_events"),
      ("group", "args", "event_tuple"),
      ("group", "args", "release_events"),
      ("group", "args", "profile_frontier"),
      ("group", "args", "maintenance_dependencies"),
      ("task", "completion_event", "event"),
      ("task", "event_uses", "event_map"),
      ("stream", "queue_id", "queue"),
    }
    if form in simple:
      if self.argument_index != -1 or self.role_id != -1:
        raise ValueError("simple relocation cannot carry argument or role indices")
      if self.scope == "task" and self.ordinal != 0:
        raise ValueError("task relocation requires ordinal=0")
    elif form == ("group", "args", "queue"):
      if self.argument_index != 0 or self.role_id != -1:
        raise ValueError("stream-init relocation requires argument_index=0")
    elif form == ("context", "global_view", "binding"):
      if self.argument_index < 0 or self.role_id != -1:
        raise ValueError("Context binding relocation requires an input index")
    elif form in (("role", "in_stream", "queue"), ("role", "out_stream", "queue")):
      if self.ordinal != 0 or self.argument_index != -1 or self.role_id < 0:
        raise ValueError("role stream relocation requires ordinal=0 and a role ID")
    elif form == ("tile_role", "global_actuals", "binding"):
      if self.argument_index < 0 or self.role_id < 0:
        raise ValueError("Tile role binding relocation requires input and role indices")
    else:
      raise ValueError("unsupported relocation form")


@dataclass(frozen=True)
class BindingGuard:
  name: str
  minimum_bytes: int
  permissions: str

  def __post_init__(self):
    _text(self.name, "binding guard name")
    _uint(self.minimum_bytes, "binding guard minimum_bytes")
    if self.permissions not in ("r", "w", "rw"):
      raise ValueError("invalid binding guard permissions")


@dataclass(frozen=True)
class CompiledProgram:
  schema_version: int
  compiler_abi: str
  source_hash: str
  source_ir: str
  registry: ProfileRegistry
  registry_hash: str
  target_hash: str
  artifact_hash: str
  entry_kind: str
  entry: ExecTileGroupTask | ExecModel
  entry_prefix: tuple[ExecDeviceOp, ...]
  call_bindings: Mapping[str, CallBinding]
  relocations: tuple[Relocation, ...]
  source_map: Mapping[str, SourceRef]
  dependency_proofs: Mapping[str, object]
  entry_profiles: ProfileState
  exit_profiles: ProfileState
  static_effects: Mapping[str, object]
  resource_budgets: Mapping[str, ResourceBudget]
  binding_guards: tuple[BindingGuard, ...]
  workload_info: WorkloadInfo

  def __post_init__(self):
    for item in fields(self):
      object.__setattr__(self, item.name, freeze(getattr(self, item.name)))
    _validate_compiled_program_value(self, allow_unsealed=True)


@dataclass(frozen=True)
class LoadedProgram:
  compiled: CompiledProgram
  actual_bindings: Mapping[str, GlobalBinding]
  target_hash: str

  def __post_init__(self):
    if not isinstance(self.compiled, CompiledProgram):
      raise ValueError("loaded program requires CompiledProgram")
    bindings = _freeze_mapping(self.actual_bindings, "actual_bindings")
    for name, binding in bindings.items():
      _text(name, "actual binding name")
      if not isinstance(binding, GlobalBinding) or binding.name != name:
        raise ValueError("actual binding map is inconsistent")
      _validate_executable_value(binding, f"actual_bindings[{name!r}]")
    object.__setattr__(self, "actual_bindings", bindings)
    if (
      type(self.target_hash) is not str
      or not _HEX.fullmatch(self.target_hash)
      or int(self.target_hash, 16) == 0
    ):
      raise ValueError("loaded target_hash must be a nonzero SHA-256 digest")


# Explicit type/opcode allowlist; the input never names importable Python code.
_CLASSES = (
  WorkloadInfo,
  Relocation,
  BindingGuard,
  CompiledProgram,
  execution.GlobalBinding,
  execution.ExecGlobalInput,
  execution.ExecMemoryView,
  execution.ExecProfiledAccess,
  execution.ExecGatherDesc,
  execution.ExecTransfer,
  execution.ExecL2Buffer,
  execution.ExecL1Buffer,
  execution.ExecTileFormal,
  execution.ExecTaskDomain,
  execution.ExecSignalPolicy,
  execution.ExecDispatchRequest,
  execution.ExecReleaseRequest,
  execution.ExecTileInst,
  execution.ExecGroupAction,
  execution.ExecStreamDesc,
  execution.ExecEngineDesc,
  execution.ExecTileProgram,
  execution.ExecTileRoleBinding,
  execution.ExecTileGroupTask,
  execution.ExecDeviceOp,
  execution.ExecModel,
  profiles.MemoryLevelTarget,
  profiles.MemoryTargetConfig,
  profiles.ProfileBytes,
  profiles.ProfileLevelSource,
  profiles.ProfileSourceConfig,
  profiles.MemoryProfile,
  profiles.ProfileRegistry,
  profiles.CacheRequirement,
  profiles.ContextResources,
  profiles.TileResources,
  profiles.BufferLayout,
  profiles.ArenaLayout,
  profiles.SourceRef,
  profiles.ProfileState,
  profiles.ProfileReconfigDesc,
  profiles.MaintenanceRange,
  profiles.MemoryMaintenanceDesc,
  profiles.CallBinding,
  profiles.ResourceBudget,
)
_TYPES = {cls.__name__: cls for cls in _CLASSES}
_ENUMS = {
  cls.__name__: cls
  for cls in (execution.ExecTileOp, execution.ExecGroupActionOp, execution.ExecGatherOutcome)
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_VALUES = (
  profiles.MemoryLevelTarget,
  profiles.MemoryTargetConfig,
  profiles.ProfileBytes,
  profiles.ProfileLevelSource,
  profiles.ProfileSourceConfig,
  profiles.MemoryProfile,
  profiles.ProfileRegistry,
  profiles.CacheRequirement,
  profiles.ContextResources,
  profiles.TileResources,
  profiles.BufferLayout,
  profiles.ArenaLayout,
  profiles.SourceRef,
  profiles.ProfileState,
  profiles.ProfileReconfigDesc,
  profiles.MaintenanceRange,
  profiles.MemoryMaintenanceDesc,
  profiles.CallBinding,
  profiles.ResourceBudget,
)


def _digest_text(value: object, name: str, *, allow_zero: bool = False) -> str:
  if type(value) is not str or not _HEX.fullmatch(value):
    raise ValueError(f"{name} must be a canonical SHA-256 digest")
  if not allow_zero and int(value, 16) == 0:
    raise ValueError(f"{name} must be nonzero")
  return value


def _shape(value: object, name: str, *, empty: bool = False) -> tuple[int, ...]:
  result = _tuple(value, name)
  if (not empty and not result) or any(
    type(item) is not int or not 0 < item <= _UINT64_MAX for item in result
  ):
    raise ValueError(f"{name} must contain positive uint64 dimensions")
  return result


def _typed_tuple(value: object, cls: type, name: str, *, nonempty: bool = False) -> tuple:
  result = _tuple(value, name)
  if (nonempty and not result) or any(type(item) is not cls for item in result):
    raise ValueError(f"{name} must contain {cls.__name__} values")
  return result


def _validate_source_ref(value: object, name: str) -> None:
  if type(value) is not SourceRef:
    raise ValueError(f"{name} requires SourceRef")


def _validate_view(view: object, name: str) -> None:
  if type(view) is not execution.ExecMemoryView:
    raise ValueError(f"{name} must be ExecMemoryView")
  if view.space not in ("global", "l2", "l1"):
    raise ValueError(f"{name}.space is invalid")
  _text(view.base, f"{name}.base")
  backing = _shape(view.backing_dims, f"{name}.backing_dims")
  dims = _shape(view.dims, f"{name}.dims")
  offsets = _tuple(view.offsets, f"{name}.offsets")
  strides = _tuple(view.strides, f"{name}.strides")
  if len(backing) != len(dims) or len(backing) != len(offsets) or len(backing) != len(strides):
    raise ValueError(f"{name} has inconsistent rank")
  for index, offset in enumerate(offsets):
    _uint(offset, f"{name}.offsets[{index}]")
  if any(type(stride) is not int or stride != 1 for stride in strides):
    raise ValueError(f"{name}.strides must contain unit integers")
  if any(offset + size > parent for offset, size, parent in zip(offsets, dims, backing)):
    raise ValueError(f"{name} exceeds its backing shape")
  if view.dtype not in _DTYPE_BYTES:
    raise ValueError(f"{name} has an invalid dtype")
  _uint(view.element_bytes, f"{name}.element_bytes", positive=True)
  _uint(view.bytes, f"{name}.bytes", positive=True)
  if view.element_bytes != _DTYPE_BYTES[view.dtype]:
    raise ValueError(f"{name} has an invalid dtype width")
  expected = math.prod(dims) * view.element_bytes
  if expected > _UINT64_MAX or view.bytes != expected:
    raise ValueError(f"{name}.bytes does not conserve its shape")
  if view.task_dim is not None and (
    type(view.task_dim) is not int or not 0 <= view.task_dim < len(dims) or view.space != "l2"
  ):
    raise ValueError(f"{name}.task_dim is invalid")


def _validate_buffer(buffer: object, name: str) -> None:
  if type(buffer) is execution.ExecL1Buffer:
    _text(buffer.name, f"{name}.name")
  elif type(buffer) is execution.ExecL2Buffer:
    _text(buffer.slot, f"{name}.slot")
    if buffer.role not in ("in", "out", "inout"):
      raise ValueError(f"{name}.role is invalid")
  else:
    raise ValueError(f"{name} is not a buffer")
  dims = _shape(buffer.dims, f"{name}.dims")
  if buffer.dtype not in _DTYPE_BYTES:
    raise ValueError(f"{name} has an invalid dtype")
  _uint(buffer.element_bytes, f"{name}.element_bytes", positive=True)
  _uint(buffer.alignment, f"{name}.alignment", positive=True)
  _uint(buffer.bytes, f"{name}.bytes", positive=True)
  if buffer.element_bytes != _DTYPE_BYTES[buffer.dtype]:
    raise ValueError(f"{name} has an invalid dtype width")
  expected = math.prod(dims) * buffer.element_bytes
  if expected > _UINT64_MAX or buffer.bytes != expected:
    raise ValueError(f"{name}.bytes does not conserve its shape")


def _validate_literal(value: object, name: str) -> None:
  if value is None or type(value) in (str, bool):
    return
  if type(value) is int:
    _signed(value, name)
    return
  if isinstance(value, tuple):
    for index, item in enumerate(value):
      _validate_literal(item, f"{name}[{index}]")
    return
  raise ValueError(f"{name} contains a non-literal {type(value).__name__}")


def _validate_instruction(inst: execution.ExecTileInst, name: str) -> None:
  if type(inst.op) is not execution.ExecTileOp:
    raise ValueError(f"{name}.op is not allowlisted")
  _optional_text(inst.dst, f"{name}.dst")
  args = _tuple(inst.args, f"{name}.args")
  _optional_text(inst.label, f"{name}.label")
  _text(inst.comment, f"{name}.comment", allow_empty=True)
  _validate_source_ref(inst.source_ref, f"{name}.source_ref")
  _text(inst.instruction_id, f"{name}.instruction_id")
  if inst.op is execution.ExecTileOp.ALLOC_L1:
    if len(args) != 2:
      raise ValueError(f"{name} has invalid alloc.l1 arguments")
    _text(args[0], f"{name}.buffer_id")
    _uint(args[1], f"{name}.layout_index")
  elif inst.op is execution.ExecTileOp.FREE_L1:
    if len(args) != 1:
      raise ValueError(f"{name} has invalid free.l1 arguments")
    _text(args[0], f"{name}.buffer_id")
  elif inst.op in (
    execution.ExecTileOp.LAUNCH_BOA,
    execution.ExecTileOp.LAUNCH_EVU,
    execution.ExecTileOp.LAUNCH_MFE,
    execution.ExecTileOp.LAUNCH_GATHER,
  ):
    if len(args) != 1:
      raise ValueError(f"{name} has invalid launch arguments")
    _text(args[0], f"{name}.descriptor")
  elif inst.op in (execution.ExecTileOp.WAIT, execution.ExecTileOp.WAITALL):
    if not args or (inst.op is execution.ExecTileOp.WAIT and len(args) != 1):
      raise ValueError(f"{name} has invalid wait arguments")
    _strings(args, f"{name}.args", unique=True)
  elif inst.op is execution.ExecTileOp.SIGNAL_PHASE:
    if len(args) != 2 or args[0] not in ("input_released", "output_ready"):
      raise ValueError(f"{name} has invalid signal arguments")
    _uint(args[1], f"{name}.task_formal")
    if args[1] != 0:
      raise ValueError(f"{name} has invalid signal arguments")
  elif inst.op is execution.ExecTileOp.RET:
    if args:
      raise ValueError(f"{name} RET cannot have arguments")
  else:
    _validate_literal(args, f"{name}.args")


def _validate_engine(desc: execution.ExecEngineDesc, name: str) -> None:
  _text(desc.name, f"{name}.name")
  _text(desc.kind, f"{name}.kind")
  _text(desc.op, f"{name}.op")
  if not isinstance(desc.params, FrozenMap):
    raise ValueError(f"{name}.params must be immutable")
  if desc.kind == "MFE" and desc.op in ("load", "store"):
    if desc.params or type(desc.transfer) is not execution.ExecTransfer:
      raise ValueError(f"{name} has invalid transfer descriptor fields")
    _validate_executable_value(desc.transfer, f"{name}.transfer")
  elif desc.kind == "MFE" and desc.op == "gather":
    if set(desc.params) != {"gather"} or type(desc.params["gather"]) is not execution.ExecGatherDesc:
      raise ValueError(f"{name} has invalid gather descriptor fields")
    _validate_executable_value(desc.params["gather"], f"{name}.gather")
    if desc.transfer is not None:
      raise ValueError(f"{name}.transfer must be absent")
  elif desc.kind == "EVU":
    required = {"ops"} | ({"bytes", "exponent"} if desc.op == "pow" else set())
    if set(desc.params) != required or desc.transfer is not None:
      raise ValueError(f"{name} has invalid EVU parameters")
    for key in required:
      if key == "exponent":
        _signed(desc.params[key], f"{name}.{key}")
      else:
        _uint(desc.params[key], f"{name}.{key}", positive=True)
  elif desc.kind == "BOA":
    if desc.transfer is not None or not {"m", "n", "k", "ops"} <= set(desc.params) <= {
      "m",
      "n",
      "k",
      "ops",
      "accumulate",
    }:
      raise ValueError(f"{name} has invalid BOA parameters")
    for key in ("m", "n", "k", "ops"):
      _uint(desc.params[key], f"{name}.{key}", positive=True)
    if "accumulate" in desc.params and type(desc.params["accumulate"]) is not bool:
      raise ValueError(f"{name}.accumulate must be bool")
  else:
    raise ValueError(f"{name} has an unsupported engine opcode")


def _validate_group_action(action: execution.ExecGroupAction, name: str) -> None:
  if type(action.op) is not execution.ExecGroupActionOp:
    raise ValueError(f"{name}.op is not allowlisted")
  args = _tuple(action.args, f"{name}.args")
  _optional_text(action.dst, f"{name}.dst")
  _text(action.comment, f"{name}.comment", allow_empty=True)
  _strings(action.dependencies, f"{name}.dependencies", unique=True)
  _strings(action.reads, f"{name}.reads", unique=True)
  _strings(action.writes, f"{name}.writes", unique=True)
  _validate_source_ref(action.source_ref, f"{name}.source_ref")
  _text(action.instruction_id, f"{name}.instruction_id")
  expected: type | None = None
  if action.op in (execution.ExecGroupActionOp.DMA_PREFETCH, execution.ExecGroupActionOp.DMA_STORE):
    if len(args) != 2:
      raise ValueError(f"{name} has invalid DMA arguments")
    _text(args[0], f"{name}.engine")
    expected = execution.ExecTransfer
    candidate = args[1]
  elif action.op is execution.ExecGroupActionOp.DISPATCH_ROLE:
    expected, candidate = execution.ExecDispatchRequest, args[0] if len(args) == 1 else None
  elif action.op is execution.ExecGroupActionOp.RELEASE_L2:
    expected, candidate = execution.ExecReleaseRequest, args[0] if len(args) == 1 else None
  elif action.op is execution.ExecGroupActionOp.PROFILE_RECONFIG:
    expected, candidate = profiles.ProfileReconfigDesc, args[0] if len(args) == 1 else None
  elif action.op is execution.ExecGroupActionOp.MEMORY_MAINTENANCE:
    expected, candidate = profiles.MemoryMaintenanceDesc, args[0] if len(args) == 1 else None
  elif action.op is execution.ExecGroupActionOp.BIND_L2_VIEW:
    if len(args) != 2:
      raise ValueError(f"{name} has invalid bind arguments")
    _text(args[0], f"{name}.buffer_id")
    _uint(args[1], f"{name}.layout_index")
    return
  elif action.op in (execution.ExecGroupActionOp.WAIT_EVENT, execution.ExecGroupActionOp.SIGNAL_EVENT):
    if len(args) != 1:
      raise ValueError(f"{name} has invalid event arguments")
    _text(args[0], f"{name}.event")
    return
  elif action.op is execution.ExecGroupActionOp.BARRIER_GROUP:
    if args:
      raise ValueError(f"{name} barrier cannot have arguments")
    return
  elif action.op is execution.ExecGroupActionOp.INIT_STREAM:
    if len(args) != 4:
      raise ValueError(f"{name} has invalid stream arguments")
    _uint(args[0], f"{name}.queue_id")
    _uint(args[1], f"{name}.depth", positive=True)
    _uint(args[2], f"{name}.producer_mask", positive=True)
    _uint(args[3], f"{name}.consumer_mask", positive=True)
    return
  else:
    _validate_literal(args, f"{name}.args")
    return
  if expected is None or type(candidate) is not expected:
    raise ValueError(f"{name} has an invalid typed argument")
  _validate_executable_value(candidate, f"{name}.args")


def _validate_executable_value(value: object, name: str) -> None:
  kind = type(value)
  if kind in _PROFILE_VALUES:
    return
  if _is_exact_type(value, WorkloadInfo):
    _plain_value(value.expected, f"{name}.expected")
    return
  if kind in (Relocation, BindingGuard):
    return
  if _is_exact_type(value, CompiledProgram):
    _validate_compiled_program_value(value, allow_unsealed=True)
    return
  if _is_exact_type(value, execution.GlobalBinding):
    _text(value.name, f"{name}.name")
    _uint(value.base_iova, f"{name}.base_iova")
    _uint(value.size_bytes, f"{name}.size_bytes", positive=True)
    if value.base_iova + value.size_bytes > _UINT64_MAX:
      raise ValueError(f"{name} address range overflows uint64")
    if value.permissions not in ("r", "w", "rw"):
      raise ValueError(f"{name}.permissions is invalid")
    return
  if _is_exact_type(value, execution.ExecGlobalInput):
    _text(value.name, f"{name}.name")
    dims = _shape(value.dims, f"{name}.dims")
    if value.dtype not in _DTYPE_BYTES:
      raise ValueError(f"{name}.dtype is invalid")
    _uint(value.size_bytes, f"{name}.size_bytes", positive=True)
    expected_size = math.prod(dims) * _DTYPE_BYTES[value.dtype]
    if expected_size > _UINT64_MAX or value.size_bytes != expected_size:
      raise ValueError(f"{name}.size_bytes does not conserve its shape")
    return
  if _is_exact_type(value, execution.ExecMemoryView):
    _validate_view(value, name)
    return
  if _is_exact_type(value, execution.ExecProfiledAccess):
    _text(value.request_id, f"{name}.request_id")
    if type(value.outcome) is not execution.ExecGatherOutcome:
      raise ValueError(f"{name}.outcome is invalid")
    _uint(value.bytes, f"{name}.bytes", positive=True)
    _optional_text(value.line_token, f"{name}.line_token")
    _optional_text(value.merge_group, f"{name}.merge_group")
    return
  if _is_exact_type(value, execution.ExecGatherDesc):
    _validate_view(value.source, f"{name}.source")
    _validate_view(value.indices, f"{name}.indices")
    _validate_view(value.destination, f"{name}.destination")
    _uint(value.result_bytes, f"{name}.result_bytes", positive=True)
    _uint(value.cache_target_bytes, f"{name}.cache_target_bytes")
    _uint(value.l1_mshr_hint, f"{name}.l1_mshr_hint", positive=True)
    accesses = _typed_tuple(value.accesses, execution.ExecProfiledAccess, f"{name}.accesses", nonempty=True)
    for index, item in enumerate(accesses):
      _validate_executable_value(item, f"{name}.accesses[{index}]")
    if sum(item.bytes for item in accesses) != value.result_bytes:
      raise ValueError(f"{name}.accesses do not conserve result_bytes")
    return
  if _is_exact_type(value, execution.ExecTransfer):
    _validate_view(value.src, f"{name}.src")
    _validate_view(value.dst, f"{name}.dst")
    _uint(value.bytes, f"{name}.bytes", positive=True)
    if value.bytes != value.src.bytes or value.bytes != value.dst.bytes:
      raise ValueError(f"{name}.bytes does not conserve its views")
    return
  if kind in (execution.ExecL1Buffer, execution.ExecL2Buffer):
    _validate_buffer(value, name)
    return
  if _is_exact_type(value, execution.ExecTileFormal):
    if value.space not in ("task", "global", "l2"):
      raise ValueError(f"{name}.space is invalid")
    dims = _shape(value.dims, f"{name}.dims", empty=value.space == "task")
    if value.space == "task":
      if dims or value.dtype != "":
        raise ValueError(f"{name} has an invalid task formal")
    elif value.dtype not in _DTYPE_BYTES:
      raise ValueError(f"{name}.dtype is invalid")
    return
  if _is_exact_type(value, execution.ExecTaskDomain):
    _uint(value.from_task, f"{name}.from_task")
    _uint(value.to_task, f"{name}.to_task", positive=True)
    if value.to_task <= value.from_task:
      raise ValueError(f"{name} is empty")
    return
  if _is_exact_type(value, execution.ExecSignalPolicy):
    _optional_text(value.input_released, f"{name}.input_released")
    _optional_text(value.output_ready, f"{name}.output_ready")
    return
  if _is_exact_type(value, execution.ExecDispatchRequest):
    _uint(value.role_id, f"{name}.role_id")
    _uint(value.dispatch_ordinal, f"{name}.dispatch_ordinal")
    if type(value.signal_policy) is not execution.ExecSignalPolicy:
      raise ValueError(f"{name}.signal_policy is invalid")
    _validate_executable_value(value.signal_policy, f"{name}.signal_policy")
    _text(value.input_released_event, f"{name}.input_released_event", allow_empty=True)
    _text(value.output_ready_event, f"{name}.output_ready_event", allow_empty=True)
    _uint(value.requested_l1_mode, f"{name}.requested_l1_mode")
    _uint(value.resolved_l1_mode, f"{name}.resolved_l1_mode")
    _text(value.binding_id, f"{name}.binding_id")
    return
  if _is_exact_type(value, execution.ExecReleaseRequest):
    _text(value.buffer_slot, f"{name}.buffer_slot")
    if value.buffer_role not in ("in", "out", "inout"):
      raise ValueError(f"{name}.buffer_role is invalid")
    for field_name in ("reader_dispatch_ordinals", "writer_dispatch_ordinals"):
      values = _tuple(getattr(value, field_name), f"{name}.{field_name}")
      if len(values) != len(set(values)):
        raise ValueError(f"{name}.{field_name} contains duplicates")
      for index, item in enumerate(values):
        _uint(item, f"{name}.{field_name}[{index}]")
    _strings(value.dependency_events, f"{name}.dependency_events", unique=True)
    return
  if _is_exact_type(value, execution.ExecTileInst):
    _validate_instruction(value, name)
    return
  if _is_exact_type(value, execution.ExecGroupAction):
    _validate_group_action(value, name)
    return
  if _is_exact_type(value, execution.ExecStreamDesc):
    _uint(value.queue_id, f"{name}.queue_id")
    _uint(value.depth, f"{name}.depth", positive=True)
    _uint(value.producer_mask, f"{name}.producer_mask", positive=True)
    _uint(value.consumer_mask, f"{name}.consumer_mask", positive=True)
    _uint(value.payload_slot_id, f"{name}.payload_slot_id")
    _uint(value.token_stride, f"{name}.token_stride", positive=True)
    _uint(value.pmu_stream_id, f"{name}.pmu_stream_id")
    return
  if _is_exact_type(value, execution.ExecEngineDesc):
    _validate_engine(value, name)
    return
  if _is_exact_type(value, execution.ExecTileProgram):
    _text(value.name, f"{name}.name")
    insts = _typed_tuple(value.insts, execution.ExecTileInst, f"{name}.insts", nonempty=True)
    for index, item in enumerate(insts):
      _validate_instruction(item, f"{name}.insts[{index}]")
    if not isinstance(value.descriptors, FrozenMap):
      raise ValueError(f"{name}.descriptors must be immutable")
    for key, item in value.descriptors.items():
      _text(key, f"{name}.descriptor name")
      if type(item) is not execution.ExecEngineDesc or item.name != key:
        raise ValueError(f"{name}.descriptors is inconsistent")
      _validate_engine(item, f"{name}.descriptors[{key!r}]")
    if not isinstance(value.labels, FrozenMap):
      raise ValueError(f"{name}.labels must be immutable")
    for label, index in value.labels.items():
      _text(label, f"{name}.label")
      _uint(index, f"{name}.labels[{label!r}]")
    _uint(value.program_id, f"{name}.program_id", positive=True)
    _uint(value.version, f"{name}.version", positive=True)
    if type(value.program_hash) is not int or not 0 < value.program_hash < (1 << 256):
      raise ValueError(f"{name}.program_hash is invalid")
    _typed_tuple(value.formals, execution.ExecTileFormal, f"{name}.formals", nonempty=True)
    for index, item in enumerate(value.formals):
      _validate_executable_value(item, f"{name}.formals[{index}]")
    _typed_tuple(value.l1_buffers, execution.ExecL1Buffer, f"{name}.l1_buffers")
    for index, item in enumerate(value.l1_buffers):
      _validate_buffer(item, f"{name}.l1_buffers[{index}]")
    if type(value.resource_contract) is not profiles.TileResources:
      raise ValueError(f"{name}.resource_contract is invalid")
    if type(value.layout) is not profiles.ArenaLayout:
      raise ValueError(f"{name}.layout is invalid")
    _uint(value.text_bytes, f"{name}.text_bytes", positive=True)
    return
  if _is_exact_type(value, execution.ExecTileRoleBinding):
    _uint(value.role_id, f"{name}.role_id")
    _uint(value.tile_mask, f"{name}.tile_mask", positive=True)
    if type(value.tile_program) is not execution.ExecTileProgram:
      raise ValueError(f"{name}.tile_program is invalid")
    _validate_executable_value(value.tile_program, f"{name}.tile_program")
    for field_name in ("in_stream", "out_stream", "context_id"):
      item = getattr(value, field_name)
      if item is not None:
        _uint(item, f"{name}.{field_name}")
    if type(value.task_domain) is not execution.ExecTaskDomain:
      raise ValueError(f"{name}.task_domain is invalid")
    _validate_executable_value(value.task_domain, f"{name}.task_domain")
    _strings(value.actuals, f"{name}.actuals")
    _typed_tuple(value.global_actuals, execution.ExecMemoryView, f"{name}.global_actuals")
    for index, item in enumerate(value.global_actuals):
      _validate_view(item, f"{name}.global_actuals[{index}]")
    _strings(value.read_actuals, f"{name}.read_actuals", unique=True)
    _strings(value.write_actuals, f"{name}.write_actuals", unique=True)
    return
  if _is_exact_type(value, execution.ExecTileGroupTask):
    _text(value.name, f"{name}.name")
    _typed_tuple(value.actions, execution.ExecGroupAction, f"{name}.actions", nonempty=True)
    for index, item in enumerate(value.actions):
      _validate_group_action(item, f"{name}.actions[{index}]")
    _typed_tuple(value.streams, execution.ExecStreamDesc, f"{name}.streams")
    for index, item in enumerate(value.streams):
      _validate_executable_value(item, f"{name}.streams[{index}]")
    if not isinstance(value.role_bindings, FrozenMap):
      raise ValueError(f"{name}.role_bindings must be immutable")
    for role_id, binding in value.role_bindings.items():
      _uint(role_id, f"{name}.role_id")
      if type(binding) is not execution.ExecTileRoleBinding or binding.role_id != role_id:
        raise ValueError(f"{name}.role_bindings is inconsistent")
      _validate_executable_value(binding, f"{name}.role_bindings[{role_id}]")
    _text(value.completion_event, f"{name}.completion_event")
    _typed_tuple(value.global_inputs, execution.ExecGlobalInput, f"{name}.global_inputs")
    for index, item in enumerate(value.global_inputs):
      _validate_executable_value(item, f"{name}.global_inputs[{index}]")
    _typed_tuple(value.l2_buffers, execution.ExecL2Buffer, f"{name}.l2_buffers")
    for index, item in enumerate(value.l2_buffers):
      _validate_buffer(item, f"{name}.l2_buffers[{index}]")
    if type(value.resource_contract) is not profiles.ContextResources:
      raise ValueError(f"{name}.resource_contract is invalid")
    if type(value.layout) is not profiles.ArenaLayout:
      raise ValueError(f"{name}.layout is invalid")
    _text(value.binding_id, f"{name}.binding_id")
    if not isinstance(value.event_uses, FrozenMap):
      raise ValueError(f"{name}.event_uses must be immutable")
    for event, count in value.event_uses.items():
      _text(event, f"{name}.event_uses event")
      _uint(count, f"{name}.event_uses[{event!r}]")
    return
  if _is_exact_type(value, execution.ExecDeviceOp):
    if value.op not in ("submit", "await", "return", "profile_reconfig", "memory_maintenance"):
      raise ValueError(f"{name}.op is not allowlisted")
    for field_name in ("ctx_name", "event_tag", "callsite_id", "binding_id"):
      _text(getattr(value, field_name), f"{name}.{field_name}", allow_empty=True)
    for index, item in enumerate(_tuple(value.actual_inputs, f"{name}.actual_inputs")):
      _uint(item, f"{name}.actual_inputs[{index}]")
    _strings(value.dependencies, f"{name}.dependencies", unique=True)
    if value.command is not None:
      command_type = (
        profiles.ProfileReconfigDesc
        if value.op == "profile_reconfig"
        else profiles.MemoryMaintenanceDesc
        if value.op == "memory_maintenance"
        else None
      )
      if command_type is None or type(value.command) is not command_type:
        raise ValueError(f"{name}.command is invalid")
    elif value.op in ("profile_reconfig", "memory_maintenance"):
      raise ValueError(f"{name}.command is missing")
    _validate_source_ref(value.source_ref, f"{name}.source_ref")
    _text(value.instruction_id, f"{name}.instruction_id")
    return
  if _is_exact_type(value, execution.ExecModel):
    _text(value.name, f"{name}.name")
    if not isinstance(value.tasks, FrozenMap) or not isinstance(value.context_pins, FrozenMap):
      raise ValueError(f"{name} task maps must be immutable")
    for binding_id, task in value.tasks.items():
      _text(binding_id, f"{name}.tasks key")
      if type(task) is not execution.ExecTileGroupTask or task.binding_id != binding_id:
        raise ValueError(f"{name}.tasks is inconsistent")
      _validate_executable_value(task, f"{name}.tasks[{binding_id!r}]")
    for binding_id, pin in value.context_pins.items():
      _text(binding_id, f"{name}.context_pins key")
      if pin is not None:
        _uint(pin, f"{name}.context_pins[{binding_id!r}]")
    _typed_tuple(value.body, execution.ExecDeviceOp, f"{name}.body", nonempty=True)
    for index, item in enumerate(value.body):
      _validate_executable_value(item, f"{name}.body[{index}]")
    _typed_tuple(value.inputs, execution.ExecGlobalInput, f"{name}.inputs")
    for index, item in enumerate(value.inputs):
      _validate_executable_value(item, f"{name}.inputs[{index}]")
    return
  raise ValueError(f"{name} has unregistered type {kind.__name__}")


def _validate_compiled_program_value(program: CompiledProgram, *, allow_unsealed: bool) -> None:
  if type(program.schema_version) is not int or program.schema_version != 1 or program.compiler_abi != "v0":
    raise ValueError("unsupported compiled schema or ABI")
  _text(program.source_ir, "source_ir")
  _digest_text(program.source_hash, "source_hash")
  if digest(program.source_ir) != program.source_hash:
    raise ValueError("source_hash does not match source_ir")
  _digest_text(program.registry_hash, "registry_hash")
  _digest_text(program.target_hash, "target_hash")
  _digest_text(program.artifact_hash, "artifact_hash", allow_zero=allow_unsealed)
  if (
    type(program.registry) is not ProfileRegistry or program.registry.registry_hash != program.registry_hash
  ):
    raise ValueError("embedded Registry hash disagrees with package")
  if program.entry_kind == "standalone":
    if type(program.entry) is not ExecTileGroupTask:
      raise ValueError("standalone entry must be ExecTileGroupTask")
  elif program.entry_kind == "model":
    if type(program.entry) is not ExecModel:
      raise ValueError("model entry must be ExecModel")
  else:
    raise ValueError("invalid compiled entry kind")
  _validate_executable_value(program.entry, "entry")
  _typed_tuple(program.entry_prefix, ExecDeviceOp, "entry_prefix")
  for index, item in enumerate(program.entry_prefix):
    _validate_executable_value(item, f"entry_prefix[{index}]")
  typed_maps = (
    ("call_bindings", program.call_bindings, CallBinding),
    ("source_map", program.source_map, SourceRef),
    ("resource_budgets", program.resource_budgets, ResourceBudget),
  )
  for map_name, mapping, value_type in typed_maps:
    if not isinstance(mapping, FrozenMap):
      raise ValueError(f"{map_name} must be immutable")
    for key, item in mapping.items():
      _text(key, f"{map_name} key")
      if type(item) is not value_type:
        raise ValueError(f"{map_name}[{key!r}] has invalid type")
  _typed_tuple(program.relocations, Relocation, "relocations")
  for map_name in ("dependency_proofs", "static_effects"):
    mapping = getattr(program, map_name)
    if not isinstance(mapping, FrozenMap):
      raise ValueError(f"{map_name} must be immutable")
    if any(type(key) is not str or not key.strip() for key in mapping):
      raise ValueError(f"{map_name} keys must be nonempty strings")
    _plain_value(mapping, map_name)
  if type(program.entry_profiles) is not ProfileState or type(program.exit_profiles) is not ProfileState:
    raise ValueError("entry_profiles and exit_profiles must be ProfileState values")
  _typed_tuple(program.binding_guards, BindingGuard, "binding_guards")
  if type(program.workload_info) is not WorkloadInfo:
    raise ValueError("workload_info must be WorkloadInfo")


def _encode(value):
  if isinstance(value, Enum):
    if type(value).__name__ not in _ENUMS:
      raise ValueError("unregistered enum")
    return {"$enum": type(value).__name__, "value": value.value}
  if is_dataclass(value) and not isinstance(value, type):
    if type(value).__name__ not in _TYPES:
      raise ValueError("unregistered executable object")
    result = {"$type": type(value).__name__}
    for f in fields(value):
      item = getattr(value, f.name)
      if f.name == "program_hash":
        if type(item) is not int or not 0 < item < (1 << 256):
          raise ValueError("program hash must be a nonzero SHA-256 integer")
        result[f.name] = f"{item:064x}"
      else:
        result[f.name] = _encode(item)
    return result
  if isinstance(value, Mapping):
    # Pairs preserve integer role/mode keys, unlike JSON object coercion.
    pairs = [[_encode(k), _encode(v)] for k, v in value.items()]
    pairs.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":")))
    return {"$map": pairs}
  if isinstance(value, (tuple, list)):
    return [_encode(item) for item in value]
  if value is None or type(value) in (str, int, bool):
    return value
  if type(value) is float and math.isfinite(value):
    return value
  raise ValueError(f"unsupported executable value: {type(value).__name__}")


def _decode(value):
  if isinstance(value, list):
    return tuple(_decode(item) for item in value)
  if not isinstance(value, dict):
    if value is None or type(value) in (str, bool, int):
      return value
    if type(value) is float and math.isfinite(value):
      return value
    raise ValueError("invalid executable scalar")
  if "$map" in value:
    if set(value) != {"$map"} or not isinstance(value["$map"], list):
      raise ValueError("invalid mapping encoding")
    result = {}
    for pair in value["$map"]:
      if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError("invalid mapping pair")
      key, item = (_decode(part) for part in pair)
      if type(key) not in (str, int) or key in result:
        raise ValueError("invalid or duplicate mapping key")
      result[key] = item
    return FrozenMap(result)
  if "$enum" in value:
    if set(value) != {"$enum", "value"} or value["$enum"] not in _ENUMS:
      raise ValueError("unknown executable enum")
    return _ENUMS[value["$enum"]](value["value"])
  name = value.get("$type")
  if not isinstance(name, str) or name not in _TYPES:
    raise ValueError("unknown executable type")
  cls = _TYPES[name]
  expected = {f.name for f in fields(cls)}
  if set(value) != expected | {"$type"}:
    raise ValueError(f"invalid fields for {name}: expected {sorted(expected)}")
  kwargs = {}
  for key in expected:
    if key == "program_hash":
      text = value[key]
      if not isinstance(text, str) or not _HEX.fullmatch(text) or int(text, 16) == 0:
        raise ValueError("invalid program identity hash")
      kwargs[key] = int(text, 16)
    else:
      kwargs[key] = _decode(value[key])
  try:
    result = cls(**kwargs)
    _validate_executable_value(result, name)
    return result
  except (TypeError, AttributeError, KeyError) as exc:
    raise ValueError(f"invalid {name}") from exc


def program_digest(program: execution.ExecTileProgram) -> int:
  """Hash executable content, not its assigned ID or diagnostic source path."""
  content = {
    f.name: _encode(getattr(program, f.name))
    for f in fields(program)
    if f.name not in ("program_id", "program_hash", "source_ref")
  }
  for instruction in content["insts"]:
    instruction.pop("source_ref", None)
    instruction.pop("instruction_id", None)
  raw = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
  return int.from_bytes(hashlib.sha256(raw.encode()).digest(), "big")


def artifact_digest(program: CompiledProgram) -> str:
  if type(program) is not CompiledProgram:
    raise ValueError("expected CompiledProgram")
  _validate_compiled_program_value(program, allow_unsealed=True)
  value = _encode(program)
  del value["artifact_hash"]
  return hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  ).hexdigest()


def seal_program(program: CompiledProgram) -> CompiledProgram:
  if type(program) is not CompiledProgram:
    raise ValueError("expected CompiledProgram")
  _validate_compiled_program_value(program, allow_unsealed=True)
  return replace(program, artifact_hash=artifact_digest(program))


def serialize_compiled_program(program: CompiledProgram) -> str:
  if type(program) is not CompiledProgram:
    raise ValueError("expected CompiledProgram")
  _validate_compiled_program_value(program, allow_unsealed=False)
  if artifact_digest(program) != program.artifact_hash:
    raise ValueError("compiled artifact hash mismatch")
  return json.dumps(_encode(program), sort_keys=True, indent=2, allow_nan=False) + "\n"


def _unique_object(pairs):
  result = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate JSON key {key!r}")
    result[key] = value
  return result


def _reject_constant(value: object) -> NoReturn:
  raise ValueError(f"non-finite JSON number {value}")


def parse_compiled_program(text: str) -> CompiledProgram:
  if type(text) is not str:
    raise ValueError("compiled artifact must be JSON text")
  try:
    program = _decode(json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant))
  except (TypeError, KeyError, OverflowError, RecursionError) as exc:
    raise ValueError("invalid compiled artifact") from exc
  if type(program) is not CompiledProgram:
    raise ValueError("expected CompiledProgram")
  _validate_compiled_program_value(program, allow_unsealed=False)
  if artifact_digest(program) != program.artifact_hash:
    raise ValueError("compiled artifact hash mismatch")
  return program


def dump_executable_ir(program: CompiledProgram) -> str:
  """A complete, inspectable dump: no hidden configuration substeps."""
  return serialize_compiled_program(program)
