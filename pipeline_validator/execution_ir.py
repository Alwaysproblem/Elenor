"""Immutable executable DTOs shared by the compiler, Loader and runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .immutable import FrozenMap, FrozenRecord
from .profiles import (
  ArenaLayout,
  ContextResources,
  MemoryMaintenanceDesc,
  ProfileReconfigDesc,
  SourceRef,
  TileResources,
)


class ExecTileOp(Enum):
  NOP = "nop"
  MOV = "mov"
  ADD = "add"
  CMP = "cmp"
  BR = "br"
  BRP = "brp"
  BR_EOS = "br_eos"
  RET = "ret"
  LAUNCH_BOA = "launch.boa"
  LAUNCH_EVU = "launch.evu"
  LAUNCH_MFE = "launch.mfe"
  LAUNCH_USE = "launch.use"
  LAUNCH_GATHER = "launch.gather"
  WAIT = "wait"
  WAITALL = "waitall"
  FENCE = "fence"
  STREAM_POP = "stream.pop"
  STREAM_PUSH = "stream.push"
  STREAM_ACQUIRE = "stream.acquire"
  STREAM_RELEASE = "stream.release"
  STREAM_PUSH_EOS = "stream.eos"
  PATCH_DESC = "patch.desc"
  LOAD_DESC = "load.desc"
  STORE_DESC = "store.desc"
  PROF_BEGIN = "prof.begin"
  PROF_END = "prof.end"
  TRAP = "trap"
  SIGNAL_PHASE = "signal.phase"
  FREE_L1 = "free.l1"
  ALLOC_L1 = "alloc.l1"


class ExecGatherOutcome(Enum):
  L1_HIT = "L1_HIT"
  L2_HIT = "L2_HIT"
  HBM_MISS = "HBM_MISS"


class ExecGroupActionOp(Enum):
  INIT_STREAM = "init.stream"
  DMA_PREFETCH = "dma.prefetch"
  DMA_STORE = "dma.store"
  DISPATCH_ROLE = "dispatch.role"
  WAIT_EVENT = "wait.event"
  BARRIER_GROUP = "barrier.group"
  COLLECTIVE_RUN = "collective.run"
  SIGNAL_EVENT = "signal.event"
  RELEASE_L2 = "release.l2"
  BIND_L2_VIEW = "bind.l2.view"
  PUBLISH_L2 = "publish.l2"
  BIND_L2_IMPORT = "bind.l2.import"
  PROFILE_RECONFIG = "profile.reconfig"
  MEMORY_MAINTENANCE = "memory.maintenance"


class ContextAdmissionStatus(Enum):
  """Cross-layer context admission lifecycle (PR 3.5).

  Shared by ``Simulator`` and ``TileGroup``: a submitted context is
  PREPARED, becomes ACTIVE once its L2 bundle commits, or waits in
  WAIT_CAPACITY until a release-driven capacity change admits it.
  CANCELLED marks a reset/fault cleanup of a never-activated context.
  """

  PREPARED = "prepared"
  WAIT_CAPACITY = "wait_capacity"
  ACTIVE = "active"
  CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Frozen value objects (logical address IR, PR 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalBinding:
  """User-side launch binding (API + CLI)."""

  name: str
  base_iova: int
  size_bytes: int
  permissions: str  # "r" | "w" | "rw"


@dataclass(frozen=True)
class ExecGlobalInput:
  """Program/context global signature item."""

  name: str
  dims: tuple[int, ...]
  dtype: str
  size_bytes: int


@dataclass(frozen=True)
class ExecMemoryView:
  """Logical view; physical address materialized in PR 2.

  Field order is fixed: ``space, base, backing_dims, dims, offsets,
  strides, dtype, element_bytes, bytes, task_dim``.  ``backing_dims`` is
  the full shape of the underlying allocation or context global formal;
  ``dims`` is the view extents.  ``strides`` preserves the source IR
  unit-stride metadata (V1 only emits unit strides).  ``element_bytes``
  is the dtype byte width materialized once at lowering; runtime uses it
  to compute byte offsets without importing the xDSL dialect.
  """

  space: str  # "global" | "l2" | "l1"
  base: str  # "global:<name>" | <l2 slot> | "formal:<i>" | "l1:<k>"
  backing_dims: tuple[int, ...]  # full shape of the backing allocation/formal
  dims: tuple[int, ...]  # view extents
  offsets: tuple[int, ...]  # element offsets
  strides: tuple[int, ...]  # source IR strides (V1: all unit)
  dtype: str
  element_bytes: int  # dtype byte width, materialized at lowering
  bytes: int
  task_dim: int | None = None


@dataclass(frozen=True)
class ExecProfiledAccess:
  request_id: str
  outcome: ExecGatherOutcome
  bytes: int
  line_token: str | None
  merge_group: str | None


@dataclass(frozen=True)
class ExecGatherDesc:
  source: ExecMemoryView
  indices: ExecMemoryView
  destination: ExecMemoryView
  result_bytes: int
  cache_target_bytes: int
  l1_mshr_hint: int
  accesses: tuple[ExecProfiledAccess, ...]


@dataclass(frozen=True)
class ExecTransfer:
  """One DMA or MFE transfer with explicit src/dst views."""

  src: ExecMemoryView
  dst: ExecMemoryView
  bytes: int


@dataclass(frozen=True)
class ExecL2Buffer:
  """Context-owned L2 buffer descriptor."""

  slot: str
  dims: tuple[int, ...]
  dtype: str
  role: str  # "in" | "out" | "inout"
  element_bytes: int  # dtype byte width, materialized at lowering
  alignment: int  # 1 when source op omits alignment
  bytes: int
  sharing: str = "private"


@dataclass(frozen=True)
class ExecSharedInput:
  """One consumer binding to an exported producer L2 buffer."""

  slot: str
  dims: tuple[int, ...]
  dtype: str
  element_bytes: int
  bytes: int
  producer_binding_id: str
  producer_slot: str


@dataclass(frozen=True)
class ExecL1Buffer:
  """Tile-local L1 buffer descriptor."""

  name: str
  dims: tuple[int, ...]
  dtype: str
  element_bytes: int  # dtype byte width, materialized at lowering
  alignment: int  # 1 when source op omits alignment
  bytes: int


@dataclass(frozen=True)
class ExecTileFormal:
  """Tile program formal parameter descriptor."""

  space: str  # "task" | "global" | "l2"
  dims: tuple[int, ...]  # task is ()
  dtype: str  # task is ""


@dataclass(frozen=True)
class ExecTaskDomain:
  """Logical task range."""

  from_task: int
  to_task: int


# ---------------------------------------------------------------------------
# Frozen value objects (signal aggregation & gated release, PR 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridInstanceId:
  """Identity of one dispatched grid (one nest.dispatch.tasks.async).

  Aggregation key for phase signals: device slot, UCE hardware
  context, physical tile and logical task stay independent fields.
  """

  context_name: str
  device_slot: int
  launch_generation: int
  dispatch_ordinal: int


@dataclass(frozen=True)
class TaskIdentity:
  """Identity of one logical task instance inside one grid."""

  grid: GridInstanceId
  task_id: int


@dataclass(frozen=True)
class PhaseSignal:
  """One ``tile.signal`` emission bound to its logical task."""

  task: TaskIdentity
  phase: str


@dataclass(frozen=True)
class ExecSignalPolicy:
  """Aggregation policy of one dispatch, per phase (``all_tasks``)."""

  input_released: str | None
  output_ready: str | None


@dataclass(frozen=True)
class ExecDispatchRequest:
  """Structured DISPATCH_ROLE request; ``dst`` carries grid_done."""

  role_id: int
  dispatch_ordinal: int
  signal_policy: ExecSignalPolicy
  input_released_event: str
  output_ready_event: str
  requested_l1_mode: int = 0
  resolved_l1_mode: int = 0
  binding_id: str = ""


@dataclass(frozen=True)
class ExecReleaseRequest:
  """Structured RELEASE_L2 request with verified reader/writer ordinals."""

  buffer_slot: str
  buffer_role: str
  reader_dispatch_ordinals: tuple[int, ...]
  writer_dispatch_ordinals: tuple[int, ...]
  dependency_events: tuple[str, ...]


@dataclass(frozen=True)
class ExecPublishRequest:
  """Structured PUBLISH_L2 request with verified reader/writer ordinals."""

  buffer_slot: str
  reader_dispatch_ordinals: tuple[int, ...]
  writer_dispatch_ordinals: tuple[int, ...]
  dependency_events: tuple[str, ...]


# ---------------------------------------------------------------------------
# Immutable program templates; launch relocation creates distinct instances.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecTileInst(FrozenRecord):
  op: ExecTileOp
  dst: str | None = None
  args: tuple = ()
  label: str | None = None
  comment: str = ""
  source_ref: SourceRef | None = None
  instruction_id: str = ""


@dataclass(frozen=True)
class ExecGroupAction(FrozenRecord):
  op: ExecGroupActionOp
  args: tuple = ()
  dst: str | None = None
  comment: str = ""
  dependencies: tuple[str, ...] = ()
  reads: tuple[str, ...] = ()
  writes: tuple[str, ...] = ()
  source_ref: SourceRef | None = None
  instruction_id: str = ""

  @property
  def output_events(self) -> tuple[str, ...]:
    events = (self.dst,) if self.dst else ()
    if self.op is ExecGroupActionOp.DISPATCH_ROLE:
      request = self.args[0]
      events += tuple(
        event for event in (request.input_released_event, request.output_ready_event) if event
      )
    elif self.op is ExecGroupActionOp.SIGNAL_EVENT:
      events += (self.args[0],)
    return events


@dataclass(frozen=True)
class ExecStreamDesc(FrozenRecord):
  queue_id: int
  depth: int
  producer_mask: int
  consumer_mask: int
  payload_slot_id: int = 0
  token_stride: int = 32
  pmu_stream_id: int = 0


@dataclass(frozen=True)
class ExecEngineDesc(FrozenRecord):
  name: str
  kind: str
  op: str
  params: Mapping[str, Any] = field(default_factory=FrozenMap)
  transfer: ExecTransfer | None = None


@dataclass(frozen=True)
class ExecTileProgram(FrozenRecord):
  name: str
  insts: tuple[ExecTileInst, ...] = ()
  descriptors: Mapping[str, ExecEngineDesc] = field(default_factory=FrozenMap)
  labels: Mapping[str, int] = field(default_factory=FrozenMap)
  program_id: int = 0
  version: int = 1
  program_hash: int = 0
  formals: tuple[ExecTileFormal, ...] = ()
  l1_buffers: tuple[ExecL1Buffer, ...] = ()
  resource_contract: TileResources | None = None
  layout: ArenaLayout | None = None
  text_bytes: int = 0

  def label_index(self, label: str) -> int:
    return self.labels[label]


@dataclass(frozen=True)
class ExecTileRoleBinding(FrozenRecord):
  role_id: int
  tile_mask: int
  tile_program: ExecTileProgram
  in_stream: int | None = None
  out_stream: int | None = None
  context_id: int | None = None
  task_domain: ExecTaskDomain | None = None
  actuals: tuple[str, ...] = ()
  global_actuals: tuple[ExecMemoryView, ...] = ()
  read_actuals: tuple[str, ...] = ()
  write_actuals: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecTileGroupTask(FrozenRecord):
  name: str
  actions: tuple[ExecGroupAction, ...] = ()
  streams: tuple[ExecStreamDesc, ...] = ()
  role_bindings: Mapping[int, ExecTileRoleBinding] = field(default_factory=FrozenMap)
  completion_event: str = "group_task_done"
  global_inputs: tuple[ExecGlobalInput, ...] = ()
  l2_buffers: tuple[ExecL2Buffer, ...] = ()
  resource_contract: ContextResources | None = None
  layout: ArenaLayout | None = None
  binding_id: str = ""
  event_uses: Mapping[str, int] = field(default_factory=FrozenMap)
  shared_inputs: tuple[ExecSharedInput, ...] = ()


@dataclass(frozen=True)
class ExecDeviceOp(FrozenRecord):
  """One device-level instruction in a model execution body."""

  op: str  # "submit" | "await" | "return"
  ctx_name: str = ""
  event_tag: str = ""
  actual_inputs: tuple[int, ...] = ()
  dependencies: tuple[str, ...] = ()
  callsite_id: str = ""
  binding_id: str = ""
  command: ProfileReconfigDesc | MemoryMaintenanceDesc | None = None
  source_ref: SourceRef | None = None
  instruction_id: str = ""


@dataclass(frozen=True)
class ExecModel(FrozenRecord):
  """Lowered model: name, per-context tasks, context pin map, body ops."""

  name: str
  tasks: Mapping[str, ExecTileGroupTask] = field(default_factory=FrozenMap)
  context_pins: Mapping[str, int | None] = field(default_factory=FrozenMap)
  body: tuple[ExecDeviceOp, ...] = ()
  inputs: tuple[ExecGlobalInput, ...] = ()
