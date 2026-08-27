"""Private execution-layer DTOs for the simulator hot path.

The public workload IR is xDSL/MLIR.  Before simulation, workload IR lowers
exactly once into these plain Python execution objects so the existing
cycle-accurate controllers keep their current field accesses and dispatch
logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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

  space: str  # "task" | "l2"
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


@dataclass(frozen=True)
class ExecReleaseRequest:
  """Structured RELEASE_L2 request with verified consumer ordinals."""

  buffer_slot: str
  buffer_role: str
  consumer_dispatch_ordinals: tuple[int, ...]
  dependency_events: tuple[str, ...]


# ---------------------------------------------------------------------------
# Container DTOs (mutable — sequencer rewrites args in place)
# ---------------------------------------------------------------------------


@dataclass
class ExecTileInst:
  op: ExecTileOp
  dst: str | None = None
  args: tuple = ()
  label: str | None = None
  comment: str = ""


@dataclass
class ExecGroupAction:
  op: ExecGroupActionOp
  args: tuple = ()
  dst: str | None = None
  comment: str = ""


@dataclass
class ExecStreamDesc:
  queue_id: int
  depth: int
  producer_mask: int
  consumer_mask: int
  payload_slot_id: int = 0
  token_stride: int = 32
  pmu_stream_id: int = 0


@dataclass
class ExecEngineDesc:
  name: str
  kind: str
  op: str
  params: dict = field(default_factory=dict)
  transfer: ExecTransfer | None = None


@dataclass
class ExecTileProgram:
  name: str
  insts: list[ExecTileInst] = field(default_factory=list)
  descriptors: dict[str, ExecEngineDesc] = field(default_factory=dict)
  labels: dict[str, int] = field(default_factory=dict)
  program_id: int = 0
  version: int = 1
  program_hash: int = 0
  formals: tuple[ExecTileFormal, ...] = ()
  l1_buffers: tuple[ExecL1Buffer, ...] = ()

  def label_index(self, label: str) -> int:
    return self.labels[label]


@dataclass
class ExecTileRoleBinding:
  role_id: int
  tile_mask: int
  tile_program: ExecTileProgram
  in_stream: int | None = None
  out_stream: int | None = None
  context_id: int | None = None
  task_domain: ExecTaskDomain | None = None
  actuals: tuple[str, ...] = ()

@dataclass
class ExecTileGroupTask:
  name: str
  actions: list[ExecGroupAction] = field(default_factory=list)
  streams: list[ExecStreamDesc] = field(default_factory=list)
  role_bindings: dict[int, ExecTileRoleBinding] = field(default_factory=dict)
  completion_event: str = "group_task_done"
  global_inputs: tuple[ExecGlobalInput, ...] = ()
  l2_buffers: tuple[ExecL2Buffer, ...] = ()


@dataclass
class ExecDeviceOp:
  """One device-level instruction in a model execution body."""

  op: str  # "submit" | "await" | "return"
  ctx_name: str = ""
  event_tag: str = ""
  actual_inputs: tuple[int, ...] = ()


@dataclass
class ExecModel:
  """Lowered model: name, per-context tasks, context pin map, body ops."""

  name: str
  tasks: dict[str, ExecTileGroupTask] = field(default_factory=dict)
  context_pins: dict[str, int | None] = field(default_factory=dict)
  body: list[ExecDeviceOp] = field(default_factory=list)
  inputs: tuple[ExecGlobalInput, ...] = ()
