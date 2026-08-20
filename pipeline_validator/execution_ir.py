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
  LAUNCH_DMA_LOAD = "dma_load"
  LAUNCH_DMA_STORE = "dma_store"
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


@dataclass
class ExecTileProgram:
  name: str
  insts: list[ExecTileInst] = field(default_factory=list)
  descriptors: dict[str, ExecEngineDesc] = field(default_factory=dict)
  labels: dict[str, int] = field(default_factory=dict)
  program_id: int = 0
  version: int = 1
  program_hash: int = 0

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

@dataclass
class ExecTileGroupTask:
  name: str
  actions: list[ExecGroupAction] = field(default_factory=list)
  streams: list[ExecStreamDesc] = field(default_factory=list)
  role_bindings: dict[int, ExecTileRoleBinding] = field(default_factory=dict)
  completion_event: str = "group_task_done"


@dataclass
class ExecDeviceOp:
  """One device-level instruction in a model execution body."""

  op: str  # "submit" | "await" | "return"
  ctx_name: str = ""
  event_tag: str = ""


@dataclass
class ExecModel:
  """Lowered model: name, per-context tasks, context pin map, body ops."""

  name: str
  tasks: dict[str, ExecTileGroupTask] = field(default_factory=dict)
  context_pins: dict[str, int | None] = field(default_factory=dict)
  body: list[ExecDeviceOp] = field(default_factory=list)
