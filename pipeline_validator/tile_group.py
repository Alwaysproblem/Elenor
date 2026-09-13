"""Tile Group: 1 Tile Group Sequencer + 4 Compute Tiles + Group SRAM + streams.

The Tile Group is the local data-reuse / synchronization unit
(design/elenor_tile_group/).  It owns the Stream Queues that connect
task roles and the Group DMA.  The simulator drives it cycle by cycle,
advancing the Tile Group Sequencer and every Compute Tile in lockstep.
"""

from __future__ import annotations

import copy
import dataclasses
import zlib
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .config import GroupSchedulerConfig, HardwareConfig
from .execution_ir import (
  ContextAdmissionStatus,
  ExecDispatchRequest,
  ExecGroupActionOp,
  ExecReleaseRequest,
  ExecSignalPolicy,
  ExecStreamDesc,
  ExecTileGroupTask,
  ExecTileOp,
  ExecTileRoleBinding,
  GridInstanceId,
  PhaseSignal,
  TaskIdentity,
)
from .group_scheduler import GroupScheduler, IssueResult, IssueStatus
from .memory import (
  L2SRAM,
  AdmissionFailure,
  AdmissionFailureKind,
  AllocationHandle,
  AllocationRequest,
  BankSegment,
  ContextBufferOwner,
  DeterministicLRUCache,
  MemoryInvariantError,
  MemoryTransaction,
  MshrTable,
  NoCRouter,
  PayloadTracker,
  ResolvedMemoryView,
  TransferOp,
)
from .pmu import PMUCounter
from .runtime import (
  EventStatus,
  EventTable,
  FaultCode,
  FaultDomain,
  FaultRecord,
  FaultRing,
  ProgramResidencyManager,
  ResetDomain,
  ResetRequest,
)
from .runtime.reset_domain import ResetState
from .stream_queue import EOSPolicy, QueueKind, StreamQueue
from .tile import ComputeTile, TileAdmission
from .tile_group_sequencer import TileGroupSequencer
from .trace import MemoryTrace, Tracer


@dataclass
class _CollectiveJob:
  """A Collective Engine command in flight (reduce/broadcast/multicast)."""

  event_id: str
  start_cycle: int
  finish_cycle: int
  desc_id: str
  op: str
  bytes_total: int
  participant_mask: int
  sequencer: TileGroupSequencer | None = None  # sequencer that issued this job


@dataclass
class _RoleTrace:
  """Bookkeeping for one dispatched role's runtime window.

  Completion fan-in is keyed by the role's completion event id, not by
  role_id, so re-dispatching the same role_id (e.g. in a future loop)
  starts with fresh completion/trace state instead of aliasing a prior
  dispatch.
  """

  role_id: int
  event_id: str
  start_cycle: int
  tile_mask: int
  out_stream: int | None
  in_stream: int | None
  sequencer: TileGroupSequencer | None = None  # sequencer that dispatched this role


@dataclass
class _GridL2Pin:
  """One logical task's access pin on one L2 allocation."""

  task: TaskIdentity
  buffer_slot: str
  handle: AllocationHandle
  consumer_id: str
  reads: bool
  writes: bool


@dataclass
class _GridSignalState:
  """Aggregation state of one dispatched grid (PR 3).

  ``expected_task_ids`` comes from the role binding's task domain;
  a phase completes exactly once when ``seen[phase]`` equals it.
  """

  grid: GridInstanceId
  expected_task_ids: frozenset[int]
  policy: ExecSignalPolicy
  phase_event_ids: dict[str, str]
  seen: dict[str, set[int]] = field(default_factory=dict)
  completed_phases: set[str] = field(default_factory=set)
  sequencer: TileGroupSequencer | None = None

  def phase_declared(self, phase: str) -> bool:
    return phase in self.phase_event_ids


class L2AdmissionStatus(Enum):
  """Outcome status of one L2 admission attempt (PR 3.5)."""

  ADMITTED = "admitted"
  WAIT_CAPACITY = "wait_capacity"
  FAULT = "fault"


@dataclass(frozen=True)
class L2AdmissionOutcome:
  """Typed result of ``try_admit_l2_buffers``."""

  status: L2AdmissionStatus
  failure: AdmissionFailure | None = None


@dataclass
class _PendingContextAdmission:
  """One prepared context waiting for strict finite Group admission.

  A waiting ticket retains an adapter slot and namespaced task, but owns no
  event reservation, L2/L1 allocation, stream, UCE context, or engine work.
  """

  sequencer: TileGroupSequencer
  task: ExecTileGroupTask
  context_name: str
  device_slot: int
  launch_generation: int
  owned_queue_ids: frozenset[int]
  enqueue_cycle: int
  retry_count: int = 0
  wait_resource: str = ""


class TileGroup:
  """One ELENOR Tile Group with 4 Compute Tiles."""

  def __init__(
    self,
    cfg: HardwareConfig,
    tracer: Tracer | None = None,
    fidelity: str = "full_memory",
    context_count: int = 1,
    memory_trace: bool = False,
    scheduler_config: GroupSchedulerConfig | None = None,
  ):
    self.cfg = cfg
    self.tracer = tracer
    self.scheduler_config = scheduler_config or GroupSchedulerConfig()
    self.fidelity = fidelity
    rt = fidelity in ("runtime", "full_memory")
    mem = fidelity == "full_memory"
    self.runtime_enabled = rt
    self.memory_enabled = mem
    # PR 5: memory lanes/counters/flows + report peaks are opt-in so a
    # plain --trace-json run still emits the pre-PR5 control-flow trace.
    self.memory_trace = MemoryTrace(tracer) if tracer is not None and memory_trace else None
    # PR 2: transfer manager must exist before tiles are created (injected
    # into ComputeTile/MFEEngine as a shared instance).
    from .memory.hbm_region import HBMRegion
    from .memory.transfer import TransferManager

    # NoC fabric must exist before the transfer manager (NoC legs are
    # router-backed: flit enqueue/traversal/credit via NoCRouter)
    if mem:
      self.noc = NoCRouter(
        vc_depth=cfg.noc_vc_depth,
        router_latency_cycles=cfg.noc_router_latency_cycles,
        trace=self.memory_trace,
      )
    self.transfer_manager = TransferManager(
      cfg, full_memory=mem, noc=self.noc if mem else None, trace=self.memory_trace
    )
    self.l2_cache = DeterministicLRUCache(cfg.l2_cache_capacity_bytes, cfg.cache_line_bytes)
    self.l2_mshr = MshrTable(cfg.l2_mshr_entries)
    self.tiles: list[ComputeTile] = [
      ComputeTile(
        i,
        cfg,
        self.tracer,
        runtime_enabled=rt,  # type: ignore[arg-type]
        memory_enabled=mem,
        context_count=context_count,
        transfer_manager=self.transfer_manager,
        l2_cache=self.l2_cache,
        l2_mshr=self.l2_mshr,
        memory_trace=self.memory_trace,
      )
      for i in range(cfg.num_tiles)
    ]
    self.sequencer = TileGroupSequencer(self)
    self._active_sequencers: list[TileGroupSequencer] = []
    self._next_launch_id: int = 0
    self.queues: dict[int, StreamQueue] = {}
    self._collective_jobs: list[_CollectiveJob] = []
    self.pmu = PMUCounter()
    self.event_table = EventTable(capacity=self.scheduler_config.event_capacity)
    self.scheduler = GroupScheduler(self, self.scheduler_config)
    self._registered_programs: dict[tuple[int, int, int], int] = {}
    # role dispatch fan-in by event id: event_id -> tile_mask / done tiles
    self._role_event_tile_mask: dict[str, int] = {}
    self._role_done_tiles: dict[str, set[int]] = {}
    self._role_trace: dict[str, _RoleTrace] = {}
    # PR 3: structured grid registries - signal aggregation + live launches
    self._grid_signals: dict[GridInstanceId, _GridSignalState] = {}
    self._live_launches: dict[tuple[str, int, int], TileGroupSequencer] = {}
    # PR 3: grid -> {task_id -> {buffer_slot -> pin}}
    self._grid_l2_pins: dict[GridInstanceId, dict[int, dict[str, _GridL2Pin]]] = {}
    self._task_done_traced: bool = False
    self._task_trace_name: str | None = None
    self._task_start_cycle: int | None = None
    self.hbm = HBMRegion(
      base_iova=0,
      size_bytes=cfg.hbm_capacity_bytes,
      bandwidth_gbs=cfg.hbm_bandwidth_gbs,
      outstanding_limit=cfg.hbm_outstanding_limit,
      trace=self.memory_trace,
    )
    # PR 2 admission state: launch generation, global/L2 handles, pins
    self._context_launch_generation: int = 0
    self._global_handles: dict[str, AllocationHandle] = {}  # binding name -> handle
    self._l2_handles: dict[tuple[int, str], AllocationHandle] = {}  # (gen, slot) -> handle
    # PR 3: launch generation -> {buffer_slot -> alloc role}
    self._l2_roles: dict[int, dict[str, str]] = {}
    # Protocol-valid L2 objects, distinct from allocated/reserved capacity.
    self._protocol_live_l2: set[tuple[int, str]] = set()
    self._l2_reserved_bytes = 0
    self._l2_live_bytes = 0
    self._l2_reserved_bytes_peak = 0
    self._l2_live_bytes_peak = 0
    # PR 3.5: L2 admission wait queue + release-driven FIFO retry
    self._pending_context_admissions: deque[_PendingContextAdmission] = deque()
    self._pending_activations: list[_PendingContextAdmission] = []
    self._l2_capacity_change_cycle: int | None = None
    self._last_retried_pool_version: int = -1
    self._last_retried_capacity_change_cycle: int = -1
    self._last_retried_event_version: int = self.event_table.version
    self._last_step_cycle: int = 0
    # role_event_id -> tile_id -> shared live L1 name/handle map.  Each
    # inner map is the same object held by that tile's UCE context.
    self._role_l1_handles: dict[str, dict[int, dict[str, AllocationHandle]]] = {}
    # transaction id -> sequencer
    self._txn_sequencer: dict[str, TileGroupSequencer] = {}
    # group transaction id -> (logical direction, visual concurrency slot)
    self._group_transfer_trace_slots: dict[str, tuple[str, int]] = {}
    self._group_transfer_trace_busy_slots: dict[str, set[int]] = {"input": set(), "output": set()}
    if self.runtime_enabled:
      self.fault_ring = FaultRing()
      self.program_table = ProgramResidencyManager(cfg)
      self.reset_domain = ResetDomain(cfg)
    if self.memory_enabled or self.runtime_enabled:
      self.l2_sram = L2SRAM(
        capacity_bytes=cfg.group_sram_bytes,
        banks=cfg.group_sram_banks,
        bank_bandwidth_gbs=cfg.l2_bank_bandwidth_gbs,
        trace=self.memory_trace,
      )
    if self.memory_enabled:
      self.payload = PayloadTracker()

  # ---- setup ----------------------------------------------------------

  def init_stream(self, desc: ExecStreamDesc) -> StreamQueue:
    # masks are tile-bit masks: a bit set means that tile participates
    producers = frozenset(i for i in range(self.cfg.num_tiles) if desc.producer_mask & (1 << i))
    consumers = frozenset(i for i in range(self.cfg.num_tiles) if desc.consumer_mask & (1 << i))
    q = StreamQueue(
      queue_id=desc.queue_id,
      depth=desc.depth,
      producers=producers,
      consumers=consumers,
      kind=QueueKind.MPSC if len(producers) > 1 else QueueKind.SPSC,
      eos_policy=EOSPolicy.ALL_PRODUCERS if len(producers) > 1 else EOSPolicy.SINGLE_PRODUCER,
    )
    q.init()
    self.queues[desc.queue_id] = q
    # bind to every tile that participates
    for t in self.tiles:
      if (desc.producer_mask | desc.consumer_mask) & (1 << t.tile_id):
        t.bind_stream(desc.queue_id, q)
    return q

  def _record_l2_occupancy(self, cycle: int) -> None:
    """Update exact L2 occupancy peaks and change-only trace counters."""

    if self._l2_reserved_bytes < 0 or self._l2_live_bytes < 0:
      raise RuntimeError("L2 occupancy accounting underflow")
    if self._l2_live_bytes > self._l2_reserved_bytes:
      raise RuntimeError("protocol-live L2 bytes exceed reserved bytes")
    self._l2_reserved_bytes_peak = max(self._l2_reserved_bytes_peak, self._l2_reserved_bytes)
    self._l2_live_bytes_peak = max(self._l2_live_bytes_peak, self._l2_live_bytes)
    if self.tracer is None:
      return
    self.tracer.counter_if_changed(
      "TileGroup", "group_l2_reserved_bytes", cycle, self._l2_reserved_bytes, "bytes", thread="Scheduler:L2"
    )
    self.tracer.counter_if_changed(
      "TileGroup", "group_l2_live_bytes", cycle, self._l2_live_bytes, "bytes", thread="Scheduler:L2"
    )

  @staticmethod
  def _group_transfer_trace_direction(op: TransferOp) -> str:
    if op is TransferOp.PREFETCH:
      return "input"
    if op is TransferOp.GLOBAL_STORE:
      return "output"
    raise ValueError(f"unsupported group transfer op {op.value}")

  def _reserve_group_transfer_trace_slot(self, transaction_id: str, op: TransferOp) -> None:
    if transaction_id in self._group_transfer_trace_slots:
      raise ValueError(f"duplicate group transfer trace slot {transaction_id}")
    direction = self._group_transfer_trace_direction(op)
    busy = self._group_transfer_trace_busy_slots[direction]
    slot = 0
    while slot in busy:
      slot += 1
    busy.add(slot)
    self._group_transfer_trace_slots[transaction_id] = (direction, slot)

  def _release_group_transfer_trace_slot(self, transaction_id: str) -> tuple[str, int] | None:
    binding = self._group_transfer_trace_slots.pop(transaction_id, None)
    if binding is None:
      return None
    direction, slot = binding
    self._group_transfer_trace_busy_slots[direction].discard(slot)
    return binding

  def _clear_group_transfer_trace_slots(self) -> None:
    self._group_transfer_trace_slots.clear()
    for busy in self._group_transfer_trace_busy_slots.values():
      busy.clear()

  @staticmethod
  def _group_transfer_trace_lane(direction: str, slot: int) -> tuple[str, str]:
    category = "HBM → L2 Input" if direction == "input" else "L2 → HBM Output"
    return f"{category} #{slot}", category

  def submit_group_transfer(
    self,
    op: str,
    event_id: str,
    cycle: int,
    desc_id: str,
    transfer,
    sequencer: TileGroupSequencer | None = None,
  ) -> bool:
    """Submit a group-level DMA transfer (prefetch/store) as a
    ``MemoryTransaction``.  Returns True on success, False on fault.

    For timing_only: collapsed latency, src/dst are None.
    For runtime/full_memory: resolve src/dst against current handles.
    """
    gen = sequencer.context_launch_generation if sequencer is not None else self._context_launch_generation
    # PR 3.5: formal→actual mapping is launch-scoped on the issuing
    # sequencer; pending B must never overwrite active A's mapping.
    formals = sequencer.formal_bindings if sequencer is not None else {}
    txn_id = f"{gen}:{event_id}"
    context_name = sequencer.context_name if sequencer is not None and sequencer.context_name else "ctx"
    buffer_id = desc_id.split(":", 1)[-1]
    bytes_total = transfer.bytes if transfer.bytes > 0 else 1024 * 1024
    if self.memory_enabled or self.runtime_enabled:
      # resolve src/dst views against admission handles
      src_view = self._resolve_view(transfer.src, "global", gen, formals)
      dst_view = self._resolve_view(transfer.dst, "l2", gen, formals)
      if op == "dma.store":
        src_view = self._resolve_view(transfer.src, "l2", gen, formals)
        dst_view = self._resolve_view(transfer.dst, "global", gen, formals)
      if src_view is None or dst_view is None:
        # missing handle → fault
        return False
      txn = MemoryTransaction(
        transaction_id=txn_id,
        op=(TransferOp.PREFETCH if op == "dma.prefetch" else TransferOp.GLOBAL_STORE),
        issuer=ContextBufferOwner(context_name, gen, buffer_id),
        src=src_view,
        dst=dst_view,
        bytes_total=bytes_total,
        completion_event=event_id,
      )
    else:
      # timing_only: collapsed
      txn = MemoryTransaction(
        transaction_id=txn_id,
        op=(TransferOp.PREFETCH if op == "dma.prefetch" else TransferOp.GLOBAL_STORE),
        issuer=ContextBufferOwner(context_name, gen, buffer_id),
        src=None,
        dst=None,
        bytes_total=bytes_total,
        completion_event=event_id,
      )
    self.transfer_manager.submit(txn, cycle, self.pmu)
    if self.tracer is not None:
      self._reserve_group_transfer_trace_slot(txn.transaction_id, txn.op)
    if sequencer is not None:
      self._txn_sequencer[txn_id] = sequencer
      sequencer.note_job_started()
    return True

  def _resolve_view(
    self, view, default_space: str, gen: int | None = None, formal_bindings: dict[str, str] | None = None
  ) -> ResolvedMemoryView | None:
    """Resolve an ``ExecMemoryView`` to a ``ResolvedMemoryView`` using
    the admission handles.  ``formal_bindings`` maps context formal
    names to actual binding names (launch-scoped, PR 3.5).
    Returns None if the handle is missing."""
    if view is None:
      return None
    space = view.space
    use_gen = gen if gen is not None else self._context_launch_generation
    if space == "global":
      # global view: resolve against HBM external binding
      name = view.base.removeprefix("global:")
      # map formal name to actual binding name (launch-scoped)
      name = (formal_bindings or {}).get(name, name)
      handle = self._global_handles.get(name)
      if handle is None:
        return None
      offset = self._view_offset_bytes(view, logical_task_id=0)
      segments = self.hbm.resolve(handle, offset, view.bytes)
      return ResolvedMemoryView(
        handle=handle,
        offset_bytes=offset,
        size_bytes=view.bytes,
        address=segments[0].address,
        segments=segments,
      )
    if space == "l2":
      # L2 view: resolve against sequencer L2 handles
      slot = view.base
      key = (use_gen, slot)
      handle = self._l2_handles.get(key)
      if handle is None:
        return None
      offset = self._view_offset_bytes(view, logical_task_id=0)
      if self.memory_enabled:
        segs = self.l2_sram.resolve_segments(handle, offset, view.bytes)
      else:
        segs = (BankSegment(0, handle.base_address + offset, view.bytes),)
      return ResolvedMemoryView(
        handle=handle, offset_bytes=offset, size_bytes=view.bytes, address=segs[0].address, segments=segs
      )
    # l1 views are resolved per-tile in dispatch admission
    return None

  @staticmethod
  def _view_offset_bytes(view, logical_task_id: int = 0) -> int:
    """Compute the byte offset of a view, applying task_dim if present."""
    offsets = list(view.offsets)
    if view.task_dim is not None and view.task_dim < len(offsets):
      offsets[view.task_dim] += logical_task_id
    element_offset = 0
    for i, off in enumerate(offsets):
      stride = 1
      for d in view.backing_dims[i + 1 :]:
        stride *= d
      element_offset += off * stride
    return element_offset * view.element_bytes

  def register_global_bindings(self, bindings, cycle: int = 0) -> None:
    """Register user-supplied ``GlobalBinding``s as HBM external handles."""
    for name, binding in bindings.items():
      if name not in self._global_handles:
        self.hbm.bind_external(binding, cycle)
        handle = self.hbm.get_handle(name)
        assert handle is not None
        self._global_handles[name] = handle

  def try_admit_l2_buffers(
    self, task: ExecTileGroupTask, *, context_name: str, launch_generation: int, cycle: int
  ) -> L2AdmissionOutcome:
    """L2 admission: plan + commit all ``l2_buffers`` as one bundle.

    Typed outcome (PR 3.5): ``WAIT_CAPACITY`` means the bundle is
    legally placeable but the current live free map cannot satisfy it
    (a future release may); ``FAULT`` covers invalid requests and
    bundles that can never fit.  A failed plan has zero side effects.
    ``launch_generation`` is the caller's ticket/sequencer generation —
    never read implicitly from shared state.
    """
    roles = {buffer.slot: buffer.role for buffer in task.l2_buffers}
    if not task.l2_buffers:
      self._l2_roles[launch_generation] = roles
      return L2AdmissionOutcome(L2AdmissionStatus.ADMITTED)
    if not (self.memory_enabled or self.runtime_enabled):
      self._l2_roles[launch_generation] = roles
      return L2AdmissionOutcome(L2AdmissionStatus.ADMITTED)
    requests = [
      AllocationRequest(
        memory_space="l2",
        buffer_id=buf.slot,
        owner=ContextBufferOwner(context_name, launch_generation, buf.slot),
        size_bytes=buf.bytes,
        alignment=max(buf.alignment, 1),
        role=buf.role,
      )
      for buf in task.l2_buffers
    ]
    plan = self.l2_sram.plan_bundle(requests)
    if isinstance(plan, AdmissionFailure):
      status = (
        L2AdmissionStatus.WAIT_CAPACITY
        if plan.kind is AdmissionFailureKind.TEMPORARY_CAPACITY
        else L2AdmissionStatus.FAULT
      )
      return L2AdmissionOutcome(status, plan)
    handles = self.l2_sram.commit(plan, cycle)
    for buf, handle in zip(task.l2_buffers, handles):
      self._l2_handles[(launch_generation, buf.slot)] = handle
    self._l2_reserved_bytes += sum(handle.size_bytes for handle in handles)
    self._record_l2_occupancy(cycle)
    self._l2_roles[launch_generation] = roles
    return L2AdmissionOutcome(L2AdmissionStatus.ADMITTED)

  @staticmethod
  def _l2_admission_fault_reason(
    outcome: L2AdmissionOutcome, fallback: str = "context admission fault"
  ) -> str:
    failure = outcome.failure
    if failure is None:
      return fallback
    if failure.kind is AdmissionFailureKind.PERMANENT_CAPACITY:
      return f"L2 capacity fault: {failure.reason}"
    return failure.reason

  def release_l2(self, request: ExecReleaseRequest, sequencer: TileGroupSequencer, cycle: int) -> bool:
    """Release one context-owned L2 buffer after a read-only preflight.

    Dependencies and declared reader/writer phases are checked in every
    fidelity.  Runtime/full-memory additionally verify and release the
    physical allocation; timing-only deliberately has no physical handles.
    No logical lifetime, pin, or allocator state changes until the complete
    preflight succeeds.
    """
    gen = sequencer.context_launch_generation
    slot = request.buffer_slot
    launch = self._live_launches.get((sequencer.context_name, sequencer.device_slot, gen))
    if launch is not sequencer:
      raise MemoryInvariantError(f"release references inactive or foreign launch generation {gen}")
    roles = self._l2_roles.get(gen)
    if roles is None or slot not in roles:
      raise MemoryInvariantError(
        f"release references unknown or released L2 buffer '{slot}' in launch generation {gen}"
      )
    declared_role = roles[slot]
    if declared_role != request.buffer_role:
      raise MemoryInvariantError(
        f"release role mismatch on slot '{slot}':"
        f" alloc '{declared_role}' != request '{request.buffer_role}'"
      )

    handle: AllocationHandle | None = None
    if self.memory_enabled or self.runtime_enabled:
      handle = self._l2_handles.get((gen, slot))
      if handle is None:
        raise MemoryInvariantError(
          f"release references missing physical L2 buffer '{slot}' in launch generation {gen}"
        )
      expected_owner = ContextBufferOwner(sequencer.context_name, gen, slot)
      self.l2_sram.assert_live(handle, expected_owner)

    for event in request.dependency_events:
      if event not in sequencer._events_done:
        raise MemoryInvariantError(f"release of slot '{slot}' has incomplete dependency '{event}'")

    reader_ordinals = request.reader_dispatch_ordinals
    writer_ordinals = request.writer_dispatch_ordinals
    if len(reader_ordinals) != len(set(reader_ordinals)):
      raise MemoryInvariantError(f"release of slot '{slot}' has duplicate reader ordinals")
    if len(writer_ordinals) != len(set(writer_ordinals)):
      raise MemoryInvariantError(f"release of slot '{slot}' has duplicate writer ordinals")

    for ordinal in reader_ordinals:
      state = self._grid_signals.get(sequencer.grid_id(ordinal))
      if state is None or state.sequencer is not sequencer:
        raise MemoryInvariantError(
          f"release of slot '{slot}' references unknown or future reader dispatch ordinal {ordinal}"
        )
      if "input_released" not in state.completed_phases:
        raise MemoryInvariantError(
          f"release of slot '{slot}' precedes input_released for reader dispatch ordinal {ordinal}"
        )

    writer_states: dict[int, _GridSignalState] = {}
    for ordinal in writer_ordinals:
      state = self._grid_signals.get(sequencer.grid_id(ordinal))
      if state is None or state.sequencer is not sequencer:
        raise MemoryInvariantError(
          f"release of slot '{slot}' references unknown or future writer dispatch ordinal {ordinal}"
        )
      if "output_ready" not in state.completed_phases:
        raise MemoryInvariantError(
          f"release of slot '{slot}' precedes output_ready for writer dispatch ordinal {ordinal}"
        )
      writer_states[ordinal] = state

    writer_pins: list[tuple[GridInstanceId, int, dict[str, _GridL2Pin], _GridL2Pin]] = []
    if handle is not None:
      for grid, task_pins in self._grid_l2_pins.items():
        for task_id, slot_pins in task_pins.items():
          for pin_slot, pin in slot_pins.items():
            if pin.handle != handle:
              continue
            if (
              grid.context_name != sequencer.context_name
              or grid.device_slot != sequencer.device_slot
              or grid.launch_generation != gen
              or pin.task.grid != grid
              or pin.task.task_id != task_id
              or pin_slot != slot
              or pin.buffer_slot != slot
            ):
              raise MemoryInvariantError(f"release of slot '{slot}' found inconsistent access pin")
            ordinal = grid.dispatch_ordinal
            if pin.reads and not pin.writes:
              raise MemoryInvariantError(
                f"release of slot '{slot}' precedes input_released for reader dispatch ordinal {ordinal}"
              )
            if not pin.writes:
              raise MemoryInvariantError(f"release of slot '{slot}' found an accessless pin")
            state = writer_states.get(ordinal)
            if state is None:
              raise MemoryInvariantError(
                f"release of slot '{slot}' has undeclared pinned writer dispatch ordinal {ordinal}"
              )
            if pin.reads and "input_released" not in state.completed_phases:
              raise MemoryInvariantError(
                f"release of readwrite slot '{slot}' precedes input_released for dispatch ordinal {ordinal}"
              )
            writer_pins.append((grid, task_id, slot_pins, pin))

      if self.transfer_manager.has_inflight_access(handle):
        raise MemoryInvariantError(f"release of slot '{slot}' has an in-flight transfer")

    for grid, task_id, slot_pins, pin in writer_pins:
      self.l2_sram.unpin(pin.handle, pin.consumer_id, cycle)
      slot_pins.pop(slot)
      task_pins = self._grid_l2_pins[grid]
      if not slot_pins:
        task_pins.pop(task_id)
      if not task_pins:
        self._grid_l2_pins.pop(grid)

    if handle is not None:
      expected_owner = ContextBufferOwner(sequencer.context_name, gen, slot)
      freed = self.l2_sram.request_release(handle, expected_owner, cycle)
      if not freed:
        raise MemoryInvariantError(f"release of slot '{slot}' left pinned consumers")
      # Only an allocator final-free makes new capacity available; signal
      # aggregates and unpins alone never wake the admission queue.
      self._l2_capacity_change_cycle = cycle
      self._l2_reserved_bytes -= handle.size_bytes
      if (gen, slot) in self._protocol_live_l2:
        self._l2_live_bytes -= handle.size_bytes
      self._record_l2_occupancy(cycle)

    roles.pop(slot)
    self._protocol_live_l2.discard((gen, slot))
    return True

  # ---- L2 admission wait queue (PR 3.5) -----------------------------

  def _enqueue_pending_admission(self, ticket: _PendingContextAdmission) -> None:
    """Enqueue a WAIT_CAPACITY ticket at the FIFO tail."""
    first = not self._pending_context_admissions
    self._pending_context_admissions.append(ticket)
    if first and (self.memory_enabled or self.runtime_enabled):
      # Baseline stamps: only pool versions / release notifications above
      # this enqueue point count as capacity changes worth a retry (no
      # submit-cycle busy-poll on pre-enqueue frees).
      self._last_retried_pool_version = self.l2_sram.pool_version
      self._last_retried_capacity_change_cycle = (
        self._l2_capacity_change_cycle if self._l2_capacity_change_cycle is not None else -1
      )
    self._last_retried_event_version = self.event_table.version
    self.pmu.add_event(f"{ticket.wait_resource or 'l2'}_admission_wait")
    peak = self.pmu.named_cycles["l2_admission_queue_peak"]
    if len(self._pending_context_admissions) > peak:
      self.pmu.named_cycles["l2_admission_queue_peak"] = len(self._pending_context_admissions)
    if self.tracer is not None:
      self.tracer.instant(
        "TileGroup",
        "Scheduler:L2",
        "context_admission_wait",
        ticket.enqueue_cycle,
        {
          "context": ticket.context_name,
          "slot": ticket.device_slot,
          "launch_generation": ticket.launch_generation,
          "cycle": ticket.enqueue_cycle,
          "wait_resource": ticket.wait_resource or "l2",
        },
      )

  def _activate_admitted_context(self, ticket: _PendingContextAdmission, cycle: int) -> None:
    """Activate an admitted context: live launch, active list, streams.

    Called at the post-sequencer barrier, so the new sequencer's first
    group action issues at ``cycle + 1``.
    """
    seq = ticket.sequencer
    if seq.admission_status is ContextAdmissionStatus.WAIT_CAPACITY:
      resource = ticket.wait_resource or "l2"
      self.pmu.add_event(f"{resource}_admission_wakeup")
      self.pmu.add_cycle(f"{resource}_admission_wait_cycles", cycle - ticket.enqueue_cycle)
    seq.admission_status = ContextAdmissionStatus.ACTIVE
    seq.admission_wait_start_cycle = None
    self._live_launches[(seq.context_name, seq.device_slot, seq.context_launch_generation)] = seq
    seq.owned_queue_ids = set(ticket.owned_queue_ids)
    self._active_sequencers.append(seq)
    for s in ticket.task.streams:
      self.init_stream(s)
    if self.tracer is not None:
      l2_live = (
        self.l2_sram.snapshot()["live_allocations"] if (self.memory_enabled or self.runtime_enabled) else 0
      )
      self.tracer.instant(
        "TileGroup",
        "Scheduler:L2",
        "context_admitted",
        cycle,
        {
          "context": seq.context_name,
          "slot": seq.device_slot,
          "launch_generation": seq.context_launch_generation,
          "cycle": cycle,
          "l2_live_allocations": l2_live,
        },
      )

  def _retry_pending_context_admissions(self, cycle: int) -> None:
    """Retry the strict FIFO head after L2 or event capacity is returned."""

    if not self._pending_context_admissions:
      return
    memory_changed = False
    if self.memory_enabled or self.runtime_enabled:
      capacity_changed = (
        self._l2_capacity_change_cycle is not None
        and self._l2_capacity_change_cycle > self._last_retried_capacity_change_cycle
      )
      memory_changed = capacity_changed and self.l2_sram.pool_version != self._last_retried_pool_version
    event_changed = self.event_table.version != self._last_retried_event_version
    wait_resource = self._pending_context_admissions[0].wait_resource or "l2"
    relevant_change = event_changed if wait_resource == "event" else memory_changed
    if not relevant_change:
      return
    if self.memory_enabled or self.runtime_enabled:
      self._last_retried_pool_version = self.l2_sram.pool_version
      if self._l2_capacity_change_cycle is not None:
        self._last_retried_capacity_change_cycle = self._l2_capacity_change_cycle
    self._last_retried_event_version = self.event_table.version
    staged: list[_PendingContextAdmission] = []
    while self._pending_context_admissions:
      ticket = self._pending_context_admissions[0]
      ticket.retry_count += 1
      ticket.sequencer.admission_retry_count = ticket.retry_count
      self.pmu.add_event(f"{ticket.wait_resource or 'l2'}_admission_retry")
      if self.tracer is not None:
        self.tracer.instant(
          "TileGroup",
          "Scheduler:L2",
          "context_admission_retry",
          cycle,
          {
            "context": ticket.context_name,
            "slot": ticket.device_slot,
            "launch_generation": ticket.launch_generation,
            "cycle": cycle,
            "retry_count": ticket.retry_count,
            "capacity_change_cycle": self._l2_capacity_change_cycle,
            "wait_resource": ticket.wait_resource or "l2",
          },
        )
      outcome = self._try_admit_prepared_context(ticket, cycle)
      if outcome.status is L2AdmissionStatus.WAIT_CAPACITY:
        ticket.wait_resource = (
          "event" if outcome.failure is not None and outcome.failure.buffer_id == "event_table" else "l2"
        )
        break
      self._pending_context_admissions.popleft()
      if outcome.status is L2AdmissionStatus.FAULT:
        ticket.sequencer.admission_status = ContextAdmissionStatus.CANCELLED
        ticket.sequencer.mark_fault(self._l2_admission_fault_reason(outcome))
        ticket.sequencer.done = True
        self._active_sequencers.append(ticket.sequencer)
        self.pmu.add_event("l2_admission_permanent_fault")
        break
      staged.append(ticket)
    self._pending_activations.extend(staged)
    if self.memory_enabled or self.runtime_enabled:
      self._last_retried_pool_version = self.l2_sram.pool_version
    self._last_retried_event_version = self.event_table.version

  def _cancel_pending_admissions(self, cycle: int | None = None, *, release_staged: bool = False) -> None:
    """Cancel every waiting or staged ticket (reset/fault cleanup).

    FIFO waiters own no allocation.  Staged activations have already
    committed their L2 bundle; callers that will not run the general
    context-memory unwind must set ``release_staged`` so those handles
    are explicitly released before the ticket is discarded.
    """
    waiters = list(self._pending_context_admissions)
    staged = list(self._pending_activations)
    for ticket in [*waiters, *staged]:
      ticket.sequencer.admission_status = ContextAdmissionStatus.CANCELLED
      terminal = cycle if cycle is not None else max(self._last_step_cycle, ticket.enqueue_cycle)
      resource = ticket.wait_resource or "l2"
      self.pmu.add_cycle(f"{resource}_admission_wait_cycles", terminal - ticket.enqueue_cycle)
      if self.tracer is not None:
        self.tracer.instant(
          "TileGroup",
          "Scheduler:L2",
          "context_admission_cancelled",
          terminal,
          {
            "context": ticket.context_name,
            "slot": ticket.device_slot,
            "launch_generation": ticket.launch_generation,
            "cycle": terminal,
            "wait_resource": resource,
          },
        )
      try:
        self.scheduler.cancel_context_events(ticket.sequencer)
      except RuntimeError:
        pass
    if release_staged and (self.memory_enabled or self.runtime_enabled):
      for ticket in staged:
        generation = ticket.launch_generation
        release_cycle = cycle if cycle is not None else max(self._last_step_cycle, ticket.enqueue_cycle)
        for key, handle in list(self._l2_handles.items()):
          if key[0] != generation:
            continue
          try:
            self.l2_sram.request_release(handle, handle.owner, release_cycle)
          except MemoryInvariantError:
            pass
          self._l2_handles.pop(key, None)
        self._l2_roles.pop(generation, None)
    self._pending_context_admissions.clear()
    self._pending_activations.clear()

  def _pin_grid_l2(
    self, grid: GridInstanceId, binding: ExecTileRoleBinding, task: TaskIdentity, cycle: int
  ) -> None:
    """Pin each accessed L2 allocation once for one logical task."""
    if not (self.memory_enabled or self.runtime_enabled):
      return
    reads = set(binding.read_actuals)
    writes = set(binding.write_actuals)
    access_slots = [slot for slot in dict.fromkeys(binding.actuals) if slot in reads or slot in writes]
    if not access_slots:
      return

    pinned: list[tuple[str, AllocationHandle, str]] = []
    try:
      for slot in access_slots:
        handle = self._l2_handles.get((grid.launch_generation, slot))
        if handle is None:
          raise MemoryInvariantError(f"missing or stale accessed L2 actual '{slot}'")
        consumer_id = (
          f"{grid.context_name}:s{grid.device_slot}"
          f":g{grid.launch_generation}:d{grid.dispatch_ordinal}"
          f":t{task.task_id}:{slot}"
        )
        self.l2_sram.pin(handle, consumer_id)
        pinned.append((slot, handle, consumer_id))
    except MemoryInvariantError:
      for _slot, handle, consumer_id in pinned:
        self.l2_sram.unpin(handle, consumer_id, cycle)
      raise

    task_pins = self._grid_l2_pins.setdefault(grid, {}).setdefault(task.task_id, {})
    for slot, handle, consumer_id in pinned:
      task_pins[slot] = _GridL2Pin(
        task=task,
        buffer_slot=slot,
        handle=handle,
        consumer_id=consumer_id,
        reads=slot in reads,
        writes=slot in writes,
      )

  def _unpin_grid_readers(self, grid: GridInstanceId, cycle: int) -> None:
    """Unpin pure readers after the grid's input_released aggregate."""
    pins = self._grid_l2_pins.get(grid)
    if not pins:
      return
    for task_id in list(pins):
      for slot in list(pins[task_id]):
        pin = pins[task_id][slot]
        if not pin.reads or pin.writes:
          continue
        self.l2_sram.unpin(pin.handle, pin.consumer_id, cycle)
        pins[task_id].pop(slot)
      if not pins[task_id]:
        pins.pop(task_id)
    if not pins:
      self._grid_l2_pins.pop(grid)

  def _unwind_grid_l2_pins(self, cycle: int, grids: list[GridInstanceId] | None = None) -> None:
    """Idempotently unpin every (or only the selected) grid's L2 pins."""
    if not (self.memory_enabled or self.runtime_enabled):
      if grids is None:
        self._grid_l2_pins.clear()
      else:
        for grid in grids:
          self._grid_l2_pins.pop(grid, None)
      return
    targets = (
      list(self._grid_l2_pins.items())
      if grids is None
      else [(grid, self._grid_l2_pins[grid]) for grid in grids if grid in self._grid_l2_pins]
    )
    for _grid, pins in targets:
      for task_id in list(pins):
        for slot in list(pins[task_id]):
          pin = pins[task_id].pop(slot)
          try:
            self.l2_sram.unpin(pin.handle, pin.consumer_id, cycle)
          except MemoryInvariantError:
            pass
        pins.pop(task_id, None)
    if grids is None:
      self._grid_l2_pins.clear()
    else:
      for grid in grids:
        self._grid_l2_pins.pop(grid, None)

  def release_context_memory(self, cycle: int) -> None:
    """Drain a faulted Group and retain only resources cleanup could not free."""

    # Waiting tickets own no allocation.  Staged tickets are covered by the
    # generation-keyed handle sweep below.
    self._cancel_pending_admissions(cycle)
    for sequencer in self._active_sequencers:
      self.scheduler.cancel_context(sequencer, cycle)
    self.transfer_manager.cancel_all(cycle)
    self._txn_sequencer.clear()
    self.scheduler.abort_inflight(cycle)
    for sequencer in self._active_sequencers:
      # MARK_EVENTS ran immediately before this reset phase.  Retiring each
      # owner returns finite event capacity while retaining RESET diagnostics.
      self.scheduler.retire_context_events(sequencer, cycle)
      sequencer.mark_fault("group reset")
      sequencer.abort_actions()
    self._clear_group_transfer_trace_slots()
    self.l2_mshr.reset()
    self.l2_cache.reset()

    if self.tracer is not None:
      for cjob in self._collective_jobs:
        self.tracer.complete(
          "TileGroup",
          "Collective",
          f"collective.{cjob.op}:{cjob.desc_id}",
          cjob.start_cycle,
          cycle,
          args={
            "event_id": cjob.event_id,
            "bytes": cjob.bytes_total,
            "participant_mask": cjob.participant_mask,
            "status": "reset",
          },
        )
      for event_id, trace in self._role_trace.items():
        expected = self._role_event_tile_mask.get(event_id, 0).bit_count()
        if len(self._role_done_tiles.get(event_id, set())) >= expected:
          continue
        self.tracer.complete(
          "TileGroup",
          f"TileRole:{trace.role_id}",
          f"dispatch:role{trace.role_id}:{event_id}:run",
          trace.start_cycle,
          cycle,
          args={
            "role_id": trace.role_id,
            "event_id": event_id,
            "tile_mask": trace.tile_mask,
            "out_stream": trace.out_stream,
            "in_stream": trace.in_stream,
            "status": "reset",
          },
        )
    self._collective_jobs.clear()

    self._unwind_grid_l2_pins(cycle)
    self._grid_signals.clear()
    self._live_launches.clear()
    remaining_l2: dict[tuple[int, str], AllocationHandle] = {}
    if self.memory_enabled or self.runtime_enabled:
      for key, handle in self._l2_handles.items():
        try:
          self.l2_sram.request_release(handle, handle.owner, cycle)
        except MemoryInvariantError:
          pass
        if not self.l2_sram.is_released(handle):
          remaining_l2[key] = handle
    self._l2_handles = remaining_l2

    for tile_l1 in self._role_l1_handles.values():
      for tile_id, handles in tile_l1.items():
        alloc = self.tiles[tile_id].l1_allocator
        for handle in tuple(handles.values()):
          try:
            alloc.request_release(handle, handle.owner, cycle)
          except MemoryInvariantError:
            pass
        handles.clear()
    self._role_l1_handles.clear()
    self._protocol_live_l2.clear()
    self._l2_roles.clear()
    self._role_trace.clear()
    self._role_event_tile_mask.clear()
    self._role_done_tiles.clear()

    self._l2_live_bytes = 0
    self._l2_reserved_bytes = (
      int(self.l2_sram.snapshot()["allocated_bytes"]) if self.memory_enabled or self.runtime_enabled else 0
    )
    self._record_l2_occupancy(cycle)

    # Cancelled transfer resources are already returned by reset cleanup;
    # now clear UCE contexts/queued engine work and invalidate L1 handles.
    for tile in self.tiles:
      tile.reset()

  def schedule_collective(
    self,
    desc_id: str,
    event_id: str,
    op: str,
    bytes_total: int,
    participant_mask: int,
    cycle: int,
    sequencer: TileGroupSequencer | None = None,
  ) -> None:
    # One-cycle runtime window: numeric reduce datapath/bandwidth is left to
    # SRAM profile/PPA exploration per the collective design spec.
    self._collective_jobs.append(
      _CollectiveJob(
        event_id=event_id,
        start_cycle=cycle,
        finish_cycle=cycle + 1,
        desc_id=desc_id,
        op=op,
        bytes_total=bytes_total,
        participant_mask=participant_mask,
        sequencer=sequencer,
      )
    )
    if sequencer is not None:
      sequencer.note_job_started()

  def can_dispatch_role(self, binding: ExecTileRoleBinding) -> bool:
    return all(
      t.can_accept_context(binding.context_id) for t in self.tiles if binding.tile_mask & (1 << t.tile_id)
    )

  def dispatch_role(
    self,
    binding: ExecTileRoleBinding,
    cycle: int,
    request: ExecDispatchRequest,
    event_id: str | None = None,
    sequencer: TileGroupSequencer | None = None,
  ) -> IssueResult:
    """Atomically plan, commit, pin, and bind one gang dispatch.

    Tile owns physical context selection and L1 plan/commit/abort.  Temporary
    context or L1 pressure returns backpressure with no mutation; malformed or
    permanently impossible requests fault.  Any late commit/pin/bind failure
    unwinds every Tile and every Group-owned pin before returning.
    """

    role_id = binding.role_id
    tile_mask = binding.tile_mask
    event = event_id or f"ev_role{role_id}"
    seq = sequencer or self.sequencer
    program = binding.tile_program
    generation = seq.context_launch_generation
    grid = seq.grid_id(request.dispatch_ordinal)
    from_task = binding.task_domain.from_task if binding.task_domain is not None else 0
    to_task = binding.task_domain.to_task if binding.task_domain is not None else 0
    expected_task_ids = frozenset(range(from_task, to_task))
    selected_tiles = [tile for tile in self.tiles if tile_mask & (1 << tile.tile_id)]
    if not expected_task_ids:
      reason = "zero-task dispatch is not supported"
      seq.mark_fault(reason)
      return IssueResult(IssueStatus.FAULT, reason=reason)
    if len(expected_task_ids) != len(selected_tiles):
      reason = "dispatch task domain must exactly match selected Tile count"
      seq.mark_fault(reason)
      return IssueResult(IssueStatus.FAULT, reason=reason)
    phase_event_ids: dict[str, str] = {}
    if request.input_released_event:
      phase_event_ids["input_released"] = request.input_released_event
    if request.output_ready_event:
      phase_event_ids["output_ready"] = request.output_ready_event
    admissions: list[TileAdmission] = []

    def fault(reason: str) -> IssueResult:
      seq.mark_fault(reason)
      seq.pmu.add_event("l1_admission_fault")
      return IssueResult(IssueStatus.FAULT, reason=reason)

    def rollback() -> None:
      self._unwind_grid_l2_pins(cycle, [grid])
      self._grid_signals.pop(grid, None)
      for admission in reversed(admissions):
        try:
          admission.tile.abort_admission(admission, cycle)
        except MemoryInvariantError:
          pass
      self._role_l1_handles.pop(event, None)
      self._role_event_tile_mask.pop(event, None)
      self._role_done_tiles.pop(event, None)
      self._role_trace.pop(event, None)

    l2_formals = [
      (formal_index, formal) for formal_index, formal in enumerate(program.formals) if formal.space == "l2"
    ]
    if len(binding.actuals) != len(l2_formals):
      return fault("L2 actual count must exactly match tile program L2 formals")
    if len(binding.read_actuals) != len(set(binding.read_actuals)):
      return fault("duplicate read actual in tile role binding")
    if len(binding.write_actuals) != len(set(binding.write_actuals)):
      return fault("duplicate write actual in tile role binding")
    actual_slots = set(binding.actuals)
    if any(slot not in actual_slots for slot in binding.read_actuals):
      return fault("read actual is not present in tile role binding actuals")
    if any(slot not in actual_slots for slot in binding.write_actuals):
      return fault("write actual is not present in tile role binding actuals")
    if binding.read_actuals and not request.signal_policy.input_released:
      return fault("L2 read actuals require an input_released policy")
    if binding.write_actuals and not request.signal_policy.output_ready:
      return fault("L2 write actuals require an output_ready policy")
    roles = self._l2_roles.get(generation, {})
    for slot in binding.actuals:
      if slot not in roles:
        return fault(f"unknown or released L2 actual '{slot}'")
    for slot in binding.write_actuals:
      if roles[slot] == "in":
        return fault(f"role=in L2 actual '{slot}' cannot be written")

    for ordinal, tile in enumerate(selected_tiles):
      logical_task_id = from_task + ordinal
      candidate = tile.plan_admission(program, grid, event, logical_task_id, binding.context_id)
      if candidate is None:
        return IssueResult(
          IssueStatus.BACKPRESSURE, reason=f"UCE context unavailable on tile {tile.tile_id}"
        )
      if isinstance(candidate, AdmissionFailure):
        if candidate.kind is AdmissionFailureKind.TEMPORARY_CAPACITY:
          return IssueResult(
            IssueStatus.BACKPRESSURE, reason=f"L1 capacity wait on tile {tile.tile_id}: {candidate.reason}"
          )
        return fault(f"L1 admission fault on tile {tile.tile_id}: {candidate.reason}")
      admissions.append(candidate)

    total_cold = 0
    if self.runtime_enabled and program.program_id != 0:
      identity = (program.program_id, program.version, program.program_hash)
      cached = self._registered_programs.get(identity)
      if cached is None:
        cached = self._program_bytes(program)
        self._registered_programs[identity] = cached
        self.program_table.register(
          program_id=program.program_id,
          version=program.version,
          program_hash=program.program_hash,
          hbm_iova=0,
          hbm_bytes=cached,
        )
      for admission in admissions:
        admission.prepare_cycles = self.program_table.ensure_resident(
          program.program_id, admission.tile.tile_id, cycle
        )
        total_cold += admission.prepare_cycles

    try:
      for admission in admissions:
        admission.tile.commit_admission(admission, program, cycle)
      for admission in admissions:
        self._pin_grid_l2(grid, binding, TaskIdentity(grid=grid, task_id=admission.logical_task_id), cycle)
      from .tile import _TileContextMemory

      for admission in admissions:
        task_identity = TaskIdentity(grid=grid, task_id=admission.logical_task_id)
        l2_handle_map: dict[int, AllocationHandle] = {}
        global_view_map: dict[int, ResolvedMemoryView] = {}
        if self.memory_enabled or self.runtime_enabled:
          for (formal_index, _formal), slot in zip(l2_formals, binding.actuals):
            handle = self._l2_handles.get((generation, slot))
            if handle is None:
              raise MemoryInvariantError("missing or stale L2 actual for tile formal")
            l2_handle_map[formal_index] = handle
          global_formals = [
            (formal_index, formal)
            for formal_index, formal in enumerate(program.formals)
            if formal.space == "global"
          ]
          if len(binding.global_actuals) != len(global_formals):
            raise MemoryInvariantError("missing global actual for tile formal")
          for (formal_index, _formal), actual in zip(global_formals, binding.global_actuals):
            resolved = self._resolve_view(actual, "global", generation, seq.formal_bindings)
            if resolved is None:
              raise MemoryInvariantError("missing or stale global actual for tile formal")
            global_view_map[formal_index] = resolved
        memory = _TileContextMemory(
          task_identity=task_identity,
          l2_formal_handles=l2_handle_map,
          global_formal_views=global_view_map,
          l1_handles=admission.l1_handles,
          l2_resolver=(self.l2_sram if self.memory_enabled or self.runtime_enabled else None),
        )
        context_id = admission.tile.load_program(
          program,
          role_id=role_id,
          role_event_id=event,
          prepare_cycles=admission.prepare_cycles,
          context_id=admission.context_id,
          memory=memory,
          task_identity=task_identity,
        )
        if context_id != admission.context_id:
          raise MemoryInvariantError(f"UCE context bind failed on tile {admission.tile.tile_id}")
        admission.bound = True
    except (MemoryInvariantError, ValueError) as exc:
      rollback()
      return fault(str(exc))

    self._role_event_tile_mask[event] = tile_mask
    self._role_done_tiles[event] = set()
    self._grid_signals[grid] = _GridSignalState(
      grid=grid,
      expected_task_ids=expected_task_ids,
      policy=request.signal_policy,
      phase_event_ids=phase_event_ids,
      sequencer=seq,
    )
    self._role_l1_handles[event] = {
      admission.tile.tile_id: admission.l1_handles for admission in admissions if admission.l1_handles
    }
    self._role_trace[event] = _RoleTrace(
      role_id=role_id,
      event_id=event,
      start_cycle=cycle,
      tile_mask=tile_mask,
      out_stream=binding.out_stream,
      in_stream=binding.in_stream,
      sequencer=seq,
    )
    context_ids = [admission.context_id for admission in admissions]
    for admission in admissions:
      tile = admission.tile
      tile.uce._phase_signal_callback = self._on_phase_signal
      for queue_id, queue in self.queues.items():
        tile.bind_stream(queue_id, queue)
      for done_event in sorted(seq._events_done):
        if "ev_dma_" in done_event:
          tile.uce.notify_event(done_event)

    if self.tracer is not None:
      context_arg = context_ids[0] if len(set(context_ids)) == 1 else context_ids
      self.tracer.instant(
        "TileGroup",
        f"TileRole:{role_id}",
        "tile_role_dispatch",
        cycle,
        {
          "role_id": role_id,
          "tile_mask": tile_mask,
          "program": program.name,
          "event_id": event,
          "out_stream": binding.out_stream,
          "in_stream": binding.in_stream,
          "ctx_id": context_arg,
          "pinned_context": binding.context_id,
          "context_count": self.tiles[0].uce.context_count,
        },
      )
    if total_cold > 0:
      self.pmu.add_cycle("program_cold_load", total_cold)
    return IssueResult(IssueStatus.ACCEPTED, asynchronous=True, completion_event=event, adapter="dispatch")

  def _on_phase_signal(self, signal: PhaseSignal, cycle: int) -> None:
    """Aggregate one task phase signal against its grid (PR 3).

    1. retired launch -> stale, ignored (+PMU ``tile_signal_stale``);
    2. live launch but unknown grid/task/phase -> protocol fault;
    3. live duplicate -> protocol fault without a second aggregate count;
    4. first legal signal completes the phase exactly once when every
       expected task has signalled, then unpins input-role pins.
    """
    grid = signal.task.grid
    tr = self.tracer

    def _signal_args() -> dict:
      return {
        "context_name": grid.context_name,
        "device_slot": grid.device_slot,
        "launch_generation": grid.launch_generation,
        "dispatch_ordinal": grid.dispatch_ordinal,
        "task_id": signal.task.task_id,
        "phase": signal.phase,
      }

    launch_key = (grid.context_name, grid.device_slot, grid.launch_generation)
    seq = self._live_launches.get(launch_key)
    if seq is None:
      self.pmu.add_event("tile_signal_stale")
      if tr is not None:
        tr.instant("TileGroup", "Scheduler:L2", "tile_signal_stale", cycle, _signal_args())
      return
    state = self._grid_signals.get(grid)
    if (
      state is None
      or signal.task.task_id not in state.expected_task_ids
      or not state.phase_declared(signal.phase)
    ):
      self.pmu.add_event("tile_signal_invalid")
      if tr is not None:
        tr.instant("TileGroup", "Scheduler:L2", "tile_signal_invalid", cycle, _signal_args())
      seq.mark_fault(f"invalid tile signal: grid {grid} task {signal.task.task_id} phase '{signal.phase}'")
      self.scheduler.cancel_context(seq, cycle)
      if self.runtime_enabled and not (self.reset_domain.is_active or self.reset_domain.is_done):
        self.trigger_fault(FaultCode.ADDRESS_FAULT, tile_id=-1, cycle=cycle, desc_id=seq.fault_reason)
      return
    seen = state.seen.setdefault(signal.phase, set())
    if signal.task.task_id in seen:
      self.pmu.add_event("tile_signal_duplicate")
      if tr is not None:
        tr.instant("TileGroup", "Scheduler:L2", "tile_signal_duplicate", cycle, _signal_args())
      seq.mark_fault(
        f"duplicate tile signal: grid {grid} task {signal.task.task_id} phase '{signal.phase}'"
      )
      self.scheduler.cancel_context(seq, cycle)
      if self.runtime_enabled and not (self.reset_domain.is_active or self.reset_domain.is_done):
        self.trigger_fault(FaultCode.ADDRESS_FAULT, tile_id=-1, cycle=cycle, desc_id=seq.fault_reason)
      return
    seen.add(signal.task.task_id)
    if seen != state.expected_task_ids:
      return
    phase_seq = state.sequencer or self.sequencer
    phase_ev = state.phase_event_ids[signal.phase]
    if signal.phase == "input_released":
      try:
        self._unpin_grid_readers(grid, cycle)
      except MemoryInvariantError as exc:
        self.pmu.add_event("release_invariant_fault")
        phase_seq.mark_fault(f"input release unpin invariant fault: {exc}")
        self.scheduler.cancel_context(phase_seq, cycle)
        if self.runtime_enabled and not (self.reset_domain.is_active or self.reset_domain.is_done):
          self.trigger_fault(
            FaultCode.ADDRESS_FAULT, tile_id=-1, cycle=cycle, desc_id=phase_seq.fault_reason
          )
        return
    state.completed_phases.add(signal.phase)
    if tr is not None:
      tr.instant(
        "TileGroup",
        "Scheduler:L2",
        "phase_aggregate",
        cycle,
        {
          **_signal_args(),
          "expected": len(state.expected_task_ids),
          "seen": len(seen),
          "event_id": phase_ev,
        },
      )
    phase_seq.notify_event(phase_ev, cycle, EventStatus.ERROR if phase_seq.faulted else EventStatus.DONE)

  def note_l2_protocol_event(self, sequencer: TileGroupSequencer, event_id: str, cycle: int) -> None:
    """Mark L2 objects protocol-valid and record physical live bytes."""

    if sequencer.task is None:
      return
    generation = sequencer.context_launch_generation
    new_live_bytes = 0
    for action in sequencer.task.actions:
      if event_id not in action.output_events:
        continue
      slots: tuple[str, ...] = ()
      if action.op is ExecGroupActionOp.DMA_PREFETCH and len(action.args) >= 2:
        transfer = action.args[1]
        if transfer.dst is not None and transfer.dst.space == "l2":
          slots = (transfer.dst.base,)
      elif action.op is ExecGroupActionOp.DISPATCH_ROLE:
        request = action.args[0]
        if isinstance(request, ExecDispatchRequest) and event_id == request.output_ready_event:
          binding = sequencer.task.role_bindings.get(request.role_id)
          if binding is not None:
            slots = binding.write_actuals
      for slot in slots:
        key = (generation, slot)
        if slot not in self._l2_roles.get(generation, {}) or key in self._protocol_live_l2:
          continue
        self._protocol_live_l2.add(key)
        handle = self._l2_handles.get(key)
        if handle is not None and not self.l2_sram.is_released(handle):
          new_live_bytes += handle.size_bytes
    if new_live_bytes:
      self._l2_live_bytes += new_live_bytes
      self._record_l2_occupancy(cycle)

  def context_cleanup_ready(self, sequencer: TileGroupSequencer) -> bool:
    """True once Tile frames and all Group-owned access pins are gone."""
    if sequencer.faulted and self.runtime_enabled and not self.reset_domain.is_done:
      return False

    for event_id, role_trace in self._role_trace.items():
      if role_trace.sequencer is sequencer and self._role_l1_handles.get(event_id):
        return False
    if self.memory_enabled or self.runtime_enabled:
      for (generation, _slot), handle in self._l2_handles.items():
        if generation != sequencer.context_launch_generation:
          continue
        if not self.l2_sram.is_released(handle):
          return False
    for grid, task_pins in self._grid_l2_pins.items():
      if grid.launch_generation != sequencer.context_launch_generation:
        continue
      if grid.context_name != sequencer.context_name:
        continue
      if any(slot_pins for slot_pins in task_pins.values()):
        return False
    return True

  @staticmethod
  def _program_bytes(prog) -> int:
    """Estimate *program text* size for residency (install to tile program SRAM).

    Counts instructions (8 B/inst) + descriptor *templates* (64 B/desc),
    NOT descriptor `params["bytes"]` which is tensor data size, not program
    text.  Minimum 1 KB so empty programs still pay a cold-install cost.
    """
    inst_bytes = len(prog.insts) * 8
    desc_template_bytes = len(prog.descriptors) * 64
    return max(inst_bytes + desc_template_bytes, 1024)

  # ---- per-cycle step -------------------------------------------------

  def step(self, cycle: int) -> bool:
    """Advance one cycle.  Returns True if the whole task is done."""
    self._last_step_cycle = cycle
    tr = self.tracer
    # 0. task trace: capture start cycle on first step
    if tr is not None and self._task_start_cycle is None:
      self._task_start_cycle = cycle

    # 1. advance the NoC fabric first, then the transfer manager: flits
    # enqueued last cycle traverse now, and the manager observes the
    # traversal in the same cycle it polls (PR 2 §4.4/§4.7).
    if self.memory_enabled:
      traversed = self.noc.step(cycle)
      self.transfer_manager.note_traversed(traversed, cycle)
    completed_txns = self.transfer_manager.step(cycle)
    for txn in completed_txns:
      # PR 2: skip tile-local transactions — MFE tick handles them
      if txn.tile_id is not None:
        continue
      trace_slot = self._release_group_transfer_trace_slot(txn.transaction_id)
      seq = self._txn_sequencer.pop(txn.transaction_id, None) or self.sequencer
      completion_status = EventStatus.ERROR if seq.faulted else EventStatus.DONE
      completion_accepted = seq.notify_event(txn.completion_event, cycle, completion_status)
      seq.note_job_done()
      # Only protocol-successful DMA may satisfy explicit Tile waits.
      if completion_accepted and completion_status is EventStatus.DONE:
        for t in self.tiles:
          if t.uce.has_active_contexts():
            t.uce.notify_event(txn.completion_event)
      # full_memory: record payload using real transaction addresses
      if self.memory_enabled and txn.src is not None and txn.dst is not None:
        from .memory.payload import Payload

        src_addr = txn.src.address
        if self.payload.get(src_addr) is None:
          self.payload.alloc(
            src_addr,
            Payload(
              iova=src_addr,
              bytes_total=txn.bytes_total,
              layout="paged_kv" if txn.op == TransferOp.PREFETCH else "row_major",
              producer_kind="DMA",
            ),
          )
        self.payload.copy(src_addr, txn.dst.address, txn.bytes_total)
      if tr is not None:
        if trace_slot is None:
          raise RuntimeError(f"group transfer {txn.transaction_id} lacks a visual slot")
        direction, visual_slot = trace_slot
        thread, category = self._group_transfer_trace_lane(direction, visual_slot)
        owner_args = MemoryTrace._owner_args(txn.issuer)
        context_name = str(owner_args.get("context_name", "ctx"))
        buffer_id = str(owner_args.get("buffer_id", txn.completion_event))
        duration_cycles = max(txn.completed_cycle - txn.start_cycle, 1)
        effective_bandwidth_gbs = round(txn.bytes_total / (duration_cycles * self.cfg.cycle_ns()), 3)
        tr.complete(
          "TileGroup",
          thread,
          f"{context_name} / {buffer_id}",
          txn.start_cycle,
          txn.completed_cycle,
          args={
            "summary_kind": "group_transfer",
            "direction": category,
            "visual_slot": visual_slot,
            "transaction_id": txn.transaction_id,
            "op": txn.op.value,
            **owner_args,
            "event_id": txn.completion_event,
            "bytes": txn.bytes_total,
            "duration_cycles": duration_cycles,
            "effective_bandwidth_gbs": effective_bandwidth_gbs,
            "start_cycle": txn.start_cycle,
            "completion_cycle": txn.completed_cycle,
            "source_address": txn.src.address if txn.src is not None else None,
            "destination_address": txn.dst.address if txn.dst is not None else None,
            **({"flow_id": tr.flow_id(txn.transaction_id)} if self.memory_trace is not None else {}),
          },
          category=category,
        )
      self.transfer_manager.acknowledge(txn.transaction_id)

    # 1b. tick Collective jobs
    remaining_coll: list[_CollectiveJob] = []
    for cjob in self._collective_jobs:
      if cycle >= cjob.finish_cycle:
        seq = cjob.sequencer or self.sequencer
        seq.notify_event(cjob.event_id, cycle, EventStatus.ERROR if seq.faulted else EventStatus.DONE)
        if cjob.sequencer is not None:
          cjob.sequencer.note_job_done()
        self.pmu.add_event("collective_complete")
        if tr is not None:
          tr.complete(
            "TileGroup",
            "Collective",
            f"collective.{cjob.op}:{cjob.desc_id}",
            cjob.start_cycle,
            cjob.finish_cycle,
            args={
              "event_id": cjob.event_id,
              "bytes": cjob.bytes_total,
              "participant_mask": cjob.participant_mask,
            },
          )
          tr.instant("TileGroup", "Collective", "collective_complete", cycle, {"event_id": cjob.event_id})
      else:
        remaining_coll.append(cjob)
    self._collective_jobs = remaining_coll
    # 2. tick stream queues (PMU occupancy counters + trace counters)
    for q in self.queues.values():
      q.tick(cycle)
      if tr is not None:
        tr.counter_if_changed(
          "TileGroup", "occupancy", cycle, q.occupancy, "tokens", thread=f"StreamQ:{q.queue_id}"
        )
        tr.counter_if_changed(
          "TileGroup",
          "credit_available",
          cycle,
          q._credit_available,
          "credits",
          thread=f"StreamQ:{q.queue_id}",
        )

    # (NoC router steps in section 1, before the transfer manager)
    freeze_new_work = self._reset_freezes_new_work()

    # 4. running engines still tick; UCE issue/queued launches freeze
    for t in self.tiles:
      t.step(cycle, freeze_new_work=freeze_new_work)
      for term in t.drain_context_terminals():
        if term.status == "fault":
          # Route fault to the sequencer that dispatched this role
          rid = term.role_event_id
          rt = self._role_trace.get(rid) if rid is not None else None
          fault_seq = rt.sequencer if rt is not None and rt.sequencer is not None else self.sequencer
          fault_seq.mark_fault(f"tile{term.tile_id}: {term.reason}")
          self.scheduler.cancel_context(fault_seq, cycle)
          if term.role_event_id is not None:
            entry = self.event_table.get(term.role_event_id)
            if entry is not None and entry.status is EventStatus.PENDING:
              fault_seq.notify_event(term.role_event_id, cycle, EventStatus.ERROR)
          self.pmu.add_event("tile_fault")
          if self.runtime_enabled and not (self.reset_domain.is_active or self.reset_domain.is_done):
            self.trigger_fault(
              self._fault_code_for_reason(term.reason),
              tile_id=term.tile_id,
              cycle=cycle,
              desc_id=term.reason,
            )
        if term.status != "done" or term.role_event_id is None:
          continue
        # PR 2: release L1 frame + allocations on tile terminal (§5.7).
        # PR 3: L2 grid pins outlive the terminal - only the matching
        # aggregate phase or a gated release may unpin them.
        tile_l1 = self._role_l1_handles.get(term.role_event_id, {})
        live_l1 = tile_l1.pop(t.tile_id, {})
        if self.memory_enabled or self.runtime_enabled:
          frame = t.l1_frames[term.ctx_id]
          frame_generation = frame.generation
          frame.release()
          if self.memory_trace is not None and tr is not None:
            tr.instant(
              f"Tile{t.tile_id}",
              "Lifecycle",
              "frame_release",
              cycle,
              {
                "ctx_id": term.ctx_id,
                "generation": frame_generation,
                "tile_id": t.tile_id,
                "reason": "tile_terminal",
              },
            )
          for handle in tuple(live_l1.values()):
            try:
              t.l1_allocator.request_release(handle, handle.owner, cycle)
            except MemoryInvariantError:
              pass  # already released or stale - terminal must not fault
        live_l1.clear()
        if not tile_l1:
          self._role_l1_handles.pop(term.role_event_id, None)
        done_set = self._role_done_tiles.setdefault(term.role_event_id, set())
        if t.tile_id in done_set:
          continue
        done_set.add(t.tile_id)
        if tr is not None:
          tr.instant(
            f"Tile{t.tile_id}",
            f"UCE CTX{term.ctx_id}",
            "tile_done",
            cycle,
            {"ctx_id": term.ctx_id, "role_id": term.role_id, "event_id": term.role_event_id},
          )
        rt = self._role_trace.get(term.role_event_id)
        mask = self._role_event_tile_mask.get(term.role_event_id, 0)
        expected = bin(mask).count("1")
        if len(done_set) >= expected:
          # Route completion to the sequencer that dispatched this role
          done_seq = rt.sequencer if rt is not None and rt.sequencer is not None else self.sequencer
          done_seq.notify_event(
            term.role_event_id, cycle, EventStatus.ERROR if done_seq.faulted else EventStatus.DONE
          )
          if tr is not None:
            if rt is not None:
              tr.complete(
                "TileGroup",
                f"TileRole:{rt.role_id}",
                f"dispatch:role{rt.role_id}:{term.role_event_id}:run",
                rt.start_cycle,
                cycle,
                args={
                  "role_id": rt.role_id,
                  "event_id": term.role_event_id,
                  "tile_mask": rt.tile_mask,
                  "out_stream": rt.out_stream,
                  "in_stream": rt.in_stream,
                },
              )
            tr.instant(
              "TileGroup",
              f"TileRole:{term.role_id}",
              "tile_role_complete",
              cycle,
              {"role_id": term.role_id, "event_id": term.role_event_id},
            )

    # 4. completions above are visible before one shared ISSUE and REGISTER.
    scheduler_frozen = freeze_new_work or (self.runtime_enabled and self.reset_domain.is_active)
    if not scheduler_frozen:
      self.scheduler.step(self._active_sequencers, cycle)
      # Admission retries stage new contexts after scheduling, so their first
      # action cannot register until the next cycle.
      self._retry_pending_context_admissions(cycle)
      for ticket in self._pending_activations:
        self._activate_admitted_context(ticket, cycle)
      self._pending_activations.clear()
    for active_seq in self._active_sequencers:
      active_seq.maybe_finish()

    # 5. aggregate PMU

    # 5b. advance reset/drain FSM if active (runtime fidelity)
    if self.runtime_enabled and self.reset_domain.is_active:
      self.reset_domain.step(cycle, group=self)
    self._aggregate_pmu()
    # Prune completed sequencers; reclaim their namespaced stream
    # queues, grid registries and retained handles so repeated submits
    # don't grow them unboundedly (PR 3 retirement).
    remaining: list[TileGroupSequencer] = []
    for s in self._active_sequencers:
      if not s.done:
        remaining.append(s)
        continue
      self._retire_sequencer(s, cycle)
      for qid in s.owned_queue_ids:
        self.queues.pop(qid, None)
        for t in self.tiles:
          t.unbind_stream(qid)
    self._active_sequencers = remaining
    # PR 3.5: pending admissions keep the group alive even with no
    # active sequencer; a waiting ticket is never "done".
    all_done = (
      len(self._active_sequencers) == 0
      and not self._pending_context_admissions
      and not self._pending_activations
    )
    if all_done and tr is not None:
      if not self._task_done_traced:
        start = self._task_start_cycle if self._task_start_cycle is not None else cycle
        if self._task_trace_name is not None:
          tr.complete(
            "TileGroup",
            "Task",
            self._task_trace_name,
            start,
            cycle,
            args={"task": self._task_trace_name.replace("task:", "", 1)},
          )
        cev = self.sequencer.task.completion_event if self.sequencer.task is not None else "group_task_done"
        tr.instant("TileGroup", "Task", "group_task_done", cycle, {"event": cev})
        self._task_done_traced = True
    return all_done

  def _retire_sequencer(self, s: TileGroupSequencer, cycle: int) -> None:
    """Retire one completed launch (PR 3).

    Drops the launch's grid signal state and live-launch registration,
    then removes only the generation's already-RELEASED retained
    handles.  Any unreleased handle or residual grid pin on a normally
    completed launch converts to an invariant fault instead of a
    silent prune.
    """
    gen = s.context_launch_generation
    grids = [g for g, st in self._grid_signals.items() if st.sequencer is s]
    pin_grids = [
      grid
      for grid in self._grid_l2_pins
      if grid.context_name == s.context_name
      and grid.device_slot == s.device_slot
      and grid.launch_generation == gen
    ]
    for grid in grids:
      self._grid_signals.pop(grid, None)
    self._live_launches.pop((s.context_name, s.device_slot, gen), None)
    if not (self.memory_enabled or self.runtime_enabled):
      self._grid_l2_pins = {g: pins for g, pins in self._grid_l2_pins.items() if g.launch_generation != gen}
      self._l2_roles.pop(gen, None)
      self._finish_sequencer_retirement(s, cycle)
      return
    retained = {key: handle for key, handle in self._l2_handles.items() if key[0] == gen}
    all_released = all(self.l2_sram.is_released(handle) for handle in retained.values())
    pin_residue = any(
      any(slot_pins for slot_pins in pins.values())
      for grid, pins in self._grid_l2_pins.items()
      if grid.context_name == s.context_name and grid.launch_generation == gen
    )
    if not all_released or pin_residue:
      if not s.faulted:
        s.mark_fault(
          f"launch retirement invariant: unreleased handles or grid pins remain for generation {gen}"
        )
        self.pmu.add_event("release_invariant_fault")
        if self.runtime_enabled and not (self.reset_domain.is_active or self.reset_domain.is_done):
          self.trigger_fault(FaultCode.ADDRESS_FAULT, cycle=cycle, desc_id=s.fault_reason)
      # Fault retirement is a rollback path; it may release only this launch.
      self._unwind_grid_l2_pins(cycle, pin_grids)
      for key, handle in retained.items():
        try:
          self.l2_sram.request_release(handle, handle.owner, cycle)
        except MemoryInvariantError:
          pass
        self._l2_handles.pop(key, None)
      self._l2_roles.pop(gen, None)
      self._finish_sequencer_retirement(s, cycle)
      return
    for key, handle in retained.items():
      if self.l2_sram.is_released(handle):
        self._l2_handles.pop(key, None)
    self._l2_roles.pop(gen, None)
    self._finish_sequencer_retirement(s, cycle)

  def _finish_sequencer_retirement(self, sequencer: TileGroupSequencer, cycle: int) -> None:
    generation = sequencer.context_launch_generation
    role_events = [event_id for event_id, trace in self._role_trace.items() if trace.sequencer is sequencer]
    external_events = {event for event in sequencer._events_done if "ev_dma_" in event}
    if external_events:
      for tile in self.tiles:
        tile.retire_external_events(external_events)
    for event_id in role_events:
      self._role_trace.pop(event_id, None)
      self._role_event_tile_mask.pop(event_id, None)
      self._role_done_tiles.pop(event_id, None)
      self._role_l1_handles.pop(event_id, None)
    self._protocol_live_l2 = {key for key in self._protocol_live_l2 if key[0] != generation}
    self.scheduler.retire_context_events(sequencer, cycle)

  def _reset_freezes_new_work(self) -> bool:
    """True after STOP_QUEUE until reset cleanup reaches DONE.

    Running transfers/engines continue to drain, but device/group/tile
    controllers must not submit new work.
    """
    if not self.runtime_enabled:
      return False
    state = self.reset_domain.state
    return ResetState.STOP_QUEUE <= state < ResetState.DONE

  @staticmethod
  def _fault_code_for_reason(reason: str) -> FaultCode:
    """Map runtime memory/engine failures to the existing fault ABI."""
    lowered = reason.lower()
    if "l1" in lowered or "slot" in lowered or "frame" in lowered:
      return FaultCode.SLOT_PERMISSION_FAULT
    if "address" in lowered or "owner" in lowered or "generation" in lowered or "release" in lowered:
      return FaultCode.ADDRESS_FAULT
    if "l2 capacity" in lowered:
      return FaultCode.L2_CAPACITY_FAULT
    if "timeout" in lowered or "credit" in lowered:
      return FaultCode.DMA_TIMEOUT
    if "descriptor" in lowered or "transaction id" in lowered:
      return FaultCode.INVALID_DESCRIPTOR
    return FaultCode.ENGINE_INTERNAL_FAULT

  def _aggregate_pmu(self) -> None:
    # Merge the single Group controller plus active context/Tile/queue PMUs.
    self.pmu.merge(self.scheduler.pmu)
    self.scheduler.pmu.reset()
    for seq in self._active_sequencers:
      self.pmu.merge(seq.pmu)
      seq.pmu.reset()
    for t in self.tiles:
      self.pmu.merge(t.pmu)
      t.pmu.reset()
    for q in self.queues.values():
      self.pmu.merge(q.pmu)
      q.pmu.reset()
    # full_memory: aggregate transfer manager + payload PMU as deltas.
    tm = self.transfer_manager
    self.pmu.add_event("memory_transaction_issued", tm.pmu_issued_count)
    self.pmu.add_event("memory_transaction_completed", tm.pmu_completed_count)
    self.pmu.add_event("memory_transaction_cancelled", tm.pmu_cancelled_count)
    self.pmu.add_event("memory_transaction_faulted", tm.pmu_faulted_count)
    self.pmu.add_cycle("noc_credit_wait", tm.pmu_noc_credit_wait_cycles)
    self.pmu.add_cycle("dma_queue_wait", tm._global_dma.wait_cycles)
    self.pmu.add_cycle("l2_bank_wait", tm._l2_read.wait_cycles + tm._l2_write.wait_cycles)
    self.pmu.add_cycle("l2_cache_wait", tm._l2_cache_lookup.wait_cycles + tm._l2_cache_fill.wait_cycles)
    l1_wait = sum(s.wait_cycles for s in list(tm._l1_read.values()) + list(tm._l1_write.values()))
    self.pmu.add_cycle("l1_bank_wait", l1_wait)
    l1_cache_wait = sum(
      stage.wait_cycles for stage in (*tm._l1_cache_lookup.values(), *tm._l1_cache_fill.values())
    )
    self.pmu.add_cycle("l1_cache_wait", l1_cache_wait)
    self.pmu.add_cycle("hbm_outstanding_wait", tm._hbm_read.wait_cycles + tm._hbm_write.wait_cycles)
    self.pmu.add_cycle("hbm_outstanding_peak", tm.pmu_hbm_outstanding_peak)
    # reset component counters so next cycle records only the delta
    tm.pmu_issued_count = 0
    tm.pmu_completed_count = 0
    tm.pmu_cancelled_count = 0
    tm.pmu_faulted_count = 0
    tm.pmu_noc_credit_wait_cycles = 0
    tm.pmu_hbm_outstanding_peak = 0
    for stage in tm._all_stages():
      stage.wait_cycles = 0
    if self.memory_enabled:
      self.pmu.add_cycle("payload_layout_faults", self.payload.layout_fault_count)
      self.payload.layout_fault_count = 0

  # ---- lifecycle ------------------------------------------------------

  def load_task(
    self, task: ExecTileGroupTask, *, input_bindings=None, formal_bindings: dict[str, str] | None = None
  ) -> None:
    previous_cycle = self._last_step_cycle
    # reset everything
    for t in self.tiles:
      t.reset()
    self.sequencer.reset()
    self.scheduler.reset()
    self._active_sequencers = []
    self.queues.clear()
    self._collective_jobs.clear()
    self._role_event_tile_mask.clear()
    self._role_done_tiles.clear()
    self._role_trace.clear()
    # PR 3.5: staged activations already own L2; unwind them before
    # clearing handle/role registries for the fresh standalone run.
    self._cancel_pending_admissions(release_staged=True)
    # PR 3: clear structured grid registries
    self._grid_signals.clear()
    self._live_launches.clear()
    self._grid_l2_pins.clear()
    self._l2_roles.clear()
    self._protocol_live_l2.clear()
    # PR 2: bump launch generation, clear admission state
    self._context_launch_generation += 1
    self._global_handles.clear()
    self._l2_handles.clear()
    self._role_l1_handles.clear()
    self._txn_sequencer.clear()
    self._clear_group_transfer_trace_slots()
    self.transfer_manager.reset()
    self.l2_mshr.reset()
    self.l2_cache.reset()
    self._l2_capacity_change_cycle = None
    self._last_retried_pool_version = -1
    self._last_retried_capacity_change_cycle = -1
    self._last_retried_event_version = self.event_table.version
    self._last_step_cycle = 0
    # A new standalone run is a fresh HBM binding epoch: old handles must
    # become stale and same-name bindings must be registered again.
    self.hbm.reset()
    # Event instances are finite in every fidelity; runtime fault/reset state
    # is conditional, while program residency remains warm.
    self.event_table.clear()
    if self.runtime_enabled:
      self.fault_ring.reset()
      self.reset_domain.reset()
      self.l2_sram.reset()
    if self.memory_enabled:
      self.noc.reset()
      self.payload.reset()
    # End the prior run's physical occupancy curve before starting fresh
    # counters.  Program residency is intentionally not reset here.
    self._l2_reserved_bytes = 0
    self._l2_live_bytes = 0
    self._record_l2_occupancy(previous_cycle)
    self._l2_reserved_bytes_peak = 0
    self._l2_live_bytes_peak = 0
    self.pmu = PMUCounter()
    self._task_trace_name = f"task:{task.name}"
    self._task_start_cycle = None
    self._task_done_traced = False
    # PR 2: register global bindings as HBM external handles
    if input_bindings:
      self.register_global_bindings(input_bindings)
    # Freeze standalone launch identity before finite event/L2 admission.
    self.sequencer.context_launch_generation = self._context_launch_generation
    self.sequencer.context_name = task.name
    self.sequencer.device_slot = 0
    self.sequencer.formal_bindings = dict(formal_bindings or {})
    self.sequencer.load(task)
    event_outcome = self.scheduler.reserve_context_events(self.sequencer, task)
    if event_outcome.status is not IssueStatus.ACCEPTED:
      self.sequencer.admission_status = ContextAdmissionStatus.CANCELLED
      self.sequencer.mark_fault(event_outcome.reason)
      self.sequencer.done = True
      self._active_sequencers = [self.sequencer]
      self.pmu.add_event("event_admission_fault")
      return
    outcome = self.try_admit_l2_buffers(
      task, context_name=task.name, launch_generation=self._context_launch_generation, cycle=0
    )
    if outcome.status is not L2AdmissionStatus.ADMITTED:
      self.scheduler.cancel_context_events(self.sequencer)
      self.sequencer.admission_status = ContextAdmissionStatus.CANCELLED
      self.sequencer.mark_fault(
        self._l2_admission_fault_reason(outcome, "L2 capacity fault during context admission")
      )
      self.sequencer.done = True
      self._active_sequencers = [self.sequencer]
      self.pmu.add_event("l2_admission_permanent_fault")
      return
    self._live_launches[(task.name, 0, self._context_launch_generation)] = self.sequencer
    self.sequencer.admission_status = ContextAdmissionStatus.ACTIVE
    self._active_sequencers = [self.sequencer]
    # pre-init streams declared in the task (some tasks init inline)
    for stream in task.streams:
      self.init_stream(stream)

  def can_accept_context_launch(self) -> bool:
    """Whether the finite Group adapter may retain another launch slot."""

    retained = (
      sum(not sequencer.done for sequencer in self._active_sequencers)
      + len(self._pending_context_admissions)
      + len(self._pending_activations)
    )
    return retained < self.scheduler_config.active_context_capacity

  def load_context_task(
    self,
    task: ExecTileGroupTask,
    slot_index: int = 0,
    *,
    context_name: str | None = None,
    input_bindings=None,
    formal_bindings: dict[str, str] | None = None,
    cycle: int = 0,
  ) -> TileGroupSequencer:
    """Load a model-mode context task without resetting shared state.

    Deep-clones the task, then namespaces every event ID and stream
    queue ID with a monotonic launch ID (``s{slot}l{launch}_`` for
    events; integer queue offset for streams) so sequential
    re-submissions on the same slot cannot consume stale completions
    and concurrent tasks cannot collide on shared group-level tracking.

    Creates a fresh TileGroupSequencer for this task.  Tiles, DMA
    channels, L2, and program residency are shared across all
    concurrently-loaded context tasks.

    ``slot_index`` is a launch namespace only.  Unpinned dispatches remain
    unpinned for Tile-local selection; explicit physical ``context_id`` pins
    are preserved and validated by each Tile's admission API.

    PR 3.5: on a transient L2 capacity miss the returned sequencer is
    ``WAIT_CAPACITY`` (not faulted) — the device slot stays reserved
    but no UCE context, L1/L2 allocation, stream, or DMA/engine work is
    held.  The group's release-driven FIFO retry activates it later.
    """
    if not self.can_accept_context_launch():
      raise RuntimeError("Group active context capacity exhausted")
    ticket = self._prepare_context_launch(
      task,
      slot_index=slot_index,
      context_name=context_name,
      input_bindings=input_bindings,
      formal_bindings=formal_bindings,
      enqueue_cycle=cycle,
    )
    outcome = self._try_admit_prepared_context(ticket, cycle)
    if outcome.status is L2AdmissionStatus.WAIT_CAPACITY:
      # transient capacity miss: reserve the slot, wait for a release
      ticket.sequencer.admission_status = ContextAdmissionStatus.WAIT_CAPACITY
      ticket.sequencer.admission_wait_start_cycle = cycle
      ticket.wait_resource = (
        "event" if outcome.failure is not None and outcome.failure.buffer_id == "event_table" else "l2"
      )
      self._enqueue_pending_admission(ticket)
      return ticket.sequencer
    if outcome.status is L2AdmissionStatus.FAULT:
      # permanent/invalid: structured fault, never queued
      seq = ticket.sequencer
      seq.admission_status = ContextAdmissionStatus.CANCELLED
      seq.mark_fault(self._l2_admission_fault_reason(outcome))
      seq.done = True
      self._active_sequencers.append(seq)
      self.pmu.add_event("l2_admission_permanent_fault")
      return seq
    self._activate_admitted_context(ticket, cycle)
    return ticket.sequencer

  def _prepare_context_launch(
    self,
    task: ExecTileGroupTask,
    slot_index: int,
    *,
    context_name: str | None,
    input_bindings,
    formal_bindings: dict[str, str] | None,
    enqueue_cycle: int,
  ) -> _PendingContextAdmission:
    """Deep-clone + namespace + validate + build sequencer state.

    No L2 allocation, stream init or live-launch registration happens
    here: admission and activation are separate phases (PR 3.5).
    """
    # Device slot is an identity namespace, never a physical Tile context.
    launch_id = self._next_launch_id
    self._next_launch_id += 1
    # Deep-clone so the caller's task stays pristine: re-submitting the
    # same context re-namespaces from the clean original.
    task = copy.deepcopy(task)
    # Explicit physical context pins remain part of the low-level ABI.
    # Namespace event IDs with (slot, launch) so sequential slot reuse
    # cannot collide with stale completions.
    prefix = f"s{slot_index}l{launch_id}_"
    for action in task.actions:
      if action.dst is not None:
        action.dst = prefix + action.dst
      action.dependencies = tuple(prefix + event for event in action.dependencies)
      if action.op == ExecGroupActionOp.WAIT_EVENT:
        action.args = tuple(prefix + a if isinstance(a, str) else a for a in action.args)
      elif action.op == ExecGroupActionOp.SIGNAL_EVENT:
        action.args = tuple(prefix + a if isinstance(a, str) else a for a in action.args)
      elif action.op == ExecGroupActionOp.DISPATCH_ROLE:
        # PR 3: args = (ExecDispatchRequest,); prefix only the event
        # strings, never the grid identity (role_id/ordinal/policy).
        request = action.args[0]
        if isinstance(request, ExecDispatchRequest):
          action.args = (
            dataclasses.replace(
              request,
              input_released_event=(
                prefix + request.input_released_event if request.input_released_event else ""
              ),
              output_ready_event=(
                prefix + request.output_ready_event if request.output_ready_event else ""
              ),
            ),
          )
      elif action.op == ExecGroupActionOp.RELEASE_L2:
        # PR 3: args = (ExecReleaseRequest,); prefix dependency events
        # only; slot/role/ordinals are grid-relative, not namespaced.
        request = action.args[0]
        if isinstance(request, ExecReleaseRequest):
          action.args = (
            dataclasses.replace(
              request,
              dependency_events=tuple(
                prefix + ev if isinstance(ev, str) else ev for ev in request.dependency_events
              ),
            ),
          )
    # Namespace stream IDs into the launch's integer queue space.
    qid_offset = (launch_id * 100 + slot_index) * 10000
    owned_qids: set[int] = set()
    for action in task.actions:
      if action.op == ExecGroupActionOp.INIT_STREAM:
        # args = (queue_id, depth, producer_mask, consumer_mask)
        action.args = (int(action.args[0]) + qid_offset, *action.args[1:])
        owned_qids.add(int(action.args[0]))
    for s in task.streams:
      s.queue_id += qid_offset
      owned_qids.add(s.queue_id)
    for binding in task.role_bindings.values():
      if binding.out_stream is not None:
        binding.out_stream += qid_offset
      if binding.in_stream is not None:
        binding.in_stream += qid_offset
    stream_ops = {
      ExecTileOp.STREAM_PUSH,
      ExecTileOp.STREAM_POP,
      ExecTileOp.STREAM_ACQUIRE,
      ExecTileOp.STREAM_RELEASE,
      ExecTileOp.STREAM_PUSH_EOS,
    }
    seen_progs: set[int] = set()
    for binding in task.role_bindings.values():
      prog = binding.tile_program
      if id(prog) in seen_progs:
        continue
      seen_progs.add(id(prog))
      for inst in prog.insts:
        if inst.op in stream_ops and len(inst.args) >= 1:
          inst.args = (int(inst.args[0]) + qid_offset, *inst.args[1:])
        elif inst.op in (ExecTileOp.WAIT, ExecTileOp.WAITALL):
          # Namespace external group-DMA waits so the tile sees the
          # forwarded namespaced DMA completion (TileUCE matches by
          # exact id against _external_events_done).
          inst.args = tuple(prefix + a if isinstance(a, str) and "ev_dma_" in a else a for a in inst.args)
    # Also namespace the completion event
    task.completion_event = prefix + task.completion_event
    # PR 2: register global bindings (shared across contexts)
    if input_bindings:
      self.register_global_bindings(input_bindings)
    seq = TileGroupSequencer(self)
    seq.context_launch_generation = launch_id
    seq.context_name = context_name or task.name
    seq.device_slot = slot_index
    # PR 3.5: formal→actual mapping is launch-scoped on this sequencer
    seq.formal_bindings = dict(formal_bindings or {})
    seq.load(task)
    return _PendingContextAdmission(
      sequencer=seq,
      task=task,
      context_name=seq.context_name,
      device_slot=slot_index,
      launch_generation=launch_id,
      owned_queue_ids=frozenset(owned_qids),
      enqueue_cycle=enqueue_cycle,
      retry_count=0,
    )

  def _try_admit_prepared_context(self, ticket: _PendingContextAdmission, cycle: int) -> L2AdmissionOutcome:
    """Atomically reserve finite event metadata and admit the L2 bundle."""

    event_outcome = self.scheduler.reserve_context_events(ticket.sequencer, ticket.task)
    if event_outcome.status is IssueStatus.BACKPRESSURE:
      return L2AdmissionOutcome(
        L2AdmissionStatus.WAIT_CAPACITY,
        AdmissionFailure(AdmissionFailureKind.TEMPORARY_CAPACITY, event_outcome.reason, "event_table"),
      )
    if event_outcome.status is IssueStatus.FAULT:
      return L2AdmissionOutcome(
        L2AdmissionStatus.FAULT,
        AdmissionFailure(AdmissionFailureKind.INVALID_REQUEST, event_outcome.reason, "event_table"),
      )
    outcome = self.try_admit_l2_buffers(
      ticket.task, context_name=ticket.context_name, launch_generation=ticket.launch_generation, cycle=cycle
    )
    if outcome.status is not L2AdmissionStatus.ADMITTED:
      self.scheduler.cancel_context_events(ticket.sequencer)
    return outcome

  def reset(self) -> None:
    previous_cycle = self._last_step_cycle
    for t in self.tiles:
      t.reset()
    self.sequencer.reset()
    self.scheduler.reset()
    self._active_sequencers = []
    self._next_launch_id = 0
    for q in self.queues.values():
      q.reset()
    self._collective_jobs.clear()
    self._role_event_tile_mask.clear()
    self._role_done_tiles.clear()
    self._role_trace.clear()
    self.pmu = PMUCounter()
    # PR 3.5: staged activations already own L2; unwind them before
    # clearing the generation-keyed handle/role registries.
    self._cancel_pending_admissions(release_staged=True)
    # PR 3: clear structured grid registries
    self._grid_signals.clear()
    self._live_launches.clear()
    self._grid_l2_pins.clear()
    self._l2_roles.clear()
    self._protocol_live_l2.clear()
    self._task_trace_name = None
    self._task_start_cycle = None
    self._task_done_traced = False
    self._registered_programs.clear()
    # PR 2: clear admission + transfer state
    self._context_launch_generation = 0
    self._global_handles.clear()
    self._l2_handles.clear()
    self._role_l1_handles.clear()
    self._txn_sequencer.clear()
    self._clear_group_transfer_trace_slots()
    self.transfer_manager.reset()
    self.l2_mshr.reset()
    self.l2_cache.reset()
    self.hbm.reset()
    self._l2_capacity_change_cycle = None
    self._last_retried_pool_version = -1
    self._last_retried_capacity_change_cycle = -1
    self._last_retried_event_version = self.event_table.version
    self._last_step_cycle = 0
    for t in self.tiles:
      t.l1_allocator.reset()
    self.event_table.clear()
    if self.runtime_enabled:
      self.fault_ring.reset()
      self.program_table.reset()
      self.reset_domain.reset()
      self.l2_sram.reset()
    if self.memory_enabled:
      self.noc.reset()
      self.payload.reset()
    self._l2_reserved_bytes = 0
    self._l2_live_bytes = 0
    self._record_l2_occupancy(previous_cycle)
    self._l2_reserved_bytes_peak = 0
    self._l2_live_bytes_peak = 0
    self._last_retried_event_version = self.event_table.version

  # ---- inspection -----------------------------------------------------
  def _scheduler_snapshot(self) -> dict:
    snapshot = self.scheduler.snapshot()
    snapshot["reserved_l2_bytes"] = self._l2_reserved_bytes
    snapshot["live_l2_bytes"] = self._l2_live_bytes
    snapshot["reserved_l2_bytes_peak"] = self._l2_reserved_bytes_peak
    snapshot["live_l2_bytes_peak"] = self._l2_live_bytes_peak
    snapshot["active_contexts"] = sum(not sequencer.done for sequencer in self._active_sequencers)
    snapshot["contexts"] = [
      {
        "context": sequencer.context_name,
        "device_slot": sequencer.device_slot,
        "launch_generation": sequencer.context_launch_generation,
        "submission_pc": sequencer.submission_pc,
        "queued": sequencer.queued_count,
        "inflight": sequencer.inflight_count,
        "closed": sequencer.submission_closed,
        "faulted": sequencer.faulted,
      }
      for sequencer in self._active_sequencers
    ]
    return snapshot

  def snapshot(self) -> dict:
    return {
      "task_done": self.sequencer.done,
      "task_submission_pc": self.sequencer.submission_pc,
      "scheduler": self._scheduler_snapshot(),
      "event_table": self.event_table.snapshot(),
      "queues": {qid: q.snapshot() for qid, q in self.queues.items()},
      "tiles": [t.snapshot() for t in self.tiles],
      "memory": {
        "fidelity": self.fidelity,
        "hbm": self.hbm.snapshot() if self.runtime_enabled else None,
        "l2": self.l2_sram.snapshot() if self.runtime_enabled else None,
        "l1": {
          tile.tile_id: {
            "allocator": (tile.l1_allocator.snapshot() if self.runtime_enabled else None),
            "frames": [frame.snapshot() for frame in tile.l1_frames],
          }
          for tile in self.tiles
        },
        "cache": {
          "l2": self.l2_cache.snapshot(),
          "l1": {tile.tile_id: tile.l1_cache.snapshot() for tile in self.tiles},
        },
        "mshr": {
          "l2": self.l2_mshr.snapshot(),
          "l1": {tile.tile_id: tile.l1_mshr.snapshot() for tile in self.tiles},
        },
        "transfers": self.transfer_manager.snapshot(),
        "noc": self.noc.snapshot() if self.memory_enabled else None,
      },
      "collective_jobs": len(self._collective_jobs),
      "pending_context_admissions": [
        {
          "context_name": ticket.context_name,
          "device_slot": ticket.device_slot,
          "launch_generation": ticket.launch_generation,
          "enqueue_cycle": ticket.enqueue_cycle,
          "retry_count": ticket.retry_count,
          "wait_resource": ticket.wait_resource or "l2",
        }
        for ticket in self._pending_context_admissions
      ],
    }

  def all_tiles_done(self) -> bool:
    return all(t.done for t in self.tiles)

  def credit_invariants_hold(self) -> bool:
    return all(q.credit_invariant_holds() for q in self.queues.values())

  def on_scheduler_fault(self, sequencer: TileGroupSequencer, reason: str, cycle: int) -> None:
    """Stop one owner and enter the existing bounded Group fault drain."""

    sequencer.mark_fault(reason)
    self.pmu.add_event("group_scheduler_fault")
    if self.runtime_enabled and not (self.reset_domain.is_active or self.reset_domain.is_done):
      self.trigger_fault(self._fault_code_for_reason(reason), cycle=cycle, desc_id=reason)

  # ---- fault / reset (runtime fidelity) -------------------------------

  def trigger_fault(self, code, tile_id: int = -1, cycle: int = 0, desc_id: str = "") -> int:
    """Inject a fault: write a FaultRecord and begin the reset/drain FSM
    (Driver-Firmware 3.3/3.4).  Returns the fault_record_index, or -1
    in timing_only fidelity (no-op).
    """
    if not self.runtime_enabled:
      return -1
    rec = FaultRecord(code=code, tile_id=tile_id, desc_id=zlib.crc32(desc_id.encode()) & 0xFFFFFFFF)
    idx = self.fault_ring.write(rec)
    domain = FaultDomain.TILE if tile_id >= 0 else FaultDomain.GROUP
    req = ResetRequest(domain=domain, tile_id=tile_id, fault_record=rec)
    self.reset_domain.begin(req, cycle)
    self.pmu.add_event("fault_record", 1)
    return idx
