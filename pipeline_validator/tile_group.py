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

from .config import HardwareConfig
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
from .memory import (
  L2SRAM,
  AdmissionFailure,
  AdmissionFailureKind,
  AllocationHandle,
  AllocationPlan,
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
  TaskBufferOwner,
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
from .tile import ComputeTile
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
class _TileAdmission:
  """One tile's staged state for atomic role dispatch."""
  tile: ComputeTile
  logical_task_id: int
  context_id: int
  l1_plan: AllocationPlan | None
  prepare_cycles: int = 0
  l1_handles: tuple[AllocationHandle, ...] = ()
  bound: bool = False


@dataclass
class _GridL2Pin:
  """One logical task's pin on one L2 allocation (PR 3)."""

  task: TaskIdentity
  buffer_slot: str
  buffer_role: str
  handle: AllocationHandle
  consumer_id: str


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
  """One prepared context waiting for L2 capacity (PR 3.5 FIFO ticket).

  A waiting ticket owns a device slot and a namespaced prepared task,
  but no L2/L1 allocation, stream, UCE context or DMA/engine work.
  """

  sequencer: TileGroupSequencer
  task: ExecTileGroupTask
  context_name: str
  device_slot: int
  launch_generation: int
  owned_queue_ids: frozenset[int]
  enqueue_cycle: int
  retry_count: int = 0


class TileGroup:
  """One ELENOR Tile Group with 4 Compute Tiles."""

  def __init__(self, cfg: HardwareConfig, tracer: Tracer | None = None,
               fidelity: str = "full_memory", context_count: int = 1,
               memory_trace: bool = False):
    self.cfg = cfg
    self.tracer = tracer
    self.fidelity = fidelity
    rt = fidelity in ("runtime", "full_memory")
    mem = fidelity == "full_memory"
    self.runtime_enabled = rt
    self.memory_enabled = mem
    # PR 5: memory lanes/counters/flows + report peaks are opt-in so a
    # plain --trace-json run still emits the pre-PR5 control-flow trace.
    self.memory_trace = (MemoryTrace(tracer)
                         if tracer is not None and memory_trace else None)
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
        trace=self.memory_trace)
    self.transfer_manager = TransferManager(
      cfg, full_memory=mem, noc=self.noc if mem else None,
      trace=self.memory_trace)
    self.l2_cache = DeterministicLRUCache(
      cfg.l2_cache_capacity_bytes, cfg.cache_line_bytes)
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
    self._active_sequencers: list[TileGroupSequencer] = [self.sequencer]
    self._next_launch_id: int = 0
    self.queues: dict[int, StreamQueue] = {}
    self._collective_jobs: list[_CollectiveJob] = []
    self.pmu = PMUCounter()
    self._registered_programs: dict[tuple[int, int, int], int] = {}
    # role dispatch fan-in by event id: event_id -> tile_mask / done tiles
    self._role_event_tile_mask: dict[str, int] = {}
    self._role_done_tiles: dict[str, set[int]] = {}
    self._role_trace: dict[str, _RoleTrace] = {}
    # PR 3: structured grid registries - signal aggregation + live launches
    self._grid_signals: dict[GridInstanceId, _GridSignalState] = {}
    self._live_launches: dict[
      tuple[str, int, int], TileGroupSequencer] = {}
    # PR 3: grid -> {task_id -> {buffer_slot -> pin}}
    self._grid_l2_pins: dict[
      GridInstanceId, dict[int, dict[str, _GridL2Pin]]] = {}
    self._task_done_traced: bool = False
    self._task_trace_name: str | None = None
    self._task_start_cycle: int | None = None
    self.hbm = HBMRegion(
      base_iova=0,
      size_bytes=cfg.hbm_capacity_bytes,
      bandwidth_gbs=cfg.hbm_bandwidth_gbs,
      outstanding_limit=cfg.hbm_outstanding_limit,
      trace=self.memory_trace)
    # PR 2 admission state: launch generation, global/L2 handles, pins
    self._context_launch_generation: int = 0
    self._global_handles: dict[str, AllocationHandle] = {}  # binding name -> handle
    self._l2_handles: dict[tuple[int, str], AllocationHandle] = {}  # (gen, slot) -> handle
    # PR 3: launch generation -> {buffer_slot -> alloc role}
    self._l2_roles: dict[int, dict[str, str]] = {}
    # PR 3.5: L2 admission wait queue + release-driven FIFO retry
    self._pending_context_admissions: deque[_PendingContextAdmission] = deque()
    self._pending_activations: list[_PendingContextAdmission] = []
    self._l2_capacity_change_cycle: int | None = None
    self._last_retried_pool_version: int = -1
    self._last_retried_capacity_change_cycle: int = -1
    self._last_step_cycle: int = 0
    # PR 2: role_event_id -> {tile_id: [L1 AllocationHandle]}
    self._role_l1_handles: dict[str, dict[int, list]] = {}
    # transaction id -> sequencer
    self._txn_sequencer: dict[str, TileGroupSequencer] = {}
    # group transaction id -> (logical direction, visual concurrency slot)
    self._group_transfer_trace_slots: dict[str, tuple[str, int]] = {}
    self._group_transfer_trace_busy_slots: dict[str, set[int]] = {
      "input": set(),
      "output": set(),
    }
    if self.runtime_enabled:
      self.event_table = EventTable()
      self.fault_ring = FaultRing()
      self.program_table = ProgramResidencyManager(cfg)
      self.reset_domain = ResetDomain(cfg)
    if self.memory_enabled or self.runtime_enabled:
      self.l2_sram = L2SRAM(
        capacity_bytes=cfg.group_sram_bytes,
        banks=cfg.group_sram_banks,
        bank_bandwidth_gbs=cfg.l2_bank_bandwidth_gbs,
        trace=self.memory_trace)
    if self.memory_enabled:
      self.payload = PayloadTracker()

  # ---- setup ----------------------------------------------------------

  def init_stream(self, desc: ExecStreamDesc) -> StreamQueue:
    # masks are tile-bit masks: a bit set means that tile participates
    producers = frozenset(i for i in range(self.cfg.num_tiles)
                          if desc.producer_mask & (1 << i))
    consumers = frozenset(i for i in range(self.cfg.num_tiles)
                          if desc.consumer_mask & (1 << i))
    q = StreamQueue(
      queue_id=desc.queue_id,
      depth=desc.depth,
      producers=producers,
      consumers=consumers,
      kind=QueueKind.MPSC if len(producers) > 1 else QueueKind.SPSC,
      eos_policy=EOSPolicy.ALL_PRODUCERS
      if len(producers) > 1 else EOSPolicy.SINGLE_PRODUCER,
    )
    q.init()
    self.queues[desc.queue_id] = q
    # bind to every tile that participates
    for t in self.tiles:
      if (desc.producer_mask | desc.consumer_mask) & (1 << t.tile_id):
        t.bind_stream(desc.queue_id, q)
    return q

  @staticmethod
  def _group_transfer_trace_direction(op: TransferOp) -> str:
    if op is TransferOp.PREFETCH:
      return "input"
    if op is TransferOp.GLOBAL_STORE:
      return "output"
    raise ValueError(f"unsupported group transfer op {op.value}")

  def _reserve_group_transfer_trace_slot(
    self,
    transaction_id: str,
    op: TransferOp,
  ) -> None:
    if transaction_id in self._group_transfer_trace_slots:
      raise ValueError(f"duplicate group transfer trace slot {transaction_id}")
    direction = self._group_transfer_trace_direction(op)
    busy = self._group_transfer_trace_busy_slots[direction]
    slot = 0
    while slot in busy:
      slot += 1
    busy.add(slot)
    self._group_transfer_trace_slots[transaction_id] = (direction, slot)

  def _release_group_transfer_trace_slot(
    self,
    transaction_id: str,
  ) -> tuple[str, int] | None:
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
    category = ("HBM → L2 Input" if direction == "input"
                else "L2 → HBM Output")
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
    gen = (sequencer.context_launch_generation
           if sequencer is not None else self._context_launch_generation)
    # PR 3.5: formal→actual mapping is launch-scoped on the issuing
    # sequencer; pending B must never overwrite active A's mapping.
    formals = sequencer.formal_bindings if sequencer is not None else {}
    txn_id = f"{gen}:{event_id}"
    context_name = (
      sequencer.context_name
      if sequencer is not None and sequencer.context_name
      else "ctx"
    )
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
        op=(TransferOp.PREFETCH if op == "dma.prefetch"
            else TransferOp.GLOBAL_STORE),
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
        op=(TransferOp.PREFETCH if op == "dma.prefetch"
            else TransferOp.GLOBAL_STORE),
        issuer=ContextBufferOwner(context_name, gen, buffer_id),
        src=None, dst=None,
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

  def _resolve_view(self, view, default_space: str,
                    gen: int | None = None,
                    formal_bindings: dict[str, str] | None = None
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
        segs = (BankSegment(0, handle.base_address + offset,
                             view.bytes),)
      return ResolvedMemoryView(
        handle=handle, offset_bytes=offset, size_bytes=view.bytes,
        address=segs[0].address, segments=segs)
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
      for d in view.backing_dims[i + 1:]:
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
    self,
    task: ExecTileGroupTask,
    *,
    context_name: str,
    launch_generation: int,
    cycle: int,
  ) -> L2AdmissionOutcome:
    """L2 admission: plan + commit all ``l2_buffers`` as one bundle.

    Typed outcome (PR 3.5): ``WAIT_CAPACITY`` means the bundle is
    legally placeable but the current live free map cannot satisfy it
    (a future release may); ``FAULT`` covers invalid requests and
    bundles that can never fit.  A failed plan has zero side effects.
    ``launch_generation`` is the caller's ticket/sequencer generation —
    never read implicitly from shared state.
    """
    if not (self.memory_enabled or self.runtime_enabled) or not task.l2_buffers:
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
        else L2AdmissionStatus.FAULT)
      return L2AdmissionOutcome(status, plan)
    handles = self.l2_sram.commit(plan, cycle)
    for buf, handle in zip(task.l2_buffers, handles):
      self._l2_handles[(launch_generation, buf.slot)] = handle
    self._l2_roles[launch_generation] = {
      buf.slot: buf.role for buf in task.l2_buffers}
    return L2AdmissionOutcome(L2AdmissionStatus.ADMITTED)

  def release_l2(self, request: ExecReleaseRequest,
                 sequencer: TileGroupSequencer, cycle: int) -> bool:
    """Gated release of one context-owned L2 buffer (PR 3).

    The issuing sequencer has already confirmed every dependency event
    fired.  Raises ``MemoryInvariantError`` on any ownership, role,
    generation or pin violation; never silently succeeds.  The handle
    stays in ``_l2_handles`` until context cleanup so a duplicate
    release reaches the allocator's double-release check.
    """
    gen = sequencer.context_launch_generation
    slot = request.buffer_slot
    handle = self._l2_handles.get((gen, slot))
    if handle is None:
      raise MemoryInvariantError(
        f"release references unknown L2 buffer '{slot}'"
        f" in launch generation {gen}")
    expected_owner = ContextBufferOwner(
      sequencer.context_name, gen, slot)
    declared_role = self._l2_roles.get(gen, {}).get(slot, "")
    if declared_role != request.buffer_role:
      raise MemoryInvariantError(
        f"release role mismatch on slot '{slot}':"
        f" alloc '{declared_role}' != request '{request.buffer_role}'")
    self.l2_sram.assert_live(handle, expected_owner)
    launch_pins = [
      (grid, pins) for grid, pins in self._grid_l2_pins.items()
      if (grid.context_name == sequencer.context_name
          and grid.launch_generation == gen)
    ]
    if request.buffer_role == "in":
      # input pins must already be gone via the input_released aggregates
      remaining = [
        (grid, task_id) for grid, pins in launch_pins
        for task_id, slot_pins in pins.items()
        if slot in slot_pins]
      if remaining:
        raise MemoryInvariantError(
          f"release of input slot '{slot}' before its input_released"
          " aggregates unpinned every consumer")
    else:
      # out/inout: final store dependency fired; drop every matching
      # consumer pin of this buffer.
      for grid, pins in launch_pins:
        for _task_id, slot_pins in pins.items():
          pin = slot_pins.get(slot)
          if pin is None or (pin.task.grid.dispatch_ordinal
                             not in request.consumer_dispatch_ordinals):
            continue
          self.l2_sram.unpin(pin.handle, pin.consumer_id, cycle)
          slot_pins.pop(slot, None)
        if pins and all(not sp for sp in pins.values()):
          self._grid_l2_pins.pop(grid, None)
      remaining = [
        (grid, task_id) for grid, pins in self._grid_l2_pins.items()
        if (grid.context_name == sequencer.context_name
            and grid.launch_generation == gen)
        for task_id, slot_pins in pins.items()
        if slot in slot_pins]
      if remaining:
        raise MemoryInvariantError(
          f"release of slot '{slot}' with undeclared pinned consumers")
    freed = self.l2_sram.request_release(handle, expected_owner, cycle)
    if not freed:
      raise MemoryInvariantError(
        f"release of slot '{slot}' left pinned consumers")
    # PR 3.5: only an allocator final-free makes new capacity available;
    # signal aggregates and unpins alone never wake the admission queue.
    self._l2_capacity_change_cycle = cycle
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
        self._l2_capacity_change_cycle
        if self._l2_capacity_change_cycle is not None else -1)
    self.pmu.add_event("l2_admission_wait")
    peak = self.pmu.named_cycles["l2_admission_queue_peak"]
    if len(self._pending_context_admissions) > peak:
      self.pmu.named_cycles["l2_admission_queue_peak"] = len(
        self._pending_context_admissions)
    if self.tracer is not None:
      self.tracer.instant("TileGroup", "Scheduler:L2",
                          "context_admission_wait", ticket.enqueue_cycle, {
                            "context": ticket.context_name,
                            "slot": ticket.device_slot,
                            "launch_generation": ticket.launch_generation,
                            "cycle": ticket.enqueue_cycle,
                          })

  def _activate_admitted_context(
    self, ticket: _PendingContextAdmission, cycle: int,
  ) -> None:
    """Activate an admitted context: live launch, active list, streams.

    Called at the post-sequencer barrier, so the new sequencer's first
    group action issues at ``cycle + 1``.
    """
    seq = ticket.sequencer
    if seq.admission_status is ContextAdmissionStatus.WAIT_CAPACITY:
      self.pmu.add_event("l2_admission_wakeup")
      self.pmu.add_cycle("l2_admission_wait_cycles",
                         cycle - ticket.enqueue_cycle)
    seq.admission_status = ContextAdmissionStatus.ACTIVE
    seq.admission_wait_start_cycle = None
    self._live_launches[
      (seq.context_name, seq.device_slot, seq.context_launch_generation)] = seq
    seq.owned_queue_ids = set(ticket.owned_queue_ids)
    self._active_sequencers.append(seq)
    for s in ticket.task.streams:
      self.init_stream(s)
    if self.tracer is not None:
      l2_live = (self.l2_sram.snapshot()["live_allocations"]
                 if (self.memory_enabled or self.runtime_enabled) else 0)
      self.tracer.instant("TileGroup", "Scheduler:L2", "context_admitted",
                          cycle, {
                            "context": seq.context_name,
                            "slot": seq.device_slot,
                            "launch_generation":
                              seq.context_launch_generation,
                            "cycle": cycle,
                            "l2_live_allocations": l2_live,
                          })

  def _retry_pending_context_admissions(self, cycle: int) -> None:
    """FIFO admission retry at the post-sequencer barrier.

    Triggered only by a ``release_l2`` final-free notification: a
    commit, unpin or signal never starts a pass (they shrink or do not
    change available capacity).  The allocator pool version dedups
    duplicate notifications.  Strict FIFO — stop at the first head
    that still cannot fit.  Successful admissions are staged in
    ``_pending_activations`` and activated after the current sequencer
    iteration, never re-entered mid-loop.
    """
    if not self._pending_context_admissions:
      return
    if not (self.memory_enabled or self.runtime_enabled):
      return
    if (self._l2_capacity_change_cycle is None
        or self._l2_capacity_change_cycle
        <= self._last_retried_capacity_change_cycle):
      return
    version = self.l2_sram.pool_version
    if version == self._last_retried_pool_version:
      return
    self._last_retried_pool_version = version
    self._last_retried_capacity_change_cycle = self._l2_capacity_change_cycle
    staged: list[_PendingContextAdmission] = []
    while self._pending_context_admissions:
      ticket = self._pending_context_admissions[0]
      ticket.retry_count += 1
      ticket.sequencer.admission_retry_count = ticket.retry_count
      self.pmu.add_event("l2_admission_retry")
      if self.tracer is not None:
        self.tracer.instant("TileGroup", "Scheduler:L2",
                            "context_admission_retry", cycle, {
                              "context": ticket.context_name,
                              "slot": ticket.device_slot,
                              "launch_generation": ticket.launch_generation,
                              "cycle": cycle,
                              "retry_count": ticket.retry_count,
                              "capacity_change_cycle":
                                self._l2_capacity_change_cycle,
                            })
      outcome = self._try_admit_prepared_context(ticket, cycle)
      if outcome.status is L2AdmissionStatus.WAIT_CAPACITY:
        break  # strict FIFO: head still cannot fit
      self._pending_context_admissions.popleft()
      if outcome.status is L2AdmissionStatus.FAULT:
        # Defensive: a queued ticket became permanently impossible.
        ticket.sequencer.admission_status = ContextAdmissionStatus.CANCELLED
        ticket.sequencer.faulted = True
        ticket.sequencer.fault_reason = (
          "L2 capacity fault during context admission")
        ticket.sequencer.done = True
        self._active_sequencers.append(ticket.sequencer)
        self.pmu.add_event("l2_admission_permanent_fault")
        break
      staged.append(ticket)
    self._pending_activations.extend(staged)
    self._last_retried_pool_version = self.l2_sram.pool_version

  def _cancel_pending_admissions(
    self, cycle: int | None = None, *, release_staged: bool = False,
  ) -> None:
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
      terminal = (
        cycle if cycle is not None
        else max(self._last_step_cycle, ticket.enqueue_cycle))
      self.pmu.add_cycle(
        "l2_admission_wait_cycles", terminal - ticket.enqueue_cycle)
      if self.tracer is not None:
        self.tracer.instant("TileGroup", "Scheduler:L2",
                            "context_admission_cancelled", terminal, {
                              "context": ticket.context_name,
                              "slot": ticket.device_slot,
                              "launch_generation": ticket.launch_generation,
                              "cycle": terminal,
                            })
    if release_staged and (self.memory_enabled or self.runtime_enabled):
      for ticket in staged:
        generation = ticket.launch_generation
        release_cycle = (
          cycle if cycle is not None
          else max(self._last_step_cycle, ticket.enqueue_cycle))
        for key, handle in list(self._l2_handles.items()):
          if key[0] != generation:
            continue
          try:
            self.l2_sram.request_release(
              handle, handle.owner, release_cycle)
          except MemoryInvariantError:
            pass
          self._l2_handles.pop(key, None)
        self._l2_roles.pop(generation, None)
    self._pending_context_admissions.clear()
    self._pending_activations.clear()

  def _pin_grid_l2(self, grid: GridInstanceId,
                   binding: ExecTileRoleBinding,
                   task: TaskIdentity, cycle: int) -> None:
    """Pin each unique L2 allocation once for one logical task of a grid.

    ``actuals`` contains both ins and outs; slot dedup prevents an
    inout buffer from receiving a duplicate pin.
    """
    if not (self.memory_enabled or self.runtime_enabled):
      return
    roles = self._l2_roles.get(grid.launch_generation, {})
    slot_pins = self._grid_l2_pins.setdefault(
      grid, {}).setdefault(task.task_id, {})
    pinned: list[tuple[str, AllocationHandle, str]] = []
    try:
      for slot in dict.fromkeys(binding.actuals):
        handle = self._l2_handles.get((grid.launch_generation, slot))
        if handle is None:
          continue
        consumer_id = (
          f"{grid.context_name}:s{grid.device_slot}"
          f":g{grid.launch_generation}:d{grid.dispatch_ordinal}"
          f":t{task.task_id}:{slot}")
        self.l2_sram.pin(handle, consumer_id)
        pinned.append((slot, handle, consumer_id))
    except MemoryInvariantError:
      for _slot, handle, consumer_id in pinned:
        self.l2_sram.unpin(handle, consumer_id, cycle)
      raise
    for slot, handle, consumer_id in pinned:
      slot_pins[slot] = _GridL2Pin(
        task=task, buffer_slot=slot,
        buffer_role=roles.get(slot, "inout"),
        handle=handle, consumer_id=consumer_id)

  def _unpin_grid_inputs(self, grid: GridInstanceId, cycle: int) -> None:
    """Unpin every role='in' allocation of one grid (input_released)."""
    pins = self._grid_l2_pins.get(grid)
    if not pins:
      return
    for task_id in list(pins):
      for slot in list(pins[task_id]):
        pin = pins[task_id][slot]
        if pin.buffer_role != "in":
          continue
        try:
          self.l2_sram.unpin(pin.handle, pin.consumer_id, cycle)
        except MemoryInvariantError:
          pass
        pins[task_id].pop(slot)
      if not pins[task_id]:
        pins.pop(task_id)
    if not pins:
      self._grid_l2_pins.pop(grid, None)

  def _unwind_grid_l2_pins(self, cycle: int,
                           grids: list[GridInstanceId] | None = None) -> None:
    """Idempotently unpin every (or only the selected) grid's L2 pins."""
    if not (self.memory_enabled or self.runtime_enabled):
      if grids is None:
        self._grid_l2_pins.clear()
      else:
        for grid in grids:
          self._grid_l2_pins.pop(grid, None)
      return
    targets = list(self._grid_l2_pins.items()) if grids is None else [
      (grid, self._grid_l2_pins[grid])
      for grid in grids if grid in self._grid_l2_pins]
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
    """Release every context-owned L2/L1 allocation, pin and frame.

    Fault-domain reset cleanup (§6.5): returns all context-owned
    handles to their allocators.  External HBM bindings are preserved.
    Release errors (double release / stale generation) are tolerated so
    reset never faults while cleaning up.
    """
    # PR 3.5: waiting tickets own no allocation — cancel without release
    self._cancel_pending_admissions(cycle)
    self.transfer_manager.cancel_all(cycle)
    self._clear_group_transfer_trace_slots()
    self.l2_mshr.reset()
    self.l2_cache.reset()
    if not (self.memory_enabled or self.runtime_enabled):
      return
    # marked a handle RELEASE_PENDING, the final reset-time unpin
    # performs the actual free.
    self._unwind_grid_l2_pins(cycle)
    # Structured grid registries belong to the drained launch.
    self._grid_signals.clear()
    self._live_launches.clear()
    for handle in list(self._l2_handles.values()):
      try:
        self.l2_sram.request_release(handle, handle.owner, cycle)
      except MemoryInvariantError:
        pass
    self._l2_handles.clear()
    for tile_l1 in self._role_l1_handles.values():
      for tile_id, handles in tile_l1.items():
        alloc = self.tiles[tile_id].l1_allocator
        for handle in handles:
          try:
            alloc.request_release(handle, handle.owner, cycle)
          except MemoryInvariantError:
            pass
    self._role_l1_handles.clear()
    # Cancelled transfer resources are already returned by reset cleanup;
    # now clear UCE contexts/queued engine work and invalidate L1 handles.
    for t in self.tiles:
      t.reset()

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
      ))
    if sequencer is not None:
      sequencer.note_job_started()
  def can_dispatch_role(self, binding: ExecTileRoleBinding) -> bool:
    return all(t.can_accept_context(binding.context_id)
               for t in self.tiles
               if binding.tile_mask & (1 << t.tile_id))

  def dispatch_role(
    self,
    binding: ExecTileRoleBinding,
    cycle: int,
    request: ExecDispatchRequest,
    event_id: str | None = None,
    sequencer: TileGroupSequencer | None = None,
  ) -> bool:
    """Atomically admit and bind one TileRole across all selected tiles.

    Phase 1 selects every UCE context and plans every L1 allocation with
    zero side effects.  Phase 2 commits/prepares every tile, pins L2
    consumers, then binds exact contexts.  Any late failure rolls back
    all earlier commits/pins/frames/contexts and faults the issuing
    sequencer (never the default sequencer by accident).

    PR 3: the dispatch's grid identity registers a ``_GridSignalState``
    whose expected task set comes from the role binding's task domain;
    every admitted context receives its ``TaskIdentity``.
    """
    role_id = binding.role_id
    tile_mask = binding.tile_mask
    ev = event_id or f"ev_role{role_id}"
    seq = sequencer or self.sequencer
    prog = binding.tile_program
    gen = seq.context_launch_generation
    grid = seq.grid_id(request.dispatch_ordinal)
    from_task = (binding.task_domain.from_task
                 if binding.task_domain is not None else 0)
    to_task = (binding.task_domain.to_task
               if binding.task_domain is not None else 0)
    expected_task_ids = frozenset(range(from_task, to_task))
    phase_event_ids: dict[str, str] = {}
    if request.input_released_event:
      phase_event_ids["input_released"] = request.input_released_event
    if request.output_ready_event:
      phase_event_ids["output_ready"] = request.output_ready_event
    admissions: list[_TileAdmission] = []

    def fail(reason: str) -> bool:
      seq.faulted = True
      seq.fault_reason = reason
      seq.done = True
      seq.pmu.add_event("l1_admission_fault")
      return False

    def rollback() -> None:
      # Undo bound contexts first; no UCE step occurred during dispatch.
      for adm in reversed(admissions):
        if adm.bound:
          adm.tile.rollback_program(adm.context_id, cycle)
        else:
          frame = adm.tile.l1_frames[adm.context_id]
          frame.release()
          if self.memory_trace is not None and self.tracer is not None:
            self.tracer.instant(
              f"Tile{adm.tile.tile_id}", "Lifecycle", "frame_release",
              cycle, {"ctx_id": adm.context_id,
                      "generation": frame.generation,
                      "tile_id": adm.tile.tile_id,
                      "reason": "dispatch_rollback"})
      # Remove grid L2 consumer pins created during phase 2.
      self._unwind_grid_l2_pins(cycle, [grid])
      self._grid_signals.pop(grid, None)
      # Return every committed L1 allocation.
      for adm in admissions:
        for handle in adm.l1_handles:
          try:
            adm.tile.l1_allocator.request_release(
              handle, handle.owner, cycle)
          except MemoryInvariantError:
            pass
      self._role_l1_handles.pop(ev, None)
      self._role_event_tile_mask.pop(ev, None)
      self._role_done_tiles.pop(ev, None)
      self._role_trace.pop(ev, None)


    # Phase 1: select exact contexts and plan all L1 bundles.  No commit,
    # frame prepare, pin or context bind is allowed in this phase.
    ordinal = 0
    for tile in self.tiles:
      if not (tile_mask & (1 << tile.tile_id)):
        continue
      logical_task_id = from_task + ordinal
      ordinal += 1
      context_id = tile.available_context_id(binding.context_id)
      if context_id is None:
        for prior in admissions:
          if prior.l1_plan is not None:
            prior.tile.l1_allocator.rollback(prior.l1_plan)
        return fail(f"UCE context unavailable on tile {tile.tile_id}")
      l1_plan = None
      if (self.memory_enabled or self.runtime_enabled) and prog.l1_buffers:
        requests = [
          AllocationRequest(
            memory_space="l1",
            buffer_id=buf.name,
            owner=TaskBufferOwner(
              seq.task.name if seq.task is not None else "ctx",
              gen, ev, logical_task_id, tile.tile_id, context_id, buf.name),
            size_bytes=buf.bytes,
            alignment=max(buf.alignment, 1),
          )
          for buf in prog.l1_buffers
        ]
        candidate = tile.l1_allocator.plan_bundle(requests)
        if isinstance(candidate, AdmissionFailure):
          for prior in admissions:
            if prior.l1_plan is not None:
              prior.tile.l1_allocator.rollback(prior.l1_plan)
          return fail(
            f"L1 capacity fault on tile {tile.tile_id}: {candidate.reason}")
        l1_plan = candidate
      admissions.append(_TileAdmission(
        tile=tile, logical_task_id=logical_task_id,
        context_id=context_id, l1_plan=l1_plan))

    # Program metadata/residency is cache state; it is safe to retain if a
    # later allocation/bind rollback occurs.
    total_cold = 0
    if self.runtime_enabled and prog.program_id != 0:
      identity = (prog.program_id, prog.version, prog.program_hash)
      cached = self._registered_programs.get(identity)
      if cached is None:
        cached = self._program_bytes(prog)
        self._registered_programs[identity] = cached
        self.program_table.register(
          program_id=prog.program_id, version=prog.version,
          program_hash=prog.program_hash,
          hbm_iova=0, hbm_bytes=cached)
      for adm in admissions:
        adm.prepare_cycles = self.program_table.ensure_resident(
          prog.program_id, adm.tile.tile_id, cycle)
        total_cold += adm.prepare_cycles

    # Phase 2a: commit all L1 plans, then prepare all exact context frames.
    try:
      for adm in admissions:
        if adm.l1_plan is None:
          continue
        adm.l1_handles = adm.tile.l1_allocator.commit(
          adm.l1_plan, cycle)
        if not adm.tile.l1_frames[adm.context_id].prepare(
            list(adm.l1_handles), list(prog.l1_buffers)):
          raise MemoryInvariantError(
            f"L1 frame prepare failed on tile {adm.tile.tile_id}")
        if self.memory_trace is not None and self.tracer is not None:
          self.tracer.instant(
              f"Tile{adm.tile.tile_id}", "Lifecycle", "frame_prepare",
              cycle, {
                "ctx_id": adm.context_id,
                "generation":
                  adm.tile.l1_frames[adm.context_id].generation,
                "tile_id": adm.tile.tile_id,
                "slots": len(prog.l1_buffers),
              })
      # Pin each unique L2 allocation once per logical task only after all
      # L1 commits/prepares succeeded.
      for adm in admissions:
        self._pin_grid_l2(
          grid, binding,
          TaskIdentity(grid=grid, task_id=adm.logical_task_id), cycle)
      # Phase 2b: bind exact contexts.  Single-threaded cycle-step means no
      # other dispatcher can steal a context between plan and bind.
      from .tile import _TileContextMemory
      for adm in admissions:
        task_identity = TaskIdentity(
          grid=grid, task_id=adm.logical_task_id)
        l2_handle_map: dict[int, AllocationHandle] = {}
        global_view_map: dict[int, ResolvedMemoryView] = {}
        if self.memory_enabled or self.runtime_enabled:
          l2_formals = [
            (formal_index, formal)
            for formal_index, formal in enumerate(prog.formals)
            if formal.space == "l2"
          ]
          if len(binding.actuals) < len(l2_formals):
            raise MemoryInvariantError("missing L2 actual for tile formal")
          for (formal_index, _formal), slot in zip(
            l2_formals, binding.actuals[:len(l2_formals)]
          ):
            handle = self._l2_handles.get((gen, slot))
            if handle is None:
              raise MemoryInvariantError("missing or stale L2 actual for tile formal")
            l2_handle_map[formal_index] = handle

          global_formals = [
            (formal_index, formal)
            for formal_index, formal in enumerate(prog.formals)
            if formal.space == "global"
          ]
          if len(binding.global_actuals) != len(global_formals):
            raise MemoryInvariantError("missing global actual for tile formal")
          for (formal_index, _formal), actual in zip(
            global_formals, binding.global_actuals
          ):
            resolved = self._resolve_view(
              actual,
              "global",
              gen,
              seq.formal_bindings,
            )
            if resolved is None:
              raise MemoryInvariantError("missing or stale global actual for tile formal")
            global_view_map[formal_index] = resolved
        l1_map = {
          buf.name: handle
          for buf, handle in zip(prog.l1_buffers, adm.l1_handles)
        }
        memory = _TileContextMemory(
          task_identity=task_identity,
          l2_formal_handles=l2_handle_map,
          global_formal_views=global_view_map,
          l1_handles=l1_map,
          l2_resolver=(
            self.l2_sram
            if (self.memory_enabled or self.runtime_enabled)
            else None
          ),
        )
        ctx_id = adm.tile.load_program(
          prog, role_id=role_id, role_event_id=ev,
          prepare_cycles=adm.prepare_cycles,
          context_id=adm.context_id, memory=memory,
          task_identity=task_identity)
        if ctx_id != adm.context_id:
          raise MemoryInvariantError(
            f"UCE context bind failed on tile {adm.tile.tile_id}")
        adm.bound = True
    except (MemoryInvariantError, ValueError) as exc:
      rollback()
      return fail(str(exc))

    # Publish role + grid bookkeeping only after every tile bound
    # successfully.
    self._role_event_tile_mask[ev] = tile_mask
    self._role_done_tiles[ev] = set()
    self._grid_signals[grid] = _GridSignalState(
      grid=grid,
      expected_task_ids=expected_task_ids,
      policy=request.signal_policy,
      phase_event_ids=phase_event_ids,
      sequencer=seq,
    )
    self._role_l1_handles[ev] = {
      adm.tile.tile_id: list(adm.l1_handles) for adm in admissions
      if adm.l1_handles
    }
    self._role_trace[ev] = _RoleTrace(
      role_id=role_id, event_id=ev, start_cycle=cycle,
      tile_mask=tile_mask, out_stream=binding.out_stream,
      in_stream=binding.in_stream, sequencer=seq)
    ctx_ids = [adm.context_id for adm in admissions]
    for adm in admissions:
      tile = adm.tile
      if self.runtime_enabled:
        tile.uce._event_done_callback = self._make_event_callback()
      tile.uce._phase_signal_callback = self._on_phase_signal
      for qid, queue in self.queues.items():
        tile.bind_stream(qid, queue)
      for done_ev in sorted(seq._events_done):
        if "ev_dma_" in done_ev:
          tile.uce.notify_event(done_ev)

    tr = self.tracer
    if tr is not None:
      ctx_trace_arg = ctx_ids[0] if len(set(ctx_ids)) == 1 else ctx_ids
      tr.instant(
        "TileGroup", f"TileRole:{role_id}", "tile_role_dispatch", cycle,
        {
          "role_id": role_id,
          "tile_mask": tile_mask,
          "program": prog.name,
          "event_id": ev,
          "out_stream": binding.out_stream,
          "in_stream": binding.in_stream,
          "ctx_id": ctx_trace_arg,
          "pinned_context": binding.context_id,
          "context_count": self.tiles[0].uce.context_count,
        })
    if total_cold > 0:
      self.pmu.add_cycle("program_cold_load", total_cold)
    return True

  def _on_phase_signal(self, signal: PhaseSignal, cycle: int) -> None:
    """Aggregate one task phase signal against its grid (PR 3).

    1. retired launch -> stale, ignored (+PMU ``tile_signal_stale``);
    2. live launch but unknown grid/task/phase -> invalid signal fault;
    3. duplicate ``(grid, phase, task)`` -> idempotent ignore;
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

    launch_key = (
      grid.context_name, grid.device_slot, grid.launch_generation)
    seq = self._live_launches.get(launch_key)
    if seq is None:
      self.pmu.add_event("tile_signal_stale")
      if tr is not None:
        tr.instant("TileGroup", "Scheduler:L2", "tile_signal_stale",
                   cycle, _signal_args())
      return
    state = self._grid_signals.get(grid)
    if (state is None
        or signal.task.task_id not in state.expected_task_ids
        or not state.phase_declared(signal.phase)):
      self.pmu.add_event("tile_signal_invalid")
      if tr is not None:
        tr.instant("TileGroup", "Scheduler:L2", "tile_signal_invalid",
                   cycle, _signal_args())
      seq.faulted = True
      seq.fault_reason = (
        f"invalid tile signal: grid {grid} task {signal.task.task_id}"
        f" phase '{signal.phase}'")
      seq.done = True
      if (self.runtime_enabled
          and not (self.reset_domain.is_active
                   or self.reset_domain.is_done)):
        self.trigger_fault(
          FaultCode.ADDRESS_FAULT, tile_id=-1, cycle=cycle,
          desc_id=seq.fault_reason)
      return
    seen = state.seen.setdefault(signal.phase, set())
    if signal.task.task_id in seen:
      self.pmu.add_event("tile_signal_duplicate")
      if tr is not None:
        tr.instant("TileGroup", "Scheduler:L2", "tile_signal_duplicate",
                   cycle, _signal_args())
      return
    seen.add(signal.task.task_id)
    if seen != state.expected_task_ids:
      return
    state.completed_phases.add(signal.phase)
    phase_seq = state.sequencer or self.sequencer
    phase_ev = state.phase_event_ids[signal.phase]
    if tr is not None:
      tr.instant("TileGroup", "Scheduler:L2", "phase_aggregate", cycle, {
        **_signal_args(),
        "expected": len(state.expected_task_ids),
        "seen": len(seen),
        "event_id": phase_ev,
      })
    phase_seq.notify_event(phase_ev)
    if signal.phase == "input_released":
      self._unpin_grid_inputs(grid, cycle)

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
      seq.notify_event(txn.completion_event)
      seq.note_job_done()
      # forward to active tiles so persistent tile programs can WAIT
      for t in self.tiles:
        if t.uce.has_active_contexts():
          t.uce.notify_event(txn.completion_event)
      # full_memory: record payload using real transaction addresses
      if self.memory_enabled and txn.src is not None and txn.dst is not None:
        from .memory.payload import Payload
        src_addr = txn.src.address
        if self.payload.get(src_addr) is None:
          self.payload.alloc(src_addr, Payload(
            iova=src_addr, bytes_total=txn.bytes_total,
            layout="paged_kv" if txn.op == TransferOp.PREFETCH
                   else "row_major", producer_kind="DMA"))
        self.payload.copy(src_addr, txn.dst.address, txn.bytes_total)
      if tr is not None:
        if trace_slot is None:
          raise RuntimeError(
            f"group transfer {txn.transaction_id} lacks a visual slot"
          )
        direction, visual_slot = trace_slot
        thread, category = self._group_transfer_trace_lane(
          direction, visual_slot)
        owner_args = MemoryTrace._owner_args(txn.issuer)
        context_name = str(owner_args.get("context_name", "ctx"))
        buffer_id = str(owner_args.get("buffer_id", txn.completion_event))
        duration_cycles = max(txn.completed_cycle - txn.start_cycle, 1)
        effective_bandwidth_gbs = round(
          txn.bytes_total / (duration_cycles * self.cfg.cycle_ns()), 3)
        tr.complete(
          "TileGroup", thread, f"{context_name} / {buffer_id}",
          txn.start_cycle, txn.completed_cycle,
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
            "destination_address":
              txn.dst.address if txn.dst is not None else None,
            **({"flow_id": tr.flow_id(txn.transaction_id)}
               if self.memory_trace is not None else {}),
          },
          category=category,
        )
      self.transfer_manager.acknowledge(txn.transaction_id)

    # 1b. tick Collective jobs
    remaining_coll: list[_CollectiveJob] = []
    for cjob in self._collective_jobs:
      if cycle >= cjob.finish_cycle:
        seq = cjob.sequencer or self.sequencer
        seq.notify_event(cjob.event_id)
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
          tr.instant("TileGroup", "Collective", "collective_complete",
                     cycle, {"event_id": cjob.event_id})
      else:
        remaining_coll.append(cjob)
    # 2. tick stream queues (PMU occupancy counters + trace counters)
    for q in self.queues.values():
      q.tick(cycle)
      if tr is not None:
        tr.counter_if_changed("TileGroup", "occupancy", cycle, q.occupancy,
                              "tokens", thread=f"StreamQ:{q.queue_id}")
        tr.counter_if_changed("TileGroup", "credit_available", cycle,
                              q._credit_available, "credits",
                              thread=f"StreamQ:{q.queue_id}")

    # (NoC router steps in section 1, before the transfer manager)
    freeze_new_work = self._reset_freezes_new_work()

    # 3. tick sequencers only while dispatch is not frozen
    if not freeze_new_work:
      for seq in self._active_sequencers:
        seq.step(cycle)
      # PR 3.5: post-sequencer barrier — retry waiting admissions on a
      # capacity change, then activate staged ones (first action issues
      # next cycle; appends never re-enter this loop).
      self._retry_pending_context_admissions(cycle)
    for ticket in self._pending_activations:
      self._activate_admitted_context(ticket, cycle)
    self._pending_activations.clear()
    # 4. running engines still tick; UCE issue/queued launches freeze
    for t in self.tiles:
      t.step(cycle, freeze_new_work=freeze_new_work)
      for term in t.drain_context_terminals():
        if term.status == "fault":
          # Route fault to the sequencer that dispatched this role
          rid = term.role_event_id
          rt = self._role_trace.get(rid) if rid is not None else None
          fault_seq = (rt.sequencer if rt is not None and rt.sequencer is not None
                       else self.sequencer)
          fault_seq.faulted = True
          fault_seq.fault_reason = f"tile{term.tile_id}: {term.reason}"
          fault_seq.done = True
          self.pmu.add_event("tile_fault")
          if (self.runtime_enabled
              and not (self.reset_domain.is_active
                       or self.reset_domain.is_done)):
            self.trigger_fault(
              self._fault_code_for_reason(term.reason),
              tile_id=term.tile_id, cycle=cycle, desc_id=term.reason)
        if term.status != "done" or term.role_event_id is None:
          continue
        # PR 2: release L1 frame + allocations on tile terminal (§5.7).
        # PR 3: L2 grid pins outlive the terminal - only the matching
        # aggregate phase or a gated release may unpin them.
        if self.memory_enabled or self.runtime_enabled:
          frame = t.l1_frames[term.ctx_id]
          frame.release()
          if self.memory_trace is not None and tr is not None:
            tr.instant(f"Tile{t.tile_id}", "Lifecycle", "frame_release",
                       cycle, {"ctx_id": term.ctx_id,
                               "generation": frame.generation,
                               "tile_id": t.tile_id,
                               "reason": "tile_terminal"})
          tile_l1 = self._role_l1_handles.get(term.role_event_id, {})
          for handle in tile_l1.pop(t.tile_id, []):
            try:
              t.l1_allocator.request_release(handle, handle.owner, cycle)
            except MemoryInvariantError:
              pass  # already released or stale - terminal must not fault
        done_set = self._role_done_tiles.setdefault(term.role_event_id, set())
        if t.tile_id in done_set:
          continue
        done_set.add(t.tile_id)
        if tr is not None:
          tr.instant(f"Tile{t.tile_id}", f"UCE CTX{term.ctx_id}", "tile_done",
                     cycle,
                     {
                       "ctx_id": term.ctx_id,
                       "role_id": term.role_id,
                       "event_id": term.role_event_id,
                     })
        rt = self._role_trace.get(term.role_event_id)
        mask = self._role_event_tile_mask.get(term.role_event_id, 0)
        expected = bin(mask).count("1")
        if len(done_set) >= expected:
          # Route completion to the sequencer that dispatched this role
          done_seq = (rt.sequencer if rt is not None and rt.sequencer is not None
                      else self.sequencer)
          done_seq.notify_event(term.role_event_id)
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
              {
                "role_id": term.role_id,
                "event_id": term.role_event_id,
              },
            )


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
    all_done = (len(self._active_sequencers) == 0
                and not self._pending_context_admissions
                and not self._pending_activations)
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
            args={
              "task": self._task_trace_name.replace("task:", "", 1)
            },
          )
        cev = (self.sequencer.task.completion_event
               if self.sequencer.task is not None else "group_task_done")
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
    for grid in grids:
      self._grid_signals.pop(grid, None)
    self._live_launches.pop((s.context_name, s.device_slot, gen), None)
    if not (self.memory_enabled or self.runtime_enabled):
      self._grid_l2_pins = {
        g: pins for g, pins in self._grid_l2_pins.items()
        if g.launch_generation != gen}
      self._l2_roles.pop(gen, None)
      return
    retained = {key: handle for key, handle in self._l2_handles.items()
                if key[0] == gen}
    all_released = all(
      self.l2_sram.is_released(handle) for handle in retained.values())
    pin_residue = any(
      any(slot_pins for slot_pins in pins.values())
      for grid, pins in self._grid_l2_pins.items()
      if grid.context_name == s.context_name
      and grid.launch_generation == gen)
    if not s.faulted and (not all_released or pin_residue):
      s.faulted = True
      s.fault_reason = (
        "launch retirement invariant: unreleased handles or grid pins"
        f" remain for generation {gen}")
      self.pmu.add_event("release_invariant_fault")
      if (self.runtime_enabled
          and not (self.reset_domain.is_active
                   or self.reset_domain.is_done)):
        self.trigger_fault(
          FaultCode.ADDRESS_FAULT, cycle=cycle, desc_id=s.fault_reason)
      # Unwind the residue so the run still terminates with the fault
      # surfaced instead of hanging on dead state.
      self._unwind_grid_l2_pins(cycle)
      for key, handle in retained.items():
        try:
          self.l2_sram.request_release(handle, handle.owner, cycle)
        except MemoryInvariantError:
          pass
        self._l2_handles.pop(key, None)
      self._l2_roles.pop(gen, None)
      return
    for key, handle in retained.items():
      if self.l2_sram.is_released(handle):
        self._l2_handles.pop(key, None)
    self._l2_roles.pop(gen, None)

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
    if ("address" in lowered or "owner" in lowered
        or "generation" in lowered or "release" in lowered):
      return FaultCode.ADDRESS_FAULT
    if "l2 capacity" in lowered:
      return FaultCode.L2_CAPACITY_FAULT
    if "timeout" in lowered or "credit" in lowered:
      return FaultCode.DMA_TIMEOUT
    if "descriptor" in lowered or "transaction id" in lowered:
      return FaultCode.INVALID_DESCRIPTOR
    return FaultCode.ENGINE_INTERNAL_FAULT

  def _aggregate_pmu(self) -> None:
    # merge all active sequencers + all tiles + all queues into group PMU
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
    self.pmu.add_event("memory_transaction_completed",
                       tm.pmu_completed_count)
    self.pmu.add_event("memory_transaction_cancelled",
                       tm.pmu_cancelled_count)
    self.pmu.add_event("memory_transaction_faulted", tm.pmu_faulted_count)
    self.pmu.add_cycle("noc_credit_wait",
                       tm.pmu_noc_credit_wait_cycles)
    self.pmu.add_cycle("dma_queue_wait",
                       tm._global_dma.wait_cycles)
    self.pmu.add_cycle("l2_bank_wait",
                       tm._l2_read.wait_cycles + tm._l2_write.wait_cycles)
    self.pmu.add_cycle(
      "l2_cache_wait",
      tm._l2_cache_lookup.wait_cycles + tm._l2_cache_fill.wait_cycles,
    )
    l1_wait = sum(
      s.wait_cycles for s in list(tm._l1_read.values())
      + list(tm._l1_write.values()))
    self.pmu.add_cycle("l1_bank_wait", l1_wait)
    l1_cache_wait = sum(
      stage.wait_cycles
      for stage in (
        *tm._l1_cache_lookup.values(),
        *tm._l1_cache_fill.values(),
      )
    )
    self.pmu.add_cycle("l1_cache_wait", l1_cache_wait)
    self.pmu.add_cycle("hbm_outstanding_wait",
                       tm._hbm_read.wait_cycles + tm._hbm_write.wait_cycles)
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
      self.pmu.add_cycle("payload_layout_faults",
                         self.payload.layout_fault_count)
      self.payload.layout_fault_count = 0
  # ---- lifecycle ------------------------------------------------------

  def load_task(self, task: ExecTileGroupTask, *,
                input_bindings=None,
                formal_bindings: dict[str, str] | None = None) -> None:
    # reset everything
    for t in self.tiles:
      t.reset()
    self.sequencer.reset()
    self._active_sequencers = [self.sequencer]
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
    self._last_step_cycle = 0
    # A new standalone run is a fresh HBM binding epoch: old handles must
    # become stale and same-name bindings must be registered again.
    self.hbm.reset()
    # runtime-level: clear event/fault tables, but PRESERVE program residency
    if self.runtime_enabled:
      self.event_table.clear()
      self.fault_ring.reset()
      self.reset_domain.reset()
      self.l2_sram.reset()
    if self.memory_enabled:
      self.noc.reset()
      self.payload.reset()
    # PR 2: register global bindings as HBM external handles
    if input_bindings:
      self.register_global_bindings(input_bindings)
    # PR 3.5: launch-scoped formal→actual binding map (standalone identity)
    self.sequencer.formal_bindings = dict(formal_bindings or {})
    self._task_trace_name = f"task:{task.name}"
    self._task_start_cycle = None
    self._task_done_traced = False
    self.pmu.reset()
    max_ctx = self.tiles[0].uce.context_count
    for binding in task.role_bindings.values():
      if binding.context_id is not None and binding.context_id >= max_ctx:
        raise ValueError(
          f"role {binding.role_id} pins context {binding.context_id} but context_count is {max_ctx}"
        )
    # PR 2: L2 admission (plan + commit all l2_buffers)
    outcome = self.try_admit_l2_buffers(
      task,
      context_name=task.name,
      launch_generation=self._context_launch_generation,
      cycle=0)
    if outcome.status is not L2AdmissionStatus.ADMITTED:
      # standalone single-context: any admission failure is a hard fault
      self.sequencer.admission_status = ContextAdmissionStatus.CANCELLED
      self.sequencer.faulted = True
      self.sequencer.fault_reason = "L2 capacity fault during context admission"
      self.sequencer.done = True
      self.pmu.add_event("l2_admission_permanent_fault")
      return
    self.sequencer.context_launch_generation = self._context_launch_generation
    self.sequencer.context_name = task.name
    self.sequencer.device_slot = 0
    self._live_launches[
      (task.name, 0, self._context_launch_generation)] = self.sequencer
    self.sequencer.load(task)
    self.sequencer.admission_status = ContextAdmissionStatus.ACTIVE
    # pre-init streams declared in the task (some tasks init inline)
    for s in task.streams:
      self.init_stream(s)

  def load_context_task(self, task: ExecTileGroupTask,
                        slot_index: int = 0, *,
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

    Dispatch bindings with ``context_id = None`` are auto-assigned to
    UCE context ``slot_index`` so each device slot runs on its own UCE
    context, enabling true concurrency on the shared tiles.  Explicit
    dispatch ``context_id`` pins are preserved (IR_SPEC §3.5); callers
    must ensure they don't conflict across concurrent slots.  Requires
    ``context_count >= device_context_count``.

    PR 3.5: on a transient L2 capacity miss the returned sequencer is
    ``WAIT_CAPACITY`` (not faulted) — the device slot stays reserved
    but no UCE context, L1/L2 allocation, stream, or DMA/engine work is
    held.  The group's release-driven FIFO retry activates it later.
    """
    ticket = self._prepare_context_launch(
      task, slot_index=slot_index,
      context_name=context_name,
      input_bindings=input_bindings,
      formal_bindings=formal_bindings,
      enqueue_cycle=cycle)
    outcome = self._try_admit_prepared_context(ticket, cycle)
    if outcome.status is L2AdmissionStatus.WAIT_CAPACITY:
      # transient capacity miss: reserve the slot, wait for a release
      ticket.sequencer.admission_status = ContextAdmissionStatus.WAIT_CAPACITY
      ticket.sequencer.admission_wait_start_cycle = cycle
      self._enqueue_pending_admission(ticket)
      return ticket.sequencer
    if outcome.status is L2AdmissionStatus.FAULT:
      # permanent/invalid: structured fault, never queued
      seq = ticket.sequencer
      seq.admission_status = ContextAdmissionStatus.CANCELLED
      seq.faulted = True
      seq.fault_reason = (
        "L2 capacity fault during context admission"
        if (outcome.failure is None
            or outcome.failure.kind is AdmissionFailureKind.PERMANENT_CAPACITY)
        else f"invalid context admission request: {outcome.failure.reason}")
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
    max_ctx = self.tiles[0].uce.context_count
    if slot_index >= max_ctx:
      raise ValueError(
        f"device slot {slot_index} requires UCE context {slot_index}"
        f" but context_count is {max_ctx}")
    launch_id = self._next_launch_id
    self._next_launch_id += 1
    # Deep-clone so the caller's task stays pristine: re-submitting the
    # same context re-namespaces from the clean original.
    task = copy.deepcopy(task)
    # Auto-assign unpinned dispatches to the slot's UCE context.
    for binding in task.role_bindings.values():
      if binding.context_id is None:
        binding.context_id = slot_index
      elif binding.context_id >= max_ctx:
        raise ValueError(
          f"role {binding.role_id} pins context {binding.context_id} but context_count is {max_ctx}"
        )
    # Namespace event IDs with (slot, launch) so sequential slot reuse
    # cannot collide with stale completions.
    prefix = f"s{slot_index}l{launch_id}_"
    for action in task.actions:
      if action.dst is not None:
        action.dst = prefix + action.dst
      if action.op == ExecGroupActionOp.WAIT_EVENT:
        action.args = tuple(prefix + a if isinstance(a, str) else a
                            for a in action.args)
      elif action.op == ExecGroupActionOp.SIGNAL_EVENT:
        action.args = tuple(prefix + a if isinstance(a, str) else a
                            for a in action.args)
      elif action.op == ExecGroupActionOp.DISPATCH_ROLE:
        # PR 3: args = (ExecDispatchRequest,); prefix only the event
        # strings, never the grid identity (role_id/ordinal/policy).
        request = action.args[0]
        if isinstance(request, ExecDispatchRequest):
          action.args = (
            dataclasses.replace(
              request,
              input_released_event=(
                prefix + request.input_released_event
                if request.input_released_event else ""),
              output_ready_event=(
                prefix + request.output_ready_event
                if request.output_ready_event else ""),
            ),)
      elif action.op == ExecGroupActionOp.RELEASE_L2:
        # PR 3: args = (ExecReleaseRequest,); prefix dependency events
        # only; slot/role/ordinals are grid-relative, not namespaced.
        request = action.args[0]
        if isinstance(request, ExecReleaseRequest):
          action.args = (
            dataclasses.replace(
              request,
              dependency_events=tuple(
                prefix + ev if isinstance(ev, str) else ev
                for ev in request.dependency_events),
            ),)
    # Namespace stream IDs into the launch's integer queue space.
    qid_offset = (launch_id * 100 + slot_index) * 10000
    owned_qids: set[int] = set()
    for action in task.actions:
      if action.op == ExecGroupActionOp.INIT_STREAM:
        # args = (queue_id, depth, producer_mask, consumer_mask)
        action.args = (
          int(action.args[0]) + qid_offset,
          *action.args[1:],
        )
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
      ExecTileOp.STREAM_PUSH, ExecTileOp.STREAM_POP,
      ExecTileOp.STREAM_ACQUIRE, ExecTileOp.STREAM_RELEASE,
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
          inst.args = (
            int(inst.args[0]) + qid_offset,
            *inst.args[1:],
          )
        elif inst.op in (ExecTileOp.WAIT, ExecTileOp.WAITALL):
          # Namespace external group-DMA waits so the tile sees the
          # forwarded namespaced DMA completion (TileUCE matches by
          # exact id against _external_events_done).
          inst.args = tuple(
            prefix + a if isinstance(a, str) and "ev_dma_" in a else a
            for a in inst.args)
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

  def _try_admit_prepared_context(
    self, ticket: _PendingContextAdmission, cycle: int,
  ) -> L2AdmissionOutcome:
    """Admission only: plan+commit the prepared context's L2 bundle."""
    return self.try_admit_l2_buffers(
      ticket.task,
      context_name=ticket.context_name,
      launch_generation=ticket.launch_generation,
      cycle=cycle,
    )

  def reset(self) -> None:
    for t in self.tiles:
      t.reset()
    self.sequencer.reset()
    self._active_sequencers = [self.sequencer]
    self._next_launch_id = 0
    self._active_sequencers = [self.sequencer]
    for q in self.queues.values():
      q.reset()
    self._collective_jobs.clear()
    self._role_event_tile_mask.clear()
    self._role_done_tiles.clear()
    self._role_trace.clear()
    self.pmu.reset()
    # PR 3.5: staged activations already own L2; unwind them before
    # clearing the generation-keyed handle/role registries.
    self._cancel_pending_admissions(release_staged=True)
    # PR 3: clear structured grid registries
    self._grid_signals.clear()
    self._live_launches.clear()
    self._grid_l2_pins.clear()
    self._l2_roles.clear()
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
    self._last_step_cycle = 0
    for t in self.tiles:
      t.l1_allocator.reset()
    if self.runtime_enabled:
      self.event_table.clear()
      self.fault_ring.reset()
      self.program_table.reset()
      self.reset_domain.reset()
      self.l2_sram.reset()
    if self.memory_enabled:
      self.noc.reset()
      self.payload.reset()

  # ---- inspection -----------------------------------------------------

  def snapshot(self) -> dict:
    return {
      "task_done": self.sequencer.done,
      "task_action_index": self.sequencer.action_index,
      "queues": {
        qid: q.snapshot() for qid, q in self.queues.items()
      },
      "tiles": [t.snapshot() for t in self.tiles],
      "memory": {
        "fidelity": self.fidelity,
        "hbm": self.hbm.snapshot() if self.runtime_enabled else None,
        "l2": self.l2_sram.snapshot() if self.runtime_enabled else None,
        "l1": {
          tile.tile_id: {
            "allocator": (
              tile.l1_allocator.snapshot()
              if self.runtime_enabled
              else None
            ),
            "frames": [frame.snapshot() for frame in tile.l1_frames],
          }
          for tile in self.tiles
        },
        "cache": {
          "l2": self.l2_cache.snapshot(),
          "l1": {
            tile.tile_id: tile.l1_cache.snapshot()
            for tile in self.tiles
          },
        },
        "mshr": {
          "l2": self.l2_mshr.snapshot(),
          "l1": {
            tile.tile_id: tile.l1_mshr.snapshot()
            for tile in self.tiles
          },
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
        }
        for ticket in self._pending_context_admissions
      ],
    }

  def all_tiles_done(self) -> bool:
    return all(t.done for t in self.tiles)

  def credit_invariants_hold(self) -> bool:
    return all(q.credit_invariant_holds() for q in self.queues.values())

  def _make_event_callback(self):
    """Create a callback that signals the group EventTable on UCE event
    completion (runtime fidelity, P0-4).  Closes over self.event_table."""
    et = self.event_table

    def _cb(event_id: str) -> None:
      et.signal(event_id, EventStatus.DONE, producer_id=-1, cycle=0)

    return _cb

  # ---- fault / reset (runtime fidelity) -------------------------------

  def trigger_fault(self, code, tile_id: int = -1, cycle: int = 0,
                     desc_id: str = "") -> int:
    """Inject a fault: write a FaultRecord and begin the reset/drain FSM
    (Driver-Firmware 3.3/3.4).  Returns the fault_record_index, or -1
    in timing_only fidelity (no-op).
    """
    if not self.runtime_enabled:
      return -1
    rec = FaultRecord(code=code, tile_id=tile_id,
                      desc_id=zlib.crc32(desc_id.encode()) & 0xFFFFFFFF)
    idx = self.fault_ring.write(rec)
    domain = (FaultDomain.TILE if tile_id >= 0 else FaultDomain.GROUP)
    req = ResetRequest(domain=domain, tile_id=tile_id, fault_record=rec)
    self.reset_domain.begin(req, cycle)
    self.pmu.add_event("fault_record", 1)
    return idx
