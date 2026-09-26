"""Tile Group: 1 Tile Group Sequencer + 4 Compute Tiles + Group SRAM + streams.

The Tile Group is the local data-reuse / synchronization unit
(design/elenor_tile_group/).  It owns the Stream Queues that connect
task roles and the Group DMA.  The simulator drives it cycle by cycle,
advancing the Tile Group Sequencer and every Compute Tile in lockstep.
"""

from __future__ import annotations

import zlib
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from .compiled_program import CompiledProgram
from .config import GroupSchedulerConfig, HardwareConfig
from .execution_ir import (
  ContextAdmissionStatus,
  ExecDispatchRequest,
  ExecGroupActionOp,
  ExecL2Buffer,
  ExecMemoryView,
  ExecModel,
  ExecPublishRequest,
  ExecReleaseRequest,
  ExecSharedInput,
  ExecSignalPolicy,
  ExecStreamDesc,
  ExecTileGroupTask,
  ExecTileRoleBinding,
  GlobalBinding,
  GridInstanceId,
  PhaseSignal,
  TaskIdentity,
)
from .group_scheduler import GroupScheduler, IssueResult, IssueStatus
from .memory import (
  AdmissionFailure,
  AdmissionFailureKind,
  AdmissionWaitReason,
  AllocationHandle,
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
from .memory.arena import ArenaHandle, ArenaPool, RootInvocation
from .memory.profile_controller import ProfileController
from .pmu import PMUCounter
from .profiles import SourceRef, build_registry
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
from .runtime.relocation import relocate_task
from .runtime.reset_domain import ResetState
from .stream_queue import EOSPolicy, QueueKind, StreamQueue
from .tile import ComputeTile, TileAdmission, _UCETerminalEvent
from .tile_group_sequencer import TileGroupSequencer
from .trace import MemoryTrace, Tracer

if TYPE_CHECKING:
  from .runtime.group_port import _RootRequestRecord


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


@dataclass
class _GridRoute:
  grid: GridInstanceId
  binding: ExecTileRoleBinding
  request: ExecDispatchRequest
  event_id: str
  sequencer: TileGroupSequencer
  expected: dict[int, TaskIdentity]
  ready_seq: int
  admissions: dict[int, TileAdmission] = field(default_factory=dict)
  retired: set[int] = field(default_factory=set)
  wait_reasons: dict[int, str] = field(default_factory=dict)
  source_ref: SourceRef | None = None


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
    byte_store=None,
  ):
    self.cfg = cfg
    self.tracer = tracer
    self.scheduler_config = scheduler_config or GroupSchedulerConfig()
    self.fidelity = fidelity
    rt = fidelity in ("runtime", "full_memory")
    mem = fidelity == "full_memory"
    self.runtime_enabled = rt
    self.memory_enabled = mem
    self.registry = build_registry(cfg)
    self.loaded_program: CompiledProgram | None = None
    self.run_generation = 0
    self.byte_store = byte_store
    self._pending_root_requests: dict[int, _RootRequestRecord] = {}
    self._grid_routes: dict[GridInstanceId, _GridRoute] = {}
    self._task_leases: dict[tuple[str, int, int], set[TaskIdentity]] = {}
    self._route_sequence = 0
    self._l2_arenas: dict[int, ArenaHandle] = {}
    self.last_admission_wait: AdmissionWaitReason | None = None
    self._retiring_tasks: dict[TaskIdentity, _UCETerminalEvent] = {}
    self._current_bindings: dict[str, GlobalBinding] = {}
    # PR 5: memory lanes/counters/flows + report peaks are opt-in so a
    # plain --trace-json run still emits the pre-PR5 control-flow trace.
    self.memory_trace = MemoryTrace(tracer) if tracer is not None and memory_trace else None
    # PR 2: transfer manager must exist before tiles are created (injected
    # into ComputeTile/MFEEngine as a shared instance).
    from .memory.hbm_region import HBMRegion
    from .memory.transfer import TransferManager

    # NoC fabric must exist before the transfer manager (NoC legs are
    # router-backed: flit enqueue/traversal/credit via NoCRouter)
    self.noc = NoCRouter(
      vc_depth=cfg.noc_vc_depth,
      router_latency_cycles=cfg.noc_router_latency_cycles,
      trace=self.memory_trace,
    )
    self.transfer_manager = TransferManager(
      cfg,
      full_memory=mem,
      noc=self.noc if mem else None,
      trace=self.memory_trace,
      byte_store=byte_store,
      generation_validator=self.validate_transaction_generation,
      reference_acquire=self._acquire_l2_transaction_references,
      reference_release=self._release_l2_transaction_references,
    )
    l2_profile = self.registry.profile("l2", cfg.memory_target.l2.reset_mode)
    self.l2_cache = DeterministicLRUCache(
      l2_profile.cache_bytes,
      cfg.cache_line_bytes,
      write_policy=l2_profile.cache_write_policy,
      level="l2",
      pool_id=0,
    )
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
      byte_store=byte_store,
    )
    # PR 2 admission state: launch generation, global/L2 handles, pins
    self._context_launch_generation: int = 0
    self._global_handles: dict[str, AllocationHandle] = {}  # binding name -> handle
    self._l2_handles: dict[tuple[int, str], AllocationHandle] = {}  # (gen, slot) -> handle
    # PR 3: launch generation -> {buffer_slot -> alloc role}
    self._l2_roles: dict[int, dict[str, str]] = {}
    self._l2_sharing: dict[int, dict[str, str]] = {}
    self._l2_published: set[tuple[int, str]] = set()
    self._l2_released: set[tuple[int, str]] = set()
    self._l2_manifest: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    self._l2_claim_export: dict[tuple[str, str], tuple[str, str]] = {}
    self._l2_export_buffers: dict[tuple[str, str], ExecL2Buffer] = {}
    self._l2_backing_by_export: dict[tuple[str, str], str] = {}
    self._l2_submit_successors: dict[str, frozenset[str]] = {}
    # Protocol-valid L2 objects, distinct from allocated/reserved capacity.
    self._protocol_live_l2: set[tuple[int, str]] = set()
    self._protocol_live_backing_bytes: dict[str, int] = {}
    self._l2_reserved_bytes = 0
    self._l2_live_bytes = 0
    self._l2_reserved_bytes_peak = 0
    self._l2_live_bytes_peak = 0
    # PR 3.5: L2 admission wait queue + release-driven FIFO retry
    self._pending_context_admissions: deque[_PendingContextAdmission] = deque()
    self._pending_activations: list[_PendingContextAdmission] = []
    self._l2_capacity_change_cycle: int | None = None
    # Set only when post-cap/fault isolation cannot be proven; cleared solely
    # by an explicit quiescent reset() (plan/01 §4.4).
    self.poisoned_reason: str | None = None
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
    self.fault_ring = FaultRing()
    self.program_table = ProgramResidencyManager(cfg)
    self.reset_domain = ResetDomain(cfg)
    self.l2_sram = ArenaPool(l2_profile, trace=self.memory_trace)
    if self.memory_enabled:
      self.payload = PayloadTracker()
    self.profile_controller = ProfileController(self, trace=self.memory_trace)
    self.profile_controller.initialize(self.registry, 0)
    self.transfer_manager.configure_profile_generations(
      {**{("l1", tile.tile_id): 0 for tile in self.tiles}, ("l2", 0): 0}
    )
    if byte_store is not None:
      byte_store.register_cache("l2", 0, self.l2_cache)
      for tile in self.tiles:
        byte_store.register_cache("l1", tile.tile_id, tile.l1_cache)

  def _acquire_l2_transaction_references(self, transaction, cycle: int) -> None:
    """Atomically register one accepted transaction per L2 endpoint view.

    Deduplicates views by allocation identity; each registration also
    reaches the physical backing record.  Any failure rolls back every
    registration performed here so a rejected transaction holds nothing.
    """
    handles: list[AllocationHandle] = []
    seen: set[str] = set()
    for view, permission in ((transaction.src, "r"), (transaction.dst, "w")):
      if view is None or view.handle.memory_space != "l2":
        continue
      self.l2_sram.assert_access(view.handle, permission)
      if view.handle.allocation_id not in seen:
        seen.add(view.handle.allocation_id)
        handles.append(view.handle)
    registered: list[AllocationHandle] = []
    try:
      for handle in handles:
        self.l2_sram.begin_inflight(handle, transaction.transaction_id)
        registered.append(handle)
    except Exception:
      self._rollback_l2_references(registered, transaction.transaction_id, cycle)
      raise

  def _rollback_l2_references(
    self, handles: list[AllocationHandle], transaction_id: str, cycle: int
  ) -> None:
    for handle in handles:
      if not self.l2_sram.is_released(handle):
        self.l2_sram.end_inflight(handle, transaction_id, cycle)
    self._sync_l2_pool_mirror(cycle)

  def _release_l2_transaction_references(self, transaction, cycle: int) -> None:
    """Release every L2 view reference at terminal acknowledgement."""
    seen: set[str] = set()
    for view in (transaction.src, transaction.dst):
      if view is None or view.handle.memory_space != "l2":
        continue
      handle = view.handle
      if handle.allocation_id in seen:
        continue
      seen.add(handle.allocation_id)
      if not self.l2_sram.is_released(handle):
        self.l2_sram.end_inflight(handle, transaction.transaction_id, cycle)
    self._sync_l2_pool_mirror(cycle)

  def validate_transaction_generation(self, transaction, phase: str) -> bool:
    """Check the actual touching domains and only the endpoint used in this leg."""
    if transaction.run_generation != self.run_generation:
      return False
    identities = {(level, pool): generation for level, pool, generation in transaction.profile_generations}
    if len(identities) != len(transaction.profile_generations):
      return False
    for (level, pool), generation in identities.items():
      if level not in ("l1", "l2") or generation != self.profile_controller.generations[level]:
        return False
      if (level == "l2" and pool != 0) or (level == "l1" and not 0 <= pool < self.cfg.num_tiles):
        return False
    view = (
      transaction.src
      if phase.startswith("source_")
      else transaction.dst
      if phase.startswith("destination_")
      else None
    )
    if view is None:
      return True
    handle = view.handle
    try:
      if handle.memory_space == "hbm":
        self.hbm.assert_live(handle)
        permission = "r" if phase.startswith("source_") else "w"
        if permission not in self.hbm.permissions(handle.owner.binding_name):
          return False
      else:
        pool = transaction.tile_id if handle.memory_space == "l1" else 0
        if (handle.memory_space, pool) not in identities:
          return False
        allocator = self.tiles[pool].l1_allocator if handle.memory_space == "l1" else self.l2_sram
        allocator.assert_live(handle)
        if handle.memory_space == "l2":
          self.l2_sram.assert_access(handle, "r" if phase.startswith("source_") else "w")
        if handle.profile_generation != identities[(handle.memory_space, pool)]:
          return False
    except (MemoryInvariantError, KeyError, IndexError):
      return False
    return True

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

  def _sync_l2_pool_mirror(self, cycle: int) -> int:
    """Apply physical backing final-free events to occupancy accounting.

    Each event is a real L2 free-map mutation: reserved bytes drop by the
    backing's padded bytes and the capacity-change cycle moves to the release
    cycle so same-profile FIFO admission retries see the new pool version.
    """
    drained = 0
    for event in self.l2_sram.drain_l2_backing_release_events():
      drained += event["padded_bytes"]
      self._l2_live_bytes -= self._protocol_live_backing_bytes.pop(event["backing_id"], 0)
      self._l2_reserved_bytes -= event["padded_bytes"]
      self._l2_capacity_change_cycle = event["release_cycle"]
      self._record_l2_occupancy(event["release_cycle"])
    return drained

  def assert_l2_closed(self) -> None:
    """Prove terminal physical, logical and transfer closure for this run."""
    snapshot = self.l2_sram.snapshot()
    if (
      snapshot["live_backings"]
      or snapshot["live_arenas"]
      or snapshot["live_views"]
      or snapshot["arena_reserved_bytes"]
      or snapshot["pending_shared_claims"]
      or snapshot["active_shared_references"]
      or snapshot["pending_release"]
      or snapshot["pin_count"]
      or snapshot["inflight_count"]
      or self.transfer_manager.outstanding_transactions
    ):
      raise MemoryInvariantError(f"live L2 objects remain: {'; '.join(self.unclosed_l2_objects())}")
    profile = self.l2_sram.profile
    for bank in snapshot["per_bank_occupancy"]:
      if bank["allocated_bytes"] + bank["free_bytes"] != profile.user_spm_per_bank:
        raise MemoryInvariantError(
          f"bank {bank['bank_id']} conservation failed:"
          f" {bank['allocated_bytes']} allocated + {bank['free_bytes']} free"
          f" != {profile.user_spm_per_bank} user SPM bytes"
        )
    self.l2_sram.close_l2_claims()

  def unclosed_l2_objects(self) -> list[str]:
    """Concrete IDs of each retained backing, claim, view, pin and transfer."""
    details: list[str] = []
    snapshot = self.l2_sram.snapshot()
    if snapshot["live_backings"]:
      details.append(f"live_backings={snapshot['live_backings']}")
    if snapshot["live_arenas"]:
      details.append(f"live_arenas={snapshot['live_arenas']}")
    if snapshot["live_views"]:
      details.append(f"live_views={snapshot['live_views']}")
    if snapshot["arena_reserved_bytes"]:
      details.append(f"reserved_bytes={snapshot['arena_reserved_bytes']}")
    if snapshot["pin_count"]:
      details.append(f"pins={snapshot['pin_count']}")
    if snapshot["inflight_count"]:
      details.append(f"inflight={snapshot['inflight_count']}")
    outstanding = self.transfer_manager.outstanding_transactions
    if outstanding:
      details.append(f"unacknowledged={sorted(outstanding)[:8]}")
    if snapshot["live_backings"]:
      details.append(f"live_backing_ids={self.l2_sram.live_backing_ids()}")
    for claim in self.l2_sram.claim_snapshot():
      if claim["state"] in ("DECLARED", "BOUND"):
        details.append(f"claim={claim['claim_id']}:{claim['state']}:{claim['backing_id']}")
    return details

  def poison(self, reason: str) -> None:
    """Mark this TileGroup unusable for further launches (plan/01 §4.4)."""
    self.poisoned_reason = reason

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
    bytes_total = transfer.bytes
    if bytes_total <= 0:
      raise MemoryInvariantError("compiled transfer must have positive bytes")
    src_view = self._resolve_view(transfer.src, gen, formals)
    dst_view = self._resolve_view(transfer.dst, gen, formals)
    if src_view is None or dst_view is None:
      return False
    txn = MemoryTransaction(
      transaction_id=txn_id,
      op=(TransferOp.PREFETCH if op == "dma.prefetch" else TransferOp.GLOBAL_STORE),
      issuer=ContextBufferOwner(context_name, gen, buffer_id),
      src=src_view,
      run_generation=self.run_generation,
      profile_generations=self.transfer_manager.transaction_identity((("l2", 0),))[1],
      dst=dst_view,
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
    self, view: ExecMemoryView | None, gen: int | None = None, formal_bindings: dict[str, str] | None = None
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
        permissions=self.hbm.permissions(name),
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
      segs = self.l2_sram.resolve_segments(handle, offset, view.bytes)
      return ResolvedMemoryView(
        handle=handle, offset_bytes=offset, size_bytes=view.bytes,
        permissions=self.l2_sram.permissions(handle), address=segs[0].address, segments=segs
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

  def bind_l2_view(
    self, buffer_id: str, layout_index: int, sequencer: TileGroupSequencer, cycle: int
  ) -> None:
    arena = self._l2_arenas[sequencer.context_launch_generation]
    if not 0 <= layout_index < len(arena.layout.buffer_layouts):
      raise MemoryInvariantError("L2 bind layout index out of range")
    if arena.layout.buffer_layouts[layout_index].buffer_id != buffer_id:
      raise MemoryInvariantError("L2 bind differs from compiled buffer identity")
    handle = self.l2_sram.bind_view(arena, buffer_id, cycle)
    self._l2_handles[(sequencer.context_launch_generation, buffer_id)] = handle

  def bind_l2_import(self, slot: str, index: int, sequencer: TileGroupSequencer) -> None:
    """Check the borrower handle installed at admission, without acquiring it twice."""
    task = sequencer.task
    if task is None or not 0 <= index < len(task.shared_inputs):
      raise MemoryInvariantError("L2 import bind index is out of range")
    shared = task.shared_inputs[index]
    if shared.slot != slot:
      raise MemoryInvariantError("L2 import bind differs from compiled shared input")
    gen = sequencer.context_launch_generation
    handle = self._l2_handles.get((gen, slot))
    if handle is None or handle.backing_id != self._l2_backing_by_export.get(
      (shared.producer_binding_id, shared.producer_slot)
    ):
      raise MemoryInvariantError("L2 import bind lacks its admitted backing")
    self.l2_sram.assert_live(handle, ContextBufferOwner(sequencer.context_name, gen, slot))
    self.l2_sram.assert_access(handle, "r")

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

  def publish_l2(self, request: ExecPublishRequest, sequencer: TileGroupSequencer, cycle: int) -> bool:
    """Seal a producer view only after its local reader/writer accesses drain."""
    gen = sequencer.context_launch_generation
    slot = request.buffer_slot
    if self._l2_sharing.get(gen, {}).get(slot) != "readonly":
      raise MemoryInvariantError(f"publish requires a local readonly export '{slot}'")
    if (gen, slot) in self._l2_published:
      raise MemoryInvariantError(f"duplicate publish of slot '{slot}'")
    role = self._l2_roles.get(gen, {}).get(slot)
    if role is None:
      raise MemoryInvariantError(f"publish references missing producer slot '{slot}'")
    handle = self._l2_handles.get((gen, slot))
    if handle is None:
      raise MemoryInvariantError(f"publish references unbound producer slot '{slot}'")
    self.l2_sram.assert_access(handle, "w")
    release_like = ExecReleaseRequest(
      slot, role, request.reader_dispatch_ordinals, request.writer_dispatch_ordinals,
      request.dependency_events,
    )
    return self._close_l2_access(release_like, sequencer, cycle, publish=True)

  def release_l2(self, request: ExecReleaseRequest, sequencer: TileGroupSequencer, cycle: int) -> bool:
    """Forfeit exactly one owner's view after a complete read/write access preflight."""
    return self._close_l2_access(request, sequencer, cycle, publish=False)

  def _close_l2_access(
    self, request: ExecReleaseRequest, sequencer: TileGroupSequencer, cycle: int, *, publish: bool
  ) -> bool:
    """Use the same pinned-access and transfer preflight for publish and release."""
    gen = sequencer.context_launch_generation
    slot = request.buffer_slot
    action_name = "publish" if publish else "release"
    launch = self._live_launches.get((sequencer.context_name, sequencer.device_slot, gen))
    if launch is not sequencer:
      raise MemoryInvariantError(f"{action_name} references inactive or foreign launch generation {gen}")
    roles = self._l2_roles.get(gen)
    if roles is None or slot not in roles:
      if not publish and (gen, slot) in self._l2_released:
        raise MemoryInvariantError(f"double release of L2 slot '{slot}'")
      raise MemoryInvariantError(
        f"{action_name} references unknown or released L2 buffer '{slot}' in launch generation {gen}"
      )
    declared_role = roles[slot]
    if declared_role != request.buffer_role:
      raise MemoryInvariantError(
        f"release role mismatch on slot '{slot}':"
        f" alloc '{declared_role}' != request '{request.buffer_role}'"
      )
    if not publish and self._l2_sharing.get(gen, {}).get(slot) == "readonly":
      if (gen, slot) not in self._l2_published:
        raise MemoryInvariantError(f"release of readonly export '{slot}' precedes publish")

    handle: AllocationHandle | None = None
    handle = self._l2_handles.get((gen, slot))
    if handle is None:
      raise MemoryInvariantError(f"release references missing physical L2 buffer '{slot}'")
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

      if publish:
        self.l2_sram.check_publish_l2(
          handle, allowed_pins=tuple(pin.consumer_id for _, _, _, pin in writer_pins)
        )
      if self.transfer_manager.has_inflight_access(handle):
        raise MemoryInvariantError(f"release of slot '{slot}' has an in-flight transfer")
      if self.l2_sram.has_inflight_references(handle):
        # Physical reference ledger: a terminal-but-unacknowledged transfer
        # still holds this view and only an acknowledgement can drop it.
        # Raising here keeps the whole release read-only (plan/01 §2.4).
        raise MemoryInvariantError(
          f"release of slot '{slot}' is blocked by an unacknowledged transfer reference"
        )

    for grid, task_id, slot_pins, pin in writer_pins:
      self.l2_sram.unpin(pin.handle, pin.consumer_id, cycle)
      slot_pins.pop(slot)
      task_pins = self._grid_l2_pins[grid]
      if not slot_pins:
        task_pins.pop(task_id)
      if not task_pins:
        self._grid_l2_pins.pop(grid)

    if publish:
      self.l2_sram.publish_l2(handle, cycle)
      self._l2_published.add((gen, slot))
      return True
    if handle is not None:
      expected_owner = ContextBufferOwner(sequencer.context_name, gen, slot)
      freed = self.l2_sram.invalidate_view(handle, expected_owner, cycle)
      if not freed:
        raise MemoryInvariantError(f"release of slot '{slot}' left pinned consumers")
      self._sync_l2_pool_mirror(cycle)

    roles.pop(slot)
    self._l2_released.add((gen, slot))
    self._protocol_live_l2.discard((gen, slot))
    return True

  # ---- L2 admission wait queue (PR 3.5) -----------------------------

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

  def _pin_grid_l2(
    self, grid: GridInstanceId, binding: ExecTileRoleBinding, task: TaskIdentity, cycle: int
  ) -> None:
    """Pin each accessed L2 allocation once for one logical task."""
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
        self.l2_sram.assert_access(handle, "w" if slot in writes else "r")
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

  def release_context_memory(self, cycle: int) -> bool:
    """Retire cancelled owners only after real transfer and member isolation."""
    for sequencer in self._active_sequencers:
      self.scheduler.cancel_context(sequencer, cycle)
    self.transfer_manager.cancel_all(cycle)
    self.profile_controller.request_cancel_all(cycle)
    if (
      self.profile_controller.cancellation_pending
      or self.transfer_manager.inflight_count
      or self.transfer_manager.cancellation_pending
    ):
      return False
    # plan/01 §4.1: terminal transfers release their view/backing references
    # at acknowledgement BEFORE view invalidation; the reverse order deadlocks
    # release-pending views on references only an ack can drop.  The drain
    # gate above plus engine accepted_count==0 guarantee no owner job will
    # re-acknowledge these records.
    self.transfer_manager.acknowledge_all_terminals(cycle)
    self._unwind_grid_l2_pins(cycle)
    for route in tuple(self._grid_routes.values()):
      for tile_id, admission in tuple(route.admissions.items()):
        tile = self.tiles[tile_id]
        for name, handle in tuple(admission.l1_handles.items()):
          if not tile.l1_allocator.is_released(handle):
            if not tile.l1_allocator.invalidate_view(handle, handle.owner, cycle):
              return False
          del admission.l1_handles[name]
        if admission.arena is not None:
          if not tile.l1_allocator.retire_arena(admission.arena, cycle):
            return False
          admission.arena = None
        tile.l1_frames[admission.context_id].release()
        task = route.expected[tile_id]
        lease_key = (route.request.binding_id, task.grid.launch_generation, tile_id)
        leases = self._task_leases.get(lease_key)
        if leases is not None:
          leases.remove(task)
          if not leases:
            del self._task_leases[lease_key]
          self._trace_task_lease("task_lease_release", route, task, tile_id, cycle)
      del self._grid_routes[route.grid]
    self._retiring_tasks.clear()
    for key, handle in tuple(self._l2_handles.items()):
      if not self.l2_sram.is_released(handle):
        if not self.l2_sram.cancel_l2_view(handle, handle.owner, cycle):
          return False
      del self._l2_handles[key]
    self._protocol_live_l2.clear()
    self.l2_sram.cancel_all_l2_claims(cycle)
    self._sync_l2_pool_mirror(cycle)
    self._record_l2_occupancy(cycle)
    for generation, arena in tuple(self._l2_arenas.items()):
      held_before = self.l2_sram.snapshot()["arena_reserved_bytes"]
      if not self.l2_sram.retire_arena(arena, cycle):
        return False
      residual = self._sync_l2_pool_mirror(cycle)
      held_after = self.l2_sram.snapshot()["arena_reserved_bytes"]
      del self._l2_arenas[generation]
      self._l2_reserved_bytes -= held_before - held_after - residual
    self._record_l2_occupancy(cycle)
    self._txn_sequencer.clear()
    self.scheduler.abort_inflight(cycle)
    for sequencer in self._active_sequencers:
      sequencer.mark_fault("group reset")
      sequencer.abort_actions()
    self._clear_group_transfer_trace_slots()
    self._collective_jobs.clear()
    self._role_l1_handles.clear()
    self._protocol_live_l2.clear()
    self._grid_signals.clear()
    self._live_launches.clear()
    self._l2_roles.clear()
    for tile in self.tiles:
      tile.reset()
    self.transfer_manager.acknowledge_all_terminals(cycle)
    self.assert_l2_closed()
    return True

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
    source_ref: SourceRef | None = None,
  ) -> IssueResult:
    """Register one bounded Grid Route, without reserving any Tile resources."""
    seq = sequencer or self.sequencer
    event = event_id or ""
    program = binding.tile_program
    grid = seq.grid_id(request.dispatch_ordinal)
    if len(self._grid_routes) >= self.scheduler_config.dispatch_capacity:
      return IssueResult(IssueStatus.BACKPRESSURE, reason="WAIT_CONTROL_RESOURCE")
    if not event or binding.task_domain is None or seq.task is None:
      return IssueResult(IssueStatus.FAULT, reason="dispatch lacks compiled Task domain/event")
    selected = [tile.tile_id for tile in self.tiles if binding.tile_mask & (1 << tile.tile_id)]
    count = binding.task_domain.to_task - binding.task_domain.from_task
    if count <= 0 or count != len(selected) or grid in self._grid_routes:
      return IssueResult(IssueStatus.FAULT, reason="invalid or duplicate Grid Route")
    if (
      program.program_id <= 0
      or program.program_hash <= 0
      or program.text_bytes <= 0
      or program.layout is None
      or program.resource_contract is None
      or request.requested_l1_mode not in program.resource_contract.allowed_profiles
      or request.resolved_l1_mode != self.profile_controller.active_modes["l1"]
    ):
      return IssueResult(IssueStatus.FAULT, reason="invalid compiled program identity or L1 profile")
    if self.profile_controller.issue_gate_closed("l1"):
      return IssueResult(IssueStatus.BACKPRESSURE, reason="profile L1 issue gate")
    l2_formals = [formal for formal in program.formals if formal.space == "l2"]
    if len(binding.actuals) != len(l2_formals):
      return IssueResult(IssueStatus.FAULT, reason="L2 actual/formal mismatch")
    roles = self._l2_roles.get(seq.context_launch_generation, {})
    if any(slot not in roles for slot in binding.actuals):
      return IssueResult(IssueStatus.FAULT, reason="unknown or released L2 actual")
    if any(roles[slot] == "in" for slot in binding.write_actuals):
      return IssueResult(IssueStatus.FAULT, reason="write to input-only L2 actual")
    expected = {
      tile_id: TaskIdentity(grid, binding.task_domain.from_task + index)
      for index, tile_id in enumerate(selected)
    }
    phases = {}
    if request.input_released_event:
      phases["input_released"] = request.input_released_event
    if request.output_ready_event:
      phases["output_ready"] = request.output_ready_event
    self._grid_signals[grid] = _GridSignalState(
      grid,
      frozenset(task.task_id for task in expected.values()),
      request.signal_policy,
      phases,
      sequencer=seq,
    )
    self._grid_routes[grid] = _GridRoute(
      grid, binding, request, event, seq, expected, self._route_sequence, source_ref=source_ref
    )
    self._route_sequence += 1
    self._role_event_tile_mask[event] = binding.tile_mask
    self._role_done_tiles[event] = set()
    self._role_l1_handles[event] = {}
    self._role_trace[event] = _RoleTrace(
      binding.role_id, event, cycle, binding.tile_mask, binding.out_stream, binding.in_stream, seq
    )
    return IssueResult(IssueStatus.ACCEPTED, asynchronous=True, completion_event=event, adapter="dispatch")

  def _step_task_admission(self, cycle: int) -> None:
    """One commit per Tile per Tick; compare only SAME/COMPATIBLE bucket heads."""
    if self.profile_controller.issue_gate_closed("l1"):
      return
    current = self.profile_controller.active_modes["l1"]
    for tile in self.tiles:
      heads: dict[bool, _GridRoute] = {}
      for queued_route in self._grid_routes.values():
        if (
          queued_route.sequencer.faulted
          or tile.tile_id not in queued_route.expected
          or tile.tile_id in queued_route.admissions
          or tile.tile_id in queued_route.retired
        ):
          continue
        same = queued_route.request.requested_l1_mode == current
        heads.setdefault(same, queued_route)
      for same in (True, False):
        route = heads.get(same)
        if route is None:
          continue
        seq = route.sequencer
        assert seq.task is not None and seq.task.resource_contract is not None
        task = route.expected[tile.tile_id]
        key = (seq.task.binding_id, seq.context_launch_generation, tile.tile_id)
        leases = self._task_leases.get(key, set())
        if len(leases) >= seq.task.resource_contract.requested_contexts_per_tile:
          route.wait_reasons[tile.tile_id] = "WAIT_CONTEXT_LIMIT"
          continue
        candidate = tile.plan_admission(
          route.binding.tile_program, route.grid, route.event_id, task.task_id, route.binding.context_id
        )
        if candidate is None:
          route.wait_reasons[tile.tile_id] = "WAIT_SLOT"
          continue
        if isinstance(candidate, AdmissionFailure):
          if candidate.kind is AdmissionFailureKind.TEMPORARY_CAPACITY:
            assert candidate.wait_reason is not None
            route.wait_reasons[tile.tile_id] = candidate.wait_reason.value
            continue
          seq.mark_fault(candidate.reason)
          self.on_scheduler_fault(seq, candidate.reason, cycle)
          break
        try:
          self._commit_route_task(route, candidate, cycle)
        except (MemoryInvariantError, ValueError) as exc:
          seq.mark_fault(f"Task admission failed: {exc}")
          self.on_scheduler_fault(seq, seq.fault_reason, cycle)
          break
        self._task_leases.setdefault(key, set()).add(task)
        route.admissions[tile.tile_id] = candidate
        route.wait_reasons.pop(tile.tile_id, None)
        self._trace_task_lease("task_lease_acquire", route, task, tile.tile_id, cycle)
        break

  def _commit_route_task(self, route: _GridRoute, admission: TileAdmission, cycle: int) -> None:
    from .tile import _TileContextMemory

    binding, seq, tile = route.binding, route.sequencer, admission.tile
    program = binding.tile_program
    task = route.expected[tile.tile_id]
    generation = seq.context_launch_generation
    l2_map = {
      index: self._l2_handles[(generation, slot)]
      for (index, _formal), slot in zip(
        ((i, f) for i, f in enumerate(program.formals) if f.space == "l2"), binding.actuals
      )
    }
    global_map = {}
    global_formals = [(i, formal) for i, formal in enumerate(program.formals) if formal.space == "global"]
    if len(global_formals) != len(binding.global_actuals):
      raise MemoryInvariantError("global actual/formal count mismatch")
    for (index, _formal), actual in zip(global_formals, binding.global_actuals):
      view = self._resolve_view(actual, generation, seq.formal_bindings)
      if view is None:
        raise MemoryInvariantError("missing global actual view")
      global_map[index] = view
    tile.commit_admission(admission, program, cycle)
    pinned = False
    try:
      self._pin_grid_l2(route.grid, binding, task, cycle)
      pinned = True
      self.program_table.register(
        program.program_id, program.version, program.program_hash, 0, program.text_bytes
      )
      admission.prepare_cycles = (
        self.program_table.ensure_resident(program.program_id, tile.tile_id, cycle)
        if self.runtime_enabled
        else 0
      )
      memory = _TileContextMemory(
        task_identity=task,
        l2_formal_handles=l2_map,
        global_formal_views=global_map,
        l1_handles=admission.l1_handles,
        l2_resolver=self.l2_sram,
        arena=admission.arena,
        binding_id=route.request.binding_id,
      )
      context = tile.load_program(
        program,
        binding.role_id,
        route.event_id,
        admission.prepare_cycles,
        admission.context_id,
        memory,
        task,
      )
      if context != admission.context_id:
        raise MemoryInvariantError("UCE binding failed after Arena commit")
      admission.bound = True
      self.profile_controller.note_user_issue(("l1",))
    except (MemoryInvariantError, ValueError):
      if pinned:
        self._unpin_task_l2(task, cycle)
      tile.abort_admission(admission, cycle)
      raise
    self._role_l1_handles[route.event_id][tile.tile_id] = admission.l1_handles
    tile.uce._phase_signal_callback = self._on_phase_signal
    for queue_id, queue in self.queues.items():
      tile.bind_stream(queue_id, queue)
    self.pmu.add_cycle("program_cold_load", admission.prepare_cycles)

  def _unpin_task_l2(self, task: TaskIdentity, cycle: int) -> None:
    grid_pins = self._grid_l2_pins.get(task.grid)
    if grid_pins is None:
      return
    for pin in grid_pins.pop(task.task_id, {}).values():
      self.l2_sram.unpin(pin.handle, pin.consumer_id, cycle)
    if not grid_pins:
      self._grid_l2_pins.pop(task.grid)
    self._sync_l2_pool_mirror(cycle)

  def _trace_task_lease(
    self, event: str, route: _GridRoute, task: TaskIdentity, tile_id: int, cycle: int
  ) -> None:
    if self.tracer is None:
      return
    admission = route.admissions[tile_id]
    program = route.binding.tile_program
    args = {
      "binding_id": route.request.binding_id,
      "task_id": task.task_id,
      "launch_generation": task.grid.launch_generation,
      "context_name": task.grid.context_name,
      "tile_id": tile_id,
      "ctx_id": admission.context_id,
      "hardware_context_id": admission.context_id,
      "role_id": route.binding.role_id,
      "grid_event": route.event_id,
      "program": program.name,
      "program_id": program.program_id,
      "program_hash": f"{program.program_hash:064x}",
      "requested_l1_mode": route.request.requested_l1_mode,
      "resolved_l1_mode": route.request.resolved_l1_mode,
      "l1_generation": self.profile_controller.generations["l1"],
    }
    if route.source_ref is not None:
      ref = route.source_ref
      args["source_ref"] = f"{ref.source_name}:{ref.symbol}:{ref.body_op_index}:{ref.op_name}"
    self.tracer.instant(f"Tile{tile_id}", "Lifecycle", event, cycle, args)

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
      if not (self.reset_domain.is_active or self.reset_domain.is_done):
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
      if not (self.reset_domain.is_active or self.reset_domain.is_done):
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
        if not (self.reset_domain.is_active or self.reset_domain.is_done):
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
          if handle.backing_id not in self._protocol_live_backing_bytes:
            self._protocol_live_backing_bytes[handle.backing_id] = handle.size_bytes
            new_live_bytes += handle.size_bytes
    if new_live_bytes:
      self._l2_live_bytes += new_live_bytes
      self._record_l2_occupancy(cycle)

  def context_cleanup_ready(self, sequencer: TileGroupSequencer) -> bool:
    generation = sequencer.context_launch_generation
    return (
      generation not in self._l2_arenas
      and not any(route.sequencer is sequencer for route in self._grid_routes.values())
      and not any(key[1] == generation for key in self._task_leases)
      and sequencer._outstanding_jobs == 0
    )

  def retire_context_arena(self, sequencer: TileGroupSequencer, cycle: int) -> bool:
    """Retire the root reserve before publishing its success completion."""
    generation = sequencer.context_launch_generation
    if any(route.sequencer is sequencer for route in self._grid_routes.values()):
      return False
    if any(key[1] == generation for key in self._task_leases):
      return False
    if sequencer._outstanding_jobs:
      return False
    handles = [handle for (gen, _slot), handle in self._l2_handles.items() if gen == generation]
    if any(not self.l2_sram.is_released(handle) for handle in handles):
      return False
    arena = self._l2_arenas.get(generation)
    if arena is None:
      return True
    held_before = self.l2_sram.snapshot()["arena_reserved_bytes"]
    if not self.l2_sram.retire_arena(arena, cycle):
      return False
    residual = self._sync_l2_pool_mirror(cycle)
    held_after = self.l2_sram.snapshot()["arena_reserved_bytes"]
    del self._l2_arenas[generation]
    self._l2_reserved_bytes -= held_before - held_after - residual
    if held_after != held_before:
      self._l2_capacity_change_cycle = cycle
    self._record_l2_occupancy(cycle)
    return True

  # ---- per-cycle step -------------------------------------------------

  def _step_group_transfers(self, cycle: int) -> None:
    tr = self.tracer
    completed_txns = self.transfer_manager.step(cycle)
    for txn in completed_txns:
      # PR 2: skip tile-local transactions — MFE tick handles them
      if txn.tile_id is not None or txn.transaction_id not in self._txn_sequencer:
        continue
      trace_slot = self._release_group_transfer_trace_slot(txn.transaction_id)
      seq = self._txn_sequencer.pop(txn.transaction_id)
      completion_status = (
        EventStatus.ERROR if seq.faulted or txn.status.value != "done" else EventStatus.DONE
      )
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
      self.transfer_manager.acknowledge(txn.transaction_id, cycle)

  def _step_collectives(self, cycle: int) -> None:
    tr = self.tracer
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

  def _step_stream_queues(self, cycle: int) -> None:
    tr = self.tracer
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

  def _step_tiles(self, cycle: int, *, freeze_new_work: bool) -> None:
    for tile in self.tiles:
      tile.step(cycle, freeze_new_work=freeze_new_work)
      for terminal in tile.drain_context_terminals():
        task = terminal.task_identity
        if task is None:
          raise MemoryInvariantError("terminal event lacks TaskIdentity")
        route = self._grid_routes.get(task.grid)
        if route is None or route.expected.get(tile.tile_id) != task:
          raise MemoryInvariantError("stale or foreign Task terminal")
        if terminal.status == "fault":
          reason = f"tile{tile.tile_id}: {terminal.reason}"
          self.on_scheduler_fault(route.sequencer, reason, cycle)
          continue
        self._retiring_tasks[task] = terminal
    for task, terminal in tuple(self._retiring_tasks.items()):
      route = self._grid_routes[task.grid]
      tile = self.tiles[terminal.tile_id]
      admission = route.admissions.get(tile.tile_id)
      if admission is None or admission.arena is None:
        raise MemoryInvariantError("terminal has no committed Task Arena")
      if any(self.transfer_manager.has_inflight_access(handle) for handle in admission.l1_handles.values()):
        continue
      pending = False
      for name, handle in tuple(admission.l1_handles.items()):
        if not tile.l1_allocator.invalidate_view(handle, handle.owner, cycle):
          pending = True
        else:
          del admission.l1_handles[name]
      if pending:
        continue
      if not tile.l1_allocator.retire_arena(admission.arena, cycle):
        continue
      self._unpin_task_l2(task, cycle)
      tile.l1_frames[terminal.ctx_id].release()
      key = (route.request.binding_id, task.grid.launch_generation, tile.tile_id)
      leases = self._task_leases[key]
      if task not in leases:
        raise MemoryInvariantError("Task retirement has no R lease")
      leases.remove(task)
      if not leases:
        del self._task_leases[key]
      self._trace_task_lease("task_lease_release", route, task, tile.tile_id, cycle)
      route.retired.add(tile.tile_id)
      self._role_l1_handles[route.event_id].pop(tile.tile_id, None)
      self._role_done_tiles[route.event_id].add(tile.tile_id)
      del self._retiring_tasks[task]
      if route.retired != set(route.expected):
        continue
      del self._grid_routes[task.grid]
      self._role_l1_handles.pop(route.event_id, None)
      route.sequencer.notify_event(
        route.event_id, cycle, EventStatus.ERROR if route.sequencer.faulted else EventStatus.DONE
      )
      if self.tracer is not None:
        trace = self._role_trace[route.event_id]
        self.tracer.complete(
          "TileGroup",
          f"TileRole:{trace.role_id}",
          f"dispatch:role{trace.role_id}:{route.event_id}:run",
          trace.start_cycle,
          cycle,
          args={
            "event_id": route.event_id,
            "binding_id": route.request.binding_id,
            "tile_mask": trace.tile_mask,
            "requested_l1_mode": route.request.requested_l1_mode,
            "resolved_l1_mode": route.request.resolved_l1_mode,
          },
        )

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
    self._step_group_transfers(cycle)
    self._step_collectives(cycle)
    self._step_stream_queues(cycle)

    # (NoC router steps in section 1, before the transfer manager)
    freeze_new_work = self._reset_freezes_new_work()

    # Running engines still tick; UCE issue/queued launches freeze.
    self._step_tiles(cycle, freeze_new_work=freeze_new_work)
    self.profile_controller.step(cycle)

    # 4. completions above are visible before one shared ISSUE and REGISTER.
    scheduler_frozen = freeze_new_work or self.reset_domain.is_active
    if not scheduler_frozen:
      self.scheduler.step(self._active_sequencers, cycle)
      self._step_task_admission(cycle)
    else:
      self.scheduler._poll_controls(cycle)
    for active_seq in self._active_sequencers:
      active_seq.maybe_finish()

    # 5. aggregate PMU

    # 5b. advance reset/drain FSM if active (runtime fidelity)
    if self.reset_domain.is_active:
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
      and not self._pending_root_requests
      and not self._grid_routes
      and not self._task_leases
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

  def _retire_sequencer(self, sequencer: TileGroupSequencer, cycle: int) -> None:
    """Release only metadata after the root Arena's confirmed retirement."""
    if not self.context_cleanup_ready(sequencer):
      raise MemoryInvariantError("Context completion preceded Arena/Task retirement")
    generation = sequencer.context_launch_generation
    self._grid_signals = {
      grid: state for grid, state in self._grid_signals.items() if state.sequencer is not sequencer
    }
    self._live_launches.pop((sequencer.context_name, sequencer.device_slot, generation), None)
    self._l2_handles = {key: handle for key, handle in self._l2_handles.items() if key[0] != generation}
    self._l2_roles.pop(generation, None)
    self._finish_sequencer_retirement(sequencer, cycle)

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

  def _build_l2_manifest(self, program: CompiledProgram) -> None:
    """Close shared claims over the actual, specialized model submit callsites."""
    self._l2_manifest.clear()
    self._l2_claim_export.clear()
    self._l2_export_buffers.clear()
    self._l2_backing_by_export.clear()
    self._l2_submit_successors.clear()
    if program.entry_kind != "model":
      return
    model = program.entry
    assert isinstance(model, ExecModel)
    event_bindings = {op.event_tag: op.binding_id for op in model.body if op.op == "submit"}
    successors: dict[str, set[str]] = {binding_id: set() for binding_id in model.tasks}
    awaited: set[str] = set()
    for op in model.body:
      if op.op == "await":
        awaited.add(op.event_tag)
      elif op.op == "submit":
        for dependency in op.dependencies:
          upstream = event_bindings.get(dependency)
          if upstream is not None:
            successors[upstream].add(op.binding_id)
        for dependency in awaited:
          upstream = event_bindings.get(dependency)
          if upstream is not None:
            successors[upstream].add(op.binding_id)
    for _ in range(len(successors)):
      for direct in successors.values():
        for consumer in tuple(direct):
          direct.update(successors.get(consumer, ()))
    self._l2_submit_successors = {
      key: frozenset(value) for key, value in successors.items()
    }
    for binding_id, task in model.tasks.items():
      for buffer in task.l2_buffers:
        if buffer.sharing == "readonly":
          self._l2_export_buffers[(binding_id, buffer.slot)] = buffer
    pending: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for binding_id, task in model.tasks.items():
      for shared in task.shared_inputs:
        claim_id = (binding_id, shared.slot)
        export = (shared.producer_binding_id, shared.producer_slot)
        if claim_id in self._l2_claim_export:
          raise MemoryInvariantError(f"duplicate L2 shared claim {claim_id}")
        if export not in self._l2_export_buffers:
          raise MemoryInvariantError(f"L2 shared claim {claim_id} has no producer export {export}")
        self._l2_claim_export[claim_id] = export
        pending.setdefault(export, []).append(claim_id)
    self._l2_manifest = {key: tuple(value) for key, value in pending.items()}

  # ---- lifecycle ------------------------------------------------------

  def begin_launch(self, program, bindings) -> None:
    """Clear retired launch bookkeeping; never reset profiles, Cache or residency."""
    if self.poisoned_reason is not None:
      raise MemoryInvariantError(f"TileGroup is poisoned and rejects new launches: {self.poisoned_reason}")
    if (
      self._active_sequencers
      or self._grid_routes
      or self._task_leases
      or self._l2_arenas
      or self._pending_root_requests
      or self.transfer_manager.inflight_count
      or self.l2_sram.snapshot()["live_backings"]
      or self.l2_sram.snapshot()["pending_shared_claims"]
      or self.l2_sram.snapshot()["active_shared_references"]
    ):
      raise MemoryInvariantError("previous launch has not retired")
    self.l2_sram.close_l2_claims()
    self._build_l2_manifest(program)
    self.loaded_program = program
    self.run_generation += 1
    self.l2_sram.run_generation = self.run_generation
    self.profile_controller.begin_run(self.run_generation)
    self.transfer_manager.begin_run(self.run_generation)
    self.scheduler.reset()
    self.event_table.clear()
    self._grid_signals.clear()
    self._grid_l2_pins.clear()
    self._live_launches.clear()
    self._l2_roles.clear()
    self._l2_sharing.clear()
    self._l2_published.clear()
    self._l2_released.clear()
    self._role_trace.clear()
    self._role_l1_handles.clear()
    self._role_done_tiles.clear()
    self._role_event_tile_mask.clear()
    self._protocol_live_l2.clear()
    self._protocol_live_backing_bytes.clear()
    self._l2_handles.clear()
    self._pending_context_admissions.clear()
    self._pending_activations.clear()
    self._task_trace_name = f"task:{program.entry.name}"
    self._task_start_cycle = None
    self._task_done_traced = False
    self.pmu.reset()
    if bindings != self._current_bindings:
      self.hbm.reset()
      self._global_handles.clear()
      self._current_bindings = dict(bindings)
    self.register_global_bindings(bindings)
    inputs = program.entry.inputs if hasattr(program.entry, "inputs") else program.entry.global_inputs
    self.profile_controller.bind_owner_inputs(
      f"device@{self.run_generation}", tuple(self._global_handles[item.name] for item in inputs)
    )
    if self.byte_store is not None:
      self.byte_store.validate_seed_coverage()

  def load_task(
    self,
    task: ExecTileGroupTask,
    *,
    input_bindings=None,
    formal_bindings: dict[str, str] | None = None,
    cycle: int = 0,
  ) -> None:
    sequencer = self.try_admit_context_task(
      task,
      0,
      context_name=task.name,
      input_bindings=input_bindings,
      formal_bindings=formal_bindings or {item.name: item.name for item in task.global_inputs},
      cycle=cycle,
    )
    if sequencer is None:
      raise MemoryInvariantError("standalone root cannot be admitted to an empty target")
    self.sequencer = sequencer

  @property
  def admission_version(self) -> tuple:
    return (
      self.l2_sram.pool_version,
      self.event_table.version,
      len(self._active_sequencers),
      tuple(self.profile_controller.generations.items()),
      tuple(self.profile_controller.active_modes.items()),
      self.profile_controller.issue_gate_closed("l2"),
    )

  def retained_shared_blocks_admission(
    self, task: ExecTileGroupTask, later_bindings: tuple[str, ...]
  ) -> bool:
    """Pure optimistic proof that FIFO-head private layout cannot fit retained backings."""
    if task.layout is None:
      raise MemoryInvariantError("retained L2 capacity proof requires a layout")
    blockers = {task.binding_id, *later_bindings}
    blockers.update(self._l2_submit_successors.get(task.binding_id, ()))
    retained: list[str] = []
    for backing_id in self._l2_backing_by_export.values():
      if any(
        claim_id[0] in blockers and state in ("DECLARED", "BOUND")
        for claim_id, state in self.l2_sram.backing_claims(backing_id)
      ):
        retained.append(backing_id)
    return not self.l2_sram.can_fit_with_retained(task.layout, tuple(retained))

  def try_admit_context_task(
    self,
    task: ExecTileGroupTask,
    slot_index: int,
    *,
    context_name: str | None = None,
    input_bindings=None,
    formal_bindings=None,
    cycle: int = 0,
  ) -> TileGroupSequencer | None:
    """Commit Slot-independent root resources only when the entire plan fits."""
    if self.poisoned_reason is not None:
      raise MemoryInvariantError(
        f"TileGroup is poisoned and rejects context admission: {self.poisoned_reason}"
      )
    self.last_admission_wait = None
    if self.loaded_program is None or task.layout is None:
      raise ValueError("Group requires a loaded compiled Context")
    binding = self.loaded_program.call_bindings[task.binding_id]
    if binding.resolved_l2_mode != self.profile_controller.active_modes["l2"]:
      raise ValueError("compiled Context resolved L2 profile differs from active mode")
    if self.profile_controller.issue_gate_closed("l2"):
      self.last_admission_wait = AdmissionWaitReason.CONTROL_RESOURCE
      return None
    if len(self._active_sequencers) >= self.scheduler_config.active_context_capacity:
      self.last_admission_wait = AdmissionWaitReason.SLOT
      return None
    launch_id = self._next_launch_id
    owner = RootInvocation(context_name or task.name, launch_id)
    imports: list[tuple[ExecSharedInput, str, ContextBufferOwner]] = []
    for shared in task.shared_inputs:
      claim_id = (task.binding_id, shared.slot)
      export_key = (shared.producer_binding_id, shared.producer_slot)
      if self._l2_claim_export.get(claim_id) != export_key:
        raise MemoryInvariantError(f"unknown or duplicate L2 shared claim {claim_id}")
      producer = self._l2_export_buffers.get(export_key)
      if producer is None or (
        shared.dims, shared.dtype, shared.element_bytes, shared.bytes
      ) != (producer.dims, producer.dtype, producer.element_bytes, producer.bytes):
        raise MemoryInvariantError(f"L2 shared input {shared.slot!r} shape/dtype differs from export")
      backing_id = self._l2_backing_by_export.get(export_key)
      if backing_id is None:
        raise MemoryInvariantError(f"L2 shared input {shared.slot!r} has no committed producer backing")
      borrower = ContextBufferOwner(owner.context_name, launch_id, shared.slot)
      self.l2_sram.check_borrow_l2_view(backing_id, borrower, claim_id)
      imports.append((shared, backing_id, borrower))
    plan = self.l2_sram.plan_arena(owner, task.layout)
    if isinstance(plan, AdmissionFailure):
      if plan.kind is AdmissionFailureKind.TEMPORARY_CAPACITY:
        self.last_admission_wait = plan.wait_reason
        return None
      raise ValueError(plan.reason)
    ticket = self._prepare_context_launch(
      task,
      slot_index,
      context_name=context_name,
      input_bindings=input_bindings,
      formal_bindings=formal_bindings,
      enqueue_cycle=cycle,
    )
    result = self.scheduler.reserve_context_events(ticket.sequencer, ticket.task)
    if result.status is IssueStatus.BACKPRESSURE:
      self.last_admission_wait = AdmissionWaitReason.CONTROL_RESOURCE
      return None
    if result.status is IssueStatus.FAULT:
      raise ValueError(result.reason)
    claims_by_slot = {
      buffer.slot: self._l2_manifest.get((task.binding_id, buffer.slot), ())
      for buffer in task.l2_buffers if buffer.sharing == "readonly"
    }
    try:
      arena = self.l2_sram.commit_arena(plan, cycle, claims_by_slot=claims_by_slot)
      borrowed = {
        shared.slot: self.l2_sram.borrow_l2_view(
          backing_id, borrower, (task.binding_id, shared.slot), cycle
        )
        for shared, backing_id, borrower in imports
      }
    except (ValueError, MemoryInvariantError):
      self.scheduler.cancel_context_events(ticket.sequencer)
      raise
    self._next_launch_id += 1
    self._l2_arenas[launch_id] = arena
    self._l2_reserved_bytes += arena.reserved_bytes
    self._l2_roles[launch_id] = {
      **{buffer.slot: buffer.role for buffer in task.l2_buffers},
      **{shared.slot: "in" for shared in task.shared_inputs},
    }
    self._l2_sharing[launch_id] = {buffer.slot: buffer.sharing for buffer in task.l2_buffers}
    for buffer in task.l2_buffers:
      if buffer.sharing == "readonly":
        self._l2_backing_by_export[(task.binding_id, buffer.slot)] = self.l2_sram.l2_backing_id(
          arena, buffer.slot
        )
    self._l2_handles.update({(launch_id, slot): handle for slot, handle in borrowed.items()})
    self._record_l2_occupancy(cycle)
    self.profile_controller.bind_owner_inputs(
      f"{task.binding_id}@{launch_id}",
      tuple(
        self._global_handles[(formal_bindings or {}).get(item.name, item.name)]
        for item in task.global_inputs
      ),
    )
    self._activate_admitted_context(ticket, cycle)
    self.profile_controller.note_user_issue(("l2",))
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
    """Instantiate only explicit compiled relocations, without reserving resources."""
    if self.loaded_program is None:
      raise ValueError("Context relocation requires a loaded artifact")
    launch_id = self._next_launch_id
    prefix = f"s{slot_index}l{launch_id}_"
    queue_offset = (launch_id * 100 + slot_index) * 10000
    task = relocate_task(task, self.loaded_program.relocations, prefix, queue_offset)
    owned_qids = {stream.queue_id for stream in task.streams}
    owned_qids.update(
      action.args[0] for action in task.actions if action.op is ExecGroupActionOp.INIT_STREAM
    )
    sequencer = TileGroupSequencer(self)
    sequencer.context_launch_generation = launch_id
    sequencer.context_name = context_name or task.name
    sequencer.device_slot = slot_index
    sequencer.formal_bindings = dict(formal_bindings or {})
    sequencer.load(task)
    return _PendingContextAdmission(
      sequencer, task, sequencer.context_name, slot_index, launch_id, frozenset(owned_qids), enqueue_cycle
    )

  def reset(self) -> None:
    """Explicit quiescent recovery, never a launch-time resource shortcut."""
    if (
      self._active_sequencers
      or self._grid_routes
      or self._task_leases
      or self._l2_arenas
      or self.transfer_manager.inflight_count
      or self._pending_root_requests
    ):
      raise MemoryInvariantError("explicit reset requires retired or isolated work")
    if self.reset_domain.is_active:
      raise MemoryInvariantError("explicit reset requires a completed reset drain")
    residual = self.unclosed_l2_objects()
    if residual:
      raise MemoryInvariantError(f"explicit reset requires physical L2 safety, found {residual}")
    cycle = self._last_step_cycle + 1
    self.profile_controller.recover(cycle)
    deadline = cycle + self.cfg.memory_target.profile_command_timeout_cycles
    while not self.profile_controller.initialized:
      if cycle >= deadline:
        raise MemoryInvariantError("explicit recovery ACK timeout")
      self.profile_controller.step(cycle)
      cycle += 1
    for tile in self.tiles:
      tile.reset()
    self.event_table.clear()
    self.scheduler.reset()
    self.program_table.invalidate_group()
    self.reset_domain.reset()
    self.fault_ring.reset()
    self.poisoned_reason = None
    self._last_step_cycle = cycle

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
      "compiled_artifact_hash": None if self.loaded_program is None else self.loaded_program.artifact_hash,
      "registry_hash": self.registry.registry_hash,
      "profile": self.profile_controller.snapshot(),
      "arenas": {
        "l2": self.l2_sram.snapshot(),
        "l1": {t.tile_id: t.l1_allocator.snapshot() for t in self.tiles},
      },
      "task_leases": {
        "active": sum(len(leases) for leases in self._task_leases.values()),
        "routes": len(self._grid_routes),
        "retiring": len(self._retiring_tasks),
      },
      "task_done": self.sequencer.done,
      "task_submission_pc": self.sequencer.submission_pc,
      "scheduler": self._scheduler_snapshot(),
      "event_table": self.event_table.snapshot(),
      "queues": {qid: q.snapshot() for qid, q in self.queues.items()},
      "tiles": [t.snapshot() for t in self.tiles],
      "memory": {
        "fidelity": self.fidelity,
        "hbm": self.hbm.snapshot(),
        "l2": self.l2_sram.snapshot(),
        "l1": {
          tile.tile_id: {
            "allocator": tile.l1_allocator.snapshot(),
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
    if not (self.reset_domain.is_active or self.reset_domain.is_done):
      self.trigger_fault(self._fault_code_for_reason(reason), cycle=cycle, desc_id=reason)

  # ---- fault / reset (runtime fidelity) -------------------------------

  def trigger_fault(self, code, tile_id: int = -1, cycle: int = 0, desc_id: str = "") -> int:
    """Inject a fault: write a FaultRecord and begin the reset/drain FSM
    (Driver-Firmware 3.3/3.4).  Returns the fault_record_index, or -1
    in timing_only fidelity (no-op).
    """
    rec = FaultRecord(code=code, tile_id=tile_id, desc_id=zlib.crc32(desc_id.encode()) & 0xFFFFFFFF)
    idx = self.fault_ring.write(rec)
    domain = FaultDomain.TILE if tile_id >= 0 else FaultDomain.GROUP
    req = ResetRequest(domain=domain, tile_id=tile_id, fault_record=rec)
    self.reset_domain.begin(req, cycle)
    self.pmu.add_event("fault_record", 1)
    return idx
