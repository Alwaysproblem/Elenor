"""Tile Group: 1 Tile Group Sequencer + 4 Compute Tiles + Group SRAM + streams.

The Tile Group is the local data-reuse / synchronization unit
(design/elenor_tile_group/).  It owns the Stream Queues that connect
task roles and the Group DMA.  The simulator drives it cycle by cycle,
advancing the Tile Group Sequencer and every Compute Tile in lockstep.
"""

from __future__ import annotations

import copy
import zlib
from dataclasses import dataclass

from .config import HardwareConfig
from .execution_ir import (
  ExecGroupActionOp,
  ExecStreamDesc,
  ExecTileGroupTask,
  ExecTileOp,
  ExecTileRoleBinding,
)
from .memory import (
  L2SRAM,
  AdmissionFailure,
  AllocationHandle,
  AllocationPlan,
  AllocationRequest,
  BankSegment,
  ContextBufferOwner,
  MemoryInvariantError,
  MemoryTransaction,
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
from .trace import Tracer


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



class TileGroup:
  """One ELENOR Tile Group with 4 Compute Tiles."""

  def __init__(self, cfg: HardwareConfig, tracer: Tracer | None = None,
               fidelity: str = "full_memory", context_count: int = 1,
               trace_prefix: str = ""):
    self.cfg = cfg
    self.tracer = tracer
    self.fidelity = fidelity
    rt = fidelity in ("runtime", "full_memory")
    mem = fidelity == "full_memory"
    self.runtime_enabled = rt
    self.memory_enabled = mem
    # PR 2: transfer manager must exist before tiles are created (injected
    # into ComputeTile/MFEEngine as a shared instance).
    from .memory.hbm_region import HBMRegion
    from .memory.transfer import TransferManager
    # NoC fabric must exist before the transfer manager (NoC legs are
    # router-backed: flit enqueue/traversal/credit via NoCRouter)
    if mem:
      self.noc = NoCRouter(
        vc_depth=cfg.noc_vc_depth,
        router_latency_cycles=cfg.noc_router_latency_cycles)
    self.transfer_manager = TransferManager(
      cfg, full_memory=mem, noc=self.noc if mem else None)
    self.tiles: list[ComputeTile] = [
      ComputeTile(i, cfg, self.tracer, runtime_enabled=rt,  # type: ignore[arg-type]
                  memory_enabled=mem, context_count=context_count,
                  transfer_manager=self.transfer_manager)
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
    self._role_phase_events: dict[str, dict[str, str]] = {}
    # role trace bookkeeping: event_id -> _RoleTrace
    self._role_trace: dict[str, _RoleTrace] = {}
    # task trace bookkeeping
    self._task_trace_name: str | None = None
    self._task_start_cycle: int | None = None
    self._task_done_traced: bool = False
    self.hbm = HBMRegion(
      base_iova=0,
      size_bytes=cfg.hbm_capacity_bytes,
      bandwidth_gbs=cfg.hbm_bandwidth_gbs,
      outstanding_limit=cfg.hbm_outstanding_limit)
    # PR 2 admission state: launch generation, global/L2 handles, pins
    self._context_launch_generation: int = 0
    self._global_handles: dict[str, AllocationHandle] = {}  # binding name -> handle
    self._l2_handles: dict[tuple[int, str], AllocationHandle] = {}  # (gen, slot) -> handle
    # role_event_id -> {tile_id: (logical_task_id, unique L2 handles)}
    self._role_l2_pins: dict[
      str, dict[int, tuple[int, list[AllocationHandle]]]] = {}
    # formal name -> actual binding name (model mode)
    self._formal_bindings: dict[str, str] = {}
    # PR 2: role_event_id -> {tile_id: [L1 AllocationHandle]}
    self._role_l1_handles: dict[str, dict[int, list]] = {}
    # transaction id -> sequencer
    self._txn_sequencer: dict[str, TileGroupSequencer] = {}
    if self.runtime_enabled:
      self.event_table = EventTable()
      self.fault_ring = FaultRing()
      self.program_table = ProgramResidencyManager(cfg)
      self.reset_domain = ResetDomain(cfg)
    if self.memory_enabled or self.runtime_enabled:
      self.l2_sram = L2SRAM(
        capacity_bytes=cfg.group_sram_bytes,
        banks=cfg.group_sram_banks,
        bank_bandwidth_gbs=cfg.l2_bank_bandwidth_gbs)
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
    txn_id = f"{gen}:{event_id}"
    bytes_total = transfer.bytes if transfer.bytes > 0 else 1024 * 1024
    if self.memory_enabled or self.runtime_enabled:
      # resolve src/dst views against admission handles
      src_view = self._resolve_view(transfer.src, "global", gen)
      dst_view = self._resolve_view(transfer.dst, "l2", gen)
      if op == "dma.store":
        src_view = self._resolve_view(transfer.src, "l2", gen)
        dst_view = self._resolve_view(transfer.dst, "global", gen)
      if src_view is None or dst_view is None:
        # missing handle → fault
        return False
      txn = MemoryTransaction(
        transaction_id=txn_id,
        op=(TransferOp.PREFETCH if op == "dma.prefetch"
            else TransferOp.GLOBAL_STORE),
        issuer=ContextBufferOwner(
          sequencer.task.name if sequencer is not None and sequencer.task is not None else "ctx",
          gen, desc_id),
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
        issuer=ContextBufferOwner("ctx", self._context_launch_generation, desc_id),
        src=None, dst=None,
        bytes_total=bytes_total,
        completion_event=event_id,
      )
    self.transfer_manager.submit(txn, cycle, self.pmu)
    if sequencer is not None:
      self._txn_sequencer[txn_id] = sequencer
      sequencer.note_job_started()
    return True

  def _resolve_view(self, view, default_space: str,
                    gen: int | None = None) -> ResolvedMemoryView | None:
    """Resolve an ``ExecMemoryView`` to a ``ResolvedMemoryView`` using
    the admission handles.  Returns None if the handle is missing."""
    if view is None:
      return None
    space = view.space
    use_gen = gen if gen is not None else self._context_launch_generation
    if space == "global":
      # global view: resolve against HBM external binding
      name = view.base.removeprefix("global:")
      # map formal name to actual binding name (model mode)
      name = self._formal_bindings.get(name, name)
      handle = self._global_handles.get(name)
      if handle is None:
        return None
      offset = self._view_offset_bytes(view, logical_task_id=0)
      seg = BankSegment(bank_id=0, address=handle.base_address + offset,
                        size_bytes=view.bytes)
      return ResolvedMemoryView(
        handle=handle, offset_bytes=offset, size_bytes=view.bytes,
        address=handle.base_address + offset, segments=(seg,))
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

  def admit_l2_buffers(self, task: ExecTileGroupTask, cycle: int,
                       context_name: str | None = None) -> bool:
    """L2 admission: plan + commit all ``l2_buffers`` as a bundle.

    Returns True on success, False on capacity fault (no handles added).
    """
    if not (self.memory_enabled or self.runtime_enabled) or not task.l2_buffers:
      return True
    ctx_name = context_name or task.name
    gen = self._context_launch_generation
    requests = [
      AllocationRequest(
        memory_space="l2",
        buffer_id=buf.slot,
        owner=ContextBufferOwner(ctx_name, gen, buf.slot),
        size_bytes=buf.bytes,
        alignment=max(buf.alignment, 1),
        role=buf.role,
      )
      for buf in task.l2_buffers
    ]
    plan = self.l2_sram.plan_bundle(requests)
    from .memory.allocator import AdmissionFailure
    if isinstance(plan, AdmissionFailure):
      return False
    handles = self.l2_sram.commit(plan, cycle)
    for buf, handle in zip(task.l2_buffers, handles):
      self._l2_handles[(gen, buf.slot)] = handle
    return True

  def release_l2_slot(self, slot: str, cycle: int,
                      generation: int | None = None) -> None:
    """Request release by slot in the owning context launch generation."""
    gen = (generation if generation is not None
           else self._context_launch_generation)
    handle = self._l2_handles.pop((gen, slot), None)
    if handle is None:
      return
    if self.memory_enabled or self.runtime_enabled:
      self.l2_sram.request_release(handle, handle.owner, cycle)

  def _pin_role_l2(self, binding: ExecTileRoleBinding,
                   role_event_id: str, logical_task_id: int,
                   tile_id: int, generation: int, cycle: int) -> None:
    """Pin each unique L2 allocation once for one logical task.

    ``actuals`` contains both ins and outs; allocation-id dedup prevents
    an inout buffer from receiving a duplicate pin.
    """
    unique: dict[str, AllocationHandle] = {}
    for slot in binding.actuals:
      handle = self._l2_handles.get((generation, slot))
      if handle is not None:
        unique.setdefault(handle.allocation_id, handle)
    consumer_id = f"{role_event_id}:{logical_task_id}"
    pinned: list[AllocationHandle] = []
    try:
      for handle in unique.values():
        self.l2_sram.pin(handle, consumer_id)
        pinned.append(handle)
    except MemoryInvariantError:
      for handle in pinned:
        self.l2_sram.unpin(handle, consumer_id, cycle)
      raise
    self._role_l2_pins.setdefault(role_event_id, {})[tile_id] = (
      logical_task_id, list(unique.values()))

  def _release_role_l2_pins(self, role_event_id: str,
                            tile_id: int, cycle: int) -> None:
    """Unpin the unique L2 allocations consumed by one terminal task."""
    tile_pins = self._role_l2_pins.get(role_event_id, {})
    entry = tile_pins.pop(tile_id, None)
    if entry is None:
      return
    logical_task_id, handles = entry
    consumer_id = f"{role_event_id}:{logical_task_id}"
    for handle in handles:
      try:
        self.l2_sram.unpin(handle, consumer_id, cycle)
      except MemoryInvariantError:
        pass
    if not tile_pins:
      self._role_l2_pins.pop(role_event_id, None)

  def release_context_memory(self, cycle: int) -> None:
    """Release every context-owned L2/L1 allocation, pin and frame.

    Fault-domain reset cleanup (§6.5): returns all context-owned
    handles to their allocators.  External HBM bindings are preserved.
    Release errors (double release / stale generation) are tolerated so
    reset never faults while cleaning up.
    """
    if not (self.memory_enabled or self.runtime_enabled):
      return
    # Remove consumer pins first.  If RELEASE_L2 already marked a handle
    # RELEASE_PENDING, the final reset-time unpin performs the actual free.
    for role_event_id, tile_pins in list(self._role_l2_pins.items()):
      for tile_id in list(tile_pins):
        self._release_role_l2_pins(role_event_id, tile_id, cycle)
    self._role_l2_pins.clear()
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
    event_id: str | None = None,
    phase_event_ids: dict[str, str] | None = None,
    sequencer: TileGroupSequencer | None = None,
  ) -> bool:
    """Atomically admit and bind one TileRole across all selected tiles.

    Phase 1 selects every UCE context and plans every L1 allocation with
    zero side effects.  Phase 2 commits/prepares every tile, pins L2
    consumers, then binds exact contexts.  Any late failure rolls back
    all earlier commits/pins/frames/contexts and faults the issuing
    sequencer (never the default sequencer by accident).
    """
    role_id = binding.role_id
    tile_mask = binding.tile_mask
    ev = event_id or f"ev_role{role_id}"
    seq = sequencer or self.sequencer
    prog = binding.tile_program
    gen = seq.context_launch_generation
    from_task = (binding.task_domain.from_task
                 if binding.task_domain is not None else 0)
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
          adm.tile.rollback_program(adm.context_id)
        else:
          adm.tile.l1_frames[adm.context_id].release()
      # Remove L2 consumer pins created during phase 2.
      for adm in admissions:
        self._release_role_l2_pins(ev, adm.tile.tile_id, cycle)
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
      self._role_phase_events.pop(ev, None)
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
      # Pin each unique L2 allocation once per logical task only after all
      # L1 commits/prepares succeeded.
      if self.memory_enabled or self.runtime_enabled:
        for adm in admissions:
          self._pin_role_l2(
            binding, ev, adm.logical_task_id,
            adm.tile.tile_id, gen, cycle)
      # Phase 2b: bind exact contexts.  Single-threaded cycle-step means no
      # other dispatcher can steal a context between plan and bind.
      from .tile import _TileContextMemory
      for adm in admissions:
        l2_handles: tuple = ()
        if self.memory_enabled or self.runtime_enabled:
          l2_handles = tuple(
            self._l2_handles.get((gen, slot))
            for slot in binding.actuals[:len(prog.formals) - 1])
        l1_map = {
          buf.name: handle
          for buf, handle in zip(prog.l1_buffers, adm.l1_handles)
        }
        memory = _TileContextMemory(
          owner=ev, logical_task_id=adm.logical_task_id,
          launch_generation=gen,
          l2_formal_handles=l2_handles, l1_handles=l1_map,
          l2_resolver=self.l2_sram
          if (self.memory_enabled or self.runtime_enabled) else None)
        ctx_id = adm.tile.load_program(
          prog, role_id=role_id, role_event_id=ev,
          prepare_cycles=adm.prepare_cycles,
          context_id=adm.context_id, memory=memory)
        if ctx_id != adm.context_id:
          raise MemoryInvariantError(
            f"UCE context bind failed on tile {adm.tile.tile_id}")
        adm.bound = True
    except (MemoryInvariantError, ValueError) as exc:
      rollback()
      return fail(str(exc))

    # Publish role bookkeeping only after every tile bound successfully.
    self._role_event_tile_mask[ev] = tile_mask
    self._role_done_tiles[ev] = set()
    self._role_phase_events[ev] = phase_event_ids or {}
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

  def _on_phase_signal(self, role_event_id: str, phase: str, tile_id: int) -> None:
    phase_map = self._role_phase_events.get(role_event_id, {})
    phase_ev = phase_map.get(phase)
    if phase_ev is None or not phase_ev:
      return
    mask = self._role_event_tile_mask.get(role_event_id, 0)
    if mask == 0:
      return
    self._role_event_tile_mask.setdefault(phase_ev, mask)
    done = self._role_done_tiles.setdefault(phase_ev, set())
    if tile_id in done:
      return
    done.add(tile_id)
    expected = bin(mask).count("1")
    if len(done) >= expected:
      rt = self._role_trace.get(role_event_id)
      phase_seq = (rt.sequencer if rt is not None and rt.sequencer is not None
                   else self.sequencer)
      phase_seq.notify_event(phase_ev)

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
        op_name = ("dma.store" if txn.op == TransferOp.GLOBAL_STORE
                   else "dma.prefetch")
        tr.complete(
          "TileGroup", "GroupDMA",
          f"{op_name}:{txn.transaction_id}",
          txn.start_cycle, txn.completed_cycle,
          args={
            "transaction_id": txn.transaction_id,
            "event_id": txn.completion_event,
            "bytes": txn.bytes_total,
            "start_cycle": txn.start_cycle,
            "completion_cycle": txn.completed_cycle,
            "source_address": txn.src.address if txn.src is not None else None,
            "destination_address":
              txn.dst.address if txn.dst is not None else None,
          })
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
    self._collective_jobs = remaining_coll

    # 2. tick stream queues (PMU occupancy counters + trace counters)
    for q in self.queues.values():
      q.tick(cycle)
      if tr is not None:
        tr.counter(f"StreamQ{q.queue_id}", "occupancy", cycle, q.occupancy,
                   "tokens")
        tr.counter(f"StreamQ{q.queue_id}", "credit_available", cycle,
                   q._credit_available, "credits")

    # (NoC router steps in section 1, before the transfer manager)
    freeze_new_work = self._reset_freezes_new_work()

    # 3. tick sequencers only while dispatch is not frozen
    if not freeze_new_work:
      for seq in self._active_sequencers:
        seq.step(cycle)
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
        # PR 2: release L1 frame + allocations on tile terminal (§5.7)
        if self.memory_enabled or self.runtime_enabled:
          t.l1_frames[term.ctx_id].release()
          tile_l1 = self._role_l1_handles.get(term.role_event_id, {})
          for handle in tile_l1.pop(t.tile_id, []):
            try:
              t.l1_allocator.request_release(handle, handle.owner, cycle)
            except MemoryInvariantError:
              pass  # already released or stale - terminal must not fault
          self._release_role_l2_pins(
            term.role_event_id, t.tile_id, cycle)
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
    # Prune completed sequencers; reclaim their namespaced stream queues
    # and tile bindings so repeated submits don't grow them unboundedly.
    remaining: list[TileGroupSequencer] = []
    for s in self._active_sequencers:
      if s.done:
        for qid in s.owned_queue_ids:
          self.queues.pop(qid, None)
          for t in self.tiles:
            t.unbind_stream(qid)
      else:
        remaining.append(s)
    self._active_sequencers = remaining
    all_done = len(self._active_sequencers) == 0
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
    l1_wait = sum(
      s.wait_cycles for s in list(tm._l1_read.values())
      + list(tm._l1_write.values()))
    self.pmu.add_cycle("l1_bank_wait", l1_wait)
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
    self._role_phase_events.clear()
    # PR 2: bump launch generation, clear admission state
    self._context_launch_generation += 1
    self._global_handles.clear()
    self._l2_handles.clear()
    self._role_l1_handles.clear()
    self._role_l2_pins.clear()
    self._txn_sequencer.clear()
    self.transfer_manager.reset()
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
    # PR 2: store formal→actual binding map (standalone: identity)
    self._formal_bindings = dict(formal_bindings or {})
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
    if not self.admit_l2_buffers(task, cycle=0):
      # L2 capacity fault: create a faulted sequencer
      self.sequencer.faulted = True
      self.sequencer.fault_reason = "L2 capacity fault during context admission"
      self.sequencer.done = True
      return
    self.sequencer.context_launch_generation = self._context_launch_generation
    self.sequencer.load(task)
    # pre-init streams declared in the task (some tasks init inline)
    for s in task.streams:
      self.init_stream(s)

  def load_context_task(self, task: ExecTileGroupTask,
                        slot_index: int = 0, *,
                        context_name: str | None = None,
                        input_bindings=None,
                        formal_bindings: dict[str, str] | None = None,
                        ) -> TileGroupSequencer:
    """Load a model-mode context task without resetting shared state.

    Deep-clones the task, then namespaces every event ID and stream
    queue ID with a monotonic launch ID (``s{slot}l{launch}_`` for
    events; integer queue offset for streams) so sequential
    re-submissions on the same slot cannot consume stale completions
    and concurrent tasks cannot collide on shared group-level tracking.

    Creates a fresh TileGroupSequencer for this task and adds it to the
    active sequencer list.  Tiles, DMA channels, L2, and program
    residency are shared across all concurrently-loaded context tasks.

    Dispatch bindings with ``context_id = None`` are auto-assigned to
    UCE context ``slot_index`` so each device slot runs on its own UCE
    context, enabling true concurrency on the shared tiles.  Explicit
    dispatch ``context_id`` pins are preserved (IR_SPEC §3.5); callers
    must ensure they don't conflict across concurrent slots.  Requires
    ``context_count >= device_context_count``.
    Returns the new sequencer so the caller can track its completion.
    """
    max_ctx = self.tiles[0].uce.context_count
    if slot_index >= max_ctx:
      raise ValueError(
        f"device slot {slot_index} requires UCE context {slot_index}"
        f" but context_count is {max_ctx}")
    launch_id = self._next_launch_id
    self._next_launch_id += 1
    # PR 2: use the monotonic launch id as the context launch generation
    self._context_launch_generation = launch_id
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
        # args = (role_id, inrel_tag, outready_tag)
        action.args = tuple(
          prefix + a if isinstance(a, str) and a else a
          for a in action.args)
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
    # PR 2: register global bindings (shared across contexts) + L2 admission
    if input_bindings:
      self.register_global_bindings(input_bindings)
    # PR 2: store formal→actual binding map for this context
    if formal_bindings:
      self._formal_bindings.update(formal_bindings)
    if not self.admit_l2_buffers(task, cycle=0, context_name=context_name):
      seq = TileGroupSequencer(self)
      seq.faulted = True
      seq.fault_reason = "L2 capacity fault during context admission"
      seq.done = True
      self._active_sequencers.append(seq)
      return seq
    seq = TileGroupSequencer(self)
    seq.context_launch_generation = self._context_launch_generation
    seq.load(task)
    seq.owned_queue_ids = owned_qids
    self._active_sequencers.append(seq)
    for s in task.streams:
      self.init_stream(s)
    return seq
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
    self._role_phase_events.clear()
    self._task_trace_name = None
    self._task_start_cycle = None
    self._task_done_traced = False
    self.pmu.reset()
    self._registered_programs.clear()
    # PR 2: clear admission + transfer state
    self._context_launch_generation = 0
    self._global_handles.clear()
    self._l2_handles.clear()
    self._role_l1_handles.clear()
    self._role_l2_pins.clear()
    self._txn_sequencer.clear()
    self.transfer_manager.reset()
    self.hbm.reset()
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
          t.tile_id: {
            "allocator": t.l1_allocator.snapshot()
            if self.runtime_enabled else None,
            "frames": [f.snapshot() for f in t.l1_frames],
          }
          for t in self.tiles
        },
        "transfers": self.transfer_manager.snapshot(),
        "noc": self.noc.snapshot() if self.memory_enabled else None,
      },
      "collective_jobs": len(self._collective_jobs),
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
