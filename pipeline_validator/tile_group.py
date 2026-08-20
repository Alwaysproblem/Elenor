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
from .memory import L2SRAM, NoCRouter, PayloadTracker
from .pmu import PMUCounter
from .runtime import (
  EventStatus,
  EventTable,
  FaultDomain,
  FaultRecord,
  FaultRing,
  ProgramResidencyManager,
  ResetDomain,
  ResetRequest,
)
from .stream_queue import EOSPolicy, QueueKind, StreamQueue
from .tile import ComputeTile
from .tile_group_sequencer import TileGroupSequencer
from .trace import Tracer


@dataclass
class _DMAJob:
  """A Group DMA in flight (prefetch/store)."""
  event_id: str
  start_cycle: int
  finish_cycle: int
  op: str
  desc_id: str
  l2_slot: str
  bytes_total: int
  channel: int
  sequencer: TileGroupSequencer | None = None  # sequencer that issued this job


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



class TileGroup:
  """One ELENOR Tile Group with 4 Compute Tiles."""

  def __init__(self, cfg: HardwareConfig, tracer: Tracer | None = None,
               fidelity: str = "timing_only", context_count: int = 1,
               trace_prefix: str = ""):
    self.cfg = cfg
    self.tracer = tracer
    self.fidelity = fidelity
    rt = fidelity in ("runtime", "full_memory")
    mem = fidelity == "full_memory"
    self.tiles: list[ComputeTile] = [
      ComputeTile(i, cfg, self.tracer, runtime_enabled=rt,  # type: ignore[arg-type]
                  memory_enabled=mem, context_count=context_count)
      for i in range(cfg.num_tiles)
    ]
    self.sequencer = TileGroupSequencer(self)
    self._active_sequencers: list[TileGroupSequencer] = [self.sequencer]
    self._next_launch_id: int = 0
    self.queues: dict[int, StreamQueue] = {}
    self._dma_jobs: list[_DMAJob] = []
    # per-channel "free cycle" for serializing DMA jobs on each channel
    self._channel_free_cycle: dict[int, int] = {}
    self._collective_jobs: list[_CollectiveJob] = []
    self.pmu = PMUCounter()
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
    # --- runtime-level components (runtime / full_memory fidelity) ---
    # Created lazily so timing_only mode pays zero cost.
    self.runtime_enabled = fidelity in ("runtime", "full_memory")
    self.memory_enabled = fidelity == "full_memory"
    self._registered_programs: dict[tuple[int, int, int], int] = {}
    if self.runtime_enabled:
      self.event_table = EventTable()
      self.fault_ring = FaultRing()
      self.program_table = ProgramResidencyManager(cfg)
      self.reset_domain = ResetDomain(cfg)
    if self.memory_enabled:
      self.l2_sram = L2SRAM(
        capacity_bytes=cfg.group_sram_bytes,
        banks=cfg.group_sram_banks,
        bank_bandwidth_gbs=cfg.l2_bank_bandwidth_gbs)
      self.noc = NoCRouter(
        vc_depth=cfg.noc_vc_depth,
        router_latency_cycles=cfg.noc_router_latency_cycles)
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

  def schedule_dma(
    self,
    event_id: str,
    latency: int,
    cycle: int,
    op: str,
    desc_id: str,
    l2_slot: str,
    bytes_total: int,
    channel: int = 0,
    sequencer: TileGroupSequencer | None = None,
  ) -> None:
    # Each channel serializes jobs: a new job starts only after the previous
    # one on the same channel finishes.  This models a real DMA channel with
    # a single outstanding request.
    ch_free = self._channel_free_cycle.get(channel, 0)
    start = max(cycle, ch_free)
    finish = start + latency
    self._channel_free_cycle[channel] = finish
    self._dma_jobs.append(
      _DMAJob(
        event_id=event_id,
        start_cycle=start,
        finish_cycle=finish,
        op=op,
        desc_id=desc_id,
        l2_slot=l2_slot,
        bytes_total=bytes_total,
        channel=channel,
        sequencer=sequencer,
      ))
    if sequencer is not None:
      sequencer.note_job_started()

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
  ) -> None:
    """Load the role's Tile Program onto the selected tiles and mark them."""
    role_id = binding.role_id
    tile_mask = binding.tile_mask
    ev = event_id or f"ev_role{role_id}"
    seq = sequencer or self.sequencer
    self._role_event_tile_mask[ev] = tile_mask
    self._role_done_tiles[ev] = set()
    self._role_phase_events[ev] = phase_event_ids or {}
    self._role_trace.pop(ev, None)
    total_cold = 0
    prog = binding.tile_program
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
    ctx_ids: list[int] = []
    for t in self.tiles:
      if not (tile_mask & (1 << t.tile_id)):
        continue
      prepare = 0
      if self.runtime_enabled and prog.program_id != 0:
        prepare = self.program_table.ensure_resident(
          prog.program_id, t.tile_id, cycle)
        total_cold += prepare
      ctx_id = t.load_program(binding.tile_program,
                              role_id=role_id,
                              role_event_id=ev,
                              prepare_cycles=prepare,
                              context_id=binding.context_id)
      assert ctx_id is not None
      ctx_ids.append(ctx_id)
      if self.runtime_enabled:
        t.uce._event_done_callback = self._make_event_callback()
      t.uce._phase_signal_callback = self._on_phase_signal
      for qid, q in self.queues.items():
        t.bind_stream(qid, q)
      for done_ev in sorted(seq._events_done):
        if "ev_dma_" in done_ev:
          t.uce.notify_event(done_ev)
    # Register _RoleTrace unconditionally: completion/phase routing
    # depends on its sequencer reference (not just trace emission).
    self._role_trace[ev] = _RoleTrace(
      role_id=role_id,
      event_id=ev,
      start_cycle=cycle,
      tile_mask=tile_mask,
      out_stream=binding.out_stream,
      in_stream=binding.in_stream,
      sequencer=seq,
    )
    tr = self.tracer
    if tr is not None:
      ctx_trace_arg = ctx_ids[0] if len(set(ctx_ids)) == 1 else ctx_ids
      tr.instant(
        "TileGroup",
        f"TileRole:{role_id}",
        "tile_role_dispatch",
        cycle,
        {
          "role_id": role_id,
          "tile_mask": tile_mask,
          "program": binding.tile_program.name,
          "event_id": ev,
          "out_stream": binding.out_stream,
          "in_stream": binding.in_stream,
          "ctx_id": ctx_trace_arg,
          "pinned_context": binding.context_id,
          "context_count": self.tiles[0].uce.context_count,
        },
      )
    if total_cold > 0:
      self.pmu.add_cycle("program_cold_load", total_cold)

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

    # 1. tick Group DMA jobs
    remaining_dma: list[_DMAJob] = []
    for job in self._dma_jobs:
      if cycle >= job.finish_cycle:
        seq = job.sequencer or self.sequencer
        seq.notify_event(job.event_id)
        if job.sequencer is not None:
          job.sequencer.note_job_done()
        # active tiles so a persistent tile program can WAIT on
        # sequencer-issued DMA events (e.g. ev_dma_A{g}).  Tiles not
        # waiting on this event ignore it.
        for t in self.tiles:
          if t.uce.has_active_contexts():
            t.uce.notify_event(job.event_id)
        # full_memory: record the DMA transfer as a payload so downstream
        # engine layout checks can validate the cross-engine ABI (P0-3).
        # Alloc a tracked src payload first, then copy to the L2 slot IOVA.
        if self.memory_enabled:
          src_iova = zlib.crc32(job.l2_slot.encode()) & 0xFFFFFFFF
          if self.payload.get(src_iova) is None:
            from .memory.payload import Payload
            self.payload.alloc(src_iova, Payload(
                iova=src_iova, bytes_total=job.bytes_total,
                layout="paged_kv" if "page" in job.op or "prefetch" in job.op
                       else "row_major", producer_kind="DMA"))
          dst_iova = (src_iova + 1) & 0xFFFFFFFF
          self.payload.copy(
              src_iova=src_iova, dst_iova=dst_iova,
              bytes_total=job.bytes_total,
              layout_transform=("packed_kv" if "page" in job.op
                                 or "prefetch" in job.op else None))
        if tr is not None:
          thread = f"DMA Ch{job.channel}"
          tr.complete(
            "TileGroup",
            thread,
            f"{job.op}:{job.desc_id}",
            job.start_cycle,
            job.finish_cycle,
            args={
              "event_id": job.event_id,
              "bytes": job.bytes_total,
              "l2_slot": job.l2_slot,
              "channel": job.channel,
            },
          )
          tr.instant("TileGroup", thread, "dma_complete", cycle,
                     {"event_id": job.event_id, "channel": job.channel})
      else:
        remaining_dma.append(job)
    self._dma_jobs = remaining_dma

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

    # 2b. tick NoC router (full_memory only)
    if self.memory_enabled:
      self.noc.step(cycle)

    # 3. tick all active Tile Group Sequencers
    for seq in self._active_sequencers:
      seq.step(cycle)

    # 4. tick all compute tiles; collect role completions by event id
    for t in self.tiles:
      t.step(cycle)
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
          self._aggregate_pmu()
          return True
        if term.status != "done" or term.role_event_id is None:
          continue
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
    # full_memory: aggregate NoC + L2 + payload PMU as deltas.
    # These component counters are cumulative, so we record the delta since
    # last aggregation and then reset — same pattern as engine/queue PMU.
    if self.memory_enabled:
      self.pmu.add_cycle("noc_switch_contention",
                         self.noc.pmu_switch_contention)
      self.pmu.add_cycle("l2_bank_conflict",
                         self.l2_sram.pmu_bank_conflict_cycles)
      self.pmu.add_cycle("l2_capacity_faults",
                         self.l2_sram.pmu_capacity_fault_count)
      self.pmu.add_cycle("payload_layout_faults",
                         self.payload.layout_fault_count)
      self.pmu.add_event("noc_vc0_stall",
                         self.noc.vcs[0].stall_cycles)
      self.pmu.add_event("noc_vc2_stall",
                         self.noc.vcs[2].stall_cycles)
      # reset component counters so next cycle records only the delta
      self.noc.pmu_switch_contention = 0
      self.noc.vcs[0].stall_cycles = 0
      self.noc.vcs[2].stall_cycles = 0
      self.l2_sram.pmu_bank_conflict_cycles = 0
      self.l2_sram.pmu_capacity_fault_count = 0
      self.payload.layout_fault_count = 0

  # ---- lifecycle ------------------------------------------------------

  def load_task(self, task: ExecTileGroupTask) -> None:
    # reset everything
    for t in self.tiles:
      t.reset()
    self.sequencer.reset()
    self._active_sequencers = [self.sequencer]
    self.queues.clear()
    self._channel_free_cycle.clear()
    self._dma_jobs.clear()
    self._collective_jobs.clear()
    self._role_event_tile_mask.clear()
    self._role_done_tiles.clear()
    self._role_trace.clear()
    self._role_phase_events.clear()
    # runtime-level: clear event/fault tables, but PRESERVE program residency
    # so warm launch works across tasks.  _registered_programs and
    # program_table are intentionally NOT cleared here — a program installed
    # in a prior task stays TILE_RESIDENT, so the next dispatch hits warm.
    if self.runtime_enabled:
      self.event_table.clear()
      self.fault_ring.reset()
      self.reset_domain.reset()
    if self.memory_enabled:
      self.l2_sram.reset()
      self.noc.reset()
      self.payload.reset()
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
    self.sequencer.load(task)
    # pre-init streams declared in the task (some tasks init inline)
    for s in task.streams:
      self.init_stream(s)

  def load_context_task(self, task: ExecTileGroupTask,
                        slot_index: int = 0) -> TileGroupSequencer:
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
    seq = TileGroupSequencer(self)
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
    self._channel_free_cycle.clear()
    self._dma_jobs.clear()
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
    if self.runtime_enabled:
      self.event_table.clear()
      self.fault_ring.reset()
      self.program_table.reset()
      self.reset_domain.reset()
    if self.memory_enabled:
      self.l2_sram.reset()
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
      "dma_jobs": len(self._dma_jobs),
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
