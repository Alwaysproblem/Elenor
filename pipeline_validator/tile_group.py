"""Tile Group: 1 Tile Group Sequencer + 4 Compute Tiles + Group SRAM + streams.

The Tile Group is the local data-reuse / synchronization unit
(design/elenor_tile_group/).  It owns the Stream Queues that connect
task roles and the Group DMA.  The simulator drives it cycle by cycle,
advancing the Tile Group Sequencer and every Compute Tile in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import HardwareConfig
from .ir import StreamDesc, TileGroupTask, TileRoleBinding
from .pmu import PMUCounter
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


class TileGroup:
  """One ELENOR Tile Group with 4 Compute Tiles."""

  def __init__(self, cfg: HardwareConfig, tracer: Tracer | None = None):
    self.cfg = cfg
    self.tracer = tracer
    self.tiles: list[ComputeTile] = [
      ComputeTile(i, cfg, tracer) for i in range(cfg.num_tiles)
    ]
    self.sequencer = TileGroupSequencer(self)
    self.queues: dict[int, StreamQueue] = {}
    self._dma_jobs: list[_DMAJob] = []
    # per-channel "free cycle" for serializing DMA jobs on each channel
    self._channel_free_cycle: dict[int, int] = {}
    self._collective_jobs: list[_CollectiveJob] = []
    self.pmu = PMUCounter()
    # role dispatch: role_id -> tile_mask of the active dispatch
    self._role_tile_mask: dict[int, int] = {}
    # role completion fan-in by event id: event_id -> set of tile ids done
    self._role_done_tiles: dict[str, set] = {}
    # role trace bookkeeping: event_id -> _RoleTrace
    self._role_trace: dict[str, _RoleTrace] = {}
    # which tile belongs to which role completion event id
    self._tile_role_event: dict[int, str] = {}
    # task trace bookkeeping
    self._task_trace_name: str | None = None
    self._task_start_cycle: int | None = None
    self._task_done_traced: bool = False

  # ---- setup ----------------------------------------------------------

  def init_stream(self, desc: StreamDesc) -> StreamQueue:
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
      ))

  def schedule_collective(
    self,
    desc_id: str,
    event_id: str,
    op: str,
    bytes_total: int,
    participant_mask: int,
    cycle: int,
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
      ))

  def dispatch_role(
    self,
    binding: TileRoleBinding,
    cycle: int,
    event_id: str | None = None,
  ) -> None:
    """Load the role's Tile Program onto the selected tiles and mark them."""
    role_id = binding.role_id
    tile_mask = binding.tile_mask
    self._role_tile_mask[role_id] = tile_mask
    ev = event_id or f"ev_role{role_id}"
    # overwrite (not setdefault) so a re-dispatched role_id in a loop
    # starts with fresh completion/trace state.
    self._role_done_tiles[ev] = set()
    # always clear stale trace metadata so a re-dispatch never inherits it
    self._role_trace.pop(ev, None)
    tr = self.tracer
    if tr is not None:
      tr.instant(
        "TileGroup",
        "TileRole",
        "tile_role_dispatch",
        cycle,
        {
          "role_id": role_id,
          "tile_mask": tile_mask,
          "program": binding.tile_program.name,
          "event_id": ev,
          "out_stream": binding.out_stream,
          "in_stream": binding.in_stream,
        },
      )
      self._role_trace[ev] = _RoleTrace(
        role_id=role_id,
        event_id=ev,
        start_cycle=cycle,
        tile_mask=tile_mask,
        out_stream=binding.out_stream,
        in_stream=binding.in_stream,
      )
    for t in self.tiles:
      if tile_mask & (1 << t.tile_id):
        t.load_program(binding.tile_program)
        t.role_id = role_id
        self._tile_role_event[t.tile_id] = ev
        # rebind streams for this role (program may use qid 0/1)
        for qid, q in self.queues.items():
          t.bind_stream(qid, q)

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
        self.sequencer.notify_event(job.event_id)
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
        self.sequencer.notify_event(cjob.event_id)
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

    # 3. tick the Tile Group Sequencer
    self.sequencer.step(cycle)

    # 4. tick all compute tiles; collect role completions by event id
    for t in self.tiles:
      t.step(cycle)
      if t.done and t.tile_id in self._tile_role_event:
        ev = self._tile_role_event[t.tile_id]
        done_set = self._role_done_tiles.setdefault(ev, set())
        if t.tile_id not in done_set:
          done_set.add(t.tile_id)
          if tr is not None:
            tr.instant(f"Tile{t.tile_id}", "UCE", "tile_done", cycle,
                       {"role_id": getattr(t, "role_id", None)})
          # when all tiles of a role are done, fire the role event
          mask = self._role_tile_mask.get(getattr(t, "role_id", -1), 0)
          expected = bin(mask).count("1")
          if len(done_set) >= expected:
            self.sequencer.notify_event(ev)
            if tr is not None:
              rt = self._role_trace.get(ev)
              if rt is not None:
                tr.complete(
                  "TileGroup",
                  "TileRole",
                  f"dispatch:role{rt.role_id}:{ev}:run",
                  rt.start_cycle,
                  cycle,
                  args={
                    "role_id": rt.role_id,
                    "event_id": ev,
                    "tile_mask": mask,
                    "out_stream": rt.out_stream,
                    "in_stream": rt.in_stream,
                  },
                )
              tr.instant(
                "TileGroup",
                "TileRole",
                "tile_role_complete",
                cycle,
                {
                  "role_id": getattr(t, "role_id", None),
                  "event_id": ev
                },
              )

    # 5. aggregate PMU
    self._aggregate_pmu()
    if self.sequencer.done and tr is not None:
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
    return self.sequencer.done

  def _aggregate_pmu(self) -> None:
    # merge sequencer + all tiles + all queues into group PMU
    self.pmu.merge(self.sequencer.pmu)
    self.sequencer.pmu.reset()
    for t in self.tiles:
      self.pmu.merge(t.pmu)
      t.pmu.reset()
    for q in self.queues.values():
      self.pmu.merge(q.pmu)
      q.pmu.reset()

  # ---- lifecycle ------------------------------------------------------

  def load_task(self, task: TileGroupTask) -> None:
    # reset everything
    for t in self.tiles:
      t.reset()
    self.sequencer.reset()
    self.queues.clear()
    self._channel_free_cycle.clear()
    self._dma_jobs.clear()
    self._collective_jobs.clear()
    self._role_tile_mask.clear()
    self._role_done_tiles.clear()
    self._role_trace.clear()
    self._tile_role_event.clear()
    self._task_trace_name = f"task:{task.name}"
    self._task_start_cycle = None
    self._task_done_traced = False
    self.pmu.reset()
    self.sequencer.load(task)
    # pre-init streams declared in the task (some tasks init inline)
    for s in task.streams:
      self.init_stream(s)

  def reset(self) -> None:
    for t in self.tiles:
      t.reset()
    self.sequencer.reset()
    for q in self.queues.values():
      q.reset()
    self._channel_free_cycle.clear()
    self._dma_jobs.clear()
    self._collective_jobs.clear()
    self._role_tile_mask.clear()
    self._role_done_tiles.clear()
    self._role_trace.clear()
    self._tile_role_event.clear()
    self._task_trace_name = None
    self._task_start_cycle = None
    self._task_done_traced = False
    self.pmu.reset()

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
