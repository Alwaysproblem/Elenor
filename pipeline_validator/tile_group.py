"""Tile Group: 1 Region Sequencer + 4 Compute Tiles + Group SRAM + streams.

The Tile Group is the local data-reuse / synchronization unit
(design/elenor_tile_group/).  It owns the Stream Queues that connect
pipeline stages and the Group DMA.  The simulator drives it cycle by
cycle, advancing the Region Sequencer and every Compute Tile in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import HardwareConfig
from .ir import RegionProgram, StreamDesc
from .pmu import PMUCounter
from .region import RegionSequencer
from .stream_queue import EOSPolicy, QueueKind, StreamQueue
from .tile import ComputeTile
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
class _StageTrace:
  """Bookkeeping for one dispatched stage's runtime window."""
  stage_id: int
  event_id: str
  start_cycle: int
  tile_mask: int
  prog_idx: int
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
    self.region_seq = RegionSequencer(self)
    self.queues: dict[int, StreamQueue] = {}
    self._dma_jobs: list[_DMAJob] = []
    self._collective_jobs: list[_CollectiveJob] = []
    self._stage_tile_mask: dict[int, int] = {}
    self._stage_prog_idx: dict[int, int] = {}
    self.pmu = PMUCounter()
    # track which tiles belong to which stage for completion aggregation
    self._tile_stage: dict[int, int] = {}
    # stage completion: stage_id -> set of tile ids that finished
    self._stage_done_tiles: dict[int, set] = {}
    # stage trace bookkeeping: stage_id -> _StageTrace
    self._stage_trace: dict[int, _StageTrace] = {}
    # region trace bookkeeping
    self._region_trace_name: str | None = None
    self._region_start_cycle: int | None = None
    self._region_done_traced: bool = False

  # ---- setup ----------------------------------------------------------

  def init_stream(self, desc: StreamDesc) -> StreamQueue:
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
  ) -> None:
    self._dma_jobs.append(
      _DMAJob(
        event_id=event_id,
        start_cycle=cycle,
        finish_cycle=cycle + latency,
        op=op,
        desc_id=desc_id,
        l2_slot=l2_slot,
        bytes_total=bytes_total,
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

  def dispatch_stage(
    self,
    stage_id: int,
    tile_mask: int,
    prog_idx: int,
    out_stream: int | None,
    in_stream: int | None,
    cycle: int,
    event_id: str | None = None,
  ) -> None:
    """Load the stage's Tile Program onto the selected tiles and mark them."""
    self._stage_tile_mask[stage_id] = tile_mask
    self._stage_prog_idx[stage_id] = prog_idx
    # overwrite (not setdefault) so a re-dispatched stage_id in a loop
    # starts with fresh completion/trace state.
    self._stage_done_tiles[stage_id] = set()
    # always clear stale trace metadata so a re-dispatch never inherits it
    self._stage_trace.pop(stage_id, None)
    tr = self.tracer
    if tr is not None and event_id is not None:
      tr.instant(
        "TileGroup",
        "Stage",
        "stage_dispatch",
        cycle,
        {
          "stage_id": stage_id,
          "tile_mask": tile_mask,
          "prog_idx": prog_idx,
          "event_id": event_id,
          "out_stream": out_stream,
          "in_stream": in_stream,
        },
      )
      self._stage_trace[stage_id] = _StageTrace(
        stage_id=stage_id,
        event_id=event_id,
        start_cycle=cycle,
        tile_mask=tile_mask,
        prog_idx=prog_idx,
        out_stream=out_stream,
        in_stream=in_stream,
      )
    assert self.region_seq.program is not None
    program = self.region_seq.program.tile_programs.get(prog_idx)
    if program is None:
      return
    for t in self.tiles:
      if tile_mask & (1 << t.tile_id):
        t.load_program(program)
        t.stage_id = stage_id
        self._tile_stage[t.tile_id] = stage_id
        # rebind streams for this stage (program may use qid 0/1)
        for qid, q in self.queues.items():
          t.bind_stream(qid, q)

  # ---- per-cycle step -------------------------------------------------

  def step(self, cycle: int) -> bool:
    """Advance one cycle.  Returns True if the whole region is done."""
    tr = self.tracer

    # 0. region trace: capture start cycle on first step
    if tr is not None and self._region_start_cycle is None:
      self._region_start_cycle = cycle

    # 1. tick Group DMA jobs
    remaining_dma: list[_DMAJob] = []
    for job in self._dma_jobs:
      if cycle >= job.finish_cycle:
        self.region_seq.notify_event(job.event_id)
        self.pmu.add_event("dma_complete")
        if tr is not None:
          tr.complete(
            "TileGroup",
            "Global DMA",
            f"{job.op}:{job.desc_id}",
            job.start_cycle,
            job.finish_cycle,
            args={
              "event_id": job.event_id,
              "bytes": job.bytes_total,
              "l2_slot": job.l2_slot,
            },
          )
          tr.instant("TileGroup", "Global DMA", "dma_complete", cycle,
                     {"event_id": job.event_id})
      else:
        remaining_dma.append(job)
    self._dma_jobs = remaining_dma

    # 1b. tick Collective jobs
    remaining_coll: list[_CollectiveJob] = []
    for cjob in self._collective_jobs:
      if cycle >= cjob.finish_cycle:
        self.region_seq.notify_event(cjob.event_id)
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

    # 3. tick region sequencer
    self.region_seq.step(cycle)

    # 4. tick all compute tiles; collect stage completions
    for t in self.tiles:
      t.step(cycle)
      if t.done and t.tile_id in self._tile_stage:
        stage_id = self._tile_stage[t.tile_id]
        done_set = self._stage_done_tiles.setdefault(stage_id, set())
        if t.tile_id not in done_set:
          done_set.add(t.tile_id)
          if tr is not None:
            tr.instant(f"Tile{t.tile_id}", "UCE", "tile_done", cycle,
                       {"stage_id": stage_id})
          # when all tiles of a stage are done, fire the stage event
          mask = self._stage_tile_mask.get(stage_id, 0)
          expected = bin(mask).count("1")
          if len(done_set) >= expected:
            ev = self.region_seq._stage_events.get(stage_id)
            if ev is not None:
              self.region_seq.notify_event(ev)
              if tr is not None:
                st = self._stage_trace.get(stage_id)
                if st is not None:
                  tr.complete(
                    "TileGroup",
                    "Stage",
                    f"stage{stage_id}:{ev}:run",
                    st.start_cycle,
                    cycle,
                    args={
                      "stage_id": stage_id,
                      "event_id": ev,
                      "tile_mask": mask,
                      "prog_idx": st.prog_idx,
                      "out_stream": st.out_stream,
                      "in_stream": st.in_stream,
                    },
                  )
                tr.instant(
                  "TileGroup",
                  "Stage",
                  "stage_complete",
                  cycle,
                  {
                    "stage_id": stage_id,
                    "event_id": ev
                  },
                )

    # 5. aggregate PMU
    self._aggregate_pmu()
    if self.region_seq.done and tr is not None:
      if not self._region_done_traced:
        start = self._region_start_cycle if self._region_start_cycle is not None else cycle
        if self._region_trace_name is not None:
          tr.complete(
            "TileGroup",
            "Region",
            self._region_trace_name,
            start,
            cycle,
            args={
              "region": self._region_trace_name.replace("region:", "", 1)
            },
          )
        tr.instant("TileGroup", "Region", "region_done", cycle, {})
        self._region_done_traced = True
    return self.region_seq.done

  def _aggregate_pmu(self) -> None:
    # merge region sequencer + all tiles + all queues into group PMU
    self.pmu.merge(self.region_seq.pmu)
    self.region_seq.pmu.reset()
    for t in self.tiles:
      self.pmu.merge(t.pmu)
      t.pmu.reset()
    for q in self.queues.values():
      self.pmu.merge(q.pmu)
      q.pmu.reset()

  # ---- lifecycle ------------------------------------------------------

  def load_region(self, program: RegionProgram) -> None:
    # reset everything
    for t in self.tiles:
      t.reset()
    self.region_seq.reset()
    self.queues.clear()
    self._dma_jobs.clear()
    self._collective_jobs.clear()
    self._stage_tile_mask.clear()
    self._stage_prog_idx.clear()
    self._tile_stage.clear()
    self._stage_done_tiles.clear()
    self._stage_trace.clear()
    self._region_trace_name = f"region:{program.name}"
    self._region_start_cycle = None
    self._region_done_traced = False
    self.pmu.reset()
    self.region_seq.load(program)
    # pre-init streams declared in the program (some regions init inline)
    for s in program.streams:
      self.init_stream(s)

  def reset(self) -> None:
    for t in self.tiles:
      t.reset()
    self.region_seq.reset()
    for q in self.queues.values():
      q.reset()
    self._dma_jobs.clear()
    self._collective_jobs.clear()
    self._stage_tile_mask.clear()
    self._stage_prog_idx.clear()
    self._tile_stage.clear()
    self._stage_done_tiles.clear()
    self._stage_trace.clear()
    self._region_trace_name = None
    self._region_start_cycle = None
    self._region_done_traced = False
    self.pmu.reset()

  # ---- inspection -----------------------------------------------------

  def snapshot(self) -> dict:
    return {
      "region_done": self.region_seq.done,
      "region_pc": self.region_seq.pc,
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
