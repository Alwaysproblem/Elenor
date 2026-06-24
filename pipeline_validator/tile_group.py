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
    finish_cycle: int


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
        self._stage_tile_mask: dict[int, int] = {}
        self._stage_prog_idx: dict[int, int] = {}
        self.pmu = PMUCounter()
        # track which tiles belong to which stage for completion aggregation
        self._tile_stage: dict[int, int] = {}
        # stage completion: stage_id -> set of tile ids that finished
        self._stage_done_tiles: dict[int, set] = {}

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

    def schedule_dma(self, event_id: str, latency: int, cycle: int) -> None:
        self._dma_jobs.append(
            _DMAJob(event_id=event_id, finish_cycle=cycle + latency))

    def dispatch_stage(self, stage_id: int, tile_mask: int, prog_idx: int,
                       out_stream: int | None, in_stream: int | None,
                       cycle: int) -> None:
        """Load the stage's Tile Program onto the selected tiles and mark them."""
        self._stage_tile_mask[stage_id] = tile_mask
        self._stage_prog_idx[stage_id] = prog_idx
        self._stage_done_tiles.setdefault(stage_id, set())
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
        # 1. tick Group DMA jobs
        remaining_dma = []
        for job in self._dma_jobs:
            if cycle >= job.finish_cycle:
                self.region_seq.notify_event(job.event_id)
                self.pmu.add_event("dma_complete")
                if tr is not None:
                    tr.instant("TileGroup", "DMA", "dma_complete", cycle,
                               {"event_id": job.event_id})
            else:
                remaining_dma.append(job)
        self._dma_jobs = remaining_dma

        # 2. tick stream queues (PMU occupancy counters + trace counters)
        for q in self.queues.values():
            q.tick(cycle)
            if tr is not None:
                tr.counter(f"StreamQ{q.queue_id}", "occupancy", cycle,
                           q.occupancy, "tokens")
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
                        tr.instant(f"Tile{t.tile_id}", "UCE", "tile_done",
                                   cycle, {"stage_id": stage_id})
                    # when all tiles of a stage are done, fire the stage event
                    mask = self._stage_tile_mask.get(stage_id, 0)
                    expected = bin(mask).count("1")
                    if len(done_set) >= expected:
                        ev = self.region_seq._stage_events.get(stage_id)
                        if ev is not None:
                            self.region_seq.notify_event(ev)
                            if tr is not None:
                                tr.instant("TileGroup", "Region",
                                           "stage_complete", cycle, {
                                               "stage_id": stage_id,
                                               "event_id": ev
                                           })

        # 5. aggregate PMU
        self._aggregate_pmu()
        if self.region_seq.done and tr is not None:
            tr.instant("TileGroup", "Region", "region_done", cycle, {})
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
        self._stage_tile_mask.clear()
        self._stage_prog_idx.clear()
        self._tile_stage.clear()
        self._stage_done_tiles.clear()
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
        self.pmu.reset()

    # ---- inspection -----------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "region_done": self.region_seq.done,
            "region_pc": self.region_seq.pc,
            "queues": {qid: q.snapshot()
                       for qid, q in self.queues.items()},
            "tiles": [t.snapshot() for t in self.tiles],
            "dma_jobs": len(self._dma_jobs),
        }

    def all_tiles_done(self) -> bool:
        return all(t.done for t in self.tiles)

    def credit_invariants_hold(self) -> bool:
        return all(q.credit_invariant_holds() for q in self.queues.values())
