"""ELENOR engine timing models.

Each engine (BOA/EVU/MFE/USE) is a cycle-accurate latency model derived from
the hardware config and the descriptor's `ops` / `bytes` fields.

Timing derivations follow the Roofline + per-engine models in
design/ELENOR_Architecture_Design_v1.md section 21:

  BOA_perf = min(BOA_peak, SRAM_bw * AI_sram, HBM_bw * AI_hbm)
  EVU      : vector FMA throughput (lanes * 2 ops/cycle)
  MFE      : bandwidth-bound (bytes / mfe_bandwidth)
  USE      : state ops on the small control core

V1: BOA/EVU/USE are non-pipelined (one job at a time; UCE blocks on
`is_busy`).  MFE is channelized (design/elenor_mfe §3.1.4):
`mfe_load_channels` load lanes plus `mfe_store_channels` store lanes,
each lane an independent serial resource with per-lane descriptor-accept
queuing (`mfe_pipeline_depth` per lane).  Launch routes store-class ops
(store/dma_store) to store lanes and everything else to load lanes,
assigning first-free within the class, so load and store lanes run in
parallel.  This keeps the double-buffered prefetch pattern in tile
programs issuing back-to-back loads without UCE stalls.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .config import HardwareConfig
from .execution_ir import ExecEngineDesc
from .memory.transfer import MemoryTransaction, TransferStatus
from .pmu import PMUCounter, StallReason
from .trace import Tracer


class EngineState(Enum):
    IDLE = 0
    RUNNING = 1
    DONE = 2
    FAULTED = 3


@dataclass
class EngineJob:
    """One in-flight or queued engine descriptor."""

    desc: ExecEngineDesc
    start_cycle: int       # actual service-start (may be later than UCE launch)
    finish_cycle: int
    event_id: str
    pmu: PMUCounter = field(default_factory=PMUCounter)


class Engine:
    """Base engine.

    pipeline_depth = 1  → non-pipelined (original V1).  UCE blocks on is_busy
                          when a job is running.
    pipeline_depth > 1  → accept up to this many total jobs (running + queued).
                          UCE blocks only when the queue is full.  Jobs still
                          execute one at a time on the single resource.
    MFEEngine overrides this queueing with per-channel lanes (see below).
    """

    kind: str = "BASE"

    def __init__(self,
                 cfg: HardwareConfig,
                 tile_id: int,
                 tracer: Tracer | None = None,
                 pipeline_depth: int = 1):
        self.cfg = cfg
        self.tile_id = tile_id
        self.pmu = PMUCounter()
        self.state = EngineState.IDLE
        self.tracer = tracer
        self._pipeline_depth = pipeline_depth
        self._running: EngineJob | None = None
        self._queue: deque[EngineJob] = deque()

    def latency(self, desc: ExecEngineDesc) -> int:
        raise NotImplementedError

    @property
    def is_busy(self) -> bool:
        """True → UCE must retry launch next cycle.

        depth=1:  busy while a job is running.
        depth>1:  busy when total accepted (running + queued) >= depth.
        """
        if self._pipeline_depth == 1:
            return self._running is not None
        accepted = len(self._queue) + (1 if self._running else 0)
        return accepted >= self._pipeline_depth

    def launch(self, desc: ExecEngineDesc, cycle: int,
               event_id: str, transaction=None) -> object | None:
        """Launch a descriptor.  Returns None if the engine cannot accept
        (queue full); the caller retries next cycle.

        For pipelined engines the returned job may not start immediately —
        ``start_cycle`` reflects actual service-start, chained after
        earlier jobs.
        """
        if self.is_busy:
            return None
        lat = self.latency(desc)
        # service-start chains from the tail of existing work
        tail = self._queue[-1] if self._queue else self._running
        service_start = max(cycle, tail.finish_cycle if tail else cycle)
        job = EngineJob(desc=desc,
                        start_cycle=service_start,
                        finish_cycle=service_start + lat,
                        event_id=event_id,
                        pmu=PMUCounter())
        self._queue.append(job)
        if self._running is None:
            self._start_next()
        return job

    def _start_next(self) -> None:
        """Pop the queue head and begin servicing it."""
        if not self._queue:
            self.state = EngineState.IDLE
            self._running = None
            return
        self._running = self._queue.popleft()
        self.state = EngineState.RUNNING
        self.pmu.add_event("launch")
        if self.tracer is not None:
            self.tracer.complete(
                f"Tile{self.tile_id}",
                self.kind,
                f"{self.kind}:{self._running.desc.op}",
                self._running.start_cycle,
                self._running.finish_cycle,
                args={
                    "event_id": self._running.event_id,
                    "ops": self._running.desc.params.get("ops", 0),
                    "bytes": self._running.desc.params.get("bytes", 0),
                    "desc": self._running.desc.name,
                    "tile_id": self.tile_id,
                    "ctx_id": self._running.desc.params.get("ctx_id"),
                    "program": self._running.desc.params.get("program"),
                    "local_event_id": self._running.desc.params.get("local_event_id"),
                })

    def tick(self, cycle: int) -> list[EngineJob]:
        """Advance one cycle; return the jobs that just completed (0 or 1)."""
        active_key = f"{self.kind.lower()}_active"
        idle_key = f"{self.kind.lower()}_idle"
        if self._running is not None and cycle >= self._running.finish_cycle:
            done = self._running
            self.pmu.add_event("complete")
            self.pmu.add_cycle(active_key, 1)
            self.pmu.add(StallReason.NONE, 1)
            self.pmu.add_cycle("total", 1)
            self._start_next()
            return [done]
        if self._running is not None:
            self.pmu.add_cycle(active_key, 1)
            self.pmu.add(StallReason.NONE, 1)
            self.pmu.add_cycle("total", 1)
        else:
            self.pmu.add_cycle(idle_key, 1)
            self.pmu.add_cycle("total", 1)
        return []

    def reset(self) -> None:
        self._running = None
        self._queue.clear()
        self.state = EngineState.IDLE
        self.pmu.reset()


class BOAEngine(Engine):
    """Block Outer-product Accelerator — dense compute.

    latency = launch_overhead + ceil(ops / peak_macs)
    peak_macs = num_opa * opa_rows * opa_cols (MACs/cycle).
    """

    kind = "BOA"

    def latency(self, desc: ExecEngineDesc) -> int:
        ops = desc.params.get("ops", 0)
        macs = ops // 2 if ops else 0
        peak_macs = (self.cfg.boa_num_opa * self.cfg.boa_opa_rows *
                     self.cfg.boa_opa_cols)
        compute = (macs + peak_macs - 1) // peak_macs if peak_macs else 0
        bytes_per_op = desc.params.get("bytes", 0)
        sram_bw_bytes_per_cycle = self.cfg.tile_l1_bandwidth_gbs * 1e9 / (
            self.cfg.clock_mhz * 1e6)
        bw_cycles = 0
        if sram_bw_bytes_per_cycle > 0 and bytes_per_op > 0:
            bw_cycles = (bytes_per_op + sram_bw_bytes_per_cycle -
                         1) // sram_bw_bytes_per_cycle
        return self.cfg.boa_launch_cycles + max(compute, bw_cycles)


class EVUEngine(Engine):
    """Enhanced Vector Unit (EVU-MT) — irregular/vector compute.

    latency = launch_overhead + ceil(ops / (lanes * 2))
    """

    kind = "EVU"

    def latency(self, desc: ExecEngineDesc) -> int:
        ops = desc.params.get("ops", 0)
        peak = self.cfg.evu_lanes * 2
        compute = (ops + peak - 1) // peak if peak else 0
        return self.cfg.evu_launch_cycles + compute


class _MFELane:
    """One serial MFE service lane (a single load or store channel).

    A lane is an independent chained-service resource: a job accepted
    while another is running starts at ``max(cycle, tail.finish_cycle)``.
    PMU and trace events stay aggregated in the owning MFEEngine.
    """

    def __init__(self, name: str, depth: int):
        self.name = name  # trace track name, e.g. "MFE_LD0" / "MFE_ST0"
        self.depth = depth  # per-lane descriptor-accept queue depth
        self.running: _MFETransferJob | None = None
        self.queue: deque[_MFETransferJob] = deque()

    def accepted(self) -> int:
        return len(self.queue) + (1 if self.running else 0)


@dataclass
class _MFETransferJob:
    """One in-flight or queued MFE descriptor with a real transfer.

    ``transaction`` is submitted to the shared ``TransferManager`` only
    when this job becomes the lane head (``start_cycle`` set).  Until
    then the lane just queues the job without occupying memory
    resources.
    """

    desc: ExecEngineDesc
    event_id: str
    transaction: MemoryTransaction | None
    enqueue_cycle: int
    start_cycle: int | None = None


class MFEEngine(Engine):
    """Memory Flow Engine — bandwidth-bound stream shaping.

    PR 2 (§5.5): tile load/store go through the shared ``TransferManager``
    as real ``MemoryTransaction``s.  Lane heads submit on start; the
    engine polls ``TransferManager.status()`` each ``tick`` and constructs
    an ``EngineJob`` with the transaction's actual start/completion cycle
    when the local route finishes.
    """

    kind = "MFE"
    _STORE_OPS = ("store", "dma_store")

    def __init__(self, cfg: HardwareConfig, tile_id: int,
                 tracer: Tracer | None = None,
                 transfer_manager=None):
        self.cfg = cfg
        self.tile_id = tile_id
        self.pmu = PMUCounter()
        self.tracer = tracer
        self.transfer_manager = transfer_manager
        self._load_lanes = [
            _MFELane(f"MFE_LD{i}", cfg.mfe_pipeline_depth)
            for i in range(cfg.mfe_load_channels)
        ]
        self._store_lanes = [
            _MFELane(f"MFE_ST{j}", cfg.mfe_pipeline_depth)
            for j in range(cfg.mfe_store_channels)
        ]

    @property
    def _lanes(self) -> list[_MFELane]:
        return self._load_lanes + self._store_lanes

    @property
    def state(self) -> EngineState:
        if any(lane.running is not None for lane in self._lanes):
            return EngineState.RUNNING
        return EngineState.IDLE

    @state.setter
    def state(self, _value: EngineState) -> None:
        pass

    @property
    def is_busy(self) -> bool:
        return all(lane.accepted() >= lane.depth for lane in self._lanes)

    def launch(self, desc: ExecEngineDesc, cycle: int,
               event_id: str,
               transaction: MemoryTransaction | None = None,
               ) -> EngineJob | _MFETransferJob | None:
        """Route a descriptor to a first-free lane of its direction class.

        Returns None when every lane of that class is full.  Raises
        ``ValueError`` on page-stream prefetch-capacity violations.
        ``MemoryInvariantError`` from ``TransferManager.submit`` propagates
        to ``TileUCE._drain_engine_queues`` which faults the context.
        """
        self._validate_stream_buffer(desc)
        lanes = (self._store_lanes if desc.op in self._STORE_OPS
                 else self._load_lanes)
        free = next((lane for lane in lanes if lane.running is None), None)
        if free is None:
            free = next(
                (lane for lane in lanes if lane.accepted() < lane.depth),
                None)
            if free is None:
                return None
        job = _MFETransferJob(
            desc=desc, event_id=event_id, transaction=transaction,
            enqueue_cycle=cycle)
        free.queue.append(job)
        if free.running is None:
            self._start_lane(free, cycle)
        # Return a non-None sentinel so TileUCE pops the FIFO entry.
        # Actual completion is reported via tick().
        return job

    def _start_lane(self, lane: _MFELane, cycle: int) -> None:
        """Pop the lane's queue head and begin servicing it."""
        if not lane.queue:
            return
        job = lane.queue.popleft()
        job.start_cycle = cycle
        lane.running = job
        self.pmu.add_event("launch")
        if job.transaction is not None and self.transfer_manager is not None:
            self.transfer_manager.submit(job.transaction, cycle, self.pmu)

    def tick(self, cycle: int,
             start_queued: bool = True) -> list[EngineJob]:
        """Advance every lane one cycle; return jobs whose local route
        finished this cycle.  ``start_queued=False`` drains only the
        running head and must not submit the next queued transfer."""
        completed: list[EngineJob] = []
        active = 0
        for lane in self._lanes:
            job = lane.running
            if job is None:
                continue
            active += 1
            done = False
            if job.transaction is not None and self.transfer_manager is not None:
                status = self.transfer_manager.status(
                    job.transaction.transaction_id)
                if status == TransferStatus.DONE:
                    done = True
            elif job.start_cycle is not None:
                # no transaction (BOA-class fallback path): finish next cycle
                if cycle > job.start_cycle:
                    done = True
            if done:
                self.pmu.add_event("complete")
                start = (job.transaction.start_cycle
                         if job.transaction is not None
                         and job.transaction.start_cycle >= 0
                         else (job.start_cycle or cycle))
                finish = (job.transaction.completed_cycle
                          if job.transaction is not None
                          and job.transaction.completed_cycle >= 0
                          else cycle)
                ej = EngineJob(
                    desc=job.desc, start_cycle=start, finish_cycle=finish,
                    event_id=job.event_id, pmu=PMUCounter())
                completed.append(ej)
                if (job.transaction is not None
                        and self.transfer_manager is not None):
                    self.transfer_manager.acknowledge(
                        job.transaction.transaction_id)
                lane.running = None
                if start_queued:
                    self._start_lane(lane, cycle)
                if self.tracer is not None:
                    self.tracer.complete(
                        f"Tile{self.tile_id}", lane.name,
                        f"MFE:{job.desc.op}", start, finish,
                        args={"event_id": job.event_id,
                              "bytes": job.desc.params.get("bytes", 0),
                              "desc": job.desc.name,
                              "tile_id": self.tile_id})
        if active:
            self.pmu.add_cycle("mfe_active", 1)
            self.pmu.add(StallReason.NONE, 1)
        else:
            self.pmu.add_cycle("mfe_idle", 1)
        self.pmu.add_cycle("mfe_channel_active", active)
        self.pmu.add_cycle("total", 1)
        return completed

    def reset(self) -> None:
        for lane in self._lanes:
            lane.running = None
            lane.queue.clear()
        self.pmu.reset()

    def _validate_stream_buffer(self, desc: ExecEngineDesc) -> None:
        if self.cfg.mfe_stream_buffer_bytes == 0:
            return
        if desc.op != "page_stream":
            return
        if "prefetch_depth" not in desc.params:
            return
        num_pages = int(desc.params["num_pages"])
        total_bytes = int(desc.params["bytes"])
        prefetch_depth = int(desc.params["prefetch_depth"])
        if num_pages <= 0:
            raise ValueError(
                "MFE page_stream num_pages must be > 0 for buffer validation")
        page_bytes = (total_bytes + num_pages - 1) // num_pages
        required_bytes = prefetch_depth * page_bytes
        if required_bytes > self.cfg.mfe_stream_buffer_bytes:
            raise ValueError(
                f"MFE page_stream prefetch requires {required_bytes} bytes, "
                f"exceeds mfe_stream_buffer_bytes="
                f"{self.cfg.mfe_stream_buffer_bytes}")

class USEEngine(Engine):
    """Unified State Engine — scan/recurrence on a small control core.

    Modelled at the slower USE clock; latency scales by the clock ratio.
    """

    kind = "USE"

    def latency(self, desc: ExecEngineDesc) -> int:
        ops = desc.params.get("ops", 0)
        ratio = self.cfg.use_clock_mhz / self.cfg.clock_mhz
        cycles = (ops / ratio) if ratio else 0
        return self.cfg.use_launch_cycles + int(cycles)
