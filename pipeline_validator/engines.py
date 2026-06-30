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
`is_busy`).  MFE supports descriptor-accept queuing (pipeline_depth >= 1):
the UCE may launch a new MFE job while an earlier job is still in flight;
jobs execute serially on the single MFE resource with chained service
start cycles.  This allows the double-buffered prefetch pattern in
tiled_matmul tile programs to issue back-to-back loads without UCE stalls.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .config import HardwareConfig
from .ir import EngineDesc
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

    desc: EngineDesc
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

    def latency(self, desc: EngineDesc) -> int:
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

    def launch(self, desc: EngineDesc, cycle: int,
               event_id: str) -> EngineJob | None:
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
                })

    def tick(self, cycle: int) -> EngineJob | None:
        """Advance one cycle; return the job if it just completed."""
        active_key = f"{self.kind.lower()}_active"
        idle_key = f"{self.kind.lower()}_idle"
        if self._running is not None and cycle >= self._running.finish_cycle:
            done = self._running
            self.pmu.add_event("complete")
            self.pmu.add_cycle(active_key, 1)
            self.pmu.add(StallReason.NONE, 1)
            self.pmu.add_cycle("total", 1)
            self._start_next()
            return done
        if self._running is not None:
            self.pmu.add_cycle(active_key, 1)
            self.pmu.add(StallReason.NONE, 1)
            self.pmu.add_cycle("total", 1)
        else:
            self.pmu.add_cycle(idle_key, 1)
            self.pmu.add_cycle("total", 1)
        return None

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

    def latency(self, desc: EngineDesc) -> int:
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

    def latency(self, desc: EngineDesc) -> int:
        ops = desc.params.get("ops", 0)
        peak = self.cfg.evu_lanes * 2
        compute = (ops + peak - 1) // peak if peak else 0
        return self.cfg.evu_launch_cycles + compute


class MFEEngine(Engine):
    """Memory Flow Engine — bandwidth-bound stream shaping.

    latency = launch_overhead + ceil(bytes / (mfe_bw_bytes_per_cycle))

    Supports descriptor-accept queuing (pipeline_depth from config)
    so the UCE can issue back-to-back MFE jobs without stalling.
    """

    kind = "MFE"

    def __init__(self, cfg: HardwareConfig, tile_id: int,
                 tracer: Tracer | None = None):
        super().__init__(cfg, tile_id, tracer,
                         pipeline_depth=cfg.mfe_pipeline_depth)

    def latency(self, desc: EngineDesc) -> int:
        nbytes = desc.params.get("bytes", 0)
        bw_bytes_per_cycle = self.cfg.mfe_bandwidth_gbs * 1e9 / (
            self.cfg.clock_mhz * 1e6)
        cycles = (nbytes + bw_bytes_per_cycle -
                  1) // bw_bytes_per_cycle if bw_bytes_per_cycle else 0
        return self.cfg.mfe_launch_cycles + cycles

    def launch(self, desc: EngineDesc, cycle: int,
               event_id: str) -> EngineJob | None:
        """Validate page-stream prefetch capacity before delegating to
        ``Engine.launch()``.  Raises ``ValueError`` when an explicit
        ``prefetch_depth`` exceeds the configured stream-buffer capacity;
        the tile launch path catches it and converts it into a modeled
        fault rather than crashing the simulation.
        """
        self._validate_stream_buffer(desc)
        return super().launch(desc, cycle, event_id)

    def _validate_stream_buffer(self, desc: EngineDesc) -> None:
        """Enforce page-stream prefetch capacity when the buffer size is
        frozen (non-zero).  Non-page-stream ops and descriptors without an
        explicit ``prefetch_depth`` are always accepted.
        """
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

    def latency(self, desc: EngineDesc) -> int:
        ops = desc.params.get("ops", 0)
        ratio = self.cfg.use_clock_mhz / self.cfg.clock_mhz
        cycles = (ops / ratio) if ratio else 0
        return self.cfg.use_launch_cycles + int(cycles)
