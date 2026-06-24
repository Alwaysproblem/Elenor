"""ELENOR engine timing models.

Each engine (BOA/EVU/MFE/USE) is a cycle-accurate latency model derived from
the hardware config and the descriptor's `ops` / `bytes` fields.

Timing derivations follow the Roofline + per-engine models in
design/ELENOR_Architecture_Design_v1.md section 21:

  BOA_perf = min(BOA_peak, SRAM_bw * AI_sram, HBM_bw * AI_hbm)
  EVU      : vector FMA throughput (lanes * 2 ops/cycle)
  MFE      : bandwidth-bound (bytes / mfe_bandwidth)
  USE      : state ops on the small control core

Engines are *non-pipelined* in V1 for simplicity: an engine issues one
descriptor at a time and the latency is ceil(ops / throughput).  The Tile
UCE models the overlap by issuing launches and waiting on events, so
memory (MFE/DMA) and compute (BOA/EVU) overlap naturally when the UCE
launches them before waiting.
"""

from __future__ import annotations

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
    """One in-flight engine descriptor with its completion cycle."""
    desc: EngineDesc
    start_cycle: int
    finish_cycle: int
    event_id: str
    pmu: PMUCounter = field(default_factory=PMUCounter)


class Engine:
    """Base engine: holds a config, a local PMU, and at most one running job."""

    kind: str = "BASE"

    def __init__(self,
                 cfg: HardwareConfig,
                 tile_id: int,
                 tracer: Tracer | None = None):
        self.cfg = cfg
        self.tile_id = tile_id
        self.pmu = PMUCounter()
        self.state = EngineState.IDLE
        self.job: EngineJob | None = None
        self.tracer = tracer

    # subclasses override
    def latency(self, desc: EngineDesc) -> int:
        raise NotImplementedError

    @property
    def is_busy(self) -> bool:
        return self.job is not None and self.state == EngineState.RUNNING

    def launch(self, desc: EngineDesc, cycle: int,
               event_id: str) -> EngineJob | None:
        """Launch a descriptor.  Returns None if the engine is busy (the
        caller should retry next cycle).  Non-pipelined V1: one job at a time.
        """
        if self.is_busy:
            return None
        lat = self.latency(desc)
        job = EngineJob(desc=desc,
                        start_cycle=cycle,
                        finish_cycle=cycle + lat,
                        event_id=event_id,
                        pmu=PMUCounter())
        self.state = EngineState.RUNNING
        self.job = job
        self.pmu.add_event("launch")
        if self.tracer is not None:
            self.tracer.complete(f"Tile{self.tile_id}",
                                 self.kind,
                                 f"{self.kind}:{desc.op}",
                                 cycle,
                                 cycle + lat,
                                 args={
                                     "event_id": event_id,
                                     "ops": desc.params.get("ops", 0),
                                     "bytes": desc.params.get("bytes", 0),
                                     "desc": desc.name,
                                     "tile_id": self.tile_id
                                 })
        return job

    def tick(self, cycle: int) -> EngineJob | None:
        """Advance one cycle; return the job if it just completed.

        Records exactly one counter per cycle:
          - <kind>_active  when busy (job in flight, not yet due)
          - <kind>_idle    when idle
        Completion is detected on the cycle the job is due.
        """
        active_key = f"{self.kind.lower()}_active"
        idle_key = f"{self.kind.lower()}_idle"
        if self.job is not None and cycle >= self.job.finish_cycle:
            done = self.job
            self.job = None
            self.state = EngineState.DONE
            self.pmu.add_event("complete")
            # the completion cycle counts as active (the job ran this cycle)
            self.pmu.add_cycle(active_key, 1)
            self.pmu.add(StallReason.NONE, 1)
            self.pmu.add_cycle("total", 1)
            return done
        if self.job is not None:
            self.pmu.add_cycle(active_key, 1)
            self.pmu.add(StallReason.NONE, 1)  # busy cycle
            self.pmu.add_cycle("total", 1)
        else:
            self.pmu.add_cycle(idle_key, 1)
            self.pmu.add_cycle("total", 1)
        return None

    def reset(self) -> None:
        self.job = None
        self.state = EngineState.IDLE
        self.pmu.reset()


class BOAEngine(Engine):
    """Block Outer-product Accelerator — dense compute.

    latency = launch_overhead + ceil(ops / peak_macs)
    where peak_macs = num_opa * opa_rows * opa_cols (MACs/cycle).
    """

    kind = "BOA"

    def latency(self, desc: EngineDesc) -> int:
        ops = desc.params.get("ops", 0)
        macs = ops // 2 if ops else 0
        peak_macs = (self.cfg.boa_num_opa * self.cfg.boa_opa_rows *
                     self.cfg.boa_opa_cols)
        compute = (macs + peak_macs - 1) // peak_macs if peak_macs else 0
        # operand bandwidth ceiling (A+B+acc read/write)
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
        peak = self.cfg.evu_lanes * 2  # FMA = 2 ops/lane/cycle
        compute = (ops + peak - 1) // peak if peak else 0
        return self.cfg.evu_launch_cycles + compute


class MFEEngine(Engine):
    """Memory Flow Engine — bandwidth-bound stream shaping.

    latency = launch_overhead + ceil(bytes / (mfe_bw_bytes_per_cycle))
    """

    kind = "MFE"

    def latency(self, desc: EngineDesc) -> int:
        nbytes = desc.params.get("bytes", 0)
        bw_bytes_per_cycle = self.cfg.mfe_bandwidth_gbs * 1e9 / (
            self.cfg.clock_mhz * 1e6)
        cycles = (nbytes + bw_bytes_per_cycle -
                  1) // bw_bytes_per_cycle if bw_bytes_per_cycle else 0
        return self.cfg.mfe_launch_cycles + cycles


class USEEngine(Engine):
    """Unified State Engine — scan/recurrence on a small control core.

    Modelled at the slower USE clock; latency scales by the clock ratio.
    """

    kind = "USE"

    def latency(self, desc: EngineDesc) -> int:
        ops = desc.params.get("ops", 0)
        # USE runs at a slower clock; convert to core cycles.
        ratio = self.cfg.use_clock_mhz / self.cfg.clock_mhz
        # one state op per USE cycle, expressed in core cycles
        cycles = (ops / ratio) if ratio else 0
        return self.cfg.use_launch_cycles + int(cycles)
