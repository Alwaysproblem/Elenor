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
from .execution_ir import (
    ExecEngineDesc,
    ExecGatherDesc,
    ExecGatherOutcome,
    ExecProfiledAccess,
)
from .memory import (
    DeterministicLRUCache,
    MemoryOwner,
    MshrTable,
    MshrWait,
    ResolvedMemoryView,
    slice_resolved_view,
)
from .memory.transfer import MemoryTransaction, TransferOp, TransferStatus
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



@dataclass
class _MFEGatherRequest:
    access: ExecProfiledAccess
    ordinal: int
    state: str = "LOOKUP"
    transaction_id: str | None = None
    l1_mshr_token: int | None = None
    l2_mshr_token: int | None = None
    wait_version: int | None = None
    response_ready: bool = False
    merged_counted: bool = False


@dataclass
class _MFEGatherJob:
    desc: ExecEngineDesc
    event_id: str
    source: ResolvedMemoryView | None
    indices: ResolvedMemoryView | None
    destination: ResolvedMemoryView | None
    issuer: MemoryOwner
    namespace: tuple[int, int, int, int, str]
    requests: list[_MFEGatherRequest]
    offsets: tuple[int, ...]
    start_cycle: int
    next_write_ordinal: int = 0
    write_transaction_id: str | None = None
    transaction_ids: set[str] = field(default_factory=set)

class MFEEngine(Engine):
    """Memory Flow Engine with load/store lanes and profiled Gather jobs."""

    kind = "MFE"
    _STORE_OPS = ("store", "dma_store")

    def __init__(
        self,
        cfg: HardwareConfig,
        tile_id: int,
        tracer: Tracer | None = None,
        transfer_manager=None,
        l1_cache: DeterministicLRUCache | None = None,
        l1_mshr: MshrTable | None = None,
        l2_cache: DeterministicLRUCache | None = None,
        l2_mshr: MshrTable | None = None,
        memory_trace=None,
    ):
        self.cfg = cfg
        self.tile_id = tile_id
        self.pmu = PMUCounter()
        self.tracer = tracer
        self.transfer_manager = transfer_manager
        self.memory_trace = memory_trace
        self.l1_cache = (
            l1_cache
            if l1_cache is not None
            else DeterministicLRUCache(
                cfg.l1_cache_capacity_bytes,
                cfg.cache_line_bytes,
            )
        )
        self.l1_mshr = l1_mshr if l1_mshr is not None else MshrTable(cfg.l1_mshr_entries)
        self.l2_cache = (
            l2_cache
            if l2_cache is not None
            else DeterministicLRUCache(
                cfg.l2_cache_capacity_bytes,
                cfg.cache_line_bytes,
            )
        )
        self.l2_mshr = l2_mshr if l2_mshr is not None else MshrTable(cfg.l2_mshr_entries)
        self._load_lanes = [
            _MFELane(f"MFE_LD{i}", cfg.mfe_pipeline_depth)
            for i in range(cfg.mfe_load_channels)
        ]
        self._store_lanes = [
            _MFELane(f"MFE_ST{j}", cfg.mfe_pipeline_depth)
            for j in range(cfg.mfe_store_channels)
        ]
        self._gather_jobs: dict[str, _MFEGatherJob] = {}
        self._current_cycle = 0

    def _emit_memory_trace(self, cycle: int) -> None:
        """Push L1/L2 cache + MSHR stats through the memory trace sink.

        Change-only sampling in the sink makes this per-transition call
        cheap; cache.py/mshr.py stay pure components.
        """
        if self.memory_trace is None:
            return
        self.memory_trace.cache("l1", self.tile_id, self.l1_cache.stats, cycle)
        self.memory_trace.mshr("l1", self.tile_id, self.l1_mshr.stats, cycle)
        self.memory_trace.cache("l2", self.tile_id, self.l2_cache.stats, cycle)
        self.memory_trace.mshr("l2", self.tile_id, self.l2_mshr.stats, cycle)

    @property
    def _lanes(self) -> list[_MFELane]:
        return self._load_lanes + self._store_lanes

    @property
    def state(self) -> EngineState:
        if self._gather_jobs or any(lane.running is not None for lane in self._lanes):
            return EngineState.RUNNING
        return EngineState.IDLE

    @state.setter
    def state(self, _value: EngineState) -> None:
        pass

    @property
    def is_busy(self) -> bool:
        lanes_full = all(lane.accepted() >= lane.depth for lane in self._lanes)
        gather_capacity = self.cfg.mfe_load_channels * self.cfg.mfe_pipeline_depth
        return lanes_full and len(self._gather_jobs) >= gather_capacity

    def launch(
        self,
        desc: ExecEngineDesc,
        cycle: int,
        event_id: str,
        transaction: MemoryTransaction | None = None,
    ) -> EngineJob | _MFETransferJob | None:
        """Route a descriptor to a first-free lane of its direction class."""
        self._validate_stream_buffer(desc)
        lanes = self._store_lanes if desc.op in self._STORE_OPS else self._load_lanes
        free = next((lane for lane in lanes if lane.running is None), None)
        if free is None:
            free = next(
                (lane for lane in lanes if lane.accepted() < lane.depth),
                None,
            )
            if free is None:
                return None
        job = _MFETransferJob(
            desc=desc,
            event_id=event_id,
            transaction=transaction,
            enqueue_cycle=cycle,
        )
        free.queue.append(job)
        if free.running is None:
            self._start_lane(free, cycle)
        return job

    def launch_gather(
        self,
        desc: ExecEngineDesc,
        cycle: int,
        event_id: str,
        *,
        source: ResolvedMemoryView | None,
        indices: ResolvedMemoryView | None,
        destination: ResolvedMemoryView | None,
        issuer: MemoryOwner,
        namespace: tuple[int, int, int, int, str],
    ) -> _MFEGatherJob | None:
        """Accept one Gather job and issue all profiled lookups concurrently."""
        capacity = self.cfg.mfe_load_channels * self.cfg.mfe_pipeline_depth
        if len(self._gather_jobs) >= capacity:
            return None
        if event_id in self._gather_jobs:
            raise ValueError("duplicate gather event id")
        gather = desc.params.get("gather")
        if not isinstance(gather, ExecGatherDesc):
            raise ValueError("gather descriptor is missing")
        if self.transfer_manager is None:
            raise ValueError("gather requires a TransferManager")

        offsets: list[int] = []
        offset = 0
        requests: list[_MFEGatherRequest] = []
        for ordinal, access in enumerate(gather.accesses):
            offsets.append(offset)
            offset += access.bytes
            requests.append(_MFEGatherRequest(access=access, ordinal=ordinal))
        job = _MFEGatherJob(
            desc=desc,
            event_id=event_id,
            source=source,
            indices=indices,
            destination=destination,
            issuer=issuer,
            namespace=namespace,
            requests=requests,
            offsets=tuple(offsets),
            start_cycle=cycle,
        )
        self._gather_jobs[event_id] = job
        for metric in (
            "gather_requests",
            "gather_l1_hits",
            "gather_l2_hits",
            "gather_hbm_misses",
            "gather_mshr_merges",
            "gather_mshr_stalls",
            "gather_reorder_wait_cycles",
            "gather_bytes",
        ):
            self.pmu.add_event(metric, 0)
        self.pmu.add_event("launch")
        self.pmu.add_event("gather_requests", len(requests))
        self.pmu.add_event("gather_bytes", gather.result_bytes)
        for request in requests:
            if request.access.outcome is ExecGatherOutcome.L1_HIT:
                self.pmu.add_event("gather_l1_hits")
                op = TransferOp.GATHER_L1_HIT
            elif request.access.outcome is ExecGatherOutcome.L2_HIT:
                self.pmu.add_event("gather_l2_hits")
                op = TransferOp.GATHER_L2_HIT
            else:
                self.pmu.add_event("gather_hbm_misses")
                op = TransferOp.GATHER_MISS_LOOKUP
            self._submit_gather_transaction(
                job,
                request,
                op,
                cycle,
                phase="lookup",
            )
        return job

    def _start_lane(self, lane: _MFELane, cycle: int) -> None:
        if not lane.queue:
            return
        job = lane.queue.popleft()
        job.start_cycle = cycle
        lane.running = job
        self.pmu.add_event("launch")
        if job.transaction is not None and self.transfer_manager is not None:
            self.transfer_manager.submit(job.transaction, cycle, self.pmu)

    def _transaction_id(
        self,
        job: _MFEGatherJob,
        request: _MFEGatherRequest,
        phase: str,
    ) -> str:
        prefix = ":".join(str(value) for value in job.namespace)
        return f"{prefix}:gather:{request.ordinal}:{phase}"

    def _submit_gather_transaction(
        self,
        job: _MFEGatherJob,
        request: _MFEGatherRequest,
        op: TransferOp,
        cycle: int,
        *,
        phase: str,
        src: ResolvedMemoryView | None = None,
        dst: ResolvedMemoryView | None = None,
    ) -> None:
        assert self.transfer_manager is not None
        transaction_id = self._transaction_id(job, request, phase)
        transaction = MemoryTransaction(
            transaction_id=transaction_id,
            op=op,
            issuer=job.issuer,
            src=src,
            dst=dst,
            bytes_total=request.access.bytes,
            completion_event=job.event_id,
            tile_id=self.tile_id,
        )
        self.transfer_manager.submit(transaction, cycle, self.pmu)
        request.transaction_id = transaction_id
        request.state = phase.upper()
        job.transaction_ids.add(transaction_id)

    def _transaction_done(self, request: _MFEGatherRequest) -> bool:
        if request.transaction_id is None or self.transfer_manager is None:
            return False
        return self.transfer_manager.status(request.transaction_id) is TransferStatus.DONE

    def _acknowledge_request_transaction(self, request: _MFEGatherRequest) -> None:
        if request.transaction_id is None or self.transfer_manager is None:
            return
        self.transfer_manager.acknowledge(request.transaction_id)
        request.transaction_id = None

    def _note_merge(self, request: _MFEGatherRequest) -> None:
        if request.merged_counted:
            return
        request.merged_counted = True
        self.pmu.add_event("gather_mshr_merges")

    def _enter_mshr_wait(
        self,
        request: _MFEGatherRequest,
        state: str,
        wait: MshrWait,
    ) -> None:
        if request.state != state:
            self.pmu.add_event("gather_mshr_stalls")
        request.state = state
        request.wait_version = wait.version
        request.transaction_id = None

    def _mark_response_ready(
        self,
        job: _MFEGatherJob,
        request: _MFEGatherRequest,
        cycle: int,
    ) -> None:
        if request.response_ready:
            return
        if job.event_id not in self._gather_jobs:
            return
        request.response_ready = True
        request.state = "RESPONSE_READY"
        request.wait_version = None
        if self.tracer is not None:
            self.tracer.instant(
                f"Tile{self.tile_id}",
                "MFE",
                "gather_response",
                cycle,
                {
                    "request_id": request.access.request_id,
                    "ordinal": request.ordinal,
                    "outcome": request.access.outcome.value,
                    "event_id": job.event_id,
                },
            )

    def _invoke_callbacks(self, callbacks) -> None:
        for callback in callbacks:
            callback()

    def _try_l1_mshr(
        self,
        job: _MFEGatherJob,
        request: _MFEGatherRequest,
        cycle: int,
    ) -> None:
        allocation = self.l1_mshr.allocate(request.access.merge_group)
        if isinstance(allocation, MshrWait):
            self._enter_mshr_wait(request, "WAIT_L1_MSHR", allocation)
            return
        request.l1_mshr_token = allocation.token
        request.wait_version = None
        if not allocation.leader:
            self._note_merge(request)
            request.state = "WAIT_L1_FILL"
            self.l1_mshr.wait(
                allocation.token,
                lambda: self._mark_response_ready(
                    job,
                    request,
                    self._current_cycle,
                ),
            )
            return
        self._try_l2_mshr(job, request, cycle)

    def _try_l2_mshr(
        self,
        job: _MFEGatherJob,
        request: _MFEGatherRequest,
        cycle: int,
    ) -> None:
        allocation = self.l2_mshr.allocate(request.access.merge_group)
        if isinstance(allocation, MshrWait):
            self._enter_mshr_wait(request, "WAIT_L2_MSHR", allocation)
            return
        request.l2_mshr_token = allocation.token
        request.wait_version = None
        if not allocation.leader:
            self._note_merge(request)
            request.state = "WAIT_L2_FILL"
            self.l2_mshr.wait(
                allocation.token,
                lambda: self._submit_l2_refill(
                    job,
                    request,
                    self._current_cycle,
                ),
            )
            return
        source = slice_resolved_view(job.source, 0, request.access.bytes)
        self._submit_gather_transaction(
            job,
            request,
            TransferOp.GATHER_HBM_REFILL,
            cycle,
            phase="hbm_refill",
            src=source,
        )

    def _submit_l2_refill(
        self,
        job: _MFEGatherJob,
        request: _MFEGatherRequest,
        cycle: int,
    ) -> None:
        if job.event_id not in self._gather_jobs:
            return
        self._submit_gather_transaction(
            job,
            request,
            TransferOp.GATHER_L2_REFILL,
            cycle,
            phase="l2_refill",
        )

    def _tick_gather_request(
        self,
        job: _MFEGatherJob,
        request: _MFEGatherRequest,
        cycle: int,
    ) -> None:
        if request.state == "WAIT_L1_MSHR":
            if request.wait_version != self.l1_mshr.version:
                self._try_l1_mshr(job, request, cycle)
            return
        if request.state == "WAIT_L2_MSHR":
            if request.wait_version != self.l2_mshr.version:
                self._try_l2_mshr(job, request, cycle)
            return
        if request.state in (
            "WAIT_L1_FILL",
            "WAIT_L2_FILL",
            "RESPONSE_READY",
            "WRITE",
            "DONE",
        ):
            return
        if not self._transaction_done(request):
            return

        state = request.state
        self._acknowledge_request_transaction(request)
        token = request.access.line_token
        if state == "LOOKUP":
            if request.access.outcome is ExecGatherOutcome.L1_HIT:
                self.l1_cache.record_hit(token)
                self._mark_response_ready(job, request, cycle)
            elif request.access.outcome is ExecGatherOutcome.L2_HIT:
                self.l1_cache.record_miss()
                self.l2_cache.record_hit(token)
                self.l1_cache.refill(token)
                self._mark_response_ready(job, request, cycle)
            else:
                self.l1_cache.record_miss()
                self.l2_cache.record_miss()
                self._try_l1_mshr(job, request, cycle)
            return

        if state == "HBM_REFILL":
            self.l2_cache.refill(token)
            assert request.l2_mshr_token is not None
            callbacks = self.l2_mshr.complete(request.l2_mshr_token)
            request.l2_mshr_token = None
            self._invoke_callbacks(callbacks)
            self._submit_l2_refill(job, request, cycle)
            return

        if state == "L2_REFILL":
            self.l1_cache.refill(token)
            assert request.l1_mshr_token is not None
            callbacks = self.l1_mshr.complete(request.l1_mshr_token)
            request.l1_mshr_token = None
            self._mark_response_ready(job, request, cycle)
            self._invoke_callbacks(callbacks)

    def _tick_gather_materialization(
        self,
        job: _MFEGatherJob,
        cycle: int,
    ) -> EngineJob | None:
        if job.write_transaction_id is not None:
            assert self.transfer_manager is not None
            if self.transfer_manager.status(job.write_transaction_id) is TransferStatus.DONE:
                request = job.requests[job.next_write_ordinal]
                self.transfer_manager.acknowledge(job.write_transaction_id)
                request.transaction_id = None
                if self.tracer is not None:
                    self.tracer.instant(
                        f"Tile{self.tile_id}",
                        "MFE",
                        "gather_destination_write",
                        cycle,
                        {
                            "request_id": request.access.request_id,
                            "ordinal": request.ordinal,
                            "outcome": request.access.outcome.value,
                            "event_id": job.event_id,
                        },
                    )
                request.state = "DONE"
                job.write_transaction_id = None
                job.next_write_ordinal += 1

        if job.next_write_ordinal >= len(job.requests):
            self.pmu.add_event("complete")
            final_request = job.requests[-1]
            if self.tracer is not None:
                self.tracer.instant(
                    f"Tile{self.tile_id}",
                    "MFE",
                    "gather_done",
                    cycle,
                    {
                        "request_id": final_request.access.request_id,
                        "ordinal": final_request.ordinal,
                        "outcome": final_request.access.outcome.value,
                        "event_id": job.event_id,
                    },
                )
            return EngineJob(
                desc=job.desc,
                start_cycle=job.start_cycle,
                finish_cycle=cycle,
                event_id=job.event_id,
                pmu=PMUCounter(),
            )

        next_request = job.requests[job.next_write_ordinal]
        if job.write_transaction_id is None and next_request.response_ready:
            destination = slice_resolved_view(
                job.destination,
                job.offsets[next_request.ordinal],
                next_request.access.bytes,
            )
            self._submit_gather_transaction(
                job,
                next_request,
                TransferOp.GATHER_DEST_WRITE,
                cycle,
                phase="write",
                dst=destination,
            )
            job.write_transaction_id = next_request.transaction_id
            return None

        if (
            not next_request.response_ready
            and any(
                request.response_ready
                for request in job.requests[job.next_write_ordinal + 1 :]
            )
        ):
            self.pmu.add_event("gather_reorder_wait_cycles")
        return None

    def tick(
        self,
        cycle: int,
        start_queued: bool = True,
    ) -> list[EngineJob]:
        """Advance transfer lanes and all active Gather state machines."""
        self._current_cycle = cycle
        completed: list[EngineJob] = []
        active_lanes = 0
        for lane in self._lanes:
            job = lane.running
            if job is None:
                continue
            active_lanes += 1
            done = False
            if job.transaction is not None and self.transfer_manager is not None:
                done = (
                    self.transfer_manager.status(job.transaction.transaction_id)
                    is TransferStatus.DONE
                )
            elif job.start_cycle is not None and cycle > job.start_cycle:
                done = True
            if not done:
                continue
            self.pmu.add_event("complete")
            start = (
                job.transaction.start_cycle
                if job.transaction is not None and job.transaction.start_cycle >= 0
                else (job.start_cycle or cycle)
            )
            finish = (
                job.transaction.completed_cycle
                if job.transaction is not None and job.transaction.completed_cycle >= 0
                else cycle
            )
            completed.append(
                EngineJob(
                    desc=job.desc,
                    start_cycle=start,
                    finish_cycle=finish,
                    event_id=job.event_id,
                    pmu=PMUCounter(),
                )
            )
            if job.transaction is not None and self.transfer_manager is not None:
                self.transfer_manager.acknowledge(job.transaction.transaction_id)
            lane.running = None
            if start_queued:
                self._start_lane(lane, cycle)
            if self.tracer is not None:
                self.tracer.complete(
                    f"Tile{self.tile_id}",
                    lane.name,
                    f"MFE:{job.desc.op}",
                    start,
                    finish,
                    args={
                        "event_id": job.event_id,
                        "bytes": job.desc.params.get("bytes", 0),
                        "desc": job.desc.name,
                        "tile_id": self.tile_id,
                    },
                )

        gather_active = len(self._gather_jobs)
        finished_gathers: list[str] = []
        for event_id, gather_job in list(self._gather_jobs.items()):
            for request in gather_job.requests:
                self._tick_gather_request(gather_job, request, cycle)
            completion = self._tick_gather_materialization(gather_job, cycle)
            if completion is not None:
                completed.append(completion)
                finished_gathers.append(event_id)
            # PR 5: gather FSM transitions are the only writers of
            # cache/MSHR state — push stats after each job's tick
            # (change-only sampling keeps this constant-cost).
            self._emit_memory_trace(cycle)
        for event_id in finished_gathers:
            self._gather_jobs.pop(event_id, None)

        blocked_gathers = sum(
            1
            for gather_job in self._gather_jobs.values()
            if any(
                request.state in ("WAIT_L1_MSHR", "WAIT_L2_MSHR")
                for request in gather_job.requests
            )
        )
        if active_lanes or gather_active:
            self.pmu.add_cycle("mfe_active", 1)
            if blocked_gathers:
                self.pmu.add(StallReason.WAIT_MSHR, blocked_gathers)
            else:
                self.pmu.add(StallReason.NONE, 1)
        else:
            self.pmu.add_cycle("mfe_idle", 1)
        self.pmu.add_cycle("mfe_channel_active", active_lanes)
        self.pmu.add_cycle("mfe_gather_active", gather_active)
        self.pmu.add_cycle("total", 1)
        return completed

    def reset(self) -> None:
        for lane in self._lanes:
            lane.running = None
            lane.queue.clear()
        self._gather_jobs.clear()
        self.l1_cache.reset()
        self.l1_mshr.reset()
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
                "MFE page_stream num_pages must be > 0 for buffer validation"
            )
        page_bytes = (total_bytes + num_pages - 1) // num_pages
        required_bytes = prefetch_depth * page_bytes
        if required_bytes > self.cfg.mfe_stream_buffer_bytes:
            raise ValueError(
                f"MFE page_stream prefetch requires {required_bytes} bytes, "
                f"exceeds mfe_stream_buffer_bytes={self.cfg.mfe_stream_buffer_bytes}"
            )

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
