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
from .execution_ir import ExecEngineDesc, ExecTileGatherDesc, ExecTileScatterDesc
from .memory import (
  DeterministicLRUCache,
  MemoryInvariantError,
  MemoryOwner,
  MshrTable,
  MshrWait,
  ResolvedMemoryView,
  slice_resolved_view,
)
from .memory.address_provider import (
  decode_index_i32,
  resolve_indexed_segments,
)
from .memory.cache import CacheLineIdentity, CacheProvenance
from .memory.transfer import MemoryTransaction, TransferOp, TransferStatus
from .pmu import PMUCounter, StallReason
from .profiles import build_registry
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
  start_cycle: int  # actual service-start (may be later than UCE launch)
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

  def __init__(
    self, cfg: HardwareConfig, tile_id: int, tracer: Tracer | None = None, pipeline_depth: int = 1
  ):
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

  @property
  def accepted_count(self) -> int:
    """Jobs already accepted by this engine, running plus service queue."""
    return len(self._queue) + (1 if self._running is not None else 0)

  @property
  def has_accepted_jobs(self) -> bool:
    return self.accepted_count > 0

  def launch(self, desc: ExecEngineDesc, cycle: int, event_id: str, transaction=None) -> object | None:
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
    job = EngineJob(
      desc=desc,
      start_cycle=service_start,
      finish_cycle=service_start + lat,
      event_id=event_id,
      pmu=PMUCounter(),
    )
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
        },
      )

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
    peak_macs = self.cfg.boa_num_opa * self.cfg.boa_opa_rows * self.cfg.boa_opa_cols
    compute = (macs + peak_macs - 1) // peak_macs if peak_macs else 0
    bytes_per_op = desc.params.get("bytes", 0)
    sram_bw_bytes_per_cycle = self.cfg.tile_l1_bandwidth_gbs * 1e9 / (self.cfg.clock_mhz * 1e6)
    bw_cycles = 0
    if sram_bw_bytes_per_cycle > 0 and bytes_per_op > 0:
      bw_cycles = int((bytes_per_op + sram_bw_bytes_per_cycle - 1) // sram_bw_bytes_per_cycle)
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
  when this job becomes the lane head.  ``transaction_submitted`` is the
  authority for that boundary; queued jobs must never be queried through
  ``TransferManager.status`` because no transaction id is registered yet.
  """

  desc: ExecEngineDesc
  event_id: str
  transaction: MemoryTransaction | None
  enqueue_cycle: int
  start_cycle: int | None = None
  transaction_submitted: bool = False


@dataclass
class _GatherIndexSlot:
  """One index slot: from INDEX_READ issue to all its payload writes."""

  ordinal: int
  state: str = "INDEX_READ"  # INDEX_READ | DECODED | DONE
  transaction_id: str | None = None
  value: int | None = None
  issued: bool = False
  released: bool = False  # all segments written to destination


@dataclass
class _GatherSegmentRequest:
  """One cache-line-shaped payload request with a global ordinal.

  The request covers exactly one (remote, local) byte pair produced by
  ``resolve_indexed_segments``; when either cache level is enabled the
  remote slice is line-aligned (first/last lines may be partial, which
  turns the request into a bypass or an inline segment of a refill).
  """

  ordinal: int
  slot_ordinal: int
  remote: ResolvedMemoryView
  local: ResolvedMemoryView
  state: str = "LOOKUP_L1"  # LOOKUP_L1 | LOOKUP_L2 | WAIT_L1_MSHR | WAIT_L2_MSHR
  #  | WAIT_L1_FILL | WAIT_L2_FILL | HBM_REFILL | L1_DIRECT_REFILL
  #  | DIRECT_RESPONSE | L2_REFILL | RESPONSE_READY | DONE
  within_line: int = 0  # byte offset of this segment inside the line slice
  transaction_id: str | None = None
  l1_mshr_token: int | None = None
  l2_mshr_token: int | None = None
  wait_version: int | None = None
  response_ready: bool = False
  merged_counted: bool = False
  bypass: bool = False
  line_identity: CacheLineIdentity | None = None
  line_provenance: CacheProvenance | None = None
  line_data: bytes | None = None
  line_validity: bytes | None = None
  hit_data: bytes | None = None
  hit_validity: bytes | None = None


@dataclass
class _MFEGatherJob:
  desc: ExecEngineDesc
  event_id: str
  gather: ExecTileGatherDesc
  source: ResolvedMemoryView | None
  indices: ResolvedMemoryView | None
  destination: ResolvedMemoryView | None
  issuer: MemoryOwner
  namespace: tuple[int, int, int, int, str]
  start_cycle: int
  element_bytes: int = 1
  slots: list[_GatherIndexSlot] = field(default_factory=list)
  next_slot: int = 0
  next_request_ordinal: int = 0
  requests: list[_GatherSegmentRequest] = field(default_factory=list)
  next_write_ordinal: int = 0
  write_request: _GatherSegmentRequest | None = None
  write_transaction_id: str | None = None
  transaction_ids: set[str] = field(default_factory=set)
  binding_id: str | None = None
  bypass_forbidden: bool = False


@dataclass
class _ScatterIndexSlot:
  """One Scatter index slot: INDEX_READ issue to decoded segments."""

  ordinal: int
  state: str = "INDEX_READ"  # INDEX_READ | DECODED | DONE
  transaction_id: str | None = None
  value: int | None = None


@dataclass
class _ScatterSegment:
  """One (i, j) Scatter payload segment with ordered-overwrite state."""

  ordinal: int
  slot_ordinal: int
  remote: ResolvedMemoryView
  local: ResolvedMemoryView
  state: str = "WAIT_INDEX"  # WAIT_INDEX | WAIT_PREDECESSOR | READY | WRITE | DONE
  read_transaction_id: str | None = None
  transaction_id: str | None = None
  data: bytes | None = None
  committed_cycle: int | None = None


@dataclass
class _MFEScatterJob:
  desc: ExecEngineDesc
  event_id: str
  scatter: ExecTileScatterDesc
  source: ResolvedMemoryView | None
  indices: ResolvedMemoryView | None
  destination: ResolvedMemoryView | None
  issuer: MemoryOwner
  namespace: tuple[int, int, int, int, str]
  start_cycle: int
  element_bytes: int = 1
  slots: list[_ScatterIndexSlot] = field(default_factory=list)
  next_slot: int = 0
  segments: list[_ScatterSegment] = field(default_factory=list)
  next_segment_ordinal: int = 0
  transaction_ids: set[str] = field(default_factory=set)
  binding_id: str | None = None
  first_failed: bool = False
  failure_reason: str = ""


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
    registry = build_registry(cfg) if l1_cache is None or l2_cache is None else None
    if l1_cache is None:
      assert registry is not None
      l1_profile = registry.profile("l1", cfg.memory_target.l1.reset_mode)
      self.l1_cache = DeterministicLRUCache(
        l1_profile.cache_bytes,
        cfg.cache_line_bytes,
        write_policy=l1_profile.cache_write_policy,
        level="l1",
        pool_id=tile_id,
      )
    else:
      self.l1_cache = l1_cache
    self.l1_mshr = l1_mshr if l1_mshr is not None else MshrTable(cfg.l1_mshr_entries)
    if l2_cache is None:
      assert registry is not None
      l2_profile = registry.profile("l2", cfg.memory_target.l2.reset_mode)
      self.l2_cache = DeterministicLRUCache(
        l2_profile.cache_bytes,
        cfg.cache_line_bytes,
        write_policy=l2_profile.cache_write_policy,
        level="l2",
        pool_id=0,
      )
    else:
      self.l2_cache = l2_cache
    self.l2_mshr = l2_mshr if l2_mshr is not None else MshrTable(cfg.l2_mshr_entries)
    self._load_lanes = [
      _MFELane(f"MFE_LD{i}", cfg.mfe_pipeline_depth) for i in range(cfg.mfe_load_channels)
    ]
    self._store_lanes = [
      _MFELane(f"MFE_ST{j}", cfg.mfe_pipeline_depth) for j in range(cfg.mfe_store_channels)
    ]
    self._gather_jobs: dict[str, _MFEGatherJob] = {}
    self._scatter_jobs: dict[str, _MFEScatterJob] = {}
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
    if self._gather_jobs or self._scatter_jobs or any(lane.running is not None for lane in self._lanes):
      return EngineState.RUNNING
    return EngineState.IDLE

  @state.setter
  def state(self, _value: EngineState) -> None:
    pass

  @property
  def is_busy(self) -> bool:
    lanes_full = all(lane.accepted() >= lane.depth for lane in self._lanes)
    indexed_capacity = self.cfg.mfe_load_channels * self.cfg.mfe_pipeline_depth
    return lanes_full and len(self._gather_jobs) + len(self._scatter_jobs) >= indexed_capacity

  @property
  def accepted_count(self) -> int:
    """Accepted lane jobs plus active Gather/Scatter state machines."""
    return (
      sum(lane.accepted() for lane in self._lanes)
      + len(self._gather_jobs)
      + len(self._scatter_jobs)
    )

  @property
  def has_accepted_jobs(self) -> bool:
    return self.accepted_count > 0

  def launch(
    self, desc: ExecEngineDesc, cycle: int, event_id: str, transaction: MemoryTransaction | None = None
  ) -> EngineJob | _MFETransferJob | None:
    """Route a descriptor to a first-free lane of its direction class."""
    self._validate_stream_buffer(desc)
    lanes = self._store_lanes if desc.op in self._STORE_OPS else self._load_lanes
    free = next((lane for lane in lanes if lane.running is None), None)
    if free is None:
      free = next((lane for lane in lanes if lane.accepted() < lane.depth), None)
      if free is None:
        return None
    job = _MFETransferJob(desc=desc, event_id=event_id, transaction=transaction, enqueue_cycle=cycle)
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
    binding_id: str | None = None,
    bypass_forbidden: bool = False,
    element_bytes: int = 1,
  ) -> _MFEGatherJob | None:
    """Accept one address-resolved Gather and start its index window."""
    capacity = self.cfg.mfe_load_channels * self.cfg.mfe_pipeline_depth
    if len(self._gather_jobs) + len(self._scatter_jobs) >= capacity:
      return None
    if event_id in self._gather_jobs:
      raise ValueError("duplicate gather event id")
    gather = desc.params.get("gather")
    if not isinstance(gather, ExecTileGatherDesc):
      raise ValueError("gather descriptor is missing")
    if self.transfer_manager is None:
      raise ValueError("gather requires a TransferManager")
    if getattr(self.transfer_manager, "byte_store", None) is None:
      raise ValueError("indexed memory requires ByteStore input data")
    if source is None or indices is None or destination is None:
      raise ValueError("gather requires resolved source, index and destination views")
    if element_bytes <= 0:
      raise ValueError("gather element_bytes must be > 0")
    index_count = indices.size_bytes // 4
    if index_count <= 0:
      raise ValueError("gather indices view is empty")
    job = _MFEGatherJob(
      desc=desc,
      event_id=event_id,
      gather=gather,
      source=source,
      indices=indices,
      destination=destination,
      issuer=issuer,
      namespace=namespace,
      start_cycle=cycle,
      element_bytes=element_bytes,
      slots=[_GatherIndexSlot(ordinal=i) for i in range(index_count)],
      binding_id=binding_id,
      bypass_forbidden=bypass_forbidden,
    )
    self._gather_jobs[event_id] = job
    for metric in (
      "gather_requests",
      "gather_l1_hits",
      "gather_l2_hits",
      "gather_hbm_misses",
      "gather_mshr_merges",
      "gather_mshr_stalls",
      "gather_cache_bypass_requests",
      "gather_index_reads",
      "gather_bytes",
      "gather_reorder_wait_cycles",
    ):
      self.pmu.add_event(metric, 0)
    self.pmu.add_event("launch")
    # ``gather_requests`` shares its unit with the hit/miss/bypass counters
    # (one payload request), so it is accumulated as segments resolve below.
    self.pmu.add_event("gather_index_reads", index_count)
    self.pmu.add_event("gather_bytes", destination.size_bytes)
    self._issue_gather_index_slots(job, cycle)
    return job

  def launch_scatter(
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
    binding_id: str | None = None,
    record_write=None,
    element_bytes: int = 1,
  ) -> _MFEScatterJob | None:
    """Accept one address-resolved Scatter (bypasses both caches)."""
    capacity = self.cfg.mfe_load_channels * self.cfg.mfe_pipeline_depth
    if len(self._gather_jobs) + len(self._scatter_jobs) >= capacity:
      return None
    if event_id in self._scatter_jobs:
      raise ValueError("duplicate scatter event id")
    scatter = desc.params.get("scatter")
    if not isinstance(scatter, ExecTileScatterDesc):
      raise ValueError("scatter descriptor is missing")
    if self.transfer_manager is None:
      raise ValueError("scatter requires a TransferManager")
    if getattr(self.transfer_manager, "byte_store", None) is None:
      raise ValueError("indexed memory requires ByteStore input data")
    if source is None or indices is None or destination is None:
      raise ValueError("scatter requires resolved source, index and destination views")
    if element_bytes <= 0:
      raise ValueError("scatter element_bytes must be > 0")
    index_count = indices.size_bytes // 4
    if index_count <= 0:
      raise ValueError("scatter indices view is empty")
    job = _MFEScatterJob(
      desc=desc,
      event_id=event_id,
      scatter=scatter,
      source=source,
      indices=indices,
      destination=destination,
      issuer=issuer,
      namespace=namespace,
      start_cycle=cycle,
      element_bytes=element_bytes,
      slots=[_ScatterIndexSlot(ordinal=i) for i in range(index_count)],
      binding_id=binding_id,
    )
    job.record_write = record_write  # type: ignore[attr-defined]
    self._scatter_jobs[event_id] = job
    for metric in (
      "scatter_segments",
      "scatter_bytes",
      "scatter_index_reads",
      "scatter_overlap_wait_cycles",
      "scatter_commit_latency_cycles",
    ):
      self.pmu.add_event(metric, 0)
    self.pmu.add_event("launch")
    self.pmu.add_event("scatter_index_reads", index_count)
    self.pmu.add_event("scatter_segments", index_count * scatter.address_map.repeat)
    self.pmu.add_event("scatter_bytes", source.size_bytes)
    self._issue_scatter_index_slots(job, cycle)
    return job

  # -- indexed window + transaction helpers ----------------------------

  def _start_lane(self, lane: _MFELane, cycle: int) -> None:
    if not lane.queue:
      return
    job = lane.queue.popleft()
    if job.transaction_submitted:
      raise MemoryInvariantError("queued MFE job was already submitted")
    job.start_cycle = cycle
    lane.running = job
    self.pmu.add_event("launch")
    if job.transaction is not None and self.transfer_manager is not None:
      self.transfer_manager.submit(job.transaction, cycle, self.pmu)
      job.transaction_submitted = True

  def _issue_gather_index_slots(self, job: _MFEGatherJob, cycle: int) -> None:
    """Fill the index window: at most window_entries slots in flight."""
    assert self.transfer_manager is not None and job.indices is not None
    window = job.gather.window_entries
    in_flight = sum(1 for slot in job.slots if slot.issued and not slot.released)
    for slot in job.slots[job.next_slot :]:
      if in_flight >= window:
        break
      index_slice = slice_resolved_view(job.indices, slot.ordinal * 4, 4)
      transaction_id = self._slot_transaction_id(job.namespace, "gather", slot.ordinal, "index_read")
      run_generation, profile_generations = self.transfer_manager.transaction_identity(
        (("l1", self.tile_id), ("l2", 0))
      )
      transaction = MemoryTransaction(
        transaction_id=transaction_id,
        op=TransferOp.INDEX_READ,
        issuer=job.issuer,
        src=index_slice,
        dst=None,
        bytes_total=4,
        completion_event=job.event_id,
        tile_id=self.tile_id,
        run_generation=run_generation,
        profile_generations=profile_generations,
      )
      self.transfer_manager.submit(transaction, cycle, self.pmu)
      slot.transaction_id = transaction_id
      slot.issued = True
      job.transaction_ids.add(transaction_id)
      job.next_slot = slot.ordinal + 1
      in_flight += 1

  def _issue_scatter_index_slots(self, job: _MFEScatterJob, cycle: int) -> None:
    assert self.transfer_manager is not None and job.indices is not None
    window = job.scatter.window_entries
    in_flight = sum(1 for slot in job.slots if slot.transaction_id is not None and slot.value is None)
    for slot in job.slots[job.next_slot :]:
      if in_flight >= window:
        break
      index_slice = slice_resolved_view(job.indices, slot.ordinal * 4, 4)
      transaction_id = self._slot_transaction_id(job.namespace, "scatter", slot.ordinal, "index_read")
      run_generation, profile_generations = self.transfer_manager.transaction_identity(
        (("l1", self.tile_id), ("l2", 0))
      )
      transaction = MemoryTransaction(
        transaction_id=transaction_id,
        op=TransferOp.INDEX_READ,
        issuer=job.issuer,
        src=index_slice,
        dst=None,
        bytes_total=4,
        completion_event=job.event_id,
        tile_id=self.tile_id,
        run_generation=run_generation,
        profile_generations=profile_generations,
      )
      self.transfer_manager.submit(transaction, cycle, self.pmu)
      slot.transaction_id = transaction_id
      job.transaction_ids.add(transaction_id)
      job.next_slot = slot.ordinal + 1
      in_flight += 1

  @staticmethod
  def _slot_transaction_id(job_namespace: tuple, kind: str, ordinal: int, phase: str) -> str:
    prefix = ":".join(str(value) for value in job_namespace)
    return f"{prefix}:{kind}:{ordinal}:{phase}"

  def _gather_transaction_id(self, job: _MFEGatherJob, ordinal: int, phase: str) -> str:
    prefix = ":".join(str(value) for value in job.namespace)
    return f"{prefix}:gather:{ordinal}:{phase}"

  def _scatter_transaction_id(self, job: _MFEScatterJob, ordinal: int, phase: str) -> str:
    prefix = ":".join(str(value) for value in job.namespace)
    return f"{prefix}:scatter:{ordinal}:{phase}"

  def _gather_segment_requests(
    self, job: _MFEGatherJob, slot: _GatherIndexSlot, cycle: int
  ) -> None:
    """Decode one index and append its line-shaped payload requests.

    Plan §2: with any cache enabled each segment is cut by cache line.
    A part whose full line stays inside the resolved remote view becomes
    a whole-line request (refill the line, then keep the part's bytes);
    a part whose line leaves the view becomes a bypass request counted
    in ``gather_cache_bypass_requests`` and faulted when the program's
    contract forbids bypass.  With both caches disabled each segment is
    one direct request.
    """
    assert self.transfer_manager is not None
    assert job.source is not None and job.destination is not None
    element_bytes = job.element_bytes
    task_id = job.namespace[2]
    assert slot.value is not None
    pairs = resolve_indexed_segments(
      job.gather.address_map,
      slot.value,
      slot.ordinal,
      task_id,
      job.source,
      job.destination,
      element_bytes,
    )
    line_bytes = self.cfg.cache_line_bytes
    view_end = job.source.offset_bytes + job.source.size_bytes
    for remote, local in pairs:
      if not (self.l1_cache.enabled or self.l2_cache.enabled):
        # No cache: one direct request per segment (plan §2 SPM pass-through).
        self.pmu.add_event("gather_cache_bypass_requests")
        if job.bypass_forbidden:
          raise MemoryInvariantError("gather_cache_bypass_forbidden")
        self._append_gather_request(job, slot, remote, local, 0, bypass=True, line_bytes=line_bytes)
        self.pmu.add_event("gather_requests")
        continue
      cursor = 0
      total = remote.size_bytes
      while cursor < total:
        absolute = remote.offset_bytes + cursor
        line_start = absolute - absolute % line_bytes
        line_end = line_start + line_bytes
        part_start = max(absolute, line_start)
        part_end = min(absolute + total - cursor, line_end)
        within = part_start - line_start
        part_bytes = part_end - part_start
        local_slice = slice_resolved_view(local, cursor, part_bytes)
        whole_line = line_end <= view_end
        if whole_line:
          # The whole line must be sliced from the source view: the segment
          # slice itself starts mid-line and would be out of bounds.
          remote_slice = slice_resolved_view(
            job.source, line_start - job.source.offset_bytes, line_bytes
          )
        else:
          remote_slice = slice_resolved_view(remote, cursor, part_bytes)
          self.pmu.add_event("gather_cache_bypass_requests")
          if job.bypass_forbidden:
            raise MemoryInvariantError("gather_cache_bypass_forbidden")
        if remote_slice is None or local_slice is None:
          raise MemoryInvariantError("gather segment slice is not resolvable")
        self._append_gather_request(
          job, slot, remote_slice, local_slice, within, bypass=not whole_line, line_bytes=line_bytes
        )
        self.pmu.add_event("gather_requests")
        cursor += part_bytes

  def _append_gather_request(
    self,
    job: _MFEGatherJob,
    slot: _GatherIndexSlot,
    remote: ResolvedMemoryView,
    local: ResolvedMemoryView,
    within_line: int,
    *,
    bypass: bool,
    line_bytes: int,
  ) -> None:
    request = _GatherSegmentRequest(
      ordinal=job.next_request_ordinal,
      slot_ordinal=slot.ordinal,
      remote=remote,
      local=local,
      within_line=within_line,
    )
    job.next_request_ordinal += 1
    if bypass:
      # Plan §2: a bypass request reads its own bytes straight from HBM and
      # never allocates or touches a cache line.
      request.bypass = True
      request.state = "DIRECT_RESPONSE"
    else:
      assert remote.handle is not None
      line_offset = (remote.offset_bytes // line_bytes) * line_bytes
      request.line_identity = CacheLineIdentity(
        remote.handle.allocation_id, remote.handle.generation, line_offset
      )
      request.line_provenance = CacheProvenance(
        getattr(remote.handle.owner, "binding_name", ""),
        remote.handle.allocation_id,
        remote.handle.generation,
        line_offset,
        line_bytes,
      )
    job.requests.append(request)

  def _submit_gather_transaction(
    self,
    job: _MFEGatherJob,
    request: _GatherSegmentRequest,
    op: TransferOp,
    cycle: int,
    *,
    phase: str,
    src: ResolvedMemoryView | None = None,
    dst: ResolvedMemoryView | None = None,
    bytes_total: int | None = None,
  ) -> None:
    assert self.transfer_manager is not None
    transaction_id = self._gather_transaction_id(job, request.ordinal, phase)
    run_generation, profile_generations = self.transfer_manager.transaction_identity(
      (("l1", self.tile_id), ("l2", 0))
    )
    total = request.remote.size_bytes if bytes_total is None else bytes_total
    captured_data = None
    captured_validity = None
    if op is TransferOp.GATHER_DEST_WRITE:
      # The destination write carries only this request's segment bytes
      # (the remote side may be a whole cache line).
      captured_data = request.hit_data
      captured_validity = request.hit_validity
      if captured_data is None or len(captured_data) != request.local.size_bytes:
        raise MemoryInvariantError("gather destination write lacks segment bytes")
      total = request.local.size_bytes
    provenance = request.line_provenance
    transaction = MemoryTransaction(
      transaction_id=transaction_id,
      op=op,
      issuer=job.issuer,
      src=src,
      dst=dst,
      bytes_total=total,
      completion_event=job.event_id,
      tile_id=self.tile_id,
      captured_data=captured_data,
      captured_validity=captured_validity,
      run_generation=run_generation,
      profile_generations=profile_generations,
      bypass_levels=self._gather_bypass_levels(request),
      conservative_source_ranges=(
        (
          provenance.allocation_id,
          provenance.allocation_generation,
          provenance.source_offset,
          provenance.bytes,
        ),
      )
      if (provenance is not None and not provenance.precise)
      else (),
    )
    self.transfer_manager.submit(transaction, cycle, self.pmu)
    request.transaction_id = transaction_id
    request.state = phase.upper()
    job.transaction_ids.add(transaction_id)

  def _gather_bypass_levels(self, request: _GatherSegmentRequest) -> tuple[str, ...]:
    """Skip cache lookup/fill legs for a bypass request or disabled level."""
    levels: list[str] = []
    if not self.l1_cache.enabled:
      levels.append("l1")
    if not self.l2_cache.enabled:
      levels.append("l2")
    return tuple(levels)

  def _scatter_submit(
    self,
    job: _MFEScatterJob,
    segment: _ScatterSegment,
    op: TransferOp,
    cycle: int,
    *,
    phase: str,
    src: ResolvedMemoryView | None = None,
    dst: ResolvedMemoryView | None = None,
  ) -> str:
    assert self.transfer_manager is not None
    transaction_id = self._scatter_transaction_id(job, segment.ordinal, phase)
    run_generation, profile_generations = self.transfer_manager.transaction_identity(
      (("l1", self.tile_id), ("l2", 0))
    )
    transaction = MemoryTransaction(
      transaction_id=transaction_id,
      op=op,
      issuer=job.issuer,
      src=src,
      dst=dst,
      bytes_total=segment.remote.size_bytes,
      completion_event=job.event_id,
      tile_id=self.tile_id,
      captured_data=segment.data if op is TransferOp.SCATTER_WRITE else None,
      run_generation=run_generation,
      profile_generations=profile_generations,
    )
    self.transfer_manager.submit(transaction, cycle, self.pmu)
    job.transaction_ids.add(transaction_id)
    return transaction_id

  def begin_reset_drain(self, cycle: int) -> tuple[str, ...]:
    """Freeze never-submitted lane entries; accepted jobs keep running."""
    return self.cancel_unissued(cycle)

  def request_cancel_active(self, cycle: int) -> int:
    """After timeout, cancel MSHR-only work with no issued transfer."""
    return self.retire_isolated(cycle, cancel_orphans=True)

  def cancel_unissued(self, cycle: int) -> tuple[str, ...]:
    """Withdraw MFE lane entries whose transfer was never submitted.

    Running lane jobs and active Gather FSMs have accepted real work and
    are left to complete/cancel-confirm.  Only queued descriptors without a
    registered TransferManager transaction are removed.
    """
    cancelled: list[str] = []
    for lane in self._lanes:
      retained: deque[_MFETransferJob] = deque()
      while lane.queue:
        job = lane.queue.popleft()
        if not job.transaction_submitted:
          if job.start_cycle is not None:
            raise MemoryInvariantError("started MFE job lacks a submitted transaction")
          cancelled.append(job.event_id)
        else:
          retained.append(job)
      lane.queue = retained
    if cancelled:
      self.pmu.add_event("cancel_unissued", len(cancelled))
      if self.tracer is not None:
        for event_id in cancelled:
          self.tracer.instant(
            f"Tile{self.tile_id}", "MFE", "mfe_cancel_unissued", cycle, {"event_id": event_id}
          )
    return tuple(cancelled)

  def retire_isolated(self, cycle: int, *, cancel_orphans: bool = False) -> int:
    """Retire cancel-confirmed/faulted jobs or proven MSHR orphans."""
    if self.transfer_manager is None:
      return 0
    retired = 0
    for lane in self._lanes:
      transfer_job = lane.running
      if transfer_job is None or transfer_job.transaction is None:
        continue
      if not transfer_job.transaction_submitted:
        raise MemoryInvariantError("running MFE transfer was never submitted")
      status = self.transfer_manager.status(transfer_job.transaction.transaction_id)
      if status not in (TransferStatus.CANCELLED, TransferStatus.FAULTED):
        continue
      self.transfer_manager.acknowledge(transfer_job.transaction.transaction_id, cycle)
      lane.running = None
      retired += 1

    for event_id, gather_job in tuple(self._gather_jobs.items()):
      transaction_ids = set(gather_job.transaction_ids)
      statuses = {
        transaction_id: self.transfer_manager.status(transaction_id)
        for transaction_id in transaction_ids
      }
      if any(
        status in (TransferStatus.PENDING, TransferStatus.RUNNING, TransferStatus.CANCEL_REQUESTED)
        for status in statuses.values()
      ):
        continue
      has_cancel = any(
        status in (TransferStatus.CANCELLED, TransferStatus.FAULTED)
        for status in statuses.values()
      )
      l1_tokens = {
        request.l1_mshr_token
        for request in gather_job.requests
        if request.l1_mshr_token is not None
      }
      l2_tokens = {
        request.l2_mshr_token
        for request in gather_job.requests
        if request.l2_mshr_token is not None
      }
      has_tokens = bool(l1_tokens or l2_tokens)
      inactive_orphan = (
        not transaction_ids
        and has_tokens
        and not any(self.l1_mshr.is_active(token) for token in l1_tokens)
        and not any(self.l2_mshr.is_active(token) for token in l2_tokens)
      )
      cancelled_orphan = cancel_orphans and not transaction_ids and has_tokens
      if not has_cancel and not inactive_orphan and not cancelled_orphan:
        continue
      for transaction_id in transaction_ids:
        self.transfer_manager.acknowledge(transaction_id, cycle)
      for token in l1_tokens:
        if self.l1_mshr.is_active(token):
          self.l1_mshr.cancel(token)
      for token in l2_tokens:
        if self.l2_mshr.is_active(token):
          self.l2_mshr.cancel(token)
      self._gather_jobs.pop(event_id)
      retired += 1

    for event_id, scatter_job in tuple(self._scatter_jobs.items()):
      live = {
        transaction_id
        for transaction_id in scatter_job.transaction_ids
        if self.transfer_manager.status(transaction_id)
        in (TransferStatus.PENDING, TransferStatus.RUNNING, TransferStatus.CANCEL_REQUESTED)
      }
      if live:
        continue
      self._isolate_scatter(scatter_job, cycle)
      self._scatter_jobs.pop(event_id)
      retired += 1

    if retired and self.tracer is not None:
      self.tracer.instant(
        f"Tile{self.tile_id}", "MFE", "mfe_isolation_retired", cycle, {"jobs": retired}
      )
    return retired

  def _transaction_done(self, transaction_id: str | None) -> bool:
    if transaction_id is None or self.transfer_manager is None:
      return False
    return self.transfer_manager.status(transaction_id) is TransferStatus.DONE

  def _transaction_terminal(self, transaction_id: str | None) -> TransferStatus | None:
    if transaction_id is None or self.transfer_manager is None:
      return None
    status = self.transfer_manager.status(transaction_id)
    if status in (TransferStatus.FAULTED, TransferStatus.CANCELLED):
      return status
    return None

  def _acknowledge_transaction(self, transaction_id: str | None, cycle: int) -> None:
    if transaction_id is None or self.transfer_manager is None:
      return
    self.transfer_manager.acknowledge(transaction_id, cycle)

  def _note_merge(self, request: _GatherSegmentRequest) -> None:
    if request.merged_counted:
      return
    request.merged_counted = True
    self.pmu.add_event("gather_mshr_merges")

  @staticmethod
  def _merge_group(request: _GatherSegmentRequest) -> str:
    assert request.line_identity is not None
    identity = request.line_identity
    return f"{identity.allocation_id}:{identity.allocation_generation}:{identity.line_offset}"

  def _enter_mshr_wait(self, request: _GatherSegmentRequest, state: str, wait: MshrWait) -> None:
    if request.state != state:
      self.pmu.add_event("gather_mshr_stalls")
    request.state = state
    request.wait_version = wait.version
    request.transaction_id = None

  @property
  def _byte_oracle_enabled(self) -> bool:
    return self.transfer_manager is not None and self.transfer_manager.byte_store is not None

  def _mark_response_ready(
    self, job: _MFEGatherJob, request: _GatherSegmentRequest, cycle: int
  ) -> None:
    if request.response_ready:
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
          "ordinal": request.ordinal,
          "index_slot": request.slot_ordinal,
          "event_id": job.event_id,
          "bypass": request.bypass,
        },
      )

  def _mark_response_from_cache(
    self,
    job: _MFEGatherJob,
    request: _GatherSegmentRequest,
    cache: DeterministicLRUCache,
    cycle: int,
  ) -> None:
    if request.line_identity is not None:
      line = cache.read_line(request.line_identity, require_data=self._byte_oracle_enabled)
      if line is not None:
        request.line_data = line
        start = request.within_line
        end = start + request.local.size_bytes
        request.hit_data = line[start:end]
        validity = cache.read_validity(request.line_identity)
        if validity is not None:
          request.hit_validity = validity[start:end]
    self._mark_response_ready(job, request, cycle)

  def _try_l1_mshr(self, job: _MFEGatherJob, request: _GatherSegmentRequest, cycle: int) -> None:
    if not self.l1_cache.enabled:
      self._try_l2_mshr(job, request, cycle)
      return
    allocation = self.l1_mshr.allocate(self._merge_group(request))
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
        lambda: self._mark_response_from_cache(job, request, self.l1_cache, self._current_cycle),
      )
      return
    self._try_l2_mshr(job, request, cycle)

  def _try_l2_mshr(self, job: _MFEGatherJob, request: _GatherSegmentRequest, cycle: int) -> None:
    if not self.l2_cache.enabled:
      if not self.l1_cache.enabled:
        raise MemoryInvariantError("both disabled cache levels must use direct Gather response")
      self._submit_gather_transaction(
        job,
        request,
        TransferOp.GATHER_DIRECT_L1_REFILL,
        cycle,
        phase="l1_direct_refill",
        src=request.remote,
        bytes_total=request.remote.size_bytes,
      )
      return
    allocation = self.l2_mshr.allocate(self._merge_group(request))
    if isinstance(allocation, MshrWait):
      self._enter_mshr_wait(request, "WAIT_L2_MSHR", allocation)
      return
    request.l2_mshr_token = allocation.token
    request.wait_version = None
    if not allocation.leader:
      self._note_merge(request)
      request.state = "WAIT_L2_FILL"
      if self.l1_cache.enabled:
        callback = lambda: self._submit_l2_refill(job, request, self._current_cycle)
      else:
        callback = lambda: self._mark_response_from_cache(job, request, self.l2_cache, self._current_cycle)
      self.l2_mshr.wait(allocation.token, callback)
      return
    self._submit_gather_transaction(
      job,
      request,
      TransferOp.GATHER_HBM_REFILL,
      cycle,
      phase="hbm_refill",
      src=request.remote,
      bytes_total=request.remote.size_bytes,
    )

  def _submit_l2_refill(self, job: _MFEGatherJob, request: _GatherSegmentRequest, cycle: int) -> None:
    self._submit_gather_transaction(
      job, request, TransferOp.GATHER_L2_REFILL, cycle, phase="l2_refill"
    )

  def _tick_gather_request(
    self, job: _MFEGatherJob, request: _GatherSegmentRequest, cycle: int
  ) -> None:
    """Advance one payload request through the lookup/fill FSM."""
    if request.state in ("WAIT_L1_MSHR", "WAIT_L2_MSHR"):
      if request.state == "WAIT_L1_MSHR":
        if request.wait_version != self.l1_mshr.version:
          self._try_l1_mshr(job, request, cycle)
      elif request.wait_version != self.l2_mshr.version:
        self._try_l2_mshr(job, request, cycle)
      return
    if request.state in ("WAIT_L1_FILL", "WAIT_L2_FILL", "RESPONSE_READY", "WRITE", "DONE"):
      return

    if request.state == "DIRECT_RESPONSE" and request.transaction_id is None:
      # Bypass request: read exactly its own bytes, never a cache line.
      self._submit_gather_transaction(
        job,
        request,
        TransferOp.GATHER_DIRECT_RESPONSE,
        cycle,
        phase="direct_response",
        src=request.remote,
        bytes_total=request.remote.size_bytes,
      )
      return

    if request.state == "LOOKUP_L1" and request.transaction_id is None:
      self._submit_gather_transaction(
        job, request, TransferOp.GATHER_L1_LOOKUP, cycle, phase="lookup_l1"
      )
      return
    if request.state == "LOOKUP_L2" and request.transaction_id is None:
      self._submit_gather_transaction(
        job, request, TransferOp.GATHER_L2_LOOKUP, cycle, phase="lookup_l2"
      )
      return

    if not self._transaction_done(request.transaction_id):
      terminal = self._transaction_terminal(request.transaction_id)
      if terminal is not None:
        raise MemoryInvariantError(
          f"gather transaction {request.transaction_id} reached {terminal.value}"
        )
      return

    state = request.state
    if state in ("HBM_REFILL", "L1_DIRECT_REFILL", "DIRECT_RESPONSE"):
      assert self.transfer_manager is not None
      line = self.transfer_manager.captured_data(request.transaction_id)
      validity = self.transfer_manager.captured_validity(request.transaction_id)
      if state == "DIRECT_RESPONSE" and line is not None and request.within_line == 0:
        request.hit_data = line
        request.hit_validity = validity
      else:
        request.line_data = line
        request.line_validity = validity
    acknowledged = request.transaction_id
    self._acknowledge_transaction(acknowledged, cycle)
    job.transaction_ids.discard(acknowledged)
    request.transaction_id = None
    identity = request.line_identity

    if state in ("LOOKUP_L1", "LOOKUP_L2"):
      if self.tracer is not None:
        self.tracer.instant(
          f"Tile{self.tile_id}",
          "MFE",
          "gather_lookup",
          cycle,
          {
            "ordinal": request.ordinal,
            "level": "l1" if state == "LOOKUP_L1" else "l2",
            "allocation_id": identity.allocation_id if identity is not None else "",
            "line_offset": identity.line_offset if identity is not None else -1,
            "bypass": request.bypass,
            "event_id": job.event_id,
          },
        )

    if state == "LOOKUP_L1":
      assert identity is not None
      if self.l1_cache.contains(identity):
        self.l1_cache.record_hit(identity, require_resident=True)
        self.pmu.add_event("gather_l1_hits")
        self._mark_response_from_cache(job, request, self.l1_cache, cycle)
      else:
        self.l1_cache.record_miss()
        if self.l2_cache.enabled:
          request.state = "LOOKUP_L2"
          self._submit_gather_transaction(
            job, request, TransferOp.GATHER_L2_LOOKUP, cycle, phase="lookup_l2"
          )
        else:
          self._try_l1_mshr(job, request, cycle)
      return

    if state == "LOOKUP_L2":
      assert identity is not None
      if self.l2_cache.contains(identity):
        self.l2_cache.record_hit(identity, require_resident=True)
        self.pmu.add_event("gather_l2_hits")
        if self.l1_cache.enabled:
          allocation = self.l1_mshr.allocate(self._merge_group(request))
          if isinstance(allocation, MshrWait):
            self._enter_mshr_wait(request, "WAIT_L1_MSHR", allocation)
            return
          request.l1_mshr_token = allocation.token
          if not allocation.leader:
            self._note_merge(request)
            request.state = "WAIT_L1_FILL"
            self.l1_mshr.wait(
              allocation.token,
              lambda: self._mark_response_from_cache(job, request, self.l1_cache, self._current_cycle),
            )
            return
          request.state = "L2_REFILL"
          self._submit_l2_refill(job, request, cycle)
        else:
          self._mark_response_from_cache(job, request, self.l2_cache, cycle)
      else:
        self.l2_cache.record_miss()
        self.pmu.add_event("gather_hbm_misses")
        self._try_l1_mshr(job, request, cycle)
      return

    if state == "HBM_REFILL":
      assert identity is not None
      if self.tracer is not None:
        self.tracer.instant(
          f"Tile{self.tile_id}",
          "MFE",
          "gather_refill",
          cycle,
          {
            "ordinal": request.ordinal,
            "allocation_id": identity.allocation_id,
            "line_offset": identity.line_offset,
            "bytes": request.remote.size_bytes,
            "event_id": job.event_id,
          },
        )
      self.l2_cache.refill(
        None,
        identity=identity,
        provenance=request.line_provenance,
        data=request.line_data,
        validity=request.line_validity,
      )
      assert request.l2_mshr_token is not None
      callbacks = self.l2_mshr.complete(request.l2_mshr_token)
      request.l2_mshr_token = None
      self._invoke_callbacks(callbacks)
      if self.l1_cache.enabled:
        self._submit_l2_refill(job, request, cycle)
      else:
        self._mark_response_from_cache(job, request, self.l2_cache, cycle)
      return

    if state == "L1_DIRECT_REFILL":
      assert identity is not None
      self.l1_cache.refill(
        None,
        identity=identity,
        provenance=request.line_provenance,
        data=request.line_data,
        validity=request.line_validity,
      )
      assert request.l1_mshr_token is not None
      callbacks = self.l1_mshr.complete(request.l1_mshr_token)
      request.l1_mshr_token = None
      self._mark_response_from_cache(job, request, self.l1_cache, cycle)
      self._invoke_callbacks(callbacks)
      return

    if state == "DIRECT_RESPONSE":
      if request.line_data is not None:
        start = request.within_line
        end = start + request.local.size_bytes
        request.hit_data = request.line_data[start:end]
        if request.line_validity is not None:
          request.hit_validity = request.line_validity[start:end]
      self._mark_response_ready(job, request, cycle)
      return

    if state == "L2_REFILL":
      assert identity is not None
      request.line_data = self.l2_cache.read_line(identity, require_data=self._byte_oracle_enabled)
      self.l1_cache.refill(
        None,
        identity=identity,
        provenance=request.line_provenance,
        data=request.line_data,
        validity=request.line_validity,
      )
      assert request.l1_mshr_token is not None
      callbacks = self.l1_mshr.complete(request.l1_mshr_token)
      request.l1_mshr_token = None
      self._mark_response_from_cache(job, request, self.l1_cache, cycle)
      self._invoke_callbacks(callbacks)

  def _invoke_callbacks(self, callbacks) -> None:
    for callback in callbacks:
      callback()

  def _tick_gather(self, job: _MFEGatherJob, cycle: int) -> EngineJob | None:
    """Advance one Gather job: index window, lookups, ordered dest writes."""
    assert self.transfer_manager is not None
    store = getattr(self.transfer_manager, "byte_store", None)
    # 1. Index window: harvest finished INDEX_READs and decode them.
    for slot in job.slots:
      if slot.state == "INDEX_READ":
        terminal = self._transaction_terminal(slot.transaction_id)
        if terminal is not None:
          raise MemoryInvariantError(
            f"gather index read {slot.transaction_id} reached {terminal.value}"
          )
        if not self._transaction_done(slot.transaction_id):
          continue
        assert store is not None
        slot.value = decode_index_i32(self.transfer_manager.captured_data(slot.transaction_id))
        acknowledged = slot.transaction_id
        self._acknowledge_transaction(acknowledged, cycle)
        job.transaction_ids.discard(acknowledged)
        slot.transaction_id = None
        slot.state = "DECODED"
        self._gather_segment_requests(job, slot, cycle)
    self._issue_gather_index_slots(job, cycle)

    # 2. Payload FSM.
    for request in job.requests:
      self._tick_gather_request(job, request, cycle)
    return self._tick_gather_materialization(job, cycle)

  def _tick_gather_materialization(self, job: _MFEGatherJob, cycle: int) -> EngineJob | None:
    if job.write_transaction_id is not None:
      assert self.transfer_manager is not None
      status = self.transfer_manager.status(job.write_transaction_id)
      if status in (TransferStatus.FAULTED, TransferStatus.CANCELLED):
        raise MemoryInvariantError(f"Gather destination write reached {status.value}")
      if status is TransferStatus.DONE:
        request = job.requests[job.next_write_ordinal]
        acknowledged = job.write_transaction_id
        self.transfer_manager.acknowledge(acknowledged, cycle)
        job.transaction_ids.discard(acknowledged)
        request.transaction_id = None
        request.state = "DONE"
        if self.tracer is not None:
          self.tracer.instant(
            f"Tile{self.tile_id}",
            "MFE",
            "gather_destination_write",
            cycle,
            {
              "ordinal": request.ordinal,
              "index_slot": request.slot_ordinal,
              "event_id": job.event_id,
            },
          )
        job.write_transaction_id = None
        job.next_write_ordinal += 1
        slot = job.slots[request.slot_ordinal]
        if all(req.state == "DONE" for req in job.requests if req.slot_ordinal == slot.ordinal):
          slot.released = True

    if job.next_write_ordinal >= len(job.requests) and all(slot.released for slot in job.slots):
      self.pmu.add_event("complete")
      if self.tracer is not None:
        self.tracer.instant(
          f"Tile{self.tile_id}", "MFE", "gather_done", cycle, {"event_id": job.event_id}
        )
      return EngineJob(
        desc=job.desc,
        start_cycle=job.start_cycle,
        finish_cycle=cycle,
        event_id=job.event_id,
        pmu=PMUCounter(),
      )

    if job.write_transaction_id is None and job.next_write_ordinal < len(job.requests):
      request = job.requests[job.next_write_ordinal]
      if request.response_ready:
        self._submit_gather_transaction(
          job,
          request,
          TransferOp.GATHER_DEST_WRITE,
          cycle,
          phase="write",
          dst=request.local,
        )
        job.write_transaction_id = request.transaction_id
        request.transaction_id = None
        return None

    if (
      job.next_write_ordinal < len(job.requests)
      and not job.requests[job.next_write_ordinal].response_ready
    ):
      if any(item.response_ready for item in job.requests[job.next_write_ordinal + 1 :]):
        self.pmu.add_event("gather_reorder_wait_cycles")
    return None

  # -- Scatter FSM ------------------------------------------------------

  def _tick_scatter(self, job: _MFEScatterJob, cycle: int) -> EngineJob | None:
    """Advance one Scatter job: index decode, ordered overwrite, commit."""
    assert self.transfer_manager is not None
    store = getattr(self.transfer_manager, "byte_store", None)
    for slot in job.slots:
      if slot.state == "INDEX_READ":
        terminal = self._transaction_terminal(slot.transaction_id)
        if terminal is not None:
          job.first_failed = True
          job.failure_reason = f"scatter index read {terminal.value}"
          continue
        if not self._transaction_done(slot.transaction_id):
          continue
        assert store is not None
        slot.value = decode_index_i32(self.transfer_manager.captured_data(slot.transaction_id))
        acknowledged = slot.transaction_id
        self._acknowledge_transaction(acknowledged, cycle)
        job.transaction_ids.discard(acknowledged)
        slot.transaction_id = None
        slot.state = "DECODED"
        self._scatter_segment_requests(job, slot)
    self._issue_scatter_index_slots(job, cycle)
    # Decoded slots release their segments into the ordered-overwrite queue.
    for segment in job.segments:
      if segment.state == "WAIT_INDEX":
        segment.state = "WAIT_PREDECESSOR"

    for segment in job.segments:
      self._tick_scatter_segment(job, segment, cycle)

    if job.first_failed:
      # Stop new issue; isolate in-flight; first-fault-wins record.
      self._isolate_scatter(job, cycle)
      raise MemoryInvariantError(f"scatter fault: {job.failure_reason}")

    for slot in job.slots:
      if slot.state != "DECODED":
        continue
      if all(
        segment.state == "DONE" for segment in job.segments if segment.slot_ordinal == slot.ordinal
      ):
        slot.state = "DONE"
    if all(slot.state == "DONE" for slot in job.slots) and all(
      segment.state == "DONE" for segment in job.segments
    ):
      self.pmu.add_event("complete")
      if self.tracer is not None:
        self.tracer.instant(
          f"Tile{self.tile_id}", "MFE", "scatter_done", cycle, {"event_id": job.event_id}
        )
      return EngineJob(
        desc=job.desc,
        start_cycle=job.start_cycle,
        finish_cycle=cycle,
        event_id=job.event_id,
        pmu=PMUCounter(),
      )
    return None

  def _scatter_segment_requests(self, job: _MFEScatterJob, slot: _ScatterIndexSlot) -> None:
    assert job.source is not None and job.destination is not None
    element_bytes = job.element_bytes
    task_id = job.namespace[2]
    pairs = resolve_indexed_segments(
      job.scatter.address_map,
      slot.value,  # type: ignore[arg-type]
      slot.ordinal,
      task_id,
      job.destination,
      job.source,
      element_bytes,
    )
    for remote, local in pairs:
      segment = _ScatterSegment(
        ordinal=job.next_segment_ordinal,
        slot_ordinal=slot.ordinal,
        remote=remote,
        local=local,
      )
      job.next_segment_ordinal += 1
      job.segments.append(segment)

  def _segment_overlaps_pending(self, job: _MFEScatterJob, segment: _ScatterSegment) -> bool:
    """True when an earlier unordered segment's byte range overlaps."""
    end = segment.remote.offset_bytes + segment.remote.size_bytes
    for other in job.segments:
      if other is segment or other.ordinal >= segment.ordinal or other.state != "WRITE":
        continue
      other_end = other.remote.offset_bytes + other.remote.size_bytes
      if segment.remote.offset_bytes < other_end and other.remote.offset_bytes < end:
        return True
    return False

  def _tick_scatter_segment(
    self, job: _MFEScatterJob, segment: _ScatterSegment, cycle: int
  ) -> None:
    assert self.transfer_manager is not None
    if segment.state in ("WAIT_INDEX", "DONE"):
      return
    if segment.state == "WAIT_PREDECESSOR":
      if not self._segment_overlaps_pending(job, segment):
        segment.state = "READY"
      else:
        self.pmu.add_event("scatter_overlap_wait_cycles")
        return
    if segment.state == "READY":
      # One SCATTER_WRITE transaction: its L1_READ leg captures the
      # source bytes and the HBM_WRITE leg commits them (plan §2).
      segment.data = None
      transaction_id = self._scatter_submit(
        job,
        segment,
        TransferOp.SCATTER_WRITE,
        cycle,
        phase="write",
        src=segment.local,
        dst=segment.remote,
      )
      segment.transaction_id = transaction_id
      segment.state = "WRITE"
      return
    if segment.state == "WRITE":
      if not self._transaction_done(segment.transaction_id):
        terminal = self._transaction_terminal(segment.transaction_id)
        if terminal is not None:
          job.first_failed = True
          job.failure_reason = f"scatter write {terminal.value}"
        return
      if segment.data is None:
        segment.data = self.transfer_manager.captured_data(segment.transaction_id)
      acknowledged = segment.transaction_id
      self.transfer_manager.acknowledge(acknowledged, cycle)
      segment.transaction_id = None
      job.transaction_ids.discard(acknowledged)
      segment.state = "DONE"
      segment.committed_cycle = cycle
      if self.tracer is not None:
        self.tracer.instant(
          f"Tile{self.tile_id}",
          "MFE",
          "scatter_write",
          cycle,
          {
            "ordinal": segment.ordinal,
            "index_slot": segment.slot_ordinal,
            "event_id": job.event_id,
            "bytes": segment.remote.size_bytes,
          },
        )
      record_write = getattr(job, "record_write", None)
      if record_write is not None:
        handle = segment.remote.handle
        record_write(
          handle.allocation_id,
          handle.generation,
          segment.remote.offset_bytes,
          segment.remote.size_bytes,
        )

  def _isolate_scatter(self, job: _MFEScatterJob, cycle: int) -> None:
    assert self.transfer_manager is not None
    for transaction_id in tuple(job.transaction_ids):
      status = self.transfer_manager.status(transaction_id)
      if status in (TransferStatus.PENDING, TransferStatus.RUNNING, TransferStatus.CANCEL_REQUESTED):
        continue
      self.transfer_manager.acknowledge(transaction_id, cycle)
      job.transaction_ids.discard(transaction_id)

  def tick(self, cycle: int, start_queued: bool = True) -> list[EngineJob]:
    """Advance transfer lanes and all active Gather state machines."""
    self._current_cycle = cycle
    completed: list[EngineJob] = []
    if not start_queued:
      # Reset drain: retire only cancel-confirmed work before the normal
      # success FSM observes it. Running/CANCEL_REQUESTED work remains.
      self.retire_isolated(cycle)
    active_lanes = 0
    for lane in self._lanes:
      job = lane.running
      if job is None:
        continue
      active_lanes += 1
      done = False
      if job.transaction is not None and self.transfer_manager is not None:
        if not job.transaction_submitted:
          raise MemoryInvariantError("running MFE transfer was never submitted")
        done = self.transfer_manager.status(job.transaction.transaction_id) is TransferStatus.DONE
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
        if (job.transaction is not None and job.transaction.completed_cycle >= 0)
        else cycle
      )
      completed.append(
        EngineJob(
          desc=job.desc, start_cycle=start, finish_cycle=finish, event_id=job.event_id, pmu=PMUCounter()
        )
      )
      if job.transaction is not None and self.transfer_manager is not None:
        self.transfer_manager.acknowledge(job.transaction.transaction_id, cycle)
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

    gather_active = len(self._gather_jobs) + len(self._scatter_jobs)
    finished: list[str] = []
    for event_id, gather_job in list(self._gather_jobs.items()):
      completion = self._tick_gather(gather_job, cycle)
      if completion is not None:
        completed.append(completion)
        finished.append(event_id)
      # Gather FSM transitions are the only writers of cache/MSHR state;
      # push stats after each job's tick (change-only sampling keeps this
      # constant-cost).
      self._emit_memory_trace(cycle)
    for event_id in finished:
      self._gather_jobs.pop(event_id, None)
    finished = []
    for event_id, scatter_job in list(self._scatter_jobs.items()):
      completion = self._tick_scatter(scatter_job, cycle)
      if completion is not None:
        completed.append(completion)
        finished.append(event_id)
      self._emit_memory_trace(cycle)
    for event_id in finished:
      self._scatter_jobs.pop(event_id, None)

    blocked_indexed = sum(
      1
      for gather_job in self._gather_jobs.values()
      if any(request.state in ("WAIT_L1_MSHR", "WAIT_L2_MSHR") for request in gather_job.requests)
    )
    if active_lanes or gather_active:
      self.pmu.add_cycle("mfe_active", 1)
      if blocked_indexed:
        self.pmu.add(StallReason.WAIT_MSHR, blocked_indexed)
      else:
        self.pmu.add(StallReason.NONE, 1)
    else:
      self.pmu.add_cycle("mfe_idle", 1)
    self.pmu.add_cycle("mfe_channel_active", active_lanes)
    self.pmu.add_cycle("mfe_gather_active", gather_active)
    self.pmu.add_cycle("total", 1)
    return completed

  def reset(self) -> None:
    self.cancel_unissued(self._current_cycle)
    if (
      any(lane.running is not None or lane.queue for lane in self._lanes)
      or self._gather_jobs
      or self._scatter_jobs
    ):
      raise MemoryInvariantError("MFE reset requires accepted work to complete or cancel-confirm")
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
      raise ValueError("MFE page_stream num_pages must be > 0 for buffer validation")
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
