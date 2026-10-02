"""Per-leg memory transfer transactions, routes and resource stages.

Replaces the old ``_DMAJob`` group-DMA model with a per-leg route state
machine.  Each transfer is a ``MemoryTransaction`` with a deterministic
id, a source and destination ``ResolvedMemoryView``, and a multi-leg
route.  The ``TransferManager`` advances transactions cycle by cycle,
issuing each leg only after the previous leg completes, and reports
per-stage wait reasons for PMU attribution.

Three fidelity modes:
  - ``timing_only``: src/dst are ``None``; one collapsed latency leg.
  - ``runtime``: real handle/address but collapsed latency (one leg).
  - ``full_memory``: full multi-leg route with HBM/NoC/DMA/bank stages.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum

from .allocator import AllocationHandle, BankSegment, MemoryInvariantError, MemoryOwner
from .noc import Flit, VCId, normalize_vc_id

# ---------------------------------------------------------------------------
# Resolved memory view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedMemoryView:
  """A physical byte range within a live allocation.

  ``address`` is the absolute physical address of the first byte;
  ``segments`` are the clipped bank segments.  Only runtime/full_memory
  creates non-``None`` views.
  """

  handle: AllocationHandle
  offset_bytes: int
  size_bytes: int
  address: int
  segments: tuple[BankSegment, ...]
  permissions: str = ""

  @property
  def end_address(self) -> int:
    return self.address + self.size_bytes


def slice_resolved_view(
  view: ResolvedMemoryView | None, offset_bytes: int, size_bytes: int
) -> ResolvedMemoryView | None:
  """Return one strict logical-byte slice over ordered physical segments."""
  if view is None:
    return None
  if offset_bytes < 0 or size_bytes <= 0:
    raise MemoryInvariantError("memory view out of bounds")
  slice_end = offset_bytes + size_bytes
  if slice_end > view.size_bytes:
    raise MemoryInvariantError("memory view out of bounds")

  segments: list[BankSegment] = []
  logical_cursor = 0
  for segment in view.segments:
    segment_logical_end = logical_cursor + segment.size_bytes
    overlap_start = max(offset_bytes, logical_cursor)
    overlap_end = min(slice_end, segment_logical_end)
    if overlap_start < overlap_end:
      physical_offset = overlap_start - logical_cursor
      segments.append(
        BankSegment(segment.bank_id, segment.address + physical_offset, overlap_end - overlap_start)
      )
    logical_cursor = segment_logical_end

  if sum(segment.size_bytes for segment in segments) != size_bytes:
    raise MemoryInvariantError("memory view out of bounds")
  return ResolvedMemoryView(
    handle=view.handle,
    offset_bytes=view.offset_bytes + offset_bytes,
    size_bytes=size_bytes,
    address=segments[0].address,
    segments=tuple(segments),
    permissions=view.permissions,
  )


# ---------------------------------------------------------------------------
# Transfer ops, legs, stages
# ---------------------------------------------------------------------------


class TransferOp(Enum):
  PREFETCH = "prefetch"
  GLOBAL_STORE = "global_store"
  TILE_LOAD = "tile_load"
  TILE_STORE = "tile_store"
  INDEX_READ = "index_read"
  GATHER_L1_LOOKUP = "gather_l1_lookup"
  GATHER_L2_LOOKUP = "gather_l2_lookup"
  GATHER_L2_RESPONSE = "gather_l2_response"
  GATHER_HBM_REFILL = "gather_hbm_refill"
  GATHER_L2_REFILL = "gather_l2_refill"
  GATHER_DEST_WRITE = "gather_dest_write"
  GATHER_DIRECT_L1_REFILL = "gather_direct_l1_refill"
  GATHER_DIRECT_RESPONSE = "gather_direct_response"
  SCATTER_WRITE = "scatter_write"
  HOST_READ = "host_read"
  HOST_WRITE = "host_write"
  CACHE_CLEAN_L1 = "cache_clean_l1"
  CACHE_CLEAN_L2 = "cache_clean_l2"


class TransferLegKind(Enum):
  HBM_READ = "hbm_read"
  HBM_WRITE = "hbm_write"
  GLOBAL_DMA = "global_dma"
  NOC_RESPONSE = "noc_response"
  NOC_REQUEST = "noc_request"
  L2_READ = "l2_read"
  L2_WRITE = "l2_write"
  LOCAL_DMA = "local_dma"
  L1_READ = "l1_read"
  L1_WRITE = "l1_write"
  L1_CACHE_LOOKUP = "l1_cache_lookup"
  L2_CACHE_LOOKUP = "l2_cache_lookup"
  L1_CACHE_FILL = "l1_cache_fill"
  L2_CACHE_FILL = "l2_cache_fill"


class StageWaitReason(Enum):
  NONE = "none"
  HBM_OUTSTANDING = "hbm_outstanding"
  DMA_QUEUE = "dma_queue"
  NOC_CREDIT = "noc_credit"
  L2_BANK = "l2_bank"
  L1_BANK = "l1_bank"
  L1_CACHE = "l1_cache"
  L2_CACHE = "l2_cache"

_GATHER_HBM_READ_OPS = frozenset(
  {
    TransferOp.GATHER_HBM_REFILL,
    TransferOp.GATHER_DIRECT_L1_REFILL,
    TransferOp.GATHER_DIRECT_RESPONSE,
  }
)
"""Gather legs whose HBM source may legitimately cover uninitialised bytes.

Plan §2: a whole-line refill and a sub-line bypass segment both cross page
tail tokens and padding, so their source capture keeps a validity mask and
only the destination write faults.
"""


class TransferStatus(Enum):
  PENDING = "pending"
  RUNNING = "running"
  CANCEL_REQUESTED = "cancel_requested"
  DONE = "done"
  FAULTED = "faulted"
  CANCELLED = "cancelled"


@dataclass(frozen=True)
class TransferLeg:
  """One leg of a transfer route."""

  kind: TransferLegKind
  src_space: str  # "hbm" | "l2" | "l1"
  dst_space: str
  bytes_total: int
  resource_id: str  # deterministic stage resource key


@dataclass(frozen=True)
class StageRequest:
  """One resource request within a leg."""

  resource_id: str
  bytes_total: int


@dataclass
class StageResult:
  """Result of a successful ``try_issue``."""

  accepted_cycle: int
  completion_cycle: int
  channels: int = 1
  # bank ids (bank-based stages) or the single channel index chosen
  resources: tuple[int, ...] = ()


@dataclass
class StageWait:
  """Result of a blocked ``try_issue`` (zero side effects)."""

  reason: StageWaitReason


# ---------------------------------------------------------------------------
# Memory transaction
# ---------------------------------------------------------------------------


@dataclass
class MemoryTransaction:
  """One DMA or MFE transfer with deterministic id and resolved views."""

  transaction_id: str
  op: TransferOp
  issuer: MemoryOwner
  src: ResolvedMemoryView | None
  dst: ResolvedMemoryView | None
  bytes_total: int
  completion_event: str
  tile_id: int | None = None
  status: TransferStatus = TransferStatus.PENDING
  legs: tuple[TransferLeg, ...] = ()
  current_leg: int = 0
  leg_start_cycle: int = -1
  leg_completion_cycle: int = -1
  wait_reason: StageWaitReason = StageWaitReason.NONE
  start_cycle: int = -1  # first leg accept cycle
  completed_cycle: int = -1  # final leg completion cycle
  noc_tag: str = ""  # NoC flit tag while a NOC leg is in flight
  noc_vc: int = 0  # virtual channel of the in-flight NOC leg
  # Byte visibility is deliberately tied to completed read/write legs.
  captured_data: bytes | None = None
  # Per-byte validity mask for relaxed whole-line refills (plan §2):
  # 1 = byte initialised in the store, 0 = never written (page tail /
  # padding).  Destination writes touching a 0 byte fault.
  captured_validity: bytes | None = None
  source_captured_cycle: int = -1
  destination_committed_cycle: int = -1
  # Runtime/profile generation captured by the submitter.  The manager's
  # validator checks it again before issue, source capture and destination
  # commit so a late old-generation return cannot modify reused memory.
  run_generation: int = 0
  profile_generations: tuple[tuple[str, int, int], ...] = ()
  bypass_levels: tuple[str, ...] = ()
  # Conservative non-oracle source provenance:
  # (allocation_id, allocation_generation, offset, bytes).
  conservative_source_ranges: tuple[tuple[str, int, int, int], ...] = ()
  cancel_requested_cycle: int = -1
  isolation_confirmed_cycle: int = -1
  fault_reason: str = ""
  # True only after the owner-side L2 reference hook registered every
  # endpoint view/backing for this accepted transaction.  Acknowledgement
  # releases references exactly when this flag is set, so a rejected or
  # failed-acquisition transaction can never release a reference it does
  # not hold.
  reference_acquired: bool = False


# ---------------------------------------------------------------------------
# Transfer stage — one resource with channels/bandwidth/latency
# ---------------------------------------------------------------------------


class TransferStage:
  """One resource stage (HBM channel, DMA channel, NoC VC, L2/L1 bank).

  ``try_issue`` atomically checks that all requested channels/banks are
  available in the current cycle, then occupies them.  If any is busy it
  returns ``StageWait`` with zero side effects.  It does not book future
  channels; the manager retries pending transactions each cycle.

  Resource ownership is tracked per transaction (``_holders``).  Unaccepted
  legs can be withdrawn synchronously; accepted legs retain resources and
  outstanding credits until real completion or explicit isolation
  confirmation.
  """

  def __init__(
    self,
    name: str,
    wait_reason: StageWaitReason,
    fixed_latency_cycles: int,
    bytes_per_cycle: float,
    resource_count: int,
    burst_bytes: int = 1,
    max_outstanding: int | None = None,
    shared_outstanding: set[str] | None = None,
  ):
    if not isinstance(wait_reason, StageWaitReason):
      raise TypeError("transfer stage wait_reason must be a StageWaitReason")
    self.name = name
    self.wait_reason = wait_reason
    self.fixed_latency_cycles = fixed_latency_cycles
    self.bytes_per_cycle = bytes_per_cycle
    self.resource_count = resource_count
    self.burst_bytes = burst_bytes
    self.max_outstanding = max_outstanding
    self._busy_until: list[int] = [0] * resource_count
    # transaction id currently holding each resource
    self._holders: list[str | None] = [None] * resource_count
    # HBM read/write pass the same set to model one global CAM pool.
    self._outstanding_txns: set[str] = shared_outstanding if shared_outstanding is not None else set()
    self.wait_cycles: int = 0  # cumulative StageWait cycles (PMU delta)

  @property
  def _outstanding(self) -> int:
    return len(self._outstanding_txns)

  def try_issue(
    self, transaction_id: str, requests: list[StageRequest], cycle: int
  ) -> StageResult | StageWait:
    """Atomically issue all requests or return wait (zero side effects).

    Bank-based stages (L2/L1): each ``resource_id`` is the bank index;
    all requested banks must be free, then occupied atomically.  The
    leg completion is the max across all segment completions
    (different banks run in parallel, same bank serializes).
    Channel/outstanding stages: pick the first free resource.
    """
    if self.max_outstanding is not None and self._outstanding >= self.max_outstanding:
      return StageWait(self.wait_reason)
    # Determine if this is a bank-based stage (resource_id is a bank index)
    bank_ids: list[int] = []
    for req in requests:
      try:
        bank_ids.append(int(req.resource_id))
      except ValueError:
        bank_ids.append(-1)
    is_bank_based = bank_ids and all(b >= 0 for b in bank_ids)
    if is_bank_based:
      # Check all required banks are free
      for bid in bank_ids:
        if bid < 0 or bid >= self.resource_count:
          return StageWait(self.wait_reason)
        if self._busy_until[bid] > cycle:
          return StageWait(self.wait_reason)
      # All free: occupy atomically, completion = max across segments
      max_completion = 0
      for req, bid in zip(requests, bank_ids):
        burst_rounded = ((req.bytes_total + self.burst_bytes - 1) // self.burst_bytes) * self.burst_bytes
        if self.bytes_per_cycle > 0:
          xfer = int(max((burst_rounded + self.bytes_per_cycle - 1) // self.bytes_per_cycle, 1))
        else:
          xfer = 0
        comp = cycle + self.fixed_latency_cycles + xfer
        self._busy_until[bid] = comp
        self._holders[bid] = transaction_id
        max_completion = max(max_completion, comp)
      if self.max_outstanding is not None:
        self._outstanding_txns.add(transaction_id)
      return StageResult(
        accepted_cycle=cycle,
        completion_cycle=max_completion,
        channels=len(bank_ids),
        resources=tuple(bank_ids),
      )
    # Channel/outstanding stage: pick first free resource
    free_idx = None
    for i in range(self.resource_count):
      if self._busy_until[i] <= cycle:
        free_idx = i
        break
    if free_idx is None:
      return StageWait(self.wait_reason)
    total_bytes = sum(r.bytes_total for r in requests)
    burst_rounded = ((total_bytes + self.burst_bytes - 1) // self.burst_bytes) * self.burst_bytes
    if self.bytes_per_cycle > 0:
      xfer_cycles = int(max((burst_rounded + self.bytes_per_cycle - 1) // self.bytes_per_cycle, 1))
    else:
      xfer_cycles = 0
    completion = cycle + self.fixed_latency_cycles + xfer_cycles
    self._busy_until[free_idx] = completion
    self._holders[free_idx] = transaction_id
    if self.max_outstanding is not None:
      self._outstanding_txns.add(transaction_id)
    return StageResult(accepted_cycle=cycle, completion_cycle=completion, channels=1, resources=(free_idx,))

  def _release_resources(self, transaction_id: str) -> None:
    self._outstanding_txns.discard(transaction_id)
    for i, holder in enumerate(self._holders):
      if holder == transaction_id:
        self._busy_until[i] = 0
        self._holders[i] = None

  def release_outstanding(self, transaction_id: str) -> None:
    """Return one outstanding credit and free the transaction's resources.

    Idempotent: safe to call after a leg completes or after cancel.
    """
    self._release_resources(transaction_id)

  def cancel(self, transaction_id: str) -> None:
    """Free every resource and outstanding credit held by a cancelled
    transaction (idempotent)."""
    self._release_resources(transaction_id)

  def step(self, cycle: int) -> None:
    """Advance one cycle: reconcile resources whose busy window expired.

    Clears expired holder references and returns outstanding credits
    for transactions whose reservation lapsed without an explicit
    release (e.g. abandoned after cancel paths).
    """
    for i, holder in enumerate(self._holders):
      if holder is not None and cycle >= self._busy_until[i]:
        self._outstanding_txns.discard(holder)
        self._busy_until[i] = 0
        self._holders[i] = None

  def reset(self) -> None:
    self._busy_until = [0] * self.resource_count
    self._holders = [None] * self.resource_count
    self._outstanding_txns.clear()
    self.wait_cycles = 0

  def snapshot(self) -> dict:
    busy = sum(1 for b in self._busy_until if b > 0)
    return {
      "name": self.name,
      "resource_count": self.resource_count,
      "busy_resources": busy,
      "outstanding": self._outstanding,
      "max_outstanding": self.max_outstanding,
      "wait_cycles": self.wait_cycles,
    }


# ---------------------------------------------------------------------------
# Transfer manager
# ---------------------------------------------------------------------------


class TransferManager:
  """Manages all in-flight memory transactions and their per-leg routes.

  Does not hold sequencer/Tile UCE references; ``TileGroup`` maps
  transaction ids to sequencers.  Consumers must ``acknowledge()`` after
  handling final completion.
  """

  def __init__(
    self,
    cfg,
    full_memory: bool = False,
    noc=None,
    trace=None,
    *,
    byte_store=None,
    generation_validator: Callable[[MemoryTransaction, str], bool] | None = None,
    tombstone_capacity: int | None = None,
    reference_acquire: Callable[[MemoryTransaction, int], None] | None = None,
    reference_release: Callable[[MemoryTransaction, int], None] | None = None,
  ):
    self.cfg = cfg
    self.full_memory = full_memory
    self.noc = noc  # NoCRouter (full_memory); None when fabric unmodeled
    self.trace = trace  # MemoryTrace sink; None disables event emission
    self.byte_store = byte_store
    self.generation_validator = generation_validator
    if (reference_acquire is None) != (reference_release is None):
      raise ValueError("transfer reference hooks must be configured together")
    self.reference_acquire = reference_acquire
    self.reference_release = reference_release
    default_tombstones = cfg.hbm_outstanding_limit + 4 * cfg.noc_vc_depth
    self.tombstone_capacity = default_tombstones if tombstone_capacity is None else tombstone_capacity
    if self.tombstone_capacity <= 0:
      raise ValueError("transfer tombstone_capacity must be > 0")
    clock_hz = cfg.clock_mhz * 1e6
    self._transactions: dict[str, MemoryTransaction] = {}
    self._completed: set[str] = set()
    self._cancelled: set[str] = set()
    self._faulted: set[str] = set()
    self._terminal_history: OrderedDict[str, TransferStatus] = OrderedDict()
    self._cancel_requested: set[str] = set()
    # Run/profile generation provider state.  Transactions must stamp
    # ``run_generation`` and ``profile_generations`` explicitly; the manager
    # never synthesises or repairs them.
    self._run_generation: int = -1
    self._profile_generations: dict[tuple[str, int], int] = {}
    self._profile_configured: bool = False
    # NoC flit traversal records: tag -> cycle the flit left the router
    self._noc_traversed: dict[str, int] = {}
    # cumulative counters (aggregated as deltas by TileGroup._aggregate_pmu)
    self.pmu_issued_count: int = 0
    self.pmu_completed_count: int = 0
    self.pmu_cancelled_count: int = 0
    self.pmu_faulted_count: int = 0
    self.pmu_noc_credit_wait_cycles: int = 0
    self.pmu_hbm_outstanding_peak: int = 0
    # all-time max (never reset per cycle) for snapshot reconciliation
    self.pmu_hbm_outstanding_peak_max: int = 0
    # Completed-transaction bytes per interface (one entry per leg kind).
    # ``_byte_deltas`` holds the delta since the last TileGroup._aggregate_pmu
    # call; ``pmu_bytes_total`` is cumulative for the run snapshot.
    self._byte_counters: dict[TransferLegKind, str] = {
      TransferLegKind.HBM_READ: "hbm_read_bytes",
      TransferLegKind.HBM_WRITE: "hbm_write_bytes",
      TransferLegKind.GLOBAL_DMA: "global_dma_bytes",
      TransferLegKind.NOC_REQUEST: "noc_request_bytes",
      TransferLegKind.NOC_RESPONSE: "noc_response_bytes",
      TransferLegKind.L2_READ: "l2_read_bytes",
      TransferLegKind.L2_WRITE: "l2_write_bytes",
      TransferLegKind.LOCAL_DMA: "local_dma_bytes",
      TransferLegKind.L1_READ: "l1_read_bytes",
      TransferLegKind.L1_WRITE: "l1_write_bytes",
    }
    self._byte_deltas: dict[str, int] = {}
    self.pmu_bytes_total: dict[str, int] = {}
    self._issued_by_op: dict[str, int] = {}
    # One global outstanding CAM/credit pool shared by HBM reads+writes.
    self._hbm_outstanding_txns: set[str] = set()
    self._hbm_read = TransferStage(
      "hbm_read",
      StageWaitReason.HBM_OUTSTANDING,
      cfg.hbm_fixed_latency_cycles,
      (cfg.hbm_bandwidth_gbs / cfg.hbm_channels) * 1e9 / clock_hz,
      cfg.hbm_channels,
      cfg.hbm_burst_bytes,
      max_outstanding=cfg.hbm_outstanding_limit,
      shared_outstanding=self._hbm_outstanding_txns,
    )
    self._hbm_write = TransferStage(
      "hbm_write",
      StageWaitReason.HBM_OUTSTANDING,
      cfg.hbm_fixed_latency_cycles,
      (cfg.hbm_bandwidth_gbs / cfg.hbm_channels) * 1e9 / clock_hz,
      cfg.hbm_channels,
      cfg.hbm_burst_bytes,
      max_outstanding=cfg.hbm_outstanding_limit,
      shared_outstanding=self._hbm_outstanding_txns,
    )
    self._global_dma = TransferStage(
      "global_dma",
      StageWaitReason.DMA_QUEUE,
      (cfg.dma_launch_cycles + cfg.dma_desc_cycles + cfg.dma_issue_cycles + cfg.dma_completion_cycles),
      cfg.group_dma_bandwidth_gbs * 1e9 / clock_hz,
      cfg.num_dma_channels,
      1,
    )
    # L2 bank stages — per-bank bandwidth; segments issue to specific banks
    l2_bw = cfg.l2_bank_bandwidth_gbs * 1e9 / clock_hz
    self._l2_read = TransferStage(
      "l2_read", StageWaitReason.L2_BANK, cfg.l2_access_latency_cycles, l2_bw, cfg.group_sram_banks, 1
    )
    self._l2_write = TransferStage(
      "l2_write", StageWaitReason.L2_BANK, cfg.l2_access_latency_cycles, l2_bw, cfg.group_sram_banks, 1
    )
    # local DMA + L1 (per-tile, created on demand)
    self._local_dma: dict[tuple[int, str], TransferStage] = {}
    self._l1_read: dict[int, TransferStage] = {}
    self._l1_write: dict[int, TransferStage] = {}
    self._l2_cache_lookup = TransferStage(
      "l2_cache_lookup", StageWaitReason.L2_CACHE, cfg.l2_cache_lookup_latency_cycles, l2_bw, 1, 1
    )
    self._l2_cache_fill = TransferStage(
      "l2_cache_fill", StageWaitReason.L2_CACHE, cfg.l2_cache_lookup_latency_cycles, l2_bw, 1, 1
    )
    self._l1_cache_lookup: dict[int, TransferStage] = {}
    self._l1_cache_fill: dict[int, TransferStage] = {}

  def _local_dma_stage(self, tile_id: int, direction: str) -> TransferStage:
    key = (tile_id, direction)
    if key not in self._local_dma:
      clock_hz = self.cfg.clock_mhz * 1e6
      count = self.cfg.mfe_load_channels if direction == "load" else self.cfg.mfe_store_channels
      self._local_dma[key] = TransferStage(
        f"local_dma_t{tile_id}_{direction}",
        StageWaitReason.DMA_QUEUE,
        self.cfg.mfe_launch_cycles,
        self.cfg.mfe_bandwidth_gbs * 1e9 / clock_hz,
        count,
        1,
      )
    return self._local_dma[key]

  def _l1_stage(self, tile_id: int, is_write: bool) -> TransferStage:
    cache = self._l1_write if is_write else self._l1_read
    if tile_id not in cache:
      clock_hz = self.cfg.clock_mhz * 1e6
      bw = (self.cfg.tile_l1_bandwidth_gbs / self.cfg.tile_l1_banks) * 1e9 / clock_hz
      cache[tile_id] = TransferStage(
        f"l1_{'write' if is_write else 'read'}_t{tile_id}",
        StageWaitReason.L1_BANK,
        self.cfg.l1_access_latency_cycles,
        bw,
        self.cfg.tile_l1_banks,
        1,
      )
    return cache[tile_id]

  def _l1_cache_stage(self, tile_id: int, is_fill: bool) -> TransferStage:
    stages = self._l1_cache_fill if is_fill else self._l1_cache_lookup
    if tile_id not in stages:
      clock_hz = self.cfg.clock_mhz * 1e6
      stages[tile_id] = TransferStage(
        f"l1_cache_{'fill' if is_fill else 'lookup'}_t{tile_id}",
        StageWaitReason.L1_CACHE,
        self.cfg.l1_cache_lookup_latency_cycles,
        self.cfg.tile_l1_bandwidth_gbs * 1e9 / clock_hz,
        1,
        1,
      )
    return stages[tile_id]

  def configure_profile_generations(self, generations) -> None:
    """Install the initial profile-generation map after Controller reset."""
    if not isinstance(generations, Mapping):
      raise TypeError("profile generations must be a mapping")
    normalized: dict[tuple[str, int], int] = {}
    for key, value in generations.items():
      if (
        not isinstance(key, tuple)
        or len(key) != 2
        or key[0] not in ("l1", "l2")
        or not isinstance(key[1], int)
        or not isinstance(value, int)
        or value < 0
      ):
        raise ValueError("profile generation entries must be ((level, pool_id), non-negative int)")
      normalized[key] = value
    if self._transactions:
      raise MemoryInvariantError("cannot configure profile generations with live transfers")
    self._profile_generations = normalized
    self._profile_configured = True

  def note_profile_commit(self, level: str, pool_ids: tuple[int, ...], generation: int) -> None:
    if level not in ("l1", "l2"):
      raise ValueError("profile level must be l1 or l2")
    if not isinstance(generation, int) or generation < 0:
      raise ValueError("profile generation must be a non-negative integer")
    if not self._profile_configured:
      raise MemoryInvariantError("profile generations are not configured")
    for pool_id in pool_ids:
      key = (level, pool_id)
      current = self._profile_generations.get(key)
      if current is None:
        raise MemoryInvariantError("profile commit references an unknown pool")
      if generation <= current:
        raise MemoryInvariantError("profile commit generation must increase monotonically")
      self._profile_generations[key] = generation

  def begin_run(self, run_generation: int) -> None:
    if not isinstance(run_generation, int) or run_generation < 0:
      raise ValueError("run generation must be a non-negative integer")
    if run_generation <= self._run_generation:
      raise MemoryInvariantError("transfer run generation must increase monotonically")
    if self._transactions:
      raise MemoryInvariantError("cannot begin a new run with live transfers")
    self._run_generation = run_generation

  def transaction_identity(
    self, domains: tuple[tuple[str, int], ...]
  ) -> tuple[int, tuple[tuple[str, int, int], ...]]:
    if not self._profile_configured:
      raise MemoryInvariantError("profile generations are not configured")
    if self._run_generation < 0:
      raise MemoryInvariantError("transfer run has not begun")
    if not isinstance(domains, tuple) or not domains:
      raise ValueError("transaction identity requires at least one domain")
    frozen: list[tuple[str, int, int]] = []
    for domain in domains:
      if not isinstance(domain, tuple) or len(domain) != 2:
        raise ValueError("transaction domains must be (level, pool_id)")
      generation = self._profile_generations.get(domain)
      if generation is None:
        raise MemoryInvariantError(f"transaction identity references unknown profile domain {domain!r}")
      frozen.append((domain[0], domain[1], generation))
    return self._run_generation, tuple(frozen)

  def _validate_identity(self, txn: MemoryTransaction) -> bool:
    if not self._profile_configured:
      return True
    if txn.run_generation != self._run_generation or not txn.profile_generations:
      return False
    seen: set[tuple[str, int]] = set()
    for level, pool_id, generation in txn.profile_generations:
      domain = (level, pool_id)
      if domain in seen:
        return False
      seen.add(domain)
      if self._profile_generations.get(domain) != generation:
        return False
    return True

  def set_generation_validator(self, validator: Callable[[MemoryTransaction, str], bool]) -> None:
    self.generation_validator = validator

  def set_byte_store(self, byte_store) -> None:
    self.byte_store = byte_store

  def _generation_valid(self, txn: MemoryTransaction, phase: str) -> bool:
    if not self._validate_identity(txn):
      return False
    if not txn.profile_generations:
      return True
    if self.generation_validator is None:
      return False
    try:
      return bool(self.generation_validator(txn, phase))
    except Exception:
      return False

  def _record_terminal(self, transaction_id: str, status: TransferStatus) -> None:
    if status not in (TransferStatus.DONE, TransferStatus.CANCELLED, TransferStatus.FAULTED):
      raise MemoryInvariantError("non-terminal status entered tombstone ledger")
    previous = self._terminal_history.pop(transaction_id, None)
    if previous is not None:
      if previous is not status:
        raise MemoryInvariantError("transaction terminal status changed")
      self._terminal_history[transaction_id] = status
      return
    while len(self._terminal_history) >= self.tombstone_capacity:
      evict_id = next(
        (terminal_id for terminal_id in self._terminal_history if terminal_id not in self._transactions),
        None,
      )
      if evict_id is None:
        raise MemoryInvariantError("transfer tombstone capacity exhausted before acknowledgement")
      evicted = self._terminal_history.pop(evict_id)
      if evicted is TransferStatus.DONE:
        self._completed.discard(evict_id)
      elif evicted is TransferStatus.CANCELLED:
        self._cancelled.discard(evict_id)
      else:
        self._faulted.discard(evict_id)
    self._terminal_history[transaction_id] = status
    if status is TransferStatus.DONE:
      self._completed.add(transaction_id)
    elif status is TransferStatus.CANCELLED:
      self._cancelled.add(transaction_id)
    else:
      self._faulted.add(transaction_id)

  def _fault_transaction(self, txn: MemoryTransaction, reason: str, cycle: int) -> None:
    if txn.status is TransferStatus.FAULTED:
      return
    txn.status = TransferStatus.FAULTED
    txn.fault_reason = reason
    txn.completed_cycle = cycle
    self._record_terminal(txn.transaction_id, TransferStatus.FAULTED)
    self._cancel_requested.discard(txn.transaction_id)
    self.pmu_faulted_count += 1

  @property
  def available_issue_slots(self) -> int:
    """Conservative capacity for work which may require a terminal tombstone."""
    return max(0, self.tombstone_capacity - len(self._transactions))

  @property
  def outstanding_transactions(self) -> tuple[str, ...]:
    """IDs of accepted transactions not yet acknowledged (leak inventory)."""
    return tuple(self._transactions)

  @property
  def can_accept_new(self) -> bool:
    retained = sum(
      txn.status in (TransferStatus.CANCELLED, TransferStatus.FAULTED)
      for txn in self._transactions.values()
    )
    return retained < self.tombstone_capacity

  def submit(self, transaction: MemoryTransaction, cycle: int, pmu=None) -> None:
    """Submit a transaction and freeze its explicit route.

    Cancel/fault tombstones are bounded.  When the bound is full, only new
    issue is blocked; existing completions and isolation continue to advance.
    """
    if (
      transaction.transaction_id in self._transactions
      or transaction.transaction_id in self._terminal_history
    ):
      raise MemoryInvariantError("duplicate transaction id")
    if not self.can_accept_new:
      raise MemoryInvariantError("transfer tombstone capacity is full")
    if transaction.bytes_total <= 0:
      raise MemoryInvariantError("transfer bytes_total must be positive")
    if transaction.src is not None and transaction.src.size_bytes != transaction.bytes_total:
      raise MemoryInvariantError("transfer source byte count mismatch")
    if transaction.dst is not None and transaction.dst.size_bytes != transaction.bytes_total:
      raise MemoryInvariantError("transfer destination byte count mismatch")
    if transaction.captured_data is not None and len(transaction.captured_data) != transaction.bytes_total:
      raise MemoryInvariantError("captured transfer data has the wrong byte count")
    self._transactions[transaction.transaction_id] = transaction
    self.pmu_issued_count += 1
    op_name = transaction.op.value
    self._issued_by_op[op_name] = self._issued_by_op.get(op_name, 0) + 1
    if not self._generation_valid(transaction, "submit"):
      self._fault_transaction(transaction, "stale profile/allocation generation at submit", cycle)
      return
    if any(level not in ("l1", "l2") for level in transaction.bypass_levels):
      raise MemoryInvariantError("transfer bypass level must be l1 or l2")
    gather_ops = (
      TransferOp.GATHER_L1_LOOKUP,
      TransferOp.GATHER_L2_LOOKUP,
      TransferOp.GATHER_L2_RESPONSE,
      TransferOp.GATHER_HBM_REFILL,
      TransferOp.GATHER_L2_REFILL,
      TransferOp.GATHER_DEST_WRITE,
      TransferOp.GATHER_DIRECT_L1_REFILL,
      TransferOp.GATHER_DIRECT_RESPONSE,
    )
    clean_ops = (TransferOp.CACHE_CLEAN_L1, TransferOp.CACHE_CLEAN_L2)
    single_endpoint_ops = (TransferOp.INDEX_READ,)
    scatter_ops = (TransferOp.SCATTER_WRITE,)
    host_ops = (TransferOp.HOST_READ, TransferOp.HOST_WRITE)
    if (
      transaction.op in gather_ops
      or transaction.op in clean_ops
      or transaction.op in single_endpoint_ops
      or transaction.op in scatter_ops
      or transaction.op in host_ops
    ):
      transaction.legs = self._build_route(transaction)
    elif transaction.src is None and transaction.dst is None:
      transaction.legs = self._collapsed_leg(transaction)
    elif transaction.src is not None and transaction.dst is not None:
      transaction.legs = (
        self._build_route(transaction) if self.full_memory else self._collapsed_leg(transaction)
      )
    else:
      self._fault_transaction(transaction, "transfer route has only one resolved endpoint", cycle)
      return
    if transaction.bypass_levels:
      blocked_kinds: set[TransferLegKind] = set()
      if "l1" in transaction.bypass_levels:
        blocked_kinds.update((TransferLegKind.L1_CACHE_LOOKUP, TransferLegKind.L1_CACHE_FILL))
      if "l2" in transaction.bypass_levels:
        blocked_kinds.update((TransferLegKind.L2_CACHE_LOOKUP, TransferLegKind.L2_CACHE_FILL))
      transaction.legs = tuple(leg for leg in transaction.legs if leg.kind not in blocked_kinds)
    if not transaction.legs:
      self._fault_transaction(transaction, "transfer route is empty", cycle)
      return
    if self._has_l2_endpoint(transaction):
      if self.reference_acquire is None:
        raise MemoryInvariantError(
          f"transfer {transaction.transaction_id} holds a physical L2 endpoint"
          " but no reference_acquire hook is configured"
        )
      # One atomic registration for every deduplicated L2 src/dst view and
      # its backing, before any transport issue.  A hook failure rolls back
      # its own partial registration; the accepted-but-unissued transaction
      # is faulted and never releases a reference it did not acquire.
      try:
        self.reference_acquire(transaction, cycle)
      except Exception as exc:
        self._fault_transaction(transaction, f"L2 reference acquisition failed: {exc}", cycle)
        return
      transaction.reference_acquired = True
    transaction.status = TransferStatus.RUNNING
    transaction.current_leg = 0
    transaction.leg_start_cycle = -1

  @staticmethod
  def _has_l2_endpoint(transaction: MemoryTransaction) -> bool:
    """True when an endpoint view is bound to a physical L2 backing.

    Synthetic unit-test views carry an L2 memory space but no backing
    identity; only physically bound views require owner reference hooks.
    """
    for view in (transaction.src, transaction.dst):
      if view is not None and view.handle.memory_space == "l2" and view.handle.backing_id:
        return True
    return False

  def _collapsed_leg(self, txn: MemoryTransaction) -> tuple[TransferLeg, ...]:
    """One collapsed leg (existing bandwidth + launch overhead) per route.

    Group ops fold onto the Global DMA stage; tile-local ops fold onto
    the tile's local DMA stage (mfe_launch_cycles + mfe bandwidth).
    """
    if txn.op in self._GATHER_ROUTE_OPS:
      raise MemoryInvariantError("gather route must not be collapsed")
    if txn.op in (TransferOp.TILE_LOAD, TransferOp.TILE_STORE):
      tid = txn.tile_id or 0
      direction = "load" if txn.op == TransferOp.TILE_LOAD else "store"
      return (
        TransferLeg(
          TransferLegKind.LOCAL_DMA,
          "l2",
          "l1",
          txn.bytes_total,
          f"local_dma:{tid}:{direction}:{txn.transaction_id}",
        ),
      )
    return (
      TransferLeg(TransferLegKind.GLOBAL_DMA, "hbm", "l2", txn.bytes_total, f"gdma:{txn.transaction_id}"),
    )

  _GATHER_ROUTE_OPS = frozenset(
    {
      TransferOp.GATHER_L1_LOOKUP,
      TransferOp.GATHER_L2_LOOKUP,
      TransferOp.GATHER_L2_RESPONSE,
      TransferOp.GATHER_HBM_REFILL,
      TransferOp.GATHER_L2_REFILL,
      TransferOp.GATHER_DEST_WRITE,
      TransferOp.GATHER_DIRECT_L1_REFILL,
      TransferOp.GATHER_DIRECT_RESPONSE,
    }
  )

  def _build_route(self, txn: MemoryTransaction) -> tuple[TransferLeg, ...]:
    op = txn.op
    tid = txn.tile_id or 0
    txn_id = txn.transaction_id
    if op == TransferOp.INDEX_READ:
      return (
        TransferLeg(
          TransferLegKind.L1_READ,
          "l1",
          "l1",
          txn.bytes_total,
          f"l1_read:{tid}:index:{txn_id}",
        ),
      )
    if op == TransferOp.GATHER_L1_LOOKUP:
      return (
        TransferLeg(
          TransferLegKind.L1_CACHE_LOOKUP,
          "l1_cache",
          "l1_cache",
          txn.bytes_total,
          f"l1_cache_lookup:{tid}:{txn_id}",
        ),
      )
    if op == TransferOp.GATHER_L2_LOOKUP:
      return (
        TransferLeg(
          TransferLegKind.L2_CACHE_LOOKUP,
          "l2_cache",
          "l2_cache",
          txn.bytes_total,
          f"l2_cache_lookup:{txn_id}",
        ),
      )
    if op == TransferOp.GATHER_L2_RESPONSE:
      return (
        TransferLeg(
          TransferLegKind.NOC_RESPONSE, "noc", "tile", txn.bytes_total, f"noc_rsp:{txn_id}"
        ),
        TransferLeg(
          TransferLegKind.LOCAL_DMA,
          "tile",
          "l1",
          txn.bytes_total,
          f"local_dma:{tid}:load:{txn_id}",
        ),
      )
    if op == TransferOp.GATHER_HBM_REFILL:
      return (
        TransferLeg(
          TransferLegKind.HBM_READ, "hbm", "noc", txn.bytes_total, f"hbm_read:{txn_id}"
        ),
        TransferLeg(
          TransferLegKind.NOC_RESPONSE, "noc", "l2_cache", txn.bytes_total, f"noc_rsp:{txn_id}"
        ),
        TransferLeg(
          TransferLegKind.L2_CACHE_FILL,
          "l2_cache",
          "l2_cache",
          txn.bytes_total,
          f"l2_cache_fill:{txn_id}",
        ),
      )
    if op == TransferOp.GATHER_L2_REFILL:
      return (
        TransferLeg(
          TransferLegKind.NOC_RESPONSE, "l2_cache", "tile", txn.bytes_total, f"noc_rsp:{txn_id}"
        ),
        TransferLeg(
          TransferLegKind.LOCAL_DMA,
          "tile",
          "l1_cache",
          txn.bytes_total,
          f"local_dma:{tid}:load:{txn_id}",
        ),
        TransferLeg(
          TransferLegKind.L1_CACHE_FILL,
          "l1_cache",
          "l1_cache",
          txn.bytes_total,
          f"l1_cache_fill:{tid}:{txn_id}",
        ),
      )
    if op == TransferOp.GATHER_DIRECT_L1_REFILL:
      return (
        TransferLeg(
          TransferLegKind.HBM_READ, "hbm", "noc", txn.bytes_total, f"hbm_read:{txn_id}"
        ),
        TransferLeg(
          TransferLegKind.NOC_RESPONSE, "noc", "tile", txn.bytes_total, f"noc_rsp:{txn_id}"
        ),
        TransferLeg(
          TransferLegKind.LOCAL_DMA,
          "tile",
          "l1_cache",
          txn.bytes_total,
          f"local_dma:{tid}:load:{txn_id}",
        ),
        TransferLeg(
          TransferLegKind.L1_CACHE_FILL,
          "l1_cache",
          "l1_cache",
          txn.bytes_total,
          f"l1_cache_fill:{tid}:{txn_id}",
        ),
      )
    if op == TransferOp.GATHER_DIRECT_RESPONSE:
      return (
        TransferLeg(
          TransferLegKind.HBM_READ, "hbm", "noc", txn.bytes_total, f"hbm_read:{txn_id}"
        ),
        TransferLeg(
          TransferLegKind.NOC_RESPONSE, "noc", "tile", txn.bytes_total, f"noc_rsp:{txn_id}"
        ),
        TransferLeg(
          TransferLegKind.LOCAL_DMA,
          "tile",
          "l1",
          txn.bytes_total,
          f"local_dma:{tid}:load:{txn_id}",
        ),
      )
    if op == TransferOp.GATHER_DEST_WRITE:
      return (
        TransferLeg(
          TransferLegKind.L1_WRITE, "l1", "l1", txn.bytes_total, f"l1_write:{tid}:{txn_id}"
        ),
      )
    if op == TransferOp.SCATTER_WRITE:
      return (
        TransferLeg(
          TransferLegKind.L1_READ, "l1", "tile", txn.bytes_total, f"l1_read:{tid}:{txn_id}"
        ),
        TransferLeg(
          TransferLegKind.LOCAL_DMA,
          "tile",
          "noc",
          txn.bytes_total,
          f"local_dma:{tid}:store:{txn_id}",
        ),
        TransferLeg(
          TransferLegKind.NOC_REQUEST, "tile", "noc", txn.bytes_total, f"noc_req:{txn_id}"
        ),
        TransferLeg(
          TransferLegKind.GLOBAL_DMA, "noc", "noc", txn.bytes_total, f"gdma:{txn_id}"
        ),
        TransferLeg(
          TransferLegKind.HBM_WRITE, "noc", "hbm", txn.bytes_total, f"hbm_write:{txn_id}"
        ),
      )
    if op == TransferOp.HOST_READ:
      return (
        TransferLeg(TransferLegKind.HBM_READ, "hbm", "noc", txn.bytes_total, f"hbm_read:{txn_id}"),
        TransferLeg(TransferLegKind.GLOBAL_DMA, "noc", "noc", txn.bytes_total, f"gdma:{txn_id}"),
        TransferLeg(
          TransferLegKind.NOC_RESPONSE, "noc", "host", txn.bytes_total, f"noc_rsp:{txn_id}"
        ),
      )
    if op == TransferOp.HOST_WRITE:
      return (
        TransferLeg(
          TransferLegKind.NOC_REQUEST, "host", "noc", txn.bytes_total, f"noc_req:{txn_id}"
        ),
        TransferLeg(TransferLegKind.GLOBAL_DMA, "noc", "noc", txn.bytes_total, f"gdma:{txn_id}"),
        TransferLeg(TransferLegKind.HBM_WRITE, "noc", "hbm", txn.bytes_total, f"hbm_write:{txn_id}"),
      )
    if op == TransferOp.CACHE_CLEAN_L1:
      return (
        TransferLeg(
          TransferLegKind.LOCAL_DMA,
          "l1_cache",
          "l2",
          txn.bytes_total,
          f"local_dma:{tid}:store:{txn.transaction_id}",
        ),
        TransferLeg(
          TransferLegKind.L2_WRITE, "tile", "l2", txn.bytes_total, f"l2_write:{txn.transaction_id}"
        ),
        TransferLeg(
          TransferLegKind.NOC_REQUEST, "l2", "noc", txn.bytes_total, f"noc_req:{txn.transaction_id}"
        ),
        TransferLeg(
          TransferLegKind.GLOBAL_DMA, "noc", "noc", txn.bytes_total, f"gdma:{txn.transaction_id}"
        ),
        TransferLeg(
          TransferLegKind.HBM_WRITE, "noc", "hbm", txn.bytes_total, f"hbm_write:{txn.transaction_id}"
        ),
      )
    if op == TransferOp.CACHE_CLEAN_L2:
      return (
        TransferLeg(
          TransferLegKind.NOC_REQUEST, "l2_cache", "noc", txn.bytes_total, f"noc_req:{txn.transaction_id}"
        ),
        TransferLeg(
          TransferLegKind.GLOBAL_DMA, "noc", "noc", txn.bytes_total, f"gdma:{txn.transaction_id}"
        ),
        TransferLeg(
          TransferLegKind.HBM_WRITE, "noc", "hbm", txn.bytes_total, f"hbm_write:{txn.transaction_id}"
        ),
      )
    if op == TransferOp.PREFETCH:
      return (
        TransferLeg(
          TransferLegKind.HBM_READ, "hbm", "noc", txn.bytes_total, f"hbm_read:{txn.transaction_id}"
        ),
        TransferLeg(
          TransferLegKind.GLOBAL_DMA, "noc", "noc", txn.bytes_total, f"gdma:{txn.transaction_id}"
        ),
        TransferLeg(
          TransferLegKind.NOC_RESPONSE, "noc", "l2", txn.bytes_total, f"noc_rsp:{txn.transaction_id}"
        ),
        TransferLeg(
          TransferLegKind.L2_WRITE, "noc", "l2", txn.bytes_total, f"l2_write:{txn.transaction_id}"
        ),
      )
    if op == TransferOp.GLOBAL_STORE:
      return (
        TransferLeg(TransferLegKind.L2_READ, "l2", "noc", txn.bytes_total, f"l2_read:{txn.transaction_id}"),
        TransferLeg(
          TransferLegKind.NOC_REQUEST, "noc", "noc", txn.bytes_total, f"noc_req:{txn.transaction_id}"
        ),
        TransferLeg(
          TransferLegKind.GLOBAL_DMA, "noc", "noc", txn.bytes_total, f"gdma:{txn.transaction_id}"
        ),
        TransferLeg(
          TransferLegKind.HBM_WRITE, "noc", "hbm", txn.bytes_total, f"hbm_write:{txn.transaction_id}"
        ),
      )
    if op == TransferOp.TILE_LOAD:
      tid = txn.tile_id or 0
      return (
        TransferLeg(
          TransferLegKind.L2_READ, "l2", "tile", txn.bytes_total, f"l2_read:{txn.transaction_id}"
        ),
        TransferLeg(
          TransferLegKind.LOCAL_DMA,
          "l2",
          "tile",
          txn.bytes_total,
          f"local_dma:{tid}:load:{txn.transaction_id}",
        ),
        TransferLeg(
          TransferLegKind.L1_WRITE, "tile", "l1", txn.bytes_total, f"l1_write:{tid}:{txn.transaction_id}"
        ),
      )
    if op == TransferOp.TILE_STORE:
      tid = txn.tile_id or 0
      return (
        TransferLeg(
          TransferLegKind.L1_READ, "l1", "tile", txn.bytes_total, f"l1_read:{tid}:{txn.transaction_id}"
        ),
        TransferLeg(
          TransferLegKind.LOCAL_DMA,
          "tile",
          "l2",
          txn.bytes_total,
          f"local_dma:{tid}:store:{txn.transaction_id}",
        ),
        TransferLeg(
          TransferLegKind.L2_WRITE, "tile", "l2", txn.bytes_total, f"l2_write:{txn.transaction_id}"
        ),
      )
    return ()

  def _stage_for_leg(self, leg: TransferLeg, txn: MemoryTransaction) -> TransferStage:
    kind = leg.kind
    if kind == TransferLegKind.HBM_READ:
      return self._hbm_read
    if kind == TransferLegKind.HBM_WRITE:
      return self._hbm_write
    if kind == TransferLegKind.GLOBAL_DMA:
      return self._global_dma
    if kind in (TransferLegKind.NOC_RESPONSE, TransferLegKind.NOC_REQUEST):
      raise ValueError(f"NoC leg {kind} is router-backed, not a stage")
    if kind == TransferLegKind.L2_READ:
      return self._l2_read
    if kind == TransferLegKind.L2_WRITE:
      return self._l2_write
    if kind == TransferLegKind.LOCAL_DMA:
      tid = txn.tile_id or 0
      direction = "store" if txn.op in (TransferOp.TILE_STORE, TransferOp.CACHE_CLEAN_L1) else "load"
      return self._local_dma_stage(tid, direction)
    if kind == TransferLegKind.L1_READ:
      return self._l1_stage(txn.tile_id or 0, is_write=False)
    if kind == TransferLegKind.L1_WRITE:
      return self._l1_stage(txn.tile_id or 0, is_write=True)
    if kind == TransferLegKind.L1_CACHE_LOOKUP:
      return self._l1_cache_stage(txn.tile_id or 0, is_fill=False)
    if kind == TransferLegKind.L2_CACHE_LOOKUP:
      return self._l2_cache_lookup
    if kind == TransferLegKind.L1_CACHE_FILL:
      return self._l1_cache_stage(txn.tile_id or 0, is_fill=True)
    if kind == TransferLegKind.L2_CACHE_FILL:
      return self._l2_cache_fill
    raise ValueError(f"unknown leg kind {kind}")

  @staticmethod
  def _leg_view(leg: TransferLeg, txn: MemoryTransaction) -> ResolvedMemoryView | None:
    """Return the resolved view whose segments a bank-based leg accesses."""
    kind = leg.kind
    if kind == TransferLegKind.L2_READ:
      return txn.src if txn.src is not None and txn.src.handle.memory_space == "l2" else None
    if kind == TransferLegKind.L1_READ:
      return txn.src if txn.src is not None and txn.src.handle.memory_space == "l1" else None
    if kind == TransferLegKind.L2_WRITE:
      return txn.dst if txn.dst is not None and txn.dst.handle.memory_space == "l2" else None
    if kind == TransferLegKind.L1_WRITE:
      return txn.dst if txn.dst is not None and txn.dst.handle.memory_space == "l1" else None
    return None

  def _requests_for_leg(self, leg: TransferLeg, txn: MemoryTransaction) -> list[StageRequest]:
    """Build deterministic resource requests for one stage-backed leg.

    HBM is channel-addressed, not first-free:
    ``(global_address // hbm_burst_bytes) % hbm_channels``.
    L2/L1 requests use allocator-resolved bank segments.
    """
    if leg.kind in (TransferLegKind.HBM_READ, TransferLegKind.HBM_WRITE):
      view = txn.src if leg.kind == TransferLegKind.HBM_READ else txn.dst
      if view is not None:
        channel = (view.address // self.cfg.hbm_burst_bytes) % self.cfg.hbm_channels
        return [StageRequest(str(channel), leg.bytes_total)]
    view = self._leg_view(leg, txn)
    if view is not None and view.segments:
      return [StageRequest(str(seg.bank_id), seg.size_bytes) for seg in view.segments]
    return [StageRequest(leg.resource_id, leg.bytes_total)]

  def _all_stages(self) -> list[TransferStage]:
    """Every stage this manager owns, including per-tile local stages."""
    stages: list[TransferStage] = [
      self._hbm_read,
      self._hbm_write,
      self._global_dma,
      self._l2_read,
      self._l2_write,
      self._l2_cache_lookup,
      self._l2_cache_fill,
    ]
    stages.extend(self._local_dma.values())
    stages.extend(self._l1_read.values())
    stages.extend(self._l1_write.values())
    stages.extend(self._l1_cache_lookup.values())
    stages.extend(self._l1_cache_fill.values())
    return stages

  def note_traversed(self, flits: list[Flit], cycle: int) -> None:
    """Register real traversals and fault/isolate unknown late returns."""
    for flit in flits:
      if not flit.tag:
        continue
      owner = next((txn for txn in self._transactions.values() if txn.noc_tag == flit.tag), None)
      if owner is None or owner.status in (
        TransferStatus.DONE,
        TransferStatus.CANCELLED,
        TransferStatus.FAULTED,
      ):
        if self.noc is not None and self.noc.traversed(flit.tag):
          self.noc.return_credit(flit.vc, 1, tag=flit.tag)
        transaction_id = flit.tag.rsplit(":", 1)[0]
        terminal = self._terminal_history.get(transaction_id)
        detail = terminal.value if terminal is not None else "unknown/evicted"
        raise MemoryInvariantError(f"late NoC return for {detail} transaction {transaction_id}")
      self._noc_traversed[flit.tag] = cycle

  @staticmethod
  def _leg_reads_source(leg: TransferLeg, txn: MemoryTransaction) -> bool:
    if txn.src is None:
      return False
    return (
      (leg.kind == TransferLegKind.HBM_READ and txn.src.handle.memory_space == "hbm")
      or (leg.kind == TransferLegKind.L2_READ and txn.src.handle.memory_space == "l2")
      or (leg.kind == TransferLegKind.L1_READ and txn.src.handle.memory_space == "l1")
    )

  @staticmethod
  def _leg_writes_destination(leg: TransferLeg, txn: MemoryTransaction) -> bool:
    if txn.dst is None:
      return False
    return (
      (leg.kind == TransferLegKind.HBM_WRITE and txn.dst.handle.memory_space == "hbm")
      or (leg.kind == TransferLegKind.L2_WRITE and txn.dst.handle.memory_space == "l2")
      or (leg.kind == TransferLegKind.L1_WRITE and txn.dst.handle.memory_space == "l1")
    )

  def _capture_source(self, txn: MemoryTransaction, cycle: int, *, force: bool = False) -> bool:
    if txn.captured_data is not None or txn.src is None or self.byte_store is None:
      return True
    if not force and txn.current_leg < len(txn.legs):
      leg = txn.legs[txn.current_leg]
      if not self._leg_reads_source(leg, txn):
        return True
    if not self._generation_valid(txn, "source_capture"):
      self._fault_transaction(txn, "old-generation source return isolated", cycle)
      return False
    try:
      if (
        txn.op in _GATHER_HBM_READ_OPS
        and txn.src.handle.memory_space == "hbm"
      ):
        # Gather reads carry per-byte validity: a whole-line refill or a
        # sub-line bypass segment legitimately covers uninitialised bytes
        # (page tail tokens, padding).  Capture data + mask without
        # faulting; only a destination write touching such a byte faults
        # (plan §2 - the fault belongs to the destination write).
        txn.captured_data, txn.captured_validity = self.byte_store.read_view_relaxed(txn.src)
      else:
        txn.captured_data = self.byte_store.read_view(txn.src)
    except Exception as exc:
      self._fault_transaction(txn, f"source byte capture failed: {exc}", cycle)
      return False
    txn.source_captured_cycle = cycle
    return True

  def _commit_destination(self, txn: MemoryTransaction, cycle: int, *, force: bool = False) -> bool:
    if txn.destination_committed_cycle >= 0:
      return True
    if txn.dst is None or self.byte_store is None:
      return True
    if not force and txn.current_leg < len(txn.legs):
      leg = txn.legs[txn.current_leg]
      if not self._leg_writes_destination(leg, txn):
        return True
    if not self._generation_valid(txn, "destination_commit"):
      self._fault_transaction(txn, "old-generation destination return isolated", cycle)
      return False
    if txn.captured_data is None:
      self._fault_transaction(txn, "destination write completed without captured bytes", cycle)
      return False
    try:
      if txn.op in (TransferOp.GATHER_DEST_WRITE, TransferOp.GATHER_DIRECT_RESPONSE):
        # A destination segment covering uninitialised bytes faults here
        # (plan §2: validity is enforced at the destination write).  A
        # bypass leg writes its L1 destination through this same commit,
        # so it is checked here too instead of in the MFE.
        self.byte_store.write_view_checked(txn.dst, txn.captured_data, txn.captured_validity)
      else:
        self.byte_store.write_view(txn.dst, txn.captured_data)
    except Exception as exc:
      self._fault_transaction(txn, f"destination byte commit failed: {exc}", cycle)
      return False
    txn.destination_committed_cycle = cycle
    return True

  def _finish_cancelled(self, txn: MemoryTransaction, cycle: int) -> None:
    txn.status = TransferStatus.CANCELLED
    txn.wait_reason = StageWaitReason.NONE
    txn.completed_cycle = cycle
    self._cancel_requested.discard(txn.transaction_id)
    self._record_terminal(txn.transaction_id, TransferStatus.CANCELLED)
    self.pmu_cancelled_count += 1
    if self.trace is not None:
      self.trace.transfer_cancelled(txn, cycle)

  def step(self, cycle: int) -> tuple[MemoryTransaction, ...]:
    """Advance all transactions one cycle.  Returns newly completed ones."""
    # Reconcile every stage first: expire finished busy windows and return
    # outstanding credits whose holder lapsed without an explicit release.
    for stage in self._all_stages():
      stage.step(cycle)
    completed: list[MemoryTransaction] = []
    for txn in list(self._transactions.values()):
      if txn.status in (TransferStatus.DONE, TransferStatus.FAULTED, TransferStatus.CANCELLED):
        continue
      if not txn.legs:
        continue
      if txn.current_leg >= len(txn.legs):
        continue
      leg = txn.legs[txn.current_leg]
      if leg.kind in (TransferLegKind.NOC_RESPONSE, TransferLegKind.NOC_REQUEST):
        self._step_noc_leg(txn, leg, cycle, completed)
        continue
      stage = self._stage_for_leg(leg, txn)
      if txn.leg_start_cycle < 0:
        if self._leg_reads_source(leg, txn):
          issue_phase = "source_issue"
        elif self._leg_writes_destination(leg, txn):
          issue_phase = "destination_issue"
        else:
          issue_phase = "transport_issue"
        if not self._generation_valid(txn, issue_phase):
          self._fault_transaction(txn, f"old-generation transfer {issue_phase} rejected", cycle)
          continue
        req = self._requests_for_leg(leg, txn)
        result = stage.try_issue(txn.transaction_id, req, cycle)
        if isinstance(result, StageWait):
          txn.wait_reason = result.reason
          stage.wait_cycles += 1
          if self.trace is not None:
            self.trace.transfer_wait(txn.transaction_id, result.reason.value)
          continue
        txn.wait_reason = StageWaitReason.NONE
        txn.leg_start_cycle = result.accepted_cycle
        txn.leg_completion_cycle = result.completion_cycle
        if txn.start_cycle < 0:
          txn.start_cycle = result.accepted_cycle
        if stage.name in ("hbm_read", "hbm_write"):
          # capture the transient: a txn may issue and complete in the
          # same cycle, so the end-of-step peak check would miss it
          peak = len(self._hbm_outstanding_txns)
          if peak > self.pmu_hbm_outstanding_peak:
            self.pmu_hbm_outstanding_peak = peak
          if peak > self.pmu_hbm_outstanding_peak_max:
            self.pmu_hbm_outstanding_peak_max = peak
        if self.trace is not None:
          self.trace.transfer_leg_issued(txn, leg, stage.name, result, result.resources, cycle)
          if stage.name in ("hbm_read", "hbm_write"):
            self._trace_hbm_outstanding(cycle)
      # check if current leg completed
      if cycle >= txn.leg_completion_cycle and txn.leg_completion_cycle > 0:
        # An accepted leg owns resources until its real completion even after
        # cancellation was requested.
        stage.release_outstanding(txn.transaction_id)
        if self.trace is not None:
          self.trace.transfer_leg_completed(txn, cycle)
          if stage.name in ("hbm_read", "hbm_write"):
            self._trace_hbm_outstanding(cycle)
        if self._leg_reads_source(leg, txn) and not self._capture_source(txn, cycle):
          continue
        if self._leg_writes_destination(leg, txn) and not self._commit_destination(txn, cycle):
          continue
        self._advance_leg(txn, cycle, completed)
    # track HBM outstanding peak after this cycle's issue activity
    peak = len(self._hbm_outstanding_txns)
    if peak > self.pmu_hbm_outstanding_peak:
      self.pmu_hbm_outstanding_peak = peak
    return tuple(completed)

  def _step_noc_leg(
    self, txn: MemoryTransaction, leg: TransferLeg, cycle: int, completed: list[MemoryTransaction]
  ) -> None:
    """Advance one router-backed NoC leg (VC1 response / VC2 request).

    First entry enqueues exactly one flit with a deterministic tag.
    While the flit is pending (queueing, arbitration or insufficient
    credit) the wait reason is ``NOC_CREDIT``.  After the flit
    traverses, the leg waits ``noc_router_latency_cycles`` and then
    returns the downstream credit.
    """
    if self.noc is None:
      # fabric unmodeled: complete immediately (defensive; collapsed
      # routes never contain NoC legs)
      self._advance_leg(txn, cycle, completed)
      return
    vc = normalize_vc_id(
      VCId.VC1_DMA_READ_RSP if leg.kind == TransferLegKind.NOC_RESPONSE else VCId.VC2_DMA_WRITE
    )
    if not txn.noc_tag:
      if not self._generation_valid(txn, "transport_issue"):
        self._fault_transaction(txn, "old-generation NoC transport issue rejected", cycle)
        return
      # first entry: enqueue one flit/tag
      txn.noc_tag = f"{txn.transaction_id}:{leg.kind.value}"
      txn.noc_vc = vc
      txn.leg_start_cycle = cycle
      if txn.start_cycle < 0:
        txn.start_cycle = cycle
      self.noc.send(vc, Flit(vc=vc, src=0, dst=1, bytes_total=leg.bytes_total, tag=txn.noc_tag), cycle)
      txn.wait_reason = StageWaitReason.NOC_CREDIT
      self.pmu_noc_credit_wait_cycles += 1
      if self.trace is not None:
        # router-backed leg: completion cycle unknown at issue (-1); the
        # completion hook substitutes the actual cycle.
        self.trace.transfer_leg_issued(txn, leg, f"noc_vc{vc}", StageResult(cycle, -1), (), cycle)
      return
    traversed = self._noc_traversed.get(txn.noc_tag)
    if traversed is None:
      # still pending: queueing, arbitration or credit exhaustion
      txn.wait_reason = StageWaitReason.NOC_CREDIT
      self.pmu_noc_credit_wait_cycles += 1
      if self.trace is not None:
        self.trace.transfer_wait(txn.transaction_id, StageWaitReason.NOC_CREDIT.value)
      return
    if cycle >= traversed + self.cfg.noc_router_latency_cycles:
      tag = txn.noc_tag
      self.noc.return_credit(txn.noc_vc, 1, tag=tag)
      self._noc_traversed.pop(tag, None)
      if self.trace is not None:
        self.trace.transfer_leg_completed(txn, cycle)
      self._advance_leg(txn, cycle, completed)
    else:
      txn.wait_reason = StageWaitReason.NOC_CREDIT
      self.pmu_noc_credit_wait_cycles += 1

  def _advance_leg(self, txn: MemoryTransaction, cycle: int, completed: list[MemoryTransaction]) -> None:
    """Advance to the next leg, respecting cancel and byte visibility."""
    if txn.noc_tag:
      self._noc_traversed.pop(txn.noc_tag, None)
    txn.leg_start_cycle = -1
    txn.leg_completion_cycle = -1
    txn.noc_tag = ""
    txn.noc_vc = 0
    if txn.status is TransferStatus.CANCEL_REQUESTED:
      self._finish_cancelled(txn, cycle)
      return
    txn.current_leg += 1
    if txn.current_leg >= len(txn.legs):
      # Collapsed fidelity routes have no explicit source/destination legs;
      # their only completion is still the visibility boundary.
      if not self._capture_source(txn, cycle, force=True):
        return
      if not self._commit_destination(txn, cycle, force=True):
        return
      txn.status = TransferStatus.DONE
      txn.wait_reason = StageWaitReason.NONE
      txn.completed_cycle = cycle
      self._record_terminal(txn.transaction_id, TransferStatus.DONE)
      self.pmu_completed_count += 1
      self._count_transaction_bytes(txn)
      completed.append(txn)

  def _count_transaction_bytes(self, txn: MemoryTransaction) -> None:
    """Attribute one completed transaction's bytes to each traversed leg.

    Every leg of the route carries the full transaction payload, so the
    counters measure bytes crossing each interface (HBM, Global DMA, NoC,
    L2 bank ports, tile-local DMA, L1 ports) rather than unique data
    volume.  Cache-internal lookup/fill legs are not counted.
    """
    for leg in txn.legs:
      name = self._byte_counters.get(leg.kind)
      if name is None:
        continue
      self._byte_deltas[name] = self._byte_deltas.get(name, 0) + txn.bytes_total
      self.pmu_bytes_total[name] = self.pmu_bytes_total.get(name, 0) + txn.bytes_total

  def consume_byte_deltas(self) -> dict[str, int]:
    """Return and clear the per-interface byte deltas completed this tick."""
    if not self._byte_deltas:
      return {}
    deltas = dict(self._byte_deltas)
    self._byte_deltas.clear()
    return deltas

  def has_inflight_access(self, handle: AllocationHandle) -> bool:
    """Return whether an unfinished transaction references ``handle``.

    Allocation identity includes memory space, allocation id and
    generation; a recycled physical address is not the same allocation.
    This query is used only on the release path and deliberately scans
    the live transaction table instead of maintaining a per-cycle index.
    """
    for transaction in self._transactions.values():
      if transaction.status not in (
        TransferStatus.PENDING,
        TransferStatus.RUNNING,
        TransferStatus.CANCEL_REQUESTED,
        TransferStatus.FAULTED,
      ):
        continue
      src = transaction.src
      if (
        src is not None
        and src.handle.memory_space == handle.memory_space
        and src.handle.allocation_id == handle.allocation_id
        and src.handle.generation == handle.generation
      ):
        return True
      dst = transaction.dst
      if (
        dst is not None
        and dst.handle.memory_space == handle.memory_space
        and dst.handle.allocation_id == handle.allocation_id
        and dst.handle.generation == handle.generation
      ):
        return True
    return False

  def status(self, transaction_id: str) -> TransferStatus:
    txn = self._transactions.get(transaction_id)
    if txn is not None:
      return txn.status
    terminal = self._terminal_history.get(transaction_id)
    if terminal is not None:
      return terminal
    raise MemoryInvariantError("unknown or evicted transaction id")

  def wait_reason(self, transaction_id: str) -> StageWaitReason:
    txn = self._transactions.get(transaction_id)
    if txn is None:
      return StageWaitReason.NONE
    return txn.wait_reason

  def captured_data(self, transaction_id: str) -> bytes | None:
    txn = self._transactions.get(transaction_id)
    if txn is None:
      raise MemoryInvariantError("unknown transfer data request")
    if txn.status is not TransferStatus.DONE:
      raise MemoryInvariantError("transfer bytes are not yet visible")
    return txn.captured_data

  def captured_validity(self, transaction_id: str) -> bytes | None:
    transaction = self._transactions.get(transaction_id)
    return None if transaction is None else transaction.captured_validity

  def acknowledge(self, transaction_id: str, cycle: int) -> None:
    """Acknowledge a confirmed terminal transaction.

    The caller acknowledges only after its own byte/owner post-processing;
    the owner-side reference hook releases every L2 view/backing reference
    here, before the manager record is removed.  Accepted faulted or
    cancelled transactions keep their references until this same terminal
    acknowledgement.
    """
    txn = self._transactions.get(transaction_id)
    if txn is None:
      raise MemoryInvariantError("unknown transfer acknowledgement")
    if txn.status not in (TransferStatus.DONE, TransferStatus.CANCELLED, TransferStatus.FAULTED):
      raise MemoryInvariantError("cannot acknowledge a non-terminal transfer")
    if txn.reference_acquired:
      if self.reference_release is None:
        raise MemoryInvariantError(
          f"transfer {transaction_id} holds acquired references but no release hook"
        )
      self.reference_release(txn, cycle)
      txn.reference_acquired = False
    self._transactions.pop(transaction_id)

  def acknowledge_all_terminals(self, cycle: int) -> int:
    """Acknowledge every confirmed terminal transaction after owner cleanup."""
    transaction_ids = tuple(
      transaction_id
      for transaction_id, txn in self._transactions.items()
      if txn.status in (TransferStatus.DONE, TransferStatus.CANCELLED, TransferStatus.FAULTED)
    )
    for transaction_id in transaction_ids:
      self.acknowledge(transaction_id, cycle)
    return len(transaction_ids)

  def _cancel_txn_resources(self, txn: MemoryTransaction) -> None:
    """Release current-leg resources after explicit isolation confirmation."""
    if not txn.legs or txn.current_leg >= len(txn.legs):
      return
    leg = txn.legs[txn.current_leg]
    if leg.kind in (TransferLegKind.NOC_RESPONSE, TransferLegKind.NOC_REQUEST):
      if txn.noc_tag and self.noc is not None:
        tag = txn.noc_tag
        if self.noc.contains(tag):
          self.noc.cancel(tag)
        elif tag in self._noc_traversed:
          self.noc.return_credit(txn.noc_vc, 1, tag=tag)
        self._noc_traversed.pop(tag, None)
        txn.noc_tag = ""
        txn.noc_vc = 0
      return
    self._stage_for_leg(leg, txn).cancel(txn.transaction_id)

  def _request_cancel(self, txn: MemoryTransaction, cycle: int) -> None:
    if txn.status in (TransferStatus.DONE, TransferStatus.CANCELLED):
      return
    if txn.status is TransferStatus.CANCEL_REQUESTED:
      return
    txn.cancel_requested_cycle = cycle
    if not txn.legs or txn.current_leg >= len(txn.legs):
      self._finish_cancelled(txn, cycle)
      return
    leg = txn.legs[txn.current_leg]
    # Work not yet accepted by a stage can be withdrawn synchronously.  A NoC
    # flit still in the upstream queue has not consumed downstream credit and
    # is likewise unissued.
    if leg.kind in (TransferLegKind.NOC_RESPONSE, TransferLegKind.NOC_REQUEST):
      if not txn.noc_tag:
        self._finish_cancelled(txn, cycle)
        return
      if self.noc is not None and self.noc.contains(txn.noc_tag):
        self.noc.cancel(txn.noc_tag)
        self._noc_traversed.pop(txn.noc_tag, None)
        txn.noc_tag = ""
        txn.noc_vc = 0
        self._finish_cancelled(txn, cycle)
        return
    elif txn.leg_start_cycle < 0:
      self._finish_cancelled(txn, cycle)
      return
    txn.status = TransferStatus.CANCEL_REQUESTED
    txn.wait_reason = StageWaitReason.NONE
    self._cancel_requested.add(txn.transaction_id)

  def cancel_owner(self, owner: MemoryOwner, cycle: int) -> bool:
    for txn in tuple(self._transactions.values()):
      if txn.issuer == owner:
        self._request_cancel(txn, cycle)
    if self.trace is not None:
      self._trace_hbm_outstanding(cycle)
    return all(
      txn.issuer != owner or txn.status is not TransferStatus.CANCEL_REQUESTED
      for txn in self._transactions.values()
    )

  def cancel_all(self, cycle: int) -> bool:
    """Request cancellation without erasing accepted transactions.

    The return value is true only after every accepted leg has completed or an
    explicit isolation confirmation released it.
    """
    for txn in tuple(self._transactions.values()):
      self._request_cancel(txn, cycle)
    if self.trace is not None:
      self._trace_hbm_outstanding(cycle)
    return not self._cancel_requested

  def confirm_isolation(self, transaction_id: str, cycle: int) -> None:
    txn = self._transactions.get(transaction_id)
    if txn is None:
      raise MemoryInvariantError("unknown transfer isolation confirmation")
    if txn.status is not TransferStatus.CANCEL_REQUESTED:
      raise MemoryInvariantError("isolation confirmation requires CANCEL_REQUESTED")
    self._cancel_txn_resources(txn)
    txn.isolation_confirmed_cycle = cycle
    self._finish_cancelled(txn, cycle)

  @property
  def cancellation_pending(self) -> bool:
    return bool(self._cancel_requested)

  def _trace_hbm_outstanding(self, cycle: int) -> None:
    """Push the shared HBM CAM pool occupancy + credits to the sink."""
    if self.trace is None:
      return
    self.trace.hbm_outstanding(len(self._hbm_outstanding_txns), self.cfg.hbm_outstanding_limit, cycle)

  @staticmethod
  def _transaction_levels(txn: MemoryTransaction) -> frozenset[str]:
    levels: set[str] = set()
    for view in (txn.src, txn.dst):
      if view is not None and view.handle.memory_space in ("l1", "l2", "hbm"):
        levels.add(view.handle.memory_space)
    for leg in txn.legs:
      for space in (leg.src_space, leg.dst_space):
        if space.startswith("l1"):
          levels.add("l1")
        elif space.startswith("l2"):
          levels.add("l2")
        elif space == "hbm":
          levels.add("hbm")
    if txn.conservative_source_ranges:
      levels.add("hbm")
    return frozenset(levels)

  def closure_snapshot(self, levels: tuple[str, ...] | None = None) -> dict[str, object]:
    selected_levels = None if levels is None else frozenset(levels)
    live_states = {
      TransferStatus.PENDING,
      TransferStatus.RUNNING,
      TransferStatus.CANCEL_REQUESTED,
      TransferStatus.FAULTED,
    }
    selected = tuple(
      txn
      for txn in self._transactions.values()
      if txn.status in live_states
      and (selected_levels is None or bool(self._transaction_levels(txn) & selected_levels))
    )
    selected_ids = {txn.transaction_id for txn in selected}
    stage_holders = tuple(
      (stage.name, index, holder)
      for stage in self._all_stages()
      for index, holder in enumerate(stage._holders)
      if holder is not None and holder in selected_ids
    )
    return {
      "transactions": tuple(
        {
          "id": txn.transaction_id,
          "status": txn.status.value,
          "op": txn.op.value,
          "levels": tuple(sorted(self._transaction_levels(txn))),
          "leg": txn.current_leg,
          "noc_tag": txn.noc_tag,
          "fault_reason": txn.fault_reason,
          "conservative_source_ranges": txn.conservative_source_ranges,
        }
        for txn in selected
      ),
      "stage_holders": stage_holders,
      "hbm_outstanding": tuple(sorted(self._hbm_outstanding_txns & selected_ids)),
      "noc_queued": tuple(
        txn.noc_tag
        for txn in selected
        if txn.noc_tag and self.noc is not None and self.noc.contains(txn.noc_tag)
      ),
      "noc_traversed": tuple(
        txn.noc_tag for txn in selected if txn.noc_tag and txn.noc_tag in self._noc_traversed
      ),
      "quiescent": not selected and not stage_holders,
    }

  def range_inflight(self, ranges) -> bool:
    """Check HBM overlap using allocation identity, generation and offsets."""
    for txn in self._transactions.values():
      if txn.status not in (
        TransferStatus.PENDING,
        TransferStatus.RUNNING,
        TransferStatus.CANCEL_REQUESTED,
        TransferStatus.FAULTED,
      ):
        continue
      for view in (txn.src, txn.dst):
        if view is None or view.handle.memory_space != "hbm":
          continue
        view_start = view.offset_bytes
        view_end = view_start + view.size_bytes
        for item in ranges:
          if (
            item.allocation_id == view.handle.allocation_id
            and item.allocation_generation == view.handle.generation
            and item.offset < view_end
            and view_start < item.end
          ):
            return True
      for allocation_id, generation, offset, size in txn.conservative_source_ranges:
        end = offset + size
        for item in ranges:
          if (
            item.allocation_id == allocation_id
            and item.allocation_generation == generation
            and item.offset < end
            and offset < item.end
          ):
            return True
    return False

  def reset(self) -> None:
    if self._transactions:
      raise MemoryInvariantError("transfer reset requires all completions/cancellations to be acknowledged")
    if self._hbm_outstanding_txns or self._noc_traversed:
      raise MemoryInvariantError("transfer reset requires confirmed fabric drain")
    if any(any(holder is not None for holder in stage._holders) for stage in self._all_stages()):
      raise MemoryInvariantError("transfer reset requires all stage holders to drain")
    if self.noc is not None and not self.noc.is_quiescent:
      raise MemoryInvariantError("transfer reset requires NoC quiescence")
    self._completed.clear()
    self._cancelled.clear()
    self._faulted.clear()
    self._terminal_history.clear()
    self._cancel_requested.clear()
    for stage in self._all_stages():
      stage.reset()
    self.pmu_issued_count = 0
    self.pmu_completed_count = 0
    self.pmu_cancelled_count = 0
    self.pmu_faulted_count = 0
    self.pmu_noc_credit_wait_cycles = 0
    self.pmu_hbm_outstanding_peak = 0
    self.pmu_hbm_outstanding_peak_max = 0
    self._byte_deltas.clear()
    self.pmu_bytes_total.clear()
    self._issued_by_op.clear()

  @property
  def inflight_count(self) -> int:
    return sum(
      txn.status
      in (
        TransferStatus.PENDING,
        TransferStatus.RUNNING,
        TransferStatus.CANCEL_REQUESTED,
        TransferStatus.FAULTED,
      )
      for txn in self._transactions.values()
    )

  def snapshot(self) -> dict[str, object]:
    transactions = tuple(self._transactions.values())
    states = {state: sum(txn.status is state for txn in transactions) for state in TransferStatus}
    stages = {
      "hbm_read": self._hbm_read.snapshot(),
      "hbm_write": self._hbm_write.snapshot(),
      "global_dma": self._global_dma.snapshot(),
      "l2_read": self._l2_read.snapshot(),
      "l2_write": self._l2_write.snapshot(),
      "l2_cache_lookup": self._l2_cache_lookup.snapshot(),
      "l2_cache_fill": self._l2_cache_fill.snapshot(),
      **{
        stage.name: stage.snapshot()
        for stage in (
          *self._local_dma.values(),
          *self._l1_read.values(),
          *self._l1_write.values(),
          *self._l1_cache_lookup.values(),
          *self._l1_cache_fill.values(),
        )
      },
    }
    return {
      "inflight": self.inflight_count,
      "retained_transactions": len(transactions),
      "pending": states[TransferStatus.PENDING],
      "running": states[TransferStatus.RUNNING],
      "cancel_requested": states[TransferStatus.CANCEL_REQUESTED],
      "completed": states[TransferStatus.DONE],
      "cancelled": states[TransferStatus.CANCELLED],
      "faulted": states[TransferStatus.FAULTED],
      "tombstone_capacity": self.tombstone_capacity,
      "can_accept_new": self.can_accept_new,
      "issued": self.pmu_issued_count,
      "completed_total": self.pmu_completed_count,
      "cancelled_total": self.pmu_cancelled_count,
      "faulted_total": self.pmu_faulted_count,
      "noc_credit_wait_cycles": self.pmu_noc_credit_wait_cycles,
      "byte_counters": dict(sorted(self.pmu_bytes_total.items())),
      "hbm_outstanding": len(self._hbm_outstanding_txns),
      "hbm_outstanding_peak": self.pmu_hbm_outstanding_peak_max,
      "issued_by_op": dict(self._issued_by_op),
      "stages": stages,
      "transactions": tuple(
        {
          "id": txn.transaction_id,
          "op": txn.op.value,
          "status": txn.status.value,
          "leg": txn.current_leg,
          "total_legs": len(txn.legs),
          "wait_reason": txn.wait_reason.value,
          "source_captured_cycle": txn.source_captured_cycle,
          "destination_committed_cycle": txn.destination_committed_cycle,
          "cancel_requested_cycle": txn.cancel_requested_cycle,
          "isolation_confirmed_cycle": txn.isolation_confirmed_cycle,
          "fault_reason": txn.fault_reason,
        }
        for txn in transactions
      ),
    }
