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

from dataclasses import dataclass
from enum import Enum

from .allocator import (
  AllocationHandle,
  BankSegment,
  MemoryInvariantError,
  MemoryOwner,
)
from .noc import Flit, VCId

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
  view: ResolvedMemoryView | None,
  offset_bytes: int,
  size_bytes: int,
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
        BankSegment(
          segment.bank_id,
          segment.address + physical_offset,
          overlap_end - overlap_start,
        )
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
  GATHER_L1_HIT = "gather_l1_hit"
  GATHER_L2_HIT = "gather_l2_hit"
  GATHER_MISS_LOOKUP = "gather_miss_lookup"
  GATHER_HBM_REFILL = "gather_hbm_refill"
  GATHER_L2_REFILL = "gather_l2_refill"
  GATHER_DEST_WRITE = "gather_dest_write"


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


class TransferStatus(Enum):
  PENDING = "pending"
  RUNNING = "running"
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

# ---------------------------------------------------------------------------
# Transfer stage — one resource with channels/bandwidth/latency
# ---------------------------------------------------------------------------


class TransferStage:
  """One resource stage (HBM channel, DMA channel, NoC VC, L2/L1 bank).

  ``try_issue`` atomically checks that all requested channels/banks are
  available in the current cycle, then occupies them.  If any is busy it
  returns ``StageWait`` with zero side effects.  It does not book future
  channels; the manager retries pending transactions each cycle.

  Resource ownership is tracked per transaction (``_holders``), so
  ``cancel(transaction_id)`` returns every resource and outstanding
  credit held by a cancelled transaction immediately, and ``step``
  reconciles resources whose busy window expired.
  """

  def __init__(self, name: str, wait_reason: StageWaitReason,
               fixed_latency_cycles: int, bytes_per_cycle: float,
               resource_count: int, burst_bytes: int = 1,
               max_outstanding: int | None = None,
               shared_outstanding: set[str] | None = None):
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
    self._outstanding_txns: set[str] = (
      shared_outstanding if shared_outstanding is not None else set())
    self.wait_cycles: int = 0  # cumulative StageWait cycles (PMU delta)

  @property
  def _outstanding(self) -> int:
    return len(self._outstanding_txns)

  def try_issue(self, transaction_id: str,
                requests: list[StageRequest],
                cycle: int) -> StageResult | StageWait:
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
        burst_rounded = ((req.bytes_total + self.burst_bytes - 1)
                         // self.burst_bytes) * self.burst_bytes
        if self.bytes_per_cycle > 0:
          xfer = int(max((burst_rounded + self.bytes_per_cycle - 1)
                         // self.bytes_per_cycle, 1))
        else:
          xfer = 0
        comp = cycle + self.fixed_latency_cycles + xfer
        self._busy_until[bid] = comp
        self._holders[bid] = transaction_id
        max_completion = max(max_completion, comp)
      if self.max_outstanding is not None:
        self._outstanding_txns.add(transaction_id)
      return StageResult(accepted_cycle=cycle,
                         completion_cycle=max_completion,
                         channels=len(bank_ids),
                         resources=tuple(bank_ids))
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
    return StageResult(accepted_cycle=cycle, completion_cycle=completion,
                       channels=1, resources=(free_idx,))

  def release_outstanding(self, transaction_id: str) -> None:
    """Return one outstanding credit and free the transaction's resources.

    Idempotent: safe to call after a leg completes or after cancel.
    """
    self._outstanding_txns.discard(transaction_id)
    for i, holder in enumerate(self._holders):
      if holder == transaction_id:
        self._busy_until[i] = 0
        self._holders[i] = None

  def cancel(self, transaction_id: str) -> None:
    """Free every resource and outstanding credit held by a cancelled
    transaction (idempotent)."""
    self._outstanding_txns.discard(transaction_id)
    for i, holder in enumerate(self._holders):
      if holder == transaction_id:
        self._busy_until[i] = 0
        self._holders[i] = None

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
  def __init__(self, cfg, full_memory: bool = False, noc=None, trace=None):
    self.cfg = cfg
    self.full_memory = full_memory
    self.noc = noc  # NoCRouter (full_memory); None when fabric unmodeled
    self.trace = trace  # MemoryTrace sink; None disables event emission
    clock_hz = cfg.clock_mhz * 1e6
    self._transactions: dict[str, MemoryTransaction] = {}
    self._completed: set[str] = set()
    self._cancelled: set[str] = set()
    self._faulted: set[str] = set()
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
    self._issued_by_op: dict[str, int] = {}
    # One global outstanding CAM/credit pool shared by HBM reads+writes.
    self._hbm_outstanding_txns: set[str] = set()
    self._hbm_read = TransferStage(
      "hbm_read", StageWaitReason.HBM_OUTSTANDING,
      cfg.hbm_fixed_latency_cycles,
      (cfg.hbm_bandwidth_gbs / cfg.hbm_channels) * 1e9 / clock_hz,
      cfg.hbm_channels, cfg.hbm_burst_bytes,
      max_outstanding=cfg.hbm_outstanding_limit,
      shared_outstanding=self._hbm_outstanding_txns)
    self._hbm_write = TransferStage(
      "hbm_write", StageWaitReason.HBM_OUTSTANDING,
      cfg.hbm_fixed_latency_cycles,
      (cfg.hbm_bandwidth_gbs / cfg.hbm_channels) * 1e9 / clock_hz,
      cfg.hbm_channels, cfg.hbm_burst_bytes,
      max_outstanding=cfg.hbm_outstanding_limit,
      shared_outstanding=self._hbm_outstanding_txns)
    self._global_dma = TransferStage(
      "global_dma", StageWaitReason.DMA_QUEUE,
      (cfg.dma_launch_cycles + cfg.dma_desc_cycles + cfg.dma_issue_cycles
       + cfg.dma_completion_cycles),
      cfg.group_dma_bandwidth_gbs * 1e9 / clock_hz,
      cfg.num_dma_channels, 1)
    # L2 bank stages — per-bank bandwidth; segments issue to specific banks
    l2_bw = cfg.l2_bank_bandwidth_gbs * 1e9 / clock_hz
    self._l2_read = TransferStage(
      "l2_read", StageWaitReason.L2_BANK,
      cfg.l2_access_latency_cycles, l2_bw, cfg.group_sram_banks, 1)
    self._l2_write = TransferStage(
      "l2_write", StageWaitReason.L2_BANK,
      cfg.l2_access_latency_cycles, l2_bw, cfg.group_sram_banks, 1)
    # local DMA + L1 (per-tile, created on demand)
    self._local_dma: dict[tuple[int, str], TransferStage] = {}
    self._l1_read: dict[int, TransferStage] = {}
    self._l1_write: dict[int, TransferStage] = {}
    self._l2_cache_lookup = TransferStage(
      "l2_cache_lookup",
      StageWaitReason.L2_CACHE,
      cfg.l2_cache_lookup_latency_cycles,
      l2_bw,
      1,
      1,
    )
    self._l2_cache_fill = TransferStage(
      "l2_cache_fill",
      StageWaitReason.L2_CACHE,
      cfg.l2_cache_lookup_latency_cycles,
      l2_bw,
      1,
      1,
    )
    self._l1_cache_lookup: dict[int, TransferStage] = {}
    self._l1_cache_fill: dict[int, TransferStage] = {}

  def _local_dma_stage(self, tile_id: int, direction: str) -> TransferStage:
    key = (tile_id, direction)
    if key not in self._local_dma:
      clock_hz = self.cfg.clock_mhz * 1e6
      count = (self.cfg.mfe_load_channels if direction == "load"
               else self.cfg.mfe_store_channels)
      self._local_dma[key] = TransferStage(
        f"local_dma_t{tile_id}_{direction}", StageWaitReason.DMA_QUEUE,
        self.cfg.mfe_launch_cycles,
        self.cfg.mfe_bandwidth_gbs * 1e9 / clock_hz, count, 1)
    return self._local_dma[key]

  def _l1_stage(self, tile_id: int, is_write: bool) -> TransferStage:
    cache = self._l1_write if is_write else self._l1_read
    if tile_id not in cache:
      clock_hz = self.cfg.clock_mhz * 1e6
      bw = (self.cfg.tile_l1_bandwidth_gbs / self.cfg.tile_l1_banks) * 1e9 / clock_hz
      cache[tile_id] = TransferStage(
        f"l1_{'write' if is_write else 'read'}_t{tile_id}",
        StageWaitReason.L1_BANK,
        self.cfg.l1_access_latency_cycles, bw,
        self.cfg.tile_l1_banks, 1)
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

  def submit(self, transaction: MemoryTransaction, cycle: int,
             pmu=None) -> None:
    """Submit a transaction.  Builds the route based on op + fidelity."""
    if transaction.transaction_id in self._transactions:
      raise MemoryInvariantError("duplicate transaction id")
    self._transactions[transaction.transaction_id] = transaction
    self.pmu_issued_count += 1
    op_name = transaction.op.value
    self._issued_by_op[op_name] = self._issued_by_op.get(op_name, 0) + 1
    # Gather always retains its explicit lookup/refill/write route in every
    # fidelity mode.  Opaque profile tokens never collapse into a fake DMA.
    gather_ops = (
      TransferOp.GATHER_L1_HIT,
      TransferOp.GATHER_L2_HIT,
      TransferOp.GATHER_MISS_LOOKUP,
      TransferOp.GATHER_HBM_REFILL,
      TransferOp.GATHER_L2_REFILL,
      TransferOp.GATHER_DEST_WRITE,
    )
    if transaction.op in gather_ops:
      transaction.legs = self._build_route(transaction)
    elif transaction.src is None and transaction.dst is None:
      transaction.legs = self._collapsed_leg(transaction)
    elif transaction.src is not None and transaction.dst is not None:
      if self.full_memory:
        transaction.legs = self._build_route(transaction)
      else:
        transaction.legs = self._collapsed_leg(transaction)
    else:
      transaction.status = TransferStatus.FAULTED
      self._faulted.add(transaction.transaction_id)
      self.pmu_faulted_count += 1
      return
    transaction.status = TransferStatus.RUNNING
    transaction.current_leg = 0
    transaction.leg_start_cycle = -1

  def _collapsed_leg(self, txn: MemoryTransaction) -> tuple[TransferLeg, ...]:
    """One collapsed leg (existing bandwidth + launch overhead) per route.

    Group ops fold onto the Global DMA stage; tile-local ops fold onto
    the tile's local DMA stage (mfe_launch_cycles + mfe bandwidth).
    """
    if txn.op in (
      TransferOp.GATHER_L1_HIT,
      TransferOp.GATHER_L2_HIT,
      TransferOp.GATHER_MISS_LOOKUP,
      TransferOp.GATHER_HBM_REFILL,
      TransferOp.GATHER_L2_REFILL,
      TransferOp.GATHER_DEST_WRITE,
    ):
      raise MemoryInvariantError("gather route must not be collapsed")
    if txn.op in (TransferOp.TILE_LOAD, TransferOp.TILE_STORE):
      tid = txn.tile_id or 0
      direction = "load" if txn.op == TransferOp.TILE_LOAD else "store"
      return (TransferLeg(
        TransferLegKind.LOCAL_DMA, "l2", "l1",
        txn.bytes_total,
        f"local_dma:{tid}:{direction}:{txn.transaction_id}"),)
    return (TransferLeg(
      TransferLegKind.GLOBAL_DMA, "hbm", "l2",
      txn.bytes_total, f"gdma:{txn.transaction_id}"),)

  def _build_route(self, txn: MemoryTransaction) -> tuple[TransferLeg, ...]:
    op = txn.op
    tid = txn.tile_id or 0
    if op == TransferOp.GATHER_L1_HIT:
      return (
        TransferLeg(
          TransferLegKind.L1_CACHE_LOOKUP,
          "l1_cache",
          "l1_cache",
          txn.bytes_total,
          f"l1_cache_lookup:{tid}:{txn.transaction_id}",
        ),
      )
    if op == TransferOp.GATHER_L2_HIT:
      return (
        TransferLeg(
          TransferLegKind.L1_CACHE_LOOKUP,
          "l1_cache",
          "l1_cache",
          txn.bytes_total,
          f"l1_cache_lookup:{tid}:{txn.transaction_id}",
        ),
        TransferLeg(
          TransferLegKind.L2_CACHE_LOOKUP,
          "l2_cache",
          "l2_cache",
          txn.bytes_total,
          f"l2_cache_lookup:{txn.transaction_id}",
        ),
        TransferLeg(
          TransferLegKind.NOC_RESPONSE,
          "noc",
          "tile",
          txn.bytes_total,
          f"noc_rsp:{txn.transaction_id}",
        ),
        TransferLeg(
          TransferLegKind.LOCAL_DMA,
          "tile",
          "l1_cache",
          txn.bytes_total,
          f"local_dma:{tid}:load:{txn.transaction_id}",
        ),
        TransferLeg(
          TransferLegKind.L1_CACHE_FILL,
          "l1_cache",
          "l1_cache",
          txn.bytes_total,
          f"l1_cache_fill:{tid}:{txn.transaction_id}",
        ),
      )
    if op == TransferOp.GATHER_MISS_LOOKUP:
      return (
        TransferLeg(
          TransferLegKind.L1_CACHE_LOOKUP,
          "l1_cache",
          "l1_cache",
          txn.bytes_total,
          f"l1_cache_lookup:{tid}:{txn.transaction_id}",
        ),
        TransferLeg(
          TransferLegKind.L2_CACHE_LOOKUP,
          "l2_cache",
          "l2_cache",
          txn.bytes_total,
          f"l2_cache_lookup:{txn.transaction_id}",
        ),
      )
    if op == TransferOp.GATHER_HBM_REFILL:
      return (
        TransferLeg(
          TransferLegKind.HBM_READ,
          "hbm",
          "noc",
          txn.bytes_total,
          f"hbm_read:{txn.transaction_id}",
        ),
        TransferLeg(
          TransferLegKind.NOC_RESPONSE,
          "noc",
          "l2_cache",
          txn.bytes_total,
          f"noc_rsp:{txn.transaction_id}",
        ),
        TransferLeg(
          TransferLegKind.L2_CACHE_FILL,
          "l2_cache",
          "l2_cache",
          txn.bytes_total,
          f"l2_cache_fill:{txn.transaction_id}",
        ),
      )
    if op == TransferOp.GATHER_L2_REFILL:
      return (
        TransferLeg(
          TransferLegKind.NOC_RESPONSE,
          "l2_cache",
          "tile",
          txn.bytes_total,
          f"noc_rsp:{txn.transaction_id}",
        ),
        TransferLeg(
          TransferLegKind.LOCAL_DMA,
          "tile",
          "l1_cache",
          txn.bytes_total,
          f"local_dma:{tid}:load:{txn.transaction_id}",
        ),
        TransferLeg(
          TransferLegKind.L1_CACHE_FILL,
          "l1_cache",
          "l1_cache",
          txn.bytes_total,
          f"l1_cache_fill:{tid}:{txn.transaction_id}",
        ),
      )
    if op == TransferOp.GATHER_DEST_WRITE:
      return (
        TransferLeg(
          TransferLegKind.L1_WRITE,
          "l1",
          "l1",
          txn.bytes_total,
          f"l1_write:{tid}:{txn.transaction_id}",
        ),
      )
    if op == TransferOp.PREFETCH:
      return (
        TransferLeg(TransferLegKind.HBM_READ, "hbm", "noc",
                    txn.bytes_total, f"hbm_read:{txn.transaction_id}"),
        TransferLeg(TransferLegKind.GLOBAL_DMA, "noc", "noc",
                    txn.bytes_total, f"gdma:{txn.transaction_id}"),
        TransferLeg(TransferLegKind.NOC_RESPONSE, "noc", "l2",
                    txn.bytes_total, f"noc_rsp:{txn.transaction_id}"),
        TransferLeg(TransferLegKind.L2_WRITE, "noc", "l2",
                    txn.bytes_total, f"l2_write:{txn.transaction_id}"),
      )
    if op == TransferOp.GLOBAL_STORE:
      return (
        TransferLeg(TransferLegKind.L2_READ, "l2", "noc",
                    txn.bytes_total, f"l2_read:{txn.transaction_id}"),
        TransferLeg(TransferLegKind.NOC_REQUEST, "noc", "noc",
                    txn.bytes_total, f"noc_req:{txn.transaction_id}"),
        TransferLeg(TransferLegKind.GLOBAL_DMA, "noc", "noc",
                    txn.bytes_total, f"gdma:{txn.transaction_id}"),
        TransferLeg(TransferLegKind.HBM_WRITE, "noc", "hbm",
                    txn.bytes_total, f"hbm_write:{txn.transaction_id}"),
      )
    if op == TransferOp.TILE_LOAD:
      tid = txn.tile_id or 0
      return (
        TransferLeg(TransferLegKind.L2_READ, "l2", "tile",
                    txn.bytes_total, f"l2_read:{txn.transaction_id}"),
        TransferLeg(TransferLegKind.LOCAL_DMA, "l2", "tile",
                    txn.bytes_total, f"local_dma:{tid}:load:{txn.transaction_id}"),
        TransferLeg(TransferLegKind.L1_WRITE, "tile", "l1",
                    txn.bytes_total, f"l1_write:{tid}:{txn.transaction_id}"),
      )
    if op == TransferOp.TILE_STORE:
      tid = txn.tile_id or 0
      return (
        TransferLeg(TransferLegKind.L1_READ, "l1", "tile",
                    txn.bytes_total, f"l1_read:{tid}:{txn.transaction_id}"),
        TransferLeg(TransferLegKind.LOCAL_DMA, "tile", "l2",
                    txn.bytes_total, f"local_dma:{tid}:store:{txn.transaction_id}"),
        TransferLeg(TransferLegKind.L2_WRITE, "tile", "l2",
                    txn.bytes_total, f"l2_write:{txn.transaction_id}"),
      )
    return ()

  def _stage_for_leg(self, leg: TransferLeg,
                     txn: MemoryTransaction) -> TransferStage:
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
      direction = "store" if txn.op == TransferOp.TILE_STORE else "load"
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
  def _leg_view(leg: TransferLeg,
                txn: MemoryTransaction) -> ResolvedMemoryView | None:
    """Return the resolved view whose segments a bank-based leg accesses."""
    kind = leg.kind
    if kind in (TransferLegKind.L2_READ, TransferLegKind.L1_READ):
      return txn.src
    if kind in (TransferLegKind.L2_WRITE, TransferLegKind.L1_WRITE):
      return txn.dst
    return None

  def _requests_for_leg(self, leg: TransferLeg,
                        txn: MemoryTransaction) -> list[StageRequest]:
    """Build deterministic resource requests for one stage-backed leg.

    HBM is channel-addressed, not first-free:
    ``(global_address // hbm_burst_bytes) % hbm_channels``.
    L2/L1 requests use allocator-resolved bank segments.
    """
    if leg.kind in (TransferLegKind.HBM_READ, TransferLegKind.HBM_WRITE):
      view = txn.src if leg.kind == TransferLegKind.HBM_READ else txn.dst
      if view is not None:
        channel = ((view.address // self.cfg.hbm_burst_bytes)
                   % self.cfg.hbm_channels)
        return [StageRequest(str(channel), leg.bytes_total)]
    view = self._leg_view(leg, txn)
    if view is not None and view.segments:
      return [StageRequest(str(seg.bank_id), seg.size_bytes)
              for seg in view.segments]
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
    """Record flits that left the router this cycle (called by TileGroup
    right after ``NoCRouter.step``)."""
    for flit in flits:
      self._noc_traversed[flit.tag] = cycle

  def step(self, cycle: int) -> tuple[MemoryTransaction, ...]:
    """Advance all transactions one cycle.  Returns newly completed ones."""
    # Reconcile every stage first: expire finished busy windows and return
    # outstanding credits whose holder lapsed without an explicit release.
    for stage in self._all_stages():
      stage.step(cycle)
    completed: list[MemoryTransaction] = []
    for txn in list(self._transactions.values()):
      if txn.status in (TransferStatus.DONE, TransferStatus.FAULTED,
                        TransferStatus.CANCELLED):
        continue
      if not txn.legs:
        continue
      if txn.current_leg >= len(txn.legs):
        continue
      leg = txn.legs[txn.current_leg]
      if leg.kind in (TransferLegKind.NOC_RESPONSE,
                      TransferLegKind.NOC_REQUEST):
        self._step_noc_leg(txn, leg, cycle, completed)
        continue
      stage = self._stage_for_leg(leg, txn)
      # if leg not yet issued, try to issue
      if txn.leg_start_cycle < 0:
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
          self.trace.transfer_leg_issued(txn, leg, stage.name, result,
                                         result.resources, cycle)
          if stage.name in ("hbm_read", "hbm_write"):
            self._trace_hbm_outstanding(cycle)
      # check if current leg completed
      if cycle >= txn.leg_completion_cycle and txn.leg_completion_cycle > 0:
        # advance to next leg; return this leg's resources and any
        # outstanding credit held by this transaction in the stage
        stage.release_outstanding(txn.transaction_id)
        if self.trace is not None:
          self.trace.transfer_leg_completed(txn, cycle)
          if stage.name in ("hbm_read", "hbm_write"):
            self._trace_hbm_outstanding(cycle)
        self._advance_leg(txn, cycle, completed)
    # track HBM outstanding peak after this cycle's issue activity
    peak = len(self._hbm_outstanding_txns)
    if peak > self.pmu_hbm_outstanding_peak:
      self.pmu_hbm_outstanding_peak = peak
    return tuple(completed)

  def _step_noc_leg(self, txn: MemoryTransaction, leg: TransferLeg,
                    cycle: int, completed: list[MemoryTransaction]) -> None:
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
    vc = (VCId.VC1_DMA_READ_RSP.value
          if leg.kind == TransferLegKind.NOC_RESPONSE
          else VCId.VC2_DMA_WRITE.value)
    if not txn.noc_tag:
      # first entry: enqueue one flit/tag
      txn.noc_tag = f"{txn.transaction_id}:{leg.kind.value}"
      txn.noc_vc = vc
      txn.leg_start_cycle = cycle
      if txn.start_cycle < 0:
        txn.start_cycle = cycle
      self.noc.send(vc, Flit(vc=vc, src=0, dst=1,
                             bytes_total=leg.bytes_total,
                             tag=txn.noc_tag), cycle)
      txn.wait_reason = StageWaitReason.NOC_CREDIT
      self.pmu_noc_credit_wait_cycles += 1
      if self.trace is not None:
        # router-backed leg: completion cycle unknown at issue (-1); the
        # completion hook substitutes the actual cycle.
        self.trace.transfer_leg_issued(
          txn, leg, f"noc_vc{vc}", StageResult(cycle, -1), (), cycle)
      return
    traversed = self._noc_traversed.get(txn.noc_tag)
    if traversed is None:
      # still pending: queueing, arbitration or credit exhaustion
      txn.wait_reason = StageWaitReason.NOC_CREDIT
      self.pmu_noc_credit_wait_cycles += 1
      if self.trace is not None:
        self.trace.transfer_wait(txn.transaction_id,
                                 StageWaitReason.NOC_CREDIT.value)
      return
    if cycle >= traversed + self.cfg.noc_router_latency_cycles:
      self.noc.return_credit(txn.noc_vc, 1)
      self._noc_traversed.pop(txn.noc_tag, None)
      if self.trace is not None:
        self.trace.transfer_leg_completed(txn, cycle)
      self._advance_leg(txn, cycle, completed)
    else:
      txn.wait_reason = StageWaitReason.NOC_CREDIT
      self.pmu_noc_credit_wait_cycles += 1

  def _advance_leg(self, txn: MemoryTransaction, cycle: int,
                   completed: list[MemoryTransaction]) -> None:
    """Advance to the next leg; complete the transaction after the last."""
    if txn.noc_tag:
      self._noc_traversed.pop(txn.noc_tag, None)
    txn.current_leg += 1
    txn.leg_start_cycle = -1
    txn.leg_completion_cycle = -1
    txn.noc_tag = ""
    txn.noc_vc = 0
    if txn.current_leg >= len(txn.legs):
      txn.status = TransferStatus.DONE
      txn.wait_reason = StageWaitReason.NONE
      txn.completed_cycle = cycle
      self._completed.add(txn.transaction_id)
      self.pmu_completed_count += 1
      completed.append(txn)

  def status(self, transaction_id: str) -> TransferStatus:
    txn = self._transactions.get(transaction_id)
    if txn is None:
      if transaction_id in self._cancelled:
        return TransferStatus.CANCELLED
      return TransferStatus.PENDING
    return txn.status

  def wait_reason(self, transaction_id: str) -> StageWaitReason:
    txn = self._transactions.get(transaction_id)
    if txn is None:
      return StageWaitReason.NONE
    return txn.wait_reason

  def acknowledge(self, transaction_id: str) -> None:
    self._transactions.pop(transaction_id, None)
    self._completed.discard(transaction_id)

  def _cancel_txn_resources(self, txn: MemoryTransaction) -> None:
    """Return every resource a transaction holds on its current leg.

    Stage-backed legs release banks/channels/credits via
    ``stage.cancel``.  Router-backed NoC legs cancel the pending flit
    (no credit consumed yet) or return the downstream credit if the
    flit already traversed.
    """
    if not txn.legs or txn.current_leg >= len(txn.legs):
      return
    leg = txn.legs[txn.current_leg]
    if leg.kind in (TransferLegKind.NOC_RESPONSE, TransferLegKind.NOC_REQUEST):
      if txn.noc_tag and self.noc is not None:
        if self.noc.contains(txn.noc_tag):
          self.noc.cancel(txn.noc_tag)
        elif txn.noc_tag in self._noc_traversed:
          self.noc.return_credit(txn.noc_vc, 1)
        self._noc_traversed.pop(txn.noc_tag, None)
        txn.noc_tag = ""
        txn.noc_vc = 0
      return
    self._stage_for_leg(leg, txn).cancel(txn.transaction_id)

  def cancel_owner(self, owner: MemoryOwner, cycle: int) -> None:
    for txn in list(self._transactions.values()):
      if txn.issuer != owner or txn.status in (
          TransferStatus.DONE, TransferStatus.CANCELLED):
        continue
      txn.status = TransferStatus.CANCELLED
      self._cancelled.add(txn.transaction_id)
      self.pmu_cancelled_count += 1
      self._cancel_txn_resources(txn)
      if self.trace is not None:
        self.trace.transfer_cancelled(txn, cycle)
        self._trace_hbm_outstanding(cycle)
      self._transactions.pop(txn.transaction_id, None)

  def cancel_all(self, cycle: int) -> None:
    """Cancel every live transaction and return all stage resources.

    Per-transaction cancel returns HBM outstanding credits, NoC
    credits, DMA channels and bank reservations; the stage resets then
    guarantee a clean slate even for resources whose holder reference
    was already expired.
    """
    for txn in list(self._transactions.values()):
      if txn.status in (TransferStatus.DONE, TransferStatus.CANCELLED):
        continue
      txn.status = TransferStatus.CANCELLED
      self._cancelled.add(txn.transaction_id)
      self.pmu_cancelled_count += 1
      self._cancel_txn_resources(txn)
      if self.trace is not None:
        self.trace.transfer_cancelled(txn, cycle)
    self._transactions.clear()
    for stage in self._all_stages():
      stage.reset()
    self._noc_traversed.clear()
    if self.trace is not None:
      self._trace_hbm_outstanding(cycle)

  def _trace_hbm_outstanding(self, cycle: int) -> None:
    """Push the shared HBM CAM pool occupancy + credits to the sink."""
    if self.trace is None:
      return
    self.trace.hbm_outstanding(len(self._hbm_outstanding_txns),
                               self.cfg.hbm_outstanding_limit, cycle)

  def reset(self) -> None:
    self._transactions.clear()
    self._completed.clear()
    self._cancelled.clear()
    self._faulted.clear()
    self._noc_traversed.clear()
    for stage in self._all_stages():
      stage.reset()
    self.pmu_issued_count = 0
    self.pmu_completed_count = 0
    self.pmu_cancelled_count = 0
    self.pmu_faulted_count = 0
    self.pmu_noc_credit_wait_cycles = 0
    self.pmu_hbm_outstanding_peak = 0
    self.pmu_hbm_outstanding_peak_max = 0
    self._issued_by_op.clear()

  @property
  def inflight_count(self) -> int:
    return sum(1 for t in self._transactions.values()
               if t.status == TransferStatus.RUNNING)

  def snapshot(self) -> dict:
    running = [t for t in self._transactions.values()
               if t.status == TransferStatus.RUNNING]
    pending = [t for t in self._transactions.values()
               if t.status == TransferStatus.PENDING]
    return {
      "inflight": len(self._transactions),
      "pending": len(pending),
      "running": len(running),
      "completed": len(self._completed),
      "cancelled": len(self._cancelled),
      "issued": self.pmu_issued_count,
      "completed_total": self.pmu_completed_count,
      "cancelled_total": self.pmu_cancelled_count,
      "faulted_total": self.pmu_faulted_count,
      "noc_credit_wait_cycles": self.pmu_noc_credit_wait_cycles,
      "hbm_outstanding_peak": self.pmu_hbm_outstanding_peak_max,
      "issued_by_op": dict(self._issued_by_op),
      "stages": {
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
      },
      "transactions": [
        {
          "id": t.transaction_id,
          "op": t.op.value,
          "status": t.status.value,
          "leg": t.current_leg,
          "total_legs": len(t.legs),
          "wait_reason": t.wait_reason.value,
        }
        for t in running
      ],
    }
