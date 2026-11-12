"""External memory-leg service for boundary-originated MEM_REQ traffic (C023 / W4-02 external leg).

Serves the wire MEMORY channel contract (C017 `wire.py`: `MEM_REQ` T2G, `MEM_RSP` G2T,
`TERMINAL_ACK` T2G) at the Group side:

    MEM_REQ (binding identity, range, permission, slice)
      -> C013-semantics validation against the LIVE HBM binding table
      -> real byte movement through ByteStore (reads return bytes; writes commit bytes via the
         permission-checked view path, never a host seeding shortcut)
      -> MEM_RSP carrying the write visibility scope
      -> TERMINAL_ACK per (txn, slice) releases the terminal reference; the ledger must be empty
         (`assert_closed`) before the run may claim zero leaks.

Scope boundary (C023): this service is the Python-side external leg for boundary-originated
requests. Python Tiles keep their in-process TransferManager legs byte-identical (C022 baseline
equality); excising L1 execution from TransferManager happens with the RTL bank/cache/MSHR in
C029/C030, not here.

Negatives are load-bearing:
- a request naming a released/unknown binding or a stale generation is REJECTED before any ledger
  or byte mutation (`ledger unchanged` is asserted by callers);
- a forged TERMINAL_ACK (unknown txn, duplicate ack, wrong slice accounting) raises and leaves the
  ledger unchanged;
- channel credits ride the `ChannelCreditLedger` conservation invariant; a forged return raises
  `CreditConservationError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..memory.allocator import MemoryInvariantError
from ..memory.byte_store import ByteStore
from ..memory.hbm_region import HBMRegion
from ..memory.transfer import ResolvedMemoryView
from .tile_port import ChannelCreditLedger, CreditConservationError

MEM_OP_READ = 0
MEM_OP_WRITE = 1

# MEM_RSP.visibility (write visibility scope, C013): 0 none (read), 1 L2-visible,
# 2 HBM-persisted. Encoding owned here; the wire field stays transport-only until frozen.
VIS_NONE = 0
VIS_L2 = 1
VIS_HBM = 2

STATUS_OK = 0
STATUS_FAULT = 1


class MemoryServiceError(RuntimeError):
  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


@dataclass(frozen=True)
class MemRequest:
  """Pure-value MEM_REQ payload (wire fields + the service-level data channel for writes)."""

  tile_id: int
  ctx_slot: int
  ctx_generation: int
  txn_id: int
  slice_id: int
  slice_count: int
  arena_alloc_gen: int
  binding_name: str
  binding_generation: int
  binding_offset: int
  scope_epoch: int
  mem_op: int  # MEM_OP_READ | MEM_OP_WRITE
  length: int
  payload: bytes | None = None  # write data travels on the service channel, not the wire header

  def __post_init__(self) -> None:
    for name in (
      "tile_id",
      "ctx_slot",
      "ctx_generation",
      "txn_id",
      "slice_id",
      "slice_count",
      "arena_alloc_gen",
      "binding_generation",
      "binding_offset",
      "scope_epoch",
      "mem_op",
      "length",
    ):
      val = getattr(self, name)
      if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        raise TypeError(f"MemRequest.{name} must be a non-negative int, got {val!r}")
    if not isinstance(self.binding_name, str) or not self.binding_name:
      raise TypeError("MemRequest.binding_name must be a non-empty str")
    if self.mem_op not in (MEM_OP_READ, MEM_OP_WRITE):
      raise ValueError(f"unknown mem_op {self.mem_op}")
    if self.slice_count < 1 or self.slice_id >= self.slice_count:
      raise ValueError("slice_id must index into slice_count")
    if self.length <= 0:
      raise ValueError("length must be positive")
    if self.mem_op == MEM_OP_WRITE and not isinstance(self.payload, bytes):
      raise TypeError("write MEM_REQ requires payload bytes")
    if self.mem_op == MEM_OP_WRITE and len(self.payload) != self.length:
      raise ValueError("write payload length disagrees with length field")


@dataclass(frozen=True)
class MemResponse:
  txn_id: int
  slice_id: int
  binding_generation: int
  visibility: int
  status: int
  data: bytes | None = None  # read data on the service channel


@dataclass
class _ServiceBinding:
  name: str
  base: int
  size: int
  permissions: str
  generation: int
  released: bool = False


@dataclass
class _TerminalRef:
  txn_id: int
  slice_id: int
  slice_count: int
  binding_name: str
  mem_op: int
  length: int


@dataclass
class MemoryServiceStats:
  requests_accepted: int = 0
  requests_rejected: int = 0
  bytes_read: int = 0
  bytes_written: int = 0
  terminal_acks: int = 0


class MemoryService:
  """Group-side external leg: MEM_REQ -> real bytes -> MEM_RSP -> TERMINAL_ACK."""

  def __init__(
    self, hbm: HBMRegion, byte_store: ByteStore | None, *, credits: ChannelCreditLedger | None = None
  ) -> None:
    self.hbm = hbm
    self.byte_store = byte_store
    self.credits = credits or ChannelCreditLedger()
    self._bindings: dict[str, _ServiceBinding] = {}
    self._terminal_refs: dict[tuple[int, int], _TerminalRef] = {}
    self._txn_slices_seen: dict[int, set[int]] = {}
    self.stats = MemoryServiceStats()
    self.visibility_scope_epoch = 0

  # -- binding lifecycle -------------------------------------------------
  def register_binding(
    self, name: str, base: int, size: int, permissions: str, *, generation: int = 1
  ) -> None:
    """Register one live HBM binding; must match the HBM region's own binding table."""
    if name in self._bindings and not self._bindings[name].released:
      raise MemoryServiceError("BINDING_ALREADY_LIVE", name)
    if self.hbm.get_handle(name) is None:
      raise MemoryServiceError("BINDING_NOT_IN_HBM", name)
    if self.hbm.permissions(name) != permissions:
      raise MemoryServiceError("BINDING_PERMISSION_MISMATCH", name)
    handle = self.hbm.get_handle(name)
    assert handle is not None
    if handle.owner.binding_name != name:
      raise MemoryServiceError("BINDING_OWNER_MISMATCH", name)
    self._bindings[name] = _ServiceBinding(name, base, size, permissions, generation)

  def release_binding(self, name: str, generation: int) -> None:
    record = self._bindings.get(name)
    if record is None:
      raise MemoryServiceError("UNKNOWN_BINDING", name)
    if record.released:
      raise MemoryServiceError("BINDING_ALREADY_RELEASED", name)
    if record.generation != generation:
      raise MemoryServiceError("BINDING_GENERATION_MISMATCH", f"{name}:{generation}")
    record.released = True

  def _live_binding(self, req: MemRequest) -> _ServiceBinding:
    record = self._bindings.get(req.binding_name)
    if record is None or record.released:
      self.stats.requests_rejected += 1
      raise MemoryServiceError("MEM_REQ_REJECTED", f"binding {req.binding_name!r} not live")
    if record.generation != req.binding_generation:
      self.stats.requests_rejected += 1
      raise MemoryServiceError("MEM_REQ_REJECTED", f"stale binding generation {req.binding_generation}")
    return record

  # -- request path ------------------------------------------------------
  def submit_mem_req(self, req: MemRequest, cycle: int = 0) -> MemResponse:
    """Validate, move real bytes, record the terminal reference, and answer MEM_RSP."""
    _ = cycle
    record = self._live_binding(req)
    needed = "r" if req.mem_op == MEM_OP_READ else "w"
    if needed not in record.permissions:
      self.stats.requests_rejected += 1
      raise MemoryServiceError("MEM_REQ_REJECTED", f"binding {record.name} lacks {needed!r} permission")
    if req.binding_offset + req.length > record.size:
      self.stats.requests_rejected += 1
      raise MemoryServiceError(
        "MEM_REQ_REJECTED", f"range [{req.binding_offset}, +{req.length}) exceeds binding size"
      )
    seen = self._txn_slices_seen.setdefault(req.txn_id, set())
    if req.slice_id in seen:
      self.stats.requests_rejected += 1
      raise MemoryServiceError("MEM_REQ_REJECTED", f"duplicate slice txn={req.txn_id} slice={req.slice_id}")

    handle = self.hbm.get_handle(req.binding_name)
    assert handle is not None
    pool = self.credits.pool("MEMORY", "T2G")
    pool.reserve_on_send()
    pool.accept_into_receiver()
    pool.consume_and_return()

    try:
      self.hbm.assert_live(handle)
      if req.mem_op == MEM_OP_READ:
        assert self.byte_store is not None
        data = self.byte_store.read_hbm(record.base + req.binding_offset, req.length)
      else:
        assert self.byte_store is not None
        segments = self.hbm.resolve(handle, req.binding_offset, req.length, "w")
        view = ResolvedMemoryView(
          handle=handle,
          offset_bytes=req.binding_offset,
          size_bytes=req.length,
          permissions=record.permissions,
          address=segments[0].address,
          segments=segments,
        )
        self.byte_store.write_view(view, req.payload)
        data = None
    except MemoryInvariantError as exc:
      seen.discard(req.slice_id)
      self.stats.requests_rejected += 1
      pool.reserve_on_send()
      pool.accept_into_receiver()
      pool.consume_and_return()
      raise MemoryServiceError("MEM_REQ_REJECTED", str(exc)) from exc

    seen.add(req.slice_id)
    self.stats.requests_accepted += 1
    if req.mem_op == MEM_OP_READ:
      self.stats.bytes_read += req.length
    else:
      self.stats.bytes_written += req.length
    self._terminal_refs[(req.txn_id, req.slice_id)] = _TerminalRef(
      txn_id=req.txn_id,
      slice_id=req.slice_id,
      slice_count=req.slice_count,
      binding_name=req.binding_name,
      mem_op=req.mem_op,
      length=req.length,
    )
    visibility = VIS_NONE if req.mem_op == MEM_OP_READ else VIS_HBM
    rsp_pool = self.credits.pool("MEMORY", "G2T")
    rsp_pool.reserve_on_send()
    rsp_pool.accept_into_receiver()
    rsp_pool.consume_and_return()
    return MemResponse(
      txn_id=req.txn_id,
      slice_id=req.slice_id,
      binding_generation=record.generation,
      visibility=visibility,
      status=STATUS_OK,
      data=data,
    )

  # -- terminal path -----------------------------------------------------
  def terminal_ack(self, txn_id: int, slice_id: int, slice_count: int, final_state: str) -> None:
    """Release one terminal reference; forged/duplicate acks leave the ledger unchanged."""
    if final_state not in ("COMPLETED", "ISOLATED", "FAULTED"):
      raise MemoryServiceError("BAD_FINAL_STATE", final_state)
    key = (txn_id, slice_id)
    ref = self._terminal_refs.get(key)
    if ref is None:
      raise MemoryServiceError("TERMINAL_ACK_UNKNOWN", f"txn={txn_id} slice={slice_id}")
    if ref.slice_count != slice_count:
      raise MemoryServiceError(
        "TERMINAL_ACK_SLICE_COUNT", f"txn={txn_id}: {slice_count} != {ref.slice_count}"
      )
    del self._terminal_refs[key]
    self.stats.terminal_acks += 1

  def forge_credit_return(self, channel: str = "MEMORY", direction: str = "T2G") -> None:
    """Test-only hook: a fabricated credit return must break conservation loudly."""
    self.credits.pool(channel, direction).consume_and_return()

  # -- closure -----------------------------------------------------------
  def unclosed(self) -> list[str]:
    return sorted(
      f"txn={ref.txn_id}:slice={ref.slice_id}:{ref.binding_name}" for ref in self._terminal_refs.values()
    )

  def assert_closed(self) -> None:
    residual = self.unclosed()
    if residual:
      raise MemoryServiceError("TERMINAL_REFS_OUTSTANDING", "; ".join(residual))
    if not self.credits.conserved():
      raise CreditConservationError("memory-service channel credit conservation broken")

  def ledger_snapshot(self) -> dict[str, Any]:
    """Ledger-only view for before/after rejection assertions (diagnostic counters excluded)."""
    return {
      "bindings": {
        name: {
          "base": b.base,
          "size": b.size,
          "permissions": b.permissions,
          "generation": b.generation,
          "released": b.released,
        }
        for name, b in self._bindings.items()
      },
      "terminal_refs": sorted(self._terminal_refs.keys()),
      "txn_slices": {txn: sorted(slices) for txn, slices in self._txn_slices_seen.items()},
      "credits": self.credits.snapshot(),
    }

  def snapshot(self) -> dict[str, Any]:
    return {
      "bindings": {
        name: {
          "base": b.base,
          "size": b.size,
          "permissions": b.permissions,
          "generation": b.generation,
          "released": b.released,
        }
        for name, b in self._bindings.items()
      },
      "terminal_refs_outstanding": len(self._terminal_refs),
      "stats": {
        "requests_accepted": self.stats.requests_accepted,
        "requests_rejected": self.stats.requests_rejected,
        "bytes_read": self.stats.bytes_read,
        "bytes_written": self.stats.bytes_written,
        "terminal_acks": self.stats.terminal_acks,
      },
      "credits": self.credits.snapshot(),
    }
