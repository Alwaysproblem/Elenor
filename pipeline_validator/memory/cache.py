"""Deterministic profiled LRU cache with byte/provenance maintenance state."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass

from .allocator import MemoryInvariantError


@dataclass(frozen=True)
class CacheStats:
  hits: int
  misses: int
  refills: int
  evictions: int
  resident_lines: int
  resident_bytes: int
  capacity_bytes: int
  dirty_lines: int = 0
  pending_cleans: int = 0
  discarded_dirty_lines: int = 0


@dataclass(frozen=True)
class CacheLineIdentity:
  """Allocation-qualified precise line or conservative opaque range."""

  allocation_id: str
  allocation_generation: int
  line_offset: int
  span_bytes: int = 0
  opaque_token: str = ""
  precise: bool = True


@dataclass(frozen=True)
class CacheProvenance:
  binding_name: str
  allocation_id: str
  allocation_generation: int
  source_offset: int
  bytes: int
  precise: bool = True

  @property
  def end_offset(self) -> int:
    return self.source_offset + self.bytes


@dataclass(frozen=True)
class CacheRange:
  allocation_id: str
  allocation_generation: int
  offset: int
  bytes: int

  @property
  def end(self) -> int:
    return self.offset + self.bytes


@dataclass(frozen=True)
class CacheCleanRequest:
  request_id: str
  identity: CacheLineIdentity
  provenance: CacheProvenance
  data: bytes
  profile_generation: int


@dataclass
class _CacheLine:
  identity: CacheLineIdentity
  token: str | None
  provenance: CacheProvenance
  data: bytes | None
  profile_generation: int
  version: int
  dirty: bool = False
  # Per-byte validity mask for partially-initialised lines (plan §2);
  # None means all bytes valid (legacy/test seeds).
  validity: bytes | None = None
  pending_clean: str | None = None
  invalidate_after_clean: bool = False


class DeterministicLRUCache:
  """Shared per-pool LRU cache.

  The IR still supplies the modeled hit/miss outcome, but resident lines carry
  a real allocation generation, provenance, optional bytes, dirty state and
  profile generation.  Capacity zero is a disabled cache, not a one-line
  fallback.  Dirty lines are never discarded by refill, maintenance, or
  reconfiguration.  Only explicit post-isolation recovery may destructively
  discard volatile dirty state.
  """

  def __init__(
    self,
    capacity_bytes: int,
    line_bytes: int,
    *,
    write_policy: str = "read_only",
    profile_generation: int = 0,
    level: str = "",
    pool_id: int = 0,
  ):
    if line_bytes <= 0 or line_bytes & (line_bytes - 1):
      raise ValueError("cache line_bytes must be a positive power of 2")
    if capacity_bytes < 0 or capacity_bytes % line_bytes:
      raise ValueError("cache capacity_bytes must be non-negative and line-aligned")
    if write_policy not in ("read_only", "write_back"):
      raise ValueError("unsupported cache write policy")
    if profile_generation < 0:
      raise ValueError("cache profile generation must be non-negative")
    self.capacity_bytes = capacity_bytes
    self.line_bytes = line_bytes
    self.write_policy = write_policy
    self.profile_generation = profile_generation
    self.level = level
    self.pool_id = pool_id
    self._capacity_lines = capacity_bytes // line_bytes
    self._lines: OrderedDict[CacheLineIdentity, _CacheLine] = OrderedDict()
    self._tokens: dict[str, set[CacheLineIdentity]] = {}
    self._pending_cleans: dict[str, CacheLineIdentity] = {}
    self._next_clean_id = 0
    self._next_version = 0
    self._hits = 0
    self._misses = 0
    self._refills = 0
    self._evictions = 0
    self._discarded_dirty_lines = 0

  @property
  def enabled(self) -> bool:
    return self._capacity_lines > 0

  @staticmethod
  def identity_for(
    allocation_id: str, allocation_generation: int, line_offset: int, span_bytes: int = 0
  ) -> CacheLineIdentity:
    if not allocation_id or allocation_generation < 0 or line_offset < 0:
      raise MemoryInvariantError("invalid cache line identity")
    return CacheLineIdentity(allocation_id, allocation_generation, line_offset, span_bytes)

  def _resolve_identity(
    self, token_or_identity: str | CacheLineIdentity | None
  ) -> CacheLineIdentity | None:
    if isinstance(token_or_identity, CacheLineIdentity):
      return token_or_identity
    if isinstance(token_or_identity, str):
      identities = self._tokens.get(token_or_identity, set())
      return next(iter(identities)) if len(identities) == 1 else None
    return None

  def contains(self, token_or_identity: str | CacheLineIdentity) -> bool:
    identity = self._resolve_identity(token_or_identity)
    return identity is not None and identity in self._lines

  def record_hit(self, token: str | CacheLineIdentity | None, *, require_resident: bool = False) -> None:
    if not self.enabled:
      raise MemoryInvariantError("explicit cache hit targets a disabled cache")
    identity = self._resolve_identity(token)
    if require_resident and (identity is None or identity not in self._lines):
      raise MemoryInvariantError("profiled cache hit has no resident line")
    self._hits += 1
    if identity is not None and identity in self._lines:
      line = self._lines[identity]
      if line.profile_generation != self.profile_generation:
        raise MemoryInvariantError("cache hit references an old profile generation")
      self._lines.move_to_end(identity)

  def read_validity(self, token_or_identity: str | CacheLineIdentity) -> bytes | None:
    if not self.enabled:
      raise MemoryInvariantError("cache validity read targets a disabled cache")
    identity = self._resolve_identity(token_or_identity)
    line = self._lines.get(identity) if identity is not None else None
    if line is None or line.profile_generation != self.profile_generation:
      raise MemoryInvariantError("cache line is not resident in the active generation")
    return line.validity

  def clear_validity_range(
    self, allocation_id: str, allocation_generation: int, offset: int, length: int
  ) -> None:
    """Oracle-only: drop validity bits of cached copies of a range.

    Plan §4: page hand-off invalidates cached validity for the old
    owner's data so the new owner can never observe stale bytes.  Tags,
    LRU order and counters are untouched and no traffic is billed.
    """
    for identity, line in self._lines.items():
      if identity.allocation_id != allocation_id:
        continue
      if identity.allocation_generation != allocation_generation:
        continue
      if line.validity is None:
        continue
      line_start = identity.line_offset
      line_end = line_start + len(line.validity)
      overlap_start = max(line_start, offset)
      overlap_end = min(line_end, offset + length)
      if overlap_start >= overlap_end:
        continue
      mask = bytearray(line.validity)
      for position in range(overlap_start - line_start, overlap_end - line_start):
        mask[position] = 0
      line.validity = bytes(mask)

  def record_miss(self) -> None:
    if self.enabled:
      self._misses += 1

  def read_line(
    self, token_or_identity: str | CacheLineIdentity, *, require_data: bool = True
  ) -> bytes | None:
    if not self.enabled:
      raise MemoryInvariantError("cache read targets a disabled cache")
    identity = self._resolve_identity(token_or_identity)
    line = self._lines.get(identity) if identity is not None else None
    if line is None or line.profile_generation != self.profile_generation:
      raise MemoryInvariantError("cache line is not resident in the active generation")
    assert identity is not None
    if require_data and line.data is None:
      raise MemoryInvariantError("cache line has no byte-oracle payload")
    self._lines.move_to_end(identity)
    return line.data

  def _index_token(self, token: str | None, identity: CacheLineIdentity) -> None:
    if token:
      self._tokens.setdefault(token, set()).add(identity)

  def _unindex_token(self, token: str | None, identity: CacheLineIdentity) -> None:
    if not token:
      return
    identities = self._tokens.get(token)
    if identities is None:
      return
    identities.discard(identity)
    if not identities:
      self._tokens.pop(token, None)

  def install_metadata(
    self, token: str | None, identity: CacheLineIdentity, provenance: CacheProvenance
  ) -> None:
    """Install allocation-qualified metadata without counting a refill."""
    if not self.enabled:
      raise MemoryInvariantError("cache metadata targets a disabled cache")
    if provenance.bytes <= 0:
      raise MemoryInvariantError("cache provenance range must be positive")
    if (
      provenance.allocation_id != identity.allocation_id
      or provenance.allocation_generation != identity.allocation_generation
      or provenance.source_offset != identity.line_offset
    ):
      raise MemoryInvariantError("cache identity/provenance mismatch")
    if provenance.precise and provenance.bytes != self.line_bytes:
      raise MemoryInvariantError("precise cache provenance must describe one line")
    if not provenance.precise and identity.precise:
      raise MemoryInvariantError("conservative provenance requires opaque identity")
    existing = self._lines.get(identity)
    if existing is not None:
      if existing.pending_clean is not None:
        raise MemoryInvariantError("cannot update metadata during pending clean")
      if token and existing.token != token:
        self._unindex_token(existing.token, identity)
        existing.token = token
      existing.provenance = provenance
      self._next_version += 1
      existing.version = self._next_version
      self._index_token(existing.token, identity)
      self._lines.move_to_end(identity)
      return
    while len(self._lines) >= self._capacity_lines:
      self._evict_one()
    self._next_version += 1
    self._lines[identity] = _CacheLine(
      identity=identity,
      token=token,
      provenance=provenance,
      data=None,
      profile_generation=self.profile_generation,
      version=self._next_version,
    )
    self._index_token(token, identity)

  def _evict_one(self) -> None:
    if not self._lines:
      return
    identity, line = next(iter(self._lines.items()))
    if line.dirty or line.pending_clean is not None:
      raise MemoryInvariantError("dirty cache line requires a real clean before eviction")
    self._lines.pop(identity)
    self._unindex_token(line.token, identity)
    self._evictions += 1

  def refill(
    self,
    token: str | None,
    *,
    identity: CacheLineIdentity | None = None,
    provenance: CacheProvenance | None = None,
    data: bytes | None = None,
    validity: bytes | None = None,
    profile_generation: int | None = None,
    dirty: bool = False,
  ) -> None:
    if not self.enabled:
      raise MemoryInvariantError("cache refill targets a disabled cache")
    generation = self.profile_generation if profile_generation is None else profile_generation
    if generation != self.profile_generation:
      raise MemoryInvariantError("old-generation cache refill rejected")
    if dirty and self.write_policy != "write_back":
      raise MemoryInvariantError("dirty line requires write_back cache policy")
    if data is not None and len(data) != self.line_bytes:
      raise MemoryInvariantError("cache refill data must be exactly one line")
    if validity is not None and len(validity) != self.line_bytes:
      raise MemoryInvariantError("cache refill validity must be exactly one line")
    if identity is None:
      if provenance is not None:
        identity = CacheLineIdentity(
          provenance.allocation_id,
          provenance.allocation_generation,
          provenance.source_offset,
          provenance.bytes,
          token or "",
          provenance.precise,
        )
      elif token:
        # Legacy component-only metadata. Formal Gather always supplies an
        # allocation-qualified identity and provenance.
        identity = CacheLineIdentity(f"opaque:{token}", 0, 0, self.line_bytes, token, False)
        provenance = CacheProvenance("", identity.allocation_id, 0, 0, self.line_bytes, False)
      else:
        raise MemoryInvariantError("cache refill requires a line identity or token")
    if provenance is None:
      provenance = CacheProvenance(
        "",
        identity.allocation_id,
        identity.allocation_generation,
        identity.line_offset,
        identity.span_bytes or self.line_bytes,
        identity.precise,
      )
    if (
      provenance.allocation_id != identity.allocation_id
      or provenance.allocation_generation != identity.allocation_generation
      or provenance.source_offset != identity.line_offset
    ):
      raise MemoryInvariantError("cache identity/provenance mismatch")
    if provenance.bytes <= 0:
      raise MemoryInvariantError("cache provenance range must be positive")
    if provenance.precise and provenance.bytes != self.line_bytes:
      raise MemoryInvariantError("precise cache provenance must describe one line")
    if not provenance.precise and identity.precise:
      raise MemoryInvariantError("conservative provenance requires opaque identity")
    self._refills += 1
    existing = self._lines.get(identity)
    if existing is not None:
      if existing.pending_clean is not None:
        raise MemoryInvariantError("cannot refill a line while its clean is pending")
      existing.provenance = provenance
      existing.data = data if data is not None else existing.data
      existing.validity = validity if validity is not None else existing.validity
      existing.dirty = existing.dirty or dirty
      existing.profile_generation = generation
      if token and existing.token != token:
        self._unindex_token(existing.token, identity)
        existing.token = token
      self._index_token(existing.token, identity)
      self._lines.move_to_end(identity)
      return
    while len(self._lines) >= self._capacity_lines:
      self._evict_one()
    self._next_version += 1
    self._lines[identity] = _CacheLine(
      identity=identity,
      token=token,
      provenance=provenance,
      data=data,
      validity=validity,
      profile_generation=generation,
      version=self._next_version,
      dirty=dirty,
    )
    self._index_token(token, identity)

  def seed_line(
    self,
    *,
    allocation_id: str,
    allocation_generation: int,
    line_offset: int,
    binding_name: str,
    data: bytes,
    dirty: bool = False,
  ) -> CacheLineIdentity:
    if line_offset % self.line_bytes:
      raise MemoryInvariantError("cache seed line offset is not aligned")
    identity = self.identity_for(allocation_id, allocation_generation, line_offset)
    self.refill(
      None,
      identity=identity,
      provenance=CacheProvenance(
        binding_name, allocation_id, allocation_generation, line_offset, self.line_bytes
      ),
      data=data,
      dirty=dirty,
    )
    return identity

  def mark_dirty(self, token_or_identity: str | CacheLineIdentity, data: bytes | None = None) -> None:
    if self.write_policy != "write_back":
      raise MemoryInvariantError("read-only cache line cannot become dirty")
    identity = self._resolve_identity(token_or_identity)
    line = self._lines.get(identity) if identity is not None else None
    if line is None:
      raise MemoryInvariantError("cannot dirty a non-resident cache line")
    assert identity is not None
    if line.pending_clean is not None:
      raise MemoryInvariantError("cannot dirty a cache line while clean is pending")
    if data is not None:
      if len(data) != self.line_bytes:
        raise MemoryInvariantError("dirty cache data must be exactly one line")
      line.data = data
    if line.data is None:
      raise MemoryInvariantError("dirty cache line requires byte data")
    line.dirty = True
    self._next_version += 1
    line.version = self._next_version
    self._lines.move_to_end(identity)

  def _matches(self, line: _CacheLine, ranges: tuple[CacheRange, ...] | None) -> bool:
    if ranges is None:
      return True
    provenance = line.provenance
    return any(
      item.allocation_id == provenance.allocation_id
      and item.allocation_generation == provenance.allocation_generation
      and item.offset < provenance.end_offset
      and provenance.source_offset < item.end
      for item in ranges
    )

  def begin_maintenance(
    self, ranges: tuple[CacheRange, ...] | None, *, clean: bool, invalidate: bool
  ) -> tuple[CacheCleanRequest, ...]:
    """Start range/full maintenance and return dirty writeback work.

    Clean lines are invalidated immediately.  Dirty lines remain resident and
    inaccessible to reconfiguration until their returned request completes.
    """
    requests: list[CacheCleanRequest] = []
    for identity, line in tuple(self._lines.items()):
      if not self._matches(line, ranges):
        continue
      if line.pending_clean is not None:
        continue
      if line.dirty:
        if not clean:
          raise MemoryInvariantError("dirty cache line cannot be invalidated without clean")
        if line.data is None:
          raise MemoryInvariantError("dirty cache line has no bytes to clean")
        self._next_clean_id += 1
        request_id = (
          f"cache-clean:{self.level}:{self.pool_id}:g{self.profile_generation}:{self._next_clean_id}"
        )
        line.pending_clean = request_id
        line.invalidate_after_clean = invalidate
        self._pending_cleans[request_id] = identity
        requests.append(
          CacheCleanRequest(
            request_id=request_id,
            identity=identity,
            provenance=line.provenance,
            data=line.data,
            profile_generation=line.profile_generation,
          )
        )
      elif invalidate:
        self._remove_line(identity)
    return tuple(requests)

  def complete_clean(self, request_id: str, *, success: bool = True) -> None:
    identity = self._pending_cleans.pop(request_id, None)
    if identity is None:
      raise MemoryInvariantError("unknown cache clean request")
    line = self._lines.get(identity)
    if line is None or line.pending_clean != request_id:
      raise MemoryInvariantError("cache clean request lost its resident line")
    line.pending_clean = None
    if not success:
      line.invalidate_after_clean = False
      return
    line.dirty = False
    if line.invalidate_after_clean:
      self._remove_line(identity)
    else:
      line.invalidate_after_clean = False

  def _remove_line(self, identity: CacheLineIdentity) -> None:
    line = self._lines.pop(identity, None)
    if line is None:
      return
    if line.pending_clean is not None:
      raise MemoryInvariantError("cannot remove a line with pending clean work")
    self._unindex_token(line.token, identity)

  def reset_after_isolation(self, *, refills_quiescent: bool) -> int:
    """Destructively clear volatile lines after external isolation proof."""
    if type(refills_quiescent) is not bool:
      raise ValueError("refills_quiescent must be a bool")
    if not refills_quiescent:
      raise MemoryInvariantError("recovery reset requires all cache refills to drain")
    if self._pending_cleans or any(line.pending_clean is not None for line in self._lines.values()):
      raise MemoryInvariantError("recovery reset requires all cache cleans to isolate")
    discarded = sum(line.dirty for line in self._lines.values())
    self._lines.clear()
    self._tokens.clear()
    self._discarded_dirty_lines += discarded
    return discarded

  def reconfigure(self, capacity_bytes: int, generation: int, *, write_policy: str | None = None) -> None:
    if capacity_bytes < 0 or capacity_bytes % self.line_bytes:
      raise MemoryInvariantError("cache profile capacity is not line-aligned")
    if generation <= self.profile_generation:
      raise MemoryInvariantError("cache profile generation must increase monotonically")
    if self._pending_cleans:
      raise MemoryInvariantError("cannot reconfigure cache with pending clean work")
    if any(line.dirty for line in self._lines.values()):
      raise MemoryInvariantError("cannot reconfigure cache with dirty lines")
    if write_policy is not None:
      if write_policy not in ("read_only", "write_back"):
        raise MemoryInvariantError("unsupported cache write policy")
      self.write_policy = write_policy
    self._lines.clear()
    self._tokens.clear()
    self.capacity_bytes = capacity_bytes
    self._capacity_lines = capacity_bytes // self.line_bytes
    self.profile_generation = generation

  @property
  def stats(self) -> CacheStats:
    resident_lines = len(self._lines)
    return CacheStats(
      hits=self._hits,
      misses=self._misses,
      refills=self._refills,
      evictions=self._evictions,
      resident_lines=resident_lines,
      resident_bytes=resident_lines * self.line_bytes,
      capacity_bytes=self.capacity_bytes,
      dirty_lines=sum(line.dirty for line in self._lines.values()),
      pending_cleans=len(self._pending_cleans),
      discarded_dirty_lines=self._discarded_dirty_lines,
    )

  def snapshot(self) -> dict[str, object]:
    return {
      **asdict(self.stats),
      "enabled": self.enabled,
      "level": self.level,
      "pool_id": self.pool_id,
      "profile_generation": self.profile_generation,
      "write_policy": self.write_policy,
      "resident_tokens": tuple(line.token for line in self._lines.values() if line.token is not None),
      "lines": tuple(
        {
          "allocation_id": identity.allocation_id,
          "allocation_generation": identity.allocation_generation,
          "line_offset": identity.line_offset,
          "span_bytes": identity.span_bytes,
          "opaque_token": identity.opaque_token,
          "precise": identity.precise,
          "version": line.version,
          "dirty": line.dirty,
          "pending_clean": line.pending_clean,
          "binding_name": line.provenance.binding_name,
          "source_offset": line.provenance.source_offset,
          "provenance_bytes": line.provenance.bytes,
          "provenance_precise": line.provenance.precise,
          "has_data": line.data is not None,
        }
        for identity, line in self._lines.items()
      ),
    }

  def reset(self) -> None:
    if self._pending_cleans or any(line.dirty for line in self._lines.values()):
      raise MemoryInvariantError("cache reset would discard uncompleted dirty state")
    self._lines.clear()
    self._tokens.clear()
    self._hits = 0
    self._misses = 0
    self._refills = 0
    self._evictions = 0
    self._discarded_dirty_lines = 0
