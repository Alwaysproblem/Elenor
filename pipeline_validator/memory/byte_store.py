"""Sparse byte-accurate memory oracle used by full-memory simulations.

The timing simulator does not allocate a dense HBM image.  ``ByteStore`` keeps
only pages which a host seeded or a completed device write touched, together
with a per-byte validity bitmap.  Reads of uninitialised bytes are errors;
there is no implicit zero-fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .allocator import AllocationHandle, BankSegment, ExternalOwner, MemoryInvariantError

if TYPE_CHECKING:
  from .cache import CacheLineIdentity, DeterministicLRUCache
  from .transfer import ResolvedMemoryView


_PAGE_BYTES = 4096


@dataclass(frozen=True)
class _Binding:
  name: str
  allocation_id: str
  generation: int
  base: int
  size: int
  permissions: str
  handle: AllocationHandle

  @property
  def end(self) -> int:
    return self.base + self.size


@dataclass(frozen=True)
class _CacheSeedSpec:
  level: str
  pool_id: int
  binding_name: str
  source_offset: int
  data: bytes
  dirty: bool

  @property
  def key(self) -> tuple[str, int, str, int]:
    return (self.level, self.pool_id, self.binding_name, self.source_offset)


class ByteStore:
  """Sparse HBM/SPM/cache data oracle.

  Host seeding is intentionally independent of device write permissions, but
  every seeded HBM byte must be covered by an actual launch binding before a
  run starts.  Device reads and writes enforce the binding permissions.
  """

  def __init__(self, *, page_bytes: int = _PAGE_BYTES) -> None:
    if type(page_bytes) is not int or page_bytes <= 0 or page_bytes & (page_bytes - 1):
      raise ValueError("byte-store page size must be a positive power of two")
    self.page_bytes = page_bytes
    self._pages: dict[tuple[str, int], bytearray] = {}
    self._valid: dict[tuple[str, int], bytearray] = {}
    self._bindings: dict[str, _Binding] = {}
    self._seed_ranges: list[tuple[int, int]] = []
    self._caches: dict[tuple[str, int], DeterministicLRUCache] = {}
    self._pending_cache_seeds: list[_CacheSeedSpec] = []
    self._seeded_cache_keys: set[tuple[str, int, str, int]] = set()

  # -- sparse storage -------------------------------------------------

  def _domain_for_view(self, view: ResolvedMemoryView) -> str:
    handle = view.handle
    if handle.memory_space == "hbm":
      return "hbm"
    profile_generation = getattr(handle, "profile_generation", None)
    if handle.memory_space == "l2":
      # Physical identity: producer bytes stay readable through view ownership
      # changes because the domain keys on the backing, not the logical view.
      return f"l2:{handle.backing_id}:{handle.generation}:{profile_generation}"
    return f"{handle.memory_space}:{handle.allocation_id}:{handle.generation}:{profile_generation}"

  def _write(self, domain: str, address: int, data: bytes) -> None:
    if type(address) is not int or address < 0:
      raise MemoryInvariantError("byte-store address must be non-negative")
    cursor = 0
    while cursor < len(data):
      absolute = address + cursor
      page_index, page_offset = divmod(absolute, self.page_bytes)
      count = min(len(data) - cursor, self.page_bytes - page_offset)
      key = (domain, page_index)
      page = self._pages.setdefault(key, bytearray(self.page_bytes))
      valid = self._valid.setdefault(key, bytearray(self.page_bytes))
      page[page_offset : page_offset + count] = data[cursor : cursor + count]
      valid[page_offset : page_offset + count] = b"\x01" * count
      cursor += count

  def _read(self, domain: str, address: int, size: int) -> bytes:
    if type(address) is not int or type(size) is not int or address < 0 or size < 0:
      raise MemoryInvariantError("invalid byte-store read range")
    data, mask = self._read_relaxed(domain, address, size)
    first = mask.find(b"\x00")
    if first >= 0:
      raise MemoryInvariantError(
        f"byte oracle read includes uninitialised bytes at {domain}:{address + first}"
      )
    return data

  def _read_relaxed(self, domain: str, address: int, size: int) -> tuple[bytes, bytes]:
    if type(address) is not int or type(size) is not int or address < 0 or size < 0:
      raise MemoryInvariantError("invalid byte-store read range")
    data = bytearray()
    mask = bytearray()
    cursor = 0
    while cursor < size:
      absolute = address + cursor
      page_index, page_offset = divmod(absolute, self.page_bytes)
      count = min(size - cursor, self.page_bytes - page_offset)
      key = (domain, page_index)
      page = self._pages.get(key)
      valid = self._valid.get(key)
      if page is None or valid is None:
        data.extend(b"\x00" * count)
        mask.extend(b"\x00" * count)
      else:
        data.extend(page[page_offset : page_offset + count])
        mask.extend(valid[page_offset : page_offset + count])
      cursor += count
    return bytes(data), bytes(mask)

  # -- HBM host/device interface -------------------------------------

  def seed_hbm(self, address: int, data: bytes, *, cycle: int | None = None) -> None:
    """Host-initialise HBM bytes without applying device permissions.

    Plan §4: seeding is a cycle-0 facility.  Passing ``cycle`` other than
    0 or None (pre-run preparation) is rejected; runtime writes must be
    billed through ``TransferManager`` rather than silently overwriting
    device-visible state from a host callback.
    """
    if cycle is not None and cycle != 0:
      raise MemoryInvariantError(
        "seed_hbm is a cycle-0 facility; runtime writes must be billed through transfers"
      )
    if not isinstance(data, bytes):
      raise TypeError("seed_hbm data must be bytes")
    if type(address) is not int or address < 0:
      raise ValueError("seed_hbm address must be non-negative")
    if not data:
      return
    self._write("hbm", address, data)
    self._seed_ranges.append((address, address + len(data)))

  def read_hbm(self, address: int, size: int) -> bytes:
    """Observe HBM bytes; uninitialised bytes fail instead of reading zero."""
    return self._read("hbm", address, size)

  def register_hbm_binding(self, handle: AllocationHandle, permissions: str) -> None:
    if not isinstance(handle.owner, ExternalOwner):
      raise MemoryInvariantError("byte-store HBM binding has a non-external owner")
    if permissions not in ("r", "w", "rw"):
      raise ValueError("invalid HBM binding permissions")
    name = handle.owner.binding_name
    prior = self._bindings.get(name)
    record = _Binding(
      name=name,
      allocation_id=handle.allocation_id,
      generation=handle.generation,
      base=handle.base_address,
      size=handle.size_bytes,
      permissions=permissions,
      handle=handle,
    )
    if prior is not None and prior != record:
      raise MemoryInvariantError("HBM binding generation replaced without unbind")
    self._bindings[name] = record

  def unregister_hbm_binding(self, name: str, allocation_id: str, generation: int) -> None:
    record = self._bindings.get(name)
    if record is None:
      return
    if record.allocation_id != allocation_id or record.generation != generation:
      raise MemoryInvariantError("stale HBM binding unregistration")
    del self._bindings[name]

  def validate_seed_coverage(self) -> None:
    """Require all host-seeded ranges to be covered by actual bindings."""
    for start, end in self._seed_ranges:
      cursor = start
      while cursor < end:
        binding = next((item for item in self._bindings.values() if item.base <= cursor < item.end), None)
        if binding is None:
          raise MemoryInvariantError(f"seeded HBM byte at {cursor} is outside actual launch bindings")
        cursor = min(end, binding.end)
    self._flush_pending_cache_seeds()
    self._seed_ranges.clear()

  def _validate_cache_seed_local(self, cache: DeterministicLRUCache, spec: _CacheSeedSpec) -> None:
    if not cache.enabled:
      raise MemoryInvariantError("cache seed targets a zero-capacity reset profile")
    if cache.level and cache.level != spec.level:
      raise MemoryInvariantError("cache seed level does not match registered cache")
    if cache.pool_id != spec.pool_id:
      raise MemoryInvariantError("cache seed pool does not match registered cache")
    if spec.source_offset % cache.line_bytes:
      raise MemoryInvariantError("cache seed source offset must be line-aligned")
    if len(spec.data) != cache.line_bytes:
      raise MemoryInvariantError("cache seed must provide exactly one cache line")
    if spec.dirty and cache.write_policy != "write_back":
      raise MemoryInvariantError("dirty cache seed requires write_back policy")

  def _prepare_cache_seed(
    self, spec: _CacheSeedSpec
  ) -> tuple[DeterministicLRUCache, _Binding, CacheLineIdentity]:
    try:
      cache = self._caches[(spec.level, spec.pool_id)]
    except KeyError as exc:
      raise MemoryInvariantError("cache seed pool is not registered before execution") from exc
    self._validate_cache_seed_local(cache, spec)
    try:
      binding = self._bindings[spec.binding_name]
    except KeyError as exc:
      raise MemoryInvariantError("cache seed references an unbound actual HBM input") from exc
    if "r" not in binding.permissions:
      raise MemoryInvariantError("cache seed source binding lacks read permission")
    if spec.source_offset + len(spec.data) > binding.size:
      raise MemoryInvariantError("cache seed exceeds its actual HBM binding")
    identity = cache.identity_for(binding.allocation_id, binding.generation, spec.source_offset)
    if spec.dirty and "w" not in binding.permissions:
      raise MemoryInvariantError("dirty cache seed source binding lacks write permission")
    return cache, binding, identity

  def _flush_pending_cache_seeds(self) -> None:
    if not self._pending_cache_seeds:
      return
    prepared = [(spec, *self._prepare_cache_seed(spec)) for spec in self._pending_cache_seeds]
    new_by_cache: dict[int, tuple[DeterministicLRUCache, set[CacheLineIdentity]]] = {}
    for _spec, cache, _binding, identity in prepared:
      cache_key = id(cache)
      _, identities = new_by_cache.setdefault(cache_key, (cache, set()))
      if not cache.contains(identity):
        identities.add(identity)
    for cache, identities in new_by_cache.values():
      capacity_lines = cache.capacity_bytes // cache.line_bytes
      if cache.stats.resident_lines + len(identities) > capacity_lines:
        raise MemoryInvariantError("cache seeds exceed active reset-profile capacity")
    for spec, cache, binding, _identity in prepared:
      cache.seed_line(
        allocation_id=binding.allocation_id,
        allocation_generation=binding.generation,
        line_offset=spec.source_offset,
        binding_name=spec.binding_name,
        data=spec.data,
        dirty=spec.dirty,
      )
      self._seeded_cache_keys.add(spec.key)
    self._pending_cache_seeds.clear()

  def _binding_for_view(self, view: ResolvedMemoryView, permission: str) -> _Binding:
    handle = view.handle
    if handle.memory_space != "hbm" or not isinstance(handle.owner, ExternalOwner):
      raise MemoryInvariantError("byte-store view is not an HBM binding")
    binding = self._bindings.get(handle.owner.binding_name)
    if (
      binding is None
      or binding.allocation_id != handle.allocation_id
      or binding.generation != handle.generation
      or view.address < binding.base
      or view.address + view.size_bytes > binding.end
    ):
      raise MemoryInvariantError("stale or out-of-bounds HBM byte-store view")
    if permission not in binding.permissions:
      raise MemoryInvariantError(f"HBM binding '{binding.name}' lacks device {permission} permission")
    return binding

  # -- transfer logical-segment interface -----------------------------

  def read_view(self, view: ResolvedMemoryView) -> bytes:
    if view.handle.memory_space == "hbm":
      self._binding_for_view(view, "r")
    domain = self._domain_for_view(view)
    result = bytearray()
    for segment in view.segments:
      result.extend(self._read(domain, segment.address, segment.size_bytes))
    if len(result) != view.size_bytes:
      raise MemoryInvariantError("resolved source segments do not cover logical bytes")
    return bytes(result)

  def read_view_relaxed(self, view: ResolvedMemoryView) -> tuple[bytes, bytes]:
    """Read bytes plus a per-byte validity mask without faulting.

    Plan §2: whole-line Gather refills legitimately cross uninitialised
    bytes (page tail tokens, padding); the mask travels with the line
    into the cache and only a destination write touching an
    uninitialised byte faults.  HBM permission checks still apply.
    """
    if view.handle.memory_space == "hbm":
      self._binding_for_view(view, "r")
    domain = self._domain_for_view(view)
    data = bytearray()
    mask = bytearray()
    for segment in view.segments:
      page_data, page_mask = self._read_relaxed(domain, segment.address, segment.size_bytes)
      data.extend(page_data)
      mask.extend(page_mask)
    if len(data) != view.size_bytes:
      raise MemoryInvariantError("resolved source segments do not cover logical bytes")
    return bytes(data), bytes(mask)

  def write_view_checked(self, view: ResolvedMemoryView, data: bytes, validity: bytes | None) -> None:
    """Commit bytes, refusing any uninitialised byte in the payload.

    Used by the Gather destination write: a segment that covers bytes
    the line refill carried as uninitialised faults here (plan §2).
    """
    if not isinstance(data, bytes) or len(data) != view.size_bytes:
      raise MemoryInvariantError("destination byte count does not match resolved view")
    if validity is not None and len(validity) != len(data):
      raise MemoryInvariantError("validity mask length does not match payload")
    if validity is not None and 0 in validity:
      offset = validity.find(b"\x00")
      raise MemoryInvariantError(
        f"gather destination segment includes uninitialised bytes at offset {offset}"
      )
    self.write_view(view, data)

  def write_view(self, view: ResolvedMemoryView, data: bytes) -> None:
    if not isinstance(data, bytes) or len(data) != view.size_bytes:
      raise MemoryInvariantError("destination byte count does not match resolved view")
    if view.handle.memory_space == "hbm":
      self._binding_for_view(view, "w")
    domain = self._domain_for_view(view)
    cursor = 0
    for segment in view.segments:
      next_cursor = cursor + segment.size_bytes
      self._write(domain, segment.address, data[cursor:next_cursor])
      cursor = next_cursor
    if cursor != len(data):
      raise MemoryInvariantError("resolved destination segments do not cover logical bytes")

  # -- cache test seeding ---------------------------------------------

  def register_cache(self, level: str, pool_id: int, cache: DeterministicLRUCache) -> None:
    if level not in ("l1", "l2") or type(pool_id) is not int or pool_id < 0:
      raise ValueError("invalid cache pool identity")
    key = (level, pool_id)
    previous = self._caches.get(key)
    if previous is not None and previous is not cache:
      raise MemoryInvariantError("cache pool registered twice")
    self._caches[key] = cache

  def binding_has_cached_state(self, binding_name: str) -> bool:
    """True when any registered cache still holds lines from ``binding_name``.

    Plan §4: the page-pool registration hook rejects a pool whose backing
    binding still carries cache copies (``managed_pool_has_cached_state``).
    Uses the public cache snapshot; no cache internals are touched.
    """
    for cache in self._caches.values():
      if not cache.enabled:
        continue
      snapshot = cache.snapshot()
      lines = snapshot.get("lines", ())
      for line in lines if isinstance(lines, tuple) else ():
        if isinstance(line, dict) and line.get("binding_name") == binding_name:
          return True
    return False

  def seed_cache_line(
    self,
    level: str,
    pool_id: int,
    binding_name: str,
    source_offset: int,
    data: bytes,
    *,
    dirty: bool = False,
  ) -> None:
    if level not in ("l1", "l2") or type(pool_id) is not int or pool_id < 0 or not binding_name:
      raise ValueError("invalid cache seed pool or binding identity")
    if type(source_offset) is not int or source_offset < 0:
      raise ValueError("cache seed source offset must be non-negative")
    if not isinstance(data, bytes) or not data:
      raise ValueError("cache seed data must be non-empty bytes")
    if type(dirty) is not bool:
      raise ValueError("cache seed dirty flag must be bool")
    spec = _CacheSeedSpec(level, pool_id, binding_name, source_offset, data, dirty)
    if spec.key in self._seeded_cache_keys or any(
      item.key == spec.key for item in self._pending_cache_seeds
    ):
      raise MemoryInvariantError("duplicate cache seed specification")

    cache = self._caches.get((level, pool_id))
    if cache is not None:
      self._validate_cache_seed_local(cache, spec)
    if cache is None or binding_name not in self._bindings:
      self._pending_cache_seeds.append(spec)
      return

    cache, binding, identity = self._prepare_cache_seed(spec)
    if (
      not cache.contains(identity)
      and cache.stats.resident_lines >= cache.capacity_bytes // cache.line_bytes
    ):
      raise MemoryInvariantError("cache seeds exceed active reset-profile capacity")
    cache.seed_line(
      allocation_id=binding.allocation_id,
      allocation_generation=binding.generation,
      line_offset=source_offset,
      binding_name=binding_name,
      data=data,
      dirty=dirty,
    )
    self._seeded_cache_keys.add(spec.key)

  def snapshot(self) -> dict[str, object]:
    return {
      "pages": len(self._pages),
      "valid_bytes": sum(sum(bitmap) for bitmap in self._valid.values()),
      "bindings": tuple(sorted(self._bindings)),
      "seed_ranges": tuple(self._seed_ranges),
      "cache_pools": tuple(sorted(self._caches)),
      "pending_cache_seeds": len(self._pending_cache_seeds),
      "seeded_cache_lines": len(self._seeded_cache_keys),
    }

  # ---------------------------------------------------------------------
  # Host runtime section (plan §4, appended; oracle-only invalidation)
  # ---------------------------------------------------------------------

  def invalidate_view(self, view: ResolvedMemoryView) -> None:
    """Clear validity bits for one resolved view range (plan §4).

    Oracle-only: the bytes themselves are not cleared, tags/LRU order is
    not touched, and no traffic is counted.  Both HBM validity bits and
    any cache copy of the range lose validity so a former owner's cached
    bytes can never satisfy a later read of a reallocated page.  A dirty
    cache copy over the range still refuses invalidation (writeback
    belongs to the maintenance protocol, not the page oracle).
    """
    if view.handle.memory_space != "hbm":
      raise MemoryInvariantError("invalidate_view targets HBM page ranges only")
    self._binding_for_view(view, "w")
    self._clear_hbm_validity(view.handle, view.offset_bytes, view.size_bytes, view.segments)
    self._clear_cache_validity(
      view.handle.allocation_id, view.handle.generation, view.offset_bytes, view.size_bytes
    )

  def invalidate_binding_range(
    self, handle: AllocationHandle, offset: int, size: int, *, binding_name: str | None = None
  ) -> None:
    """Clear validity for [offset, offset+size) inside one HBM binding."""
    del binding_name  # identity comes from the handle
    if handle.memory_space != "hbm":
      raise MemoryInvariantError("invalidate_binding_range targets HBM ranges only")
    if offset < 0 or size <= 0 or offset + size > handle.size_bytes:
      raise MemoryInvariantError("invalidate range exceeds its HBM binding")
    self._clear_hbm_validity(handle, offset, size, ())
    self._clear_cache_validity(
      handle.allocation_id, handle.generation, offset, size
    )

  def _clear_hbm_validity(
    self,
    handle: AllocationHandle,
    offset: int,
    size: int,
    segments: tuple[BankSegment, ...] | tuple[()],
  ) -> None:
    """Zero HBM validity bits for a binding-relative byte range."""
    if offset < 0 or offset + size > handle.size_bytes:
      raise MemoryInvariantError("invalidate range exceeds its HBM binding")
    physical_ranges: tuple[tuple[int, int], ...] = (
      tuple((segment.address, segment.size_bytes) for segment in segments)
      if segments
      else ((handle.base_address + offset, size),)
    )
    for address, length in physical_ranges:
      cursor = 0
      while cursor < length:
        absolute = address + cursor
        page_index, page_offset = divmod(absolute, self.page_bytes)
        count = min(length - cursor, self.page_bytes - page_offset)
        valid = self._valid.get(("hbm", page_index))
        if valid is not None:
          valid[page_offset : page_offset + count] = b"\x00" * count
        cursor += count

  def _clear_cache_validity(
    self, allocation_id: str, allocation_generation: int, offset: int, size: int
  ) -> None:
    """Drop cache copies of one binding-relative range, oracle-only."""
    from .cache import CacheRange

    if size <= 0:
      return
    ranges = (CacheRange(allocation_id, allocation_generation, offset, size),)
    for cache in self._caches.values():
      if cache.enabled:
        cache.begin_maintenance(ranges, clean=False, invalidate=True)
