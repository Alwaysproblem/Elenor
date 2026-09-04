"""Deterministic metadata-only LRU cache for profiled Gather requests."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CacheStats:
  hits: int
  misses: int
  refills: int
  evictions: int
  resident_lines: int
  resident_bytes: int
  capacity_bytes: int


class DeterministicLRUCache:
  """Metadata cache whose hit/miss outcome is supplied by the IR profile.

  Line tokens are opaque identities.  They affect only deterministic LRU
  residency; they are never interpreted as addresses, banks, or sets.
  """

  def __init__(self, capacity_bytes: int, line_bytes: int):
    if line_bytes <= 0 or line_bytes & (line_bytes - 1):
      raise ValueError("cache line_bytes must be a positive power of 2")
    if capacity_bytes < line_bytes or capacity_bytes % line_bytes:
      raise ValueError("cache capacity_bytes must be line-aligned and at least one line")
    self.capacity_bytes = capacity_bytes
    self.line_bytes = line_bytes
    self._capacity_lines = capacity_bytes // line_bytes
    self._lines: OrderedDict[str, None] = OrderedDict()
    self._hits = 0
    self._misses = 0
    self._refills = 0
    self._evictions = 0

  def record_hit(self, token: str | None) -> None:
    self._hits += 1
    if token and token in self._lines:
      self._lines.move_to_end(token)

  def record_miss(self) -> None:
    self._misses += 1

  def refill(self, token: str | None) -> None:
    self._refills += 1
    if not token:
      return
    if token in self._lines:
      self._lines.move_to_end(token)
      return
    if len(self._lines) == self._capacity_lines:
      self._lines.popitem(last=False)
      self._evictions += 1
    self._lines[token] = None

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
    )

  def snapshot(self) -> dict[str, object]:
    return {**asdict(self.stats), "resident_tokens": tuple(self._lines)}

  def reset(self) -> None:
    self._lines.clear()
    self._hits = 0
    self._misses = 0
    self._refills = 0
    self._evictions = 0
