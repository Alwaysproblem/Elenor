"""Deterministic leader/waiter MSHR table with profile generations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from .allocator import MemoryInvariantError

MshrCallback = Callable[[], None]


@dataclass(frozen=True)
class MshrAllocation:
  token: int
  leader: bool
  merge_group: str | None
  generation: int = 0


@dataclass(frozen=True)
class MshrWait:
  reason: str
  version: int


@dataclass(frozen=True)
class MshrStats:
  active: int
  merged: int
  stalls: int
  callbacks: int
  capacity: int
  version: int
  generation: int = 0


@dataclass
class _MshrEntry:
  token: int
  merge_group: str | None
  generation: int
  callbacks: list[MshrCallback] = field(default_factory=list)


class MshrTable:
  """Capacity-bounded MSHR table with deterministic merge and wakeup.

  Entries from different profile generations never merge.  Reconfiguration is
  permitted only after all refill callbacks and active entries have drained.
  """

  def __init__(self, capacity: int, *, generation: int = 0):
    if capacity <= 0:
      raise ValueError("MSHR capacity must be > 0")
    if generation < 0:
      raise ValueError("MSHR generation must be non-negative")
    self.capacity = capacity
    self.generation = generation
    self._entries: dict[int, _MshrEntry] = {}
    self._groups: dict[tuple[int, str], int] = {}
    self._next_token = 0
    self._merged = 0
    self._stalls = 0
    self._callbacks = 0
    self._version = 0

  @property
  def version(self) -> int:
    return self._version

  @property
  def active_count(self) -> int:
    return len(self._entries)

  def allocate(
    self, merge_group: str | None = None, *, generation: int | None = None
  ) -> MshrAllocation | MshrWait:
    requested_generation = self.generation if generation is None else generation
    if requested_generation != self.generation:
      raise MemoryInvariantError("MSHR allocation uses an old profile generation")
    if merge_group is not None:
      existing = self._groups.get((requested_generation, merge_group))
      if existing is not None:
        self._merged += 1
        return MshrAllocation(existing, False, merge_group, requested_generation)
    if len(self._entries) >= self.capacity:
      self._stalls += 1
      return MshrWait("mshr_full", self._version)

    token = self._next_token
    self._next_token += 1
    self._entries[token] = _MshrEntry(token, merge_group, requested_generation)
    if merge_group is not None:
      self._groups[(requested_generation, merge_group)] = token
    return MshrAllocation(token, True, merge_group, requested_generation)

  def wait(self, token: int, callback: MshrCallback) -> None:
    entry = self._entries.get(token)
    if entry is None:
      raise MemoryInvariantError("unknown MSHR token")
    if entry.generation != self.generation:
      raise MemoryInvariantError("MSHR waiter targets an old generation")
    entry.callbacks.append(callback)
    self._callbacks += 1

  def is_active(self, token: int) -> bool:
    return token in self._entries

  def cancel(self, token: int, *, generation: int | None = None) -> int:
    """Isolate one refill entry without invoking success callbacks."""
    entry = self._entries.get(token)
    if entry is None:
      raise MemoryInvariantError("unknown or completed MSHR token")
    expected = self.generation if generation is None else generation
    if entry.generation != expected or expected != self.generation:
      raise MemoryInvariantError("old-generation MSHR cancellation rejected")
    self._entries.pop(token)
    self._callbacks -= len(entry.callbacks)
    if entry.merge_group is not None:
      self._groups.pop((entry.generation, entry.merge_group), None)
    self._version += 1
    return len(entry.callbacks)

  def complete(self, token: int, *, generation: int | None = None) -> tuple[MshrCallback, ...]:
    entry = self._entries.get(token)
    if entry is None:
      raise MemoryInvariantError("unknown or completed MSHR token")
    expected = self.generation if generation is None else generation
    if entry.generation != expected or expected != self.generation:
      raise MemoryInvariantError("old-generation MSHR refill rejected")
    self._entries.pop(token)
    self._callbacks -= len(entry.callbacks)
    if entry.merge_group is not None:
      self._groups.pop((entry.generation, entry.merge_group), None)
    self._version += 1
    return tuple(entry.callbacks)

  def reconfigure(self, generation: int) -> None:
    if generation <= self.generation:
      raise MemoryInvariantError("MSHR generation must increase monotonically")
    if self._entries or self._groups:
      raise MemoryInvariantError("cannot reconfigure MSHR with active refills")
    self.generation = generation
    self._version += 1

  @property
  def stats(self) -> MshrStats:
    return MshrStats(
      active=len(self._entries),
      merged=self._merged,
      stalls=self._stalls,
      callbacks=self._callbacks,
      capacity=self.capacity,
      version=self._version,
      generation=self.generation,
    )

  def snapshot(self) -> dict[str, object]:
    return {
      **asdict(self.stats),
      "entries": tuple(
        {
          "token": entry.token,
          "merge_group": entry.merge_group,
          "generation": entry.generation,
          "callbacks": len(entry.callbacks),
        }
        for entry in self._entries.values()
      ),
    }

  def reset(self) -> None:
    if self._entries:
      raise MemoryInvariantError("MSHR reset would discard active refills")
    self._groups.clear()
    self._merged = 0
    self._stalls = 0
    self._version += 1
