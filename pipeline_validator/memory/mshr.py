"""Deterministic leader/waiter MSHR table for profiled Gather misses."""

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


@dataclass
class _MshrEntry:
  token: int
  merge_group: str | None
  callbacks: list[MshrCallback] = field(default_factory=list)


class MshrTable:
  """Capacity-bounded MSHR table with deterministic merge and wakeup."""

  def __init__(self, capacity: int):
    if capacity <= 0:
      raise ValueError("MSHR capacity must be > 0")
    self.capacity = capacity
    self._entries: dict[int, _MshrEntry] = {}
    self._groups: dict[str, int] = {}
    self._next_token = 0
    self._merged = 0
    self._stalls = 0
    self._version = 0

  @property
  def version(self) -> int:
    return self._version

  def allocate(self, merge_group: str | None = None) -> MshrAllocation | MshrWait:
    if merge_group is not None:
      existing = self._groups.get(merge_group)
      if existing is not None:
        self._merged += 1
        return MshrAllocation(existing, False, merge_group)
    if len(self._entries) >= self.capacity:
      self._stalls += 1
      return MshrWait("mshr_full", self._version)

    token = self._next_token
    self._next_token += 1
    self._entries[token] = _MshrEntry(token, merge_group)
    if merge_group is not None:
      self._groups[merge_group] = token
    return MshrAllocation(token, True, merge_group)

  def wait(self, token: int, callback: MshrCallback) -> None:
    entry = self._entries.get(token)
    if entry is None:
      raise MemoryInvariantError("unknown MSHR token")
    entry.callbacks.append(callback)

  def complete(self, token: int) -> tuple[MshrCallback, ...]:
    entry = self._entries.pop(token, None)
    if entry is None:
      raise MemoryInvariantError("unknown or completed MSHR token")
    if entry.merge_group is not None:
      self._groups.pop(entry.merge_group, None)
    self._version += 1
    return tuple(entry.callbacks)

  @property
  def stats(self) -> MshrStats:
    return MshrStats(
      active=len(self._entries),
      merged=self._merged,
      stalls=self._stalls,
      callbacks=sum(len(entry.callbacks) for entry in self._entries.values()),
      capacity=self.capacity,
      version=self._version,
    )

  def snapshot(self) -> dict[str, int]:
    return asdict(self.stats)

  def reset(self) -> None:
    self._entries.clear()
    self._groups.clear()
    self._next_token = 0
    self._merged = 0
    self._stalls = 0
    self._version = 0
