"""Bounded generation-safe event storage for runtime and Group scheduling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class EventStatus(Enum):
  """Runtime event states.

  ``PENDING`` plus ``producer_bound`` distinguishes reserved/unbound from a
  registered producer.  Terminal failures are never overwritten by success.
  """

  PENDING = 0
  DONE = 1
  ERROR = 2
  TIMEOUT = 3
  RESET = 4


class EventProtocolError(RuntimeError):
  """An event owner, generation, or producer violated the event protocol."""


@dataclass
class EventEntry:
  """One bounded event instance keyed by its launch-namespaced name."""

  name: str
  id: int
  sequence: int = 0
  status: EventStatus = EventStatus.PENDING
  producer_id: int = 0
  timestamp: int = 0
  error_code: int = 0
  expected_sequence: int = 0
  owner: str = ""
  generation: int = 0
  producer_bound: bool = False
  consumer_refs: int = 0


class EventTable:
  """Finite event instances with explicit reservation and producer binding.

  The compatibility ``register``/``signal`` surface remains available to the
  runtime ABI.  Group scheduling uses ``reserve_many`` and ``bind_producer``
  so admission is all-or-nothing and an unbound dependency is distinguishable
  from a pending producer.
  """

  def __init__(self, capacity: int = 4096) -> None:
    if capacity < 1:
      raise ValueError("event table capacity must be positive")
    self.capacity = capacity
    self._entries: dict[str, EventEntry] = {}
    self._next_id: int = 0
    self._retired: deque[dict] = deque(maxlen=capacity)
    self.peak_reserved: int = 0
    self.pmu_stale_sequence_count: int = 0
    self.pmu_wrong_owner_count: int = 0
    self.pmu_duplicate_signal_count: int = 0
    self.pmu_capacity_reject_count: int = 0
    self.pmu_protocol_error_count: int = 0
    self.version: int = 0

  @property
  def reserved(self) -> int:
    return len(self._entries)

  def _allocate(self, name: str, owner: str, generation: int) -> EventEntry:
    if len(self._entries) >= self.capacity:
      self.pmu_capacity_reject_count += 1
      raise EventProtocolError("event table capacity exhausted")
    entry = EventEntry(name=name, id=self._next_id, owner=owner, generation=generation)
    self._next_id += 1
    self._entries[name] = entry
    self.peak_reserved = max(self.peak_reserved, len(self._entries))
    self.version += 1
    return entry

  def register(self, name: str, *, owner: str = "", generation: int = 0) -> EventEntry:
    """Compatibility registration for one event.

    Existing names may only be reopened by the same owner/generation.
    """

    entry = self._entries.get(name)
    if entry is not None:
      self._validate_identity(entry, owner, generation, allow_unspecified=True)
      return entry
    return self._allocate(name, owner, generation)

  def reserve_many(self, names: tuple[str, ...], *, owner: str, generation: int) -> bool:
    """Atomically reserve all unique names for one launch.

    Returns ``False`` only for temporary capacity pressure.  Identity clashes
    are protocol errors because waiting cannot make a foreign live owner safe.
    """

    unique = tuple(dict.fromkeys(names))
    missing: list[str] = []
    for name in unique:
      entry = self._entries.get(name)
      if entry is None:
        missing.append(name)
      else:
        self._validate_identity(entry, owner, generation)
    if len(self._entries) + len(missing) > self.capacity:
      self.pmu_capacity_reject_count += 1
      return False
    for name in missing:
      self._allocate(name, owner, generation)
    return True

  def cancel_reservation(self, *, owner: str, generation: int) -> None:
    """Drop an unactivated launch reservation without diagnostic retirement."""

    removed = False
    for name, entry in tuple(self._entries.items()):
      if entry.owner != owner or entry.generation != generation:
        continue
      if entry.producer_bound or entry.status is not EventStatus.PENDING:
        raise EventProtocolError("cannot cancel an active event reservation")
      self._entries.pop(name)
      removed = True
    if removed:
      self.version += 1

  def bind_producer(self, name: str, *, owner: str, generation: int, producer_id: int) -> EventEntry:
    """Bind a reserved event to exactly one registered action producer."""

    entry = self._entries.get(name)
    if entry is None:
      raise EventProtocolError(f"event {name!r} was not reserved")
    self._validate_identity(entry, owner, generation)
    if entry.producer_bound and entry.producer_id != producer_id:
      self.pmu_protocol_error_count += 1
      raise EventProtocolError(f"event {name!r} already has producer {entry.producer_id}")
    entry.producer_bound = True
    entry.producer_id = producer_id
    return entry

  def add_consumer(self, name: str, *, owner: str, generation: int) -> EventEntry:
    """Register a dependency consumer after its producer has been bound."""

    entry = self._entries.get(name)
    if entry is None:
      self.pmu_protocol_error_count += 1
      raise EventProtocolError(f"dependency event {name!r} is not reserved")
    self._validate_identity(entry, owner, generation)
    if not entry.producer_bound:
      self.pmu_protocol_error_count += 1
      raise EventProtocolError(f"dependency event {name!r} has no bound producer")
    entry.consumer_refs += 1
    return entry

  def get(self, name: str) -> EventEntry | None:
    return self._entries.get(name)

  def signal(
    self,
    name: str,
    status: EventStatus,
    producer_id: int,
    cycle: int,
    error_code: int = 0,
    *,
    expected_owner: str | None = None,
    expected_generation: int | None = None,
    expected_sequence: int | None = None,
  ) -> bool:
    """Signal one event after validating sequence and launch identity.

    Missing events are auto-registered only for the legacy ABI form with no
    expected identity.  Group completions always provide identity and can
    therefore never create or overwrite the wrong bounded slot.
    """

    entry = self._entries.get(name)
    if entry is None:
      if expected_owner is not None or expected_generation is not None:
        self.pmu_stale_sequence_count += 1
        return False
      try:
        entry = self.register(name)
      except EventProtocolError:
        return False
    try:
      self._validate_identity(
        entry,
        expected_owner if expected_owner is not None else entry.owner,
        expected_generation if expected_generation is not None else entry.generation,
      )
    except EventProtocolError:
      self.pmu_wrong_owner_count += 1
      return False
    sequence = expected_sequence
    if sequence is None and entry.expected_sequence != 0:
      sequence = entry.expected_sequence
    if sequence is not None and entry.sequence != sequence:
      self.pmu_stale_sequence_count += 1
      return False
    if entry.producer_bound and producer_id != entry.producer_id:
      self.pmu_wrong_owner_count += 1
      return False
    if entry.status is not EventStatus.PENDING:
      self.pmu_duplicate_signal_count += 1
      return False
    entry.status = status
    entry.producer_id = producer_id
    entry.timestamp = cycle
    entry.error_code = error_code
    return True

  def wait(
    self,
    name: str,
    expected_sequence: int | None = None,
    *,
    expected_owner: str | None = None,
    expected_generation: int | None = None,
  ) -> EventStatus | None:
    """Return terminal status, or ``None`` while absent/unbound/pending."""

    entry = self._entries.get(name)
    if entry is None:
      return None
    if expected_owner is not None and entry.owner != expected_owner:
      self.pmu_wrong_owner_count += 1
      return None
    if expected_generation is not None and entry.generation != expected_generation:
      self.pmu_stale_sequence_count += 1
      return None
    if expected_sequence is not None:
      entry.expected_sequence = expected_sequence
      if entry.sequence != expected_sequence:
        return None
    if not entry.producer_bound and entry.owner:
      return None
    if entry.status is EventStatus.PENDING:
      return None
    return entry.status

  def advance_sequence(self, name: str) -> int:
    entry = self._entries.get(name)
    if entry is None:
      entry = self.register(name)
    entry.sequence += 1
    entry.status = EventStatus.PENDING
    entry.producer_bound = False
    entry.consumer_refs = 0
    return entry.sequence

  def release_owner(self, *, owner: str, generation: int, cycle: int) -> int:
    """Reclaim all slots for one retired launch into bounded diagnostics."""

    released = 0
    for name, entry in tuple(self._entries.items()):
      if entry.owner != owner or entry.generation != generation:
        continue
      self._retired.append(
        {
          "name": entry.name,
          "id": entry.id,
          "seq": entry.sequence,
          "status": entry.status.name,
          "producer": entry.producer_id,
          "owner": entry.owner,
          "generation": entry.generation,
          "retired_cycle": cycle,
        }
      )
      self._entries.pop(name)
      released += 1
    if released:
      self.version += 1
    return released

  def reset(self) -> None:
    """Reset marks pending live events RESET; terminal failures stay terminal."""

    for entry in self._entries.values():
      if entry.status is EventStatus.PENDING:
        entry.status = EventStatus.RESET

  def clear(self) -> None:
    self._entries.clear()
    self._retired.clear()
    self._next_id = 0
    self.peak_reserved = 0
    self.pmu_stale_sequence_count = 0
    self.pmu_wrong_owner_count = 0
    self.pmu_duplicate_signal_count = 0
    self.pmu_capacity_reject_count = 0
    self.pmu_protocol_error_count = 0
    self.version += 1

  def snapshot(self) -> dict:
    return {
      "capacity": self.capacity,
      "reserved": len(self._entries),
      "peak_reserved": self.peak_reserved,
      "entries": {
        name: {
          "id": entry.id,
          "seq": entry.sequence,
          "status": entry.status.name,
          "producer": entry.producer_id,
          "producer_bound": entry.producer_bound,
          "owner": entry.owner,
          "generation": entry.generation,
          "consumer_refs": entry.consumer_refs,
        }
        for name, entry in self._entries.items()
      },
      "retired": list(self._retired),
      "stale_sequence": self.pmu_stale_sequence_count,
      "wrong_owner": self.pmu_wrong_owner_count,
      "duplicate_signal": self.pmu_duplicate_signal_count,
      "capacity_reject": self.pmu_capacity_reject_count,
      "protocol_error": self.pmu_protocol_error_count,
    }

  def _validate_identity(
    self, entry: EventEntry, owner: str, generation: int, *, allow_unspecified: bool = False
  ) -> None:
    if allow_unspecified and not owner and generation == 0:
      return
    if entry.owner != owner or entry.generation != generation:
      self.pmu_protocol_error_count += 1
      raise EventProtocolError(
        f"event {entry.name!r} belongs to {entry.owner!r}/{entry.generation}, not {owner!r}/{generation}"
      )
