"""Bounded generation-safe event storage for runtime and Group scheduling."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
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
  future_uses: int = 0
  managed: bool = False


@dataclass
class _EventQuota:
  """Launch quota over immutable compiler-provided event metadata."""

  owner: str
  generation: int
  limit: int
  event_uses: Mapping[str, int]
  declared_references: int
  produced: set[str]


class EventTable:
  """Finite generation-safe event storage with launch quotas.

  A managed launch reserves only its compiler-proved frontier quota.  Event
  entries are allocated just in time when an action registers its outputs.
  The compatibility ``register``/``signal`` surface remains available for
  standalone runtime users and keeps its original persistent-entry behavior.
  """

  def __init__(self, capacity: int = 4096) -> None:
    if capacity < 1:
      raise ValueError("event table capacity must be positive")
    self.capacity = capacity
    self._entries: dict[str, EventEntry] = {}
    self._quotas: dict[tuple[str, int], _EventQuota] = {}
    self._next_id: int = 0
    self._retired: deque[dict] = deque(maxlen=capacity)
    self.peak_reserved: int = 0
    self.peak_active: int = 0
    self.pmu_stale_sequence_count: int = 0
    self.pmu_wrong_owner_count: int = 0
    self.pmu_duplicate_signal_count: int = 0
    self.pmu_capacity_reject_count: int = 0
    self.pmu_protocol_error_count: int = 0
    self.version: int = 0

  @property
  def reserved(self) -> int:
    """Storage claimed by quotas plus unmanaged compatibility entries."""

    return sum(quota.limit for quota in self._quotas.values()) + sum(
      not entry.managed for entry in self._entries.values()
    )

  @property
  def active(self) -> int:
    """Number of event names with a currently allocated hardware slot."""

    return len(self._entries)

  def _quota(self, owner: str, generation: int) -> _EventQuota:
    quota = self._quotas.get((owner, generation))
    if quota is None:
      self.pmu_protocol_error_count += 1
      raise EventProtocolError(f"event quota for {owner!r}/{generation} was not reserved")
    return quota

  def _managed_live(self, owner: str, generation: int) -> int:
    return sum(
      entry.managed and entry.owner == owner and entry.generation == generation
      for entry in self._entries.values()
    )

  def _allocate(self, name: str, owner: str, generation: int, *, managed: bool = False) -> EventEntry:
    if name in self._entries:
      self.pmu_protocol_error_count += 1
      raise EventProtocolError(f"event {name!r} already has a live slot")
    if managed:
      quota = self._quota(owner, generation)
      if self._managed_live(owner, generation) >= quota.limit:
        self.pmu_protocol_error_count += 1
        raise EventProtocolError(
          f"event frontier for {owner!r}/{generation} exceeds reserved quota {quota.limit}"
        )
    elif self.reserved >= self.capacity:
      self.pmu_capacity_reject_count += 1
      raise EventProtocolError("event table capacity exhausted")
    entry = EventEntry(name=name, id=self._next_id, owner=owner, generation=generation, managed=managed)
    self._next_id += 1
    self._entries[name] = entry
    self.peak_active = max(self.peak_active, len(self._entries))
    self.peak_reserved = max(self.peak_reserved, self.reserved)
    if not managed:
      self.version += 1
    return entry

  def register(self, name: str, *, owner: str = "", generation: int = 0) -> EventEntry:
    """Compatibility registration for one persistent event entry."""

    entry = self._entries.get(name)
    if entry is not None:
      self._validate_identity(entry, owner, generation, allow_unspecified=True)
      return entry
    return self._allocate(name, owner, generation)

  def reserve_many(self, names: tuple[str, ...], *, owner: str, generation: int) -> bool:
    """Compatibility atomic reservation of persistent event entries."""

    if (owner, generation) in self._quotas:
      self.pmu_protocol_error_count += 1
      raise EventProtocolError("managed launches must reserve outputs just in time")
    unique = tuple(dict.fromkeys(names))
    missing: list[str] = []
    for name in unique:
      entry = self._entries.get(name)
      if entry is None:
        missing.append(name)
      else:
        self._validate_identity(entry, owner, generation)
    if self.reserved + len(missing) > self.capacity:
      self.pmu_capacity_reject_count += 1
      return False
    for name in missing:
      self._allocate(name, owner, generation)
    return True

  def reserve_quota(
    self, limit: int, event_uses: Mapping[str, int], *, owner: str, generation: int
  ) -> bool:
    """Atomically claim one launch's finite event frontier.

    This allocates no per-event entry.  ``event_uses`` is the compiler-emitted
    count of future dependency references and is the only lifetime source.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
      raise EventProtocolError("event quota must be a non-negative integer")
    key = (owner, generation)
    if key in self._quotas:
      self.pmu_protocol_error_count += 1
      raise EventProtocolError(f"event quota for {owner!r}/{generation} already exists")
    if any(entry.owner == owner and entry.generation == generation for entry in self._entries.values()):
      self.pmu_protocol_error_count += 1
      raise EventProtocolError("event quota identity already owns compatibility entries")
    declared_references = 0
    for name, count in event_uses.items():
      if not isinstance(name, str) or not name:
        raise EventProtocolError("event_uses keys must be non-empty strings")
      if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise EventProtocolError(f"event_uses[{name!r}] must be a non-negative integer")
      declared_references += count
    if self.reserved + limit > self.capacity:
      self.pmu_capacity_reject_count += 1
      return False
    self._quotas[key] = _EventQuota(owner, generation, limit, event_uses, declared_references, set())
    self.peak_reserved = max(self.peak_reserved, self.reserved)
    self.version += 1
    return True

  def cancel_reservation(self, *, owner: str, generation: int) -> None:
    """Drop an unactivated launch reservation without diagnostic retirement."""

    key = (owner, generation)
    quota = self._quotas.get(key)
    if quota is not None:
      if quota.produced or any(
        entry.managed and entry.owner == owner and entry.generation == generation
        for entry in self._entries.values()
      ):
        raise EventProtocolError("cannot cancel an event quota after output registration")
      del self._quotas[key]
      self.version += 1
      return
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

  @staticmethod
  def _unique_names(names: tuple[str, ...], kind: str) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(names))
    if len(unique) != len(names):
      raise EventProtocolError(f"duplicate {kind} event reference")
    return unique

  def _preflight_outputs(self, names: tuple[str, ...], *, owner: str, generation: int) -> tuple[str, ...]:
    names = self._unique_names(names, "output")
    quota = self._quota(owner, generation)
    for name in names:
      if name in quota.produced:
        raise EventProtocolError(f"output event {name!r} was already produced in this launch")
      if name not in quota.event_uses:
        raise EventProtocolError(f"output event {name!r} is absent from compiler event_uses")
      if name in self._entries:
        raise EventProtocolError(f"output event {name!r} already has a live slot")
    if self._managed_live(owner, generation) + len(names) > quota.limit:
      raise EventProtocolError(
        f"event outputs exceed compiler frontier quota {quota.limit} for {owner!r}/{generation}"
      )
    return names

  def preflight_action(
    self, output_names: tuple[str, ...], dependencies: tuple[str, ...], *, owner: str, generation: int
  ) -> None:
    """Validate one explicit action registration without changing state."""

    self._preflight_outputs(output_names, owner=owner, generation=generation)
    dependencies = self._unique_names(dependencies, "dependency")
    self._quota(owner, generation)
    for name in dependencies:
      entry = self._entries.get(name)
      if entry is None:
        raise EventProtocolError(
          f"dependency event {name!r} has no live slot; historical events cannot be resurrected"
        )
      self._validate_identity(entry, owner, generation)
      if not entry.managed:
        raise EventProtocolError(f"dependency event {name!r} is not launch-managed")
      if not entry.producer_bound:
        raise EventProtocolError(f"dependency event {name!r} has no bound producer")
      if entry.consumer_refs >= entry.future_uses:
        raise EventProtocolError(f"dependency event {name!r} exceeds compiler future-use count")

  def reserve_outputs(
    self, names: tuple[str, ...], *, owner: str, generation: int
  ) -> tuple[EventEntry, ...]:
    """Atomically allocate only this action's declared output slots."""

    names = self._preflight_outputs(names, owner=owner, generation=generation)
    quota = self._quota(owner, generation)
    entries = tuple(self._allocate(name, owner, generation, managed=True) for name in names)
    for entry in entries:
      entry.future_uses = quota.event_uses[entry.name]
    quota.produced.update(names)
    return entries

  def bind_producer(self, name: str, *, owner: str, generation: int, producer_id: int) -> EventEntry:
    """Bind a reserved event to exactly one registered action producer."""

    entry = self._entries.get(name)
    if entry is None:
      raise EventProtocolError(f"event {name!r} was not reserved")
    self._validate_identity(entry, owner, generation)
    if entry.status is not EventStatus.PENDING:
      raise EventProtocolError(f"event {name!r} is already terminal")
    if entry.producer_bound and entry.producer_id != producer_id:
      self.pmu_protocol_error_count += 1
      raise EventProtocolError(f"event {name!r} already has producer {entry.producer_id}")
    entry.producer_bound = True
    entry.producer_id = producer_id
    return entry

  def add_consumer(self, name: str, *, owner: str, generation: int) -> EventEntry:
    """Pin a dependency for a registered, not-yet-accepted action."""

    entry = self._entries.get(name)
    if entry is None:
      self.pmu_protocol_error_count += 1
      raise EventProtocolError(f"dependency event {name!r} has no live slot")
    self._validate_identity(entry, owner, generation)
    if not entry.producer_bound:
      self.pmu_protocol_error_count += 1
      raise EventProtocolError(f"dependency event {name!r} has no bound producer")
    if entry.managed and entry.consumer_refs >= entry.future_uses:
      self.pmu_protocol_error_count += 1
      raise EventProtocolError(f"dependency event {name!r} exceeds compiler future-use count")
    entry.consumer_refs += 1
    return entry

  def consume_dependencies(
    self, names: tuple[str, ...], *, owner: str, generation: int, cycle: int
  ) -> None:
    """Consume ready dependencies after, and only after, action acceptance."""

    names = self._unique_names(names, "dependency")
    self._quota(owner, generation)
    entries: list[EventEntry] = []
    for name in names:
      entry = self._entries.get(name)
      if entry is None:
        raise EventProtocolError(f"accepted dependency event {name!r} has no live slot")
      self._validate_identity(entry, owner, generation)
      if (
        not entry.managed
        or not entry.producer_bound
        or entry.status is not EventStatus.DONE
        or entry.consumer_refs <= 0
        or entry.future_uses <= 0
      ):
        raise EventProtocolError(f"dependency event {name!r} cannot be consumed")
      entries.append(entry)
    for entry in entries:
      entry.consumer_refs -= 1
      entry.future_uses -= 1
      self._maybe_reclaim(entry, cycle, "last dependency consumed")

  def cancel_dependencies(self, names: tuple[str, ...], *, owner: str, generation: int, cycle: int) -> None:
    """Unpin dependencies of an action that never reached acceptance."""

    names = self._unique_names(names, "dependency")
    self._quota(owner, generation)
    entries: list[EventEntry] = []
    for name in names:
      entry = self._entries.get(name)
      if entry is None:
        raise EventProtocolError(f"cancelled dependency event {name!r} has no live slot")
      self._validate_identity(entry, owner, generation)
      if not entry.managed or entry.consumer_refs <= 0 or entry.future_uses <= 0:
        raise EventProtocolError(f"dependency event {name!r} has no registered consumer")
      entries.append(entry)
    for entry in entries:
      entry.consumer_refs -= 1
      entry.future_uses -= 1
      self._maybe_reclaim(entry, cycle, "registered consumer cancelled")

  def abandon_future_uses(self, *, owner: str, generation: int, cycle: int) -> None:
    """Abandon all unregistered references after a Context is closed by fault."""

    self._quota(owner, generation)
    live = [
      entry
      for entry in self._entries.values()
      if entry.managed and entry.owner == owner and entry.generation == generation
    ]
    if any(entry.consumer_refs for entry in live):
      raise EventProtocolError("cannot abandon future uses while registered consumers remain")
    for entry in tuple(live):
      entry.future_uses = 0
      self._maybe_reclaim(entry, cycle, "future references abandoned")

  def cancel_outputs(
    self, names: tuple[str, ...], *, owner: str, generation: int, producer_id: int, cycle: int
  ) -> None:
    """Retire outputs of a registered action which was never accepted."""

    names = self._unique_names(names, "output")
    entries: list[EventEntry] = []
    for name in names:
      entry = self._entries.get(name)
      if entry is None:
        raise EventProtocolError(f"cancelled output event {name!r} has no live slot")
      self._validate_identity(entry, owner, generation)
      if not entry.managed or not entry.producer_bound or entry.producer_id != producer_id:
        raise EventProtocolError(f"cancelled output event {name!r} has a foreign producer")
      entries.append(entry)
    for entry in entries:
      if entry.status is EventStatus.PENDING:
        entry.status = EventStatus.RESET
        entry.timestamp = cycle
      self._maybe_reclaim(entry, cycle, "producer cancelled before acceptance")

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

    Missing events are auto-registered only for the standalone ABI form with
    no expected identity.  Managed completions can therefore never resurrect a
    reclaimed historical name.
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
    self._maybe_reclaim(entry, cycle, "terminal event has no remaining references")
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
    if entry.managed:
      raise EventProtocolError("managed event sequences are fixed for one launch")
    entry.sequence += 1
    entry.status = EventStatus.PENDING
    entry.producer_bound = False
    entry.consumer_refs = 0
    return entry.sequence

  def _maybe_reclaim(self, entry: EventEntry, cycle: int, reason: str) -> bool:
    if not entry.managed or entry.status is EventStatus.PENDING or entry.consumer_refs or entry.future_uses:
      return False
    self._retire_entry(entry, cycle, reason)
    return True

  def _retire_entry(self, entry: EventEntry, cycle: int, reason: str) -> None:
    if self._entries.get(entry.name) is not entry:
      raise EventProtocolError(f"event {entry.name!r} is not the live instance")
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
        "reason": reason,
      }
    )
    del self._entries[entry.name]
    if not entry.managed:
      self.version += 1

  def release_owner(self, *, owner: str, generation: int, cycle: int) -> int:
    """Release a retired launch quota after all event work is terminal."""

    key = (owner, generation)
    quota = self._quotas.get(key)
    entries = [
      entry for entry in self._entries.values() if entry.owner == owner and entry.generation == generation
    ]
    if quota is not None:
      if any(entry.status is EventStatus.PENDING for entry in entries):
        raise EventProtocolError("cannot release event quota with pending producers")
      if any(entry.consumer_refs for entry in entries):
        raise EventProtocolError("cannot release event quota with registered consumers")
      if any(entry.future_uses for entry in entries):
        raise EventProtocolError("cannot release event quota with future dependency references")
      for entry in tuple(entries):
        self._retire_entry(entry, cycle, "owner retired")
      del self._quotas[key]
      self.version += 1
      return len(entries)
    for entry in tuple(entries):
      self._retire_entry(entry, cycle, "compatibility owner retired")
    return len(entries)

  def reset(self) -> None:
    """Reset marks pending live events RESET; terminal failures stay terminal."""

    for entry in self._entries.values():
      if entry.status is EventStatus.PENDING:
        entry.status = EventStatus.RESET

  def clear(self) -> None:
    self._entries.clear()
    self._quotas.clear()
    self._retired.clear()
    self._next_id = 0
    self.peak_reserved = 0
    self.peak_active = 0
    self.pmu_stale_sequence_count = 0
    self.pmu_wrong_owner_count = 0
    self.pmu_duplicate_signal_count = 0
    self.pmu_capacity_reject_count = 0
    self.pmu_protocol_error_count = 0
    self.version += 1

  def snapshot(self) -> dict:
    return {
      "capacity": self.capacity,
      "reserved": self.reserved,
      "quota_reserved": sum(quota.limit for quota in self._quotas.values()),
      "active": len(self._entries),
      "peak_reserved": self.peak_reserved,
      "peak_active": self.peak_active,
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
          "future_uses": entry.future_uses,
          "managed": entry.managed,
        }
        for name, entry in self._entries.items()
      },
      "quotas": {
        f"{owner}@{generation}": {
          "owner": owner,
          "generation": generation,
          "limit": quota.limit,
          "active": self._managed_live(owner, generation),
          "declared_events": len(quota.event_uses),
          "declared_references": quota.declared_references,
          "produced_events": len(quota.produced),
          "remaining_references": sum(
            entry.future_uses
            for entry in self._entries.values()
            if entry.managed and entry.owner == owner and entry.generation == generation
          ),
        }
        for (owner, generation), quota in self._quotas.items()
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
