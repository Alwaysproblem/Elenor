"""Event Table with event_id + sequence (Runtime ABI 3.2, review P0-4).

The simulator keeps the existing *symbolic string event names* as the public
IR (TileInst.dst, GroupAction.dst, trace args["event_id"] are all strings).
This EventTable is an overlay used in runtime/full_memory fidelity: it maps
each symbolic event name to a (numeric id, sequence) pair and enforces the
P0-4 rule that wait must match expected_sequence, not just status.

This keeps V1 builders / tests / traces unchanged while adding the
sequence/generation semantics that prevent stale-completion reuse after
event table wrap or reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EventStatus(Enum):
  """elenor_event_status_v0_t (Runtime ABI 4.3).

  Plain Enum (not IntEnum) so ERROR/TIMEOUT/RESET are not truthy —
  callers must explicitly compare `status is EventStatus.DONE` and
  route other terminal states into the fault/reset path.
  """
  PENDING = 0
  DONE = 1
  ERROR = 2
  TIMEOUT = 3
  RESET = 4


@dataclass
class EventEntry:
  """One event table entry keyed by symbolic name.

  `name` is the existing string event id (e.g. "ev_role0").
  `id` is the numeric event_id (Runtime ABI 4.3 layout).
  `sequence` is the reuse-generation counter (P0-4).
  """
  name: str
  id: int
  sequence: int = 0
  status: EventStatus = EventStatus.PENDING
  producer_id: int = 0
  timestamp: int = 0
  error_code: int = 0
  expected_sequence: int = 0  # set by wait(); used to reject stale signals


class EventTable:
  """Event table overlay: symbolic name -> (id, sequence, status).

  In runtime fidelity, every signal/wait goes through this table so that
  sequence mismatches are detected.  In timing_only fidelity the table is
  bypassed and the existing string-set mechanism is used directly.
  """

  def __init__(self) -> None:
    self._entries: dict[str, EventEntry] = {}
    self._next_id: int = 0
    self.pmu_stale_sequence_count: int = 0

  def register(self, name: str) -> EventEntry:
    """Register a symbolic event name, assigning a numeric id."""
    if name in self._entries:
      return self._entries[name]
    e = EventEntry(name=name, id=self._next_id)
    self._next_id += 1
    self._entries[name] = e
    return e

  def get(self, name: str) -> EventEntry | None:
    return self._entries.get(name)

  def signal(self, name: str, status: EventStatus, producer_id: int,
             cycle: int, error_code: int = 0) -> bool:
    """Signal an event.  Returns False if the signal is stale (sequence
    mismatch) — this is the P0-4 stale-completion rejection path.

    If the event was not pre-registered (timing_only callers that bypass
    register), it is auto-registered so signal still succeeds.
    """
    e = self._entries.get(name)
    if e is None:
      e = self.register(name)
    if e.expected_sequence != 0 and e.sequence != e.expected_sequence:
      # stale completion: the producer signalled an old sequence
      self.pmu_stale_sequence_count += 1
      return False
    e.status = status
    e.producer_id = producer_id
    e.timestamp = cycle
    e.error_code = error_code
    return True

  def wait(self, name: str,
           expected_sequence: int | None = None) -> EventStatus | None:
    """Check whether an event is satisfied with the expected sequence.

    Returns:
      None           — not yet satisfied (still pending, keep waiting).
      EventStatus.X  — satisfied; caller must distinguish DONE from
                       ERROR/TIMEOUT/RESET and route the latter into the
                       fault/reset path.  Do NOT treat non-None as success.

    If expected_sequence is None, only status is checked (V1-compatible).
    Otherwise the entry's sequence must match (P0-4).
    """
    e = self._entries.get(name)
    if e is None:
      return None
    if expected_sequence is not None:
      e.expected_sequence = expected_sequence
      if e.sequence != expected_sequence:
        return None
    if e.status == EventStatus.PENDING:
      return None
    return e.status  # DONE / ERROR / TIMEOUT / RESET — caller must check

  def advance_sequence(self, name: str) -> int:
    """Advance the sequence counter for an event (on reuse/reset)."""
    e = self._entries.get(name)
    if e is None:
      e = self.register(name)
    e.sequence += 1
    return e.sequence

  def reset(self) -> None:
    """Runtime ABI 3.2: reset marks pending events as RESET, never silent."""
    for e in self._entries.values():
      if e.status == EventStatus.PENDING:
        e.status = EventStatus.RESET

  def clear(self) -> None:
    self._entries.clear()
    self._next_id = 0
    self.pmu_stale_sequence_count = 0

  def snapshot(self) -> dict:
    return {
      n: {"id": e.id, "seq": e.sequence,
          "status": e.status.name, "producer": e.producer_id}
      for n, e in self._entries.items()
    }
