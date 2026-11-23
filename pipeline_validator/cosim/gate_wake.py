"""Event-driven profile gate wake registry and GATE_NOTIFY broadcast (C036 / W1-04).

Ownership boundary:
- ``ProfileController`` is the sole writer of ``_profile_gates`` and calls
  :meth:`GateWakeCoordinator.register` inside the same step where a gate flips
  (``initialize`` / ``CLOSE_ISSUE`` / ``OPEN_ISSUE`` / ``_fault`` / ``recover``).
- ``GateWakeCoordinator`` maintains monotonic per-level gate epochs, wakes
  gate-parked scheduler actions (Route registration) and ``AdmissionCoordinator``
  (Task submission), and queues/flushes ``GATE_NOTIFY`` (G2T, CTRL) to every
  attached boundary ``TilePort`` without per-tick polling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .tile_port import BoundaryMessage

if TYPE_CHECKING:
  from ..tile_group import TileGroup

_GATE_LEVELS = ("l1", "l2")


@dataclass(frozen=True)
class GateStateRecord:
  """One authoritative gate state snapshot for a profile level."""

  level: str
  closed: bool
  profile_generation: int
  epoch: int
  cycle: int


class GateWakeCoordinator:
  """Single source of truth for profile gate epochs, wakeups, and ``GATE_NOTIFY``."""

  def __init__(self, group: TileGroup) -> None:
    self.group = group
    self._closed: dict[str, bool] = {"l1": True, "l2": True}
    self._epochs: dict[str, int] = {"l1": 0, "l2": 0}
    self._generations: dict[str, int] = {"l1": 0, "l2": 0}
    # Pending GATE_NOTIFY payload per (tile_id, level). Keeping only the latest
    # state per (tile_id, level) bounds CTRL traffic and coalesces rapid flips
    # while a CTRL link is backpressured.
    self._pending_notify: dict[tuple[int, str], GateStateRecord] = {}
    self.stats: dict[str, int] = {
      "transitions": 0,
      "open_events": 0,
      "close_events": 0,
      "scheduler_wakes": 0,
      "admission_wakes": 0,
      "gate_notifies_queued": 0,
      "gate_notifies_sent": 0,
      "gate_notifies_coalesced": 0,
    }

  def epoch(self, level: str) -> int:
    if level not in _GATE_LEVELS:
      raise ValueError(f"unknown gate level {level!r}")
    return self._epochs[level]

  @property
  def gate_epoch(self) -> tuple[int, int]:
    """Unified monotonic gate epoch tuple ``(l1_epoch, l2_epoch)`` for wake tokens."""
    return (self._epochs["l1"], self._epochs["l2"])

  def is_closed(self, level: str) -> bool:
    if level not in _GATE_LEVELS:
      raise ValueError(f"unknown gate level {level!r}")
    return self._closed[level]

  def register(
    self,
    level: str,
    *,
    closed: bool | None = None,
    profile_generation: int | None = None,
    cycle: int = 0,
    notify: bool = True,
  ) -> bool:
    """Record a profile gate state change in-step, wake parked paths, and queue GATE_NOTIFY.

    Returns ``True`` iff the effective gate state (``closed`` or ``profile_generation``
    on an open/close transition) changed and advanced ``epoch(level)``.
    """
    if level not in _GATE_LEVELS:
      raise ValueError(f"unknown gate level {level!r}")
    controller = self.group.profile_controller
    new_closed = controller.issue_gate_closed(level) if closed is None else bool(closed)
    new_gen = int(controller.generations[level]) if profile_generation is None else int(profile_generation)
    if new_gen < 0:
      raise ValueError(f"profile_generation must be non-negative, got {new_gen}")

    prev_closed = self._closed[level]
    prev_gen = self._generations[level]
    if new_closed == prev_closed and new_gen == prev_gen:
      return False

    self._closed[level] = new_closed
    self._generations[level] = new_gen
    self._epochs[level] += 1
    epoch = self._epochs[level]
    self.stats["transitions"] += 1
    if new_closed:
      self.stats["close_events"] += 1
    else:
      self.stats["open_events"] += 1

    record = GateStateRecord(
      level=level, closed=new_closed, profile_generation=new_gen, epoch=epoch, cycle=cycle
    )

    if not new_closed:
      woken = self.group.scheduler.wake_gate_waiters(level, epoch, cycle)
      if woken:
        self.stats["scheduler_wakes"] += woken
      if level == "l1":
        if self.group.admission.wake_on_l1_gate(epoch, cycle):
          self.stats["admission_wakes"] += 1

    if notify:
      self._queue_notify(record)
      self.flush(cycle)
    self._trace("gate_transition", record)
    return True

  def _queue_notify(self, record: GateStateRecord) -> None:
    if not self.group.boundary_ports:
      return
    for tile_id in self.group.boundary_ports:
      key = (tile_id, record.level)
      if key in self._pending_notify:
        self.stats["gate_notifies_coalesced"] += 1
      self._pending_notify[key] = record
      self.stats["gate_notifies_queued"] += 1

  def flush(self, cycle: int) -> int:
    """Transmit pending ``GATE_NOTIFY`` messages on ready boundary CTRL links."""
    if not self._pending_notify or not self.group.boundary_ports:
      return 0
    sent = 0
    for key in tuple(self._pending_notify):
      tile_id, _level = key
      port = self.group.boundary_ports.get(tile_id)
      if port is None:
        del self._pending_notify[key]
        continue
      if not port.outbox_ready("CTRL"):
        continue
      record = self._pending_notify.pop(key)
      msg = BoundaryMessage(
        message_type="GATE_NOTIFY",
        channel="CTRL",
        direction="G2T",
        fields={
          "level": record.level,
          "state": "closed" if record.closed else "open",
          "profile_generation": record.profile_generation,
        },
      )
      port.send(msg)
      sent += 1
      self.stats["gate_notifies_sent"] += 1
      self._trace("gate_notify_sent", record, tile_id=tile_id, send_cycle=cycle)
    return sent

  def clear_pending_notifies(self) -> None:
    """Drop unsent notifications across an explicit run/reset boundary."""
    self._pending_notify.clear()

  def snapshot(self) -> dict[str, object]:
    return {
      "closed": dict(self._closed),
      "epochs": dict(self._epochs),
      "generations": dict(self._generations),
      "pending_notifies": len(self._pending_notify),
      "stats": dict(self.stats),
    }

  def _trace(self, name: str, record: GateStateRecord, **extra: object) -> None:
    # Keep pure-Python baseline traces identical to C022; emit Cosim:GateWake
    # instrumentation only when at least one external boundary TilePort is attached.
    tracer = self.group.tracer
    if tracer is None or not self.group.boundary_ports:
      return
    tracer.instant(
      "TileGroup",
      "Cosim:GateWake",
      name,
      record.cycle,
      {
        "level": record.level,
        "state": "closed" if record.closed else "open",
        "profile_generation": record.profile_generation,
        "epoch": record.epoch,
        **extra,
      },
    )
