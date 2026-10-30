"""Bounded BOA result/AM service with explicit L1-output backpressure.

Compute completion is not the UCE completion event.  A prepared result first
occupies a finite result FIFO and O/R credits, then takes the AM commit stage,
then (for FINAL) waits for an L1 output credit, writes/drains the output, and
only then releases the AM lease and signals the event.  If the L1 output port
is blocked, the FINAL remains live in ``WAIT_OUTPUT``/``DRAINING`` -- the AM
slot and quota are not returned early, and no UCE polling is required for the
event to fire after recovery.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..config import HardwareConfig
from ..execution_ir import ExecBoaMatmulDesc
from ..trace import Tracer


def _optional_cycle(value: int | None) -> int:
  return -1 if value is None else value


class BoaResultState(Enum):
  WAIT_COMMIT = "wait_commit"
  WAIT_OUTPUT = "wait_output"
  DRAINING = "draining"


@dataclass(frozen=True)
class BoaPreparedResult:
  """Compute-done payload plus the AM/output transitions it must perform."""

  matmul: ExecBoaMatmulDesc
  partial: bytes
  final: bool
  commit: Callable[[int], None]
  commit_final: Callable[[int], None]
  write_output: Callable[[int], None]
  release: Callable[[int], None]
  abort: Callable[[int, str], None]


@dataclass
class _PendingResult:
  job: Any
  prepared: BoaPreparedResult
  enqueued_cycle: int
  commit_due_cycle: int
  state: BoaResultState
  drain_end_cycle: int | None = None
  claimed_output_credit: bool = False


@dataclass(frozen=True)
class BoaResultRecord:
  event_id: str
  am: str
  final: bool
  compute_done_cycle: int
  commit_cycle: int
  output_cycle: int | None
  event_cycle: int
  status: str
  engine_id: int | None = None


class BoaResultService:
  """Finite result FIFO, AM commit and L1-output drain service."""

  def __init__(self, cfg: HardwareConfig, *, tile_id: int = 0, tracer: Tracer | None = None):
    self.cfg = cfg
    self.tile_id = tile_id
    self.tracer = tracer
    self.fifo_depth = cfg.boa_result_fifo_depth
    self.commit_cycles = cfg.boa_commit_cycles
    self.drain_cycles = cfg.boa_output_drain_cycles
    self.credit_capacity = cfg.boa_am_credit_capacity
    self.output_credit_capacity = cfg.boa_l1_output_credits
    self._credits = self.credit_capacity
    self._reservations = 0
    self._output_in_use = 0
    self._output_blocked = False
    self._output_block_until_cycle: int | None = None
    self._pending: deque[_PendingResult] = deque()
    self._failures: list[tuple[str, str]] = []
    self._records: list[BoaResultRecord] = []
    self._record_capacity = 64
    self.peak_fifo = 0
    self.enqueued_count = 0
    self.commit_count = 0
    self.output_write_count = 0
    self.release_count = 0
    self.event_count = 0
    self.blocked_cycles = 0
    self.overflow_count = 0

  # -- capacity / backpressure ---------------------------------------

  @property
  def credits(self) -> int:
    return self._credits

  @property
  def pending_count(self) -> int:
    return len(self._pending)

  def has_capacity(self) -> bool:
    return len(self._pending) + self._reservations < self.fifo_depth and self._credits >= 2

  def reserve(self) -> bool:
    """Atomically reserve one result slot and O/R credits at launch."""
    if not self.has_capacity():
      return False
    self._credits -= 2
    self._reservations += 1
    return True

  def cancel_reservation(self) -> None:
    if self._reservations <= 0:
      raise ValueError("no BOA result reservation to cancel")
    self._reservations -= 1
    self._credits += 2

  def set_output_blocked(self, blocked: bool, *, until_cycle: int | None = None) -> None:
    """Inject L1 output backpressure; ``until_cycle`` models downstream recovery."""
    self._output_blocked = blocked
    self._output_block_until_cycle = until_cycle if blocked else None

  def pop_failures(self) -> tuple[tuple[str, str], ...]:
    failures = tuple(self._failures)
    self._failures.clear()
    return failures

  # -- lifecycle -----------------------------------------------------

  def submit(
    self, job: Any, prepared: BoaPreparedResult, cycle: int, *, reserved: bool = False
  ) -> None:
    if reserved:
      if self._reservations <= 0:
        self.overflow_count += 1
        raise ValueError("BOA result reservation is missing")
      self._reservations -= 1
    else:
      if not self.has_capacity():
        self.overflow_count += 1
        raise ValueError("BOA result FIFO or O/R credit capacity exhausted")
      self._credits -= 2
    self.enqueued_count += 1
    self._pending.append(
      _PendingResult(
        job,
        prepared,
        cycle,
        cycle + self.commit_cycles,
        BoaResultState.WAIT_COMMIT,
      )
    )
    self.peak_fifo = max(self.peak_fifo, len(self._pending))
    self._trace(
      cycle,
      "boa_result_enqueue",
      {
        "event_id": getattr(job, "event_id", ""),
        "engine_id": getattr(job, "engine_id", None),
        "am": prepared.matmul.am,
        "final": prepared.final,
        "compute_done": getattr(job, "compute_done_cycle", cycle),
        "fifo": len(self._pending),
        "credits": self._credits,
      },
    )

  def advance(self, cycle: int) -> list[Any]:
    """Advance one cycle; return jobs whose UCE event is now ready."""
    if (
      self._output_blocked
      and self._output_block_until_cycle is not None
      and cycle >= self._output_block_until_cycle
    ):
      self._output_blocked = False
      self._output_block_until_cycle = None
    ready: list[Any] = []
    for pending in list(self._pending):
      try:
        if pending.state is BoaResultState.WAIT_COMMIT:
          if cycle < pending.commit_due_cycle:
            continue
          if not pending.prepared.final:
            pending.prepared.commit(cycle)
            self.commit_count += 1
            pending.job.commit_cycle = cycle
            self._credits += 2
            self._finish(pending, cycle, "step_committed")
            ready.append(pending.job)
          else:
            pending.state = BoaResultState.WAIT_OUTPUT
            self._trace(
              cycle,
              "boa_result_wait_output",
              {"event_id": getattr(pending.job, "event_id", ""), "am": pending.prepared.matmul.am,
               "engine_id": getattr(pending.job, "engine_id", None)},
            )
        if pending.state is BoaResultState.WAIT_OUTPUT:
          if self._output_blocked or self._output_in_use >= self.output_credit_capacity:
            self.blocked_cycles += 1
            continue
          self._output_in_use += 1
          pending.claimed_output_credit = True
          pending.prepared.write_output(cycle)
          pending.job.output_cycle = cycle
          pending.prepared.commit_final(cycle)
          self.output_write_count += 1
          self.commit_count += 1
          pending.job.commit_cycle = cycle
          pending.state = BoaResultState.DRAINING
          pending.drain_end_cycle = cycle + self.drain_cycles
          self._trace(
            cycle,
            "boa_result_output_written",
            {
              "event_id": getattr(pending.job, "event_id", ""),
              "engine_id": getattr(pending.job, "engine_id", None),
              "am": pending.prepared.matmul.am,
              "output_cycle": cycle,
              "drain_end": pending.drain_end_cycle,
            },
          )
        if pending.state is BoaResultState.DRAINING:
          if cycle < (pending.drain_end_cycle or cycle):
            continue
          pending.prepared.release(cycle)
          self.release_count += 1
          if pending.claimed_output_credit:
            self._output_in_use -= 1
          self._credits += 2
          self._finish(pending, cycle, "chain_done")
          ready.append(pending.job)
      except Exception as exc:
        self._fail_pending(pending, cycle, str(exc))
    return ready

  def _finish(self, pending: _PendingResult, cycle: int, status: str) -> None:
    self._pending.remove(pending)
    pending.job.event_cycle = cycle
    self.event_count += 1
    self._records.append(
      BoaResultRecord(
        getattr(pending.job, "event_id", ""),
        pending.prepared.matmul.am,
        pending.prepared.final,
        _optional_cycle(getattr(pending.job, "compute_done_cycle", None)),
        _optional_cycle(getattr(pending.job, "commit_cycle", None)),
        getattr(pending.job, "output_cycle", None),
        cycle,
        status,
        getattr(pending.job, "engine_id", None),
      )
    )
    if len(self._records) > self._record_capacity:
      del self._records[: len(self._records) - self._record_capacity]
    self._trace(
      cycle,
      "boa_result_ready",
      {
        "event_id": getattr(pending.job, "event_id", ""),
        "engine_id": getattr(pending.job, "engine_id", None),
        "am": pending.prepared.matmul.am,
        "status": status,
        "event_cycle": cycle,
      },
    )

  def _fail_pending(self, pending: _PendingResult, cycle: int, reason: str) -> None:
    if pending not in self._pending:
      return
    self._pending.remove(pending)
    if pending.claimed_output_credit:
      self._output_in_use -= 1
    try:
      pending.prepared.abort(cycle, reason)
    except Exception:
      pass
    self._credits += 2
    self._failures.append((getattr(pending.job, "event_id", ""), reason))
    self._records.append(
      BoaResultRecord(
        getattr(pending.job, "event_id", ""),
        pending.prepared.matmul.am,
        pending.prepared.final,
        _optional_cycle(getattr(pending.job, "compute_done_cycle", None)),
        -1,
        None,
        cycle,
        "fault",
        getattr(pending.job, "engine_id", None),
      )
    )

  def reset(self, cycle: int = 0) -> None:
    for pending in tuple(self._pending):
      try:
        pending.prepared.abort(cycle, "reset")
      except Exception:
        pass
    self._pending.clear()
    self._failures.clear()
    self._credits = self.credit_capacity
    self._reservations = 0
    self._reservations = 0
    self._output_in_use = 0
    self._output_blocked = False
    self._output_block_until_cycle = None

  def _trace(self, cycle: int, name: str, args: dict) -> None:
    if self.tracer is None:
      return
    self.tracer.instant(f"Tile{self.tile_id}", "BoaResult", name, cycle, {"tile_id": self.tile_id, **args})

  def snapshot(self) -> dict:
    states = {state.value: 0 for state in BoaResultState}
    for pending in self._pending:
      states[pending.state.value] += 1
    return {
      "fifo_depth": self.fifo_depth,
      "pending": len(self._pending),
      "reservations": self._reservations,
      "states": states,
      "credits": self._credits,
      "credit_capacity": self.credit_capacity,
      "output_in_use": self._output_in_use,
      "output_credit_capacity": self.output_credit_capacity,
      "output_blocked": self._output_blocked,
      "output_block_until_cycle": self._output_block_until_cycle,
      "peak_fifo": self.peak_fifo,
      "enqueued_count": self.enqueued_count,
      "commit_count": self.commit_count,
      "output_write_count": self.output_write_count,
      "release_count": self.release_count,
      "event_count": self.event_count,
      "blocked_cycles": self.blocked_cycles,
      "overflow_count": self.overflow_count,
      "records": [
        {
          "event_id": record.event_id,
          "engine_id": record.engine_id,
          "am": record.am,
          "final": record.final,
          "compute_done_cycle": record.compute_done_cycle,
          "commit_cycle": record.commit_cycle,
          "output_cycle": record.output_cycle,
          "event_cycle": record.event_cycle,
          "status": record.status,
        }
        for record in self._records
      ],
    }
