"""Reference consumer for the W6 event / terminal / reset contracts (C015).

This is a CONTRACT CONSUMER, not W6 RTL and not a simulator: it is the smallest state machine that follows
`protocol/w6_event_facility.contract.yaml` and `protocol/w6_terminal_reset.contract.yaml`, so the contract
can be exercised with real event, wait, reset, cancel, fault and debug sequences. A test that deletes a
rule from the YAML changes what this module accepts, because every rule below is read from the YAML.

What it is NOT: it proves no RTL behaviour. C033/C034 still have to show concurrency and backpressure on
real hardware; a pass here is a contract-level result only.

Every capacity is a caller-supplied model parameter. The contracts leave them `unresolved`, and nothing in
this file turns an unresolved value into a default.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
H = LIB.parents[1]
FACILITY = H / "protocol" / "w6_event_facility.contract.yaml"
TERMINAL = H / "protocol" / "w6_terminal_reset.contract.yaml"
RANK = {"PENDING": 0, "DONE": 1, "ERROR": 2, "TIMEOUT": 3, "RESET": 4}


class W6Error(Exception):
  """Coded rejection; `code` is stable so tests can pin it."""

  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}" + (f": {detail}" if detail else ""))
    self.code = code


def load(path: Path) -> dict[str, Any]:
  return yaml.safe_load(path.read_text())  # type: ignore[no-any-return]


def load_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
  return load(FACILITY), load(TERMINAL)


def _count(value: Any, name: str, *, minimum: int = 0) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
    raise W6Error("BAD_PARAMETER", f"{name} must be an integer >= {minimum}, got {value!r}")
  return value


# ---------------------------------------------------------------------------------------------------
# Wait-for graphs: used to prove a contract has no circular wait.
# ---------------------------------------------------------------------------------------------------
def find_cycle(edges: dict[str, list[str]]) -> list[str] | None:
  """Return one cycle in a `node -> nodes it waits on` graph, or None."""
  state: dict[str, int] = {}
  stack: list[str] = []

  def visit(node: str) -> list[str] | None:
    state[node] = 1
    stack.append(node)
    for nxt in edges.get(node, []):
      if state.get(nxt, 0) == 1:
        return [*stack[stack.index(nxt) :], nxt]
      if state.get(nxt, 0) == 0:
        found = visit(nxt)
        if found:
          return found
    stack.pop()
    state[node] = 2
    return None

  for start in list(edges):
    if state.get(start, 0) == 0:
      found = visit(start)
      if found:
        return found
  return None


def reaches(edges: dict[str, list[str]], start: str, goal: str) -> bool:
  seen: set[str] = set()
  todo = [start]
  while todo:
    node = todo.pop()
    if node == goal:
      return True
    if node in seen:
      continue
    seen.add(node)
    todo.extend(edges.get(node, []))
  return False


# ---------------------------------------------------------------------------------------------------
# Local event table + WAIT/WAITALL
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EventKey:
  tile_id: int
  ctx_slot: int
  ctx_generation: int
  local_event_id: int
  event_generation: int
  sequence: int
  scope: str


@dataclass
class Entry:
  key: EventKey
  kind: str
  producer_id: int
  status: str = "PENDING"
  error_code: int = 0
  consumer_refs: int = 0
  future_uses: int = 0


@dataclass
class WaitRecord:
  ctx: tuple[int, int, int]
  kind: str
  operands: list[EventKey]
  armed: bool = True


class EventFacility:
  """Event table: per-Context frontier slots, a separate terminal reserve, one wait record per Context."""

  def __init__(
    self,
    contract: dict[str, Any],
    *,
    contexts: int,
    frontier: int,
    terminal_reserve: int,
    tile_slots: int,
    response_inflight: int,
    waitall_max: int,
  ) -> None:
    self.c = contract
    self.contexts = _count(contexts, "contexts", minimum=1)
    self.frontier = _count(frontier, "frontier", minimum=1)
    self.reserve = _count(
      terminal_reserve,
      "terminal_reserve",
      minimum=contract["capacity"]["terminal_reserve_per_context"]["minimum"],
    )
    self.tile_slots = _count(tile_slots, "tile_slots", minimum=1)
    self.waitall_max = _count(waitall_max, "waitall_max", minimum=2)
    self.response_inflight = _count(response_inflight, "response_inflight")
    # the budget expression of the contract, every term in entries
    if self.contexts * (self.frontier + self.reserve) + self.response_inflight > self.tile_slots:
      raise W6Error("BUDGET_EXCEEDED", "contexts*(frontier+reserve)+response_inflight > tile_slots")
    self.entries: dict[tuple[int, int, int], Entry] = {}
    self.terminal_records: dict[tuple[int, int], list[dict[str, Any]]] = {}
    self.waits: dict[tuple[int, int, int], WaitRecord] = {}
    self.wakes: list[tuple[tuple[int, int, int], str]] = []
    self.counters = dict.fromkeys(contract["counters"], 0)
    self.sticky = dict.fromkeys(contract["counters"], False)

  # ---- counters ----
  def _bump(self, name: str) -> None:
    self.counters[name] += 1
    self.sticky[name] = True

  # ---- registration ----
  def _live(self, tile_id: int, ctx_slot: int) -> int:
    return sum(1 for (t, s, _), _e in self.entries.items() if (t, s) == (tile_id, ctx_slot))

  def register(self, key: EventKey, kind: str, producer_id: int, *, future_uses: int = 0) -> bool:
    """Allocate one entry. A full Context frontier is backpressure (False), never a fault."""
    ev = self.c["event_kinds"]
    if kind in {k["name"] for k in ev["not_local_events"]}:
      raise W6Error("NOT_A_LOCAL_EVENT", kind)
    if kind in ev["not_waitable"]:
      raise W6Error("NOT_WAITABLE", kind)
    if kind not in ev["local_waitable"]:
      raise W6Error("UNKNOWN_KIND", kind)
    if key.scope not in self.c["identity"]["scopes"]:
      raise W6Error("UNKNOWN_SCOPE", key.scope)
    slot = (key.tile_id, key.ctx_slot, key.local_event_id)
    if slot in self.entries:
      self._bump("protocol_error")
      raise W6Error("SLOT_IN_USE", f"{slot}")
    if self._live(key.tile_id, key.ctx_slot) >= self.frontier:
      self._bump("capacity_reject")
      return False
    self.entries[slot] = Entry(key, kind, producer_id, future_uses=_count(future_uses, "future_uses"))
    return True

  def add_consumer(self, key: EventKey) -> None:
    entry = self._resolve(key)
    entry.consumer_refs += 1

  def consume(self, key: EventKey) -> None:
    entry = self._resolve(key)
    if entry.consumer_refs <= 0 or entry.future_uses <= 0 or entry.status != "DONE":
      raise W6Error("CANNOT_CONSUME", f"{key.local_event_id}")
    entry.consumer_refs -= 1
    entry.future_uses -= 1
    self._reclaim(entry)

  def abandon_future_uses(self, tile_id: int, ctx_slot: int, ctx_generation: int) -> None:
    live = [
      e
      for e in self.entries.values()
      if (e.key.tile_id, e.key.ctx_slot, e.key.ctx_generation) == (tile_id, ctx_slot, ctx_generation)
    ]
    if any(e.consumer_refs for e in live):
      raise W6Error("ABANDON_WITH_CONSUMERS")
    for entry in live:
      entry.future_uses = 0
      self._reclaim(entry)

  def _reclaim(self, entry: Entry) -> bool:
    # predicate from the contract: terminal and no consumer refs and no future uses
    slot = (entry.key.tile_id, entry.key.ctx_slot, entry.key.local_event_id)
    if self.entries.get(slot) is not entry:
      return False  # already reclaimed through another path
    if entry.status == "PENDING" or entry.consumer_refs or entry.future_uses:
      return False
    del self.entries[slot]
    return True

  def _resolve(self, key: EventKey) -> Entry:
    entry = self.entries.get((key.tile_id, key.ctx_slot, key.local_event_id))
    if entry is None or entry.key != key:
      raise W6Error("UNRESOLVED_OPERAND", f"{key}")
    return entry

  # ---- signal ----
  def signal(self, key: EventKey, status: str, producer_id: int, error_code: int = 0) -> bool:
    """One completion changes at most one entry; a rejected one changes counters only."""
    if status not in self.c["states"]["terminal"]:
      raise W6Error("NOT_TERMINAL", status)
    entry = self.entries.get((key.tile_id, key.ctx_slot, key.local_event_id))
    if entry is None:
      self._bump("unknown_event_notify")
      return False
    live = entry.key
    if (live.ctx_generation, live.scope) != (key.ctx_generation, key.scope):
      self._bump("wrong_owner")
      return False
    if (live.event_generation, live.sequence) != (key.event_generation, key.sequence):
      self._bump("stale_sequence")
      return False
    if producer_id != entry.producer_id:
      self._bump("wrong_owner")
      return False
    if entry.status != "PENDING":
      self._bump("duplicate_signal")
      return False
    entry.status = status
    entry.error_code = error_code
    self._evaluate_waits()
    self._reclaim(entry)
    return True

  # ---- WAIT / WAITALL ----
  def _verdict(self, record: WaitRecord) -> str | None:
    states = []
    for op in record.operands:
      # an armed record pins every operand, so a live record can never see a missing entry
      entry = self.entries.get((op.tile_id, op.ctx_slot, op.local_event_id))
      if entry is None:
        raise W6Error("PINNED_OPERAND_MISSING", f"{op}")
      states.append(entry.status)
    if any(s not in ("PENDING", "DONE") for s in states):
      return "FAIL_EXIT"
    if all(s == "DONE" for s in states):
      return "PROCEED"
    return None

  def _pin(self, operands: list[EventKey]) -> None:
    """The wait is a registered consumer of each operand; the compiler's future-use count must cover it."""
    entries = [self._resolve(op) for op in operands]
    for entry in entries:
      if entry.consumer_refs >= entry.future_uses:
        self._bump("protocol_error")
        raise W6Error("WAIT_EXCEEDS_FUTURE_USES", f"{entry.key.local_event_id}")
    for entry in entries:
      entry.consumer_refs += 1

  def _settle(self, record: WaitRecord, verdict: str) -> None:
    """Release the pins: a success consumes each operand, a failure cancels the registered consumer."""
    for op in record.operands:
      entry = self._resolve(op)
      entry.consumer_refs -= 1
      entry.future_uses -= 1
      self._reclaim(entry)
    self.wakes.append((record.ctx, "WAKE" if verdict == "PROCEED" else "FAIL_EXIT"))

  def issue_wait(self, ctx: tuple[int, int, int], kind: str, operands: list[EventKey]) -> str:
    """Atomic check-register: PROCEED, FAIL_EXIT, or PARK with an armed record."""
    w = self.c["wait"]
    if kind not in w["kinds"]:
      raise W6Error("UNKNOWN_WAIT_KIND", kind)
    if kind == "WAIT" and len(operands) != w["wait_operands"]:
      raise W6Error("WAIT_OPERAND_COUNT", f"{len(operands)}")
    if kind == "WAITALL" and not (w["waitall_operands"]["min"] <= len(operands) <= self.waitall_max):
      raise W6Error("WAITALL_OPERAND_COUNT", f"{len(operands)}")
    if len({(o.tile_id, o.ctx_slot, o.local_event_id) for o in operands}) != len(operands):
      raise W6Error("DUPLICATE_OPERAND")
    if ctx in self.waits:
      raise W6Error("WAIT_RECORD_BUSY", "one wait record per Context")
    self._pin(operands)  # also rejects an operand that does not resolve by its full key
    record = WaitRecord(ctx, kind, list(operands))
    verdict = self._verdict(record)
    if verdict:
      self._settle(record, verdict)
      self.wakes.pop()  # the issuing Context sees the verdict directly, nothing to wake
      return verdict
    self.waits[ctx] = record
    return "PARK"

  def _evaluate_waits(self) -> None:
    for ctx, record in list(self.waits.items()):
      verdict = self._verdict(record)
      if verdict:
        record.armed = False
        del self.waits[ctx]
        self._settle(record, verdict)

  def step(
    self,
    completions: list[tuple[EventKey, str, int]],
    waits: list[tuple[tuple[int, int, int], str, list[EventKey]]],
  ) -> list[str]:
    """One cycle: completions are applied first, so a same-cycle completion wins over park."""
    for key, status, producer in completions:
      self.signal(key, status, producer)
    return [self.issue_wait(ctx, kind, ops) for ctx, kind, ops in waits]

  def fault_context(self, ctx: tuple[int, int, int]) -> None:
    """A faulted Context's record is voided: it matches nothing later and releases its pins."""
    record = self.waits.pop(ctx, None)
    if record is None:
      return
    for op in record.operands:
      entry = self._resolve(op)
      entry.consumer_refs -= 1
      entry.future_uses -= 1
      self._reclaim(entry)

  # ---- terminal reserve ----
  def store_terminal(self, tile_id: int, ctx_slot: int, record: dict[str, Any]) -> None:
    """Terminal records live in their own reserve, so a full frontier cannot block them."""
    held = self.terminal_records.setdefault((tile_id, ctx_slot), [])
    if len(held) >= self.reserve:
      raise W6Error("TERMINAL_RESERVE_EXHAUSTED")
    held.append(record)

  # ---- MARK_EVENTS ----
  def mark_events(self) -> int:
    n = 0
    for entry in self.entries.values():
      if entry.status == "PENDING":
        entry.status = "RESET"
        n += 1
    self._evaluate_waits()
    return n

  def admit_task(self, static_live_bound: int) -> None:
    """Load-time admission: a static bound above the hardware slot count is a permanent rejection."""
    if _count(static_live_bound, "static_live_bound") > self.frontier:
      raise W6Error(self.c["capacity"]["load_time_rejection"]["code"])


def assert_writer(contract: dict[str, Any], resource: str, actor: str) -> None:
  """A resource has exactly one writer; any other actor is rejected."""
  writers = contract["writers"]
  if resource not in writers:
    raise W6Error("UNKNOWN_RESOURCE", resource)
  if writers[resource] != actor:
    raise W6Error("WRONG_WRITER", f"{resource} is written by {writers[resource]!r}, not {actor!r}")


# ---------------------------------------------------------------------------------------------------
# first-fault latch
# ---------------------------------------------------------------------------------------------------
class FirstFaultLatch:
  def __init__(self, terminal: dict[str, Any]) -> None:
    self.c = terminal["first_fault_latch"]
    self.record: dict[str, Any] | None = None
    self.suppressed = 0
    self.suppressed_sticky = False
    self.reset_active = False
    self.reset_done = False

  def _check(self, fault: dict[str, Any]) -> None:
    missing = [f for f in self.c["record_fields"] if f not in fault]
    if missing:
      raise W6Error("FAULT_FIELDS_MISSING", ",".join(missing))
    if fault["code"] not in self.c["fault_codes"]["names"]:
      raise W6Error("UNKNOWN_FAULT_CODE", str(fault["code"]))
    for name in ("ctx_slot", "ctx_generation", "owner_op_seq"):
      if name not in fault.get("owner_tag", {}):
        raise W6Error("OWNER_TAG_MISSING", name)

  @staticmethod
  def _order(fault: dict[str, Any]) -> tuple[int, int, int, int]:
    tag = fault["owner_tag"]
    return (fault["source_id"], fault["engine_id"], tag["ctx_slot"], tag["owner_op_seq"])

  def report_cycle(self, faults: list[dict[str, Any]]) -> None:
    """Same-cycle faults are ordered by direct fields; only the first one overall may arm the latch."""
    for fault in faults:
      self._check(fault)
    for fault in sorted(faults, key=self._order):
      if self.record is None and not (self.reset_active or self.reset_done):
        self.record = dict(fault)
      else:
        self.suppressed += 1
        self.suppressed_sticky = True

  def functional_reset(self) -> None:
    self.reset_active = False
    self.reset_done = True  # the latch itself is untouched

  def clear(self, by: str) -> None:
    if by not in self.c["cleared_only_by"]:
      raise W6Error("LATCH_CLEAR_REFUSED", by)
    self.record = None
    self.reset_active = False
    self.reset_done = False


# ---------------------------------------------------------------------------------------------------
# Reset endpoint, cancel, poison
# ---------------------------------------------------------------------------------------------------
class ResetEndpoint:
  """The Tile segment of the Group reset: passive, ordered, observable. It has no timer."""

  def __init__(self, terminal: dict[str, Any], facility: EventFacility) -> None:
    self.c = terminal["reset_endpoint"]
    self.facility = facility
    self.phases = {p["name"]: p["requires"] for p in self.c["phases"]}
    self.done: list[str] = []
    self.active = False
    self.poisoned = False
    self.pending_to_reset = 0

  def begin(self) -> None:
    self.done, self.active, self.pending_to_reset = [], True, 0

  def start_task(self) -> None:
    if self.poisoned:
      raise W6Error("POISONED", "rejects every new start until an explicit quiescent reset")
    if self.active:
      raise W6Error("RESET_ACTIVE", "new work is rejected until the reset is done")

  def advance(self, phase: str, evidence: dict[str, bool] | None = None, *, actor: str = "group") -> Any:
    if actor != self.c["authority"]:
      raise W6Error("WRONG_AUTHORITY", f"phases are advanced by {self.c['authority']}, not {actor}")
    if not self.active:
      raise W6Error("NOT_RESETTING")
    if phase not in self.phases:
      raise W6Error("UNKNOWN_PHASE", phase)
    if phase in self.done:
      return None  # idempotent repeat of an acknowledged phase
    unmet = [r for r in self.phases[phase] if r not in self.done]
    if unmet:
      raise W6Error("PHASE_ORDER", f"{phase} requires {unmet}")
    result: Any = None
    if phase == "DRAIN":
      self._check_drain(evidence)
    elif phase == "MARK_EVENTS":
      self.pending_to_reset = self.facility.mark_events()
      result = {"pending_to_reset": self.pending_to_reset}
    self.done.append(phase)
    if phase == "RESUME_DONE":
      self.active = False
    return result

  def _check_drain(self, evidence: dict[str, bool] | None) -> None:
    drain = self.c["drain"]
    evidence = evidence or {}
    for name in drain["evidence_excluded"]:
      if name in evidence:
        raise W6Error("DRAIN_WAITS_ON_PENDING", name)
    missing = [n for n in drain["evidence_required"] if n not in evidence]
    if missing:
      raise W6Error("DRAIN_EVIDENCE_MISSING", ",".join(missing))
    unknown = [n for n in evidence if n not in drain["evidence_required"]]
    if unknown:
      raise W6Error("DRAIN_EVIDENCE_UNKNOWN", ",".join(unknown))
    open_ = [n for n, ok in evidence.items() if ok is not True]
    if open_:
      raise W6Error("DRAIN_NOT_QUIESCENT", ",".join(open_))

  def invalidate_ack(self, references_outstanding: int) -> None:
    if references_outstanding:
      raise W6Error("INVALIDATE_ACK_EARLY", f"{references_outstanding} references remain")

  def poison(self) -> None:
    self.poisoned = True

  def explicit_reset(self, *, quiescent: bool, zero_evidence: bool) -> None:
    if not (quiescent and zero_evidence):
      raise W6Error("RECOVERY_REFUSED", "needs a quiescent Tile and zero evidence")
    self.poisoned = False


class CancelTracker:
  """CANCEL_TASK classification: stale, live-unknown/conflict and duplicate are three different outcomes."""

  def __init__(self, terminal: dict[str, Any]) -> None:
    self.c = terminal["cancel"]
    self.live: dict[Any, int] = {}
    self.applied: set[tuple[Any, int]] = set()
    self.stale_count = 0

  def bind(self, task_key: Any, ctx_generation: int) -> None:
    self.live[task_key] = ctx_generation

  def retire(self, task_key: Any) -> None:
    self.live.pop(task_key, None)

  def cancel(self, task_key: Any, ctx_generation: int) -> str:
    if task_key not in self.live:
      self.stale_count += 1
      return "STALE_ACK"
    if self.live[task_key] != ctx_generation:
      raise W6Error(
        "CANCEL_GENERATION_CONFLICT", "a live Task with another ctx_generation is a protocol error"
      )
    if (task_key, ctx_generation) in self.applied:
      return "IDEMPOTENT_ACK"
    self.applied.add((task_key, ctx_generation))
    return "APPLIED"


def settle_external_write(
  contract: dict[str, Any], outcome: str, *, timed_out: bool = False
) -> dict[str, Any]:
  """An accepted write ends only on a real outcome; a timeout or a failed isolation never completes it."""
  c = contract["cancel"]["external_write_settlement"]
  if outcome == "TIMEOUT" or (timed_out and outcome not in c["states"]):
    raise W6Error("TIMEOUT_IS_NOT_SETTLEMENT")
  if outcome not in c["states"] and outcome != "ISOLATION_FAILED":
    raise W6Error("UNKNOWN_OUTCOME", outcome)
  if outcome == "ISOLATION_FAILED":
    return {"state": "UNCLOSED", "released": False, "poison": True}
  return {"state": outcome, "released": True, "poison": outcome == "FAULTED"}


# ---------------------------------------------------------------------------------------------------
# Task terminal / retirement chain
# ---------------------------------------------------------------------------------------------------
class Retirement:
  def __init__(self, terminal: dict[str, Any]) -> None:
    self.c = terminal["task_terminal"]
    self.requires = {s["step"]: s["requires"] for s in self.c["chain"]}
    self.done: list[str] = []
    self.terminal_state: str | None = None

  def do(self, step: str) -> None:
    if step not in self.requires:
      raise W6Error("UNKNOWN_STEP", step)
    if step in self.done:
      raise W6Error("STEP_REPEATED", step)
    unmet = [r for r in self.requires[step] if r not in self.done]
    if unmet:
      raise W6Error("STEP_ORDER", f"{step} requires {unmet}")
    self.done.append(step)

  def publish(self, state: str, *, fault_chain: bool = False, w4_evidence: bool = True) -> None:
    if state not in self.c["states"]:
      raise W6Error("NO_SUCH_TERMINAL", state)
    if fault_chain and state == "DONE":
      raise W6Error("FAULT_CHAIN_SUCCESS")
    if state == "TIMEOUT" and not w4_evidence:
      raise W6Error("TIMEOUT_WITHOUT_EVIDENCE")
    if self.terminal_state is not None:
      raise W6Error("TERMINAL_NOT_UNIQUE", self.terminal_state)
    self.terminal_state = state


class CompletionStore:
  """Stable under backpressure: the identity is kept until the send is accepted."""

  def __init__(self, capacity: int) -> None:
    self.capacity = _count(capacity, "capacity", minimum=1)
    self.held: deque[dict[str, Any]] = deque()

  def store(self, record: dict[str, Any]) -> None:
    if len(self.held) >= self.capacity:
      raise W6Error("COMPLETION_STORE_FULL")
    self.held.append(record)

  def send(self, accepted: bool) -> dict[str, Any] | None:
    if not self.held:
      return None
    if not accepted:
      return None  # the record and its identity stay exactly as they were
    return self.held.popleft()


class WrapAuthority:
  """Bit-pattern reuse is authorized by the Group, never by Tile-local quiescence."""

  def __init__(self, terminal: dict[str, Any]) -> None:
    self.c = terminal["wrap_quiesce"]

  def authorize_reuse(
    self, response: dict[str, Any] | None, *, old_copy_may_arrive: bool, outer_epoch: bool
  ) -> None:
    if response is None or response.get("authorized") is not True:
      raise W6Error("NO_GROUP_AUTHORIZATION")
    missing = [c for c in self.c["covers"] if c not in response.get("drained", ())]
    if missing:
      raise W6Error("WRAP_COVERAGE_INCOMPLETE", ",".join(missing))
    if "evidence_seq" not in response:
      raise W6Error("WRAP_RESPONSE_UNVERIFIABLE")
    if old_copy_may_arrive and not outer_epoch:
      raise W6Error("ABA_REUSE_FORBIDDEN")


# ---------------------------------------------------------------------------------------------------
# Clock gating
# ---------------------------------------------------------------------------------------------------
class Gating:
  def __init__(self, terminal: dict[str, Any], contexts: int) -> None:
    self.c = terminal["clock_gating"]
    self.parked = [False] * _count(contexts, "contexts", minimum=1)
    self.wfi = [False] * contexts

  def park(self, ctx: int, *, wfi: bool = False) -> None:
    self.parked[ctx] = True
    self.wfi[ctx] = wfi

  def wake(self, ctx: int) -> None:
    self.parked[ctx] = False
    self.wfi[ctx] = False

  def context_front_end_gated(
    self, ctx: int, *, completion_arrival: bool, writeback_port_busy: bool
  ) -> bool:
    if not self.parked[ctx] or completion_arrival or writeback_port_busy:
      return False
    return True

  def tile_gated(
    self, in_flight: dict[str, int], completion_channel_empty: bool, *, arrival_this_cycle: bool
  ) -> bool:
    need = {"lsu", "mfe", "rvv", "vme"}
    if set(in_flight) != need:
      raise W6Error("IN_FLIGHT_UNITS_INCOMPLETE", f"need {sorted(need)}")
    if arrival_this_cycle or not completion_channel_empty or any(in_flight.values()):
      return False
    return all(self.parked)  # WFI alone never reaches here


# ---------------------------------------------------------------------------------------------------
# Debug FIFO and export
# ---------------------------------------------------------------------------------------------------
@dataclass
class DebugState:
  depth: int
  enabled: bool = False
  fifo: list[dict[str, Any]] = field(default_factory=list)
  dropped: int = 0
  overflow: bool = False
  window_incomplete: bool = False
  pending_writes: dict[int, dict[str, Any]] = field(default_factory=dict)
  poisoned: bool = False
  closing: bool = False


class DebugPath:
  def __init__(self, terminal: dict[str, Any], depth: int) -> None:
    self.c = terminal["debug"]
    self.s = DebugState(_count(depth, "depth", minimum=1))

  def enable(self, *, privileged: bool) -> None:
    if not privileged:
      raise W6Error("DEBUG_NOT_PRIVILEGED")
    self.s.enabled, self.s.closing = True, False

  def push(self, record: dict[str, Any]) -> bool:
    """Never blocks. Off: nothing is kept. Full: the NEW record is dropped and counted."""
    if not self.s.enabled or self.s.closing:
      return False
    allowed = set(self.c["restricted_export_fields"]["allowed"])
    bad = set(record) - allowed
    if bad & set(self.c["restricted_export_fields"]["forbidden"]):
      raise W6Error("EXPORT_FIELD_FORBIDDEN", ",".join(sorted(bad)))
    if bad:
      raise W6Error("EXPORT_FIELD_UNKNOWN", ",".join(sorted(bad)))
    if len(self.s.fifo) >= self.s.depth:
      self.s.dropped += 1
      self.s.overflow = True
      self.s.window_incomplete = True
      return False
    self.s.fifo.append(record)
    return True

  def accept_log_write(self, txn: int, owner: int, addr: int, credit: int) -> None:
    if not self.s.enabled:
      raise W6Error("DEBUG_OFF")
    self.s.pending_writes[txn] = {"owner": owner, "addr": addr, "credit": credit}

  def close(self) -> None:
    self.s.closing = True

  def clear_fifo(self) -> None:
    self.s.fifo.clear()

  def settle(self, txn: int, outcome: str) -> None:
    if txn not in self.s.pending_writes:
      raise W6Error("UNKNOWN_TXN", str(txn))
    if outcome == "TIMEOUT":
      raise W6Error("TIMEOUT_IS_NOT_SETTLEMENT")
    if outcome == "ISOLATION_FAILED":
      self.s.poisoned = True
      return
    if outcome not in ("COMPLETED", "ISOLATED", "FAULTED"):
      raise W6Error("UNKNOWN_OUTCOME", outcome)
    del self.s.pending_writes[txn]

  def status(self) -> str:
    if self.s.poisoned:
      return "poison"
    return (
      "pending" if self.s.pending_writes else "closed" if (self.s.closing or not self.s.enabled) else "open"
    )

  def ring_reusable(self) -> bool:
    return not self.s.pending_writes and not self.s.poisoned

  def after_reset(self) -> None:
    self.s.enabled = False
    self.s.fifo.clear()
    self.s.closing = True
