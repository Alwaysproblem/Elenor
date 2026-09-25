"""Profiling trace collector for Perfetto / Chrome `chrome://tracing`.

Produces a Chrome Trace Format JSON (`trace_event` schema) that can be
loaded directly into Perfetto (perfetto.dev) or Chrome's built-in
`chrome://tracing` viewer.  An optional standalone HTML wrapper embeds
the same JSON with a minimal `catapult:trace_viewer` shim so it can be
opened in any browser without a server.

The trace has three kinds of events:

  * **slice**  (ph=B/E):  engine jobs (BOA/EVU/MFE/USE), UCE instruction
    phases (wait/issue/stream), task role dispatches, DMA jobs.
    These show up as horizontal bars on a Gantt timeline.
  * **counter** (ph=C):   stream-queue occupancy and credit, sampled
    per cycle.  These render as line graphs in Perfetto/Chrome.
  * **instant** (ph=i): CPU request lifecycle, distinct Group
    REGISTER/ISSUE/COMPLETE markers, UCE issue attempts, stream events, and
    tile/group completion markers.

Cycle → time mapping: one cycle = `hw.cycle_ns()` nanoseconds.  Chrome
trace uses microseconds, so cycles are converted to µs with 3 decimal
places to preserve sub-µs resolution at 1 GHz.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import HardwareConfig

# ---------------------------------------------------------------------------
# Lane sort registry (PR 5 §2.1) — deterministic track ordering metadata
# ---------------------------------------------------------------------------

# Numeric suffixes occupy a fixed-width band, so Tile999 stays before the
# next process family and MFE_LD999 stays before BOA.  Larger IDs fail
# explicitly instead of silently crossing trace groups.
_SORT_BAND_WIDTH = 1000

# Process sort: "Tile" is a prefix family (Tile0..Tile999 -> 200000+n).
_PROCESS_SORT = {"CPU Device": 0, "Device": 1, "TileGroup": 100, "Tile": 200}
_UNKNOWN_SORT = 900 * _SORT_BAND_WIDTH

# Thread sort: per-process table, longest-prefix match, trailing digits of
# the thread name become an in-band offset ("L2 Bank:3" -> 60000+3).
_THREAD_SORT = {
  "CPU Device": {"Controller": 10, "Pending": 20, "Completion": 30, "Request:": 40},
  "Device": {"Slot:": 10},
  "TileGroup": {
    "Task": 10,
    "TileRole:": 20,
    "Scheduler:Control": 25,
    "Scheduler:L2": 30,
    "Profile:Command": 26,
    "Profile:Step": 27,
    "Profile:Member": 28,
    "HBM → L2 Input #": 40,
    "L2 → HBM Output #": 50,
    "Memory:HBM": 60,
    "Memory:L2 Read": 70,
    "Memory:L2 Write": 71,
    "Memory:L2 State": 72,
    "Memory:L2 Arena": 73,
    "L2 Bank:": 80,
    "Global DMA Ch:": 90,
    "HBM Ch:": 100,
    "NoC:VC": 110,
    "Collective": 120,
    "StreamQ:": 130,
  },
  "Tile": {
    "UCE CTX": 10,
    "MFE_LD": 20,
    "BOA": 30,
    "EVU": 31,
    "MFE": 32,
    "USE": 33,
    "MFE_ST": 40,
    "Memory:L1 Read": 50,
    "Memory:L1 Write": 51,
    "Memory:L1 State": 52,
    "Memory:L1 Arena": 53,
    "L1 Bank:": 60,
    "Local DMA Load": 70,
    "Local DMA Store": 71,
    "MSHR": 80,
    "UCE": 90,
    "Lifecycle": 95,
  },
}


def _prefix_offset(prefix: str, base: int, name: str) -> int:
  remainder = name[len(prefix) :]
  offset = int(remainder) if remainder.isdigit() else 0
  if offset >= _SORT_BAND_WIDTH:
    raise ValueError(
      f"trace lane {name!r} numeric suffix {offset} exceeds reserved range 0..{_SORT_BAND_WIDTH - 1}"
    )
  return base * _SORT_BAND_WIDTH + offset


def _process_sort(track: str) -> int:
  """Sort index for a process track name (longest registered prefix)."""
  best = _UNKNOWN_SORT
  best_len = -1
  for prefix, base in _PROCESS_SORT.items():
    if track.startswith(prefix) and len(prefix) > best_len:
      best = _prefix_offset(prefix, base, track)
      best_len = len(prefix)
  return best


def _thread_sort(track: str, thread: str) -> int:
  """Sort index for a thread lane within its process table."""
  table: dict[str, int] | None = None
  best_len = -1
  for process in _THREAD_SORT:
    if track.startswith(process) and len(process) > best_len:
      table = _THREAD_SORT[process]
      best_len = len(process)
  if table is None:
    return _UNKNOWN_SORT
  best = _UNKNOWN_SORT
  best_len = -1
  for prefix, base in table.items():
    if thread.startswith(prefix) and len(prefix) > best_len:
      best = _prefix_offset(prefix, base, thread)
      best_len = len(prefix)
  return best


# TransferLegKind values (memory/transfer.py) — kept as a literal set so
# trace.py never imports the memory package (components import this module).
_LEG_KIND_VALUES = frozenset(
  {
    "hbm_read",
    "hbm_write",
    "global_dma",
    "noc_response",
    "noc_request",
    "l2_read",
    "l2_write",
    "local_dma",
    "l1_read",
    "l1_write",
    "l1_cache_lookup",
    "l2_cache_lookup",
    "l1_cache_fill",
    "l2_cache_fill",
  }
)
_LEG_REQUIRED_ARGS = frozenset(
  {
    "transaction_id",
    "flow_id",
    "bytes",
    "accepted_cycle",
    "completion_cycle",
    "source_space",
    "destination_space",
  }
)
_GROUP_ACTION_EVENTS = frozenset({"group_action_register", "group_action_issue", "group_action_complete"})
_GROUP_ACTION_REQUIRED_ARGS = frozenset({"context", "launch_generation", "ordinal", "kind", "cycle"})
_PROFILE_SLICE_EVENTS = frozenset({"profile_command", "profile_step"})
_PROFILE_MEMBER_EVENTS = frozenset({"profile_member_request", "profile_member_ack"})
_PROFILE_REQUIRED_ARGS = frozenset(
  {"command_id", "level", "member", "transaction", "generation", "stage", "status", "source_ref"}
)


@dataclass
class Tracer:
  """Collects Chrome trace events during a simulation run.

  Call `begin`/`end` for slice events, `counter` for sampled values,
  `instant` for point-in-time markers.  After the run, `to_chrome_json`
  produces the Perfetto-loadable JSON.
  """

  hw: HardwareConfig
  _events: list[dict] = field(default_factory=list)
  # track open slices: (pid, tid, name) -> (start_us, args)
  _open: dict[tuple, tuple] = field(default_factory=dict)
  _pid_counter: int = 0
  _tid_counter: int = 0
  _pids: dict[str, int] = field(default_factory=dict)  # track_name -> pid
  _tids: dict[tuple[int, str], int] = field(default_factory=dict)
  # flow key -> dense flow id, assigned in first-appearance order
  _flow_ids: dict[str, int] = field(default_factory=dict)
  # (track, effective thread, counter name) -> last sampled value
  _last_counter: dict[tuple[str, str, str], float] = field(default_factory=dict)

  # ---- helpers ---------------------------------------------------------

  def cycle_to_us(self, cycle: int) -> float:
    """Convert a cycle number to wall-clock microseconds."""
    ns = cycle * self.hw.cycle_ns()
    return round(ns / 1000.0, 3)

  def _pid(self, track: str) -> int:
    if track not in self._pids:
      self._pid_counter += 1
      pid = self._pid_counter
      self._pids[track] = pid
      self._events.append(
        {"name": "process_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": track}}
      )
      self._events.append(
        {
          "name": "process_sort_index",
          "ph": "M",
          "pid": pid,
          "tid": 0,
          "args": {"sort_index": _process_sort(track)},
        }
      )
    return self._pids[track]

  def _tid(self, pid: int, thread: str, track: str = "") -> int:
    key = (pid, thread)
    if key not in self._tids:
      self._tid_counter += 1
      tid = self._tid_counter
      self._tids[key] = tid
      self._events.append(
        {"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": thread}}
      )
      self._events.append(
        {
          "name": "thread_sort_index",
          "ph": "M",
          "pid": pid,
          "tid": tid,
          "args": {"sort_index": _thread_sort(track, thread)},
        }
      )
    return self._tids[key]

  # ---- slice events (Gantt bars) --------------------------------------

  def begin(self, track: str, thread: str, name: str, cycle: int, args: dict | None = None) -> None:
    """Start a slice.  Pair with `end` using the same (track, thread, name)."""
    pid = self._pid(track)
    tid = self._tid(pid, thread, track)
    key = (pid, tid, name)
    self._open[key] = (self.cycle_to_us(cycle), args or {})
    self._events.append(
      {
        "name": name,
        "ph": "B",
        "pid": pid,
        "tid": tid,
        "ts": self.cycle_to_us(cycle),
        "cat": thread,
        "args": dict(args) if args else {},
      }
    )

  def end(self, track: str, thread: str, name: str, cycle: int) -> None:
    """End a previously-started slice."""
    pid = self._pid(track)
    tid = self._tid(pid, thread, track)
    key = (pid, tid, name)
    self._events.append(
      {"name": name, "ph": "E", "pid": pid, "tid": tid, "ts": self.cycle_to_us(cycle), "cat": thread}
    )
    self._open.pop(key, None)

  def complete(
    self,
    track: str,
    thread: str,
    name: str,
    start_cycle: int,
    end_cycle: int,
    args: dict | None = None,
    category: str | None = None,
  ) -> None:
    """Emit a complete slice (X event) with known start and end."""
    pid = self._pid(track)
    tid = self._tid(pid, thread, track)
    dur = self.cycle_to_us(end_cycle) - self.cycle_to_us(start_cycle)
    self._events.append(
      {
        "name": name,
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": self.cycle_to_us(start_cycle),
        "dur": max(dur, 0.001),
        "cat": category if category is not None else thread,
        "args": dict(args) if args else {},
      }
    )

  # ---- counter events (line graphs) -----------------------------------

  def counter(
    self, track: str, name: str, cycle: int, value: float, unit: str = "", thread: str | None = None
  ) -> None:
    """Sample a counter value at a given cycle.

    ``thread`` selects the lane the counter renders on; when omitted
    the counter name itself is the thread name (legacy behavior).
    """
    eff_thread = thread if thread is not None else name
    pid = self._pid(track)
    tid = self._tid(pid, eff_thread, track)
    self._events.append(
      {
        "name": name,
        "ph": "C",
        "pid": pid,
        "tid": tid,
        "ts": self.cycle_to_us(cycle),
        "args": {name: value, "unit": unit} if unit else {name: value},
      }
    )

  def counter_if_changed(
    self, track: str, name: str, cycle: int, value: float, unit: str = "", thread: str | None = None
  ) -> None:
    """Sample a counter only when its value changed (PR 5 §2.5).

    The first sample for each (track, thread, name) is always
    emitted; subsequent samples are dropped while the value is
    unchanged.  Keeps per-cycle emission sites cheap without
    flooding the trace with constant samples.
    """
    eff_thread = thread if thread is not None else name
    key = (track, eff_thread, name)
    if self._last_counter.get(key) == value:
      return
    self._last_counter[key] = value
    self.counter(track, name, cycle, value, unit, thread)

  # ---- instant events (markers) ---------------------------------------

  def instant(self, track: str, thread: str, name: str, cycle: int, args: dict | None = None) -> None:
    pid = self._pid(track)
    tid = self._tid(pid, thread, track)
    self._events.append(
      {
        "name": name,
        "ph": "i",
        "pid": pid,
        "tid": tid,
        "ts": self.cycle_to_us(cycle),
        "cat": thread,
        "s": "t",
        "args": dict(args) if args else {},
      }
    )

  # ---- flow events (cross-lane arrows) ---------------------------------

  def flow_id(self, key: str) -> int:
    """Dense flow id for a stable key, assigned on first appearance."""
    fid = self._flow_ids.get(key)
    if fid is None:
      fid = len(self._flow_ids) + 1
      self._flow_ids[key] = fid
    return fid

  def _flow_event(
    self, ph: str, track: str, thread: str, name: str, cycle: int, key: str, args: dict | None
  ) -> None:
    pid = self._pid(track)
    tid = self._tid(pid, thread, track)
    self._events.append(
      {
        "name": name,
        "ph": ph,
        "pid": pid,
        "tid": tid,
        "ts": self.cycle_to_us(cycle),
        "cat": thread,
        "id": self.flow_id(key),
        "args": dict(args) if args else {},
      }
    )

  def flow_start(
    self, track: str, thread: str, name: str, cycle: int, key: str, args: dict | None = None
  ) -> None:
    """Open a flow (ph=s) anchored on one lane."""
    self._flow_event("s", track, thread, name, cycle, key, args)

  def flow_step(
    self, track: str, thread: str, name: str, cycle: int, key: str, args: dict | None = None
  ) -> None:
    """Continue a flow (ph=t) on another lane."""
    self._flow_event("t", track, thread, name, cycle, key, args)

  def flow_end(
    self, track: str, thread: str, name: str, cycle: int, key: str, args: dict | None = None
  ) -> None:
    """Close a flow (ph=f) anchored on its final lane."""
    self._flow_event("f", track, thread, name, cycle, key, args)

  # ---- well-formedness (test entry, not a hot path) --------------------

  def assert_well_formed(self) -> None:
    """Validate the collected trace; raise AssertionError on violation.

    Checks (PR 5 §2.10): metadata coverage for every process/thread,
    counter args shape and change-only sampling, non-negative X
    durations, closed B/E pairs, one (s, f) pair per flow id with
    steps in between, and required identity args on transfer-leg
    slices.  Collects all violations before raising.
    """
    errors: list[str] = []
    proc_names: dict[Any, bool] = {}
    proc_sorts: dict[Any, bool] = {}
    thread_names: dict[Any, bool] = {}
    thread_sorts: dict[Any, bool] = {}
    last_counter: dict[Any, object] = {}
    open_stack: dict[Any, int] = {}
    flow_start_ts: dict[Any, float] = {}
    flow_end_ts: dict[Any, float] = {}
    flow_steps: dict[Any, list[float]] = {}
    for ev in self._events:
      ph = ev.get("ph")
      if ph == "M":
        mname = ev.get("name")
        if mname == "process_name":
          proc_names[ev["pid"]] = True
        elif mname == "process_sort_index":
          proc_sorts[ev["pid"]] = True
        elif mname == "thread_name":
          thread_names[(ev["pid"], ev["tid"])] = True
        elif mname == "thread_sort_index":
          thread_sorts[(ev["pid"], ev["tid"])] = True
        continue
      pid = ev.get("pid")
      tid = ev.get("tid")
      if ph in ("B", "E", "X", "C", "i", "s", "t", "f"):
        if not proc_names.get(pid):
          errors.append(f"pid {pid} lacks process_name")
          proc_names[pid] = True  # report once
        if not proc_sorts.get(pid):
          errors.append(f"pid {pid} lacks process_sort_index")
          proc_sorts[pid] = True
        if tid is not None and not thread_names.get((pid, tid)):
          errors.append(f"pid {pid} tid {tid} lacks thread_name")
          thread_names[(pid, tid)] = True
        if tid is not None and not thread_sorts.get((pid, tid)):
          errors.append(f"pid {pid} tid {tid} lacks thread_sort_index")
          thread_sorts[(pid, tid)] = True
      if ph == "C":
        name = ev.get("name", "")
        cargs = ev.get("args", {})
        extra = set(cargs) - {name, "unit"}
        if extra:
          errors.append(f"counter '{name}' has extra args {sorted(extra)}")
        if name not in cargs:
          errors.append(f"counter '{name}' lacks its own value key")
          continue
        ckey = (pid, tid, name)
        value = cargs[name]
        if ckey in last_counter and last_counter[ckey] == value:
          errors.append(f"counter '{name}' has consecutive duplicate value {value}")
        last_counter[ckey] = value
      elif ph == "X":
        if ev.get("dur", 0) < 0:
          errors.append(f"X '{ev.get('name')}' has negative dur")
        if ev.get("name") in _LEG_KIND_VALUES:
          missing = _LEG_REQUIRED_ARGS - set(ev.get("args", {}))
          if missing:
            errors.append(f"leg slice '{ev.get('name')}' missing args {sorted(missing)}")
        if ev.get("name") in _PROFILE_SLICE_EVENTS:
          missing = _PROFILE_REQUIRED_ARGS - set(ev.get("args", {}))
          if missing:
            errors.append(f"profile slice '{ev.get('name')}' missing args {sorted(missing)}")
      elif ph == "i" and ev.get("name") in _GROUP_ACTION_EVENTS:
        missing = _GROUP_ACTION_REQUIRED_ARGS - set(ev.get("args", {}))
        if missing:
          errors.append(f"Group action instant '{ev.get('name')}' missing args {sorted(missing)}")
      elif ph == "i" and ev.get("name") in _PROFILE_MEMBER_EVENTS:
        missing = _PROFILE_REQUIRED_ARGS - set(ev.get("args", {}))
        if missing:
          errors.append(f"profile member instant '{ev.get('name')}' missing args {sorted(missing)}")
      elif ph == "B":
        bkey = (pid, tid, ev.get("name"))
        open_stack[bkey] = open_stack.get(bkey, 0) + 1
      elif ph == "E":
        ekey = (pid, tid, ev.get("name"))
        if open_stack.get(ekey, 0) <= 0:
          errors.append(f"unmatched E for '{ev.get('name')}'")
        else:
          open_stack[ekey] -= 1
      elif ph in ("s", "t", "f"):
        fid = ev.get("id")
        if fid is None:
          errors.append("flow event without id")
        elif ph == "s":
          if fid in flow_start_ts:
            errors.append(f"flow {fid} has duplicate start")
          flow_start_ts[fid] = ev.get("ts", 0)
        elif ph == "f":
          if fid in flow_end_ts:
            errors.append(f"flow {fid} has duplicate end")
          flow_end_ts[fid] = ev.get("ts", 0)
        else:
          flow_steps.setdefault(fid, []).append(ev.get("ts", 0))
    for key, depth in open_stack.items():
      if depth > 0:
        errors.append(f"unclosed B slice '{key[2]}'")
    for fid in sorted(set(flow_start_ts) | set(flow_end_ts) | set(flow_steps)):
      if fid not in flow_start_ts:
        errors.append(f"flow {fid} lacks a start event")
      if fid not in flow_end_ts:
        errors.append(f"flow {fid} lacks an end event")
      if fid in flow_start_ts and fid in flow_end_ts:
        if flow_start_ts[fid] > flow_end_ts[fid]:
          errors.append(f"flow {fid} starts after it ends")
        for ts in flow_steps.get(fid, []):
          if not flow_start_ts[fid] <= ts <= flow_end_ts[fid]:
            errors.append(f"flow {fid} step outside [s, f] window")
    if errors:
      raise AssertionError("trace not well-formed:\n  " + "\n  ".join(errors))

  # ---- output ---------------------------------------------------------

  def to_chrome_json(self) -> str:
    """Produce the Perfetto/Chrome-loadable JSON trace."""
    # flush any open slices
    for (pid, tid, name), (_, _) in list(self._open.items()):
      self._events.append(
        {"name": name, "ph": "E", "pid": pid, "tid": tid, "ts": self.cycle_to_us(9999999), "cat": "open"}
      )
    return json.dumps({"traceEvents": self._events}, indent=None, separators=(",", ":"))

  def to_chrome_json_pretty(self) -> str:
    return json.dumps({"traceEvents": self._events}, indent=2)

  @property
  def event_count(self) -> int:
    return len(self._events)


class ScopedTracer:
  """Tracer wrapper prefixing every track (process lane) name.

  Thread names (and thus ``cat``) are unchanged, so per-thread
  assertions and HTML colors are unaffected.  Gives each device
  execution slot its own lane set (``G0:Tile0``, ``G0:TileGroup``...).
  """

  def __init__(self, inner: Tracer, prefix: str):
    self._inner = inner
    self._prefix = prefix

  def cycle_to_us(self, cycle: int) -> float:
    return self._inner.cycle_to_us(cycle)

  def begin(self, track, thread, name, cycle, args=None):
    self._inner.begin(self._prefix + track, thread, name, cycle, args)

  def end(self, track, thread, name, cycle):
    self._inner.end(self._prefix + track, thread, name, cycle)

  def complete(self, track, thread, name, start_cycle, end_cycle, args=None, category=None):
    self._inner.complete(self._prefix + track, thread, name, start_cycle, end_cycle, args, category)

  def counter(self, track, name, cycle, value, unit="", thread=None):
    self._inner.counter(self._prefix + track, name, cycle, value, unit, thread)

  def counter_if_changed(self, track, name, cycle, value, unit="", thread=None):
    self._inner.counter_if_changed(self._prefix + track, name, cycle, value, unit, thread)

  def instant(self, track, thread, name, cycle, args=None):
    self._inner.instant(self._prefix + track, thread, name, cycle, args)

  def flow_id(self, key):
    return self._inner.flow_id(key)

  def flow_start(self, track, thread, name, cycle, key, args=None):
    self._inner.flow_start(self._prefix + track, thread, name, cycle, key, args)

  def flow_step(self, track, thread, name, cycle, key, args=None):
    self._inner.flow_step(self._prefix + track, thread, name, cycle, key, args)

  def flow_end(self, track, thread, name, cycle, key, args=None):
    self._inner.flow_end(self._prefix + track, thread, name, cycle, key, args)


# ---------------------------------------------------------------------------
# MemoryTrace — semantic sink for memory-subsystem events (PR 5 §2.2)
# ---------------------------------------------------------------------------


@dataclass
class _LegRecord:
  """Per-leg bookkeeping between issue and completion (PR 5 §2.3)."""

  leg_index: int
  stage_name: str
  accepted_cycle: int
  completion_cycle: int  # -1 when unknown at issue (router-backed legs)
  resources: tuple[int, ...]
  noc_vc: int
  stall_reason: str
  channel_thread: str | None = None  # Ch lane for channel-type stages


@dataclass
class _ProfileTrace:
  """In-flight command/step timing sourced from the controller FSM."""

  runtime_id: str
  command_id: str
  start_cycle: int
  step_start_cycle: int
  current_step: str
  base_args: dict[str, Any]
  generation: int | None = None


class MemoryTrace:
  """Semantic sink memory components call with DTOs they already hold.

  Components never build Chrome JSON themselves: they call the methods
  below with live snapshots/handles/transactions, and this class maps
  them onto the fixed lane layout (PR 5 §2.4/§2.5).  Leg slices are
  recorded when a leg is accepted and emitted when it completes, so a
  cancelled leg never leaves a fabricated prediction behind.
  """

  def __init__(self, tracer: Tracer):
    self.tracer = tracer
    # transaction id -> FIFO of issued-but-not-emitted leg records
    self._leg_records: dict[str, list[_LegRecord]] = {}
    # transactions with at least one emitted leg slice
    self._emitted_flows: set[str] = set()
    self._closed_flows: set[str] = set()
    # transaction id -> (track, thread) of its last emitted leg slice
    self._last_leg_lane: dict[str, tuple[str, str]] = {}
    # transaction id -> most recent StageWait reason before the next issue
    self._last_wait: dict[str, str] = {}
    # runtime command id -> currently open command and FSM step.  Complete
    # slices are emitted only at real transitions/terminal states.
    self._profile_records: dict[str, _ProfileTrace] = {}
    self._arena_flows: set[str] = set()
    # Retained identity metadata lets late/ignored member responses remain
    # attributable after their command slice has reached a terminal state.
    self._profile_metadata: dict[str, dict[str, Any]] = {}

  # -- lane helpers -----------------------------------------------------

  @staticmethod
  def _space_lane(space: str, tile_id: int | None) -> tuple[str, str]:
    """State/capacity lane for one memory space ('l2' group-owned, 'l1' tile)."""
    if space == "l1":
      return f"Tile{tile_id}", "Memory:L1 State"
    return "TileGroup", "Memory:L2 State"

  @staticmethod
  def _channel_thread(stage_name: str, resources: tuple[int, ...]) -> str | None:
    if not resources:
      return None
    if stage_name == "global_dma":
      return f"Global DMA Ch:{resources[0]}"
    if stage_name in ("hbm_read", "hbm_write"):
      return f"HBM Ch:{resources[0]}"
    return None

  @staticmethod
  def _leg_lane(txn, leg, rec: _LegRecord) -> tuple[str, str]:
    kind = leg.kind.value
    if kind in ("hbm_read", "hbm_write"):
      ch = rec.resources[0] if rec.resources else 0
      return "TileGroup", f"HBM Ch:{ch}"
    if kind == "global_dma":
      ch = rec.resources[0] if rec.resources else 0
      return "TileGroup", f"Global DMA Ch:{ch}"
    if kind in ("noc_request", "noc_response"):
      return "TileGroup", f"NoC:VC{rec.noc_vc}"
    if kind in ("l2_read", "l2_cache_lookup"):
      return "TileGroup", "Memory:L2 Read"
    if kind in ("l2_write", "l2_cache_fill"):
      return "TileGroup", "Memory:L2 Write"
    if kind == "local_dma":
      load = leg.dst_space in ("l1", "l1_cache")
      return f"Tile{txn.tile_id}", ("Local DMA Load" if load else "Local DMA Store")
    if kind in ("l1_read", "l1_cache_lookup"):
      return f"Tile{txn.tile_id}", "Memory:L1 Read"
    if kind in ("l1_write", "l1_cache_fill"):
      return f"Tile{txn.tile_id}", "Memory:L1 Write"
    raise ValueError(f"unsupported memory trace leg kind {kind!r}")

  @staticmethod
  def _owner_args(owner) -> dict:
    args: dict = {"owner_kind": type(owner).__name__}
    if hasattr(owner, "context_name"):
      args["context_name"] = owner.context_name
      generation = getattr(owner, "context_launch_generation", None)
      if generation is None:
        generation = getattr(owner, "launch_generation", None)
      if generation is not None:
        args["context_launch_generation"] = generation
    if hasattr(owner, "buffer_id"):
      args["buffer_id"] = owner.buffer_id
    if hasattr(owner, "role_event_id"):
      args["role_event_id"] = owner.role_event_id
      args["task_id"] = owner.logical_task_id
    return args

  @staticmethod
  def _allocation_args(handle) -> dict:
    owner = handle.owner
    return {
      "allocation_id": handle.allocation_id,
      "owner_kind": type(owner).__name__,
      "buffer_id": getattr(owner, "buffer_id", getattr(owner, "binding_name", "")),
      "backing_id": getattr(handle, "backing_id", ""),
      "generation": handle.generation,
      "base_address": handle.base_address,
      "size_bytes": handle.size_bytes,
      "allocate_cycle": handle.allocate_cycle,
    }

  @staticmethod
  def _source_ref_args(source_ref) -> dict[str, Any]:
    if source_ref is None:
      return {
        "source_ref": "",
        "source_name": "",
        "source_symbol": "",
        "source_body_op_index": -1,
        "source_op_name": "",
        "generated_by": "",
        "source_reason": "",
      }
    source_name = getattr(source_ref, "source_name", "")
    symbol = getattr(source_ref, "symbol", "")
    body_op_index = getattr(source_ref, "body_op_index", -1)
    op_name = getattr(source_ref, "op_name", "")
    return {
      "source_ref": f"{source_name}:{symbol}:{body_op_index}:{op_name}",
      "source_name": source_name,
      "source_symbol": symbol,
      "source_body_op_index": body_op_index,
      "source_op_name": op_name,
      "generated_by": getattr(source_ref, "generated_by", ""),
      "source_reason": getattr(source_ref, "reason", ""),
    }

  @staticmethod
  def _member_name(member) -> str:
    if isinstance(member, (tuple, list)):
      return ":".join(str(item) for item in member)
    return "" if member is None else str(member)

  @classmethod
  def _profile_record_args(cls, record) -> dict[str, Any]:
    descriptor = record.descriptor
    level = getattr(descriptor, "level", None)
    if level is None:
      level = ",".join(getattr(descriptor, "levels", ()))
    members = tuple(getattr(descriptor, "member_ids", ()))
    owner = getattr(record, "owner", "")
    binding_id = owner.rpartition("@")[0] if "@" in owner else owner
    args = {
      "command_id": getattr(descriptor, "command_id", ""),
      "level": level,
      "member": ",".join(cls._member_name(member) for member in members),
      "transaction": record.runtime_id,
      "runtime_id": record.runtime_id,
      "generation": None,
      "stage": "command",
      "accepted_cycle": record.accepted_cycle,
      "status": record.state,
      "kind": record.kind,
      "owner": owner,
      "binding_id": binding_id,
      **cls._source_ref_args(getattr(descriptor, "source_ref", None)),
    }
    for name in ("expected_mode", "target_mode", "registry_hash", "exclusive_binding_id"):
      if hasattr(descriptor, name):
        args[name] = getattr(descriptor, name)
    for name in ("wait_instruction_ids", "frontier", "affected_domains", "dependencies", "levels"):
      if hasattr(descriptor, name):
        args[name] = ",".join(str(value) for value in getattr(descriptor, name))
    return args

  @classmethod
  def _arena_args(cls, arena, snapshot: dict) -> dict[str, Any]:
    owner = arena.owner
    args = {
      "arena_id": arena.arena_id,
      "space": arena.memory_space,
      "pool_id": snapshot.get("pool_id"),
      "tile_id": snapshot.get("tile_id"),
      "owner_kind": type(owner).__name__,
      "profile_mode": arena.profile.mode,
      "profile_generation": arena.profile_generation,
      "allocation_generation": arena.allocation_generation,
      "reserved_bytes": arena.reserved_bytes,
      "allocate_cycle": arena.allocate_cycle,
      "layout_hash": arena.layout.layout_hash,
      "stripe_bytes": arena.layout.stripe_bytes,
      "live_arenas": snapshot.get("live_arenas"),
      "live_views": snapshot.get("live_views"),
      "pool_reserved_bytes": snapshot.get("arena_reserved_bytes"),
      "pool_live_view_bytes": snapshot.get("live_view_bytes"),
      "pool_padding_bytes": snapshot.get("padding_bytes"),
      "pool_free_bytes": snapshot.get("free_bytes"),
    }
    args.update(cls._owner_args(owner))
    task = getattr(owner, "grid", None)
    if task is not None:
      args.update(
        {
          "context_name": task.context_name,
          "device_slot": task.device_slot,
          "launch_generation": task.launch_generation,
          "dispatch_ordinal": task.dispatch_ordinal,
          "task_id": owner.task_id,
        }
      )
    elif hasattr(owner, "launch_generation"):
      args["launch_generation"] = owner.launch_generation
    return args

  @classmethod
  def _view_args(cls, view, snapshot: dict) -> dict[str, Any]:
    return {
      **cls._allocation_args(view),
      "arena_id": view.arena_id,
      "profile_generation": view.profile_generation,
      "pool_id": snapshot.get("pool_id"),
      "tile_id": snapshot.get("tile_id"),
      "live_views": snapshot.get("live_views"),
      "pool_reserved_bytes": snapshot.get("arena_reserved_bytes"),
      "pool_live_view_bytes": snapshot.get("live_view_bytes"),
      "pool_padding_bytes": snapshot.get("padding_bytes"),
      "pool_free_bytes": snapshot.get("free_bytes"),
    }

  # -- capacity / bank counters (PR 5 §2.5) ----------------------------

  def capacity(self, space: str, tile_id: int | None, snapshot: dict, cycle: int) -> None:
    track, thread = self._space_lane(space, tile_id)
    tr = self.tracer
    for key in (
      "allocated_bytes",
      "free_bytes",
      "largest_free_extent",
      "live_allocations",
      "pending_release",
      "arena_reserved_bytes",
      "live_view_bytes",
      "padding_bytes",
      "system_reserved_bytes",
      "cache_bytes",
    ):
      if key in snapshot:
        tr.counter_if_changed(track, f"{space}_{key}", cycle, snapshot[key], thread=thread)

  def banks(self, space: str, tile_id: int | None, per_bank: list[dict], cycle: int) -> None:
    track = self._space_lane(space, tile_id)[0]
    fields = {
      "allocated_bytes": "allocated_bytes",
      "arena_reserved_bytes": "reserved_bytes",
      "live_view_bytes": "live_bytes",
      "padding_bytes": "padding_bytes",
      "system_reserved_bytes": "system_reserved_bytes",
      "cache_bytes": "cache_bytes",
      "free_bytes": "free_bytes",
    }
    for item in per_bank:
      thread = f"{space.upper()} Bank:{item['bank_id']}"
      for counter_field, suffix in fields.items():
        if counter_field in item:
          self.tracer.counter_if_changed(
            track, f"{space}_bank_{suffix}", cycle, item[counter_field], thread=thread
          )

  def cache(self, space: str, tile_id: int | None, stats, cycle: int) -> None:
    track, thread = self._space_lane(space, tile_id)
    tr = self.tracer
    tr.counter_if_changed(
      track, f"{space}_cache_resident_bytes", cycle, stats.resident_bytes, thread=thread
    )
    tr.counter_if_changed(
      track, f"{space}_cache_resident_lines", cycle, stats.resident_lines, thread=thread
    )

  def mshr(self, space: str, tile_id: int | None, stats, cycle: int) -> None:
    if space == "l1":
      track, thread = f"Tile{tile_id}", "MSHR"
    else:
      track, thread = "TileGroup", "Memory:L2 State"
    tr = self.tracer
    tr.counter_if_changed(track, f"{space}_mshr_active", cycle, stats.active, thread=thread)
    tr.counter_if_changed(track, f"{space}_mshr_merged", cycle, stats.merged, thread=thread)
    tr.counter_if_changed(track, f"{space}_mshr_stalls", cycle, stats.stalls, thread=thread)

  def hbm(self, snapshot: dict, cycle: int) -> None:
    tr = self.tracer
    tr.counter_if_changed(
      "TileGroup", "hbm_allocated_bytes", cycle, snapshot["used_bytes"], thread="Memory:HBM"
    )
    tr.counter_if_changed(
      "TileGroup",
      "hbm_free_bytes",
      cycle,
      snapshot["capacity_bytes"] - snapshot["used_bytes"],
      thread="Memory:HBM",
    )

  def hbm_outstanding(self, outstanding: int, limit: int, cycle: int) -> None:
    tr = self.tracer
    tr.counter_if_changed("TileGroup", "hbm_outstanding", cycle, outstanding, thread="Memory:HBM")
    tr.counter_if_changed("TileGroup", "hbm_credits", cycle, limit - outstanding, thread="Memory:HBM")

  def noc_vc(self, vc_id: int, occupancy: int, credit: int, cycle: int) -> None:
    thread = f"NoC:VC{vc_id}"
    tr = self.tracer
    tr.counter_if_changed("TileGroup", "noc_occupancy", cycle, occupancy, thread=thread)
    tr.counter_if_changed("TileGroup", "noc_credit_available", cycle, credit, thread=thread)

  # -- allocation lifecycle (PR 5 §2.6) --------------------------------

  def alloc_committed(self, space: str, tile_id: int | None, handle, cycle: int) -> None:
    track, thread = self._space_lane(space, tile_id)
    args = self._allocation_args(handle)
    name = f"{space}_alloc"
    self.tracer.instant(track, thread, name, cycle, args)
    self.tracer.flow_start(
      track, thread, name, cycle, handle.allocation_id, {"allocation_id": handle.allocation_id}
    )

  def alloc_released(self, space: str, tile_id: int | None, handle, cycle: int, reason: str) -> None:
    track, thread = self._space_lane(space, tile_id)
    args = self._allocation_args(handle)
    args["reason"] = reason
    name = f"{space}_release"
    self.tracer.instant(track, thread, name, cycle, args)
    self.tracer.flow_end(
      track, thread, name, cycle, handle.allocation_id, {"allocation_id": handle.allocation_id}
    )

  def _pool_snapshot(self, space: str, tile_id: int | None, snapshot: dict, cycle: int) -> None:
    """Emit counters from an ArenaPool-owned post-mutation snapshot."""
    self.capacity(space, tile_id, snapshot, cycle)
    self.banks(space, tile_id, snapshot.get("per_bank_occupancy", []), cycle)

  def arena_reserve(self, space: str, tile_id: int | None, arena, snapshot: dict, cycle: int) -> None:
    """Record a committed whole-Arena reservation and actual pool state."""
    track = self._space_lane(space, tile_id)[0]
    args = self._arena_args(arena, snapshot)
    self.tracer.instant(track, f"Memory:{space.upper()} Arena", "arena_reserve", cycle, args)
    self.tracer.flow_start(
      track,
      f"Memory:{space.upper()} Arena",
      "arena_reserve",
      cycle,
      arena.arena_id,
      {"arena_id": arena.arena_id},
    )
    self._arena_flows.add(arena.arena_id)
    self._pool_snapshot(space, tile_id, snapshot, cycle)

  def arena_retire(self, space: str, tile_id: int | None, arena, snapshot: dict, cycle: int) -> None:
    """Record only a successful whole-Arena retirement."""
    track = self._space_lane(space, tile_id)[0]
    thread = f"Memory:{space.upper()} Arena"
    args = self._arena_args(arena, snapshot)
    args["retire_cycle"] = cycle
    self.tracer.instant(track, thread, "arena_retire", cycle, args)
    self.tracer.complete(track, thread, "arena_lifetime", arena.allocate_cycle, cycle, args)
    if arena.arena_id in self._arena_flows:
      self.tracer.flow_end(
        track, thread, "arena_retire", cycle, arena.arena_id, {"arena_id": arena.arena_id}
      )
      self._arena_flows.remove(arena.arena_id)
    self._pool_snapshot(space, tile_id, snapshot, cycle)

  def buffer_view_invalidate(
    self, space: str, tile_id: int | None, view, snapshot: dict, cycle: int, backing=None
  ) -> None:
    """Record final view invalidation without implying Arena capacity release.

    ``backing`` (an ``_L2BackingRecord`` or None) carries the view's physical
    backing ID and its current lifecycle state (plan/01 §5).
    """
    track = self._space_lane(space, tile_id)[0]
    args = self._view_args(view, snapshot)
    if backing is not None:
      args["backing_id"] = backing.backing_id
      args["backing_state"] = backing.state
      args["backing_pinned"] = len(backing.pins)
      args["backing_inflight"] = len(backing.inflight)
    self.tracer.instant(
      track,
      f"Memory:{space.upper()} Arena",
      "buffer_view_invalidate",
      cycle,
      args,
    )
    self._pool_snapshot(space, tile_id, snapshot, cycle)

  def l2_extent_release(
    self, tile_id: int | None, backing, snapshot: dict, event: dict, cycle: int
  ) -> None:
    """Record one real physical backing final-free with post-mutation state."""
    track = self._space_lane("l2", tile_id)[0]
    thread = "Memory:L2 Arena"
    args = {
      **event,
      "pool_id": snapshot.get("pool_id"),
      "tile_id": snapshot.get("tile_id"),
      "pool_reserved_bytes": snapshot.get("arena_reserved_bytes"),
      "pool_free_bytes": snapshot.get("free_bytes"),
      "pool_live_backings": snapshot.get("live_backings"),
    }
    self.tracer.instant(track, thread, "l2_extent_release", cycle, args)
    self.tracer.complete(
      track, thread, "l2_backing_lifetime", backing.commit_cycle, cycle, dict(args)
    )
    self._pool_snapshot("l2", tile_id, snapshot, cycle)

  # -- profile controller protocol -------------------------------------

  def _emit_profile_step(self, trace: _ProfileTrace, cycle: int, status: str, reason: str = "") -> None:
    args = {
      **trace.base_args,
      "generation": trace.generation,
      "stage": trace.current_step,
      "status": status,
    }
    if reason:
      args["reason"] = reason
    self.tracer.complete("TileGroup", "Profile:Step", "profile_step", trace.step_start_cycle, cycle, args)

  def profile_command(self, record, cycle: int) -> None:
    """Open or finish the actual controller transaction."""
    trace = self._profile_records.get(record.runtime_id)
    if trace is None:
      base_args = self._profile_record_args(record)
      trace = _ProfileTrace(
        runtime_id=record.runtime_id,
        command_id=record.command_id,
        start_cycle=record.accepted_cycle,
        step_start_cycle=record.accepted_cycle,
        current_step=record.step,
        base_args=base_args,
      )
      self._profile_records[record.runtime_id] = trace
      self._profile_metadata[record.runtime_id] = base_args
    if record.state not in ("completed", "faulted", "cancelled"):
      return
    self._emit_profile_step(trace, cycle, record.state, getattr(record, "reason", ""))
    args = {
      **trace.base_args,
      "generation": trace.generation,
      "stage": "command",
      "status": record.state,
      "completed_cycle": record.completed_cycle,
    }
    if record.reason:
      args["reason"] = record.reason
    self.tracer.complete("TileGroup", "Profile:Command", "profile_command", trace.start_cycle, cycle, args)
    del self._profile_records[record.runtime_id]

  def profile_step(self, record, cycle: int) -> None:
    """Close the prior FSM step and start the controller-reported step."""
    trace = self._profile_records.get(record.runtime_id)
    if trace is None:
      self.profile_command(record, record.accepted_cycle)
      trace = self._profile_records.get(record.runtime_id)
    if trace is None or trace.current_step == record.step:
      return
    self._emit_profile_step(trace, cycle, "completed")
    trace.current_step = record.step
    trace.step_start_cycle = cycle

  def profile_member_ack(self, item, cycle: int, status: str) -> None:
    """Record one real member-bus request or validated response."""
    runtime_id = getattr(item, "runtime_id", "")
    trace = self._profile_records.get(runtime_id)
    generation = getattr(item, "generation", None)
    if trace is not None:
      trace.generation = generation
      base_args = trace.base_args
    else:
      base_args = self._profile_metadata.get(runtime_id, self._source_ref_args(None))
    metadata = self._profile_metadata.get(runtime_id)
    if metadata is not None:
      metadata["generation"] = generation
    member = getattr(item, "member_id", None)
    args = {
      **base_args,
      "command_id": getattr(item, "static_command_id", ""),
      "level": member[0] if isinstance(member, (tuple, list)) and member else "",
      "member": self._member_name(member),
      "transaction": runtime_id,
      "runtime_id": runtime_id,
      "generation": generation,
      "stage": getattr(item, "stage", ""),
      "status": status,
    }
    if hasattr(item, "success"):
      args["success"] = bool(item.success)
    if hasattr(item, "ready_cycle"):
      args["ready_cycle"] = item.ready_cycle
    if getattr(item, "reason", ""):
      args["reason"] = item.reason
    event = "profile_member_request" if status == "request" else "profile_member_ack"
    self.tracer.instant("TileGroup", "Profile:Member", event, cycle, args)

  def _profile_event(self, name: str, cycle: int, args: dict[str, Any], status: str) -> None:
    self.tracer.instant("TileGroup", "Profile:Command", name, cycle, {**args, "status": status})

  def profile_initialize(self, cycle: int, args: dict[str, Any]) -> None:
    self._profile_event("profile_initialize", cycle, args, "running")

  def profile_initialized(self, cycle: int, args: dict[str, Any]) -> None:
    self._profile_event("profile_initialized", cycle, args, "completed")

  def profile_recover(self, cycle: int, args: dict[str, Any]) -> None:
    self._profile_event("profile_recover", cycle, args, "running")

  def profile_recovered(self, cycle: int, args: dict[str, Any]) -> None:
    self._profile_event("profile_recovered", cycle, args, "completed")

  def profile_recover_failed(self, cycle: int, args: dict[str, Any]) -> None:
    self._profile_event("profile_recover_failed", cycle, args, "faulted")

  def profile_cancel_requested(self, cycle: int, args: dict[str, Any]) -> None:
    runtime_id = args.get("runtime_id", "")
    context = self._profile_metadata.get(runtime_id, {})
    self._profile_event("profile_cancel_requested", cycle, {**context, **args}, "cancel_requested")

  def hbm_bind(self, binding, handle, cycle: int) -> None:
    self.tracer.instant(
      "TileGroup",
      "Memory:HBM",
      "hbm_bind",
      cycle,
      {
        "binding": binding.name,
        "base_iova": binding.base_iova,
        "size_bytes": binding.size_bytes,
        "permissions": binding.permissions,
        "allocation_id": handle.allocation_id,
        "generation": handle.generation,
      },
    )

  def hbm_unbind(self, binding_name: str, cycle: int) -> None:
    self.tracer.instant("TileGroup", "Memory:HBM", "hbm_unbind", cycle, {"binding": binding_name})

  # -- transfer legs (PR 5 §2.3/§2.4) ----------------------------------

  def transfer_wait(self, transaction_id: str, reason: str) -> None:
    self._last_wait[transaction_id] = reason

  def transfer_leg_issued(
    self, txn, leg, stage_name: str, result, resources: tuple[int, ...], cycle: int
  ) -> None:
    rec = _LegRecord(
      leg_index=txn.current_leg,
      stage_name=stage_name,
      accepted_cycle=result.accepted_cycle,
      completion_cycle=result.completion_cycle,
      resources=tuple(resources),
      noc_vc=getattr(txn, "noc_vc", 0),
      stall_reason=self._last_wait.pop(txn.transaction_id, ""),
    )
    rec.channel_thread = self._channel_thread(stage_name, rec.resources)
    self._leg_records.setdefault(txn.transaction_id, []).append(rec)
    if rec.channel_thread is not None:
      self.tracer.counter_if_changed("TileGroup", "busy", cycle, 1, thread=rec.channel_thread)

  def _leg_args(self, txn, leg, rec: _LegRecord, end_cycle: int) -> dict:
    args = {
      "transaction_id": txn.transaction_id,
      "op": txn.op.value,
      "flow_id": self.tracer.flow_id(txn.transaction_id),
      "leg_index": rec.leg_index,
      "leg_count": len(txn.legs),
      **self._owner_args(txn.issuer),
      "tile_id": txn.tile_id,
      "source_space": leg.src_space,
      "destination_space": leg.dst_space,
      "source_address": txn.src.address if txn.src is not None else None,
      "destination_address": txn.dst.address if txn.dst is not None else None,
      "bytes": leg.bytes_total,
      "accepted_cycle": rec.accepted_cycle,
      "completion_cycle": end_cycle,
    }
    if rec.resources:
      args["banks"] = list(rec.resources)
    if leg.kind.value in ("noc_request", "noc_response"):
      args["noc_vc"] = rec.noc_vc
    if rec.stall_reason:
      args["stall_reason"] = rec.stall_reason
    return args

  def transfer_leg_completed(self, txn, cycle: int) -> None:
    queue = self._leg_records.get(txn.transaction_id)
    if not queue:
      return
    rec = queue.pop(0)
    if not queue:
      self._leg_records.pop(txn.transaction_id, None)
    leg = txn.legs[rec.leg_index]
    end_cycle = rec.completion_cycle if rec.completion_cycle > 0 else cycle
    track, thread = self._leg_lane(txn, leg, rec)
    self.tracer.complete(
      track, thread, leg.kind.value, rec.accepted_cycle, end_cycle, self._leg_args(txn, leg, rec, end_cycle)
    )
    flow_args = {"transaction_id": txn.transaction_id}
    key = txn.transaction_id
    if key not in self._emitted_flows:
      self.tracer.flow_start(track, thread, leg.kind.value, rec.accepted_cycle, key, flow_args)
      self._emitted_flows.add(key)
    else:
      self.tracer.flow_step(track, thread, leg.kind.value, rec.accepted_cycle, key, flow_args)
    self._last_leg_lane[key] = (track, thread)
    if rec.leg_index + 1 >= len(txn.legs):
      self.tracer.flow_end(track, thread, leg.kind.value, rec.accepted_cycle, key, flow_args)
      self._closed_flows.add(key)
    if rec.channel_thread is not None:
      self.tracer.counter_if_changed("TileGroup", "busy", end_cycle, 0, thread=rec.channel_thread)

  def transfer_cancelled(self, txn, cycle: int) -> None:
    queue = self._leg_records.pop(txn.transaction_id, None)
    if queue and queue[0].channel_thread is not None:
      # the issued-but-never-completed leg leaves its channel lane busy
      self.tracer.counter_if_changed("TileGroup", "busy", cycle, 0, thread=queue[0].channel_thread)
    self._last_wait.pop(txn.transaction_id, None)
    args = {"transaction_id": txn.transaction_id, "op": txn.op.value, **self._owner_args(txn.issuer)}
    self.tracer.instant("TileGroup", "Scheduler:L2", "transfer_cancelled", cycle, args)
    key = txn.transaction_id
    if key in self._emitted_flows and key not in self._closed_flows:
      track, thread = self._last_leg_lane.get(key, ("TileGroup", "Scheduler:L2"))
      self.tracer.flow_end(track, thread, "transfer_cancelled", cycle, key, {"transaction_id": key})
      self._closed_flows.add(key)


# ---------------------------------------------------------------------------
# Standalone HTML wrapper
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ELENOR Pipeline Validator — Trace</title>
<style>
body { margin:0; background:#1a1a1a; color:#eee; font-family:monospace; }
#header { padding:8px 12px; background:#222; border-bottom:1px solid #333; }
#header h2 { margin:0; font-size:14px; font-weight:normal; }
#chart { width:100%; min-height:90vh; }
.bar { position:absolute; height:14px; border-radius:2px; overflow:hidden;
       white-space:nowrap; font-size:10px; color:#fff; padding:1px 3px;
       box-sizing:border-box; cursor:pointer; }
.bar:hover { outline:1px solid #fff; }
.bar.flow-hi { outline:2px solid #fff200; z-index:5; }
#tooltip { position:fixed; background:#333; border:1px solid #555; padding:6px 8px;
           font-size:11px; border-radius:3px; pointer-events:none; z-index:100;
           max-width:480px; display:none; }
.axis { position:absolute; color:#888; font-size:10px; }
.counter-row { position:absolute; }
.inst { position:absolute; width:7px; height:7px; background:#f1c40f;
        transform:rotate(45deg); cursor:pointer; }
.inst:hover { outline:1px solid #fff; }
svg.flow-line { position:absolute; pointer-events:none; z-index:4; }
</style>
</head>
<body>
<div id="header"><h2>ELENOR Pipeline Validator — Gantt + Counters (Chrome Trace Format embedded)</h2></div>
<div id="chart"></div>
<div id="tooltip"></div>
<script>
const TRACE = __TRACE_JSON__;
(function() {
  const events = TRACE.traceEvents || [];
  // metadata maps
  const procName = {}, procSort = {};
  const threadName = {}, threadSort = {};
  for (const e of events) {
    if (e.ph !== "M") continue;
    if (e.name === "process_name") procName[e.pid] = e.args.name;
    else if (e.name === "process_sort_index") procSort[e.pid] = e.args.sort_index;
    else if (e.name === "thread_name") threadName[e.pid+":"+e.tid] = e.args.name;
    else if (e.name === "thread_sort_index") threadSort[e.pid+":"+e.tid] = e.args.sort_index;
  }
  // pair B/E into complete slices
  const slices = [];
  const openSlices = {};
  for (const e of events) {
    if (e.ph === "X") slices.push(e);
    else if (e.ph === "B") openSlices[e.pid+":"+e.tid+":"+e.name] = e;
    else if (e.ph === "E") {
      const key = e.pid+":"+e.tid+":"+e.name;
      if (openSlices[key]) {
        const b = openSlices[key];
        slices.push({name:b.name, cat:b.cat, pid:e.pid, tid:e.tid,
                     ts:b.ts, dur:Math.max(e.ts-b.ts,0.01), args:b.args||{}});
        delete openSlices[key];
      }
    }
  }
  const instants = events.filter(e => e.ph === "i");
  const counters = events.filter(e => e.ph === "C");
  const flows = events.filter(e => e.ph === "s" || e.ph === "t" || e.ph === "f");
  // row set: every (pid,tid) that has a thread_name
  const laneKeys = Object.keys(threadName);
  // sort processes by (sort_index, pid) then threads by (thread sort, tid)
  function pSort(pid){ return [procSort[pid] ?? 900, pid]; }
  function tSort(k){ return [threadSort[k] ?? 900, parseInt(k.split(":")[1])]; }
  const pidsOrdered = [...new Set(laneKeys.map(k => k.split(":")[0]))]
      .sort((a,b) => pSort(+a) < pSort(+b) ? -1 : pSort(+a) > pSort(+b) ? 1 : 0);
  let lanes = [];
  for (const pid of pidsOrdered) {
    const ks = laneKeys.filter(k => k.split(":")[0] === pid).sort((a,b)=> {
      const sa = tSort(a), sb = tSort(b);
      return sa < sb ? -1 : sa > sb ? 1 : 0;
    });
    lanes.push(...ks);
  }
  const slicesByLane = {};
  for (const s of slices) (slicesByLane[s.pid+":"+s.tid] ||= []).push(s);
  const instByLane = {};
  for (const e of instants) (instByLane[e.pid+":"+e.tid] ||= []).push(e);
  const countersByLane = {};
  for (const c of counters) {
    const k = c.pid+":"+c.tid;
    (countersByLane[k] ||= {})[c.name] = (countersByLane[k][c.name] || []);
    countersByLane[k][c.name].push({ts:c.ts, v:c.args[c.name], unit:c.args.unit||""});
  }
  const allTs = slices.map(s => s.ts).concat(instants.map(i=>i.ts), counters.map(c=>c.ts));
  const minTs = Math.min(...allTs, 0);
  const maxTs = Math.max(...slices.map(s => s.ts+(s.dur||0)),
                        ...instants.map(i=>i.ts), ...counters.map(c=>c.ts), 1);
  const range = Math.max(maxTs - minTs, 0.001);
  const chart = document.getElementById("chart");
  const barH = 14, rowH = 22, counterH = 40, leftPad = 200;
  const colors = {BOA:"#e74c3c", EVU:"#27ae60", MFE:"#3498db", USE:"#f39c12",
                   UCE:"#9b59b6", DMA:"#1abc9c", Task:"#e67e22",
                   Stream:"#95a5a6", TileRole:"#2ecc71",
                   "HBM → L2 Input":"#3498db", "L2 → HBM Output":"#e67e22",
                   Collective:"#d35400", "Memory:HBM":"#8e44ad",
                   "Memory:L2":"#2980b9", "Memory:L1":"#16a085",
                   "Scheduler:L2":"#c0392b", hbm_read:"#8e44ad",
                   hbm_write:"#9b59b6", global_dma:"#16a085",
                   noc_request:"#e67e22", noc_response:"#f39c12",
                   l2_read:"#2980b9", l2_write:"#3498db",
                   local_dma:"#1abc9c", l1_read:"#16a085", l1_write:"#2ecc71",
                   default:"#888"};
  const tooltip = document.getElementById("tooltip");
  function showTip(text, x, y) {
    tooltip.innerHTML = text.replace(/\\n/g, "<br>");
    tooltip.style.display = "block";
    tooltip.style.left = (x+12)+"px"; tooltip.style.top = (y+12)+"px";
  }
  let y = 4;
  const laneY = {};
  function xFor(ts){ return leftPad + ((ts - minTs)/range)*(chart.clientWidth - leftPad - 10); }
  for (const k of lanes) {
    const [pidStr, tidStr] = k.split(":");
    const pid = parseInt(pidStr);
    const pn = procName[pid] || ("proc "+pid);
    const tn = threadName[k] || ("thread "+tidStr);
    const lbl = document.createElement("div");
    lbl.className = "axis"; lbl.style.left="4px"; lbl.style.top=(y+2)+"px";
    lbl.style.width=(leftPad-8)+"px"; lbl.textContent = pn+" / "+tn;
    chart.appendChild(lbl);
    laneY[k] = y;
    for (const s of (slicesByLane[k]||[])) {
      const x = xFor(s.ts);
      const w = Math.max(((s.dur||0)/range)*(chart.clientWidth-leftPad-10), 1);
      const bar = document.createElement("div");
      bar.className="bar"; bar.style.left=x+"px"; bar.style.top=y+"px";
      bar.style.width=w+"px";
      bar.style.background = colors[s.name] || colors[s.cat] || colors.default;
      bar.textContent = s.name;
      bar.dataset.flow = (s.args && s.args.flow_id != null) ? s.args.flow_id : "";
      chart.appendChild(bar);
      const sRef = s;
      bar.addEventListener("mouseenter", (ev) => {
        let txt = sRef.name + "  " + sRef.ts.toFixed(1) + "µs  dur=" + (sRef.dur||0).toFixed(1) + "µs";
        for (const [kk,vv] of Object.entries(sRef.args||{})) txt += "\\n  "+kk+": "+vv;
        showTip(txt, ev.clientX, ev.clientY);
      });
      bar.addEventListener("mouseleave", () => tooltip.style.display="none");
    }
    for (const e of (instByLane[k]||[])) {
      const x = xFor(e.ts) - 3;
      const m = document.createElement("div");
      m.className="inst"; m.style.left=x+"px"; m.style.top=(y+4)+"px";
      m.addEventListener("mouseenter", (ev) => {
        let txt = e.name + "  " + e.ts.toFixed(1)+"µs";
        for (const [kk,vv] of Object.entries(e.args||{})) txt += "\\n  "+kk+": "+vv;
        showTip(txt, ev.clientX, ev.clientY);
      });
      m.addEventListener("mouseleave", () => tooltip.style.display="none");
      chart.appendChild(m);
    }
    y += rowH;
    // counter rows for this lane
    const cset = countersByLane[k];
    if (cset) {
      for (const cname of Object.keys(cset).sort()) {
        const samples = cset[cname];
        const values = samples.map(s => s.v);
        const cmin = Math.min(...values, 0), cmax = Math.max(...values, 1);
        const crange = Math.max(cmax - cmin, 1e-9);
        const points = samples.map(s =>
          xFor(s.ts).toFixed(1) + "," + (y + counterH - 4 - ((s.v-cmin)/crange)*(counterH-8)).toFixed(1)
        ).join(" ");
        const svg = document.createElementNS("http://www.w3.org/2000/svg","svg");
        svg.setAttribute("class","flow-line");
        svg.style.left="0px"; svg.style.top="0px";
        svg.setAttribute("width", chart.clientWidth); svg.setAttribute("height", y+counterH);
        const pol = document.createElementNS("http://www.w3.org/2000/svg","polyline");
        pol.setAttribute("points", points);
        pol.setAttribute("fill","none");
        pol.setAttribute("stroke", "#3498db");
        pol.setAttribute("stroke-width","1.2");
        svg.appendChild(pol);
        chart.appendChild(svg);
        const clbl = document.createElement("div");
        clbl.className="axis"; clbl.style.left="4px"; clbl.style.top=(y+2)+"px";
        clbl.style.width=(leftPad-8)+"px";
        clbl.textContent = pn+"/"+tn+" · "+cname;
        chart.appendChild(clbl);
        y += counterH;
      }
    }
  }
  // flow highlight: click a bar → highlight same flow_id bars + draw connectors
  let activeFlow = null;
  function clearFlow() {
    activeFlow = null;
    document.querySelectorAll(".bar.flow-hi").forEach(b => b.classList.remove("flow-hi"));
    document.querySelectorAll("svg.flow-conn").forEach(s => s.remove());
  }
  chart.addEventListener("click", (ev) => {
    const bar = ev.target.closest(".bar");
    if (!bar || !bar.dataset.flow) { clearFlow(); return; }
    clearFlow();
    activeFlow = bar.dataset.flow;
    const bars = [...document.querySelectorAll(".bar")].filter(b => b.dataset.flow === activeFlow);
    bars.sort((a,b) => parseFloat(a.style.left) - parseFloat(b.style.left));
    const conn = document.createElementNS("http://www.w3.org/2000/svg","svg");
    conn.setAttribute("class","flow-conn flow-line");
    conn.style.left="0px"; conn.style.top="0px";
    conn.setAttribute("width", chart.clientWidth); conn.setAttribute("height", y);
    for (let i=0;i<bars.length;i++){
      bars[i].classList.add("flow-hi");
      if (i+1<bars.length){
        const a = bars[i].getBoundingClientRect(), b = bars[i+1].getBoundingClientRect();
        const cc = chart.getBoundingClientRect();
        const x1 = a.right - cc.left, y1 = a.top + a.height/2 - cc.top;
        const x2 = b.left - cc.left, y2 = b.top + b.height/2 - cc.top;
        const ln = document.createElementNS("http://www.w3.org/2000/svg","path");
        ln.setAttribute("d", `M${x1},${y1} L${x2},${y2}`);
        ln.setAttribute("stroke","#fff200"); ln.setAttribute("stroke-width","1.4");
        ln.setAttribute("fill","none");
        conn.appendChild(ln);
      }
    }
    chart.appendChild(conn);
  });
  // time axis
  const axisDiv = document.createElement("div");
  axisDiv.style.position="absolute"; axisDiv.style.left=leftPad+"px";
  axisDiv.style.top=(y+4)+"px"; axisDiv.style.width="100%";
  axisDiv.className="axis";
  axisDiv.textContent = "0µs".padEnd(20) + " ... " + maxTs.toFixed(0) + "µs";
  chart.appendChild(axisDiv);
})();
</script>
</body>
</html>
"""


def trace_to_html(tracer: Tracer) -> str:
  """Wrap the Chrome trace JSON in a standalone HTML page with a Gantt chart."""
  json_str = tracer.to_chrome_json()
  return _HTML_TEMPLATE.replace("__TRACE_JSON__", json_str)
