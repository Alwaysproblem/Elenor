"""Compute Tile and Tile UCE controller.

The Compute Tile is the kernel execution domain
(design/elenor_compute_tile/ELENOR_Compute_Tile_Design.md).  It contains:

  - Tile UCE (control):     interprets a Tile Program cycle by cycle.
  - BOA / EVU / MFE / USE:  engine instances.
  - L1 SRAM:                modeled as a bandwidth budget (bank conflicts
                            optional; V1 leaves them frozen).

The Tile UCE is the heart of the Tile-SPMD model.  It fetches one
instruction per cycle, launches engines, waits on events, and moves
stream tokens.  The stream-pipeline tile program (pop -> acquire ->
launch -> wait -> push -> release) drives the producer-consumer pipeline.

Event model:  `launch.*` produces an event id (str).  `wait`/`waitall`
blocks the UCE until the named event(s) complete.  While blocked the
UCE records a WAIT_EVENT stall so PMU can attribute it.

V2: For non-stream straight-line programs, the UCE lowers the TileProgram
into a tile-local task list with Shared Lightweight Command Buffer and
per-engine FIFOs.  `wait`/`waitall` become dependency edges rather than
blocking the PC; the UCE scheduler issues ready tasks subject to dep
counters and engine queue backpressure.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from .config import HardwareConfig
from .engines import BOAEngine, Engine, EngineJob, EngineState, EVUEngine, MFEEngine, USEEngine
from .ir import EngineDesc, TileInst, TileOp, TileProgram
from .memory.l1_slot_frame import SlotFrame
from .pmu import PMUCounter, StallReason
from .stream_queue import StreamQueue, StreamToken
from .trace import Tracer

# ---------------------------------------------------------------------------
# Private types for PC interpreter
# ---------------------------------------------------------------------------


@dataclass
class _PendingWait:
    """A UCE blocked waiting on event(s)."""
    events: tuple
    all: bool  # True=waitall, False=wait single
    started_cycle: int


# ---------------------------------------------------------------------------
# Private types for task-list scheduler (V2)
# ---------------------------------------------------------------------------


class _TileTaskStatus(Enum):
    """Tile-local task entry state in the task-list scheduler."""
    PENDING = "pending"
    READY = "ready"
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"


@dataclass
class _TileTaskEntry:
    """One lowered task entry in the Shared Lightweight Command Buffer."""
    cmd_tag: int
    inst: TileInst
    deps: tuple[str, ...]
    dep_counter: int
    state: _TileTaskStatus = _TileTaskStatus.PENDING
    engine_kind: str | None = None
    event_id: str | None = None


@dataclass
class _EngineQueueEntry:
    """One entry in a per-engine execution FIFO (pointer/tag only).

    ``engine_kind`` is the *physical* engine ("BOA"/"EVU"/"MFE"/"USE")
    the entry drains into; both MFE_LOAD and MFE_STORE map to "MFE".
    """
    cmd_tag: int
    event_id: str
    desc_ref: str
    op: TileOp
    engine_kind: str = "MFE"


# ---------------------------------------------------------------------------
# Task-list lowering: ops allowed in the non-stream scheduler
# ---------------------------------------------------------------------------

_ALLOWED_TASK_LIST_OPS = frozenset({
    TileOp.NOP,
    TileOp.RET,
    TileOp.WAIT,
    TileOp.WAITALL,
    TileOp.FENCE,
    TileOp.LAUNCH_BOA,
    TileOp.LAUNCH_EVU,
    TileOp.LAUNCH_MFE,
    TileOp.LAUNCH_USE,
    TileOp.LAUNCH_DMA_LOAD,
    TileOp.LAUNCH_DMA_STORE,
    TileOp.PATCH_DESC,
    TileOp.PROF_BEGIN,
    TileOp.PROF_END,
})

_ENGINE_KIND_MAP: dict[TileOp, str] = {
    TileOp.LAUNCH_BOA: "BOA",
    TileOp.LAUNCH_EVU: "EVU",
    TileOp.LAUNCH_MFE: "MFE",
    TileOp.LAUNCH_USE: "USE",
    TileOp.LAUNCH_DMA_LOAD: "MFE",
    TileOp.LAUNCH_DMA_STORE: "MFE",
}


# ===========================================================================
# Tile UCE
# ===========================================================================


class TileUCE:
    """Tile Unified Control Engine.

    Issues one instruction per cycle when not blocked.  Tracks outstanding
    engine events and stream-token registers.

    V2: For non-stream straight-line TilePrograms the UCE lowers the program
    into a tile-local task list.  `wait`/`waitall` form dependency edges
    (dep counters) rather than blocking the PC; the scheduler issues ready
    tasks subject to per-engine FIFO backpressure.  Stream/branch programs
    continue to use the legacy PC interpreter.
    """

    def __init__(self,
                 tile_id: int,
                 cfg: HardwareConfig,
                 tracer: Tracer | None = None):
        self.tile_id = tile_id
        self.cfg = cfg
        self.tracer = tracer
        self.pmu = PMUCounter()
        self.pc = 0
        self.program: TileProgram | None = None
        self._events_done: set[str] = set()
        self._pending: _PendingWait | None = None
        # token registers: name -> StreamToken
        self._tokens: dict[str, StreamToken] = {}
        # event -> StreamQueue id (for stream-token-based events)
        self._event_meta: dict[str, dict] = {}
        # runtime fidelity: optional callback fired on engine event completion
        # so the group-level EventTable can track sequence/status (P0-4).
        self._event_done_callback = None
        self.done = False
        self.faulted: bool = False
        self.fault_reason: str = ""

        # ---- V2 task-list scheduler state ----
        self._task_mode: bool = False
        self._task_entries: list[_TileTaskEntry] = []
        self._ready_tasks: list[int] = []  # sorted cmd_tags
        self._dependents: dict[str, list[int]] = {}
        self._event_producer: dict[str, int] = {}
        self._running_events: dict[str, int] = {}  # event_id -> cmd_tag
        self._command_buffer: dict[int, _TileTaskEntry] = {}
        # UCE-side ingress FIFOs. MFE ingress is split into load/store so
        # each can be sized independently; both still drain into the single
        # physical MFEEngine (cfg.mfe_pipeline_depth limits accept depth).
        self._engine_queues: dict[str, list[_EngineQueueEntry]] = {
            "BOA": [], "EVU": [], "MFE_LOAD": [], "MFE_STORE": [], "USE": [],
        }
        self._engine_queue_depths: dict[str, int] = {
            "BOA": 1,
            "EVU": 1,
            "USE": 1,
            "MFE_LOAD": cfg.mfe_load_queue_depth,
            "MFE_STORE": cfg.mfe_store_queue_depth,
        }
        self._completion_queue: deque[tuple[str, bool]] = deque()

    def load(self, program: TileProgram) -> None:
        self.program = program
        self.pc = 0
        self._events_done.clear()
        self._pending = None
        self._tokens.clear()
        self._event_meta.clear()
        self.done = False
        self.faulted = False
        self.fault_reason = ""

        # Reset task-list state
        self._task_mode = False
        self._task_entries.clear()
        self._ready_tasks.clear()
        self._dependents.clear()
        self._event_producer.clear()
        self._running_events.clear()
        self._command_buffer.clear()
        self._engine_queues = {
            "BOA": [], "EVU": [], "MFE_LOAD": [], "MFE_STORE": [], "USE": [],
        }
        self._completion_queue.clear()

        # Try task-list lowering for non-stream straight-line programs
        if self._can_lower_to_task_list(program):
            self._task_mode = True
            self._lower_to_task_list(program)

    # ------------------------------------------------------------------
    # Per-cycle step
    # ------------------------------------------------------------------
    def step(self, cycle: int, tile: ComputeTile) -> str | None:
        """Execute one cycle of the UCE.  Returns an event id if an engine
        just completed that the UCE was *not* blocking on (rare), else None.

        The UCE returns 'done' when the program returns.
        """
        # Drain completion queue first so _events_done is current
        # for the legacy PC interpreter and late callbacks fire
        # even when UCE is already done.
        completion_drained = self._drain_completion_queue()

        if self.done or self.program is None:
            if completion_drained:
                self.pmu.add_cycle("total", 1)
            else:
                self.pmu.add_cycle("idle", 1)
                self.pmu.add_cycle("total", 1)
            return None

        if self._task_mode:
            return self._step_task_scheduler(cycle, tile,
                                             completion_drained=completion_drained)

        # ---- legacy PC interpreter ----
        # If blocked on a wait, check completion.
        if self._pending is not None:
            completed = [
                e for e in self._pending.events if e in self._events_done
            ]
            satisfied = (len(completed) == len(self._pending.events)
                         if self._pending.all else len(completed) > 0)
            if satisfied:
                self._pending = None
                # fall through to issue next inst this cycle?  For simplicity,
                # consume this cycle as the wait-resolve and advance next.
                self.pmu.add(StallReason.WAIT_EVENT,
                             0)  # resolved, no stall counted
                self.pmu.add_cycle("wait_resolved", 1)
                self.pmu.add_cycle("total", 1)
                return None
            else:
                self.pmu.add(StallReason.WAIT_EVENT, 1)
                self.pmu.add_cycle("wait_event", 1)
                self.pmu.add_cycle("total", 1)
                return None

        # Fetch + issue one instruction.
        if self.pc >= len(self.program.insts):
            self.done = True
            self.pmu.add_cycle("total", 1)
            return "done"

        ins = self.program.insts[self.pc]
        self.pmu.add_cycle("total", 1)
        self._issue(ins, cycle, tile)
        return None

    # ------------------------------------------------------------------
    # Instruction issue  (legacy PC interpreter)
    # ------------------------------------------------------------------

    def _issue(self, ins: TileInst, cycle: int, tile: ComputeTile) -> None:
        op = ins.op
        assert self.program is not None, "UCE has no TileProgram loaded"
        prog = self.program
        if op == TileOp.NOP:
            self.pc += 1
        elif op == TileOp.RET:
            self.done = True
            self.pmu.add_event("tile_done")
            self.pc += 1
        elif op == TileOp.MOV:
            self.pc += 1
        elif op == TileOp.BR:
            self.pc = prog.label_index(ins.args[0])
        elif op == TileOp.BR_EOS:
            tok = self._tokens.get(ins.args[0])
            if tok is not None and tok.is_eos:
                self.pc = prog.label_index(ins.args[1])
            else:
                self.pc += 1
        elif op == TileOp.WAIT:
            ev = ins.args[0]
            self._pending = _PendingWait(events=(ev, ),
                                         all=False,
                                         started_cycle=cycle)
            self.pc += 1
        elif op == TileOp.WAITALL:
            evs = tuple(ins.args)
            self._pending = _PendingWait(events=evs,
                                         all=True,
                                         started_cycle=cycle)
            self.pc += 1
        elif op == TileOp.FENCE:
            # one-cycle fence
            self.pmu.add_cycle("fence", 1)
            self.pc += 1
        elif op == TileOp.LAUNCH_BOA:
            if self._launch_engine(tile.boa, ins, cycle, tile):
                self.pc += 1
        elif op == TileOp.LAUNCH_EVU:
            if self._launch_engine(tile.evu, ins, cycle, tile):
                self.pc += 1
        elif op == TileOp.LAUNCH_MFE:
            if self._launch_engine(tile.mfe, ins, cycle, tile):
                self.pc += 1
        elif op == TileOp.LAUNCH_USE:
            if self._launch_engine(tile.use, ins, cycle, tile):
                self.pc += 1
        elif op == TileOp.LAUNCH_DMA_LOAD:
            if self._launch_dma(ins, cycle, tile, store=False):
                self.pc += 1
        elif op == TileOp.LAUNCH_DMA_STORE:
            if self._launch_dma(ins, cycle, tile, store=True):
                self.pc += 1
        elif op == TileOp.STREAM_POP:
            qid = ins.args[0]
            q = tile.get_stream(qid)
            tok = q.pop(cycle)
            if tok is not None:
                assert ins.dst is not None
                self._tokens[ins.dst] = tok
                self.pmu.add_event("stream_pop")
                self.pc += 1
            else:
                self.pmu.add(StallReason.STREAM_CREDIT, 1)
                self.pmu.add_cycle("stream_empty", 1)
        elif op == TileOp.STREAM_ACQUIRE:
            qid = ins.args[0]
            if qid < 0:
                self.pc += 1  # sink tile: no output stream
            else:
                q = tile.get_stream(qid)
                if q.acquire(cycle):
                    assert ins.dst is not None
                    self._tokens[ins.dst] = StreamToken(
                        token_id=-1, producer_id=tile.tile_id)
                    self.pc += 1
                else:
                    self.pmu.add(StallReason.STREAM_CREDIT, 1)
                    self.pmu.add_cycle("stream_full", 1)
        elif op == TileOp.STREAM_PUSH:
            qid, tok_reg, producer_id = ins.args
            if qid < 0:
                self.pc += 1  # sink tile: no output stream
            else:
                q = tile.get_stream(qid)
                tok = self._tokens.get(tok_reg)
                if tok is None:
                    tok = StreamToken(token_id=q._next_token_id,
                                      producer_id=producer_id)
                tok.producer_id = producer_id
                tok.token_id = q._next_token_id
                q._next_token_id += 1
                q.push(tok, cycle)
                self.pmu.add_event("stream_push")
                self.pc += 1
        elif op == TileOp.STREAM_RELEASE:
            qid, tok_reg = ins.args
            if qid >= 0:
                q = tile.get_stream(qid)
                tok = self._tokens.get(tok_reg)
                if tok is not None:
                    q.release(tok, cycle)
                    del self._tokens[tok_reg]
                    self.pmu.add_event("stream_release")
            self.pc += 1
        elif op == TileOp.STREAM_PUSH_EOS:
            qid, producer_id = ins.args
            if qid >= 0:
                q = tile.get_stream(qid)
                q.push_eos(producer_id, cycle)
                self.pmu.add_event("stream_eos")
            self.pc += 1
        elif op == TileOp.PATCH_DESC:
            self.pmu.add_cycle("patch_desc", 1)
            self.pc += 1
        elif op == TileOp.PROF_BEGIN or op == TileOp.PROF_END:
            self.pc += 1
        else:
            self.pc += 1

    # ------------------------------------------------------------------
    # Engine launch helpers  (shared by PC and task-list paths)
    # ------------------------------------------------------------------

    def _fault(self, reason: str) -> None:
        """Mark the UCE as faulted.  Sets done so the scheduler stops
        dispatching; the group propagates this to the sequencer fault
        path so ``SimResult`` reports ``completed=False``.
        """
        self.faulted = True
        self.fault_reason = reason
        self.done = True
        self.pmu.add_event("tile_fault")

    def _launch_engine(self, engine: Engine, ins: TileInst, cycle: int,
                       tile: ComputeTile) -> bool:
        """Launch an engine descriptor.  Returns False (UCE blocks, no PC
        advance) when the engine is still busy with a prior job.

        A ``ValueError`` from the engine (e.g. MFE stream-buffer prefetch
        capacity validation) is converted into a modeled tile fault: the
        instruction is consumed (returns True) but no event is registered.
        """
        if engine.is_busy:
            self.pmu.add(StallReason.WAIT_OPERAND, 1)
            self.pmu.add_cycle("engine_busy", 1)
            return False
        assert self.program is not None
        dname = ins.args[0]
        desc = self.program.descriptors[dname]
        # patch in tile_id for SPMD differentiation
        desc = EngineDesc(name=desc.name,
                          kind=desc.kind,
                          op=desc.op,
                          params={
                              **desc.params, "tile_id": self.tile_id
                          })
        assert ins.dst is not None
        try:
            job = engine.launch(desc, cycle, ins.dst)
        except ValueError as exc:
            self._fault(str(exc))
            return True
        if job is None:
            self.pmu.add(StallReason.WAIT_OPERAND, 1)
            self.pmu.add_cycle("engine_busy", 1)
            return False
        self._event_meta[ins.dst] = {"engine": engine.kind, "job": job}
        return True

    def _launch_dma(self, ins: TileInst, cycle: int, tile: ComputeTile,
                    store: bool) -> bool:
        """Model DMA load/store as an MFE descriptor.  Blocks if MFE busy."""
        if tile.mfe.is_busy:
            self.pmu.add(StallReason.WAIT_OPERAND, 1)
            self.pmu.add_cycle("engine_busy", 1)
            return False
        nbytes = ins.args[1] if len(ins.args) > 1 else 4096
        desc = EngineDesc(name="dma",
                          kind="MFE",
                          op="dma_load" if not store else "dma_store",
                          params={"bytes": nbytes, "ops": 0})
        assert ins.dst is not None
        try:
            job = tile.mfe.launch(desc, cycle, ins.dst)
        except ValueError as exc:
            self._fault(str(exc))
            return True
        if job is None:
            self.pmu.add(StallReason.WAIT_OPERAND, 1)
            return False
        self._event_meta[ins.dst] = {"engine": "DMA", "job": job}
        return True

    # ------------------------------------------------------------------
    # Event notification  (shared, V2 adds dep-counter update)
    # ------------------------------------------------------------------

    def notify_event(self, event_id: str) -> None:
        """Enqueue a completion event for processing in the next UCE step.

        The actual dependency release and ready-queue update happen in
        ``_drain_completion_queue()``, called at the top of ``step()``.
        """
        self._completion_queue.append((event_id, True))

    def _complete_event(self, event_id: str, external: bool = False) -> None:
        """Record an event as done and update dependent dep counters.

        ``external=True`` → engine/device completion; also fires the
        group-level ``_event_done_callback`` so the TileGroup EventTable
        stays consistent.

        ``external=False`` → synthetic control event (PATCH_DESC, NOP, …);
        only updates tile-local dependents and does NOT expose this event
        to the group EventTable.
        """
        # Duplicate completion: already consumed
        if event_id in self._events_done:
            return
        self._events_done.add(event_id)

        # Mark producer task as DONE (task-list mode)
        if self._task_mode and event_id in self._running_events:
            cmd_tag = self._running_events.pop(event_id)
            entry = self._command_buffer.get(cmd_tag)
            if entry is not None:
                entry.state = _TileTaskStatus.DONE

        # Decrement dependent dep counters
        if event_id in self._dependents:
            for cmd_tag in self._dependents[event_id]:
                entry = self._command_buffer.get(cmd_tag)
                if entry is not None and entry.dep_counter > 0:
                    entry.dep_counter -= 1
                    if entry.dep_counter == 0 and entry.state == _TileTaskStatus.PENDING:
                        entry.state = _TileTaskStatus.READY
                        self._ready_tasks.append(cmd_tag)

        # External events propagate to group-level callback
        if external and self._event_done_callback is not None:
            self._event_done_callback(event_id)

    def _drain_completion_queue(self) -> bool:
        """Process all pending completions from the completion queue.

        Returns True if at least one completion was processed.
        """
        drained = False
        while self._completion_queue:
            event_id, external = self._completion_queue.popleft()
            self._complete_event(event_id, external=external)
            drained = True
        return drained

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.pc = 0
        self._events_done.clear()
        self._pending = None
        self._tokens.clear()
        self._event_meta.clear()
        self.done = False
        self.faulted = False
        self.fault_reason = ""
        self.pmu.reset()

        # Reset task-list state
        self._task_mode = False
        self._task_entries.clear()
        self._ready_tasks.clear()
        self._dependents.clear()
        self._event_producer.clear()
        self._running_events.clear()
        self._command_buffer.clear()
        self._engine_queues = {
            "BOA": [], "EVU": [], "MFE_LOAD": [], "MFE_STORE": [], "USE": [],
        }
        self._completion_queue.clear()

    # ==================================================================
    # V2: Task-list scheduler  (private)
    # ==================================================================

    def _can_lower_to_task_list(self, program: TileProgram) -> bool:
        """Return True iff every op in *program* is allowed in the
        non-stream task-list scheduler."""
        for ins in program.insts:
            if ins.op not in _ALLOWED_TASK_LIST_OPS:
                return False
        return True

    def _lower_to_task_list(self, program: TileProgram) -> None:
        """Lower a straight-line TileProgram into a tile-local task list.

        Produces ``_TileTaskEntry`` objects stored in ``_command_buffer``,
        initialises ``_dependents`` and ``_event_producer``, and populates
        the initial ``_ready_tasks`` list.

        Raises ``ValueError`` for:
          - duplicate event ids
          - wait referencing an unknown event
        """
        active_deps: set[str] = set()
        known_events: set[str] = set()
        all_engine_events: list[str] = []

        for cmd_tag, ins in enumerate(program.insts):
            op = ins.op

            # ---- wait / waitall: accumulate active deps, no task ----
            if op in (TileOp.WAIT, TileOp.WAITALL):
                evs = (tuple(ins.args) if op == TileOp.WAITALL
                       else (ins.args[0],) if ins.args else ())
                for ev in evs:
                    if ev not in known_events:
                        # External (cross-level) event — produced by the
                        # group sequencer (DMA completion forwarded via
                        # the TileGroup event bridge) rather than by an
                        # engine launch in this program.  Only sequencer
                        # DMA events (``ev_dma_*``) are bridged today, so
                        # restrict acceptance to that prefix to keep
                        # catching typos / unknown tile-local events.
                        if not ev.startswith("ev_dma_"):
                            raise ValueError(
                                f"wait references unknown event {ev}")
                        known_events.add(ev)
                active_deps.update(evs)
                continue

            # ---- determine engine kind and event id ----
            engine_kind = _ENGINE_KIND_MAP.get(op, None)
            event_id: str | None = ins.dst

            is_control = engine_kind is None
            if is_control and op not in (TileOp.RET,):
                event_id = f"__uce_ctrl_{cmd_tag}"
            # ---- duplicate check ----
            if event_id is not None:
                if event_id in known_events:
                    raise ValueError(
                        f"duplicate tile event {event_id}")
                known_events.add(event_id)

            # ---- FENCE: add all engine events to active deps ----
            if op == TileOp.FENCE:
                active_deps.update(all_engine_events)

            # ---- build deps from current active set ----
            deps = tuple(sorted(active_deps))
            dep_counter = len(deps)

            entry = _TileTaskEntry(
                cmd_tag=cmd_tag,
                inst=ins,
                deps=deps,
                dep_counter=dep_counter,
                state=(_TileTaskStatus.READY if dep_counter == 0
                       else _TileTaskStatus.PENDING),
                engine_kind=engine_kind,
                event_id=event_id,
            )

            self._task_entries.append(entry)
            self._command_buffer[cmd_tag] = entry

            # Register dependents
            for dep in deps:
                self._dependents.setdefault(dep, []).append(cmd_tag)

            # Register event producer
            if event_id is not None:
                self._event_producer[event_id] = cmd_tag

            # Track engine events (for FENCE and RET)
            if engine_kind is not None:
                assert event_id is not None, "engine task must have event_id"
                all_engine_events.append(event_id)

            # Control tasks: add synthetic event to active deps
            if is_control and op not in (TileOp.RET,) and event_id is not None:
                active_deps.add(event_id)
            # ---- RET: expand deps to active set union all engine events ----
            if op == TileOp.RET:
                ret_deps = tuple(sorted(set(deps) | set(all_engine_events)))
                entry.deps = ret_deps
                entry.dep_counter = len(ret_deps)
                # Register dependents for additional deps
                for dep in ret_deps:
                    if dep not in deps or cmd_tag not in self._dependents.get(dep, []):
                        self._dependents.setdefault(dep, []).append(cmd_tag)
                entry.state = (_TileTaskStatus.READY if entry.dep_counter == 0
                               else _TileTaskStatus.PENDING)

            if entry.state == _TileTaskStatus.READY:
                self._ready_tasks.append(cmd_tag)

    # ------------------------------------------------------------------
    # Scheduler step
    # ------------------------------------------------------------------

    def _classify_queue_key(self, entry: _TileTaskEntry) -> str:
        """Map a ready engine task to its UCE-side ingress FIFO key.

        MFE ingress is split: LAUNCH_MFE inspects the referenced
        descriptor's ``op`` (store-class → MFE_STORE, else MFE_LOAD);
        LAUNCH_DMA_LOAD → MFE_LOAD, LAUNCH_DMA_STORE → MFE_STORE.
        BOA/EVU/USE keep their engine kind as the key.
        """
        op = entry.inst.op
        if op == TileOp.LAUNCH_DMA_LOAD:
            return "MFE_LOAD"
        if op == TileOp.LAUNCH_DMA_STORE:
            return "MFE_STORE"
        if op == TileOp.LAUNCH_MFE:
            desc_ref = (entry.inst.args[0]
                        if entry.inst.args else "")
            assert self.program is not None
            d = self.program.descriptors.get(desc_ref)
            if d is not None and d.op in ("store", "dma_store"):
                return "MFE_STORE"
            return "MFE_LOAD"
        return entry.engine_kind or "BOA"

    def _step_task_scheduler(self, cycle: int, tile: ComputeTile,
                              completion_drained: bool = False) -> str | None:
        """One cycle of the non-blocking task-list scheduler.

        Phases:
          0. Completion drain (if not already done at top of step()).
          1. Drain engine queues (launch FIFO heads to engines).
          2. Dispatch ready tasks to per-engine FIFOs (up to
             ``uce_dispatch_per_cycle`` times).
          3. Drain again so just-queued tasks may enter engines
             in the same cycle.
          4. Attribute stall / idle.
        """
        did_work = completion_drained

        # Phase 1: drain engine queues
        did_work |= self._drain_engine_queues(cycle, tile)

        # Keep _ready_tasks sorted by cmd_tag (completions may append
        # out of order).
        self._ready_tasks.sort()

        # Phase 2: dispatch ready tasks
        for _ in range(self.cfg.uce_dispatch_per_cycle):
            if not self._ready_tasks:
                break
            issued = False
            for idx, cmd_tag in enumerate(self._ready_tasks):
                entry = self._command_buffer.get(cmd_tag)
                if entry is None or entry.state != _TileTaskStatus.READY:
                    continue

                if entry.engine_kind is None:
                    # Control task: execute immediately, no FIFO
                    self._execute_control_task(entry, cycle)
                    del self._ready_tasks[idx]
                    did_work = True
                    issued = True
                    # Drain any synthetic control completions so
                    # subsequent dispatch iterations see them.
                    did_work |= self._drain_completion_queue()
                    self._ready_tasks.sort()
                    break
                else:
                    queue_key = self._classify_queue_key(entry)
                    fifo = self._engine_queues[queue_key]
                    if len(fifo) < self._engine_queue_depths[queue_key]:
                        fifo.append(_EngineQueueEntry(
                            cmd_tag=cmd_tag,
                            event_id=entry.event_id or "",
                            desc_ref=(entry.inst.args[0]
                                      if entry.inst.args else ""),
                            op=entry.inst.op,
                            engine_kind=entry.engine_kind or "MFE",
                        ))
                        entry.state = _TileTaskStatus.QUEUED
                        del self._ready_tasks[idx]
                        did_work = True
                        issued = True
                        break
                    # FIFO full → skip, try next ready task
            if not issued and self._ready_tasks:
                # All ready tasks are engine tasks whose target FIFOs
                # are full, or there are no ready control tasks.
                break

        # Phase 3: drain again (just-queued tasks may enter engines
        # this same cycle)
        did_work |= self._drain_engine_queues(cycle, tile)
        # Phase 4: stall / idle attribution
        self.pmu.add_cycle("total", 1)

        if did_work:
            # work was done: no stall to attribute
            pass
        elif self._ready_tasks:
            self.pmu.add(StallReason.WAIT_OPERAND, 1)
            self.pmu.add_cycle("engine_queue_full", 1)
        elif any(e.state in (_TileTaskStatus.PENDING,
                             _TileTaskStatus.QUEUED,
                             _TileTaskStatus.RUNNING)
                 for e in self._command_buffer.values()):
            self.pmu.add(StallReason.WAIT_EVENT, 1)
            self.pmu.add_cycle("wait_event", 1)
        elif self._running_events:
            self.pmu.add(StallReason.WAIT_EVENT, 1)
            self.pmu.add_cycle("wait_event", 1)
        else:
            self.pmu.add_cycle("idle", 1)

        # Auto-done: no RET in program AND all work drained
        if not self.done and self.program is not None:
            has_ret = any(e.inst.op == TileOp.RET
                          for e in self._command_buffer.values())
            if not has_ret and not self._running_events:
                all_done = all(
                    e.state == _TileTaskStatus.DONE
                    for e in self._command_buffer.values())
                all_fifos_empty = all(
                    len(f) == 0 for f in self._engine_queues.values())
                if all_done and all_fifos_empty:
                    self.done = True
                    self.pmu.add_event("tile_done")

        return None

    # ------------------------------------------------------------------
    # Engine queue drain
    # ------------------------------------------------------------------

    def _drain_engine_queues(self, cycle: int,
                             tile: ComputeTile) -> bool:
        """Try to launch the head of each per-engine FIFO.

        Iterates in deterministic order
        ``("BOA", "EVU", "MFE_LOAD", "MFE_STORE", "USE")``; both MFE
        ingress queues map to the single physical ``tile.mfe`` so the
        datapath stays serialized and ``cfg.mfe_pipeline_depth`` remains
        the internal accept-depth limiter.

        Returns True if at least one FIFO head was successfully launched.
        """
        did_work = False
        for queue_key in ("BOA", "EVU", "MFE_LOAD", "MFE_STORE", "USE"):
            fifo = self._engine_queues[queue_key]
            if not fifo:
                continue
            head = fifo[0]
            cmd_tag = head.cmd_tag
            entry = self._command_buffer.get(cmd_tag)
            if entry is None:
                fifo.pop(0)
                continue

            ins = entry.inst
            success = False
            if head.op in (TileOp.LAUNCH_DMA_LOAD, TileOp.LAUNCH_DMA_STORE):
                success = self._launch_dma(
                    ins, cycle, tile,
                    store=(head.op == TileOp.LAUNCH_DMA_STORE))
            else:
                engine = getattr(tile, head.engine_kind.lower())
                success = self._launch_engine(engine, ins, cycle, tile)

            if success:
                fifo.pop(0)
                if self.faulted:
                    # Validation fault: consume the entry without
                    # registering an event or marking it running.
                    entry.state = _TileTaskStatus.DONE
                    did_work = True
                    return did_work
                entry.state = _TileTaskStatus.RUNNING
                if entry.event_id:
                    self._running_events[entry.event_id] = cmd_tag
                did_work = True
            # else: engine busy / queue full → keep in FIFO (backpressure)
        return did_work

    # ------------------------------------------------------------------
    # Control task execution
    # ------------------------------------------------------------------

    def _execute_control_task(self, entry: _TileTaskEntry,
                               cycle: int) -> None:
        """Execute a control (non-engine) task in the UCE locally.

        NOP / PROF_BEGIN / PROF_END complete instantly.
        PATCH_DESC consumes one named cycle "patch_desc".
        FENCE consumes one named cycle "fence".
        RET records PMU event "tile_done" and sets ``self.done = True``.
        """
        op = entry.inst.op
        if op in (TileOp.NOP, TileOp.PROF_BEGIN, TileOp.PROF_END):
            entry.state = _TileTaskStatus.DONE
        elif op == TileOp.PATCH_DESC:
            self.pmu.add_cycle("patch_desc", 1)
            entry.state = _TileTaskStatus.DONE
        elif op == TileOp.FENCE:
            self.pmu.add_cycle("fence", 1)
            entry.state = _TileTaskStatus.DONE
        elif op == TileOp.RET:
            self.pmu.add_event("tile_done")
            entry.state = _TileTaskStatus.DONE
            self.done = True
        # Enqueue synthetic control completion for drain in next
        # _drain_completion_queue() call (do NOT expose to group).
        if (entry.event_id is not None
                and entry.event_id.startswith("__uce_ctrl_")):
            self._completion_queue.append((entry.event_id, False))

    # ------------------------------------------------------------------
    # Scheduler observability
    # ------------------------------------------------------------------

    def has_queued_or_running_work(self) -> bool:
        """True when the scheduler has in-flight (QUEUED or RUNNING) tasks."""
        if not self._task_mode:
            return False
        return any(
            e.state in (_TileTaskStatus.QUEUED, _TileTaskStatus.RUNNING)
            for e in self._command_buffer.values())

    def scheduler_snapshot(self) -> dict:
        """Return a fixed-shape public snapshot of the task-list scheduler.

        Tests MUST only assert against this dict — never reach into
        private ``_command_buffer`` or ``_engine_queues`` directly.
        """
        return {
            "completion_queue": [eid for eid, _ext in self._completion_queue],
            "mode": "task_list" if self._task_mode else "pc",
            "ready": sorted(self._ready_tasks),
            "queued": {
                **{
                    k: [e.cmd_tag for e in v]
                    for k, v in self._engine_queues.items()
                },
                # Compatibility aggregate: MFE = MFE_LOAD + MFE_STORE.
                # Existing snapshot readers that only check for queued MFE
                # work keep working; new tests assert the split keys.
                "MFE": [e.cmd_tag for e in self._engine_queues["MFE_LOAD"]]
                       + [e.cmd_tag for e in self._engine_queues["MFE_STORE"]],
            },
            "running": dict(self._running_events),
            "dep_counters": {
                cmd_tag: e.dep_counter
                for cmd_tag, e in self._command_buffer.items()
            },
            "events_done": sorted(self._events_done),
            "tasks": {
                cmd_tag: {
                    "op": e.inst.op.value,
                    "event_id": e.event_id,
                    "state": e.state.value,
                    "dep_counter": e.dep_counter,
                }
                for cmd_tag, e in self._command_buffer.items()
            },
            "command_buffer_size": len(self._command_buffer),
        }


# ===========================================================================
# Compute Tile
# ===========================================================================


class TileTaskState(Enum):
    """Compute Tile task accept FSM (Compute Tile design 3).

    In timing_only fidelity this is bypassed (tiles go straight to RUN).
    In runtime/full_memory fidelity the tile advances through:
      TASK_ACCEPT -> PREPARED_TASK_CHECK -> FRAME_BIND -> PROGRAM_RUN -> DRAIN
    Each pre-RUN state consumes cycles, modelling the runtime overhead that
    the existing validator treats as zero.
    """
    IDLE = 0
    TASK_ACCEPT = 1
    PREPARED_TASK_CHECK = 2
    FRAME_BIND = 3
    PROGRAM_RUN = 4
    DRAIN = 5
    FAULTED = 6
    DONE = 7


class ComputeTile:
    """One Compute Tile: UCE + BOA/EVU/MFE/USE + L1 SRAM + stream ports.

    Holds references to the StreamQueues it participates in (shared with
    the TileGroup).  PMU aggregates the per-engine + UCE + queue PMUs.
    """

    def __init__(self,
                 tile_id: int,
                 cfg: HardwareConfig,
                 tracer: Tracer | None = None,
                 runtime_enabled: bool = False,
                 memory_enabled: bool = False):
        self.tile_id = tile_id
        self.cfg = cfg
        self.tracer = tracer
        self.runtime_enabled = runtime_enabled
        self.memory_enabled = memory_enabled
        self.uce = TileUCE(tile_id, cfg, tracer)
        self.boa = BOAEngine(cfg, tile_id, tracer)
        self.evu = EVUEngine(cfg, tile_id, tracer)
        self.mfe = MFEEngine(cfg, tile_id, tracer)
        self.use = USEEngine(cfg, tile_id, tracer)
        self.streams: dict[int, StreamQueue] = {}
        self.pmu = PMUCounter()
        self.role_id: int | None = None
        # TileTaskState FSM (runtime fidelity only)
        self.task_state: TileTaskState = TileTaskState.IDLE
        self._prepared_check_remaining: int = 0
        self._frame_bind_remaining: int = 0
        # L1 Slot Frame — always instantiated as scratchpad metadata;
        # timing_only does not consume frame-bind cycles but the object
        # is present so snapshots are uniform across fidelities.
        self.l1_frame: SlotFrame = SlotFrame(l1_bytes=cfg.tile_l1_bytes)

    def bind_stream(self, qid: int, q: StreamQueue) -> None:
        self.streams[qid] = q

    def get_stream(self, qid: int) -> StreamQueue:
        return self.streams[qid]

    def load_program(self, program: TileProgram,
                      prepare_cycles: int = 0) -> None:
        self.uce.load(program)
        if self.runtime_enabled:
            # Enter the task FSM: TASK_ACCEPT -> PREPARED -> FRAME_BIND -> RUN.
            # prepare_cycles = cold/warm residency penalty (per-tile).
            self.task_state = TileTaskState.TASK_ACCEPT
            # PREPARED_TASK_CHECK always costs 1 cycle (the id+version+hash+
            # epoch gate); cold miss adds prepare_cycles for install latency.
            # warm path: 1 cycle (gate only); cold path: 1 + prepare_cycles.
            self._prepared_check_remaining = 1 + prepare_cycles
            self._frame_bind_remaining = self.cfg.frame_bind_cycles

    # ---- per-cycle step -------------------------------------------------

    def step(self, cycle: int) -> EngineJob | None:
        """Advance one cycle.  Returns a completed EngineJob if any engine
        finished this cycle (so the TileGroup can fire events).

        In runtime fidelity, the TileTaskState FSM runs first: the tile
        spends cycles in TASK_ACCEPT / PREPARED_TASK_CHECK / FRAME_BIND
        before the UCE starts executing (PROGRAM_RUN).  In timing_only the
        FSM is bypassed (task_state stays IDLE/RUN) and behavior is
        identical to the original validator.
        """
        # tick engines first; collect completions
        completed = None
        for eng in (self.boa, self.evu, self.mfe, self.use):
            job = eng.tick(cycle)
            if job is not None:
                completed = job
                # notify UCE the event is done
                self.uce.notify_event(job.event_id)

        # ---- TileTaskState FSM (runtime fidelity) ----
        if self.runtime_enabled and self.task_state != TileTaskState.PROGRAM_RUN:
            self._advance_task_fsm(cycle)
            self._aggregate_pmu()
            return completed

        # tick UCE (only in PROGRAM_RUN, or always in timing_only)
        if not self.runtime_enabled or self.task_state == TileTaskState.PROGRAM_RUN:
            self.uce.step(cycle, self)
        # aggregate PMU snapshot (lightweight: copy counters)
        self._aggregate_pmu()
        return completed

    def _advance_task_fsm(self, cycle: int) -> None:
        """Advance the TileTaskState FSM one cycle (Compute Tile design 3)."""
        if self.task_state == TileTaskState.TASK_ACCEPT:
            self.pmu.add_cycle("task_accept", 1)
            # always enter PREPARED_TASK_CHECK (remaining >= 1 = gate cycle);
            # warm path pays 1 cycle (gate), cold pays 1 + install_cycles.
            self.task_state = TileTaskState.PREPARED_TASK_CHECK
        elif self.task_state == TileTaskState.PREPARED_TASK_CHECK:
            self.pmu.add_cycle("prepared_check", 1)
            if self._prepared_check_remaining > 1:
                self._prepared_check_remaining -= 1
            else:
                # last prepared-check cycle: transition this cycle
                self._prepared_check_remaining = 0
                self.task_state = TileTaskState.FRAME_BIND
        elif self.task_state == TileTaskState.FRAME_BIND:
            self.pmu.add_cycle("frame_bind", 1)
            if self._frame_bind_remaining > 1:
                self._frame_bind_remaining -= 1
            else:
                # last frame-bind cycle: call SlotFrame.bind() for
                # metadata validation; on fault → FAULTED.
                ok, _cycles = self.l1_frame.bind(
                    cycle, self.cfg.frame_bind_cycles)
                if not ok:
                    self.task_state = TileTaskState.FAULTED
                    self.uce.done = True
                    self.pmu.add_event("l1_frame_fault")
                    return
                self._frame_bind_remaining = 0
                self.task_state = TileTaskState.PROGRAM_RUN
        elif self.task_state == TileTaskState.PROGRAM_RUN:
            pass  # handled by UCE in step()
        elif self.task_state == TileTaskState.DRAIN:
            self.pmu.add_cycle("drain", 1)
            if self._outstanding_zero():
                self.task_state = TileTaskState.DONE
        # FAULTED/DONE/IDLE: no-op

    def _aggregate_pmu(self) -> None:
        for eng in (self.boa, self.evu, self.mfe, self.use):
            self.pmu.merge(eng.pmu)
            eng.pmu.reset()
        self.pmu.merge(self.uce.pmu)
        self.uce.pmu.reset()

    @property
    def faulted(self) -> bool:
        return self.uce.faulted

    @property
    def fault_reason(self) -> str:
        return self.uce.fault_reason

    @property
    def done(self) -> bool:
        # A UCE validation fault makes the tile done immediately so the
        # group can propagate the fault without waiting for engines.
        if self.uce.faulted:
            return True
        if self.runtime_enabled:
            # In runtime mode, done requires the FSM to reach DONE/FAULTED
            # *and* UCE + engines to be idle.
            if self.task_state in (TileTaskState.TASK_ACCEPT,
                                   TileTaskState.PREPARED_TASK_CHECK,
                                   TileTaskState.FRAME_BIND,
                                   TileTaskState.DRAIN):
                return False
            if self.task_state in (TileTaskState.DONE, TileTaskState.FAULTED):
                return True
            # PROGRAM_RUN: check UCE + engines, then transition to DRAIN
            uce_done = self.uce.done
            engs_idle = all(
                eng.state in (EngineState.IDLE, EngineState.DONE)
                for eng in (self.boa, self.evu, self.mfe, self.use))
            if uce_done and engs_idle:
                self.task_state = TileTaskState.DRAIN
                return False
            return False
        return self.uce.done and all(
            eng.state in (EngineState.IDLE, EngineState.DONE)
            for eng in (self.boa, self.evu, self.mfe, self.use))

    def _outstanding_zero(self) -> bool:
        """Check all engines idle (DRAIN exit condition, Compute Tile 3)."""
        return all(
            eng.state in (EngineState.IDLE, EngineState.DONE)
            for eng in (self.boa, self.evu, self.mfe, self.use))

    def reset(self) -> None:
        self.uce.reset()
        for eng in (self.boa, self.evu, self.mfe, self.use):
            eng.reset()
        self.pmu.reset()
        self.role_id = None
        self.task_state = TileTaskState.IDLE
        self._prepared_check_remaining = 0
        self._frame_bind_remaining = 0
        self.l1_frame.reset()

    def snapshot(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "uce_pc": self.uce.pc,
            "uce_done": self.uce.done,
            "faulted": self.uce.faulted,
            "fault_reason": self.uce.fault_reason,
            "task_state": self.task_state.name,
            "boa_state": self.boa.state.name,
            "evu_state": self.evu.state.name,
            "mfe_state": self.mfe.state.name,
            "use_state": self.use.state.name,
            "uce_scheduler": self.uce.scheduler_snapshot(),
            "l1_frame": self.l1_frame.snapshot(),
        }
