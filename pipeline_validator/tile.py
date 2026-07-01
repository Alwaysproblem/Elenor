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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import HardwareConfig
from .engines import BOAEngine, Engine, EngineJob, EngineState, EVUEngine, MFEEngine, USEEngine
from .ir import EngineDesc, TileInst, TileOp, TileProgram
from .pmu import PMUCounter, StallReason
from .stream_queue import StreamQueue, StreamToken
from .trace import Tracer


@dataclass
class _PendingWait:
    """A UCE blocked waiting on event(s)."""
    events: tuple
    all: bool  # True=waitall, False=wait single
    started_cycle: int


@dataclass
class _PreparedTileTask:
    """A Tile Program admitted into the UCE issue window."""
    program: TileProgram
    prepare_remaining: int
    frame_bind_remaining: int
    role_id: int | None
    role_event: str | None
    overlap_started: bool = False
    window_id: int = 0
    lookahead_pc: int = 0
    lookahead_event_aliases: dict[str, str] = field(default_factory=dict)
    lookahead_done: set[str] = field(default_factory=set)
    lookahead_stopped: bool = False


@dataclass
class _CompletedTileTask:
    """A role completion latched for TileGroup fan-in accounting."""
    role_id: int | None
    role_event: str | None


class TileUCE:
    """Tile Unified Control Engine.

    Issues one instruction per cycle when not blocked.  Tracks outstanding
    engine events and stream-token registers.
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

    def load(self, program: TileProgram) -> None:
        self.program = program
        self.pc = 0
        self._events_done.clear()
        self._pending = None
        self._tokens.clear()
        self._event_meta.clear()
        self.done = False

    # ---- per-cycle step -------------------------------------------------

    def step(self, cycle: int, tile: ComputeTile) -> str | None:
        """Execute one cycle of the UCE.  Returns an event id if an engine
        just completed that the UCE was *not* blocking on (rare), else None.

        The UCE returns 'done' when the program returns.
        """
        if self.done or self.program is None:
            self.pmu.add_cycle("idle", 1)
            self.pmu.add_cycle("total", 1)
            return None

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

    # ---- instruction issue ----------------------------------------------

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

    def _launch_engine(self, engine: Engine, ins: TileInst, cycle: int,
                       tile: ComputeTile) -> bool:
        """Launch an engine descriptor.  Returns False (UCE blocks, no PC
        advance) when the engine is still busy with a prior job.
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
        job = engine.launch(desc, cycle, ins.dst)
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
        job = tile.mfe.launch(desc, cycle, ins.dst)
        if job is None:
            self.pmu.add(StallReason.WAIT_OPERAND, 1)
            return False
        self._event_meta[ins.dst] = {"engine": "DMA", "job": job}
        return True
    def notify_event(self, event_id: str) -> None:
        self._events_done.add(event_id)
        if self._event_done_callback is not None:
            self._event_done_callback(event_id)


    def reset(self) -> None:
        self.pc = 0
        self._events_done.clear()
        self._pending = None
        self._tokens.clear()
        self._event_meta.clear()
        self.done = False
        self.pmu.reset()


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
        self._current_role_event: str | None = None
        self._active_task_valid: bool = False
        self._window_queue: list[_PreparedTileTask] = []
        self._just_completed: _CompletedTileTask | None = None
        self._completion_latched: bool = False
        self._next_window_id: int = 0
        self._active_event_aliases: dict[str, str] = {}
        # L1 Slot Frame (full_memory fidelity only)
        self.l1_frame: object | None = None  # SlotFrame, created on demand

    def bind_stream(self, qid: int, q: StreamQueue) -> None:
        self.streams[qid] = q

    def get_stream(self, qid: int) -> StreamQueue:
        return self.streams[qid]

    @property
    def has_active_task(self) -> bool:
        return self._active_task_valid

    @property
    def current_role_event(self) -> str | None:
        return self._current_role_event

    def window_active_entries(self) -> int:
        current = 1 if self._active_task_valid else 0
        return current + len(self._window_queue)

    def can_accept_program(self) -> bool:
        if not self.runtime_enabled:
            # timing_only has no window queue: a tile accepts only when it
            # has no active task or the active task has completed and been
            # retired.  Keeps the precheck in dispatch_role() consistent
            # with load_program()'s rejection path.
            return (not self._active_task_valid) or self._completion_latched
        return self.window_active_entries() < max(1, self.cfg.uce_window_size)

    def load_program(self,
                     program: TileProgram,
                     prepare_cycles: int = 0,
                     role_id: int | None = None,
                     role_event: str | None = None) -> bool:
        if not self.runtime_enabled:
            # timing_only has no window queue: reject a second program while
            # one is active so the sequencer retries the dispatch after the
            # current role completes (prevents overwriting a running
            # TileProgram before its role event fires).
            if self._active_task_valid and not self._completion_latched:
                return False
            if self._completion_latched:
                self.pop_completed_task()
            self.uce.load(program)
            self.role_id = role_id
            self._current_role_event = role_event
            self._active_task_valid = True
            return True

        if self._completion_latched:
            self.pop_completed_task()
        if not self.can_accept_program():
            return False

        entry = _PreparedTileTask(
            program=program,
            prepare_remaining=1 + prepare_cycles,
            frame_bind_remaining=self.cfg.frame_bind_cycles,
            role_id=role_id,
            role_event=role_event,
            window_id=self._next_window_id,
        )
        self._next_window_id += 1
        if not self._active_task_valid:
            self._start_window_entry(entry)
        else:
            self._window_queue.append(entry)
            self.pmu.add_event("uce_window_entry_queued")
        return True

    def _start_window_entry(self, entry: _PreparedTileTask) -> None:
        self.uce.load(entry.program)
        # Promote lookahead progress: a queued entry may have already issued
        # its leading MFE-load prefix and crossed lookahead-only waits.  Set
        # the UCE PC past those instructions and seed events_done so the
        # active UCE never re-executes already-issued loads.  Bridge any
        # in-flight namespaced lookahead loads to the original event ids the
        # program waits on.
        self.uce.pc = entry.lookahead_pc
        self.uce._events_done.update(entry.lookahead_done)
        self._active_event_aliases = {
            namespaced: original
            for namespaced, original in entry.lookahead_event_aliases.items()
            if original not in entry.lookahead_done
        }
        if entry.lookahead_pc > 0:
            self.pmu.add_event("uce_window_mfe_lookahead_promote")
        self.role_id = entry.role_id
        self._current_role_event = entry.role_event
        self._active_task_valid = True
        self._completion_latched = False
        self._just_completed = None
        if self.runtime_enabled:
            self._prepared_check_remaining = max(0, entry.prepare_remaining)
            self._frame_bind_remaining = max(0, entry.frame_bind_remaining)
            if self._prepared_check_remaining > 0:
                self.task_state = (TileTaskState.PREPARED_TASK_CHECK
                                   if entry.overlap_started else TileTaskState.TASK_ACCEPT)
            elif self._frame_bind_remaining > 0:
                self.task_state = TileTaskState.FRAME_BIND
            else:
                self.task_state = TileTaskState.PROGRAM_RUN
    # ---- per-cycle step -------------------------------------------------

    def step(self, cycle: int) -> EngineJob | None:
        """Advance one cycle and latch role completion without auto-retiring it.

        Runtime/full_memory fidelity uses a small UCE issue window: at most one
        Tile Program is executing while one queued entry may overlap its
        prepared-task/frame-bind work.  Completion fan-in is exposed through
        pop_completed_task(), so TileGroup accounting is not derived from the
        mutable ``done`` state after the next queued entry starts.
        """
        if not self._completion_latched:
            self._just_completed = None

        # tick engines first; collect completions
        completed = None
        for eng in (self.boa, self.evu, self.mfe, self.use):
            job = eng.tick(cycle)
            if job is not None:
                completed = job
                queued_lookahead = self._notify_window_event(job.event_id)
                if job.event_id in self._active_event_aliases:
                    self.uce.notify_event(self._active_event_aliases.pop(job.event_id))
                elif not queued_lookahead:
                    self.uce.notify_event(job.event_id)

        if self.runtime_enabled:
            self._advance_window_queue(cycle)
            active_uce_ran = False
            if self._active_task_valid:
                if self.task_state != TileTaskState.PROGRAM_RUN:
                    self._advance_task_fsm(cycle)
                else:
                    self.uce.step(cycle, self)
                    self._maybe_enter_drain()
                    active_uce_ran = True
                self._maybe_latch_completion()
            # Issue queued-entry MFE-load lookahead ONLY when the active UCE
            # had its step this cycle (PROGRAM_RUN, including wait-stall/
            # wait-resolve cycles that do not advance pc) or the active task
            # is already draining.  This blocks lookahead on the
            # FRAME_BIND->PROGRAM_RUN transition cycle, where the active UCE
            # never got a turn and its first LAUNCH_MFE would hit
            # engine_busy next cycle.
            if active_uce_ran or self.task_state == TileTaskState.DRAIN:
                self._advance_window_mfe_lookahead(cycle)
        elif self._active_task_valid:
            self.uce.step(cycle, self)
            self._maybe_latch_completion()

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
                # last frame-bind cycle: enter PROGRAM_RUN this cycle
                self._frame_bind_remaining = 0
                self.task_state = TileTaskState.PROGRAM_RUN
        elif self.task_state == TileTaskState.PROGRAM_RUN:
            pass  # handled by UCE in step()
        elif self.task_state == TileTaskState.DRAIN:
            self.pmu.add_cycle("drain", 1)
            if self._outstanding_zero():
                self.task_state = TileTaskState.DONE
        # FAULTED/DONE/IDLE: no-op

    def _advance_window_queue(self, cycle: int) -> None:
        del cycle  # overlap accounting is cycle-local; no timestamp needed here
        if not (self.runtime_enabled and self._active_task_valid and self._window_queue):
            return
        if self._completion_latched:
            return
        if self.task_state not in (TileTaskState.PROGRAM_RUN, TileTaskState.DRAIN):
            return

        entry = self._window_queue[0]
        entry.overlap_started = True
        if entry.prepare_remaining > 0:
            entry.prepare_remaining -= 1
            self.pmu.add_cycle("uce_window_prepare_overlap", 1)
        elif entry.frame_bind_remaining > 0:
            entry.frame_bind_remaining -= 1
            self.pmu.add_cycle("uce_window_frame_bind_overlap", 1)

    def _can_mfe_lookahead(self, program: TileProgram) -> bool:
        """Stream TilePrograms are excluded from lookahead (stream token
        semantics are unsafe to pre-execute under another TileProgram)."""
        stream_ops = {
            TileOp.STREAM_POP,
            TileOp.STREAM_PUSH,
            TileOp.STREAM_ACQUIRE,
            TileOp.STREAM_RELEASE,
            TileOp.STREAM_PUSH_EOS,
            TileOp.BR_EOS,
        }
        return all(ins.op not in stream_ops for ins in program.insts)

    def _lookahead_event_id(self, entry: _PreparedTileTask,
                            original_event: str) -> str:
        return f"lookahead:{entry.window_id}:{original_event}"

    def _advance_window_mfe_lookahead(self, cycle: int) -> None:
        """Issue at most one MFE-load instruction from the queued entry's
        leading MFE-load prefix while the active TileProgram is still running.

        Never stalls the active UCE: it is called after the active UCE step,
        and only issues when the active task is in PROGRAM_RUN/DRAIN and
        the MFE engine is free.  Stops at the first non-load instruction or
        at a wait for an event not issued by this lookahead context.
        """
        if not (self.runtime_enabled and self._active_task_valid and self._window_queue):
            return
        if self._completion_latched:
            return
        if self.task_state not in (TileTaskState.PROGRAM_RUN, TileTaskState.DRAIN):
            return
        entry = self._window_queue[0]
        if entry.lookahead_stopped:
            return
        if entry.prepare_remaining > 0 or entry.frame_bind_remaining > 0:
            return
        program = entry.program
        if not self._can_mfe_lookahead(program):
            return
        if entry.lookahead_pc >= len(program.insts):
            return
        ins = program.insts[entry.lookahead_pc]
        # ---- WAIT / WAITALL ----
        if ins.op in (TileOp.WAIT, TileOp.WAITALL):
            waited = tuple(ins.args)
            issued_originals = set(entry.lookahead_event_aliases.values())
            foreign = [e for e in waited if e not in issued_originals]
            if foreign:
                entry.lookahead_stopped = True
                return
            done = [e for e in waited if e in entry.lookahead_done]
            satisfied = (len(done) == len(waited)
                         if ins.op == TileOp.WAITALL else len(done) > 0)
            if satisfied:
                entry.lookahead_pc += 1
                self.pmu.add_event("uce_window_mfe_lookahead_wait_resolved")
            else:
                self.pmu.add_cycle("uce_window_mfe_lookahead_wait", 1)
            return
        # ---- non-MFE launch stops lookahead ----
        if ins.op != TileOp.LAUNCH_MFE:
            entry.lookahead_stopped = True
            return
        desc = program.descriptors[ins.args[0]]
        if desc.kind != "MFE" or desc.op != "load":
            entry.lookahead_stopped = True
            return
        if self.mfe.is_busy:
            self.pmu.add_cycle("uce_window_mfe_lookahead_engine_busy", 1)
            return
        assert ins.dst is not None, "LAUNCH_MFE requires dst event id"
        namespaced = self._lookahead_event_id(entry, ins.dst)
        patched = EngineDesc(
            name=desc.name,
            kind=desc.kind,
            op=desc.op,
            params={**desc.params, "tile_id": self.tile_id},
        )
        job = self.mfe.launch(patched, cycle, namespaced)
        if job is None:
            self.pmu.add_cycle("uce_window_mfe_lookahead_engine_busy", 1)
            return
        entry.lookahead_event_aliases[namespaced] = ins.dst
        entry.lookahead_pc += 1
        self.pmu.add_event("uce_window_mfe_lookahead_launch")
        if self.tracer is not None:
            self.tracer.instant(
                f"Tile{self.tile_id}",
                "UCE",
                "uce_window_mfe_lookahead_launch",
                cycle,
                {
                    "window_id": entry.window_id,
                    "role_event": entry.role_event,
                    "program": program.name,
                    "event_id": namespaced,
                    "original_event": ins.dst,
                    "desc": desc.name,
                },
            )

    def _notify_window_event(self, event_id: str) -> bool:
        """Route a namespaced lookahead event completion to its queued entry.

        Returns True if the event belongs to a queued lookahead context (the
        active UCE must NOT be notified with the namespaced id).  Returns
        False for ordinary active-UCE events.
        """
        for entry in self._window_queue:
            if event_id in entry.lookahead_event_aliases:
                entry.lookahead_done.add(entry.lookahead_event_aliases[event_id])
                return True
        return False

    def _maybe_enter_drain(self) -> None:
        if not (self.runtime_enabled and self._active_task_valid):
            return
        if self.task_state != TileTaskState.PROGRAM_RUN:
            return
        if self.uce.done and self._outstanding_zero():
            self.task_state = TileTaskState.DRAIN

    def _maybe_latch_completion(self) -> None:
        if not self._active_task_valid or self._completion_latched:
            return

        if self.runtime_enabled:
            completed = self.task_state == TileTaskState.DONE and self._outstanding_zero()
        else:
            completed = self.uce.done and self._outstanding_zero()
        if not completed:
            return

        self._completion_latched = True
        self._just_completed = _CompletedTileTask(
            role_id=self.role_id,
            role_event=self._current_role_event,
        )
        self.pmu.add_event("tile_task_completed")

    def pop_completed_task(self) -> _CompletedTileTask | None:
        completed = self._just_completed
        if completed is None:
            return None

        self._just_completed = None
        self._completion_latched = False
        self._active_task_valid = False

        if self._window_queue:
            self._start_window_entry(self._window_queue.pop(0))
        else:
            self.role_id = None
            self._current_role_event = None
            if self.runtime_enabled:
                self.task_state = TileTaskState.IDLE

        return completed

    def _aggregate_pmu(self) -> None:
        for eng in (self.boa, self.evu, self.mfe, self.use):
            self.pmu.merge(eng.pmu)
            eng.pmu.reset()
        self.pmu.merge(self.uce.pmu)
        self.uce.pmu.reset()

    @property
    def done(self) -> bool:
        if not self._active_task_valid:
            return True
        if self._completion_latched:
            return True
        if self.runtime_enabled:
            return self.task_state == TileTaskState.DONE and self._outstanding_zero()
        return self.uce.done and self._outstanding_zero()

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
        self._current_role_event = None
        self._active_task_valid = False
        self._window_queue.clear()
        self._just_completed = None
        self._completion_latched = False
        self._active_event_aliases.clear()
        self._next_window_id = 0
        self.task_state = TileTaskState.IDLE
        self._prepared_check_remaining = 0
        self._frame_bind_remaining = 0

    def snapshot(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "uce_pc": self.uce.pc,
            "uce_done": self.uce.done,
            "task_state": self.task_state.name,
            "role_id": self.role_id,
            "current_role_event": self._current_role_event,
            "uce_window_size": self.cfg.uce_window_size,
            "uce_window_active": self.window_active_entries(),
            "uce_window_queued": len(self._window_queue),
            "uce_window_completion_latched": self._completion_latched,
            "uce_window_queue": [
                {
                    "role_id": entry.role_id,
                    "event_id": entry.role_event,
                    "prepare_remaining": entry.prepare_remaining,
                    "frame_bind_remaining": entry.frame_bind_remaining,
                    "overlap_started": entry.overlap_started,
                    "window_id": entry.window_id,
                    "lookahead_pc": entry.lookahead_pc,
                    "lookahead_done": sorted(entry.lookahead_done),
                    "lookahead_event_aliases": dict(entry.lookahead_event_aliases),
                }
                for entry in self._window_queue
            ],
            "boa_state": self.boa.state.name,
            "evu_state": self.evu.state.name,
            "mfe_state": self.mfe.state.name,
            "use_state": self.use.state.name,
        }
