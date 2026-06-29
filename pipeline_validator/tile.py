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

from dataclasses import dataclass
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
        # L1 Slot Frame (full_memory fidelity only)
        self.l1_frame: object | None = None  # SlotFrame, created on demand

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

    def _aggregate_pmu(self) -> None:
        for eng in (self.boa, self.evu, self.mfe, self.use):
            self.pmu.merge(eng.pmu)
            eng.pmu.reset()
        self.pmu.merge(self.uce.pmu)
        self.uce.pmu.reset()

    @property
    def done(self) -> bool:
        if self.runtime_enabled:
            # In runtime mode, done requires the FSM to reach DONE/FAULTED
            # *and* UCE + engines to be idle.
            if self.task_state in (TileTaskState.TASK_ACCEPT,
                                   TileTaskState.PREPARED_TASK_CHECK,
                                   TileTaskState.FRAME_BIND,
                                   TileTaskState.DRAIN):
                return False
            if self.task_state == TileTaskState.DONE:
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
    def snapshot(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "uce_pc": self.uce.pc,
            "uce_done": self.uce.done,
            "task_state": self.task_state.name,
            "boa_state": self.boa.state.name,
            "evu_state": self.evu.state.name,
            "mfe_state": self.mfe.state.name,
            "use_state": self.use.state.name,
        }
