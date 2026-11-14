// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package coralnpu

import chisel3._
import chisel3.util._

/** Context slot lifecycle states (C027 / W2-04; superset of validator `_UCEContextState`).
  *
  * FREE -> HELD -> ACCEPTED -> INIT -> READY -> WAIT_*(sub)/PARKED -> DONE_DRAINING
  *   -> EXEC_DRAINED -> RETIRE_READY -> TASK_COMPLETION -> (ack) -> FREE ;
  * FAULT is the fail-stop terminal (recovery is a Group reset, never local).
  */
object ContextState extends ChiselEnum {
  val FREE, HELD, ACCEPTED, INIT, READY = Value
  val WAIT_EVENT, WAIT_ENGINE_QUEUE, WAIT_STREAM, WAIT_REGION, WAIT_FETCH = Value
  val WAIT_OPERAND, WAIT_DEST = Value
  val PARKED = Value
  val DONE_DRAINING, EXEC_DRAINED, RETIRE_READY, TASK_COMPLETION = Value
  val FAULT = Value
}

/** Scheduler-side wait reasons (C028 drives these once real fetch/dispatch lands).
  *
  * EVENT and REGION waits are NOT in here: they derive from the W6/W3 read-only
  * projections `ext.waitBusy` / `ext.regionBusy` (the records live in W6/W3).
  */
object ContextWait extends ChiselEnum {
  val NONE, ENGINE_QUEUE, STREAM, FETCH, OPERAND, DEST = Value
}

/** Per-Context control inputs owned by W6/W3 projections (read-only here). */
class CtxExternalIO(p: Parameters) extends Bundle {
  val waitBusy   = Input(Bool()) // W6: a wait record of this Context is not yet satisfied
  val regionBusy = Input(Bool()) // W3: this Context holds a pending VME region interaction
}

/** Named IO so integration harnesses (and C040 top assembly) can type it (C027 / W2-04). */
class ContextManagerIO(p: Parameters) extends Bundle {
  val activeCtx = Input(UInt(p.ctxBits.W)) // selected Context (CoreContextIO.ctxSelect)

  // --- admission / lifecycle (W1 boundary coordinator; C035 wires the real ones) ---
  val holdReq   = Input(Vec(p.contextCount, Bool()))
  val accept    = Input(Vec(p.contextCount, Bool()))
  val initStart = Input(Vec(p.contextCount, Bool())) // ACCEPTED -> INIT
  val initDone  = Input(Vec(p.contextCount, Bool())) // INIT -> READY
  val retIssued = Input(Vec(p.contextCount, Bool())) // RET instruction retired this Context

  // --- generation-qualified park (WFI / WAIT / region-head parking; level) ---
  val parkValid = Input(Vec(p.contextCount, Bool()))
  val parkGen   = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))

  // --- tagged wake from W6 event completion (satisfies a WAIT_EVENT at once) ---
  val wakeValid = Input(Vec(p.contextCount, Bool()))
  val wakeGen   = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))

  // --- scheduler-side wait reason (C028) ---
  val waitKind = Input(Vec(p.contextCount, ContextWait()))

  // --- W6/W3 read-only summaries ---
  val ext = Input(Vec(p.contextCount, new CtxExternalIO(p)))

  // --- local quiescence / terminal inputs ---
  val robEmpty           = Input(Vec(p.contextCount, Bool()))
  val pendingLaunch      = Input(Vec(p.contextCount, Bool())) // any un-completed engine launch
  // SCore `perCtxBackendDrained` projection: this Context's RF/FPR outstanding writes
  // (scoreboard | flushed_scoreboard) and other backend write obligations are closed.
  val backendDrained     = Input(Vec(p.contextCount, Bool()))
  val faultCtx           = Input(Vec(p.contextCount, Bool())) // fault raised for this Context
  val taskCompletion     = Input(Vec(p.contextCount, Bool())) // W6 reliably stored TASK_COMPLETION
  val taskCompletionGen  = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))
  val completionAck      = Input(Vec(p.contextCount, Bool())) // Group consumed the completion
  val completionAckGen   = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))

  // --- cancel (W6 CANCEL_TASK): only THIS Context's issue stops; peers untouched ---
  val cancelValid = Input(Vec(p.contextCount, Bool()))
  val cancelGen   = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))

  // --- outputs ---
  val ctxState   = Output(Vec(p.contextCount, ContextState()))
  val perCtxGen  = Output(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))
  val eligible   = Output(Vec(p.contextCount, Bool())) // may issue NEW instructions
  val perCtxParked    = Output(Vec(p.contextCount, Bool()))
  val perCtxWaiting   = Output(Vec(p.contextCount, Bool()))
  val perCtxCancelled = Output(Vec(p.contextCount, Bool()))
  val cancelAck  = Output(Vec(p.contextCount, Bool())) // 1-cycle pulse: live cancel honored
  val execDrained = Output(Vec(p.contextCount, Bool()))
  val retireReady = Output(Vec(p.contextCount, Bool()))
  val advanceGen  = Output(Vec(p.contextCount, Bool())) // reuse edge: bump gen & clear banks
  val tileIssueFreeze = Output(Bool())                  // fail-stop: any FAULT stops ALL issue
  val tileQuiescent   = Output(Bool())                  // all parked/idle AND backends empty
  val staleCancelRejected     = Output(UInt(16.W))
  val staleParkRejected       = Output(UInt(16.W))
  val staleWakeRejected       = Output(UInt(16.W))
  val staleCompletionRejected = Output(UInt(16.W))
  val staleAckRejected        = Output(UInt(16.W))
}

/** Context slot manager: park / cancel / fail-stop separation + EXEC_DRAINED + slot reuse
  * (C027 / W2-04, W2-D06 / CTX-15 / SCH-07 / SCH-11 / RET-06).
  *
  * This module NEVER gates the core clock (park only stops NEW issue for its Context) and
  * NEVER touches another Context's state on cancel (RET-06 hard invariant).
  *
  * Semantics pinned by W2-D06:
  *   - park (level, generation-qualified) stops new issue only; accepted work keeps
  *     executing and retiring. A parked slot with a non-empty ROB is NOT quiescent.
  *   - cancel (generation-qualified) stops new issue in the cancel beat itself. Work that
  *     was accepted still drains to its terminal path; un-accepted payloads are dropped
  *     (HELD cancel returns FREE directly). Only THIS Context is affected.
  *   - fail-stop: a fault on ANY occupied Context latches FAULT and freezes issue for the
  *     whole Tile in the fault beat itself (input fault included); recovery is reset only.
  *   - reuse: generation advances EXACTLY ONCE per slot-lifecycle end, emitting a 1-cycle
  *     `advanceGen` pulse that the parent wires to per-Context bank clears.
  */
class ContextManager(p: Parameters) extends Module {
  val io = IO(new ContextManagerIO(p))

  val states    = RegInit(VecInit.fill(p.contextCount)(ContextState.FREE))
  val gens      = RegInit(VecInit.fill(p.contextCount)(0.U(p.ctxGenerationBits.W)))
  val cancelled = RegInit(VecInit.fill(p.contextCount)(false.B))

  // Per-Context stale-reject counters (W2 plan §8 step 7): they belong to the slot's
  // CURRENT owner and clear ONLY at that slot's own reuse edge (together with the owned
  // bank clears), so one Context's reject history never disturbs a peer's stats.
  val staleCancel     = RegInit(VecInit.fill(p.contextCount)(0.U(16.W)))
  val stalePark       = RegInit(VecInit.fill(p.contextCount)(0.U(16.W)))
  val staleWake       = RegInit(VecInit.fill(p.contextCount)(0.U(16.W)))
  val staleCompletion = RegInit(VecInit.fill(p.contextCount)(0.U(16.W)))
  val staleAck       = RegInit(VecInit.fill(p.contextCount)(0.U(16.W)))

  val nextState   = Wire(Vec(p.contextCount, ContextState()))
  val obligationsClosed = Wire(Vec(p.contextCount, Bool()))
  val parkLive    = Wire(Vec(p.contextCount, Bool()))
  val cancelLive  = Wire(Vec(p.contextCount, Bool()))
  val wakeLive    = Wire(Vec(p.contextCount, Bool()))
  val cancelHonored = Wire(Vec(p.contextCount, Bool()))

  for (c <- 0 until p.contextCount) {
    val s        = states(c)
    val occupied = s =/= ContextState.FREE
    val inWait   = s.isOneOf(
      ContextState.WAIT_EVENT, ContextState.WAIT_ENGINE_QUEUE, ContextState.WAIT_STREAM,
      ContextState.WAIT_REGION, ContextState.WAIT_FETCH, ContextState.WAIT_OPERAND,
      ContextState.WAIT_DEST,
    )
    // Issue-capable preterminal states: a live cancel is honored here (HELD handled apart).
    val preDrain = s.isOneOf(ContextState.ACCEPTED, ContextState.INIT, ContextState.READY,
      ContextState.PARKED) || inWait

    // --- generation qualification: only the live generation is honoured; stale signals
    // --- are counted, never applied (CTX-15 / INV-15). Signals on a FREE slot are
    // --- ignored entirely (nothing is occupied; no phantom counts).
    parkLive(c)    := io.parkValid(c) && occupied && (io.parkGen(c) === gens(c))
    val parkStale  = io.parkValid(c) && occupied && (io.parkGen(c) =/= gens(c))
    cancelLive(c)  := io.cancelValid(c) && occupied && (io.cancelGen(c) === gens(c))
    val cancelStale = io.cancelValid(c) && occupied && (io.cancelGen(c) =/= gens(c))
    wakeLive(c)    := io.wakeValid(c) && occupied && (io.wakeGen(c) === gens(c))
    val wakeStale  = io.wakeValid(c) && occupied && (io.wakeGen(c) =/= gens(c))
    val completionLive = io.taskCompletion(c) && occupied &&
      (io.taskCompletionGen(c) === gens(c))
    val completionStale = io.taskCompletion(c) && occupied &&
      (io.taskCompletionGen(c) =/= gens(c))
    val ackLive  = io.completionAck(c) && (io.completionAckGen(c) === gens(c))
    val ackStale = io.completionAck(c) && (io.completionAckGen(c) =/= gens(c))

    when(occupied) {
      when(cancelStale) { staleCancel(c) := staleCancel(c) + 1.U }
      when(parkStale) { stalePark(c) := stalePark(c) + 1.U }
      when(wakeStale) { staleWake(c) := staleWake(c) + 1.U }
      when(completionStale) { staleCompletion(c) := staleCompletion(c) + 1.U }
      // An ack is only meaningful for an occupied slot; a stale-generation ack is an
      // old Group confirmation that must never free a newer owner's slot.
      when(ackStale) { staleAck(c) := staleAck(c) + 1.U }
    }

    // Local obligations: ROB partition, engine launches, W6 wait records, W3 regions,
    // and the SCore backend projection (RF/FPR outstanding writes). True when ALL of
    // them are closed (drained) — the ack cannot free the slot before this holds.
    obligationsClosed(c) := io.robEmpty(c) && !io.pendingLaunch(c) &&
      !io.ext(c).waitBusy && !io.ext(c).regionBusy && io.backendDrained(c)

    // --- cancel honoring: HELD drops the un-accepted payload straight to FREE; every
    // --- other issue-capable state stops new issue NOW and drains accepted work.
    cancelHonored(c) := cancelLive(c) && ((s === ContextState.HELD) || preDrain)
    when(cancelHonored(c)) { cancelled(c) := true.B }

    val wk        = io.waitKind(c)
    val schedWait = wk =/= ContextWait.NONE
    val schedWaitState = WireInit(ContextState.WAIT_ENGINE_QUEUE)
    switch(wk) {
      is(ContextWait.ENGINE_QUEUE) { schedWaitState := ContextState.WAIT_ENGINE_QUEUE }
      is(ContextWait.STREAM) { schedWaitState := ContextState.WAIT_STREAM }
      is(ContextWait.FETCH) { schedWaitState := ContextState.WAIT_FETCH }
      is(ContextWait.OPERAND) { schedWaitState := ContextState.WAIT_OPERAND }
      is(ContextWait.DEST) { schedWaitState := ContextState.WAIT_DEST }
    }

    // Uniform exit chain for WAIT_* states. Priorities: cancel > park > tagged W6 wake
    // (WAIT_EVENT only) > W6 wait record > W3 region > scheduler wait reason > READY.
    // Architectural waits (ext levels) outrank scheduler waits; a wake token satisfies
    // the wait record even if the level summary lags one beat.
    val waitNext = WireInit(ContextState.READY)
    when(cancelLive(c)) {
      waitNext := ContextState.DONE_DRAINING
    }.elsewhen(io.retIssued(c)) {
      waitNext := ContextState.DONE_DRAINING
    }.elsewhen(parkLive(c)) {
      waitNext := ContextState.PARKED
    }.elsewhen((s === ContextState.WAIT_EVENT) && wakeLive(c)) {
      waitNext := ContextState.READY
    }.elsewhen(io.ext(c).waitBusy) {
      waitNext := ContextState.WAIT_EVENT
    }.elsewhen(io.ext(c).regionBusy) {
      waitNext := ContextState.WAIT_REGION
    }.elsewhen(schedWait) {
      waitNext := schedWaitState
    }.otherwise {
      waitNext := ContextState.READY
    }

    nextState(c) := s
    switch(s) {
      is(ContextState.FREE) {
        when(io.holdReq(c)) {
          nextState(c) := ContextState.HELD
        }
      }
      is(ContextState.HELD) {
        when(cancelLive(c)) {
          // Un-accepted payload dropped (W2-D06): immediate safe reuse iff drained.
          nextState(c) := Mux(obligationsClosed(c), ContextState.FREE, ContextState.DONE_DRAINING)
        }.elsewhen(io.accept(c)) {
          nextState(c) := ContextState.ACCEPTED
        }.elsewhen(io.holdReq(c)) {
          nextState(c) := ContextState.HELD // hold may persist until admission authorizes
        }
      }
      is(ContextState.ACCEPTED) {
        when(cancelLive(c)) {
          nextState(c) := ContextState.DONE_DRAINING
        }.elsewhen(io.initStart(c)) {
          nextState(c) := ContextState.INIT
        }
      }
      is(ContextState.INIT) {
        when(cancelLive(c)) {
          nextState(c) := ContextState.DONE_DRAINING
        }.elsewhen(io.initDone(c)) {
          nextState(c) := ContextState.READY
        }
      }
      is(ContextState.READY) {
        when(cancelLive(c)) {
          nextState(c) := ContextState.DONE_DRAINING
        }.elsewhen(io.retIssued(c)) {
          nextState(c) := ContextState.DONE_DRAINING
        }.elsewhen(parkLive(c)) {
          nextState(c) := ContextState.PARKED
        }.elsewhen(io.ext(c).waitBusy) {
          nextState(c) := ContextState.WAIT_EVENT
        }.elsewhen(io.ext(c).regionBusy) {
          nextState(c) := ContextState.WAIT_REGION
        }.elsewhen(schedWait) {
          nextState(c) := schedWaitState
        }
      }
      is(ContextState.PARKED) {
        when(cancelLive(c)) {
          nextState(c) := ContextState.DONE_DRAINING
        }.elsewhen(io.retIssued(c)) {
          nextState(c) := ContextState.DONE_DRAINING // retire never stops for park
        }.elsewhen(!parkLive(c)) {
          nextState(c) := ContextState.READY // wake resumes at the ORIGINAL head (no flush)
        }
      }
      is(ContextState.WAIT_EVENT, ContextState.WAIT_ENGINE_QUEUE,
        ContextState.WAIT_STREAM, ContextState.WAIT_REGION, ContextState.WAIT_FETCH,
        ContextState.WAIT_OPERAND, ContextState.WAIT_DEST) {
        nextState(c) := waitNext
      }
      is(ContextState.DONE_DRAINING) {
        when(obligationsClosed(c)) {
          nextState(c) := ContextState.EXEC_DRAINED
        }
      }
      is(ContextState.EXEC_DRAINED) {
        nextState(c) := ContextState.RETIRE_READY
      }
      is(ContextState.RETIRE_READY) {
        when(completionLive) {
          nextState(c) := ContextState.TASK_COMPLETION
        }
      }
      is(ContextState.TASK_COMPLETION) {
        // CTX-15: only a LIVE-generation ack can free the slot, and never while ANY
        // local obligation is still live.
        when(ackLive && obligationsClosed(c)) {
          nextState(c) := ContextState.FREE
        }
      }
      is(ContextState.FAULT) {
        nextState(c) := ContextState.FAULT // fail-stop terminal until reset (W6)
      }
    }

    // Fail-stop override: a fault on any occupied Context wins every concurrent
    // transition in its beat (FLT-01), including a same-beat cancel/wake/accept input.
    when(occupied && io.faultCtx(c)) {
      nextState(c) := ContextState.FAULT
    }
  }

  states := nextState

  // Reuse edge: generation advances EXACTLY ONCE (registered state leaves FREE-entry
  // pulses one-cycle long even if the triggering input stays high), and the pulse is
  // what the parent wires to per-Context bank clears (safe reuse only).
  for (c <- 0 until p.contextCount) {
    val reuseEdge = (nextState(c) === ContextState.FREE) && (states(c) =/= ContextState.FREE)
    io.advanceGen(c) := reuseEdge
    when(reuseEdge) {
      gens(c) := gens(c) + 1.U
      cancelled(c) := false.B
      // §8 step 7: the OLD owner's reject counters clear together with its banks;
      // a peer's counters are untouched (this block is per-slot).
      staleCancel(c) := 0.U
      stalePark(c) := 0.U
      staleWake(c) := 0.U
      staleCompletion(c) := 0.U
      staleAck(c) := 0.U
    }
  }

  // Fail-stop aggregate: latched FAULT states OR a live fault input on an occupied
  // Context freeze issue for the WHOLE Tile in the fault beat itself (SCH-11/FLT-01).
  io.tileIssueFreeze := VecInit.tabulate(p.contextCount) { c =>
    (states(c) === ContextState.FAULT) ||
      ((states(c) =/= ContextState.FREE) && io.faultCtx(c))
  }.asUInt.orR

  // Issue eligibility: READY only (WAIT_*/PARKED/draining states hold the head), and
  // the same-beat inputs already block issue in their own beat — park, cancel, fault,
  // a newly-asserted wait reason (ext levels or waitKind), and a same-beat RET. This
  // closes the one-beat window where PARKED wakes into READY while another wait reason
  // is still busy, or a wait/RET asserts in the READY beat itself.
  for (c <- 0 until p.contextCount) {
    io.eligible(c) := !io.tileIssueFreeze && !io.faultCtx(c) &&
      (states(c) === ContextState.READY) && !parkLive(c) && !cancelLive(c) &&
      !io.ext(c).waitBusy && !io.ext(c).regionBusy &&
      (io.waitKind(c) === ContextWait.NONE) && !io.retIssued(c)
  }

  io.ctxState        := states
  io.perCtxGen       := gens
  io.cancelAck       := cancelHonored
  io.execDrained     := VecInit.tabulate(p.contextCount)(c =>
    states(c) === ContextState.EXEC_DRAINED)
  io.retireReady     := VecInit.tabulate(p.contextCount)(c =>
    states(c) === ContextState.RETIRE_READY)
  io.perCtxParked    := VecInit.tabulate(p.contextCount)(c =>
    (states(c) === ContextState.PARKED) || parkLive(c))
  io.perCtxWaiting   := VecInit.tabulate(p.contextCount)(c =>
    states(c).isOneOf(
      ContextState.WAIT_EVENT, ContextState.WAIT_ENGINE_QUEUE, ContextState.WAIT_STREAM,
      ContextState.WAIT_REGION, ContextState.WAIT_FETCH, ContextState.WAIT_OPERAND,
      ContextState.WAIT_DEST,
    ))
  io.perCtxCancelled := VecInit.tabulate(p.contextCount)(c => cancelled(c) || cancelHonored(c))

  // Tile quiescence: quiet ONLY in parked/terminal states AND with every backend
  // obligation drained (ROB partition empty, no pending launch, no W6 wait record, no
  // W3 region interaction). A parked Context with a non-empty ROB still has accepted
  // work to retire; WAIT_*/READY/lifecycle states and FAULT (clock must stay alive for
  // the Group reset) are never quiet (T-27 / CLK-01..06).
  io.tileQuiescent := VecInit.tabulate(p.contextCount) { c =>
    states(c).isOneOf(
      ContextState.FREE, ContextState.PARKED, ContextState.DONE_DRAINING,
      ContextState.EXEC_DRAINED, ContextState.RETIRE_READY, ContextState.TASK_COMPLETION,
    ) && obligationsClosed(c)
  }.asUInt.andR

  io.staleCancelRejected     := staleCancel.reduce(_ + _)
  io.staleParkRejected       := stalePark.reduce(_ + _)
  io.staleWakeRejected       := staleWake.reduce(_ + _)
  io.staleCompletionRejected := staleCompletion.reduce(_ + _)
  io.staleAckRejected        := staleAck.reduce(_ + _)
}

/** Named IO for the bounded round-robin issue arbiter (C027 / W2-04, C028 consumes). */
class ContextIssueArbiterIO(p: Parameters) extends Bundle {
  val eligible      = Input(Vec(p.contextCount, Bool()))  // ContextManager.io.eligible
  val blocked       = Input(Vec(p.contextCount, Bool()))  // head blocked: interlock/hazard
  val issueAccepted = Input(Bool()) // the granted bundle was actually issued this cycle
  val grant      = Output(Vec(p.contextCount, Bool()))    // one-hot offer
  val grantValid = Output(Bool())
  val grantIdx   = Output(UInt(p.ctxBits.W))
  val rrPtr      = Output(UInt(p.ctxBits.W))              // rotating pointer (audit)
  val serviceCount  = Output(Vec(p.contextCount, UInt(16.W))) // per-ctx accepted services
  val acceptedTotal = Output(UInt(16.W))                     // accepted issue opportunities
}

/** Bounded round-robin issue arbitration over ContextManager eligibility (C027 / SCH-03).
  *
  * The rotating pointer advances ONLY on an actually-accepted issue, and blocked heads
  * receive no grant, so with N grantable Contexts (N = `p.contextCount`, the C019
  * capacity constraint: cs1_c2 `context_count: 2`) any CONTINUOUSLY grantable Context is
  * granted again within N accepted issue opportunities (k*N + c service form). Measuring
  * the bound in accepted ISSUE OPPORTUNITIES (never cycles) keeps it independent of any
  * stall reason timing. This is a standalone production component: C028 consumes it for
  * per-context fetch/dispatch selection; nothing wires it into SCore implicitly.
  */
class ContextIssueArbiter(p: Parameters) extends Module {
  val io = IO(new ContextIssueArbiterIO(p))

  val ready = VecInit.tabulate(p.contextCount)(c => io.eligible(c) && !io.blocked(c))
  val rr    = RegInit(0.U(p.ctxBits.W))

  private def rotIdx(i: Int): UInt = {
    if (p.contextCount == 1) 0.U
    else {
      val sum = rr +& i.U // expanding add: the compare below must see the true sum
      val sub = sum - p.contextCount.U
      Mux(sum >= p.contextCount.U, sub(p.ctxBits - 1, 0), rr + i.U)
    }
  }
  private def wrapNext(x: UInt): UInt =
    if (p.contextCount == 1) 0.U
    else Mux(x === (p.contextCount - 1).U, 0.U, x +% 1.U)

  val anyReady = ready.asUInt.orR
  io.grantValid := anyReady

  val sel = Wire(UInt(p.ctxBits.W))
  if (p.contextCount == 1) {
    sel := 0.U
    io.grant(0) := ready(0)
  } else {
    val order       = Seq.tabulate(p.contextCount)(i => rotIdx(i))
    val rotatedRdy  = VecInit.tabulate(p.contextCount)(i => ready(rotIdx(i)))
    sel := PriorityMux(rotatedRdy.toSeq, order)
    for (c <- 0 until p.contextCount) {
      io.grant(c) := anyReady && (sel === c.U)
    }
  }

  io.grantIdx := Mux(anyReady, sel, rr)
  io.rrPtr    := rr

  // RR advances ONLY on an accepted issue; an offered-but-unaccepted grant retries the
  // same Context so acceptance ordering is never lost.
  when(io.grantValid && io.issueAccepted) {
    rr := wrapNext(sel)
  }

  val service  = RegInit(VecInit.fill(p.contextCount)(0.U(16.W)))
  val accepted = RegInit(0.U(16.W))
  when(io.grantValid && io.issueAccepted) {
    accepted := accepted + 1.U
  }
  for (c <- 0 until p.contextCount) {
    when(io.grant(c) && io.issueAccepted) {
      service(c) := service(c) + 1.U
    }
  }
  io.serviceCount  := service
  io.acceptedTotal := accepted
}

/** Tile clock-gate predicate (C027 / T-27, ISA-04, CLK-01..06). CoreAxi is the caller.
  *
  * enable = wakeInterrupt || debugWake || !(gateCtl && tileQuiescent)
  *
  * The clock may only be gated when the W6 gate_ctl policy requests it AND the whole
  * Tile is quiescent; any interrupt or debug wake force-enables. The caller folds the
  * reset override into `wakeInterrupt` (parent CoreAxi decision, 2026-11-13).
  */
object ContextClockEnable {
  def apply(gateCtl: Bool, tileQuiescent: Bool, wakeInterrupt: Bool, debugWake: Bool): Bool =
    wakeInterrupt || debugWake || !(gateCtl && tileQuiescent)
}
