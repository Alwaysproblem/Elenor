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
  * FREE -> HELD -> ACCEPTED -> INIT -> READY -> WAIT_*(sub) -> DONE_DRAINING -> EXEC_DRAINED
  *   -> RETIRE_READY -> TASK_COMPLETION -> (ack) -> FREE ; FAULT is a terminal for fail-stop.
  */
object ContextState extends ChiselEnum {
  val FREE, HELD, ACCEPTED, INIT, READY = Value
  val WAIT_EVENT, WAIT_ENGINE_QUEUE, WAIT_STREAM, WAIT_REGION, WAIT_FETCH = Value
  val PARKED = Value
  val DONE_DRAINING, EXEC_DRAINED, RETIRE_READY, TASK_COMPLETION = Value
  val FAULT = Value
}

/** Per-Context control inputs owned by W6/W3 projections (read-only here). */
class CtxExternalIO(p: Parameters) extends Bundle {
  val waitBusy   = Input(Bool()) // W6: a wait record of this Context is not yet satisfied
  val regionBusy = Input(Bool()) // W3: this Context holds a pending VME region interaction
}

/** Context slot manager: park / cancel / fail-stop separation + EXEC_DRAINED + slot reuse
  * (C027 / W2-04, W2-D06 / CTX-15 / SCH-07 / SCH-11 / RET-06).
  *
  * This module NEVER gates the core clock (park only stops NEW issue for its Context) and
  * NEVER touches another Context's state on cancel (RET-06 hard invariant).
  */
class ContextManager(p: Parameters) extends Module {
  val io = IO(new Bundle {
    val activeCtx = Input(UInt(p.ctxBits.W))

    // Admission / lifecycle inputs (from W1 boundary coordinator; C035 wires the real ones).
    val holdReq   = Input(Vec(p.contextCount, Bool()))
    val accept    = Input(Vec(p.contextCount, Bool()))
    val initDone  = Input(Vec(p.contextCount, Bool()))
    val retIssued = Input(Vec(p.contextCount, Bool())) // RET instruction retired this Context

    // Park reasons (level; only stop new issue of THIS Context, SCH-07).
    val parkReq = Input(Vec(p.contextCount, Bool())) // WFI / WAIT / region head parking
    // Wait-record / region projections from W6/W3 (read-only summaries).
    val ext = Input(Vec(p.contextCount, new CtxExternalIO(p)))

    // Local quiescence inputs for EXEC_DRAINED (this Context's own obligations).
    val robEmpty       = Input(Vec(p.contextCount, Bool()))
    val pendingLaunch  = Input(Vec(p.contextCount, Bool())) // any un-completed engine launch
    val faultCtx       = Input(Vec(p.contextCount, Bool())) // fault raised for this Context
    val taskCompletion = Input(Vec(p.contextCount, Bool())) // W6 reliably stored TASK_COMPLETION
    val completionAck  = Input(Vec(p.contextCount, Bool())) // Group consumed the completion

    // cancel (W6 CANCEL_TASK): only THIS Context's issue stops; peers untouched.
    val cancelValid = Input(Vec(p.contextCount, Bool()))
    val cancelGen   = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))

    // Outputs.
    val ctxState   = Output(Vec(p.contextCount, ContextState()))
    val perCtxGen  = Output(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))
    val eligible   = Output(Vec(p.contextCount, Bool())) // may issue NEW instructions
    val execDrained = Output(Vec(p.contextCount, Bool()))
    val retireReady = Output(Vec(p.contextCount, Bool()))
    val advanceGen  = Output(Vec(p.contextCount, Bool())) // slot reuse: bump generation & clear banks
    val tileIssueFreeze = Output(Bool())                  // fail-stop: any FAULT stops ALL issue
    val tileQuiescent   = Output(Bool())                  // all parked/idle AND backends empty
    val staleCancelRejected = Output(UInt(16.W))
  })

  val states = RegInit(VecInit.fill(p.contextCount)(ContextState.FREE))
  val gens   = RegInit(VecInit.fill(p.contextCount)(0.U(p.ctxGenerationBits.W)))
  val staleCancel = RegInit(0.U(16.W))

  for (c <- 0 until p.contextCount) {
    val s = states(c)

    // --- cancel acceptance: only the live generation is honoured (stale = idempotent reject).
    val cancelLive = io.cancelValid(c) && (io.cancelGen(c) === gens(c))
    val cancelStale = io.cancelValid(c) && (io.cancelGen(c) =/= gens(c))
    when(cancelStale) {
      staleCancel := staleCancel + 1.U
    }

    // --- park (level, from WFI/WAIT/region): never flushes, never clears RF/ROB (SCH-07).
    val parked = io.parkReq(c) || io.ext(c).waitBusy || io.ext(c).regionBusy

    switch(s) {
      is(ContextState.FREE) {
        when(io.holdReq(c)) {
          s := ContextState.HELD
        }
      }
      is(ContextState.HELD) {
        when(io.accept(c)) {
          s := ContextState.ACCEPTED
        }.elsewhen(io.holdReq(c)) {
          s := ContextState.HELD // hold may persist until admission authorizes
        }
      }
      is(ContextState.ACCEPTED) {
        when(io.initDone(c)) {
          s := ContextState.READY
        }
      }
      is(ContextState.READY) {
        when(io.faultCtx(c)) {
          s := ContextState.FAULT
        }.elsewhen(cancelLive) {
          s := ContextState.DONE_DRAINING // cancel stops new issue; accepted work still drains
        }.elsewhen(io.retIssued(c)) {
          s := ContextState.DONE_DRAINING
        }.elsewhen(parked) {
          s := ContextState.PARKED
        }
      }
      is(ContextState.PARKED) {
        when(io.faultCtx(c)) {
          s := ContextState.FAULT
        }.elsewhen(cancelLive) {
          s := ContextState.DONE_DRAINING
        }.elsewhen(!parked) {
          s := ContextState.READY // wake resumes at the ORIGINAL head (no flush happened)
        }
      }
      is(ContextState.WAIT_EVENT) {
        when(!io.ext(c).waitBusy) { s := ContextState.READY }
      }
      is(ContextState.WAIT_ENGINE_QUEUE) {
        s := ContextState.READY // engine-queue waits are re-probed by the scheduler each tick
      }
      is(ContextState.WAIT_STREAM) {
        s := ContextState.READY
      }
      is(ContextState.WAIT_REGION) {
        when(!io.ext(c).regionBusy) { s := ContextState.READY }
      }
      is(ContextState.WAIT_FETCH) {
        s := ContextState.READY
      }
      is(ContextState.DONE_DRAINING) {
        when(io.faultCtx(c)) {
          s := ContextState.FAULT
        }.elsewhen(io.robEmpty(c) && !io.pendingLaunch(c) && !io.ext(c).waitBusy && !io.ext(c).regionBusy) {
          s := ContextState.EXEC_DRAINED
        }
      }
      is(ContextState.EXEC_DRAINED) {
        s := ContextState.RETIRE_READY
      }
      is(ContextState.RETIRE_READY) {
        when(io.taskCompletion(c)) {
          s := ContextState.TASK_COMPLETION
        }
      }
      is(ContextState.TASK_COMPLETION) {
        when(io.completionAck(c)) {
          s := ContextState.FREE // conservative CTX-15: all obligations closed before reuse
        }
      }
      is(ContextState.FAULT) {
        s := ContextState.FAULT // fail-stop terminal; recovery is a Group reset (W6), not here
      }
    }
  }

  // Generation bump happens exactly at the FREE transition (slot reuse).
  for (c <- 0 until p.contextCount) {
    io.advanceGen(c) := (states(c) === ContextState.TASK_COMPLETION) && io.completionAck(c)
    when(io.advanceGen(c)) {
      gens(c) := gens(c) + 1.U
    }
  }

  // Issue eligibility: READY (and transient WAIT_* re-probe states) and not parked; the
  // scheduler still applies its own RR / prefix rules on top of this.
  for (c <- 0 until p.contextCount) {
    val s = states(c)
    val waitProbe = s.isOneOf(
      ContextState.WAIT_EVENT, ContextState.WAIT_ENGINE_QUEUE, ContextState.WAIT_STREAM,
      ContextState.WAIT_REGION, ContextState.WAIT_FETCH,
    )
    val parked = io.parkReq(c) || io.ext(c).waitBusy || io.ext(c).regionBusy
    io.eligible(c) := (s === ContextState.READY || waitProbe) && !parked
  }

  io.ctxState    := states
  io.perCtxGen   := gens
  io.execDrained := VecInit.tabulate(p.contextCount)(c => states(c) === ContextState.EXEC_DRAINED)
  io.retireReady := VecInit.tabulate(p.contextCount)(c => states(c) === ContextState.RETIRE_READY)

  // fail-stop (SCH-11/FLT-01): ANY Context FAULT freezes issue for the whole Tile; the
  // flush itself still only clears the faulting Context's partitions (owned elsewhere).
  io.tileIssueFreeze := VecInit.tabulate(p.contextCount)(c => states(c) === ContextState.FAULT).asUInt.orR

  // Tile quiescence: a Context is quiet ONLY if it is parked/terminal AND its real
  // backend obligations have drained — ROB partition empty, no pending engine launch,
  // no wait record and no region interaction in flight (W2 plan §8 item 8). A parked
  // Context with a non-empty ROB still has accepted work to retire, so the Tile must
  // NOT be called quiescent (gate_ctl must keep the clock running).
  val allQuiet = VecInit.tabulate(p.contextCount) { c =>
    val s = states(c)
    val parkedOrTerminal = s.isOneOf(
      ContextState.PARKED, ContextState.DONE_DRAINING, ContextState.EXEC_DRAINED,
      ContextState.RETIRE_READY, ContextState.TASK_COMPLETION, ContextState.FAULT,
      ContextState.FREE,
    )
    val backendsDrained = io.robEmpty(c) && !io.pendingLaunch(c) &&
      !io.ext(c).waitBusy && !io.ext(c).regionBusy
    parkedOrTerminal && backendsDrained
  }.asUInt.andR
  io.tileQuiescent := allQuiet

  io.staleCancelRejected := staleCancel
}
