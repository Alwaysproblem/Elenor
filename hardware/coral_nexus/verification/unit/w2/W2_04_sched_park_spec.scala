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
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec

/** C027 / W2-04 sched/park behavioral spec (W2 plan §8 item 8, T-04/T-27/T-28/T-62,
  * ISA-04, CLK-01..06, SCH-08, SCH-11/FLT-01, MAT-12).
  *
  * Real modules only: ContextManager + ContextIssueArbiter + RetirementPartitionBank
  * (mini) + the ContextClockEnable predicate. WAIT_* heads stall issue only; the peer
  * keeps issuing and retiring; wakes resume at the ORIGINAL head (no flush ever fires
  * from park/wait); WFI + peer outstanding never reaches tile quiescence.
  */
class W2_04_SchedParkSpec extends AnyFreeSpec with ChiselSim {
  def makeParams(ctxCount: Int): Parameters = {
    val p = new Parameters
    p.contextCount       = ctxCount
    p.enableVerification = false
    p.enableFloat        = false
    p.enableRvv          = false
    p.enableVme          = false
    p
  }

  val ADDI_X5_7 = "h00700293".U(32.W)

  /** Real-module harness: arbiter eligibility and bank identity/reset are bridged from
    * the ContextManager so park/cancel behavior is observed on the live banks.
    */
  class Harness(p: Parameters) extends Module {
    val cm   = Module(new ContextManager(p))
    val arb  = Module(new ContextIssueArbiter(p))
    val bank = Module(new RetirementPartitionBank(p, mini = true))
    val io = IO(new Bundle {
      val cmio   = new ContextManagerIO(p)
      val arbio  = new ContextIssueArbiterIO(p)
      val bankio = new RetirementPartitionBankIO(p)
    })
    io.cmio <> cm.io
    io.arbio <> arb.io
    io.bankio <> bank.io
    // Internal bridges MUST come after the passthrough connects (last-connect-wins).
    arb.io.eligible := cm.io.eligible
    bank.io.perCtxGen := cm.io.perCtxGen
    bank.io.perCtxReset := cm.io.advanceGen
    cm.io.robEmpty := bank.io.perCtxEmpty
  }

  /** Wraps the ContextClockEnable predicate (function object) as a testable module. */
  class ClockPredDut extends Module {
    val io = IO(new Bundle {
      val gateCtl       = Input(Bool())
      val tileQuiescent = Input(Bool())
      val wakeInterrupt = Input(Bool())
      val debugWake     = Input(Bool())
      val enable        = Output(Bool())
    })
    io.enable := ContextClockEnable(io.gateCtl, io.tileQuiescent, io.wakeInterrupt, io.debugWake)
  }

  def idleCm(h: Harness): Unit = {
    val c = h.io.cmio
    c.activeCtx.poke(0.U)
    for (i <- 0 until c.holdReq.length) {
      c.holdReq(i).poke(false.B)
      c.accept(i).poke(false.B)
      c.initStart(i).poke(false.B)
      c.initDone(i).poke(false.B)
      c.retIssued(i).poke(false.B)
      c.parkValid(i).poke(false.B)
      c.parkGen(i).poke(0.U)
      c.wakeValid(i).poke(false.B)
      c.wakeGen(i).poke(0.U)
      c.waitKind(i).poke(ContextWait.NONE)
      c.ext(i).waitBusy.poke(false.B)
      c.ext(i).regionBusy.poke(false.B)
      c.pendingLaunch(i).poke(false.B)
      c.faultCtx(i).poke(false.B)
      c.taskCompletion(i).poke(false.B)
      c.taskCompletionGen(i).poke(0.U)
      c.completionAck(i).poke(false.B)
      c.completionAckGen(i).poke(0.U)
      c.backendDrained(i).poke(true.B)
      c.cancelValid(i).poke(false.B)
      c.cancelGen(i).poke(0.U)
    }
  }

  def idleBank(b: RetirementPartitionBankIO): Unit = {
    b.activeCtx.poke(0.U)
    for (i <- 0 until b.inst.length) {
      b.dispatchOpSeq(i).poke(0.U)
      b.inst(i).valid.poke(false.B)
      b.inst(i).ready.poke(true.B)
      b.inst(i).bits.addr.poke(0.U)
      b.inst(i).bits.inst.poke(0.U)
      b.inst(i).bits.brchFwd.poke(false.B)
      b.targets(i).poke(0.U)
      b.jalrTargets(i).poke(0.U)
      b.jump(i).poke(false.B)
      b.branch(i).poke(false.B)
      b.writeAddrScalar(i).valid.poke(false.B)
      b.writeAddrScalar(i).addr.poke(0.U)
    }
    b.storeComplete.valid.poke(false.B)
    b.storeComplete.bits.ctx_slot.poke(0.U)
    b.storeComplete.bits.ctx_generation.poke(0.U)
    b.storeComplete.bits.owner_op_seq.poke(0.U)
    b.storeComplete.bits.subop_id.poke(0.U)
    for (j <- 0 until b.writeDataScalar.length) {
      b.writeDataScalar(j).valid.poke(false.B)
      b.writeDataScalar(j).bits.addr.poke(0.U)
      b.writeDataScalar(j).bits.data.poke(0.U)
      b.writeDataScalar(j).bits.owner.ctx_slot.poke(0.U)
      b.writeDataScalar(j).bits.owner.ctx_generation.poke(0.U)
      b.writeDataScalar(j).bits.owner.owner_op_seq.poke(0.U)
      b.writeDataScalar(j).bits.owner.subop_id.poke(0.U)
    }
    b.fault.valid.poke(false.B)
    b.fault.bits.mepc.poke(0.U)
    b.fault.bits.mtval.poke(0.U)
    b.fault.bits.mcause.poke(0.U)
    b.fault.bits.decode.poke(false.B)
    b.fault.bits.owner.ctx_slot.poke(0.U)
    b.fault.bits.owner.ctx_generation.poke(0.U)
    b.fault.bits.owner.owner_op_seq.poke(0.U)
    b.fault.bits.owner.subop_id.poke(0.U)
    for (j <- 0 until b.flagsDelta.length) {
      b.flagsDelta(j).valid.poke(false.B)
      b.flagsDelta(j).bits.valid.poke(false.B)
      b.flagsDelta(j).bits.owner.ctx_slot.poke(0.U)
      b.flagsDelta(j).bits.owner.ctx_generation.poke(0.U)
      b.flagsDelta(j).bits.owner.owner_op_seq.poke(0.U)
      b.flagsDelta(j).bits.owner.subop_id.poke(0.U)
      b.flagsDelta(j).bits.rob_tag.ctx_slot.poke(0.U)
      b.flagsDelta(j).bits.rob_tag.rob_slot.poke(0.U)
      b.flagsDelta(j).bits.rob_tag.slot_generation.poke(0.U)
      b.flagsDelta(j).bits.fflags.poke(0.U)
      b.flagsDelta(j).bits.vxsat.poke(false.B)
      b.flagsDelta(j).bits.replaceFflags.poke(false.B)
      b.flagsDelta(j).bits.replaceVxsat.poke(false.B)
    }
    b.csrPark.valid.poke(false.B)
    b.csrPark.bits.owner.ctx_slot.poke(0.U)
    b.csrPark.bits.owner.ctx_generation.poke(0.U)
    b.csrPark.bits.owner.owner_op_seq.poke(0.U)
    b.csrPark.bits.owner.subop_id.poke(0.U)
    b.csrPark.bits.fflags.poke(0.U)
    b.csrPark.bits.vxsat.poke(false.B)
    b.csrPark.bits.replaceFflags.poke(false.B)
    b.csrPark.bits.replaceVxsat.poke(false.B)
    b.csrPark.bits.commitsFrm.poke(false.B)
    b.csrPark.bits.frmValue.poke(0.U)
  }

  def idleArb(a: ContextIssueArbiterIO): Unit = {
    for (i <- 0 until a.blocked.length) { a.blocked(i).poke(false.B) }
    a.issueAccepted.poke(false.B)
  }

  def step(h: Harness): Unit = {
    h.clock.step()
  }

  /** HELD -> ACCEPTED -> INIT -> READY. */
  def toReady(h: Harness, ctx: Int): Unit = {
    val c = h.io.cmio
    c.holdReq(ctx).poke(true.B)
    step(h)
    c.holdReq(ctx).poke(false.B)
    c.accept(ctx).poke(true.B)
    step(h)
    c.accept(ctx).poke(false.B)
    c.initStart(ctx).poke(true.B)
    step(h)
    c.initStart(ctx).poke(false.B)
    c.initDone(ctx).poke(true.B)
    step(h)
    c.initDone(ctx).poke(false.B)
    c.eligible(ctx).expect(true.B)
  }

  def dispatch(h: Harness, ctx: Int, seq: Int, addr: BigInt): Unit = {
    val b = h.io.bankio
    b.activeCtx.poke(ctx.U)
    b.dispatchOpSeq(0).poke(seq.U)
    b.inst(0).valid.poke(true.B)
    b.inst(0).bits.addr.poke(addr.U)
    b.inst(0).bits.inst.poke(ADDI_X5_7)
    b.writeAddrScalar(0).valid.poke(true.B)
    b.writeAddrScalar(0).addr.poke(5.U)
    step(h)
    b.inst(0).valid.poke(false.B)
    b.writeAddrScalar(0).valid.poke(false.B)
    b.dispatchOpSeq(0).poke(0.U)
  }

  def writebackScalar(h: Harness, ctx: Int, gen: Int, seq: Int): Unit = {
    val w = h.io.bankio.writeDataScalar(0)
    w.valid.poke(true.B)
    w.bits.addr.poke(5.U)
    w.bits.data.poke(seq.U)
    w.bits.owner.ctx_slot.poke(ctx.U)
    w.bits.owner.ctx_generation.poke(gen.U)
    w.bits.owner.owner_op_seq.poke(seq.U)
    w.bits.owner.subop_id.poke(0.U)
    step(h)
    w.valid.poke(false.B)
  }

  /** Step until the Context's ROB partition is empty (retire latency tolerated). */
  def waitEmpty(h: Harness, ctx: Int, maxCycles: Int = 8): Unit = {
    var i = 0
    while (i < maxCycles && h.io.bankio.perCtxEmpty(ctx).peek().litValue == 0) {
      h.clock.step()
      i += 1
    }
    h.io.bankio.perCtxEmpty(ctx).expect(true.B)
  }

  def start(h: Harness): Harness = {
    idleCm(h)
    idleBank(h.io.bankio)
    idleArb(h.io.arbio)
    h.reset.poke(true.B)
    step(h)
    h.reset.poke(false.B)
    step(h)
    h
  }

  "Region wait parks only owner and wake resumes original head" in {
    simulate(new Harness(makeParams(2))) { dut =>
      start(dut)
      toReady(dut, 0)
      toReady(dut, 1)

      dispatch(dut, 0, 20, BigInt(0x1000))
      dispatch(dut, 0, 21, BigInt(0x1004))
      dut.io.bankio.perCtxEmpty(0).expect(false.B)
      val aSpace = dut.io.bankio.perCtxSpace(0).peek().litValue

      // Region head: W3 says this Context holds a pending region interaction.
      dut.io.cmio.ext(0).regionBusy.poke(true.B)
      step(dut)
      dut.io.cmio.perCtxWaiting(0).expect(true.B)
      dut.io.cmio.eligible(0).expect(false.B)

      // B keeps being served; A never gets a grant while waiting.
      dut.io.arbio.issueAccepted.poke(true.B)
      for (_ <- 0 until 4) {
        dut.io.arbio.grant(0).expect(false.B)
        dut.io.arbio.grant(1).expect(true.B)
        step(dut)
      }
      // Park/wait NEVER flushes: A's partition content is untouched (original head).
      assert(dut.io.bankio.perCtxSpace(0).peek().litValue == aSpace)
      dut.io.bankio.perCtxEmpty(0).expect(false.B)

      // region_done: A wakes and re-issues from the SAME head (entries still in order).
      dut.io.cmio.ext(0).regionBusy.poke(false.B)
      step(dut)
      dut.io.cmio.perCtxWaiting(0).expect(false.B)
      dut.io.cmio.eligible(0).expect(true.B)
      dut.io.bankio.perCtxEmpty(0).expect(false.B) // nothing was flushed away

      // The pre-wake entries retire in their ORIGINAL dispatch order: entry 20 leaves
      // first; entry 21 empties the partition only after its own completion.
      writebackScalar(dut, 0, 0, 20)
      dut.io.bankio.perCtxEmpty(0).expect(false.B) // younger entry still parked
      writebackScalar(dut, 0, 0, 21)
      waitEmpty(dut, 0)
    }
  }

  "Event wait rejects stale wake and accepts live wake" in {
    simulate(new Harness(makeParams(2))) { dut =>
      start(dut)
      toReady(dut, 0)
      toReady(dut, 1)

      dut.io.cmio.ext(0).waitBusy.poke(true.B)
      step(dut)
      dut.io.cmio.perCtxWaiting(0).expect(true.B)
      dut.io.cmio.eligible(0).expect(false.B)

      // Stale wake (generation 1 while slot 0 is generation 0): rejected + counted.
      dut.io.cmio.wakeValid(0).poke(true.B)
      dut.io.cmio.wakeGen(0).poke(1.U)
      step(dut)
      dut.io.cmio.wakeValid(0).poke(false.B)
      assert(dut.io.cmio.staleWakeRejected.peek().litValue == 1)
      dut.io.cmio.perCtxWaiting(0).expect(true.B)
      dut.io.cmio.eligible(0).expect(false.B)

      // Live tagged wake satisfies the record even though the level summary lags.
      dut.io.cmio.wakeValid(0).poke(true.B)
      dut.io.cmio.wakeGen(0).poke(0.U)
      step(dut)
      dut.io.cmio.wakeValid(0).poke(false.B)
      dut.io.cmio.ext(0).waitBusy.poke(false.B)
      step(dut)
      dut.io.cmio.perCtxWaiting(0).expect(false.B)
      dut.io.cmio.eligible(0).expect(true.B)
    }
  }

  "Park allows retirement and outstanding work prevents clock gating" in {
    simulate(new Harness(makeParams(2))) { dut =>
      start(dut)
      toReady(dut, 0)
      toReady(dut, 1)
      dispatch(dut, 0, 20, BigInt(0x1000))
      dispatch(dut, 1, 30, BigInt(0x2000))

      // WFI park: observable in the request beat itself.
      dut.io.cmio.parkValid(0).poke(true.B)
      dut.io.cmio.parkGen(0).poke(0.U)
      dut.io.cmio.eligible(0).expect(false.B)
      dut.io.cmio.perCtxParked(0).expect(true.B)
      step(dut)

      // B outstanding => never quiescent.
      dut.io.cmio.tileQuiescent.expect(false.B)

      // Park does NOT stop retirement: A's entry retires while A stays parked.
      writebackScalar(dut, 0, 0, 20)
      waitEmpty(dut, 0)
      dut.io.cmio.perCtxParked(0).expect(true.B)
      dut.io.cmio.tileQuiescent.expect(false.B) // B still owes work

      // B retires and parks too: now the whole Tile is quiet.
      writebackScalar(dut, 1, 0, 30)
      waitEmpty(dut, 1)
      dut.io.cmio.parkValid(1).poke(true.B)
      dut.io.cmio.parkGen(1).poke(0.U)
      step(dut)
      dut.io.cmio.tileQuiescent.expect(true.B)
    }
  }

  "Clock gating requires quiescence and permits interrupt or debug wake" in {
    simulate(new ClockPredDut) { dut =>
      def drive(gate: Boolean, quiet: Boolean, irq: Boolean, dbg: Boolean): Unit = {
        dut.io.gateCtl.poke(gate.B)
        dut.io.tileQuiescent.poke(quiet.B)
        dut.io.wakeInterrupt.poke(irq.B)
        dut.io.debugWake.poke(dbg.B)
      }
      drive(false, true, false, false)
      dut.io.enable.expect(true.B) // no gating policy: always on
      drive(true, false, false, false)
      dut.io.enable.expect(true.B) // not quiescent: keep clocking
      drive(true, true, false, false)
      dut.io.enable.expect(false.B) // policy + quiescent: gate
      drive(true, true, true, false)
      dut.io.enable.expect(true.B) // interrupt wake
      drive(true, true, false, true)
      dut.io.enable.expect(true.B) // debug wake
    }
  }

  "Same beat wake preserves completing peer grant" in {
    simulate(new Harness(makeParams(2))) { dut =>
      start(dut)
      toReady(dut, 0)
      toReady(dut, 1)
      dispatch(dut, 0, 20, BigInt(0x1000))
      dispatch(dut, 1, 30, BigInt(0x2000))

      // A waits on a region; B stays READY with an outstanding entry.
      dut.io.cmio.ext(0).regionBusy.poke(true.B)
      step(dut)
      dut.io.cmio.perCtxWaiting(0).expect(true.B)

      // Beat t: A's wake arrives in the SAME beat as B's completing writeback.
      dut.io.cmio.ext(0).regionBusy.poke(false.B)
      val w = dut.io.bankio.writeDataScalar(0)
      w.valid.poke(true.B)
      w.bits.addr.poke(5.U)
      w.bits.data.poke(30.U)
      w.bits.owner.ctx_slot.poke(1.U)
      w.bits.owner.ctx_generation.poke(0.U)
      w.bits.owner.owner_op_seq.poke(30.U)
      w.bits.owner.subop_id.poke(0.U)

      // Eligibility is state-based, so A cannot be granted in the wake beat itself.
      dut.io.arbio.issueAccepted.poke(true.B)
      dut.io.arbio.grant(0).expect(false.B)
      step(dut)
      w.valid.poke(false.B)

      // B's retirement (completion -> retire) proceeds; A only becomes grantable the
      // beat AFTER its wake beat (the rotation itself may still favor B — not asserted).
      dut.io.cmio.eligible(0).expect(true.B)
      waitEmpty(dut, 1)
    }
  }

  "Fault immediately freezes all issue until reset" in {
    simulate(new Harness(makeParams(2))) { dut =>
      start(dut)
      toReady(dut, 0)
      toReady(dut, 1)
      dut.io.arbio.issueAccepted.poke(true.B)

      // The fault input beat itself: freeze covers the peer's issue immediately.
      dut.io.cmio.faultCtx(0).poke(true.B)
      dut.io.cmio.tileIssueFreeze.expect(true.B)
      dut.io.cmio.eligible(0).expect(false.B)
      dut.io.cmio.eligible(1).expect(false.B) // peer frozen same beat
      dut.io.arbio.grantValid.expect(false.B)
      step(dut)

      // Latched: freeze survives the fault pulse dropping.
      dut.io.cmio.faultCtx(0).poke(false.B)
      for (_ <- 0 until 3) {
        dut.io.cmio.tileIssueFreeze.expect(true.B)
        dut.io.cmio.eligible(1).expect(false.B)
        step(dut)
      }
      dut.io.cmio.tileQuiescent.expect(false.B) // FAULT keeps the clock alive for reset

      // Recovery is a reset, nothing local.
      dut.reset.poke(true.B)
      step(dut)
      dut.reset.poke(false.B)
      step(dut)
      dut.io.cmio.tileIssueFreeze.expect(false.B)
    }
  }

  "Every wait reason can enter and leave ready state" in {
    simulate(new Harness(makeParams(2))) { dut =>
      start(dut)
      toReady(dut, 0)
      toReady(dut, 1)

      // W6 wait record -> WAIT_EVENT; clears when the record is satisfied.
      dut.io.cmio.ext(0).waitBusy.poke(true.B)
      step(dut)
      dut.io.cmio.perCtxWaiting(0).expect(true.B)
      dut.io.cmio.eligible(0).expect(false.B)
      dut.io.cmio.ext(0).waitBusy.poke(false.B)
      step(dut)
      dut.io.cmio.perCtxWaiting(0).expect(false.B)
      dut.io.cmio.eligible(0).expect(true.B)

      // W3 region -> WAIT_REGION.
      dut.io.cmio.ext(1).regionBusy.poke(true.B)
      step(dut)
      dut.io.cmio.perCtxWaiting(1).expect(true.B)
      dut.io.cmio.ext(1).regionBusy.poke(false.B)
      step(dut)
      dut.io.cmio.perCtxWaiting(1).expect(false.B)

      // Scheduler-side reasons (C028 drives these): ENGINE_QUEUE / STREAM / FETCH.
      for (
        k <- Seq(
          ContextWait.ENGINE_QUEUE, ContextWait.STREAM, ContextWait.FETCH,
          ContextWait.OPERAND, ContextWait.DEST,
        )
      ) {
        dut.io.cmio.waitKind(0).poke(k)
        step(dut)
        dut.io.cmio.perCtxWaiting(0).expect(true.B)
        dut.io.cmio.eligible(0).expect(false.B)
        dut.io.cmio.waitKind(0).poke(ContextWait.NONE)
        step(dut)
        dut.io.cmio.perCtxWaiting(0).expect(false.B)
        dut.io.cmio.eligible(0).expect(true.B)
      }
    }
  }

  "Same beat waits and return block issue until all obligations clear" in {
    simulate(new Harness(makeParams(2))) { dut =>
      start(dut)
      toReady(dut, 0)
      toReady(dut, 1)

      // A newly-asserted W6 wait record blocks issue in the SAME beat (before any
      // state move), as do a scheduler wait reason and a same-beat RET.
      dut.io.cmio.ext(0).waitBusy.poke(true.B)
      dut.io.cmio.eligible(0).expect(false.B) // blocked in the assertion beat itself
      step(dut) // READY -> WAIT_EVENT
      dut.io.cmio.perCtxWaiting(0).expect(true.B)
      dut.io.cmio.ext(0).waitBusy.poke(false.B)
      dut.io.cmio.eligible(0).expect(false.B) // still WAIT_EVENT until the edge
      step(dut) // WAIT_EVENT -> READY
      dut.io.cmio.eligible(0).expect(true.B)

      dut.io.cmio.waitKind(0).poke(ContextWait.OPERAND)
      dut.io.cmio.eligible(0).expect(false.B) // blocked in the assertion beat itself
      step(dut) // READY -> WAIT_OPERAND
      dut.io.cmio.perCtxWaiting(0).expect(true.B)
      dut.io.cmio.waitKind(0).poke(ContextWait.NONE)
      dut.io.cmio.eligible(0).expect(false.B) // still WAIT_OPERAND until the edge
      step(dut) // WAIT_OPERAND -> READY
      dut.io.cmio.eligible(0).expect(true.B)

      dut.io.cmio.retIssued(0).poke(true.B)
      dut.io.cmio.eligible(0).expect(false.B)
      step(dut)
      dut.io.cmio.retIssued(0).poke(false.B)

      // PARKED wake with the W6 wait record still busy: the one-beat READY transit
      // must NOT open an issue window (slot 1 is still READY here; slot 0 draining).
      dut.io.cmio.parkValid(1).poke(true.B)
      dut.io.cmio.parkGen(1).poke(0.U)
      step(dut)
      dut.io.cmio.ext(1).waitBusy.poke(true.B) // busy while parked
      step(dut)
      dut.io.cmio.parkValid(1).poke(false.B) // wake beat: PARKED -> READY next cycle
      step(dut)
      dut.io.cmio.eligible(1).expect(false.B) // READY transit, but waitBusy holds
      dut.io.cmio.perCtxWaiting(1).expect(false.B) // not yet WAIT_EVENT
      step(dut)
      dut.io.cmio.perCtxWaiting(1).expect(true.B) // now properly WAIT_EVENT
      dut.io.cmio.eligible(1).expect(false.B)
      dut.io.cmio.ext(1).waitBusy.poke(false.B)
      step(dut)
      dut.io.cmio.eligible(1).expect(true.B)
    }
  }

  "C1 park resumes its only eligible context" in {
    simulate(new Harness(makeParams(1))) { dut =>
      start(dut)
      toReady(dut, 0)
      dispatch(dut, 0, 20, BigInt(0x1000))

      dut.io.arbio.issueAccepted.poke(true.B)
      dut.io.arbio.grant(0).expect(true.B)
      dut.io.arbio.grantIdx.expect(0.U)

      dut.io.cmio.parkValid(0).poke(true.B)
      dut.io.cmio.parkGen(0).poke(0.U)
      dut.io.cmio.eligible(0).expect(false.B)
      dut.io.arbio.grantValid.expect(false.B)
      step(dut)

      // Retire continues under park, then quiescence can settle.
      writebackScalar(dut, 0, 0, 20)
      waitEmpty(dut, 0)
      dut.io.cmio.tileQuiescent.expect(true.B)

      dut.io.cmio.parkValid(0).poke(false.B)
      step(dut)
      dut.io.cmio.eligible(0).expect(true.B)
      dut.io.arbio.grant(0).expect(true.B)
    }
  }
}
