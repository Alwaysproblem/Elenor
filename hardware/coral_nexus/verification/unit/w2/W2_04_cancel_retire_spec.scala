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
import chisel3.util.log2Ceil
import org.scalatest.freespec.AnyFreeSpec

import scala.collection.mutable

/** C027 / W2-04 cancel, retire, and bounded-RR behavioral spec (W2 plan §8 item 9).
  *
  * Everything runs on the REAL banks — `BankedRf` (scoreboard, outstanding writes,
  * stale-write reject) and `RetirementPartitionBank` (owner-keyed retirement, live and
  * ready slot masks) — wired to the live ContextManager generations and reuse pulses; no
  * stand-in arrays:
  *   - live cancel stops new issue in the cancel beat, never disturbs the peer: B's RF
  *     scoreboard, ROB live and ready masks, and nonzero reject counter stay bit-for-bit
  *     intact across A's whole lifecycle (RET-06, T-10);
  *   - retire records are captured per retiring lane from the combinational
  *     `perCtxRetiredDelta.owner` (assigned for every retired entry, flags or not) and
  *     asserted as an exact owner sequence; full architectural records are checked
  *     against the independent pinned C021 Spike oracle by test_w2_c027.py;
  *   - bounded RR: a continuously eligible Context is serviced within N accepted issue
  *     opportunities under a saturated peer, with exact 64-vs-32-vs-32 accounting;
  *     blocked heads get no grant; the pointer advances only on accepted issue (SCH-03,
  *     N = p.contextCount from the C019 cs1_c2 constraint context_count: 2);
  *   - slot reuse bumps the generation exactly once, is blocked while RF outstanding
  *     writes are pending (SCore perCtxBackendDrained projection consumed by
  *     obligationsClosed), clears ONLY the owner's banks, and every old-generation
  *     signal — cancel, park, wake, task completion, completion ACK, RF write — is
  *     rejected with exact per-Context counters that clear at the owner's own reuse
  *     (CTX-15, T-08, INV-15);
  *   - same-PC store completion with a wrong op_seq is rejected here too (RET-01; this
  *     is the behavioral target for mutant 0202-pc-key-store.patch).
  *
  * Depends on the parent 0404 additions: `BankedRf.perCtxOutstanding` and
  * `RetirementPartitionBank.perCtxLiveMask`/`perCtxReadyMask`.
  */
class W2_04_CancelRetireSpec extends AnyFreeSpec with ChiselSim {
  def makeParams(ctxCount: Int): Parameters = {
    val p = new Parameters
    p.contextCount       = ctxCount
    p.enableVerification = true
    p.enableFloat        = false
    p.enableRvv          = false
    p.enableVme          = false
    p
  }

  val ADDI_X5_7  = "h00700293".U(32.W)
  val SW_X5_0_X1 = "h0050a023".U(32.W)

  /** Real-module harness: ContextManager + ContextIssueArbiter + BankedRf +
    * RetirementPartitionBank. Identity (perCtxGen) and bank clears (advanceGen) are
    * bridged from the ContextManager exactly like the parent's production wiring, and
    * the SCore backend projection (perCtxBackendDrained) is modeled from the real RF
    * outstanding bitmaps.
    */
  class Harness(p: Parameters) extends Module {
    val cm   = Module(new ContextManager(p))
    val arb  = Module(new ContextIssueArbiter(p))
    val rf   = Module(new BankedRf(p, extraWritePorts = 2))
    val bank = Module(new RetirementPartitionBank(p, mini = false))
    // Host-side observation only; captures EVERY retiring lane before its clock edge.
    val observedRetires = mutable.ArrayBuffer[(Int, BigInt, BigInt)]()
    val architecturalRetires = mutable.ArrayBuffer[String]()

    val io = IO(new Bundle {
      val cmio   = new ContextManagerIO(p)
      val arbio  = new ContextIssueArbiterIO(p)
      val bankio = new RetirementPartitionBankIO(p)

      // Shared decode and execute Context selection for the RF and the ROB partitions
      // (the internal bridges below consume THIS signal, not bankio.activeCtx).
      val selCtx = Input(UInt(p.ctxBits.W))

      // Scoreboard decode-marks (writeAddr) into the selected Context's bank.
      val rfMarkValid = Input(Bool())
      val rfMarkAddr  = Input(UInt(log2Ceil(p.scalarRegCount).W))

      // One test-driven owned writeback port (first extra write port).
      val rfWriteValid = Input(Bool())
      val rfWriteAddr  = Input(UInt(log2Ceil(p.scalarRegCount).W))
      val rfWriteData  = Input(UInt(p.xlen.W))
      val rfWriteOwner = Input(new OwnerTag(p))

      // GPR read-back for the banks-clear-only-at-reuse proof (combinational VecAt).
      val rfDbgIdx  = Input(UInt(log2Ceil(p.scalarRegCount).W))
      val rfDbgData = Output(UInt(p.xlen.W))

      val rfScoreboard  = Output(Vec(p.contextCount, UInt(p.scalarRegCount.W)))
      val rfOutstanding = Output(Vec(p.contextCount, UInt(p.scalarRegCount.W)))
      val rfStale       = Output(UInt(16.W))
    })

    io.cmio <> cm.io
    io.arbio <> arb.io
    io.bankio <> bank.io
    // Internal bridges MUST come after the passthrough connects (last-connect-wins).
    arb.io.eligible := cm.io.eligible
    bank.io.perCtxGen := cm.io.perCtxGen
    bank.io.perCtxReset := cm.io.advanceGen
    rf.io.perCtxGen := cm.io.perCtxGen
    rf.io.perCtxReset := cm.io.advanceGen
    cm.io.robEmpty := bank.io.perCtxEmpty
    // SCore perCtxBackendDrained: the Context's RF outstanding writes all closed.
    cm.io.backendDrained := VecInit.tabulate(p.contextCount)(
      c => rf.io.perCtxOutstanding(c) === 0.U
    )
    // Trap-retired flush of a partition also flushes that Context's Regfile bank.
    rf.io.perCtxFlush := bank.io.perCtxTrapRetired
    bank.io.activeCtx := io.selCtx
    rf.io.activeCtx := io.selCtx

    // Default-drive every BankedRf input (its IO is anonymous; the harness owns defaults).
    for (i <- 0 until p.instructionLanes * 2) {
      rf.io.readAddr(i).valid := false.B
      rf.io.readAddr(i).addr := 0.U
      rf.io.readSet(i).valid := false.B
      rf.io.readSet(i).value := 0.U
    }
    for (i <- 0 until p.instructionLanes) {
      rf.io.writeAddr(i).valid := false.B
      rf.io.writeAddr(i).addr := 0.U
      rf.io.busAddr(i).valid := false.B
      rf.io.busAddr(i).immen := false.B
      rf.io.busAddr(i).immed := 0.U
    }
    rf.io.debugBusPort.idx := io.rfDbgIdx
    rf.io.debugWriteValid := false.B
    for (j <- 0 until rf.totalWritePorts) {
      rf.io.writeData(j).valid := false.B
      rf.io.writeData(j).bits.addr := 0.U
      rf.io.writeData(j).bits.data := 0.U
      rf.io.writeData(j).bits.owner := OwnerTag.zero(p)
      rf.io.writeMask(j).valid := false.B
    }
    // Test-driven decode mark and owned writeback.
    rf.io.writeAddr(0).valid := io.rfMarkValid
    rf.io.writeAddr(0).addr := io.rfMarkAddr
    val wb = rf.io.writeData(p.instructionLanes)
    wb.valid := io.rfWriteValid
    wb.bits.addr := io.rfWriteAddr
    wb.bits.data := io.rfWriteData
    wb.bits.owner := io.rfWriteOwner

    io.rfScoreboard  := rf.io.perCtxScoreboardRegd
    io.rfOutstanding := rf.io.perCtxOutstanding
    io.rfStale       := rf.io.staleWriteRejected
    io.rfDbgData     := rf.io.debugBusPort.data
  }

  def hStep(h: Harness): Unit = {
    for (ctx <- 0 until h.io.bankio.perCtxDeqReady.length) {
      val count = h.io.bankio.perCtxDeqReady(ctx).peek().litValue.toInt
      for (lane <- 0 until count) {
        val owner = h.io.bankio.perCtxRetiredDelta(ctx)(lane).owner
        h.observedRetires += ((
          owner.ctx_slot.peek().litValue.toInt,
          owner.ctx_generation.peek().litValue,
          owner.owner_op_seq.peek().litValue
        ))
        val record = h.io.bankio.perCtxDebug.get(ctx).inst(lane)
        record.valid.expect(true.B)
        val delta = h.io.bankio.perCtxRetiredDelta(ctx)(lane)
        val flags = delta.fflags.peek().litValue |
          (if (delta.vxsat.peek().litToBoolean) BigInt(32) else BigInt(0))
        val rdIndex = record.bits.idx.peek().litValue
        val scalarRd = if (rdIndex > 0 && rdIndex < 32) rdIndex.toString else "null"
        val json = new StringBuilder("C027_RETIRE {")
          .append("\"ctx_slot\":").append(owner.ctx_slot.peek().litValue)
          .append(",\"ctx_generation\":").append(owner.ctx_generation.peek().litValue)
          .append(",\"owner_op_seq\":").append(owner.owner_op_seq.peek().litValue)
          .append(",\"pc_rdata\":").append(record.bits.pc.peek().litValue)
          .append(",\"insn\":").append(record.bits.inst.peek().litValue)
          .append(",\"x_rd\":").append(scalarRd)
          .append(",\"x_rd_val\":").append(record.bits.data.peek().litValue)
          .append(",\"trap\":").append(record.bits.trap.peek().litValue)
          .append(",\"flags_delta\":").append(flags).append("}")
        h.architecturalRetires += json.toString
      }
    }
    h.clock.step()
  }

  /** Step until the Context's ROB partition is empty (retire latency tolerated), then
    * hard-expect emptiness.
    */
  def waitEmpty(h: Harness, ctx: Int, maxCycles: Int = 8): Unit = {
    var i = 0
    while (i < maxCycles && h.io.bankio.perCtxEmpty(ctx).peek().litValue == 0) {
      hStep(h)
      i += 1
    }
    h.io.bankio.perCtxEmpty(ctx).expect(true.B)
  }

  /** Drain the partition and return its complete in-beat retirement history.
    * No deduplication: a repeated retirement is a real correctness failure.
    */
  def watchRetires(
    h: Harness,
    ctx: Int,
    maxCycles: Int = 10
  ): Seq[(Int, BigInt, BigInt)] = {
    waitEmpty(h, ctx, maxCycles)
    h.observedRetires.filter(_._1 == ctx).toVector
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
    b.flagsCapable.foreach(_.poke(false.B))
    b.configCapable.foreach(_.poke(false.B))
    b.cfgPayloadExpected.foreach(_.poke(false.B))
    b.configDelta.foreach(_.valid.poke(false.B))
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

  def idleRf(h: Harness): Unit = {
    h.io.selCtx.poke(0.U)
    h.io.rfMarkValid.poke(false.B)
    h.io.rfMarkAddr.poke(0.U)
    h.io.rfWriteValid.poke(false.B)
    h.io.rfWriteAddr.poke(0.U)
    h.io.rfWriteData.poke(0.U)
    h.io.rfWriteOwner.ctx_slot.poke(0.U)
    h.io.rfWriteOwner.ctx_generation.poke(0.U)
    h.io.rfWriteOwner.owner_op_seq.poke(0.U)
    h.io.rfWriteOwner.subop_id.poke(0.U)
    h.io.rfDbgIdx.poke(0.U)
  }

  def idleArb(a: ContextIssueArbiterIO): Unit = {
    for (i <- 0 until a.blocked.length) { a.blocked(i).poke(false.B) }
    a.issueAccepted.poke(false.B)
  }

  def start(h: Harness): Harness = {
    idleCm(h)
    idleBank(h.io.bankio)
    idleRf(h)
    idleArb(h.io.arbio)
    h.reset.poke(true.B)
    h.clock.step()
    h.reset.poke(false.B)
    h.clock.step()
    h
  }

  /** HELD, ACCEPTED, INIT, READY. */
  def toReady(h: Harness, ctx: Int): Unit = {
    val c = h.io.cmio
    c.holdReq(ctx).poke(true.B)
    hStep(h)
    c.holdReq(ctx).poke(false.B)
    c.accept(ctx).poke(true.B)
    hStep(h)
    c.accept(ctx).poke(false.B)
    c.initStart(ctx).poke(true.B)
    hStep(h)
    c.initStart(ctx).poke(false.B)
    c.initDone(ctx).poke(true.B)
    hStep(h)
    c.initDone(ctx).poke(false.B)
    c.eligible(ctx).expect(true.B)
  }

  def markScoreboard(h: Harness, ctx: Int, addr: Int): Unit = {
    h.io.selCtx.poke(ctx.U)
    h.io.rfMarkValid.poke(true.B)
    h.io.rfMarkAddr.poke(addr.U)
    hStep(h)
    h.io.rfMarkValid.poke(false.B)
  }

  /** Owned RF writeback on the harness port; also the real completion that drains an
    * outstanding scoreboard bit.
    */
  def rfWrite(h: Harness, ctx: Int, gen: Int, seq: Int, addr: Int, data: BigInt): Unit = {
    h.io.rfWriteValid.poke(true.B)
    h.io.rfWriteAddr.poke(addr.U)
    h.io.rfWriteData.poke(data.U)
    h.io.rfWriteOwner.ctx_slot.poke(ctx.U)
    h.io.rfWriteOwner.ctx_generation.poke(gen.U)
    h.io.rfWriteOwner.owner_op_seq.poke(seq.U)
    h.io.rfWriteOwner.subop_id.poke(0.U)
    hStep(h)
    h.io.rfWriteValid.poke(false.B)
  }

  def dispatch(h: Harness, ctx: Int, seq: Int, addr: BigInt, inst: UInt): Unit = {
    h.io.selCtx.poke(ctx.U) // the bridges consume selCtx, not bankio.activeCtx
    val b = h.io.bankio
    b.dispatchOpSeq(0).poke(seq.U)
    b.inst(0).valid.poke(true.B)
    b.inst(0).bits.addr.poke(addr.U)
    b.inst(0).bits.inst.poke(inst)
    val hasRd = inst.litValue == ADDI_X5_7.litValue
    if (hasRd) {
      b.writeAddrScalar(0).valid.poke(true.B)
      b.writeAddrScalar(0).addr.poke(5.U)
    }
    hStep(h)
    b.inst(0).valid.poke(false.B)
    b.writeAddrScalar(0).valid.poke(false.B)
    b.dispatchOpSeq(0).poke(0.U)
  }

  def writebackScalar(h: Harness, ctx: Int, gen: Int, seq: Int): Unit = {
    val w = h.io.bankio.writeDataScalar(0)
    w.valid.poke(true.B)
    w.bits.addr.poke(5.U)
    w.bits.data.poke(7.U) // ADDI x5,x0,7; checked independently against Spike.
    w.bits.owner.ctx_slot.poke(ctx.U)
    w.bits.owner.ctx_generation.poke(gen.U)
    w.bits.owner.owner_op_seq.poke(seq.U)
    w.bits.owner.subop_id.poke(0.U)
    hStep(h)
    w.valid.poke(false.B)
  }

  def dbgGpr(h: Harness, ctx: Int, addr: Int): BigInt = {
    h.io.selCtx.poke(ctx.U)
    h.io.rfDbgIdx.poke(addr.U)
    h.io.rfDbgData.peek().litValue
  }

  /** From RETIRE_READY: TASK_COMPLETION, then a LIVE-generation ack frees the slot with
    * an exactly-once reuse pulse and a generation bump. Callers must have drained the
    * Context's RF outstanding bits first (obligationsClosed consumes backendDrained).
    */
  def finishTask(h: Harness, ctx: Int, gen: Int): Unit = {
    val c = h.io.cmio
    c.taskCompletion(ctx).poke(true.B)
    c.taskCompletionGen(ctx).poke(gen.U)
    hStep(h)
    c.taskCompletion(ctx).poke(false.B)
    c.completionAck(ctx).poke(true.B)
    c.completionAckGen(ctx).poke(gen.U)
    c.advanceGen(ctx).expect(true.B) // pulse fires in the ack beat (obligations closed)
    hStep(h)
    c.advanceGen(ctx).expect(false.B) // exactly once: slot is FREE, ack may stay high
    hStep(h)
    c.completionAck(ctx).poke(false.B)
    c.perCtxGen(ctx).expect((gen + 1).U)
  }


  "Live cancel preserves peer progress while accepted store drains" in {
    var bObserved = Seq.empty[(Int, BigInt, BigInt)]
    var aObserved = Seq.empty[(Int, BigInt, BigInt)]
    simulate(new Harness(makeParams(2))) { dut =>
      start(dut)
      toReady(dut, 0)
      toReady(dut, 1)
      dispatch(dut, 0, 20, BigInt(0x1000), SW_X5_0_X1)
      dispatch(dut, 0, 21, BigInt(0x1004), ADDI_X5_7)
      dispatch(dut, 1, 30, BigInt(0x2000), ADDI_X5_7)
      dispatch(dut, 1, 31, BigInt(0x2004), ADDI_X5_7)
      dut.io.arbio.issueAccepted.poke(true.B)

      // The cancel beat itself: issue for A stops immediately (combinational eligible).
      dut.io.cmio.cancelValid(0).poke(true.B)
      dut.io.cmio.cancelGen(0).poke(0.U)
      dut.io.cmio.eligible(0).expect(false.B)
      dut.io.cmio.cancelAck(0).expect(true.B)
      dut.io.cmio.perCtxCancelled(0).expect(true.B)
      dut.io.arbio.grant(0).expect(false.B)
      hStep(dut)
      dut.io.cmio.cancelValid(0).poke(false.B)
      dut.io.cmio.eligible(0).expect(false.B) // DONE_DRAINING holds
      dut.io.cmio.perCtxCancelled(0).expect(true.B)

      // B retires to completion while A's completions are withheld (A head stalled).
      val aSpace = dut.io.bankio.perCtxSpace(0).peek().litValue
      writebackScalar(dut, 1, 0, 30)
      writebackScalar(dut, 1, 0, 31)
      val bRetired = watchRetires(dut, 1)
      assert(dut.io.bankio.perCtxSpace(0).peek().litValue == aSpace) // A untouched

      // Accepted work of the canceled Context still completes and retires.
      val done = dut.io.bankio.storeComplete
      done.bits.ctx_slot.poke(0.U)
      done.bits.ctx_generation.poke(0.U)
      done.bits.owner_op_seq.poke(20.U)
      done.bits.subop_id.poke(0.U)
      done.valid.poke(true.B)
      hStep(dut)
      done.valid.poke(false.B)
      writebackScalar(dut, 0, 0, 21)
      val aRetired = watchRetires(dut, 0)
      hStep(dut) // DONE_DRAINING -> EXEC_DRAINED
      hStep(dut) // EXEC_DRAINED -> RETIRE_READY
      dut.io.cmio.retireReady(0).expect(true.B)
      bObserved = bRetired
      aObserved = aRetired
      dut.architecturalRetires.foreach(println)
    }

    // Exact retire-record content and order, not bare counts.
    assert(aObserved == Seq((0, BigInt(0), BigInt(20)), (0, BigInt(0), BigInt(21))))
    assert(bObserved == Seq((1, BigInt(0), BigInt(30)), (1, BigInt(0), BigInt(31))))
  }

  "Cancel and reuse preserve every peer scoreboard and ROB bit" in {
    simulate(new Harness(makeParams(2))) { dut =>
      start(dut)
      toReady(dut, 0)
      toReady(dut, 1)

      // B carries a nonzero stale-reject history before A's lifecycle runs.
      dut.io.selCtx.poke(1.U)
      dut.io.rfWriteValid.poke(true.B)
      dut.io.rfWriteAddr.poke(1.U)
      dut.io.rfWriteData.poke(0xdead.U)
      dut.io.rfWriteOwner.ctx_slot.poke(1.U)
      dut.io.rfWriteOwner.ctx_generation.poke(5.U) // wrong generation for B
      dut.io.rfWriteOwner.owner_op_seq.poke(9.U)
      hStep(dut)
      dut.io.rfWriteValid.poke(false.B)
      val bStaleBefore = dut.io.rfStale.peek().litValue
      assert(bStaleBefore >= 1)

      // Scoreboard patterns: A {2,4}, B {1,3}.
      markScoreboard(dut, 0, 2)
      markScoreboard(dut, 0, 4)
      markScoreboard(dut, 1, 1)
      markScoreboard(dut, 1, 3)
      val aPat = (BigInt(1) << 2) | (BigInt(1) << 4)
      val bPat = (BigInt(1) << 1) | (BigInt(1) << 3)
      assert(dut.io.rfScoreboard(0).peek().litValue == aPat)
      assert(dut.io.rfScoreboard(1).peek().litValue == bPat)

      dispatch(dut, 0, 20, BigInt(0x1000), ADDI_X5_7)
      dispatch(dut, 1, 30, BigInt(0x2000), ADDI_X5_7) // B head: stays pending
      dispatch(dut, 1, 31, BigInt(0x2004), ADDI_X5_7) // B younger entry
      writebackScalar(dut, 1, 0, 31) // completes EARLY, cannot retire behind 30

      // B's ROB slot bitmaps before the cancel: BOTH masks nonzero (an older pending
      // entry keeps liveMask populated; the younger completed entry keeps readyMask
      // populated), so an accidental clearing of completed OR live B entries during
      // A's lifecycle is observable instead of silently hanging B.
      val bLive  = dut.io.bankio.perCtxLiveMask(1).peek().litValue
      val bReady = dut.io.bankio.perCtxReadyMask(1).peek().litValue
      assert(bLive.bitCount == 2) // both B entries live
      assert(bReady.bitCount == 1) // exactly the younger entry complete
      assert(bReady != 0 && (bLive & bReady) == bReady)

      // Cancel A; the cancel itself must NOT clear any bank (only safe reuse does).
      dut.io.cmio.cancelValid(0).poke(true.B)
      dut.io.cmio.cancelGen(0).poke(0.U)
      hStep(dut)
      dut.io.cmio.cancelValid(0).poke(false.B)
      assert(dut.io.rfScoreboard(0).peek().litValue == aPat) // survives cancel
      assert(dut.io.rfScoreboard(1).peek().litValue == bPat)
      assert(dut.io.bankio.perCtxLiveMask(1).peek().litValue == bLive)
      assert(dut.io.bankio.perCtxReadyMask(1).peek().litValue == bReady)

      // A's ROB entry retires, but the drain CANNOT reach RETIRE_READY while A's RF
      // outstanding writes (x2, x4) are still pending.
      writebackScalar(dut, 0, 0, 20)
      waitEmpty(dut, 0)
      for (_ <- 0 until 4) {
        dut.io.cmio.retireReady(0).expect(false.B) // blocked by backendDrained=0
        dut.io.cmio.advanceGen(0).expect(false.B)
        hStep(dut)
      }
      assert(dut.io.rfOutstanding(0).peek().litValue == aPat)

      // Real RF completions for A's x2 and x4 drain the outstanding bits; the written
      // architectural data is now readable and must survive until the reuse pulse.
      rfWrite(dut, 0, 0, 20, 2, BigInt(0xa5a5))
      rfWrite(dut, 0, 0, 21, 4, BigInt(0x5a5a))
      assert(dut.io.rfOutstanding(0).peek().litValue == 0)
      assert(dbgGpr(dut, 0, 2) == BigInt(0xa5a5)) // NOT cleared yet
      hStep(dut) // DONE_DRAINING -> EXEC_DRAINED
      hStep(dut) // EXEC_DRAINED -> RETIRE_READY
      dut.io.cmio.retireReady(0).expect(true.B)
      finishTask(dut, 0, 0)
      assert(dbgGpr(dut, 0, 2) == 0) // GPR cleared at safe reuse only
      assert(dbgGpr(dut, 0, 4) == 0)
      assert(dut.io.rfOutstanding(0).peek().litValue == 0)

      // Peer preservation, bit-for-bit, across A's whole lifecycle including the pulse.
      assert(dut.io.rfScoreboard(1).peek().litValue == bPat)
      assert(dut.io.bankio.perCtxLiveMask(1).peek().litValue == bLive)
      assert(dut.io.bankio.perCtxReadyMask(1).peek().litValue == bReady)
      dut.io.bankio.perCtxEmpty(1).expect(false.B) // B's partition content intact
      assert(dut.io.rfStale.peek().litValue == bStaleBefore) // B's counter survived

      // B finishes in the correct record order afterward: completing the older head
      // retires 30 THEN the early-completed 31.
      writebackScalar(dut, 1, 0, 30)
      val bRetired = watchRetires(dut, 1)
      assert(bRetired == Seq((1, BigInt(0), BigInt(30)), (1, BigInt(0), BigInt(31))))
    }
  }

  "Bounded RR advances only on accepted eligible issue" in {
    simulate(new Harness(makeParams(2))) { dut =>
      start(dut)
      toReady(dut, 0)
      toReady(dut, 1)
      val n = 2 // p.contextCount == 2; C019 cs1_c2 constraint context_count: 2

      // Saturated peers: every offered grant is accepted; one accept per cycle.
      dut.io.arbio.issueAccepted.poke(true.B)
      var lastAService = BigInt(-1)
      val cycles       = 64
      val total0       = dut.io.arbio.acceptedTotal.peek().litValue
      for (_ <- 0 until cycles) {
        val before0 = dut.io.arbio.serviceCount(0).peek().litValue
        hStep(dut)
        val s0  = dut.io.arbio.serviceCount(0).peek().litValue
        val s1  = dut.io.arbio.serviceCount(1).peek().litValue
        val acc = dut.io.arbio.acceptedTotal.peek().litValue
        if (s0 > before0) {
          if (lastAService >= 0) {
            assert(acc - lastAService <= n, // serviced within N accepted opportunities
              s"A waited ${acc - lastAService} accepted opportunities (bound $n)")
          }
          lastAService = acc
        }
        // Fairness under saturation: service counts never diverge by more than one.
        assert((s0 - s1).abs <= 1)
      }
      // Exact accounting: every cycle accepted one issue, split evenly.
      assert(dut.io.arbio.acceptedTotal.peek().litValue - total0 == BigInt(cycles))
      assert(dut.io.arbio.serviceCount(0).peek().litValue == BigInt(cycles / 2))
      assert(dut.io.arbio.serviceCount(1).peek().litValue == BigInt(cycles / 2))

      // Blocked heads receive NO grant (and do not consume the pointer).
      dut.io.arbio.blocked(1).poke(true.B)
      for (_ <- 0 until 5) {
        dut.io.arbio.grant(1).expect(false.B)
        dut.io.arbio.grant(0).expect(true.B)
        hStep(dut)
      }
      dut.io.arbio.blocked(0).poke(true.B)
      for (_ <- 0 until 3) {
        dut.io.arbio.grantValid.expect(false.B)
        hStep(dut)
      }
      val frozen = dut.io.arbio.acceptedTotal.peek().litValue
      hStep(dut)
      assert(dut.io.arbio.acceptedTotal.peek().litValue == frozen) // nothing accepted

      // Offered-but-unaccepted grants hold the pointer: order is preserved.
      dut.io.arbio.blocked(0).poke(false.B)
      dut.io.arbio.blocked(1).poke(false.B)
      dut.io.arbio.issueAccepted.poke(false.B)
      val held        = dut.io.arbio.grantIdx.peek().litValue
      val heldService = dut.io.arbio.serviceCount(held.toInt).peek().litValue
      for (_ <- 0 until 5) {
        dut.io.arbio.grantIdx.expect(held.U) // same Context re-offered every cycle
        hStep(dut)
      }
      assert(dut.io.arbio.serviceCount(held.toInt).peek().litValue == heldService)
      dut.io.arbio.issueAccepted.poke(true.B)
      hStep(dut) // the held offer is accepted; the pointer must now move past it
      dut.io.arbio.grantIdx.expect((1 - held).U)
    }
  }

  "Context reuse rejects and counts stale generation signals" in {
    simulate(new Harness(makeParams(2))) { dut =>
      start(dut)
      toReady(dut, 0)
      toReady(dut, 1)

      // First lifecycle: normal RET path into reuse (gen 0 -> 1); A has no RF marks.
      dispatch(dut, 0, 20, BigInt(0x1000), ADDI_X5_7)
      writebackScalar(dut, 0, 0, 20)
      waitEmpty(dut, 0)
      dut.io.cmio.retIssued(0).poke(true.B)
      hStep(dut)
      dut.io.cmio.retIssued(0).poke(false.B)
      hStep(dut) // DONE_DRAINING -> EXEC_DRAINED (robEmpty and backendDrained both live)
      hStep(dut) // EXEC_DRAINED -> RETIRE_READY
      dut.io.cmio.retireReady(0).expect(true.B)
      finishTask(dut, 0, 0)
      dut.io.cmio.perCtxGen(0).expect(1.U)

      // A second task occupies the slot at generation 1.
      toReady(dut, 0)

      // Old-generation cancel: counted once, NOT honored (A keeps issuing).
      dut.io.cmio.cancelValid(0).poke(true.B)
      dut.io.cmio.cancelGen(0).poke(0.U)
      dut.io.cmio.cancelAck(0).expect(false.B)
      dut.io.cmio.eligible(0).expect(true.B)
      hStep(dut)
      dut.io.cmio.cancelValid(0).poke(false.B)
      assert(dut.io.cmio.staleCancelRejected.peek().litValue == 1)
      dut.io.cmio.perCtxCancelled(0).expect(false.B)

      // Old-generation park and wake: counted once each, never applied.
      dut.io.cmio.parkValid(0).poke(true.B)
      dut.io.cmio.parkGen(0).poke(0.U)
      hStep(dut)
      dut.io.cmio.parkValid(0).poke(false.B)
      assert(dut.io.cmio.staleParkRejected.peek().litValue == 1)
      dut.io.cmio.perCtxParked(0).expect(false.B)
      dut.io.cmio.eligible(0).expect(true.B)

      dut.io.cmio.wakeValid(0).poke(true.B)
      dut.io.cmio.wakeGen(0).poke(0.U)
      hStep(dut)
      dut.io.cmio.wakeValid(0).poke(false.B)
      assert(dut.io.cmio.staleWakeRejected.peek().litValue == 1)

      // Old-generation TASK_COMPLETION at RETIRE_READY: counted, ignored — a then-ack
      // cannot free the slot (it never reached TASK_COMPLETION).
      dut.io.cmio.retIssued(0).poke(true.B)
      hStep(dut)
      dut.io.cmio.retIssued(0).poke(false.B)
      hStep(dut)
      hStep(dut)
      dut.io.cmio.retireReady(0).expect(true.B)
      dut.io.cmio.taskCompletion(0).poke(true.B)
      dut.io.cmio.taskCompletionGen(0).poke(0.U) // stale
      hStep(dut)
      dut.io.cmio.taskCompletion(0).poke(false.B)
      assert(dut.io.cmio.staleCompletionRejected.peek().litValue == 1)
      dut.io.cmio.retireReady(0).expect(true.B)
      dut.io.cmio.completionAck(0).poke(true.B)
      dut.io.cmio.completionAckGen(0).poke(1.U) // live gen, but no TASK_COMPLETION
      dut.io.cmio.advanceGen(0).expect(false.B) // never reached TASK_COMPLETION
      hStep(dut)
      dut.io.cmio.completionAck(0).poke(false.B)
      dut.io.cmio.retireReady(0).expect(true.B)

      // Live completion moves to TASK_COMPLETION; an OLD-generation ACK is counted and
      // cannot free the newer owner; the valid ACK then advances exactly once.
      dut.io.cmio.taskCompletion(0).poke(true.B)
      dut.io.cmio.taskCompletionGen(0).poke(1.U)
      hStep(dut)
      dut.io.cmio.taskCompletion(0).poke(false.B)
      dut.io.cmio.completionAck(0).poke(true.B)
      dut.io.cmio.completionAckGen(0).poke(0.U) // stale ACK
      dut.io.cmio.advanceGen(0).expect(false.B)
      hStep(dut)
      dut.io.cmio.completionAck(0).poke(false.B)
      assert(dut.io.cmio.staleAckRejected.peek().litValue == 1)
      dut.io.cmio.completionAck(0).poke(true.B)
      dut.io.cmio.completionAckGen(0).poke(1.U) // valid ACK
      dut.io.cmio.advanceGen(0).expect(true.B)
      hStep(dut)
      dut.io.cmio.completionAck(0).poke(false.B)
      dut.io.cmio.advanceGen(0).expect(false.B)
      dut.io.cmio.perCtxGen(0).expect(2.U)

      // All five old-owner counters cleared at the reuse edge.
      assert(dut.io.cmio.staleCancelRejected.peek().litValue == 0)
      assert(dut.io.cmio.staleParkRejected.peek().litValue == 0)
      assert(dut.io.cmio.staleWakeRejected.peek().litValue == 0)
      assert(dut.io.cmio.staleCompletionRejected.peek().litValue == 0)
      assert(dut.io.cmio.staleAckRejected.peek().litValue == 0)

      // Old-generation RF write after reuse: rejected by the REAL BankedRf, counted,
      // and no scoreboard bit changes anywhere.
      markScoreboard(dut, 1, 3) // B carries a live pending bit
      dut.io.selCtx.poke(0.U)
      dut.io.rfWriteValid.poke(true.B)
      dut.io.rfWriteAddr.poke(6.U)
      dut.io.rfWriteData.poke(0xcafe.U)
      dut.io.rfWriteOwner.ctx_slot.poke(0.U)
      dut.io.rfWriteOwner.ctx_generation.poke(0.U) // stale for A (now generation 2)
      dut.io.rfWriteOwner.owner_op_seq.poke(20.U)
      hStep(dut)
      dut.io.rfWriteValid.poke(false.B)
      assert(dut.io.rfStale.peek().litValue >= 1)
      assert(dut.io.rfScoreboard(0).peek().litValue == 0)
      assert(dut.io.rfScoreboard(1).peek().litValue == (BigInt(1) << 3))
    }
  }

  "Same PC stores require exact parent identity" in {
    simulate(new Harness(makeParams(2))) { dut =>
      start(dut)
      toReady(dut, 0)
      toReady(dut, 1)
      val sharedPc = BigInt(0x30001000)
      dispatch(dut, 0, 7, sharedPc, SW_X5_0_X1)
      dispatch(dut, 1, 7, sharedPc, SW_X5_0_X1) // identical PC, identical op_seq
      dut.io.bankio.perCtxEmpty(0).expect(false.B)
      dut.io.bankio.perCtxEmpty(1).expect(false.B)

      // Complete B's store: only B retires; A's same-PC store stays in flight.
      val sc = dut.io.bankio.storeComplete
      sc.valid.poke(true.B)
      sc.bits.ctx_slot.poke(1.U)
      sc.bits.ctx_generation.poke(0.U)
      sc.bits.owner_op_seq.poke(7.U)
      sc.bits.subop_id.poke(0.U)
      hStep(dut)
      sc.valid.poke(false.B)
      hStep(dut)
      dut.io.bankio.perCtxEmpty(1).expect(true.B)
      dut.io.bankio.perCtxEmpty(0).expect(false.B)

      // Wrong op_seq (same PC legacy key): must NOT complete A's store.
      sc.valid.poke(true.B)
      sc.bits.ctx_slot.poke(0.U)
      sc.bits.ctx_generation.poke(0.U)
      sc.bits.owner_op_seq.poke(8.U)
      hStep(dut)
      sc.valid.poke(false.B)
      hStep(dut)
      dut.io.bankio.perCtxEmpty(0).expect(false.B)

      // Matching OwnerTag completes A cleanly.
      sc.valid.poke(true.B)
      sc.bits.ctx_slot.poke(0.U)
      sc.bits.ctx_generation.poke(0.U)
      sc.bits.owner_op_seq.poke(7.U)
      hStep(dut)
      sc.valid.poke(false.B)
      hStep(dut)
      dut.io.bankio.perCtxEmpty(0).expect(true.B)
    }
  }

  "C1 cancel waits for RF drainage before reuse" in {
    simulate(new Harness(makeParams(1))) { dut =>
      start(dut)
      toReady(dut, 0)
      markScoreboard(dut, 0, 2)
      dispatch(dut, 0, 20, BigInt(0x1000), ADDI_X5_7)

      dut.io.cmio.cancelValid(0).poke(true.B)
      dut.io.cmio.cancelGen(0).poke(0.U)
      dut.io.cmio.cancelAck(0).expect(true.B)
      hStep(dut)
      dut.io.cmio.cancelValid(0).poke(false.B)

      writebackScalar(dut, 0, 0, 20)
      waitEmpty(dut, 0)
      // ROB drained, but the outstanding x2 write still blocks the terminal states.
      for (_ <- 0 until 3) {
        dut.io.cmio.retireReady(0).expect(false.B)
        hStep(dut)
      }
      rfWrite(dut, 0, 0, 20, 2, BigInt(0xb7))
      assert(dut.io.rfOutstanding(0).peek().litValue == 0)
      assert(dbgGpr(dut, 0, 2) == BigInt(0xb7))
      hStep(dut)
      hStep(dut)
      dut.io.cmio.retireReady(0).expect(true.B)
      finishTask(dut, 0, 0)
      assert(dbgGpr(dut, 0, 2) == 0) // cleared at the reuse pulse only
      dut.io.cmio.perCtxGen(0).expect(1.U)
    }
  }
}
