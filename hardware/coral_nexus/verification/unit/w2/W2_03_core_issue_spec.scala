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
import chisel3.util.{Valid, log2Ceil}
import org.scalatest.freespec.AnyFreeSpec

import scala.collection.mutable

/** C028 / W2-03 core-level behavioral spec: the REAL `SCore` — `PerCtxFetchMux` +
  * `Scheduler` + `DispatchV2` + ALU/BRU/MLU/DVU/LSU + `BankedRf` +
  * `RetirementPartitionBank` + `CtxCsr` — executes TWO instruction streams under
  * AUTOMATIC eligible round-robin. Nobody drives issue selection: the spec drives
  * only the physical buses (legacy fixed-next-cycle IBus, DTCM-row DBus) and the
  * lifecycle boundary (accepted / programReady / entry pulse), then observes the
  * COMBINATIONAL lane accepts (`debug.dispatch(i).instFire`) with the same-beat
  * `issueCtx` grant.
  *
  * Proven per Context (CTX/SCH/FET acceptance):
  *   - exact architectural results: real MLU/DVU arithmetic, real LSU store+load
  *     round trip through DTCM (LUI-based data base), own-PC branch loops taken and
  *     mispredicted-not-taken with resolution redirects, per-Context GPR isolation
  *     checked through the banked-RF observation port;
  *   - exact-once issue/retire accounting: `owner_op_seq` and accumulated retires
  *     match the executed program length exactly, no cross-Context lane ever mixes
  *     into a bundle, zero RF/ROB/CSR stale rejects (frontend stale rejects are
  *     legitimate around redirects and are not pinned to zero);
  *   - automatic RR interleave with a REAL operand stall: while Context 0's head is
  *     WAIT_OPERAND on its own scoreboard, the peer accepts work in those very beats;
  *   - WFI parks each Context independently and the Tile reaches tileQuiescent with
  *     both backends drained.
  *
  * No fake injected completion: every architectural value comes from the real units'
  * writebacks through the real ROB/RF banks.
  */
class W2_03_CoreIssueSpec extends AnyFreeSpec with ChiselSim {
  import W2_03Encoding._

  private val ctxCount = 2
  require(
    C028TargetParams.contextCounts.contains(ctxCount),
    "sealed C019 targets must contain the cs1_c2 context_count used here"
  )

  def makeParams: Parameters = {
    val p = new Parameters(m = Seq(
      new MemoryRegion(0x00000000, 0x2000, MemoryRegionType.IMEM),
      new MemoryRegion(0x00010000, 0x10000, MemoryRegionType.DMEM)
    ))
    p.contextCount = ctxCount
    p.enableVerification = true
    p.enableFloat = false
    p.enableRvv = false
    p.enableVme = false
    p.fetchDataBits = 128
    p.lsuDataBits = 128
    p
  }


  // Both programs address DTCM through x10 = lui 0x10 -> 0x00010000.
  // The wide DIV has an immediate result consumer, so A really stalls while B
  // has independent ALU/MLU/memory work. Merely checking a DIV at program end
  // would not exercise the scheduler's operand-blocked head.
  val p0 = Seq[BigInt](
    lui(10, 0x10), addi(1, 0, 5), addi(2, 0, 7), lui(12, 0x80000),
    addi(12, 12, -1), div(9, 12, 1), addi(11, 9, 0), mul(3, 1, 2),
    addi(4, 3, 0), sw(3, 10, 0), lw(5, 10, 0), addi(6, 0, 3),
    addi(6, 6, -1), bne(6, 0, -4), addi(7, 6, 100), WFI
  )
  // Context 1 @0x400: distinct constants, 5-iteration loop, word at 0x10004.
  val p1 = Seq[BigInt](
    lui(10, 0x10), addi(1, 0, 11), addi(2, 0, 13), mul(3, 1, 2),
    addi(4, 3, 0), sw(3, 10, 4), lw(5, 10, 4), addi(6, 0, 5),
    div(9, 2, 1), addi(6, 6, -1), bne(6, 0, -4), addi(7, 6, 200), WFI
  )
  val entry0   = BigInt(0x000)
  val entry1   = BigInt(0x400)
  val nRetire0 = 12 + 2 * 3 + 2 // 20 executed
  val nRetire1 = 9 + 2 * 5 + 2 // 21

  val imem: Map[BigInt, BigInt] = {
    val rows = mutable.ArrayBuffer[(BigInt, BigInt)]()
    rows ++= p0.grouped(4).toSeq.zipWithIndex.map { case (ws, i) =>
      (entry0 + 16 * i) -> line(ws: _*)
    } :+ (entry0 + 16 * ((p0.size + 3) / 4) -> line(NOP, NOP, NOP, NOP))
    rows ++= p1.grouped(4).toSeq.zipWithIndex.map { case (ws, i) =>
      (entry1 + 16 * i) -> line(ws: _*)
    } :+ (entry1 + 16 * ((p1.size + 3) / 4) -> line(NOP, NOP, NOP, NOP))
    rows.toMap
  }

  val expected0 = Map(1 -> 5, 2 -> 7, 3 -> 35, 4 -> 35, 5 -> 35, 6 -> 0, 7 -> 100,
    9 -> (Int.MaxValue / 5), 10 -> 0x10000, 11 -> (Int.MaxValue / 5), 12 -> Int.MaxValue)
  val expected1 = Map(1 -> 11, 2 -> 13, 3 -> 143, 4 -> 143, 5 -> 143, 6 -> 0, 7 -> 200,
    9 -> 1, 10 -> 0x10000)

  /** Legacy-timing bus functional model, driven from the Scala test thread. Phases
    * are observed AFTER the pokes of each beat; the IBus has a deterministic 1-in-3
    * ready backpressure so the frontend key bridge and LSU-priority arbitration are
    * actually exercised. DBus rows are lsuDataBits/8 bytes (16B for this target).
    */
  class BusModel(dut: SCore, p: Parameters) {
    private val rowLsb    = log2Ceil(p.lsuDataBits / 8)
    private var ibusPend  = Option.empty[BigInt]
    private var dbusPend  = Option.empty[BigInt]
    private val dtcm      = mutable.Map[BigInt, BigInt]()
    private var cycle     = 0

    def poke(): Unit = {
      // --- IBus request phase (fetch / LSU ITCM traffic share this port) ---
      val iReady = (cycle % 3) != 2
      dut.io.ibus.ready.poke(iReady.B)
      dut.io.ibus.rdata.poke(ibusPend.getOrElse(BigInt(0)).U)
      dut.io.ibus.fault.valid.poke(false.B)
      dut.io.ibus.fault.bits.write.poke(false.B)
      dut.io.ibus.fault.bits.addr.poke(0.U)
      dut.io.ibus.fault.bits.epc.poke(0.U)
      // Observe the request phase AFTER the ready poke.
      val iValid = dut.io.ibus.valid.peek().litToBoolean
      val iAddr  = dut.io.ibus.addr.peek().litValue
      val iFire  = iValid && iReady
      ibusPend = if (iFire) Some(imem.getOrElse(iAddr, BigInt(0))) else None

      // --- DBus (DTCM row interface: row write with byte mask, row read) ---
      dut.io.dbus.ready.poke(true.B)
      dut.io.dbus.rdata.poke(dbusPend.getOrElse(BigInt(0)).U)
      val dValid = dut.io.dbus.valid.peek().litToBoolean
      val dWrite = dut.io.dbus.write.peek().litToBoolean
      val dRow   = dut.io.dbus.addr.peek().litValue >> rowLsb
      if (dValid) {
        if (dWrite) {
          val data = dut.io.dbus.wdata.peek().litValue
          val mask = expandMask(dut.io.dbus.wmask.peek().litValue, p.lsuDataBits)
          dtcm(dRow) = dtcm.getOrElse(dRow, BigInt(0)) & ~mask | (data & mask)
          dbusPend = None
        } else {
          dbusPend = Some(dtcm.getOrElse(dRow, BigInt(0)))
        }
      } else dbusPend = None

      // --- EBus (unused external port) ---
      dut.io.ebus.dbus.ready.poke(true.B)
      dut.io.ebus.dbus.rdata.poke(0.U)
      dut.io.ebus.fault.valid.poke(false.B)
      dut.io.ebus.fault.bits.write.poke(false.B)
      dut.io.ebus.fault.bits.addr.poke(0.U)
      dut.io.ebus.fault.bits.epc.poke(0.U)

      cycle += 1
    }

    def expandMask(mask: BigInt, bits: Int): BigInt = {
      var acc = BigInt(0)
      var m   = mask
      var b   = 0
      while (b < bits / 8) {
        if (m.testBit(0)) acc |= BigInt(0xff) << (8 * b)
        m >>= 1
        b += 1
      }
      acc
    }

    def dtcmWord(row: BigInt, word: Int): BigInt =
      (dtcm.getOrElse(row, BigInt(0)) >> (32 * word)) & BigInt(0xffffffffL)
  }

  "Two contexts execute real streams with automatic RR and exact results" in {
    val p = makeParams
    simulate(new SCore(p)) { dut =>
      // --- quiescent inputs (leaf pokes only; no hardware construction here) ---
      dut.io.irq.poke(false.B)
      dut.io.timer_irq.poke(false.B)
      dut.io.software_irq.poke(false.B)
      dut.io.dm.debug_req.poke(false.B)
      dut.io.dm.resume_req.poke(false.B)
      dut.io.dm.csr.valid.poke(false.B)
      dut.io.dm.csr.bits.addr.poke(0.U)
      dut.io.dm.csr.bits.index.poke(0.U)
      dut.io.dm.csr.bits.rs1.poke(0.U)
      dut.io.dm.csr.bits.op.poke(CsrOp.CSRRW)
      dut.io.dm.csr_rs1.poke(0.U)
      dut.io.dm.scalar_rd.valid.poke(false.B)
      dut.io.dm.scalar_rd.bits.addr.poke(0.U)
      dut.io.dm.scalar_rd.bits.data.poke(0.U)
      dut.io.dm.scalar_rs.idx.poke(0.U)
      dut.io.csr.in.value.foreach(_.poke(0.U))

      // --- Context lifecycle boundary ---
      dut.io.ctx.ctxSelect.poke(0.U) // debug/DM bank ownership only; no issue effect
      dut.io.ctx.advanceGen.foreach(_.poke(false.B))
      dut.io.ctx.perCtxPark.foreach(_.poke(false.B))
      dut.io.ctx.parkGen.foreach(_.poke(0.U))
      dut.io.ctx.cancelValid.foreach(_.poke(false.B))
      dut.io.ctx.cancelGen.foreach(_.poke(0.U))
      dut.io.ctx.pendingLaunch.foreach(_.poke(false.B))
      dut.io.ctx.waitBusy.foreach(_.poke(false.B))
      dut.io.ctx.regionBusy.foreach(_.poke(false.B))
      dut.io.ctx.ctxAccepted.foreach(_.poke(false.B))
      dut.io.ctx.programReady.foreach(_.poke(false.B))
      dut.io.ctx.entryStart.foreach(_.valid.poke(false.B))
      dut.io.ctx.entryStart.foreach(_.bits.poke(0.U))
      dut.io.ctx.rfObserve.poke(true.B)
      dut.io.ctx.rfDebugAddr.poke(0.U)

      val bus = new BusModel(dut, p)
      dut.reset.poke(true.B)
      bus.poke()
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      // --- accept both Tasks; program READY; one entry-start pulse each ---
      def launch(ctx: Int, entry: BigInt): Unit = {
        dut.io.ctx.ctxAccepted(ctx).poke(true.B)
        dut.io.ctx.programReady(ctx).poke(true.B)
        dut.io.ctx.entryStart(ctx).valid.poke(true.B)
        dut.io.ctx.entryStart(ctx).bits.poke(entry.U)
        bus.poke()
        dut.clock.step()
        dut.io.ctx.entryStart(ctx).valid.poke(false.B)
      }
      launch(0, entry0)
      launch(1, entry1)

      // --- run: pair same-beat combinational accepts with the same-beat grant ---
      val fires = dut.io.debug.get.dispatch.map(_.instFire) // combinational lane accepts
      def firesOf(c: Int): Int =
        (0 until fires.length).count(i =>
          fires(i).peek().litToBoolean && dut.io.ctx.issueCtx.peek().litValue == c)

      val retired    = Array(0L, 0L)
      val accepted   = Array(0L, 0L)
      var sawInterleave         = false
      var sawPeerThroughStall   = false
      var bothDone              = false
      var cycle                 = 0
      val maxCycles             = 8000
      while (!bothDone && cycle < maxCycles) {
        bus.poke()
        val sel = dut.io.ctx.issueCtx.peek().litValue.toInt
        val f0  = firesOf(0)
        val f1  = firesOf(1)
        assert(!(f0 > 0 && f1 > 0), "two Contexts accepted lanes in the same beat")
        if (f0 > 0) assert(sel == 0, "accepted lanes without the matching RR grant")
        if (f1 > 0) assert(sel == 1, "accepted lanes without the matching RR grant")
        accepted(0) += f0
        accepted(1) += f1
        if ((f0 > 0 && accepted(1) > 0) || (f1 > 0 && accepted(0) > 0)) sawInterleave = true
        // REAL operand stall: Context 0's head is WAIT_OPERAND on its own scoreboard
        // while the peer ACCEPTS work in this very beat.
        if (dut.io.ctx.perCtxWaitReason(0).peek().litValue ==
              IssueStallReason.WAIT_OPERAND.litValue && f1 > 0) sawPeerThroughStall = true
        dut.clock.step()
        retired(0) += dut.io.ctx.perCtxNRetired(0).peek().litValue.toLong
        retired(1) += dut.io.ctx.perCtxNRetired(1).peek().litValue.toLong
        bothDone = retired(0) >= nRetire0 && retired(1) >= nRetire1
        cycle += 1
      }
      assert(bothDone, s"streams did not finish: retired=${retired.toList}, cycle=$cycle")
      assert(retired(0) == nRetire0, s"ctx0 retired ${retired(0)} != $nRetire0")
      assert(retired(1) == nRetire1, s"ctx1 retired ${retired(1)} != $nRetire1")
      assert(accepted(0) == nRetire0.toLong, s"ctx0 accepted ${accepted(0)} lanes")
      assert(accepted(1) == nRetire1.toLong, s"ctx1 accepted ${accepted(1)} lanes")
      assert(
        dut.io.ctx.perCtxNextOpSeq(0).peek().litValue == BigInt(nRetire0) &&
          dut.io.ctx.perCtxNextOpSeq(1).peek().litValue == BigInt(nRetire1),
        "owner_op_seq must equal the accepted lane counts"
      )
      assert(accepted(0) > 0 && accepted(1) > 0, "automatic RR never served both")
      assert(sawInterleave, "no interleaved issue: RR degenerated to one Context")
      assert(sawPeerThroughStall,
        "peer never accepted work through a same-beat WAIT_OPERAND head")

      // --- identity: zero stale rejects across RF/ROB/CSR (frontend stale rejects
      // around branch redirects are legitimate and deliberately not pinned) ---
      dut.io.ctx.staleRfRejected.expect(0.U)
      dut.io.ctx.staleRobRejected.expect(0.U)
      dut.io.ctx.staleCsrRejected.expect(0.U)
      dut.io.ctx.perCtxCancelled.foreach(_.expect(false.B))

      // --- architectural results: per-Context GPR banks ---
      def checkRf(ctx: Int, expect: Map[Int, Int]): Unit =
        expect.toSeq.sortBy(_._1).foreach { case (reg, v) =>
          dut.io.ctx.rfDebugAddr.poke(reg.U)
          dut.io.ctx.perCtxRfData(ctx).expect(v.U)
        }
      checkRf(0, expected0)
      checkRf(1, expected1)

      // --- DTCM round trip through the real LSU (rows are lsuDataBits/8 bytes) ---
      val dtcmRow = BigInt(0x10000) >> log2Ceil(p.lsuDataBits / 8)
      assert(bus.dtcmWord(dtcmRow, 0) == 35, s"ctx0 store missing: ${bus.dtcmWord(dtcmRow, 0)}")
      assert(bus.dtcmWord(dtcmRow, 1) == 143, s"ctx1 store missing: ${bus.dtcmWord(dtcmRow, 1)}")

      // --- WFI parks both Contexts; backends drain; Tile quiesces ---
      var quiet = false
      var wq    = 0
      while (!quiet && wq < 200) {
        bus.poke()
        if (dut.io.ctx.perCtxRobEmpty(0).peek().litToBoolean &&
            dut.io.ctx.perCtxRobEmpty(1).peek().litToBoolean &&
            dut.io.ctx.perCtxBackendDrained(0).peek().litToBoolean &&
            dut.io.ctx.perCtxBackendDrained(1).peek().litToBoolean) quiet = true
        dut.clock.step()
        wq += 1
      }
      assert(quiet, "backends never drained after WFI")
      dut.io.ctx.tileQuiescent.expect(true.B)
    }
  }

  "A parked context is not quiescent while an accepted fetch still drains" in {
    val p = makeParams
    simulate(new SCore(p)) { dut =>
      dut.io.irq.poke(false.B)
      dut.io.timer_irq.poke(false.B)
      dut.io.software_irq.poke(false.B)
      dut.io.dm.debug_req.poke(false.B)
      dut.io.dm.resume_req.poke(false.B)
      dut.io.dm.csr.valid.poke(false.B)
      dut.io.dm.csr.bits.addr.poke(0.U)
      dut.io.dm.csr.bits.index.poke(0.U)
      dut.io.dm.csr.bits.rs1.poke(0.U)
      dut.io.dm.csr.bits.op.poke(CsrOp.CSRRW)
      dut.io.dm.csr_rs1.poke(0.U)
      dut.io.dm.scalar_rd.valid.poke(false.B)
      dut.io.dm.scalar_rd.bits.addr.poke(0.U)
      dut.io.dm.scalar_rd.bits.data.poke(0.U)
      dut.io.dm.scalar_rs.idx.poke(0.U)
      dut.io.csr.in.value.foreach(_.poke(0.U))
      dut.io.ctx.ctxSelect.poke(0.U)
      dut.io.ctx.advanceGen.foreach(_.poke(false.B))
      dut.io.ctx.perCtxPark.foreach(_.poke(false.B))
      dut.io.ctx.parkGen.foreach(_.poke(0.U))
      dut.io.ctx.cancelValid.foreach(_.poke(false.B))
      dut.io.ctx.cancelGen.foreach(_.poke(0.U))
      dut.io.ctx.pendingLaunch.foreach(_.poke(false.B))
      dut.io.ctx.waitBusy.foreach(_.poke(false.B))
      dut.io.ctx.regionBusy.foreach(_.poke(false.B))
      dut.io.ctx.ctxAccepted.foreach(_.poke(false.B))
      dut.io.ctx.programReady.foreach(_.poke(false.B))
      dut.io.ctx.entryStart.foreach(_.valid.poke(false.B))
      dut.io.ctx.entryStart.foreach(_.bits.poke(0.U))
      dut.io.ctx.rfObserve.poke(false.B)
      dut.io.ctx.rfDebugAddr.poke(0.U)

      // Bus backpressure model: the FIRST accepted fetch gets its fixed next-cycle
      // response; afterwards the IBus hard-stalls until the test releases it, so an
      // ACCEPTED transaction (and later a stable unaccepted offer) stays pending.
      // [addi x1,x0,1; wfi; nop; nop]
      val firstLine = BigInt("00000013000000131050007300100093", 16)
      var served    = false
      var rdataNext = BigInt(0)
      def busStep(stall: Boolean): Unit = {
        dut.io.ibus.ready.poke((!stall).B)
        dut.io.ibus.rdata.poke(rdataNext.U)
        dut.io.ibus.fault.valid.poke(false.B)
        dut.io.ibus.fault.bits.write.poke(false.B)
        dut.io.ibus.fault.bits.addr.poke(0.U)
        dut.io.ibus.fault.bits.epc.poke(0.U)
        if (!stall && dut.io.ibus.valid.peek().litToBoolean) {
          rdataNext = firstLine
          served = true
        }
        dut.io.dbus.ready.poke(true.B)
        dut.io.dbus.rdata.poke(0.U)
        dut.io.ebus.dbus.ready.poke(true.B)
        dut.io.ebus.dbus.rdata.poke(0.U)
        dut.io.ebus.fault.valid.poke(false.B)
        dut.io.ebus.fault.bits.write.poke(false.B)
        dut.io.ebus.fault.bits.addr.poke(0.U)
        dut.io.ebus.fault.bits.epc.poke(0.U)
      }

      dut.reset.poke(true.B)
      busStep(stall = false)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      // Launch ctx0 at 0x0, run to its WFI, then park; hold ctx1 parked so the
      // whole-Tile predicate waits only on ctx0's frontend obligations.
      dut.io.ctx.ctxAccepted(0).poke(true.B)
      dut.io.ctx.programReady(0).poke(true.B)
      dut.io.ctx.entryStart(0).valid.poke(true.B)
      dut.io.ctx.entryStart(0).bits.poke(0.U)
      busStep(stall = false)
      dut.clock.step()
      dut.io.ctx.entryStart(0).valid.poke(false.B)
      dut.io.ctx.perCtxPark(1).poke(true.B)

      var cycles = 0
      while (!dut.io.ctx.perCtxParked(0).peek().litToBoolean && cycles < 300) {
        busStep(stall = served) // hard-stall the memory once the first line landed
        dut.clock.step()
        cycles += 1
      }
      assert(cycles < 300, "ctx0 never parked at its WFI")

      // Park != quiescent: an accepted-but-unanswered fetch transaction (credit
      // outstanding) or a stable unaccepted offer keeps perCtxBackendDrained false.
      var sawNotDrained = false
      var quiesced      = false
      cycles = 0
      while (!quiesced && cycles < 200) {
        if (!dut.io.ctx.perCtxBackendDrained(0).peek().litToBoolean) sawNotDrained = true
        // cycles < 20 keeps the IBus stalled so the pending accepted transaction and
        // the stable unaccepted offer stay outstanding; afterwards the backpressure
        // releases and the REAL bridge completes the response (credit settles).
        busStep(stall = cycles < 20)
        dut.clock.step()
        cycles += 1
        if (dut.io.ctx.tileQuiescent.peek().litToBoolean &&
            dut.io.ctx.perCtxBackendDrained(0).peek().litToBoolean) quiesced = true
      }
      assert(sawNotDrained, "parked Context looked drained while its fetch was in flight")
      assert(quiesced, "Tile never quiesced after the fetch response settled")
      dut.io.ctx.tileQuiescent.expect(true.B)
    }
  }

  class BruCsrPhases(p: Parameters) extends Module {
    val csr = Module(new CtxCsr(p))
    val bru = Module(new Bru(p, first = true))
    val io = IO(new Bundle {
      val selected = Input(UInt(p.ctxBits.W))
      val write = Input(Valid(new CsrCmd(p)))
      val writeOwner = Input(new OwnerTag(p))
      val writeData = Input(UInt(p.xlen.W))
      val branch = Input(Valid(new BruCmd(p)))
      val branchOwner = Input(new OwnerTag(p))
      val fault = Input(Valid(new FaultManagerOutput(p)))
      val irq = Input(Bool())
      val debugReq = Input(Bool())
      val taken = Output(Valid(UInt(p.programCounterBits.W)))
      val executionOwner = Output(new OwnerTag(p))
      val executionMode = Output(UInt(CsrMode().getWidth.W))
      val wfi = Output(Vec(p.contextCount, Bool()))
      val halted = Output(Vec(p.contextCount, Bool()))
      val faulted = Output(Vec(p.contextCount, Bool()))
      val mepc = Output(Vec(p.contextCount, UInt(p.programCounterBits.W)))
      val mcause = Output(Vec(p.contextCount, UInt(p.xlen.W)))
    })
    csr.io.activeCtx := io.selected
    csr.io.dmCtx := io.selected
    csr.io.advanceGen.foreach(_ := false.B)
    csr.io.csr.in.value.foreach(_ := 0.U)
    csr.io.req := io.write
    csr.io.reqOwner := io.writeOwner
    csr.io.reqFromDm := true.B
    csr.io.rs1.valid := true.B
    csr.io.rs1.data := io.writeData
    csr.io.perCtxNRetired.foreach(_ := 0.U)
    csr.io.perCtxTrapRetired.foreach(_ := false.B)
    csr.io.retiredDelta := 0.U.asTypeOf(csr.io.retiredDelta)
    csr.io.perCtxRetiredConfig := 0.U.asTypeOf(csr.io.perCtxRetiredConfig)
    csr.io.irq := io.irq
    csr.io.timer_irq := false.B
    csr.io.software_irq := false.B
    csr.io.dm.debug_req := io.debugReq
    csr.io.dm.resume_req := false.B
    csr.io.dm.current_pc := 0.U
    csr.io.dm.next_pc := 0.U
    bru.io.csr.get <> csr.io.bru
    csr.io.bruOwner := bru.io.trapOwner.get
    csr.io.bruRequestOwner := bru.io.requestOwner.get
    csr.io.bruExecOwner := bru.io.rdOwner
    bru.io.req := io.branch
    bru.io.reqOwner := io.branchOwner
    bru.io.fault_manager.get := io.fault
    bru.io.rs1.valid := true.B
    bru.io.rs1.data := 0.U
    bru.io.rs2.valid := true.B
    bru.io.rs2.data := 0.U
    bru.io.target.data := 0.U
    io.taken.valid := bru.io.taken.valid
    io.taken.bits := bru.io.taken.value
    io.executionOwner := bru.io.rdOwner
    io.executionMode := csr.io.bru.out.mode.asUInt
    io.wfi := csr.io.perCtxWfi
    io.halted := csr.io.perCtxHalted
    io.faulted := csr.io.perCtxFault
    io.mepc := csr.io.perCtxMepc
    io.mcause := csr.io.perCtxMcause
  }

  "BRU request execution and fault owners retain their own CSR targets" in {
    simulate(new BruCsrPhases(makeParams)) { d =>
      def owner(port: OwnerTag, ctx: Int, seq: Int = 0): Unit = {
        port.ctx_slot.poke(ctx.U)
        port.ctx_generation.poke(0.U)
        port.owner_op_seq.poke(seq.U)
        port.subop_id.poke(0.U)
      }
      d.io.selected.poke(0.U)
      d.io.debugReq.poke(false.B)
      d.io.write.valid.poke(false.B)
      d.io.write.bits.op.poke(CsrOp.CSRRW)
      d.io.write.bits.index.poke(0.U)
      d.io.write.bits.addr.poke(0.U)
      d.io.write.bits.rs1.poke(1.U)
      d.io.writeData.poke(0.U)
      owner(d.io.writeOwner, 0)
      d.io.branch.valid.poke(false.B)
      d.io.branch.bits.op.poke(BruOp.JAL)
      d.io.branch.bits.pc.poke(0.U)
      d.io.branch.bits.target.poke(0.U)
      d.io.branch.bits.link.poke(0.U)
      d.io.branch.bits.inst.poke(0.U)
      d.io.branch.bits.fwd.poke(false.B)
      owner(d.io.branchOwner, 0)
      d.io.fault.valid.poke(false.B)
      d.io.fault.bits.mepc.poke(0.U)
      d.io.fault.bits.mtval.poke(0.U)
      d.io.fault.bits.mcause.poke(2.U)
      d.io.fault.bits.decode.poke(true.B)
      owner(d.io.fault.bits.owner, 1, 77)
      d.io.irq.poke(false.B)
      d.reset.poke(true.B); d.clock.step(); d.reset.poke(false.B)

      def writeCsr(ctx: Int, index: Int, value: Int): Unit = {
        owner(d.io.writeOwner, ctx)
        d.io.write.bits.index.poke(index.U)
        d.io.writeData.poke(value.U)
        d.io.write.valid.poke(true.B)
        d.clock.step()
        d.io.write.valid.poke(false.B)
        d.clock.step()
      }
      var sequence = 0
      def request(ctx: Int, op: BruOp.Type, pc: Int): Unit = {
        sequence += 1
        owner(d.io.branchOwner, ctx, sequence)
        d.io.branch.bits.op.poke(op)
        d.io.branch.bits.pc.poke(pc.U)
        d.io.branch.bits.target.poke(0x80.U)
        d.io.branch.valid.poke(true.B)
      }
      def idleBranch(): Unit = {
        d.io.branch.valid.poke(false.B)
        d.io.fault.valid.poke(false.B)
        d.clock.step()
      }

      writeCsr(0, 0x305, 0x100)
      writeCsr(1, 0x305, 0x200)
      writeCsr(0, 0x341, 0x300)
      writeCsr(1, 0x341, 0x400)
      // A executes while B's MRET captures B's MEPC; issue selection is noise.
      d.io.selected.poke(0.U)
      request(0, BruOp.JAL, 0x40); d.clock.step()
      request(1, BruOp.MRET, 0x104); d.clock.step()
      d.io.taken.valid.expect(true.B)
      d.io.taken.bits.expect(0x400.U)
      d.io.executionOwner.ctx_slot.expect(1.U)
      idleBranch()
      request(1, BruOp.JAL, 0x108); d.clock.step()
      request(0, BruOp.ECALL, 0x44); d.clock.step()
      d.io.taken.valid.expect(true.B)
      d.io.taken.bits.expect(0x100.U)
      d.io.executionOwner.ctx_slot.expect(0.U)
      idleBranch()

      // A's execution does not retag the newly arriving B fault or its target.
      request(0, BruOp.JAL, 0x48); d.clock.step()
      d.io.branch.valid.poke(false.B)
      d.io.fault.valid.poke(true.B)
      d.io.fault.bits.mepc.poke(0x500.U)
      d.clock.step()
      d.io.fault.valid.poke(false.B)
      d.io.taken.valid.expect(true.B)
      d.io.taken.bits.expect(0x200.U)
      d.io.executionOwner.ctx_slot.expect(1.U)
      idleBranch()

      // A's live interrupt vector and B's incoming MRET target are different
      // reads in the SAME beat; both must survive the subsequent owner switch.
      writeCsr(0, 0x304, 0x800)
      writeCsr(0, 0x300, 8)
      d.io.irq.poke(true.B)
      request(0, BruOp.JAL, 0x60); d.clock.step()
      d.io.selected.poke(1.U)
      request(1, BruOp.MRET, 0x10C)
      d.io.taken.valid.expect(true.B)
      d.io.taken.bits.expect(0x100.U)
      d.io.executionOwner.ctx_slot.expect(0.U)
      d.clock.step()
      d.io.irq.poke(false.B)
      d.io.branch.valid.poke(false.B)
      d.io.taken.valid.expect(true.B)
      d.io.taken.bits.expect(0x500.U)
      d.io.executionOwner.ctx_slot.expect(1.U)

      // Registered A controls and an incoming B fault coexist. Neither group
      // may be retagged to the other's owner by trapOwner's fault priority.
      for ((op, parks, halts, faults) <- Seq(
        (BruOp.WFI, true, false, false),
        (BruOp.MPAUSE, false, true, false),
        (BruOp.EBREAK, false, true, true),
        (BruOp.MRET, false, false, false)
      )) {
        d.io.branch.valid.poke(false.B)
        d.io.fault.valid.poke(false.B)
        d.io.write.valid.poke(false.B)
        d.io.irq.poke(false.B)
        d.io.debugReq.poke(false.B)
        d.reset.poke(true.B); d.clock.step(); d.reset.poke(false.B)

        if (op == BruOp.MRET) {
          // Both banks begin in Debug. Only A's MRET may return to Machine.
          for (ctx <- 0 until 2) {
            d.io.selected.poke(ctx.U)
            request(ctx, BruOp.JAL, 0x30 + 4 * ctx)
            d.io.debugReq.poke(true.B)
            d.clock.step()
            d.io.debugReq.poke(false.B)
            d.io.executionMode.expect(CsrMode.Debug.litValue.U)
            idleBranch()
          }
        }
        request(0, op, 0x60)
        d.clock.step() // A is now the registered execution owner.
        d.io.executionOwner.ctx_slot.expect(0.U)
        d.io.branch.valid.poke(false.B)
        owner(d.io.fault.bits.owner, 1, 77)
        d.io.fault.bits.mepc.poke(0x700.U)
        d.io.fault.bits.mtval.poke(0x34.U)
        d.io.fault.bits.mcause.poke(2.U)
        d.io.fault.valid.poke(true.B)
        d.clock.step()
        d.io.fault.valid.poke(false.B)
        d.io.wfi(0).expect(parks.B)
        d.io.wfi(1).expect(false.B)
        d.io.halted(0).expect(halts.B)
        d.io.halted(1).expect(false.B)
        d.io.faulted(0).expect(faults.B)
        d.io.faulted(1).expect(false.B)
        d.io.mepc(0).expect(0.U)
        d.io.mcause(0).expect(0.U)
        d.io.mepc(1).expect(0x700.U)
        d.io.mcause(1).expect(2.U)
        if (op == BruOp.MRET) {
          d.io.executionMode.expect(CsrMode.Debug.litValue.U) // B unchanged.
          request(0, BruOp.JAL, 0x80)
          d.clock.step()
          d.io.executionMode.expect(CsrMode.Machine.litValue.U) // A committed.
        }
        idleBranch()
      }
    }
  }
}
