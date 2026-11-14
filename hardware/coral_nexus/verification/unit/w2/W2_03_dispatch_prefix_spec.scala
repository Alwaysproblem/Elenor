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
import chisel3.util._
import org.scalatest.freespec.AnyFreeSpec

import scala.collection.mutable
import scala.util.Random

/** C028 / W2-03 issue-slice behavioral spec: REAL `Scheduler` + `PerCtxFetchMux` +
  * `DispatchV2` on the actual dispatch path (`perctx_issue.yaml` bundle rules).
  *
  * Everything issue-related runs on production modules; the harness provides only the
  * physical instruction-memory endpoint (a tagged request/response model with
  * programmable PER-CONTEXT delay and independent completion — B lines return while an
  * A line is held for 1000 cycles) and the unit-level ready jitter. No fixed-ready-only
  * scheduler, no static ready vector.
  *
  * Coverage (W2 plan §7 item 6):
  *   - one bundle per cycle, all lanes from ONE owner, program-order continuous
  *     prefix, exact-once issue (SCH-01/06, T-11);
  *   - partial accept: 2 accepted / 2 held through >=100 cycles of readiness jitter
  *     with bit-stable payload, `owner_op_seq` and PC advancing only for the accepted
  *     prefix (SCH-05);
  *   - branch redirect beat suppresses every suffix lane of the OWNING context while
  *     the peer keeps issuing (FET-05);
  *   - A blocked by region projection / CSR order / operand hazard lets B progress,
  *     with unique stall reasons (WAIT_REGION/WAIT_CSR/WAIT_OPERAND/WAIT_FETCH)
  *     counted per Context;
  *   - RR service is bounded: services stay within N accepted grants of each other
  *     (SCH-02/SCH-03, N = context_count from the sealed C019 cs1_c2 target);
  *   - a 1000-cycle instruction-memory miss on A never prevents B service (FET-03).
  *
  * The non-prefix negative mutant for this card is `DispatchV2.lastReady(i+1) := true.B`
  * (Decode.scala try-dispatch loop): it pops suffix lanes without unit acceptance and
  * must turn the prefix/hold assertions below RED before the green run.
  */
class W2_03_DispatchPrefixSpec extends AnyFreeSpec with ChiselSim {
  import W2_03Encoding._

  private val ctxCount = 2
  require(
    C028TargetParams.contextCounts.contains(ctxCount),
    "sealed C019 targets must contain the cs1_c2 context_count used here"
  )
  require(C028TargetParams.fetchOutstandingPerContext > 0)

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


  class Txn(p: Parameters) extends Bundle {
    val key  = new CtxFetchKey(p)
    val addr = UInt(p.fetchAddrBits.W)
    val cnt  = UInt(12.W)
    val live = Bool()
  }

  /** Tagged instruction-memory endpoint: independently completing transactions
    * (one slot per Context at its sealed C019 in-flight bound).
    * The response delay is selected by the request key's Context slot, so one
    * Context's 1000-cycle miss never delays another Context's line.
    */
  class FetchEndpoint(p: Parameters, image: Seq[(BigInt, BigInt)]) extends Module {
    // One slot per Context at its sealed C019 in-flight bound.
    private val nTxn = p.contextCount * C028TargetParams.fetchOutstandingPerContext
    val io = IO(new Bundle {
      val req   = Flipped(Decoupled(new CtxFetchRequest(p)))
      val resp  = Decoupled(new CtxFetchResponse(p))
      val ready = Input(Bool())
      val delay = Input(Vec(p.contextCount, UInt(12.W)))
    })

    val txns = RegInit(VecInit.fill(nTxn)(0.U.asTypeOf(new Txn(p))))
    val freeMask = ~VecInit(txns.map(_.live)).asUInt
    val allocIdx = PriorityEncoder(freeMask)
    val canAlloc = freeMask =/= 0.U && io.ready

    io.req.ready := canAlloc
    when(io.req.fire) {
      val slot = io.req.bits.key.ctx_slot
      val dly = MuxLookup(slot, 0.U(12.W))(
        (0 until p.contextCount).map(c => c.U -> io.delay(c)))
      txns(allocIdx).live := true.B
      txns(allocIdx).key  := io.req.bits.key
      txns(allocIdx).addr := io.req.bits.addr
      txns(allocIdx).cnt  := Mux(slot < p.contextCount.U, dly, 0.U)
    }
    for (i <- 0 until nTxn) {
      when(txns(i).live && txns(i).cnt =/= 0.U) {
        txns(i).cnt := txns(i).cnt - 1.U
      }
    }

    val imageRows = image.map { case (a, d) => (a.U(p.fetchAddrBits.W), d.U(p.fetchDataBits.W)) }
    def lookup(addr: UInt): UInt =
      MuxCase(0.U, imageRows.map { case (a, d) => (addr === a) -> d })

    val readyMask = VecInit(txns.map(t => t.live && t.cnt === 0.U)).asUInt
    val selIdx    = PriorityEncoder(readyMask)
    val respQ     = Module(new Queue(new CtxFetchResponse(p), 2))
    respQ.io.enq.valid      := readyMask.orR
    respQ.io.enq.bits.key   := txns(selIdx).key
    respQ.io.enq.bits.data  := lookup(txns(selIdx).addr)
    respQ.io.enq.bits.fault := false.B
    when(respQ.io.enq.fire) {
      txns(selIdx).live := false.B
    }
    respQ.io.deq <> io.resp
  }

  /** Harness: the REAL frontend, scheduler and dispatch path. */
  class Harness(p: Parameters, image: Seq[(BigInt, BigInt)]) extends Module {
    val fetch   = Module(new PerCtxFetchMux(p, C028TargetParams.fetchOutstandingPerContext))
    val sched   = Module(new Scheduler(p))
    val dispatch = Module(new DispatchV2(p))
    val mem     = Module(new FetchEndpoint(p, image))

    val io = IO(new Bundle {
      val accepted      = Input(Vec(p.contextCount, Bool()))
      val programReady  = Input(Vec(p.contextCount, Bool()))
      val entryStart    = Input(Vec(p.contextCount, Valid(UInt(p.fetchAddrBits.W))))
      val cancel        = Input(Vec(p.contextCount, Bool()))
      val park          = Input(Vec(p.contextCount, Bool()))
      val redirect      = Input(Vec(p.contextCount, Valid(UInt(p.fetchAddrBits.W))))
      val regionBusy    = Input(Vec(p.contextCount, Bool()))
      val sbRegd        = Input(Vec(p.contextCount, UInt(32.W)))
      val sbComb        = Input(Vec(p.contextCount, UInt(32.W)))
      val robSpace      = Input(Vec(p.contextCount, UInt(log2Ceil(p.retirementBufferSize + 1).W)))
      val robEmpty      = Input(Vec(p.contextCount, Bool()))
      val trapPending   = Input(Vec(p.contextCount, Bool()))
      val flagsInFlight = Input(Vec(p.contextCount, Bool()))
      val mluReady      = Input(Bool())
      val dvuReady      = Input(Bool())
      val lsuReady      = Input(Bool())
      val lsuQCap       = Input(UInt(3.W))
      val memReady      = Input(Bool())
      val memDelay      = Input(Vec(p.contextCount, UInt(12.W)))

      val selValid      = Output(Bool())
      val selIdx        = Output(UInt(p.ctxBits.W))
      val service       = Output(Vec(p.contextCount, UInt(16.W)))
      val acceptedTotal = Output(UInt(16.W))
      val opSeq         = Output(Vec(p.contextCount, UInt(p.ownerOpSeqBits.W)))
      val laneSeq       = Output(Vec(p.instructionLanes, UInt(p.ownerOpSeqBits.W)))
      val laneOwnerSlot = Output(Vec(p.instructionLanes, UInt(p.ctxBits.W)))
      val laneFire      = Output(Vec(p.instructionLanes, Bool()))
      val laneValid     = Output(Vec(p.instructionLanes, Bool()))
      val laneAddr      = Output(Vec(p.instructionLanes, UInt(p.programCounterBits.W)))
      val laneInst      = Output(Vec(p.instructionLanes, UInt(p.instructionBits.W)))
      val fetchPc       = Output(Vec(p.contextCount, UInt(p.fetchAddrBits.W)))
      val fetchEpoch    = Output(Vec(p.contextCount, UInt(p.ownerOpSeqBits.W)))
      val fetchLaneValid = Output(Vec(p.contextCount, Vec(p.instructionLanes, Bool())))
      val fetchLaneAddr  = Output(Vec(p.contextCount, Vec(p.instructionLanes,
        UInt(p.programCounterBits.W))))
      val fetchLaneInst  = Output(Vec(p.contextCount, Vec(p.instructionLanes,
        UInt(p.instructionBits.W))))
      val waitReason    = Output(Vec(p.contextCount, UInt(3.W)))
      val stallFetch    = Output(Vec(p.contextCount, UInt(16.W)))
      val stallRegion   = Output(Vec(p.contextCount, UInt(16.W)))
      val stallCsr      = Output(Vec(p.contextCount, UInt(16.W)))
      val stallOperand  = Output(Vec(p.contextCount, UInt(16.W)))
      val csrValid      = Output(Bool())
    })

    // --- frontend lifecycle ---
    fetch.io.accepted     := io.accepted
    fetch.io.programReady := io.programReady
    fetch.io.generation   := VecInit(Seq.fill(p.contextCount)(0.U(p.ctxGenerationBits.W)))
    fetch.io.parked       := io.park
    fetch.io.cancel       := io.cancel
    fetch.io.start        := io.entryStart
    fetch.io.redirect     := io.redirect
    fetch.io.request      <> mem.io.req
    fetch.io.response     <> mem.io.resp
    mem.io.ready := io.memReady
    mem.io.delay := io.memDelay

    // --- scheduler ---
    sched.io.fetch <> fetch.io.inst
    sched.io.accepted        := io.accepted
    sched.io.programReady    := io.programReady
    sched.io.advanceGen      := VecInit(Seq.fill(p.contextCount)(false.B))
    sched.io.parked          := io.park
    sched.io.cancelled       := io.cancel
    sched.io.halted          := VecInit(Seq.fill(p.contextCount)(false.B))
    sched.io.tileIssueFreeze := false.B
    sched.io.waitBusy        := VecInit(Seq.fill(p.contextCount)(false.B))
    sched.io.regionBusy      := io.regionBusy
    sched.io.perCtxGen       := VecInit(Seq.fill(p.contextCount)(0.U(p.ctxGenerationBits.W)))
    sched.io.scoreboardRegd  := io.sbRegd
    sched.io.scoreboardComb  := io.sbComb
    sched.io.robSpace        := io.robSpace
    sched.io.robEmpty        := io.robEmpty
    sched.io.robTrapPending  := io.trapPending
    sched.io.flagsInFlight   := io.flagsInFlight
    sched.io.configInFlight  := VecInit(Seq.fill(p.contextCount)(false.B))
    sched.io.frmPending      := VecInit(Seq.fill(p.contextCount)(false.B))
    sched.io.rvvOrderedPending := VecInit(Seq.fill(p.contextCount)(false.B))
    sched.io.lsuActive       := false.B
    sched.io.lsuQueueCapacity := io.lsuQCap
    sched.io.lsuReqReady     := io.lsuReady
    sched.io.mluReqReady     := io.mluReady
    sched.io.dvuReqReady     := io.dvuReady
    sched.io.perCtxFrm       := VecInit(Seq.fill(p.contextCount)(0.U(3.W)))

    // --- dispatch (selected-Context projections exactly as SCore wires them) ---
    dispatch.io.inst <> sched.io.instOut
    dispatch.io.halted := false.B
    dispatch.io.lsuActive := false.B
    dispatch.io.scoreboard.regd := Mux1H(sched.io.selOH, io.sbRegd)
    dispatch.io.scoreboard.comb := Mux1H(sched.io.selOH, io.sbComb)
    dispatch.io.branchTaken := false.B
    dispatch.io.interlock := false.B
    dispatch.io.retirement_buffer_nSpace := Mux1H(sched.io.selOH, io.robSpace)
    dispatch.io.retirement_buffer_empty := Mux1H(sched.io.selOH, io.robEmpty)
    dispatch.io.retirement_buffer_trap_pending := Mux1H(sched.io.selOH, io.trapPending)
    dispatch.io.csrFlagsPending := Mux1H(sched.io.selOH, io.flagsInFlight)
    dispatch.io.csrConfigPending := false.B
    dispatch.io.csrFrmPending := false.B
    dispatch.io.rvvOrderedPending := false.B
    dispatch.io.single_step := false.B
    dispatch.io.debug_mode := false.B
    dispatch.io.lsuQueueCapacity := io.lsuQCap
    for (i <- 0 until p.instructionLanes) {
      dispatch.io.jalrTarget(i).data := 0.U
      dispatch.io.mlu(i).ready := io.mluReady
      dispatch.io.dvu(i).ready := io.dvuReady
      dispatch.io.lsu(i).ready := io.lsuReady
    }

    // --- observation ---
    io.selValid      := sched.io.selValid
    io.selIdx        := sched.io.selIdx
    io.service       := sched.io.serviceCount
    io.acceptedTotal := sched.io.acceptedTotal
    io.opSeq         := sched.io.perCtxOpSeq
    io.laneSeq       := sched.io.laneOpSeq
    io.laneOwnerSlot := sched.io.laneOwners.map(_.ctx_slot)
    io.laneFire      := sched.io.instOut.map(_.fire)
    io.laneValid     := sched.io.instOut.map(_.valid)
    io.laneAddr      := sched.io.instOut.map(_.bits.addr)
    io.laneInst      := sched.io.instOut.map(_.bits.inst)
    io.fetchPc       := fetch.io.pc
    io.fetchEpoch    := fetch.io.epoch
    io.fetchLaneValid := fetch.io.inst.map(bank => VecInit(bank.map(_.valid)))
    io.fetchLaneAddr  := fetch.io.inst.map(bank => VecInit(bank.map(_.bits.addr)))
    io.fetchLaneInst  := fetch.io.inst.map(bank => VecInit(bank.map(_.bits.inst)))
    io.waitReason    := sched.io.waitReason
    io.stallFetch    := sched.io.stallCount.map(_.fetch)
    io.stallRegion   := sched.io.stallCount.map(_.region)
    io.stallCsr      := sched.io.stallCount.map(_.csr)
    io.stallOperand  := sched.io.stallCount.map(_.operand)
    io.csrValid      := dispatch.io.csr.valid
  }

  def resetHarness(h: Harness): Unit = {
    h.io.accepted.foreach(_.poke(false.B))
    h.io.programReady.foreach(_.poke(false.B))
    h.io.entryStart.foreach(e => e.valid.poke(false.B))
    h.io.entryStart.foreach(_.bits.poke(0.U))
    h.io.cancel.foreach(_.poke(false.B))
    h.io.park.foreach(_.poke(false.B))
    h.io.redirect.foreach(r => r.valid.poke(false.B))
    h.io.redirect.foreach(_.bits.poke(0.U))
    h.io.regionBusy.foreach(_.poke(false.B))
    h.io.sbRegd.foreach(_.poke(0.U))
    h.io.sbComb.foreach(_.poke(0.U))
    h.io.robSpace.foreach(_.poke(16.U))
    h.io.robEmpty.foreach(_.poke(true.B))
    h.io.trapPending.foreach(_.poke(false.B))
    h.io.flagsInFlight.foreach(_.poke(false.B))
    h.io.mluReady.poke(true.B)
    h.io.dvuReady.poke(true.B)
    h.io.lsuReady.poke(true.B)
    h.io.lsuQCap.poke(7.U)
    h.io.memReady.poke(true.B)
    h.io.memDelay.foreach(_.poke(0.U))
    h.reset.poke(true.B)
    h.clock.step()
    h.reset.poke(false.B)
    h.clock.step()
  }

  def launch(h: Harness, ctx: Int, entry: BigInt): Unit = {
    h.io.accepted(ctx).poke(true.B)
    h.io.programReady(ctx).poke(true.B)
    h.io.entryStart(ctx).valid.poke(true.B)
    h.io.entryStart(ctx).bits.poke(entry.U)
    h.clock.step()
    h.io.entryStart(ctx).valid.poke(false.B)
  }

  def waitHeads(h: Harness, maxCycles: Int = 200): Unit = {
    var w = 0
    while ((!h.io.fetchLaneValid(0)(0).peek().litToBoolean ||
            !h.io.fetchLaneValid(1)(0).peek().litToBoolean) && w < maxCycles) {
      h.clock.step()
      w += 1
    }
    assert(h.io.fetchLaneValid(0)(0).peek().litToBoolean, "ctx0 head never arrived")
    assert(h.io.fetchLaneValid(1)(0).peek().litToBoolean, "ctx1 head never arrived")
  }

  def firedCount(h: Harness): Int =
    (0 until h.io.laneFire.length).count(i => h.io.laneFire(i).peek().litToBoolean)

  "One owner issues a four lane continuous prefix exactly once with pc advance" in {
    val p = makeParams
    val base0 = BigInt(0x1000)
    val image = Seq(
      base0 -> line(addi(5, 0, 1), addi(6, 0, 2), addi(7, 0, 3), addi(8, 0, 4)),
      base0 + 16 -> line(addi(9, 0, 5), addi(10, 0, 6), addi(11, 0, 7), addi(12, 0, 8)),
      base0 + 32 -> line(NOP, NOP, NOP, NOP)
    )
    simulate(new Harness(p, image)) { h =>
      resetHarness(h)
      // Only Context 0 launches; Context 1 must never be selected.
      launch(h, 0, base0)
      val firedAddrs = mutable.ArrayBuffer[BigInt]()
      var beats      = 0
      while (firedAddrs.size < 8 && beats < 300) {
        val n = firedCount(h)
        if (n > 0) {
          assert(h.io.selValid.peek().litToBoolean, "lanes fired without a grant")
          assert(n == 4, s"expected the full 4-lane prefix, got $n")
          assert(h.io.selIdx.peek().litValue == 0, "only Context 0 is launchable")
          val base = firedAddrs.size
          val seqs = (0 until 4).map(i => h.io.laneSeq(i).peek().litValue)
          assert(seqs == (base until base + 4).map(BigInt(_)), s"owner_op_seq mismatch: $seqs")
          (0 until 4).foreach { i =>
            assert(h.io.laneOwnerSlot(i).peek().litValue == 0, "lane owner must be ctx0")
            firedAddrs += h.io.laneAddr(i).peek().litValue
          }
        }
        h.clock.step()
        beats += 1
      }
      assert(firedAddrs.size == 8, s"stream did not complete: ${firedAddrs.size}")
      assert(firedAddrs.distinct == firedAddrs, "an instruction issued more than once")
      assert(firedAddrs == firedAddrs.sorted, "instructions issued out of program order")
      assert(h.io.opSeq(0).peek().litValue == 8, "owner_op_seq must count accepted lanes")
      assert(h.io.service(1).peek().litValue == 0, "non-launched Context must never be served")
    }
  }

  "Partial accept holds suffix payload stable across long unit ready jitter" in {
    val p = makeParams
    val base0 = BigInt(0x2000)
    // Lanes 0/1 are ALU (accepted); lane 2 is a MUL the unit will not accept; lane 3
    // is a YOUNGER INDEPENDENT ADDI that must never accept past the blocked MUL
    // (prefix_only) — the non-prefix mutant would let exactly this ADDI fire.
    val image = Seq(
      base0 -> line(addi(5, 0, 1), addi(6, 0, 2), mul(7, 5, 6), addi(8, 0, 7)),
      base0 + 16 -> line(addi(9, 0, 9), NOP, NOP, NOP)
    )
    simulate(new Harness(p, image)) { h =>
      resetHarness(h)
      launch(h, 0, base0)
      h.io.mluReady.poke(false.B)
      // Wait for the partial-accept beat: lanes 0/1 (ALU) fire while the MUL head
      // cannot accept. Capture the held identities BEFORE the accepting edge.
      var partialBeat = -1
      var beats       = 0
      while (partialBeat < 0 && beats < 300) {
        if (firedCount(h) == 2) partialBeat = beats
        if (partialBeat < 0) { h.clock.step(); beats += 1 }
      }
      assert(partialBeat >= 0, "partial accept never happened")
      assert(h.io.fetchLaneAddr(0)(2).peek().litValue == base0 + 8, "held head must be the MUL")
      val mulAddr  = h.io.fetchLaneAddr(0)(2).peek().litValue
      val mulInst  = h.io.fetchLaneInst(0)(2).peek().litValue
      val addiAddr = h.io.fetchLaneAddr(0)(3).peek().litValue
      val addiInst = h.io.fetchLaneInst(0)(3).peek().litValue
      assert(addiAddr == base0 + 12, "held suffix must be the younger ADDI")
      h.clock.step() // the accepting edge: op_seq / PC advance only for the prefix
      assert(h.io.opSeq(0).peek().litValue == 2, "owner_op_seq advanced past accepted prefix")
      assert(h.io.fetchPc(0).peek().litValue == base0 + 8, "next unaccepted PC must be the held MUL")

      // >=100 cycles of readiness jitter on the other units and the memory endpoint
      // while the MUL stays unaccepted: the real buffer has COMPACTED the two held
      // instructions to lanes 0/1; their identities must stay bit-stable and nothing
      // — in particular the younger ADDI — may accept past the blocked head.
      val rng = new Random(20261114)
      var heldCycles = 0
      while (heldCycles < 120) {
        h.io.dvuReady.poke(rng.nextBoolean().B)
        h.io.lsuReady.poke(rng.nextBoolean().B)
        h.io.memReady.poke(rng.nextBoolean().B)
        assert(firedCount(h) == 0, "held suffix must not issue (prefix rule)")
        assert(h.io.fetchLaneValid(0)(0).peek().litToBoolean, "held MUL lost its valid")
        assert(h.io.fetchLaneValid(0)(1).peek().litToBoolean, "held ADDI lost its valid")
        assert(h.io.fetchLaneAddr(0)(0).peek().litValue == mulAddr, "MUL payload deformed")
        assert(h.io.fetchLaneInst(0)(0).peek().litValue == mulInst, "MUL inst deformed")
        assert(h.io.fetchLaneAddr(0)(1).peek().litValue == addiAddr, "ADDI payload deformed")
        assert(h.io.fetchLaneInst(0)(1).peek().litValue == addiInst, "ADDI inst deformed")
        assert(h.io.fetchPc(0).peek().litValue == mulAddr, "PC advanced past unaccepted lanes")
        assert(h.io.opSeq(0).peek().litValue == 2, "owner_op_seq moved while held")
        h.clock.step()
        heldCycles += 1
      }
      // Release: the held prefix (MUL, then the younger ADDI) accepts with the exact
      // original identities and continues the op_seq at 2, 3.
      h.io.dvuReady.poke(true.B)
      h.io.lsuReady.poke(true.B)
      h.io.memReady.poke(true.B)
      h.io.mluReady.poke(true.B)
      var resumedAt = -1
      beats = 0
      while (resumedAt < 0 && beats < 60) {
        if (firedCount(h) >= 1 && h.io.laneAddr(0).peek().litValue == mulAddr) {
          assert(h.io.laneInst(0).peek().litValue == mulInst, "MUL issued deformed inst")
          assert(h.io.laneSeq(0).peek().litValue == 2, "MUL must carry owner_op_seq 2")
          resumedAt = beats
        }
        if (resumedAt < 0) { h.clock.step(); beats += 1 }
      }
      assert(resumedAt >= 0, "held MUL never issued after release")
      // The younger ADDI accepts after (same beat or a later one) with identity+seq 3.
      var addiDone = false
      beats = 0
      while (!addiDone && beats < 40) {
        (0 until 4).foreach { i =>
          if (h.io.laneFire(i).peek().litToBoolean &&
              h.io.laneAddr(i).peek().litValue == addiAddr) {
            assert(h.io.laneInst(i).peek().litValue == addiInst, "ADDI issued deformed inst")
            assert(h.io.laneSeq(i).peek().litValue == 3, "ADDI must carry owner_op_seq 3")
            addiDone = true
          }
        }
        h.clock.step()
        beats += 1
      }
      assert(addiDone, "younger ADDI never issued after the MUL")
      assert(h.io.opSeq(0).peek().litValue >= 4, "owner_op_seq did not advance after accept")
    }
  }

  "Branch redirect beat suppresses owner suffix lanes while the peer issues" in {
    val p = makeParams
    val base0 = BigInt(0x3000)
    val base1 = BigInt(0x3800)
    val image = Seq(
      base0 -> line(bne(1, 2, 8), addi(5, 0, 1), addi(6, 0, 2), addi(7, 0, 3))
    ) ++ nopLines(base0 + 16, 6) ++ Seq(
      base1 -> line(addi(15, 0, 1), addi(16, 0, 2), addi(17, 0, 3), addi(18, 0, 4))
    ) ++ nopLines(base1 + 16, 30)
    simulate(new Harness(p, image)) { h =>
      resetHarness(h)
      launch(h, 0, base0)
      launch(h, 1, base1)
      // Wait for ctx0's BNE head: lane 0 fires (BRU is a Valid unit) and the
      // same-bundle suffix lanes must not issue (jumped/branch scan).
      var branchBeat = -1
      var beats      = 0
      while (branchBeat < 0 && beats < 300) {
        if (h.io.selValid.peek().litToBoolean && h.io.selIdx.peek().litValue == 0 &&
            firedCount(h) >= 1 && h.io.laneAddr(0).peek().litValue == base0) {
          assert(firedCount(h) == 1, "suffix lanes after a branch must not issue")
          branchBeat = beats
        }
        if (branchBeat < 0) { h.clock.step(); beats += 1 }
      }
      assert(branchBeat >= 0, "ctx0 branch never issued")
      // Emulate Bru resolution one beat later: redirect ctx0 to pc+8.
      h.clock.step()
      val epochBefore    = h.io.fetchEpoch(0).peek().litValue
      val peerEpochBefore = h.io.fetchEpoch(1).peek().litValue
      h.io.redirect(0).valid.poke(true.B)
      h.io.redirect(0).bits.poke((base0 + 8).U)
      assert(
        !h.io.selValid.peek().litToBoolean || h.io.selIdx.peek().litValue != 0 ||
          !h.io.fetchLaneValid(0)(0).peek().litToBoolean,
        "redirected Context's lanes must be suppressed in the redirect beat"
      )
      h.clock.step()
      h.io.redirect(0).valid.poke(false.B)
      assert(h.io.fetchEpoch(0).peek().litValue != epochBefore, "epoch must advance on redirect")
      assert(h.io.fetchEpoch(1).peek().litValue == peerEpochBefore, "peer epoch must not move")
      // Park ctx0 once its post-redirect stream settled; scan for wrong-path suffix.
      var settled = false
      var w       = 0
      while (!settled && w < 60) {
        if (h.io.opSeq(0).peek().litValue >= 4) settled = true
        else { h.clock.step(); w += 1 }
      }
      assert(settled, "ctx0 never resumed at the branch target")
      h.io.park(0).poke(true.B)
      var c0 = 0
      val suffixFired = mutable.ArrayBuffer[BigInt]()
      while (c0 < 20) {
        (0 until 4).foreach(i =>
          if (h.io.laneFire(i).peek().litToBoolean && h.io.selIdx.peek().litValue == 0)
            suffixFired += h.io.laneAddr(i).peek().litValue
        )
        h.clock.step()
        c0 += 1
      }
      val badSuffix = suffixFired.filter(a => a > base0 && a < base0 + 8)
      assert(badSuffix.isEmpty, s"wrong-path suffix issued: $badSuffix")
      // The peer kept issuing through ctx0's redirect.
      var peerProgress = false
      var c1           = 0
      while (c1 < 100 && !peerProgress) {
        if (h.io.selValid.peek().litToBoolean && h.io.selIdx.peek().litValue == 1 &&
            firedCount(h) > 0) peerProgress = true
        h.clock.step()
        c1 += 1
      }
      assert(peerProgress, "peer made no progress across the redirect")
    }
  }

  "Region blocked A lets B issue continuously and RR resumes at A plus one" in {
    val p = makeParams
    val base0 = BigInt(0x4000)
    val base1 = BigInt(0x4800)
    val image = Seq(
      base0 -> line(addi(5, 0, 1), addi(6, 0, 1), addi(7, 0, 1), addi(8, 0, 1)),
      base0 + 16 -> line(addi(9, 0, 1), NOP, NOP, NOP)
    ) ++ Seq(
      base1 -> line(addi(15, 0, 2), addi(16, 0, 2), addi(17, 0, 2), addi(18, 0, 2))
    ) ++ nopLines(base1 + 16, 60)
    simulate(new Harness(p, image)) { h =>
      resetHarness(h)
      launch(h, 0, base0)
      launch(h, 1, base1)
      waitHeads(h)
      // C012 region projection parks A at its head for a long window.
      h.io.regionBusy(0).poke(true.B)
      val svc0AtBlock  = h.io.service(0).peek().litValue
      val svc1AtBlock  = h.io.service(1).peek().litValue
      val region0Start = h.io.stallRegion(0).peek().litValue
      assert(h.io.waitReason(0).peek().litValue == 2, "expected WAIT_REGION") // NONE,EVENT,REGION
      var blocked = 0
      while (blocked < 150) {
        assert(h.io.waitReason(0).peek().litValue == 2, "A must record WAIT_REGION")
        h.clock.step()
        blocked += 1
      }
      assert(h.io.service(0).peek().litValue == svc0AtBlock, "blocked A was serviced")
      assert(h.io.stallRegion(0).peek().litValue > region0Start, "WAIT_REGION counter did not grow")
      assert(h.io.service(1).peek().litValue > svc1AtBlock + 40, "B starved behind parked A")

      // Release: A is granted within N accepted opportunities and RR continues A,B.
      h.io.regionBusy(0).poke(false.B)
      val grantOrder = mutable.ArrayBuffer[Int]()
      var release    = 0
      while (release < 60 && grantOrder.size < 8) {
        if (h.io.selValid.peek().litToBoolean && firedCount(h) > 0) {
          grantOrder += h.io.selIdx.peek().litValue.toInt
          if (h.io.opSeq(0).peek().litValue >= 8) h.io.park(0).poke(true.B)
        }
        h.clock.step()
        release += 1
      }
      assert(grantOrder.size >= 6, s"RR did not resume: $grantOrder")
      assert(grantOrder.contains(0), "A never resumed after release")
      assert(grantOrder.contains(1), "B stopped after A resumed")
      val firstA = grantOrder.indexOf(0)
      assert(grantOrder.drop(firstA).take(4).contains(1), "scan did not resume at A+1")
    }
  }

  "CSR ordered head waits on flags in flight while the peer issues" in {
    val p = makeParams
    val base0 = BigInt(0x5000)
    val base1 = BigInt(0x5800)
    val image = Seq(
      base0 -> line(csrrs(5, 1, 1), addi(6, 0, 1), NOP, NOP)
    ) ++ nopLines(base0 + 16, 4) ++ Seq(
      base1 -> line(addi(15, 0, 1), addi(16, 0, 2), NOP, NOP)
    ) ++ nopLines(base1 + 16, 30)
    simulate(new Harness(p, image)) { h =>
      resetHarness(h)
      h.io.flagsInFlight(0).poke(true.B)
      launch(h, 0, base0)
      launch(h, 1, base1)
      waitHeads(h)
      val csr0 = h.io.stallCsr(0).peek().litValue
      var blocked = 0
      while (blocked < 60) {
        assert(h.io.waitReason(0).peek().litValue == IssueStallReason.WAIT_CSR.litValue,
          "expected WAIT_CSR")
        h.clock.step()
        blocked += 1
      }
      assert(h.io.stallCsr(0).peek().litValue > csr0, "WAIT_CSR counter did not grow")
      assert(h.io.service(1).peek().litValue > 5, "peer starved behind CSR-ordered head")
      assert(h.io.opSeq(0).peek().litValue == 0, "CSR-ordered head must not issue")
      h.io.flagsInFlight(0).poke(false.B)
      var issued = false
      var r      = 0
      while (!issued && r < 40) {
        if (h.io.selIdx.peek().litValue == 0 && h.io.csrValid.peek().litToBoolean) issued = true
        h.clock.step()
        r += 1
      }
      assert(issued, "CSR head never issued after the ordered wait cleared")
    }
  }

  "Operand hazard on the own scoreboard stalls only the owner" in {
    val p = makeParams
    val base0 = BigInt(0x6000)
    val base1 = BigInt(0x6800)
    val image = Seq(
      base0 -> line(addi(6, 6, 1), NOP, NOP, NOP)
    ) ++ nopLines(base0 + 16, 4) ++ Seq(
      base1 -> line(addi(15, 0, 1), NOP, NOP, NOP)
    ) ++ nopLines(base1 + 16, 30)
    simulate(new Harness(p, image)) { h =>
      resetHarness(h)
      h.io.sbComb(0).poke((1 << 6).U)
      launch(h, 0, base0)
      launch(h, 1, base1)
      waitHeads(h)
      val op0 = h.io.stallOperand(0).peek().litValue
      var blocked = 0
      while (blocked < 60) {
        assert(h.io.waitReason(0).peek().litValue == IssueStallReason.WAIT_OPERAND.litValue,
          "expected WAIT_OPERAND")
        h.clock.step()
        blocked += 1
      }
      assert(h.io.stallOperand(0).peek().litValue > op0, "WAIT_OPERAND counter did not grow")
      assert(h.io.opSeq(0).peek().litValue == 0, "hazardous head must not issue")
      assert(h.io.opSeq(1).peek().litValue > 0, "peer stalled behind another Context's operand")
      h.io.sbComb(0).poke(0.U)
      var issued = false
      var r      = 0
      while (!issued && r < 40) {
        if (h.io.selIdx.peek().litValue == 0 && firedCount(h) > 0) issued = true
        h.clock.step()
        r += 1
      }
      assert(issued, "head never issued after the hazard cleared")
    }
  }

  "Round robin service is bounded and advances only on accepted issue" in {
    val p = makeParams
    val base0 = BigInt(0x7000)
    val base1 = BigInt(0x7800)
    val image = Seq(
      base0 -> line(addi(5, 0, 1), addi(6, 0, 1), addi(7, 0, 1), addi(8, 0, 1)),
      base0 + 16 -> line(addi(9, 0, 1), NOP, NOP, NOP)
    ) ++ Seq(
      base1 -> line(addi(15, 0, 2), addi(16, 0, 2), addi(17, 0, 2), addi(18, 0, 2)),
      base1 + 16 -> line(addi(19, 0, 2), NOP, NOP, NOP)
    )
    simulate(new Harness(p, image)) { h =>
      resetHarness(h)
      launch(h, 0, base0)
      launch(h, 1, base1)
      val accStart  = h.io.acceptedTotal.peek().litValue
      var parked    = Array(false, false)
      var cycles    = 0
      while (cycles < 80) {
        // Park each Context when its stream is exhausted so no phantom head sticks.
        if (!parked(0) && h.io.opSeq(0).peek().litValue >= 8) {
          h.io.park(0).poke(true.B); parked(0) = true
        }
        if (!parked(1) && h.io.opSeq(1).peek().litValue >= 8) {
          h.io.park(1).poke(true.B); parked(1) = true
        }
        h.clock.step()
        cycles += 1
        val s0 = h.io.service(0).peek().litValue
        val s1 = h.io.service(1).peek().litValue
        assert((s0 - s1).abs <= 2, s"unbounded service gap: $s0 vs $s1")
      }
      assert(parked(0) && parked(1), "streams did not complete in the window")
      val acc = h.io.acceptedTotal.peek().litValue - accStart
      assert(acc >= 4, "RR never accepted issues")
      assert(
        h.io.service(0).peek().litValue + h.io.service(1).peek().litValue ==
          h.io.acceptedTotal.peek().litValue,
        "service counts must sum to accepted opportunities"
      )
      assert(
        h.io.opSeq(0).peek().litValue + h.io.opSeq(1).peek().litValue == BigInt(16),
        "both streams must complete exactly once"
      )
    }
  }

  "Long instruction memory miss stalls only the missing context" in {
    val p = makeParams
    val base0 = BigInt(0x8800)
    val base1 = BigInt(0x9000)
    val image = Seq(
      base0 -> line(addi(5, 0, 1), addi(6, 0, 1), addi(7, 0, 1), addi(8, 0, 1))
    ) ++ nopLines(base0 + 16, 2) ++ Seq(
      base1 -> line(addi(15, 0, 2), addi(16, 0, 2), NOP, NOP)
    ) ++ nopLines(base1 + 16, 40)
    simulate(new Harness(p, image)) { h =>
      resetHarness(h)
      launch(h, 0, base0)
      launch(h, 1, base1)
      // Let A issue its first line, then strand it behind a 1000-cycle line miss.
      var w = 0
      while (h.io.opSeq(0).peek().litValue < 4 && w < 200) { h.clock.step(); w += 1 }
      assert(h.io.opSeq(0).peek().litValue == 4, "A never issued its first line")
      h.io.memDelay(0).poke(1000.U)
      h.io.redirect(0).valid.poke(true.B)
      h.io.redirect(0).bits.poke(base0.U)
      h.clock.step()
      h.io.redirect(0).valid.poke(false.B)
      val seq0 = h.io.opSeq(0).peek().litValue
      val svc1 = h.io.service(1).peek().litValue
      var d            = 0
      var sawFetchWait = false
      while (d < 200) {
        if (h.io.waitReason(0).peek().litValue == 3) sawFetchWait = true // FETCH
        h.clock.step()
        d += 1
      }
      assert(sawFetchWait, "A never recorded WAIT_FETCH behind its long miss")
      assert(h.io.opSeq(0).peek().litValue == seq0, "A issued while its fetch was stuck")
      assert(h.io.service(1).peek().litValue > svc1 + 20, "B starved behind A's fetch miss")
      h.io.memDelay(0).poke(0.U)
    }
  }
}
