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
import common.MuxOR

/** (IssueStallReason / SchedulerStallCounters live in the base library next to
  * `CoreContextIO` — see UopTag.scala.) */

/** Real per-Context head predecode, eligible round-robin selection and continuous
  * prefix mux into DispatchV2 (C028 / W2-03, `perctx_issue.yaml` bundle rules).
  *
  * Production module consumed by SCore:
  *   - selection reuses the C027 `ContextIssueArbiter`: the rotating pointer advances
  *     ONLY on an actually-accepted issue (at least one lane fired into a unit), a
  *     blocked head never receives a grant, and after a selection the scan resumes at
  *     that slot + 1 (SCH-02/SCH-03);
  *   - `eligible` requires the Context to be launchable (Task accepted AND program
  *     READY, never inferred from reset or selection) with a fetched head and no hold
  *     (park / cancel / halt / Tile fault freeze / W6 wait / C012 region projection);
  *     `blocked` carries the real head predecode gates: per-Context scoreboard
  *     RAW/WAW, per-Context ROB space and CSR-order projections, shared unit queue
  *     capacity and occupancy probes, plus the C012 region projection at the head
  *     before shared vector ingress;
  *   - shared-unit acceptability participates in `blocked` through SIDE-EFFECT-FREE
  *     occupancy probes only (registered state: the LSU reserve-station space and
  *     flush command, the DVU state machine, the FloatCore queue count, the RVV queue
  *     capacity — the same signal DispatchV2's own interlock trusts — and the
  *     SCore-computed MLU pipeline occupancy), so eligibility can never loop through
  *     Dispatch's request valids, and a Context stalled 1000 cycles on a busy shared
  *     unit is NOT eligible and cannot block the peer (no global HOL, SCH-02/FET-03);
  *   - `issueAccepted` accounting keeps the C026 contract: `owner_op_seq` advances only
  *     for ACCEPTED lanes (the fired continuous prefix), lane owners are
  *     `(ctx_slot, ctx_generation, owner_op_seq)`, and per-Context sequence counters
  *     reset only on that Context's `advanceGen`;
  *   - a live cancel drops ONLY the selected Context's unaccepted payload (frontend
  *     lanes are drained), never RF/ROB state.
  *
  * The bundle issued to DispatchV2 always comes from ONE Context and DispatchV2's own
  * `lastReady` chain enforces the program-order continuous-prefix rule; this module
  * additionally routes the per-Context buffer ready so exactly the accepted prefix of
  * the selected bank pops (partial accept keeps the suffix payload stable in the
  * owner's buffer, SCH-05/SCH-06).
  */
class Scheduler(p: Parameters) extends Module {
  val io = IO(new Bundle {
    // Per-Context fetch banks (PerCtxFetchMux.io.inst): the selected bank's head is
    // the candidate bundle; `ready` pops exactly the accepted prefix.
    val fetch =
      Vec(p.contextCount, Vec(p.instructionLanes, Flipped(Decoupled(new FetchInstruction(p)))))

    // Lifecycle (W1/W5 boundary via CoreContextIO). `accepted && programReady` gates
    // launchability; both are explicit inputs, never inferred.
    val accepted     = Input(Vec(p.contextCount, Bool()))
    val programReady = Input(Vec(p.contextCount, Bool()))
    val advanceGen   = Input(Vec(p.contextCount, Bool())) // safe slot-reuse clear beat
    // Hold reasons that stop NEW issue without being counted as resource stalls.
    val parked  = Input(Vec(p.contextCount, Bool())) // park hold (WFI/region park)
    val cancelled = Input(Vec(p.contextCount, Bool())) // live cancel: drop payload
    val halted  = Input(Vec(p.contextCount, Bool())) // CtxCsr perCtxHalted
    val tileIssueFreeze = Input(Bool()) // fail-stop fault freeze (whole Tile)

    // W6/W3 read-only projections (recorded as WAIT_EVENT / WAIT_REGION).
    val waitBusy   = Input(Vec(p.contextCount, Bool()))
    val regionBusy = Input(Vec(p.contextCount, Bool())) // C012 frozen projection

    // Identity.
    val perCtxGen = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))

    // Per-Context scoreboards (BankedRf / BankedFRegfile).
    val scoreboardRegd = Input(Vec(p.contextCount, UInt(p.scalarRegCount.W)))
    val scoreboardComb = Input(Vec(p.contextCount, UInt(p.scalarRegCount.W)))
    val fscoreboard =
      Option.when(p.enableFloat)(Input(Vec(p.contextCount, UInt(p.floatRegCount.W))))

    // Per-Context ROB / CSR-order gates (RetirementPartitionBank / CtxCsr).
    val robSpace = Input(Vec(p.contextCount, UInt(log2Ceil(p.retirementBufferSize + 1).W)))
    val robEmpty = Input(Vec(p.contextCount, Bool()))
    val robTrapPending = Input(Vec(p.contextCount, Bool()))
    val flagsInFlight = Input(Vec(p.contextCount, Bool()))
    val configInFlight = Input(Vec(p.contextCount, Bool()))
    val frmPending = Input(Vec(p.contextCount, Bool()))
    val rvvOrderedPending = Input(Vec(p.contextCount, Bool()))

    // Shared unit probes (queue capacities + owning unit live readiness).
    val lsuActive = Input(Bool())
    val lsuQueueCapacity = Input(UInt(3.W))
    val lsuReqReady = Input(Bool()) // lane-0 LSU request port readiness
    val mluReqReady = Input(Bool())
    val dvuReqReady = Input(Bool())
    val floatReqReady = Option.when(p.enableFloat)(Input(Bool()))
    val rvvQueueCapacity = Option.when(p.enableRvv)(Input(UInt(4.W)))

    // Predecode rounding mode (CtxCsr perCtxFrm).
    val perCtxFrm = Input(Vec(p.contextCount, UInt(3.W)))

    // Selected-Context bundle into DispatchV2 (single owner per beat).
    val instOut = Vec(p.instructionLanes, Decoupled(new FetchInstruction(p)))

    // Selection and issue identity.
    val selValid = Output(Bool())
    val selIdx   = Output(UInt(p.ctxBits.W))
    val selOH    = Output(Vec(p.contextCount, Bool()))
    val laneOpSeq  = Output(Vec(p.instructionLanes, UInt(p.ownerOpSeqBits.W)))
    val laneOwners = Output(Vec(p.instructionLanes, new OwnerTag(p)))
    val perCtxOpSeq = Output(Vec(p.contextCount, UInt(p.ownerOpSeqBits.W)))
    val issueAccepted = Output(Bool()) // the granted bundle actually issued >= 1 lane
    val perCtxHeadValid = Output(Vec(p.contextCount, Bool()))
    val perCtxAnyValid  = Output(Vec(p.contextCount, Bool()))

    // Unique stall accounting (raw reason for PMU + ContextWait mapping for the
    // ContextManager). The UInt form avoids peeking enum ports in ChiselSim.
    val waitReason = Output(
      Vec(p.contextCount, UInt(IssueStallReason().getWidth.W))
    )
    val waitKind   = Output(Vec(p.contextCount, ContextWait()))
    val stallCount = Output(Vec(p.contextCount, new SchedulerStallCounters))

    // Bounded-RR observability (ContextIssueArbiter audit ports).
    val rrPtr         = Output(UInt(p.ctxBits.W))
    val serviceCount  = Output(Vec(p.contextCount, UInt(16.W)))
    val acceptedTotal = Output(UInt(16.W))
  })

  // ---------------------------------------------------------------------------
  // Head predecode (real DecodeInstruction/DecodedInstruction reuse).
  // Only lane 0 gates eligibility: a bundle is a program-order prefix, so the head
  // decides whether the Context can make progress this beat. Decode garbage when the
  // head is empty is harmless (undef, no resource claims).
  val heads = (0 until p.contextCount).map { c =>
    val lane = io.fetch(c)(0)
    val inst = Mux(lane.valid, lane.bits.inst, 0.U(p.instructionBits.W))
    DecodeInstruction(p, 0, lane.bits.addr, inst, io.perCtxFrm(c))
  }
  val headValid = (0 until p.contextCount).map(c => io.fetch(c)(0).valid)
  val anyValid = (0 until p.contextCount).map(c => io.fetch(c).map(_.valid).reduce(_ || _))
  io.perCtxHeadValid := VecInit(headValid)
  io.perCtxAnyValid := VecInit(anyValid)

  // ---------------------------------------------------------------------------
  // Per-Context gate computation (all real projections; no static ready vector).
  val launchable = (0 until p.contextCount).map(c =>
    io.accepted(c) && io.programReady(c) && !io.advanceGen(c)
  )
  val hold = (0 until p.contextCount).map(c =>
    io.parked(c) || io.cancelled(c) || io.halted(c) ||
      io.waitBusy(c) || io.regionBusy(c) || io.tileIssueFreeze
  )

  // Scalar scoreboard RAW/WAW on the head, against THIS Context's bank.
  val headRawWaw = (0 until p.contextCount).map { c =>
    val d      = heads(c)
    val inst   = io.fetch(c)(0).bits.inst
    val rs1OH  = UIntToOH(inst(19, 15), p.scalarRegCount)
    val rs2OH  = UIntToOH(inst(24, 20), p.scalarRegCount)
    val rdAddr = inst(11, 7)
    val usesRs1Regd = d.jalr || d.isLsu()
    val usesRs2Regd = d.isLsu() && d.readsRs2()
    val readRegd = MuxOR(usesRs1Regd, rs1OH) | MuxOR(usesRs2Regd, rs2OH)
    val readComb = MuxOR(d.readsRs1(), rs1OH) | MuxOR(d.readsRs2(), rs2OH)
    // VERBATIM DispatchV2's scalar write mask (Decode.scala `writesRd`):
    // (!isScalarStore && !isCondBr) || (isFloat && floatWritesRd) || rvvWritesRd.
    // The eligibility prediction consumes the exact predicate the dispatch gate
    // uses, so this head is blocked exactly when DispatchV2 would block it.
    val writesRd = (!d.isScalarStore() && !d.isCondBr()) ||
      (d.isFloat() && d.floatWritesRd()) || d.rvvWritesRd()
    val rdOH = Mux(writesRd, UIntToOH(rdAddr, p.scalarRegCount), 0.U(p.scalarRegCount.W))
    val raw  = (readRegd & io.scoreboardRegd(c)) =/= 0.U ||
      (readComb & io.scoreboardComb(c)) =/= 0.U
    val waw  = (rdOH & io.scoreboardComb(c)) =/= 0.U
    val fltRaw = io.fscoreboard.map { sb =>
      val rs3OH = UIntToOH(inst(31, 27), p.floatRegCount)
      val fRead = MuxOR(d.floatReadsFloatRs1() || d.rvvReadsFloatRs1(), rs1OH) |
        MuxOR(d.floatReadsRs2(), rs2OH) | MuxOR(d.floatReadsRs3(), rs3OH)
      val writesFloatRd = (d.isFloat() && !d.floatWritesRd()) || d.rvvWritesFrd()
      val fWaw = MuxOR(writesFloatRd, UIntToOH(rdAddr, p.floatRegCount))
      (fRead & sb(c)) =/= 0.U || (fWaw & sb(c)) =/= 0.U
    }.getOrElse(false.B)
    raw || waw || fltRaw
  }

  // CSR-order gate on the head, exactly DispatchV2's per-CSR classes evaluated on
  // THIS Context's pending projections (CFG-01..05).
  val headCsrBlocked = (0 until p.contextCount).map { c =>
    val d   = heads(c)
    val idx = io.fetch(c)(0).bits.inst(31, 20)
    val (isF, isC, isM) = CsrOrderedClassify(p)(idx)
    val readsArch = io.fetch(c)(0).bits.inst(11, 7) =/= 0.U
    val isCsrrw   = io.fetch(c)(0).bits.inst(13, 12) === "b01".U
    val pureWrite = !readsArch && isCsrrw
    val fOk = !isF || pureWrite || (!io.flagsInFlight(c) && !io.frmPending(c))
    val cOk = !isC || (!io.configInFlight(c) && !io.flagsInFlight(c))
    val mOk = !isM || io.robEmpty(c)
    d.isCsr() && !(fOk && cOk && mOk)
  }

  // Vector-op ordered interlock (Context-local, W2-D04 a+b).
  val headRvvOrderedBlocked = (0 until p.contextCount).map(c =>
    heads(c).rvv.map(_.valid).getOrElse(false.B) && io.rvvOrderedPending(c)
  )

  // Destination partition gates: ROB space for the head entry, no pending trap.
  val headDestBlocked = (0 until p.contextCount).map(c =>
    io.robSpace(c) === 0.U || io.robTrapPending(c)
  )

  // Shared unit gates: queue capacity plus the owning unit's live readiness, so a
  // granted head is always acceptable and a busy unit cannot hold the Tile hostage.
  val headEngineBlocked = (0 until p.contextCount).map { c =>
    val d = heads(c)
    val isMul = d.mul || d.mulh || d.mulhsu || d.mulhu || d.mulw
    val isDiv = d.div || d.divu || d.rem || d.remu || d.divw || d.divuw || d.remw || d.remuw
    val lsuGate = d.isLsu() && (!io.lsuReqReady || io.lsuQueueCapacity === 0.U)
    val mluGate = isMul && !io.mluReqReady
    val dvuGate = isDiv && !io.dvuReqReady
    val fltGate = io.floatReqReady.map(fr => d.isFloat() && !d.isFloatLoad() && !d.isFloatStore() && !fr)
      .getOrElse(false.B)
    val rvvGate = io.rvvQueueCapacity.map { cap =>
      d.rvv.map(_.valid).getOrElse(false.B) && (cap === 0.U)
    }.getOrElse(false.B)
    val fenceGate = d.isFency() && io.lsuActive
    lsuGate || mluGate || dvuGate || fltGate || rvvGate || fenceGate
  }

  val headBlocked = (0 until p.contextCount).map(c =>
    headCsrBlocked(c) || headRawWaw(c) || headDestBlocked(c) || headEngineBlocked(c) ||
      headRvvOrderedBlocked(c)
  )

  // ---------------------------------------------------------------------------
  // Bounded round-robin arbitration (C027 ContextIssueArbiter, unchanged semantics).
  val arb = Module(new ContextIssueArbiter(p))
  val eligible = (0 until p.contextCount).map(c => launchable(c) && !hold(c) && headValid(c))
  arb.io.eligible := VecInit(eligible)
  arb.io.blocked := VecInit((0 until p.contextCount).map(c => eligible(c) && headBlocked(c)))
  arb.io.issueAccepted := io.issueAccepted

  io.selValid := arb.io.grantValid
  io.selIdx := arb.io.grantIdx
  io.selOH := arb.io.grant
  io.rrPtr := arb.io.rrPtr
  io.serviceCount := arb.io.serviceCount
  io.acceptedTotal := arb.io.acceptedTotal

  // ---------------------------------------------------------------------------
  // Selected-bank bundle mux (W2-D02): one owner per beat, suffix lanes never
  // leak across Contexts. A live cancel discards only the selected Context's
  // unaccepted payload (frontend drain), never RF/ROB state.
  val selCancelled = (0 until p.contextCount).map(c => io.selOH(c) && io.cancelled(c))
    .reduce(_ || _)
  for (i <- 0 until p.instructionLanes) {
    val validSel = (0 until p.contextCount).map(c => io.selOH(c) && io.fetch(c)(i).valid)
    val bitsSel = io.fetch.map(_(i).bits)
    if (p.contextCount == 1) {
      io.instOut(i).valid := io.selValid && validSel.head && !selCancelled
      io.instOut(i).bits := bitsSel.head
    } else {
      io.instOut(i).valid := io.selValid && validSel.reduce(_ || _) && !selCancelled
      io.instOut(i).bits := Mux1H(io.selOH, bitsSel)
    }
    // Pop exactly the accepted prefix of the selected bank; a cancelled selection
    // drains its unaccepted payload instead.
    for (c <- 0 until p.contextCount) {
      io.fetch(c)(i).ready := io.selOH(c) && (selCancelled || io.instOut(i).ready)
    }
  }
  io.issueAccepted := io.instOut.map(_.fire).reduce(_ || _)

  // ---------------------------------------------------------------------------
  // Accepted-lane owner identity: owner_op_seq advances ONLY for accepted lanes of
  // the selected Context (C026/C010 `owner_op_seq_advance`).
  val perCtxOpSeqReg = RegInit(VecInit.fill(p.contextCount)(0.U(p.ownerOpSeqBits.W)))
  val laneFires = io.instOut.map(_.fire)
  for (i <- 0 until p.instructionLanes) {
    val prevFires = PopCount((0 until i).map(j => laneFires(j)))
    val seq = perCtxOpSeqReg(io.selIdx) + prevFires
    io.laneOpSeq(i) := seq
    io.laneOwners(i) := OwnerTag.make(p)(io.selIdx, io.perCtxGen(io.selIdx), seq)
  }
  val totalFires = PopCount(laneFires)
  for (c <- 0 until p.contextCount) {
    perCtxOpSeqReg(c) := Mux(
      io.advanceGen(c),
      0.U,
      Mux(io.selOH(c), perCtxOpSeqReg(c) + totalFires, perCtxOpSeqReg(c))
    )
  }
  io.perCtxOpSeq := perCtxOpSeqReg

  // ---------------------------------------------------------------------------
  // Unique stall reason + per-Context counters. A Context that issued this beat
  // records nothing; parked/cancelled/halted/frozen Contexts record nothing (park is
  // an architectural hold, not a resource stall).
  val stallRegs = RegInit(
    VecInit.fill(p.contextCount)(0.U.asTypeOf(new SchedulerStallCounters))
  )
  val issuedC = (0 until p.contextCount).map(c => io.selOH(c) && io.issueAccepted)
  for (c <- 0 until p.contextCount) {
    val counting = launchable(c) && !io.parked(c) && !io.cancelled(c) &&
      !io.halted(c) && !io.tileIssueFreeze && !issuedC(c)
    val eventStall  = counting && io.waitBusy(c)
    val regionStall = counting && !io.waitBusy(c) && io.regionBusy(c)
    val headBase    = counting && !io.waitBusy(c) && !io.regionBusy(c)
    val fetchStall  = headBase && !headValid(c)
    val csrStall    = headBase && headValid(c) && headCsrBlocked(c)
    val operandStall = headBase && headValid(c) && !headCsrBlocked(c) && headRawWaw(c)
    val destStall = headBase && headValid(c) && !headCsrBlocked(c) && !headRawWaw(c) &&
      headDestBlocked(c)
    val engineStall = headBase && headValid(c) && !headCsrBlocked(c) && !headRawWaw(c) &&
      !headDestBlocked(c) && (headEngineBlocked(c) || headRvvOrderedBlocked(c))

    val reason = MuxCase(
      IssueStallReason.NONE,
      Seq(
        eventStall   -> IssueStallReason.WAIT_EVENT,
        regionStall  -> IssueStallReason.WAIT_REGION,
        fetchStall   -> IssueStallReason.WAIT_FETCH,
        csrStall     -> IssueStallReason.WAIT_CSR,
        operandStall -> IssueStallReason.WAIT_OPERAND,
        destStall    -> IssueStallReason.WAIT_DEST,
        engineStall  -> IssueStallReason.WAIT_ENGINE_QUEUE
      )
    )
    io.waitReason(c) := reason.asUInt
    io.waitKind(c) := MuxCase(
      ContextWait.NONE,
      Seq(
        fetchStall   -> ContextWait.FETCH,
        operandStall -> ContextWait.OPERAND,
        destStall    -> ContextWait.DEST,
        engineStall  -> ContextWait.ENGINE_QUEUE
      )
    )
    when(io.advanceGen(c)) {
      stallRegs(c) := 0.U.asTypeOf(new SchedulerStallCounters)
    }.otherwise {
      stallRegs(c).event  := stallRegs(c).event + eventStall.asUInt
      stallRegs(c).region := stallRegs(c).region + regionStall.asUInt
      stallRegs(c).fetch  := stallRegs(c).fetch + fetchStall.asUInt
      stallRegs(c).csr    := stallRegs(c).csr + csrStall.asUInt
      stallRegs(c).operand := stallRegs(c).operand + operandStall.asUInt
      stallRegs(c).dest   := stallRegs(c).dest + destStall.asUInt
      stallRegs(c).engineQueue := stallRegs(c).engineQueue + engineStall.asUInt
    }
  }
  io.stallCount := stallRegs

  // Bundle integrity: every offered lane belongs to the granted Context and the
  // accepted set is a prefix (perctx_issue.yaml same_context / prefix_only).
  if (p.enableVerification) {
    for (i <- 0 until p.instructionLanes) {
      assert(!(io.instOut(i).fire) || io.selValid, "issue lane fired without a grant")
      if (i > 0) {
        assert(
          !io.instOut(i).fire || io.instOut(i - 1).fire,
          "non-prefix issue: lane fired while the preceding lane did not"
        )
      }
    }
  }
}
