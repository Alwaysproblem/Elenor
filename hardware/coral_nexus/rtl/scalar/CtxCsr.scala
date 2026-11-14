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
import coralnpu.float.CsrFloatIO

/** Bank-indexed native state and owner-qualified CSR writes. Configuration
  * readback itself lives in the scalar retirement bank, not shared engine state.
  */
class CtxRvvIO(p: Parameters) extends Bundle {
  val vstart = Input(Vec(p.contextCount, UInt(log2Ceil(p.rvvVlen).W)))
  val vxrm = Input(Vec(p.contextCount, UInt(2.W)))
  val vxsat = Output(Bool())
  val frm = Output(UInt(3.W))
  val vstart_write = Output(Valid(UInt(log2Ceil(p.rvvVlen).W)))
  val vxrm_write = Output(Valid(UInt(2.W)))
  val writeOwner = Output(new OwnerTag(p))
}

/** Per-Context CSR bank (`W2-02` C026 + `W2-04` C027).
  *
  * Each Context slot `b` owns an independent `Csr` bank whose `mhartid` encodes `(tileId, b)`,
  * isolated trap CSRs (`mtvec`, `mepc`, `mcause`, `mtval`, `mscratch`), counters (`mcycle`,
  * `minstret`), and floating/vector configuration & flags state.
  *
  * W2-04 ordered merge (CFG-04): the architectural `fflags` accumulator is ONLY updated by
  * `retiredDelta` at each entry's own retirement beat, in its Context's program order.
  * FloatCore's direct `float.in.fflags` path is disabled in full mode (deltas park in the
  * ROB); software `csrw fflags/fcsr` likewise parks and commits at retirement. `frm`/`fcsr`
  * frm-bits retire-commit with the LAST retiring `commitsFrm` lane winning (program order).
  *
  * Generation isolation (CTX-08/09, INV-15): `advanceGen(b)` increments `perCtxGen(b)` and
  * synchronously resets bank `b`; any request, trap update, delta, or software CSR write
  * carrying a stale `ctx_generation` is rejected and counted in `staleRejected`.
  */
class CtxCsrIO(p: Parameters) extends Bundle {
    val activeCtx  = Input(UInt(p.ctxBits.W))
    val advanceGen = Input(Vec(p.contextCount, Bool()))
    val perCtxGen  = Output(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))

    val csr          = new CsrInOutIO(p)
    val req          = Flipped(Valid(new CsrCmd(p)))
    val reqOwner     = Input(new OwnerTag(p))
    val rs1          = Flipped(new RegfileReadDataIO(p))
    val rd           = Output(Valid(new OwnedRegfileWriteDataIO(p)))
    val bru          = Flipped(new CsrBruIO(p))
    val bruOwner     = Input(new OwnerTag(p))
    val float        = Option.when(p.enableFloat)(Flipped(new CsrFloatIO(p)))
    val floatOwner   = Option.when(p.enableFloat)(Input(new OwnerTag(p)))
    val rvv          = Option.when(p.enableRvv)(new CtxRvvIO(p))
    val float_dirty  = Option.when(p.enableFloat)(Input(Bool()))
    val perCtxRvvDirty = Option.when(p.enableRvv)(Input(Vec(p.contextCount, Bool())))
    val vme_discard  = Option.when(p.enableVme)(Input(Bool()))
    val vme_ms_dirty = Option.when(p.enableVme)(Input(Bool()))
    val vme_vs_dirty = Option.when(p.enableVme)(Input(Bool()))
    val vme_fp_dirty = Option.when(p.enableVme && p.enableFloat)(Input(Bool()))
    val vmeOwner = Option.when(p.enableVme)(Input(new OwnerTag(p)))
    val mstatus_ms   = Option.when(p.enableVme)(Output(UInt(2.W)))

    val perCtxNRetired = Input(Vec(p.contextCount, UInt(log2Ceil(p.retirementLanes + 1).W)))
    // Flat bus: context c's retiring lanes occupy c*retirementLanes + lane (program order
    // within a Context preserved; cross-Context order irrelevant — banks filter by owner).
    val retiredDelta = Input(Vec(p.contextCount * p.retirementLanes, new RetiredFlagsDeltaIO(p)))
    // Winning bank's software-CSR park export (observed one beat after dispatch).
    val csrParkOut = Output(Valid(new ParkedCsrDeltaIO(p)))

    val halted       = Output(Bool())
    val fault        = Output(Bool())
    val wfi          = Output(Bool())
    val perCtxHalted = Output(Vec(p.contextCount, Bool()))
    val perCtxFault  = Output(Vec(p.contextCount, Bool()))
    val perCtxWfi    = Output(Vec(p.contextCount, Bool()))

    val irq = Input(Bool())
    val dm  = new Bundle {
      val debug_req           = Input(Bool())
      val resume_req          = Input(Bool())
      val debug_mode          = Output(Bool())
      val entering_debug_mode = Output(Bool())
      val single_step         = Output(Bool())
      val dcsr_step           = Output(Bool())
      val current_pc          = Input(UInt(p.programCounterBits.W))
      val next_pc             = Input(UInt(p.programCounterBits.W))
      val debug_pc            = Valid(UInt(p.fetchAddrBits.W))
    }
    val timer_irq     = Input(Bool())
    val software_irq  = Input(Bool())
    val trace         = Output(new CsrTraceIO(p))
    val staleRejected = Output(UInt(16.W))

    // Direct per-Context readouts for unit & integration verification.
    val perCtxFflags   = Output(Vec(p.contextCount, UInt(5.W)))
    val perCtxFrm      = Output(Vec(p.contextCount, UInt(3.W)))
    val perCtxMscratch = Output(Vec(p.contextCount, UInt(p.xlen.W)))
    val perCtxMepc     = Output(Vec(p.contextCount, UInt(p.programCounterBits.W)))
    val perCtxMcause   = Output(Vec(p.contextCount, UInt(p.xlen.W)))
    // W2-04 / C027: software-CSR park owner must ride the SAME one-beat pipeline as the
    // bank's internal `req` register (Csr computes wdata one beat after io.req); DM
    // (debug-module) CSR writes bypass the park and apply architecturally at the req beat.
    val reqFromDm         = Input(Bool())
    // Per-Context trap flush: kills in-flight parked frm/config writers of that Context
    // (their deltas die with the flush), so the frm pending-shadow must drop immediately.
    val perCtxTrapRetired = Input(Vec(p.contextCount, Bool()))
    // Dispatch-beat frm snapshot state: true while an older frm-class write is observed
    // but un-retired (younger FP ops consume the shadow value, not the architectural frm).
    val perCtxFrmPending  = Output(Vec(p.contextCount, Bool()))
    // Retire-point config commit records from the ROB (per-op immutable payloads).
    val perCtxRetiredConfig = Input(Vec(p.contextCount, Valid(new RetiredConfigIO(p))))
    // Retire-point vxsat commit for software `csrw vxsat/vcsr` (last retiring lane wins).
    val perCtxRetiredVxsat = Output(Vec(p.contextCount, Valid(UInt(1.W))))
    // Per-Context stale-reject counters: cleared ONLY by that Context's own advanceGen
    // (safe slot reuse), never globally — peer Contexts keep their exact counts (INV-15).
    val perCtxStaleRejected = Output(Vec(p.contextCount, UInt(16.W)))}

class CtxCsr(p: Parameters) extends Module {
  val io = IO(new CtxCsrIO(p))

  val effActiveCtx = if (p.contextCount == 1) 0.U(1.W) else io.activeCtx

  val genRegs = RegInit(VecInit.fill(p.contextCount)(0.U(p.ctxGenerationBits.W)))
  for (b <- 0 until p.contextCount) {
    when(io.advanceGen(b)) {
      genRegs(b) := genRegs(b) + 1.U
    }
  }
  io.perCtxGen := genRegs

  def selCtx[T <: Data](items: Seq[T], idx: UInt): T = {
    if (items.length == 1) items.head else VecInit(items)(idx)
  }

  def copyParamsWithHart(hart: Int): Parameters = {
    val q = new Parameters(p.m, hartId = hart, xlen = p.xlen)
    q.enableVerification        = p.enableVerification
    q.rawExposeDebugPorts       = p.rawExposeDebugPorts
    q.enableRvv                 = p.enableRvv
    q.enableVme                 = p.enableVme
    q.enableFloat               = p.enableFloat
    q.enableZfbfmin             = p.enableZfbfmin
    q.enableVectorBf16          = p.enableVectorBf16
    q.fpuPipeRegs               = p.fpuPipeRegs
    q.enableAxiInstructionFetch = p.enableAxiInstructionFetch
    q.fetchDataBits             = p.fetchDataBits
    q.lsuDataBits               = p.lsuDataBits
    q.itcmSizeKBytes            = p.itcmSizeKBytes
    q.dtcmSizeKBytes            = p.dtcmSizeKBytes
    q.tileId                    = p.tileId
    q.contextCount              = p.contextCount
    q.ctxGenerationBits         = p.ctxGenerationBits
    q.ownerOpSeqBits            = p.ownerOpSeqBits
    q.slotGenerationBits        = p.slotGenerationBits
    q.subopIdBits               = p.subopIdBits
    q
  }

  val banks = Seq.tabulate(p.contextCount) { b =>
    val encodedHart = (p.tileId << p.ctxBits) | b
    withReset(reset.asBool || io.advanceGen(b)) {
      Module(new Csr(copyParamsWithHart(encodedHart)))
    }
  }

  def isLiveOwner(owner: OwnerTag): Bool = {
    val slot      = owner.ctx_slot
    val slotInRng = if (p.contextCount == 1) (slot === 0.U) else (slot < p.contextCount.U)
    val liveGen   = if (p.contextCount == 1) genRegs(0) else genRegs(Mux(slotInRng, slot, 0.U))
    slotInRng && (owner.ctx_generation === liveGen)
  }


  // W2-04: the bank's internal `req` register delays every request by one beat, so the
  // owner / source tags captured WITH a request must be pipelined by the same amount to
  // stay aligned with the beat in which the bank observes it (park export, DM bypass).
  val reqOwnerReg = RegNext(io.reqOwner, OwnerTag.zero(p))
  val reqFromDmReg = RegNext(io.reqFromDm, false.B)
  val nDeltaLanes = p.contextCount * p.retirementLanes

  // W2-04 retire-point architectural config: per-Context committed vxsat / vl / vtype /
  // mtype. These — not the shared engine's live state — are the authoritative CSR
  // readback: a not-yet-retired or flushed config writer must never be observable.
  val committedVxsat  = RegInit(VecInit.fill(p.contextCount)(false.B))
  val committedVl     = RegInit(VecInit.fill(p.contextCount)(0.U(log2Ceil(p.rvvVlen + 1).W)))
  val resetVtype = (BigInt(1) << (p.xlen - 1)).U(p.xlen.W)
  val committedVtype  = RegInit(VecInit.fill(p.contextCount)(resetVtype))
  val committedMtype  = RegInit(VecInit.fill(p.contextCount)(0.U(p.xlen.W)))
  val reqStale = io.req.valid && !isLiveOwner(io.reqOwner)
  val bruAnyValid = io.bru.in.mode.valid || io.bru.in.mcause.valid ||
    io.bru.in.mepc.valid || io.bru.in.mtval.valid ||
    io.bru.in.halt || io.bru.in.fault || io.bru.in.wfi
  val bruStale = bruAnyValid && !isLiveOwner(io.bruOwner)
  val fltStale = io.float
    .zip(io.floatOwner)
    .map { case (f, own) => f.in.fflags.valid && !isLiveOwner(own) }
    .getOrElse(false.B)
  val deltaStaleVec = VecInit((0 until nDeltaLanes).map { k =>
    io.retiredDelta(k).valid && !isLiveOwner(io.retiredDelta(k).owner)
  })

  val staleCounter = RegInit(0.U(16.W))
  staleCounter := staleCounter +
    reqStale.asUInt +& bruStale.asUInt +& fltStale.asUInt +&
    PopCount(deltaStaleVec)
  io.staleRejected := staleCounter

  // Per-Context attribution of the same stale events, keyed by the TARGET Context slot.
  // Each counter clears ONLY on its own Context's advanceGen (safe slot reuse, §8 step 7),
  // so reusing Context A never wipes Context B's exact reject count (INV-15).
  val stalePerCtx = RegInit(VecInit.fill(p.contextCount)(0.U(16.W)))
  def ctxIdx(slot: UInt): UInt = if (p.contextCount == 1) 0.U else slot
  val staleEvents = Seq(
    (reqStale, io.reqOwner.ctx_slot),
    (bruStale, io.bruOwner.ctx_slot)
  ) ++ io.float.zip(io.floatOwner).map { case (_, own) => (fltStale, own.ctx_slot) } ++
    (0 until nDeltaLanes).map(k => (deltaStaleVec(k), io.retiredDelta(k).owner.ctx_slot))
  for (c <- 0 until p.contextCount) {
    val inc = PopCount(VecInit(staleEvents.map { case (ev, slot) =>
      ev && (ctxIdx(slot) === c.U)
    }))
    when(io.advanceGen(c)) {
      stalePerCtx(c) := 0.U
    }.otherwise {
      stalePerCtx(c) := stalePerCtx(c) + inc
    }
  }
  io.perCtxStaleRejected := stalePerCtx

  for (b <- 0 until p.contextCount) {
    val bank      = banks(b)
    val isSel     = (effActiveCtx === b.U)
    val liveGen   = genRegs(b)
    val notAdv    = !io.advanceGen(b)
    val reqMatch  = io.req.valid && notAdv && OwnerTag.sameContext(io.reqOwner, b.U, liveGen)
    val bruMatch  = notAdv && OwnerTag.sameContext(io.bruOwner, b.U, liveGen)

    bank.io.csr.in       := io.csr.in
    bank.io.req.valid    := reqMatch
    bank.io.req.bits     := io.req.bits
    // The bank registers `req` for one beat before computing wdata/csrParkOut, so the
    // park owner must ride the SAME pipeline: pass the owner captured at acceptance,
    // not the combinational input (which by then belongs to the next request).
    bank.io.reqOwner     := reqOwnerReg
    bank.io.reqFromDm    := reqFromDmReg
    bank.io.rs1          := io.rs1
    bank.io.counters.nRetired := io.perCtxNRetired(b)

    bank.io.bru.in.mode.valid   := bruMatch && io.bru.in.mode.valid
    bank.io.bru.in.mode.bits    := io.bru.in.mode.bits
    bank.io.bru.in.mcause.valid := bruMatch && io.bru.in.mcause.valid
    bank.io.bru.in.mcause.bits  := io.bru.in.mcause.bits
    bank.io.bru.in.mepc.valid   := bruMatch && io.bru.in.mepc.valid
    bank.io.bru.in.mepc.bits    := io.bru.in.mepc.bits
    bank.io.bru.in.mtval.valid  := bruMatch && io.bru.in.mtval.valid
    bank.io.bru.in.mtval.bits   := io.bru.in.mtval.bits
    bank.io.bru.in.halt         := bruMatch && io.bru.in.halt
    bank.io.bru.in.fault        := bruMatch && io.bru.in.fault
    bank.io.bru.in.wfi          := bruMatch && io.bru.in.wfi

    // ---- W2-04 ordered merge: only retiring-lane deltas touch the accumulator ----
    val matchingDeltas = (0 until nDeltaLanes).map { k =>
      val d = io.retiredDelta(k)
      d.valid && notAdv && OwnerTag.sameContext(d.owner, b.U, liveGen)
    }
    val mergedDeltaFflags = (0 until nDeltaLanes).foldLeft(0.U(5.W)) { (acc, k) =>
      val d   = io.retiredDelta(k)
      val hit = matchingDeltas(k)
      Mux(hit, Mux(d.replaceFflags, d.fflags, acc | d.fflags), acc)
    }
    val anyReplaceFflags = (0 until nDeltaLanes)
      .map(k => matchingDeltas(k) && io.retiredDelta(k).replaceFflags)
      .reduce(_ || _)
    bank.io.retiredFflagsReplace := anyReplaceFflags
    bank.io.retiredFflagsValid   := VecInit(matchingDeltas).asUInt.orR
    bank.io.retiredFflagsBits    := mergedDeltaFflags
    // frm retire-point commit: the LAST retiring lane with commitsFrm wins (program
    // order — within one Context higher lane index = younger). PriorityMux selects the
    // FIRST hit, so scan from the END of the flat bus (cross-Context hits belong to
    // other banks and never match here).
    val frmHits = VecInit((0 until nDeltaLanes).map { k =>
      matchingDeltas(k) && io.retiredDelta(k).commitsFrm
    })
    bank.io.retiredFrmCommit.valid := frmHits.asUInt.orR
    bank.io.retiredFrmCommit.bits  := PriorityMux(
      frmHits.reverse,
      (0 until nDeltaLanes).map(k => io.retiredDelta(nDeltaLanes - 1 - k).frmValue)
    )

    // Fold every retiring vxsat action in program order: a younger OR must
    // survive an older replace/clear in the same retirement beat.
    val mergedVxsat = (0 until nDeltaLanes).foldLeft(committedVxsat(b)) { (acc, k) =>
      val d = io.retiredDelta(k)
      Mux(matchingDeltas(k), Mux(d.replaceVxsat, d.vxsat, acc || d.vxsat), acc)
    }
    io.perCtxRetiredVxsat(b).valid := VecInit((0 until nDeltaLanes).map { k =>
      matchingDeltas(k) && (io.retiredDelta(k).replaceVxsat || io.retiredDelta(k).vxsat)
    }).asUInt.orR
    io.perCtxRetiredVxsat(b).bits := mergedVxsat.asUInt

    // frm pending shadow flush: a trap flush kills every in-flight frm-class writer of
    // this Context, so the shadow must drop in the same beat (their parked deltas die).
    bank.io.flushPending   := io.perCtxTrapRetired(b)
    // Exact frm retire count (several writers may retire in one beat; the coalesced
    // last-lane value alone would under-decrement the pending-shadow counter).
    bank.io.retiredFrmCommitCount := PopCount(frmHits)
    // Per-Context committed config: vxsat merges in retire order; vl/vtype/mtype latch
    // at a config-class writer's own retirement; advanceGen (safe slot reuse) clears.
    committedVxsat(b) := mergedVxsat
    when(io.perCtxRetiredConfig(b).valid) {
      val cfg = io.perCtxRetiredConfig(b).bits
      // Masks include all architectural effects, including vset's MTYPE clear
      // and msetmtype's VTYPE update/VL clear; narrow setters preserve other fields.
      if (p.enableRvv) {
        when(cfg.wrVl) { committedVl(b) := cfg.vl }
        when(cfg.wrVtype) { committedVtype(b) := cfg.vtype }
      }
      if (p.enableVme) {
        when(cfg.wrMtype) { committedMtype(b) := cfg.mtype }
      }
    }
    when(io.advanceGen(b)) {
      committedVxsat(b)  := false.B
      committedVl(b)     := 0.U
      committedVtype(b)  := resetVtype
      committedMtype(b)  := 0.U
    }
    io.perCtxFrmPending(b) := bank.io.frmPendingOut

    if (p.enableFloat) {
      // CFG-04: the architectural fflags accumulator updates ONLY via retiredDelta.
      // FloatCore's direct path is disabled in full mode; its deltas park in the ROB.
      val fltMatch = notAdv && OwnerTag.sameContext(io.floatOwner.get, b.U, liveGen)
      bank.io.float.get.in.fflags.valid := false.B
      bank.io.float.get.in.fflags.bits  := 0.U
      bank.io.float_dirty.get           := (fltMatch && io.float_dirty.get) || anyReplaceFflags
    }

    if (p.enableRvv) {
      bank.io.rvv.get.vstart := io.rvv.get.vstart(b)
      bank.io.rvv.get.vxrm := io.rvv.get.vxrm(b)
      bank.io.rvv.get.vxsat := committedVxsat(b)
      bank.io.rvv.get.vl := committedVl(b)
      bank.io.rvv.get.vtype := committedVtype(b)
      bank.io.rvv.get.mtype := committedMtype(b)
      bank.io.rvv_dirty.get := notAdv && (io.perCtxRvvDirty.get(b) ||
        io.perCtxRetiredConfig(b).valid || io.perCtxRetiredVxsat(b).valid)
      // Debug-module CSR writes bypass instruction retirement but still own one bank.
      when(notAdv && bank.io.rvv.get.vxsat_write.valid) {
        committedVxsat(b) := bank.io.rvv.get.vxsat_write.bits
      }
    }

    if (p.enableVme) {
      val vmeMatch = notAdv && OwnerTag.sameContext(io.vmeOwner.get, b.U, liveGen)
      bank.io.vme_discard.get  := vmeMatch && io.vme_discard.get
      bank.io.vme_ms_dirty.get := vmeMatch && io.vme_ms_dirty.get
      bank.io.vme_vs_dirty.get := vmeMatch && io.vme_vs_dirty.get
      if (p.enableFloat) {
        bank.io.vme_fp_dirty.get := vmeMatch && io.vme_fp_dirty.get
      }
    }

    bank.io.irq             := io.irq
    bank.io.timer_irq       := io.timer_irq
    bank.io.software_irq    := io.software_irq
    bank.io.dm.debug_req    := isSel && io.dm.debug_req
    bank.io.dm.resume_req   := isSel && io.dm.resume_req
    bank.io.dm.current_pc   := io.dm.current_pc
    bank.io.dm.next_pc      := io.dm.next_pc

    io.perCtxHalted(b)   := bank.io.halted
    io.perCtxFault(b)    := bank.io.fault
    io.perCtxWfi(b)      := bank.io.wfi
    io.perCtxFflags(b)   := bank.io.obsFflags
    io.perCtxFrm(b)      := bank.io.obsFrm
    io.perCtxMscratch(b) := bank.io.obsMscratch
    io.perCtxMepc(b)     := bank.io.csr.out.value(1)
    io.perCtxMcause(b)   := bank.io.csr.out.value(3)
  }

  // Software CSR park export: the winning (live) bank's observation wins; stale owners
  // are rejected upstream so at most one bank observes a live write at a time.
  val parkHits = VecInit(banks.map(_.io.csrParkOut.valid))
  io.csrParkOut.valid := parkHits.asUInt.orR
  io.csrParkOut.bits  := PriorityMux(parkHits, banks.map(_.io.csrParkOut.bits))

  io.csr.out                 := selCtx(banks.map(_.io.csr.out), effActiveCtx)
  io.bru.out.mode            := selCtx(banks.map(_.io.bru.out.mode), effActiveCtx)
  io.bru.out.mepc            := selCtx(banks.map(_.io.bru.out.mepc), effActiveCtx)
  io.bru.out.mtvec           := selCtx(banks.map(_.io.bru.out.mtvec), effActiveCtx)
  io.bru.out.interrupt       := selCtx(banks.map(_.io.bru.out.interrupt), effActiveCtx)
  io.bru.out.interrupt_cause := selCtx(banks.map(_.io.bru.out.interrupt_cause), effActiveCtx)
  io.trace                   := selCtx(banks.map(_.io.trace), effActiveCtx)

  // Any bank whose registered `req` fires produces `rd.valid` one cycle after `io.req.valid`.
  // (reqOwnerReg itself is declared above the bank loop — see the W2-04 pipeline note.)
  val rdHits      = VecInit(banks.map(_.io.rd.valid))
  assert(PopCount(rdHits) <= 1.U)
  io.rd.valid      := rdHits.asUInt.orR
  io.rd.bits.addr  := Mux1H(rdHits, banks.map(_.io.rd.bits.addr))
  io.rd.bits.data  := Mux1H(rdHits, banks.map(_.io.rd.bits.data))
  io.rd.bits.owner := reqOwnerReg

  if (p.enableFloat) {
    io.float.get.out.frm := selCtx(banks.map(_.io.float.get.out.frm), effActiveCtx)
  }
  if (p.enableRvv) {
    val vstartHits = VecInit(banks.zipWithIndex.map { case (bank, b) =>
      bank.io.rvv.get.vstart_write.valid && !io.advanceGen(b)
    })
    val vxrmHits = VecInit(banks.zipWithIndex.map { case (bank, b) =>
      bank.io.rvv.get.vxrm_write.valid && !io.advanceGen(b)
    })
    io.rvv.get.vstart_write.valid := vstartHits.asUInt.orR
    io.rvv.get.vstart_write.bits := Mux1H(vstartHits, banks.map(_.io.rvv.get.vstart_write.bits))
    io.rvv.get.vxrm_write.valid := vxrmHits.asUInt.orR
    io.rvv.get.vxrm_write.bits := Mux1H(vxrmHits, banks.map(_.io.rvv.get.vxrm_write.bits))
    io.rvv.get.writeOwner := reqOwnerReg
    io.rvv.get.vxsat := selCtx(committedVxsat, effActiveCtx)
    io.rvv.get.frm := selCtx(banks.map(_.io.rvv.get.frm), effActiveCtx)
  }
  if (p.enableVme) {
    io.mstatus_ms.get := selCtx(banks.map(_.io.mstatus_ms.get), effActiveCtx)
  }

  io.halted                 := selCtx(io.perCtxHalted, effActiveCtx)
  io.fault                  := io.perCtxFault.reduce(_ || _)
  io.wfi                    := io.perCtxWfi.reduce(_ && _)
  io.dm.debug_mode          := selCtx(banks.map(_.io.dm.debug_mode), effActiveCtx)
  io.dm.entering_debug_mode := selCtx(banks.map(_.io.dm.entering_debug_mode), effActiveCtx)
  io.dm.single_step         := selCtx(banks.map(_.io.dm.single_step), effActiveCtx)
  io.dm.dcsr_step           := selCtx(banks.map(_.io.dm.dcsr_step), effActiveCtx)
  io.dm.debug_pc            := selCtx(banks.map(_.io.dm.debug_pc), effActiveCtx)
}
