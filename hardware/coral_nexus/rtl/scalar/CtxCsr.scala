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

/** Retired flags-delta update port consumed at instruction retirement (W2-02/W2-04). */
class RetiredFlagsDeltaIO(p: Parameters) extends Bundle {
  val valid         = Bool()
  val owner         = new OwnerTag(p)
  val fflags        = UInt(5.W)
  val vxsat         = Bool()
  val replaceFflags = Bool()
  val replaceVxsat  = Bool()
}

/** Per-Context CSR bank (`W2-02` / C026).
  *
  * Each Context slot `b` owns an independent `Csr` bank whose `mhartid` encodes `(tileId, b)`,
  * isolated trap CSRs (`mtvec`, `mepc`, `mcause`, `mtval`, `mscratch`), counters (`mcycle`,
  * `minstret`), and floating/vector configuration & flags state.
  *
  * Generation isolation (`CTX-08/09`, `INV-15`):
  *   - `advanceGen(b)` increments `perCtxGen(b)` and synchronously resets bank `b` (clearing
  *     its flags accumulator and per-Context writable CSRs) without disturbing any peer Context.
  *   - Any CSR request, trap update, or flags delta carrying an `OwnerTag` whose
  *     `ctx_generation =/= perCtxGen(ctx_slot)` is rejected and counted in `staleRejected`.
  */
class CtxCsr(p: Parameters) extends Module {
  val io = IO(new Bundle {
    val activeCtx   = Input(UInt(p.ctxBits.W))
    val advanceGen  = Input(Vec(p.contextCount, Bool()))
    val perCtxGen   = Output(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))

    val csr          = new CsrInOutIO(p)
    val req          = Flipped(Valid(new CsrCmd(p)))
    val reqOwner     = Input(new OwnerTag(p))
    val rs1          = Flipped(new RegfileReadDataIO(p))
    val rd           = Output(Valid(new OwnedRegfileWriteDataIO(p)))
    val bru          = Flipped(new CsrBruIO(p))
    val bruOwner     = Input(new OwnerTag(p))
    val float        = Option.when(p.enableFloat)(Flipped(new CsrFloatIO(p)))
    val floatOwner   = Option.when(p.enableFloat)(Input(new OwnerTag(p)))
    val rvv          = Option.when(p.enableRvv)(new CsrRvvIO(p))
    val rvvOwner     = Option.when(p.enableRvv)(Input(new OwnerTag(p)))
    val float_dirty  = Option.when(p.enableFloat)(Input(Bool()))
    val rvv_dirty    = Option.when(p.enableRvv)(Input(Bool()))
    val vme_discard  = Option.when(p.enableVme)(Input(Bool()))
    val vme_ms_dirty = Option.when(p.enableVme)(Input(Bool()))
    val vme_vs_dirty = Option.when(p.enableVme)(Input(Bool()))
    val vme_fp_dirty = Option.when(p.enableVme && p.enableFloat)(Input(Bool()))
    val mstatus_ms   = Option.when(p.enableVme)(Output(UInt(2.W)))

    val perCtxNRetired = Input(Vec(p.contextCount, UInt(log2Ceil(p.retirementLanes + 1).W)))
    val retiredDelta   = Input(Vec(p.retirementLanes, new RetiredFlagsDeltaIO(p)))

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
  })

  val effActiveCtx = if (p.contextCount == 1) 0.U(1.W) else io.activeCtx

  val genRegs = RegInit(VecInit.fill(p.contextCount)(0.U(p.ctxGenerationBits.W)))
  for (b <- 0 until p.contextCount) {
    when(io.advanceGen(b)) {
      genRegs(b) := genRegs(b) + 1.U
    }
  }
  io.perCtxGen := genRegs

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

  def selCtx[T <: Data](items: Seq[T], idx: UInt): T = {
    if (items.length == 1) items.head else VecInit(items)(idx)
  }

  def isLiveOwner(owner: OwnerTag): Bool = {
    val slot      = owner.ctx_slot
    val slotInRng = if (p.contextCount == 1) (slot === 0.U) else (slot < p.contextCount.U)
    val liveGen   = if (p.contextCount == 1) genRegs(0) else genRegs(Mux(slotInRng, slot, 0.U))
    slotInRng && (owner.ctx_generation === liveGen)
  }

  val reqStale = io.req.valid && !isLiveOwner(io.reqOwner)
  val bruAnyValid = io.bru.in.mode.valid || io.bru.in.mcause.valid ||
    io.bru.in.mepc.valid || io.bru.in.mtval.valid ||
    io.bru.in.halt || io.bru.in.fault || io.bru.in.wfi
  val bruStale = bruAnyValid && !isLiveOwner(io.bruOwner)
  val fltStale = io.float
    .zip(io.floatOwner)
    .map { case (f, own) => f.in.fflags.valid && !isLiveOwner(own) }
    .getOrElse(false.B)
  val rvvStale = io.rvv
    .zip(io.rvvOwner)
    .map { case (r, own) => r.fflags.valid && !isLiveOwner(own) }
    .getOrElse(false.B)
  val deltaStaleVec = VecInit((0 until p.retirementLanes).map { k =>
    io.retiredDelta(k).valid && !isLiveOwner(io.retiredDelta(k).owner)
  })

  val staleCounter = RegInit(0.U(16.W))
  staleCounter := staleCounter +
    reqStale.asUInt +& bruStale.asUInt +& fltStale.asUInt +& rvvStale.asUInt +&
    PopCount(deltaStaleVec)
  io.staleRejected := staleCounter

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

    val matchingDeltas = (0 until p.retirementLanes).map { k =>
      val d = io.retiredDelta(k)
      d.valid && notAdv && OwnerTag.sameContext(d.owner, b.U, liveGen)
    }
    val hasDeltaHit = VecInit(matchingDeltas).asUInt.orR
    val mergedDeltaFflags = (0 until p.retirementLanes).foldLeft(0.U(5.W)) { (acc, k) =>
      val d   = io.retiredDelta(k)
      val hit = matchingDeltas(k)
      Mux(hit, Mux(d.replaceFflags, d.fflags, acc | d.fflags), acc)
    }
    val anyReplaceFflags = (0 until p.retirementLanes)
      .map(k => matchingDeltas(k) && io.retiredDelta(k).replaceFflags)
      .reduce(_ || _)
    bank.io.retiredFflagsReplace := anyReplaceFflags
    bank.io.retiredFflagsBits    := mergedDeltaFflags

    if (p.enableFloat) {
      val fltMatch = notAdv && OwnerTag.sameContext(io.floatOwner.get, b.U, liveGen)
      val fflagsInValid = (fltMatch && io.float.get.in.fflags.valid) || (hasDeltaHit && !anyReplaceFflags)
      val fflagsInBits = Mux(fltMatch && io.float.get.in.fflags.valid, io.float.get.in.fflags.bits, 0.U(5.W)) |
        Mux(hasDeltaHit && !anyReplaceFflags, mergedDeltaFflags, 0.U(5.W))
      bank.io.float.get.in.fflags.valid := fflagsInValid
      bank.io.float.get.in.fflags.bits  := fflagsInBits
      bank.io.float_dirty.get           := (fltMatch && io.float_dirty.get) || fflagsInValid || anyReplaceFflags
    }

    if (p.enableRvv) {
      val rvvMatch = notAdv && OwnerTag.sameContext(io.rvvOwner.get, b.U, liveGen)
      bank.io.rvv.get.vstart       := io.rvv.get.vstart
      bank.io.rvv.get.vl           := io.rvv.get.vl
      bank.io.rvv.get.vtype        := io.rvv.get.vtype
      bank.io.rvv.get.vxrm         := io.rvv.get.vxrm
      bank.io.rvv.get.vxsat        := io.rvv.get.vxsat
      bank.io.rvv.get.fflags.valid := rvvMatch && io.rvv.get.fflags.valid
      bank.io.rvv.get.fflags.bits  := io.rvv.get.fflags.bits
      if (p.enableVme) {
        bank.io.rvv.get.mtype := io.rvv.get.mtype
      } else {
        bank.io.rvv.get.mtype := 0.U
      }
      bank.io.rvv_dirty.get := rvvMatch && io.rvv_dirty.get
    }

    if (p.enableVme) {
      bank.io.vme_discard.get  := isSel && notAdv && io.vme_discard.get
      bank.io.vme_ms_dirty.get := isSel && notAdv && io.vme_ms_dirty.get
      bank.io.vme_vs_dirty.get := isSel && notAdv && io.vme_vs_dirty.get
      if (p.enableFloat) {
        bank.io.vme_fp_dirty.get := isSel && notAdv && io.vme_fp_dirty.get
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

  io.csr.out                 := selCtx(banks.map(_.io.csr.out), effActiveCtx)
  io.bru.out.mode            := selCtx(banks.map(_.io.bru.out.mode), effActiveCtx)
  io.bru.out.mepc            := selCtx(banks.map(_.io.bru.out.mepc), effActiveCtx)
  io.bru.out.mtvec           := selCtx(banks.map(_.io.bru.out.mtvec), effActiveCtx)
  io.bru.out.interrupt       := selCtx(banks.map(_.io.bru.out.interrupt), effActiveCtx)
  io.bru.out.interrupt_cause := selCtx(banks.map(_.io.bru.out.interrupt_cause), effActiveCtx)
  io.trace                   := selCtx(banks.map(_.io.trace), effActiveCtx)

  // Any bank whose registered `req` fires produces `rd.valid` one cycle after `io.req.valid`.
  val reqOwnerReg = RegNext(io.reqOwner, OwnerTag.zero(p))
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
    io.rvv.get.vstart_write := selCtx(banks.map(_.io.rvv.get.vstart_write), effActiveCtx)
    io.rvv.get.vxrm_write   := selCtx(banks.map(_.io.rvv.get.vxrm_write), effActiveCtx)
    io.rvv.get.vxsat_write  := selCtx(banks.map(_.io.rvv.get.vxsat_write), effActiveCtx)
    io.rvv.get.frm          := selCtx(banks.map(_.io.rvv.get.frm), effActiveCtx)
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
