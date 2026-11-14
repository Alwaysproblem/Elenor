// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
package coralnpu

import chisel3._
import chisel3.util._
import chisel3.simulator.scalatest.ChiselSim
import coralnpu.rvv.RvvConfigState
import org.scalatest.freespec.AnyFreeSpec

/** Real Decode -> ROB -> CSR request/park/retire chain. Only asynchronous engine
  * completions are driven by the test; CSR requests and completions are NOT injected.
  */
class W2_04_CsrConfigSpec extends AnyFreeSpec with ChiselSim {
  val p = new Parameters
  p.contextCount = 2
  p.enableFloat = true
  p.enableRvv = true
  p.enableVme = true
  p.enableVerification = false

  case class Ticket(ctx: Int, seq: Int, slot: BigInt, slotGen: BigInt, gen: BigInt)
  val fadd = BigInt("003100d3", 16) // fadd.s f1,f2,f3
  val vsadd = BigInt("862180d7", 16) // vsadd.vv v1,v2,v3, unmasked
  val vset0 = BigInt("0000f057", 16) // vsetvli x0,x1,e8,m1; AVL supplied by engine
  val vset5 = BigInt("0000f2d7", 16) // same config, writes x5
  def csrInst(address: Int, funct3: Int, rd: Int, rs1: Int): BigInt =
    (BigInt(address) << 20) | (BigInt(rs1) << 15) | (BigInt(funct3) << 12) | (rd << 7) | 0x73

  class Harness extends Module {
    val dispatch = Module(new DispatchV2(p))
    val bank = Module(new RetirementPartitionBank(p, mini = true))
    val csr = Module(new CtxCsr(p))
    val io = IO(new Bundle {
      val sel = Input(UInt(p.ctxBits.W))
      val seq = Input(UInt(p.ownerOpSeqBits.W))
      val inst = Input(Vec(p.instructionLanes, Valid(UInt(32.W))))
      val rs1 = Input(UInt(p.xlen.W))
      val rvvIdle = Input(Bool())
      val floatReady = Input(Bool())
      val workingVl = Input(UInt(log2Ceil(p.rvvVlen + 1).W))
      val advance = Input(Vec(p.contextCount, Bool()))
      val bankIO = new RetirementPartitionBankIO(p)
      val accepted = Output(Vec(p.instructionLanes, Bool()))
      val csrRequest = Output(Bool())
      val csrResult = Output(Valid(new OwnedRegfileWriteDataIO(p)))
      val fflags = Output(Vec(p.contextCount, UInt(5.W)))
      val frm = Output(Vec(p.contextCount, UInt(3.W)))
      val frmSnapshot = Output(UInt(3.W))
      val frmPending = Output(Vec(p.contextCount, Bool()))
      val vxsat = Output(Bool())
      val generation = Output(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))
    })
    io.bankIO <> bank.io
    bank.io.activeCtx := io.sel
    bank.io.perCtxGen := csr.io.perCtxGen
    bank.io.perCtxReset := io.advance
    bank.io.inst := dispatch.io.inst
    bank.io.targets := dispatch.io.bruTarget
    bank.io.jalrTargets.foreach(_ := 0.U)
    bank.io.jump := dispatch.io.jump
    bank.io.branch := dispatch.io.branch
    bank.io.writeAddrScalar := dispatch.io.rdMark
    bank.io.writeAddrFloat.get := dispatch.io.rdMark_flt.get
    bank.io.writeAddrVector.get := dispatch.io.rvvRdMark.get
    bank.io.isVector.get := dispatch.io.isVector.get
    bank.io.isTile.get := dispatch.io.isTile.get
    bank.io.flagsCapable := dispatch.io.flagsClass
    bank.io.configCapable := dispatch.io.configClass
    bank.io.cfgPayloadExpected := dispatch.io.cfgPayloadExpected
    for (i <- 0 until p.instructionLanes) {
      dispatch.io.inst(i).valid := io.inst(i).valid
      dispatch.io.inst(i).bits.inst := io.inst(i).bits
      dispatch.io.inst(i).bits.addr := 0x1000.U + ((io.seq + i.U) << 2)
      dispatch.io.inst(i).bits.brchFwd := false.B
      dispatch.io.jalrTarget(i).data := 0.U
      bank.io.dispatchOpSeq(i) := io.seq + PopCount((0 until i).map(j => dispatch.io.inst(j).fire))
      io.accepted(i) := dispatch.io.inst(i).fire
      dispatch.io.lsu(i).ready := true.B
      dispatch.io.mlu(i).ready := true.B
      dispatch.io.dvu(i).ready := true.B
      dispatch.io.rvv.get(i).ready := true.B
    }
    dispatch.io.halted := false.B
    dispatch.io.lsuActive := false.B
    dispatch.io.lsuQueueCapacity := 7.U
    dispatch.io.scoreboard.regd := 0.U
    dispatch.io.scoreboard.comb := 0.U
    dispatch.io.fscoreboard.get := 0.U
    dispatch.io.branchTaken := false.B
    dispatch.io.interlock := false.B
    dispatch.io.single_step := false.B
    dispatch.io.debug_mode := false.B
    dispatch.io.float.get.ready := io.floatReady
    dispatch.io.csrFrm.get := csr.io.float.get.out.frm
    dispatch.io.rvvIdle.get := io.rvvIdle
    dispatch.io.rvvQueueCapacity.get := 15.U
    dispatch.io.rvvState.get.valid := true.B
    dispatch.io.rvvState.get.bits := 0.U.asTypeOf(new RvvConfigState(p))
    dispatch.io.rvvState.get.bits.vl := io.workingVl
    dispatch.io.mstatusMs.get := csr.io.mstatus_ms.get
    dispatch.io.retirement_buffer_nSpace := bank.io.nSpace
    dispatch.io.retirement_buffer_empty := bank.io.empty
    dispatch.io.retirement_buffer_trap_pending := bank.io.trapPending
    dispatch.io.csrFlagsPending := bank.io.perCtxFlagsInFlight(io.sel)
    dispatch.io.csrConfigPending := bank.io.perCtxConfigInFlight(io.sel)
    dispatch.io.rvvOrderedPending := bank.io.perCtxParkedFlags(io.sel) || bank.io.perCtxConfigInFlight(io.sel)
    dispatch.io.csrFrmPending := csr.io.perCtxFrmPending(io.sel)

    val owner = OwnerTag.make(p)(io.sel, csr.io.perCtxGen(io.sel), io.seq)
    csr.io.activeCtx := io.sel
    csr.io.advanceGen := io.advance
    csr.io.req := dispatch.io.csr
    csr.io.reqOwner := owner
    csr.io.reqFromDm := false.B
    csr.io.rs1.valid := true.B
    csr.io.rs1.data := RegEnable(io.rs1, 0.U(p.xlen.W), dispatch.io.csr.valid)
    bank.io.writeDataScalar(0) := csr.io.rd
    bank.io.csrPark := csr.io.csrParkOut
    csr.io.retiredDelta := VecInit(bank.io.perCtxRetiredDelta.flatMap(_.toSeq))
    csr.io.perCtxRetiredConfig := bank.io.perCtxRetiredConfig
    csr.io.perCtxTrapRetired := bank.io.perCtxTrapRetired
    csr.io.perCtxNRetired := bank.io.perCtxNRetired
    csr.io.csr.in.value.foreach(_ := 0.U)
    csr.io.bru.in := 0.U.asTypeOf(csr.io.bru.in)
    csr.io.bruOwner := OwnerTag.zero(p)
    csr.io.float.get.in.fflags.valid := false.B
    csr.io.float.get.in.fflags.bits := 0.U
    csr.io.floatOwner.get := OwnerTag.zero(p)
    csr.io.float_dirty.get := false.B
    csr.io.rvv.get.vstart.foreach(_ := 0.U)
    csr.io.rvv.get.vxrm.foreach(_ := 0.U)
    csr.io.perCtxRvvDirty.get.foreach(_ := false.B)
    csr.io.vmeOwner.get := RegNext(owner, OwnerTag.zero(p))
    csr.io.vme_discard.get := RegNext(dispatch.io.vmeDiscard.get, false.B)
    csr.io.vme_ms_dirty.get := RegNext(dispatch.io.vmeMsDirty.get, false.B)
    csr.io.vme_vs_dirty.get := RegNext(dispatch.io.vmeVsDirty.get, false.B)
    csr.io.vme_fp_dirty.get := RegNext(dispatch.io.vmeFpDirty.get, false.B)
    csr.io.irq := false.B
    csr.io.timer_irq := false.B
    csr.io.software_irq := false.B
    csr.io.dm.debug_req := false.B
    csr.io.dm.resume_req := false.B
    csr.io.dm.current_pc := 0.U
    csr.io.dm.next_pc := 0.U
    io.csrRequest := dispatch.io.csr.valid
    io.csrResult := csr.io.rd
    io.fflags := csr.io.perCtxFflags
    io.frm := csr.io.perCtxFrm
    io.frmSnapshot := csr.io.float.get.out.frm
    io.frmPending := csr.io.perCtxFrmPending
    io.vxsat := csr.io.rvv.get.vxsat
    io.generation := csr.io.perCtxGen
  }

  def owner(o: OwnerTag, t: Ticket): Unit = {
    o.ctx_slot.poke(t.ctx.U); o.ctx_generation.poke(t.gen.U)
    o.owner_op_seq.poke(t.seq.U); o.subop_id.poke(0.U)
  }
  def tag(r: RobTag, t: Ticket): Unit = {
    r.ctx_slot.poke(t.ctx.U); r.rob_slot.poke(t.slot.U); r.slot_generation.poke(t.slotGen.U)
  }
  def idle(d: Harness): Unit = {
    d.io.sel.poke(0.U); d.io.seq.poke(0.U); d.io.rs1.poke(0.U)
    d.io.inst.foreach { i => i.valid.poke(false.B); i.bits.poke(0.U) }
    d.io.advance.foreach(_.poke(false.B))
    d.io.rvvIdle.poke(true.B); d.io.floatReady.poke(true.B); d.io.workingVl.poke(0.U)
    val b = d.io.bankIO
    b.writeDataScalar.foreach(_.valid.poke(false.B))
    b.writeDataFloat.get.foreach(_.valid.poke(false.B))
    b.writeDataVector.get.foreach(_.valid.poke(false.B))
    b.writeDataTile.get.valid.poke(false.B)
    b.vmeMtypeWb.get.foreach(_.valid.poke(false.B))
    b.flagsDelta.foreach(_.valid.poke(false.B))
    b.configDelta.foreach(_.valid.poke(false.B))
    b.csrPark.valid.poke(false.B); b.storeComplete.valid.poke(false.B); b.fault.valid.poke(false.B)
  }
  def start(d: Harness): Unit = {
    idle(d); d.reset.poke(true.B); d.clock.step(); d.reset.poke(false.B); d.clock.step()
  }
  def offer(d: Harness, ctx: Int, seq: Int, inst: BigInt, rs1: BigInt = 0): Unit = {
    d.io.sel.poke(ctx.U); d.io.seq.poke(seq.U); d.io.rs1.poke(rs1.U)
    d.io.inst(0).bits.poke(inst.U(32.W)); d.io.inst(0).valid.poke(true.B)
  }
  def issue(d: Harness, ctx: Int, seq: Int, inst: BigInt, rs1: BigInt = 0): Ticket = {
    offer(d, ctx, seq, inst, rs1)
    d.io.accepted(0).expect(true.B)
    val t = Ticket(ctx, seq, d.io.bankIO.enqPtr.peek().litValue,
      d.io.bankIO.enqSlotGen(0).peek().litValue, d.io.generation(ctx).peek().litValue)
    d.clock.step(); d.io.inst(0).valid.poke(false.B)
    t
  }
  def readCsr(d: Harness, ctx: Int, seq: Int, address: Int): BigInt = {
    issue(d, ctx, seq, csrInst(address, 2, 10, 0))
    d.io.csrResult.valid.expect(true.B)
    val value = d.io.csrResult.bits.data.peek().litValue
    d.clock.step()
    value
  }
  def flags(d: Harness, t: Ticket, f: Int = 0, sat: Boolean = false, port: Int = 0): Unit = {
    val delta = d.io.bankIO.flagsDelta(port)
    delta.valid.poke(true.B); delta.bits.valid.poke(true.B)
    owner(delta.bits.owner, t); tag(delta.bits.rob_tag, t)
    delta.bits.fflags.poke(f.U); delta.bits.vxsat.poke(sat.B)
    delta.bits.replaceFflags.poke(false.B); delta.bits.replaceVxsat.poke(false.B)
  }
  def floatDone(d: Harness, t: Ticket): Unit = {
    val w = d.io.bankIO.writeDataFloat.get(0)
    w.valid.poke(true.B); owner(w.bits.owner, t); w.bits.addr.poke(1.U); w.bits.data.poke(0.U)
  }
  def vectorDone(d: Harness, t: Ticket): Unit = {
    val w = d.io.bankIO.writeDataVector.get(0)
    w.valid.poke(true.B); w.bits.w_valid.poke(true.B); w.bits.addr.poke(1.U)
    w.bits.last_uop_valid.poke(true.B); tag(w.bits.rob_tag, t)
  }
  def config(d: Harness, t: Ticket, vl: Int, vt: BigInt, mt: Int, port: Int = 0,
             wrVl: Boolean = true, wrVtype: Boolean = true, wrMtype: Boolean = true): Unit = {
    val x = d.io.bankIO.configDelta(port)
    x.valid.poke(true.B); x.bits.valid.poke(true.B)
    owner(x.bits.owner, t); tag(x.bits.rob_tag, t)
    x.bits.wrVl.poke(wrVl.B); x.bits.wrVtype.poke(wrVtype.B); x.bits.wrMtype.poke(wrMtype.B)
    x.bits.vl.poke(vl.U); x.bits.vtype.poke(vt.U); x.bits.mtype.poke(mt.U)
  }
  def clearCompletions(d: Harness): Unit = {
    d.io.bankIO.flagsDelta.foreach(_.valid.poke(false.B))
    d.io.bankIO.configDelta.foreach(_.valid.poke(false.B))
    d.io.bankIO.writeDataFloat.get.foreach(_.valid.poke(false.B))
    d.io.bankIO.writeDataVector.get.foreach(_.valid.poke(false.B))
    d.io.bankIO.writeDataScalar.foreach(_.valid.poke(false.B))
  }

  "CSR interlock is context-local even when peer vector and floating engines are busy" in {
    simulate(new Harness) { d =>
      start(d)
      val older = issue(d, 0, 100, fadd)
      d.io.rvvIdle.poke(false.B); d.io.floatReady.poke(false.B)
      offer(d, 0, 101, csrInst(0x001, 2, 5, 0))
      d.io.csrRequest.expect(false.B) // mutation target: no own pending flag may be skipped
      d.io.accepted(0).expect(false.B)
      offer(d, 1, 200, csrInst(0x001, 2, 5, 0))
      d.io.csrRequest.expect(true.B)
      d.clock.step(); d.io.inst(0).valid.poke(false.B); d.clock.step()
      // rd=x0 is not enough for a bypass: CSRRS/CSRRC still read-modify-write.
      for (op <- Seq(2, 3, 6, 7)) {
        offer(d, 0, 101, csrInst(0x001, op, 0, 1), rs1 = 1)
        d.io.csrRequest.expect(false.B)
      }
      d.io.inst(0).valid.poke(false.B)
      flags(d, older, f = 8); floatDone(d, older); d.clock.step(); clearCompletions(d)
      assert(readCsr(d, 0, 102, 0x001) == 8)
    }
  }

  "vxsat clear and younger saturation in the same retirement beat preserve the younger delta" in {
    simulate(new Harness) { d =>
      start(d)
      val older = issue(d, 0, 10, vsadd)
      issue(d, 0, 11, csrInst(0x009, 1, 0, 1), rs1 = 0)
      val younger = issue(d, 0, 12, vsadd)
      flags(d, younger, sat = true); vectorDone(d, younger)
      d.clock.step(); clearCompletions(d)
      d.io.vxsat.expect(false.B) // all effects are still behind the older instruction
      flags(d, older, sat = true); vectorDone(d, older)
      d.io.bankIO.perCtxDeqReady(0).expect(3.U)
      d.clock.step(); clearCompletions(d)
      d.io.vxsat.expect(true.B)
      assert(readCsr(d, 0, 13, 0x009) == 1)
    }
  }

  "frm preview is distinct from retirement state and clears its pending count after commit" in {
    simulate(new Harness) { d =>
      start(d)
      val older = issue(d, 0, 20, fadd)
      issue(d, 0, 21, csrInst(0x002, 1, 0, 1), rs1 = 3)
      d.io.frmSnapshot.expect(3.U)
      d.io.frm(0).expect(0.U)
      d.clock.step()
      offer(d, 0, 22, csrInst(0x003, 2, 5, 0))
      d.io.csrRequest.expect(false.B)
      d.io.inst(0).valid.poke(false.B)
      floatDone(d, older); d.clock.step(); clearCompletions(d)
      d.io.frm(0).expect(3.U)
      d.io.frmPending(0).expect(false.B)
      assert(readCsr(d, 0, 23, 0x003) == (3 << 5))
    }
  }

  "early GPR completion is retained until a valid all-zero config payload arrives" in {
    simulate(new Harness) { d =>
      start(d)
      val t = issue(d, 0, 30, vset5)
      val w = d.io.bankIO.writeDataScalar(1)
      w.valid.poke(true.B); w.bits.addr.poke(5.U); w.bits.data.poke(0.U); owner(w.bits.owner, t)
      d.clock.step(); w.valid.poke(false.B)
      d.clock.step(2)
      d.io.bankIO.perCtxEmpty(0).expect(false.B)
      d.io.bankIO.perCtxDeqReady(0).expect(0.U)
      config(d, t, vl = 0, vt = 0, mt = 0)
      d.io.bankIO.perCtxDeqReady(0).expect(1.U)
      d.clock.step(); clearCompletions(d)
      d.io.bankIO.perCtxEmpty(0).expect(true.B)
      assert(readCsr(d, 0, 31, 0xc20) == 0)
      assert(readCsr(d, 0, 32, 0xc21) == 0)
    }
  }

  "simultaneous config results in two contexts use immutable payloads and preserve unmasked fields" in {
    simulate(new Harness) { d =>
      start(d)
      val a = issue(d, 0, 40, vset0)
      val b = issue(d, 1, 40, vset0)
      assert(a.slot == b.slot)
      d.io.workingVl.poke(99.U)
      config(d, a, vl = 3, vt = 0x10, mt = 0x4001, port = 0)
      config(d, b, vl = 7, vt = 0x18, mt = 0x8001, port = 1)
      d.io.bankIO.perCtxDeqReady(0).expect(1.U)
      d.io.bankIO.perCtxDeqReady(1).expect(1.U)
      d.clock.step(); clearCompletions(d)
      assert(readCsr(d, 0, 41, 0xc20) == 3)
      assert(readCsr(d, 1, 41, 0xc20) == 7)
      assert(readCsr(d, 0, 42, 0xc21) == 0x10)
      assert(readCsr(d, 1, 42, 0xc23) == 0x8001)
      val later = issue(d, 0, 43, vset0)
      config(d, later, vl = 11, vt = 0, mt = 0, wrVtype = false, wrMtype = false)
      d.clock.step(); clearCompletions(d)
      assert(readCsr(d, 0, 44, 0xc20) == 11)
      assert(readCsr(d, 0, 45, 0xc21) == 0x10)
      assert(readCsr(d, 0, 46, 0xc23) == 0x4001)
    }
  }

  "multiple config payload lanes retiring together fold each field in program order" in {
    simulate(new Harness) { d =>
      start(d)
      d.io.seq.poke(50.U)
      // Distinct destinations avoid the upstream conservative x0 WAW prefix stop.
      for ((inst, i) <- Seq(vset5, vset0).zipWithIndex) {
        d.io.inst(i).bits.poke(inst.U); d.io.inst(i).valid.poke(true.B)
      }
      d.io.accepted(0).expect(true.B); d.io.accepted(1).expect(true.B)
      val startSlot = d.io.bankIO.enqPtr.peek().litValue
      val ts = (0 until 2).map(i => Ticket(0, 50 + i, startSlot + i,
        d.io.bankIO.enqSlotGen(i).peek().litValue, 0))
      d.clock.step(); d.io.inst.foreach(_.valid.poke(false.B))
      val w = d.io.bankIO.writeDataScalar(1)
      w.valid.poke(true.B); w.bits.addr.poke(5.U); w.bits.data.poke(4.U)
      owner(w.bits.owner, ts(0))
      config(d, ts(0), vl = 4, vt = 0x10, mt = 0x4001, port = 0)
      config(d, ts(1), vl = 8, vt = 0x18, mt = 0, port = 1, wrMtype = false)
      d.io.bankIO.perCtxDeqReady(0).expect(2.U)
      d.clock.step(); clearCompletions(d)
      assert(readCsr(d, 0, 52, 0xc20) == 8)
      assert(readCsr(d, 0, 53, 0xc21) == 0x18)
      assert(readCsr(d, 0, 54, 0xc23) == 0x4001)
    }
  }

  "task reuse rejects stale config flags and writeback without disturbing the peer" in {
    simulate(new Harness) { d =>
      start(d)
      val old = issue(d, 0, 0, fadd)
      floatDone(d, old); d.clock.step(); clearCompletions(d)
      val peer = issue(d, 1, 0, fadd)
      val peerMask = d.io.bankIO.perCtxLiveMask(1).peek().litValue
      flags(d, peer.copy(gen = 5), f = 1)
      d.clock.step(); clearCompletions(d)
      d.io.bankIO.perCtxStaleRejected(1).expect(1.U)
      d.io.advance(0).poke(true.B); d.clock.step(); d.io.advance(0).poke(false.B)
      val fresh = issue(d, 0, 0, fadd)
      assert(fresh.slot == old.slot && fresh.gen == 1 && old.gen == 0)
      flags(d, old, f = 31); floatDone(d, old); config(d, old, vl = 99, vt = 99, mt = 99)
      d.clock.step(); clearCompletions(d)
      d.io.fflags(0).expect(0.U)
      d.io.bankIO.perCtxDeqReady(0).expect(0.U)
      d.io.bankIO.perCtxStaleRejected(0).expect(3.U)
      d.io.bankIO.perCtxLiveMask(1).expect(peerMask.U)
      d.io.bankIO.perCtxStaleRejected(1).expect(1.U)
      flags(d, fresh, f = 4); floatDone(d, fresh); d.clock.step(); clearCompletions(d)
      d.io.fflags(0).expect(4.U)
    }
  }
}
