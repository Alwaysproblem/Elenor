// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
package coralnpu

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import coralnpu.rvv.{RvvCoreShim, RvvCsrIO}
import org.scalatest.freespec.AnyFreeSpec

class W2_04_CoreControlSpec extends AnyFreeSpec with ChiselSim {
  def parameters(n: Int): Parameters = {
    val p = new Parameters(m = Seq(
      new MemoryRegion(0x00000000, 0x2000, MemoryRegionType.IMEM),
      new MemoryRegion(0x00010000, 0x10000, MemoryRegionType.DMEM)
    ))
    p.contextCount = n
    p.enableVerification = true
    p.enableFloat = false
    p.enableRvv = false
    p.fetchDataBits = 128
    p.lsuDataBits = 128
    p
  }

  class NativeControlHarness(p: Parameters) extends Module {
    val native = Module(new RvvCoreShim(p))
    val io = IO(new Bundle {
      val csr = new RvvCsrIO(p)
      val selected = Input(UInt(p.ctxBits.W))
      val generation = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))
      val resetContext = Input(Vec(p.contextCount, Bool()))
      val clearVstart = Input(Vec(p.contextCount, Bool()))
      val configState = Output(chiselTypeOf(native.io.configState))
      val stale = Output(UInt(16.W))
    })
    io.csr <> native.io.csr
    native.io.ctxSelect := io.selected
    native.io.ctxGeneration := io.generation
    native.io.ctxReset := io.resetContext
    native.io.nativeControlAccept := true.B
    native.io.inst.foreach { port =>
      port.valid := false.B
      port.bits := 0.U.asTypeOf(port.bits)
    }
    native.io.rs.foreach(port => port := 0.U.asTypeOf(port))
    native.io.frs.foreach(_ := 0.U)
    native.io.lsu2rvv.foreach { port =>
      port.valid := false.B
      port.bits := 0.U.asTypeOf(port.bits)
    }
    native.io.rvv2lsu.foreach(_.ready := true.B)
    native.io.async_rd.ready := true.B
    native.io.async_frd.ready := true.B
    native.io.flush := false.B
    native.io.clear_vstart := io.clearVstart
    io.stale := native.io.staleControlRejected
    io.configState := native.io.configState
  }

  "Concurrent native fault and peer CSR updates retain separate owners across reuse" in {
    val p = parameters(2)
    p.enableRvv = true
    simulate(new NativeControlHarness(p)) { d =>
      def owner(port: OwnerTag, ctx: Int, gen: Int): Unit = {
        port.ctx_slot.poke(ctx.U)
        port.ctx_generation.poke(gen.U)
        port.owner_op_seq.poke(7.U)
        port.subop_id.poke(0.U)
      }
      def expectVstart(ctx: Int, value: Int): Unit = {
        d.io.selected.poke(ctx.U)
        d.io.csr.vstart(ctx).expect(value.U)
        d.io.configState.bits.vstart.expect(value.U)
      }
      d.io.selected.poke(1.U)
      d.io.generation(0).poke(1.U)
      d.io.generation(1).poke(2.U)
      d.io.resetContext.foreach(_.poke(false.B))
      d.io.clearVstart.foreach(_.poke(false.B))
      d.io.csr.frm.poke(0.U)
      d.io.csr.vxsat.poke(false.B)
      d.io.csr.vstart_write.valid.poke(false.B)
      d.io.csr.vstart_write.bits.poke(0.U)
      d.io.csr.vxrm_write.valid.poke(false.B)
      d.io.csr.vxrm_write.bits.poke(0.U)
      d.io.csr.faultVstart.valid.poke(false.B)
      d.io.csr.faultVstart.bits.poke(0.U)
      owner(d.io.csr.writeOwner, 1, 2)
      owner(d.io.csr.faultOwner, 0, 1)
      d.reset.poke(true.B); d.clock.step(); d.reset.poke(false.B)

      d.io.csr.vstart_write.valid.poke(true.B)
      d.io.csr.vstart_write.bits.poke(5.U)
      d.io.csr.faultVstart.valid.poke(true.B)
      d.io.csr.faultVstart.bits.poke(3.U)
      d.clock.step()
      d.io.csr.vstart(0).expect(3.U)
      d.io.csr.vstart(1).expect(5.U)
      d.io.csr.vstart_write.valid.poke(false.B)
      d.io.csr.vxrm_write.valid.poke(true.B)
      d.io.csr.vxrm_write.bits.poke(2.U)
      d.io.csr.faultVstart.bits.poke(4.U)
      d.io.clearVstart(0).poke(true.B)
      d.clock.step()
      d.io.csr.vstart(0).expect(4.U)
      d.io.csr.vstart(1).expect(5.U)
      d.io.csr.vxrm(0).expect(0.U)
      d.io.csr.vxrm(1).expect(2.U)
      d.io.stale.expect(0.U)

      // The fault wins over same-Context clear. Its frontend copy must also
      // survive the following beat after all update inputs are withdrawn.
      d.io.csr.faultVstart.valid.poke(false.B)
      d.io.csr.vxrm_write.valid.poke(false.B)
      d.io.clearVstart(0).poke(false.B)
      d.clock.step()
      expectVstart(0, 4)
      expectVstart(1, 5)

      // A live CSR VSTART write wins over clear by the same arbitration rule.
      d.io.csr.vstart_write.valid.poke(true.B)
      d.io.csr.vstart_write.bits.poke(6.U)
      d.io.clearVstart(1).poke(true.B)
      d.clock.step()
      d.io.csr.vstart_write.valid.poke(false.B)
      d.io.clearVstart(1).poke(false.B)
      d.clock.step()
      expectVstart(0, 4)
      expectVstart(1, 6)

      // The architectural bank clears at retirement, before the registered
      // notification reaches the native frontend. The peer remains untouched.
      d.io.csr.faultVstart.valid.poke(false.B)
      d.io.csr.vxrm_write.valid.poke(false.B)
      d.io.clearVstart(0).poke(true.B)
      d.clock.step()
      d.io.csr.vstart(0).expect(0.U)
      d.io.csr.vstart(1).expect(6.U)
      d.io.clearVstart(0).poke(false.B)
      d.clock.step()
      expectVstart(0, 0)
      expectVstart(1, 6)
      d.io.csr.faultVstart.valid.poke(true.B)
      d.clock.step()
      d.io.csr.vstart(0).expect(4.U)
      d.io.csr.vxrm_write.valid.poke(true.B)

      owner(d.io.csr.faultOwner, 0, 0)
      d.io.csr.faultVstart.bits.poke(7.U)
      d.io.csr.vxrm_write.bits.poke(1.U)
      d.clock.step()
      expectVstart(0, 4)
      expectVstart(1, 6)
      d.io.csr.vxrm(1).expect(1.U)
      d.io.stale.expect(1.U)
      d.io.csr.faultVstart.valid.poke(false.B)
      d.io.csr.vxrm_write.valid.poke(false.B)
      d.io.resetContext(0).poke(true.B)
      d.io.generation(0).poke(2.U)
      d.clock.step()
      expectVstart(0, 0)
      expectVstart(1, 6)
      d.io.csr.vxrm(1).expect(1.U)
      d.io.stale.expect(0.U)
    }
  }

  def idle(core: Core): Unit = {
    core.io.irq.poke(false.B)
    core.io.timer_irq.poke(false.B)
    core.io.software_irq.poke(false.B)
    core.io.debug_req.poke(false.B)
    core.io.ctx.ctxSelect.poke(0.U)
    for (c <- 0 until core.io.ctx.advanceGen.length) {
      core.io.ctx.advanceGen(c).poke(false.B)
      core.io.ctx.perCtxPark(c).poke(false.B)
      core.io.ctx.parkGen(c).poke(0.U)
      core.io.ctx.cancelValid(c).poke(false.B)
      core.io.ctx.cancelGen(c).poke(0.U)
      core.io.ctx.pendingLaunch(c).poke(false.B)
      core.io.ctx.waitBusy(c).poke(false.B)
      core.io.ctx.regionBusy(c).poke(false.B)
    }
    core.io.csr.in.value.foreach(_.poke(0.U))
    core.io.ibus.ready.poke(false.B)
    core.io.ibus.rdata.poke(0.U)
    core.io.ibus.fault.valid.poke(false.B)
    core.io.ibus.fault.bits.write.poke(false.B)
    core.io.ibus.fault.bits.addr.poke(0.U)
    core.io.ibus.fault.bits.epc.poke(0.U)
    core.io.dbus.ready.poke(true.B)
    core.io.dbus.rdata.poke(0.U)
    core.io.ebus.dbus.ready.poke(true.B)
    core.io.ebus.dbus.rdata.poke(0.U)
    core.io.ebus.fault.valid.poke(false.B)
    core.io.ebus.fault.bits.write.poke(false.B)
    core.io.ebus.fault.bits.addr.poke(0.U)
    core.io.ebus.fault.bits.epc.poke(0.U)
    core.io.dm.debug_req.poke(false.B)
    core.io.dm.resume_req.poke(false.B)
    core.io.dm.csr.valid.poke(false.B)
    core.io.dm.csr.bits.addr.poke(0.U)
    core.io.dm.csr.bits.index.poke(0.U)
    core.io.dm.csr.bits.rs1.poke(0.U)
    core.io.dm.csr.bits.op.poke(CsrOp.CSRRW)
    core.io.dm.csr_rs1.poke(0.U)
    core.io.dm.scalar_rd.valid.poke(false.B)
    core.io.dm.scalar_rd.bits.addr.poke(0.U)
    core.io.dm.scalar_rd.bits.data.poke(0.U)
    core.io.dm.scalar_rs.idx.poke(5.U)
    core.io.dm.float_rd.foreach { port =>
      port.valid.poke(false.B)
      port.addr.poke(0.U)
      port.data.sign.poke(false.B)
      port.data.exponent.poke(0.U)
      port.data.mantissa.poke(0.U)
    }
    core.io.dm.float_rs.foreach { port =>
      port.valid.poke(false.B)
      port.addr.poke(0.U)
    }
  }

  /** One-cycle instruction bus response, with accepted data-store observation. */
  class InstructionMemory(core: Core, lines: Vector[BigInt]) {
    val nops = BigInt("00000013000000130000001300000013", 16)
    private var response = nops
    var storeCount = 0
    var lastStoreAddress = BigInt(0)

    def step(): Unit = {
      core.io.ibus.rdata.poke(response.U(128.W))
      val next = if (core.io.ibus.valid.peek().litToBoolean) {
        val line = (core.io.ibus.addr.peek().litValue >> 4).toInt
        if (line < lines.length) lines(line) else nops
      } else response
      if (core.io.dbus.valid.peek().litToBoolean &&
          core.io.dbus.write.peek().litToBoolean &&
          core.io.dbus.ready.peek().litToBoolean) {
        storeCount += 1
        lastStoreAddress = core.io.dbus.addr.peek().litValue
      }
      core.clock.step()
      response = next
    }
  }

  "Real WFI parks without fencing an accepted backpressured store and the store still drains" in {
    simulate(new Core(parameters(2), "C027WfiDrain")) { core =>
      core.reset.poke(true.B)
      idle(core)
      core.clock.step()
      core.reset.poke(false.B)
      core.io.ctx.perCtxPark(1).poke(true.B)
      core.io.ibus.ready.poke(true.B)
      core.io.dbus.ready.poke(false.B)
      // lui x1,0x10; sw x0,0(x1); wfi; nop.
      val memory = new InstructionMemory(core,
        Vector(BigInt("00000013105000730000a023000100b7", 16)))
      var cycles = 0
      while (!core.io.ctx.perCtxParked(0).peek().litToBoolean && cycles < 80) {
        memory.step()
        cycles += 1
      }
      assert(cycles < 80, "WFI incorrectly waited for global LSU idle")
      core.io.ctx.perCtxRobEmpty(0).expect(false.B)
      core.io.ctx.tileQuiescent.expect(false.B)
      assert(memory.storeCount == 0)
      core.io.dbus.ready.poke(true.B)
      while (!core.io.ctx.tileQuiescent.peek().litToBoolean && cycles < 120) {
        memory.step()
        cycles += 1
      }
      core.io.ctx.tileQuiescent.expect(true.B)
      assert(memory.storeCount == 1, "park lost or duplicated the accepted store")
      assert(memory.lastStoreAddress == BigInt(0x10000))
    }
  }

  "Real FloatCore preserves ordered flags and dynamic rounding snapshots without RVV" in {
    val p = parameters(2)
    p.enableFloat = true
    simulate(new Core(p, "C027ScalarFloat")) { core =>
      core.reset.poke(true.B)
      idle(core)
      core.clock.step()
      core.reset.poke(false.B)
      core.io.ctx.perCtxPark(1).poke(true.B)
      core.io.ibus.ready.poke(true.B)
      // addi x1,0,1; fcvt.s.w f1,x1; fmv.w.x f2,x0; fdiv.s f3,f1,f2;
      // csrr x6,fflags; lui x2,0x33800; fmv.w.x f2,x2; csrrwi x0,frm,3;
      // fadd.s f4,f1,f2,dyn; csrrwi x0,frm,0; csrr x7,fflags; wfi.
      // 1.0 / +0.0 = +inf (DZ=8). RUP(1.0 + 2^-24) = 0x3f800001 (NX=1);
      // the younger RNE write must not alter the FADD's dispatch-time snapshot.
      val lines = Vector(
        BigInt("182081d3f0000153d00080d300100093", 16),
        BigInt("0021d073f00101533380013700102373", 16),
        BigInt("10500073001023f3002050730020f253", 16)
      )
      val memory = new InstructionMemory(core, lines)
      var cycles = 0
      while (!core.io.ctx.perCtxParked(0).peek().litToBoolean && cycles < 400) {
        memory.step()
        cycles += 1
      }
      assert(cycles < 400, "floating producer or dependent CSR failed to retire")
      core.io.ctx.tileIssueFreeze.expect(false.B)
      core.io.dm.scalar_rs.idx.poke(6.U)
      core.io.dm.scalar_rs.data.expect(8.U)
      core.io.dm.scalar_rs.idx.poke(7.U)
      core.io.dm.scalar_rs.data.expect(9.U)
      core.io.dm.float_rs.get.valid.poke(true.B)
      core.io.dm.float_rs.get.addr.poke(3.U)
      core.io.dm.float_rs.get.data.sign.expect(false.B)
      core.io.dm.float_rs.get.data.exponent.expect(255.U)
      core.io.dm.float_rs.get.data.mantissa.expect(0.U)
      core.io.dm.float_rs.get.addr.poke(4.U)
      core.io.dm.float_rs.get.data.sign.expect(false.B)
      core.io.dm.float_rs.get.data.exponent.expect(127.U)
      core.io.dm.float_rs.get.data.mantissa.expect(1.U)
      core.io.ctx.ctxSelect.poke(1.U)
      core.io.dm.scalar_rs.data.expect(0.U)
      core.io.dm.float_rs.get.data.exponent.expect(0.U)
      core.io.dm.float_rs.get.data.mantissa.expect(0.U)
    }
  }

  "Real native RVV returns commit configuration saturation and floating flags through Core" in {
    val p = parameters(2)
    p.enableFloat = true
    p.enableRvv = true
    p.enableVme = true
    simulate(new Core(p, "C027NativeReturns")) { core =>
      core.reset.poke(true.B)
      idle(core)
      core.clock.step()
      core.reset.poke(false.B)
      core.io.ctx.perCtxPark(1).poke(true.B)
      core.io.ibus.ready.poke(true.B)
      // e8: saturate 127+1, observe VXSAT, clear it, and observe the ordered clear.
      // e32: divide vector 1.0 by 0.0, then read architectural DZ.
      // Finally a VL=0/VTYPE=0 configuration must commit zeros, not mean "no update".
      val words = Vector(
        "c00272d7", // vsetivli x5,4,e8,m1
        "07f00093", // addi x1,x0,127
        "5e00c0d7", // vmv.v.x v1,x1
        "8610b157", // vsadd.vi v2,v1,1
        "00902373", // csrr x6,vxsat
        "00905073", // csrrwi x0,vxsat,0
        "009023f3", // csrr x7,vxsat
        "c1027657", // vsetivli x12,4,e32,m1
        "00100093", // addi x1,x0,1
        "d00080d3", // fcvt.s.w f1,x1
        "f0000053", // fmv.w.x f0,x0
        "5e00d0d7", // vfmv.v.f v1,f1
        "5e005057", // vfmv.v.f v0,f0
        "82101157", // vfdiv.vv v2,v1,v0
        "00102473", // csrr x8,fflags
        "c0007057", // vsetivli x0,0,e8,m1
        "c20024f3", // csrr x9,vl
        "c2102573", // csrr x10,vtype
        "c23025f3", // csrr x11,mtype
        "10500073"  // wfi
      ).map(BigInt(_, 16))
      val lines = words.grouped(4).map { group =>
        group.zipWithIndex.foldLeft(BigInt(0)) { case (line, (word, lane)) =>
          line | (word << (32 * lane))
        }
      }.toVector
      val memory = new InstructionMemory(core, lines)
      var cycles = 0
      while (!core.io.ctx.perCtxParked(0).peek().litToBoolean && cycles < 2000) {
        memory.step()
        cycles += 1
      }
      assert(cycles < 2000, "native producers or dependent architectural CSR failed to retire")
      core.io.ctx.tileIssueFreeze.expect(false.B)
      for ((reg, value) <- Seq(5 -> 4, 6 -> 1, 7 -> 0, 8 -> 8, 9 -> 0, 10 -> 0, 11 -> 0, 12 -> 4)) {
        core.io.dm.scalar_rs.idx.poke(reg.U)
        core.io.dm.scalar_rs.data.expect(value.U)
      }
      core.io.ctx.ctxSelect.poke(1.U)
      for (reg <- Seq(5, 6, 7, 8, 9, 10, 11, 12)) {
        core.io.dm.scalar_rs.idx.poke(reg.U)
        core.io.dm.scalar_rs.data.expect(0.U)
      }
    }
  }

  "A real decode fault freezes issue for both Contexts until Group reset" in {
    simulate(new Core(parameters(2), "C027FailStop")) { core =>
      core.reset.poke(true.B)
      idle(core)
      core.clock.step()
      core.reset.poke(false.B)
      core.io.ibus.ready.poke(true.B)
      core.io.ibus.rdata.poke(BigInt("000000130000001300000013ffffffff", 16).U(128.W))
      var cycles = 0
      while (!core.io.ctx.tileIssueFreeze.peek().litToBoolean && cycles < 40) {
        core.clock.step()
        cycles += 1
      }
      assert(cycles < 40, "real FaultManager path did not latch fail-stop")
      core.io.ctx.ctxSelect.poke(1.U)
      core.io.ibus.rdata.poke(BigInt("00000013000000130000001302a00293", 16).U(128.W))
      for (_ <- 0 until 8) {
        core.io.ctx.tileIssueFreeze.expect(true.B)
        core.io.debug.get.dispatch.foreach(_.instFire.expect(false.B))
        core.clock.step()
      }
      core.io.ctx.perCtxGeneration(1).expect(0.U)
      core.io.dm.scalar_rs.data.expect(0.U)
    }
  }

  for (count <- Seq(1, 2)) {
    s"Core c$count WFI parks only its owner and stale control cannot cancel a reused slot" in {
      simulate(new Core(parameters(count), s"C027Control$count")) { core =>
        core.reset.poke(true.B)
        idle(core)
        core.clock.step()
        core.reset.poke(false.B)
        core.io.ctx.perCtxPark(0).poke(true.B)
        // addi x5,x0,42; wfi; nop; nop. A real instruction stream, not injected retirement.
        val line = BigInt("00000013000000131050007302a00293", 16)
        core.io.ibus.rdata.poke(line.U(128.W))
        core.io.ibus.ready.poke(true.B)
        for (_ <- 0 until 8) {
          core.io.debug.get.dispatch.foreach(_.instFire.expect(false.B))
          core.clock.step()
        }
        core.io.ctx.perCtxPark(0).poke(false.B)
        var cycles = 0
        while (!core.io.ctx.perCtxParked(0).peek().litToBoolean && cycles < 80) {
          core.clock.step()
          cycles += 1
        }
        assert(cycles < 80, "owner never reached its real WFI")
        core.io.dm.scalar_rs.data.expect(42.U)
        if (count == 2) {
          core.io.ctx.perCtxParked(1).expect(false.B)
          core.io.ctx.tileQuiescent.expect(false.B)
          core.io.ctx.perCtxPark(1).poke(true.B)
        }
        core.clock.step(4)
        for (_ <- 0 until 5) {
          core.io.debug.get.dispatch.foreach(_.instFire.expect(false.B))
          core.clock.step()
        }
        core.io.ctx.tileQuiescent.expect(true.B)
        core.io.ctx.pendingLaunch(count - 1).poke(true.B)
        core.io.ctx.tileQuiescent.expect(false.B)
        core.io.ctx.perCtxBackendDrained(count - 1).expect(false.B)
        core.io.ctx.pendingLaunch(count - 1).poke(false.B)

        // Cancel is Context-local, latched immediately, and does not erase architectural RF.
        core.io.ctx.cancelValid(0).poke(true.B)
        core.io.ctx.cancelGen(0).poke(0.U)
        core.io.ctx.perCtxCancelled(0).expect(true.B)
        core.clock.step()
        core.io.ctx.cancelValid(0).poke(false.B)
        core.io.dm.scalar_rs.data.expect(42.U)
        if (count == 2) core.io.ctx.perCtxCancelled(1).expect(false.B)

        // The coordinator's safe-reuse pulse clears only bank A and increments its generation.
        core.io.ctx.advanceGen(0).poke(true.B)
        core.clock.step()
        core.io.ctx.advanceGen(0).poke(false.B)
        core.io.ctx.perCtxPark(0).poke(true.B)
        core.io.ctx.parkGen(0).poke(1.U)
        core.io.ctx.perCtxGeneration(0).expect(1.U)
        core.io.ctx.perCtxCancelled(0).expect(false.B)
        core.io.dm.scalar_rs.data.expect(0.U)
        if (count == 2) core.io.ctx.perCtxGeneration(1).expect(0.U)
        val before = core.io.ctx.staleControlRejected.peek().litValue
        core.io.ctx.cancelValid(0).poke(true.B)
        core.io.ctx.cancelGen(0).poke(0.U)
        core.clock.step()
        core.io.ctx.cancelValid(0).poke(false.B)
        core.io.ctx.perCtxCancelled(0).expect(false.B)
        core.io.ctx.staleControlRejected.expect((before + 1).U)
        core.io.ctx.perCtxParked(0).expect(true.B)
      }
    }
  }
}
