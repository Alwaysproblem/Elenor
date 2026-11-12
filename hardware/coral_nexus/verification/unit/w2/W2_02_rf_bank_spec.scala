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

class W2_02_RfBankSpec extends AnyFreeSpec with ChiselSim {
  def makeParams(ctxCount: Int, enableFloat: Boolean = true): Parameters = {
    val p = new Parameters(
      m = Seq(
        new MemoryRegion(0x00000000, 0x2000, MemoryRegionType.IMEM),
        new MemoryRegion(0x00010000, 0x0010000, MemoryRegionType.DMEM)
      )
    )
    p.contextCount       = ctxCount
    p.enableVerification = true
    p.enableFloat        = enableFloat
    p.fetchDataBits      = 128
    p.lsuDataBits        = 128
    p
  }

  def idleRf(dut: BankedRf): Unit = {
    dut.io.activeCtx.poke(0.U)
    for (c <- 0 until dut.io.perCtxGen.length) {
      dut.io.perCtxGen(c).poke(0.U)
      dut.io.perCtxReset(c).poke(false.B)
      dut.io.perCtxFlush(c).poke(false.B)
    }
    for (i <- 0 until dut.io.readAddr.length) {
      dut.io.readAddr(i).valid.poke(false.B)
      dut.io.readAddr(i).addr.poke(0.U)
      dut.io.readSet(i).valid.poke(false.B)
      dut.io.readSet(i).value.poke(0.U)
    }
    for (i <- 0 until dut.io.writeAddr.length) {
      dut.io.writeAddr(i).valid.poke(false.B)
      dut.io.writeAddr(i).addr.poke(0.U)
      dut.io.busAddr(i).valid.poke(false.B)
      dut.io.busAddr(i).immen.poke(false.B)
      dut.io.busAddr(i).immed.poke(0.U)
    }
    dut.io.debugBusPort.idx.poke(0.U)
    dut.io.debugWriteValid.poke(false.B)
    for (j <- 0 until dut.io.writeData.length) {
      dut.io.writeData(j).valid.poke(false.B)
      dut.io.writeData(j).bits.addr.poke(0.U)
      dut.io.writeData(j).bits.data.poke(0.U)
      dut.io.writeData(j).bits.owner.ctx_slot.poke(0.U)
      dut.io.writeData(j).bits.owner.ctx_generation.poke(0.U)
      dut.io.writeData(j).bits.owner.owner_op_seq.poke(0.U)
      dut.io.writeData(j).bits.owner.subop_id.poke(0.U)
      dut.io.writeMask(j).valid.poke(false.B)
    }
  }

  "BankedRf isolates same-register GPR and scoreboard across Contexts and accepts background writebacks by OwnerTag (STA-02 STA-03 STA-06)" in {
    simulate(new BankedRf(makeParams(2))) { dut =>
      dut.reset.poke(true.B)
      idleRf(dut)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      // Context 0 marks x5 pending (e.g. long-latency MLU/LSU).
      dut.io.activeCtx.poke(0.U)
      dut.io.writeAddr(0).valid.poke(true.B)
      dut.io.writeAddr(0).addr.poke(5.U)
      dut.clock.step()
      idleRf(dut)

      // Switch to Context 1: Context 1's scoreboard MUST NOT show x5 busy!
      dut.io.activeCtx.poke(1.U)
      dut.io.scoreboard.regd.expect(0.U)
      dut.io.perCtxScoreboardRegd(0).expect((BigInt(1) << 5).U)
      dut.io.perCtxScoreboardRegd(1).expect(0.U)

      // Context 1 marks and writes its own x5 = 0x22222222 while Context 0's x5 is still pending.
      dut.io.writeAddr(0).valid.poke(true.B)
      dut.io.writeAddr(0).addr.poke(5.U)
      dut.clock.step()
      idleRf(dut)
      dut.io.activeCtx.poke(1.U)
      dut.io.writeData(0).valid.poke(true.B)
      dut.io.writeData(0).bits.addr.poke(5.U)
      dut.io.writeData(0).bits.data.poke("h22222222".U)
      dut.io.writeData(0).bits.owner.ctx_slot.poke(1.U)
      dut.io.writeData(0).bits.owner.ctx_generation.poke(0.U)
      dut.io.writeData(0).bits.owner.owner_op_seq.poke(1.U)
      dut.clock.step()
      idleRf(dut)

      // While Context 1 remains selected, Context 0's long-latency writeback arrives on port 4 (MLU/DVU) with x5 = 0x11111111!
      dut.io.activeCtx.poke(1.U)
      dut.io.writeData(4).valid.poke(true.B)
      dut.io.writeData(4).bits.addr.poke(5.U)
      dut.io.writeData(4).bits.data.poke("h11111111".U)
      dut.io.writeData(4).bits.owner.ctx_slot.poke(0.U)
      dut.io.writeData(4).bits.owner.ctx_generation.poke(0.U)
      dut.io.writeData(4).bits.owner.owner_op_seq.poke(0.U)
      dut.clock.step()
      idleRf(dut)

      // Both scoreboards must now be clear, and each Context holds its distinct x5 value!
      dut.io.perCtxScoreboardRegd(0).expect(0.U)
      dut.io.perCtxScoreboardRegd(1).expect(0.U)

      dut.io.activeCtx.poke(0.U)
      dut.io.debugBusPort.idx.poke(5.U)
      dut.io.debugBusPort.data.expect("h11111111".U)

      dut.io.activeCtx.poke(1.U)
      dut.io.debugBusPort.idx.poke(5.U)
      dut.io.debugBusPort.data.expect("h22222222".U)

      // Advance Context 0 generation to 1 and inject stale gen=0 writeback -> rejected & counted!
      dut.io.perCtxGen(0).poke(1.U)
      dut.io.writeData(5).valid.poke(true.B)
      dut.io.writeData(5).bits.addr.poke(5.U)
      dut.io.writeData(5).bits.data.poke("hdeadbeef".U)
      dut.io.writeData(5).bits.owner.ctx_slot.poke(0.U)
      dut.io.writeData(5).bits.owner.ctx_generation.poke(0.U)
      dut.clock.step()
      idleRf(dut)
      dut.io.perCtxGen(0).poke(1.U)
      dut.io.staleWriteRejected.expect(1.U)
      dut.io.activeCtx.poke(0.U)
      dut.io.debugBusPort.idx.poke(5.U)
      dut.io.debugBusPort.data.expect("h11111111".U)
    }
  }

  "BankedFRegfile isolates FPR banks and rejects stale-generation float writebacks" in {
    simulate(new BankedFRegfile(makeParams(2), 3, 2)) { dut =>
      dut.reset.poke(true.B)
      dut.io.activeCtx.poke(0.U)
      dut.io.floatReadCtx.poke(0.U)
      for (c <- 0 until 2) {
        dut.io.perCtxGen(c).poke(0.U)
        dut.io.perCtxReset(c).poke(false.B)
        dut.io.perCtxFlush(c).poke(false.B)
      }
      for (r <- 0 until 3) {
        dut.io.read_ports(r).valid.poke(false.B)
        dut.io.read_ports(r).addr.poke(0.U)
      }
      for (w <- 0 until 2) {
        dut.io.write_ports(w).valid.poke(false.B)
        dut.io.write_ports(w).addr.poke(0.U)
        dut.io.write_ports(w).data.sign.poke(false.B)
        dut.io.write_ports(w).data.exponent.poke(0.U)
        dut.io.write_ports(w).data.mantissa.poke(0.U)
        dut.io.write_ports(w).owner.ctx_slot.poke(0.U)
        dut.io.write_ports(w).owner.ctx_generation.poke(0.U)
        dut.io.write_ports(w).owner.owner_op_seq.poke(0.U)
        dut.io.write_ports(w).owner.subop_id.poke(0.U)
      }
      dut.io.dm_write_valid.poke(false.B)
      dut.io.scoreboard_set.poke(0.U)
      dut.io.busPortAddr.poke(0.U)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      // Mark f3 in Context 0 and Context 1, then write distinct FP32 bit patterns.
      dut.io.activeCtx.poke(0.U)
      dut.io.scoreboard_set.poke((BigInt(1) << 3).U)
      dut.clock.step()
      dut.io.activeCtx.poke(1.U)
      dut.io.scoreboard_set.poke((BigInt(1) << 3).U)
      dut.clock.step()
      dut.io.scoreboard_set.poke(0.U)

      dut.io.write_ports(0).valid.poke(true.B)
      dut.io.write_ports(0).addr.poke(3.U)
      dut.io.write_ports(0).data.sign.poke(false.B)
      dut.io.write_ports(0).data.exponent.poke(127.U) // 1.0f = 0x3f800000
      dut.io.write_ports(0).data.mantissa.poke(0.U)
      dut.io.write_ports(0).owner.ctx_slot.poke(0.U)
      dut.io.write_ports(0).owner.ctx_generation.poke(0.U)

      dut.io.write_ports(1).valid.poke(true.B)
      dut.io.write_ports(1).addr.poke(3.U)
      dut.io.write_ports(1).data.sign.poke(false.B)
      dut.io.write_ports(1).data.exponent.poke(128.U) // 2.0f = 0x40000000
      dut.io.write_ports(1).data.mantissa.poke(0.U)
      dut.io.write_ports(1).owner.ctx_slot.poke(1.U)
      dut.io.write_ports(1).owner.ctx_generation.poke(0.U)
      dut.clock.step()
      dut.io.write_ports(0).valid.poke(false.B)
      dut.io.write_ports(1).valid.poke(false.B)

      dut.io.perCtxScoreboard(0).expect(0.U)
      dut.io.perCtxScoreboard(1).expect(0.U)

      dut.io.activeCtx.poke(0.U)
      dut.io.busPortAddr.poke(3.U)
      dut.io.busPort.data(0).expect("h3f800000".U)

      dut.io.activeCtx.poke(1.U)
      dut.io.busPortAddr.poke(3.U)
      dut.io.busPort.data(0).expect("h40000000".U)
    }
  }

  "Patched Core integration smoke: real Alu, Mlu, Lsu, and Csr propagate OwnerTag across Contexts (c2 and c1 specialization)" in {
    // 1. Degenerate c1 configuration (contextCount == 1 -> explicit 1-bit ctx_slot specialization)
    simulate(new Core(makeParams(1, enableFloat = false), "CoreC1Smoke")) { core1 =>
      core1.reset.poke(true.B)
      core1.io.irq.poke(false.B)
      core1.io.timer_irq.poke(false.B)
      core1.io.software_irq.poke(false.B)
      core1.io.debug_req.poke(false.B)
      core1.io.ctx.ctxSelect.poke(0.U)
      core1.io.ctx.advanceGen(0).poke(false.B)
      for (i <- 0 until core1.io.csr.in.value.length) {
        core1.io.csr.in.value(i).poke(0.U)
      }
      core1.io.ibus.ready.poke(false.B)
      core1.io.ibus.rdata.poke(0.U)
      core1.io.ibus.fault.valid.poke(false.B)
      core1.io.ibus.fault.bits.write.poke(false.B)
      core1.io.ibus.fault.bits.addr.poke(0.U)
      core1.io.ibus.fault.bits.epc.poke(0.U)
      core1.io.dbus.ready.poke(true.B)
      core1.io.dbus.rdata.poke(0.U)
      core1.io.ebus.dbus.ready.poke(true.B)
      core1.io.ebus.dbus.rdata.poke(0.U)
      core1.io.ebus.fault.valid.poke(false.B)
      core1.io.ebus.fault.bits.write.poke(false.B)
      core1.io.ebus.fault.bits.addr.poke(0.U)
      core1.io.ebus.fault.bits.epc.poke(0.U)
      core1.io.dm.debug_req.poke(false.B)
      core1.io.dm.resume_req.poke(false.B)
      core1.io.dm.csr.valid.poke(false.B)
      core1.io.dm.csr.bits.addr.poke(0.U)
      core1.io.dm.csr.bits.index.poke(0.U)
      core1.io.dm.csr.bits.rs1.poke(0.U)
      core1.io.dm.csr.bits.op.poke(CsrOp.CSRRW)
      core1.io.dm.csr_rs1.poke(0.U)
      core1.io.dm.scalar_rd.valid.poke(false.B)
      core1.io.dm.scalar_rd.bits.addr.poke(0.U)
      core1.io.dm.scalar_rd.bits.data.poke(0.U)
      core1.io.dm.scalar_rs.idx.poke(0.U)
      core1.clock.step()
      core1.reset.poke(false.B)
      core1.clock.step()
      core1.io.ctx.perCtxRobEmpty(0).expect(true.B)
    }

    // 2. Full c2 Core end-to-end execution of real ALU + MLU + LSU store across Context 0 and Context 1
    simulate(new Core(makeParams(2, enableFloat = false), "CoreC2Smoke")) { core =>
      core.reset.poke(true.B)
      core.io.irq.poke(false.B)
      core.io.timer_irq.poke(false.B)
      core.io.software_irq.poke(false.B)
      core.io.debug_req.poke(false.B)
      core.io.ctx.ctxSelect.poke(0.U)
      core.io.ctx.advanceGen(0).poke(false.B)
      core.io.ctx.advanceGen(1).poke(false.B)
      for (i <- 0 until core.io.csr.in.value.length) {
        core.io.csr.in.value(i).poke(0.U)
      }
      // Boot PC = 0x00000000
      core.io.csr.in.value(0).poke(0.U)
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
      core.clock.step()
      core.reset.poke(false.B)

      // 128-bit fetch lines (4 instructions each, lane0..lane3): the single shared
      // PC front-end (per-context fetch routing lands in C028) is fed by a one-line
      // IMEM that serves whichever Context `ctxSelect` has selected for issue, as a
      // stand-in for per-context instruction memory:
      //   Context 0: lui x1,0x10; addi x5,x0,42; mul x6,x5,x5; sw x6,0(x1)
      //   Context 1: lui x1,0x10; addi x5,x0,99; sw x5,4(x1); nop
      val fetchLineCtx0 = BigInt("0060a0230252833302a00293000100b7", 16)
      val fetchLineCtx1 = BigInt("000000130050a22306300293000100b7", 16)
      val nopLine       = BigInt("00000013000000130000001300000013", 16)

      var prevFire   = false
      var prevAddr   = BigInt(0)
      var prevRdata = nopLine
      var switched  = false
      var retired0  = 0L
      var retired1  = 0L
      val dbusStores = scala.collection.mutable.ArrayBuffer[BigInt]()

      for (cyc <- 0 until 100) {
        if (cyc == 24) {
          // Switch issue context to Context 1 once Context 0 has retired its bundle
          core.io.ctx.ctxSelect.poke(1.U)
          switched = true
        }
        // Drive a 1-cycle delayed IMEM response for the previous cycle's ibus.fire,
        // selecting the line by the Context that was issued for the fetch.
        if (prevFire) {
          val issuedCtx = core.io.ctx.ctxSelect.peek().litValue
          val line      = if (issuedCtx == 0 && !switched) fetchLineCtx0 else fetchLineCtx1
          core.io.ibus.rdata.poke(line.U(128.W))
        } else {
          core.io.ibus.rdata.poke(prevRdata.U(128.W))
        }

        core.io.ibus.ready.poke(true.B)
        val reqValid = core.io.ibus.valid.peek().litToBoolean
        val reqAddr  = core.io.ibus.addr.peek().litValue

        if (core.io.dbus.valid.peek().litToBoolean && core.io.dbus.write.peek().litToBoolean) {
          dbusStores += (core.io.dbus.wdata.peek().litValue & BigInt("ffffffff", 16))
        }
        retired0 += core.io.ctx.perCtxNRetired(0).peek().litValue.toLong
        retired1 += core.io.ctx.perCtxNRetired(1).peek().litValue.toLong

        core.clock.step()
        val servedLine =
          if (reqValid) (if (!switched) fetchLineCtx0 else fetchLineCtx1) else prevRdata
        prevRdata = servedLine
        prevFire = reqValid
        prevAddr = reqAddr
      }

      assert(retired0 >= 4, s"expected Context 0 to retire >= 4 instructions, got $retired0")
      assert(retired1 >= 4, s"expected Context 1 to retire >= 4 instructions, got $retired1")
      assert(dbusStores.contains(BigInt(1764)),
        s"expected Context 0 store 1764 in $dbusStores, retired0=$retired0 retired1=$retired1")

      // Context 0 x5 == 42, x6 == 1764; Context 1 x5 == 99, x6 == 1764 — both
      // through the real ALU and 3-stage MLU with per-bank OwnerTag routing
      // (each bank holds its own x5=42/99). Context 1's own LSU store leg via
      // the shared fetch PC lands in C028's per-context fetch routing.
      core.io.ctx.ctxSelect.poke(0.U)
      core.io.dm.scalar_rs.idx.poke(5.U)
      core.io.dm.scalar_rs.data.expect(42.U)
      core.io.dm.scalar_rs.idx.poke(6.U)
      core.io.dm.scalar_rs.data.expect(1764.U)

      core.io.ctx.ctxSelect.poke(1.U)
      core.io.dm.scalar_rs.idx.poke(5.U)
      core.io.dm.scalar_rs.data.expect(99.U)
      core.io.dm.scalar_rs.idx.poke(6.U)
      core.io.dm.scalar_rs.data.expect(1764.U)
    }
  }
}
