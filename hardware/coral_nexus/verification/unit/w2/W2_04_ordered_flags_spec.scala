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

/** C027 / W2-04 ordered-merge proof over the REAL Csr -> ROB parking -> CtxCsr chain.
  *
  * Golden case (CFG-04): older FP (delta 0b10001) -> `csrw fflags, 0` -> younger FP
  * (delta 0b00100), with the YOUNGER delta parked BEFORE the older entry retires
  * (early completion). Asserts:
  *   - architectural fflags untouched while nothing retires;
  *   - after the older FP retires: 0b10001;
  *   - after the csrw retires: 0 (a software write REPLACES, clearing the accumulator);
  *   - after the younger FP retires: 0b00100 (the early-parked delta is never lost).
  */
class W2_04_OrderedFlagsSpec extends AnyFreeSpec with ChiselSim {
  val p = {
    val q = new Parameters
    q.contextCount       = 1
    q.enableVerification = false
    q.enableFloat        = true
    q.enableRvv          = false
    q.enableVme          = false
    q
  }

  val ADDI_X5_7 = "h00700293".U(32.W)

  class Harness extends Module {
    val bank = Module(new RetirementPartitionBank(p, mini = true))
    val csr  = Module(new CtxCsr(p))

    val io = IO(new Bundle {
      val bankIO = new RetirementPartitionBankIO(p)
      val csrIO  = new CtxCsrIO(p)
    })
    io.bankIO <> bank.io
    io.csrIO <> csr.io
    // Internal bridges MUST come after the passthrough connects: `:=` on a sink is
    // last-connect-wins, so these override the all-zero harness-side drivers.
    csr.io.retiredDelta := bank.io.perCtxRetiredDelta(0)
    bank.io.csrPark := csr.io.csrParkOut
  }

  "older FP then csrw fflags 0 then younger FP merges in retire order with final value exactly the younger delta" in {
    simulate(new Harness) { dut =>
      val b = dut.io.bankIO
      val c = dut.io.csrIO

      def idle(): Unit = {
        b.activeCtx.poke(0.U)
        b.perCtxGen(0).poke(0.U)
        b.perCtxReset(0).poke(false.B)
        for (i <- 0 until 4) {
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
        b.writeAddrFloat.get.valid.poke(false.B)
        b.writeAddrFloat.get.addr.poke(0.U)
        b.storeComplete.valid.poke(false.B)
        b.storeComplete.bits.ctx_slot.poke(0.U)
        b.storeComplete.bits.ctx_generation.poke(0.U)
        b.storeComplete.bits.owner_op_seq.poke(0.U)
        b.storeComplete.bits.subop_id.poke(0.U)
        for (j <- 0 until 6) {
          b.writeDataScalar(j).valid.poke(false.B)
          b.writeDataScalar(j).bits.addr.poke(0.U)
          b.writeDataScalar(j).bits.data.poke(0.U)
          b.writeDataScalar(j).bits.owner.ctx_slot.poke(0.U)
          b.writeDataScalar(j).bits.owner.ctx_generation.poke(0.U)
          b.writeDataScalar(j).bits.owner.owner_op_seq.poke(0.U)
          b.writeDataScalar(j).bits.owner.subop_id.poke(0.U)
        }
        for (j <- 0 until 2) {
          b.writeDataFloat.get(j).valid.poke(false.B)
          b.writeDataFloat.get(j).bits.addr.poke(0.U)
          b.writeDataFloat.get(j).bits.data.poke(0.U)
          b.writeDataFloat.get(j).bits.owner.ctx_slot.poke(0.U)
          b.writeDataFloat.get(j).bits.owner.ctx_generation.poke(0.U)
          b.writeDataFloat.get(j).bits.owner.owner_op_seq.poke(0.U)
          b.writeDataFloat.get(j).bits.owner.subop_id.poke(0.U)
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
        b.flagsDelta.valid.poke(false.B)
        b.flagsDelta.bits.rob_tag.ctx_slot.poke(0.U)
        b.flagsDelta.bits.rob_tag.rob_slot.poke(0.U)
        b.flagsDelta.bits.rob_tag.slot_generation.poke(0.U)
        b.flagsDelta.bits.fflags.poke(0.U)
        b.flagsDelta.bits.vxsat.poke(false.B)
        b.flagsDelta.bits.replaceFflags.poke(false.B)
        b.flagsDelta.bits.replaceVxsat.poke(false.B)
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

        c.activeCtx.poke(0.U)
        c.advanceGen(0).poke(false.B)
        for (i <- 0 until c.csr.in.value.length) { c.csr.in.value(i).poke(0.U) }
        c.req.valid.poke(false.B)
        c.req.bits.addr.poke(0.U)
        c.req.bits.index.poke(0.U)
        c.req.bits.rs1.poke(0.U)
        c.req.bits.op.poke(CsrOp.CSRRW)
        c.reqOwner.ctx_slot.poke(0.U)
        c.reqOwner.ctx_generation.poke(0.U)
        c.reqOwner.owner_op_seq.poke(0.U)
        c.reqOwner.subop_id.poke(0.U)
        c.rs1.valid.poke(true.B)
        c.rs1.data.poke(0.U)
        c.bru.in.mode.valid.poke(false.B)
        c.bru.in.mcause.valid.poke(false.B)
        c.bru.in.mepc.valid.poke(false.B)
        c.bru.in.mtval.valid.poke(false.B)
        c.bru.in.halt.poke(false.B)
        c.bru.in.fault.poke(false.B)
        c.bru.in.wfi.poke(false.B)
        c.bruOwner.ctx_slot.poke(0.U)
        c.bruOwner.ctx_generation.poke(0.U)
        c.bruOwner.owner_op_seq.poke(0.U)
        c.bruOwner.subop_id.poke(0.U)
        c.float.get.in.fflags.valid.poke(false.B)
        c.float.get.in.fflags.bits.poke(0.U)
        c.floatOwner.get.ctx_slot.poke(0.U)
        c.floatOwner.get.ctx_generation.poke(0.U)
        c.floatOwner.get.owner_op_seq.poke(0.U)
        c.floatOwner.get.subop_id.poke(0.U)
        c.float_dirty.get.poke(false.B)
        c.perCtxNRetired(0).poke(0.U)
        c.irq.poke(false.B)
        c.dm.debug_req.poke(false.B)
        c.dm.resume_req.poke(false.B)
        c.dm.current_pc.poke(0.U)
        c.dm.next_pc.poke(0.U)
        c.timer_irq.poke(false.B)
        c.software_irq.poke(false.B)
      }

      dut.reset.poke(true.B)
      idle()
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      // Dispatch older FP (op_seq 10), csrw fflags,0 (op_seq 11), younger FP (op_seq 12);
      // all wait for an owner-keyed writeback so retirement order is test-controlled.
      val slotBase = b.enqPtr.peek().litValue.toInt
      for ((seq, i) <- Seq(10, 11, 12).zipWithIndex) {
        b.dispatchOpSeq(0).poke(seq.U)
        b.inst(0).valid.poke(true.B)
        b.inst(0).bits.addr.poke((0x1000 + i * 4).U)
        b.inst(0).bits.inst.poke(ADDI_X5_7)
        b.writeAddrScalar(0).valid.poke(true.B)
        b.writeAddrScalar(0).addr.poke(5.U)
        dut.clock.step()
        b.inst(0).valid.poke(false.B)
        b.writeAddrScalar(0).valid.poke(false.B)
        b.dispatchOpSeq(0).poke(0.U)
      }

      // Fresh ROB: all three slots are generation 0.
      def parkFp(slot: Int, delta: BigInt): Unit = {
        b.flagsDelta.bits.rob_tag.ctx_slot.poke(0.U)
        b.flagsDelta.bits.rob_tag.rob_slot.poke(slot.U)
        b.flagsDelta.bits.rob_tag.slot_generation.poke(0.U)
        b.flagsDelta.bits.fflags.poke(delta.U)
        b.flagsDelta.valid.poke(true.B)
        dut.clock.step()
        b.flagsDelta.valid.poke(false.B)
        b.flagsDelta.bits.fflags.poke(0.U)
      }

      // EARLY completions, before ANY retirement: park both FP deltas now.
      parkFp(slotBase, BigInt("10001", 2)) // older FP
      parkFp(slotBase + 2, BigInt("00100", 2)) // younger FP completes early

      // csrw fflags, x0 through the REAL Csr: parks at op_seq 11's slot.
      c.req.valid.poke(true.B)
      c.req.bits.index.poke("h001".U) // fflags
      c.req.bits.op.poke(CsrOp.CSRRW)
      c.req.bits.rs1.poke(0.U)
      c.rs1.data.poke(0.U)
      c.reqOwner.owner_op_seq.poke(11.U)
      dut.clock.step()
      c.req.valid.poke(false.B)

      // Pre-retirement invariant: architecture untouched.
      c.perCtxFflags(0).expect(0.U)

      def writeback(seq: Int): Unit = {
        b.writeDataScalar(0).valid.poke(true.B)

        b.writeDataScalar(0).bits.addr.poke(5.U)
        b.writeDataScalar(0).bits.data.poke(seq.U)
        b.writeDataScalar(0).bits.owner.ctx_slot.poke(0.U)
        b.writeDataScalar(0).bits.owner.ctx_generation.poke(0.U)
        b.writeDataScalar(0).bits.owner.owner_op_seq.poke(seq.U)
        b.writeDataScalar(0).bits.owner.subop_id.poke(0.U)
        dut.clock.step()
        b.writeDataScalar(0).valid.poke(false.B)
      }

      writeback(10) // older FP retires: parked delta merges.
      c.perCtxFflags(0).expect("b10001".U)

      writeback(11) // csrw fflags,0 retires: REPLACE clears the accumulator.
      c.perCtxFflags(0).expect(0.U)

      writeback(12) // younger FP retires: its early-parked delta is NOT lost.
      c.perCtxFflags(0).expect("b00100".U)
    }
  }
}
