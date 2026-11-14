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

class W2_02_RobOwnerSpec extends AnyFreeSpec with ChiselSim {
  def makeParams(ctxCount: Int): Parameters = {
    val p = new Parameters
    p.contextCount       = ctxCount
    p.enableVerification = false
    p.enableFloat        = true
    p.enableRvv          = true
    p.enableVme          = false
    p
  }

  // RISC-V encodings:
  // sw x5, 0(x1)  -> imm=0, rs2=5, rs1=1, funct3=010, opcode=0100011 => 0x0050A023
  // addi x5, x0, 7 -> 0x00700293
  // vle32.v v1, (x1) -> 0x0200E087
  val SW_X5_0_X1  = "h0050a023".U(32.W)
  val ADDI_X5_7   = "h00700293".U(32.W)
  val VLE32_V1_X1 = "h0200e087".U(32.W)

  def idleInputs(dut: RetirementPartitionBank): Unit = {
    dut.io.activeCtx.poke(0.U)
    for (c <- 0 until dut.io.perCtxGen.length) {
      dut.io.perCtxGen(c).poke(0.U)
      dut.io.perCtxReset(c).poke(false.B)
    }
    for (i <- 0 until dut.io.inst.length) {
      dut.io.dispatchOpSeq(i).poke(0.U)
      dut.io.inst(i).valid.poke(false.B)
      dut.io.inst(i).ready.poke(true.B)
      dut.io.inst(i).bits.addr.poke(0.U)
      dut.io.inst(i).bits.inst.poke(0.U)
      dut.io.inst(i).bits.brchFwd.poke(false.B)
      dut.io.targets(i).poke(0.U)
      dut.io.jalrTargets(i).poke(0.U)
      dut.io.jump(i).poke(false.B)
      dut.io.branch(i).poke(false.B)
      dut.io.writeAddrScalar(i).valid.poke(false.B)
      dut.io.writeAddrScalar(i).addr.poke(0.U)
      dut.io.isVector.get(i).poke(false.B)
      dut.io.writeAddrVector.get(i).valid.poke(false.B)
      dut.io.writeAddrVector.get(i).addr.poke(0.U)
    }
    dut.io.writeAddrFloat.get.valid.poke(false.B)
    dut.io.writeAddrFloat.get.addr.poke(0.U)
    dut.io.storeComplete.valid.poke(false.B)
    dut.io.storeComplete.bits.ctx_slot.poke(0.U)
    dut.io.storeComplete.bits.ctx_generation.poke(0.U)
    dut.io.storeComplete.bits.owner_op_seq.poke(0.U)
    dut.io.storeComplete.bits.subop_id.poke(0.U)
    for (j <- 0 until dut.io.writeDataScalar.length) {
      dut.io.writeDataScalar(j).valid.poke(false.B)
      dut.io.writeDataScalar(j).bits.addr.poke(0.U)
      dut.io.writeDataScalar(j).bits.data.poke(0.U)
      dut.io.writeDataScalar(j).bits.owner.ctx_slot.poke(0.U)
      dut.io.writeDataScalar(j).bits.owner.ctx_generation.poke(0.U)
      dut.io.writeDataScalar(j).bits.owner.owner_op_seq.poke(0.U)
      dut.io.writeDataScalar(j).bits.owner.subop_id.poke(0.U)
    }
    for (j <- 0 until dut.io.writeDataFloat.get.length) {
      dut.io.writeDataFloat.get(j).valid.poke(false.B)
      dut.io.writeDataFloat.get(j).bits.addr.poke(0.U)
      dut.io.writeDataFloat.get(j).bits.data.poke(0.U)
      dut.io.writeDataFloat.get(j).bits.owner.ctx_slot.poke(0.U)
      dut.io.writeDataFloat.get(j).bits.owner.ctx_generation.poke(0.U)
      dut.io.writeDataFloat.get(j).bits.owner.owner_op_seq.poke(0.U)
      dut.io.writeDataFloat.get(j).bits.owner.subop_id.poke(0.U)
    }
    for (j <- 0 until dut.io.writeDataVector.get.length) {
      dut.io.writeDataVector.get(j).valid.poke(false.B)
      dut.io.writeDataVector.get(j).bits.w_valid.poke(false.B)
      dut.io.writeDataVector.get(j).bits.addr.poke(0.U)
      dut.io.writeDataVector.get(j).bits.last_uop_valid.poke(false.B)
      dut.io.writeDataVector.get(j).bits.rob_tag.ctx_slot.poke(0.U)
      dut.io.writeDataVector.get(j).bits.rob_tag.rob_slot.poke(0.U)
      dut.io.writeDataVector.get(j).bits.rob_tag.slot_generation.poke(0.U)
    }
    dut.io.fault.valid.poke(false.B)
    dut.io.fault.bits.mepc.poke(0.U)
    dut.io.fault.bits.mtval.poke(0.U)
    dut.io.fault.bits.mcause.poke(0.U)
    dut.io.fault.bits.decode.poke(false.B)
    dut.io.fault.bits.is_rvv.get.poke(false.B)
    dut.io.fault.bits.owner.ctx_slot.poke(0.U)
    dut.io.fault.bits.owner.ctx_generation.poke(0.U)
    dut.io.fault.bits.owner.owner_op_seq.poke(0.U)
    dut.io.fault.bits.owner.subop_id.poke(0.U)
    dut.io.fault.bits.rob_tag.get.ctx_slot.poke(0.U)
    dut.io.fault.bits.rob_tag.get.rob_slot.poke(0.U)
    dut.io.fault.bits.rob_tag.get.slot_generation.poke(0.U)
    for (j <- 0 until dut.io.flagsDelta.length) {
      dut.io.flagsDelta(j).valid.poke(false.B)
      dut.io.flagsDelta(j).bits.valid.poke(false.B)
      dut.io.flagsDelta(j).bits.owner.ctx_slot.poke(0.U)
      dut.io.flagsDelta(j).bits.owner.ctx_generation.poke(0.U)
      dut.io.flagsDelta(j).bits.owner.owner_op_seq.poke(0.U)
      dut.io.flagsDelta(j).bits.owner.subop_id.poke(0.U)
      dut.io.flagsDelta(j).bits.rob_tag.ctx_slot.poke(0.U)
      dut.io.flagsDelta(j).bits.rob_tag.rob_slot.poke(0.U)
      dut.io.flagsDelta(j).bits.rob_tag.slot_generation.poke(0.U)
      dut.io.flagsDelta(j).bits.fflags.poke(0.U)
      dut.io.flagsDelta(j).bits.vxsat.poke(false.B)
      dut.io.flagsDelta(j).bits.replaceFflags.poke(false.B)
      dut.io.flagsDelta(j).bits.replaceVxsat.poke(false.B)
    }
    for (i <- 0 until dut.io.flagsCapable.length) {
      dut.io.flagsCapable(i).poke(false.B)
      dut.io.configCapable(i).poke(false.B)
    }
    dut.io.csrPark.valid.poke(false.B)
    dut.io.csrPark.bits.owner.ctx_slot.poke(0.U)
    dut.io.csrPark.bits.owner.ctx_generation.poke(0.U)
    dut.io.csrPark.bits.owner.owner_op_seq.poke(0.U)
    dut.io.csrPark.bits.owner.subop_id.poke(0.U)
    dut.io.csrPark.bits.fflags.poke(0.U)
    dut.io.csrPark.bits.vxsat.poke(false.B)
    dut.io.csrPark.bits.replaceFflags.poke(false.B)
    dut.io.csrPark.bits.replaceVxsat.poke(false.B)
    dut.io.csrPark.bits.commitsFrm.poke(false.B)
    dut.io.csrPark.bits.frmValue.poke(0.U)
  }

  "Same-PC concurrent stores across two Contexts match strictly by OwnerTag and retire independently (RET-01 RET-03 T-09)" in {
    simulate(new RetirementPartitionBank(makeParams(2), mini = false)) { dut =>
      dut.reset.poke(true.B)
      idleInputs(dut)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      val sharedPc = "h20001000".U(32.W)

      // Cycle 0: Context 0 dispatches sw at sharedPc with op_seq = 7
      dut.io.activeCtx.poke(0.U)
      dut.io.dispatchOpSeq(0).poke(7.U)
      dut.io.inst(0).valid.poke(true.B)
      dut.io.inst(0).bits.addr.poke(sharedPc)
      dut.io.inst(0).bits.inst.poke(SW_X5_0_X1)
      dut.clock.step()
      idleInputs(dut)

      // Cycle 1: Context 1 dispatches sw at the EXACT SAME PC AND SAME op_seq = 7
      dut.io.activeCtx.poke(1.U)
      dut.io.dispatchOpSeq(0).poke(7.U)
      dut.io.inst(0).valid.poke(true.B)
      dut.io.inst(0).bits.addr.poke(sharedPc)
      dut.io.inst(0).bits.inst.poke(SW_X5_0_X1)
      dut.clock.step()
      idleInputs(dut)

      dut.io.perCtxEmpty(0).expect(false.B)
      dut.io.perCtxEmpty(1).expect(false.B)

      // Complete ONLY Context 1's store (ctx_slot=1, gen=0, op_seq=7).
      // Context 0's store at the identical PC is still in flight and MUST NOT retire!
      dut.io.storeComplete.valid.poke(true.B)
      dut.io.storeComplete.bits.ctx_slot.poke(1.U)
      dut.io.storeComplete.bits.ctx_generation.poke(0.U)
      dut.io.storeComplete.bits.owner_op_seq.poke(7.U)
      dut.clock.step()
      idleInputs(dut)

      // Pipe(storeComplete) takes 1 cycle inside RetirementBuffer -> retires on this cycle.
      dut.io.perCtxDeqReady(1).expect(1.U)
      dut.io.perCtxDeqReady(0).expect(0.U)
      dut.clock.step()

      // Context 1 retired independently while Context 0 head remained stalled!
      dut.io.perCtxEmpty(1).expect(true.B)
      dut.io.perCtxEmpty(0).expect(false.B)

      // Inject a wrong-op_seq storeComplete for Context 0 (same PC in legacy model, wrong op_seq=8): must NOT match.
      dut.io.storeComplete.valid.poke(true.B)
      dut.io.storeComplete.bits.ctx_slot.poke(0.U)
      dut.io.storeComplete.bits.ctx_generation.poke(0.U)
      dut.io.storeComplete.bits.owner_op_seq.poke(8.U)
      dut.clock.step()
      idleInputs(dut)
      dut.clock.step()
      dut.io.perCtxEmpty(0).expect(false.B)

      // Inject a stale-generation storeComplete for Context 0 (gen=1 while live gen=0): rejected & counted!
      dut.io.storeComplete.valid.poke(true.B)
      dut.io.storeComplete.bits.ctx_slot.poke(0.U)
      dut.io.storeComplete.bits.ctx_generation.poke(1.U)
      dut.io.storeComplete.bits.owner_op_seq.poke(7.U)
      dut.clock.step()
      idleInputs(dut)
      dut.clock.step()
      dut.io.perCtxEmpty(0).expect(false.B)
      assert(dut.io.staleRejected.peek().litValue >= 1)

      // Finally send matching OwnerTag (slot=0, gen=0, op_seq=7) -> Context 0 retires cleanly.
      dut.io.storeComplete.valid.poke(true.B)
      dut.io.storeComplete.bits.ctx_slot.poke(0.U)
      dut.io.storeComplete.bits.ctx_generation.poke(0.U)
      dut.io.storeComplete.bits.owner_op_seq.poke(7.U)
      dut.clock.step()
      idleInputs(dut)
      dut.io.perCtxDeqReady(0).expect(1.U)
      dut.clock.step()
      dut.io.perCtxEmpty(0).expect(true.B)
    }
  }

  "Same-register scalar writebacks match strictly by OwnerTag, not register number, and fault flush is isolated per Context (RET-06)" in {
    simulate(new RetirementPartitionBank(makeParams(2), mini = false)) { dut =>
      dut.reset.poke(true.B)
      idleInputs(dut)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      // Context 0 dispatches two back-to-back writes to x5 (op_seq 10 and 11).
      dut.io.activeCtx.poke(0.U)
      dut.io.dispatchOpSeq(0).poke(10.U)
      dut.io.dispatchOpSeq(1).poke(11.U)
      dut.io.inst(0).valid.poke(true.B)
      dut.io.inst(0).bits.addr.poke("h1000".U)
      dut.io.inst(0).bits.inst.poke(ADDI_X5_7)
      dut.io.writeAddrScalar(0).valid.poke(true.B)
      dut.io.writeAddrScalar(0).addr.poke(5.U)
      dut.io.inst(1).valid.poke(true.B)
      dut.io.inst(1).bits.addr.poke("h1004".U)
      dut.io.inst(1).bits.inst.poke(ADDI_X5_7)
      dut.io.writeAddrScalar(1).valid.poke(true.B)
      dut.io.writeAddrScalar(1).addr.poke(5.U)
      dut.clock.step()
      idleInputs(dut)

      // Context 1 also dispatches an instruction writing x5 (op_seq 10).
      dut.io.activeCtx.poke(1.U)
      dut.io.dispatchOpSeq(0).poke(10.U)
      dut.io.inst(0).valid.poke(true.B)
      dut.io.inst(0).bits.addr.poke("h1000".U)
      dut.io.inst(0).bits.inst.poke(ADDI_X5_7)
      dut.io.writeAddrScalar(0).valid.poke(true.B)
      dut.io.writeAddrScalar(0).addr.poke(5.U)
      dut.clock.step()
      idleInputs(dut)

      // Deliver writeback ONLY for Context 0's YOUNGER op_seq=11 (x5 = 0xBBBB).
      // Under legacy regno matching BOTH entries in Context 0 would mark ready!
      // With OwnerTag matching, op_seq=10 stays unready so deqReady(0) MUST remain 0.
      dut.io.writeDataScalar(4).valid.poke(true.B)
      dut.io.writeDataScalar(4).bits.addr.poke(5.U)
      dut.io.writeDataScalar(4).bits.data.poke("h0000bbbb".U)
      dut.io.writeDataScalar(4).bits.owner.ctx_slot.poke(0.U)
      dut.io.writeDataScalar(4).bits.owner.ctx_generation.poke(0.U)
      dut.io.writeDataScalar(4).bits.owner.owner_op_seq.poke(11.U)
      dut.clock.step()
      idleInputs(dut)
      dut.io.perCtxDeqReady(0).expect(0.U)
      dut.io.perCtxEmpty(0).expect(false.B)

      // Now inject a non-RVV load fault on Context 0 op_seq=10 (even with mepc != 0x1000 to prove mepc is NOT the key).
      dut.io.fault.valid.poke(true.B)
      dut.io.fault.bits.mepc.poke("hdeadbeef".U)
      dut.io.fault.bits.mcause.poke(5.U)
      dut.io.fault.bits.decode.poke(false.B)
      dut.io.fault.bits.is_rvv.get.poke(false.B)
      dut.io.fault.bits.owner.ctx_slot.poke(0.U)
      dut.io.fault.bits.owner.ctx_generation.poke(0.U)
      dut.io.fault.bits.owner.owner_op_seq.poke(10.U)
      dut.clock.step()
      idleInputs(dut)

      // Fault retires & flushes Context 0 ONLY; Context 1's entry at 0x1000 MUST remain intact!
      dut.io.perCtxTrapRetired(0).expect(true.B)
      dut.io.perCtxTrapRetired(1).expect(false.B)
      dut.clock.step()
      dut.io.perCtxEmpty(0).expect(true.B)
      dut.io.perCtxEmpty(1).expect(false.B)

      // Complete Context 1's x5 writeback -> Context 1 retires normally!
      dut.io.writeDataScalar(0).valid.poke(true.B)
      dut.io.writeDataScalar(0).bits.addr.poke(5.U)
      dut.io.writeDataScalar(0).bits.data.poke("h12345678".U)
      dut.io.writeDataScalar(0).bits.owner.ctx_slot.poke(1.U)
      dut.io.writeDataScalar(0).bits.owner.ctx_generation.poke(0.U)
      dut.io.writeDataScalar(0).bits.owner.owner_op_seq.poke(10.U)
      dut.io.perCtxDeqReady(1).expect(1.U)
      dut.clock.step()
      idleInputs(dut)
      dut.io.perCtxEmpty(1).expect(true.B)
    }
  }

  "Composite RobTag (ctx_slot, rob_slot, slot_generation) rejects reused-slot ABA across wraparound" in {
    val p = makeParams(2)
    simulate(new RetirementPartitionBank(p, mini = false)) { dut =>
      dut.reset.poke(true.B)
      idleInputs(dut)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      // Dispatch 16 scalar instructions in Context 0 and retire them to wrap rob_slot 0 to slot_generation = 1.
      for (wave <- 0 until 4) {
        dut.io.activeCtx.poke(0.U)
        for (lane <- 0 until 4) {
          val seq = wave * 4 + lane
          dut.io.dispatchOpSeq(lane).poke(seq.U)
          dut.io.inst(lane).valid.poke(true.B)
          dut.io.inst(lane).bits.addr.poke((0x3000 + seq * 4).U)
          dut.io.inst(lane).bits.inst.poke(ADDI_X5_7)
          dut.io.writeAddrScalar(lane).valid.poke(true.B)
          dut.io.writeAddrScalar(lane).addr.poke((lane + 1).U)
        }
        dut.clock.step()
        idleInputs(dut)
        for (lane <- 0 until 4) {
          val seq = wave * 4 + lane
          dut.io.writeDataScalar(lane).valid.poke(true.B)
          dut.io.writeDataScalar(lane).bits.addr.poke((lane + 1).U)
          dut.io.writeDataScalar(lane).bits.data.poke(seq.U)
          dut.io.writeDataScalar(lane).bits.owner.ctx_slot.poke(0.U)
          dut.io.writeDataScalar(lane).bits.owner.ctx_generation.poke(0.U)
          dut.io.writeDataScalar(lane).bits.owner.owner_op_seq.poke(seq.U)
        }
        dut.io.perCtxDeqReady(0).expect(4.U)
        dut.clock.step()
        idleInputs(dut)
      }
      dut.io.perCtxEmpty(0).expect(true.B)
      // Slot 0 now has slot_generation == 1!
      dut.io.activeCtx.poke(0.U)
      val liveSlotGen = dut.io.enqSlotGen(0).peek().litValue
      assert(liveSlotGen == 1, s"expected slot 0 generation to advance to 1, got $liveSlotGen")

      // Enqueue a vector instruction at wrapped rob_slot = 0 (slot_generation = 1).
      dut.io.dispatchOpSeq(0).poke(100.U)
      dut.io.inst(0).valid.poke(true.B)
      dut.io.inst(0).bits.addr.poke("h4000".U)
      dut.io.inst(0).bits.inst.poke(VLE32_V1_X1)
      dut.io.isVector.get(0).poke(true.B)
      dut.io.writeAddrVector.get(0).valid.poke(true.B)
      dut.io.writeAddrVector.get(0).addr.poke(1.U)
      dut.clock.step()
      idleInputs(dut)

      val staleBefore = dut.io.staleRejected.peek().litValue
      // Stale ABA vector writeback for (ctx_slot=0, rob_slot=0, slot_generation=0) MUST be rejected!
      dut.io.writeDataVector.get(0).valid.poke(true.B)
      dut.io.writeDataVector.get(0).bits.w_valid.poke(true.B)
      dut.io.writeDataVector.get(0).bits.addr.poke(1.U)
      dut.io.writeDataVector.get(0).bits.last_uop_valid.poke(true.B)
      dut.io.writeDataVector.get(0).bits.rob_tag.ctx_slot.poke(0.U)
      dut.io.writeDataVector.get(0).bits.rob_tag.rob_slot.poke(0.U)
      dut.io.writeDataVector.get(0).bits.rob_tag.slot_generation.poke(0.U)
      dut.io.perCtxDeqReady(0).expect(0.U)
      dut.clock.step()
      idleInputs(dut)
      dut.io.perCtxEmpty(0).expect(false.B)
      assert(dut.io.staleRejected.peek().litValue == staleBefore + 1)

      // Matching (ctx_slot=0, rob_slot=0, slot_generation=1) retires cleanly!
      dut.io.writeDataVector.get(0).valid.poke(true.B)
      dut.io.writeDataVector.get(0).bits.w_valid.poke(true.B)
      dut.io.writeDataVector.get(0).bits.addr.poke(1.U)
      dut.io.writeDataVector.get(0).bits.last_uop_valid.poke(true.B)
      dut.io.writeDataVector.get(0).bits.rob_tag.ctx_slot.poke(0.U)
      dut.io.writeDataVector.get(0).bits.rob_tag.rob_slot.poke(0.U)
      dut.io.writeDataVector.get(0).bits.rob_tag.slot_generation.poke(1.U)
      dut.io.perCtxDeqReady(0).expect(1.U)
      dut.clock.step()
      idleInputs(dut)
      dut.io.perCtxEmpty(0).expect(true.B)
    }
  }
}
