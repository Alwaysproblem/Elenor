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

class W2_02_CsrFlagDeltaSpec extends AnyFreeSpec with ChiselSim {
  def makeParams(ctxCount: Int = 2, tileId: Int = 3): Parameters = {
    val p = new Parameters
    p.contextCount       = ctxCount
    p.tileId             = tileId
    p.enableVerification = true
    p.enableFloat        = true
    p.enableRvv          = true
    p.enableVme          = true
    p
  }

  def idleCsr(dut: CtxCsr): Unit = {
    dut.io.activeCtx.poke(0.U)
    for (c <- 0 until dut.io.advanceGen.length) {
      dut.io.advanceGen(c).poke(false.B)
      dut.io.perCtxNRetired(c).poke(0.U)
    }
    for (i <- 0 until dut.io.csr.in.value.length) {
      dut.io.csr.in.value(i).poke(0.U)
    }
    dut.io.req.valid.poke(false.B)
    dut.io.req.bits.addr.poke(0.U)
    dut.io.req.bits.index.poke(0.U)
    dut.io.req.bits.rs1.poke(0.U)
    dut.io.req.bits.op.poke(CsrOp.CSRRW)
    dut.io.reqOwner.ctx_slot.poke(0.U)
    dut.io.reqOwner.ctx_generation.poke(0.U)
    dut.io.reqOwner.owner_op_seq.poke(0.U)
    dut.io.reqOwner.subop_id.poke(0.U)
    dut.io.rs1.valid.poke(true.B)
    dut.io.rs1.data.poke(0.U)

    dut.io.bru.in.mode.valid.poke(false.B)
    dut.io.bru.in.mode.bits.poke(CsrMode.Machine)
    dut.io.bru.in.mcause.valid.poke(false.B)
    dut.io.bru.in.mcause.bits.poke(0.U)
    dut.io.bru.in.mepc.valid.poke(false.B)
    dut.io.bru.in.mepc.bits.poke(0.U)
    dut.io.bru.in.mtval.valid.poke(false.B)
    dut.io.bru.in.mtval.bits.poke(0.U)
    dut.io.bru.in.halt.poke(false.B)
    dut.io.bru.in.fault.poke(false.B)
    dut.io.bru.in.wfi.poke(false.B)
    dut.io.bruOwner.ctx_slot.poke(0.U)
    dut.io.bruOwner.ctx_generation.poke(0.U)
    dut.io.bruOwner.owner_op_seq.poke(0.U)
    dut.io.bruOwner.subop_id.poke(0.U)

    dut.io.float.get.in.fflags.valid.poke(false.B)
    dut.io.float.get.in.fflags.bits.poke(0.U)
    dut.io.floatOwner.get.ctx_slot.poke(0.U)
    dut.io.floatOwner.get.ctx_generation.poke(0.U)
    dut.io.floatOwner.get.owner_op_seq.poke(0.U)
    dut.io.floatOwner.get.subop_id.poke(0.U)

    dut.io.rvv.get.vstart.poke(0.U)
    dut.io.rvv.get.vl.poke(0.U)
    dut.io.rvv.get.vtype.poke(0.U)
    dut.io.rvv.get.vxrm.poke(0.U)
    dut.io.rvv.get.vxsat.poke(false.B)
    dut.io.rvv.get.fflags.valid.poke(false.B)
    dut.io.rvv.get.fflags.bits.poke(0.U)
    dut.io.rvv.get.mtype.poke(0.U)
    dut.io.rvvOwner.get.ctx_slot.poke(0.U)
    dut.io.rvvOwner.get.ctx_generation.poke(0.U)
    dut.io.rvvOwner.get.owner_op_seq.poke(0.U)
    dut.io.rvvOwner.get.subop_id.poke(0.U)

    dut.io.float_dirty.get.poke(false.B)
    dut.io.rvv_dirty.get.poke(false.B)
    dut.io.vme_discard.get.poke(false.B)
    dut.io.vme_ms_dirty.get.poke(false.B)
    dut.io.vme_vs_dirty.get.poke(false.B)
    dut.io.vme_fp_dirty.get.poke(false.B)

    for (k <- 0 until dut.io.retiredDelta.length) {
      dut.io.retiredDelta(k).valid.poke(false.B)
      dut.io.retiredDelta(k).fflags.poke(0.U)
      dut.io.retiredDelta(k).vxsat.poke(false.B)
      dut.io.retiredDelta(k).replaceFflags.poke(false.B)
      dut.io.retiredDelta(k).replaceVxsat.poke(false.B)
      dut.io.retiredDelta(k).owner.ctx_slot.poke(0.U)
      dut.io.retiredDelta(k).owner.ctx_generation.poke(0.U)
      dut.io.retiredDelta(k).owner.owner_op_seq.poke(0.U)
      dut.io.retiredDelta(k).owner.subop_id.poke(0.U)
    }

    dut.io.irq.poke(false.B)
    dut.io.timer_irq.poke(false.B)
    dut.io.software_irq.poke(false.B)
    dut.io.dm.debug_req.poke(false.B)
    dut.io.dm.resume_req.poke(false.B)
    dut.io.dm.current_pc.poke(0.U)
    dut.io.dm.next_pc.poke(0.U)
  }

  "CtxCsr encodes tileId and ctx_slot in mhartid, isolates trap and fflags per Context, and rejects stale deltas on generation bump" in {
    simulate(new CtxCsr(makeParams(ctxCount = 2, tileId = 3))) { dut =>
      dut.reset.poke(true.B)
      idleCsr(dut)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      // 1. Read mhartid (0xF14) on Context 0 -> (3 << 1) | 0 == 6; Context 1 -> (3 << 1) | 1 == 7.
      dut.io.activeCtx.poke(0.U)
      dut.io.req.valid.poke(true.B)
      dut.io.req.bits.addr.poke(10.U)
      dut.io.req.bits.index.poke("hf14".U)
      dut.io.req.bits.op.poke(CsrOp.CSRRS)
      dut.io.req.bits.rs1.poke(0.U)
      dut.io.reqOwner.ctx_slot.poke(0.U)
      dut.io.reqOwner.ctx_generation.poke(0.U)
      dut.clock.step()
      idleCsr(dut)
      dut.io.rd.valid.expect(true.B)
      dut.io.rd.bits.data.expect(6.U)
      dut.io.rd.bits.owner.ctx_slot.expect(0.U)

      dut.io.activeCtx.poke(1.U)
      dut.io.req.valid.poke(true.B)
      dut.io.req.bits.addr.poke(11.U)
      dut.io.req.bits.index.poke("hf14".U)
      dut.io.req.bits.op.poke(CsrOp.CSRRS)
      dut.io.req.bits.rs1.poke(0.U)
      dut.io.reqOwner.ctx_slot.poke(1.U)
      dut.io.reqOwner.ctx_generation.poke(0.U)
      dut.clock.step()
      idleCsr(dut)
      dut.io.rd.valid.expect(true.B)
      dut.io.rd.bits.data.expect(7.U)
      dut.io.rd.bits.owner.ctx_slot.expect(1.U)

      // 2. Accumulate fflags & trap CSRs independently in Context 0 vs Context 1.
      dut.io.retiredDelta(0).valid.poke(true.B)
      dut.io.retiredDelta(0).fflags.poke("b10101".U)
      dut.io.retiredDelta(0).owner.ctx_slot.poke(0.U)
      dut.io.retiredDelta(0).owner.ctx_generation.poke(0.U)

      dut.io.retiredDelta(1).valid.poke(true.B)
      dut.io.retiredDelta(1).fflags.poke("b00010".U)
      dut.io.retiredDelta(1).owner.ctx_slot.poke(1.U)
      dut.io.retiredDelta(1).owner.ctx_generation.poke(0.U)

      dut.io.bru.in.mepc.valid.poke(true.B)
      dut.io.bru.in.mepc.bits.poke("h20001008".U)
      dut.io.bru.in.mcause.valid.poke(true.B)
      dut.io.bru.in.mcause.bits.poke(11.U)
      dut.io.bruOwner.ctx_slot.poke(0.U)
      dut.io.bruOwner.ctx_generation.poke(0.U)
      dut.clock.step()
      idleCsr(dut)

      dut.io.perCtxFflags(0).expect("b10101".U)
      dut.io.perCtxFflags(1).expect("b00010".U)
      dut.io.perCtxMepc(0).expect("h20001008".U)
      dut.io.perCtxMepc(1).expect(0.U)

      // 3. Advance generation on Context 0 ONLY: Context 0 clears to 0 while Context 1 stays intact!
      dut.io.advanceGen(0).poke(true.B)
      dut.clock.step()
      idleCsr(dut)

      dut.io.perCtxGen(0).expect(1.U)
      dut.io.perCtxGen(1).expect(0.U)
      dut.io.perCtxFflags(0).expect(0.U)
      dut.io.perCtxMepc(0).expect(0.U)
      dut.io.perCtxFflags(1).expect("b00010".U)

      // 4. Inject stale gen=0 retired delta, float fflags, and CSR write on Context 0 -> ALL rejected & counted!
      dut.io.retiredDelta(0).valid.poke(true.B)
      dut.io.retiredDelta(0).fflags.poke("b11111".U)
      dut.io.retiredDelta(0).owner.ctx_slot.poke(0.U)
      dut.io.retiredDelta(0).owner.ctx_generation.poke(0.U) // stale!

      dut.io.float.get.in.fflags.valid.poke(true.B)
      dut.io.float.get.in.fflags.bits.poke("b01100".U)
      dut.io.floatOwner.get.ctx_slot.poke(0.U)
      dut.io.floatOwner.get.ctx_generation.poke(0.U) // stale!

      dut.io.req.valid.poke(true.B)
      dut.io.req.bits.index.poke("h340".U) // mscratch
      dut.io.req.bits.rs1.poke(1.U)
      dut.io.req.bits.op.poke(CsrOp.CSRRW)
      dut.io.reqOwner.ctx_slot.poke(0.U)
      dut.io.reqOwner.ctx_generation.poke(0.U) // stale!
      dut.io.rs1.data.poke("hbadc0de".U)
      dut.clock.step()
      idleCsr(dut)
      dut.clock.step()

      dut.io.perCtxFflags(0).expect(0.U)
      dut.io.perCtxMscratch(0).expect(0.U)
      dut.io.staleRejected.expect(3.U)

      // 5. Live gen=1 delta on Context 0 succeeds!
      dut.io.retiredDelta(0).valid.poke(true.B)
      dut.io.retiredDelta(0).fflags.poke("b00101".U)
      dut.io.retiredDelta(0).owner.ctx_slot.poke(0.U)
      dut.io.retiredDelta(0).owner.ctx_generation.poke(1.U)
      dut.clock.step()
      idleCsr(dut)
      dut.io.perCtxFflags(0).expect("b00101".U)
    }
  }
}
