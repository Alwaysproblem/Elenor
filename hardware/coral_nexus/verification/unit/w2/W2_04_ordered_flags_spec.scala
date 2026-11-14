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
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec

/** C027 / W2-04 ordered-merge proof over the REAL FlagsParkMap -> ROB parking ->
  * RetirementBuffer -> CtxCsr chain (pure Float config: enableFloat=true, enableRvv=false —
  * the unified slot table must cover scalar-float programs with no RVV tables).
  *
  * The float-delta producer drives exactly the FloatCore interface pair
  * (`fflagsOwner`, `csr.in.fflags`) into FlagsParkMap, whose alloc ports are fed by the
  * bank's own dispatch fires — bit-for-bit the wiring SCore performs.
  *
  * Cases (G2_scalar.md C027 step 4 / W2 plan §8.2):
  *   - golden: older FP (0b10001) -> `csrw fflags, 0` -> younger FP (0b00100), younger
  *     parked EARLY; per-retire diagnostic log of every retiring delta; final value
  *     exactly the younger delta.
  *   - same-beat arrival+retire: a delta landing on its slot's retirement beat is
  *     published (merged view), never lost to the clearing connect.
  *   - frm/fcsr retire-commit: architectural frm untouched pre-retire; LAST retiring
  *     commitsFrm lane (younger) wins.
  *   - CSR pure read (CSRRS rs1=x0) parks nothing.
  *   - slot reuse after perCtxReset: an old-generation delta carrying the SAME physical
  *     slot AND SAME slot_generation as the new occupant is rejected and counted
  *     (owner.ctx_generation is the discriminator); a late delta after retire likewise.
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
    val fpm  = Module(new FlagsParkMap(p))

    val io = IO(new Bundle {
      val bankIO     = new RetirementPartitionBankIO(p)
      val csrIO      = new CtxCsrIO(p)
      val floatOwner = Input(new OwnerTag(p))
      val floatFlags = Input(Valid(UInt(5.W)))
      // Direct rob_tag-carrying producer (the port contract every HW producer uses;
      // W3 RVV deltas arrive this way). Lets a stale-identity delta bypass the owner
      // CAM so the BANK-side (rob_tag, owner) pair check is exercised directly.
      val injectDelta = Input(Valid(new ParkedFlagsDeltaIO(p)))
    })
    io.bankIO <> bank.io
    io.csrIO <> csr.io
    // Internal bridges MUST come after the passthrough connects: `:=` on a sink is
    // last-connect-wins, so these override the all-zero harness-side drivers.
    bank.io.perCtxGen := csr.io.perCtxGen
    csr.io.retiredDelta := bank.io.perCtxRetiredDelta(0)
    bank.io.csrPark := csr.io.csrParkOut
    bank.io.perCtxReset := csr.io.advanceGen
    csr.io.perCtxTrapRetired := bank.io.perCtxTrapRetired
    csr.io.perCtxNRetired := bank.io.perCtxNRetired
    csr.io.perCtxRetiredConfig := bank.io.perCtxRetiredConfig
    bank.io.configDelta.foreach { d => d.valid := false.B; d.bits := 0.U.asTypeOf(d.bits) }

    // FlagsParkMap fed exactly as SCore feeds it: one alloc per dispatch fire, owner
    // and RobTag built from the bank's own generation/enqueue metadata.
    val slotW = log2Ceil(p.retirementBufferSize)
    for (i <- 0 until p.instructionLanes) {
      val prevFires = PopCount((0 until i).map(j => bank.io.inst(j).fire))
      val slot = if (p.retirementBufferSize > 1) {
        (bank.io.enqPtr + prevFires)(slotW - 1, 0)
      } else { 0.U(1.W) }
      fpm.io.alloc(i).valid := bank.io.inst(i).fire
      fpm.io.alloc(i).bits.owner := OwnerTag.make(p)(
        0.U(p.ctxBits.W),
        csr.io.perCtxGen(0),
        bank.io.dispatchOpSeq(i)
      )
      fpm.io.alloc(i).bits.rob_tag := RobTag.make(p)(
        0.U(p.ctxBits.W),
        slot,
        bank.io.enqSlotGen(i)
      )
    }

    fpm.io.liveMask := bank.io.perCtxLiveMask
    fpm.io.generation := csr.io.perCtxGen
    fpm.io.pendingData := bank.io.perCtxPendingData
    fpm.io.trap := bank.io.perCtxTrapRetired
    fpm.io.settled.foreach { d => d.valid := false.B; d.bits := 0.U }
    // Injection bypasses the lookup so the ROB's independent identity check is exercised.
    fpm.io.float.req := io.floatOwner
    val fpmDeltaValid = io.floatFlags.valid && fpm.io.float.resp.valid
    val fpmDeltaBits = {
      val w = Wire(new ParkedFlagsDeltaIO(p))
      w.valid := fpmDeltaValid
      w.owner := io.floatOwner
      w.rob_tag := fpm.io.float.resp.bits
      w.fflags := io.floatFlags.bits
      w.vxsat := false.B
      w.replaceFflags := false.B
      w.replaceVxsat := false.B
      w
    }
    bank.io.flagsDelta(0).valid := io.injectDelta.valid || fpmDeltaValid
    bank.io.flagsDelta(0).bits := Mux(io.injectDelta.valid, io.injectDelta.bits, fpmDeltaBits)
  }

  def pokeOwner(o: OwnerTag, seq: BigInt, gen: BigInt = 0): Unit = {
    o.ctx_slot.poke(0.U)
    o.ctx_generation.poke(gen.U)
    o.owner_op_seq.poke(seq.U)
    o.subop_id.poke(0.U)
  }

  "older FP then csrw fflags 0 then younger FP merges in retire order with final value exactly the younger delta, logged per retire" in {
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
          b.flagsCapable(i).poke(false.B)
          b.configCapable(i).poke(false.B)
        }
        b.writeAddrFloat.get.valid.poke(false.B)
        b.writeAddrFloat.get.addr.poke(0.U)
        b.storeComplete.valid.poke(false.B)
        pokeOwner(b.storeComplete.bits, 0)
        for (j <- 0 until 6) {
          b.writeDataScalar(j).valid.poke(false.B)
          b.writeDataScalar(j).bits.addr.poke(0.U)
          b.writeDataScalar(j).bits.data.poke(0.U)
          pokeOwner(b.writeDataScalar(j).bits.owner, 0)
        }
        for (j <- 0 until 2) {
          b.writeDataFloat.get(j).valid.poke(false.B)
          b.writeDataFloat.get(j).bits.addr.poke(0.U)
          b.writeDataFloat.get(j).bits.data.poke(0.U)
          pokeOwner(b.writeDataFloat.get(j).bits.owner, 0)
        }
        b.fault.valid.poke(false.B)
        b.fault.bits.mepc.poke(0.U)
        b.fault.bits.mtval.poke(0.U)
        b.fault.bits.mcause.poke(0.U)
        b.fault.bits.decode.poke(false.B)
        pokeOwner(b.fault.bits.owner, 0)
        pokeOwner(dut.io.floatOwner, 0)
        dut.io.floatFlags.valid.poke(false.B)
        dut.io.floatFlags.bits.poke(0.U)

        c.activeCtx.poke(0.U)
        c.advanceGen(0).poke(false.B)
        for (i <- 0 until c.csr.in.value.length) { c.csr.in.value(i).poke(0.U) }
        c.req.valid.poke(false.B)
        c.req.bits.addr.poke(0.U)
        c.req.bits.index.poke(0.U)
        c.req.bits.rs1.poke(0.U)
        c.req.bits.op.poke(CsrOp.CSRRW)
        pokeOwner(c.reqOwner, 0)
        c.reqFromDm.poke(false.B)
        c.perCtxTrapRetired(0).poke(false.B)
        c.rs1.valid.poke(true.B)
        c.rs1.data.poke(0.U)
        c.bru.in.mode.valid.poke(false.B)
        c.bru.in.mcause.valid.poke(false.B)
        c.bru.in.mepc.valid.poke(false.B)
        c.bru.in.mtval.valid.poke(false.B)
        c.bru.in.halt.poke(false.B)
        c.bru.in.fault.poke(false.B)
        c.bru.in.wfi.poke(false.B)
        pokeOwner(c.bruOwner, 0)
        c.float.get.in.fflags.valid.poke(false.B)
        c.float.get.in.fflags.bits.poke(0.U)
        pokeOwner(c.floatOwner.get, 0)
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

      // EARLY completions, before ANY retirement: park both FP deltas now, through the
      // real owner->slot CAM (FlagsParkMap), exactly as FloatCore would report them.
      def parkFp(seq: BigInt, delta: BigInt, gen: BigInt = 0): Unit = {
        pokeOwner(dut.io.floatOwner, seq, gen)
        dut.io.floatFlags.bits.poke(delta.U)
        dut.io.floatFlags.valid.poke(true.B)
        dut.clock.step()
        dut.io.floatFlags.valid.poke(false.B)
        dut.io.floatFlags.bits.poke(0.U)
      }

      parkFp(10, BigInt("10001", 2)) // older FP
      parkFp(12, BigInt("00100", 2)) // younger FP completes early

      // csrw fflags, x0 through the REAL Csr: parks at op_seq 11's slot (the owner rides
      // the same one-beat req pipeline as the bank's internal request register).
      c.req.valid.poke(true.B)
      c.req.bits.index.poke("h001".U) // fflags
      c.req.bits.op.poke(CsrOp.CSRRW)
      c.req.bits.rs1.poke(0.U)
      c.req.bits.addr.poke(0.U) // rd = x0: pure write
      c.rs1.data.poke(0.U)
      pokeOwner(c.reqOwner, 11)
      dut.clock.step()
      c.req.valid.poke(false.B)

      // Pre-retirement invariant: architecture untouched.
      c.perCtxFflags(0).expect(0.U)
      c.perCtxFrm(0).expect(0.U)

      // Per-retire diagnostic log: the retiring-lane delta of this Context is printed
      // every beat (design section 8 table rows). Peek is in-beat: the writeback below
      // retires its entry in the SAME combinational beat.
      def logRetires(tag: String): Unit = {
        for (lane <- 0 until p.retirementLanes) {
          val d = b.perCtxRetiredDelta(0)(lane)
          if (d.valid.peek().litValue == 1) {
            println(
              s"[W2_04_OrderedFlagsSpec] retire $tag lane=$lane " +
                s"op_seq=${d.owner.owner_op_seq.peek().litValue} " +
                s"fflags=0b${d.fflags.peek().litValue.toString(2)} " +
                s"replace=${d.replaceFflags.peek().litValue} frm=${d.commitsFrm.peek().litValue}"
            )
          }
        }
      }

      def writeback(seq: BigInt, port: Int = 0): Unit = {
        b.writeDataScalar(port).valid.poke(true.B)
        b.writeDataScalar(port).bits.addr.poke(5.U)
        b.writeDataScalar(port).bits.data.poke(seq.U)
        pokeOwner(b.writeDataScalar(port).bits.owner, seq)
        logRetires(s"wb($seq)")
        dut.clock.step()
        b.writeDataScalar(port).valid.poke(false.B)
      }

      writeback(10) // older FP retires: parked delta merges.
      c.perCtxFflags(0).expect("b10001".U)

      writeback(11) // csrw fflags,0 retires: REPLACE clears the accumulator.
      c.perCtxFflags(0).expect(0.U)

      writeback(12) // younger FP retires: its early-parked delta is NOT lost.
      c.perCtxFflags(0).expect("b00100".U)
    }
  }

  "a delta arriving on the same beat its slot retires is merged, not lost" in {
    simulate(new Harness) { dut =>
      val b = dut.io.bankIO
      val c = dut.io.csrIO
      def minimal(): Unit = {
        b.activeCtx.poke(0.U)
        b.perCtxGen(0).poke(0.U)
        b.perCtxReset(0).poke(false.B)
        b.dispatchOpSeq(0).poke(0.U)
        for (i <- 0 until 4) {
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
          b.flagsCapable(i).poke(false.B)
          b.configCapable(i).poke(false.B)
        }
        for (j <- 0 until 6) {
          b.writeDataScalar(j).valid.poke(false.B)
          pokeOwner(b.writeDataScalar(j).bits.owner, 0)
        }
        b.writeAddrFloat.get.valid.poke(false.B)
        for (j <- 0 until 2) {
          b.writeDataFloat.get(j).valid.poke(false.B)
          pokeOwner(b.writeDataFloat.get(j).bits.owner, 0)
        }
        b.storeComplete.valid.poke(false.B)
        pokeOwner(b.storeComplete.bits, 0)
        b.fault.valid.poke(false.B)
        pokeOwner(b.fault.bits.owner, 0)
        pokeOwner(dut.io.floatOwner, 0)
        dut.io.floatFlags.valid.poke(false.B)
        dut.io.floatFlags.bits.poke(0.U)
        c.activeCtx.poke(0.U)
        c.advanceGen(0).poke(false.B)
        c.req.valid.poke(false.B)
        c.reqFromDm.poke(false.B)
        c.perCtxTrapRetired(0).poke(false.B)
        pokeOwner(c.reqOwner, 0)
        c.rs1.valid.poke(true.B)
        c.rs1.data.poke(0.U)
        c.bru.in.mode.valid.poke(false.B)
        c.bru.in.mcause.valid.poke(false.B)
        c.bru.in.mepc.valid.poke(false.B)
        c.bru.in.mtval.valid.poke(false.B)
        c.bru.in.halt.poke(false.B)
        c.bru.in.fault.poke(false.B)
        c.bru.in.wfi.poke(false.B)
        pokeOwner(c.bruOwner, 0)
        c.float.get.in.fflags.valid.poke(false.B)
        pokeOwner(c.floatOwner.get, 0)
        c.float_dirty.get.poke(false.B)
        c.perCtxNRetired(0).poke(0.U)
        c.irq.poke(false.B)
        c.dm.debug_req.poke(false.B)
        c.dm.resume_req.poke(false.B)
        c.dm.current_pc.poke(0.U)
        c.dm.next_pc.poke(0.U)
        c.timer_irq.poke(false.B)
        c.software_irq.poke(false.B)
        for (i <- 0 until c.csr.in.value.length) { c.csr.in.value(i).poke(0.U) }
      }

      dut.reset.poke(true.B)
      minimal()
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      b.dispatchOpSeq(0).poke(40.U)
      b.inst(0).valid.poke(true.B)
      b.inst(0).bits.addr.poke(0x2000.U)
      b.inst(0).bits.inst.poke(ADDI_X5_7)
      b.writeAddrScalar(0).valid.poke(true.B)
      b.writeAddrScalar(0).addr.poke(5.U)
      dut.clock.step()
      b.inst(0).valid.poke(false.B)
      b.writeAddrScalar(0).valid.poke(false.B)
      b.dispatchOpSeq(0).poke(0.U)

      // Same beat: writeback (retire) AND the FP delta for the very same owner.
      b.writeDataScalar(0).valid.poke(true.B)
      b.writeDataScalar(0).bits.addr.poke(5.U)
      b.writeDataScalar(0).bits.data.poke(40.U)
      pokeOwner(b.writeDataScalar(0).bits.owner, 40)
      pokeOwner(dut.io.floatOwner, 40)
      dut.io.floatFlags.bits.poke("b01000".U)
      dut.io.floatFlags.valid.poke(true.B)
      dut.clock.step()
      b.writeDataScalar(0).valid.poke(false.B)
      dut.io.floatFlags.valid.poke(false.B)
      pokeOwner(dut.io.floatOwner, 0)

      // The same-beat delta merged into the retiring publication: not lost, and no
      // residue parks on the (now free) slot for the next occupant.
      c.perCtxFflags(0).expect("b01000".U)
      b.perCtxParkedFlags(0).expect(false.B)
    }
  }

  "frm retire-commit leaves architecture untouched pre-retire and the LAST retiring lane wins" in {
    simulate(new Harness) { dut =>
      val b = dut.io.bankIO
      val c = dut.io.csrIO
      // (Reuse the same port driving discipline as case 2; elided here via a local copy.)
      def drive(): Unit = {
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
          b.flagsCapable(i).poke(false.B)
          b.configCapable(i).poke(false.B)
        }
        for (j <- 0 until 6) {
          b.writeDataScalar(j).valid.poke(false.B)
          pokeOwner(b.writeDataScalar(j).bits.owner, 0)
        }
        b.writeAddrFloat.get.valid.poke(false.B)
        for (j <- 0 until 2) {
          b.writeDataFloat.get(j).valid.poke(false.B)
          pokeOwner(b.writeDataFloat.get(j).bits.owner, 0)
        }
        b.storeComplete.valid.poke(false.B)
        pokeOwner(b.storeComplete.bits, 0)
        b.fault.valid.poke(false.B)
        pokeOwner(b.fault.bits.owner, 0)
        pokeOwner(dut.io.floatOwner, 0)
        dut.io.floatFlags.valid.poke(false.B)
        dut.io.floatFlags.bits.poke(0.U)
        c.activeCtx.poke(0.U)
        c.advanceGen(0).poke(false.B)
        c.req.valid.poke(false.B)
        c.reqFromDm.poke(false.B)
        c.perCtxTrapRetired(0).poke(false.B)
        pokeOwner(c.reqOwner, 0)
        c.rs1.valid.poke(true.B)
        c.rs1.data.poke(0.U)
        c.bru.in.mode.valid.poke(false.B)
        c.bru.in.mcause.valid.poke(false.B)
        c.bru.in.mepc.valid.poke(false.B)
        c.bru.in.mtval.valid.poke(false.B)
        c.bru.in.halt.poke(false.B)
        c.bru.in.fault.poke(false.B)
        c.bru.in.wfi.poke(false.B)
        pokeOwner(c.bruOwner, 0)
        c.float.get.in.fflags.valid.poke(false.B)
        pokeOwner(c.floatOwner.get, 0)
        c.float_dirty.get.poke(false.B)
        c.perCtxNRetired(0).poke(0.U)
        c.irq.poke(false.B)
        c.dm.debug_req.poke(false.B)
        c.dm.resume_req.poke(false.B)
        c.dm.current_pc.poke(0.U)
        c.dm.next_pc.poke(0.U)
        c.timer_irq.poke(false.B)
        c.software_irq.poke(false.B)
        for (i <- 0 until c.csr.in.value.length) { c.csr.in.value(i).poke(0.U) }
      }

      dut.reset.poke(true.B)
      drive()
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      // Two csrw frm ops (op_seq 20 older -> frm 1, op_seq 21 younger -> frm 2).
      for ((seq, valBits) <- Seq((20, 1), (21, 2))) {
        b.dispatchOpSeq(0).poke(seq.U)
        b.inst(0).valid.poke(true.B)
        b.inst(0).bits.addr.poke((0x3000 + seq).U)
        b.inst(0).bits.inst.poke(ADDI_X5_7)
        b.writeAddrScalar(0).valid.poke(true.B)
        b.writeAddrScalar(0).addr.poke(5.U)
        dut.clock.step()
        b.inst(0).valid.poke(false.B)
        b.writeAddrScalar(0).valid.poke(false.B)
        b.dispatchOpSeq(0).poke(0.U)

        c.req.valid.poke(true.B)
        c.req.bits.index.poke("h002".U) // frm
        c.req.bits.op.poke(CsrOp.CSRRW)
        c.req.bits.rs1.poke(1.U)
        c.req.bits.addr.poke(0.U)
        c.rs1.data.poke(valBits.U)
        pokeOwner(c.reqOwner, seq)
        dut.clock.step()
        c.req.valid.poke(false.B)
      }

      // Let the second park land (the observation beat is the bank-internal req beat).
      dut.clock.step()

      // Pre-retire: architectural frm untouched, but the dispatch snapshot (frmView,
      // exposed as the float rounding-mode output) already shows the newest older write.
      c.perCtxFrm(0).expect(0.U)
      c.perCtxFrmPending(0).expect(true.B)
      c.float.get.out.frm.expect(2.U)

      // Both retire in ONE beat (two write ports): lane 0 = older, lane 1 = younger.
      // The LAST retiring commitsFrm lane must win.
      b.writeDataScalar(0).valid.poke(true.B)
      b.writeDataScalar(0).bits.addr.poke(5.U)
      b.writeDataScalar(0).bits.data.poke(20.U)
      pokeOwner(b.writeDataScalar(0).bits.owner, 20)
      b.writeDataScalar(1).valid.poke(true.B)
      b.writeDataScalar(1).bits.addr.poke(5.U)
      b.writeDataScalar(1).bits.data.poke(21.U)
      pokeOwner(b.writeDataScalar(1).bits.owner, 21)
      dut.clock.step()
      b.writeDataScalar(0).valid.poke(false.B)
      b.writeDataScalar(1).valid.poke(false.B)

      c.perCtxFrm(0).expect(2.U) // younger (lane 1, op_seq 21) wins
      c.perCtxFrmPending(0).expect(false.B)
      c.float.get.out.frm.expect(2.U)
    }
  }

  "a pure CSR read (CSRRS rs1 x0) parks nothing" in {
    simulate(new Harness) { dut =>
      val b = dut.io.bankIO
      val c = dut.io.csrIO
      def drive(): Unit = {
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
          b.flagsCapable(i).poke(false.B)
          b.configCapable(i).poke(false.B)
        }
        for (j <- 0 until 6) {
          b.writeDataScalar(j).valid.poke(false.B)
          pokeOwner(b.writeDataScalar(j).bits.owner, 0)
        }
        b.writeAddrFloat.get.valid.poke(false.B)
        for (j <- 0 until 2) {
          b.writeDataFloat.get(j).valid.poke(false.B)
          pokeOwner(b.writeDataFloat.get(j).bits.owner, 0)
        }
        b.storeComplete.valid.poke(false.B)
        pokeOwner(b.storeComplete.bits, 0)
        b.fault.valid.poke(false.B)
        pokeOwner(b.fault.bits.owner, 0)
        pokeOwner(dut.io.floatOwner, 0)
        dut.io.floatFlags.valid.poke(false.B)
        dut.io.floatFlags.bits.poke(0.U)
        c.activeCtx.poke(0.U)
        c.advanceGen(0).poke(false.B)
        c.req.valid.poke(false.B)
        c.reqFromDm.poke(false.B)
        c.perCtxTrapRetired(0).poke(false.B)
        pokeOwner(c.reqOwner, 0)
        c.rs1.valid.poke(true.B)
        c.rs1.data.poke(0.U)
        c.bru.in.mode.valid.poke(false.B)
        c.bru.in.mcause.valid.poke(false.B)
        c.bru.in.mepc.valid.poke(false.B)
        c.bru.in.mtval.valid.poke(false.B)
        c.bru.in.halt.poke(false.B)
        c.bru.in.fault.poke(false.B)
        c.bru.in.wfi.poke(false.B)
        pokeOwner(c.bruOwner, 0)
        c.float.get.in.fflags.valid.poke(false.B)
        pokeOwner(c.floatOwner.get, 0)
        c.float_dirty.get.poke(false.B)
        c.perCtxNRetired(0).poke(0.U)
        c.irq.poke(false.B)
        c.dm.debug_req.poke(false.B)
        c.dm.resume_req.poke(false.B)
        c.dm.current_pc.poke(0.U)
        c.dm.next_pc.poke(0.U)
        c.timer_irq.poke(false.B)
        c.software_irq.poke(false.B)
        for (i <- 0 until c.csr.in.value.length) { c.csr.in.value(i).poke(0.U) }
      }

      dut.reset.poke(true.B)
      drive()
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      // csrrs fflags, x0, x5: a read (rs1 == x0 parks nothing).
      c.req.valid.poke(true.B)
      c.req.bits.index.poke("h001".U)
      c.req.bits.op.poke(CsrOp.CSRRS)
      c.req.bits.rs1.poke(0.U)
      c.req.bits.addr.poke(5.U)
      c.rs1.data.poke(0.U)
      pokeOwner(c.reqOwner, 50)
      dut.clock.step()
      c.req.valid.poke(false.B)

      // THE bank-internal request register holds the read THIS beat: a read
      // (rs1 == x0 on CSRRS) must produce no park while it is being observed.
      c.csrParkOut.valid.expect(false.B)
      dut.clock.step()
      c.perCtxFflags(0).expect(0.U)
      b.perCtxParkedFlags(0).expect(false.B)
    }
  }

  "context reuse rejects and counts an old-generation delta on the same physical slot with the same slot_generation" in {
    simulate(new Harness) { dut =>
      val b = dut.io.bankIO
      val c = dut.io.csrIO
      def drive(): Unit = {
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
          b.flagsCapable(i).poke(false.B)
          b.configCapable(i).poke(false.B)
        }
        for (j <- 0 until 6) {
          b.writeDataScalar(j).valid.poke(false.B)
          pokeOwner(b.writeDataScalar(j).bits.owner, 0)
        }
        b.writeAddrFloat.get.valid.poke(false.B)
        for (j <- 0 until 2) {
          b.writeDataFloat.get(j).valid.poke(false.B)
          pokeOwner(b.writeDataFloat.get(j).bits.owner, 0)
        }
        b.storeComplete.valid.poke(false.B)
        pokeOwner(b.storeComplete.bits, 0)
        b.fault.valid.poke(false.B)
        pokeOwner(b.fault.bits.owner, 0)
        pokeOwner(dut.io.floatOwner, 0)
        dut.io.floatFlags.valid.poke(false.B)
        dut.io.floatFlags.bits.poke(0.U)
        c.activeCtx.poke(0.U)
        c.advanceGen(0).poke(false.B)
        c.req.valid.poke(false.B)
        c.reqFromDm.poke(false.B)
        c.perCtxTrapRetired(0).poke(false.B)
        pokeOwner(c.reqOwner, 0)
        c.rs1.valid.poke(true.B)
        c.rs1.data.poke(0.U)
        c.bru.in.mode.valid.poke(false.B)
        c.bru.in.mcause.valid.poke(false.B)
        c.bru.in.mepc.valid.poke(false.B)
        c.bru.in.mtval.valid.poke(false.B)
        c.bru.in.halt.poke(false.B)
        c.bru.in.fault.poke(false.B)
        c.bru.in.wfi.poke(false.B)
        pokeOwner(c.bruOwner, 0)
        c.float.get.in.fflags.valid.poke(false.B)
        pokeOwner(c.floatOwner.get, 0)
        c.float_dirty.get.poke(false.B)
        c.perCtxNRetired(0).poke(0.U)
        c.irq.poke(false.B)
        c.dm.debug_req.poke(false.B)
        c.dm.resume_req.poke(false.B)
        c.dm.current_pc.poke(0.U)
        c.dm.next_pc.poke(0.U)
        c.timer_irq.poke(false.B)
        c.software_irq.poke(false.B)
        for (i <- 0 until c.csr.in.value.length) { c.csr.in.value(i).poke(0.U) }
      }

      dut.reset.poke(true.B)
      drive()
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()

      // Task 1 (generation 0): dispatch op_seq 30, merge its delta, retire it.
      val oldSlot = b.enqPtr.peek().litValue.toInt
      val oldSlotGeneration = b.enqSlotGen(0).peek().litValue
      b.dispatchOpSeq(0).poke(30.U)
      b.inst(0).valid.poke(true.B)
      b.inst(0).bits.addr.poke(0x4000.U)
      b.inst(0).bits.inst.poke(ADDI_X5_7)
      b.writeAddrScalar(0).valid.poke(true.B)
      b.writeAddrScalar(0).addr.poke(5.U)
      dut.clock.step()
      b.inst(0).valid.poke(false.B)
      b.writeAddrScalar(0).valid.poke(false.B)
      b.dispatchOpSeq(0).poke(0.U)

      pokeOwner(dut.io.floatOwner, 30, 0)
      dut.io.floatFlags.bits.poke("b00001".U)
      dut.io.floatFlags.valid.poke(true.B)
      b.writeDataScalar(0).valid.poke(true.B)
      b.writeDataScalar(0).bits.addr.poke(5.U)
      b.writeDataScalar(0).bits.data.poke(30.U)
      pokeOwner(b.writeDataScalar(0).bits.owner, 30)
      dut.clock.step()
      dut.io.floatFlags.valid.poke(false.B)
      b.writeDataScalar(0).valid.poke(false.B)
      pokeOwner(dut.io.floatOwner, 0)
      c.perCtxFflags(0).expect("b00001".U)

      // Safe slot reuse: advanceGen resets the partition (slotGen back to 0) and bumps
      // the Context generation to 1. Per-context reject counters clear with it.
      c.advanceGen(0).poke(true.B)
      dut.clock.step()
      c.advanceGen(0).poke(false.B)
      dut.clock.step()
      c.perCtxFflags(0).expect(0.U) // bank reset cleared the accumulator
      b.perCtxStaleRejected(0).expect(0.U)

      // Task 2 (generation 1): dispatch op_seq 30 AGAIN into the SAME physical slot
      // (buffer empty => enqPtr back at slotBase), so its rob_tag is bit-identical to
      // task 1's {slot, slot_generation 0}. Only owner.ctx_generation differs.
      val slotBase = b.enqPtr.peek().litValue.toInt
      assert(slotBase == oldSlot, "stale-generation test must reuse the same physical slot")
      b.enqSlotGen(0).expect(oldSlotGeneration.U)
      b.dispatchOpSeq(0).poke(30.U)
      b.inst(0).valid.poke(true.B)
      b.inst(0).bits.addr.poke(0x4000.U)
      b.inst(0).bits.inst.poke(ADDI_X5_7)
      b.writeAddrScalar(0).valid.poke(true.B)
      b.writeAddrScalar(0).addr.poke(5.U)
      dut.clock.step()
      b.inst(0).valid.poke(false.B)
      b.writeAddrScalar(0).valid.poke(false.B)
      b.dispatchOpSeq(0).poke(0.U)

      // Stale delta from task 1, injected with a FULL rob_tag (as a W3-style producer
      // would): same slot, same slot_generation, OLD ctx_generation. The RobTag alone
      // aliases the new occupant; only the (rob_tag, owner) pair rejects it.
      def injectStale(fflags: BigInt, ownerGen: BigInt, seq: BigInt, slotGen: BigInt = 0): Unit = {
        pokeOwner(dut.io.injectDelta.bits.owner, seq, ownerGen)
        dut.io.injectDelta.bits.rob_tag.ctx_slot.poke(0.U)
        dut.io.injectDelta.bits.rob_tag.rob_slot.poke(slotBase.U)
        dut.io.injectDelta.bits.rob_tag.slot_generation.poke(slotGen.U)
        dut.io.injectDelta.bits.fflags.poke(fflags.U)
        dut.io.injectDelta.bits.vxsat.poke(false.B)
        dut.io.injectDelta.bits.replaceFflags.poke(false.B)
        dut.io.injectDelta.bits.replaceVxsat.poke(false.B)
        dut.io.injectDelta.valid.poke(true.B)
        dut.clock.step()
        dut.io.injectDelta.valid.poke(false.B)
      }

      injectStale(BigInt("10000", 2), ownerGen = 0, seq = 30)

      // Rejected + counted; the new occupant's parked state stays clean.
      b.perCtxStaleRejected(0).expect(1.U)
      b.perCtxParkedFlags(0).expect(false.B)

      // The new occupant retires with NO stale flags merged.
      b.writeDataScalar(0).valid.poke(true.B)
      b.writeDataScalar(0).bits.addr.poke(5.U)
      b.writeDataScalar(0).bits.data.poke(30.U)
      pokeOwner(b.writeDataScalar(0).bits.owner, 30)
      b.writeDataScalar(0).bits.owner.ctx_generation.poke(1.U)
      dut.clock.step()
      b.writeDataScalar(0).valid.poke(false.B)
      c.perCtxFflags(0).expect(0.U)

      // A LATE delta for the retired occupant (current generation, but its slot is
      // already free) is likewise rejected and counted.
      val before = b.perCtxStaleRejected(0).peek().litValue
      injectStale(BigInt("01000", 2), ownerGen = 1, seq = 30)
      b.perCtxStaleRejected(0).expect((before + 1).U)
      c.perCtxFflags(0).expect(0.U)
    }
  }
}
