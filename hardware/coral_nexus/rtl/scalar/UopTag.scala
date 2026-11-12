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
import common.Fp32

/** Context identity `(ctx_slot, ctx_generation)` per C010 `rob_tag.yaml` / `ctx_ctrl.yaml`.
  *
  * When `p.contextCount == 1`, `p.ctxBits` is explicitly specialized to 1 bit (`max(1, log2Ceil(1))`)
  * so zero-width fields are never generated.
  */
class ContextKeyBundle(p: Parameters) extends Bundle {
  val ctx_slot       = UInt(p.ctxBits.W)
  val ctx_generation = UInt(p.ctxGenerationBits.W)
}

object ContextKeyBundle {
  def zero(p: Parameters): ContextKeyBundle = {
    val w = Wire(new ContextKeyBundle(p))
    w.ctx_slot       := 0.U(p.ctxBits.W)
    w.ctx_generation := 0.U(p.ctxGenerationBits.W)
    w
  }

  def make(p: Parameters)(slot: UInt, generation: UInt): ContextKeyBundle = {
    val w = Wire(new ContextKeyBundle(p))
    w.ctx_slot       := (if (p.contextCount == 1) 0.U(1.W) else slot(p.ctxBits - 1, 0))
    w.ctx_generation := generation(p.ctxGenerationBits - 1, 0)
    w
  }

  def matches(a: ContextKeyBundle, b: ContextKeyBundle): Bool = {
    (a.ctx_slot === b.ctx_slot) && (a.ctx_generation === b.ctx_generation)
  }
}

/** Per-instruction / per-subop return identity (`rob_tag.yaml` `owner_tag`).
  *
  * Replaces PC-keyed store/fault completions and bare register-number scalar/float writebacks.
  */
class OwnerTag(p: Parameters) extends Bundle {
  val ctx_slot       = UInt(p.ctxBits.W)
  val ctx_generation = UInt(p.ctxGenerationBits.W)
  val owner_op_seq   = UInt(p.ownerOpSeqBits.W)
  val subop_id       = UInt(p.subopIdBits.W)
}

object OwnerTag {
  def zero(p: Parameters): OwnerTag = {
    val w = Wire(new OwnerTag(p))
    w.ctx_slot       := 0.U(p.ctxBits.W)
    w.ctx_generation := 0.U(p.ctxGenerationBits.W)
    w.owner_op_seq   := 0.U(p.ownerOpSeqBits.W)
    w.subop_id       := 0.U(p.subopIdBits.W)
    w
  }

  def make(
    p: Parameters
  )(
    slot: UInt,
    generation: UInt,
    opSeq: UInt,
    subop: UInt = 0.U
  ): OwnerTag = {
    val w = Wire(new OwnerTag(p))
    w.ctx_slot       := (if (p.contextCount == 1) 0.U(1.W) else slot(p.ctxBits - 1, 0))
    w.ctx_generation := generation(p.ctxGenerationBits - 1, 0)
    w.owner_op_seq   := opSeq(p.ownerOpSeqBits - 1, 0)
    w.subop_id       := subop(p.subopIdBits - 1, 0)
    w
  }

  /** Match parent instruction identity `(ctx_slot, ctx_generation, owner_op_seq)`. */
  def matches(a: OwnerTag, b: OwnerTag): Bool = {
    (a.ctx_slot === b.ctx_slot) &&
    (a.ctx_generation === b.ctx_generation) &&
    (a.owner_op_seq === b.owner_op_seq)
  }

  /** Match full subop identity including `subop_id`. */
  def matchesExact(a: OwnerTag, b: OwnerTag): Bool = {
    matches(a, b) && (a.subop_id === b.subop_id)
  }

  /** Check whether `tag` belongs to live occupant `(slot, generation)`. */
  def sameContext(tag: OwnerTag, slot: UInt, generation: UInt): Bool = {
    (tag.ctx_slot === slot) && (tag.ctx_generation === generation)
  }
}

/** Composite RVV/VME retirement tag `(ctx_slot, rob_slot, slot_generation)` (`rob_tag.yaml`).
  *
  * Prevents cross-Context slot aliasing AND same-Context reused-slot ABA.
  */
class RobTag(p: Parameters) extends Bundle {
  val robSlotWidth = math.max(1, log2Ceil(p.retirementBufferSize))

  val ctx_slot        = UInt(p.ctxBits.W)
  val rob_slot        = UInt(robSlotWidth.W)
  val slot_generation = UInt(p.slotGenerationBits.W)
}

object RobTag {
  def zero(p: Parameters): RobTag = {
    val w = Wire(new RobTag(p))
    w.ctx_slot        := 0.U(p.ctxBits.W)
    w.rob_slot        := 0.U(w.robSlotWidth.W)
    w.slot_generation := 0.U(p.slotGenerationBits.W)
    w
  }

  def make(
    p: Parameters
  )(
    slot: UInt,
    robSlot: UInt,
    slotGen: UInt
  ): RobTag = {
    val w = Wire(new RobTag(p))
    w.ctx_slot        := (if (p.contextCount == 1) 0.U(1.W) else slot(p.ctxBits - 1, 0))
    w.rob_slot        := robSlot(w.robSlotWidth - 1, 0)
    w.slot_generation := slotGen(p.slotGenerationBits - 1, 0)
    w
  }

  def matches(a: RobTag, b: RobTag): Bool = {
    (a.ctx_slot === b.ctx_slot) &&
    (a.rob_slot === b.rob_slot) &&
    (a.slot_generation === b.slot_generation)
  }
}

/** Retired flags-delta published at an entry's own retirement beat (W2-02/W2-04, CFG-04). */
class RetiredFlagsDeltaIO(p: Parameters) extends Bundle {
  val valid         = Bool()
  val owner         = new OwnerTag(p)
  val fflags        = UInt(5.W)
  val vxsat         = Bool()
  val replaceFflags = Bool()
  val replaceVxsat  = Bool()
  val commitsFrm    = Bool()
  val frmValue      = UInt(3.W)
}

/** HW fflags/vxsat delta (FloatCore/RVV); parks at its ROB slot until retirement. */
class ParkedFlagsDeltaIO(p: Parameters) extends Bundle {
  val valid         = Bool()
  val rob_tag       = new RobTag(p)
  val fflags        = UInt(5.W)
  val vxsat         = Bool()
  val replaceFflags = Bool()
  val replaceVxsat  = Bool()
}

/** Software fflags/fcsr/frm CSR write observed one beat after dispatch; parks at its slot. */
class ParkedCsrDeltaIO(p: Parameters) extends Bundle {
  val valid         = Bool()
  val owner         = new OwnerTag(p)
  val fflags        = UInt(5.W)
  val vxsat         = Bool()
  val replaceFflags = Bool()
  val replaceVxsat  = Bool()
  val commitsFrm    = Bool()
  val frmValue      = UInt(3.W)
}

/** Owned scalar/float/vector/VME writeback bundles carrying `OwnerTag` / `RobTag` (`W2-02`). */
class OwnedRegfileWriteDataIO(p: Parameters) extends Bundle {
  val addr  = UInt(p.scalarRegCountWidth.W)
  val data  = UInt(p.xlen.W)
  val owner = new OwnerTag(p)
}

class OwnedFloatWriteDataIO(p: Parameters) extends Bundle {
  val addr  = UInt(p.floatRegCountWidth.W)
  val data  = UInt(32.W)
  val owner = new OwnerTag(p)
}

class OwnedFRegfileWrite(p: Parameters) extends Bundle {
  val valid = Input(Bool())
  val addr  = Input(UInt(p.floatRegCountWidth.W))
  val data  = Input(new Fp32)
  val owner = Input(new OwnerTag(p))
}

class OwnedVectorWriteDataIO(p: Parameters) extends Bundle {
  val w_valid        = Bool()
  val addr           = UInt(p.rvvRegCountWidth.W)
  val data           = Option.when(p.enableVerification)(UInt(p.lsuDataBits.W))
  val uop_pc         = Option.when(p.enableVerification)(UInt(p.programCounterBits.W))
  val last_uop_valid = Bool()
  val rob_tag        = new RobTag(p)
}

class OwnedTileWriteDataIO(p: Parameters) extends Bundle {
  val rob_tag  = new RobTag(p)
  val is_store = Bool()
  val mask     = UInt(p.vmeMaxTileWrites.W)
  val idx      = UInt(p.vmeRegCountWidth.W)
  val data = Option.when(p.enableVerification)(
    Vec(p.vmeMaxTileWrites, Vec(p.vmeNumSubtiles, UInt(p.vmeSubtileBits.W)))
  )
  val pc = Option.when(p.enableVerification)(UInt(p.programCounterBits.W))
}

class OwnedMtypeWriteDataIO(p: Parameters) extends Bundle {
  val rob_tag = new RobTag(p)
  val data    = Option.when(p.enableVerification)(UInt(p.xlen.W))
}

/** Per-Context control/observability interface exposed by `SCore` and `Core` (C026 / W2-02). */
class CoreContextIO(p: Parameters) extends Bundle {
  val ctxSelect         = Input(UInt(p.ctxBits.W))
  val advanceGen        = Input(Vec(p.contextCount, Bool()))
  val perCtxGeneration  = Output(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))
  val perCtxNextOpSeq   = Output(Vec(p.contextCount, UInt(p.ownerOpSeqBits.W)))
  val perCtxRobSpace    = Output(Vec(p.contextCount, UInt(log2Ceil(p.retirementBufferSize + 1).W)))
  val perCtxRobEmpty    = Output(Vec(p.contextCount, Bool()))
  val perCtxNRetired    = Output(Vec(p.contextCount, UInt(log2Ceil(p.retirementLanes + 1).W)))
  val perCtxTrapRetired = Output(Vec(p.contextCount, Bool()))
  val staleRfRejected   = Output(UInt(16.W))
  val staleRobRejected  = Output(UInt(16.W))
  val staleCsrRejected  = Output(UInt(16.W))
}
