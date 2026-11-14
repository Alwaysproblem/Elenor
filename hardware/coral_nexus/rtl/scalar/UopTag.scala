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

/** Opaque native return identity. Low bits are ctx_slot; all remaining parent
  * identity fields survive the native queues. Parent instructions have subop_id=0.
  * The transport word is private to this simulation target, not a frozen wire ABI.
  */
class NativeRobIdentity(p: Parameters) extends Bundle {
  val ctx_generation = UInt(p.ctxGenerationBits.W)
  val owner_op_seq = UInt(p.ownerOpSeqBits.W)
  val slot_generation = UInt(p.slotGenerationBits.W)
  val rob_slot = UInt(math.max(1, log2Ceil(p.retirementBufferSize)).W)
  val ctx_slot = UInt(p.ctxBits.W)
  require(p.ctxGenerationBits + p.ownerOpSeqBits + p.slotGenerationBits +
    math.max(1, log2Ceil(p.retirementBufferSize)) + p.ctxBits <= p.nativeRobTagBits)

  def owner: OwnerTag = {
    val result = Wire(new OwnerTag(p))
    result.ctx_slot := ctx_slot
    result.ctx_generation := ctx_generation
    result.owner_op_seq := owner_op_seq
    result.subop_id := 0.U
    result
  }
  def robTag: RobTag = {
    val result = Wire(new RobTag(p))
    result.ctx_slot := ctx_slot
    result.rob_slot := rob_slot
    result.slot_generation := slot_generation
    result
  }
}

object NativeRobIdentity {
  def pack(p: Parameters)(owner: OwnerTag, tag: RobTag): UInt = {
    val key = Cat(owner.ctx_generation, owner.owner_op_seq, tag.slot_generation,
      tag.rob_slot, owner.ctx_slot)
    require(key.getWidth <= p.nativeRobTagBits)
    key.pad(p.nativeRobTagBits)
  }
  def unpack(p: Parameters)(bits: UInt): NativeRobIdentity = {
    val key = Wire(new NativeRobIdentity(p))
    val slotEnd = p.ctxBits + math.max(1, log2Ceil(p.retirementBufferSize))
    val slotGenEnd = slotEnd + p.slotGenerationBits
    val opSeqEnd = slotGenEnd + p.ownerOpSeqBits
    key.ctx_slot := bits(p.ctxBits - 1, 0)
    key.rob_slot := bits(slotEnd - 1, p.ctxBits)
    key.slot_generation := bits(slotGenEnd - 1, slotEnd)
    key.owner_op_seq := bits(opSeqEnd - 1, slotGenEnd)
    key.ctx_generation := bits(opSeqEnd + p.ctxGenerationBits - 1, opSeqEnd)
    key
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

/** HW fflags/vxsat delta (FloatCore/RVV); parks at its ROB slot until retirement.
  *
  * Identity is the PAIR `(rob_tag, owner)`: `slot_generation` alone cannot reject a delta
  * from a previous task, because a Context-slot reset (perCtxReset) clears `slotGen` back
  * to 0, so an old `{slot 0, gen 0}` tag would alias the new occupant of the same physical
  * slot. `owner.ctx_generation` is monotonic across task reuse (incremented by `advanceGen`,
  * never reset), so requiring BOTH matches closes the ABA and stale-generation windows
  * without changing the frozen `rob_tag.yaml` RobTag encoding (W2-04 / CTX-08/09, INV-15).
  */
class ParkedFlagsDeltaIO(p: Parameters) extends Bundle {
  val valid         = Bool()
  val owner         = new OwnerTag(p)
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

/** Per-op architectural config payload (vset/mset) captured at the op's own
  * execution and committed ONLY at its ROB retirement (W2-04 / C027). Masks include
  * the pinned ISA's implicit effects: vset* clears MTYPE; msetmtype/i also clears
  * VL and updates VTYPE; msettn writes VL; msettm/tk writes MTYPE. The payload is
  * the op's IMMUTABLE computed state, never shared engine state sampled at retire.
  */
class ParkedConfigIO(p: Parameters) extends Bundle {
  val valid     = Bool()
  val owner     = new OwnerTag(p)
  val rob_tag   = new RobTag(p)
  val wrVl      = Bool()
  val wrVtype   = Bool()
  val wrMtype   = Bool()
  val vl        = UInt(log2Ceil(p.rvvVlen + 1).W)
  val vtype     = UInt(p.xlen.W)
  val mtype     = UInt(p.xlen.W)
}

/** Retire-point config commit record published at the entry's own retirement beat. */
class RetiredConfigIO(p: Parameters) extends Bundle {
  val valid     = Bool()
  val rob_tag   = new RobTag(p)
  val wrVl      = Bool()
  val wrVtype   = Bool()
  val wrMtype   = Bool()
  val vl        = UInt(log2Ceil(p.rvvVlen + 1).W)
  val vtype     = UInt(p.xlen.W)
  val mtype     = UInt(p.xlen.W)
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
  // W2-04 lifecycle extension (C027/C040; field names fixed by parent contract — SCore
  // wiring is layered by patch 0404, do not renumber or rename):
  val perCtxPark           = Input(Vec(p.contextCount, Bool()))
  val parkGen              = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))
  val cancelValid          = Input(Vec(p.contextCount, Bool()))
  val cancelGen            = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))
  val pendingLaunch        = Input(Vec(p.contextCount, Bool()))
  val waitBusy             = Input(Vec(p.contextCount, Bool()))
  val regionBusy           = Input(Vec(p.contextCount, Bool()))
  val perCtxParked         = Output(Vec(p.contextCount, Bool()))
  val perCtxCancelled      = Output(Vec(p.contextCount, Bool()))
  val perCtxBackendDrained = Output(Vec(p.contextCount, Bool()))
  val tileIssueFreeze      = Output(Bool())
  val tileQuiescent        = Output(Bool())
  val staleControlRejected = Output(UInt(16.W))
}


/** Live parent-instruction identities indexed by (Context, physical ROB slot).
  * FloatCore returns an OwnerTag and uses the associative lookup; native units
  * return the complete opaque key and validate it by direct slot lookup.
  */
class FlagsParkMap(p: Parameters) extends Module {
  val slotW = math.max(1, log2Ceil(p.retirementBufferSize))
  val nEntries = p.contextCount * p.retirementBufferSize
  private val indexW = math.max(1, log2Ceil(nEntries))
  val io = IO(new Bundle {
    val alloc = Vec(p.instructionLanes, Flipped(Valid(new Bundle {
      val owner = new OwnerTag(p)
      val rob_tag = new RobTag(p)
    })))
    val liveMask = Input(Vec(p.contextCount, UInt(p.retirementBufferSize.W)))
    val generation = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))
    val pendingData = Input(Vec(p.contextCount, UInt(p.retirementBufferSize.W)))
    val trap = Input(Vec(p.contextCount, Bool()))
    val settled = Input(Vec(p.instructionLanes + 2, Valid(UInt(p.nativeRobTagBits.W))))
    val expectedKeys = Output(Vec(nEntries, UInt(p.nativeRobTagBits.W)))
    val live = Output(Vec(nEntries, Bool()))
    val returnable = Output(Vec(nEntries, Bool()))
    val float = new Bundle {
      val req = Input(new OwnerTag(p))
      val resp = Output(Valid(new RobTag(p)))
    }
  })
  val keys = RegInit(VecInit.fill(nEntries)(0.U(p.nativeRobTagBits.W)))
  val allocated = RegInit(VecInit.fill(nEntries)(false.B))
  val flushed = RegInit(VecInit.fill(nEntries)(false.B))
  for (i <- 0 until p.instructionLanes) {
    val slot = io.alloc(i).bits.rob_tag.rob_slot
    val ctx = io.alloc(i).bits.owner.ctx_slot
    val idx = ((ctx << slotW) | slot)(indexW - 1, 0)
    when(io.alloc(i).valid) {
      assert(ctx < p.contextCount.U)
      assert(io.alloc(i).bits.owner.ctx_slot === io.alloc(i).bits.rob_tag.ctx_slot)
      keys(idx) := NativeRobIdentity.pack(p)(io.alloc(i).bits.owner, io.alloc(i).bits.rob_tag)
      allocated(idx) := true.B
    }
  }
  val identities = keys.map(key => NativeRobIdentity.unpack(p)(key))
  for (i <- 0 until nEntries) {
    val ctx = i / p.retirementBufferSize
    val slot = i % p.retirementBufferSize
    val generationMatches = identities(i).ctx_generation === io.generation(ctx)
    io.live(i) := allocated(i) && io.liveMask(ctx)(slot) && generationMatches
    io.returnable(i) := allocated(i) && generationMatches && (io.liveMask(ctx)(slot) || flushed(i))
    val allocating = io.alloc.map(a => a.valid && a.bits.owner.ctx_slot === ctx.U &&
      a.bits.rob_tag.rob_slot === slot.U).reduce(_ || _)
    val settled = io.settled.map(s => s.valid && s.bits === keys(i)).reduce(_ || _)
    when(allocating || !generationMatches || settled) {
      flushed(i) := false.B
    }.elsewhen(io.trap(ctx) && io.liveMask(ctx)(slot) && io.pendingData(ctx)(slot)) {
      // Keep only the unfinished writebacks of the flushed partition eligible
      // for RF settlement; the RF's flushed scoreboard masks architectural writes.
      flushed(i) := true.B
    }
  }
  io.expectedKeys := keys
  val hits = VecInit((0 until nEntries).map { i =>
    io.live(i) && OwnerTag.matches(io.float.req, identities(i).owner)
  })
  assert(PopCount(hits) <= 1.U)
  io.float.resp.valid := hits.asUInt.orR
  io.float.resp.bits := Mux1H(hits, identities.map(_.robTag))
}
