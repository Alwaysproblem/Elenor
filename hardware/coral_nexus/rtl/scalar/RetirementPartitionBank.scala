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

/** Static per-Context partitioned RetirementBuffer bank (`W2-02` / C026, `W2-D01` full mode).
  *
  * Each Context owns an independent `RetirementBuffer` partition (`enqPtr`, `deqPtr`, `nSpace`,
  * `nRetired`, `trapRetired`) so:
  *   - Dispatch enqueues exclusively into `activeCtx`.
  *   - Completion streams (`storeComplete`, `writeDataScalar`, `writeDataFloat`,
  *     `writeDataVector`, `writeDataTile`, `vmeMtypeWb`, `fault`) demux by `OwnerTag` / `RobTag`
  *     and validate `ctx_generation` + `slot_generation`.
  *   - Head stall in Context A never blocks retirement in Context B (`RET-03` / `T-09`).
  *   - Trap flush (`trapRetired`) clears ONLY the faulting Context partition (`RET-06`).
  */
class RetirementPartitionBankIO(p: Parameters) extends Bundle {
    val activeCtx     = Input(UInt(p.ctxBits.W))
    val perCtxGen     = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))
    val perCtxReset   = Input(Vec(p.contextCount, Bool()))
    val dispatchOpSeq = Input(Vec(p.instructionLanes, UInt(p.ownerOpSeqBits.W)))

    val inst            = Input(Vec(p.instructionLanes, Decoupled(new FetchInstruction(p))))
    val targets         = Input(Vec(p.instructionLanes, UInt(p.programCounterBits.W)))
    val jalrTargets     = Input(Vec(p.instructionLanes, UInt(p.programCounterBits.W)))
    val jump            = Input(Vec(p.instructionLanes, Bool()))
    val branch          = Input(Vec(p.instructionLanes, Bool()))
    val storeComplete   = Input(Valid(new OwnerTag(p)))
    val writeAddrScalar = Input(Vec(p.instructionLanes, new RegfileWriteAddrIO(p)))
    val writeDataScalar = Input(Vec(p.instructionLanes + 2, Valid(new OwnedRegfileWriteDataIO(p))))
    val writeAddrFloat  = Option.when(p.enableFloat)(Input(new RegfileWriteAddrIO(p)))
    val writeDataFloat  = Option.when(p.enableFloat)(Input(Vec(2, Valid(new OwnedRegfileWriteDataIO(p)))))
    val writeAddrVector =
      Option.when(p.enableRvv)(Input(Vec(p.instructionLanes, new RegfileWriteAddrIO(p))))
    val writeDataVector =
      Option.when(p.enableRvv)(Input(Vec(p.rvvRetireLanes, Valid(new OwnedVectorWriteDataIO(p)))))
    val fault = Input(Valid(new FaultManagerOutput(p)))

    val isVector      = Option.when(p.enableRvv)(Input(Vec(p.instructionLanes, Bool())))
    val isTile        = Option.when(p.enableVme)(Input(Vec(p.instructionLanes, Bool())))
    val clearVstart   = Option.when(p.enableRvv)(Output(Bool()))
    val writeDataTile = Option.when(p.enableVme)(Input(Valid(new OwnedTileWriteDataIO(p))))
    val vmeMtypeWb =
      Option.when(p.enableVme)(Input(Vec(p.instructionLanes, Valid(new OwnedMtypeWriteDataIO(p)))))
    val flagsDelta = Input(Valid(new ParkedFlagsDeltaIO(p)))
    val csrPark    = Input(Valid(new ParkedCsrDeltaIO(p)))
    val perCtxRetiredDelta =
      Output(Vec(p.contextCount, Vec(p.retirementLanes, new RetiredFlagsDeltaIO(p))))
    // Selected-Context projections for DispatchV2 compatibility.
    val enqPtr      = Output(UInt(math.max(1, log2Ceil(p.retirementBufferSize)).W))
    val enqSlotGen  = Output(Vec(p.instructionLanes, UInt(p.slotGenerationBits.W)))
    val nSpace      = Output(UInt(log2Ceil(p.retirementBufferSize + 1).W))
    val nRetired    = Output(UInt(log2Ceil(p.retirementLanes + 1).W))
    val empty       = Output(Bool())
    val trapPending = Output(Bool())
    val trapRetired = Output(Bool())

    // Per-Context independent outputs.
    val perCtxSpace       = Output(Vec(p.contextCount, UInt(log2Ceil(p.retirementBufferSize + 1).W)))
    val perCtxEmpty       = Output(Vec(p.contextCount, Bool()))
    val perCtxNRetired    = Output(Vec(p.contextCount, UInt(log2Ceil(p.retirementLanes + 1).W)))
    val perCtxDeqReady    = Output(Vec(p.contextCount, UInt(log2Ceil(p.retirementLanes + 1).W)))
    val perCtxTrapPending = Output(Vec(p.contextCount, Bool()))
    val perCtxTrapRetired = Output(Vec(p.contextCount, Bool()))
    val staleRejected     = Output(UInt(16.W))

    val debug = Option.when(p.shouldExposeDebugPorts)(Output(new RetirementBufferDebugIO(p)))
    val perCtxDebug = Option.when(p.shouldExposeDebugPorts)(
      Output(Vec(p.contextCount, new RetirementBufferDebugIO(p)))
    )
}

class RetirementPartitionBank(p: Parameters, mini: Boolean = false) extends Module {
  val robSlotWidth = math.max(1, log2Ceil(p.retirementBufferSize))
  val io           = IO(new RetirementPartitionBankIO(p))

  val effActiveCtx = if (p.contextCount == 1) 0.U(1.W) else io.activeCtx

  val partitions = Seq.tabulate(p.contextCount) { c =>
    withReset(reset.asBool || io.perCtxReset(c)) {
      Module(new RetirementBuffer(p, mini = mini, ctxSlotId = c))
    }
  }

  def selCtx[T <: Data](items: Seq[T], idx: UInt): T = {
    if (items.length == 1) items.head else VecInit(items)(idx)
  }

  for (c <- 0 until p.contextCount) {
    val part  = partitions(c)
    val isSel = (effActiveCtx === c.U) && !io.perCtxReset(c)

    part.io.ctxGeneration := io.perCtxGen(c)
    part.io.dispatchOpSeq := io.dispatchOpSeq

    for (i <- 0 until p.instructionLanes) {
      part.io.inst(i).valid       := isSel && io.inst(i).valid
      part.io.inst(i).ready       := isSel && io.inst(i).ready
      part.io.inst(i).bits        := io.inst(i).bits
      part.io.targets(i)          := io.targets(i)
      part.io.jalrTargets(i)      := io.jalrTargets(i)
      part.io.jump(i)             := isSel && io.jump(i)
      part.io.branch(i)           := isSel && io.branch(i)
      part.io.writeAddrScalar(i).valid := isSel && io.writeAddrScalar(i).valid
      part.io.writeAddrScalar(i).addr  := io.writeAddrScalar(i).addr
    }

    part.io.storeComplete.valid := io.storeComplete.valid && (io.storeComplete.bits.ctx_slot === c.U)
    part.io.storeComplete.bits  := io.storeComplete.bits

    for (j <- 0 until p.instructionLanes + 2) {
      part.io.writeDataScalar(j).valid := io.writeDataScalar(j).valid &&
        (io.writeDataScalar(j).bits.owner.ctx_slot === c.U)
      part.io.writeDataScalar(j).bits := io.writeDataScalar(j).bits
    }

    if (p.enableFloat) {
      part.io.writeAddrFloat.get.valid := isSel && io.writeAddrFloat.get.valid
      part.io.writeAddrFloat.get.addr  := io.writeAddrFloat.get.addr
      for (j <- 0 until 2) {
        part.io.writeDataFloat.get(j).valid := io.writeDataFloat.get(j).valid &&
          (io.writeDataFloat.get(j).bits.owner.ctx_slot === c.U)
        part.io.writeDataFloat.get(j).bits := io.writeDataFloat.get(j).bits
      }
    }

    if (p.enableRvv) {
      for (i <- 0 until p.instructionLanes) {
        part.io.writeAddrVector.get(i).valid := isSel && io.writeAddrVector.get(i).valid
        part.io.writeAddrVector.get(i).addr  := io.writeAddrVector.get(i).addr
        part.io.isVector.get(i)              := isSel && io.isVector.get(i)
      }
      for (j <- 0 until p.rvvRetireLanes) {
        part.io.writeDataVector.get(j).valid := io.writeDataVector.get(j).valid &&
          (io.writeDataVector.get(j).bits.rob_tag.ctx_slot === c.U)
        part.io.writeDataVector.get(j).bits := io.writeDataVector.get(j).bits
      }
    }

    if (p.enableVme) {
      for (i <- 0 until p.instructionLanes) {
        part.io.isTile.get(i) := isSel && io.isTile.get(i)
        part.io.vmeMtypeWb.get(i).valid := io.vmeMtypeWb.get(i).valid &&
          (io.vmeMtypeWb.get(i).bits.rob_tag.ctx_slot === c.U)
        part.io.vmeMtypeWb.get(i).bits := io.vmeMtypeWb.get(i).bits
      }
      part.io.writeDataTile.get.valid := io.writeDataTile.get.valid &&
        (io.writeDataTile.get.bits.rob_tag.ctx_slot === c.U)
      part.io.writeDataTile.get.bits := io.writeDataTile.get.bits
    }

    // W2-04 ordered deltas: route by identity to the owning partition.
    part.io.flagsDelta.valid := io.flagsDelta.valid &&
      (io.flagsDelta.bits.rob_tag.ctx_slot === c.U)
    part.io.flagsDelta.bits := io.flagsDelta.bits
    part.io.csrPark.valid := io.csrPark.valid && (io.csrPark.bits.owner.ctx_slot === c.U)
    part.io.csrPark.bits := io.csrPark.bits

    io.perCtxRetiredDelta(c) := part.io.retiredDelta

    val isRvvFault = io.fault.bits.is_rvv.getOrElse(false.B)
    val faultSlot  = Mux(
      isRvvFault,
      io.fault.bits.rob_tag.map(_.ctx_slot).getOrElse(0.U),
      io.fault.bits.owner.ctx_slot
    )
    part.io.fault.valid := io.fault.valid && (faultSlot === c.U)
    part.io.fault.bits  := io.fault.bits

    io.perCtxSpace(c)       := part.io.nSpace
    io.perCtxEmpty(c)       := part.io.empty
    io.perCtxNRetired(c)    := part.io.nRetired
    io.perCtxDeqReady(c)    := part.io.deqReadyOut
    io.perCtxTrapPending(c) := part.io.trapPending
    io.perCtxTrapRetired(c) := part.io.trapRetired
  }

  io.enqPtr      := selCtx(partitions.map(_.io.enqPtr), effActiveCtx)
  io.enqSlotGen  := selCtx(partitions.map(_.io.enqSlotGen), effActiveCtx)
  io.nSpace      := selCtx(partitions.map(_.io.nSpace), effActiveCtx)
  io.nRetired    := selCtx(partitions.map(_.io.nRetired), effActiveCtx)
  io.empty       := selCtx(partitions.map(_.io.empty), effActiveCtx)
  io.trapPending := selCtx(partitions.map(_.io.trapPending), effActiveCtx)
  io.trapRetired := selCtx(partitions.map(_.io.trapRetired), effActiveCtx)

  if (p.enableRvv) {
    io.clearVstart.get := partitions.map(_.io.clearVstart.get).reduce(_ || _)
  }

  val outOfRangeStore = io.storeComplete.valid && (io.storeComplete.bits.ctx_slot >= p.contextCount.U)
  val outOfRangeReg   = RegInit(0.U(16.W))
  outOfRangeReg := outOfRangeReg + outOfRangeStore.asUInt
  io.staleRejected := partitions.map(_.io.staleCompletionRejected).reduce(_ + _) + outOfRangeReg

  io.perCtxDebug.foreach { pdbg =>
    for (c <- 0 until p.contextCount) {
      pdbg(c) := partitions(c).io.debug.get
    }
  }
  io.debug.foreach { dbg =>
    val anyRetiring = VecInit(partitions.map(pt => pt.io.debug.get.inst(0).valid))
    val chosenDbg   = Mux(anyRetiring.asUInt.orR, PriorityEncoder(anyRetiring), effActiveCtx)
    dbg := selCtx(partitions.map(_.io.debug.get), chosenDbg)
  }
}
