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

/** Per-Context banked scalar GPR + scoreboard wrapper (`W2-02` / C026).
  *
  * Instantiates `p.contextCount` full `Regfile` banks so every Context owns isolated
  * `regfile`, `scoreboard`, `flushed_scoreboard`, and `linkPort` state:
  *   - Decode-cycle reads/marks (`readAddr`, `readSet`, `writeAddr`, `busAddr`) apply ONLY to
  *     `activeCtx`.
  *   - Execute-cycle `readData` comes from `RegNext(activeCtx)` matching `Regfile`'s 1-cycle
  *     registered read response.
  *   - Writebacks (`writeData`) route by `OwnerTag.ctx_slot` and are accepted iff
  *     `OwnerTag.ctx_generation === perCtxGen(slot)`; stale-generation writebacks are rejected
  *     and counted without touching any architectural register or scoreboard (INV-15).
  */
class BankedRf(p: Parameters, extraWritePorts: Int = 2) extends Module {
  val totalWritePorts = p.instructionLanes + extraWritePorts
  val io              = IO(new Bundle {
    val activeCtx   = Input(UInt(p.ctxBits.W))
    val perCtxGen   = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))
    val perCtxReset = Input(Vec(p.contextCount, Bool()))
    val perCtxFlush = Input(Vec(p.contextCount, Bool()))

    // Decode cycle (selected Context).
    val readAddr  = Vec(p.instructionLanes * 2, new RegfileReadAddrIO(p))
    val readSet   = Vec(p.instructionLanes * 2, new RegfileReadSetIO(p))
    val writeAddr = Vec(p.instructionLanes, new RegfileWriteAddrIO(p))
    val busAddr   = Vec(p.instructionLanes, Input(new RegfileBusAddrIO(p)))
    val target    = Vec(p.instructionLanes, new RegfileBranchTargetIO(p))
    val linkPort  = new RegfileLinkPortIO(p)
    val busPort   = new RegfileBusPortIO(p)

    val debugBusPort = new Bundle {
      val idx  = Input(UInt(log2Ceil(p.scalarRegCount).W))
      val data = Output(UInt(p.xlen.W))
    }
    val debugWriteValid = Input(Bool())

    // Execute cycle.
    val readData = Vec(p.instructionLanes * 2, new RegfileReadDataIO(p))
    val writeData = Vec(
      totalWritePorts,
      new Bundle {
        val valid = Input(Bool())
        val bits  = Input(new OwnedRegfileWriteDataIO(p))
      }
    )
    val writeMask = Vec(totalWritePorts, new Bundle { val valid = Input(Bool()) })

    val scoreboard = new Bundle {
      val regd = Output(UInt(p.scalarRegCount.W))
      val comb = Output(UInt(p.scalarRegCount.W))
    }
    val perCtxScoreboardRegd = Output(Vec(p.contextCount, UInt(p.scalarRegCount.W)))
    val perCtxScoreboardComb = Output(Vec(p.contextCount, UInt(p.scalarRegCount.W)))
    val staleWriteRejected   = Output(UInt(16.W))
  })

  val effActiveCtx = if (p.contextCount == 1) 0.U(1.W) else io.activeCtx
  val execCtx      = RegNext(effActiveCtx, 0.U(p.ctxBits.W))

  val banks = Seq.tabulate(p.contextCount) { b =>
    withReset(reset.asBool || io.perCtxReset(b)) {
      Module(new Regfile(p))
    }
  }

  def selCtx[T <: Data](items: Seq[T], idx: UInt): T = {
    if (items.length == 1) items.head else VecInit(items)(idx)
  }

  val staleHits = Wire(Vec(totalWritePorts, Bool()))
  for (j <- 0 until totalWritePorts) {
    val slot      = io.writeData(j).bits.owner.ctx_slot
    val slotInRng = if (p.contextCount == 1) (slot === 0.U) else (slot < p.contextCount.U)
    val liveGen   = if (p.contextCount == 1) io.perCtxGen(0) else io.perCtxGen(Mux(slotInRng, slot, 0.U))
    val genMatch  = slotInRng && (io.writeData(j).bits.owner.ctx_generation === liveGen)
    staleHits(j) := io.writeData(j).valid && !io.writeMask(j).valid && !genMatch
  }
  val staleCounter = RegInit(0.U(16.W))
  staleCounter          := staleCounter + PopCount(staleHits)
  io.staleWriteRejected := staleCounter

  for (b <- 0 until p.contextCount) {
    val bank  = banks(b)
    val isSel = (effActiveCtx === b.U)

    for (i <- 0 until p.instructionLanes * 2) {
      bank.io.readAddr(i).valid := isSel && io.readAddr(i).valid
      bank.io.readAddr(i).addr  := io.readAddr(i).addr
      bank.io.readSet(i).valid  := isSel && io.readSet(i).valid
      bank.io.readSet(i).value  := io.readSet(i).value
    }
    for (i <- 0 until p.instructionLanes) {
      bank.io.writeAddr(i).valid := isSel && io.writeAddr(i).valid
      bank.io.writeAddr(i).addr  := io.writeAddr(i).addr
      bank.io.busAddr(i).valid   := isSel && io.busAddr(i).valid
      bank.io.busAddr(i).immen   := io.busAddr(i).immen
      bank.io.busAddr(i).immed   := io.busAddr(i).immed
    }

    bank.io.debugBusPort.idx := io.debugBusPort.idx
    bank.io.debugWriteValid  := isSel && io.debugWriteValid
    bank.io.pipelineFlush    := io.perCtxFlush(b)

    for (j <- 0 until totalWritePorts) {
      val ownerMatch = (io.writeData(j).bits.owner.ctx_slot === b.U) &&
        (io.writeData(j).bits.owner.ctx_generation === io.perCtxGen(b))
      bank.io.writeData(j).valid      := io.writeData(j).valid && ownerMatch && !io.perCtxReset(b)
      bank.io.writeData(j).bits.addr  := io.writeData(j).bits.addr
      bank.io.writeData(j).bits.data  := io.writeData(j).bits.data
      // Single-bank Regfile stores per-bank GPRs/scoreboard; owner routing was validated above.
      bank.io.writeMask(j).valid      := io.writeMask(j).valid
    }

    io.perCtxScoreboardRegd(b) := bank.io.scoreboard.regd
    io.perCtxScoreboardComb(b) := bank.io.scoreboard.comb
  }

  io.target            := selCtx(banks.map(_.io.target), effActiveCtx)
  io.linkPort          := selCtx(banks.map(_.io.linkPort), effActiveCtx)
  io.busPort           := selCtx(banks.map(_.io.busPort), effActiveCtx)
  io.debugBusPort.data := selCtx(banks.map(_.io.debugBusPort.data), effActiveCtx)
  io.readData          := selCtx(banks.map(_.io.readData), execCtx)
  io.scoreboard.regd   := selCtx(banks.map(_.io.scoreboard.regd), effActiveCtx)
  io.scoreboard.comb   := selCtx(banks.map(_.io.scoreboard.comb), effActiveCtx)
}

/** Per-Context banked scalar FPR + scoreboard wrapper (`W2-02` / C026). */
class BankedFRegfile(p: Parameters, n_read: Int, n_write: Int) extends Module {
  val io = IO(new Bundle {
    val activeCtx    = Input(UInt(p.ctxBits.W))
    val floatReadCtx = Input(UInt(p.ctxBits.W))
    val perCtxGen    = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))
    val perCtxReset  = Input(Vec(p.contextCount, Bool()))
    val perCtxFlush  = Input(Vec(p.contextCount, Bool()))

    val read_ports     = Vec(n_read, new FRegfileRead(p))
    val write_ports    = Vec(n_write, new OwnedFRegfileWrite(p))
    val dm_write_valid = Input(Bool())

    val scoreboard_set   = Input(UInt(p.floatRegCount.W))
    val scoreboard       = Output(UInt(p.floatRegCount.W))
    val perCtxScoreboard = Output(Vec(p.contextCount, UInt(p.floatRegCount.W)))
    val exception        = Output(Bool())

    val busPort            = new RegfileBusPortIO(p)
    val busPortAddr        = Input(UInt(log2Ceil(p.floatRegCount).W))
    val staleWriteRejected = Output(UInt(16.W))
  })

  val effActiveCtx    = if (p.contextCount == 1) 0.U(1.W) else io.activeCtx
  val effFloatReadCtx = if (p.contextCount == 1) 0.U(1.W) else io.floatReadCtx

  val banks = Seq.tabulate(p.contextCount) { b =>
    withReset(reset.asBool || io.perCtxReset(b)) {
      Module(new FRegfile(p, n_read, n_write))
    }
  }

  def selCtx[T <: Data](items: Seq[T], idx: UInt): T = {
    if (items.length == 1) items.head else VecInit(items)(idx)
  }

  val staleHits = Wire(Vec(n_write, Bool()))
  for (w <- 0 until n_write) {
    val slot      = io.write_ports(w).owner.ctx_slot
    val slotInRng = if (p.contextCount == 1) (slot === 0.U) else (slot < p.contextCount.U)
    val liveGen   = if (p.contextCount == 1) io.perCtxGen(0) else io.perCtxGen(Mux(slotInRng, slot, 0.U))
    val genMatch  = slotInRng && (io.write_ports(w).owner.ctx_generation === liveGen)
    staleHits(w) := io.write_ports(w).valid && !genMatch
  }
  val staleCounter = RegInit(0.U(16.W))
  staleCounter          := staleCounter + PopCount(staleHits)
  io.staleWriteRejected := staleCounter

  for (b <- 0 until p.contextCount) {
    val bank      = banks(b)
    val isDispSel = (effActiveCtx === b.U)
    val isFpuSel  = (effFloatReadCtx === b.U)

    for (r <- 0 until n_read) {
      val useDispCtx = (r == 3)
      val portSel    = if (useDispCtx) isDispSel else isFpuSel
      bank.io.read_ports(r).valid := portSel && io.read_ports(r).valid
      bank.io.read_ports(r).addr  := io.read_ports(r).addr
    }

    for (w <- 0 until n_write) {
      val ownerMatch = (io.write_ports(w).owner.ctx_slot === b.U) &&
        (io.write_ports(w).owner.ctx_generation === io.perCtxGen(b))
      bank.io.write_ports(w).valid := io.write_ports(w).valid && ownerMatch && !io.perCtxReset(b)
      bank.io.write_ports(w).addr  := io.write_ports(w).addr
      bank.io.write_ports(w).data  := io.write_ports(w).data
      // Single-bank FRegfile stores per-bank FPRs/scoreboard; owner routing was validated above.
    }

    bank.io.dm_write_valid := isDispSel && io.dm_write_valid
    bank.io.pipelineFlush  := io.perCtxFlush(b)
    bank.io.scoreboard_set := Mux(isDispSel, io.scoreboard_set, 0.U)
    bank.io.busPortAddr    := io.busPortAddr
    io.perCtxScoreboard(b) := bank.io.scoreboard
  }

  for (r <- 0 until n_read) {
    val idx = if (r == 3) effActiveCtx else effFloatReadCtx
    io.read_ports(r).data := selCtx(banks.map(_.io.read_ports(r).data), idx)
  }

  io.scoreboard := selCtx(banks.map(_.io.scoreboard), effActiveCtx)
  io.busPort    := selCtx(banks.map(_.io.busPort), effActiveCtx)
  io.exception  := banks.map(_.io.exception).reduce(_ || _)
}
