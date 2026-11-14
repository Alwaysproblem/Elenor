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
import common._

/** C028 / W2-03 frozen directed-fetch frontend identity types.
  *
  * A fetch transaction is identified ONLY by `CtxFetchKey` (slot + generation +
  * epoch + reqid) — never by PC/address. Responses carry no address: the ledger
  * inside `PerCtxFetchMux` supplies the line PC at delivery time, so two requests
  * for the same line under different epochs (same-PC ABA) can never alias.
  */
class CtxFetchKey(p: Parameters) extends Bundle {
  val ctx_slot       = UInt(p.ctxBits.W)
  val ctx_generation = UInt(p.ctxGenerationBits.W)
  val fetch_epoch    = UInt(p.ownerOpSeqBits.W)
  val fetch_reqid    = UInt(p.ownerOpSeqBits.W)
}

class CtxFetchRequest(p: Parameters) extends Bundle {
  val key  = new CtxFetchKey(p)
  val addr = UInt(p.fetchAddrBits.W) // physical aligned line address
}

class CtxFetchResponse(p: Parameters) extends Bundle {
  val key   = new CtxFetchKey(p)
  val data  = UInt(p.fetchDataBits.W)
  val fault = Bool()
}

/** Native Coral line predecode, shared by every Context's fetch bank. */
class PredecodeOutput(p: Parameters) extends Bundle {
  val insts     = Vec(p.fetchInstrSlots, new FetchInstruction(p))
  val count     = UInt(4.W)
  val nextPc    = UInt(p.instructionBits.W)
  val hasJumped = Bool()
}

object FetchPredecode {
  def PredictJump(p: Parameters)(addr: UInt, inst: UInt): ValidIO[UInt] = {
    assert(p.instructionBits == 32)
    val jal    = inst === BitPat("b????????????????????_?????_1101111")
    val immjal = Cat(Fill(12, inst(31)), inst(19, 12), inst(20), inst(30, 21), 0.U(1.W))
    val bxx    = inst === BitPat("b???????_?????_?????_???_?????_1100011") &&
      inst(31) && inst(14, 13) =/= 1.U
    val immbxx = Cat(Fill(20, inst(31)), inst(7), inst(30, 25), inst(11, 8), 0.U(1.W))
    val immed  = Mux(inst(2), immjal, immbxx)

    val valid  = jal || bxx
    val target = addr + immed

    MakeValid(valid, target)
  }

  def Predecode(p: Parameters)(addr: UInt, inst: Vec[UInt]): PredecodeOutput = {
    val lsb = log2Ceil(p.fetchDataBits / 8)
    assert((p.fetchDataBits == 128 && lsb == 4) || (p.fetchDataBits == 256 && lsb == 5))
    val baseAddr  = addr(p.fetchAddrBits - 1, lsb)
    val startElem = addr(lsb - 1, lsb - log2Ceil(p.fetchInstrSlots))

    val insts = ShiftVectorRight(inst, startElem)
    val addrs = VecInit.tabulate(p.fetchInstrSlots)(i => addr + (i * 4).U)

    val branchTargets = VecInit.tabulate(p.fetchInstrSlots)(i => PredictJump(p)(addrs(i), insts(i)))

    val validsIn = VecInit.tabulate(p.fetchInstrSlots)(i => i.U < p.fetchInstrSlots.U - startElem)
    val jumped   = VecInit.tabulate(p.fetchInstrSlots)(i => validsIn(i) && branchTargets(i).valid)
    val firstJumpOH = VecInit(PriorityEncoderOH(jumped))

    // Have we jumped before the instruction i
    val hasJumpedBefore = VecInit(jumped.scan(false.B)(_ || _).take(p.fetchInstrSlots))

    val validsOut = VecInit.tabulate(p.fetchInstrSlots)(i => validsIn(i) && !hasJumpedBefore(i))

    val nextFetchPc = MuxUpTo1H(
      Cat(baseAddr + 1.U, 0.U(lsb.W)),
      (0 until p.fetchInstrSlots).map(i => firstJumpOH(i) -> branchTargets(i).bits)
    )

    MakeWireBundle[PredecodeOutput](
      new PredecodeOutput(p),
      _.insts -> VecInit.tabulate(p.fetchInstrSlots)(i =>
        MakeWireBundle[FetchInstruction](
          new FetchInstruction(p),
          _.addr    -> addrs(i),
          _.inst    -> insts(i),
          _.brchFwd -> jumped(i)
        )
      ),
      _.count     -> validsOut.count(x => x),
      _.nextPc    -> nextFetchPc,
      _.hasJumped -> jumped.reduce(_ || _)
    )
  }
}

/** Frozen C028 frontend interface (named class so integration harnesses and specs can
  * hold real references; see the C028 contract and W2 design §3.5).
  *
  * `fetchServices` counts ACCEPTED fetch requests (request-channel fires), matching
  * the frozen metadata definition — not buffered-line deliveries.
  */
class PerCtxFetchIO(p: Parameters, maxOutstanding: Int) extends Bundle {
  val accepted     = Input(Vec(p.contextCount, Bool()))
  val programReady = Input(Vec(p.contextCount, Bool()))
  val parked       = Input(Vec(p.contextCount, Bool()))
  val cancel       = Input(Vec(p.contextCount, Bool()))
  val generation   = Input(Vec(p.contextCount, UInt(p.ctxGenerationBits.W)))

  /** Entry initialization pulse of a newly accepted Task. A start on a safely reused
    * slot must NOT discard old transaction identities or credits: it only bumps the
    * owning epoch (invalidating undelivered payload) and re-arms fetching at `bits`.
    */
  val start    = Input(Vec(p.contextCount, Valid(UInt(p.fetchAddrBits.W))))
  /** Resolved-branch / LSU-flush redirect. Advances only the owning Context's epoch
    * and flushes only that Context's instruction buffer.
    */
  val redirect = Input(Vec(p.contextCount, Valid(UInt(p.fetchAddrBits.W))))

  val request  = Decoupled(new CtxFetchRequest(p))
  val response = Flipped(Decoupled(new CtxFetchResponse(p)))

  val inst = Vec(p.contextCount, Vec(p.instructionLanes, Decoupled(new FetchInstruction(p))))

  /** Next unaccepted PC: the live buffer head while instructions are buffered, else
    * the saved next-unaccepted PC (explicit start/redirect target, the sequential PC
    * after the last accepted lane, or the predecoded jump target). Never a stale
    * ledger entry and never the speculatively advanced fetch line address.
    */
  val pc     = Output(Vec(p.contextCount, UInt(p.fetchAddrBits.W)))
  val fault  = Output(Vec(p.contextCount, Valid(UInt(p.fetchAddrBits.W))))
  val epoch  = Output(Vec(p.contextCount, UInt(p.ownerOpSeqBits.W)))
  val outstanding = Output(Vec(p.contextCount, UInt(math.max(1, log2Ceil(maxOutstanding + 1)).W)))
  val staleRejected   = Output(Vec(p.contextCount, UInt(32.W)))
  val creditsReturned = Output(Vec(p.contextCount, UInt(32.W)))
  val fetchServices   = Output(Vec(p.contextCount, UInt(32.W)))
}

/** Directed per-Context fetch frontend (C028 / W2-03; W2 design §3.5, FET-01/02/03/05,
  * CTX-09, SCH-02).
  *
  * Banked replacement of the retired single-stream `FetchControl` + `Fetcher` +
  * `InstructionBuffer` assembly for the multi-Context core:
  *   - per-Context fetch PC, predecode (shared `FetchPredecode`), instruction buffer
  *     and identity-keyed reorder ledger (`CtxFetchKey`: slot + generation + epoch +
  *     reqid — never a PC match);
  *   - one shared tagged request channel arbitrated with the existing
  *     `ContextIssueArbiter`; while a request is staged the arbiter only sees the
  *     staged owner, so the rotating pointer advances from the ACTUAL staged owner on
  *     its fire and a Context whose responses are delayed cannot prevent peer service
  *     (FET-03);
  *   - bounded per-Context in-flight transactions: `maxOutstanding` is the explicit
  *     constructor bound (any positive integer — ring membership is modulo-free) the
  *     integrator takes from the reopened C019 constraint
  *     `constraints.fetch_outstanding_per_context` (via `C028TargetParams`);
  *   - directed invalidation: branch redirect, task cancel and entry (re)start bump
  *     ONLY the owning Context's epoch and flush ONLY that Context's instruction
  *     buffer; a generation-only change (reuse edge without a pulse yet) also flushes
  *     and disarms that Context until an explicit start. In-flight old-identity
  *     transactions still occupy their credit until their response completes the
  *     handshake (exact credit return, CTX-09); the payload is never written;
  *   - responses may be delayed, out of order, duplicated or unknown: only a response
  *     matching a live awaiting ledger entry (or arriving in the very beat its request
  *     is accepted — the enqueue bypass) consumes it and returns its credit; anything
  *     else is counted in `staleRejected` and never fabricates or frees credit.
  *
  * The physical instruction-memory port bridge (registering the accepted request key
  * onto the existing fixed-latency IBus) belongs to the core integration, not here.
  */
class PerCtxFetchMux(p: Parameters, val maxOutstanding: Int) extends Module {
  require(maxOutstanding > 0, "maxOutstanding must be an explicit positive bound")

  val io = IO(new PerCtxFetchIO(p, maxOutstanding))

  private val nCtx      = p.contextCount
  private val ptrW      = math.max(1, log2Ceil(maxOutstanding))
  private val ctrW      = log2Ceil(maxOutstanding + 1)
  private val lineLsb   = log2Ceil(p.fetchDataBits / 8)
  private val bufWindow = p.fetchInstrSlots * 2

  private def alignLine(pc: UInt): UInt  = Cat(pc(p.fetchAddrBits - 1, lineLsb), 0.U(lineLsb.W))
  private def nextLinePc(pc: UInt): UInt = alignLine(pc) + (p.fetchDataBits / 8).U
  private def wrapInc(x: UInt): UInt =
    if (maxOutstanding == 1) 0.U else Mux(x === (maxOutstanding - 1).U, 0.U, x + 1.U)

  // ---------------------------------------------------------------- ledger state
  private class LedgerEntry extends Bundle {
    val key       = new CtxFetchKey(p)
    val pc        = UInt(p.fetchAddrBits.W) // unaligned fetch PC (predecode startElem)
    val respValid = Bool()
    val data      = UInt(p.fetchDataBits.W)
    val fault     = Bool()
  }

  private val entries = RegInit(
    VecInit.fill(nCtx)(VecInit.fill(maxOutstanding)(0.U.asTypeOf(new LedgerEntry)))
  )
  private val headPtr = RegInit(VecInit.fill(nCtx)(0.U(ptrW.W)))
  private val qLen    = RegInit(VecInit.fill(nCtx)(0.U(ctrW.W))) // entries in delivery order
  private val credits = RegInit(VecInit.fill(nCtx)(0.U(ctrW.W))) // in-flight transactions

  private val epochReg = RegInit(VecInit.fill(nCtx)(0.U(p.ownerOpSeqBits.W)))
  private val reqidReg = RegInit(VecInit.fill(nCtx)(0.U(p.ownerOpSeqBits.W)))
  private val genReg   = RegInit(VecInit.fill(nCtx)(0.U(p.ctxGenerationBits.W)))
  private val fetchPc  = RegInit(VecInit.fill(nCtx)(MakeInvalid(UInt(p.fetchAddrBits.W))))
  private val started  = RegInit(VecInit.fill(nCtx)(false.B))
  private val cancelled = RegInit(VecInit.fill(nCtx)(false.B))
  private val faulted  = RegInit(VecInit.fill(nCtx)(false.B))
  private val faultReg = RegInit(VecInit.fill(nCtx)(MakeInvalid(UInt(p.fetchAddrBits.W))))
  private val pcSaved  = RegInit(VecInit.fill(nCtx)(0.U(p.fetchAddrBits.W)))
  private val nStale   = RegInit(VecInit.fill(nCtx)(0.U(32.W)))
  private val nCredit  = RegInit(VecInit.fill(nCtx)(0.U(32.W)))
  private val nService = RegInit(VecInit.fill(nCtx)(0.U(32.W)))

  // Ring membership for ARBITRARY positive bounds: distance is computed with an
  // explicit wrap correction, never a power-of-two truncation.
  private def ringDist(c: Int, i: Int): UInt = {
    val fwd = i.U((ptrW + 1).W) +% (maxOutstanding.U((ptrW + 1).W) - headPtr(c))
    Mux(fwd >= maxOutstanding.U, fwd - maxOutstanding.U, fwd)
  }
  private val liveAt = Seq.tabulate(nCtx, maxOutstanding) { (c, i) =>
    ringDist(c, i) < qLen(c)
  }
  private val liveVec = VecInit.tabulate(nCtx, maxOutstanding)((c, i) => liveAt(c)(i))

  // ------------------------------------------------------ per-Context instruction buffers
  private val buffers = Seq.tabulate(nCtx) { c =>
    Module(new InstructionBuffer(new FetchInstruction(p), p.fetchInstrSlots, bufWindow))
  }

  // ---------------------------------------------------------------- request staging
  // The single outstanding offer. Bits are registers, so an offered request stays bit
  // stable while valid && !ready, including across redirect/cancel/reuse of its owner;
  // an obsolete offer completes and its response is later rejected as stale. A new
  // offer is staged only while the stage is empty (one bubble after a fire), so the
  // grant never races the fire.
  private val stageValid = RegInit(false.B)
  private val stageOwner = RegInit(0.U(p.ctxBits.W))
  private val stageReq   = RegInit(0.U.asTypeOf(new CtxFetchRequest(p)))
  private val stagePc    = RegInit(0.U(p.fetchAddrBits.W))

  io.request.valid := stageValid
  io.request.bits  := stageReq
  private val stageFire = stageValid && io.request.ready

  // Per-cycle event strobes. Each is written by exactly one producer and only read
  // afterwards, so no wire ever reads its own (partially) connected value.
  private val enqInc     = WireDefault(VecInit.fill(nCtx)(0.U(1.W)))
  private val deqInc     = WireDefault(VecInit.fill(nCtx)(0.U(1.W)))
  private val creditDec  = WireDefault(VecInit.fill(nCtx)(0.U(1.W)))
  private val respStale  = WireDefault(VecInit.fill(nCtx)(false.B)) // unknown/dup packet
  private val dropStale  = WireDefault(VecInit.fill(nCtx)(false.B)) // stale commit drop
  private val creditInc  = WireDefault(VecInit.fill(nCtx)(false.B))
  private val serviceInc = WireDefault(VecInit.fill(nCtx)(false.B))

  private val genMismatch = VecInit.tabulate(nCtx)(c => genReg(c) =/= io.generation(c))
  private val startedNow  = VecInit.tabulate(nCtx)(c => started(c) && !cancelled(c))
  // Armed only under the generation the entry start sampled: a generation-only reuse
  // edge disarms until the explicit start of the new Task.
  private val armed = VecInit.tabulate(nCtx)(c => startedNow(c) && !genMismatch(c))

  // A Context may start fetching only after its Task was accepted AND the program
  // install reported ready; never inferred from reset or Context selection. Park
  // inhibits new work only; a fetch fault stops requests until a redirect clears it.
  private val wantOffer = VecInit.tabulate(nCtx) { c =>
    armed(c) && io.accepted(c) && io.programReady(c) && !io.parked(c) &&
      !faulted(c) && fetchPc(c).valid &&
      !io.start(c).valid && !io.redirect(c).valid && !io.cancel(c) &&
      (qLen(c) + Mux(stageFire && (stageOwner === c.U), 1.U, 0.U)) < maxOutstanding.U
  }

  // While a request is staged, the arbiter sees ONLY the staged owner: on its fire the
  // rotating pointer advances from the actual staged owner (never from a grant whose
  // owner may already be capacity-blocked).
  private val stagedOneHot = VecInit.tabulate(nCtx)(c => stageValid && (stageOwner === c.U))
  private val arb          = Module(new ContextIssueArbiter(p))
  arb.io.eligible      := Mux(stageValid, stagedOneHot, wantOffer)
  arb.io.blocked       := VecInit.fill(nCtx)(false.B)
  arb.io.issueAccepted := stageFire

  private val fillNow = !stageValid && arb.io.grantValid
  when(fillNow) {
    val g = arb.io.grantIdx
    val candidate = MakeWireBundle[CtxFetchKey](
      new CtxFetchKey(p),
      _.ctx_slot -> g,
      _.ctx_generation -> io.generation(g),
      _.fetch_epoch -> epochReg(g),
      _.fetch_reqid -> reqidReg(g)
    )
    stageValid := true.B
    stageOwner := g
    stageReq.key := candidate
    stageReq.addr               := alignLine(fetchPc(g).bits)
    stagePc                     := fetchPc(g).bits
    reqidReg(g) := reqidReg(g) + 1.U
    fetchPc(g)  := MakeValid(nextLinePc(fetchPc(g).bits))

    // Compare the new candidate, not the previous registered offer.
    val clash = VecInit.tabulate(maxOutstanding)(i =>
      liveVec(g)(i) && (entries(g)(i).key === candidate)
    ).asUInt.orR
    assert(!clash, "new fetch identity aliases live work")
  }
  when(stageFire) {
    stageValid := false.B // one bubble before the next selection
  }

  // ---------------------------------------------------------------- response demux
  // Always consume. A response matching the request accepted in THIS beat is handled
  // by the enqueue bypass below; a response matching a live awaiting ledger entry
  // fills it and returns its credit; everything else is rejected, and never
  // fabricates or frees credit.
  io.response.ready := true.B

  private val respKey   = io.response.bits.key
  private val slotKnown = respKey.ctx_slot < nCtx.U

  private val bypassHit = stageFire && io.response.valid && slotKnown &&
    (respKey === stageReq.key)

  private val respMatch = Seq.tabulate(nCtx, maxOutstanding) { (c, i) =>
    liveAt(c)(i) && (entries(c)(i).key === respKey) && !entries(c)(i).respValid
  }
  private val matchAny = VecInit.tabulate(nCtx)(c => VecInit(respMatch(c)).exists(x => x))
  private val matchIdx = VecInit.tabulate(nCtx)(c => VecInit(respMatch(c)).indexWhere(x => x))

  when(io.response.valid && slotKnown && !bypassHit) {
    val sc = respKey.ctx_slot
    when(matchAny(sc)) {
      // Exactly the transaction that issued this request completes: fill the ledger
      // entry (delivery happens in request order at the head) and return its credit.
      val e = entries(sc)(matchIdx(sc))
      e.respValid    := true.B
      e.data         := io.response.bits.data
      e.fault        := io.response.bits.fault
      creditDec(sc)  := 1.U
      creditInc(sc)  := true.B
    }.otherwise {
      // Unknown or duplicate packet: rejected, no credit fabricated.
      respStale(sc) := true.B
    }
  }

  // Live ledger entries always carry distinct identities (monotonic reqid per slot,
  // distinct slots), so at most one entry can match any response.
  if (maxOutstanding > 1) {
    for (c <- 0 until nCtx; i <- 0 until maxOutstanding; j <- i + 1 until maxOutstanding) {
      assert(
        !(liveAt(c)(i) && liveAt(c)(j) && (entries(c)(i).key === entries(c)(j).key)),
        "live ledger entries must have distinct fetch identities"
      )
    }
  }

  // Ledger enqueue at request fire (the transaction is on the bus from this beat),
  // carrying the staged identity and unaligned fetch PC. A response arriving in the
  // SAME beat with the exact staged key completes the transaction immediately: the
  // entry is created pre-filled and the credit is enqueued and returned in one
  // combined event — never freed twice, never counted unknown.
  when(stageFire) {
    val sc   = stageOwner
    val t    = headPtr(sc) +& qLen(sc)
    val tail = Mux(t >= maxOutstanding.U, t - maxOutstanding.U, t)(ptrW - 1, 0)
    val e    = entries(sc)(tail)
    e.key       := stageReq.key
    e.pc        := stagePc
    e.respValid := bypassHit
    e.fault     := bypassHit && io.response.bits.fault
    when(bypassHit) {
      e.data := io.response.bits.data
    }
    enqInc(sc) := 1.U
    when(bypassHit) {
      creditDec(sc) := 1.U
      creditInc(sc) := true.B
    }
    serviceInc(sc) := true.B // fetchServices counts accepted requests
  }

  // ---------------------------------------------------------------- per-Context commit
  for (c <- 0 until nCtx) {
    val buf        = buffers(c)
    val pulse      = io.start(c).valid || io.redirect(c).valid || io.cancel(c)
    val invalidate = pulse || genMismatch(c)

    // Directed flush: entry init, branch redirect, task cancel and a generation-only
    // reuse edge clear only THIS Context's unaccepted frontend payload (never the
    // ledger, never peers). InstructionBuffer combinationally gates out.valid with
    // !flush, so the lanes die in the invalidation beat itself.
    buf.io.flush := invalidate

    for (i <- 0 until p.instructionLanes) {
      io.inst(c)(i) <> buf.io.out(i)
    }
    buf.io.out.drop(p.instructionLanes).foreach(_.ready := false.B)

    val head       = entries(c)(headPtr(c))
    val headFilled = qLen(c) =/= 0.U && head.respValid

    // C028 NEGATIVE MUTANT SITE (fetched-identity currency).
    // Delivery currency is decided by FULL identity: armed Task (started, uncancelled,
    // generation current, no committed fetch fault) AND live generation AND live
    // epoch. Replacing this predicate with `armed(c)` or a PC/address comparison
    // revives stale acceptance — W2_03_FetchEpochSpec "same PC identity is not
    // accepted across redirects" must then FAIL on staleRejected / delivered-data /
    // credit assertions. !faulted also forbids YOUNGER already-issued lines from
    // feeding instructions after a fault commits (older buffered instructions stay).
    val headCurrent = armed(c) && !faulted(c) &&
      (head.key.ctx_generation === io.generation(c)) &&
      (head.key.fetch_epoch === epochReg(c))

    val pre = FetchPredecode.Predecode(p)(head.pc, UIntToVec(head.data, p.instructionBits))

    buf.io.feedIn.nValid := 0.U
    buf.io.feedIn.bits  := pre.insts

    when(headFilled) {
      when(!headCurrent) {
        // Old epoch/generation (redirect, reuse, cancel) or disarmed Task: drop the
        // payload. The handshake completed and the credit already returned.
        deqInc(c)  := 1.U
        dropStale(c) := true.B
      }.elsewhen(!invalidate && head.fault) {
        deqInc(c)   := 1.U
        faulted(c)  := true.B
        faultReg(c) := MakeValid(head.pc)
      }.elsewhen(!invalidate && buf.io.feedIn.nReady >= pre.count) {
        deqInc(c) := 1.U
        buf.io.feedIn.nValid := pre.count
        when(pre.hasJumped) {
          // Predecoded jump: continue at the target under a new epoch. Successor
          // lines already issued on the sequential path drain as stale; instructions
          // after the jump inside this line were masked by the shared predecode.
          fetchPc(c)  := MakeValid(pre.nextPc)
          epochReg(c) := epochReg(c) + 1.U
        }
      }
    }

    headPtr(c) := Mux(deqInc(c) === 1.U, wrapInc(headPtr(c)), headPtr(c))
    qLen(c)    := qLen(c) + enqInc(c) - deqInc(c)
    credits(c) := credits(c) + enqInc(c) - creditDec(c)
    nStale(c)   := nStale(c) +& PopCount(Seq(respStale(c), dropStale(c)))
    nCredit(c)  := nCredit(c) +& creditInc(c).asUInt
    nService(c) := nService(c) +& serviceInc(c).asUInt

    // Save the successor of the last accepted instruction, not a speculative
    // line's target. Several predicted jumps may coexist in the buffer.
    val fired = VecInit.tabulate(p.instructionLanes)(i => io.inst(c)(i).fire)
    val anyFired = fired.asUInt.orR
    val lastAccepted = PriorityMux(fired.reverse, io.inst(c).reverse.map(_.bits))
    val predicted = FetchPredecode.PredictJump(p)(lastAccepted.addr, lastAccepted.inst)
    val nextAfterLast = Mux(lastAccepted.brchFwd, predicted.bits, lastAccepted.addr + 4.U)
    pcSaved(c) := MuxCase(pcSaved(c), Seq(
      io.redirect(c).valid -> io.redirect(c).bits,
      io.start(c).valid    -> io.start(c).bits,
      anyFired             -> nextAfterLast
    ))

    io.pc(c) := Mux(buf.io.out(0).valid && !invalidate, buf.io.out(0).bits.addr, pcSaved(c))

    // A counter wrap may not revive an older identity. Check only actual
    // invalidation or prediction commits; an idle frontend never bumps epoch.
    val epochBump = pulse ||
      (headFilled && !invalidate && headCurrent && !head.fault &&
        buf.io.feedIn.nReady >= pre.count && pre.hasJumped)
    val epochNext = epochReg(c) + 1.U
    when(epochBump) {
      assert(epochNext =/= 0.U, "fetch epoch exhausted")
      assert(!VecInit.tabulate(maxOutstanding)(i =>
        liveAt(c)(i) && entries(c)(i).key.fetch_epoch === epochNext
      ).asUInt.orR, "epoch bump aliases live fetch work")
    }

    // Directed pulses (last-connect priority over speculative/jump fetch-PC updates).
    // Every payload invalidation bumps the owning epoch so undelivered ledger lines
    // can never deliver after the event, while their identities/credits keep draining.
    when(io.start(c).valid) {
      started(c)   := true.B
      cancelled(c) := false.B
      genReg(c)    := io.generation(c)
      fetchPc(c)   := MakeValid(io.start(c).bits)
      pcSaved(c)   := io.start(c).bits
      epochReg(c)  := epochReg(c) + 1.U
      faulted(c)   := false.B
      faultReg(c)  := MakeInvalid(UInt(p.fetchAddrBits.W))
    }
    when(io.redirect(c).valid) {
      fetchPc(c)  := MakeValid(io.redirect(c).bits)
      pcSaved(c)  := io.redirect(c).bits
      epochReg(c) := epochReg(c) + 1.U
      faulted(c)  := false.B
      faultReg(c) := MakeInvalid(UInt(p.fetchAddrBits.W))
    }
    when(io.cancel(c)) {
      cancelled(c) := true.B
      epochReg(c)  := epochReg(c) + 1.U
      faulted(c)   := false.B
      faultReg(c)  := MakeInvalid(UInt(p.fetchAddrBits.W))
    }

    io.fault(c)           := faultReg(c)
    io.epoch(c)           := epochReg(c)
    io.outstanding(c)     := credits(c)
    io.staleRejected(c)   := nStale(c)
    io.creditsReturned(c) := nCredit(c)
    io.fetchServices(c)   := nService(c)
  }
}
