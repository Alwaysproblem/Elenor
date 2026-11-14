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

/** C028 / W2-03 directed per-Context fetch behavioral spec (W2 plan §7 item 5; W2
  * design §3.5 FET-01/02/03/05, CTX-09, SCH-02; card C028 spec list).
  *
  * Real module only: `PerCtxFetchMux` driven through its frozen tagged
  * request/response channels with true delayed, out-of-order, duplicated, unknown and
  * request-accept-beat responses. Capacity and Context counts come from the sealed
  * C019 targets through the generated consumer `C028TargetParams` — no literal
  * capacity fixtures. Every expectation on identities is derived from OBSERVED fires
  * (the endpoint's transaction record), not from incidental reset literals.
  *
  * Harness contract. Every attempt-006 green failure traced to ONE harness defect:
  * helper-level `clock.step()` beats silently ACCEPTED whatever request the DUT had
  * staged (request.ready idled high), so transactions were consumed without being
  * recorded, answered or drained:
  *   - old-epoch/peer case: ctx1's FIRST line was accepted as an in-flight
  *     pass-through inside `awaitFire(0)`'s polling steps, so waiting for ctx1's
  *     next offer returned the SECOND line and the peer buffer never received its
  *     first response (lane 0 of ctx 1 never became valid);
  *   - credit-conservation / same-PC / start-reuse / c1: the next sequential line
  *     (0x140 / 0x100 / 0x400 / 0x140) was consumed inside expectLanes / drainLanes /
  *     expectLanesDead beats, so the following awaitFire observed the SECOND new
  *     address (0x160 / 0x120 / 0x420 / 0x160);
  *   - redirect-and-response same beat: the 0x300 re-fetch was consumed inside
  *     expectLanesDead beats and both refill lines stayed unanswered, so awaitFire
  *     timed out after 300 cycles;
  *   - cancel / park: the 0x120 successor line was consumed inside expectLanes'
  *     poll beat, so the answered transaction was not the ledger head — the drop
  *     never happened (staleRejected 0 vs 1) or the buffered 0x120 line never
  *     delivered (lane 0 of ctx 0 stayed invalid);
  *   - 1000-cycle case: B's first line was accepted before the loop, but the loop
  *     answered only loop-observed fires: B's head stayed unanswered and B stalled
  *     at its credit bound after 2 fires.
  *
  * `Endpoint` below is therefore the ONLY clock driver of this spec: request.ready
  * is LOW outside explicit accepting windows (an offered request just holds
  * bit-stable — the DUT's own staging contract), and every valid&&ready fire
  * inside a window is recorded in the log and RETAINED (key + address) until an
  * explicit response answers it; pass-through peer fires are never lost.
  *
  * Negative mutant (0303-pc-only-fetch, parent-registered): weakening the
  * ledger-head delivery currency check in `PerCtxFetchMux` from the full
  * armed+generation+epoch+unfaulted identity to a PC-only or unarmed comparison must
  * FAIL "same PC identity is not accepted across redirects" (and the generation-only
  * test) on consumer-visible wrong instructions, staleRejected and credit assertions.
  */
class W2_03_FetchEpochSpec extends AnyFreeSpec with ChiselSim {

  // C019 (reopened, protocol_seq=3) is the ONLY capacity / Context-count authority.
  private val cap       = C028TargetParams.fetchOutstandingPerContext
  private val ctxCounts = C028TargetParams.contextCounts

  private def makeParams(ctxCount: Int): Parameters = {
    val p = new Parameters
    p.contextCount       = ctxCount
    p.enableVerification = false
    p.enableFloat        = false
    p.enableRvv          = false
    p.enableVme          = false
    p
  }

  // ADDI encodings (opcode 0x13): never a predicted jal/bxx.
  private val NOP = BigInt("13", 16)
  private def word(i: Int): BigInt = NOP + i * 0x100

  private case class Key(slot: BigInt, gen: BigInt, epoch: BigInt, reqid: BigInt)

  private def lineData(p: Parameters, words: Seq[BigInt]): BigInt =
    words.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (w, i)) => acc | (w << (32 * i)) }

  private def lineWords(base: BigInt): Seq[BigInt] = (0 until 8).map(i => word((base >> 2).toInt + i))

  /** Deterministic test-side fetch endpoint: the ONLY clock driver of this spec.
    *
    * It owns `request.ready`, the response channel and every clock edge:
    *   - `step`-based helpers take edges with request.ready LOW, so an offered
    *     request holds bit-stable (the DUT contract) — nothing is ever accepted
    *     outside an explicit accepting window;
    *   - `awaitFire` / `stepAccept` / `acceptWithResponse` are the only accepting
    *     windows: every valid&&ready fire of that edge is recorded in the fire log
    *     and RETAINED (key + address) until an explicit `respond`/`armResponse`
    *     answers it — a pass-through peer fire (the staged offer of another
    *     Context, which MUST be accepted before the single request stage can
    *     re-grant) is never lost;
    *   - `respond`/`armResponse` are fail-closed: they answer a RETAINED
    *     transaction only; `injectRejected` drives a deliberately bogus identity
    *     (duplicate / unknown response) and refuses to touch retained work.
    */
  private class Endpoint(dut: PerCtxFetchMux, p: Parameters) {

    // Every accepted request fire, in acceptance order; retained = accepted but
    // not yet answered by a driven response.
    private val fired    = scala.collection.mutable.ArrayBuffer[(Key, BigInt)]()
    private val retained = scala.collection.mutable.ArrayBuffer[(Key, BigInt)]()
    private var responseArmed = false

    def firesOf(slot: Int): Seq[(Key, BigInt)]  = fired.filter(_._1.slot == BigInt(slot)).toSeq
    def fireCount(slot: Int): Int                = firesOf(slot).length
    def totalFires: Int                          = fired.length
    def unansweredOf(slot: Int): Seq[(Key, BigInt)] =
      retained.filter(_._1.slot == BigInt(slot)).toSeq

    def offeredKey: Key =
      Key(
        dut.io.request.bits.key.ctx_slot.peek().litValue,
        dut.io.request.bits.key.ctx_generation.peek().litValue,
        dut.io.request.bits.key.fetch_epoch.peek().litValue,
        dut.io.request.bits.key.fetch_reqid.peek().litValue
      )
    def offeredAddr: BigInt = dut.io.request.bits.addr.peek().litValue

    /** One clock edge with request.ready LOW: an offered request holds; nothing
      * is accepted, so no helper beat can silently consume a transaction.
      */
    def step(): Unit = {
      require(!responseArmed, "response must be withdrawn before a plain step")
      dut.io.request.ready.poke(false.B)
      dut.clock.step()
    }

    /** One clock edge in the accepting window: records and retains every fire. */
    def stepAccept(): Option[(Key, BigInt)] = {
      dut.io.request.ready.poke(true.B)
      val f =
        if (dut.io.request.valid.peek().litToBoolean) Some((offeredKey, offeredAddr)) else None
      dut.clock.step()
      f.foreach { case (k, a) =>
        fired += ((k, a))
        retained += ((k, a))
      }
      f
    }

    /** Wait for an offer of `slot` and fire it. Every other fire on the way (the
      * staged peer offer that must be accepted before the single request stage
      * can re-grant `slot`) is recorded and retained, never dropped.
      */
    def awaitFire(slot: Int, maxWait: Int = 300): (Key, BigInt) = {
      var w = 0
      var hit: Option[(Key, BigInt)] = None
      while (hit.isEmpty && w < maxWait) {
        stepAccept() match {
          case Some(t) if t._1.slot == BigInt(slot) => hit = Some(t)
          case _                                    => ()
        }
        w += 1
      }
      hit.getOrElse(fail(s"no fetch request for slot $slot within $maxWait cycles"))
    }

    private def pokeResponse(k: Key, data: BigInt, fault: Boolean): Unit = {
      dut.io.response.bits.key.ctx_slot.poke(k.slot.U)
      dut.io.response.bits.key.ctx_generation.poke(k.gen.U)
      dut.io.response.bits.key.fetch_epoch.poke(k.epoch.U)
      dut.io.response.bits.key.fetch_reqid.poke(k.reqid.U)
      dut.io.response.bits.data.poke(data.U(p.fetchDataBits.W))
      dut.io.response.bits.fault.poke(fault.B)
      dut.io.response.valid.poke(true.B)
    }

    /** Answer a RETAINED transaction: exactly one response beat, request.ready
      * low so no fire can hide inside the response beat.
      */
    def respond(k: Key, data: BigInt, fault: Boolean = false): Unit = {
      armResponse(k, data, fault)
      dut.io.request.ready.poke(false.B)
      dut.clock.step()
      withdrawResponse()
    }

    /** Drive a deliberately bogus response (duplicate of an already answered
      * transaction, or an unknown identity): rejected by the DUT, and it may
      * never touch retained work.
      */
    def injectRejected(k: Key, data: BigInt): Unit = {
      require(!retained.exists(_._1 == k), s"injected packet matches retained work: $k")
      pokeResponse(k, data, fault = false)
      responseArmed = true
      dut.io.request.ready.poke(false.B)
      dut.clock.step()
      withdrawResponse()
    }

    /** Arm a response for a retained transaction WITHOUT taking an edge (loops
      * answer before their own accepting step); removes it from retained.
      */
    def armResponse(k: Key, data: BigInt, fault: Boolean = false): Unit = {
      val i = retained.indexWhere(_._1 == k)
      require(i >= 0, s"response target was never accepted (or already answered): $k")
      retained.remove(i)
      pokeResponse(k, data, fault)
      responseArmed = true
    }

    def withdrawResponse(): Unit = {
      responseArmed = false
      dut.io.response.valid.poke(false.B)
    }

    /** Same-beat settlement: the offered request fires and its response is
      * consumed in the SAME edge (the DUT's enqueue bypass). Recorded once and
      * answered at its own accept beat, so it is never retained.
      */
    def acceptWithResponse(k: Key, data: BigInt, fault: Boolean = false): Unit = {
      pokeResponse(k, data, fault)
      responseArmed = true
      dut.io.request.ready.poke(true.B)
      require(dut.io.request.valid.peek().litToBoolean, "no offered request to accept")
      val t = (offeredKey, offeredAddr)
      require(t._1 == k, s"offered request $t does not match the armed response $k")
      dut.clock.step()
      withdrawResponse()
      dut.io.request.ready.poke(false.B)
      fired += t
    }

    // -------------------------------------------------- consumer-side observers

    /** Poll until the first instruction lane is valid, then assert lanes. The
      * poll beats never accept a request (ready stays low).
      */
    def expectLanes(c: Int, base: BigInt, words: Seq[BigInt]): Unit = {
      var w = 0
      while (!dut.io.inst(c)(0).valid.peek().litToBoolean && w < 16) {
        step()
        w += 1
      }
      for (i <- words.indices) {
        dut.io.inst(c)(i).valid.expect(true.B, s"lane $i of ctx $c")
        dut.io.inst(c)(i).bits.addr.expect((base + 4 * i).U)
        dut.io.inst(c)(i).bits.inst.expect(words(i).U)
      }
    }

    def expectLanesDead(c: Int, cycles: Int = 4): Unit = {
      for (_ <- 0 until cycles) {
        for (i <- 0 until p.instructionLanes) {
          dut.io.inst(c)(i).valid.expect(false.B)
        }
        step()
      }
    }

    def drainLanes(c: Int): Unit = {
      for (i <- 0 until p.instructionLanes) {
        dut.io.inst(c)(i).ready.poke(true.B)
      }
      step()
      for (i <- 0 until p.instructionLanes) {
        dut.io.inst(c)(i).ready.poke(false.B)
      }
    }

    def expectNoReq(cycles: Int = 4): Unit = {
      for (_ <- 0 until cycles) {
        dut.io.request.valid.expect(false.B)
        step()
      }
    }

    // ------------------------------------------------------------- input pulses

    /** Entry-initialization pulse; `accepted` and `programReady` levels are set with it. */
    def pulseStart(c: Int, pc: BigInt, accepted: Boolean = true, ready: Boolean = true): Unit = {
      dut.io.accepted(c).poke(accepted.B)
      dut.io.programReady(c).poke(ready.B)
      dut.io.start(c).valid.poke(true.B)
      dut.io.start(c).bits.poke(pc.U)
      step()
      dut.io.start(c).valid.poke(false.B)
    }

    def pulseRedirect(c: Int, pc: BigInt): Unit = {
      dut.io.redirect(c).valid.poke(true.B)
      dut.io.redirect(c).bits.poke(pc.U)
      step()
      dut.io.redirect(c).valid.poke(false.B)
    }

    def pulseCancel(c: Int): Unit = {
      dut.io.cancel(c).poke(true.B)
      step()
      dut.io.cancel(c).poke(false.B)
    }

    def reset(): Unit = {
      for (c <- 0 until p.contextCount) {
        dut.io.accepted(c).poke(false.B)
        dut.io.programReady(c).poke(false.B)
        dut.io.parked(c).poke(false.B)
        dut.io.cancel(c).poke(false.B)
        dut.io.generation(c).poke(0.U)
        dut.io.start(c).valid.poke(false.B)
        dut.io.start(c).bits.poke(0.U)
        dut.io.redirect(c).valid.poke(false.B)
        dut.io.redirect(c).bits.poke(0.U)
        for (i <- 0 until p.instructionLanes) {
          dut.io.inst(c)(i).ready.poke(false.B)
        }
      }
      dut.io.request.ready.poke(false.B)
      dut.io.response.valid.poke(false.B)
      dut.io.response.bits.key.ctx_slot.poke(0.U)
      dut.io.response.bits.key.ctx_generation.poke(0.U)
      dut.io.response.bits.key.fetch_epoch.poke(0.U)
      dut.io.response.bits.key.fetch_reqid.poke(0.U)
      dut.io.response.bits.data.poke(0.U)
      dut.io.response.bits.fault.poke(false.B)
      dut.reset.poke(true.B)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()
    }
  }

  // ---------------------------------------------------------------------------
  // Acceptance: ready/install gate
  // ---------------------------------------------------------------------------

  "accepted and programReady together gate the first fetch" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()

      // Gate inputs high without an accepted Task entry: nothing fetches (the gate is
      // never inferred from reset or Context selection).
      dut.io.accepted(0).poke(true.B)
      dut.io.programReady(0).poke(true.B)
      ep.expectNoReq(4)

      // Entry pulse with neither gate asserted.
      ep.pulseStart(0, 0x100, accepted = false, ready = false)
      ep.expectNoReq(4)

      // programReady alone is not sufficient.
      dut.io.programReady(0).poke(true.B)
      ep.expectNoReq(4)

      // accepted completes the gate: the first request fires under the observed
      // start epoch and the definitionally-first reqid.
      dut.io.accepted(0).poke(true.B)
      val epochAtStart = dut.io.epoch(0).peek().litValue
      val (k, addr) = ep.awaitFire(0)
      assert(addr == 0x100, s"first fetch PC $addr")
      assert(k.slot == 0 && k.gen == dut.io.generation(0).peek().litValue && k.reqid == 0, s"$k")
      assert(k.epoch == epochAtStart, s"$k")
    }
  }

  // ---------------------------------------------------------------------------
  // Acceptance: old epoch / old generation rejection + peer buffer stability
  // ---------------------------------------------------------------------------

  "old epoch response is rejected with credit while the peer buffer is unchanged" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)
      ep.pulseStart(1, 0x200)

      val (kA1, _) = ep.awaitFire(0)
      val (kA2, _) = ep.awaitFire(0) // speculative successor line
      // B's FIRST transaction was accepted as a pass-through inside the second
      // awaitFire window (the staged peer offer must be accepted before ctx0 is
      // re-granted); the endpoint retained it, so the peer's first line — not its
      // second — gets its first response.
      val (kB1, kB1Addr) = ep.unansweredOf(1).head
      assert(kB1Addr == 0x200, s"peer first line $kB1Addr")

      val bWords = lineWords(0x200)
      ep.respond(kB1, lineData(p, bWords))
      ep.expectLanes(1, 0x200, bWords.take(4))

      // Redirect A away; its outstanding lines become old-epoch transactions.
      ep.pulseRedirect(0, 0x300)

      // Late response for A's entry line under the OLD epoch: rejected, but the
      // handshake completes and its credit returns (CTX-09).
      val staleWords = lineWords(0xF00)
      ep.respond(kA1, lineData(p, staleWords))
      ep.step()
      ep.step()
      ep.expectLanesDead(0, 3)
      dut.io.staleRejected(0).expect(1.U)
      dut.io.creditsReturned(0).expect(1.U)

      // Peer Context: no rejection, delivered payload bit-stable.
      dut.io.staleRejected(1).expect(0.U)
      assert(dut.io.fetchServices(1).peek().litValue >= 1, "peer B must have been served")
      ep.expectLanes(1, 0x200, bWords.take(4))

      // A fetches from the redirect target under a new epoch.
      val (kA3, aA3) = ep.awaitFire(0)
      assert(aA3 == 0x300 && kA3.epoch == kA1.epoch + 1, s"$kA3")

      // Drain the stale speculative successor blocking the head, then deliver the
      // target line.
      ep.respond(kA2, lineData(p, staleWords))
      ep.step()
      dut.io.staleRejected(0).expect(2.U)
      val tWords = lineWords(0x300)
      ep.respond(kA3, lineData(p, tWords))
      ep.expectLanes(0, 0x300, tWords.take(4))
      // every observed A fire returned exactly one credit
      dut.io.creditsReturned(0).expect(3.U)
    }
  }

  "generation change alone flushes buffered payload and rejects old results" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)
      val (k1, _) = ep.awaitFire(0)
      val (k2, _) = ep.awaitFire(0) // younger line stays in flight
      val w1 = lineWords(0x100)
      ep.respond(k1, lineData(p, w1))
      ep.expectLanes(0, 0x100, w1.take(4)) // instructions are buffered and visible

      val epochBefore = dut.io.epoch(0).peek().litValue
      val genBefore   = dut.io.generation(0).peek().litValue

      // Reuse edge WITHOUT any pulse: the buffered payload dies in the very
      // invalidation beat (combinationally), before any clock edge.
      dut.io.generation(0).poke((genBefore + 1).U)
      for (i <- 0 until p.instructionLanes) {
        dut.io.inst(0)(i).valid.expect(false.B, s"lane $i must die in the invalidation beat")
      }
      ep.step()
      ep.expectLanesDead(0, 3)

      // Disarmed until an explicit start: no new requests from the old fetch PC.
      ep.expectNoReq(4)

      // The younger in-flight old-generation line: rejected by identity alone (the
      // epoch register never moved), handshake completes, credit returns.
      val garbage = lineWords(0xE00)
      ep.respond(k2, lineData(p, garbage))
      ep.step()
      dut.io.staleRejected(0).expect(1.U)
      dut.io.creditsReturned(0).expect(2.U) // k1's genuine response plus k2's drain
      ep.expectLanesDead(0, 2)
      dut.io.epoch(0).expect(epochBefore.U) // rejection came from generation alone
    }
  }

  // ---------------------------------------------------------------------------
  // Acceptance: duplicate / unknown responses and credit conservation
  // ---------------------------------------------------------------------------

  "duplicate and unknown responses never fabricate or free credit" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)
      val (k1, _) = ep.awaitFire(0) // entry line
      val (k2, _) = ep.awaitFire(0) // speculative successor, left in flight

      val w1 = lineWords(0x100)
      ep.respond(k1, lineData(p, w1))
      ep.expectLanes(0, 0x100, w1.take(4))
      dut.io.creditsReturned(0).expect(1.U)

      // Freeze speculation (park) so the exact credit arithmetic is observable;
      // delivery still proceeds under park.
      dut.io.parked(0).poke(true.B)
      ep.step()
      val liveBefore = dut.io.outstanding(0).peek().litValue

      // Duplicate of an already-consumed transaction and an unknown reqid: both are
      // rejected, and neither may fabricate or free credit.
      val garbage = lineWords(0xD00)
      ep.injectRejected(k1, lineData(p, garbage))
      dut.io.staleRejected(0).expect(1.U)
      dut.io.creditsReturned(0).expect(1.U)
      ep.injectRejected(k2.copy(reqid = k2.reqid + 77), lineData(p, garbage))
      dut.io.staleRejected(0).expect(2.U)
      dut.io.creditsReturned(0).expect(1.U)
      val liveAfterBogus = dut.io.outstanding(0).peek().litValue
      assert(liveAfterBogus == liveBefore, "bogus packet freed credit")

      // The genuine k2 response frees exactly its own credit and delivers in order.
      val w2 = lineWords(0x120)
      ep.respond(k2, lineData(p, w2))
      dut.io.creditsReturned(0).expect(2.U)
      assert(
        dut.io.outstanding(0).peek().litValue == liveAfterBogus - 1,
        "genuine response must free exactly one credit"
      )
      ep.drainLanes(0)
      ep.drainLanes(0)
      ep.expectLanes(0, 0x120, w2.take(4)) // in-order delivery of the younger line
      // Neither bogus packet corrupted the buffered payload.
    }
  }

  "response in the request accept beat delivers and credits exactly once" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)

      // The endpoint holds the offer without accepting it (no fire outside an
      // explicit window).
      var w = 0
      while (!dut.io.request.valid.peek().litToBoolean && w < 16) {
        ep.step()
        w += 1
      }
      val k    = ep.offeredKey
      val addr = ep.offeredAddr

      // The endpoint answers in the very beat the request is accepted: the enqueue
      // bypass settles that transaction immediately — one single response delivers
      // and credits exactly once, nothing is counted unknown, nothing is retransmitted.
      val sameWords = lineWords(addr)
      ep.acceptWithResponse(k, lineData(p, sameWords))
      // Park immediately: no successor can stage, so the credit arithmetic is exact.
      dut.io.parked(0).poke(true.B)
      ep.step() // absorb the bubble beat; nothing else may fire

      dut.io.creditsReturned(0).expect(1.U)
      dut.io.staleRejected(0).expect(0.U)
      ep.expectLanes(0, addr, sameWords.take(4))

      // Exactly one credit was consumed for this transaction: outstanding equals the
      // observed fires minus the one settled response.
      assert(
        dut.io.outstanding(0).peek().litValue == ep.fireCount(0) - 1,
        "same-beat settlement must leave exactly the unanswered fires outstanding"
      )
    }
  }

  "credit conservation with reverse order responses and capacity backpressure" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)

      val (k1, a1) = ep.awaitFire(0)
      assert(a1 == 0x100)
      dut.io.outstanding(0).expect(1.U)
      val (k2, a2) = ep.awaitFire(0)
      assert(a2 == 0x120 && k2.reqid == k1.reqid + 1)
      dut.io.outstanding(0).expect(2.U)

      // Capacity bound: no third request while both transactions are in flight.
      ep.expectNoReq(8)

      // Out-of-order responses: the younger line answers first. In-order delivery
      // holds: nothing is buffered until the head line has answered.
      val w2 = lineWords(0x120)
      ep.respond(k2, lineData(p, w2))
      dut.io.outstanding(0).expect(1.U)
      dut.io.creditsReturned(0).expect(1.U)
      ep.expectLanesDead(0, 3)

      val w1 = lineWords(0x100)
      ep.respond(k1, lineData(p, w1))
      ep.expectLanes(0, 0x100, w1.take(4))
      ep.drainLanes(0)
      ep.drainLanes(0)
      ep.expectLanes(0, 0x120, w2.take(4)) // both lines delivered, in order
      dut.io.creditsReturned(0).expect(2.U)

      // Space freed: sequential fetching resumes.
      val (k3, a3) = ep.awaitFire(0)
      assert(a3 == 0x140 && k3.reqid == k2.reqid + 1, s"$k3")
    }
  }

  // ---------------------------------------------------------------------------
  // Acceptance: same-PC identity negative (the pc-only mutant catcher)
  // ---------------------------------------------------------------------------

  "same PC identity is not accepted across redirects" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)

      // Old request at PC 0x100 accepted and LIVE in the ledger across two redirects.
      val (kOld, aOld) = ep.awaitFire(0)
      assert(aOld == 0x100)
      val (kOld2, _) = ep.awaitFire(0) // speculative successor, also live

      ep.pulseRedirect(0, 0x300) // away ...
      ep.pulseRedirect(0, 0x100) // ... and back to the SAME PC

      // The endpoint now returns the EXACT old key for the same line.
      val oldWords = lineWords(0xA00)
      ep.respond(kOld, lineData(p, oldWords))
      ep.step()
      ep.step()

      // Identity, not PC, decides: the old-epoch payload must never reach a consumer.
      ep.expectLanesDead(0, 3)
      dut.io.staleRejected(0).expect(1.U)

      // Drain the stale successor, then the frontend re-requests the SAME line under
      // the new epoch and only the new response may deliver.
      ep.respond(kOld2, lineData(p, oldWords))
      ep.step()
      dut.io.staleRejected(0).expect(2.U)
      val (kNew, aNew) = ep.awaitFire(0)
      assert(aNew == 0x100 && kNew.epoch == kOld.epoch + 2 && kNew.reqid == kOld2.reqid + 1, s"$kNew")

      val newWords = lineWords(0x400)
      ep.respond(kNew, lineData(p, newWords))
      ep.expectLanes(0, 0x100, newWords.take(4))

      // Credit ledger: every observed fire returned exactly one credit. The
      // successor line is only STAGED (offered, held bit-stable outside the
      // accepting window), so every recorded fire is answered and no credit is
      // left outstanding.
      ep.step()
      dut.io.creditsReturned(0).expect(3.U)
      assert(dut.io.outstanding(0).peek().litValue == ep.fireCount(0) - 3)
    }
  }

  // ---------------------------------------------------------------------------
  // Acceptance: request stability, redirect/cancel overlaps
  // ---------------------------------------------------------------------------

  "offered request holds bit stability across redirect until it completes" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)

      var w = 0
      while (!dut.io.request.valid.peek().litToBoolean && w < 16) {
        ep.step()
        w += 1
      }
      val k    = ep.offeredKey
      val addr = ep.offeredAddr
      assert(addr == 0x100)

      // Redirect mid-stall: the obsolete offer must remain bit stable — it is
      // draining work, never withdrawn.
      ep.pulseRedirect(0, 0x300)

      // Churn around the stalled offer: peer start and park jitter, 120 beats. No
      // beat accepts anything (ready stays low), so the offer must hold exactly.
      ep.pulseStart(1, 0x200)
      for (i <- 0 until 120) {
        dut.io.request.valid.expect(true.B)
        assert(ep.offeredKey == k, "offered request identity changed under backpressure")
        assert(ep.offeredAddr == addr, "offered addr changed")
        dut.io.parked(1).poke((i % 2 == 0).B)
        ep.step()
      }

      // Release: the obsolete offer completes in an explicit accepting window, its
      // response is rejected as stale with exact credit, and the owner continues at
      // the redirect target.
      val (kF, aF) = ep.awaitFire(0)
      assert(kF == k && aF == addr, "the released fire must be the held offer")
      ep.respond(k, lineData(p, lineWords(0xB00)))
      ep.step()
      dut.io.staleRejected(0).expect(1.U)
      dut.io.creditsReturned(0).expect(1.U)

      val (k2, a2) = ep.awaitFire(0)
      assert(a2 == 0x300 && k2.epoch == k.epoch + 1, s"$k2")
    }
  }

  "redirect and response in the same beat drop the wrong path payload" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)
      val (k1, _) = ep.awaitFire(0)

      // Response and redirect in the same beat: the handshake completes and the
      // credit returns, but delivery is suppressed and the payload never buffers.
      dut.io.redirect(0).valid.poke(true.B)
      dut.io.redirect(0).bits.poke(0x300.U)
      ep.respond(k1, lineData(p, lineWords(0x900)))
      dut.io.redirect(0).valid.poke(false.B)

      dut.io.creditsReturned(0).expect(1.U)
      ep.step()
      dut.io.staleRejected(0).expect(1.U)
      ep.expectLanesDead(0, 3)

      val (k2, a2) = ep.awaitFire(0)
      assert(a2 == 0x300 && k2.epoch == k1.epoch + 1, s"$k2")
    }
  }

  "cancel stops requests clears payload and drains credits" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)
      val (k1, _) = ep.awaitFire(0)
      val w1 = lineWords(0x100)
      ep.respond(k1, lineData(p, w1))
      ep.expectLanes(0, 0x100, w1.take(4))
      val (k2, _) = ep.awaitFire(0) // in flight when the cancel lands

      ep.pulseCancel(0)
      ep.expectLanesDead(0, 3)
      ep.expectNoReq(6)

      // Accepted response drains: handshake completes, credit returns, no payload.
      ep.respond(k2, lineData(p, lineWords(0x800)))
      ep.step()
      dut.io.creditsReturned(0).expect(2.U)
      dut.io.staleRejected(0).expect(1.U)
      dut.io.outstanding(0).expect(0.U)
      ep.expectLanesDead(0, 2)

      // Reuse: generation edge plus a fresh entry start re-arms fetching.
      val genBefore = dut.io.generation(0).peek().litValue
      dut.io.generation(0).poke((genBefore + 1).U)
      ep.pulseStart(0, 0x400)
      val (k3, a3) = ep.awaitFire(0)
      assert(a3 == 0x400 && k3.gen == genBefore + 1 && k3.epoch == k2.epoch + 2, s"$k3")
    }
  }

  "start reuse keeps old transaction identities draining with exact credits" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)
      val (k1, _) = ep.awaitFire(0)
      val (k2, _) = ep.awaitFire(0)
      dut.io.outstanding(0).expect(2.U)

      // Reuse edge in one beat: generation bump plus the new Task's entry pulse.
      val genBefore = dut.io.generation(0).peek().litValue
      dut.io.generation(0).poke((genBefore + 1).U)
      dut.io.start(0).valid.poke(true.B)
      dut.io.start(0).bits.poke(0x400.U)
      ep.step()
      dut.io.start(0).valid.poke(false.B)

      // start did NOT discard the old transaction identities or their credits.
      dut.io.outstanding(0).expect(2.U)
      ep.expectNoReq(4) // ledger full: the new entry waits for the drain

      val garbage = lineWords(0x700)
      ep.respond(k1, lineData(p, garbage))
      ep.step()
      dut.io.staleRejected(0).expect(1.U)
      dut.io.creditsReturned(0).expect(1.U)
      ep.respond(k2, lineData(p, garbage))
      ep.step()
      dut.io.staleRejected(0).expect(2.U)
      dut.io.creditsReturned(0).expect(2.U)

      val (k3, a3) = ep.awaitFire(0)
      assert(a3 == 0x400 && k3.gen == genBefore + 1 && k3.reqid == k2.reqid + 1, s"$k3")
      ep.expectLanesDead(0, 2) // old payload never delivered
    }
  }

  // ---------------------------------------------------------------------------
  // Acceptance: park, independent progress and bounded service
  // ---------------------------------------------------------------------------

  "park inhibits new fetch work while buffered payload survives" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)
      val (k1, _) = ep.awaitFire(0)
      val w1 = lineWords(0x100)
      ep.respond(k1, lineData(p, w1))
      ep.expectLanes(0, 0x100, w1.take(4))
      val (k2, _) = ep.awaitFire(0)

      dut.io.parked(0).poke(true.B)
      ep.step()

      // An accepted response returns while parked: it buffers (accepted returns
      // survive) but no new request is offered.
      val w2 = lineWords(0x120)
      ep.respond(k2, lineData(p, w2))
      ep.expectNoReq(6)

      // Buffered payload stays bit stable under park, with peer churn alongside.
      ep.pulseStart(1, 0x200)
      for (_ <- 0 until 100) {
        dut.io.inst(0)(0).valid.expect(true.B)
        dut.io.inst(0)(0).bits.addr.expect(0x100.U)
        dut.io.inst(0)(0).bits.inst.expect(w1.head.U)
        ep.step()
      }

      // The younger line delivered while parked (in order, after draining the head).
      ep.drainLanes(0)
      ep.drainLanes(0)
      ep.expectLanes(0, 0x120, w2.take(4))

      // Unpark resumes new work at the next sequential line.
      dut.io.parked(0).poke(false.B)
      val (k3, a3) = ep.awaitFire(0)
      assert(a3 == 0x140, s"$a3")
    }
  }

  "peer B progresses while a delayed A response is withheld a thousand cycles" in {
    val p = makeParams(ctxCounts.last)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)
      ep.pulseStart(1, 0x200)

      // A consumes its whole credit budget, then its responses are withheld. Every
      // fire on the way (including B's pass-through) is recorded — nothing is lost.
      ep.awaitFire(0)
      ep.awaitFire(0)
      assert(ep.fireCount(0) == 2)
      assert(dut.io.outstanding(0).peek().litValue == cap)

      // B consumes its lanes continuously so its pipeline never fills.
      for (i <- 0 until p.instructionLanes) {
        dut.io.inst(1)(i).ready.poke(true.B)
      }

      // Exactly ONE clock edge per iteration, taken in the accepting window; at
      // most one B response is armed before the edge and withdrawn after it. The
      // loop always answers the OLDEST retained B transaction — including the one
      // accepted before the loop — so B's head is never left unanswered while
      // A's responses are withheld.
      var lastBTotal = -1
      var iter       = 0
      while (iter < 1000) {
        ep.unansweredOf(1).headOption.foreach { case (k, a) =>
          ep.armResponse(k, lineData(p, lineWords(a)))
        }
        ep.stepAccept() match {
          case Some((k, _)) if k.slot == 1 =>
            if (lastBTotal >= 0) {
              assert(
                ep.totalFires - lastBTotal <= ctxCounts.last,
                s"B waited ${ep.totalFires - lastBTotal} accepted requests (bound N=${ctxCounts.last})"
              )
            }
            lastBTotal = ep.totalFires
          case _ => ()
        }
        ep.withdrawResponse()
        assert(dut.io.outstanding(0).peek().litValue <= cap, "A exceeded its own bound")
        iter += 1
      }

      // B was serviced continuously under the accepted-opportunity bound while A's
      // delayed responses blocked only A.
      val bFires = ep.fireCount(1)
      assert(bFires >= 100, s"B served only $bFires times")
      // Accepted-request counters equal the OBSERVED fires, per Context.
      dut.io.fetchServices(1).expect(bFires.U)
      dut.io.fetchServices(0).expect(ep.fireCount(0).U)
      dut.io.staleRejected(0).expect(0.U)
      dut.io.outstanding(0).expect(cap.U)

      // The withheld A response is still current (A never redirected): it delivers.
      val (kA1, aA1) = ep.firesOf(0).head
      val aWords = lineWords(0x600)
      ep.respond(kA1, lineData(p, aWords))
      ep.expectLanes(0, aA1, aWords.take(4))
    }
  }

  // ---------------------------------------------------------------------------
  // Acceptance: c1 specialization (C019 cs1_c1), predecode jump, fault
  // ---------------------------------------------------------------------------

  "c1 target specializes the single context frontend with jump and fault" in {
    val p = makeParams(ctxCounts.head)
    simulate(new PerCtxFetchMux(p, cap)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      // Unaligned entry PC exercises the predecode startElem path.
      ep.pulseStart(0, 0x104)
      val (k1, a1) = ep.awaitFire(0)
      assert(a1 == 0x100, "request must carry the aligned line address")
      val (k1b, a1b) = ep.awaitFire(0)
      assert(a1b == 0x120)

      // jal +0x20 at 0x10C (slot 3) targets 0x12C; slot 0 is masked by the entry
      // offset. Lanes: 0x104 NOP, 0x108 NOP, 0x10C jal — suffix suppressed.
      val jal = BigInt("0200006F", 16)
      val w1  = Seq(NOP, NOP, NOP, jal, word(4), word(5), word(6), word(7))
      ep.respond(k1, lineData(p, w1))
      ep.expectLanes(0, 0x104, Seq(NOP, NOP, jal))
      dut.io.inst(0)(3).valid.expect(false.B)

      // Drain the jump-line lanes: the buffer empties and pc shows the SAVED target
      // (the target request itself carries the aligned line address 0x120). The
      // target offer stages during this beat but is only ever fired explicitly.
      for (i <- 0 until p.instructionLanes) {
        dut.io.inst(0)(i).ready.poke(true.B)
      }
      ep.step()
      for (i <- 0 until p.instructionLanes) {
        dut.io.inst(0)(i).ready.poke(false.B)
      }
      var w = 0
      while (dut.io.pc(0).peek().litValue != 0x12C && w < 8) {
        ep.step()
        w += 1
      }
      dut.io.pc(0).expect(0x12C.U)

      // The speculative sequential line is stale under the jump epoch: drain it.
      ep.respond(k1b, lineData(p, lineWords(0x500)))
      ep.step()
      dut.io.staleRejected(0).expect(1.U)

      // Fetch continues at the predecoded target under a new epoch.
      val (k2, a2) = ep.awaitFire(0)
      assert(a2 == 0x120, "request addr is the aligned target line")
      assert(k2.epoch == k1.epoch + 1, s"$k2")

      // Deliver the target line. Entry pc 0x12C sits at slot 3 of line 0x120, so the
      // visible lanes are 0x12C..0x138 (slots 3..6).
      val tWords = lineWords(0x120)
      ep.respond(k2, lineData(p, tWords))
      ep.expectLanes(0, 0x12C, tWords.drop(3).take(4))

      // Younger lines fire sequentially while queue space allows.
      val (k3, a3) = ep.awaitFire(0)
      assert(a3 == 0x140)
      val (k4, a4) = ep.awaitFire(0) // younger line, still outstanding
      assert(a4 == 0x160)

      // The 0x140 line FAULTS: the fault commits, but the older buffered target-line
      // instructions stay visible and no new request is offered.
      ep.respond(k3, lineData(p, lineWords(0x140)), fault = true)
      ep.step()
      dut.io.fault(0).valid.expect(true.B)
      dut.io.fault(0).bits.expect(0x140.U)
      ep.expectNoReq(4)

      // Younger outstanding line returned AFTER the fault: credit settles, but NO
      // younger instruction ever becomes visible (older lanes intact).
      ep.respond(k4, lineData(p, lineWords(0x160)))
      ep.step()
      dut.io.creditsReturned(0).expect(5.U) // k1, k1b, k2, k3(fault), k4 all settled
      dut.io.fault(0).valid.expect(true.B)
      ep.expectLanes(0, 0x12C, tWords.drop(3).take(4)) // unchanged, no younger payload

      // Recovery: an explicit redirect clears the fault and re-arms fetching.
      ep.pulseRedirect(0, 0x200)
      dut.io.fault(0).valid.expect(false.B)
      val (k5, a5) = ep.awaitFire(0)
      assert(a5 == 0x200 && k5.epoch == k4.epoch + 1, s"$k5")
    }
  }

  "Ledger wrap preserves pending transactions at an irregular positive bound" in {
    val p = makeParams(ctxCounts.head)
    p.fetchDataBits = 128
    // Exercise the constructor's positive-bound contract, not a new CS1 profile.
    val bound = cap + 1
    simulate(new PerCtxFetchMux(p, bound)) { dut =>
      val ep = new Endpoint(dut, p)
      ep.reset()
      ep.pulseStart(0, 0x100)
      dut.io.inst(0).foreach(_.ready.poke(true.B))
      val pending  = scala.collection.mutable.ArrayBuffer[(Key, BigInt, Int)]()
      val observed = scala.collection.mutable.ArrayBuffer[BigInt]()
      var cycles    = 0
      var requests  = 0
      var responses = 0
      while (observed.size < 48 && cycles < 400) {
        val reply = pending.headOption.filter(_._3 <= cycles)
        reply match {
          case Some((key, addr, _)) =>
            val words = (0 until p.fetchInstrSlots).map(i => word((addr >> 2).toInt + i))
            ep.armResponse(key, lineData(p, words))
          case None => ep.withdrawResponse()
        }
        for (lane <- 0 until p.instructionLanes if dut.io.inst(0)(lane).valid.peek().litToBoolean) {
          val addr = dut.io.inst(0)(lane).bits.addr.peek().litValue
          dut.io.inst(0)(lane).bits.inst.expect(word((addr >> 2).toInt).U)
          observed += addr
        }
        ep.stepAccept() match {
          case Some((k, a)) =>
            pending += ((k, a, cycles + 7))
            requests += 1
          case None => ()
        }
        if (reply.isDefined) {
          pending.remove(0)
          responses += 1
        }
        cycles += 1
      }
      ep.withdrawResponse()
      assert(observed.toSeq == (0 until 48).map(i => BigInt(0x100 + i * 4)),
        s"lost, duplicated or reordered instructions at ledger wrap: $observed")
      dut.io.creditsReturned(0).expect(responses.U)
      dut.io.fetchServices(0).expect(requests.U)
      dut.io.outstanding(0).expect((requests - responses).U)
      dut.io.staleRejected(0).expect(0.U)
    }
  }
}
