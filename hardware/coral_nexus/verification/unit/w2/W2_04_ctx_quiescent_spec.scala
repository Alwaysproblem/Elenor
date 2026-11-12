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

/** C027 / W2-04: `tileQuiescent` must reflect real backend obligations.
  *
  * Regression (W2 plan §8 item 8): a fully parked Context with zero pending launches but a
  * NON-empty ROB partition still has accepted work to retire — the Tile is NOT quiescent
  * and `gate_ctl` must keep the core clock running.
  */
class W2_04_CtxQuiescentSpec extends AnyFreeSpec with ChiselSim {
  def makeParams(ctxCount: Int): Parameters = {
    val p = new Parameters
    p.contextCount       = ctxCount
    p.enableVerification = false
    p.enableFloat        = false
    p.enableRvv          = false
    p.enableVme          = false
    p
  }

  def drive(dut: ContextManager, parked: Boolean, robEmpty: Boolean, launch: Boolean): Unit = {
    for (c <- 0 until dut.io.holdReq.length) {
      dut.io.holdReq(c).poke(false.B)
      dut.io.accept(c).poke(false.B)
      dut.io.initDone(c).poke(false.B)
      dut.io.retIssued(c).poke(false.B)
      dut.io.parkReq(c).poke(parked.B)
      dut.io.ext(c).waitBusy.poke(false.B)
      dut.io.ext(c).regionBusy.poke(false.B)
      dut.io.robEmpty(c).poke(robEmpty.B)
      dut.io.pendingLaunch(c).poke(launch.B)
      dut.io.faultCtx(c).poke(false.B)
      dut.io.taskCompletion(c).poke(false.B)
      dut.io.completionAck(c).poke(false.B)
      dut.io.cancelValid(c).poke(false.B)
      dut.io.cancelGen(c).poke(0.U)
    }
    dut.io.activeCtx.poke(0.U)
  }

  "A fully parked Context with an empty ROB and no launches IS quiescent" in {
    simulate(new ContextManager(makeParams(2))) { dut =>
      dut.reset.poke(true.B)
      drive(dut, parked = true, robEmpty = true, launch = false)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()
      // FREE + drained counts as quiet (no obligations exist).
      dut.io.tileQuiescent.expect(true.B)
    }
  }

  "A parked Context with pendingLaunch still in flight is NOT quiescent" in {
    simulate(new ContextManager(makeParams(2))) { dut =>
      dut.reset.poke(true.B)
      drive(dut, parked = true, robEmpty = true, launch = true)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()
      dut.io.tileQuiescent.expect(false.B)
    }
  }

  "REGRESSION: fully parked, pendingLaunch=0, but ROB non-empty is NOT quiescent" in {
    simulate(new ContextManager(makeParams(2))) { dut =>
      dut.reset.poke(true.B)
      drive(dut, parked = true, robEmpty = true, launch = false)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()
      // Drive context 0 through HELD -> ACCEPTED -> INIT -> READY -> PARKED.
      dut.io.holdReq(0).poke(true.B)
      dut.clock.step()
      dut.io.holdReq(0).poke(false.B)
      dut.io.accept(0).poke(true.B)
      dut.clock.step()
      dut.io.accept(0).poke(false.B)
      dut.io.initDone(0).poke(true.B)
      dut.clock.step()
      dut.io.initDone(0).poke(false.B)
      dut.io.parkReq(0).poke(true.B)
      dut.clock.step()
      // PARKED is observable via eligibility=0 while quiescence is judged below.

      // Accepted-but-unretired work in the ROB must keep the clock eligible: the old
      // predicate (parked||terminal && !pendingLaunch) wrongly reported quiet here.
      dut.io.robEmpty(0).poke(false.B)
      dut.io.pendingLaunch(0).poke(false.B)
      dut.clock.step()
      dut.io.tileQuiescent.expect(false.B)

      // Once the ROB drains, the same parked configuration becomes quiet.
      dut.io.robEmpty(0).poke(true.B)
      dut.clock.step()
      dut.io.tileQuiescent.expect(true.B)
    }
  }

  "A READY (not parked) Context is never quiescent even when drained" in {
    simulate(new ContextManager(makeParams(2))) { dut =>
      dut.reset.poke(true.B)
      drive(dut, parked = false, robEmpty = true, launch = false)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.clock.step()
      // FREE is terminal-quiet, but park only via parkReq; with holdReq=false the slot
      // stays FREE and quiet — so drive one context into READY to break quietness.
      dut.io.holdReq(0).poke(true.B)
      dut.clock.step()
      dut.io.accept(0).poke(true.B)
      dut.clock.step()
      dut.io.initDone(0).poke(true.B)
      dut.clock.step()
      dut.io.robEmpty(0).poke(true.B)
      dut.io.tileQuiescent.expect(false.B)
    }
  }
}
