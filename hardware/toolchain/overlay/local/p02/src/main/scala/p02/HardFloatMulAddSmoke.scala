// SPDX-License-Identifier: BSD-3-Clause
//
// P02 FP32 fused multiply-add smoke top.
//
// Wraps the pinned Berkeley HardFloat MulAddRecFN(expWidth=8, sigWidth=24)
// exactly as Gemmini instantiates it (gemmini Arithmetic.scala MulAddRecFN with
// io.op = 0), converting IEEE-754 FP32 operands/results at the boundary with
// recFNFromFN / fNFromRecFN.
//
// IO contract (P02Runner + smoke/hardfloat.cpp):
//   clock, reset   implicit Module ports
//   a, b, c        input  [31:0]  IEEE-754 FP32 bit patterns
//   out            output [31:0]  IEEE-754 FP32 a*b+c, single rounding
//   flags          output [4:0]   HardFloat exception flags:
//                                 bit4 invalid, bit3 infinite, bit2 overflow,
//                                 bit1 underflow, bit0 inexact
//
// roundingMode is hardwired to round_near_even and detectTininess to
// tininess_afterRounding (HardFloat consts; RISC-V semantics, same choice
// Gemmini's scale args make). They are deliberately NOT top-level ports.
package p02

import chisel3._
import hardfloat.consts
import hardfloat.{MulAddRecFN, fNFromRecFN, recFNFromFN}

class HardFloatMulAddSmoke extends Module {
  val a = IO(Input(UInt(32.W)))
  val b = IO(Input(UInt(32.W)))
  val c = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))
  val flags = IO(Output(UInt(5.W)))

  val madd = Module(new MulAddRecFN(8, 24))
  madd.io.op := 0.U(2.W) // op(1)=0, op(0)=0: a*b + c
  madd.io.a := recFNFromFN(8, 24, a)
  madd.io.b := recFNFromFN(8, 24, b)
  madd.io.c := recFNFromFN(8, 24, c)
  madd.io.roundingMode := consts.round_near_even // hardwired RNE
  madd.io.detectTininess := consts.tininess_afterRounding

  out := fNFromRecFN(8, 24, madd.io.out)
  flags := madd.io.exceptionFlags

  // Built-in self check, exercised by smoke/hardfloat.cpp vector exact_basic:
  // 1.5 * 2.0 + 0.5 = 3.5 exactly representable in FP32, no exceptions.
  when(a === "h3fc00000".U && b === "h40000000".U && c === "h3f000000".U) {
    assert(out === "h40600000".U, "p02 smoke: 1.5*2.0+0.5 must equal 3.5")
    assert(flags === 0.U(5.W), "p02 smoke: exact 3.5 must not raise FP flags")
  }
}
