// SPDX-License-Identifier: BSD-3-Clause
//
// P03 HardFloat primitive probe top.
//
// Wraps the pinned Berkeley HardFloat f32 primitives exactly as the numeric
// freeze requires, with the full IEEE<->recFN conversion at the boundary:
//   op 0 fma  : MulAddRecFN(8, 24), io.op = 0 (a*b + c, single rounding)
//   op 1 add  : AddRecFN(8, 24), subOp = 0
//   op 2 mul  : MulRecFN(8, 24) recoded operands -> rounded out/flags
//   op 3 bf16 : RecFNToRecFN(8, 24, 8, 8) + fNFromRecFN(8, 8)
//               (expWidth 8, sigWidth 8: the 16-bit "brain float" image)
//   op 4 i32  : RecFNToIN(8, 24, 32), signedOut = 1
//
// RecFNToIN is a chisel3.Module, so this top is a Module with real clock and
// reset ports; the C++ harness must toggle and settle them (it does NOT
// assume RawModule). Every submodule is purely combinational, so one settled
// clock edge per vector is sufficient.
//
// Rounding: roundingMode hardwired to round_near_even and detectTininess to
// tininess_afterRounding (HardFloat consts; RISC-V semantics). Deliberately
// NOT top-level ports, mirroring the P02 smoke top and Gemmini's scale args.
//
// Output protocol (PRIVATE probe protocol, NOT the BOA wire ABI):
//   out    [31:0]  IEEE bit pattern; op 3 (bf16) carries the 16-bit pattern in
//                  the low half with zero extension; op 4 carries the 32-bit
//                  two's-complement integer result.
//   flags  [4:0]   standard 5-bit diagnostic layout NV16/DZ8/OF4/UF2/NX1:
//                  ops 0-3 pass HardFloat exceptionFlags through (same bit
//                  order as RISC-V fflags); op 4 maps RecFNToIN
//                  intExceptionFlags {invalid, overflow, inexact} as
//                  invalid|overflow -> NV, inexact -> NX.
package p03

import chisel3._
import chisel3.util.MuxLookup
import hardfloat.consts
import hardfloat.{
  AddRecFN,
  MulAddRecFN,
  MulRecFN,
  RecFNToIN,
  RecFNToRecFN,
  fNFromRecFN,
  recFNFromFN,
}

class NumericProbe extends Module {
  private val rne = consts.round_near_even
  private val tininessAfter = consts.tininess_afterRounding

  val op = IO(Input(UInt(3.W)))
  val a = IO(Input(UInt(32.W)))
  val b = IO(Input(UInt(32.W)))
  val c = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))
  val flags = IO(Output(UInt(5.W)))

  // IEEE-754 f32 bit patterns -> recoded f32 at every operand boundary.
  private val recA = recFNFromFN(8, 24, a)
  private val recB = recFNFromFN(8, 24, b)

  // op 0: f32 fused multiply-add, single rounding.
  private val madd = Module(new MulAddRecFN(8, 24))
  madd.io.op := 0.U(2.W) // op(1)=0, op(0)=0: a*b + c
  madd.io.a := recA
  madd.io.b := recB
  madd.io.c := recFNFromFN(8, 24, c)
  madd.io.roundingMode := rne
  madd.io.detectTininess := tininessAfter

  // op 1: f32 add. (AddRecFN's detectTininess port is Bool, unlike the
  // UInt(1.W) port of MulAddRecFN.)
  private val add = Module(new AddRecFN(8, 24))
  add.io.subOp := false.B
  add.io.a := recA
  add.io.b := recB
  add.io.roundingMode := rne
  add.io.detectTininess := tininessAfter.asBool

  // op 2: f32 mul. MulRecFN takes recoded f32 operands and returns the
  // rounded f32 result with flags directly (internally: MulRawFN product +
  // RoundRawFNToRecFN with option 0).
  private val mul = Module(new MulRecFN(8, 24))
  mul.io.a := recA
  mul.io.b := recB
  mul.io.roundingMode := rne
  mul.io.detectTininess := tininessAfter.asBool

  // op 3: f32 -> BF16. Narrowing conversion in the recoded domain; the 17-bit
  // recoded output is converted back to the raw 16-bit IEEE-style pattern.
  private val toBf16 = Module(new RecFNToRecFN(8, 24, 8, 8))
  toBf16.io.in := recA
  toBf16.io.roundingMode := rne
  toBf16.io.detectTininess := tininessAfter

  // op 4: f32 -> signed 32-bit integer, RNE.
  private val toI32 = Module(new RecFNToIN(8, 24, 32))
  toI32.io.in := recA
  toI32.io.roundingMode := rne
  toI32.io.signedOut := true.B

  // RecFNToIN intExceptionFlags = invalid ## overflow ## inexact (bits 2..0).
  // Map to the standard 5-bit layout: invalid|overflow -> NV(16), inexact -> NX(1).
  private val i32Flags: UInt =
    toI32.io.intExceptionFlags(2, 1).orR ## 0.U(3.W) ## toI32.io.intExceptionFlags(0)

  out := MuxLookup(op, 0.U)(Seq(
    0.U -> fNFromRecFN(8, 24, madd.io.out),
    1.U -> fNFromRecFN(8, 24, add.io.out),
    2.U -> fNFromRecFN(8, 24, mul.io.out),
    3.U -> fNFromRecFN(8, 8, toBf16.io.out),
    4.U -> toI32.io.out,
  ))
  flags := MuxLookup(op, 0.U)(Seq(
    0.U -> madd.io.exceptionFlags,
    1.U -> add.io.exceptionFlags,
    2.U -> mul.io.exceptionFlags,
    3.U -> toBf16.io.exceptionFlags,
    4.U -> i32Flags,
  ))
}
