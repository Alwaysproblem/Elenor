// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
package coralnpu

/** RV32 fixture encoders use unsigned Long arithmetic before producing BigInt words. */
object W2_03Encoding {
  def addi(rd: Int, rs1: Int, imm: Int): BigInt =
    BigInt(((imm.toLong & 0xfffL) << 20) | (rs1.toLong << 15) | (rd.toLong << 7) | 0x13L)

  def lui(rd: Int, imm20: Int): BigInt =
    BigInt(((imm20.toLong & 0xfffffL) << 12) | (rd.toLong << 7) | 0x37L)

  def mul(rd: Int, rs1: Int, rs2: Int): BigInt =
    BigInt((1L << 25) | (rs2.toLong << 20) | (rs1.toLong << 15) | (rd.toLong << 7) | 0x33L)

  def div(rd: Int, rs1: Int, rs2: Int): BigInt =
    BigInt((1L << 25) | (rs2.toLong << 20) | (rs1.toLong << 15) | (4L << 12) | (rd.toLong << 7) | 0x33L)

  def sw(rs2: Int, rs1: Int, imm: Int): BigInt = {
    val i = imm.toLong & 0xfffL
    BigInt(((i >> 5) << 25) | (rs2.toLong << 20) | (rs1.toLong << 15) |
      (2L << 12) | ((i & 0x1fL) << 7) | 0x23L)
  }

  def lw(rd: Int, rs1: Int, imm: Int): BigInt =
    BigInt(((imm.toLong & 0xfffL) << 20) | (rs1.toLong << 15) | (2L << 12) | (rd.toLong << 7) | 0x03L)

  def csrrs(rd: Int, csr: Int, rs1: Int): BigInt =
    BigInt((csr.toLong << 20) | (rs1.toLong << 15) | (2L << 12) | (rd.toLong << 7) | 0x73L)

  def bne(rs1: Int, rs2: Int, imm: Int): BigInt = {
    val i = imm.toLong & 0x1fffL
    BigInt((((i >> 12) & 1L) << 31) | (((i >> 5) & 0x3fL) << 25) |
      (rs2.toLong << 20) | (rs1.toLong << 15) | (1L << 12) |
      (((i >> 1) & 0xfL) << 8) | (((i >> 11) & 1L) << 7) | 0x63L)
  }

  val WFI: BigInt = BigInt("10500073", 16)
  val NOP: BigInt = addi(0, 0, 0)

  def line(insts: BigInt*): BigInt = {
    require(insts.length <= 4)
    var row = BigInt(0)
    var i = 0
    while (i < 4) {
      val instruction = if (i < insts.length) insts(i) else NOP
      row |= instruction << (32 * i)
      i += 1
    }
    row
  }

  def nopLines(base: BigInt, n: Int): Seq[(BigInt, BigInt)] =
    (0 until n).map(i => (base + 16 * i) -> line())
}
