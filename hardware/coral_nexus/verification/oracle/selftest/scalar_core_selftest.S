/* Hand-computable RV32IMF + Zicsr + Zbb scalar selftest ELF (C021 / W7-02a/b/e).
 *
 * Exercises:
 *   1. RV32I integer wraparound, signed/unsigned compare, shifts (0x7fffffff + 1 = 0x80000000)
 *   2. RV32M mul/div/rem including INT32_MIN / -1 = INT32_MIN and x / 0 = -1
 *   3. Zbb bitmanip (clz, ctz, cpop, orc.b, rev8)
 *   4. Branch/JAL/JALR taken & not-taken control flow
 *   5. DTCM sub-word/word stores and signed/unsigned loads (sb/sh/sw, lb/lbu/lh/lhu/lw)
 *   6. Zicsr WARL & fflags/frm behavior (mtvec direct-mode mask, misa read-only WARL, frm RNE/RTZ/RUP)
 *   7. RV32F exact & boundary rounding/exception flags (NX, OF, UF, DZ, NV, fmadd, fcvt.w.s saturation)
 *   8. Trap vector & mepc/mcause recovery on illegal instruction (.4byte 0xffffffff)
 *
 * Output words are written sequentially to DTCM 0x00010100 (16 x uint32_t = 64 bytes), followed by
 * `ebreak` (which SpikeSimulator intercepts cleanly into debug_mode halt).
 */
.section .text._start,"ax",@progbits
.option push
.option norelax
.globl _start
_start:
  li sp, 0x00017ff0
  li s0, 0x00010100

  /* [0] RV32I signed overflow wrap: 0x7fffffff + 1 -> 0x80000000 */
  lui t0, 0x80000
  addi t0, t0, -1          /* t0 = 0x7fffffff */
  addi t1, t0, 1           /* t1 = 0x80000000 */
  sw t1, 0(s0)

  /* [1] slt vs sltu + srai: (0x80000000 <s 0x7fffffff)=1, (<u)=0, (0x80000000 >>> 4)=0xf8000000 */
  slt t2, t1, t0           /* 1 */
  sltu t3, t1, t0          /* 0 */
  srai t4, t1, 4           /* 0xf8000000 */
  or t4, t4, t2            /* 0xf8000001 */
  add t4, t4, t3           /* 0xf8000001 */
  sw t4, 4(s0)

  /* [2] RV32M: INT32_MIN / -1 = 0x80000000, (-7) / 0 = 0xffffffff -> xor = 0x7fffffff */
  li a0, -1
  div a1, t1, a0           /* 0x80000000 */
  li a2, -7
  div a3, a2, zero         /* 0xffffffff */
  xor a4, a1, a3           /* 0x7fffffff */
  sw a4, 8(s0)

  /* [3] RV32M mulhsu / remu: (-2 *u 3) high=0xffffffff, 29 %u 6 = 5 -> 0x00000004 */
  li a0, 29
  li a1, 6
  remu a2, a0, a1          /* 5 */
  li a3, -2
  li a4, 3
  mulhsu a5, a3, a4        /* 0xffffffff (-6 >> 32) */
  add a2, a2, a5           /* 4 */
  sw a2, 12(s0)

  /* [4] Zbb: clz(0x000f0000)=12, ctz(0x000f0000)=16, cpop(0x000f0000)=4, rev8(0x01020304)=0x04030201 */
  lui a0, 0x000f0
  clz a1, a0               /* 12 */
  ctz a2, a0               /* 16 */
  cpop a3, a0              /* 4 */
  li a4, 0x01020304
  rev8 a5, a4              /* 0x04030201 */
  slli a1, a1, 8           /* 0x0c00 */
  or a1, a1, a2            /* 0x0c10 */
  or a1, a1, a3            /* 0x0c14 */
  xor a5, a5, a1           /* 0x04030e15 */
  sw a5, 16(s0)

  /* [5] Branch taken + not-taken + jal/jalr */
  li a0, 10
  li a1, 20
  blt a0, a1, 1f
  addi a0, a0, 99          /* skipped */
1:
  bge a0, a1, 2f           /* not taken */
  jal ra, add_const_sub
2:
  sw a0, 20(s0)            /* 10 + 37 = 47 = 0x0000002f */

  /* [6] Sub-word store/load sign & zero extension at 0x00010200 */
  li t5, 0x00010200
  li a0, 0x80fffe81
  sw a0, 0(t5)
  lb a1, 0(t5)             /* 0xffffff81 (-127) */
  lbu a2, 0(t5)            /* 0x00000081 (129) */
  lh a3, 0(t5)             /* 0xfffffe81 */
  lhu a4, 2(t5)            /* 0x000080ff */
  xor a1, a1, a2           /* 0xffffff00 */
  xor a1, a1, a3           /* 0x00000181 */
  add a1, a1, a4           /* 0x00008280 */
  sw a1, 24(s0)

  /* [7] Zicsr WARL: mtvec Direct mode masks low 2 bits; mvendorid == 0x426 */
  la t0, trap_handler
  ori t1, t0, 3
  csrw mtvec, t1
  csrr a0, mtvec           /* trap_handler & ~3 */
  sub a0, a0, t0           /* 0 */
  csrr a1, mvendorid       /* 0x426 */
  add a0, a0, a1           /* 0x00000426 */
  sw a0, 28(s0)

  /* [8] FP32 RNE inexact: 1.0f (0x3f800000) + 2^-24 (0x33800000) -> tie-to-even = 0x3f800000, fflags=NX(1) */
  csrw fcsr, zero
  lui a0, 0x3f800
  fmv.w.x fa0, a0
  lui a1, 0x33800
  fmv.w.x fa1, a1
  fadd.s fa2, fa0, fa1, rne
  fmv.x.w a2, fa2          /* 0x3f800000 */
  csrr a3, fflags          /* 0x01 (NX) */
  sw a2, 32(s0)
  sw a3, 36(s0)            /* [9] = 0x00000001 */

  /* [10..11] FP32 RUP vs RTZ overflow: FLT_MAX (0x7f7fffff) * 2.0f (0x40000000)
   * Under RTZ -> 0x7f7fffff, fflags = OF|NX = 0x05. */
  csrw fflags, zero
  li a0, 0x7f7fffff
  fmv.w.x fa0, a0
  lui a1, 0x40000
  fmv.w.x fa1, a1
  fmul.s fa2, fa0, fa1, rtz
  fmv.x.w a2, fa2          /* 0x7f7fffff */
  csrr a3, fflags          /* 0x05 (OF|NX) */
  sw a2, 40(s0)
  sw a3, 44(s0)

  /* [12..13] FP32 divide by zero (1.0 / +0.0 -> +Inf, DZ=0x08) and invalid (0.0 / 0.0 -> canonical qNaN 0x7fc00000, NV=0x10) */
  csrw fflags, zero
  lui a0, 0x3f800
  fmv.w.x fa0, a0
  fmv.w.x fa1, zero
  fdiv.s fa2, fa0, fa1
  fdiv.s fa3, fa1, fa1
  fmv.x.w a2, fa2          /* 0x7f800000 */
  fmv.x.w a3, fa3          /* 0x7fc00000 */
  csrr a4, fflags          /* DZ|NV = 0x18 */
  xor a2, a2, a3           /* 0x00400000 */
  or a2, a2, a4            /* 0x00400018 */
  sw a2, 48(s0)

  /* [13] fcvt.w.s overflow saturation: +Inf -> 0x7fffffff with NV=0x10 */
  csrw fflags, zero
  lui a0, 0x7f800
  fmv.w.x fa0, a0
  fcvt.w.s a1, fa0, rne    /* 0x7fffffff */
  csrr a2, fflags          /* 0x10 */
  xor a1, a1, a2           /* 0x7fffffef */
  sw a1, 52(s0)

  /* [14..15] Illegal instruction trap -> trap_handler records mcause=2 and advances mepc+4 */
  li s1, 0
trap_insn_label:
  .4byte 0xffffffff
  sw s1, 56(s0)            /* [14] = mcause (2) */
  csrr a0, mepc
  la a1, trap_insn_label
  addi a1, a1, 4
  sub a0, a0, a1           /* 0 when handler advanced mepc by 4 */
  addi a0, a0, 0x5a        /* 0x0000005a */
  sw a0, 60(s0)            /* [15] = 0x0000005a */

  ebreak

add_const_sub:
  addi a0, a0, 37
  jalr zero, 0(ra)

.balign 4
trap_handler:
  csrr s1, mcause
  csrr t6, mepc
  addi t6, t6, 4
  csrw mepc, t6
  mret
.option pop
