/* SPDX-License-Identifier: BSD-3-Clause
 *
 * P02 RISC-V rv64gc_zicsr_zifencei / lp64d bare-metal smoke.
 *
 * Build as two explicit steps so the ELF FILE symbol does not contain a
 * random GCC temporary object name (the runner supplies absolute paths):
 *   riscv64-unknown-elf-gcc -march=rv64gc_zicsr_zifencei -mabi=lp64d -c riscv.S -o smoke.o
 *   riscv64-unknown-elf-gcc -march=rv64gc_zicsr_zifencei -mabi=lp64d -nostdlib -T riscv.ld smoke.o -o smoke.elf
 *
 * The ELF is verified statically by the runner (ELF64 / EM_RISC-V / ET_EXEC,
 * entry == address of _start at 0x80000000, Tag_RISCV_arch containing rv64 and
 * the letters m a f d c zicsr zifencei, and double-float ABI in ELF e_flags), plus
 * objdump instruction-presence checks below.
 *
 * Machine-greppable verification contract:
 *
 * CHECK-SYMBOL: _start
 * CHECK-SYMBOL: smoke_amo
 * CHECK-SYMBOL: smoke_result
 * CHECK-SYMBOL: smoke_checksum
 * CHECK-SYMBOL: smoke_f32
 * CHECK-SYMBOL: smoke_f64
 * CHECK-SYMBOL: rv64_smoke_complete
 * CHECK-INSN: mul
 * CHECK-INSN: divu
 * CHECK-INSN: amoadd.d
 * CHECK-INSN: lr.d
 * CHECK-INSN: sc.d
 * CHECK-INSN: csrr
 * CHECK-INSN: fence.i
 * CHECK-INSN: c.li
 * CHECK-INSN: c.add
 * CHECK-INSN: flw
 * CHECK-INSN: fsw
 * CHECK-INSN: fmul.s
 * CHECK-INSN: fadd.s
 * CHECK-INSN: fld
 * CHECK-INSN: fsd
 * CHECK-INSN: fmul.d
 * CHECK-INSN: fadd.d
 * CHECK-INSN: fmv.x.w
 * CHECK-INSN: fmv.x.d
 *
 * This program is compiled, linked and inspected by P02, not executed on a
 * CPU or RTL backend. Its instruction mix exercises the assembler and ELF
 * target contract; no runtime checksum or CSR execution coverage is claimed.
 */

    .section .text.init, "ax", @progbits
    .globl _start
_start:
    /* I extension */
    li      t0, 12
    li      t1, 10
    add     t2, t0, t1              /* 22 */

    /* M extension */
    li      a0, 7
    li      a1, 3
    mul     a2, a0, a1              /* 21 */
    divu    a3, t2, a1              /* 7 */

    /* A extension against .data scratch */
    la      t3, smoke_amo
    li      t4, 5
    amoadd.d a4, t4, (t3)           /* old=0, mem=5 */
    lr.d    a5, (t3)                /* 5 */
    li      t5, 1
    li      t6, 8                   /* bounded SC retry */
1:
    sc.d    a6, t5, (t3)
    beq     a6, zero, 2f
    addi    t6, t6, -1
    bnez    t6, 1b
2:
    /* Zicsr: executed for decode coverage; values unused, checksum-safe. */
    csrr    s0, cycle
    csrr    s1, fflags

    /* Zifencei */
    fence.i

    /* C extension: explicit compressed encodings.
     * c.li/c.addi immediates are signed 6-bit (-32..31), so the 55 constant
     * uses a regular li; the c.li/c.addi/c.add encodings are still exercised. */
    c.li    s5, 0
    c.addi  s5, 1
    li      s6, 55
    c.add   s5, s6                  /* s5 = 56 */

    /* F extension (FP32) */
    la      t3, flt_a
    flw     fa0, 0(t3)              /* 1.5f */
    la      t3, flt_b
    flw     fa1, 0(t3)              /* 2.0f */
    fmul.s  fa2, fa0, fa1           /* 3.0f */
    la      t3, flt_c
    flw     fa3, 0(t3)              /* 0.5f */
    fadd.s  fa4, fa2, fa3           /* 3.5f */
    la      t3, smoke_f32
    fsw     fa4, 0(t3)

    /* D extension (FP64) */
    la      t3, dbl_a
    fld     ft0, 0(t3)              /* 1.5 */
    la      t3, dbl_b
    fld     ft1, 0(t3)              /* 2.0 */
    fmul.d  ft2, ft0, ft1           /* 3.0 */
    la      t3, dbl_c
    fld     ft3, 0(t3)              /* 0.25 */
    fadd.d  ft4, ft2, ft3           /* 3.25 */
    la      t3, smoke_f64
    fsd     ft4, 0(t3)

    /* Fold deterministic contributions into the checksum */
    li      s2, 0
    add     s2, s2, t2
    add     s2, s2, a2
    add     s2, s2, a3
    add     s2, s2, a4
    add     s2, s2, a5
    add     s2, s2, s5
    fmv.x.w s3, fa4                 /* sext(0x40600000) */
    add     s2, s2, s3
    fmv.x.d s4, ft4                 /* 0x400a000000000000 */
    add     s2, s2, s4
    la      t3, smoke_checksum
    sd      s2, 0(t3)

    /* Stash results for any runtime inspection (statically verified symbols) */
    la      t3, smoke_result
    sd      t2, 0(t3)
    sd      a2, 8(t3)
    sd      a3, 16(t3)
    sd      a4, 24(t3)
    sd      a5, 32(t3)
    sd      a6, 40(t3)
    sd      s5, 48(t3)
    sd      s2, 56(t3)

    /* Completion marker, then park */
    fence
    la      t3, rv64_smoke_complete
    li      t4, 0x600dcafe
    sw      t4, 0(t3)
3:
    j       3b

    .section .rodata, "a", @progbits
    .p2align 2
flt_a:  .word 0x3fc00000        /* 1.5f  */
flt_b:  .word 0x40000000        /* 2.0f  */
flt_c:  .word 0x3f000000        /* 0.5f  */
    .p2align 3
dbl_a:  .dword 0x3ff8000000000000   /* 1.5  */
dbl_b:  .dword 0x4000000000000000   /* 2.0  */
dbl_c:  .dword 0x3fd0000000000000   /* 0.25 */

    .section .data, "aw", @progbits
    .p2align 3
    .globl smoke_amo
smoke_amo:
    .dword 0
    .p2align 3
    .globl smoke_result
smoke_result:
    .zero 64
    .p2align 3
    .globl smoke_checksum
smoke_checksum:
    .zero 8
    .p2align 2
    .globl smoke_f32
smoke_f32:
    .zero 4
    .p2align 3
    .globl smoke_f64
smoke_f64:
    .zero 8
    .p2align 2
    .globl rv64_smoke_complete
rv64_smoke_complete:
    .word 0
