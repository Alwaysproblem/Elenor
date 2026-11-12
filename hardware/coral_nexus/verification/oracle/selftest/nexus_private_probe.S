/* Nexus-private CSR probe ELF (C021 / W7-02a & W7-02c):
 *   1. On raw upstream Spike (`enable_nexus_extend=False`), `csrw 0x7ca, a0` (`nexus.mregion.start`)
 *      raises illegal-instruction trap (mcause=2) -> proves upstream Spike is `NONE` for Nexus-private CSRs.
 *   2. With `nexus_extend` enabled (`enable_nexus_extend=True`), executes the sealed helper CSRs
 *      (`nexus.mctx.dispatch_snapshot` = 0x7c8, `nexus.msnap.config` = 0x7c9) without trapping,
 *      reads back the previous CSR value into rd, and stores the result to DTCM 0x00010100.
 */
.section .text._start,"ax",@progbits
.option push
.option norelax
.globl _start
_start:
  li sp, 0x00017ff0
  li s0, 0x00010100
  la t0, nexus_trap
  csrw mtvec, t0

  li a0, 0x13572468
  csrrw a1, 0x7c8, a0      /* nexus.mctx.dispatch_snapshot: traps on raw Spike; rd=0 under nexus_extend */
  li a2, 0x00000011
  csrrs a3, 0x7c8, a2      /* rd=0x13572468, new=0x13572479 under nexus_extend */
  csrr a4, 0x7c8           /* rd=0x13572479 */
  sw a1, 0(s0)
  sw a3, 4(s0)
  sw a4, 8(s0)
  li a5, 0
  sw a5, 12(s0)            /* [3] = 0 -> no trap */
  ebreak

.balign 4
nexus_trap:
  csrr t1, mcause
  sw t1, 12(s0)            /* [3] = mcause (2 on raw Spike) */
  ebreak
.option pop
