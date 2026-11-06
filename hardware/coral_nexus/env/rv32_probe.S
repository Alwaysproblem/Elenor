/* Cross-compiled with the sha256-locked toolchain_coralnpu_v2 GCC using the
 * exact upstream march/mabi (cc_toolchain_config.bzl @ f795221):
 *   -march=rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma
 *   -mabi=ilp32 -mcmodel=medany -nostdlib
 * verify_inside.sh asserts the produced artifact is an ELF32 RISC-V image. */
.section .text
.globl _start
_start:
  li      a0, 0
  csrw    mstatus, zero
  wfi
  j       _start
