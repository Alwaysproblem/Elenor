/* Minimal freestanding entry (C018): set gp/sp and tail-call the measured control sequence.
 * No crt, no libc, no semihosting: this image is linked to be measured, never run. */
.section .text._start,"ax",@progbits
.option push
.option norelax
.globl _start
.type _start, @function
_start:
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop
  la sp, __stack_end
  tail cn_bootstrap_entry
.size _start, .-_start
.option pop
