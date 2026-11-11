/* Bootstrap link-layout template (C018). @@TEXT_BASE@@ / @@DATA_BASE@@ / @@STACK_BYTES@@ are substituted
 * per candidate layout by scripts/lib/measure_kernel.py; never a hand-typed literal in a fixed encoding. */
OUTPUT_ARCH(riscv)
ENTRY(_start)

/* Region bases follow the upstream generator (rules/linker.bzl): ITCM at 0x0, DTCM at 0x10000.
 * @@TEXT_BASE@@ / @@DATA_BASE@@ select the SLOT inside each region; the two regions never overlap. */
MEMORY {
  ITCM (rx)  : ORIGIN = 0x00000000 + @@TEXT_BASE@@, LENGTH = 8K
  DTCM (rwx) : ORIGIN = 0x00010000 + @@DATA_BASE@@, LENGTH = 32K
}

STACK_BYTES = @@STACK_BYTES@@;

SECTIONS {
  .text : ALIGN(16) {
    *(.text._start)
    *(.text .text.*)
    . = ALIGN(16);
  } > ITCM

  .rodata : ALIGN(16) {
    *(.srodata .srodata.*)
    *(.rodata .rodata.*)
    . = ALIGN(16);
  } > ITCM

  .data : ALIGN(16) {
    __data_start = .;
    _global_pointer = . + 0x800;
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
    *(.data .data.*)
    . = ALIGN(16);
    __data_end = .;
  } > DTCM

  .bss (NOLOAD) : ALIGN(16) {
    __bss_start = .;
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    . = ALIGN(16);
    __bss_end = .;
  } > DTCM

  .stack (NOLOAD) : ALIGN(16) {
    . += STACK_BYTES;
    __stack_end = .;
  } > DTCM

  /DISCARD/ : { *(.eh_frame .eh_frame_hdr .comment .note*) }
}
