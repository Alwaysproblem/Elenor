/* C021 scalar oracle selftest linker layout: ITCM 0x00000000 (8 KiB), DTCM 0x00010000 (32 KiB). */
OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY {
  ITCM (rx)  : ORIGIN = 0x00000000, LENGTH = 0x00002000
  DTCM (rw)  : ORIGIN = 0x00010000, LENGTH = 0x00008000
}

SECTIONS {
  .text 0x00000000 : ALIGN(16) {
    *(.text._start)
    *(.text .text.*)
    *(.rodata .rodata.*)
  } > ITCM

  .data 0x00010000 : ALIGN(16) {
    __global_pointer$ = . + 0x800;
    *(.data .data.*)
    *(.sdata .sdata.*)
  } > DTCM

  .bss (NOLOAD) : ALIGN(16) {
    *(.bss .bss.*)
    *(.sbss .sbss.*)
  } > DTCM

  .stack 0x00017f00 (NOLOAD) : ALIGN(16) {
    . = . + 0x100;
    __stack_end = .;
  } > DTCM
}
