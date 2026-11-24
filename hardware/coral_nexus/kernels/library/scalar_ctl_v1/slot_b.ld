/* K-SCALAR-ADD7 / scalar_ctl_v1, slot B (program_id 2): the second fixed pre-linked address (C038).
 *
 * The regions are not chosen here. They are slot 1 of the sealed target: itcm_bytes and dtcm_bytes of
 * configs/cs1_c*.yaml split by program_slots over the pinned upstream origins (rules/linker.bzl). Before
 * anything is linked, scripts/lib/scalar_kernel_build.py reads both regions back and refuses this script
 * unless it states exactly that slot, disjoint from slot A.
 *
 * The image is text only: the kernel owns no static data and no stack, and every byte it touches is
 * Frame-relative to the launch register a0. DTCM is declared so that a stray data section lands in this
 * slot's own range, where the build refuses it as undeclared, never in another slot.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
  ITCM (rx)  : ORIGIN = 0x00001000, LENGTH = 0x00001000
  DTCM (rwx) : ORIGIN = 0x00014000, LENGTH = 0x00004000
}

SECTIONS {
  .text : ALIGN(16) {
    *(.text._start)
    *(.text .text.*)
  } > ITCM

  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame .eh_frame_hdr) }
}
