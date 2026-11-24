/* K-SCALAR-ADD7 / scalar_ctl_v1, slot A (program_id 1): one of the two fixed pre-linked addresses (C038).
 *
 * The regions are not chosen here. They are slot 0 of the sealed target: itcm_bytes and dtcm_bytes of
 * configs/cs1_c*.yaml split by program_slots over the pinned upstream origins (rules/linker.bzl). Before
 * anything is linked, scripts/lib/scalar_kernel_build.py reads both regions back and refuses this script
 * unless it states exactly that slot.
 *
 * The image is text only: the kernel owns no static data and no stack, and every byte it touches is
 * Frame-relative to the launch register a0. DTCM is declared so that a stray data section lands in this
 * slot's own range, where the build refuses it as undeclared, never in another slot.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
  ITCM (rx)  : ORIGIN = 0x00000000, LENGTH = 0x00001000
  DTCM (rwx) : ORIGIN = 0x00010000, LENGTH = 0x00004000
}

SECTIONS {
  .text : ALIGN(16) {
    *(.text._start)
    *(.text .text.*)
  } > ITCM

  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame .eh_frame_hdr) }
}
