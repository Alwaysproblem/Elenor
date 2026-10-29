/* SPDX-License-Identifier: BSD-3-Clause
 *
 * P02 RISC-V smoke linker script.
 *
 * Bare-metal rv64gc_zicsr_zifencei/lp64d ELF:
 *   - ENTRY(_start), .text.init placed first so e_entry == 0x80000000
 *   - .riscv.attributes is preserved on purpose: the runner reads
 *     Tag_RISCV_arch (rv64 + m/a/f/d/c/zicsr/zifencei) and
 *     Tag_RISCV_float_abi (double) from `readelf -A`
 *   - stack reservation + __stack_top symbol for completeness
 */

OUTPUT_FORMAT("elf64-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 16M
}

SECTIONS
{
    .text :
    {
        KEEP(*(.text.init))
        *(.text .text.*)
    } > RAM

    .rodata : ALIGN(8)
    {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    } > RAM

    .data : ALIGN(8)
    {
        __global_pointer$ = . + 0x800;
        *(.sdata .sdata.*)
        *(.data .data.*)
        *(.got .got.*)
    } > RAM

    .bss (NOLOAD) : ALIGN(8)
    {
        __bss_start = .;
        *(.sbss .sbss.*)
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(16);
        __bss_end = .;
    } > RAM

    .stack (NOLOAD) : ALIGN(16)
    {
        . += 0x4000;
        __stack_top = .;
    } > RAM

    /DISCARD/ :
    {
        *(.eh_frame .eh_frame_hdr)
        *(.comment)
    }
}
