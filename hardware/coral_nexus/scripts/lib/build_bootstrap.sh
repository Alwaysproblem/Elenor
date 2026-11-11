#!/usr/bin/env bash
# Build one bootstrap ELF with the pinned C004 toolchain (C018). No bazel, no crt, no libc: freestanding
# -nostdlib with the upstream rv32 arch/abi/model flags (toolchain/cc_toolchain_config.bzl:164-178).
# usage: build_bootstrap.sh <toolchain-bin> <src-dir> <include-dir> <out-dir> <elf-name>
set -euo pipefail
bin="$1"; src="$2"; inc="$3"; out="$4"; name="$5"
mkdir -p "$out"
MARCH="rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma"
obj="$out/${name}.o"
"$bin/clang" --target=riscv32-unknown-elf -march="$MARCH" -mabi=ilp32 -mcmodel=medany \
  -ffreestanding -fno-builtin -fno-stack-protector -fno-jump-tables \
  -ffunction-sections -fdata-sections -Oz -g -fstack-usage \
  -Wall -Wextra -Werror -I"$inc" -c "$src/bootstrap_control.c" -o "$obj"
"$bin/clang" --target=riscv32-unknown-elf -march="$MARCH" -mabi=ilp32 -mcmodel=medany \
  -ffreestanding -fno-builtin -Wa,--no-warn -c "$src/bootstrap_start.S" -o "$out/${name}_start.o"
"$bin/riscv64-unknown-elf-ld.bfd" -m elf32lriscv -T "$out/${name}.ld" -Map "$out/${name}.map" \
  --gc-sections -o "$out/${name}.elf" "$out/${name}_start.o" "$obj"
"$bin/llvm-objdump" -d "$out/${name}.elf" > "$out/${name}.dis"
"$bin/llvm-size" -A "$out/${name}.elf" > "$out/${name}.size"
