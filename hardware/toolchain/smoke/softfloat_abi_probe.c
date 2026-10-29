/* SPDX-License-Identifier: BSD-3-Clause
 *
 * P02 SoftFloat ABI probe.
 *
 * Compiled once by the parent at image build time (against the pinned
 * /opt/softfloat headers) into softfloat_abi_probe; smoke/softfloat.py invokes
 * it to verify the native ABI facts its ctypes mappings assume, before any
 * vector is run. No linking against libsoftfloat is required: all facts are
 * compile-time sizes of the pinned public headers.
 *
 * Suggested parent build command (image build, CC is the pinned GCC 13.2):
 *   $CC -I/opt/softfloat/include -O2 -o /opt/softfloat/bin/softfloat_abi_probe
 * \ /workspace/hardware/toolchain/smoke/softfloat_abi_probe.c
 *
 * Output: one JSON line, no trailing newline beyond the final \n.
 */

#include <stdint.h>
#include <stdio.h>

#include "softfloat_types.h"

int main(void) {
  printf(
      "{\"sizeof_uint_fast8_t\":%zu,"
      "\"sizeof_float32_t\":%zu,"
      "\"alignof_float32_t\":%zu,"
      "\"sizeof_uint32_t\":%zu}\n",
      sizeof(uint_fast8_t), sizeof(float32_t), (size_t)_Alignof(float32_t),
      sizeof(uint32_t));
  return 0;
}
