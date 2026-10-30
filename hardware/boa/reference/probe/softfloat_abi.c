/* SPDX-License-Identifier: BSD-3-Clause
 *
 * P03 native ABI reporter for the SoftFloat ctypes binding.
 *
 * Compiled by the parent against the pinned image headers
 * (/opt/softfloat/include, CC/CXX from the image ENV); no libsoftfloat link
 * is required: every reported fact is a compile-time size of the pinned
 * public headers.
 *   suggested build: $CC -I/opt/softfloat/include -O2 -o softfloat_abi \
 *                    softfloat_abi.c
 *
 * The binding (boa_reference/softfloat_binding.py) loads this program's one
 * JSON line and must map its ctypes argument/return types from these probed
 * sizes; it never assumes LP64 fast-type widths.
 *
 * Output: exactly one JSON line with a final newline, fixed key order:
 * {"sizeof_float32_t":..,"alignof_float32_t":..,"sizeof_uint_fast8_t":..,
 *  "sizeof_int_fast32_t":..,"sizeof_uint_fast32_t":..,"sizeof_bool":..}
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "softfloat_types.h"

int main(void) {
  printf(
      "{\"sizeof_float32_t\":%zu,"
      "\"alignof_float32_t\":%zu,"
      "\"sizeof_uint_fast8_t\":%zu,"
      "\"sizeof_int_fast32_t\":%zu,"
      "\"sizeof_uint_fast32_t\":%zu,"
      "\"sizeof_bool\":%zu}\n",
      sizeof(float32_t), (size_t)_Alignof(float32_t), sizeof(uint_fast8_t),
      sizeof(int_fast32_t), sizeof(uint_fast32_t), sizeof(bool));
  return 0;
}
