#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""P02 SoftFloat 3e ctypes smoke (build product check, NOT the P03 numeric oracle).

Loads the pinned PIC libsoftfloat.so, sets explicit rounding/tininess state, and
calls f32_add / f32_mulAdd against known exact bit patterns (IEEE-754 FP32,
roundNearestTiesToEven, RISC-V tininess-after-rounding). Writes deterministic
actual/golden JSON. Any mismatch or library/ABI problem -> nonzero exit.

Usage:
    softfloat.py <actual.json> <golden.json> [<libsoftfloat.so> [<abi_probe>]]

    argv1  actual JSON output path
    argv2  golden JSON output path
    argv3  optional shared-library override, else
           /opt/softfloat/lib/libsoftfloat.so (pinned image path)
    argv4  optional ABI probe binary, else
           /opt/softfloat/bin/softfloat_abi_probe

The compiled ABI probe is mandatory: native uint_fast8_t, uint32_t and
float32_t sizes/alignment must match ctypes before any numeric calls.

JSON convention (P02Runner): vectors entries carry exactly
{"op","a","b","c","out","flags"} with "0x"-prefixed lowercase hex strings,
plus fixed top-level metadata. No timestamps, no paths, sort_keys=True.

Exit codes: 0 pass, 1 mismatch/library/IO failure, 2 usage error.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys

TOOL = "softfloat-f32-smoke"
DEFAULT_LIB = "/opt/softfloat/lib/libsoftfloat.so"
DEFAULT_PROBE = "/opt/softfloat/bin/softfloat_abi_probe"

# SoftFloat 3e constants (include/softfloat.h, source/RISCV/specialize.h):
# softfloat_round_near_even = 0; init_detectTininess for RISCV =
# softfloat_tininess_afterRounding = 1.
ROUND_NEAR_EVEN = 0
TININESS_AFTER_ROUNDING = 1

# softfloat_flag_inexact=1, underflow=2, overflow=4, infinite=8, invalid=16.
# Same bit order as HardFloat exceptionFlags (== RISC-V fflags).
FLAG_BITS = {"inexact": 1, "underflow": 2, "overflow": 4, "infinite": 8, "invalid": 16}


class F32(ctypes.Structure):
  """softfloat_types.h: typedef struct { uint32_t v; } float32_t;"""

  _fields_ = [("v", ctypes.c_uint32)]


def muladd_case(name, a, b, c, out, flags, why):
  return {
    "op": "mulAdd",
    "name": name,
    "a": f"0x{a:08x}",
    "b": f"0x{b:08x}",
    "c": f"0x{c:08x}",
    "out": f"0x{out:08x}",
    "flags": f"0x{flags:02x}",
    "why": why,
  }


def add_case(name, a, b, out, flags, why):
  case = muladd_case(name, a, b, 0, out, flags, why)
  case["op"] = "add"
  return case


def cases():
  # mulAdd cases mirror hardware/toolchain/smoke/hardfloat.cpp bit-for-bit so
  # the pinned SoftFloat build and the elaborated HardFloat RTL are checked
  # against the same mathematically justified expectations.
  return [
    add_case("add_exact", 0x3FC00000, 0x3F000000, 0x40000000, 0x00, "1.5+0.5=2.0 exact"),
    add_case(
      "add_tie_to_even",
      0x3F800000,
      0x33800000,
      0x3F800000,
      0x01,
      "1.0+2^-24 exact tie; RNE to even -> 1.0, inexact",
    ),
    add_case("add_normal_exact", 0x00800000, 0x00800000, 0x01000000, 0x00, "2^-126+2^-126=2^-125 exact"),
    add_case("add_zero_sign", 0x00000000, 0x80000000, 0x00000000, 0x00, "(+0)+(-0)=+0 under RNE"),
    add_case(
      "add_inf_minus_inf",
      0x7F800000,
      0xFF800000,
      0x7FC00000,
      0x10,
      "inf+(-inf) invalid -> canonical qNaN 0x7fc00000",
    ),
    muladd_case(
      "muladd_exact_basic", 0x3FC00000, 0x40000000, 0x3F000000, 0x40600000, 0x00, "1.5*2.0+0.5=3.5 exact"
    ),
    muladd_case(
      "muladd_fma_exact_neg_tiny",
      0x3F800001,
      0x3F7FFFFE,
      0xBF800000,
      0xA8800000,
      0x00,
      "(1+2^-23)(1-2^-23)-1=-2^-46 exact; distinguishes single-rounding FMA "
      "from mul-then-add (which would return 0)",
    ),
    muladd_case(
      "muladd_exact_subnormal",
      0x00800000,
      0x3F000000,
      0x00000000,
      0x00400000,
      0x00,
      "2^-126*2^-1=2^-127 exact subnormal; tininess-after keeps flags 0",
    ),
    muladd_case(
      "muladd_round_down_inexact",
      0x3F800001,
      0x3F800002,
      0x00000000,
      0x3F800003,
      0x01,
      "(1+2^-23)(1+2^-22)=1+3*2^-23+2^-45 rounds down to 1+3*2^-23, inexact",
    ),
    muladd_case(
      "muladd_overflow_to_inf",
      0x7F7FFFFF,
      0x40000000,
      0x00000000,
      0x7F800000,
      0x05,
      "2*FLT_MAX -> +inf with overflow|inexact",
    ),
    muladd_case(
      "muladd_invalid_inf_times_zero",
      0x7F800000,
      0x00000000,
      0x00000000,
      0x7FC00000,
      0x10,
      "inf*0+0 invalid -> canonical qNaN",
    ),
    muladd_case(
      "muladd_neg_zero_result",
      0xBFC00000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00,
      "(-1.5)*0+0 = +0 under RNE",
    ),
  ]


def load_library(path):
  lib = ctypes.CDLL(path)
  lib.f32_add.argtypes = [F32, F32]
  lib.f32_add.restype = F32
  lib.f32_mulAdd.argtypes = [F32, F32, F32]
  lib.f32_mulAdd.restype = F32
  # Process-global state (platform.h leaves THREAD_LOCAL undefined in the
  # pinned build); this smoke is single-threaded and sets state before every
  # call, so the values below are the ones observed by each operation.
  rounding_mode = ctypes.c_uint8.in_dll(lib, "softfloat_roundingMode")
  detect_tininess = ctypes.c_uint8.in_dll(lib, "softfloat_detectTininess")
  exception_flags = ctypes.c_uint8.in_dll(lib, "softfloat_exceptionFlags")
  return lib, rounding_mode, detect_tininess, exception_flags


def find_probe(explicit):
  probe = explicit or DEFAULT_PROBE
  if not os.path.isfile(probe) or not os.access(probe, os.X_OK):
    raise RuntimeError(f"required SoftFloat ABI probe is missing: {probe}")
  return probe


def run_probe(probe_path):
  """Validate the native header ABI against the ctypes representation."""
  proc = subprocess.run([probe_path], capture_output=True, text=True, timeout=30)
  if proc.returncode != 0:
    sys.stderr.write(
      f"{TOOL}: ABI probe {probe_path} exited {proc.returncode:d}: {proc.stderr.strip()}\n"
    )
    sys.exit(1)
  data = json.loads(proc.stdout)
  problems = []
  if data.get("sizeof_uint_fast8_t") != 1:
    problems.append(
      "uint_fast8_t is {!r} bytes, ctypes state mapping assumes 1".format(data.get("sizeof_uint_fast8_t"))
    )
  if data.get("sizeof_float32_t") != 4:
    problems.append("float32_t is {!r} bytes, F32 struct assumes 4".format(data.get("sizeof_float32_t")))
  if data.get("sizeof_uint32_t") != 4:
    problems.append("uint32_t is {!r} bytes".format(data.get("sizeof_uint32_t")))
  if data.get("alignof_float32_t") != ctypes.alignment(F32):
    problems.append("float32_t alignment does not match ctypes")
  if problems:
    sys.stderr.write("{}: ABI probe failure: {}\n".format(TOOL, "; ".join(problems)))
    sys.exit(1)
  return data


def vector_json(case):
  """Runner JSON convention: exactly op,a,b,c,out,flags per vector."""
  return {
    "op": case["op"],
    "a": case["a"],
    "b": case["b"],
    "c": case["c"],
    "out": case["out"],
    "flags": case["flags"],
  }


def write_json(path, kind, vectors, extra_meta):
  doc = {
    "tool": TOOL,
    "kind": kind,
    "count": len(vectors),
    "rounding_mode": f"rne({ROUND_NEAR_EVEN:d})",
    "detect_tininess": f"after_rounding({TININESS_AFTER_ROUNDING:d})",
    "flag_bits": FLAG_BITS,
    "vectors": vectors,
  }
  doc.update(extra_meta)
  with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(doc, f, indent=2, sort_keys=True)
    f.write("\n")


def main(argv):
  if len(argv) not in (3, 4, 5):
    sys.stderr.write("usage: softfloat.py <actual.json> <golden.json> [<libsoftfloat.so> [<abi_probe>]]\n")
    return 2

  actual_path, golden_path = argv[1], argv[2]
  lib_path = argv[3] if len(argv) >= 4 else DEFAULT_LIB
  probe_explicit = argv[4] if len(argv) >= 5 else None

  try:
    lib, rounding_mode, detect_tininess, exception_flags = load_library(lib_path)
  except OSError as exc:
    sys.stderr.write(f"{TOOL}: cannot load SoftFloat library '{lib_path}': {exc}\n")
    return 1
  except AttributeError as exc:
    sys.stderr.write(f"{TOOL}: library '{lib_path}' is missing required symbols: {exc}\n")
    return 1

  probe_data = run_probe(find_probe(probe_explicit))

  expected = cases()
  measured = []
  mismatches = 0
  for case in expected:
    rounding_mode.value = ROUND_NEAR_EVEN
    detect_tininess.value = TININESS_AFTER_ROUNDING
    exception_flags.value = 0

    a = F32(int(case["a"], 16))
    b = F32(int(case["b"], 16))
    c = F32(int(case["c"], 16))
    if case["op"] == "mulAdd":
      result = lib.f32_mulAdd(a, b, c)
    else:
      result = lib.f32_add(a, b)
    got_flags = exception_flags.value

    row = vector_json(case)
    row["out"] = f"0x{result.v:08x}"
    row["flags"] = f"0x{got_flags:02x}"
    measured.append(row)

    if result.v != int(case["out"], 16) or got_flags != int(case["flags"], 16):
      mismatches += 1
      sys.stderr.write(
        "{}: MISMATCH {} a={} b={} c={} measured(out={} flags={}) expected(out={} flags={}) [{}]\n".format(
          TOOL,
          case["name"],
          case["a"],
          case["b"],
          case["c"],
          row["out"],
          row["flags"],
          case["out"],
          case["flags"],
          case["why"],
        )
      )

  probe_meta = {"abi_probe": probe_data}
  try:
    write_json(golden_path, "golden", [vector_json(c) for c in expected], {})
    write_json(actual_path, "actual", measured, probe_meta)
  except OSError as exc:
    sys.stderr.write(f"{TOOL}: failed writing result files: {exc}\n")
    return 1

  if mismatches:
    sys.stderr.write(f"{TOOL}: {mismatches:d}/{len(expected):d} vectors mismatched\n")
    return 1
  sys.stdout.write(f"{TOOL}: PASS {len(expected):d}/{len(expected):d} vectors matched (lib={lib_path})\n")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
