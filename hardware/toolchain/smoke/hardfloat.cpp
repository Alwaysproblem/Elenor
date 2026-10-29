/* SPDX-License-Identifier: BSD-3-Clause
 *
 * P02 Verilator C++ harness for the HardFloat FP32 MulAddRecFN smoke top
 * (VHardFloatMulAddSmoke, generated from overlay project p02).
 *
 * Contract (P02Runner / container_check.py):
 *   - invoked as:  obj_dir/vsim <actual.json> <golden.json>
 *   - argv1 = actual JSON path, argv2 = golden JSON path (deterministic bytes:
 *     fixed key order, no timestamps, no paths, no environment data)
 *   - any input/output mismatch -> nonzero exit (explicit check here, never a
 *     disabled C assert), after both JSON files are still written
 *   - no file dependencies beyond verilated.h + VHardFloatMulAddSmoke.h
 *
 * Verification contract: known input/output bit vectors with mathematically
 * justified expected values (IEEE-754 FP32, single-rounding FMA, roundNearest-
 * TiesToEven, tininess-after-rounding). The top hardwires roundingMode=0 (RNE)
 * and detectTininess=1 (after rounding); exceptionFlags bit order is
 * HardFloat's
 * (== RISC-V fflags): bit4 invalid, bit3 infinite, bit2 overflow, bit1
 * underflow, bit0 inexact.
 */

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>

#include "VHardFloatMulAddSmoke.h"
#include "verilated.h"

namespace {

struct Vector {
  const char* name;
  uint32_t a;
  uint32_t b;
  uint32_t c;
  uint32_t out;     // expected IEEE-754 FP32 bits of a*b+c (single rounding)
  uint8_t flags;    // expected HardFloat exception flags (5 bits used)
  const char* why;  // mathematical justification for the expected bits
};

// All vectors use roundingMode=RNE, detectTininess=after-rounding.
const Vector kVectors[] = {
    {"exact_basic", 0x3fc00000u, 0x40000000u, 0x3f000000u, 0x40600000u, 0x00,
     "1.5*2.0+0.5=3.5; exact, representable, no exceptions"},
    {"fma_exact_neg_tiny", 0x3f800001u, 0x3f7ffffeu, 0xbf800000u, 0xa8800000u,
     0x00,
     "(1+2^-23)(1-2^-23)-1 = 1-2^-46-1 = -2^-46 exact in FP32 (exp -46, "
     "mantissa 0); "
     "naive mul-then-add would round the product to 1.0 first and return 0, "
     "so this pins single rounding"},
    {"exact_subnormal", 0x00800000u, 0x3f000000u, 0x00000000u, 0x00400000u,
     0x00,
     "2^-126*2^-1=2^-127, exactly representable subnormal; "
     "tininess-after-rounding "
     "raises underflow only when inexact, so flags stay 0"},
    {"add_tie_to_even", 0x3f800000u, 0x3f800000u, 0x33800000u, 0x3f800000u,
     0x01,
     "1.0+2^-24 lies exactly halfway between 1.0 and 1+2^-23; RNE ties to even "
     "-> 1.0, inexact"},
    {"mul_round_down_inexact", 0x3f800001u, 0x3f800002u, 0x00000000u,
     0x3f800003u, 0x01,
     "(1+2^-23)(1+2^-22)=1+3*2^-23+2^-45; nearest grid point 1+3*2^-23, "
     "remainder < half ulp, "
     "rounds down, inexact"},
    {"overflow_to_inf", 0x7f7fffffu, 0x40000000u, 0x00000000u, 0x7f800000u,
     0x05, "2*FLT_MAX overflows; RNE rounds to +inf with overflow|inexact"},
    {"invalid_inf_times_zero", 0x7f800000u, 0x00000000u, 0x00000000u,
     0x7fc00000u, 0x10,
     "inf*0+0 is invalid; HardFloat emits the canonical qNaN 0x7fc00000 with "
     "invalid flag"},
    {"neg_zero_result", 0xbfc00000u, 0x00000000u, 0x00000000u, 0x00000000u,
     0x00, "(-1.5)*0 = -0; -0 + (+0) = +0 under RNE, no exceptions"},
};

constexpr int kVectorCount =
    static_cast<int>(sizeof(kVectors) / sizeof(kVectors[0]));
constexpr const char* kTool = "hardfloat-fp32-muladd-smoke";

/* Writes the deterministic result file. `kind` is "golden" (expected values) or
 * "actual" (measured values). Vector keys are exactly a,b,c,out,flags. */
bool writeJson(const char* path, const char* kind, const uint32_t* out,
               const uint8_t* flags) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    std::fprintf(stderr, "%s: cannot open '%s' for writing\n", kTool, path);
    return false;
  }
  f << "{\n";
  f << "  \"tool\": \"" << kTool << "\",\n";
  f << "  \"kind\": \"" << kind << "\",\n";
  f << "  \"count\": " << kVectorCount << ",\n";
  f << "  \"rounding_mode\": \"rne(0)\",\n";
  f << "  \"detect_tininess\": \"after_rounding(1)\",\n";
  f << "  \"flag_bits\": {\"inexact\": 1, \"underflow\": 2, \"overflow\": 4,"
       " \"infinite\": 8, \"invalid\": 16},\n";
  f << "  \"vectors\": [\n";
  f << std::hex << std::setfill('0');
  for (int i = 0; i < kVectorCount; ++i) {
    const Vector& v = kVectors[i];
    f << "    {\"a\": \"0x" << std::setw(8) << v.a << "\", \"b\": \"0x"
      << std::setw(8) << v.b << "\", \"c\": \"0x" << std::setw(8) << v.c
      << "\", \"out\": \"0x" << std::setw(8) << out[i] << "\", \"flags\": \"0x"
      << std::setw(2) << static_cast<int>(flags[i]) << "\"}"
      << (i + 1 < kVectorCount ? "," : "") << "\n";
  }
  f << std::dec << std::setfill(' ');
  f << "  ]\n";
  f << "}\n";
  f.flush();
  if (!f.good()) {
    std::fprintf(stderr, "%s: failed while writing '%s'\n", kTool, path);
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <actual.json> <golden.json>\n",
                 argc > 0 ? argv[0] : "vsim");
    return 2;
  }

  auto* top = new VHardFloatMulAddSmoke;

  top->clock = 0;
  top->reset = 1;
  top->eval();
  top->reset = 0;
  top->eval();

  uint32_t measuredOut[kVectorCount];
  uint8_t measuredFlags[kVectorCount];
  uint32_t expectedOut[kVectorCount];
  uint8_t expectedFlags[kVectorCount];
  int mismatches = 0;

  for (int i = 0; i < kVectorCount; ++i) {
    const Vector& v = kVectors[i];
    top->a = v.a;
    top->b = v.b;
    top->c = v.c;
    top->eval();     // combinational settle
    top->clock = 1;  // one full clock pulse so RTL asserts see the vector
    top->eval();
    top->clock = 0;
    top->eval();

    measuredOut[i] = static_cast<uint32_t>(top->out);
    measuredFlags[i] = static_cast<uint8_t>(top->flags);
    expectedOut[i] = v.out;
    expectedFlags[i] = v.flags;

    if (measuredOut[i] != v.out || measuredFlags[i] != v.flags) {
      ++mismatches;
      std::fprintf(stderr,
                   "%s: MISMATCH %s a=%08x b=%08x c=%08x"
                   " measured(out=%08x flags=%02x)"
                   " expected(out=%08x flags=%02x) [%s]\n",
                   kTool, v.name, v.a, v.b, v.c, measuredOut[i],
                   static_cast<unsigned>(measuredFlags[i]), v.out,
                   static_cast<unsigned>(v.flags), v.why);
    }
  }

  // Both result files are always written, deterministically, in a fixed order
  // (golden first), even when mismatches exist.
  const bool goldenOk =
      writeJson(argv[2], "golden", expectedOut, expectedFlags);
  const bool actualOk =
      writeJson(argv[1], "actual", measuredOut, measuredFlags);

  delete top;

  if (mismatches > 0) {
    std::fprintf(stderr, "%s: %d/%d vectors mismatched\n", kTool, mismatches,
                 kVectorCount);
    return 1;
  }
  if (!goldenOk || !actualOk) {
    return 1;
  }
  std::printf("%s: PASS %d/%d vectors matched\n", kTool, kVectorCount,
              kVectorCount);
  return 0;
}
