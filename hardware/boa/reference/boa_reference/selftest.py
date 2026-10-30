"""Behavioral checks for the exact reference; not a production-model oracle."""

from __future__ import annotations

import struct
import sys
from fractions import Fraction

from . import PROFILE
from .convolution import conv2d_reference
from .generators import encoded, generate_documents, generator_hash
from .model_cases import model_checks
from .scalar import FPResult, fma_f32_rne, from_fraction, round_to_int_rne, saturate_i8
from .scalar_cases import primitive_checks
from .tensor import MatmulStep, matmul_reference


def _binary64_fma_baseline(a: int, b: int, c: int) -> int:
  """Deliberately wrong legacy computation, used only to expose double rounding."""
  values = [struct.unpack("<f", word.to_bytes(4, "little"))[0] for word in (a, b, c)]
  return int.from_bytes(struct.pack("<f", values[0] * values[1] + values[2]), "little")


def run_selftest(seed: int, tile_k: int = 8) -> dict:
  observations = [*primitive_checks(), *model_checks()]
  default_step = MatmulStep(1, (0x3F80,), (0x3F80,), True, True, 0)
  default_mm = matmul_reference(1, 1, [default_step])
  default_conv = conv2d_reference((0x3F80,), (0x3F80,), (1, 1, 1, 1), (1, 1, 1, 1))
  observations.append(
    (
      "N-P03-6/default-final-writes-f32",
      default_mm.output_bytes == bytes.fromhex("0000803f") and default_conv["output_hex"] == "0000803f",
      "FINAL without optional stages still writes the f32 result",
    )
  )
  third = from_fraction(Fraction(1, 3))
  observations.append(("exact-rational-third", third == FPResult(0x3EAAAAAB, 1), str(third)))
  huge = round_to_int_rne(0x7F000000)
  observations.append(("unbounded-integer-rounding", huge == 1 << 127, str(huge)))
  observations.append(
    (
      "integer-saturation",
      saturate_i8(huge) == 127 and saturate_i8(-huge) == -128,
      "finite integers saturate only after exact rounding and zero-point addition",
    )
  )
  fused = fma_f32_rne(0x3F800800, 0x3F800800, 0x0D800000)
  legacy = _binary64_fma_baseline(0x3F800800, 0x3F800800, 0x0D800000)
  observations.append(
    (
      "N-P03-8/single-fma-double-rounding",
      fused.bits == 0x3F801001 and legacy == 0x3F801000,
      f"exact=0x{fused.bits:08x}; binary64-then-f32=0x{legacy:08x}",
    )
  )
  first = generate_documents(seed, tile_k)
  second = generate_documents(seed, tile_k)
  deterministic = first.keys() == second.keys() and all(
    encoded(first[name]) == encoded(second[name]) for name in first
  )
  observations.append(
    (
      "N-P03-10/same-seed-identical-documents",
      deterministic,
      "all generated JSON and embedded numerical bytes compared exactly",
    )
  )
  forbidden = sorted(
    name
    for name in sys.modules
    if name == "numpy"
    or name.startswith("numpy.")
    or name == "pipeline_validator"
    or name.startswith("pipeline_validator.")
  )
  observations.append(("independent-runtime-imports", not forbidden, repr(forbidden)))
  checks = [{"id": name, "pass": bool(passed), "detail": detail} for name, passed, detail in observations]
  failed = sum(not row["pass"] for row in checks)
  return {
    "schema": 1,
    "profile": PROFILE,
    "seed": seed,
    "tile_k": tile_k,
    "generator_hash": generator_hash(),
    "status": "FAIL" if failed else "PASS",
    "counts": {"checks": len(checks), "passed": len(checks) - failed, "failed": failed},
    "checks": checks,
  }
