"""Batch exact-reference/SoftFloat/HardFloat comparison with raw failure evidence."""

from __future__ import annotations

import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from . import PROFILE
from .generators import generator_hash
from .scalar_cases import evaluate, result_dict, scalar_vectors
from .softfloat_binding import SoftFloat


def _write(path: Path, value: dict) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def _row(item: dict, result: tuple[int, int]) -> dict:
  return {"id": item["id"], "op": item["op"], **result_dict(item["op"], result)}


def _read_probe(path: Path, cases: list[dict]) -> list[dict]:
  lines = path.read_text(encoding="ascii").splitlines()
  if len(lines) != len(cases):
    raise ValueError(f"HardFloat returned {len(lines)} rows for {len(cases)} input vectors")
  result = []
  for index, (line, item) in enumerate(zip(lines, cases, strict=True)):
    columns = line.split()
    if len(columns) != 4 or columns[:2] != [item["id"], item["op"]]:
      raise ValueError(f"HardFloat response identity/shape mismatch at row {index}: {line!r}")
    bits, flags = int(columns[2], 16), int(columns[3], 16)
    width = 16 if item["op"] == "bf16" else 32
    if not 0 <= bits < 1 << width or not 0 <= flags < 32:
      raise ValueError(f"HardFloat response exceeds its declared bit width at row {index}")
    result.append(_row(item, (bits, flags)))
  return result


def crosscheck(
  seed: int,
  random_cases: int,
  probe: Path,
  abi_json: Path,
  out: Path,
  *,
  specials_only: bool = False,
  library: Path = Path("/opt/softfloat/lib/libsoftfloat.so"),
) -> dict:
  out.mkdir(parents=True, exist_ok=True)
  cases = scalar_vectors(seed, random_cases, specials_only=specials_only)
  if len({item["id"] for item in cases}) != len(cases):
    raise ValueError("numeric vector IDs must be unique")
  expected = [_row(item, (value.bits, value.flags)) for item in cases for value in [evaluate(item)]]
  _write(out / "inputs.json", {"schema": 1, "seed": seed, "vectors": cases})
  _write(out / "expected.json", {"schema": 1, "vectors": expected})
  tsv = out / "inputs.tsv"
  tsv.write_text(
    "".join(" ".join(str(item[key]) for key in ("id", "op", "a", "b", "c")) + "\n" for item in cases),
    encoding="ascii",
  )
  report: dict = {
    "schema": 1,
    "profile": PROFILE,
    "generator_hash": generator_hash(),
    "seed": seed,
    "random_cases_per_op": 0 if specials_only else random_cases,
    "specials_only": specials_only,
    "status": "FAIL",
    "mismatches": [],
    "counts": {"vectors": len(cases), "compared": 0},
    "backends": {
      "oracle": "independent Fraction/integer exact arithmetic followed by one RNE",
      "softfloat_library_sha256": _sha(library),
      "hardfloat_probe_sha256": _sha(probe),
      "softfloat_abi": json.loads(abi_json.read_text()),
      "bf16_softfloat": "independent adapter using SoftFloat integer RNE; not a native BF16 API",
    },
  }
  sf_rows: list[dict] = []
  try:
    binding = SoftFloat(library, abi_json)
    for item in cases:
      values = [int(item[key], 16) for key in ("a", "b", "c")]
      sf_rows.append(_row(item, binding.compute(item["op"], *values)))
  except Exception as exc:
    index = len(sf_rows)
    failure = {
      "backend_error": "softfloat",
      "error": f"{type(exc).__name__}: {exc}",
      "input": cases[index] if index < len(cases) else None,
    }
    report["mismatches"] = [failure]
    _write(out / "first_failure.json", failure)
    _write(out / "softfloat.json", {"schema": 1, "vectors": sf_rows})
    _write(out / "report.json", report)
    return report
  _write(out / "softfloat.json", {"schema": 1, "vectors": sf_rows})

  measured_tsv = out / "hardfloat.tsv"
  execution = subprocess.run(
    [str(probe), str(tsv), str(measured_tsv)], capture_output=True, text=True, timeout=600
  )
  (out / "probe.stdout.log").write_text(execution.stdout, encoding="utf-8")
  (out / "probe.stderr.log").write_text(execution.stderr, encoding="utf-8")
  if execution.returncode:
    report["mismatches"] = [{"backend_error": "hardfloat", "exit_status": execution.returncode}]
    _write(out / "first_failure.json", report["mismatches"][0])
    _write(out / "report.json", report)
    return report
  try:
    hf_rows = _read_probe(measured_tsv, cases)
  except (OSError, ValueError) as exc:
    report["mismatches"] = [{"backend_error": "hardfloat-protocol", "error": str(exc)}]
    _write(out / "first_failure.json", report["mismatches"][0])
    _write(out / "report.json", report)
    return report
  _write(out / "hardfloat.json", {"schema": 1, "vectors": hf_rows})

  for index, (item, golden, sf, hf) in enumerate(zip(cases, expected, sf_rows, hf_rows, strict=True)):
    report["counts"]["compared"] = index + 1
    analytic = item.get("analytic_expected")
    if (
      golden != sf
      or golden != hf
      or (analytic is not None and any(golden[key] != analytic[key] for key in ("bits", "flags")))
    ):
      failure = {"input": item, "oracle": golden, "softfloat": sf, "hardfloat": hf}
      report["mismatches"] = [failure]
      _write(out / "first_failure.json", failure)
      break
  loaded_production = sorted(
    name for name in sys.modules if name == "pipeline_validator" or name.startswith("pipeline_validator.")
  )
  report["production_modules_loaded"] = loaded_production
  if loaded_production:
    report["mismatches"].append({"independence_error": loaded_production})
  report["counts"]["by_operation"] = dict(sorted(collections.Counter(item["op"] for item in cases).items()))
  report["counts"]["by_category"] = dict(
    sorted(collections.Counter(item["category"] for item in cases).items())
  )
  observations: dict[str, dict] = {}
  for op in ("fma", "add", "mul", "bf16"):
    words = [int(item["bits"], 16) for item in hf_rows if item["op"] == op]
    fraction_bits = 7 if op == "bf16" else 23
    magnitude_mask = (1 << (fraction_bits + 8)) - 1
    exponent_mask = 255 << fraction_bits
    fraction_mask = (1 << fraction_bits) - 1
    nan_words = sorted(
      {word for word in words if word & exponent_mask == exponent_mask and word & fraction_mask}
    )
    observations[op] = {
      "observed_nan_words": [f"0x{word:0{4 if op == 'bf16' else 8}x}" for word in nan_words],
      "positive_zero_count": words.count(0),
      "negative_zero_count": words.count(1 << (fraction_bits + 8)),
      "subnormal_result_count": sum(
        word & exponent_mask == 0 and word & fraction_mask != 0 for word in words
      ),
      "infinite_result_count": sum(word & magnitude_mask == exponent_mask for word in words),
    }
  report["observations"] = observations
  report["status"] = "PASS" if not report["mismatches"] else "FAIL"
  _write(out / "report.json", report)
  return report
