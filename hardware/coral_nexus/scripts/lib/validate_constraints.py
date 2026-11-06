"""Validate target constraint configs and sourced parameters (C008, stdlib only).

Checks implemented (each maps to a C008 rejection class):

- schema conformance: required fields, const/enum, integer types, and the
  schema's minimum/maximum bounds (context_count=0 or >8 is rejected here,
  not merely documented in the schema file).
- encodability: a parameter that declares width_bits must satisfy
  value < 2**width_bits (static encodability; runtime peaks are hearsay).
- unit normalization: capacity params carry a unit from {B, KiB}; the same
  parameter expressed in different units must agree AFTER normalization
  (2048 B vs 2 KiB agree; 2048 B vs 2048 KiB is a unit conflict).
- reserved region + static upper bound: user_spm_per_bank +
  system_reserved_spm_per_bank must stay within the sourced bank capacity
  (l1_spm_bytes_per_bank_mode0), with the reserved part itself >= 0.
- multi-source conflicts: one parameter name defined by two params.d sources
  with different normalized values is rejected outright (no silent override).
- C1/C2 degradation: when both a c1 and c2 target exist, context_count must
  not increase from C2 to C1 (C1 is the degraded regression), and every other
  constraint must be identical (same-RTL, only active-context-count differs).

`validate` takes the config path plus the params.d directory; `main` keeps a
single-config mode for the runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs" / "target.schema.json"
PARAMS_DIR = Path(__file__).resolve().parents[2] / "configs" / "params.d"

_UNIT_TO_BYTES = {"B": 1, "KiB": 1024}
_CAPACITY_UNITS = {"B", "KiB"}


def _normalize(value: object, unit: str) -> float:
  return float(cast(int, value)) * _UNIT_TO_BYTES[unit]


def _load_schema() -> dict:
  return json.loads(SCHEMA_PATH.read_text())


def _check_schema_bounds(config: dict, schema: dict, errors: list[str]) -> None:
  props = schema.get("properties", {}).get("constraints", {}).get("properties", {})
  constraints = config.get("constraints", {})
  for key, rule in props.items():
    if key not in constraints:
      continue
    val = constraints[key]
    if "type" in rule and rule["type"] == "integer":
      if not isinstance(val, int) or isinstance(val, bool):
        errors.append(f"constraint {key}: expected integer, got {type(val).__name__}")
        continue
      if "minimum" in rule and val < rule["minimum"]:
        errors.append(f"constraint {key}: {val} < minimum {rule['minimum']}")
      if "maximum" in rule and val > rule["maximum"]:
        errors.append(f"constraint {key}: {val} > maximum {rule['maximum']}")
    if "enum" in rule and val not in rule["enum"]:
      errors.append(f"constraint {key}: {val!r} not in {rule['enum']}")


def load_params(params_dir: Path = PARAMS_DIR) -> tuple[dict[str, dict], list[str]]:
  """Merge params.d sources -> {name: definition}; conflicts become errors.

  Returns (merged, errors). A name defined twice with the same normalized
  value but different provenance is legal (cross-source agreement); a
  different value is a multi-source conflict.
  """
  merged: dict[str, dict] = {}
  errors: list[str] = []
  for path in sorted(params_dir.glob("*.json")):
    doc = json.loads(path.read_text())
    for param in doc.get("params", []):
      name = param["name"]
      unit = param.get("unit", "count")
      if "value" in param and unit in _CAPACITY_UNITS:
        norm = _normalize(param["value"], unit)
      else:
        norm = param.get("value")
      if name not in merged:
        merged[name] = {
          "value": norm,
          "raw": param.get("value"),
          "unit": unit,
          "width_bits": param.get("width_bits"),
          "sources": [doc["source"]["kind"]],
          "provenance": [param.get("provenance", {})],
        }
        continue
      existing = merged[name]
      if existing["value"] != norm:
        errors.append(
          f"param {name}: multi-source conflict "
          f"{existing['raw']!r} {existing['unit']} ({existing['sources'][0]}) "
          f"vs {param.get('value')!r} {unit} ({doc['source']['kind']})"
        )
        continue
      existing["sources"].append(doc["source"]["kind"])
      existing["provenance"].append(param.get("provenance", {}))
  return merged, errors


def _check_encodability(params: dict, errors: list[str]) -> None:
  for name, definition in params.items():
    width = definition.get("width_bits")
    value = definition.get("raw")
    if width is None or not isinstance(value, int):
      continue
    if value >= 2**width:
      errors.append(
        f"param {name}: {value} not encodable in {width} bits (max {2**width - 1})"
      )


def _check_reserved_and_bound(
  params: dict, config: dict, errors: list[str]
) -> None:
  bank = params.get("l1_spm_bytes_per_bank_mode0")
  reserved_src = params.get("l1_system_reserved_spm_per_bank")
  constraints = config.get("constraints", {})
  user = constraints.get("user_spm_per_bank")
  reserved_cfg = constraints.get("system_reserved_spm_per_bank")

  if reserved_src is not None and reserved_cfg is not None:
    if reserved_src["value"] != reserved_cfg:
      errors.append(
        f"system_reserved_spm_per_bank: config {reserved_cfg} conflicts with "
        f"sourced {reserved_src['raw']} {reserved_src['unit']}"
      )
  if bank is None:
    return
  total_used = (user or 0) + (reserved_cfg if reserved_cfg is not None
                              else (reserved_src["value"] if reserved_src else 0))
  if total_used > bank["value"]:
    errors.append(
      f"L1 SPM overflow: user {user or 0}B + reserved "
      f"{reserved_cfg if reserved_cfg is not None else (reserved_src['value'] if reserved_src else 0)}B "
      f"= {total_used}B > bank {int(bank['value'])}B"
    )


def _check_context_bound(params: dict, config: dict, errors: list[str]) -> None:
  ctx = config.get("constraints", {}).get("context_count")
  if not isinstance(ctx, int):
    return
  lo = params.get("context_count_min")
  hi = params.get("context_count_max")
  if lo is not None and ctx < lo["value"]:
    errors.append(f"context_count {ctx} < sourced minimum {int(lo['value'])}")
  if hi is not None and ctx > hi["value"]:
    errors.append(f"context_count {ctx} > sourced maximum {int(hi['value'])}")


def _check_c1_c2_direction(configs: dict[str, dict], errors: list[str]) -> None:
  """C1 is the degraded regression: fewer or equal contexts, all else equal."""
  c1 = configs.get("c1")
  c2 = configs.get("c2")
  if c1 is None or c2 is None:
    return
  n1 = c1.get("constraints", {}).get("context_count")
  n2 = c2.get("constraints", {}).get("context_count")
  if isinstance(n1, int) and isinstance(n2, int) and n1 > n2:
    errors.append(
      f"C1/C2 degradation violated: c1 context_count {n1} > c2 {n2}"
    )
  rest1 = {k: v for k, v in c1.get("constraints", {}).items() if k != "context_count"}
  rest2 = {k: v for k, v in c2.get("constraints", {}).items() if k != "context_count"}
  if rest1 != rest2:
    errors.append(
      "C1/C2 must share every constraint except context_count "
      f"(same RTL hash; got c1={rest1} c2={rest2})"
    )


def validate(
  config_path: Path, params_dir: Path = PARAMS_DIR, extra_targets: dict | None = None
) -> dict:
  config = json.loads(config_path.read_text())
  schema = _load_schema()
  errors: list[str] = []

  for field in schema.get("required", []):
    if field not in config:
      errors.append(f"missing required field: {field}")
  if config.get("schema") != schema.get("properties", {}).get("schema", {}).get("const"):
    errors.append(f"schema mismatch: {config.get('schema')!r}")
  kind = config.get("kind")
  if kind not in {"p0_upstream", "cs1_target"}:
    errors.append(f"invalid kind: {kind!r}")

  _check_schema_bounds(config, schema, errors)

  params, param_errors = load_params(params_dir)
  errors.extend(param_errors)
  _check_encodability(params, errors)
  _check_reserved_and_bound(params, config, errors)
  _check_context_bound(params, config, errors)

  targets = {"main": config}
  if extra_targets:
    targets.update(extra_targets)
  _check_c1_c2_direction(targets, errors)

  return {
    "schema": "cn_constraint_report/v1",
    "config": str(config_path),
    "params_merged": len(params),
    "valid": not errors,
    "errors": errors,
  }


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--params-dir", default=str(PARAMS_DIR))
  args = parser.parse_args()
  report = validate(Path(args.config).resolve(), Path(args.params_dir).resolve())
  out = Path(args.out).resolve()
  out.mkdir(parents=True, exist_ok=True)
  (out / "constraint_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(report, indent=2, sort_keys=True))
  if not report["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
