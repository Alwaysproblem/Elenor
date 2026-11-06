"""Validate target constraint configs against the C008 schema (stdlib only)."""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs" / "target.schema.json"


def validate(config_path: Path) -> dict:
  config = json.loads(config_path.read_text())
  schema = json.loads(SCHEMA_PATH.read_text())
  errors = []
  for field in schema.get("required", []):
    if field not in config:
      errors.append(f"missing required field: {field}")
  if config.get("schema") != schema.get("properties", {}).get("schema", {}).get("const"):
    errors.append(f"schema mismatch: {config.get('schema')!r}")
  kind = config.get("kind")
  if kind not in {"p0_upstream", "cs1_target"}:
    errors.append(f"invalid kind: {kind!r}")
  constraints = config.get("constraints", {})
  for key, val in constraints.items():
    if isinstance(val, int) and val < 0:
      errors.append(f"constraint {key} must be >= 0, got {val}")
  report = {
    "schema": "cn_constraint_report/v1",
    "config": str(config_path),
    "valid": not errors,
    "errors": errors,
  }
  return report


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  report = validate(Path(args.config).resolve())
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
