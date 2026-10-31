"""Shared ``--case/--out/--seed`` CLI plumbing for scenario test modules.

Several container targets (P09 ``shared-am-functional`` / ``two-slot-progress``
/ ``shared-am-cycle``) consume modules whose real gate is the pytest scenario
set itself.  This helper turns a module's own regression functions into the
report-JSON CLI shape used by ``test_boa_numeric_profile`` / ``test_soft_conv``
without duplicating any scenario: each case is a list of
``(check_id, callable)`` pairs where the callable is the module's permanent
test function, so the container gate can never drift from pytest.

The helper only builds reports/CLI plumbing; no scenario logic lives here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

SCHEMA = 1

Check = tuple[str, Callable[[], object]]


def build_parser(prog: str, description: str, cases: list[str], *, config: bool = False) -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog=prog, description=description)
  parser.add_argument("--case", required=True, choices=cases)
  parser.add_argument("--out", required=True, help="output directory for the case report (and artifacts)")
  parser.add_argument("--seed", type=int, default=1, help="recorded run seed; feeds SimConfig where one is built")
  if config:
    parser.add_argument("--config", default=None, help="optional schema-1 config YAML consumed by this module")
  return parser


def _detail(exc: BaseException) -> str:
  text = str(exc) or type(exc).__name__
  return text if len(text) <= 2000 else text[:2000] + f"... ({len(text)} chars)"


def run_checks(checks: list[Check]) -> list[dict]:
  """Execute the real scenario callables; a raised error is a failed check."""
  results: list[dict] = []
  for check_id, fn in checks:
    try:
      fn()
    except AssertionError as exc:
      results.append({"id": check_id, "pass": False, "detail": _detail(exc)})
    except Exception as exc:  # scenario crash is a failure, never a skip
      results.append({"id": check_id, "pass": False, "detail": f"{type(exc).__name__}: {_detail(exc)}"})
    else:
      results.append({"id": check_id, "pass": True, "detail": ""})
  return results


def build_report(
  case: str,
  module: str,
  seed: int,
  checks: list[dict],
  *,
  config: dict | None = None,
  metrics: dict | None = None,
) -> dict:
  failed = sum(1 for item in checks if not item["pass"])
  report: dict = {
    "schema": SCHEMA,
    "case": case,
    "module": module,
    "checks": checks,
    "counts": {"checks": len(checks), "failed": failed},
    "metrics": metrics or {},
    "status": "PASS" if checks and not failed else "FAIL",
  }
  if config is not None:
    report["config"] = config
  return report


def write_report(out: Path, report: dict) -> Path:
  out.mkdir(parents=True, exist_ok=True)
  path = out / f"{report['case']}.report.json"
  path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return path


def finish(report: dict) -> int:
  print(
    json.dumps(
      {
        "case": report["case"],
        "status": report["status"],
        "counts": report["counts"],
        "failed_checks": [item["id"] for item in report["checks"] if not item["pass"]],
      },
      sort_keys=True,
    )
  )
  return 0 if report["status"] == "PASS" else 1


def load_config_or_exit(loader: Callable[[str | None], T], path: str | None) -> T:
  """Parse ``--config`` with the module's real parser; exit 2 on misuse."""
  try:
    return loader(path)
  except (ValueError, TypeError, OSError) as exc:
    print(f"config-error: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc
