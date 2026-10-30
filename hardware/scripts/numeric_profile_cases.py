"""P06 numeric-profile case adapters for the existing container runner.

Drives the real production numeric gate
(``pipeline_validator.tests.test_boa_numeric_profile``: parse -> compile ->
load -> Simulator full_memory ByteStore against committed P03/P06 golden
bytes) inside the locked container.  No numerical algorithms live here.

The committed golden vectors pin each case's own ``boa_numeric_tile_k``
(2/4/8/16), so this target deliberately consumes no ``validator``
configuration: ``validator.hw_overrides`` must be empty and is rejected
otherwise (fail closed, never silently discarded).  ``--seed`` is recorded
and feeds the simulator; the goldens are fixed and never change with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numeric_cases


def _module_config(ctx) -> Path:
  """Strict module config: profile only; paths default inside the module."""
  payload = {"schema": 1, "profile": numeric_cases.PROFILE}
  path = ctx.work / "numeric-profile-config.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return path


def _run_profile_case(ctx, case: str) -> bool:
  overrides = ctx.config.get("validator", {}).get("hw_overrides", {})
  if not ctx.check(
    "fixed-vector-mode-no-overrides",
    not overrides,
    f"golden vectors pin per-case boa_numeric_tile_k; validator.hw_overrides "
    f"must be empty, got {overrides}",
  ):
    return False

  out = ctx.artifacts_dir()
  argv = [
    sys.executable,
    "-m",
    "pipeline_validator.tests.test_boa_numeric_profile",
    "--case",
    case,
    "--out",
    str(out),
    "--seed",
    str(ctx.request["seed"]),
    "--config",
    str(_module_config(ctx)),
  ]
  code = ctx.run(
    argv,
    ctx.repo,
    f"numeric-profile-{case}",
    env={"PYTHONPATH": str(ctx.repo)},
    timeout=3600,
  )
  ok = ctx.check(f"{case}-exit", code == 0, f"exit={code}; see command log")

  report = None
  report_path = out / f"{case}.report.json"
  if report_path.is_file():
    try:
      report = json.loads(report_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
      ok &= ctx.check(f"{case}-report-parses", False, f"invalid JSON: {exc}")
  else:
    ok &= ctx.check(f"{case}-report-written", False, f"{report_path} missing")

  if report is not None:
    ok &= ctx.check(
      f"{case}-status",
      report.get("status") == "PASS",
      f"status={report.get('status')!r} counts={report.get('counts')}",
    )
    for check in report.get("checks", []):
      ok &= ctx.check(f"{case}:{check['id']}", check["pass"], check.get("detail", ""))
    counts = report.get("counts", {})
    ctx.metrics[f"{case}_vectors"] = counts.get("vectors")
    ctx.metrics[f"{case}_cycles_total"] = counts.get("cycles_total")

  # Wall-clock timing is deliberately recorded with deterministic=False so
  # offline reproduction compares outputs, never elapsed time.
  for name, deterministic in (
    (f"{case}.report.json", True),
    (f"{case}.vectors.json", True),
    (f"{case}.timing.json", False),
  ):
    candidate = out / name
    if candidate.is_file():
      ctx.record_artifact(candidate, deterministic=deterministic)
  sources = out / "sources"
  if sources.is_dir():
    for source in sorted(sources.glob("*.mlir")):
      ctx.record_artifact(source, deterministic=True)
  return bool(ok)


def case_vector_parity(ctx) -> bool:
  """Committed P03 matmul vectors (specials/tails/masks/chains/illegal) plus
  the directed tile_k 8-vs-16 counterexamples, through the production path."""
  return _run_profile_case(ctx, "vector-parity")


def case_zero_k(ctx) -> bool:
  """N-P03-11 zero-K / zero-M / zero-N group through the production path."""
  return _run_profile_case(ctx, "zero-k")


def case_cutover_reject(ctx) -> bool:
  """Legacy v4 version-gate rejection, current-format replay, and the
  independent wrong-target fingerprint rejection."""
  return _run_profile_case(ctx, "cutover-reject")
