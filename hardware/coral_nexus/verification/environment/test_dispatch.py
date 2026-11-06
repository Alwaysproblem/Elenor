"""Behavioral tests for the C003 unified cn entry.

Drives scripts/cn.sh through a real subprocess for every case: argument
passing with spaces, unknown command/suite refusals, run-manifest recording,
and the actual environment suite execution (sources + container lock tests).
No fake success paths: the refusals must exit non-zero and must not print the
command as supported.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CN = REPO_ROOT / "hardware/coral_nexus/scripts/cn.sh"
RUNS = REPO_ROOT / "out/coral_nexus/cn-runs"


def _cn(*args: str) -> subprocess.CompletedProcess:
  return subprocess.run(
    ["bash", str(CN), *args], capture_output=True, text=True, cwd=REPO_ROOT
  )


def test_unknown_command_exits_three_and_lists_known() -> None:
  proc = _cn("frobnicate")
  assert proc.returncode == 3
  assert "unknown command" in proc.stderr
  assert "env" in proc.stderr and "sources" in proc.stderr and "unit" in proc.stderr


def test_missing_suite_value_is_usage_error() -> None:
  proc = _cn("unit", "--suite")
  assert proc.returncode == 2


def test_unknown_suite_exits_three() -> None:
  proc = _cn("unit", "--suite", "w9", "--out", "out/coral_nexus/tmp-w9")
  assert proc.returncode == 3
  assert "unknown suite" in proc.stderr
  assert "environment" in proc.stderr


def test_env_requires_registered_subcommand() -> None:
  proc = _cn("env", "teardown")
  assert proc.returncode == 2
  assert "build" in proc.stderr and "verify" in proc.stderr


def test_sources_verify_runs_and_records_manifest() -> None:
  out_dir = REPO_ROOT / "out/coral_nexus/C003/sources-verify"
  proc = _cn(
    "sources",
    "verify",
    "--lock",
    "hardware/coral_nexus/env/sources.lock",
    "--out",
    str(out_dir.relative_to(REPO_ROOT)),
  )
  assert proc.returncode == 0, proc.stderr
  report = json.loads((out_dir / "sources_report.json").read_text())
  assert report["pass"] is True
  manifests = sorted(RUNS.glob("sources-*.json"))
  assert manifests, "run manifest not written"
  manifest = json.loads(manifests[-1].read_text())
  assert manifest["command"] == "sources"
  assert manifest["exit_code"] == 0


def test_unit_environment_suite_actually_runs_tests() -> None:
  out_dir = REPO_ROOT / "out/coral_nexus/C003/unit-environment"
  proc = _cn(
    "unit",
    "--suite",
    "environment",
    "--out",
    str(out_dir.relative_to(REPO_ROOT)),
  )
  assert proc.returncode == 0, (out_dir / "pytest.log").read_text()[-2000:]
  summary = json.loads((out_dir / "summary.json").read_text())
  assert summary["suite"] == "environment"
  assert summary["passed"] >= 20  # 10 sources + 10 environment tests
  assert summary["failed"] == 0
  manifests = sorted(RUNS.glob("unit-*.json"))
  manifest = json.loads(manifests[-1].read_text())
  assert manifest["command"] == "unit"
  assert manifest["exit_code"] == 0


def test_arguments_with_spaces_reach_the_handler() -> None:
  # unit rejects an unknown positional with a usage error; the error message
  # must carry the spaced token verbatim (no shell word splitting).
  proc = _cn("unit", "--suite", "environment", "two words")
  assert proc.returncode == 2
  assert "two words" in proc.stderr
