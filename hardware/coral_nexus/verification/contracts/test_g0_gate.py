"""Tests for the C007 hardened G0 gate.

The gate must RE-HASH report files and compare against the manifest, check
required artifacts, and refuse forged/stale/missing evidence. These tests
construct fake evidence directories to exercise each failure mode.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
GATES_PY = H / "verification/gates.py"
GATE_SH = H / "scripts/handlers/gate.sh"


def _run_gate(evidence_index: Path, out: Path) -> subprocess.CompletedProcess:
  return subprocess.run(
    [
      sys.executable,
      str(GATES_PY),
      "--evidence-index",
      str(evidence_index),
      "--out",
      str(out),
    ],
    capture_output=True,
    text=True,
  )


def _build_manifest(evidence_index: Path) -> None:
  subprocess.run(
    [sys.executable, str(GATES_PY), "--build-manifest", "--evidence-index", str(evidence_index)],
    capture_output=True,
    text=True,
    check=True,
  )


def _make_evidence_dir(
  base: Path,
  name: str,
  report_name: str,
  report_content: dict,
  artifacts: dict[str, str] | None = None,
) -> Path:
  d = base / name
  d.mkdir(parents=True, exist_ok=True)
  (d / report_name).write_text(json.dumps(report_content, indent=2) + "\n")
  for fname, content in (artifacts or {}).items():
    if fname == report_name:
      continue  # report already written above with real content
    (d / fname).write_text(content)
  return d


def _make_full_evidence_index(base: Path) -> Path:
  """Create a complete, valid evidence index with all 5 required entries."""
  _make_evidence_dir(
    base,
    "sources_verify",
    "sources_report.json",
    {"pass": True, "checks": []},
    {"sources_report.json": ""},
  )
  _make_evidence_dir(
    base,
    "environment_verify",
    "environment_checks.json",
    {"pass": True, "passed": 12, "failed": 0},
    {"environment_checks.json": "", "verify_inside.log": "PASS lines\n"},
  )
  _make_evidence_dir(
    base,
    "seed_export",
    "export_manifest.json",
    {"complete": True, "seeds": {}},
    {"export_manifest.json": ""},
  )
  _make_evidence_dir(
    base,
    "baseline_run",
    "report.json",
    {"pass": True, "workloads": []},
    {"report.json": "", "summary.txt": "PASS hello\nPASS p0\n"},
  )
  _make_evidence_dir(
    base,
    "bf16_baseline",
    "bf16_baseline_report.json",
    {"test_result": "PASSED", "bf16_cases": ["vtfmm_bf16_mt0"]},
    {"bf16_baseline_report.json": ""},
  )
  _build_manifest(base)
  return base


def test_gate_passes_with_complete_real_evidence(tmp_path: Path) -> None:
  idx = _make_full_evidence_index(tmp_path / "evidence")
  out = tmp_path / "gate_out"
  proc = _run_gate(idx, out)
  assert proc.returncode == 0
  report = json.loads((out / "g0_gate_report.json").read_text())
  assert report["pass"] is True
  assert len(report["checks"]) == 20  # 4 per evidence * 5


def test_gate_fails_on_forged_pass_boolean(tmp_path: Path) -> None:
  """A directory with {"pass": true} but no real artifacts must fail."""
  idx = tmp_path / "evidence"
  for name in ["sources_verify", "environment_verify", "seed_export", "baseline_run", "bf16_baseline"]:
    d = idx / name
    d.mkdir(parents=True)
    if name == "bf16_baseline":
      (d / "bf16_baseline_report.json").write_text(
        json.dumps({"test_result": "PASSED"}) + "\n"
      )
    elif name == "seed_export":
      (d / "export_manifest.json").write_text(
        json.dumps({"complete": True}) + "\n"
      )
    else:
      (d / "sources_report.json" if name == "sources_verify"
       else d / "environment_checks.json" if name == "environment_verify"
       else d / "report.json").write_text(
        json.dumps({"pass": True}) + "\n"
      )
  _build_manifest(idx)
  proc = _run_gate(idx, tmp_path / "out")
  assert proc.returncode != 0
  report = json.loads((tmp_path / "out/g0_gate_report.json").read_text())
  # Missing required artifacts (no summary.txt, verify_inside.log, etc.)
  failed = [c for c in report["checks"] if not c["pass"]]
  assert any("artifacts" in c["check"] for c in failed)


def test_gate_fails_on_stale_hash(tmp_path: Path) -> None:
  """Tampering with a report after manifest creation must fail."""
  idx = _make_full_evidence_index(tmp_path / "evidence")
  # Tamper: modify a report file AFTER the manifest was built.
  report = idx / "sources_verify/sources_report.json"
  report.write_text(json.dumps({"pass": True, "tampered": True}, indent=2) + "\n")
  proc = _run_gate(idx, tmp_path / "out")
  assert proc.returncode != 0
  report_json = json.loads((tmp_path / "out/g0_gate_report.json").read_text())
  failed = [c for c in report_json["checks"] if not c["pass"]]
  assert any("report_sha256" in c["check"] for c in failed)


def test_gate_fails_on_missing_bf16_evidence(tmp_path: Path) -> None:
  """Missing BF16 baseline directory must fail the gate."""
  idx = _make_full_evidence_index(tmp_path / "evidence")
  shutil.rmtree(idx / "bf16_baseline")
  _build_manifest(idx)
  proc = _run_gate(idx, tmp_path / "out")
  assert proc.returncode != 0
  report = json.loads((tmp_path / "out/g0_gate_report.json").read_text())
  assert any(
    c["check"] == "bf16_baseline.present" and not c["pass"]
    for c in report["checks"]
  )


def test_gate_fails_on_missing_seed_export(tmp_path: Path) -> None:
  idx = _make_full_evidence_index(tmp_path / "evidence")
  shutil.rmtree(idx / "seed_export")
  _build_manifest(idx)
  proc = _run_gate(idx, tmp_path / "out")
  assert proc.returncode != 0
  report = json.loads((tmp_path / "out/g0_gate_report.json").read_text())
  assert any(
    c["check"] == "seed_export.present" and not c["pass"]
    for c in report["checks"]
  )


def test_gate_fails_on_bf16_failed_verdict(tmp_path: Path) -> None:
  """BF16 report with test_result != PASSED must fail."""
  idx = _make_full_evidence_index(tmp_path / "evidence")
  (idx / "bf16_baseline/bf16_baseline_report.json").write_text(
    json.dumps({"test_result": "FAILED", "bf16_cases": []}) + "\n"
  )
  _build_manifest(idx)
  proc = _run_gate(idx, tmp_path / "out")
  assert proc.returncode != 0
  report = json.loads((tmp_path / "out/g0_gate_report.json").read_text())
  assert any(
    c["check"] == "bf16_baseline.verdict" and not c["pass"]
    for c in report["checks"]
  )
