"""Tests for the C007/C008 hardened G0 gate (pin-based root of trust).

The gate verifies each required report and artifact against a COMMITTED pin
(verification/expected_evidence.json). Re-hashing against a manifest rebuilt
from the same directory proves nothing, so these tests pin a pristine fixture
first and then tamper copies: every tamper must fail, and a missing pin must
fail closed.
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
REPO_PIN = H / "verification/expected_evidence.json"

ELF_HASH = "a9f9d0b4124cf4beab7037f889850a76f44566482ed0f6f61d992909c22336ea"

SOURCES = ("sources_verify", "sources_report.json", {"pass": True, "checks": []})
ENVIRONMENT = (
  "environment_verify",
  "environment_checks.json",
  {"pass": True, "passed": 12, "failed": 0},
)
SEED_EXPORT = ("seed_export", "export_manifest.json", {"complete": True, "seeds": {}})
BASELINE = (
  "baseline_run",
  "report.json",
  {"pass": True, "workloads": [{"name": "p0_scalar_add", "elf_sha256": ELF_HASH}]},
)
BF16 = (
  "bf16_baseline",
  "bf16_baseline_report.json",
  {"test_result": "PASSED", "bf16_cases": ["vtfmm_bf16_mt0", "vtfmm_bf16_mt8"]},
)
OFFLINE = (
  "offline_rebuild",
  "offline_rebuild.json",
  {"pass": True, "network": "none", "elfs": {"p0_scalar_add.elf": ELF_HASH}},
)
DEVCONTAINER = (
  "devcontainer_probe",
  "devcontainer_report.json",
  {"pass": True, "status": "passed", "checks": {"up": True}},
)
ARTIFACTS = {
  "sources_verify": {},
  "environment_verify": {"verify_inside.log": "PASS env\n"},
  "seed_export": {},
  "baseline_run": {"summary.txt": "PASS hello\nPASS p0\n"},
  "bf16_baseline": {},
  "offline_rebuild": {"build.log": "Build completed successfully\n"},
  "devcontainer_probe": {"probes.log": "builder\nbazel 9.1.0\n/workspace\n"},
}


def _run_gate(
  evidence_index: Path, out: Path, pin: Path | None = None
) -> subprocess.CompletedProcess:
  cmd = [sys.executable, str(GATES_PY), "--evidence-index", str(evidence_index)]
  cmd += ["--out", str(out)]
  if pin is not None:
    cmd += ["--pin", str(pin)]
  return subprocess.run(cmd, capture_output=True, text=True)


def _write_pin(evidence_index: Path, pin: Path) -> None:
  subprocess.run(
    [
      sys.executable,
      str(GATES_PY),
      "--evidence-index",
      str(evidence_index),
      "--pin",
      str(pin),
      "--write-pin",
    ],
    capture_output=True,
    text=True,
    check=True,
  )


def _make_evidence_index(base: Path) -> Path:
  groups = (
    SOURCES,
    ENVIRONMENT,
    SEED_EXPORT,
    BASELINE,
    BF16,
    OFFLINE,
    DEVCONTAINER,
  )
  for name, report_name, content in groups:
    d = base / name
    d.mkdir(parents=True)
    (d / report_name).write_text(json.dumps(content, indent=2) + "\n")
    for artifact, text in ARTIFACTS[name].items():
      (d / artifact).write_text(text)
  return base


def _pinned_fixture(tmp_path: Path) -> tuple[Path, Path]:
  """Pristine fixture plus a pin generated from it."""
  idx = _make_evidence_index(tmp_path / "evidence")
  pin = tmp_path / "pin.json"
  _write_pin(idx, pin)
  return idx, pin


def _failed_checks(out: Path) -> list[dict]:
  report = json.loads((out / "g0_gate_report.json").read_text())
  return [c for c in report["checks"] if not c["pass"]]


def test_gate_passes_when_evidence_matches_pin(tmp_path: Path) -> None:
  idx, pin = _pinned_fixture(tmp_path)
  proc = _run_gate(idx, tmp_path / "out", pin)
  assert proc.returncode == 0, proc.stdout + proc.stderr
  report = json.loads((tmp_path / "out/g0_gate_report.json").read_text())
  assert report["pass"] is True
  # pin.present, pin.coverage, 4 per evidence * 7, plus the cross-check
  assert len(report["checks"]) == 31


def test_gate_fails_closed_without_pin(tmp_path: Path) -> None:
  """No pin file means no root of trust: the gate must not pass."""
  idx, _ = _pinned_fixture(tmp_path)
  proc = _run_gate(idx, tmp_path / "out", tmp_path / "absent.json")
  assert proc.returncode != 0
  assert any(c["check"] == "pin.present" for c in _failed_checks(tmp_path / "out"))


def test_gate_fails_on_forged_pass_boolean(tmp_path: Path) -> None:
  """A directory holding only {"pass": true} must fail against a real pin."""
  idx = tmp_path / "forged"
  groups = (SOURCES, ENVIRONMENT, SEED_EXPORT, BASELINE, BF16, OFFLINE, DEVCONTAINER)
  for name, report_name, _ in groups:
    d = idx / name
    d.mkdir(parents=True)
    (d / report_name).write_text(json.dumps({"pass": True}) + "\n")
  for name, artifacts in ARTIFACTS.items():
    for artifact, _text in artifacts.items():
      (idx / name / artifact).write_text("")
  proc = _run_gate(idx, tmp_path / "out", REPO_PIN)
  assert proc.returncode != 0
  failed = _failed_checks(tmp_path / "out")
  assert any("report_sha256" in c["check"] for c in failed)
  assert any("artifacts" in c["check"] for c in failed)


def test_gate_fails_on_tampered_report(tmp_path: Path) -> None:
  idx, pin = _pinned_fixture(tmp_path)
  report = idx / "sources_verify/sources_report.json"
  report.write_text(json.dumps({"pass": True, "tampered": True}) + "\n")
  proc = _run_gate(idx, tmp_path / "out", pin)
  assert proc.returncode != 0
  assert any(
    "report_sha256" in c["check"] for c in _failed_checks(tmp_path / "out")
  )


def test_gate_fails_on_tampered_artifact(tmp_path: Path) -> None:
  idx, pin = _pinned_fixture(tmp_path)
  (idx / "baseline_run/summary.txt").write_text("PASS hello\nFAIL p0\n")
  proc = _run_gate(idx, tmp_path / "out", pin)
  assert proc.returncode != 0
  failed = _failed_checks(tmp_path / "out")
  assert any(c["check"] == "baseline_run.artifacts" for c in failed)


def test_gate_fails_on_missing_bf16_evidence(tmp_path: Path) -> None:
  idx, pin = _pinned_fixture(tmp_path)
  shutil.rmtree(idx / "bf16_baseline")
  proc = _run_gate(idx, tmp_path / "out", pin)
  assert proc.returncode != 0
  assert any(
    c["check"] == "bf16_baseline.present"
    for c in _failed_checks(tmp_path / "out")
  )


def test_gate_fails_on_missing_seed_export(tmp_path: Path) -> None:
  idx, pin = _pinned_fixture(tmp_path)
  shutil.rmtree(idx / "seed_export")
  proc = _run_gate(idx, tmp_path / "out", pin)
  assert proc.returncode != 0
  assert any(
    c["check"] == "seed_export.present"
    for c in _failed_checks(tmp_path / "out")
  )


def test_gate_rejects_pin_of_failing_verdict(tmp_path: Path) -> None:
  """Re-pinning a FAILED BF16 verdict must still fail: pass is not opt-in."""
  idx = _make_evidence_index(tmp_path / "evidence")
  (idx / "bf16_baseline/bf16_baseline_report.json").write_text(
    json.dumps({"test_result": "FAILED", "bf16_cases": []}) + "\n"
  )
  pin = tmp_path / "pin.json"
  _write_pin(idx, pin)
  proc = _run_gate(idx, tmp_path / "out", pin)
  assert proc.returncode != 0
  assert any(
    c["check"] == "bf16_baseline.verdict"
    for c in _failed_checks(tmp_path / "out")
  )


def test_gate_fails_when_pin_misses_required_evidence(tmp_path: Path) -> None:
  idx, pin = _pinned_fixture(tmp_path)
  data = json.loads(pin.read_text())
  del data["entries"]["bf16_baseline"]
  pin.write_text(json.dumps(data, indent=2) + "\n")
  proc = _run_gate(idx, tmp_path / "out", pin)
  assert proc.returncode != 0
  failed = _failed_checks(tmp_path / "out")
  assert any(c["check"] == "pin.coverage" for c in failed)


def test_repo_pin_verifies_repo_evidence_index() -> None:
  """The committed pin must accept the real recorded evidence set."""
  index = REPO_ROOT / "out/coral_nexus/C007/evidence_index"
  if not index.is_dir():
    import pytest

    pytest.skip("recorded evidence index absent in this checkout")
  proc = _run_gate(index, REPO_ROOT / "out/coral_nexus/C007/gate-test", REPO_PIN)
  assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]


def test_gate_fails_when_offline_rebuild_differs_from_baseline(tmp_path: Path) -> None:
  """An offline rebuild with different bytes than the baseline is not a rebuild."""
  idx = _make_evidence_index(tmp_path / "evidence")
  offline = idx / "offline_rebuild/offline_rebuild.json"
  data = json.loads(offline.read_text())
  data["elfs"]["p0_scalar_add.elf"] = "de" * 32
  offline.write_text(json.dumps(data) + "\n")
  pin = tmp_path / "pin.json"
  _write_pin(idx, pin)
  proc = _run_gate(idx, tmp_path / "out", pin)
  assert proc.returncode != 0
  assert any(
    c["check"] == "offline_rebuild.matches_baseline"
    for c in _failed_checks(tmp_path / "out")
  )


def test_gate_fails_on_missing_devcontainer_evidence(tmp_path: Path) -> None:
  idx, pin = _pinned_fixture(tmp_path)
  shutil.rmtree(idx / "devcontainer_probe")
  proc = _run_gate(idx, tmp_path / "out", pin)
  assert proc.returncode != 0
  assert any(
    c["check"] == "devcontainer_probe.present"
    for c in _failed_checks(tmp_path / "out")
  )
