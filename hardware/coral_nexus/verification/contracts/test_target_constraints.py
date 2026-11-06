"""Tests for C008 target constraint schema validation."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
VALIDATE = H / "scripts/lib/validate_constraints.py"
SCHEMA = H / "configs/target.schema.json"


def test_schema_has_required_fields() -> None:
  schema = json.loads(SCHEMA.read_text())
  assert schema["required"] == ["schema", "kind", "name"]
  assert "p0_upstream" in schema["properties"]["kind"]["enum"]
  assert "cs1_target" in schema["properties"]["kind"]["enum"]


def test_valid_p0_config_passes() -> None:
  with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(
      {
        "schema": "cn_target/v1",
        "kind": "p0_upstream",
        "name": "p0_coral",
        "constraints": {
          "context_count": 1,
          "tile_count": 1,
          "retirement_buffer_mode": "full",
        },
      },
      f,
    )
    f.flush()
    proc = subprocess.run(
      [sys.executable, str(VALIDATE), "--config", f.name, "--out", "/tmp/cn-ct1"],
      capture_output=True,
      text=True,
    )
  assert proc.returncode == 0
  report = json.loads(Path("/tmp/cn-ct1/constraint_report.json").read_text())
  assert report["valid"] is True


def test_missing_kind_fails() -> None:
  with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump({"schema": "cn_target/v1", "name": "bad"}, f)
    f.flush()
    proc = subprocess.run(
      [sys.executable, str(VALIDATE), "--config", f.name, "--out", "/tmp/cn-ct2"],
      capture_output=True,
      text=True,
    )
  assert proc.returncode != 0
  report = json.loads(Path("/tmp/cn-ct2/constraint_report.json").read_text())
  assert "kind" in " ".join(report["errors"])


def test_negative_constraint_fails() -> None:
  with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(
      {
        "schema": "cn_target/v1",
        "kind": "cs1_target",
        "name": "bad",
        "constraints": {"context_count": -1},
      },
      f,
    )
    f.flush()
    proc = subprocess.run(
      [sys.executable, str(VALIDATE), "--config", f.name, "--out", "/tmp/cn-ct3"],
      capture_output=True,
      text=True,
    )
  assert proc.returncode != 0
  report = json.loads(Path("/tmp/cn-ct3/constraint_report.json").read_text())
  assert any("context_count" in e for e in report["errors"])
