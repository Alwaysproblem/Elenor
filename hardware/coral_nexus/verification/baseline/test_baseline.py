"""Tests for the C005 P0 baseline handler.

Unit tests (no docker) verify config-kind rejection, workload-registry
parsing and handler invariants. Integration tests (gated by
.coral_docker_ok) run the real baseline through the locked container and
assert both P0 workloads pass with exit code 0.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
BASELINE_SH = H / "scripts/handlers/baseline.sh"
P0_CONFIG = "hardware/coral_nexus/configs/p0_coral.yaml"
P0_REGISTRY = H / "configs/p0_workloads.yaml"

CN = H / "scripts/cn.sh"


def _load_yaml(path: Path | str) -> dict:
  import sys

  sys.path.insert(0, str(H / "scripts/lib"))
  from sources_tool import load_yaml_limited

  return load_yaml_limited(Path(path))


def test_p0_config_is_p0_upstream_kind() -> None:
  cfg = _load_yaml(P0_CONFIG)
  assert cfg["kind"] == "p0_upstream"
  assert cfg["full_retirement_buffer"] == "true"


def test_p0_config_declares_real_bazel_targets() -> None:
  cfg = _load_yaml(P0_CONFIG)
  targets = cfg["bazel"]["targets"]
  assert "//tests/verilator_sim:core_mini_axi_sim" in targets
  assert "//examples:coralnpu_v2_hello_world_add_floats" in targets
  assert "//nexus_p0:p0_scalar_add" in targets


def test_p0_workload_registry_has_hand_computable_expectations() -> None:
  doc = _load_yaml(P0_REGISTRY)
  names = [w["name"] for w in doc["workloads"]]
  assert "hello_world_add_floats" in names
  assert "p0_scalar_add" in names
  for w in doc["workloads"]:
    assert w["expected_exit"] == "0"
    assert w["expected_outputs"], f"{w['name']} missing expected_outputs"
  scalar = next(w for w in doc["workloads"] if w["name"] == "p0_scalar_add")
  assert "1.5" in scalar["expected_outputs"] and "10" in scalar["expected_outputs"]


def test_baseline_rejects_cs1_config(tmp_path: Path) -> None:
  cfg = tmp_path / "cs1.yaml"
  cfg.write_text("kind: cs1_target\n")
  proc = subprocess.run(
    ["bash", str(BASELINE_SH), "--config", str(cfg), "--out", str(tmp_path / "out")],
    capture_output=True,
    text=True,
    cwd=REPO_ROOT,
  )
  assert proc.returncode == 3
  assert "p0_upstream" in proc.stderr


def test_cn_baseline_dispatches_to_handler() -> None:
  proc = subprocess.run(
    ["bash", str(CN), "baseline", "--config", "nonexistent.yaml"],
    capture_output=True,
    text=True,
    cwd=REPO_ROOT,
  )
  # The handler runs and fails on the missing config, not on dispatch.
  assert proc.returncode != 0


@pytest.mark.skipif(
  subprocess.run(["docker", "info"], capture_output=True).returncode != 0
  or not (REPO_ROOT / ".coral_docker_ok").exists(),
  reason="docker daemon not reachable or .coral_docker_ok flag absent",
)
class TestBaselineRoundTrip:
  """Real P0 baseline through the locked container."""

  def test_both_workloads_pass(self, tmp_path: Path) -> None:
    proc = subprocess.run(
      [
        "bash",
        str(BASELINE_SH),
        "--config",
        P0_CONFIG,
        "--out",
        str(tmp_path / "baseline"),
      ],
      capture_output=True,
      text=True,
      cwd=REPO_ROOT,
      timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    report = json.loads((tmp_path / "baseline/report.json").read_text())
    assert report["pass"] is True
    assert len(report["workloads"]) == 2
    for w in report["workloads"]:
      assert w["pass"] is True
      assert w["exit_code"] == "0"
      assert w["elf_sha256"] is not None
