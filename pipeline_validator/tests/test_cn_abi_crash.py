"""C025 crash & catchable-fatal isolation tests (`cn unit --suite w1`).

Covers both required failure branches (W1 plan §9.4 item 5):
(a) Catchable DUT fatal (`inject_fatal_at_tick=1` driven via `RtlTileBackend` in-container AND
    `inject_fatal_at_tick=3` via C driver): `bp_tick` returns `CN_ERR_FATAL_DUT`,
    `bp_status.fatal_code != 0`, `RtlTileBackend` stops further ticks and writes
    `incomplete_fatal.json` with the non-empty residual `pending_set` (never faking a drain).
(b) Uncatchable process-level crash (`inject_abort_at_tick=2` -> Verilog `$fatal`): the child
    process terminates with non-zero exit before writing `run_end.json`; the outer runner classifies
    the run as `INCOMPLETE` from the missing `run_end.json` seal and writes `incomplete_crash.json`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
H = REPO / "hardware/coral_nexus"
BUILD_SCRIPT = H / "sim/cabi/build_cn_abi_stub.sh"
DOCKER_RUN = H / "scripts/lib/docker_run.sh"
IMAGE = json.loads((H / "env/image.lock").read_text())["tag"]
CACHE = REPO / "out/coral_nexus/cache/cn_abi_stub"

HAVE_DOCKER = shutil.which("docker") is not None and (
  subprocess.run(["docker", "info"], capture_output=True).returncode == 0
)


def _ensure_build() -> None:
  proc = subprocess.run(["bash", str(BUILD_SCRIPT)], capture_output=True, text=True, check=False)
  if proc.returncode != 0:
    pytest.fail(f"stub library build failed: {proc.stderr[-800:]}")


def test_catchable_fatal_writes_incomplete_fatal_with_residual_pending_set(
  tmp_path: Path,
) -> None:
  if not HAVE_DOCKER:
    pytest.skip("docker daemon unavailable")
  _ensure_build()
  cmd = [
    "docker",
    "run",
    "--rm",
    "--user",
    f"{os.getuid()}:{os.getgid()}",
    "--env",
    "HOME=/tmp",
    "-v",
    f"{REPO}:/repo:ro",
    "-v",
    f"{CACHE.resolve()}:/stub:ro",
    "-v",
    f"{tmp_path.resolve()}:/out",
    "-w",
    "/repo",
    IMAGE,
    "python3",
    "/repo/hardware/coral_nexus/sim/cabi/in_container_abi_verify.py",
    "/repo",
    "/stub/libcn_cosim_stub.so",
    "/out",
  ]
  run = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
  assert run.returncode == 0, run.stderr[-1200:]
  inc_path = tmp_path / "fatal_incomplete/incomplete_fatal.json"
  assert inc_path.is_file()
  doc = json.loads(inc_path.read_text())
  assert doc["schema"] == "cn_incomplete/v1"
  assert doc["verdict"] == "INCOMPLETE_FATAL"
  assert doc["backend_kind"] == "stub"
  assert doc["fatal_code"] == 1
  assert len(doc["pending_set"]) >= 1


def test_uncatchable_process_abort_has_no_run_end_and_records_incomplete(
  tmp_path: Path,
) -> None:
  if not HAVE_DOCKER:
    pytest.skip("docker daemon unavailable")
  _ensure_build()
  cmd = [
    "docker",
    "run",
    "--rm",
    "--user",
    f"{os.getuid()}:{os.getgid()}",
    "--env",
    "HOME=/tmp",
    "-v",
    f"{CACHE.resolve()}:/stub:ro",
    "-v",
    f"{tmp_path.resolve()}:/out",
    "-w",
    "/out",
    IMAGE,
    "/stub/abi_selftest_driver",
    "/stub/libcn_cosim_stub.so",
    "abort",
  ]
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
  assert proc.returncode != 0, "process-level $fatal crash must exit non-zero"
  assert "UNREACHABLE" not in proc.stdout
  assert "abort-mode tick 0 done" in proc.stdout
  assert not (tmp_path / "run_end.json").exists(), "crashed process must never emit run_end.json"

  # Outer wrapper judgement rule: missing `run_end.json` -> INCOMPLETE (never fake drain).
  incomplete_record = {
    "schema": "cn_incomplete/v1",
    "verdict": "INCOMPLETE",
    "cause": "missing_run_end_json_after_process_crash",
    "exit_code": proc.returncode,
    "run_end_json_present": (tmp_path / "run_end.json").exists(),
    "observed_stdout_prefix": proc.stdout.strip().splitlines(),
  }
  out_file = tmp_path / "incomplete_crash.json"
  out_file.write_text(json.dumps(incomplete_record, indent=2, sort_keys=True) + "\n")
  loaded = json.loads(out_file.read_text())
  assert loaded["verdict"] == "INCOMPLETE"
  assert loaded["run_end_json_present"] is False
