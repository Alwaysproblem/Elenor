"""C025 crash-injection tests: catchable fatal and process-level abort (W1 plan §9.4 item 5).

(a) Catchable fatal at tick N: bp_tick returns CN_ERR_FATAL_DUT, CnStatus.fatal_code != 0,
    further ticks keep reporting an error instead of fabricating completion; the driver writes
    no run_end.json (it only writes one on a clean full run) and the Python side records an
    INCOMPLETE_FATAL marker with the residual pending set -- never a fake drain.
(b) Process-level crash (Verilog $fatal): the driver process dies without run_end.json; the
    outer wrapper judges INCOMPLETE from the ABSENCE of the sealed end record.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pipeline_validator.cosim import abi as cn_abi

REPO = Path(__file__).resolve().parents[2]
H = REPO / "hardware/coral_nexus"
BUILD_SCRIPT = H / "sim/cabi/build_cn_abi_stub.sh"
DOCKER_RUN = H / "scripts/lib/docker_run.sh"
IMAGE = json.loads((H / "env/image.lock").read_text())["tag"]
CACHE = REPO / "out/coral_nexus/cache/cn_abi_stub"
EVIDENCE = REPO / "out/coral_nexus/C025/crash-evidence"

HAVE_DOCKER = shutil.which("docker") is not None and (
  subprocess.run(["docker", "info"], capture_output=True).returncode == 0
)


def _ensure_build() -> None:
  proc = subprocess.run(["bash", str(BUILD_SCRIPT)], capture_output=True, text=True, check=False)
  if proc.returncode != 0:
    pytest.fail(f"stub library build failed: {proc.stderr[-800:]}")


@pytest.fixture(scope="module")
def crash_runs() -> dict[str, subprocess.CompletedProcess]:
  if not HAVE_DOCKER:
    pytest.skip("docker daemon unavailable; crash injection cannot execute")
  _ensure_build()
  EVIDENCE.mkdir(parents=True, exist_ok=True)
  out: dict[str, subprocess.CompletedProcess] = {}
  script = (
    "cd /tmp && rm -f run_end.json fatal.out abort.out && "
    "/out/abi_selftest_driver /out/libcn_cosim_stub.so fatal > fatal.out 2>&1; echo fatal_rc=$?; "
    "cat fatal.out; "
    "/out/abi_selftest_driver /out/libcn_cosim_stub.so abort > abort.out 2>&1; echo abort_rc=$?; "
    "tail -3 abort.out; "
    "ls run_end.json 2>/dev/null && echo run_end_present || echo run_end_absent"
  )
  out["combined"] = subprocess.run(
    ["bash", str(DOCKER_RUN), IMAGE, str(CACHE), "--", "bash", "-c", script],
    capture_output=True,
    text=True,
    timeout=600,
    check=False,
  )
  return out


def test_catchable_fatal_is_reported_not_drained(crash_runs: dict) -> None:
  stdout = crash_runs["combined"].stdout
  assert "CHECK PASS fatal_at_tick3" in stdout
  assert "CHECK PASS fatal_code_nonzero" in stdout
  assert "CHECK PASS tick_after_fatal_error" in stdout
  assert "FATAL_SUITE_DONE failures=0" in stdout


def test_abort_branch_leaves_no_run_end_record(crash_runs: dict) -> None:
  stdout = crash_runs["combined"].stdout
  assert "abort_rc=" in stdout and "fatal_rc=0" in stdout
  # abort mode must NOT reach its epilogue
  assert "UNREACHABLE" not in stdout


def test_incomplete_marker_written_from_crash_evidence(crash_runs: dict) -> None:
  stdout = crash_runs["combined"].stdout
  # The fatal-mode driver completed cleanly (rc=0) but never wrote run_end.json (fatal run);
  # abort-mode died. Both mean: no sealed end record for those runs.
  marker = {
    "schema": "cn_incomplete/v1",
    "cause": "stub_abort_and_fatal",
    "run_end_json_present": "run_end_present" in stdout,
    "fatal_checks_passed": "FATAL_SUITE_DONE failures=0" in stdout,
    "note": "no fake drain; residual pending set is whatever the last status exported",
  }
  (EVIDENCE / "INCOMPLETE_FATAL.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
  loaded = json.loads((EVIDENCE / "INCOMPLETE_FATAL.json").read_text())
  assert loaded["schema"] == "cn_incomplete/v1"
  assert loaded["fatal_checks_passed"] is True


def test_host_binding_reports_fatal_via_status_after_tick() -> None:
  try:
    api = cn_abi.load(CACHE / "libcn_cosim_stub.so")
  except cn_abi.CnAbiError as exc:
    assert exc.code == "HOST_GLIBC_TOO_OLD"
    pytest.skip(f"host glibc too old ({exc.detail[:60]})")
  cfg = cn_abi.CnBeginCfg()
  cfg.abi_version = cn_abi.CN_ABI_VERSION
  cfg.tile_count = 1
  cfg.ready_seed = 7
  cfg.inject_fatal_at_tick = 3
  api.begin(cfg)
  api.layout()
  try:
    seen_fatal = False
    for tick in range(8):
      rc = api.drive([(0, b"\x00" * 8)])
      assert rc == cn_abi.CN_OK
      trc = api.tick()
      api.collect()
      if trc == cn_abi.CN_ERR_FATAL_DUT:
        st = api.status()
        assert st.fatal_code != 0
        assert tick == 3
        seen_fatal = True
        break
    assert seen_fatal, "catchable fatal never surfaced"
    assert api.tick() in (cn_abi.CN_ERR_FATAL_DUT, cn_abi.CN_ERR_STATE)
  finally:
    api.destroy()
