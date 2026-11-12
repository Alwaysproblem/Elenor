"""C025 stub-ABI five-phase tests.

Because the stub library is built inside `coral-nexus-builder:c002` (Debian trixie glibc),
the authoritative execution runs the in-container selftest driver via docker; the host-side
tests parse its structured CHECK output and ALSO assert the negative behaviors through the
driver's exit code. When docker or the built library is unavailable the suite skips with the
reason recorded (never silently passes).

Covered (card C025 / W1 plan §9.4):
- bp_begin -> bp_layout -> (bp_drive -> bp_tick -> bp_collect) x N happy path;
- posedge_count advances by exactly +1 per tick;
- ready lowering (fixed seed): payload stable while valid && !ready, accepted set committed
  exactly once, credit conservation (accepted == returned after drain);
- negatives: missing bp_drive -> CN_ERR_STATE; double collect -> CN_ERR_STATE;
  collect buffer too small -> CN_ERR_RANGE (no truncation);
- bp_abi_info hash equals sha256(cn_protocol_v1.yaml) ++ sha256(cn_layout.json);
- layout queried more than once -> CN_ERR_STATE; short slab array -> CN_ERR_RANGE.
"""

from __future__ import annotations

import hashlib
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
SO = CACHE / "libcn_cosim_stub.so"
DRIVER = CACHE / "abi_selftest_driver"

HAVE_DOCKER = shutil.which("docker") is not None and (
  subprocess.run(["docker", "info"], capture_output=True).returncode == 0
)


def _ensure_build() -> None:
  proc = subprocess.run(["bash", str(BUILD_SCRIPT)], capture_output=True, text=True, check=False)
  if proc.returncode != 0:
    pytest.fail(f"stub library build failed: {proc.stderr[-800:]}")


@pytest.fixture(scope="module")
def phase_output() -> str:
  if not HAVE_DOCKER:
    pytest.skip("docker daemon unavailable; stub library cannot execute")
  _ensure_build()
  script = "cd /tmp && /out/abi_selftest_driver /out/libcn_cosim_stub.so phase"
  run = subprocess.run(
    ["bash", str(DOCKER_RUN), IMAGE, str(CACHE), "--", "bash", "-c", script],
    capture_output=True,
    text=True,
    timeout=600,
    check=False,
  )
  assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
  return run.stdout


def test_phase_suite_all_checks_pass(phase_output: str) -> None:
  assert phase_output.count("CHECK FAIL") == 0
  assert phase_output.count("CHECK PASS") >= 200  # 48-tick loop x4 checks + negatives
  assert "PHASE_SUITE_DONE failures=0" in phase_output


def test_missing_drive_then_tick_is_state_error(phase_output: str) -> None:
  assert "CHECK PASS tick_without_drive_state" in phase_output


def test_double_collect_is_state_error(phase_output: str) -> None:
  assert "CHECK PASS double_collect_state" in phase_output


def test_collect_short_buffer_is_range_error(phase_output: str) -> None:
  assert "CHECK PASS collect_small_range" in phase_output
  assert "CHECK PASS collect_after_range_ok" in phase_output


def test_posedge_advances_exactly_once_per_tick(phase_output: str) -> None:
  # The driver checks posedge == before+t+1 on every one of the 48 ticks.
  assert phase_output.count("CHECK PASS posedge_exactly_one_per_tick") == 48


def test_echo_payload_bit_exact_and_credit_conserved(phase_output: str) -> None:
  assert "CHECK PASS echo_payload_exact" in phase_output
  assert "CHECK PASS credit_conservation" in phase_output


def test_abi_identity_hash_matches_sealed_protocol(phase_output: str) -> None:
  assert "CHECK PASS abi_hash_len_128" in phase_output
  expected = cn_abi.expected_abi_hash()
  assert len(expected) == 128
  proto = (H / "protocol/cn_protocol_v1.yaml").read_bytes()
  layout = (H / "protocol/generated/cn_layout.json").read_bytes()
  assert expected == hashlib.sha256(proto).hexdigest() + hashlib.sha256(layout).hexdigest()


def test_host_ctypes_binding_or_documented_fallback() -> None:
  """Host dlopen works when glibc is new enough; otherwise the error names the fallback."""
  try:
    api = cn_abi.load(SO)
  except cn_abi.CnAbiError as exc:
    assert exc.code == "HOST_GLIBC_TOO_OLD"
    pytest.skip(f"host glibc too old for the container-built stub ({exc.detail[:80]})")
  info = api.abi_info()
  assert info == cn_abi.expected_abi_hash()


def test_build_meta_pinned_to_current_sources() -> None:
  if not SO.is_file():
    pytest.skip("stub library not built yet")
  meta = cn_abi.build_meta()
  assert meta["schema"] == "cn_abi_stub_build/v1"
  assert meta["library_sha256"] == hashlib.sha256(SO.read_bytes()).hexdigest()
  assert meta["driver_sha256"] == hashlib.sha256(DRIVER.read_bytes()).hexdigest()
