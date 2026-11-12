"""C025 five-phase C ABI & `RtlTileBackend` wire-path tests (`cn unit --suite w1`).

Executes BOTH:
1. Real Python `wire.py` + `abi.py` + `RtlTileBackend(TilePort)` inside `coral-nexus-builder:c002`
   against `libcn_cosim_stub.so` (`in_container_abi_verify.py`):
   - All 10 `(channel, dir)` slots match `cn_layout.json` `payload_words` (`CTRL.G2T=104B`,
     `PROGRAM.T2G=128B`, `MEMORY.T2G=136B`);
   - `bp_layout` is repeatable and read-only (zero functional state mutation);
   - Full-width `wire.pack` messages (`TASK_PLAN` 96B, `TASK_COMMIT` 104B, `EVENT_NOTIFY` 104B,
     `PROGRAM_INSTALL` 96B, `PROGRAM_READ_RESP` 96B -> `PROGRAM_INSTALL_ACK` 128B, `MEM_RSP` 72B,
     and Tile-originated `MEM_REQ` 136B) round-trip through SV DUT pins and decode via `wire.unpack`;
   - Staged T2G output isolation (P2 output never leaks into same-tick P3 collect);
   - Negatives: missing `bp_drive` -> `CN_ERR_STATE`, double `bp_collect` -> `CN_ERR_STATE`,
     uncollected records preserved across ticks, short `bp_collect` -> `CN_ERR_RANGE` without
     truncation, mutating stalled payload -> `CN_EP_PAYLOAD_READY_MISMATCH`, post-edge glitch ->
     `CN_EP_PAYLOAD_READY_MISMATCH`, and tampering SV DUT T2G output pin (`CN_STUB_FLAG_CORRUPT_T2G`)
     is caught immediately (proving no C-side self-echo).
2. C `dlopen` driver (`abi_selftest_driver phase`) checking `posedge_count == +1` on every tick and
   credit conservation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

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
    pytest.fail(f"stub library build failed:\n{proc.stdout[-1200:]}\n{proc.stderr[-1200:]}")


@pytest.fixture(scope="module")
def python_in_container_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  if not HAVE_DOCKER:
    pytest.skip("docker daemon unavailable")
  _ensure_build()
  out_dir = tmp_path_factory.mktemp("c025_py_verify")
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
    f"{out_dir.resolve()}:/out",
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
  assert run.returncode == 0, f"stdout:\n{run.stdout}\nstderr:\n{run.stderr}"
  return json.loads((out_dir / "python_abi_verify.json").read_text())


@pytest.fixture(scope="module")
def c_phase_stdout() -> str:
  if not HAVE_DOCKER:
    pytest.skip("docker daemon unavailable")
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


def test_slot_geometries_match_cn_layout_payload_words() -> None:
  slots = cn_abi.expected_slot_geometries()
  assert len(slots) == 10
  by_link = {(s["channel"], s["dir"]): s["len"] for s in slots}
  # Must accommodate TASK_PLAN (96B), TASK_COMMIT (104B), PROGRAM_INSTALL_ACK (128B), MEM_REQ (136B)!
  assert by_link[("CTRL", "G2T")] == 104
  assert by_link[("CTRL", "T2G")] == 96
  assert by_link[("EVENT", "G2T")] == 104
  assert by_link[("EVENT", "T2G")] == 96
  assert by_link[("PROGRAM", "G2T")] == 96
  assert by_link[("PROGRAM", "T2G")] == 128
  assert by_link[("MEMORY", "G2T")] == 72
  assert by_link[("MEMORY", "T2G")] == 136


def test_in_container_python_wire_and_rtl_backend_roundtrip(
  python_in_container_report: dict[str, Any],
) -> None:
  checks = set(python_in_container_report["checks"])
  required = {
    "layout_repeatable_readonly",
    "layout_mutates_no_state",
    "slot_ctrl_g2t_104b",
    "slot_program_t2g_128b",
    "slot_memory_t2g_136b",
    "missing_drive_returns_state_err",
    "double_collect_returns_state_err",
    "unstable_payload_rejected",
    "unstable_payload_sets_proto_err",
    "staged_t2g_visible_only_as_next_tick_candidate",
    "short_collect_returns_range",
    "uncollected_records_preserved_across_ticks",
    "caps_resp_decoded_via_wire_unpack",
    "post_edge_glitch_rejected",
    "post_edge_glitch_sets_proto_err",
    "rtl_backend_implements_tile_port",
    "unit_stub_backend_kind_tagged_stub",
    "cs1_guard_rejects_stub",
    "send_does_not_tick_verilator",
    "t2g_136b_mem_req_decoded",
    "ack_task_plan_96b_accepted",
    "ack_task_commit_104b_hold_ready",
    "ack_stale_ctx_gen_computed_by_sv_dut",
    "event_channel_104b_to_96b_roundtrip",
    "program_install_digest32_roundtrip",
    "program_128b_install_ack_roundtrip",
    "memory_terminal_ack_roundtrip",
    "rtl_backend_credits_conserved",
    "dut_t2g_tamper_detected_not_self_echo",
  }
  missing = required - checks
  assert not missing, f"missing verified checks: {sorted(missing)}"


def test_c_driver_posedge_and_96b_task_plan_roundtrip(c_phase_stdout: str) -> None:
  assert c_phase_stdout.count("CHECK FAIL") == 0
  assert c_phase_stdout.count("CHECK PASS posedge_exactly_one_per_tick") == 64
  assert "CHECK PASS layout_repeatable_readonly" in c_phase_stdout
  assert "CHECK PASS ctrl_g2t_slot_104b" in c_phase_stdout
  assert "CHECK PASS memory_t2g_slot_136b" in c_phase_stdout
  assert "CHECK PASS all_12_task_plans_acked_by_dut" in c_phase_stdout
  assert "CHECK PASS credit_conservation" in c_phase_stdout
  assert "PHASE_SUITE_DONE failures=0" in c_phase_stdout


def test_abi_identity_hash_and_build_meta(c_phase_stdout: str) -> None:
  assert "CHECK PASS abi_hash_len_128" in c_phase_stdout
  expected = cn_abi.expected_abi_hash()
  assert len(expected) == 128
  meta = cn_abi.build_meta()
  assert meta["schema"] == "cn_abi_stub_build/v1"
  assert meta["library_sha256"] == hashlib.sha256(SO.read_bytes()).hexdigest()
  assert meta["driver_sha256"] == hashlib.sha256(DRIVER.read_bytes()).hexdigest()
