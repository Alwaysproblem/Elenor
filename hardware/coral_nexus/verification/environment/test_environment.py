"""Behavioral tests for the C002 locked build environment.

Split into two layers:

* Lock/dockerfile invariants (always run, no docker required): the image is
  built from a digest-pinned base, the expected tool versions exist in
  image.lock, tools.lock sha256s come from the pinned upstream sources, and
  neither the Dockerfile nor the handler ever elevates or mounts the docker
  socket.
* Container integration (skipped without a docker daemon): a real
  ``env.sh build`` + ``env.sh verify`` round trip whose report must pass.

The integration tests are heavy (image build ~1 GB); they are controlled by
a ``.coral_docker_ok`` flag file so the fast suite stays fast.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
ENV_SH = H / "scripts/handlers/env.sh"
DOCKERFILE = H / "env/Dockerfile"
IMAGE_LOCK = H / "env/image.lock"
TOOLS_LOCK = H / "env/tools.lock"


def _load(path: Path) -> dict:
  return json.loads(path.read_text())


def _strip_comments(text: str) -> str:
  return "\n".join(
    line for line in text.splitlines() if not line.lstrip().startswith("#")
  )


def test_dockerfile_pins_base_by_digest() -> None:
  text = DOCKERFILE.read_text()
  from_lines = re.findall(r"^FROM (.+)$", text, re.M)
  assert len(from_lines) == 1
  assert re.fullmatch(r"\S+@sha256:[0-9a-f]{64}", from_lines[0]), from_lines[0]


def test_dockerfile_matches_locked_digest() -> None:
  text = DOCKERFILE.read_text()
  digest = _load(IMAGE_LOCK)["base"]["digest"]
  assert f"@{digest}" in text


def test_dockerfile_never_elevates_or_mounts_socket() -> None:
  text = _strip_comments(DOCKERFILE.read_text())
  assert "privileged" not in text
  assert "docker.sock" not in text
  assert "sudoers" not in text
  assert re.search(r"^USER \$\{_USERNAME\}$", text, re.M)


def test_dockerfile_installs_locked_bazel_and_symlink() -> None:
  text = _strip_comments(DOCKERFILE.read_text())
  lock = _load(IMAGE_LOCK)
  assert f"apt-get install -y -qq {lock['bazel']['package']}" in text
  assert "bazel-9.1.0 /usr/local/bin/bazel" in text


def test_image_lock_records_real_versions_and_deviation() -> None:
  lock = _load(IMAGE_LOCK)
  assert lock["schema"] == "cn_image_lock/v1"
  assert lock["expected_versions"]["buildifier"] == "7.3.1"
  assert lock["rv32_cross"]["mabi"] == "ilp32"
  assert lock["rv32_cross"]["march"].startswith("rv32imf_zve32f")
  assert lock["deviations_from_upstream_dockerfile"], "deviations must be recorded"


def test_tools_lock_hashes_match_upstream_pins() -> None:
  lock = _load(TOOLS_LOCK)
  assert (
    lock["rv32_toolchain"]["sha256"]
    == "de06690c2da5cd783d76b2998208bd4db4dcdc22dec146c7b0a5ee1af40d3db7"
  )
  assert (
    lock["verilator"]["sha256"]
    == "8c8d2e11e6ad32f641dd250742a94195ddecb912e2e2dabe2f42ddbbb99c1092"
  )
  for entry in lock.values():
    if isinstance(entry, dict) and "sha256" in entry:
      assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry


def test_env_handler_rejects_unknown_subcommand() -> None:
  proc = subprocess.run(
    ["bash", str(ENV_SH), "export"], capture_output=True, text=True
  )
  assert proc.returncode == 2
  assert "env.sh" in proc.stderr


def test_env_handler_never_mounts_docker_socket() -> None:
  text = _strip_comments(ENV_SH.read_text())
  assert "docker.sock" not in text
  assert "--privileged" not in text
  assert '--user "$(id -u):$(id -g)"' in text


def test_verify_inside_asserts_exact_host_output_and_rv32_target() -> None:
  text = (H / "env/verify_inside.sh").read_text()
  assert "host-probe-ok 44" in text
  assert "rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma" in text
  assert "ELF32" in text and "RISC-V" in text
  assert "var/run/docker.sock" in text


def test_host_probe_output_is_deterministic() -> None:
  source = (H / "env/host_probe.c").read_text()
  assert "host-probe-ok" in source
  # 1..8 sums to 36; +8 == 44 asserted by verify_inside.sh.
  assert "44" in source


@pytest.mark.skipif(
  subprocess.run(["docker", "info"], capture_output=True).returncode != 0
  or not (REPO_ROOT / ".coral_docker_ok").exists(),
  reason="docker daemon not reachable or .coral_docker_ok flag absent",
)
class TestContainerRoundTrip:
  """Real build + verify round trip (opt-in via .coral_docker_ok)."""

  def test_build_and_verify(self, tmp_path: Path) -> None:
    build = subprocess.run(
      ["bash", str(ENV_SH), "build", "--out", str(tmp_path / "build")],
      capture_output=True,
      text=True,
      cwd=REPO_ROOT,
    )
    assert build.returncode == 0, build.stderr[-2000:]
    verify = subprocess.run(
      ["bash", str(ENV_SH), "verify", "--out", str(tmp_path / "verify")],
      capture_output=True,
      text=True,
      cwd=REPO_ROOT,
    )
    assert verify.returncode == 0, verify.stderr[-2000:]
    report = json.loads((tmp_path / "verify/environment_checks.json").read_text())
    assert report["pass"] is True
    assert report["failed_checks"] == []

  def test_tampered_toolchain_sha_fails(self, tmp_path: Path) -> None:
    """Negative: a wrong locked sha256 must fail the in-container verify."""
    import shutil

    fake_h = tmp_path / "hardware/coral_nexus"
    (fake_h / "env").mkdir(parents=True)
    tools = json.loads(TOOLS_LOCK.read_text())
    tools["rv32_toolchain"]["sha256"] = "0" * 64
    (fake_h / "env/tools.lock").write_text(json.dumps(tools))
    for name in ("host_probe.c", "rv32_probe.S"):
      shutil.copy(H / "env" / name, fake_h / "env" / name)
    tag = _load(IMAGE_LOCK)["tag"]
    proc = subprocess.run(
      [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "REPORT_DIR=/report",
        "--env",
        "TOOLCHAIN_DIR=/toolchain",
        "--env",
        "H_ROOT=/fake/hardware/coral_nexus",
        "-v",
        f"{tmp_path}:/fake:ro",
        "-v",
        f"{tmp_path}:/report",
        "-v",
        f"{REPO_ROOT}:/workspace:ro",
        "-v",
        f"{REPO_ROOT / 'out/coral_nexus/cache/toolchain'}:/toolchain",
        tag,
        "bash",
        "/workspace/hardware/coral_nexus/env/verify_inside.sh",
      ],
      capture_output=True,
      text=True,
      cwd=REPO_ROOT,
    )
    assert proc.returncode != 0
    log = (tmp_path / "verify_inside.log").read_text()
    assert "toolchain_sha256" in log and "FAIL" in log
