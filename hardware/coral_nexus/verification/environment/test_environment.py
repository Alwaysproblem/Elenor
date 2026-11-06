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
    ["bash", str(ENV_SH), "teardown"], capture_output=True, text=True
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


SEEDS = H / "env/seed_targets.json"
DEVC = REPO_ROOT / ".devcontainer/coral_nexus/devcontainer.json"


def test_seed_inventory_entries_all_carry_license_and_hash() -> None:
  seeds = _load(SEEDS)["seeds"]
  notices = (H / "env/notices.md").read_text()
  names = {s["name"] for s in seeds}
  assert {
    "rv32_toolchain",
    "verilator",
    "riscv_isa_sim",
    "trace_processor_shell",
    "bazel_vendor_cache",
  } <= names
  for seed in seeds:
    if seed["kind"] not in {"deferred", "not-applicable"}:
      assert re.fullmatch(r"[0-9a-f]{64}", seed["sha256"]), seed["name"]
      assert seed["name"] in notices, f"{seed['name']} missing from notices.md"


def test_seed_export_completes_from_cache(tmp_path: Path) -> None:
  """Real export run (uses the shared cache built during C004)."""
  proc = subprocess.run(
    [
      "bash",
      str(H / "scripts/handlers/env.sh"),
      "export",
      "--out",
      str(tmp_path / "export"),
    ],
    capture_output=True,
    text=True,
    cwd=REPO_ROOT,
  )
  assert proc.returncode == 0, proc.stderr[-2000:]
  manifest = json.loads((tmp_path / "export/export_manifest.json").read_text())
  assert manifest["complete"] is True
  assert manifest["failures"] == []
  exported = {n for n, e in manifest["seeds"].items() if e["status"] == "exported"}
  assert "riscv_isa_sim" in exported and "trace_processor_shell" in exported
  assert manifest["seeds"]["bazel_vendor_cache"]["status"] == "exported"


def test_seed_export_refuses_non_empty_directory(tmp_path: Path) -> None:
  out = tmp_path / "export"
  out.mkdir()
  (out / "stale.txt").write_text("stale\n")
  proc = subprocess.run(
    [
      "bash",
      str(H / "scripts/handlers/env.sh"),
      "export",
      "--out",
      str(out),
    ],
    capture_output=True,
    text=True,
    cwd=REPO_ROOT,
  )
  assert proc.returncode != 0
  assert "not empty" in proc.stderr


def test_devcontainer_references_locked_image_without_elevation() -> None:
  doc = json.loads(DEVC.read_text())
  lock = _load(IMAGE_LOCK)
  assert doc["image"] == lock["tag"]
  assert doc["remoteUser"] == "builder"
  assert "build" not in doc  # image is built by env.sh, not rebuilt here
  rendered = json.dumps(doc)
  assert "docker.sock" not in rendered
  assert "privileged" not in rendered


def _export(seeds: Path, cache: Path, out: Path) -> subprocess.CompletedProcess:
  return subprocess.run(
    [
      "python3",
      str(H / "scripts/lib/seed_export.py"),
      "--seeds",
      str(seeds),
      "--cache",
      str(cache),
      "--out",
      str(out),
    ],
    capture_output=True,
    text=True,
  )


def test_seed_export_fails_precisely_when_a_directory_seed_is_missing(
  tmp_path: Path,
) -> None:
  """Deleting a required archive must fail, naming it (no silent refill)."""
  seeds = tmp_path / "seeds.json"
  seeds.write_text(
    json.dumps(
      {
        "seeds": [
          {
            "name": "bazel_vendor_cache",
            "kind": "local-dir",
            "path": str(tmp_path / "absent"),
            "sha256": "0" * 64,
            "license": "test",
          }
        ]
      }
    )
  )
  proc = _export(seeds, tmp_path / "cache", tmp_path / "export")
  assert proc.returncode == 1
  manifest = json.loads((tmp_path / "export/export_manifest.json").read_text())
  assert manifest["complete"] is False
  assert any("bazel_vendor_cache" in f and "missing" in f for f in manifest["failures"])


def test_seed_export_fails_on_directory_digest_mismatch(tmp_path: Path) -> None:
  src = tmp_path / "src"
  (src / "bin").mkdir(parents=True)
  (src / "bin/tool").write_text("payload\n")
  seeds = tmp_path / "seeds.json"
  seeds.write_text(
    json.dumps(
      {
        "seeds": [
          {
            "name": "dirseed",
            "kind": "local-dir",
            "path": str(src),
            "sha256": "1" * 64,
            "license": "test",
          }
        ]
      }
    )
  )
  proc = _export(seeds, tmp_path / "cache", tmp_path / "export")
  assert proc.returncode == 1
  manifest = json.loads((tmp_path / "export/export_manifest.json").read_text())
  assert any("dirseed" in f and "digest" in f for f in manifest["failures"])


def test_directory_seed_archive_preserves_in_tree_symlinks(tmp_path: Path) -> None:
  """bazel vendor aliases (bin/python3 -> python3.11) must survive archiving."""
  import hashlib
  import os

  src = tmp_path / "src"
  (src / "bin").mkdir(parents=True)
  (src / "bin/python3.11").write_text("interp\n")
  os.symlink("python3.11", src / "bin/python3")
  os.symlink("/nonexistent/outside", src / "bin/dead")

  digest = hashlib.sha256()
  for rel, h in sorted(
    [
      ("bin/python3.11", hashlib.sha256(b"interp\n").hexdigest()),
      ("bin/python3", "symlink:python3.11"),
    ]
  ):
    digest.update(rel.encode())
    digest.update(b"\0")
    digest.update(h.encode())
    digest.update(b"\0")

  seeds = tmp_path / "seeds.json"
  seeds.write_text(
    json.dumps(
      {
        "seeds": [
          {
            "name": "dirseed",
            "kind": "local-dir",
            "path": str(src),
            "sha256": digest.hexdigest(),
            "license": "test",
          }
        ]
      }
    )
  )
  proc = _export(seeds, tmp_path / "cache", tmp_path / "export")
  assert proc.returncode == 0, proc.stdout + proc.stderr
  archive = tmp_path / "export/dirseed"
  assert (archive / "bin/python3").is_symlink()
  assert os.readlink(archive / "bin/python3") == "python3.11"
  assert not (archive / "bin/dead").exists()
  manifest = json.loads((tmp_path / "export/export_manifest.json").read_text())
  assert manifest["seeds"]["dirseed"]["dropped_links"] == ["bin/dead"]
