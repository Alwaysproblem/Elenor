#!/usr/bin/env python3
"""Shared host-side helpers for the P02 hardware scripts.

The host launcher environment is intentionally stdlib-only (contract: no host
installs, no YAML dependency). Everything here is read-only with respect to
the repository; all execution happens inside the locked toolchain container.

See hardware/scripts/README.md for the command contract this module serves.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent

LOCK_REL = "hardware/toolchain/toolchain.lock"
REGISTRY_REL = "hardware/scripts/targets.json"
CONTAINER_CHECK_REL = "hardware/scripts/container_check.py"

# Exit codes shared by check, reproduce, and the in-container runner.
EXIT_OK = 0
EXIT_USAGE = 2  # bad arguments, unknown target/case, missing lock/image/docker
EXIT_NOT_IMPLEMENTED = 3  # registered target without an implementation
EXIT_BLOCKED = 4  # capability blocked (fpga-synth without Vivado)
EXIT_DOCKER = 5  # docker itself failed (spawn error, timeout, daemon error)
EXIT_CONFIG = 6  # config rejected by schema/target/section validation
EXIT_CASES = 7  # at least one executed case failed its assertions
EXIT_ENV = 8  # container environment/identity verification failed
EXIT_MANIFEST = 9  # reproduce: manifest missing/invalid/PASS-source mismatch
EXIT_MISMATCH = 10  # reproduce: deterministic outputs differ

IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Identity scope of the runtime source manifest (contract: hardware inputs,
# pipeline_validator, used examples; no host modifications).
SOURCE_SCOPE = ("hardware", "pipeline_validator", "examples")

CONTAINER_REPO = "/workspace"
CONTAINER_OUT = "/out"
CONTAINER_HOME = "/tmp/nexus-home"
CONTAINER_PYTHON = "/opt/validator/bin/python"

# Hard wall-clock cap for one container run (sbt cold elaboration dominates).
HOST_TIMEOUT_SECONDS = 4 * 3600


def fail(message: str, code: int) -> NoReturn:
  print(f"error: {message}", file=sys.stderr)
  raise SystemExit(code)


def repo_path(relative: str) -> Path:
  return REPO_ROOT / relative


def load_json(path: Path) -> dict:
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeError, ValueError) as exc:
    fail(f"cannot read JSON {path}: {exc}", EXIT_USAGE)


def write_json(path: Path, payload: dict) -> None:
  """Deterministic JSON output (sorted keys, stable indent, trailing newline)."""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_path(path: Path) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def sha256_text(text: str) -> str:
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_registry() -> dict:
  registry = load_json(repo_path(REGISTRY_REL))
  if registry.get("schema") != 1 or not isinstance(registry.get("targets"), dict):
    fail(f"{REGISTRY_REL} must have schema 1 and a 'targets' mapping", EXIT_USAGE)
  return registry


def load_lock() -> dict:
  """Load and minimally validate the parent-owned toolchain lock.

  The runner only *consumes* the lock; it never writes it. Required here:
  schema==1 and image.id in exact Docker image-config form (sha256:<64hex>).
  """
  path = repo_path(LOCK_REL)
  if not path.is_file():
    fail(f"missing {LOCK_REL}; run `hardware/scripts/env build --locked` first", EXIT_USAGE)
  lock = load_json(path)
  if lock.get("schema") != 1:
    fail(f"{LOCK_REL} schema must be 1, got {lock.get('schema')!r}", EXIT_USAGE)
  image = lock.get("image")
  if not isinstance(image, dict) or not IMAGE_ID_RE.match(str(image.get("id", ""))):
    fail(
      f"{LOCK_REL} image.id must be an exact Docker image-config ID (sha256:<64hex>); "
      "mutable tags are not accepted",
      EXIT_USAGE,
    )
  return lock


def lock_sha256() -> str:
  return sha256_path(repo_path(LOCK_REL))


def git(args: list[str], binary: bool = False):
  result = subprocess.run(
    ["git", *args], cwd=str(REPO_ROOT), check=True, capture_output=True, text=not binary
  )
  return result.stdout


def source_identity() -> dict:
  """Reproducible identity of the working tree within SOURCE_SCOPE.

  - head: HEAD commit (full)
  - dirty: True when tracked files in scope differ from HEAD
  - diff_sha256: sha256 over full binary `git diff HEAD -- <scope>`
  - untracked_files: sorted map path -> sha256 for untracked, non-ignored files
  - untracked_sha256: sha256 over the serialized untracked map
  """
  head = git(["rev-parse", "HEAD"]).strip()
  diff = git(
    [
      "diff",
      "--binary",
      "--full-index",
      "--no-ext-diff",
      "--no-textconv",
      "--no-color",
      "HEAD",
      "--",
      *SOURCE_SCOPE,
    ]
  )
  listing = git(["ls-files", "--others", "--exclude-standard", "-z", "--", *SOURCE_SCOPE], binary=True)
  untracked: dict[str, str] = {}
  for item in listing.split(b"\0"):
    if not item:
      continue
    rel = item.decode("utf-8")
    path = REPO_ROOT / rel
    if path.is_file():
      untracked[rel] = sha256_path(path)
  untracked_serialized = "".join(f"{path}\0{untracked[path]}\0" for path in sorted(untracked))
  return {
    "head": head,
    "dirty": bool(diff.strip()),
    "diff_sha256": sha256_text(diff),
    "untracked_sha256": sha256_text(untracked_serialized),
    "untracked_files": untracked,
  }


def identity_mismatches(a: dict, b: dict) -> list[str]:
  """Compare two source_identity() results; returns human-readable diffs."""
  problems: list[str] = []
  for key in ("head", "dirty", "diff_sha256", "untracked_sha256"):
    if a.get(key) != b.get(key):
      problems.append(f"source.{key}: manifest={a.get(key)!r} current={b.get(key)!r}")
  a_files = a.get("untracked_files", {})
  b_files = b.get("untracked_files", {})
  for path in sorted(set(a_files) - set(b_files)):
    problems.append(f"source.untracked: {path} no longer present")
  for path in sorted(set(b_files) - set(a_files)):
    problems.append(f"source.untracked: {path} is new")
  for path in sorted(set(a_files) & set(b_files)):
    if a_files[path] != b_files[path]:
      problems.append(f"source.untracked: {path} content changed")
  return problems


def docker_binary() -> str:
  found = shutil_which("docker")
  if found is None:
    fail("docker CLI not found on PATH; the host launcher requires docker", EXIT_USAGE)
  return found


def shutil_which(name: str):
  from shutil import which

  return which(name)


def require_image(docker: str, image_id: str) -> None:
  result = subprocess.run(
    [docker, "image", "inspect", "--format", "{{.Id}}", image_id], capture_output=True, text=True
  )
  if result.returncode != 0:
    fail(
      f"locked image {image_id} is not present locally "
      "(build it with `hardware/scripts/env build --locked`)",
      EXIT_USAGE,
    )
  observed = result.stdout.strip()
  if observed != image_id:
    fail(f"docker reports image ID {observed!r} for requested {image_id!r}", EXIT_USAGE)


def container_run_argv(
  docker: str, image_id: str, repo_root: Path, out_dir: Path, expect_lock_sha: str, container_name: str
) -> list[str]:
  """Exact docker invocation for one check/reproduce execution.

  Contract mounts: repository read-only at /workspace, output writable at
  /out, host UID:GID, no network, all capabilities dropped, no privilege
  escalation, no socket/privileged/device access, no installs.
  """
  return [
    docker,
    "run",
    "--rm",
    "--name",
    container_name,
    "--network",
    "none",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--user",
    f"{os.getuid()}:{os.getgid()}",
    "-e",
    f"HOME={CONTAINER_HOME}",
    "--entrypoint",
    CONTAINER_PYTHON,
    "-w",
    CONTAINER_REPO,
    "-v",
    f"{repo_root}:{CONTAINER_REPO}:ro",
    "-v",
    f"{out_dir}:{CONTAINER_OUT}",
    image_id,
    f"{CONTAINER_REPO}/{CONTAINER_CHECK_REL}",
    "--request",
    f"{CONTAINER_OUT}/request.json",
    "--repo",
    CONTAINER_REPO,
    "--out",
    CONTAINER_OUT,
    "--expect-lock-sha256",
    expect_lock_sha,
    "--expect-image-id",
    image_id,
  ]


def run_container(argv: list[str], container_name: str, log_path: Path) -> int:
  """Run the container, streaming combined output to log_path.

  Returns the container exit code. On host-side timeout the container is
  force-removed and EXIT_DOCKER is raised; nothing is left running.
  """
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with open(log_path, "w", encoding="utf-8") as log:
    log.write(f"# argv: {json.dumps(argv)}\n\n")
    log.flush()
    try:
      proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    except OSError as exc:
      fail(f"cannot start docker: {exc}", EXIT_DOCKER)
    try:
      code = proc.wait(timeout=HOST_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
      proc.kill()
      proc.wait()
      subprocess.run([argv[0], "rm", "-f", "--force", container_name], capture_output=True)
      fail(f"container exceeded {HOST_TIMEOUT_SECONDS}s wall clock and was removed", EXIT_DOCKER)
  return code
