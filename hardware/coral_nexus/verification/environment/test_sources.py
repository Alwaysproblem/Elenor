"""Behavioral tests for the C001 source lock tooling.

Every case drives the real scripts (``materialize.sh`` / ``verify_sources.sh``)
through a subprocess against a fixture repository built in ``tmp_path``: a
tiny upstream git repo pinned as a real gitlink in a fixture superproject.
No test touches the pristine submodules under ``hardware/coral_nexus/upstream``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
MATERIALIZE = REPO_ROOT / "hardware/coral_nexus/scripts/lib/materialize.sh"
VERIFY = REPO_ROOT / "hardware/coral_nexus/scripts/lib/verify_sources.sh"


def _git(repo: Path, *args: str, check: bool = True) -> str:
  proc = subprocess.run(
    ["git", "-C", str(repo), *args], capture_output=True, text=True
  )
  if check and proc.returncode != 0:
    raise AssertionError(f"git {args} failed: {proc.stderr}")
  return proc.stdout


def _digest(repo: Path, commit: str) -> str:
  tool = REPO_ROOT / "hardware/coral_nexus/scripts/lib/sources_tool.py"
  proc = subprocess.run(
    [sys.executable, str(tool), "archive-digest", "--repo", str(repo), "--commit", commit],
    capture_output=True,
    text=True,
    check=True,
  )
  return proc.stdout.strip()


class Fixture:
  """Upstream repo pinned as a gitlink inside a fixture superproject."""

  def __init__(self, root: Path) -> None:
    self.root = root
    self.upstream = root / "upstream_src"
    self.super = root / "super"
    self._build()

  def _build(self) -> None:
    self.upstream.mkdir(parents=True)
    _git(self.upstream, "init", "-q", "-b", "main")
    (self.upstream / "LICENSE").write_text("fixture license\n")
    (self.upstream / "README.md").write_text("fixture upstream\n")
    code = self.upstream / "code"
    code.mkdir()
    (code / "main.c").write_text("int main(void) { return 7; }\n")
    _git(self.upstream, "add", "-A")
    _git(
      self.upstream,
      "-c",
      "user.name=fixture",
      "-c",
      "user.email=fixture@example.invalid",
      "commit",
      "-qm",
      "upstream base",
    )
    # A second commit lets tests move HEAD away from the pin.
    (code / "second.c").write_text("int second = 2;\n")
    _git(self.upstream, "add", "-A")
    _git(
      self.upstream,
      "-c",
      "user.name=fixture",
      "-c",
      "user.email=fixture@example.invalid",
      "commit",
      "-qm",
      "upstream second",
    )
    _git(self.upstream, "checkout", "-q", "HEAD~1")
    self.pinned = _git(self.upstream, "rev-parse", "HEAD").strip()

    self.super.mkdir(parents=True)
    _git(self.super, "init", "-q", "-b", "main")
    _git(
      self.super,
      "-c",
      "protocol.file.allow=always",
      "submodule",
      "add",
      "-q",
      "--name",
      "src",
      self.upstream.resolve().as_uri(),
      "sub/src",
    )
    _git(self.super, "add", "-A")
    _git(
      self.super,
      "-c",
      "user.name=fixture",
      "-c",
      "user.email=fixture@example.invalid",
      "commit",
      "-qm",
      "super base",
    )
    self.env = self.super / "env"
    self.env.mkdir()
    self.patches = self.super / "patches"
    self.patches.mkdir()

  def write_lock(self, **overrides: str) -> Path:
    entry = {
      "name": "src",
      "kind": "submodule",
      "url": self.upstream.resolve().as_uri(),
      "path": "sub/src",
      "commit": self.pinned,
      "archive_sha256": _digest(self.upstream, self.pinned),
      "licenses": ["LICENSE"],
    }
    entry.update(overrides)
    lock = {
      "schema": "cn_sources_lock/v1",
      "sources": [entry],
      "embedded_pins": [],
      "overlay": "env/overlay.yaml",
      "patches": {
        "series": "patches/series",
        "owners": "patches/file_owners.yaml",
      },
    }
    path = self.env / "sources.lock"
    path.write_text(json.dumps(lock, indent=2) + "\n")
    (self.env / "overlay.yaml").write_text(
      "schema: cn_overlay/v1\nentries: []\n"
    )
    (self.patches / "series").write_text("# empty series\n")
    (self.patches / "file_owners.yaml").write_text(
      "schema: cn_patch_owners/v1\nowners:\n  T: [code/**]\n"
    )
    return path

  def materialize(self, out: Path, lock: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
      ["bash", str(MATERIALIZE), "--lock", str(lock), "--out", str(out)],
      capture_output=True,
      text=True,
      cwd=self.super,
    )

  def verify(self, out: Path, lock: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
      ["bash", str(VERIFY), "--lock", str(lock), "--out", str(out)],
      capture_output=True,
      text=True,
      cwd=self.super,
    )


@pytest.fixture()
def fixture(tmp_path: Path) -> Fixture:
  return Fixture(tmp_path)


def _report(out: Path) -> dict:
  return json.loads((out / "sources_report.json").read_text())


def test_materialize_twice_produces_identical_digest(fixture: Fixture) -> None:
  lock = fixture.write_lock()
  first, second = fixture.root / "w1", fixture.root / "w2"
  proc1 = fixture.materialize(first, lock)
  proc2 = fixture.materialize(second, lock)
  assert proc1.returncode == 0, proc1.stderr
  assert proc2.returncode == 0, proc2.stderr
  digest1 = json.loads((first / "materialize_manifest.json").read_text())[
    "work_tree_digest"
  ]
  digest2 = json.loads((second / "materialize_manifest.json").read_text())[
    "work_tree_digest"
  ]
  assert digest1 == digest2
  assert (first / "src/code/main.c").read_text().startswith("int main")
  assert not (first / "src/code/second.c").exists()
  verify = fixture.verify(fixture.root / "report", lock)
  assert verify.returncode == 0, verify.stdout
  assert _report(fixture.root / "report")["pass"] is True


def test_tampered_commit_is_rejected(fixture: Fixture) -> None:
  lock = fixture.write_lock()
  good_content = lock.read_text()
  other = _git(fixture.upstream, "rev-parse", "main").strip()
  tampered = fixture.write_lock(commit=other)
  assert tampered == lock  # same file, now holding the wrong commit
  proc = fixture.verify(fixture.root / "report", tampered)
  assert proc.returncode == 1
  report = _report(fixture.root / "report")
  failed = [c for c in report["checks"] if not c["pass"]]
  assert any(c["check"] == "src.gitlink" for c in failed)
  assert any(c["check"] == "src.archive_sha256" for c in failed)
  # Restore the sound lock: the fixture pin itself still verifies.
  lock.write_text(good_content)
  assert fixture.verify(fixture.root / "report2", lock).returncode == 0


def test_dirty_submodule_is_rejected(fixture: Fixture) -> None:
  lock = fixture.write_lock()
  (fixture.super / "sub/src/README.md").write_text("dirty\n")
  proc = fixture.verify(fixture.root / "report", lock)
  assert proc.returncode == 1
  report = _report(fixture.root / "report")
  assert any(
    c["check"] == "src.pristine" and not c["pass"] for c in report["checks"]
  )


def test_drifted_lock_digest_is_rejected(fixture: Fixture) -> None:
  fixture.write_lock()
  lock = fixture.write_lock(archive_sha256="0" * 64)
  proc = fixture.verify(fixture.root / "report", lock)
  assert proc.returncode == 1
  report = _report(fixture.root / "report")
  failed = [c for c in report["checks"] if not c["pass"]]
  assert any(c["check"] == "src.archive_sha256" for c in failed)


def test_bad_patch_context_is_rejected(fixture: Fixture) -> None:
  lock = fixture.write_lock()
  patch = fixture.patches / "t" / "0001-break.patch"
  patch.parent.mkdir()
  patch.write_text(
    "--- a/code/main.c\n"
    "+++ b/code/main.c\n"
    "@@ -1 +1 @@\n"
    "-int main(void) { return 42; }\n"
    "+int main(void) { return 8; }\n"
  )
  (fixture.patches / "series").write_text("src:t/0001-break.patch\n")
  # (the hunk below claims the original line was different from the pin)
  out = fixture.root / "work"
  proc = fixture.materialize(out, lock)
  assert proc.returncode != 0
  assert "does not apply cleanly" in proc.stderr


def test_valid_patch_applies_in_work_root_only(fixture: Fixture) -> None:
  lock = fixture.write_lock()
  patch = fixture.patches / "t" / "0001-ok.patch"
  patch.parent.mkdir()
  patch.write_text(
    "--- a/code/main.c\n"
    "+++ b/code/main.c\n"
    "@@ -1 +1 @@\n"
    "-int main(void) { return 7; }\n"
    "+int main(void) { return 9; }\n"
  )
  (fixture.patches / "series").write_text("src:t/0001-ok.patch\n")
  out = fixture.root / "work"
  proc = fixture.materialize(out, lock)
  assert proc.returncode == 0, proc.stderr
  assert "return 9" in (out / "src/code/main.c").read_text()
  # Pristine submodule untouched by patch application.
  assert "return 7" in (fixture.super / "sub/src/code/main.c").read_text()


def test_overlay_undeclared_overwrite_is_rejected(fixture: Fixture) -> None:
  lock = fixture.write_lock()
  overlay_file = fixture.env / "over README.md"
  overlay_file.write_text("replaced\n")
  (fixture.env / "overlay.yaml").write_text(
    "schema: cn_overlay/v1\n"
    "entries:\n"
    "  - source: src\n"
    "    src: env/over README.md\n"
    "    dst: README.md\n"
    "    mode: add\n"
  )
  out = fixture.root / "work"
  proc = fixture.materialize(out, lock)
  assert proc.returncode != 0
  assert "mode is not 'overwrite'" in proc.stderr


def test_double_owner_is_rejected(fixture: Fixture) -> None:
  fixture.write_lock()
  for name, body in {
    "a.patch": "--- a/code/main.c\n+++ b/code/main.c\n@@ -1 +1 @@\n-x\n+y\n",
    "b.patch": "--- a/code/main.c\n+++ b/code/main.c\n@@ -1 +1 @@\n-x\n+z\n",
  }.items():
    (fixture.patches / name).write_text(body)
  (fixture.patches / "series").write_text("src:a.patch\nsrc:b.patch\n")
  (fixture.patches / "file_owners.yaml").write_text(
    "schema: cn_patch_owners/v1\n"
    "owners:\n"
    "  A: [code/**]\n"
    "  B: [code/main.c]\n"
  )
  lock = fixture.env / "sources.lock"
  proc = fixture.verify(fixture.root / "report", lock)
  assert proc.returncode == 1
  report = _report(fixture.root / "report")
  failed = [c for c in report["checks"] if not c["pass"]]
  assert any(c["check"] == "patches.ownership" for c in failed)
  assert "multiple owners" in next(
    c["detail"] for c in failed if c["check"] == "patches.ownership"
  )


def test_unowned_patch_file_is_rejected(fixture: Fixture) -> None:
  fixture.write_lock()
  (fixture.patches / "free.patch").write_text(
    "--- a/LICENSE\n+++ b/LICENSE\n@@ -1 +1 @@\n-fixture license\n-x\n"
  )
  (fixture.patches / "series").write_text("src:free.patch\n")
  proc = fixture.verify(fixture.root / "report", fixture.env / "sources.lock")
  assert proc.returncode == 1
  report = _report(fixture.root / "report")
  assert any(
    c["check"] == "patches.ownership" and "no owning package" in c["detail"]
    for c in report["checks"]
    if not c["pass"]
  )


def test_refuses_non_empty_work_root(fixture: Fixture) -> None:
  lock = fixture.write_lock()
  out = fixture.root / "work"
  out.mkdir()
  (out / "stale.txt").write_text("stale\n")
  proc = fixture.materialize(out, lock)
  assert proc.returncode != 0
  assert "not empty" in proc.stderr


def test_materialize_applies_patches_when_out_is_inside_superproject(fixture: Fixture) -> None:
  """Regression: when `--out` lives inside the superproject tree (e.g. `out/...`),
  `git apply` must NOT walk up to `super/.git` and silently skip hunks."""
  lock = fixture.write_lock()
  patch_body = (
    "--- a/code/main.c\n"
    "+++ b/code/main.c\n"
    "@@ -1 +1 @@\n"
    "-int main(void) { return 7; }\n"
    "+int main(void) { return 42; }\n"
  )
  (fixture.patches / "ret42.patch").write_text(patch_body)
  (fixture.patches / "series").write_text("src:ret42.patch\n")
  nested_out = fixture.super / "out/work"
  nested_out.parent.mkdir(parents=True, exist_ok=True)
  proc = fixture.materialize(nested_out, lock)
  assert proc.returncode == 0, proc.stderr
  assert "return 42;" in (nested_out / "src/code/main.c").read_text()
