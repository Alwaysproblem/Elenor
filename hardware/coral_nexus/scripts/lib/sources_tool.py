"""Source-lock tooling for the Coral Nexus hardware workspace.

Subcommands
-----------
archive-digest
    Print the sha256 of ``git archive --format=tar <commit>`` for one source.
materialize
    Build an isolated derived work root (``out/coral_nexus/work/<name>``) from
    the pinned commits, overlay entries and patch series. Refuses to reuse a
    non-empty output directory and applies patches strictly (``git apply
    --check`` first, no fuzzy matching).
verify
    Re-check every lock invariant (gitlink commit, pristine submodule, archive
    digest, embedded pins, overlay declarations, patch/owner consistency) and
    write a JSON report. Exits non-zero on the first failed check group.

All paths inside ``sources.lock`` are relative to the repository root, which is
located by walking up from the lock file until a ``.git`` entry is found. The
tool never writes into the pristine submodules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import NoReturn, cast

LOCK_SCHEMA = "cn_sources_lock/v1"
OVERLAY_SCHEMA = "cn_overlay/v1"
OWNERS_SCHEMA = "cn_patch_owners/v1"
MANIFEST_SCHEMA = "cn_materialize_manifest/v1"
REPORT_SCHEMA = "cn_sources_report/v1"

DIGEST_EXEC = ("git",)  # indirection for tests that want to stub git


def _fail(message: str) -> NoReturn:
  print(f"error: {message}", file=sys.stderr)
  raise SystemExit(1)


def run_git(repo: Path, *args: str, check: bool = True) -> str:
  proc = subprocess.run(
    [*DIGEST_EXEC, "-C", str(repo), *args],
    capture_output=True,
    text=True,
  )
  if check and proc.returncode != 0:
    _fail(f"git {' '.join(args)} in {repo} failed: {proc.stderr.strip()}")
  return proc.stdout


def repo_root(start: Path) -> Path:
  cur = start.resolve()
  while True:
    if (cur / ".git").exists():
      return cur
    if cur.parent == cur:
      _fail(f"no repository root found above {start}")
    cur = cur.parent


def load_lock(lock_path: Path) -> dict:
  try:
    lock = json.loads(lock_path.read_text())
  except (OSError, json.JSONDecodeError) as exc:
    _fail(f"cannot read lock {lock_path}: {exc}")
  if lock.get("schema") != LOCK_SCHEMA:
    _fail(f"lock {lock_path} schema {lock.get('schema')!r} != {LOCK_SCHEMA!r}")
  if not lock.get("sources"):
    _fail(f"lock {lock_path} declares no sources")
  return lock


def load_yaml_limited(path: Path) -> dict:
  """Parse the small YAML subset used by overlay/owner files (stdlib only).

  Supports nested mappings, block lists of mappings or scalars, and
  flow-style string lists. The workspace writes these files itself, so a
  full YAML parser is not a dependency.
  """
  try:
    raw_lines = path.read_text().splitlines()
  except OSError as exc:
    _fail(f"cannot read {path}: {exc}")
  lines = []
  for lineno, raw in enumerate(raw_lines, 1):
    if "\t" in raw:
      _fail(f"{path}:{lineno} uses tabs")
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
      continue
    indent = len(raw) - len(raw.lstrip())
    lines.append((lineno, indent, stripped))

  def scalar(value: str, lineno: int) -> object:
    if value.startswith("[") and value.endswith("]"):
      inner = value[1:-1].strip()
      items = [v.strip().strip("'\"") for v in inner.split(",")] if inner else []
      if any(not v for v in items):
        _fail(f"{path}:{lineno} empty list item")
      return items
    return value.strip("'\"")

  def parse_block(idx: int, indent: int) -> tuple[object, int]:
    if lines[idx][1] != indent:
      _fail(f"{path}:{lines[idx][0]} bad indentation")
    if lines[idx][2].startswith("- "):
      items: list[object] = []
      while (
        idx < len(lines)
        and lines[idx][1] == indent
        and lines[idx][2].startswith("- ")
      ):
        lineno, _, body = lines[idx]
        rest = body[2:].strip()
        if not rest:
          idx += 1
          if idx >= len(lines):
            _fail(f"{path}:{lineno} list item without value")
          child, idx = parse_block(idx, lines[idx][1])
          items.append(child)
          continue
        if ":" not in rest:
          items.append(scalar(rest, lineno))
          idx += 1
          continue
        # Distinguish scalar-with-colon (e.g. bazel labels //path:target) from
        # mapping items: if the part before the first ':' contains '/' or
        # starts with '//', it is not a YAML key, so treat the whole rest as a
        # scalar string.
        pre_colon = rest.partition(":")[0].strip()
        if "/" in pre_colon or pre_colon.startswith("//"):
          items.append(scalar(rest, lineno))
          idx += 1
          continue
        key, sep, value = rest.partition(":")
        if not sep or not key.strip():
          _fail(f"{path}:{lineno} malformed mapping item")
        item: dict = {}
        items.append(item)
        idx += 1
        item_indent = indent + 2
        if value.strip():
          item[key.strip()] = scalar(value.strip(), lineno)
        else:
          if idx >= len(lines) or lines[idx][1] <= indent:
            _fail(f"{path}:{lineno} missing nested value for {key!r}")
          child, idx = parse_block(idx, lines[idx][1])
          item[key.strip()] = child
        while (
          idx < len(lines)
          and lines[idx][1] == item_indent
          and not lines[idx][2].startswith("- ")
        ):
          lineno2, _, body2 = lines[idx]
          key2, sep2, value2 = body2.partition(":")
          if not sep2:
            _fail(f"{path}:{lineno2} malformed mapping entry {body2!r}")
          if value2.strip():
            item[key2.strip()] = scalar(value2.strip(), lineno2)
            idx += 1
          else:
            idx += 1
            if idx >= len(lines) or lines[idx][1] <= item_indent:
              _fail(f"{path}:{lineno2} missing nested value for {key2!r}")
            child2, idx = parse_block(idx, lines[idx][1])
            item[key2.strip()] = child2
      return items, idx
    mapping: dict = {}
    while (
      idx < len(lines)
      and lines[idx][1] == indent
      and not lines[idx][2].startswith("- ")
    ):
      lineno, _, body = lines[idx]
      key, sep, value = body.partition(":")
      if not sep or not key.strip():
        _fail(f"{path}:{lineno} malformed mapping entry {body!r}")
      idx += 1
      if value.strip():
        mapping[key.strip()] = scalar(value.strip(), lineno)
      else:
        if idx >= len(lines) or lines[idx][1] <= indent:
          _fail(f"{path}:{lineno} missing nested value for {key!r}")
        child, idx = parse_block(idx, lines[idx][1])
        mapping[key.strip()] = child
    return mapping, idx

  if not lines:
    return {}
  result, next_idx = parse_block(0, lines[0][1])
  if next_idx != len(lines):
    _fail(f"{path}:{lines[next_idx][0]} unexpected trailing content")
  return cast(dict, result)


def archive_digest(repo: Path, commit: str) -> str:
  proc = subprocess.run(
    [*DIGEST_EXEC, "-C", str(repo), "archive", "--format=tar", commit],
    capture_output=True,
    check=False,
  )
  if proc.returncode != 0:
    _fail(
      f"git archive {commit} in {repo} failed: "
      f"{proc.stderr.decode(errors='replace').strip()}"
    )
  return hashlib.sha256(proc.stdout).hexdigest()


def tree_digest(root: Path) -> str:
  """Deterministic content digest over every regular file in ``root``."""
  entries = []
  for path in sorted(root.rglob("*")):
    if path.is_file() and not path.is_symlink():
      rel = path.relative_to(root).as_posix()
      entries.append((rel, hashlib.sha256(path.read_bytes()).hexdigest()))
    elif path.is_symlink():
      rel = path.relative_to(root).as_posix()
      entries.append((rel, "symlink:" + str(os.readlink(path))))
  digest = hashlib.sha256()
  for rel, file_hash in entries:
    digest.update(rel.encode())
    digest.update(b"\0")
    digest.update(file_hash.encode())
    digest.update(b"\0")
  return digest.hexdigest()


def parse_series(series_path: Path) -> list[tuple[str, str]]:
  """Return [(source_name, patch_relative_path)] from the series file."""
  entries: list[tuple[str, str]] = []
  try:
    lines = series_path.read_text().splitlines()
  except OSError as exc:
    _fail(f"cannot read patch series {series_path}: {exc}")
  for lineno, raw in enumerate(lines, 1):
    line = raw.strip()
    if not line or line.startswith("#"):
      continue
    source, sep, patch = line.partition(":")
    if not sep or not source or not patch or ("/" in patch and patch.startswith("/")):
      _fail(f"{series_path}:{lineno} malformed entry {line!r}")
    if patch.startswith("/") or ".." in patch.split("/"):
      _fail(f"{series_path}:{lineno} patch path {patch!r} escapes patch root")
    entries.append((source, patch))
  return entries


def patch_touched_files(patch_path: Path) -> list[str]:
  """Extract the upstream paths a patch modifies.

  Accepts both ``diff --git a/X b/X`` headers and plain unified diffs
  (``--- a/X`` followed by ``+++ b/X``). Renames and creations are rejected:
  the series only supports in-place modifications of upstream files.
  """
  files: list[str] = []
  git_header = re.compile(r"^diff --git a/(.+?) b/(.+)$")
  minus_header = re.compile(r"^--- a/(.+)$")
  plus_header = re.compile(r"^\+\+\+ b/(.+)$")
  try:
    lines = patch_path.read_text().splitlines()
  except OSError as exc:
    _fail(f"cannot read patch {patch_path}: {exc}")
  pending_minus: str | None = None
  for line in lines:
    git_match = git_header.match(line)
    if git_match:
      a, b = git_match.group(1), git_match.group(2)
      if a != b:
        _fail(f"{patch_path}: rename patches are not supported ({a} -> {b})")
      files.append(a)
      pending_minus = None
      continue
    minus_match = minus_header.match(line)
    if minus_match:
      pending_minus = minus_match.group(1)
      continue
    plus_match = plus_header.match(line)
    if plus_match and pending_minus is not None:
      a, b = pending_minus, plus_match.group(1)
      if a != b:
        _fail(f"{patch_path}: rename patches are not supported ({a} -> {b})")
      if a not in files:
        files.append(a)
      pending_minus = None
  if not files:
    _fail(f"{patch_path}: no diff headers found")
  return files


def glob_match(pattern: str, path: str) -> bool:
  """fnmatch-style glob with ``**`` support, anchored at the left."""
  import fnmatch

  if pattern.startswith("/"):
    pattern = pattern[1:]
  if "**" in pattern:
    prefix = pattern.split("**")[0]
    rest = pattern.split("**", 1)[1].lstrip("/")
    if not path.startswith(prefix):
      return False
    if not rest:
      return True
    return any(
      fnmatch.fnmatch(part, rest) for part in _suffixes(path[len(prefix) :])
    )
  return fnmatch.fnmatch(path, pattern)


def _suffixes(rel: str) -> list[str]:
  parts = rel.split("/")
  return ["/".join(parts[i:]) for i in range(len(parts) + 1)]


def resolve_owners(touched: dict[str, set[str]], owners: dict) -> list[str]:
  """Map each touched upstream file to owner names; report conflicts."""
  errors: list[str] = []
  file_owners: dict[str, set[str]] = {}
  for patch, files in sorted(touched.items()):
    for path in files:
      matched = {
        owner
        for owner, patterns in owners.items()
        if any(glob_match(pattern, path) for pattern in patterns or [])
      }
      if not matched:
        errors.append(f"{patch}: {path} has no owning package")
        continue
      file_owners.setdefault(path, set()).update(matched)
  for path, owners_hit in sorted(file_owners.items()):
    if len(owners_hit) > 1:
      errors.append(f"{path} claimed by multiple owners: {sorted(owners_hit)}")
  return errors


def cmd_archive_digest(args: argparse.Namespace) -> None:
  repo = Path(args.repo)
  print(archive_digest(repo, args.commit))


def _extract_source(source: dict, root: Path, work: Path) -> str:
  name = source["name"]
  repo = root / source["path"]
  commit = source["commit"]
  digest = archive_digest(repo, commit)
  locked = source.get("archive_sha256")
  if locked and digest != locked:
    _fail(
      f"source {name}: archive digest {digest} != locked {locked}; "
      "upstream content drifted from the lock"
    )
  dest = work / name
  dest.mkdir(parents=True)
  payload = subprocess.run(
    [*DIGEST_EXEC, "-C", str(repo), "archive", "--format=tar", commit],
    capture_output=True,
    check=False,
  )
  if payload.returncode != 0:
    _fail(f"git archive failed for {name}")
  with tempfile.NamedTemporaryFile(suffix=".tar") as tmp:
    tmp.write(payload.stdout)
    tmp.flush()
    with tarfile.open(tmp.name) as tar:
      try:
        tar.extractall(dest, filter="data")
      except TypeError:  # python < 3.12 compatibility
        dest_rel = dest.resolve()
        for member in tar.getmembers():
          target = (dest / member.name).resolve()
          if not str(target).startswith(str(dest_rel)):
            _fail(f"archive for {name} contains unsafe path {member.name}")
        tar.extractall(dest)
  return digest


def _apply_overlay(lock: dict, root: Path, overlay_path: Path, work: Path) -> list[str]:
  if not overlay_path.exists():
    return []
  overlay = load_yaml_limited(overlay_path)
  if overlay.get("schema") != OVERLAY_SCHEMA:
    _fail(f"{overlay_path} schema {overlay.get('schema')!r} != {OVERLAY_SCHEMA!r}")
  applied = []
  for entry in overlay.get("entries") or []:
    source = entry.get("source")
    src = root / entry["src"]
    dst_rel = entry["dst"]
    mode = entry.get("mode", "add")
    if mode not in {"add", "overwrite"}:
      _fail(f"overlay entry {dst_rel}: unknown mode {mode!r}")
    if ".." in dst_rel.split("/") or dst_rel.startswith("/"):
      _fail(f"overlay entry {dst_rel}: escapes work root")
    dst = work / source / dst_rel
    if dst.exists():
      if mode != "overwrite":
        _fail(
          f"overlay entry {dst_rel}: target exists in upstream tree and mode "
          "is not 'overwrite'; declare the overwrite explicitly"
        )
    elif mode == "overwrite":
      _fail(
        f"overlay entry {dst_rel}: mode 'overwrite' but target missing in "
        "upstream tree; use mode 'add'"
      )
    if not src.is_file():
      _fail(f"overlay entry {dst_rel}: source file {src} not found")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    applied.append(f"{source}:{dst_rel}")
  return applied


def _apply_patches(lock: dict, root: Path, work: Path) -> list[str]:
  patches_cfg = lock.get("patches") or {}
  series_rel = patches_cfg.get("series")
  if not series_rel:
    return []
  series = root / series_rel
  patch_root = Path(series).parent
  applied = []
  for source, patch_rel in parse_series(series):
    patch_file = patch_root / patch_rel
    if not patch_file.is_file():
      _fail(f"series entry {source}:{patch_rel}: patch file missing")
    target = work / source
    if not target.is_dir():
      _fail(f"series entry {source}:{patch_rel}: source {source} not materialized")
    # Isolate `git apply` from any enclosing superproject `.git` (e.g. when
    # `work` lives under `out/coral_nexus/...` inside the main repo), otherwise
    # `git apply` resolves `a/...` against the outer repo root and silently
    # prints `Skipped patch ...` with exit code 0.
    git_env = {
      **os.environ,
      "GIT_CEILING_DIRECTORIES": str(target.resolve().parent),
    }
    check = subprocess.run(
      ["git", "apply", "--check", "--verbose", str(patch_file.absolute())],
      cwd=target,
      env=git_env,
      capture_output=True,
      text=True,
    )
    if check.returncode != 0 or "Skipped patch" in check.stderr:
      _fail(
        f"patch {patch_rel} does not apply cleanly to {source}: "
        f"{check.stderr.strip()}"
      )
    apply = subprocess.run(
      ["git", "apply", "--whitespace=nowarn", "--verbose", str(patch_file.absolute())],
      cwd=target,
      env=git_env,
      capture_output=True,
      text=True,
    )
    if apply.returncode != 0 or "Skipped patch" in apply.stderr:
      _fail(f"patch {patch_rel} failed during apply: {apply.stderr.strip()}")
    applied.append(f"{source}:{patch_rel}")
  return applied


def cmd_materialize(args: argparse.Namespace) -> None:
  lock_path = Path(args.lock).resolve()
  lock = load_lock(lock_path)
  root = repo_root(lock_path.parent)
  work = Path(args.out).resolve()
  if work.exists() and any(work.iterdir()):
    _fail(f"work root {work} exists and is not empty; use a fresh attempt directory")
  work.mkdir(parents=True)
  overlay_path = root / lock["overlay"] if lock.get("overlay") else None
  sources = lock["sources"]
  if args.sources:
    wanted = set(args.sources.split(","))
    unknown = wanted - {s["name"] for s in sources}
    if unknown:
      _fail(f"unknown sources requested: {sorted(unknown)}")
    sources = [s for s in sources if s["name"] in wanted]
  digests = {}
  for source in sources:
    digests[source["name"]] = _extract_source(source, root, work)
  overlay_applied = _apply_overlay(lock, root, overlay_path, work) if overlay_path else []
  patches_applied = _apply_patches(lock, root, work)
  manifest = {
    "schema": MANIFEST_SCHEMA,
    "lock": str(lock_path.relative_to(root)),
    "sources": {
      name: {"archive_sha256": digest} for name, digest in sorted(digests.items())
    },
    "overlay_applied": sorted(overlay_applied),
    "patches_applied": sorted(patches_applied),
    "work_tree_digest": tree_digest(work),
  }
  manifest_path = work / "materialize_manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(json.dumps(manifest, indent=2, sort_keys=True))


def cmd_verify(args: argparse.Namespace) -> None:
  lock_path = Path(args.lock).resolve()
  lock = load_lock(lock_path)
  root = repo_root(lock_path.parent)
  report_dir = Path(args.out).resolve()
  report_dir.mkdir(parents=True, exist_ok=True)
  checks: list[dict] = []

  def record(name: str, ok: bool, detail: str) -> bool:
    checks.append({"check": name, "pass": ok, "detail": detail})
    return ok

  any_failed = False

  for source in lock["sources"]:
    name = source["name"]
    path = root / source["path"]
    if not path.is_dir():
      any_failed |= not record(f"{name}.present", False, f"{path} missing")
      continue
    any_failed |= not record(f"{name}.present", True, str(path))
    gitlink = run_git(root, "ls-files", "-s", "--", source["path"]).split()
    if not gitlink:
      any_failed |= not record(
        f"{name}.gitlink", False, "gitlink not recorded in the index"
      )
    else:
      mode, sha = gitlink[0], gitlink[1]
      ok = mode == "160000" and sha == source["commit"]
      any_failed |= not record(
        f"{name}.gitlink",
        ok,
        f"index {mode} {sha[:12]} vs locked {source['commit'][:12]}",
      )
    dirty = run_git(path, "status", "--porcelain", check=False).strip()
    any_failed |= not record(
      f"{name}.pristine", not dirty, dirty or "work tree clean"
    )
    digest = archive_digest(path, source["commit"])
    locked = source.get("archive_sha256")
    ok = bool(locked) and digest == locked
    any_failed |= not record(
      f"{name}.archive_sha256",
      ok,
      digest if ok else f"actual {digest} vs locked {locked}",
    )
    if not locked:
      checks[-1]["detail"] = f"lock missing archive_sha256 (actual {digest})"

  for pin in lock.get("embedded_pins") or []:
    holder = root / next(s["path"] for s in lock["sources"] if s["name"] == pin["holder"])
    entry = run_git(holder, "ls-tree", HEAD_REF, "--", pin["path"]).split()
    actual = entry[2] if len(entry) >= 3 else ""
    ok = actual == pin["commit"]
    any_failed |= not record(
      f"embedded.{pin['holder']}.{pin['path']}",
      ok,
      f"{actual[:12]} vs locked {pin['commit'][:12]} (policy {pin.get('policy')})",
    )

  overlay_rel = lock.get("overlay")
  if overlay_rel:
    overlay_path = root / overlay_rel
    if overlay_path.exists():
      overlay = load_yaml_limited(overlay_path)
      ok = overlay.get("schema") == OVERLAY_SCHEMA
      any_failed |= not record(
        "overlay.schema", ok, f"schema={overlay.get('schema')!r}"
      )
      for entry in overlay.get("entries") or []:
        src = root / entry["src"]
        if not src.is_file():
          any_failed |= not record(
            f"overlay.{entry['dst']}", False, f"source file {src} missing"
          )
        else:
          record(f"overlay.{entry['dst']}", True, str(src))
    else:
      any_failed |= not record("overlay.present", False, str(overlay_path))

  patches_cfg = lock.get("patches") or {}
  owners_rel = patches_cfg.get("owners")
  series_rel = patches_cfg.get("series")
  if series_rel:
    series_path = root / series_rel
    if not series_path.exists():
      any_failed |= not record("patches.series", False, str(series_path))
    else:
      entries = parse_series(series_path)
      record("patches.series", True, f"{len(entries)} patch entries")
      owners_data: dict = {}
      if owners_rel:
        owners_path = root / owners_rel
        if not owners_path.exists():
          any_failed |= not record("patches.owners", False, str(owners_path))
        else:
          owners_doc = load_yaml_limited(owners_path)
          if owners_doc.get("schema") != OWNERS_SCHEMA:
            any_failed |= not record(
              "patches.owners",
              False,
              f"schema={owners_doc.get('schema')!r}",
            )
          else:
            owners_data = owners_doc.get("owners") or {}
            record("patches.owners", True, f"{len(owners_data)} owners")
      touched: dict[str, set[str]] = {}
      for source, patch_rel in entries:
        patch_file = (root / series_rel).parent / patch_rel
        if not patch_file.is_file():
          any_failed |= not record(
            f"patches.{patch_rel}", False, "patch file missing"
          )
          continue
        record(f"patches.{patch_rel}", True, f"source={source}")
        touched[patch_rel] = set(patch_touched_files(patch_file))
      if owners_data:
        conflicts = resolve_owners(touched, owners_data)
        any_failed |= not record(
          "patches.ownership",
          not conflicts,
          "; ".join(conflicts) if conflicts else "single owner per file",
        )

  report = {
    "schema": REPORT_SCHEMA,
    "lock": str(lock_path.relative_to(root)),
    "checks": checks,
    "pass": not any_failed,
  }
  (report_dir / "sources_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(report, indent=2, sort_keys=True))
  if any_failed:
    raise SystemExit(1)


HEAD_REF = "HEAD"


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)

  digest = sub.add_parser("archive-digest", help="print archive sha256 for a commit")
  digest.add_argument("--repo", required=True)
  digest.add_argument("--commit", required=True)
  digest.set_defaults(func=cmd_archive_digest)

  materialize = sub.add_parser("materialize", help="build isolated work root")
  materialize.add_argument("--lock", required=True)
  materialize.add_argument("--out", required=True)
  materialize.add_argument(
    "--sources", help="comma-separated subset of source names"
  )
  materialize.set_defaults(func=cmd_materialize)

  verify = sub.add_parser("verify", help="verify lock invariants and report")
  verify.add_argument("--lock", required=True)
  verify.add_argument("--out", required=True)
  verify.set_defaults(func=cmd_verify)

  args = parser.parse_args(argv)
  args.func(args)


if __name__ == "__main__":
  main()
