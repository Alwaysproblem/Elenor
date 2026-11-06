"""Offline seed exporter (C004).

Reads env/seed_targets.json, fetches or copies every concrete seed into a
shared download cache (sha256-verified), then populates a fresh export
directory and writes export_manifest.json. Deferred/not-applicable entries
are recorded with their status instead of being silently skipped. Export
fails when any concrete seed is unreachable or fails its hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources_tool import tree_digest

MANIFEST_SCHEMA = "cn_seed_export/v1"


def sha256_of(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def fetch(url: str, dest: Path) -> None:
  tmp = dest.with_suffix(dest.suffix + ".part")
  with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
    shutil.copyfileobj(response, out, 1 << 20)
  tmp.replace(dest)


def archive_tree(src: Path, dest: Path) -> list[str]:
  """Copy src to dest preserving in-tree relative symlinks.

  Symlinks are what bazel's vendor layout uses for interpreters and tool
  aliases (for example ``bin/python3 -> python3.11``); dropping them turns a
  rebuildable archive into one that fails offline. Links whose target is
  absolute or escapes the tree are dropped (they point at a producer-local
  path) and reported, so the archive stays self-contained.
  """
  dropped: list[str] = []
  for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
    rel_dir = Path(dirpath).relative_to(src)
    (dest / rel_dir).mkdir(parents=True, exist_ok=True)
    for name in list(dirnames) + filenames:
      src_path = Path(dirpath) / name
      rel = (rel_dir / name).as_posix()
      if src_path.is_symlink():
        target = os.readlink(src_path)
        keep = (
          not os.path.isabs(target)
          and (src_path.parent / target).resolve().is_relative_to(src.resolve())
          and (src_path.parent / target).exists()
        )
        if keep:
          os.symlink(target, dest / rel)
        else:
          dropped.append(rel)
        if name in dirnames:
          dirnames.remove(name)
        continue
      if src_path.is_dir():
        continue
      (dest / rel).parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(src_path, dest / rel)
  return sorted(dropped)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seeds", required=True)
  parser.add_argument("--cache", required=True)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()

  seeds_path = Path(args.seeds).resolve()
  cache = Path(args.cache).resolve()
  out = Path(args.out).resolve()
  if out.exists() and any(out.iterdir()):
    print(f"error: export directory {out} exists and is not empty", file=sys.stderr)
    raise SystemExit(1)
  out.mkdir(parents=True)
  cache.mkdir(parents=True, exist_ok=True)

  doc = json.loads(seeds_path.read_text())
  entries: dict[str, dict] = {}
  failures: list[str] = []
  for seed in doc["seeds"]:
    name = seed["name"]
    kind = seed["kind"]
    if kind in {"deferred", "not-applicable"}:
      entries[name] = {"status": seed.get("status", kind), "kind": kind}
      continue
    if kind == "local-dir":
      # A directory seed is only archived when the tree is COPIED into the
      # export and the copy re-hashes to the declared digest. Recording
      # "exported" for an uncopied source (or a missing one) would make the
      # offline proof circular.
      src = Path(seed.get("path", ""))
      if not src.is_dir():
        failures.append(f"{name}: local-dir source missing: {src}")
        continue
      try:
        dest = out / name
        dropped = archive_tree(src, dest)
        actual = tree_digest(dest)
        if actual != seed["sha256"]:
          failures.append(
            f"{name}: archived tree digest {actual} != {seed['sha256']}"
          )
          continue
        entries[name] = {
          "status": "exported",
          "kind": kind,
          "sha256": actual,
          "file": f"{name}/",
          "bytes": sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()),
          "files": sum(1 for f in dest.rglob("*") if f.is_file()),
          "dropped_links": dropped,
        }
      except OSError as exc:
        failures.append(f"{name}: {exc}")
      continue
    expected = seed["sha256"]
    filename = name + "-" + seed["url"].rsplit("/", 1)[-1] if "url" in seed else name
    cached = cache / filename
    try:
      if not cached.is_file() or sha256_of(cached) != expected:
        if kind.startswith("url"):
          fetch(seed["url"], cached)
        elif kind == "repo-file":
          shutil.copy2(seeds_path.parents[3] / seed["path"], cached)
        else:
          raise ValueError(f"unknown kind {kind}")
      actual = sha256_of(cached)
      if actual != expected:
        failures.append(f"{name}: sha256 {actual} != {expected}")
        continue
      shutil.copy2(cached, out / filename)
      entries[name] = {"status": "exported", "sha256": actual, "file": filename}
    except Exception as exc:  # report every seed, then fail below
      failures.append(f"{name}: {exc}")

  manifest = {
    "schema": MANIFEST_SCHEMA,
    "seeds": entries,
    "failures": failures,
    "complete": not failures,
    "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  }
  (out / "export_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(manifest, indent=2, sort_keys=True))
  if failures:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
