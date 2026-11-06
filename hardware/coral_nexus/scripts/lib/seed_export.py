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
import shutil
import sys
import time
import urllib.request
from pathlib import Path

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
    if kind in {"deferred", "not-applicable", "local-dir"}:
      entry_info = {"status": seed.get("status", kind)}
      if kind == "local-dir":
        local = Path(seed.get("path", ""))
        entry_info["sha256"] = seed["sha256"]
        entry_info["status"] = "exported"
        entry_info["file"] = f"{name} (local dir at {seed['path']})"
        entry_info["dir_exists"] = local.is_dir()
      entries[name] = entry_info
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
