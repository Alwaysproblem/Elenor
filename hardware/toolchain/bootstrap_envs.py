"""Build-only acquisition and offline reconstruction of the two pinned environments."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import urllib.parse
import urllib.request
from shutil import copyfile

ROOT = pathlib.Path("/opt")
CONDA = ROOT / "bootstrap/bin/conda"


def download(url: str, target: pathlib.Path, expected: str, algorithm: str = "sha256") -> dict:
  target.parent.mkdir(parents=True, exist_ok=True)
  digest = hashlib.new(algorithm)
  partial = target.with_suffix(target.suffix + ".partial")
  seed = ROOT / "package-seed" / target.name
  cache = ROOT / "acquisition-cache" / f"{expected}-{target.name}"
  if not seed.is_file() and cache.is_file():
    seed = cache
  transport = "verified-seed" if seed.is_file() else "download"
  print(f"Acquiring ({transport}) {url}", flush=True)
  received = 0
  try:
    source = seed.open("rb") if seed.is_file() else urllib.request.urlopen(url, timeout=300)
    with source as response, partial.open("wb") as output:
      while chunk := response.read1(1024 * 1024):
        digest.update(chunk)
        output.write(chunk)
        received += len(chunk)
  except OSError as exc:
    raise RuntimeError(f"package download failed after {received} bytes: {url}: {exc}") from exc
  if digest.hexdigest() != expected:
    partial.unlink()
    raise ValueError(f"{algorithm} mismatch for {url}")
  partial.replace(target)
  if not cache.is_file():
    cache.parent.mkdir(parents=True, exist_ok=True)
    cached_partial = cache.with_suffix(cache.suffix + ".partial")
    copyfile(target, cached_partial)
    cached_partial.replace(cache)
  with target.open("rb") as stream:
    sha256 = (
      digest.hexdigest() if algorithm == "sha256" else hashlib.file_digest(stream, "sha256").hexdigest()
    )
  print(f"Verified {target.name}: sha256={sha256}", flush=True)
  return {"url": url, "path": str(target), "sha256": sha256, "transport": transport}


def run(*args: str, **kwargs) -> None:
  print("+", " ".join(args), flush=True)
  subprocess.run(args, check=True, **kwargs)


def create_environment(name: str, explicit: pathlib.Path) -> list[dict]:
  records = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    jobs = []
    for line in explicit.read_text().splitlines():
      if not line or line.startswith(("#", "@")):
        continue
      url, separator, checksum = line.rpartition("#")
      if not separator or len(checksum) != 64:
        raise ValueError(f"explicit package lacks SHA-256: {line}")
      target = ROOT / "packages" / name / pathlib.PurePosixPath(urllib.parse.urlparse(url).path).name
      jobs.append(pool.submit(download, url, target, checksum))
    records = [job.result() for job in jobs]
  local = ROOT / "locks" / f"{name}.local.explicit.txt"
  local.write_text("@EXPLICIT\n" + "".join(f"file://{p['path']}#{p['sha256']}\n" for p in records))
  env = dict(os.environ, CONDA_PKGS_DIRS=str(ROOT / "conda-cache" / name))
  # All archive inputs are local and hash checked; the environment is new.
  run(
    str(CONDA),
    "create",
    "--yes",
    "--offline",
    "--no-default-packages",
    "--prefix",
    str(ROOT / name),
    "--file",
    str(local),
    env=env,
  )
  return records


def main() -> None:
  inventory = {
    "validator": create_environment("validator", ROOT / "locks/environment.lock.explicit.txt"),
    "native": create_environment("native", ROOT / "locks/native.explicit.txt"),
  }
  python = str(ROOT / "validator/bin/python")
  wheelhouse = ROOT / "packages/pip"
  wheelhouse.mkdir(parents=True)
  requirements = str(ROOT / "locks/environment.lock.pip.txt")
  run(
    python,
    "-m",
    "pip",
    "download",
    "--no-cache-dir",
    "--no-deps",
    "--dest",
    str(wheelhouse),
    "--require-hashes",
    "-r",
    requirements,
  )
  run(
    python,
    "-m",
    "pip",
    "install",
    "--no-index",
    "--no-deps",
    "--find-links",
    str(wheelhouse),
    "--require-hashes",
    "-r",
    requirements,
  )
  run(python, "-m", "pip", "check")
  inventory["pip"] = [
    {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in sorted(wheelhouse.glob("*.whl"))
  ]
  (ROOT / "environment-downloads.json").write_text(json.dumps(inventory, indent=2) + "\n")


if __name__ == "__main__":
  main()
