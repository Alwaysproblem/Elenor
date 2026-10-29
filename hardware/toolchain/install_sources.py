"""Docker-build-only source acquisition and PIC SoftFloat construction."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import pathlib
import shutil
import subprocess
import tarfile

from bootstrap_envs import download, run

ROOT = pathlib.Path("/opt")
CC = "/opt/native/bin/x86_64-conda-linux-gnu-gcc"
AR = "/opt/native/bin/x86_64-conda-linux-gnu-ar"


def checkout(source: dict) -> None:
  path = pathlib.Path(source["path"])
  path.mkdir(parents=True, exist_ok=True)
  run("git", "init", str(path))
  run(
    "git",
    "-C",
    str(path),
    "-c",
    "http.version=HTTP/1.1",
    "fetch",
    "--depth=1",
    source["url"],
    source["commit"],
  )
  run("git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD")
  actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
  if actual != source["commit"]:
    raise ValueError(f"source commit mismatch at {path}")


def check_gitlink(parent: pathlib.Path, relative: str, expected: str) -> None:
  entry = subprocess.check_output(["git", "-C", str(parent), "ls-tree", "HEAD", relative], text=True)
  if entry.split()[:3] != ["160000", "commit", expected]:
    raise ValueError(f"pinned parent/submodule mismatch: {relative}: {entry}")


def main() -> None:
  lock = json.loads((ROOT / "sources.json").read_text())
  sources = lock["sources"]
  checkout(sources["chipyard"])
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(checkout, [source for name, source in sources.items() if name != "chipyard"]))
  chipyard = pathlib.Path(sources["chipyard"]["path"])
  for name, relative in {
    "rocket": "generators/rocket-chip",
    "gemmini": "generators/gemmini",
    "hardfloat": "generators/hardfloat",
    "diplomacy": "generators/diplomacy",
    "cde": "tools/cde",
    "firesim": "sims/firesim",
  }.items():
    check_gitlink(chipyard, relative, sources[name]["commit"])
  check_gitlink(
    pathlib.Path(sources["hardfloat"]["path"]), "berkeley-softfloat-3", sources["softfloat"]["commit"]
  )
  # Deliberately leave Chipyard's Gemmini and Rocket's duplicate nested dependencies uninitialized.
  if any((chipyard / "generators/gemmini").iterdir()):
    raise ValueError("duplicate Chipyard Gemmini source must not be initialized")

  firtool = lock["firtool"]
  archive = ROOT / "packages/firrtl-bin-linux-x64.tar.gz"
  downloads = [download(firtool["url"], archive, firtool["sha256"])]
  unpack = ROOT / "circt-unpack"
  with tarfile.open(archive) as tar:
    tar.extractall(unpack, filter="data")
  binaries = [p for p in unpack.rglob("firtool") if p.is_file()]
  if len(binaries) != 1:
    raise ValueError("firtool archive must contain exactly one firtool binary")
  shutil.copytree(binaries[0].parent.parent, ROOT / "circt")
  shutil.rmtree(unpack)
  launcher = lock["sbt_launcher"]
  downloads.append(download(launcher["url"], ROOT / "sbt-launch.jar", launcher["sha1"], "sha1"))

  source = pathlib.Path(sources["softfloat"]["path"])
  build = source / "build/Linux-x86_64-GCC"
  compile_c = (
    f"{CC} -c -fPIC -Werror-implicit-function-declaration -DSOFTFLOAT_FAST_INT64 "
    "-DSOFTFLOAT_ROUND_ODD -DINLINE_LEVEL=5 -DSOFTFLOAT_FAST_DIV32TO16 "
    "-DSOFTFLOAT_FAST_DIV64TO32 -I. -I../../source/RISCV -I../../source/include "
    "-O2 -ffile-prefix-map=/opt/deps/softfloat=. -o $@"
  )
  run(
    "make",
    "-C",
    str(build),
    "-j4",
    "SPECIALIZE_TYPE=RISCV",
    f"COMPILE_C={compile_c}",
    f"MAKELIB={AR} crs $@",
  )
  library = ROOT / "softfloat/lib/libsoftfloat.so"
  library.parent.mkdir(parents=True)
  run(
    CC,
    "-shared",
    "-Wl,--build-id=none",
    "-o",
    str(library),
    "-Wl,--whole-archive",
    str(build / "softfloat.a"),
    "-Wl,--no-whole-archive",
  )
  include = ROOT / "softfloat/include"
  shutil.copytree(source / "source/include", include)
  shutil.copy2(source / "source/RISCV/specialize.h", include / "specialize.h")
  shutil.copy2(build / "platform.h", include / "platform.h")
  shutil.copy2(source / "COPYING.txt", ROOT / "softfloat/COPYING.txt")
  shutil.copytree(source / "doc", ROOT / "softfloat/doc")
  (ROOT / "source-build.json").write_text(
    json.dumps(
      {
        "sources": sources,
        "downloads": downloads,
        "softfloat": {
          "compile_c": compile_c,
          "specialization": "RISCV",
          "thread_local": False,
          "library": str(library),
          "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        },
      },
      indent=2,
    )
    + "\n"
  )


if __name__ == "__main__":
  main()
