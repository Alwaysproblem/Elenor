"""Seal observed image inputs after the pinned Scala graph has compiled."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/opt")


def digest(path: Path) -> str:
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(directory: Path, *, skip_git: bool = False) -> list[dict]:
  records: list[dict] = []
  for path in sorted(directory.rglob("*")):
    relative = path.relative_to(directory)
    if skip_git and ".git" in relative.parts:
      continue
    if path.is_symlink():
      target = os.readlink(path)
      records.append(
        {"path": str(relative), "symlink": target, "sha256": hashlib.sha256(target.encode()).hexdigest()}
      )
    elif path.is_file() and not path.name.endswith((".lock", ".lck")):
      records.append(
        {
          "path": str(relative),
          "size": path.stat().st_size,
          "sha256": digest(path),
          "executable": bool(path.stat().st_mode & 0o111),
        }
      )
  return records


def version(command: list[str], *, env: dict | None = None) -> dict:
  result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
  return {"command": command, "output": (result.stdout + result.stderr).strip()}


def main() -> None:
  env = dict(os.environ, NEXUS_GEMMINI="/opt/gemmini/upstream", SBT_OFFLINE="true")
  tools = {
    "java": version(["/opt/native/bin/java", "-version"]),
    "gcc": version([os.environ["CC"], "--version"]),
    "gxx": version([os.environ["CXX"], "--version"]),
    "verilator": version(["verilator", "--version"]),
    "riscv_gcc": version(["riscv64-unknown-elf-gcc", "--version"]),
    "firtool": version([os.environ["FIRTOOL_BIN"], "--version"]),
    "python": version(["/opt/validator/bin/python", "--version"]),
    "sbt_scala": version(["/opt/bin/sbt", "show sbtVersion", "p02/scalaVersion"], env=env),
    "softfloat_abi": version(["/opt/softfloat/bin/softfloat_abi_probe"]),
  }
  environments = {}
  for name in ("native", "validator", "bootstrap"):
    environments[name] = json.loads(
      subprocess.check_output(
        ["/opt/bootstrap/bin/conda", "list", "-p", f"/opt/{name}", "--json"], text=True
      )
    )
  sources = json.loads((ROOT / "sources.json").read_text())
  native_lock = ROOT / "locks/chipyard-native.lock.yml.txt"
  if digest(native_lock) != sources["native_lock"]["sha256"]:
    raise ValueError("native package lock differs from pinned input")
  bundle = ROOT / "sbt-bundle"
  for directory in ("boot", "coursier", "ivy"):
    (bundle / directory).mkdir(parents=True, exist_ok=True)
  dependencies = {
    "environments": environments,
    "package_archives": json.loads((ROOT / "environment-downloads.json").read_text()),
    "sbt_bundle": inventory(bundle),
    "sbt_launcher_sha256": digest(ROOT / "sbt-launch.jar"),
    "native_upstream_lock_sha256": digest(native_lock),
    "softfloat_abi_probe_sha256": digest(ROOT / "softfloat/bin/softfloat_abi_probe"),
  }
  gemmini_tree = inventory(ROOT / "gemmini/upstream", skip_git=True)
  gemmini = {
    "schema": 1,
    "source": sources["sources"]["gemmini"],
    "tree_sha256": hashlib.sha256(json.dumps(gemmini_tree, sort_keys=True).encode()).hexdigest(),
    "files": gemmini_tree,
  }
  (ROOT / "gemmini-upstream.lock").write_text(json.dumps(gemmini, indent=2) + "\n")
  build = {
    "schema": 1,
    "sources": sources,
    "tools": tools,
    "dependencies": dependencies,
    "source_build": json.loads((ROOT / "source-build.json").read_text()),
    "softfloat_headers": inventory(ROOT / "softfloat/include"),
    "gemmini_tree_sha256": gemmini["tree_sha256"],
    "validation": {"scala_graph_compiled": True, "runtime_smoke": "required separately"},
  }
  (ROOT / "toolchain-build.json").write_text(json.dumps(build, indent=2) + "\n")
  print(
    json.dumps(
      {
        "tools": tools,
        "sbt_dependency_files": len(dependencies["sbt_bundle"]),
        "gemmini_files": len(gemmini_tree),
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
