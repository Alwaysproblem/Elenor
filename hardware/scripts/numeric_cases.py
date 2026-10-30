"""P03 case adapters for the existing container runner; no numerical algorithms here."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

PROFILE = "bf16_fp32_mac_rne_v1"
VECTOR_FILES = ("primitives.json", "matmul.json", "epilogue.json", "conv.json")


def validate_numeric_config(value: dict) -> dict:
  allowed = {"profile", "tile_k", "random_cases"}
  if set(value) - allowed:
    raise ValueError(f"unknown numeric configuration keys: {sorted(set(value) - allowed)}")
  result = {"profile": PROFILE, "tile_k": 8, "random_cases": 1024, **value}
  if result["profile"] != PROFILE:
    raise ValueError(f"numeric.profile must be {PROFILE}")
  for name, minimum in (("tile_k", 1), ("random_cases", 0)):
    if type(result[name]) is not int or result[name] < minimum:
      raise ValueError(f"numeric.{name} must be an integer >= {minimum}")
  return result


def _sha(path: Path) -> str:
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def _record_tree(ctx, path: Path) -> None:
  for artifact in sorted(path.rglob("*")):
    if artifact.is_file() and artifact.suffix != ".log":
      ctx.record_artifact(artifact, deterministic=True)


def _cli(ctx) -> str:
  return str(ctx.repo / "hardware/boa/reference/cli")


def _run(ctx, argv: list[str], cwd: Path, name: str, *, env: dict | None = None) -> None:
  code = ctx.run(argv, cwd, name, env=env, timeout=10800)
  if not ctx.check(f"{name}-exit", code == 0, f"exit={code}; see command log"):
    raise RuntimeError(f"numeric prerequisite command failed: {name}")


def _ensure_probe(ctx) -> tuple[Path, Path]:
  if "numeric-probe" in ctx.tools:
    return Path(ctx.tools["numeric-probe"]), Path(ctx.tools["numeric-abi-json"])
  reference = ctx.repo / "hardware/boa/reference"
  work = ctx.work / "numeric-probe"
  overlay = work / "overlay"
  dependencies = work / "dependencies"
  work.mkdir(parents=True, exist_ok=False)
  shutil.copytree(
    ctx.repo / "hardware/toolchain/overlay",
    overlay,
    ignore=shutil.ignore_patterns("target", ".bloop", ".metals", ".bsp", "__pycache__"),
  )
  build = overlay / "build.sbt"
  build.write_text(build.read_text() + "\n" + (reference / "probe/numeric.sbt").read_text())
  for name in ("boot", "coursier", "ivy"):
    shutil.copytree(Path("/opt/sbt-bundle") / name, dependencies / name, symlinks=True)
  env = {
    "SBT_BOOT_DIRECTORY": str(dependencies / "boot"),
    "SBT_IVY_HOME": str(dependencies / "ivy"),
    "COURSIER_CACHE": str(dependencies / "coursier"),
    "NEXUS_DEPS": ctx.chipyard_root,
    "NEXUS_GEMMINI": ctx.gemmini_root,
    "NEXUS_NUMERIC_PROBE_SOURCE": str(reference / "probe"),
    "FIRTOOL_BIN": ctx.tools["firtool"],
  }
  sv = work / "NumericProbe.sv"
  _run(
    ctx,
    [ctx.tools["sbt"], f"numericProbe/runMain p03.Elaborate {sv}"],
    overlay,
    "numeric-elaborate",
    env=env,
  )
  obj = work / "obj_dir"
  _run(
    ctx,
    [
      ctx.tools["verilator"],
      "--cc",
      "--exe",
      "--build",
      "-j",
      "4",
      "--top-module",
      "NumericProbe",
      "--Mdir",
      str(obj),
      "-o",
      "numeric_probe",
      "--CFLAGS",
      "-O2",
      str(sv),
      str(reference / "probe/harness/numeric_probe.cpp"),
    ],
    work,
    "numeric-verilator-build",
  )
  compiler = os.environ["CC"]
  abi_object, abi_binary, abi_json = work / "abi.o", work / "abi_probe", work / "abi.json"
  _run(
    ctx,
    [
      compiler,
      "-std=c11",
      "-O2",
      "-I/opt/softfloat/include",
      "-c",
      str(reference / "probe/softfloat_abi.c"),
      "-o",
      str(abi_object),
    ],
    work,
    "abi-compile",
  )
  _run(ctx, [compiler, str(abi_object), "-o", str(abi_binary)], work, "abi-link")
  status, abi_text = ctx.run_capture([str(abi_binary)], work, "abi-run")
  if not ctx.check("native-abi-probe", status == 0, f"ABI probe exit={status}"):
    raise RuntimeError("native ABI probe failed")
  abi = json.loads(abi_text)
  abi_json.write_text(json.dumps(abi, indent=2, sort_keys=True) + "\n")
  executable = obj / "numeric_probe"
  artifacts = ctx.artifacts_dir() / "probe"
  artifacts.mkdir()
  for source, name in (
    (sv, "NumericProbe.sv"),
    (executable, "numeric_probe"),
    (abi_binary, "abi_probe"),
    (abi_json, "abi.json"),
  ):
    shutil.copy2(source, artifacts / name)
  source_hashes = {
    str(p.relative_to(reference)): _sha(p) for p in sorted((reference / "probe").rglob("*")) if p.is_file()
  }
  (artifacts / "build-inputs.json").write_text(
    json.dumps(
      {
        "sources": source_hashes,
        "softfloat_library_sha256": _sha(Path("/opt/softfloat/lib/libsoftfloat.so")),
        "native_abi": abi,
      },
      indent=2,
      sort_keys=True,
    )
    + "\n"
  )
  _record_tree(ctx, artifacts)
  # Exercise the exported binaries, not work/ copies deleted after PASS.
  executable, abi_json = artifacts / "numeric_probe", artifacts / "abi.json"
  ctx.tools["numeric-probe"] = str(executable)
  ctx.tools["numeric-abi-json"] = str(abi_json)
  return executable, abi_json


def case_selftest(ctx) -> bool:
  options = ctx.config["numeric"]
  out = ctx.artifacts_dir()
  _run(
    ctx,
    [
      sys.executable,
      _cli(ctx),
      "selftest",
      "--seed",
      str(ctx.request["seed"]),
      "--tile-k",
      str(options["tile_k"]),
      "--out",
      str(out),
    ],
    ctx.repo,
    "reference-selftest",
  )
  report = json.loads((out / "report.json").read_text())
  ok = ctx.check("reference-selftest-status", report["status"] == "PASS")
  for check in report["checks"]:
    ok &= ctx.check(check["id"], check["pass"], check["detail"])
  ctx.metrics["reference_checks"] = report["counts"]["checks"]
  _record_tree(ctx, out)
  return bool(ok)


def _crosscheck(ctx, specials_only: bool) -> bool:
  executable, abi = _ensure_probe(ctx)
  out = ctx.artifacts_dir() / "comparison"
  argv = [
    sys.executable,
    _cli(ctx),
    "crosscheck",
    "--seed",
    str(ctx.request["seed"]),
    "--random-cases",
    str(ctx.config["numeric"]["random_cases"]),
    "--probe",
    str(executable),
    "--abi-json",
    str(abi),
    "--out",
    str(out),
  ]
  if specials_only:
    argv.append("--specials-only")
  code = ctx.run(argv, ctx.repo, "three-way-crosscheck", timeout=10800)
  ok = ctx.check("three-way-process", code == 0, f"crosscheck exit={code}")
  report = json.loads((out / "report.json").read_text())
  ok &= ctx.check(
    "all-three-backends-match",
    report["status"] == "PASS" and not report["mismatches"],
    json.dumps(report["mismatches"][:1], sort_keys=True),
  )
  ok &= ctx.check("all-vectors-compared", report["counts"]["compared"] == report["counts"]["vectors"])
  ctx.metrics.update(report["counts"])
  _record_tree(ctx, out)
  return bool(ok)


def case_crosscheck(ctx) -> bool:
  return _crosscheck(ctx, False)


def case_probe_nan(ctx) -> bool:
  return _crosscheck(ctx, True)


def _generate(ctx, out: Path, seed: int, tile_k: int, label: str) -> None:
  for command in ("gen-matmul", "gen-epilogue", "gen-conv"):
    _run(
      ctx,
      [sys.executable, _cli(ctx), command, "--seed", str(seed), "--tile-k", str(tile_k), "--out", str(out)],
      ctx.repo,
      f"{label}-{command}",
    )


def case_vectors(ctx) -> bool:
  seed, tile_k = ctx.request["seed"], ctx.config["numeric"]["tile_k"]
  first, second = ctx.artifacts_dir() / "generated", ctx.work / "vectors-second"
  _generate(ctx, first, seed, tile_k, "first")
  _generate(ctx, second, seed, tile_k, "repeat")
  committed = ctx.repo / "hardware/boa/reference/vectors"
  baseline_metadata = json.loads((committed / "matmul.json").read_text())
  baseline_seed = baseline_metadata["seed"]
  baseline_k = baseline_metadata["parameters"]["tile_k"]
  baseline = first
  if (seed, tile_k) != (baseline_seed, baseline_k):
    baseline = ctx.work / "vectors-baseline"
    _generate(ctx, baseline, baseline_seed, baseline_k, "baseline")
  hashes = {}
  ok = True
  for name in VECTOR_FILES:
    one, two, old, regenerated = (_sha(path / name) for path in (first, second, committed, baseline))
    ok &= ctx.check(f"same-seed-{name}", one == two, f"{one} != {two}")
    ok &= ctx.check(
      f"committed-golden-{name}",
      old == regenerated,
      f"stale committed vector {name}: regenerate explicitly, not inside check",
    )
    hashes[name] = {"first": one, "repeat": two, "committed": old, "baseline": regenerated}
  (ctx.artifacts_dir() / "hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
  _record_tree(ctx, ctx.artifacts_dir())
  ctx.metrics["golden_documents"] = len(hashes)
  return bool(ok)
