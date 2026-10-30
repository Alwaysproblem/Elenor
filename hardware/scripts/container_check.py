#!/usr/bin/env python3
"""In-container case runner for registered hardware validation targets.

Executed by the host launchers (hardware/scripts/check, hardware/scripts/
reproduce) inside the locked toolchain image via
`--entrypoint /opt/validator/bin/python`. The repository is mounted read-only
at /workspace and the output directory writable at /out; the request
(config content, seed, case selection, expected lock/image identity) is
serialized in /out/request.json, never re-read from outside /out.

Fail closed: every case must pass explicit assertions on actual outputs
(byte hashes, parsed reports, ELF structure); PASS is never derived from
logs, version prints, or file existence alone. Exit codes:

  0  all requested cases passed
  6  request config rejected by schema/target/section validation
  7  at least one case failed (manifest/summary still written)
  8  environment or identity verification failed
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
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import numeric_cases
import numeric_profile_cases

EXIT_OK = 0
EXIT_CONFIG = 6
EXIT_CASES = 7
EXIT_ENV = 8

SCHEMA = 1
GOLDEN_C_SHA256 = "ebd1555712c10fb5636e5d21fe415d82e2ce4a46dd9a10901a9849669e7eaaea"

CONFIG_TOP_KEYS = {"schema", "target", "topology", "numeric", "validator", "board"}
VALIDATOR_KEYS = {"hw_overrides"}
OVERRIDE_KEY_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
OVERRIDE_VALUE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+-]*$")

OVERLAY_DIR = "hardware/toolchain/overlay"
SMOKE_DIR = "hardware/toolchain/smoke"
EXCLUDED_COPY_DIRS = {"target", ".bloop", ".metals", ".bsp", "__pycache__"}

PYTHON_ENV_FIXTURES = (
  "examples/workloads/boa_am_matmul.mlir",
  "examples/workloads/boa_am_inputs/boa_am_matmul/A.bin",
  "examples/workloads/boa_am_inputs/boa_am_matmul/B.bin",
)

REQUIRED_IMAGE_PATHS = (
  "/opt/bin/sbt",
  "/opt/sbt-bundle/boot",
  "/opt/sbt-bundle/coursier",
  "/opt/sbt-bundle/ivy",
  "/opt/softfloat/lib/libsoftfloat.so",
  "/opt/softfloat/include/softfloat.h",
  "/opt/toolchain-build.json",
)

# Timeout for one host-identity digest of the P01 byte-level replay
# (fixture-derived constant; see plan/evidence/P01/boa-am-matmul.bytes.json).
REPLAY_SCRIPT = r'''#!/usr/bin/env python3
"""P02 python-env byte-level replay.

Reloads the CLI-written compiled artifact on a fresh ByteStore with identical
bindings (exact P01 procedure, plan/evidence/P01/commands.json) and compares
the C output bytes against the P01 fixed-input golden SHA-256. The golden
constant comes from P01's independent integer BF16 oracle (exact 1/8-unit
decode, exact dot products); this script deliberately imports neither
boa_numeric nor any test oracle.
"""
import hashlib
import json
import pathlib
import sys

from pipeline_validator.compiled_program import parse_compiled_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.simulator import Simulator


def main() -> int:
  compiled_json, target_yaml, seed_text, out_dir, golden_sha = sys.argv[1:6]
  out = pathlib.Path(out_dir)
  inputs = pathlib.Path("examples/workloads/boa_am_inputs/boa_am_matmul")
  a_bytes = inputs.joinpath("A.bin").read_bytes()
  b_bytes = inputs.joinpath("B.bin").read_bytes()
  program = parse_compiled_program(pathlib.Path(compiled_json).read_text())
  hw = HardwareConfig.from_yaml(target_yaml)
  sim_cfg = SimConfig(
    fidelity="full_memory",
    context_count=1,
    device_context_count=1,
    max_cycles=200000,
    seed=int(seed_text),
  )
  bindings = {
    "A": GlobalBinding("A", 0x100000, len(a_bytes), "r"),
    "B": GlobalBinding("B", 0x200000, len(b_bytes), "r"),
    "C": GlobalBinding("C", 0x300000, 65536, "w"),
  }
  store = ByteStore()
  store.seed_hbm(0x100000, a_bytes)
  store.seed_hbm(0x200000, b_bytes)
  result = Simulator(hw, sim_cfg, byte_store=store).run(
    load_program(program, hw, sim_cfg, actual_bindings=bindings)
  )
  actual = store.read_hbm(0x300000, 65536)
  actual_sha = hashlib.sha256(actual).hexdigest()
  record = {
    "completed": result.completed,
    "cycles": result.cycles,
    "artifact_hash": program.artifact_hash,
    "fidelity": sim_cfg.fidelity,
    "seed": sim_cfg.seed,
    "compared_bytes": len(actual),
    "actual_sha256": actual_sha,
    "golden_sha256": golden_sha,
    "bytes_equal": actual_sha == golden_sha,
  }
  out.joinpath("replay.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  out.joinpath("c_output.bin").write_bytes(actual)
  print(json.dumps(record, sort_keys=True))
  return 0 if result.completed and actual_sha == golden_sha else 1


if __name__ == "__main__":
  raise SystemExit(main())
'''


def fail_env(message: str) -> NoReturn:
  print(f"environment-error: {message}", file=sys.stderr)
  raise SystemExit(EXIT_ENV)


def fail_config(message: str) -> NoReturn:
  print(f"config-error: {message}", file=sys.stderr)
  raise SystemExit(EXIT_CONFIG)


def sha256_path(path: Path) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class CaseContext:
  """Per-run state shared by all case runners."""

  def __init__(self, repo: Path, out: Path, request: dict, config: dict):
    self.repo = repo
    self.out = out
    self.request = request
    self.config = config
    self.work = out / "work"
    self.work.mkdir(parents=True, exist_ok=True)
    self.tools: dict[str, str] = {}
    self.chipyard_root = ""
    self.gemmini_root = ""
    self.case_id = ""
    self.layer = ""
    self.assertions: list[dict] = []
    self.commands: list[dict] = []
    self.artifacts: list[dict] = []
    self.metrics: dict = {}
    self.log_index = 0

  def begin_case(self, case_id: str, layer: str) -> None:
    self.case_id = case_id
    self.layer = layer
    self.assertions = []
    self.commands = []
    self.artifacts = []
    self.metrics = {}
    self.log_index = 0

  def log_dir(self) -> Path:
    path = self.out / "logs" / self.case_id
    path.mkdir(parents=True, exist_ok=True)
    return path

  def artifacts_dir(self, case_id: str | None = None) -> Path:
    path = self.out / "artifacts" / (case_id or self.case_id)
    path.mkdir(parents=True, exist_ok=True)
    return path

  def check(self, name: str, passed: bool, detail: str = "") -> bool:
    self.assertions.append({"name": name, "pass": bool(passed), "detail": detail})
    if not passed:
      print(f"case-failure: {self.case_id}: assertion '{name}' failed: {detail}", file=sys.stderr)
    return bool(passed)

  def run(self, argv: list[str], cwd: Path, name: str, env: dict | None = None, timeout: int = 600) -> int:
    code, _ = self._run(argv, cwd, name, env=env, timeout=timeout)
    return code

  def run_capture(self, argv: list[str], cwd: Path, name: str, timeout: int = 600) -> tuple[int, str]:
    return self._run(argv, cwd, name, timeout=timeout, capture=True)

  def _run(
    self,
    argv: list[str],
    cwd: Path,
    name: str,
    env: dict | None = None,
    timeout: int = 600,
    capture: bool = False,
  ) -> tuple[int, str]:
    self.log_index += 1
    log_rel = f"logs/{self.case_id}/{self.log_index:02d}-{name}.log"
    log_path = self.out / log_rel
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = dict(os.environ)
    deltas = {}
    for key, value in (env or {}).items():
      deltas[key] = value
      run_env[key] = value
    captured: list[str] = []
    log_handle = open(log_path, "w", encoding="utf-8")
    try:
      log_handle.write(f"# argv: {json.dumps(argv)}\n")
      log_handle.write(f"# cwd: {cwd}\n")
      log_handle.write(f"# env: {json.dumps(deltas, sort_keys=True)}\n\n")
      log_handle.flush()
      try:
        proc = subprocess.run(
          argv,
          cwd=str(cwd),
          env=run_env,
          stdin=subprocess.DEVNULL,
          stdout=subprocess.PIPE if capture else log_handle,
          stderr=subprocess.STDOUT,
          text=capture,
          timeout=timeout,
        )
        code = proc.returncode
        if capture and proc.stdout is not None:
          captured.append(proc.stdout)
          log_handle.write(proc.stdout)
      except subprocess.TimeoutExpired as exc:
        code = 124
        partial = (
          exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        )
        captured.append(f"# timed out after {timeout}s\n{partial or ''}")
        log_handle.write(f"{partial or ''}\n# timed out after {timeout}s\n")
      except OSError as exc:
        code = 127
        captured.append(f"# spawn failed: {exc}\n")
        log_handle.write(f"# spawn failed: {exc}\n")
      log_handle.write(f"\n# exit: {code}\n")
    finally:
      log_handle.close()
    self.commands.append({"argv": argv, "cwd": str(cwd), "exit": code, "log": log_rel, "env": deltas})
    return code, "".join(captured)

  def record_artifact(self, path: Path, deterministic: bool) -> None:
    self.artifacts.append(
      {
        "path": str(path.relative_to(self.out)),
        "sha256": sha256_path(path),
        "bytes": path.stat().st_size,
        "deterministic": bool(deterministic),
      }
    )

  def result(self) -> dict:
    failed_commands = [
      f"{command['log']} exit={command['exit']}" for command in self.commands if command["exit"] != 0
    ]
    if failed_commands:
      self.assertions.append(
        {"name": "commands-exit-zero", "pass": False, "detail": "; ".join(failed_commands)}
      )
    passed = bool(self.assertions) and all(item["pass"] for item in self.assertions)
    return {
      "layer": self.layer,
      "status": "PASS" if passed else "FAIL",
      "assertions": self.assertions,
      "commands": self.commands,
      "artifacts": self.artifacts,
      "metrics": self.metrics,
    }


def resolve_tool(name: str, fallback: str | None = None) -> str:
  env_value = os.environ.get(name.upper().replace("-", "_"))
  if env_value and Path(env_value).is_file():
    return env_value
  if fallback and Path(fallback).is_file():
    return fallback
  found = shutil.which(name)
  if found:
    return found
  fail_env(
    f"required tool '{name}' not found (PATH={os.environ.get('PATH', '')!r}"
    + (f", fallback {fallback} missing" if fallback else "")
    + ")"
  )
  raise AssertionError  # unreachable


def validate_config(text: str, registry_entry: dict, request: dict) -> dict:
  try:
    import yaml
  except ImportError as exc:  # pragma: no cover - PyYAML is part of the locked env
    fail_env(f"PyYAML unavailable in /opt/validator environment: {exc}")

  try:
    config = yaml.safe_load(text)
  except yaml.YAMLError as exc:
    fail_config(f"config YAML parse error: {exc}")
  if not isinstance(config, dict):
    fail_config("config must be a YAML mapping")

  unknown = sorted(set(config) - CONFIG_TOP_KEYS)
  if unknown:
    fail_config(f"unknown config keys {unknown}; allowed: {sorted(CONFIG_TOP_KEYS)}")
  if config.get("schema") != SCHEMA:
    fail_config(f"config schema must be {SCHEMA}, got {config.get('schema')!r}")

  target = config.get("target")
  required = registry_entry.get("config", {}).get("required_target")
  if target != request["target"] or target != required:
    fail_config(
      f"config target {target!r} does not match requested target "
      f"{request['target']!r} (registry requires {required!r})"
    )

  consumed = set(registry_entry.get("config", {}).get("consumed_sections", []))
  for section in ("topology", "numeric", "validator", "board"):
    if section not in config:
      continue
    value = config[section]
    if not isinstance(value, dict):
      fail_config(f"config section '{section}' must be a mapping")
    if section not in consumed and value != {}:
      fail_config(
        f"config section '{section}' is reserved and has no registered consumer "
        f"for target '{target}'; it must be empty or absent"
      )

  validator = config.get("validator", {})
  unknown_validator = sorted(set(validator) - VALIDATOR_KEYS)
  if unknown_validator:
    fail_config(f"unknown validator keys {unknown_validator}; allowed: {sorted(VALIDATOR_KEYS)}")
  overrides = validator.get("hw_overrides", {})
  if not isinstance(overrides, dict):
    fail_config("validator.hw_overrides must be a mapping")
  for key, value in overrides.items():
    if not isinstance(key, str) or not OVERRIDE_KEY_RE.match(key):
      fail_config(f"invalid hw_overrides key {key!r}")
    if isinstance(value, bool):
      continue
    if isinstance(value, (int, float)):
      continue
    if isinstance(value, str) and OVERRIDE_VALUE_RE.match(value):
      continue
    fail_config(f"invalid hw_overrides value for '{key}': {value!r}")
  if target == "numeric":
    try:
      config["numeric"] = numeric_cases.validate_numeric_config(config.get("numeric", {}))
    except ValueError as exc:
      fail_config(str(exc))
  if target == "numeric-profile" and overrides:
    fail_config(
      "numeric-profile golden vectors pin each case's boa_numeric_tile_k; "
      "validator.hw_overrides must be empty (nothing is silently discarded)"
    )
  return config


def override_value(value) -> str:
  if isinstance(value, bool):
    return "true" if value else "false"
  return str(value)


def case_elaborate(ctx: CaseContext) -> bool:
  """Layer R: clean sbt elaboration of the HardFloat FP32 MulAddRecFN smoke
  top plus a Verilator build/run of the C++ assertion harness."""
  ok = True
  overlay_src = ctx.repo / OVERLAY_DIR
  harness_src = ctx.repo / SMOKE_DIR / "hardfloat.cpp"
  if not (overlay_src / "build.sbt").is_file():
    return ctx.check("overlay-present", False, f"missing {overlay_src}/build.sbt")
  if not harness_src.is_file():
    return ctx.check("harness-present", False, f"missing {harness_src}")

  shutil.rmtree(ctx.work, ignore_errors=True)
  work_overlay = ctx.work / "overlay"
  dependencies = ctx.work / "dependencies"
  elaborate_dir = ctx.work / "elaborate"
  verilator_dir = ctx.work / "verilator"
  for directory in (dependencies, elaborate_dir, verilator_dir):
    directory.mkdir(parents=True)

  shutil.copytree(
    overlay_src, work_overlay, ignore=shutil.ignore_patterns(*EXCLUDED_COPY_DIRS), symlinks=True
  )
  for item in ("boot", "coursier", "ivy"):
    source = Path("/opt/sbt-bundle") / item
    if not source.is_dir():
      return ctx.check("sbt-bundle", False, f"missing image bundle {source}")
    shutil.copytree(source, dependencies / item, symlinks=True)

  sv_path = elaborate_dir / "HardFloatMulAddSmoke.sv"
  sbt_env = {
    "SBT_BOOT_DIRECTORY": str(dependencies / "boot"),
    "COURSIER_CACHE": str(dependencies / "coursier"),
    "SBT_IVY_HOME": str(dependencies / "ivy"),
    "FIRTOOL_BIN": ctx.tools["firtool"],
    "NEXUS_DEPS": ctx.chipyard_root,
    "NEXUS_GEMMINI": ctx.gemmini_root,
  }
  code = ctx.run(
    [ctx.tools["sbt"], f"p02/runMain p02.Elaborate {sv_path}"],
    work_overlay,
    "sbt-elaborate",
    env=sbt_env,
    timeout=10800,
  )
  ok &= ctx.check(
    "sbt-elaborate-exit",
    code == 0,
    f"sbt exit={code} (log logs/{ctx.case_id}/{ctx.log_index:02d}-sbt-elaborate.log)",
  )

  sv_text = ""
  if sv_path.is_file():
    sv_text = sv_path.read_text(encoding="utf-8", errors="replace")
  ok &= ctx.check(
    "sv-written", sv_path.is_file() and sv_path.stat().st_size > 0, f"{sv_path} missing or empty"
  )
  ok &= ctx.check(
    "sv-top-module",
    "module HardFloatMulAddSmoke" in sv_text,
    "elaborated SV must declare module HardFloatMulAddSmoke",
  )

  obj_dir = verilator_dir / "obj_dir"
  code = ctx.run(
    [
      ctx.tools["verilator"],
      "--cc",
      "--exe",
      "--build",
      "--top-module",
      "HardFloatMulAddSmoke",
      "--Mdir",
      str(obj_dir),
      "-o",
      "vsim",
      "--CFLAGS",
      "-O2",
      str(sv_path),
      str(harness_src),
    ],
    verilator_dir,
    "verilator-build",
    timeout=3600,
  )
  ok &= ctx.check("verilator-build-exit", code == 0, f"verilator exit={code}")

  vsim = obj_dir / "vsim"
  artifacts = ctx.artifacts_dir()
  actual_path = artifacts / "actual.json"
  golden_path = artifacts / "golden.json"
  if vsim.is_file():
    code = ctx.run([str(vsim), str(actual_path), str(golden_path)], verilator_dir, "vsim-run", timeout=600)
    ok &= ctx.check("vsim-exit", code == 0, f"simulation exit={code} (mismatch or crash)")
  else:
    ok &= ctx.check("vsim-binary", False, f"{vsim} was not produced")

  if actual_path.is_file() and golden_path.is_file():
    try:
      payload = json.loads(actual_path.read_text(encoding="utf-8"))
      golden = json.loads(golden_path.read_text(encoding="utf-8"))
      ok &= ctx.check(
        "actual-equals-golden",
        payload.get("vectors") == golden.get("vectors"),
        "measured input/output bits and exception flags must match golden vectors",
      )
      vectors = payload.get("vectors")
      ok &= ctx.check(
        "vectors-nonempty",
        isinstance(vectors, list) and len(vectors) > 0,
        f"vectors must be a non-empty list, got {type(vectors).__name__}",
      )
      if isinstance(vectors, list) and vectors:
        bad = [
          index
          for index, vector in enumerate(vectors)
          if not isinstance(vector, dict) or not {"a", "b", "c", "out", "flags"} <= set(vector)
        ]
        ok &= ctx.check("vector-fields", not bad, f"vectors missing a/b/c/out/flags at indices {bad[:5]}")
        ctx.metrics["vector_count"] = len(vectors)
    except (ValueError, UnicodeError) as exc:
      ok &= ctx.check("actual-json-parses", False, f"invalid JSON: {exc}")
  else:
    ok &= ctx.check("actual-golden-files", False, "harness JSON outputs missing")

  sv_artifact = artifacts / "HardFloatMulAddSmoke.sv"
  if sv_path.is_file():
    shutil.copyfile(sv_path, sv_artifact)
    ctx.record_artifact(sv_artifact, deterministic=True)
  for json_name in ("actual.json", "golden.json"):
    if (artifacts / json_name).is_file():
      ctx.record_artifact(artifacts / json_name, deterministic=True)
  if vsim.is_file():
    binary_artifact = artifacts / "vsim"
    shutil.copyfile(vsim, binary_artifact)
    ctx.record_artifact(binary_artifact, deterministic=True)
  return bool(ok)


def case_softfloat(ctx: CaseContext) -> bool:
  """Layer F: actual ctypes calls into the pinned SoftFloat 3e PIC shared
  library against known exact bit patterns (independent P01-style vectors)."""
  ok = True
  script = ctx.repo / SMOKE_DIR / "softfloat.py"
  if not script.is_file():
    return ctx.check("softfloat-script", False, f"missing {script}")
  library = Path("/opt/softfloat/lib/libsoftfloat.so")
  if not library.is_file():
    return ctx.check("softfloat-library", False, f"missing {library}")

  artifacts = ctx.artifacts_dir()
  code = ctx.run(
    [sys.executable, str(script), str(artifacts / "actual.json"), str(artifacts / "golden.json")],
    ctx.work,
    "softfloat-smoke",
    timeout=600,
  )
  ok &= ctx.check("softfloat-exit", code == 0, f"smoke exit={code} (mismatch or load failure)")

  actual_path = artifacts / "actual.json"
  golden_path = artifacts / "golden.json"
  if actual_path.is_file() and golden_path.is_file():
    try:
      payload = json.loads(actual_path.read_text(encoding="utf-8"))
      golden = json.loads(golden_path.read_text(encoding="utf-8"))
      ok &= ctx.check(
        "actual-equals-golden",
        payload.get("vectors") == golden.get("vectors"),
        "measured input/output bits and exception flags must match golden vectors",
      )
      vectors = payload.get("vectors")
      ok &= ctx.check(
        "vectors-nonempty",
        isinstance(vectors, list) and len(vectors) > 0,
        "SoftFloat vectors must be a non-empty list",
      )
      if isinstance(vectors, list) and vectors:
        bad = [
          index
          for index, vector in enumerate(vectors)
          if not isinstance(vector, dict) or not {"a", "b", "c", "out", "flags"} <= set(vector)
        ]
        ok &= ctx.check("vector-fields", not bad, f"vectors missing a/b/c/out/flags at indices {bad[:5]}")
        ctx.metrics["vector_count"] = len(vectors)
    except (ValueError, UnicodeError) as exc:
      ok &= ctx.check("actual-json-parses", False, f"invalid JSON: {exc}")
    for json_name in ("actual.json", "golden.json"):
      ctx.record_artifact(artifacts / json_name, deterministic=True)
  else:
    ok &= ctx.check("actual-golden-files", False, "SoftFloat JSON outputs missing")
  return bool(ok)


def readelf_field(text: str, field: str) -> str | None:
  match = re.search(rf"^\s*{re.escape(field)}:\s*(.+?)\s*$", text, re.M)
  return match.group(1) if match else None


def case_riscv(ctx: CaseContext) -> bool:
  """Layer F/tooling: compile and link rv64gc_zicsr_zifencei/lp64d, then
  inspect ELF/ISA/disassembly. This does not execute a CPU or RTL backend."""
  ok = True
  smoke_dir = ctx.repo / SMOKE_DIR
  source = smoke_dir / "riscv.S"
  linker = smoke_dir / "riscv.ld"
  for path in (source, linker):
    if not path.is_file():
      return ctx.check(f"smoke-input-{path.name}", False, f"missing {path}")

  work = ctx.work / "riscv"
  shutil.rmtree(work, ignore_errors=True)
  work.mkdir(parents=True)
  elf_path = work / "smoke.elf"
  object_path = work / "smoke.o"

  gcc = ctx.tools["riscv-gcc"]
  target_flags = ["-march=rv64gc_zicsr_zifencei", "-mabi=lp64d"]
  compile_code = ctx.run(
    [gcc, *target_flags, "-c", str(source), "-o", str(object_path)], work, "riscv-assemble", timeout=600
  )
  if not ctx.check("riscv-assemble-exit", compile_code == 0, f"assembler exit={compile_code}"):
    return False
  # A direct .S -> ELF invocation gives the linker a random ccXXXX.o FILE
  # symbol. Keep the intermediate object name stable instead of stripping it.
  code = ctx.run(
    [gcc, *target_flags, "-nostdlib", "-T", str(linker), str(object_path), "-o", str(elf_path)],
    work,
    "riscv-link",
    timeout=600,
  )
  ok &= ctx.check("riscv-gcc-exit", code == 0, f"gcc exit={code}")
  if not elf_path.is_file():
    return ctx.check("elf-produced", False, f"{elf_path} missing")

  readelf = ctx.tools["riscv-readelf"]
  header_code, header_text = ctx.run_capture(
    [readelf, "-h", str(elf_path)], work, "readelf-header", timeout=120
  )
  attrs_code, attrs_text = ctx.run_capture(
    [readelf, "-A", str(elf_path)], work, "readelf-attrs", timeout=120
  )
  symbols_code, symbols_text = ctx.run_capture(
    [readelf, "-s", str(elf_path)], work, "readelf-symbols", timeout=120
  )
  disasm_code, disasm_text = ctx.run_capture(
    [ctx.tools["riscv-objdump"], "-d", "-M", "no-aliases", str(elf_path)],
    work,
    "objdump-disasm",
    timeout=120,
  )
  ok &= ctx.check(
    "elf-inspection-exit",
    header_code == 0 and attrs_code == 0 and symbols_code == 0 and disasm_code == 0,
    f"readelf/objdump exits h={header_code} A={attrs_code} s={symbols_code} d={disasm_code}",
  )

  info_text = (
    "# readelf -h\n" + header_text + "\n# readelf -A\n" + attrs_text + "\n# readelf -s\n" + symbols_text
  )
  info_path = work / "elf-info.txt"
  info_path.write_text(info_text, encoding="utf-8")
  disasm_path = work / "disasm.txt"
  disasm_path.write_text(disasm_text, encoding="utf-8")

  ok &= ctx.check(
    "elf-class",
    "ELF64" in (readelf_field(header_text, "Class") or ""),
    f"Class={readelf_field(header_text, 'Class')!r}",
  )
  ok &= ctx.check(
    "elf-machine",
    "RISC-V" in (readelf_field(header_text, "Machine") or ""),
    f"Machine={readelf_field(header_text, 'Machine')!r}",
  )
  entry = readelf_field(header_text, "Entry point address")
  ok &= ctx.check(
    "elf-entry-nonzero",
    entry is not None and entry.lower() not in ("0x0", "0"),
    f"Entry point address={entry!r}",
  )
  elf_type = readelf_field(header_text, "Type")
  ok &= ctx.check("elf-type-exec", elf_type is not None and "EXEC" in elf_type, f"Type={elf_type!r}")

  start_address = None
  for line in symbols_text.splitlines():
    match = re.search(r"^\s*\d+:\s+([0-9a-fA-F]+)\s+\d+\s+\S+\s+GLOBAL\s+\S+\s+\S+\s+(\S+)$", line)
    if match and match.group(2) == "_start":
      start_address = match.group(1)
      break
  ok &= ctx.check("symbol-start", start_address is not None, "global FUNC symbol '_start' required")
  if entry is not None and start_address is not None:
    ok &= ctx.check(
      "entry-equals-start",
      int(entry, 16) == int(start_address, 16),
      f"entry {entry} != _start {start_address}",
    )

  arch = (readelf_field(attrs_text, "Tag_RISCV_arch") or "").strip('"')
  ok &= ctx.check("arch-rv64", arch.startswith("rv64"), f"Tag_RISCV_arch={arch!r} must start with rv64")
  ok &= ctx.check(
    "arch-double-float", "_d2p" in arch, f"Tag_RISCV_arch={arch!r} must include the D extension"
  )
  # Float ABI lives in e_flags (EF_RISCV_FLOAT_ABI), printed by readelf -h as
  # ", double-float ABI" — binutils defines no Tag_RISCV_float_abi attribute.
  flags = readelf_field(header_text, "Flags") or ""
  ok &= ctx.check(
    "abi-double",
    "double-float ABI" in flags,
    f"ELF Flags={flags!r} must include double-float ABI (EF_RISCV_FLOAT_ABI_DOUBLE)",
  )

  ok &= ctx.check(
    "disasm-float-ops",
    "fmul.d" in disasm_text and "fadd.d" in disasm_text,
    "disassembly must contain linked D-extension ops (fmul.d, fadd.d)",
  )

  # Data-driven expectations declared machine-greppably in the smoke source
  # header (CHECK-SYMBOL: / CHECK-INSN: lines) so the runner never hardcodes
  # a private convention.
  source_text = source.read_text(encoding="utf-8", errors="replace")
  expected_symbols = sorted(set(re.findall(r"^\s*(?:\*\s*)?CHECK-SYMBOL:\s*(\S+)", source_text, re.M)))
  expected_insns = sorted(set(re.findall(r"^\s*(?:\*\s*)?CHECK-INSN:\s*(\S+)", source_text, re.M)))
  defined_symbols = set(re.findall(r"\s(\S+)$", symbols_text, re.M))
  missing_symbols = [s for s in expected_symbols if s not in defined_symbols]
  ok &= ctx.check(
    "declared-symbols-defined",
    not missing_symbols,
    f"CHECK-SYMBOL names missing from the symbol table: {missing_symbols}",
  )
  missing_insns = [insn for insn in expected_insns if insn not in disasm_text]
  ok &= ctx.check(
    "declared-insns-linked",
    not missing_insns,
    f"CHECK-INSN mnemonics missing from disassembly: {missing_insns}",
  )

  artifacts = ctx.artifacts_dir()
  elf_artifact = artifacts / "smoke.elf"
  shutil.copyfile(elf_path, elf_artifact)
  ctx.record_artifact(elf_artifact, deterministic=True)
  shutil.copyfile(info_path, artifacts / "elf-info.txt")
  ctx.record_artifact(artifacts / "elf-info.txt", deterministic=True)
  shutil.copyfile(disasm_path, artifacts / "disasm.txt")
  ctx.record_artifact(artifacts / "disasm.txt", deterministic=True)
  return bool(ok)


def case_python_env(ctx: CaseContext) -> bool:
  """Layer F: the real rebuilt validator executes the boa-am-matmul typed
  workload end to end (report checks all pass) and a byte-level replay of the
  CLI-written artifact reproduces the P01 fixed-input golden C bytes."""
  ok = True
  mlir = ctx.repo / PYTHON_ENV_FIXTURES[0]
  inputs = ctx.repo / "examples/workloads/boa_am_inputs/boa_am_matmul"
  for rel in PYTHON_ENV_FIXTURES[1:]:
    if not (ctx.repo / rel).is_file():
      return ctx.check("fixture-present", False, f"missing {rel}")

  artifacts = ctx.artifacts_dir()
  compiled = artifacts / "boa-am-matmul.compiled.json"
  report_path = artifacts / "report.json"

  overrides = sorted(ctx.config.get("validator", {}).get("hw_overrides", {}).items())
  argv = [sys.executable, "-m", "pipeline_validator", "--ir-file", str(mlir)]
  for key, value in overrides:
    argv.extend(["--hw-override", f"{key}={override_value(value)}"])
  argv.extend(
    [
      "--input-binding",
      "A=0x100000:16384:r",
      "--input-binding",
      "B=0x200000:65536:r",
      "--input-binding",
      "C=0x300000:65536:w",
      "--input-data",
      f"A={inputs / 'A.bin'}",
      "--input-data",
      f"B={inputs / 'B.bin'}",
      "--max-cycles",
      "200000",
      "--json",
      "--report",
      str(report_path),
      "--compiled-output",
      str(compiled),
    ]
  )
  code = ctx.run(argv, ctx.repo, "validator-cli", timeout=3600)
  ok &= ctx.check("validator-cli-exit", code == 0, f"pipeline_validator exit={code}")

  workload = None
  if report_path.is_file():
    try:
      reports = json.loads(report_path.read_text(encoding="utf-8"))
      ok &= ctx.check(
        "report-one-workload",
        isinstance(reports, list) and len(reports) == 1,
        f"expected exactly one workload report, got {len(reports) if isinstance(reports, list) else '?'}",
      )
      if isinstance(reports, list) and reports:
        workload = reports[0]
    except (ValueError, UnicodeError) as exc:
      ok &= ctx.check("report-parses", False, f"invalid report JSON: {exc}")
  else:
    ok &= ctx.check("report-written", False, f"{report_path} missing")

  if workload is not None:
    ok &= ctx.check(
      "workload-name", workload.get("name") == "boa_am_matmul", f"name={workload.get('name')!r}"
    )
    ok &= ctx.check(
      "workload-completed",
      workload.get("completed") is True,
      f"completed={workload.get('completed')!r} reason={workload.get('reason')!r}",
    )
    checks = workload.get("checks", [])
    failed = [c.get("check") for c in checks if not c.get("pass")]
    ok &= ctx.check(
      "report-checks-all-pass", len(checks) > 0 and not failed, f"failed report checks: {failed}"
    )
    ctx.metrics["cycles"] = workload.get("cycles")
    ctx.metrics["checks_passed"] = sum(1 for c in checks if c.get("pass"))

  target_yaml = artifacts / "boa-am-matmul.compiled.target.yaml"
  if compiled.is_file() and target_yaml.is_file():
    script_path = artifacts / "replay_script.py"
    script_path.write_text(REPLAY_SCRIPT, encoding="utf-8")
    # A script run by path puts its own directory (not the cwd) on sys.path,
    # so pipeline_validator must be provided explicitly.
    replay_code = ctx.run(
      [
        sys.executable,
        str(script_path),
        str(compiled),
        str(target_yaml),
        str(ctx.request["seed"]),
        str(artifacts),
        GOLDEN_C_SHA256,
      ],
      ctx.repo,
      "byte-replay",
      env={"PYTHONPATH": str(ctx.repo)},
      timeout=3600,
    )
    ok &= ctx.check(
      "byte-replay-exit",
      replay_code == 0,
      f"replay must complete and reproduce the P01 golden C bytes (SHA-256 {GOLDEN_C_SHA256})",
    )
    replay_path = artifacts / "replay.json"
    if replay_path.is_file():
      try:
        record = json.loads(replay_path.read_text(encoding="utf-8"))
        ctx.metrics["replay_cycles"] = record.get("cycles")
        ok &= ctx.check(
          "replay-bytes-equal",
          record.get("bytes_equal") is True and record.get("completed") is True,
          f"replay record={record}",
        )
      except (ValueError, UnicodeError) as exc:
        ok &= ctx.check("replay-json-parses", False, f"invalid replay JSON: {exc}")

  if compiled.is_file():
    ctx.record_artifact(compiled, deterministic=True)
  for name in (
    "boa-am-matmul.compiled.exec.txt",
    "boa-am-matmul.compiled.compiled.mlir.txt",
    "boa-am-matmul.compiled.target.yaml",
  ):
    candidate = artifacts / name
    if candidate.is_file():
      ctx.record_artifact(candidate, deterministic=True)
  for name in ("report.json", "replay_script.py", "replay.json", "c_output.bin"):
    candidate = artifacts / name
    if candidate.is_file():
      ctx.record_artifact(candidate, deterministic=True)
  return bool(ok)


CASE_RUNNERS = {
  "elaborate": (case_elaborate, "R"),
  "softfloat": (case_softfloat, "F"),
  "riscv": (case_riscv, "F"),
  "python-env": (case_python_env, "F"),
  "selftest": (numeric_cases.case_selftest, "F"),
  "crosscheck": (numeric_cases.case_crosscheck, "R"),
  "probe-nan": (numeric_cases.case_probe_nan, "R"),
  "vectors": (numeric_cases.case_vectors, "F"),
  "vector-parity": (numeric_profile_cases.case_vector_parity, "F"),
  "zero-k": (numeric_profile_cases.case_zero_k, "F"),
  "cutover-reject": (numeric_profile_cases.case_cutover_reject, "F"),
}


def load_request(path: Path) -> dict:
  try:
    request = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeError, ValueError) as exc:
    fail_env(f"cannot read request {path}: {exc}")
  if request.get("schema") != SCHEMA:
    fail_env(f"request schema must be {SCHEMA}")
  for key in ("mode", "target", "task_id", "cases", "seed", "config_content", "expect"):
    if key not in request:
      fail_env(f"request missing required key '{key}'")
  return request


def verify_lock(repo: Path, request: dict) -> dict:
  lock_path = repo / request["lock_path"]
  if not lock_path.is_file():
    fail_env(f"lock not found at {lock_path}")
  observed_sha = sha256_path(lock_path)
  if observed_sha != request["expect"]["lock_sha256"]:
    fail_env(
      f"mounted {request['lock_path']} sha256 {observed_sha} != expected "
      f"{request['expect']['lock_sha256']} (repo mount/request mismatch)"
    )
  lock = json.loads(lock_path.read_text(encoding="utf-8"))
  if lock.get("schema") != SCHEMA:
    fail_env(f"lock schema must be {SCHEMA}")
  if lock.get("image", {}).get("id") != request["expect"]["image_id"]:
    fail_env(
      f"lock image.id {lock.get('image', {}).get('id')!r} != running image "
      f"{request['expect']['image_id']!r}"
    )
  build_inputs = lock.get("build_inputs")
  if not isinstance(build_inputs, dict) or not build_inputs:
    fail_env("lock must carry a non-empty build_inputs mapping for source verification")
  problems = []
  for rel in sorted(build_inputs):
    expected = build_inputs[rel]
    path = repo / rel
    if not path.is_file():
      problems.append(f"{rel}: missing from repository mount")
      continue
    observed = sha256_path(path)
    if observed != expected:
      problems.append(f"{rel}: sha256 {observed} != lock {expected}")
  if problems:
    listing = "\n".join(f"  {problem}" for problem in problems[:20])
    fail_env(f"lock build_inputs verification failed for {len(problems)} file(s):\n{listing}")
  return lock


def verify_environment(ctx: CaseContext, lock: dict, request: dict) -> dict:
  if not str(sys.executable).startswith("/opt/validator/"):
    fail_env(f"runner must execute under /opt/validator, got {sys.executable!r}")
  home = os.environ.get("HOME") or "/tmp/nexus-home"
  os.makedirs(home, exist_ok=True)

  for required in REQUIRED_IMAGE_PATHS:
    if not Path(required).exists():
      fail_env(f"image contract violated: required path {required} is missing")

  chipyard_root = os.environ.get("NEXUS_DEPS") or "/opt/deps/chipyard"
  if not Path(chipyard_root).is_dir():
    fail_env(f"chipyard source root {chipyard_root} missing")
  gemmini_root = os.environ.get("NEXUS_GEMMINI") or str(ctx.repo / "hardware/gemmini/upstream")
  if not Path(gemmini_root).is_dir():
    fail_env(f"gemmini authoritative source root {gemmini_root} missing")

  tools = {
    "sbt": "/opt/bin/sbt",
    "verilator": resolve_tool("verilator"),
    "riscv-gcc": resolve_tool("riscv64-unknown-elf-gcc"),
    "riscv-readelf": resolve_tool("riscv64-unknown-elf-readelf"),
    "riscv-objdump": resolve_tool("riscv64-unknown-elf-objdump"),
    "java": resolve_tool("java", "/opt/native/bin/java"),
  }
  firtool = os.environ.get("FIRTOOL_BIN") or "/opt/circt/bin/firtool"
  if not Path(firtool).is_file():
    fail_env(f"pinned firtool binary not found: {firtool}")
  tools["firtool"] = firtool
  for name, path in tools.items():
    if not Path(path).is_file():
      fail_env(f"resolved tool {name} -> {path} is not a file")
  ctx.tools = tools
  ctx.chipyard_root = chipyard_root
  ctx.gemmini_root = gemmini_root

  probes: dict = {"tools": {}, "binaries": {}, "python": {}}
  probe_specs = {
    "firtool": ([tools["firtool"], "--version"], True),
    "verilator": ([tools["verilator"], "--version"], True),
    "riscv-gcc": ([tools["riscv-gcc"], "--version"], True),
    "java": ([tools["java"], "-version"], False),
  }
  for name, (argv, from_stdout) in probe_specs.items():
    proc = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    if proc.returncode != 0:
      fail_env(f"tool probe {' '.join(argv)} failed with exit {proc.returncode}")
    stream = proc.stdout if from_stdout else proc.stderr
    first_line = (stream or "").strip().splitlines()
    probes["tools"][name] = first_line[0] if first_line else ""
    probes["binaries"][argv[0]] = sha256_path(Path(argv[0]))

  dumpversion = subprocess.run(
    [tools["riscv-gcc"], "-dumpversion"], capture_output=True, text=True, timeout=120
  )
  probes["tools"]["riscv-gcc-dumpversion"] = dumpversion.stdout.strip()

  import importlib.metadata as metadata

  probes["python"] = {"executable": sys.executable, "version": sys.version.split()[0], "packages": {}}
  for distribution in ("xdsl", "PyYAML"):
    try:
      probes["python"]["packages"][distribution] = metadata.version(distribution)
    except metadata.PackageNotFoundError:
      probes["python"]["packages"][distribution] = "unavailable"

  probes["binaries"]["/opt/softfloat/lib/libsoftfloat.so"] = sha256_path(
    Path("/opt/softfloat/lib/libsoftfloat.so")
  )
  probes["binaries"]["/opt/toolchain-build.json"] = sha256_path(Path("/opt/toolchain-build.json"))
  return probes


def main() -> int:
  parser = argparse.ArgumentParser(description="In-container hardware validation runner")
  parser.add_argument("--request", required=True)
  parser.add_argument("--repo", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--expect-lock-sha256", required=True)
  parser.add_argument("--expect-image-id", required=True)
  args = parser.parse_args()

  repo = Path(args.repo).resolve()
  out = Path(args.out).resolve()
  request = load_request(Path(args.request))
  if (
    request["expect"]["lock_sha256"] != args.expect_lock_sha256
    or request["expect"]["image_id"] != args.expect_image_id
  ):
    fail_env("request expect block does not match launcher-provided identity")

  registry_path = repo / "hardware/scripts/targets.json"
  registry = json.loads(registry_path.read_text(encoding="utf-8"))
  entry = registry.get("targets", {}).get(request["target"])
  if entry is None:
    fail_env(f"request target '{request['target']}' is not registered")
  if not entry.get("implemented"):
    fail_env(f"target '{request['target']}' is registered but not implemented")
  task_id = entry["owner"]
  if request["task_id"] != task_id:
    fail_env(f"request task_id must match registered owner {task_id}")
  registered_cases = list(entry.get("cases", []))
  requested = list(request["cases"])
  unknown = [case for case in requested if case not in registered_cases]
  if unknown:
    fail_env(f"request cases {unknown} not registered for target '{request['target']}'")
  if not requested:
    fail_env("request must select at least one case")
  unbacked = [case for case in registered_cases if case not in CASE_RUNNERS]
  if unbacked:
    fail_env(f"registry/case-runner drift: {unbacked} registered without an implemented runner")

  config = validate_config(request["config_content"], entry, request)

  (out / "logs").mkdir(parents=True, exist_ok=True)
  ctx = CaseContext(repo, out, request, config)
  lock = verify_lock(repo, request)
  probes = verify_environment(ctx, lock, request)

  case_results: dict = {}
  failures: list[dict] = []
  for case_id in (case for case in registered_cases if case in requested):
    runner, layer = CASE_RUNNERS[case_id]
    ctx.begin_case(case_id, layer)
    print(f"[case] {case_id} (layer {layer}) start", flush=True)
    try:
      runner(ctx)
    except Exception as exc:
      ctx.check("case-internal-error", False, f"{type(exc).__name__}: {exc}")
    result = ctx.result()
    case_results[case_id] = result
    print(f"[case] {case_id} -> {result['status']}", flush=True)
    if result["status"] != "PASS":
      failures.append(
        {
          "case": case_id,
          "failed_assertions": [
            {"name": a["name"], "detail": a["detail"]} for a in result["assertions"] if not a["pass"]
          ],
        }
      )

  overall = "PASS" if not failures else "FAIL"
  work_removed = False
  if overall == "PASS":
    shutil.rmtree(ctx.work)
    work_removed = not ctx.work.exists()

  fixtures = {}
  if "python-env" in requested:
    for rel in PYTHON_ENV_FIXTURES:
      path = repo / rel
      fixtures[rel] = sha256_path(path) if path.is_file() else "missing"
  if request["target"] == "numeric":
    for path in sorted((repo / "hardware/boa/reference/vectors").glob("*.json")):
      fixtures[str(path.relative_to(repo))] = sha256_path(path)
  if request["target"] == "numeric-profile":
    for rel in (
      "hardware/boa/reference/vectors/matmul.json",
      "pipeline_validator/tests/data/numeric_profile_directed_golden.json",
      "pipeline_validator/tests/data/legacy_v4_boa_am.json",
    ):
      path = repo / rel
      fixtures[rel] = sha256_path(path) if path.is_file() else "missing"

  cleanup = {
    "work_root": "/out/work",
    "removed": work_removed,
    "policy": "removed on PASS; kept on FAIL for failure diagnosis",
    "network": "none (host-enforced)",
    "user": f"{os.getuid()}:{os.getgid()}",
    "capabilities": "dropped (cap-drop ALL, no-new-privileges)",
    "mounts": ["/workspace:ro", "/out:rw"],
  }

  counts = {
    "cases": len(case_results),
    "passed": sum(1 for r in case_results.values() if r["status"] == "PASS"),
    "failed": len(failures),
    "assertions": sum(len(r["assertions"]) for r in case_results.values()),
    "assertions_failed": sum(1 for r in case_results.values() for a in r["assertions"] if not a["pass"]),
    "commands": sum(len(r["commands"]) for r in case_results.values()),
  }

  manifest = {
    "schema": SCHEMA,
    "task_id": task_id,
    "mode": request["mode"],
    "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "status": overall,
    "request": request,
    "config": config,
    "lock": {"path": request["lock_path"], "sha256": request["expect"]["lock_sha256"], "content": lock},
    "image": {"id": request["expect"]["image_id"]},
    "source": {"host": request["source"], "fixtures": fixtures},
    "probes": probes,
    "environment": {
      "uid": os.getuid(),
      "gid": os.getgid(),
      "network": "none",
      "work": "/out/work",
      "home": os.environ.get("HOME"),
    },
    "cases": case_results,
    "resource_cleanup": cleanup,
    "counts": counts,
  }
  write_json(out / "manifest.json", manifest)

  summary = {
    "schema": SCHEMA,
    "task_id": task_id,
    "status": overall,
    "validation_layer": {case: result["layer"] for case, result in case_results.items()},
    "cases": list(case_results),
    "failures": failures,
    "counts": counts,
    "resource_cleanup": cleanup,
    "manifest": "manifest.json",
  }
  write_json(out / "summary.json", summary)

  print(
    f"[summary] status={overall} cases={counts['cases']} passed={counts['passed']} "
    f"failed={counts['failed']} assertions={counts['assertions']} "
    f"failed_assertions={counts['assertions_failed']}",
    flush=True,
  )
  return EXIT_OK if overall == "PASS" else EXIT_CASES


if __name__ == "__main__":
  raise SystemExit(main())
