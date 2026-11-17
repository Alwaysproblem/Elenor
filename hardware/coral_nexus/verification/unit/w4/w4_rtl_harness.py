"""Shared Docker/Verilator harness for the C030 RTL acceptance runners.

Every C030 slice (MSHR, program port, integrated miss path, native C028 fetch integration)
needs the same scaffolding: the sealed-target selection, a source/seal hash fence, a materialized
`coralnpu` work tree, the offline Bazel Verilator toolchain inside the locked builder image, one
`docker run`, and the same behavioral-mutation verdict rules. This module owns all of it once so
`cn unit --suite w4` pays for the toolchain build per runner invocation, not per copy of the code,
and so a rule such as "a mutant is only killed by an executed `C030_ASSERT` after a clean build"
cannot drift between slices.

A slice supplies only what differs: its benches, its mutant glob, and (optionally) extra shell
steps that run after the toolchain is ready (the fetch slice emits Chisel Verilog there).
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[5]
H = REPO / "hardware/coral_nexus"
sys.path.insert(0, str(H / "scripts/lib"))

import gen_memory_params as memory_params  # noqa: E402

RTL = H / "rtl/memory"
MUTANTS = H / "verification/unit/w4/mutants"
CACHE = REPO / "out/coral_nexus/cache/w2_bazel"

TOOLCHAIN = [
  "set -euo pipefail",
  "export GIT_CEILING_DIRECTORIES=/work:/out",
  "cd /work/coralnpu",
  'BZL="bazel --output_user_root=/tmp/ob"',
  'COMMON="-c opt --disk_cache=/bcache/disk --repository_cache=/bcache/repo '
  '--vendor_dir=/bcache/vendor --symlink_prefix=/tmp/bazel-"',
]
VERILATOR_READY = [
  "$BZL build $COMMON @verilator//:verilator_bin > /out/verilator-tool.log 2>&1",
  "VEXE=(/tmp/bazel-bin/external/*/verilator_bin)",
  'test "${#VEXE[@]}" -eq 1',
  'export VERILATOR_ROOT="${VEXE[0]}.runfiles/$(basename "$(dirname "${VEXE[0]}")")"',
  'test -f "$VERILATOR_ROOT/include/verilated.mk"',
  "export VERILATOR_CXX=g++ VERILATOR_AR=ar VERILATOR_PYTHON3=python3",
]


@dataclass(frozen=True)
class Bench:
  """One Verilator top with the sources it needs and the Context counts it must pass at."""

  top: str
  sources: tuple[str, ...]
  contexts: tuple[int, ...] | None = None  # None: the bench has no Context dimension
  # `{contexts}` is substituted per run; absolute in-container paths (e.g. generated Verilog).
  extra_sources: tuple[str, ...] = ()
  args_by_context: dict[int, str] = field(default_factory=dict)
  timeout_s: int = 300
  run_args: tuple[str, ...] = ()  # {label}: unique per mutant/green service endpoint


@dataclass
class Slice:
  """Everything a slice declares; the harness derives the rest."""

  name: str  # evidence directory stem and result card
  benches: Sequence[Bench]
  mutant_globs: tuple[str, ...]  # every glob must match at least one mutant
  mutant_bench: Bench  # the bench the mutants are run against
  min_mutants: int = 1
  pre_toolchain_steps: Sequence[str] = field(default_factory=list)
  post_toolchain_steps: Sequence[str] = field(default_factory=list)
  mutant_contexts: int | None = None  # None: the requested target's Context count
  assert_markers: tuple[str, ...] = ("C030_ASSERT",)
  service: Callable[[Path, Sequence[str]], AbstractContextManager[None]] | None = None
  host_sources: tuple[Path, ...] = ()


def selected_target() -> dict[str, Any]:
  configured = os.environ.get("CN_UNIT_CONFIG")
  path = Path(configured) if configured else memory_params.gt.OUTPUTS["cs1_c2"]
  data = yaml.safe_load(path.read_text())
  assert isinstance(data, dict) and data.get("name") in ("cs1_c1", "cs1_c2"), path
  expected = yaml.safe_load(memory_params.gt.OUTPUTS[data["name"]].read_text())
  assert data == expected, f"{path}: unit config differs from the sealed C019 target"
  return data


def _sha(paths: Sequence[Path]) -> dict[str, str]:
  return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def _label(bench: Bench, contexts: int | None, prefix: str) -> str:
  suffix = "" if contexts is None else f"-c{contexts}"
  return f"{prefix}{suffix}-{bench.top}"


def _compile_and_run(label: str, bench: Bench, root: str, contexts: int | None) -> list[str]:
  sources = " ".join(shlex.quote(f"{root}/{path}") for path in bench.sources)
  extra = " ".join(shlex.quote(path.format(contexts=contexts)) for path in bench.extra_sources)
  gen = "" if contexts is None else f"-GCONTEXT_COUNT={contexts} "
  args = bench.args_by_context.get(contexts, "") if contexts is not None else ""
  run_args = " ".join(shlex.quote(arg.format(label=label)) for arg in bench.run_args)
  return [
    "rc=0",
    f'"${{VEXE[0]}}" --binary --timing --assert -j 4 --top-module {bench.top} {gen}{args} '
    f"--Mdir /out/obj-{label} {sources} {extra} > /out/{label}.build.log 2>&1 || rc=$?",
    f'printf "%s\\n" "$rc" > /out/{label}.build.rc',
    f"if (( rc == 0 )); then timeout {bench.timeout_s} /out/obj-{label}/V{bench.top} {run_args} "
    f"> /out/{label}.log 2>&1 || rc=$?; fi",
    f'printf "%s\\n" "$rc" > /out/{label}.rc',
  ]


def run_slice(spec: Slice, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  """Build the toolchain once, run the mutants then every green bench, return the evidence."""
  assert shutil.which("docker"), "C030 requires the locked Docker builder"
  memory_params.check()
  params = memory_params.parameters()
  target = selected_target()
  requested = int(target["constraints"]["context_count"])
  # C2 also proves the C019 same-RTL C1 degradation direction.
  counts = sorted({int(params["C1_CONTEXTS"]), requested})
  root = tmp_path_factory.mktemp(spec.name)
  evidence, work = root / "evidence", root / "work"
  evidence.mkdir()
  (evidence / "requested-target.json").write_text(json.dumps(target, indent=2) + "\n")

  mutants = []
  for glob in spec.mutant_globs:
    matched = sorted(MUTANTS.glob(glob))
    # A glob that silently matches nothing would drop a whole mutant family from acceptance.
    assert matched, f"{spec.name}: mutant glob {glob!r} matches no patch"
    mutants.extend(matched)
  assert len(mutants) >= spec.min_mutants, f"{spec.name}: expected >= {spec.min_mutants} mutants"
  required = {RTL / s for b in [*spec.benches, spec.mutant_bench] for s in b.sources} | set(mutants)
  required.update(spec.host_sources)
  before = _sha(sorted(required))
  (evidence / "source-sha256.json").write_text(json.dumps(before, indent=2) + "\n")

  materialize = subprocess.run(
    [
      sys.executable,
      str(H / "scripts/lib/sources_tool.py"),
      "materialize",
      "--lock",
      str(H / "env/sources.lock"),
      "--sources",
      "coralnpu",
      "--out",
      str(work),
    ],
    capture_output=True,
    text=True,
    check=False,
  )
  (evidence / "materialize.log").write_text(materialize.stdout + materialize.stderr)
  assert materialize.returncode == 0, materialize.stderr
  for name in ("disk", "repo", "vendor"):
    assert (CACHE / name).is_dir(), f"missing pinned offline Bazel cache {name}"

  script = [*TOOLCHAIN, *spec.pre_toolchain_steps, *VERILATOR_READY, *spec.post_toolchain_steps]
  negative = []
  mutant_contexts = spec.mutant_contexts if spec.mutant_contexts is not None else requested
  for index, patch in enumerate(mutants):
    label = f"red-{index}"
    variant = f"/out/variant-{index}"
    patch_path = shlex.quote(f"/mutants/{patch.name}")
    script.extend(
      [
        f"cp -a /rtl {variant}",
        f"git -C {variant} apply --check {patch_path} > /out/{label}.patch.log 2>&1",
        f"git -C {variant} apply {patch_path} >> /out/{label}.patch.log 2>&1",
        *_compile_and_run(
          label, spec.mutant_bench, variant, mutant_contexts if spec.mutant_bench.contexts else None
        ),
      ]
    )
    negative.append({"label": label, "patch": patch.name})
  positive = []
  for bench in spec.benches:
    for contexts in bench.contexts if bench.contexts else (None,):
      if contexts is not None and contexts not in counts:
        continue
      label = _label(bench, contexts, "green")
      script.extend(_compile_and_run(label, bench, "/rtl", contexts))
      positive.append({"label": label, "top": bench.top, "context_count": contexts})
  (evidence / "run.sh").write_text("\n".join(script) + "\n")

  image = json.loads((H / "env/image.lock").read_text())["tag"]
  command = [
    "docker",
    "run",
    "--rm",
    "--network=none",
    "--user",
    f"{os.getuid()}:{os.getgid()}",
    "-e",
    "HOME=/tmp",
    "-v",
    f"{CACHE.resolve()}:/bcache",
    "-v",
    f"{work.resolve()}:/work",
    "-v",
    f"{evidence.resolve()}:/out",
    "-v",
    f"{RTL.resolve()}:/rtl:ro",
    "-v",
    f"{MUTANTS.resolve()}:/mutants:ro",
    image,
    "bash",
    "/out/run.sh",
  ]
  cases: list[dict[str, Any]] = [*negative, *positive]
  labels = [case["label"] for case in cases]
  service: AbstractContextManager[None] = (
    spec.service(evidence, labels) if spec.service else nullcontext()
  )
  with service:
    run = subprocess.run(command, capture_output=True, text=True, check=False, timeout=3600)
  (evidence / "runner.log").write_text(run.stdout + run.stderr)
  assert run.returncode == 0, f"{spec.name} runner failed; evidence={evidence}\n{run.stderr}"
  assert before == _sha(sorted(required)), f"{spec.name}: source changed during acceptance"
  seal = hashlib.sha256((H / "protocol/PROTOCOL_FREEZE.json").read_bytes()).hexdigest()
  assert seal == params["SEAL_SHA256"], "C020 seal changed"
  return {
    "path": evidence,
    "positive": positive,
    "negative": negative,
    "requested_target": target["name"],
    "seal_sha256": params["SEAL_SHA256"],
    "assert_markers": spec.assert_markers,
  }


def check_verdicts(evidence: dict[str, Any], *, pass_marker: str) -> None:
  """Mutants die by an executed assertion after a clean build; every green bench passes."""
  path = evidence["path"]
  for case in evidence["negative"]:
    label = case["label"]
    build = (path / f"{label}.build.log").read_text()
    assert int((path / f"{label}.build.rc").read_text()) == 0, build[-8000:]
    log = (path / f"{label}.log").read_text()
    rc = int((path / f"{label}.rc").read_text())
    assert rc in (1, 134), f"{case['patch']}: not a simulator assertion exit ({rc})\n{log[-8000:]}"
    assert any(
      ("%Error:" in line or "%Fatal:" in line)
      and any(marker in line for marker in evidence["assert_markers"])
      and "watchdog" not in line.lower()
      for line in log.splitlines()
    ), f"{case['patch']}: no executed RTL assertion\n{log[-8000:]}"
  for case in evidence["positive"]:
    label = case["label"]
    build = (path / f"{label}.build.log").read_text()
    assert int((path / f"{label}.build.rc").read_text()) == 0, build[-8000:]
    log = (path / f"{label}.log").read_text()
    assert int((path / f"{label}.rc").read_text()) == 0, log[-8000:]
    assert f"{pass_marker} {case['top']}" in log, log[-8000:]
