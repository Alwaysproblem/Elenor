"""C029 real RTL acceptance: behavioral mutants before bank/Frame/ledger cases."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
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
TARGETS = {
  "tb_l1_bank": ("l1_bank.sv", "tb/tb_l1_bank.sv"),
  "tb_slot_frame": ("slot_frame.sv", "tb/tb_slot_frame.sv"),
  "tb_arena_ledger": ("slot_frame.sv", "l1_bank.sv", "arena_ledger.sv", "tb/tb_arena_ledger.sv"),
}
MUTANT_TOPS = {"bank": "tb_l1_bank", "frame": "tb_slot_frame", "ledger": "tb_arena_ledger"}


def _selected_target() -> dict[str, Any]:
  configured = os.environ.get("CN_UNIT_CONFIG")
  path = Path(configured) if configured else memory_params.gt.OUTPUTS["cs1_c2"]
  data = yaml.safe_load(path.read_text())
  assert isinstance(data, dict) and data.get("name") in ("cs1_c1", "cs1_c2"), path
  expected = yaml.safe_load(memory_params.gt.OUTPUTS[data["name"]].read_text())
  assert data == expected, f"{path}: unit config differs from the sealed C019 target"
  return data


def _compile_and_run(label: str, top: str, root: str, contexts: int) -> list[str]:
  files = ("c029_params_pkg.sv", *TARGETS[top])
  sources = " ".join(shlex.quote(f"{root}/{path}") for path in files)
  return [
    "rc=0",
    f'"${{VEXE[0]}}" --binary --timing --assert -j 4 --top-module {top} '
    f"-GCONTEXT_COUNT={contexts} --Mdir /out/obj-{label} {sources} "
    f"> /out/{label}.build.log 2>&1 || rc=$?",
    f'printf "%s\\n" "$rc" > /out/{label}.build.rc',
    f"if (( rc == 0 )); then timeout 180 /out/obj-{label}/V{top} "
    f"> /out/{label}.log 2>&1 || rc=$?; fi",
    f'printf "%s\\n" "$rc" > /out/{label}.rc',
  ]


@pytest.fixture(scope="module")
def c029_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  assert shutil.which("docker"), "C029 requires the locked Docker builder"
  memory_params.check()
  params = memory_params.parameters()
  target = _selected_target()
  requested_contexts = int(target["constraints"]["context_count"])
  # C2 also proves the C019 same-RTL C1 degradation direction.
  context_counts = sorted({int(params["C1_CONTEXTS"]), requested_contexts})
  root = tmp_path_factory.mktemp("c029")
  evidence, work = root / "evidence", root / "work"
  evidence.mkdir()
  (evidence / "requested-target.json").write_text(json.dumps(target, indent=2) + "\n")
  required = {RTL / "c029_params_pkg.sv"}
  required.update(RTL / path for paths in TARGETS.values() for path in paths)
  mutants: list[tuple[Path, str]] = []
  for family, top in MUTANT_TOPS.items():
    paths = sorted(MUTANTS.glob(f"{family}-*.patch"))
    assert paths, f"missing behavioral {family} mutant"
    mutants.extend((path, top) for path in paths)
    required.update(paths)
  before = {
    str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(required)
  }
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
  script = [
    "set -euo pipefail",
    "export GIT_CEILING_DIRECTORIES=/work:/out",
    "cd /work/coralnpu",
    "bazel --output_user_root=/tmp/ob build -c opt --disk_cache=/bcache/disk "
    "--repository_cache=/bcache/repo --vendor_dir=/bcache/vendor "
    "--symlink_prefix=/tmp/bazel- @verilator//:verilator_bin > /out/verilator-tool.log 2>&1",
    "VEXE=(/tmp/bazel-bin/external/*/verilator_bin)",
    'test "${#VEXE[@]}" -eq 1',
    'export VERILATOR_ROOT="${VEXE[0]}.runfiles/$(basename "$(dirname "${VEXE[0]}")")"',
    'test -f "$VERILATOR_ROOT/include/verilated.mk"',
    "export VERILATOR_CXX=g++ VERILATOR_AR=ar VERILATOR_PYTHON3=python3",
  ]
  negative = []
  for index, (patch, top) in enumerate(mutants):
    label = f"red-{index}"
    variant = f"/out/variant-{index}"
    patch_path = shlex.quote(f"/mutants/{patch.name}")
    script.extend(
      [
        f"cp -a /rtl {variant}",
        f"git -C {variant} apply --check {patch_path} > /out/{label}.patch.log 2>&1",
        f"git -C {variant} apply {patch_path} >> /out/{label}.patch.log 2>&1",
        *_compile_and_run(label, top, variant, requested_contexts),
      ]
    )
    negative.append({"label": label, "patch": patch.name, "top": top})
  positive = []
  for contexts in context_counts:
    for top in TARGETS:
      label = f"green-c{contexts}-{top}"
      script.extend(_compile_and_run(label, top, "/rtl", contexts))
      positive.append({"label": label, "top": top, "context_count": contexts})
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
  run = subprocess.run(command, capture_output=True, text=True, check=False, timeout=3600)
  (evidence / "runner.log").write_text(run.stdout + run.stderr)
  assert run.returncode == 0, f"C029 runner failed; evidence={evidence}\n{run.stderr}"
  assert before == {
    str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(required)
  }, "C029 source changed during acceptance"
  assert hashlib.sha256((H / "protocol/PROTOCOL_FREEZE.json").read_bytes()).hexdigest() == params[
    "SEAL_SHA256"
  ], "C020 seal changed"
  return {
    "path": evidence,
    "positive": positive,
    "negative": negative,
    "requested_target": target["name"],
    "seal_sha256": params["SEAL_SHA256"],
  }


def test_c029_behavior_and_mutation_acceptance(c029_evidence: dict[str, Any]) -> None:
  evidence = c029_evidence["path"]
  for case in c029_evidence["negative"]:
    label = case["label"]
    build = (evidence / f"{label}.build.log").read_text()
    assert int((evidence / f"{label}.build.rc").read_text()) == 0, build[-8000:]
    log = (evidence / f"{label}.log").read_text()
    rc = int((evidence / f"{label}.rc").read_text())
    assert rc in (1, 134), f"{case['patch']}: not a simulator assertion exit ({rc})\n{log[-8000:]}"
    assert any(
      ("%Error:" in line or "%Fatal:" in line) and "C029_ASSERT" in line for line in log.splitlines()
    ), (
      f"{case['patch']}: no executed RTL assertion\n{log[-8000:]}"
    )
  for case in c029_evidence["positive"]:
    label = case["label"]
    build = (evidence / f"{label}.build.log").read_text()
    assert int((evidence / f"{label}.build.rc").read_text()) == 0, build[-8000:]
    log = (evidence / f"{label}.log").read_text()
    assert int((evidence / f"{label}.rc").read_text()) == 0, log[-8000:]
    assert f"C029_PASS {case['top']}" in log, log[-8000:]
  result = {key: value for key, value in c029_evidence.items() if key != "path"}
  result["card"] = "C029"
  (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")
