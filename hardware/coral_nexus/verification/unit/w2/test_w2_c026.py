"""W2-02 / C026 unit & Core-integration verification suite (`cn unit --suite w2`).

Proves:
1. `cn sources verify` passes with disjoint single-owner patch assignments and both
   pinned submodules remain pristine.
2. RED -> GREEN mutation proof (sharing one warm Bazel JVM inside `coral-nexus-builder:c002`
   with `--network=none`):
   - Mutant `w2/mutants/0202-pc-key-store.patch` (ignoring `OwnerTag` on `storeComplete`)
     MUST fail `//hdl/chisel/src/coralnpu:w2_02_rob_owner_spec`.
   - Mutant `w2/mutants/0203-tie-owner-const.patch` (tying MLU/LSU return `OwnerTag` to 0)
     MUST fail `//hdl/chisel/src/coralnpu:w2_02_rf_bank_spec`.
3. Full ChiselSim behavioral + Core-level integration suite passes:
   - `w2_02_rob_owner_spec`: same-PC concurrent dual-Context stores, independent partition
     retirement (RET-03 / T-09), per-Context fault flush isolation (RET-06), composite
     `RobTag` wraparound ABA rejection.
   - `w2_02_rf_bank_spec`: `BankedRf` + `BankedFRegfile` isolation, background writeback
     while peer Context is selected (STA-06), `c1` explicit 1-bit specialization, and
     end-to-end patched `Core` (`c2`) executing real ALU + 3-stage MLU + LSU store across
     Context 0 and Context 1.
   - `w2_02_csr_flag_delta_spec`: per-Context `(tileId, ctx_slot)` `mhartid`, isolated trap
     & flags state, generation bump clearing, and stale-generation delta/CSR rejection.
   - Upstream scalar unit regressions (`coralnpu_alu_tests`, `coralnpu_mlu_tests`,
     `coralnpu_fregfile_tests`) remain green.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
H_ROOT = REPO_ROOT / "hardware/coral_nexus"
SOURCES_TOOL = H_ROOT / "scripts/lib/sources_tool.py"
VERIFY_SH = H_ROOT / "scripts/lib/verify_sources.sh"
LOCK_FILE = H_ROOT / "env/sources.lock"
IMAGE_LOCK = H_ROOT / "env/image.lock"
MUT_PC_STORE = H_ROOT / "patches/w2/mutants/0202-pc-key-store.patch"
MUT_TIE_OWNER = H_ROOT / "patches/w2/mutants/0203-tie-owner-const.patch"
BAZEL_CACHE = REPO_ROOT / "out/coral_nexus/cache/w2_bazel"


def _ensure_bazel_cache() -> None:
  (BAZEL_CACHE / "disk").mkdir(parents=True, exist_ok=True)
  (BAZEL_CACHE / "repo").mkdir(parents=True, exist_ok=True)
  (BAZEL_CACHE / "vendor").mkdir(parents=True, exist_ok=True)
  c005_work = REPO_ROOT / "out/coral_nexus/C005/work-coralnpu"
  if not any((BAZEL_CACHE / "vendor").iterdir()) and (c005_work / "vendor-cache").is_dir():
    subprocess.run(
      ["cp", "-a", f"{c005_work / 'vendor-cache'}/.", str(BAZEL_CACHE / "vendor")],
      check=True,
    )
  if not any((BAZEL_CACHE / "repo").iterdir()) and (c005_work / "repo-cache").is_dir():
    subprocess.run(
      ["cp", "-a", f"{c005_work / 'repo-cache'}/.", str(BAZEL_CACHE / "repo")],
      check=True,
    )


def _docker_available() -> bool:
  if shutil.which("docker") is None:
    return False
  probe = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
  return probe.returncode == 0


def _image_tag() -> str:
  return str(json.loads(IMAGE_LOCK.read_text())["tag"])


@pytest.fixture(scope="module")
def w2_c026_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  if not _docker_available():
    pytest.skip("docker daemon unavailable")

  _ensure_bazel_cache()
  base = tmp_path_factory.mktemp("w2_c026")
  work = base / "work"
  out_dir = base / "out"
  out_dir.mkdir(parents=True, exist_ok=True)

  mat = subprocess.run(
    [
      sys.executable,
      str(SOURCES_TOOL),
      "materialize",
      "--lock",
      str(LOCK_FILE),
      "--out",
      str(work),
      "--sources",
      "coralnpu",
    ],
    capture_output=True,
    text=True,
    check=False,
  )
  assert mat.returncode == 0, mat.stderr

  green_targets = [
    "//hdl/chisel/src/coralnpu:w2_02_rob_owner_spec",
    "//hdl/chisel/src/coralnpu:w2_02_rf_bank_spec",
    "//hdl/chisel/src/coralnpu:w2_02_csr_flag_delta_spec",
    "//hdl/chisel/src/coralnpu:coralnpu_alu_tests",
    "//hdl/chisel/src/coralnpu:coralnpu_mlu_tests",
    "//hdl/chisel/src/coralnpu:coralnpu_fregfile_tests",
  ]
  green_str = " ".join(green_targets)

  in_container_script = f"""
set -uo pipefail
export GIT_CEILING_DIRECTORIES=/work
cd /work/coralnpu

BZL="bazel --output_user_root=/tmp/ob"
COMMON="-c opt --disk_cache=/bcache/disk --repository_cache=/bcache/repo \\
  --vendor_dir=/bcache/vendor --symlink_prefix=/tmp/bazel- --test_output=errors"

# 1. RED mutant #1: revive legacy store matching without OwnerTag
git apply --whitespace=nowarn /mutants/0202-pc-key-store.patch
$BZL test $COMMON //hdl/chisel/src/coralnpu:w2_02_rob_owner_spec > /out/red1.log 2>&1
rc_red1=$?
git apply -R --whitespace=nowarn /mutants/0202-pc-key-store.patch

# 2. RED mutant #2: tie MLU/LSU return OwnerTag to constant 0
git apply --whitespace=nowarn /mutants/0203-tie-owner-const.patch
$BZL test $COMMON //hdl/chisel/src/coralnpu:w2_02_rf_bank_spec > /out/red2.log 2>&1
rc_red2=$?
git apply -R --whitespace=nowarn /mutants/0203-tie-owner-const.patch

# 3. GREEN full C026 specs + Core c1/c2 integration + upstream scalar regressions
$BZL test $COMMON {green_str} > /out/green.log 2>&1
rc_green=$?

printf '{{"red1_rc": %d, "red2_rc": %d, "green_rc": %d}}\\n' \\
  "$rc_red1" "$rc_red2" "$rc_green" > /out/rc.json
"""

  cmd = [
    "docker",
    "run",
    "--rm",
    "--network=none",
    "--user",
    f"{os.getuid()}:{os.getgid()}",
    "-e",
    "HOME=/tmp",
    "-v",
    f"{BAZEL_CACHE.resolve()}:/bcache",
    "-v",
    f"{work.resolve()}:/work",
    "-v",
    f"{out_dir.resolve()}:/out",
    "-v",
    f"{MUT_PC_STORE.parent.resolve()}:/mutants:ro",
    _image_tag(),
    "bash",
    "-lc",
    in_container_script,
  ]
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
  assert (out_dir / "rc.json").is_file(), f"container failed early:\n{proc.stdout}\n{proc.stderr}"
  rcs = json.loads((out_dir / "rc.json").read_text())

  result = {
    "schema": "cn_w2_c026_result/v1",
    "negatives": {
      "0202_pc_key_store_killed": rcs["red1_rc"] != 0,
      "0202_pc_key_store_rc": rcs["red1_rc"],
      "0203_tie_owner_const_killed": rcs["red2_rc"] != 0,
      "0203_tie_owner_const_rc": rcs["red2_rc"],
    },
    "green": {
      "targets": green_targets,
      "rc": rcs["green_rc"],
      "passed": rcs["green_rc"] == 0,
    },
  }
  out_record = REPO_ROOT / "out/coral_nexus/C026/latest_unit_result.json"
  out_record.parent.mkdir(parents=True, exist_ok=True)
  out_record.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return {
    "result": result,
    "red1_out": (out_dir / "red1.log").read_text(),
    "red2_out": (out_dir / "red2.log").read_text(),
    "green_out": (out_dir / "green.log").read_text(),
  }


def test_sources_verify_single_owner_and_pristine_submodules(tmp_path: Path) -> None:
  proc = subprocess.run(
    ["bash", str(VERIFY_SH), "--lock", str(LOCK_FILE), "--out", str(tmp_path / "verify")],
    capture_output=True,
    text=True,
    check=False,
  )
  assert proc.returncode == 0, proc.stderr + proc.stdout
  report = json.loads((tmp_path / "verify/sources_report.json").read_text())
  assert report["pass"] is True
  by_name = {c["check"]: c for c in report["checks"]}
  assert by_name["patches.ownership"]["pass"] is True
  assert by_name["coralnpu.pristine"]["pass"] is True
  assert by_name["coralnpu-compiler.pristine"]["pass"] is True


def test_negative_mutants_fail_before_green_pass(w2_c026_evidence: dict[str, Any]) -> None:
  neg = w2_c026_evidence["result"]["negatives"]
  assert neg["0202_pc_key_store_killed"] is True, (
    f"Mutant 0202-pc-key-store survived!\n{w2_c026_evidence['red1_out'][-1500:]}"
  )
  assert neg["0203_tie_owner_const_killed"] is True, (
    f"Mutant 0203-tie-owner-const survived!\n{w2_c026_evidence['red2_out'][-1500:]}"
  )


def test_w2_02_chiselsim_specs_and_core_integration_pass(w2_c026_evidence: dict[str, Any]) -> None:
  green = w2_c026_evidence["result"]["green"]
  assert green["passed"] is True, (
    f"C026 ChiselSim / Core integration failed (rc={green['rc']}):\n"
    f"{w2_c026_evidence['green_out'][-2500:]}"
  )
  for target in green["targets"]:
    assert f"{target} " in w2_c026_evidence["green_out"] and "PASSED" in w2_c026_evidence["green_out"]
