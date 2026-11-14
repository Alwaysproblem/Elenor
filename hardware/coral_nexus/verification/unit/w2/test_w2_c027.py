"""C027 RTL acceptance, isolated from the already-passing C026 suite.

Bazel build failures are not mutation kills: a negative run must exit 3 and
contain a ScalaTest behavioral assertion failure. Logs and JUnit reports stay
in the pytest temporary directory (use --basetemp to retain card evidence).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
H_ROOT = REPO_ROOT / "hardware/coral_nexus"
BAZEL_CACHE = REPO_ROOT / "out/coral_nexus/cache/w2_bazel"
ORACLE_ROOT = H_ROOT / "verification/oracle"
sys.path.insert(0, str(ORACLE_ROOT))

import oracle_main as om  # noqa: E402
import spike_ref_driver as srd  # noqa: E402

TARGET_PREFIX = "//hdl/chisel/src/coralnpu:"
TARGETS = (
  "w2_04_ordered_flags_spec",
  "w2_04_csr_config_spec",
  "w2_04_ctx_quiescent_spec",
  "w2_04_sched_park_spec",
  "w2_04_cancel_retire_spec",
  "w2_04_core_control_spec",
)
MUTANTS = (
  ("0402-disable-csr-interlock.patch", "w2_04_csr_config_spec"),
  ("0404-global-cancel.patch", "w2_04_cancel_retire_spec"),
  ("0202-pc-key-store.patch", "w2_04_cancel_retire_spec"),
  ("0404-shared-retire-head.patch", "w2_04_cancel_retire_spec"),
)
NATIVE_TOPS = ("tb_native_config", "tb_native_flags", "tb_native_matrix_tags")


def _test_cases(path: Path) -> list[ET.Element]:
  assert path.is_file(), f"missing Bazel JUnit report: {path}"
  cases = list(ET.parse(path).getroot().iter("testcase"))
  assert cases, f"no executed test cases in {path}"
  assert not any(case.find("skipped") is not None for case in cases), path
  return cases


@pytest.fixture(scope="module")
def c027_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  assert shutil.which("docker"), "C027 requires the locked Docker builder"
  base = tmp_path_factory.mktemp("c027")
  work = base / "work"
  evidence = base / "evidence"
  evidence.mkdir()
  freeze = H_ROOT / "protocol/PROTOCOL_FREEZE.json"
  freeze_hash = hashlib.sha256(freeze.read_bytes()).hexdigest()
  materialize = subprocess.run(
    [
      sys.executable,
      str(H_ROOT / "scripts/lib/sources_tool.py"),
      "materialize",
      "--lock",
      str(H_ROOT / "env/sources.lock"),
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
    assert (BAZEL_CACHE / name).is_dir(), f"missing offline Bazel cache {name}"
  image = json.loads((H_ROOT / "env/image.lock").read_text())["tag"]
  script = [
    "set -euo pipefail",
    "export GIT_CEILING_DIRECTORIES=/work",
    "cd /work/coralnpu",
    'BZL="bazel --output_user_root=/tmp/ob"',
    'COMMON="-c opt --disk_cache=/bcache/disk --repository_cache=/bcache/repo '
    '--vendor_dir=/bcache/vendor --symlink_prefix=/tmp/bazel- --test_output=all"',
  ]
  script.extend(
    [
      "RV=/work/coralnpu/hdl/verilog/rvv",
      "FP=/bcache/vendor/+coralnpu_deps_ext+cvfpu/src",
      "CC=/bcache/vendor/+coralnpu_deps_ext+common_cells",
      "$BZL build $COMMON @verilator//:verilator_bin > /out/verilator-tool.log 2>&1",
      "$BZL build $COMMON "
      + " ".join(TARGET_PREFIX + target for target in TARGETS)
      + " > /out/test-build.log 2>&1",
      "VEXE=(/tmp/bazel-bin/external/*/verilator_bin)",
      'test "${#VEXE[@]}" -eq 1',
      'export VERILATOR_ROOT="${VEXE[0]}.runfiles/$(basename "$(dirname "${VEXE[0]}")")"',
      'test -f "$VERILATOR_ROOT/include/verilated.mk"',
      "export VERILATOR_CXX=g++ VERILATOR_AR=ar VERILATOR_PYTHON3=python3",
    ]
  )
  for top in NATIVE_TOPS:
    script.extend(
      [
        "rc=0",
        f'"${{VEXE[0]}}" --binary --timing -j 4 -Wno-fatal --top-module {top} '
        f"--Mdir /out/obj-{top} -DVLEN_128 -DZVT_ON -DZVE32F_ON "
        '-I"$RV/inc" -I"$CC/include" -y "$RV/design" -y "$RV/design/Zvt" '
        '-y "$RV/common" -y "$CC/src" +libext+.sv '
        f'"$CC/src/cf_math_pkg.sv" "$FP/fpnew_pkg.sv" /native/{top}.sv '
        f"> /out/{top}.build.log 2>&1 || rc=$?",
        f'printf "%s\\n" "$rc" > /out/{top}.build.rc',
        f"if (( rc == 0 )); then timeout 180 /out/obj-{top}/V{top} > /out/{top}.log 2>&1 || rc=$?; fi",
        f'printf "%s\\n" "$rc" > /out/{top}.rc',
      ]
    )
  for index, (patch, target) in enumerate(MUTANTS):
    label = f"red-{index}"
    script.extend(
      [
        f"git apply --check /mutants/{shlex.quote(patch)}",
        f"git apply --whitespace=nowarn /mutants/{shlex.quote(patch)}",
        "rc=0",
        f"$BZL test $COMMON {TARGET_PREFIX}{target} > /out/{label}.log 2>&1 || rc=$?",
        f'printf "%s\\n" "$rc" > /out/{label}.rc',
        f"cp /tmp/bazel-testlogs/hdl/chisel/src/coralnpu/{target}/test.xml /out/{label}.xml",
        f"git apply -R --check /mutants/{shlex.quote(patch)}",
        f"git apply -R --whitespace=nowarn /mutants/{shlex.quote(patch)}",
      ]
    )
  labels = " ".join(TARGET_PREFIX + name for name in TARGETS)
  script.extend(
    [
      "rc=0",
      f"$BZL test $COMMON {labels} > /out/green.log 2>&1 || rc=$?",
      'printf "%s\\n" "$rc" > /out/green.rc',
    ]
  )
  for target in TARGETS:
    script.append(f"cp /tmp/bazel-testlogs/hdl/chisel/src/coralnpu/{target}/test.xml /out/{target}.xml")
    script.append(f"cp /tmp/bazel-testlogs/hdl/chisel/src/coralnpu/{target}/test.log /out/{target}.log")
  script.extend(
    [
      "rc=0",
      "$BZL build -c opt --disk_cache=/bcache/disk --repository_cache=/bcache/repo "
      "--vendor_dir=/bcache/vendor --symlink_prefix=/tmp/bazel- "
      f"{TARGET_PREFIX}core_mini_verification_axi_cc_library_emit_verilog "
      "> /out/coreaxi.log 2>&1 || rc=$?",
      'printf "%s\\n" "$rc" > /out/coreaxi.rc',
    ]
  )
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
    f"{BAZEL_CACHE.resolve()}:/bcache",
    "-v",
    f"{work.resolve()}:/work",
    "-v",
    f"{evidence.resolve()}:/out",
    "-v",
    f"{H_ROOT / 'patches/w2/mutants'}:/mutants:ro",
    "-v",
    f"{H_ROOT / 'verification/unit/w2'}:/native:ro",
    image,
    "bash",
    "-lc",
    "\n".join(script),
  ]
  run = subprocess.run(command, capture_output=True, text=True, check=False)
  (evidence / "runner.log").write_text(run.stdout + run.stderr)
  assert run.returncode == 0, f"C027 runner failed; evidence={evidence}\n{run.stderr}"
  assert hashlib.sha256(freeze.read_bytes()).hexdigest() == freeze_hash, "C020 seal changed"
  return {"path": evidence, "seal_sha256": freeze_hash}


def test_c027_mutants_fail_on_behavior(c027_evidence: dict[str, Any]) -> None:
  evidence = c027_evidence["path"]
  for index, (patch, _target) in enumerate(MUTANTS):
    label = f"red-{index}"
    rc = int((evidence / f"{label}.rc").read_text())
    log = (evidence / f"{label}.log").read_text()
    assert rc == 3, f"{patch}: expected Bazel test failure, not build/tool failure ({rc})\n{log[-4000:]}"
    assert "*** FAILED ***" in log, f"{patch}: no ScalaTest failure"
    assert (
      "FailedExpectationException" in log
      or "TestFailedException" in log
      or re.search(r"\*\*\* FAILED \*\*\*[^\n]*\n[^\n]*(?:did not equal|was not|failed)", log)
    ), f"{patch}: failure is not a behavioral assertion\n{log[-4000:]}"
    cases = _test_cases(evidence / f"{label}.xml")
    assert any(case.find("failure") is not None for case in cases), patch


def test_c027_all_registered_rtl_cases_pass(c027_evidence: dict[str, Any]) -> None:
  evidence = c027_evidence["path"]
  log = (evidence / "green.log").read_text()
  assert int((evidence / "green.rc").read_text()) == 0, log[-8000:]
  for target in TARGETS:
    assert re.search(rf"^{re.escape(TARGET_PREFIX + target)}\s+.*\bPASSED\b", log, re.M), target
    target_log = (evidence / f"{target}.log").read_text()
    executed = re.search(r"Total number of tests run: (\d+)", target_log)
    assert executed and int(executed[1]) > 0, f"{target}: ScalaTest executed no cases"
    assert "All tests passed." in target_log, target_log[-3000:]
    for case in _test_cases(evidence / f"{target}.xml"):
      assert case.find("failure") is None and case.find("error") is None, (target, case.attrib)
  assert int((evidence / "coreaxi.rc").read_text()) == 0, (evidence / "coreaxi.log").read_text()[-6000:]
  for top in NATIVE_TOPS:
    assert int((evidence / f"{top}.build.rc").read_text()) == 0, (
      evidence / f"{top}.build.log"
    ).read_text()[-8000:]
    assert int((evidence / f"{top}.rc").read_text()) == 0, (evidence / f"{top}.log").read_text()[-8000:]
    assert f"{top.upper()} ALL PASS" in (evidence / f"{top}.log").read_text()
  (evidence / "result.json").write_text(
    json.dumps(
      {
        "card": "C027",
        "targets": TARGETS,
        "native_tops": NATIVE_TOPS,
        "seal_sha256": c027_evidence["seal_sha256"],
      },
      indent=2,
    )
    + "\n"
  )


def test_c027_cancel_retirement_matches_independent_spike(c027_evidence: dict[str, Any]) -> None:
  evidence = c027_evidence["path"]
  for case in _test_cases(evidence / "w2_04_cancel_retire_spec.xml"):
    assert case.find("failure") is None and case.find("error") is None, case.attrib
  log = (evidence / "w2_04_cancel_retire_spec.log").read_text()
  actual = [json.loads(match[1]) for match in re.finditer(r"^C027_RETIRE (\{[^\n]+\})$", log, re.M)]
  # No timestamp alignment, deduplication or PC-only pairing.
  assert [(row["ctx_slot"], row["owner_op_seq"]) for row in actual] == [(1, 30), (1, 31), (0, 20), (0, 21)]
  oracle_dir = evidence / "cancel-oracle"
  oracle_dir.mkdir()
  assembly = oracle_dir / "retire.S"
  linker = oracle_dir / "retire.ld"
  assembly.write_text(
    '.option norvc\n.section .text.a,"ax"\n.global _start\n_start:\n'
    "sw x5,0(x1)\naddi x5,x0,7\n"
    '.section .text.b,"ax"\naddi x5,x0,7\naddi x5,x0,7\n'
  )
  linker.write_text(
    "OUTPUT_ARCH(riscv)\nENTRY(_start)\nSECTIONS {\n"
    " .text.a 0x1000 : { *(.text.a) }\n .text.b 0x2000 : { *(.text.b) }\n}\n"
  )
  elf = om.build_asm_elf(assembly, linker, oracle_dir / "retire.elf")
  fields = (
    "ctx_slot",
    "ctx_generation",
    "owner_op_seq",
    "pc_rdata",
    "insn",
    "x_rd",
    "x_rd_val",
    "trap",
    "flags_delta",
  )
  references = []
  for ctx, pc, seq_base in ((0, 0x1000, 20), (1, 0x2000, 30)):
    trace = srd.run_trace_request(
      srd.TraceRequest(
        elf_path=str(elf),
        entry_pc=pc,
        ctx_slot=ctx,
        ctx_generation=0,
        max_steps=2,
        memory_regions=[(0, 0x10000), (0x10000, 0x10000)],
        initial_gpr={"x1": 0x10000, "x5": 7},
      )
    )
    assert trace["steps_executed"] == 2 and trace["final_pc"] == pc + 8
    expected = []
    for reference in trace["retire_stream"]:
      assert reference["source"] == "spike_isa_ref"
      row = {field: reference[field] for field in fields}
      row["owner_op_seq"] += seq_base
      expected.append(row)
    assert [row for row in actual if row["ctx_slot"] == ctx] == expected
    references.append(trace)
  (oracle_dir / "comparison.json").write_text(
    json.dumps({"actual": actual, "independent_references": references}, indent=2) + "\n"
  )
