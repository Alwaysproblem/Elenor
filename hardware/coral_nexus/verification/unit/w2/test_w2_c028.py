"""C028 acceptance: real frontend and Core behavior, RED mutations before GREEN.

Run this file directly for C028 without rerunning C026 or C027. Retain --basetemp
under a fresh card attempt so materialization, Scala/JUnit and mutation evidence survive.
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

REPO = Path(__file__).resolve().parents[5]
H = REPO / "hardware/coral_nexus"
sys.path.insert(0, str(H / "scripts/lib"))

import gen_frontend_params as frontend_params  # noqa: E402

CACHE = REPO / "out/coral_nexus/cache/w2_bazel"
PREFIX = "//hdl/chisel/src/coralnpu:"
TARGETS = ("w2_03_fetch_epoch_spec", "w2_03_dispatch_prefix_spec", "w2_03_core_issue_spec")
MUTANTS = (
  ("0303-pc-only-fetch.patch", "w2_03_fetch_epoch_spec", "same PC identity"),
  ("0302-nonprefix-dispatch.patch", "w2_03_dispatch_prefix_spec", "Partial accept holds"),
)


def _cases(path: Path) -> list[ET.Element]:
  assert path.is_file(), f"missing executed JUnit evidence: {path}"
  cases = list(ET.parse(path).getroot().iter("testcase"))
  assert cases and not any(case.find("skipped") is not None for case in cases), path
  return cases


@pytest.fixture(scope="module")
def c028_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  assert shutil.which("docker"), "C028 requires the locked Docker builder"
  frontend_params.check()
  base = tmp_path_factory.mktemp("c028")
  evidence = base / "evidence"
  evidence.mkdir()
  work = base / "work"
  seal = H / "protocol/PROTOCOL_FREEZE.json"
  seal_hash = hashlib.sha256(seal.read_bytes()).hexdigest()
  materialize = subprocess.run(
    [sys.executable, str(H / "scripts/lib/sources_tool.py"), "materialize", "--lock",
     str(H / "env/sources.lock"), "--sources", "coralnpu", "--out", str(work)],
    capture_output=True, text=True, check=False,
  )  # fmt: skip
  (evidence / "materialize.log").write_text(materialize.stdout + materialize.stderr)
  assert materialize.returncode == 0, materialize.stderr
  for directory in ("disk", "repo", "vendor"):
    assert (CACHE / directory).is_dir(), f"missing offline Bazel cache: {directory}"
  labels = " ".join(PREFIX + target for target in TARGETS)
  script = [
    "set -euo pipefail",
    "export GIT_CEILING_DIRECTORIES=/work",
    "cd /work/coralnpu",
    'BZL="bazel --output_user_root=/tmp/ob"',
    'COMMON="-c opt --disk_cache=/bcache/disk --repository_cache=/bcache/repo '
    '--vendor_dir=/bcache/vendor --symlink_prefix=/tmp/bazel- --test_output=all"',
    f"$BZL build $COMMON {labels} > /out/test-build.log 2>&1",
  ]
  for index, (patch, target, case_filter) in enumerate(MUTANTS):
    mutation = shlex.quote("/mutants/" + patch)
    script.extend(
      [
        f"git apply --check {mutation}",
        f"git apply --whitespace=nowarn {mutation}",
        "rc=0",
        f"$BZL test $COMMON --test_arg=-z --test_arg={shlex.quote(case_filter)} "
        f"{PREFIX}{target} > /out/red-{index}.log 2>&1 || rc=$?",
        f'printf "%s\\n" "$rc" > /out/red-{index}.rc',
        f"cp /tmp/bazel-testlogs/hdl/chisel/src/coralnpu/{target}/test.xml /out/red-{index}.xml",
        f"git apply -R --check {mutation}",
        f"git apply -R --whitespace=nowarn {mutation}",
      ]
    )
  script.extend(
    [
      "rc=0",
      f"$BZL test $COMMON {labels} > /out/green.log 2>&1 || rc=$?",
      'printf "%s\\n" "$rc" > /out/green.rc',
    ]
  )
  for target in TARGETS:
    for suffix in ("xml", "log"):
      script.append(
        f"cp /tmp/bazel-testlogs/hdl/chisel/src/coralnpu/{target}/test.{suffix} /out/{target}.{suffix}"
      )
  script.extend(
    [
      "rc=0",
      f"$BZL build $COMMON {PREFIX}core_mini_verification_axi_cc_library_emit_verilog "
      "> /out/coreaxi.log 2>&1 || rc=$?",
      'printf "%s\\n" "$rc" > /out/coreaxi.rc',
    ]
  )
  image = json.loads((H / "env/image.lock").read_text())["tag"]
  command = [
    "docker", "run", "--rm", "--network=none", "--user", f"{os.getuid()}:{os.getgid()}",
    "-e", "HOME=/tmp", "-v", f"{CACHE.resolve()}:/bcache", "-v", f"{work.resolve()}:/work",
    "-v", f"{evidence.resolve()}:/out", "-v", f"{H / 'patches/w2/mutants'}:/mutants:ro",
    image, "bash", "-lc", "\n".join(script),
  ]  # fmt: skip
  run = subprocess.run(command, capture_output=True, text=True, check=False)
  (evidence / "runner.log").write_text(run.stdout + run.stderr)
  assert run.returncode == 0, f"C028 runner failed: {evidence}\n{run.stderr}"
  assert hashlib.sha256(seal.read_bytes()).hexdigest() == seal_hash, "joint seal changed during C028"
  return {"path": evidence, "seal_sha256": seal_hash}


def test_c028_mutants_fail_on_behavior(c028_evidence: dict[str, Any]) -> None:
  evidence = c028_evidence["path"]
  for index, (patch, _target, _case_filter) in enumerate(MUTANTS):
    label = f"red-{index}"
    rc = int((evidence / f"{label}.rc").read_text())
    log = (evidence / f"{label}.log").read_text()
    assert rc == 3, f"{patch}: not a Bazel test failure ({rc})\n{log[-6000:]}"
    assert "*** FAILED ***" in log, f"{patch}: no executed Scala failure"
    assert (
      "FailedExpectationException" in log
      or "TestFailedException" in log
      or re.search(r"\*\*\* FAILED \*\*\*[^\n]*\n[^\n]*(?:did not equal|was not|failed)", log)
    ), f"{patch}: no behavioral assertion failure\n{log[-6000:]}"
    assert any(case.find("failure") is not None for case in _cases(evidence / f"{label}.xml")), patch


def test_c028_frontend_and_core_behavior_pass(c028_evidence: dict[str, Any]) -> None:
  evidence = c028_evidence["path"]
  green = (evidence / "green.log").read_text()
  assert int((evidence / "green.rc").read_text()) == 0, green[-10000:]
  counts = {}
  for target in TARGETS:
    assert re.search(rf"^{re.escape(PREFIX + target)}\s+.*\bPASSED\b", green, re.M), target
    log = (evidence / f"{target}.log").read_text()
    count = re.search(r"Total number of tests run: (\d+)", log)
    assert count and int(count[1]) > 0 and "All tests passed." in log, log[-6000:]
    counts[target] = int(count[1])
    for case in _cases(evidence / f"{target}.xml"):
      assert case.find("failure") is None and case.find("error") is None, case.attrib
  assert int((evidence / "coreaxi.rc").read_text()) == 0, (evidence / "coreaxi.log").read_text()[-8000:]
  (evidence / "result.json").write_text(
    json.dumps({"card": "C028", "targets": counts, "seal_sha256": c028_evidence["seal_sha256"]}, indent=2)
    + "\n"
  )
