"""C019 CS1 target generation tests.

What is proven here:

* the committed cs1_c1/cs1_c2 targets, params.d contribution and rtl/targets.bzl equal a fresh render
  from the SOURCES (decisions read back from their documents, params.d envelope, W5 capacities, the
  committed C018 measurement record) -- a hand edit or a stale file is an error;
* C1 and C2 differ ONLY in context_count (same-RTL rule), C1 still indexes its single Context with >= 1 bit
  (no zero-width index), and both fit the measured envelope (slots x text <= ITCM, stack <= DTCM);
* the two targets carry distinct constraints fingerprints, and rtl/targets.bzl carries the same ones;
* `cn build --prepare-only` really runs: it re-checks generation, validates the target through the
  C008 consumer, compiles a GENERATED C consumer with the pinned RV32 toolchain and claims no
  executability;
  claims no executability; kind p0_upstream is refused, and a missing prepare-only subcommand is a usage
  error;
* generation refuses: a missing/stale measurement, a non-static bound, an envelope that does not fit, fewer
  slots than residency min_coexisting, a decision whose needle no longer reads back, and a C1/C2 direction
  violation.

Toolchain-dependent legs (fresh measurement + the prepare-only compile) are skipped with the reason stated
when the pinned toolchain is absent; every refusal still runs without it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
sys.path.insert(0, str(H / "scripts/lib"))

import gen_targets as gt  # noqa: E402
import measure_kernel as mk  # noqa: E402
import validate_constraints as vc  # noqa: E402

HAVE_TOOLCHAIN = mk.TOOLCHAIN.is_dir()
MEASUREMENT = H / "verification/kernel_bootstrap/kernel_measurement.json"
CN = H / "scripts/cn.sh"


def _measurement(**override: Any) -> dict[str, Any]:
  doc = json.loads(MEASUREMENT.read_text())
  doc.update(override)
  return doc


def _write_measurement(tmp_path: Path, doc: dict[str, Any]) -> Path:
  path = tmp_path / "kernel_measurement.json"
  path.write_text(json.dumps(doc, indent=2))
  return path


def _expect(code: str, fn: Any, *args: Any) -> None:
  with pytest.raises(gt.TargetError) as exc:
    fn(*args)
  assert exc.value.code == code, exc.value


# ---------------------------------------------------------------------------------------------------
# committed artifacts
# ---------------------------------------------------------------------------------------------------
def test_committed_targets_equal_a_fresh_render() -> None:
  assert gt.check(MEASUREMENT) == []


@pytest.mark.skipif(not HAVE_TOOLCHAIN, reason="pinned C004 toolchain not extracted")
def test_the_committed_measurement_record_reproduces_from_the_sources(tmp_path: Path) -> None:
  record = mk.measure(tmp_path)
  committed = json.loads(MEASUREMENT.read_text())
  assert record["resource_manifest"] == committed["resource_manifest"]
  assert record["candidate_layouts"] == committed["candidate_layouts"]
  assert record["provenance"]["elf_sha256_layout_A"] == committed["provenance"]["elf_sha256_layout_A"]


def test_c1_and_c2_differ_only_in_context_count_and_index_their_contexts() -> None:
  c1 = yaml.safe_load((H / "configs/cs1_c1.yaml").read_text())
  c2 = yaml.safe_load((H / "configs/cs1_c2.yaml").read_text())
  assert (c1["kind"], c2["kind"]) == ("cs1_target", "cs1_target")
  assert (c1["constraints"]["context_count"], c2["constraints"]["context_count"]) == (1, 2)
  for key in set(c1["constraints"]) | set(c2["constraints"]):
    if key != "context_count":
      assert c1["constraints"][key] == c2["constraints"][key], key
  for target in (c1, c2):
    assert target["index_widths"]["ctx_slot_bits"] >= 1, "a zero-width index is forbidden (C1 included)"


def test_both_targets_fit_the_measured_envelope() -> None:
  measured = json.loads(MEASUREMENT.read_text())
  itcm = next(int(x) for x in [measured["capacities"]["itcm_bytes"]])
  dtcm = int(measured["capacities"]["dtcm_bytes"])
  slots = len(measured["candidate_layouts"])
  rm = measured["resource_manifest"]
  assert slots * rm["text_bytes"] <= itcm
  assert rm["stack_bytes"] <= dtcm
  for name in ("cs1_c1", "cs1_c2"):
    target = yaml.safe_load((H / f"configs/{name}.yaml").read_text())
    assert target["constraints"]["program_slots"] == slots
    assert target["envelope"]["measured"] == {
      k: rm[k] for k in ("text_bytes", "static_data_bytes", "stack_bytes", "scratch_bytes")
    }


def test_fingerprints_are_distinct_and_shared_with_the_bzl_registry() -> None:
  fingerprints = {}
  for name in ("cs1_c1", "cs1_c2"):
    target = yaml.safe_load((H / f"configs/{name}.yaml").read_text())
    fingerprints[name] = target["fingerprints"]["constraints_sha256"]
  assert fingerprints["cs1_c1"] != fingerprints["cs1_c2"]
  bzl = (H / "rtl/targets.bzl").read_text()
  for name, fp in fingerprints.items():
    assert f'"{name}": {{' in bzl
    assert fp in bzl
  assert sorted(re.findall(r'"(cs1_c\d)": \{', bzl)) == ["cs1_c1", "cs1_c2"]
  # the fingerprint covers the constraints only; no generated file may point at the seal
  for path in (H / "configs/cs1_c1.yaml", H / "configs/cs1_c2.yaml", H / "rtl/targets.bzl"):
    assert "PROTOCOL_FREEZE" not in path.read_text()


def test_the_params_contribution_merges_without_conflict_and_matches_the_targets() -> None:
  merged, errors = vc.load_params(H / "configs/params.d")
  assert errors == []
  for name in ("cs1_c1", "cs1_c2"):
    target = yaml.safe_load((H / f"configs/{name}.yaml").read_text())
    assert merged[f"{name}.context_count"]["value"] == target["constraints"]["context_count"]
  assert merged["cs1.program_slots"]["value"] == 2


# ---------------------------------------------------------------------------------------------------
# prepare-only
# ---------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_TOOLCHAIN, reason="pinned C004 toolchain not extracted")
@pytest.mark.parametrize("name", ["cs1_c1", "cs1_c2"])
def test_prepare_only_compiles_a_generated_consumer(tmp_path: Path, name: str) -> None:
  out = tmp_path / name
  proc = subprocess.run(
    ["bash", str(CN), "build", "--prepare-only", "--config", f"hardware/coral_nexus/configs/{name}.yaml",
     "--out", str(out)],
    capture_output=True, text=True, cwd=REPO_ROOT,
  )  # fmt: skip
  assert proc.returncode == 0, proc.stderr
  verdict = json.loads((out / "prepare_only_verdict.json").read_text())
  assert verdict["target"] == name
  assert verdict["consumer_compiled"] is True
  assert verdict["executable_claimed"] is False
  assert (out / f"{name}_consumer.o").is_file()
  header = (out / f"{name}_params.h").read_text()
  assert "CTX_SLOT_BITS 1" in header  # the never-zero index is visible to C consumers too


def test_p0_upstream_is_refused_and_a_missing_subcommand_is_usage() -> None:
  proc = subprocess.run(
    ["bash", str(CN), "build", "--prepare-only", "--config", "hardware/coral_nexus/configs/p0_coral.yaml",
     "--out", "out/coral_nexus/C019/refused"],
    capture_output=True, text=True, cwd=REPO_ROOT,
  )  # fmt: skip
  assert proc.returncode == 3
  assert "cs1_target" in proc.stderr
  proc = subprocess.run(
    ["bash", str(CN), "build", "--config", "hardware/coral_nexus/configs/cs1_c1.yaml"],
    capture_output=True,
    text=True,
    cwd=REPO_ROOT,
  )
  assert proc.returncode == 2
  assert shutil.which("bash") is not None


# ---------------------------------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------------------------------
def test_a_missing_measurement_is_refused(tmp_path: Path) -> None:
  _expect("MISSING_MEASUREMENT", gt.build, tmp_path / "nowhere.json")


def test_a_bound_that_is_not_static_analysis_is_refused(tmp_path: Path) -> None:
  doc = _measurement()
  doc["resource_manifest"]["bound_source"] = "measured_peak"
  _expect("BAD_MEASUREMENT", gt.build, _write_measurement(tmp_path, doc))


def test_an_envelope_that_does_not_fit_is_refused(tmp_path: Path) -> None:
  doc = _measurement()
  for layout in doc["candidate_layouts"]:
    layout["text_bytes"] = 6000  # the second slot runs past the 8 KiB ITCM end
  _expect("ENVELOPE", gt.build, _write_measurement(tmp_path, doc))
  doc = _measurement()
  doc["candidate_layouts"][0]["data_base"] = 32000  # data+stack exceeds DTCM, no overlap involved
  doc["candidate_layouts"][0]["data_bytes"] = 1000
  _expect("ENVELOPE", gt.build, _write_measurement(tmp_path, doc))
  doc = _measurement()
  first, second = doc["candidate_layouts"]
  first["data_base"] = second["data_base"]
  first["data_bytes"] = second["data_bytes"] = 64  # the two data ranges collide
  _expect("ENVELOPE", gt.build, _write_measurement(tmp_path, doc))
  doc = _measurement()
  doc["candidate_layouts"] = doc["candidate_layouts"][:1]
  _expect("BAD_MEASUREMENT", gt.build, _write_measurement(tmp_path, doc))


def test_fewer_slots_than_residency_min_coexisting_is_refused(tmp_path: Path) -> None:
  contract = yaml.safe_load((H / "protocol/w5_package.contract.yaml").read_text())
  contract["residency"]["slots"]["min_coexisting"] = 3
  contract_path = tmp_path / "w5_contract.yaml"
  contract_path.write_text(yaml.safe_dump(contract))
  _expect("ENVELOPE", gt.build, MEASUREMENT, contract_path)


def test_the_pair_rule_and_freeze_loop_guards_hold() -> None:
  base = {"context_count": 1, "tile_count": 1}
  gt._check_pair(base, {**base, "context_count": 2})
  _expect("CONFLICT", gt._check_pair, base, {**base, "context_count": 1})
  _expect("CONFLICT", gt._check_pair, base, {**base, "context_count": 2, "tile_count": 2})
  gt._reject_freeze_references({"x": "clean"})
  _expect("FREEZE_LOOP", gt._reject_freeze_references, {"x": "see PROTOCOL_FREEZE.json"})


def test_a_measurement_with_foreign_capacities_is_refused(tmp_path: Path) -> None:
  doc = _measurement()
  doc["capacities"]["itcm_bytes"] = 65536
  _expect("CONFLICT", gt.build, _write_measurement(tmp_path, doc))


def test_a_decision_needle_that_no_longer_reads_back_is_refused(tmp_path: Path, monkeypatch: Any) -> None:
  monkeypatch.setitem(gt.DECISIONS["cs1_c1"], "needle", ("design/coral_nexus_tile/plan.md", "cs1_c9"))
  _expect("MISSING_SOURCE", gt.build, MEASUREMENT)
  monkeypatch.setitem(
    gt.DECISIONS["cs1_c1"], "needle", ("design/coral_nexus_tile/does_not_exist.md", "cs1_c1")
  )
  _expect("MISSING_SOURCE", gt.build, MEASUREMENT)


def test_a_c1_c2_direction_violation_is_refused(tmp_path: Path, monkeypatch: Any) -> None:
  monkeypatch.setitem(gt.DECISIONS["cs1_c1"], "context_count", 2)
  _expect("CONFLICT", gt.build, MEASUREMENT)


def test_the_committed_record_stays_schema_clean() -> None:
  doc = json.loads(MEASUREMENT.read_text())
  assert doc["schema"] == "cn_kernel_measurement/v1"
  checks = {c["name"]: c["ok"] for c in doc["checks"]}
  assert checks == {
    "coexist_A_B": True,
    "itcm_capacity": True,
    "dtcm_capacity": True,
    "stack_reservation_covers_bound": True,
  }
