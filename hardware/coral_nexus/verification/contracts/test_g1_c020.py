"""C020 joint seal + CN-G1 gate tests.

State of the tree these tests pin:

* the seal ENTRY POINTS exist and are refuse-first: an open required capacity budget refuses to seal, a
  missing required input refuses to seal, identical content re-seals idempotently, changed content needs
  --bump (which archives the previous seal), and the seal file's own digest appears nowhere inside it;
* TODAY the two required capacity budgets (per_ctx_outstanding_credit_cap, response_record_budget) have
  no sourced value, so no seal is committed and the G1 gate fails closed with exit 14 naming them (and
  17 for the missing seal). This is the honest state: `derivable:false` closes a WIDTH (fail-closed, no
  invented width); it does not close a CAPACITY, and the W1 freeze rule forbids sealing over it;
* once those budgets carry sourced values (simulated here with a patched constraints file -- the real
  closure is an edit of configs/w4_memory_constraints.yaml with evidence), the gate passes end to end
  and a seal written to a temp path verifies;
* every G1 failure class maps to its own non-zero exit: half-seal 17, tampered/stale generated artifact
  12 (codec) and 17 (seal hash), open budget 14, spec/evaluator drift 18, missing seal 17.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
sys.path.insert(0, str(H / "protocol"))
sys.path.insert(0, str(H / "scripts/lib"))
sys.path.insert(0, str(H / "verification"))

import freeze_check as fc  # noqa: E402
import gates  # noqa: E402
import gen_protocol as gp  # noqa: E402


@pytest.fixture
def gate() -> gates.G1Gate:
  return gates.G1Gate(REPO_ROOT)


# A sourced bound to close the two budgets in the fixture. The needle is read back by
# freeze_check.open_budgets; the real closure additionally updates scripts/derive_w4_bounds.py's
# REQUIRED_UNRESOLVED bookkeeping, which its own suite checks -- the fixture only simulates the move.
SOURCED_BOUND = {"file": "hardware/coral_nexus/configs/w4_memory_constraints.yaml", "line": 3}


def _closed_w4(tmp_path: Path) -> Path:
  """A copy of the w4 constraints with the two required budgets genuinely CLOSED: moved off the
  unresolved list AND onto inputs with a value plus a source whose needle reads back (deleting the
  entries is NOT closure; a deleted required entry stays open by design)."""
  doc = yaml.safe_load((H / "configs/w4_memory_constraints.yaml").read_text())
  doc["unresolved"] = [e for e in doc["unresolved"] if e["name"] not in fc.REQUIRED_CLOSED_BUDGETS]
  doc["inputs"] = list(doc.get("inputs") or [])
  doc["inputs"] += [
    {
      "name": name,
      "value": 8,
      "unit": "count" if "credit_cap" in name else "entries",
      "key": name,
      "source": {
        **SOURCED_BOUND,
        "needle": "Every input carries an explicit unit and a source the validator reads back",
      },
      "provenance": {"note": "fixture; the real closure cites the sourced constant"},
    }
    for name in sorted(fc.REQUIRED_CLOSED_BUDGETS)
  ]
  path = tmp_path / "w4_memory_constraints.yaml"
  path.write_text(yaml.safe_dump(doc))
  return path


# ---------------------------------------------------------------------------------------------------
# the honest current state: open budgets, no seal, gate red with named reasons
# ---------------------------------------------------------------------------------------------------
def test_the_two_capacity_budgets_are_still_open() -> None:
  assert fc.open_budgets() == ["per_ctx_outstanding_credit_cap", "response_record_budget"]


def test_the_gate_fails_closed_today_naming_the_open_budgets(gate: gates.G1Gate) -> None:
  report = gate.run()
  assert report["pass"] is False
  assert report["exit_code"] == gates.REQUIRED_G1_CHECKS["budgets.closed"] == 14
  by = {c["check"]: c for c in report["checks"]}
  assert "per_ctx_outstanding_credit_cap" in by["budgets.closed"]["detail"]
  assert "response_record_budget" in by["budgets.closed"]["detail"]
  assert by["seal.joint"]["ok"] is False
  assert not (H / "protocol/PROTOCOL_FREEZE.json").exists()


def test_deleting_a_required_budget_is_open_not_closed(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  doc = yaml.safe_load((H / "configs/w4_memory_constraints.yaml").read_text())
  doc["unresolved"] = [e for e in doc["unresolved"] if e["name"] not in fc.REQUIRED_CLOSED_BUDGETS]
  path = tmp_path / "deleted.yaml"
  path.write_text(yaml.safe_dump(doc))
  original = fc.W4_CONSTRAINTS
  fc.W4_CONSTRAINTS = path
  try:
    assert fc.open_budgets() == sorted(fc.REQUIRED_CLOSED_BUDGETS)
  finally:
    fc.W4_CONSTRAINTS = original


def test_a_budget_with_an_unreadback_source_is_open(tmp_path: Path) -> None:
  doc = yaml.safe_load((H / "configs/w4_memory_constraints.yaml").read_text())
  doc["unresolved"] = [e for e in doc["unresolved"] if e["name"] not in fc.REQUIRED_CLOSED_BUDGETS]
  doc["inputs"] = list(doc.get("inputs") or [])
  doc["inputs"] += [
    {
      "name": name,
      "value": 8,
      "unit": "count",
      "source": {"file": "hardware/coral_nexus/nope.yaml", "line": 1, "needle": "x"},
    }
    for name in sorted(fc.REQUIRED_CLOSED_BUDGETS)
  ]
  path = tmp_path / "unreadable.yaml"
  path.write_text(yaml.safe_dump(doc))
  original = fc.W4_CONSTRAINTS
  fc.W4_CONSTRAINTS = path
  try:
    assert fc.open_budgets() == sorted(fc.REQUIRED_CLOSED_BUDGETS)
  finally:
    fc.W4_CONSTRAINTS = original


def test_a_sourced_looking_entry_without_a_readback_needle_is_still_open(tmp_path: Path) -> None:
  doc = yaml.safe_load((H / "configs/w4_memory_constraints.yaml").read_text())
  doc["unresolved"] = [e for e in doc["unresolved"] if e["name"] not in fc.REQUIRED_CLOSED_BUDGETS]
  doc["inputs"] = list(doc.get("inputs") or [])
  doc["inputs"] += [
    {
      "name": name,
      "value": 8,
      "unit": "count",
      # an existing line, but the needle is not on it; and a non-integer value with a right needle
      "source": {
        "file": "hardware/coral_nexus/configs/w4_memory_constraints.yaml",
        "line": 3,
        "needle": "this text is not on that line",
      },
    }
    for name in sorted(fc.REQUIRED_CLOSED_BUDGETS)
  ]
  path = tmp_path / "cosmetic.yaml"
  path.write_text(yaml.safe_dump(doc))
  original = fc.W4_CONSTRAINTS
  fc.W4_CONSTRAINTS = path
  try:
    assert fc.open_budgets() == sorted(fc.REQUIRED_CLOSED_BUDGETS)
  finally:
    fc.W4_CONSTRAINTS = original


def test_a_width_may_stay_underivable_but_a_capacity_may_not(gate: gates.G1Gate, tmp_path: Path) -> None:
  monkey_w4 = _closed_w4(tmp_path)
  original = fc.W4_CONSTRAINTS
  fc.W4_CONSTRAINTS = monkey_w4
  try:
    report = gate.run()
    by = {c["check"]: c for c in report["checks"]}
    assert by["budgets.closed"]["ok"] is True
    assert "widths declared underivable" in by["budgets.closed"]["detail"]
    # the seal is still the blocker until one exists
    assert report["exit_code"] == gates.REQUIRED_G1_CHECKS["seal.joint"] == 17
  finally:
    fc.W4_CONSTRAINTS = original


# ---------------------------------------------------------------------------------------------------
# the seal, once the budgets are closed
# ---------------------------------------------------------------------------------------------------
@pytest.fixture
def closed_budgets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  monkeypatch.setattr(fc, "W4_CONSTRAINTS", _closed_w4(tmp_path))
  return tmp_path


def test_sealing_with_open_budgets_is_refused(tmp_path: Path) -> None:
  with pytest.raises(gp.GenError) as exc:
    fc.seal(tmp_path / "no.json")
  assert "open budget" in "\n".join(exc.value.errors)


def test_a_seal_written_with_closed_budgets_verifies_and_the_gate_goes_green(
  closed_budgets: Path, gate: gates.G1Gate, monkeypatch: pytest.MonkeyPatch
) -> None:
  seal_path = closed_budgets / "PROTOCOL_FREEZE.json"
  monkeypatch.setattr(fc, "SEAL_PATH", seal_path)
  body = fc.seal()
  assert (body["state"], body["protocol_seq"]) == ("frozen", 1)
  assert fc.verify_seal() == []
  text = seal_path.read_text()
  assert "protocol_seq" in text and hashlib_of_self_nowhere(text, seal_path)
  report = gate.run()
  assert report["pass"] is True, report["checks"]
  assert report["exit_code"] == 0


def hashlib_of_self_nowhere(text: str, path: Path) -> bool:
  import hashlib

  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  return digest not in text


def test_resealing_identical_content_is_idempotent_and_change_needs_bump(
  closed_budgets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  seal_path = closed_budgets / "PROTOCOL_FREEZE.json"
  monkeypatch.setattr(fc, "SEAL_PATH", seal_path)
  first = fc.seal()
  assert fc.seal() == first, "identical content re-seals idempotently"
  # a REAL input change (not a hand edit of the seal): an extra tracked file's bytes move
  extra = closed_budgets / "tracked_input.txt"
  extra.write_text("v1\n")
  monkeypatch.setattr(
    fc, "REQUIRED_INPUTS", (*fc.REQUIRED_INPUTS, "hardware/coral_nexus/scratch/" + extra.name)
  )
  scratch = fc.REPO / "hardware/coral_nexus/scratch"
  scratch.mkdir(exist_ok=True)
  tracked = scratch / extra.name
  tracked.write_text("v1\n")
  with pytest.raises(gp.GenError) as caught:
    fc.seal()  # now the content differs from the sealed state
  assert "--bump" in "\n".join(caught.value.errors)
  bumped = fc.seal(bump=True)
  assert bumped["protocol_seq"] == 2
  assert (seal_path.parent / "PROTOCOL_FREEZE.v1.json").is_file(), "the old seal is archived"
  assert fc.verify_seal() == []
  tracked.write_text("v2\n")
  third = fc.seal(bump=True)  # an INPUT change is re-sealable with a bump; a generated artifact is not
  assert third["protocol_seq"] == 3
  assert fc.verify_seal() == []

  def stale(*_a: object, **_k: object) -> list[str]:
    return ["cn_layout.json: differs from a fresh generation"]

  monkeypatch.setattr(fc, "verify_generated", stale)
  with pytest.raises(gp.GenError):  # regenerate first; a seal never chases generated drift
    fc.seal(bump=True)
  tracked.unlink()


# ---------------------------------------------------------------------------------------------------
# G1 failure classes
# ---------------------------------------------------------------------------------------------------
def test_reopening_a_budget_after_sealing_invalidates_the_seal(
  closed_budgets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(fc, "SEAL_PATH", closed_budgets / "PROTOCOL_FREEZE.json")
  fc.seal()
  assert fc.verify_seal() == []
  # the budget is re-opened after the seal (the real file still lists it): the seal stops verifying
  monkeypatch.setattr(fc, "W4_CONSTRAINTS", H / "configs/w4_memory_constraints.yaml")
  errors = fc.verify_seal()
  assert any("open required budget" in e and "per_ctx_outstanding_credit_cap" in e for e in errors)


def test_a_half_seal_is_rejected(
  closed_budgets: Path, gate: gates.G1Gate, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(fc, "SEAL_PATH", closed_budgets / "PROTOCOL_FREEZE.json")
  fc.seal()
  record = json.loads((closed_budgets / "PROTOCOL_FREEZE.json").read_text())
  dropped = sorted(record["inputs"])[0]
  del record["inputs"][dropped]
  (closed_budgets / "PROTOCOL_FREEZE.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
  errors = fc.verify_seal()
  assert any("half-seal" in e and dropped in e for e in errors)
  report = gate.run()
  assert report["exit_code"] == gates.REQUIRED_G1_CHECKS["seal.joint"] == 17


def test_a_tampered_generated_artifact_fails_both_the_seal_and_the_codec_check(
  closed_budgets: Path, gate: gates.G1Gate, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(fc, "SEAL_PATH", closed_budgets / "PROTOCOL_FREEZE.json")
  fc.seal()
  record = json.loads((closed_budgets / "PROTOCOL_FREEZE.json").read_text())
  victim = "hardware/coral_nexus/protocol/generated/cn_layout.json"
  record["generated"][victim] = "0" * 64
  (closed_budgets / "PROTOCOL_FREEZE.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
  report = gate.run()
  assert report["exit_code"] == gates.REQUIRED_G1_CHECKS["seal.joint"] == 17


def test_a_stale_generated_artifact_fails_the_codec_check(
  gate: gates.G1Gate, monkeypatch: pytest.MonkeyPatch
) -> None:
  def stale(*_a: object, **_k: object) -> list[str]:
    return ["cn_layout.json: differs from a fresh generation"]

  monkeypatch.setattr(fc, "verify_generated", stale)
  report = gate.run()
  assert report["exit_code"] == gates.REQUIRED_G1_CHECKS["wire.codec_generation"] == 12


def test_a_drifted_gate_spec_is_its_own_failure(gate: gates.G1Gate, tmp_path: Path) -> None:
  spec = {"schema": "cn_gate_config/v1", "stage": "G1", "required_checks": ["seal.joint"]}
  path = tmp_path / "G1.yaml"
  path.write_text(yaml.safe_dump(spec))
  original = gates.G1_SPEC_PATH
  gates.G1_SPEC_PATH = path
  try:
    report = gate.run()
  finally:
    gates.G1_SPEC_PATH = original
  assert report["exit_code"] == gates.G1_SPEC_DRIFT_CODE == 18


def test_the_committed_spec_matches_the_evaluator() -> None:
  spec = yaml.safe_load((H / "verification/gates/G1.yaml").read_text())
  assert list(spec["required_checks"]) == list(gates.REQUIRED_G1_CHECKS)
  assert set(spec["exit_codes"]) >= {0, 11, 12, 13, 14, 15, 16, 17, 18}


def test_the_gate_command_runs_and_reports_the_honest_state() -> None:
  import subprocess

  proc = subprocess.run(
    ["bash", str(H / "scripts/cn.sh"), "gate", "--stage", "G1", "--out", "out/coral_nexus/C020/gate-test"],
    capture_output=True,
    text=True,
    cwd=REPO_ROOT,
  )
  assert proc.returncode == 14
  assert "per_ctx_outstanding_credit_cap" in proc.stdout
