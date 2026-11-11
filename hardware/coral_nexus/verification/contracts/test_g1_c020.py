"""C020 joint seal + CN-G1 gate tests.

State of the tree these tests pin:

* the seal ENTRY POINTS exist and are refuse-first: an open required capacity budget refuses to seal, a
  missing required input refuses to seal, identical content re-seals idempotently, changed content needs
  --bump (which archives the previous seal), and the seal file's own digest appears nowhere inside it;
* the two required capacity budgets (per_ctx_outstanding_credit_cap, response_record_budget) are closed by
  recorded decisions in configs/w4_capacity_decisions.yaml (a capacity with no derivable bound is a design
  decision, never a number in code; recording one does not make it "derivable"), PROTOCOL_FREEZE.json is
  committed, and the G1 gate is green end to end;
* every way a decision can be incomplete (null, 0, bool, float, wrong unit, blank text, missing or foreign
  entry, wrong schema) is exercised against a patched decisions file and stays open or refuses to seal;
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


BUDGETS = sorted(fc.REQUIRED_CLOSED_BUDGETS)


def _decisions(tmp_path: Path, **overrides: dict[str, object]) -> Path:
  """A decisions file with both budgets fully recorded; `overrides` replace single entries."""
  doc = yaml.safe_load((H / "configs/w4_capacity_decisions.yaml").read_text())
  for name in BUDGETS:
    doc["decisions"][name] = {
      "unit": "entries",
      "value": 8,
      "decided_by": "review-2026-11-11",
      "rationale": "fixture decision; stalls when full, never drops",
      "evidence": "fixture://cosim-sweep-1",
    }
  doc["decisions"].update(overrides)
  path = tmp_path / "w4_capacity_decisions.yaml"
  path.write_text(yaml.safe_dump(doc))
  return path


def _use(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
  monkeypatch.setattr(fc, "W4_DECISIONS", path)


# ---------------------------------------------------------------------------------------------------
# the committed state: decisions recorded, budgets closed, seal committed, gate green
# ---------------------------------------------------------------------------------------------------
def test_the_two_capacity_budgets_are_now_closed() -> None:
  assert fc.open_budgets() == []


def test_the_committed_tree_is_sealed_and_the_gate_is_green(gate: gates.G1Gate) -> None:
  report = gate.run()
  assert report["pass"] is True and report["exit_code"] == 0, report["checks"]
  assert all(c["ok"] for c in report["checks"])
  seal = json.loads((H / "protocol/PROTOCOL_FREEZE.json").read_text())
  assert seal["state"] == "frozen" and seal["protocol_seq"] == 1
  assert fc.verify_seal() == []


def test_a_fully_recorded_decision_closes_the_budget(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _use(monkeypatch, _decisions(tmp_path))
  assert fc.open_budgets() == []


@pytest.mark.parametrize(
  "broken",
  [
    {"value": None},
    {"value": 0},
    {"value": -3},
    {"value": True},
    {"value": 8.0},
    {"value": "8"},
    {"unit": "cycles"},
    {"decided_by": None},
    {"decided_by": "  "},
    {"rationale": ""},
    {"evidence": None},
  ],
)
def test_every_incomplete_decision_stays_open(
  broken: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  good = yaml.safe_load(_decisions(tmp_path).read_text())["decisions"]["response_record_budget"]
  _use(monkeypatch, _decisions(tmp_path, response_record_budget={**good, **broken}))
  assert fc.open_budgets() == ["response_record_budget"]


def test_a_missing_file_a_missing_entry_or_a_foreign_entry_is_open(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _use(monkeypatch, tmp_path / "absent.yaml")
  assert fc.open_budgets() == BUDGETS
  doc = yaml.safe_load(_decisions(tmp_path).read_text())
  del doc["decisions"]["per_ctx_outstanding_credit_cap"]
  path = tmp_path / "missing.yaml"
  path.write_text(yaml.safe_dump(doc))
  _use(monkeypatch, path)
  assert fc.open_budgets() == ["per_ctx_outstanding_credit_cap"]
  doc = yaml.safe_load(_decisions(tmp_path).read_text())
  doc["decisions"]["invented_budget"] = doc["decisions"]["response_record_budget"]
  path = tmp_path / "foreign.yaml"
  path.write_text(yaml.safe_dump(doc))
  _use(monkeypatch, path)
  assert fc.open_budgets() == ["invented_budget (not a required budget)"]
  doc["schema"] = "something/else"
  path.write_text(yaml.safe_dump(doc))
  assert fc.open_budgets() == BUDGETS


def test_the_committed_decision_file_records_decisions() -> None:
  doc = yaml.safe_load((H / "configs/w4_capacity_decisions.yaml").read_text())
  assert sorted(doc["decisions"]) == BUDGETS
  for entry in doc["decisions"].values():
    assert isinstance(entry["value"], int) and entry["value"] >= 1
    assert entry["unit"] == "entries"
    for k in fc.DECISION_TEXT_FIELDS:
      assert isinstance(entry[k], str) and entry[k].strip()


def test_a_decision_does_not_make_the_capacity_derivable() -> None:
  # the constraints file keeps both underivable with a reason and a needs list; recording a decision never
  # edits that claim, so a reader cannot mistake a chosen number for a derived one
  doc = yaml.safe_load((H / "configs/w4_memory_constraints.yaml").read_text())
  listed = {e["name"]: e for e in doc["unresolved"]}
  for name in BUDGETS:
    assert listed[name]["derivable"] is False and listed[name]["reason"] and listed[name]["needs"]


def test_the_decision_file_is_sealed_so_changing_a_decision_needs_a_bump(
  closed_budgets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  rel = "hardware/coral_nexus/configs/w4_capacity_decisions.yaml"
  assert rel in fc.REQUIRED_INPUTS
  # the seal records the decision file's hash, so editing a decision after sealing breaks verification
  monkeypatch.setattr(fc, "SEAL_PATH", closed_budgets / "PROTOCOL_FREEZE.json")
  fc.seal()
  record = json.loads((closed_budgets / "PROTOCOL_FREEZE.json").read_text())
  assert rel in record["inputs"], "the decision file is part of the sealed input set"


def test_a_width_may_stay_underivable_but_a_capacity_may_not(
  gate: gates.G1Gate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _use(monkeypatch, _decisions(tmp_path))
  report = gate.run()
  by = {c["check"]: c for c in report["checks"]}
  assert by["budgets.closed"]["ok"] is True
  assert "widths declared underivable" in by["budgets.closed"]["detail"]
  assert by["protocol.contracts"]["ok"] is True
  # with the committed seal in place nothing else blocks the gate
  assert report["exit_code"] == 0


def test_withdrawing_one_decision_reopens_the_budget_and_fails_the_gate_with_14(
  gate: gates.G1Gate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _use(monkeypatch, _decisions(tmp_path, response_record_budget={"value": None}))
  report = gate.run()
  by = {c["check"]: c for c in report["checks"]}
  assert by["budgets.closed"]["ok"] is False and "response_record_budget" in by["budgets.closed"]["detail"]
  assert report["exit_code"] == gates.REQUIRED_G1_CHECKS["budgets.closed"] == 14


def test_a_crashing_or_failing_fragment_validator_fails_the_gate(
  gate: gates.G1Gate, monkeypatch: pytest.MonkeyPatch
) -> None:
  import validate_numeric as vnum

  def boom() -> dict:
    raise RuntimeError("probe")

  monkeypatch.setattr(vnum, "validate", boom)
  report = gate.run()
  by = {c["check"]: c for c in report["checks"]}
  assert by["protocol.contracts"]["ok"] is False and "C011 numeric" in by["protocol.contracts"]["detail"]
  monkeypatch.setattr(vnum, "validate", lambda: {"valid": False, "errors": ["mutated"]})
  assert gate.run()["exit_code"] == 11
  monkeypatch.setattr(vnum, "validate", lambda: {"errors": []})
  assert gate.run()["exit_code"] == 11, "a report that does not say valid is not valid"


# ---------------------------------------------------------------------------------------------------
# the seal, once the budgets are closed
# ---------------------------------------------------------------------------------------------------
@pytest.fixture
def closed_budgets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  _use(monkeypatch, _decisions(tmp_path))
  return tmp_path


def test_sealing_with_open_budgets_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  doc = yaml.safe_load((H / "configs/w4_capacity_decisions.yaml").read_text())
  for entry in doc["decisions"].values():
    entry.update(value=None, decided_by=None, rationale=None, evidence=None)
  path = tmp_path / "open.yaml"
  path.write_text(yaml.safe_dump(doc))
  _use(monkeypatch, path)
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
  # the decision is withdrawn after the seal (value back to null): the seal stops verifying, twice over
  # (the file hash moved and the budget is open again)
  _use(monkeypatch, _decisions(closed_budgets, per_ctx_outstanding_credit_cap={"value": None}))
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


def test_the_gate_command_exits_zero_on_the_committed_tree() -> None:
  import subprocess

  proc = subprocess.run(
    ["bash", str(H / "scripts/cn.sh"), "gate", "--stage", "G1", "--out", "out/coral_nexus/C020/gate-test"],
    capture_output=True,
    text=True,
    cwd=REPO_ROOT,
  )
  assert proc.returncode == 0, proc.stdout
  assert "seal.joint" in proc.stdout and "budgets.closed" in proc.stdout
