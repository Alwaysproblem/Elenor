"""C011 numeric-contract file checks (contracts suite).

The profile's citations, derived parameters, rule/vector coverage and cvfpu encodings are checked
against the pinned tree; each check is shown to fire by mutating its input. Numeric behaviour lives
in verification/numerics/test_numeric_bootstrap.py (numeric-bootstrap suite).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

H = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(H / "scripts/lib"))

import numeric_ref as nr  # noqa: E402
import validate_numeric as vn  # noqa: E402


@pytest.fixture(scope="module")
def profile() -> dict[str, Any]:
  return vn.load_profile()


@pytest.fixture(scope="module")
def vectors() -> dict[str, Any]:
  return vn.load_vectors()


def _fresh_vectors() -> dict[str, Any]:
  return copy.deepcopy(vn.load_vectors())


def _fresh_profile() -> dict[str, Any]:
  return copy.deepcopy(vn.load_profile())


def test_committed_numeric_contract_validates() -> None:
  report = vn.validate()
  assert report["valid"], report["errors"]


def test_profile_must_stay_experimental() -> None:
  profile = _fresh_profile()
  profile["support_status"] = "supported"
  tmp = Path(__file__).parent / "_scratch_profile.yaml"
  import yaml

  tmp.write_text(yaml.safe_dump(profile))
  try:
    report = vn.validate(profile_path=tmp)
  finally:
    tmp.unlink()
  assert any("experimental" in e for e in report["errors"])


def test_every_rule_has_at_least_one_directed_vector(
  profile: dict[str, Any], vectors: dict[str, Any]
) -> None:
  ids = {v["id"] for v in vectors["vectors"]}
  for rule in profile["rules"]:
    assert rule["vectors"], rule["id"]
    assert set(rule["vectors"]) <= ids, rule["id"]


def test_rule_without_a_vector_is_rejected() -> None:
  profile, vectors = _fresh_profile(), _fresh_vectors()
  profile["rules"][0]["vectors"] = []
  errors: list[str] = []
  vn.check_rule_vector_coverage(profile, vectors, errors)
  assert any("no directed vector" in e for e in errors)


def test_vector_claimed_by_no_rule_is_rejected() -> None:
  profile, vectors = _fresh_profile(), _fresh_vectors()
  for rule in profile["rules"]:
    rule["vectors"] = [v for v in rule["vectors"] if v != "spc_single_inf"]
  errors: list[str] = []
  vn.check_rule_vector_coverage(profile, vectors, errors)
  assert any("claimed by no rule" in e for e in errors)


# --- parameters re-derived, anchors re-read ---------------------------------------------------


def test_generator_parameters_are_rederived_not_trusted() -> None:
  derived = nr.derive_parameters(("FP32", "FP16ALT"))
  assert (derived.vec_len, derived.guard_bits, derived.prod_exp_bits, derived.prod_sig_bits) == (
    4,
    2,
    10,
    26,
  )
  assert derived.lanes == {"FP32": 1, "FP16ALT": 2}
  # enabling a format that does not exist, or none, is an error
  with pytest.raises(nr.NumericError):
    nr.derive_parameters(("FP128",))
  with pytest.raises(nr.NumericError):
    nr.derive_parameters(())


def test_profile_parameter_drift_is_caught() -> None:
  profile = _fresh_profile()
  profile["generator"]["derived"]["prod_sig_bits"] = 27
  errors: list[str] = []
  vn.check_generator(profile, errors)
  assert any("prod_sig_bits" in e for e in errors)


def test_fp_fmt_config_literal_must_match_enabled_formats() -> None:
  profile = _fresh_profile()
  profile["generator"]["fp_fmt_config"]["literal"] = "5'b11000"
  errors: list[str] = []
  vn.check_generator(profile, errors)
  assert any("literal" in e for e in errors)


def test_every_cited_line_is_read_back() -> None:
  errors: list[str] = []
  assert vn.check_anchors(vn.load_profile(), errors) >= 30
  assert errors == []


@pytest.mark.parametrize("mutation", ["line", "anchor", "file"])
def test_drifted_source_citation_is_caught(mutation: str) -> None:
  profile = _fresh_profile()
  src = profile["rules"][0]["sources"][0]
  if mutation == "line":
    src["line"] += 7
  elif mutation == "anchor":
    src["anchor"] = "something the file does not say"
  else:
    src["file"] = "verilog/rvv/design/Zvt/does_not_exist.sv"
  errors: list[str] = []
  vn.check_anchors(profile, errors)
  assert errors


def test_cvfpu_encoding_drift_is_caught() -> None:
  profile = _fresh_profile()
  profile["cvfpu_pin"]["encodings"]["ROD"]["value"] = "3'b011"
  errors: list[str] = []
  vn.check_cvfpu(profile, errors, [])
  if vn.CVFPU_VENDOR.joinpath("src", "fpnew_pkg.sv").is_file():
    assert errors
  profile = _fresh_profile()
  profile["cvfpu_pin"]["read_sha256"] = "0" * 64
  errors = []
  vn.check_cvfpu(profile, errors, [])
  if vn.CVFPU_VENDOR.joinpath("src", "fpnew_pkg.sv").is_file():
    assert any("sha256" in e for e in errors)


def test_missing_cvfpu_copy_is_reported_not_passed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  monkeypatch.setattr(vn, "CVFPU_VENDOR", tmp_path)
  notes: list[str] = []
  assert vn.check_cvfpu(vn.load_profile(), [], notes) is False
  assert any("NOT re-read" in n for n in notes)


# --- the required contract is held by the validator, not by the file it constrains ---------------


def _validate_mutated(
  tmp_path: Path, mutate_profile: Any = None, mutate_vectors: Any = None
) -> dict[str, Any]:
  """Run the WHOLE validator on mutated copies, the path a synchronized edit would take."""
  import yaml

  profile, vectors = _fresh_profile(), _fresh_vectors()
  if mutate_profile:
    mutate_profile(profile)
  if mutate_vectors:
    mutate_vectors(vectors)
  (tmp_path / "profile.yaml").write_text(yaml.safe_dump(profile))
  (tmp_path / "vectors.json").write_text(json.dumps(vectors))
  return vn.validate(tmp_path / "profile.yaml", tmp_path / "vectors.json")


def _drop_rule_and_vectors(rule_id: str, kinds: set[str]) -> tuple[Any, Any]:
  def mutate_profile(profile: dict[str, Any]) -> None:
    profile["rules"] = [r for r in profile["rules"] if r["id"] != rule_id]

  def mutate_vectors(vectors: dict[str, Any]) -> None:
    vectors["vectors"] = [
      v for v in vectors["vectors"] if v["kind"] not in kinds and rule_id not in v.get("rules", [])
    ]

  return mutate_profile, mutate_vectors


@pytest.mark.parametrize(
  ("rule_id", "kinds"),
  [
    ("NAR-01", {"narrow"}),
    ("FRM-01", {"frm"}),
    ("TIL-01", {"record_case"}),
    ("JAM-01", {"ablation"}),
    ("TKG-01", {"tk_sequence"}),
    ("OVF-ACC-01", set()),
    ("OVF-MUL-01", set()),
    ("MSK-01", set()),
    ("SPC-01", set()),
    ("CAN-01", set()),
    ("SUB-01", set()),
    ("FLG-01", set()),
    ("MUL-01", set()),
    ("ROD-01", set()),
  ],
)
def test_deleting_a_rule_together_with_its_vectors_invalidates_the_contract(
  tmp_path: Path, rule_id: str, kinds: set[str]
) -> None:
  """The advisory case: rule and vectors removed in one edit leave only self-consistent leftovers."""
  mutate_profile, mutate_vectors = _drop_rule_and_vectors(rule_id, kinds)
  report = _validate_mutated(tmp_path, mutate_profile, mutate_vectors)
  assert report["valid"] is False
  assert any(rule_id in e and "required rule" in e and "missing" in e for e in report["errors"]), report[
    "errors"
  ][:4]


def test_the_narrowing_boundary_cannot_vanish_even_if_every_leftover_is_consistent(tmp_path: Path) -> None:
  mutate_profile, mutate_vectors = _drop_rule_and_vectors("NAR-01", {"narrow"})
  report = _validate_mutated(tmp_path, mutate_profile, mutate_vectors)
  # the leftovers really are self-consistent: the rule/vector cross-check alone finds nothing wrong
  profile, vectors = _fresh_profile(), _fresh_vectors()
  mutate_profile(profile)
  mutate_vectors(vectors)
  assert _coverage_errors(profile, vectors) == []
  assert report["valid"] is False
  assert any("'narrow'" in e and "absent" in e for e in report["errors"])


def _coverage_errors(profile: dict[str, Any], vectors: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  vn.check_rule_vector_coverage(profile, vectors, errors)
  return errors


@pytest.mark.parametrize("kind", sorted(vn.REQUIRED_VECTOR_KINDS))
def test_removing_every_vector_of_a_kind_invalidates_the_contract(tmp_path: Path, kind: str) -> None:
  def mutate_profile(profile: dict[str, Any]) -> None:
    gone = {v["id"] for v in _fresh_vectors()["vectors"] if v["kind"] == kind}
    for rule in profile["rules"]:
      rule["vectors"] = [v for v in rule["vectors"] if v not in gone]
    # keep every rule non-empty so only the kind check can object
    for rule in profile["rules"]:
      if not rule["vectors"]:
        rule["vectors"] = ["mul_basic_exact"]

  def mutate_vectors(vectors: dict[str, Any]) -> None:
    vectors["vectors"] = [v for v in vectors["vectors"] if v["kind"] != kind]

  report = _validate_mutated(tmp_path, mutate_profile, mutate_vectors)
  assert report["valid"] is False
  assert any(kind in e for e in report["errors"])


@pytest.mark.parametrize(
  ("rule_id", "weaker"),
  [("NAR-01", "upstream_source"), ("TIL-01", "upstream_source"), ("OVF-ACC-01", "decided_by_card")],
)
def test_weakening_or_inflating_an_evidence_class_is_rejected(
  tmp_path: Path, rule_id: str, weaker: str
) -> None:
  def mutate_profile(profile: dict[str, Any]) -> None:
    next(r for r in profile["rules"] if r["id"] == rule_id)["evidence"] = weaker

  report = _validate_mutated(tmp_path, mutate_profile)
  assert report["valid"] is False
  assert any(rule_id in e and "evidence class" in e for e in report["errors"])


def test_a_rule_that_loses_the_vector_kind_it_needs_is_rejected(tmp_path: Path) -> None:
  """NAR-01 keeps a vector but of the wrong kind: its narrowing evidence is gone."""

  def mutate_vectors(vectors: dict[str, Any]) -> None:
    for v in vectors["vectors"]:
      if v["kind"] == "narrow":
        v["kind"] = "cell"

  report = _validate_mutated(tmp_path, None, mutate_vectors)
  assert report["valid"] is False


def test_an_unlisted_extra_rule_must_be_added_to_the_validator_deliberately(tmp_path: Path) -> None:
  def mutate_profile(profile: dict[str, Any]) -> None:
    profile["rules"].append({**profile["rules"][0], "id": "NEW-01", "vectors": ["mul_basic_exact"]})

  report = _validate_mutated(tmp_path, mutate_profile)
  assert report["valid"] is False
  assert any("NEW-01" in e and "required set" in e for e in report["errors"])


@pytest.mark.parametrize("item", sorted(vn.REQUIRED_OPEN_ITEMS))
def test_an_open_item_cannot_be_dropped_or_quietly_closed(tmp_path: Path, item: str) -> None:
  def drop(profile: dict[str, Any]) -> None:
    profile["open_items"] = [i for i in profile["open_items"] if i["id"] != item]

  def close(profile: dict[str, Any]) -> None:
    next(i for i in profile["open_items"] if i["id"] == item)["status"] = "closed"

  for mutate in (drop, close):
    report = _validate_mutated(tmp_path, mutate)
    assert report["valid"] is False
    assert any(item in e for e in report["errors"])


def test_kd01_and_the_closed_probe_item_cannot_be_removed(tmp_path: Path) -> None:
  def drop_kd(profile: dict[str, Any]) -> None:
    profile["known_differences"] = []

  def reopen(profile: dict[str, Any]) -> None:
    next(i for i in profile["open_items"] if i["id"] == "OPEN-01")["status"] = "open"

  assert any("KD-01" in e for e in _validate_mutated(tmp_path, drop_kd)["errors"])
  assert any("OPEN-01" in e for e in _validate_mutated(tmp_path, reopen)["errors"])


def test_required_set_is_not_read_from_the_profile() -> None:
  """The set lives in the validator module: mutating the profile cannot change it."""
  profile = _fresh_profile()
  before = set(vn.REQUIRED_RULES)
  profile["rules"] = []
  assert set(vn.REQUIRED_RULES) == before and len(before) == 14


def test_every_required_rule_is_in_the_committed_profile_with_a_matching_class(
  profile: dict[str, Any],
) -> None:
  rules = {r["id"]: r for r in profile["rules"]}
  assert set(rules) == set(vn.REQUIRED_RULES)
  for rid, need in vn.REQUIRED_RULES.items():
    assert rules[rid]["evidence"] == need["evidence"], rid


# --- a record_case vector must point at a real tail case ------------------------------------------


def _tail_vector(**override: Any) -> dict[str, Any]:
  vec = next(v for v in _fresh_vectors()["vectors"] if v["kind"] == "record_case")
  return {**vec, **override}


def test_the_committed_tail_vector_points_at_a_real_tail_case() -> None:
  assert vn.check_record_case(_tail_vector()) == []


@pytest.mark.parametrize(
  ("override", "needle"),
  [
    ({"case": 0}, "full tile"),  # full 16x16 tile: no tail evidence
    ({"case": 1}, "full tile"),
    ({"case": 99}, "no case 99"),
    ({"record": "other.json"}, "unknown record"),
  ],
)
def test_a_tail_vector_that_names_the_wrong_case_is_rejected(override: dict[str, Any], needle: str) -> None:
  assert any(needle in e for e in vn.check_record_case(_tail_vector(**override)))


def test_a_tail_case_that_disturbed_cells_outside_the_window_is_rejected(tmp_path: Path) -> None:
  record = json.loads(vn.RECORD.read_text())
  case = record["cases"][3]
  outside = next(i for i in range(256) if i // 16 >= case["tm"] or i % 16 >= case["tn"])
  case["actual_u32"][outside] ^= 1
  path = tmp_path / "record.json"
  path.write_text(json.dumps(record))
  assert any("outside 9x11 changed" in e for e in vn.check_record_case(_tail_vector(), path))


def test_a_tail_case_without_preload_is_rejected(tmp_path: Path) -> None:
  record = json.loads(vn.RECORD.read_text())
  record["cases"][3]["init_mode"] = 0
  path = tmp_path / "record.json"
  path.write_text(json.dumps(record))
  assert any("did not preload" in e for e in vn.check_record_case(_tail_vector(), path))


def test_a_tail_case_that_updated_nothing_inside_is_rejected(tmp_path: Path) -> None:
  record = json.loads(vn.RECORD.read_text())
  case = record["cases"][3]
  case["actual_u32"] = list(case["c_init_u32"])
  path = tmp_path / "record.json"
  path.write_text(json.dumps(record))
  assert any("no cell inside" in e for e in vn.check_record_case(_tail_vector(), path))


def test_a_malformed_vector_is_reported_not_raised(tmp_path: Path) -> None:
  def mutate_vectors(vectors: dict[str, Any]) -> None:
    for v in vectors["vectors"]:
      if v["kind"] == "narrow":
        v["kind"] = "cell"  # fields no longer fit the kind

  report = _validate_mutated(tmp_path, None, mutate_vectors)  # must not raise
  assert report["valid"] is False
  assert any("needs fields" in e for e in report["errors"])
  assert vn.run_vector({"kind": "mystery"}) == ["unknown vector kind 'mystery'"]
  bad = {
    "kind": "cell",
    "old_fp32_bits": 0,
    "pairs_bf16_bits": 5,
    "tk": 1,
    "expected_fp32_bits": 0,
    "expected_flags": {"nv": 0, "of": 0},
  }
  assert any("malformed" in e for e in vn.run_vector(bad))
  assert vn.run_vector({"kind": "frm", "frm": "x", "expected_error": "E"}) == [
    "error FRM_RANGE != expected E"
  ]


def test_a_rule_repointed_at_the_wrong_kind_of_vector_is_rejected_by_the_whole_contract(
  tmp_path: Path,
) -> None:
  """NAR-01 keeps a real vector, but a cell vector: the global kinds are all still present."""

  def mutate_profile(profile: dict[str, Any]) -> None:
    next(r for r in profile["rules"] if r["id"] == "NAR-01")["vectors"] = ["mul_basic_exact"]

  report = _validate_mutated(tmp_path, mutate_profile)
  assert report["valid"] is False
  assert any("NAR-01" in e and "'narrow'" in e and "no" in e for e in report["errors"])


def test_a_tail_vector_pointing_at_a_full_tile_invalidates_the_whole_contract(tmp_path: Path) -> None:
  def mutate_vectors(vectors: dict[str, Any]) -> None:
    next(v for v in vectors["vectors"] if v["kind"] == "record_case")["case"] = 0

  report = _validate_mutated(tmp_path, None, mutate_vectors)
  assert report["valid"] is False
  assert any("til_tail_9x11" in e and "full tile" in e for e in report["errors"])
