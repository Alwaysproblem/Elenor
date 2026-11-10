"""C011 numeric-contract file checks (contracts suite).

The profile's citations, derived parameters, rule/vector coverage and cvfpu encodings are checked
against the pinned tree; each check is shown to fire by mutating its input. Numeric behaviour lives
in verification/numerics/test_numeric_bootstrap.py (numeric-bootstrap suite).
"""

from __future__ import annotations

import copy
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
