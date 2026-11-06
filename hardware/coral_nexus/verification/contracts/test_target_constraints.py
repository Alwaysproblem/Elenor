"""C008 target-constraint validation tests.

Every C008 rejection class gets a test that the validator actually rejects
it (and a positive case that legal constraints parse). The suites run
without --config (pure data checks over fixtures), per the card.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
sys.path.insert(0, str(H / "scripts/lib"))

import validate_constraints as vc  # noqa: E402

PARAMS_DIR = H / "configs/params.d"


def _config(**constraints: object) -> dict:
  return {
    "schema": "cn_target/v1",
    "kind": "p0_upstream",
    "name": "fixture",
    "constraints": constraints,
  }


def _validate(config: dict, params_dir: Path | None = None,
              extra: dict | None = None) -> dict:
  path = Path("/tmp") / "cn_c008_fixture.json"
  path.write_text(json.dumps(config))
  return vc.validate(path, params_dir or PARAMS_DIR, extra_targets=extra)


def _params_dir(tmp_path: Path, docs: list[dict]) -> Path:
  d = tmp_path / "params.d"
  d.mkdir(exist_ok=True)
  for i, doc in enumerate(docs):
    (d / f"src{i}.json").write_text(json.dumps(doc))
  return d


def test_valid_constraints_parse() -> None:
  report = _validate(_config(
    context_count=1,
    user_spm_per_bank=63488,
    system_reserved_spm_per_bank=2048,
    retirement_buffer_mode="full",
  ))
  assert report["valid"], report["errors"]
  assert report["params_merged"] >= 6  # three sources merged


def test_missing_kind_is_rejected() -> None:
  cfg = _config(context_count=1)
  del cfg["kind"]
  assert not _validate(cfg)["valid"]


def test_negative_constraint_is_rejected() -> None:
  assert not _validate(_config(max_in_flight_per_context=-1))["valid"]


def test_zero_context_count_is_rejected() -> None:
  report = _validate(_config(context_count=0))
  assert not report["valid"]
  assert any("minimum" in e for e in report["errors"])


def test_context_count_above_sourced_max_is_rejected() -> None:
  report = _validate(_config(context_count=9))
  assert not report["valid"]
  assert any("maximum" in e or "sourced maximum" in e for e in report["errors"])


def test_unit_conflict_is_rejected_after_normalization(tmp_path: Path) -> None:
  """2048 B vs 2048 KiB is a unit conflict; 2048 B vs 2 KiB agrees."""
  conflicting = _params_dir(tmp_path, [
    {"schema": "cn_params/v1", "source": {"kind": "a"},
     "params": [{"name": "cap", "value": 2048, "unit": "B"}]},
    {"schema": "cn_params/v1", "source": {"kind": "b"},
     "params": [{"name": "cap", "value": 2048, "unit": "KiB"}]},
  ])
  report = _validate(_config(), conflicting)
  assert not report["valid"]
  assert any("multi-source conflict" in e for e in report["errors"])

  agreeing = _params_dir(tmp_path, [
    {"schema": "cn_params/v1", "source": {"kind": "a"},
     "params": [{"name": "cap", "value": 2048, "unit": "B"}]},
    {"schema": "cn_params/v1", "source": {"kind": "b"},
     "params": [{"name": "cap", "value": 2, "unit": "KiB"}]},
  ])
  assert _validate(_config(), agreeing)["valid"]


def test_reserved_plus_user_overflows_sourced_bank() -> None:
  report = _validate(_config(
    user_spm_per_bank=70000, system_reserved_spm_per_bank=2048
  ))
  assert not report["valid"]
  assert any("L1 SPM overflow" in e for e in report["errors"])


def test_config_reserved_conflicts_with_sourced_reserved() -> None:
  report = _validate(_config(system_reserved_spm_per_bank=4096))
  assert not report["valid"]
  assert any("conflicts with sourced" in e for e in report["errors"])


def test_unencodable_width_is_rejected(tmp_path: Path) -> None:
  d = _params_dir(tmp_path, [
    {"schema": "cn_params/v1", "source": {"kind": "fixture"},
     "params": [
       {"name": "queue_depth", "value": 16, "unit": "count", "width_bits": 4}
     ]},
  ])
  report = _validate(_config(), d)
  assert not report["valid"]
  assert any("not encodable" in e for e in report["errors"])


def test_c1_c2_degradation_direction_enforced() -> None:
  base = {"schema": "cn_target/v1", "kind": "cs1_target", "name": "t"}
  c1 = dict(base, constraints={"context_count": 1, "tile_count": 4})
  c2 = dict(base, constraints={"context_count": 2, "tile_count": 4})
  report = _validate(c1, extra={"c1": c1, "c2": c2})
  assert report["valid"], report["errors"]

  bad_c1 = dict(base, constraints={"context_count": 2, "tile_count": 4})
  bad_c2 = dict(base, constraints={"context_count": 1, "tile_count": 4})
  report = _validate(bad_c1, extra={"c1": bad_c1, "c2": bad_c2})
  assert not report["valid"]
  assert any("degradation" in e for e in report["errors"])

  drift_c1 = dict(base, constraints={"context_count": 1, "tile_count": 8})
  drift_c2 = dict(base, constraints={"context_count": 2, "tile_count": 4})
  report = _validate(drift_c1, extra={"c1": drift_c1, "c2": drift_c2})
  assert not report["valid"]
  assert any("share every constraint" in e for e in report["errors"])


def test_committed_params_d_have_no_internal_conflict() -> None:
  """The real params.d must merge cleanly (multi-source agreement)."""
  merged, errors = vc.load_params(PARAMS_DIR)
  assert errors == [], errors
  assert {"context_count_min", "context_count_max",
          "l1_spm_bytes_per_bank_mode0"} <= set(merged)
