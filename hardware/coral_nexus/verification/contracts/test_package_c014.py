"""C014 cn_package/v1 contract tests.

The consumer (`package_contract`) is built FROM protocol/w5_package.contract.yaml, and the contract's
required set is held in `validate_package_contract`, so these tests use real structured packages: a
complete binding is accepted, every kind of damage is rejected with a code that names the field, and a
synchronized edit that weakens the YAML together with its guard is rejected.

Card pass-condition clauses -> tests:
  complete binding accepted ........................... test_a_complete_binding_*
  missing actual view ................................. test_missing_actual_view_*
  illegal alias ....................................... test_*alias*
  undeclared resource ................................. test_undeclared_helper / test_scratch_*
  unknown field ....................................... test_unknown_field_*
  wrong version domain ................................ test_*version*
  image with no controlled source ..................... test_image_with_no_controlled_source
  a consumer can generate a real call and check effect  test_helpers_resolve_*
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

H = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(H / "scripts/lib"))

import package_contract as pc  # noqa: E402
import validate_package_contract as vp  # noqa: E402


@pytest.fixture
def contract() -> dict[str, Any]:
  return copy.deepcopy(pc.load_contract())


@pytest.fixture
def request_doc() -> dict[str, Any]:
  return copy.deepcopy(yaml.safe_load(vp.REQUEST.read_text()))


@pytest.fixture
def good() -> dict[str, Any]:
  return copy.deepcopy(json.loads(vp.EXAMPLE.read_text()))


def _kernel(package: dict[str, Any]) -> dict[str, Any]:
  return package["binding_manifest"]["kernels"][0]  # type: ignore[no-any-return]


def _resealed(package: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
  """Recompute the hash after an edit, so only the SEMANTIC checks can reject it."""
  body = {k: v for k, v in package.items() if k != "package_hash"}
  return {**body, "package_hash": pc.package_hash(body, contract)}


def _code(package: dict[str, Any], contract: dict[str, Any]) -> tuple[str, str]:
  with pytest.raises(pc.PackageError) as exc:
    pc.validate_package(package, contract)
  return exc.value.code, exc.value.field


def _report(contract: Any, request_doc: Any) -> dict[str, Any]:
  return vp.validate_docs(contract, request_doc)


# --- the committed contract and example -------------------------------------------------------


def test_committed_contract_validates(contract: dict, request_doc: dict) -> None:
  report = _report(contract, request_doc)
  assert report["valid"], report["errors"]
  assert report["anchors_read_back"] >= 9


def test_cli_exits_zero_on_the_committed_files(tmp_path: Path) -> None:
  run = subprocess.run(
    [sys.executable, str(H / "scripts/lib/validate_package_contract.py"), "--out", str(tmp_path)],
    capture_output=True,
    text=True,
  )
  assert run.returncode == 0
  assert json.loads((tmp_path / "package_contract_report.json").read_text())["valid"]


def test_a_complete_binding_is_accepted_sealed_and_reloaded(good: dict, contract: dict) -> None:
  pc.validate_package(good, contract)
  sealed = pc.seal({k: v for k, v in good.items() if k != "package_hash"}, contract)
  assert sealed["package_hash"] == good["package_hash"]
  assert pc.load(json.dumps(sealed), contract) == sealed


def test_the_committed_example_is_a_real_resealable_package(good: dict, contract: dict) -> None:
  assert good["package_hash"] != "0" * 64 and len(good["package_hash"]) == 64
  assert _resealed(good, contract)["package_hash"] == good["package_hash"]


def test_round_trip_is_field_for_field_equal(good: dict, contract: dict) -> None:
  assert json.loads(json.dumps(pc.load(json.dumps(good), contract))) == good


# --- missing / empty / unknown fields ---------------------------------------------------------


@pytest.mark.parametrize(
  "field", ["ins", "outs", "events", "initial", "scratch", "clobber", "region", "helpers", "mode_table"]
)
def test_a_missing_kernel_field_is_rejected_and_named(good: dict, contract: dict, field: str) -> None:
  _kernel(good).pop(field)
  code, where = _code(_resealed(good, contract), contract)
  assert code == "MISSING_FIELD" and where.endswith(f".{field}")


@pytest.mark.parametrize("field", pc.load_contract()["package"]["required_fields"])
def test_a_missing_outer_field_is_rejected_and_named(good: dict, contract: dict, field: str) -> None:
  good.pop(field)
  code, where = _code(good, contract)
  assert code == "MISSING_FIELD" and where == f"$.{field}"


@pytest.mark.parametrize("side", ["ins", "outs"])
def test_explicit_ins_and_outs_must_not_be_empty(good: dict, contract: dict, side: str) -> None:
  _kernel(good)[side] = []
  assert _code(_resealed(good, contract), contract)[0] == "MISSING_VIEWS"


def test_missing_actual_view_is_rejected_not_inferred(good: dict, contract: dict) -> None:
  _kernel(good)["ins"][0].pop("actual_view")
  assert _code(_resealed(good, contract), contract) == (
    "MISSING_FIELD",
    "$.binding_manifest.kernels[0].ins[0].actual_view",
  )
  _kernel(good)["ins"][0]["actual_view"] = ""
  assert _code(_resealed(good, contract), contract)[0] == "MISSING_ACTUAL_VIEW"


@pytest.mark.parametrize(
  ("where", "inject"),
  [
    ("top", lambda p: p.__setitem__("extra", 1)),
    ("kernel", lambda p: _kernel(p).__setitem__("hint", 1)),
    ("view", lambda p: _kernel(p)["ins"][0].__setitem__("stride", 1)),
    ("image", lambda p: p["image"].__setitem__("compressed", False)),
    ("link_layout", lambda p: p["image"]["link_layout"].__setitem__("align", 16)),
    ("provenance", lambda p: p["image"]["provenance"].__setitem__("note", "x")),
    ("events", lambda p: _kernel(p)["events"].__setitem__("extra", "e")),
    ("resource_manifest", lambda p: p["resource_manifest"].__setitem__("heap_bytes", 1)),
    ("target_fingerprint", lambda p: p["target_fingerprint"].__setitem__("mhz", 1000)),
    ("patch_mask", lambda p: p["patch_mask"][0].__setitem__("why", "x")),
  ],
)
def test_unknown_field_is_rejected_at_every_level_and_named(
  good: dict, contract: dict, where: str, inject: Any
) -> None:
  inject(good)
  code, field = _code(_resealed(good, contract), contract)
  assert code == "UNKNOWN_FIELD", (where, code, field)


# --- version domains --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["cn_package/v2", "cn_package/v0", 3, 1, None, "", True])
def test_wrong_package_version_is_rejected(good: dict, contract: dict, value: Any) -> None:
  good["package_schema"] = value
  assert _code(good, contract) == ("WRONG_VERSION", "$.package_schema")


@pytest.mark.parametrize("value", [4, 2, 0, True, "3", 3.0])
def test_wrong_control_plane_schema_is_rejected(good: dict, contract: dict, value: Any) -> None:
  good["control_plane"]["schema_version"] = value
  assert _code(good, contract)[0] == "WRONG_VERSION"


def test_wrong_control_plane_abi_is_rejected(good: dict, contract: dict) -> None:
  good["control_plane"]["compiler_abi"] = "v2"
  assert _code(good, contract) == ("WRONG_VERSION", "$.control_plane.compiler_abi")


def test_the_package_version_domain_never_shares_a_number_with_another(contract: dict) -> None:
  dom = contract["version_domains"]
  assert dom["package"]["id"] == "cn_package/v1"
  assert dom["control_plane"]["schema_version"] == 3 and dom["hardware_yaml"]["schema_version"] == 2
  assert (
    dom["legacy_package"]["package_abi_version"] == 1 and dom["legacy_package"]["status"] == "not_inherited"
  )


def test_changing_the_package_domain_leaves_the_other_domains_alone(contract: dict) -> None:
  contract["version_domains"]["package"]["id"] = "cn_package/v2"
  assert contract["version_domains"]["control_plane"]["schema_version"] == 3


# --- views: dtype, shape, layout, permission, alias -------------------------------------------


@pytest.mark.parametrize(
  ("mutate", "code"),
  [
    (lambda v: v.__setitem__("dtype", "int8"), "BAD_DTYPE"),
    (lambda v: v.__setitem__("dtype", "f16"), "BAD_DTYPE"),
    (lambda v: v.__setitem__("layout", "strided"), "BAD_LAYOUT"),
    (lambda v: v.__setitem__("permission", "x"), "BAD_PERMISSION"),
    (lambda v: v.__setitem__("shape", [16, 0]), "BAD_SHAPE"),
    (lambda v: v.__setitem__("shape", []), "BAD_SHAPE"),
    (lambda v: v.__setitem__("shape", [16, True]), "BAD_SHAPE"),
    (lambda v: v.__setitem__("shape", [16, -2]), "BAD_SHAPE"),
    (lambda v: v.__setitem__("shape", [16.0, 2]), "BAD_SHAPE"),
    (lambda v: v.__setitem__("shape", "16x2"), "BAD_SHAPE"),
    (lambda v: v.__setitem__("handle", ""), "WRONG_TYPE"),
  ],
)
def test_each_view_defect_is_rejected_with_its_own_code(
  good: dict, contract: dict, mutate: Any, code: str
) -> None:
  mutate(_kernel(good)["ins"][0])
  got, field = _code(_resealed(good, contract), contract)
  assert got == code and "ins[0]" in field


def test_an_output_may_not_alias_a_read_only_input(good: dict, contract: dict) -> None:
  _kernel(good)["outs"][0]["alias"] = "a"
  assert _code(_resealed(good, contract), contract)[0] == "ILLEGAL_ALIAS"


def test_an_output_may_alias_a_writable_input(good: dict, contract: dict) -> None:
  _kernel(good)["ins"][0]["permission"] = "rw"
  _kernel(good)["outs"][0]["alias"] = "a"
  pc.validate_package(_resealed(good, contract), contract)


def test_an_alias_to_a_handle_that_does_not_exist_is_rejected(good: dict, contract: dict) -> None:
  _kernel(good)["outs"][0]["alias"] = "ghost"
  assert _code(_resealed(good, contract), contract)[0] == "ALIAS_UNKNOWN"


def test_view_handles_are_unique_within_a_kernel(good: dict, contract: dict) -> None:
  _kernel(good)["outs"][0]["handle"] = "a"
  assert _code(_resealed(good, contract), contract)[0] == "DUPLICATE_HANDLE"


def test_the_tensor_output_may_not_be_external_fp32(good: dict, contract: dict) -> None:
  _kernel(good)["outs"][0]["dtype"] = "f32"
  assert _code(_resealed(good, contract), contract)[0] == "EXTERNAL_DTYPE_FORBIDDEN"


def test_a_scalar_i32_output_is_a_legal_companion(good: dict, contract: dict) -> None:
  _kernel(good)["outs"].append(
    {
      "handle": "idx",
      "shape": [1],
      "dtype": "i32",
      "layout": "contiguous",
      "permission": "w",
      "alias": None,
      "actual_view": "l1.idx",
    }
  )
  pc.validate_package(_resealed(good, contract), contract)


# --- kernels, mode table, numeric profile -----------------------------------------------------


def test_an_unregistered_kernel_is_rejected(good: dict, contract: dict) -> None:
  _kernel(good)["kernel_id"] = "K-SOFTMAX"
  assert _code(_resealed(good, contract), contract)[0] == "UNREGISTERED_KERNEL"


def test_a_kernel_version_mismatch_is_rejected(good: dict, contract: dict) -> None:
  _kernel(good)["kernel_version"] = 2
  assert _code(_resealed(good, contract), contract)[0] == "KERNEL_VERSION_MISMATCH"


def test_a_mode_table_the_kernel_was_not_sealed_with_is_rejected(good: dict, contract: dict) -> None:
  _kernel(good)["mode_table"] = "tail"
  assert _code(_resealed(good, contract), contract)[0] == "MODE_TABLE_MISMATCH"


def test_numeric_profile_must_match_in_the_package_and_in_the_kernel(good: dict, contract: dict) -> None:
  good["numeric_profile_id"] = "w3_other"
  assert _code(good, contract)[0] == "NUMERIC_PROFILE_MISMATCH"
  good["numeric_profile_id"] = "w3_bf16_fp32_rod_v1"
  _kernel(good)["numeric_profile"] = "w3_other"
  assert _code(_resealed(good, contract), contract)[0] == "NUMERIC_PROFILE_MISMATCH"


def test_a_supported_numeric_profile_cannot_be_sealed_against_here(
  good: dict, contract: dict, monkeypatch: Any, tmp_path: Path
) -> None:
  profile = yaml.safe_load(pc.PROFILE.read_text())
  profile["support_status"] = "supported"
  fake = tmp_path / "profile.yaml"
  fake.write_text(yaml.safe_dump(profile))
  monkeypatch.setattr(pc, "PROFILE", fake)
  assert _code(good, contract)[0] == "NUMERIC_PROFILE_STATUS"


@pytest.mark.parametrize("method", ["implicit", "none", "zero", ""])
def test_a_chain_with_no_declared_seed_is_rejected(good: dict, contract: dict, method: str) -> None:
  _kernel(good)["initial"]["method"] = method
  assert _code(_resealed(good, contract), contract)[0] == "BAD_INITIAL"


@pytest.mark.parametrize("method", ["vtzero", "vtmv", "tile_load"])
def test_each_declared_seed_method_is_accepted(good: dict, contract: dict, method: str) -> None:
  _kernel(good)["initial"]["method"] = method
  pc.validate_package(_resealed(good, contract), contract)


# --- scratch, events, helpers -----------------------------------------------------------------


def test_a_scratch_bound_must_come_from_static_analysis(good: dict, contract: dict) -> None:
  _kernel(good)["scratch"][0]["bound_source"] = "measured"
  assert _code(_resealed(good, contract), contract)[0] == "BOUND_NOT_STATIC"


def test_scratch_without_a_positive_byte_count_is_rejected(good: dict, contract: dict) -> None:
  _kernel(good)["scratch"][0]["bytes"] = 0
  assert _code(_resealed(good, contract), contract)[0] == "BAD_SCRATCH"


def test_the_completion_event_is_required(good: dict, contract: dict) -> None:
  _kernel(good)["events"].pop("completion")
  assert _code(_resealed(good, contract), contract)[0] == "MISSING_FIELD"


def test_an_event_name_may_not_be_reused(good: dict, contract: dict) -> None:
  _kernel(good)["events"]["prerequisite"] = _kernel(good)["events"]["completion"]
  assert _code(_resealed(good, contract), contract)[0] == "DUPLICATE_EVENT"


def test_an_event_name_may_not_repeat_across_kernels_in_one_context(good: dict, contract: dict) -> None:
  second = copy.deepcopy(_kernel(good))
  second["ins"][0]["handle"], second["ins"][1]["handle"], second["outs"][0]["handle"] = "a2", "b2", "c2"
  good["binding_manifest"]["kernels"].append(second)
  assert _code(_resealed(good, contract), contract)[0] == "DUPLICATE_EVENT"


def test_helpers_resolve_to_a_generated_csr_logical_name_and_never_a_number(
  good: dict, contract: dict
) -> None:
  """A consumer can generate the real call: helper -> CSR logical name -> declaration kind."""
  names = {d["name"] for d in yaml.safe_load(pc.CSR_NAMES.read_text())["declared"]}
  for helper in _kernel(good)["helpers"]:
    assert helper["csr"] in names and not helper["csr"].startswith("0x")
  kinds = {h["name"]: h["kind"] for h in contract["helpers"]["registered"]}
  assert kinds["region_start"] == "REGION_START" and kinds["chain_final"] == "CHAIN_FINAL"


@pytest.mark.parametrize("csr", ["0x7c0", "1984", "nexus.mregion.bogus", "nexus.mregion.end"])
def test_a_helper_csr_that_is_a_number_or_the_wrong_name_is_rejected(
  good: dict, contract: dict, csr: str
) -> None:
  _kernel(good)["helpers"][0]["csr"] = csr
  assert _code(_resealed(good, contract), contract)[0] == "HELPER_CSR_MISMATCH"


def test_a_helper_that_is_not_declared_cannot_be_called(good: dict, contract: dict) -> None:
  _kernel(good)["helpers"][0]["name"] = "region_nuke"
  assert _code(_resealed(good, contract), contract)[0] == "UNDECLARED_HELPER"


@pytest.mark.parametrize("barrier", ["none", "", "asm", None])
def test_a_helper_without_a_compiler_barrier_is_rejected(good: dict, contract: dict, barrier: Any) -> None:
  _kernel(good)["helpers"][0]["barrier"] = barrier
  assert _code(_resealed(good, contract), contract)[0] == "BAD_BARRIER"


def test_a_helper_must_state_what_it_makes_observable(good: dict, contract: dict) -> None:
  _kernel(good)["helpers"][0]["observable"] = "   "
  assert _code(_resealed(good, contract), contract)[0] == "MISSING_OBSERVABLE"


# --- region declarations ----------------------------------------------------------------------


@pytest.mark.parametrize(
  ("steps", "code"),
  [
    (["CHAIN_CONTINUE", "CHAIN_FINAL"], "CHAIN_WITHOUT_FIRST"),
    (["CHAIN_FIRST", "CHAIN_FIRST", "CHAIN_FINAL"], "DUPLICATE_FIRST"),
    (["CHAIN_FIRST", "CHAIN_CONTINUE"], "CHAIN_WITHOUT_FINAL"),
    (["CHAIN_FIRST", "CHAIN_FINAL", "CHAIN_FINAL"], "CHAIN_WITHOUT_FINAL"),
    ([], "CHAIN_WITHOUT_FIRST"),
    (["CHAIN_FIRST", "BOGUS", "CHAIN_FINAL"], "BAD_CHAIN_KIND"),
  ],
)
def test_chain_declarations_are_checked_against_the_abi(
  good: dict, contract: dict, steps: list, code: str
) -> None:
  _kernel(good)["region"]["chains"][0]["steps"] = steps
  assert _code(_resealed(good, contract), contract)[0] == code


def test_a_single_step_first_and_final_chain_is_not_silently_accepted(good: dict, contract: dict) -> None:
  """FIRST|FINAL on one step is a different declaration shape; the package form needs both steps listed."""
  _kernel(good)["region"]["chains"][0]["steps"] = ["CHAIN_FIRST"]
  assert _code(_resealed(good, contract), contract)[0] == "CHAIN_WITHOUT_FINAL"


@pytest.mark.parametrize("flag", ["start", "end"])
def test_a_region_needs_both_start_and_end(good: dict, contract: dict, flag: str) -> None:
  _kernel(good)["region"][flag] = False
  assert _code(_resealed(good, contract), contract)[0] == "REGION_BOUNDS_MISSING"


# --- image and resources ----------------------------------------------------------------------


@pytest.mark.parametrize("key", ["builder", "toolchain_lock_sha256", "source_sha256"])
def test_an_image_with_no_controlled_source_is_rejected(good: dict, contract: dict, key: str) -> None:
  good["image"]["provenance"][key] = ""
  assert _code(_resealed(good, contract), contract) == ("NO_CONTROLLED_SOURCE", f"$.image.provenance.{key}")


def test_text_bytes_must_be_the_linked_length_everywhere(good: dict, contract: dict) -> None:
  good["image"]["text_bytes"] = 1024  # the abstract-IR estimate floor, not the linked length
  assert _code(_resealed(good, contract), contract)[0] == "TEXT_BYTES_DISAGREE"
  good = copy.deepcopy(json.loads(vp.EXAMPLE.read_text()))
  good["resource_manifest"]["text_bytes"] = 1000
  assert _code(_resealed(good, contract), contract)[0] == "TEXT_BYTES_DISAGREE"


def test_text_beyond_the_upstream_itcm_default_is_rejected(good: dict, contract: dict) -> None:
  for place in (good["image"], good["image"]["link_layout"], good["resource_manifest"]):
    place["text_bytes"] = 9000
  assert _code(_resealed(good, contract), contract)[0] == "TEXT_EXCEEDS_CAPACITY"


def test_text_and_data_segments_may_not_overlap(good: dict, contract: dict) -> None:
  good["image"]["link_layout"]["data_base"] = 0x80
  assert _code(_resealed(good, contract), contract)[0] == "SEGMENT_OVERLAP"


def test_the_entry_point_must_lie_inside_text(good: dict, contract: dict) -> None:
  good["image"]["entry"] = 0x9999
  assert _code(_resealed(good, contract), contract)[0] == "ENTRY_OUTSIDE_TEXT"


@pytest.mark.parametrize("bad", ["", "abc", "z" * 63, 5])
def test_the_image_hash_must_be_a_sha256_hex_string(good: dict, contract: dict, bad: Any) -> None:
  good["image"]["bytes_sha256"] = bad
  assert _code(_resealed(good, contract), contract)[0] == "BAD_HASH"


def test_the_resource_bound_must_be_static_analysis(good: dict, contract: dict) -> None:
  good["resource_manifest"]["bound_source"] = "measured"
  assert _code(_resealed(good, contract), contract)[0] == "BOUND_NOT_STATIC"


def test_target_dtypes_must_be_bf16_in_fp32_acc_bf16_out(good: dict, contract: dict) -> None:
  good["target_fingerprint"]["tensor_output_bytes"] = 4
  assert _code(_resealed(good, contract), contract)[0] == "TARGET_DTYPE_MISMATCH"


# --- hashing, strict text, patch mask, residency ----------------------------------------------


def test_a_tampered_package_fails_the_hash(good: dict, contract: dict) -> None:
  good["resource_manifest"]["scratch_bytes"] = 1
  assert _code(good, contract) == ("HASH_MISMATCH", "$.package_hash")


def test_deleting_a_required_contract_step_is_rejected_even_when_the_hash_is_recomputed(
  good: dict, contract: dict
) -> None:
  """CMP-14: the semantic checks do not depend on the hash, so a recomputed hash is not a pass."""
  _kernel(good)["events"].pop("completion")
  resealed = _resealed(good, contract)
  assert resealed["package_hash"] == pc.package_hash(resealed, contract)
  assert _code(resealed, contract)[0] == "MISSING_FIELD"


@pytest.mark.parametrize(
  "part", ["binding_manifest", "resource_manifest", "target_fingerprint", "patch_mask"]
)
def test_the_hash_covers_each_declared_part(good: dict, contract: dict, part: str) -> None:
  before = pc.package_hash(good, contract)
  changed = copy.deepcopy(good)
  if part == "binding_manifest":
    _kernel(changed)["clobber"].append("t4")
  elif part == "resource_manifest":
    changed["resource_manifest"]["scratch_bytes"] += 1
  elif part == "target_fingerprint":
    changed["target_fingerprint"]["rvv_vlen"] = 256
  else:
    changed["patch_mask"].append({"field": "param_area.other", "layer": "uce"})
  assert pc.package_hash(changed, contract) != before


def test_the_hash_covers_the_image_bytes_and_the_control_plane_and_the_profile(
  good: dict, contract: dict
) -> None:
  before = pc.package_hash(good, contract)
  for mutate in (
    lambda p: p["image"].__setitem__("bytes_sha256", "f" * 64),
    lambda p: p["control_plane"].__setitem__("artifact_hash", "9" * 64),
    lambda p: p.__setitem__("numeric_profile_id", "w3_other"),
  ):
    changed = copy.deepcopy(good)
    mutate(changed)
    assert pc.package_hash(changed, contract) != before


def test_the_hash_does_not_cover_itself(good: dict, contract: dict) -> None:
  a = pc.package_hash(good, contract)
  good["package_hash"] = "1" * 64
  assert pc.package_hash(good, contract) == a


def test_the_hash_is_independent_of_key_order(good: dict, contract: dict) -> None:
  reordered = json.loads(json.dumps(good, sort_keys=True))
  assert pc.package_hash(reordered, contract) == pc.package_hash(good, contract)


def test_a_hash_coverage_entry_this_consumer_cannot_compute_is_an_error(good: dict, contract: dict) -> None:
  contract["package"]["package_hash"]["covers"].append("host_upload_time")
  with pytest.raises(pc.PackageError) as exc:
    pc.package_hash(good, contract)
  assert exc.value.code == "HASH_COVERAGE_UNKNOWN"


def test_sealing_a_broken_package_is_refused(good: dict, contract: dict) -> None:
  _kernel(good).pop("outs")
  with pytest.raises(pc.PackageError):
    pc.seal({k: v for k, v in good.items() if k != "package_hash"}, contract)


@pytest.mark.parametrize(
  ("edit", "code"),
  [
    (lambda t: t.replace('"entry": 256', '"entry": 256, "entry": 300'), "DUPLICATE_KEY"),
    (
      lambda t: t.replace('"kernel_version": 1', '"kernel_version": 1, "kernel_version": 2'),
      "DUPLICATE_KEY",
    ),
    (lambda t: t.replace('"rvv_vlen": 128', '"rvv_vlen": NaN'), "NON_FINITE_NUMBER"),
    (lambda t: t.replace('"rvv_vlen": 128', '"rvv_vlen": Infinity'), "NON_FINITE_NUMBER"),
    (lambda t: t.replace('"rvv_vlen": 128', '"rvv_vlen": -Infinity'), "NON_FINITE_NUMBER"),
    (lambda t: t.replace('"rvv_vlen": 128', '"rvv_vlen": 1e999'), "NON_FINITE_NUMBER"),
    (lambda t: t.replace('"rvv_vlen": 128', '"rvv_vlen": 128.0'), "WRONG_TYPE"),
    (lambda t: t[:-10], "INVALID_JSON"),
    (lambda t: "[1, 2]", "WRONG_TYPE"),
    (lambda t: "{}", "MISSING_FIELD"),
  ],
)
def test_strict_text_parsing(good: dict, contract: dict, edit: Any, code: str) -> None:
  with pytest.raises(pc.PackageError) as exc:
    pc.load(edit(json.dumps(good)), contract)
  assert exc.value.code == code


def test_a_patch_to_a_field_not_in_the_mask_is_refused(good: dict, contract: dict) -> None:
  pc.patch_allowed(good, "param_area.shape", "runtime")
  with pytest.raises(pc.PackageError) as undeclared:
    pc.patch_allowed(good, "text.word0", "runtime")
  assert undeclared.value.code == "PATCH_UNDECLARED"
  with pytest.raises(pc.PackageError) as wrong_layer:
    pc.patch_allowed(good, "param_area.shape", "uce")
  assert wrong_layer.value.code == "PATCH_WRONG_LAYER"


def test_patch_mask_layers_and_duplicates_are_checked(good: dict, contract: dict) -> None:
  good["patch_mask"][0]["layer"] = "host"
  assert _code(_resealed(good, contract), contract)[0] == "BAD_PATCH_LAYER"
  good = copy.deepcopy(json.loads(vp.EXAMPLE.read_text()))
  good["patch_mask"].append(dict(good["patch_mask"][0]))
  assert _code(_resealed(good, contract), contract)[0] == "DUPLICATE_PATCH_FIELD"


def _layout(good: dict, text_base: int, data_base: int) -> dict:
  other = copy.deepcopy(good)
  other["image"]["link_layout"].update(text_base=text_base, data_base=data_base)
  return other


def test_two_prelinked_images_with_disjoint_ranges_coexist(good: dict) -> None:
  pc.layouts_coexist(good, _layout(good, 0x4000, 0x20000))


@pytest.mark.parametrize(
  ("text_base", "data_base"), [(0x100, 0x20000), (0x4000, 0x10080), (0x10010, 0x30000), (0x0, 0x10000)]
)
def test_overlapping_images_cannot_coexist(good: dict, text_base: int, data_base: int) -> None:
  with pytest.raises(pc.PackageError) as exc:
    pc.layouts_coexist(good, _layout(good, text_base, data_base))
  assert exc.value.code == "IMAGES_OVERLAP"


@pytest.mark.parametrize(
  ("refs", "ack", "code"),
  [(1, True, "SLOT_IN_USE"), (0, False, "INVALIDATION_NOT_ACKED"), (3, False, "SLOT_IN_USE")],
)
def test_a_slot_is_reusable_only_at_zero_references_after_the_invalidation_ack(
  refs: int, ack: bool, code: str
) -> None:
  with pytest.raises(pc.PackageError) as exc:
    pc.residency_reuse_allowed(refs, ack)
  assert exc.value.code == code
  pc.residency_reuse_allowed(0, True)


# --- the required contract is held by the validator -------------------------------------------


def test_the_schema_json_is_derived_from_the_contract_and_cannot_drift(contract: dict) -> None:
  errors: list[str] = []
  vp.check_schema_matches(contract, errors)
  assert errors == []
  contract["binding_manifest"]["view"]["dtypes"].append("f16")
  vp.check_schema_matches(contract, errors)
  assert any("schema" in e and "differs" in e for e in errors)


def test_a_hand_edited_schema_is_caught(contract: dict, tmp_path: Path) -> None:
  schema = json.loads(vp.SCHEMA.read_text())
  schema["properties"]["kernels"]["items"]["additionalProperties"] = True
  path = tmp_path / "schema.json"
  path.write_text(json.dumps(schema))
  errors: list[str] = []
  vp.check_schema_matches(contract, errors, path)
  assert errors


def test_a_broken_example_is_caught(contract: dict, good: dict, tmp_path: Path) -> None:
  _kernel(good).pop("outs")
  path = tmp_path / "example.json"
  path.write_text(json.dumps(good))
  errors: list[str] = []
  vp.check_example(contract, errors, path)
  assert any("rejected" in e for e in errors)
  errors = []
  vp.check_example(contract, errors, tmp_path / "absent.json")
  assert any("missing" in e for e in errors)


@pytest.mark.parametrize("domain", vp.REQUIRED_DOMAINS)
def test_deleting_a_version_domain_is_rejected(contract: dict, request_doc: dict, domain: str) -> None:
  contract["version_domains"].pop(domain)
  report = _report(contract, request_doc)
  assert report["valid"] is False and any(domain in e for e in report["errors"])


@pytest.mark.parametrize("field", vp.REQUIRED_OUTER_FIELDS)
def test_dropping_an_outer_required_field_from_the_contract_is_rejected(
  contract: dict, request_doc: dict, field: str
) -> None:
  contract["package"]["required_fields"].remove(field)
  assert _report(contract, request_doc)["valid"] is False


@pytest.mark.parametrize("field", vp.REQUIRED_ENTRY_FIELDS)
def test_dropping_a_kernel_entry_field_from_the_contract_is_rejected(
  contract: dict, request_doc: dict, field: str
) -> None:
  contract["binding_manifest"]["entry_required_fields"].remove(field)
  report = _report(contract, request_doc)
  assert report["valid"] is False and any("entry_required_fields" in e for e in report["errors"])


@pytest.mark.parametrize("field", vp.REQUIRED_VIEW_FIELDS)
def test_dropping_a_view_field_from_the_contract_is_rejected(
  contract: dict, request_doc: dict, field: str
) -> None:
  contract["binding_manifest"]["view_required_fields"].remove(field)
  assert _report(contract, request_doc)["valid"] is False


@pytest.mark.parametrize("part", vp.REQUIRED_HASH_COVERS)
def test_narrowing_what_the_hash_covers_is_rejected(contract: dict, request_doc: dict, part: str) -> None:
  contract["package"]["package_hash"]["covers"].remove(part)
  report = _report(contract, request_doc)
  assert report["valid"] is False and any("covers" in e for e in report["errors"])


@pytest.mark.parametrize("key", vp.REQUIRED_STRICT)
def test_relaxing_a_strictness_rule_is_rejected(contract: dict, request_doc: dict, key: str) -> None:
  contract["package"]["strict"][key] = "accept"
  report = _report(contract, request_doc)
  assert report["valid"] is False and any(key in e for e in report["errors"])


@pytest.mark.parametrize(
  ("mutate", "needle"),
  [
    (lambda c: c["binding_manifest"]["view"]["dtypes"].append("int8"), "dtypes"),
    (lambda c: c["binding_manifest"]["view"]["permissions"].append("x"), "permissions"),
    (lambda c: c["binding_manifest"]["initial"]["methods"].append("implicit"), "initial.methods"),
    (
      lambda c: c["binding_manifest"]["scratch"]["bound_source_values"].append("measured"),
      "bound_source_values",
    ),
    (lambda c: c["binding_manifest"]["events"].__setitem__("required", []), "events"),
    (
      lambda c: c["binding_manifest"]["tensor_dtype_rule"].__setitem__(
        "int8_new_on_tensor_path", "allowed"
      ),
      "INT8",
    ),
    (
      lambda c: c["binding_manifest"]["required_kernel_families"].remove("softmax"),
      "required_kernel_families",
    ),
    (lambda c: c["helpers"]["barrier_kinds"].append("none"), "barrier_kinds"),
    (lambda c: c["helpers"].__setitem__("hand_written_numeric_address", "allowed"), "hand_written"),
    (lambda c: c["helpers"]["registered"].pop(), "helpers.registered"),
    (lambda c: c["patch_mask"]["layers"].remove("mfe"), "patch_mask.layers"),
    (lambda c: c["patch_mask"].__setitem__("undeclared_field_patch", "allow"), "undeclared_field_patch"),
    (lambda c: c["build"]["states"].remove("MEASURED"), "build.states"),
    (lambda c: c["build"].__setitem__("loader_patches", "allowed"), "build"),
    (lambda c: c["build"]["bounds"].__setitem__("measured_is_evidence_only", False), "build.bounds"),
    (
      lambda c: c["trust_boundary"].__setitem__("loader_proves_machine_code", True),
      "loader_proves_machine_code",
    ),
    (
      lambda c: c["trust_boundary"].__setitem__("delete_required_step_with_recomputed_hash", "accept"),
      "delete_required_step",
    ),
    (lambda c: c["trust_boundary"].__setitem__("machine_code_trusted_via", []), "machine_code_trusted_via"),
    (lambda c: c["image"]["link_layout"].__setitem__("overlap", "allow"), "overlap"),
    (lambda c: c["image"]["link_layout"].__setitem__("relocation_by_loader", "allowed"), "relocation"),
    (lambda c: c["image"]["text_bytes"].pop("estimate_rejected"), "estimate_rejected"),
    (lambda c: c["image"]["provenance"]["required_fields"].remove("toolchain_lock_sha256"), "provenance"),
    (lambda c: c["residency"]["slots"].__setitem__("min_coexisting", 1), "residency.slots"),
    (lambda c: c["residency"]["slots"]["reuse_requires"].remove("invalidation_ack"), "reuse_requires"),
    (lambda c: c["residency"]["mutable_area"].__setitem__("isolated_from_text", False), "mutable_area"),
    (lambda c: c["package"]["numeric_profile"]["status_allowed"].append("supported"), "status_allowed"),
    (lambda c: c.__setitem__("status", "frozen"), "status"),
  ],
)
def test_each_contract_weakening_is_rejected(
  contract: dict, request_doc: dict, mutate: Any, needle: str
) -> None:
  mutate(contract)
  report = _report(contract, request_doc)
  assert report["valid"] is False and any(needle in e for e in report["errors"]), report["errors"][:3]


def test_registering_an_extra_kernel_is_a_deliberate_validator_change(
  contract: dict, request_doc: dict
) -> None:
  contract["binding_manifest"]["supported_kernels"].append(
    {"id": "K-NEW", "version": 1, "family": "add", "mode_table": "reference"}
  )
  assert any("K-NEW" in e and "required set" in e for e in _report(contract, request_doc)["errors"])


def test_the_required_kernel_cannot_be_deleted_or_re_pointed(contract: dict, request_doc: dict) -> None:
  contract["binding_manifest"]["supported_kernels"][0]["mode_table"] = "tail"
  assert any("mode_table" in e for e in _report(contract, request_doc)["errors"])
  contract["binding_manifest"]["supported_kernels"] = []
  assert any("K-VME-BF16-ACC" in e for e in _report(contract, request_doc)["errors"])


@pytest.mark.parametrize("name", sorted(vp.UNRESOLVED_REQUIRED))
def test_unsourced_slot_quantities_accept_only_unresolved(
  contract: dict, request_doc: dict, name: str
) -> None:
  for invention in (4, "4", "2+2", "log2Ceil(4)", 16384):
    c = copy.deepcopy(contract)
    c["image"][name] = invention
    assert any(name in e for e in _report(c, request_doc)["errors"]), invention
    c = copy.deepcopy(contract)
    c["image"][name]["value"] = invention
    assert any(name in e and "unresolved" in e for e in _report(c, request_doc)["errors"]), invention


@pytest.mark.parametrize("name", sorted(vp.UNRESOLVED_REQUIRED))
@pytest.mark.parametrize("key", ["width_bits", "bits", "count", "max", "expr", "rule"])
def test_no_key_can_smuggle_a_number_into_an_unresolved_slot_quantity(
  contract: dict, request_doc: dict, name: str, key: str
) -> None:
  contract["image"][name][key] = 4
  assert any(name in e and "carries" in e for e in _report(contract, request_doc)["errors"])


def test_deleting_an_unresolved_slot_quantity_does_not_make_it_derivable(
  contract: dict, request_doc: dict
) -> None:
  for name in vp.UNRESOLVED_REQUIRED:
    c = copy.deepcopy(contract)
    c["image"].pop(name)
    assert any(name in e for e in _report(c, request_doc)["errors"])


@pytest.mark.parametrize("name", sorted(vp.REQUIRED_CAPACITIES))
def test_a_hardware_default_cannot_be_widened_or_unsourced(
  contract: dict, request_doc: dict, name: str
) -> None:
  c = copy.deepcopy(contract)
  c["image"]["capacities"][name]["value"] *= 2
  assert any(name in e and "required" in e for e in _report(c, request_doc)["errors"])
  c = copy.deepcopy(contract)
  c["image"]["capacities"][name].pop("source")
  assert any(name in e and "source" in e for e in _report(c, request_doc)["errors"])


def test_every_cited_line_is_read_back_and_a_drifted_one_is_caught(
  contract: dict, request_doc: dict
) -> None:
  errors: list[str] = []
  assert vp.check_sources(contract, errors) >= 9 and errors == []
  for _, src in vp.iter_sources(contract):
    src["line"] += 1
  errors = _report(contract, request_doc)["errors"]
  assert sum("SOURCE_LINE_MISMATCH" in e for e in errors) >= 6


@pytest.mark.parametrize("mutation", ["line", "needle", "file"])
def test_one_nested_citation_that_drifts_is_caught(
  contract: dict, request_doc: dict, mutation: str
) -> None:
  src = contract["version_domains"]["control_plane"]["source"]
  if mutation == "line":
    src["line"] += 5
  elif mutation == "needle":
    src["needle"] = "something the file does not say"
  else:
    src["file"] = "pipeline_validator/does_not_exist.py"
  assert any("control_plane" in e for e in _report(contract, request_doc)["errors"])


def test_the_cited_upstream_defaults_really_say_8_32_128() -> None:
  for name in ("itcm_bytes", "dtcm_bytes", "stack_bytes"):
    src = pc.load_contract()["image"]["capacities"][name]["source"]
    text = vp._read_line(src["file"], src["line"])
    assert text is not None and src["needle"] in text


def test_the_legacy_package_object_really_is_version_1_and_unrelated() -> None:
  src = pc.load_contract()["version_domains"]["legacy_package"]["source"]
  assert "package_abi_version: int = 1" in (vp._read_line(src["file"], src["line"]) or "")


# --- the W5 field request ---------------------------------------------------------------------


def test_the_field_request_is_clean(contract: dict, request_doc: dict) -> None:
  errors: list[str] = []
  vp.check_field_request(request_doc, contract, errors)
  assert errors == []


@pytest.mark.parametrize(
  ("mutate", "needle"),
  [
    (lambda r: r["helpers"].pop(), "must be exactly"),
    (lambda r: r["helpers"][0].__setitem__("csr", "nexus.bogus"), "csr_name_map"),
    (lambda r: r["helpers"][0].__setitem__("address", 0x7C0), "INVENTED_VALUE"),
    (lambda r: r["helpers"][0].__setitem__("bits", 8), "INVENTED_VALUE"),
    (lambda r: r["package_fields"].pop(), "package_fields"),
    (lambda r: r["package_fields"].append({"name": "extra"}), "package_fields"),
    (
      lambda r: r["messages"].append({"message": "PROGRAM_INSTALL", "new": True}),
      "already in the C009 catalog",
    ),
    (lambda r: r["messages"][0].pop("new"), "not in the C009 catalog"),
    (lambda r: r.__setitem__("schema", "x"), "schema/package"),
    (lambda r: r.__setitem__("package", "W9"), "schema/package"),
  ],
)
def test_each_field_request_mutation_is_rejected(
  contract: dict, request_doc: dict, mutate: Any, needle: str
) -> None:
  mutate(request_doc)
  errors: list[str] = []
  vp.check_field_request(request_doc, contract, errors)
  assert any(needle in e for e in errors), errors


# --- structurally broken documents are reported, never raised ---------------------------------


@pytest.mark.parametrize("junk", [[], None, "x", 5])
def test_a_non_mapping_root_is_reported(contract: dict, request_doc: dict, junk: Any) -> None:
  for args in ((junk, request_doc), (contract, junk)):
    report = _report(*args)
    assert report["valid"] is False and any("root must be a mapping" in e for e in report["errors"])


@pytest.mark.parametrize(
  "section",
  [
    "version_domains",
    "package",
    "image",
    "residency",
    "binding_manifest",
    "helpers",
    "patch_mask",
    "build",
    "trust_boundary",
  ],
)
def test_deleting_any_contract_section_is_reported_not_raised(
  contract: dict, request_doc: dict, section: str
) -> None:
  contract.pop(section)
  report = _report(contract, request_doc)
  assert report["valid"] is False and report["errors"]


def test_the_required_sets_are_module_constants_not_read_from_the_yaml(contract: dict) -> None:
  before = (tuple(vp.REQUIRED_OUTER_FIELDS), dict(vp.REQUIRED_HELPERS), dict(vp.REQUIRED_KERNELS))
  contract["package"]["required_fields"] = []
  contract["helpers"]["registered"] = []
  assert (tuple(vp.REQUIRED_OUTER_FIELDS), dict(vp.REQUIRED_HELPERS), dict(vp.REQUIRED_KERNELS)) == before
  assert len(vp.REQUIRED_OUTER_FIELDS) == 9 and len(vp.REQUIRED_HELPERS) == 5


def test_the_contract_stays_draft_and_assigns_no_slot_count_or_encoding(contract: dict) -> None:
  assert contract["status"] == "draft"
  assert contract["image"]["slot_count"]["value"] == "unresolved"
  assert contract["image"]["slot_capacity"]["value"] == "unresolved"
  assert contract["helpers"]["hand_written_numeric_address"] == "forbidden"
