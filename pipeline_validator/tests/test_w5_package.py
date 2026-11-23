"""C037 / W5-01: strict cn_package/v1 codec, seal and read-only Loader.

Card pass-condition -> tests:
  round-trip field-for-field equality ............. test_round_trip_*
  seal + reload identical ......................... test_seal_*, test_real_build_*
  tampered contract layer, hash RECOMPUTED ........ test_recomputed_hash_does_not_rescue_*
  missing ins/outs, bad alias, duplicate keys,
  unknown field, non-finite, wrong target ......... test_negative_* (each names the field)
  independent cross-check .......................... test_matches_the_c014_reference_consumer
The C014 reference consumer (hardware/.../package_contract.py) is deliberately NOT imported by the
implementation; it is used here as a second, separately written oracle.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from pipeline_validator.package import (
  PackageError,
  encode,
  layouts_coexist,
  load_file,
  load_text,
  package_hash,
  parse_strict,
  patch_allowed,
  seal,
  validate,
  verify,
  verify_image,
)

REPO = Path(__file__).resolve().parents[2]
H = REPO / "hardware/coral_nexus"
EXAMPLE = H / "kernels/bindings/examples/k_vme_bf16_acc.package.json"
BINDING = H / "kernels/bindings/selftest_roundtrip.yaml"
CONFIG = H / "configs/cs1_c2.yaml"
TOOLCHAIN = REPO / "out/coral_nexus/cache/toolchain/extracted/toolchain_coralnpu_v2/bin"

sys.path.insert(0, str(H / "scripts/lib"))
import package_contract as ref  # noqa: E402  (independent C014 oracle, never imported by the code under test)


# Both consumers reject, but this implementation names the cause more precisely: a hand-written numeric CSR
# (`0x7c0`) is HAND_WRITTEN_ADDRESS here, while the reference's earlier name-mismatch check masks it as
# HELPER_CSR_MISMATCH. Every other mutation must agree on the exact code.
KNOWN_CODE_REFINEMENTS = {("HAND_WRITTEN_ADDRESS", "HELPER_CSR_MISMATCH")}


@pytest.fixture
def good() -> dict[str, Any]:
  return copy.deepcopy(json.loads(EXAMPLE.read_text()))


def kernel(package: dict[str, Any]) -> dict[str, Any]:
  return package["binding_manifest"]["kernels"][0]  # type: ignore[no-any-return]


def resealed(package: dict[str, Any]) -> dict[str, Any]:
  """Recompute the hash after an edit, so ONLY the semantic checks can reject it."""
  body = {k: v for k, v in package.items() if k != "package_hash"}
  return {**body, "package_hash": package_hash(body)}


def rejects(package: dict[str, Any], code: str, field_part: str) -> PackageError:
  with pytest.raises(PackageError) as caught:
    verify(package)
  assert caught.value.code == code, (caught.value.code, str(caught.value))
  assert field_part in caught.value.field, (field_part, caught.value.field)
  return caught.value


# ------------------------------------------------------------------------------------------ round trip
def test_round_trip_is_field_for_field_equal(good: dict[str, Any]) -> None:
  verify(good)
  again = parse_strict(encode(good))
  assert again == good
  assert encode(again) == encode(good)  # deterministic bytes


def test_loader_returns_a_frozen_view_equal_to_the_source(good: dict[str, Any]) -> None:
  loaded = load_text(encode(good))
  assert loaded.package_hash == good["package_hash"]
  assert loaded.to_dict() == good
  with pytest.raises(TypeError):
    loaded.view["package_hash"] = "x"  # type: ignore[index]
  with pytest.raises(TypeError):
    loaded.view["binding_manifest"]["kernels"][0]["kernel_id"] = "x"  # type: ignore[index]
  # to_dict hands out a copy, so editing it cannot reach the loaded package
  copy_out = loaded.to_dict()
  copy_out["numeric_profile_id"] = "other"
  assert loaded.to_dict() == good


# ------------------------------------------------------------------------------------------------- seal
def test_seal_fills_the_hash_and_never_mutates_the_input(good: dict[str, Any]) -> None:
  body = {k: v for k, v in good.items() if k != "package_hash"}
  before = copy.deepcopy(body)
  sealed = seal(body)
  assert body == before
  assert sealed["package_hash"] == good["package_hash"]
  verify(sealed)


def test_seal_refuses_a_package_that_breaks_the_contract(good: dict[str, Any]) -> None:
  body = {k: v for k, v in good.items() if k != "package_hash"}
  kernel(body)["ins"][0].pop("actual_view")
  with pytest.raises(PackageError) as caught:
    seal(body)
  assert caught.value.code == "MISSING_FIELD" and "ins[0].actual_view" in caught.value.field


def test_any_covered_byte_changes_the_hash(good: dict[str, Any]) -> None:
  base = good["package_hash"]
  edits = [
    lambda p: p["image"].__setitem__("bytes_sha256", "f" * 64),
    lambda p: p["resource_manifest"].__setitem__("stack_bytes", 64),
    lambda p: p["target_fingerprint"].__setitem__("context_count", 3),
    lambda p: p["patch_mask"].append({"field": "param_area.extra", "layer": "uce"}),
    lambda p: p["control_plane"].__setitem__("artifact_hash", "1" * 64),
    lambda p: kernel(p).__setitem__("source_node", "matmul_9"),
  ]
  for edit in edits:
    changed = copy.deepcopy(good)
    edit(changed)
    assert package_hash(changed) != base


def test_a_stale_hash_is_a_mismatch_that_names_the_field(good: dict[str, Any]) -> None:
  good["resource_manifest"]["stack_bytes"] = 64
  rejects(good, "HASH_MISMATCH", "$.package_hash")


def test_recomputed_hash_does_not_rescue_a_tampered_contract_layer(good: dict[str, Any]) -> None:
  """Delete a required step AND recompute the hash: the edit is rejected on its meaning."""
  cases = [
    (
      lambda p: kernel(p)["region"]["chains"][0]["steps"].remove("CHAIN_FINAL"),
      "CHAIN_WITHOUT_FINAL",
      "chains[0]",
    ),
    (lambda p: kernel(p).__setitem__("ins", []), "MISSING_VIEWS", "ins"),
    (
      lambda p: kernel(p)["scratch"][0].__setitem__("bound_source", "measured_peak"),
      "BOUND_NOT_STATIC",
      "bound_source",
    ),
    (lambda p: kernel(p)["region"].__setitem__("end", False), "REGION_BOUNDS_MISSING", "region"),
    (lambda p: kernel(p)["helpers"][0].__setitem__("barrier", "none"), "BAD_BARRIER", "helpers[0].barrier"),
    (
      lambda p: p.__setitem__("numeric_profile_id", "w3_int8"),
      "NUMERIC_PROFILE_MISMATCH",
      "numeric_profile_id",
    ),
  ]
  for edit, code, where in cases:
    tampered = copy.deepcopy(good)
    edit(tampered)
    rejects(resealed(tampered), code, where)


def test_image_bytes_must_match_the_named_identity(good: dict[str, Any]) -> None:
  with pytest.raises(PackageError) as caught:
    verify_image(good, b"not the linked image")
  assert caught.value.code == "IMAGE_BYTES_MISMATCH"


# --------------------------------------------------------------------------------------------- negatives
def test_negative_missing_ins_and_outs(good: dict[str, Any]) -> None:
  for side in ("ins", "outs"):
    broken = copy.deepcopy(good)
    del kernel(broken)[side]
    rejects(resealed(broken), "MISSING_FIELD", f".{side}")


def test_negative_alias_rules(good: dict[str, Any]) -> None:
  to_readonly = copy.deepcopy(good)
  kernel(to_readonly)["outs"][0]["alias"] = "a"  # `a` is a read-only input
  rejects(resealed(to_readonly), "ILLEGAL_ALIAS", "outs[0].alias")
  to_nothing = copy.deepcopy(good)
  kernel(to_nothing)["outs"][0]["alias"] = "no_such_handle"
  rejects(resealed(to_nothing), "ALIAS_UNKNOWN", "outs[0].alias")


def test_negative_duplicate_keys_are_rejected_at_parse() -> None:
  with pytest.raises(PackageError) as caught:
    parse_strict('{"package_schema": "cn_package/v1", "package_schema": "cn_package/v1"}')
  assert caught.value.code == "DUPLICATE_KEY"


def test_negative_non_finite_numbers_are_rejected_everywhere() -> None:
  for token in ("NaN", "Infinity", "-Infinity"):
    with pytest.raises(PackageError) as caught:
      parse_strict(f'{{"x": {token}}}')
    assert caught.value.code == "NON_FINITE_NUMBER"
  with pytest.raises(PackageError) as caught:
    encode({"x": float("nan")})  # type: ignore[arg-type]
  assert caught.value.code == "NON_FINITE_NUMBER"


def test_negative_unknown_fields_at_every_level(good: dict[str, Any]) -> None:
  for path in (
    ("extra",),
    ("image", "extra"),
    ("resource_manifest", "extra"),
    ("target_fingerprint", "extra"),
  ):
    broken = copy.deepcopy(good)
    node = broken
    for part in path[:-1]:
      node = node[part]
    node[path[-1]] = 1
    err = rejects(resealed(broken), "UNKNOWN_FIELD", path[-1])
    assert err.field.endswith("." + path[-1])
  nested = copy.deepcopy(good)
  kernel(nested)["surprise"] = True
  rejects(resealed(nested), "UNKNOWN_FIELD", "kernels[0].surprise")


def test_negative_wrong_version_domains(good: dict[str, Any]) -> None:
  for edit, where in (
    (lambda p: p.__setitem__("package_schema", "cn_package/v2"), "package_schema"),
    (lambda p: p["control_plane"].__setitem__("schema_version", 2), "control_plane.schema_version"),
    (lambda p: p["control_plane"].__setitem__("compiler_abi", "v2"), "control_plane.compiler_abi"),
  ):
    broken = copy.deepcopy(good)
    edit(broken)
    rejects(resealed(broken), "WRONG_VERSION", where)


def test_negative_types_are_strict(good: dict[str, Any]) -> None:
  broken = copy.deepcopy(good)
  broken["image"]["text_bytes"] = True  # a bool is not an integer length
  rejects(resealed(broken), "BAD_TEXT_BYTES", "text_bytes")
  broken = copy.deepcopy(good)
  kernel(broken)["kernel_version"] = 1.0
  rejects(resealed(broken), "KERNEL_VERSION_MISMATCH", "kernel_version")


def test_negative_unregistered_kernel_and_events(good: dict[str, Any]) -> None:
  broken = copy.deepcopy(good)
  kernel(broken)["kernel_id"] = "K-NOT-REGISTERED"
  rejects(resealed(broken), "UNREGISTERED_KERNEL", "kernel_id")
  second = copy.deepcopy(good)
  other = copy.deepcopy(kernel(second))
  second["binding_manifest"]["kernels"].append(other)  # same completion event name twice
  rejects(resealed(second), "DUPLICATE_EVENT", "events")


def test_negative_image_without_controlled_source_or_over_capacity(good: dict[str, Any]) -> None:
  broken = copy.deepcopy(good)
  broken["image"]["provenance"]["toolchain_lock_sha256"] = ""
  rejects(resealed(broken), "NO_CONTROLLED_SOURCE", "toolchain_lock_sha256")
  big = copy.deepcopy(good)
  big["image"]["text_bytes"] = big["image"]["link_layout"]["text_bytes"] = 8193
  big["resource_manifest"]["text_bytes"] = 8193
  rejects(resealed(big), "TEXT_EXCEEDS_CAPACITY", "text_bytes")


def test_negative_target_mismatch_refuses_the_load_and_names_the_field(good: dict[str, Any]) -> None:
  target = dict(good["target_fingerprint"])
  load_text(encode(good), target=target)  # the matching target loads
  target["rvv_vlen"] = 256
  with pytest.raises(PackageError) as caught:
    load_text(encode(good), target=target)
  assert caught.value.code == "TARGET_MISMATCH" and caught.value.field.endswith("rvv_vlen")
  del target["rvv_vlen"]
  with pytest.raises(PackageError) as missing:
    load_text(encode(good), target=target)
  assert missing.value.code == "TARGET_FIELD_MISSING"


# ----------------------------------------------------------------------------------------- patch / reuse
def test_patch_mask_is_a_whitelist_with_a_declared_layer(good: dict[str, Any]) -> None:
  patch_allowed(good, "param_area.shape", "runtime")
  with pytest.raises(PackageError) as wrong_layer:
    patch_allowed(good, "param_area.shape", "uce")
  assert wrong_layer.value.code == "PATCH_WRONG_LAYER"
  with pytest.raises(PackageError) as undeclared:
    patch_allowed(good, "text.bytes", "runtime")
  assert undeclared.value.code == "PATCH_UNDECLARED"


def test_fixed_images_coexist_only_when_disjoint(good: dict[str, Any]) -> None:
  other = copy.deepcopy(good)
  with pytest.raises(PackageError) as same:
    layouts_coexist(good, other)
  assert same.value.code == "IMAGES_OVERLAP"
  other["image"]["link_layout"].update(text_base=4096, data_base=69632)
  layouts_coexist(good, other)


# ---------------------------------------------------------------------------- independent cross-check
def test_matches_the_c014_reference_consumer(good: dict[str, Any]) -> None:
  """A second implementation of the same contract agrees on accept/reject and on the code."""
  contract = ref.load_contract()
  mutations: list[Any] = [
    lambda p: None,
    lambda p: p.__setitem__("extra", 1),
    lambda p: kernel(p).pop("ins"),
    lambda p: kernel(p)["outs"][0].__setitem__("alias", "a"),
    lambda p: kernel(p)["outs"][0].__setitem__("alias", "ghost"),
    lambda p: kernel(p)["ins"][0].__setitem__("actual_view", ""),
    lambda p: kernel(p)["ins"][0].__setitem__("dtype", "i8"),
    lambda p: kernel(p)["scratch"][0].__setitem__("bound_source", "measured_peak"),
    lambda p: kernel(p)["region"]["chains"][0]["steps"].pop(),
    lambda p: kernel(p)["helpers"][0].__setitem__("csr", "0x7c0"),
    lambda p: p["control_plane"].__setitem__("schema_version", 2),
    lambda p: p["image"]["provenance"].__setitem__("builder", " "),
    lambda p: p["target_fingerprint"].__setitem__("accumulator_bytes", 2),
  ]
  for index, mutate in enumerate(mutations):
    candidate = copy.deepcopy(good)
    mutate(candidate)
    candidate = resealed(candidate)
    try:
      verify(candidate)
      mine: str | None = None
    except PackageError as exc:
      mine = exc.code
    try:
      ref.validate_package(candidate, contract)
      theirs: str | None = None
    except ref.PackageError as exc:
      theirs = exc.code
    assert (mine is None) == (theirs is None), (
      f"mutation {index}: accept/reject differs ({mine!r} vs {theirs!r})"
    )
    assert mine == theirs or (mine, theirs) in KNOWN_CODE_REFINEMENTS, (
      f"mutation {index}: implementation {mine!r} vs reference {theirs!r}"
    )


def test_the_sealed_hash_equals_the_reference_hash(good: dict[str, Any]) -> None:
  assert package_hash(good) == ref.package_hash(good, ref.load_contract()) == good["package_hash"]


# --------------------------------------------------------------------------------------- real build
HAVE_TOOLCHAIN = TOOLCHAIN.is_dir()


@pytest.mark.skipif(not HAVE_TOOLCHAIN, reason="pinned C004 toolchain not extracted (cn env seed)")
def test_real_build_chain_seals_and_reloads_identically(tmp_path: Path) -> None:
  proc = subprocess.run(
    [
      "bash",
      str(H / "scripts/cn.sh"),
      "package",
      "--binding",
      str(BINDING),
      "--config",
      str(CONFIG),
      "--out",
      str(tmp_path),
    ],
    capture_output=True,
    text=True,
    timeout=300,
  )
  assert proc.returncode == 0, proc.stdout + proc.stderr
  report = json.loads((tmp_path / "package_report.json").read_text())
  assert report["reload_identical"] is True
  package = tmp_path / "package.json"
  image = tmp_path / "image.elf"
  loaded = load_file(
    package, image_path=image, target=json.loads(package.read_text())["target_fingerprint"]
  )
  assert loaded.package_hash == report["package_hash"]
  # the image really is the C018 bootstrap link: its hash is the one the committed measurement records
  measurement = json.loads((H / "verification/kernel_bootstrap/kernel_measurement.json").read_text())
  assert report["image_sha256"] == measurement["provenance"]["elf_sha256_layout_A"]
  assert report["resource_manifest"]["text_bytes"] == measurement["resource_manifest"]["text_bytes"]
  # flipping one byte of the shipped image is refused by name
  tampered = bytearray(image.read_bytes())
  tampered[0] ^= 0xFF
  bad = tmp_path / "tampered.elf"
  bad.write_bytes(bytes(tampered))
  with pytest.raises(PackageError) as caught:
    load_file(package, image_path=bad)
  assert caught.value.code == "IMAGE_BYTES_MISMATCH"


@pytest.mark.skipif(not HAVE_TOOLCHAIN, reason="pinned C004 toolchain not extracted (cn env seed)")
def test_real_build_refuses_a_binding_for_a_different_target(tmp_path: Path) -> None:
  bad = tmp_path / "bad.yaml"
  bad.write_text(BINDING.read_text().replace("context_count: 2", "context_count: 5"))
  proc = subprocess.run(
    [
      "bash",
      str(H / "scripts/cn.sh"),
      "package",
      "--binding",
      str(bad),
      "--config",
      str(CONFIG),
      "--out",
      str(tmp_path / "o"),
    ],
    capture_output=True,
    text=True,
    timeout=300,
  )
  assert proc.returncode == 1
  assert "TARGET_MISMATCH" in proc.stderr and not (tmp_path / "o/package.json").exists()


@pytest.mark.skipif(not HAVE_TOOLCHAIN, reason="pinned C004 toolchain not extracted (cn env seed)")
def test_real_build_checks_vlen_against_the_pinned_upstream_not_the_binding(tmp_path: Path) -> None:
  """A binding that states another positive VLEN must not seal and report LOADED: the Loader's target
  takes rvv_vlen from the upstream parameter record, not from the binding that authored the package."""
  bad = tmp_path / "bad.yaml"
  bad.write_text(BINDING.read_text().replace("rvv_vlen: 128", "rvv_vlen: 256"))
  proc = subprocess.run(
    [
      "bash",
      str(H / "scripts/cn.sh"),
      "package",
      "--binding",
      str(bad),
      "--config",
      str(CONFIG),
      "--out",
      str(tmp_path / "o"),
    ],
    capture_output=True,
    text=True,
    timeout=300,
  )
  assert proc.returncode == 1
  assert "TARGET_MISMATCH" in proc.stderr and "rvv_vlen" in proc.stderr
  assert not (tmp_path / "o/package.json").exists()


def test_the_package_command_is_registered_and_usage_errors_exit_two() -> None:
  proc = subprocess.run(
    ["bash", str(H / "scripts/cn.sh"), "package"], capture_output=True, text=True, timeout=60
  )
  assert proc.returncode == 2 and "--binding" in proc.stderr
