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
import hashlib
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


IMAGE = b"C037 test image: stands in for the linked ELF the package names by hash"
# The Coral target a load is checked against. It is read from the committed C014 example, never from the
# package under test, so a package that edits its own fingerprint cannot make itself match.
CORAL_TARGET: dict[str, Any] = json.loads(EXAMPLE.read_text())["target_fingerprint"]


def SEMANTIC() -> dict[str, Any]:
  """The validator target and actual launch bindings the wrapped pow body is semantically loaded against
  (what `cn package` uses: the default hardware, runtime fidelity, the pow CLI binding)."""
  from pipeline_validator.config import HardwareConfig, SimConfig
  from pipeline_validator.execution_ir import GlobalBinding

  return {
    "hw": HardwareConfig(),
    "sim": SimConfig(fidelity="runtime"),
    "actual_bindings": {"Y": GlobalBinding("Y", 0x100000, 131072, "rw")},
  }


def TARGET() -> dict[str, Any]:
  """Everything `load_text` needs besides the package and body texts: SEMANTIC plus the shipped image bytes
  and the independent Coral target (both mandatory)."""
  return {**SEMANTIC(), "image_bytes": IMAGE, "target": copy.deepcopy(CORAL_TARGET)}


@pytest.fixture(scope="module")
def compiled_text() -> str:
  """A real serialized CompiledProgram (the pow workload the build wraps)."""
  from pipeline_validator.compiled_program import serialize_compiled_program
  from pipeline_validator.compiler import compile_program
  from pipeline_validator.config import HardwareConfig, SimConfig
  from pipeline_validator.workloads import PowWorkload

  hw = HardwareConfig()
  return serialize_compiled_program(
    compile_program(PowWorkload(num_group_chunks=1, hw=hw).module, hw, SimConfig(fidelity="runtime"))
  )


@pytest.fixture
def good(compiled_text: str) -> dict[str, Any]:
  """The committed C014 example, whose placeholder control-plane digests are replaced by the real
  program's, then resealed (the committed example itself stays byte-identical)."""
  from pipeline_validator.compiled_program import parse_compiled_program

  package = copy.deepcopy(json.loads(EXAMPLE.read_text()))
  package["image"]["bytes_sha256"] = hashlib.sha256(IMAGE).hexdigest()
  program = parse_compiled_program(compiled_text)
  package["control_plane"] = {
    "schema_version": program.schema_version,
    "compiler_abi": program.compiler_abi,
    "artifact_hash": program.artifact_hash,
    "source_hash": program.source_hash,
  }
  return resealed(package)


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


def test_loader_returns_a_frozen_view_equal_to_the_source(good: dict[str, Any], compiled_text: str) -> None:
  loaded = load_text(encode(good), compiled_text=compiled_text, **TARGET())
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
def test_seal_fills_the_hash_and_never_mutates_the_input(good: dict[str, Any], compiled_text: str) -> None:
  body = {k: v for k, v in good.items() if k != "package_hash"}
  before = copy.deepcopy(body)
  sealed = seal(body, compiled_text=compiled_text, **SEMANTIC())
  assert body == before
  assert sealed["package_hash"] == good["package_hash"]
  verify(sealed)


def test_seal_refuses_a_package_that_breaks_the_contract(good: dict[str, Any], compiled_text: str) -> None:
  body = {k: v for k, v in good.items() if k != "package_hash"}
  kernel(body)["ins"][0].pop("actual_view")
  with pytest.raises(PackageError) as caught:
    seal(body, compiled_text=compiled_text, **SEMANTIC())
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


def test_negative_target_mismatch_refuses_the_load_and_names_the_field(
  good: dict[str, Any], compiled_text: str
) -> None:
  load_text(encode(good), compiled_text=compiled_text, **TARGET())  # the matching target loads
  for key, value, code in (
    ("rvv_vlen", 256, "TARGET_MISMATCH"),
    ("rvv_vlen", None, "TARGET_FIELD_MISSING"),
  ):
    target = copy.deepcopy(CORAL_TARGET)
    if value is None:
      del target[key]
    else:
      target[key] = value
    with pytest.raises(PackageError) as caught:
      load_text(encode(good), compiled_text=compiled_text, **{**TARGET(), "target": target})
    assert caught.value.code == code and caught.value.field.endswith(key)


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


# ----------------------------------------------------------------------------------- control plane body
def test_a_substituted_digest_with_a_recomputed_outer_hash_is_refused(
  good: dict[str, Any], compiled_text: str
) -> None:
  """The digest must belong to the wrapped program: a syntax-valid '9'*64 plus a fresh package_hash
  used to load."""
  forged = copy.deepcopy(good)
  forged["control_plane"]["artifact_hash"] = "9" * 64
  forged = resealed(forged)
  verify(forged)  # structurally fine, and the outer hash is consistent: only the body can refuse it
  with pytest.raises(PackageError) as caught:
    load_text(encode(forged), compiled_text=compiled_text, **TARGET())
  assert caught.value.code == "CONTROL_PLANE_MISMATCH"
  assert caught.value.field == "$.control_plane.artifact_hash"
  with pytest.raises(PackageError) as at_seal:
    seal(
      {k: v for k, v in forged.items() if k != "package_hash"}, compiled_text=compiled_text, **SEMANTIC()
    )
  assert at_seal.value.code == "CONTROL_PLANE_MISMATCH"


def test_a_body_edit_without_a_recomputed_inner_digest_is_refused_by_the_codec(
  good: dict[str, Any], compiled_text: str
) -> None:
  """Codec layer only: the inner digest is NOT recomputed here, so parse_compiled_program refuses it.
  The rehash case CMP-14 is about is test_a_semantically_forged_body_with_fresh_inner_and_outer_seals."""
  body = json.loads(compiled_text)
  body["source_hash"] = "a" * 64  # an inner field that the program's own digest covers
  edited = json.dumps(body, sort_keys=True, indent=2) + "\n"
  with pytest.raises(PackageError) as caught:
    load_text(encode(good), compiled_text=edited, **TARGET())
  assert (
    caught.value.code == "CONTROL_PLANE_INVALID"
  )  # the existing loader's recomputed artifact_hash disagrees
  # and a package re-pointed at the edited body's claimed digest cannot be sealed either
  forged = copy.deepcopy(good)
  forged["control_plane"]["source_hash"] = "a" * 64
  with pytest.raises(PackageError) as at_seal:
    seal({k: v for k, v in forged.items() if k != "package_hash"}, compiled_text=edited, **SEMANTIC())
  assert at_seal.value.code == "CONTROL_PLANE_INVALID"


def test_a_control_plane_body_that_is_not_a_compiled_program_is_refused(
  good: dict[str, Any], compiled_text: str
) -> None:
  for bad in ("", "{}", "not json", compiled_text.replace('"v3"', '"v2"', 1), '{"a": 1, "a": 2}'):
    with pytest.raises(PackageError) as caught:
      load_text(encode(good), compiled_text=bad, **TARGET())
    assert caught.value.code == "CONTROL_PLANE_INVALID", bad[:30]


def _two_buffer_artifact() -> tuple[Any, Any, Any, Any]:
  """The test_double_release_rejected fixture (a data-dependent matmul/pow with >=2 L2 buffers)."""
  from pipeline_validator.compiler import compile_program
  from pipeline_validator.config import HardwareConfig, SimConfig
  from pipeline_validator.execution_ir import ExecModel
  from pipeline_validator.workload_ir import parse_workload_ir

  hw = HardwareConfig().with_overrides(num_dma_channels=2, hbm_fixed_latency_cycles=10)
  sim = SimConfig(max_cycles=200000, context_count=4)
  module = parse_workload_ir((REPO / "examples/workloads/matmul_pow_data_dep.mlir").read_text())
  artifact = compile_program(module, hw, sim)
  assert isinstance(artifact.entry, ExecModel)
  _id, task = next(
    (key, task)
    for key, task in artifact.entry.tasks.items()
    if task.layout is not None and len(task.layout.buffer_layouts) >= 2
  )
  return artifact, hw, sim, task


def _wrap(good: dict[str, Any], program: Any) -> tuple[dict[str, Any], str]:
  """A package whose control plane names `program`, with a freshly computed outer package_hash."""
  from pipeline_validator.compiled_program import serialize_compiled_program

  package = copy.deepcopy(good)
  package["control_plane"] = {
    "schema_version": program.schema_version,
    "compiler_abi": program.compiler_abi,
    "artifact_hash": program.artifact_hash,
    "source_hash": program.source_hash,
  }
  return resealed(package), serialize_compiled_program(program)


def test_a_semantically_forged_body_with_fresh_inner_and_outer_seals_is_refused(
  good: dict[str, Any],
) -> None:
  """CMP-14: a tampered contract layer is rejected even after rehash. A double RELEASE_L2 is inserted into
  a real task and the program is re-sealed (fresh inner artifact_hash), then wrapped in a package with a
  fresh outer package_hash. The codec accepts every digest; only the semantic Loader sees the forgery."""
  from dataclasses import replace

  from pipeline_validator.compiled_program import (
    parse_compiled_program,
    seal_program,
    serialize_compiled_program,
  )
  from pipeline_validator.execution_ir import ExecGroupActionOp, GlobalBinding

  artifact, hw, sim, task = _two_buffer_artifact()
  bindings = {
    guard.name: GlobalBinding(
      guard.name, 0x100000 + index * 0x1000000, guard.minimum_bytes, guard.permissions
    )
    for index, guard in enumerate(artifact.binding_guards)
  }
  target = {"hw": hw, "sim": sim, "actual_bindings": bindings}
  load_args = {**target, "image_bytes": IMAGE, "target": copy.deepcopy(CORAL_TARGET)}

  # control: the genuine program wraps, seals and loads
  genuine, genuine_text = _wrap(good, artifact)
  seal({k: v for k, v in genuine.items() if k != "package_hash"}, compiled_text=genuine_text, **target)
  load_text(encode(genuine), compiled_text=genuine_text, **load_args)

  last = max(i for i, a in enumerate(task.actions) if a.op is ExecGroupActionOp.RELEASE_L2)
  release = task.actions[last]
  forged_action = replace(
    release, dst=f"{release.dst}:forged", instruction_id=f"{task.binding_id}:group:forged_release"
  )
  actions = (*task.actions[: last + 1], forged_action, *task.actions[last + 1 :])
  model = replace(
    artifact.entry, tasks={**artifact.entry.tasks, task.binding_id: replace(task, actions=actions)}
  )
  corrupted = seal_program(replace(artifact, entry=model))  # fresh inner digest
  assert corrupted.artifact_hash != artifact.artifact_hash
  forged, forged_text = _wrap(good, corrupted)  # fresh outer hash
  assert serialize_compiled_program(corrupted) == forged_text

  # every digest is self-consistent: the codec and the structural package check both accept it
  assert parse_compiled_program(forged_text).artifact_hash == corrupted.artifact_hash
  verify(forged)
  assert forged["package_hash"] == package_hash({k: v for k, v in forged.items() if k != "package_hash"})

  with pytest.raises(PackageError) as at_load:
    load_text(encode(forged), compiled_text=forged_text, **load_args)
  assert at_load.value.code == "CONTROL_PLANE_REJECTED"
  assert "already released L2 view" in str(at_load.value)
  with pytest.raises(PackageError) as at_seal:
    seal({k: v for k, v in forged.items() if k != "package_hash"}, compiled_text=forged_text, **target)
  assert at_seal.value.code == "CONTROL_PLANE_REJECTED"


def test_the_semantic_loader_also_checks_the_actual_bindings(
  good: dict[str, Any], compiled_text: str
) -> None:
  """The same program with a binding that violates its compiled range guard is not loadable."""
  from pipeline_validator.execution_ir import GlobalBinding

  args = TARGET()
  args["actual_bindings"] = {"Y": GlobalBinding("Y", 0x100000, 16, "rw")}  # smaller than the guard
  with pytest.raises(PackageError) as caught:
    load_text(encode(good), compiled_text=compiled_text, **args)
  assert caught.value.code == "CONTROL_PLANE_REJECTED" and "range guard" in str(caught.value)


# ---------------------------------------------------------------- installation proof is not optional
def test_load_text_requires_the_image_and_the_independent_coral_target(
  good: dict[str, Any], compiled_text: str
) -> None:
  """A load that omits the ELF or the Coral fingerprint proves less than it looks like, so it cannot be
  expressed: both are required keywords at the entry point (metadata-only inspection is `verify`)."""
  full = TARGET()
  load_text(encode(good), compiled_text=compiled_text, **full)
  for missing in ("image_bytes", "target"):
    args = {k: v for k, v in full.items() if k != missing}
    with pytest.raises(TypeError, match=missing):
      load_text(encode(good), compiled_text=compiled_text, **args)
  verify(good)  # metadata-only inspection stays available and needs neither


def test_load_file_requires_the_image_and_the_independent_coral_target(tmp_path: Path) -> None:
  package = tmp_path / "package.json"
  compiled = tmp_path / "compiled_program.json"
  package.write_text("{}")
  compiled.write_text("{}")
  for missing in ("image_path", "target"):
    args: dict[str, Any] = {**SEMANTIC(), "image_path": tmp_path / "image.elf", "target": CORAL_TARGET}
    del args[missing]
    with pytest.raises(TypeError, match=missing):
      load_file(package, compiled_path=compiled, **args)


def test_a_wrong_image_or_a_foreign_target_is_refused_at_load_text(
  good: dict[str, Any], compiled_text: str
) -> None:
  with pytest.raises(PackageError) as wrong_image:
    load_text(encode(good), compiled_text=compiled_text, **{**TARGET(), "image_bytes": IMAGE + b"!"})
  assert wrong_image.value.code == "IMAGE_BYTES_MISMATCH"
  # a package cannot vouch for itself: its own fingerprint is not an acceptable `target`
  foreign = {**CORAL_TARGET, "context_count": CORAL_TARGET["context_count"] + 1}
  with pytest.raises(PackageError) as wrong_target:
    load_text(encode(good), compiled_text=compiled_text, **{**TARGET(), "target": foreign})
  assert wrong_target.value.code == "TARGET_MISMATCH"


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
    package,
    compiled_path=tmp_path / "compiled_program.json",
    **SEMANTIC(),
    image_path=image,
    target=copy.deepcopy(CORAL_TARGET),  # independent of the package's own fingerprint
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
    load_file(
      package,
      compiled_path=tmp_path / "compiled_program.json",
      **SEMANTIC(),
      image_path=bad,
      target=copy.deepcopy(CORAL_TARGET),
    )
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
