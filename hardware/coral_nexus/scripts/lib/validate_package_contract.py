"""Validate the C014 cn_package/v1 contract (w5_package.contract.yaml) and its derived artifacts.

What it checks:

1. Every `source: {file, line, needle}` anywhere in the contract is READ BACK from the pinned tree or the
   validator (one whole-tree walk, so a nested citation cannot escape).
2. The required contract is held HERE, not in the YAML: version domains, required outer/image/entry/view/
   helper fields, dtype and permission sets, the supported-kernel registry, build states, and the explicit
   trust boundary. A synchronized edit that weakens the YAML and its guard together cannot pass.
3. Unsourced quantities stay `unresolved` with an owner and a reason: slot count and slot capacity may not
   be filled with a number (the C010/C013 rule), and the fixed upstream defaults are read back.
4. `kernels/bindings/schema.json` is DERIVED from the YAML; it is recomputed here and must match, so the two
   cannot drift. The committed legal example is run through the strict consumer.
5. The consumer is exercised on the committed example: a complete binding seals and reloads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB))

import package_contract as pc  # noqa: E402

H = LIB.parents[1]
REPO = LIB.parents[3]
UPSTREAM = H / "upstream" / "coralnpu"
SCHEMA = H / "kernels" / "bindings" / "schema.json"
EXAMPLE = H / "kernels" / "bindings" / "examples" / "k_vme_bf16_acc.package.json"
REQUEST = H / "protocol" / "field_requests" / "w5.yaml"
CATALOG = H / "protocol" / "cn_protocol_v1.yaml"

# ---------------------------------------------------------------------------------------------------
# The required contract. Held here so one edit cannot weaken the YAML and its guard together.
# ---------------------------------------------------------------------------------------------------
REQUIRED_DOMAINS = ("package", "control_plane", "hardware_yaml", "legacy_package")
REQUIRED_DOMAIN_VALUES: dict[str, Any] = {
  "package": {"id": "cn_package/v1"},
  "control_plane": {"schema_version": 3, "compiler_abi": "v3", "wrapped_unchanged": True},
  "hardware_yaml": {"schema_version": 2},
  "legacy_package": {"package_abi_version": 1, "status": "not_inherited"},
}
REQUIRED_OUTER_FIELDS = (
  "package_schema",
  "control_plane",
  "image",
  "binding_manifest",
  "resource_manifest",
  "target_fingerprint",
  "numeric_profile_id",
  "patch_mask",
  "package_hash",
)
REQUIRED_STRICT = ("unknown_fields", "duplicate_keys", "non_finite_numbers", "wrong_version", "wrong_types")
REQUIRED_HASH_COVERS = (
  "image_sha256",
  "binding_manifest",
  "resource_manifest",
  "target_fingerprint",
  "patch_mask",
  "numeric_profile_id",
  "control_plane_artifact_hash",
)
REQUIRED_IMAGE_FIELDS = ("bytes_sha256", "text_bytes", "entry", "link_layout", "provenance")
REQUIRED_LINK_FIELDS = ("text_base", "text_bytes", "data_base", "data_bytes")
REQUIRED_PROVENANCE = ("builder", "toolchain_lock_sha256", "source_sha256")
REQUIRED_ENTRY_FIELDS = (
  "kernel_id",
  "kernel_version",
  "source_node",
  "ins",
  "outs",
  "initial",
  "scratch",
  "clobber",
  "events",
  "numeric_profile",
  "mode_table",
  "region",
  "patch_mask",
  "helpers",
)
REQUIRED_VIEW_FIELDS = ("handle", "shape", "dtype", "layout", "permission", "alias", "actual_view")
REQUIRED_DTYPES = ("bf16", "f32", "i32")
REQUIRED_PERMISSIONS = ("r", "w", "rw")
REQUIRED_INITIAL = ("vtzero", "vtmv", "tile_load")
REQUIRED_FAMILIES = (
  "matmul",
  "gather",
  "scatter",
  "softmax",
  "norm",
  "reduce_sum",
  "pow",
  "add",
  "recurrence",
)
REQUIRED_KERNELS = {"K-VME-BF16-ACC": {"version": 1, "family": "matmul", "mode_table": "reference"}}
REQUIRED_HELPERS = {
  "region_start": ("nexus.mregion.start", "REGION_START"),
  "region_end": ("nexus.mregion.end", "REGION_END"),
  "chain_first": ("nexus.mregion.first", "CHAIN_FIRST"),
  "chain_continue": ("nexus.mregion.continue", "CHAIN_CONTINUE"),
  "chain_final": ("nexus.mregion.final", "CHAIN_FINAL"),
}
REQUIRED_HELPER_FIELDS = ("name", "csr", "observable", "barrier", "params", "resources")
REQUIRED_BARRIERS = ("volatile_asm_memory_clobber", "compiler_fence_seq_cst")
REQUIRED_PATCH_LAYERS = ("runtime", "uce", "mfe")
REQUIRED_BUILD_STATES = ("AUTHORED", "COMPILED", "MEASURED", "RECHECKED", "SEALED", "LOADED")
# Upstream defaults the contract cites; each is also read back from the pinned tree through its source.
REQUIRED_CAPACITIES = {"itcm_bytes": 8192, "dtcm_bytes": 32768, "stack_bytes": 128}
UNRESOLVED_REQUIRED = {"slot_count": "C018", "slot_capacity": "C018"}
TRUST_FALSE = ("loader_proves_machine_code",)
TRUST_REQUIRED_TRUE_KEYS = ("delete_required_step_with_recomputed_hash",)


# ---------------------------------------------------------------------------------------------------
def _read_line(file: str, line: int) -> str | None:
  for root in (REPO, UPSTREAM):
    path = root / file
    if path.is_file():
      lines = path.read_text().splitlines()
      return lines[line - 1] if 0 < line <= len(lines) else None
  return None


def iter_sources(node: Any, where: str = "") -> list[tuple[str, dict]]:
  found: list[tuple[str, dict]] = []
  if isinstance(node, dict):
    if isinstance(node.get("file"), str) and isinstance(node.get("line"), int):
      found.append((where, node))
    for key, value in node.items():
      found.extend(iter_sources(value, f"{where}.{key}"))
  elif isinstance(node, list):
    for index, value in enumerate(node):
      found.extend(iter_sources(value, f"{where}[{index}]"))
  return found


def check_sources(contract: dict[str, Any], errors: list[str]) -> int:
  count = 0
  for where, src in iter_sources(contract):
    count += 1
    if "needle" not in src:
      errors.append(f"contract{where}: MISSING_SOURCE (need file/line/needle)")
      continue
    text = _read_line(src["file"], src["line"])
    if text is None:
      errors.append(f"contract{where}: SOURCE_FILE_MISSING {src['file']}:{src['line']}")
    elif str(src["needle"]) not in text:
      errors.append(
        f"contract{where}: SOURCE_LINE_MISMATCH {src['file']}:{src['line']} lacks {src['needle']!r}"
      )
  return count


def _eq(errors: list[str], where: str, got: Any, want: Any) -> None:
  if got != want:
    errors.append(f"{where}: {got!r} != required {want!r}")


def check_required(contract: dict[str, Any], errors: list[str]) -> None:
  dom = contract["version_domains"]
  for name in REQUIRED_DOMAINS:
    if name not in dom:
      errors.append(f"version_domains.{name}: required domain is missing")
      continue
    for key, want in REQUIRED_DOMAIN_VALUES[name].items():
      _eq(errors, f"version_domains.{name}.{key}", dom[name].get(key), want)
  if len(
    {
      json.dumps(dom[n].get("id", dom[n].get("schema_version", dom[n].get("package_abi_version"))))
      for n in dom
      if n != "rule"
    }
  ) != len(REQUIRED_DOMAINS):
    errors.append(
      "version_domains: two domains share a version number (a number is never reused across domains)"
    )
  if sorted(dom["package"].get("independent_of", [])) != sorted(
    ["control_plane", "hardware_yaml", "legacy_package"]
  ):
    errors.append(
      "version_domains.package.independent_of: must name control_plane, hardware_yaml, legacy_package"
    )
  pkg = contract["package"]
  _eq(errors, "package.required_fields", tuple(pkg["required_fields"]), REQUIRED_OUTER_FIELDS)
  for key in REQUIRED_STRICT:
    if pkg["strict"].get(key) != "reject":
      errors.append(f"package.strict.{key}: must be 'reject'")
  _eq(errors, "package.package_hash.covers", tuple(pkg["package_hash"]["covers"]), REQUIRED_HASH_COVERS)
  if pkg["package_hash"].get("excludes_itself") is not True:
    errors.append(
      "package.package_hash.excludes_itself: must be true (a hash that covers itself is a cycle)"
    )
  if pkg["package_hash"].get("algorithm") != "sha256":
    errors.append("package.package_hash.algorithm: must be sha256")
  if pkg["numeric_profile"].get("status_allowed") != ["experimental"]:
    errors.append("package.numeric_profile.status_allowed: only 'experimental' may be sealed here")
  if pkg["numeric_profile"].get("must_equal_id") != "w3_bf16_fp32_rod_v1":
    errors.append("package.numeric_profile.must_equal_id: must name the C011 profile id")

  img = contract["image"]
  _eq(errors, "image.required_fields", tuple(img["required_fields"]), REQUIRED_IMAGE_FIELDS)
  _eq(
    errors,
    "image.link_layout.required_fields",
    tuple(img["link_layout"]["required_fields"]),
    REQUIRED_LINK_FIELDS,
  )
  _eq(
    errors,
    "image.provenance.required_fields",
    tuple(img["provenance"]["required_fields"]),
    REQUIRED_PROVENANCE,
  )
  if (
    img["link_layout"].get("overlap") != "reject"
    or img["link_layout"].get("relocation_by_loader") != "forbidden"
  ):
    errors.append("image.link_layout: overlap must be rejected and loader relocation forbidden")
  if img["link_layout"].get("fixed_prelink") is not True:
    errors.append("image.link_layout.fixed_prelink: must be true")
  if "estimate_rejected" not in img["text_bytes"]:
    errors.append("image.text_bytes.estimate_rejected: an estimated text length must stay rejected")
  for name, want in REQUIRED_CAPACITIES.items():
    cap = img["capacities"].get(name)
    if cap is None:
      errors.append(f"image.capacities.{name}: required capacity is missing")
    else:
      _eq(errors, f"image.capacities.{name}.value", cap.get("value"), want)
      if "source" not in cap:
        errors.append(f"image.capacities.{name}: a hardware default needs its own source")
  for name, owner in UNRESOLVED_REQUIRED.items():
    item = img.get(name)
    if not isinstance(item, dict) or item.get("value") != "unresolved":
      errors.append(f"image.{name}: must stay 'unresolved' (a number here would be invented)")
      continue
    _eq(errors, f"image.{name}.owner", item.get("owner"), owner)
    if not str(item.get("reason") or "").strip():
      errors.append(f"image.{name}: an unresolved quantity needs a reason")
    extra = set(item) - {"value", "owner", "reason"}
    if extra:
      errors.append(
        f"image.{name}: carries {sorted(extra)}; an unsourced entry holds only value/owner/reason"
      )

  res = contract["residency"]
  if res["slots"].get("min_coexisting") != 2 or res["slots"].get("non_overlapping") is not True:
    errors.append("residency.slots: at least two non-overlapping images must coexist")
  if sorted(res["slots"].get("reuse_requires", [])) != ["invalidation_ack", "zero_references"]:
    errors.append("residency.slots.reuse_requires: must be zero_references and invalidation_ack")
  if (
    res["mutable_area"].get("isolated_from_text") is not True
    or res["mutable_area"].get("per_context_instance") is not True
  ):
    errors.append("residency.mutable_area: must be isolated from text and per Context")
  _eq(
    errors, "residency.identity", tuple(res["identity"]), ("program_id", "version", "package_hash", "epoch")
  )

  bm = contract["binding_manifest"]
  _eq(
    errors,
    "binding_manifest.entry_required_fields",
    tuple(bm["entry_required_fields"]),
    REQUIRED_ENTRY_FIELDS,
  )
  _eq(
    errors, "binding_manifest.view_required_fields", tuple(bm["view_required_fields"]), REQUIRED_VIEW_FIELDS
  )
  _eq(errors, "binding_manifest.view.dtypes", tuple(bm["view"]["dtypes"]), REQUIRED_DTYPES)
  _eq(errors, "binding_manifest.view.permissions", tuple(bm["view"]["permissions"]), REQUIRED_PERMISSIONS)
  _eq(errors, "binding_manifest.view.layouts", tuple(bm["view"]["layouts"]), ("contiguous",))
  _eq(errors, "binding_manifest.initial.methods", tuple(bm["initial"]["methods"]), REQUIRED_INITIAL)
  _eq(
    errors,
    "binding_manifest.required_kernel_families",
    tuple(bm["required_kernel_families"]),
    REQUIRED_FAMILIES,
  )
  rule = bm["tensor_dtype_rule"]
  if (rule.get("tensor_in"), rule.get("tensor_out"), rule.get("accumulator")) != ("bf16", "bf16", "f32"):
    errors.append("binding_manifest.tensor_dtype_rule: BF16 in, BF16 out, FP32 accumulator")
  if rule.get("int8_new_on_tensor_path") != "forbidden" or rule.get("external_f32_output") != "forbidden":
    errors.append("binding_manifest.tensor_dtype_rule: INT8 and external FP32 output must stay forbidden")
  if bm["scratch"].get("bound_source_values") != ["static_analysis"]:
    errors.append("binding_manifest.scratch.bound_source_values: only static_analysis is a bound")
  if (
    bm["events"].get("required") != ["completion"] or bm["events"].get("unique_within_context") is not True
  ):
    errors.append("binding_manifest.events: completion is required and names are unique within a Context")
  kernels = {k["id"]: k for k in bm["supported_kernels"]}
  for kid, want in REQUIRED_KERNELS.items():
    k = kernels.get(kid)
    if k is None:
      errors.append(f"binding_manifest.supported_kernels: required kernel {kid} is missing")
      continue
    for key, val in want.items():
      _eq(errors, f"binding_manifest.supported_kernels.{kid}.{key}", k.get(key), val)
  for kid in sorted(set(kernels) - set(REQUIRED_KERNELS)):
    errors.append(
      f"binding_manifest.supported_kernels.{kid}: not in the validator's required set; add it deliberately"
    )

  hp = contract["helpers"]
  _eq(errors, "helpers.entry_required_fields", tuple(hp["entry_required_fields"]), REQUIRED_HELPER_FIELDS)
  _eq(errors, "helpers.barrier_kinds", tuple(hp["barrier_kinds"]), REQUIRED_BARRIERS)
  if hp.get("hand_written_numeric_address") != "forbidden":
    errors.append("helpers.hand_written_numeric_address: must stay forbidden")
  got = {h["name"]: (h["csr"], h["kind"]) for h in hp["registered"]}
  if got != REQUIRED_HELPERS:
    errors.append(f"helpers.registered: must be exactly {REQUIRED_HELPERS} (got {got})")

  pm = contract["patch_mask"]
  _eq(errors, "patch_mask.layers", tuple(pm["layers"]), REQUIRED_PATCH_LAYERS)
  if pm.get("undeclared_field_patch") != "reject":
    errors.append("patch_mask.undeclared_field_patch: must be 'reject'")

  b = contract["build"]
  _eq(errors, "build.states", tuple(b["states"]), REQUIRED_BUILD_STATES)
  if (
    b.get("failure_returns_to") != "AUTHORED"
    or b.get("loader_patches") != "forbidden"
    or b.get("immutable_after") != "SEALED"
  ):
    errors.append(
      "build: failure must return to AUTHORED, the Loader must not patch, and SEALED is immutable"
    )
  if (
    b["bounds"].get("measured_is_evidence_only") is not True
    or b["bounds"].get("bound_requires") != "static_analysis"
  ):
    errors.append("build.bounds: a measured value is evidence only; a bound needs static analysis")

  tb = contract["trust_boundary"]
  for key in TRUST_FALSE:
    if tb.get(key) is not False:
      errors.append(f"trust_boundary.{key}: must be false (the Loader does not prove machine code)")
  if tb.get("delete_required_step_with_recomputed_hash") != "reject":
    errors.append("trust_boundary.delete_required_step_with_recomputed_hash: must be 'reject'")
  if sorted(tb.get("machine_code_trusted_via", [])) != sorted(
    ["controlled_kernel", "trusted_build_chain", "numeric_and_protocol_verification"]
  ):
    errors.append("trust_boundary.machine_code_trusted_via: the trust sources must be listed explicitly")
  if contract.get("status") != "draft":
    errors.append("status: must stay draft (C020 seals, not this card)")


# ---------------------------------------------------------------------------------------------------
SCHEMA_COMMENT = (
  "DERIVED from protocol/w5_package.contract.yaml (the authority). It is a structural aid for tools that "
  "have a schema library; the strict consumer in scripts/lib/package_contract.py is what the tests run, "
  "because this environment has no jsonschema package. Cross-field rules (alias target, unique events, "
  "mode table, chain order) are NOT expressible here and live in the consumer."
)


def derive_schema(contract: dict[str, Any]) -> dict[str, Any]:
  """Recompute schema.json from the contract (the same derivation that wrote the committed file)."""
  b = contract["binding_manifest"]
  v = b["view"]
  view = {
    "type": "object",
    "additionalProperties": False,
    "required": b["view_required_fields"],
    "properties": {
      "handle": {"type": "string", "minLength": 1},
      "shape": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 1}},
      "dtype": {"enum": v["dtypes"]},
      "layout": {"enum": v["layouts"]},
      "permission": {"enum": v["permissions"]},
      "alias": {"type": ["string", "null"]},
      "actual_view": {"type": "string", "minLength": 1},
    },
  }
  entry = {
    "type": "object",
    "additionalProperties": False,
    "required": b["entry_required_fields"],
    "properties": {
      "kernel_id": {"type": "string"},
      "kernel_version": {"type": "integer"},
      "source_node": {"type": "string"},
      "ins": {"type": "array", "minItems": 1, "items": view},
      "outs": {"type": "array", "minItems": 1, "items": view},
      "initial": {
        "type": "object",
        "required": ["method"],
        "additionalProperties": False,
        "properties": {"method": {"enum": b["initial"]["methods"]}, "note": {"type": "string"}},
      },
      "scratch": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": False,
          "required": b["scratch"]["required_fields"],
          "properties": {
            "name": {"type": "string"},
            "bytes": {"type": "integer", "minimum": 1},
            "bound_source": {"enum": b["scratch"]["bound_source_values"]},
          },
        },
      },
      "clobber": {"type": "array", "items": {"type": "string", "minLength": 1}},
      "events": {
        "type": "object",
        "required": b["events"]["required"],
        "additionalProperties": False,
        "properties": {
          k: {"type": "string", "minLength": 1} for k in b["events"]["required"] + b["events"]["optional"]
        },
      },
      "numeric_profile": {"const": contract["package"]["numeric_profile"]["must_equal_id"]},
      "mode_table": {"type": "string"},
      "region": {
        "type": "object",
        "required": ["start", "end", "chains"],
        "additionalProperties": False,
        "properties": {"start": {"const": True}, "end": {"const": True}, "chains": {"type": "array"}},
      },
      "patch_mask": {"type": "array"},
      "helpers": {"type": "array"},
    },
  }
  return {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "cn_package/v1 binding manifest",
    "$comment": SCHEMA_COMMENT,
    "type": "object",
    "additionalProperties": False,
    "required": ["kernels"],
    "properties": {"kernels": {"type": "array", "minItems": 1, "items": entry}},
  }


def check_schema_matches(contract: dict[str, Any], errors: list[str], path: Path = SCHEMA) -> None:
  if not path.is_file():
    errors.append(f"schema: {path} is missing")
    return
  committed = json.loads(path.read_text())
  if committed != derive_schema(contract):
    errors.append(
      "schema: kernels/bindings/schema.json differs from what the contract derives; regenerate it"
    )


def check_example(contract: dict[str, Any], errors: list[str], path: Path = EXAMPLE) -> None:
  if not path.is_file():
    errors.append(f"example: {path} is missing")
    return
  try:
    package = pc.load(path.read_text(), contract)
  except pc.PackageError as exc:
    errors.append(f"example: the committed legal example is rejected: {exc}")
    return
  sealed = pc.seal({k: v for k, v in package.items() if k != "package_hash"}, contract)
  if sealed["package_hash"] != package["package_hash"]:
    errors.append("example: re-sealing the committed example gives a different package_hash")


def check_field_request(request: dict[str, Any], contract: dict[str, Any], errors: list[str]) -> None:
  if request.get("schema") != "cn_field_request/v1" or request.get("package") != "W5":
    errors.append("field_request: schema/package must be cn_field_request/v1 + W5")
  catalog = {m["name"] for m in yaml.safe_load(CATALOG.read_text())["messages"]}
  names = {d["name"] for d in yaml.safe_load(pc.CSR_NAMES.read_text())["declared"]}
  for item in request.get("messages") or []:
    known = item.get("message") in catalog
    if known and item.get("new") is True:
      errors.append(f"field_request.messages.{item.get('message')}: already in the C009 catalog, not new")
    if not known and item.get("new") is not True:
      errors.append(
        f"field_request.messages.{item.get('message')}: not in the C009 catalog; mark new: true"
      )
  for item in request.get("helpers") or []:
    if item.get("csr") not in names:
      errors.append(
        f"field_request.helpers.{item.get('name')}: csr {item.get('csr')!r} is not in csr_name_map.yaml"
      )
    for banned in ("address", "bits", "opcode", "width_bits", "encoding"):
      if banned in item:
        errors.append(f"field_request.helpers.{item.get('name')}.{banned}: invented value (INVENTED_VALUE)")
  asked = {h.get("name") for h in request.get("helpers") or []}
  if asked != set(REQUIRED_HELPERS):
    errors.append(f"field_request.helpers: must be exactly {sorted(REQUIRED_HELPERS)}")
  fields = request.get("package_fields") or []
  if tuple(f.get("name") for f in fields) != REQUIRED_OUTER_FIELDS:
    errors.append(f"field_request.package_fields: must list exactly {list(REQUIRED_OUTER_FIELDS)}")
  contract_fields = set(contract["package"]["required_fields"])
  for f in fields:
    if f.get("name") not in contract_fields:
      errors.append(f"field_request.package_fields.{f.get('name')}: not a field of the contract")


# ---------------------------------------------------------------------------------------------------
def validate(contract_path: Path = pc.CONTRACT) -> dict[str, Any]:
  contract = pc.load_contract(contract_path)
  request = yaml.safe_load(REQUEST.read_text())
  return validate_docs(contract, request)


def validate_docs(contract: Any, request: Any) -> dict[str, Any]:
  errors: list[str] = []
  for name, doc in (("contract", contract), ("field request", request)):
    if not isinstance(doc, dict):
      errors.append(f"{name}: root must be a mapping, got {type(doc).__name__}")
  if errors:
    return {
      "schema": "cn_package_contract_report/v1",
      "valid": False,
      "errors": errors,
      "anchors_read_back": 0,
    }
  anchors = 0

  def guarded(label: str, fn: Any, *args: Any) -> Any:
    try:
      return fn(*args)
    except (KeyError, TypeError, AttributeError, IndexError, ValueError, pc.PackageError) as exc:
      errors.append(f"{label}: document is malformed ({type(exc).__name__}: {exc})")
      return None

  def sources() -> int:
    local: list[str] = []
    n = check_sources(contract, local)
    errors.extend(local)
    return n

  anchors = guarded("sources", sources) or 0
  guarded("required", check_required, contract, errors)
  guarded("schema", check_schema_matches, contract, errors)
  guarded("example", check_example, contract, errors)
  guarded("field request", check_field_request, request, contract, errors)
  return {
    "schema": "cn_package_contract_report/v1",
    "valid": not errors,
    "errors": errors,
    "anchors_read_back": anchors,
    "open": [
      "slot_count",
      "slot_capacity",
      "kernel families beyond K-VME-BF16-ACC",
      "package runtime (C037)",
    ],
  }


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  report = validate()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / "package_contract_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  if not report["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
