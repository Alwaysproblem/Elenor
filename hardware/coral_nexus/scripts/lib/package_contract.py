"""Reference consumer for the cn_package/v1 contract (C014).

This is a CONTRACT CONSUMER, not the package runtime: encode/decode of real packages and the Loader are
C037. It exists so the contract can be tested with real structured samples: a complete binding is
accepted and every kind of damage is rejected with a code that names the field.

Everything it enforces is read from `protocol/w5_package.contract.yaml`: the required fields, the dtype /
layout / permission sets, the helper table, the version domains. A test that deletes a required field
from the YAML therefore changes what this module accepts.

Strictness (CMP-14), implemented here and not delegated to a schema library:
  - duplicate JSON keys are rejected at parse time
  - NaN / Infinity are rejected at parse time
  - unknown fields are rejected at every level that declares its fields
  - types are exact (a bool is not an int, a float is not an int)
  - the package hash covers the declared parts and recomputing it after deleting a required contract
    step still fails, because the semantic checks run independently of the hash
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
H = LIB.parents[1]
CONTRACT = H / "protocol" / "w5_package.contract.yaml"
CSR_NAMES = H / "protocol" / "csr_name_map.yaml"
ABI = H / "protocol" / "w3_vme_decl_abi_v1.json"
TABLES = H / "protocol" / "w3_vme_mode_tables_v1.json"
PROFILE = H / "protocol" / "w3_numeric_profile_v1.yaml"

PACKAGE_SCHEMA = "cn_package/v1"


class PackageError(Exception):
  """Coded rejection. `code` is stable; `field` names the offending field path."""

  def __init__(self, code: str, field: str, detail: str = "") -> None:
    super().__init__(f"{code} at {field}" + (f": {detail}" if detail else ""))
    self.code = code
    self.field = field


def load_contract(path: Path = CONTRACT) -> dict[str, Any]:
  return yaml.safe_load(path.read_text())  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# strict JSON
# ---------------------------------------------------------------------------
def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  out: dict[str, Any] = {}
  for key, value in pairs:
    if key in out:
      raise PackageError("DUPLICATE_KEY", key)
    out[key] = value
  return out


def _no_constant(value: str) -> Any:
  raise PackageError("NON_FINITE_NUMBER", value)


def parse_strict(text: str) -> Any:
  """json.loads with duplicate-key and NaN/Infinity rejection (the same hooks the control plane uses)."""
  try:
    return json.loads(text, object_pairs_hook=_unique, parse_constant=_no_constant)
  except json.JSONDecodeError as exc:
    raise PackageError("INVALID_JSON", "$", exc.msg) from exc


def _finite(node: Any, path: str) -> None:
  if isinstance(node, float) and not math.isfinite(node):
    raise PackageError("NON_FINITE_NUMBER", path)
  if isinstance(node, dict):
    for k, v in node.items():
      _finite(v, f"{path}.{k}")
  elif isinstance(node, list):
    for i, v in enumerate(node):
      _finite(v, f"{path}[{i}]")


def canonical(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _is_int(value: Any) -> bool:
  return type(value) is int


def _need_fields(obj: Any, required: list[str], path: str, *, allowed: list[str] | None = None) -> None:
  if not isinstance(obj, dict):
    raise PackageError("WRONG_TYPE", path, f"expected an object, got {type(obj).__name__}")
  for name in required:
    if name not in obj:
      raise PackageError("MISSING_FIELD", f"{path}.{name}")
  allowed_set = set(allowed if allowed is not None else required)
  for name in obj:
    if name not in allowed_set:
      raise PackageError("UNKNOWN_FIELD", f"{path}.{name}")


# ---------------------------------------------------------------------------
# package hash
# ---------------------------------------------------------------------------
def package_hash(package: dict[str, Any], contract: dict[str, Any]) -> str:
  """sha256 over exactly the parts the contract says it covers (never over itself)."""
  covers = contract["package"]["package_hash"]["covers"]
  parts = {
    "image_sha256": (package.get("image") or {}).get("bytes_sha256"),
    "binding_manifest": package.get("binding_manifest"),
    "resource_manifest": package.get("resource_manifest"),
    "target_fingerprint": package.get("target_fingerprint"),
    "patch_mask": package.get("patch_mask"),
    "numeric_profile_id": package.get("numeric_profile_id"),
    "control_plane_artifact_hash": (package.get("control_plane") or {}).get("artifact_hash"),
  }
  missing = [name for name in covers if name not in parts]
  if missing:
    raise PackageError(
      "HASH_COVERAGE_UNKNOWN", "package_hash", f"contract names {missing} that this consumer cannot hash"
    )
  return hashlib.sha256(canonical({name: parts[name] for name in covers})).hexdigest()


def seal(package: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
  """Return a copy with package_hash filled in (validation first: a broken package is never sealed)."""
  body = {k: v for k, v in package.items() if k != "package_hash"}
  validate_package({**body, "package_hash": "0" * 64}, contract, check_hash=False)
  sealed = dict(body)
  sealed["package_hash"] = package_hash(sealed, contract)
  return sealed


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _check_view(view: Any, path: str, spec: dict[str, Any], contract_view: dict[str, Any]) -> None:
  _need_fields(view, spec["view_required_fields"], path)
  if not isinstance(view["handle"], str) or not view["handle"].strip():
    raise PackageError("WRONG_TYPE", f"{path}.handle")
  shape = view["shape"]
  if not isinstance(shape, list) or not shape or not all(_is_int(d) and d > 0 for d in shape):
    raise PackageError("BAD_SHAPE", f"{path}.shape", "a shape is a non-empty list of positive integers")
  if view["dtype"] not in contract_view["dtypes"]:
    raise PackageError("BAD_DTYPE", f"{path}.dtype", f"{view['dtype']!r} not in {contract_view['dtypes']}")
  if view["layout"] not in contract_view["layouts"]:
    raise PackageError("BAD_LAYOUT", f"{path}.layout", f"{view['layout']!r}")
  if view["permission"] not in contract_view["permissions"]:
    raise PackageError("BAD_PERMISSION", f"{path}.permission", f"{view['permission']!r}")
  actual = view["actual_view"]
  if not isinstance(actual, str) or not actual.strip():
    raise PackageError(
      "MISSING_ACTUAL_VIEW", f"{path}.actual_view", "a binding must name the view it really reads or writes"
    )
  alias = view["alias"]
  if alias is not None and (not isinstance(alias, str) or not alias.strip()):
    raise PackageError("WRONG_TYPE", f"{path}.alias")


def validate_package(package: Any, contract: dict[str, Any], *, check_hash: bool = True) -> None:
  """Raise PackageError for the first violation; return None when the whole package is acceptable."""
  spec = contract["package"]
  _need_fields(package, spec["required_fields"], "$")
  if package["package_schema"] != contract["version_domains"]["package"]["id"]:
    raise PackageError("WRONG_VERSION", "$.package_schema", f"{package['package_schema']!r}")
  _finite(package, "$")

  # control plane is wrapped unchanged and keeps ITS OWN version domain
  cp = package["control_plane"]
  _need_fields(
    cp,
    ["schema_version", "compiler_abi", "artifact_hash"],
    "$.control_plane",
    allowed=["schema_version", "compiler_abi", "artifact_hash", "source_hash"],
  )
  dom = contract["version_domains"]["control_plane"]
  if cp["schema_version"] != dom["schema_version"] or type(cp["schema_version"]) is not int:
    raise PackageError("WRONG_VERSION", "$.control_plane.schema_version")
  if cp["compiler_abi"] != dom["compiler_abi"]:
    raise PackageError("WRONG_VERSION", "$.control_plane.compiler_abi")

  _check_image(package["image"], contract)
  _check_numeric(package, contract)
  mtables = json.loads(TABLES.read_text())
  abi = json.loads(ABI.read_text())
  names = {d["name"] for d in yaml.safe_load(CSR_NAMES.read_text())["declared"]}
  _check_manifest(package["binding_manifest"], contract, mtables, abi, names)
  _check_resources(package, contract)
  _check_patch_mask(package["patch_mask"], contract, package["binding_manifest"])

  if check_hash:
    if package["package_hash"] != package_hash(package, contract):
      raise PackageError("HASH_MISMATCH", "$.package_hash")


def _check_image(image: Any, contract: dict[str, Any]) -> None:
  spec = contract["image"]
  _need_fields(image, spec["required_fields"], "$.image")
  if not isinstance(image["bytes_sha256"], str) or len(image["bytes_sha256"]) != 64:
    raise PackageError("BAD_HASH", "$.image.bytes_sha256")
  if not _is_int(image["text_bytes"]) or image["text_bytes"] <= 0:
    raise PackageError("BAD_TEXT_BYTES", "$.image.text_bytes")
  if not _is_int(image["entry"]) or image["entry"] < 0:
    raise PackageError("WRONG_TYPE", "$.image.entry")
  prov = image["provenance"]
  _need_fields(prov, spec["provenance"]["required_fields"], "$.image.provenance")
  for key in spec["provenance"]["required_fields"]:
    if not isinstance(prov[key], str) or not prov[key].strip():
      raise PackageError(
        "NO_CONTROLLED_SOURCE",
        f"$.image.provenance.{key}",
        "an image with no controlled source is not an identity",
      )
  layout = image["link_layout"]
  _need_fields(layout, spec["link_layout"]["required_fields"], "$.image.link_layout")
  for key in spec["link_layout"]["required_fields"]:
    if not _is_int(layout[key]) or layout[key] < 0:
      raise PackageError("WRONG_TYPE", f"$.image.link_layout.{key}")
  if layout["text_bytes"] != image["text_bytes"]:
    raise PackageError(
      "TEXT_BYTES_DISAGREE",
      "$.image.link_layout.text_bytes",
      "text_bytes must equal the linked length everywhere it appears",
    )
  if layout["text_base"] <= layout["data_base"] < layout["text_base"] + layout["text_bytes"] or (
    layout["data_base"] <= layout["text_base"] < layout["data_base"] + layout["data_bytes"]
  ):
    raise PackageError("SEGMENT_OVERLAP", "$.image.link_layout", "text and data segments overlap")
  if not layout["text_base"] <= image["entry"] < layout["text_base"] + layout["text_bytes"]:
    raise PackageError("ENTRY_OUTSIDE_TEXT", "$.image.entry")
  itcm = spec["capacities"]["itcm_bytes"]["value"]
  if image["text_bytes"] > itcm:
    raise PackageError(
      "TEXT_EXCEEDS_CAPACITY",
      "$.image.text_bytes",
      f"{image['text_bytes']} > {itcm} B ITCM default; a different target needs a new target hash",
    )


def _check_numeric(package: dict[str, Any], contract: dict[str, Any]) -> None:
  want = contract["package"]["numeric_profile"]["must_equal_id"]
  if package["numeric_profile_id"] != want:
    raise PackageError(
      "NUMERIC_PROFILE_MISMATCH", "$.numeric_profile_id", f"{package['numeric_profile_id']!r} != {want!r}"
    )
  profile = yaml.safe_load(PROFILE.read_text())
  if profile["numeric_profile_id"] != want:
    raise PackageError(
      "NUMERIC_PROFILE_MISMATCH",
      "protocol/w3_numeric_profile_v1.yaml",
      "the profile file no longer carries the id this contract names",
    )
  if profile["support_status"] not in contract["package"]["numeric_profile"]["status_allowed"]:
    raise PackageError(
      "NUMERIC_PROFILE_STATUS", "protocol/w3_numeric_profile_v1.yaml", profile["support_status"]
    )


def _check_manifest(
  manifest: Any, contract: dict[str, Any], mtables: dict[str, Any], abi: dict[str, Any], csr_names: set[str]
) -> None:
  spec = contract["binding_manifest"]
  _need_fields(manifest, ["kernels"], "$.binding_manifest")
  kernels = manifest["kernels"]
  if not isinstance(kernels, list) or not kernels:
    raise PackageError("EMPTY_MANIFEST", "$.binding_manifest.kernels")
  registered = {k["id"]: k for k in spec["supported_kernels"]}
  seen_events: set[str] = set()
  for index, entry in enumerate(kernels):
    path = f"$.binding_manifest.kernels[{index}]"
    _need_fields(entry, spec["entry_required_fields"], path)
    kid = entry["kernel_id"]
    if kid not in registered:
      raise PackageError("UNREGISTERED_KERNEL", f"{path}.kernel_id", f"{kid!r}")
    if entry["kernel_version"] != registered[kid]["version"] or not _is_int(entry["kernel_version"]):
      raise PackageError("KERNEL_VERSION_MISMATCH", f"{path}.kernel_version")
    for side in ("ins", "outs"):
      views = entry[side]
      if not isinstance(views, list) or not views:
        raise PackageError("MISSING_VIEWS", f"{path}.{side}", "a kernel needs explicit ins and outs")
      for j, view in enumerate(views):
        _check_view(view, f"{path}.{side}[{j}]", spec, spec["view"])
    _check_alias(entry, path)
    _check_dtype_rule(entry, path, spec)
    _check_initial(entry["initial"], path, spec)
    _check_scratch(entry["scratch"], path, spec)
    _check_events(entry["events"], path, spec, seen_events)
    if not isinstance(entry["clobber"], list) or not all(
      isinstance(c, str) and c for c in entry["clobber"]
    ):
      raise PackageError("WRONG_TYPE", f"{path}.clobber")
    if entry["numeric_profile"] != contract["package"]["numeric_profile"]["must_equal_id"]:
      raise PackageError("NUMERIC_PROFILE_MISMATCH", f"{path}.numeric_profile")
    _check_mode_table(entry, path, registered[kid], mtables)
    _check_region(entry["region"], path, abi)
    _check_helpers(entry["helpers"], path, contract, csr_names)


def _check_alias(entry: dict[str, Any], path: str) -> None:
  ins = {v["handle"]: v for v in entry["ins"]}
  for j, out in enumerate(entry["outs"]):
    if out["alias"] is None:
      continue
    target = ins.get(out["alias"])
    if target is None:
      raise PackageError(
        "ALIAS_UNKNOWN", f"{path}.outs[{j}].alias", f"{out['alias']!r} is not an input handle"
      )
    if target["permission"] == "r":
      raise PackageError(
        "ILLEGAL_ALIAS", f"{path}.outs[{j}].alias", "an output may not alias a read-only input"
      )
  handles = [v["handle"] for v in entry["ins"]] + [v["handle"] for v in entry["outs"]]
  if len(handles) != len(set(handles)):
    raise PackageError("DUPLICATE_HANDLE", path, "view handles must be unique within a kernel")


def _check_dtype_rule(entry: dict[str, Any], path: str, spec: dict[str, Any]) -> None:
  rule = spec["tensor_dtype_rule"]
  for j, out in enumerate(entry["outs"]):
    if out["dtype"] != rule["tensor_out"] and out["dtype"] != "i32":
      raise PackageError(
        "EXTERNAL_DTYPE_FORBIDDEN",
        f"{path}.outs[{j}].dtype",
        f"{out['dtype']!r}: the tensor output is {rule['tensor_out']}",
      )
  for j, view in enumerate(entry["ins"]):
    if view["dtype"] not in (rule["tensor_in"], "i32", "f32"):
      raise PackageError("BAD_DTYPE", f"{path}.ins[{j}].dtype")


def _check_initial(initial: Any, path: str, spec: dict[str, Any]) -> None:
  _need_fields(initial, ["method"], f"{path}.initial", allowed=["method", "note"])
  if initial["method"] not in spec["initial"]["methods"]:
    raise PackageError(
      "BAD_INITIAL",
      f"{path}.initial.method",
      f"{initial['method']!r}: a seed is one of {spec['initial']['methods']}",
    )


def _check_scratch(scratch: Any, path: str, spec: dict[str, Any]) -> None:
  if not isinstance(scratch, list):
    raise PackageError("WRONG_TYPE", f"{path}.scratch")
  for j, item in enumerate(scratch):
    _need_fields(item, spec["scratch"]["required_fields"], f"{path}.scratch[{j}]")
    if not _is_int(item["bytes"]) or item["bytes"] <= 0:
      raise PackageError("BAD_SCRATCH", f"{path}.scratch[{j}].bytes")
    if item["bound_source"] not in spec["scratch"]["bound_source_values"]:
      raise PackageError(
        "BOUND_NOT_STATIC",
        f"{path}.scratch[{j}].bound_source",
        f"{item['bound_source']!r}: a measured peak is evidence, not a bound",
      )


def _check_events(events: Any, path: str, spec: dict[str, Any], seen: set[str]) -> None:
  _need_fields(
    events,
    spec["events"]["required"],
    f"{path}.events",
    allowed=spec["events"]["required"] + spec["events"]["optional"],
  )
  names: list[str] = []
  for key, value in events.items():
    if not isinstance(value, str) or not value.strip():
      raise PackageError("WRONG_TYPE", f"{path}.events.{key}")
    names.append(value)
  if len(names) != len(set(names)):
    raise PackageError("DUPLICATE_EVENT", f"{path}.events", "event names are unique within a Context")
  for name in names:
    if name in seen:
      raise PackageError("DUPLICATE_EVENT", f"{path}.events", f"{name!r} is already used by another kernel")
    seen.add(name)


def _check_mode_table(
  entry: dict[str, Any], path: str, reg: dict[str, Any], mtables: dict[str, Any]
) -> None:
  mode = entry["mode_table"]
  if mode != reg["mode_table"]:
    raise PackageError(
      "MODE_TABLE_MISMATCH", f"{path}.mode_table", f"{mode!r} is not the table sealed for {reg['id']}"
    )
  if mode not in mtables:
    raise PackageError("MODE_TABLE_UNKNOWN", f"{path}.mode_table", f"{mode!r}")


def _check_region(region: Any, path: str, abi: dict[str, Any]) -> None:
  _need_fields(region, ["start", "end", "chains"], f"{path}.region")
  if region["start"] is not True or region["end"] is not True:
    raise PackageError("REGION_BOUNDS_MISSING", f"{path}.region", "START and END must both be declared")
  chains = region["chains"]
  if not isinstance(chains, list):
    raise PackageError("WRONG_TYPE", f"{path}.region.chains")
  kinds = set(abi["declaration"]["kinds"])
  for j, chain in enumerate(chains):
    _need_fields(chain, ["id", "steps"], f"{path}.region.chains[{j}]")
    steps = chain["steps"]
    if not isinstance(steps, list) or any(s not in kinds for s in steps):
      raise PackageError(
        "BAD_CHAIN_KIND", f"{path}.region.chains[{j}].steps", "every step must be an ABI declaration kind"
      )
    if not steps or steps[0] != "CHAIN_FIRST":
      raise PackageError("CHAIN_WITHOUT_FIRST", f"{path}.region.chains[{j}]")
    if steps.count("CHAIN_FIRST") != 1:
      raise PackageError("DUPLICATE_FIRST", f"{path}.region.chains[{j}]")
    if steps[-1] != "CHAIN_FINAL" or steps.count("CHAIN_FINAL") != 1:
      raise PackageError("CHAIN_WITHOUT_FINAL", f"{path}.region.chains[{j}]")


def _check_helpers(helpers: Any, path: str, contract: dict[str, Any], csr_names: set[str]) -> None:
  spec = contract["helpers"]
  if not isinstance(helpers, list):
    raise PackageError("WRONG_TYPE", f"{path}.helpers")
  registered = {h["name"]: h for h in spec["registered"]}
  for j, helper in enumerate(helpers):
    hp = f"{path}.helpers[{j}]"
    _need_fields(helper, spec["entry_required_fields"], hp)
    reg = registered.get(helper["name"])
    if reg is None:
      raise PackageError("UNDECLARED_HELPER", f"{hp}.name", f"{helper['name']!r} cannot be called")
    if helper["csr"] != reg["csr"] or helper["csr"] not in csr_names:
      raise PackageError(
        "HELPER_CSR_MISMATCH", f"{hp}.csr", "a CSR is referenced by its generated logical name only"
      )
    if (
      not isinstance(helper["csr"], str)
      or helper["csr"].startswith(("0x", "0X"))
      or helper["csr"].isdigit()
    ):
      raise PackageError("HAND_WRITTEN_ADDRESS", f"{hp}.csr")
    if helper["barrier"] not in spec["barrier_kinds"]:
      raise PackageError(
        "BAD_BARRIER",
        f"{hp}.barrier",
        f"{helper['barrier']!r}: a helper needs a compiler-reordering barrier",
      )
    if not isinstance(helper["observable"], str) or not helper["observable"].strip():
      raise PackageError(
        "MISSING_OBSERVABLE", f"{hp}.observable", "a helper must state what it makes observable"
      )


def _check_resources(package: dict[str, Any], contract: dict[str, Any]) -> None:
  rm = package["resource_manifest"]
  _need_fields(
    rm,
    ["text_bytes", "static_data_bytes", "stack_bytes", "scratch_bytes", "bound_source"],
    "$.resource_manifest",
  )
  for key in ("text_bytes", "static_data_bytes", "stack_bytes", "scratch_bytes"):
    if not _is_int(rm[key]) or rm[key] < 0:
      raise PackageError("WRONG_TYPE", f"$.resource_manifest.{key}")
  if rm["bound_source"] != "static_analysis":
    raise PackageError(
      "BOUND_NOT_STATIC", "$.resource_manifest.bound_source", "only bounded static analysis is a bound"
    )
  if rm["text_bytes"] != package["image"]["text_bytes"]:
    raise PackageError(
      "TEXT_BYTES_DISAGREE", "$.resource_manifest.text_bytes", "must equal the real image length"
    )
  caps = contract["image"]["capacities"]
  if rm["stack_bytes"] > caps["dtcm_bytes"]["value"]:
    raise PackageError("STACK_EXCEEDS_CAPACITY", "$.resource_manifest.stack_bytes")
  tf = package["target_fingerprint"]
  _need_fields(
    tf,
    ["tensor_input_bytes", "accumulator_bytes", "tensor_output_bytes", "rvv_vlen", "context_count"],
    "$.target_fingerprint",
  )
  if (tf["tensor_input_bytes"], tf["accumulator_bytes"], tf["tensor_output_bytes"]) != (2, 4, 2):
    raise PackageError(
      "TARGET_DTYPE_MISMATCH", "$.target_fingerprint", "BF16 in, FP32 accumulator, BF16 out"
    )
  for key in ("rvv_vlen", "context_count"):
    if not _is_int(tf[key]) or tf[key] <= 0:
      raise PackageError("WRONG_TYPE", f"$.target_fingerprint.{key}")


def _check_patch_mask(mask: Any, contract: dict[str, Any], manifest: dict[str, Any]) -> None:
  spec = contract["patch_mask"]
  if not isinstance(mask, list):
    raise PackageError("WRONG_TYPE", "$.patch_mask")
  fields: set[str] = set()
  for j, item in enumerate(mask):
    _need_fields(item, spec["required_fields"], f"$.patch_mask[{j}]")
    if item["layer"] not in spec["layers"]:
      raise PackageError("BAD_PATCH_LAYER", f"$.patch_mask[{j}].layer", f"{item['layer']!r}")
    if item["field"] in fields:
      raise PackageError("DUPLICATE_PATCH_FIELD", f"$.patch_mask[{j}].field")
    fields.add(item["field"])


def patch_allowed(package: dict[str, Any], field: str, layer: str) -> None:
  """A field absent from patch_mask is never modified; the declared layer is the only one allowed."""
  for item in package["patch_mask"]:
    if item["field"] == field:
      if item["layer"] != layer:
        raise PackageError(
          "PATCH_WRONG_LAYER", field, f"declared for {item['layer']!r}, attempted by {layer!r}"
        )
      return
  raise PackageError("PATCH_UNDECLARED", field, "a field not in patch_mask is never modified at run time")


def load(text: str, contract: dict[str, Any]) -> dict[str, Any]:
  """Read-only load: strict parse, full validation, hash check. The Loader repairs nothing."""
  package = parse_strict(text)
  validate_package(package, contract)
  return package  # type: ignore[no-any-return]


def residency_reuse_allowed(references: int, invalidation_acked: bool) -> None:
  """A slot is reusable only at zero references AND after the invalidation ACK."""
  if references != 0:
    raise PackageError("SLOT_IN_USE", "slot", f"{references} references")
  if not invalidation_acked:
    raise PackageError("INVALIDATION_NOT_ACKED", "slot")


def layouts_coexist(a: dict[str, Any], b: dict[str, Any]) -> None:
  """Two fixed pre-linked images may coexist only if their address ranges are disjoint."""
  la, lb = a["image"]["link_layout"], b["image"]["link_layout"]
  ra = [(la["text_base"], la["text_bytes"]), (la["data_base"], la["data_bytes"])]
  rb = [(lb["text_base"], lb["text_bytes"]), (lb["data_base"], lb["data_bytes"])]
  for sa, na in ra:
    for sb, nb in rb:
      if sa < sb + nb and sb < sa + na:
        raise PackageError("IMAGES_OVERLAP", "image.link_layout", f"[{sa:#x},+{na}) vs [{sb:#x},+{nb})")
