"""cn_package/v1 strict codec and structural validation (C037 / W5-01).

The rules come from ``hardware/coral_nexus/protocol/w5_package.contract.yaml`` (C014, sealed in G1) and
are read from it at import time, so this module holds no second copy of a capacity, a version or a
whitelist. A package is a plain JSON object; nothing here guesses a field: an unknown, missing, duplicated
or non-finite value is rejected and the error names the field path.

Layering: this module checks the package *as data*. Hashing and sealing are ``seal.py``; reading a sealed
file back is ``loader.py``. The C014 reference consumer under ``hardware/coral_nexus/scripts/lib`` is NOT
imported; the contract tests use it as an independent cross-check of this implementation.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / "hardware/coral_nexus/protocol"
CONTRACT_PATH = PROTOCOL / "w5_package.contract.yaml"
CSR_NAMES_PATH = PROTOCOL / "csr_name_map.yaml"
ABI_PATH = PROTOCOL / "w3_vme_decl_abi_v1.json"
TABLES_PATH = PROTOCOL / "w3_vme_mode_tables_v1.json"
PROFILE_PATH = PROTOCOL / "w3_numeric_profile_v1.yaml"

PACKAGE_SCHEMA = "cn_package/v1"


class PackageError(Exception):
  """A coded rejection. ``code`` is stable and ``field`` is the JSON path of the offending value."""

  def __init__(self, code: str, field: str, detail: str = "") -> None:
    super().__init__(f"{code} at {field}" + (f": {detail}" if detail else ""))
    self.code = code
    self.field = field
    self.detail = detail


@lru_cache(maxsize=1)
def contract() -> dict[str, Any]:
  loaded = yaml.safe_load(CONTRACT_PATH.read_text())
  if not isinstance(loaded, dict) or loaded.get("version") != 1:
    raise PackageError("BAD_CONTRACT", str(CONTRACT_PATH), "w5_package.contract.yaml is unreadable")
  return loaded


@lru_cache(maxsize=1)
def _references() -> tuple[dict[str, Any], dict[str, Any], set[str], dict[str, Any]]:
  abi = json.loads(ABI_PATH.read_text())
  tables = json.loads(TABLES_PATH.read_text())
  names = {d["name"] for d in yaml.safe_load(CSR_NAMES_PATH.read_text())["declared"]}
  profile = yaml.safe_load(PROFILE_PATH.read_text())
  return abi, tables, names, profile


# ---------------------------------------------------------------------------------------------------
# strict JSON
# ---------------------------------------------------------------------------------------------------
def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  out: dict[str, Any] = {}
  for key, value in pairs:
    if key in out:
      raise PackageError("DUPLICATE_KEY", f"$.{key}", "a JSON object may not repeat a key")
    out[key] = value
  return out


def _reject_constant(token: str) -> Any:
  raise PackageError("NON_FINITE_NUMBER", "$", f"{token} is not a finite number")


def parse_strict(text: str) -> Any:
  """``json.loads`` with duplicate-key and NaN/Infinity rejection (the control plane's own hooks)."""
  if type(text) is not str:
    raise PackageError("WRONG_TYPE", "$", "a package is JSON text")
  try:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
  except json.JSONDecodeError as exc:
    raise PackageError("INVALID_JSON", "$", exc.msg) from exc


def canonical(value: Any) -> bytes:
  """The one byte form the hash is computed over (matches the control plane's canonical JSON)."""
  try:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  except ValueError as exc:
    raise PackageError("NON_FINITE_NUMBER", "$", str(exc)) from exc


def encode(package: dict[str, Any]) -> str:
  """Deterministic text form of a package (stable key order, one trailing newline)."""
  try:
    return json.dumps(package, sort_keys=True, indent=2, allow_nan=False) + "\n"
  except ValueError as exc:
    raise PackageError("NON_FINITE_NUMBER", "$", str(exc)) from exc


def _is_int(value: Any) -> bool:
  return type(value) is int


def _finite(node: Any, path: str) -> None:
  if isinstance(node, float) and not math.isfinite(node):
    raise PackageError("NON_FINITE_NUMBER", path)
  if isinstance(node, dict):
    for key, value in node.items():
      _finite(value, f"{path}.{key}")
  elif isinstance(node, list):
    for index, value in enumerate(node):
      _finite(value, f"{path}[{index}]")


def _fields(obj: Any, required: list[str], path: str, *, allowed: list[str] | None = None) -> None:
  if not isinstance(obj, dict):
    raise PackageError("WRONG_TYPE", path, f"expected an object, got {type(obj).__name__}")
  for name in required:
    if name not in obj:
      raise PackageError("MISSING_FIELD", f"{path}.{name}")
  permitted = set(allowed if allowed is not None else required)
  for name in obj:
    if name not in permitted:
      raise PackageError("UNKNOWN_FIELD", f"{path}.{name}")


def _hex64(value: Any, path: str) -> None:
  if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
    raise PackageError("BAD_HASH", path, "expected 64 lowercase hex digits")


# ---------------------------------------------------------------------------------------------------
# structural validation (everything except the hash, which is seal.py's)
# ---------------------------------------------------------------------------------------------------
def validate(package: Any) -> None:
  """Raise ``PackageError`` for the first violation; return None when the structure is acceptable."""
  spec = contract()
  _fields(package, spec["package"]["required_fields"], "$")
  if package["package_schema"] != spec["version_domains"]["package"]["id"]:
    raise PackageError("WRONG_VERSION", "$.package_schema", f"{package['package_schema']!r}")
  _finite(package, "$")
  _control_plane(package["control_plane"], spec)
  _image(package["image"], spec)
  _numeric(package, spec)
  abi, tables, csr_names, _ = _references()
  _manifest(package["binding_manifest"], spec, tables, abi, csr_names)
  _resources(package, spec)
  _patch_mask(package["patch_mask"], spec)
  _hex64(package["package_hash"], "$.package_hash")


def _control_plane(cp: Any, spec: dict[str, Any]) -> None:
  _fields(
    cp,
    ["schema_version", "compiler_abi", "artifact_hash"],
    "$.control_plane",
    allowed=["schema_version", "compiler_abi", "artifact_hash", "source_hash"],
  )
  domain = spec["version_domains"]["control_plane"]
  if not _is_int(cp["schema_version"]) or cp["schema_version"] != domain["schema_version"]:
    raise PackageError("WRONG_VERSION", "$.control_plane.schema_version")
  if cp["compiler_abi"] != domain["compiler_abi"]:
    raise PackageError("WRONG_VERSION", "$.control_plane.compiler_abi")
  _hex64(cp["artifact_hash"], "$.control_plane.artifact_hash")
  if "source_hash" in cp:
    _hex64(cp["source_hash"], "$.control_plane.source_hash")


def _image(image: Any, spec: dict[str, Any]) -> None:
  rules = spec["image"]
  _fields(image, rules["required_fields"], "$.image")
  _hex64(image["bytes_sha256"], "$.image.bytes_sha256")
  if not _is_int(image["text_bytes"]) or image["text_bytes"] <= 0:
    raise PackageError("BAD_TEXT_BYTES", "$.image.text_bytes", "must be the real linked length")
  if not _is_int(image["entry"]) or image["entry"] < 0:
    raise PackageError("WRONG_TYPE", "$.image.entry")
  prov = image["provenance"]
  _fields(prov, rules["provenance"]["required_fields"], "$.image.provenance")
  for key in rules["provenance"]["required_fields"]:
    if not isinstance(prov[key], str) or not prov[key].strip():
      raise PackageError(
        "NO_CONTROLLED_SOURCE", f"$.image.provenance.{key}", "a hash of unknown bytes is not an identity"
      )
  layout = image["link_layout"]
  _fields(layout, rules["link_layout"]["required_fields"], "$.image.link_layout")
  for key in rules["link_layout"]["required_fields"]:
    if not _is_int(layout[key]) or layout[key] < 0:
      raise PackageError("WRONG_TYPE", f"$.image.link_layout.{key}")
  if layout["text_bytes"] != image["text_bytes"]:
    raise PackageError("TEXT_BYTES_DISAGREE", "$.image.link_layout.text_bytes")
  if _overlap(layout["text_base"], layout["text_bytes"], layout["data_base"], layout["data_bytes"]):
    raise PackageError("SEGMENT_OVERLAP", "$.image.link_layout", "text and data segments overlap")
  if not layout["text_base"] <= image["entry"] < layout["text_base"] + layout["text_bytes"]:
    raise PackageError("ENTRY_OUTSIDE_TEXT", "$.image.entry")
  itcm = rules["capacities"]["itcm_bytes"]["value"]
  if image["text_bytes"] > itcm:
    raise PackageError("TEXT_EXCEEDS_CAPACITY", "$.image.text_bytes", f"{image['text_bytes']} > {itcm}")


def _overlap(a_base: int, a_len: int, b_base: int, b_len: int) -> bool:
  return a_base < b_base + b_len and b_base < a_base + a_len


def _numeric(package: dict[str, Any], spec: dict[str, Any]) -> None:
  want = spec["package"]["numeric_profile"]["must_equal_id"]
  if package["numeric_profile_id"] != want:
    raise PackageError(
      "NUMERIC_PROFILE_MISMATCH", "$.numeric_profile_id", f"{package['numeric_profile_id']!r}"
    )
  profile = _references()[3]
  if profile["numeric_profile_id"] != want:
    raise PackageError(
      "NUMERIC_PROFILE_MISMATCH", str(PROFILE_PATH), "the profile no longer carries this id"
    )
  if profile["support_status"] not in spec["package"]["numeric_profile"]["status_allowed"]:
    raise PackageError("NUMERIC_PROFILE_STATUS", str(PROFILE_PATH), str(profile["support_status"]))


def _view(view: Any, path: str, spec: dict[str, Any]) -> None:
  rules = spec["view"]
  _fields(view, spec["view_required_fields"], path)
  if not isinstance(view["handle"], str) or not view["handle"].strip():
    raise PackageError("WRONG_TYPE", f"{path}.handle")
  shape = view["shape"]
  if not isinstance(shape, list) or not shape or not all(_is_int(d) and d > 0 for d in shape):
    raise PackageError("BAD_SHAPE", f"{path}.shape", "a non-empty list of positive integers")
  for key, allowed, code in (
    ("dtype", rules["dtypes"], "BAD_DTYPE"),
    ("layout", rules["layouts"], "BAD_LAYOUT"),
    ("permission", rules["permissions"], "BAD_PERMISSION"),
  ):
    if view[key] not in allowed:
      raise PackageError(code, f"{path}.{key}", f"{view[key]!r} not in {allowed}")
  if not isinstance(view["actual_view"], str) or not view["actual_view"].strip():
    raise PackageError(
      "MISSING_ACTUAL_VIEW", f"{path}.actual_view", "a binding names the view it really touches"
    )
  alias = view["alias"]
  if alias is not None and (not isinstance(alias, str) or not alias.strip()):
    raise PackageError("WRONG_TYPE", f"{path}.alias")


def _manifest(
  manifest: Any, spec_all: dict[str, Any], tables: dict[str, Any], abi: dict[str, Any], csr_names: set[str]
) -> None:
  spec = spec_all["binding_manifest"]
  _fields(manifest, ["kernels"], "$.binding_manifest")
  kernels = manifest["kernels"]
  if not isinstance(kernels, list) or not kernels:
    raise PackageError("EMPTY_MANIFEST", "$.binding_manifest.kernels")
  registered = {k["id"]: k for k in spec["supported_kernels"]}
  seen_events: set[str] = set()
  for index, entry in enumerate(kernels):
    path = f"$.binding_manifest.kernels[{index}]"
    _fields(entry, spec["entry_required_fields"], path)
    reg = registered.get(entry["kernel_id"])
    if reg is None:
      raise PackageError("UNREGISTERED_KERNEL", f"{path}.kernel_id", f"{entry['kernel_id']!r}")
    if not _is_int(entry["kernel_version"]) or entry["kernel_version"] != reg["version"]:
      raise PackageError("KERNEL_VERSION_MISMATCH", f"{path}.kernel_version")
    for side in ("ins", "outs"):
      views = entry[side]
      if not isinstance(views, list) or not views:
        raise PackageError("MISSING_VIEWS", f"{path}.{side}", "a kernel needs explicit ins and outs")
      for j, view in enumerate(views):
        _view(view, f"{path}.{side}[{j}]", spec)
    _aliases(entry, path)
    _dtype_rule(entry, path, spec)
    _initial(entry["initial"], path, spec)
    _scratch(entry["scratch"], path, spec)
    _events(entry["events"], path, spec, seen_events)
    if not isinstance(entry["clobber"], list) or not all(
      isinstance(c, str) and c for c in entry["clobber"]
    ):
      raise PackageError("WRONG_TYPE", f"{path}.clobber")
    if entry["numeric_profile"] != spec_all["package"]["numeric_profile"]["must_equal_id"]:
      raise PackageError("NUMERIC_PROFILE_MISMATCH", f"{path}.numeric_profile")
    if entry["mode_table"] != reg["mode_table"]:
      raise PackageError("MODE_TABLE_MISMATCH", f"{path}.mode_table", f"{entry['mode_table']!r}")
    if entry["mode_table"] not in tables:
      raise PackageError("MODE_TABLE_UNKNOWN", f"{path}.mode_table", f"{entry['mode_table']!r}")
    _region(entry["region"], path, abi)
    _helpers(entry["helpers"], path, spec_all, csr_names)
    _patch_mask_entry(entry["patch_mask"], path, spec_all)


def _aliases(entry: dict[str, Any], path: str) -> None:
  handles = [v["handle"] for v in entry["ins"]] + [v["handle"] for v in entry["outs"]]
  if len(handles) != len(set(handles)):
    raise PackageError("DUPLICATE_HANDLE", path, "view handles are unique within a kernel")
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


def _dtype_rule(entry: dict[str, Any], path: str, spec: dict[str, Any]) -> None:
  rule = spec["tensor_dtype_rule"]
  for j, out in enumerate(entry["outs"]):
    if out["dtype"] not in (rule["tensor_out"], "i32"):
      raise PackageError("EXTERNAL_DTYPE_FORBIDDEN", f"{path}.outs[{j}].dtype", f"{out['dtype']!r}")
  for j, view in enumerate(entry["ins"]):
    if view["dtype"] not in (rule["tensor_in"], "i32", "f32"):
      raise PackageError("BAD_DTYPE", f"{path}.ins[{j}].dtype")


def _initial(initial: Any, path: str, spec: dict[str, Any]) -> None:
  _fields(initial, ["method"], f"{path}.initial", allowed=["method", "note"])
  if initial["method"] not in spec["initial"]["methods"]:
    raise PackageError(
      "BAD_INITIAL", f"{path}.initial.method", f"a seed is one of {spec['initial']['methods']}"
    )


def _scratch(scratch: Any, path: str, spec: dict[str, Any]) -> None:
  if not isinstance(scratch, list):
    raise PackageError("WRONG_TYPE", f"{path}.scratch")
  for j, item in enumerate(scratch):
    _fields(item, spec["scratch"]["required_fields"], f"{path}.scratch[{j}]")
    if not _is_int(item["bytes"]) or item["bytes"] <= 0:
      raise PackageError("BAD_SCRATCH", f"{path}.scratch[{j}].bytes")
    if item["bound_source"] not in spec["scratch"]["bound_source_values"]:
      raise PackageError(
        "BOUND_NOT_STATIC", f"{path}.scratch[{j}].bound_source", "a measured peak is evidence, not a bound"
      )


def _events(events: Any, path: str, spec: dict[str, Any], seen: set[str]) -> None:
  rules = spec["events"]
  _fields(events, rules["required"], f"{path}.events", allowed=rules["required"] + rules["optional"])
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


def _region(region: Any, path: str, abi: dict[str, Any]) -> None:
  _fields(region, ["start", "end", "chains"], f"{path}.region")
  if region["start"] is not True or region["end"] is not True:
    raise PackageError("REGION_BOUNDS_MISSING", f"{path}.region", "START and END must both be declared")
  chains = region["chains"]
  if not isinstance(chains, list):
    raise PackageError("WRONG_TYPE", f"{path}.region.chains")
  kinds = set(abi["declaration"]["kinds"])
  for j, chain in enumerate(chains):
    cpath = f"{path}.region.chains[{j}]"
    _fields(chain, ["id", "steps"], cpath)
    steps = chain["steps"]
    if not isinstance(steps, list) or any(s not in kinds for s in steps):
      raise PackageError("BAD_CHAIN_KIND", f"{cpath}.steps", "every step is an ABI declaration kind")
    if not steps or steps[0] != "CHAIN_FIRST":
      raise PackageError("CHAIN_WITHOUT_FIRST", cpath)
    if steps.count("CHAIN_FIRST") != 1:
      raise PackageError("DUPLICATE_FIRST", cpath)
    if steps[-1] != "CHAIN_FINAL" or steps.count("CHAIN_FINAL") != 1:
      raise PackageError("CHAIN_WITHOUT_FINAL", cpath)


def _helpers(helpers: Any, path: str, spec_all: dict[str, Any], csr_names: set[str]) -> None:
  spec = spec_all["helpers"]
  if not isinstance(helpers, list):
    raise PackageError("WRONG_TYPE", f"{path}.helpers")
  registered = {h["name"]: h for h in spec["registered"]}
  for j, helper in enumerate(helpers):
    hp = f"{path}.helpers[{j}]"
    _fields(helper, spec["entry_required_fields"], hp)
    reg = registered.get(helper["name"])
    if reg is None:
      raise PackageError("UNDECLARED_HELPER", f"{hp}.name", f"{helper['name']!r} cannot be called")
    csr = helper["csr"]
    if not isinstance(csr, str) or csr.startswith(("0x", "0X")) or csr.isdigit():
      raise PackageError("HAND_WRITTEN_ADDRESS", f"{hp}.csr")
    if csr != reg["csr"] or csr not in csr_names:
      raise PackageError(
        "HELPER_CSR_MISMATCH", f"{hp}.csr", "a CSR is referenced by its generated logical name"
      )
    if helper["barrier"] not in spec["barrier_kinds"]:
      raise PackageError("BAD_BARRIER", f"{hp}.barrier", f"{helper['barrier']!r}")
    if not isinstance(helper["observable"], str) or not helper["observable"].strip():
      raise PackageError("MISSING_OBSERVABLE", f"{hp}.observable")


def _patch_mask_entry(mask: Any, path: str, spec_all: dict[str, Any]) -> None:
  if not isinstance(mask, list):
    raise PackageError("WRONG_TYPE", f"{path}.patch_mask")
  seen: set[str] = set()
  spec = spec_all["patch_mask"]
  for j, item in enumerate(mask):
    _fields(item, spec["required_fields"], f"{path}.patch_mask[{j}]")
    if item["layer"] not in spec["layers"]:
      raise PackageError("BAD_PATCH_LAYER", f"{path}.patch_mask[{j}].layer", f"{item['layer']!r}")
    if item["field"] in seen:
      raise PackageError("DUPLICATE_PATCH_FIELD", f"{path}.patch_mask[{j}].field")
    seen.add(item["field"])


def _resources(package: dict[str, Any], spec: dict[str, Any]) -> None:
  rm = package["resource_manifest"]
  keys = ["text_bytes", "static_data_bytes", "stack_bytes", "scratch_bytes", "bound_source"]
  _fields(rm, keys, "$.resource_manifest")
  for key in keys[:4]:
    if not _is_int(rm[key]) or rm[key] < 0:
      raise PackageError("WRONG_TYPE", f"$.resource_manifest.{key}")
  if rm["bound_source"] not in spec["binding_manifest"]["scratch"]["bound_source_values"]:
    raise PackageError(
      "BOUND_NOT_STATIC", "$.resource_manifest.bound_source", "only static analysis is a bound"
    )
  if rm["text_bytes"] != package["image"]["text_bytes"]:
    raise PackageError("TEXT_BYTES_DISAGREE", "$.resource_manifest.text_bytes")
  caps = spec["image"]["capacities"]
  layout = package["image"]["link_layout"]
  if rm["static_data_bytes"] != layout["data_bytes"]:
    raise PackageError("DATA_BYTES_DISAGREE", "$.resource_manifest.static_data_bytes")
  if rm["static_data_bytes"] + rm["stack_bytes"] + rm["scratch_bytes"] > caps["dtcm_bytes"]["value"]:
    raise PackageError(
      "DATA_EXCEEDS_CAPACITY", "$.resource_manifest", f"> {caps['dtcm_bytes']['value']} B DTCM"
    )
  if rm["stack_bytes"] > caps["stack_bytes"]["value"]:
    raise PackageError(
      "STACK_EXCEEDS_CAPACITY", "$.resource_manifest.stack_bytes", f"> {caps['stack_bytes']['value']}"
    )
  tf = package["target_fingerprint"]
  fp = ["tensor_input_bytes", "accumulator_bytes", "tensor_output_bytes", "rvv_vlen", "context_count"]
  _fields(tf, fp, "$.target_fingerprint")
  for key in fp:
    if not _is_int(tf[key]) or tf[key] <= 0:
      raise PackageError("WRONG_TYPE", f"$.target_fingerprint.{key}")
  if (tf["tensor_input_bytes"], tf["accumulator_bytes"], tf["tensor_output_bytes"]) != (2, 4, 2):
    raise PackageError(
      "TARGET_DTYPE_MISMATCH", "$.target_fingerprint", "BF16 in, FP32 accumulator, BF16 out"
    )


def _patch_mask(mask: Any, spec: dict[str, Any]) -> None:
  rules = spec["patch_mask"]
  if not isinstance(mask, list):
    raise PackageError("WRONG_TYPE", "$.patch_mask")
  seen: set[str] = set()
  for j, item in enumerate(mask):
    _fields(item, rules["required_fields"], f"$.patch_mask[{j}]")
    if item["layer"] not in rules["layers"]:
      raise PackageError("BAD_PATCH_LAYER", f"$.patch_mask[{j}].layer", f"{item['layer']!r}")
    if item["field"] in seen:
      raise PackageError("DUPLICATE_PATCH_FIELD", f"$.patch_mask[{j}].field")
    seen.add(item["field"])


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


def layouts_coexist(a: dict[str, Any], b: dict[str, Any]) -> None:
  """Two fixed pre-linked images may be resident together only if their address ranges are disjoint."""
  la, lb = a["image"]["link_layout"], b["image"]["link_layout"]
  for ab, an in ((la["text_base"], la["text_bytes"]), (la["data_base"], la["data_bytes"])):
    for bb, bn in ((lb["text_base"], lb["text_bytes"]), (lb["data_base"], lb["data_bytes"])):
      if _overlap(ab, an, bb, bn):
        raise PackageError("IMAGES_OVERLAP", "image.link_layout", f"[{ab:#x},+{an}) vs [{bb:#x},+{bn})")
