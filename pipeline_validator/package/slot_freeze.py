"""Cross-check the C038 two-image handoff after both packages passed the read-only Loader."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .binding_ir import PackageError, canonical, layouts_coexist

if TYPE_CHECKING:
  from .loader import LoadedPackage


def freeze_digest(measurements: dict[str, Any]) -> str:
  """The packages sign the common measurements, not their own final package hashes."""
  return hashlib.sha256(canonical({k: v for k, v in measurements.items() if k != "packages"})).hexdigest()


def check_measurements(package: dict, freeze: dict | None) -> None:
  for entry in package["binding_manifest"]["kernels"]:
    if "linked_slot" not in entry:
      continue  # Structural validation requires this field for every scalar entry.
    linked = entry["linked_slot"]
    if not isinstance(freeze, dict) or linked["freeze_sha256"] != freeze_digest(freeze):
      raise PackageError("SLOT_FREEZE_MISMATCH", "$.linked_slot.freeze_sha256", "missing or stale freeze")
    rows = freeze.get("slots")
    if not isinstance(rows, list) or len(rows) != 2 or any(not isinstance(row, dict) for row in rows):
      raise PackageError("SLOT_SET_MISMATCH", "$.slots", "two measured slots are required")
    by_name = {row.get("name"): row for row in rows}
    if set(by_name) != {"A", "B"}:
      raise PackageError("SLOT_SET_MISMATCH", "$.slots", "expected measured slots A and B")
    row = by_name[linked["slot"]]
    image = package["image"]
    fields = {**image["link_layout"], "entry": image["entry"], "image_sha256": image["bytes_sha256"]}
    for key, value in fields.items():
      if row.get(key) != value or linked.get(key) != value:
        raise PackageError("SLOT_MEASUREMENT_MISMATCH", f"$.linked_slot.{key}", "freeze and image disagree")
    if row.get("program_id") != linked["program_id"]:
      raise PackageError("SLOT_IDENTITY_MISMATCH", "$.linked_slot.program_id")
    if "packages" in freeze and "package_hash" in package:
      slot = linked["slot"]
      index = freeze["packages"]
      expected = {
        "path": f"{slot}/package.json",
        "program_id": linked["program_id"],
        "version": entry["kernel_version"],
        "image_sha256": image["bytes_sha256"],
        "package_hash": package["package_hash"],
      }
      if not isinstance(index, dict) or canonical(index.get(slot)) != canonical(expected):
        raise PackageError(
          "SLOT_IDENTITY_MISMATCH", f"$.packages.{slot}", "not the sealed program identity"
        )


def verify_slot_freeze(freeze: dict[str, Any], loaded: Mapping[str, LoadedPackage]) -> None:
  """Do not use paths or registration IDs from an unchecked external index for installation."""
  index = freeze.get("packages")
  if not isinstance(index, dict) or set(index) != {"A", "B"} or set(index) != set(loaded):
    raise PackageError("SLOT_SET_MISMATCH", "$.packages", "both fixed images are required")
  packages = [loaded[slot].to_dict() for slot in ("A", "B")]
  layouts_coexist(*packages)
  for slot, pkg in zip(("A", "B"), packages, strict=True):
    check_measurements(pkg, freeze)
    (entry,) = pkg["binding_manifest"]["kernels"]
    linked = entry.get("linked_slot", {})
    if linked.get("slot") != slot:
      raise PackageError(
        "SLOT_FREEZE_MISMATCH", f"$.slots.{slot}", "not the measurements sealed by this image"
      )
