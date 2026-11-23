"""cn_package/v1 sealing (C037 / W5-01).

``package_hash`` is sha256 over exactly the parts ``w5_package.contract.yaml`` says it covers, and never
over itself. Two rules make the seal more than a checksum:

* ``seal`` validates first: a package that breaks the contract is never given a hash.
* ``verify`` recomputes the hash AND re-runs the full structural validation. Recomputing a hash after
  editing the contract layer therefore does not help: the edit is rejected on its meaning, not its digest.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .binding_ir import PackageError, canonical, contract, validate


def _parts(package: dict[str, Any]) -> dict[str, Any]:
  image = package.get("image")
  control = package.get("control_plane")
  return {
    "image_sha256": image.get("bytes_sha256") if isinstance(image, dict) else None,
    "binding_manifest": package.get("binding_manifest"),
    "resource_manifest": package.get("resource_manifest"),
    "target_fingerprint": package.get("target_fingerprint"),
    "patch_mask": package.get("patch_mask"),
    "numeric_profile_id": package.get("numeric_profile_id"),
    "control_plane_artifact_hash": control.get("artifact_hash") if isinstance(control, dict) else None,
  }


def package_hash(package: dict[str, Any]) -> str:
  """sha256 over the parts the contract names (a part this code cannot hash is an error, not a skip)."""
  covers = contract()["package"]["package_hash"]["covers"]
  parts = _parts(package)
  unknown = [name for name in covers if name not in parts]
  if unknown:
    raise PackageError(
      "HASH_COVERAGE_UNKNOWN", "package_hash", f"the contract covers {unknown}, which this code cannot hash"
    )
  for name in covers:
    if parts[name] is None:
      raise PackageError("MISSING_FIELD", f"$.{name}", "a covered part is absent")
  return hashlib.sha256(canonical({name: parts[name] for name in covers})).hexdigest()


def seal(package: dict[str, Any]) -> dict[str, Any]:
  """Return a copy carrying its ``package_hash``; the input is validated first and is never mutated."""
  body = {k: v for k, v in package.items() if k != "package_hash"}
  validate({**body, "package_hash": "0" * 64})
  return {**body, "package_hash": package_hash(body)}


def verify(package: Any) -> None:
  """Validate the structure, then check the stored hash against a recomputation."""
  validate(package)
  recomputed = package_hash(package)
  if package["package_hash"] != recomputed:
    raise PackageError(
      "HASH_MISMATCH",
      "$.package_hash",
      f"stored {package['package_hash'][:12]}…, recomputed {recomputed[:12]}…",
    )


def verify_image(package: dict[str, Any], image_bytes: bytes) -> None:
  """The image identity is its bytes: a package that names an image must match the bytes it ships."""
  actual = hashlib.sha256(image_bytes).hexdigest()
  if actual != package["image"]["bytes_sha256"]:
    raise PackageError(
      "IMAGE_BYTES_MISMATCH", "$.image.bytes_sha256", f"image bytes hash to {actual[:12]}…"
    )
