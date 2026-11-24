"""Read-only cn_package/v1 Loader (C037 / W5-01).

The Loader never edits what it reads: it parses strictly, validates, verifies the seal and, when asked,
the image bytes and the target. Every refusal is a ``PackageError`` that names the offending field. The
returned object is a frozen view; patching at run time is only ever through ``patch_mask`` (``binding_ir.
patch_allowed``) and never changes ``package_hash`` or resident text.

Trust boundary (contract section 9): this proves the CONTRACT layer. It does not prove arbitrary machine
code; the image is trusted through its controlled build and the numeric/protocol verification.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..config import HardwareConfig, SimConfig
from ..execution_ir import GlobalBinding
from .binding_ir import PackageError, parse_strict
from .seal import verify, verify_control_plane, verify_image
from .slot_freeze import check_measurements


def _freeze(value: Any) -> Any:
  if isinstance(value, dict):
    return MappingProxyType({k: _freeze(v) for k, v in value.items()})
  if isinstance(value, list):
    return tuple(_freeze(v) for v in value)
  return value


@dataclass(frozen=True)
class LoadedPackage:
  """An immutable, verified package. ``raw`` is a private deep copy used only to re-serialize."""

  view: Any
  package_hash: str
  text: bytes  # executable PT_LOAD payload of the exact ELF verified at load
  _raw: dict[str, Any]

  def to_dict(self) -> dict[str, Any]:
    return copy.deepcopy(self._raw)


def check_target(package: dict[str, Any], target: dict[str, Any]) -> None:
  """A package is bound to one target fingerprint; any differing field refuses the load by name."""
  want = package["target_fingerprint"]
  for key in sorted(want):
    if key not in target:
      raise PackageError(
        "TARGET_FIELD_MISSING", f"$.target_fingerprint.{key}", "the target did not state it"
      )
    if target[key] != want[key]:
      raise PackageError(
        "TARGET_MISMATCH", f"$.target_fingerprint.{key}", f"package {want[key]!r}, target {target[key]!r}"
      )


def load_text(
  text: str,
  *,
  compiled_text: str,
  hw: HardwareConfig,
  sim: SimConfig,
  actual_bindings: Mapping[str, GlobalBinding],
  image_bytes: bytes,
  target: dict[str, Any],
  slot_freeze: dict | None = None,
) -> LoadedPackage:
  """Load a package as installation proof: the contract layer, the wrapped CompiledProgram (codec and
  semantic Loader), the shipped image bytes AND the independent Coral target are all mandatory. Use
  ``verify`` for metadata-only inspection of a package whose image or target is not at hand."""
  package = parse_strict(text)
  verify(package)
  check_measurements(package, slot_freeze)
  verify_control_plane(package, compiled_text, hw=hw, sim=sim, actual_bindings=actual_bindings)
  payload = verify_image(package, image_bytes)
  check_target(package, target)
  return LoadedPackage(
    view=_freeze(package),
    package_hash=package["package_hash"],
    text=bytes(payload),
    _raw=copy.deepcopy(package),
  )


def load_file(
  path: Path,
  *,
  compiled_path: Path,
  hw: HardwareConfig,
  sim: SimConfig,
  actual_bindings: Mapping[str, GlobalBinding],
  image_path: Path,
  target: dict[str, Any],
  slot_freeze: dict | None = None,
) -> LoadedPackage:
  try:
    text = Path(path).read_text()
  except OSError as exc:
    raise PackageError("UNREADABLE", str(path), str(exc)) from exc
  try:
    compiled = Path(compiled_path).read_text()
  except OSError as exc:
    raise PackageError("UNREADABLE", str(compiled_path), str(exc)) from exc
  try:
    image = Path(image_path).read_bytes()
  except OSError as exc:
    raise PackageError("UNREADABLE", str(image_path), str(exc)) from exc
  return load_text(
    text,
    compiled_text=compiled,
    hw=hw,
    sim=sim,
    actual_bindings=actual_bindings,
    image_bytes=image,
    target=target,
    slot_freeze=slot_freeze,
  )
