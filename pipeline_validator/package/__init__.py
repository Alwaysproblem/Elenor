"""Executable Package IR for the runtime-level simulator."""

from __future__ import annotations

from .binding_ir import PackageError, encode, layouts_coexist, parse_strict, patch_allowed, validate
from .loader import LoadedPackage, check_target, load_file, load_text
from .package_ir import (
  DeviceCaps,
  ElenorPackage,
  GraphSchedule,
  PackageHeader,
  PackageSection,
  RelocationEntry,
  validate_package,
)
from .seal import package_hash, seal, verify, verify_control_plane, verify_image

__all__ = [
  "DeviceCaps",
  "ElenorPackage",
  "GraphSchedule",
  "LoadedPackage",
  "PackageError",
  "PackageHeader",
  "PackageSection",
  "RelocationEntry",
  "check_target",
  "encode",
  "layouts_coexist",
  "load_file",
  "load_text",
  "package_hash",
  "parse_strict",
  "patch_allowed",
  "seal",
  "validate",
  "validate_package",
  "verify",
  "verify_control_plane",
  "verify_image",
]
