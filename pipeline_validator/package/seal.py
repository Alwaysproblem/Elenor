"""cn_package/v1 sealing (C037 / W5-01).

``package_hash`` is sha256 over exactly the parts ``w5_package.contract.yaml`` says it covers, and never
over itself. Two rules make the seal more than a checksum:

* ``seal`` validates first: a package that breaks the contract is never given a hash.
* ``verify`` recomputes the hash AND re-runs the full structural validation. Recomputing a hash after
  editing the contract layer therefore does not help: the edit is rejected on its meaning, not its digest.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from typing import Any

from ..compiled_program import parse_compiled_program
from ..config import HardwareConfig, SimConfig
from ..execution_ir import GlobalBinding
from ..loader import load_program
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


def verify_control_plane(
  package: dict[str, Any],
  compiled_text: str,
  *,
  hw: HardwareConfig,
  sim: SimConfig,
  actual_bindings: Mapping[str, GlobalBinding],
) -> None:
  """The wrapped CompiledProgram is the control plane, not a digest about it.

  Two layers, both required (CMP-14, "a tampered contract layer is rejected even after rehash"):

  * the CODEC (``parse_compiled_program``): strict JSON, schema/ABI, structure, and a recomputed
    ``artifact_hash`` that is never trusted. This only proves the body is self-consistent;
  * the SEMANTIC Loader (``pipeline_validator.loader.load_program``) against the real validator target
    and actual bindings: an artifact whose inner digest was recomputed after a semantic edit (a forged
    double release, a rebound layout, a source/IR mismatch) is rejected on what it does.

  The package's ``control_plane`` must then name exactly that program. Recomputing the outer
  ``package_hash`` therefore cannot rescue a substituted digest or an edited body.
  """
  try:
    program = parse_compiled_program(compiled_text)
  except (ValueError, TypeError) as exc:
    raise PackageError("CONTROL_PLANE_INVALID", "$.control_plane", str(exc)) from exc
  try:
    load_program(program, hw, sim, actual_bindings=actual_bindings)
  except (ValueError, TypeError) as exc:
    raise PackageError("CONTROL_PLANE_REJECTED", "$.control_plane", str(exc)) from exc
  cp = package["control_plane"]
  actual = {
    "schema_version": program.schema_version,
    "compiler_abi": program.compiler_abi,
    "artifact_hash": program.artifact_hash,
  }
  if "source_hash" in cp:
    actual["source_hash"] = program.source_hash
  for key, value in actual.items():
    if cp[key] != value:
      raise PackageError(
        "CONTROL_PLANE_MISMATCH",
        f"$.control_plane.{key}",
        f"package {cp[key]!r}, wrapped CompiledProgram {value!r}",
      )


def seal(
  package: dict[str, Any],
  *,
  compiled_text: str,
  hw: HardwareConfig,
  sim: SimConfig,
  actual_bindings: Mapping[str, GlobalBinding],
) -> dict[str, Any]:
  """Return a copy carrying its ``package_hash``; validated first, checked against the wrapped
  CompiledProgram by the codec and the semantic Loader, and never mutated."""
  body = {k: v for k, v in package.items() if k != "package_hash"}
  validate({**body, "package_hash": "0" * 64})
  verify_control_plane(body, compiled_text, hw=hw, sim=sim, actual_bindings=actual_bindings)
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


_PT_LOAD, _PF_X, _PF_W = 1, 1, 2
_EM_RISCV = 243


def _elf_layout(image_bytes: bytes) -> dict[str, int | None]:
  """The execution-relevant identity of a linked RV32 ELF, from its header and PT_LOAD segments only.

  Nothing here analyses machine code: it reads ``e_entry`` and the loaded segments' addresses and
  lengths, so a manifest cannot claim a layout the shipped bytes do not have.
  """

  def fail(why: str) -> PackageError:
    return PackageError("IMAGE_NOT_ELF", "$.image", why)

  if len(image_bytes) < 52 or image_bytes[:4] != b"\x7fELF":
    raise fail("not an ELF file")
  if image_bytes[4] != 1 or image_bytes[5] != 1:
    raise fail("only 32-bit little-endian ELF is a Coral image")
  if struct.unpack_from("<H", image_bytes, 18)[0] != _EM_RISCV:
    raise fail("not a RISC-V ELF")
  entry, phoff = struct.unpack_from("<II", image_bytes, 24)[0], struct.unpack_from("<I", image_bytes, 28)[0]
  phentsize, phnum = struct.unpack_from("<HH", image_bytes, 42)
  if phentsize < 32 or phoff + phnum * phentsize > len(image_bytes):
    raise fail("program header table is outside the file")
  text: list[tuple[int, int]] = []
  data: list[tuple[int, int, int]] = []
  for index in range(phnum):
    kind, _off, vaddr, _paddr, filesz, memsz, flags, _align = struct.unpack_from(
      "<IIIIIIII", image_bytes, phoff + index * phentsize
    )
    if kind != _PT_LOAD or memsz == 0:
      continue
    if flags & _PF_X:
      text.append((vaddr, filesz))
    elif flags & _PF_W:
      data.append((vaddr, filesz, memsz))
  if len(text) != 1:
    raise fail(f"expected exactly one executable PT_LOAD segment, found {len(text)}")
  return {
    "entry": entry,
    "text_base": text[0][0],
    "text_bytes": text[0][1],
    "data_base": min(v for v, _f, _m in data) if data else None,
    "data_filesz": sum(f for _v, f, _m in data),
    "data_memsz": sum(m for _v, _f, m in data),
  }


def verify_image(package: dict[str, Any], image_bytes: bytes) -> None:
  """The image identity is its bytes, and the manifest's execution metadata must be what those bytes are.

  ``image.bytes_sha256`` authenticates the bytes but ``entry`` and ``link_layout`` are not under the
  package hash, so each is checked against the actual ELF (entry point, executable segment address and
  length, data segment address and extent). Changing one of them without relinking is refused here.
  """
  image = package["image"]
  actual = hashlib.sha256(image_bytes).hexdigest()
  if actual != image["bytes_sha256"]:
    raise PackageError(
      "IMAGE_BYTES_MISMATCH", "$.image.bytes_sha256", f"image bytes hash to {actual[:12]}…"
    )
  elf = _elf_layout(image_bytes)
  layout = image["link_layout"]
  checks = (
    ("$.image.entry", image["entry"], elf["entry"], "ENTRY_MISMATCH", "e_entry"),
    (
      "$.image.link_layout.text_base",
      layout["text_base"],
      elf["text_base"],
      "LAYOUT_MISMATCH",
      "text base",
    ),
    ("$.image.text_bytes", image["text_bytes"], elf["text_bytes"], "TEXT_BYTES_MISMATCH", "text length"),
  )
  for field, claimed, linked, code, what in checks:
    if claimed != linked:
      raise PackageError(code, field, f"manifest {claimed}, linked ELF {what} {linked}")
  resource = package["resource_manifest"]
  if elf["data_base"] is None:
    if layout["data_bytes"] or resource["static_data_bytes"] or resource["stack_bytes"]:
      raise PackageError(
        "LAYOUT_MISMATCH",
        "$.image.link_layout.data_base",
        "the manifest declares data the ELF does not load",
      )
    return
  if layout["data_base"] != elf["data_base"]:
    raise PackageError(
      "LAYOUT_MISMATCH",
      "$.image.link_layout.data_base",
      f"manifest {layout['data_base']}, linked ELF data base {elf['data_base']}",
    )
  assert elf["data_filesz"] is not None and elf["data_memsz"] is not None
  if elf["data_filesz"] > layout["data_bytes"]:
    raise PackageError(
      "LAYOUT_MISMATCH",
      "$.image.link_layout.data_bytes",
      f"the ELF ships {elf['data_filesz']} initialized data bytes, "
      f"the manifest declares {layout['data_bytes']}",
    )
  need = layout["data_bytes"] + resource["stack_bytes"] + resource["scratch_bytes"]
  if need > elf["data_memsz"]:
    raise PackageError(
      "LAYOUT_MISMATCH",
      "$.image.link_layout.data_bytes",
      f"declared data + proven stack + scratch is {need}, the linked data region is {elf['data_memsz']}",
    )
