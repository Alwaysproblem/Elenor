"""C038: consume both published packages as independently linked instances of the scalar program."""

from __future__ import annotations

import json
import struct
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.cosim.program_service import ProgramImage, ProgramService, ProgramServiceError
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.package import PackageError, load_file, package_hash, verify
from pipeline_validator.package.slot_freeze import check_measurements, verify_slot_freeze

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "hardware/coral_nexus"


def executable_segment(elf: bytes) -> bytes:
  phoff = struct.unpack_from("<I", elf, 28)[0]
  stride, count = struct.unpack_from("<HH", elf, 42)
  for index in range(count):
    kind, offset, _, _, size, _, flags, _ = struct.unpack_from("<8I", elf, phoff + stride * index)
    if kind == 1 and flags & 1:
      return elf[offset : offset + size]
  raise AssertionError("no loaded text")


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> Path:
  out = tmp_path_factory.mktemp("scalar-package")
  subprocess.run(
    [
      "bash",
      str(H / "scripts/cn.sh"),
      "package",
      "--binding",
      str(H / "kernels/bindings/scalar_ctl_v1.yaml"),
      "--config",
      str(H / "configs/cs1_c2.yaml"),
      "--out",
      str(out),
    ],
    check=True,
    timeout=300,
  )
  return out


@pytest.fixture
def target() -> dict[str, int]:
  vlen = next(
    p["value"]
    for p in json.loads((H / "configs/params.d/upstream_coralnpu.json").read_bytes())["params"]
    if p["name"] == "rvv_vlen_bits"
  )
  return {
    "context_count": 2,
    "rvv_vlen": vlen,
    "tensor_input_bytes": 2,
    "tensor_output_bytes": 2,
    "accumulator_bytes": 4,
  }


@pytest.fixture
def bindings() -> dict[str, GlobalBinding]:
  return {"X": GlobalBinding("X", 0x100000, 32, "r"), "Y": GlobalBinding("Y", 0x200000, 32, "w")}


def test_two_slot_freeze_is_consumable_and_identity_cannot_be_rebound(out, target, bindings) -> None:
  freeze = json.loads((out / "slot_freeze.json").read_bytes())
  registry = ProgramService()
  images, loaded_slots = [], {}
  for slot in ("A", "B"):
    identity = freeze["packages"][slot]
    path = out / slot / "package.json"
    elf_path = path.with_name("image.elf")
    loaded = load_file(
      path,
      compiled_path=path.with_name("compiled_program.json"),
      hw=HardwareConfig(),
      sim=SimConfig(fidelity="runtime"),
      actual_bindings=bindings,
      target=target,
      image_path=elf_path,
      slot_freeze=freeze,
    )
    assert loaded.package_hash == identity["package_hash"]
    pkg = loaded.to_dict()
    (kernel,) = pkg["binding_manifest"]["kernels"]
    assert kernel["kernel_id"] == "K-SCALAR-ADD7" and kernel["kernel_version"] == identity["version"]
    text = executable_segment(elf_path.read_bytes())
    assert len(text) == pkg["resource_manifest"]["text_bytes"]
    image = ProgramImage(
      kernel["linked_slot"]["program_id"],
      kernel["kernel_version"],
      bytes.fromhex(loaded.package_hash),
      0,
      text,
    )
    registry.register_image(image)
    images.append(image)
    loaded_slots[slot] = loaded
  verify_slot_freeze(freeze, loaded_slots)
  from pipeline_validator.package import seal

  a_path = out / "A/package.json"
  body = a_path.with_name("compiled_program.json")
  hw, sim = HardwareConfig(), SimConfig(fidelity="runtime")
  with pytest.raises(PackageError, match="SLOT_FREEZE_MISMATCH"):
    seal(
      loaded_slots["A"].to_dict(), compiled_text=body.read_text(), hw=hw, sim=sim, actual_bindings=bindings
    )
  with pytest.raises(PackageError, match="SLOT_FREEZE_MISMATCH"):
    load_file(
      a_path,
      compiled_path=body,
      hw=hw,
      sim=sim,
      actual_bindings=bindings,
      target=target,
      image_path=a_path.with_name("image.elf"),
    )
  with pytest.raises(PackageError, match="SLOT_FREEZE_MISMATCH"):
    verify_slot_freeze({**freeze, "slots": list(reversed(freeze["slots"]))}, loaded_slots)
  substituted = {**freeze, "packages": {**freeze["packages"], "A": freeze["packages"]["B"]}}
  with pytest.raises(PackageError, match="SLOT_IDENTITY_MISMATCH"):
    verify_slot_freeze(substituted, loaded_slots)
  assert images[0].package_hash != images[1].package_hash
  with pytest.raises(ProgramServiceError, match="IMAGE_IDENTITY_CONFLICT"):
    registry.register_image(replace(images[1], program_id=images[0].program_id))
  registry.register_image(images[0])  # original still registered, idempotent
  assert registry.stats["images_registered"] == 2


def test_aligned_output_offset_must_match_compiled_transfer(out, target, bindings, tmp_path) -> None:
  slot = out / "A"
  pkg = json.loads((slot / "package.json").read_bytes())
  output = pkg["binding_manifest"]["kernels"][0]["launch_descriptors"][1]
  assert (output["direction"], output["frame_offset"]) == ("frame_to_l2", 1024)
  output["frame_offset"] = 1028
  pkg["package_hash"] = package_hash(pkg)
  freeze = json.loads((out / "slot_freeze.json").read_bytes())
  freeze["packages"]["A"]["package_hash"] = pkg["package_hash"]
  verify(pkg)
  check_measurements(pkg, freeze)

  forged = tmp_path / "package.json"
  forged.write_text(json.dumps(pkg))
  with pytest.raises(PackageError) as caught:
    load_file(
      forged,
      compiled_path=slot / "compiled_program.json",
      image_path=slot / "image.elf",
      hw=HardwareConfig(),
      sim=SimConfig(fidelity="runtime"),
      actual_bindings=bindings,
      target=target,
      slot_freeze=freeze,
    )
  assert (caught.value.code, caught.value.field) == (
    "CONTROL_PLANE_MISMATCH",
    "$.binding_manifest.launch_descriptors",
  )
