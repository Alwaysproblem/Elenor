"""Prepare a PROGRAM-channel endpoint from a C038 release; never builds or installs an image."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import HardwareConfig, SimConfig
from ..execution_ir import GlobalBinding
from ..memory.byte_store import ByteStore
from ..memory.hbm_region import HBMRegion
from ..package import LoadedPackage, PackageError, load_file, parse_strict
from ..package.slot_freeze import verify_slot_freeze
from ..runtime.program_table import ProgramIdentity, ProgramResidencyManager, ResidencyStorage
from .program_service import ProgramImage, ProgramService

H = Path(__file__).resolve().parents[2] / "hardware/coral_nexus"


@dataclass
class PreparedPrograms:
  scenario: dict[str, Any]
  packages: dict[str, LoadedPackage]
  images: dict[str, ProgramImage]
  hbm: HBMRegion
  service: ProgramService
  storage: ResidencyStorage

  def attach(self, manager: ProgramResidencyManager, tile_id: int) -> dict[str, ProgramIdentity]:
    """Prepare one live Group manager for this Tile from measured, Loader-verified placements."""
    manager.configure_storage(self.storage)
    manager.attach(tile_id, self.service)
    return {
      slot: manager.register_image(
        self.images[slot],
        text_base=pkg.view["image"]["link_layout"]["text_base"],
        patch_mask=pkg.view["patch_mask"],
      )
      for slot, pkg in self.packages.items()
    }


def prepare_programs(
  scenario_path: Path, package_root: Path, *, config_path: Path, hbm: HBMRegion | None = None
) -> PreparedPrograms:
  scenario = yaml.safe_load(scenario_path.read_text())
  if scenario.get("schema") != "cn_program_scenario/v1" or set(scenario["slots"]) != {"A", "B"}:
    raise ValueError("expected a two-slot cn_program_scenario/v1")
  if scenario["slot_freeze"] != "slot_freeze.json":
    raise ValueError("unrecognized freeze reference")
  freeze = parse_strict((package_root / scenario["slot_freeze"]).read_text())
  config = yaml.safe_load(config_path.read_text())
  if config.get("kind") != "cs1_target":
    raise ValueError("a CS1 target is required")
  params = json.loads((H / "configs/params.d/upstream_coralnpu.json").read_text())["params"]
  target = {
    "rvv_vlen": next(p["value"] for p in params if p["name"] == "rvv_vlen_bits"),
    "context_count": config["constraints"]["context_count"],
    "tensor_input_bytes": 2,
    "accumulator_bytes": 4,
    "tensor_output_bytes": 2,
  }
  hw, sim = HardwareConfig(), SimConfig(fidelity="runtime")
  bindings = {
    name: GlobalBinding(name=name, **fields) for name, fields in scenario["launch_bindings"].items()
  }
  packages: dict[str, LoadedPackage] = {}
  for slot, spec in scenario["slots"].items():
    if spec["package"] != f"{slot}/package.json":
      raise ValueError(f"invalid package path for slot {slot}")
    path = package_root / spec["package"]
    loaded = load_file(
      path,
      compiled_path=path.with_name("compiled_program.json"),
      hw=hw,
      sim=sim,
      actual_bindings=bindings,
      target=target,
      image_path=path.with_name("image.elf"),
      slot_freeze=freeze,
    )
    if loaded.package_hash != spec["package_hash"]:
      raise PackageError("SCENARIO_HASH_MISMATCH", f"$.slots.{slot}.package_hash")
    packages[slot] = loaded
  verify_slot_freeze(freeze, packages)
  geometry = freeze["slot_geometry"]
  for field in ("itcm_bytes", "dtcm_bytes", "program_slots"):
    if config["constraints"][field] != geometry[field]:
      raise PackageError("TARGET_MISMATCH", f"$.slot_geometry.{field}")
  owns_hbm = hbm is None
  if hbm is None:
    hbm = HBMRegion(size_bytes=hw.hbm_capacity_bytes, byte_store=ByteStore())
  # Preflight all ranges against the live Group without exposing a partial upload on a bad second slot.
  ranges = list(hbm.bound_ranges())
  missing = []
  for binding in bindings.values():
    live = hbm.get_handle(binding.name)
    if live is None:
      missing.append(binding)
    elif (live.base_address, live.size_bytes, hbm.permissions(binding.name)) != (
      binding.base_iova,
      binding.size_bytes,
      binding.permissions,
    ):
      raise ValueError(f"incompatible launch binding {binding.name}")
  source_ranges = []
  for slot, loaded in packages.items():
    (kernel,) = loaded.view["binding_manifest"]["kernels"]
    name = f"program:{kernel['linked_slot']['program_id']}:{kernel['kernel_version']}"
    if name in bindings or hbm.get_handle(name) is not None:
      raise ValueError(f"program source binding already in use: {name}")
    source_ranges.append((scenario["slots"][slot]["hbm_iova"], len(loaded.text)))
  for base, length in [*((b.base_iova, b.size_bytes) for b in missing), *source_ranges]:
    end = base + length
    if type(base) is not int or base < hbm.base_iova or end > hbm.base_iova + hbm.size_bytes:
      raise ValueError("program prepare exceeds HBM address range")
    if any(base < high and low < end for low, high in ranges):
      raise ValueError("overlapping program prepare HBM binding")
    ranges.append((base, end))
  service = ProgramService(hbm=hbm)
  for binding in missing:
    hbm.bind_external(binding)
  images = {}
  for slot, loaded in packages.items():
    (kernel,) = loaded.view["binding_manifest"]["kernels"]
    image = ProgramImage(
      kernel["linked_slot"]["program_id"],
      kernel["kernel_version"],
      bytes.fromhex(loaded.package_hash),
      0,
      loaded.text,
    )
    service.register_image(image, hbm_iova=scenario["slots"][slot]["hbm_iova"])
    images[slot] = image
  assert hbm.byte_store is not None
  if owns_hbm:
    hbm.byte_store.validate_seed_coverage()
  return PreparedPrograms(
    scenario, packages, images, hbm, service, ResidencyStorage.from_slot_freeze(freeze)
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scenario", required=True, type=Path)
  parser.add_argument("--package-root", required=True, type=Path)
  parser.add_argument("--config", required=True, type=Path)
  args = parser.parse_args()
  run = prepare_programs(args.scenario, args.package_root, config_path=args.config)
  print(
    json.dumps(
      {
        slot: {
          "package_hash": im.package_hash.hex(),
          "text_bytes": len(im.text),
          "text_sha256": hashlib.sha256(im.text).hexdigest(),
          "hbm_iova": run.scenario["slots"][slot]["hbm_iova"],
        }
        for slot, im in run.images.items()
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
