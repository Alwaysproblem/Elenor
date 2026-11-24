"""cn package: build -> resource recheck -> seal -> reload (C037 / W5-01).

The chain follows the contract's build order (AUTHORED, COMPILED, MEASURED, RECHECKED, SEALED, LOADED).
A failure at any step returns to AUTHORED with a named reason and writes no package; nothing is patched in
the Loader. Bootstrap selftests reuse the measured C018 image. The scalar control binding links two
fixed-address images, verifies both, and publishes their slot freeze only after both loads succeed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
H = LIB.parents[1]
REPO = H.parents[1]
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(REPO))

import measure_kernel as mk  # noqa: E402

from pipeline_validator.package import (  # noqa: E402
  PackageError,
  encode,
  load_file,
  seal,
  verify_image,
)
from pipeline_validator.package.slot_freeze import freeze_digest, verify_slot_freeze  # noqa: E402

STATES = ("AUTHORED", "COMPILED", "MEASURED", "RECHECKED", "SEALED", "LOADED")


class BuildError(Exception):
  """A build refusal: ``state`` is the step that failed; the build then returns to AUTHORED."""

  def __init__(self, state: str, code: str, detail: str) -> None:
    super().__init__(f"{state}: {code}: {detail}")
    self.state = state
    self.code = code
    self.detail = detail


def _sha(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def read_binding(path: Path) -> dict[str, Any]:
  doc = yaml.safe_load(path.read_text())
  if not isinstance(doc, dict) or doc.get("schema") != "cn_binding/v1":
    raise BuildError("AUTHORED", "BAD_BINDING", f"{path}: schema must be cn_binding/v1")
  for key in ("kernel", "layout", "manifest_kernel", "numeric_profile_id", "target"):
    if key not in doc:
      raise BuildError("AUTHORED", "MISSING_FIELD", f"{path}: {key}")
  return doc


def control_plane(kernel: str) -> tuple[dict[str, Any], str, tuple[Any, Any, dict[str, Any]]]:
  """Compile the control/transfer program and return its independent late-bound launch environment."""
  from pipeline_validator.compiled_program import artifact_digest, serialize_compiled_program
  from pipeline_validator.compiler import compile_program
  from pipeline_validator.config import HardwareConfig, SimConfig
  from pipeline_validator.execution_ir import GlobalBinding
  from pipeline_validator.workload_ir import parse_workload_ir
  from pipeline_validator.workloads import PowWorkload

  hw = HardwareConfig()
  sim = SimConfig(fidelity="runtime")
  if kernel == "scalar_ctl_v1":
    module = parse_workload_ir((H / "kernels/library/scalar_ctl_v1.mlir").read_text())
    bindings = {"X": GlobalBinding("X", 0x100000, 32, "r"), "Y": GlobalBinding("Y", 0x200000, 32, "w")}
  elif kernel == "bootstrap":
    module = PowWorkload(num_group_chunks=1, hw=hw).module
    bindings = {"Y": GlobalBinding("Y", 0x100000, 131072, "rw")}
  else:
    raise BuildError("AUTHORED", "UNKNOWN_KERNEL", kernel)
  program = compile_program(module, hw, sim, workload_info=None)
  fields = {
    "schema_version": program.schema_version,
    "compiler_abi": program.compiler_abi,
    "artifact_hash": artifact_digest(program),
    "source_hash": program.source_hash,
  }
  return fields, serialize_compiled_program(program), (hw, sim, bindings)


UPSTREAM_PARAMS = H / "configs" / "params.d" / "upstream_coralnpu.json"


def independent_target(binding_target: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
  """The target the Loader checks against: rvv_vlen from the pinned upstream parameter record and
  context_count from the config, never from the binding that authored the package. Byte widths have no
  independent source in either file, so they are taken from the binding."""
  params = {p["name"]: p["value"] for p in json.loads(UPSTREAM_PARAMS.read_text())["params"]}
  return {
    **binding_target,
    "rvv_vlen": params["rvv_vlen_bits"],
    "context_count": config["constraints"]["context_count"],
  }


def build(binding_path: Path, config_path: Path, out: Path) -> dict[str, Any]:
  binding = read_binding(binding_path)
  config = yaml.safe_load(config_path.read_text())
  if config.get("kind") != "cs1_target":
    raise BuildError("AUTHORED", "BAD_CONFIG", f"{config_path}: a cs1_target config is required")
  if binding["target"].get("context_count") != config["constraints"]["context_count"]:
    raise BuildError(
      "AUTHORED",
      "TARGET_MISMATCH",
      f"binding context_count {binding['target'].get('context_count')} != "
      f"config {config['constraints']['context_count']}",
    )
  out.mkdir(parents=True, exist_ok=True)

  control = control_plane(binding["kernel"])
  if binding["kernel"] == "scalar_ctl_v1":
    return _build_scalar(binding, config, out, control)
  # COMPILED + MEASURED: the real ELF and the real resource record.
  try:
    record = mk.measure(out / "measure")
  except mk.MeasureError as exc:
    raise BuildError("MEASURED", exc.code, str(exc)) from exc
  layout = next(x for x in record["candidate_layouts"] if x["name"] == binding["layout"])
  elf_path = out / "measure" / binding["layout"] / "bootstrap.elf"
  report, _ = _seal_image(binding, config, out, record, layout, elf_path, control)
  return report


def _build_scalar(binding: dict, config: dict, out: Path, control: tuple) -> dict:
  import scalar_kernel_build as scalar

  if binding["layout"] != ["A", "B"] or "linked_slot" in binding["manifest_kernel"]:
    raise BuildError("AUTHORED", "BAD_LAYOUT", "select layouts [A,B]; linked_slot is build-derived")
  # Do not mix two attempts: the freeze is the publication point for C039/C040.
  if any(out.iterdir()):
    raise BuildError("AUTHORED", "OUTPUT_NOT_EMPTY", str(out))
  with TemporaryDirectory(dir=out) as scratch:
    stage = Path(scratch)
    try:
      images = scalar.measure(binding, config, stage)
    except mk.MeasureError as exc:
      raise BuildError("RECHECKED", exc.code, str(exc)) from exc
    reports = {}
    loaded_slots = {}
    freeze_path = stage / "slot_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    rows = {row["name"]: row for row in freeze["slots"]}
    commitment = freeze_digest(freeze)
    for record, layout, elf_path in images:
      slot = layout["name"]
      slot_dir = stage / slot
      slot_dir.mkdir(exist_ok=True)
      linked = {
        key: rows[slot][key]
        for key in (
          "program_id",
          "entry",
          "text_base",
          "text_bytes",
          "data_base",
          "data_bytes",
          "image_sha256",
        )
      }
      entry = {
        **binding["manifest_kernel"],
        "linked_slot": {**linked, "slot": slot, "freeze_sha256": commitment},
      }
      reports[slot], loaded_slots[slot] = _seal_image(
        {**binding, "manifest_kernel": entry},
        config,
        slot_dir,
        record,
        layout,
        elf_path,
        control,
        published=out / slot,
        slot_freeze=freeze,
      )
    if set(reports) != {"A", "B"}:
      raise BuildError("RECHECKED", "MISSING_SLOT", str(sorted(reports)))
    freeze["packages"] = {
      slot: {
        "path": f"{slot}/package.json",
        "program_id": rows[slot]["program_id"],
        "version": binding["manifest_kernel"]["kernel_version"],
        "package_hash": reports[slot]["package_hash"],
        "image_sha256": reports[slot]["image_sha256"],
      }
      for slot in ("A", "B")
    }
    verify_slot_freeze(freeze, loaded_slots)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    for product in stage.iterdir():
      if product != freeze_path:
        product.rename(out / product.name)
    freeze_path.rename(out / freeze_path.name)
  return {"packages": freeze["packages"], "reload_identical": True}


def _seal_image(
  binding: dict,
  config: dict,
  out: Path,
  record: dict,
  layout: dict,
  elf_path: Path,
  control: tuple,
  published: Path | None = None,
  slot_freeze: dict | None = None,
):
  published = out if published is None else published
  elf = elf_path.read_bytes()
  image_path = out / "image.elf"
  image_path.write_bytes(elf)

  # RECHECKED: capacity against the target the config names, before a hash exists.
  c = config["constraints"]
  rm = record["resource_manifest"]
  if layout["text_bytes"] > c["itcm_bytes"]:
    raise BuildError("RECHECKED", "TEXT_EXCEEDS_CAPACITY", f"{layout['text_bytes']} > {c['itcm_bytes']}")
  if rm["static_data_bytes"] + rm["stack_bytes"] + rm["scratch_bytes"] > c["dtcm_bytes"]:
    raise BuildError("RECHECKED", "DATA_EXCEEDS_CAPACITY", "static data + stack + scratch exceeds DTCM")

  control_fields, compiled_text, (hw, sim, bindings) = control
  compiled_path = out / "compiled_program.json"
  compiled_path.write_text(compiled_text)
  package = {
    "package_schema": "cn_package/v1",
    "control_plane": control_fields,
    "image": {
      "bytes_sha256": _sha(elf),
      "text_bytes": layout["text_bytes"],
      "entry": record["entry"],
      "link_layout": {
        "text_base": layout["text_base"],
        "text_bytes": layout["text_bytes"],
        "data_base": layout["data_base"],
        "data_bytes": layout["data_bytes"],
      },
      "provenance": {
        "builder": json.loads((H / "env/image.lock").read_text())["tag"],
        "toolchain_lock_sha256": record["provenance"]["toolchain_lock_sha256"],
        "source_sha256": _sha(json.dumps(record["provenance"]["source_sha256"], sort_keys=True).encode()),
      },
    },
    "binding_manifest": {"kernels": [binding["manifest_kernel"]]},
    "resource_manifest": {
      "text_bytes": rm["text_bytes"],
      "static_data_bytes": rm["static_data_bytes"],
      "stack_bytes": rm["stack_bytes"],
      "scratch_bytes": rm["scratch_bytes"],
      "bound_source": rm["bound_source"],
    },
    "target_fingerprint": binding["target"],
    "numeric_profile_id": binding["numeric_profile_id"],
    "patch_mask": binding.get("patch_mask", []),
  }
  # SEALED: validation first; a package that breaks the contract is never hashed.
  try:
    sealed = seal(
      package,
      compiled_text=compiled_text,
      hw=hw,
      sim=sim,
      actual_bindings=bindings,
      slot_freeze=slot_freeze,
    )
  except PackageError as exc:
    raise BuildError("SEALED", exc.code, str(exc)) from exc
  package_path = out / "package.json"
  package_path.write_text(encode(sealed))

  # LOADED: re-read through the read-only Loader and require the identical sealed object.
  target = independent_target(binding["target"], config)
  try:
    loaded = load_file(
      package_path,
      compiled_path=compiled_path,
      hw=hw,
      sim=sim,
      actual_bindings=bindings,
      image_path=image_path,
      target=target,
      slot_freeze=slot_freeze,
    )
    verify_image(sealed, image_path.read_bytes())
    if loaded.to_dict() != sealed:
      raise PackageError("RELOAD_DIFFERS", "$", "the reloaded package is not the sealed package")
  except PackageError as exc:
    package_path.unlink(missing_ok=True)  # a package the Loader refuses is never left behind
    raise BuildError("LOADED", exc.code, str(exc)) from exc
  report = {
    "schema": "cn_package_report/v1",
    "states": list(STATES),
    "package": str(published / package_path.name),
    "image": str(published / image_path.name),
    "compiled_program": str(published / compiled_path.name),
    "package_hash": sealed["package_hash"],
    "image_sha256": sealed["image"]["bytes_sha256"],
    "resource_manifest": sealed["resource_manifest"],
    "layout": layout,
    "reload_identical": True,
  }
  (out / "package_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  return report, loaded


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--binding", required=True)
  parser.add_argument("--config", required=True)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  try:
    report = build(Path(args.binding), Path(args.config), Path(args.out))
  except BuildError as exc:
    print(f"package: {exc}  (build returns to AUTHORED; nothing written to a sealed path)", file=sys.stderr)
    raise SystemExit(1) from exc
  keys = (
    ("packages", "reload_identical")
    if "packages" in report
    else ("package_hash", "image_sha256", "reload_identical")
  )
  print(json.dumps({key: report[key] for key in keys}, indent=2))


if __name__ == "__main__":
  main()
