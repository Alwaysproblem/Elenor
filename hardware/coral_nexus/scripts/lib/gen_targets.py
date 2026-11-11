"""Generate the sourced CS1 experimental targets (C019 / W0-04).

Inputs (each read back, never trusted):

* the confirmed decisions -- cs1_c1 single-Context degenerate regression, cs1_c2 two Contexts, one VME
  instance (N=1), full RetirementBuffer (design/coral_nexus_tile/plan.md; upstream default 16-entry
  RetirementBuffer);
* the C008 constraint schema bounds and every params.d source (validate_constraints);
* the C013 static memory envelope (configs/w4_memory_constraints.yaml);
* the W5 image capacities (protocol/w5_package.contract.yaml image.capacities) and residency
  ``min_coexisting``;
* the REAL C018 measurement record (out/coral_nexus/C018/kernel_measurement.json): text/data/stack bound
  and the candidate layouts. Without it generation refuses -- no fallback guess.

Outputs (rendered in memory, written atomically; a refused run writes nothing):

* configs/cs1_c1.yaml, configs/cs1_c2.yaml -- kind cs1_target with per-value provenance and a constraints
  fingerprint (sha256 over the canonical constraints map, never over a file that contains the fingerprint);
* configs/params.d/coral_cs1.json -- the CS1 parameter contribution, same numbers as the targets;
* rtl/targets.bzl -- the generated target registry keyed by the same fingerprints.

Refusal classes: MISSING_MEASUREMENT, MISSING_SOURCE (a provenance needle that does not read back),
CONFLICT (two sources disagree), ZERO_WIDTH (C1 must still index its single Context with >= 1 bit),
ENVELOPE (the measured images do not fit capacity x slots), SCHEMA (constraint outside the C008 bounds),
FREEZE_LOOP (a target file referencing PROTOCOL_FREEZE.json; the seal is C020 and must not be referenced by
its own input).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

H = Path(__file__).resolve().parents[2]
REPO = H.parents[1]
sys.path.insert(0, str(H / "scripts/lib"))
sys.path.insert(0, str(H / "protocol"))

import validate_constraints as vc  # noqa: E402
import validate_protocol as vp  # noqa: E402

CONFIGS = H / "configs"
MEASUREMENT = H / "verification/kernel_bootstrap/kernel_measurement.json"
OUT_MEASUREMENT = REPO / "out/coral_nexus/C018/kernel_measurement.json"
OUTPUTS = {
  "cs1_c1": CONFIGS / "cs1_c1.yaml",
  "cs1_c2": CONFIGS / "cs1_c2.yaml",
  "params": CONFIGS / "params.d/coral_cs1.json",
  "targets": H / "rtl/targets.bzl",
}

# The confirmed decisions, each with a needle the generator reads back from the decision document.
DECISIONS: dict[str, dict[str, Any]] = {
  "cs1_c1": {
    "context_count": 1,
    "needle": ("design/coral_nexus_tile/plan.md", "cs1_c1"),
    "why": "cs1_c1 is the single-Context degenerate functional regression",
  },
  "cs1_c2": {
    "context_count": 2,
    "needle": ("design/coral_nexus_tile/plan.md", "cs1_c2"),
    "why": "cs1_c2 is the two-Context target all P1/PN performance comparisons share",
  },
  "vme_instances": {
    "value": 1,
    "needle": ("design/coral_nexus_tile/plan.md", "VME N=1"),
    "why": "N=1 is the confirmed first cut; N>1 work does not auto-start",
  },
  "retirement_buffer_mode": {
    "value": "full",
    "needle": ("hardware/coral_nexus/configs/p0_coral.yaml", "full_retirement_buffer: true"),
    "why": "the confirmed full-RetirementBuffer decision (upstream default is the 16-entry buffer)",
  },
  "retirement_buffer_entries": {
    "value": 16,
    "needle": (
      "hardware/coral_nexus/upstream/coralnpu/hdl/chisel/src/coralnpu/Parameters.scala",
      "val retirementBufferSize          = 16",
    ),
    "why": "the pinned upstream default the full mode keeps",
  },
}


class TargetError(Exception):
  def __init__(self, code: str, detail: str) -> None:
    super().__init__(f"{code}: {detail}")
    self.code = code
    self.detail = detail


def _read_back(rel: str, needle: str) -> None:
  for base in (REPO, H):
    path = base / rel
    if path.is_file():
      if needle not in path.read_text():
        raise TargetError("MISSING_SOURCE", f"{needle!r} not found in {rel}")
      return
  raise TargetError("MISSING_SOURCE", f"{rel} does not exist")


def _canonical(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fingerprint(constraints: dict[str, Any]) -> str:
  return hashlib.sha256(_canonical(constraints)).hexdigest()


def _check_pair(c1: dict[str, Any], c2: dict[str, Any]) -> None:
  """C1/C2 degradation direction and the same-RTL rule (everything but context_count equal)."""
  if c1["context_count"] >= c2["context_count"]:
    raise TargetError("CONFLICT", "C1 must have fewer Contexts than C2")
  for key in set(c1) | set(c2):
    if key != "context_count" and c1.get(key) != c2.get(key):
      raise TargetError("CONFLICT", f"same-RTL rule broken: {key} differs between C1 and C2")


def _reject_freeze_references(rendered: dict[str, str]) -> None:
  """A generated target may not reference the seal: the seal covers its inputs (no hash loop)."""
  for text in rendered.values():
    if "PROTOCOL_FREEZE" in text:
      raise TargetError("FREEZE_LOOP", "a generated target references the seal; the seal covers its inputs")


def build(measurement_path: Path = MEASUREMENT, contract_path: Path | None = None) -> dict[Path, str]:
  """Validate every input and render all outputs; raises TargetError without writing."""
  for decision in DECISIONS.values():
    rel, needle = decision["needle"]
    _read_back(rel, needle)

  if not measurement_path.is_file():
    raise TargetError("MISSING_MEASUREMENT", f"{measurement_path} (run C018 first)")
  measured = json.loads(measurement_path.read_text())
  rm = measured["resource_manifest"]
  if rm["bound_source"] != "static_analysis":
    raise TargetError("BAD_MEASUREMENT", "bound_source is not static_analysis")
  layouts = measured["candidate_layouts"]
  if len(layouts) < 2:
    raise TargetError("BAD_MEASUREMENT", "fewer than two candidate layouts")

  contract = yaml.safe_load((contract_path or H / "protocol/w5_package.contract.yaml").read_text())
  caps = contract["image"]["capacities"]
  itcm, dtcm = int(caps["itcm_bytes"]["value"]), int(caps["dtcm_bytes"]["value"])
  # the capacities the measurement ran against must be the contract's; a stale record is refused
  recorded = measured.get("capacities") or {}
  if (recorded.get("itcm_bytes"), recorded.get("dtcm_bytes")) != (itcm, dtcm):
    raise TargetError("CONFLICT", "the measurement record carries different ITCM/DTCM capacities")
  min_coex = int(contract["residency"]["slots"]["min_coexisting"])
  slots = len(layouts)
  if slots < min_coex:
    raise TargetError("ENVELOPE", f"{slots} candidate slots < residency min_coexisting {min_coex}")

  # the C013 envelope feeds user_spm_per_bank; conflicts between sources refuse generation
  params, param_errors = vc.load_params(CONFIGS / "params.d")
  if param_errors:
    raise TargetError("CONFLICT", "; ".join(param_errors))
  bank = params.get("l1_spm_bytes_per_bank_mode0")
  reserved = params.get("l1_system_reserved_spm_per_bank")
  if not bank or not reserved:
    raise TargetError("MISSING_SOURCE", "the C013 bank/reserved parameters are absent from params.d")
  user_spm = int(bank["value"]) - int(reserved["value"])

  envelope_ok = all(
    x["text_base"] + x["text_bytes"] <= itcm and x["data_base"] + x["data_bytes"] + x["stack_bytes"] <= dtcm
    for x in layouts
  )
  if not envelope_ok:
    raise TargetError("ENVELOPE", "a measured candidate layout does not fit the W5 capacity")
  for i, a in enumerate(layouts):
    for b in layouts[i + 1 :]:
      clash = (
        a["text_base"] < b["text_base"] + b["text_bytes"]
        and b["text_base"] < a["text_base"] + a["text_bytes"]
      )
      clash = clash or (
        a["data_base"] < b["data_base"] + b["data_bytes"]
        and b["data_base"] < a["data_base"] + a["data_bytes"]
      )
      if clash:
        raise TargetError("ENVELOPE", "candidate layouts overlap; they cannot be program slots")

  targets: dict[str, dict[str, Any]] = {}
  for name in ("cs1_c1", "cs1_c2"):
    context_count = int(DECISIONS[name]["context_count"])
    constraints = {
      "context_count": context_count,
      "tile_count": 1,
      "vme_instances": int(DECISIONS["vme_instances"]["value"]),
      "user_spm_per_bank": user_spm,
      "system_reserved_spm_per_bank": int(reserved["value"]),
      "retirement_buffer_mode": str(DECISIONS["retirement_buffer_mode"]["value"]),
      "retirement_buffer_entries": int(DECISIONS["retirement_buffer_entries"]["value"]),
      "program_slots": slots,
      "itcm_bytes": itcm,
      "dtcm_bytes": dtcm,
    }
    # C1 must still index its single Context: the width is never zero (C009 rule, enforced by
    # derive_index_width itself and pinned here so a drift is visible in the artifact)
    width = vp.derive_index_width(f"{name}.ctx_slot", context_count)
    targets[name] = {
      "schema": "cn_target/v1",
      "kind": "cs1_target",
      "name": name,
      "generated_by": "hardware/coral_nexus/scripts/lib/gen_targets.py",
      "constraints": constraints,
      "index_widths": {"ctx_slot_bits": width.bits},
      "envelope": {
        "measured": {
          "text_bytes": rm["text_bytes"],
          "static_data_bytes": rm["static_data_bytes"],
          "stack_bytes": rm["stack_bytes"],
          "scratch_bytes": rm["scratch_bytes"],
        },
        "record": "hardware/coral_nexus/verification/kernel_bootstrap/kernel_measurement.json",
        "elf_sha256_layout_A": measured["provenance"]["elf_sha256_layout_A"],
        "toolchain_lock_sha256": measured["provenance"]["toolchain_lock_sha256"],
      },
      "support": {
        "numeric": "experimental (C011 profile; W7 oracle pending)",
        "kernels": ["bootstrap (measure-only, C018)"],
        "executable": False,
        "note": "no full Tile top exists yet; cn build only reaches prepare-only (C040 provides the top)",
      },
      "provenance": {
        "decision": f"{DECISIONS[name]['needle'][0]} :: {DECISIONS[name]['needle'][1]}",
        "vme_instances": "design/coral_nexus_tile/plan.md :: VME N=1",
        "retirement_buffer_mode": DECISIONS["retirement_buffer_mode"]["needle"],
        "retirement_buffer_entries": DECISIONS["retirement_buffer_entries"]["needle"],
        "user_spm_per_bank": "l1_spm_bytes_per_bank_mode0 - l1_system_reserved_spm_per_bank (params.d)",
        "system_reserved_spm_per_bank": "params.d validator_model l1_system_reserved_spm_per_bank",
        "itcm_bytes": "rules/coralnpu_v2.bzl:224 (read via w5_package.contract.yaml image.capacities)",
        "dtcm_bytes": "rules/coralnpu_v2.bzl:225 (read via w5_package.contract.yaml image.capacities)",
        "w4_constraints": "configs/w4_memory_constraints.yaml",
      },
      "fingerprints": {"constraints_sha256": _fingerprint(constraints)},
    }
  # validate each target through the C008 consumer; C1 is validated WITH C2 as its peer so the
  # C008 C1/C2 degradation rule is part of generation, not an afterthought
  for name, target in targets.items():
    tmp = Path(tempfile.mkdtemp()) / f"{name}.json"
    tmp.write_text(json.dumps(target))
    report = vc.validate(tmp, extra_targets={"cs1_c2": targets["cs1_c2"]} if name == "cs1_c1" else {})
    if report["errors"]:
      raise TargetError("SCHEMA", f"{name}: {report['errors']}")

  # C1/C2 degradation direction and everything-but-context equality (C008 rule, re-checked here)
  _check_pair(targets["cs1_c1"]["constraints"], targets["cs1_c2"]["constraints"])

  params_doc = {
    "schema": "cn_params/v1",
    "source": {
      "repo": "coral_nexus",
      "kind": "cs1_target",
      "note": "generated with the CS1 targets (C019)",
    },
    "params": [
      {
        "name": f"{name}.context_count",
        "value": t["constraints"]["context_count"],
        "unit": "count",
        "provenance": {"file": f"hardware/coral_nexus/configs/{name}.yaml"},
      }
      for name, t in sorted(targets.items())
    ]
    + [
      {
        "name": "cs1.program_slots",
        "value": slots,
        "unit": "count",
        "provenance": {"file": "hardware/coral_nexus/configs/cs1_c2.yaml"},
      }
    ],
  }

  bzl = [
    "# GENERATED by hardware/coral_nexus/scripts/lib/gen_targets.py (C019); do not edit.",
    "# Target registry keyed by the constraints fingerprint of each cs1 target. The fingerprints come",
    "# from configs/cs1_c{1,2}.yaml and never from a file that contains them (no hash loop).",
    "",
    "CS1_TARGETS = {",
  ]
  for name, t in sorted(targets.items()):
    fp = t["fingerprints"]["constraints_sha256"]
    bzl += [
      f'    "{name}": {{',
      f'        "constraints_sha256": "{fp}",',
      f'        "context_count": {t["constraints"]["context_count"]},',
      f'        "tile_count": {t["constraints"]["tile_count"]},',
      f'        "vme_instances": {t["constraints"]["vme_instances"]},',
      f'        "program_slots": {t["constraints"]["program_slots"]},',
      f'        "retirement_buffer_mode": "{t["constraints"]["retirement_buffer_mode"]}",',
      "    },",
    ]
  bzl += [
    "}",
    "",
    "def cs1_target_fingerprint(name):",
    '    """Return the constraints fingerprint of one registered cs1 target."""',
    "    if name not in CS1_TARGETS:",
    '        fail("unknown cs1 target: %s" % name)',
    '    return CS1_TARGETS[name]["constraints_sha256"]',
    "",
  ]

  rendered: dict[str, str] = {
    "cs1_c1": yaml.safe_dump(targets["cs1_c1"], sort_keys=False, width=1000),
    "cs1_c2": yaml.safe_dump(targets["cs1_c2"], sort_keys=False, width=1000),
    "params": json.dumps(params_doc, indent=2) + "\n",
    "targets": "\n".join(bzl),
  }
  _reject_freeze_references(rendered)
  return {OUTPUTS[key]: value for key, value in rendered.items()}


def _atomic_write(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
  try:
    with os.fdopen(fd, "w") as handle:
      handle.write(text)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
  except BaseException:
    Path(tmp).unlink(missing_ok=True)
    raise


def generate(measurement_path: Path = MEASUREMENT, out: dict[Path, Path] | None = None) -> dict[Path, str]:
  rendered = build(measurement_path)
  for path, text in rendered.items():
    _atomic_write((out or {}).get(path, path), text)
  return rendered


def check(measurement_path: Path = MEASUREMENT) -> list[str]:
  """Committed outputs must equal a fresh render; stale or hand-edited files are errors."""
  try:
    rendered = build(measurement_path)
  except TargetError as exc:
    return [str(exc)]
  errors = []
  for path, text in rendered.items():
    if not path.is_file():
      errors.append(f"{path.name}: generated file is missing")
    elif path.read_text() != text:
      errors.append(f"{path.name}: differs from a fresh generation (hand-edited or stale)")
  return errors


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--measurement", default=str(MEASUREMENT))
  parser.add_argument("--check", action="store_true")
  args = parser.parse_args()
  try:
    if args.check:
      errors = check(Path(args.measurement))
      if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
      print("gen_targets: ok")
    else:
      for path in generate(Path(args.measurement)):
        print(f"wrote {path}")
  except TargetError as exc:
    print(f"gen_targets: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc


if __name__ == "__main__":
  main()
