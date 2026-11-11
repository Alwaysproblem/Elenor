"""Evidence schema and G0 gate logic (C007).

Every cn run produces evidence: a manifest with producer commit ID, code
content digest, input/config/tool hashes, command and retrievable file
index. The G0 gate re-verifies the registered evidence set by RE-HASHING
report files and comparing against the manifest recorded at setup time,
then checking that required artifact files exist. It refuses stale, forged
or incomplete artifacts — a directory containing only {"pass": true} with
no real artifacts cannot pass.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA = "cn_evidence/v1"
GATE_SCHEMA = "cn_gate/v1"
MANIFEST_SCHEMA = "cn_evidence_manifest/v1"
PIN_SCHEMA = "cn_evidence_pin/v1"
G0_STAGE = "G0"
G1_STAGE = "G1"
G1_SPEC_PATH = Path(__file__).with_name("gates") / "G1.yaml"
# The required G1 check set lives HERE (the yaml mirrors it for readers); a drifted spec is itself a
# failure class. (check name -> failure exit code)
REQUIRED_G1_CHECKS = {
  "protocol.contracts": 11,
  "wire.codec_generation": 12,
  "csr.bindings": 13,
  "budgets.closed": 14,
  "resources.measured": 15,
  "targets.generated": 16,
  "seal.joint": 17,
}
G1_SPEC_DRIFT_CODE = 18
DEFAULT_PIN_PATH = Path(__file__).with_name("expected_evidence.json")

# (evidence_name, report_filename, verdict_field, required_artifacts)
EVIDENCE_SPECS = [
  ("sources_verify", "sources_report.json", "pass", ["sources_report.json"]),
  (
    "environment_verify",
    "environment_checks.json",
    "pass",
    ["environment_checks.json", "verify_inside.log"],
  ),
  ("seed_export", "export_manifest.json", "complete", ["export_manifest.json"]),
  ("baseline_run", "report.json", "pass", ["report.json", "summary.txt"]),
  (
    "bf16_baseline",
    "bf16_kat_record.json",
    "replay_identical",
    ["bf16_kat_record.json", "bf16_kat_run_a.json", "bf16_kat_run_b.json"],
  ),
  ("offline_rebuild", "offline_rebuild.json", "pass", ["offline_rebuild.json", "build.log"]),
  ("devcontainer_probe", "devcontainer_report.json", "pass", ["devcontainer_report.json", "probes.log"]),
]

REQUIRED_G0_EVIDENCE = [spec[0] for spec in EVIDENCE_SPECS]


def _verdict_passes(value: object) -> bool:
  return value is True or value == "PASSED"


def build_pin(evidence_index: Path) -> dict:
  """Freeze the sha256 of every required report and artifact.

  The pin is the gate's root of trust: it is committed and reviewed, so a
  self-consistent forged directory cannot pass. Re-running legwork
  regenerates evidence and therefore requires a reviewed pin update.
  """
  entries: dict[str, dict] = {}
  for name, report_name, verdict_field, artifacts in EVIDENCE_SPECS:
    entry_path = evidence_index / name
    report_path = entry_path / report_name
    if not report_path.is_file():
      raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text())
    entries[name] = {
      "report": f"{name}/{report_name}",
      "report_sha256": sha256_file(report_path),
      "verdict_field": verdict_field,
      "expected_verdict": report.get(verdict_field),
      "artifacts": {a: sha256_file(entry_path / a) for a in artifacts if (entry_path / a).is_file()},
    }
  return {
    "schema": PIN_SCHEMA,
    "stage": G0_STAGE,
    "producer_commit": git_head(Path.cwd()),
    "entries": entries,
  }


@dataclass
class EvidenceRecord:
  """One piece of evidence from one cn run."""

  producer_commit: str
  code_digest: str
  input_hashes: dict[str, str] = field(default_factory=dict)
  config_hash: str = ""
  tool_hashes: dict[str, str] = field(default_factory=dict)
  command: str = ""
  files: list[str] = field(default_factory=list)
  evidence_tier: str = ""
  recorded_at: str = ""

  def to_dict(self) -> dict:
    d = asdict(self)
    d["schema"] = EVIDENCE_SCHEMA
    return d


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_head(root: Path) -> str:
  try:
    proc = subprocess.run(
      ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()
  except (subprocess.CalledProcessError, OSError):
    return "unknown"


def build_evidence_manifest(evidence_index: Path) -> dict:
  """Scan the evidence index and record real hashes + required artifacts.

  Called by gate.sh BEFORE running the gate. Each required evidence
  directory must contain its report file; the manifest records the
  report's content sha256 and lists artifact files that must exist.
  If a required evidence directory is missing or has no report, the
  manifest entry is recorded with status=missing so the gate fails.
  """
  # (evidence_name, report_filename, pass_field, required_artifacts)
  evidence_specs = EVIDENCE_SPECS

  entries = []
  for name, report_name, pass_field, artifacts in evidence_specs:
    entry_path = evidence_index / name
    report_path = entry_path / report_name
    entry: dict = {"name": name, "required_artifacts": artifacts}

    if not entry_path.is_dir():
      entry["status"] = "missing"
      entry["detail"] = f"directory {entry_path} not found"
      entries.append(entry)
      continue

    if not report_path.is_file():
      entry["status"] = "missing_report"
      entry["detail"] = f"{report_name} not found in {entry_path}"
      entries.append(entry)
      continue

    entry["report_path"] = str(report_path.relative_to(evidence_index))
    entry["report_sha256"] = sha256_file(report_path)

    try:
      report = json.loads(report_path.read_text())
      entry["report_verdict"] = report.get(pass_field)
      verdict = report.get(pass_field)
      entry["verdict_pass"] = verdict is True or verdict == "PASSED"
    except (json.JSONDecodeError, OSError):
      entry["report_verdict"] = None
      entry["verdict_pass"] = False

    missing_artifacts = [a for a in artifacts if not (entry_path / a).is_file()]
    entry["missing_artifacts"] = missing_artifacts
    entry["status"] = "present" if not missing_artifacts else "incomplete"

    entries.append(entry)

  manifest = {
    "schema": MANIFEST_SCHEMA,
    "stage": G0_STAGE,
    "entries": entries,
    "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
  }
  (evidence_index / "evidence_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
  )
  return manifest


class G0Gate:
  """CN-G0 evidence gate: verifies evidence against a committed pin.

  The pin (expected_evidence.json) is the root of trust. Re-hashing
  against a manifest rebuilt from the same directory proves nothing, so
  the gate fails closed when the pin is absent, unreadable, or does not
  cover every required evidence entry. A directory containing only
  {"pass": true} cannot pass: its report hash never matches the pin.
  """

  def __init__(self, repo_root: Path, pin_path: Path | None = None) -> None:
    self.root = repo_root
    self.pin_path = pin_path or DEFAULT_PIN_PATH
    self.checks: list[dict] = []

  def _record(self, name: str, ok: bool, detail: str) -> bool:
    self.checks.append({"check": name, "pass": ok, "detail": detail})
    return ok

  def _report(self, passed: bool) -> dict:
    return {
      "schema": GATE_SCHEMA,
      "stage": G0_STAGE,
      "checks": self.checks,
      "pass": passed,
      "pin": str(self.pin_path),
      "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

  def run(self, evidence_index: Path) -> dict:
    """Verify every pinned report and artifact in the evidence index."""
    try:
      pin = json.loads(self.pin_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
      self._record("pin.present", False, f"{self.pin_path}: {exc}")
      return self._report(False)

    if pin.get("schema") != PIN_SCHEMA:
      self._record("pin.schema", False, str(pin.get("schema")))
      return self._report(False)
    self._record("pin.present", True, str(self.pin_path))

    pinned = pin.get("entries", {})
    uncovered = [n for n in REQUIRED_G0_EVIDENCE if n not in pinned]
    any_failed = not self._record(
      "pin.coverage", not uncovered, f"missing {uncovered}" if uncovered else "complete"
    )

    for name in REQUIRED_G0_EVIDENCE:
      expected = pinned.get(name)
      if expected is None:
        any_failed |= not self._record(f"{name}.pinned", False, "absent from pin")
        continue

      report_path = evidence_index / expected["report"]
      if not report_path.is_file():
        any_failed |= not self._record(f"{name}.present", False, str(report_path))
        continue
      self._record(f"{name}.present", True, str(expected["report"]))

      actual = sha256_file(report_path)
      expected_hash = expected["report_sha256"]
      any_failed |= not self._record(
        f"{name}.report_sha256",
        actual == expected_hash,
        f"actual {actual[:12]} vs pin {expected_hash[:12]}",
      )

      verdict_ok = False
      detail = "unreadable"
      try:
        report = json.loads(report_path.read_text())
        value = report.get(expected["verdict_field"])
        detail = str(value)
        verdict_ok = value == expected.get("expected_verdict") and _verdict_passes(value)
      except (json.JSONDecodeError, OSError) as exc:
        detail = f"parse error: {exc}"
      any_failed |= not self._record(f"{name}.verdict", verdict_ok, detail)

      bad_artifacts = []
      expected_artifacts = expected.get("artifacts", {})
      spec_artifacts = next(artifacts for n, _, _, artifacts in EVIDENCE_SPECS if n == name)
      for artifact in spec_artifacts:
        artifact_path = evidence_index / name / artifact
        pinned_hash = expected_artifacts.get(artifact)
        if pinned_hash is None:
          bad_artifacts.append(f"{artifact}: not pinned")
        elif not artifact_path.is_file():
          bad_artifacts.append(f"{artifact}: missing")
        elif sha256_file(artifact_path) != pinned_hash:
          bad_artifacts.append(f"{artifact}: hash mismatch")
      any_failed |= not self._record(
        f"{name}.artifacts",
        not bad_artifacts,
        "; ".join(bad_artifacts) if bad_artifacts else "all match pin",
      )

    # Cross-evidence invariant: an offline rebuild that produces different
    # bytes than the recorded baseline did not reproduce the baseline.
    baseline_report = evidence_index / "baseline_run" / "report.json"
    offline_report = evidence_index / "offline_rebuild" / "offline_rebuild.json"
    try:
      baseline = json.loads(baseline_report.read_text())
      offline = json.loads(offline_report.read_text())
      by_name = {w["name"] + ".elf": w["elf_sha256"] for w in baseline.get("workloads", [])}
      rebuilt = offline.get("elfs", {})
      mismatched = [
        f"{elf}: {digest[:12]} != {by_name[elf][:12]}"
        for elf, digest in rebuilt.items()
        if elf in by_name and digest != by_name[elf]
      ]
      compared = [elf for elf in rebuilt if elf in by_name]
      any_failed |= not self._record(
        "offline_rebuild.matches_baseline",
        bool(compared) and not mismatched,
        "; ".join(mismatched) if mismatched else f"{len(compared)} elf(s) identical",
      )
    except (json.JSONDecodeError, OSError, KeyError) as exc:
      any_failed |= not self._record("offline_rebuild.matches_baseline", False, f"unreadable: {exc}")

    return self._report(not any_failed)


def _fragment_validators() -> list[tuple[str, Any]]:
  """The C010-C016 producer validators G1 re-runs. The set is held here, not in any YAML."""
  import derive_w4_bounds as w4
  import validate_native_owner as vno
  import validate_numeric as vnum
  import validate_package_contract as vpc
  import validate_vme_decl as vvd
  import validate_w6_contract as vw6
  import validate_w7_contract as vw7

  return [
    ("C010 native owner", vno.validate),
    ("C011 numeric", vnum.validate),
    ("C012 vme decl", vvd.validate),
    ("C013 memory", lambda: w4.validate_all(constraints_path=_w4_path())),
    ("C014 package", vpc.validate),
    ("C015 event/reset", vw6.validate),
    ("C016 trace", vw7.validate),
  ]


def _w4_path() -> Path:
  import freeze_check as fc

  return fc.W4_CONSTRAINTS


class G1Gate:
  """CN-G1 joint-contract gate (C020).

  Runs the REAL checkers (catalog validation, fresh re-render of every generated artifact, the CSR
  allocation lint, the budget derivation, the measured-resource record and the joint seal); it never
  accepts a producer's "DONE" status in place of artifact consistency. Each failure class maps to its
  own non-zero exit code; a spec/evaluator drift is a failure of its own.
  """

  def __init__(self, repo_root: Path) -> None:
    self.repo_root = repo_root
    self.results: list[dict] = []

  def _record(self, name: str, ok: bool, detail: str) -> bool:
    self.results.append({"check": name, "ok": ok, "detail": detail})
    return ok

  def _report(self, exit_code: int) -> dict:
    return {
      "schema": GATE_SCHEMA,
      "stage": G1_STAGE,
      "checks": self.results,
      "pass": exit_code == 0,
      "exit_code": exit_code,
      "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

  def run(self) -> dict:
    import yaml

    sys.path.insert(0, str(self.repo_root / "hardware/coral_nexus/protocol"))
    sys.path.insert(0, str(self.repo_root / "hardware/coral_nexus/scripts/lib"))
    sys.path.insert(0, str(self.repo_root / "hardware/coral_nexus/scripts"))
    import csr_lint  # noqa: F401  (imported for the side effect of path sanity)
    import csr_lint as cl
    import derive_w4_bounds as w4
    import derive_widths as dw
    import freeze_check as fc
    import gen_targets as gtr
    import validate_protocol as vpr

    exit_code = 0

    spec = yaml.safe_load(G1_SPEC_PATH.read_text())
    listed = list(spec.get("required_checks") or [])
    if listed != list(REQUIRED_G1_CHECKS):
      self._record(
        "gate.spec", False, f"G1.yaml lists {listed}, evaluator holds {list(REQUIRED_G1_CHECKS)}"
      )
      exit_code = G1_SPEC_DRIFT_CODE
    else:
      self._record("gate.spec", True, "spec matches the evaluator")

    catalog = vpr.validate(self.repo_root / "hardware/coral_nexus/protocol")
    # Every producer contract (C010-C016) is re-validated HERE, not trusted from its own commit: the seal
    # only hashes these files, so without this call nothing would read their anchors back at gate time.
    contract_errors = [f"protocol: {e}" for e in catalog["errors"]]
    for label, check in _fragment_validators():
      try:
        report = check()
        contract_errors += [f"{label}: {e}" for e in report.get("errors", [])]
        if report.get("valid") is not True:
          contract_errors.append(f"{label}: validator did not report valid")
      except Exception as exc:  # a crashed validator is a failed validator, never a skipped one
        contract_errors.append(f"{label}: validator raised {type(exc).__name__}: {exc}")
    if not self._record(
      "protocol.contracts",
      not contract_errors,
      "; ".join(contract_errors)[:400] or f"catalog + {len(_fragment_validators())} contracts valid",
    ):
      exit_code = exit_code or REQUIRED_G1_CHECKS["protocol.contracts"]

    drift = fc.verify_generated()
    if not self._record(
      "wire.codec_generation", not drift, "; ".join(drift)[:400] or "all artifacts fresh"
    ):
      exit_code = exit_code or REQUIRED_G1_CHECKS["wire.codec_generation"]

    csr_errors = cl.lint()
    if not self._record("csr.bindings", not csr_errors, "; ".join(csr_errors)[:400] or "bindings match"):
      exit_code = exit_code or REQUIRED_G1_CHECKS["csr.bindings"]

    data = vpr.load_all(self.repo_root / "hardware/coral_nexus/protocol")
    # widths may stay declared-underivable (fail closed, no invented width); capacity budgets may not:
    # a budget on the w4 unresolved list is open and the gate fails on it by name
    budgets_open = fc.open_budgets()
    try:
      result = dw.derive(data["budgets"], data["lifetimes"])
      # the W4 envelope is validated by its own owner (units, partition conservation, unresolved guard);
      # a budget cannot read as closed while that validator rejects the file that closes it
      w4_report = w4.validate_all(constraints_path=fc.W4_CONSTRAINTS)
      if not w4_report["valid"] and not budgets_open:
        raise dw.WidthError("w4 envelope rejected: " + "; ".join(w4_report["errors"])[:300])
      if budgets_open:
        self._record(
          "budgets.closed", False, f"open required budget, no sourced value: {', '.join(budgets_open)}"
        )
        exit_code = exit_code or REQUIRED_G1_CHECKS["budgets.closed"]
      else:
        detail = f"{len(result.derived)} derived, {len(result.unresolved)} widths declared underivable"
        self._record("budgets.closed", True, detail)
    except dw.WidthError as exc:
      self._record("budgets.closed", False, str(exc)[:400])
      exit_code = exit_code or REQUIRED_G1_CHECKS["budgets.closed"]

    measurement = (
      self.repo_root / "hardware/coral_nexus/verification/kernel_bootstrap/kernel_measurement.json"
    )
    if not measurement.is_file():
      self._record("resources.measured", False, f"{measurement} missing")
      exit_code = exit_code or REQUIRED_G1_CHECKS["resources.measured"]
    else:
      record = json.loads(measurement.read_text())
      ok = record.get("schema") == "cn_kernel_measurement/v1" and all(c["ok"] for c in record["checks"])
      ok = ok and record["resource_manifest"]["bound_source"] == "static_analysis"
      self._record(
        "resources.measured", ok, "record schema-clean, all checks OK" if ok else "record not all-OK"
      )
      if not ok:
        exit_code = exit_code or REQUIRED_G1_CHECKS["resources.measured"]

    target_errors = gtr.check()
    if not self._record(
      "targets.generated", not target_errors, "; ".join(target_errors)[:400] or "targets fresh"
    ):
      exit_code = exit_code or REQUIRED_G1_CHECKS["targets.generated"]

    seal_errors = fc.verify_seal()
    if not self._record("seal.joint", not seal_errors, "; ".join(seal_errors)[:400] or "seal verifies"):
      exit_code = exit_code or REQUIRED_G1_CHECKS["seal.joint"]

    return self._report(exit_code)


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description="CN stage gates")
  parser.add_argument("--stage", default=G0_STAGE)
  parser.add_argument("--evidence-index", required=False)
  parser.add_argument("--out", required=False)
  parser.add_argument("--pin", required=False, default=str(DEFAULT_PIN_PATH))
  parser.add_argument("--build-manifest", action="store_true")
  parser.add_argument("--write-pin", action="store_true")
  args = parser.parse_args()

  if args.stage == G1_STAGE and not args.out:
    print("error: --out is required", file=sys.stderr)
    raise SystemExit(2)
  if args.stage != G1_STAGE and not args.evidence_index:
    print("error: --evidence-index is required for this stage", file=sys.stderr)
    raise SystemExit(2)
  evidence_index = Path(args.evidence_index).resolve() if args.evidence_index else Path("/nonexistent")
  if args.write_pin:
    pin = build_pin(evidence_index)
    pin_path = Path(args.pin).resolve()
    pin_path.write_text(json.dumps(pin, indent=2, sort_keys=True) + "\n")
    print(f"pin written: {pin_path} ({len(pin['entries'])} entries)")
    return
  if args.build_manifest:
    m = build_evidence_manifest(evidence_index)
    print(f"manifest built: {len(m['entries'])} entries")
    return
  if not args.out:
    print("error: --out is required unless --build-manifest/--write-pin", file=sys.stderr)
    raise SystemExit(2)
  if args.stage == G1_STAGE:
    report = G1Gate(Path.cwd()).run()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "g1_gate_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["pass"] else report["exit_code"])

  gate = G0Gate(Path.cwd(), Path(args.pin))
  report = gate.run(evidence_index)
  out = Path(args.out).resolve()
  out.mkdir(parents=True, exist_ok=True)
  (out / "g0_gate_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  if not report["pass"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
