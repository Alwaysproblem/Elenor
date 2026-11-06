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

EVIDENCE_SCHEMA = "cn_evidence/v1"
GATE_SCHEMA = "cn_gate/v1"
MANIFEST_SCHEMA = "cn_evidence_manifest/v1"
PIN_SCHEMA = "cn_evidence_pin/v1"
G0_STAGE = "G0"
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
  ("bf16_baseline", "bf16_baseline_report.json", "test_result",
   ["bf16_baseline_report.json"]),
  ("offline_rebuild", "offline_rebuild.json", "pass",
   ["offline_rebuild.json", "build.log"]),
  ("devcontainer_probe", "devcontainer_report.json", "pass",
   ["devcontainer_report.json", "probes.log"]),
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
      "artifacts": {
        a: sha256_file(entry_path / a)
        for a in artifacts
        if (entry_path / a).is_file()
      },
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
      ["git", "-C", str(root), "rev-parse", "HEAD"],
      capture_output=True,
      text=True,
      check=True,
    )
    return proc.stdout.strip()
  except (subprocess.CalledProcessError, OSError):
    return "unknown"


def build_evidence_manifest(
  evidence_index: Path,
) -> dict:
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

    missing_artifacts = [
      a for a in artifacts if not (entry_path / a).is_file()
    ]
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
        verdict_ok = (
          value == expected.get("expected_verdict") and _verdict_passes(value)
        )
      except (json.JSONDecodeError, OSError) as exc:
        detail = f"parse error: {exc}"
      any_failed |= not self._record(f"{name}.verdict", verdict_ok, detail)

      bad_artifacts = []
      expected_artifacts = expected.get("artifacts", {})
      spec_artifacts = next(
        artifacts for n, _, _, artifacts in EVIDENCE_SPECS if n == name
      )
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
      by_name = {
        w["name"] + ".elf": w["elf_sha256"] for w in baseline.get("workloads", [])
      }
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
      any_failed |= not self._record(
        "offline_rebuild.matches_baseline", False, f"unreadable: {exc}"
      )

    return self._report(not any_failed)


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description="CN-G0 evidence gate")
  parser.add_argument("--evidence-index", required=True)
  parser.add_argument("--out", required=False)
  parser.add_argument("--pin", required=False, default=str(DEFAULT_PIN_PATH))
  parser.add_argument("--build-manifest", action="store_true")
  parser.add_argument("--write-pin", action="store_true")
  args = parser.parse_args()

  evidence_index = Path(args.evidence_index).resolve()
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
    print("error: --out is required unless --build-manifest/--write-pin",
          file=sys.stderr)
    raise SystemExit(2)
  gate = G0Gate(Path.cwd(), Path(args.pin))
  report = gate.run(evidence_index)
  out = Path(args.out).resolve()
  out.mkdir(parents=True, exist_ok=True)
  (out / "g0_gate_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(report, indent=2, sort_keys=True))
  if not report["pass"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
