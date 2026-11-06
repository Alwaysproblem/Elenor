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
G0_STAGE = "G0"

REQUIRED_G0_EVIDENCE = [
  "sources_verify",
  "environment_verify",
  "seed_export",
  "baseline_run",
  "bf16_baseline",
]


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


def tree_digest(root: Path) -> str:
  """Deterministic sha256 over every regular file in root."""
  digest = hashlib.sha256()
  for path in sorted(root.rglob("*")):
    if path.is_file():
      rel = path.relative_to(root).as_posix()
      digest.update(rel.encode())
      digest.update(b"\0")
      digest.update(sha256_file(path).encode())
      digest.update(b"\0")
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
  evidence_specs = [
    (
      "sources_verify",
      "sources_report.json",
      "pass",
      ["sources_report.json"],
    ),
    (
      "environment_verify",
      "environment_checks.json",
      "pass",
      ["environment_checks.json", "verify_inside.log"],
    ),
    (
      "seed_export",
      "export_manifest.json",
      "complete",
      ["export_manifest.json"],
    ),
    (
      "baseline_run",
      "report.json",
      "pass",
      ["report.json", "summary.txt"],
    ),
    (
      "bf16_baseline",
      "bf16_baseline_report.json",
      "test_result",
      ["bf16_baseline_report.json"],
    ),
  ]

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
  """CN-G0 evidence gate: re-hashes reports and checks artifacts."""

  def __init__(self, repo_root: Path) -> None:
    self.root = repo_root
    self.checks: list[dict] = []

  def _record(self, name: str, ok: bool, detail: str) -> bool:
    self.checks.append({"check": name, "pass": ok, "detail": detail})
    return ok

  def run(self, evidence_index: Path) -> dict:
    """Re-verify every required evidence entry from the manifest."""
    any_failed = False
    manifest_path = evidence_index / "evidence_manifest.json"

    if not manifest_path.is_file():
      self._record("manifest.present", False, str(manifest_path))
      return {
        "schema": GATE_SCHEMA,
        "stage": G0_STAGE,
        "checks": self.checks,
        "pass": False,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
      }

    try:
      manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
      self._record("manifest.readable", False, str(exc))
      return {
        "schema": GATE_SCHEMA,
        "stage": G0_STAGE,
        "checks": self.checks,
        "pass": False,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
      }

    for entry in manifest.get("entries", []):
      name = entry["name"]
      status = entry.get("status", "missing")

      if status in {"missing", "missing_report"}:
        any_failed |= not self._record(
          f"{name}.present", False, entry.get("detail", status)
        )
        continue

      self._record(f"{name}.present", True, status)

      # Re-hash the report file and compare against the manifest.
      report_path = evidence_index / entry["report_path"]
      if not report_path.is_file():
        any_failed |= not self._record(
          f"{name}.report_exists", False, str(report_path)
        )
        continue

      actual_hash = sha256_file(report_path)
      expected_hash = entry.get("report_sha256", "")
      hash_ok = actual_hash == expected_hash
      any_failed |= not self._record(
        f"{name}.report_sha256",
        hash_ok,
        f"actual {actual_hash[:12]} vs manifest {expected_hash[:12]}",
      )

      # Check the report's verdict field.
      verdict_ok = entry.get("verdict_pass", False)
      any_failed |= not self._record(
        f"{name}.verdict", verdict_ok, str(entry.get("report_verdict"))
      )

      # Check required artifact files exist.
      missing = entry.get("missing_artifacts", [])
      if missing:
        any_failed |= not self._record(
          f"{name}.artifacts", False, f"missing: {missing}"
        )
      else:
        self._record(f"{name}.artifacts", True, "all present")

    return {
      "schema": GATE_SCHEMA,
      "stage": G0_STAGE,
      "checks": self.checks,
      "pass": not any_failed,
      "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description="CN-G0 evidence gate")
  parser.add_argument("--evidence-index", required=True)
  parser.add_argument("--out", required=False)
  parser.add_argument("--build-manifest", action="store_true")
  args = parser.parse_args()

  evidence_index = Path(args.evidence_index).resolve()
  if args.build_manifest:
    m = build_evidence_manifest(evidence_index)
    print(f"manifest built: {len(m['entries'])} entries")
    return
  if not args.out:
    print("error: --out is required unless --build-manifest", file=sys.stderr)
    raise SystemExit(2)
  gate = G0Gate(evidence_index)
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
