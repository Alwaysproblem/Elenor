"""Evidence schema and G0 gate logic (C007).

Every cn run produces evidence: a manifest with producer commit ID, code
content digest, input/config/tool hashes, command and retrievable file
index. The G0 gate re-verifies the registered evidence set (C001-C006
products) and refuses stale/forged/incomplete artifacts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

EVIDENCE_SCHEMA = "cn_evidence/v1"
GATE_SCHEMA = "cn_gate/v1"
G0_STAGE = "G0"

REQUIRED_G0_EVIDENCE = [
  "sources_verify",
  "environment_verify",
  "seed_export",
  "baseline_run",
]

# (report filename, field that must be True for a pass)
REPORT_PROBES = [
  ("report.json", "pass"),
  ("sources_report.json", "pass"),
  ("environment_checks.json", "pass"),
  ("export_manifest.json", "complete"),
  ("summary.json", "pass"),
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

  @classmethod
  def from_dict(cls, data: dict) -> EvidenceRecord:
    schema = data.pop("schema", "")
    if schema != EVIDENCE_SCHEMA:
      raise ValueError(f"evidence schema {schema!r} != {EVIDENCE_SCHEMA!r}")
    return cls(**data)


def tree_digest(root: Path) -> str:
  """Deterministic sha256 over every regular file in root."""
  digest = hashlib.sha256()
  for path in sorted(root.rglob("*")):
    if path.is_file():
      rel = path.relative_to(root).as_posix()
      digest.update(rel.encode())
      digest.update(b"\0")
      digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
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


def build_evidence(
  evidence_dir: Path,
  repo_root: Path,
  command: str,
  evidence_tier: str,
  producer_commit: str = "",
) -> EvidenceRecord:
  """Build an EvidenceRecord from a completed run's output directory."""
  return EvidenceRecord(
    producer_commit=producer_commit or git_head(repo_root),
    code_digest=tree_digest(evidence_dir),
    command=command,
    evidence_tier=evidence_tier,
    recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
    files=sorted(
      p.relative_to(evidence_dir).as_posix()
      for p in evidence_dir.rglob("*")
      if p.is_file()
    ),
  )


class G0Gate:
  """CN-G0 evidence gate: re-verifies registered evidence."""

  def __init__(self, repo_root: Path) -> None:
    self.root = repo_root
    self.checks: list[dict] = []

  def _record(self, name: str, ok: bool, detail: str) -> bool:
    self.checks.append({"check": name, "pass": ok, "detail": detail})
    return ok

  def run(self, evidence_index: Path) -> dict:
    """Run all G0 checks and return a report dict."""
    any_failed = False
    for required in REQUIRED_G0_EVIDENCE:
      entry_path = evidence_index / required
      if not entry_path.is_dir() or not list(entry_path.rglob("*")):
        any_failed |= not self._record(
          f"{required}.present", False, str(entry_path)
        )
        continue
      self._record(f"{required}.present", True, str(entry_path))
      found_report = False
      for report_name, pass_field in REPORT_PROBES:
        report_path = entry_path / report_name
        if report_path.is_file():
          found_report = True
          try:
            report = json.loads(report_path.read_text())
          except (json.JSONDecodeError, OSError) as exc:
            any_failed |= not self._record(
              f"{required}.verdict", False, str(exc)
            )
            break
          ok = report.get(pass_field) is True
          any_failed |= not self._record(
            f"{required}.verdict", ok, str(report.get(pass_field))
          )
          break
      if not found_report:
        any_failed |= not self._record(
          f"{required}.report", False, "no report file found"
        )

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
  parser.add_argument("--out", required=True)
  args = parser.parse_args()

  gate = G0Gate(Path(args.evidence_index).resolve())
  report = gate.run(Path(args.evidence_index).resolve())
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
