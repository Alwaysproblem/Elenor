"""Run-manifest writer shared by cn dispatch and bash handlers.

Every cn invocation records what ran, with what inputs, and how it ended.
Manifests never contain pass/fail judgments — those live in the reports the
handlers write; the manifest only captures the invocation envelope.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "cn_run_manifest/v1"


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


def write(
  path: Path,
  *,
  command: str,
  args: list[str],
  executor: str,
  handler: str,
  exit_code: int,
  duration_seconds: float,
  root: Path,
) -> dict:
  manifest = {
    "schema": SCHEMA,
    "command": command,
    "args": args,
    "executor": executor,
    "handler": handler,
    "exit_code": exit_code,
    "duration_seconds": round(duration_seconds, 3),
    "git_head": git_head(root),
    "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  return manifest


def main() -> None:
  if len(sys.argv) != 3:
    print("usage: run_manifest.py <fields-json> <output-path>", file=sys.stderr)
    raise SystemExit(2)
  fields = json.loads(Path(sys.argv[1]).read_text())
  write(
    Path(sys.argv[2]),
    command=fields["command"],
    args=fields["args"],
    executor=fields["executor"],
    handler=fields["handler"],
    exit_code=fields["exit_code"],
    duration_seconds=fields["duration_seconds"],
    root=Path(fields["root"]),
  )


if __name__ == "__main__":
  main()
