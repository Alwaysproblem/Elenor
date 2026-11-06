"""cn command dispatcher (host side).

Parses scripts/commands.yaml, runs the registered handler with the remaining
argv, and records a run manifest for every invocation. Unknown commands,
unknown subcommands and unknown suites exit 3 without pretending to be
supported.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from sources_tool import load_yaml_limited

EXIT_UNKNOWN = 3
EXIT_NOT_IMPLEMENTED = 4
EXIT_USAGE = 2
EXIT_FAILURE = 1


def repo_root_from(root_arg: str) -> Path:
  root = Path(root_arg).resolve()
  if not (root / "hardware/coral_nexus/scripts/commands.yaml").exists():
    raise SystemExit("error: --root does not contain hardware/coral_nexus")
  return root


def main() -> None:
  argv = sys.argv[1:]
  if len(argv) < 2 or argv[0] != "--root":
    print("usage: cn --root <repo-root> <command> [args]", file=sys.stderr)
    raise SystemExit(EXIT_USAGE)
  root = repo_root_from(argv[1])
  h = root / "hardware/coral_nexus"
  registry = load_yaml_limited(h / "scripts/commands.yaml")
  commands = registry["commands"]

  words = argv[2:]
  if not words:
    print(
      "usage: cn <command> [args]; known: " + ", ".join(sorted(commands)),
      file=sys.stderr,
    )
    raise SystemExit(EXIT_USAGE)

  command, rest = words[0], words[1:]
  spec = commands.get(command)
  if spec is None:
    print(
      f"error: unknown command {command!r}; known: {', '.join(sorted(commands))}",
      file=sys.stderr,
    )
    raise SystemExit(EXIT_UNKNOWN)

  handler_rel = spec["handler"]
  handler = h / handler_rel
  if not handler.is_file():
    print(f"error: handler {handler_rel} missing", file=sys.stderr)
    raise SystemExit(EXIT_UNKNOWN)

  argv: list[str]
  if command == "unit":
    # unit: --suite NAME [...] is validated against the registry here.
    suite = None
    consume = list(rest)
    cleaned: list[str] = []
    while consume:
      token = consume.pop(0)
      if token == "--suite":
        if not consume:
          print("error: --suite requires a value", file=sys.stderr)
          raise SystemExit(EXIT_USAGE)
        suite = consume.pop(0)
        continue
      cleaned.append(token)
    suites = spec.get("suites") or {}
    if suite is None:
      print(f"error: unit requires --suite; known: {', '.join(sorted(suites))}", file=sys.stderr)
      raise SystemExit(EXIT_USAGE)
    if suite not in suites:
      print(
        f"error: unknown suite {suite!r}; known: {', '.join(sorted(suites))}",
        file=sys.stderr,
      )
      raise SystemExit(EXIT_UNKNOWN)
    argv = ["--suite", suite, *cleaned]
  elif spec.get("subcommands"):
    subcommands = spec["subcommands"]
    if not rest or rest[0] not in subcommands:
      print(
        f"error: {command} requires one of {subcommands}; got {rest[:1]!r}",
        file=sys.stderr,
      )
      raise SystemExit(EXIT_USAGE)
    argv = rest
  else:
    argv = rest

  manifest_dir = root / "out/coral_nexus/cn-runs"
  manifest_dir.mkdir(parents=True, exist_ok=True)
  stamp = time.strftime("%Y%m%dT%H%M%S")
  manifest_path = manifest_dir / f"{command}-{stamp}-{os.getpid()}.json"

  started = time.monotonic()
  proc = subprocess.run(["bash", str(handler), *argv], cwd=root)
  duration = time.monotonic() - started

  manifest = {
    "schema": "cn_run_manifest/v1",
    "command": command,
    "args": argv,
    "executor": spec["executor"],
    "handler": handler_rel,
    "exit_code": proc.returncode,
    "duration_seconds": round(duration, 3),
    "root": ".",
  }
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  raise SystemExit(proc.returncode)


if __name__ == "__main__":
  main()
