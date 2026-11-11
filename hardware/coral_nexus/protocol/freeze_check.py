"""Freeze checker for the boundary protocol (C017); it never seals.

``collect`` records the sha256 of every input and generated artifact with state ``collected-not-frozen``;
``verify`` re-hashes them against such a record and, independently of any record, re-renders every
generated artifact in memory and compares bytes with what is on disk, so a hand-edited generated file
fails even when no record exists. Writing the one formal ``PROTOCOL_FREEZE.json`` is C020's job: this
module has no seal entry point, and ``verify`` refuses a record whose state is not
``collected-not-frozen`` or ``frozen``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

H = Path(__file__).resolve().parents[1]
REPO = H.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_protocol as gp  # noqa: E402

STATES = ("collected-not-frozen", "frozen")


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
  return str(path.resolve().relative_to(REPO))


def collect(directory: Path = gp.PROTOCOL_DIR) -> dict[str, Any]:
  """Hash inputs and generated outputs. Generation must already be consistent (verify_generated)."""
  errors = verify_generated(directory)
  if errors:
    raise gp.GenError(errors)
  inputs = {f"hardware/coral_nexus/protocol/{n}": _sha(directory / n) for n in gp._INPUTS}
  outputs = {_rel(p): _sha(p) for p in sorted(gp.render_all(gp.build_model(directory)))}
  return {
    "schema": "cn_protocol_freeze_record/v1",
    "state": "collected-not-frozen",
    "inputs": inputs,
    "generated": outputs,
  }


def verify_generated(
  directory: Path = gp.PROTOCOL_DIR, on_disk: dict[Path, Path] | None = None
) -> list[str]:
  """Re-render in memory and compare with disk: catches hand edits and stale outputs.

  `on_disk` maps a default output path to the file to compare instead (tests use a private copy).
  """
  try:
    rendered = gp.render_all(gp.build_model(directory))
  except gp.GenError as exc:
    return [f"inputs do not generate: {e}" for e in exc.errors]
  errors: list[str] = []
  for path, text in rendered.items():
    actual = (on_disk or {}).get(path, path)
    if not actual.is_file():
      errors.append(f"{_rel(path)}: generated file is missing")
    elif actual.read_text() != text:
      errors.append(f"{_rel(path)}: differs from a fresh generation (hand-edited or stale)")
  return errors


def verify(record: dict[str, Any], root: Path = REPO) -> list[str]:
  errors: list[str] = []
  if record.get("schema") != "cn_protocol_freeze_record/v1":
    errors.append("record: unknown schema")
  if record.get("state") not in STATES:
    errors.append(f"record: state {record.get('state')!r} is not one of {STATES}")
  for section in ("inputs", "generated"):
    entries = record.get(section)
    if not isinstance(entries, dict) or not entries:
      errors.append(f"record.{section}: missing or empty")
      continue
    for rel, digest in sorted(entries.items()):
      path = root / rel
      if not path.is_file():
        errors.append(f"{rel}: listed in the record but missing")
      elif _sha(path) != digest:
        errors.append(f"{rel}: sha256 differs from the record")
  return errors


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="cmd", required=True)
  c = sub.add_parser("collect")
  c.add_argument("--out", required=True)
  v = sub.add_parser("verify")
  v.add_argument("--record")
  args = parser.parse_args()
  if args.cmd == "collect":
    try:
      record = collect()
    except gp.GenError as exc:
      print("\n".join(exc.errors), file=sys.stderr)
      raise SystemExit(1) from exc
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} (state={record['state']})")
    return
  errors = verify_generated()
  if args.record:
    errors += verify(json.loads(Path(args.record).read_text()))
  if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
  print("freeze_check: ok")


if __name__ == "__main__":
  main()
