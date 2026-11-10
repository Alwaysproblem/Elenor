"""Extract the upstream-occupied CSR address table from the pinned source.

Reads ``CsrAddress`` from the pristine, lock-pinned Coral submodule and writes
``upstream_csr_occupied.json``. The numbers are parsed, never hand-copied; the
output records the source commit and file sha256 so a stale table is
detectable. Refuses to run when the submodule HEAD differs from the commit in
``env/sources.lock`` (a table from a different pin proves nothing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
CSR_REL = "hdl/chisel/src/coralnpu/scalar/Csr.scala"
_ENTRY = re.compile(r"^\s*val\s+(\w+)\s*=\s*Value\(0x([0-9a-fA-F]+)\.U\((\d+)\.W\)\)")


def locked_commit(lock_path: Path, name: str = "coralnpu") -> str:
  lock = json.loads(lock_path.read_text())
  for source in lock["sources"]:
    if source["name"] == name:
      return str(source["commit"])
  raise SystemExit(f"error: source {name!r} not in {lock_path}")


def submodule_head(path: Path) -> str:
  proc = subprocess.run(
    ["git", "-C", str(path), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=True,
  )
  return proc.stdout.strip()


def extract(csr_scala: Path) -> list[dict]:
  """Parse the CsrAddress enum block into [{name, address, width_bits, line}]."""
  entries: list[dict] = []
  in_enum = False
  for number, line in enumerate(csr_scala.read_text().splitlines(), start=1):
    if "object CsrAddress extends ChiselEnum" in line:
      in_enum = True
      continue
    if not in_enum:
      continue
    if line.startswith("}"):
      break
    match = _ENTRY.match(line)
    if match:
      entries.append(
        {
          "name": match.group(1),
          "address": int(match.group(2), 16),
          "width_bits": int(match.group(3)),
          "line": number,
        }
      )
  return entries


def build(upstream: Path, lock_path: Path) -> dict:
  pinned = locked_commit(lock_path)
  head = submodule_head(upstream)
  if head != pinned:
    raise SystemExit(f"error: submodule HEAD {head} != locked commit {pinned}")
  csr_scala = upstream / CSR_REL
  entries = extract(csr_scala)
  if not entries:
    raise SystemExit(f"error: no CsrAddress entries parsed from {csr_scala}")
  addresses = [e["address"] for e in entries]
  if len(set(addresses)) != len(addresses):
    raise SystemExit("error: duplicate addresses in upstream CsrAddress")
  return {
    "schema": "cn_upstream_csr/v1",
    "source": {
      "repo": "coralnpu",
      "commit": pinned,
      "file": CSR_REL,
      "file_sha256": hashlib.sha256(csr_scala.read_bytes()).hexdigest(),
    },
    # 0x000 is Chisel's internal "invalid address" marker, not a CSR.
    "internal_invalid_address": 0,
    "occupied": [e for e in entries if e["address"] != 0],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--upstream", default=str(H / "upstream/coralnpu"))
  parser.add_argument("--lock", default=str(H / "env/sources.lock"))
  parser.add_argument("--out", default=str(H / "protocol/upstream_csr_occupied.json"))
  args = parser.parse_args()
  doc = build(Path(args.upstream), Path(args.lock))
  Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
  print(f"wrote {args.out}: {len(doc['occupied'])} occupied CSRs @ {doc['source']['commit'][:12]}")


if __name__ == "__main__":
  try:
    main()
  except subprocess.CalledProcessError as exc:
    sys.exit(f"error: git failed: {exc}")
