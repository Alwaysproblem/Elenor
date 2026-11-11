"""Lint the generated CSR bindings against their sources (C017).

Checks the allocation recomputed from ``csr_name_map.yaml`` + the upstream occupancy table against every
generated artifact (layout JSON, C header, SystemVerilog offsets, Python wire module), and the allocation
itself against the upstream table: a collision, a window escape or a duplicate address fails and names
the occupant. It compares generated artifacts with a fresh allocation; it does not grep other source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import csr_allocator  # noqa: E402

PROTOCOL = H / "protocol"
GENERATED = PROTOCOL / "generated"


def _ident(name: str) -> str:
  return re.sub(r"[^0-9A-Za-z]", "_", name).upper()


def lint_table(rows: list[dict[str, Any]], upstream: dict[str, Any], window: tuple[int, int]) -> list[str]:
  """Collision/window/duplicate checks on an allocation table (rows: name + address)."""
  errors: list[str] = []
  occupied = {int(e["address"]): str(e["name"]) for e in upstream["occupied"]}
  seen: dict[int, str] = {}
  for row in rows:
    name, addr = row["name"], row["address"]
    if addr in occupied:
      errors.append(f"csr {name}: address {addr:#x} is occupied by upstream {occupied[addr]}")
    if addr in seen:
      errors.append(f"csr {name}: address {addr:#x} already assigned to {seen[addr]}")
    seen[addr] = name
    if not window[0] <= addr <= window[1]:
      errors.append(f"csr {name}: address {addr:#x} is outside the allocation window")
  return errors


def lint(directory: Path = PROTOCOL, generated: Path = GENERATED, wire_py: Path | None = None) -> list[str]:
  csr_map = yaml.safe_load((directory / "csr_name_map.yaml").read_text())
  upstream = json.loads((directory / "upstream_csr_occupied.json").read_text())
  errors: list[str] = []
  try:
    expected = csr_allocator.allocate(csr_map, upstream)
    window = csr_allocator.window_of(csr_map)
  except csr_allocator.CsrAllocError as exc:
    return [f"csr: {exc}"]
  errors += lint_table(expected, upstream, window)
  want = {r["name"]: r["address"] for r in expected}

  layout = json.loads((generated / "cn_layout.json").read_text())
  got = {r["name"]: r["address"] for r in layout["csr"]}
  errors += _diff("cn_layout.json", got, want)

  header = (generated / "cn_messages.h").read_text()
  got = {n: int(v, 16) for n, v in re.findall(r"#define CN_CSR_(\w+) 0x([0-9a-f]+)", header)}
  errors += _diff("cn_messages.h", got, {_ident(n): a for n, a in want.items()})

  svh = (generated / "cn_offsets.svh").read_text()
  got = {n: int(v, 16) for n, v in re.findall(r"localparam int CN_CSR_(\w+) = 12'h([0-9a-f]+);", svh)}
  errors += _diff("cn_offsets.svh", got, {_ident(n): a for n, a in want.items()})

  py = wire_py or (H.parents[1] / "pipeline_validator/cosim/wire.py")
  match = re.search(r"^CSR: dict\[str, int\] = (\{.*?\})$", py.read_text(), re.S | re.M)
  got_py = {}
  if match:
    import ast

    got_py = ast.literal_eval(match.group(1))
  errors += _diff("wire.py", got_py, want)
  return errors


def _diff(where: str, got: dict[str, int], want: dict[str, int]) -> list[str]:
  errors: list[str] = []
  for name in sorted(set(got) | set(want)):
    if name not in got:
      errors.append(f"{where}: CSR {name} is missing")
    elif name not in want:
      errors.append(f"{where}: CSR {name} is not a declared name")
    elif got[name] != want[name]:
      errors.append(f"{where}: CSR {name} is {got[name]:#x}, the allocator gives {want[name]:#x}")
  return errors


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dir", default=str(PROTOCOL))
  args = parser.parse_args()
  errors = lint(Path(args.dir))
  if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
  print("csr_lint: ok")


if __name__ == "__main__":
  main()
