"""Width derivation for the boundary protocol (C017 / W1-01).

A thin, strict front end over ``validate_protocol.derive_all``: every domain in
``lifetime_bounds.yaml`` must end up either with a derived width or with an
explicit ``derivable: false`` record (reason + needs). A domain that is neither
is an error, so a width can never be silently invented or silently dropped.

The generator never reads a width from anywhere else; the fixed 64-bit transport
word of ``wire_schema_v1.yaml`` is a container, not a width.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "scripts/lib"))

import validate_protocol as vp  # noqa: E402


@dataclass(frozen=True)
class DerivedWidths:
  derived: dict[str, dict[str, int]]
  unresolved: list[dict[str, Any]]


class WidthError(Exception):
  def __init__(self, errors: list[str]) -> None:
    super().__init__("; ".join(errors))
    self.errors = errors


def derive(budgets: dict[str, Any], lifetimes: dict[str, Any]) -> DerivedWidths:
  widths, errors = vp.derive_all(budgets, lifetimes)
  derived = {w.name: {"window": w.window, "bits": w.bits} for w in widths}
  unresolved: list[dict[str, Any]] = []
  for dom in lifetimes["domains"]:
    name = dom["name"]
    if dom.get("derivable") is False:
      unresolved.append(
        {"name": name, "reason": dom.get("reason", ""), "needs": list(dom.get("needs") or [])}
      )
  if errors:
    raise WidthError(errors)
  return DerivedWidths(derived=derived, unresolved=unresolved)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dir", default=str(H / "protocol"))
  args = parser.parse_args()
  data = vp.load_all(Path(args.dir))
  try:
    result = derive(data["budgets"], data["lifetimes"])
  except WidthError as exc:
    print("\n".join(exc.errors), file=sys.stderr)
    raise SystemExit(1) from exc
  print(json.dumps({"derived": result.derived, "unresolved": result.unresolved}, indent=2))


if __name__ == "__main__":
  main()
