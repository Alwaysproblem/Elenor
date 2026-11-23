"""Compare real per-Tile PMU between a reference tree (C035) and the current tree, per scenario.

usage: compare.py <ref_dir> <cur_dir> <out.json>
Each dir holds `<scenario>.tile_pmu.json` written by the sitecustomize hook. Fails on any difference
and on empty telemetry (an all-empty PMU would make equality vacuous).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCENARIOS = (
  "l2-admission-wait",
  "transformer-prefill-attention",
  "transformer-prefill-attention-baseline",
  "profile-reconfiguration",
)


def _total(doc: dict) -> int:
  return sum(sum(v.values()) for tile in doc.values() for v in tile.values())


def main() -> int:
  ref_dir, cur_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
  report: dict = {"schema": "cn_tile_pmu_witness/v1", "scenarios": {}}
  ok = True
  for name in SCENARIOS:
    ref = json.loads((ref_dir / f"{name}.tile_pmu.json").read_text())
    cur = json.loads((cur_dir / f"{name}.tile_pmu.json").read_text())
    nonempty = bool(ref) and _total(ref) > 0 and all(any(v for v in t.values()) for t in ref.values())
    equal = ref == cur
    ok &= nonempty and equal
    report["scenarios"][name] = {
      "equal": equal,
      "nonempty_telemetry": nonempty,
      "tiles": sorted(ref),
      "counter_total": _total(ref),
      "per_tile_total": {
        tile: sum(sum(v.values()) for v in data.values()) for tile, data in sorted(ref.items())
      },
    }
    print(name, "equal=", equal, "nonempty=", nonempty, "tiles=", sorted(ref), "total=", _total(ref))
    if not equal:
      for tile in sorted(set(ref) | set(cur)):
        if ref.get(tile) != cur.get(tile):
          print("  DIFF tile", tile)
  report["pass"] = ok
  out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
