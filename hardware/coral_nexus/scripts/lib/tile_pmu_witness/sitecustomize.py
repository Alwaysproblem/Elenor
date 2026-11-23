"""Per-Tile PMU witness hook for the C022 equivalence gate (loaded via PYTHONPATH).

`python_baseline_summary.py` reads PMU keys that the JSON report never emits, so the committed
baseline holds `pmu: {}`/`tiles: []`. This hook records the REAL per-Tile PMU without touching
validator sources: it wraps `TileGroup._aggregate_pmu`, accumulating each Tile's counters (events,
named cycles, stall cycles) before they are merged into the Group PMU and reset.

Active only when CN_TILE_PMU_OUT is set; writes the accumulated JSON at interpreter exit.
"""

from __future__ import annotations

import atexit
import json
import os
from collections import defaultdict

_OUT = os.environ.get("CN_TILE_PMU_OUT")

if _OUT is not None:
  _out_path: str = _OUT
  from pipeline_validator.tile_group import TileGroup

  _acc: dict[str, dict[str, dict[str, int]]] = defaultdict(
    lambda: {"events": defaultdict(int), "named_cycles": defaultdict(int), "stall_cycles": defaultdict(int)}
  )
  _orig = TileGroup._aggregate_pmu

  def _wrapped(self) -> None:
    for tile in self.tiles:
      slot = _acc[str(tile.tile_id)]
      for name, value in tile.pmu.events.items():
        slot["events"][str(name)] += int(value)
      for name, value in tile.pmu.named_cycles.items():
        slot["named_cycles"][str(name)] += int(value)
      for name, value in tile.pmu.stall_cycles.items():
        slot["stall_cycles"][getattr(name, "name", str(name))] += int(value)
    _orig(self)

  TileGroup._aggregate_pmu = _wrapped  # type: ignore[method-assign]

  def _dump() -> None:
    doc = {
      tile_id: {kind: dict(sorted(values.items())) for kind, values in sorted(slot.items())}
      for tile_id, slot in sorted(_acc.items())
    }
    with open(_out_path, "w") as handle:
      json.dump(doc, handle, indent=2, sort_keys=True)
      handle.write("\n")

  atexit.register(_dump)
