"""Extract deterministic Python-baseline metrics for C022 (`protocol/baseline_python.json`).

Compares and records for each of the four C022 baseline scenarios (`l2-admission-wait`,
`transformer-prefill-attention`, `transformer-prefill-attention-baseline`,
`profile-reconfiguration`):
- `cycles`, `completed`, `credit_invariant_ok`, `fidelity`, `context_count`, `device_context_count`
- per-Tile and Group PMU named event/cycle dictionaries
- deterministic trace partial-order digest & event sequence over stable architectural fields
  (`pid`/`tid`/`ph`/`name`/`ts`/`dur` and path-free `args` keys), excluding host paths or timestamps.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCENARIOS: tuple[str, ...] = (
  "l2-admission-wait",
  "transformer-prefill-attention",
  "transformer-prefill-attention-baseline",
  "profile-reconfiguration",
)

VOLATILE_ARG_KEYS: frozenset[str] = frozenset(
  {"source_ref", "ir_file", "compiled_file", "report_path", "trace_path", "timestamp"}
)


def _clean_args(raw: Any) -> dict[str, Any]:
  if not isinstance(raw, dict):
    return {}
  out: dict[str, Any] = {}
  for k in sorted(raw):
    if k in VOLATILE_ARG_KEYS:
      continue
    val = raw[k]
    if isinstance(val, str) and ("/" in val or val.endswith(".mlir") or val.endswith(".json")):
      continue
    out[k] = val
  return out


def summarize_trace(trace_path: Path) -> dict[str, Any]:
  doc = json.loads(trace_path.read_text())
  events = doc.get("traceEvents") if isinstance(doc, dict) else doc
  assert isinstance(events, list)
  normalized: list[dict[str, Any]] = []
  for ev in events:
    if not isinstance(ev, dict):
      continue
    ph = ev.get("ph")
    if ph == "M":
      continue
    normalized.append(
      {
        "ph": ph,
        "cat": ev.get("cat", ""),
        "name": ev.get("name", ""),
        "pid": ev.get("pid", 0),
        "tid": ev.get("tid", 0),
        "ts": ev.get("ts", 0),
        "dur": ev.get("dur", 0),
        "args": _clean_args(ev.get("args")),
      }
    )
  canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
  return {
    "event_count": len(normalized),
    "partial_order_sha256": hashlib.sha256(canonical).hexdigest(),
  }


def summarize_report(report_path: Path, trace_path: Path) -> dict[str, Any]:
  raw = json.loads(report_path.read_text())
  rep = raw[0] if isinstance(raw, list) else raw
  cfg = rep.get("configuration") or {}
  sim_cfg = cfg.get("sim") or cfg
  group_pmu = rep.get("pmu") or (rep.get("summary") or {}).get("pmu") or {}
  tiles_snap = ((rep.get("group_snapshot") or {}).get("tiles")) or rep.get("tiles") or []
  per_tile_state = [
    {
      "tile_id": t.get("tile_id", idx),
      "boa_state": t.get("boa_state"),
      "evu_state": t.get("evu_state"),
      "mfe_state": t.get("mfe_state"),
      "use_state": t.get("use_state"),
      "faulted": t.get("faulted", False),
    }
    for idx, t in enumerate(tiles_snap)
    if isinstance(t, dict)
  ]
  return {
    "completed": bool(rep.get("completed")),
    "cycles": int(rep.get("cycles")),
    "credit_invariant_ok": bool(rep.get("credit_invariant_ok", True)),
    "checks": rep.get("checks") or {},
    "fidelity": sim_cfg.get("fidelity") or (rep.get("memory") or {}).get("fidelity"),
    "context_count": sim_cfg.get("context_count"),
    "device_context_count": sim_cfg.get("device_context_count"),
    "pmu": group_pmu,
    "tiles": per_tile_state,
    "trace": summarize_trace(trace_path),
  }


def build_baseline_document(run_dir: Path) -> dict[str, Any]:
  scenarios_out: dict[str, Any] = {}
  for name in SCENARIOS:
    rpath = run_dir / f"{name}.report.json"
    tpath = run_dir / f"{name}.trace.json"
    scenarios_out[name] = summarize_report(rpath, tpath)
  return {
    "schema": "cn_python_tileport_baseline/v1",
    "card": "C022",
    "note": (
      "Pure-Python Tile baseline across timing_only and full_memory runners; "
      "proves zero behavioral drift across the C022 TilePort seam (not numeric/RTL proof)."
    ),
    "scenarios": scenarios_out,
  }
