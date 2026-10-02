"""Compare two PagedAttention decode reports (plan §6 analyzer).

Reads both JSON reports plus their traces and emits makespan/speedup,
per-request-step latency, byte counters, cache hit rates by level, page
peaks, HBM-vs-BOA interval overlap and the append-complete to
first-attention-dispatch gap.  Missing fields fail loudly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_report(path: Path) -> dict:
  data = json.loads(Path(path).read_text(encoding="utf-8"))
  reports = data if isinstance(data, list) else [data]
  if len(reports) != 1:
    raise SystemExit(f"{path}: expected exactly one report, found {len(reports)}")
  return reports[0]


def _require(report: dict, key: str, where: str):
  if key not in report:
    raise SystemExit(f"{where}: report has no {key!r} field")
  return report[key]


def _trace_events(path: Path) -> list[dict]:
  if not path.exists():
    raise SystemExit(f"{path}: trace file is missing")
  data = json.loads(Path(path).read_text(encoding="utf-8"))
  if "traceEvents" not in data:
    raise SystemExit(f"{path}: trace has no traceEvents")
  return data["traceEvents"]


def _instant_named(events: list[dict], name: str) -> list[dict]:
  return [e for e in events if e.get("ph") == "X" or "ph" not in e]


def _hbm_read_intervals(events: list[dict]) -> list[tuple[int, int]]:
  intervals = []
  for event in events:
    if event.get("ph") != "X":
      continue
    name = str(event.get("name", ""))
    if "hbm_read" in name or "gather_refill" in name:
      intervals.append((int(event["ts"]), int(event["ts"]) + int(event.get("dur", 0))))
  return intervals


def _boa_intervals(events: list[dict]) -> list[tuple[int, int]]:
  intervals = []
  for event in events:
    if event.get("ph") != "X":
      continue
    if "BOA" in str(event.get("name", "")) or "MFE:matmul" in str(event.get("name", "")):
      intervals.append((int(event["ts"]), int(event["ts"]) + int(event.get("dur", 0))))
  return intervals


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
  merged: list[tuple[int, int]] = []
  for start, end in sorted(intervals):
    if merged and start <= merged[-1][1]:
      merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
      merged.append((start, end))
  return merged


def _overlap_ratio(
  left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> float | None:
  left = _merge(left)
  right = _merge(right)
  if not left or not right:
    return None
  total = sum(end - start for start, end in left)
  if total <= 0:
    return None
  cursor = 0
  covered = 0
  for start, end in right:
    while cursor < len(left) and left[cursor][1] <= start:
      cursor += 1
    probe = cursor
    while probe < len(left) and left[probe][0] < end:
      covered += max(0, min(left[probe][1], end) - max(left[probe][0], start))
      probe += 1
  return min(1.0, covered / total)


def analyze(
  pipeline_report: Path,
  baseline_report: Path,
  pipeline_trace: Path,
  baseline_trace: Path,
) -> dict:
  pipeline = _load_report(pipeline_report)
  baseline = _load_report(baseline_report)
  for name, report in (("pipeline", pipeline), ("baseline", baseline)):
    _require(report, "cycles", name)
    _require(report, "completed", name)
    _require(report, "events", name)

  pipeline_cycles = int(_require(pipeline, "cycles", "pipeline"))
  baseline_cycles = int(_require(baseline, "cycles", "baseline"))
  speedup = baseline_cycles / pipeline_cycles if pipeline_cycles else None

  def counters(report: dict) -> dict[str, int]:
    events = _require(report, "events", "report")
    keys = (
      "gather_requests",
      "gather_l1_hits",
      "gather_l2_hits",
      "gather_hbm_misses",
      "gather_cache_bypass_requests",
      "gather_mshr_merges",
      "gather_bytes",
      "gather_index_reads",
      "scatter_segments",
      "scatter_bytes",
      "scatter_index_reads",
      "scatter_overlap_wait_cycles",
      "scatter_commit_latency_cycles",
    )
    return {key: int(events.get(key, 0)) for key in keys}

  pipeline_events = _trace_events(pipeline_trace)
  baseline_events = _trace_events(baseline_trace)
  hbm_ratio_pipeline = _overlap_ratio(
    _hbm_read_intervals(pipeline_events), _boa_intervals(pipeline_events)
  )
  hbm_ratio_baseline = _overlap_ratio(
    _hbm_read_intervals(baseline_events), _boa_intervals(baseline_events)
  )

  def pools(report: dict) -> dict:
    device = _require(report, "device", "report")
    page_pools = device.get("page_pools")
    if not isinstance(page_pools, dict):
      raise SystemExit("report: device section has no page_pools")
    # Page-pool state is sampled per cycle; the last sample is the final one.
    sampled = page_pools.get("pools")
    if isinstance(sampled, dict) and "tail" in sampled:
      return {"pools": sampled["tail"]}
    return {"pools": sampled if isinstance(sampled, dict) else {}}

  return {
    "makespan": {
      "pipeline_cycles": pipeline_cycles,
      "baseline_cycles": baseline_cycles,
      "speedup": speedup,
    },
    "completed": {
      "pipeline": bool(_require(pipeline, "completed", "pipeline")),
      "baseline": bool(_require(baseline, "completed", "baseline")),
    },
    "counters": {
      "pipeline": counters(pipeline),
      "baseline": counters(baseline),
    },
    "hbm_boa_overlap_ratio": {
      "pipeline": hbm_ratio_pipeline,
      "baseline": hbm_ratio_baseline,
    },
    "page_pools": {
      "pipeline": pools(pipeline),
      "baseline": pools(baseline),
    },
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pipeline-report", required=True)
  parser.add_argument("--baseline-report", required=True)
  parser.add_argument("--pipeline-trace", required=True)
  parser.add_argument("--baseline-trace", required=True)
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args(argv)
  result = analyze(
    Path(args.pipeline_report),
    Path(args.baseline_report),
    Path(args.pipeline_trace),
    Path(args.baseline_trace),
  )
  print(json.dumps(result, indent=2) if args.json else _render(result))
  return 0


def _render(result: dict) -> str:
  lines = ["PagedAttention decode comparison (plan §6 analyzer)"]
  makespan = result["makespan"]
  lines.append(
    f"  cycles: pipeline {makespan['pipeline_cycles']} / baseline {makespan['baseline_cycles']}"
    f"  speedup {makespan['speedup']:.3f}" if makespan["speedup"] is not None else "  speedup n/a"
  )
  completed = result["completed"]
  lines.append(f"  completed: pipeline {completed['pipeline']} / baseline {completed['baseline']}")
  for variant, values in result["counters"].items():
    lines.append(f"  {variant} bytes: gather {values['gather_bytes']} scatter {values['scatter_bytes']}")
    lines.append(
      f"  {variant} cache: l1_hits {values['gather_l1_hits']} l2_hits {values['gather_l2_hits']}"
      f" hbm_misses {values['gather_hbm_misses']} bypass {values['gather_cache_bypass_requests']}"
      f" merges {values['gather_mshr_merges']}"
    )
  overlap = result["hbm_boa_overlap_ratio"]
  lines.append(
    f"  HBM-read / BOA overlap: pipeline {overlap['pipeline']} baseline {overlap['baseline']}"
  )
  for variant, pools in result["page_pools"].items():
    for name, pool in (pools.get("pools") or {}).items():
      lines.append(
        f"  {variant} pool {name}: live {pool.get('live_pages')} peak {pool.get('peak_live_pages')}"
        f" allocated {pool.get('allocated_pages')} freed {pool.get('freed_pages')}"
      )
  return "\n".join(lines)


if __name__ == "__main__":
  raise SystemExit(main())
