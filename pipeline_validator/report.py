"""Report generation for execution, scheduler, device, and PMU state.

The legacy workload checks remain unchanged.  Structured Group/CPU snapshots
are copied from their owning models; report generation does not infer hardware
events or resource use from trace names.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise

from .pmu import StallReason
from .simulator import SimResult
from .workloads import Workload


@dataclass
class WorkloadReport:
  name: str
  description: str
  cycles: int
  completed: bool
  reason: str
  utilization: float
  stall_breakdown: dict
  engine_active: dict  # engine -> active cycles
  stream_counters: dict  # per-queue occupancy/full/empty cycles
  events: dict
  checks: list[dict]  # pass/fail items
  credit_invariant_ok: bool
  num_tiles: int = 4
  gather_fidelity: str | None = None
  memory: dict = field(default_factory=dict)  # PR 5 peak reconciliation
  scheduler: dict = field(default_factory=dict)
  device: dict = field(default_factory=dict)
  resources: dict = field(default_factory=dict)
  makespan: dict = field(default_factory=dict)
  request_timing: dict = field(default_factory=dict)


def _ratio(counters: dict, total: int) -> float:
  return (counters.get("active", 0) / total) if total else 0.0


_REPORT_RECORD_LIMIT = 16
_TAIL_QUANTILE_MIN_SAMPLES = 20
_STEADY_INTERVAL_MIN_COMPLETIONS = 8
_RECORD_COLLECTION_NAMES = ("record", "history", "request", "launch", "completion")


def _is_record_collection_name(name: str) -> bool:
  lowered = name.lower()
  return any(token in lowered for token in _RECORD_COLLECTION_NAMES)


def _is_collection_mapping(name: str, value: Mapping) -> bool:
  """Identify mapping-valued state collections, not scalar/config maps."""
  if name.lower() in {"configuration", "config", "counters", "counter", "pmu", "named_cycles"}:
    return False
  return bool(value) and all(isinstance(item, Mapping) for item in value.values())


def _bounded_snapshot(value, *, name: str = ""):
  """Copy a structured snapshot while bounding state collections.

  Scalar counter/configuration maps remain intact.  Sequences and
  mapping-valued collections become counted tails, so a long simulation
  cannot make the report itself an unbounded event log.
  """
  if isinstance(value, Mapping):
    if _is_collection_mapping(name, value):
      tail = list(deque(value.items(), maxlen=_REPORT_RECORD_LIMIT))
      return {
        "sample_count": len(value),
        "tail_sample_count": len(tail),
        "tail": {key: _bounded_snapshot(item, name=str(key)) for key, item in tail},
      }
    return {key: _bounded_snapshot(item, name=str(key)) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    copied_tail = [_bounded_snapshot(item, name=name) for item in value[-_REPORT_RECORD_LIMIT:]]
    if _is_record_collection_name(name) or len(value) > _REPORT_RECORD_LIMIT:
      return {"sample_count": len(value), "tail_sample_count": len(copied_tail), "tail": copied_tail}
    return copied_tail
  return value


def _request_records(device: Mapping) -> list[dict]:
  """Use the CPU controller's single lifecycle-record contract."""
  return device.get("launch_records", [])


def _cycle(record: Mapping, name: str) -> int | None:
  value = record.get(name)
  return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _duration(start: int | float | None, end: int | float | None) -> int | float | None:
  if start is None or end is None or end < start:
    return None
  return end - start


def _nearest_rank(values: list[int | float], percentile: int) -> int | float:
  ordered = sorted(values)
  rank = (percentile * len(ordered) + 99) // 100
  return ordered[max(rank - 1, 0)]


def _distribution(values: list[int | float]) -> dict:
  if not values:
    return {}
  result = {
    "sample_count": len(values),
    "min_cycles": min(values),
    "mean_cycles": round(sum(values) / len(values), 3),
    "max_cycles": max(values),
  }
  if len(values) >= _TAIL_QUANTILE_MIN_SAMPLES:
    result["p95_cycles"] = _nearest_rank(values, 95)
  if len(values) >= 100:
    result["p99_cycles"] = _nearest_rank(values, 99)
  return result


def _request_timing_summary(device: Mapping) -> dict:
  records = _request_records(device)
  if not records:
    return {}
  normalized = []
  series: dict[str, list[int | float]] = {
    "submit_to_complete": [],
    "admit_to_complete": [],
    "port_accept_to_complete": [],
    "wait_dependencies": [],
    "wait_admission": [],
    "wait_port_accept": [],
    "wait_group_admission": [],
    "active": [],
    "drain": [],
  }
  completion_cycles: list[int | float] = []
  for record in records:
    submit = _cycle(record, "submit_cycle")
    deps_ready = _cycle(record, "dependencies_ready_cycle")
    admitted = _cycle(record, "admission_cycle")
    active = _cycle(record, "active_cycle")
    draining = _cycle(record, "drain_cycle")
    complete = _cycle(record, "completion_cycle")
    row = {
      key: value
      for key, value in (
        ("request_id", record.get("request_id")),
        ("context", record.get("context")),
        ("event", record.get("event")),
        ("group_affinity", record.get("group_affinity")),
        ("status", record.get("status")),
        ("reason", record.get("reason")),
        ("submit_cycle", submit),
        ("dependencies_ready_cycle", deps_ready),
        ("admission_cycle", admitted),
        ("active_cycle", active),
        ("drain_cycle", draining),
        ("completion_cycle", complete),
      )
      if value is not None
    }
    durations = {
      "submit_to_complete_cycles": _duration(submit, complete),
      "admit_to_complete_cycles": _duration(active, complete),
      "port_accept_to_complete_cycles": _duration(admitted, complete),
      "wait_dependencies_cycles": _duration(submit, deps_ready),
      "wait_admission_cycles": _duration(deps_ready, active),
      "wait_port_accept_cycles": _duration(deps_ready, admitted),
      "wait_group_admission_cycles": _duration(admitted, active),
      "active_cycles": _duration(active, draining if draining is not None else complete),
      "drain_cycles": _duration(draining, complete),
    }
    for key, value in durations.items():
      if value is None:
        continue
      row[key] = value
      series[key.removesuffix("_cycles")].append(value)
    if complete is not None and record.get("status") == "success":
      completion_cycles.append(complete)
    normalized.append(row)

  result = {
    "record_count": len(records),
    "latency": {name: summary for name, values in series.items() if (summary := _distribution(values))},
    "record_tail_sample_count": min(len(normalized), _REPORT_RECORD_LIMIT),
    "record_tail": normalized[-_REPORT_RECORD_LIMIT:],
  }
  completion_cycles.sort()
  intervals = [later - earlier for earlier, later in pairwise(completion_cycles) if later >= earlier]
  if intervals:
    result["completion_intervals"] = {
      "completed_request_count": len(completion_cycles),
      **_distribution(intervals),
    }
    if len(completion_cycles) >= _STEADY_INTERVAL_MIN_COMPLETIONS:
      steady_start = len(completion_cycles) // 2
      steady = [
        later - earlier for earlier, later in pairwise(completion_cycles[steady_start:]) if later >= earlier
      ]
      steady_tail = steady[-_REPORT_RECORD_LIMIT:]
      result["steady_completion_intervals"] = {
        "definition": ("completion intervals within the second half of cycle-ordered completed requests"),
        "warmup_completion_count": steady_start,
        **_distribution(steady),
        "tail_sample_count": len(steady_tail),
        "tail_cycles": steady_tail,
      }
  return result


def _effective_resources(
  result: SimResult, scheduler: Mapping, device: Mapping, fallback_tiles: int
) -> dict:
  tiles = result.group_snapshot.get("tiles", [])
  contexts_by_tile = {}
  if isinstance(tiles, (list, tuple)):
    for index, tile in enumerate(tiles):
      if not isinstance(tile, Mapping):
        continue
      tile_id = tile.get("tile_id", index)
      uce = tile.get("uce")
      if isinstance(uce, Mapping) and uce.get("context_count") is not None:
        contexts_by_tile[tile_id] = uce["context_count"]
  tile_count = len(tiles) if isinstance(tiles, (list, tuple)) and tiles else fallback_tiles
  hardware = {
    "actual_tile_count": tile_count,
    "actual_contexts_per_tile": contexts_by_tile,
    "actual_tile_contexts_total": sum(contexts_by_tile.values()),
  }
  unique_context_counts = set(contexts_by_tile.values())
  if len(unique_context_counts) == 1:
    hardware["actual_context_count"] = next(iter(unique_context_counts))
  device_config = device.get("configuration")
  if not isinstance(device_config, Mapping):
    device_config = {}
  cpu_limit = device_config.get("outstanding_launch_limit")
  cpu_limit_source = "device.configuration.outstanding_launch_limit"
  if cpu_limit is None:
    cpu_limit = device_config.get("device_context_count")
    cpu_limit_source = "device.configuration.device_context_count"
  if cpu_limit is None:
    cpu_limit = result.slot_count
    cpu_limit_source = "result.slot_count"

  resources = {
    "configuration": result.configuration,
    "tile_hardware": hardware,
    "cpu_controller": {
      "outstanding_launch_limit": cpu_limit,
      "outstanding_launch_limit_source": cpu_limit_source,
      "configuration": dict(device_config),
    },
  }
  group_config = scheduler.get("configuration")
  if isinstance(group_config, Mapping):
    resources["group_scheduler"] = {"configuration": dict(group_config)}
  return resources


def build_report(wl: Workload, result: SimResult, num_tiles: int = 4) -> WorkloadReport:
  pmu = result.pmu
  total = result.cycles or 1
  tile_snapshots = result.group_snapshot.get("tiles", [])
  actual_num_tiles = (
    len(tile_snapshots) if isinstance(tile_snapshots, (list, tuple)) and tile_snapshots else num_tiles
  )
  # engine/uce stall cycles are aggregated across all tiles, so the
  # correct denominator for per-tile engine ratios uses actual hardware.
  tile_cycles = total * actual_num_tiles

  engine_active = {
    "BOA": pmu.named_cycles.get("boa_active", 0),
    "EVU": pmu.named_cycles.get("evu_active", 0),
    "MFE": pmu.named_cycles.get("mfe_active", 0),
    "USE": pmu.named_cycles.get("use_active", 0),
  }

  stream_counters = {}
  for k, v in pmu.named_cycles.items():
    if k.startswith("queue") or k in ("occupancy", "credit_full", "credit_empty", "credit_fault"):
      stream_counters[k] = v

  checks = _run_checks(wl, result, engine_active, tile_cycles)
  scheduler_raw = result.group_snapshot.get("scheduler")
  if not isinstance(scheduler_raw, Mapping):
    scheduler_raw = {}
  device_raw = getattr(result, "device_snapshot", {})
  if not isinstance(device_raw, Mapping):
    device_raw = {}
  scheduler = _bounded_snapshot(scheduler_raw)
  group_pmu = pmu.prefixed_snapshot("group_")
  if any(group_pmu.values()):
    scheduler["pmu"] = group_pmu
  device = _bounded_snapshot(device_raw)
  resources = _effective_resources(result, scheduler_raw, device_raw, actual_num_tiles)
  makespan = {
    "origin_cycle": 0,
    "terminal_cycle": result.cycles,
    "span_cycles": result.cycles,
    "convention": (
      "span_cycles is terminal_cycle - origin_cycle; legacy cycles "
      "remains the simulator termination cycle value"
    ),
  }
  request_timing = _request_timing_summary(device_raw)

  return WorkloadReport(
    name=wl.name,
    description=wl.description,
    cycles=result.cycles,
    completed=result.completed,
    reason=result.reason,
    utilization=result.utilization(actual_num_tiles),
    stall_breakdown=pmu.stall_breakdown(),
    engine_active=engine_active,
    stream_counters=stream_counters,
    events=dict(pmu.events),
    checks=checks,
    credit_invariant_ok=result.credit_invariant_ok,
    num_tiles=actual_num_tiles,
    gather_fidelity=(
      "deterministic_profiled_not_address_or_value_accurate"
      if pmu.events.get("gather_requests", 0) > 0
      else None
    ),
    memory=(_memory_summary(result.group_snapshot.get("memory")) if result.memory_trace else {}),
    scheduler=scheduler,
    device=device,
    resources=resources,
    makespan=makespan,
    request_timing=request_timing,
  )


def _memory_summary(group_memory: dict | None) -> dict:
  """Extract memory peak reconciliation values from the group snapshot.

  Values are None (and the corresponding key omitted) when the run
  used a fidelity without memory accounting - the report never
  reconstructs memory state from trace events.
  """
  if not group_memory:
    return {}
  summary: dict = {}
  l2 = group_memory.get("l2") or {}
  if l2.get("peak_allocated_bytes") is not None:
    summary["l2_peak_allocated_bytes"] = l2["peak_allocated_bytes"]
  l1 = group_memory.get("l1") or {}
  l1_peaks = {
    tile_id: (entry or {}).get("allocator", {}).get("peak_allocated_bytes")
    for tile_id, entry in l1.items()
    if (entry or {}).get("allocator")
  }
  if any(value is not None for value in l1_peaks.values()):
    summary["l1_peak_allocated_bytes"] = l1_peaks
  transfers = group_memory.get("transfers") or {}
  if transfers.get("hbm_outstanding_peak") is not None:
    summary["hbm_outstanding_peak"] = transfers["hbm_outstanding_peak"]
  hbm = group_memory.get("hbm") or {}
  if hbm.get("used_bytes") is not None:
    summary["hbm_used_bytes"] = hbm["used_bytes"]
  return summary


def _run_checks(wl: Workload, result: SimResult, engine_active: dict, total: int) -> list[dict]:
  checks: list[dict] = []
  exp = wl.expected

  # 1. completion
  checks.append(
    {"check": "task_completed", "expected": True, "actual": result.completed, "pass": result.completed}
  )
  # 2. credit invariant
  checks.append(
    {
      "check": "credit_invariant",
      "expected": True,
      "actual": result.credit_invariant_ok,
      "pass": result.credit_invariant_ok,
    }
  )

  boa_ratio = engine_active.get("BOA", 0) / total if total else 0
  evu_ratio = engine_active.get("EVU", 0) / total if total else 0
  mfe_ratio = engine_active.get("MFE", 0) / total if total else 0
  stream_stall = result.pmu.stall_cycles.get(StallReason.STREAM_CREDIT, 0)
  stream_stall_ratio = stream_stall / total if total else 0

  if "boa_active_ratio_min" in exp:
    checks.append(
      {
        "check": "boa_active_ratio",
        "expected_min": exp["boa_active_ratio_min"],
        "actual": round(boa_ratio, 3),
        "pass": boa_ratio >= exp["boa_active_ratio_min"],
      }
    )
  if "mfe_active_ratio_min" in exp:
    checks.append(
      {
        "check": "mfe_active_ratio",
        "expected_min": exp["mfe_active_ratio_min"],
        "actual": round(mfe_ratio, 3),
        "pass": mfe_ratio >= exp["mfe_active_ratio_min"],
      }
    )
  if "evu_active_ratio_min" in exp:
    checks.append(
      {
        "check": "evu_active_ratio",
        "expected_min": exp["evu_active_ratio_min"],
        "actual": round(evu_ratio, 3),
        "pass": evu_ratio >= exp["evu_active_ratio_min"],
      }
    )
  if "stream_stall_ratio_max" in exp:
    checks.append(
      {
        "check": "stream_stall_ratio",
        "expected_max": exp["stream_stall_ratio_max"],
        "actual": round(stream_stall_ratio, 3),
        "pass": stream_stall_ratio <= exp["stream_stall_ratio_max"],
      }
    )
  if exp.get("stream_s0_occupancy_seen"):
    occ = result.pmu.named_cycles.get("occupancy", 0)
    checks.append({"check": "stream_occupancy_seen", "expected": True, "actual": occ > 0, "pass": occ > 0})
  if exp.get("producer_consumer_overlap"):
    # overlap: both BOA (role1) and EVU (role1) active in same window
    overlap = engine_active.get("BOA", 0) > 0 and engine_active.get("EVU", 0) > 0
    checks.append(
      {"check": "producer_consumer_overlap", "expected": True, "actual": overlap, "pass": overlap}
    )
  if exp.get("mfe_page_stream_active"):
    # MFE Page Stream active: MFE ran (page-stream gather + store)
    mfe_active_total = engine_active.get("MFE", 0)
    checks.append(
      {
        "check": "mfe_page_stream_active",
        "expected": True,
        "actual": mfe_active_total > 0,
        "pass": mfe_active_total > 0,
      }
    )
  if exp.get("dual_boa_qk_pv"):
    # dual BOA: BOA ran (QK + PV are both BOA matmuls)
    boa_active_total = engine_active.get("BOA", 0)
    checks.append(
      {
        "check": "dual_boa_qk_pv",
        "expected": True,
        "actual": boa_active_total > 0,
        "pass": boa_active_total > 0,
      }
    )
  if exp.get("tiled_overlap"):
    # tiled overlap: BOA and MFE both active, proving the double-buffer
    # overlap issued MFE loads while BOA was computing
    boa_active_total = engine_active.get("BOA", 0)
    mfe_active_total = engine_active.get("MFE", 0)
    overlap_ok = boa_active_total > 0 and mfe_active_total > 0
    checks.append({"check": "tiled_overlap", "expected": True, "actual": overlap_ok, "pass": overlap_ok})

  if exp.get("multi_stage_group_io"):
    # Multiple DMA prefetch + dispatch stages prove the group-level
    # task was unrolled into stages (the trace test verifies actual
    # temporal overlap; this check only confirms staged structure).
    dma_prefetch_count = result.pmu.events.get("tgs_dma_prefetch", 0)
    dispatch_count = result.pmu.events.get("tgs_dispatch_role", 0)
    ok = dma_prefetch_count >= 2 and dispatch_count >= 2
    checks.append(
      {
        "check": "multi_stage_group_io",
        "expected": True,
        "actual": ok,
        "detail": (f"dma_prefetch={dma_prefetch_count}, dispatch={dispatch_count}"),
        "pass": ok,
      }
    )
  gather_requests = result.pmu.events.get("gather_requests", 0)
  if gather_requests > 0:
    l1_hits = result.pmu.events.get("gather_l1_hits", 0)
    l2_hits = result.pmu.events.get("gather_l2_hits", 0)
    hbm_misses = result.pmu.events.get("gather_hbm_misses", 0)
    conserved = gather_requests == l1_hits + l2_hits + hbm_misses
    checks.append(
      {
        "check": "gather_request_conservation",
        "expected": True,
        "actual": {
          "requests": gather_requests,
          "l1_hits": l1_hits,
          "l2_hits": l2_hits,
          "hbm_misses": hbm_misses,
        },
        "pass": conserved,
      }
    )

    snapshot = result.group_snapshot
    memory = snapshot.get("memory", {})
    mshr = memory.get("mshr", {})
    l2_mshr = mshr.get("l2", {})
    l1_mshrs = mshr.get("l1", {}).values()
    transfers = memory.get("transfers", {})
    stages = transfers.get("stages", {}).values()
    l2_allocator = memory.get("l2") or {}
    l1_allocators = [tile_memory.get("allocator") or {} for tile_memory in memory.get("l1", {}).values()]
    gather_jobs = [tile.get("gather_active_jobs", 0) for tile in snapshot.get("tiles", [])]
    zero_leak = (
      l2_mshr.get("active", 0) == 0
      and l2_mshr.get("callbacks", 0) == 0
      and all(item.get("active", 0) == 0 for item in l1_mshrs)
      and all(item.get("callbacks", 0) == 0 for item in mshr.get("l1", {}).values())
      and all(count == 0 for count in gather_jobs)
      and transfers.get("inflight", 0) == 0
      and all(stage.get("busy_resources", 0) == 0 and stage.get("outstanding", 0) == 0 for stage in stages)
      and l2_allocator.get("live_allocations", 0) == 0
      and l2_allocator.get("pending_release", 0) == 0
      and all(
        allocator.get("live_allocations", 0) == 0 and allocator.get("pending_release", 0) == 0
        for allocator in l1_allocators
      )
    )
    checks.append({"check": "gather_zero_leak", "expected": True, "actual": zero_leak, "pass": zero_leak})
  return checks


def _append_structured(lines: list[str], value, indent: int) -> None:
  prefix = " " * indent
  if isinstance(value, Mapping):
    if not value:
      lines.append(f"{prefix}(none)")
      return
    for key, item in value.items():
      if isinstance(item, (Mapping, list, tuple)):
        lines.append(f"{prefix}{key}:")
        _append_structured(lines, item, indent + 2)
      else:
        lines.append(f"{prefix}{key}: {item}")
    return
  if isinstance(value, (list, tuple)):
    if not value:
      lines.append(f"{prefix}(none)")
      return
    for item in value:
      if isinstance(item, Mapping):
        lines.append(f"{prefix}-")
        _append_structured(lines, item, indent + 2)
      else:
        lines.append(f"{prefix}- {item}")
    return
  lines.append(f"{prefix}{value}")


def _append_report_section(lines: list[str], title: str, value: Mapping) -> None:
  lines.append(f"  {title}:")
  _append_structured(lines, value, 4)
  lines.append("")


def report_to_text(r: WorkloadReport) -> str:
  lines = []
  lines.append("=" * 72)
  lines.append(f"Workload: {r.name}")
  lines.append("-" * 72)
  lines.append(f"  {r.description}")
  lines.append("")
  lines.append(f"  Cycles:          {r.cycles}")
  lines.append(
    "  Makespan:        "
    f"{r.makespan.get('span_cycles', r.cycles)} cycles "
    f"(cycle {r.makespan.get('origin_cycle', 0)}"
    f" → {r.makespan.get('terminal_cycle', r.cycles)})"
  )
  lines.append(f"  Completed:       {r.completed}  ({r.reason})")
  lines.append(f"  Utilization:     {r.utilization:.1%}")
  lines.append(f"  Credit inv OK:   {r.credit_invariant_ok}")
  if r.gather_fidelity is not None:
    lines.append(f"  Gather fidelity: {r.gather_fidelity}")
  if r.memory:
    lines.append("  Memory peaks:")
    for key, value in sorted(r.memory.items()):
      if isinstance(value, dict):
        inner = ", ".join(f"tile{tid}={v}" for tid, v in sorted(value.items()))
        lines.append(f"    {key:<30}: {inner}")
      else:
        lines.append(f"    {key:<30}: {value}")
  lines.append("")
  _append_report_section(lines, "Configured/effective resources", r.resources)
  if r.scheduler:
    _append_report_section(lines, "Group scheduler", r.scheduler)
  if r.device:
    _append_report_section(lines, "CPU device controller", r.device)
  if r.request_timing:
    _append_report_section(lines, "CPU request timing", r.request_timing)
  lines.append("  Engine active cycles:")
  for eng, c in r.engine_active.items():
    lines.append(f"    {eng:<4}: {c:>8}  ({c / max(r.cycles * r.num_tiles, 1):.1%})")
  lines.append("")
  lines.append("  Stall breakdown (primary owner):")
  if r.stall_breakdown:
    for label, c in sorted(r.stall_breakdown.items(), key=lambda x: -x[1]):
      lines.append(f"    {label:<28}: {c:>8}")
  else:
    lines.append("    (no stalls)")
  lines.append("")
  lines.append("  Stream counters:")
  if r.stream_counters:
    for k, v in sorted(r.stream_counters.items()):
      lines.append(f"    {k:<28}: {v:>8}")
  else:
    lines.append("    (none)")
  lines.append("")
  lines.append("  Events:")
  for k, v in sorted(r.events.items()):
    lines.append(f"    {k:<28}: {v:>8}")
  lines.append("")
  lines.append("  Checks:")
  all_pass = True
  for ch in r.checks:
    ok = ch.get("pass", False)
    mark = "PASS" if ok else "FAIL"
    if not ok:
      all_pass = False
    detail = ""
    if "expected" in ch:
      detail = f"  expected={ch['expected']} actual={ch['actual']}"
    elif "expected_min" in ch:
      detail = f"  min={ch['expected_min']} actual={ch['actual']}"
    elif "expected_max" in ch:
      detail = f"  max={ch['expected_max']} actual={ch['actual']}"
    lines.append(f"    [{mark}] {ch['check']}{detail}")
  lines.append("")
  lines.append(f"  Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
  lines.append("=" * 72)
  return "\n".join(lines)


def report_to_json(r: WorkloadReport) -> str:
  return json.dumps(
    {
      "name": r.name,
      "description": r.description,
      "cycles": r.cycles,
      "completed": r.completed,
      "reason": r.reason,
      "utilization": r.utilization,
      "credit_invariant_ok": r.credit_invariant_ok,
      "gather_fidelity": r.gather_fidelity,
      "engine_active": r.engine_active,
      "stall_breakdown": r.stall_breakdown,
      "stream_counters": r.stream_counters,
      "events": r.events,
      "memory": r.memory,
      "scheduler": r.scheduler,
      "device": r.device,
      "resources": r.resources,
      "makespan": r.makespan,
      "request_timing": r.request_timing,
      "checks": r.checks,
    },
    indent=2,
  )
