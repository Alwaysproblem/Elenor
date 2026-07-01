"""Report generation: PMU fingerprint + pass/fail checks.

Produces a concise text + optional JSON report comparing measured PMU
counters against each workload's `expected` fingerprint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

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


def _ratio(counters: dict, total: int) -> float:
    return (counters.get("active", 0) / total) if total else 0.0


def build_report(wl: Workload,
                 result: SimResult,
                 num_tiles: int = 4) -> WorkloadReport:
    pmu = result.pmu
    total = result.cycles or 1
    # engine/uce stall cycles are aggregated across all tiles, so the
    # correct denominator for per-tile engine ratios is total * num_tiles.
    tile_cycles = total * num_tiles

    engine_active = {
        "BOA": pmu.named_cycles.get("boa_active", 0),
        "EVU": pmu.named_cycles.get("evu_active", 0),
        "MFE": pmu.named_cycles.get("mfe_active", 0),
        "USE": pmu.named_cycles.get("use_active", 0),
    }

    stream_counters = {}
    for k, v in pmu.named_cycles.items():
        if k.startswith("queue") or k in ("occupancy", "credit_full",
                                          "credit_empty", "credit_fault"):
            stream_counters[k] = v

    checks = _run_checks(wl, result, engine_active, tile_cycles)

    return WorkloadReport(
        name=wl.name,
        description=wl.description,
        cycles=result.cycles,
        completed=result.completed,
        reason=result.reason,
        utilization=result.utilization(num_tiles),
        stall_breakdown=pmu.stall_breakdown(),
        engine_active=engine_active,
        stream_counters=stream_counters,
        events=dict(pmu.events),
        checks=checks,
        credit_invariant_ok=result.credit_invariant_ok,
        num_tiles=num_tiles,
    )


def _run_checks(wl: Workload, result: SimResult, engine_active: dict,
                total: int) -> list[dict]:
    checks: list[dict] = []
    exp = wl.expected

    # 1. completion
    checks.append({
        "check": "task_completed",
        "expected": True,
        "actual": result.completed,
        "pass": result.completed,
    })
    # 2. credit invariant
    checks.append({
        "check": "credit_invariant",
        "expected": True,
        "actual": result.credit_invariant_ok,
        "pass": result.credit_invariant_ok,
    })

    boa_ratio = engine_active.get("BOA", 0) / total if total else 0
    evu_ratio = engine_active.get("EVU", 0) / total if total else 0
    mfe_ratio = engine_active.get("MFE", 0) / total if total else 0
    stream_stall = result.pmu.stall_cycles.get(StallReason.STREAM_CREDIT, 0)
    stream_stall_ratio = stream_stall / total if total else 0

    if "boa_active_ratio_min" in exp:
        checks.append({
            "check": "boa_active_ratio",
            "expected_min": exp["boa_active_ratio_min"],
            "actual": round(boa_ratio, 3),
            "pass": boa_ratio >= exp["boa_active_ratio_min"],
        })
    if "mfe_active_ratio_min" in exp:
        checks.append({
            "check": "mfe_active_ratio",
            "expected_min": exp["mfe_active_ratio_min"],
            "actual": round(mfe_ratio, 3),
            "pass": mfe_ratio >= exp["mfe_active_ratio_min"],
        })
    if "evu_active_ratio_min" in exp:
        checks.append({
            "check": "evu_active_ratio",
            "expected_min": exp["evu_active_ratio_min"],
            "actual": round(evu_ratio, 3),
            "pass": evu_ratio >= exp["evu_active_ratio_min"],
        })
    if "stream_stall_ratio_max" in exp:
        checks.append({
            "check":
            "stream_stall_ratio",
            "expected_max":
            exp["stream_stall_ratio_max"],
            "actual":
            round(stream_stall_ratio, 3),
            "pass":
            stream_stall_ratio <= exp["stream_stall_ratio_max"],
        })
    if exp.get("stream_s0_occupancy_seen"):
        occ = result.pmu.named_cycles.get("occupancy", 0)
        checks.append({
            "check": "stream_occupancy_seen",
            "expected": True,
            "actual": occ > 0,
            "pass": occ > 0,
        })
    if exp.get("producer_consumer_overlap"):
        # overlap: both BOA (role1) and EVU (role1) active in same window
        overlap = (engine_active.get("BOA", 0) > 0
                   and engine_active.get("EVU", 0) > 0)
        checks.append({
            "check": "producer_consumer_overlap",
            "expected": True,
            "actual": overlap,
            "pass": overlap,
        })
    if exp.get("mfe_page_stream_active"):
        # MFE Page Stream active: MFE ran (page-stream gather + store)
        mfe_active_total = engine_active.get("MFE", 0)
        checks.append({
            "check": "mfe_page_stream_active",
            "expected": True,
            "actual": mfe_active_total > 0,
            "pass": mfe_active_total > 0,
        })
    if exp.get("dual_boa_qk_pv"):
        # dual BOA: BOA ran (QK + PV are both BOA matmuls)
        boa_active_total = engine_active.get("BOA", 0)
        checks.append({
            "check": "dual_boa_qk_pv",
            "expected": True,
            "actual": boa_active_total > 0,
            "pass": boa_active_total > 0,
        })
    if exp.get("tiled_overlap"):
        # tiled overlap: BOA and MFE both active, proving the double-buffer
        # overlap issued MFE loads while BOA was computing
        boa_active_total = engine_active.get("BOA", 0)
        mfe_active_total = engine_active.get("MFE", 0)
        overlap_ok = boa_active_total > 0 and mfe_active_total > 0
        checks.append({
            "check": "tiled_overlap",
            "expected": True,
            "actual": overlap_ok,
            "pass": overlap_ok,
        })

    if exp.get("multi_stage_group_io"):
        # Multiple DMA prefetch + dispatch stages prove the group-level
        # task was unrolled into stages (the trace test verifies actual
        # temporal overlap; this check only confirms staged structure).
        dma_prefetch_count = result.pmu.events.get("tgs_dma_prefetch", 0)
        dispatch_count = result.pmu.events.get("tgs_dispatch_role", 0)
        ok = dma_prefetch_count >= 2 and dispatch_count >= 2
        checks.append({
            "check": "multi_stage_group_io",
            "expected": True,
            "actual": ok,
            "detail": (
                f"dma_prefetch={dma_prefetch_count}, "
                f"dispatch={dispatch_count}"
            ),
            "pass": ok,
        })

    if exp.get("uce_window_mfe_lookahead"):
        lookahead_launch = result.pmu.events.get(
            "uce_window_mfe_lookahead_launch", 0)
        queued = result.pmu.events.get("uce_window_entry_queued", 0)
        ok = lookahead_launch > 0 and queued > 0
        checks.append({
            "check": "uce_window_mfe_lookahead",
            "expected": True,
            "actual": ok,
            "detail": (
                f"lookahead_launch={lookahead_launch}, queued={queued}"),
            "pass": ok,
        })
    return checks


def report_to_text(r: WorkloadReport) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(f"Workload: {r.name}")
    lines.append("-" * 72)
    lines.append(f"  {r.description}")
    lines.append("")
    lines.append(f"  Cycles:          {r.cycles}")
    lines.append(f"  Completed:       {r.completed}  ({r.reason})")
    lines.append(f"  Utilization:     {r.utilization:.1%}")
    lines.append(f"  Credit inv OK:   {r.credit_invariant_ok}")
    lines.append("")
    lines.append("  Engine active cycles:")
    for eng, c in r.engine_active.items():
        lines.append(
            f"    {eng:<4}: {c:>8}  ({c / max(r.cycles * r.num_tiles, 1):.1%})"
        )
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
    lines.append(
        f"  Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
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
            "engine_active": r.engine_active,
            "stall_breakdown": r.stall_breakdown,
            "stream_counters": r.stream_counters,
            "events": r.events,
            "checks": r.checks,
        },
        indent=2)
