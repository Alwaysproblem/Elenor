"""Summarize Transformer workload reports into the guide's metric tables.

Usage:
  python examples/generators/analyze_transformer.py \
      --prefill /tmp/xf/pre_pipe.report.json /tmp/xf/pre_base.report.json \
      --decode /tmp/xf/dec_pipe.report.json /tmp/xf/dec_base.report.json \
      --decode-trace /tmp/xf/dec_pipe.trace.json

Prefill KPIs (guide section 10): total cycles, per-engine active cycles,
HBM/L2/L1 bytes, stall categories, peak concurrent contexts, and
``useful_matrix_utilization = useful_MACs / (BOA_peak_per_cycle * cycles)``.

Decode KPIs (guide sections 21-22): KV-block initiation interval from the
per-block PV BOA completions in the trace, the analytic
``ideal_resource_II`` lower bound, ``pipeline_efficiency``, and
``effective_kv_bandwidth = useful_KV_bytes / steady_state_time``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BOA_PEAK_MACS_PER_TILE = 4 * 16 * 16  # 4 OPA x 16x16 outer product
PREFILL_USEFUL_MACS = (
  512 * 1024 * 1536  # QKV projection
  + 4 * 16 * 4 * (128 * 128 * 64 + 128 * 64 * 128)  # attention, per tile x 4 tiles
  + 512 * 1024 * 1024  # output projection
)
DECODE_KV_BYTES = 2048 * 4 * 64 * 2 * 2  # K 1 MiB + V 1 MiB, BF16
DECODE_BLOCKS = 8


def load_report(path: str) -> dict:
  data = json.loads(Path(path).read_text(encoding="utf-8"))
  return data[0] if isinstance(data, list) else data


def traffic_bytes(report: dict, *keys: str) -> int:
  return sum(report["traffic"][key] for key in keys)


def prefill_table(piped_path: str, base_path: str) -> dict:
  piped = load_report(piped_path)
  base = load_report(base_path)
  peak_per_cycle = BOA_PEAK_MACS_PER_TILE * 4

  rows = [
    ("total cycles", base["cycles"], piped["cycles"]),
    ("speedup", 1.0, round(base["cycles"] / piped["cycles"], 3)),
    (
      "useful BOA utilization",
      f"{PREFILL_USEFUL_MACS / (peak_per_cycle * base['cycles']):.1%}",
      f"{PREFILL_USEFUL_MACS / (peak_per_cycle * piped['cycles']):.1%}",
    ),
    ("BOA busy (tile-cycles)", base["engine_active"]["BOA"], piped["engine_active"]["BOA"]),
    ("EVU busy (tile-cycles)", base["engine_active"]["EVU"], piped["engine_active"]["EVU"]),
    ("MFE busy (tile-cycles)", base["engine_active"]["MFE"], piped["engine_active"]["MFE"]),
    ("HBM bytes", traffic_bytes(base, "hbm_read_bytes", "hbm_write_bytes"),
     traffic_bytes(piped, "hbm_read_bytes", "hbm_write_bytes")),
    ("L2 bytes", traffic_bytes(base, "l2_read_bytes", "l2_write_bytes"),
     traffic_bytes(piped, "l2_read_bytes", "l2_write_bytes")),
    ("memory stall", base["stall_categories"]["memory_stall_cycles"],
     piped["stall_categories"]["memory_stall_cycles"]),
    # Older reports double-count engine-queue retries here; derive the
    # disjoint dependency stall from stall_breakdown instead.
    ("dependency stall", base["stall_breakdown"].get("engine_wait_event", 0),
     piped["stall_breakdown"].get("engine_wait_event", 0)),
    ("context peak", base["scheduler"].get("active_context_peak"),
     piped["scheduler"].get("active_context_peak")),
  ]
  return {"baseline": base, "optimized": piped, "rows": rows}


def ideal_decode_ii(kv_block: int) -> int:
  """max() over per-resource service times for one KV block (analytic)."""
  hbm_bytes_per_block = 2 * (4 * kv_block * 64 * 2)  # K + V, all four heads
  hbm = hbm_bytes_per_block / (819.2e9 / 1e9)  # 819.2 B/cycle
  dma_channels = 2  # num_dma_channels used by the benchmark runs
  dma = hbm_bytes_per_block / dma_channels / (256.0e9 / 1e9)
  l2 = hbm_bytes_per_block / (16 * 64.0e9 / 1e9)  # 16 banks x 64 B/cycle
  mfe = (hbm_bytes_per_block / 4 + 512 + 4224 / 4) / (256.0e9 / 1e9)  # per-tile share
  # One BOA per QK and one per PV; each is 4*kv_block*64 MACs at 1024
  # MACs/cycle plus the 4-cycle launch overhead.
  boa = (4 * kv_block * 64 / 1024 + 4) * 2
  evu = 3 + (4 * kv_block + 4 * 66) / 64
  return int(max(hbm, dma, l2, mfe, boa, evu))


def decode_metrics(report_path: str, trace_path: str | None, kv_block: int = 256) -> dict:
  report = load_report(report_path)
  blocks = 2048 // kv_block
  out = {
    "cycles": report["cycles"],
    "engine_active": report["engine_active"],
    "traffic": report["traffic"],
    "stall_categories": report["stall_categories"],
    "stall_breakdown": report["stall_breakdown"],
    "active_context_peak": report["scheduler"].get("active_context_peak"),
  }
  if trace_path and Path(trace_path).exists():
    events = json.loads(Path(trace_path).read_text(encoding="utf-8"))["traceEvents"]
    pv = sorted(
      (
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("name") == "BOA:matmul"
        and e.get("args", {}).get("local_event_id") == "pv_boa"
      ),
      key=lambda e: e["ts"],
    )
    first = min(pv, key=lambda e: e["ts"] + e["dur"])  # first PV completion
    last = pv[-1]
    # 1 ts unit == 1000 cycles at 1 GHz (Tracer.cycle_to_us).
    t0 = (first["ts"] + first["dur"]) * 1000.0
    t7 = (last["ts"] + last["dur"]) * 1000.0
    actual_ii = (t7 - t0) / (blocks - 1)
    ideal = ideal_decode_ii(kv_block)
    out.update(
      kv_block_actual_ii=round(actual_ii, 1),
      kv_block_reference_ii=ideal,
      pipeline_efficiency=round(ideal / actual_ii, 3),
      effective_kv_bandwidth_gbs=round(
        DECODE_KV_BYTES / ((t7 - t0) + actual_ii) * 1e9 / 1e9, 1
      ),
    )
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prefill", nargs=2, metavar=("OPTIMIZED", "BASELINE"))
  parser.add_argument("--decode", nargs=2, metavar=("OPTIMIZED", "BASELINE"))
  parser.add_argument("--decode-trace", default=None)
  parser.add_argument(
    "--kv-block",
    type=int,
    default=256,
    help="KV block size of the decode run; scales the II divisor and the "
    "analytic reference II (2048 must divide evenly; default 256)",
  )
  args = parser.parse_args()

  if args.prefill:
    table = prefill_table(*args.prefill)
    print("== Prefill (guide section 30) ==")
    print(f"{'Metric':<26}{'Baseline':>16}{'Optimized':>16}")
    for label, base, pipe in table["rows"]:
      print(f"{label:<26}{base!s:>16}{pipe!s:>16}")
  if args.decode:
    if args.kv_block <= 0 or 2048 % args.kv_block:
      parser.error("--kv-block must be a positive divisor of 2048")
    pipe = decode_metrics(args.decode[0], args.decode_trace, kv_block=args.kv_block)
    base = decode_metrics(args.decode[1], None, kv_block=args.kv_block)
    print("\n== Decode (guide section 30) ==")
    print(f"{'Metric':<26}{'Baseline':>16}{'Pipelined':>16}")
    print(f"{'token latency (cycles)':<26}{base['cycles']:>16}{pipe['cycles']:>16}")
    if "kv_block_actual_ii" in pipe:
      print(f"{'KV block II':<26}{'-':>16}{pipe['kv_block_actual_ii']:>16}")
      print(f"{'KV reference II':<26}{'-':>16}{pipe['kv_block_reference_ii']:>16}")
      print(f"{'pipeline efficiency':<26}{'-':>16}{pipe['pipeline_efficiency']:>16}")
      print(f"{'effective KV BW (B/cyc)':<26}{'-':>16}{pipe['effective_kv_bandwidth_gbs']:>16}")
    print(f"{'MFE busy':<26}{base['engine_active']['MFE']:>16}{pipe['engine_active']['MFE']:>16}")
    print(f"{'BOA busy':<26}{base['engine_active']['BOA']:>16}{pipe['engine_active']['BOA']:>16}")
    print(f"{'EVU busy':<26}{base['engine_active']['EVU']:>16}{pipe['engine_active']['EVU']:>16}")
    for label in ("HBM bytes", "L2 bytes"):
      read_key = "hbm_read_bytes" if label == "HBM bytes" else "l2_read_bytes"
      write_key = "hbm_write_bytes" if label == "HBM bytes" else "l2_write_bytes"
      print(
        f"{label:<26}"
        f"{traffic_bytes(base, read_key, write_key):>16}"
        f"{traffic_bytes(pipe, read_key, write_key):>16}"
      )
    print(
      f"{'memory stall':<26}"
      f"{base['stall_categories']['memory_stall_cycles']:>16}"
      f"{pipe['stall_categories']['memory_stall_cycles']:>16}"
    )
    # Reports generated before the stall-category fix double-counted
    # engine-queue retries inside dependency_stall_cycles; derive it from
    # the disjoint stall_breakdown instead.
    def dep_stall(report: dict) -> int:
      return report["stall_breakdown"].get("engine_wait_event", 0)

    print(f"{'dependency stall':<26}{dep_stall(base):>16}{dep_stall(pipe):>16}")


if __name__ == "__main__":
  main()
