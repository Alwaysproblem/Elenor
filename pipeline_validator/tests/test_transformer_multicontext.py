"""Actual single-request parallelism and safe handoffs in the multicontext examples."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from math import prod
from pathlib import Path

import pytest

from pipeline_validator.compiled_program import WorkloadInfo
from pipeline_validator.compiler import compile_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.dialects.elenor import NestContextOp, NexusProgramOp
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.report import build_report
from pipeline_validator.simulator import Simulator
from pipeline_validator.workload_ir import load_workload_ir

WORKLOADS = Path(__file__).resolve().parents[2] / "examples/workloads"


@pytest.fixture(scope="module", params=("prefill_attention", "decode_kv"))
def multicontext_run(request):
  name = f"transformer_{request.param}_multicontext"
  module = load_workload_ir(WORKLOADS / f"{name}.mlir")
  entry = next(op for op in module.ops if isinstance(op, NexusProgramOp))
  context = next(op for op in module.ops if isinstance(op, NestContextOp))
  context_count = context.resource_contract.to_contract().requested_contexts_per_tile
  bindings = {}
  address = 0x1000000
  for arg in entry.body.block.args:
    shape = [dim.value.data for dim in arg.type.dims]
    size = prod(shape) * {"bf16": 2, "f32": 4}[arg.type.dtype.data]
    offset = 64 if arg.name_hint == "V_CACHE_mc" else 0
    bindings[arg.name_hint] = GlobalBinding(arg.name_hint, address + offset, size, "rw")
    address += ((size + offset + 0xFFFFF) // 0x100000) * 0x100000
  hw = replace(HardwareConfig(), num_dma_channels=2, hbm_fixed_latency_cycles=10)
  config = SimConfig(context_count=context_count, memory_trace=True, max_cycles=2_000_000)
  artifact = compile_program(module, hw, config, source_name=name)
  simulator = Simulator(hw, config, enable_tracer=True)
  result = simulator.run(load_program(artifact, hw, config, actual_bindings=bindings))
  assert result.completed, result.reason
  assert simulator.tracer is not None
  simulator.tracer.assert_well_formed()
  events = json.loads(simulator.tracer.to_chrome_json())["traceEvents"]
  report = build_report(WorkloadInfo(name, "single-request multicontext regression", {}), result)
  return request.param, result, report, events, context_count


def test_independent_tasks_overlap_without_leaking(multicontext_run):
  _, result, report, events, context_count = multicontext_run
  assert result.device_snapshot["counters"]["submitted"] == 1
  zero_leak = next(check for check in report.checks if check["check"] == "arena_zero_leak")
  assert zero_leak["pass"], zero_leak
  deltas = defaultdict(lambda: defaultdict(int))
  for event in events:
    if event.get("name") not in ("task_lease_acquire", "task_lease_release"):
      continue
    args = event["args"]
    delta = 1 if event["name"] == "task_lease_acquire" else -1
    deltas[args["tile_id"]][event["ts"]] += delta
  assert set(deltas) == {0, 1, 2, 3}
  for tile, points in deltas.items():
    active = peak = 0
    for _, delta in sorted(points.items()):
      active += delta
      assert 0 <= active <= context_count, (tile, active)
      peak = max(peak, active)
    assert active == 0
    assert peak >= 2, f"tile {tile} never admitted concurrent Tasks"


def test_compute_overlaps_other_contexts_loads(multicontext_run):
  _, _, _, events, _context_count = multicontext_run
  compute = [
    event for event in events if event.get("ph") == "X" and event["name"].startswith(("BOA:", "EVU:"))
  ]
  loads = [event for event in events if event.get("ph") == "X" and event["name"] == "MFE:load"]
  overlapping_tiles = set()
  for load in loads:
    args = load["args"]
    # MFE trace encodes the tile-local UCE slot in ctxN:event; BOA/EVU
    # additionally provide the same identity in args.ctx_id.
    slot = int(args["event_id"].split(":", 1)[0].removeprefix("ctx"))
    if any(
      op["args"]["tile_id"] == args["tile_id"]
      and op["args"]["ctx_id"] != slot
      and op["ts"] < load["ts"] + load["dur"]
      and load["ts"] < op["ts"] + op["dur"]
      for op in compute
    ):
      overlapping_tiles.add(args["tile_id"])
  assert overlapping_tiles == {0, 1, 2, 3}


def test_same_logical_work_and_accounted_traffic(multicontext_run):
  kind, _, report, events, _context_count = multicontext_run
  matrix_flops = sum(
    event["args"]["ops"] for event in events if event.get("ph") == "X" and event["name"].startswith("BOA:")
  )
  if kind == "prefill_attention":
    assert matrix_flops == 2 * (512 * 1024 * 1536 + 2 * 16 * 512 * 512 * 64 + 512 * 1024 * 1024)
    assert report.traffic["hbm_read_bytes"] == 6291456
    assert report.traffic["hbm_write_bytes"] == 1048576
  else:
    assert matrix_flops == 2 * (2 * 16 * 2048 * 64 + 2 * 4 * 1024 * 64)
    history_reads = [
      e
      for e in events
      if e.get("ph") == "X"
      and e["name"] == "hbm_read"
      and any(tag in e["args"]["transaction_id"] for tag in ("pre_k_block_", "pre_v_block_"))
    ]
    assert sum(e["args"]["bytes"] for e in history_reads) == 2097152
    # Packet gaps are address padding only and must never become DMA payload.
    assert all(e["args"]["bytes"] == 4 * 256 * 64 * 2 for e in history_reads)
    assert report.traffic["hbm_read_bytes"] == 2097152 + 2048 + 16896 + 2048 + 2 * 524288
    assert report.traffic["hbm_write_bytes"] == 4096 + 2 * 512


def test_merge_and_refill_obey_partition_phases(multicontext_run):
  kind, _, _, events, _context_count = multicontext_run
  if kind != "decode_kv":
    pytest.skip("split-KV partition handoff is decode-specific")
  phases = {
    event["args"]["event_id"].split("_attn_block_", 1)[1]: event["ts"]
    for event in events
    if event.get("name") == "phase_aggregate" and "_attn_block_" in event["args"]["event_id"]
  }
  for block in (1, 3, 5, 7):
    for tensor in ("k", "v"):
      prefetch = next(
        event
        for event in events
        if event.get("ph") == "X"
        and event["name"] == "hbm_read"
        and f"pre_{tensor}_block_{block}_event" in event["args"]["transaction_id"]
      )
      assert prefetch["ts"] >= phases[f"{block - 1}_inrel"]
  final_partition_ready = max(phases[f"{block}_out"] for block in (1, 3, 5, 7))
  merge = [
    e
    for e in events
    if e.get("ph") == "X"
    and e["name"].startswith("EVU:")
    and e.get("args", {}).get("program") == "decode_stable_partition_merge"
  ]
  assert len(merge) == 4
  assert min(e["ts"] for e in merge) >= final_partition_ready
