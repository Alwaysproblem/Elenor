"""Transformer prefill/decode workload acceptance tests (guide section 29).

The committed workload files are the test fixtures:

- ``examples/workloads/transformer_prefill_attention_pipeline.mlir`` and its
  ``_baseline`` variant,
- ``examples/workloads/transformer_decode_kv_pipeline.mlir`` and its
  ``_baseline`` variant.

All tests run the same compile -> load -> simulate pipeline as the CLI and
only assert observables (report counters, trace slices); none of them
inspect simulator internals.  Timing model only -- nothing here claims
numerical correctness of the attention computation.
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

from pipeline_validator.compiler import compile_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.simulator import Simulator
from pipeline_validator.workload_ir import load_workload_ir

REPO = Path(__file__).resolve().parents[2]
WORKLOADS = REPO / "examples" / "workloads"

PREFILL_BINDINGS = {
  "X": GlobalBinding("X", 0x1000000, 1048576, "r"),
  "WQ": GlobalBinding("WQ", 0x2000000, 2097152, "r"),
  "WK": GlobalBinding("WK", 0x3000000, 524288, "r"),
  "WV": GlobalBinding("WV", 0x3100000, 524288, "r"),
  "WO": GlobalBinding("WO", 0x4000000, 2097152, "r"),
  "OUT": GlobalBinding("OUT", 0x5000000, 1048576, "w"),
}

# Minimum useful KV scan traffic per request: K 1 MiB + V 1 MiB (BF16).
DECODE_KV_BYTES = 2048 * 4 * 64 * 2 * 2
DECODE_BINDINGS = {
  "K_CACHE": GlobalBinding("K_CACHE", 0x1000000, 1048576, "r"),
  "V_CACHE": GlobalBinding("V_CACHE", 0x2000000, 1048576, "r"),
  "Q_IN": GlobalBinding("Q_IN", 0x3000000, 2048, "r"),
  "S_INIT": GlobalBinding("S_INIT", 0x3001000, 4224, "rw"),
  "OUT": GlobalBinding("OUT", 0x3010000, 4096, "w"),
  "H_T": GlobalBinding("H_T", 0x3011000, 2048, "r"),
  "WK_A": GlobalBinding("WK_A", 0x3012000, 524288, "r"),
  "WV_A": GlobalBinding("WV_A", 0x3092000, 524288, "r"),
  "K_APPEND": GlobalBinding("K_APPEND", 0x3200000, 512, "w"),
  "V_APPEND": GlobalBinding("V_APPEND", 0x3201000, 512, "w"),
}


def make_sim(hw_overrides: dict | None = None, **sim) -> Simulator:
  hw = replace(HardwareConfig(), num_dma_channels=2, hbm_fixed_latency_cycles=10, **(hw_overrides or {}))
  config = SimConfig(fidelity="full_memory", max_cycles=1000000, **sim)
  return Simulator(hw, config, enable_tracer=True)


def run_file(name: str, bindings: dict, simulator: Simulator):
  module = load_workload_ir(WORKLOADS / name)
  artifact = compile_program(module, simulator.hw, simulator.sim, source_name=name)
  loaded = load_program(artifact, simulator.hw, simulator.sim, actual_bindings=bindings)
  return simulator.run(loaded)


def chrome_events(simulator: Simulator) -> list[dict]:
  assert simulator.tracer is not None
  return json.loads(simulator.tracer.to_chrome_json())["traceEvents"]


@lru_cache(maxsize=1)
def _decode_results() -> tuple[object, object, Simulator, Simulator]:
  pipe_sim = make_sim(memory_trace=True)
  base_sim = make_sim(memory_trace=True)
  return (
    run_file("transformer_decode_kv_pipeline.mlir", DECODE_BINDINGS, pipe_sim),
    run_file("transformer_decode_kv_baseline.mlir", DECODE_BINDINGS, base_sim),
    pipe_sim,
    base_sim,
  )


@pytest.fixture(scope="module")
def decode_results():
  return _decode_results()


@pytest.fixture(scope="module")
def prefill_results():
  pipe_sim = make_sim(memory_trace=True)
  base_sim = make_sim(memory_trace=True)
  piped = run_file("transformer_prefill_attention_pipeline.mlir", PREFILL_BINDINGS, pipe_sim)
  baseline = run_file("transformer_prefill_attention_baseline.mlir", PREFILL_BINDINGS, base_sim)
  return piped, baseline, pipe_sim, base_sim


# ---------------------------------------------------------------------------
# Compile + run acceptance
# ---------------------------------------------------------------------------


def test_prefill_compiles_and_runs(prefill_results):
  piped, baseline, _, _ = prefill_results
  assert piped.completed and piped.reason == "model complete"
  assert baseline.completed and baseline.reason == "model complete"
  # Optimized prefetch/compute overlap must not be slower than the fenced
  # baseline; the guard is loose because scheduling noise is model-visible.
  assert piped.cycles <= baseline.cycles


def test_decode_compiles_and_runs(decode_results):
  piped, baseline, _, _ = decode_results
  assert piped.completed and piped.reason == "model complete"
  assert baseline.completed and baseline.reason == "model complete"


# ---------------------------------------------------------------------------
# Decode ping/pong pipelining
# ---------------------------------------------------------------------------


def _engine_slices(events: list[dict], name_prefix: str) -> list[dict]:
  out = []
  for event in events:
    if event.get("ph") != "X":
      continue
    name = event.get("name", "")
    if name.startswith(name_prefix):
      out.append(event)
  return out


def _args(event: dict) -> dict:
  return event.get("args") or {}


def test_decode_kv_pingpong_overlap(decode_results):
  """Pipelined: context-level KV prefetches overlap tile attention compute.

  Dispatch b+1 waits for dispatch b's ``output_ready``, so tile-local
  loads cannot overlap the previous block.  The pipeline overlap is the
  HBM/Global-DMA prefetch of block b+2 running while blocks b and b+1
  compute; the observable is hbm_read prefetch legs intersecting BOA
  slices.
  """
  piped, baseline, pipe_sim, _ = decode_results
  assert piped.cycles < baseline.cycles, "ping/pong pipelining must beat the fenced baseline"

  events = chrome_events(pipe_sim)
  boas = [e for e in _engine_slices(events, "BOA:") if _args(e).get("local_event_id") == "pv_boa"]
  prefetches = [
    e
    for e in _engine_slices(events, "hbm_read")
    if _args(e).get("op") == "prefetch"
    and ("pre_k" in str(_args(e).get("transaction_id")) or "pre_v" in str(_args(e).get("transaction_id")))
  ]
  assert boas and prefetches, "expected PV BOA slices and KV prefetch legs in the trace"
  overlapped = 0
  for boa in boas:
    if any(e["ts"] < boa["ts"] + boa["dur"] and e["ts"] + e["dur"] > boa["ts"] for e in prefetches):
      overlapped += 1
  assert overlapped >= 4, (
    f"expected KV prefetch to overlap attention compute on most blocks, saw {overlapped}/{len(boas)}"
  )


def test_decode_no_buffer_overwrite_before_input_released():
  """L2 ping/pong refill must be gated on the two-older block's inrel.

  Checked on the parsed source IR: for every KV block b >= 2, both the K
  and V prefetches of block b depend on the ``input_released`` result of
  dispatch b-2 (SSA identity, not name matching).  The runtime enforces
  the same edges on the compiled artifact, so a passing compile + run
  (test_decode_compiles_and_runs) proves the protocol executes; this
  test pins the dependency structure that makes the gate correct.
  """
  from pipeline_validator.dialects.elenor import NestDispatchOp, NestPrefetchOp

  module = load_workload_ir(WORKLOADS / "transformer_decode_kv_pipeline.mlir")
  contexts = [op for op in module.ops if op.name == "nest.context"]
  assert len(contexts) == 1
  inrel_block: dict[object, int] = {}  # input_released SSAValue -> block
  gated = 0
  for op in contexts[0].body.block.ops:
    if isinstance(op, NestDispatchOp):
      grid_tag = op.grid_done.type.tag.data
      if grid_tag.startswith("d") and grid_tag.endswith("_grid"):
        inrel_block[op.input_released] = int(grid_tag[1:-5])
      continue
    if isinstance(op, NestPrefetchOp):
      tag = op.result.type.tag.data
      if not (tag.startswith("pre_k_") or tag.startswith("pre_v_")):
        continue
      block = int(tag.rsplit("_", 1)[1])
      expected = {value for value, value_block in inrel_block.items() if value_block == block - 2}
      deps = set(op.depends_on)
      if block < 2:
        assert not (deps & set(inrel_block)), f"prefetch {tag} must be un-gated"
      else:
        # Dispatches are emitted before the prefetch that consumes their
        # inrel (prefetch(b) precedes dispatch(b) but follows dispatch(b-2)).
        assert expected & deps, (
          f"prefetch {tag} must depend on block {block - 2}'s input_released"
        )
        gated += 1
  assert gated == 2 * 6, f"expected 12 ping/pong-gated prefetches, saw {gated}"


def test_decode_block_count():
  """8 statically unrolled KV-block dispatches plus the fixed-position append."""
  text = (WORKLOADS / "transformer_decode_kv_pipeline.mlir").read_text(encoding="utf-8")
  dispatches = [line for line in text.splitlines() if "nest.dispatch.tasks.async" in line]
  assert len(dispatches) == 9  # 8 KV blocks + 1 fixed-position append
  assert sum("decode_attention_block" in line for line in dispatches) == 8
  assert sum("kv_append_tile" in line for line in dispatches) == 1
  assert "valid_sequence_length = 2048" in text


def test_decode_hbm_bytes_match_expected_kv_bytes(decode_results):
  """The KV scan must explain its HBM traffic; extra bytes must be accounted."""
  piped, _, _, _ = decode_results
  assert piped.completed
  traffic = piped.group_snapshot["memory"]["transfers"]["byte_counters"]
  expected_reads = (
    DECODE_KV_BYTES  # K + V history: 2 MiB
    + 2048  # Q_IN
    + 4224  # S_INIT
    + 2048  # H_T (append projection input)
    + 2 * 524288  # WK_A + WV_A (hidden x head_dim per tile)
  )
  expected_writes = 4096 + 2 * 512  # OUT + K_APPEND/V_APPEND
  assert traffic["hbm_read_bytes"] == expected_reads, (
    f"HBM reads {traffic['hbm_read_bytes']} != expected {expected_reads}: every byte "
    "beyond the 2 MiB KV scan must be explained"
  )
  assert traffic["hbm_write_bytes"] == expected_writes


# ---------------------------------------------------------------------------
# Prefill L2 lifetime
# ---------------------------------------------------------------------------


def test_prefill_l2_lifetime(prefill_results):
  """All L2 extents are final-freed exactly once; no leaked backings/views."""
  piped, _, _, _ = prefill_results
  assert piped.completed
  arenas = piped.group_snapshot["arenas"]["l2"]
  assert arenas["live_backings"] == 0
  assert arenas["live_views"] == 0
  assert arenas["pending_shared_claims"] == 0
  assert arenas["pin_count"] == 0
  assert arenas["inflight_count"] == 0
  assert arenas["peak_allocated_bytes"] == 6815744  # declared root Arena contract


def test_prefill_traffic_and_stalls(prefill_results):
  """Prefill reports byte traffic per interface and non-negative stalls."""
  piped, baseline, _, _ = prefill_results
  for result in (piped, baseline):
    traffic = result.group_snapshot["memory"]["transfers"]["byte_counters"]
    # X + WQ + WK + WV + WO reads; OUT is the only HBM write.
    assert traffic["hbm_read_bytes"] == 1048576 + 2097152 + 524288 * 2 + 2097152
    assert traffic["hbm_write_bytes"] == 1048576
    assert traffic["l1_write_bytes"] > 0
    stalls = result.pmu.named_cycles
    assert stalls.get("l2_bank_wait", 0) >= 0 and stalls.get("l1_bank_wait", 0) >= 0
