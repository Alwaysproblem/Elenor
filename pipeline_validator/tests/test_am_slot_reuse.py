"""Q09 permanent regression: one owner reuses AM logical slot 0 across chains.

Two sequential typed Matmul chains run from the SAME tile UCE context on the
same logical AM slot0 under quota 1 (one group x one physical slot).  Chain 0
reaches its accepted FINAL while the L1 output port is blocked until cycle
3000, so its lease stays live: chain 1's ALLOC_AM is issued once, parks exactly
once, and is granted by the autonomous slot_free BEFORE the UCE awaits chain
0's completion event.  The old event remains awaitable after the new grant,
both chains drain, and both HBM outputs match an independent integer oracle
built from small BF16 operands (every partial sum is an exact small integer,
so pre-P06 whole-step accumulation and any future per-MAC RNE profile agree).
The final makespan is intentionally not pinned.  Since P07 each chain_done
terminal is published at its slot_free, never before, and no later than its
event.

pytest:  conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_am_slot_reuse.py -q
smoke:   conda run -n elenor-validator python -m pipeline_validator.tests.test_am_slot_reuse
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import pytest

from pipeline_validator.compiler import compile_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import ExecTileOp, GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.simulator import SimResult, Simulator
from pipeline_validator.workload_ir import parse_workload_ir

M, N, K = 16, 16, 8
OUT_BYTES = M * N * 4
BLOCKED_UNTIL = 3000
A_BASE, B_BASE, OLD_C_BASE, NEW_C_BASE = 0x100000, 0x200000, 0x300000, 0x400000
SOURCE_NAME = "q09_am_slot0_reuse"

# Same owner, two sequential INIT+FINAL chains, both on logical slot 0.  The
# second `tile.alloc ... am logical_slot = 0` is legal only because chain 0's
# `output` step already ended its source liveness; at runtime its ALLOC_AM
# still parks because chain 0's physical lease is held by the output
# backpressure.  `tile.await %k1` / `tile.await %k0` both come after the second
# ALLOC, so the old completion is awaited strictly after the new grant.
_Q09_IR = """builtin.module {
  tile.program @q09_slot_reuse_mm(
    %task: !nest.task, %a_l2: !nest.l2_buffer<2x16x8xbf16>,
    %b_l2: !nest.l2_buffer<2x8x16xbf16>, %c0_l2: !nest.l2_buffer<16x16xf32>,
    %c1_l2: !nest.l2_buffer<16x16xf32>)
        resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
      tile_l1_spm_bytes_per_context = 24576> {
    %a0_v = tile.subview %a_l2 offsets = [0, 0, 0] sizes = [1, 16, 8] strides = [1, 1, 1]
      : !nest.l2_view<1x16x8xbf16>
    %a1_v = tile.subview %a_l2 offsets = [1, 0, 0] sizes = [1, 16, 8] strides = [1, 1, 1]
      : !nest.l2_view<1x16x8xbf16>
    %b0_v = tile.subview %b_l2 offsets = [0, 0, 0] sizes = [1, 8, 16] strides = [1, 1, 1]
      : !nest.l2_view<1x8x16xbf16>
    %b1_v = tile.subview %b_l2 offsets = [1, 0, 0] sizes = [1, 8, 16] strides = [1, 1, 1]
      : !nest.l2_view<1x8x16xbf16>
    %c0_v = tile.subview %c0_l2 offsets = [0, 0] sizes = [16, 16] strides = [1, 1]
      : !nest.l2_view<16x16xf32>
    %c1_v = tile.subview %c1_l2 offsets = [0, 0] sizes = [16, 16] strides = [1, 1]
      : !nest.l2_view<16x16xf32>
    %a0 = tile.alloc shape = [16, 8] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x8xbf16>
    %b0 = tile.alloc shape = [8, 16] dtype = "bf16" alignment = 256 : !tile.l1_buffer<8x16xbf16>
    %out0 = tile.alloc memory_space = "l1" shape = [16, 16] dtype = "f32" alignment = 256
      : !tile.l1_buffer<16x16xf32>
    %a1 = tile.alloc shape = [16, 8] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x8xbf16>
    %b1 = tile.alloc shape = [8, 16] dtype = "bf16" alignment = 256 : !tile.l1_buffer<8x16xbf16>
    %out1 = tile.alloc memory_space = "l1" shape = [16, 16] dtype = "f32" alignment = 256
      : !tile.l1_buffer<16x16xf32>
    %acc0 = tile.alloc memory_space = "am" logical_slot = 0 output(%out0) shape = [16, 16]
      dtype = "f32" alignment = 256 : !tile.am_buffer<16x16xf32>
    %la0 = tile.load.async %a0_v into %a0 : !tile.event<"la0">
    %lb0 = tile.load.async %b0_v into %b0 : !tile.event<"lb0">
    %la1 = tile.load.async %a1_v into %a1 : !tile.event<"la1">
    %lb1 = tile.load.async %b1_v into %b1 : !tile.event<"lb1">
    tile.await %la0, %lb0
    %k0 = tile.boa.async "matmul" ins(%a0, %b0) outs(%out0) acc(%acc0) m = 16 n = 16 k = 8 seq = 0
      output : !tile.event<"k0">
    tile.await %la1, %lb1
    tile.signal input_released(%task)
    %acc1 = tile.alloc memory_space = "am" logical_slot = 0 output(%out1) shape = [16, 16]
      dtype = "f32" alignment = 256 : !tile.am_buffer<16x16xf32>
    %k1 = tile.boa.async "matmul" ins(%a1, %b1) outs(%out1) acc(%acc1) m = 16 n = 16 k = 8 seq = 0
      output : !tile.event<"k1">
    tile.await %k1
    tile.await %k0
    %st0 = tile.store.async %out0 into %c0_v : !tile.event<"st0">
    %st1 = tile.store.async %out1 into %c1_v : !tile.event<"st1">
    tile.await %st0, %st1
    tile.free %a0
    tile.free %b0
    tile.free %a1
    tile.free %b1
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @q09_slot_reuse_ctx(
    %A: !nest.global_memref<2x16x8xbf16>, %B: !nest.global_memref<2x8x16xbf16>,
    %C0: !nest.global_memref<16x16xf32>, %C1: !nest.global_memref<16x16xf32>)
    placement = 1
        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
      logical_tasks = 1, l2_spm_bytes = 16384, requested_contexts_per_tile = 1> {
    %a_blk = nest.subview %A offsets = [0, 0, 0] sizes = [2, 16, 8] strides = [1, 1, 1]
      : !nest.global_view<2x16x8xbf16>
    %b_blk = nest.subview %B offsets = [0, 0, 0] sizes = [2, 8, 16] strides = [1, 1, 1]
      : !nest.global_view<2x8x16xbf16>
    %c0_blk = nest.subview %C0 offsets = [0, 0] sizes = [16, 16] strides = [1, 1]
      : !nest.global_view<16x16xf32>
    %c1_blk = nest.subview %C1 offsets = [0, 0] sizes = [16, 16] strides = [1, 1]
      : !nest.global_view<16x16xf32>
    %a_l2_1 = nest.alloc slot = "a" role = "in" shape = [2, 16, 8] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<2x16x8xbf16>
    %b_l2_1 = nest.alloc slot = "b" role = "in" shape = [2, 8, 16] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<2x8x16xbf16>
    %c0_l2_1 = nest.alloc slot = "c0" role = "out" shape = [16, 16] dtype = "f32" alignment = 256
      : !nest.l2_buffer<16x16xf32>
    %c1_l2_1 = nest.alloc slot = "c1" role = "out" shape = [16, 16] dtype = "f32" alignment = 256
      : !nest.l2_buffer<16x16xf32>
    %a_pref = nest.dma.prefetch.async %a_blk into %a_l2_1 : !nest.event<"a_pref">
    %b_pref = nest.dma.prefetch.async %b_blk into %b_l2_1 : !nest.event<"b_pref">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %read, %ready = nest.dispatch.tasks.async @q09_slot_reuse_mm l1_mode = 0 context = 0
      tasks(%tasks) globals() bindings(%a_l2_1, %b_l2_1, %c0_l2_1, %c1_l2_1)
      ins(%a_l2_1, %b_l2_1) outs(%c0_l2_1, %c1_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%a_pref, %b_pref)
      : (!nest.event<"grid">, !nest.event<"read">, !nest.event<"ready">)
    %c0_store = nest.dma.store.async %c0_l2_1 into %c0_blk depends_on(%ready)
      : !nest.event<"c0_store">
    %c1_store = nest.dma.store.async %c1_l2_1 into %c1_blk depends_on(%ready)
      : !nest.event<"c1_store">
    nest.release %a_l2_1 depends_on(%read, %a_pref)
    nest.release %b_l2_1 depends_on(%read, %b_pref)
    nest.release %c0_l2_1 depends_on(%c0_store)
    nest.release %c1_l2_1 depends_on(%c1_store)
    nest.await %grid, %c0_store, %c1_store
    nest.return
  }
  nexus.program @q09_slot_reuse(
    %A_1: !nest.global_memref<2x16x8xbf16>, %B_1: !nest.global_memref<2x8x16xbf16>,
    %C0_1: !nest.global_memref<16x16xf32>, %C1_1: !nest.global_memref<16x16xf32>) {
    %done_1 = nexus.submit_context.async @q09_slot_reuse_ctx(%A_1, %B_1, %C0_1, %C1_1)
      : !nexus.event<"done">
    nexus.await %done_1
    nexus.return
  }
}
"""


# --- independent integer oracle (no production numeric code) -----------------
#
# |value| <= 3 and |partial sum| <= 48 < 2^24, so every per-MAC RNE prefix and
# the whole-step sum are exact in both binary64 and binary32: pre-P06 and any
# future per-MAC profile must produce these identical bytes.


def _old_a(row: int, kk: int) -> int:
  return ((row * K + kk) % 5) - 2


def _old_b(kk: int, col: int) -> int:
  return ((kk * N + col) % 3) - 1


def _new_a(row: int, kk: int) -> int:
  return ((row + 2 * kk) % 7) - 3


def _new_b(kk: int, col: int) -> int:
  return ((3 * kk + col) % 5) - 2


def _bf16_pair(value: int) -> bytes:
  f32_bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
  return struct.pack("<H", f32_bits >> 16)


def _chain_operands(a_values, b_values) -> tuple[bytes, bytes, bytes]:
  a = b"".join(_bf16_pair(a_values(row, kk)) for row in range(M) for kk in range(K))
  b = b"".join(_bf16_pair(b_values(kk, col)) for kk in range(K) for col in range(N))
  expected = [
    sum(a_values(row, kk) * b_values(kk, col) for kk in range(K)) for row in range(M) for col in range(N)
  ]
  return a, b, struct.pack(f"<{M * N}f", *(float(item) for item in expected))


# --- full parser/compiler/loader/Simulator/ByteStore scenario ----------------


@dataclass
class _Q09Run:
  result: SimResult
  simulator: Simulator
  store: ByteStore
  am_pcs: tuple[int, int]
  old_await_pc: int
  expected_old: bytes
  expected_new: bytes


def _tile_program(artifact):
  task = next(iter(artifact.entry.tasks.values()))
  return next(iter(task.role_bindings.values())).tile_program


def _run_q09(*, seed: int = 0) -> _Q09Run:
  old_a, old_b, expected_old = _chain_operands(_old_a, _old_b)
  new_a, new_b, expected_new = _chain_operands(_new_a, _new_b)
  assert expected_old != expected_new
  hw = HardwareConfig().with_overrides(
    boa_am_group_count=1, boa_am_slots_per_group=1, boa_am_slots_per_context=1
  )
  sim_cfg = SimConfig(fidelity="full_memory", context_count=1, seed=seed)
  bindings = {
    "A": GlobalBinding("A", A_BASE, len(old_a) + len(new_a), "r"),
    "B": GlobalBinding("B", B_BASE, len(old_b) + len(new_b), "r"),
    "C0": GlobalBinding("C0", OLD_C_BASE, OUT_BYTES, "w"),
    "C1": GlobalBinding("C1", NEW_C_BASE, OUT_BYTES, "w"),
  }
  artifact = compile_program(
    parse_workload_ir(_Q09_IR),
    hw,
    sim_cfg,
    binding_assumptions=bindings,
    source_name=SOURCE_NAME,
  )
  loaded = load_program(artifact, hw, sim_cfg, actual_bindings=bindings)
  store = ByteStore()
  store.seed_hbm(A_BASE, old_a + new_a)
  store.seed_hbm(B_BASE, old_b + new_b)
  simulator = Simulator(hw, sim_cfg, enable_tracer=True, byte_store=store)
  simulator.group.tiles[0].boa_results.set_output_blocked(True, until_cycle=BLOCKED_UNTIL)
  result = simulator.run(loaded)
  program = _tile_program(artifact)
  alloc_pcs = [index for index, inst in enumerate(program.insts) if inst.op is ExecTileOp.ALLOC_AM]
  assert len(alloc_pcs) == 2
  am_pcs = (alloc_pcs[0], alloc_pcs[1])
  old_await_pc = next(
    index
    for index, inst in enumerate(program.insts)
    if inst.op is ExecTileOp.WAIT and tuple(inst.args) == ("k0",)
  )
  return _Q09Run(result, simulator, store, am_pcs, old_await_pc, expected_old, expected_new)


# --- real trace / service / UCE observations ---------------------------------


def _events(simulator: Simulator) -> list[dict]:
  assert simulator.tracer is not None
  return json.loads(simulator.tracer.to_chrome_json())["traceEvents"]


def _instant(events: list[dict], name: str, cat: str) -> dict:
  hits = [event for event in events if event["name"] == name and event["cat"] == cat]
  assert len(hits) == 1, f"expected exactly one {name} on {cat}, saw {len(hits)}"
  return hits[0]


def _cycle_of(simulator: Simulator, event: dict) -> int:
  assert simulator.tracer is not None
  return round(event["ts"] / simulator.tracer.cycle_to_us(1))


def _issue_cycle(simulator: Simulator, events: list[dict], pc: int) -> int:
  hits = [event for event in events if event["name"] == "uce_issue" and event["args"].get("pc") == pc]
  assert len(hits) == 1, f"expected exactly one issue of pc {pc}, saw {len(hits)}"
  return _cycle_of(simulator, hits[0])


def test_q09_same_owner_reuses_slot0_before_awaiting_old_completion():
  run = _run_q09()
  result, simulator, store = run.result, run.simulator, run.store
  tile = simulator.group.tiles[0]
  assert result.completed
  assert result.cycles >= BLOCKED_UNTIL

  # Per-Tile manager counters: 2 requests/grants, exactly one park/wake cycle,
  # no faults, everything returned.
  manager = tile.shared_am
  snapshot = manager.snapshot()
  assert snapshot["tile_id"] == 0
  assert snapshot["request_count"] == 2
  assert snapshot["acquire_count"] == 2
  assert snapshot["parked_count"] == 1
  assert snapshot["wake_count"] == 1
  assert snapshot["cancelled_request_count"] == 0
  assert snapshot["fault_count"] == 0
  assert snapshot["commit_count"] == 2
  assert snapshot["final_count"] == 2
  assert snapshot["slot_free_count"] == 2
  assert snapshot["pending_requests"] == 0
  assert snapshot["reserved_slots"] == 0
  assert snapshot["live_leases"] == 0
  assert snapshot["requests"] == []
  assert snapshot["leases"] == []
  assert snapshot["bundles"] == []
  assert snapshot["total_slots"] == snapshot["free_slots"] == 1
  assert snapshot["peak_live"] == 1
  assert snapshot["peak_pending"] == 1

  # Both chain_done terminals belong to the SAME owner identity and to two
  # generations of the same physical slot 0.0.
  terminals = snapshot["terminals"]
  assert len(terminals) == 2
  old_term, new_term = terminals
  assert old_term["kind"] == new_term["kind"] == "chain_done"
  assert old_term["success"] is True and new_term["success"] is True
  assert old_term["lease_id"] != new_term["lease_id"]
  assert new_term["generation"] == old_term["generation"] + 1
  assert (old_term["owner"], old_term["ctx_id"], old_term["owner_epoch"]) == (
    new_term["owner"],
    new_term["ctx_id"],
    new_term["owner_epoch"],
  )
  assert snapshot["slots"][0]["generation"] == new_term["generation"]

  # Result service: two FINAL records; the old chain's output wrote no earlier
  # than the backpressure release point and its event only after that output.
  service = tile.boa_results.snapshot()
  assert service["event_count"] == 2
  assert service["blocked_cycles"] > 0
  assert service["overflow_count"] == 0
  assert service["output_blocked"] is False
  assert service["output_block_until_cycle"] is None
  records = sorted(service["records"], key=lambda record: record["event_cycle"])
  assert len(records) == 2
  old_rec, new_rec = records
  for record in records:
    assert record["final"] is True
    assert record["status"] == "chain_done"
    assert record["compute_done_cycle"] < record["output_cycle"] <= record["event_cycle"]
    terminal = old_term if record is old_rec else new_term
    assert terminal["lease_id"] == record["lease_id"]
    assert record["output_cycle"] <= record["slot_free_cycle"] == terminal["cycle"]
    assert terminal["cycle"] <= record["event_cycle"]
  assert old_rec["output_cycle"] >= BLOCKED_UNTIL
  assert old_rec["event_cycle"] < new_rec["event_cycle"]
  # The new chain only ran after the old chain's event had fired.
  assert new_rec["compute_done_cycle"] >= old_rec["event_cycle"]

  # Trace: the second ALLOC parked exactly once while the old chain still held
  # the slot, and the wake is the slot_free that also fires the old event.
  events = _events(simulator)
  parked = _instant(events, "am_request_parked", "SharedAM")
  wake = _instant(events, "am_request_wake", "SharedAM")
  assert wake["args"]["request_id"] == parked["args"]["request_id"]
  assert parked["args"]["buffer"] == "am:1"
  assert parked["args"]["kind"] == "continuation"
  assert parked["args"]["free_slots"] == 0
  assert parked["args"]["reserved_slots"] == 0
  assert wake["args"]["lease_id"] is not None
  assert wake["args"]["waited_cycles"] > 0
  park_cycle = _cycle_of(simulator, parked)
  grant_cycle = _cycle_of(simulator, wake)
  assert park_cycle < old_rec["output_cycle"]
  assert grant_cycle == old_rec["event_cycle"]
  assert park_cycle == grant_cycle - wake["args"]["waited_cycles"]

  # Both grants hit physical slot 0.0 of logical slot 0 for the same owner.
  allocates = sorted(
    (event for event in events if event["name"] == "am_allocate" and event["cat"] == "SharedAM"),
    key=lambda event: event["ts"],
  )
  assert len(allocates) == 2
  assert [event["args"]["generation"] for event in allocates] == [
    old_term["generation"],
    new_term["generation"],
  ]
  for event in allocates:
    assert event["args"]["slot"] == "0.0"
    assert event["args"]["logical_slot"] == 0
  assert allocates[0]["args"]["owner"] == allocates[1]["args"]["owner"]
  assert allocates[0]["args"]["ctx_id"] == allocates[1]["args"]["ctx_id"]

  # UCE program order proof: the second ALLOC (parked above) is issued before
  # the UCE ever issues the await of the old completion, and that await is
  # issued at/after the grant that fired the old event.
  assert len(run.am_pcs) == 2
  first_alloc_cycle = _issue_cycle(simulator, events, run.am_pcs[0])
  second_alloc_cycle = _issue_cycle(simulator, events, run.am_pcs[1])
  old_await_cycle = _issue_cycle(simulator, events, run.old_await_pc)
  assert first_alloc_cycle < second_alloc_cycle == park_cycle
  assert second_alloc_cycle < old_await_cycle
  assert old_await_cycle >= grant_cycle
  simulator.tracer.assert_well_formed()

  # Both output buffers carry independent, exactly predicted bytes.
  assert store.read_hbm(OLD_C_BASE, OUT_BYTES) == run.expected_old
  assert store.read_hbm(NEW_C_BASE, OUT_BYTES) == run.expected_new


# ---------------------------------------------------------------------------
# Module CLI (container target shared-am-functional, case q09-old-event-cross-slot)
# ---------------------------------------------------------------------------


CLI_CASE = "q09-old-event-cross-slot"


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
  """Run the real Q09 regression and write the report/trace artifacts."""
  import hashlib
  from pathlib import Path

  from pipeline_validator.tests import case_cli

  args = case_cli.build_parser(
    "python -m pipeline_validator.tests.test_am_slot_reuse",
    "P04 Q09: one owner reuses AM slot 0 across chains, old event awaitable "
    "strictly after the new grant.",
    [CLI_CASE],
  ).parse_args(argv)
  captured: dict = {}

  def _capture_artifacts() -> None:
    run = _run_q09(seed=args.seed)
    assert run.result.completed, run.result.reason
    old_c = run.store.read_hbm(OLD_C_BASE, OUT_BYTES)
    new_c = run.store.read_hbm(NEW_C_BASE, OUT_BYTES)
    assert old_c == run.expected_old and new_c == run.expected_new
    assert run.simulator.tracer is not None
    run.simulator.tracer.assert_well_formed()
    captured["trace"] = run.simulator.tracer.to_chrome_json()
    captured["cycles"] = run.result.cycles
    captured["old_c_sha256"] = hashlib.sha256(old_c).hexdigest()
    captured["new_c_sha256"] = hashlib.sha256(new_c).hexdigest()

  checks = case_cli.run_checks(
    [
      ("q09-same-owner-reuses-slot0-before-awaiting-old-completion",
       test_q09_same_owner_reuses_slot0_before_awaiting_old_completion),
      ("q09-trace-and-output-bytes-capture", _capture_artifacts),
    ]
  )
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  trace_path = out / f"{CLI_CASE}.trace.json"
  trace_path.write_text(captured.get("trace", ""), encoding="utf-8")
  report = case_cli.build_report(
    CLI_CASE,
    "test_am_slot_reuse",
    args.seed,
    checks,
    metrics={key: captured[key] for key in ("cycles", "old_c_sha256", "new_c_sha256") if key in captured},
  )
  case_cli.write_report(out, report)
  return case_cli.finish(report)


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
