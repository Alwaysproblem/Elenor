"""P05 real-runtime engine-pool gates and standalone schema-1 config smoke.

No manufactured affinity or increased context quota: up to four UCE contexts
hold two chains each. All answers use independent integer dot products. The
port-blocking observer calls the real group step and only supplies external
output backpressure; it never substitutes engine, allocation or event behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from pipeline_validator.compiler import compile_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.report import build_report, report_to_json
from pipeline_validator.simulator import SimResult, Simulator
from pipeline_validator.workload_ir import parse_workload_ir

M = N = 16
LONG_ESTIMATE_OPS = 524288
ROOT = Path(__file__).resolve().parents[2]


def _operands(seed: int, k: int, steps: int) -> tuple[bytes, bytes, bytes]:
  a = [((row + 2 * kk + seed) % 5) - 2 for row in range(M) for kk in range(k)]
  b = [((3 * kk + col + seed) % 5) - 2 for kk in range(k) for col in range(N)]
  c = [steps * sum(a[row * k + kk] * b[kk * N + col] for kk in range(k))
       for row in range(M) for col in range(N)]

  def bf16(values: list[int]) -> bytes:
    return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16)
                    for value in values)

  # Every operand and accumulation prefix is an exact small integer.
  return bf16(a), bf16(b), struct.pack(f"<{M * N}f", *c)


def _source(contexts: int, k: int, steps: int, estimates: tuple[int, ...], before: bool,
            grant_before_estimates: bool) -> str:
  programs, globals_, actuals, body, waits = [], [], [], [], []
  setup, preload, retire = [], [], []
  for ctx in range(contexts):
    program = f"p{ctx}"
    # One independent blocker; other contexts must not wait to enqueue their
    # own blocker before they can compete for typed service.
    context_estimates = estimates if ctx == 0 else ()
    formals = ["%task: !nest.task"]
    tile_lines, loads, inputs, outputs, buffers, prefs = [], [], [], [], [], []
    for chain in range(2):
      for kind, dims, dtype in (("a", (M, k), "bf16"), ("b", (k, N), "bf16"),
                                ("c", (M, N), "f32")):
        local = f"{kind}{chain}"
        name = f"p{ctx}_{local}"
        shape = "x".join(str(d) for d in dims)
        formals.append(f"%{local}_l2: !nest.l2_buffer<{shape}x{dtype}>")
        globals_.append(f"%{name}: !nest.global_memref<{shape}x{dtype}>")
        actuals.append(f"%{name}")
        role = "out" if kind == "c" else "in"
        setup.extend([
          f'    %{name}_buf = nest.alloc slot = "{name}" role = "{role}" '
          f'shape = [{dims[0]}, {dims[1]}] dtype = "{dtype}" alignment = 256 '
          f': !nest.l2_buffer<{shape}x{dtype}>',
          f"    %{name}_view = nest.subview %{name} offsets = [0, 0] "
          f"sizes = [{dims[0]}, {dims[1]}] strides = [1, 1] : !nest.global_view<{shape}x{dtype}>",
        ])
        buffers.append(f"%{name}_buf")
        tile_lines.extend([
          f"    %{local}_view = tile.subview %{local}_l2 offsets = [0, 0] "
          f"sizes = [{dims[0]}, {dims[1]}] strides = [1, 1] : !nest.l2_view<{shape}x{dtype}>",
          f'    %{local} = tile.alloc shape = [{dims[0]}, {dims[1]}] dtype = "{dtype}" '
          f'alignment = 256 : !tile.l1_buffer<{shape}x{dtype}>',
        ])
        if kind == "c":
          outputs.append(f"%{name}_buf")
        else:
          inputs.append(f"%{name}_buf")
          prefs.append(f"%{name}_pref")
          preload.append(f"%{name}_pref")
          setup.append(f"    %{name}_pref = nest.dma.prefetch.async %{name}_view into %{name}_buf "
                      f': !nest.event<"{name}_pref">')
          loads.append(f"%load_{local}")
          tile_lines.append(f"    %load_{local} = tile.load.async %{local}_view into %{local} "
                            f': !tile.event<"load_{local}">')
    tile_lines += [f"    tile.await {', '.join(loads)}", "    tile.signal input_released(%task)"]
    estimate_lines = [
      f'    %estimate{i} = tile.boa.estimate.async "matmul" m = 16 n = 16 k = 16 '
      f'ops = {ops} : !tile.event<"estimate{i}">' for i, ops in enumerate(context_estimates)
    ]
    if before and not grant_before_estimates:
      tile_lines.extend(estimate_lines)
    # Both leases exist before any FINAL is accepted: static bundle is TWO.
    for chain in range(2):
      tile_lines.append(
        f'    %acc{chain} = tile.alloc memory_space = "am" logical_slot = {chain} '
        f'output(%c{chain}) shape = [16, 16] dtype = "f32" alignment = 256 '
        ': !tile.am_buffer<16x16xf32>'
      )
    if before and grant_before_estimates:
      tile_lines.extend(estimate_lines)
    for seq in range(steps):
      for chain in range(2):
        flags = (" accumulate" if seq else "") + (" output" if seq == steps - 1 else "")
        tile_lines.append(
          f'    %k{chain}_{seq} = tile.boa.async "matmul" ins(%a{chain}, %b{chain}) '
          f'outs(%c{chain}) acc(%acc{chain}) m = 16 n = 16 k = {k} seq = {seq}'
          f'{flags} : !tile.event<"k{chain}_{seq}">'
        )
      if seq == steps - 1 and not before:
        tile_lines.extend(estimate_lines)
      tile_lines.append(f"    tile.await %k0_{seq}, %k1_{seq}")
    if context_estimates:
      tile_lines.append("    tile.await " + ", ".join(
        f"%estimate{i}" for i in range(len(context_estimates))))
    for chain in range(2):
      tile_lines.append(f"    %store{chain} = tile.store.async %c{chain} into %c{chain}_view "
                        f': !tile.event<"store{chain}">')
    tile_lines.append("    tile.await %store0, %store1")
    tile_lines.extend(f"    tile.free %{kind}{chain}" for chain in range(2) for kind in ("a", "b"))
    tile_lines += ["    tile.signal output_ready(%task)", "    tile.return"]
    programs.append(
      f"  tile.program @{program}({', '.join(formals)}) "
      "resource_contract = #tile.resources<allowed_profiles = [0], "
      "tile_l1_spm_bytes_per_context = 24576> {\n" + "\n".join(tile_lines) + "\n  }"
    )
    body.extend([
      f"    %tasks{ctx} = nest.task.range from = 0 to = 1 : !nest.task_range",
      f"    %grid{ctx}, %read{ctx}, %ready{ctx} = nest.dispatch.tasks.async @{program} "
      f"l1_mode = 0 context = {ctx} tasks(%tasks{ctx}) globals() bindings({', '.join(buffers)}) "
      f"ins({', '.join(inputs)}) outs({', '.join(outputs)}) "
      "signal_policy { input_released = #nest.aggregate<all_tasks> "
      "output_ready = #nest.aggregate<all_tasks> } "
      f"depends_on({', '.join(prefs)}) "
      f': (!nest.event<"grid{ctx}">, !nest.event<"read{ctx}">, !nest.event<"ready{ctx}">)',
    ])
    waits.append(f"%grid{ctx}")
    for chain in range(2):
      name = f"p{ctx}_c{chain}"
      retire.extend([
        f"    %{name}_store = nest.dma.store.async %{name}_buf into %{name}_view "
        f'depends_on(%ready{ctx}) : !nest.event<"{name}_store">',
        f"    nest.release %{name}_buf depends_on(%{name}_store)",
      ])
      waits.append(f"%{name}_store")
      for kind in ("a", "b"):
        inp = f"p{ctx}_{kind}{chain}"
        retire.append(f"    nest.release %{inp}_buf depends_on(%read{ctx}, %{inp}_pref)")
  return "\n".join([
    "builtin.module {", *programs,
    f"  nest.context @ctx({', '.join(globals_)}) placement = 1 "
    "resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0], "
    f"logical_tasks = {contexts}, l2_spm_bytes = {24576 * contexts}, "
    f"requested_contexts_per_tile = {contexts}> {{",
    # Register dispatches before blocked stores/releases fill the finite action window.
    *setup, f"    nest.await {', '.join(preload)}", *body, *retire,
    f"    nest.await {', '.join(waits)}", "    nest.return", "  }",
    f"  nexus.program @run({', '.join(globals_)}) {{",
    f"    %done = nexus.submit_context.async @ctx({', '.join(actuals)}) : !nexus.event<\"done\">",
    "    nexus.await %done", "    nexus.return", "  }", "}",
  ])


@dataclass
class Case:
  simulator: Simulator
  result: SimResult
  expected: dict[str, bytes]
  bindings: dict[str, GlobalBinding]
  target_hash: str
  events: list[dict]
  jobs: list[dict]
  grants: list[dict]
  report: dict
  resume_cycle: int | None


def run_case(hw: HardwareConfig, *, contexts: int = 1, steps: int = 2, k: int = 8,
             estimates: tuple[int, ...] = (), before: bool = False, block: bool = False,
             on_cycle=None, grant_before_estimates: bool = False, seed: int = 0) -> Case:
  bindings: dict[str, GlobalBinding] = {}
  seeds: dict[str, bytes] = {}
  expected: dict[str, bytes] = {}
  for ctx in range(contexts):
    for chain in range(2):
      a, b, c = _operands(ctx * 2 + chain, k, steps)
      for kind, data in (("a", a), ("b", b), ("c", c)):
        name = f"p{ctx}_{kind}{chain}"
        base = 0x100000 + len(bindings) * 0x10000
        bindings[name] = GlobalBinding(name, base, len(data), "w" if kind == "c" else "r")
        (expected if kind == "c" else seeds)[name] = data
  source = _source(contexts, k, steps, estimates, before, grant_before_estimates)
  sim_cfg = SimConfig(fidelity="full_memory", context_count=contexts, seed=seed)
  artifact = compile_program(parse_workload_ir(source),
                             hw, sim_cfg,
                             binding_assumptions=bindings, source_name="p05_engine_pool")
  store = ByteStore()
  for name, data in seeds.items():
    store.seed_hbm(bindings[name].base_iova, data)
  simulator = Simulator(hw, sim_cfg,
                        enable_tracer=True, byte_store=store)
  service = simulator.group.tiles[0].boa_results
  resume_cycle = None
  if block:
    service.set_output_blocked(True)
  original_step = simulator.group.step

  def step(cycle: int) -> bool:
    nonlocal resume_cycle
    manager = simulator.group.tiles[0].shared_am
    if block and resume_cycle is None and manager.acquire_count == contexts * 2:
      if service.snapshot()["states"]["wait_output"]:
        resume_cycle = cycle + 11
    if resume_cycle is not None and cycle >= resume_cycle:
      service.set_output_blocked(False)
    if on_cycle is not None:
      on_cycle(simulator, cycle)
    return original_step(cycle)

  with pytest.MonkeyPatch.context() as patch:
    if block or on_cycle is not None:
      patch.setattr(simulator.group, "step", step)
    result = simulator.run(load_program(artifact, hw, simulator.sim, actual_bindings=bindings))
  assert simulator.tracer is not None
  simulator.tracer.assert_well_formed()
  events = json.loads(simulator.tracer.to_chrome_json())["traceEvents"]
  unit = simulator.tracer.cycle_to_us(1)
  jobs = [dict(e["args"], start=round(e["ts"] / unit), end=round((e["ts"] + e["dur"]) / unit))
          for e in events if e.get("ph") == "X" and e.get("name", "").startswith("BOA:")]
  grants = [dict(e["args"], cycle=round(e["ts"] / unit))
            for e in events if e.get("name") == "am_allocate"]
  report = json.loads(report_to_json(build_report(artifact.workload_info, result, num_tiles=hw.num_tiles)))
  return Case(simulator, result, expected, bindings, artifact.target_hash,
              events, jobs, grants, report, resume_cycle)


def check_case(case: Case, *, contexts: int, steps: int, estimates: int = 0) -> dict:
  assert case.result.completed, case.result.reason
  tile = case.simulator.group.tiles[0]
  hw = case.simulator.hw
  manager = tile.shared_am.snapshot()
  chains = 2 * contexts
  assert manager["request_count"] == manager["acquire_count"] == manager["final_count"] == chains
  assert manager["commit_count"] == chains * steps
  assert manager["slot_free_count"] == chains
  assert manager["live_leases"] == manager["pending_requests"] == manager["reserved_slots"] == 0
  assert manager["fault_count"] == manager["parked_count"] == 0
  assert manager["free_slots"] == manager["total_slots"]
  assert manager["engine_handle_counts"] == dict.fromkeys(range(hw.boa_arrays_per_tile), 0)
  grants = {g["lease_id"]: g for g in case.grants}
  assert len(grants) == chains
  typed = [job for job in case.jobs if job.get("lease_id") is not None]
  assert len(typed) == chains * steps
  assert len(case.jobs) - len(typed) == estimates
  for lease_id, grant in grants.items():
    jobs = [j for j in typed if j["lease_id"] == lease_id]
    assert sorted(j["seq"] for j in jobs) == list(range(steps))
    assert all(j["engine_id"] == j["bound_engine"] == grant["bound_engine"] for j in jobs)
    assert all(j["chain_id"] == grant["chain_id"] and j["start"] >= grant["cycle"] for j in jobs)
    assert grant["quota_slots"] == 2 and grant["owner_live_handles"] <= 2
  for engine in tile.boa_engines:
    jobs = sorted((j for j in case.jobs if j["engine_id"] == engine.engine_id), key=lambda j: j["start"])
    assert all(a["end"] <= b["start"] for a, b in pairwise(jobs))
    snapshot = engine.snapshot()
    assert snapshot["complete_count"] == snapshot["launch_count"] == len(jobs)
    assert snapshot["active_cycles"] == sum(j["end"] - j["start"] for j in jobs)
    assert snapshot["accepted_count"] == 0
  service = tile.boa_results.snapshot()
  assert service["enqueued_count"] == service["event_count"] == chains * steps
  assert service["output_write_count"] == service["release_count"] == chains
  assert service["pending"] == service["reservations"] == service["output_in_use"] == 0
  assert service["credits"] == service["credit_capacity"] and service["overflow_count"] == 0
  # Duplicate advance() calls in one Tile cycle inflate this counter even if
  # absolute commit deadlines still prevent double completion.
  finals = [r for r in service["records"] if r["final"]]
  assert len(finals) == chains
  assert service["blocked_cycles"] == sum(
    r["output_cycle"] - r["compute_done_cycle"] - hw.boa_commit_cycles for r in finals
  )
  by_event = {j["event_id"]: j for j in typed}
  for record in service["records"]:
    assert record["engine_id"] == by_event[record["event_id"]]["engine_id"]
  store = tile.mfe.transfer_manager.byte_store
  for name, expected in case.expected.items():
    assert store.read_hbm(case.bindings[name].base_iova, len(expected)) == expected, name
  assert all(check["pass"] for check in case.report["checks"]), case.report["checks"]
  overlap = any(a["engine_id"] != b["engine_id"] and max(a["start"], b["start"]) < min(a["end"], b["end"])
                for i, a in enumerate(typed) for b in typed[i + 1:])
  return {"cycles": case.result.cycles, "target_hash": case.target_hash,
          "peak_live": manager["peak_live"], "grant_engines": [g["bound_engine"] for g in case.grants],
          "overlap": overlap, "engines": [e.snapshot() for e in tile.boa_engines],
          "blocked_cycles": service["blocked_cycles"], "resume_cycle": case.resume_cycle,
          "output_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in case.expected.items()}}


def _four_live_handles_progress(hw: HardwareConfig) -> tuple[Case, dict]:
  """T06-P05-3/-4: two contexts x two chains live at once under per-context quota 2."""
  arrays = hw.boa_arrays_per_tile
  estimates = (LONG_ESTIMATE_OPS,) if arrays == 1 else ()
  case = run_case(
    hw, contexts=2, block=True, estimates=estimates, before=True, grant_before_estimates=True
  )
  info = check_case(case, contexts=2, steps=2, estimates=len(estimates))
  assert info["peak_live"] == 4
  assert info["grant_engines"] == [i % arrays for i in range(4)]
  assert info["blocked_cycles"] > 0
  if arrays > 1:
    assert info["overlap"]
  else:
    typed = [job for job in case.jobs if job.get("lease_id") is not None]
    assert min(job["start"] for job in typed) >= max(g["cycle"] for g in case.grants)
    order = [job["ctx_id"] for job in typed]
    assert set(order[:2]) == {0, 1} and order[:2] == order[2:4], order
  return case, info


@pytest.mark.parametrize("arrays", [1, 2, 4])
def test_legal_four_live_handles_progress_with_context_quota_two(arrays):
  _four_live_handles_progress(HardwareConfig(boa_arrays_per_tile=arrays))


def _multistep_overlap(hw: HardwareConfig) -> dict:
  case = run_case(hw, steps=3)
  info = check_case(case, contexts=1, steps=3)
  assert info["grant_engines"] == [0, 1] and info["overlap"]
  return info


@pytest.mark.parametrize("arrays", [2, 4])
def test_single_context_multistep_affinity_and_independent_output_overlap(arrays):
  _multistep_overlap(HardwareConfig(boa_arrays_per_tile=arrays))


def check_busy(hw: HardwareConfig) -> dict:
  case = run_case(hw, steps=1, estimates=(LONG_ESTIMATE_OPS,), before=True)
  info = check_case(case, contexts=1, steps=1, estimates=1)
  estimate = next(j for j in case.jobs if j.get("lease_id") is None)
  assert estimate["engine_id"] == 0
  first, second = case.grants
  assert first["bound_engine"] == 0 and second["bound_engine"] == 1
  assert estimate["start"] <= first["cycle"] < estimate["end"]
  blocked = next(j for j in case.jobs if j.get("lease_id") == first["lease_id"])
  neighbor = next(j for j in case.jobs if j.get("lease_id") == second["lease_id"])
  assert neighbor["end"] < blocked["start"] and blocked["start"] >= estimate["end"]
  waits = [e["args"] for e in case.events if e.get("name") == "boa_engine_wait"]
  assert any(w["reason"] == "engine_busy" and w["engine_id"] == 0 and
             w["lease_id"] == first["lease_id"] for w in waits)
  assert info["engines"][0]["busy_stall_cycles"] > 0
  return info


def test_busy_affinity_never_migrates_and_neighbor_queue_progresses():
  check_busy(HardwareConfig(boa_arrays_per_tile=2))


def _two_engines_four_contexts(hw: HardwareConfig) -> dict:
  """T06-P05-4: 2/2 handle distribution is least-live's result, not a cap."""
  case = run_case(hw, contexts=4, steps=1, block=True)
  info = check_case(case, contexts=4, steps=1)
  assert info["peak_live"] == 8 and info["grant_engines"] == [0, 1] * 4
  assert [g["engine_live_handles"] for g in case.grants] == [1, 1, 2, 2, 3, 3, 4, 4]
  assert {g["ctx_id"] for g in case.grants} == {0, 1, 2, 3}
  assert all(g["quota_slots"] == 2 for g in case.grants)
  return info


def test_two_engines_can_each_own_four_handles_without_increasing_context_quota():
  _two_engines_four_contexts(HardwareConfig(boa_arrays_per_tile=2))


def _mixed_estimates(hw: HardwareConfig) -> list[list[int]]:
  """Estimates run without AM and pick engines deterministically."""
  cases = [run_case(hw, k=64, steps=1, estimates=(8192, 8192)) for _ in range(2)]
  for case in cases:
    check_case(case, contexts=1, steps=1, estimates=2)
  selection = [[j["engine_id"] for j in c.jobs if j.get("lease_id") is None] for c in cases]
  assert selection[0] == selection[1]
  if hw.boa_arrays_per_tile == 4:
    assert selection[0] == [2, 3]
  return selection


@pytest.mark.parametrize("arrays", [1, 2, 4])
def test_mixed_estimates_run_without_am_and_select_deterministically(arrays):
  _mixed_estimates(HardwareConfig(boa_arrays_per_tile=arrays))


def parse_schema1_config(path: str | Path) -> dict:
  raw = yaml.safe_load(Path(path).read_text())
  if not isinstance(raw, dict) or type(raw.get("schema")) is not int or raw["schema"] != 1:
    raise ValueError("expected schema 1 wrapper")
  if raw.get("target") != "two-slot-progress":
    raise ValueError("expected two-slot-progress target")
  if set(raw) - {"schema", "target", "topology", "numeric", "validator", "board"}:
    raise ValueError("unknown wrapper fields")
  if any(raw.get(key, {}) != {} for key in ("topology", "numeric", "board")):
    raise ValueError("unconsumed configuration sections must be empty")
  validator = raw.get("validator", {})
  if not isinstance(validator, dict) or set(validator) - {"hw_overrides"}:
    raise ValueError("invalid validator configuration")
  overrides = validator.get("hw_overrides", {})
  if not isinstance(overrides, dict) or any(not isinstance(key, str) for key in overrides):
    raise ValueError("hw_overrides must be a mapping of field names")
  HardwareConfig().with_overrides(**overrides)
  return overrides


def test_schema1_geometry_changes_real_execution_not_only_report_metadata(tmp_path):
  path = tmp_path / "config.yaml"
  path.write_text(yaml.safe_dump({"schema": 1, "target": "two-slot-progress", "validator": {
    "hw_overrides": {"boa_arrays_per_tile": 2, "boa_array_rows": 8}}}))
  configured = run_case(HardwareConfig().with_overrides(**parse_schema1_config(path)))
  reference = run_case(HardwareConfig(boa_arrays_per_tile=2))
  check_case(configured, contexts=1, steps=2)
  check_case(reference, contexts=1, steps=2)
  assert configured.target_hash != reference.target_hash
  assert configured.result.cycles > reference.result.cycles
  assert configured.expected == reference.expected


def test_schema1_rejects_wrong_target_and_unconsumed_fields(tmp_path):
  path = tmp_path / "bad.yaml"
  for doc in ({"schema": 1, "target": "numeric"},
              {"schema": 1, "target": "two-slot-progress", "topology": {"arrays": 2}}):
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):
      parse_schema1_config(path)


CLI_CASES = ["t06-one-engine-many-handles", "engine-pool-progress"]
_REGISTERED_ENGINE_POOLS = (1, 2, 4)


def _run_cli_case(args: argparse.Namespace, hw: HardwareConfig, overrides: dict) -> int:
  """Run one registered container case through the real config-driven scenarios."""
  from pipeline_validator.tests import case_cli

  arrays = hw.boa_arrays_per_tile
  captured: dict = {}

  def _registered_pool() -> None:
    assert arrays in _REGISTERED_ENGINE_POOLS, (
      f"this case consumes the registered hardware/configs/boa-{{1,2,4}}engine.yaml "
      f"instances; boa_arrays_per_tile={arrays}"
    )

  def _single_engine() -> None:
    assert arrays == 1, (
      f"case t06-one-engine-many-handles must run under boa-1engine.yaml "
      f"(boa_arrays_per_tile=1), got {arrays}"
    )

  def _progress() -> None:
    case, info = _four_live_handles_progress(hw)
    assert case.simulator.tracer is not None
    captured["progress_trace"] = case.simulator.tracer.to_chrome_json()
    captured["progress"] = {
      "cycles": info["cycles"], "peak_live": info["peak_live"],
      "grant_engines": info["grant_engines"], "target_hash": info["target_hash"],
      "blocked_cycles": info["blocked_cycles"], "resume_cycle": info["resume_cycle"],
    }

  def _mixed() -> None:
    captured["mixed_selection"] = _mixed_estimates(hw)

  spec: list = [
    ("config-single-engine-pool" if args.case == "t06-one-engine-many-handles"
     else "config-registered-engine-pool",
     _single_engine if args.case == "t06-one-engine-many-handles" else _registered_pool),
    ("four-live-handles-quota-two-progress" if args.case != "t06-one-engine-many-handles"
     else "four-live-handles-one-engine-fair-progress", _progress),
    ("mixed-estimates-deterministic-selection", _mixed),
  ]
  if args.case == "engine-pool-progress" and arrays >= 2:
    spec += [
      ("busy-affinity-never-migrates-neighbor-progresses", lambda: check_busy(hw)),
      ("single-context-multistep-affinity-and-overlap", lambda: _multistep_overlap(hw)),
    ]
  if args.case == "engine-pool-progress" and arrays == 2:
    spec += [
      ("two-engines-four-contexts-eight-handles-quota-two",
       lambda: _two_engines_four_contexts(hw)),
      ("fault-drain-waits-for-inflight-before-pool-reset",
       lambda: _fault_drain_waits_for_inflight(hw)),
    ]

  checks = case_cli.run_checks(spec)
  out = args.out
  assert out is not None, "--out DIR is required with --case"
  out.mkdir(parents=True, exist_ok=True)
  trace_name = f"{args.case}.progress.trace.json"
  (out / trace_name).write_text(captured.get("progress_trace", ""), encoding="utf-8")
  report = case_cli.build_report(
    args.case,
    "test_boa_multi_engine",
    args.seed,
    checks,
    config={"hw_overrides": overrides, "boa_arrays_per_tile": arrays,
            "geometry": [hw.boa_array_rows, hw.boa_array_cols, hw.boa_tile_m, hw.boa_tile_n]},
    metrics={key: captured[key] for key in ("progress", "mixed_selection") if key in captured},
  )
  case_cli.write_report(out, report)
  return case_cli.finish(report)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(prog="python -m pipeline_validator.tests.test_boa_multi_engine",
                                   description=__doc__.splitlines()[0])
  parser.add_argument("--arrays-per-tile", type=int, choices=range(1, 9))
  parser.add_argument("--config")
  parser.add_argument("--trace-json", type=Path, help="save the actual progress-case trace")
  parser.add_argument("--case", choices=CLI_CASES,
                      help="registered container case (writes a report JSON to --out)")
  parser.add_argument("--out", type=Path, help="output directory for the case report (--case)")
  parser.add_argument("--seed", type=int, default=1,
                      help="recorded run seed; feeds SimConfig (inputs are deterministic)")
  args = parser.parse_args(argv)
  if args.case is not None and args.out is None:
    parser.error("--case requires --out DIR")
  try:
    overrides = parse_schema1_config(args.config) if args.config else {}
    if args.arrays_per_tile is not None:
      if overrides.get("boa_arrays_per_tile", args.arrays_per_tile) != args.arrays_per_tile:
        raise ValueError("config and --arrays-per-tile disagree")
      overrides["boa_arrays_per_tile"] = args.arrays_per_tile
    hw = HardwareConfig().with_overrides(**overrides)
    if args.case is not None:
      return _run_cli_case(args, hw, overrides)
    progress = run_case(hw, contexts=2, block=True, seed=args.seed)
    info = check_case(progress, contexts=2, steps=2)
    assert info["peak_live"] == 4
    mixed = run_case(hw, k=64, steps=1, estimates=(8192, 8192), seed=args.seed)
    report = {"ok": True, "arrays_per_tile": hw.boa_arrays_per_tile,
              "geometry": [hw.boa_array_rows, hw.boa_array_cols, hw.boa_tile_m, hw.boa_tile_n],
              "progress": info, "mixed": check_case(mixed, contexts=1, steps=1, estimates=2)}
    if hw.boa_arrays_per_tile >= 2:
      report["busy_no_migration"] = check_busy(hw)
    if args.trace_json is not None:
      assert progress.simulator.tracer is not None
      args.trace_json.parent.mkdir(parents=True, exist_ok=True)
      args.trace_json.write_text(progress.simulator.tracer.to_chrome_json())
  except (ValueError, TypeError, OSError) as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
    return 2
  except AssertionError as exc:
    print(json.dumps({"ok": False, "failure": str(exc)}))
    return 1
  print(json.dumps(report, indent=2, sort_keys=True))
  return 0

def _fault_drain_waits_for_inflight(hw: HardwareConfig) -> list[int]:
  from pipeline_validator.runtime import FaultCode

  injected: list[int] = []

  def fault_when_only_engine_one_is_busy(simulator: Simulator, cycle: int) -> None:
    pool = simulator.group.tiles[0].boa_engines
    if not injected and pool[1].is_busy and not pool[0].is_busy:
      simulator.group.trigger_fault(FaultCode.ENGINE_INTERNAL_FAULT, tile_id=0, cycle=cycle)
      injected.append(cycle)

  case = run_case(hw, steps=1, estimates=(2, LONG_ESTIMATE_OPS), before=True,
                  on_cycle=fault_when_only_engine_one_is_busy)
  assert len(injected) == 1 and not case.result.completed
  long_job = next(job for job in case.jobs if job["engine_id"] == 1)
  assert long_job["start"] <= injected[0] < long_job["end"] <= case.result.cycles
  assert case.simulator.group.reset_domain.is_done
  tile = case.simulator.group.tiles[0]
  assert all(not engine.has_accepted_jobs for engine in tile.engines)
  assert tile.boa_results.pending_count == 0
  assert tile.boa_results.credits == tile.boa_results.credit_capacity
  assert tile.shared_am.snapshot()["engine_handle_counts"] == dict.fromkeys(
    range(hw.boa_arrays_per_tile), 0
  )
  return injected


def test_fault_drain_waits_for_nonzero_engine_before_resetting_the_pool():
  _fault_drain_waits_for_inflight(HardwareConfig(boa_arrays_per_tile=2))


if __name__ == "__main__":
  raise SystemExit(main())
