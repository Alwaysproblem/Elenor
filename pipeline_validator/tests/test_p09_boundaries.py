"""P09 negative/boundary set: only cases without an existing regression.

Each test names the already-covered siblings it complements:

- zero-K/zero-dimension work (bytes == committed P03 golden, including the
  zero-K-with-bias vector) is owned by
  ``test_boa_numeric_profile.py::test_zero_work_production_paths``;
- the runtime bundle/quota rejection face (two unaccepted FINAL chains, third
  request rejected with no side effect, T01) is owned by
  ``test_boa_am.py::test_ir09_allocation_beyond_bundle_is_rejected_without_side_effect``;
- the legacy v4 artifact version gate is owned by
  ``test_boa_am.py`` (LEGACY_V4_ARTIFACT) and
  ``test_boa_numeric_profile.py::test_cutover_rejections``; the full edited
  version matrix by ``test_compiler.py::TestExplicitCompileLoadReplay``;
- unknown ``numeric_profile`` values at config load are owned by
  ``test_boa_numeric_config.py::test_invalid_numeric_configuration_is_not_loaded``;
- INT8 illegal epilogue parameters are owned by
  ``test_boa_am.py::test_am_i8_output_requires_quantization_parameters`` and
  ``test_soft_conv.py::test_param_rejects_at_real_boundaries``;
- f32 FINAL output backpressure (Q07) is owned by
  ``test_boa_am.py::test_ir14_runtime_final_output_backpressure_recovers_without_polling``
  and ``test_am_terminal_order.py``.

New here: the real retained pre-P08 (schema 5 / ABI v5) artifact rejection,
v6 wrong-target rejection through the CLI, the numeric-profile fingerprint
identity, the static third-chain faces (T01), engine-queue FULL keeping seq
(T14), INT8 FINAL backpressure recovery (Q07), and mixed-fault counting with
the separate denominators mandated by the shared contract §5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from xdsl.utils.exceptions import VerifyException

from pipeline_validator.cli import main as cli_main
from pipeline_validator.compiled_program import parse_compiled_program, serialize_compiled_program
from pipeline_validator.compiler import compile_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.execution_verifier import target_fingerprint
from pipeline_validator.loader import load_program
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.simulator import Simulator
from pipeline_validator.tests.test_boa_multi_engine import _operands, _source, run_case
from pipeline_validator.tests.test_boa_numeric_profile import _layout, build_case_module
from pipeline_validator.workload_ir import parse_workload_ir

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "examples" / "workloads" / "boa_am_matmul.mlir"
INPUTS = ROOT / "examples" / "workloads" / "boa_am_inputs" / "boa_am_matmul"
# Verbatim pre-P08 capture (schema 5 / compiler ABI v5, retained by the P06
# evidence); parsing it under the current compiler must fail at the version
# gate exactly like the v4 capture, never be silently upgraded.
LEGACY_V5_ARTIFACT = ROOT / "pipeline_validator" / "tests" / "data" / "legacy_v5_boa_am.json"
VECTORS = ROOT / "hardware" / "boa" / "reference" / "vectors" / "matmul.json"


# ---------------------------------------------------------------------------
# Case 2: old artifacts and wrong-target v6 artifacts (public CLI entries)
# ---------------------------------------------------------------------------


def test_real_pre_p08_v5_artifact_is_rejected_on_version(capsys):
  text = LEGACY_V5_ARTIFACT.read_text(encoding="utf-8")
  document = json.loads(text)
  assert (document["schema_version"], document["compiler_abi"]) == (5, "v5")
  with pytest.raises(ValueError, match="unsupported compiled schema or ABI") as excinfo:
    parse_compiled_program(text)
  assert "recompile" in str(excinfo.value)
  assert "fingerprint" not in str(excinfo.value)
  # Same gate through the CLI entry: exit code 2 with the reason on stderr.
  assert cli_main(["--compiled-file", str(LEGACY_V5_ARTIFACT)]) == 2
  stderr = capsys.readouterr().err
  assert "unsupported compiled schema or ABI" in stderr


@pytest.fixture(scope="module")
def v6_artifact(tmp_path_factory):
  """One fresh v6 artifact compiled through the real compiler entry."""
  a = (INPUTS / "A.bin").read_bytes()
  b = (INPUTS / "B.bin").read_bytes()
  bindings = {
    "A": GlobalBinding("A", 0x100000, len(a), "r"),
    "B": GlobalBinding("B", 0x200000, len(b), "r"),
    "C": GlobalBinding("C", 0x300000, 65536, "w"),
  }
  hw = HardwareConfig().with_overrides(boa_am_slot_capacity_bytes=65536)
  sim = SimConfig(fidelity="full_memory", context_count=1)
  program = compile_program(
    parse_workload_ir(FIXTURE.read_text()), hw, sim, binding_assumptions=bindings, source_name=str(FIXTURE)
  )
  assert (program.schema_version, program.compiler_abi) == (6, "v6")
  path = tmp_path_factory.mktemp("p09_v6") / "v6_boa_am.json"
  path.write_text(serialize_compiled_program(program), encoding="utf-8")
  return path, bindings


@pytest.mark.parametrize(
  ("key", "value"),
  [
    ("boa_array_rows", 8),  # different valid geometry
    ("boa_arrays_per_tile", 2),  # different valid engine pool
    ("boa_numeric_tile_k", 16),  # different valid numeric block length
  ],
)
def test_v6_artifact_rejected_on_target_mismatch_via_cli(v6_artifact, capsys, key, value):
  path, _bindings = v6_artifact
  # Hold every other fingerprint field at the artifact's compiled value (the
  # fixture compiles with slot capacity 65536, not the 262144 default), so the
  # mismatch can only come from the one field under test.
  args = [
    "--compiled-file",
    str(path),
    "--hw-override",
    "boa_am_slot_capacity_bytes=65536",
    "--hw-override",
    f"{key}={value}",
  ]
  assert cli_main(args) == 2
  stderr = capsys.readouterr().err
  assert "target fingerprint mismatch" in stderr
  # Discriminating control: with only the capacity pinned back, the load
  # passes the fingerprint stage (it fails later on the missing bindings,
  # a different message), so the tested field alone moves the target.
  assert cli_main(args[:-2]) == 2
  control_stderr = capsys.readouterr().err
  assert "target fingerprint mismatch" not in control_stderr
  assert "bindings" in control_stderr


def test_v6_artifact_loads_under_its_own_target(v6_artifact):
  path, bindings = v6_artifact
  hw = HardwareConfig().with_overrides(boa_am_slot_capacity_bytes=65536)
  sim = SimConfig(fidelity="full_memory", context_count=1)
  program = parse_compiled_program(path.read_text(encoding="utf-8"))
  load_program(program, hw, sim, actual_bindings=bindings)  # not raising is the control


# ---------------------------------------------------------------------------
# Case 3: unknown numeric profile vs fingerprint identity
# ---------------------------------------------------------------------------


def test_unknown_numeric_profile_rejected_before_any_target_hash(capsys):
  # Rejection happens while the configuration object is being built: an
  # invalid profile never yields a HardwareConfig, so no target hash can be
  # requested for it (the artifact/compile stage is never reached).
  with pytest.raises(ValueError, match="boa_numeric_profile"):
    HardwareConfig(boa_numeric_profile="binary64")
  with pytest.raises(ValueError, match="boa_numeric_profile"):
    HardwareConfig().with_overrides(boa_numeric_profile="binary64")
  assert cli_main(["--ir-file", str(FIXTURE), "--hw-override", "boa_numeric_profile=binary64"]) == 2
  stderr = capsys.readouterr().err
  assert "failed to load hardware configuration" in stderr
  assert "boa_numeric_profile must be 'bf16_fp32_mac_rne_v1'" in stderr


def test_legal_numeric_profile_identity_is_in_the_fingerprint_payload(v6_artifact):
  path, bindings = v6_artifact
  sim = SimConfig(fidelity="full_memory", context_count=1)
  # Same target as the artifact except for the profile identity under probe.
  hw = HardwareConfig().with_overrides(boa_am_slot_capacity_bytes=65536)
  base = target_fingerprint(hw, sim)
  assert hw.boa_numeric_profile == "bf16_fp32_mac_rne_v1"
  program = parse_compiled_program(path.read_text(encoding="utf-8"))
  load_program(program, hw, sim, actual_bindings=bindings)  # identical target loads
  # Digest probe only: bypass the frozen field to prove the profile identity
  # string itself is a fingerprint input (it is not derived away).
  object.__setattr__(hw, "boa_numeric_profile", "probe_identity_v0")
  assert target_fingerprint(hw, sim) != base
  # A v6 artifact compiled under the legal profile therefore cannot load under
  # any other profile identity: wrong semantics can never execute.
  with pytest.raises(ValueError, match="target fingerprint mismatch"):
    load_program(program, hw, sim)


# ---------------------------------------------------------------------------
# Case 4: quota/bundle static faces (T01) -- runtime face see test_boa_am.py
# ---------------------------------------------------------------------------


def _third_chain_source() -> str:
  """The legal two-chain program plus an inadmissible third AM allocation."""
  source = _source(1, 8, 2, (), False, False)
  lines = source.split("\n")
  out: list[str] = []
  for line in lines:
    out.append(line)
    if "logical_slot = 1" in line and 'memory_space = "am"' in line:
      indent = line[: len(line) - len(line.lstrip())]
      out.append(
        f'{indent}%outX = tile.alloc memory_space = "l1" shape = [16, 16] dtype = "f32" '
        "alignment = 256 : !tile.l1_buffer<16x16xf32>"
      )
      out.append(
        f'{indent}%accX = tile.alloc memory_space = "am" logical_slot = 0 output(%outX) '
        'shape = [16, 16] dtype = "f32" alignment = 256 : !tile.am_buffer<16x16xf32>'
      )
  return "\n".join(out)


def test_third_am_chain_is_rejected_by_the_source_verifier():
  # Two chains already hold logical slots 0 and 1 with unaccepted FINALs; a
  # third concurrent chain cannot even be expressed: the only two logical
  # slots are live, so the third allocation is a static contract error.
  with pytest.raises((VerifyException, ValueError), match=r"already live in this tile\.program"):
    parse_workload_ir(_third_chain_source())


def test_two_live_chains_exceeding_configured_quota_are_rejected_at_compile():
  # Same program, but the target advertises a per-context quota of one lease:
  # the compiler's static bundle proof must refuse it before any runtime.
  source = _source(1, 8, 2, (), False, False)
  hw = HardwareConfig(boa_am_slots_per_context=1)
  sim = SimConfig(fidelity="full_memory", context_count=1)
  with pytest.raises(ValueError, match="exceeds the per-context AM lease quota"):
    compile_program(parse_workload_ir(source), hw, sim)


# ---------------------------------------------------------------------------
# Case 4: FULL -> WAIT_ENGINE_QUEUE keeps seq (T14)
# ---------------------------------------------------------------------------


def _context_state_slices(events, unit, state):
  return [
    {
      "ctx": int(e["cat"].split("CTX")[-1]),
      "start": round(e["ts"] / unit),
      "end": round((e["ts"] + e["dur"]) / unit),
    }
    for e in events
    if e.get("ph") == "X"
    and e.get("cat", "").startswith("UCE CTX")
    and str(e.get("name", "")).split(":")[0] == state
  ]


def test_engine_queue_full_keeps_seq_and_neighbor_contexts_progress():
  # One BOA engine (depth-1 launch FIFO) shared by two contexts: while a
  # queued step occupies the FIFO, the other context's launch is answered
  # FULL, the context parks in WAIT_ENGINE_QUEUE, and its instruction stays
  # at the same PC with the same descriptor seq -- nothing is consumed until
  # the enqueue really happens.
  from pipeline_validator.tests.test_boa_multi_engine import check_case

  case = run_case(HardwareConfig(boa_arrays_per_tile=1), contexts=2, steps=2, k=64)
  info = check_case(case, contexts=2, steps=2)  # completion, bytes, counters
  unit = case.simulator.tracer.cycle_to_us(1)
  blocked = _context_state_slices(case.events, unit, "WAIT_ENGINE_QUEUE")
  assert blocked, "the single-engine contention must actually produce FULL waits"
  typed = [job for job in case.jobs if job.get("lease_id") is not None]
  # seq kept: every chain committed each descriptor seq exactly once (a
  # consumed seq at the FULL attempt would have faulted the later begin_step).
  assert info["engines"][0]["complete_count"] == len(typed) == 8  # 4 chains x 2 steps
  for slice_ in blocked:
    waiter = slice_["ctx"]
    neighbors = [
      j
      for j in typed
      if j["ctx_id"] != waiter and j["start"] < slice_["end"] and j["end"] > slice_["start"]
    ]
    assert neighbors, "another context must keep progressing during the FULL wait"


# ---------------------------------------------------------------------------
# Case 5: INT8 FINAL backpressure recovery (Q07)
# ---------------------------------------------------------------------------


def test_i8_final_backpressure_holds_lease_then_completes_autonomously():
  document = json.loads(VECTORS.read_text(encoding="utf-8"))
  vector = next(item for item in document["cases"] if item["id"] == "mm-stages-i8")
  inp, golden = vector["input"], vector["expected"]["output_hex"]
  module = build_case_module("mm_stages_i8_bp", inp["m"], inp["n"], inp["steps"], "i8", inp["epilogue"])
  bindings, seeds, c_address, c_bytes = _layout(inp["m"], inp["n"], inp["steps"], "i8")
  hw = HardwareConfig().with_overrides(boa_numeric_tile_k=inp["tile_k"])
  sim = SimConfig(fidelity="full_memory", context_count=1, seed=20261031)
  artifact = compile_program(
    parse_workload_ir(module), hw, sim, binding_assumptions=bindings, source_name="p09-i8-bp"
  )
  loaded = load_program(
    parse_compiled_program(serialize_compiled_program(artifact)), hw, sim, actual_bindings=bindings
  )
  store = ByteStore()
  for address, data in seeds:
    store.seed_hbm(address, data)
  simulator = Simulator(hw, sim, enable_tracer=True, byte_store=store)
  tile = simulator.group.tiles[0]
  observed: dict = {}
  state = {"resume": None}
  original_step = simulator.group.step
  tile.boa_results.set_output_blocked(True)

  def step(cycle: int) -> bool:
    service = tile.boa_results
    if state["resume"] is None and service.snapshot()["states"]["wait_output"]:
      manager = tile.shared_am.snapshot()
      observed["blocked_at"] = cycle
      observed["leases"] = manager["leases"]
      observed["live"] = manager["live_leases"]
      observed["free_slots"] = manager["free_slots"]
      state["resume"] = cycle + 50
    if state["resume"] is not None and cycle >= state["resume"]:
      service.set_output_blocked(False)
    return original_step(cycle)

  with pytest.MonkeyPatch.context() as patch:
    patch.setattr(simulator.group, "step", step)
    result = simulator.run(loaded)
  assert result.completed, result.reason
  # While blocked the lease is kept: FINAL committed, one open AM access,
  # slot not returned, no artificial release by the UCE.
  assert observed["live"] == 1 and observed["free_slots"] == 7
  (lease,) = observed["leases"]
  assert lease["accesses"] == 1 and lease["output"] != ""
  service = tile.boa_results.snapshot()
  manager = tile.shared_am.snapshot()
  (terminal,) = manager["terminals"]
  # After the block lifts the chain writes, frees and publishes on its own.
  assert service["blocked_cycles"] > 0 and service["output_write_count"] == 1
  assert (terminal["kind"], terminal["success"]) == ("chain_done", True)
  assert store.read_hbm(c_address, c_bytes) == bytes.fromhex(golden)
  # Event order: commit -> output -> slot_free -> terminal -> event.
  unit = simulator.tracer.cycle_to_us(1)
  marks = [
    dict(event["args"], name=event["name"], cycle=round(event["ts"] / unit))
    for event in json.loads(simulator.tracer.to_chrome_json())["traceEvents"]
    if event.get("ph") == "i" and event.get("cat") in ("SharedAM", "BoaResult")
  ]

  def one(name, **match):
    hits = [m for m in marks if m["name"] == name and all(m.get(k) == v for k, v in match.items())]
    assert len(hits) == 1, (name, match, hits)
    return hits[0]

  record = next(r for r in service["records"] if r["final"])
  commit = one("am_commit", lease_id=terminal["lease_id"], final=True)
  output = one("boa_result_output_written", event_id=record["event_id"])
  release = one("am_release", lease_id=terminal["lease_id"])
  chain_terminal = one("am_chain_terminal", lease_id=terminal["lease_id"])
  ready = one("boa_result_ready", event_id=record["event_id"])
  cycles = [commit["cycle"], output["cycle"], release["cycle"], chain_terminal["cycle"], ready["cycle"]]
  assert cycles == sorted(cycles) and output["cycle"] >= observed["blocked_at"] + 50


# ---------------------------------------------------------------------------
# Case 6: mixed-fault counting with separate denominators
# ---------------------------------------------------------------------------


def test_mixed_fault_run_counts_rejections_park_cancels_and_faults_separately():
  # Three contexts, one BOA engine and only two physical AM slots: the first
  # context's two chains hold both slots, the other contexts' requests park.
  # Faulting one parked owner cancels its request; the group fault handling
  # reclaims the granted leases. The success identity (request == grant ==
  # ALLOC) must NOT be applied to this run; the park closure
  # parked == normal wakes + parked cancels must hold, and every granted
  # lease ends in a fault terminal.
  contexts, k, steps = 3, 64, 2
  hw = HardwareConfig(boa_arrays_per_tile=1, boa_am_group_count=1, boa_am_slots_per_group=2)
  sim_cfg = SimConfig(fidelity="full_memory", context_count=contexts)
  bindings: dict[str, GlobalBinding] = {}
  seeds: list[tuple[int, bytes]] = []
  for ctx in range(contexts):
    for chain in range(2):
      a, b, _c = _operands(ctx * 2 + chain, k, steps)
      for kind, data in (("a", a), ("b", b)):
        name = f"p{ctx}_{kind}{chain}"
        base = 0x100000 + len(bindings) * 0x10000
        bindings[name] = GlobalBinding(name, base, len(data), "r")
        seeds.append((base, data))
      name = f"p{ctx}_c{chain}"
      bindings[name] = GlobalBinding(name, 0x100000 + len(bindings) * 0x10000, 16 * 16 * 4, "w")
  source = _source(contexts, k, steps, (), False, False)
  artifact = compile_program(
    parse_workload_ir(source), hw, sim_cfg, binding_assumptions=bindings, source_name="p09-mixed-fault"
  )
  store = ByteStore()
  for address, data in seeds:
    store.seed_hbm(address, data)
  simulator = Simulator(hw, sim_cfg, enable_tracer=True, byte_store=store)
  tile = simulator.group.tiles[0]
  injected: list[tuple[str, int]] = []
  original_step = simulator.group.step

  def step(cycle: int) -> bool:
    manager = tile.shared_am
    if not injected and manager.snapshot()["pending_requests"] >= 2:
      uce = tile.uce
      uce._fault_context(uce.contexts[2], "cancel parked ctx2", cycle)
      injected.append(("fault_ctx2", cycle))
    return original_step(cycle)

  with pytest.MonkeyPatch.context() as patch:
    patch.setattr(simulator.group, "step", step)
    result = simulator.run(load_program(artifact, hw, sim_cfg, actual_bindings=bindings))
  assert injected and not result.completed
  assert "fault" in str(result.reason).lower()
  manager = tile.shared_am.snapshot()
  # Separate denominators, recorded independently.
  parked_cancels = manager["cancelled_request_count"]
  granted_faulted = manager["fault_count"]
  terminals = manager["terminals"]
  assert parked_cancels >= 1, "the parked request of the faulted owner must be cancelled"
  assert granted_faulted >= 1, "granted leases must be reclaimed as faults"
  assert any(t["kind"] == "fault" and not t["success"] for t in terminals)
  # Closure: every parked request either woke exactly once or was cancelled.
  assert manager["parked_count"] == manager["wake_count"] + parked_cancels
  assert manager["pending_requests"] == 0
  # Mixed bookkeeping identity (not the all-success F03 identity).
  assert manager["request_count"] == manager["acquire_count"] + parked_cancels
  assert manager["request_count"] != manager["acquire_count"]
  # Quiesced end state: nothing live, nothing reserved, every slot returned.
  assert manager["live_leases"] == 0 and manager["reserved_slots"] == 0
  assert manager["free_slots"] == manager["total_slots"]
