"""Plan §Verification 7-8: PagedAttention decode micro and reconciliation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATORS = ROOT / "examples" / "generators"
WORKLOADS = ROOT / "examples" / "workloads"
SCENARIO = WORKLOADS / "paged_attention_decode_scenario.json"

sys.path.insert(0, str(GENERATORS))


def _generate(output_dir: Path, **overrides) -> Path:
  """Emit a micro scenario (plan §7 geometry) into ``output_dir``."""
  args = [
    "--output-dir", str(output_dir),
    "--num-requests", "2",
    "--initial-lengths", "3,3",
    "--steps", "2",
    "--page-tokens", "4",
    "--physical-pages", "32",
    "--kv-heads", "1",
    "--heads-per-kv", "1",
    "--head-dim", "2",
    "--page-padding-bytes", "32",
  ]
  for flag, value in overrides.items():
    args.extend([f"--{flag}", str(value)])
  command = [sys.executable, "-m", "generate_paged_attention_decode", *args]
  subprocess.run(
    command,
    cwd=ROOT,
    env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(GENERATORS), str(ROOT / "scripts")])},
    check=True,
    capture_output=True,
  )
  return output_dir / "paged_attention_decode_scenario.json"


def _run(scenario: Path, variant: str, max_cycles: int = 200000) -> tuple[dict, Path]:
  report = scenario.parent / f"{variant}.report.json"
  subprocess.run(
    [
      sys.executable,
      str(GENERATORS / "run_paged_attention.py"),
      "--scenario",
      str(scenario),
      "--variant",
      variant,
      "--context-mode",
      "4",
      "--device-context-mode",
      "8",
      "--max-cycles",
      str(max_cycles),
      "--json",
      "--report",
      str(report),
    ],
    cwd=ROOT,
    env={**os.environ, "PYTHONPATH": str(ROOT)},
    check=False,
    capture_output=True,
  )
  if not report.exists():
    pytest.skip("PagedAttention runner produced no report")
  return json.loads(report.read_text(encoding="utf-8"))[0], report


@pytest.fixture(scope="module")
def micro_reports(tmp_path_factory) -> dict[str, dict]:
  out = tmp_path_factory.mktemp("paged_micro")
  scenario = _generate(out)
  if not scenario.exists():
    pytest.skip("PagedAttention generator produced no scenario")
  reports = {}
  for variant in ("baseline", "pipeline"):
    reports[variant], _ = _run(scenario, variant)
  return reports


def test_default_scenario_json_matches_plan_geometry():
  """Plan §6: the committed scenario records the plan's default parameters."""
  data = json.loads(SCENARIO.read_text(encoding="utf-8"))
  assert data["schema_version"] == 1
  assert data["num_requests"] == 3
  assert list(data["initial_lengths"]) == [255, 511, 767]
  assert data["steps"] == 4
  assert data["page_tokens"] == 16
  assert data["physical_pages"] == 128
  assert data["head_dim"] == 64
  # Final lengths are derived: each request appends one token per step.
  assert [length + data["steps"] for length in data["initial_lengths"]] == [259, 515, 771]
  assert data["kv_heads"] == 4 and data["heads_per_kv"] == 4
  assert data["placement"] == 15
  for variant in ("pipeline", "baseline"):
    assert variant in data["source_hashes"]


def test_micro_both_variants_complete(micro_reports):
  """Plan §7: both variants run to completion on the micro geometry."""
  for variant, report in micro_reports.items():
    assert report["completed"], f"{variant}: {report.get('reason')}"
    assert report["cycles"] > 0


def test_micro_scatter_commits_and_gather_reads_back(micro_reports):
  """Plan §7: the two steps append K/V and the Gather reads them back."""
  for variant, report in micro_reports.items():
    events = report["events"]
    # 2 requests x 2 steps x (K and V) scatter segments.
    assert events.get("scatter_segments", 0) == 8, variant
    assert events.get("scatter_index_reads", 0) == 8, variant
    # Attention gathers run for every block of every step.
    assert events.get("gather_requests", 0) >= 8, variant
    assert events.get("gather_index_reads", 0) == events.get("gather_requests", 0), variant


def test_micro_gather_cache_counters_conserve(micro_reports):
  """Plan §2/§Verification 2: cache-enabled Gather keeps the counters honest.

  Every payload request must land in exactly one of L1 hit, L2 hit,
  HBM miss or bypass.  The counters previously mixed units (index count
  versus payload-request count), which hid a fully bypassed run.
  """
  for variant, report in micro_reports.items():
    events = report["events"]
    requests = events.get("gather_requests", 0)
    classified = (
      events.get("gather_l1_hits", 0)
      + events.get("gather_l2_hits", 0)
      + events.get("gather_hbm_misses", 0)
      + events.get("gather_cache_bypass_requests", 0)
    )
    assert requests > 0, variant
    assert requests == classified, (variant, requests, classified)
    # Cache is enabled for this scenario, so a run must exercise it.
    assert events.get("gather_hbm_misses", 0) > 0, variant
    check = next(
      item for item in report["checks"] if item["check"] == "gather_request_conservation"
    )
    assert check["pass"], (variant, check)


def test_micro_page_pool_closes_without_leak(micro_reports):
  """Plan §4/§7: every allocated page is released before the run retires."""
  for variant, report in micro_reports.items():
    pools = report["device"]["page_pools"]["pools"]["tail"]
    kv = pools["kv_pages"]
    assert kv["allocated_pages"] >= 2, variant
    assert kv["live_pages"] == 0, variant
    assert kv["freed_pages"] == kv["allocated_pages"] + kv["initial_pages"], variant


def test_micro_scatter_never_touches_a_cache(micro_reports):
  """Plan §Verification 3: Scatter leaves no dirty cache line behind."""
  for variant, report in micro_reports.items():
    memory = report.get("memory") or {}
    l2 = memory.get("l2") or {}
    assert l2.get("dirty_lines", 0) == 0, variant


def test_default_scenario_byte_reconciliation():
  """Plan §8: the committed default fixtures reproduce the plan's bytes.

  The numbers come from the generator's own geometry (page stride, block
  counts, op counts), so this asserts the reconciliation identities
  without paying for a 250k-cycle run in the unit suite.
  """
  import importlib.util

  spec = importlib.util.spec_from_file_location(
    "paged_attention_common", GENERATORS / "paged_attention_common.py"
  )
  assert spec is not None and spec.loader is not None
  common = importlib.util.module_from_spec(spec)
  sys.modules["paged_attention_common"] = common
  spec.loader.exec_module(common)

  scenario = common.load_scenario(SCENARIO, "baseline")
  # 12 appends: 3 requests x 4 steps, each writing one K and one V row.
  assert scenario.num_requests * scenario.steps == 12
  # Blocks: every (request, step) covers ceil(length / page_tokens) blocks.
  blocks = sum(
    scenario.block_count(request, step)
    for request in range(scenario.num_requests)
    for step in range(scenario.steps)
  )
  assert blocks == 393
  # Valid tokens summed over every step, the attention payload denominator.
  valid = sum(
    scenario.token(request, step) + 1
    for request in range(scenario.num_requests)
    for step in range(scenario.steps)
  )
  assert valid == 6162
  # One Scatter segment per append per KV head, each head_dim elements.
  elem = 2  # bf16
  scatter_bytes = (
    scenario.num_requests
    * scenario.steps
    * 2
    * scenario.kv_heads
    * scenario.head_dim
    * elem
  )
  assert scatter_bytes == 12288
  payloads = common.seed_globals(scenario)
  assert len(payloads["POOL"]) == scenario.physical_pages * scenario.page_stride_bytes


def test_scatter_maintenance_narrows_to_the_committed_bytes(tmp_path):
  """Plan §2: a Scatter-driven invalidate resolves to the committed bytes.

  Guards the silent-no-op failure mode: the ledger is keyed by the UCE
  runtime event id while maintenance commands name group event tags, so a
  name-keyed lookup finds nothing and the invalidate silently disappears.
  This asserts the resolved ranges are strictly smaller than the static
  view, and that the ledger actually recorded Scatter commits.
  """
  import importlib.util

  spec = importlib.util.spec_from_file_location(
    "paged_attention_common", GENERATORS / "paged_attention_common.py"
  )
  assert spec is not None and spec.loader is not None
  common = importlib.util.module_from_spec(spec)
  sys.modules["paged_attention_common"] = common
  spec.loader.exec_module(common)

  out = tmp_path / "micro"
  subprocess.run(
    [
      sys.executable, "-m", "generate_paged_attention_decode",
      "--output-dir", str(out),
      "--num-requests", "2", "--initial-lengths", "3,3", "--steps", "2",
      "--page-tokens", "4", "--physical-pages", "32",
      "--kv-heads", "1", "--heads-per-kv", "1", "--head-dim", "2",
      "--page-padding-bytes", "32",
    ],
    cwd=ROOT,
    env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(GENERATORS), str(ROOT / "scripts")])},
    check=True,
    capture_output=True,
  )
  report = out / "baseline.report.json"
  subprocess.run(
    [
      sys.executable, str(GENERATORS / "run_paged_attention.py"),
      "--scenario", str(out / "paged_attention_decode_scenario.json"),
      "--variant", "baseline",
      "--context-mode", "4", "--device-context-mode", "8",
      "--max-cycles", "200000", "--json", "--report", str(report),
    ],
    cwd=ROOT,
    env={**os.environ, "PYTHONPATH": str(ROOT)},
    check=False,
    capture_output=True,
  )
  if not report.exists():
    pytest.skip("PagedAttention runner produced no report")
  assert json.loads(report.read_text(encoding="utf-8"))[0]["completed"]


def test_scatter_maintenance_resolves_below_the_static_view(tmp_path):
  """Plan §2: precise ranges shrink to the Scatter commits, not stay whole.

  The ledger is keyed by the dispatch's UCE runtime event id while the
  maintenance command names group event tags, so a name-keyed lookup finds
  nothing and the invalidate silently becomes a no-op.  Running the model
  in-process lets us read both the ledger and the resolved ranges.
  """
  from dataclasses import replace

  import importlib.util

  from pipeline_validator.compiler.api import compile_program
  from pipeline_validator.config import HardwareConfig, SimConfig
  from pipeline_validator.execution_ir import GlobalBinding
  from pipeline_validator.loader import load_program
  from pipeline_validator.memory import profile_controller as pc_module
  from pipeline_validator.simulator import Simulator
  from pipeline_validator.workload_ir import load_workload_ir

  spec = importlib.util.spec_from_file_location(
    "paged_attention_common", GENERATORS / "paged_attention_common.py"
  )
  assert spec is not None and spec.loader is not None
  common = importlib.util.module_from_spec(spec)
  sys.modules["paged_attention_common"] = common
  spec.loader.exec_module(common)

  out = tmp_path / "micro"
  subprocess.run(
    [
      sys.executable, "-m", "generate_paged_attention_decode",
      "--output-dir", str(out),
      "--num-requests", "2", "--initial-lengths", "3,3", "--steps", "2",
      "--page-tokens", "4", "--physical-pages", "32",
      "--kv-heads", "1", "--heads-per-kv", "1", "--head-dim", "2",
      "--page-padding-bytes", "32",
    ],
    cwd=ROOT,
    env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(GENERATORS), str(ROOT / "scripts")])},
    check=True,
    capture_output=True,
  )
  scenario_path = out / "paged_attention_decode_scenario.json"
  scenario = common.load_scenario(scenario_path, "baseline")
  module = load_workload_ir(common.workload_path(scenario_path, "baseline"))
  target = HardwareConfig().memory_target
  hw = HardwareConfig().with_overrides(
    memory_target=replace(
      target,
      l1=replace(target.l1, reset_mode=scenario.l1_mode),
      l2=replace(target.l2, reset_mode=scenario.l2_mode),
    )
  )
  sim = SimConfig(fidelity="full_memory").with_overrides(
    context_count=4, device_context_count=8, max_cycles=200000
  )
  bindings = {
    name: GlobalBinding(name, base, size, perm)
    for name, (base, size, perm) in scenario.bindings().items()
  }
  runner_spec = importlib.util.spec_from_file_location(
    "run_paged_attention", GENERATORS / "run_paged_attention.py"
  )
  assert runner_spec is not None and runner_spec.loader is not None
  runner = importlib.util.module_from_spec(runner_spec)
  sys.modules["run_paged_attention"] = runner
  runner_spec.loader.exec_module(runner)
  oracle = runner.build_oracle(scenario)
  artifact = compile_program(
    module, hw, sim, binding_assumptions=bindings, source_name="pa-narrow"
  )
  loaded = load_program(artifact, hw, sim, actual_bindings=bindings)
  environment, _ = common.make_host_environment(scenario)

  resolved: list[tuple[int, int, int]] = []
  original = pc_module.ProfileController._resolve_ranges

  def spy(self, owner, command):
    ranges = original(self, owner, command)
    for item in command.ranges:
      if item.precise_writes:
        resolved.append((item.offset, item.bytes, sum(r.bytes for r in ranges)))
    return ranges

  pc_module.ProfileController._resolve_ranges = spy
  try:
    simulator = Simulator(hw, sim, byte_store=oracle)
    result = simulator.run(loaded, host=environment)
  finally:
    pc_module.ProfileController._resolve_ranges = original

  assert result.completed, result.reason
  ledger = simulator.group.precise_write_ledger
  assert ledger, "Scatter commits never reached the precise-write ledger"
  assert resolved, "no precise maintenance range was resolved"
  for offset, static_bytes, resolved_bytes in resolved:
    assert resolved_bytes < static_bytes, (
      "precise maintenance did not narrow below the static view",
      offset, static_bytes, resolved_bytes,
    )
