"""Plan §Verification 7-8: PagedAttention decode micro and reconciliation.

Everything drives the unified ``python -m pipeline_validator`` CLI:
``--scenario`` + ``--variant`` for source runs, ``--compiled-file`` for
source-free replay, and ``--l1-mode`` for the source-side L1 profile
rewrite.  The in-process reconciliation tests import the canonical
``pipeline_validator.paged_attention`` helpers directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline_validator import paged_attention as common

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = ROOT / "examples" / "workloads"
SCENARIO = WORKLOADS / "paged_attention_decode_scenario.json"

MICRO_GENERATE_ARGS = (
  "--num-requests", "2",
  "--initial-lengths", "3,3",
  "--steps", "2",
  "--page-tokens", "4",
  "--physical-pages", "32",
  "--kv-heads", "1",
  "--heads-per-kv", "1",
  "--head-dim", "2",
  "--page-padding-bytes", "32",
)


def _generate(output_dir: Path, **overrides) -> Path:
  """Emit a micro scenario (plan §7 geometry) into ``output_dir``."""
  args = ["--output-dir", str(output_dir)]
  args.extend(MICRO_GENERATE_ARGS)
  for flag, value in overrides.items():
    args.extend([f"--{flag.replace('_', '-')}", str(value)])
  subprocess.run(
    [sys.executable, "-m", "examples.generators.generate_paged_attention_decode", *args],
    cwd=ROOT,
    env={**os.environ, "PYTHONPATH": str(ROOT)},
    check=True,
    capture_output=True,
  )
  return output_dir / "paged_attention_decode_scenario.json"


def _cli(scenario: Path, *options: str, variant: str | None = "pipeline") -> subprocess.CompletedProcess:
  """Run the unified CLI on a scenario and return the completed process."""
  command = [sys.executable, "-m", "pipeline_validator", "--scenario", str(scenario)]
  if variant is not None:
    command.extend(["--variant", variant])
  return subprocess.run(
    [*command, *options],
    cwd=ROOT,
    env={**os.environ, "PYTHONPATH": str(ROOT)},
    capture_output=True,
    text=True,
    check=False,
  )


def _report(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))[0]


def _run(scenario: Path, variant: str, max_cycles: int = 200000) -> dict:
  """Run one variant to completion through the unified CLI."""
  report_path = scenario.parent / f"{variant}.report.json"
  process = _cli(
    scenario,
    "--context-mode", "4",
    "--device-context-mode", "8",
    "--max-cycles", str(max_cycles),
    "--json",
    "--report", str(report_path),
    variant=variant,
  )
  assert process.returncode == 0, (variant, process.returncode, process.stderr[-2000:])
  return _report(report_path)


@pytest.fixture(scope="module")
def micro_reports(tmp_path_factory) -> dict[str, dict]:
  out = tmp_path_factory.mktemp("paged_micro")
  scenario = _generate(out)
  return {variant: _run(scenario, variant) for variant in ("baseline", "pipeline")}


def test_cli_scenario_cycle_cap_and_explicit_override_precedence(tmp_path):
  scenario = _generate(tmp_path)
  data = json.loads(scenario.read_text(encoding="utf-8"))
  data["max_cycles"] = 1
  scenario.write_text(json.dumps(data), encoding="utf-8")
  report = tmp_path / "scenario_cap.report.json"
  process = _cli(scenario, "--device-context-mode", "8", "--json", "--report", str(report))
  assert process.returncode == 1, process.stderr
  capped = _report(report)
  assert not capped["completed"]
  assert "cycle cap 1 reached" in capped["reason"]
  report = tmp_path / "override_cap.report.json"
  process = _cli(
    scenario, "--device-context-mode", "8", "--json", "--report", str(report),
    "--sim-override", "max_cycles=200000",
  )
  assert process.returncode == 0, process.stderr
  assert _report(report)["completed"]
  report = tmp_path / "dedicated_cap.report.json"
  process = _cli(
    scenario, "--device-context-mode", "8", "--json", "--report", str(report),
    "--sim-override", "max_cycles=200000", "--max-cycles", "1",
  )
  assert process.returncode == 1, process.stderr
  capped = _report(report)
  assert not capped["completed"]
  assert "cycle cap 1 reached" in capped["reason"]


def test_cli_scenario_latency_hardware_file_and_override_precedence(tmp_path):
  scenario = _generate(tmp_path)
  data = json.loads(scenario.read_text(encoding="utf-8"))
  data["hbm_fixed_latency_cycles"] = 400
  scenario.write_text(json.dumps(data), encoding="utf-8")

  def _run_named(name: str, *options: str) -> tuple[int, dict]:
    report = tmp_path / f"{name}.report.json"
    process = _cli(
      scenario, "--device-context-mode", "8", "--json", "--report", str(report), *options,
    )
    assert report.exists(), process.stderr
    return process.returncode, _report(report)

  code, slow = _run_named("scenario_slow")
  assert code == 0 and slow["completed"], slow["reason"]
  data["hbm_fixed_latency_cycles"] = 10
  scenario.write_text(json.dumps(data), encoding="utf-8")
  code, fast = _run_named("scenario_fast")
  assert code == 0 and fast["completed"], fast["reason"]
  assert fast["cycles"] < slow["cycles"]

  data["hbm_fixed_latency_cycles"] = 400
  scenario.write_text(json.dumps(data), encoding="utf-8")
  hardware = tmp_path / "hardware.yaml"
  hardware.write_text(
    "memory:\n  hbm:\n    fixed_latency_cycles: 10\nfabric:\n  dma:\n    channels: 2\n",
    encoding="utf-8",
  )
  code, selected = _run_named("hardware_file", "--hw-config", str(hardware))
  assert code == 0 and selected["cycles"] == fast["cycles"], selected["reason"]
  code, overridden = _run_named(
    "hardware_override", "--hw-config", str(hardware),
    "--hw-override", "hbm_fixed_latency_cycles=400",
  )
  assert code == 0 and overridden["cycles"] == slow["cycles"], overridden["reason"]


def test_micro_both_variants_complete(micro_reports):
  """Plan §7: both variants run to completion with every check green."""
  for variant, report in micro_reports.items():
    assert report["completed"], f"{variant}: {report.get('reason')}"
    # Arena/page/gather/scatter conservation invariants all hold.
    assert all(check["pass"] for check in report["checks"]), (
      variant,
      [check for check in report["checks"] if not check["pass"]],
    )


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


def test_scatter_maintenance_resolves_below_the_static_view(tmp_path):
  """Plan §2: precise ranges shrink to the Scatter commits, not stay whole.

  The ledger is keyed by the dispatch's UCE runtime event id while the
  maintenance command names group event tags, so a name-keyed lookup finds
  nothing and the invalidate silently becomes a no-op.  Running the model
  in-process lets us read both the ledger and the resolved ranges.
  """
  from dataclasses import replace

  from pipeline_validator.compiler.api import compile_program
  from pipeline_validator.config import HardwareConfig, SimConfig
  from pipeline_validator.loader import load_program
  from pipeline_validator.memory import profile_controller as pc_module
  from pipeline_validator.simulator import Simulator
  from pipeline_validator.workload_ir import load_workload_ir

  out = tmp_path / "micro"
  scenario_path = _generate(out)
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
  bindings = common.build_bindings(scenario)
  oracle = common.build_oracle(scenario, bindings)
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


def test_cli_l1_mode_switches_actual_l1_cache_behavior(tmp_path):
  """--l1-mode rewires the compiled L1 cache without touching any file.

  The baseline retains mode 0; the pipeline contract explicitly permits
  only modes 1–3.

  The geometry keeps a full page untouched across steps (lengths 8 grow
  to 10, so each step-1 append lands in a third page): the step-1 blocks
  re-gather pages 0 and 1 unchanged, so the cache-enabled run must show
  real L1 hits, while --l1-mode 0 disables the L1 cache and the same
  source runs with no L1 hits at all.
  """
  scenario = _generate(tmp_path, initial_lengths="8,8")
  before = {
    path: path.read_bytes()
    for path in tmp_path.iterdir()
    if path.suffix in (".mlir", ".json")
  }

  enabled_report = tmp_path / "l1_enabled.report.json"
  process = _cli(
    scenario,
    "--context-mode", "4", "--device-context-mode", "8",
    "--max-cycles", "200000", "--l1-mode", "3",
    "--json", "--report", str(enabled_report),
    variant="baseline",
  )
  assert process.returncode == 0, process.stderr
  report = _report(enabled_report)
  assert report["completed"], report["reason"]
  events = report["events"]
  requests = events["gather_requests"]
  assert requests > 0
  assert requests == (
    events["gather_l1_hits"]
    + events["gather_l2_hits"]
    + events["gather_hbm_misses"]
    + events["gather_cache_bypass_requests"]
  )
  assert events["gather_l1_hits"] > 0, "cache-enabled run never reused a resident line"

  disabled_report = tmp_path / "l1_disabled.report.json"
  process = _cli(
    scenario,
    "--context-mode", "4", "--device-context-mode", "8",
    "--max-cycles", "200000", "--l1-mode", "0",
    "--json", "--report", str(disabled_report),
    variant="baseline",
  )
  assert process.returncode == 0, (process.returncode, process.stderr[-2000:])
  report = _report(disabled_report)
  assert report["completed"], report["reason"]
  events = report["events"]
  requests = events["gather_requests"]
  assert requests > 0
  assert requests == (
    events["gather_l1_hits"]
    + events["gather_l2_hits"]
    + events["gather_hbm_misses"]
    + events["gather_cache_bypass_requests"]
  )
  assert events["gather_l1_hits"] == 0, "L1 cache served hits while disabled"
  # L2 stays enabled, so the re-gathered untouched pages must still hit
  # there — an all-bypass mode-0 run would be a silent downgrade.
  assert events["gather_l2_hits"] > 0, "L2 cache never reused a resident line"

  after = {path: path.read_bytes() for path in before}
  assert after == before, "run modified the source fixtures or scenario"
  # The recorded source hashes still verify after both invocations.
  common.load_scenario(scenario, "baseline")


def test_cli_pipeline_rejects_spm_only_l1_mode(tmp_path):
  """A runtime override cannot widen the pipeline's allowed profile contract."""
  scenario = _generate(tmp_path)
  artifact = tmp_path / "disallowed.json"
  report = tmp_path / "disallowed.report.json"
  process = _cli(
    scenario, "--l1-mode", "0", "--compiled-output", str(artifact),
    "--json", "--report", str(report),
  )
  assert process.returncode == 2, process.stderr
  assert not artifact.exists()
  assert not report.exists()


def test_cli_compiled_replay_runs_without_source_or_compiler(tmp_path):
  """--scenario + --compiled-file replays after the .mlir is deleted.

  The replay subprocess blocks every ``pipeline_validator.compiler`` and
  ``examples.generators`` import, so a successful run proves the source-
  free path needs neither the compiler nor the generator package.
  """
  scenario = _generate(tmp_path, initial_lengths="8,8")
  artifact = tmp_path / "replay.json"
  process = _cli(
    scenario,
    "--context-mode", "4", "--device-context-mode", "8",
    "--max-cycles", "200000", "--l1-mode", "3",
    "--compile-only", "--compiled-output", str(artifact),
  )
  assert process.returncode == 0, process.stderr
  target = tmp_path / "replay.target.yaml"
  assert artifact.exists() and target.exists()

  for variant in ("pipeline", "baseline"):
    (tmp_path / f"paged_attention_decode_{variant}.mlir").unlink()

  blocker = tmp_path / "no_compiler_imports"
  blocker.mkdir()
  (blocker / "sitecustomize.py").write_text(
    "import sys\n"
    "_BLOCKED = ('pipeline_validator.compiler', 'examples.generators')\n"
    "class _Blocked:\n"
    "    def find_spec(self, fullname, path=None, target=None):\n"
    "        for prefix in _BLOCKED:\n"
    "            if fullname == prefix or fullname.startswith(prefix + '.'):\n"
    "                raise ImportError('blocked by test: ' + fullname)\n"
    "        return None\n"
    "sys.meta_path.insert(0, _Blocked())\n",
    encoding="utf-8",
  )
  replay_report = tmp_path / "replay.run.report.json"
  process = subprocess.run(
    [
      sys.executable, "-m", "pipeline_validator",
      "--scenario", str(scenario),
      "--compiled-file", str(artifact),
      "--hw-config", str(target),
      "--context-mode", "4", "--device-context-mode", "8",
      "--json", "--report", str(replay_report),
    ],
    cwd=ROOT,
    env={
      **os.environ,
      "PYTHONPATH": os.pathsep.join([str(blocker), str(ROOT)]),
    },
    capture_output=True,
    text=True,
    check=False,
  )
  assert process.returncode == 0, process.stderr
  report = _report(replay_report)
  assert report["completed"], report["reason"]
  assert report["events"]["gather_l1_hits"] > 0
  assert all(check["pass"] for check in report["checks"]), (
    [check for check in report["checks"] if not check["pass"]]
  )


def test_cli_source_hash_mismatch_fails_before_artifacts(tmp_path):
  """A tampered fixture fails scenario load before any artifact or report."""
  scenario = _generate(tmp_path)
  source = tmp_path / "paged_attention_decode_pipeline.mlir"
  source.write_bytes(source.read_bytes() + b"\n// tampered\n")
  report = tmp_path / "mismatch.report.json"
  artifact = tmp_path / "mismatch.json"
  process = _cli(
    scenario,
    "--json", "--report", str(report), "--compiled-output", str(artifact),
  )
  assert process.returncode == 2, (process.returncode, process.stderr)
  assert "failed to load scenario" in process.stderr
  assert "source_hash mismatch" in process.stderr
  assert not report.exists()
  assert not artifact.exists()
  assert not artifact.with_suffix("").with_name("mismatch.target.yaml").exists()


def test_cli_l1_mode_rejected_with_compiled_file(tmp_path):
  """--l1-mode is a source-compilation option; replay rejects it upfront."""
  scenario = _generate(tmp_path)
  artifact = tmp_path / "replay.json"
  process = _cli(
    scenario,
    "--context-mode", "4", "--device-context-mode", "8",
    "--max-cycles", "200000", "--compile-only", "--compiled-output", str(artifact),
  )
  assert process.returncode == 0, process.stderr
  report = tmp_path / "rejected.report.json"
  process = _cli(
    scenario,
    "--compiled-file", str(artifact), "--l1-mode", "3",
    "--report", str(report),
  )
  assert process.returncode == 2, (process.returncode, process.stderr)
  assert "--l1-mode" in process.stderr
  assert not report.exists()
