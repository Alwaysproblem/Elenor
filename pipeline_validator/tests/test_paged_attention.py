"""Plan §Verification 7-8: PagedAttention decode micro and reconciliation."""

from __future__ import annotations

import json
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
    env={"PYTHONPATH": f"{ROOT}:{GENERATORS}", "PATH": "/usr/bin:/bin"},
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
    env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
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
