"""Observable ordering contracts shared by CPU and FPGA-oriented models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from xdsl.utils.exceptions import VerifyException

from pipeline_validator.compiler import compile_program
from pipeline_validator.config import DeviceConfig, GroupSchedulerConfig, HardwareConfig, SimConfig
from pipeline_validator.dialects.elenor import NexusProgramOp, NexusSubmitContextOp
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.simulator import Simulator
from pipeline_validator.workload_ir import parse_workload_ir, verify_workload_ir

SCENARIOS = Path(__file__).resolve().parents[2] / "examples" / "scenarios"
BRANCH = (SCENARIOS / "ready_action_branch.mlir").read_text()
BINDINGS = {"arena": GlobalBinding("arena", 0x100000, 8192, "rw")}


def run_source(simulator: Simulator, module, bindings: dict[str, GlobalBinding] | None = None):
  """Exercise the public compile -> load -> run path used by every source fixture."""
  artifact = compile_program(module, simulator.hw, simulator.sim)
  loaded = load_program(artifact, simulator.hw, simulator.sim, actual_bindings=bindings)
  return simulator.run(loaded)


def run_branch(policy: str, text: str = BRANCH, *, window: int = 16):
  hardware = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  config = SimConfig(
    context_count=2,
    max_cycles=100000,
    group=GroupSchedulerConfig(
      policy=policy, action_capacity=window, context_action_quota=min(window, 8), scan_width=min(window, 4)
    ),
  )
  simulator = Simulator(hardware, config, enable_tracer=True)
  result = run_source(simulator, parse_workload_ir(text), BINDINGS)
  assert result.completed, result.reason
  assert result.credit_invariant_ok
  assert result.tracer is not None
  events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
  transfers = {
    event["args"]["event_id"].rsplit("_", 1)[-1]: event["args"]
    for event in events
    if event.get("ph") == "X" and event.get("args", {}).get("summary_kind") == "group_transfer"
  }
  return result, transfers, events


def test_ready_action_bypasses_independent_branch_but_head_policy_cannot():
  _, head, _ = run_branch("s0")
  _, ready, _ = run_branch("s1")
  assert head["pb"]["start_cycle"] >= head["sa"]["completion_cycle"]
  assert ready["pb"]["start_cycle"] < ready["sa"]["completion_cycle"]
  assert set(ready) == {"pa", "sa", "pb", "sb"}


@pytest.mark.parametrize("policy", ["s0", "s1"])
def test_single_action_slot_still_drains_both_branches(policy):
  result, transfers, _ = run_branch(policy, window=1)
  assert set(transfers) == {"pa", "sa", "pb", "sb"}
  assert all(transfer["completion_cycle"] <= result.cycles for transfer in transfers.values())
  assert result.group_snapshot["scheduler"]["queued_peak"] <= 1


def test_global_store_to_prefetch_alias_preserves_real_data_dependency():
  dependent = BRANCH.replace("offsets = [2048]", "offsets = [1024]")
  _, transfers, _ = run_branch("s1", dependent)
  assert transfers["pb"]["start_cycle"] >= transfers["sa"]["completion_cycle"]


def test_explicit_context_barrier_prevents_independent_branch_bypass():
  fenced = BRANCH.replace(
    "    %pb = nest.dma.prefetch.async", "    nest.barrier\n    %pb = nest.dma.prefetch.async"
  )
  _, transfers, _ = run_branch("s1", fenced)
  assert transfers["pb"]["start_cycle"] >= transfers["sa"]["completion_cycle"]


def test_epoch_policy_blocks_only_incompatible_compute_not_prefetch():
  hardware = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  config = SimConfig(
    context_count=2, max_cycles=100000, group=GroupSchedulerConfig(epoch_policy="same_program")
  )
  simulator = Simulator(hardware, config, enable_tracer=True)
  result = run_source(simulator, parse_workload_ir(BRANCH), BINDINGS)
  assert result.completed, result.reason
  assert result.tracer is not None
  events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
  jobs = [event for event in events if event.get("ph") == "X"]
  slow = next(event for event in jobs if event.get("name") == "EVU:slow")
  fast = next(event for event in jobs if event.get("name") == "BOA:matmul")
  prefetch_b = next(
    event
    for event in jobs
    if event.get("args", {}).get("summary_kind") == "group_transfer"
    and event["args"]["event_id"].endswith("_pb")
  )
  assert fast["ts"] >= slow["ts"] + slow["dur"]
  assert prefetch_b["ts"] < slow["ts"] + slow["dur"]


def test_cpu_dependency_wait_does_not_block_independent_hardware_work():
  bindings = {
    name: GlobalBinding(name, address, 2048, permission)
    for name, address, permission in (
      ("src", 0x100000, "r"),
      ("a", 0x110000, "rw"),
      ("b", 0x120000, "w"),
      ("c", 0x130000, "w"),
    )
  }
  hardware = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  simulator = Simulator(
    hardware, SimConfig(context_count=2, device_context_count=2, max_cycles=100000), enable_tracer=True
  )
  module = parse_workload_ir((SCENARIOS / "device_dependency_submit.mlir").read_text())
  result = run_source(simulator, module, bindings)
  assert result.completed, result.reason
  assert result.tracer is not None
  transfers = [
    event["args"]
    for event in json.loads(result.tracer.to_chrome_json())["traceEvents"]
    if event.get("ph") == "X" and event.get("args", {}).get("summary_kind") == "group_transfer"
  ]
  store_a = next(t for t in transfers if t["destination_address"] == 0x110000)
  store_b = next(t for t in transfers if t["destination_address"] == 0x120000)
  load_c = next(t for t in transfers if t["source_address"] == 0x110000)
  load_b = next(
    t
    for t in transfers
    if t["op"] == "prefetch" and t["context_launch_generation"] == store_b["context_launch_generation"]
  )
  assert load_c["start_cycle"] >= store_a["completion_cycle"]
  assert load_b["start_cycle"] < store_a["completion_cycle"]


def test_t11_device_dependency_rejects_foreign_producer_with_matching_tag():
  module = parse_workload_ir((SCENARIOS / "device_dependency_submit.mlir").read_text())
  program = next(op for op in module.body.block.ops if isinstance(op, NexusProgramOp))
  submissions = [op for op in program.body.block.ops if isinstance(op, NexusSubmitContextOp)]
  first, dependent = submissions[:2]
  foreign = NexusSubmitContextOp("worker", "a_done", actuals=first.actuals)
  dependent.operands = [*dependent.actuals, foreign.result]
  with pytest.raises(VerifyException):
    verify_workload_ir(module)


def test_cross_context_hbm_hazard_requires_an_explicit_completion_dependency():
  text = (SCENARIOS / "device_dependency_submit.mlir").read_text()
  with pytest.raises(VerifyException, match="overlapping global accesses"):
    parse_workload_ir(text.replace(" depends_on(%a_done)", ""))


def test_group_event_budget_recycles_without_l2_release_notifications():
  module = parse_workload_ir("""
    builtin.module {
      nest.context @empty placement = 1
          resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
              logical_tasks = 0, l2_spm_bytes = 0, requested_contexts_per_tile = 1> { nest.return }
      nexus.program @run {
        %a = nexus.submit_context.async @empty : !nexus.event<"a">
        %b = nexus.submit_context.async @empty : !nexus.event<"b">
        nexus.await %a, %b
        nexus.return
      }
    }
  """)
  simulator = Simulator(
    HardwareConfig(),
    SimConfig(device_context_count=2, max_cycles=100, group=GroupSchedulerConfig(event_capacity=1)),
  )
  result = run_source(simulator, module)
  assert result.completed, result.reason
  assert result.group_snapshot["scheduler"]["event_peak"] == 1
  assert result.group_snapshot["scheduler"]["event_reserved"] == 0
  assert result.group_snapshot["memory"]["l2"]["peak_allocated_bytes"] == 0


def test_cpu_completion_budget_recycles_chain_and_rejects_blocked_frontier():
  chain = """
    builtin.module {
      nest.context @empty placement = 1
          resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
              logical_tasks = 0, l2_spm_bytes = 0, requested_contexts_per_tile = 1> { nest.return }
      nexus.program @run {
        %a = nexus.submit_context.async @empty : !nexus.event<"a">
        %b = nexus.submit_context.async @empty depends_on(%a) : !nexus.event<"b">
        nexus.await %b
        nexus.return
      }
    }
  """
  config = SimConfig(
    device_context_count=2, max_cycles=100, device=DeviceConfig(completion_capacity=1, pending_capacity=1)
  )
  result = run_source(Simulator(HardwareConfig(), config), parse_workload_ir(chain))
  assert result.completed, result.reason
  assert result.device_snapshot["counters"]["completion_peak"] <= 1
  blocked = chain.replace(" depends_on(%a)", "").replace("nexus.await %b", "nexus.await %b, %a")
  result = run_source(Simulator(HardwareConfig(), config), parse_workload_ir(blocked))
  assert not result.completed
  assert result.cycles < config.max_cycles
  assert result.device_snapshot["faulted"]
  assert result.device_snapshot["counters"]["reserved_completions"] == 0
