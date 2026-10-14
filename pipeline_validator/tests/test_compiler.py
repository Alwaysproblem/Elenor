from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from itertools import pairwise
from pathlib import Path

import pytest
from xdsl.utils.exceptions import VerifyException

from pipeline_validator.compiled_program import (
  parse_compiled_program,
  seal_program,
  serialize_compiled_program,
)
from pipeline_validator.compiler import compile_program
from pipeline_validator.compiler.resources import conservative_arena_bytes, layout_buffers
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import ExecGroupActionOp, ExecModel, ExecTileGroupTask, GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.profiles import build_registry
from pipeline_validator.simulator import Simulator
from pipeline_validator.workload_ir import parse_workload_ir, print_workload_ir
from pipeline_validator.workloads import PowWorkload

ROOT = Path(__file__).resolve().parents[2]
POW_BINDINGS = {"Y": GlobalBinding("Y", 0x100000, 524288, "rw")}


COMPATIBLE_IR = """builtin.module {
  nest.context @mode2 placement = 1
      resource_contract = #nest.context_resources<l2_mode = 2, allowed_profiles = [2],
          logical_tasks = 0, l2_spm_bytes = 0, requested_contexts_per_tile = 1> {
    nest.return
  }
  nest.context @baseline0 placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 2],
          logical_tasks = 0, l2_spm_bytes = 0, requested_contexts_per_tile = 1> {
    nest.return
  }
  nexus.program @compatible {
    %first = nexus.submit_context.async @mode2 : !nexus.event<"first">
    nexus.await %first
    %second = nexus.submit_context.async @baseline0 : !nexus.event<"second">
    nexus.await %second
    nexus.return
  }
}
"""


ALIAS_IR = """builtin.module {
  nest.context @copy(
      %X : !nest.global_memref<1024xi8>,
      %Y : !nest.global_memref<1024xi8>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
          logical_tasks = 0, l2_spm_bytes = 2048, requested_contexts_per_tile = 1> {
    %x = nest.alloc slot = "x" role = "in" shape = [1024] dtype = "i8"
        : !nest.l2_buffer<1024xi8>
    %y = nest.alloc slot = "y" role = "in" shape = [1024] dtype = "i8"
        : !nest.l2_buffer<1024xi8>
    %xv = nest.subview %X offsets = [0] sizes = [1024] strides = [1]
        : !nest.global_view<1024xi8>
    %yv = nest.subview %Y offsets = [0] sizes = [1024] strides = [1]
        : !nest.global_view<1024xi8>
    %px = nest.dma.prefetch.async %xv into %x : !nest.event<"px">
    %py = nest.dma.prefetch.async %yv into %y : !nest.event<"py">
    %sx = nest.dma.store.async %x into %xv : !nest.event<"sx">
    %sy = nest.dma.store.async %y into %xv : !nest.event<"sy">
    nest.release %x depends_on(%px, %sx)
    nest.release %y depends_on(%py, %sy)
    nest.await %px, %py, %sx, %sy
    nest.return
  }
  nexus.program @aliases(
      %A : !nest.global_memref<1024xi8>,
      %B : !nest.global_memref<1024xi8>) {
    %same = nexus.submit_context.async @copy(%A, %A) : !nexus.event<"same">
    nexus.await %same
    %different = nexus.submit_context.async @copy(%A, %B) : !nexus.event<"different">
    nexus.await %different
    nexus.return
  }
}
"""


def _compile_pow(hw: HardwareConfig | None = None, sim: SimConfig | None = None):
  hw = hw or HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  sim = sim or SimConfig(max_cycles=200000)
  workload = PowWorkload(hw=hw)
  return hw, sim, workload, compile_program(workload.module, hw, sim, workload_info=workload.info)


def _replace_task(program, task: ExecTileGroupTask):
  entry: ExecTileGroupTask | ExecModel
  if isinstance(program.entry, ExecTileGroupTask):
    entry = task
  else:
    assert isinstance(program.entry, ExecModel)
    tasks = dict(program.entry.tasks)
    key = next(key for key, value in tasks.items() if value.binding_id == task.binding_id)
    tasks[key] = task
    entry = replace(program.entry, tasks=tasks)
  return seal_program(replace(program, entry=entry))


def _all_tasks(program):
  if isinstance(program.entry, ExecTileGroupTask):
    return (program.entry,)
  return tuple(program.entry.tasks.values())


def _compile_shared_program(*, allow_mode2: bool = False):
  from pipeline_validator.tests.test_l2_sharing_source import SHARED_IR

  source = SHARED_IR.replace(
    "nexus.program @run(\n      %W : !nest.global_memref<4xi8>) {",
    "nexus.program @run(\n"
    "      %W : !nest.global_memref<4xi8>,\n"
    "      %OUT : !nest.global_memref<4xi8>) {",
    1,
  ).replace("@reader(%W, %shared_w)", "@reader(%OUT, %shared_w)", 1)
  source = source.replace(
    '        : !nexus.event<"reader_done">\n    nexus.return',
    '        : !nexus.event<"reader_done">\n'
    "    nexus.await %loaded\n"
    "    nexus.await %reader_done\n"
    "    nexus.return",
    1,
  )
  source = source.replace("l2_spm_bytes = 4", "l2_spm_bytes = 1024")
  source = source.replace(
    "tile_l1_spm_bytes_per_context = 4", "tile_l1_spm_bytes_per_context = 1024"
  )
  if allow_mode2:
    source = source.replace("allowed_profiles = [0]", "allowed_profiles = [0, 2]")
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  sim = SimConfig(max_cycles=200000)
  artifact = compile_program(parse_workload_ir(source), hw, sim)
  bindings = {
    "W": GlobalBinding("W", 0x100000, 4, "r"),
    "OUT": GlobalBinding("OUT", 0x200000, 4, "w"),
  }
  return hw, sim, artifact, bindings


class TestExplicitCompileLoadReplay:
  def test_cb08_source_run_is_rejected_but_serialized_artifact_replays(self):
    hw, sim, workload, artifact = _compile_pow()
    simulator = Simulator(hw, sim)
    with pytest.raises(ValueError, match="LoadedProgram"):
      simulator.run(workload.module)

    text = serialize_compiled_program(artifact)
    assert artifact.schema_version == 3
    assert artifact.compiler_abi == "v3"
    # Plan §1: v3 clean cutover - every earlier schema/ABI pair is rejected
    # with a recompile hint; no upgrade shim exists.
    for schema_version, compiler_abi in (
      (1, "v0"),
      (2, "v0"),
      (1, "v1"),
      (2, "v1"),
      (2, "v2"),
      (3, "v2"),
    ):
      legacy = json.loads(text)
      legacy["schema_version"] = schema_version
      legacy["compiler_abi"] = compiler_abi
      with pytest.raises(ValueError, match="recompile.*source"):
        parse_compiled_program(json.dumps(legacy))
    replay = parse_compiled_program(text)
    loaded = load_program(replay, hw, sim, actual_bindings=POW_BINDINGS)
    result = simulator.run(loaded)
    assert result.completed, result.reason
    assert serialize_compiled_program(replay) == text

  def test_compiled_templates_and_maps_are_deeply_readonly(self):
    _hw, _sim, _workload, artifact = _compile_pow()
    task = _all_tasks(artifact)[0]
    with pytest.raises(FrozenInstanceError):
      task.actions = ()
    with pytest.raises(TypeError):
      task.role_bindings[0] = task.role_bindings[0]
    descriptor = next(iter(task.role_bindings[0].tile_program.descriptors.values()))
    with pytest.raises(TypeError):
      descriptor.params["new"] = 1

  def test_cb03_missing_or_inconsistent_program_identity_is_rejected(self):
    hw, sim, _workload, artifact = _compile_pow()
    task = _all_tasks(artifact)[0]
    role = task.role_bindings[0]
    for bad_program in (
      replace(role.tile_program, program_id=0),
      replace(role.tile_program, program_hash=0),
      replace(role.tile_program, program_hash=role.tile_program.program_hash ^ 1),
    ):
      roles = dict(task.role_bindings)
      roles[0] = replace(role, tile_program=bad_program)
      with pytest.raises(ValueError):
        corrupted = _replace_task(artifact, replace(task, role_bindings=roles))
        load_program(corrupted, hw, sim, actual_bindings=POW_BINDINGS)

  def test_cb07_actual_binding_guards_reject_range_permission_alias_violations(self):
    hw, sim, _workload, artifact = _compile_pow()
    with pytest.raises(ValueError):
      load_program(artifact, hw, sim, actual_bindings={"Y": GlobalBinding("Y", 0x100000, 524287, "rw")})
    with pytest.raises(ValueError):
      load_program(artifact, hw, sim, actual_bindings={"Y": GlobalBinding("Y", 0x100000, 524288, "r")})


class TestReadOnlyVerifierCorruption:
  def test_cb01_recomputed_hash_missing_dependency_is_rejected_without_source_mutation(self):
    hw, sim, workload, artifact = _compile_pow()
    before = print_workload_ir(workload.module)
    task = _all_tasks(artifact)[0]
    index = next(
      index
      for index, action in enumerate(task.actions)
      if action.op is ExecGroupActionOp.DMA_STORE and action.dependencies
    )
    actions = list(task.actions)
    actions[index] = replace(actions[index], dependencies=())
    corrupted = _replace_task(artifact, replace(task, actions=tuple(actions)))
    with pytest.raises(ValueError):
      load_program(corrupted, hw, sim, actual_bindings=POW_BINDINGS)
    assert print_workload_ir(workload.module) == before
    assert artifact.artifact_hash != corrupted.artifact_hash

  def test_t02_rv09_missing_await_or_incomplete_profile_command_is_rejected(self):
    hw = HardwareConfig()
    sim = SimConfig(max_cycles=200000)
    module = parse_workload_ir((ROOT / "examples/scenarios/profile_reconfiguration.mlir").read_text())
    artifact = compile_program(module, hw, sim)
    assert isinstance(artifact.entry, ExecModel)
    controls = [op for op in artifact.entry.body if op.op == "profile_reconfig"]
    assert controls

    control = controls[0]
    body_without_await = tuple(
      op
      for op in artifact.entry.body
      if not (op.op == "await" and op.instruction_id in control.command.wait_instruction_ids)
    )
    missing_await = seal_program(replace(artifact, entry=replace(artifact.entry, body=body_without_await)))
    with pytest.raises(ValueError):
      load_program(missing_await, hw, sim)

    with pytest.raises(ValueError):
      shortened = replace(control.command, steps=control.command.steps[:-1])
      body = tuple(replace(op, command=shortened) if op is control else op for op in artifact.entry.body)
      incomplete = seal_program(replace(artifact, entry=replace(artifact.entry, body=body)))
      load_program(incomplete, hw, sim)

  def test_cb05_runtime_cannot_infer_a_deleted_profile_command(self):
    hw = HardwareConfig()
    sim = SimConfig(max_cycles=200000)
    module = parse_workload_ir((ROOT / "examples/scenarios/profile_reconfiguration.mlir").read_text())
    artifact = compile_program(module, hw, sim)
    assert isinstance(artifact.entry, ExecModel)
    body = tuple(op for op in artifact.entry.body if op.op != "profile_reconfig")
    assert len(body) < len(artifact.entry.body)
    corrupted = seal_program(replace(artifact, entry=replace(artifact.entry, body=body)))
    with pytest.raises(ValueError):
      load_program(corrupted, hw, sim)

  def test_strict_codec_rejects_unknown_fields_duplicate_keys_and_nonfinite_numbers(self):
    _hw, _sim, _workload, artifact = _compile_pow()
    encoded = json.loads(serialize_compiled_program(artifact))
    encoded["unknown"] = 1
    with pytest.raises(ValueError):
      parse_compiled_program(json.dumps(encoded))
    with pytest.raises(ValueError):
      parse_compiled_program('{"a":1,"a":2}')
    with pytest.raises(ValueError):
      parse_compiled_program("NaN")

  def test_rv05_l2_reconfiguration_inside_live_root_is_rejected(self):
    hw = HardwareConfig()
    sim = SimConfig(max_cycles=200000)
    module = parse_workload_ir((ROOT / "examples/scenarios/profile_reconfiguration.mlir").read_text())
    artifact = compile_program(module, hw, sim)
    task = next(
      task
      for task in _all_tasks(artifact)
      if any(action.op is ExecGroupActionOp.PROFILE_RECONFIG for action in task.actions)
    )
    actions = list(task.actions)
    index = next(
      index for index, action in enumerate(actions) if action.op is ExecGroupActionOp.PROFILE_RECONFIG
    )
    action = actions[index]
    command = action.args[0]
    target = artifact.registry.profile("l2", command.target_mode)
    actions[index] = replace(
      action, args=(replace(command, level="l2", member_ids=target.member_ids, affected_domains=("l2",)),)
    )
    corrupted = _replace_task(artifact, replace(task, actions=tuple(actions)))
    with pytest.raises(ValueError):
      load_program(corrupted, hw, sim)

  def test_rv06_rv07_internal_l1_switch_has_grid_retirement_and_device_exclusivity_proof(self):
    hw = HardwareConfig()
    sim = SimConfig(max_cycles=200000)
    module = parse_workload_ir((ROOT / "examples/scenarios/profile_reconfiguration.mlir").read_text())
    artifact = compile_program(module, hw, sim)
    assert isinstance(artifact.entry, ExecModel)
    exclusive = [binding for binding in artifact.call_bindings.values() if binding.requires_l1_exclusive]
    assert exclusive
    task = next(
      task for task in artifact.entry.tasks.values() if task.binding_id == exclusive[0].binding_id
    )
    control_index = next(
      index for index, action in enumerate(task.actions) if action.op is ExecGroupActionOp.PROFILE_RECONFIG
    )
    assert task.actions[control_index - 1].op is ExecGroupActionOp.WAIT_EVENT
    assert task.actions[control_index - 1].args

    body = tuple(op for op in artifact.entry.body if op.op != "await")
    corrupted = seal_program(replace(artifact, entry=replace(artifact.entry, body=body)))
    with pytest.raises(ValueError):
      load_program(corrupted, hw, sim)

  def test_cb04_wait_and_waitall_retain_only_their_source_events(self):
    _hw, _sim, _workload, artifact = _compile_pow()
    task = _all_tasks(artifact)[0]
    program = task.role_bindings[0].tile_program
    waits = [inst for inst in program.insts if inst.op.value in ("wait", "waitall")]
    assert waits
    assert all(inst.args in (("e_load",), ("e_pow",), ("e_store",)) for inst in waits)
    group_waits = [action for action in task.actions if action.op is ExecGroupActionOp.WAIT_EVENT]
    assert group_waits
    assert all(tuple(action.args) == tuple(dict.fromkeys(action.args)) for action in group_waits)


class TestProfileAndAliasCompilation:
  def test_t01_rv10_compatible_current_mode_does_not_switch_back_to_baseline(self):
    hw = HardwareConfig()
    sim = SimConfig()
    artifact = compile_program(parse_workload_ir(COMPATIBLE_IR), hw, sim)
    assert isinstance(artifact.entry, ExecModel)
    baseline = next(
      binding for binding in artifact.call_bindings.values() if binding.context_name == "baseline0"
    )
    assert baseline.requested_l2_mode == 0
    assert baseline.resolved_l2_mode == 2
    l2_controls = [
      op
      for op in (*artifact.entry_prefix, *artifact.entry.body)
      if op.op == "profile_reconfig" and op.command.level == "l2"
    ]
    assert len(l2_controls) == 1

  def test_t28_explicit_same_profile_await_is_preserved_without_commit(self):
    text = COMPATIBLE_IR.replace(
      "l2_mode = 2, allowed_profiles = [2]", "l2_mode = 0, allowed_profiles = [0]"
    )
    hw = HardwareConfig()
    sim = SimConfig(max_cycles=200000)
    artifact = compile_program(parse_workload_ir(text), hw, sim)
    assert isinstance(artifact.entry, ExecModel)
    assert any(op.op == "await" and op.event_tag == "first" for op in artifact.entry.body)
    assert not [op for op in (*artifact.entry_prefix, *artifact.entry.body) if op.op == "profile_reconfig"]
    result = Simulator(hw, sim).run(load_program(artifact, hw, sim))
    assert result.completed, result.reason
    assert result.group_snapshot["profile"]["generations"] == {"l1": 0, "l2": 0}

  def test_cb02_each_alias_call_is_specialized_from_a_clean_template(self):
    module = parse_workload_ir(ALIAS_IR)
    before = print_workload_ir(module)
    artifact = compile_program(module, HardwareConfig(), SimConfig())
    assert isinstance(artifact.entry, ExecModel)
    tasks = sorted(artifact.entry.tasks.values(), key=lambda task: task.binding_id)
    assert len(tasks) == 2
    stores = [
      tuple(action.dependencies for action in task.actions if action.op is ExecGroupActionOp.DMA_STORE)
      for task in tasks
    ]
    assert stores[0] != stores[1]
    again = compile_program(module, HardwareConfig(), SimConfig())
    assert serialize_compiled_program(again) == serialize_compiled_program(artifact)
    assert print_workload_ir(module) == before

  def test_rv02_dispatch_baseline_must_belong_to_tile_allowed_profiles(self):
    text = (
      (ROOT / "examples/workloads/gather_indexed.mlir")
      .read_text()
      .replace(
        "nest.dispatch.tasks.async @gather_tile l1_mode = 1",
        "nest.dispatch.tasks.async @gather_tile l1_mode = 0",
        1,
      )
    )
    with pytest.raises((ValueError, VerifyException)):
      compile_program(parse_workload_ir(text), HardwareConfig(), SimConfig())

  def test_t24_cache_target_hint_is_not_a_private_capacity_quota(self):
    text = (
      (ROOT / "examples/workloads/gather_indexed.mlir")
      .read_text()
      .replace("target_bytes = 65536", "target_bytes = 99999999")
    )
    artifact = compile_program(parse_workload_ir(text), HardwareConfig(), SimConfig())
    assert artifact.resource_budgets

  def test_t26_zero_cache_profile_is_a_legal_bypass_path(self):
    """Plan §2: a disabled cache level is a bypass path, not a rejection."""
    text = (
      (ROOT / "examples/workloads/gather_indexed.mlir")
      .read_text()
      .replace("#tile.resources<allowed_profiles = [1, 2]", "#tile.resources<allowed_profiles = [0, 1]", 1)
    )
    compile_program(parse_workload_ir(text), HardwareConfig(), SimConfig())

  def test_t26_forbidden_bypass_on_zero_cache_profile_is_rejected(self):
    text = (
      (ROOT / "examples/workloads/gather_indexed.mlir")
      .read_text()
      .replace("#tile.resources<allowed_profiles = [1, 2]", "#tile.resources<allowed_profiles = [0, 1]", 1)
      .replace('l1_cache = {required = false, access = "read", bypass = "allowed"',
               'l1_cache = {required = false, access = "read", bypass = "forbidden"', 1)
    )
    with pytest.raises(ValueError, match="bypass"):
      compile_program(parse_workload_ir(text), HardwareConfig(), SimConfig())

  def test_t27_underdeclared_layout_padding_is_rejected(self):
    text = (
      (ROOT / "examples/workloads/gather_indexed.mlir")
      .read_text()
      .replace("tile_l1_spm_bytes_per_context = 2048", "tile_l1_spm_bytes_per_context = 1024", 1)
    )
    with pytest.raises(ValueError):
      compile_program(parse_workload_ir(text), HardwareConfig(), SimConfig())

  def test_rv03_t32_cross_layer_cache_combination_is_checked_not_cartesian_assumed(self):
    """Plan §2: both levels may be bypassed, but the dispatch must stay in-contract."""
    source = (ROOT / "examples/workloads/gather_indexed.mlir").read_text()
    # Plan §2: a zero-cache profile on both levels is a legal bypass path.
    bypassed = (
      source.replace(
        "l2_mode = 1, allowed_profiles = [1, 2]", "l2_mode = 0, allowed_profiles = [0, 1]", 1
      )
      .replace("#tile.resources<allowed_profiles = [1, 2]",
               "#tile.resources<allowed_profiles = [0, 1]", 1)
      .replace("nest.dispatch.tasks.async @gather_tile l1_mode = 1",
               "nest.dispatch.tasks.async @gather_tile l1_mode = 0", 1)
    )
    assert bypassed != source
    compile_program(parse_workload_ir(bypassed), HardwareConfig(), SimConfig())
    # A dispatch that asks for a mode outside the tile contract is rejected.
    outside = source.replace(
      "nest.dispatch.tasks.async @gather_tile l1_mode = 1",
      "nest.dispatch.tasks.async @gather_tile l1_mode = 7",
      1,
    )
    assert outside != source
    with pytest.raises(VerifyException):
      parse_workload_ir(outside)

  def test_c02_event_frontier_bounds_live_events_not_historical_waves(self):
    from pipeline_validator.config import GroupSchedulerConfig

    waves = "\n".join(
      f"""%g{i}, %r{i}, %w{i} = nest.dispatch.tasks.async @empty l1_mode = 0
          tasks(%tasks) globals() bindings() ins() outs() signal_policy {{}}
          : (!nest.event<"g{i}">, !nest.event<"">, !nest.event<"">)
        nest.await %g{i}"""
      for i in range(32)
    )
    text = (
      """builtin.module {
      tile.program @empty(%t : !nest.task)
        resource_contract = #tile.resources<allowed_profiles = [0], tile_l1_spm_bytes_per_context = 0> {
        tile.return
      }
      nest.context @waves placement = 1
        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 32, l2_spm_bytes = 0, requested_contexts_per_tile = 1> {
        %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
      """
      + waves
      + "\nnest.return\n}}"
    )
    hw = HardwareConfig()
    config = SimConfig(group=GroupSchedulerConfig(event_capacity=1), max_cycles=10000)
    program = compile_program(parse_workload_ir(text), hw, config)
    assert len(program.entry.event_uses) == 33
    result = Simulator(hw, config).run(load_program(program, hw, config))
    assert result.completed, result.reason
    assert result.pmu.events["tile_done"] == 32
    assert result.group_snapshot["event_table"]["peak_active"] == 1
    assert result.group_snapshot["event_table"]["reserved"] == 0

  def test_c02_large_mode0_only_task_is_not_charged_to_small_mode2_envelope(self):
    text = """builtin.module {
      tile.program @large(%t : !nest.task)
        resource_contract = #tile.resources<allowed_profiles = [0],
          tile_l1_spm_bytes_per_context = 901120> {
        %a = tile.alloc shape = [901120] dtype = "i8" alignment = 64 : !tile.l1_buffer<901120xi8>
        tile.return
      }
      tile.program @small(%t : !nest.task)
        resource_contract = #tile.resources<allowed_profiles = [2], tile_l1_spm_bytes_per_context = 1024> {
        %a = tile.alloc shape = [64] dtype = "i8" alignment = 64 : !tile.l1_buffer<64xi8>
        tile.return
      }
      nest.context @switch placement = 1
        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 2, l2_spm_bytes = 0, requested_contexts_per_tile = 1> {
        %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
        %a, %ar, %aw = nest.dispatch.tasks.async @large l1_mode = 0
          tasks(%tasks) globals() bindings() ins() outs() signal_policy {}
          : (!nest.event<"a">, !nest.event<"">, !nest.event<"">)
        nest.await %a
        %b, %br, %bw = nest.dispatch.tasks.async @small l1_mode = 2
          tasks(%tasks) globals() bindings() ins() outs() signal_policy {}
          : (!nest.event<"b">, !nest.event<"">, !nest.event<"">)
        nest.await %b
        nest.return
      }
    }"""
    hw, config = HardwareConfig(), SimConfig(max_cycles=10000)
    program = compile_program(parse_workload_ir(text), hw, config)
    result = Simulator(hw, config).run(load_program(program, hw, config))
    assert result.completed, result.reason
    assert result.group_snapshot["profile"]["generations"] == {"l1": 1, "l2": 0}
    impossible = text.replace("requested_contexts_per_tile = 1", "requested_contexts_per_tile = 2")
    with pytest.raises(ValueError):
      compile_program(parse_workload_ir(impossible), hw, replace(config, context_count=2))


class TestLoaderDefenses:
  def test_t11_loader_rejects_forward_submit_dependency(self):
    hw = HardwareConfig()
    sim = SimConfig(max_cycles=200000)
    module = parse_workload_ir((ROOT / "examples/scenarios/profile_reconfiguration.mlir").read_text())
    artifact = compile_program(module, hw, sim)
    assert isinstance(artifact.entry, ExecModel)
    submits = [op for op in artifact.entry.body if op.op == "submit"]
    assert len(submits) >= 2
    first, second = submits[0], submits[1]
    body = tuple(
      replace(op, dependencies=(second.event_tag,)) if op is first else op for op in artifact.entry.body
    )
    corrupted = seal_program(replace(artifact, entry=replace(artifact.entry, body=body)))
    with pytest.raises(ValueError):
      load_program(corrupted, hw, sim)

  def test_t11_loader_rejects_foreign_owner_group_event(self):
    hw = HardwareConfig()
    sim = SimConfig(max_cycles=200000)
    module = parse_workload_ir((ROOT / "examples/scenarios/profile_reconfiguration.mlir").read_text())
    artifact = compile_program(module, hw, sim)
    tasks = {task.binding_id: task for task in _all_tasks(artifact)}
    foreign_events = {
      event for task in tasks.values() for action in task.actions for event in action.output_events
    }
    victim = next(iter(tasks.values()))
    foreign = next(
      event for event in foreign_events if event not in {e for a in victim.actions for e in a.output_events}
    )
    index = next(
      index for index, action in enumerate(victim.actions) if action.op is ExecGroupActionOp.WAIT_EVENT
    )
    actions = list(victim.actions)
    actions[index] = replace(actions[index], dependencies=(*actions[index].dependencies, foreign))
    corrupted = _replace_task(artifact, replace(victim, actions=tuple(actions)))
    with pytest.raises(ValueError):
      load_program(corrupted, hw, sim)

  def test_cb07_pairwise_alias_overlap_between_distinct_bindings_rejected(self):
    hw = HardwareConfig()
    sim = SimConfig(max_cycles=200000)
    artifact = compile_program(parse_workload_ir(ALIAS_IR), hw, sim)
    overlapping = {
      "A": GlobalBinding("A", 0x100000, 1024, "rw"),
      "B": GlobalBinding("B", 0x100200, 1024, "rw"),
    }
    with pytest.raises(ValueError):
      load_program(artifact, hw, sim, actual_bindings=overlapping)
    disjoint = {
      "A": GlobalBinding("A", 0x100000, 1024, "rw"),
      "B": GlobalBinding("B", 0x200000, 1024, "rw"),
    }
    loaded = load_program(artifact, hw, sim, actual_bindings=disjoint)
    assert loaded is not None

  def test_c07_relocation_table_corruption_is_rejected(self):
    hw, sim, _workload, artifact = _compile_pow()
    assert artifact.relocations
    missing = seal_program(replace(artifact, relocations=artifact.relocations[:-1]))
    with pytest.raises(ValueError):
      load_program(missing, hw, sim, actual_bindings=POW_BINDINGS)
    victim = artifact.relocations[0]
    swapped = replace(victim, ordinal=victim.ordinal + 1000)
    wrong = tuple(swapped if item is victim else item for item in artifact.relocations)
    corrupted = seal_program(replace(artifact, relocations=wrong))
    with pytest.raises(ValueError):
      load_program(corrupted, hw, sim, actual_bindings=POW_BINDINGS)

  def test_c09_source_map_corruption_is_rejected(self):
    hw, sim, _workload, artifact = _compile_pow()
    assert artifact.source_map
    key = next(iter(artifact.source_map))
    source = artifact.source_map[key]

    forged = replace(source, symbol=f"{source.symbol}:forged") if hasattr(source, "symbol") else source
    corrupted_map = dict(artifact.source_map)
    corrupted_map[key] = forged
    corrupted = seal_program(replace(artifact, source_map=corrupted_map))
    with pytest.raises(ValueError):
      load_program(corrupted, hw, sim, actual_bindings=POW_BINDINGS)

  def test_loader_rejects_use_after_release_l2_view(self):
    from pipeline_validator.compiler.api import _dependency_proofs, _relocations, _source_map
    from pipeline_validator.compiler.resources import finalize_event_resources
    from pipeline_validator.execution_ir import ExecGroupAction
    from pipeline_validator.tests.test_profile_runtime import COPY_IR

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = SimConfig(max_cycles=200000)
    artifact = compile_program(parse_workload_ir(COPY_IR), hw, sim)
    task = _all_tasks(artifact)[0]
    release_index = next(
      index for index, action in enumerate(task.actions) if action.op is ExecGroupActionOp.RELEASE_L2
    )
    store = next(action for action in task.actions if action.op is ExecGroupActionOp.DMA_STORE)
    release_done = task.actions[release_index].dst
    forged_store = ExecGroupAction(
      ExecGroupActionOp.DMA_STORE,
      args=store.args,
      dst="forged_store_done",
      dependencies=(release_done,),
      reads=store.reads,
      writes=store.writes,
      source_ref=store.source_ref,
      instruction_id=f"{task.binding_id}:group:{len(task.actions)}:forged_store",
    )
    forged_task = replace(task, actions=(*task.actions, forged_store))
    entry, budgets = finalize_event_resources(forged_task, dict(artifact.resource_budgets), sim)
    corrupted = seal_program(
      replace(
        artifact,
        entry=entry,
        resource_budgets=budgets,
        relocations=_relocations(entry),
        source_map=_source_map(entry, artifact.entry_prefix),
        dependency_proofs=_dependency_proofs(entry),
      )
    )
    with pytest.raises(ValueError, match="released L2 view"):
      load_program(corrupted, hw, sim)

  def test_c06_t16_write_back_prewrite_clean_is_emitted_and_required(self):
    base = HardwareConfig()
    wb_l2 = replace(base.memory_target.l2, cache_write_policy="write_back")
    hw = replace(base, memory_target=replace(base.memory_target, l2=wb_l2))
    sim = SimConfig(max_cycles=200000)
    module = parse_workload_ir((ROOT / "examples/workloads/matmul_gather_add.mlir").read_text())
    artifact = compile_program(module, hw, sim)
    assert isinstance(artifact.entry, ExecModel)
    cleans = [
      op
      for op in artifact.entry.body
      if op.op == "memory_maintenance" and op.instruction_id.endswith(":prewrite_clean")
    ]
    assert cleans
    assert all("l2" in op.command.levels for op in cleans)
    from pipeline_validator.compiler.api import _source_map

    body = tuple(op for op in artifact.entry.body if op not in cleans)
    new_entry = replace(artifact.entry, body=body)
    corrupted = seal_program(
      replace(artifact, entry=new_entry, source_map=_source_map(new_entry, artifact.entry_prefix))
    )
    with pytest.raises(ValueError, match="dirty-cache clean"):
      load_program(corrupted, hw, sim)


class TestL2NoRebindLayout:
  """plan/01 §1: every local L2 buffer owns a permanent, disjoint span."""

  def test_compiled_workload_layouts_are_pairwise_disjoint_and_conserved(self):
    from pipeline_validator.execution_verifier import _verify_l2_no_rebind_layout

    hw = HardwareConfig().with_overrides(num_dma_channels=2, hbm_fixed_latency_cycles=10)
    sim = SimConfig(max_cycles=200000, context_count=4)
    module = parse_workload_ir((ROOT / "examples/workloads/matmul_pow_data_dep.mlir").read_text())
    artifact = compile_program(module, hw, sim)
    assert isinstance(artifact.entry, ExecModel)
    checked = 0
    for task in artifact.entry.tasks.values():
      layout = task.layout
      assert layout is not None
      if len(layout.buffer_layouts) < 2:
        continue
      round_bytes = layout.stripe_bytes * len(layout.per_bank_bytes)
      spans = []
      for item in layout.buffer_layouts:
        padded = -(-item.logical_bytes // round_bytes) * round_bytes
        spans.append((item.arena_offset, item.arena_offset + padded))
      spans.sort()
      for (_, left_end), (right_start, _) in pairwise(spans):
        assert right_start >= left_end, "compiled L2 padded spans overlap"
      high_water = max(end for _, end in spans)
      assert layout.reserved_bytes >= high_water
      _verify_l2_no_rebind_layout(layout, "test")
      checked += 1
    assert checked >= 1

  def test_no_rebind_layout_ignores_release_ordering(self):
    """Removing L2 alias reuse: a released buffer's span is never handed to a
    later bind, so layout spans stay disjoint for any action order."""
    from pipeline_validator.execution_ir import ExecL2Buffer

    hw = HardwareConfig()
    registry = build_registry(hw)
    profile = registry.profile("l2", 0)
    buffers = (
      ExecL2Buffer("a_input", (1024,), "i8", "in", 1, 64, 1024),
      ExecL2Buffer("b_late", (1024,), "i8", "inout", 1, 64, 1024),
    )
    reserved = conservative_arena_bytes([(b.bytes, b.alignment) for b in buffers], profile)
    # lifetimes would previously allow b_late to reuse a_input's released
    # region; the no-rebind layout ignores lifetimes entirely.
    layout = layout_buffers(buffers, profile, reserved, lifetimes=None, slot_capacity=2)
    spans = sorted(
      (item.arena_offset, item.arena_offset + item.logical_bytes) for item in layout.buffer_layouts
    )
    assert spans[0][1] <= spans[1][0]
    assert layout.buffer_layouts[1].arena_offset >= layout.buffer_layouts[0].arena_offset


class TestIndependentSharedArtifactVerification:
  def test_shared_artifact_round_trips_codec_and_loads(self):
    hw, sim, artifact, bindings = _compile_shared_program()
    text = serialize_compiled_program(artifact)
    replay = parse_compiled_program(text)

    loaded = load_program(replay, hw, sim, actual_bindings=bindings)

    assert loaded.compiled.schema_version == 3
    assert loaded.compiled.compiler_abi == "v3"
    assert serialize_compiled_program(loaded.compiled) == text


  def test_publish_dependency_events_have_a_required_relocation(self):
    hw, sim, artifact, bindings = _compile_shared_program()
    publish_relocations = [item for item in artifact.relocations if item.kind == "publish_events"]
    assert len(publish_relocations) == 1
    corrupted = seal_program(
      replace(
        artifact,
        relocations=tuple(item for item in artifact.relocations if item.kind != "publish_events"),
      )
    )

    assert corrupted.artifact_hash != artifact.artifact_hash
    with pytest.raises(ValueError, match="relocation table differs"):
      load_program(corrupted, hw, sim, actual_bindings=bindings)

  @pytest.mark.parametrize(
    ("changes", "message"),
    [
      ({"producer_binding_id": "missing"}, "unknown producer binding"),
      ({"producer_slot": "missing"}, "unpublished producer slot"),
      ({"dims": (2,), "bytes": 2}, "incompatible L2 actual"),
      ({"dtype": "f16", "element_bytes": 2, "bytes": 8}, "incompatible L2 actual"),
    ],
  )
  def test_shared_identity_shape_and_dtype_tampering_is_semantically_rejected(
    self, changes, message
  ):
    hw, sim, artifact, bindings = _compile_shared_program()
    reader = next(task for task in _all_tasks(artifact) if task.name == "reader")
    shared = replace(reader.shared_inputs[0], **changes)
    corrupted = _replace_task(artifact, replace(reader, shared_inputs=(shared,)))

    assert corrupted.artifact_hash != artifact.artifact_hash
    with pytest.raises(ValueError, match=message):
      load_program(corrupted, hw, sim, actual_bindings=bindings)

  def test_shared_backing_counts_against_reader_per_bank_capacity(self):
    hw, sim, artifact, bindings = _compile_shared_program()
    profile = artifact.registry.profile("l2", 0)
    reader = next(task for task in _all_tasks(artifact) if task.name == "reader")
    local_layout = layout_buffers((), profile, profile.user_spm_bytes, slot_capacity=1)
    contract = replace(reader.resource_contract, l2_spm_bytes=profile.user_spm_bytes)
    corrupted = _replace_task(
      artifact, replace(reader, layout=local_layout, resource_contract=contract)
    )

    assert corrupted.artifact_hash != artifact.artifact_hash
    with pytest.raises(ValueError, match="exceeds per-bank L2 capacity"):
      load_program(corrupted, hw, sim, actual_bindings=bindings)


  def test_publish_requires_complete_prefetch_initialization_after_rehash(self):
    from pipeline_validator.execution_ir import ExecGroupActionOp

    hw, sim, artifact, bindings = _compile_shared_program()
    producer = next(task for task in _all_tasks(artifact) if task.name == "loader")
    actions = list(producer.actions)
    prefetch_index = next(
      index for index, action in enumerate(actions) if action.op is ExecGroupActionOp.DMA_PREFETCH
    )
    prefetch = actions[prefetch_index]
    transfer = prefetch.args[1]
    partial_src = replace(transfer.src, dims=(2,), bytes=2)
    partial_dst = replace(transfer.dst, dims=(2,), bytes=2)
    actions[prefetch_index] = replace(
      prefetch,
      args=(prefetch.args[0], replace(transfer, src=partial_src, dst=partial_dst, bytes=2)),
    )
    corrupted = _replace_task(artifact, replace(producer, actions=tuple(actions)))

    assert corrupted.artifact_hash != artifact.artifact_hash
    with pytest.raises(ValueError, match="complete initialization"):
      load_program(corrupted, hw, sim, actual_bindings=bindings)

  def test_publish_dependencies_are_recomputed_from_real_accesses(self):
    from pipeline_validator.execution_ir import ExecGroupActionOp

    hw, sim, artifact, bindings = _compile_shared_program()
    producer = next(task for task in _all_tasks(artifact) if task.name == "loader")
    prefetch = next(
      action for action in producer.actions if action.op is ExecGroupActionOp.DMA_PREFETCH
    )
    actions = list(producer.actions)
    publish_index = next(
      index for index, action in enumerate(actions) if action.op is ExecGroupActionOp.PUBLISH_L2
    )
    publish = actions[publish_index]
    dependencies = tuple(event for event in publish.dependencies if event != prefetch.dst)
    request = replace(publish.args[0], dependency_events=dependencies)
    actions[publish_index] = replace(publish, dependencies=dependencies, args=(request,))
    corrupted = _replace_task(artifact, replace(producer, actions=tuple(actions)))

    assert corrupted.artifact_hash != artifact.artifact_hash
    with pytest.raises(ValueError, match="publish dependencies do not exactly close initialization"):
      load_program(corrupted, hw, sim, actual_bindings=bindings)

  def test_import_write_is_rejected_from_tile_descriptors_not_permissions(self):
    from pipeline_validator.compiled_program import program_digest
    from pipeline_validator.execution_ir import ExecTileOp

    hw, sim, artifact, bindings = _compile_shared_program()
    reader = next(task for task in _all_tasks(artifact) if task.name == "reader")
    role_id, role = next(iter(reader.role_bindings.items()))
    program = role.tile_program
    descriptor_name, descriptor = next(
      (name, item) for name, item in program.descriptors.items() if item.op == "load"
    )
    transfer = descriptor.transfer
    assert transfer is not None
    imported_destination = replace(transfer.src, base="formal:1")
    forged_transfer = replace(transfer, src=transfer.dst, dst=imported_destination)
    descriptors = dict(program.descriptors)
    descriptors[descriptor_name] = replace(descriptor, op="store", transfer=forged_transfer)
    instructions = tuple(
      replace(instruction, args=("output_ready", 0))
      if instruction.op is ExecTileOp.SIGNAL_PHASE and instruction.args[0] == "input_released"
      else instruction
      for instruction in program.insts
    )
    unhashed = replace(program, descriptors=descriptors, insts=instructions)
    forged_program = replace(unhashed, program_hash=program_digest(unhashed))
    roles = dict(reader.role_bindings)
    roles[role_id] = replace(
      role, tile_program=forged_program, read_actuals=(), write_actuals=("weight",)
    )
    corrupted = _replace_task(artifact, replace(reader, role_bindings=roles))

    assert corrupted.artifact_hash != artifact.artifact_hash
    with pytest.raises(ValueError, match="writes through a readonly shared import"):
      load_program(corrupted, hw, sim, actual_bindings=bindings)

  def test_consumer_must_retain_producer_completion_dependency_ancestry(self):
    hw, sim, artifact, bindings = _compile_shared_program()
    assert isinstance(artifact.entry, ExecModel)
    reader_submit = next(op for op in artifact.entry.body if op.op == "submit" and op.ctx_name == "reader")
    body = tuple(
      replace(op, dependencies=()) if op is reader_submit else op for op in artifact.entry.body
    )
    corrupted = seal_program(replace(artifact, entry=replace(artifact.entry, body=body)))

    assert corrupted.artifact_hash != artifact.artifact_hash
    with pytest.raises(ValueError, match="producer completion dependency ancestry"):
      load_program(corrupted, hw, sim, actual_bindings=bindings)

  def test_shared_reference_cannot_cross_an_l2_profile_epoch(self):
    from pipeline_validator.compiler.api import _source_map
    from pipeline_validator.execution_ir import ExecDeviceOp
    from pipeline_validator.profiles import ProfileState

    hw, sim, artifact, bindings = _compile_shared_program(allow_mode2=True)
    assert isinstance(artifact.entry, ExecModel)
    profile_artifact = compile_program(parse_workload_ir(COMPATIBLE_IR), hw, sim)
    switch_template = next(
      op.command
      for op in (*profile_artifact.entry_prefix, *profile_artifact.entry.body)
      if op.op == "profile_reconfig" and op.command.level == "l2"
    )
    producer_submit = next(
      op for op in artifact.entry.body if op.op == "submit" and op.ctx_name == "loader"
    )
    reader_submit = next(
      op for op in artifact.entry.body if op.op == "submit" and op.ctx_name == "reader"
    )
    wait_id = f"{producer_submit.instruction_id}:epoch_test_wait"
    command_id = f"{producer_submit.instruction_id}:epoch_test_switch"
    command = replace(
      switch_template,
      command_id=command_id,
      expected_mode=0,
      target_mode=2,
      registry_hash=artifact.registry_hash,
      wait_instruction_ids=(wait_id,),
      frontier=(producer_submit.event_tag,),
      member_ids=artifact.registry.profile("l2", 2).member_ids,
    )
    wait = ExecDeviceOp(
      "await",
      event_tag=producer_submit.event_tag,
      source_ref=command.source_ref,
      instruction_id=wait_id,
    )
    switch = ExecDeviceOp(
      "profile_reconfig",
      command=command,
      source_ref=command.source_ref,
      instruction_id=command_id,
    )
    body = []
    for op in artifact.entry.body:
      if op.op == "await" and op.event_tag == producer_submit.event_tag:
        continue
      body.append(op)
      if op is producer_submit:
        body.extend((wait, switch))
    entry = replace(artifact.entry, body=tuple(body))
    calls = dict(artifact.call_bindings)
    reader_call = calls[reader_submit.binding_id]
    calls[reader_submit.binding_id] = replace(
      reader_call,
      resolved_l2_mode=2,
      permitted_profiles=tuple(ProfileState(state.l1_mode, 2) for state in reader_call.permitted_profiles),
    )
    corrupted = seal_program(
      replace(
        artifact,
        entry=entry,
        call_bindings=calls,
        exit_profiles=ProfileState(artifact.exit_profiles.l1_mode, 2),
        source_map=_source_map(entry, artifact.entry_prefix),
      )
    )
    assert corrupted.artifact_hash != artifact.artifact_hash

    with pytest.raises(ValueError, match="crosses an L2 profile epoch"):
      load_program(corrupted, hw, sim, actual_bindings=bindings)


def test_context_local_release_must_cover_writer_even_without_reader_or_store() -> None:
  from pipeline_validator.tests.test_l2_sharing_source import CONTEXT_LOCAL_WRITER_ONLY_IR

  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10, num_dma_channels=2)
  sim = SimConfig(context_count=1, max_cycles=200000)
  artifact = compile_program(parse_workload_ir(CONTEXT_LOCAL_WRITER_ONLY_IR), hw, sim)
  bindings = {"SOURCE": GlobalBinding("SOURCE", 0x100000, 32768, "r")}
  load_program(artifact, hw, sim, actual_bindings=bindings)
  task = _all_tasks(artifact)[0]
  actions = list(task.actions)
  writer = next(
    action for action in actions if action.op is ExecGroupActionOp.DISPATCH_ROLE
  )
  release_index = next(
    index for index, action in enumerate(actions)
    if action.op is ExecGroupActionOp.RELEASE_L2 and action.args[0].buffer_slot == "scratch"
  )
  release = actions[release_index]
  dependencies = tuple(event for event in release.dependencies if event != writer.args[0].output_ready_event)
  assert len(dependencies) + 1 == len(release.dependencies)
  request = replace(release.args[0], dependency_events=dependencies)
  actions[release_index] = replace(release, dependencies=dependencies, args=(request,))
  corrupted = _replace_task(artifact, replace(task, actions=tuple(actions)))
  with pytest.raises(ValueError, match="does not retire every real buffer access"):
    load_program(corrupted, hw, sim, actual_bindings=bindings)


def test_context_local_final_store_must_cover_writers_after_resealed_reorder() -> None:
  from pipeline_validator.tests.test_l2_sharing_source import CONTEXT_LOCAL_IR

  source = CONTEXT_LOCAL_IR.replace(
    "    nest.release %scratch depends_on(%ready_produce, %read_mutate, %ready_mutate, %read_scratch)",
    "    %saved = nest.dma.store.async %scratch into %output"
    " depends_on(%ready_produce, %ready_mutate)\n"
    '        : !nest.event<"saved">\n'
    "    nest.release %scratch"
    " depends_on(%ready_produce, %read_mutate, %ready_mutate, %read_scratch, %saved)",
    1,
  )
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10, num_dma_channels=2)
  sim = SimConfig(context_count=1, max_cycles=200000)
  artifact = compile_program(parse_workload_ir(source), hw, sim)
  bindings = {
    "SOURCE": GlobalBinding("SOURCE", 0x100000, 32768, "r"),
    "OUT": GlobalBinding("OUT", 0x200000, 32768, "w"),
  }
  load_program(artifact, hw, sim, actual_bindings=bindings)
  task = _all_tasks(artifact)[0]
  actions = list(task.actions)
  store_index = next(
    index for index, action in enumerate(actions)
    if action.op is ExecGroupActionOp.DMA_STORE and action.args[1].src.base == "scratch"
  )
  store = actions.pop(store_index)
  store = replace(
    store, dependencies=tuple(event for event in store.dependencies if event != "ready_mutate")
  )
  mutate_index = next(
    index for index, action in enumerate(actions)
    if action.op is ExecGroupActionOp.DISPATCH_ROLE and action.args[0].dispatch_ordinal == 1
  )
  mutate = actions[mutate_index]
  actions[mutate_index] = replace(mutate, dependencies=(*mutate.dependencies, store.dst))
  actions.insert(mutate_index, store)
  corrupted = _replace_task(artifact, replace(task, actions=tuple(actions)))
  with pytest.raises(ValueError, match="final store does not cover every actual writer"):
    load_program(corrupted, hw, sim, actual_bindings=bindings)
