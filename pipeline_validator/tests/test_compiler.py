from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from xdsl.utils.exceptions import VerifyException

from pipeline_validator.compiled_program import (
  parse_compiled_program,
  seal_program,
  serialize_compiled_program,
)
from pipeline_validator.compiler import compile_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import ExecGroupActionOp, ExecModel, ExecTileGroupTask, GlobalBinding
from pipeline_validator.loader import load_program
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


class TestExplicitCompileLoadReplay:
  def test_cb08_source_run_is_rejected_but_serialized_artifact_replays(self):
    hw, sim, workload, artifact = _compile_pow()
    simulator = Simulator(hw, sim)
    with pytest.raises(ValueError, match="LoadedProgram"):
      simulator.run(workload.module)

    text = serialize_compiled_program(artifact)
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
      (ROOT / "examples/workloads/gather_profiled.mlir")
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
      (ROOT / "examples/workloads/gather_profiled.mlir")
      .read_text()
      .replace("target_bytes = 65536", "target_bytes = 99999999")
      .replace("cache_target_bytes = 65536", "cache_target_bytes = 99999999")
    )
    artifact = compile_program(parse_workload_ir(text), HardwareConfig(), SimConfig())
    assert artifact.resource_budgets

  def test_t26_complete_program_cache_path_rejects_zero_cache_allowed_mode(self):
    text = (
      (ROOT / "examples/workloads/gather_profiled.mlir")
      .read_text()
      .replace("#tile.resources<allowed_profiles = [1, 2]", "#tile.resources<allowed_profiles = [0, 1]", 1)
    )
    with pytest.raises(ValueError):
      compile_program(parse_workload_ir(text), HardwareConfig(), SimConfig())

  def test_t27_underdeclared_layout_padding_is_rejected(self):
    text = (
      (ROOT / "examples/workloads/gather_profiled.mlir")
      .read_text()
      .replace("tile_l1_spm_bytes_per_context = 2048", "tile_l1_spm_bytes_per_context = 1024", 1)
    )
    with pytest.raises(ValueError):
      compile_program(parse_workload_ir(text), HardwareConfig(), SimConfig())

  def test_rv03_t32_cross_layer_cache_combination_is_checked_not_cartesian_assumed(self):
    text = (
      (ROOT / "examples/workloads/gather_profiled.mlir")
      .read_text()
      .replace("l2_mode = 1, allowed_profiles = [1, 2]", "l2_mode = 0, allowed_profiles = [0]", 1)
    )
    with pytest.raises(ValueError):
      compile_program(parse_workload_ir(text), HardwareConfig(), SimConfig())

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
