"""Top-level explicit compiler pipeline and inspectable executable-control dump."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import pairwise

from xdsl.dialects.builtin import ModuleOp

from ..compiled_program import (
  BindingGuard,
  CompiledProgram,
  Relocation,
  WorkloadInfo,
  program_digest,
  seal_program,
)
from ..config import HardwareConfig, SimConfig
from ..dialects.elenor import NestContextOp
from ..execution_ir import (
  ExecDeviceOp,
  ExecGroupAction,
  ExecGroupActionOp,
  ExecMemoryView,
  ExecModel,
  ExecTileGroupTask,
  ExecTileInst,
  ExecTileProgram,
  GlobalBinding,
)
from ..immutable import canonical_value, digest
from ..profiles import SourceRef, build_registry
from ..workload_ir import print_workload_ir, verify_workload_ir
from .lowering import lower_model_ir, lower_workload_ir, normalize_action_dependencies
from .profile_pass import _mapped_accesses, bind_profiles
from .resources import finalize_event_resources, prepare_resources


def _entry_tasks(entry: ExecModel | ExecTileGroupTask) -> Mapping[str, ExecTileGroupTask]:
  return entry.tasks if isinstance(entry, ExecModel) else {entry.binding_id: entry}


def _finalize_action_dependencies(entry: ExecModel | ExecTileGroupTask) -> ExecModel | ExecTileGroupTask:
  """Recompute compiler retirement closures after generated control actions."""

  tasks = {
    binding_id: replace(task, actions=normalize_action_dependencies(task.actions, task.role_bindings))
    for binding_id, task in _entry_tasks(entry).items()
  }
  if isinstance(entry, ExecModel):
    return replace(entry, tasks=tasks)
  return tasks[entry.binding_id]


def _finalize_program_identities(
  entry: ExecModel | ExecTileGroupTask, hw: HardwareConfig
) -> ExecModel | ExecTileGroupTask:
  next_id = 1
  identities: dict[int, int] = {}
  finalized_objects: dict[int, ExecTileProgram] = {}
  tasks: dict[str, ExecTileGroupTask] = {}
  for binding_id, task in _entry_tasks(entry).items():
    roles = {}
    for role_id in sorted(task.role_bindings):
      role = task.role_bindings[role_id]
      program = finalized_objects.get(id(role.tile_program))
      if program is None:
        text_bytes = max(len(role.tile_program.insts) * 4 + len(role.tile_program.descriptors) * 16, 1024)
        if text_bytes > hw.tile_program_sram_bytes:
          raise ValueError(
            f"Tile Program {role.tile_program.name!r} text requires {text_bytes} bytes, "
            f"target program SRAM provides {hw.tile_program_sram_bytes}"
          )
        sized = replace(role.tile_program, text_bytes=text_bytes, program_id=0, program_hash=0)
        program_hash = program_digest(sized)
        if not 0 < program_hash < (1 << 256):
          raise ValueError(f"Tile Program {sized.name!r} has an invalid SHA-256 identity")
        program_id = identities.get(program_hash)
        if program_id is None:
          program_id = next_id
          next_id += 1
          identities[program_hash] = program_id
        program = replace(sized, program_id=program_id, version=1, program_hash=program_hash)
        finalized_objects[id(role.tile_program)] = program
      roles[role_id] = replace(role, tile_program=program)
    tasks[binding_id] = replace(task, role_bindings=roles)
  if isinstance(entry, ExecModel):
    return replace(entry, tasks=tasks)
  return tasks[entry.binding_id]


def _input_index(task: ExecTileGroupTask, view: ExecMemoryView) -> int:
  if not view.base.startswith("global:"):
    raise ValueError("relocation view is not backed by a Context global input")
  name = view.base.removeprefix("global:")
  for index, item in enumerate(task.global_inputs):
    if item.name == name:
      return index
  raise ValueError(f"relocation references absent Context input {name!r}")


def _relocations(entry: ExecModel | ExecTileGroupTask) -> tuple[Relocation, ...]:
  result: list[Relocation] = []
  for binding_id, task in _entry_tasks(entry).items():
    first_dispatch: dict[int, int] = {}
    for ordinal, action in enumerate(task.actions):
      if action.dst:
        result.append(Relocation(binding_id, "group", ordinal, "dst", "event"))
      if action.dependencies:
        result.append(Relocation(binding_id, "group", ordinal, "dependencies", "event_tuple"))
      if action.op in (ExecGroupActionOp.DMA_PREFETCH, ExecGroupActionOp.DMA_STORE):
        transfer = action.args[1]
        view = transfer.src if action.op is ExecGroupActionOp.DMA_PREFETCH else transfer.dst
        result.append(
          Relocation(binding_id, "context", ordinal, "global_view", "binding", _input_index(task, view), -1)
        )
      elif action.op is ExecGroupActionOp.DISPATCH_ROLE:
        first_dispatch.setdefault(action.args[0].role_id, ordinal)
        result.append(Relocation(binding_id, "group", ordinal, "args", "dispatch_events"))
      elif action.op in (ExecGroupActionOp.WAIT_EVENT, ExecGroupActionOp.SIGNAL_EVENT):
        result.append(Relocation(binding_id, "group", ordinal, "args", "event_tuple"))
      elif action.op is ExecGroupActionOp.RELEASE_L2:
        result.append(Relocation(binding_id, "group", ordinal, "args", "release_events"))
      elif action.op is ExecGroupActionOp.PUBLISH_L2:
        result.append(Relocation(binding_id, "group", ordinal, "args", "publish_events"))
      elif action.op is ExecGroupActionOp.PROFILE_RECONFIG:
        result.append(Relocation(binding_id, "group", ordinal, "args", "profile_frontier"))
      elif action.op is ExecGroupActionOp.MEMORY_MAINTENANCE:
        result.append(Relocation(binding_id, "group", ordinal, "args", "maintenance_dependencies"))
      elif action.op is ExecGroupActionOp.INIT_STREAM:
        result.append(Relocation(binding_id, "group", ordinal, "args", "queue", 0))
    result.append(Relocation(binding_id, "task", 0, "completion_event", "event"))
    result.append(Relocation(binding_id, "task", 0, "event_uses", "event_map"))
    for ordinal, _stream in enumerate(task.streams):
      result.append(Relocation(binding_id, "stream", ordinal, "queue_id", "queue"))
    for role_id in sorted(task.role_bindings):
      role = task.role_bindings[role_id]
      if role.in_stream is not None:
        result.append(Relocation(binding_id, "role", 0, "in_stream", "queue", -1, role_id))
      if role.out_stream is not None:
        result.append(Relocation(binding_id, "role", 0, "out_stream", "queue", -1, role_id))
      dispatch_ordinal = first_dispatch.get(role_id)
      if dispatch_ordinal is None:
        raise ValueError(f"role {role_id} in {binding_id!r} is never dispatched")
      for view in role.global_actuals:
        result.append(
          Relocation(
            binding_id,
            "tile_role",
            dispatch_ordinal,
            "global_actuals",
            "binding",
            _input_index(task, view),
            role_id,
          )
        )

  # Preserve deterministic traversal while rejecting an ambiguous duplicate.
  keys = [tuple(vars(item).values()) for item in result]
  if len(keys) != len(set(keys)):
    raise ValueError("compiler produced duplicate relocation coordinates")
  return tuple(result)


def _source_map(
  entry: ExecModel | ExecTileGroupTask, prefix: Sequence[ExecDeviceOp]
) -> dict[str, SourceRef]:
  result: dict[str, SourceRef] = {}

  def add(item: ExecDeviceOp | ExecGroupAction | ExecTileInst) -> None:
    if not item.instruction_id or item.source_ref is None:
      raise ValueError("every executable instruction requires identity and source_ref")
    if item.instruction_id in result:
      raise ValueError(f"duplicate executable instruction_id {item.instruction_id!r}")
    result[item.instruction_id] = item.source_ref

  for prefix_op in prefix:
    add(prefix_op)
  if isinstance(entry, ExecModel):
    for device_op in entry.body:
      add(device_op)
  for task in _entry_tasks(entry).values():
    for group_action in task.actions:
      add(group_action)
    seen: set[tuple[int, tuple[str, ...]]] = set()
    for role in task.role_bindings.values():
      key = (role.tile_program.program_hash, tuple(inst.instruction_id for inst in role.tile_program.insts))
      if key in seen:
        continue
      seen.add(key)
      for tile_inst in role.tile_program.insts:
        add(tile_inst)
  return result


def _dependency_proofs(entry: ExecModel | ExecTileGroupTask) -> dict[str, object]:
  proofs: dict[str, object] = {}
  if isinstance(entry, ExecModel):
    for op in entry.body:
      proofs[op.instruction_id] = {
        "scope": "device",
        "dependencies": tuple(op.dependencies),
        "event": op.event_tag,
      }
  for task in _entry_tasks(entry).values():
    for action in task.actions:
      proofs[action.instruction_id] = {
        "scope": "group",
        "dependencies": tuple(action.dependencies),
        "reads": tuple(action.reads),
        "writes": tuple(action.writes),
        "outputs": tuple(action.output_events),
      }
  return proofs


def _binding_guards_and_effects(
  entry: ExecModel | ExecTileGroupTask,
) -> tuple[tuple[BindingGuard, ...], dict[str, object]]:
  inputs = entry.inputs if isinstance(entry, ExecModel) else entry.global_inputs
  minimum = [item.size_bytes for item in inputs]
  permissions: list[set[str]] = [set() for _ in inputs]
  effects: dict[str, object] = {}
  if isinstance(entry, ExecModel):
    operations = (item for item in entry.body if item.op == "submit")
    for op in operations:
      task = entry.tasks[op.binding_id]
      accesses = _mapped_accesses(task, op.actual_inputs)
      effects[op.binding_id] = tuple(
        {
          "input_index": access.input_index,
          "offset": access.start,
          "bytes": access.end - access.start,
          "access": "write" if access.writing else "read",
          "cache_levels": access.levels,
        }
        for access in accesses
      )
      for access in accesses:
        minimum[access.input_index] = max(minimum[access.input_index], access.end)
        permissions[access.input_index].add("w" if access.writing else "r")
  else:
    accesses = _mapped_accesses(entry, tuple(range(len(entry.global_inputs))))
    effects[entry.binding_id] = tuple(
      {
        "input_index": access.input_index,
        "offset": access.start,
        "bytes": access.end - access.start,
        "access": "write" if access.writing else "read",
        "cache_levels": access.levels,
      }
      for access in accesses
    )
    for access in accesses:
      minimum[access.input_index] = max(minimum[access.input_index], access.end)
      permissions[access.input_index].add("w" if access.writing else "r")
  guards = tuple(
    BindingGuard(
      item.name,
      minimum[index],
      "rw" if permissions[index] == {"r", "w"} else next(iter(permissions[index]), "r"),
    )
    for index, item in enumerate(inputs)
  )
  return guards, effects


def _validate_binding_assumptions(
  assumptions: Mapping[str, GlobalBinding] | None, guards: Sequence[BindingGuard]
) -> None:
  if assumptions is None:
    return
  if set(assumptions) != {guard.name for guard in guards}:
    raise ValueError("binding_assumptions must name every entry input exactly once")
  ordered = []
  for guard in guards:
    binding = assumptions[guard.name]
    if not isinstance(binding, GlobalBinding) or binding.name != guard.name:
      raise ValueError(f"binding assumption {guard.name!r} has inconsistent identity")
    if binding.size_bytes < guard.minimum_bytes:
      raise ValueError(f"binding assumption {guard.name!r} is smaller than compiled access range")
    if binding.permissions not in ("r", "w", "rw") or not set(guard.permissions) <= set(
      binding.permissions
    ):
      raise ValueError(f"binding assumption {guard.name!r} weakens required permissions")
    if type(binding.base_iova) is not int or binding.base_iova < 0 or type(binding.size_bytes) is not int:
      raise ValueError(f"binding assumption {guard.name!r} has invalid address or size")
    ordered.append(binding)
  ordered.sort(key=lambda item: item.base_iova)
  for left, right in pairwise(ordered):
    if right.base_iova < left.base_iova + left.size_bytes:
      raise ValueError(f"binding assumptions {left.name!r} and {right.name!r} overlap")


def compile_program(
  module: ModuleOp,
  hw: HardwareConfig,
  sim: SimConfig,
  *,
  binding_assumptions: Mapping[str, GlobalBinding] | None = None,
  source_name: str = "<memory>",
  workload_info: WorkloadInfo | None = None,
) -> CompiledProgram:
  """Compile source IR into a sealed, immutable, target-bound executable."""
  if not isinstance(module, ModuleOp):
    raise ValueError("compile_program requires an xDSL builtin.module")
  if not isinstance(hw, HardwareConfig) or not isinstance(sim, SimConfig):
    raise ValueError("compile_program requires HardwareConfig and SimConfig")
  if not isinstance(source_name, str) or not source_name:
    raise ValueError("source_name must be a nonempty string")
  source_ir = print_workload_ir(module)
  source_hash = digest(source_ir)
  source_entry = verify_workload_ir(module)
  entry_kind = "model" if source_entry.name == "nexus.program" else "standalone"
  if isinstance(source_entry, NestContextOp) and source_entry.context_id is not None:
    if int(source_entry.context_id.value.data) != 0:
      raise ValueError("standalone nest.context must use Group context slot 0")
  entry = (
    lower_model_ir(module, source_name=source_name)
    if entry_kind == "model"
    else lower_workload_ir(module, source_name=source_name)
  )
  registry = build_registry(hw)
  entry, budgets, resource_effects = prepare_resources(entry, registry, hw, sim)
  entry, prefix, call_bindings, entry_profiles, exit_profiles = bind_profiles(entry, registry)
  entry = _finalize_action_dependencies(entry)
  entry, budgets = finalize_event_resources(entry, budgets, sim)
  entry = _finalize_program_identities(entry, hw)
  guards, access_effects = _binding_guards_and_effects(entry)
  _validate_binding_assumptions(binding_assumptions, guards)
  if print_workload_ir(module) != source_ir:
    raise ValueError("compiler pass mutated its source IR")
  if workload_info is None:
    name = entry.name
    workload_info = WorkloadInfo(name, f"compiled executable for {name}", {})
  elif not isinstance(workload_info, WorkloadInfo):
    raise ValueError("workload_info must be WorkloadInfo")
  from ..execution_verifier import target_fingerprint, verify_compiled_program

  program = CompiledProgram(
    2,
    "v1",
    source_hash,
    source_ir,
    registry,
    registry.registry_hash,
    target_fingerprint(hw, sim),
    "0" * 64,
    entry_kind,
    entry,
    tuple(prefix),
    call_bindings,
    _relocations(entry),
    _source_map(entry, prefix),
    _dependency_proofs(entry),
    entry_profiles,
    exit_profiles,
    {
      "resources": resource_effects,
      "accesses": access_effects,
      "program_text_bytes": {
        binding_id: tuple(sorted({role.tile_program.text_bytes for role in task.role_bindings.values()}))
        for binding_id, task in _entry_tasks(entry).items()
      },
    },
    budgets,
    guards,
    workload_info,
  )
  program = seal_program(program)
  verify_compiled_program(program, hw, sim)
  return program


def _attr(value: object) -> str:
  return json.dumps(json.dumps(canonical_value(value), sort_keys=True, separators=(",", ":")))


def dump_compiled_source(program: CompiledProgram) -> str:
  """Render executable control as generic MLIR operations.

  This is deliberately derived only from the immutable package.  Generated
  ordinary awaits and control descriptors are operations, never comments or
  source-name inference.
  """
  lines = ["builtin.module {"]

  def emit(name: str, item, indent: str = "  ") -> None:
    attrs = {"instruction_id": item.instruction_id, "source_ref": canonical_value(item.source_ref)}
    if isinstance(item, ExecDeviceOp):
      attrs.update(
        {
          "event": item.event_tag,
          "dependencies": item.dependencies,
          "binding_id": item.binding_id,
          "command": canonical_value(item.command),
        }
      )
    else:
      attrs.update(
        {"dependencies": item.dependencies, "args": canonical_value(item.args), "dst": item.dst or ""}
      )
    rendered = ", ".join(
      f"{key} = {_attr(value)}" for key, value in attrs.items() if value not in ("", (), None)
    )
    lines.append(f'{indent}"{name}"() {{{rendered}}} : () -> ()')

  for item in program.entry_prefix:
    name = "nexus.await" if item.op == "await" else f"compiler.{item.op}"
    emit(name, item)
  if isinstance(program.entry, ExecModel):
    for item in program.entry.body:
      name = "nexus.await" if item.op == "await" else f"compiler.{item.op}"
      emit(name, item)
  for binding_id, task in _entry_tasks(program.entry).items():
    lines.append('  "compiler.context"() ({')
    for action in task.actions:
      name = "nest.await" if action.op is ExecGroupActionOp.WAIT_EVENT else action.op.value
      emit(name, action, "    ")
    lines.append(
      f"  }}) {{binding_id = {_attr(binding_id)}, "
      f"resource_contract = {_attr(task.resource_contract)}}} : () -> ()"
    )
  lines.append("}")
  return "\n".join(lines) + "\n"
