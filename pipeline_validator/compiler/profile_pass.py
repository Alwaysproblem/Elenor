"""Bind layered profiles and compile explicit synchronization/control sequences."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..execution_ir import (
  ExecDeviceOp,
  ExecGatherDesc,
  ExecGatherOutcome,
  ExecGroupAction,
  ExecGroupActionOp,
  ExecMemoryView,
  ExecModel,
  ExecTileGroupTask,
)
from ..profiles import (
  CallBinding,
  ContextResources,
  MaintenanceRange,
  MemoryMaintenanceDesc,
  ProfileReconfigDesc,
  ProfileRegistry,
  ProfileState,
  SourceRef,
  TileResources,
)
from ..workload_ir import _view_offset_bytes
from .lowering import normalize_action_dependencies


def _generated(source: SourceRef | None, generated_by: str, reason: str) -> SourceRef:
  if source is None:
    raise ValueError("compiler operation lacks a source reference")
  return replace(source, generated_by=generated_by, reason=reason)


def _profile_command(
  registry: ProfileRegistry,
  level: str,
  old: int,
  new: int,
  command_id: str,
  waits: tuple[str, ...],
  frontier: tuple[str, ...],
  exclusive: str,
  source: SourceRef,
) -> ProfileReconfigDesc:
  return ProfileReconfigDesc(
    command_id,
    level,
    old,
    new,
    registry.registry_hash,
    waits,
    frontier,
    (level,),
    registry.profile(level, new).member_ids,
    exclusive,
    source,
  )


def _has_l1(task: ExecTileGroupTask) -> bool:
  # Even a zero-byte Task owns a profile-bound Arena, Frame and UCE lease.
  return any(action.op is ExecGroupActionOp.DISPATCH_ROLE for action in task.actions)


def _first_l1(task: ExecTileGroupTask, current: int) -> int:
  for action in task.actions:
    if action.op is not ExecGroupActionOp.DISPATCH_ROLE:
      continue
    request = action.args[0]
    program = task.role_bindings[request.role_id].tile_program
    contract = program.resource_contract
    if not isinstance(contract, TileResources):
      raise ValueError(f"Tile Program {program.name!r} requires resource_contract")
    if request.requested_l1_mode not in contract.allowed_profiles:
      raise ValueError(
        f"dispatch {action.instruction_id!r} requests L1 mode {request.requested_l1_mode} "
        f"outside Tile Program {program.name!r} allowed_profiles"
      )
    return current if current in contract.allowed_profiles else request.requested_l1_mode
  return current


@dataclass(frozen=True)
class _GlobalAccess:
  input_index: int
  start: int
  end: int
  writing: bool
  event: str
  levels: tuple[str, ...]


def _view_access(
  task: ExecTileGroupTask, view: ExecMemoryView, writing: bool, event: str, levels: tuple[str, ...] = ()
) -> _GlobalAccess:
  if not view.base.startswith("global:"):
    raise ValueError("compiler visibility analysis requires a resolved global view")
  name = view.base.removeprefix("global:")
  input_index = next((index for index, item in enumerate(task.global_inputs) if item.name == name), None)
  if input_index is None:
    raise ValueError(f"global view {name!r} has no Context formal backing")
  start = _view_offset_bytes(view.offsets, view.backing_dims, view.element_bytes)
  return _GlobalAccess(input_index, start, start + view.bytes, writing, event, levels)


def _action_accesses(task: ExecTileGroupTask, action: ExecGroupAction) -> tuple[_GlobalAccess, ...]:
  if action.op is ExecGroupActionOp.DMA_PREFETCH:
    transfer = action.args[1]
    return (_view_access(task, transfer.src, False, action.dst or ""),)
  if action.op is ExecGroupActionOp.DMA_STORE:
    transfer = action.args[1]
    return (_view_access(task, transfer.dst, True, action.dst or ""),)
  if action.op is not ExecGroupActionOp.DISPATCH_ROLE:
    return ()
  role = task.role_bindings[action.args[0].role_id]
  global_formals = [
    index for index, formal in enumerate(role.tile_program.formals) if formal.space == "global"
  ]
  actuals = dict(zip(global_formals, role.global_actuals))
  result: list[_GlobalAccess] = []
  for descriptor in role.tile_program.descriptors.values():
    gather = descriptor.params.get("gather")
    if isinstance(gather, ExecGatherDesc):
      formal = int(gather.source.base.removeprefix("formal:"))
      levels = (
        ("l1",)
        if all(access.outcome is ExecGatherOutcome.L1_HIT for access in gather.accesses)
        else ("l1", "l2")
      )
      result.append(_view_access(task, actuals[formal], False, action.dst or "", levels))
  return tuple(result)


def _overlaps(left: _GlobalAccess, right: _GlobalAccess) -> bool:
  return left.input_index == right.input_index and left.start < right.end and right.start < left.end


def _maintenance_range(
  registry: ProfileRegistry, inputs, access: _GlobalAccess, levels: tuple[str, ...]
) -> MaintenanceRange:
  full_domain = any(
    "invalidate_range" not in registry.target.__getattribute__(level).maintenance_caps for level in levels
  )
  if full_domain:
    unsupported = [
      level
      for level in levels
      if "clean_invalidate_all" not in registry.target.__getattribute__(level).maintenance_caps
    ]
    if unsupported:
      raise ValueError(
        f"target lacks range or full-domain maintenance for levels {unsupported}; "
        "cannot preserve HBM visibility"
      )
    return MaintenanceRange(access.input_index, 0, inputs[access.input_index].size_bytes, levels)
  return MaintenanceRange(access.input_index, access.start, access.end - access.start, levels)


def _writeback_levels(registry: ProfileRegistry) -> tuple[str, ...]:
  result = []
  for level in ("l1", "l2"):
    table = getattr(registry, level)
    if registry.target.__getattribute__(level).cache_write_policy == "write_back" and any(
      profile.cache_bytes for profile in table.values()
    ):
      result.append(level)
  return tuple(result)


def _writeback_levels_for(
  registry: ProfileRegistry, l1_modes: tuple[int, ...], l2_mode: int
) -> tuple[str, ...]:
  """Write-back Cache levels that are actually active under the given modes."""
  result: list[str] = []
  l1_target = registry.target.l1
  if l1_target.cache_write_policy == "write_back" and any(
    registry.profile("l1", mode).cache_bytes for mode in l1_modes
  ):
    result.append("l1")
  l2_target = registry.target.l2
  if l2_target.cache_write_policy == "write_back" and registry.profile("l2", l2_mode).cache_bytes:
    result.append("l2")
  return tuple(result)


def _task_l1_modes(task: ExecTileGroupTask, entry_l1: int, exit_l1: int) -> tuple[int, ...]:
  modes = {entry_l1, exit_l1}
  for action in task.actions:
    if action.op is ExecGroupActionOp.DISPATCH_ROLE:
      modes.add(action.args[0].resolved_l1_mode)
  return tuple(sorted(modes))


def _prewrite_clean(
  registry: ProfileRegistry,
  inputs,
  command_id: str,
  writes: tuple[_GlobalAccess, ...],
  dependencies: tuple[str, ...],
  source: SourceRef,
  writeback_levels: tuple[str, ...],
) -> ExecDeviceOp | None:
  """Emit a device-level clean of stale dirty lines before HBM writes.

  Under a write-back policy the shared Cache may hold an older dirty line for
  a range a producer is about to overwrite.  The verifier requires a covering
  maintenance command before every such write, regardless of tracked reads.
  """
  if not writes or not writeback_levels:
    return None
  ranges = tuple(
    dict.fromkeys(_maintenance_range(registry, inputs, item, writeback_levels) for item in writes)
  )
  desc = MemoryMaintenanceDesc(command_id, writeback_levels, ranges, dependencies, source)
  return ExecDeviceOp(
    "memory_maintenance",
    dependencies=dependencies,
    command=desc,
    source_ref=source,
    instruction_id=desc.command_id,
  )


def insert_task_maintenance(task: ExecTileGroupTask, registry: ProfileRegistry) -> ExecTileGroupTask:
  """Insert pre-clean and post-write invalidation at legal Group boundaries."""
  output: list[ExecGroupAction] = []
  prior_writes: list[_GlobalAccess] = []
  prior_cached_reads: list[_GlobalAccess] = []
  writeback_levels = _writeback_levels(registry)
  for action in task.actions:
    accesses = _action_accesses(task, action)
    writes = tuple(item for item in accesses if item.writing)
    cached_reads = tuple(item for item in accesses if not item.writing and item.levels)

    # A write-back cache may contain an older dirty line.  Clean it before a
    # new HBM producer writes, so the old value can never overwrite the new.
    if writes and writeback_levels:
      overlapping = [old for old in prior_cached_reads for new in writes if _overlaps(old, new)]
      if overlapping:
        dependencies = tuple(sorted(dict.fromkeys(item.event for item in overlapping)))
        source = _generated(
          action.source_ref, "memory_visibility_pass", "clean old dirty cache lines before HBM write"
        )
        ranges = tuple(
          dict.fromkeys(
            _maintenance_range(registry, task.global_inputs, item, writeback_levels) for item in writes
          )
        )
        desc = MemoryMaintenanceDesc(
          f"{action.instruction_id}:preclean", writeback_levels, ranges, dependencies, source
        )
        output.append(
          ExecGroupAction(
            ExecGroupActionOp.MEMORY_MAINTENANCE,
            args=(desc,),
            dependencies=dependencies,
            source_ref=source,
            instruction_id=desc.command_id,
          )
        )

    if cached_reads:
      overlapping_writes = [
        old for old in prior_writes for current in cached_reads if _overlaps(old, current)
      ]
      if overlapping_writes:
        dependencies = tuple(sorted(dict.fromkeys(item.event for item in overlapping_writes)))
        levels = tuple(dict.fromkeys(level for item in cached_reads for level in item.levels))
        source = _generated(
          action.source_ref, "memory_visibility_pass", "invalidate cache after overlapping HBM write"
        )
        ranges = tuple(
          dict.fromkeys(
            _maintenance_range(registry, task.global_inputs, item, item.levels)
            for item in cached_reads
            if any(_overlaps(old, item) for old in overlapping_writes)
          )
        )
        desc = MemoryMaintenanceDesc(
          f"{action.instruction_id}:postwrite_invalidate", levels, ranges, dependencies, source
        )
        output.append(
          ExecGroupAction(
            ExecGroupActionOp.MEMORY_MAINTENANCE,
            args=(desc,),
            dependencies=dependencies,
            source_ref=source,
            instruction_id=desc.command_id,
          )
        )
    output.append(action)
    prior_writes.extend(writes)
    prior_cached_reads.extend(cached_reads)
  return replace(task, actions=normalize_action_dependencies(output, task.role_bindings))


def bind_context_profiles(
  task: ExecTileGroupTask, registry: ProfileRegistry, entry_l1: int, l2_mode: int
) -> tuple[ExecTileGroupTask, CallBinding]:
  contract = task.resource_contract
  if not isinstance(contract, ContextResources):
    raise ValueError(f"Context {task.name!r} requires resource_contract")
  if l2_mode not in contract.allowed_profiles:
    raise ValueError(f"resolved L2 mode {l2_mode} is outside Context {task.name!r} allowed_profiles")
  actions: list[ExecGroupAction] = []
  current = entry_l1
  frontier: list[str] = []
  awaited: dict[str, str] = {}
  permitted: list[ProfileState] = [ProfileState(entry_l1, l2_mode)]
  exclusive = False
  for action in task.actions:
    if action.op is ExecGroupActionOp.WAIT_EVENT:
      awaited[action.args[0]] = action.instruction_id
    if action.op is ExecGroupActionOp.DISPATCH_ROLE:
      request = action.args[0]
      program = task.role_bindings[request.role_id].tile_program
      tile_contract = program.resource_contract
      if not isinstance(tile_contract, TileResources):
        raise ValueError(f"Tile Program {program.name!r} requires resource_contract")
      if request.requested_l1_mode not in tile_contract.allowed_profiles:
        raise ValueError(
          f"dispatch {action.instruction_id!r} requested mode {request.requested_l1_mode} "
          f"outside Tile Program {program.name!r} allowed_profiles"
        )
      selected = current if current in tile_contract.allowed_profiles else request.requested_l1_mode
      if selected != current:
        if not frontier:
          raise ValueError(
            f"dispatch {action.instruction_id!r} changes L1 without a preceding finite Grid frontier"
          )
        source = _generated(action.source_ref, "profile_boundary_pass", "retire prior L1 Grid frontier")
        for event in frontier:
          if event not in awaited:
            instruction_id = f"{action.instruction_id}:generated_wait:{event}"
            actions.append(
              ExecGroupAction(
                ExecGroupActionOp.WAIT_EVENT,
                args=(event,),
                dependencies=(event,),
                source_ref=source,
                instruction_id=instruction_id,
              )
            )
            awaited[event] = instruction_id
        wait_ids = tuple(dict.fromkeys(awaited[event] for event in frontier))
        command = _profile_command(
          registry,
          "l1",
          current,
          selected,
          f"{action.instruction_id}:profile:l1",
          wait_ids,
          tuple(frontier),
          task.binding_id,
          source,
        )
        actions.append(
          ExecGroupAction(
            ExecGroupActionOp.PROFILE_RECONFIG,
            args=(command,),
            dependencies=tuple(frontier),
            source_ref=source,
            instruction_id=command.command_id,
          )
        )
        current = selected
        exclusive = True
        frontier.clear()
      request = replace(request, resolved_l1_mode=current, binding_id=task.binding_id)
      action = replace(action, args=(request,))
      if action.dst is None:
        raise ValueError("dispatch requires grid_done")
      frontier.append(action.dst)
      state = ProfileState(current, l2_mode)
      if state not in permitted:
        permitted.append(state)
    actions.append(action)
  binding = CallBinding(
    task.binding_id,
    task.name,
    contract.l2_mode,
    l2_mode,
    entry_l1,
    current,
    exclusive,
    (),
    tuple(permitted),
  )
  return replace(task, actions=tuple(actions)), binding


def _mapped_accesses(task: ExecTileGroupTask, actual_inputs: tuple[int, ...]) -> tuple[_GlobalAccess, ...]:
  return tuple(
    replace(access, input_index=actual_inputs[access.input_index])
    for action in task.actions
    for access in _action_accesses(task, action)
  )


def _device_maintenance(
  registry: ProfileRegistry,
  inputs,
  trigger: ExecDeviceOp,
  previous: list[tuple[str, tuple[_GlobalAccess, ...]]],
  current: tuple[_GlobalAccess, ...],
) -> ExecDeviceOp | None:
  cached_reads = tuple(item for item in current if not item.writing and item.levels)
  overlaps = [
    (event, old, new)
    for event, accesses in previous
    for old in accesses
    if old.writing
    for new in cached_reads
    if _overlaps(old, new)
  ]
  if not overlaps:
    return None
  dependencies = tuple(sorted(dict.fromkeys(event for event, _, _ in overlaps)))
  levels = tuple(dict.fromkeys(level for _, _, item in overlaps for level in item.levels))
  ranges = tuple(
    dict.fromkeys(_maintenance_range(registry, inputs, item, item.levels) for _, _, item in overlaps)
  )
  source = _generated(trigger.source_ref, "memory_visibility_pass", "cross-Context HBM-to-cache visibility")
  desc = MemoryMaintenanceDesc(
    f"{trigger.instruction_id}:memory_maintenance", levels, ranges, dependencies, source
  )
  return ExecDeviceOp(
    "memory_maintenance",
    dependencies=dependencies,
    command=desc,
    source_ref=source,
    instruction_id=desc.command_id,
  )


def bind_profiles(
  entry: ExecModel | ExecTileGroupTask, registry: ProfileRegistry
) -> tuple[
  ExecModel | ExecTileGroupTask,
  tuple[ExecDeviceOp, ...],
  dict[str, CallBinding],
  ProfileState,
  ProfileState,
]:
  """Resolve compatible modes and emit all required ordinary awaits/control."""
  initial = ProfileState(registry.target.l1.reset_mode, registry.target.l2.reset_mode)
  current = initial
  bindings: dict[str, CallBinding] = {}
  prefix: list[ExecDeviceOp] = []
  if isinstance(entry, ExecTileGroupTask):
    task = insert_task_maintenance(entry, registry)
    contract = task.resource_contract
    if not isinstance(contract, ContextResources):
      raise ValueError(f"Context {task.name!r} requires resource_contract")
    target_l2 = current.l2_mode if current.l2_mode in contract.allowed_profiles else contract.l2_mode
    target_l1 = _first_l1(task, current.l1_mode)
    source = (
      task.actions[0].source_ref if task.actions else SourceRef("<memory>", task.name, 0, "nest.context")
    )
    for level, old, new in (("l2", current.l2_mode, target_l2), ("l1", current.l1_mode, target_l1)):
      if old == new:
        continue
      ref = _generated(source, "profile_boundary_pass", "verified initialization proof")
      command = _profile_command(registry, level, old, new, f"entry:profile:{level}", (), (), "", ref)
      prefix.append(
        ExecDeviceOp("profile_reconfig", command=command, source_ref=ref, instruction_id=command.command_id)
      )
      current = ProfileState(new, current.l2_mode) if level == "l1" else ProfileState(current.l1_mode, new)
    task, binding = bind_context_profiles(task, registry, target_l1, target_l2)
    writes = tuple(
      item for action in task.actions for item in _action_accesses(task, action) if item.writing
    )
    clean = _prewrite_clean(
      registry,
      task.global_inputs,
      "entry:prewrite_clean",
      writes,
      (),
      _generated(source, "memory_visibility_pass", "clean old dirty cache lines before HBM write"),
      _writeback_levels_for(
        registry,
        _task_l1_modes(task, binding.entry_l1_mode, binding.exit_l1_mode),
        binding.resolved_l2_mode,
      ),
    )
    if clean is not None:
      prefix.append(clean)
    bindings[task.binding_id] = binding
    return task, tuple(prefix), bindings, initial, ProfileState(binding.exit_l1_mode, target_l2)

  tasks = {binding_id: insert_task_maintenance(task, registry) for binding_id, task in entry.tasks.items()}
  output: list[ExecDeviceOp] = []
  active: dict[str, CallBinding] = {}
  history: dict[str, list[str]] = {"l1": [], "l2": []}
  awaited: dict[str, str] = {}
  submitted: set[str] = set()
  previous_accesses: list[tuple[str, tuple[_GlobalAccess, ...]]] = []

  def wait_events(events, trigger: ExecDeviceOp, reason: str) -> None:
    ref = _generated(trigger.source_ref, "profile_boundary_pass", reason)
    for event in tuple(dict.fromkeys(events)):
      if event not in submitted:
        raise ValueError(f"profile await producer {event!r} has not been submitted")
      if event not in awaited:
        instruction_id = f"{trigger.instruction_id}:generated_wait:{event}"
        output.append(ExecDeviceOp("await", event_tag=event, source_ref=ref, instruction_id=instruction_id))
        awaited[event] = instruction_id
      active.pop(event, None)

  def reconfigure(level: str, old: int, new: int, trigger: ExecDeviceOp) -> None:
    if old == new:
      return
    frontier = tuple(history[level])
    wait_events(frontier, trigger, f"retire complete {level} root history")
    ref = _generated(
      trigger.source_ref,
      "profile_boundary_pass",
      f"change {level} profile" if frontier else "verified initialization proof",
    )
    command = _profile_command(
      registry,
      level,
      old,
      new,
      f"{trigger.instruction_id}:profile:{level}",
      tuple(dict.fromkeys(awaited[event] for event in frontier)),
      frontier,
      "",
      ref,
    )
    output.append(
      ExecDeviceOp("profile_reconfig", command=command, source_ref=ref, instruction_id=command.command_id)
    )
    history[level].clear()

  for source_op in entry.body:
    if source_op.op == "await":
      if source_op.event_tag not in submitted:
        raise ValueError(
          f"Device await {source_op.instruction_id!r} references unsubmitted event {source_op.event_tag!r}"
        )
      if source_op.event_tag not in awaited:
        awaited[source_op.event_tag] = source_op.instruction_id
        active.pop(source_op.event_tag, None)
        output.append(source_op)
      continue
    if source_op.op == "return":
      output.append(source_op)
      continue
    if source_op.op != "submit":
      raise ValueError(f"unsupported lowered Device op {source_op.op!r}")
    if not set(source_op.dependencies) <= submitted:
      raise ValueError(
        f"Device submit {source_op.instruction_id!r} has forward or unknown dependencies "
        f"{sorted(set(source_op.dependencies) - submitted)}"
      )
    task = tasks[source_op.binding_id]
    contract = task.resource_contract
    if not isinstance(contract, ContextResources):
      raise ValueError(f"Context {task.name!r} requires resource_contract")
    target_l2 = current.l2_mode if current.l2_mode in contract.allowed_profiles else contract.l2_mode
    target_l1 = _first_l1(task, current.l1_mode)
    bound, binding = bind_context_profiles(task, registry, target_l1, target_l2)
    binding = replace(binding, actual_inputs=source_op.actual_inputs)
    mapped = _mapped_accesses(bound, source_op.actual_inputs)

    # Compile model-level RAW/WAR/WAW edges from clean per-call summaries.
    hazard_events = []
    for event, old_accesses in previous_accesses:
      if any(
        _overlaps(old, new) and (old.writing or new.writing) for old in old_accesses for new in mapped
      ):
        hazard_events.append(event)
    submit = replace(
      source_op, dependencies=tuple(dict.fromkeys((*source_op.dependencies, *hazard_events)))
    )
    maintenance = _device_maintenance(registry, entry.inputs, submit, previous_accesses, mapped)
    if maintenance is not None:
      output.append(maintenance)
    prewrite = _prewrite_clean(
      registry,
      entry.inputs,
      f"{submit.instruction_id}:prewrite_clean",
      tuple(item for item in mapped if item.writing),
      submit.dependencies,
      _generated(submit.source_ref, "memory_visibility_pass", "clean stale dirty lines before HBM write"),
      _writeback_levels_for(
        registry,
        _task_l1_modes(bound, binding.entry_l1_mode, binding.exit_l1_mode),
        binding.resolved_l2_mode,
      ),
    )
    if prewrite is not None:
      output.append(prewrite)

    if _has_l1(bound):
      exclusive_roots = [event for event, old in active.items() if old.requires_l1_exclusive]
      wait_events(exclusive_roots, submit, "previous root owns exclusive L1 control")
    if binding.requires_l1_exclusive:
      producers = [event for event, old in active.items() if _has_l1(tasks[old.binding_id])]
      wait_events(producers, submit, "acquire exclusive L1 root interval")
    reconfigure("l2", current.l2_mode, target_l2, submit)
    reconfigure("l1", current.l1_mode, target_l1, submit)
    tasks[submit.binding_id] = bound
    bindings[submit.binding_id] = binding
    current = ProfileState(binding.exit_l1_mode, target_l2)
    output.append(submit)
    submitted.add(submit.event_tag)
    active[submit.event_tag] = binding
    history["l2"].append(submit.event_tag)
    if _has_l1(bound):
      history["l1"].append(submit.event_tag)
    previous_accesses.append((submit.event_tag, mapped))
  return replace(entry, tasks=tasks, body=tuple(output)), (), bindings, initial, current
