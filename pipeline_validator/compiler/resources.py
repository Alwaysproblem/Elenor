"""Static striped layout, capability, and finite-resource analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import lcm

from ..execution_ir import (
  ExecGroupActionOp,
  ExecL1Buffer,
  ExecL2Buffer,
  ExecModel,
  ExecTileGatherDesc,
  ExecTileGroupTask,
  ExecTileOp,
  ExecTileProgram,
  ExecTileRoleBinding,
)
from ..immutable import digest
from ..profiles import (
  UINT64_MAX,
  ArenaLayout,
  BufferLayout,
  CacheRequirement,
  ContextResources,
  MemoryProfile,
  ProfileRegistry,
  ResourceBudget,
  TileResources,
  uint64,
)


def _event_ancestors(
  dependencies: Sequence[str], ancestors: Mapping[str, frozenset[str]]
) -> frozenset[str]:
  result = set(dependencies)
  for event in dependencies:
    result.update(ancestors.get(event, frozenset()))
  return frozenset(result)


def conservative_arena_bytes(byte_alignments: Sequence[tuple[int, int]], profile: MemoryProfile) -> int:
  """Return the no-reuse contract size for logical ``(bytes, alignment)`` buffers.

  This is an authoring helper.  The compiler still derives and checks the
  authoritative layout from the actual executable allocations.
  """
  stripe = profile.alignment
  for logical_bytes, alignment in byte_alignments:
    uint64(logical_bytes, "buffer bytes", positive=True)
    uint64(alignment, "buffer alignment", positive=True)
    stripe = uint64(lcm(stripe, alignment), "stripe_bytes", positive=True)
  round_bytes = uint64(stripe * profile.banks, "stripe round", positive=True)
  result = 0
  for logical_bytes, _ in byte_alignments:
    result = uint64(
      result + ((logical_bytes + round_bytes - 1) // round_bytes) * round_bytes, "arena reservation"
    )
  return result


def layout_buffers(
  buffers: Sequence[ExecL1Buffer | ExecL2Buffer],
  profile: MemoryProfile,
  reserved_bytes: int,
  *,
  lifetimes: Mapping[str, tuple[int, int]] | None = None,
  slot_capacity: int = 16,
) -> ArenaLayout:
  """Deterministically first-fit source allocations in a striped Arena.

  Reuse occurs only when the supplied lifetime proves the previous view has
  ended before the next bind.  Every placement consumes whole all-bank stripe
  rounds, so tail padding is reserved but never exposed through a view.
  """
  uint64(reserved_bytes, "declared arena bytes")
  uint64(slot_capacity, "Frame Slot capacity", positive=bool(buffers))
  stripe = profile.alignment
  for buffer in buffers:
    uint64(buffer.bytes, "buffer bytes", positive=True)
    uint64(buffer.alignment, "buffer alignment", positive=True)
    stripe = uint64(lcm(stripe, buffer.alignment), "stripe_bytes", positive=True)
  round_bytes = uint64(stripe * profile.banks, "stripe round", positive=True)
  if reserved_bytes % round_bytes:
    raise ValueError(
      f"declared {profile.level} Arena bytes {reserved_bytes} are not a whole "
      f"{round_bytes}-byte stripe round"
    )
  live: list[tuple[int, int, int, int]] = []
  layouts: list[BufferLayout] = []
  high_water = 0
  names: set[str] = set()
  for index, buffer in enumerate(buffers):
    name = buffer.name if isinstance(buffer, ExecL1Buffer) else buffer.slot
    if name in names:
      raise ValueError(f"duplicate buffer layout {name!r}")
    names.add(name)
    start, end = lifetimes[name] if lifetimes is not None else (index, len(buffers) + 1)
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
      raise ValueError(f"invalid buffer lifetime for {name!r}")
    live = [item for item in live if item[3] > start]
    span = uint64(
      ((buffer.bytes + round_bytes - 1) // round_bytes) * round_bytes, f"padded span for {name}"
    )
    offset = 0
    for previous, size, _slot, _end in sorted(live):
      if offset + span <= previous:
        break
      offset = max(offset, previous + size)
    used_slots = {item[2] for item in live}
    slot = next((candidate for candidate in range(slot_capacity) if candidate not in used_slots), None)
    if slot is None:
      raise ValueError(f"live Frame Slot budget exceeds {slot_capacity} at buffer {name!r}")
    high_water = max(high_water, uint64(offset + span, "arena high-water mark"))
    live.append((offset, span, slot, end))
    layouts.append(
      BufferLayout(
        name,
        buffer.bytes,
        slot,
        offset,
        stripe,
        profile.banks,
        buffer.role if isinstance(buffer, ExecL2Buffer) else "",
      )
    )
  if high_water > reserved_bytes:
    raise ValueError(
      f"resource contract {reserved_bytes} does not cover {high_water} bytes of layout and padding"
    )
  if lifetimes is None:
    # No-rebind layout: distinct buffers must own disjoint padded spans.
    for index, left in enumerate(layouts):
      for right in layouts[index + 1 :]:
        if _segments_overlap(left, right):
          raise ValueError(
            f"no-rebind layout overlaps padded spans of {left.buffer_id!r} and {right.buffer_id!r}"
          )
  per_bank = tuple(reserved_bytes // profile.banks for _ in range(profile.banks))
  layout = ArenaLayout(stripe, stripe, reserved_bytes, per_bank, tuple(layouts), "")
  layout = replace(layout, layout_hash=digest(layout))
  check_layout_capacity(layout, profile)
  return layout


def check_layout_capacity(layout: ArenaLayout, profile: MemoryProfile, *, copies: int = 1) -> None:
  uint64(copies, "Arena copies", positive=True)
  if len(layout.per_bank_bytes) != profile.banks:
    raise ValueError("Arena bank count differs from target")
  if layout.stripe_bytes % profile.alignment or layout.alignment % profile.alignment:
    raise ValueError("Arena alignment is incompatible with target")
  if sum(layout.per_bank_bytes) != layout.reserved_bytes:
    raise ValueError("Arena per-bank reservation does not conserve bytes")
  for bank, length in enumerate(layout.per_bank_bytes):
    demand = uint64(copies * length, "R x per-bank Arena bytes")
    if profile.system_reserved_spm_per_bank + demand > profile.spm_bytes_per_bank:
      raise ValueError(
        f"permanent capacity: {profile.level} bank {bank} needs {demand} user bytes for R={copies}, "
        f"but profile {profile.mode} provides {profile.user_spm_per_bank}"
      )
  if copies * layout.reserved_bytes > UINT64_MAX:
    raise ValueError("R x Arena reservation overflows uint64")


def allowed_layouts(
  buffers: Sequence[ExecL1Buffer | ExecL2Buffer],
  registry: ProfileRegistry,
  level: str,
  allowed: Sequence[int],
  reserved_bytes: int,
  *,
  copies: int = 1,
  lifetimes: Mapping[str, tuple[int, int]] | None = None,
  slot_capacity: int = 16,
) -> ArenaLayout:
  selected: ArenaLayout | None = None
  for mode in allowed:
    profile = registry.profile(level, mode)
    layout = layout_buffers(
      buffers, profile, reserved_bytes, lifetimes=lifetimes, slot_capacity=slot_capacity
    )
    check_layout_capacity(layout, profile, copies=copies)
    if selected is not None and layout != selected:
      raise ValueError(f"{level} allowed profiles require incompatible static layouts")
    selected = layout
  if selected is None:
    raise ValueError(f"{level} allowed_profiles cannot be empty")
  return selected


def _check_requirement(
  requirement: CacheRequirement | None,
  profile: MemoryProfile,
  *,
  used: bool,
  bypass_needed: bool,
  required_needed: bool,
  writes: bool = False,
) -> None:
  if requirement is not None:
    if used and requirement.access == "none":
      raise ValueError(f"{profile.level} cache declaration understates actual accesses")
    if required_needed and not requirement.required:
      raise ValueError(f"{profile.level} cache declaration understates required access")
    if writes and requirement.access != "read_write":
      raise ValueError(f"{profile.level} cache declaration understates write access")
    if requirement.required and not profile.cache_bytes:
      raise ValueError(f"{profile.level} required Cache cannot be satisfied by bypass")
    if requirement.access == "read_write" and profile.cache_write_policy != "write_back":
      raise ValueError(f"{profile.level} write Cache requires write_back capability")
  if used and not profile.cache_bytes:
    raise ValueError(f"{profile.level} actual Cache access requires nonzero capacity")
  if bypass_needed and (
    "bypass" not in profile.maintenance_caps
    or (requirement is not None and requirement.bypass == "forbidden")
  ):
    raise ValueError(f"{profile.level} disabled Cache path has no permitted bypass")


def _gathers(program: ExecTileProgram) -> tuple[ExecTileGatherDesc, ...]:
  return tuple(
    gather
    for descriptor in program.descriptors.values()
    if isinstance(gather := descriptor.params.get("gather"), ExecTileGatherDesc)
  )


def _cache_path_requirements(
  gathers: Sequence[ExecTileGatherDesc], profile: MemoryProfile
) -> tuple[bool, bool, bool]:
  """Return ``(cache_used, bypass_needed, cache_required)`` for one path.

  Plan §2: a program containing a Gather uses any level whose profile
  enables a cache; a disabled level is a bypass path.  Scatter never
  touches the cache and needs no bypass, so it does not enter here.
  """
  if not gathers:
    return False, False, False
  if profile.cache_bytes > 0:
    return True, False, False
  return False, True, False


def check_program_capabilities(
  program: ExecTileProgram,
  tile_contract: TileResources,
  context_contract: ContextResources,
  l1: MemoryProfile,
  l2: MemoryProfile,
  *,
  l1_mshr_entries: int,
) -> None:
  """Validate the complete program for one actual L1/L2 combination."""
  gathers = _gathers(program)
  l1_used, l1_bypass, l1_required = _cache_path_requirements(gathers, l1)
  l2_used, l2_bypass, l2_required = _cache_path_requirements(gathers, l2)
  _check_requirement(
    tile_contract.l1_cache, l1, used=l1_used, bypass_needed=l1_bypass, required_needed=l1_required
  )
  # A Tile declaration and its parent Context declaration are independent
  # hard lower bounds; neither can weaken the other.
  _check_requirement(
    tile_contract.l2_cache, l2, used=l2_used, bypass_needed=l2_bypass, required_needed=l2_required
  )
  _check_requirement(
    context_contract.l2_cache, l2, used=l2_used, bypass_needed=l2_bypass, required_needed=l2_required
  )
  del l1_mshr_entries


def _tile_lifetimes(program: ExecTileProgram) -> dict[str, tuple[int, int]]:
  starts: dict[str, int] = {}
  ends: dict[str, int] = {}
  for index, inst in enumerate(program.insts):
    if inst.op is ExecTileOp.ALLOC_L1:
      starts[inst.args[0]] = index
    elif inst.op is ExecTileOp.FREE_L1:
      ends[inst.args[0]] = index
  expected = {buffer.name for buffer in program.l1_buffers}
  if set(starts) != expected or not set(ends) <= expected:
    raise ValueError(f"program {program.name!r} has incomplete L1 allocation lifetimes")
  return {name: (starts[name], ends.get(name, len(program.insts))) for name in expected}


def _segments_overlap(left: BufferLayout, right: BufferLayout) -> bool:
  for left_bank, left_start, left_size in left.segments():
    for right_bank, right_start, right_size in right.segments():
      if (
        left_bank == right_bank
        and left_start < right_start + right_size
        and right_start < left_start + left_size
      ):
        return True
  return False


def _prepare_program(
  program: ExecTileProgram, registry: ProfileRegistry, slot_capacity: int
) -> ExecTileProgram:
  contract = program.resource_contract
  if not isinstance(contract, TileResources):
    raise ValueError(f"Tile Program {program.name!r} requires resource_contract")
  layout = allowed_layouts(
    program.l1_buffers,
    registry,
    "l1",
    contract.allowed_profiles,
    contract.tile_l1_spm_bytes_per_context,
    lifetimes=_tile_lifetimes(program),
    slot_capacity=slot_capacity,
  )
  return replace(program, layout=layout)


def _logical_task_count(task: ExecTileGroupTask) -> int:
  result = 0
  for action in task.actions:
    if action.op is ExecGroupActionOp.DISPATCH_ROLE:
      binding = task.role_bindings[action.args[0].role_id]
      domain = binding.task_domain
      if domain is None:
        raise ValueError(f"Context {task.name!r} dispatch lacks a task range")
      result += domain.to_task - domain.from_task
  return result


_REGISTRATION_FENCES = frozenset(
  {
    ExecGroupActionOp.WAIT_EVENT,
    ExecGroupActionOp.BARRIER_GROUP,
    ExecGroupActionOp.PROFILE_RECONFIG,
    ExecGroupActionOp.MEMORY_MAINTENANCE,
  }
)


def _event_resource_requirements(task: ExecTileGroupTask) -> tuple[dict[str, int], int]:
  """Return exact future-use counts and a conservative live-slot frontier.

  Outputs are allocated atomically when their action registers, before any
  dependency can retire.  Outside a completed registration fence no issue or
  completion order is assumed.  A completed ordinary wait/control fence proves
  only its dependency ancestry terminal; a completed Group barrier proves the
  whole registered prefix terminal.  An Event Table slot is reclaimed only
  when its event is terminal and every compiled consumer action is proven
  issued, which is the point where runtime drops both its registered reference
  and future-use count.
  """

  producers: dict[str, int] = {}
  consumers: dict[str, set[int]] = {}
  ancestors: dict[str, frozenset[str]] = {}
  outputs_by_action: list[tuple[str, ...]] = []
  uses: dict[str, int] = {}
  for index, action in enumerate(task.actions):
    if len(action.dependencies) != len(set(action.dependencies)):
      raise ValueError(f"action {action.instruction_id!r} has duplicate event dependencies")
    unknown = set(action.dependencies) - set(producers)
    if unknown:
      raise ValueError(
        f"action {action.instruction_id!r} has unknown or forward event dependencies {sorted(unknown)}"
      )
    for event in action.dependencies:
      uses[event] += 1
      consumers[event].add(index)

    outputs = action.output_events
    if len(outputs) != len(set(outputs)):
      raise ValueError(f"action {action.instruction_id!r} produces duplicate events")
    base = set(action.dependencies)
    for event in action.dependencies:
      base.update(ancestors[event])
    for event in outputs:
      if not isinstance(event, str) or not event:
        raise ValueError(f"action {action.instruction_id!r} produces an invalid event")
      if event in producers:
        raise ValueError(f"event {event!r} has multiple producers")
      event_ancestors = set(base)
      if action.op is ExecGroupActionOp.DISPATCH_ROLE and event == action.dst:
        request = action.args[0]
        event_ancestors.update(
          phase for phase in (request.input_released_event, request.output_ready_event) if phase
        )
      producers[event] = index
      consumers[event] = set()
      ancestors[event] = frozenset(event_ancestors)
      uses[event] = 0
    outputs_by_action.append(outputs)

  live: set[str] = set()
  terminal: set[str] = set()
  issued_actions: set[int] = set()
  peak = 0
  for index, action in enumerate(task.actions):
    live.update(outputs_by_action[index])
    peak = max(peak, len(live))
    if action.op not in _REGISTRATION_FENCES:
      continue

    issued_actions.add(index)
    if action.op is ExecGroupActionOp.BARRIER_GROUP:
      issued_actions.update(range(index))
      terminal.update(event for prefix_outputs in outputs_by_action[:index] for event in prefix_outputs)
    else:
      completed = set(action.dependencies)
      for event in action.dependencies:
        completed.update(ancestors[event])
      terminal.update(completed)
      issued_actions.update(producers[event] for event in completed)

    live.difference_update(
      {event for event in live if event in terminal and consumers[event] <= issued_actions}
    )

  return uses, peak


def _resource_budget(task: ExecTileGroupTask) -> ResourceBudget:
  active_grids: set[str] = set()
  maximum_grids = 0
  _, event_frontier = _event_resource_requirements(task)
  engine_kinds: set[str] = set()
  ancestors: dict[str, frozenset[str]] = {}
  for action in task.actions:
    completed = _event_ancestors(action.dependencies, ancestors)
    active_grids.difference_update(completed)
    if action.op is ExecGroupActionOp.DISPATCH_ROLE:
      if action.dst is None:
        raise ValueError("dispatch lacks grid_done")
      active_grids.add(action.dst)
      engine_kinds.add("dispatch")
      maximum_grids = max(maximum_grids, len(active_grids))
    elif action.op is ExecGroupActionOp.DMA_PREFETCH:
      engine_kinds.add("mfe_load")
    elif action.op is ExecGroupActionOp.DMA_STORE:
      engine_kinds.add("mfe_store")
    elif action.op is ExecGroupActionOp.BARRIER_GROUP:
      active_grids.clear()
    base = set(completed)
    for event in action.output_events:
      event_ancestors = set(base)
      if action.op is ExecGroupActionOp.DISPATCH_ROLE and event == action.dst:
        request = action.args[0]
        event_ancestors.update(
          signal for signal in (request.input_released_event, request.output_ready_event) if signal
        )
      ancestors[event] = frozenset(event_ancestors)
  frame_slots = 0
  for role in task.role_bindings.values():
    layout = role.tile_program.layout
    if not isinstance(layout, ArenaLayout):
      raise ValueError(f"Tile Program {role.tile_program.name!r} is missing a prepared layout")
    for item in layout.buffer_layouts:
      frame_slots = max(frame_slots, item.slot_id + 1)
  return ResourceBudget(
    maximum_grids,
    event_frontier,
    frame_slots,
    dict.fromkeys(sorted(engine_kinds), 1) | {"inflight": int(bool(engine_kinds))},
  )


def prepare_resources(
  entry: ExecModel | ExecTileGroupTask, registry: ProfileRegistry, hw, sim
) -> tuple[ExecModel | ExecTileGroupTask, dict[str, ResourceBudget], dict[str, object]]:
  """Attach layouts and prove C01/C02/C04 contracts for every call binding."""
  slot_capacity = hw.frame_slot_capacity
  tasks = entry.tasks if isinstance(entry, ExecModel) else {entry.binding_id: entry}
  prepared: dict[str, ExecTileGroupTask] = {}
  budgets: dict[str, ResourceBudget] = {}
  effects: dict[str, object] = {}
  for binding_id, task in tasks.items():
    contract = task.resource_contract
    if not isinstance(contract, ContextResources):
      raise ValueError(f"Context {task.name!r} requires resource_contract")
    if contract.requested_contexts_per_tile > sim.context_count:
      raise ValueError(
        f"Context {task.name!r} requests R={contract.requested_contexts_per_tile}, "
        f"target provides {sim.context_count} Tile contexts"
      )
    logical_tasks = _logical_task_count(task)
    if logical_tasks != contract.logical_tasks:
      raise ValueError(
        f"Context {task.name!r} declares logical_tasks={contract.logical_tasks}, "
        f"executable dispatches use {logical_tasks}"
      )
    # L2 layout is permanently no-rebind: every local buffer owns an
    # independent, non-overlapping stripe-rounded padded span for the whole
    # Context.  When that raises the high-water above the source resource
    # declaration, the executable contract is corrected to the existing
    # conservative no-reuse result before the normal per-profile validation.
    conservative_bytes = [
      (buffer.bytes, buffer.alignment) for buffer in task.l2_buffers
    ]
    required_reserved = max(
      (
        conservative_arena_bytes(conservative_bytes, registry.profile("l2", mode))
        for mode in contract.allowed_profiles
      ),
      default=0,
    )
    if required_reserved > contract.l2_spm_bytes:
      contract = replace(contract, l2_spm_bytes=required_reserved)
      task = replace(task, resource_contract=contract)
    l2_layout = allowed_layouts(
      task.l2_buffers,
      registry,
      "l2",
      contract.allowed_profiles,
      contract.l2_spm_bytes,
      lifetimes=None,
      slot_capacity=max(1, len(task.l2_buffers)),
    )
    for l2_mode in contract.allowed_profiles:
      _check_requirement(
        contract.l2_cache,
        registry.profile("l2", l2_mode),
        used=False,
        bypass_needed=False,
        required_needed=False,
      )
    program_cache: dict[int, ExecTileProgram] = {}
    roles: dict[int, ExecTileRoleBinding] = {}
    envelopes: dict[int, list[int]] = {}
    child_modes: set[int] = set()
    l1_layout_hashes: set[str] = set()
    for role_id, role in task.role_bindings.items():
      program = program_cache.get(id(role.tile_program))
      if program is None:
        program = _prepare_program(role.tile_program, registry, slot_capacity)
        program_cache[id(role.tile_program)] = program
      tile_contract = program.resource_contract
      tile_layout = program.layout
      if not isinstance(tile_contract, TileResources) or not isinstance(tile_layout, ArenaLayout):
        raise ValueError(f"Tile Program {program.name!r} has incomplete prepared resources")
      child_modes.update(tile_contract.allowed_profiles)
      l1_layout_hashes.add(tile_layout.layout_hash)
      for mode in tile_contract.allowed_profiles:
        profile = registry.profile("l1", mode)
        envelope = envelopes.setdefault(mode, [0] * profile.banks)
        if len(tile_layout.per_bank_bytes) != profile.banks:
          raise ValueError("L1 child layout bank count differs from target")
        for index, amount in enumerate(tile_layout.per_bank_bytes):
          envelope[index] = max(envelope[index], amount)
      # Validate every declared cross-layer combination, not just the first
      # Gather path or the requested baseline.
      for l1_mode in tile_contract.allowed_profiles:
        for l2_mode in contract.allowed_profiles:
          check_program_capabilities(
            program,
            tile_contract,
            contract,
            registry.profile("l1", l1_mode),
            registry.profile("l2", l2_mode),
            l1_mshr_entries=hw.l1_mshr_entries,
          )
      roles[role_id] = replace(role, tile_program=program)
    for mode, envelope in envelopes.items():
      profile = registry.profile("l1", mode)
      if len(envelope) != profile.banks:
        raise ValueError("L1 envelope bank count differs from target")
      for bank, amount in enumerate(envelope):
        demand = contract.requested_contexts_per_tile * amount
        if demand > profile.user_spm_per_bank:
          raise ValueError(
            f"permanent capacity: Context {task.name!r} R x child bank {bank} needs {demand} bytes "
            f"under L1 mode {mode}, only {profile.user_spm_per_bank} available"
          )
    task = replace(task, role_bindings=roles, layout=l2_layout)
    budget = _resource_budget(task)
    if budget.max_live_grids > sim.group.dispatch_capacity:
      raise ValueError(
        f"Context {task.name!r} needs {budget.max_live_grids} live Grid routes, "
        f"target capacity is {sim.group.dispatch_capacity}"
      )
    if budget.frame_slots > slot_capacity:
      raise ValueError(
        f"Context {task.name!r} needs {budget.frame_slots} Frame Slots, target provides {slot_capacity}"
      )
    prepared[binding_id] = task
    budgets[binding_id] = budget
    effects[binding_id] = {
      "logical_tasks": logical_tasks,
      "l2_layout_hash": l2_layout.layout_hash,
      "l1_layout_hashes": tuple(sorted(l1_layout_hashes)),
      "l1_modes": tuple(sorted(child_modes)),
      "l2_modes": tuple(contract.allowed_profiles),
    }
  _prove_shared_l2_capacity(prepared, registry)
  if isinstance(entry, ExecModel):
    return replace(entry, tasks=prepared), budgets, effects
  return prepared[entry.binding_id], budgets, effects


def _prove_shared_l2_capacity(
  tasks: Mapping[str, ExecTileGroupTask], registry: ProfileRegistry
) -> None:
  """Prove each reader's local reservation plus shared padded spans fit per bank."""
  for consumer in tasks.values():
    if not consumer.shared_inputs:
      continue
    consumer_contract = consumer.resource_contract
    consumer_layout = consumer.layout
    if not isinstance(consumer_contract, ContextResources) or not isinstance(
      consumer_layout, ArenaLayout
    ):
      raise ValueError(f"Context {consumer.name!r} has incomplete shared L2 resource metadata")

    imported: dict[tuple[str, str], tuple[ExecTileGroupTask, BufferLayout]] = {}
    for shared_input in consumer.shared_inputs:
      key = (shared_input.producer_binding_id, shared_input.producer_slot)
      producer = tasks.get(shared_input.producer_binding_id)
      if producer is None:
        raise ValueError(
          f"shared input references unknown producer {shared_input.producer_binding_id!r}"
        )
      producer_contract = producer.resource_contract
      producer_layout = producer.layout
      if not isinstance(producer_contract, ContextResources) or not isinstance(
        producer_layout, ArenaLayout
      ):
        raise ValueError(f"shared producer {producer.name!r} has incomplete L2 resource metadata")
      producer_buffer = next(
        (
          buffer
          for buffer in producer.l2_buffers
          if buffer.slot == shared_input.producer_slot and buffer.sharing == "readonly"
        ),
        None,
      )
      if producer_buffer is None or (
        producer_buffer.dims,
        producer_buffer.dtype,
        producer_buffer.element_bytes,
        producer_buffer.bytes,
      ) != (
        shared_input.dims,
        shared_input.dtype,
        shared_input.element_bytes,
        shared_input.bytes,
      ):
        raise ValueError("shared input descriptor does not match its readonly producer buffer")
      allocation = next(
        (item for item in producer_layout.buffer_layouts if item.buffer_id == shared_input.producer_slot),
        None,
      )
      if allocation is None or allocation.logical_bytes != producer_buffer.bytes:
        raise ValueError(f"shared producer layout omits slot {shared_input.producer_slot!r}")
      imported.setdefault(key, (producer, allocation))

    for mode in consumer_contract.allowed_profiles:
      profile = registry.profile("l2", mode)
      if len(consumer_layout.per_bank_bytes) != profile.banks:
        raise ValueError("consumer L2 layout bank count differs from its shared profile")
      shared_per_bank = 0
      for _producer, allocation in imported.values():
        if allocation.banks != profile.banks:
          raise ValueError("shared producer and consumer profile have different L2 bank counts")
        round_bytes = uint64(
          allocation.stripe_bytes * allocation.banks, "shared producer stripe round", positive=True
        )
        padded_bytes = uint64(
          ((allocation.logical_bytes + round_bytes - 1) // round_bytes) * round_bytes,
          "shared producer padded span",
        )
        shared_per_bank += padded_bytes // allocation.banks
      for bank, local_bytes in enumerate(consumer_layout.per_bank_bytes):
        demand = uint64(local_bytes + shared_per_bank, "shared plus local L2 bank demand")
        if profile.system_reserved_spm_per_bank + demand > profile.spm_bytes_per_bank:
          raise ValueError(
            f"permanent capacity: Context {consumer.name!r} L2 bank {bank} needs {demand} "
            f"bytes for its local arena and shared inputs under mode {mode}, "
            f"but only {profile.user_spm_per_bank} user bytes are available"
          )


def finalize_event_resources(
  entry: ExecModel | ExecTileGroupTask, budgets: Mapping[str, ResourceBudget], sim
) -> tuple[ExecModel | ExecTileGroupTask, dict[str, ResourceBudget]]:
  """Seal event metadata after profile and maintenance action generation."""

  tasks = entry.tasks if isinstance(entry, ExecModel) else {entry.binding_id: entry}
  if set(tasks) != set(budgets):
    raise ValueError("event finalization requires one resource budget per call binding")
  finalized_tasks: dict[str, ExecTileGroupTask] = {}
  finalized_budgets: dict[str, ResourceBudget] = {}
  for binding_id, task in tasks.items():
    event_uses, event_frontier = _event_resource_requirements(task)
    if event_frontier > sim.group.event_capacity:
      raise ValueError(
        f"Context {task.name!r} event frontier {event_frontier} exceeds {sim.group.event_capacity}"
      )
    finalized_tasks[binding_id] = replace(task, event_uses=event_uses)
    finalized_budgets[binding_id] = replace(budgets[binding_id], event_frontier=event_frontier)
  if isinstance(entry, ExecModel):
    return replace(entry, tasks=finalized_tasks), finalized_budgets
  return finalized_tasks[entry.binding_id], finalized_budgets
