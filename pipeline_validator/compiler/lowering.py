"""Pure lowering from verified source IR to immutable executable templates.

The lowering owns source identities, call-site specialization, memory effects,
and the exact source-level lifetime operations.  It never mutates either the
xDSL source or a previously published executable object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from xdsl.ir import Attribute, SSAValue
from xdsl.utils.exceptions import VerifyException

from ..dialects.elenor import (
  DTYPE_BYTES,
  NestAllocOp,
  NestAwaitOp,
  NestBarrierOp,
  NestBuffer,
  NestContextOp,
  NestDispatchOp,
  NestDMAStoreOp,
  NestEvent,
  NestGlobalMemref,
  NestGlobalView,
  NestL2View,
  NestPrefetchOp,
  NestPublishOp,
  NestReleaseOp,
  NestReturnOp,
  NestSubviewOp,
  NestTaskRangeOp,
  NexusAwaitOp,
  NexusEvent,
  NexusHostCallOp,
  NexusProgramOp,
  NexusReturnOp,
  NexusSharedRefOp,
  NexusSubmitContextOp,
  TileAllocOp,
  TileAwaitOp,
  TileBoaOp,
  TileEvent,
  TileEvuOp,
  TileFreeOp,
  TileGatherOp,
  TileL1Buffer,
  TileLoadOp,
  TilePowOp,
  TileProgramDefOp,
  TileReturnOp,
  TileScatterOp,
  TileSignalOp,
  TileStoreOp,
  TileSubviewOp,
  _int_list,
)
from ..execution_ir import (
  ExecDeviceOp,
  ExecDispatchRequest,
  ExecEngineDesc,
  ExecGlobalInput,
  ExecGroupAction,
  ExecGroupActionOp,
  ExecHostAccess,
  ExecHostCall,
  ExecIndexedMap,
  ExecL1Buffer,
  ExecL2Buffer,
  ExecMemoryView,
  ExecModel,
  ExecPublishRequest,
  ExecReleaseRequest,
  ExecSharedInput,
  ExecSignalPolicy,
  ExecTaskDomain,
  ExecTileFormal,
  ExecTileGatherDesc,
  ExecTileGroupTask,
  ExecTileInst,
  ExecTileOp,
  ExecTileProgram,
  ExecTileRoleBinding,
  ExecTileScatterDesc,
  ExecTransfer,
)
from ..profiles import SourceRef
from ..workload_ir import (
  _body_ops,
  _view_bytes,
  _view_offset_bytes,
  effective_submit_dependencies,
  verify_workload_ir,
)


def _ref(source_name: str, symbol: str, index: int, op) -> SourceRef:
  return SourceRef(source_name, symbol, index, op.name)


def _iid(binding_id: str, scope: str, index: int, suffix: str = "") -> str:
  tail = f":{suffix}" if suffix else ""
  return f"{binding_id}:{scope}:{index}{tail}"


def lower_workload_ir(module, *, source_name: str, binding_id: str | None = None) -> ExecTileGroupTask:
  """Lower the single standalone Context in *module*.

  This is intentionally an internal compiler API.  Public callers use
  :func:`pipeline_validator.compiler.compile_program`.
  """
  context = verify_workload_ir(module)
  if not isinstance(context, NestContextOp):
    raise VerifyException("standalone lowering requires a nest.context entry")
  identity = binding_id or f"standalone:{context.sym_name.data}"
  return _lower_context(module, context, source_name, identity)


def lower_model_ir(module, *, source_name: str) -> ExecModel:
  """Lower and independently specialize every model submit call site."""
  program = verify_workload_ir(module)
  if not isinstance(program, NexusProgramOp):
    raise VerifyException("model lowering requires a nexus.program entry")
  program_args = list(program.body.block.args)
  inputs = tuple(_global_input(arg, index) for index, arg in enumerate(program_args))
  contexts = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, NestContextOp)}
  model_ops = tuple(_body_ops(program))
  symbol = program.sym_name.data
  binding_ids = {
    op: f"call:{symbol}:{index}:{_event_tag(op.result.type)}"
    for index, op in enumerate(model_ops)
    if isinstance(op, NexusSubmitContextOp)
  }
  tasks: dict[str, ExecTileGroupTask] = {}
  pins: dict[str, int | None] = {}
  body: list[ExecDeviceOp] = []
  for index, body_op in enumerate(model_ops):
    source = _ref(source_name, symbol, index, body_op)
    if isinstance(body_op, NexusSubmitContextOp):
      context_name = body_op.context_sym.data
      context = contexts.get(context_name)
      if context is None:
        raise VerifyException(f"submit references unknown Context '@{context_name}'")
      context_args = tuple(context.body.block.args)
      global_formals = tuple(arg for arg in context_args if isinstance(arg.type, NestGlobalMemref))
      import_formals = tuple(arg for arg in context_args if isinstance(arg.type, NestBuffer))
      global_actuals = tuple(
        actual for actual in body_op.actuals if isinstance(actual.type, NestGlobalMemref)
      )
      shared_actuals = tuple(actual for actual in body_op.actuals if isinstance(actual.type, NestBuffer))
      if len(global_actuals) != len(global_formals) or len(shared_actuals) != len(import_formals):
        raise VerifyException(f"submit '@{context_name}' actuals do not match global/shared formals")
      actual_indices = tuple(_block_arg_index(actual, program_args) for actual in global_actuals)
      aliases = {
        _global_input(formal, formal_index).name: inputs[actual_index].name
        for formal_index, (formal, actual_index) in enumerate(zip(global_formals, actual_indices))
      }
      shared_inputs: list[ExecSharedInput] = []
      for formal, actual in zip(import_formals, shared_actuals):
        reference = actual.owner
        if not isinstance(reference, NexusSharedRefOp):
          raise VerifyException("shared Context actual must come from nexus.shared.ref")
        producer_submit = reference.producer.owner
        if not isinstance(producer_submit, NexusSubmitContextOp):
          raise VerifyException("shared reference producer must be a submit_context call site")
        producer_binding_id = binding_ids.get(producer_submit)
        if producer_binding_id is None:
          raise VerifyException("shared reference producer submit is not in this model")
        producer_context = contexts.get(producer_submit.context_sym.data)
        if producer_context is None:
          raise VerifyException("shared reference producer Context is unknown")
        producer_slot = reference.slot.data
        producer_buffer = next(
          (
            op
            for op in _body_ops(producer_context)
            if isinstance(op, NestAllocOp) and op.slot.data == producer_slot
          ),
          None,
        )
        if producer_buffer is None or producer_buffer.sharing.data != "readonly":
          raise VerifyException("shared reference does not name a readonly producer allocation")
        dims, dtype = _dims_dtype(formal.type)
        producer_dims, producer_dtype = _dims_dtype(producer_buffer.result.type)
        if (dims, dtype) != (producer_dims, producer_dtype):
          raise VerifyException("shared reference shape/dtype differs from producer allocation")
        shared_inputs.append(
          ExecSharedInput(
            formal.name_hint or "",
            dims,
            dtype,
            DTYPE_BYTES[dtype],
            _view_bytes(dims, dtype),
            producer_binding_id,
            producer_slot,
          )
        )
      binding_id = binding_ids[body_op]
      task = _lower_context(
        module, context, source_name, binding_id, aliases, tuple(shared_inputs)
      )
      tasks[binding_id] = task
      pins[binding_id] = None if context.context_id is None else int(context.context_id.value.data)
      body.append(
        ExecDeviceOp(
          "submit",
          ctx_name=context_name,
          event_tag=_event_tag(body_op.result.type),
          actual_inputs=actual_indices,
          dependencies=tuple(
            _event_tag(dep.type) for dep in effective_submit_dependencies(body_op)
          ),
          callsite_id=f"{symbol}:call:{index}",
          binding_id=binding_id,
          source_ref=source,
          instruction_id=_iid(binding_id, "device", index, "submit"),
        )
      )
    elif isinstance(body_op, NexusSharedRefOp):
      continue
    elif isinstance(body_op, NexusHostCallOp):
      accesses: list[ExecHostAccess] = []
      for binding, (offset, byte_count, mode) in zip(body_op.bindings, body_op.access_list):
        actual_index = _block_arg_index(binding, program_args)
        if actual_index is None:
          raise VerifyException("nexus.host.call.async binding must be a nexus.program input")
        accesses.append(ExecHostAccess(actual_index, offset, byte_count, mode))
      body.append(
        ExecDeviceOp(
          "host_call",
          event_tag=_event_tag(body_op.result.type),
          command=ExecHostCall(
            body_op.host_routine.data,
            tuple(accesses),
            tuple(body_op.scope_list),
          ),
          dependencies=tuple(_event_tag(dep.type) for dep in body_op.depends_on),
          callsite_id=f"{symbol}:host:{index}",
          source_ref=source,
          instruction_id=f"model:{symbol}:device:{index}:host_call",
        )
      )
    elif isinstance(body_op, NexusAwaitOp):
      for operand_index, event in enumerate(body_op.events):
        body.append(
          ExecDeviceOp(
            "await",
            event_tag=_event_tag(event.type),
            source_ref=source,
            instruction_id=f"model:{symbol}:device:{index}:await:{operand_index}",
          )
        )
    elif isinstance(body_op, NexusReturnOp):
      body.append(
        ExecDeviceOp("return", source_ref=source, instruction_id=f"model:{symbol}:device:{index}:return")
      )
    else:
      raise VerifyException(f"unexpected nexus.program body op '{body_op.name}'")
  return ExecModel(symbol, tasks, pins, tuple(body), inputs)


def _lower_context(
  module,
  context: NestContextOp,
  source_name: str,
  binding_id: str,
  global_aliases: Mapping[str, str] | None = None,
  shared_inputs: tuple[ExecSharedInput, ...] = (),
) -> ExecTileGroupTask:
  programs = {
    op.sym_name.data: _lower_program(op, source_name, binding_id)
    for op in module.body.block.ops
    if isinstance(op, TileProgramDefOp)
  }
  context_args = tuple(context.body.block.args)
  global_args = tuple(arg for arg in context_args if isinstance(arg.type, NestGlobalMemref))
  import_args = tuple(arg for arg in context_args if isinstance(arg.type, NestBuffer))
  if len(import_args) != len(shared_inputs):
    raise VerifyException("shared input metadata does not match Context L2 formals")
  global_inputs = tuple(_global_input(arg, index) for index, arg in enumerate(global_args))
  formal_names: dict[SSAValue, str] = {
    arg: item.name for arg, item in zip(global_args, global_inputs)
  }
  formal_dims: dict[SSAValue, tuple[int, ...]] = {
    arg: item.dims for arg, item in zip(global_args, global_inputs)
  }
  objects: dict[SSAValue, ExecMemoryView | ExecL2Buffer] = {}
  task_domains: dict[SSAValue, ExecTaskDomain] = {}
  l2_buffers: list[ExecL2Buffer] = []
  bind_events: dict[str, str] = {}
  layout_indices: dict[str, int] = {}
  import_indices: dict[str, int] = {}
  actions: list[ExecGroupAction] = []
  for shared_index, (formal, shared_input) in enumerate(zip(import_args, shared_inputs)):
    dims, dtype = _dims_dtype(formal.type)
    slot = formal.name_hint or ""
    if (
      shared_input.slot != slot
      or shared_input.dims != dims
      or shared_input.dtype != dtype
      or shared_input.element_bytes != DTYPE_BYTES[dtype]
      or shared_input.bytes != _view_bytes(dims, dtype)
    ):
      raise VerifyException("shared input metadata differs from its Context formal")
    descriptor = ExecL2Buffer(
      slot, dims, dtype, "in", DTYPE_BYTES[dtype], 1, _view_bytes(dims, dtype)
    )
    objects[formal] = descriptor
    bind_source = SourceRef(
      source_name,
      context.sym_name.data,
      shared_index,
      "nest.context.import",
      "shared_l2_lowering",
      f"bind imported readonly L2 formal {slot!r}",
    )
    bind_event = f"__{binding_id}:l2.import.bind:{shared_index}"
    if slot in bind_events:
      raise VerifyException(f"duplicate L2 bind slot {slot!r}")
    bind_events[slot] = bind_event
    import_indices[slot] = shared_index
    actions.append(
      ExecGroupAction(
        ExecGroupActionOp.BIND_L2_IMPORT,
        args=(slot, shared_index),
        dst=bind_event,
        source_ref=bind_source,
        instruction_id=_iid(binding_id, "group", shared_index, "bind_l2_import"),
      )
    )
  body_ops = tuple(_body_ops(context))
  dispatch_ordinals: dict[NestDispatchOp, int] = {}
  ins_consumers: dict[SSAValue, list[int]] = {}
  outs_producers: dict[SSAValue, list[int]] = {}
  for body_op in body_ops:
    if isinstance(body_op, NestDispatchOp):
      ordinal = len(dispatch_ordinals)
      dispatch_ordinals[body_op] = ordinal
      for actual in body_op.ins:
        ins_consumers.setdefault(actual, []).append(ordinal)
      for actual in body_op.outs:
        outs_producers.setdefault(actual, []).append(ordinal)

  role_bindings: dict[int, ExecTileRoleBinding] = {}
  placement = int(context.placement.value.data)
  symbol = context.sym_name.data
  for index, body_op in enumerate(body_ops):
    source = _ref(source_name, symbol, index, body_op)
    if isinstance(body_op, NestAllocOp):
      dims, dtype = _dims_dtype(body_op.result.type)
      alignment = int(body_op.alignment.value.data) if body_op.alignment is not None else 1
      buffer = ExecL2Buffer(
        body_op.slot.data,
        dims,
        dtype,
        body_op.role.data,
        DTYPE_BYTES[dtype],
        alignment,
        _view_bytes(dims, dtype),
        body_op.sharing.data,
      )
      layout_index = len(l2_buffers)
      l2_buffers.append(buffer)
      objects[body_op.result] = buffer
      layout_indices[buffer.slot] = layout_index
      event = f"__{binding_id}:l2.bind:{layout_index}"
      bind_events[buffer.slot] = event
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.BIND_L2_VIEW,
          args=(buffer.slot, layout_index),
          dst=event,
          source_ref=source,
          instruction_id=_iid(binding_id, "group", index, "bind_l2"),
        )
      )
    elif isinstance(body_op, NestTaskRangeOp):
      task_domains[body_op.result] = ExecTaskDomain(
        int(body_op.from_task.value.data), int(body_op.to_task.value.data)
      )
    elif isinstance(body_op, NestSubviewOp):
      formal_name = formal_names.get(body_op.src)
      backing_dims = formal_dims.get(body_op.src)
      if formal_name is None or backing_dims is None:
        raise VerifyException("nest.subview source has no lowered global formal")
      dims = tuple(_int_list(body_op.sizes))
      dtype = body_op.result.type.dtype.data
      objects[body_op.result] = ExecMemoryView(
        "global",
        f"global:{formal_name}",
        backing_dims,
        dims,
        tuple(_int_list(body_op.offsets)),
        tuple(_int_list(body_op.strides)),
        dtype,
        DTYPE_BYTES[dtype],
        _view_bytes(dims, dtype),
      )
    elif isinstance(body_op, NestPrefetchOp):
      src = _memory_view(objects, body_op.src, body_op.name)
      dst_buffer = _l2_buffer(objects, body_op.dst, body_op.name)
      deps = (*tuple(_event_tag(dep.type) for dep in body_op.depends_on), bind_events[dst_buffer.slot])
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.DMA_PREFETCH,
          args=(
            f"gdma_prefetch:{dst_buffer.slot}",
            ExecTransfer(src, _l2_buffer_view(dst_buffer), src.bytes),
          ),
          dst=_event_tag(body_op.result.type),
          dependencies=tuple(dict.fromkeys(deps)),
          writes=(dst_buffer.slot,),
          source_ref=source,
          instruction_id=_iid(binding_id, "group", index, "prefetch"),
        )
      )
    elif isinstance(body_op, NestDMAStoreOp):
      src_buffer = _l2_buffer(objects, body_op.src, body_op.name)
      dst = _memory_view(objects, body_op.dst, body_op.name)
      deps = (*tuple(_event_tag(dep.type) for dep in body_op.depends_on), bind_events[src_buffer.slot])
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.DMA_STORE,
          args=(f"gdma_store:{src_buffer.slot}", ExecTransfer(_l2_buffer_view(src_buffer), dst, dst.bytes)),
          dst=_event_tag(body_op.result.type),
          dependencies=tuple(dict.fromkeys(deps)),
          reads=(src_buffer.slot,),
          source_ref=source,
          instruction_id=_iid(binding_id, "group", index, "store"),
        )
      )
    elif isinstance(body_op, NestDispatchOp):
      program_name = body_op.program.data
      task_domain = task_domains.get(body_op.tasks)
      if task_domain is None:
        raise VerifyException("dispatch task range has no lowered task domain")
      global_actuals = tuple(_memory_view(objects, actual, body_op.name) for actual in body_op.global_views)
      actuals = tuple(_l2_buffer(objects, actual, body_op.name).slot for actual in body_op.bindings)
      read_slots = {_l2_buffer(objects, actual, body_op.name).slot for actual in body_op.ins}
      write_slots = {_l2_buffer(objects, actual, body_op.name).slot for actual in body_op.outs}
      unique_actuals = tuple(dict.fromkeys(actuals))
      read_actuals = tuple(slot for slot in unique_actuals if slot in read_slots)
      write_actuals = tuple(slot for slot in unique_actuals if slot in write_slots)
      role_id = _register_role(
        role_bindings,
        programs,
        program_name,
        placement,
        None if body_op.context_id is None else int(body_op.context_id.value.data),
        task_domain,
        actuals,
        global_actuals,
        read_actuals,
        write_actuals,
      )
      dispatch_request = ExecDispatchRequest(
        role_id,
        dispatch_ordinals[body_op],
        ExecSignalPolicy(
          body_op.signal_policy.get("input_released"), body_op.signal_policy.get("output_ready")
        ),
        _event_tag(body_op.input_released.type),
        _event_tag(body_op.output_ready.type),
        int(body_op.l1_mode.value.data),
        int(body_op.l1_mode.value.data),
        binding_id,
      )
      dispatch_dependencies = [*(_event_tag(dep.type) for dep in body_op.depends_on)]
      dispatch_dependencies.extend(bind_events[slot] for slot in unique_actuals)
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.DISPATCH_ROLE,
          args=(dispatch_request,),
          dst=_event_tag(body_op.grid_done.type),
          dependencies=tuple(dict.fromkeys(dispatch_dependencies)),
          reads=read_actuals,
          writes=write_actuals,
          source_ref=source,
          instruction_id=_iid(binding_id, "group", index, "dispatch"),
        )
      )
    elif isinstance(body_op, NestPublishOp):
      buffer = _l2_buffer(objects, body_op.buffer, body_op.name)
      explicit = tuple(_event_tag(dep.type) for dep in body_op.depends_on)
      publish_request = ExecPublishRequest(
        buffer.slot,
        tuple(sorted(set(ins_consumers.get(body_op.buffer, ())))),
        tuple(sorted(set(outs_producers.get(body_op.buffer, ())))),
        explicit,
      )
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.PUBLISH_L2,
          args=(publish_request,),
          dst=_event_tag(body_op.result.type),
          dependencies=tuple(dict.fromkeys((*explicit, bind_events[buffer.slot]))),
          writes=(buffer.slot,),
          source_ref=source,
          instruction_id=_iid(binding_id, "group", index, "publish_l2"),
        )
      )
    elif isinstance(body_op, NestReleaseOp):
      buffer = _l2_buffer(objects, body_op.buffer, body_op.name)
      explicit = tuple(_event_tag(dep.type) for dep in body_op.depends_on)
      release_request = ExecReleaseRequest(
        buffer.slot,
        buffer.role,
        tuple(sorted(set(ins_consumers.get(body_op.buffer, ())))),
        tuple(sorted(set(outs_producers.get(body_op.buffer, ())))),
        explicit,
      )
      release_event = (
        f"__{binding_id}:l2.release:{layout_indices[buffer.slot]}"
        if buffer.slot in layout_indices
        else f"__{binding_id}:l2.import.release:{import_indices[buffer.slot]}"
      )
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.RELEASE_L2,
          args=(release_request,),
          dst=release_event,
          dependencies=tuple(dict.fromkeys((*explicit, bind_events[buffer.slot]))),
          source_ref=source,
          instruction_id=_iid(binding_id, "group", index, "release_l2"),
        )
      )
    elif isinstance(body_op, NestAwaitOp):
      for operand_index, operand in enumerate(body_op.events):
        event = _event_tag(operand.type)
        actions.append(
          ExecGroupAction(
            ExecGroupActionOp.WAIT_EVENT,
            args=(event,),
            dependencies=(event,),
            source_ref=source,
            instruction_id=_iid(binding_id, "group", index, f"await:{operand_index}"),
          )
        )
    elif isinstance(body_op, NestBarrierOp):
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.BARRIER_GROUP,
          source_ref=source,
          instruction_id=_iid(binding_id, "group", index, "barrier"),
        )
      )
    elif isinstance(body_op, NestReturnOp):
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.SIGNAL_EVENT,
          args=(context.completion_event.data,),
          source_ref=source,
          instruction_id=_iid(binding_id, "group", index, "return"),
        )
      )
    else:
      raise VerifyException(f"unexpected nest context body op '{body_op.name}'")

  actions = list(normalize_action_dependencies(actions, role_bindings, global_aliases))
  return ExecTileGroupTask(
    symbol,
    tuple(actions),
    (),
    role_bindings,
    context.completion_event.data,
    global_inputs,
    tuple(l2_buffers),
    context.resource_contract.to_contract(),
    None,
    binding_id,
    shared_inputs=shared_inputs,
  )


def _event_ancestors(
  dependencies: Sequence[str], ancestors: Mapping[str, frozenset[str]]
) -> frozenset[str]:
  result = set(dependencies)
  for event in dependencies:
    result.update(ancestors[event])
  return frozenset(result)


def _maximal_event_frontier(
  candidates: set[str], ancestors: Mapping[str, frozenset[str]]
) -> tuple[str, ...]:
  """Return the deterministic antichain whose completion covers *candidates*."""

  return tuple(
    sorted(
      event
      for event in candidates
      if not any(event != later and event in ancestors[later] for later in candidates)
    )
  )


def _append_unique_dependency(event: str, dependencies: list[str], seen: set[str]) -> None:
  if event not in seen:
    seen.add(event)
    dependencies.append(event)


def normalize_action_dependencies(
  actions: Sequence[ExecGroupAction],
  role_bindings: Mapping[int, ExecTileRoleBinding],
  global_aliases: Mapping[str, str] | None = None,
) -> tuple[ExecGroupAction, ...]:
  """Return a graph with complete hazards and a minimal retirement closure.

  Author dependencies and direct RAW/WAR/WAW/release edges remain explicit.
  A barrier or terminal return only acquires the maximal outstanding event
  antichain: every older completion is then covered transitively instead of
  being retained as an independent historical Event Table reference.
  """

  last_writer: dict[str, str] = {}
  readers: dict[str, set[str]] = {}
  produced: set[str] = set()
  outstanding: set[str] = set()
  ancestors: dict[str, frozenset[str]] = {}
  global_accesses: list[tuple[str, int, int, bool, str, str | None]] = []
  aliases = global_aliases or {}
  output: list[ExecGroupAction] = []
  for original in actions:
    dependencies = list(dict.fromkeys(original.dependencies))
    if original.op is ExecGroupActionOp.WAIT_EVENT:
      dependencies = list(dict.fromkeys(original.args))
    dependency_set = set(dependencies)

    for slot in (*original.reads, *original.writes):
      if slot in last_writer:
        _append_unique_dependency(last_writer[slot], dependencies, dependency_set)
    for slot in original.writes:
      for event in sorted(readers.get(slot, ())):
        _append_unique_dependency(event, dependencies, dependency_set)
    if original.op is ExecGroupActionOp.RELEASE_L2:
      slot = original.args[0].buffer_slot
      if slot in last_writer:
        _append_unique_dependency(last_writer[slot], dependencies, dependency_set)
      for event in sorted(readers.get(slot, ())):
        _append_unique_dependency(event, dependencies, dependency_set)
    if original.op in (ExecGroupActionOp.BARRIER_GROUP, ExecGroupActionOp.SIGNAL_EVENT):
      # These source operations have no dependency operands.  Any incoming
      # edges are from an earlier normalization pass and must be recomputed
      # after generated maintenance/profile fences change the frontier.
      dependencies = sorted(outstanding)
      dependency_set = set(dependencies)
    accesses: list[tuple[ExecMemoryView, bool, str | None]] = []
    if original.op in (ExecGroupActionOp.DMA_PREFETCH, ExecGroupActionOp.DMA_STORE):
      transfer = original.args[1]
      writing = original.op is ExecGroupActionOp.DMA_STORE
      accesses.append((transfer.dst if writing else transfer.src, writing, None))
    elif original.op is ExecGroupActionOp.DISPATCH_ROLE:
      binding = role_bindings[original.args[0].role_id]
      global_formals = [
        index for index, formal in enumerate(binding.tile_program.formals) if formal.space == "global"
      ]
      actuals = dict(zip(global_formals, binding.global_actuals))
      for descriptor in binding.tile_program.descriptors.values():
        gather = descriptor.params.get("gather")
        if isinstance(gather, ExecTileGatherDesc):
          accesses.append(
            (actuals[int(gather.source.base.removeprefix("formal:"))], False, gather.scope)
          )
        scatter = descriptor.params.get("scatter")
        if isinstance(scatter, ExecTileScatterDesc):
          accesses.append(
            (actuals[int(scatter.destination.base.removeprefix("formal:"))], True, scatter.scope)
          )
    for view, writing, scope in dict.fromkeys(accesses):
      name = aliases.get(view.base.removeprefix("global:"), view.base.removeprefix("global:"))
      start = _view_offset_bytes(view.offsets, view.backing_dims, view.element_bytes)
      end = start + view.bytes
      for prior_name, prior_start, prior_end, prior_write, event, prior_scope in global_accesses:
        # Different non-empty scopes are statically disjoint page owners
        # (plan §1), so only same-scope or unscoped pairs serialise.
        if scope and prior_scope and scope != prior_scope:
          continue
        if name == prior_name and start < prior_end and prior_start < end and (writing or prior_write):
          _append_unique_dependency(event, dependencies, dependency_set)
      if not original.dst:
        raise VerifyException(f"global access '{original.instruction_id}' has no completion event")
      global_accesses.append((name, start, end, writing, original.dst, scope))

    if not dependency_set <= produced:
      missing = sorted(dependency_set - produced)
      raise VerifyException(
        f"action '{original.instruction_id}' depends on unbound producer events: {missing}"
      )
    deps = tuple(dependencies)
    if original.op in (ExecGroupActionOp.PROFILE_RECONFIG, ExecGroupActionOp.MEMORY_MAINTENANCE):
      # Typed control descriptors carry the same ordered frontier and Loader
      # verifies byte-for-byte agreement with the action.
      deps = tuple(original.dependencies)
    args = original.args
    if original.op in (ExecGroupActionOp.RELEASE_L2, ExecGroupActionOp.PUBLISH_L2):
      args = (replace(original.args[0], dependency_events=deps),)
    action = replace(original, args=args, dependencies=deps)
    output.append(action)
    read_done = action.dst
    write_done = action.dst
    if action.op is ExecGroupActionOp.DISPATCH_ROLE:
      read_done = action.args[0].input_released_event
      write_done = action.args[0].output_ready_event
    for slot in action.writes:
      if not write_done:
        raise VerifyException(f"write of '{slot}' has no completion event")
      last_writer[slot] = write_done
      readers[slot] = set()
    for slot in action.reads:
      if not read_done:
        raise VerifyException(f"read of '{slot}' has no completion event")
      readers.setdefault(slot, set()).add(read_done)

    outputs = action.output_events
    if len(outputs) != len(set(outputs)):
      raise VerifyException(f"action '{action.instruction_id}' produces duplicate events")
    base = _event_ancestors(action.dependencies, ancestors)
    for event in outputs:
      if event in produced:
        raise VerifyException(f"duplicate action producer event '{event}'")
      event_ancestors = set(base)
      if action.op is ExecGroupActionOp.DISPATCH_ROLE and event == action.dst:
        request = action.args[0]
        event_ancestors.update(
          phase for phase in (request.input_released_event, request.output_ready_event) if phase
        )
      ancestors[event] = frozenset(event_ancestors)
      produced.add(event)
    outstanding.update(outputs)
    outstanding = set(_maximal_event_frontier(outstanding, ancestors))
    if action.op is ExecGroupActionOp.BARRIER_GROUP:
      outstanding.clear()
    elif action.op in (
      ExecGroupActionOp.WAIT_EVENT,
      ExecGroupActionOp.PROFILE_RECONFIG,
      ExecGroupActionOp.MEMORY_MAINTENANCE,
    ):
      outstanding.difference_update(_event_ancestors(action.dependencies, ancestors))
  return tuple(output)


def _register_role(
  bindings: dict[int, ExecTileRoleBinding],
  programs: Mapping[str, ExecTileProgram],
  program_name: str,
  tile_mask: int,
  context_id: int | None,
  task_domain: ExecTaskDomain,
  actuals: tuple[str, ...],
  global_actuals: tuple[ExecMemoryView, ...],
  read_actuals: tuple[str, ...],
  write_actuals: tuple[str, ...],
) -> int:
  for role_id, binding in bindings.items():
    if (
      binding.tile_program.name,
      binding.tile_mask,
      binding.context_id,
      binding.task_domain,
      binding.actuals,
      binding.global_actuals,
      binding.read_actuals,
      binding.write_actuals,
    ) == (
      program_name,
      tile_mask,
      context_id,
      task_domain,
      actuals,
      global_actuals,
      read_actuals,
      write_actuals,
    ):
      return role_id
  program = programs.get(program_name)
  if program is None:
    raise VerifyException(f"dispatch references unknown tile program '@{program_name}'")
  role_id = len(bindings)
  bindings[role_id] = ExecTileRoleBinding(
    role_id,
    tile_mask,
    program,
    None,
    None,
    context_id,
    task_domain,
    actuals,
    global_actuals,
    read_actuals,
    write_actuals,
  )
  return role_id


def _lower_engine_descriptor(
  op, name: str, objects: Mapping[SSAValue, ExecMemoryView]
) -> tuple[ExecEngineDesc, ExecTileOp]:
  if isinstance(op, (TileLoadOp, TileStoreOp)):
    src = _memory_view(objects, op.src, op.name)
    dst = _memory_view(objects, op.dst, op.name)
    return ExecEngineDesc(
      name, "MFE", "load" if isinstance(op, TileLoadOp) else "store", {}, ExecTransfer(src, dst, src.bytes)
    ), ExecTileOp.LAUNCH_MFE
  if isinstance(op, TileGatherOp):
    address_map = ExecIndexedMap(*op.address_map.values)
    gather = ExecTileGatherDesc(
      _memory_view(objects, op.source, op.name),
      _memory_view(objects, op.indices, op.name),
      _memory_view(objects, op.destination, op.name),
      address_map,
      int(op.window_entries.value.data),
      None if op.scope is None else op.scope.data,
    )
    return ExecEngineDesc(name, "MFE", "gather", {"gather": gather}), ExecTileOp.LAUNCH_GATHER
  if isinstance(op, TileScatterOp):
    address_map = ExecIndexedMap(*op.address_map.values)
    scatter = ExecTileScatterDesc(
      _memory_view(objects, op.source, op.name),
      _memory_view(objects, op.indices, op.name),
      _memory_view(objects, op.destination, op.name),
      address_map,
      int(op.window_entries.value.data),
      None if op.scope is None else op.scope.data,
    )
    return ExecEngineDesc(name, "MFE", "scatter", {"scatter": scatter}), ExecTileOp.LAUNCH_SCATTER
  if isinstance(op, TilePowOp):
    return ExecEngineDesc(
      name,
      "EVU",
      "pow",
      {
        "bytes": int(op.bytes_total.value.data),
        "exponent": int(op.exponent.value.data),
        "ops": int(op.pow_ops.value.data),
      },
    ), ExecTileOp.LAUNCH_EVU
  if isinstance(op, TileEvuOp):
    return ExecEngineDesc(
      name, "EVU", op.op_name.data, {"ops": int(op.evu_ops.value.data)}
    ), ExecTileOp.LAUNCH_EVU
  if isinstance(op, TileBoaOp):
    params = {
      "m": int(op.m.value.data),
      "n": int(op.n.value.data),
      "k": int(op.k.value.data),
      "ops": int(op.boa_ops.value.data),
    }
    if op.accumulate is not None:
      params["accumulate"] = True
    return ExecEngineDesc(name, "BOA", op.op_name.data, params), ExecTileOp.LAUNCH_BOA
  raise VerifyException(f"unexpected tile program body op '{op.name}'")


def _lower_program(op: TileProgramDefOp, source_name: str, binding_id: str) -> ExecTileProgram:
  descriptors: dict[str, ExecEngineDesc] = {}
  insts: list[ExecTileInst] = []
  l1_buffers: list[ExecL1Buffer] = []
  objects: dict[SSAValue, ExecMemoryView] = {}
  formals: list[ExecTileFormal] = []
  for index, arg in enumerate(op.body.block.args):
    if index == 0:
      formals.append(ExecTileFormal("task", (), ""))
      continue
    dims, dtype = _dims_dtype(arg.type)
    space = (
      "global" if isinstance(arg.type, NestGlobalView) else "l2" if isinstance(arg.type, NestBuffer) else ""
    )
    if not space:
      raise VerifyException(f"unsupported tile formal type '{type(arg.type).__name__}'")
    formals.append(ExecTileFormal(space, dims, dtype))
    objects[arg] = ExecMemoryView(
      space,
      f"formal:{index}",
      dims,
      dims,
      (0,) * len(dims),
      (1,) * len(dims),
      dtype,
      DTYPE_BYTES[dtype],
      _view_bytes(dims, dtype),
    )

  symbol = op.sym_name.data
  descriptor_index = 0
  for index, body_op in enumerate(_body_ops(op)):
    source = _ref(source_name, symbol, index, body_op)
    instruction_id = _iid(binding_id, f"tile:{symbol}", index)
    if isinstance(body_op, TileAllocOp):
      dims, dtype = _dims_dtype(body_op.result.type)
      alignment = int(body_op.alignment.value.data) if body_op.alignment is not None else 1
      name = f"l1:{len(l1_buffers)}"
      l1_buffers.append(
        ExecL1Buffer(name, dims, dtype, DTYPE_BYTES[dtype], alignment, _view_bytes(dims, dtype))
      )
      objects[body_op.result] = ExecMemoryView(
        "l1",
        name,
        dims,
        dims,
        (0,) * len(dims),
        (1,) * len(dims),
        dtype,
        DTYPE_BYTES[dtype],
        _view_bytes(dims, dtype),
      )
      insts.append(
        ExecTileInst(
          ExecTileOp.ALLOC_L1,
          args=(name, len(l1_buffers) - 1),
          source_ref=source,
          instruction_id=f"{instruction_id}:alloc_l1",
        )
      )
    elif isinstance(body_op, TileFreeOp):
      buffer = _memory_view(objects, body_op.buffer, body_op.name)
      insts.append(
        ExecTileInst(
          ExecTileOp.FREE_L1,
          args=(buffer.base,),
          source_ref=source,
          instruction_id=f"{instruction_id}:free_l1",
        )
      )
    elif isinstance(body_op, TileSubviewOp):
      src = _memory_view(objects, body_op.src, body_op.name)
      dims = tuple(_int_list(body_op.sizes))
      dtype = body_op.result.type.dtype.data
      objects[body_op.result] = ExecMemoryView(
        "l2",
        src.base,
        src.backing_dims,
        dims,
        tuple(_int_list(body_op.offsets)),
        tuple(_int_list(body_op.strides)),
        dtype,
        src.element_bytes,
        _view_bytes(dims, dtype),
        None if body_op.task_dim is None else int(body_op.task_dim.value.data),
      )
    elif isinstance(
      body_op,
      (TileLoadOp, TileStoreOp, TileGatherOp, TileScatterOp, TilePowOp, TileEvuOp, TileBoaOp),
    ):
      descriptor_name = f"d{descriptor_index}"
      descriptor_index += 1
      descriptor, launch_op = _lower_engine_descriptor(body_op, descriptor_name, objects)
      descriptors[descriptor_name] = descriptor
      insts.append(
        ExecTileInst(
          launch_op,
          dst=_event_tag(body_op.result.type),
          args=(descriptor_name,),
          source_ref=source,
          instruction_id=f"{instruction_id}:launch",
        )
      )
    elif isinstance(body_op, TileAwaitOp):
      events = tuple(_event_tag(operand.type) for operand in body_op.events)
      if not events:
        raise VerifyException("tile.await requires at least one event operand")
      insts.append(
        ExecTileInst(
          ExecTileOp.WAIT if len(events) == 1 else ExecTileOp.WAITALL,
          args=events,
          source_ref=source,
          instruction_id=f"{instruction_id}:await",
        )
      )
    elif isinstance(body_op, TileSignalOp):
      task_index = next((i for i, arg in enumerate(op.body.block.args) if arg is body_op.task), None)
      if task_index is None:
        raise VerifyException("tile.signal operand must be a tile.program block argument")
      insts.append(
        ExecTileInst(
          ExecTileOp.SIGNAL_PHASE,
          args=(body_op.phase.data, task_index),
          source_ref=source,
          instruction_id=f"{instruction_id}:signal",
        )
      )
    elif isinstance(body_op, TileReturnOp):
      insts.append(
        ExecTileInst(ExecTileOp.RET, source_ref=source, instruction_id=f"{instruction_id}:return")
      )
    else:
      raise VerifyException(f"unexpected tile program body op '{body_op.name}'")
  return ExecTileProgram(
    symbol,
    tuple(insts),
    descriptors,
    {},
    0,
    1,
    0,
    tuple(formals),
    tuple(l1_buffers),
    op.resource_contract.to_contract(),
    None,
  )


def _event_tag(event_type) -> str:
  if not isinstance(event_type, (NestEvent, TileEvent, NexusEvent)):
    raise VerifyException(f"expected event type, got {type(event_type).__name__}")
  return event_type.tag.data


def _dims_dtype(type_attr: Attribute) -> tuple[tuple[int, ...], str]:
  if not isinstance(type_attr, (NestBuffer, NestGlobalMemref, NestGlobalView, NestL2View, TileL1Buffer)):
    raise VerifyException(f"expected shape-typed memory attribute, got {type(type_attr).__name__}")
  return tuple(_int_list(type_attr.dims)), type_attr.dtype.data


def _global_input(arg: SSAValue, index: int) -> ExecGlobalInput:
  dims, dtype = _dims_dtype(arg.type)
  return ExecGlobalInput(arg.name_hint or f"arg{index}", dims, dtype, _view_bytes(dims, dtype))


def _block_arg_index(value: SSAValue, args: Sequence[SSAValue]) -> int:
  for index, arg in enumerate(args):
    if value is arg:
      return index
  raise VerifyException("submit actual is not a nexus.program block argument")


def _memory_view(
  objects: Mapping[SSAValue, ExecMemoryView | ExecL2Buffer], value: SSAValue, op_name: str
) -> ExecMemoryView:
  result = objects.get(value)
  if not isinstance(result, ExecMemoryView):
    raise VerifyException(f"{op_name} references an unknown memory view")
  return result


def _l2_buffer(
  objects: Mapping[SSAValue, ExecMemoryView | ExecL2Buffer], value: SSAValue, op_name: str
) -> ExecL2Buffer:
  result = objects.get(value)
  if not isinstance(result, ExecL2Buffer):
    raise VerifyException(f"{op_name} references an unknown L2 buffer")
  return result


def _l2_buffer_view(buffer: ExecL2Buffer) -> ExecMemoryView:
  return ExecMemoryView(
    "l2",
    buffer.slot,
    buffer.dims,
    buffer.dims,
    (0,) * len(buffer.dims),
    (1,) * len(buffer.dims),
    buffer.dtype,
    buffer.element_bytes,
    buffer.bytes,
  )
