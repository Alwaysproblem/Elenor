"""Lower public xDSL workload IR to private execution DTOs.

Lowers the function-call style IR (see reference.mlir) to the
``ExecTileGroupTask`` consumed by the cycle-accurate simulator.  The
lowering is a direct 1:1 walk of the IR body, so the trace (engine jobs,
event ids, PMU counters) corresponds exactly to the IR ops.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from xdsl.ir import Attribute, SSAValue
from xdsl.utils.exceptions import VerifyException

from .dialects.elenor import (
  NestAllocOp,
  NestAwaitOp,
  NestBarrierOp,
  NestBuffer,
  NestCollectiveOp,
  NestContextOp,
  NestDispatchOp,
  NestDMAStoreOp,
  NestEvent,
  NestGlobalMemref,
  NestGlobalView,
  NestL2View,
  NestPrefetchOp,
  NestReleaseOp,
  NestReturnOp,
  NestSubviewOp,
  NestTaskRangeOp,
  NexusAwaitOp,
  NexusEvent,
  NexusProgramOp,
  NexusReturnOp,
  NexusSubmitContextOp,
  TileAllocOp,
  TileAwaitOp,
  TileBoaOp,
  TileEvent,
  TileEvuOp,
  TileL1Buffer,
  TileLoadOp,
  TilePowOp,
  TileProgramDefOp,
  TileReturnOp,
  TileSignalOp,
  TileStoreOp,
  TileSubviewOp,
)
from .execution_ir import (
  ExecDeviceOp,
  ExecEngineDesc,
  ExecGlobalInput,
  ExecGroupAction,
  ExecGroupActionOp,
  ExecL1Buffer,
  ExecL2Buffer,
  ExecMemoryView,
  ExecModel,
  ExecTaskDomain,
  ExecTileFormal,
  ExecTileGroupTask,
  ExecTileInst,
  ExecTileOp,
  ExecTileProgram,
  ExecTileRoleBinding,
  ExecTransfer,
)
from .workload_ir import _view_bytes, verify_workload_ir


def lower_workload_ir(module) -> ExecTileGroupTask:
  context = verify_workload_ir(module)
  assert isinstance(context, NestContextOp)
  return _lower_context(module, context)


def lower_model_ir(module) -> ExecModel:
  program = verify_workload_ir(module)
  assert isinstance(program, NexusProgramOp)
  program_args = list(program.body.block.args)
  inputs = tuple(
    _global_input(arg, i) for i, arg in enumerate(program_args)
  )
  tasks: dict[str, ExecTileGroupTask] = {}
  context_pins: dict[str, int | None] = {}
  for op in module.body.block.ops:
    if isinstance(op, NestContextOp):
      tasks[op.sym_name.data] = _lower_context(module, op)
      context_pins[op.sym_name.data] = (
        None if op.context_id is None else int(op.context_id.value.data)
      )
  body: list[ExecDeviceOp] = []
  for body_op in _body_ops(program):
    if isinstance(body_op, NexusSubmitContextOp):
      body.append(
        ExecDeviceOp(
          "submit",
          ctx_name=body_op.context_sym.data,
          event_tag=_event_tag(body_op.result.type),
          actual_inputs=tuple(
            _block_arg_index(actual, program_args)
            for actual in body_op.actuals
          ),
        )
      )
    elif isinstance(body_op, NexusAwaitOp):
      for ev in body_op.events:
        body.append(ExecDeviceOp("await", event_tag=_event_tag(ev.type)))
    elif isinstance(body_op, NexusReturnOp):
      body.append(ExecDeviceOp("return"))
    else:
      raise VerifyException(f"unexpected nexus.program body op '{body_op.name}'")
  return ExecModel(
    name=program.sym_name.data,
    tasks=tasks,
    context_pins=context_pins,
    body=body,
    inputs=inputs,
  )


# ---------------------------------------------------------------------------
# Context (nest.context) lowering
# ---------------------------------------------------------------------------


def _lower_context(module, context: NestContextOp) -> ExecTileGroupTask:
  # Collect tile program definitions.
  programs: dict[str, ExecTileProgram] = {}
  for op in module.body.block.ops:
    if isinstance(op, TileProgramDefOp):
      programs[op.sym_name.data] = _lower_program(op)

  context_args = list(context.body.block.args)
  global_inputs = tuple(
    _global_input(arg, i) for i, arg in enumerate(context_args)
  )
  formal_names: dict[SSAValue, str] = {
    arg: global_input.name
    for arg, global_input in zip(context_args, global_inputs)
  }
  objects: dict[SSAValue, ExecMemoryView | ExecL2Buffer] = {}
  task_domains: dict[SSAValue, ExecTaskDomain] = {}
  l2_buffers: list[ExecL2Buffer] = []
  actions: list[ExecGroupAction] = []
  role_bindings: dict[int, ExecTileRoleBinding] = {}
  placement = int(context.placement.value.data)

  for body_op in _body_ops(context):
    if isinstance(body_op, NestAllocOp):
      dims, dtype = _dims_dtype(body_op.result.type)
      buffer = ExecL2Buffer(
        slot=body_op.slot.data,
        dims=dims,
        dtype=dtype,
        role=body_op.role.data,
        bytes=_view_bytes(dims, dtype),
      )
      l2_buffers.append(buffer)
      objects[body_op.result] = buffer
    elif isinstance(body_op, NestTaskRangeOp):
      task_domains[body_op.result] = ExecTaskDomain(
        from_task=int(body_op.from_task.value.data),
        to_task=int(body_op.to_task.value.data),
      )
    elif isinstance(body_op, NestSubviewOp):
      formal_name = formal_names.get(body_op.src)
      if formal_name is None:
        raise VerifyException("nest.subview source has no lowered global formal")
      dims = tuple(_int_list(body_op.sizes))
      dtype = body_op.result.type.dtype.data
      objects[body_op.result] = ExecMemoryView(
        space="global",
        base=f"global:{formal_name}",
        dims=dims,
        offsets=tuple(_int_list(body_op.offsets)),
        dtype=dtype,
        bytes=_view_bytes(dims, dtype),
      )
    elif isinstance(body_op, NestPrefetchOp):
      src = _memory_view(objects, body_op.src, body_op.name)
      dst_buffer = _l2_buffer(objects, body_op.dst, body_op.name)
      transfer = ExecTransfer(
        src=src,
        dst=_l2_buffer_view(dst_buffer),
        bytes=src.bytes,
      )
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.DMA_PREFETCH,
          args=(f"gdma_prefetch:{dst_buffer.slot}", transfer),
          dst=_event_tag(body_op.result.type),
        )
      )
    elif isinstance(body_op, NestDMAStoreOp):
      # depends_on -> WAIT actions first
      for dep in body_op.depends_on:
        actions.append(
          ExecGroupAction(
            ExecGroupActionOp.WAIT_EVENT,
            args=(_event_tag(dep.type),),
          )
        )
      src_buffer = _l2_buffer(objects, body_op.src, body_op.name)
      dst = _memory_view(objects, body_op.dst, body_op.name)
      transfer = ExecTransfer(
        src=_l2_buffer_view(src_buffer),
        dst=dst,
        bytes=dst.bytes,
      )
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.DMA_STORE,
          args=(f"gdma_store:{src_buffer.slot}", transfer),
          dst=_event_tag(body_op.result.type),
        )
      )
    elif isinstance(body_op, NestDispatchOp):
      # depends_on -> WAIT actions first
      for dep in body_op.depends_on:
        actions.append(
          ExecGroupAction(
            ExecGroupActionOp.WAIT_EVENT,
            args=(_event_tag(dep.type),),
          )
        )
      prog_sym = body_op.program.data
      ctx_id = (
        None
        if body_op.context_id is None
        else int(body_op.context_id.value.data)
      )
      task_domain = task_domains.get(body_op.tasks)
      if task_domain is None:
        raise VerifyException("dispatch task range has no lowered task domain")
      actuals = tuple(
        _l2_buffer(objects, actual, body_op.name).slot
        for actual in (*body_op.ins, *body_op.outs)
      )
      role_id = _register_role(
        role_bindings,
        programs,
        prog_sym,
        placement,
        context_id=ctx_id,
        task_domain=task_domain,
        actuals=actuals,
      )
      grid_tag = _event_tag(body_op.grid_done.type)
      inrel_tag = _event_tag(body_op.input_released.type)
      outready_tag = _event_tag(body_op.output_ready.type)
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.DISPATCH_ROLE,
          args=(role_id, inrel_tag, outready_tag),
          dst=grid_tag,
        )
      )
    elif isinstance(body_op, NestCollectiveOp):
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.COLLECTIVE_RUN,
          args=(
            body_op.collective.data,
            body_op.collective.data,
            int(body_op.bytes_total.value.data),
            int(body_op.participant_mask.value.data),
          ),
          dst=_event_tag(body_op.result.type),
        )
      )
    elif isinstance(body_op, NestReleaseOp):
      # depends_on -> WAIT actions first
      for dep in body_op.depends_on:
        actions.append(
          ExecGroupAction(
            ExecGroupActionOp.WAIT_EVENT,
            args=(_event_tag(dep.type),),
          )
        )
      buffer = _l2_buffer(objects, body_op.buffer, body_op.name)
      actions.append(
        ExecGroupAction(ExecGroupActionOp.RELEASE_L2, args=(buffer.slot,))
      )
    elif isinstance(body_op, NestAwaitOp):
      for operand in body_op.events:
        actions.append(
          ExecGroupAction(
            ExecGroupActionOp.WAIT_EVENT,
            args=(_event_tag(operand.type),),
          )
        )
    elif isinstance(body_op, NestBarrierOp):
      actions.append(ExecGroupAction(ExecGroupActionOp.BARRIER_GROUP))
    elif isinstance(body_op, NestReturnOp):
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.SIGNAL_EVENT,
          args=(context.completion_event.data,),
        )
      )
    else:
      raise VerifyException(f"unexpected nest context body op '{body_op.name}'")

  return ExecTileGroupTask(
    name=context.sym_name.data,
    actions=actions,
    streams=[],
    role_bindings=role_bindings,
    completion_event=context.completion_event.data,
    global_inputs=global_inputs,
    l2_buffers=tuple(l2_buffers),
  )

def _register_role(
  bindings: dict[int, ExecTileRoleBinding],
  programs: dict[str, ExecTileProgram],
  prog_sym: str,
  tile_mask: int,
  context_id: int | None = None,
  task_domain: ExecTaskDomain | None = None,
  actuals: tuple[str, ...] = (),
) -> int:
  """Register one logical tile-program binding and return its role id."""
  for rid, binding in bindings.items():
    if (
      binding.tile_program.name == prog_sym
      and binding.tile_mask == tile_mask
      and binding.context_id == context_id
      and binding.task_domain == task_domain
      and binding.actuals == actuals
    ):
      return rid
  role_id = len(bindings)
  prog = programs.get(prog_sym)
  if prog is None:
    raise VerifyException(f"dispatch references unknown tile program '@{prog_sym}'")
  bindings[role_id] = ExecTileRoleBinding(
    role_id=role_id,
    tile_mask=tile_mask,
    tile_program=prog,
    context_id=context_id,
    task_domain=task_domain,
    actuals=actuals,
  )
  return role_id


# ---------------------------------------------------------------------------
# Tile program lowering
# ---------------------------------------------------------------------------


def _lower_program(op: TileProgramDefOp) -> ExecTileProgram:
  descriptors: dict[str, ExecEngineDesc] = {}
  insts: list[ExecTileInst] = []
  l1_buffers: list[ExecL1Buffer] = []
  objects: dict[SSAValue, ExecMemoryView] = {}
  formals: list[ExecTileFormal] = []
  for i, arg in enumerate(op.body.block.args):
    if i == 0:
      formals.append(ExecTileFormal(space="task", dims=(), dtype=""))
      continue
    dims, dtype = _dims_dtype(arg.type)
    formals.append(ExecTileFormal(space="l2", dims=dims, dtype=dtype))
    objects[arg] = ExecMemoryView(
      space="l2",
      base=f"formal:{i}",
      dims=dims,
      offsets=(0,) * len(dims),
      dtype=dtype,
      bytes=_view_bytes(dims, dtype),
    )

  desc_counter = 0
  l1_counter = 0
  for body_op in _body_ops(op):
    if isinstance(body_op, TileAllocOp):
      dims, dtype = _dims_dtype(body_op.result.type)
      name = f"l1:{l1_counter}"
      l1_counter += 1
      l1_buffers.append(
        ExecL1Buffer(
          name=name,
          dims=dims,
          dtype=dtype,
          bytes=_view_bytes(dims, dtype),
        )
      )
      objects[body_op.result] = ExecMemoryView(
        space="l1",
        base=name,
        dims=dims,
        offsets=(0,) * len(dims),
        dtype=dtype,
        bytes=_view_bytes(dims, dtype),
      )
    elif isinstance(body_op, TileSubviewOp):
      src = _memory_view(objects, body_op.src, body_op.name)
      dims = tuple(_int_list(body_op.sizes))
      dtype = body_op.result.type.dtype.data
      objects[body_op.result] = ExecMemoryView(
        space="l2",
        base=src.base,
        dims=dims,
        offsets=tuple(_int_list(body_op.offsets)),
        dtype=dtype,
        bytes=_view_bytes(dims, dtype),
        task_dim=(
          None
          if body_op.task_dim is None
          else int(body_op.task_dim.value.data)
        ),
      )
    elif isinstance(body_op, TileLoadOp):
      desc_name = f"d{desc_counter}"
      desc_counter += 1
      src = _memory_view(objects, body_op.src, body_op.name)
      dst = _memory_view(objects, body_op.dst, body_op.name)
      descriptors[desc_name] = ExecEngineDesc(
        name=desc_name,
        kind="MFE",
        op="load",
        params={},
        transfer=ExecTransfer(src=src, dst=dst, bytes=src.bytes),
      )
      insts.append(
        ExecTileInst(
          ExecTileOp.LAUNCH_MFE,
          dst=_event_tag(body_op.result.type),
          args=(desc_name,),
        )
      )
    elif isinstance(body_op, TileStoreOp):
      desc_name = f"d{desc_counter}"
      desc_counter += 1
      src = _memory_view(objects, body_op.src, body_op.name)
      dst = _memory_view(objects, body_op.dst, body_op.name)
      descriptors[desc_name] = ExecEngineDesc(
        name=desc_name,
        kind="MFE",
        op="store",
        params={},
        transfer=ExecTransfer(src=src, dst=dst, bytes=src.bytes),
      )
      insts.append(
        ExecTileInst(
          ExecTileOp.LAUNCH_MFE,
          dst=_event_tag(body_op.result.type),
          args=(desc_name,),
        )
      )
    elif isinstance(body_op, TilePowOp):
      desc_name = f"d{desc_counter}"
      desc_counter += 1
      descriptors[desc_name] = ExecEngineDesc(
        name=desc_name,
        kind="EVU",
        op="pow",
        params={
          "bytes": int(body_op.bytes_total.value.data),
          "exponent": int(body_op.exponent.value.data),
          "ops": int(body_op.pow_ops.value.data),
        },
      )
      insts.append(
        ExecTileInst(
          ExecTileOp.LAUNCH_EVU,
          dst=_event_tag(body_op.result.type),
          args=(desc_name,),
        )
      )
    elif isinstance(body_op, TileEvuOp):
      desc_name = f"d{desc_counter}"
      desc_counter += 1
      descriptors[desc_name] = ExecEngineDesc(
        name=desc_name,
        kind="EVU",
        op=body_op.op_name.data,
        params={"ops": int(body_op.evu_ops.value.data)},
      )
      insts.append(
        ExecTileInst(
          ExecTileOp.LAUNCH_EVU,
          dst=_event_tag(body_op.result.type),
          args=(desc_name,),
        )
      )
    elif isinstance(body_op, TileBoaOp):
      desc_name = f"d{desc_counter}"
      desc_counter += 1
      params = {
        "m": int(body_op.m.value.data),
        "n": int(body_op.n.value.data),
        "k": int(body_op.k.value.data),
        "ops": int(body_op.boa_ops.value.data),
      }
      if body_op.accumulate is not None:
        params["accumulate"] = True
      descriptors[desc_name] = ExecEngineDesc(
        name=desc_name,
        kind="BOA",
        op=body_op.op_name.data,
        params=params,
      )
      insts.append(
        ExecTileInst(
          ExecTileOp.LAUNCH_BOA,
          dst=_event_tag(body_op.result.type),
          args=(desc_name,),
        )
      )
    elif isinstance(body_op, TileAwaitOp):
      events = [_event_tag(operand.type) for operand in body_op.events]
      if len(events) == 1:
        insts.append(ExecTileInst(ExecTileOp.WAIT, args=(events[0],)))
      elif len(events) > 1:
        insts.append(ExecTileInst(ExecTileOp.WAITALL, args=tuple(events)))
      else:
        raise VerifyException("tile.await requires at least one event operand")
    elif isinstance(body_op, TileSignalOp):
      insts.append(
        ExecTileInst(ExecTileOp.SIGNAL_PHASE, args=(body_op.phase.data,))
      )
    elif isinstance(body_op, TileReturnOp):
      insts.append(ExecTileInst(ExecTileOp.RET))
    else:
      raise VerifyException(f"unexpected tile program body op '{body_op.name}'")

  return ExecTileProgram(
    name=op.sym_name.data,
    insts=insts,
    descriptors=descriptors,
    labels={},
    formals=tuple(formals),
    l1_buffers=tuple(l1_buffers),
  )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body_ops(op) -> list:
  """Return the ops in the single-block body region of a definition op."""
  region = op.body if hasattr(op, "body") else op.regions[0]
  if len(region.blocks) != 1:
    raise VerifyException("expected exactly one block in body region")
  return list(region.blocks[0].ops)


def _event_tag(event_type) -> str:
  """Extract the runtime event id from a ``!nest.event<tag>``,
  ``!tile.event<tag>``, or ``!nexus.event<"tag">`` type."""
  if not isinstance(event_type, (NestEvent, TileEvent, NexusEvent)):
    raise VerifyException(f"expected event type, got {type(event_type).__name__}")
  return event_type.tag.data


def _dims_dtype(type_attr: Attribute) -> tuple[tuple[int, ...], str]:
  if not isinstance(
    type_attr, (NestBuffer, NestGlobalMemref, NestGlobalView, NestL2View, TileL1Buffer)
  ):
    raise VerifyException(f"expected shape-typed memory attribute, got {type(type_attr).__name__}")
  return tuple(_int_list(type_attr.dims)), type_attr.dtype.data


def _int_list(arr) -> list[int]:
  return [int(value.value.data) for value in arr.data]


def _global_input(arg: SSAValue, index: int) -> ExecGlobalInput:
  dims, dtype = _dims_dtype(arg.type)
  return ExecGlobalInput(
    name=arg.name_hint or f"arg{index}",
    dims=dims,
    dtype=dtype,
    size_bytes=_view_bytes(dims, dtype),
  )


def _block_arg_index(value: SSAValue, args: Sequence[SSAValue]) -> int:
  for index, arg in enumerate(args):
    if value is arg:
      return index
  raise VerifyException("submit actual is not a nexus.program block argument")


def _memory_view(
  objects: Mapping[SSAValue, ExecMemoryView | ExecL2Buffer],
  value: SSAValue,
  op_name: str,
) -> ExecMemoryView:
  result = objects.get(value)
  if not isinstance(result, ExecMemoryView):
    raise VerifyException(f"{op_name} references an unknown memory view")
  return result


def _l2_buffer(
  objects: Mapping[SSAValue, ExecMemoryView | ExecL2Buffer],
  value: SSAValue,
  op_name: str,
) -> ExecL2Buffer:
  result = objects.get(value)
  if not isinstance(result, ExecL2Buffer):
    raise VerifyException(f"{op_name} references an unknown L2 buffer")
  return result


def _l2_buffer_view(buffer: ExecL2Buffer) -> ExecMemoryView:
  return ExecMemoryView(
    space="l2",
    base=buffer.slot,
    dims=buffer.dims,
    offsets=(0,) * len(buffer.dims),
    dtype=buffer.dtype,
    bytes=buffer.bytes,
  )
