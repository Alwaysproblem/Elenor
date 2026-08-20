"""Lower public xDSL workload IR to private execution DTOs.

Lowers the function-call style IR (see reference.mlir) to the
``ExecTileGroupTask`` consumed by the cycle-accurate simulator.  The
lowering is a direct 1:1 walk of the IR body, so the trace (engine jobs,
event ids, PMU counters) corresponds exactly to the IR ops.
"""

from __future__ import annotations

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
  NestPrefetchOp,
  NestReleaseOp,
  NestReturnOp,
  NestTaskRangeOp,
  TileAwaitOp,
  TileBoaOp,
  TileEvent,
  TileEvuOp,
  TileLoadOp,
  TilePowOp,
  TileProgramDefOp,
  TileReturnOp,
  TileSignalOp,
  TileStoreOp,
)
from .execution_ir import (
  ExecEngineDesc,
  ExecGroupAction,
  ExecGroupActionOp,
  ExecTileGroupTask,
  ExecTileInst,
  ExecTileOp,
  ExecTileProgram,
  ExecTileRoleBinding,
)
from .workload_ir import verify_workload_ir


def lower_workload_ir(module) -> ExecTileGroupTask:
  context = verify_workload_ir(module)
  return _lower_context(module, context)


# ---------------------------------------------------------------------------
# Context (nest.context) lowering
# ---------------------------------------------------------------------------


def _lower_context(module, context: NestContextOp) -> ExecTileGroupTask:
  # Collect tile program definitions
  programs: dict[str, ExecTileProgram] = {}
  for op in module.body.block.ops:
    if isinstance(op, TileProgramDefOp):
      programs[op.sym_name.data] = _lower_program(op)

  actions: list[ExecGroupAction] = []
  role_bindings: dict[int, ExecTileRoleBinding] = {}
  placement = int(context.placement.value.data)

  for body_op in _body_ops(context):
    if isinstance(body_op, NestAllocOp):
      # No runtime action; the L2 slot is allocated lazily by the DMA
      # latency model in full_memory fidelity.
      pass
    elif isinstance(body_op, NestTaskRangeOp):
      # No runtime action; the task range is informational (logical task
      # count vs physical tile count is validated by the verifier).
      pass
    elif isinstance(body_op, NestPrefetchOp):
      slot = _buffer_slot(body_op.buffer.type)
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.DMA_PREFETCH,
          args=(f"gdma_prefetch:{slot}", slot, int(body_op.bytes_total.value.data)),
          dst=_event_tag(body_op.result.type),
        )
      )
    elif isinstance(body_op, NestDMAStoreOp):
      # depends_on -> WAIT actions first
      for dep in body_op.depends_on:
        actions.append(ExecGroupAction(ExecGroupActionOp.WAIT_EVENT, args=(_event_tag(dep.type),)))
      slot = _buffer_slot(body_op.buffer.type)
      actions.append(
        ExecGroupAction(
          ExecGroupActionOp.DMA_STORE,
          args=(f"gdma_store:{slot}", slot, int(body_op.bytes_total.value.data)),
          dst=_event_tag(body_op.result.type),
        )
      )
    elif isinstance(body_op, NestDispatchOp):
      # depends_on -> WAIT actions first
      for dep in body_op.depends_on:
        actions.append(ExecGroupAction(ExecGroupActionOp.WAIT_EVENT, args=(_event_tag(dep.type),)))
      prog_sym = body_op.program.data
      ctx_id = None if body_op.context_id is None else int(body_op.context_id.value.data)
      role_id = _register_role(role_bindings, programs, prog_sym, placement, context_id=ctx_id)
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
        actions.append(ExecGroupAction(ExecGroupActionOp.WAIT_EVENT, args=(_event_tag(dep.type),)))
      # Release the L2 slot (frees capacity in full_memory fidelity)
      slot = _buffer_slot(body_op.buffer.type)
      actions.append(ExecGroupAction(ExecGroupActionOp.RELEASE_L2, args=(slot,)))
    elif isinstance(body_op, NestAwaitOp):
      for operand in body_op.events:
        actions.append(ExecGroupAction(ExecGroupActionOp.WAIT_EVENT, args=(_event_tag(operand.type),)))
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
  )

def _register_role(
  bindings: dict[int, ExecTileRoleBinding],
  programs: dict[str, ExecTileProgram],
  prog_sym: str,
  tile_mask: int,
  context_id: int | None = None,
) -> int:
  """Register a (program, tile_mask, context_id) binding if not already registered.

  Returns the role_id.  Re-dispatching the same program with the same
  mask and context pin reuses the existing binding (matching the old
  role semantics).  Different context pins produce distinct roles.
  """
  for rid, binding in bindings.items():
    if (binding.tile_program.name == prog_sym
            and binding.tile_mask == tile_mask
            and binding.context_id == context_id):
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
  )
  return role_id


# ---------------------------------------------------------------------------
# Tile program lowering
# ---------------------------------------------------------------------------


def _lower_program(op: TileProgramDefOp) -> ExecTileProgram:
  descriptors: dict[str, ExecEngineDesc] = {}
  insts: list[ExecTileInst] = []
  desc_counter = 0

  for body_op in _body_ops(op):
    if isinstance(body_op, TileLoadOp):
      desc_name = f"d{desc_counter}"
      desc_counter += 1
      descriptors[desc_name] = ExecEngineDesc(
        name=desc_name, kind="MFE", op="load",
        params={"bytes": int(body_op.bytes_total.value.data), "ops": 0},
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
      descriptors[desc_name] = ExecEngineDesc(
        name=desc_name, kind="MFE", op="store",
        params={"bytes": int(body_op.bytes_total.value.data), "ops": 0},
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
        name=desc_name, kind="EVU", op=body_op.op_name.data, params={"ops": int(body_op.evu_ops.value.data)}
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
        name=desc_name, kind="BOA", op=body_op.op_name.data, params=params
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
  """Extract the runtime event id from a ``!nest.event<tag>`` or
  ``!tile.event<tag>`` type."""
  if not isinstance(event_type, (NestEvent, TileEvent)):
    raise VerifyException(f"expected event type, got {type(event_type).__name__}")
  return event_type.tag.data


def _buffer_slot(buffer_type) -> str:
  """Extract the L2 slot id from a ``!nest.l2_buffer<slot>`` type."""
  if not isinstance(buffer_type, NestBuffer):
    raise VerifyException(f"expected nest.l2_buffer type, got {type(buffer_type).__name__}")
  return buffer_type.slot.data
