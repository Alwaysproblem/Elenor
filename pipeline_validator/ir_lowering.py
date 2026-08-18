"""Lower public xDSL workload IR to private execution DTOs."""

from __future__ import annotations

from xdsl.dialects.builtin import StringAttr
from xdsl.utils.exceptions import VerifyException

from .dialects.elenor import (
  AddOp,
  BOADescriptorOp,
  BranchEosOp,
  BranchOp,
  BranchPredicateOp,
  CollectiveRunOp,
  CompareOp,
  DispatchRoleOp,
  DMALoadOp,
  DMAStoreOp,
  EVUDescriptorOp,
  FenceOp,
  GroupBarrierOp,
  GroupDMAPrefetchOp,
  GroupDMAStoreOp,
  GroupWaitEventOp,
  InitStreamOp,
  LabelOp,
  LaunchBOAOp,
  LaunchEVUOp,
  LaunchMFEOp,
  LaunchUSEOp,
  LoadDescriptorOp,
  MFEDescriptorOp,
  MoveOp,
  NopOp,
  PatchDescriptorOp,
  ProfileBeginOp,
  ProfileEndOp,
  ReturnOp,
  SignalEventOp,
  StoreDescriptorOp,
  StreamAcquireOp,
  StreamDescOp,
  StreamEosOp,
  StreamPopOp,
  StreamPushOp,
  StreamReleaseOp,
  TileGroupTaskOp,
  TileProgramOp,
  TileRoleBindingOp,
  TrapOp,
  USEDescriptorOp,
  WaitAllOp,
  WaitOp,
  _decode_array,
  _decode_params,
)
from .execution_ir import (
  ExecEngineDesc,
  ExecGroupAction,
  ExecGroupActionOp,
  ExecStreamDesc,
  ExecTileGroupTask,
  ExecTileInst,
  ExecTileOp,
  ExecTileProgram,
  ExecTileRoleBinding,
)
from .workload_ir import _int_attr_value, _region_ops, verify_workload_ir


def lower_workload_ir(module) -> ExecTileGroupTask:
  task = verify_workload_ir(module)
  return _lower_task(task)


def _lower_task(task: TileGroupTaskOp) -> ExecTileGroupTask:
  streams = [_lower_stream(op) for op in _region_ops(task.streams)]
  roles = [_lower_role(op) for op in _region_ops(task.roles)]
  actions = [_lower_group_action(op) for op in _region_ops(task.actions)]
  return ExecTileGroupTask(
    name=task.task_name.data,
    streams=streams,
    role_bindings={role.role_id: role for role in roles},
    actions=actions,
    completion_event=task.completion_event.data,
  )


def _lower_stream(op: StreamDescOp) -> ExecStreamDesc:
  return ExecStreamDesc(
    queue_id=int(op.queue_id.value.data),
    depth=int(op.depth.value.data),
    producer_mask=int(op.producer_mask.value.data),
    consumer_mask=int(op.consumer_mask.value.data),
    payload_slot_id=int(op.payload_slot_id.value.data),
    token_stride=int(op.token_stride.value.data),
    pmu_stream_id=int(op.pmu_stream_id.value.data),
  )


def _lower_role(op: TileRoleBindingOp) -> ExecTileRoleBinding:
  program_ops = _region_ops(op.program)
  if len(program_ops) != 1 or not isinstance(program_ops[0], TileProgramOp):
    raise VerifyException("role must contain exactly one tile program")
  return ExecTileRoleBinding(
    role_id=int(op.role_id.value.data),
    tile_mask=int(op.tile_mask.value.data),
    tile_program=_lower_program(program_ops[0]),
    in_stream=_int_attr_value(op.in_stream),
    out_stream=_int_attr_value(op.out_stream),
  )


def _lower_program(op: TileProgramOp) -> ExecTileProgram:
  descriptors: dict[str, ExecEngineDesc] = {}
  for desc_op in _region_ops(op.descriptors):
    lowered = _lower_descriptor(desc_op)
    descriptors[lowered.name] = lowered

  insts: list[ExecTileInst] = []
  labels: dict[str, int] = {}
  for inst_op in _region_ops(op.instructions):
    if isinstance(inst_op, LabelOp):
      labels[inst_op.label.data] = len(insts)
      continue
    insts.append(_lower_instruction(inst_op))

  return ExecTileProgram(
    name=op.program_name.data,
    insts=insts,
    descriptors=descriptors,
    labels=labels,
    version=int(op.version.value.data),
  )


def _lower_descriptor(op) -> ExecEngineDesc:
  if isinstance(op, BOADescriptorOp):
    kind = "BOA"
  elif isinstance(op, EVUDescriptorOp):
    kind = "EVU"
  elif isinstance(op, MFEDescriptorOp):
    kind = "MFE"
  elif isinstance(op, USEDescriptorOp):
    kind = "USE"
  else:
    raise VerifyException(f"unexpected descriptor op '{op.name}'")
  return ExecEngineDesc(
    name=op.descriptor_name.data, kind=kind, op=op.op_name.data, params=_decode_params(op.params)
  )


def _comment(op) -> str:
  comment = getattr(op, "comment", None)
  return "" if comment is None else comment.data


def _lower_instruction(op) -> ExecTileInst:
  if isinstance(op, NopOp):
    return ExecTileInst(ExecTileOp.NOP, comment=_comment(op))
  if isinstance(op, MoveOp):
    return ExecTileInst(ExecTileOp.MOV, dst=op.dst.data, args=_decode_array(op.args), comment=_comment(op))
  if isinstance(op, AddOp):
    return ExecTileInst(ExecTileOp.ADD, dst=op.dst.data, args=_decode_array(op.args), comment=_comment(op))
  if isinstance(op, CompareOp):
    return ExecTileInst(ExecTileOp.CMP, dst=op.dst.data, args=_decode_array(op.args), comment=_comment(op))
  if isinstance(op, BranchOp):
    return ExecTileInst(ExecTileOp.BR, args=(op.target.data,), comment=_comment(op))
  if isinstance(op, BranchPredicateOp):
    return ExecTileInst(ExecTileOp.BRP, args=(op.target.data,), comment=_comment(op))
  if isinstance(op, BranchEosOp):
    return ExecTileInst(
      ExecTileOp.BR_EOS, args=(op.token_register.data, op.target.data), comment=_comment(op)
    )
  if isinstance(op, ReturnOp):
    return ExecTileInst(ExecTileOp.RET, comment=_comment(op))
  if isinstance(op, LaunchBOAOp):
    return ExecTileInst(
      ExecTileOp.LAUNCH_BOA, dst=op.event.data, args=(op.descriptor.data,), comment=_comment(op)
    )
  if isinstance(op, LaunchEVUOp):
    return ExecTileInst(
      ExecTileOp.LAUNCH_EVU, dst=op.event.data, args=(op.descriptor.data,), comment=_comment(op)
    )
  if isinstance(op, LaunchMFEOp):
    return ExecTileInst(
      ExecTileOp.LAUNCH_MFE, dst=op.event.data, args=(op.descriptor.data,), comment=_comment(op)
    )
  if isinstance(op, LaunchUSEOp):
    return ExecTileInst(
      ExecTileOp.LAUNCH_USE, dst=op.event.data, args=(op.descriptor.data,), comment=_comment(op)
    )
  if isinstance(op, DMALoadOp):
    bytes_total = _int_attr_value(op.bytes_total)
    return ExecTileInst(
      ExecTileOp.LAUNCH_DMA_LOAD,
      dst=op.event.data,
      args=("dma", 4096 if bytes_total is None else bytes_total),
      comment=_comment(op),
    )
  if isinstance(op, DMAStoreOp):
    bytes_total = _int_attr_value(op.bytes_total)
    return ExecTileInst(
      ExecTileOp.LAUNCH_DMA_STORE,
      dst=op.event.data,
      args=("dma", 4096 if bytes_total is None else bytes_total),
      comment=_comment(op),
    )
  if isinstance(op, WaitOp):
    return ExecTileInst(ExecTileOp.WAIT, args=(op.event.data,), comment=_comment(op))
  if isinstance(op, WaitAllOp):
    events: list[str] = []
    for attr in op.events.data:
      if not isinstance(attr, StringAttr):
        raise VerifyException("waitall events must be strings")
      events.append(attr.data)
    return ExecTileInst(ExecTileOp.WAITALL, args=tuple(events), comment=_comment(op))
  if isinstance(op, FenceOp):
    return ExecTileInst(ExecTileOp.FENCE, comment=_comment(op))
  if isinstance(op, StreamPopOp):
    return ExecTileInst(
      ExecTileOp.STREAM_POP,
      dst=op.destination_token.data,
      args=(int(op.queue_id.value.data),),
      comment=_comment(op),
    )
  if isinstance(op, StreamPushOp):
    return ExecTileInst(
      ExecTileOp.STREAM_PUSH,
      args=(int(op.queue_id.value.data), op.token_register.data, int(op.producer_id.value.data)),
      comment=_comment(op),
    )
  if isinstance(op, StreamAcquireOp):
    return ExecTileInst(
      ExecTileOp.STREAM_ACQUIRE,
      dst=op.destination_token.data,
      args=(int(op.queue_id.value.data),),
      comment=_comment(op),
    )
  if isinstance(op, StreamReleaseOp):
    return ExecTileInst(
      ExecTileOp.STREAM_RELEASE,
      args=(int(op.queue_id.value.data), op.token_register.data),
      comment=_comment(op),
    )
  if isinstance(op, StreamEosOp):
    return ExecTileInst(
      ExecTileOp.STREAM_PUSH_EOS,
      args=(int(op.queue_id.value.data), int(op.producer_id.value.data)),
      comment=_comment(op),
    )
  if isinstance(op, PatchDescriptorOp):
    return ExecTileInst(ExecTileOp.PATCH_DESC, args=_decode_array(op.args), comment=_comment(op))
  if isinstance(op, LoadDescriptorOp):
    return ExecTileInst(
      ExecTileOp.LOAD_DESC, dst=op.destination.data, args=_decode_array(op.args), comment=_comment(op)
    )
  if isinstance(op, StoreDescriptorOp):
    return ExecTileInst(ExecTileOp.STORE_DESC, args=_decode_array(op.args), comment=_comment(op))
  if isinstance(op, ProfileBeginOp):
    return ExecTileInst(ExecTileOp.PROF_BEGIN, comment=_comment(op))
  if isinstance(op, ProfileEndOp):
    return ExecTileInst(ExecTileOp.PROF_END, comment=_comment(op))
  if isinstance(op, TrapOp):
    return ExecTileInst(ExecTileOp.TRAP, comment=_comment(op))
  raise VerifyException(f"unexpected tile instruction '{op.name}'")


def _lower_group_action(op) -> ExecGroupAction:
  if isinstance(op, InitStreamOp):
    return ExecGroupAction(
      ExecGroupActionOp.INIT_STREAM,
      args=(
        int(op.queue_id.value.data),
        int(op.depth.value.data),
        int(op.producer_mask.value.data),
        int(op.consumer_mask.value.data),
      ),
      comment=_comment(op),
    )
  if isinstance(op, GroupDMAPrefetchOp):
    prefetch_args: list[str | int] = [op.descriptor.data, op.l2_slot.data]
    bytes_total = _int_attr_value(op.bytes_total)
    if bytes_total is not None:
      prefetch_args.append(bytes_total)
    return ExecGroupAction(
      ExecGroupActionOp.DMA_PREFETCH,
      args=tuple(prefetch_args),
      dst=op.event.data,
      comment=_comment(op),
    )
  if isinstance(op, GroupDMAStoreOp):
    store_args: list[str | int] = [op.descriptor.data, op.l2_slot.data]
    bytes_total = _int_attr_value(op.bytes_total)
    if bytes_total is not None:
      store_args.append(bytes_total)
    return ExecGroupAction(
      ExecGroupActionOp.DMA_STORE,
      args=tuple(store_args),
      dst=op.event.data,
      comment=_comment(op),
    )
  if isinstance(op, DispatchRoleOp):
    return ExecGroupAction(
      ExecGroupActionOp.DISPATCH_ROLE,
      args=(int(op.role_id.value.data),),
      dst=op.event.data,
      comment=_comment(op),
    )
  if isinstance(op, GroupWaitEventOp):
    return ExecGroupAction(ExecGroupActionOp.WAIT_EVENT, args=(op.event.data,), comment=_comment(op))
  if isinstance(op, GroupBarrierOp):
    return ExecGroupAction(ExecGroupActionOp.BARRIER_GROUP, comment=_comment(op))
  if isinstance(op, CollectiveRunOp):
    return ExecGroupAction(
      ExecGroupActionOp.COLLECTIVE_RUN,
      args=(
        op.descriptor.data,
        op.collective.data,
        int(op.bytes_total.value.data),
        int(op.participant_mask.value.data),
      ),
      dst=op.event.data,
      comment=_comment(op),
    )
  if isinstance(op, SignalEventOp):
    return ExecGroupAction(ExecGroupActionOp.SIGNAL_EVENT, args=(op.event.data,), comment=_comment(op))
  raise VerifyException(f"unexpected group action '{op.name}'")
