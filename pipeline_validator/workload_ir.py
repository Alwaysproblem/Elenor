"""Public workload IR I/O and verification API.

The single public workload IR is xDSL generic assembly rooted at
``builtin.module`` and containing exactly one
``elenor.runtime.tile_group_task``.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import cast

from xdsl.context import Context
from xdsl.dialects.builtin import (
  Builtin,
  DictionaryAttr,
  FloatAttr,
  IndexType,
  IntegerAttr,
  ModuleOp,
  StringAttr,
  f64,
  i1,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
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
  Elenor,
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
  TileInstructionLike,
  TileProgramOp,
  TileRoleBindingOp,
  TrapOp,
  USEDescriptorOp,
  WaitAllOp,
  WaitOp,
  _decode_array,
)

DescriptorOp = BOADescriptorOp | EVUDescriptorOp | MFEDescriptorOp | USEDescriptorOp
ProducingGroupOp = GroupDMAPrefetchOp | GroupDMAStoreOp | DispatchRoleOp | CollectiveRunOp | SignalEventOp
LaunchOp = LaunchBOAOp | LaunchEVUOp | LaunchMFEOp | LaunchUSEOp | DMALoadOp | DMAStoreOp


def make_elenor_context() -> Context:
  ctx = Context(allow_unregistered=False)
  ctx.load_dialect(Builtin)
  ctx.load_dialect(Elenor)
  return ctx


def parse_workload_ir(text: str, source_name: str = "<memory>") -> ModuleOp:
  module = Parser(make_elenor_context(), text, name=source_name).parse_module()
  verify_workload_ir(module)
  return module


def print_workload_ir(module: ModuleOp) -> str:
  stream = StringIO()
  Printer(stream=stream, print_generic_format=True).print_op(module)
  text = stream.getvalue()
  return text.rstrip("\n") + "\n"


def print_workload_ir_pretty(module: ModuleOp) -> str:
  """Print the workload IR in short custom-assembly format (non-generic)."""
  stream = StringIO()
  Printer(stream=stream, print_generic_format=False).print_op(module)
  text = stream.getvalue()
  return text.rstrip("\n") + "\n"


def parse_workload_ir_pretty(text: str, source_name: str = "<memory>") -> ModuleOp:
  """Parse short custom-assembly workload IR (the same Parser also accepts generic)."""
  module = Parser(make_elenor_context(), text, name=source_name).parse_module()
  verify_workload_ir(module)
  return module


def load_workload_ir_pretty(path: str | Path) -> ModuleOp:
  actual = Path(path)
  text = actual.read_text(encoding="utf-8")
  return parse_workload_ir_pretty(text, source_name=str(actual))


def load_workload_ir(path: str | Path) -> ModuleOp:
  actual = Path(path)
  text = actual.read_text(encoding="utf-8")
  return parse_workload_ir(text, source_name=str(actual))


def verify_workload_ir(module: ModuleOp) -> TileGroupTaskOp:
  module.verify()
  top_ops = list(module.body.block.ops)
  if len(top_ops) != 1 or not isinstance(top_ops[0], TileGroupTaskOp):
    raise VerifyException("expected exactly one elenor.runtime.tile_group_task")
  task = cast(TileGroupTaskOp, top_ops[0])
  _verify_task(task)
  return task


def _region_ops(region) -> list:
  if len(region.blocks) != 1:
    raise VerifyException("expected exactly one block in single-block region")
  return list(region.blocks[0].ops)


def _int_attr_value(attr: IntegerAttr | None) -> int | None:
  if attr is None:
    return None
  return int(attr.value.data)


def _string_attr_value(attr: StringAttr | None) -> str | None:
  if attr is None:
    return None
  return attr.data


def _task_stream_signature(stream: StreamDescOp) -> tuple[int, int, int, int]:
  return (
    int(stream.queue_id.value.data),
    int(stream.depth.value.data),
    int(stream.producer_mask.value.data),
    int(stream.consumer_mask.value.data),
  )


def _verify_task(task: TileGroupTaskOp) -> None:
  streams = _region_ops(task.streams)
  roles = _region_ops(task.roles)
  actions = _region_ops(task.actions)

  stream_map: dict[int, StreamDescOp] = {}
  for op in streams:
    if not isinstance(op, StreamDescOp):
      raise VerifyException(f"unexpected op '{op.name}' in task streams region")
    queue_id = int(op.queue_id.value.data)
    if queue_id in stream_map:
      raise VerifyException(f"duplicate stream {queue_id}")
    depth = int(op.depth.value.data)
    if depth <= 0:
      raise VerifyException(f"stream {queue_id} depth must be > 0")
    producer_mask = int(op.producer_mask.value.data)
    consumer_mask = int(op.consumer_mask.value.data)
    if producer_mask == 0:
      raise VerifyException(f"stream {queue_id} producer_mask must be non-zero")
    if consumer_mask == 0:
      raise VerifyException(f"stream {queue_id} consumer_mask must be non-zero")
    stream_map[queue_id] = op

  role_map: dict[int, TileRoleBindingOp] = {}
  role_masks: dict[int, int] = {}
  for op in roles:
    if not isinstance(op, TileRoleBindingOp):
      raise VerifyException(f"unexpected op '{op.name}' in task roles region")
    role_id = int(op.role_id.value.data)
    if role_id in role_map:
      raise VerifyException(f"duplicate role_id {role_id}")
    tile_mask = int(op.tile_mask.value.data)
    if tile_mask == 0:
      raise VerifyException(f"role {role_id} tile_mask must be non-zero")
    in_stream = _int_attr_value(op.in_stream)
    out_stream = _int_attr_value(op.out_stream)
    for stream_id in (in_stream, out_stream):
      if stream_id is None:
        continue
      if stream_id < 0 or stream_id not in stream_map:
        raise VerifyException(f"stream {stream_id} is not declared")
    program_ops = _region_ops(op.program)
    if len(program_ops) != 1 or not isinstance(program_ops[0], TileProgramOp):
      raise VerifyException(f"role {role_id} must contain exactly one tile program")
    role_map[role_id] = op
    role_masks[role_id] = tile_mask

  group_dma_events: set[str] = set()
  produced_group_events: set[str] = set()
  init_streams: dict[int, tuple[int, int, int, int]] = {}
  for op in actions:
    if isinstance(op, InitStreamOp):
      qid = int(op.queue_id.value.data)
      sig = (
        qid,
        int(op.depth.value.data),
        int(op.producer_mask.value.data),
        int(op.consumer_mask.value.data),
      )
      prev = init_streams.get(qid)
      if prev is not None and prev != sig:
        raise VerifyException(f"duplicate init.stream {qid} with mismatched fields")
      init_streams[qid] = sig
      if qid in stream_map and sig != _task_stream_signature(stream_map[qid]):
        raise VerifyException(f"init.stream {qid} does not match declared stream")
      continue

    if isinstance(op, GroupDMAPrefetchOp):
      event = op.event.data
      _register_group_event(event, produced_group_events)
      group_dma_events.add(event)
      continue

    if isinstance(op, GroupDMAStoreOp):
      event = op.event.data
      _register_group_event(event, produced_group_events)
      group_dma_events.add(event)
      continue

    if isinstance(op, DispatchRoleOp):
      role_id = int(op.role_id.value.data)
      if role_id not in role_map:
        raise VerifyException(f"unknown role_id {role_id}")
      _register_group_event(op.event.data, produced_group_events)
      continue

    if isinstance(op, CollectiveRunOp):
      _register_group_event(op.event.data, produced_group_events)
      continue

    if isinstance(op, SignalEventOp):
      _register_group_event(op.event.data, produced_group_events)
      continue

    if isinstance(op, GroupWaitEventOp):
      event = op.event.data
      if event not in produced_group_events:
        raise VerifyException(f"group wait references unknown event {event}")
      continue

    if isinstance(op, GroupBarrierOp):
      continue

    raise VerifyException(f"unexpected op '{op.name}' in task actions region")

  for role in role_map.values():
    program = cast(TileProgramOp, _region_ops(role.program)[0])
    _verify_program(program, stream_map, group_dma_events)


def _register_group_event(event: str, produced: set[str]) -> None:
  if event in produced:
    raise VerifyException(f"duplicate group event '{event}'")
  produced.add(event)


def _verify_program(
  program: TileProgramOp, stream_map: dict[int, StreamDescOp], group_dma_events: set[str]
) -> None:
  descriptor_ops = _region_ops(program.descriptors)
  instruction_ops = _region_ops(program.instructions)

  descriptors: dict[str, DescriptorOp] = {}
  for op in descriptor_ops:
    if not isinstance(op, (BOADescriptorOp, EVUDescriptorOp, MFEDescriptorOp, USEDescriptorOp)):
      raise VerifyException(f"unexpected op '{op.name}' in descriptors region")
    name = op.descriptor_name.data
    if name in descriptors:
      raise VerifyException(f"duplicate descriptor '{name}'")
    _verify_descriptor_params(op.params)
    descriptors[name] = cast(DescriptorOp, op)

  labels: dict[str, int] = {}
  for idx, op in enumerate(instruction_ops):
    if isinstance(op, LabelOp):
      label = op.label.data
      if label in labels:
        raise VerifyException(f"duplicate label '{label}'")
      labels[label] = idx

  defined_tokens: set[str] = set()
  produced_local_events: dict[str, int] = {}
  for idx, op in enumerate(instruction_ops):
    _verify_instruction(
      op, idx, descriptors, labels, stream_map, group_dma_events, defined_tokens, produced_local_events
    )


def _verify_descriptor_params(params: DictionaryAttr) -> None:
  for key, attr in params.data.items():
    if isinstance(attr, IntegerAttr):
      if attr.type != i1 and attr.type != IndexType():
        raise VerifyException(f"descriptor field '{key}' has invalid scalar type")
      if key in {"ops", "bytes"}:
        if attr.type == i1 or int(attr.value.data) < 0:
          raise VerifyException(f"descriptor field '{key}' must be a non-negative integer")
      continue
    if isinstance(attr, FloatAttr):
      if attr.type != f64:
        raise VerifyException(f"descriptor field '{key}' has invalid scalar type")
      continue
    if isinstance(attr, StringAttr):
      continue
    raise VerifyException(f"descriptor field '{key}' has invalid scalar type")


def _verify_instruction(
  op: TileInstructionLike,
  idx: int,
  descriptors: dict[str, DescriptorOp],
  labels: dict[str, int],
  stream_map: dict[int, StreamDescOp],
  group_dma_events: set[str],
  defined_tokens: set[str],
  produced_local_events: dict[str, int],
) -> None:
  if isinstance(op, LabelOp | NopOp | FenceOp | ReturnOp | ProfileBeginOp | ProfileEndOp | TrapOp):
    return

  if isinstance(op, MoveOp | AddOp | CompareOp | PatchDescriptorOp | StoreDescriptorOp):
    _ = _decode_array(op.args)
    return

  if isinstance(op, LoadDescriptorOp):
    _ = _decode_array(op.args)
    return

  if isinstance(op, BranchOp | BranchPredicateOp):
    if op.target.data not in labels:
      raise VerifyException(f"branch target '{op.target.data}' is not defined")
    return

  if isinstance(op, BranchEosOp):
    if op.token_register.data not in defined_tokens:
      raise VerifyException(f"token register '{op.token_register.data}' is not defined")
    if op.target.data not in labels:
      raise VerifyException(f"branch target '{op.target.data}' is not defined")
    return

  if isinstance(op, LaunchBOAOp):
    _verify_launch_descriptor(op.descriptor.data, descriptors, BOADescriptorOp)
    produced_local_events[op.event.data] = idx
    return

  if isinstance(op, LaunchEVUOp):
    _verify_launch_descriptor(op.descriptor.data, descriptors, EVUDescriptorOp)
    produced_local_events[op.event.data] = idx
    return

  if isinstance(op, LaunchMFEOp):
    _verify_launch_descriptor(op.descriptor.data, descriptors, MFEDescriptorOp)
    produced_local_events[op.event.data] = idx
    return

  if isinstance(op, LaunchUSEOp):
    _verify_launch_descriptor(op.descriptor.data, descriptors, USEDescriptorOp)
    produced_local_events[op.event.data] = idx
    return

  if isinstance(op, DMALoadOp | DMAStoreOp):
    produced_local_events[op.event.data] = idx
    return

  if isinstance(op, WaitOp):
    _verify_wait_event(op.event.data, produced_local_events, group_dma_events)
    return

  if isinstance(op, WaitAllOp):
    for attr in op.events.data:
      if not isinstance(attr, StringAttr):
        raise VerifyException("waitall events must be strings")
      _verify_wait_event(attr.data, produced_local_events, group_dma_events)
    return

  if isinstance(op, StreamPopOp):
    qid = int(op.queue_id.value.data)
    _verify_stream_qid(qid, stream_map, allow_disabled=False)
    if op.destination_token.data in defined_tokens:
      raise VerifyException(f"token register '{op.destination_token.data}' is already live")
    defined_tokens.add(op.destination_token.data)
    return

  if isinstance(op, StreamAcquireOp):
    qid = int(op.queue_id.value.data)
    disabled = _verify_stream_qid(qid, stream_map, allow_disabled=True)
    if disabled:
      return
    if op.destination_token.data in defined_tokens:
      raise VerifyException(f"token register '{op.destination_token.data}' is already live")
    defined_tokens.add(op.destination_token.data)
    return

  if isinstance(op, StreamPushOp):
    qid = int(op.queue_id.value.data)
    disabled = _verify_stream_qid(qid, stream_map, allow_disabled=True)
    if not disabled and op.token_register.data not in defined_tokens:
      raise VerifyException(f"token register '{op.token_register.data}' is not defined")
    defined_tokens.discard(op.token_register.data)
    return

  if isinstance(op, StreamReleaseOp):
    qid = int(op.queue_id.value.data)
    disabled = _verify_stream_qid(qid, stream_map, allow_disabled=True)
    if not disabled and op.token_register.data not in defined_tokens:
      raise VerifyException(f"token register '{op.token_register.data}' is not defined")
    defined_tokens.discard(op.token_register.data)
    return

  if isinstance(op, StreamEosOp):
    qid = int(op.queue_id.value.data)
    _verify_stream_qid(qid, stream_map, allow_disabled=True)
    return

  raise VerifyException(f"unexpected tile instruction '{op.name}'")


def _verify_stream_qid(qid: int, stream_map: dict[int, StreamDescOp], *, allow_disabled: bool) -> bool:
  if allow_disabled and qid == -1:
    return True
  if qid < 0 or qid not in stream_map:
    raise VerifyException(f"stream {qid} is not declared")
  return False


def _verify_launch_descriptor(
  descriptor_name: str, descriptors: dict[str, DescriptorOp], expected_type: type[DescriptorOp]
) -> None:
  op = descriptors.get(descriptor_name)
  if op is None or not isinstance(op, expected_type):
    raise VerifyException(f"tile launch references unknown descriptor '{descriptor_name}'")


def _verify_wait_event(
  event: str, produced_local_events: dict[str, int], group_dma_events: set[str]
) -> None:
  if event in produced_local_events:
    return
  if event.startswith("ev_dma_") and event in group_dma_events:
    return
  raise VerifyException(f"wait references unknown event {event}")
