"""ELENOR xDSL dialect for pipeline validator workload IR."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self, TypeAlias

from xdsl.dialects.builtin import (
  ArrayAttr,
  DictionaryAttr,
  FloatAttr,
  IndexType,
  IntegerAttr,
  StringAttr,
  f64,
  i1,
)
from xdsl.ir import Attribute, Block, Dialect, Operation, Region
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_prop_def, prop_def, region_def, traits_def
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import NoTerminator

ScalarParam: TypeAlias = bool | int | float | str


def _index_attr(value: int) -> IntegerAttr:
  return IntegerAttr(value, IndexType())


def _i1_attr(value: bool) -> IntegerAttr:
  return IntegerAttr(1 if value else 0, i1)


def _string_attr(value: str | None) -> StringAttr | None:
  return None if value is None else StringAttr(value)


def _props(mapping: dict[str, Attribute | None]) -> dict[str, Attribute]:
  return {k: v for k, v in mapping.items() if v is not None}


def _single_block_region(ops: Sequence[IRDLOperation]) -> Region:
  return Region([Block(list(ops))])


def _encode_scalar(value: ScalarParam) -> Attribute:
  if isinstance(value, bool):
    return _i1_attr(value)
  if isinstance(value, int):
    return _index_attr(value)
  if isinstance(value, float):
    return FloatAttr(value, f64)
  if isinstance(value, str):
    return StringAttr(value)
  raise TypeError(f"descriptor param must be bool/int/float/str, got {type(value).__name__}")


def _decode_scalar(attr: Attribute) -> ScalarParam:
  if isinstance(attr, IntegerAttr):
    if attr.type == i1:
      return attr.value.data != 0
    return int(attr.value.data)
  if isinstance(attr, FloatAttr):
    return float(attr.value.data)
  if isinstance(attr, StringAttr):
    return attr.data
  raise TypeError(f"cannot decode attribute {attr!r} to scalar")


def _encode_params(params: dict[str, ScalarParam]) -> DictionaryAttr:
  return DictionaryAttr({k: _encode_scalar(v) for k, v in sorted(params.items())})


def _decode_params(attr: DictionaryAttr) -> dict[str, ScalarParam]:
  return {k: _decode_scalar(v) for k, v in attr.data.items()}


def _encode_array(items: tuple[ScalarParam, ...]) -> ArrayAttr:
  return ArrayAttr([_encode_scalar(item) for item in items])

def _decode_array(attr: ArrayAttr) -> tuple[ScalarParam, ...]:
  return tuple(_decode_scalar(item) for item in attr.data)


def _print_optional_comment(printer: Printer, comment: StringAttr | None) -> None:
  if comment is not None:
    printer.print_string(" comment=")
    printer.print_string_literal(comment.data)


def _parse_optional_comment(parser: Parser) -> str | None:
  if parser.parse_optional_keyword("comment") is None:
    return None
  parser.parse_punctuation("=")
  return parser.parse_str_literal()


def _print_optional_int_kw(printer: Printer, keyword: str, value: int) -> None:
  printer.print_string(f" {keyword}=")
  printer.print_int(value)


def _parse_optional_int_kw(parser: Parser, keyword: str) -> int | None:
  if parser.parse_optional_keyword(keyword) is None:
    return None
  parser.parse_punctuation("=")
  return parser.parse_integer()


def _print_region(printer: Printer, region: Region) -> None:
  printer.print_string(" ")
  printer.print_region(region, print_empty_block=False)


def _parse_region(parser: Parser) -> Region:
  region = parser.parse_region()
  if not region.blocks:
    region.add_block(Block())
  if len(region.blocks) != 1:
    parser.raise_error("expected exactly one block in region")
  return region

def _parse_region_ops(parser: Parser) -> list:
  region = _parse_region(parser)
  ops = list(region.blocks[0].ops)
  for op in ops:
    op.detach()
  return ops


def _parse_array_attr(parser: Parser, context_msg: str) -> ArrayAttr:
  attr = parser.parse_attribute()
  if not isinstance(attr, ArrayAttr):
    parser.raise_error(f"expected {context_msg}")
  return attr


EngineDescriptorLike: TypeAlias = "BOADescriptorOp | EVUDescriptorOp | MFEDescriptorOp | USEDescriptorOp"
GroupActionLike: TypeAlias = (
  "InitStreamOp | GroupDMAPrefetchOp | GroupDMAStoreOp | DispatchRoleOp"
  " | GroupWaitEventOp | GroupBarrierOp | CollectiveRunOp | SignalEventOp"
)
TileInstructionLike: TypeAlias = (
  "LabelOp | NopOp | MoveOp | AddOp | CompareOp | BranchOp | BranchPredicateOp"
  " | BranchEosOp | ReturnOp | LaunchBOAOp | LaunchEVUOp | LaunchMFEOp"
  " | LaunchUSEOp | DMALoadOp | DMAStoreOp | WaitOp | WaitAllOp | FenceOp"
  " | StreamPopOp | StreamPushOp | StreamAcquireOp | StreamReleaseOp"
  " | StreamEosOp | PatchDescriptorOp | LoadDescriptorOp | StoreDescriptorOp"
  " | ProfileBeginOp | ProfileEndOp | TrapOp"
)


class _BaseDescriptorOp(IRDLOperation):
  ENGINE_KIND: ClassVar[str] = ""

  descriptor_name = prop_def(StringAttr)
  op_name = prop_def(StringAttr)
  params = prop_def(DictionaryAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, name: str, op_name: str, params: dict[str, ScalarParam], comment: str | None = None):
    super().__init__(
      properties=_props(
        {
          "descriptor_name": StringAttr(name),
          "op_name": StringAttr(op_name),
          "params": _encode_params(params),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.descriptor_name.data)
    printer.print_string(" ")
    printer.print_string_literal(self.op_name.data)
    printer.print_string(" ")
    printer.print_attribute(self.params)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    name = parser.parse_str_literal()
    op_name = parser.parse_str_literal()
    params_attr = parser.parse_attribute()
    if not isinstance(params_attr, DictionaryAttr):
      parser.raise_error("expected descriptor parameter dictionary")
    comment = _parse_optional_comment(parser)
    return cls(name, op_name, _decode_params(params_attr), comment)


@irdl_op_definition
class BOADescriptorOp(_BaseDescriptorOp):
  name = "elenor.boa.descriptor"
  ENGINE_KIND: ClassVar[str] = "BOA"


@irdl_op_definition
class EVUDescriptorOp(_BaseDescriptorOp):
  name = "elenor.evu.descriptor"
  ENGINE_KIND: ClassVar[str] = "EVU"


@irdl_op_definition
class MFEDescriptorOp(_BaseDescriptorOp):
  name = "elenor.mfe.descriptor"
  ENGINE_KIND: ClassVar[str] = "MFE"


@irdl_op_definition
class USEDescriptorOp(_BaseDescriptorOp):
  name = "elenor.use.descriptor"
  ENGINE_KIND: ClassVar[str] = "USE"


@irdl_op_definition
class TileProgramOp(IRDLOperation):
  name = "elenor.runtime.tile_program"

  program_name = prop_def(StringAttr)
  version = prop_def(IntegerAttr, default_value=_index_attr(1))
  comment = opt_prop_def(StringAttr)

  descriptors = region_def("single_block")
  instructions = region_def("single_block")

  traits = traits_def(NoTerminator())

  def __init__(
    self,
    name: str,
    descriptors: Sequence[EngineDescriptorLike],
    instructions: Sequence[TileInstructionLike],
    version: int = 1,
    comment: str | None = None,
  ):
    super().__init__(
      properties=_props(
        {
          "program_name": StringAttr(name),
          "version": _index_attr(version),
          "comment": _string_attr(comment),
        }
      ),
      regions=[_single_block_region(list(descriptors)), _single_block_region(list(instructions))],
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.program_name.data)
    if self.version.value.data != 1:
      _print_optional_int_kw(printer, "version", self.version.value.data)
    _print_optional_comment(printer, self.comment)
    _print_region(printer, self.descriptors)
    _print_region(printer, self.instructions)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    name = parser.parse_str_literal()
    version = _parse_optional_int_kw(parser, "version")
    comment = _parse_optional_comment(parser)
    descriptors = _parse_region_ops(parser)
    instructions = _parse_region_ops(parser)
    return cls(
      name,
      descriptors,
      instructions,
      version=1 if version is None else version,
      comment=comment,
    )


@irdl_op_definition
class StreamDescOp(IRDLOperation):
  name = "elenor.runtime.stream_desc"

  queue_id = prop_def(IntegerAttr)
  depth = prop_def(IntegerAttr)
  producer_mask = prop_def(IntegerAttr)
  consumer_mask = prop_def(IntegerAttr)
  payload_slot_id = prop_def(IntegerAttr, default_value=_index_attr(0))
  token_stride = prop_def(IntegerAttr, default_value=_index_attr(32))
  pmu_stream_id = prop_def(IntegerAttr, default_value=_index_attr(0))
  comment = opt_prop_def(StringAttr)

  def __init__(
    self,
    queue_id: int,
    depth: int,
    producer_mask: int,
    consumer_mask: int,
    payload_slot_id: int = 0,
    token_stride: int = 32,
    pmu_stream_id: int = 0,
    comment: str | None = None,
  ):
    super().__init__(
      properties=_props(
        {
          "queue_id": _index_attr(queue_id),
          "depth": _index_attr(depth),
          "producer_mask": _index_attr(producer_mask),
          "consumer_mask": _index_attr(consumer_mask),
          "payload_slot_id": _index_attr(payload_slot_id),
          "token_stride": _index_attr(token_stride),
          "pmu_stream_id": _index_attr(pmu_stream_id),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_int(self.queue_id.value.data)
    printer.print_string(" ")
    printer.print_int(self.depth.value.data)
    printer.print_string(" ")
    printer.print_int(self.producer_mask.value.data)
    printer.print_string(" ")
    printer.print_int(self.consumer_mask.value.data)
    if self.payload_slot_id.value.data != 0:
      _print_optional_int_kw(printer, "payload_slot", self.payload_slot_id.value.data)
    if self.token_stride.value.data != 32:
      _print_optional_int_kw(printer, "token_stride", self.token_stride.value.data)
    if self.pmu_stream_id.value.data != 0:
      _print_optional_int_kw(printer, "pmu_stream", self.pmu_stream_id.value.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    queue_id = parser.parse_integer()
    depth = parser.parse_integer()
    producer_mask = parser.parse_integer()
    consumer_mask = parser.parse_integer()
    payload_slot_id = _parse_optional_int_kw(parser, "payload_slot")
    token_stride = _parse_optional_int_kw(parser, "token_stride")
    pmu_stream_id = _parse_optional_int_kw(parser, "pmu_stream")
    comment = _parse_optional_comment(parser)
    return cls(
      queue_id,
      depth,
      producer_mask,
      consumer_mask,
      payload_slot_id=0 if payload_slot_id is None else payload_slot_id,
      token_stride=32 if token_stride is None else token_stride,
      pmu_stream_id=0 if pmu_stream_id is None else pmu_stream_id,
      comment=comment,
    )


@irdl_op_definition
class TileRoleBindingOp(IRDLOperation):
  name = "elenor.runtime.tile_role"

  role_id = prop_def(IntegerAttr)
  tile_mask = prop_def(IntegerAttr)
  in_stream = opt_prop_def(IntegerAttr)
  out_stream = opt_prop_def(IntegerAttr)
  comment = opt_prop_def(StringAttr)

  program = region_def("single_block")

  traits = traits_def(NoTerminator())

  def __init__(
    self,
    role_id: int,
    tile_mask: int,
    program: TileProgramOp,
    in_stream: int | None = None,
    out_stream: int | None = None,
    comment: str | None = None,
  ):
    super().__init__(
      properties=_props(
        {
          "role_id": _index_attr(role_id),
          "tile_mask": _index_attr(tile_mask),
          "in_stream": None if in_stream is None else _index_attr(in_stream),
          "out_stream": None if out_stream is None else _index_attr(out_stream),
          "comment": _string_attr(comment),
        }
      ),
      regions=[_single_block_region([program])],
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" role=")
    printer.print_int(self.role_id.value.data)
    printer.print_string(" tile_mask=")
    printer.print_int(self.tile_mask.value.data)
    if self.in_stream is not None:
      _print_optional_int_kw(printer, "in", self.in_stream.value.data)
    if self.out_stream is not None:
      _print_optional_int_kw(printer, "out", self.out_stream.value.data)
    _print_optional_comment(printer, self.comment)
    _print_region(printer, self.program)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    parser.parse_keyword("role")
    parser.parse_punctuation("=")
    role_id = parser.parse_integer()
    parser.parse_keyword("tile_mask")
    parser.parse_punctuation("=")
    tile_mask = parser.parse_integer()
    in_stream = _parse_optional_int_kw(parser, "in")
    out_stream = _parse_optional_int_kw(parser, "out")
    comment = _parse_optional_comment(parser)
    program = _parse_region_ops(parser)
    if len(program) != 1:
      parser.raise_error("expected exactly one elenor.runtime.tile_program in role region")
    return cls(
      role_id,
      tile_mask,
      program[0],
      in_stream=in_stream,
      out_stream=out_stream,
      comment=comment,
    )


@irdl_op_definition
class TileGroupTaskOp(IRDLOperation):
  name = "elenor.runtime.tile_group_task"

  task_name = prop_def(StringAttr)
  completion_event = prop_def(StringAttr, default_value=StringAttr("group_task_done"))
  comment = opt_prop_def(StringAttr)

  streams = region_def("single_block")
  roles = region_def("single_block")
  actions = region_def("single_block")

  traits = traits_def(NoTerminator())

  def __init__(
    self,
    name: str,
    streams: Sequence[StreamDescOp],
    roles: Sequence[TileRoleBindingOp],
    actions: Sequence[GroupActionLike],
    completion_event: str = "group_task_done",
    comment: str | None = None,
  ):
    super().__init__(
      properties=_props(
        {
          "task_name": StringAttr(name),
          "completion_event": StringAttr(completion_event),
          "comment": _string_attr(comment),
        }
      ),
      regions=[
        _single_block_region(list(streams)),
        _single_block_region(list(roles)),
        _single_block_region(list(actions)),
      ],
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.task_name.data)
    if self.completion_event.data != "group_task_done":
      printer.print_string(" completion=")
      printer.print_string_literal(self.completion_event.data)
    _print_optional_comment(printer, self.comment)
    _print_region(printer, self.streams)
    _print_region(printer, self.roles)
    _print_region(printer, self.actions)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    task_name = parser.parse_str_literal()
    completion_event = "group_task_done"
    if parser.parse_optional_keyword("completion") is not None:
      parser.parse_punctuation("=")
      completion_event = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    streams = _parse_region_ops(parser)
    roles = _parse_region_ops(parser)
    actions = _parse_region_ops(parser)
    return cls(
      task_name,
      streams,
      roles,
      actions,
      completion_event=completion_event,
      comment=comment,
    )


@irdl_op_definition
class InitStreamOp(IRDLOperation):
  name = "elenor.runtime.group.init_stream"

  queue_id = prop_def(IntegerAttr)
  depth = prop_def(IntegerAttr)
  producer_mask = prop_def(IntegerAttr)
  consumer_mask = prop_def(IntegerAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(
    self, queue_id: int, depth: int, producer_mask: int, consumer_mask: int, comment: str | None = None
  ):
    super().__init__(
      properties=_props(
        {
          "queue_id": _index_attr(queue_id),
          "depth": _index_attr(depth),
          "producer_mask": _index_attr(producer_mask),
          "consumer_mask": _index_attr(consumer_mask),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_int(self.queue_id.value.data)
    printer.print_string(" ")
    printer.print_int(self.depth.value.data)
    printer.print_string(" ")
    printer.print_int(self.producer_mask.value.data)
    printer.print_string(" ")
    printer.print_int(self.consumer_mask.value.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    queue_id = parser.parse_integer()
    depth = parser.parse_integer()
    producer_mask = parser.parse_integer()
    consumer_mask = parser.parse_integer()
    comment = _parse_optional_comment(parser)
    return cls(queue_id, depth, producer_mask, consumer_mask, comment)


@irdl_op_definition
class GroupDMAPrefetchOp(IRDLOperation):
  name = "elenor.runtime.group.dma_prefetch"

  descriptor = prop_def(StringAttr)
  l2_slot = prop_def(StringAttr)
  event = prop_def(StringAttr)
  bytes_total = opt_prop_def(IntegerAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(
    self,
    descriptor: str,
    l2_slot: str,
    event: str,
    bytes_total: int | None = None,
    comment: str | None = None,
  ):
    super().__init__(
      properties=_props(
        {
          "descriptor": StringAttr(descriptor),
          "l2_slot": StringAttr(l2_slot),
          "event": StringAttr(event),
          "bytes_total": None if bytes_total is None else _index_attr(bytes_total),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.descriptor.data)
    printer.print_string(" ")
    printer.print_string_literal(self.l2_slot.data)
    printer.print_string(" -> ")
    printer.print_string_literal(self.event.data)
    if self.bytes_total is not None:
      _print_optional_int_kw(printer, "bytes", self.bytes_total.value.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    descriptor = parser.parse_str_literal()
    l2_slot = parser.parse_str_literal()
    parser.parse_punctuation("->")
    event = parser.parse_str_literal()
    bytes_total = _parse_optional_int_kw(parser, "bytes")
    comment = _parse_optional_comment(parser)
    return cls(descriptor, l2_slot, event, bytes_total=bytes_total, comment=comment)


@irdl_op_definition
class GroupDMAStoreOp(IRDLOperation):
  name = "elenor.runtime.group.dma_store"

  descriptor = prop_def(StringAttr)
  l2_slot = prop_def(StringAttr)
  event = prop_def(StringAttr)
  bytes_total = opt_prop_def(IntegerAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(
    self,
    descriptor: str,
    l2_slot: str,
    event: str,
    bytes_total: int | None = None,
    comment: str | None = None,
  ):
    super().__init__(
      properties=_props(
        {
          "descriptor": StringAttr(descriptor),
          "l2_slot": StringAttr(l2_slot),
          "event": StringAttr(event),
          "bytes_total": None if bytes_total is None else _index_attr(bytes_total),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.descriptor.data)
    printer.print_string(" ")
    printer.print_string_literal(self.l2_slot.data)
    printer.print_string(" -> ")
    printer.print_string_literal(self.event.data)
    if self.bytes_total is not None:
      _print_optional_int_kw(printer, "bytes", self.bytes_total.value.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    descriptor = parser.parse_str_literal()
    l2_slot = parser.parse_str_literal()
    parser.parse_punctuation("->")
    event = parser.parse_str_literal()
    bytes_total = _parse_optional_int_kw(parser, "bytes")
    comment = _parse_optional_comment(parser)
    return cls(descriptor, l2_slot, event, bytes_total=bytes_total, comment=comment)


@irdl_op_definition
class DispatchRoleOp(IRDLOperation):
  name = "elenor.runtime.group.dispatch_role"

  role_id = prop_def(IntegerAttr)
  event = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, role_id: int, event: str, comment: str | None = None):
    super().__init__(
      properties=_props(
        {"role_id": _index_attr(role_id), "event": StringAttr(event), "comment": _string_attr(comment)}
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_int(self.role_id.value.data)
    printer.print_string(" -> ")
    printer.print_string_literal(self.event.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    role_id = parser.parse_integer()
    parser.parse_punctuation("->")
    event = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(role_id, event, comment)


@irdl_op_definition
class GroupWaitEventOp(IRDLOperation):
  name = "elenor.runtime.group.wait_event"

  event = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, event: str, comment: str | None = None):
    super().__init__(properties=_props({"event": StringAttr(event), "comment": _string_attr(comment)}))

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.event.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    event = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(event, comment)


@irdl_op_definition
class GroupBarrierOp(IRDLOperation):
  name = "elenor.runtime.group.barrier"

  comment = opt_prop_def(StringAttr)

  def __init__(self, comment: str | None = None):
    super().__init__(properties=_props({"comment": _string_attr(comment)}))

  def print(self, printer: Printer) -> None:
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    return cls(_parse_optional_comment(parser))


@irdl_op_definition
class CollectiveRunOp(IRDLOperation):
  name = "elenor.runtime.group.collective_run"

  descriptor = prop_def(StringAttr)
  collective = prop_def(StringAttr)
  bytes_total = prop_def(IntegerAttr)
  participant_mask = prop_def(IntegerAttr)
  event = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(
    self,
    descriptor: str,
    collective: str,
    bytes_total: int,
    participant_mask: int,
    event: str,
    comment: str | None = None,
  ):
    super().__init__(
      properties=_props(
        {
          "descriptor": StringAttr(descriptor),
          "collective": StringAttr(collective),
          "bytes_total": _index_attr(bytes_total),
          "participant_mask": _index_attr(participant_mask),
          "event": StringAttr(event),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.descriptor.data)
    printer.print_string(" ")
    printer.print_string_literal(self.collective.data)
    printer.print_string(" ")
    printer.print_int(self.bytes_total.value.data)
    printer.print_string(" ")
    printer.print_int(self.participant_mask.value.data)
    printer.print_string(" -> ")
    printer.print_string_literal(self.event.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    descriptor = parser.parse_str_literal()
    collective = parser.parse_str_literal()
    bytes_total = parser.parse_integer()
    participant_mask = parser.parse_integer()
    parser.parse_punctuation("->")
    event = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(descriptor, collective, bytes_total, participant_mask, event, comment)


@irdl_op_definition
class SignalEventOp(IRDLOperation):
  name = "elenor.runtime.group.signal_event"

  event = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, event: str, comment: str | None = None):
    super().__init__(properties=_props({"event": StringAttr(event), "comment": _string_attr(comment)}))

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.event.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    event = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(event, comment)


class _CommentOnlyTileOp(IRDLOperation):
  comment = opt_prop_def(StringAttr)

  def __init__(self, comment: str | None = None):
    super().__init__(properties=_props({"comment": _string_attr(comment)}))

  def print(self, printer: Printer) -> None:
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    return cls(_parse_optional_comment(parser))


@irdl_op_definition
class LabelOp(IRDLOperation):
  name = "elenor.runtime.tile.label"

  label = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, name: str, comment: str | None = None):
    super().__init__(properties=_props({"label": StringAttr(name), "comment": _string_attr(comment)}))

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.label.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    name = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(name, comment)


@irdl_op_definition
class NopOp(_CommentOnlyTileOp):
  name = "elenor.runtime.tile.nop"


@irdl_op_definition
class MoveOp(IRDLOperation):
  name = "elenor.runtime.tile.move"

  dst = prop_def(StringAttr)
  args = prop_def(ArrayAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, dst: str, args: tuple[ScalarParam, ...], comment: str | None = None):
    super().__init__(
      properties=_props(
        {"dst": StringAttr(dst), "args": _encode_array(args), "comment": _string_attr(comment)}
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.dst.data)
    printer.print_string(" ")
    printer.print_attribute(self.args)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    dst = parser.parse_str_literal()
    args = _parse_array_attr(parser, "argument array")
    comment = _parse_optional_comment(parser)
    return cls(dst, _decode_array(args), comment)


@irdl_op_definition
class AddOp(IRDLOperation):
  name = "elenor.runtime.tile.add"

  dst = prop_def(StringAttr)
  args = prop_def(ArrayAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, dst: str, args: tuple[ScalarParam, ...], comment: str | None = None):
    super().__init__(
      properties=_props(
        {"dst": StringAttr(dst), "args": _encode_array(args), "comment": _string_attr(comment)}
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.dst.data)
    printer.print_string(" ")
    printer.print_attribute(self.args)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    dst = parser.parse_str_literal()
    args = _parse_array_attr(parser, "argument array")
    comment = _parse_optional_comment(parser)
    return cls(dst, _decode_array(args), comment)


@irdl_op_definition
class CompareOp(IRDLOperation):
  name = "elenor.runtime.tile.cmp"

  dst = prop_def(StringAttr)
  args = prop_def(ArrayAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, dst: str, args: tuple[ScalarParam, ...], comment: str | None = None):
    super().__init__(
      properties=_props(
        {"dst": StringAttr(dst), "args": _encode_array(args), "comment": _string_attr(comment)}
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.dst.data)
    printer.print_string(" ")
    printer.print_attribute(self.args)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    dst = parser.parse_str_literal()
    args = _parse_array_attr(parser, "argument array")
    comment = _parse_optional_comment(parser)
    return cls(dst, _decode_array(args), comment)


@irdl_op_definition
class BranchOp(IRDLOperation):
  name = "elenor.runtime.tile.br"

  target = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, target: str, comment: str | None = None):
    super().__init__(properties=_props({"target": StringAttr(target), "comment": _string_attr(comment)}))

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.target.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    target = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(target, comment)


@irdl_op_definition
class BranchPredicateOp(IRDLOperation):
  name = "elenor.runtime.tile.brp"

  target = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, target: str, comment: str | None = None):
    super().__init__(properties=_props({"target": StringAttr(target), "comment": _string_attr(comment)}))

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.target.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    target = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(target, comment)


@irdl_op_definition
class BranchEosOp(IRDLOperation):
  name = "elenor.runtime.tile.br_eos"

  token_register = prop_def(StringAttr)
  target = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, token_register: str, target: str, comment: str | None = None):
    super().__init__(
      properties=_props(
        {
          "token_register": StringAttr(token_register),
          "target": StringAttr(target),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.token_register.data)
    printer.print_string(" ")
    printer.print_string_literal(self.target.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    token_register = parser.parse_str_literal()
    target = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(token_register, target, comment)


@irdl_op_definition
class ReturnOp(_CommentOnlyTileOp):
  name = "elenor.runtime.tile.ret"


class _BaseLaunchOp(IRDLOperation):
  descriptor = prop_def(StringAttr)
  event = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, descriptor: str, event: str, comment: str | None = None):
    super().__init__(
      properties=_props(
        {"descriptor": StringAttr(descriptor), "event": StringAttr(event), "comment": _string_attr(comment)}
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.descriptor.data)
    printer.print_string(" -> ")
    printer.print_string_literal(self.event.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    descriptor = parser.parse_str_literal()
    parser.parse_punctuation("->")
    event = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(descriptor, event, comment)


@irdl_op_definition
class LaunchBOAOp(_BaseLaunchOp):
  name = "elenor.runtime.tile.launch.boa"


@irdl_op_definition
class LaunchEVUOp(_BaseLaunchOp):
  name = "elenor.runtime.tile.launch.evu"


@irdl_op_definition
class LaunchMFEOp(_BaseLaunchOp):
  name = "elenor.runtime.tile.launch.mfe"


@irdl_op_definition
class LaunchUSEOp(_BaseLaunchOp):
  name = "elenor.runtime.tile.launch.use"


@irdl_op_definition
class DMALoadOp(IRDLOperation):
  name = "elenor.runtime.tile.dma_load"

  event = prop_def(StringAttr)
  bytes_total = opt_prop_def(IntegerAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, event: str, bytes_total: int | None = None, comment: str | None = None):
    super().__init__(
      properties=_props(
        {
          "event": StringAttr(event),
          "bytes_total": None if bytes_total is None else _index_attr(bytes_total),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.event.data)
    if self.bytes_total is not None:
      _print_optional_int_kw(printer, "bytes", self.bytes_total.value.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    event = parser.parse_str_literal()
    bytes_total = _parse_optional_int_kw(parser, "bytes")
    comment = _parse_optional_comment(parser)
    return cls(event, bytes_total=bytes_total, comment=comment)


@irdl_op_definition
class DMAStoreOp(IRDLOperation):
  name = "elenor.runtime.tile.dma_store"

  event = prop_def(StringAttr)
  bytes_total = opt_prop_def(IntegerAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, event: str, bytes_total: int | None = None, comment: str | None = None):
    super().__init__(
      properties=_props(
        {
          "event": StringAttr(event),
          "bytes_total": None if bytes_total is None else _index_attr(bytes_total),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.event.data)
    if self.bytes_total is not None:
      _print_optional_int_kw(printer, "bytes", self.bytes_total.value.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    event = parser.parse_str_literal()
    bytes_total = _parse_optional_int_kw(parser, "bytes")
    comment = _parse_optional_comment(parser)
    return cls(event, bytes_total=bytes_total, comment=comment)


@irdl_op_definition
class WaitOp(IRDLOperation):
  name = "elenor.runtime.tile.wait"

  event = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, event: str, comment: str | None = None):
    super().__init__(properties=_props({"event": StringAttr(event), "comment": _string_attr(comment)}))

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.event.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    event = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(event, comment)


@irdl_op_definition
class WaitAllOp(IRDLOperation):
  name = "elenor.runtime.tile.waitall"

  events = prop_def(ArrayAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, events: Sequence[str], comment: str | None = None):
    super().__init__(
      properties=_props(
        {"events": ArrayAttr([StringAttr(event) for event in events]), "comment": _string_attr(comment)}
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_attribute(self.events)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    events = _parse_array_attr(parser, "event array")
    names = []
    for attr in events.data:
      if not isinstance(attr, StringAttr):
        parser.raise_error("waitall events must be strings")
      names.append(attr.data)
    comment = _parse_optional_comment(parser)
    return cls(names, comment)


@irdl_op_definition
class FenceOp(_CommentOnlyTileOp):
  name = "elenor.runtime.tile.fence"


@irdl_op_definition
class StreamPopOp(IRDLOperation):
  name = "elenor.runtime.tile.stream.pop"

  queue_id = prop_def(IntegerAttr)
  destination_token = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, queue_id: int, destination_token: str, comment: str | None = None):
    super().__init__(
      properties=_props(
        {
          "queue_id": _index_attr(queue_id),
          "destination_token": StringAttr(destination_token),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_int(self.queue_id.value.data)
    printer.print_string(" ")
    printer.print_string_literal(self.destination_token.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    queue_id = parser.parse_integer()
    destination_token = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(queue_id, destination_token, comment)


@irdl_op_definition
class StreamPushOp(IRDLOperation):
  name = "elenor.runtime.tile.stream.push"

  queue_id = prop_def(IntegerAttr)
  token_register = prop_def(StringAttr)
  producer_id = prop_def(IntegerAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, queue_id: int, token_register: str, producer_id: int, comment: str | None = None):
    super().__init__(
      properties=_props(
        {
          "queue_id": _index_attr(queue_id),
          "token_register": StringAttr(token_register),
          "producer_id": _index_attr(producer_id),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_int(self.queue_id.value.data)
    printer.print_string(" ")
    printer.print_string_literal(self.token_register.data)
    printer.print_string(" ")
    printer.print_int(self.producer_id.value.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    queue_id = parser.parse_integer()
    token_register = parser.parse_str_literal()
    producer_id = parser.parse_integer()
    comment = _parse_optional_comment(parser)
    return cls(queue_id, token_register, producer_id, comment)


@irdl_op_definition
class StreamAcquireOp(IRDLOperation):
  name = "elenor.runtime.tile.stream.acquire"

  queue_id = prop_def(IntegerAttr)
  destination_token = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, queue_id: int, destination_token: str, comment: str | None = None):
    super().__init__(
      properties=_props(
        {
          "queue_id": _index_attr(queue_id),
          "destination_token": StringAttr(destination_token),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_int(self.queue_id.value.data)
    printer.print_string(" ")
    printer.print_string_literal(self.destination_token.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    queue_id = parser.parse_integer()
    destination_token = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(queue_id, destination_token, comment)


@irdl_op_definition
class StreamReleaseOp(IRDLOperation):
  name = "elenor.runtime.tile.stream.release"

  queue_id = prop_def(IntegerAttr)
  token_register = prop_def(StringAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, queue_id: int, token_register: str, comment: str | None = None):
    super().__init__(
      properties=_props(
        {
          "queue_id": _index_attr(queue_id),
          "token_register": StringAttr(token_register),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_int(self.queue_id.value.data)
    printer.print_string(" ")
    printer.print_string_literal(self.token_register.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    queue_id = parser.parse_integer()
    token_register = parser.parse_str_literal()
    comment = _parse_optional_comment(parser)
    return cls(queue_id, token_register, comment)


@irdl_op_definition
class StreamEosOp(IRDLOperation):
  name = "elenor.runtime.tile.stream.eos"

  queue_id = prop_def(IntegerAttr)
  producer_id = prop_def(IntegerAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, queue_id: int, producer_id: int, comment: str | None = None):
    super().__init__(
      properties=_props(
        {
          "queue_id": _index_attr(queue_id),
          "producer_id": _index_attr(producer_id),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_int(self.queue_id.value.data)
    printer.print_string(" ")
    printer.print_int(self.producer_id.value.data)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    queue_id = parser.parse_integer()
    producer_id = parser.parse_integer()
    comment = _parse_optional_comment(parser)
    return cls(queue_id, producer_id, comment)


@irdl_op_definition
class PatchDescriptorOp(IRDLOperation):
  name = "elenor.runtime.tile.patch.desc"

  args = prop_def(ArrayAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, args: tuple[ScalarParam, ...], comment: str | None = None):
    super().__init__(properties=_props({"args": _encode_array(args), "comment": _string_attr(comment)}))

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_attribute(self.args)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    args = _parse_array_attr(parser, "argument array")
    comment = _parse_optional_comment(parser)
    return cls(_decode_array(args), comment)


@irdl_op_definition
class LoadDescriptorOp(IRDLOperation):
  name = "elenor.runtime.tile.load.desc"

  destination = prop_def(StringAttr)
  args = prop_def(ArrayAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, destination: str, args: tuple[ScalarParam, ...], comment: str | None = None):
    super().__init__(
      properties=_props(
        {
          "destination": StringAttr(destination),
          "args": _encode_array(args),
          "comment": _string_attr(comment),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.destination.data)
    printer.print_string(" ")
    printer.print_attribute(self.args)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    destination = parser.parse_str_literal()
    args = _parse_array_attr(parser, "argument array")
    comment = _parse_optional_comment(parser)
    return cls(destination, _decode_array(args), comment)


@irdl_op_definition
class StoreDescriptorOp(IRDLOperation):
  name = "elenor.runtime.tile.store.desc"

  args = prop_def(ArrayAttr)
  comment = opt_prop_def(StringAttr)

  def __init__(self, args: tuple[ScalarParam, ...], comment: str | None = None):
    super().__init__(properties=_props({"args": _encode_array(args), "comment": _string_attr(comment)}))

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_attribute(self.args)
    _print_optional_comment(printer, self.comment)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    args = _parse_array_attr(parser, "argument array")
    comment = _parse_optional_comment(parser)
    return cls(_decode_array(args), comment)


@irdl_op_definition
class ProfileBeginOp(_CommentOnlyTileOp):
  name = "elenor.runtime.tile.prof.begin"


@irdl_op_definition
class ProfileEndOp(_CommentOnlyTileOp):
  name = "elenor.runtime.tile.prof.end"


@irdl_op_definition
class TrapOp(_CommentOnlyTileOp):
  name = "elenor.runtime.tile.trap"


operations: list[type[Operation]] = [
  BOADescriptorOp,
  EVUDescriptorOp,
  MFEDescriptorOp,
  USEDescriptorOp,
  TileGroupTaskOp,
  StreamDescOp,
  TileRoleBindingOp,
  TileProgramOp,
  InitStreamOp,
  GroupDMAPrefetchOp,
  GroupDMAStoreOp,
  DispatchRoleOp,
  GroupWaitEventOp,
  GroupBarrierOp,
  CollectiveRunOp,
  SignalEventOp,
  LabelOp,
  NopOp,
  MoveOp,
  AddOp,
  CompareOp,
  BranchOp,
  BranchPredicateOp,
  BranchEosOp,
  ReturnOp,
  LaunchBOAOp,
  LaunchEVUOp,
  LaunchMFEOp,
  LaunchUSEOp,
  DMALoadOp,
  DMAStoreOp,
  WaitOp,
  WaitAllOp,
  FenceOp,
  StreamPopOp,
  StreamPushOp,
  StreamAcquireOp,
  StreamReleaseOp,
  StreamEosOp,
  PatchDescriptorOp,
  LoadDescriptorOp,
  StoreDescriptorOp,
  ProfileBeginOp,
  ProfileEndOp,
  TrapOp,
]

Elenor = Dialect("elenor", operations, [])

__all__ = [
  "AddOp",
  "BOADescriptorOp",
  "BranchEosOp",
  "BranchOp",
  "BranchPredicateOp",
  "CollectiveRunOp",
  "CompareOp",
  "DMALoadOp",
  "DMAStoreOp",
  "DispatchRoleOp",
  "EVUDescriptorOp",
  "Elenor",
  "EngineDescriptorLike",
  "FenceOp",
  "GroupActionLike",
  "GroupBarrierOp",
  "GroupDMAPrefetchOp",
  "GroupDMAStoreOp",
  "GroupWaitEventOp",
  "InitStreamOp",
  "LabelOp",
  "LaunchBOAOp",
  "LaunchEVUOp",
  "LaunchMFEOp",
  "LaunchUSEOp",
  "LoadDescriptorOp",
  "MFEDescriptorOp",
  "MoveOp",
  "NopOp",
  "PatchDescriptorOp",
  "ProfileBeginOp",
  "ProfileEndOp",
  "ReturnOp",
  "ScalarParam",
  "SignalEventOp",
  "StoreDescriptorOp",
  "StreamAcquireOp",
  "StreamDescOp",
  "StreamEosOp",
  "StreamPopOp",
  "StreamPushOp",
  "StreamReleaseOp",
  "TileGroupTaskOp",
  "TileInstructionLike",
  "TileProgramOp",
  "TileRoleBindingOp",
  "TrapOp",
  "USEDescriptorOp",
  "WaitAllOp",
  "WaitOp",
  "_decode_array",
  "_decode_params",
  "_decode_scalar",
]
