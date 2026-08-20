"""ELENOR xDSL dialect for pipeline validator workload IR.

Function-call style IR (see reference.mlir): the module contains top-level
named definitions - ``nest.context @name { ... }`` and ``tile.program @name
{ ... }`` - and the context body dispatches tile programs by symbol
reference.  Async engine/DMA ops produce SSA event values typed
``!nest.event<tag>`` / ``!tile.event<tag>``; ``await`` consumes them.

Key design points mirrored from reference.mlir:

  - placement (tile mask) lives on ``nest.context``, not on dispatch;
  - dispatch consumes a logical task range plus ins/outs L2 buffers and a
    ``depends_on`` event, and returns THREE aggregated events:
    ``grid_done`` / ``input_released`` / ``output_ready``;
  - tile programs mark their L2-read / L2-write phases with
    ``tile.signal``, which drives the dispatch phase events;
  - buffers are SSA values (``!nest.l2_buffer<slot>``), and the event type
    tag doubles as the runtime event id shared by the simulator and trace.

Prefix hierarchy: ``tile.*`` (tile level), ``nest.*`` (group level),
``nexus.*`` (host level, deferred - not yet implemented).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self, TypeAlias

from xdsl.dialects.builtin import (
  ArrayAttr,
  IndexType,
  IntegerAttr,
  StringAttr,
)
from xdsl.ir import (
  Attribute,
  Block,
  Dialect,
  Operation,
  ParametrizedAttribute,
  Region,
  TypeAttribute,
)
from xdsl.irdl import (
  IRDLOperation,
  irdl_attr_definition,
  irdl_op_definition,
  operand_def,
  opt_prop_def,
  prop_def,
  region_def,
  result_def,
  traits_def,
  var_operand_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import NoTerminator

# ---------------------------------------------------------------------------
# SSA types: events, L2 buffers, task range
# ---------------------------------------------------------------------------


@irdl_attr_definition
class NestEvent(ParametrizedAttribute, TypeAttribute):
  """Group-level async event: ``!nest.event<tag>``.

  The tag is the runtime event id shared by the simulator and the trace.
  """

  name = "nest.event"
  tag: StringAttr


@irdl_attr_definition
class NexusEvent(ParametrizedAttribute, TypeAttribute):
  """Device-level async event: ``!nexus.event<"tag">``.

  Tag is the runtime event id shared by the device scheduler and trace.
  """
  name = "nexus.event"
  tag: StringAttr

@irdl_attr_definition
class TileEvent(ParametrizedAttribute, TypeAttribute):
  """Tile-level async event: ``!tile.event<tag>``."""
  name = "tile.event"
  tag: StringAttr

@irdl_attr_definition
class NestBuffer(ParametrizedAttribute, TypeAttribute):
  """Context-owned L2 buffer: ``!nest.l2_buffer<slot>``.

  The slot name is the L2 slot id used by the group DMA latency model.
  """

  name = "nest.l2_buffer"
  slot: StringAttr


@irdl_attr_definition
class NestGlobalMemref(ParametrizedAttribute, TypeAttribute):
  """Declarative global memref: ``!nest.global_memref<4x128x128xbf16>``.

  V1: parse/print only — no runtime data movement.  Used as a
  declaration-only entry-block argument type for ``nexus.program``.
  """

  name = "nest.global_memref"
  dims: ArrayAttr  # list of IntegerAttr(IndexType)
  dtype: StringAttr

  @staticmethod
  def of(dims: Sequence[int], dtype: str) -> NestGlobalMemref:
    return NestGlobalMemref(
      ArrayAttr([_index_attr(d) for d in dims]), StringAttr(dtype))

  @classmethod
  def parse_parameters(cls, parser: AttrParser) -> list:
    # The lexer treats ``4x128x128xbf16`` as ``4`` then a single identifier
    # ``x128x128xbf16`` (BARE_IDENT), so we parse the first integer and the
    # trailing identifier, then split on ``x``: dims + dtype.
    parser.parse_punctuation("<")
    first = parser.parse_integer()
    rest = parser.parse_identifier()
    parser.parse_punctuation(">")
    # rest starts with 'x': 'x128x128xbf16' -> split -> ['', '128', '128', 'bf16']
    parts = rest.split("x")
    dims = [first] + [int(p) for p in parts[1:-1]]
    dtype = parts[-1]
    return [ArrayAttr([_index_attr(d) for d in dims]), StringAttr(dtype)]

  def print_parameters(self, printer: Printer) -> None:
    printer.print_string("<")
    printer.print_string("x".join(str(d.value.data) for d in self.dims.data))  # type: ignore[attr-defined]
    printer.print_string("x" + self.dtype.data + ">")

@irdl_attr_definition
class TaskRange(ParametrizedAttribute, TypeAttribute):
  """Logical task domain: ``!nest.task_range``.

  Logical task ids are NOT physical Tile ids (reference.mlir section 3).
  """

  name = "nest.task_range"


def _index_attr(value: int) -> IntegerAttr:
  return IntegerAttr(value, IndexType())


def _props(mapping: dict[str, Attribute | None]) -> dict[str, Attribute]:
  return {k: v for k, v in mapping.items() if v is not None}


def _single_block_region(ops: Sequence[IRDLOperation]) -> Region:
  return Region([Block(list(ops))])


def _print_int_kw(printer: Printer, keyword: str, value: int) -> None:
  printer.print_string(f" {keyword} = ")
  printer.print_int(value)


def _parse_int_kw(parser: Parser, keyword: str) -> int:
  parser.parse_keyword(keyword)
  parser.parse_punctuation("=")
  return parser.parse_integer()


def _parse_opt_int_kw(parser: Parser, keyword: str) -> int | None:
  if parser.parse_optional_keyword(keyword) is None:
    return None
  parser.parse_punctuation("=")
  return parser.parse_integer()


def _print_str_kw(printer: Printer, keyword: str, value: str) -> None:
  printer.print_string(f" {keyword} = ")
  printer.print_string_literal(value)


def _parse_str_kw(parser: Parser, keyword: str) -> str:
  parser.parse_keyword(keyword)
  parser.parse_punctuation("=")
  return parser.parse_str_literal()


def _print_region(printer: Printer, region: Region) -> None:
  printer.print_string(" ")
  printer.print_region(region, print_empty_block=False)


def _parse_region_ops(parser: Parser) -> list:
  region = parser.parse_region()
  if not region.blocks:
    region.add_block(Block())
  if len(region.blocks) != 1:
    parser.raise_error("expected exactly one block in region")
  ops = list(region.blocks[0].ops)
  for op in ops:
    op.detach()
  return ops


def _parse_event_type(parser: AttrParser, cls: type[Attribute]) -> Attribute:
  parser.parse_punctuation(":")
  attr = parser.parse_attribute()
  if not isinstance(attr, cls):
    parser.raise_error(f"expected {cls.name} type")
  return attr


def _print_event_type(printer: Printer, event_type: Attribute) -> None:
  printer.print_string(" : ")
  printer.print_attribute(event_type)


def _parse_symbol(parser: Parser) -> str:
  return parser.parse_symbol_name().data


def _print_symbol(printer: Printer, sym: str) -> None:
  printer.print_string(" ")
  printer.print_symbol_name(sym)


def _parse_operand_group(parser: Parser, keyword: str) -> list:
  parser.parse_keyword(keyword)
  return parser.parse_comma_separated_list(
    parser.Delimiter.PAREN, parser.parse_operand, f" in {keyword}(...) operand list"
  )


def _print_operand_group(printer: Printer, keyword: str, operands: Sequence) -> None:
  printer.print_string(f" {keyword}(")
  for i, operand in enumerate(operands):
    if i:
      printer.print_string(", ")
    printer.print_operand(operand)
  printer.print_string(")")


def _parse_depends_on(parser: Parser) -> list:
  if parser.parse_optional_keyword("depends_on") is None:
    return []
  return list(
    parser.parse_comma_separated_list(
      parser.Delimiter.PAREN, parser.parse_operand, " in depends_on(...) operand list"
    )
  )


def _print_depends_on(printer: Printer, operands: Sequence) -> None:
  if not operands:
    return
  _print_operand_group(printer, "depends_on", operands)


NestActionLike: TypeAlias = (  # noqa: UP040
  "NestAllocOp | NestTaskRangeOp | NestPrefetchOp | NestDMAStoreOp"
  " | NestDispatchOp | NestCollectiveOp | NestReleaseOp | NestAwaitOp"
  " | NestBarrierOp | NestReturnOp"
)
TileActionLike: TypeAlias = (  # noqa: UP040
  "TileLoadOp | TileStoreOp | TilePowOp | TileEvuOp | TileBoaOp"
  " | TileAwaitOp | TileSignalOp | TileReturnOp"
)
NexusActionLike: TypeAlias = (  # noqa: UP040
  "NexusSubmitContextOp | NexusAwaitOp | NexusReturnOp"
)


# ---------------------------------------------------------------------------
# Top-level definitions
# ---------------------------------------------------------------------------


@irdl_op_definition
class NestContextOp(IRDLOperation):
  """``nest.context @name placement = M context = N { ... }`` - one tile-group context.

  ``placement`` is the tile-group placement mask (reference.mlir
  ``#nest.tile_group<mask = 0xF>``): the physical tile set every dispatch
  in this context runs on.

  Optional ``context = N`` pins this context to **device execution slot N**
  when submitted via ``nexus.submit_context.async`` (mirrors the UCE
  context pin of ``nest.dispatch.tasks.async`` one level up).  Omitted =
  first available slot; occupied slot = submission waits (backpressure).
  Legal range ``0..device_context_count-1``; out-of-range rejected at
  model load.  In a legacy single-context module the pin selects the
  (only) slot and must be 0.
  """

  name = "nest.context"

  sym_name = prop_def(StringAttr)
  placement = prop_def(IntegerAttr)
  context_id = opt_prop_def(IntegerAttr)
  completion_event = prop_def(StringAttr, default_value=StringAttr("context_done"))

  body = region_def("single_block")

  traits = traits_def(NoTerminator())

  def __init__(
    self,
    sym_name: str,
    body: Sequence[NestActionLike],
    placement: int = 0x0F,
    completion_event: str = "context_done",
    context_id: int | None = None,
  ):
    super().__init__(
      properties=_props(
        {
          "sym_name": StringAttr(sym_name),
          "placement": _index_attr(placement),
          "context_id": None if context_id is None else _index_attr(context_id),
          "completion_event": StringAttr(completion_event),
        }
      ),
      regions=[_single_block_region(list(body))],
    )

  def print(self, printer: Printer) -> None:
    _print_symbol(printer, self.sym_name.data)
    _print_int_kw(printer, "placement", self.placement.value.data)
    if self.context_id is not None:
      _print_int_kw(printer, "context", self.context_id.value.data)
    if self.completion_event.data != "context_done":
      _print_str_kw(printer, "completion", self.completion_event.data)
    _print_region(printer, self.body)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    sym_name = _parse_symbol(parser)
    placement = _parse_int_kw(parser, "placement")
    context_id = _parse_opt_int_kw(parser, "context")
    completion_event = "context_done"
    if parser.parse_optional_keyword("completion") is not None:
      parser.parse_punctuation("=")
      completion_event = parser.parse_str_literal()
    body = _parse_region_ops(parser)
    return cls(sym_name, body, placement=placement,
               completion_event=completion_event, context_id=context_id)


@irdl_op_definition
class TileProgramDefOp(IRDLOperation):
  """``tile.program @name { ... }`` - one tile program definition."""

  name = "tile.program"

  sym_name = prop_def(StringAttr)

  body = region_def("single_block")

  traits = traits_def(NoTerminator())

  def __init__(self, sym_name: str, body: Sequence[TileActionLike]):
    super().__init__(
      properties=_props({"sym_name": StringAttr(sym_name)}),
      regions=[_single_block_region(list(body))],
    )

  def print(self, printer: Printer) -> None:
    _print_symbol(printer, self.sym_name.data)
    _print_region(printer, self.body)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    sym_name = _parse_symbol(parser)
    body = _parse_region_ops(parser)
    return cls(sym_name, body)


# ---------------------------------------------------------------------------
# nest.* context-body actions
# ---------------------------------------------------------------------------


@irdl_op_definition
class NestAllocOp(IRDLOperation):
  """``%b = nest.alloc slot = "s" bytes = N : !nest.l2_buffer<s>``.

  Context-owned L2 buffer (reference.mlir section 1).
  """

  name = "nest.alloc"

  slot = prop_def(StringAttr)
  bytes_total = prop_def(IntegerAttr)

  result = result_def(NestBuffer)

  def __init__(self, slot: str, bytes_total: int):
    super().__init__(
      result_types=[NestBuffer(StringAttr(slot))],
      properties=_props(
        {
          "slot": StringAttr(slot),
          "bytes_total": _index_attr(bytes_total),
        }
      ),
    )
    self.result.name_hint = slot

  def print(self, printer: Printer) -> None:
    _print_str_kw(printer, "slot", self.slot.data)
    _print_int_kw(printer, "bytes", self.bytes_total.value.data)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    slot = _parse_str_kw(parser, "slot")
    bytes_total = _parse_int_kw(parser, "bytes")
    buffer_type = _parse_event_type(parser, NestBuffer)
    if buffer_type.slot.data != slot:  # type: ignore[attr-defined]
      parser.raise_error("nest.alloc slot and !nest.l2_buffer tag must match")
    return cls(slot, bytes_total)


@irdl_op_definition
class NestTaskRangeOp(IRDLOperation):
  """``%t = nest.task.range from = 0 to = 4 : !nest.task_range``.

  Logical task domain (reference.mlir section 3): task ids are logical,
  not physical Tile ids.
  """

  name = "nest.task.range"

  from_task = prop_def(IntegerAttr)
  to_task = prop_def(IntegerAttr)

  result = result_def(TaskRange)

  def __init__(self, from_task: int, to_task: int):
    super().__init__(
      result_types=[TaskRange()],
      properties=_props(
        {
          "from_task": _index_attr(from_task),
          "to_task": _index_attr(to_task),
        }
      ),
    )

  @property
  def num_tasks(self) -> int:
    return self.to_task.value.data - self.from_task.value.data

  def print(self, printer: Printer) -> None:
    _print_int_kw(printer, "from", self.from_task.value.data)
    _print_int_kw(printer, "to", self.to_task.value.data)
    printer.print_string(" : ")
    printer.print_attribute(self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    from_task = _parse_int_kw(parser, "from")
    to_task = _parse_int_kw(parser, "to")
    parser.parse_punctuation(":")
    attr = parser.parse_attribute()
    if not isinstance(attr, TaskRange):
      parser.raise_error("expected !nest.task_range type")
    return cls(from_task, to_task)


class _NestAsyncOp(IRDLOperation):
  """Base for nest-body async ops producing ``!nest.event<tag>``."""

  result = result_def(NestEvent)

  def _finish(self, tag: str, **kwargs) -> None:
    super().__init__(result_types=[NestEvent(StringAttr(tag))], **kwargs)
    self.result.name_hint = tag


@irdl_op_definition
class NestPrefetchOp(_NestAsyncOp):
  """``%e = nest.dma.prefetch.async %buf bytes = N : !nest.event<t>``

  HBM -> L2 prefetch into the context-owned buffer.
  """

  name = "nest.dma.prefetch.async"

  buffer = operand_def(NestBuffer)
  bytes_total = prop_def(IntegerAttr)

  def __init__(self, buffer, bytes_total: int, tag: str):
    self._finish(
      tag,
      properties=_props({"bytes_total": _index_attr(bytes_total)}),
      operands=[buffer],
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_operand(self.buffer)
    _print_int_kw(printer, "bytes", self.bytes_total.value.data)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    buffer = parser.parse_operand()
    bytes_total = _parse_int_kw(parser, "bytes")
    event_type = _parse_event_type(parser, NestEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(buffer, bytes_total, tag)


@irdl_op_definition
class NestDMAStoreOp(_NestAsyncOp):
  """``%e = nest.dma.store.async %buf bytes = N depends_on(%o) : !nest.event<t>``

  L2 -> HBM final store, gated on the dispatch ``output_ready`` event.
  """

  name = "nest.dma.store.async"

  buffer = operand_def(NestBuffer)
  bytes_total = prop_def(IntegerAttr)
  depends_on = var_operand_def(NestEvent)

  def __init__(self, buffer, bytes_total: int, tag: str, depends_on: Sequence = ()):
    self._finish(
      tag,
      properties=_props({"bytes_total": _index_attr(bytes_total)}),
      operands=[[buffer], list(depends_on)],
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_operand(self.buffer)
    _print_int_kw(printer, "bytes", self.bytes_total.value.data)
    _print_depends_on(printer, self.depends_on)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    buffer = parser.parse_operand()
    bytes_total = _parse_int_kw(parser, "bytes")
    depends_on = _parse_depends_on(parser)
    event_type = _parse_event_type(parser, NestEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(buffer, bytes_total, tag, depends_on=depends_on)


@irdl_op_definition
class NestDispatchOp(IRDLOperation):
  """``%grid, %inrel, %out = nest.dispatch.tasks.async @prog``

  ``tasks(%t) ins(%b) outs(%b) depends_on(%e) : (three !nest.event types)``

  Function-call dispatch per reference.mlir section 4: the tile program is
  referenced by symbol; the placement comes from the enclosing
  ``nest.context``.  Returns three aggregated events:

    - grid_done      - all logical tasks returned;
    - input_released - all tasks completed their L2 read phase
                       (``tile.signal input_released``);
    - output_ready   - all tasks completed their L2 write phase
                       (``tile.signal output_ready``).

  Optional ``context = N`` pins every task of this dispatch to the
  tile-local UCE context index ``N`` (same index on every tile in the
  placement, not a physical tile id).  Omitted = first available
  context (existing behaviour).  Legal range is ``0..context_count-1``;
  out-of-range is rejected at task load, not at IR verify.
  """

  name = "nest.dispatch.tasks.async"

  program = prop_def(StringAttr)
  context_id = opt_prop_def(IntegerAttr)
  tasks = operand_def(TaskRange)
  ins = operand_def(NestBuffer)
  outs = operand_def(NestBuffer)
  depends_on = var_operand_def(NestEvent)

  grid_done = result_def(NestEvent)
  input_released = result_def(NestEvent)
  output_ready = result_def(NestEvent)

  def __init__(
    self,
    program: str,
    tasks,
    ins,
    outs,
    grid_tag: str,
    inrel_tag: str,
    outready_tag: str,
    depends_on: Sequence = (),
    context_id: int | None = None,
  ):
    super().__init__(
      result_types=[
        NestEvent(StringAttr(grid_tag)),
        NestEvent(StringAttr(inrel_tag)),
        NestEvent(StringAttr(outready_tag)),
      ],
      properties=_props({
        "program": StringAttr(program),
        "context_id": None if context_id is None else _index_attr(context_id),
      }),
      operands=[[tasks], ins, outs, list(depends_on)],
    )
    self.grid_done.name_hint = grid_tag
    if inrel_tag:
      self.input_released.name_hint = inrel_tag
    if outready_tag:
      self.output_ready.name_hint = outready_tag

  def print(self, printer: Printer) -> None:
    _print_symbol(printer, self.program.data)
    if self.context_id is not None:
      _print_int_kw(printer, "context", self.context_id.value.data)
    printer.print_string(" tasks(")
    printer.print_operand(self.tasks)
    printer.print_string(")")
    printer.print_string(" ins(")
    printer.print_operand(self.ins)
    printer.print_string(") outs(")
    printer.print_operand(self.outs)
    printer.print_string(")")
    _print_depends_on(printer, self.depends_on)
    printer.print_string(" : (")
    for i, result in enumerate(self.results):
      if i:
        printer.print_string(", ")
      printer.print_attribute(result.type)
    printer.print_string(")")

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    program = _parse_symbol(parser)
    context_id = _parse_opt_int_kw(parser, "context")
    tasks = _parse_operand_group(parser, "tasks")
    parser.parse_keyword("ins")
    parser.parse_punctuation("(")
    ins_op = parser.parse_operand()
    parser.parse_punctuation(")")
    parser.parse_keyword("outs")
    parser.parse_punctuation("(")
    outs_op = parser.parse_operand()
    parser.parse_punctuation(")")
    depends_on = _parse_depends_on(parser)
    if len(tasks) != 1:
      parser.raise_error("dispatch tasks(...) expects exactly one task range")
    parser.parse_punctuation(":")
    parser.parse_punctuation("(")
    types = parser.parse_comma_separated_list(
      parser.Delimiter.NONE, parser.parse_attribute, " in dispatch result types"
    )
    parser.parse_punctuation(")")
    if len(types) != 3 or not all(isinstance(t, NestEvent) for t in types):
      parser.raise_error("dispatch expects three !nest.event results")
    tags = [t.tag.data for t in types]  # type: ignore[attr-defined]
    return cls(program, tasks[0], ins_op, outs_op, tags[0], tags[1], tags[2],
               depends_on=depends_on, context_id=context_id)


@irdl_op_definition
class NestCollectiveOp(_NestAsyncOp):
  """``%e = nest.collective.async "reduce" bytes = N mask = M : !nest.event<t>``"""

  name = "nest.collective.async"

  collective = prop_def(StringAttr)
  bytes_total = prop_def(IntegerAttr)
  participant_mask = prop_def(IntegerAttr)

  def __init__(self, collective: str, bytes_total: int, participant_mask: int, tag: str):
    self._finish(
      tag,
      properties=_props(
        {
          "collective": StringAttr(collective),
          "bytes_total": _index_attr(bytes_total),
          "participant_mask": _index_attr(participant_mask),
        }
      ),
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.collective.data)
    _print_int_kw(printer, "bytes", self.bytes_total.value.data)
    _print_int_kw(printer, "mask", self.participant_mask.value.data)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    collective = parser.parse_str_literal()
    bytes_total = _parse_int_kw(parser, "bytes")
    participant_mask = _parse_int_kw(parser, "mask")
    event_type = _parse_event_type(parser, NestEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(collective, bytes_total, participant_mask, tag)


@irdl_op_definition
class NestReleaseOp(IRDLOperation):
  """``nest.release %buf depends_on(%e)`` - reclaim the context-owned buffer."""

  name = "nest.release"

  buffer = operand_def(NestBuffer)
  depends_on = var_operand_def(NestEvent)

  def __init__(self, buffer, depends_on: Sequence = ()):
    super().__init__(operands=[[buffer], list(depends_on)])

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_operand(self.buffer)
    _print_depends_on(printer, self.depends_on)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    buffer = parser.parse_operand()
    depends_on = _parse_depends_on(parser)
    return cls(buffer, depends_on=depends_on)


@irdl_op_definition
class NestAwaitOp(IRDLOperation):
  """``nest.await %e0, %e1`` - wait for one or more nest events."""

  name = "nest.await"

  events = var_operand_def(NestEvent)

  def __init__(self, events: Sequence):
    super().__init__(operands=[list(events)])

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    for i, operand in enumerate(self.events):
      if i:
        printer.print_string(", ")
      printer.print_operand(operand)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    events = parser.parse_comma_separated_list(
      parser.Delimiter.NONE, parser.parse_operand, " in nest.await event list"
    )
    return cls(events)


@irdl_op_definition
class NestBarrierOp(IRDLOperation):
  """``nest.barrier`` - group barrier."""

  name = "nest.barrier"

  def __init__(self):
    super().__init__()

  def print(self, printer: Printer) -> None:
    pass

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    return cls()


@irdl_op_definition
class NestReturnOp(IRDLOperation):
  """``nest.return`` - context completion (signals the completion event)."""

  name = "nest.return"

  def __init__(self):
    super().__init__()

  def print(self, printer: Printer) -> None:
    pass

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    return cls()
# ---------------------------------------------------------------------------
# nexus.* model-level ops
# ---------------------------------------------------------------------------


@irdl_op_definition
class NexusProgramOp(IRDLOperation):
  """``nexus.program @name (%a : !nest.global_memref<...>) { ... }`` - model entry.

  Body: ``nexus.submit_context.async`` / ``nexus.await`` / ``nexus.return``.
  V1: entry block args are declarative (parsed/printed); body ops must
  not reference them (no memref plumbing yet).  The block carries real
  args (``Block(..., arg_types=...)``); the signature is printed from
  those args, and the region is printed with
  ``print_entry_block_args=False`` so no duplicate ``^bb0`` label appears.
  """

  name = "nexus.program"
  sym_name = prop_def(StringAttr)
  body = region_def("single_block")
  traits = traits_def(NoTerminator())

  def __init__(self, sym_name: str, body: Sequence, arg_types: Sequence = ()):
    super().__init__(
      properties=_props({"sym_name": StringAttr(sym_name)}),
      regions=[Region([Block(list(body), arg_types=list(arg_types))])],
    )

  def print(self, printer: Printer) -> None:
    _print_symbol(printer, self.sym_name.data)
    block = self.body.block
    if block.args:
      printer.print_string(" (")
      for i, arg in enumerate(block.args):
        if i:
          printer.print_string(", ")
        printer.print_block_argument(arg)
      printer.print_string(")")
    printer.print_string(" ")
    printer.print_region(self.body, print_entry_block_args=False, print_empty_block=False)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    sym_name = _parse_symbol(parser)
    # Parse signature args without registering SSA names
    arguments = []
    if parser.parse_optional_punctuation("(") is not None:
      while True:
        arg = parser.parse_optional_argument()
        if arg is not None:
          arguments.append(arg)
        if parser.parse_optional_punctuation(",") is None:
          break
      parser.parse_punctuation(")")
    # Parse the region, passing arguments so parse_region creates real
    # block args and registers them.
    region = parser.parse_region(arguments=arguments if arguments else None)
    if not region.blocks:
      region.add_block(Block())
    if len(region.blocks) != 1:
      parser.raise_error("expected exactly one block in region")
    block = region.blocks[0]
    arg_types = [a.type for a in block.args]
    ops = list(block.ops)
    for op in ops:
      op.detach()
    return cls(sym_name, ops, arg_types=arg_types)


@irdl_op_definition
class NexusSubmitContextOp(IRDLOperation):
  """``%e = nexus.submit_context.async @ctx : !nexus.event<"tag">``"""

  name = "nexus.submit_context.async"
  context_sym = prop_def(StringAttr)
  result = result_def(NexusEvent)

  def __init__(self, context_sym: str, tag: str):
    super().__init__(
      result_types=[NexusEvent(StringAttr(tag))],
      properties=_props({"context_sym": StringAttr(context_sym)}),
    )
    self.result.name_hint = tag

  def print(self, printer: Printer) -> None:
    _print_symbol(printer, self.context_sym.data)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    context_sym = _parse_symbol(parser)
    event_type = _parse_event_type(parser, NexusEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(context_sym, tag)


@irdl_op_definition
class NexusAwaitOp(IRDLOperation):
  """``nexus.await %e0, %e1`` - wait for one or more nexus events."""

  name = "nexus.await"
  events = var_operand_def(NexusEvent)

  def __init__(self, events: Sequence):
    super().__init__(operands=[list(events)])

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    for i, operand in enumerate(self.events):
      if i:
        printer.print_string(", ")
      printer.print_operand(operand)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    events = parser.parse_comma_separated_list(
      parser.Delimiter.NONE, parser.parse_operand, " in nexus.await event list"
    )
    return cls(events)


@irdl_op_definition
class NexusReturnOp(IRDLOperation):
  """``nexus.return`` - model completion."""

  name = "nexus.return"

  def __init__(self):
    super().__init__()

  def print(self, printer: Printer) -> None:
    pass

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    return cls()

# ---------------------------------------------------------------------------
# tile.* program-body actions
# ---------------------------------------------------------------------------


class _TileAsyncOp(IRDLOperation):
  """Base for tile-body async ops producing ``!tile.event<tag>``."""

  result = result_def(TileEvent)

  def _finish(self, tag: str, **kwargs) -> None:
    super().__init__(result_types=[TileEvent(StringAttr(tag))], **kwargs)
    self.result.name_hint = tag


@irdl_op_definition
class TileLoadOp(_TileAsyncOp):
  """``%e = tile.load.async bytes = N : !tile.event<t>`` - MFE L2->L1 load."""

  name = "tile.load.async"

  bytes_total = prop_def(IntegerAttr)

  def __init__(self, bytes_total: int, tag: str):
    self._finish(tag, properties=_props({"bytes_total": _index_attr(bytes_total)}))

  def print(self, printer: Printer) -> None:
    _print_int_kw(printer, "bytes", self.bytes_total.value.data)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    bytes_total = _parse_int_kw(parser, "bytes")
    event_type = _parse_event_type(parser, TileEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(bytes_total, tag)


@irdl_op_definition
class TileStoreOp(_TileAsyncOp):
  """``%e = tile.store.async bytes = N : !tile.event<t>`` - MFE L1->L2 store."""

  name = "tile.store.async"

  bytes_total = prop_def(IntegerAttr)

  def __init__(self, bytes_total: int, tag: str):
    self._finish(tag, properties=_props({"bytes_total": _index_attr(bytes_total)}))

  def print(self, printer: Printer) -> None:
    _print_int_kw(printer, "bytes", self.bytes_total.value.data)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    bytes_total = _parse_int_kw(parser, "bytes")
    event_type = _parse_event_type(parser, TileEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(bytes_total, tag)


@irdl_op_definition
class TilePowOp(_TileAsyncOp):
  """``%e = tile.pow.async bytes = N exponent = E ops = K : !tile.event<t>``"""

  name = "tile.pow.async"

  bytes_total = prop_def(IntegerAttr)
  exponent = prop_def(IntegerAttr)
  pow_ops = prop_def(IntegerAttr)

  def __init__(self, bytes_total: int, exponent: int, pow_ops: int, tag: str):
    self._finish(
      tag,
      properties=_props(
        {
          "bytes_total": _index_attr(bytes_total),
          "exponent": _index_attr(exponent),
          "pow_ops": _index_attr(pow_ops),
        }
      ),
    )

  def print(self, printer: Printer) -> None:
    _print_int_kw(printer, "bytes", self.bytes_total.value.data)
    _print_int_kw(printer, "exponent", self.exponent.value.data)
    _print_int_kw(printer, "ops", self.pow_ops.value.data)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    bytes_total = _parse_int_kw(parser, "bytes")
    exponent = _parse_int_kw(parser, "exponent")
    pow_ops = _parse_int_kw(parser, "ops")
    event_type = _parse_event_type(parser, TileEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(bytes_total, exponent, pow_ops, tag)


@irdl_op_definition
class TileEvuOp(_TileAsyncOp):
  """``%e = tile.evu.async "relu" ops = K : !tile.event<t>`` - generic EVU op."""

  name = "tile.evu.async"

  op_name = prop_def(StringAttr)
  evu_ops = prop_def(IntegerAttr)

  def __init__(self, op_name: str, evu_ops: int, tag: str):
    self._finish(
      tag,
      properties=_props(
        {
          "op_name": StringAttr(op_name),
          "evu_ops": _index_attr(evu_ops),
        }
      ),
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.op_name.data)
    _print_int_kw(printer, "ops", self.evu_ops.value.data)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    op_name = parser.parse_str_literal()
    evu_ops = _parse_int_kw(parser, "ops")
    event_type = _parse_event_type(parser, TileEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(op_name, evu_ops, tag)


@irdl_op_definition
class TileBoaOp(_TileAsyncOp):
  """``%e = tile.boa.async "matmul" m = M n = N k = K ops = S : !tile.event<t>``"""

  name = "tile.boa.async"

  op_name = prop_def(StringAttr)
  m = prop_def(IntegerAttr)
  n = prop_def(IntegerAttr)
  k = prop_def(IntegerAttr)
  boa_ops = prop_def(IntegerAttr)
  accumulate = opt_prop_def(IntegerAttr)

  def __init__(
    self,
    op_name: str,
    m: int,
    n: int,
    k: int,
    boa_ops: int,
    tag: str,
    accumulate: bool = False,
  ):
    self._finish(
      tag,
      properties=_props(
        {
          "op_name": StringAttr(op_name),
          "m": _index_attr(m),
          "n": _index_attr(n),
          "k": _index_attr(k),
          "boa_ops": _index_attr(boa_ops),
          "accumulate": None if not accumulate else _index_attr(1),
        }
      ),
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string_literal(self.op_name.data)
    _print_int_kw(printer, "m", self.m.value.data)
    _print_int_kw(printer, "n", self.n.value.data)
    _print_int_kw(printer, "k", self.k.value.data)
    _print_int_kw(printer, "ops", self.boa_ops.value.data)
    if self.accumulate is not None:
      printer.print_string(" accumulate")
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    op_name = parser.parse_str_literal()
    m = _parse_int_kw(parser, "m")
    n = _parse_int_kw(parser, "n")
    k = _parse_int_kw(parser, "k")
    boa_ops = _parse_int_kw(parser, "ops")
    accumulate = parser.parse_optional_keyword("accumulate") is not None
    event_type = _parse_event_type(parser, TileEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(op_name, m, n, k, boa_ops, tag, accumulate=accumulate)


@irdl_op_definition
class TileAwaitOp(IRDLOperation):
  """``tile.await %e0, %e1`` - wait for one or more tile events."""

  name = "tile.await"

  events = var_operand_def(TileEvent)

  def __init__(self, events: Sequence):
    super().__init__(operands=[list(events)])

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    for i, operand in enumerate(self.events):
      if i:
        printer.print_string(", ")
      printer.print_operand(operand)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    events = parser.parse_comma_separated_list(
      parser.Delimiter.NONE, parser.parse_operand, " in tile.await event list"
    )
    return cls(events)


@irdl_op_definition
class TileSignalOp(IRDLOperation):
  """``tile.signal input_released`` / ``tile.signal output_ready``.

  Phase signal (reference.mlir tile.signal): when every dispatched task of
  one dispatch has signalled a phase, the corresponding dispatch result
  event fires.  ``input_released`` = this task will not read its L2 input
  subview again; ``output_ready`` = this task's output is visible in L2.
  """

  name = "tile.signal"

  phase = prop_def(StringAttr)

  PHASES: ClassVar[tuple[str, ...]] = ("input_released", "output_ready")

  def __init__(self, phase: str):
    super().__init__(properties=_props({"phase": StringAttr(phase)}))

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string(self.phase.data)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    phase = parser.parse_identifier()
    if phase not in cls.PHASES:
      parser.raise_error(f"unknown tile.signal phase '{phase}'")
    return cls(phase)


@irdl_op_definition
class TileReturnOp(IRDLOperation):
  """``tile.return`` - tile program completion (contributes to grid_done)."""

  name = "tile.return"

  def __init__(self):
    super().__init__()

  def print(self, printer: Printer) -> None:
    pass

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    return cls()


operations: list[type[Operation]] = [
  NestContextOp,
  NestAllocOp,
  NestTaskRangeOp,
  NestPrefetchOp,
  NestDMAStoreOp,
  NestDispatchOp,
  NestCollectiveOp,
  NestReleaseOp,
  NestAwaitOp,
  NestBarrierOp,
  NestReturnOp,
  TileProgramDefOp,
  TileLoadOp,
  TileStoreOp,
  TilePowOp,
  TileEvuOp,
  TileBoaOp,
  TileAwaitOp,
  TileSignalOp,
  TileReturnOp,
  NexusProgramOp,
  NexusSubmitContextOp,
  NexusAwaitOp,
  NexusReturnOp,
]

Elenor = Dialect(
  "elenor", operations, [NestEvent, TileEvent, NestBuffer, TaskRange, NexusEvent, NestGlobalMemref])

__all__ = [
  "Elenor",
  "NestActionLike",
  "NestAllocOp",
  "NestAwaitOp",
  "NestBarrierOp",
  "NestBuffer",
  "NestCollectiveOp",
  "NestContextOp",
  "NestDMAStoreOp",
  "NestDispatchOp",
  "NestEvent",
  "NestGlobalMemref",
  "NestPrefetchOp",
  "NestReleaseOp",
  "NestReturnOp",
  "NestTaskRangeOp",
  "NexusActionLike",
  "NexusAwaitOp",
  "NexusEvent",
  "NexusProgramOp",
  "NexusReturnOp",
  "NexusSubmitContextOp",
  "TaskRange",
  "TileActionLike",
  "TileAwaitOp",
  "TileBoaOp",
  "TileEvent",
  "TileEvuOp",
  "TileLoadOp",
  "TilePowOp",
  "TileProgramDefOp",
  "TileReturnOp",
  "TileSignalOp",
  "TileStoreOp",
]
