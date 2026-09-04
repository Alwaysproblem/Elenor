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
  - global inputs flow as real SSA values: ``nexus.program`` block args
    → ``nexus.submit_context.async`` actuals → ``nest.context`` formals
    → ``nest.subview`` → prefetch/store; tile-side L2 formals →
    ``tile.subview`` → load/store.  Bytes derive from view/buffer shapes,
    not from a ``bytes`` property on transfer ops.

Prefix hierarchy: ``tile.*`` (tile level), ``nest.*`` (group level),
``nexus.*`` (host level).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, Self, TypeAlias, cast

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
  AttrSizedOperandSegments,
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
# Element byte widths and shape-type parse/print helpers
# ---------------------------------------------------------------------------

DTYPE_BYTES: dict[str, int] = {"i8": 1, "bf16": 2, "f16": 2, "i32": 4, "f32": 4}
"""Element byte widths accepted by every shape-typed ELENOR type."""

def _parse_dims_dtype(parser: AttrParser) -> tuple[list[int], str]:
  """Parse ``<Dx...xDtype>`` returning (dims, dtype).

  The lexer treats ``4x128x128xbf16`` as ``4`` then a single BARE_IDENT
  ``x128x128xbf16``; we split on ``x`` to recover dims + dtype.
  """
  parser.parse_punctuation("<")
  first = parser.parse_integer()
  rest = parser.parse_identifier()
  parser.parse_punctuation(">")
  parts = rest.split("x")
  dims = [first] + [int(p) for p in parts[1:-1]]
  dtype = parts[-1]
  if dtype not in DTYPE_BYTES:
    parser.raise_error(f"unknown dtype '{dtype}'")
  return dims, dtype


def _print_dims_dtype(printer: Printer, dims: ArrayAttr, dtype: str) -> None:
  printer.print_string("<")
  printer.print_string("x".join(str(d.value.data) for d in dims.data))  # type: ignore[attr-defined]
  printer.print_string("x" + dtype + ">")


def _int_list(arr: ArrayAttr) -> list[int]:
  return [int(d.value.data) for d in arr.data]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# SSA types: events, buffers, views, task handles
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
  """Context-owned L2 buffer: ``!nest.l2_buffer<4x128x128xbf16>``.

  Shape-typed; the L2 slot id lives on the defining ``nest.alloc``.
  """

  name = "nest.l2_buffer"
  dims: ArrayAttr
  dtype: StringAttr

  @staticmethod
  def of(dims: Sequence[int], dtype: str) -> NestBuffer:
    return NestBuffer(
      ArrayAttr([_index_attr(d) for d in dims]), StringAttr(dtype))

  @classmethod
  def parse_parameters(cls, parser: AttrParser) -> list:
    dims, dtype = _parse_dims_dtype(parser)
    return [ArrayAttr([_index_attr(d) for d in dims]), StringAttr(dtype)]

  def print_parameters(self, printer: Printer) -> None:
    _print_dims_dtype(printer, self.dims, self.dtype.data)


@irdl_attr_definition
class NestGlobalMemref(ParametrizedAttribute, TypeAttribute):
  """Declarative global memref: ``!nest.global_memref<4x128x128xbf16>``.

  Host-visible global input; appears as a ``nexus.program`` or
  ``nest.context`` block-arg formal.
  """

  name = "nest.global_memref"
  dims: ArrayAttr
  dtype: StringAttr

  @staticmethod
  def of(dims: Sequence[int], dtype: str) -> NestGlobalMemref:
    return NestGlobalMemref(
      ArrayAttr([_index_attr(d) for d in dims]), StringAttr(dtype))

  @classmethod
  def parse_parameters(cls, parser: AttrParser) -> list:
    dims, dtype = _parse_dims_dtype(parser)
    return [ArrayAttr([_index_attr(d) for d in dims]), StringAttr(dtype)]

  def print_parameters(self, printer: Printer) -> None:
    _print_dims_dtype(printer, self.dims, self.dtype.data)


@irdl_attr_definition
class NestGlobalView(ParametrizedAttribute, TypeAttribute):
  """Logical view of a global memref: ``!nest.global_view<4x128x128xbf16>``.

  Produced by ``nest.subview``; consumed by prefetch/store.
  """

  name = "nest.global_view"
  dims: ArrayAttr
  dtype: StringAttr

  @staticmethod
  def of(dims: Sequence[int], dtype: str) -> NestGlobalView:
    return NestGlobalView(
      ArrayAttr([_index_attr(d) for d in dims]), StringAttr(dtype))

  @classmethod
  def parse_parameters(cls, parser: AttrParser) -> list:
    dims, dtype = _parse_dims_dtype(parser)
    return [ArrayAttr([_index_attr(d) for d in dims]), StringAttr(dtype)]

  def print_parameters(self, printer: Printer) -> None:
    _print_dims_dtype(printer, self.dims, self.dtype.data)


@irdl_attr_definition
class NestL2View(ParametrizedAttribute, TypeAttribute):
  """Logical per-task view of an L2 buffer: ``!nest.l2_view<1x128x128xbf16>``.

  Produced by ``tile.subview``; consumed by tile.load/store.
  """

  name = "nest.l2_view"
  dims: ArrayAttr
  dtype: StringAttr

  @staticmethod
  def of(dims: Sequence[int], dtype: str) -> NestL2View:
    return NestL2View(
      ArrayAttr([_index_attr(d) for d in dims]), StringAttr(dtype))

  @classmethod
  def parse_parameters(cls, parser: AttrParser) -> list:
    dims, dtype = _parse_dims_dtype(parser)
    return [ArrayAttr([_index_attr(d) for d in dims]), StringAttr(dtype)]

  def print_parameters(self, printer: Printer) -> None:
    _print_dims_dtype(printer, self.dims, self.dtype.data)


@irdl_attr_definition
class TileL1Buffer(ParametrizedAttribute, TypeAttribute):
  """Tile-local L1 buffer: ``!tile.l1_buffer<128x128xbf16>``.

  Produced by ``tile.alloc``; consumed by tile.load (dst) / store (src).
  """

  name = "tile.l1_buffer"
  dims: ArrayAttr
  dtype: StringAttr

  @staticmethod
  def of(dims: Sequence[int], dtype: str) -> TileL1Buffer:
    return TileL1Buffer(
      ArrayAttr([_index_attr(d) for d in dims]), StringAttr(dtype))

  @classmethod
  def parse_parameters(cls, parser: AttrParser) -> list:
    dims, dtype = _parse_dims_dtype(parser)
    return [ArrayAttr([_index_attr(d) for d in dims]), StringAttr(dtype)]

  def print_parameters(self, printer: Printer) -> None:
    _print_dims_dtype(printer, self.dims, self.dtype.data)


@irdl_attr_definition
class NestTask(ParametrizedAttribute, TypeAttribute):
  """Logical task handle: ``!nest.task``.

  First formal of every ``tile.program``; ``tile.subview task = %task``
  offsets a view by the logical task id along ``task_dim``.
  """
  name = "nest.task"


@irdl_attr_definition
class TaskRange(ParametrizedAttribute, TypeAttribute):
  """Logical task domain: ``!nest.task_range``.

  Logical task ids are NOT physical Tile ids (reference.mlir section 3).
  """

  name = "nest.task_range"


@irdl_attr_definition
class NestAggregate(ParametrizedAttribute):
  """Signal aggregation policy: ``#nest.aggregate<all_tasks>``.

  Declares how one dispatch aggregates ``tile.signal`` phase events
  (PR 3).  V1 supports only ``all_tasks``: the phase result fires when
  every logical task of the dispatch's task range has signalled.
  """

  name = "nest.aggregate"

  mode: StringAttr

  MODES: ClassVar[tuple[str, ...]] = ("all_tasks",)

  @staticmethod
  def of(mode: str) -> NestAggregate:
    if mode not in NestAggregate.MODES:
      raise ValueError(f"unsupported nest.aggregate mode '{mode}'")
    return NestAggregate(StringAttr(mode))

  @classmethod
  def parse_parameters(cls, parser: AttrParser) -> list:
    parser.parse_punctuation("<")
    mode = parser.parse_identifier()
    parser.parse_punctuation(">")
    if mode not in cls.MODES:
      parser.raise_error(f"unsupported nest.aggregate mode '{mode}'")
    return [StringAttr(mode)]

  def print_parameters(self, printer: Printer) -> None:
    printer.print_string(f"<{self.mode.data}>")


# ---------------------------------------------------------------------------
# Shared parse/print helpers
# ---------------------------------------------------------------------------


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


def _print_int_list_kw(printer: Printer, keyword: str, values: Sequence[int]) -> None:
  printer.print_string(f" {keyword} = [")
  for i, value in enumerate(values):
    if i:
      printer.print_string(", ")
    printer.print_int(value)
  printer.print_string("]")


def _parse_int_list_kw(parser: Parser, keyword: str) -> list[int]:
  parser.parse_keyword(keyword)
  parser.parse_punctuation("=")
  return list(
    parser.parse_comma_separated_list(
      parser.Delimiter.SQUARE, parser.parse_integer, f" in {keyword} = [...] list"
    )
  )


def _print_str_kw(printer: Printer, keyword: str, value: str) -> None:
  printer.print_string(f" {keyword} = ")
  printer.print_string_literal(value)


def _parse_str_kw(parser: Parser, keyword: str) -> str:
  parser.parse_keyword(keyword)
  parser.parse_punctuation("=")
  return parser.parse_str_literal()


def _parse_opt_str_kw(parser: Parser, keyword: str) -> str | None:
  if parser.parse_optional_keyword(keyword) is None:
    return None
  parser.parse_punctuation("=")
  return parser.parse_str_literal()

def _print_body_region(printer: Printer, region: Region) -> None:
  printer.print_string(" ")
  printer.print_region(region, print_entry_block_args=False, print_empty_block=False)

def _parse_body_region(parser: Parser, arguments: list | None = None) -> Region:
  """Parse one single-block region, preserving SSA identity of block args.

  Returns the Region intact (block args + ops still attached) so that
  body ops referencing block-arg formals keep their SSA identity.
  """
  region = parser.parse_region(arguments=arguments if arguments else None)
  if not region.blocks:
    region.add_block(Block())
  if len(region.blocks) != 1:
    parser.raise_error("expected exactly one block in region")
  return region




def _parse_block_args(parser: Parser) -> list:
  """Parse an optional ``(%a : type, %b : type)`` signature."""
  arguments: list = []
  if parser.parse_optional_punctuation("(") is None:
    return arguments
  while True:
    arg = parser.parse_optional_argument()
    if arg is not None:
      arguments.append(arg)
    if parser.parse_optional_punctuation(",") is None:
      break
  parser.parse_punctuation(")")
  return arguments


def _print_block_args(printer: Printer, block: Block) -> None:
  if not block.args:
    return
  printer.print_string(" (")
  for i, arg in enumerate(block.args):
    if i:
      printer.print_string(", ")
    printer.print_block_argument(arg)
  printer.print_string(")")


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


def _parse_signal_policy(parser: Parser) -> dict[str, str]:
  """Parse the mandatory ``signal_policy { ... }`` block (PR 3).

  Entries are ``<phase> = #nest.aggregate<mode>`` in any order; the
  printer always emits ``input_released`` before ``output_ready``.
  """
  parser.parse_keyword("signal_policy")
  parser.parse_punctuation("{")
  policy: dict[str, str] = {}
  while parser.parse_optional_punctuation("}") is None:
    phase = parser.parse_identifier()
    if phase not in TileSignalOp.PHASES:
      parser.raise_error(f"unknown signal policy phase '{phase}'")
    if phase in policy:
      parser.raise_error(f"duplicate signal policy phase '{phase}'")
    parser.parse_punctuation("=")
    attr = parser.parse_attribute()
    if not isinstance(attr, NestAggregate):
      parser.raise_error(
        f"signal policy phase '{phase}' expects a #nest.aggregate attribute")
    policy[phase] = attr.mode.data
    parser.parse_optional_punctuation(",")
  return policy


def _print_signal_policy(printer: Printer, op: NestDispatchOp) -> None:
  printer.print_string(" signal_policy {")
  policy = op.signal_policy
  entries = [p for p in TileSignalOp.PHASES if p in policy]
  for i, phase in enumerate(entries):
    if i:
      printer.print_string(", ")
    printer.print_string(f" {phase} = #nest.aggregate<{policy[phase]}>")
  if entries:
    printer.print_string(" ")
  printer.print_string("}")


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


def _set_arg_names(block: Block, arg_names: Sequence[str]) -> None:
  for arg, name in zip(block.args, arg_names):
    if name:
      arg.name_hint = name


NestActionLike: TypeAlias = (  # noqa: UP040
  "NestAllocOp | NestTaskRangeOp | NestSubviewOp | NestPrefetchOp"
  " | NestDMAStoreOp | NestDispatchOp | NestCollectiveOp | NestReleaseOp"
  " | NestAwaitOp | NestBarrierOp | NestReturnOp"
)
TileActionLike: TypeAlias = (  # noqa: UP040
  "TileSubviewOp | TileAllocOp | TileLoadOp | TileStoreOp | TileGatherOp"
  " | TilePowOp | TileEvuOp | TileBoaOp | TileAwaitOp | TileSignalOp"
  " | TileReturnOp"
)
NexusActionLike: TypeAlias = (  # noqa: UP040
  "NexusSubmitContextOp | NexusAwaitOp | NexusReturnOp"
)


# ---------------------------------------------------------------------------
# Top-level definitions
# ---------------------------------------------------------------------------


@irdl_op_definition
class NestContextOp(IRDLOperation):
  """``nest.context @name(%Y : !nest.global_memref<...>) placement = M ... { ... }``.

  One tile-group context.  Entry block args are the context's global
  formals (HBM inputs); the body dispatches tile programs by symbol
  reference.
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
    body: Sequence[NestActionLike] = (),
    placement: int = 0x0F,
    completion_event: str = "context_done",
    context_id: int | None = None,
    arg_types: Sequence = (),
    arg_names: Sequence[str] = (),
    _region: Region | None = None,
  ):
    if _region is not None:
      region = _region
    else:
      region = Region([Block(list(body), arg_types=list(arg_types))])
      _set_arg_names(region.blocks[0], arg_names)
    super().__init__(
      properties=_props(
        {
          "sym_name": StringAttr(sym_name),
          "placement": _index_attr(placement),
          "context_id": None if context_id is None else _index_attr(context_id),
          "completion_event": StringAttr(completion_event),
        }
      ),
      regions=[region],
    )

  def print(self, printer: Printer) -> None:
    _print_symbol(printer, self.sym_name.data)
    _print_block_args(printer, self.body.block)
    _print_int_kw(printer, "placement", self.placement.value.data)
    if self.context_id is not None:
      _print_int_kw(printer, "context", self.context_id.value.data)
    if self.completion_event.data != "context_done":
      _print_str_kw(printer, "completion", self.completion_event.data)
    _print_body_region(printer, self.body)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    sym_name = _parse_symbol(parser)
    arguments = _parse_block_args(parser)
    placement = _parse_int_kw(parser, "placement")
    context_id = _parse_opt_int_kw(parser, "context")
    completion_event = "context_done"
    if parser.parse_optional_keyword("completion") is not None:
      parser.parse_punctuation("=")
      completion_event = parser.parse_str_literal()
    region = _parse_body_region(parser, arguments)
    return cls(
      sym_name, placement=placement,
      completion_event=completion_event, context_id=context_id,
      _region=region,
    )

@irdl_op_definition
class TileProgramDefOp(IRDLOperation):
  """``tile.program @name(%task : !nest.task, %global : !nest.global_view<...>, ...)``.

  The first formal must be ``!nest.task``.  It is followed by zero or
  more ``!nest.global_view`` formals, then zero or more
  ``!nest.l2_buffer`` formals; global and L2 formals may not interleave.
  """

  name = "tile.program"

  sym_name = prop_def(StringAttr)

  body = region_def("single_block")

  traits = traits_def(NoTerminator())

  def __init__(self, sym_name: str, body: Sequence[TileActionLike] = (),
               arg_types: Sequence = (), arg_names: Sequence[str] = (),
               _region: Region | None = None):
    if _region is not None:
      region = _region
    else:
      region = Region([Block(list(body), arg_types=list(arg_types))])
      _set_arg_names(region.blocks[0], arg_names)
    super().__init__(
      properties=_props({"sym_name": StringAttr(sym_name)}),
      regions=[region],
    )

  def print(self, printer: Printer) -> None:
    _print_symbol(printer, self.sym_name.data)
    _print_block_args(printer, self.body.block)
    _print_body_region(printer, self.body)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    sym_name = _parse_symbol(parser)
    arguments = _parse_block_args(parser)
    region = _parse_body_region(parser, arguments)
    return cls(sym_name, _region=region)
# ---------------------------------------------------------------------------
# nest.* context-body actions
# ---------------------------------------------------------------------------


@irdl_op_definition
class NestAllocOp(IRDLOperation):
  """``%b = nest.alloc slot = "s" role = "inout" shape = [..] dtype = "bf16" : !nest.l2_buffer<..>``.

  Context-owned L2 buffer (reference.mlir section 1).  ``slot`` is the
  runtime L2 object id; bytes derive from shape x dtype.
  """

  name = "nest.alloc"

  slot = prop_def(StringAttr)
  role = prop_def(StringAttr)
  shape = prop_def(ArrayAttr)
  dtype = prop_def(StringAttr)
  alignment = opt_prop_def(IntegerAttr)

  result = result_def(NestBuffer)

  def __init__(self, slot: str, role: str, shape: Sequence[int], dtype: str,
               alignment: int | None = None):
    super().__init__(
      result_types=[NestBuffer.of(shape, dtype)],
      properties=_props(
        {
          "slot": StringAttr(slot),
          "role": StringAttr(role),
          "shape": ArrayAttr([_index_attr(d) for d in shape]),
          "dtype": StringAttr(dtype),
          "alignment": None if alignment is None else _index_attr(alignment),
        }
      ),
    )
    self.result.name_hint = slot

  def print(self, printer: Printer) -> None:
    _print_str_kw(printer, "slot", self.slot.data)
    _print_str_kw(printer, "role", self.role.data)
    _print_int_list_kw(printer, "shape", _int_list(self.shape))
    _print_str_kw(printer, "dtype", self.dtype.data)
    if self.alignment is not None:
      _print_int_kw(printer, "alignment", self.alignment.value.data)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    slot = _parse_str_kw(parser, "slot")
    role = _parse_str_kw(parser, "role")
    shape = _parse_int_list_kw(parser, "shape")
    dtype = _parse_str_kw(parser, "dtype")
    alignment = _parse_opt_int_kw(parser, "alignment")
    buffer_type = cast(NestBuffer, _parse_event_type(parser, NestBuffer))
    buf_dims = _int_list(buffer_type.dims)
    if buf_dims != shape or buffer_type.dtype.data != dtype:
      parser.raise_error("nest.alloc shape/dtype and !nest.l2_buffer type must match")
    return cls(slot, role, shape, dtype, alignment=alignment)


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


@irdl_op_definition
class NestSubviewOp(IRDLOperation):
  """``%v = nest.subview %Y offsets = [..] sizes = [..] strides = [..] : !nest.global_view<..>``.

  Logical view of one context global formal (HBM side of a transfer).
  The source must be a ``nest.context`` block arg; V1 disallows view chains.
  """

  name = "nest.subview"

  src = operand_def(NestGlobalMemref)
  offsets = prop_def(ArrayAttr)
  sizes = prop_def(ArrayAttr)
  strides = prop_def(ArrayAttr)

  result = result_def(NestGlobalView)

  def __init__(self, src, offsets: Sequence[int], sizes: Sequence[int],
               strides: Sequence[int], view_type: NestGlobalView):
    super().__init__(
      operands=[src],
      result_types=[view_type],
      properties=_props(
        {
          "offsets": ArrayAttr([_index_attr(v) for v in offsets]),
          "sizes": ArrayAttr([_index_attr(v) for v in sizes]),
          "strides": ArrayAttr([_index_attr(v) for v in strides]),
        }
      ),
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_operand(self.src)
    _print_int_list_kw(printer, "offsets", _int_list(self.offsets))
    _print_int_list_kw(printer, "sizes", _int_list(self.sizes))
    _print_int_list_kw(printer, "strides", _int_list(self.strides))
    printer.print_string(" : ")
    printer.print_attribute(self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    src = parser.parse_operand()
    offsets = _parse_int_list_kw(parser, "offsets")
    sizes = _parse_int_list_kw(parser, "sizes")
    strides = _parse_int_list_kw(parser, "strides")
    parser.parse_punctuation(":")
    attr = parser.parse_attribute()
    if not isinstance(attr, NestGlobalView):
      parser.raise_error("expected !nest.global_view type")
    return cls(src, offsets, sizes, strides, attr)


class _NestAsyncOp(IRDLOperation):
  """Base for nest-body async ops producing ``!nest.event<tag>``."""

  result = result_def(NestEvent)

  def _finish(self, tag: str, **kwargs) -> None:
    super().__init__(result_types=[NestEvent(StringAttr(tag))], **kwargs)
    self.result.name_hint = tag


@irdl_op_definition
class NestPrefetchOp(_NestAsyncOp):
  """``%e = nest.dma.prefetch.async %src into %dst : !nest.event<t>``

  HBM -> L2 prefetch from a global view into the context-owned buffer.
  """

  name = "nest.dma.prefetch.async"

  src = operand_def(NestGlobalView)
  dst = operand_def(NestBuffer)

  def __init__(self, src, dst, tag: str):
    self._finish(tag, operands=[src, dst])

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_operand(self.src)
    printer.print_string(" into ")
    printer.print_operand(self.dst)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    src = parser.parse_operand()
    parser.parse_keyword("into")
    dst = parser.parse_operand()
    event_type = _parse_event_type(parser, NestEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(src, dst, tag)


@irdl_op_definition
class NestDMAStoreOp(_NestAsyncOp):
  """``%e = nest.dma.store.async %src into %dst depends_on(%o) : !nest.event<t>``

  L2 -> HBM final store, gated on the dispatch ``output_ready`` event.
  """

  name = "nest.dma.store.async"

  src = operand_def(NestBuffer)
  dst = operand_def(NestGlobalView)
  depends_on = var_operand_def(NestEvent)

  def __init__(self, src, dst, tag: str, depends_on: Sequence = ()):

    self._finish(tag, operands=[src, dst, list(depends_on)])
  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_operand(self.src)
    printer.print_string(" into ")
    printer.print_operand(self.dst)
    _print_depends_on(printer, self.depends_on)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    src = parser.parse_operand()
    parser.parse_keyword("into")
    dst = parser.parse_operand()
    depends_on = _parse_depends_on(parser)
    event_type = _parse_event_type(parser, NestEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(src, dst, tag, depends_on=depends_on)


@irdl_op_definition
class NestDispatchOp(IRDLOperation):
  """``%grid, %inrel, %out = nest.dispatch.tasks.async @prog``

  ``tasks(%t) globals(%g...) ins(%b...) outs(%b...) signal_policy { ... }``

  Function-call dispatch per reference.mlir section 4: the tile program is
  referenced by symbol; the placement comes from the enclosing
  ``nest.context``.  ``globals`` bind positionally to the tile program's
  global-view formals.  ``ins`` and ``outs`` each bind positionally to all
  L2 formals.  Formal 0 is always ``!nest.task``.
  Returns three aggregated events:

    - grid_done      - all logical tasks returned;
    - input_released - all tasks completed their L2 read phase
                       (``tile.signal input_released(%task)``);
    - output_ready   - all tasks completed their L2 write phase
                       (``tile.signal output_ready(%task)``).

  ``signal_policy`` declares, per phase, how the dispatch aggregates
  task signals (``#nest.aggregate<all_tasks>``).  The block is always
  printed, possibly empty; a phase with a policy entry must carry a
  non-empty result tag and a phase without entry an empty tag.

  Optional ``context = N`` pins every task of this dispatch to the
  tile-local UCE context index ``N``.
  """

  name = "nest.dispatch.tasks.async"

  irdl_options = (AttrSizedOperandSegments(),)

  program = prop_def(StringAttr)
  context_id = opt_prop_def(IntegerAttr)
  input_released_policy = opt_prop_def(NestAggregate)
  output_ready_policy = opt_prop_def(NestAggregate)
  tasks = operand_def(TaskRange)
  global_views = var_operand_def(NestGlobalView)
  ins = var_operand_def(NestBuffer)
  outs = var_operand_def(NestBuffer)
  depends_on = var_operand_def(NestEvent)

  grid_done = result_def(NestEvent)
  input_released = result_def(NestEvent)
  output_ready = result_def(NestEvent)

  def __init__(
    self,
    program: str,
    tasks,
    global_views,
    ins,
    outs,
    grid_tag: str,
    inrel_tag: str,
    outready_tag: str,
    *,
    signal_policy: Mapping[str, str],
    depends_on: Sequence = (),
    context_id: int | None = None,
  ):
    for phase, mode in signal_policy.items():
      if phase not in TileSignalOp.PHASES:
        raise ValueError(f"unknown signal policy phase '{phase}'")
      if mode not in NestAggregate.MODES:
        raise ValueError(f"unsupported signal policy mode '{mode}'")
    super().__init__(
      result_types=[
        NestEvent(StringAttr(grid_tag)),
        NestEvent(StringAttr(inrel_tag)),
        NestEvent(StringAttr(outready_tag)),
      ],
      properties=_props({
        "program": StringAttr(program),
        "context_id": None if context_id is None else _index_attr(context_id),
        "input_released_policy": (
          None if "input_released" not in signal_policy
          else NestAggregate.of(signal_policy["input_released"])),
        "output_ready_policy": (
          None if "output_ready" not in signal_policy
          else NestAggregate.of(signal_policy["output_ready"])),
      }),
      operands=[[tasks], list(global_views), list(ins), list(outs), list(depends_on)],
    )
    self.grid_done.name_hint = grid_tag
    if inrel_tag:
      self.input_released.name_hint = inrel_tag
    if outready_tag:
      self.output_ready.name_hint = outready_tag

  @property
  def signal_policy(self) -> dict[str, str]:
    """Declared phase → aggregation mode (source contract, PR 3)."""
    policy: dict[str, str] = {}
    if self.input_released_policy is not None:
      policy["input_released"] = self.input_released_policy.mode.data
    if self.output_ready_policy is not None:
      policy["output_ready"] = self.output_ready_policy.mode.data
    return policy

  def print(self, printer: Printer) -> None:
    _print_symbol(printer, self.program.data)
    if self.context_id is not None:
      _print_int_kw(printer, "context", self.context_id.value.data)
    printer.print_string(" tasks(")
    printer.print_operand(self.tasks)
    printer.print_string(")")
    _print_operand_group(printer, "globals", self.global_views)
    _print_operand_group(printer, "ins", self.ins)
    _print_operand_group(printer, "outs", self.outs)
    _print_signal_policy(printer, self)
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
    global_ops = _parse_operand_group(parser, "globals")
    ins_ops = _parse_operand_group(parser, "ins")
    outs_ops = _parse_operand_group(parser, "outs")
    signal_policy = _parse_signal_policy(parser)
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
    return cls(
      program,
      tasks[0],
      global_ops,
      ins_ops,
      outs_ops,
      tags[0],
      tags[1],
      tags[2],
      signal_policy=signal_policy,
      depends_on=depends_on,
      context_id=context_id,
    )


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
  Entry block args are the model's named global inputs; submit ops pass
  them as actuals to ``nest.context`` formals.
  """

  name = "nexus.program"
  sym_name = prop_def(StringAttr)
  body = region_def("single_block")
  traits = traits_def(NoTerminator())

  def __init__(self, sym_name: str, body: Sequence = (), arg_types: Sequence = (),
               arg_names: Sequence[str] = (), _region: Region | None = None):
    if _region is not None:
      region = _region
    else:
      region = Region([Block(list(body), arg_types=list(arg_types))])
      _set_arg_names(region.blocks[0], arg_names)
    super().__init__(
      properties=_props({"sym_name": StringAttr(sym_name)}),
      regions=[region],
    )

  def print(self, printer: Printer) -> None:
    _print_symbol(printer, self.sym_name.data)
    _print_block_args(printer, self.body.block)
    printer.print_string(" ")
    printer.print_region(self.body, print_entry_block_args=False, print_empty_block=False)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    sym_name = _parse_symbol(parser)
    arguments = _parse_block_args(parser)
    region = _parse_body_region(parser, arguments)
    return cls(sym_name, _region=region)

@irdl_op_definition
class NexusSubmitContextOp(IRDLOperation):
  """``%e = nexus.submit_context.async @ctx(%Y) : !nexus.event<"tag">``

  Submits a ``nest.context`` for execution, passing global-input actuals
  that bind positionally to the context's formals.  Zero-actual form
  ``@ctx : ...`` is accepted for legacy modules without global inputs.
  """

  name = "nexus.submit_context.async"
  context_sym = prop_def(StringAttr)
  actuals = var_operand_def(NestGlobalMemref)
  result = result_def(NexusEvent)

  def __init__(self, context_sym: str, tag: str, actuals: Sequence = ()):
    super().__init__(
      result_types=[NexusEvent(StringAttr(tag))],
      properties=_props({"context_sym": StringAttr(context_sym)}),
      operands=[list(actuals)],
    )
    self.result.name_hint = tag

  def print(self, printer: Printer) -> None:
    _print_symbol(printer, self.context_sym.data)
    if self.actuals:
      printer.print_string("(")
      for i, operand in enumerate(self.actuals):
        if i:
          printer.print_string(", ")
        printer.print_operand(operand)
      printer.print_string(")")
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    context_sym = _parse_symbol(parser)
    actuals: list = []
    if parser.parse_optional_punctuation("(") is not None:
      actuals = list(
        parser.parse_comma_separated_list(
          parser.Delimiter.NONE, parser.parse_operand, " in submit actuals list"
        )
      )
      parser.parse_punctuation(")")
    event_type = _parse_event_type(parser, NexusEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(context_sym, tag, actuals=actuals)


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
class TileSubviewOp(IRDLOperation):
  """``%v = tile.subview %l2_buf task = %task task_dim = 0 offsets = [..] sizes = [..] : !nest.l2_view<..>``

  Logical per-task view of one tile.program L2 formal.  ``task_dim``
  (requires the ``task = %task`` operand) adds the logical task id to
  ``offsets[task_dim]``.  V1 strides must be unit; V1 disallows view chains.
  """

  name = "tile.subview"

  irdl_options = (AttrSizedOperandSegments(),)

  src = operand_def(NestBuffer)
  task = var_operand_def(NestTask)
  task_dim = opt_prop_def(IntegerAttr)
  offsets = prop_def(ArrayAttr)
  sizes = prop_def(ArrayAttr)
  strides = prop_def(ArrayAttr)

  result = result_def(NestL2View)

  def __init__(self, src, task, task_dim: int | None,
               offsets: Sequence[int], sizes: Sequence[int],
               strides: Sequence[int], view_type: NestL2View):
    super().__init__(
      operands=[[src], [task] if task is not None else []],
      result_types=[view_type],

      properties=_props(
        {
          "task_dim": None if task_dim is None else _index_attr(task_dim),
          "offsets": ArrayAttr([_index_attr(v) for v in offsets]),
          "sizes": ArrayAttr([_index_attr(v) for v in sizes]),
          "strides": ArrayAttr([_index_attr(v) for v in strides]),
        }
      ),
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_operand(self.src)
    if self.task:
      printer.print_string(" task = ")
      printer.print_operand(self.task[0])
    if self.task_dim is not None:
      _print_int_kw(printer, "task_dim", self.task_dim.value.data)
    _print_int_list_kw(printer, "offsets", _int_list(self.offsets))
    _print_int_list_kw(printer, "sizes", _int_list(self.sizes))
    _print_int_list_kw(printer, "strides", _int_list(self.strides))
    printer.print_string(" : ")
    printer.print_attribute(self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    src = parser.parse_operand()
    task = None
    if parser.parse_optional_keyword("task") is not None:
      parser.parse_punctuation("=")
      task = parser.parse_operand()
    task_dim = _parse_opt_int_kw(parser, "task_dim")
    offsets = _parse_int_list_kw(parser, "offsets")
    sizes = _parse_int_list_kw(parser, "sizes")
    strides = _parse_int_list_kw(parser, "strides")
    parser.parse_punctuation(":")
    attr = parser.parse_attribute()
    if not isinstance(attr, NestL2View):
      parser.raise_error("expected !nest.l2_view type")
    return cls(src, task, task_dim, offsets, sizes, strides, attr)


@irdl_op_definition
class TileAllocOp(IRDLOperation):
  """``%l1 = tile.alloc shape = [..] dtype = "bf16" alignment = 256 : !tile.l1_buffer<..>``

  Tile-local L1 scratch buffer.  Bytes derive from shape x dtype.
  """

  name = "tile.alloc"

  shape = prop_def(ArrayAttr)
  dtype = prop_def(StringAttr)
  alignment = opt_prop_def(IntegerAttr)

  result = result_def(TileL1Buffer)

  def __init__(self, shape: Sequence[int], dtype: str, alignment: int | None = None):
    super().__init__(
      result_types=[TileL1Buffer.of(shape, dtype)],
      properties=_props(
        {
          "shape": ArrayAttr([_index_attr(d) for d in shape]),
          "dtype": StringAttr(dtype),
          "alignment": None if alignment is None else _index_attr(alignment),
        }
      ),
    )

  def print(self, printer: Printer) -> None:
    _print_int_list_kw(printer, "shape", _int_list(self.shape))
    _print_str_kw(printer, "dtype", self.dtype.data)
    if self.alignment is not None:
      _print_int_kw(printer, "alignment", self.alignment.value.data)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    shape = _parse_int_list_kw(parser, "shape")
    dtype = _parse_str_kw(parser, "dtype")
    alignment = _parse_opt_int_kw(parser, "alignment")
    buf_type = cast(TileL1Buffer, _parse_event_type(parser, TileL1Buffer))
    buf_dims = _int_list(buf_type.dims)
    if buf_dims != shape or buf_type.dtype.data != dtype:
      parser.raise_error("tile.alloc shape/dtype and !tile.l1_buffer type must match")
    return cls(shape, dtype, alignment=alignment)


@irdl_op_definition
class TileLoadOp(_TileAsyncOp):
  """``%e = tile.load.async %src into %dst : !tile.event<t>`` - MFE L2->L1 load."""

  name = "tile.load.async"

  src = operand_def(NestL2View)
  dst = operand_def(TileL1Buffer)

  def __init__(self, src, dst, tag: str):
    self._finish(tag, operands=[src, dst])

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_operand(self.src)
    printer.print_string(" into ")
    printer.print_operand(self.dst)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    src = parser.parse_operand()
    parser.parse_keyword("into")
    dst = parser.parse_operand()
    event_type = _parse_event_type(parser, TileEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(src, dst, tag)


@irdl_op_definition
class TileStoreOp(_TileAsyncOp):
  """``%e = tile.store.async %src into %dst : !tile.event<t>`` - MFE L1->L2 store."""

  name = "tile.store.async"

  src = operand_def(TileL1Buffer)
  dst = operand_def(NestL2View)

  def __init__(self, src, dst, tag: str):
    self._finish(tag, operands=[src, dst])

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_operand(self.src)
    printer.print_string(" into ")
    printer.print_operand(self.dst)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    src = parser.parse_operand()
    parser.parse_keyword("into")
    dst = parser.parse_operand()
    event_type = _parse_event_type(parser, TileEvent)
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(src, dst, tag)


@irdl_op_definition
class TileProfiledAccessOp(IRDLOperation):
  """One deterministic profiled request inside ``tile.gather.global.async``."""

  name = "tile.profiled.access"

  request_id = prop_def(StringAttr)
  outcome = prop_def(StringAttr)
  bytes = prop_def(IntegerAttr)
  line_token = opt_prop_def(StringAttr)
  merge_group = opt_prop_def(StringAttr)

  def __init__(
    self,
    request_id: str,
    outcome: str,
    num_bytes: int,
    line_token: str | None = None,
    merge_group: str | None = None,
  ):
    super().__init__(
      properties=_props(
        {
          "request_id": StringAttr(request_id),
          "outcome": StringAttr(outcome),
          "bytes": _index_attr(num_bytes),
          "line_token": None if line_token is None else StringAttr(line_token),
          "merge_group": None if merge_group is None else StringAttr(merge_group),
        }
      )
    )

  def print(self, printer: Printer) -> None:
    _print_str_kw(printer, "id", self.request_id.data)
    _print_str_kw(printer, "outcome", self.outcome.data)
    _print_int_kw(printer, "bytes", self.bytes.value.data)
    if self.line_token is not None:
      _print_str_kw(printer, "line", self.line_token.data)
    if self.merge_group is not None:
      _print_str_kw(printer, "merge", self.merge_group.data)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    request_id = _parse_str_kw(parser, "id")
    outcome = _parse_str_kw(parser, "outcome")
    num_bytes = _parse_int_kw(parser, "bytes")
    line_token = _parse_opt_str_kw(parser, "line")
    merge_group = _parse_opt_str_kw(parser, "merge")
    return cls(request_id, outcome, num_bytes, line_token=line_token, merge_group=merge_group)


@irdl_op_definition
class TileGatherOp(_TileAsyncOp):
  """Deterministic profiled global gather into one tile-local L1 buffer."""

  name = "tile.gather.global.async"

  source = operand_def(NestGlobalView)
  indices = operand_def(TileL1Buffer)
  destination = operand_def(TileL1Buffer)
  result_bytes = prop_def(IntegerAttr)
  cache_min_bytes = prop_def(IntegerAttr)
  cache_target_bytes = prop_def(IntegerAttr)
  l1_mshr_hint = prop_def(IntegerAttr)
  profile = region_def("single_block")

  traits = traits_def(NoTerminator())

  def __init__(
    self,
    source,
    indices,
    destination,
    result_bytes: int,
    cache_min_bytes: int,
    cache_target_bytes: int,
    l1_mshr_hint: int,
    accesses: Sequence[TileProfiledAccessOp],
    tag: str,
    _region: Region | None = None,
  ):
    profile = _single_block_region(accesses) if _region is None else _region
    self._finish(
      tag,
      operands=[source, indices, destination],
      properties=_props(
        {
          "result_bytes": _index_attr(result_bytes),
          "cache_min_bytes": _index_attr(cache_min_bytes),
          "cache_target_bytes": _index_attr(cache_target_bytes),
          "l1_mshr_hint": _index_attr(l1_mshr_hint),
        }
      ),
      regions=[profile],
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_operand(self.source)
    printer.print_string(" indices(")
    printer.print_operand(self.indices)
    printer.print_string(") into ")
    printer.print_operand(self.destination)
    _print_int_kw(printer, "result_bytes", self.result_bytes.value.data)
    _print_int_kw(printer, "cache_min_bytes", self.cache_min_bytes.value.data)
    _print_int_kw(printer, "cache_target_bytes", self.cache_target_bytes.value.data)
    _print_int_kw(printer, "l1_mshr_hint", self.l1_mshr_hint.value.data)
    _print_body_region(printer, self.profile)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    source = parser.parse_operand()
    indices = _parse_operand_group(parser, "indices")
    parser.parse_keyword("into")
    destination = parser.parse_operand()
    result_bytes = _parse_int_kw(parser, "result_bytes")
    cache_min_bytes = _parse_int_kw(parser, "cache_min_bytes")
    cache_target_bytes = _parse_int_kw(parser, "cache_target_bytes")
    l1_mshr_hint = _parse_int_kw(parser, "l1_mshr_hint")
    profile = _parse_body_region(parser)
    event_type = _parse_event_type(parser, TileEvent)
    if len(indices) != 1:
      parser.raise_error("gather indices(...) expects exactly one L1 buffer")
    tag = event_type.tag.data  # type: ignore[attr-defined]
    return cls(
      source,
      indices[0],
      destination,
      result_bytes,
      cache_min_bytes,
      cache_target_bytes,
      l1_mshr_hint,
      (),
      tag,
      _region=profile,
    )


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
    _print_int_kw(printer, "pow_ops", self.pow_ops.value.data)
    _print_event_type(printer, self.result.type)

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    bytes_total = _parse_int_kw(parser, "bytes")
    exponent = _parse_int_kw(parser, "exponent")
    pow_ops = _parse_int_kw(parser, "pow_ops")
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
  """``tile.signal input_released(%task)`` / ``tile.signal output_ready(%task)``.

  Phase signal (reference.mlir tile.signal), bound to the logical task
  that emits it (PR 3): when every dispatched task of one dispatch has
  signalled a phase, the corresponding dispatch result event fires.
  ``input_released`` = this task will not read its L2 input subview
  again; ``output_ready`` = this task's output is visible in L2.

  The operand must be the tile program's ``!nest.task`` formal (block
  arg 0); the legacy operand-less syntax is a parse error.
  """

  name = "tile.signal"

  phase = prop_def(StringAttr)
  task = operand_def(NestTask)

  PHASES: ClassVar[tuple[str, ...]] = ("input_released", "output_ready")

  def __init__(self, phase: str, task):
    super().__init__(
      properties=_props({"phase": StringAttr(phase)}),
      operands=[[task]],
    )

  def print(self, printer: Printer) -> None:
    printer.print_string(" ")
    printer.print_string(self.phase.data)
    printer.print_string("(")
    printer.print_operand(self.task)
    printer.print_string(")")

  @classmethod
  def parse(cls, parser: Parser) -> Self:
    phase = parser.parse_identifier()
    if phase not in cls.PHASES:
      parser.raise_error(f"unknown tile.signal phase '{phase}'")
    parser.parse_punctuation("(")
    task = parser.parse_operand()
    parser.parse_punctuation(")")
    return cls(phase, task)


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
  NestSubviewOp,
  NestPrefetchOp,
  NestDMAStoreOp,
  NestDispatchOp,
  NestCollectiveOp,
  NestReleaseOp,
  NestAwaitOp,
  NestBarrierOp,
  NestReturnOp,
  TileProgramDefOp,
  TileSubviewOp,
  TileAllocOp,
  TileLoadOp,
  TileStoreOp,
  TileProfiledAccessOp,
  TileGatherOp,
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
  "elenor",
  operations,
  [
    NestEvent,
    TileEvent,
    NestBuffer,
    NestGlobalView,
    NestL2View,
    TileL1Buffer,
    NestTask,
    TaskRange,
    NestAggregate,
    NexusEvent,
    NestGlobalMemref,
  ],
)

__all__ = [
  "DTYPE_BYTES",
  "Elenor",
  "NestActionLike",
  "NestAggregate",
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
  "NestGlobalView",
  "NestL2View",
  "NestPrefetchOp",
  "NestReleaseOp",
  "NestReturnOp",
  "NestSubviewOp",
  "NestTask",
  "NestTaskRangeOp",
  "NexusActionLike",
  "NexusAwaitOp",
  "NexusEvent",
  "NexusProgramOp",
  "NexusReturnOp",
  "NexusSubmitContextOp",
  "TaskRange",
  "TileActionLike",
  "TileAllocOp",
  "TileAwaitOp",
  "TileBoaOp",
  "TileEvent",
  "TileEvuOp",
  "TileGatherOp",
  "TileL1Buffer",
  "TileLoadOp",
  "TilePowOp",
  "TileProfiledAccessOp",
  "TileProgramDefOp",
  "TileReturnOp",
  "TileSignalOp",
  "TileStoreOp",
  "TileSubviewOp",
]
