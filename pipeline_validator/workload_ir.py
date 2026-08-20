"""Public workload IR I/O and verification API.

The public workload IR is xDSL custom assembly rooted at
``builtin.module``.  The module contains top-level named definitions:
``nest.context @name { ... }`` (exactly one) and ``tile.program @name
{ ... }`` (zero or more).  The context body dispatches tile programs by
symbol reference (``@prog_name``).
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import cast

from xdsl.context import Context
from xdsl.dialects.builtin import (
  Builtin,
  ModuleOp,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from .dialects.elenor import (
  Elenor,
  NestContextOp,
  NestEvent,
  NestTaskRangeOp,
  NexusAwaitOp,
  NexusProgramOp,
  NexusReturnOp,
  NexusSubmitContextOp,
  TileEvent,
  TileProgramDefOp,
)


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
  """Print the workload IR in custom-assembly format (non-generic)."""
  stream = StringIO()
  Printer(stream=stream, print_generic_format=False).print_op(module)
  text = stream.getvalue()
  return text.rstrip("\n") + "\n"


def load_workload_ir(path: str | Path) -> ModuleOp:
  actual = Path(path)
  text = actual.read_text(encoding="utf-8")
  return parse_workload_ir(text, source_name=str(actual))

def verify_workload_ir(module: ModuleOp) -> NestContextOp | NexusProgramOp:
  module.verify()
  top_ops = list(module.body.block.ops)

  context: NestContextOp | None = None
  contexts: dict[str, NestContextOp] = {}
  programs: dict[str, TileProgramDefOp] = {}
  nexus_programs: list[NexusProgramOp] = []

  for op in top_ops:
    if isinstance(op, TileProgramDefOp):
      name = op.sym_name.data
      if name in programs:
        raise VerifyException(f"duplicate tile program '@{name}'")
      programs[name] = op
    elif isinstance(op, NestContextOp):
      name = op.sym_name.data
      if name in contexts:
        raise VerifyException(f"duplicate nest.context name '@{name}'")
      contexts[name] = op
      context = op
    elif isinstance(op, NexusProgramOp):
      nexus_programs.append(op)
    else:
      raise VerifyException(f"unexpected top-level op '{op.name}'")

  if not nexus_programs:
    # Legacy path: exactly one nest.context, no nexus.program
    if context is None:
      raise VerifyException("expected exactly one nest.context")
    if len(contexts) > 1:
      raise VerifyException("expected exactly one nest.context")
    _verify_context(context, programs)
    for prog in programs.values():
      _verify_program(prog)
    return context

  # Model path: exactly one nexus.program + at least one nest.context
  if len(nexus_programs) != 1:
    raise VerifyException("expected exactly one nexus.program")
  if not contexts:
    raise VerifyException("model requires at least one nest.context")
  program = nexus_programs[0]
  for ctx in contexts.values():
    _verify_context(ctx, programs)
  for prog in programs.values():
    _verify_program(prog)
  _verify_nexus_program(program, contexts)
  return program


def _verify_context(context: NestContextOp, programs: dict[str, TileProgramDefOp]) -> None:
  placement = int(context.placement.value.data)
  if placement == 0:
    raise VerifyException("nest.context placement must be non-zero")
  if context.context_id is not None and int(context.context_id.value.data) < 0:
    raise VerifyException("nest.context context must be >= 0")

  from .dialects.elenor import (
    NestAllocOp,
    NestAwaitOp,
    NestBarrierOp,
    NestCollectiveOp,
    NestDispatchOp,
    NestDMAStoreOp,
    NestPrefetchOp,
    NestReleaseOp,
    NestReturnOp,
  )

  body = _body_ops(context)
  seen_events: set[str] = set()
  defined_events: set[str] = set()
  seen_buffers: set[str] = set()

  for op in body:
    if isinstance(op, NestAllocOp):
      slot = op.result.type.slot.data
      if slot in seen_buffers:
        raise VerifyException(f"duplicate L2 buffer slot '{slot}'")
      seen_buffers.add(slot)
      continue

    if isinstance(op, NestTaskRangeOp):
      if op.num_tasks <= 0:
        raise VerifyException("nest.task.range requires from < to")
      continue

    # Single-result async ops: prefetch, store, collective
    if isinstance(op, (NestPrefetchOp, NestDMAStoreOp, NestCollectiveOp)):
      tag = op.result.type.tag.data
      if tag in seen_events:
        raise VerifyException(f"duplicate event tag '{tag}'")
      seen_events.add(tag)
      defined_events.add(tag)
      # check depends_on
      for dep in getattr(op, "depends_on", ()):
        dep_tag = dep.type.tag.data
        if dep_tag not in defined_events:
          raise VerifyException(f"depends_on references undefined event '{dep_tag}'")
      continue

    if isinstance(op, NestDispatchOp):
      if op.context_id is not None and int(op.context_id.value.data) < 0:
        raise VerifyException("dispatch context must be >= 0")
      prog_sym = op.program.data
      if prog_sym not in programs:
        raise VerifyException(f"dispatch references unknown tile program '@{prog_sym}'")
      # Validate 1:1 logical-task-to-tile mapping
      # one tile per logical task; reference.mlir's logical tasks are not
      # physical tiles, but this validator uses a 1:1 mapping).
      task_op = cast(NestTaskRangeOp, op.tasks.owner)
      num_tasks = int(task_op.to_task.value.data) - int(task_op.from_task.value.data)
      expected_tiles = bin(placement).count("1")
      if num_tasks != expected_tiles:
        raise VerifyException(
          f"dispatch task range ({num_tasks}) must match placement popcount ({expected_tiles})"
        )
      # phase tags (input_released / output_ready) are optional (empty = no phase)
      for r in op.results:
        if not isinstance(r.type, NestEvent):
          continue
        tag = r.type.tag.data
        if not tag:
          continue
        if tag in seen_events:
          raise VerifyException(f"duplicate event tag '{tag}'")
        seen_events.add(tag)
        defined_events.add(tag)
      for dep in op.depends_on:
        if not isinstance(dep.type, NestEvent):
          continue
        dep_tag = dep.type.tag.data
        if dep_tag not in defined_events:
          raise VerifyException(f"dispatch depends_on references undefined event '{dep_tag}'")
      continue

    if isinstance(op, NestReleaseOp):
      for dep in op.depends_on:
        if not isinstance(dep.type, NestEvent):
          continue
        dep_tag = dep.type.tag.data
        if dep_tag not in defined_events:
          raise VerifyException(f"nest.release depends_on references undefined event '{dep_tag}'")
      continue

    if isinstance(op, NestAwaitOp):
      for operand in op.events:
        if not isinstance(operand.type, NestEvent):
          continue
        tag = operand.type.tag.data
        if tag not in defined_events:
          raise VerifyException(f"nest.await references undefined event '{tag}'")
      continue

    if isinstance(op, (NestBarrierOp, NestReturnOp)):
      continue

    raise VerifyException(f"unexpected nest context body op '{op.name}'")

def _verify_program(prog: TileProgramDefOp) -> None:
  from .dialects.elenor import (
    TileAwaitOp,
    TileBoaOp,
    TileEvuOp,
    TileLoadOp,
    TilePowOp,
    TileReturnOp,
    TileSignalOp,
    TileStoreOp,
  )

  body = _body_ops(prog)
  seen_events: set[str] = set()
  defined_events: set[str] = set()

  for op in body:
    if isinstance(op, (TileLoadOp, TileStoreOp, TilePowOp, TileEvuOp, TileBoaOp)):
      if not isinstance(op.result.type, TileEvent):
        raise VerifyException(f"expected tile.event result type in '{op.name}'")
      tag = op.result.type.tag.data
      if tag in seen_events:
        raise VerifyException(
          f"duplicate event tag '{tag}' in tile program '@{prog.sym_name.data}'"
        )
      seen_events.add(tag)
      defined_events.add(tag)
      continue

    if isinstance(op, TileAwaitOp):
      for operand in op.events:
        if not isinstance(operand.type, TileEvent):
          continue
        tag = operand.type.tag.data
        if tag not in defined_events:
          raise VerifyException(f"tile.await references undefined event '{tag}'")
      continue

    if isinstance(op, TileSignalOp):
      if op.phase.data not in TileSignalOp.PHASES:
        raise VerifyException(f"unknown tile.signal phase '{op.phase.data}'")
      continue

    if isinstance(op, TileReturnOp):
      continue

    raise VerifyException(f"unexpected tile program body op '{op.name}'")


def _body_ops(op) -> list:
  region = op.body if hasattr(op, "body") else op.regions[0]
  if len(region.blocks) != 1:
    raise VerifyException("expected exactly one block in body region")
  return list(region.blocks[0].ops)


def _verify_nexus_program(program: NexusProgramOp, contexts: dict[str, NestContextOp]) -> None:
  body = _body_ops(program)
  if not body or not isinstance(body[-1], NexusReturnOp):
    raise VerifyException("nexus.program body must end with nexus.return")

  seen_events: set[str] = set()
  defined_events: set[str] = set()

  for op in body:
    if isinstance(op, NexusSubmitContextOp):
      ctx_sym = op.context_sym.data
      if ctx_sym not in contexts:
        raise VerifyException(f"submit_context references unknown nest.context '@{ctx_sym}'")
      tag = op.result.type.tag.data
      if not tag:
        raise VerifyException("submit_context event tag must be non-empty")
      if tag in seen_events:
        raise VerifyException(f"duplicate event tag '{tag}'")
      seen_events.add(tag)
      defined_events.add(tag)
      continue

    if isinstance(op, NexusAwaitOp):
      for operand in op.events:
        tag = operand.type.tag.data  # type: ignore[attr-defined]
        if tag not in defined_events:
          raise VerifyException(f"nexus.await references undefined event '{tag}'")
      continue

    if isinstance(op, NexusReturnOp):
      continue

    raise VerifyException(f"unexpected nexus.program body op '{op.name}'")
