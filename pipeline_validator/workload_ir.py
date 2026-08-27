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
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.ir import Attribute
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from .dialects.elenor import (
  DTYPE_BYTES,
  Elenor,
  NestBuffer,
  NestContextOp,
  NestEvent,
  NestGlobalMemref,
  NestGlobalView,
  NestL2View,
  NestSubviewOp,
  NestTask,
  NestTaskRangeOp,
  NexusAwaitOp,
  NexusProgramOp,
  NexusReturnOp,
  NexusSubmitContextOp,
  TileEvent,
  TileL1Buffer,
  TileProgramDefOp,
  TileSignalOp,
  TileSubviewOp,
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


# ---------------------------------------------------------------------------
# Byte-count helper (single source of truth, shared with lowering)
# ---------------------------------------------------------------------------


def _view_bytes(dims, dtype: str) -> int:
  """prod(dims) * dtype_size."""
  n = 1
  for d in dims:
    n *= int(d)
  return n * DTYPE_BYTES[dtype]


def _int_list(arr) -> list[int]:
  return [int(d.value.data) for d in arr.data]


def _shape_type(
  type_attr: Attribute,
) -> NestBuffer | NestGlobalMemref | NestGlobalView | NestL2View | TileL1Buffer:
  if not isinstance(type_attr, (NestBuffer, NestGlobalMemref, NestGlobalView, NestL2View, TileL1Buffer)):
    raise VerifyException(f"expected shape-typed memory attribute, got {type(type_attr).__name__}")
  return type_attr


def _shape_bytes(type_attr: Attribute) -> int:
  shaped = _shape_type(type_attr)
  return _view_bytes(_int_list(shaped.dims), shaped.dtype.data)


def _assert_contiguous_subview(sizes: list[int], backing_dims: list[int], op_name: str) -> None:
  """V1 only allows contiguous row-major subviews so every transfer resolves
  to one logical byte range.  For any dim ``i`` with ``sizes[i] > 1``, every
  trailing dim ``j > i`` must satisfy ``sizes[j] == backing_dims[j]``."""
  for i in range(len(sizes)):
    if sizes[i] <= 1:
      continue
    for j in range(i + 1, len(sizes)):
      if sizes[j] != backing_dims[j]:
        raise VerifyException("non-contiguous subviews are not supported by the physical transfer model")


def _shape_key(type_attr: Attribute) -> tuple[tuple[int, ...], str]:
  """Comparable key for shape-typed attributes (dims tuple + dtype str)."""
  shaped = _shape_type(type_attr)
  return (tuple(_int_list(shaped.dims)), shaped.dtype.data)


def _formal_index(value, block) -> int | None:
  """Return the block-arg index of ``value`` if it is a block arg, else None."""
  for i, arg in enumerate(block.args):
    if arg is value:
      return i
  return None


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
    for prog in programs.values():
      _verify_program(prog)
    _verify_context(context, programs)
    return context

  # Model path: exactly one nexus.program + at least one nest.context
  if len(nexus_programs) != 1:
    raise VerifyException("expected exactly one nexus.program")
  if not contexts:
    raise VerifyException("model requires at least one nest.context")
  program = nexus_programs[0]
  for prog in programs.values():
    _verify_program(prog)
  for ctx in contexts.values():
    _verify_context(ctx, programs)
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

  ctx_block = context.body.block
  # Rule 2: context formals must be !nest.global_memref
  for i, arg in enumerate(ctx_block.args):
    if not isinstance(arg.type, NestGlobalMemref):
      raise VerifyException(
        f"nest.context '@{context.sym_name.data}' formal {i} must be !nest.global_memref"
      )

  body = _body_ops(context)
  seen_events: set[str] = set()
  defined_events: set[str] = set()
  seen_buffers: set[str] = set()

  for op in body:
    if isinstance(op, NestAllocOp):
      slot = op.slot.data
      if op.role.data not in ("in", "out", "inout"):
        raise VerifyException(f'nest.alloc slot \'{slot}\' role must be "in", "out" or "inout"')
      if slot in seen_buffers:
        raise VerifyException(f"duplicate L2 buffer slot '{slot}'")
      seen_buffers.add(slot)
      continue

    if isinstance(op, NestTaskRangeOp):
      if op.num_tasks <= 0:
        raise VerifyException("nest.task.range requires from < to")
      continue

    if isinstance(op, NestSubviewOp):
      # Rule 10: src must be a context global formal
      idx = _formal_index(op.src, ctx_block)
      if idx is None:
        raise VerifyException("nest.subview source must be a context global formal")
      formal = ctx_block.args[idx]
      formal_type = _shape_type(formal.type)
      parent = _int_list(formal_type.dims)
      name = formal.name_hint or f"arg{idx}"
      offsets = _int_list(op.offsets)
      sizes = _int_list(op.sizes)
      strides = _int_list(op.strides)
      if len(offsets) != len(parent) or len(sizes) != len(parent) or len(strides) != len(parent):
        raise VerifyException(f"nest.subview rank mismatch on '{name}': expected {len(parent)} dims")
      # Rule 8: V1 strides must be unit
      if any(s != 1 for s in strides):
        raise VerifyException("non-unit strides are not supported in V1")
      # Rule 8b: V1 only supports contiguous row-major subviews — every
      # transfer resolves to one logical byte range (PR 2 physical model).
      _assert_contiguous_subview(sizes, parent, "nest.subview")
      view_type = _shape_type(op.result.type)
      if _int_list(view_type.dims) != sizes or view_type.dtype.data != formal_type.dtype.data:
        raise VerifyException("nest.subview result type must match sizes and source dtype")
      # Rule 6: bounds
      for d, (o, s, pd) in enumerate(zip(offsets, sizes, parent)):
        if o < 0 or s < 1:
          raise VerifyException(f"nest.subview dim {d} requires offset >= 0 and size >= 1")
        if o + s > pd:
          raise VerifyException(
            f"nest.subview exceeds bounds of '{name}' dim {d}: offset {o} + size {s} > {pd}"
          )
      # Rule 6: byte overflow
      if _view_bytes(sizes, formal_type.dtype.data) >= 2**63:
        raise VerifyException("nest.subview byte count overflows int64")
      continue

    # Single-result async ops: prefetch, store, collective
    if isinstance(op, (NestPrefetchOp, NestDMAStoreOp, NestCollectiveOp)):
      # Rule 9: transfer byte equality (prefetch/store only)
      if isinstance(op, NestPrefetchOp):
        src_bytes = _shape_bytes(op.src.type)
        dst_bytes = _shape_bytes(op.dst.type)
        if src_bytes != dst_bytes:
          raise VerifyException(f"transfer '{op.name}' src bytes ({src_bytes}) != dst bytes ({dst_bytes})")
      elif isinstance(op, NestDMAStoreOp):
        src_bytes = _shape_bytes(op.src.type)
        dst_bytes = _shape_bytes(op.dst.type)
        if src_bytes != dst_bytes:
          raise VerifyException(f"transfer '{op.name}' src bytes ({src_bytes}) != dst bytes ({dst_bytes})")
      tag = op.result.type.tag.data
      if tag in seen_events:
        raise VerifyException(f"duplicate event tag '{tag}'")
      seen_events.add(tag)
      defined_events.add(tag)
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
      prog_def = programs[prog_sym]
      # Rule 5: dispatch↔tile.program binding
      l2_formals = list(prog_def.body.block.args[1:])
      ins_list = list(op.ins)
      outs_list = list(op.outs)
      total = len(ins_list) + len(outs_list)
      if len(ins_list) != len(l2_formals) or len(outs_list) != len(l2_formals):
        raise VerifyException(
          f"dispatch '@{prog_sym}' passes {total} actuals"
          f" but tile.program declares {len(l2_formals)} l2 formals"
        )
      for i, (actual, formal) in enumerate(zip(ins_list, l2_formals)):
        if not isinstance(actual.type, NestBuffer) or _shape_key(actual.type) != _shape_key(formal.type):
          raise VerifyException(
            f"dispatch actual {i} type does not match tile.program '@{prog_sym}' formal {i}"
          )
      for i, (actual, formal) in enumerate(zip(outs_list, l2_formals)):
        if not isinstance(actual.type, NestBuffer) or _shape_key(actual.type) != _shape_key(formal.type):
          raise VerifyException(
            f"dispatch actual {i} type does not match tile.program '@{prog_sym}' formal {i}"
          )
      # Validate 1:1 logical-task-to-tile mapping
      task_op = cast(NestTaskRangeOp, op.tasks.owner)
      num_tasks = int(task_op.to_task.value.data) - int(task_op.from_task.value.data)
      expected_tiles = bin(placement).count("1")
      if num_tasks != expected_tiles:
        raise VerifyException(
          f"dispatch task range ({num_tasks}) must match placement popcount ({expected_tiles})"
        )
      # Rule 7: tile.subview bounds at dispatch checkpoint
      to_task = int(task_op.to_task.value.data)
      for i, formal in enumerate(l2_formals):
        parent = _int_list(_shape_type(formal.type).dims)
        for sv in _program_subviews_of_formal(prog_def, i + 1):
          offsets = _int_list(sv.offsets)
          sizes = _int_list(sv.sizes)
          td = None if sv.task_dim is None else int(sv.task_dim.value.data)
          if td is None:
            for d, (o, s, pd) in enumerate(zip(offsets, sizes, parent)):
              if o < 0 or s < 1 or o + s > pd:
                raise VerifyException(
                  f"tile.subview exceeds bounds of formal {i} dim {d}: offset {o} + size {s} > {pd}"
                )
          else:
            if td < 0 or td >= len(parent):
              raise VerifyException("tile.subview task_dim must be a valid dimension index")
            o = offsets[td]
            s = sizes[td]
            pd = parent[td]
            tmax = to_task - 1
            if o + tmax + s > pd:
              raise VerifyException(
                f"tile.subview on formal {i} dim {td}: offset {o} + max task {tmax} + size {s} exceeds {pd}"
              )
      # PR 3: signal_policy must exactly match the program's emitted
      # phases; V1 only supports all_tasks aggregation.
      prog_phases = _program_signal_phases(prog_def)
      policy = op.signal_policy
      if set(policy) != prog_phases:
        raise VerifyException(
          f"dispatch '@{prog_sym}' signal_policy phases {sorted(policy)}"
          f" must match tile.program signal phases {sorted(prog_phases)}"
        )
      for mode in policy.values():
        if mode != "all_tasks":
          raise VerifyException(
            f"dispatch '@{prog_sym}' signal policy mode '{mode}' is not supported in V1"
          )
      inrel_tag = op.input_released.type.tag.data
      outready_tag = op.output_ready.type.tag.data
      if ("input_released" in policy) != bool(inrel_tag):
        raise VerifyException(
          f"dispatch '@{prog_sym}' input_released policy and event tag must be declared together"
        )
      if ("output_ready" in policy) != bool(outready_tag):
        raise VerifyException(
          f"dispatch '@{prog_sym}' output_ready policy and event tag must be declared together"
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

  _verify_release_graph(body)


def _program_signal_phases(prog: TileProgramDefOp) -> frozenset[str]:
  """Phases the tile program actually emits via ``tile.signal``."""
  return frozenset(op.phase.data for op in _body_ops(prog) if isinstance(op, TileSignalOp))


def _verify_release_graph(body: list) -> None:
  """PR 3 static ownership graph: every ``nest.alloc`` has exactly one
  role-legal ``nest.release`` before ``nest.return``, gated on the
  matching aggregate events / final store of its consumers.
  """
  from .dialects.elenor import NestAllocOp, NestDispatchOp, NestDMAStoreOp, NestReleaseOp, NestReturnOp

  allocs: dict = {}
  releases: dict = {}
  stores: dict = {}
  dispatches: list = []
  return_index = len(body)
  for idx, op in enumerate(body):
    if isinstance(op, NestAllocOp):
      allocs[op.result] = op
    elif isinstance(op, NestReleaseOp):
      releases.setdefault(op.buffer, []).append((idx, op))
    elif isinstance(op, NestDMAStoreOp):
      stores.setdefault(op.src, []).append(op)
    elif isinstance(op, NestReturnOp):
      return_index = idx
    elif isinstance(op, NestDispatchOp):
      dispatches.append(op)

  for buffer, alloc in allocs.items():
    slot = alloc.slot.data
    role = alloc.role.data
    rels = releases.get(buffer, [])
    if len(rels) != 1:
      raise VerifyException(
        f"nest.alloc slot '{slot}' requires exactly one nest.release in the same context"
      )
    rel_idx, rel = rels[0]
    if rel_idx > return_index:
      raise VerifyException(f"nest.release of slot '{slot}' must appear before nest.return")
    if not rel.depends_on:
      raise VerifyException(f"nest.release of slot '{slot}' requires at least one depends_on event")
    deps = list(rel.depends_on)

    ins_consumers = [d for d in dispatches if buffer in d.ins]
    outs_producers = [d for d in dispatches if buffer in d.outs]
    if role == "in":
      expected = []
      for d in ins_consumers:
        if "input_released" not in d.signal_policy:
          raise VerifyException(
            f"release of input slot '{slot}' lacks a matching input_released consumer dispatch"
          )
        expected.append(d.input_released)
      if sorted(deps, key=str) != sorted(expected, key=str) or len(set(deps)) != len(deps):
        raise VerifyException(
          f"nest.release of input slot '{slot}' must depend on exactly"
          " the input_released results of every consuming dispatch"
        )
    else:
      if not outs_producers:
        raise VerifyException(
          f"nest.release of slot '{slot}' (role '{role}') lacks a matching dispatch producer in outs"
        )
      buffer_stores = stores.get(buffer, [])
      if not buffer_stores:
        raise VerifyException(
          f"nest.release of slot '{slot}' (role '{role}') requires a final nest.dma.store.async"
        )
      last_store = buffer_stores[-1]
      if len(deps) != 1 or deps[0] is not last_store.result:
        raise VerifyException(
          f"nest.release of slot '{slot}' (role '{role}') must depend"
          " only on its final nest.dma.store.async result"
        )
      expected_out = []
      for d in outs_producers:
        if "output_ready" not in d.signal_policy:
          raise VerifyException(f"release of slot '{slot}' lacks a matching output_ready producer dispatch")
        expected_out.append(d.output_ready)
      missing = [e for e in expected_out if e not in last_store.depends_on]
      if missing:
        raise VerifyException(
          f"final store of slot '{slot}' must depend on every producing dispatch output_ready result"
        )

  for buffer, _rels in releases.items():
    if buffer not in allocs:
      raise VerifyException("nest.release operand must be a nest.alloc result from the same context")


def _program_subviews_of_formal(prog: TileProgramDefOp, formal_pos: int) -> list:
  """Return all TileSubviewOp ops in prog whose src is block.args[formal_pos]."""
  result = []
  block = prog.body.block
  for op in block.ops:
    if isinstance(op, TileSubviewOp):
      if _formal_index(op.src, block) == formal_pos:
        result.append(op)
  return result


def _verify_program(prog: TileProgramDefOp) -> None:
  from .dialects.elenor import (
    TileAllocOp,
    TileAwaitOp,
    TileBoaOp,
    TileEvuOp,
    TileLoadOp,
    TilePowOp,
    TileSignalOp,
    TileStoreOp,
  )

  block = prog.body.block
  args = list(block.args)
  if not args or not isinstance(args[0].type, NestTask):
    raise VerifyException(f"tile.program '@{prog.sym_name.data}' first formal must be !nest.task")
  for i, arg in enumerate(args[1:], start=1):
    if not isinstance(arg.type, NestBuffer):
      raise VerifyException(f"tile.program '@{prog.sym_name.data}' formal {i} must be !nest.l2_buffer")

  body = _body_ops(prog)
  seen_events: set[str] = set()
  defined_events: set[str] = set()

  for op in body:
    if isinstance(op, TileSubviewOp):
      # Rule 10: src must be a tile.program L2 formal (not task)
      idx = _formal_index(op.src, block)
      if idx is None or idx == 0:
        raise VerifyException("tile.subview source must be a tile.program l2 formal")
      # task operand and task_dim must be used together
      if bool(op.task) != (op.task_dim is not None):
        raise VerifyException("tile.subview task operand and task_dim must be used together")
      source_type = _shape_type(args[idx].type)
      parent = _int_list(source_type.dims)
      offsets = _int_list(op.offsets)
      sizes = _int_list(op.sizes)
      strides = _int_list(op.strides)
      if len(offsets) != len(parent) or len(sizes) != len(parent) or len(strides) != len(parent):
        raise VerifyException(f"tile.subview rank mismatch: expected {len(parent)} dims")
      # Rule 8: V1 strides must be unit
      if any(s != 1 for s in strides):
        raise VerifyException("non-unit strides are not supported in V1")
      # Rule 8b: contiguous row-major subviews only (PR 2 physical model)
      _assert_contiguous_subview(sizes, parent, "tile.subview")
      # task_dim range
      if op.task_dim is not None:
        td = int(op.task_dim.value.data)
        if td < 0 or td >= len(parent):
          raise VerifyException("tile.subview task_dim must be a valid dimension index")
      # result type must match sizes and source dtype
      view_type = _shape_type(op.result.type)
      if _int_list(view_type.dims) != sizes or view_type.dtype.data != source_type.dtype.data:
        raise VerifyException("tile.subview result type must match sizes and source dtype")
      continue

    if isinstance(op, (TileLoadOp, TileStoreOp, TilePowOp, TileEvuOp, TileBoaOp)):
      if not isinstance(op.result.type, TileEvent):
        raise VerifyException(f"expected tile.event result type in '{op.name}'")
      tag = op.result.type.tag.data
      if tag in seen_events:
        raise VerifyException(f"duplicate event tag '{tag}' in tile program '@{prog.sym_name.data}'")
      seen_events.add(tag)
      defined_events.add(tag)
      # Rule 9: transfer byte equality (load/store only)
      if isinstance(op, TileLoadOp):
        src_bytes = _shape_bytes(op.src.type)
        dst_bytes = _shape_bytes(op.dst.type)
        if src_bytes != dst_bytes:
          raise VerifyException(f"transfer '{op.name}' src bytes ({src_bytes}) != dst bytes ({dst_bytes})")
      elif isinstance(op, TileStoreOp):
        src_bytes = _shape_bytes(op.src.type)
        dst_bytes = _shape_bytes(op.dst.type)
        if src_bytes != dst_bytes:
          raise VerifyException(f"transfer '{op.name}' src bytes ({src_bytes}) != dst bytes ({dst_bytes})")
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
      if _formal_index(op.task, block) != 0:
        raise VerifyException("tile.signal operand must be the tile.program task formal (block arg 0)")
      continue

    if isinstance(op, TileAllocOp):
      continue


def _body_ops(op) -> list:
  region = op.body if hasattr(op, "body") else op.regions[0]
  if len(region.blocks) != 1:
    raise VerifyException("expected exactly one block in body region")
  return list(region.blocks[0].ops)


def _verify_nexus_program(program: NexusProgramOp, contexts: dict[str, NestContextOp]) -> None:
  body = _body_ops(program)
  if not body or not isinstance(body[-1], NexusReturnOp):
    raise VerifyException("nexus.program body must end with nexus.return")

  block = program.body.block
  # Rule 1: each block arg must be !nest.global_memref and have a non-empty name
  for i, arg in enumerate(block.args):
    if not isinstance(arg.type, NestGlobalMemref):
      raise VerifyException(f"nexus.program input {i} must be !nest.global_memref")
    if not (arg.name_hint or "").strip():
      raise VerifyException(
        f"nexus.program input {i} has no name; global inputs must be named for input binding"
      )

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
      # Rule 3: submit↔context signature
      formal_types = [a.type for a in contexts[ctx_sym].body.block.args]
      if len(op.actuals) != len(formal_types):
        raise VerifyException(
          f"submit_context '@{ctx_sym}' passes {len(op.actuals)} actuals"
          f" but nest.context '@{ctx_sym}' declares {len(formal_types)} formals"
        )
      for i, (actual, formal) in enumerate(zip(op.actuals, formal_types)):
        if _shape_key(actual.type) != _shape_key(formal):
          raise VerifyException(
            f"submit_context actual {i} type does not match nest.context '@{ctx_sym}' formal {i}"
          )
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
