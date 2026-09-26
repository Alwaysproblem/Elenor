"""Public workload IR I/O and verification API.

The public workload IR is xDSL custom assembly rooted at
``builtin.module``.  The module contains top-level named definitions:
``nest.context @name { ... }`` (exactly one) and ``tile.program @name
{ ... }`` (zero or more).  The context body dispatches tile programs by
symbol reference (``@prog_name``).
"""

from __future__ import annotations

from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from typing import cast

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.ir import Attribute, Block, SSAValue
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from .dialects.elenor import (
  DTYPE_BYTES,
  Elenor,
  NestBuffer,
  NestContextOp,
  NestContextResourcesAttr,
  NestDispatchOp,
  NestEvent,
  NestGlobalMemref,
  NestGlobalView,
  NestL2View,
  NestPublishOp,
  NestSubviewOp,
  NestTask,
  NestTaskRangeOp,
  NexusAwaitOp,
  NexusEvent,
  NexusProgramOp,
  NexusReturnOp,
  NexusSharedRefOp,
  NexusSubmitContextOp,
  TileEvent,
  TileGatherOp,
  TileL1Buffer,
  TileProfiledAccessOp,
  TileProgramDefOp,
  TileResourcesAttr,
  TileSignalOp,
  TileSubviewOp,
  _int_list,
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


def _view_offset_bytes(offsets: Sequence[int], backing_dims: Sequence[int], element_bytes: int) -> int:
  offset = 0
  for dim, component in enumerate(offsets):
    stride = 1
    for extent in backing_dims[dim + 1 :]:
      stride *= extent
    offset += component * stride * element_bytes
  return offset


def _shape_type(
  type_attr: Attribute,
) -> NestBuffer | NestGlobalMemref | NestGlobalView | NestL2View | TileL1Buffer:
  if not isinstance(type_attr, (NestBuffer, NestGlobalMemref, NestGlobalView, NestL2View, TileL1Buffer)):
    raise VerifyException(f"expected shape-typed memory attribute, got {type(type_attr).__name__}")
  return type_attr


def _shape_bytes(type_attr: Attribute) -> int:
  shaped = _shape_type(type_attr)
  return _view_bytes(_int_list(shaped.dims), shaped.dtype.data)


def _assert_transfer_bytes_equal(src_type: Attribute, dst_type: Attribute, op_name: str) -> None:
  src_bytes = _shape_bytes(src_type)
  dst_bytes = _shape_bytes(dst_type)
  if src_bytes != dst_bytes:
    raise VerifyException(f"transfer '{op_name}' src bytes ({src_bytes}) != dst bytes ({dst_bytes})")


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


def _verify_context_subview(op: NestSubviewOp, block: Block) -> None:
  # Rule 10: src must be a context global formal
  idx = _formal_index(op.src, block)
  if idx is None:
    raise VerifyException("nest.subview source must be a context global formal")
  formal = block.args[idx]
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


def _verify_dispatch_contract(
  op: NestDispatchOp,
  programs: dict[str, TileProgramDefOp],
  program_accesses: dict[str, tuple[frozenset[int], frozenset[int]]],
  context_buffers: set[SSAValue],
  context_shared_imports: set[SSAValue],
  placement: int,
) -> None:
  if op.context_id is not None and int(op.context_id.value.data) < 0:
    raise VerifyException("dispatch context must be >= 0")
  prog_sym = op.program.data
  if prog_sym not in programs:
    raise VerifyException(f"dispatch references unknown tile program '@{prog_sym}'")
  prog_def = programs[prog_sym]
  if not isinstance(prog_def.resource_contract, TileResourcesAttr):
    raise VerifyException(f"tile.program '@{prog_sym}' must own a #tile.resources contract")
  tile_resources = _resource_contract(
    prog_def.resource_contract, f"tile.program '@{prog_sym}' resource_contract"
  )
  requested_l1_mode = int(op.l1_mode.value.data)
  if requested_l1_mode not in tile_resources.allowed_profiles:
    raise VerifyException(
      f"dispatch '@{prog_sym}' l1_mode {requested_l1_mode} must belong to tile.program allowed_profiles"
    )
  # Rule 5: dispatch↔tile.program binding.  Program data formals are
  # zero or more globals followed by zero or more L2 buffers.
  data_formals = list(enumerate(prog_def.body.block.args[1:], start=1))
  global_formals = [
    (formal_pos, formal) for formal_pos, formal in data_formals if isinstance(formal.type, NestGlobalView)
  ]
  l2_formals = [
    (formal_pos, formal) for formal_pos, formal in data_formals if isinstance(formal.type, NestBuffer)
  ]
  globals_list = list(op.global_views)
  bindings_list = list(op.bindings)
  ins_list = list(op.ins)
  outs_list = list(op.outs)
  if len(globals_list) != len(global_formals):
    raise VerifyException(
      f"dispatch '@{prog_sym}' passes {len(globals_list)} global actuals"
      f" but tile.program declares {len(global_formals)} global formals"
    )
  for i, (actual, (_, formal)) in enumerate(zip(globals_list, global_formals)):
    if not isinstance(actual.type, NestGlobalView) or _shape_key(actual.type) != _shape_key(formal.type):
      raise VerifyException(
        f"dispatch global actual {i} type does not match tile.program '@{prog_sym}' global formal {i}"
      )
  if len(bindings_list) != len(l2_formals):
    raise VerifyException(
      f"dispatch bindings for '@{prog_sym}' pass {len(bindings_list)} actuals"
      f" but tile.program declares {len(l2_formals)} l2 formals"
    )
  for i, (actual, (_, formal)) in enumerate(zip(bindings_list, l2_formals)):
    if actual not in context_buffers:
      raise VerifyException(
        f"dispatch bindings actual {i} for '@{prog_sym}'"
        " must be a local allocation or readonly import from the current nest.context"
      )
    if not isinstance(actual.type, NestBuffer) or _shape_key(actual.type) != _shape_key(formal.type):
      raise VerifyException(
        f"dispatch bindings actual {i} type does not match tile.program '@{prog_sym}' l2 formal {i}"
      )

  binding_set = set(bindings_list)
  for label, actuals in (("ins", ins_list), ("outs", outs_list)):
    if len(set(actuals)) != len(actuals):
      raise VerifyException(f"dispatch {label} for '@{prog_sym}' contains a duplicate actual")
    if any(actual not in binding_set for actual in actuals):
      raise VerifyException(
        f"dispatch {label} for '@{prog_sym}' contains an actual that is not present in bindings"
      )

  read_formals, write_formals = program_accesses[prog_sym]
  actual_by_formal = {formal_pos: actual for (formal_pos, _), actual in zip(l2_formals, bindings_list)}
  expected_ins = {actual_by_formal[formal_pos] for formal_pos in read_formals}
  expected_outs = {actual_by_formal[formal_pos] for formal_pos in write_formals}
  if any(actual in context_shared_imports for actual in expected_outs):
    raise VerifyException(f"dispatch '@{prog_sym}' may not write a readonly L2 import")
  if set(ins_list) != expected_ins:
    raise VerifyException(
      f"dispatch ins for '@{prog_sym}' must exactly declare the buffers read by the tile program"
    )
  if set(outs_list) != expected_outs:
    raise VerifyException(
      f"dispatch outs for '@{prog_sym}' must exactly declare the buffers written by the tile program"
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
  for i, (formal_pos, formal) in enumerate(l2_formals):
    parent = _int_list(_shape_type(formal.type).dims)
    for sv in _program_subviews_of_formal(prog_def, formal_pos):
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
      raise VerifyException(f"dispatch '@{prog_sym}' signal policy mode '{mode}' is not supported in V1")
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


def _resource_contract(attr, owner: str):
  """Validate a resource-contract attribute into an immutable value.

  Structurally-valid attributes constructed outside the typed path can
  still carry invalid allowed/cache fields; surface those as
  ``VerifyException`` so source verification and the CLI rejection
  path handle them uniformly.
  """
  try:
    return attr.to_contract()
  except ValueError as exc:
    raise VerifyException(f"invalid {owner}: {exc}") from exc


def _verify_tile_subview(op: TileSubviewOp, block: Block) -> None:
  # tile.subview remains L2-only; gather is the only tile-side global
  # consumer in this PR.
  idx = _formal_index(op.src, block)
  if idx is None or idx == 0 or not isinstance(block.args[idx].type, NestBuffer):
    raise VerifyException("tile.subview source must be a tile.program l2 formal")
  # task operand and task_dim must be used together
  if bool(op.task) != (op.task_dim is not None):
    raise VerifyException("tile.subview task operand and task_dim must be used together")
  source_type = _shape_type(block.args[idx].type)
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


def _verify_gather_profile(op: TileGatherOp) -> None:
  accesses = list(op.profile.block.ops)
  if not accesses:
    raise VerifyException("gather profile must contain at least one tile.profiled.access")
  if any(not isinstance(access, TileProfiledAccessOp) for access in accesses):
    raise VerifyException("gather profile may contain only tile.profiled.access operations")
  profiled_accesses = [cast(TileProfiledAccessOp, access) for access in accesses]

  result_bytes = int(op.result_bytes.value.data)
  cache_target_bytes = int(op.cache_target_bytes.value.data)
  l1_mshr_hint = int(op.l1_mshr_hint.value.data)
  if result_bytes <= 0:
    raise VerifyException("gather result_bytes must be > 0")
  if result_bytes > _shape_bytes(op.destination.type):
    raise VerifyException("gather result_bytes exceeds destination extent")
  if cache_target_bytes < 0:
    raise VerifyException("gather cache_target_bytes must be >= 0")
  if l1_mshr_hint <= 0:
    raise VerifyException("gather l1_mshr_hint must be > 0")

  request_ids: set[str] = set()
  merge_contracts: dict[str, tuple[str, int]] = {}
  profiled_bytes = 0
  for access in profiled_accesses:
    request_id = access.request_id.data
    outcome = access.outcome.data
    access_bytes = int(access.bytes.value.data)
    line_token = None if access.line_token is None else access.line_token.data
    merge_group = None if access.merge_group is None else access.merge_group.data
    if not request_id.strip():
      raise VerifyException("gather request_id must be non-empty")
    if request_id in request_ids:
      raise VerifyException(f"duplicate gather request_id '{request_id}'")
    request_ids.add(request_id)
    if access_bytes <= 0:
      raise VerifyException(f"gather request '{request_id}' bytes must be > 0")
    if access_bytes > _shape_bytes(op.source.type):
      raise VerifyException(f"gather request '{request_id}' exceeds source extent")
    if outcome not in ("L1_HIT", "L2_HIT", "HBM_MISS"):
      raise VerifyException(f"unknown gather outcome '{outcome}'")
    if merge_group:
      if outcome != "HBM_MISS":
        raise VerifyException("gather merge_group is only valid for HBM_MISS")
      if not line_token:
        raise VerifyException("gather merge_group requires a non-empty line_token")
      contract = (line_token, access_bytes)
      previous = merge_contracts.setdefault(merge_group, contract)
      if previous != contract:
        raise VerifyException(f"gather merge_group '{merge_group}' must use one line_token and byte size")
    profiled_bytes += access_bytes
  if profiled_bytes != result_bytes:
    raise VerifyException(
      f"gather profile bytes ({profiled_bytes}) must equal result_bytes ({result_bytes})"
    )


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
    program_accesses = {name: _verify_program(prog) for name, prog in programs.items()}
    _verify_context(context, programs, program_accesses, allow_shared_imports=False)
    return context

  # Model path: exactly one nexus.program + at least one nest.context
  if len(nexus_programs) != 1:
    raise VerifyException("expected exactly one nexus.program")
  if not contexts:
    raise VerifyException("model requires at least one nest.context")
  program = nexus_programs[0]
  program_accesses = {name: _verify_program(prog) for name, prog in programs.items()}
  for ctx in contexts.values():
    _verify_context(ctx, programs, program_accesses, allow_shared_imports=True)
  _verify_nexus_program(program, contexts, programs)
  return program


def _verify_context(
  context: NestContextOp,
  programs: dict[str, TileProgramDefOp],
  program_accesses: dict[str, tuple[frozenset[int], frozenset[int]]],
  *,
  allow_shared_imports: bool,
) -> None:
  if not isinstance(context.resource_contract, NestContextResourcesAttr):
    raise VerifyException(
      f"nest.context '@{context.sym_name.data}' must own a #nest.context_resources contract"
    )
  context_resources = _resource_contract(
    context.resource_contract, f"nest.context '@{context.sym_name.data}' resource_contract"
  )
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
    NestTaskRangeOp,
  )

  ctx_block = context.body.block
  context_imports: set[SSAValue] = set()
  import_names: set[str] = set()
  seen_import = False
  for i, arg in enumerate(ctx_block.args):
    if isinstance(arg.type, NestGlobalMemref):
      if seen_import:
        raise VerifyException(
          f"nest.context '@{context.sym_name.data}' global formal {i} may not follow an L2 import"
        )
      continue
    if isinstance(arg.type, NestBuffer):
      seen_import = True
      if not allow_shared_imports:
        raise VerifyException("readonly L2 context formals require a nexus.program model")
      name = (arg.name_hint or "").strip()
      if not name:
        raise VerifyException(
          f"nest.context '@{context.sym_name.data}' L2 import formal {i} must have a non-empty name"
        )
      if name in import_names:
        raise VerifyException(
          f"nest.context '@{context.sym_name.data}' has duplicate L2 import formal name '{name}'"
        )
      import_names.add(name)
      context_imports.add(arg)
      continue
    raise VerifyException(
      f"nest.context '@{context.sym_name.data}' formal {i} must be !nest.global_memref"
      " or a readonly !nest.l2_buffer import"
    )

  body = _body_ops(context)
  context_allocs: set[SSAValue] = {op.result for op in body if isinstance(op, NestAllocOp)}
  context_buffers = context_allocs | context_imports
  seen_events: set[str] = set()
  defined_events: set[SSAValue] = set()
  seen_buffers: set[str] = set(import_names)
  logical_tasks = 0

  for op in body:
    if isinstance(op, NestAllocOp):
      slot = op.slot.data
      if op.role.data not in ("in", "out", "inout"):
        raise VerifyException(f'nest.alloc slot \'{slot}\' role must be "in", "out" or "inout"')
      if op.sharing.data not in ("private", "readonly", "context-local"):
        raise VerifyException(
          f"nest.alloc slot '{slot}' sharing must be 'private', 'readonly' or 'context-local'"
        )
      if slot in seen_buffers:
        raise VerifyException(f"duplicate L2 buffer slot or import name '{slot}'")
      seen_buffers.add(slot)
      continue

    if isinstance(op, NestTaskRangeOp):
      if op.num_tasks <= 0:
        raise VerifyException("nest.task.range requires from < to")
      continue

    if isinstance(op, NestSubviewOp):
      _verify_context_subview(op, ctx_block)
      continue

    if isinstance(op, (NestPrefetchOp, NestDMAStoreOp, NestCollectiveOp, NestPublishOp)):
      if isinstance(op, (NestPrefetchOp, NestDMAStoreOp)):
        _assert_transfer_bytes_equal(op.src.type, op.dst.type, op.name)
      tag = op.result.type.tag.data
      if tag in seen_events:
        raise VerifyException(f"duplicate event tag '{tag}'")
      seen_events.add(tag)
      for dep in getattr(op, "depends_on", ()):
        dep_tag = dep.type.tag.data
        if dep not in defined_events:
          raise VerifyException(f"depends_on references undefined event '{dep_tag}'")
      defined_events.add(op.result)
      continue

    if isinstance(op, NestDispatchOp):
      task_op = op.tasks.owner
      if not isinstance(task_op, NestTaskRangeOp):
        raise VerifyException("dispatch tasks operand must be a nest.task.range result")
      logical_tasks += task_op.num_tasks
      _verify_dispatch_contract(
        op, programs, program_accesses, context_buffers, context_imports, placement
      )
      for result in op.results:
        if not isinstance(result.type, NestEvent):
          continue
        tag = result.type.tag.data
        if not tag:
          continue
        if tag in seen_events:
          raise VerifyException(f"duplicate event tag '{tag}'")
        seen_events.add(tag)
      for dependency in op.depends_on:
        dependency_tag = cast(NestEvent, dependency.type).tag.data
        if dependency not in defined_events:
          raise VerifyException(f"dispatch depends_on references undefined event '{dependency_tag}'")
      defined_events.update(op.results)
      continue

    if isinstance(op, NestReleaseOp):
      for dependency in op.depends_on:
        dependency_tag = cast(NestEvent, dependency.type).tag.data
        if dependency not in defined_events:
          raise VerifyException(f"nest.release depends_on references undefined event '{dependency_tag}'")
      continue

    if isinstance(op, NestAwaitOp):
      for operand in op.events:
        tag = cast(NestEvent, operand.type).tag.data
        if operand not in defined_events:
          raise VerifyException(f"nest.await references undefined event '{tag}'")
      continue

    if isinstance(op, (NestBarrierOp, NestReturnOp)):
      continue

    raise VerifyException(f"unexpected nest context body op '{op.name}'")

  _verify_release_graph(body, ctx_block.args, programs)
  if logical_tasks != context_resources.logical_tasks:
    raise VerifyException(
      f"nest.context '@{context.sym_name.data}' resource_contract logical_tasks"
      f" ({context_resources.logical_tasks}) does not match dispatched tasks ({logical_tasks})"
    )


def _program_signal_phases(prog: TileProgramDefOp) -> frozenset[str]:
  """Phases the tile program actually emits via ``tile.signal``."""
  return frozenset(op.phase.data for op in _body_ops(prog) if isinstance(op, TileSignalOp))


def _dispatch_write_intervals(
  dispatch: NestDispatchOp,
  buffer: SSAValue,
  programs: dict[str, TileProgramDefOp],
) -> list[tuple[int, int]]:
  """Return the byte intervals actually written to ``buffer`` by one dispatch."""
  from .dialects.elenor import TileStoreOp

  program = programs[dispatch.program.data]
  block = program.body.block
  l2_formals = [
    (index, arg)
    for index, arg in enumerate(block.args)
    if index and isinstance(arg.type, NestBuffer)
  ]
  bindings = list(dispatch.bindings)
  task_range = dispatch.tasks.owner
  assert isinstance(task_range, NestTaskRangeOp)
  task_ids = range(int(task_range.from_task.value.data), int(task_range.to_task.value.data))
  intervals: list[tuple[int, int]] = []
  for (formal_index, formal), actual in zip(l2_formals, bindings):
    if actual is not buffer:
      continue
    formal_type = cast(NestBuffer, formal.type)
    dims = _int_list(formal_type.dims)
    dtype = formal_type.dtype.data
    element_bytes = DTYPE_BYTES[dtype]
    for op in _body_ops(program):
      if not isinstance(op, TileStoreOp):
        continue
      view = op.dst.owner
      if not isinstance(view, TileSubviewOp) or _formal_index(view.src, block) != formal_index:
        continue
      offsets = _int_list(view.offsets)
      sizes = _int_list(view.sizes)
      if view.task_dim is None:
        start = _view_offset_bytes(offsets, dims, element_bytes)
        intervals.append((start, start + _view_bytes(sizes, dtype)))
        continue
      task_dim = int(view.task_dim.value.data)
      for task_id in task_ids:
        task_offsets = list(offsets)
        task_offsets[task_dim] += task_id
        start = _view_offset_bytes(task_offsets, dims, element_bytes)
        intervals.append((start, start + _view_bytes(sizes, dtype)))
  return intervals


def _covers_full_buffer(intervals: Sequence[tuple[int, int]], byte_count: int) -> bool:
  cursor = 0
  for start, end in sorted(intervals):
    if start < 0 or end > byte_count:
      return False
    if start > cursor:
      return False
    cursor = max(cursor, end)
    if cursor >= byte_count:
      return True
  return byte_count == 0


def _verify_release_graph(
  body: list,
  context_args: Sequence[SSAValue],
  programs: dict[str, TileProgramDefOp],
) -> None:
  """Verify local and imported L2 view closure, publication and release dependencies."""
  from .dialects.elenor import (
    NestAllocOp,
    NestDispatchOp,
    NestDMAStoreOp,
    NestPrefetchOp,
    NestReleaseOp,
    NestReturnOp,
  )

  allocs: dict[SSAValue, NestAllocOp] = {}
  imports = {arg for arg in context_args if isinstance(arg.type, NestBuffer)}
  releases: dict[SSAValue, list[tuple[int, NestReleaseOp]]] = {}
  stores: dict[SSAValue, list[tuple[int, NestDMAStoreOp]]] = {}
  prefetches: dict[SSAValue, list[tuple[int, NestPrefetchOp]]] = {}
  publishes: dict[SSAValue, list[tuple[int, NestPublishOp]]] = {}
  dispatches: list[tuple[int, NestDispatchOp]] = []
  return_index = next((idx for idx, op in enumerate(body) if isinstance(op, NestReturnOp)), len(body))
  for idx, op in enumerate(body):
    if isinstance(op, NestAllocOp):
      allocs[op.result] = op
    elif isinstance(op, NestReleaseOp):
      releases.setdefault(op.buffer, []).append((idx, op))
    elif isinstance(op, NestDMAStoreOp):
      stores.setdefault(op.src, []).append((idx, op))
    elif isinstance(op, NestPrefetchOp):
      prefetches.setdefault(op.dst, []).append((idx, op))
    elif isinstance(op, NestPublishOp):
      publishes.setdefault(op.buffer, []).append((idx, op))
    elif isinstance(op, NestDispatchOp):
      dispatches.append((idx, op))

  tracked_buffers = set(allocs) | imports
  for buffer, alloc in allocs.items():
    slot = alloc.slot.data
    role = alloc.role.data
    sharing = alloc.sharing.data
    if sharing not in ("private", "readonly", "context-local"):
      raise VerifyException(
        f"nest.alloc slot '{slot}' sharing must be 'private', 'readonly' or 'context-local'"
      )
    rels = releases.get(buffer, [])
    if len(rels) != 1:
      raise VerifyException(
        f"nest.alloc slot '{slot}' requires exactly one nest.release in the same context"
      )
    rel_idx, rel = rels[0]
    if rel_idx >= return_index:
      raise VerifyException(f"nest.release of slot '{slot}' must appear before nest.return")
    for later_op in body[rel_idx + 1 :]:
      if any(operand is buffer for operand in later_op.operands):
        raise VerifyException(f"nest.release of slot '{slot}' must follow every use of the allocation")

    readers = [
      (idx, dispatch) for idx, dispatch in dispatches if any(actual is buffer for actual in dispatch.ins)
    ]
    writers = [
      (idx, dispatch) for idx, dispatch in dispatches if any(actual is buffer for actual in dispatch.outs)
    ]
    buffer_prefetches = prefetches.get(buffer, [])
    buffer_stores = stores.get(buffer, [])
    buffer_publishes = publishes.get(buffer, [])
    if role == "in" and writers:
      raise VerifyException(f"nest.alloc input slot '{slot}' may not be written by a tile dispatch")
    if role in ("out", "inout") and not writers:
      raise VerifyException(
        f"nest.release of slot '{slot}' (role '{role}') requires at least one actual tile writer"
      )
    if role in ("out", "inout") and sharing == "private" and not buffer_stores:
      raise VerifyException(
        f"nest.release of slot '{slot}' (role '{role}') requires at least one nest.dma.store.async"
      )

    for store_idx, store in buffer_stores:
      prior_writer_events = [
        dispatch.output_ready for dispatch_idx, dispatch in writers if dispatch_idx < store_idx
      ]
      if any(all(dep is not event for dep in store.depends_on) for event in prior_writer_events):
        raise VerifyException(
          f"store of slot '{slot}' must depend on every previously defined"
          " actual writer output_ready result"
        )
    if buffer_stores:
      last_store = buffer_stores[-1][1]
      writer_events = [dispatch.output_ready for _, dispatch in writers]
      if any(all(dep is not event for dep in last_store.depends_on) for event in writer_events):
        raise VerifyException(
          f"final store of slot '{slot}' must depend on every actual writer output_ready result"
        )

    expected_publish_deps = [
      *(prefetch.result for _, prefetch in buffer_prefetches),
      *(store.result for _, store in buffer_stores),
      *(dispatch.input_released for _, dispatch in readers),
      *(dispatch.output_ready for _, dispatch in writers),
    ]
    if sharing != "readonly":
      if buffer_publishes:
        raise VerifyException(f"{sharing} L2 slot '{slot}' may not be published")
      publish_event = None
    else:
      if len(buffer_publishes) != 1:
        raise VerifyException(f"readonly L2 slot '{slot}' requires exactly one nest.publish")
      publish_idx, publish = buffer_publishes[0]
      publish_event = publish.result
      if not publish.result.type.tag.data:
        raise VerifyException(f"nest.publish of readonly L2 slot '{slot}' requires a non-empty event tag")
      deps = list(publish.depends_on)
      if len(set(deps)) != len(deps) or set(deps) != set(expected_publish_deps):
        raise VerifyException(
          f"nest.publish of readonly L2 slot '{slot}' must depend on exactly all"
          " prefetch/store completions, reader input_released, and writer output_ready events"
        )
      if buffer_prefetches:
        # Prefetch targets the complete alloc result, and the transfer verifier
        # proves equal bytes even when its HBM view has different dimensions.
        initialized = True
      else:
        intervals = [
          interval
          for _, writer in writers
          for interval in _dispatch_write_intervals(writer, buffer, programs)
        ]
        initialized = _covers_full_buffer(intervals, _shape_bytes(buffer.type))
      if not initialized:
        raise VerifyException(
          f"readonly L2 slot '{slot}' must be fully initialized by a complete prefetch or tile writer"
        )
      for later_op in body[publish_idx + 1 :]:
        if isinstance(later_op, NestReleaseOp) and later_op.buffer is buffer:
          continue
        if any(operand is buffer for operand in later_op.operands):
          raise VerifyException(
            f"readonly L2 slot '{slot}' may not be accessed after nest.publish"
          )

    expected_release_deps = [
      *(dispatch.input_released for _, dispatch in readers),
      *(dispatch.output_ready for _, dispatch in writers if sharing == "context-local"),
      *(prefetch.result for _, prefetch in buffer_prefetches),
      *(store.result for _, store in buffer_stores),
      *([] if publish_event is None else [publish_event]),
    ]
    deps = list(rel.depends_on)
    if len(set(deps)) != len(deps) or set(deps) != set(expected_release_deps):
      if sharing == "context-local":
        raise VerifyException(
          f"nest.release of context-local slot '{slot}' must depend on exactly all reader"
          " input_released, writer output_ready, prefetch, and store completion events"
        )
      raise VerifyException(
        f"nest.release of slot '{slot}' must depend on exactly all reader input_released,"
        " prefetch, store, and required publish completion events"
      )

  for buffer in imports:
    name = (buffer.name_hint or "import").strip()
    rels = releases.get(buffer, [])
    if len(rels) != 1:
      raise VerifyException(f"readonly L2 import '{name}' requires exactly one nest.release")
    rel_idx, rel = rels[0]
    if rel_idx >= return_index:
      raise VerifyException(f"nest.release of readonly L2 import '{name}' must appear before nest.return")
    for later_op in body[rel_idx + 1 :]:
      if any(operand is buffer for operand in later_op.operands):
        raise VerifyException(f"nest.release of readonly L2 import '{name}' must follow every use")
    if prefetches.get(buffer):
      raise VerifyException(f"readonly L2 import '{name}' may not be a prefetch destination")
    if publishes.get(buffer):
      raise VerifyException(f"readonly L2 import '{name}' may not be published or re-exported")
    import_readers = [
      dispatch for _, dispatch in dispatches if any(actual is buffer for actual in dispatch.ins)
    ]
    if any(any(actual is buffer for actual in dispatch.outs) for _, dispatch in dispatches):
      raise VerifyException(f"readonly L2 import '{name}' may not be a dispatch write destination")
    buffer_stores = stores.get(buffer, [])
    expected = [
      *(dispatch.input_released for dispatch in import_readers),
      *(store.result for _, store in buffer_stores),
    ]
    deps = list(rel.depends_on)
    if len(set(deps)) != len(deps) or set(deps) != set(expected):
      raise VerifyException(
        f"nest.release of readonly L2 import '{name}' must depend on exactly all"
        " reader input_released and store completion events"
      )

  for buffer in releases:
    if buffer not in tracked_buffers:
      raise VerifyException(
        "nest.release operand must be a local nest.alloc result or readonly context import"
      )
  for buffer in publishes:
    if buffer not in allocs:
      raise VerifyException("nest.publish operand must be a local nest.alloc result")
  for buffer in stores:
    if buffer not in tracked_buffers:
      raise VerifyException("nest.dma.store.async source must be a local L2 allocation or readonly import")
  for buffer in prefetches:
    if buffer not in allocs:
      raise VerifyException("nest.dma.prefetch.async destination must be a local nest.alloc result")

def _program_subviews_of_formal(prog: TileProgramDefOp, formal_pos: int) -> list:
  """Return all TileSubviewOp ops in prog whose src is block.args[formal_pos]."""
  result = []
  block = prog.body.block
  for op in block.ops:
    if isinstance(op, TileSubviewOp):
      if _formal_index(op.src, block) == formal_pos:
        result.append(op)
  return result


def _verify_program(prog: TileProgramDefOp) -> tuple[frozenset[int], frozenset[int]]:
  from .dialects.elenor import (
    TileAllocOp,
    TileAwaitOp,
    TileBoaOp,
    TileEvuOp,
    TileFreeOp,
    TileGatherOp,
    TileLoadOp,
    TilePowOp,
    TileReturnOp,
    TileSignalOp,
    TileStoreOp,
  )

  if not isinstance(prog.resource_contract, TileResourcesAttr):
    raise VerifyException(f"tile.program '@{prog.sym_name.data}' must own a #tile.resources contract")
  _resource_contract(prog.resource_contract, f"tile.program '@{prog.sym_name.data}' resource_contract")
  block = prog.body.block
  args = list(block.args)
  if not args or not isinstance(args[0].type, NestTask):
    raise VerifyException(f"tile.program '@{prog.sym_name.data}' first formal must be !nest.task")
  seen_l2_formal = False
  for i, arg in enumerate(args[1:], start=1):
    if isinstance(arg.type, NestGlobalView):
      if seen_l2_formal:
        raise VerifyException(
          f"tile.program '@{prog.sym_name.data}' global formal {i} may not follow an l2 formal"
        )
      continue
    if isinstance(arg.type, NestBuffer):
      seen_l2_formal = True
      continue
    raise VerifyException(
      f"tile.program '@{prog.sym_name.data}' formal {i} must be !nest.global_view or !nest.l2_buffer"
    )

  body = _body_ops(prog)
  returns = [op for op in body if isinstance(op, TileReturnOp)]
  if len(returns) != 1 or body[-1] is not returns[0]:
    raise VerifyException(
      f"tile.program '@{prog.sym_name.data}' body must contain exactly one terminal tile.return"
    )

  seen_events: set[str] = set()
  defined_events: set = set()
  awaited_events: set = set()
  load_events: set = set()
  store_events: set = set()
  read_formals: set[int] = set()
  write_formals: set[int] = set()
  phase_counts = dict.fromkeys(TileSignalOp.PHASES, 0)
  defined_l1: set = set()
  live_l1: set = set()
  freed_l1: set = set()
  l1_access_events: dict = {}
  opaque_compute_events: set = set()

  def require_live_l1(buffer, operand_description: str) -> None:
    owner = buffer.owner
    if buffer in freed_l1:
      raise VerifyException(
        f"tile.free in '@{prog.sym_name.data}' is followed by"
        f" {operand_description} access to the freed allocation"
      )
    if (
      not isinstance(owner, TileAllocOp)
      or owner not in body
      or buffer not in defined_l1
      or buffer not in live_l1
    ):
      raise VerifyException(
        f"{operand_description} must be an earlier tile.alloc result from the current tile.program"
      )

  for op in body:
    if isinstance(op, TileSubviewOp):
      _verify_tile_subview(op, block)
      continue

    if isinstance(op, TileFreeOp):
      buffer = op.buffer
      owner = buffer.owner
      if not isinstance(owner, TileAllocOp) or owner not in body or buffer not in defined_l1:
        raise VerifyException(
          f"tile.free in '@{prog.sym_name.data}' requires an earlier"
          " tile.alloc result from the current tile.program"
        )
      if buffer in freed_l1:
        raise VerifyException(
          f"tile.free in '@{prog.sym_name.data}' may not free the same allocation more than once"
        )
      pending_accesses = l1_access_events[buffer] - awaited_events
      if pending_accesses:
        raise VerifyException(
          f"tile.free in '@{prog.sym_name.data}' requires every preceding"
          " asynchronous access to the allocation to be awaited"
        )
      if opaque_compute_events - awaited_events:
        raise VerifyException(
          f"tile.free in '@{prog.sym_name.data}' requires every preceding"
          " opaque compute event to be awaited"
        )
      live_l1.remove(buffer)
      freed_l1.add(buffer)
      continue

    if isinstance(op, (TileLoadOp, TileStoreOp, TileGatherOp, TilePowOp, TileEvuOp, TileBoaOp)):
      if not isinstance(op.result.type, TileEvent):
        raise VerifyException(f"expected tile.event result type in '{op.name}'")
      tag = op.result.type.tag.data
      if tag in seen_events:
        raise VerifyException(f"duplicate event tag '{tag}' in tile program '@{prog.sym_name.data}'")
      seen_events.add(tag)
      defined_events.add(op.result)
      if isinstance(op, (TilePowOp, TileEvuOp, TileBoaOp)):
        opaque_compute_events.add(op.result)
      # Rule 9: transfer byte equality (load/store only)
      if isinstance(op, TileLoadOp):
        require_live_l1(op.dst, "tile.load destination")
        l1_access_events[op.dst].add(op.result)
        view = op.src.owner
        if not isinstance(view, TileSubviewOp) or view not in body:
          raise VerifyException("tile.load source must be a tile.subview from the current tile.program")
        formal_index = _formal_index(view.src, block)
        if formal_index is None or not isinstance(args[formal_index].type, NestBuffer):
          raise VerifyException("tile.load source must resolve directly to a tile.program l2 formal")
        if phase_counts["input_released"]:
          raise VerifyException(
            f"tile.signal input_released in '@{prog.sym_name.data}' may not be followed by another L2 load"
          )
        read_formals.add(formal_index)
        load_events.add(op.result)
        _assert_transfer_bytes_equal(op.src.type, op.dst.type, op.name)
      elif isinstance(op, TileStoreOp):
        require_live_l1(op.src, "tile.store source")
        l1_access_events[op.src].add(op.result)
        view = op.dst.owner
        if not isinstance(view, TileSubviewOp) or view not in body:
          raise VerifyException(
            "tile.store destination must be a tile.subview from the current tile.program"
          )
        formal_index = _formal_index(view.src, block)
        if formal_index is None or not isinstance(args[formal_index].type, NestBuffer):
          raise VerifyException("tile.store destination must resolve directly to a tile.program l2 formal")
        if phase_counts["output_ready"]:
          raise VerifyException(
            f"tile.signal output_ready in '@{prog.sym_name.data}' may not be followed by another L2 store"
          )
        write_formals.add(formal_index)
        store_events.add(op.result)
        _assert_transfer_bytes_equal(op.src.type, op.dst.type, op.name)
      elif isinstance(op, TileGatherOp):
        source_index = _formal_index(op.source, block)
        if source_index is None or not isinstance(args[source_index].type, NestGlobalView):
          raise VerifyException("gather source must be a global formal of the current tile.program")
        require_live_l1(op.indices, "gather indices")
        require_live_l1(op.destination, "gather destination")
        if op.indices is op.destination:
          raise VerifyException("gather indices and destination must be different tile.alloc results")
        l1_access_events[op.indices].add(op.result)
        l1_access_events[op.destination].add(op.result)

        _verify_gather_profile(op)
      continue

    if isinstance(op, TileAwaitOp):
      for operand in op.events:
        if not isinstance(operand.type, TileEvent):
          continue
        tag = operand.type.tag.data
        if operand not in defined_events:
          raise VerifyException(f"tile.await references undefined event '{tag}'")
        awaited_events.add(operand)
      continue

    if isinstance(op, TileSignalOp):
      phase = op.phase.data
      if phase not in TileSignalOp.PHASES:
        raise VerifyException(f"unknown tile.signal phase '{phase}'")
      if _formal_index(op.task, block) != 0:
        raise VerifyException("tile.signal operand must be the tile.program task formal (block arg 0)")
      phase_counts[phase] += 1
      if phase_counts[phase] > 1:
        raise VerifyException(f"tile.signal {phase} in '@{prog.sym_name.data}' may appear at most once")
      if phase == "input_released" and any(event not in awaited_events for event in load_events):
        raise VerifyException(
          f"tile.signal input_released in '@{prog.sym_name.data}' requires"
          " every preceding L2 load completion event to be awaited"
        )
      if phase == "output_ready" and any(event not in awaited_events for event in store_events):
        raise VerifyException(
          f"tile.signal output_ready in '@{prog.sym_name.data}' requires"
          " every preceding L2 store completion event to be awaited"
        )
      continue

    if isinstance(op, TileAllocOp):
      defined_l1.add(op.result)
      live_l1.add(op.result)
      l1_access_events[op.result] = set()
      continue

    if isinstance(op, TileReturnOp):
      continue

    raise VerifyException(f"unexpected tile program body op '{op.name}'")

  if read_formals and phase_counts["input_released"] != 1:
    raise VerifyException(
      f"tile.signal input_released in '@{prog.sym_name.data}' is required"
      " exactly once for a program with L2 reads"
    )
  if write_formals and phase_counts["output_ready"] != 1:
    raise VerifyException(
      f"tile.signal output_ready in '@{prog.sym_name.data}' is required"
      " exactly once for a program with L2 writes"
    )
  return frozenset(read_formals), frozenset(write_formals)


def _body_ops(op) -> list:
  region = op.body if hasattr(op, "body") else op.regions[0]
  if len(region.blocks) != 1:
    raise VerifyException("expected exactly one block in body region")
  return list(region.blocks[0].ops)


def effective_submit_dependencies(submit: NexusSubmitContextOp) -> tuple[SSAValue, ...]:
  """Return explicit dependencies plus producer completions for shared actuals.

  Keeping this source-level closure in one helper makes hazard verification
  and compiler lowering agree about the dependencies a submit actually has.
  """
  dependencies = list(submit.depends_on)
  seen = set(dependencies)
  for actual in submit.actuals:
    owner = actual.owner
    if isinstance(owner, NexusSharedRefOp) and owner.producer not in seen:
      dependencies.append(owner.producer)
      seen.add(owner.producer)
  return tuple(dependencies)


def _verify_shared_l2_epochs(
  body: list,
  contexts: dict[str, NestContextOp],
  consumer_producers: Sequence[tuple[NexusSubmitContextOp, NexusSubmitContextOp]],
) -> None:
  """Prove shared submit pairs remain inside one L2 profile generation."""
  contracts = {
    name: _resource_contract(
      context.resource_contract, f"nest.context '@{name}' resource_contract"
    )
    for name, context in contexts.items()
  }
  possible_modes = {
    mode
    for contract in contracts.values()
    for mode in (*contract.allowed_profiles, contract.l2_mode)
  }
  initial_modes = set(possible_modes)
  initial_modes.add(max(possible_modes, default=-1) + 1)
  for initial_mode in initial_modes:
    current_mode = initial_mode
    epoch = 0
    submit_epochs: dict[NexusSubmitContextOp, tuple[int, int]] = {}
    for op in body:
      if not isinstance(op, NexusSubmitContextOp):
        continue
      contract = contracts[op.context_sym.data]
      resolved_mode = current_mode if current_mode in contract.allowed_profiles else contract.l2_mode
      if resolved_mode != current_mode:
        epoch += 1
      current_mode = resolved_mode
      submit_epochs[op] = (current_mode, epoch)
    for consumer, producer in consumer_producers:
      if submit_epochs[consumer] != submit_epochs[producer]:
        raise VerifyException(
          "nexus.shared.ref consumer crosses the producer's L2 profile epoch"
        )


def _verify_nexus_program(
  program: NexusProgramOp, contexts: dict[str, NestContextOp], programs: dict[str, TileProgramDefOp]
) -> None:
  from .dialects.elenor import NestAllocOp

  body = _body_ops(program)
  if not body or not isinstance(body[-1], NexusReturnOp):
    raise VerifyException("nexus.program body must end with nexus.return")

  block = program.body.block
  for i, arg in enumerate(block.args):
    if not isinstance(arg.type, NestGlobalMemref):
      raise VerifyException(f"nexus.program input {i} must be !nest.global_memref")
    if not (arg.name_hint or "").strip():
      raise VerifyException(
        f"nexus.program input {i} has no name; global inputs must be named for input binding"
      )

  seen_events: set[str] = set()
  defined_events: set[SSAValue] = set()
  shared_refs: dict[SSAValue, NexusSubmitContextOp] = {}
  consumer_producers: list[tuple[NexusSubmitContextOp, NexusSubmitContextOp]] = []

  for op in body:
    if isinstance(op, NexusSubmitContextOp):
      ctx_sym = op.context_sym.data
      if ctx_sym not in contexts:
        raise VerifyException(f"submit_context references unknown nest.context '@{ctx_sym}'")
      tag = cast(NexusEvent, op.result.type).tag.data
      if not tag:
        raise VerifyException("submit_context event tag must be non-empty")
      if tag in seen_events:
        raise VerifyException(f"duplicate event tag '{tag}'")
      seen_events.add(tag)
      for dependency in op.depends_on:
        if dependency not in defined_events:
          dependency_tag = cast(NexusEvent, dependency.type).tag.data
          raise VerifyException(
            f"submit_context depends_on references undefined event '{dependency_tag}'"
          )
      if len(set(op.depends_on)) != len(op.depends_on):
        raise VerifyException("submit_context depends_on events must be unique")

      context = contexts[ctx_sym]
      formal_args = list(context.body.block.args)
      if len(op.actuals) != len(formal_args):
        raise VerifyException(
          f"submit_context '@{ctx_sym}' passes {len(op.actuals)} actuals"
          f" but nest.context '@{ctx_sym}' declares {len(formal_args)} formals"
        )
      imported_backings: set[tuple[NexusSubmitContextOp, str]] = set()
      for i, (actual, formal) in enumerate(zip(op.actuals, formal_args)):
        if isinstance(formal.type, NestGlobalMemref):
          if (
            not isinstance(actual.type, NestGlobalMemref)
            or _shape_key(actual.type) != _shape_key(formal.type)
            or _formal_index(actual, block) is None
          ):
            raise VerifyException(
              f"submit_context actual {i} must be a matching nexus.program HBM input"
            )
          continue
        if not isinstance(formal.type, NestBuffer):
          raise VerifyException(f"nest.context '@{ctx_sym}' formal {i} has an unsupported type")
        if not isinstance(actual.type, NestBuffer) or _shape_key(actual.type) != _shape_key(formal.type):
          raise VerifyException(
            f"submit_context shared actual {i} type does not exactly match"
            f" nest.context '@{ctx_sym}' formal {i}"
          )
        ref = actual.owner
        if not isinstance(ref, NexusSharedRefOp) or actual not in shared_refs:
          raise VerifyException(
            f"submit_context L2 actual {i} must be a nexus.shared.ref to a published readonly slot"
          )
        producer = shared_refs[actual]
        slot = ref.slot.data
        backing = (producer, slot)
        if backing in imported_backings:
          raise VerifyException(
            f"submit_context '@{ctx_sym}' may not import the same producer slot more than once"
          )
        imported_backings.add(backing)
        consumer_producers.append((op, producer))
      defined_events.add(op.result)
      continue

    if isinstance(op, NexusSharedRefOp):
      if not op.slot.data.strip():
        raise VerifyException("nexus.shared.ref slot must be non-empty")
      if op.producer not in defined_events:
        raise VerifyException("nexus.shared.ref references an unknown or forward producer submit")
      referenced_producer = op.producer.owner
      if (
        not isinstance(referenced_producer, NexusSubmitContextOp)
        or referenced_producer.result is not op.producer
      ):
        raise VerifyException("nexus.shared.ref producer must be a nexus.submit_context.async result")
      producer_context = contexts[referenced_producer.context_sym.data]
      matches = [
        candidate
        for candidate in _body_ops(producer_context)
        if isinstance(candidate, NestAllocOp) and candidate.slot.data == op.slot.data
      ]
      if not matches:
        raise VerifyException(
          f"nexus.shared.ref references unknown producer slot '{op.slot.data}'"
        )
      export = matches[0]
      if export.sharing.data != "readonly":
        raise VerifyException(
          f"nexus.shared.ref producer slot '{op.slot.data}' must be a readonly published export"
        )
      if _shape_key(op.result.type) != _shape_key(export.result.type):
        raise VerifyException(
          f"nexus.shared.ref type does not match producer slot '{op.slot.data}' shape and dtype"
        )
      shared_refs[op.result] = referenced_producer
      continue

    if isinstance(op, NexusAwaitOp):
      for operand in op.events:
        tag = cast(NexusEvent, operand.type).tag.data
        if operand not in defined_events:
          raise VerifyException(f"nexus.await references undefined event '{tag}'")
      continue

    if isinstance(op, NexusReturnOp):
      continue

    raise VerifyException(f"unexpected nexus.program body op '{op.name}'")

  _verify_shared_l2_epochs(body, contexts, consumer_producers)
  _verify_device_memory_dependencies(body, contexts, programs)


def _context_global_accesses(
  context: NestContextOp, programs: dict[str, TileProgramDefOp]
) -> list[tuple[int, int, int, bool]]:
  """Conservative byte intervals for actual global reads/writes of a launch."""
  from .dialects.elenor import NestDispatchOp, NestDMAStoreOp, NestPrefetchOp, TileGatherOp

  accesses: list[tuple[SSAValue, bool]] = []
  for op in _body_ops(context):
    if isinstance(op, NestPrefetchOp):
      accesses.append((op.src, False))
    elif isinstance(op, NestDMAStoreOp):
      accesses.append((op.dst, True))
    elif isinstance(op, NestDispatchOp):
      program = programs[op.program.data]
      used = {nested.source for nested in _body_ops(program) if isinstance(nested, TileGatherOp)}
      formals = (arg for arg in program.body.block.args if isinstance(arg.type, NestGlobalView))
      accesses.extend((actual, False) for formal, actual in zip(formals, op.global_views) if formal in used)
  intervals: list[tuple[int, int, int, bool]] = []
  for value, writing in dict.fromkeys(accesses):
    view = value.owner
    assert isinstance(view, NestSubviewOp)
    index = _formal_index(view.src, context.body.block)
    assert index is not None
    backing = _shape_type(view.src.type)
    dims = _int_list(backing.dims)
    offset = _view_offset_bytes(_int_list(view.offsets), dims, DTYPE_BYTES[backing.dtype.data])
    intervals.append((index, offset, offset + _shape_bytes(value.type), writing))
  return intervals


def _verify_device_memory_dependencies(
  body: list, contexts: dict[str, NestContextOp], programs: dict[str, TileProgramDefOp]
) -> None:
  accesses = {name: _context_global_accesses(context, programs) for name, context in contexts.items()}
  ancestors: dict[SSAValue, set[SSAValue]] = {}
  awaited: set[SSAValue] = set()
  prior: list[tuple[SSAValue, list[tuple[SSAValue, int, int, bool]]]] = []
  for op in body:
    if isinstance(op, NexusAwaitOp):
      for event in op.events:
        awaited.add(event)
        awaited.update(ancestors[event])
    elif isinstance(op, NexusSubmitContextOp):
      ordered = set(awaited)
      for event in effective_submit_dependencies(op):
        ordered.add(event)
        ordered.update(ancestors[event])
      actuals = list(op.actuals)
      current = [
        (actuals[index], start, end, writing)
        for index, start, end, writing in accesses[op.context_sym.data]
      ]
      for producer, previous in prior:
        if producer in ordered:
          continue
        if any(
          actual is old_actual and start < old_end and old_start < end and (writing or old_write)
          for actual, start, end, writing in current
          for old_actual, old_start, old_end, old_write in previous
        ):
          raise VerifyException(
            "overlapping global accesses across context submissions require"
            " depends_on or a preceding nexus.await"
          )
      ancestors[op.result] = ordered
      prior.append((op.result, current))
