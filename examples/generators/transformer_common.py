"""Shared helpers for the Transformer prefill/decode workload generators.

The generators build author source IR directly with the xDSL ops from
``pipeline_validator.dialects.elenor`` (same construction style as
``pipeline_validator.workload_builders``), print it in custom assembly with
``print_workload_ir``, and prepend a header comment block.  The generated
``.mlir`` files are committed to the repository so that runs are
deterministic; the generators exist so nobody hand-writes the unrolled
attention bodies.
"""

from __future__ import annotations

from pathlib import Path

from pipeline_validator.config import HardwareConfig
from pipeline_validator.dialects.elenor import (
  NestAwaitOp,
  NestBuffer,
  NestL2View,
  TileAllocOp,
  TileAwaitOp,
  TileBoaOp,
  TileEvuOp,
  TileLoadOp,
  TileStoreOp,
  TileSubviewOp,
)
from pipeline_validator.profiles import build_registry
from pipeline_validator.workload_ir import print_workload_ir
from scripts.format_mlir import format_text

DTYPE_BYTES = {"bf16": 2, "f32": 4}


def contract_bytes(
  hw: HardwareConfig, level: str, byte_alignments: list[tuple[int, int]]
) -> tuple[int, tuple[int, ...]]:
  """Return ``(declared_bytes, allowed_profiles)`` for one Arena contract.

  ``declared_bytes`` is the no-reuse conservative reservation of the given
  buffers against mode 0 (matching the compiler's authoritative layout when
  lifetimes do not prove reuse); ``allowed_profiles`` lists every bundled
  mode whose per-bank user SPM covers the reservation with
  ``requested_contexts_per_tile == 1``.
  """
  from pipeline_validator.compiler.resources import conservative_arena_bytes

  registry = build_registry(hw)
  profiles = getattr(registry, level)
  baseline = registry.profile(level, 0)
  reserved = conservative_arena_bytes(byte_alignments, baseline)
  allowed = tuple(
    mode for mode, profile in profiles.items() if reserved // profile.banks <= profile.user_spm_per_bank
  )
  if 0 not in allowed:
    raise ValueError(f"{level}.mode0 cannot satisfy authored Arena reservation {reserved}")
  return reserved, allowed


def write_workload(path: Path, header: list[str], module) -> Path:
  """Write ``// header`` lines plus the printed, 100-column-wrapped module."""
  text = (
    "".join(f"// {line}".rstrip() + "\n" for line in header) + "\n" + format_text(print_workload_ir(module))
  )
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")
  return path


# ---------------------------------------------------------------------------
# Small tile.* emit helpers (straight op constructors, no hidden semantics)
# ---------------------------------------------------------------------------


def l2_view(src, task, task_dim: int | None, offsets, sizes, view_shape, dtype: str):
  if task_dim is not None and task is None:
    raise ValueError("task_dim requires the task handle")
  return TileSubviewOp(
    src, task, task_dim, list(offsets), list(sizes), [1] * len(offsets), NestL2View.of(view_shape, dtype)
  )


def l1_alloc(shape, dtype: str, name: str, alignment: int = 256):
  buf = TileAllocOp(list(shape), dtype, alignment=alignment)
  buf.result.name_hint = name
  return buf


def l1_load(l2_view_op, l1_buf, tag: str):
  ev = TileLoadOp(l2_view_op.result, l1_buf.result, tag)
  return ev


def l1_store(l1_buf, l2_view_op, tag: str):
  ev = TileStoreOp(l1_buf.result, l2_view_op.result, tag)
  return ev


def tile_await(events):
  return TileAwaitOp(list(events))


def boa_matmul(m: int, n: int, k: int, tag: str, accumulate: bool = False):
  ops = 2 * m * n * k
  return TileBoaOp("matmul", m, n, k, ops, tag, accumulate=accumulate)


def evu(op_name: str, ops: int, tag: str):
  return TileEvuOp(op_name, ops, tag)


def nest_await(events):
  return NestAwaitOp(list(events))


def buffer_of(shape, dtype: str) -> NestBuffer:
  return NestBuffer.of(shape, dtype)
