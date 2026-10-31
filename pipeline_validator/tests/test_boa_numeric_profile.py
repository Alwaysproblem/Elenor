"""P06 numeric-profile gate: production-path vector parity against P03 goldens.

Every numeric comparison here runs the real production path
``parse_workload_ir -> compile_program -> load_program -> Simulator`` with
``fidelity="full_memory"`` and a seeded ``ByteStore``, then compares the
observed HBM output bytes against committed golden bytes.  Golden authority:

- ``hardware/boa/reference/vectors/matmul.json`` (P03, unchanged): full
  matmul vectors including specials, tails, masks, INIT/ADD chains and the
  N-P03-11 zero-K/zero-dimension group.
- ``pipeline_validator/tests/data/numeric_profile_directed_golden.json``
  (P06, generated from the unchanged P03 reference in a separate process):
  the tile_k 8-vs-16 grouping counterexamples and the k=64 eight-block case.
- ``pipeline_validator/tests/data/legacy_v4_boa_am.json`` (verbatim pre-P06
  v4 artifact): legacy-version rejection.

This module never imports ``hardware/boa/reference`` and never derives
expected numeric bytes itself.  Since P08 every legal vector, including FINAL
epilogues (bias/scale/ReLU/clip) and INT8 output, is emitted with the source
AM-allocation epilogue syntax and compared against the golden FINAL bytes.

Standalone CLI (same engine as the pytest tests):

  python -m pipeline_validator.tests.test_boa_numeric_profile \\
      --case {vector-parity,zero-k,cutover-reject} --out DIR --seed N \\
      [--config PATH]

``--config`` is an optional strict JSON document (unknown keys rejected)::

  {"schema": 1, "profile": "bf16_fp32_mac_rne_v1",
   "vectors": "hardware/boa/reference/vectors/matmul.json",
   "directed": "pipeline_validator/tests/data/numeric_profile_directed_golden.json",
   "legacy_v4": "pipeline_validator/tests/data/legacy_v4_boa_am.json"}

Relative paths resolve against the repository root.  ``--seed`` is recorded
in every report and seeds the simulator; the committed goldens are fixed, so
the seed never changes the expected bytes.  Reports contain only
deterministic content; wall-clock time is written to ``<case>.timing.json``
and marked non-deterministic for offline reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from xdsl.utils.exceptions import ParseError, VerifyException

from pipeline_validator.boa_numeric import matmul_f32
from pipeline_validator.compiled_program import (
  parse_compiled_program,
  program_digest,
  seal_program,
  serialize_compiled_program,
)
from pipeline_validator.compiler import compile_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.execution_verifier import target_fingerprint
from pipeline_validator.loader import load_program
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.simulator import Simulator
from pipeline_validator.workload_ir import parse_workload_ir, print_workload_ir

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "bf16_fp32_mac_rne_v1"
DEFAULT_VECTORS = ROOT / "hardware/boa/reference/vectors/matmul.json"
DEFAULT_DIRECTED = ROOT / "pipeline_validator/tests/data/numeric_profile_directed_golden.json"
DEFAULT_LEGACY_V4 = ROOT / "pipeline_validator/tests/data/legacy_v4_boa_am.json"
CASES = ("vector-parity", "zero-k", "cutover-reject")

L1_CONTRACT_BYTES = 32768
L2_BASE = 0x100000
ALIGN = 256
_DTYPE_BYTES = {"f32": 4, "bf16": 2, "i8": 1}

_ZERO_WORK_IDS = ("mm-add-k0-negzero", "mm-init-k0-bias-scale", "mm-m0-empty", "mm-n0-empty")

# Illegal golden vectors -> production contract check.  ``kind`` selects the
# mechanism: ``config`` (hardware field validation), ``module`` (source/IR
# rejection), ``binding`` (loader input byte-count enforcement),
# ``scalar`` exercises byte-oriented production argument checks; ``input``
# rejects BF16 words that cannot be represented in a 16-bit operand.
_NEGATIVE_PLAN: dict[str, dict] = {
  "mm-illegal-tile-k-zero": {"kind": "config", "stage": ("config",), "pattern": "boa_numeric_tile_k"},
  "mm-illegal-empty-steps": {
    "kind": "module",
    "stage": ("parse", "compile"),
    "pattern": "output|AM|FINAL|boa",
  },
  "mm-illegal-seq-gap": {"kind": "module", "stage": ("compile",), "pattern": "seq"},
  "mm-illegal-init-not-first": {"kind": "module", "stage": ("compile",), "pattern": "INIT|accumulate|seq"},
  "mm-illegal-init-on-add": {"kind": "module", "stage": ("compile",), "pattern": "INIT|accumulate|seq"},
  "mm-illegal-final-missing": {
    "kind": "module",
    "stage": ("compile",),
    "pattern": "terminate with one output|output|FINAL",
  },
  "mm-illegal-final-early": {"kind": "module", "stage": ("compile",), "pattern": "finalized|output|FINAL"},
  "mm-illegal-storage-below-k": {"kind": "module", "stage": ("compile",), "pattern": "storage|k "},
  "mm-illegal-a-length": {"kind": "binding", "binding": "A0", "stage": ("load",), "pattern": "binding"},
  "mm-illegal-b-length": {"kind": "binding", "binding": "B0", "stage": ("load",), "pattern": "binding"},
  "mm-illegal-mask-length": {"kind": "module", "stage": ("compile",), "pattern": "mask|k_mask"},
  "mm-illegal-mask-tail-true": {"kind": "module", "stage": ("compile",), "pattern": "mask|tail"},
  "mm-illegal-mask-not-bool": {"kind": "scalar"},
  "mm-illegal-word-range": {"kind": "input"},
  "mm-illegal-word-negative": {"kind": "input"},
  "mm-illegal-negative-m": {"kind": "scalar"},
  "mm-illegal-negative-k": {"kind": "scalar"},
}


# ---------------------------------------------------------------------------
# Strict configuration
# ---------------------------------------------------------------------------


def load_config(path: str | None) -> dict:
  config: dict = {
    "profile": PROFILE,
    "vectors": DEFAULT_VECTORS,
    "directed": DEFAULT_DIRECTED,
    "legacy_v4": DEFAULT_LEGACY_V4,
  }
  if path is None:
    return config
  raw = json.loads(Path(path).read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    raise ValueError("numeric-profile config must be a JSON object")
  unknown = sorted(set(raw) - {"schema", "profile", "vectors", "directed", "legacy_v4"})
  if unknown:
    raise ValueError(f"unknown numeric-profile config keys: {unknown}")
  if "schema" in raw and (type(raw["schema"]) is not int or raw["schema"] != 1):
    raise ValueError(f"numeric-profile config schema must be 1, got {raw['schema']!r}")
  if "profile" in raw and raw["profile"] != PROFILE:
    raise ValueError(f"numeric-profile config profile must be {PROFILE!r}")
  for key in ("vectors", "directed", "legacy_v4"):
    if key in raw:
      if not isinstance(raw[key], str):
        raise ValueError(f"numeric-profile config '{key}' must be a path string")
      candidate = Path(raw[key])
      config[key] = candidate if candidate.is_absolute() else ROOT / candidate
  for key in ("vectors", "directed", "legacy_v4"):
    if not Path(config[key]).is_file():
      raise ValueError(f"numeric-profile config '{key}' file missing: {config[key]}")
  return config


# ---------------------------------------------------------------------------
# Golden document helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
  document = json.loads(Path(path).read_text(encoding="utf-8"))
  if document.get("profile") != PROFILE:
    raise ValueError(f"{path} profile must be {PROFILE!r}")
  return document


def _hex_bytes(document_hex: str) -> bytes:
  return bytes.fromhex(document_hex)


def _pack_operands(words: list[str]) -> bytes:
  """Pack vector inputs without narrowing, padding or replacing invalid words."""
  values = [int(word, 16) for word in words]
  if any(value < 0 or value > 0xFFFF for value in values):
    raise ValueError("BF16 input word is outside the unsigned 16-bit range")
  return struct.pack(f"<{len(values)}H", *values)


def _epilogue_text(epilogue: dict | None) -> str:
  """Source AM-allocation epilogue keywords mirroring a golden epilogue."""
  if not epilogue:
    return ""
  parts = []
  if epilogue.get("bias") is not None:
    parts.append("bias = [" + ", ".join(epilogue["bias"]) + "]")
  if epilogue.get("scale") is not None:
    parts.append(f"scale = {epilogue['scale']}")
  if epilogue.get("relu"):
    parts.append("relu")
  if epilogue.get("clip") is not None:
    parts.append("clip = [" + ", ".join(epilogue["clip"]) + "]")
  if epilogue.get("inv_scale") is not None:
    parts.append(f"inv_scale = {epilogue['inv_scale']}")
  if epilogue.get("output_dtype") == "i8":
    parts.append(f"zero_point = {epilogue.get('zero_point', 0)}")
  return "".join(f" {part}" for part in parts)


def _plan_steps(steps: list[dict]) -> list[dict]:
  """Normalize golden steps into builder form (storage_k defaults to k)."""
  planned = []
  for step in steps:
    storage_k = step.get("storage_k")
    planned.append(
      {
        "seq": step["seq"],
        "k": step["k"],
        "storage_k": step["k"] if storage_k is None else storage_k,
        "mask": step.get("mask"),
        "init": bool(step["init"]),
        "final": bool(step["final"]),
        "a": list(step["a"]),
        "b": list(step["b"]),
      }
    )
  return planned


# ---------------------------------------------------------------------------
# Source module builder (one tile.program / nest.context per vector)
# ---------------------------------------------------------------------------


def _ssa(record_id: str) -> str:
  return "".join(char if char.isalnum() else "_" for char in record_id)


def _shape(left: int, right: int, dtype: str) -> str:
  return f"{left}x{right}x{dtype}"


Transpose = tuple[bool, bool] | list[tuple[bool, bool]]


def _step_transpose(transpose: Transpose, index: int) -> tuple[bool, bool]:
  """One (transpose_a, transpose_b) pair for all steps, or one per step."""
  return transpose[index] if isinstance(transpose, list) else transpose


def _a_shape(m: int, step: dict, transpose: tuple[bool, bool]) -> tuple[int, int]:
  return (step["storage_k"], m) if transpose[0] else (m, step["storage_k"])


def _b_shape(n: int, step: dict, transpose: tuple[bool, bool]) -> tuple[int, int]:
  return (n, step["storage_k"]) if transpose[1] else (step["storage_k"], n)


def _transpose_words(words: list[str], rows: int, cols: int) -> list[str]:
  """Row-major ``[rows, cols]`` words re-stored as row-major ``[cols, rows]``."""
  return [words[r * cols + c] for c in range(cols) for r in range(rows)]


def build_case_module(
  record_id: str,
  m: int,
  n: int,
  steps: list[dict],
  out_dtype: str,
  epilogue: dict | None = None,
  transpose: Transpose = (False, False),
) -> str:
  """Emit one typed BOA matmul module for a golden vector.

  Each step gets its own A[m, storage_k] / B[storage_k, n] BF16 buffers; the
  chain runs seq 0..S-1 on one AM slot with ``accumulate`` from step 1 and
  ``output`` only on the FINAL step.  ``k_mask`` mirrors the golden mask
  (boolean ints, storage_k long, lanes >= k false).  Zero K/M/N emit real
  empty shapes end to end; no step data is compacted or padded.  Illegal
  golden vectors reuse this builder verbatim so production must reject the
  actual violation.
  """
  planned = _plan_steps(steps)
  name = _ssa(record_id)
  dtype_bytes = _DTYPE_BYTES[out_dtype]
  lines: list[str] = []

  program_params = ["%task: !nest.task"]
  for index, step in enumerate(planned):
    program_params.append(
      f"%a{index}_l2: !nest.l2_buffer<"
      f"{_shape(*_a_shape(m, step, _step_transpose(transpose, index)), 'bf16')}>"
    )
    program_params.append(
      f"%b{index}_l2: !nest.l2_buffer<"
      f"{_shape(*_b_shape(n, step, _step_transpose(transpose, index)), 'bf16')}>"
    )
  program_params.append(f"%c_l2: !nest.l2_buffer<{_shape(m, n, out_dtype)}>")
  lines.append("builtin.module {")
  lines.append(f"  tile.program @{name}({', '.join(program_params)})")
  lines.append(
    "      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],\n"
    f"    tile_l1_spm_bytes_per_context = {L1_CONTRACT_BYTES}> {{"
  )
  for index, step in enumerate(planned):
    a_shape, b_shape = (
      _a_shape(m, step, _step_transpose(transpose, index)),
      _b_shape(n, step, _step_transpose(transpose, index)),
    )
    lines.append(
      f"    %a{index}_v = tile.subview %a{index}_l2 offsets = [0, 0] "
      f"sizes = [{a_shape[0]}, {a_shape[1]}] strides = [1, 1]\n"
      f"      : !nest.l2_view<{_shape(*a_shape, 'bf16')}>"
    )
    lines.append(
      f"    %b{index}_v = tile.subview %b{index}_l2 offsets = [0, 0] "
      f"sizes = [{b_shape[0]}, {b_shape[1]}] strides = [1, 1]\n"
      f"      : !nest.l2_view<{_shape(*b_shape, 'bf16')}>"
    )
  lines.append(
    f"    %c_v = tile.subview %c_l2 offsets = [0, 0] sizes = [{m}, {n}] strides = [1, 1]\n"
    f"      : !nest.l2_view<{_shape(m, n, out_dtype)}>"
  )
  for index, step in enumerate(planned):
    a_shape, b_shape = (
      _a_shape(m, step, _step_transpose(transpose, index)),
      _b_shape(n, step, _step_transpose(transpose, index)),
    )
    lines.append(
      f'    %a{index} = tile.alloc shape = [{a_shape[0]}, {a_shape[1]}] dtype = "bf16" '
      f"alignment = {ALIGN} : !tile.l1_buffer<{_shape(*a_shape, 'bf16')}>"
    )
    lines.append(
      f'    %b{index} = tile.alloc shape = [{b_shape[0]}, {b_shape[1]}] dtype = "bf16" '
      f"alignment = {ALIGN} : !tile.l1_buffer<{_shape(*b_shape, 'bf16')}>"
    )
  lines.append(
    f'    %out = tile.alloc memory_space = "l1" shape = [{m}, {n}] dtype = "{out_dtype}" '
    f"alignment = {ALIGN}\n      : !tile.l1_buffer<{_shape(m, n, out_dtype)}>"
  )
  lines.append(
    '    %acc = tile.alloc memory_space = "am" logical_slot = 0 output(%out) '
    f'shape = [{m}, {n}]\n      dtype = "f32" alignment = {ALIGN}{_epilogue_text(epilogue)}'
    f" : !tile.am_buffer<{_shape(m, n, 'f32')}>"
  )
  load_events: list[str] = []
  for index in range(len(planned)):
    lines.append(f'    %la{index} = tile.load.async %a{index}_v into %a{index} : !tile.event<"la{index}">')
    lines.append(f'    %lb{index} = tile.load.async %b{index}_v into %b{index} : !tile.event<"lb{index}">')
    load_events.extend((f"%la{index}", f"%lb{index}"))
  if load_events:
    lines.append(f"    tile.await {', '.join(load_events)}")
  lines.append("    tile.signal input_released(%task)")
  for index, step in enumerate(planned):
    modifiers = []
    if not step["init"]:
      modifiers.append("accumulate")
    if step["final"]:
      modifiers.append("output")
    mask_text = ""
    if step["mask"] is not None:
      entries = ", ".join("1" if flag else "0" for flag in step["mask"])
      mask_text = f" k_mask = [{entries}]"
    step_tr = _step_transpose(transpose, index)
    flags = [name for name, on in (("transpose_a", step_tr[0]), ("transpose_b", step_tr[1])) if on]
    tail = " ".join(flags + modifiers)
    lines.append(
      f'    %k{index} = tile.boa.async "matmul" ins(%a{index}, %b{index}) outs(%out) acc(%acc)\n'
      f"        m = {m} n = {n} k = {step['k']} seq = {step['seq']}{mask_text}"
      f'{(" " + tail) if tail else ""} : !tile.event<"k{index}">'
    )
    lines.append(f"    tile.await %k{index}")
  for index in range(len(planned)):
    lines.append(f"    tile.free %a{index}")
    lines.append(f"    tile.free %b{index}")
  lines.append('    %stored = tile.store.async %out into %c_v : !tile.event<"stored">')
  lines.append("    tile.await %stored")
  lines.append("    tile.signal output_ready(%task)")
  lines.append("    tile.return")
  lines.append("  }")

  context_params = [
    parameter
    for index, step in enumerate(planned)
    for parameter in (
      f"%A{index}: !nest.global_memref<"
      f"{_shape(*_a_shape(m, step, _step_transpose(transpose, index)), 'bf16')}>",
      f"%B{index}: !nest.global_memref<"
      f"{_shape(*_b_shape(n, step, _step_transpose(transpose, index)), 'bf16')}>",
    )
  ]
  context_params.append(f"%C: !nest.global_memref<{_shape(m, n, out_dtype)}>")
  l2_bytes = (
    sum(2 * (m * step["storage_k"] + step["storage_k"] * n) for step in planned) + m * n * dtype_bytes
  )
  # Minimal physical reservation floor for the slot bookkeeping of an
  # otherwise all-empty context; the logical payload stays the exact bytes.
  l2_spm = max(l2_bytes, ALIGN)
  lines.append(f"  nest.context @{name}_ctx({', '.join(context_params)})")
  lines.append(
    "    placement = 1\n"
    "        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],\n"
    f"      logical_tasks = 1, l2_spm_bytes = {l2_spm}, requested_contexts_per_tile = 1> {{"
  )
  pref_events: list[str] = []
  for index, step in enumerate(planned):
    a_shape, b_shape = (
      _a_shape(m, step, _step_transpose(transpose, index)),
      _b_shape(n, step, _step_transpose(transpose, index)),
    )
    lines.append(
      f"    %a{index}_blk = nest.subview %A{index} offsets = [0, 0] "
      f"sizes = [{a_shape[0]}, {a_shape[1]}] strides = [1, 1]\n"
      f"      : !nest.global_view<{_shape(*a_shape, 'bf16')}>"
    )
    lines.append(
      f"    %b{index}_blk = nest.subview %B{index} offsets = [0, 0] "
      f"sizes = [{b_shape[0]}, {b_shape[1]}] strides = [1, 1]\n"
      f"      : !nest.global_view<{_shape(*b_shape, 'bf16')}>"
    )
    lines.append(
      f'    %a{index}_l2_1 = nest.alloc slot = "a{index}" role = "in" '
      f'shape = [{a_shape[0]}, {a_shape[1]}] dtype = "bf16" alignment = {ALIGN}'
      f" : !nest.l2_buffer<{_shape(*a_shape, 'bf16')}>"
    )
    lines.append(
      f'    %b{index}_l2_1 = nest.alloc slot = "b{index}" role = "in" '
      f'shape = [{b_shape[0]}, {b_shape[1]}] dtype = "bf16" alignment = {ALIGN}'
      f" : !nest.l2_buffer<{_shape(*b_shape, 'bf16')}>"
    )
    lines.append(
      f"    %a{index}_pref = nest.dma.prefetch.async %a{index}_blk into %a{index}_l2_1"
      f' : !nest.event<"a{index}_pref">'
    )
    lines.append(
      f"    %b{index}_pref = nest.dma.prefetch.async %b{index}_blk into %b{index}_l2_1"
      f' : !nest.event<"b{index}_pref">'
    )
    pref_events.extend((f"%a{index}_pref", f"%b{index}_pref"))
  lines.append(
    f"    %c_blk = nest.subview %C offsets = [0, 0] sizes = [{m}, {n}] strides = [1, 1]\n"
    f"      : !nest.global_view<{_shape(m, n, out_dtype)}>"
  )
  lines.append(
    f'    %c_l2_1 = nest.alloc slot = "c" role = "out" shape = [{m}, {n}] '
    f'dtype = "{out_dtype}" alignment = {ALIGN} : !nest.l2_buffer<{_shape(m, n, out_dtype)}>'
  )
  lines.append("    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range")
  binding_names = [
    name_part for index in range(len(planned)) for name_part in (f"%a{index}_l2_1", f"%b{index}_l2_1")
  ]
  bindings_list = ", ".join([*binding_names, "%c_l2_1"])
  ins_list = ", ".join(binding_names)
  depends = f" depends_on({', '.join(pref_events)})" if pref_events else ""
  lines.append(
    f"    %grid, %read, %ready = nest.dispatch.tasks.async @{name} l1_mode = 0 context = 0\n"
    f"      tasks(%tasks) globals() bindings({bindings_list}) ins({ins_list})\n"
    f"      outs(%c_l2_1)\n"
    "      signal_policy {\n"
    "        input_released = #nest.aggregate<all_tasks>\n"
    "        output_ready = #nest.aggregate<all_tasks>\n"
    f"      }}{depends}\n"
    '      : (!nest.event<"grid">, !nest.event<"read">, !nest.event<"ready">)'
  )
  lines.append(
    '    %c_store = nest.dma.store.async %c_l2_1 into %c_blk depends_on(%ready) : !nest.event<"c_store">'
  )
  for index in range(len(planned)):
    lines.append(f"    nest.release %a{index}_l2_1 depends_on(%read, %a{index}_pref)")
    lines.append(f"    nest.release %b{index}_l2_1 depends_on(%read, %b{index}_pref)")
  lines.append("    nest.release %c_l2_1 depends_on(%c_store)")
  lines.append("    nest.await %grid, %c_store")
  lines.append("    nest.return")
  lines.append("  }")

  program_formals = context_params
  submit_args = [name_part for index in range(len(planned)) for name_part in (f"%A{index}", f"%B{index}")]
  submit_args.append("%C")
  lines.append(f"  nexus.program @{name}_entry({', '.join(program_formals)}) {{")
  lines.append(
    f'    %done_1 = nexus.submit_context.async @{name}_ctx({", ".join(submit_args)}) : !nexus.event<"done">'
  )
  lines.append("    nexus.await %done_1")
  lines.append("    nexus.return")
  lines.append("  }")
  lines.append("}")
  return "\n".join(lines) + "\n"


def _layout(m: int, n: int, steps: list[dict], out_dtype: str, transpose: Transpose = (False, False)):
  """Bindings, HBM seed data and the C output location for one module.

  Buffer sizes come from declared shapes. Operand bytes stay verbatim, so
  malformed vectors cannot be silently repaired by padding or truncation.
  """
  planned = _plan_steps(steps)
  dtype_bytes = _DTYPE_BYTES[out_dtype]
  bindings: dict[str, GlobalBinding] = {}
  seeds: list[tuple[int, bytes]] = []
  cursor = L2_BASE

  def reserve(name: str, size: int, permission: str, data: bytes) -> None:
    nonlocal cursor
    cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1)
    bindings[name] = GlobalBinding(name, cursor, size, permission)
    if data:
      seeds.append((cursor, data))
    cursor += size

  for index, step in enumerate(planned):
    a_count, b_count = m * step["storage_k"], step["storage_k"] * n
    reserve(
      f"A{index}",
      a_count * 2,
      "r",
      _pack_operands(
        _transpose_words(step["a"], m, step["storage_k"])
        if _step_transpose(transpose, index)[0]
        else step["a"]
      ),
    )
    reserve(
      f"B{index}",
      b_count * 2,
      "r",
      _pack_operands(
        _transpose_words(step["b"], step["storage_k"], n)
        if _step_transpose(transpose, index)[1]
        else step["b"]
      ),
    )
  c_size = m * n * dtype_bytes
  reserve("C", c_size, "w", b"")
  return bindings, seeds, bindings["C"].base_iova, c_size


def _compile_load_run(
  module_text: str,
  bindings: dict[str, GlobalBinding],
  seeds: list[tuple[int, bytes]],
  c_address: int,
  c_bytes: int,
  *,
  tile_k: int,
  seed: int,
  steps: int,
  actual_bindings: dict[str, GlobalBinding] | None = None,
):
  hw = HardwareConfig().with_overrides(boa_numeric_tile_k=tile_k)
  sim = SimConfig(fidelity="full_memory", context_count=1, seed=seed)
  artifact = compile_program(
    parse_workload_ir(print_workload_ir(parse_workload_ir(module_text))),
    hw,
    sim,
    binding_assumptions=bindings,
    source_name="numeric-profile",
  )
  replay = parse_compiled_program(serialize_compiled_program(artifact))
  loaded = load_program(replay, hw, sim, actual_bindings=actual_bindings or bindings)
  store = ByteStore()
  for address, data in seeds:
    store.seed_hbm(address, data)
  simulator = Simulator(hw, sim, byte_store=store)
  result = simulator.run(loaded)
  if result.completed:
    pools = result.group_snapshot["shared_am_per_tile"]
    assert sum(pool["acquire_count"] for pool in pools) == 1
    assert sum(pool["commit_count"] for pool in pools) == steps
    assert sum(pool["final_count"] for pool in pools) == 1
    assert sum(pool["slot_free_count"] for pool in pools) == 1
    assert all(
      pool["fault_count"] == pool["live_leases"] == pool["pending_requests"] == pool["reserved_slots"] == 0
      and pool["free_slots"] == pool["total_slots"]
      for pool in pools
    )
  actual = store.read_hbm(c_address, c_bytes) if result.completed and c_bytes else b""
  return result, actual, hw, sim, artifact


# ---------------------------------------------------------------------------
# Vector execution and classification
# ---------------------------------------------------------------------------


def _record(
  record_id: str,
  *,
  label: str,
  passed: bool,
  detail: str,
  inp: dict | None = None,
  golden_hex: str = "",
  expected_hex: str = "",
  actual_hex: str = "",
  cycles: int | None = None,
  stage: str = "",
  error: str = "",
) -> dict:
  record = {
    "id": record_id,
    "label": label,
    "pass": bool(passed),
    "detail": detail,
    "golden_final_hex": golden_hex,
    "expected_hex": expected_hex,
    "actual_hex": actual_hex,
  }
  if inp is not None:
    record.update(
      {
        "m": inp.get("m"),
        "n": inp.get("n"),
        "tile_k": inp.get("tile_k"),
        "steps": [
          {
            "seq": step["seq"],
            "k": step["k"],
            "storage_k": step.get("storage_k"),
            "mask": step.get("mask"),
            "init": bool(step["init"]),
            "final": bool(step["final"]),
          }
          for step in (inp.get("steps") or [])
        ],
      }
    )
  if cycles is not None:
    record["cycles"] = cycles
  if stage:
    record["stage"] = stage
  if error:
    record["error"] = error
  return record


def _run_negative(record_id: str, inp: dict, plan: dict, seed: int, sources: list[tuple[str, str]]) -> dict:
  """Exercise each invalid vector at its actual production/input boundary."""
  kind = plan["kind"]

  def outcome(stage: str, error: Exception | None) -> dict:
    pattern = plan.get("pattern")
    passed = error is not None and (pattern is None or re.search(pattern, str(error), re.I) is not None)
    return _record(
      record_id,
      label="negative",
      passed=passed,
      detail=f"invalid vector must be rejected at {stage}",
      inp=inp,
      stage=stage,
      error="" if error is None else str(error),
    )

  if kind == "config":
    try:
      HardwareConfig().with_overrides(boa_numeric_tile_k=inp["tile_k"])
    except ValueError as exc:
      return outcome("config", exc)
    return outcome("config", None)
  if kind in ("scalar", "input"):
    step = inp["steps"][0]
    stage = "numeric-api" if kind == "scalar" else "input"
    try:
      a, b = _pack_operands(step["a"]), _pack_operands(step["b"])
      if kind == "scalar":
        matmul_f32(
          a,
          b,
          m=inp["m"],
          n=inp["n"],
          k=step["k"],
          accumulate=not step["init"],
          partial=None,
          tile_k=inp["tile_k"],
          storage_k=step.get("storage_k"),
          mask=step.get("mask"),
        )
    except ValueError as exc:
      return outcome(stage, exc)
    return outcome(stage, None)

  m, n = inp["m"], inp["n"]
  module = build_case_module(record_id, m, n, inp["steps"], "f32")
  sources.append((f"{_ssa(record_id)}.mlir", module))
  bindings, seeds, c_address, c_bytes = _layout(m, n, inp["steps"], "f32")
  if kind == "binding":
    actual = dict(bindings)
    target = plan["binding"]
    supplied = len(inp["steps"][int(target[1:])][target[0].lower()]) * 2
    actual[target] = GlobalBinding(target, bindings[target].base_iova, supplied, "r")
    try:
      _compile_load_run(
        module,
        bindings,
        seeds,
        c_address,
        c_bytes,
        tile_k=inp["tile_k"],
        seed=seed,
        steps=len(inp["steps"]),
        actual_bindings=actual,
      )
    except ValueError as exc:
      return outcome("load", exc)
    return outcome("load", None)

  try:
    parsed = parse_workload_ir(module)
  except (ParseError, ValueError, VerifyException) as exc:
    return outcome("parse", exc)
  try:
    compile_program(
      parsed,
      HardwareConfig().with_overrides(boa_numeric_tile_k=inp["tile_k"]),
      SimConfig(fidelity="full_memory", context_count=1, seed=seed),
      binding_assumptions=bindings,
      source_name="numeric-profile",
    )
  except (ValueError, VerifyException) as exc:
    return outcome("compile", exc)
  return outcome("compile", None)


def _run_golden_case(case: dict, seed: int, sources: list[tuple[str, str]]) -> dict:
  record_id = case["id"]
  inp = case.get("input") or {}
  expected_block = case.get("expected") or {}
  if "error" in expected_block or not inp.get("steps"):
    plan = _NEGATIVE_PLAN.get(record_id)
    if plan is None:
      return _record(
        record_id,
        label="unmapped-negative",
        passed=False,
        detail="illegal golden vector has no production negative mapping",
        inp=inp,
      )
    return _run_negative(record_id, inp, plan, seed, sources)

  m, n = inp["m"], inp["n"]
  epilogue = inp.get("epilogue") or {}
  out_dtype = epilogue.get("output_dtype") or "f32"
  tile_k = inp["tile_k"]
  golden_final_hex = expected_block.get("output_hex", "")
  golden_final = _hex_bytes(golden_final_hex)
  module = build_case_module(record_id, m, n, inp["steps"], out_dtype, epilogue)
  sources.append((f"{_ssa(record_id)}.mlir", module))
  bindings, seeds, c_address, c_bytes = _layout(m, n, inp["steps"], out_dtype)
  try:
    result, actual, hw, sim, _artifact = _compile_load_run(
      module, bindings, seeds, c_address, c_bytes, tile_k=tile_k, seed=seed, steps=len(inp["steps"])
    )
  except (ParseError, ValueError, VerifyException) as exc:
    return _record(
      record_id,
      label="construction-error",
      passed=False,
      detail="legal golden vector failed production construction",
      inp=inp,
      golden_hex=golden_final_hex,
      stage="compile",
      error=str(exc),
    )
  fingerprint = target_fingerprint(hw, sim)
  passed = result.completed and actual == golden_final
  record = _record(
    record_id,
    label="full-golden-bytes",
    passed=passed,
    detail="completed production run (FINAL epilogue included) must equal the golden output bytes",
    inp=inp,
    golden_hex=golden_final_hex,
    expected_hex=golden_final_hex,
    actual_hex=actual.hex(),
    cycles=result.cycles,
  )
  record["executed_tile_k"] = hw.boa_numeric_tile_k
  record["target_fingerprint"] = fingerprint
  record["output_dtype"] = out_dtype
  record["epilogue_stages"] = sorted(
    key for key in ("bias", "scale", "clip", "inv_scale") if epilogue.get(key) is not None
  ) + (["relu"] if epilogue.get("relu") else [])
  return record


def _run_directed_case(case: dict, seed: int, sources: list[tuple[str, str]]):
  inp = case["input"]
  expected_hex = case["expected"]["output_hex"]
  golden_final = _hex_bytes(expected_hex)
  module = build_case_module(case["id"], inp["m"], inp["n"], inp["steps"], "f32")
  sources.append((f"{_ssa(case['id'])}.mlir", module))
  bindings, seeds, c_address, c_bytes = _layout(inp["m"], inp["n"], inp["steps"], "f32")
  result, actual, hw, sim, artifact = _compile_load_run(
    module, bindings, seeds, c_address, c_bytes, tile_k=inp["tile_k"], seed=seed, steps=len(inp["steps"])
  )
  record = _record(
    case["id"],
    label="directed-golden-bytes",
    passed=result.completed and actual == golden_final,
    detail="directed tile_k counterexample must reproduce the reference bytes",
    inp=inp,
    golden_hex=expected_hex,
    expected_hex=expected_hex,
    actual_hex=actual.hex(),
    cycles=result.cycles,
  )
  record["executed_tile_k"] = hw.boa_numeric_tile_k
  record["target_fingerprint"] = target_fingerprint(hw, sim)
  return record, hw, sim, actual, artifact, bindings, seeds, c_address, c_bytes


def _binary64_whole_step_bits(a_words: list[str], b_words: list[str]) -> int:
  """The deliberately wrong pre-P06 model: binary64 accumulation of the whole
  step with one final FP32 encode.  Used only to prove the counterexample
  still separates the old model from the frozen profile."""

  def value(word: str) -> float:
    bits = int(word, 16)
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]

  total = 0.0
  for a_word, b_word in zip(a_words, b_words):
    total += value(a_word) * value(b_word)
  return struct.unpack("<I", struct.pack("<f", total))[0]


# ---------------------------------------------------------------------------
# Case engines
# ---------------------------------------------------------------------------


def _finish_report(case: str, config: dict, checks: list[dict], vectors: list[dict]) -> dict:
  counts = {
    "vectors": len(vectors),
    "passed": sum(1 for item in vectors if item["pass"]),
    "failed": sum(1 for item in vectors if not item["pass"]),
    "full_bytes": sum(1 for item in vectors if item["label"] == "full-golden-bytes"),
    "directed": sum(1 for item in vectors if item["label"] == "directed-golden-bytes"),
    "negative": sum(1 for item in vectors if item["label"] == "negative"),
    "cycles_total": sum(item.get("cycles", 0) for item in vectors),
  }
  hard_failures = [item for item in vectors if not item["pass"]]
  status = "PASS" if checks and all(item["pass"] for item in checks) and not hard_failures else "FAIL"
  return {
    "schema": 1,
    "case": case,
    "profile": config["profile"],
    "counts": counts,
    "checks": checks,
    "status": status,
  }


def _inputs_echo(config: dict, keys: tuple[str, ...]) -> dict:
  echo = {"profile": config["profile"]}
  for key in keys:
    path = Path(config[key])
    echo[key] = {"path": str(path), "sha256": _sha256(path)}
  return echo


def _engine_vectors(
  config: dict, seed: int, subset: set[str] | None
) -> tuple[dict, list[dict], list[tuple[str, str]]]:
  document = _load_json(config["vectors"])
  sources: list[tuple[str, str]] = []
  vectors: list[dict] = []
  checks: list[dict] = []
  for case in document["cases"]:
    if subset is not None and case["id"] not in subset:
      continue
    record = _run_golden_case(case, seed, sources)
    vectors.append(record)
    checks.append({"id": f"vector:{record['id']}", "pass": record["pass"], "detail": record["detail"]})

  if subset is None:
    # Directed P06 counterexamples (tile_k 8 vs 16 and the k=64 chain).
    directed = _load_json(config["directed"])
    directed_records: dict[str, dict] = {}
    hw_by_id: dict[str, HardwareConfig] = {}
    sim_by_id: dict[str, SimConfig] = {}
    actual_by_id: dict[str, bytes] = {}
    for case in directed["cases"]:
      record, hw, sim, actual, _artifact, _b, _s, _ca, _cb = _run_directed_case(case, seed, sources)
      directed_records[case["id"]] = record
      hw_by_id[case["id"]] = hw
      sim_by_id[case["id"]] = sim
      actual_by_id[case["id"]] = actual
      vectors.append(record)
      checks.append({"id": f"vector:{record['id']}", "pass": record["pass"], "detail": record["detail"]})
    both = "p06-grouping-k8" in directed_records and "p06-grouping-k16" in directed_records
    fp8 = target_fingerprint(hw_by_id["p06-grouping-k8"], sim_by_id["p06-grouping-k8"]) if both else ""
    fp16 = target_fingerprint(hw_by_id["p06-grouping-k16"], sim_by_id["p06-grouping-k16"]) if both else ""
    fingerprint_ok = bool(fp8) and bool(fp16) and fp8 != fp16
    checks.append(
      {
        "id": "tile-k-fingerprint-inequality",
        "pass": fingerprint_ok,
        "detail": f"tile_k=8 {fp8} vs tile_k=16 {fp16}",
      }
    )
    bytes_ok = (
      both
      and actual_by_id["p06-grouping-k8"] != actual_by_id["p06-grouping-k16"]
      and directed_records["p06-grouping-k8"]["pass"]
    )
    checks.append(
      {
        "id": "tile-k-counterexample-bytes-differ",
        "pass": bytes_ok,
        "detail": (
          f"k8={directed_records.get('p06-grouping-k8', {}).get('actual_hex', '')} "
          f"k16={directed_records.get('p06-grouping-k16', {}).get('actual_hex', '')}"
        ),
      }
    )

    # Regression guard: the old binary64 whole-step model must stay
    # separated from the frozen profile on N-P03-8 (mm-grouping-tile8).
    separator = next(case for case in document["cases"] if case["id"] == "mm-grouping-tile8")
    step = separator["input"]["steps"][0]
    old_bits = _binary64_whole_step_bits(step["a"], step["b"])
    golden_bits = int(separator["expected"]["output_bits"][0], 16)
    production_bits = int(next(item["actual_hex"] for item in vectors if item["id"] == separator["id"]), 16)
    contrast_ok = old_bits == 0x3F800000 and golden_bits == 0x00000000 and production_bits == golden_bits
    checks.append(
      {
        "id": "binary64-contrast-preserved",
        "pass": contrast_ok,
        "detail": (
          f"binary64 whole-step bits {old_bits:#010x} (old model), golden "
          f"{golden_bits:#010x}, production {production_bits:#010x}"
        ),
      }
    )
  report = _finish_report("vector-parity" if subset is None else "zero-k", config, checks, vectors)
  report["inputs"] = _inputs_echo(config, ("vectors",) if subset is not None else ("vectors", "directed"))
  return report, vectors, sources


def _engine_vector_parity(config: dict, seed: int):
  return _engine_vectors(config, seed, None)


def _engine_zero_k(config: dict, seed: int):
  return _engine_vectors(config, seed, set(_ZERO_WORK_IDS))


def _engine_cutover(config: dict, seed: int):
  checks: list[dict] = []
  vectors: list[dict] = []
  sources: list[tuple[str, str]] = []

  legacy_path = Path(config["legacy_v4"])
  legacy_text = legacy_path.read_text(encoding="utf-8")
  legacy_doc = json.loads(legacy_text)
  legacy_is_v4 = legacy_doc.get("schema_version") == 4 and legacy_doc.get("compiler_abi") == "v4"
  checks.append(
    {
      "id": "legacy-artifact-is-unmodified-v4",
      "pass": legacy_is_v4,
      "detail": f"{legacy_path.name} sha256 {_sha256(legacy_path)}",
    }
  )
  try:
    parse_compiled_program(legacy_text)
    version_rejection = None
  except ValueError as exc:
    version_rejection = exc
  version_ok = (
    version_rejection is not None
    and "fingerprint" not in str(version_rejection)
    and ("recompile" in str(version_rejection) or "unsupported compiled schema" in str(version_rejection))
  )
  checks.append(
    {
      "id": "legacy-v4-rejected-at-version-gate",
      "pass": version_ok,
      "detail": "rejected at parse on schema/ABI version, never fingerprint: "
      + (str(version_rejection) if version_rejection else "accepted"),
    }
  )
  vectors.append(
    _record(
      "legacy-v4-parse",
      label="negative",
      passed=version_ok,
      detail="verbatim pre-P06 v4 artifact must fail the version gate",
      stage="parse",
      error=str(version_rejection) if version_rejection else "",
    )
  )

  # Positive replay of the current format, then the independent wrong-target
  # rejection on the same fresh artifact.
  directed = _load_json(config["directed"])
  case = next(item for item in directed["cases"] if item["id"] == "p06-grouping-k16")
  record, hw, _sim, _actual, artifact, bindings, seeds, c_address, c_bytes = _run_directed_case(
    case, seed, sources
  )
  vectors.append(record)
  checks.append({"id": f"vector:{record['id']}", "pass": record["pass"], "detail": record["detail"]})
  golden_hex = case["expected"]["output_hex"]
  serialized = serialize_compiled_program(artifact)
  reparsed = parse_compiled_program(serialized)
  sim2 = SimConfig(fidelity="full_memory", context_count=1, seed=seed)
  replay_loaded = load_program(reparsed, hw, sim2, actual_bindings=bindings)
  replay_store = ByteStore()
  for address, data in seeds:
    replay_store.seed_hbm(address, data)
  replay_result = Simulator(hw, sim2, byte_store=replay_store).run(replay_loaded)
  replay_actual = replay_store.read_hbm(c_address, c_bytes) if c_bytes else b""
  replay_ok = replay_result.completed and replay_actual == _hex_bytes(golden_hex)
  checks.append(
    {
      "id": "current-format-replay-runs",
      "pass": replay_ok,
      "detail": f"serialize->parse->load->run reproduces {golden_hex}",
    }
  )
  wrong_hw = HardwareConfig().with_overrides(boa_numeric_tile_k=8)
  try:
    load_program(reparsed, wrong_hw, sim2, actual_bindings=bindings)
    wrong_target_rejection = None
  except ValueError as exc:
    wrong_target_rejection = exc
  wrong_ok = wrong_target_rejection is not None and "target fingerprint mismatch" in str(
    wrong_target_rejection
  )
  checks.append(
    {
      "id": "current-format-wrong-target-rejected",
      "pass": wrong_ok,
      "detail": "fresh artifact loaded under a different numeric tile_k: "
      + (str(wrong_target_rejection) if wrong_target_rejection else "accepted"),
    }
  )
  vectors.append(
    _record(
      "current-format-wrong-target",
      label="negative",
      passed=wrong_ok,
      detail="independent of the legacy-version rejection above",
      stage="load",
      error=str(wrong_target_rejection) if wrong_target_rejection else "",
    )
  )
  report = _finish_report("cutover-reject", config, checks, vectors)
  report["inputs"] = _inputs_echo(config, ("directed", "legacy_v4"))
  return report, vectors, sources


ENGINES = {
  "vector-parity": _engine_vector_parity,
  "zero-k": _engine_zero_k,
  "cutover-reject": _engine_cutover,
}


# ---------------------------------------------------------------------------
# CLI + outputs
# ---------------------------------------------------------------------------


def _write_outputs(
  out: Path, case: str, report: dict, vectors: list[dict], sources: list[tuple[str, str]]
) -> None:
  out.mkdir(parents=True, exist_ok=True)
  (out / f"{case}.report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  (out / f"{case}.vectors.json").write_text(
    json.dumps(
      {"schema": 1, "case": case, "profile": report["profile"], "vectors": vectors},
      indent=2,
      sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
  )
  sources_dir = out / "sources"
  sources_dir.mkdir(parents=True, exist_ok=True)
  for name, text in sources:
    (sources_dir / name).write_text(text, encoding="utf-8")


def _run_case(case: str, out: Path, *, seed: int, config: dict | None = None) -> dict:
  if case not in ENGINES:
    raise ValueError(f"unknown numeric-profile case {case!r} (registered: {', '.join(CASES)})")
  resolved = config if config is not None else load_config(None)
  report, vectors, sources = ENGINES[case](resolved, seed)
  _write_outputs(Path(out), case, report, vectors, sources)
  return report


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    prog="python -m pipeline_validator.tests.test_boa_numeric_profile",
    description="P06 numeric-profile production-path vector gate",
  )
  parser.add_argument("--case", required=True, choices=CASES)
  parser.add_argument("--out", required=True, help="output directory for report/vectors/sources")
  parser.add_argument("--seed", type=int, default=1, help="recorded run seed and simulator seed")
  parser.add_argument("--config", default=None, help="optional strict JSON config (see module docstring)")
  args = parser.parse_args(argv)
  try:
    config = load_config(args.config)
  except ValueError as exc:
    print(f"config-error: {exc}", file=sys.stderr)
    return 2
  started = time.monotonic()
  report = _run_case(args.case, Path(args.out), seed=args.seed, config=config)
  elapsed = time.monotonic() - started
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / f"{args.case}.timing.json").write_text(
    json.dumps(
      {
        "schema": 1,
        "case": args.case,
        "seed": args.seed,
        "wall_clock_seconds": elapsed,
        "deterministic": False,
        "note": "elapsed time is environment-dependent; excluded from deterministic reproduction",
      },
      indent=2,
      sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
  )
  failed_checks = [item["id"] for item in report["checks"] if not item["pass"]]
  print(
    json.dumps(
      {
        "case": args.case,
        "status": report["status"],
        "counts": report["counts"],
        "failed_checks": failed_checks,
      },
      sort_keys=True,
    )
  )
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


def test_vector_parity_matches_committed_goldens(tmp_path):
  report = _run_case("vector-parity", Path(tmp_path), seed=20261028)
  assert report["status"] == "PASS", [c for c in report["checks"] if not c["pass"]]
  assert report["counts"]["full_bytes"] > 0
  assert report["counts"]["directed"] == 3
  assert report["counts"]["negative"] > 0


def test_zero_work_production_paths(tmp_path):
  report = _run_case("zero-k", Path(tmp_path), seed=20261028)
  assert report["status"] == "PASS", [c for c in report["checks"] if not c["pass"]]
  assert report["counts"]["vectors"] == len(_ZERO_WORK_IDS)


def test_cutover_rejections(tmp_path):
  report = _run_case("cutover-reject", Path(tmp_path), seed=20261028)
  assert report["status"] == "PASS", [c for c in report["checks"] if not c["pass"]]


@pytest.mark.parametrize("m", [0, 1])
def test_inline_epilogue_constants_respect_program_sram_capacity(m):
  """Constants occupy resident bytes even when the Matmul has no MAC work."""
  n = 300
  steps = [{
    "seq": 0, "k": 0, "storage_k": 1, "init": True, "final": True,
    "a": ["0x3f80"] * m, "b": ["0x3f80"] * n,
  }]
  epilogue = {
    "output_dtype": "i8", "bias": ["0x3fc00000"] * n, "scale": "0x40000000",
    "relu": True, "clip": ["0x00000000", "0x40400000"],
    "inv_scale": "0x3f000000", "zero_point": 1,
  }
  source = parse_workload_ir(build_case_module("epi_capacity", m, n, steps, "i8", epilogue))
  bindings, seeds, c_address, c_bytes = _layout(m, n, steps, "i8")
  sim = SimConfig(fidelity="full_memory", context_count=1)
  hw = HardwareConfig()
  artifact = compile_program(source, hw, sim, binding_assumptions=bindings)
  task = next(iter(artifact.entry.tasks.values()))
  program = next(iter(task.role_bindings.values())).tile_program
  # Bias alone needs 1200 B; the old estimate incorrectly remained 1024 B.
  for capacity in (1024, program.text_bytes - 1):
    with pytest.raises(ValueError, match="target program SRAM"):
      compile_program(
        source, hw.with_overrides(tile_program_sram_bytes=capacity), sim, binding_assumptions=bindings
      )
  exact_hw = hw.with_overrides(tile_program_sram_bytes=program.text_bytes)
  exact = compile_program(source, exact_hw, sim, binding_assumptions=bindings)
  replay = parse_compiled_program(serialize_compiled_program(exact))
  loaded = load_program(replay, exact_hw, sim, actual_bindings=bindings)
  store = ByteStore()
  for address, data in seeds:
    store.seed_hbm(address, data)
  result = Simulator(exact_hw, sim, byte_store=store).run(loaded)
  assert result.completed, result.reason
  # INIT K0: bias 1.5 * scale 2 = 3; quantize 3 * 0.5 -> RNE 2, + zp 1 = 3.
  assert store.read_hbm(c_address, c_bytes) == bytes([3]) * (m * n)
  pools = result.group_snapshot["shared_am_per_tile"]
  assert sum(pool["final_count"] for pool in pools) == 1
  assert all(pool["live_leases"] == 0 and pool["free_slots"] == pool["total_slots"] for pool in pools)


def test_loader_rejects_resealed_underreported_epilogue_footprint():
  steps = [{"seq": 0, "k": 0, "init": True, "final": True, "a": [], "b": []}]
  source = build_case_module("epi_tamper", 1, 300, steps, "f32", {"bias": ["0x3f800000"] * 300})
  bindings, _seeds, _address, _size = _layout(1, 300, steps, "f32")
  hw, sim = HardwareConfig(), SimConfig(fidelity="full_memory", context_count=1)
  artifact = compile_program(parse_workload_ir(source), hw, sim, binding_assumptions=bindings)
  task = next(iter(artifact.entry.tasks.values()))
  role = next(iter(task.role_bindings.values()))
  underreported = replace(role.tile_program, text_bytes=1024)
  underreported = replace(underreported, program_hash=program_digest(underreported))
  task = replace(task, role_bindings={role.role_id: replace(role, tile_program=underreported)})
  tampered = seal_program(replace(
    artifact, entry=replace(artifact.entry, tasks={task.binding_id: task}),
    static_effects={**artifact.static_effects, "program_text_bytes": {task.binding_id: (1024,)}},
  ))
  replay = parse_compiled_program(serialize_compiled_program(tampered))
  with pytest.raises(ValueError, match=r"text_bytes.*epilogue constants"):
    load_program(replay, hw, sim, actual_bindings=bindings)


def _typed_steps(artifact) -> list:
  task = next(iter(artifact.entry.tasks.values()))
  program = next(iter(task.role_bindings.values())).tile_program
  steps = [desc.params["matmul"] for desc in program.descriptors.values() if desc.kind == "BOA"]
  return sorted(steps, key=lambda matmul: matmul.seq)


@pytest.mark.parametrize("transpose", [(False, False), (True, False), (False, True), (True, True)])
def test_e_p08_11_transposed_layouts_replay_golden_bytes(transpose):
  """E-P08-11: asymmetric (m=3, n=2) storage-padded (storage_k=16 > k) A/B
  in all four stored layouts, through the serialized v6 artifact, must
  reproduce the frozen untransposed P03 golden bytes: the logical product
  is layout-independent."""
  case = next(item for item in _load_json(DEFAULT_VECTORS)["cases"] if item["id"] == "mm-rand-04")
  inp = case["input"]
  m, n, steps = inp["m"], inp["n"], inp["steps"]
  module = build_case_module("tr", m, n, steps, "f32", transpose=transpose)
  bindings, seeds, c_address, c_bytes = _layout(m, n, steps, "f32", transpose)
  result, actual, _hw, _sim, artifact = _compile_load_run(
    module, bindings, seeds, c_address, c_bytes, tile_k=inp["tile_k"], seed=20261028, steps=len(steps)
  )
  assert result.completed, result.reason
  assert actual == _hex_bytes(case["expected"]["output_hex"])
  replay = parse_compiled_program(serialize_compiled_program(artifact))
  descriptors = _typed_steps(replay)
  assert [(d.transpose_a, d.transpose_b) for d in descriptors] == [transpose] * len(steps)
  assert all(d.storage_k == 16 and d.k < d.storage_k for d in descriptors)


def test_e_p08_11_transpose_is_per_step_descriptor_state():
  """Transposing only step 0's A in a two-step chain must still give the
  golden bytes: flags live on each step's descriptor and never leak into the
  next step or the AM chain.  A flag contradicting the stored shape is a
  source-verifier error."""
  case = next(item for item in _load_json(DEFAULT_VECTORS)["cases"] if item["id"] == "mm-rand-04")
  inp = case["input"]
  m, n, steps = inp["m"], inp["n"], inp["steps"]
  per_step = [(True, False), (False, False)]
  module = build_case_module("tr", m, n, steps, "f32", transpose=per_step)
  bindings, seeds, c_address, c_bytes = _layout(m, n, steps, "f32", per_step)
  result, actual, _hw, _sim, artifact = _compile_load_run(
    module, bindings, seeds, c_address, c_bytes, tile_k=inp["tile_k"], seed=1, steps=len(steps)
  )
  assert result.completed, result.reason
  assert actual == _hex_bytes(case["expected"]["output_hex"])
  assert [(d.transpose_a, d.transpose_b) for d in _typed_steps(artifact)] == per_step
  plain = build_case_module("tr", m, n, steps, "f32")
  with pytest.raises((ParseError, ValueError, VerifyException)):
    parse_workload_ir(plain.replace("k = 6 seq = 1", "k = 6 seq = 1 transpose_a"))


def test_cli_rejects_unknown_case(tmp_path):
  with pytest.raises(SystemExit) as excinfo:
    main(["--case", "bogus", "--out", str(tmp_path), "--seed", "1"])
  assert excinfo.value.code == 2


def test_cli_rejects_unknown_config_key(tmp_path):
  config_path = Path(tmp_path) / "bad.json"
  config_path.write_text(json.dumps({"schema": 1, "bogus": True}), encoding="utf-8")
  assert (
    main(["--case", "zero-k", "--out", str(tmp_path / "o"), "--seed", "1", "--config", str(config_path)])
    == 2
  )


def test_empty_actual_binding_does_not_hide_nonempty_overlap():
  case = next(item for item in _load_json(DEFAULT_VECTORS)["cases"] if item["id"] == "mm-add-k0-negzero")
  inp = case["input"]
  source = build_case_module(case["id"], inp["m"], inp["n"], inp["steps"], "f32")
  bindings, _seeds, _address, _size = _layout(inp["m"], inp["n"], inp["steps"], "f32")
  hw, sim = HardwareConfig(), SimConfig(fidelity="full_memory")
  artifact = compile_program(parse_workload_ir(source), hw, sim, binding_assumptions=bindings)
  actual = dict(bindings)
  actual["A0"] = GlobalBinding("A0", 0, 8, "r")
  actual["A1"] = GlobalBinding("A1", 2, 0, "r")
  actual["B1"] = GlobalBinding("B1", 3, 0, "r")
  load_program(artifact, hw, sim, actual_bindings=actual)
  actual["B0"] = GlobalBinding("B0", 4, 8, "r")
  with pytest.raises(ValueError, match="overlap"):
    load_program(artifact, hw, sim, actual_bindings=actual)
