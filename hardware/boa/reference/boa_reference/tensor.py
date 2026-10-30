"""Independent golden Matmul oracle for profile bf16_fp32_mac_rne_v1.

Frozen per-cell schedule (P03 contract; design proposal section 3.2):
for every output cell (i, j), steps are applied in seq order.  Within a
step the active k lanes are cut along the physical contribution
boundaries [0:tile_k), [tile_k:2*tile_k), ...  Each active block starts
p = +0 and accumulates p = fma_f32_rne(a, b, p) once per active lane in
ascending k.  The first active block of an INIT step overwrites AM
without reading it; every later active block of any step updates
AM = add_f32_rne(AM, p) exactly once.  FINAL applies the epilogue once
per logical chain; intermediate steps never touch the epilogue.

Frozen mask bypass rule (stronger than zero fill, deliberately explicit,
not implicit):

* Physical contribution boundaries use the ORIGINAL k indices.  Masked
  lanes are never compacted or regrouped into different blocks.
* A masked or padded lane executes no MAC at all.  Substituting
  fma(0, 0, p) is forbidden: -0 + +0 would flip the sign of a zero
  partial, and 0 x Inf would fabricate a NaN.
* A contribution block whose lanes are all inactive is skipped without
  touching AM (no phantom add of +0).
* An INIT step with no active block initializes the cell to +0.  An ADD
  step with no active block (including k == 0) leaves the previous AM
  bits exactly untouched.

The oracle starts every AM cell at a poison NaN internally so that any
INIT wrongly reading the previous occupant is caught by exact bit
comparison.  This is an internal proof-of-overwrite device, NOT a public
initial-C input mode.

m == 0 or n == 0 produces no output, no MAC and no contribution block,
but all parameters and steps are still validated.  storage_k defaults to
k (never a rounded-up allocation); lanes at k..storage_k are inactive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import scalar
from .epilogue import (
  Epilogue,
  apply_epilogue,
  json_trace,
  output_dtype_width,
  result_dtype,
  validate_epilogue,
)

_POISON_AM = scalar.CANONICAL_NAN
_PLUS_ZERO = 0x00000000


@dataclass(frozen=True)
class MatmulStep:
  k: int
  a: tuple[int, ...]
  b: tuple[int, ...]
  init: bool
  final: bool
  seq: int
  storage_k: int | None = None
  mask: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class TensorResult:
  am_bits: tuple[int, ...]
  output_bytes: bytes
  output_bits: tuple[int, ...]
  blocks: list[dict]
  epilogue_trace: list[dict]
  mac_count: int
  am_add_count: int
  initialization_count: int
  m: int
  n: int
  tile_k: int

  def as_dict(self) -> dict:
    dtype = result_dtype(self.epilogue_trace)
    return {
      "m": self.m,
      "n": self.n,
      "tile_k": self.tile_k,
      "am_bits": [f"0x{v:08x}" for v in self.am_bits],
      "output_bits": [f"0x{v:0{2 * output_dtype_width(dtype)}x}" for v in self.output_bits],
      "output_hex": bytes(self.output_bytes).hex(),
      "blocks": [dict(record) for record in self.blocks],
      "epilogue_trace": json_trace(self.epilogue_trace),
      "mac_count": self.mac_count,
      "am_add_count": self.am_add_count,
      "initialization_count": self.initialization_count,
    }


def _require_index(value, name, minimum, maximum=None):
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"{name} must be an int")
  if value < minimum or (maximum is not None and value > maximum):
    raise ValueError(f"{name} out of range: {value!r}")


def _require_words(words, length, limit, name):
  if isinstance(words, (str, bytes)) or not isinstance(words, (list, tuple)):
    raise TypeError(f"{name} must be a sequence of bit patterns")
  if len(words) != length:
    raise ValueError(f"{name} has {len(words)} words but {length} are required")
  for index, word in enumerate(words):
    if isinstance(word, bool) or not isinstance(word, int):
      raise TypeError(f"{name}[{index}] must be an integer bit pattern")
    if not 0 <= word <= limit:
      raise ValueError(f"{name}[{index}] out of range: {word!r}")


def _parse_steps(m, n, steps):
  if isinstance(steps, (str, bytes)) or not isinstance(steps, (list, tuple)):
    raise TypeError("steps must be a sequence of MatmulStep")
  if not steps:
    raise ValueError("step sequence must be nonempty")
  last = len(steps) - 1
  parsed = []
  for index, step in enumerate(steps):
    if not isinstance(step, MatmulStep):
      raise TypeError(f"steps[{index}] must be a MatmulStep")
    _require_index(step.seq, f"steps[{index}].seq", 0)
    if step.seq != index:
      raise ValueError(f"steps[{index}].seq is {step.seq}, expected {index}")
    if not isinstance(step.init, bool) or not isinstance(step.final, bool):
      raise TypeError(f"steps[{index}].init/.final must be bool")
    if step.init != (index == 0):
      raise ValueError(f"steps[{index}] must be INIT exactly when seq==0")
    if step.final != (index == last):
      raise ValueError(f"steps[{index}] must be FINAL exactly on the last step")
    _require_index(step.k, f"steps[{index}].k", 0)
    if step.storage_k is None:
      storage = step.k
    else:
      _require_index(step.storage_k, f"steps[{index}].storage_k", 0)
      if step.storage_k < step.k:
        raise ValueError(f"steps[{index}].storage_k {step.storage_k} is below k {step.k}")
      storage = step.storage_k
    _require_words(step.a, m * storage, 0xFFFF, f"steps[{index}].a")
    _require_words(step.b, storage * n, 0xFFFF, f"steps[{index}].b")
    if step.mask is None:
      mask = None
    else:
      if isinstance(step.mask, (str, bytes)) or not isinstance(step.mask, (list, tuple)):
        raise TypeError(f"steps[{index}].mask must be a sequence of bool")
      if len(step.mask) != storage:
        raise ValueError(f"steps[{index}].mask has {len(step.mask)} entries but storage_k={storage}")
      for lane, flag in enumerate(step.mask):
        if not isinstance(flag, bool):
          raise TypeError(f"steps[{index}].mask[{lane}] must be bool")
      if any(step.mask[step.k :]):
        raise ValueError(f"steps[{index}].mask tail entries at k..storage_k must be false")
      mask = tuple(step.mask)
    parsed.append((step, storage, mask))
  return parsed


def _active_blocks(k, tile_k, mask):
  out = []
  for start in range(0, k, tile_k):
    end = min(start + tile_k, k)
    lanes = [kk for kk in range(start, end) if mask is None or mask[kk]]
    out.append((start, lanes))
  return out


def _contrib_record(i, j, step, start, lanes, partials, am_word, fma_flags, add_flags, overwrite):
  return {
    "kind": "contrib",
    "i": i,
    "j": j,
    "seq": step.seq,
    "start_k": start,
    "valid_indices": list(lanes),
    "partial_bits": [f"0x{v:08x}" for v in partials],
    "am_bits": f"0x{am_word:08x}",
    "fma_flags": fma_flags,
    "add_flags": add_flags,
    "flags": fma_flags | add_flags,
    "overwrite": overwrite,
  }


def _init_zero_record(i, j, step):
  return {
    "kind": "init_zero",
    "i": i,
    "j": j,
    "seq": step.seq,
    "start_k": None,
    "valid_indices": [],
    "partial_bits": [],
    "am_bits": f"0x{_PLUS_ZERO:08x}",
    "fma_flags": 0,
    "add_flags": 0,
    "flags": 0,
    "overwrite": True,
  }


def matmul_reference(
  m: int, n: int, steps: Sequence[MatmulStep], tile_k: int = 8, epilogue: Epilogue | None = None
) -> TensorResult:
  _require_index(m, "m", 0)
  _require_index(n, "n", 0)
  _require_index(tile_k, "tile_k", 1)
  parsed = _parse_steps(m, n, steps)
  epilogue = Epilogue() if epilogue is None else epilogue
  validate_epilogue(epilogue, n)
  am = [_POISON_AM] * (m * n)
  blocks: list[dict] = []
  mac_count = 0
  am_add_count = 0
  initialization_count = 0
  for step, storage, mask in parsed:
    for i in range(m):
      a_row = step.a[i * storage : (i + 1) * storage]
      for j in range(n):
        cell = i * n + j
        initialized = False
        for start, lanes in _active_blocks(step.k, tile_k, mask):
          if not lanes:
            continue
          p = _PLUS_ZERO
          partials = []
          fma_flags = 0
          for k in lanes:
            result = scalar.fma_f32_rne(
              scalar.bf16_to_f32(a_row[k]), scalar.bf16_to_f32(step.b[k * n + j]), p
            )
            p = result.bits
            fma_flags |= result.flags
            partials.append(p)
            mac_count += 1
          add_flags = 0
          if step.init and not initialized:
            am[cell] = p
            initialization_count += 1
            initialized = True
            overwrite = True
          else:
            result = scalar.add_f32_rne(am[cell], p)
            am[cell] = result.bits
            add_flags = result.flags
            am_add_count += 1
            overwrite = False
          blocks.append(
            _contrib_record(i, j, step, start, lanes, partials, am[cell], fma_flags, add_flags, overwrite)
          )
        if step.init and not initialized:
          am[cell] = _PLUS_ZERO
          initialization_count += 1
          blocks.append(_init_zero_record(i, j, step))
  applied = apply_epilogue(am, m, n, epilogue)
  output_bytes = applied.output_bytes
  output_bits = applied.output_bits
  epilogue_trace = applied.trace
  return TensorResult(
    am_bits=tuple(am),
    output_bytes=output_bytes,
    output_bits=output_bits,
    blocks=blocks,
    epilogue_trace=epilogue_trace,
    mac_count=mac_count,
    am_add_count=am_add_count,
    initialization_count=initialization_count,
    m=m,
    n=n,
    tile_k=tile_k,
  )
