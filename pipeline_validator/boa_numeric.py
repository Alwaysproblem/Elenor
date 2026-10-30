"""Exact integer FP32 numerics for typed BOA matmul (``bf16_fp32_mac_rne_v1``).

Every numeric result is derived from raw bit patterns using stdlib exact
integer arithmetic.  Python ``float`` values never enter a numeric result and
there is deliberately no binary64/float fallback path.  Rounding is
round-to-nearest, ties-to-even, applied exactly once per operation (the FMA
multiplies and adds unrounded), subnormals are preserved (no FTZ/DAZ), and
RNE overflow produces same-sign infinity.  Tininess (UNDERFLOW) is judged
after rounding at target significand precision with an unbounded exponent
range -- not from the final encoded exponent -- so an encoded minimum normal
can still carry UF.  Exception flags are diagnostic IEEE bits (NV=16 OF=4
UF=2 NX=1; this module never sets DZ), not a runtime exception ABI.

Arithmetic special-value rules (frozen): arithmetic NaN output is positive
canonical ``0x7fc00000`` (``0x7fc0`` for BF16 conversion) with input sign and
payload dropped; only sNaN inputs raise NV; ``0 x Inf`` raises NV even when
the third FMA operand is qNaN; opposite-sign infinity addition raises NV;
opposite-sign exact zero cancellation yields ``+0`` (``-0 + +0 = +0``,
``-0 + -0 = -0``); multiplication zero sign is the operand sign XOR; an FMA
zero product and a same-signed zero accumulator keep that sign; a tiny
negative result rounded to zero keeps ``-0``.

Matmul semantics: inputs are row-major BF16, the AM accumulator is row-major
FP32.  The active ``k`` range is split into ``ceil(k / tile_k)`` contribution
blocks at the original indices (no compaction).  Each non-empty block starts
from ``p = +0`` and performs one ``fma_f32_rne`` per active lane in ascending
k; the first non-empty block of an INIT step overwrites AM without reading
it, and every other non-empty block applies ``add_f32_rne(AM, p)``.  Masked
or tail lanes beyond ``k`` perform no MAC, and a block without active lanes
performs no AM update at all (never ``AM + (+0)``).  An INIT step with no
active lanes initializes every cell to ``+0``; an ADD step with no active
lanes preserves the partial bytes bit-for-bit, including ``-0`` and sNaN
payloads.  Validation runs even for empty work.  INT8 output stays rejected
until P08 freezes the conversion policy.

This module does not import ``hardware/boa/reference`` (or anything outside
the standard library); parity with the frozen oracle is enforced only through
golden vectors consumed by ``pipeline_validator/tests/test_boa_numeric.py``.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

__all__ = [
  "add_f32_rne",
  "convert_output",
  "decode_f32_bits",
  "encode_f32_bits",
  "f32_to_bf16_rne",
  "fma_f32_rne",
  "matmul_f32",
]

# Diagnostic exception flags; NV=16 OF=4 UF=2 NX=1 per the frozen profile.
_NV = 0x10
_OF = 0x04
_UF = 0x02
_NX = 0x01
_CANONICAL_F32_NAN = 0x7FC00000
_CANONICAL_BF16_NAN = 0x7FC0
_F32_FRACTION_BITS = 23
_BF16_FRACTION_BITS = 7


def _bits_word(value, width: int = 32) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise ValueError(f"bit pattern must be an int, got {type(value).__name__}")
  if not 0 <= value < (1 << width):
    raise ValueError(f"bit pattern out of range for a {width}-bit word: {value!r}")
  return value


def _dimension(value, name: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise ValueError(f"{name} must be an int, got {type(value).__name__}")
  if value < 0:
    raise ValueError(f"{name} must be non-negative, got {value!r}")
  return value


def _buffer(value, name: str):
  if not isinstance(value, (bytes, bytearray, memoryview)):
    raise ValueError(f"{name} must be a bytes-like payload")
  return value


def _is_nan(bits: int) -> bool:
  return bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF != 0


def _is_snan(bits: int) -> bool:
  return bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF != 0 and not bits & 0x00400000


def _is_inf(bits: int) -> bool:
  return bits & 0x7FFFFFFF == 0x7F800000


def _is_zero(bits: int) -> bool:
  return bits & 0x7FFFFFFF == 0


def _decode_finite(bits: int) -> tuple[int, int, int]:
  """Exact finite value as ``(negative, magnitude, shift)``:

  ``value = (-1 if negative else 1) * magnitude / 2**shift`` with
  ``magnitude >= 0`` and ``shift >= 0``.  Zero decodes as magnitude 0.
  """
  fraction = bits & 0x007FFFFF
  exponent = (bits >> 23) & 0xFF
  if exponent:
    magnitude = fraction | 0x00800000
    shift = 150 - exponent
  else:
    magnitude = fraction
    shift = 149
  if shift < 0:
    magnitude <<= -shift
    shift = 0
  return bits >> 31, magnitude, shift


def _rne_quotient(magnitude: int, divisor: int) -> tuple[int, bool]:
  """RNE quotient ``magnitude // divisor`` plus the inexact indicator."""
  quotient, remainder = divmod(magnitude, divisor)
  remainder += remainder
  inexact = remainder != 0
  if remainder > divisor or (remainder == divisor and quotient & 1):
    quotient += 1
  return quotient, inexact


def _rne_round(
  negative: bool, magnitude: int, shift: int, fraction_bits: int, zero_negative: int
) -> tuple[int, int]:
  """Round the exact value ``magnitude / 2**shift`` once under RNE into an
  8-exponent-bit binary format with ``fraction_bits`` fraction bits.

  ``zero_negative`` selects the sign of an exactly-zero result; tininess is
  judged with an unbounded exponent range (tininess-after).
  """
  sign_mask = 1 << (8 + fraction_bits)
  if not magnitude:
    return (sign_mask if zero_negative else 0), 0
  sign_bits = sign_mask if negative else 0
  # The denominator is a power of two, so |value| lies in [2**e, 2**(e+1)]
  # with no binade correction needed.
  exponent = magnitude.bit_length() - 1 - shift
  quantum = (exponent if exponent > -126 else -126) - fraction_bits
  tail = -shift - quantum
  if tail >= 0:
    rounded = magnitude << tail
    inexact = False
  else:
    rounded, inexact = _rne_quotient(magnitude, 1 << -tail)
  flags = _NX if inexact else 0
  implicit = 1 << fraction_bits
  if exponent >= -126:
    if rounded >= implicit << 1:
      rounded >>= 1
      exponent += 1
    if exponent > 127:
      return sign_bits | (255 << fraction_bits), _OF | _NX
    return sign_bits | ((exponent + 127) << fraction_bits) | (rounded - implicit), flags
  # Tininess-after: re-round at target precision over an unbounded exponent
  # range; an encoded minimum normal (rounded == implicit) can still be tiny.
  tiny = exponent < -127
  if exponent == -127:
    head = 127 + fraction_bits - shift
    if head >= 0:
      unbounded = magnitude << head
    else:
      unbounded, _ = _rne_quotient(magnitude, 1 << -head)
    tiny = unbounded < implicit << 1
  if tiny and inexact:
    flags |= _UF
  return sign_bits | rounded, flags


def _fma_bits(a: int, b: int, c: int) -> tuple[int, int]:
  invalid_product = (_is_inf(a) and _is_zero(b)) or (_is_inf(b) and _is_zero(a))
  if _is_nan(a) or _is_nan(b) or _is_nan(c):
    invalid = invalid_product or _is_snan(a) or _is_snan(b) or _is_snan(c)
    return _CANONICAL_F32_NAN, (_NV if invalid else 0)
  if invalid_product:
    return _CANONICAL_F32_NAN, _NV
  product_negative = (a >> 31) ^ (b >> 31)
  if _is_inf(a) or _is_inf(b):
    if _is_inf(c) and product_negative != (c >> 31):
      return _CANONICAL_F32_NAN, _NV
    return (product_negative << 31) | 0x7F800000, 0
  if _is_inf(c):
    return c, 0
  if _is_zero(a) or _is_zero(b):
    # Exact zero product: a finite c passes through unchanged, and a zero c
    # keeps its sign only when the product zero carries the same sign.
    if _is_zero(c) and product_negative != (c >> 31):
      return 0, 0
    return c, 0
  _a_negative, a_magnitude, a_shift = _decode_finite(a)
  _b_negative, b_magnitude, b_shift = _decode_finite(b)
  c_negative, c_magnitude, c_shift = _decode_finite(c)
  product_shift = a_shift + b_shift
  common = product_shift if product_shift > c_shift else c_shift
  scaled_c = c_magnitude << (common - c_shift)
  if c_negative:
    scaled_c = -scaled_c
  product_scaled = a_magnitude * b_magnitude << (common - product_shift)
  if product_negative:
    product_scaled = -product_scaled
  total = product_scaled + scaled_c
  # a and b are nonzero here (zero operands took the fast path above), so an
  # exact-zero total is an opposite-sign cancellation and yields +0.
  return _rne_round(total < 0, abs(total), common, _F32_FRACTION_BITS, 0)


def _add_bits(a: int, b: int) -> tuple[int, int]:
  if _is_nan(a) or _is_nan(b):
    return _CANONICAL_F32_NAN, (_NV if (_is_snan(a) or _is_snan(b)) else 0)
  if _is_inf(a) and _is_inf(b) and (a >> 31) != (b >> 31):
    return _CANONICAL_F32_NAN, _NV
  if _is_inf(a):
    return a, 0
  if _is_inf(b):
    return b, 0
  if _is_zero(a) and _is_zero(b):
    # -0 + -0 = -0; any other zero combination is +0.  Exact, no flags.
    return (a if a == b else 0), 0
  a_negative, a_magnitude, a_shift = _decode_finite(a)
  b_negative, b_magnitude, b_shift = _decode_finite(b)
  common = a_shift if a_shift > b_shift else b_shift
  left = a_magnitude << (common - a_shift)
  right = b_magnitude << (common - b_shift)
  if a_negative:
    left = -left
  if b_negative:
    right = -right
  total = left + right
  # Exact opposite-sign cancellation yields +0, so zero_negative is fixed 0.
  return _rne_round(total < 0, abs(total), common, _F32_FRACTION_BITS, 0)


def _bf16_bits(bits: int) -> tuple[int, int]:
  if _is_nan(bits):
    return _CANONICAL_BF16_NAN, (_NV if _is_snan(bits) else 0)
  if _is_inf(bits):
    return bits >> 16, 0
  negative, magnitude, shift = _decode_finite(bits)
  return _rne_round(bool(negative), magnitude, shift, _BF16_FRACTION_BITS, bits >> 31)


def fma_f32_rne(a_bits: int, b_bits: int, c_bits: int) -> tuple[int, int]:
  """Single-rounding fused multiply-add of raw FP32 bit patterns.

  Returns ``(result_bits, flags)``; the product and the addend are combined
  unrounded and rounded once under RNE.  Inputs must be unsigned 32-bit int
  bit patterns (bool is rejected).
  """
  return _fma_bits(_bits_word(a_bits), _bits_word(b_bits), _bits_word(c_bits))


def add_f32_rne(a_bits: int, b_bits: int) -> tuple[int, int]:
  """RNE FP32 addition of raw bit patterns, returning ``(bits, flags)``."""
  return _add_bits(_bits_word(a_bits), _bits_word(b_bits))


def f32_to_bf16_rne(bits: int) -> tuple[int, int]:
  """RNE FP32 -> BF16 conversion of a raw bit pattern.

  Returns ``(bf16_bits, flags)``.  NaN canonicalizes to positive ``0x7fc0``
  (sNaN raises NV), infinity keeps its sign, and tininess follows the
  unbounded-exponent rule.
  """
  return _bf16_bits(_bits_word(bits))


def decode_f32_bits(data: bytes) -> list[int]:
  """Decode little-endian FP32 bytes into raw 32-bit bit patterns.

  No float decoding and no NaN canonicalization happens here; payloads and
  signed zeros survive verbatim.
  """
  _buffer(data, "FP32 payload")
  if len(data) % 4:
    raise ValueError("FP32 payload must be a multiple of 4 bytes")
  count = len(data) // 4
  if not count:
    return []
  return list(struct.unpack_from(f"<{count}I", data))


def encode_f32_bits(words) -> bytes:
  """Encode raw 32-bit FP32 bit patterns as little-endian bytes."""
  words = tuple(words)
  if not words:
    return b""
  return struct.pack(f"<{len(words)}I", *(_bits_word(word) for word in words))


def convert_output(words, dtype: str) -> bytes:
  """Convert final AM words (raw FP32 bit patterns) to the output dtype.

  ``f32`` writes the raw bits verbatim (sNaN payloads and ``-0`` survive);
  ``bf16`` canonicalizes NaN to ``0x7fc0`` and applies one final RNE; ``i8``
  stays rejected until the P08 conversion policy lands.
  """
  if dtype == "f32":
    return encode_f32_bits(words)
  if dtype == "bf16":
    return b"".join(
      _bf16_bits(_bits_word(word))[0].to_bytes(2, "little") for word in words
    )
  if dtype == "i8":
    raise ValueError("INT8 AM output conversion policy is not frozen")
  raise ValueError(f"unsupported AM output dtype '{dtype}'")


def _bf16_operand_words(data, count: int) -> tuple[int, ...]:
  """Widen raw BF16 words to FP32 bit patterns by an exact 16-bit shift."""
  if not count:
    return ()
  return tuple(word << 16 for word in struct.unpack_from(f"<{count}H", data))


def _active_blocks(
  k: int, tile_k: int, mask: Sequence[bool] | None
) -> list[tuple[int, Sequence[int]]]:
  """Contribution blocks ``(start_k, lanes)`` over the original indices.

  Blocks cover ``[0, k)`` in ``tile_k`` strides; masked-out (or tail) lanes
  are never listed, and blocks without active lanes are omitted entirely --
  they must not execute a MAC or an AM update.
  """
  blocks: list[tuple[int, Sequence[int]]] = []
  for start in range(0, k, tile_k):
    end = start + tile_k
    if end > k:
      end = k
    if mask is None:
      blocks.append((start, range(start, end)))
    else:
      lanes = [lane for lane in range(start, end) if mask[lane]]
      if lanes:
        blocks.append((start, lanes))
  return blocks


def _init_zero_record(i: int, j: int) -> dict:
  return {
    "kind": "init_zero",
    "i": i,
    "j": j,
    "start_k": None,
    "valid_indices": [],
    "partial_bits": [],
    "am_bits": 0,
    "fma_flags": 0,
    "add_flags": 0,
    "flags": 0,
    "overwrite": True,
  }


def matmul_f32(
  a: bytes,
  b: bytes,
  *,
  m: int,
  n: int,
  k: int,
  accumulate: bool,
  partial: bytes | None,
  tile_k: int = 8,
  storage_k: int | None = None,
  mask=None,
  block_trace: list | None = None,
) -> bytes:
  """Evaluate one typed matmul step under ``bf16_fp32_mac_rne_v1``.

  ``a`` is row-major ``[m, storage_k]`` BF16 and ``b`` is row-major
  ``[storage_k, n]`` BF16 (``storage_k`` defaults to ``k``).  ``k`` is the
  logical K extent before masking; ``mask`` is an optional boolean sequence
  of ``storage_k`` length whose tail at ``k..storage_k`` must be inactive.
  ``accumulate=False`` marks the INIT step: the first non-empty contribution
  block overwrites AM and no partial is read.  ``accumulate=True`` marks an
  ADD step and requires ``partial`` with exactly ``m*n*4`` bytes.  Zero
  dimensions and zero active lanes are legal: INIT yields ``+0`` cells, ADD
  preserves the partial bytes exactly, and ``m*n == 0`` yields empty output.
  Validation always runs, even for empty work.

  ``block_trace`` is a diagnostic-only list; when provided, one record per
  executed block is appended with raw integer bit patterns: ``kind`` (``"contrib"`` or
  ``"init_zero"``), ``i``, ``j``, ``start_k`` (``None`` for ``init_zero``),
  ``valid_indices``, ``partial_bits`` (per-MAC FP32 bit patterns),
  ``am_bits``, ``fma_flags``, ``add_flags``, ``flags`` and ``overwrite``.
  Callers that need a step ``seq`` attach it outside this call.  Leaving it
  ``None`` keeps the default path free of per-block allocations.
  """
  _dimension(m, "m")
  _dimension(n, "n")
  _dimension(k, "k")
  if isinstance(tile_k, bool) or not isinstance(tile_k, int) or tile_k < 1:
    raise ValueError("tile_k must be a positive integer")
  storage = k if storage_k is None else _dimension(storage_k, "storage_k")
  if storage < k:
    raise ValueError(f"storage_k {storage} is below k {k}")
  _buffer(a, "A operand")
  _buffer(b, "B operand")
  if len(a) != m * storage * 2:
    raise ValueError("A byte count does not match (m, storage_k) BF16")
  if len(b) != storage * n * 2:
    raise ValueError("B byte count does not match (storage_k, n) BF16")
  mask_tuple = None
  if mask is not None:
    if len(mask) != storage:
      raise ValueError(f"mask has {len(mask)} entries but storage_k={storage}")
    for index, flag in enumerate(mask):
      if not isinstance(flag, bool):
        raise ValueError(f"mask[{index}] must be a boolean")
    mask_tuple = tuple(mask)
    if any(mask_tuple[k:]):
      raise ValueError("mask lanes at k..storage_k must be inactive")
  if block_trace is not None and not isinstance(block_trace, list):
    raise ValueError("block_trace must be a list when provided")
  cells = m * n
  if accumulate:
    if partial is None:
      raise ValueError("ADD step requires a full FP32 partial")
    _buffer(partial, "ADD partial")
    partial_bytes = bytes(partial)
    if len(partial_bytes) != cells * 4:
      raise ValueError("ADD partial byte count does not match (m, n) FP32")
    partial_words = struct.unpack_from(f"<{cells}I", partial_bytes)
  else:
    partial_bytes = None
    partial_words = None
  if not cells:
    return b""

  blocks = _active_blocks(k, tile_k, mask_tuple)
  if not blocks:
    if accumulate:
      # ADD with no active lanes: preserve the partial bytes bit-exactly.
      assert partial_bytes is not None
      return partial_bytes
    if block_trace is not None:
      for row in range(m):
        for col in range(n):
          block_trace.append(_init_zero_record(row, col))
    return bytes(cells * 4)

  a_words = _bf16_operand_words(a, m * storage)
  b_words = _bf16_operand_words(b, storage * n)
  fma = _fma_bits
  add = _add_bits
  init = not accumulate
  out = [0] * cells
  for row in range(m):
    a_base = row * storage
    out_base = row * n
    for col in range(n):
      am = partial_words[out_base + col] if partial_words is not None else 0
      first = init
      for start, lanes in blocks:
        p = 0
        fma_flags = 0
        partial_bits: list[int] | None = [] if block_trace is not None else None
        for lane in lanes:
          p, flags = fma(a_words[a_base + lane], b_words[lane * n + col], p)
          if partial_bits is not None:
            fma_flags |= flags
            partial_bits.append(p)
        if first:
          am = p
          first = False
          overwrite = True
          add_flags = 0
        else:
          am, add_flags = add(am, p)
          overwrite = False
        if block_trace is not None:
          block_trace.append(
            {
              "kind": "contrib",
              "i": row,
              "j": col,
              "start_k": start,
              "valid_indices": list(lanes),
              "partial_bits": partial_bits,
              "am_bits": am,
              "fma_flags": fma_flags,
              "add_flags": add_flags,
              "flags": fma_flags | add_flags,
              "overwrite": overwrite,
            }
          )
      out[out_base + col] = am
  return struct.pack(f"<{cells}I", *out)
