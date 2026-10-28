"""Numerical BOA Matmul primitives for the full-memory simulator.

Inputs are row-major BF16; the AM accumulator is row-major FP32; the final
output is converted once, after the last partial sum.  BF16 conversion uses
round-to-nearest-even.  INT8 output is intentionally rejected until the
quantization policy (scale, clamp, exceptional values) is frozen; the source
dialect already accepts the dtype so the contract is visible, but the
numeric backend must not invent a policy.
"""

from __future__ import annotations

import struct

_DTYPE_BYTES = {"f32": 4, "bf16": 2, "i8": 1}


def _bf16_to_float(bits: int) -> float:
  return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _float_to_bf16_bits(value: float) -> int:
  bits = struct.unpack("<I", struct.pack("<f", value))[0]
  if bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF:
    return 0x7FC0 | ((bits >> 16) & 0x8000)  # quiet NaN, keep sign
  rounded = bits + 0x7FFF + ((bits >> 16) & 1)
  return (rounded >> 16) & 0xFFFF


def decode_bf16(data: bytes) -> list[float]:
  if len(data) % 2:
    raise ValueError("BF16 payload must have an even byte count")
  return [_bf16_to_float(int.from_bytes(data[i : i + 2], "little")) for i in range(0, len(data), 2)]


def encode_f32(values: list[float]) -> bytes:
  return struct.pack(f"<{len(values)}f", *values)


def decode_f32(data: bytes) -> list[float]:
  if len(data) % 4:
    raise ValueError("FP32 payload must be a multiple of 4 bytes")
  return list(struct.unpack(f"<{len(data) // 4}f", data))


def encode_bf16(values: list[float]) -> bytes:
  return b"".join(_float_to_bf16_bits(value).to_bytes(2, "little") for value in values)


def convert_output(values: list[float], dtype: str) -> bytes:
  if dtype == "f32":
    return encode_f32(values)
  if dtype == "bf16":
    return encode_bf16(values)
  if dtype == "i8":
    raise ValueError("INT8 AM output conversion policy is not frozen")
  raise ValueError(f"unsupported AM output dtype '{dtype}'")


def matmul_f32(
  a: bytes,
  b: bytes,
  *,
  m: int,
  n: int,
  k: int,
  accumulate: bool,
  partial: bytes | None,
) -> bytes:
  """Return row-major FP32 ``A[m,k] @ B[k,n]`` plus an optional partial."""
  if m < 1 or n < 1 or k < 1:
    raise ValueError("matmul dimensions must be positive")
  if len(a) != m * k * 2:
    raise ValueError("A byte count does not match (m, k) BF16")
  if len(b) != k * n * 2:
    raise ValueError("B byte count does not match (k, n) BF16")
  a_values = decode_bf16(a)
  b_values = decode_bf16(b)
  if accumulate:
    if partial is None or len(partial) != m * n * 4:
      raise ValueError("ADD step requires a full FP32 partial")
    result = decode_f32(partial)
  else:
    result = [0.0] * (m * n)
  for row in range(m):
    a_base = row * k
    out_base = row * n
    for kk in range(k):
      avalue = a_values[a_base + kk]
      b_base = kk * n
      for col in range(n):
        result[out_base + col] += avalue * b_values[b_base + col]
  return encode_f32(result)
