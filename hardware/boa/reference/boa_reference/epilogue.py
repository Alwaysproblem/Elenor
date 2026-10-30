"""FINAL-only epilogue for the bf16_fp32_mac_rne_v1 numeric profile.

Frozen policy (P03 contract; design proposal section 3.2):

* Stage order per element: per-column bias add, scalar scale multiply,
  ReLU, clip, output conversion.  Disabled stages execute nothing.
  Enabled stages always execute their FP32 RNE primitive, even when the
  parameter is a numeric identity (scale=+1.0 still multiplies).
* ReLU propagates the canonical NaN; -0, negative finite values and -Inf
  become +0.  Positive values keep their exact bits.
* Clip propagates the canonical NaN.  Comparisons are strict, so an input
  equal to a bound keeps its exact input bits (including -0).
* A signalling NaN seen by ReLU or clip is diagnosed NV in the trace.
  This is a software diagnostic, not a runtime trap ABI.
* f32 output preserves AM bits verbatim (no arithmetic, no
  canonicalization).  bf16 converts exactly once via f32_to_bf16_rne.
* i8: mul_f32_rne(y, inv_scale), then exact integer RNE, then exact
  integer + zero_point, then saturation to [-128, 127].  Reviewed and
  frozen special policy: scaled NaN -> zero_point, scaled +Inf -> 127,
  scaled -Inf -> -128.  A product that overflows to Inf follows the same
  endpoints.  This is software policy; it is not derived from a
  HardFloat probe result.

All parameters are validated before any output is produced, including
for m == 0 or n == 0.  Bias must be exactly n FP32 words.  Exceptions:
TypeError for wrong Python types, ValueError for out-of-range or
inconsistent parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import scalar

_NV_FLAG = 0x10
_DTYPES = ("f32", "bf16", "i8")
_I8_MIN = -128
_I8_MAX = 127
_POS_INF = 0x7F800000
_WIDTHS = {"f32": 4, "bf16": 2, "i8": 1}
_BIT_KEYS = ("before", "after", "scaled", "input")


@dataclass(frozen=True)
class Epilogue:
  bias: tuple[int, ...] | None = None
  scale: int | None = None
  relu: bool = False
  clip: tuple[int, int] | None = None
  output_dtype: str = "f32"
  inv_scale: int | None = None
  zero_point: int = 0


@dataclass(frozen=True)
class EpilogueResult:
  output_bytes: bytes
  output_bits: tuple[int, ...]
  trace: list[dict]

  def as_dict(self) -> dict:
    dtype = result_dtype(self.trace)
    return {
      "output_bits": [_hex_bits(v, output_dtype_width(dtype)) for v in self.output_bits],
      "output_hex": bytes(self.output_bytes).hex(),
      "trace": json_trace(self.trace),
    }


def output_dtype_width(dtype: str) -> int:
  if dtype not in _WIDTHS:
    raise ValueError(f"unknown output dtype {dtype!r}")
  return _WIDTHS[dtype]


def _hex_bits(value: int, width_bytes: int) -> str:
  return f"0x{value:0{width_bytes * 2}x}"


def result_dtype(trace: list[dict]) -> str:
  for entry in trace:
    for stage in entry.get("stages", ()):
      if stage.get("stage") == "output":
        return str(stage.get("dtype", "f32"))
  return "f32"


def _stage_json(stage: dict, dtype: str) -> dict:
  is_output = stage.get("stage") == "output"
  record = {}
  for key, value in stage.items():
    if key in _BIT_KEYS and isinstance(value, int) and not isinstance(value, bool):
      if key == "after" and is_output:
        width = output_dtype_width(dtype)
      else:
        width = 4
      record[key] = _hex_bits(value, width)
    else:
      record[key] = value
  return record


def json_trace(trace: list[dict]) -> list[dict]:
  out = []
  for entry in trace:
    dtype = "f32"
    for stage in entry["stages"]:
      if stage.get("stage") == "output":
        dtype = str(stage.get("dtype", "f32"))
    out.append(
      {
        "index": entry["index"],
        "row": entry["row"],
        "col": entry["col"],
        "input": _hex_bits(entry["input"], 4),
        "stages": [_stage_json(stage, dtype) for stage in entry["stages"]],
      }
    )
  return out


def _require_index(value, name, minimum, maximum=None):
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"{name} must be an int")
  if value < minimum or (maximum is not None and value > maximum):
    raise ValueError(f"{name} out of range: {value!r}")


def _require_word(value, name):
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"{name} must be an integer FP32 bit pattern")
  if not 0 <= value <= 0xFFFFFFFF:
    raise ValueError(f"{name} is not a 32-bit pattern: {value!r}")


def _require_words(words, name):
  if isinstance(words, (str, bytes)) or not isinstance(words, (list, tuple)):
    raise TypeError(f"{name} must be a sequence of FP32 bit patterns")
  for index, word in enumerate(words):
    _require_word(word, f"{name}[{index}]")


def validate_epilogue(spec: Epilogue, n: int) -> None:
  if not isinstance(spec, Epilogue):
    raise TypeError("spec must be an Epilogue")
  _require_index(n, "n", 0)
  if spec.output_dtype not in _DTYPES:
    raise ValueError(f"unknown output_dtype {spec.output_dtype!r}")
  if not isinstance(spec.relu, bool):
    raise TypeError("relu must be bool")
  if spec.bias is not None:
    if isinstance(spec.bias, (str, bytes)) or not isinstance(spec.bias, (list, tuple)):
      raise TypeError("bias must be a sequence of FP32 bit patterns")
    if len(spec.bias) != n:
      raise ValueError(f"bias has {len(spec.bias)} words but n={n}")
    _require_words(spec.bias, "bias")
  if spec.scale is not None:
    _require_word(spec.scale, "scale")
  if spec.clip is not None:
    if isinstance(spec.clip, (str, bytes)) or not isinstance(spec.clip, (list, tuple)):
      raise TypeError("clip must be a pair of FP32 bit patterns")
    if len(spec.clip) != 2:
      raise TypeError("clip must be a pair of FP32 bit patterns")
    lo, hi = spec.clip
    _require_word(lo, "clip[0]")
    _require_word(hi, "clip[1]")
    if scalar.is_nan(lo) or scalar.is_nan(hi):
      raise ValueError("clip bounds must not be NaN")
    if scalar.compare_f32(lo, hi) > 0:
      raise ValueError("clip bounds are unordered")
  if spec.output_dtype == "i8":
    if spec.inv_scale is None:
      raise ValueError("i8 output requires inv_scale")
    _require_word(spec.inv_scale, "inv_scale")
    if scalar.is_nan(spec.inv_scale) or scalar.is_inf(spec.inv_scale):
      raise ValueError("inv_scale must be finite")
    if scalar.compare_f32(spec.inv_scale, 0) <= 0:
      raise ValueError("inv_scale must be > +0")
    _require_index(spec.zero_point, "zero_point", _I8_MIN, _I8_MAX)
  else:
    if spec.inv_scale is not None:
      raise ValueError("inv_scale is only valid for i8 output")
    if spec.zero_point != 0:
      raise ValueError("zero_point is only valid for i8 output")


def _relu(bits: int) -> tuple[int, int]:
  if scalar.is_nan(bits):
    flags = _NV_FLAG if scalar.is_snan(bits) else 0
    return scalar.CANONICAL_NAN, flags
  if bits >> 31:
    return 0, 0
  return bits, 0


def _clip(bits: int, bounds) -> tuple[int, int]:
  lo, hi = bounds
  flags = _NV_FLAG if scalar.is_snan(bits) else 0
  if scalar.is_nan(bits):
    return scalar.CANONICAL_NAN, flags
  if scalar.compare_f32(bits, lo) < 0:
    return lo, flags
  if scalar.compare_f32(bits, hi) > 0:
    return hi, flags
  return bits, flags


def _convert_output(value: int, spec: Epilogue) -> tuple[int, dict]:
  if spec.output_dtype == "f32":
    return value, {"stage": "output", "dtype": "f32", "before": value, "after": value, "flags": 0}
  if spec.output_dtype == "bf16":
    result = scalar.f32_to_bf16_rne(value)
    return result.bits, {
      "stage": "output",
      "dtype": "bf16",
      "before": value,
      "after": result.bits,
      "flags": result.flags,
    }
  assert spec.inv_scale is not None  # validate_epilogue checked i8 parameters before any output.
  result = scalar.mul_f32_rne(value, spec.inv_scale)
  scaled = result.bits
  record = {
    "stage": "output",
    "dtype": "i8",
    "before": value,
    "scaled": scaled,
    "flags": result.flags,
    "special": None,
    "rounded": None,
    "biased": None,
  }
  if scalar.is_nan(scaled):
    word = spec.zero_point
    record["special"] = "nan"
  elif scalar.is_inf(scaled):
    positive = scaled == _POS_INF
    word = _I8_MAX if positive else _I8_MIN
    record["special"] = "inf+" if positive else "inf-"
  else:
    rounded = scalar.round_to_int_rne(scaled)
    biased = rounded + spec.zero_point
    record["rounded"] = rounded
    record["biased"] = biased
    word = scalar.saturate_i8(biased)
  record["after"] = word & 0xFF
  return word & 0xFF, record


def apply_epilogue(am: Sequence[int], m: int, n: int, spec: Epilogue) -> EpilogueResult:
  validate_epilogue(spec, n)
  _require_index(m, "m", 0)
  if isinstance(am, (str, bytes)) or not isinstance(am, (list, tuple)):
    raise TypeError("am must be a sequence of FP32 bit patterns")
  if len(am) != m * n:
    raise ValueError(f"am has {len(am)} words but m*n={m * n}")
  _require_words(am, "am")
  output_bits = []
  output_bytes = bytearray()
  trace = []
  width = output_dtype_width(spec.output_dtype)
  for i in range(m):
    for j in range(n):
      index = i * n + j
      value = am[index]
      stages = [{"stage": "input", "before": value, "after": value, "flags": 0}]
      if spec.bias is not None:
        result = scalar.add_f32_rne(value, spec.bias[j])
        stages.append({"stage": "bias", "before": value, "after": result.bits, "flags": result.flags})
        value = result.bits
      if spec.scale is not None:
        result = scalar.mul_f32_rne(value, spec.scale)
        stages.append({"stage": "scale", "before": value, "after": result.bits, "flags": result.flags})
        value = result.bits
      if spec.relu:
        after, flags = _relu(value)
        stages.append({"stage": "relu", "before": value, "after": after, "flags": flags})
        value = after
      if spec.clip is not None:
        after, flags = _clip(value, spec.clip)
        stages.append({"stage": "clip", "before": value, "after": after, "flags": flags})
        value = after
      out_word, out_stage = _convert_output(value, spec)
      stages.append(out_stage)
      output_bits.append(out_word)
      output_bytes += out_word.to_bytes(width, "little")
      trace.append({"index": index, "row": i, "col": j, "input": am[index], "stages": stages})
  return EpilogueResult(bytes(output_bytes), tuple(output_bits), trace)
