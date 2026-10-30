"""Exact IEEE arithmetic for the independent reference, with one explicit RNE.

Finite operations use fractions.Fraction and integer quotient/remainder only.
No Python float, production numerical code, SoftFloat or HardFloat is used to
produce these answers. Exception bits are diagnostic IEEE flags, not a BOA ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

INVALID = 0x10
DIVIDE_BY_ZERO = 0x08
OVERFLOW = 0x04
UNDERFLOW = 0x02
INEXACT = 0x01
CANONICAL_NAN = 0x7FC00000
CANONICAL_BF16_NAN = 0x7FC0


@dataclass(frozen=True)
class FPResult:
  bits: int
  flags: int = 0


def _word(value: int, width: int = 32) -> int:
  if type(value) is not int or not 0 <= value < (1 << width):
    raise ValueError(f"expected an unsigned {width}-bit integer bit pattern")
  return value


def _nan(bits: int) -> bool:
  return bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF != 0


def _snan(bits: int) -> bool:
  return _nan(bits) and bits & 0x00400000 == 0


def _inf(bits: int) -> bool:
  return bits & 0x7FFFFFFF == 0x7F800000


def _zero(bits: int) -> bool:
  return bits & 0x7FFFFFFF == 0


def is_nan(bits: int) -> bool:
  return _nan(_word(bits))


def is_snan(bits: int) -> bool:
  return _snan(_word(bits))


def is_inf(bits: int) -> bool:
  return _inf(_word(bits))


def is_zero(bits: int) -> bool:
  return _zero(_word(bits))


def finite_fraction(bits: int) -> Fraction:
  """Decode a finite binary32 exactly; signed zero is handled by the caller."""
  bits = _word(bits)
  exponent = (bits >> 23) & 255
  fraction = bits & 0x007FFFFF
  if exponent == 255:
    raise ValueError("non-finite FP32 has no rational value")
  significand = fraction | (0x00800000 if exponent else 0)
  power = exponent - 150 if exponent else -149
  if bits >> 31:
    significand = -significand
  return Fraction(significand << power) if power >= 0 else Fraction(significand, 1 << -power)


def _rne_quotient(numerator: int, denominator: int) -> tuple[int, bool]:
  quotient, remainder = divmod(numerator, denominator)
  twice = remainder << 1
  if twice > denominator or (twice == denominator and quotient & 1):
    quotient += 1
  return quotient, remainder != 0


def _round_fraction(value: Fraction, fraction_bits: int, zero_sign: int = 0) -> FPResult:
  """Round once to an 8-exponent-bit binary format (FP32 or BF16)."""
  if type(zero_sign) is not int or zero_sign not in (0, 1):
    raise ValueError("zero_sign must be 0 or 1")
  sign = int(value < 0) if value else zero_sign
  sign_bits = sign << (8 + fraction_bits)
  if not value:
    return FPResult(sign_bits)
  numerator = abs(value.numerator)
  denominator = value.denominator
  # floor(log2(numerator/denominator)) without a floating-point logarithm.
  exponent = numerator.bit_length() - denominator.bit_length()
  if exponent >= 0:
    if numerator < denominator << exponent:
      exponent -= 1
  elif numerator << -exponent < denominator:
    exponent -= 1
  quantum = max(exponent, -126) - fraction_bits
  scaled_numerator = numerator << -quantum if quantum < 0 else numerator
  scaled_denominator = denominator << quantum if quantum >= 0 else denominator
  rounded, inexact = _rne_quotient(scaled_numerator, scaled_denominator)
  flags = INEXACT if inexact else 0
  implicit_one = 1 << fraction_bits
  if exponent >= -126:
    if rounded >= implicit_one << 1:
      rounded >>= 1
      exponent += 1
    if exponent > 127:
      return FPResult(sign_bits | (255 << fraction_bits), OVERFLOW | INEXACT)
    fraction = rounded - implicit_one
    return FPResult(sign_bits | ((exponent + 127) << fraction_bits) | fraction, flags)
  # Tininess-after uses precision rounding with an unbounded exponent range,
  # not the final encoding after rounding onto the subnormal grid.
  tiny = exponent < -127
  if exponent == -127:
    unbounded, _ = _rne_quotient(numerator << (127 + fraction_bits), denominator)
    tiny = unbounded < implicit_one << 1
  if tiny and inexact:
    flags |= UNDERFLOW
  return FPResult(sign_bits | rounded, flags)


def from_fraction(value: Fraction | int, zero_sign: int = 0) -> FPResult:
  if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
    raise ValueError("exact input must be Fraction or int, never Python float")
  return _round_fraction(Fraction(value), 23, zero_sign)


def fma_exact(a: int, b: int, c: int) -> Fraction:
  """Unrounded finite a*b+c. Special-value classification belongs to fma_f32_rne."""
  return finite_fraction(a) * finite_fraction(b) + finite_fraction(c)


def fma_f32_rne(a: int, b: int, c: int) -> FPResult:
  a, b, c = _word(a), _word(b), _word(c)
  invalid_product = (_inf(a) and _zero(b)) or (_inf(b) and _zero(a))
  invalid = invalid_product or _snan(a) or _snan(b) or _snan(c)
  if _nan(a) or _nan(b) or _nan(c):
    return FPResult(CANONICAL_NAN, INVALID if invalid else 0)
  if invalid_product:
    return FPResult(CANONICAL_NAN, INVALID)
  product_sign = (a ^ b) >> 31
  if _inf(a) or _inf(b):
    if _inf(c) and product_sign != c >> 31:
      return FPResult(CANONICAL_NAN, INVALID)
    return FPResult((product_sign << 31) | 0x7F800000)
  if _inf(c):
    return FPResult(c)
  exact = fma_exact(a, b, c)
  same_signed_zeros = (_zero(a) or _zero(b)) and _zero(c) and product_sign == c >> 31
  zero_sign = product_sign if same_signed_zeros else 0
  return _round_fraction(exact, 23, zero_sign)


def add_f32_rne(a: int, b: int) -> FPResult:
  a, b = _word(a), _word(b)
  if _nan(a) or _nan(b):
    return FPResult(CANONICAL_NAN, INVALID if _snan(a) or _snan(b) else 0)
  if _inf(a) and _inf(b) and (a ^ b) >> 31:
    return FPResult(CANONICAL_NAN, INVALID)
  if _inf(a):
    return FPResult(a)
  if _inf(b):
    return FPResult(b)
  zero_sign = a >> 31 if _zero(a) and _zero(b) and a == b else 0
  return _round_fraction(finite_fraction(a) + finite_fraction(b), 23, zero_sign)


def mul_f32_rne(a: int, b: int) -> FPResult:
  a, b = _word(a), _word(b)
  if _nan(a) or _nan(b):
    return FPResult(CANONICAL_NAN, INVALID if _snan(a) or _snan(b) else 0)
  if (_inf(a) and _zero(b)) or (_inf(b) and _zero(a)):
    return FPResult(CANONICAL_NAN, INVALID)
  sign = (a ^ b) >> 31
  if _inf(a) or _inf(b):
    return FPResult((sign << 31) | 0x7F800000)
  return _round_fraction(finite_fraction(a) * finite_fraction(b), 23, sign)


def bf16_to_f32(bits: int) -> int:
  return _word(bits, 16) << 16


def f32_to_bf16_rne(bits: int) -> FPResult:
  bits = _word(bits)
  if _nan(bits):
    return FPResult(CANONICAL_BF16_NAN, INVALID if _snan(bits) else 0)
  if _inf(bits):
    return FPResult(bits >> 16)
  return _round_fraction(finite_fraction(bits), 7, bits >> 31)


def round_to_int_rne(bits: int) -> int:
  value = finite_fraction(bits)
  rounded, _ = _rne_quotient(abs(value.numerator), value.denominator)
  return -rounded if value < 0 else rounded


def f32_to_i32_rne(bits: int) -> FPResult:
  """RISC-V signed-i32 conversion for primitive crosscheck, not INT8 policy."""
  bits = _word(bits)
  if _nan(bits):
    return FPResult(0x7FFFFFFF, INVALID)
  if _inf(bits):
    return FPResult(0x80000000 if bits >> 31 else 0x7FFFFFFF, INVALID)
  integer = round_to_int_rne(bits)
  if integer < -(1 << 31):
    return FPResult(0x80000000, INVALID)
  if integer >= 1 << 31:
    return FPResult(0x7FFFFFFF, INVALID)
  return FPResult(integer & 0xFFFFFFFF, INEXACT if Fraction(integer) != finite_fraction(bits) else 0)


def saturate_i8(value: int) -> int:
  if type(value) is not int:
    raise ValueError("INT8 saturation requires an integer")
  return max(-128, min(127, value))


def compare_f32(a: int, b: int) -> int:
  a, b = _word(a), _word(b)
  if _nan(a) or _nan(b):
    raise ValueError("NaN is unordered")
  if a == b or (_zero(a) and _zero(b)):
    return 0
  if (a ^ b) >> 31:
    return -1 if a >> 31 else 1
  magnitude_order = (a > b) - (a < b)
  return -magnitude_order if a >> 31 else magnitude_order
