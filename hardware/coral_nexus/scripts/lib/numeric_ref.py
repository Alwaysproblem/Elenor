"""Exact bit-level reference of the pinned Zvt BF16 matrix-multiply numeric contract (C011).

This is the CONTRACT written as executable integer arithmetic. It is a DUT
contract check, NOT the independent oracle (C047 owns that). It never calls
float(); every value is a Python int or a `fractions.Fraction`, so a result is
a bit pattern, never "close to" one.

Each function names the rule id it implements (see protocol/w3_numeric_profile_v1.yaml).
The generator parameters the model uses are NOT hard-coded numbers: `derive_parameters`
re-computes them from the format table and the format-enable mask, and the
profile validator compares them to what the pinned RTL source says.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

# fpnew_pkg::FP_ENCODINGS (pinned cvfpu): (exp_bits, man_bits), index = fp_format_e.
FP_FORMATS: dict[str, tuple[int, int]] = {
  "FP32": (8, 23),
  "FP64": (11, 52),
  "FP16": (5, 10),
  "FP8": (5, 2),
  "FP16ALT": (8, 7),
}
FORMAT_ORDER = ("FP32", "FP64", "FP16", "FP8", "FP16ALT")  # enum value = index

# fpnew_pkg::roundmode_e
RNE, RTZ, RDN, RUP, RMM, ROD = 0b000, 0b001, 0b010, 0b011, 0b100, 0b101
ROUND_MODE_NAMES = {RNE: "RNE", RTZ: "RTZ", RDN: "RDN", RUP: "RUP", RMM: "RMM", ROD: "ROD"}

QNAN_FP32 = 0x7FC00000
INF_FP32 = 0x7F800000


class NumericError(Exception):
  """Coded error so tests pin the failure kind, not a message."""

  def __init__(self, code: str, detail: str) -> None:
    super().__init__(f"{code}: {detail}")
    self.code = code


@dataclass(frozen=True)
class Status:
  nv: int = 0
  of: int = 0
  uf: int = 0
  nx: int = 0


def log2_ceil(n: int) -> int:
  if n < 1:
    raise NumericError("LOG2_DOMAIN", f"log2Ceil needs n >= 1, got {n}")
  return (n - 1).bit_length()


@dataclass(frozen=True)
class Params:
  """Parameters the pinned generator derives for FP_FMT_CONFIG (rule PAR-*)."""

  enabled: tuple[str, ...]
  super_exp_bits: int
  super_man_bits: int
  width: int
  vec_len: int
  guard_bits: int
  overflow_bits: int
  prod_exp_bits: int
  prod_sig_bits: int
  lanes: dict[str, int]


def derive_parameters(enabled: tuple[str, ...]) -> Params:
  """Re-derive the generator parameters from the format table.

  Mirrors `zvt_pe_mulbulk_fp_lane.sv` localparams: VEC_LEN is the maximum of WIDTH / format_width
  over ALL five formats (enabled or not); GUARD_BITS = OVERFLOW_BITS = clog2(VEC_LEN).
  """
  for name in enabled:
    if name not in FP_FORMATS:
      raise NumericError("FORMAT_UNKNOWN", name)
  if not enabled:
    raise NumericError("FORMAT_NONE", "no format enabled")
  super_e = max(FP_FORMATS[n][0] for n in enabled)
  super_m = max(FP_FORMATS[n][1] for n in enabled)
  width = max(sum((*FP_FORMATS[n], 1)) for n in enabled)
  vec_len = max(width // (e + m + 1) for e, m in FP_FORMATS.values())
  guard = log2_ceil(vec_len)
  return Params(
    enabled=tuple(enabled),
    super_exp_bits=super_e,
    super_man_bits=super_m,
    width=width,
    vec_len=vec_len,
    guard_bits=guard,
    overflow_bits=guard,
    prod_exp_bits=super_e + 2,
    prod_sig_bits=super_m + guard + 1,
    lanes={n: width // sum((*FP_FORMATS[n], 1)) for n in enabled},
  )


# The block instantiates the multiplier with FP_FMT_CONFIG = 5'b10001, a vector indexed [0:4];
# index 0 is the leftmost bit, so the set bits are FP32 (index 0) and FP16ALT (index 4).
BLOCK_FP_FMT_CONFIG = ("FP32", "FP16ALT")


# ---------------------------------------------------------------------------
# BF16 / FP32 bit helpers (hand-written, no float())
# ---------------------------------------------------------------------------
def bf16_to_fp32_bits(bits: int) -> int:
  if not 0 <= bits <= 0xFFFF:
    raise NumericError("BF16_RANGE", f"{bits:#x}")
  return bits << 16


def fp32_to_bf16_bits_exact(bits: int) -> int:
  """Narrow an FP32 pattern whose low 16 bits are zero (exactly representable)."""
  if bits & 0xFFFF:
    raise NumericError("NOT_EXACT_BF16", f"{bits:#010x}")
  return bits >> 16


def decode(bits: int, exp_bits: int, man_bits: int) -> tuple[str, int, Fraction | None]:
  """Classify a pattern. Returns (kind, sign, exact value or None for inf/nan)."""
  total = 1 + exp_bits + man_bits
  if not 0 <= bits < 1 << total:
    raise NumericError("PATTERN_RANGE", f"{bits:#x} does not fit {total} bits")
  sign = bits >> (exp_bits + man_bits)
  exp = (bits >> man_bits) & ((1 << exp_bits) - 1)
  man = bits & ((1 << man_bits) - 1)
  if exp == (1 << exp_bits) - 1:
    if man == 0:
      return "inf", sign, None
    return ("snan" if not (man >> (man_bits - 1)) & 1 else "qnan"), sign, None
  bias = (1 << (exp_bits - 1)) - 1
  if exp == 0:
    value = Fraction(man, 1 << man_bits) * Fraction(2) ** (1 - bias)
    return ("zero" if man == 0 else "subnormal"), sign, -value if sign else value
  value = (1 + Fraction(man, 1 << man_bits)) * Fraction(2) ** (exp - bias)
  return "normal", sign, -value if sign else value


# ---------------------------------------------------------------------------
# Rounding (fp_rounding.sv), on an already-aligned significand
# ---------------------------------------------------------------------------
def round_decision(mode: int, guard: int, round_bit: int, sticky: int, sign: int) -> int:
  """The `rounding_decision` case. RNE/RTZ/RDN/RUP/RMM/ROD only; anything else is a coded error."""
  rs = round_bit | sticky
  if mode == RNE:
    return 0 if (round_bit, sticky) in ((0, 0), (0, 1)) else (guard if (round_bit, sticky) == (1, 0) else 1)
  if mode == RTZ:
    return 0
  if mode == RDN:
    return sign if rs else 0
  if mode == RUP:
    return (1 - sign) if rs else 0
  if mode == RMM:
    return round_bit
  if mode == ROD:
    return (1 - guard) & rs
  raise NumericError("ROUND_MODE_INVALID", f"{mode:#05b} has no rounding decision")


def overflow_to_max_norm(mode: int, sign: int) -> int:
  """The `of_saturate_decision` case: does an overflow saturate to max-normal instead of Inf?"""
  if mode in (RNE, RMM):
    return 0
  if mode in (RTZ, ROD):
    return 1
  if mode == RDN:
    return 1 - sign
  if mode == RUP:
    return sign
  raise NumericError("ROUND_MODE_INVALID", f"{mode:#05b} has no overflow rule")


def round_exact(
  value: Fraction, man_bits: int, exp_bits: int, mode: int, *, widen_overflow_inf: bool
) -> tuple[int, Status]:
  """Round an exact rational to an (exp_bits, man_bits) pattern with a given mode.

  `widen_overflow_inf` is the multiplier-stage rule (fp_rounding + zvt_pe_mulbulk_fp_lane overflow mux):
  when source and destination formats differ an overflow is Inf even if the mode would saturate. The
  accumulate stage passes False, so ROD overflow saturates to max-normal there.
  """
  bias = (1 << (exp_bits - 1)) - 1
  emax = (1 << exp_bits) - 2  # largest finite biased exponent
  sign = 1 if value < 0 else 0
  mag = -value if sign else value
  if mag == 0:
    return sign << (exp_bits + man_bits), Status()
  e = mag.numerator.bit_length() - mag.denominator.bit_length()
  while Fraction(2) ** e > mag:
    e -= 1
  while Fraction(2) ** (e + 1) <= mag:
    e += 1
  biased = e + bias
  if biased < 1:  # subnormal: fixed grid 2**(1-bias-man_bits)
    scale = Fraction(2) ** (1 - bias - man_bits)
    biased = 0
  else:
    scale = Fraction(2) ** (e - man_bits)
  units = mag / scale
  floor = units.numerator // units.denominator
  rem = units - floor
  guard = floor & 1
  round_bit = 1 if rem >= Fraction(1, 2) else 0
  sticky = 1 if rem - Fraction(round_bit, 2) != 0 else 0
  up = round_decision(mode, guard, round_bit, sticky, sign)
  total = floor + up
  if biased == 0:
    exp_field, mant_field = 0, total
    if total >> man_bits:  # rounded up into the smallest normal
      exp_field, mant_field = 1, total & ((1 << man_bits) - 1)
  else:
    if total >> (man_bits + 1):  # mantissa carry
      total >>= 1
      biased += 1
    exp_field, mant_field = biased, total & ((1 << man_bits) - 1)
  if exp_field > emax:
    if overflow_to_max_norm(mode, sign) and not widen_overflow_inf:
      pattern = (sign << (exp_bits + man_bits)) | (emax << man_bits) | ((1 << man_bits) - 1)
    else:
      pattern = (sign << (exp_bits + man_bits)) | (((1 << exp_bits) - 1) << man_bits)
    return pattern, Status(of=1, nx=1)
  pattern = (sign << (exp_bits + man_bits)) | (exp_field << man_bits) | mant_field
  inexact = 1 if rem != 0 else 0
  return pattern, Status(nx=inexact, uf=1 if (exp_field == 0 and inexact) else 0)


# ---------------------------------------------------------------------------
# multiplier lane (fp_mulfront.sv) and the batched multiply tree (zvt_pe_mulbulk_fp_lane.sv)
# ---------------------------------------------------------------------------
BF16 = FP_FORMATS["FP16ALT"]
SIG_BITS_ALIGNED = 26  # PROD_SIG_BITS for the block's FP_FMT_CONFIG (derive_parameters)


@dataclass(frozen=True)
class Lane:
  """One K lane after masking, as fp_mulfront sees it."""

  zero: bool  # a_zero | b_zero -> contributes nothing and takes no part in Em
  sign: int
  product: Fraction  # exact signed product (0 when zero)
  norm_exp: int  # floor(log2|a|) + floor(log2|b|) + 1 (weight exponent of product bit 15 - lzc sum)
  nan: bool
  inf: bool
  invalid: bool


def _floor_log2(x: Fraction) -> int:
  x = -x if x < 0 else x
  e = x.numerator.bit_length() - x.denominator.bit_length()
  while Fraction(2) ** e > x:
    e -= 1
  while Fraction(2) ** (e + 1) <= x:
    e += 1
  return e


def mul_lane(a_bits: int, b_bits: int, enabled: bool) -> Lane:
  """Mask FIRST, then classify (rule MSK-01): a disabled lane is 0 x 0, so it can never raise NV."""
  if not enabled:
    a_bits = b_bits = 0
  ka, sa, va = decode(a_bits, *BF16)
  kb, sb, vb = decode(b_bits, *BF16)
  sign = sa ^ sb
  inf_zero = (ka == "inf" and kb == "zero") or (kb == "inf" and ka == "zero")
  if inf_zero:
    return Lane(True, sign, Fraction(0), 0, True, False, True)
  if ka in ("snan", "qnan") or kb in ("snan", "qnan"):
    return Lane(True, sign, Fraction(0), 0, True, False, ka == "snan" or kb == "snan")
  if ka == "inf" or kb == "inf":
    return Lane(True, sign, Fraction(0), 0, False, True, False)
  assert va is not None and vb is not None
  if va == 0 or vb == 0:
    return Lane(True, sign, Fraction(0), 0, False, False, False)
  return Lane(False, sign, va * vb, _floor_log2(va) + _floor_log2(vb) + 1, False, False, False)


def mulbulk(lanes: list[Lane]) -> tuple[int, Status]:
  """Sum the K lanes with the finite-width tree; one ROD rounding to FP32 (multiplier stage).

  Rules: NaN beats Inf; Inf-Inf and NaN-lane/inf-zero raise NV; otherwise lanes are aligned to
  Em = max(norm_exp) on a 26-bit grid, bits below the grid are JAMMED into the LSB, the aligned
  signed integers are added exactly, and the result is rounded to FP32 with ROD where an overflow
  is Inf (widening).
  """
  nan = any(lane.nan for lane in lanes)
  infs = [lane for lane in lanes if lane.inf]
  inf_minus_inf = any(lane.sign for lane in infs) and any(not lane.sign for lane in infs)
  if nan or inf_minus_inf:
    return QNAN_FP32, Status(nv=1 if (inf_minus_inf or any(lane.invalid for lane in lanes)) else 0)
  if infs:
    return (infs[0].sign << 31) | INF_FP32, Status()
  live = [lane for lane in lanes if not lane.zero]
  if not live:
    return 0, Status()  # all-zero products: sum is +0 whatever the operand signs
  em = max(lane.norm_exp for lane in live)
  unit = Fraction(2) ** (em - (SIG_BITS_ALIGNED - 1))
  total = 0
  for lane in live:
    mag = lane.product if lane.product > 0 else -lane.product
    q = mag / unit
    fl = q.numerator // q.denominator
    if q != fl:
      fl |= 1  # jam: lost bits OR-ed into the LSB
    total += -fl if lane.sign else fl
  # total == 0 (exact cancellation) rounds to +0 below: a zero rational has no sign, matching fmt_zero
  return round_exact(Fraction(total) * unit, 23, 8, ROD, widen_overflow_inf=True)


# ---------------------------------------------------------------------------
# accumulate stage (zvt_pe_adder_fp_lane.sv + fp_addfront.sv)
# ---------------------------------------------------------------------------
def _fp32_parts(bits: int) -> tuple[str, int, int, int, int]:
  kind, sign, _ = decode(bits, 8, 23)
  exp = (bits >> 23) & 0xFF
  man = bits & 0x7FFFFF
  return kind, sign, exp, man, ((1 << 23) | man if exp else man)


def accumulate(prod_bits: int, old_bits: int) -> tuple[int, Status]:
  """new = ROD(prod + old) with the adder's 26-bit-plus-round-plus-sticky alignment (rule ACC-*)."""
  ka, sa, ea, _ma, siga = _fp32_parts(prod_bits)  # operand 0 = multiplier result
  kb, sb, eb, _mb, sigb = _fp32_parts(old_bits)  # operand 1 = mt
  if ka == "inf" and kb == "inf" and (sa ^ sb):
    return QNAN_FP32, Status(nv=1)
  if ka in ("snan", "qnan") or kb in ("snan", "qnan"):
    return QNAN_FP32, Status(nv=1 if "snan" in (ka, kb) else 0)
  if ka == "inf":
    return (sa << 31) | INF_FP32, Status()
  if kb == "inf":
    return (sb << 31) | INF_FP32, Status()
  ea1, eb1 = max(ea, 1), max(eb, 1)  # subnormals align as exponent 1
  top = max(ea, eb)
  fix = top if top > 1 else 1  # subnormal_fix_align_exp
  unit27 = Fraction(2) ** (
    fix - 127 - 27
  )  # weight of one unit of the 28-bit {sig26, round, sticky} integer

  def to_n(sig: int, e1: int) -> int:
    value = Fraction(sig) * Fraction(2) ** (e1 - 127 - 23)
    q = value / (2 * unit27)  # floor to the round-bit grid
    fl = q.numerator // q.denominator
    return 2 * fl + (1 if q != fl else 0)

  na, nb = to_n(siga, ea1), to_n(sigb, eb1)
  signed = (-na if sa else na) + (-nb if sb else nb)
  if signed == 0:
    eff_sub = sa ^ sb
    return (0 if eff_sub else sa << 31), Status()  # RDN is the only mode that gives -0 on cancellation
  return round_exact(Fraction(signed) * unit27, 23, 8, ROD, widen_overflow_inf=False)


def vtfmm_cell(old_bits: int, pairs: list[tuple[int, int]], tk: int) -> tuple[int, Status, Status]:
  """One output element: old += sum over the first `tk` (a, b) pairs.

  Returns (bits, mul_status, add_status).

  VME only forwards OF and NV (zvt_pe_block.sv fpexp), so the flags the contract exposes are
  mul.of | add.of and mul.nv | add.nv.
  """
  if tk not in (1, 2):
    raise NumericError("TK_INVALID", f"tk={tk}")
  lanes = [mul_lane(a, b, k < tk) for k, (a, b) in enumerate(pairs[:2])]
  prod, mstat = mulbulk(lanes)
  new, astat = accumulate(prod, old_bits)
  return new, mstat, astat


def vtfmm_tile(
  a_rows: list[list[int]],
  b_rows: list[list[int]],
  c_old: list[int],
  tm: int,
  tn: int,
  tk: int,
  dim: int = 16,
) -> list[int]:
  """16x16 FP32 tile: cells inside tm x tn are updated, the rest are undisturbed."""
  out = list(c_old)
  for m in range(tm):
    for n in range(tn):
      pairs = [(a_rows[k][m], b_rows[k][n]) for k in range(2)]
      out[m * dim + n] = vtfmm_cell(c_old[m * dim + n], pairs, tk)[0]
  return out


# ---------------------------------------------------------------------------
# frm admission (rule FRM-01) and final narrowing (rule NAR-01)
# ---------------------------------------------------------------------------
ADMITTED_FRM = (RNE, RTZ, RDN, RUP, RMM)


def check_frm(frm: int) -> int:
  """The profile admits the five RISC-V rounding modes. 5 (ROD), 6 and 7 (DYN) are rejected."""
  if frm not in range(8):
    raise NumericError("FRM_RANGE", f"frm={frm} is not a 3-bit value")
  if frm == 7:
    raise NumericError("FRM_DYN", "frm=7 (DYN) reaches an undefined rounding path")
  if frm == ROD:
    raise NumericError("FRM_ROD", "frm=5 is ROD: not a software rounding mode for this profile")
  if frm not in ADMITTED_FRM:
    raise NumericError("FRM_UNASSIGNED", f"frm={frm} has no enumerator")
  return frm


def narrow_fp32_to_bf16(bits: int, frm: int = RNE) -> tuple[int, Status]:
  """FP32 -> BF16 (the RVV narrowing boundary). Result bits and NV/OF/NX only.

  A NaN input gives the canonical BF16 quiet NaN 0x7FC0; a signalling NaN also raises NV.
  This rule is decided by the card (see profile NAR-01); the RTL path is not read back here.
  """
  check_frm(frm)
  kind, sign, value = decode(bits, 8, 23)
  if kind in ("snan", "qnan"):
    return 0x7FC0, Status(nv=1 if kind == "snan" else 0)
  if kind == "inf":
    return (sign << 15) | 0x7F80, Status()
  assert value is not None
  pattern, status = round_exact(value, 7, 8, frm, widen_overflow_inf=False)
  return pattern, Status(of=status.of, nx=status.nx)
