"""Narrow bitwise scalar / CSR / FP32 flags & rounding supplement (C021 / W7-02b).

Scope per C021:
- Does NOT re-implement all of RV32IMF (pinned Spike is the primary scalar ISA reference).
- Implements ONLY the narrow delta / rounding / CSR semantics that Spike either merges into
  running CSRs or does not govern on the Coral Nexus boundary:
  1. IEEE-754 binary32 exact integer/rational arithmetic and rounding under RNE/RTZ/RDN/RUP/RMM
     with per-instruction `fflags` delta (`NV=0x10, DZ=0x08, OF=0x04, UF=0x02, NX=0x01`).
  2. Retire-ordered `flags_delta` (`fflags[4:0] | (vxsat << 5)`) and CSR read/write WARL rules
     matching the pinned CoralNPU `Csr.scala` and C010 `ctx_ctrl.yaml` / `retire_record.yaml`.
- Invariants:
  - Pure Python integer bit operations and `fractions.Fraction`; NEVER `float`, `struct` float
    formats, `numpy`, or `scipy`.
  - Never imports or reads DUT RTL Verilog files to derive numeric results.
  - BF16/ROD/VME matrix tile bitwise oracle belongs to C047 (CN-G3) and is explicitly not claimed
    here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

MASK32 = 0xFFFFFFFF

# RISC-V fflags bit positions (CSR 0x001, bits [4:0]): NV, DZ, OF, UF, NX.
FLAG_NX = 0b00001
FLAG_UF = 0b00010
FLAG_OF = 0b00100
FLAG_DZ = 0b01000
FLAG_NV = 0b10000
FFLAGS_MASK = 0x1F
VXSAT_SHIFT = 5
FLAGS_DELTA_MASK = 0x3F

# RISC-V frm encodings (CSR 0x002, bits [2:0]).
FRM_RNE = 0b000
FRM_RTZ = 0b001
FRM_RDN = 0b010
FRM_RUP = 0b011
FRM_RMM = 0b100
FRM_DYN = 0b111
LEGAL_SCALAR_FRM = frozenset({FRM_RNE, FRM_RTZ, FRM_RDN, FRM_RUP, FRM_RMM})

FP32_SIGN_SHIFT = 31
FP32_EXP_BITS = 8
FP32_MAN_BITS = 23
FP32_BIAS = 127
FP32_EMAX_BIASED = 254
FP32_CANONICAL_QNAN = 0x7FC00000
FP32_POS_INF = 0x7F800000
FP32_NEG_INF = 0xFF800000


class BitwiseScalarError(Exception):
  """Coded failure for illegal rounding modes, CSR accesses, or bit-width violations."""

  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


@dataclass(frozen=True)
class Fp32Result:
  bits: int
  fflags_delta: int
  flags_delta: int


@dataclass(frozen=True)
class CsrStepExpectation:
  old_value: int
  new_value: int
  rd_value: int
  commit_timing: str  # "dispatch_beat" | "retire_commit" | "read_only" | "flags_delta"
  flags_delta: int = 0


def _check_u32(value: int, name: str = "u32") -> int:
  if isinstance(value, bool) or not isinstance(value, int) or not (0 <= value <= MASK32):
    raise BitwiseScalarError("U32_RANGE", f"{name}={value!r}")
  return value


def pack_flags_delta(fflags_delta: int, vxsat_delta: int = 0) -> int:
  if isinstance(fflags_delta, bool) or not (0 <= fflags_delta <= FFLAGS_MASK):
    raise BitwiseScalarError("FFLAGS_RANGE", repr(fflags_delta))
  if isinstance(vxsat_delta, bool) or vxsat_delta not in (0, 1):
    raise BitwiseScalarError("VXSAT_RANGE", repr(vxsat_delta))
  return fflags_delta | (vxsat_delta << VXSAT_SHIFT)


def resolve_rm(insn_rm: int, csr_frm: int) -> int:
  rm = csr_frm if insn_rm == FRM_DYN else insn_rm
  if rm not in LEGAL_SCALAR_FRM:
    raise BitwiseScalarError("ILLEGAL_FRM", f"insn_rm={insn_rm} csr_frm={csr_frm} -> rm={rm}")
  return rm


def decode_fp32(bits: int) -> tuple[str, int, Fraction | None]:
  """Decode binary32 bits into `(kind, sign, exact_rational)` without `float`."""
  _check_u32(bits, "fp32")
  sign = (bits >> FP32_SIGN_SHIFT) & 1
  exp = (bits >> FP32_MAN_BITS) & 0xFF
  man = bits & 0x7FFFFF
  if exp == 0xFF:
    if man == 0:
      return ("inf", sign, None)
    is_quiet = (man >> (FP32_MAN_BITS - 1)) & 1
    return ("qnan" if is_quiet else "snan", sign, None)
  if exp == 0:
    if man == 0:
      return ("zero", sign, Fraction(0))
    mag = Fraction(man, 1 << FP32_MAN_BITS) * (Fraction(1, 2) ** 126)
    return ("subnormal", sign, -mag if sign else mag)
  mag = (1 + Fraction(man, 1 << FP32_MAN_BITS)) * (
    Fraction(2, 1) ** (exp - FP32_BIAS)
    if exp >= FP32_BIAS
    else Fraction(1, 2) ** (FP32_BIAS - exp)
  )
  return ("normal", sign, -mag if sign else mag)


def _round_increment(rm: int, lsb: int, guard: int, sticky: int, sign: int) -> int:
  inexact = guard | sticky
  if not inexact:
    return 0
  if rm == FRM_RNE:
    return 1 if (guard == 1 and (sticky == 1 or lsb == 1)) else 0
  if rm == FRM_RTZ:
    return 0
  if rm == FRM_RDN:
    return 1 if sign == 1 else 0
  if rm == FRM_RUP:
    return 1 if sign == 0 else 0
  if rm == FRM_RMM:
    return 1 if guard == 1 else 0
  raise BitwiseScalarError("ILLEGAL_FRM", f"rm={rm}")


def _overflow_result(rm: int, sign: int) -> int:
  max_norm = (sign << 31) | (FP32_EMAX_BIASED << 23) | 0x7FFFFF
  inf = (sign << 31) | (0xFF << 23)
  if rm in (FRM_RNE, FRM_RMM):
    return inf
  if rm == FRM_RTZ:
    return max_norm
  if rm == FRM_RDN:
    return inf if sign == 1 else max_norm
  if rm == FRM_RUP:
    return max_norm if sign == 1 else inf
  raise BitwiseScalarError("ILLEGAL_FRM", f"rm={rm}")


def _floor_log2_pos(mag: Fraction) -> int:
  est = mag.numerator.bit_length() - mag.denominator.bit_length()
  while (Fraction(2, 1) ** est if est >= 0 else Fraction(1, 2) ** (-est)) > mag:
    est -= 1
  while (Fraction(2, 1) ** (est + 1) if est + 1 >= 0 else Fraction(1, 2) ** (-(est + 1))) <= mag:
    est += 1
  return est


def round_rational_to_fp32(value: Fraction, sign_if_zero: int, rm: int) -> Fp32Result:
  """IEEE-754 binary32 rounding with RISC-V tininess-after-rounding UF semantics."""
  if rm not in LEGAL_SCALAR_FRM:
    raise BitwiseScalarError("ILLEGAL_FRM", f"rm={rm}")
  if value == 0:
    bits = (sign_if_zero & 1) << 31
    return Fp32Result(bits=bits, fflags_delta=0, flags_delta=0)

  sign = 1 if value < 0 else 0
  mag = -value if sign else value
  unbiased_exp = _floor_log2_pos(mag)
  biased_exp = unbiased_exp + FP32_BIAS

  # Check tininess AFTER rounding per RISC-V IEEE 754-2008 choice (Spike softfloat_tininess_afterRounding):
  # test whether an unbounded-exponent normal rounding would still have exponent < -126.
  norm_scale_exp = unbiased_exp - FP32_MAN_BITS
  norm_scale = (
    Fraction(2, 1) ** norm_scale_exp if norm_scale_exp >= 0 else Fraction(1, 2) ** (-norm_scale_exp)
  )
  norm_units = mag / norm_scale
  norm_floor = norm_units.numerator // norm_units.denominator
  norm_rem = norm_units - norm_floor
  norm_guard = 1 if norm_rem >= Fraction(1, 2) else 0
  norm_sticky = 1 if (norm_rem - Fraction(norm_guard, 2)) != 0 else 0
  norm_inc = _round_increment(rm, norm_floor & 1, norm_guard, norm_sticky, sign)
  norm_sig = norm_floor + norm_inc
  norm_unbiased = unbiased_exp + (1 if norm_sig >= (1 << (FP32_MAN_BITS + 1)) else 0)
  tiny_after_rounding = norm_unbiased < (1 - FP32_BIAS)

  if biased_exp < 1:
    sub_scale = Fraction(1, 2) ** 149
    units = mag / sub_scale
    q = units.numerator // units.denominator
    rem = units - q
    guard = 1 if rem >= Fraction(1, 2) else 0
    sticky = 1 if (rem - Fraction(guard, 2)) != 0 else 0
    inc = _round_increment(rm, q & 1, guard, sticky, sign)
    sig = q + inc
    inexact = 1 if rem != 0 else 0
    uf = 1 if (inexact and tiny_after_rounding) else 0
    if sig >= (1 << FP32_MAN_BITS):
      exp_out = 1
      man_out = sig & 0x7FFFFF
    else:
      exp_out = 0
      man_out = sig
    flags = (FLAG_UF if uf else 0) | (FLAG_NX if inexact else 0)
    bits = (sign << 31) | (exp_out << 23) | man_out
    return Fp32Result(bits=bits, fflags_delta=flags, flags_delta=pack_flags_delta(flags))

  inexact = 1 if norm_rem != 0 else 0
  exp_out = biased_exp
  sig = norm_sig
  if sig >= (1 << (FP32_MAN_BITS + 1)):
    sig >>= 1
    exp_out += 1
  if exp_out > FP32_EMAX_BIASED:
    flags = FLAG_OF | FLAG_NX
    bits = _overflow_result(rm, sign)
    return Fp32Result(bits=bits, fflags_delta=flags, flags_delta=pack_flags_delta(flags))
  man_out = sig & 0x7FFFFF
  flags = FLAG_NX if inexact else 0
  bits = (sign << 31) | (exp_out << 23) | man_out
  return Fp32Result(bits=bits, fflags_delta=flags, flags_delta=pack_flags_delta(flags))


def fadd_s(a_bits: int, b_bits: int, *, insn_rm: int = FRM_DYN, csr_frm: int = FRM_RNE) -> Fp32Result:
  rm = resolve_rm(insn_rm, csr_frm)
  ka, sa, va = decode_fp32(a_bits)
  kb, sb, vb = decode_fp32(b_bits)
  if ka in ("snan", "qnan") or kb in ("snan", "qnan"):
    nv = FLAG_NV if (ka == "snan" or kb == "snan") else 0
    return Fp32Result(FP32_CANONICAL_QNAN, nv, pack_flags_delta(nv))
  if ka == "inf" and kb == "inf":
    if sa != sb:
      return Fp32Result(FP32_CANONICAL_QNAN, FLAG_NV, pack_flags_delta(FLAG_NV))
    return Fp32Result(FP32_NEG_INF if sa else FP32_POS_INF, 0, 0)
  if ka == "inf":
    return Fp32Result(FP32_NEG_INF if sa else FP32_POS_INF, 0, 0)
  if kb == "inf":
    return Fp32Result(FP32_NEG_INF if sb else FP32_POS_INF, 0, 0)
  assert va is not None and vb is not None
  total = va + vb
  if total == 0:
    zero_sign = (sa & sb) if sa == sb else (1 if rm == FRM_RDN else 0)
    return Fp32Result(zero_sign << 31, 0, 0)
  return round_rational_to_fp32(total, 0, rm)


def fsub_s(a_bits: int, b_bits: int, *, insn_rm: int = FRM_DYN, csr_frm: int = FRM_RNE) -> Fp32Result:
  _check_u32(b_bits, "b_bits")
  return fadd_s(a_bits, b_bits ^ (1 << 31), insn_rm=insn_rm, csr_frm=csr_frm)


def fmul_s(a_bits: int, b_bits: int, *, insn_rm: int = FRM_DYN, csr_frm: int = FRM_RNE) -> Fp32Result:
  rm = resolve_rm(insn_rm, csr_frm)
  ka, sa, va = decode_fp32(a_bits)
  kb, sb, vb = decode_fp32(b_bits)
  sign = sa ^ sb
  if ka in ("snan", "qnan") or kb in ("snan", "qnan"):
    nv = FLAG_NV if (ka == "snan" or kb == "snan") else 0
    return Fp32Result(FP32_CANONICAL_QNAN, nv, pack_flags_delta(nv))
  if (ka == "inf" and kb == "zero") or (kb == "inf" and ka == "zero"):
    return Fp32Result(FP32_CANONICAL_QNAN, FLAG_NV, pack_flags_delta(FLAG_NV))
  if ka == "inf" or kb == "inf":
    return Fp32Result(FP32_NEG_INF if sign else FP32_POS_INF, 0, 0)
  assert va is not None and vb is not None
  prod = va * vb
  return round_rational_to_fp32(prod, sign, rm)


def fdiv_s(a_bits: int, b_bits: int, *, insn_rm: int = FRM_DYN, csr_frm: int = FRM_RNE) -> Fp32Result:
  rm = resolve_rm(insn_rm, csr_frm)
  ka, sa, va = decode_fp32(a_bits)
  kb, sb, vb = decode_fp32(b_bits)
  sign = sa ^ sb
  if ka in ("snan", "qnan") or kb in ("snan", "qnan"):
    nv = FLAG_NV if (ka == "snan" or kb == "snan") else 0
    return Fp32Result(FP32_CANONICAL_QNAN, nv, pack_flags_delta(nv))
  if (ka == "inf" and kb == "inf") or (ka == "zero" and kb == "zero"):
    return Fp32Result(FP32_CANONICAL_QNAN, FLAG_NV, pack_flags_delta(FLAG_NV))
  if kb == "zero":
    return Fp32Result(FP32_NEG_INF if sign else FP32_POS_INF, FLAG_DZ, pack_flags_delta(FLAG_DZ))
  if ka == "inf":
    return Fp32Result(FP32_NEG_INF if sign else FP32_POS_INF, 0, 0)
  if kb == "inf" or ka == "zero":
    return Fp32Result(sign << 31, 0, 0)
  assert va is not None and vb is not None
  return round_rational_to_fp32(va / vb, sign, rm)


def fmadd_s(
  a_bits: int, b_bits: int, c_bits: int, *, insn_rm: int = FRM_DYN, csr_frm: int = FRM_RNE
) -> Fp32Result:
  rm = resolve_rm(insn_rm, csr_frm)
  ka, sa, va = decode_fp32(a_bits)
  kb, sb, vb = decode_fp32(b_bits)
  kc, sc, vc = decode_fp32(c_bits)
  psign = sa ^ sb
  inf_zero = (ka == "inf" and kb == "zero") or (kb == "inf" and ka == "zero")
  any_snan = "snan" in (ka, kb, kc)
  if any_snan or "qnan" in (ka, kb, kc) or inf_zero:
    nv = FLAG_NV if (any_snan or inf_zero) else 0
    return Fp32Result(FP32_CANONICAL_QNAN, nv, pack_flags_delta(nv))
  prod_inf = ka == "inf" or kb == "inf"
  if prod_inf and kc == "inf":
    if psign != sc:
      return Fp32Result(FP32_CANONICAL_QNAN, FLAG_NV, pack_flags_delta(FLAG_NV))
    return Fp32Result(FP32_NEG_INF if psign else FP32_POS_INF, 0, 0)
  if prod_inf:
    return Fp32Result(FP32_NEG_INF if psign else FP32_POS_INF, 0, 0)
  if kc == "inf":
    return Fp32Result(FP32_NEG_INF if sc else FP32_POS_INF, 0, 0)
  assert va is not None and vb is not None and vc is not None
  exact = va * vb + vc
  if exact == 0:
    if va * vb == 0 and vc == 0 and psign == sc:
      zsign = psign
    else:
      zsign = 1 if rm == FRM_RDN else 0
    return Fp32Result(zsign << 31, 0, 0)
  return round_rational_to_fp32(exact, psign, rm)


def fcvt_w_s(
  a_bits: int, *, signed_out: bool = True, insn_rm: int = FRM_DYN, csr_frm: int = FRM_RNE
) -> Fp32Result:
  """FP32 -> i32/u32 conversion with exact RISC-V saturation and NV/NX flags."""
  rm = resolve_rm(insn_rm, csr_frm)
  ka, sa, va = decode_fp32(a_bits)
  lo = -(1 << 31) if signed_out else 0
  hi = ((1 << 31) - 1) if signed_out else 0xFFFFFFFF
  pos_sat = 0x7FFFFFFF if signed_out else 0xFFFFFFFF
  neg_sat = 0x80000000 if signed_out else 0x00000000
  if ka in ("snan", "qnan"):
    return Fp32Result(pos_sat, FLAG_NV, pack_flags_delta(FLAG_NV))
  if ka == "inf":
    return Fp32Result(neg_sat if sa else pos_sat, FLAG_NV, pack_flags_delta(FLAG_NV))
  assert va is not None
  sign = 1 if va < 0 else 0
  mag = -va if sign else va
  q = mag.numerator // mag.denominator
  rem = mag - q
  guard = 1 if rem >= Fraction(1, 2) else 0
  sticky = 1 if (rem - Fraction(guard, 2)) != 0 else 0
  inc = _round_increment(rm, q & 1, guard, sticky, sign)
  rounded_mag = q + inc
  ival = -rounded_mag if sign else rounded_mag
  if ival < lo:
    return Fp32Result(neg_sat, FLAG_NV, pack_flags_delta(FLAG_NV))
  if ival > hi:
    return Fp32Result(pos_sat, FLAG_NV, pack_flags_delta(FLAG_NV))
  flags = FLAG_NX if rem != 0 else 0
  return Fp32Result(ival & MASK32, flags, pack_flags_delta(flags))


def fcvt_s_int(
  int_bits: int, *, signed_in: bool = True, insn_rm: int = FRM_DYN, csr_frm: int = FRM_RNE
) -> Fp32Result:
  rm = resolve_rm(insn_rm, csr_frm)
  _check_u32(int_bits, "int_bits")
  if signed_in and (int_bits & 0x80000000):
    val = int_bits - (1 << 32)
  else:
    val = int_bits
  return round_rational_to_fp32(Fraction(val), 0, rm)


def merge_retire_flags(arch_fflags: int, arch_vxsat: int, flags_delta: int) -> tuple[int, int]:
  """Ordered retire-time flag accumulation (`ctx_ctrl.yaml` `csr_commit.flags_delta`)."""
  if not (0 <= arch_fflags <= FFLAGS_MASK):
    raise BitwiseScalarError("FFLAGS_RANGE", repr(arch_fflags))
  if arch_vxsat not in (0, 1):
    raise BitwiseScalarError("VXSAT_RANGE", repr(arch_vxsat))
  if not (0 <= flags_delta <= FLAGS_DELTA_MASK):
    raise BitwiseScalarError("FLAGS_DELTA_RANGE", repr(flags_delta))
  new_fflags = arch_fflags | (flags_delta & FFLAGS_MASK)
  new_vxsat = arch_vxsat | ((flags_delta >> VXSAT_SHIFT) & 1)
  return new_fflags, new_vxsat


# Pinned CoralNPU CSR WARL expectations (Csr.scala + patch 0004/0006/0009 + C010 ctx_ctrl.yaml).
READ_ONLY_CSRS: dict[int, int | None] = {
  0x301: None,  # MISA: WARL write_mask=0 on CoralNPU (patch 0006)
  0xC22: 16,  # VLENB = 128b / 8 = 16
  0xF11: 0x426,  # MVENDORID
  0xF12: 0,  # MARCHID
  0xF13: 0,  # MIMPID
  0xF14: 0,  # MHARTID (hart 0)
}


def csr_step_expectation(
  csr_addr: int,
  op: str,
  old_value: int,
  operand: int,
  *,
  rs1_is_x0: bool = False,
) -> CsrStepExpectation:
  """Compute `(old_value, new_value, rd_value, commit_timing)` for scalar Zicsr ops."""
  _check_u32(old_value, "old_value")
  _check_u32(operand, "operand")
  if op not in ("csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci"):
    raise BitwiseScalarError("ILLEGAL_CSR_OP", op)
  is_write = not (op in ("csrrs", "csrrc", "csrrsi", "csrrci") and rs1_is_x0)
  if csr_addr >= 0xC00 and is_write:
    # RISC-V Privileged Spec: CSR[11:10] == 2'b11 is read-only; writing traps.
    raise BitwiseScalarError("CSR_READ_ONLY_TRAP", f"csr={csr_addr:#05x}")

  if op in ("csrrw", "csrrwi"):
    raw = operand
  elif op in ("csrrs", "csrrsi"):
    raw = old_value | operand if is_write else old_value
  else:
    raw = old_value & (~operand & MASK32) if is_write else old_value

  if csr_addr in READ_ONLY_CSRS:
    fixed = READ_ONLY_CSRS[csr_addr]
    rd = old_value if fixed is None else fixed
    return CsrStepExpectation(rd, rd, rd, "read_only")

  if csr_addr == 0x001:  # fflags
    old_m = old_value & FFLAGS_MASK
    new_m = (raw & FFLAGS_MASK) if is_write else old_m
    return CsrStepExpectation(old_m, new_m, old_m, "retire_commit")
  if csr_addr == 0x002:  # frm
    old_m = old_value & 0x7
    new_m = (raw & 0x7) if is_write else old_m
    return CsrStepExpectation(old_m, new_m, old_m, "retire_commit")
  if csr_addr == 0x003:  # fcsr = {frm[2:0], fflags[4:0]}
    old_m = old_value & 0xFF
    new_m = (raw & 0xFF) if is_write else old_m
    return CsrStepExpectation(old_m, new_m, old_m, "retire_commit")
  if csr_addr == 0x009:  # vxsat
    old_m = old_value & 0x1
    new_m = (raw & 0x1) if is_write else old_m
    return CsrStepExpectation(old_m, new_m, old_m, "retire_commit")
  if csr_addr == 0x305:  # mtvec: Direct mode only on CoralNPU (MODE bits [1:0] masked to 0)
    new_m = (raw & ~0x3) & MASK32 if is_write else (old_value & ~0x3)
    return CsrStepExpectation(old_value & ~0x3, new_m, old_value & ~0x3, "dispatch_beat")
  if csr_addr == 0x340:  # mscratch
    new_m = raw if is_write else old_value
    return CsrStepExpectation(old_value, new_m, old_value, "dispatch_beat")
  if csr_addr == 0x300:  # mstatus: CoralNPU legalizes FS=0 -> non-zero and VS=0 -> non-zero
    new_m = raw if is_write else old_value
    if is_write:
      if ((new_m >> 13) & 0x3) == 0:
        new_m |= 0x2000
      if ((new_m >> 9) & 0x3) == 0:
        new_m |= 0x0200
    return CsrStepExpectation(old_value, new_m & MASK32, old_value, "dispatch_beat")
  if 0x7C0 <= csr_addr <= 0x7C7:  # CoralNPU MCONTEXT0..7
    new_m = raw if is_write else old_value
    return CsrStepExpectation(old_value, new_m, old_value, "dispatch_beat")

  raise BitwiseScalarError("UNSUPPORTED_SCALAR_CSR", f"csr={csr_addr:#05x}")
