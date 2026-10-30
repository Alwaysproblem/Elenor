# SPDX-License-Identifier: BSD-3-Clause
"""P03 SoftFloat 3e ctypes binding (primitive probe crosscheck backend).

Loads the pinned image library /opt/softfloat/lib/libsoftfloat.so
(Specialization RISCV) through ctypes with exact C ABI types taken from the
mandatory native ABI probe (probe/softfloat_abi.c): float32_t struct size and
alignment, uint_fast8_t, int_fast32_t, uint_fast32_t and _Bool widths are
probed values, never guessed (int_fast32_t/uint_fast32_t are NOT assumed to
be 32-bit types; on the pinned LP64 image they are 8 bytes).

Backend independence contract: this module NEVER imports the scalar oracle or
any tensor/reference code, and it normalizes nothing: the arithmetic ops
(fma, add, mul) and the signed i32 conversion return SoftFloat's raw result
bits and its diagnostic flag byte directly. SoftFloat's flag layout
(inexact=1, underflow=2, overflow=4, infinite=8, invalid=16) is exactly the
standard 5-bit NV16/DZ8/OF4/UF2/NX1 layout, identical to HardFloat
exceptionFlags and RISC-V fflags, so those ops pass the flag byte through
unmapped.

SoftFloat 3e has NO native BF16 (bfloat16) API: op "bf16" runs the clearly
labeled independent adapter below. It delegates the single RNE decision to
SoftFloat integer rounding (f32_to_ui32, round_near_even, exact=True) over
provably exactly-representable fixed-point images, and derives
tiny-after/inexact/overflow flags explicitly. Its canonical NaN policy
candidate is +0x7fc0 (quiet, zero sign; signaling NaN raises NV); the parent
freezes the policy only against the measured HardFloat probe bits, never by
normalizing DUT output.

For op "i32" SoftFloat raises invalid (not overflow) on NaN/Inf and
out-of-range results and saturates per the RISCV specialization
(0x7fffffff / 0x80000000 / NaN->0x7fffffff); the binding maps
invalid -> NV and inexact -> NX, matching the HardFloat RecFNToIN probe
mapping (invalid|overflow -> NV, inexact -> NX).

Process-global state: this pinned build has no TLS. The binding is single-threaded,
pins round_near_even and tininess_after, and clears flags before every compute().
The ABI argument is the JSON emitted by the mandatory native ABI executable.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

ROUND_NEAR_EVEN = 0
TININESS_AFTER_ROUNDING = 1

# Standard 5-bit diagnostic layout (NV16/DZ8/OF4/UF2/NX1).
FLAG_NX = 1
FLAG_UF = 2
FLAG_OF = 4
FLAG_DZ = 8
FLAG_NV = 16

# SoftFloat flag byte (same bit positions; named separately so the passthrough
# vs. mapped distinction stays visible at call sites).
SF_FLAG_INEXACT = 1

OPS = ("fma", "add", "mul", "bf16", "i32")

_F32_2POW_NEG16_BITS = 0x37800000  # +2.0^-16, exact power-of-two scale operand

_UNSIGNED_BY_SIZE = {1: ctypes.c_uint8, 2: ctypes.c_uint16, 4: ctypes.c_uint32, 8: ctypes.c_uint64}
_SIGNED_BY_SIZE = {1: ctypes.c_int8, 2: ctypes.c_int16, 4: ctypes.c_int32, 8: ctypes.c_int64}


class F32(ctypes.Structure):
  """softfloat_types.h: typedef struct { uint32_t v; } float32_t;"""

  _fields_ = [("v", ctypes.c_uint32)]


def _unsigned_type(size: int):
  try:
    return _UNSIGNED_BY_SIZE[size]
  except KeyError:
    raise ValueError(f"unsupported probed unsigned width: {size}") from None


def _signed_type(size: int):
  try:
    return _SIGNED_BY_SIZE[size]
  except KeyError:
    raise ValueError(f"unsupported probed signed width: {size}") from None


class SoftFloat:
  """ctypes binding for the pinned SoftFloat 3e (RISCV) library.

  compute(op, a, b=0, c=0) -> (bits, flags): raw measured result words, no
  missing-probe fallback and no post-hoc correction of any value.
  """

  _ABI_FIELDS = (
    "sizeof_float32_t",
    "alignof_float32_t",
    "sizeof_uint_fast8_t",
    "sizeof_int_fast32_t",
    "sizeof_uint_fast32_t",
    "sizeof_bool",
  )

  def __init__(self, library: str | Path, abi_probe: str | Path) -> None:
    abi = self._load_abi(Path(abi_probe))
    self._abi = abi
    if abi["sizeof_float32_t"] != ctypes.sizeof(F32) or abi["alignof_float32_t"] != ctypes.alignment(F32):
      raise ValueError(
        "float32_t ABI mismatch: probe reports "
        f"size={abi['sizeof_float32_t']} align={abi['alignof_float32_t']}, "
        f"ctypes maps size={ctypes.sizeof(F32)} "
        f"align={ctypes.alignment(F32)}"
      )
    self._u_fast8 = _unsigned_type(abi["sizeof_uint_fast8_t"])
    self._u_fast32 = _unsigned_type(abi["sizeof_uint_fast32_t"])
    self._i_fast32 = _signed_type(abi["sizeof_int_fast32_t"])
    self._bool_t = ctypes.c_bool
    if abi["sizeof_bool"] != ctypes.sizeof(self._bool_t):
      raise ValueError("native bool ABI does not match ctypes.c_bool")

    self._lib = ctypes.CDLL(str(library))
    self._bind()
    self._rounding_mode = self._u_fast8.in_dll(self._lib, "softfloat_roundingMode")
    self._detect_tininess = self._u_fast8.in_dll(self._lib, "softfloat_detectTininess")
    self._exception_flags = self._u_fast8.in_dll(self._lib, "softfloat_exceptionFlags")
    self._rounding_mode.value = ROUND_NEAR_EVEN
    self._detect_tininess.value = TININESS_AFTER_ROUNDING
    self._exception_flags.value = 0

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------
  def compute(self, op: str, a: int, b: int = 0, c: int = 0) -> tuple[int, int]:
    """Returns (result bits, standard 5-bit diagnostic flags)."""
    if op not in OPS:
      raise ValueError(f"unknown probe op: {op!r}")
    ai = self._word("a", a)
    bi = self._word("b", b)
    ci = self._word("c", c)
    self._rounding_mode.value = ROUND_NEAR_EVEN
    self._detect_tininess.value = TININESS_AFTER_ROUNDING
    self._exception_flags.value = 0
    if op == "fma":
      bits = self._f32_mul_add(F32(ai), F32(bi), F32(ci)).v
      return bits, self._exception_flags.value
    if op == "add":
      bits = self._f32_add(F32(ai), F32(bi)).v
      return bits, self._exception_flags.value
    if op == "mul":
      bits = self._f32_mul(F32(ai), F32(bi)).v
      return bits, self._exception_flags.value
    if op == "bf16":
      return self._f32_to_bf16(ai)
    raw = self._f32_to_i32(F32(ai), ROUND_NEAR_EVEN, True)
    return int(raw) & 0xFFFFFFFF, int(self._exception_flags.value)

  # ------------------------------------------------------------------
  # Independent BF16 adapter.
  #
  # Clearly labeled independent code: SoftFloat 3e offers f16/f32/f64/extF80/
  # f128 only, never a BF16 API, and nothing here is claimed to be one. The
  # single RNE decision is delegated to SoftFloat integer rounding
  # (f32_to_ui32, round_near_even, exact=True) over fixed-point images that
  # are provably exactly representable:
  #   value = m * 2^q exactly; the BF16 grid is n * 2^-133 (8-bit significand,
  #   min normal 2^-126), so k = RNE(value * 2^133) is the grid index.
  #   f32 subnormal input: value = frac * 2^-149 -> k = RNE(frac * 2^-16),
  #     computed as ui32_to_f32(frac) * 2^-16 then f32_to_ui32.
  #   f32 normal input: value = m24 * 2^(E-23); only m24/2^16 in [128, 256)
  #     has to be rounded, computed as ui32_to_f32(m24) * 2^-16 then
  #     f32_to_ui32; the exponent is tracked as an integer with carry.
  # The 2^-16 rescale is an f32 multiply whose product is exactly
  # representable (<= 24 significant bits, exponent far from the subnormal
  # and overflow ranges), so it cannot round or flag under any rounding mode.
  # Flags are derived explicitly (tininess AFTER rounding, no FTZ):
  #   NX whenever the grid decision was inexact;
  #   NV for signaling-NaN input only (quiet NaN converts silently);
  #   OF only when the significand carry leaves the exponent range (-> +-Inf;
  #     the carry implies inexact, so OF|NX like the arithmetic path);
  #   UF when precision rounding with an unbounded exponent remains tiny and
  #     the final grid decision is inexact (even if its encoding is min-normal).
  # Canonical NaN policy candidate: +0x7fc0.
  # ------------------------------------------------------------------
  def _f32_to_bf16(self, a: int) -> tuple[int, int]:
    sign = a >> 31
    exp = (a >> 23) & 0xFF
    frac = a & 0x007FFFFF
    if exp == 0xFF:
      if frac == 0:
        return (sign << 15) | 0x7F80, 0  # +/-Inf is exact, no flag
      nv = 0 if (frac & 0x00400000) else FLAG_NV
      return 0x7FC0, nv
    if exp == 0 and frac == 0:
      return sign << 15, 0
    if exp == 0:
      t = self._ui32_to_f32(frac)
      k = self._f32_to_ui32_rne_exact(self._exact_rescale(t))
      sf = self._exception_flags.value
      flags = FLAG_NX if (sf & SF_FLAG_INEXACT) else 0
      # The virtual BF16 normal precision below 2^-126 has quantum 2^-134.
      # Its halfway point below min-normal is 2^-126 - 2^-135, or frac=0x7fc000.
      # RNE at that tie carries into min-normal. This independent integer
      # threshold is not the final subnormal-grid encoding test (k < 128).
      if frac < 0x007FC000 and flags & FLAG_NX:
        flags |= FLAG_UF
      return (sign << 15) | k, flags
    m24 = 0x00800000 | frac
    e = exp - 127
    t = self._ui32_to_f32(m24)
    k8 = self._f32_to_ui32_rne_exact(self._exact_rescale(t))
    sf = self._exception_flags.value
    flags = FLAG_NX if (sf & SF_FLAG_INEXACT) else 0
    if k8 == 256:
      k8 = 128
      e += 1
    if e > 127:
      return (sign << 15) | 0x7F80, flags | FLAG_OF
    return (sign << 15) | ((e + 127) << 7) | (k8 - 128), flags

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------
  def _bind(self) -> None:
    lib = self._lib

    def fn(name: str, args: list, res):
      f = getattr(lib, name)
      f.argtypes = args
      f.restype = res
      return f

    self._f32_mul_add = fn("f32_mulAdd", [F32, F32, F32], F32)
    self._f32_add = fn("f32_add", [F32, F32], F32)
    self._f32_mul = fn("f32_mul", [F32, F32], F32)
    self._f32_to_ui32 = fn("f32_to_ui32", [F32, self._u_fast8, self._bool_t], self._u_fast32)
    self._f32_to_i32 = fn("f32_to_i32", [F32, self._u_fast8, self._bool_t], self._i_fast32)
    self._ui32_to_f32 = fn("ui32_to_f32", [ctypes.c_uint32], F32)

  def _exact_rescale(self, t: F32) -> F32:
    """Multiplies by exactly 2^-16 inside SoftFloat (exact, never flags)."""
    scaled = self._f32_mul(t, F32(_F32_2POW_NEG16_BITS))
    if self._exception_flags.value != 0:
      raise AssertionError("bf16 adapter: exact 2^-16 rescale raised flags")
    return scaled

  def _f32_to_ui32_rne_exact(self, t: F32) -> int:
    return int(self._f32_to_ui32(t, ROUND_NEAR_EVEN, True))

  @staticmethod
  def _load_abi(path: Path) -> dict[str, int]:
    try:
      data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
      raise ValueError(f"cannot read ABI probe JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
      raise ValueError(f"ABI probe JSON must be an object: {path}")
    out: dict[str, int] = {}
    for name in SoftFloat._ABI_FIELDS:
      v = data.get(name)
      if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ValueError(f"ABI probe JSON field {name!r} missing/invalid: {path}")
      out[name] = v
    return out

  @staticmethod
  def _word(name: str, v: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
      raise ValueError(f"{name} must be an int bit pattern, got {v!r}")
    if not 0 <= v <= 0xFFFFFFFF:
      raise ValueError(f"{name} out of unsigned 32-bit range: {v!r}")
    return v
