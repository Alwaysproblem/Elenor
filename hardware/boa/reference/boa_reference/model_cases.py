"""Deterministic generator cases and behavior checks for the tensor slice.

Case entries are JSON-ready dicts {id, input, expected, acceptance}.  All
bit patterns are hex strings; "expected" is the golden result of the
slice's own oracle (TensorResult.as_dict / conv2d_reference dict /
EpilogueResult.as_dict) or {"error": <ExceptionName>} for cases that
must be rejected.  Generators are driven by an internal LCG so the same
seed always produces byte-identical cases on any Python version.

model_checks() executes the real functions and returns
list[tuple[str, bool, str]]: check name, actual comparison outcome, and
observed detail.  It never fabricates a passing constant.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

from . import scalar
from .convolution import conv2d_reference
from .epilogue import Epilogue, apply_epilogue, validate_epilogue
from .tensor import MatmulStep, matmul_reference

_DEFAULT_SEED = 20261028
_RANDOM_MATMUL_CASES = 8
_RANDOM_EPILOGUE_CASES = 6
_RANDOM_CONV_CASES = 4

_BF16_ONE = 0x3F80
_BF16_TWO = 0x4000
_BF16_NAN = 0x7FC0
_BF16_INF = 0x7F80
_BF16_NEGZERO = 0x8000
_DIRECT_B = (0x4B80, 0x0000, 0x0000, 0x0000, 0x3F80, 0xCB80, 0x0000, 0x0000)

_F32_PLUS_ZERO = 0x00000000
_F32_ONE = 0x3F800000
_F32_TWO = 0x40000000
_F32_THREE = 0x40400000
_F32_SIX = 0x40C00000
_F32_NAN = 0x7FC00000
_F32_INF = 0x7F800000

_BF16_POOL = (
  0x0000,
  0x8000,
  0x3F80,
  0xBF80,
  0x4000,
  0xC000,
  0x3FC0,
  0x4040,
  0x40A0,
  0xC040,
  0x0001,
  0x8001,
  0x0080,
  0x7F80,
  0xFF80,
  0x7FC0,
  0x7F81,
  0x3F01,
  0x42C8,
)
_F32_POOL = (
  0x3F800000,
  0xBF800000,
  0x3F000000,
  0xBF000000,
  0x40A00000,
  0xC0A00000,
  0x3FC00000,
  0x80000000,
  0x00000000,
  0x7F800000,
  0xFF800000,
  0x7FC00000,
  0x7F800001,
  0x7F7FFFFF,
  0x00000001,
  0x3F808000,
  0x40200000,
)
_BIAS_POOL = (0x3F800000, 0xBF800000, 0x40000000, 0x00000000, 0x80000000)
_CLIP_POOL = (
  (0xC0400000, 0x40A00000),
  (0xFF800000, 0x7F800000),
  (0x00000000, 0x3F800000),
  (0x80000000, 0x00000000),
)


class _Lcg:
  """Small deterministic LCG; avoids any dependence on stdlib RNG details."""

  def __init__(self, seed: int):
    self._state = (int(seed) ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF

  def _next(self) -> int:
    self._state = (self._state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return self._state

  def below(self, bound: int) -> int:
    return self._next() % bound

  def bits(self, width: int) -> int:
    return self._next() & ((1 << width) - 1)

  def choice(self, seq):
    return seq[self.below(len(seq))]


def _h16(v: int) -> str:
  return f"0x{v:04x}"


def _h32(v: int) -> str:
  return f"0x{v:08x}"


def _step_json(step: MatmulStep) -> dict:
  return {
    "k": step.k,
    "a": [_h16(v) for v in step.a],
    "b": [_h16(v) for v in step.b],
    "init": step.init,
    "final": step.final,
    "seq": step.seq,
    "storage_k": step.storage_k,
    "mask": None if step.mask is None else list(step.mask),
  }


def _epilogue_json(spec: Epilogue | None) -> dict | None:
  if spec is None:
    return None
  return {
    "bias": None if spec.bias is None else [_h32(v) for v in spec.bias],
    "scale": None if spec.scale is None else _h32(spec.scale),
    "relu": spec.relu,
    "clip": None if spec.clip is None else [_h32(v) for v in spec.clip],
    "output_dtype": spec.output_dtype,
    "inv_scale": None if spec.inv_scale is None else _h32(spec.inv_scale),
    "zero_point": spec.zero_point,
  }


def _golden(fn, *, input_payload: dict, case_id: str, acceptance: list) -> dict:
  try:
    expected = fn()
  except Exception as exc:  # captured deliberately: rejection is the golden
    if "illegal" not in case_id:
      raise
    expected = {"error": type(exc).__name__, "message": str(exc)}
  return {"id": case_id, "input": input_payload, "expected": expected, "acceptance": list(acceptance)}


def _matmul_case(case_id, m, n, steps, tile_k, epilogue, acceptance):
  payload = {
    "m": m,
    "n": n,
    "tile_k": tile_k,
    "epilogue": _epilogue_json(epilogue),
    "steps": [_step_json(s) for s in steps],
  }
  return _golden(
    lambda: matmul_reference(m, n, steps, tile_k=tile_k, epilogue=epilogue).as_dict(),
    input_payload=payload,
    case_id=case_id,
    acceptance=acceptance,
  )


def _epilogue_case(case_id, am, m, n, spec, acceptance):
  payload = {"m": m, "n": n, "am": [_h32(v) for v in am], "epilogue": _epilogue_json(spec)}
  return _golden(
    lambda: apply_epilogue(am, m, n, spec).as_dict(),
    input_payload=payload,
    case_id=case_id,
    acceptance=acceptance,
  )


def _conv_case(case_id, x, weights, shape, kernel, stride, padding, dilation, tile_k, epilogue, acceptance):
  payload = {
    "x": [_h16(v) for v in x],
    "weights": [_h16(v) for v in weights],
    "shape": list(shape),
    "kernel": list(kernel),
    "stride": list(stride),
    "padding": list(padding),
    "dilation": list(dilation),
    "tile_k": tile_k,
    "epilogue": _epilogue_json(epilogue),
  }
  return _golden(
    lambda: conv2d_reference(
      x,
      weights,
      shape,
      kernel,
      stride=stride,
      padding=padding,
      dilation=dilation,
      tile_k=tile_k,
      epilogue=epilogue,
    ),
    input_payload=payload,
    case_id=case_id,
    acceptance=acceptance,
  )


def _simple_step(**overrides) -> MatmulStep:
  params: dict = {
    "k": 2,
    "a": (_BF16_ONE, _BF16_TWO),
    "b": (_BF16_TWO, 0x4040),
    "init": True,
    "final": True,
    "seq": 0,
  }
  params.update(overrides)
  return MatmulStep(**params)


def _directed_steps() -> tuple:
  a = (_BF16_ONE,) * 8
  return (MatmulStep(k=8, a=a, b=_DIRECT_B, init=True, final=True, seq=0),)


def _matmul_directed_cases() -> list:
  cases = []
  for tk in (4, 8):
    cases.append(
      _matmul_case(
        f"mm-grouping-tile{tk}", 1, 1, _directed_steps(), tk, Epilogue(), ["N-P03-5", "N-P03-8", "N-P03-1"]
      )
    )
  k3 = (
    MatmulStep(
      k=3,
      a=(_BF16_ONE, _BF16_ONE, _BF16_ONE),
      b=(_BF16_ONE, _BF16_TWO, 0x4040),
      init=True,
      final=True,
      seq=0,
    ),
  )
  for tk in (2, 8):
    cases.append(_matmul_case(f"mm-grouping-invariant-tile{tk}", 1, 1, k3, tk, Epilogue(), ["N-P03-5"]))

  a = (_BF16_ONE, _BF16_ONE, 0x0000, 0x0000)
  b = (_BF16_ONE, _BF16_ONE, _BF16_INF, _BF16_NEGZERO)
  cases.append(
    _matmul_case(
      "mm-tail-poison-nomask",
      1,
      1,
      (MatmulStep(k=2, a=a, b=b, init=True, final=True, seq=0, storage_k=4),),
      8,
      Epilogue(),
      ["N-P03-4"],
    )
  )
  cases.append(
    _matmul_case(
      "mm-tail-poison-mask",
      1,
      1,
      (
        MatmulStep(
          k=2, a=a, b=b, init=True, final=True, seq=0, storage_k=4, mask=(True, True, False, False)
        ),
      ),
      8,
      Epilogue(),
      ["N-P03-4"],
    )
  )
  mask_b = (_DIRECT_B[0], _BF16_NAN, _BF16_ONE, _BF16_ONE)
  cases.append(
    _matmul_case(
      "mm-mask-block-boundaries",
      1,
      1,
      (
        MatmulStep(
          k=4, a=(_BF16_ONE,) * 4, b=mask_b, init=True, final=True, seq=0, mask=(True, False, True, True)
        ),
      ),
      2,
      Epilogue(),
      ["N-P03-4", "N-P03-5"],
    )
  )

  chain = (
    MatmulStep(k=2, a=(_BF16_TWO, 0x4040), b=(0x4080, 0x40A0), init=True, final=False, seq=0),
    MatmulStep(k=2, a=(_BF16_TWO, 0x4040), b=(0x4080, 0x40A0), init=False, final=True, seq=1),
  )
  cases.append(_matmul_case("mm-init-add-chain", 1, 1, chain, 8, Epilogue(), ["N-P03-6"]))
  cases.append(
    _matmul_case(
      "mm-add-k0-negzero",
      1,
      1,
      (
        MatmulStep(k=1, a=(0x8001,), b=(0x0001,), init=True, final=False, seq=0),
        MatmulStep(k=0, a=(), b=(), init=False, final=True, seq=1),
      ),
      8,
      Epilogue(),
      ["N-P03-2", "N-P03-11"],
    )
  )
  cases.append(
    _matmul_case(
      "mm-init-k0-bias-scale",
      1,
      1,
      (MatmulStep(k=0, a=(), b=(), init=True, final=True, seq=0),),
      8,
      Epilogue(bias=(_F32_TWO,), scale=_F32_THREE),
      ["N-P03-11"],
    )
  )
  tie = (MatmulStep(k=3, a=(_BF16_ONE,) * 3, b=(0x4B80, 0x4000, 0x3F80), init=True, final=True, seq=0),)
  cases.append(_matmul_case("mm-rne-tie-up", 1, 1, tie, 8, Epilogue(), ["N-P03-1"]))
  cases.append(
    _matmul_case(
      "mm-m0-empty",
      0,
      2,
      (
        MatmulStep(
          k=3, a=(), b=(0x3F80, 0x4000, 0x4080, 0x4040, 0x40A0, 0x40C0), init=True, final=True, seq=0
        ),
      ),
      8,
      Epilogue(),
      ["N-P03-11"],
    )
  )
  cases.append(
    _matmul_case(
      "mm-n0-empty",
      2,
      0,
      (MatmulStep(k=3, a=(0x3F80,) * 6, b=(), init=True, final=True, seq=0),),
      8,
      Epilogue(),
      ["N-P03-11"],
    )
  )
  cases.append(
    _matmul_case(
      "mm-nonsymmetric-3x5",
      3,
      5,
      (
        MatmulStep(
          k=7,
          a=tuple(_BF16_POOL[i % len(_BF16_POOL)] for i in range(27)),
          b=tuple((0x3F80 + i) & 0xFFFF for i in range(45)),
          init=True,
          final=True,
          seq=0,
          storage_k=9,
          mask=(True, True, True, False, True, True, False, False, False),
        ),
      ),
      4,
      Epilogue(bias=(_F32_ONE, 0xBF800000, _F32_TWO, 0x80000000, 0x3F000000)),
      ["N-P03-6", "N-P03-4"],
    )
  )

  stage_steps = (
    MatmulStep(
      k=3,
      a=(_BF16_ONE, _BF16_TWO, 0x4080),
      b=(0xBF80, _BF16_TWO, 0xBF80, _BF16_TWO, 0xBF80, _BF16_TWO),
      init=True,
      final=True,
      seq=0,
    ),
  )
  stage_specs = (
    ("bias", Epilogue(bias=(0x3F000000, 0xBF800000))),
    ("scale", Epilogue(scale=_F32_TWO)),
    ("relu", Epilogue(relu=True)),
    ("clip", Epilogue(clip=(0xC1000000, 0x41200000))),
    (
      "all",
      Epilogue(bias=(0x3F000000, 0x3F000000), scale=_F32_TWO, relu=True, clip=(0x00000000, 0x41A00000)),
    ),
    ("none", Epilogue()),
    ("bf16", Epilogue(output_dtype="bf16")),
    ("i8", Epilogue(output_dtype="i8", inv_scale=0x3F000000, zero_point=3)),
  )
  for name, spec in stage_specs:
    tag = ["N-P03-7"] if spec.output_dtype == "i8" else ["N-P03-3"]
    cases.append(_matmul_case(f"mm-stages-{name}", 1, 2, stage_steps, 8, spec, tag))
  cases.extend(_matmul_illegal_cases())
  return cases


def _matmul_illegal_cases() -> list:
  a = (0x3F80, 0x4000)
  b = (0x4000, 0x4040)
  tag = ["N-P03-11"]
  entries: list[tuple[str, dict]] = [
    ("tile-k-zero", {"steps": (_simple_step(),), "tile_k": 0}),
    ("empty-steps", {"steps": ()}),
    ("seq-gap", {"steps": (MatmulStep(k=2, a=a, b=b, init=True, final=True, seq=1),)}),
    ("init-not-first", {"steps": (MatmulStep(k=2, a=a, b=b, init=False, final=True, seq=0),)}),
    (
      "init-on-add",
      {
        "steps": (
          MatmulStep(k=2, a=a, b=b, init=True, final=False, seq=0),
          MatmulStep(k=2, a=a, b=b, init=True, final=True, seq=1),
        )
      },
    ),
    ("final-missing", {"steps": (MatmulStep(k=2, a=a, b=b, init=True, final=False, seq=0),)}),
    (
      "final-early",
      {
        "steps": (
          MatmulStep(k=2, a=a, b=b, init=True, final=True, seq=0),
          MatmulStep(k=2, a=a, b=b, init=False, final=True, seq=1),
        )
      },
    ),
    (
      "storage-below-k",
      {
        "steps": (
          MatmulStep(
            k=3,
            a=(0x3F80, 0x4000, 0x4040),
            b=(0x4000, 0x4040, 0x4080),
            init=True,
            final=True,
            seq=0,
            storage_k=2,
          ),
        )
      },
    ),
    ("a-length", {"steps": (MatmulStep(k=2, a=(0x3F80,), b=b, init=True, final=True, seq=0),)}),
    ("b-length", {"steps": (MatmulStep(k=2, a=a, b=(0x4000,), init=True, final=True, seq=0),)}),
    ("word-range", {"steps": (MatmulStep(k=2, a=(0x10000, 0x4000), b=b, init=True, final=True, seq=0),)}),
    ("word-negative", {"steps": (MatmulStep(k=2, a=(-1, 0x4000), b=b, init=True, final=True, seq=0),)}),
    (
      "mask-length",
      {"steps": (MatmulStep(k=2, a=a, b=b, init=True, final=True, seq=0, mask=(True, True, False)),)},
    ),
    (
      "mask-tail-true",
      {
        "steps": (
          MatmulStep(
            k=1,
            a=(0x3F80, 0x4000, 0x4040),
            b=(0x4000, 0x4040, 0x4080),
            init=True,
            final=True,
            seq=0,
            storage_k=3,
            mask=(True, False, True),
          ),
        )
      },
    ),
    # Deliberately malformed runtime payloads exercise rejection, not static type conformance.
    (
      "mask-not-bool",
      {"steps": (MatmulStep(k=2, a=a, b=b, init=True, final=True, seq=0, mask=cast(Any, (1, 0))),)},
    ),
    ("negative-m", {"m": -1}),
    ("negative-k", {"steps": (MatmulStep(k=-1, a=(), b=(), init=True, final=True, seq=0),)}),
  ]
  cases = []
  for name, kwargs in entries:
    m = kwargs.get("m", 1)
    n = kwargs.get("n", 1)
    steps = kwargs.get("steps", (_simple_step(),))
    tile_k = kwargs.get("tile_k", 8)
    cases.append(_matmul_case(f"mm-illegal-{name}", m, n, steps, tile_k, None, tag))
  return cases


def _combination_epilogue_cases() -> list:
  am = (_F32_NAN, 0x7F800001, 0x80000000, _F32_INF, 0xBF800000, 0x3FC00000)
  bias = (_F32_TWO, 0xC0000000, _F32_PLUS_ZERO)
  cases = []
  for combo in range(16):
    for dtype in ("f32", "bf16", "i8"):
      spec = Epilogue(
        bias=bias if combo & 1 else None,
        scale=0x40400000 if combo & 2 else None,
        relu=bool(combo & 4),
        clip=(0xC0400000, 0x40A00000) if combo & 8 else None,
        output_dtype=dtype,
        inv_scale=0x3F800000 if dtype == "i8" else None,
      )
      cases.append(_epilogue_case(f"epi-combo-{combo:02d}-{dtype}", am, 2, 3, spec, ["N-P03-2", "N-P03-3"]))
  return cases


def _epilogue_directed_cases() -> list:
  cases = []
  i8 = lambda **kw: Epilogue(output_dtype="i8", **kw)
  cases.append(
    _epilogue_case(
      "epi-i8-ties",
      (0x3F000000, 0x3FC00000, 0x40200000, 0x40600000, 0xBF000000, 0xBFC00000),
      1,
      6,
      i8(inv_scale=_F32_ONE),
      ["N-P03-7", "N-P03-1"],
    )
  )
  cases.append(
    _epilogue_case(
      "epi-i8-saturation",
      (0x7F7FFFFF, 0xFF7FFFFF, _F32_INF, 0xFF800000),
      1,
      4,
      i8(inv_scale=_F32_ONE),
      ["N-P03-7"],
    )
  )
  cases.append(
    _epilogue_case(
      "epi-i8-overflow-product",
      (0x7F7FFFFF, 0xFF7FFFFF),
      1,
      2,
      i8(inv_scale=0x7F7FFFFF),
      ["N-P03-3", "N-P03-7"],
    )
  )
  cases.append(
    _epilogue_case(
      "epi-i8-nan-zp", (_F32_NAN, 0x7F800001), 1, 2, i8(inv_scale=_F32_ONE, zero_point=-1), ["N-P03-7"]
    )
  )
  cases.append(
    _epilogue_case(
      "epi-i8-zp-boundaries",
      (_F32_PLUS_ZERO, 0x3F000000, 0xBF000000, _F32_INF),
      1,
      4,
      i8(inv_scale=_F32_ONE, zero_point=127),
      ["N-P03-7"],
    )
  )
  cases.append(
    _epilogue_case(
      "epi-i8-zp-min",
      (_F32_PLUS_ZERO, 0xBF000000),
      1,
      2,
      i8(inv_scale=_F32_ONE, zero_point=-128),
      ["N-P03-7"],
    )
  )
  cases.append(
    _epilogue_case(
      "epi-i8-tiny-inv-scale", (_F32_ONE, 0x40A00000), 1, 2, i8(inv_scale=0x00000001), ["N-P03-7"]
    )
  )
  cases.append(
    _epilogue_case(
      "epi-bf16-ties",
      (0x3F808000, 0x3F818000, 0x3F807FFF, 0x3F808001),
      1,
      4,
      Epilogue(output_dtype="bf16"),
      ["N-P03-1"],
    )
  )
  cases.append(
    _epilogue_case(
      "epi-bf16-nan-canonical",
      (_F32_NAN, 0x7F800001, 0xFFC00001),
      1,
      3,
      Epilogue(output_dtype="bf16"),
      ["N-P03-3"],
    )
  )
  cases.append(
    _epilogue_case(
      "epi-relu-negzero-inf",
      (0x80000000, 0xFF800000, 0xBF800000, _F32_ONE),
      1,
      4,
      Epilogue(relu=True),
      ["N-P03-2", "N-P03-3"],
    )
  )
  cases.append(
    _epilogue_case(
      "epi-clip-equality-negzero",
      (0x80000000, _F32_INF, 0xFF800000, _F32_ONE),
      1,
      4,
      Epilogue(clip=(0x00000000, _F32_TWO)),
      ["N-P03-2"],
    )
  )
  cases.append(
    _epilogue_case(
      "epi-clip-inf-bounds",
      (_F32_NAN, _F32_INF, 0x80000000, 0x40A00000),
      1,
      4,
      Epilogue(clip=(0xFF800000, 0x7F800000)),
      ["N-P03-3"],
    )
  )
  cases.append(
    _epilogue_case(
      "epi-scale1-executes", (0x7F800001, 0x80000000), 1, 2, Epilogue(scale=_F32_ONE), ["N-P03-3"]
    )
  )
  cases.append(_epilogue_case("epi-empty-n0", (), 0, 0, Epilogue(bias=()), ["N-P03-11"]))
  cases.extend(_epilogue_illegal_cases())
  return cases


def _epilogue_illegal_cases() -> list:
  tag = ["N-P03-11"]
  entries = [
    ("bias-length", Epilogue(bias=(_F32_ONE, _F32_TWO)), 1, 3),
    ("bias-word-range", Epilogue(bias=(_F32_ONE, _F32_TWO, 0x100000000)), 1, 3),
    ("bias-word-negative", Epilogue(bias=(_F32_ONE, _F32_TWO, -1)), 1, 3),
    ("scale-word-range", Epilogue(scale=0x100000000), 1, 1),
    ("clip-nan-bound", Epilogue(clip=(_F32_NAN, _F32_ONE)), 1, 1),
    ("clip-reversed", Epilogue(clip=(_F32_ONE, _F32_PLUS_ZERO)), 1, 1),
    ("clip-triple", Epilogue(clip=cast(Any, (_F32_PLUS_ZERO, _F32_ONE, _F32_TWO))), 1, 1),
    ("i8-missing-inv-scale", Epilogue(output_dtype="i8"), 1, 1),
    ("inv-scale-zero", Epilogue(output_dtype="i8", inv_scale=_F32_PLUS_ZERO), 1, 1),
    ("inv-scale-negzero", Epilogue(output_dtype="i8", inv_scale=0x80000000), 1, 1),
    ("inv-scale-nan", Epilogue(output_dtype="i8", inv_scale=_F32_NAN), 1, 1),
    ("inv-scale-inf", Epilogue(output_dtype="i8", inv_scale=_F32_INF), 1, 1),
    ("inv-scale-negative", Epilogue(output_dtype="i8", inv_scale=0xBF800000), 1, 1),
    ("zero-point-high", Epilogue(output_dtype="i8", inv_scale=_F32_ONE, zero_point=200), 1, 1),
    ("zero-point-low", Epilogue(output_dtype="i8", inv_scale=_F32_ONE, zero_point=-129), 1, 1),
    ("zero-point-bool", Epilogue(output_dtype="i8", inv_scale=_F32_ONE, zero_point=True), 1, 1),
    ("quant-on-f32-inv", Epilogue(inv_scale=_F32_ONE), 1, 1),
    ("quant-on-f32-zp", Epilogue(zero_point=1), 1, 1),
    ("unknown-dtype", Epilogue(output_dtype="fp16"), 1, 1),
    ("relu-not-bool", Epilogue(relu=cast(Any, 1)), 1, 1),
  ]
  cases = []
  for name, spec, m, n in entries:
    cases.append(_epilogue_case(f"epi-illegal-{name}", (_F32_ONE,), m, n, spec, tag))
  return cases


def _rand_bf16(rng: _Lcg) -> int:
  roll = rng.below(100)
  if roll < 60:
    return rng.choice(_BF16_POOL)
  if roll < 85:
    sign = rng.below(2) << 15
    expo = (118 + rng.below(18)) << 7
    return sign | expo | rng.bits(7)
  return rng.bits(16)


def _split_k(rng: _Lcg, k: int, count: int) -> list:
  parts = [k // count] * count
  for index in range(k % count):
    parts[index] += 1
  return parts


def _random_epilogue(rng: _Lcg, n: int) -> Epilogue | None:
  if rng.below(10) == 0:
    return None
  dtype = ("f32", "bf16", "i8")[rng.below(3)]
  bias = None
  if rng.below(2):
    bias = tuple(rng.choice(_BIAS_POOL) for _ in range(n))
  scale = rng.choice((None, 0x3F800000, 0x40000000, 0x3F000000))
  relu = rng.below(2) == 1
  clip = None
  if rng.below(3) == 1:
    clip = rng.choice(_CLIP_POOL)
  if dtype == "i8":
    return Epilogue(
      bias=bias,
      scale=scale,
      relu=relu,
      clip=clip,
      output_dtype=dtype,
      inv_scale=rng.choice((0x3F800000, 0x3F000000, 0x40000000)),
      zero_point=rng.choice((0, -1, 3, 127, -128)),
    )
  return Epilogue(bias=bias, scale=scale, relu=relu, clip=clip, output_dtype=dtype)


def _random_matmul_case(rng: _Lcg, index: int, default_tile_k: int) -> dict:
  m = 1 + rng.below(4)
  n = 1 + rng.below(4)
  k = 1 + rng.below(16)
  storage = k + rng.choice((0, 0, 1, 3))
  count = 1 + rng.below(3)
  tile_k = default_tile_k if rng.below(10) < 7 else 2 + rng.below(7)
  parts = _split_k(rng, k, count)
  steps = []
  for seq, step_k in enumerate(parts):
    a_words = tuple(_rand_bf16(rng) for _ in range(m * storage))
    b_words = tuple(_rand_bf16(rng) for _ in range(storage * n))
    mask = None
    if rng.below(10) < 4:
      mask = tuple(rng.below(10) < 8 for _ in range(step_k))
      mask = mask + (False,) * (storage - step_k)
    steps.append(
      MatmulStep(
        k=step_k,
        a=a_words,
        b=b_words,
        init=(seq == 0),
        final=(seq == count - 1),
        seq=seq,
        storage_k=storage if storage != step_k else None,
        mask=mask,
      )
    )
  epilogue = _random_epilogue(rng, n)
  return _matmul_case(f"mm-rand-{index:02d}", m, n, tuple(steps), tile_k, epilogue, ["N-P03-6", "N-P03-10"])


def _random_epilogue_case(rng: _Lcg, index: int) -> dict:
  n = 2 + rng.below(4)
  am = tuple(rng.choice(_F32_POOL) for _ in range(n))
  spec = _random_epilogue(rng, n)
  if spec is None:
    spec = Epilogue()
  return _epilogue_case(f"epi-rand-{index:02d}", am, 1, n, spec, ["N-P03-3", "N-P03-10"])


def _random_conv_case(rng: _Lcg, index: int, default_tile_k: int) -> dict:
  n = 1 + rng.below(2)
  c = 1 + rng.below(3)
  h = 1 + rng.below(5)
  w = 1 + rng.below(5)
  o = 1 + rng.below(3)
  kh = 1 + rng.below(3)
  kw = 1 + rng.below(3)
  stride = (1 + rng.below(2), 1 + rng.below(2))
  dilation = (1 + rng.below(2), 1 + rng.below(2))
  padding = (rng.below(3), rng.below(3), rng.below(3), rng.below(3))
  x = tuple(_rand_bf16(rng) for _ in range(n * c * h * w))
  weights = tuple(_rand_bf16(rng) for _ in range(o * c * kh * kw))
  tile_k = default_tile_k if rng.below(10) < 7 else 2 + rng.below(7)
  epilogue = _random_epilogue(rng, o)
  return _conv_case(
    f"conv-rand-{index:02d}",
    x,
    weights,
    (n, c, h, w),
    (o, c, kh, kw),
    stride,
    padding,
    dilation,
    tile_k,
    epilogue,
    ["N-P03-9", "N-P03-10"],
  )


_BF16_INT1_29 = (
  0x3F80,
  0x4000,
  0x4040,
  0x4080,
  0x40A0,
  0x40C0,
  0x40E0,
  0x4100,
  0x4110,
  0x4120,
  0x4130,
  0x4140,
  0x4150,
  0x4160,
  0x4170,
  0x4180,
  0x4188,
  0x4190,
  0x4198,
  0x41A0,
  0x41A8,
  0x41B0,
  0x41B8,
  0x41C0,
  0x41C8,
  0x41D0,
  0x41D8,
  0x41E0,
  0x41E8,
)


def _conv_directed_cases() -> list:
  cases = []
  x9 = (0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0, 0x40E0, 0x4100, 0x4110)
  ones = (0x3F80, 0x3F80, 0x3F80, 0x3F80)
  cases.append(
    _conv_case(
      "conv-ones-2x2",
      x9,
      ones,
      (1, 1, 3, 3),
      (1, 1, 2, 2),
      (1, 1),
      (0, 0, 0, 0),
      (1, 1),
      8,
      Epilogue(),
      ["N-P03-9"],
    )
  )

  x_words = tuple(
    _BF16_INT1_29[(n_i * 40 + c_i * 20 + h_i * 5 + w_i) % 29]
    for n_i in range(2)
    for c_i in range(2)
    for h_i in range(4)
    for w_i in range(5)
  )
  w_pool = (0x3F80, 0x3F00, 0xBF80, 0x4000, 0x3E80, 0xBF00)
  weights = tuple(w_pool[i % len(w_pool)] for i in range(36))
  cases.append(
    _conv_case(
      "conv-asymmetric",
      x_words,
      weights,
      (2, 2, 4, 5),
      (3, 2, 2, 3),
      (2, 1),
      (1, 2, 0, 3),
      (2, 1),
      8,
      Epilogue(),
      ["N-P03-9"],
    )
  )

  cases.append(
    _conv_case(
      "conv-empty",
      (0x3F80, 0x4000, 0x4080, 0x40A0),
      (0x3F80,) * 9,
      (1, 1, 2, 2),
      (1, 1, 3, 3),
      (1, 1),
      (0, 0, 0, 0),
      (1, 1),
      8,
      None,
      ["N-P03-11", "N-P03-9"],
    )
  )
  cases.append(
    _conv_case(
      "conv-pad-poison",
      (0x3F80, 0x4000, 0x4080, 0x40A0),
      (_BF16_INF, 0x3F80, 0x3F80, 0x3F80),
      (1, 1, 2, 2),
      (1, 1, 2, 2),
      (1, 1),
      (1, 0, 1, 0),
      (1, 1),
      8,
      Epilogue(),
      ["N-P03-3", "N-P03-9"],
    )
  )
  for tk in (4, 8):
    cases.append(
      _conv_case(
        f"conv-tile{tk}",
        (_BF16_ONE,) * 8,
        _DIRECT_B,
        (1, 1, 1, 8),
        (1, 1, 1, 8),
        (1, 1),
        (0, 0, 0, 0),
        (1, 1),
        tk,
        Epilogue(),
        ["N-P03-5", "N-P03-8", "N-P03-9"],
      )
    )

  x_pow = (0x3F80, 0x4000, 0x4080, 0x4100, 0x4180, 0x4200, 0x4280, 0x4300)
  w_pow = tuple((127 + idx) << 7 for idx in range(16))
  cases.append(
    _conv_case(
      "conv-k-order",
      x_pow,
      w_pow,
      (1, 2, 2, 2),
      (2, 2, 2, 2),
      (1, 1),
      (0, 0, 0, 0),
      (1, 1),
      8,
      Epilogue(),
      ["N-P03-9"],
    )
  )
  cases.append(
    _conv_case(
      "conv-i8-out",
      x9,
      ones,
      (1, 1, 3, 3),
      (1, 1, 2, 2),
      (1, 1),
      (0, 0, 0, 0),
      (1, 1),
      4,
      Epilogue(output_dtype="i8", inv_scale=0x3F000000, zero_point=3),
      ["N-P03-7", "N-P03-9"],
    )
  )
  return cases


def generate_matmul_cases(seed: int, tile_k: int = 8) -> list[dict]:
  rng = _Lcg(seed)
  cases = _matmul_directed_cases()
  cases.extend(_random_matmul_case(rng, index, tile_k) for index in range(_RANDOM_MATMUL_CASES))
  return cases


def generate_epilogue_cases(seed: int) -> list[dict]:
  rng = _Lcg(seed)
  cases = _combination_epilogue_cases()
  cases.extend(_epilogue_directed_cases())
  cases.extend(_random_epilogue_case(rng, index) for index in range(_RANDOM_EPILOGUE_CASES))
  return cases


def generate_conv_cases(seed: int, tile_k: int = 8) -> list[dict]:
  rng = _Lcg(seed)
  cases = _conv_directed_cases()
  cases.extend(_random_conv_case(rng, index, tile_k) for index in range(_RANDOM_CONV_CASES))
  return cases


def _manual_am(b_words, blocks):
  """Independent per-cell recomputation following the frozen contract text."""
  am = None
  for active in blocks:
    p = _F32_PLUS_ZERO
    for k in active:
      p = scalar.fma_f32_rne(_F32_ONE, scalar.bf16_to_f32(b_words[k]), p).bits
    am = p if am is None else scalar.add_f32_rne(am, p).bits
  return am


def _expect_raises(fn):
  try:
    fn()
  except (ValueError, TypeError) as exc:
    return type(exc).__name__
  return None


def model_checks() -> list[tuple[str, bool, str]]:
  results: list[tuple[str, bool, str]] = []

  def check(name, fn):
    try:
      ok, detail = fn()
    except Exception as exc:  # a raising check is a failed check
      results.append((name, False, f"unexpected {type(exc).__name__}: {exc}"))
    else:
      results.append((name, bool(ok), detail))

  def mc_grouping():
    r4 = matmul_reference(1, 1, _directed_steps(), tile_k=4, epilogue=Epilogue())
    r8 = matmul_reference(1, 1, _directed_steps(), tile_k=8, epilogue=Epilogue())
    ok = r4.am_bits == (_F32_ONE,) and r8.am_bits == (_F32_PLUS_ZERO,) and r4.am_bits != r8.am_bits
    return ok, f"tile4={_h32(r4.am_bits[0])} tile8={_h32(r8.am_bits[0])}"

  def mc_binary64_contrast():
    total = Fraction(0)
    for word in _DIRECT_B:
      total += scalar.finite_fraction(scalar.bf16_to_f32(word)) * scalar.finite_fraction(_F32_ONE)
    baseline = scalar.from_fraction(total)
    r8 = matmul_reference(1, 1, _directed_steps(), tile_k=8, epilogue=Epilogue())
    ok = (
      total == 1
      and baseline.bits == _F32_ONE
      and r8.am_bits[0] == _F32_PLUS_ZERO
      and baseline.bits != r8.am_bits[0]
    )
    return ok, f"exact-sum={total} baseline={_h32(baseline.bits)} per-mac={_h32(r8.am_bits[0])}"

  def mc_invariant_control():
    steps = (
      MatmulStep(
        k=3,
        a=(_BF16_ONE, _BF16_ONE, _BF16_ONE),
        b=(_BF16_ONE, _BF16_TWO, 0x4040),
        init=True,
        final=True,
        seq=0,
      ),
    )
    r2 = matmul_reference(1, 1, steps, tile_k=2, epilogue=Epilogue())
    r8 = matmul_reference(1, 1, steps, tile_k=8, epilogue=Epilogue())
    ok = r2.am_bits == r8.am_bits == (_F32_SIX,)
    return ok, f"tile2={_h32(r2.am_bits[0])} tile8={_h32(r8.am_bits[0])}"

  def mc_poison_overwrite():
    steps = (
      MatmulStep(k=2, a=(0x4000, 0x4040), b=(0x4080, 0x40A0), init=True, final=False, seq=0),
      MatmulStep(k=2, a=(0x4000, 0x4040), b=(0x4080, 0x40A0), init=False, final=True, seq=1),
    )
    r = matmul_reference(1, 1, steps, tile_k=8, epilogue=Epilogue())
    init_expected = scalar.from_fraction(23).bits
    final_expected = scalar.from_fraction(46).bits
    ok = (
      r.am_bits == (final_expected,)
      and r.blocks[0]["am_bits"] == _h32(init_expected)
      and r.initialization_count == 1
      and r.am_add_count == 1
      and not scalar.is_nan(final_expected)
    )
    return ok, "init_block={} final={} poison-start-not-read".format(
      r.blocks[0]["am_bits"], _h32(r.am_bits[0])
    )

  def mc_add_k0_negzero():
    steps = (
      MatmulStep(k=1, a=(0x8001,), b=(0x0001,), init=True, final=False, seq=0),
      MatmulStep(k=0, a=(), b=(), init=False, final=True, seq=1),
    )
    r = matmul_reference(1, 1, steps, tile_k=8, epilogue=Epilogue())
    ok = r.am_bits == (0x80000000,)
    return ok, f"am={_h32(r.am_bits[0])} (-0 preserved through k0 ADD)"

  def mc_init_k0_bias_scale():
    steps = (MatmulStep(k=0, a=(), b=(), init=True, final=True, seq=0),)
    r = matmul_reference(1, 1, steps, tile_k=8, epilogue=Epilogue(bias=(_F32_TWO,), scale=_F32_THREE))
    ok = r.output_bits == (_F32_SIX,)
    return ok, f"output={_h32(r.output_bits[0])} (6.0 proves INIT +0 then bias then scale)"

  def mc_tail_bypass():
    a = (0x3F80, 0x3F80, 0x0000, 0x0000)
    b = (0x3F80, 0x3F80, 0x7F80, 0x8000)
    for mask in (None, (True, True, False, False)):
      step = MatmulStep(k=2, a=a, b=b, init=True, final=True, seq=0, storage_k=4, mask=mask)
      r = matmul_reference(1, 1, (step,), tile_k=8, epilogue=Epilogue())
      manual = _manual_am(b, [(0, 1)])
      if r.am_bits != (manual,) or manual != 0x40000000:
        return False, f"mask={mask!r} am={_h32(r.am_bits[0])} manual={_h32(manual)}"
    return True, "both unmasked and masked tail bypass give 2.0"

  def mc_mask_boundaries():
    b = (_DIRECT_B[0], _BF16_NAN, _BF16_ONE, _BF16_ONE)
    step = MatmulStep(
      k=4, a=(_BF16_ONE,) * 4, b=b, init=True, final=True, seq=0, mask=(True, False, True, True)
    )
    r = matmul_reference(1, 1, (step,), tile_k=2, epilogue=Epilogue())
    manual = _manual_am(b, [(0,), (2, 3)])
    ok = r.am_bits == (manual,) == (0x4B800001,)
    return ok, f"am={_h32(r.am_bits[0])} manual={_h32(manual)} (dense compaction would give 0x4b800000)"

  def mc_empty_work():
    steps_m0 = (MatmulStep(k=3, a=(), b=(0x3F80,) * 6, init=True, final=True, seq=0),)
    r_m0 = matmul_reference(0, 2, steps_m0, tile_k=8, epilogue=Epilogue())
    steps_n0 = (MatmulStep(k=3, a=(0x3F80,) * 6, b=(), init=True, final=True, seq=0),)
    r_n0 = matmul_reference(2, 0, steps_n0, tile_k=8, epilogue=Epilogue())
    for r in (r_m0, r_n0):
      if (r.mac_count, r.am_add_count, r.initialization_count) != (0, 0, 0):
        return False, "nonzero counts for empty work"
      if r.am_bits or r.blocks or r.output_bytes or r.output_bits:
        return False, "nonempty output for empty work"
    return True, "m=0 and n=0: no MAC, no blocks, no output, still validated"

  def mc_rne_tie_up():
    steps = (MatmulStep(k=3, a=(_BF16_ONE,) * 3, b=(0x4B80, 0x4000, 0x3F80), init=True, final=True, seq=0),)
    r = matmul_reference(1, 1, steps, tile_k=8, epilogue=Epilogue())
    ok = r.am_bits == (0x4B800002,)
    return ok, f"2^24+3 ties to even -> {_h32(r.am_bits[0])}"

  def mc_matmul_illegal():
    cases = _matmul_illegal_cases()
    bad = []
    for case in cases:
      if "error" not in case["expected"]:
        bad.append(case["id"])
    ok = not bad
    return ok, f"{len(cases)} rejection vectors, failures={bad!r}"

  def mc_stage_order_bias_scale():
    am = (_F32_ONE,)
    scaled = apply_epilogue(am, 1, 1, Epilogue(bias=(_F32_TWO,), scale=_F32_THREE))
    ok = scaled.output_bits == (0x41100000,)
    return ok, f"(1+2)*3 = {_h32(scaled.output_bits[0])} (9.0)"

  def mc_stage_order_relu_clip():
    spec = Epilogue(bias=(0x41200000,), relu=True, clip=(0x00000000, 0x40400000))
    out = apply_epilogue((0xC0A00000,), 1, 1, spec)
    ok = out.output_bits == (_F32_THREE,)
    return ok, f"(-5+10)->relu 5->clip3 = {_h32(out.output_bits[0])} (3.0)"

  def mc_scale_one_executes():
    out = apply_epilogue((0x7F800001,), 1, 1, Epilogue(scale=_F32_ONE))
    stage = out.trace[0]["stages"][1]
    ok = out.output_bits == (_F32_NAN,) and stage["stage"] == "scale" and stage["flags"] & 0x10
    return ok, f"sNaN x 1.0 -> {_h32(out.output_bits[0])} with NV"

  def mc_relu_negzero():
    out = apply_epilogue((0x80000000, 0xFF800000), 1, 2, Epilogue(relu=True))
    ok = out.output_bits == (_F32_PLUS_ZERO, _F32_PLUS_ZERO)
    return ok, "-0 and -Inf -> +0"

  def mc_clip_equality():
    out = apply_epilogue((0x80000000,), 1, 1, Epilogue(clip=(_F32_PLUS_ZERO, _F32_ONE)))
    ok = out.output_bits == (0x80000000,)
    return ok, "clip lo==+0 keeps -0 bits"

  def mc_bf16_ties():
    out = apply_epilogue(
      (0x3F808000, 0x3F818000, 0x3F807FFF, 0x3F808001), 1, 4, Epilogue(output_dtype="bf16")
    )
    ok = out.output_bits == (0x3F80, 0x3F82, 0x3F80, 0x3F81)
    return ok, f"ties to even: {[hex(v) for v in out.output_bits]}"

  def mc_i8_ties():
    out = apply_epilogue(
      (0x3F000000, 0x3FC00000, 0x40200000, 0x40600000, 0xBF000000, 0xBFC00000),
      1,
      6,
      Epilogue(output_dtype="i8", inv_scale=_F32_ONE),
    )
    ok = out.output_bits == (0, 2, 2, 4, 0, 254)
    return ok, f"0.5,1.5,2.5,3.5,-0.5,-1.5 -> {out.output_bits!r}"

  def mc_i8_specials():
    out = apply_epilogue(
      (_F32_NAN, 0x7F800001, _F32_INF, 0xFF800000, 0x7F7FFFFF, 0xFF7FFFFF),
      1,
      6,
      Epilogue(output_dtype="i8", inv_scale=_F32_ONE, zero_point=-1),
    )
    ok = out.output_bits == (0xFF, 0xFF, 0x7F, 0x80, 0x7F, 0x80)
    return ok, "NaN/zp=-1 -> 0xff, +/-Inf and saturating extremes -> 0x7f/0x80"

  def mc_epilogue_illegal():
    cases = _epilogue_illegal_cases()
    bad = [c["id"] for c in cases if "error" not in c["expected"]]
    validate_reject = _expect_raises(lambda: validate_epilogue(Epilogue(bias=(_F32_ONE,)), 0))
    ok = not bad and validate_reject == "ValueError"
    return ok, f"{len(cases)} rejection vectors, failures={bad!r}, n=0 validation first: {validate_reject}"

  def mc_conv_ones():
    x9 = (0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0, 0x40E0, 0x4100, 0x4110)
    r = conv2d_reference(x9, (0x3F80,) * 4, (1, 1, 3, 3), (1, 1, 2, 2), tile_k=8, epilogue=Epilogue())
    expected = [f"0x{v:08x}" for v in (0x41400000, 0x41800000, 0x41C00000, 0x41E00000)]
    ok = (
      r["c_bits"] == expected
      and r["out_shape"] == [1, 1, 2, 2]
      and r["mac_count"] == 16
      and r["initialization_count"] == 4
      and r["am_add_count"] == 0
    )
    return ok, "C={} mac=16".format(r["c_bits"])

  def mc_conv_asymmetric():
    x_words = tuple(
      _BF16_INT1_29[(n_i * 40 + c_i * 20 + h_i * 5 + w_i) % 29]
      for n_i in range(2)
      for c_i in range(2)
      for h_i in range(4)
      for w_i in range(5)
    )
    w_pool = (0x3F80, 0x3F00, 0xBF80, 0x4000, 0x3E80, 0xBF00)
    weights = tuple(w_pool[i % len(w_pool)] for i in range(36))
    r = conv2d_reference(
      x_words,
      weights,
      (2, 2, 4, 5),
      (3, 2, 2, 3),
      stride=(2, 1),
      padding=(1, 2, 0, 3),
      dilation=(2, 1),
      tile_k=8,
      epilogue=Epilogue(),
    )
    ok = (
      r["out_shape"] == [2, 3, 3, 6]
      and r["m"] == 36
      and r["k"] == 12
      and len(r["c_bits"]) == 36 * 3
      and len(r["a_bits"]) == 36 * 12
      and r["mac_count"] == 36 * 3 * 12
      and r["traffic"]["valid_patch_elements"] == r["traffic_direct"]["valid_patch_elements"]
      and r["traffic"]["padding_zero_elements"]
      == r["traffic"]["expanded_elements"] - r["traffic"]["valid_patch_elements"]
    )
    return ok, (
      f"out={r['out_shape']} m={r['m']} mac={r['mac_count']} "
      f"valid={r['traffic']['valid_patch_elements']} pad={r['traffic']['padding_zero_elements']}"
    )

  def mc_conv_empty():
    r = conv2d_reference(
      (0x3F80, 0x4000, 0x4080, 0x40A0), (0x3F80,) * 9, (1, 1, 2, 2), (1, 1, 3, 3), tile_k=8, epilogue=None
    )
    ok = (
      r["out_shape"] == [1, 1, 0, 0]
      and r["m"] == 0
      and r["mac_count"] == 0
      and r["c_bits"] == []
      and r["output_hex"] == ""
      and r["traffic"]["expanded_elements"] == 0
      and r["traffic"]["valid_patch_elements"] == 0
    )
    return ok, "OH=OW=0: no MAC/output; expanded and valid patch element counts are zero"

  def mc_conv_padding_macs():
    r = conv2d_reference(
      (0x3F80, 0x4000, 0x4080, 0x40A0),
      (0x7F80, 0x3F80, 0x3F80, 0x3F80),
      (1, 1, 2, 2),
      (1, 1, 2, 2),
      padding=(1, 0, 1, 0),
      tile_k=8,
      epilogue=Epilogue(),
    )
    c_words = [int(v, 16) for v in r["c_bits"]]
    nan_present = any(scalar.is_nan(v) for v in c_words)
    ok = (
      r["out_shape"] == [1, 1, 2, 2]
      and nan_present
      and r["mac_count"] == 16
      and any(scalar.is_inf(v) for v in c_words)
    )
    return ok, "c={} (NaN only possible from pad +0 x Inf)".format(r["c_bits"])

  def mc_conv_tile_grouping():
    x = (_BF16_ONE,) * 8
    r4 = conv2d_reference(x, _DIRECT_B, (1, 1, 1, 8), (1, 1, 1, 8), tile_k=4, epilogue=Epilogue())
    r8 = conv2d_reference(x, _DIRECT_B, (1, 1, 1, 8), (1, 1, 1, 8), tile_k=8, epilogue=Epilogue())
    ok = (
      r4["c_bits"] == [_h32(_F32_ONE)]
      and r8["c_bits"] == [_h32(_F32_PLUS_ZERO)]
      and r4["c_bits"] != r8["c_bits"]
    )
    return ok, "conv tile4={} tile8={}".format(r4["c_bits"][0], r8["c_bits"][0])

  def mc_conv_k_order():
    x_pow = (0x3F80, 0x4000, 0x4080, 0x4100, 0x4180, 0x4200, 0x4280, 0x4300)
    w_pow = tuple((127 + idx) << 7 for idx in range(16))
    r = conv2d_reference(x_pow, w_pow, (1, 2, 2, 2), (2, 2, 2, 2), tile_k=8, epilogue=Epilogue())
    ok = len(r["b_bits"]) == 16
    for o in range(2):
      for k_index in range(8):
        c_i, rem = divmod(k_index, 4)
        kh_i, kw_i = divmod(rem, 2)
        src = w_pow[((o * 2 + c_i) * 2 + kh_i) * 2 + kw_i]
        if int(r["b_bits"][k_index * 2 + o], 16) != src:
          ok = False
    return ok, "B[k,o] == weights[o,ic,kh,kw] for all 16 entries" if ok else "B layout mismatch"

  def mc_generators_deterministic():
    seed = _DEFAULT_SEED
    ok = (
      generate_matmul_cases(seed) == generate_matmul_cases(seed)
      and generate_epilogue_cases(seed) == generate_epilogue_cases(seed)
      and generate_conv_cases(seed) == generate_conv_cases(seed)
    )
    return ok, f"three generators reproduce byte-identical cases at seed {seed}"

  check("matmul/tile-grouping-directed", mc_grouping)
  check("matmul/binary64-contrast", mc_binary64_contrast)
  check("matmul/grouping-invariant-control", mc_invariant_control)
  check("matmul/init-overwrites-poison", mc_poison_overwrite)
  check("matmul/add-k0-preserves-negzero", mc_add_k0_negzero)
  check("matmul/init-k0-bias-scale-six", mc_init_k0_bias_scale)
  check("matmul/tail-bypass-independent", mc_tail_bypass)
  check("matmul/mask-keeps-block-boundaries", mc_mask_boundaries)
  check("matmul/empty-m-and-n", mc_empty_work)
  check("matmul/rne-tie-to-even", mc_rne_tie_up)
  check("matmul/illegal-params-rejected", mc_matmul_illegal)
  check("epilogue/order-bias-then-scale", mc_stage_order_bias_scale)
  check("epilogue/order-relu-then-clip", mc_stage_order_relu_clip)
  check("epilogue/scale-one-executes", mc_scale_one_executes)
  check("epilogue/relu-negzero-inf", mc_relu_negzero)
  check("epilogue/clip-equality-preserves-bits", mc_clip_equality)
  check("epilogue/bf16-tie-to-even", mc_bf16_ties)
  check("epilogue/i8-tie-to-even", mc_i8_ties)
  check("epilogue/i8-nan-inf-saturation", mc_i8_specials)
  check("epilogue/illegal-params-rejected", mc_epilogue_illegal)
  check("conv/ones-2x2-known-answer", mc_conv_ones)
  check("conv/asymmetric-geometry-traffic", mc_conv_asymmetric)
  check("conv/empty-output-zero-work", mc_conv_empty)
  check("conv/padding-zero-operands-mac", mc_conv_padding_macs)
  check("conv/tile-grouping-matches-profile", mc_conv_tile_grouping)
  check("conv/b-layout-declared-k-order", mc_conv_k_order)
  check("generators/deterministic-per-seed", mc_generators_deterministic)
  return results
