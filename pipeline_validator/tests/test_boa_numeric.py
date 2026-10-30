"""Golden boundary tests for the bf16_fp32_mac_rne_v1 numeric core.

Consumes the frozen P03 vector JSON (``hardware/boa/reference/vectors``) and
the permanent directed golden fixture
(``pipeline_validator/tests/data/numeric_profile_directed_golden.json``) as
black-box expectations.  The reference oracle is neither imported nor copied:
every comparison is raw bits and diagnostic flags against JSON goldens, plus
a deliberate binary64 counterexample pinning the removal of the old
whole-step accumulation.  Production-path (parse/compile/load/simulate)
vector parity lives in ``tests/test_boa_numeric_profile.py``.

pytest:  conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_boa_numeric.py -q
smoke:   conda run -n elenor-validator python -m pipeline_validator.tests.test_boa_numeric
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from pipeline_validator.boa_numeric import (
  add_f32_rne,
  convert_output,
  decode_f32_bits,
  encode_f32_bits,
  f32_to_bf16_rne,
  fma_f32_rne,
  matmul_f32,
)

_REPO = Path(__file__).resolve().parents[2]
_VECTORS = _REPO / "hardware" / "boa" / "reference" / "vectors"
_DIRECTED = Path(__file__).resolve().parent / "data" / "numeric_profile_directed_golden.json"

_FLAGS_ORDER = ("fma_flags", "add_flags", "flags")


def _golden(path: Path) -> dict:
  return json.loads(path.read_text())


def _h(hex_string: str) -> int:
  return int(hex_string, 16)


def _bf16_bytes(words) -> bytes:
  return b"".join(_h(word).to_bytes(2, "little") for word in words)


def _run_case(case: dict):
  """Execute every step of a golden matmul case, returning (am, trace, spans)."""
  inputs = case["input"]
  m = inputs["m"]
  n = inputs["n"]
  tile_k = inputs.get("tile_k", 8)
  am: bytes | None = None
  trace: list = []
  spans: list[tuple[int, int]] = []
  for step in inputs["steps"]:
    begin = len(trace)
    am = matmul_f32(
      _bf16_bytes(step["a"]),
      _bf16_bytes(step["b"]),
      m=m,
      n=n,
      k=step["k"],
      accumulate=not step["init"],
      partial=am,
      tile_k=tile_k,
      storage_k=step.get("storage_k"),
      mask=step.get("mask"),
      block_trace=trace,
    )
    spans.append((begin, len(trace)))
  assert am is not None, f"{case['id']} produced no AM bytes"
  return am, trace, spans


def _assert_case_bytes(case: dict, am: bytes) -> str:
  """Compare final bytes against golden; returns 'full' or 'epilogue-deferred'."""
  expected = case["expected"]
  am_words = decode_f32_bits(am)
  assert am_words == [_h(word) for word in expected["am_bits"]], case["id"]
  epilogue = case["input"].get("epilogue")
  dtype = (epilogue or {}).get("output_dtype", "f32")
  staged = bool(epilogue) and (
    epilogue.get("bias") is not None
    or epilogue.get("scale") is not None
    or bool(epilogue.get("relu"))
    or epilogue.get("clip") is not None
  )
  if staged or dtype == "i8":
    # P08 owns epilogue stages and INT8 conversion; only the frozen
    # pre-epilogue f32 AM bits are asserted here.
    return "epilogue-deferred"
  if dtype == "bf16":
    assert convert_output(am_words, "bf16") == bytes.fromhex(expected["output_hex"]), case["id"]
  else:
    assert am == bytes.fromhex(expected["output_hex"]), case["id"]
  return "full"


def _assert_case_blocks(case: dict, trace: list, spans: list[tuple[int, int]]) -> None:
  expected_blocks = case["expected"]["blocks"]
  assert len(trace) == len(expected_blocks), case["id"]
  for index, (got, want) in enumerate(zip(trace, expected_blocks)):
    seq = want["seq"]
    begin, end = spans[seq]
    assert begin <= index < end, (case["id"], index, seq)
    assert got["kind"] == want["kind"], (case["id"], index)
    assert (got["i"], got["j"], got["start_k"]) == (want["i"], want["j"], want["start_k"])
    assert got["valid_indices"] == want["valid_indices"], (case["id"], index)
    assert got["partial_bits"] == [_h(word) for word in want["partial_bits"]], (case["id"], index)
    assert got["am_bits"] == _h(want["am_bits"]), (case["id"], index)
    for flag in _FLAGS_ORDER:
      assert got[flag] == want[flag], (case["id"], index, flag)
    assert got["overwrite"] == want["overwrite"], (case["id"], index)


def test_fma_primitive_golden() -> None:
  data = _golden(_VECTORS / "primitives.json")
  cases = [case for case in data["cases"] if case["op"] == "fma"]
  assert cases, "fma golden subset missing"
  for case in cases:
    got = fma_f32_rne(_h(case["a"]), _h(case["b"]), _h(case["c"]))
    assert got == (_h(case["expected"]["bits"]), _h(case["expected"]["flags"])), case["id"]


def test_add_primitive_golden() -> None:
  data = _golden(_VECTORS / "primitives.json")
  cases = [case for case in data["cases"] if case["op"] == "add"]
  assert cases, "add golden subset missing"
  for case in cases:
    got = add_f32_rne(_h(case["a"]), _h(case["b"]))
    assert got == (_h(case["expected"]["bits"]), _h(case["expected"]["flags"])), case["id"]


def test_bf16_conversion_golden() -> None:
  data = _golden(_VECTORS / "primitives.json")
  cases = [case for case in data["cases"] if case["op"] == "bf16"]
  assert len(cases) > 100, "bf16 conversion golden subset missing"
  for case in cases:
    got = f32_to_bf16_rne(_h(case["a"]))
    assert got == (_h(case["expected"]["bits"]), _h(case["expected"]["flags"])), case["id"]




def test_freeze_tininess_after_pins() -> None:
  # numeric_freeze.md section 2.1 boundary rows (BF16 conversion side).
  assert f32_to_bf16_rne(0x007F8000) == (0x0080, 0x03)
  assert f32_to_bf16_rne(0x007FC000) == (0x0080, 0x01)
  # A tiny negative value rounding to zero keeps -0 (magnitudes below half
  # of the smallest BF16 subnormal tie to zero with the sign preserved).
  assert f32_to_bf16_rne(0x00000001) == (0x0000, 0x03)
  assert f32_to_bf16_rne(0x80000001) == (0x8000, 0x03)


def test_zero_and_nan_flag_pins() -> None:
  plus_zero, minus_zero = 0x00000000, 0x80000000
  # Signed-zero rules.
  assert add_f32_rne(minus_zero, minus_zero) == (minus_zero, 0)
  assert add_f32_rne(minus_zero, plus_zero) == (plus_zero, 0)
  assert add_f32_rne(_h("0x3f800000"), _h("0xbf800000")) == (plus_zero, 0)
  # FMA signed zeros need an actual zero PRODUCT: -0 x 1 keeps the product
  # sign, which survives only when the zero accumulator carries the same
  # sign; opposite signs collapse to +0.
  neg_one = _h("0xbf800000")
  assert fma_f32_rne(minus_zero, _h("0x3f800000"), minus_zero) == (minus_zero, 0)
  assert fma_f32_rne(minus_zero, _h("0x3f800000"), plus_zero) == (plus_zero, 0)
  assert fma_f32_rne(plus_zero, _h("0x3f800000"), minus_zero) == (plus_zero, 0)
  # Exact nonzero cancellation (-2 x 0.5 + 1) yields +0.
  assert fma_f32_rne(_h("0xc0000000"), _h("0x3f000000"), _h("0x3f800000")) == (plus_zero, 0)
  # A negative finite product is negative: -1 x 1 + 0 = -1.
  assert fma_f32_rne(neg_one, _h("0x3f800000"), plus_zero) == (neg_one, 0)
  # NaN outputs canonicalize; NV comes only from sNaN inputs or 0 x Inf.
  qnan, snan = 0x7FC00001, 0x7F800001
  assert add_f32_rne(qnan, _h("0x3f800000")) == (0x7FC00000, 0)
  assert add_f32_rne(snan, plus_zero) == (0x7FC00000, 0x10)
  assert fma_f32_rne(plus_zero, _h("0x7f800000"), qnan) == (0x7FC00000, 0x10)
  assert fma_f32_rne(_h("0x7f800000"), _h("0x3f800000"), _h("0xbf800000")) == (
    0x7F800000,
    0,
  )
  assert add_f32_rne(_h("0x7f800000"), _h("0xff800000")) == (0x7FC00000, 0x10)
  # Exact opposite-sign infinity addition is invalid; same-sign passes.
  assert add_f32_rne(_h("0x7f800000"), _h("0x7f800000")) == (0x7F800000, 0)


def test_fma_differs_from_removed_binary64_path() -> None:
  # Frozen counterexample (numeric_freeze.md section 3): a single-rounding
  # fma yields 0x3f801001, while the removed binary64 whole-step path
  # double-rounded to 0x3f801000.  The legacy value is recomputed with the
  # old float algorithm on purpose; it must never match the profile result.
  a = b = 0x3F800800
  c = 0x0D800000
  bits, flags = fma_f32_rne(a, b, c)
  assert (bits, flags) == (0x3F801001, 0x01)
  product = struct.unpack("<f", struct.pack("<I", a))[0] * struct.unpack(
    "<f", struct.pack("<I", b)
  )[0]
  acc = struct.unpack("<f", struct.pack("<I", c))[0] + product
  legacy_bits = struct.unpack("<I", struct.pack("<f", acc))[0]
  assert legacy_bits == 0x3F801000
  assert bits != legacy_bits


def test_matmul_golden_vectors() -> None:
  data = _golden(_VECTORS / "matmul.json")
  cases = [
    case
    for case in data["cases"]
    if case.get("input") and "error" not in case.get("expected", {})
  ]
  assert cases, "matmul golden subset missing"
  full = 0
  for case in cases:
    am, trace, spans = _run_case(case)
    if _assert_case_bytes(case, am) == "full":
      full += 1
    _assert_case_blocks(case, trace, spans)
    expected = case["expected"]
    mac_count = sum(len(record["partial_bits"]) for record in trace)
    add_count = sum(1 for record in trace if record["kind"] == "contrib" and not record["overwrite"])
    init_count = sum(1 for record in trace if record["overwrite"])
    assert mac_count == expected["mac_count"], case["id"]
    assert add_count == expected["am_add_count"], case["id"]
    assert init_count == expected["initialization_count"], case["id"]
  assert full >= 1, "no golden case compared whole output bytes"


# Illegal golden cases rejected directly by the scalar numeric core.
_MATMUL_REJECT_VALUE_ERROR = {
  "mm-illegal-tile-k-zero",
  "mm-illegal-storage-below-k",
  "mm-illegal-a-length",
  "mm-illegal-b-length",
  "mm-illegal-mask-length",
  "mm-illegal-mask-tail-true",
  "mm-illegal-mask-not-bool",
  "mm-illegal-negative-m",
  "mm-illegal-negative-k",
}
# Chain structure is exercised through the IR/runtime gate. BF16 word-range
# failures happen at vector ingestion, before a byte-oriented numerical API.


@pytest.mark.parametrize("case_id", sorted(_MATMUL_REJECT_VALUE_ERROR))
def test_matmul_illegal_golden_rejections(case_id: str) -> None:
  data = _golden(_VECTORS / "matmul.json")
  case = next(case for case in data["cases"] if case["id"] == case_id)
  inputs = case["input"]
  step = inputs["steps"][0]
  with pytest.raises(ValueError):
    matmul_f32(
      _bf16_bytes(step["a"]),
      _bf16_bytes(step["b"]),
      m=inputs["m"],
      n=inputs["n"],
      k=step["k"],
      accumulate=not step["init"],
      partial=None,
      tile_k=inputs.get("tile_k", 8),
      storage_k=step.get("storage_k"),
      mask=step.get("mask"),
    )


def test_directed_golden_vector_parity() -> None:
  data = _golden(_DIRECTED)
  assert data["profile"] == "bf16_fp32_mac_rne_v1"
  by_id = {case["id"]: case for case in data["cases"]}
  assert {"p06-grouping-k8", "p06-grouping-k16", "p06-k64-eight-blocks"} <= set(by_id)
  results = {}
  for case in data["cases"]:
    am, trace, spans = _run_case(case)
    assert _assert_case_bytes(case, am) == "full", case["id"]
    _assert_case_blocks(case, trace, spans)
    results[case["id"]] = am
  # Grouping counterexample: the same operands must produce different bits
  # under tile_k 8 vs 16, and each must match its own golden.
  assert results["p06-grouping-k8"] != results["p06-grouping-k16"]
  # k=64 chain of eight blocks: INIT overwrites exactly the first block.
  case = by_id["p06-k64-eight-blocks"]
  overwrites = [record["start_k"] for record in case["expected"]["blocks"] if record["overwrite"]]
  assert overwrites == [0]


def test_empty_step_semantics() -> None:
  # INIT with no active lanes initializes every cell to +0.
  assert matmul_f32(b"", b"", m=2, n=2, k=0, accumulate=False, partial=None) == bytes(16)
  # ADD with no active lanes preserves the partial bytes bit-exactly,
  # including -0, sNaN payloads and infinities.
  partial = encode_f32_bits([0x80000000, 0xFF800001, 0x7F800000, 0x00000000])
  assert matmul_f32(b"", b"", m=2, n=2, k=0, accumulate=True, partial=partial) == partial
  # Zero output dimensions still run validation and produce empty bytes.
  assert matmul_f32(b"", b"\x00\x00", m=0, n=1, k=1, accumulate=False, partial=None) == b""
  assert matmul_f32(bytes(6), b"", m=1, n=0, k=3, accumulate=True, partial=b"") == b""
  # INIT without partial works; ADD without partial is rejected.
  with pytest.raises(ValueError):
    matmul_f32(b"", b"", m=1, n=1, k=0, accumulate=True, partial=None)
  with pytest.raises(ValueError):
    matmul_f32(b"", b"", m=1, n=1, k=0, accumulate=True, partial=b"\x00\x00")


def test_storage_mask_and_validation_rules() -> None:
  a = _bf16_bytes(["0x3f80"] * 6)
  b = _bf16_bytes(["0x3f80"] * 6)
  # Storage tail beyond k must be inactive.
  tail_out = matmul_f32(
    a, b, m=1, n=1, k=4, accumulate=False, partial=None, storage_k=6, mask=[True] * 4 + [False] * 2
  )
  assert tail_out == encode_f32_bits([_h("0x40800000")])  # 4 exact unit MACs
  with pytest.raises(ValueError):
    matmul_f32(a, b, m=1, n=1, k=4, accumulate=False, partial=None, storage_k=6, mask=[True] * 5 + [False])
  with pytest.raises(ValueError):
    matmul_f32(a, b, m=1, n=1, k=4, accumulate=False, partial=None, storage_k=6, mask=[True] * 5)
  with pytest.raises(ValueError):
    matmul_f32(a, b, m=1, n=1, k=4, accumulate=False, partial=None, storage_k=6, mask=[1, 2, 0, 0, 0, 0])
  with pytest.raises(ValueError):
    matmul_f32(a, b, m=1, n=1, k=4, accumulate=False, partial=None, storage_k=6, mask=[1, 0, 0, 0, 0, 0])
  with pytest.raises(ValueError):
    matmul_f32(a, b, m=1, n=1, k=4, accumulate=False, partial=None, storage_k=3)
  with pytest.raises(ValueError):
    matmul_f32(a[:10], b, m=1, n=1, k=4, accumulate=False, partial=None, storage_k=6)
  with pytest.raises(ValueError):
    matmul_f32(a, b, m=1, n=1, k=4, accumulate=False, partial=None, tile_k=0)
  with pytest.raises(ValueError):
    matmul_f32(a, b, m=-1, n=1, k=4, accumulate=False, partial=None)
  # Inactive lanes execute no MAC: a masked-out NaN operand never poisons.
  nan_b = _bf16_bytes(["0x7fc0", "0x7fc0", "0x7fc0", "0x3f80"])
  out = matmul_f32(
    _bf16_bytes(["0x3f80"] * 4),
    nan_b,
    m=1,
    n=1,
    k=4,
    accumulate=False,
    partial=None,
    mask=[False, False, False, True],
    block_trace=[],
  )
  assert out == encode_f32_bits([_h("0x3f800000")])
  trace_records: list[dict] = []
  matmul_f32(
    _bf16_bytes(["0x3f80"] * 4),
    nan_b,
    m=1,
    n=1,
    k=4,
    accumulate=False,
    partial=None,
    mask=[False, False, False, True],
    block_trace=trace_records,
  )
  assert [(r["start_k"], r["valid_indices"]) for r in trace_records] == [(0, [3])]
  assert trace_records[0]["overwrite"] is True
  assert trace_records[0]["partial_bits"] == [_h("0x3f800000")]


def test_block_trace_int_fields() -> None:
  trace: list = []
  matmul_f32(
    _bf16_bytes(["0x3f80"] * 3),
    _bf16_bytes(["0x4000"] * 3),
    m=1,
    n=1,
    k=3,
    accumulate=False,
    partial=None,
    tile_k=2,
    block_trace=trace,
  )
  assert [record["start_k"] for record in trace] == [0, 2]
  assert trace[0]["overwrite"] is True and trace[1]["overwrite"] is False
  assert trace[0]["add_flags"] == 0 and trace[1]["add_flags"] == 0
  assert trace[1]["flags"] == trace[1]["fma_flags"] == 0
  # 1.0 * 2.0 accumulated: block0 partials [+0->2.0, 2.0->4.0] overwrite AM
  # with 4.0; block1 partial 2.0 adds to 6.0.  All exact, no flags.
  assert trace[0]["partial_bits"] == [_h("0x40000000"), _h("0x40800000")]
  assert trace[1]["partial_bits"] == [_h("0x40000000")]
  assert [record["am_bits"] for record in trace] == [
    _h("0x40800000"),
    _h("0x40c00000"),
  ]


def test_bit_codec_and_convert_output() -> None:
  words = [0x7FC00001, 0xFF800000, 0x00000001, 0x80000000]
  assert decode_f32_bits(b"") == []
  assert encode_f32_bits([]) == b""
  data = encode_f32_bits(words)
  assert decode_f32_bits(data) == words  # raw payloads survive verbatim
  assert convert_output(words, "f32") == data
  converted = convert_output(words, "bf16")
  # Positive tiny -> +0; -0 retains its sign. Every word is little-endian.
  assert converted == bytes.fromhex("c07f80ff00000080")
  with pytest.raises(ValueError):
    convert_output(words, "i8")
  with pytest.raises(ValueError):
    convert_output(words, "f16")
  with pytest.raises(ValueError):
    decode_f32_bits(b"\x00\x00\x00")
  with pytest.raises(ValueError):
    encode_f32_bits([0x100000000])
  with pytest.raises(ValueError):
    encode_f32_bits([True])


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(pytest.main([__file__, "-q"]))
