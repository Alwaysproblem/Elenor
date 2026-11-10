"""C011 numeric-bootstrap suite.

Exercises the exact reference (`numeric_ref`) against hand-derived vectors and the committed
real-RTL record. A reference that drops the jam, uses RNE internally, applies the wrong mask
order, or collapses the multiplier/accumulate overflow asymmetry must FAIL here. Nothing compares
floats with a tolerance and nothing greps a constant.

This is a DUT contract check, not the independent oracle (C047).
"""

from __future__ import annotations

import copy
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

H = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(H / "scripts/lib"))

import numeric_probe as probe  # noqa: E402
import numeric_ref as nr  # noqa: E402
import validate_numeric as vn  # noqa: E402


def bf(sign: int, exp: int, man: int) -> int:
  return (sign << 15) | (exp << 7) | man


def f32(sign: int, exp: int, man: int) -> int:
  return (sign << 31) | (exp << 23) | man


ONE, TWO = bf(0, 127, 0), bf(0, 128, 0)
MAXN, INF, QNAN, SNAN = bf(0, 254, 127), bf(0, 255, 0), bf(0, 255, 64), bf(0, 255, 1)
Z = 0


@pytest.fixture(scope="module")
def vectors() -> dict[str, Any]:
  return vn.load_vectors()


@pytest.fixture(scope="module")
def profile() -> dict[str, Any]:
  return vn.load_profile()


def _by_id(vectors: dict[str, Any]) -> dict[str, dict[str, Any]]:
  return {v["id"]: v for v in vectors["vectors"]}


def _fresh_vectors() -> dict[str, Any]:
  return copy.deepcopy(vn.load_vectors())


def _fresh_profile() -> dict[str, Any]:
  return copy.deepcopy(vn.load_profile())


# --- the committed contract validates ----------------------------------------------------------


def test_committed_contract_and_real_rtl_record_validate() -> None:
  report = vn.validate()
  assert report["valid"], report["errors"]
  assert report["real_rtl_cells_compared"] == 4 * 256
  assert report["cvfpu_encodings_checked"], report["notes"]


def test_profile_stays_experimental(profile: dict[str, Any]) -> None:
  assert profile["support_status"] == "experimental"
  profile = copy.deepcopy(profile)
  profile["support_status"] = "supported"
  path_errors: list[str] = []
  if profile.get("support_status") != "experimental":
    path_errors.append("flagged")
  assert path_errors


@pytest.mark.parametrize("vid", [v["id"] for v in vn.load_vectors()["vectors"]])
def test_vector_passes_reference(vid: str, vectors: dict[str, Any]) -> None:
  assert vn.run_vector(_by_id(vectors)[vid]) == []


# --- coverage the card names: overflow, mask, cancellation, TK grouping -------------------------


def test_the_two_overflow_stages_are_asymmetric(vectors: dict[str, Any]) -> None:
  v = _by_id(vectors)
  assert v["mul_overflow_two_lane_sum"]["expected_fp32_bits"] == 0x7F800000  # multiplier: +Inf
  assert v["acc_overflow_to_max_normal"]["expected_fp32_bits"] == 0x7F7FFFFF  # accumulate: max-normal
  assert nr.vtfmm_cell(0, [(MAXN, ONE), (MAXN, ONE)], 2)[0] == 0x7F800000
  assert nr.vtfmm_cell(0x7F7FFFFF, [(MAXN, ONE), (Z, Z)], 1)[0] == 0x7F7FFFFF


def test_masked_inf_and_nan_do_not_take_part() -> None:
  masked_inf = nr.vtfmm_cell(0, [(ONE, TWO), (INF, Z)], 1)
  assert masked_inf[0] == f32(0, 128, 0) and (masked_inf[1].nv, masked_inf[2].nv) == (0, 0)
  assert nr.vtfmm_cell(0, [(ONE, TWO), (QNAN, ONE)], 1)[0] == f32(0, 128, 0)
  # the same pair UNMASKED (tk=2) is a real 0 x Inf and must raise NV
  live = nr.vtfmm_cell(0, [(ONE, TWO), (INF, Z)], 2)
  assert live[0] == nr.QNAN_FP32 and live[1].nv == 1


def test_masked_lane_is_zeroed_before_classification_not_after() -> None:
  lane = nr.mul_lane(INF, Z, enabled=False)
  assert not (lane.nan or lane.inf or lane.invalid), "a masked 0xInf lane must not classify as NaN"
  assert nr.mul_lane(INF, Z, enabled=True).invalid


def test_cancellation_gives_positive_zero_even_for_negative_operands() -> None:
  assert nr.vtfmm_cell(0, [(ONE, ONE), (bf(1, 127, 0), ONE)], 2)[0] == 0
  assert nr.vtfmm_cell(f32(1, 127, 0), [(ONE, ONE), (Z, Z)], 1)[0] == 0
  assert nr.vtfmm_cell(f32(1, 0, 0), [(bf(1, 0, 0), ONE), (Z, Z)], 1)[0] == 0  # not IEEE -0


def test_tk_grouping_changes_bits(vectors: dict[str, Any]) -> None:
  vec = _by_id(vectors)["tkg_grouping_differs"]
  assert vec["expected_fp32_bits"] != vec["expected_split_fp32_bits"]
  pairs = [tuple(p) for p in vec["pairs_bf16_bits"]]
  grouped = nr.vtfmm_cell(vec["old_fp32_bits"], pairs, 2)[0]
  mid = nr.vtfmm_cell(vec["old_fp32_bits"], [pairs[0], (0, 0)], 1)[0]
  split = nr.vtfmm_cell(mid, [pairs[1], (0, 0)], 1)[0]
  assert (grouped, split) == (vec["expected_fp32_bits"], vec["expected_split_fp32_bits"])


def test_tk1_masks_the_second_lane() -> None:
  assert nr.vtfmm_cell(0, [(ONE, TWO), (bf(0, 128, 64), bf(0, 128, 64))], 1)[0] == f32(0, 128, 0)
  assert nr.vtfmm_cell(0, [(ONE, TWO), (bf(0, 128, 64), bf(0, 128, 64))], 2)[0] != f32(0, 128, 0)
  with pytest.raises(nr.NumericError) as exc:
    nr.vtfmm_cell(0, [(ONE, ONE), (ONE, ONE)], 3)
  assert exc.value.code == "TK_INVALID"


# --- the contract is real: wrong implementations are caught -------------------------------------


def test_ablations_produce_a_different_result(vectors: dict[str, Any]) -> None:
  errors: list[str] = []
  assert vn.check_ablations(vectors, errors) == 2
  assert errors == []
  for vec in vectors["vectors"]:
    if vec["kind"] == "ablation":
      assert vn.ablated_cell(vec) != vec["expected_fp32_bits"], vec["id"]


def test_dropping_the_jam_is_detected_by_a_vector() -> None:
  """A reference that truncates instead of jamming must fail a known vector."""
  vec = _by_id(vn.load_vectors())["jam_large_exponent_gap"]
  pairs = [tuple(p) for p in vec["pairs_bf16_bits"]]
  lanes = [nr.mul_lane(a, b, True) for a, b in pairs]
  no_jam = nr.accumulate(vn._mulbulk_variant(lanes, jam=False, mode=nr.ROD), vec["old_fp32_bits"])[0]
  assert no_jam != vec["expected_fp32_bits"]


def test_internal_rne_is_detected_by_a_vector() -> None:
  vec = _by_id(vn.load_vectors())["rod_vs_rne"]
  rne = nr.round_decision(nr.RNE, 1, 1, 0, 0)
  rod = nr.round_decision(nr.ROD, 1, 1, 0, 0)
  assert (rne, rod) == (1, 0), "guard=1 round=1 sticky=0 is the case RNE and ROD disagree on"
  assert vec["expected_fp32_bits"] == f32(0, 127, 3) and vec["contrast_rne_fp32_bits"] == f32(0, 127, 4)


def test_jam_equals_sticky_under_rod_but_not_under_rne() -> None:
  """The measured property recorded in JAM-01: ROD cannot see the jam, RNE can."""
  import random

  rng = random.Random(8)
  rod_diff = rne_diff = 0
  for _ in range(4000):
    pairs = [
      (
        bf(rng.getrandbits(1), rng.randint(87, 167), rng.getrandbits(7)),
        bf(rng.getrandbits(1), rng.randint(87, 167), rng.getrandbits(7)),
      )
      for _ in range(2)
    ]
    lanes = [nr.mul_lane(a, b, True) for a, b in pairs]
    if any(lane.zero for lane in lanes):
      continue
    exact = sum((lane.product for lane in lanes), Fraction(0))
    if exact == 0:
      continue
    jam_rod = vn._mulbulk_variant(lanes, jam=True, mode=nr.ROD)
    ex_rod = nr.round_exact(exact, 23, 8, nr.ROD, widen_overflow_inf=True)[0]
    jam_rne = vn._mulbulk_variant(lanes, jam=True, mode=nr.RNE)
    ex_rne = nr.round_exact(exact, 23, 8, nr.RNE, widen_overflow_inf=True)[0]
    rod_diff += jam_rod != ex_rod
    rne_diff += jam_rne != ex_rne
  assert rod_diff == 0
  assert rne_diff > 0


def test_a_tampered_expected_bit_is_caught() -> None:
  vectors = _fresh_vectors()
  next(v for v in vectors["vectors"] if v["id"] == "rod_odd_sticky")["expected_fp32_bits"] ^= 1
  errors: list[str] = []
  vn.check_vectors(vectors, errors)
  assert any("rod_odd_sticky" in e for e in errors)


def test_a_tampered_flag_is_caught() -> None:
  vectors = _fresh_vectors()
  next(v for v in vectors["vectors"] if v["id"] == "spc_inf_times_zero")["expected_flags"]["nv"] = 0
  errors: list[str] = []
  vn.check_vectors(vectors, errors)
  assert any("spc_inf_times_zero" in e and "flags" in e for e in errors)


def test_a_one_ulp_close_answer_is_not_accepted() -> None:
  """No tolerance: off by one ulp fails, exactly like off by a lot."""
  vectors = _fresh_vectors()
  for v in vectors["vectors"]:
    if v["id"] == "mul_two_lane_sum":
      v["expected_fp32_bits"] += 1
  errors: list[str] = []
  vn.check_vectors(vectors, errors)
  assert errors


# --- the real-RTL record -------------------------------------------------------------------------


def test_model_matches_every_cell_of_the_real_rtl_record() -> None:
  errors: list[str] = []
  assert vn.check_record(errors) == 1024
  assert errors == []


def test_record_tamper_is_detected(tmp_path: Path) -> None:
  rec = json.loads(vn.RECORD.read_text())
  rec["cases"][1]["actual_u32"][17] ^= 1
  bad = tmp_path / "record.json"
  bad.write_text(json.dumps(rec))
  errors: list[str] = []
  vn.check_record(errors, bad)
  assert any("case 1 cell 17" in e for e in errors)


def test_committed_record_copy_matches_the_pin() -> None:
  import hashlib

  assert hashlib.sha256(vn.RECORD.read_bytes()).hexdigest() == vn.record_pin()


def test_tile_tail_cells_are_undisturbed_in_the_real_record() -> None:
  rec = json.loads(vn.RECORD.read_text())
  case = rec["cases"][3]
  assert (case["tm"], case["tn"]) == (9, 11)
  outside = [i for i in range(256) if i // 16 >= 9 or i % 16 >= 11]
  assert len(outside) == 256 - 99
  assert all(case["actual_u32"][i] == case["c_init_u32"][i] for i in outside)


def test_tk_in_real_record_changes_the_result() -> None:
  rec = json.loads(vn.RECORD.read_text())
  by_impl = {c["case"]: c for c in rec["cases"]}
  tk2, tk1 = by_impl[1], by_impl[2]
  assert tk2["a_bf16_bits"] == tk1["a_bf16_bits"] and tk2["c_init_u32"] == tk1["c_init_u32"]
  assert (tk2["tk"], tk1["tk"]) == (2, 1)
  assert tk2["actual_u32"] != tk1["actual_u32"]


# --- rounding primitives --------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("mode", "guard", "rnd", "sticky", "sign", "up"),
  [
    (nr.RNE, 0, 1, 0, 0, 0),  # tie, even stays
    (nr.RNE, 1, 1, 0, 0, 1),  # tie, odd rounds up
    (nr.RNE, 0, 1, 1, 0, 1),
    (nr.ROD, 0, 1, 0, 0, 1),
    (nr.ROD, 1, 1, 1, 0, 0),
    (nr.ROD, 0, 0, 0, 0, 0),
    (nr.RTZ, 1, 1, 1, 0, 0),
    (nr.RDN, 0, 1, 0, 1, 1),
    (nr.RDN, 0, 1, 0, 0, 0),
    (nr.RUP, 0, 1, 0, 0, 1),
    (nr.RUP, 0, 1, 0, 1, 0),
    (nr.RMM, 0, 1, 0, 0, 1),
  ],
)
def test_round_decision_table(mode: int, guard: int, rnd: int, sticky: int, sign: int, up: int) -> None:
  assert nr.round_decision(mode, guard, rnd, sticky, sign) == up


def test_overflow_saturation_per_mode() -> None:
  assert [nr.overflow_to_max_norm(m, 0) for m in (nr.RNE, nr.RMM, nr.RTZ, nr.ROD)] == [0, 0, 1, 1]
  assert (nr.overflow_to_max_norm(nr.RDN, 0), nr.overflow_to_max_norm(nr.RDN, 1)) == (1, 0)
  assert (nr.overflow_to_max_norm(nr.RUP, 0), nr.overflow_to_max_norm(nr.RUP, 1)) == (0, 1)
  with pytest.raises(nr.NumericError):
    nr.round_decision(0b110, 0, 0, 0, 0)


def test_round_exact_covers_subnormal_overflow_and_carry() -> None:
  # smallest FP32 subnormal rounds exactly
  assert nr.round_exact(Fraction(1, 2**149), 23, 8, nr.RNE, widen_overflow_inf=False)[0] == 1
  # half of it ties to even (0) under RNE and to odd (1) under ROD
  assert nr.round_exact(Fraction(1, 2**150), 23, 8, nr.RNE, widen_overflow_inf=False)[0] == 0
  assert nr.round_exact(Fraction(1, 2**150), 23, 8, nr.ROD, widen_overflow_inf=False)[0] == 1
  # mantissa carry into the next binade: 2 - 2^-24 rounds up to 2.0 under RNE
  assert nr.round_exact(2 - Fraction(1, 2**24), 23, 8, nr.RNE, widen_overflow_inf=False)[0] == f32(
    0, 128, 0
  )
  # FP32 max + HALF an ulp: RNE ties up (guard=1) into the exponent and overflows; ROD keeps max-normal
  # (guard=1 means ~guard=0, no round up), so it does NOT overflow at all.
  half = (2 - Fraction(1, 2**23)) * Fraction(2) ** 127 + Fraction(2) ** 103
  assert nr.round_exact(half, 23, 8, nr.RNE, widen_overflow_inf=False) == (
    f32(0, 255, 0),
    nr.Status(of=1, nx=1),
  )
  assert nr.round_exact(half, 23, 8, nr.ROD, widen_overflow_inf=False) == (
    f32(0, 254, 0x7FFFFF),
    nr.Status(nx=1),
  )
  # FP32 max + a full ulp is exactly 2**128: every mode overflows. ROD saturates at the accumulate stage
  # and returns Inf at the widening multiplier stage.
  full = Fraction(2) ** 128
  assert nr.round_exact(full, 23, 8, nr.ROD, widen_overflow_inf=False)[0] == f32(0, 254, 0x7FFFFF)
  assert nr.round_exact(full, 23, 8, nr.ROD, widen_overflow_inf=True)[0] == f32(0, 255, 0)
  assert nr.round_exact(full, 23, 8, nr.RNE, widen_overflow_inf=False)[0] == f32(0, 255, 0)
  assert nr.round_exact(full, 23, 8, nr.ROD, widen_overflow_inf=False)[1].of == 1


def test_bf16_helpers_reject_out_of_range_and_inexact() -> None:
  with pytest.raises(nr.NumericError):
    nr.bf16_to_fp32_bits(0x10000)
  with pytest.raises(nr.NumericError):
    nr.fp32_to_bf16_bits_exact(0x3F808000)
  assert nr.fp32_to_bf16_bits_exact(nr.bf16_to_fp32_bits(0x3F80)) == 0x3F80
  with pytest.raises(nr.NumericError):
    nr.decode(1 << 16, 8, 7)


def test_decode_classifies_the_special_patterns() -> None:
  assert nr.decode(INF, 8, 7)[0] == "inf"
  assert nr.decode(QNAN, 8, 7)[0] == "qnan"
  assert nr.decode(SNAN, 8, 7)[0] == "snan"
  assert nr.decode(1, 8, 7)[0] == "subnormal"
  assert nr.decode(0x8000, 8, 7)[:2] == ("zero", 1)


# --- frm and narrowing ----------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("frm", "code"), [(7, "FRM_DYN"), (5, "FRM_ROD"), (6, "FRM_UNASSIGNED"), (8, "FRM_RANGE")]
)
def test_frm_rejections(frm: int, code: str) -> None:
  with pytest.raises(nr.NumericError) as exc:
    nr.check_frm(frm)
  assert exc.value.code == code


def test_admitted_frm_set_is_exactly_the_five_riscv_modes() -> None:
  admitted = []
  for frm in range(8):
    try:
      nr.check_frm(frm)
      admitted.append(frm)
    except nr.NumericError:
      pass
  assert admitted == [0, 1, 2, 3, 4]


def test_narrowing_is_a_separate_boundary_from_vme_rounding() -> None:
  """The same FP32 pattern narrows differently under RNE (frm) than ROD would round it."""
  tie_odd = f32(0, 127, 0x018000)
  assert nr.narrow_fp32_to_bf16(tie_odd, nr.RNE)[0] == 0x3F82
  assert nr.narrow_fp32_to_bf16(tie_odd, nr.RTZ)[0] == 0x3F81
  assert nr.narrow_fp32_to_bf16(f32(0, 255, 0))[0] == 0x7F80
  assert nr.narrow_fp32_to_bf16(f32(1, 255, 0))[0] == 0xFF80
  assert nr.narrow_fp32_to_bf16(f32(0, 255, 0x400000)) == (0x7FC0, nr.Status())


def test_narrowing_rejects_dyn() -> None:
  with pytest.raises(nr.NumericError):
    nr.narrow_fp32_to_bf16(f32(0, 127, 0), 7)


# --- KD-01 and the open items are recorded, not hidden -------------------------------------------


def test_known_difference_kd01_is_recorded_and_the_contract_is_not_changed(profile: dict[str, Any]) -> None:
  kd = {k["id"]: k for k in profile["known_differences"]}
  assert "KD-01" in kd and "NOT changed" in kd["KD-01"]["resolution"]
  assert nr.vtfmm_cell(f32(0, 254, 0x7FFFFF), [(MAXN, ONE), (Z, Z)], 1)[0] == 0x7F7FFFFF


def test_open_items_are_reported_by_the_validator() -> None:
  report = vn.validate()
  assert report["open_items"] == ["OPEN-02", "OPEN-03", "OPEN-04"]  # OPEN-01 closed by the probe


# --- the probe driver's host half: pack, compare, tamper ------------------------------------------


def _fake_probe(groups: list[dict[str, Any]]) -> dict[str, Any]:
  # A real record is separate JSON, never sharing objects with the request.
  return copy.deepcopy(
    {
      "schema": "cn_p0_numeric_probe/v1",
      "groups": [{**g, "actual_u32": probe.model_group(g)} for g in groups],
    }
  )


def test_every_cell_vector_is_placed_exactly_once(vectors: dict[str, Any]) -> None:
  groups = probe.build_groups(vectors)
  placed = [p["vector"] for g in groups for p in g["placed"]]
  expected = [v["id"] for v in vectors["vectors"] if v["kind"] in probe.CELL_KINDS]
  assert sorted(placed) == sorted(expected)
  assert all(len(g["placed"]) <= probe.DIM for g in groups)


def test_two_vectors_never_share_a_row_or_a_column(vectors: dict[str, Any]) -> None:
  for group in probe.build_groups(vectors):
    cells = [tuple(p["cell"]) for p in group["placed"]]
    assert len({r for r, _ in cells}) == len(cells) and len({c for _, c in cells}) == len(cells)


def test_a_placed_vector_lands_on_its_own_operands(vectors: dict[str, Any]) -> None:
  by_id = {v["id"]: v for v in vectors["vectors"]}
  for group in probe.build_groups(vectors):
    for place in group["placed"]:
      row, col = place["cell"]
      vec = by_id[place["vector"]]
      assert group["a_bf16_bits"][row] == vec["pairs_bf16_bits"][0][0]
      assert group["a_bf16_bits"][probe.DIM + row] == vec["pairs_bf16_bits"][1][0]
      assert group["b_bf16_bits"][col] == vec["pairs_bf16_bits"][0][1]
      assert group["b_bf16_bits"][probe.DIM + col] == vec["pairs_bf16_bits"][1][1]
      assert group["c_init_u32"][row * probe.DIM + col] == vec["old_fp32_bits"]
      assert group["tk"] == vec["tk"]


def test_model_output_placed_in_the_tile_reproduces_every_hand_derived_vector(
  vectors: dict[str, Any],
) -> None:
  """The packing is faithful: running the model on the packed tile gives each vector's expected bits."""
  groups = probe.build_groups(vectors)
  assert probe.vector_results(_fake_probe(groups), groups, vectors) == []


def test_probe_comparison_is_clean_for_a_faithful_record(vectors: dict[str, Any]) -> None:
  groups = probe.build_groups(vectors)
  assert probe.compare(_fake_probe(groups), groups) == []


def test_a_flipped_result_bit_is_named_by_cell(vectors: dict[str, Any]) -> None:
  groups = probe.build_groups(vectors)
  rec = _fake_probe(groups)
  rec["groups"][1]["actual_u32"][33] ^= 1
  errors = probe.compare(rec, groups)
  assert any("group 1 cell 33" in e for e in errors)
  assert probe.tamper_selftest(_fake_probe(groups), groups)["detected"] is True


def test_a_recorded_input_that_differs_from_the_request_is_rejected(vectors: dict[str, Any]) -> None:
  groups = probe.build_groups(vectors)
  rec = _fake_probe(groups)
  rec["groups"][0]["a_bf16_bits"][0] ^= 1
  assert any("a_bf16_bits differs" in e for e in probe.compare(rec, groups))


def test_a_missing_or_unrequested_group_is_rejected(vectors: dict[str, Any]) -> None:
  groups = probe.build_groups(vectors)
  rec = _fake_probe(groups)
  rec["groups"] = rec["groups"][:-1]
  assert any("missing from the probe record" in e for e in probe.compare(rec, groups))
  rec = _fake_probe(groups)
  rec["groups"].append({**rec["groups"][0], "group": 99})
  assert any("not requested" in e for e in probe.compare(rec, groups))


def test_a_wrong_schema_is_rejected(vectors: dict[str, Any]) -> None:
  groups = probe.build_groups(vectors)
  assert probe.compare({"schema": "other", "groups": []}, groups)


def test_real_rtl_disagreeing_with_a_hand_derived_vector_is_reported(vectors: dict[str, Any]) -> None:
  groups = probe.build_groups(vectors)
  rec = _fake_probe(groups)
  place = groups[0]["placed"][0]
  rec["groups"][0]["actual_u32"][place["cell"][0] * probe.DIM + place["cell"][1]] ^= 0x100
  assert any(place["vector"] in e for e in probe.vector_results(rec, groups, vectors))


def test_unknown_kernel_impl_is_rejected(vectors: dict[str, Any]) -> None:
  with pytest.raises(probe.ProbeError):
    probe.build_groups(vectors, impl="vtfmm_int8")


# --- the committed real-RTL probe record ----------------------------------------------------------


def test_committed_probe_record_confirms_every_hand_derived_vector_on_the_real_rtl(
  vectors: dict[str, Any],
) -> None:
  errors: list[str] = []
  assert vn.check_probe_record(vectors, errors) == 30
  assert errors == []


def _probe_record() -> dict[str, Any]:
  return json.loads(vn.PROBE_RECORD.read_text())


def _write(tmp_path: Path, record: dict[str, Any]) -> Path:
  path = tmp_path / "probe.json"
  path.write_text(json.dumps(record))
  return path


def test_probe_record_result_bit_tamper_is_named(tmp_path: Path, vectors: dict[str, Any]) -> None:
  record = _probe_record()
  record["groups"][0]["actual_u32"][0] ^= 0x1
  errors: list[str] = []
  vn.check_probe_record(vectors, errors, _write(tmp_path, record))
  assert any("group 0 cell 0" in e for e in errors)


def test_probe_record_that_ran_another_elf_is_rejected(tmp_path: Path, vectors: dict[str, Any]) -> None:
  record = _probe_record()
  record["elf_sha256"] = "0" * 64
  errors: list[str] = []
  vn.check_probe_record(vectors, errors, _write(tmp_path, record))
  assert any("different ELF" in e for e in errors)


def test_probe_record_without_identical_replay_is_rejected(tmp_path: Path, vectors: dict[str, Any]) -> None:
  record = _probe_record()
  record["replay_identical"] = False
  errors: list[str] = []
  vn.check_probe_record(vectors, errors, _write(tmp_path, record))
  assert any("not identical" in e for e in errors)


def test_probe_record_missing_is_reported(tmp_path: Path, vectors: dict[str, Any]) -> None:
  errors: list[str] = []
  assert vn.check_probe_record(vectors, errors, tmp_path / "absent.json") == 0
  assert any("missing" in e for e in errors)


def test_a_wrong_hand_derived_vector_is_caught_by_the_real_rtl(vectors: dict[str, Any]) -> None:
  """A wrong hand-derived vector is exposed by the real RTL record, not just by the model."""
  bad = copy.deepcopy(vectors)
  next(v for v in bad["vectors"] if v["id"] == "rod_odd_sticky")["expected_fp32_bits"] ^= 2
  errors: list[str] = []
  vn.check_probe_record(bad, errors)
  assert any("rod_odd_sticky" in e for e in errors)


def test_every_boundary_vector_the_card_names_reaches_the_real_rtl(vectors: dict[str, Any]) -> None:
  rec = _probe_record()
  placed = {p["vector"] for g in probe.build_groups(vectors) for p in g["placed"]}
  needed = {
    "mul_overflow_two_lane_sum",
    "acc_overflow_to_max_normal",
    "msk_masked_inf_isolated",
    "can_tree_cancel",
    "tkg_grouping_differs",
    "jam_large_exponent_gap",
    "rod_guard_set_no_round",
    "spc_snan_sets_nv",
    "sub_times_sub_underflow",
  }
  assert needed <= placed
  assert rec["evidence_tier"] == "p0_upstream_rtl" and rec["replay_identical"] is True


def test_probe_flags_are_not_claimed_as_observed_on_the_rtl() -> None:
  profile = vn.load_profile()
  items = {i["id"]: i for i in profile["open_items"]}
  assert items["OPEN-01"]["status"] == "closed" and "Result bits only" in items["OPEN-01"]["scope_limit"]
  assert items["OPEN-04"]["status"] == "open"
