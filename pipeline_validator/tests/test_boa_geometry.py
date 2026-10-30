"""P05 BOA engine-pool configuration, geometry timing, and identity gates.

Boundary tests for the new ``engines.boa`` configuration keys
(``arrays_per_tile``, ``array_rows``/``array_cols``, ``tile_m``/``tile_n``),
the closed-form spatial-wave timing model in ``BOAEngine``, the static
per-engine L1 share, the target-fingerprint coupling, and per-engine
runtime identity/snapshot counters.  Everything here is deterministic:
exact closed-form expectations, no scenario cycle pins, no wiring echoes.

Run with:  python -m pytest pipeline_validator/tests/test_boa_geometry.py -q
"""

from __future__ import annotations

import math

import pytest

from pipeline_validator.config import MAX_BOA_ENGINES, HardwareConfig, SimConfig
from pipeline_validator.engines import BOAEngine
from pipeline_validator.execution_ir import (
  ExecBoaMatmulDesc,
  ExecEngineDesc,
  ExecMemoryView,
)
from pipeline_validator.execution_verifier import target_fingerprint

_UINT64_MAX = (1 << 64) - 1

# Default geometry: 1 array of 16x16, logical chunk 16x16, boost 1.0,
# launch 4 cycles, tile L1 512 GB/s at 1 GHz = 512 B/cycle.
_DEFAULT_ROWS = 16
_DEFAULT_LANES = 16 * 16

# Pool fixture for per-engine identity tests: ids 0..3 are all in-pool.
_FOUR_ARRAYS = HardwareConfig(boa_arrays_per_tile=4)


def _l1_view() -> ExecMemoryView:
  return ExecMemoryView("l1", "l1:0", (1,), (1,), (0,), (1,), "bf16", 2, 2)


def _typed(m: int, n: int, k: int, *, final: bool = False) -> ExecEngineDesc:
  matmul = ExecBoaMatmulDesc(_l1_view(), _l1_view(), _l1_view(), "acc0", m, n, k, 0, True, final)
  return ExecEngineDesc(name="k", kind="BOA", op="matmul", params={"matmul": matmul})


def _estimate(
  ops: int,
  m: int | None = None,
  n: int | None = None,
  k: int | None = None,
  *,
  nbytes: int = 0,
) -> ExecEngineDesc:
  """Timing-only estimate descriptor; shapes optional (legacy ops-only form)."""
  params: dict = {"ops": ops}
  if nbytes:
    params["bytes"] = nbytes
  if m is not None:
    params.update({"m": m, "n": n, "k": k, "estimate": True})
  return ExecEngineDesc(name="e", kind="BOA", op="matmul", params=params)


# ---------------------------------------------------------------------------
# Configuration boundary
# ---------------------------------------------------------------------------


class TestBoaArrayConfig:
  def test_max_boa_engines_is_the_software_cap(self):
    assert MAX_BOA_ENGINES == 8

  @pytest.mark.parametrize("count", [1, 2, 3, 4, 7, 8])
  def test_array_count_accepts_exact_ints(self, count):
    assert HardwareConfig(boa_arrays_per_tile=count).boa_arrays_per_tile == count

  @pytest.mark.parametrize(
    "count",
    [True, False, 0, -1, 9, 64, 2**64, 1.0, 2.5, "2", None],
  )
  def test_array_count_rejects_bool_float_zero_negative_huge(self, count):
    with pytest.raises(ValueError, match="boa_arrays_per_tile"):
      HardwareConfig(boa_arrays_per_tile=count)

  @pytest.mark.parametrize(
    "field",
    ["boa_array_rows", "boa_array_cols", "boa_tile_m", "boa_tile_n"],
  )
  @pytest.mark.parametrize("value", [1, 17, _UINT64_MAX])
  def test_geometry_accepts_exact_positive_uint64(self, field, value):
    assert getattr(HardwareConfig(**{field: value}), field) == value

  @pytest.mark.parametrize(
    "field",
    ["boa_array_rows", "boa_array_cols", "boa_tile_m", "boa_tile_n"],
  )
  @pytest.mark.parametrize(
    "value",
    [0, -1, _UINT64_MAX + 1, True, False, 16.0, "16", None],
  )
  def test_geometry_rejects_non_exact_or_out_of_range(self, field, value):
    with pytest.raises(ValueError, match=field):
      HardwareConfig(**{field: value})

  @pytest.mark.parametrize("value", [1, 0.5, 2.0, 1e6])
  def test_clock_multiplier_accepts_finite_positive(self, value):
    cfg = HardwareConfig(boa_clock_multiplier=value)
    assert cfg.boa_clock_multiplier == pytest.approx(value)

  @pytest.mark.parametrize(
    "value",
    [0, -1.0, 0.0, float("inf"), float("-inf"), float("nan"), True, "1.0", None],
  )
  def test_clock_multiplier_rejects_non_finite_or_non_positive(self, value):
    with pytest.raises(ValueError, match="boa_clock_multiplier"):
      HardwareConfig(boa_clock_multiplier=value)

  def test_flat_override_drives_pool_size(self):
    assert HardwareConfig().with_overrides(boa_arrays_per_tile=4).boa_arrays_per_tile == 4

  def test_yaml_partial_override_drives_pool_and_geometry(self, tmp_path):
    path = tmp_path / "hw.yaml"
    path.write_text(
      "engines:\n  boa:\n    arrays_per_tile: 4\n    array_rows: 8\n    tile_n: 32\n",
      encoding="utf-8",
    )
    cfg = HardwareConfig.from_yaml(path)
    assert (cfg.boa_arrays_per_tile, cfg.boa_array_rows, cfg.boa_tile_n) == (4, 8, 32)

  def test_removed_opa_fields_rejected_on_flat_api(self):
    with pytest.raises(TypeError):
      HardwareConfig(boa_num_opa=8)
    with pytest.raises(TypeError):
      HardwareConfig(boa_opa_rows=16, boa_opa_cols=16)
    with pytest.raises(TypeError):
      HardwareConfig().with_overrides(boa_num_opa=8)

  def test_removed_opa_subtree_rejected_on_yaml(self, tmp_path):
    path = tmp_path / "hw.yaml"
    path.write_text(
      "engines:\n  boa:\n    opa:\n      count: 8\n      rows: 16\n      cols: 16\n",
      encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"unknown hardware config path 'engines\.boa\.opa'"):
      HardwareConfig.from_yaml(path)

  def test_numeric_tile_k_reserved_for_p06_not_available(self, tmp_path):
    path = tmp_path / "hw.yaml"
    path.write_text("engines:\n  boa:\n    numeric_tile_k: 8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown hardware config path"):
      HardwareConfig.from_yaml(path)
    assert not hasattr(HardwareConfig(), "boa_numeric_tile_k")


# ---------------------------------------------------------------------------
# Target identity
# ---------------------------------------------------------------------------


class TestBoaTargetIdentity:
  @pytest.mark.parametrize(
    ("field", "value"),
    [
      ("boa_arrays_per_tile", 2),
      ("boa_array_rows", 32),
      ("boa_array_cols", 8),
      ("boa_tile_m", 32),
      ("boa_tile_n", 8),
    ],
  )
  def test_each_boa_field_changes_target_fingerprint(self, field, value):
    base = target_fingerprint(HardwareConfig(), SimConfig())
    changed = target_fingerprint(HardwareConfig().with_overrides(**{field: value}), SimConfig())
    assert changed != base

  def test_fingerprint_is_stable_for_equal_configs(self):
    assert target_fingerprint(HardwareConfig(), SimConfig()) == target_fingerprint(
      HardwareConfig(), SimConfig()
    )

  def test_timing_only_knob_stays_out_of_target_fingerprint(self):
    base = target_fingerprint(HardwareConfig(), SimConfig())
    assert target_fingerprint(
      HardwareConfig().with_overrides(boa_launch_cycles=7), SimConfig()
    ) == base


# ---------------------------------------------------------------------------
# Spatial-wave timing (one engine = one physical array)
# ---------------------------------------------------------------------------


class TestBoaSpatialTiming:
  def test_single_chunk_single_wave_compute(self):
    eng = BOAEngine(HardwareConfig(), 0)
    # waves 1*1, k=16; byte roofline: 2048 B / 512 B/cycle = 4 < 16.
    assert eng.breakdown(_typed(16, 16, 16)) == (16, 2, 0)
    assert eng.breakdown(_typed(16, 16, 16, final=True)) == (16, 2, 1)

  def test_partial_wave_tail_counts_a_full_pass(self):
    eng = BOAEngine(HardwareConfig(), 0)
    # m=33: two full 16-chunks plus a 1-element remainder → 3 row passes.
    compute, _commit, _drain = eng.breakdown(_typed(33, 16, 16))
    assert compute == 3 * 16

  def test_tile_m_changes_partial_wave_timing(self):
    # m=48, k=8: tile_m=16 → 3 chunks; tile_m=24 with rows=16 → each
    # 24-wide chunk needs ceil(24/16)=2 row passes → 2*2=4 passes.
    c16 = BOAEngine(HardwareConfig(), 0).breakdown(_typed(48, 16, 8))[0]
    c24 = BOAEngine(HardwareConfig(boa_tile_m=24), 0).breakdown(_typed(48, 16, 8))[0]
    assert c16 == 3 * 8
    assert c24 == 4 * 8
    assert c16 != c24

  def test_array_width_changes_remainder_tail(self):
    # m=17, tile_m=16, k=8: rows=16 → 1 full pass + 1 tail pass; rows=8
    # → 2 passes for the full chunk + 1 tail pass.
    c16 = BOAEngine(HardwareConfig(), 0).breakdown(_typed(17, 16, 8))[0]
    c8 = BOAEngine(HardwareConfig(boa_array_rows=8), 0).breakdown(_typed(17, 16, 8))[0]
    assert c16 == 2 * 8
    assert c8 == 3 * 8

  def test_rectangular_aspect_changes_timing_at_equal_area(self):
    # 8x32 and 32x8 have the same 512-MAC area as 16x16, but a 16-tall
    # logical chunk straddles the 8-row array: the wave count grows.
    k = 4
    square = BOAEngine(HardwareConfig(), 0).breakdown(_typed(16, 48, k))[0]
    wide = BOAEngine(HardwareConfig(boa_array_rows=8, boa_array_cols=32), 0).breakdown(
      _typed(16, 48, k)
    )[0]
    assert square == 3 * k  # waves 1*3
    assert wide == 6 * k  # waves 2*3

  def test_clock_multiplier_scales_compute(self):
    eng = BOAEngine(HardwareConfig(boa_clock_multiplier=2.0), 0)
    compute, _commit, _drain = eng.breakdown(_typed(16, 16, 17))
    assert compute == math.ceil(17 / 2.0) == 9

  def test_typed_byte_roofline_uses_static_per_engine_l1_share(self):
    # m=n=64, k=1: compute waves 4*4=16; touched 16640 B → 33 cycles at
    # the full 512 B/cycle share, 65 cycles at the 2-engine 256 B/cycle
    # share.  The share is static: no dynamic borrowing between engines.
    one = BOAEngine(HardwareConfig(), 0)
    two = BOAEngine(HardwareConfig(boa_arrays_per_tile=2), 0)
    assert one.breakdown(_typed(64, 64, 1))[0] == 33
    assert two.breakdown(_typed(64, 64, 1))[0] == 65


# ---------------------------------------------------------------------------
# Estimate timing (timing-only, no tensor data, no AM lease)
# ---------------------------------------------------------------------------


class TestBoaEstimateTiming:
  def test_estimate_matches_typed_for_same_shape(self):
    eng = BOAEngine(HardwareConfig(), 0)
    m, n, k = 64, 256, 64
    estimate = eng.breakdown(_estimate(2 * m * n * k, m, n, k))[0]
    typed = eng.breakdown(_typed(m, n, k))[0]
    assert estimate == typed == 4 * 16 * k

  def test_estimate_scales_with_declared_ops(self):
    eng = BOAEngine(HardwareConfig(), 0)
    m, n, k = 64, 256, 64
    full = eng.breakdown(_estimate(2 * m * n * k, m, n, k))[0]
    half = eng.breakdown(_estimate(m * n * k, m, n, k))[0]
    assert full == 4 * 16 * k
    assert half == math.ceil(full / 2)

  def test_estimate_without_shapes_uses_effective_lanes(self):
    eng = BOAEngine(HardwareConfig(), 0)
    # min(rows, tile_m) * min(cols, tile_n) = 256 effective lanes.
    assert eng.breakdown(_estimate(262144))[0] == math.ceil(131072 / _DEFAULT_LANES)
    smaller = BOAEngine(HardwareConfig(boa_tile_m=8, boa_tile_n=8), 0)
    assert smaller.breakdown(_estimate(262144))[0] == math.ceil(131072 / 64)

  def test_estimate_byte_roofline_uses_per_engine_l1_share(self):
    one = BOAEngine(HardwareConfig(), 0)
    two = BOAEngine(HardwareConfig(boa_arrays_per_tile=2), 0)
    # 1024 B: 2 cycles at the 512 B/cycle share, 4 at the 2-engine share.
    assert one.breakdown(_estimate(2, nbytes=1024))[0] == 2
    assert two.breakdown(_estimate(2, nbytes=1024))[0] == 4


# ---------------------------------------------------------------------------
# Per-engine runtime identity and snapshot counters
# ---------------------------------------------------------------------------


class TestBoaEngineIdentity:

  def test_engine_id_rejects_non_exact_or_out_of_pool(self):
    with pytest.raises(ValueError, match="engine_id"):
      BOAEngine(HardwareConfig(), 0, engine_id=-1)
    with pytest.raises(ValueError, match="engine_id"):
      BOAEngine(HardwareConfig(), 0, engine_id=True)
    with pytest.raises(ValueError, match="engine_id"):
      BOAEngine(HardwareConfig(), 0, engine_id=1.0)
    # engine_id == pool size (default pool 1) and any id beyond the pool
    # are outside the configured pool.
    with pytest.raises(ValueError, match="engine_id"):
      BOAEngine(HardwareConfig(), 0, engine_id=1)
    with pytest.raises(ValueError, match="engine_id"):
      BOAEngine(HardwareConfig(boa_arrays_per_tile=2), 0, engine_id=2)



  def test_counters_cover_execution_and_reset_cancels_outstanding_work(self):
    engine = BOAEngine(_FOUR_ARRAYS, 0, engine_id=3)
    job = engine.launch(_estimate(32768), 10, "ctx0:complete")
    assert job is not None
    # Follow the Tile's tick-before-issue convention, not a pinned latency.
    for cycle in range(11, job.finish_cycle + 1):
      completed = engine.tick(cycle)
      if cycle < job.finish_cycle:
        assert completed == [] and engine.is_busy
    assert completed == [job] and not engine.has_accepted_jobs
    assert engine.snapshot()["active_cycles"] == job.finish_cycle - job.start_cycle
    assert engine.pmu.named_cycles["boa_active"] == engine.snapshot()["active_cycles"]
    assert engine.snapshot()["launch_count"] == engine.snapshot()["complete_count"] == 1
    running = engine.launch(_estimate(65536), job.finish_cycle + 1, "ctx0:cancel")
    assert running is not None and engine.is_busy
    engine.reset()
    assert not engine.has_accepted_jobs
    assert engine.tick(running.finish_cycle) == []
    assert engine.snapshot()["launch_count"] == engine.snapshot()["complete_count"] == 0
    assert engine.snapshot()["active_cycles"] == 0





def test_fractional_l1_share_rounds_up_without_integer_bandwidth_assumption():
  engine = BOAEngine(HardwareConfig(boa_arrays_per_tile=3), 0)
  assert engine.breakdown(_estimate(2, nbytes=171))[0] == 2
  assert engine.breakdown(_typed(3, 24, 1))[0] == 3
  # Exact boundary must not drift to seven cycles after a float round-trip.
  assert engine.breakdown(_estimate(2, nbytes=1024))[0] == 6
