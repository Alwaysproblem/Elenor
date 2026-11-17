"""C030 integrated miss path: cache + MSHR + outbridge over a real l1_bank."""

from __future__ import annotations

from typing import Any

import pytest
from w4_rtl_harness import Bench, Slice, check_verdicts, run_slice

BENCH = Bench(
  top="tb_l1_miss_path",
  sources=(
    "c029_params_pkg.sv",
    "l1_bank.sv",
    "mshr_table.sv",
    "cache.sv",
    "transfer_outbridge.sv",
    "l1_miss_path.sv",
    "tb/tb_l1_miss_path.sv",
  ),
  contexts=(1, 2),
)
SPEC = Slice(
  name="c030miss", benches=[BENCH], mutant_globs=("cache-*.patch", "bridge-*.patch"), mutant_bench=BENCH
)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  return run_slice(SPEC, tmp_path_factory)


def test_c030_miss_path_acceptance(evidence: dict[str, Any]) -> None:
  check_verdicts(evidence, pass_marker="C030_PASS")
