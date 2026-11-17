"""C030 MSHR slice: behavioral mutants before the real RTL token-lifetime bench."""

from __future__ import annotations

from typing import Any

import pytest
from w4_rtl_harness import Bench, Slice, check_verdicts, run_slice

BENCH = Bench(
  top="tb_mshr_table",
  sources=("c029_params_pkg.sv", "mshr_table.sv", "tb/tb_mshr_table.sv"),
  contexts=(1, 2),
)
SPEC = Slice(name="c030mshr", benches=[BENCH], mutant_globs=("mshr-*.patch",), mutant_bench=BENCH)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  return run_slice(SPEC, tmp_path_factory)


def test_c030_mshr_token_lifetime_and_mutation_acceptance(evidence: dict[str, Any]) -> None:
  check_verdicts(evidence, pass_marker="C030_PASS")
