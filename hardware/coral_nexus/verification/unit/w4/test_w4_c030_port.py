"""C030 program-port slice: behavioral mutants before the real shared-port bench."""

from __future__ import annotations

from typing import Any

import pytest
from w4_rtl_harness import Bench, Slice, check_verdicts, run_slice

BENCH = Bench(
  top="tb_l1_program_port",
  sources=("c029_params_pkg.sv", "l1_program_port.sv", "tb/tb_l1_program_port.sv"),
  contexts=(1, 2),
)
SPEC = Slice(name="c030port", benches=[BENCH], mutant_globs=("port-*.patch",), mutant_bench=BENCH)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  return run_slice(SPEC, tmp_path_factory)


def test_c030_program_port_acceptance(evidence: dict[str, Any]) -> None:
  check_verdicts(evidence, pass_marker="C030_PASS")
