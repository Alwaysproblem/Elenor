"""C032: MFE direct move, per-txn settlement and cancel isolation over the real service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from c032_mfe_peer import serve
from w4_rtl_harness import Bench, Slice, check_verdicts, run_slice

# The MFE has no Context dimension; the bench only carries a registration shim.
BENCH = Bench(
  top="tb_mfe_top",
  sources=(
    "c029_params_pkg.sv",
    "c032_mfe_pkg.sv",
    "transfer_cancel_tracker.sv",
    "mfe_top.sv",
    "tb/tb_mfe_top.sv",
  ),
  run_args=("+SERVICE=/out/{label}",),
  timeout_s=300,
)

W4 = Path(__file__).resolve().parent
REPO = W4.parents[4]
HOST_SOURCES = (
  Path(__file__),
  W4 / "c032_mfe_peer.py",
  W4 / "w4_rtl_harness.py",
  REPO / "pipeline_validator/cosim/memory_service.py",
)

SPEC = Slice(
  name="c032mfe",
  benches=(BENCH,),
  mutant_globs=("mfe-*.patch",),
  mutant_bench=BENCH,
  min_mutants=6,
  assert_markers=("C032_ASSERT", "C029_ASSERT"),
  service=serve,
  host_sources=HOST_SOURCES,
)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  return run_slice(SPEC, tmp_path_factory)


def test_mfe_direct_move_settlement_and_isolation(evidence: dict[str, Any]) -> None:
  check_verdicts(evidence, pass_marker="C032_PASS")
  report = json.loads((evidence["path"] / "memory-service.json").read_text())
  assert report, "the real service peer ran"
  # Only green runs must close their ledger; a killed mutant dies mid-flight
  # and legitimately leaves an obligation open (C031 precedent).
  for case in evidence["positive"]:
    label = case["label"]
    assert label in report, (label, sorted(report))
    result = report[label]
    assert result["error"] is None, (label, result)
    assert result["done"] is True, (label, result)
    assert result["service"]["terminal_refs_outstanding"] == 0, (label, result)
    assert len(result["requests"]) == len(result["terminals"]), (label, result)
    assert len({item["txn"] for item in result["terminals"]}) == len(result["terminals"]), (
      label,
      result,
    )
    assert all(item["final"] in (0, 1, 2) for item in result["terminals"]), (label, result)
