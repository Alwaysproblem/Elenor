"""C031 native LSU acceptance: real bank, external service and full C026 ROB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from c031_memory_peer import serve
from c031_native_rob import NATIVE_STEPS
from w4_rtl_harness import Bench, Slice, check_verdicts, run_slice

CORE = Bench(
  top="tb_lsu_mslot",
  sources=("c029_params_pkg.sv", "c031_lsu_pkg.sv", "l1_bank.sv", "lsu_mslot.sv", "tb/tb_lsu_mslot.sv"),
  contexts=(1, 2),
)
MEMORY = Bench(
  top="tb_c031_memory",
  sources=(
    "c029_params_pkg.sv", "c031_lsu_pkg.sv", "l1_bank.sv", "cache.sv", "mshr_table.sv",
    "transfer_outbridge.sv", "l1_miss_path.sv", "lsu_cache_path.sv", "lsu_mslot.sv",
    "tb/tb_c031_memory.sv",
  ),
  contexts=(1, 2),
  extra_sources=("/out/c031_rob_c{contexts}.sv",),
  args_by_context={1: "+define+C031_C1"},
  run_args=("+SERVICE=/out/{label}",),
)
W4 = Path(__file__).resolve().parent
REPO = W4.parents[4]
HOST_SOURCES = (
  Path(__file__), W4 / "c031_memory_peer.py", W4 / "c031_native_rob.py",
  W4 / "w4_rtl_harness.py", REPO / "pipeline_validator/cosim/memory_service.py",
)


@pytest.fixture(scope="module")
def core_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  return run_slice(
    Slice(name="c031core", benches=(CORE,), mutant_globs=("lsu-core-*.patch",),
          mutant_bench=CORE, assert_markers=("C031_ASSERT", "C029_ASSERT"), min_mutants=10),
    tmp_path_factory,
  )


@pytest.fixture(scope="module")
def memory_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  return run_slice(
    Slice(name="c031memory", benches=(MEMORY,), mutant_globs=("lsu-cache-*.patch",),
          mutant_bench=MEMORY, assert_markers=("C031_ASSERT", "C030_ASSERT", "C029_ASSERT"),
          min_mutants=10, post_toolchain_steps=NATIVE_STEPS, service=serve,
          host_sources=HOST_SOURCES),
    tmp_path_factory,
  )


def test_real_lsu_bank_owner_credit_and_cancel(core_evidence: dict[str, Any]) -> None:
  check_verdicts(core_evidence, pass_marker="C031_PASS")


def test_real_lsu_cache_service_and_native_retirement(memory_evidence: dict[str, Any]) -> None:
  check_verdicts(memory_evidence, pass_marker="C031_PASS")
  report = json.loads((memory_evidence["path"] / "memory-service.json").read_text())
  for case in memory_evidence["positive"]:
    result = report[case["label"]]
    assert result["error"] is None, result
    assert result["done"] is True, result
    assert result["service"]["terminal_refs_outstanding"] == 0, result
    requests = result["requests"]
    terminals = result["terminals"]
    assert {item["txn"] for item in requests} == {item["txn"] for item in terminals}
    assert len(terminals) == len({item["txn"] for item in terminals}), result
    assert {item["offset"] for item in requests} >= {0, 64, 128, 256, 768, 1024}
    assert all(item["generation"] == 1 and item["arena"] == 99 and item["scope"] == 77
               for item in requests)
    assert {(item["binding"], item["binding_generation"]) for item in requests} == {(7, 1), (8, 1), (7, 2)}
    assert sum(item["final"] == 2 for item in terminals) == 1
    if case["context_count"] == 2:
      assert {item["ctx"] for item in requests} == {0, 1}
