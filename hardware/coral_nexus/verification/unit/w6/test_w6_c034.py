"""C034: reset/drain endpoint over the real W6 RTL with a live Group boundary.

Unit scope: `tb_c034` instantiates the real `rtl/w6` `event_table`,
`terminal_store`, `first_fault_latch` and `reset_endpoint` next to the real
C032 `mfe_top` (internal `transfer_cancel_tracker`) and the real C023
MemoryService — W6-03 module evidence only: no full-Tile assembly (C040), no
retirement chain or its reset policy (C035), no C035 config/admission.
`tb_c034_memory.sv` compiles immediately before `tb_c034.sv` and carries the
C023 service tasks, split from the reset behavioral checks. CONTEXT_COUNT is
the only generated dimension (C1/C2 from the sealed target); identities are
full 64-bit transport words.

The nine `u3?_*.yaml` scenarios are validated fail-closed and materialized
once as `evidence/c034-vectors.txt` — nine lowercase-hex words per row
(inject generation/tag/hold_cycles, then expect
winner/suppressed/marked/terminals/stale/poisoned), rows sorted a..i — behind
`+VECTORS=/out/c034-vectors.txt`. The bench owns the comparison and emits one
`C034_CASE U3-x` line per case plus a single
`C034_PASS <top> contexts=N checks=M` record; this runner never hard-codes
green output. `+PEER=/out/<label>` connects each invocation to the live
`c034_group_peer.py` (real Group reset cycles for U3-f/h/i, real C023
windows for U3-e/f, real FaultRing observations). Scenario/vector/fragment
IO is shared with the C033 runner through `w6_scenario_io.py`; both C015
fragments are hashed against the real C020 seal.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

W6 = Path(__file__).resolve().parent
W4 = W6.parent / "w4"
H = W6.parents[2]
REPO = W6.parents[4]
sys.path.insert(0, str(W4))
sys.path.insert(0, str(H / "scripts/lib"))

import gen_memory_params as memory_params  # noqa: E402
import w6_scenario_io  # noqa: E402
from c034_group_peer import serve as serve_live_peer  # noqa: E402
from w4_rtl_harness import Bench, Slice, check_verdicts, run_slice, selected_target  # noqa: E402

from pipeline_validator.cosim import wire  # noqa: E402
from pipeline_validator.cosim.memory_service import MEM_OP_READ, MEM_OP_WRITE  # noqa: E402

SCENARIOS = H / "verification/scenarios/w6"
SEAL = H / "protocol/PROTOCOL_FREEZE.json"
FRAGMENTS = (
  H / "protocol/w6_event_facility.contract.yaml",
  H / "protocol/w6_terminal_reset.contract.yaml",
)
CASE_IDS = tuple(f"U3-{letter}" for letter in "abcdefghi")
INJECT_KEYS = ("generation", "tag", "hold_cycles")
EXPECT_KEYS = ("winner", "suppressed", "marked", "terminals", "stale", "poisoned")
PASS_RE = re.compile(r"^C034_PASS (\S+) contexts=(\d+) checks=(\d+)$", re.M)
WIRE_PHASES = tuple(wire.ENUMS["reset_phase"])

BENCH = Bench(
  top="tb_c034",
  sources=(
    "w6/c033_event_pkg.sv",
    "w6/c034_reset_pkg.sv",
    "w6/wait_record.sv",
    "w6/completion_bus.sv",
    "w6/event_table.sv",
    "w6/terminal_store.sv",
    "memory/c029_params_pkg.sv",
    "memory/c032_mfe_pkg.sv",
    "memory/transfer_cancel_tracker.sv",
    "memory/mfe_top.sv",
    "w6/first_fault_latch.sv",
    "w6/reset_endpoint.sv",
    "w6/tb/tb_c034_memory.sv",
    "w6/tb/tb_c034.sv",
  ),
  contexts=(1, 2),
  run_args=("+VECTORS=/out/c034-vectors.txt", "+PEER=/out/{label}"),
  timeout_s=600,
)


def load_scenarios() -> list[dict[str, Any]]:
  """Validate exactly the nine U3-a..U3-i inject+expect scenario rows, sorted a..i."""
  return w6_scenario_io.load_scenarios(
    SCENARIOS, "u3?_*.yaml", CASE_IDS, INJECT_KEYS, EXPECT_KEYS, seed_name="inject"
  )


def vectors(scenarios: Sequence[dict[str, Any]]) -> str:
  """One row per case: nine lowercase-hex words, three inject then six expected."""
  return w6_scenario_io.vectors(scenarios, INJECT_KEYS, EXPECT_KEYS, seed_name="inject")


def fragment_receipts() -> dict[str, str]:
  """Hash both C015 fragments and cross-check them against the real C020 seal."""
  return w6_scenario_io.fragment_receipts(FRAGMENTS, SEAL, REPO)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  scenarios = load_scenarios()  # fail closed before any toolchain work
  fragments = fragment_receipts()
  rows = vectors(scenarios)
  receipt = {
    "scenarios": [
      {
        "case": row["case"],
        "file": str(row["file"].relative_to(REPO)),
        "sha256": hashlib.sha256(row["file"].read_bytes()).hexdigest(),
        "inject": row["inject"],
        "expect": row["expect"],
      }
      for row in scenarios
    ]
  }

  @contextmanager
  def serve(out: Path, _labels: Sequence[str]) -> Iterator[None]:
    (out / "c034-vectors.txt").write_text(rows)
    (out / "c034-scenarios.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (out / "c015-fragments.json").write_text(json.dumps(fragments, indent=2, sort_keys=True) + "\n")
    with serve_live_peer(out, _labels):
      yield

  return run_slice(
    Slice(
      name="c034w6",
      benches=(BENCH,),
      mutant_globs=("reset-*.patch",),
      mutant_bench=BENCH,
      min_mutants=9,
      assert_markers=("C034_ASSERT",),
      service=serve,
      host_sources=(
        W6 / "test_w6_c034.py",
        W6 / "c034_group_peer.py",
        W6 / "w6_scenario_io.py",
        W4 / "w4_rtl_harness.py",
        W4 / "c032_mfe_peer.py",
        REPO / "pipeline_validator/cosim/memory_service.py",
        REPO / "pipeline_validator/cosim/tile_port.py",
        REPO / "pipeline_validator/cosim/wire.py",
        REPO / "pipeline_validator/cosim/reset_phase.py",
        REPO / "pipeline_validator/runtime/reset_domain.py",
        REPO / "pipeline_validator/runtime/fault_ring.py",
        REPO / "pipeline_validator/memory/byte_store.py",
        REPO / "pipeline_validator/tile_group.py",
        REPO / "pipeline_validator/tests/test_runtime.py",
        *FRAGMENTS,
        *(row["file"] for row in scenarios),
      ),
      rtl_dir=H / "rtl",
      mutants_dir=H / "verification/unit/w6/mutants",
    ),
    tmp_path_factory,
  )


def _check_live_cycles(label: str, result: dict[str, Any]) -> None:
  """Three completed live Group resets (U3-f with poison+recovery, U3-h, U3-i)."""
  assert result["errors"] == [], (label, result["errors"])
  assert not result["open_scenario"], f"{label}: a live Group cycle never closed"
  assert not result["open_memory"], f"{label}: a MemoryService window never closed"
  scenarios = result["scenarios"]
  assert len(scenarios) == 3, f"{label}: expected live cycles f/h/i, got {len(scenarios)}"
  for scenario in scenarios:
    where = (label, scenario["fault_idx"], scenario["domain"])
    assert scenario["fault_idx"] >= 0, where
    assert scenario["domain"] == "GROUP", where
    assert [request["phase"] for request in scenario["requests"]] == list(WIRE_PHASES), where
    assert [ack["phase"] for ack in scenario["acks"]] == list(WIRE_PHASES), where
    drain = next(ack for ack in scenario["acks"] if ack["phase"] == "DRAIN")
    assert drain["mask"] == 0x3F, where
    assert all(ack["mask"] == 0 for ack in scenario["acks"] if ack["phase"] != "DRAIN"), where
    assert all(ack["generation"] == scenario["generation"] for ack in scenario["acks"]), where
    closed = scenario["z"]
    assert closed is not None and closed["done"], where
    assert closed["phases"] == list(WIRE_PHASES), where
    assert all(not residue for residue in closed["leak_checks"].values()), where
    assert len(closed["leak_checks"]) == 7, where  # five structural + L2 + credits
  first = scenarios[0]  # case order f < h < i: U3-f is the first live cycle
  assert first["poison"] is not None and first["poison"]["observed"], (label, first)
  assert [(attempt["allowed"], attempt["poisoned"]) for attempt in first["recoveries"]] == [
    (0, 1), (1, 0)
  ], (label, first["recoveries"])  # refused while poisoned, then allowed and cleared
  for scenario in scenarios[1:]:
    assert scenario["poison"] is None and scenario["recoveries"] == [], (label, scenario)


def _check_memory_windows(label: str, result: dict[str, Any]) -> None:
  """Two closed real C023 windows: U3-e's four accepted legs, U3-f's one inbound."""
  windows = result["memory_windows"]
  assert sorted(len(window["requests"]) for window in windows) == [1, 4], (label, windows)
  for window in windows:
    assert window["service"]["terminal_refs_outstanding"] == 0, (label, window)
    assert sorted(request["txn"] for request in window["requests"]) == sorted(
      terminal["txn"] for terminal in window["terminals"]
    ), (label, window)
  eager = next(window for window in windows if len(window["requests"]) == 4)
  inbound = next(window for window in windows if len(window["requests"]) == 1)
  assert [terminal["final"] for terminal in eager["terminals"]] == [0, 1, 2, 0], (label, eager)
  assert [terminal["final"] for terminal in inbound["terminals"]] == [1], (label, inbound)
  assert sum(request["op"] == MEM_OP_WRITE for request in eager["requests"]) == 1, (label, eager)
  assert inbound["requests"][0]["op"] == MEM_OP_READ, (label, inbound)


def _check_rings(label: str, result: dict[str, Any]) -> None:
  """Real bounded-ring observations: retention, oldest source, physical index."""
  rings = result["rings"]
  assert rings, (label, "no FaultRing observation")
  for ring in rings:
    where = (label, ring)
    assert ring["write_count"] == ring["writes"], where
    assert ring["len"] == min(ring["depth"], ring["writes"]), where
    assert ring["oldest_source"] == ring["writes"] - ring["len"], where
    assert ring["last_index"] == (ring["writes"] - 1) % ring["depth"], where


def test_reset_endpoint_live_group_and_memory(evidence: dict[str, Any]) -> None:
  check_verdicts(evidence, pass_marker="C034_PASS")
  path = evidence["path"]
  # C1/C2 comes from the real sealed config, not from this runner.
  params = memory_params.parameters()
  requested = int(selected_target()["constraints"]["context_count"])
  counts = sorted({int(params["C1_CONTEXTS"]), requested})
  pairs = {(case["top"], case["context_count"]) for case in evidence["positive"]}
  assert pairs == {(BENCH.top, count) for count in counts}
  report = json.loads((path / "group-peer.json").read_text())
  assert report, "the live peer ran"
  for case in evidence["positive"]:
    label = case["label"]
    log = (path / f"{label}.log").read_text()
    records = [line.split() for line in log.splitlines() if line.startswith("C034_CASE")]
    assert all(len(record) == 2 and record[1] in CASE_IDS for record in records), records
    ids = [record[1] for record in records]
    assert sorted(ids) == list(CASE_IDS), f"{case['label']}: case records {ids}"
    passes = PASS_RE.findall(log)
    assert len(passes) == 1, f"{case['label']}: expected one C034_PASS record, got {passes}"
    top, contexts, checks = passes[0]
    assert (top, int(contexts)) == (case["top"], case["context_count"]), (case["label"], passes)
    assert int(checks) >= 1, f"{case['label']}: PASS record executed no checks"
    assert label in report, (label, sorted(report))
    _check_live_cycles(label, report[label])
    _check_memory_windows(label, report[label])
    _check_rings(label, report[label])
