"""C033: persistent local events, WAIT records and phase signals over the real W6 RTL.

Unit scope: `tb_c033` instantiates the real `rtl/w6` `event_table` and
`terminal_store` (with their internal `wait_record`/`completion_bus`) — W6-02
module evidence only: no full-Tile assembly (C040), no retirement chain (C035),
no reset/drain FSM (C034).

Geometry: CONTEXT_COUNT is the only generated dimension (C1/C2 from the sealed
target). FRONTIER=2, OPERANDS=3 and the terminal reserve of 1 follow the C015
`make` test fixture; COUNTER_BITS=4 is deliberately narrowed so stress runs
saturate the counters; SOURCES=4 stands for the four producer classes
(engine/MFE/VME/Group). All are verification geometry, not silicon values;
identities are full 64-bit transport words.

The runner validates the nine scenario YAMLs (`verification/scenarios/w6/
u2?_*.yaml`, cases U2-a..U2-i) fail-closed, then materializes them once as
`evidence/c033-vectors.txt` — one row per case, twelve space-separated
lowercase-hex 64-bit words (seed id/generation/sequence/future_uses, then
expect armed/success/failure/consumer_refs/future_uses/phases/retire_ready/
completions), rows sorted a..i — exposed to every simulator invocation as
`+VECTORS=/out/c033-vectors.txt`. Greens are exactly tb_c033 at the C1 and
requested counts plus tb_c033_min (the same fixture with OPERANDS=2 — this
bench's configured WAITALL ceiling, matching the explicit make(...,
waitall_max=2) C015 boundary test; the sealed contract makes 2 the minimum
and leaves the maximum unresolved) at C1; mutants only run against tb_c033.
Each bench owns the behavioral comparison and emits one `C033_CASE U2-x` line per case
plus a single `C033_PASS <top> contexts=N checks=M` record bound to that
run's top and Context count; this runner never hard-codes green output. Both
C015 fragment files are hashed against the real C020 seal (whose bytes the
shared harness already fences).
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
from c033_phase_replay import verify_group_phases  # noqa: E402
from w4_rtl_harness import Bench, Slice, check_verdicts, run_slice, selected_target  # noqa: E402

SCENARIOS = H / "verification/scenarios/w6"
SEAL = H / "protocol/PROTOCOL_FREEZE.json"
FRAGMENTS = (
  H / "protocol/w6_event_facility.contract.yaml",
  H / "protocol/w6_terminal_reset.contract.yaml",
)
CASE_IDS = tuple(f"U2-{letter}" for letter in "abcdefghi")
SEED_KEYS = ("id", "generation", "sequence", "future_uses")
EXPECT_KEYS = (
  "armed",
  "success",
  "failure",
  "consumer_refs",
  "future_uses",
  "phases",
  "retire_ready",
  "completions",
)
PASS_RE = re.compile(r"^C033_PASS (\S+) contexts=(\d+) checks=(\d+)$", re.M)

BENCH = Bench(
  top="tb_c033",
  sources=(
    "c033_event_pkg.sv",
    "wait_record.sv",
    "completion_bus.sv",
    "event_table.sv",
    "terminal_store.sv",
    "tb/tb_c033.sv",
  ),
  contexts=(1, 2),
  run_args=("+VECTORS=/out/c033-vectors.txt",),
)

# Same sources and plusargs, C1 only: the tb_c033_min wrapper instantiates the
# fixture with OPERANDS=2 — this bench's configured WAITALL ceiling, matching
# the explicit make(..., waitall_max=2) C015 boundary test; the sealed
# contract makes 2 the WAITALL minimum and leaves the maximum unresolved
# (owner C018), so this is not a C015 architectural maximum. Mutants only run
# against BENCH.
MIN_BENCH = Bench(top="tb_c033_min", sources=BENCH.sources, contexts=(1,),
                   run_args=BENCH.run_args)


def load_scenarios() -> list[dict[str, Any]]:
  """Validate exactly the nine U2-a..U2-i seed+expect scenario rows, sorted a..i."""
  rows = w6_scenario_io.load_scenarios(SCENARIOS, "u2?_*.yaml", CASE_IDS, SEED_KEYS, EXPECT_KEYS)
  for row in rows:  # C033-only: armed is a 0/1 flag, not a counter
    if row["expect"]["armed"] not in (0, 1):
      raise AssertionError(f"{row['file']}: expect.armed must be 0 or 1, "
                           f"got {row['expect']['armed']}")
  return rows


def vectors(scenarios: Sequence[dict[str, Any]]) -> str:
  """One row per case: twelve lowercase-hex words, four seed then eight expected."""
  return w6_scenario_io.vectors(scenarios, SEED_KEYS, EXPECT_KEYS)


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
        "seed": row["seed"],
        "expect": row["expect"],
      }
      for row in scenarios
    ]
  }

  @contextmanager
  def serve(out: Path, _labels: Sequence[str]) -> Iterator[None]:
    (out / "c033-vectors.txt").write_text(rows)
    (out / "c033-scenarios.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (out / "c015-fragments.json").write_text(json.dumps(fragments, indent=2, sort_keys=True) + "\n")
    yield

  return run_slice(
    Slice(
      name="c033w6",
      benches=(BENCH, MIN_BENCH),
      mutant_globs=("event-*.patch",),
      mutant_bench=BENCH,
      min_mutants=9,
      assert_markers=("C033_ASSERT",),
      service=serve,
      host_sources=(
        W6 / "test_w6_c033.py",
        W6 / "c033_phase_replay.py",
        W6 / "w6_scenario_io.py",
        W4 / "w4_rtl_harness.py",
        REPO / "pipeline_validator/cosim/event_service.py",
        REPO / "pipeline_validator/tile_group.py",
        REPO / "pipeline_validator/tests/test_runtime.py",
        *FRAGMENTS,
        *(row["file"] for row in scenarios),
      ),
      rtl_dir=H / "rtl/w6",
      mutants_dir=H / "verification/unit/w6/mutants",
    ),
    tmp_path_factory,
  )


def test_persistent_events_wait_and_phase_signals(evidence: dict[str, Any]) -> None:
  check_verdicts(evidence, pass_marker="C033_PASS")
  path = evidence["path"]
  # C1/C2 comes from the real sealed config, not from this runner.
  params = memory_params.parameters()
  requested = int(selected_target()["constraints"]["context_count"])
  counts = sorted({int(params["C1_CONTEXTS"]), requested})
  pairs = {(case["top"], case["context_count"]) for case in evidence["positive"]}
  assert pairs == {(BENCH.top, count) for count in counts} | {(MIN_BENCH.top, counts[0])}
  for case in evidence["positive"]:
    log = (path / f"{case['label']}.log").read_text()
    records = [line.split() for line in log.splitlines() if line.startswith("C033_CASE")]
    assert all(len(r) == 2 and r[1] in CASE_IDS for r in records), records
    ids = [record[1] for record in records]
    assert sorted(ids) == list(CASE_IDS), f"{case['label']}: case records {ids}"
    passes = PASS_RE.findall(log)
    assert len(passes) == 1, f"{case['label']}: expected one C033_PASS record, got {passes}"
    top, contexts, checks = passes[0]
    assert (top, int(contexts)) == (case["top"], case["context_count"]), (case["label"], passes)
    assert int(checks) >= 1, f"{case['label']}: PASS record executed no checks"
    verify_group_phases(log)
