"""C038: scalar helper issue adapter decode/qualification against the real W4/W6 owners.

Unit scope: `tb_scalar_helper_issue` instantiates the real
`rtl/scalar/scalar_helper_issue` adapter driving the real `rtl/memory` mfe_top
(an SV memory peer answers the external channel) and the real `rtl/w6`
event_table + terminal_store. This is the instruction-side decode/qualify
evidence only: no full-Tile assembly (C040), no retire-chain ownership
(C033/C035), no scalar core (W2).

The adapter's CSR constants are the GENERATED C017 allocator output only:
`c038_scalar_pkg` includes `cn_offsets.svh`, which this runner stages into
the container's /out include dir (the same file the seal covers) and whose
sha256 is receipted alongside the ABI contract hash. If Protocol has not yet
regenerated the v5 layout the run fails closed at build time, never at a
guessed address.

Geometry: CONTEXT_COUNT from the sealed target (C1/C2). The bench runs one
live task owner on Context slot 0 at every count; DESC_COUNT=EVENTS=2 is the
K-SCALAR-ADD7 launch-table geometry (an ABI parameter, not silicon).
OPERANDS=4 stands for the event-table issue-head width the C040 wiring
carries; all are verification fixtures, not silicon values.

Each green run must emit exactly one `C038_PASS <top> contexts=N checks=M`
record with the observable ledger (desc=2 phases=2 rr=1 comp=1); the runner
never hard-codes bench output. Mutants die by an executed assertion after a
clean build, per the shared harness rule.
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

W5 = Path(__file__).resolve().parent
W4 = W5.parent / "w4"
H = W5.parents[2]
REPO = W5.parents[4]
sys.path.insert(0, str(W4))
sys.path.insert(0, str(H / "scripts/lib"))

import gen_memory_params as memory_params  # noqa: E402
from w4_rtl_harness import Bench, Slice, check_verdicts, run_slice, selected_target  # noqa: E402

SVH = H / "protocol/generated/cn_offsets.svh"
ABI = H / "protocol/scalar_helper_abi_v1.json"
CONTRACT = H / "protocol/w5_package.contract.yaml"

PASS_RE = re.compile(
  r"^C038_PASS (\S+) contexts=(\d+) checks=(\d+) desc=(\d+) phases=(\d+) rr=(\d+) comp=(\d+)$",
  re.M,
)

BENCH = Bench(
  top="tb_scalar_helper_issue",
  sources=(
    "memory/c029_params_pkg.sv",
    "memory/c032_mfe_pkg.sv",
    "memory/transfer_cancel_tracker.sv",
    "memory/mfe_top.sv",
    "w6/c033_event_pkg.sv",
    "w6/completion_bus.sv",
    "w6/wait_record.sv",
    "w6/event_table.sv",
    "w6/terminal_store.sv",
    "w6/c034_reset_pkg.sv",
    "scalar/c038_scalar_pkg.sv",
    "scalar/scalar_helper_issue.sv",
    "scalar/tb/tb_scalar_helper_issue.sv",
  ),
  contexts=(1, 2),
  # The generated layout header lives outside rtl/; /out is the evidence dir
  # where the service stages the canonical (sealed) copy.
  args_by_context={1: "+incdir+/out", 2: "+incdir+/out", 4: "+incdir+/out"},
  timeout_s=300,
)

HOST_SOURCES = (
  Path(__file__),
  W4 / "w4_rtl_harness.py",
  SVH,
  ABI,
  CONTRACT,
)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  @contextmanager
  def serve(out: Path, _labels: Sequence[str]) -> Iterator[None]:
    # Stage the canonical generated header into the container include dir
    # and receipt the sealed inputs the decode is built from.
    svh_bytes = SVH.read_bytes()
    (out / "cn_offsets.svh").write_bytes(svh_bytes)
    receipt = {
      "generated/cn_offsets.svh": hashlib.sha256(svh_bytes).hexdigest(),
      "protocol/scalar_helper_abi_v1.json": hashlib.sha256(ABI.read_bytes()).hexdigest(),
      "protocol/w5_package.contract.yaml": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
      "csr_names": [
        "nexus.mfe.submit",
        "nexus.event.wait",
        "nexus.phase.signal",
        "nexus.task.ret",
      ],
      "abi": "cn_scalar_helper_abi/v1",
    }
    (out / "c038-sealed-inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")
    yield

  return run_slice(
    Slice(
      name="c038decode",
      benches=(BENCH,),
      mutant_globs=("decode-*.patch",),
      mutant_bench=BENCH,
      min_mutants=9,
      assert_markers=("C038_ASSERT", "C032_ASSERT", "C033_ASSERT"),
      service=serve,
      host_sources=HOST_SOURCES,
      rtl_dir=H / "rtl",
      mutants_dir=W5 / "mutants",
    ),
    tmp_path_factory,
  )


def test_scalar_helper_issue_decode_and_routing(evidence: dict[str, Any]) -> None:
  check_verdicts(evidence, pass_marker="C038_PASS")
  path = evidence["path"]
  params = memory_params.parameters()
  requested = int(selected_target()["constraints"]["context_count"])
  counts = sorted({int(params["C1_CONTEXTS"]), requested})
  pairs = {(case["top"], case["context_count"]) for case in evidence["positive"]}
  assert pairs == {(BENCH.top, count) for count in counts}
  for case in evidence["positive"]:
    log = (path / f"{case['label']}.log").read_text()
    passes = PASS_RE.findall(log)
    assert len(passes) == 1, f"{case['label']}: expected one C038_PASS record, got {passes}"
    top, contexts, checks, desc, phases, rr, comp = passes[0]
    assert (top, int(contexts)) == (case["top"], case["context_count"]), (case["label"], passes)
    assert int(checks) >= 60, f"{case['label']}: PASS record ran too few checks"
    # The observable ledger: exactly three accepted descriptors (desc0 twice
    # to fill both INFLIGHT records for the stall, then desc1), two
    # PHASE_SIGNAL messages, one RETIRE_READY, one TASK_COMPLETION.
    assert (int(desc), int(phases), int(rr), int(comp)) == (3, 2, 1, 1), (
      case["label"],
      passes,
    )
