"""Structural invariants of the dispatch-parallel prefill comparison workload.

The example's claim is precise: the three producer programs do no load
hoisting (no fill is issued across the BOA that reads its buffer, so
``input_released`` still fires after the final L2 load) and every
``tile.await`` is sunk to the last dependency-safe point — the weight fill
co-issues with the first X fill plus the first partials behind one merged
await, each store overlaps the next independent fill, and no store ever
co-issues with an outstanding BOA.  The parallelism the producers gave up is
recovered by independent Grids — a Q chain and a K/V chain on separate
allocations plus per-half output projections.  All of that is visible in the
IR without running a simulation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pipeline_validator.dialects.elenor import (
  NestAllocOp,
  NestContextOp,
  NestDispatchOp,
  NexusProgramOp,
)
from pipeline_validator.workload_ir import load_workload_ir

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/workloads"
DISPATCH = EXAMPLES / "transformer_prefill_attention_dispatchparallel_multicontext.mlir"
BASELINE = EXAMPLES / "transformer_prefill_attention_multicontext.mlir"
TAIL_PROGRAMS = (
  "prefill_attention_q0",
  "prefill_attention_q1",
  "prefill_attention_q2",
  "prefill_attention_q3",
)
PRODUCERS = (
  "qkv_q_init",
  "qkv_q_accum",
  "qkv_kv_init",
  "qkv_kv_accum",
  "prefill_outproj_lo",
  "prefill_outproj_hi",
)
QKV_PRODUCERS = (
  "qkv_q_init",
  "qkv_q_accum",
  "qkv_kv_init",
  "qkv_kv_accum",
)
_TILE_OP = re.compile(
  r"tile\.(load|store|boa|evu)\.async|tile\.await|tile\.signal (input_released|output_ready)"
)


def _program_bodies(path: Path) -> dict[str, str]:
  text = path.read_text()
  bodies: dict[str, str] = {}
  for match in re.finditer(r"^  tile\.program @(\w+)\(", text, flags=re.MULTILINE):
    start = match.start()
    following = text.find("\n  tile.program @", start + 1)
    end = following if following != -1 else text.find("\n  nest.context @", start)
    bodies[match.group(1)] = text[start:end]
  return bodies


def _tile_ops(program_body: str) -> list[str]:
  """Tile async op kinds in program order (load/store/boa/evu/await/signal_*)."""
  ops: list[str] = []
  for match in _TILE_OP.finditer(program_body):
    if match.group(0).startswith("tile.await"):
      ops.append("await")
    else:
      ops.append(match.group(1) or match.group(2))
  return ops


def _normalize(body: str) -> str:
  return re.sub(r"\s+", " ", re.sub(r"%\w+", "%v", body)).strip()


def _context(module) -> NestContextOp:
  return next(op for op in module.ops if isinstance(op, NestContextOp))


def _slot_of(operand) -> str:
  owner = getattr(operand, "owner", operand)
  assert isinstance(owner, NestAllocOp)
  return owner.slot.data


def _dispatches(module, program: str) -> list[NestDispatchOp]:
  context = _context(module)
  return [
    op
    for op in context.body.block.ops
    if isinstance(op, NestDispatchOp) and op.program.data == program
  ]


@pytest.fixture(scope="module")
def dispatch_module():
  return load_workload_ir(DISPATCH)


@pytest.mark.parametrize("program", PRODUCERS)
def test_producers_issue_the_first_fills_before_any_await(dispatch_module, program):
  """The weight fill co-issues with the first X fill (plus first partials)."""
  ops = _tile_ops(_program_bodies(DISPATCH)[program])
  first_await = ops.index("await")
  loads_before = sum(1 for kind in ops[:first_await] if kind == "load")
  assert loads_before >= 2, f"{program} awaits before the weight and X fills overlap"


@pytest.mark.parametrize("program", PRODUCERS)
def test_producers_keep_fill_bursts_bounded(dispatch_module, program):
  """Sinking still honors the single-buffer WAR chain: at most five fills."""
  in_flight = 0
  for kind in _tile_ops(_program_bodies(DISPATCH)[program]):
    if kind == "load":
      in_flight += 1
      assert in_flight <= 5, f"{program} has {in_flight} loads in flight"
    elif kind == "await":
      in_flight = 0


@pytest.mark.parametrize("program", PRODUCERS)
def test_producers_never_issue_a_store_under_an_outstanding_boa(dispatch_module, program):
  """RAW on the accumulator: a store waits for its own BOA's await."""
  outstanding_boas = 0
  for kind in _tile_ops(_program_bodies(DISPATCH)[program]):
    if kind == "boa":
      outstanding_boas += 1
    elif kind == "store":
      assert outstanding_boas == 0, f"{program} stores under an outstanding BOA"
    elif kind == "await":
      outstanding_boas = 0


@pytest.mark.parametrize("program", QKV_PRODUCERS)
def test_producers_co_issue_every_store_with_the_next_fill(dispatch_module, program):
  """The sunk stores overlap a later independent fill (the old drain is gone)."""
  overlapped = False
  pending_store = False
  for kind in _tile_ops(_program_bodies(DISPATCH)[program]):
    if kind == "store":
      pending_store = True
    elif kind == "await":
      pending_store = False
    elif kind == "load" and pending_store:
      overlapped = True
  assert overlapped, f"{program} never overlaps a store with a later fill"


@pytest.mark.parametrize("program", PRODUCERS)
def test_input_released_fires_after_the_final_load(dispatch_module, program):
  ops = _tile_ops(_program_bodies(DISPATCH)[program])
  loads = [index for index, kind in enumerate(ops) if kind == "load"]
  assert loads, f"{program} never loads"
  release = next(index for index, kind in enumerate(ops) if kind == "input_released")
  assert release > loads[-1]


def test_q_chain_and_kv_chain_are_independent_grids(dispatch_module):
  """Q Grids write q_l2 only, K/V Grids write k_l2/v_l2 only: no cross edges."""
  q = _dispatches(dispatch_module, "qkv_q_accum") + _dispatches(dispatch_module, "qkv_q_init")
  kv = _dispatches(dispatch_module, "qkv_kv_accum") + _dispatches(dispatch_module, "qkv_kv_init")
  assert len(q) == 8 and len(kv) == 8
  for family, slots in ((q, {"q_l2"}), (kv, {"k_l2", "v_l2"})):
    for dispatch in family:
      assert {_slot_of(o) for o in dispatch.outs} == slots
      for dependency in dispatch.depends_on:
        owner = getattr(dependency, "owner", dependency)
        if isinstance(owner, NestDispatchOp):
          other = owner.program.data
          if family is q:
            assert not other.startswith("qkv_kv"), "a Q Grid waits on a K/V Grid"
          else:
            assert not other.startswith("qkv_q"), "a K/V Grid waits on a Q Grid"


def test_outproj_halves_are_independent_grids(dispatch_module):
  lo = _dispatches(dispatch_module, "prefill_outproj_lo")
  hi = _dispatches(dispatch_module, "prefill_outproj_hi")
  assert len(lo) == 4 and len(hi) == 4
  for lo_dispatch, hi_dispatch in zip(lo, hi, strict=True):
    lo_outs = {_slot_of(o) for o in lo_dispatch.outs}
    hi_outs = {_slot_of(o) for o in hi_dispatch.outs}
    assert lo_outs.isdisjoint(hi_outs), "row halves share an output buffer"


def test_attention_tail_matches_the_pipelined_baseline():
  baseline = _program_bodies(BASELINE)
  staged = _program_bodies(DISPATCH)
  for name in TAIL_PROGRAMS:
    assert _normalize(baseline[name]) == _normalize(staged[name]), f"{name} diverged"


def test_logical_shapes_and_dispatch_count(dispatch_module):
  entry = next(op for op in dispatch_module.ops if isinstance(op, NexusProgramOp))
  assert [arg.name_hint for arg in entry.body.block.args] == ["X", "WQ", "WK", "WV", "WO", "OUT"]
  context = _context(dispatch_module)
  total = sum(1 for op in context.body.block.ops if isinstance(op, NestDispatchOp))
  # 16 QKV + 4 attention + 8 outproj Grids.
  assert total == 28
