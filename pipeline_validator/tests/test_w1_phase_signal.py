"""C035: all_tasks releases only actual pure-read pins; L1/R/terminal ownership is independent."""

from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline_validator.compiled_program import seal_program
from pipeline_validator.config import SimConfig

from pipeline_validator.cosim import abi
from pipeline_validator.cosim.tile_port import BoundaryMessage
from pipeline_validator.execution_ir import ExecGroupActionOp, ExecTileGroupTask
from pipeline_validator.loader import load_program
from pipeline_validator.memory import MemoryInvariantError
from pipeline_validator.tests.w1_lifecycle import ObservedNative, native_cases, start


def phase_case(reads: bool, fault: str | None = None) -> dict:
  with ExitStack() as stack:
    ports = {}
    for i in range(4):
      port = ObservedNative(
        i, ready_seed=3 + i, stub_flags=abi.CN_STUB_FLAG_LIFECYCLE | abi.CN_STUB_FLAG_INPUT_RELEASED
      )
      stack.callback(port.close)
      ports[i] = port
    run = start(participants=4, ports=ports, reads=reads)
    g = run.group
    assert not g.tiles  # The Python Group has no shadow L1/engine execution owner.
    for _ in range(500):
      run.tick()
      for port in ports.values():
        if port.messages("RESOURCE_UPDATE"):
          port.set_t2g_ready("EVENT", False)
      if len(g.admission.tickets) == 4 and all(t.accepted for t in g.admission.tickets.values()):
        if all(p.messages("RESOURCE_UPDATE") for p in ports.values()):
          break
    else:
      pytest.fail("not all four physical holds became accepted Tasks")
    (route,) = g._grid_routes.values()
    key = route.grid
    assert run.leases == 4 and run.pins == (4 if reads else 0)
    assert not any(p.messages("PHASE_SIGNAL") for p in ports.values())
    # A genuine ACK replayed on another physical link is not a duplicate from the owning Tile.
    other_ack = next(
      m for m in ports[0].messages("ACK") if m.fields["ack_for_message_type"] == "TASK_COMMIT"
    )
    with pytest.raises(MemoryInvariantError):
      g.admission.handle(1, other_ack, run.cycle)
    for port in ports.values():
      port.set_t2g_ready("CTRL", False)
    for tid in (0, 1, 2):
      ports[tid].set_t2g_ready("EVENT", True)
    run.until(lambda: len(g._grid_signals[key].seen.get("input_released", ())) == 3)
    assert not g._grid_signals[key].completed_phases and run.pins == (4 if reads else 0)
    assert run.leases == 4 and not route.retired

    if fault == "duplicate":
      # Inject another valid instruction's identical value (not a transport sequence replay).
      (first,) = ports[0].messages("PHASE_SIGNAL")
      again = BoundaryMessage.from_wire(
        BoundaryMessage(
          "PHASE_SIGNAL", "EVENT", "T2G", first.fields, message_seq=first.message_seq + 1
        ).pack_wire()
      )
      g.retirement.handle(0, again, run.cycle)
      assert g.pmu.events["tile_signal_duplicate"] == 1 and g.sequencer.faulted
      assert not g._grid_signals[key].completed_phases and run.leases == run.pins == 4
    elif fault in ("premature", "not_fired"):
      # Enqueueing RETIRE also isn't a physical grant. Stop before the very next native edge.
      t = next(t for t in g.admission.tickets.values() if t.tile_id == 0)
      if fault == "not_fired":
        ports[0].set_t2g_ready("CTRL", True)
        run.until(lambda: bool(ports[0].messages("RETIRE", sent=True)))
        (grant,) = ports[0].messages("RETIRE", sent=True)
        assert ports[0].g2t_accepted_seq("CTRL") < grant.message_seq
      premature = BoundaryMessage.from_wire(
        BoundaryMessage(
          "TASK_COMPLETION",
          "CTRL",
          "T2G",
          {**t.key.to_wire_fields(), "status": "DONE", "first_fault_ref": 0},
        ).pack_wire()
      )
      g.retirement.handle(0, premature, run.cycle)
      assert g.sequencer.faulted and g.reset_domain.is_active
      assert not route.retired and run.leases == run.pins == 4
    else:
      ports[0].set_t2g_ready("CTRL", True)
      run.until(lambda: route.retired == {0})
      assert run.leases == 3 and run.pins == (3 if reads else 0) and key in g._grid_routes
      assert not g._grid_signals[key].completed_phases and g._l2_arenas and not g.sequencer.done
      ports[3].set_t2g_ready("EVENT", True)
      run.until(lambda: g._grid_signals[key].completed_phases == {"input_released"})
      assert run.pins == 0 and run.leases == 3 and route.retired == {0}
      assert g.event_service.stats["phase_aggregates"] == 1
      assert all(not p.messages("TASK_COMPLETION") for i, p in ports.items() if i != 0)

    for port in ports.values():
      port.set_t2g_ready("CTRL", True)
      port.set_t2g_ready("EVENT", True)
    run.finish(fault=fault is not None)
    return {
      "events": g.event_service.snapshot()["stats"],
      "unclosed": g.unclosed_l2_objects(),
      "wire": [p.summary() for p in ports.values()],
    }


def test_loader_keeps_explicit_empty_phase_bound_to_the_emitting_program() -> None:
  run = start(reads=False)
  g = run.group
  artifact = g.loaded_program
  assert artifact is not None and isinstance(artifact.entry, ExecTileGroupTask)
  task = artifact.entry
  actions = list(task.actions)
  idx = next(i for i, action in enumerate(actions) if action.op is ExecGroupActionOp.DISPATCH_ROLE)
  dispatch = actions[idx]
  (req,) = dispatch.args
  # The request's policy/tag still agree with EACH OTHER, but suppress the actual instruction.
  bad = replace(req, input_released_event="", signal_policy=replace(req.signal_policy, input_released=None))
  actions[idx] = replace(dispatch, args=(bad,))
  corrupt = seal_program(replace(artifact, entry=replace(task, actions=tuple(actions))))
  with pytest.raises(ValueError, match="phase events differ from Tile Program signals"):
    load_program(corrupt, g.cfg, SimConfig(fidelity=g.fidelity, context_count=2))


def test_partial_phase_and_early_retirement_on_real_links(tmp_path: Path) -> None:
  result = native_cases("test_w1_phase_signal", tmp_path)
  for case in ("readers", "no_inputs"):
    assert result[case]["events"]["phase_aggregates"] == 1
    assert result[case]["events"]["phase_signals"] == 4 and result[case]["unclosed"] == []
  assert result["duplicate"]["events"]["phase_duplicate"] == 1


if __name__ == "__main__":
  results = {
    "readers": phase_case(True),
    "no_inputs": phase_case(False),
    "duplicate": phase_case(True, "duplicate"),
    "premature": phase_case(True, "premature"),
    "not_fired": phase_case(True, "not_fired"),
  }
  (Path(sys.argv[1]) / "c035.json").write_text(json.dumps(results, indent=2) + "\n")
