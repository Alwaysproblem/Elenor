"""Replay real RTL PHASE_SIGNAL handshakes into the C024 Group owning the grid."""

from __future__ import annotations

from pipeline_validator.cosim import wire
from pipeline_validator.cosim.event_service import PHASE_DIRECTIONS, EventService
from pipeline_validator.cosim.tile_port import BoundaryMessage
from pipeline_validator.execution_ir import TaskIdentity
from pipeline_validator.tests.test_runtime import TestGridSignalAggregation, fault_drain_and_reset


def verify_group_phases(log: str) -> None:
  traces: dict[str, list[tuple[int, ...]]] = {"U2-f": [], "U2-g": []}
  for line in log.splitlines():
    if line.startswith("C033_PHASE "):
      _, case, *words = line.split()
      assert len(words) == 7, line  # five TaskKey fields, direction, transport sequence
      traces[case].append(tuple(int(word, 16) for word in words))

  channel = wire.CHANNELS[wire.MESSAGES[wire.MSG_INDEX["PHASE_SIGNAL"]]["channel"]]
  for case, rows in traces.items():
    group, sequencer, grid, state = TestGridSignalAggregation._dispatch_and_get_grid()
    service = EventService(group)
    # Functional replay at the live dispatch point: message order matters;
    # Group simulation ticks are not synchronized with bench clock edges.
    cycle = group._last_step_cycle
    received: dict[str, set[int]] = {direction: set() for direction in PHASE_DIRECTIONS}
    ids = set(state.expected_task_ids)
    expected_messages = 2 if case == "U2-f" else 2 * len(ids)
    assert len(rows) == expected_messages, (case, rows)
    assert len({row[-1] for row in rows}) == expected_messages
    for i, (launch, root_gen, ordinal, tid, handle, direction, seq) in enumerate(rows):
      # Resolve the transmitted TaskKey. Do not replace a stale RTL field with
      # the fixture's identity to make an otherwise rejected message succeed.
      assert (launch, root_gen, ordinal, handle) == (
        grid.launch_generation, grid.launch_generation, grid.dispatch_ordinal, grid.device_slot
      )
      assert tid in ids and direction in (0, 1)
      phase = PHASE_DIRECTIONS[direction]
      fields = {
        "launch_generation": launch, "root_launch_generation": root_gen, "dispatch_ordinal": ordinal,
        "task_id": tid, "context_handle": handle, "direction": phase,
      }
      msg = BoundaryMessage(
        message_type="PHASE_SIGNAL", channel=channel, direction="T2G", fields=fields, message_seq=seq
      )
      verdict = service.handle_phase_signal(msg, grid=TaskIdentity(grid, tid), cycle=cycle)
      if case == "U2-f":
        assert tid == 0 and phase == "input_released"
        assert (verdict == "duplicate") == (i == 1)
        assert not state.completed_phases
      else:
        assert tid not in received[phase]
        received[phase].add(tid)
        # A phase must neither fire early nor wait after the last contributor.
        for name, seen in received.items():
          assert (name in state.completed_phases) == (seen == ids)
          assert (state.phase_event_ids[name] in sequencer._events_done) == (seen == ids)
    assert group.pmu.events.get("tile_signal_duplicate", 0) == (1 if case == "U2-f" else 0)
    assert sequencer.faulted == (case == "U2-f")
    fault_drain_and_reset(group)
