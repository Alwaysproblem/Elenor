"""C034 Group-side regressions: real reset state cannot outrun an attached Tile."""

from __future__ import annotations

from itertools import count

import pytest

from pipeline_validator.cosim import wire
from pipeline_validator.cosim.reset_phase import ResetPhaseError
from pipeline_validator.cosim.tile_port import BoundaryMessage
from pipeline_validator.memory.allocator import MemoryInvariantError
from pipeline_validator.runtime.event_table import EventStatus
from pipeline_validator.runtime.fault_ring import FaultCode, FaultDomain
from pipeline_validator.runtime.reset_domain import ResetRequest, ResetState
from pipeline_validator.tests.test_runtime import make_sim

GEN = (1 << 63) - 1
PHASES = tuple(wire.ENUMS["reset_phase"])
DRAIN_FIELDS = tuple(wire.BITSETS["drain_evidence"])
REJECT = (ResetPhaseError, MemoryInvariantError, ValueError, wire.WireError)


def from_tile(kind: str, **fields: object) -> BoundaryMessage:
  # Exercise the actual C017 integer/enum transport, not a Python-only DTO.
  outbound = BoundaryMessage(message_type=kind, channel="CTRL", direction="T2G", fields=fields)
  return BoundaryMessage.from_wire(outbound.pack_wire())


def ack(phase: str, *, epoch: int = GEN, marked: int = 0, mask: int = 0) -> BoundaryMessage:
  return from_tile(
    "RESET_PHASE_ACK", phase=phase, run_generation=epoch, pending_to_reset=marked, evidence_mask=mask
  )


def fault(tile: int) -> BoundaryMessage:
  return from_tile("FAULT_REPORT", tile_id=tile, ctx_slot=0, ctx_generation=GEN + 1,
                   code="ADDRESS_FAULT", severity=2, source_id=8, engine_id=3, op_seq=GEN + 2)


def test_group_waits_for_each_real_link_and_valid_drain_evidence() -> None:
  group = make_sim().group
  ticks = count()
  while not group.profile_controller.initialized:
    group.profile_controller.step(next(ticks))
  domain = group.reset_domain
  tiles = (0, 1)
  for tile in tiles:
    domain.attach_tile_reset(tile, GEN)
  with pytest.raises(REJECT):
    domain.observe_tile_reset_ack(0, ack("FREEZE_ISSUE"))  # no request from this epoch yet
  group.event_table.register("group-pending")
  index = group.accept_tile_fault(fault(1), cycle=next(ticks))
  assert index >= 0 and domain.request is not None and domain.request.domain is FaultDomain.GROUP
  record = group.fault_ring.latest()
  assert record is not None and record.fault_record_index == index
  assert (record.context_id, record.aux1, record.aux0) == (0, GEN + 1, GEN + 2)
  assert (record.tile_id, record.code, record.severity, record.source, record.engine_id) == (
    1, FaultCode.ADDRESS_FAULT, 2, 8, 3
  )

  for phase in PHASES:
    received = {}
    for _ in range(16):
      group.step(next(ticks))
      for tile in tiles:
        req = domain.take_tile_reset_request(tile)
        if req is not None:
          assert tile not in received and req.message_type == "RESET_PHASE_REQ"
          received[tile] = BoundaryMessage.from_wire(req.pack_wire()).fields
      if len(received) == len(tiles):
        break
    assert set(received) == set(tiles)
    for fields in received.values():
      assert fields["phase"] in (phase, PHASES.index(phase)) and fields["run_generation"] == GEN
    if phase == "DRAIN":
      # Every individual missing closure bit must reject without consuming the real request.
      for bit in range(len(DRAIN_FIELDS)):
        with pytest.raises(REJECT):
          domain.observe_tile_reset_ack(0, ack(phase, mask=0x3F ^ (1 << bit)))
      with pytest.raises(REJECT):
        domain.observe_tile_reset_ack(0, ack(phase, mask=0x3F, epoch=GEN ^ (1 << 32)))
    if phase != "MARK_EVENTS":
      with pytest.raises(REJECT):
        domain.observe_tile_reset_ack(0, ack(phase, marked=1, mask=0x3F if phase == "DRAIN" else 0))
    if phase == "MARK_EVENTS":
      with pytest.raises(REJECT):
        # Direct callers can also present a DTO that never survived wire.pack.
        domain.observe_tile_reset_ack(0, BoundaryMessage(
          "RESET_PHASE_ACK", "CTRL", "T2G",
          {"phase": phase, "run_generation": GEN, "pending_to_reset": 1 << 64, "evidence_mask": 0},
        ))
    if phase != "DRAIN":
      with pytest.raises(REJECT):
        domain.observe_tile_reset_ack(0, ack(phase, mask=1))
    mask = 0x3F if phase == "DRAIN" else 0
    first = ack(phase, mask=mask, marked=5 if phase == "MARK_EVENTS" else 0)
    domain.observe_tile_reset_ack(0, first)
    with pytest.raises(REJECT):
      domain.observe_tile_reset_ack(0, first)  # cannot stand in for tile 1
    held_state = domain.state
    delay = 105 if phase == "DRAIN" else 3
    for _ in range(delay):
      group.step(next(ticks))
      assert domain.state is held_state
      assert domain.take_tile_reset_request(0) is None and domain.take_tile_reset_request(1) is None
    if phase == "DRAIN":
      assert held_state is ResetState.DRAIN_SAFE
      assert domain.drain_cycles > 100  # the Group watchdog ran; it did not manufacture closure
      assert group.poisoned_reason is None, "a delayed control ACK is not a failed isolation"
      pending = group.event_table.get("group-pending")
      assert pending is not None and pending.status is EventStatus.PENDING
    domain.observe_tile_reset_ack(1, ack(phase, mask=mask, marked=2 if phase == "MARK_EVENTS" else 0))

  for _ in range(3):
    group.step(next(ticks))
    if domain.is_done:
      break
  settled = group.event_table.get("group-pending")
  assert domain.is_done and settled is not None and settled.status is EventStatus.RESET
  for tile, marked in zip(tiles, (5, 2), strict=True):
    obs = domain.tile_reset_observation(tile)
    assert tuple(obs["phases"]) == PHASES and obs["pending_to_reset"] == marked
    assert obs["is_quiescent_snapshot"] == dict.fromkeys(DRAIN_FIELDS, True)
  assert not group.unclosed_l2_objects()
  # These bytes can be legitimately replayed by an idempotent RTL endpoint.
  # They cannot attest to a second drain of this already-finished epoch.
  with pytest.raises(MemoryInvariantError):
    group.note_tile_isolation_failure(1, GEN)
  with pytest.raises(MemoryInvariantError):
    domain.begin(ResetRequest(domain=FaultDomain.GROUP), next(ticks))
  with pytest.raises(REJECT):
    domain.observe_tile_reset_ack(1, ack("DRAIN", mask=0x3F))
  assert domain.is_done and tuple(domain.tile_reset_observation(1)["phases"]) == PHASES
  group.reset()
  for replay in (GEN - 1, GEN):
    with pytest.raises(ValueError):
      domain.attach_tile_reset(0, replay)
  domain.attach_tile_reset(0, GEN + 1)
  with pytest.raises(REJECT):
    domain.observe_tile_reset_ack(0, ack("DRAIN", mask=0x3F))
  assert not domain.is_active


def test_quiet_group_cannot_clear_a_tile_isolation_failure_without_tile_closure() -> None:
  group = make_sim().group
  ticks = count()
  while not group.profile_controller.initialized:
    group.profile_controller.step(next(ticks))
  group.reset_domain.attach_tile_reset(0, GEN)
  group.note_tile_isolation_failure(0, GEN)
  assert group.transfer_manager.inflight_count == 0 and group.poisoned_reason
  # No Python work is live, but the external Tile has not even acked DRAIN.
  with pytest.raises(MemoryInvariantError):
    group.reset()
  assert group.poisoned_reason
  assert group.reset_domain.tile_reset_observation(0)["is_quiescent_snapshot"] is None


def test_isolation_failure_poisons_only_the_bound_run_without_clearing_work() -> None:
  group = make_sim().group
  domain = group.reset_domain
  domain.attach_tile_reset(1, GEN)
  with pytest.raises(REJECT):
    group.note_tile_isolation_failure(1, GEN - 1)
  assert group.poisoned_reason is None
  assert group.accept_tile_fault(fault(1), cycle=3) >= 0
  group.note_tile_isolation_failure(1, GEN)
  group.note_tile_isolation_failure(1, GEN)  # retry, not a second incident
  assert group.pmu.events["tile_isolation_failure"] == 1
  with pytest.raises(MemoryInvariantError):
    domain.reset()  # discarding the link cannot stand in for Tile closure
  assert group.poisoned_reason and domain.is_active
  with pytest.raises(MemoryInvariantError, match="poisoned"):
    group.begin_launch(None, None)
  with pytest.raises(MemoryInvariantError):
    group.reset()
  assert group.poisoned_reason and domain.is_active and not domain.is_done
