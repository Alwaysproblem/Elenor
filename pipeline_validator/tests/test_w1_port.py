"""C022 / W1-02 TilePort seam unit tests (pure Python; no C ABI, no Verilator).

Covers the W1 plan §8.3 unit layer:
- `TilePort` protocol structural consistency via the unit-only `TileProtocolStub`
  (`backend_kind == "stub"`; `assert_cs1_backend_kind` rejects it for CS1 evidence).
- `BoundaryMessage` wire round-trip through the generated `wire.py`.
- Pure-value boundary guard: runtime objects (`ComputeTile`, `TileUCE`, ...) raise TypeError.
- `PythonTileBackend.send()` does NOT advance the Tile (`pmu.total` unchanged); only
  `step(cycle)` advances, exactly once per cycle (double-step raises).
- A `TASK_PLAN` from a retired `launch_generation` is rejected (`stale_plan_rejected` += 1)
  and answers `ACK(outcome=STALE)` with zero Tile state mutation; a large instruction
  `op_seq_hint` never substitutes for launch identity.
- Channel credit conservation (`available + in_flight + occupancy == capacity`) holds on the
  happy path and is broken loudly on underflow/duplicate-return.
"""

from __future__ import annotations

import pytest

from pipeline_validator.config import HardwareConfig
from pipeline_validator.cosim import (
  BoundaryMessage,
  ChannelCreditLedger,
  CreditConservationError,
  LinkCreditPool,
  PythonTileBackend,
  TaskKey,
  TilePort,
  TileProtocolStub,
  assert_cs1_backend_kind,
  validate_boundary_value,
  wire,
)
from pipeline_validator.tile import ComputeTile


def _task_key(launch_generation: int = 1, task_id: int = 3) -> TaskKey:
  return TaskKey(
    launch_generation=launch_generation,
    root_launch_generation=launch_generation,
    dispatch_ordinal=0,
    task_id=task_id,
    context_handle=0,
  )


def _task_plan(launch_generation: int, txn_id: int = 9, seq: int = 0) -> BoundaryMessage:
  return BoundaryMessage(
    message_type="TASK_PLAN",
    channel="CTRL",
    direction="G2T",
    fields={
      "launch_generation": launch_generation,
      "root_launch_generation": launch_generation,
      "dispatch_ordinal": 0,
      "task_id": 3,
      "context_handle": 0,
      "txn_id": txn_id,
      "pool_epoch": 1,
      "pool_version": 1,
    },
    message_seq=seq,
  )


def _make_tile() -> ComputeTile:
  return ComputeTile(0, HardwareConfig(), runtime_enabled=True, memory_enabled=True, context_count=2)


def test_tile_port_protocol_structural_consistency_via_stub() -> None:
  stub = TileProtocolStub(tile_id=3)
  assert isinstance(stub, TilePort)
  assert stub.backend_kind == "stub"
  assert stub.tile_id == 3
  with pytest.raises(ValueError):
    assert_cs1_backend_kind(stub.backend_kind)
  assert_cs1_backend_kind("python")
  assert_cs1_backend_kind("rtl")
  with pytest.raises(ValueError):
    assert_cs1_backend_kind("quantum")


def test_boundary_message_wire_round_trip() -> None:
  msg = _task_plan(launch_generation=2, txn_id=7, seq=4)
  blob = msg.pack_wire()
  port = (wire.CHANNELS.index("CTRL"), wire.DIRS.index("G2T"))
  back = BoundaryMessage.from_wire(blob, expected_port=port)
  assert back.message_type == "TASK_PLAN"
  assert back.channel == "CTRL" and back.direction == "G2T"
  assert back.message_seq == 4
  for name in (
    "launch_generation",
    "root_launch_generation",
    "dispatch_ordinal",
    "task_id",
    "context_handle",
    "txn_id",
    "pool_epoch",
    "pool_version",
  ):
    assert back.fields[name] == msg.fields[name], name
  # message/direction mismatch is rejected at construction
  with pytest.raises(ValueError):
    BoundaryMessage(message_type="TASK_PLAN", channel="EVENT", direction="T2G", fields={}, message_seq=0)


def test_runtime_objects_never_cross_the_boundary() -> None:
  validate_boundary_value(TaskKey(1, 1, 0, 0, 0))
  validate_boundary_value({"k": (1, "s", b"bytes", memoryview(b"mv"), None, True)})
  with pytest.raises(TypeError):
    validate_boundary_value(_make_tile())
  with pytest.raises(TypeError):
    validate_boundary_value({"tile": _make_tile()})
  with pytest.raises(TypeError):
    validate_boundary_value([object()])  # list container is itself rejected
  with pytest.raises(TypeError):
    BoundaryMessage(
      message_type="TASK_PLAN",
      channel="CTRL",
      direction="G2T",
      fields={"launch_generation": _make_tile()},
      message_seq=0,
    )


def test_send_does_not_advance_and_step_advances_exactly_once() -> None:
  backend = PythonTileBackend(_make_tile())
  before = backend.tile.pmu.named_cycles.get("total", 0) + sum(backend.tile.pmu.events.values())
  backend.send(_task_plan(launch_generation=1, txn_id=5, seq=0))
  assert backend.tile.pmu.named_cycles.get("total", 0) + sum(
    backend.tile.pmu.events.values()
  ) == before, "send() must not advance the Tile"
  assert backend._outbox, "send() should enqueue the G2T message"
  backend.step(0)
  assert backend.steps_run == 1
  with pytest.raises(RuntimeError):
    backend.step(0)
  backend.step(1)


def test_stale_launch_generation_plan_rejected_with_zero_side_effects() -> None:
  backend = PythonTileBackend(_make_tile())
  backend.advance_launch_watermark(5)
  snapshot_before = backend.tile.snapshot()
  backend.send(_task_plan(launch_generation=4, txn_id=11, seq=0))
  backend.step(10)
  assert backend.stale_plan_rejected == 1
  assert backend.accepted_plans == {}
  assert backend.poll_inbound()
  ack = backend.pop_inbound()
  assert ack is not None and ack.message_type == "ACK"
  assert ack.fields["outcome"] == "STALE"
  assert ack.fields["ack_for_message_type"] == "TASK_PLAN"
  assert ack.fields["txn_id"] == 11
  assert backend.tile.snapshot()["uce"] == snapshot_before["uce"]
  assert backend.credit_invariant_holds()


def test_op_seq_hint_never_substitutes_for_launch_identity() -> None:
  backend = PythonTileBackend(_make_tile())
  backend.advance_launch_watermark(7)
  stale = _task_plan(launch_generation=3, txn_id=12, seq=0)
  fresh_fields = dict(stale.fields)
  stale_with_big_opseq = BoundaryMessage(
    message_type="TASK_PLAN",
    channel="CTRL",
    direction="G2T",
    fields=fresh_fields,
    message_seq=0,
    op_seq_hint=0xFFFFFFFF,
  )
  backend.send(stale_with_big_opseq)
  backend.step(0)
  assert backend.stale_plan_rejected == 1
  assert backend.accepted_plans == {}
  _ = stale  # silence linters; the point is op_seq_hint cannot rescue the stale launch


def test_credit_ledger_conservation_and_negatives() -> None:
  pool = LinkCreditPool(channel="CTRL", direction="G2T", capacity=2)
  pool.reserve_on_send()
  pool.accept_into_receiver()
  assert pool.snapshot() == {"capacity": 2, "available": 1, "in_flight": 0, "occupancy": 1}
  pool.consume_and_return()
  assert pool.available == 2 and pool.occupancy == 0
  with pytest.raises(CreditConservationError):
    pool.consume_and_return()  # duplicate/unmatched return
  full = LinkCreditPool(channel="CTRL", direction="G2T", capacity=1)
  full.reserve_on_send()
  with pytest.raises(CreditConservationError):
    full.reserve_on_send()  # over-subscription
  empty_accept = LinkCreditPool(channel="EVENT", direction="G2T", capacity=4)
  with pytest.raises(CreditConservationError):
    empty_accept.accept_into_receiver()  # accept without in-flight
  ledger = ChannelCreditLedger({"CTRL": 3})
  assert ledger.conserved()
  ledger.pool("CTRL", "G2T").reserve_on_send()
  assert ledger.conserved()
  with pytest.raises(ValueError):
    ledger.pool("NOPE", "G2T")


def test_stub_round_trip_and_stale_rejection_keep_credits_conserved() -> None:
  stub = TileProtocolStub(tile_id=1)
  blob = stub.send(_task_plan(launch_generation=2, txn_id=21, seq=0))
  assert isinstance(blob, bytes) and blob
  stub.step(0)
  assert stub.poll_inbound()
  ack = stub.pop_inbound()
  assert ack is not None and ack.fields["outcome"] == "ACCEPTED"
  assert stub.credit_invariant_holds()
  stub.send(_task_plan(launch_generation=1, txn_id=22, seq=1))
  stub.step(1)
  assert stub.stale_rejected == 1
  assert stub.credit_invariant_holds()
  stub.reset()
  assert stub.credit_invariant_holds() and stub.steps_run == 2  # reset clears queues, not history
