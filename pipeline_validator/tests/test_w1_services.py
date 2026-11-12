"""C024 outer-service tests: caps/program/event/reset/stream driven through real APIs.

Behavioral results, not wiring:
- PHASE_SIGNAL aggregation fires exactly once through `EventService` (preserved `_on_phase_signal`
  semantics); a duplicate adds exactly +1 to `tile_signal_duplicate`;
- unauthorized POST_EVENT is rejected and counted; an authorized producer is accepted;
- PROGRAM install completes ONLY on the Tile's INSTALL_ACK with verified len/hash; read chunks
  serve the registered image bytes exactly;
- RESET_PHASE mapping round-trips and unsolicited ACKs are rejected;
- STREAM sub-queue backpressure is independent of the MEMORY main queue;
- CAPS_RESP reports live configuration values.
"""

from __future__ import annotations

import hashlib

import pytest
from test_runtime import POW_BINDINGS, fault_drain_and_reset, finish_group_and_reset, prepare_group_source

from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.cosim.caps import CapsService, CapsServiceError
from pipeline_validator.cosim.event_service import EventService, EventServiceError
from pipeline_validator.cosim.program_service import ProgramImage, ProgramService, ProgramServiceError
from pipeline_validator.cosim.reset_phase import ResetPhaseError, ResetPhaseService
from pipeline_validator.cosim.stream_subqueue import (
  DEFAULT_STREAM_DEPTH,
  StreamSubqueue,
  StreamSubqueueError,
)
from pipeline_validator.cosim.tile_port import BoundaryMessage
from pipeline_validator.execution_ir import TaskIdentity
from pipeline_validator.simulator import Simulator
from pipeline_validator.workloads import PowWorkload


def _msg(message_type: str, fields: dict, direction: str = "T2G") -> BoundaryMessage:
  from pipeline_validator.cosim import wire

  spec = wire.MESSAGES[wire.MSG_INDEX[message_type]]
  return BoundaryMessage(
    message_type=message_type,
    channel=wire.CHANNELS[spec["channel"]],
    direction=direction,
    fields=fields,
    message_seq=0,
  )


def _dispatched_group():
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  sim_config = SimConfig(fidelity="runtime", max_cycles=200000)
  sim = Simulator(hw, sim_config)
  group = sim.group
  workload = PowWorkload(num_group_chunks=1, hw=hw)
  artifact = prepare_group_source(group, workload.module, POW_BINDINGS, sim=sim_config)
  group.load_task(artifact.entry, input_bindings=POW_BINDINGS)
  for c in range(5000):
    group.step(c)
    if group._grid_signals:
      break
  assert group._grid_signals
  grid = next(iter(group._grid_signals))
  state = group._grid_signals[grid]
  return group, grid, state


# ---------------------------------------------------------------- caps
def test_caps_response_reports_live_configuration() -> None:
  svc = CapsService(context_count=2, program_slots=4, stream_supported=False)
  resp = svc.handle(_msg("QUERY_CAPS", {"txn_id": 5}, direction="G2T"))
  assert resp.message_type == "CAPS_RESP"
  assert resp.fields == {"txn_id": 5, "context_count": 2, "program_slots": 4, "stream_supported": False}
  assert svc.stats == {"queries": 1, "responses": 1}
  with pytest.raises((CapsServiceError, ValueError)):
    svc.handle(_msg("QUERY_CAPS", {"txn_id": 6}, direction="T2G"))


# ---------------------------------------------------------------- program
def _image(text: bytes = b"PROG" * 32) -> ProgramImage:
  return ProgramImage(
    program_id=1, version=1, package_hash=hashlib.sha256(text).digest(), program_epoch=0, text=text
  )


def _img_fields(img: ProgramImage) -> dict:
  return {
    "program_id": img.program_id,
    "version": img.version,
    "package_hash": img.package_hash,
    "program_epoch": img.program_epoch,
  }


def test_program_install_completes_only_on_verified_tile_ack() -> None:
  svc = ProgramService(chunk_bytes=64)
  img = _image()
  svc.register_image(img)
  install = _msg("PROGRAM_INSTALL", {**_img_fields(img), "slot": 0}, direction="G2T")
  svc.handle_install(install)
  assert not svc.install_complete(img.program_id, img.version, img.package_hash)

  served = b""
  offset = 0
  while offset < len(img.text):
    req = _msg("PROGRAM_READ_REQ", _img_fields(img))
    resp, chunk = svc.handle_read_req(req, offset=offset)
    assert resp.message_type == "PROGRAM_READ_RESP"
    assert resp.fields["hbm_chunk_len"] == len(chunk)
    served += chunk
    offset += len(chunk)
  assert served == img.text  # real image bytes served exactly, no echo

  # fake completion is impossible: no READY before ACK, len mismatch and hash mismatch rejected
  with pytest.raises(ProgramServiceError) as ready_early:
    svc.handle_ready(_msg("PROGRAM_READY", {**_img_fields(img), "local_handle": 3}, direction="G2T"))
  assert ready_early.value.code == "READY_WITHOUT_ACK"
  with pytest.raises(ProgramServiceError) as len_bad:
    svc.handle_install_ack(
      _msg(
        "PROGRAM_INSTALL_ACK",
        {
          **_img_fields(img),
          "installed_len": len(img.text) - 1,
          "measured_hash": hashlib.sha256(img.text).digest(),
        },
      )
    )
  assert len_bad.value.code == "INSTALL_LEN_MISMATCH"
  with pytest.raises(ProgramServiceError) as hash_bad:
    svc.handle_install_ack(
      _msg(
        "PROGRAM_INSTALL_ACK",
        {**_img_fields(img), "installed_len": len(img.text), "measured_hash": b"\x00" * 32},
      )
    )
  assert hash_bad.value.code == "INSTALL_HASH_MISMATCH"
  assert not svc.install_complete(img.program_id, img.version, img.package_hash)

  svc.handle_install_ack(
    _msg(
      "PROGRAM_INSTALL_ACK",
      {
        **_img_fields(img),
        "installed_len": len(img.text),
        "measured_hash": hashlib.sha256(img.text).digest(),
      },
    )
  )
  assert svc.install_complete(img.program_id, img.version, img.package_hash)
  assert (
    svc.handle_ready(_msg("PROGRAM_READY", {**_img_fields(img), "local_handle": 3}, direction="G2T")) == 3
  )
  ack = svc.handle_invalidate(_msg("PROGRAM_INVALIDATE", _img_fields(img), direction="G2T"))
  assert ack.message_type == "INVALIDATE_ACK"
  assert not svc.install_complete(img.program_id, img.version, img.package_hash)


def test_program_service_rejects_unknown_image_and_double_install() -> None:
  svc = ProgramService()
  img = _image()
  with pytest.raises(ProgramServiceError) as unknown:
    svc.handle_install(_msg("PROGRAM_INSTALL", {**_img_fields(img), "slot": 0}, direction="G2T"))
  assert unknown.value.code == "UNKNOWN_IMAGE"
  svc.register_image(img)
  svc.handle_install(_msg("PROGRAM_INSTALL", {**_img_fields(img), "slot": 0}, direction="G2T"))
  with pytest.raises(ProgramServiceError) as dup:
    svc.handle_install(_msg("PROGRAM_INSTALL", {**_img_fields(img), "slot": 1}, direction="G2T"))
  assert dup.value.code == "INSTALL_ALREADY_ARMED"


# ---------------------------------------------------------------- event
def test_phase_signal_aggregates_exactly_once_and_duplicate_counts_exactly_one() -> None:
  group, grid, state = _dispatched_group()
  svc = EventService(group)
  expected = len(state.expected_task_ids)

  for tid in range(expected - 1):
    result = svc.handle_phase_signal(
      _msg(
        "PHASE_SIGNAL",
        {
          "launch_generation": grid.launch_generation,
          "root_launch_generation": grid.launch_generation,
          "dispatch_ordinal": grid.dispatch_ordinal,
          "task_id": tid,
          "context_handle": 0,
          "direction": "input_released",
        },
      ),
      grid=TaskIdentity(grid, tid),
      cycle=1,
    )
    assert result in ("recorded", "aggregated")
  assert "input_released" not in state.completed_phases
  assert svc.stats["phase_duplicate"] == 0

  result_last = svc.handle_phase_signal(
    _msg(
      "PHASE_SIGNAL",
      {
        "launch_generation": grid.launch_generation,
        "root_launch_generation": grid.launch_generation,
        "dispatch_ordinal": grid.dispatch_ordinal,
        "task_id": expected - 1,
        "context_handle": 0,
        "direction": "input_released",
      },
    ),
    grid=TaskIdentity(grid, expected - 1),
    cycle=2,
  )
  assert "input_released" in state.completed_phases
  before = group.pmu.events.get("tile_signal_duplicate", 0)
  dup = svc.handle_phase_signal(
    _msg(
      "PHASE_SIGNAL",
      {
        "launch_generation": grid.launch_generation,
        "root_launch_generation": grid.launch_generation,
        "dispatch_ordinal": grid.dispatch_ordinal,
        "task_id": 0,
        "context_handle": 0,
        "direction": "input_released",
      },
    ),
    grid=TaskIdentity(grid, 0),
    cycle=3,
  )
  after = group.pmu.events.get("tile_signal_duplicate", 0)
  assert dup == "duplicate"
  assert after - before == 1  # exactly +1 protocol-fault count
  assert svc.stats["phase_duplicate"] == 1
  _ = result_last
  fault_drain_and_reset(group)


def test_unauthorized_post_event_rejected_and_counted() -> None:
  group, _grid, _state = _dispatched_group()
  svc = EventService(group, authorized_post_producers=frozenset({"sealed-producer-1"}))
  post = _msg(
    "POST_EVENT",
    {
      "tile_id": 0,
      "ctx_slot": 0,
      "ctx_generation": 1,
      "local_event_id": 7,
      "event_generation": 0,
      "sequence": 0,
      "scope": "group_l2",
      "status": "DONE",
    },
  )
  with pytest.raises(EventServiceError) as exc:
    svc.handle_post_event(post, producer="rogue", cycle=1)
  assert exc.value.code == "POST_EVENT_UNAUTHORIZED"
  assert svc.stats["post_event_unauthorized"] == 1
  assert svc.stats["post_event_accepted"] == 0
  assert svc.handle_post_event(post, producer="sealed-producer-1", cycle=2) == "accepted"
  assert svc.stats["post_event_accepted"] == 1
  finish_group_and_reset(group)


def test_event_retire_broadcast_reaches_every_tile() -> None:
  group, _grid, _state = _dispatched_group()
  svc = EventService(group)
  retired = svc.broadcast_event_retire({"ev_dma_1"}, group.tiles)
  assert retired == len(group.tiles)
  assert svc.stats["event_retire_broadcasts"] == 1
  finish_group_and_reset(group)


# ---------------------------------------------------------------- reset phase
def test_reset_phase_mapping_and_unsolicited_ack() -> None:
  from pipeline_validator.runtime.reset_domain import ResetState

  svc = ResetPhaseService()
  req = svc.emit_request(ResetState.STOP_QUEUE, run_generation=3)
  assert req.fields == {"phase": "FREEZE_ISSUE", "run_generation": 3}
  assert svc.phase_for(ResetState.MARK_EVENTS) == "MARK_EVENTS"
  assert svc.phase_for(ResetState.DONE) == "RESUME_DONE"
  with pytest.raises(ResetPhaseError):
    svc.phase_for(ResetState.IDLE)
  ack = _msg(
    "RESET_PHASE_ACK",
    {"phase": "FREEZE_ISSUE", "run_generation": 3, "pending_to_reset": 0, "evidence_mask": 0},
  )
  svc.handle_ack(ack)
  assert svc.snapshot()["pending"] == []
  with pytest.raises(ResetPhaseError) as exc:
    svc.handle_ack(ack)  # unsolicited duplicate
  assert exc.value.code == "UNSOLICITED_RESET_ACK"


# ---------------------------------------------------------------- stream
def test_stream_subqueue_backpressure_is_independent_of_memory_main() -> None:
  q = StreamSubqueue(depth=2)
  assert q.depth == DEFAULT_STREAM_DEPTH or q.depth == 2
  for i in range(2):
    assert q.handle(_msg("STREAM_PUSH", {"token_id": i, "queue_generation": 0, "seq": i})) == "pushed"
  assert q.full()
  assert (
    q.handle(_msg("STREAM_PUSH", {"token_id": 2, "queue_generation": 0, "seq": 2})) == "stall:stream_full"
  )
  assert q.stats["stall_cycles_stream_full"] == 1

  # MEMORY main queue backpressure does NOT stall the stream sub-queue and vice versa
  q.note_memory_main_backpressure(True, cycle=5)
  assert q.memory_main_stalled is True
  assert q.handle(_msg("STREAM_POP", {"token_id": 0, "queue_generation": 0, "seq": 0})).startswith(
    "popped:"
  )
  assert not q.full()  # stream drained while MEMORY main is still backpressured
  q.note_memory_main_backpressure(False, cycle=6)

  q.handle(_msg("STREAM_POP", {"token_id": 1, "queue_generation": 0, "seq": 1}))  # drain last entry
  with pytest.raises(StreamSubqueueError) as empty:
    q.handle(_msg("STREAM_POP", {"token_id": 9, "queue_generation": 0, "seq": 9}))
  assert empty.value.code == "POP_EMPTY"
  assert q.handle(_msg("STREAM_RELEASE", {"token_id": 0, "queue_generation": 0, "seq": 0})) == "released"
  snap = q.snapshot()
  assert snap["occupancy"] == 0 and snap["memory_main_stalled"] is False
  assert (
    snap["stats"]["stall_cycles_stream_full"] == 1
  )  # stream stalled exactly once, memory main never stalled it
