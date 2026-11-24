"""C039 / W5-04 residency state-machine tests.

The manager under test is ``pipeline_validator/runtime/program_table.py``.  Every scenario is
driven against a REAL ``pipeline_validator.cosim.program_service.ProgramService`` (the C024
program-channel endpoint) — the same object the Bridge's RTL cosim drives — never a fixture
that pretends ACKs.  A fake Tile pull is emulated by real ``PROGRAM_READ_REQ`` / ``PROGRAM_
INSTALL_ACK`` messages built here and routed through ``manager.handle`` so the manager-side
state transitions are exercised end-to-end against the live channel bookkeeping
(``manager.assert_lockstep`` after every phase).

The analytic timing model (``register``/``ensure_resident``) is asserted to stay a coherent,
separate account — no real record consults it and it never manufactures residency proof.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from pipeline_validator.cosim.program_service import ProgramImage, ProgramService
from pipeline_validator.cosim.tile_port import BoundaryMessage
from pipeline_validator.runtime.program_table import (
  ProgramIdentity,
  ProgramResidencyManager,
  ResidencyError,
  ResidencyState,
  ResidencyStorage,
)


class _Cfg:
  """The analytic path only ever reads bandwidth/clock config."""

  clock_mhz = 1000.0
  group_dma_bandwidth_gbs = 8.0
  tile_l1_bandwidth_gbs = 4.0


def _mgr(capacity: int = 8192, slot_bytes: int = 4096, slots: int = 2, tiles: int = 1):
  manager = ProgramResidencyManager(
    _Cfg(), storage=ResidencyStorage(capacity, slot_bytes, slots)
  )
  services = {}
  for tile_id in range(tiles):
    service = ProgramService()
    manager.attach(tile_id, service)
    services[tile_id] = service
  return manager, services


def _image(pid: int, size: int, hash_byte: bytes, version: int = 1) -> ProgramImage:
  return ProgramImage(
    program_id=pid,
    version=version,
    package_hash=hash_byte * 32,
    program_epoch=0,
    text=bytes((pid + i) % 251 for i in range(size)),
  )


def _t2g(message_type: str, fields: dict) -> BoundaryMessage:
  return BoundaryMessage(
    message_type=message_type,
    channel="PROGRAM",
    direction="T2G",
    fields=fields,
    message_seq=0,
  )


def _ack_fields(ident: ProgramIdentity, text: bytes, epoch: int = 0) -> dict:
  return {
    **ident.wire_fields(),
    "program_epoch": epoch,
    "installed_len": len(text),
    "measured_hash": hashlib.sha256(text).digest(),
  }


def _pull(manager, tile: int, ident: ProgramIdentity, epoch: int = 0) -> bytes:
  """The Tile's ordered text pull: one PROGRAM_READ_REQ per chunk, real served bytes."""
  text_bytes = manager.image_info(ident).text_bytes
  fields = {**ident.wire_fields(), "program_epoch": epoch}
  pulled = bytearray()
  while len(pulled) < text_bytes:
    step = manager.handle(tile, _t2g("PROGRAM_READ_REQ", dict(fields)))
    pulled += step.data
  return bytes(pulled)


def _install(manager, tile: int, ident: ProgramIdentity, epoch: int = 0, handle: int = 7):
  """Full cold-install: pull every byte, then ACK with the real measured digest."""
  pulled = _pull(manager, tile, ident, epoch)
  return manager.handle(
    tile, _t2g("PROGRAM_INSTALL_ACK", _ack_fields(ident, pulled, epoch)), local_handle=handle
  )


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_register_derives_slot_and_setup_metadata():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 144, b"\x11"), text_base=4096)
  info = mgr.image_info(ident)
  assert info.slot == 1 and info.slot_address == 4096 and info.text_base == 4096
  assert info.text_bytes == 144 and info.text_sha256 == hashlib.sha256(bytes(range(1, 145))).digest()


def test_register_is_idempotent_for_identical_image():
  mgr, _ = _mgr()
  image = _image(1, 64, b"\x22")
  a = mgr.register_image(image, slot=0)
  assert mgr.register_image(image, slot=0) == a


@pytest.mark.parametrize(
  "kwargs, code",
  [
    ({"slot": 0, "text_base": 4096}, "SLOT_ADDRESS_MISMATCH"),
    ({"text_base": 100}, "TEXT_BASE_UNALIGNED"),
    ({"slot": 9}, "SLOT_OUT_OF_RANGE"),
    ({}, "PLACEMENT_UNDECLARED"),
  ],
)
def test_register_rejects_bad_placement(kwargs, code):
  mgr, _ = _mgr()
  with pytest.raises(ResidencyError, match=code):
    mgr.register_image(_image(1, 64, b"\x33"), **kwargs)


def test_register_rejects_oversized_and_identity_conflict():
  mgr, _ = _mgr()
  with pytest.raises(ResidencyError, match="PROGRAM_EXCEEDS_SLOT"):
    mgr.register_image(_image(1, 5000, b"\x44"), slot=0)
  ident = mgr.register_image(_image(2, 64, b"\x55"), slot=0)
  with pytest.raises(ResidencyError, match="IDENTITY_CONFLICT"):
    mgr.register_image(_image(2, 64, b"\x66"), slot=0)  # same key, different hash
  with pytest.raises(ResidencyError, match="IDENTITY_CONFLICT"):
    mgr.register_image(_image(2, 128, b"\x55"), slot=1)  # same key, different bytes
  assert mgr.register_image(_image(2, 64, b"\x55"), slot=0) == ident  # identical still ok


# ---------------------------------------------------------------------------
# cold install -> resident (positive transitions)
# ---------------------------------------------------------------------------


def test_cold_install_arms_one_install_and_publishes_ready_after_verified_ack():
  mgr, svc = _mgr()
  ident = mgr.register_image(_image(1, 144, b"\x77"), slot=0)
  decision = mgr.acquire(ident, 0, holder=("task", 1))
  assert decision.kind == "pending" and decision.completion_id is not None
  assert len(decision.outbox) == 1
  install = decision.outbox[0]
  assert install.message_type == "PROGRAM_INSTALL"
  assert install.fields["slot"] == 0 and install.fields["program_epoch"] == 0
  assert mgr.view(ident, 0).state is ResidencyState.PENDING_INSTALL

  step = _install(mgr, 0, ident)
  assert [m.message_type for m in step.outbox] == ["PROGRAM_READY"]
  assert step.outbox[0].fields["local_handle"] == 7
  mgr.assert_lockstep(0)

  view = mgr.view(ident, 0)
  assert view.state is ResidencyState.RESIDENT and view.installed
  assert view.local_handle == 7 and view.refs == 1
  assert svc[0].snapshot()["installs"]


def test_shared_pending_install_arms_once_and_completes_all_waiters():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 128, b"\x88"), slot=0)
  d1 = mgr.acquire(ident, 0, holder=("task", 1))
  d2 = mgr.acquire(ident, 0, holder=("task", 2))
  assert d1.kind == "pending" and d2.kind == "pending"
  assert d1.completion_id == d2.completion_id  # ONE shared install
  assert len(d1.outbox) == 1 and not d2.outbox  # only the starter sends the header
  assert mgr.counters()["shared_installs"] == 1
  _install(mgr, 0, ident)
  mgr.assert_lockstep(0)
  assert mgr.view(ident, 0).refs == 2


def test_warm_hit_takes_reference_and_never_fetches():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\x99"), slot=0)
  mgr.acquire(ident, 0)
  _install(mgr, 0, ident)
  tile_counts = dict(mgr.tile_counters(0))
  assert tile_counts["read_requests"] > 0
  hit = mgr.acquire(ident, 0)
  assert hit.kind == "hit" and hit.handle.local_handle == 7
  assert hit.handle.epoch == 0 and not hit.outbox
  assert dict(mgr.tile_counters(0)) == tile_counts  # zero external bytes for a warm hit
  mgr.assert_lockstep(0)


def test_read_after_install_is_a_protocol_error_not_an_abort():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xAA"), slot=0)
  mgr.acquire(ident, 0)
  _install(mgr, 0, ident)
  fields = {**ident.wire_fields(), "program_epoch": 0}
  with pytest.raises(ResidencyError, match="READ_AFTER_INSTALL"):
    mgr.handle(0, _t2g("PROGRAM_READ_REQ", fields))
  assert mgr.view(ident, 0).state is ResidencyState.RESIDENT  # resident record untouched


def test_two_identities_resident_in_both_slots():
  mgr, _ = _mgr()
  a = mgr.register_image(_image(1, 144, b"\xB1"), slot=0)
  b = mgr.register_image(_image(2, 144, b"\xB2"), slot=1)
  mgr.acquire(a, 0)
  mgr.acquire(b, 0)
  _install(mgr, 0, a)
  _install(mgr, 0, b)
  mgr.assert_lockstep(0)
  assert mgr.slot_bitmap(0) == 0b11 and mgr.resident_bitmap(0) == 0b11
  assert mgr.used_bytes(0) == 288


# ---------------------------------------------------------------------------
# bounded waits
# ---------------------------------------------------------------------------


def test_wait_slot_in_use_until_explicit_invalidation():
  mgr, _ = _mgr()
  a = mgr.register_image(_image(1, 64, b"\xC1"), slot=0)
  b = mgr.register_image(_image(2, 64, b"\xC2"), slot=0)  # same slot, different identity
  mgr.acquire(a, 0)
  _install(mgr, 0, a)
  decision = mgr.acquire(b, 0)
  assert decision.kind == "wait" and decision.reason == "WAIT_SLOT_IN_USE"
  assert decision.blocked_by == (a,)
  # explicit invalidation of the owner unblocks the wait
  mgr.begin_invalidate(a, 0)
  s = mgr.release(a, 0)  # anonymous ref retires -> INVALIDATE emitted
  assert [m.message_type for m in s.outbox] == ["PROGRAM_INVALIDATE"]
  mgr.handle(0, _t2g("INVALIDATE_ACK", {**a.wire_fields(), "program_epoch": 0}))
  mgr.assert_lockstep(0)
  assert mgr.acquire(b, 0).kind == "pending"


def test_wait_invalidating_never_hits():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xC3"), slot=0)
  mgr.acquire(ident, 0)
  _install(mgr, 0, ident)
  mgr.begin_invalidate(ident, 0)
  decision = mgr.acquire(ident, 0)
  assert decision.kind == "wait" and decision.reason == "WAIT_PROGRAM_INVALIDATING"
  assert decision.blocked_by == (ident,)
  assert mgr.counters()["waits"]["WAIT_PROGRAM_INVALIDATING"] == 1


def test_capacity_wait_names_blockers_and_resolves_on_invalidation():
  # A non-tiling geometry: the cumulative byte ledger, not the slot count, is what binds.
  mgr, _ = _mgr(capacity=200, slot_bytes=100, slots=3)
  x = mgr.register_image(_image(10, 90, b"\xD1"), slot=0)
  y = mgr.register_image(_image(11, 90, b"\xD2"), slot=1)
  z = mgr.register_image(_image(12, 90, b"\xD3"), slot=2)
  mgr.acquire(x, 0)
  _install(mgr, 0, x)
  mgr.acquire(y, 0)
  _install(mgr, 0, y)
  decision = mgr.acquire(z, 0)
  assert decision.kind == "wait" and decision.reason == "WAIT_PROGRAM_CAPACITY"
  assert set(decision.blocked_by) == {x, y}
  assert mgr.used_bytes(0) == 180 and mgr.free_bytes(0) == 20
  # a release keeps RESIDENT bytes resident (warm invariant); explicit invalidation frees them
  mgr.release(x, 0)
  assert mgr.used_bytes(0) == 180
  mgr.begin_invalidate(x, 0)
  mgr.handle(0, _t2g("INVALIDATE_ACK", {**x.wire_fields(), "program_epoch": 0}))
  assert mgr.used_bytes(0) == 90
  assert mgr.acquire(z, 0).kind == "pending"


# ---------------------------------------------------------------------------
# invalidation lifecycle, epochs
# ---------------------------------------------------------------------------


def test_invalidation_drains_through_ack_and_frees_slot_and_bytes():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xE1"), slot=0)
  mgr.acquire(ident, 0)
  _install(mgr, 0, ident)
  assert mgr.used_bytes(0) == 64
  mgr.begin_invalidate(ident, 0)
  s = mgr.release(ident, 0)
  assert [m.message_type for m in s.outbox] == ["PROGRAM_INVALIDATE"]
  assert mgr.view(ident, 0).state is ResidencyState.INVALIDATING
  mgr.handle(0, _t2g("INVALIDATE_ACK", {**ident.wire_fields(), "program_epoch": 0}))
  mgr.assert_lockstep(0)
  assert mgr.view(ident, 0) is None and mgr.slot_bitmap(0) == 0 and mgr.used_bytes(0) == 0


def test_epoch_advances_per_tile_install_transaction_and_stale_is_rejected():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xE2"), slot=0)
  mgr.acquire(ident, 0)
  _install(mgr, 0, ident)
  mgr.begin_invalidate(ident, 0)
  mgr.release(ident, 0)
  mgr.handle(0, _t2g("INVALIDATE_ACK", {**ident.wire_fields(), "program_epoch": 0}))
  # the next install transaction carries epoch+1, mirroring ProgramService._drop_record
  decision = mgr.acquire(ident, 0)
  assert decision.outbox[0].fields["program_epoch"] == 1
  stale = {**ident.wire_fields(), "program_epoch": 0}
  with pytest.raises(ResidencyError, match="STALE_READ_REQUEST"):
    mgr.handle(0, _t2g("PROGRAM_READ_REQ", stale))
  with pytest.raises(ResidencyError, match="STALE_INSTALL_ACK"):
    mgr.handle(
      0,
      _t2g(
        "PROGRAM_INSTALL_ACK",
        {**stale, "installed_len": 64, "measured_hash": hashlib.sha256(b"").digest()},
      ),
      local_handle=1,
    )
  _install(mgr, 0, ident, epoch=1)
  mgr.assert_lockstep(0)


def test_single_tile_invalidation_leaves_other_tiles_resident():
  mgr, svc = _mgr(tiles=2)
  ident = mgr.register_image(_image(1, 64, b"\xE3"), slot=0)
  for tile in (0, 1):
    mgr.acquire(ident, tile)
    _install(mgr, tile, ident)
  mgr.assert_lockstep()
  mgr.invalidate_tile(0)
  assert mgr.slot_bitmap(0) == 0 and mgr.resident_bitmap(1) == 1
  assert mgr.view(ident, 0) is None and mgr.view(ident, 1).state is ResidencyState.RESIDENT
  # tile 0's epoch advanced; tile 1's did not
  d0 = mgr.acquire(ident, 0)
  assert d0.outbox[0].fields["program_epoch"] == 1
  hit = mgr.acquire(ident, 1)
  assert hit.kind == "hit"
  mgr.assert_lockstep()


def test_group_reset_voids_every_tile_and_advances_epochs():
  mgr, _ = _mgr(tiles=2)
  a = mgr.register_image(_image(1, 64, b"\xE4"), slot=0)
  b = mgr.register_image(_image(2, 64, b"\xE5"), slot=1)
  for tile in (0, 1):
    mgr.acquire(a, tile)
    mgr.acquire(b, tile)
    _install(mgr, tile, a)
    _install(mgr, tile, b)
  mgr.invalidate_group()
  assert mgr.resident_bitmap(0) == 0 and mgr.resident_bitmap(1) == 0
  d = mgr.acquire(a, 0)
  assert d.outbox[0].fields["program_epoch"] == 1
  mgr.assert_lockstep()


def test_invalidate_while_pulling_never_publishes_ready():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 128, b"\xE6"), slot=0)
  mgr.acquire(ident, 0)  # anonymous ref
  # begin invalidating with only the anonymous acquire ref held and a pull mid-flight
  _pull(mgr, 0, ident)
  step = mgr.begin_invalidate(ident, 0)
  assert not step.outbox  # still has a reference; INVALIDATE waits
  fields = {**ident.wire_fields(), "program_epoch": 0}
  acked = mgr.handle(
    0,
    _t2g(
      "PROGRAM_INSTALL_ACK",
      {**fields, "installed_len": 128, "measured_hash": hashlib.sha256(
        bytes((1 + i) % 251 for i in range(128))
      ).digest()},
    ),
    local_handle=3,
  )
  # the pull was invalidated in flight: the ACK is consumed, READY is never emitted
  assert acked.outbox == ()
  assert mgr.view(ident, 0).state is ResidencyState.INVALIDATING
  # installed-but-invalidated: lockstep holds at the point READY was withheld (no bound handle)
  mgr.assert_lockstep(0)
  s = mgr.release(ident, 0)
  assert [m.message_type for m in s.outbox] == ["PROGRAM_INVALIDATE"]
  mgr.handle(0, _t2g("INVALIDATE_ACK", fields))
  mgr.assert_lockstep(0)


def test_unexpected_invalidate_ack_rejected():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xE7"), slot=0)
  mgr.acquire(ident, 0)
  _install(mgr, 0, ident)
  with pytest.raises(ResidencyError, match="UNEXPECTED_INVALIDATE_ACK"):
    mgr.handle(0, _t2g("INVALIDATE_ACK", {**ident.wire_fields(), "program_epoch": 0}))
  mgr.assert_lockstep(0)

def test_faulted_manager_refuses_every_real_operation():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xE8"), slot=0)
  mgr.acquire(ident, 0)
  mgr.fail_install(ident, 0, code="INSTALL_HW_FAULT")
  assert mgr.fault is not None
  # fail-stop is global: even a tile that was never faulted refuses real-path work
  assert mgr.acquire(ident, 0).kind == "refused"
  with pytest.raises(ResidencyError, match="RESIDENCY_FAULTED"):
    mgr.release(ident, 0)
  with pytest.raises(ResidencyError, match="RESIDENCY_FAULTED"):
    mgr.begin_invalidate(ident, 0)
  with pytest.raises(ResidencyError, match="RESIDENCY_FAULTED"):
    mgr.patch(ident, 0, field="f", layer="l")
  with pytest.raises(ResidencyError, match="RESIDENCY_FAULTED"):
    mgr.handle(0, _t2g("PROGRAM_READ_REQ", {**ident.wire_fields(), "program_epoch": 0}))

def test_install_epoch_mirrors_channel_install_counter():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xE9"), slot=0)
  assert mgr.install_epoch(ident, 0) == 0
  mgr.acquire(ident, 0)
  _install(mgr, 0, ident)
  mgr.begin_invalidate(ident, 0)
  mgr.release(ident, 0)
  mgr.handle(0, _t2g("INVALIDATE_ACK", {**ident.wire_fields(), "program_epoch": 0}))
  # the next transaction must carry the post-drop epoch — matching ProgramService._drop_record
  assert mgr.install_epoch(ident, 0) == 1
  d = mgr.acquire(ident, 0)
  assert d.outbox[0].fields["program_epoch"] == 1


# ---------------------------------------------------------------------------
# fault abort: no half-resident record, latch, refused, reset recovery
# ---------------------------------------------------------------------------


def test_hash_mismatch_aborts_and_latches_fault():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xF1"), slot=0)
  mgr.acquire(ident, 0, holder=("waiter", 1))
  _pull(mgr, 0, ident)
  bad = {
    **ident.wire_fields(),
    "program_epoch": 0,
    "installed_len": 64,
    "measured_hash": b"\x00" * 32,
  }
  with pytest.raises(ResidencyError, match="INSTALL_HASH_MISMATCH"):
    mgr.handle(0, _t2g("PROGRAM_INSTALL_ACK", bad), local_handle=1)
  fault = mgr.fault
  assert fault is not None and fault.code == "INSTALL_HASH_MISMATCH"
  assert mgr.view(ident, 0) is None  # nothing half-resident
  assert mgr.slot_bitmap(0) == 0 and mgr.used_bytes(0) == 0
  decision = mgr.acquire(ident, 0)
  assert decision.kind == "refused" and decision.reason == "RESIDENCY_FAULTED"
  mgr.invalidate_group()  # a reset clears the latch
  assert mgr.fault is None
  assert mgr.acquire(ident, 0).kind == "pending"


def test_incomplete_install_aborts():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 128, b"\xF2"), slot=0)
  mgr.acquire(ident, 0)
  _pull(mgr, 0, ident)  # serves all 128
  # A Tile that ACKs early is impossible through the pull API, so emulate a short pull:
  mgr2, _ = _mgr()
  ident2 = mgr2.register_image(_image(1, 128, b"\xF2"), slot=0)
  mgr2.acquire(ident2, 0)
  f = {**ident2.wire_fields(), "program_epoch": 0}
  mgr2.handle(0, _t2g("PROGRAM_READ_REQ", dict(f)))  # served only 64 of 128
  with pytest.raises(ResidencyError, match="INCOMPLETE_INSTALL"):
    mgr2.handle(
      0,
      _t2g(
        "PROGRAM_INSTALL_ACK",
        {
          **f,
          "installed_len": 128,
          "measured_hash": hashlib.sha256(bytes((1 + i) % 251 for i in range(128))).digest(),
        },
      ),
      local_handle=1,
    )
  assert mgr2.view(ident2, 0) is None and mgr2.fault.code == "INCOMPLETE_INSTALL"


def test_bad_local_handle_aborts_install():
  for bad_handle in (None, -1, 1 << 64):  # absent sideband, negative, wider than one wire u64
    mgr, _ = _mgr()
    ident = mgr.register_image(_image(1, 64, b"\xF3"), slot=0)
    mgr.acquire(ident, 0)
    pulled = _pull(mgr, 0, ident)
    with pytest.raises(ResidencyError, match="BAD_LOCAL_HANDLE"):
      mgr.handle(
        0, _t2g("PROGRAM_INSTALL_ACK", _ack_fields(ident, pulled)), local_handle=bad_handle
      )
    assert mgr.view(ident, 0) is None and mgr.fault.code == "BAD_LOCAL_HANDLE"


def test_malformed_installed_len_aborts_install():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xF4"), slot=0)
  mgr.acquire(ident, 0)
  with pytest.raises(ResidencyError, match="BAD_PROGRAM_MESSAGE"):
    mgr.handle(
      0,
      _t2g(
        "PROGRAM_INSTALL_ACK",
        {**ident.wire_fields(), "program_epoch": 0, "installed_len": "x", "measured_hash": b"\x00" * 32},
      ),
      local_handle=1,
    )
  # malformed ACK killed the install transaction; no stuck pending remains
  assert mgr.view(ident, 0) is None


def test_fail_install_public_path_releases_waiters_and_latches():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xF5"), slot=0)
  mgr.acquire(ident, 0, holder=("w", 1))
  mgr.acquire(ident, 0, holder=("w", 2))
  mgr.fail_install(ident, 0, code="INSTALL_HW_FAULT")
  fault = mgr.fault
  assert fault.code == "INSTALL_HW_FAULT" and set(fault.affected) == {"('w', 1)", "('w', 2)"}
  assert mgr.view(ident, 0) is None
  mgr.invalidate_group()
  assert mgr.fault is None


def test_unhashable_holder_never_arms_an_install():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xF6"), slot=0)
  with pytest.raises(ResidencyError, match="BAD_HOLDER"):
    mgr.acquire(ident, 0, holder=["unhashable", "list"])
  # no PENDING ref0 zombie: the preflight ran before the arm
  assert mgr.view(ident, 0) is None and mgr.slot_bitmap(0) == 0


def test_release_underflow_and_duplicate_reference():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xF7"), slot=0)
  with pytest.raises(ResidencyError, match="REF_NOT_HELD"):
    mgr.release(ident, 0)
  mgr.acquire(ident, 0, holder=("h", 1))
  with pytest.raises(ResidencyError, match="REF_DUPLICATE"):
    mgr.acquire(ident, 0, holder=("h", 1))
  with pytest.raises(ResidencyError, match="REF_NOT_HELD"):
    mgr.release(ident, 0, holder=("h", 2))
  mgr.release(ident, 0, holder=("h", 1))
  with pytest.raises(ResidencyError, match="REF_UNDERFLOW"):
    mgr.release(ident, 0)


# ---------------------------------------------------------------------------
# patch-mask rules (RES-07)
# ---------------------------------------------------------------------------


def test_patch_within_mask_succeeds():
  mgr, _ = _mgr()
  ident = mgr.register_image(
    _image(1, 64, b"\xA1"), slot=0, patch_mask=[{"field": "param_area.shape", "layer": "runtime"}]
  )
  mgr.acquire(ident, 0)
  _install(mgr, 0, ident)
  mgr.patch(ident, 0, field="param_area.shape", layer="runtime")
  assert mgr.counters()["descriptor_patches"] == 1


def test_patch_wrong_layer_is_invalid_descriptor_fault():
  mgr, _ = _mgr()
  ident = mgr.register_image(
    _image(1, 64, b"\xA2"), slot=0, patch_mask=[{"field": "param_area.shape", "layer": "runtime"}]
  )
  mgr.acquire(ident, 0)
  _install(mgr, 0, ident)
  with pytest.raises(ResidencyError, match="INVALID_DESCRIPTOR"):
    mgr.patch(ident, 0, field="param_area.shape", layer="accept")
  assert mgr.fault.code == "INVALID_DESCRIPTOR"
  # fail-stop: every subsequent acquire is refused until a reset clears the latch
  refused = mgr.acquire(ident, 0)
  assert refused.kind == "refused" and refused.reason == "RESIDENCY_FAULTED"


def test_patch_text_is_never_allowed_and_undeclared_fields_fault():
  mgr, _ = _mgr()
  ident = mgr.register_image(
    _image(1, 64, b"\xA3"), slot=0, patch_mask=[{"field": "text.bytes", "layer": "runtime"}]
  )
  mgr.acquire(ident, 0)
  _install(mgr, 0, ident)
  # even a sealed mask entry naming resident text is refused (resident text is never patched)
  with pytest.raises(ResidencyError, match="INVALID_DESCRIPTOR"):
    mgr.patch(ident, 0, field="text.bytes", layer="runtime")
  assert mgr.fault.code == "INVALID_DESCRIPTOR"  # fail-stop latched; residency record survives
  assert mgr.view(ident, 0).state is ResidencyState.RESIDENT

  # an undeclared field on a separate, unfaulted manager faults identically
  mgr2, _ = _mgr()
  ident2 = mgr2.register_image(_image(2, 64, b"\xA4"), slot=1)
  mgr2.acquire(ident2, 0)
  _install(mgr2, 0, ident2)
  with pytest.raises(ResidencyError, match="INVALID_DESCRIPTOR"):
    mgr2.patch(ident2, 0, field="undeclared.field", layer="runtime")
  assert mgr2.fault.code == "INVALID_DESCRIPTOR"


def test_patch_requires_resident_and_rejects_invalidating():
  mgr, _ = _mgr()
  ident = mgr.register_image(_image(1, 64, b"\xA5"), slot=0)
  with pytest.raises(ResidencyError, match="PROGRAM_NOT_RESIDENT"):
    mgr.patch(ident, 0, field="f", layer="l")
  mgr.acquire(ident, 0)
  _install(mgr, 0, ident)
  mgr.begin_invalidate(ident, 0)
  with pytest.raises(ResidencyError, match="PROGRAM_INVALIDATING"):
    mgr.patch(ident, 0, field="f", layer="l")


# ---------------------------------------------------------------------------
# analytic isolation
# ---------------------------------------------------------------------------


def test_analytic_timing_model_is_coherently_separate_from_real_residency():
  mgr, _ = _mgr()
  # legacy entry: no bytes, no ACK, no references — never residency proof
  entry = mgr.register(7, 3, 0xBEEF, hbm_iova=0x1000, hbm_bytes=144)
  assert entry.epoch == 0
  cycles = mgr.ensure_resident(7, 0, cycle=100)
  assert cycles > 0 and mgr.cold_load_cycles == cycles
  snap = mgr.snapshot()
  assert 7 in snap["analytic"] and snap["analytic"][7]["tile_states"][0] == "RESIDENT"
  # the analytic mark is NOT on the real path: no image, no record, acquire refuses
  real = snap["installed"]
  assert real["images"] == {} and real["tiles"][0]["slots"] == {}
  with pytest.raises(ResidencyError, match="IDENTITY_UNKNOWN"):
    mgr.acquire(ProgramIdentity(7, 3, b"\x00" * 32), 0)
  # a real install does not consult the analytic entry
  ident = mgr.register_image(_image(7, 64, b"\xB0"), slot=0)
  assert mgr.acquire(ident, 0).kind == "pending"
  # analytic residency is still just timing state, untouched by the real install
  _install(mgr, 0, ident)
  snap2 = mgr.snapshot()
  assert snap2["analytic"][7]["tile_states"][0] == "RESIDENT"
  assert snap2["installed"]["tiles"][0]["resident_bitmap"] == 1


def test_analytic_group_reset_bumps_epoch_for_legacy_consumers():
  mgr, _ = _mgr()
  mgr.register(9, 1, 0x1234, hbm_iova=0, hbm_bytes=64)
  mgr.ensure_resident(9, 0, cycle=0)
  mgr.invalidate_group()
  e = mgr.snapshot()["analytic"][9]
  assert e["epoch"] == 1 and e["tile_states"] == {}

# ---------------------------------------------------------------------------
# REAL C038 attempt-002 images — deterministic rebuild + pinned release digests
# ---------------------------------------------------------------------------

_H = Path(__file__).resolve().parents[2] / "hardware/coral_nexus"
# Pinned C038 attempt-002 release digests (t58_cold_warm scenario). The cn package rebuild of
# scalar_ctl_v1.yaml is byte-identical to the accepted evidence; the scenario still asserts it.
_PINNED_PACKAGE_HASH = {
  "A": "f062e954f3523106ec5fe8ee5729d37d419b1c68ed88d47f0aebea5e9d3f8bdc",
  "B": "1cf65e4510afb7802601d7eba2495e3ccbc26e20a281a6f1cb14ba8b3ca4829d",
}


@pytest.fixture(scope="module")
def c038_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
  """Rebuild the two sealed slot images exactly as the C038 attempt-002 evidence did — the same
  ``cn package`` invocation as ``test_w5_scalar_package``; never skipped, never a fixture."""
  out = tmp_path_factory.mktemp("c038-package")
  subprocess.run(
    [
      "bash",
      str(_H / "scripts/cn.sh"),
      "package",
      "--binding",
      str(_H / "kernels/bindings/scalar_ctl_v1.yaml"),
      "--config",
      str(_H / "configs/cs1_c2.yaml"),
      "--out",
      str(out),
    ],
    check=True,
    timeout=300,
  )
  return out


@pytest.fixture(scope="module")
def c038_scenario(c038_package: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
  """A minimal cn_program_scenario/v1 pointing at the rebuilt tree, with the pinned release
  package hashes so a byte drift fails the Loader rather than guessing."""
  scenario = tmp_path_factory.mktemp("c038-scenario") / "scenario.yaml"
  scenario.write_text(
    "schema: cn_program_scenario/v1\n"
    "name: c039_residency_state\n"
    "slot_freeze: slot_freeze.json\n"
    "slots:\n"
    f"  A: {{ package: A/package.json, package_hash: {_PINNED_PACKAGE_HASH['A']}, hbm_iova: 0x300000 }}\n"
    f"  B: {{ package: B/package.json, package_hash: {_PINNED_PACKAGE_HASH['B']}, hbm_iova: 0x301000 }}\n"
    "launch_bindings:\n"
    "  X: { base_iova: 0x100000, size_bytes: 32, permissions: r }\n"
    "  Y: { base_iova: 0x200000, size_bytes: 32, permissions: w }\n"
    "steps: {}\n"
  )
  return scenario


def test_real_c038_attempt002_images_drive_residency(c038_package, c038_scenario):
  """The two real sealed slot images install, serve real bytes, and invalidate correctly.

  Deterministic: the images are rebuilt by ``cn package`` into a tmpdir and must equal the
  pinned attempt-002 package hashes.  Manager-side assertions are the scenario's expected
  values: 144-byte texts, 3 read chunks per cold install, bitmap 3 with both resident, bitmap
  0 after group reset.
  """
  from pipeline_validator.cosim.program_prepare import prepare_programs

  prepared = prepare_programs(
    c038_scenario, c038_package, config_path=_H / "configs/cs1_c2.yaml"
  )
  mgr = ProgramResidencyManager(_Cfg())
  identities = prepared.attach(mgr, 0)  # configure_storage + attach + register_image, safe order
  ident_a, ident_b = identities["A"], identities["B"]
  info_a, info_b = mgr.image_info(ident_a), mgr.image_info(ident_b)
  assert info_a.text_bytes == 144 and info_b.text_bytes == 144
  assert info_a.slot == 0 and info_b.slot == 1

  # cold install of A: exactly the scenario's 3 read chunks / 144 bytes
  d = mgr.acquire(ident_a, 0, holder=("task", 0))
  assert d.kind == "pending" and d.outbox[0].fields["slot"] == info_a.text_base
  pulled = _pull(mgr, 0, ident_a)
  assert pulled == prepared.images["A"].text  # real bytes, byte-for-byte
  mgr.handle(0, _t2g("PROGRAM_INSTALL_ACK", _ack_fields(ident_a, pulled)), local_handle=1)
  mgr.assert_lockstep(0)
  counts = mgr.tile_counters(0)
  assert counts["read_requests"] == 3 and counts["external_text_bytes"] == 144
  assert mgr.slot_bitmap(0) == 1 and mgr.resident_bitmap(0) == 1

  # warm hit adds a reference, no fetch
  hit = mgr.acquire(ident_a, 0, holder=("task", 1))
  assert hit.kind == "hit"
  assert dict(mgr.tile_counters(0)) == counts  # zero external bytes on a warm hit

  # cold install of B in the other slot; A untouched
  mgr.acquire(ident_b, 0, holder=("task", 0))
  pulled_b = _pull(mgr, 0, ident_b)
  mgr.handle(0, _t2g("PROGRAM_INSTALL_ACK", _ack_fields(ident_b, pulled_b)), local_handle=2)
  mgr.assert_lockstep(0)
  assert mgr.resident_bitmap(0) == 3
  assert mgr.tile_counters(0)["external_text_bytes"] == 288

  # invalidate A: ref holder retires, INVALIDATE drains, bitmap drops to B only
  mgr.begin_invalidate(ident_a, 0)
  mgr.release(ident_a, 0, holder=("task", 0))
  s = mgr.release(ident_a, 0, holder=("task", 1))
  assert [m.message_type for m in s.outbox] == ["PROGRAM_INVALIDATE"]
  mgr.handle(0, _t2g("INVALIDATE_ACK", {**ident_a.wire_fields(), "program_epoch": 0}))
  mgr.assert_lockstep(0)
  assert mgr.resident_bitmap(0) == 2

  # group reset: nothing resident; next install carries epoch+1
  mgr.invalidate_group()
  assert mgr.resident_bitmap(0) == 0
  d = mgr.acquire(ident_b, 0)
  assert d.outbox[0].fields["program_epoch"] == 1
