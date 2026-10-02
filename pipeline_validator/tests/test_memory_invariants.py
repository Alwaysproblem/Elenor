"""Allocator invariant tests for the PR 2 physical memory model.

Covers the ``BankedFreeExtentAllocator``, ``HBMRegion`` external binding
registry and per-tile L1 owner isolation with small-capacity / few-bank
configurations.  Fixed diagnostic fragments are asserted so the
allocator contract is testable.
"""

from __future__ import annotations

import pytest

from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.memory import (
  L2SRAM,
  AdmissionFailure,
  AdmissionFailureKind,
  AllocationRequest,
  BankedFreeExtentAllocator,
  ContextBufferOwner,
  DeterministicLRUCache,
  ExternalOwner,
  HBMRegion,
  MemoryInvariantError,
  MshrAllocation,
  MshrTable,
  MshrWait,
  TaskBufferOwner,
)


def _ctx_owner(name: str = "ctx", gen: int = 0, buf: str = "b") -> ContextBufferOwner:
  return ContextBufferOwner(name, gen, buf)


def _task_owner(tile: int = 0, ctx: int = 0, task: int = 0, buf: str = "l1") -> TaskBufferOwner:
  return TaskBufferOwner("ctx", 0, "ev_role", task, tile, ctx, buf)


def _req(owner, size: int, align: int = 1, buf_id: str = "b", space: str = "l2") -> AllocationRequest:
  return AllocationRequest(space, buf_id, owner, size, align)


# ---------------------------------------------------------------------------
# Free-extent split / merge / alignment / cross-bank
# ---------------------------------------------------------------------------


class TestFreeExtent:
  def test_split_and_merge_on_release(self):
    alloc = BankedFreeExtentAllocator("l2", 4096, 4)
    o = _ctx_owner()
    h = alloc.commit(alloc.plan_bundle([_req(o, 1024)]) or [], 0)[0]
    snap = alloc.snapshot()
    assert snap["allocated_bytes"] == 1024
    assert alloc.request_release(h, o, 10) is True
    snap2 = alloc.snapshot()
    assert snap2["allocated_bytes"] == 0
    assert snap2["free_bytes"] == 4096
    # after release the free extent should be fully merged
    assert snap2["largest_free_extent"] == 1024  # per-bank largest

  def test_alignment_rounds_up_base(self):
    alloc = BankedFreeExtentAllocator("l2", 4096, 4)
    o = _ctx_owner()
    # place a 1-byte alloc to create a gap, then align the next to 256
    h1 = alloc.commit(alloc.plan_bundle([_req(o, 1, align=1)]) or [], 0)[0]
    assert h1.base_address == 0
    h2 = alloc.commit(alloc.plan_bundle([_req(o, 1024, align=256, buf_id="b2")]) or [], 0)[0]
    assert h2.base_address % 256 == 0

  def test_cross_bank_segments(self):
    # 2 banks of 512 each; request 768 → spans bank 0 (512) + bank 1 (256)
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    plan = alloc.plan_bundle([_req(o, 768)])
    assert not isinstance(plan, AdmissionFailure)
    h = alloc.commit(plan, 0)[0]
    assert len(h.bank_segments) == 2
    assert h.size_bytes == 768
    assert h.base_address == h.bank_segments[0].address

  def test_arbitrary_release_order_no_overlap(self):
    alloc = BankedFreeExtentAllocator("l2", 4096, 4)
    owners = [_ctx_owner(buf=f"b{i}") for i in range(4)]
    plan = alloc.plan_bundle([_req(owners[i], 512, buf_id=f"b{i}") for i in range(4)])
    handles = alloc.commit(plan, 0)
    # release in reverse order
    for h, o in zip(reversed(handles), reversed(owners)):
      alloc.request_release(h, o, 10)
    assert alloc.snapshot()["allocated_bytes"] == 0
    # re-allocate the full capacity — no overlap means it succeeds
    plan2 = alloc.plan_bundle([_req(_ctx_owner(buf="big"), 4096, buf_id="big")])
    assert not isinstance(plan2, AdmissionFailure)

  def test_exact_capacity_succeeds(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    plan = alloc.plan_bundle([_req(o, 1024)])
    assert not isinstance(plan, AdmissionFailure)
    alloc.commit(plan, 0)
    assert alloc.snapshot()["allocated_bytes"] == 1024

  def test_one_byte_over_capacity_fails(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    plan = alloc.plan_bundle([_req(o, 1025)])
    assert isinstance(plan, AdmissionFailure)
    assert plan.reason == "allocation capacity exceeded"

  def test_zero_size_fails(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    plan = alloc.plan_bundle([_req(_ctx_owner(), 0)])
    assert isinstance(plan, AdmissionFailure)
    assert plan.reason == "invalid allocation size"

  def test_non_power_of_two_alignment_fails(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    plan = alloc.plan_bundle([_req(_ctx_owner(), 64, align=3)])
    assert isinstance(plan, AdmissionFailure)
    assert plan.reason == "invalid allocation alignment"


# ---------------------------------------------------------------------------
# Atomic plan / commit / rollback / stale plan
# ---------------------------------------------------------------------------


class TestPlanCommitRollback:
  def test_atomic_commit_all_or_nothing(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    # one fits, one doesn't → whole bundle fails
    plan = alloc.plan_bundle([_req(o, 512, buf_id="ok"), _req(o, 600, buf_id="big")])
    assert isinstance(plan, AdmissionFailure)
    # nothing committed
    assert alloc.snapshot()["allocated_bytes"] == 0

  def test_rollback_uncommitted_plan_no_side_effect(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    plan = alloc.plan_bundle([_req(o, 512)])
    alloc.rollback(plan)
    assert alloc.snapshot()["allocated_bytes"] == 0
    # can still commit a fresh plan
    plan2 = alloc.plan_bundle([_req(o, 512)])
    alloc.commit(plan2, 0)
    assert alloc.snapshot()["allocated_bytes"] == 512

  def test_stale_plan_rejected(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    plan = alloc.plan_bundle([_req(o, 512)])
    alloc.commit(plan, 0)  # bumps pool_version
    with pytest.raises(MemoryInvariantError, match="stale allocation plan"):
      alloc.commit(plan, 1)


# ---------------------------------------------------------------------------
# Owner / generation / use-after-release
# ---------------------------------------------------------------------------


class TestOwnerGeneration:
  def test_wrong_owner_release(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o1 = _ctx_owner(buf="a")
    o2 = _ctx_owner(buf="b")
    h = alloc.commit(alloc.plan_bundle([_req(o1, 512)]) or [], 0)[0]
    with pytest.raises(MemoryInvariantError, match="wrong-owner release"):
      alloc.request_release(h, o2, 10)

  def test_double_release(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    h = alloc.commit(alloc.plan_bundle([_req(o, 512)]) or [], 0)[0]
    alloc.request_release(h, o, 10)
    with pytest.raises(MemoryInvariantError, match="double release"):
      alloc.request_release(h, o, 11)

  def test_use_after_release(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    h = alloc.commit(alloc.plan_bundle([_req(o, 512)]) or [], 0)[0]
    alloc.request_release(h, o, 10)
    with pytest.raises(MemoryInvariantError, match="use-after-release"):
      alloc.resolve_segments(h, 0, 64)

  def test_stale_generation_after_reset(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    h = alloc.commit(alloc.plan_bundle([_req(o, 512)]) or [], 0)[0]
    alloc.reset()
    with pytest.raises(MemoryInvariantError, match="stale allocation generation"):
      alloc.assert_live(h)

  def test_resolve_out_of_bounds(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    h = alloc.commit(alloc.plan_bundle([_req(o, 256)]) or [], 0)[0]
    with pytest.raises(MemoryInvariantError, match="memory view out of bounds"):
      alloc.resolve_segments(h, 0, 512)
    with pytest.raises(MemoryInvariantError, match="memory view out of bounds"):
      alloc.resolve_segments(h, -1, 64)


# ---------------------------------------------------------------------------
# Pin / pending-release / final unpin
# ---------------------------------------------------------------------------


class TestPinUnpin:
  def test_pin_then_release_immediate(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    h = alloc.commit(alloc.plan_bundle([_req(o, 256)]) or [], 0)[0]
    alloc.pin(h, "consumer1")
    # release while pinned → pending
    assert alloc.request_release(h, o, 5) is False
    assert alloc.snapshot()["pending_release"] == 1
    # final unpin → actual release
    assert alloc.unpin(h, "consumer1", 10) is True
    assert alloc.snapshot()["allocated_bytes"] == 0

  def test_duplicate_pin_rejected(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    h = alloc.commit(alloc.plan_bundle([_req(o, 256)]) or [], 0)[0]
    alloc.pin(h, "c1")
    with pytest.raises(MemoryInvariantError, match="duplicate allocation pin"):
      alloc.pin(h, "c1")

  def test_unknown_pin_rejected(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    h = alloc.commit(alloc.plan_bundle([_req(o, 256)]) or [], 0)[0]
    with pytest.raises(MemoryInvariantError, match="unknown allocation pin"):
      alloc.unpin(h, "nope", 10)

  def test_multiple_pins_last_unpin_releases(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    o = _ctx_owner()
    h = alloc.commit(alloc.plan_bundle([_req(o, 256)]) or [], 0)[0]
    alloc.pin(h, "c1")
    alloc.pin(h, "c2")
    assert alloc.request_release(h, o, 5) is False
    assert alloc.unpin(h, "c1", 6) is False  # still pinned by c2
    assert alloc.unpin(h, "c2", 7) is True  # last pin → release
    assert alloc.snapshot()["allocated_bytes"] == 0


# ---------------------------------------------------------------------------
# HBM external binding registry
# ---------------------------------------------------------------------------


class TestHBMRegion:
  def test_bind_external_ok(self):
    hbm = HBMRegion(size_bytes=16 * 1024 * 1024 * 1024)
    gb = GlobalBinding("Y", 0x100000, 4096, "rw")
    h = hbm.bind_external(gb)
    assert h.base_address == 0x100000
    assert h.size_bytes == 4096
    assert isinstance(h.owner, ExternalOwner)

  def test_overlap_rejected(self):
    hbm = HBMRegion(size_bytes=16 * 1024 * 1024 * 1024)
    hbm.bind_external(GlobalBinding("A", 0x100000, 4096, "rw"))
    with pytest.raises(ValueError, match="overlaps existing binding"):
      hbm.bind_external(GlobalBinding("B", 0x100000 + 2048, 4096, "rw"))

  def test_exceeds_capacity_rejected(self):
    hbm = HBMRegion(size_bytes=4096)
    with pytest.raises(ValueError, match="exceeds HBM capacity"):
      hbm.bind_external(GlobalBinding("Y", 0x1000, 4096, "rw"))

  def test_zero_size_rejected(self):
    hbm = HBMRegion(size_bytes=4096)
    with pytest.raises(ValueError, match="size must be > 0"):
      hbm.bind_external(GlobalBinding("Y", 0, 0, "rw"))

  def test_resolve_view_bounds(self):
    hbm = HBMRegion(size_bytes=16 * 1024 * 1024 * 1024)
    h = hbm.bind_external(GlobalBinding("Y", 0x100000, 4096, "rw"))
    segs = hbm.resolve(h, 100, 200)
    assert len(segs) == 1
    assert segs[0].address == 0x100000 + 100
    assert segs[0].size_bytes == 200
    with pytest.raises(MemoryInvariantError, match="memory view out of bounds"):
      hbm.resolve(h, 0, 8192)

  def test_unbind_and_reset(self):
    hbm = HBMRegion(size_bytes=16 * 1024 * 1024 * 1024)
    hbm.bind_external(GlobalBinding("Y", 0x100000, 4096, "rw"))
    assert hbm.snapshot()["external_bindings"] == 1
    hbm.unbind_external("Y")
    assert hbm.snapshot()["external_bindings"] == 0
    hbm.bind_external(GlobalBinding("Z", 0x100000, 4096, "rw"))
    hbm.reset()
    assert hbm.snapshot()["external_bindings"] == 0


# ---------------------------------------------------------------------------
# Per-tile L1 owner isolation
# ---------------------------------------------------------------------------


class TestPerTileL1Isolation:
  def test_same_local_base_different_owners_no_conflict(self):
    # two independent per-tile allocators can use the same local base
    tile0 = BankedFreeExtentAllocator("l1", 4096, 4)
    tile1 = BankedFreeExtentAllocator("l1", 4096, 4)
    o0 = _task_owner(tile=0)
    o1 = _task_owner(tile=1)
    h0 = tile0.commit(tile0.plan_bundle([_req(o0, 512, space="l1")]) or [], 0)[0]
    h1 = tile1.commit(tile1.plan_bundle([_req(o1, 512, space="l1")]) or [], 0)[0]
    assert h0.base_address == h1.base_address  # same local base
    assert h0.owner != h1.owner  # different owners
    # releasing on tile0 doesn't affect tile1
    tile0.request_release(h0, o0, 10)
    assert tile1.snapshot()["live_allocations"] == 1

  def test_l1_exact_capacity_succeeds(self):
    alloc = BankedFreeExtentAllocator("l1", 2048, 2)
    o = _task_owner()
    plan = alloc.plan_bundle([_req(o, 2048, space="l1")])
    assert not isinstance(plan, AdmissionFailure)
    alloc.commit(plan, 0)

  def test_l1_one_byte_over_fails(self):
    alloc = BankedFreeExtentAllocator("l1", 2048, 2)
    plan = alloc.plan_bundle([_req(_task_owner(), 2049, space="l1")])
    assert isinstance(plan, AdmissionFailure)

  def test_reset_clears_live_allocations(self):
    alloc = BankedFreeExtentAllocator("l1", 2048, 2)
    o = _task_owner()
    alloc.commit(alloc.plan_bundle([_req(o, 512, space="l1")]) or [], 0)
    assert alloc.snapshot()["live_allocations"] == 1
    alloc.reset()
    assert alloc.snapshot()["live_allocations"] == 0
    assert alloc.snapshot()["generation"] == 1


# ---------------------------------------------------------------------------
# L2SRAM wrapper
# ---------------------------------------------------------------------------


class TestL2SRAMWrapper:
  def test_plan_commit_release(self):
    l2 = L2SRAM(capacity_bytes=4096, banks=4)
    o = _ctx_owner()
    plan = l2.plan_bundle([_req(o, 1024)])
    handles = l2.commit(plan, 0)
    assert len(handles) == 1
    assert l2.snapshot()["live_allocations"] == 1
    l2.request_release(handles[0], o, 10)
    assert l2.snapshot()["live_allocations"] == 0

  def test_capacity_fault(self):
    l2 = L2SRAM(capacity_bytes=1024, banks=2)
    plan = l2.plan_bundle([_req(_ctx_owner(), 2048)])
    assert isinstance(plan, AdmissionFailure)
    assert plan.reason == "allocation capacity exceeded"

  def test_reset_clears(self):
    l2 = L2SRAM(capacity_bytes=4096, banks=4)
    o = _ctx_owner()
    plan = l2.plan_bundle([_req(o, 1024)])
    l2.commit(plan, 0)
    l2.reset()
    assert l2.snapshot()["live_allocations"] == 0
    assert l2.snapshot()["generation"] == 1


# ---------------------------------------------------------------------------
# Transfer stage / manager cancellation (PR 2 §4.1, §6.5)
# ---------------------------------------------------------------------------


class TestStageIsolationRelease:
  def test_confirmed_isolation_returns_bank_and_outstanding(self):
    """The stage-level isolation primitive returns all held resources."""
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import StageRequest, StageWait, StageWaitReason, TransferStage

    cfg = HardwareConfig()
    stage = TransferStage(
      "hbm_read",
      StageWaitReason.HBM_OUTSTANDING,
      cfg.hbm_fixed_latency_cycles,
      1024.0,
      1,
      cfg.hbm_burst_bytes,
      max_outstanding=1,
    )
    result = stage.try_issue("t0", [StageRequest("x", 4096)], cycle=0)
    assert not isinstance(result, StageWait)
    # outstanding limit reached -> wait
    blocked = stage.try_issue("t1", [StageRequest("x", 4096)], cycle=0)
    assert isinstance(blocked, StageWait)
    assert blocked.reason == StageWaitReason.HBM_OUTSTANDING
    # ``TransferStage.cancel`` is invoked only after the manager has confirmed
    # isolation; at that point the resource and credit may be returned.
    stage.cancel("t0")
    assert stage._outstanding == 0
    assert all(h is None for h in stage._holders)
    retry = stage.try_issue("t1", [StageRequest("x", 4096)], cycle=0)
    assert not isinstance(retry, StageWait)

  def test_confirmed_isolation_frees_all_bank_segments(self):
    """Confirmed isolation frees every segment of a multi-bank issue."""
    from pipeline_validator.memory.transfer import StageRequest, StageWait, StageWaitReason, TransferStage

    stage = TransferStage("l2_write", StageWaitReason.L2_BANK, 4, 12.8, 16, 1)
    result = stage.try_issue("t0", [StageRequest("0", 512), StageRequest("3", 512)], cycle=0)
    assert not isinstance(result, StageWait)
    assert stage._holders[0] == "t0"
    assert stage._holders[3] == "t0"
    stage.cancel("t0")
    assert stage._holders[0] is None
    assert stage._holders[3] is None
    assert stage._busy_until[0] == 0
    assert stage._busy_until[3] == 0
    # the same banks are usable after isolation is confirmed
    retry = stage.try_issue("t1", [StageRequest("0", 512), StageRequest("3", 512)], cycle=0)
    assert not isinstance(retry, StageWait)

  def test_step_reconciles_expired_holder(self):
    """step() clears an expired busy window and returns the credit."""
    from pipeline_validator.memory.transfer import StageRequest, StageWaitReason, TransferStage

    stage = TransferStage("hbm_read", StageWaitReason.HBM_OUTSTANDING, 0, 100.0, 1, 1, max_outstanding=1)
    stage.try_issue("t0", [StageRequest("x", 100)], cycle=0)
    assert stage._outstanding == 1
    # window = 1 cycle; step past it without an explicit release
    stage.step(cycle=5)
    assert stage._outstanding == 0
    assert stage._holders[0] is None

  def test_confirmed_isolation_is_idempotent(self):
    from pipeline_validator.memory.transfer import StageRequest, StageWaitReason, TransferStage

    stage = TransferStage("hbm_read", StageWaitReason.HBM_OUTSTANDING, 0, 100.0, 1, 1, max_outstanding=1)
    stage.try_issue("t0", [StageRequest("x", 100)], cycle=0)
    stage.cancel("t0")
    stage.cancel("t0")  # second cancel is a no-op, must not raise
    assert stage._outstanding == 0


class TestManagerCancel:
  def _manager(self, cfg):
    from pipeline_validator.memory.transfer import TransferManager

    return TransferManager(cfg, full_memory=True)

  @staticmethod
  def _txn(txn_id, owner, op, tile_id=None):
    from pipeline_validator.memory.transfer import MemoryTransaction

    return MemoryTransaction(
      transaction_id=txn_id,
      op=op,
      issuer=owner,
      src=None,
      dst=None,
      bytes_total=4096,
      completion_event=txn_id,
      tile_id=tile_id,
    )

  @staticmethod
  def _view():
    """A minimal resolved HBM/L2 view (single segment on bank 0)."""
    from pipeline_validator.memory.allocator import AllocationHandle, BankSegment
    from pipeline_validator.memory.transfer import ResolvedMemoryView

    seg = (BankSegment(0, 0, 4096),)
    handle = AllocationHandle(
      allocation_id="l2:0:1",
      memory_space="l2",
      owner=_ctx_owner(),
      base_address=0,
      size_bytes=4096,
      alignment=1,
      bank_segments=seg,
      generation=0,
      allocate_cycle=0,
    )
    return ResolvedMemoryView(handle=handle, offset_bytes=0, size_bytes=4096, address=0, segments=seg)

  def test_cancel_owner_holds_hbm_credit_until_leg_completion(self):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import TransferOp, TransferStatus

    cfg = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=4, hbm_outstanding_limit=1)
    tm = self._manager(cfg)
    o1 = _task_owner(tile=0)
    o2 = _task_owner(tile=1)
    t1 = self._txn("t1", o1, TransferOp.PREFETCH)
    t2 = self._txn("t2", o2, TransferOp.PREFETCH)
    # Real views force the full-memory HBM_READ leg.
    t1.src = self._view()
    t1.dst = self._view()
    t2.src = self._view()
    t2.dst = self._view()
    tm.submit(t1, cycle=0)
    tm.submit(t2, cycle=0)
    tm.step(cycle=0)
    completion_cycle = t1.leg_completion_cycle
    assert completion_cycle > 0
    assert tm._hbm_read._outstanding == 1
    assert tm.cancel_owner(o1, cycle=0) is False
    assert tm.status("t1") is TransferStatus.CANCEL_REQUESTED
    assert tm._hbm_read._outstanding == 1
    assert t2.leg_start_cycle == -1

    tm.step(cycle=completion_cycle - 1)
    assert tm.status("t1") is TransferStatus.CANCEL_REQUESTED
    assert tm._hbm_read._outstanding == 1
    assert t2.leg_start_cycle == -1

    # The real accepted-leg completion returns the credit.  The waiting owner
    # can then issue in that same deterministic completion window.
    tm.step(cycle=completion_cycle)
    assert tm.status("t1") is TransferStatus.CANCELLED
    assert t2.leg_start_cycle == completion_cycle
    assert tm._hbm_read._outstanding == 1

    assert tm.cancel_all(cycle=completion_cycle) is False
    assert tm.status("t2") is TransferStatus.CANCEL_REQUESTED
    tm.confirm_isolation("t2", cycle=completion_cycle)
    assert tm.cancel_all(cycle=completion_cycle) is True
    assert tm._hbm_read._outstanding == 0
    assert tm.inflight_count == 0

  def test_cancel_all_waits_for_accepted_leg_or_confirmed_isolation(self):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import TransferOp, TransferStatus

    cfg = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=4, hbm_outstanding_limit=1)
    tm = self._manager(cfg)
    owner = _task_owner(tile=0)
    accepted = self._txn("accepted", owner, TransferOp.PREFETCH)
    waiting = self._txn("waiting", owner, TransferOp.PREFETCH)
    accepted.src = self._view()
    accepted.dst = self._view()
    waiting.src = self._view()
    waiting.dst = self._view()
    tm.submit(accepted, cycle=0)
    tm.submit(waiting, cycle=0)
    tm.step(cycle=0)
    assert accepted.leg_start_cycle == 0
    assert waiting.leg_start_cycle == -1

    assert tm.cancel_all(cycle=0) is False
    assert tm.status("accepted") is TransferStatus.CANCEL_REQUESTED
    assert tm.status("waiting") is TransferStatus.CANCELLED
    assert tm._hbm_read._outstanding == 1
    assert tm.inflight_count == 1

    # Explicit isolation confirmation is the other legal way to release an
    # accepted leg before its physical completion window.
    tm.confirm_isolation("accepted", cycle=1)
    assert tm.cancel_all(cycle=1) is True
    assert tm.status("accepted") is TransferStatus.CANCELLED
    assert tm._hbm_read._outstanding == 0
    assert tm.inflight_count == 0
    for stage in tm._all_stages():
      assert stage._outstanding == 0, stage.name
      assert all(holder is None for holder in stage._holders), stage.name

  def test_other_owner_waits_for_cancelled_bank_leg_completion(self):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.allocator import AllocationHandle, BankSegment
    from pipeline_validator.memory.transfer import ResolvedMemoryView, TransferOp, TransferStatus

    cfg = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    tm = self._manager(cfg)
    o1 = _task_owner(tile=0)
    o2 = _task_owner(tile=1)
    seg = (BankSegment(0, 0, 4096),)
    h = AllocationHandle(
      allocation_id="l2:0:1",
      memory_space="l2",
      owner=o1,
      base_address=0,
      size_bytes=4096,
      alignment=1,
      bank_segments=seg,
      generation=0,
      allocate_cycle=0,
    )
    view = ResolvedMemoryView(handle=h, offset_bytes=0, size_bytes=4096, address=0, segments=seg)
    t1 = self._txn("t1", o1, TransferOp.GLOBAL_STORE)
    t1.src = view
    t1.dst = view
    tm.submit(t1, cycle=0)
    tm.step(cycle=0)
    completion_cycle = t1.leg_completion_cycle
    assert completion_cycle > 0
    assert tm._l2_read._holders[0] == "t1"
    assert tm.cancel_owner(o1, cycle=0) is False
    assert tm.status("t1") is TransferStatus.CANCEL_REQUESTED

    t2 = self._txn("t2", o2, TransferOp.GLOBAL_STORE)
    t2.src = view
    t2.dst = view
    tm.submit(t2, cycle=0)
    tm.step(cycle=completion_cycle - 1)
    assert tm._l2_read._holders[0] == "t1"
    assert t2.leg_start_cycle == -1

    tm.step(cycle=completion_cycle)
    assert tm.status("t1") is TransferStatus.CANCELLED
    assert tm._l2_read._holders[0] == "t2"
    assert t2.leg_start_cycle == completion_cycle
    assert tm.cancel_owner(o2, cycle=completion_cycle) is False
    tm.confirm_isolation("t2", cycle=completion_cycle)


# ---------------------------------------------------------------------------
# NoC router path (PR 2 §4.4 / §4.7): flit/tag enqueue, traversal, credit
# ---------------------------------------------------------------------------


class TestNoCPath:
  @staticmethod
  def _manager(cfg, noc):
    from pipeline_validator.memory.transfer import TransferManager

    return TransferManager(cfg, full_memory=True, noc=noc)

  @staticmethod
  def _txn_on_leg(tm, txn_id, owner, kind, vc_name):
    """Submit a transaction and pin its single leg to the given NoC kind."""
    from pipeline_validator.memory.transfer import MemoryTransaction, TransferLeg, TransferOp

    txn = MemoryTransaction(
      transaction_id=txn_id,
      op=TransferOp.PREFETCH,
      issuer=owner,
      src=None,
      dst=None,
      bytes_total=64,
      completion_event=txn_id,
    )
    tm.submit(txn, 0)
    txn.legs = (TransferLeg(kind, "noc", "l2", 64, vc_name),)
    txn.current_leg = 0
    txn.leg_start_cycle = -1
    return txn

  @staticmethod
  def _step(tm, noc, cycle):
    """TileGroup ordering: fabric steps first, then the manager polls."""
    traversed = noc.step(cycle)
    tm.note_traversed(traversed, cycle)
    return tm.step(cycle)

  def test_noc_leg_enqueues_traverses_and_returns_credit(self):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.noc import NoCRouter, VCId
    from pipeline_validator.memory.transfer import StageWaitReason, TransferLegKind, TransferStatus

    cfg = HardwareConfig()
    noc = NoCRouter(vc_depth=cfg.noc_vc_depth, router_latency_cycles=cfg.noc_router_latency_cycles)
    tm = self._manager(cfg, noc)
    txn = self._txn_on_leg(tm, "t1", _task_owner(tile=0), TransferLegKind.NOC_RESPONSE, "vc1")
    vc1 = noc.vcs[VCId.VC1_DMA_READ_RSP.value]
    # cycle 0: enqueue one flit/tag, wait NOC_CREDIT
    self._step(tm, noc, 0)
    assert txn.noc_tag == "t1:noc_response"
    assert noc.contains(txn.noc_tag)
    assert vc1.credit_available == cfg.noc_vc_depth  # not consumed yet
    assert txn.wait_reason == StageWaitReason.NOC_CREDIT
    # cycle 1: flit traverses (credit consumed), still waiting latency
    self._step(tm, noc, 1)
    assert not noc.contains(txn.noc_tag)
    assert vc1.credit_available == cfg.noc_vc_depth - 1
    assert txn.wait_reason == StageWaitReason.NOC_CREDIT
    # cycle 1 + router_latency: leg completes, credit returned
    for cycle in range(2, 10):
      self._step(tm, noc, cycle)
      if tm.status("t1") == TransferStatus.DONE:
        break
    assert tm.status("t1") == TransferStatus.DONE
    assert txn.completed_cycle == 1 + cfg.noc_router_latency_cycles
    assert vc1.credit_available == cfg.noc_vc_depth

  def test_credit_exhaustion_holds_pending_flit(self):
    """With one downstream credit, a second NoC flit stays pending
    (NOC_CREDIT) until the first leg returns its credit."""
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.noc import NoCRouter, VCId
    from pipeline_validator.memory.transfer import TransferLegKind, TransferStatus

    cfg = HardwareConfig().with_overrides(noc_vc_depth=1)
    noc = NoCRouter(vc_depth=1, router_latency_cycles=cfg.noc_router_latency_cycles)
    tm = self._manager(cfg, noc)
    self._txn_on_leg(tm, "t1", _task_owner(tile=0), TransferLegKind.NOC_REQUEST, "vc2")
    self._txn_on_leg(tm, "t2", _task_owner(tile=1), TransferLegKind.NOC_REQUEST, "vc2")
    vc2 = noc.vcs[VCId.VC2_DMA_WRITE.value]
    for cycle in range(0, 3):
      self._step(tm, noc, cycle)
    # t1 traversed (credit 0), t2 pending on the exhausted VC
    assert noc.contains("t2:noc_request")
    assert vc2.credit_available == 0
    assert tm.pmu_noc_credit_wait_cycles > 0
    # after t1 completes and returns credit, t2 traverses and completes
    for cycle in range(3, 20):
      self._step(tm, noc, cycle)
      if tm.status("t2") == TransferStatus.DONE:
        break
    assert tm.status("t1") == TransferStatus.DONE
    assert tm.status("t2") == TransferStatus.DONE
    assert vc2.credit_available == 1

  def test_cancel_pending_flit_removes_it(self):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.noc import NoCRouter, VCId
    from pipeline_validator.memory.transfer import TransferLegKind

    cfg = HardwareConfig()
    noc = NoCRouter(vc_depth=cfg.noc_vc_depth, router_latency_cycles=cfg.noc_router_latency_cycles)
    tm = self._manager(cfg, noc)
    self._txn_on_leg(tm, "t1", _task_owner(tile=0), TransferLegKind.NOC_RESPONSE, "vc1")
    self._step(tm, noc, 0)
    assert noc.contains("t1:noc_response")
    tm.cancel_all(cycle=0)
    assert not noc.contains("t1:noc_response")
    # credit never consumed (flit was pending), stays full
    vc1 = noc.vcs[VCId.VC1_DMA_READ_RSP.value]
    assert vc1.credit_available == cfg.noc_vc_depth

  def test_cancelled_traversed_flit_holds_credit_until_real_completion(self):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.noc import NoCRouter, VCId
    from pipeline_validator.memory.transfer import TransferLegKind, TransferStatus

    cfg = HardwareConfig().with_overrides(noc_vc_depth=1)
    noc = NoCRouter(vc_depth=1, router_latency_cycles=cfg.noc_router_latency_cycles)
    tm = self._manager(cfg, noc)
    first_owner = _task_owner(tile=0)
    second_owner = _task_owner(tile=1)
    self._txn_on_leg(tm, "t1", first_owner, TransferLegKind.NOC_RESPONSE, "vc1")
    self._step(tm, noc, 0)
    self._step(tm, noc, 1)  # the flit traverses and consumes downstream credit
    vc1 = noc.vcs[VCId.VC1_DMA_READ_RSP.value]
    assert vc1.credit_available == 0
    assert tm.cancel_owner(first_owner, cycle=1) is False
    assert tm.status("t1") is TransferStatus.CANCEL_REQUESTED
    assert vc1.credit_available == 0

    # A second owner may enqueue, but cannot traverse while the cancelled,
    # already-accepted leg still owns the only downstream credit.
    self._txn_on_leg(tm, "t2", second_owner, TransferLegKind.NOC_RESPONSE, "vc1")
    tm.step(cycle=1)
    assert noc.contains("t2:noc_response")
    release_cycle = 1 + cfg.noc_router_latency_cycles
    for cycle in range(2, release_cycle):
      self._step(tm, noc, cycle)
      assert tm.status("t1") is TransferStatus.CANCEL_REQUESTED
      assert vc1.credit_available == 0
      assert noc.contains("t2:noc_response")

    self._step(tm, noc, release_cycle)
    assert tm.status("t1") is TransferStatus.CANCELLED
    assert vc1.credit_available == 1
    assert noc.contains("t2:noc_response")

    # The waiting owner traverses only after the credit-return window.
    self._step(tm, noc, release_cycle + 1)
    assert not noc.contains("t2:noc_response")
    assert vc1.credit_available == 0
    assert tm.cancel_owner(second_owner, cycle=release_cycle + 1) is False
    assert tm.status("t2") is TransferStatus.CANCEL_REQUESTED
    tm.confirm_isolation("t2", cycle=release_cycle + 1)
    assert vc1.credit_available == 1

  def test_router_contains_and_cancel(self):
    from pipeline_validator.memory.noc import Flit, NoCRouter

    noc = NoCRouter(vc_depth=4)
    noc.send(1, Flit(vc=1, src=0, dst=1, bytes_total=64, tag="a"), cycle=0)
    noc.send(2, Flit(vc=2, src=0, dst=1, bytes_total=64, tag="b"), cycle=0)
    assert noc.contains("a")
    assert noc.contains("b")
    assert not noc.contains("missing")
    assert noc.cancel("a") == 1
    assert not noc.contains("a")
    assert noc.contains("b")
    assert noc.cancel("missing") is None


class TestHBMChannelMapping:
  @staticmethod
  def _view(address: int):
    from pipeline_validator.memory import ExternalOwner
    from pipeline_validator.memory.allocator import AllocationHandle, BankSegment
    from pipeline_validator.memory.transfer import ResolvedMemoryView

    owner = ExternalOwner("Y")
    segments = (BankSegment(0, address, 64),)
    handle = AllocationHandle(
      allocation_id=f"global:Y:{address}",
      memory_space="hbm",
      owner=owner,
      base_address=address,
      size_bytes=64,
      alignment=64,
      bank_segments=segments,
      generation=0,
      allocate_cycle=0,
    )
    return ResolvedMemoryView(
      handle=handle, offset_bytes=0, size_bytes=64, address=address, segments=segments
    )

  @staticmethod
  def _txn(tm, txn_id: str, address: int, kind, op, owner=None):
    from pipeline_validator.memory.transfer import MemoryTransaction, TransferLeg

    view = TestHBMChannelMapping._view(address)
    txn = MemoryTransaction(
      transaction_id=txn_id,
      op=op,
      issuer=owner if owner is not None else _task_owner(),
      src=view,
      dst=view,
      bytes_total=64,
      completion_event=txn_id,
    )
    tm.submit(txn, cycle=0)
    txn.legs = (TransferLeg(kind, "hbm", "noc", 64, txn_id),)
    txn.current_leg = 0
    txn.leg_start_cycle = -1
    return txn

  def test_same_address_channel_serializes(self):
    """Addresses 0 and 128 both map to channel 0 for burst=64/channels=2;
    the second HBM read must wait rather than take a free channel 1."""
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import (
      StageWaitReason,
      TransferLegKind,
      TransferManager,
      TransferOp,
    )

    cfg = HardwareConfig().with_overrides(hbm_channels=2, hbm_burst_bytes=64, hbm_fixed_latency_cycles=100)
    tm = TransferManager(cfg, full_memory=True)
    t0 = self._txn(tm, "t0", 0, TransferLegKind.HBM_READ, TransferOp.PREFETCH)
    t1 = self._txn(tm, "t1", 128, TransferLegKind.HBM_READ, TransferOp.PREFETCH)
    tm.step(cycle=0)
    assert tm._hbm_read._holders == ["t0", None]
    assert t0.leg_start_cycle == 0
    assert t1.leg_start_cycle == -1
    assert t1.wait_reason == StageWaitReason.HBM_OUTSTANDING

  def test_different_address_channels_overlap(self):
    """Addresses 0 and 64 map to channels 0 and 1, so HBM writes issue
    in the same cycle and overlap."""
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import TransferLegKind, TransferManager, TransferOp

    cfg = HardwareConfig().with_overrides(hbm_channels=2, hbm_burst_bytes=64, hbm_fixed_latency_cycles=100)
    tm = TransferManager(cfg, full_memory=True)
    t0 = self._txn(tm, "t0", 0, TransferLegKind.HBM_WRITE, TransferOp.GLOBAL_STORE)
    t1 = self._txn(tm, "t1", 64, TransferLegKind.HBM_WRITE, TransferOp.GLOBAL_STORE)
    tm.step(cycle=0)
    assert tm._hbm_write._holders == ["t0", "t1"]
    assert t0.leg_start_cycle == 0
    assert t1.leg_start_cycle == 0
    assert t0.leg_completion_cycle == t1.leg_completion_cycle

  def test_read_and_write_share_one_global_outstanding_limit(self):
    """A cancelled accepted read retains the global credit until completion."""
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import (
      StageWaitReason,
      TransferLegKind,
      TransferManager,
      TransferOp,
      TransferStatus,
    )

    cfg = HardwareConfig().with_overrides(
      hbm_channels=2, hbm_burst_bytes=64, hbm_fixed_latency_cycles=4, hbm_outstanding_limit=1
    )
    tm = TransferManager(cfg, full_memory=True)
    read = self._txn(
      tm, "read", 0, TransferLegKind.HBM_READ, TransferOp.PREFETCH, owner=_task_owner(tile=0)
    )
    write = self._txn(
      tm, "write", 64, TransferLegKind.HBM_WRITE, TransferOp.GLOBAL_STORE, owner=_task_owner(tile=1)
    )
    tm.step(cycle=0)
    completion_cycle = read.leg_completion_cycle
    assert completion_cycle > 0
    assert tm._hbm_outstanding_txns == {"read"}
    assert tm._hbm_read._holders == ["read", None]
    assert tm._hbm_write._holders == [None, None]
    assert write.leg_start_cycle == -1
    assert write.wait_reason == StageWaitReason.HBM_OUTSTANDING

    assert tm.cancel_owner(read.issuer, cycle=0) is False
    assert tm.status("read") is TransferStatus.CANCEL_REQUESTED
    assert tm._hbm_outstanding_txns == {"read"}
    tm.step(cycle=completion_cycle - 1)
    assert tm._hbm_outstanding_txns == {"read"}
    assert write.leg_start_cycle == -1

    tm.step(cycle=completion_cycle)
    assert tm.status("read") is TransferStatus.CANCELLED
    assert tm._hbm_outstanding_txns == {"write"}
    assert tm._hbm_write._holders == [None, "write"]
    assert write.leg_start_cycle == completion_cycle
    assert tm.cancel_owner(write.issuer, cycle=completion_cycle) is False
    tm.confirm_isolation("write", cycle=completion_cycle)


# ---------------------------------------------------------------------------
# Admission failure classification (PR 3.5)
# ---------------------------------------------------------------------------


class TestAdmissionClassification:
  """Typed admission failures: invalid / permanent / temporary, and
  zero side effects on every failed ``plan_bundle``."""

  @staticmethod
  def _state(alloc: BankedFreeExtentAllocator) -> dict:
    snap = alloc.snapshot()
    return {
      "free": alloc._free,
      "live": dict(alloc._live),
      "version": alloc._pool_version,
      "counter": alloc._counter,
      "peak": alloc._peak_allocated,
      "allocated": snap["allocated_bytes"],
    }

  def test_invalid_size_and_alignment_are_invalid_request(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    zero = alloc.plan_bundle([_req(_ctx_owner(), 0)])
    assert isinstance(zero, AdmissionFailure)
    assert zero.kind is AdmissionFailureKind.INVALID_REQUEST
    assert zero.reason == "invalid allocation size"
    bad_align = alloc.plan_bundle([_req(_ctx_owner(), 64, align=3)])
    assert isinstance(bad_align, AdmissionFailure)
    assert bad_align.kind is AdmissionFailureKind.INVALID_REQUEST
    assert bad_align.reason == "invalid allocation alignment"

  def test_empty_pool_impossible_bundle_is_permanent(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    plan = alloc.plan_bundle([_req(_ctx_owner(), 1025)])
    assert isinstance(plan, AdmissionFailure)
    assert plan.kind is AdmissionFailureKind.PERMANENT_CAPACITY
    assert plan.reason == "allocation capacity exceeded"
    # a multi-request bundle whose sum exceeds capacity is also permanent
    plan = alloc.plan_bundle([_req(_ctx_owner(buf="b1"), 512), _req(_ctx_owner(buf="b2"), 513)])
    assert isinstance(plan, AdmissionFailure)
    assert plan.kind is AdmissionFailureKind.PERMANENT_CAPACITY

  def test_fragmentation_miss_is_temporary_and_merges(self):
    # 2 banks x 32 bytes.  Two 16-byte requests aligned to 32 force one
    # 16-byte gap per bank (free = 32 >= 32) yet no single aligned
    # extent: a 32-byte aligned request fails on the live map but fits
    # the empty pool — temporary, not permanent.  Releasing one blocker
    # merges its bank back into a 32-byte extent and the retry fits.
    alloc = BankedFreeExtentAllocator("l2", 64, 2)
    for name in ("a", "b"):
      plan = alloc.plan_bundle([_req(_ctx_owner(gen=1, buf=name), 16, align=32, buf_id=name)])
      assert not isinstance(plan, AdmissionFailure)
      alloc.commit(plan, 0)
    assert alloc.snapshot()["free_bytes"] == 32
    frag_req = [AllocationRequest("l2", "frag", _ctx_owner(gen=2, buf="frag"), 32, 32)]
    before = self._state(alloc)
    frag = alloc.plan_bundle(frag_req)
    assert isinstance(frag, AdmissionFailure)
    assert frag.kind is AdmissionFailureKind.TEMPORARY_CAPACITY
    assert self._state(alloc) == before
    assert alloc.can_ever_fit_bundle(frag_req)
    # release blocker a: bank 0 merges back into one 32-byte extent
    handle_a = next(h for h in alloc._live.values() if h.handle.owner.buffer_id == "a").handle
    assert alloc.request_release(handle_a, handle_a.owner, 1)
    retry = alloc.plan_bundle(frag_req)
    assert not isinstance(retry, AdmissionFailure)

  def test_failed_plan_never_mutates_free_map(self):
    alloc = BankedFreeExtentAllocator("l2", 1024, 2)
    committed = alloc.plan_bundle([_req(_ctx_owner(gen=1), 256)])
    alloc.commit(committed, 0)
    before = self._state(alloc)
    outcomes = [
      alloc.plan_bundle([_req(_ctx_owner(), 0)]),
      alloc.plan_bundle([_req(_ctx_owner(), 8, align=3)]),
      alloc.plan_bundle([_req(_ctx_owner(), 4096)]),
      alloc.plan_bundle([_req(_ctx_owner(), 1024)]),
    ]
    assert all(isinstance(o, AdmissionFailure) for o in outcomes)
    after = self._state(alloc)
    assert after == before


# ---------------------------------------------------------------------------
# Deterministic Gather cache and MSHR metadata
# ---------------------------------------------------------------------------


class TestDeterministicLRUCache:
  def test_two_line_lru_touch_preserves_recent_line(self):
    cache = DeterministicLRUCache(capacity_bytes=128, line_bytes=64)
    cache.refill("A")
    cache.refill("B")
    cache.record_hit("A")
    cache.refill("C")
    snapshot = cache.snapshot()
    assert snapshot["resident_tokens"] == ("A", "C")
    assert snapshot["evictions"] == 1
    assert snapshot["hits"] == 1

  def test_anonymous_refill_is_rejected_without_fabricating_residency(self):
    cache = DeterministicLRUCache(capacity_bytes=128, line_bytes=64)
    cache.record_hit(None)
    cache.record_miss()
    with pytest.raises(MemoryInvariantError):
      cache.refill(None)
    snapshot = cache.snapshot()
    assert snapshot["hits"] == 1
    assert snapshot["misses"] == 1
    assert snapshot["refills"] == 0
    assert snapshot["resident_lines"] == 0
    assert snapshot["resident_tokens"] == ()

  def test_reset_clears_residency_and_statistics_but_preserves_configuration(self):
    cache = DeterministicLRUCache(capacity_bytes=64, line_bytes=64)
    cache.record_miss()
    cache.refill("A")
    assert cache.snapshot()["resident_tokens"] == ("A",)
    cache.reset()
    snapshot = cache.snapshot()
    assert snapshot["hits"] == 0
    assert snapshot["misses"] == 0
    assert snapshot["refills"] == 0
    assert snapshot["evictions"] == 0
    assert snapshot["resident_lines"] == 0
    assert snapshot["resident_bytes"] == 0
    assert snapshot["resident_tokens"] == ()
    assert snapshot["capacity_bytes"] == 64

  def test_cache_metadata_does_not_consume_spm_allocator_capacity(self):
    l1 = BankedFreeExtentAllocator("l1", 1024, 2)
    l2 = BankedFreeExtentAllocator("l2", 2048, 2)
    before_l1 = l1.snapshot()
    before_l2 = l2.snapshot()
    cache = DeterministicLRUCache(capacity_bytes=128, line_bytes=64)
    cache.refill("A")
    cache.refill("B")
    cache.record_hit("A")
    assert l1.snapshot() == before_l1
    assert l2.snapshot() == before_l2


class TestMshrTable:
  def test_non_null_group_has_one_leader_and_waiters(self):
    table = MshrTable(capacity=2)
    leader = table.allocate("line42")
    waiter = table.allocate("line42")
    assert isinstance(leader, MshrAllocation)
    assert isinstance(waiter, MshrAllocation)
    assert leader.leader
    assert not waiter.leader
    assert leader.token == waiter.token
    assert table.snapshot()["active"] == 1
    assert table.snapshot()["merged"] == 1

  def test_anonymous_misses_never_merge(self):
    table = MshrTable(capacity=2)
    first = table.allocate()
    second = table.allocate()
    assert isinstance(first, MshrAllocation)
    assert isinstance(second, MshrAllocation)
    assert first.leader and second.leader
    assert first.token != second.token
    assert table.snapshot()["active"] == 2
    assert table.snapshot()["merged"] == 0

  def test_capacity_wait_is_structured_and_version_gated(self):
    table = MshrTable(capacity=1)
    leader = table.allocate("A")
    assert isinstance(leader, MshrAllocation)
    wait = table.allocate("B")
    assert wait == MshrWait(reason="mshr_full", version=0)
    assert table.snapshot()["active"] == 1
    assert table.version == wait.version
    table.complete(leader.token)
    assert table.version != wait.version
    retry = table.allocate("B")
    assert isinstance(retry, MshrAllocation)
    assert retry.leader

  def test_complete_returns_callbacks_exactly_once(self):
    table = MshrTable(capacity=1)
    allocation = table.allocate("A")
    assert isinstance(allocation, MshrAllocation)
    callbacks: list[str] = []
    table.wait(allocation.token, lambda: callbacks.append("ready"))
    ready = table.complete(allocation.token)
    assert len(ready) == 1
    ready[0]()
    assert callbacks == ["ready"]
    with pytest.raises(MemoryInvariantError, match="unknown or completed MSHR token"):
      table.complete(allocation.token)

  def test_reset_rejects_active_refills_then_clears_completed_state(self):
    table = MshrTable(capacity=1)
    allocation = table.allocate("A")
    assert isinstance(allocation, MshrAllocation)
    callbacks: list[str] = []
    table.wait(allocation.token, lambda: callbacks.append("ready"))
    assert isinstance(table.allocate("B"), MshrWait)
    before_reset = table.snapshot()

    with pytest.raises(MemoryInvariantError):
      table.reset()
    assert table.snapshot()["active"] == 1
    assert table.snapshot()["callbacks"] == 1
    assert table.snapshot()["entries"] == before_reset["entries"]

    ready = table.complete(allocation.token)
    assert len(ready) == 1
    ready[0]()
    assert callbacks == ["ready"]
    version_after_completion = table.version
    table.reset()
    snapshot = table.snapshot()
    assert snapshot["active"] == 0
    assert snapshot["callbacks"] == 0
    assert snapshot["entries"] == ()
    assert snapshot["merged"] == 0
    assert snapshot["stalls"] == 0
    assert snapshot["capacity"] == 1
    assert table.version > version_after_completion


class TestGatherTransferRoutes:
  @staticmethod
  def _transaction(transaction_id, op, *, src=None, dst=None, owner=None, bytes_total=64):
    from pipeline_validator.memory.transfer import MemoryTransaction

    return MemoryTransaction(
      transaction_id=transaction_id,
      op=op,
      issuer=_task_owner() if owner is None else owner,
      src=src,
      dst=dst,
      bytes_total=bytes_total,
      completion_event=transaction_id,
      tile_id=0,
    )

  @staticmethod
  def _view(space: str, owner=None):
    from pipeline_validator.memory.allocator import AllocationHandle, BankSegment
    from pipeline_validator.memory.transfer import ResolvedMemoryView

    actual_owner = _task_owner() if owner is None else owner
    segments = (BankSegment(0, 0, 64), BankSegment(1, 64, 64))
    handle = AllocationHandle(
      allocation_id=f"{space}:0:1",
      memory_space=space,
      owner=actual_owner,
      base_address=0,
      size_bytes=128,
      alignment=1,
      bank_segments=segments,
      generation=0,
      allocate_cycle=0,
    )
    return ResolvedMemoryView(
      handle=handle, offset_bytes=0, size_bytes=128, address=0, segments=segments, permissions="r"
    )

  @pytest.mark.parametrize("full_memory", [False, True])
  def test_gather_routes_have_exact_leg_sequences(self, full_memory):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import TransferLegKind, TransferManager, TransferOp

    # Plan §2: address-driven Gather route set.  The MISS_LOOKUP leg
    # sequence from the profiled era is gone; lookups are standalone
    # transactions and SCATTER_WRITE writes through both caches.
    expected = {
      TransferOp.INDEX_READ: (TransferLegKind.L1_READ,),
      TransferOp.GATHER_L1_LOOKUP: (TransferLegKind.L1_CACHE_LOOKUP,),
      TransferOp.GATHER_L2_LOOKUP: (TransferLegKind.L2_CACHE_LOOKUP,),
      TransferOp.GATHER_L2_RESPONSE: (TransferLegKind.NOC_RESPONSE, TransferLegKind.LOCAL_DMA),
      TransferOp.GATHER_HBM_REFILL: (
        TransferLegKind.HBM_READ,
        TransferLegKind.NOC_RESPONSE,
        TransferLegKind.L2_CACHE_FILL,
      ),
      TransferOp.GATHER_L2_REFILL: (
        TransferLegKind.NOC_RESPONSE,
        TransferLegKind.LOCAL_DMA,
        TransferLegKind.L1_CACHE_FILL,
      ),
      TransferOp.GATHER_DEST_WRITE: (TransferLegKind.L1_WRITE,),
      TransferOp.SCATTER_WRITE: (
        TransferLegKind.L1_READ,
        TransferLegKind.LOCAL_DMA,
        TransferLegKind.NOC_REQUEST,
        TransferLegKind.GLOBAL_DMA,
        TransferLegKind.HBM_WRITE,
      ),
    }
    manager = TransferManager(HardwareConfig(), full_memory=full_memory)
    for index, (op, expected_legs) in enumerate(expected.items()):
      transaction = self._transaction(f"route:{index}", op)
      manager.submit(transaction, cycle=0)
      assert tuple(leg.kind for leg in transaction.legs) == expected_legs
      if op is not TransferOp.SCATTER_WRITE:
        # Scatter writes through the Global DMA to HBM (plan §2); gathers
        # never take the Global DMA or write the L2 data path.
        assert TransferLegKind.GLOBAL_DMA not in expected_legs
        assert TransferLegKind.L2_WRITE not in expected_legs

  def test_gather_route_cannot_collapse(self):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import TransferManager, TransferOp

    manager = TransferManager(HardwareConfig(), full_memory=False)
    transaction = self._transaction("no-collapse", TransferOp.GATHER_L1_LOOKUP)
    with pytest.raises(MemoryInvariantError, match="gather route must not be collapsed"):
      manager._collapsed_leg(transaction)

  def test_slice_resolved_view_clips_segments_and_checks_bounds(self):
    from pipeline_validator.memory.transfer import slice_resolved_view

    view = self._view("l1")
    sliced = slice_resolved_view(view, 32, 64)
    assert sliced is not None
    assert sliced.address == 32
    assert sliced.offset_bytes == 32
    assert [(segment.bank_id, segment.address, segment.size_bytes) for segment in sliced.segments] == [
      (0, 32, 32),
      (1, 64, 32),
    ]
    assert slice_resolved_view(None, 0, 64) is None
    with pytest.raises(MemoryInvariantError, match="memory view out of bounds"):
      slice_resolved_view(view, 96, 64)

  def test_slice_resolved_view_uses_logical_cursor_for_fragmented_allocation(self):
    from pipeline_validator.memory.transfer import ResolvedMemoryView, slice_resolved_view

    allocator = BankedFreeExtentAllocator("l1", 256, 2)
    owner_a = _task_owner(buf="a")
    owner_b = _task_owner(buf="b")
    initial = allocator.plan_bundle(
      [_req(owner_a, 32, buf_id="a", space="l1"), _req(owner_b, 32, buf_id="b", space="l1")]
    )
    assert not isinstance(initial, AdmissionFailure)
    handle_a, _handle_b = allocator.commit(initial, cycle=0)
    allocator.request_release(handle_a, owner_a, cycle=1)

    destination_owner = _task_owner(buf="destination")
    fragmented = allocator.plan_bundle([_req(destination_owner, 96, buf_id="destination", space="l1")])
    assert not isinstance(fragmented, AdmissionFailure)
    handle = allocator.commit(fragmented, cycle=2)[0]
    assert [(segment.bank_id, segment.address, segment.size_bytes) for segment in handle.bank_segments] == [
      (0, 0, 32),
      (0, 64, 64),
    ]
    view = ResolvedMemoryView(
      handle=handle,
      offset_bytes=0,
      size_bytes=96,
      address=handle.bank_segments[0].address,
      segments=handle.bank_segments,
    )
    sliced = slice_resolved_view(view, 16, 64)
    assert sliced is not None
    assert sliced.address == 16
    assert [(segment.bank_id, segment.address, segment.size_bytes) for segment in sliced.segments] == [
      (0, 16, 16),
      (0, 64, 48),
    ]

  def test_cancel_holds_gather_resources_until_completion_or_isolation(self):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.noc import NoCRouter, VCId
    from pipeline_validator.memory.transfer import TransferManager, TransferOp, TransferStatus

    config = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=4, hbm_outstanding_limit=1)
    noc = NoCRouter(vc_depth=config.noc_vc_depth, router_latency_cycles=config.noc_router_latency_cycles)
    manager = TransferManager(config, full_memory=True, noc=noc)
    owner = _task_owner()

    lookup = self._transaction("lookup", TransferOp.GATHER_L1_LOOKUP, owner=owner)
    manager.submit(lookup, cycle=0)
    manager.step(cycle=0)
    lookup_completion = lookup.leg_completion_cycle
    assert lookup_completion > 0
    assert manager._l1_cache_lookup[0]._holders[0] == "lookup"
    assert manager.cancel_owner(owner, cycle=0) is False
    assert manager.status("lookup") is TransferStatus.CANCEL_REQUESTED
    assert manager._l1_cache_lookup[0]._holders[0] == "lookup"
    manager.step(cycle=lookup_completion - 1)
    assert manager._l1_cache_lookup[0]._holders[0] == "lookup"
    manager.step(cycle=lookup_completion)
    assert manager.status("lookup") is TransferStatus.CANCELLED
    assert manager._l1_cache_lookup[0]._holders[0] is None

    hbm_start = lookup_completion + 1
    hbm = self._transaction("hbm", TransferOp.GATHER_HBM_REFILL, src=self._view("hbm", owner), owner=owner, bytes_total=128)
    manager.submit(hbm, cycle=hbm_start)
    manager.step(cycle=hbm_start)
    hbm_completion = hbm.leg_completion_cycle
    assert hbm_completion > hbm_start
    assert manager._hbm_read._outstanding == 1
    assert manager.cancel_owner(owner, cycle=hbm_start) is False
    assert manager.status("hbm") is TransferStatus.CANCEL_REQUESTED
    assert manager._hbm_read._outstanding == 1
    manager.step(cycle=hbm_completion - 1)
    assert manager._hbm_read._outstanding == 1
    manager.step(cycle=hbm_completion)
    assert manager.status("hbm") is TransferStatus.CANCELLED
    assert manager._hbm_read._outstanding == 0

    refill_start = hbm_completion + 1
    refill = self._transaction("refill", TransferOp.GATHER_L2_REFILL, owner=owner)
    manager.submit(refill, cycle=refill_start)
    manager.step(cycle=refill_start)
    traversal_cycle = refill_start + 1
    traversed = noc.step(cycle=traversal_cycle)
    manager.note_traversed(traversed, cycle=traversal_cycle)
    manager.step(cycle=traversal_cycle)
    vc1 = noc.vcs[VCId.VC1_DMA_READ_RSP.value]
    assert vc1.credit_available == config.noc_vc_depth - 1
    assert manager.cancel_owner(owner, cycle=traversal_cycle) is False
    assert manager.status("refill") is TransferStatus.CANCEL_REQUESTED
    assert vc1.credit_available == config.noc_vc_depth - 1

    noc_completion = traversal_cycle + config.noc_router_latency_cycles
    for cycle in range(traversal_cycle + 1, noc_completion):
      traversed = noc.step(cycle)
      manager.note_traversed(traversed, cycle)
      manager.step(cycle)
      assert manager.status("refill") is TransferStatus.CANCEL_REQUESTED
      assert vc1.credit_available == config.noc_vc_depth - 1
    traversed = noc.step(noc_completion)
    manager.note_traversed(traversed, noc_completion)
    manager.step(noc_completion)
    assert manager.status("refill") is TransferStatus.CANCELLED
    assert vc1.credit_available == config.noc_vc_depth

    destination_start = noc_completion + 1
    destination = self._transaction(
      "destination", TransferOp.GATHER_DEST_WRITE, dst=self._view("l1", owner), owner=owner, bytes_total=128
    )
    manager.submit(destination, cycle=destination_start)
    manager.step(cycle=destination_start)
    assert manager._l1_write[0]._holders[0] == "destination"
    assert manager.cancel_owner(owner, cycle=destination_start) is False
    assert manager.status("destination") is TransferStatus.CANCEL_REQUESTED
    assert manager._l1_write[0]._holders[0] == "destination"
    manager.confirm_isolation("destination", cycle=destination_start)
    assert manager.status("destination") is TransferStatus.CANCELLED

    snapshot = manager.snapshot()
    assert manager.inflight_count == 0
    assert manager._hbm_read._outstanding == 0
    assert vc1.credit_available == config.noc_vc_depth
    assert all(
      stage["busy_resources"] == 0 and stage["outstanding"] == 0 for stage in snapshot["stages"].values()
    )


class TestTransferReferenceHooks:
  """plan/01 §3/§6.3: submit-level acquisition failure faults the accepted
  transaction and never releases a reference it does not hold."""

  @staticmethod
  def _physical_view(allocation_id: str):
    from pipeline_validator.memory.allocator import AllocationHandle, BankSegment
    from pipeline_validator.memory.transfer import ResolvedMemoryView

    seg = (BankSegment(0, 0, 4096),)
    handle = AllocationHandle(
      allocation_id=allocation_id,
      memory_space="l2",
      owner=_ctx_owner(),
      base_address=0,
      size_bytes=4096,
      alignment=1,
      bank_segments=seg,
      generation=0,
      allocate_cycle=0,
      backing_id=f"{allocation_id}:backing",
    )
    return ResolvedMemoryView(handle=handle, offset_bytes=0, size_bytes=4096, address=0, segments=seg)

  def test_acquire_failure_faults_transaction_without_release(self):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import (
      MemoryTransaction,
      TransferManager,
      TransferOp,
      TransferStatus,
    )

    acquired: list[str] = []
    released: list[str] = []

    def acquire(txn, cycle):
      acquired.append(txn.transaction_id)
      raise MemoryInvariantError("second endpoint went stale after rollback")

    def release(txn, cycle):
      released.append(txn.transaction_id)

    tm = TransferManager(
      HardwareConfig(), full_memory=True, reference_acquire=acquire, reference_release=release
    )
    txn = MemoryTransaction(
      transaction_id="t1",
      op=TransferOp.PREFETCH,
      issuer=_task_owner(tile=0),
      src=self._physical_view("l2:a"),
      dst=self._physical_view("l2:b"),
      bytes_total=4096,
      completion_event="e",
    )
    tm.submit(txn, cycle=0)
    assert tm.status("t1") is TransferStatus.FAULTED
    assert txn.reference_acquired is False
    assert acquired == ["t1"]
    assert released == []
    # Terminal acknowledgement of the failed acquisition releases nothing.
    tm.acknowledge("t1", cycle=5)
    assert released == []
    assert tm.outstanding_transactions == ()


class TestAcceptedNotIssuedPrefetch:
  """plan/01 §6.3: an accepted-but-not-issued prefetch through the real
  submit path holds its physical backing until a terminal acknowledgement."""

  @staticmethod
  def _pool_with_manager():
    from dataclasses import replace

    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.arena import ArenaPool
    from pipeline_validator.memory.transfer import TransferManager
    from pipeline_validator.profiles import build_registry

    profile = replace(
      build_registry(HardwareConfig()).profile("l2", 0),
    )
    pool = ArenaPool(profile)
    acquired: list[str] = []
    released: list[str] = []

    def acquire(txn, cycle):
      acquired.append(txn.transaction_id)
      for view in (txn.src, txn.dst):
        if view is not None and view.handle.memory_space == "l2" and view.handle.backing_id:
          pool.begin_inflight(view.handle, txn.transaction_id)

    def release(txn, cycle):
      released.append(txn.transaction_id)
      for view in (txn.src, txn.dst):
        if view is not None and view.handle.memory_space == "l2" and view.handle.backing_id:
          if not pool.is_released(view.handle):
            pool.end_inflight(view.handle, txn.transaction_id, cycle)

    manager = TransferManager(
      HardwareConfig(),
      full_memory=True,
      reference_acquire=acquire,
      reference_release=release,
      generation_validator=lambda transaction, phase: True,
    )
    manager.configure_profile_generations({("l2", 0): 0})
    manager.begin_run(1)
    return pool, manager, acquired, released

  def test_not_issued_prefetch_holds_backing_until_cancel_and_ack(self):
    from pipeline_validator.compiler.resources import conservative_arena_bytes, layout_buffers
    from pipeline_validator.execution_ir import ExecL2Buffer
    from pipeline_validator.memory import AdmissionFailure
    from pipeline_validator.memory.allocator import AllocationHandle, BankSegment, ExternalOwner
    from pipeline_validator.memory.arena import RootInvocation
    from pipeline_validator.memory.transfer import (
      MemoryTransaction,
      ResolvedMemoryView,
      TransferOp,
      TransferStatus,
    )

    pool, manager, acquired, released = self._pool_with_manager()
    buffer = ExecL2Buffer("buf", (4096,), "i8", "in", 1, 64, 4096)
    reserved = conservative_arena_bytes([(buffer.bytes, buffer.alignment)], pool.profile)
    layout = layout_buffers((buffer,), pool.profile, reserved, lifetimes=None, slot_capacity=1)
    plan = pool.plan_arena(RootInvocation("ctx", 0), layout)
    assert not isinstance(plan, AdmissionFailure)
    arena = pool.commit_arena(plan, 0)
    view = pool.bind_view(arena, "buf", 0)
    committed_free = pool.snapshot()["free_bytes"]

    seg = (BankSegment(0, 0, view.size_bytes),)
    hbm_handle = AllocationHandle(
      allocation_id="hbm:src",
      memory_space="hbm",
      owner=ExternalOwner("src"),
      base_address=0x100000,
      size_bytes=view.size_bytes,
      alignment=1,
      bank_segments=seg,
      generation=0,
      allocate_cycle=0,
    )
    src_view = ResolvedMemoryView(
      handle=hbm_handle, offset_bytes=0, size_bytes=view.size_bytes,
      address=0x100000, segments=seg,
    )
    dst_view = ResolvedMemoryView(
      handle=view, offset_bytes=0, size_bytes=view.size_bytes,
      address=view.base_address, segments=view.bank_segments,
    )
    transaction = MemoryTransaction(
      transaction_id="prefetch:held",
      op=TransferOp.PREFETCH,
      issuer=view.owner,
      src=src_view,
      dst=dst_view,
      bytes_total=view.size_bytes,
      completion_event="held_done",
      run_generation=1,
      profile_generations=(("l2", 0, 0),),
    )
    manager.submit(transaction, 0)
    # Accepted but never issued: no leg has started, the ledger holds the
    # backing, and the owner's invalidation can only stay pending.
    assert transaction.leg_start_cycle == -1
    assert acquired == ["prefetch:held"]
    assert not pool.invalidate_view(view, view.owner, 1)
    held = pool.snapshot()
    assert held["live_backings"] == 1
    assert held["free_bytes"] == committed_free

    # A never-issued leg cancels synchronously; the reference survives until
    # the terminal acknowledgement.
    assert manager.cancel_all(2) is True
    assert transaction.status is TransferStatus.CANCELLED
    still_held = pool.snapshot()
    assert still_held["live_backings"] == 1
    assert still_held["free_bytes"] == committed_free

    manager.acknowledge("prefetch:held", 3)
    assert released == ["prefetch:held"]
    # The pending invalidation completed with the reference drop: the full
    # padded span is back and the view is released.
    assert pool.is_released(view)
    after = pool.snapshot()
    assert after["live_backings"] == 0
    assert after["free_bytes"] == after["user_spm_capacity_bytes"]


class TestExplicitL2PoolClaims:
  @staticmethod
  def _pool(*, trace: bool = False):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.arena import ArenaPool
    from pipeline_validator.profiles import build_registry

    hw = HardwareConfig()
    tracer = None
    if trace:
      from pipeline_validator.trace import MemoryTrace, Tracer

      tracer = Tracer(hw)
      memory_trace = MemoryTrace(tracer)
    else:
      memory_trace = None
    profile = build_registry(hw).profile("l2", 0)
    return ArenaPool(profile, trace=memory_trace), profile, tracer

  @staticmethod
  def _layout(pool, buffer_bytes: int = 4096):
    from pipeline_validator.compiler.resources import conservative_arena_bytes, layout_buffers
    from pipeline_validator.execution_ir import ExecL2Buffer

    buffer = ExecL2Buffer("weight", (buffer_bytes,), "i8", "inout", 1, 64, buffer_bytes)
    reserved = conservative_arena_bytes([(buffer.bytes, buffer.alignment)], pool.profile)
    layout = layout_buffers((buffer,), pool.profile, reserved, lifetimes=None, slot_capacity=1)
    return buffer, layout

  def test_claim_validation_failure_preserves_plan_pool_version_and_trace(self):
    from pipeline_validator.memory import AdmissionFailure
    from pipeline_validator.memory.arena import RootInvocation

    pool, _profile, tracer = self._pool(trace=True)
    _buffer, layout = self._layout(pool)
    plan = pool.plan_arena(RootInvocation("producer", 0), layout)
    assert not isinstance(plan, AdmissionFailure)
    before = pool.snapshot()
    trace_before = tracer.to_chrome_json()

    with pytest.raises(MemoryInvariantError, match="unknown arena buffer"):
      pool.commit_arena(plan, 1, claims_by_slot={"missing": (("reader", "weight"),)})

    assert pool.snapshot() == before
    assert pool.pool_version == before["pool_version"]
    assert tracer.to_chrome_json() == trace_before

    arena = pool.commit_arena(plan, 1)
    view = pool.bind_view(arena, "weight", 2)
    assert pool.invalidate_view(view, view.owner, 3)
    assert pool.retire_arena(arena, 4)

  def test_published_claim_survives_origin_retirement_until_last_alias_ack(self):
    from pipeline_validator.compiler.resources import layout_buffers
    from pipeline_validator.memory import AdmissionFailure
    from pipeline_validator.memory.arena import RootInvocation

    pool, _profile, _tracer = self._pool()
    _buffer, layout = self._layout(pool)
    claim_id = ("reader-submit-1", "weight")
    producer_plan = pool.plan_arena(RootInvocation("producer", 0), layout)
    assert not isinstance(producer_plan, AdmissionFailure)
    producer_arena = pool.commit_arena(
      producer_plan, 0, claims_by_slot={"weight": (claim_id,)}
    )
    producer = pool.bind_view(producer_arena, "weight", 1)
    backing_id = producer.backing_id
    initial = pool.snapshot()
    assert initial["pending_shared_claims"] == 1
    assert initial["active_shared_references"] == 0
    assert not pool.can_borrow_l2_view(
      backing_id, ContextBufferOwner("reader", 7, "weight"), claim_id
    )
    assert pool.snapshot() == initial

    assert pool.permissions(producer) == "rw"
    pool.assert_access(producer, "w")
    pool.pin(producer, "writer:0")
    before_publish_check = pool.snapshot()
    with pytest.raises(MemoryInvariantError, match="unexpected active pins"):
      pool.check_publish_l2(producer)
    assert pool.snapshot() == before_publish_check
    pool.check_publish_l2(producer, allowed_pins=("writer:0",))
    assert pool.snapshot() == before_publish_check
    assert not pool.unpin(producer, "writer:0", 2)
    pool.publish_l2(producer, 2)
    assert pool.permissions(producer) == "r"
    with pytest.raises(MemoryInvariantError, match="read-only"):
      pool.assert_access(producer, "w")
    assert pool.can_borrow_l2_view(
      backing_id, ContextBufferOwner("reader", 7, "weight"), claim_id
    )

    assert pool.invalidate_view(producer, producer.owner, 3)
    committed_version = pool.pool_version
    assert pool.retire_arena(producer_arena, 4)
    origin_retired = pool.snapshot()
    assert pool.pool_version == committed_version
    assert origin_retired["live_arenas"] == 0
    assert origin_retired["live_backings"] == 1
    assert origin_retired["origin_retired_backings"][0]["backing_id"] == backing_id
    assert origin_retired["arena_reserved_bytes"] == producer_arena.reserved_bytes
    assert pool.arena_held_bytes(producer_arena) == 0

    reader_owner = RootInvocation("reader", 7)
    reader_layout = layout_buffers((), pool.profile, 0, lifetimes=None, slot_capacity=1)
    reader_plan = pool.plan_arena(reader_owner, reader_layout)
    assert not isinstance(reader_plan, AdmissionFailure)
    reader_arena = pool.commit_arena(reader_plan, 5)
    assert pool.pool_version == committed_version
    reader = pool.borrow_l2_view(
      backing_id, ContextBufferOwner("reader", 7, "weight"), claim_id, 6
    )
    assert reader.allocation_id != producer.allocation_id
    assert reader.backing_id == producer.backing_id
    assert reader.generation == producer.generation
    assert reader.profile_generation == producer.profile_generation
    assert reader.bank_segments == producer.bank_segments
    assert pool.pool_version == committed_version
    active = pool.snapshot()
    assert active["live_backings"] == 1
    assert active["pending_shared_claims"] == 0
    assert active["active_shared_references"] == 1
    assert active["arena_reserved_bytes"] == initial["arena_reserved_bytes"]
    assert active["live_view_bytes"] == producer.size_bytes
    assert active["logical_live_view_bytes"] == reader.size_bytes
    for bank in active["per_bank_occupancy"]:
      assert bank["allocated_bytes"] + bank["free_bytes"] == pool.profile.user_spm_per_bank

    pool.begin_inflight(reader, "reader:transfer")
    assert not pool.has_inflight_references(producer)
    assert pool.has_inflight_references(reader)
    assert not pool.invalidate_view(reader, reader.owner, 7)
    assert pool.permissions(reader) == "r"
    before_denied_write = pool.snapshot()
    with pytest.raises(MemoryInvariantError, match="read-only"):
      pool.assert_access(reader, "w")
    assert pool.snapshot() == before_denied_write
    assert pool.end_inflight(reader, "reader:transfer", 8)

    released = pool.snapshot()
    assert released["live_backings"] == 0
    assert released["pending_shared_claims"] == 0
    assert released["active_shared_references"] == 0
    assert released["arena_reserved_bytes"] == 0
    assert released["free_bytes"] == released["user_spm_capacity_bytes"]
    assert released["pool_version"] == committed_version + 1
    assert pool.backing_claims(backing_id) == ()
    assert pool.claim_snapshot()[0]["state"] == "RELEASED"
    assert len(pool.drain_l2_backing_release_events()) == 1

    before_double_release = pool.snapshot()
    with pytest.raises(MemoryInvariantError, match="double release"):
      pool.invalidate_view(reader, reader.owner, 9)
    assert pool.snapshot() == before_double_release
    assert pool.retire_arena(reader_arena, 10)
    assert pool.pool_version == before_double_release["pool_version"]
    pool.close_l2_claims()
    assert pool.claim_snapshot() == ()

  def test_retained_capacity_query_uses_physical_spans_without_mutation(self):
    from pipeline_validator.compiler.resources import layout_buffers
    from pipeline_validator.memory import AdmissionFailure
    from pipeline_validator.memory.arena import RootInvocation

    pool, profile, _tracer = self._pool()
    _buffer, layout = self._layout(pool)
    claim_id = ("reader-submit", "weight")
    plan = pool.plan_arena(RootInvocation("producer", 0), layout)
    assert not isinstance(plan, AdmissionFailure)
    arena = pool.commit_arena(plan, 0, claims_by_slot={"weight": (claim_id,)})
    producer = pool.bind_view(arena, "weight", 1)
    pool.publish_l2(producer, 2)
    backing_id = producer.backing_id

    full_layout = layout_buffers(
      (), profile, profile.user_spm_bytes, lifetimes=None, slot_capacity=1
    )
    before = pool.snapshot()
    assert pool.can_fit_with_retained(full_layout, ())
    assert not pool.can_fit_with_retained(full_layout, (backing_id,))
    assert pool.snapshot() == before

    assert pool.invalidate_view(producer, producer.owner, 3)
    assert pool.cancel_l2_claim(backing_id, claim_id, 4)
    after_cancel = pool.snapshot()
    assert not pool.cancel_l2_claim(backing_id, claim_id, 5)
    assert pool.snapshot() == after_cancel
    assert pool.retire_arena(arena, 5)
    pool.close_l2_claims()

  def test_fault_cancelled_bound_alias_is_idempotent_but_public_release_is_strict(self):
    from pipeline_validator.compiler.resources import layout_buffers
    from pipeline_validator.memory import AdmissionFailure
    from pipeline_validator.memory.arena import RootInvocation

    pool, _profile, _tracer = self._pool()
    _buffer, layout = self._layout(pool)
    claim_id = ("reader-submit", "weight")
    producer_plan = pool.plan_arena(RootInvocation("producer", 0), layout)
    assert not isinstance(producer_plan, AdmissionFailure)
    producer_arena = pool.commit_arena(
      producer_plan, 0, claims_by_slot={"weight": (claim_id,)}
    )
    producer = pool.bind_view(producer_arena, "weight", 1)
    pool.publish_l2(producer, 2)

    reader_owner = RootInvocation("reader", 7)
    reader_layout = layout_buffers((), pool.profile, 0, lifetimes=None, slot_capacity=1)
    reader_plan = pool.plan_arena(reader_owner, reader_layout)
    assert not isinstance(reader_plan, AdmissionFailure)
    reader_arena = pool.commit_arena(reader_plan, 3)
    reader = pool.borrow_l2_view(
      producer.backing_id, ContextBufferOwner("reader", 7, "weight"), claim_id, 4
    )
    assert pool.invalidate_view(producer, producer.owner, 5)
    assert pool.retire_arena(producer_arena, 6)
    assert pool.snapshot()["active_shared_references"] == 1

    assert pool.cancel_l2_view(reader, reader.owner, 7)
    cancelled = pool.snapshot()
    assert cancelled["live_backings"] == 0
    assert cancelled["active_shared_references"] == 0
    assert pool.claim_snapshot()[0]["state"] == "CANCELLED"
    assert len(pool.drain_l2_backing_release_events()) == 1
    assert not pool.cancel_l2_view(reader, reader.owner, 8)
    assert pool.snapshot() == cancelled
    with pytest.raises(MemoryInvariantError, match="double release"):
      pool.invalidate_view(reader, reader.owner, 9)
    assert pool.snapshot() == cancelled

    assert pool.retire_arena(reader_arena, 10)
    pool.close_l2_claims()
