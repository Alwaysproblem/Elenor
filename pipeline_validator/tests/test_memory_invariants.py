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
  AllocationRequest,
  BankedFreeExtentAllocator,
  ContextBufferOwner,
  ExternalOwner,
  HBMRegion,
  MemoryInvariantError,
  TaskBufferOwner,
)


def _ctx_owner(name: str = "ctx", gen: int = 0, buf: str = "b") -> ContextBufferOwner:
  return ContextBufferOwner(name, gen, buf)


def _task_owner(tile: int = 0, ctx: int = 0, task: int = 0,
                buf: str = "l1") -> TaskBufferOwner:
  return TaskBufferOwner("ctx", 0, "ev_role", task, tile, ctx, buf)


def _req(owner, size: int, align: int = 1, buf_id: str = "b",
         space: str = "l2") -> AllocationRequest:
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
    h2 = alloc.commit(
      alloc.plan_bundle([_req(o, 1024, align=256, buf_id="b2")]) or [], 0)[0]
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
    plan = alloc.plan_bundle([_req(owners[i], 512, buf_id=f"b{i}")
                              for i in range(4)])
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
    plan = alloc.plan_bundle([
      _req(o, 512, buf_id="ok"),
      _req(o, 600, buf_id="big"),
    ])
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


class TestStageCancel:
  def test_cancel_returns_bank_and_outstanding(self):
    """Cancelling a transaction frees its bank and outstanding credit
    immediately, so a later issue succeeds at the same cycle."""
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import StageRequest, StageWait, StageWaitReason, TransferStage
    cfg = HardwareConfig()
    stage = TransferStage(
      "hbm_read", StageWaitReason.HBM_OUTSTANDING,
      cfg.hbm_fixed_latency_cycles, 1024.0, 1,
      cfg.hbm_burst_bytes, max_outstanding=1)
    result = stage.try_issue("t0", [StageRequest("x", 4096)], cycle=0)
    assert not isinstance(result, StageWait)
    # outstanding limit reached -> wait
    blocked = stage.try_issue("t1", [StageRequest("x", 4096)], cycle=0)
    assert isinstance(blocked, StageWait)
    assert blocked.reason == StageWaitReason.HBM_OUTSTANDING
    # cancel t0 -> resource + credit returned
    stage.cancel("t0")
    assert stage._outstanding == 0
    assert all(h is None for h in stage._holders)
    retry = stage.try_issue("t1", [StageRequest("x", 4096)], cycle=0)
    assert not isinstance(retry, StageWait)

  def test_cancel_bank_based_stage_frees_all_segments(self):
    """Cancel frees every bank segment a multi-segment issue occupied."""
    from pipeline_validator.memory.transfer import StageRequest, StageWait, StageWaitReason, TransferStage
    stage = TransferStage(
      "l2_write", StageWaitReason.L2_BANK, 4, 12.8, 16, 1)
    result = stage.try_issue(
      "t0", [StageRequest("0", 512), StageRequest("3", 512)], cycle=0)
    assert not isinstance(result, StageWait)
    assert stage._holders[0] == "t0"
    assert stage._holders[3] == "t0"
    stage.cancel("t0")
    assert stage._holders[0] is None
    assert stage._holders[3] is None
    assert stage._busy_until[0] == 0
    assert stage._busy_until[3] == 0
    # same banks usable immediately
    retry = stage.try_issue(
      "t1", [StageRequest("0", 512), StageRequest("3", 512)], cycle=0)
    assert not isinstance(retry, StageWait)

  def test_step_reconciles_expired_holder(self):
    """step() clears an expired busy window and returns the credit."""
    from pipeline_validator.memory.transfer import StageRequest, StageWaitReason, TransferStage
    stage = TransferStage(
      "hbm_read", StageWaitReason.HBM_OUTSTANDING,
      0, 100.0, 1, 1, max_outstanding=1)
    stage.try_issue("t0", [StageRequest("x", 100)], cycle=0)
    assert stage._outstanding == 1
    # window = 1 cycle; step past it without an explicit release
    stage.step(cycle=5)
    assert stage._outstanding == 0
    assert stage._holders[0] is None

  def test_cancel_idempotent(self):
    from pipeline_validator.memory.transfer import StageRequest, StageWaitReason, TransferStage
    stage = TransferStage(
      "hbm_read", StageWaitReason.HBM_OUTSTANDING, 0, 100.0, 1, 1,
      max_outstanding=1)
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
      transaction_id=txn_id, op=op, issuer=owner,
      src=None, dst=None, bytes_total=4096,
      completion_event=txn_id, tile_id=tile_id)

  @staticmethod
  def _view():
    """A minimal resolved HBM/L2 view (single segment on bank 0)."""
    from pipeline_validator.memory.allocator import AllocationHandle, BankSegment
    from pipeline_validator.memory.transfer import ResolvedMemoryView
    seg = (BankSegment(0, 0, 4096),)
    handle = AllocationHandle(
      allocation_id="l2:0:1", memory_space="l2", owner=_ctx_owner(),
      base_address=0, size_bytes=4096, alignment=1,
      bank_segments=seg, generation=0, allocate_cycle=0)
    return ResolvedMemoryView(
      handle=handle, offset_bytes=0, size_bytes=4096, address=0,
      segments=seg)

  def test_cancel_owner_returns_hbm_outstanding(self):
    """cancel_owner returns the HBM outstanding credit and lets the
    channel be reused immediately."""
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import TransferOp, TransferStatus
    cfg = HardwareConfig().with_overrides(
      hbm_fixed_latency_cycles=1000, hbm_outstanding_limit=1)
    tm = self._manager(cfg)
    o1 = _task_owner(tile=0)
    o2 = _task_owner(tile=1)
    t1 = self._txn("t1", o1, TransferOp.PREFETCH)
    t2 = self._txn("t2", o2, TransferOp.PREFETCH)
    # real views so the full_memory route (HBM_READ first leg) builds
    t1.src = self._view()
    t1.dst = self._view()
    t2.src = self._view()
    t2.dst = self._view()
    tm.submit(t1, cycle=0)
    tm.submit(t2, cycle=0)
    tm.step(cycle=0)
    # limit=1: t1 holds the credit; t2 waits (outstanding full)
    assert tm._hbm_read._outstanding == 1
    tm.cancel_owner(o1, cycle=0)
    assert tm.status("t1") == TransferStatus.CANCELLED
    # credit returned immediately on cancel
    assert tm._hbm_read._outstanding == 0
    # t2 issues on the next step and takes the credit
    tm.step(cycle=1)
    assert tm._hbm_read._outstanding == 1
    # after cancel_all nothing is held
    tm.cancel_all(cycle=1)
    assert tm._hbm_read._outstanding == 0
    assert tm._hbm_write._outstanding == 0
    assert tm.inflight_count == 0

  def test_cancel_all_returns_bank_and_channel_resources(self):
    """cancel_all returns every bank/channel/credit reservation."""
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import TransferOp
    cfg = HardwareConfig().with_overrides(
      hbm_fixed_latency_cycles=1000, hbm_outstanding_limit=32)
    tm = self._manager(cfg)
    o = _task_owner(tile=0)
    for i in range(3):
      tm.submit(self._txn(f"t{i}", o, TransferOp.PREFETCH), cycle=0)
    tm.step(cycle=0)
    tm.step(cycle=1)
    assert tm.inflight_count == 3
    tm.cancel_all(cycle=1)
    assert tm.inflight_count == 0
    for stage in tm._all_stages():
      assert stage._outstanding == 0, stage.name
      assert all(h is None for h in stage._holders), stage.name
      assert all(b == 0 for b in stage._busy_until), stage.name

  def test_owner_after_cancel_does_not_block_others(self):
    """A cancelled owner's bank reservation must not serialize a later
    transaction on the same bank."""
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.allocator import AllocationHandle, BankSegment
    from pipeline_validator.memory.transfer import ResolvedMemoryView, TransferOp
    cfg = HardwareConfig().with_overrides(
      hbm_fixed_latency_cycles=10)
    tm = self._manager(cfg)
    o1 = _task_owner(tile=0)
    o2 = _task_owner(tile=1)
    seg = (BankSegment(0, 0, 4096),)
    h = AllocationHandle(
      allocation_id="l2:0:1", memory_space="l2", owner=o1,
      base_address=0, size_bytes=4096, alignment=1,
      bank_segments=seg, generation=0, allocate_cycle=0)
    view = ResolvedMemoryView(
      handle=h, offset_bytes=0, size_bytes=4096, address=0,
      segments=seg)
    t1 = self._txn("t1", o1, TransferOp.GLOBAL_STORE)
    t1.src = view
    t1.dst = view
    tm.submit(t1, cycle=0)
    tm.step(cycle=0)
    # t1 holds l2_read bank 0
    assert tm._l2_read._holders[0] == "t1"
    tm.cancel_owner(o1, cycle=0)
    # bank returned: a new transaction issues on bank 0 immediately
    t2 = self._txn("t2", o2, TransferOp.GLOBAL_STORE)
    t2.src = view
    t2.dst = view
    tm.submit(t2, cycle=0)
    tm.step(cycle=0)
    assert tm._l2_read._holders[0] == "t2"


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
    from pipeline_validator.memory.transfer import (
      MemoryTransaction,
      TransferLeg,
      TransferOp,
    )
    txn = MemoryTransaction(
      transaction_id=txn_id, op=TransferOp.PREFETCH, issuer=owner,
      src=None, dst=None, bytes_total=64, completion_event=txn_id)
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
    noc = NoCRouter(vc_depth=cfg.noc_vc_depth,
                    router_latency_cycles=cfg.noc_router_latency_cycles)
    tm = self._manager(cfg, noc)
    txn = self._txn_on_leg(tm, "t1", _task_owner(tile=0),
                           TransferLegKind.NOC_RESPONSE, "vc1")
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
    self._txn_on_leg(tm, "t1", _task_owner(tile=0),
                     TransferLegKind.NOC_REQUEST, "vc2")
    self._txn_on_leg(tm, "t2", _task_owner(tile=1),
                     TransferLegKind.NOC_REQUEST, "vc2")
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
    noc = NoCRouter(vc_depth=cfg.noc_vc_depth,
                    router_latency_cycles=cfg.noc_router_latency_cycles)
    tm = self._manager(cfg, noc)
    self._txn_on_leg(tm, "t1", _task_owner(tile=0),
                     TransferLegKind.NOC_RESPONSE, "vc1")
    self._step(tm, noc, 0)
    assert noc.contains("t1:noc_response")
    tm.cancel_all(cycle=0)
    assert not noc.contains("t1:noc_response")
    # credit never consumed (flit was pending), stays full
    vc1 = noc.vcs[VCId.VC1_DMA_READ_RSP.value]
    assert vc1.credit_available == cfg.noc_vc_depth

  def test_cancel_traversed_flit_returns_credit(self):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.noc import NoCRouter, VCId
    from pipeline_validator.memory.transfer import TransferLegKind
    cfg = HardwareConfig()
    noc = NoCRouter(vc_depth=cfg.noc_vc_depth,
                    router_latency_cycles=cfg.noc_router_latency_cycles)
    tm = self._manager(cfg, noc)
    self._txn_on_leg(tm, "t1", _task_owner(tile=0),
                     TransferLegKind.NOC_RESPONSE, "vc1")
    self._step(tm, noc, 0)
    self._step(tm, noc, 1)  # flit traverses, credit consumed
    vc1 = noc.vcs[VCId.VC1_DMA_READ_RSP.value]
    assert vc1.credit_available == cfg.noc_vc_depth - 1
    tm.cancel_all(cycle=1)
    assert vc1.credit_available == cfg.noc_vc_depth

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
      allocation_id=f"global:Y:{address}", memory_space="hbm",
      owner=owner, base_address=address, size_bytes=64, alignment=64,
      bank_segments=segments, generation=0, allocate_cycle=0)
    return ResolvedMemoryView(
      handle=handle, offset_bytes=0, size_bytes=64, address=address,
      segments=segments)

  @staticmethod
  def _txn(tm, txn_id: str, address: int, kind, op, owner=None):
    from pipeline_validator.memory.transfer import MemoryTransaction, TransferLeg
    view = TestHBMChannelMapping._view(address)
    txn = MemoryTransaction(
      transaction_id=txn_id, op=op,
      issuer=owner if owner is not None else _task_owner(),
      src=view, dst=view, bytes_total=64, completion_event=txn_id)
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
    cfg = HardwareConfig().with_overrides(
      hbm_channels=2, hbm_burst_bytes=64,
      hbm_fixed_latency_cycles=100)
    tm = TransferManager(cfg, full_memory=True)
    t0 = self._txn(
      tm, "t0", 0, TransferLegKind.HBM_READ, TransferOp.PREFETCH)
    t1 = self._txn(
      tm, "t1", 128, TransferLegKind.HBM_READ, TransferOp.PREFETCH)
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
    cfg = HardwareConfig().with_overrides(
      hbm_channels=2, hbm_burst_bytes=64,
      hbm_fixed_latency_cycles=100)
    tm = TransferManager(cfg, full_memory=True)
    t0 = self._txn(
      tm, "t0", 0, TransferLegKind.HBM_WRITE,
      TransferOp.GLOBAL_STORE)
    t1 = self._txn(
      tm, "t1", 64, TransferLegKind.HBM_WRITE,
      TransferOp.GLOBAL_STORE)
    tm.step(cycle=0)
    assert tm._hbm_write._holders == ["t0", "t1"]
    assert t0.leg_start_cycle == 0
    assert t1.leg_start_cycle == 0
    assert t0.leg_completion_cycle == t1.leg_completion_cycle

  def test_read_and_write_share_one_global_outstanding_limit(self):
    """With limit=1, an HBM read blocks an HBM write even when their
    address-derived channels differ."""
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import (
      StageWaitReason,
      TransferLegKind,
      TransferManager,
      TransferOp,
    )
    cfg = HardwareConfig().with_overrides(
      hbm_channels=2, hbm_burst_bytes=64,
      hbm_fixed_latency_cycles=100, hbm_outstanding_limit=1)
    tm = TransferManager(cfg, full_memory=True)
    read = self._txn(
      tm, "read", 0, TransferLegKind.HBM_READ, TransferOp.PREFETCH,
      owner=_task_owner(tile=0))
    write = self._txn(
      tm, "write", 64, TransferLegKind.HBM_WRITE,
      TransferOp.GLOBAL_STORE, owner=_task_owner(tile=1))
    tm.step(cycle=0)
    assert tm._hbm_outstanding_txns == {"read"}
    assert tm._hbm_read._holders == ["read", None]
    assert tm._hbm_write._holders == [None, None]
    assert read.leg_start_cycle == 0
    assert write.leg_start_cycle == -1
    assert write.wait_reason == StageWaitReason.HBM_OUTSTANDING
    tm.cancel_owner(read.issuer, cycle=0)
    assert tm._hbm_outstanding_txns == set()
    tm.step(cycle=1)
    assert tm._hbm_outstanding_txns == {"write"}
    assert tm._hbm_write._holders == [None, "write"]
