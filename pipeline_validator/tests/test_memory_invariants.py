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
    plan = alloc.plan_bundle([
      _req(_ctx_owner(buf="b1"), 512),
      _req(_ctx_owner(buf="b2"), 513),
    ])
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
      plan = alloc.plan_bundle([
        _req(_ctx_owner(gen=1, buf=name), 16, align=32, buf_id=name)])
      assert not isinstance(plan, AdmissionFailure)
      alloc.commit(plan, 0)
    assert alloc.snapshot()["free_bytes"] == 32
    frag_req = [AllocationRequest(
      "l2", "frag", _ctx_owner(gen=2, buf="frag"), 32, 32)]
    before = self._state(alloc)
    frag = alloc.plan_bundle(frag_req)
    assert isinstance(frag, AdmissionFailure)
    assert frag.kind is AdmissionFailureKind.TEMPORARY_CAPACITY
    assert self._state(alloc) == before
    assert alloc.can_ever_fit_bundle(frag_req)
    # release blocker a: bank 0 merges back into one 32-byte extent
    handle_a = next(h for h in alloc._live.values()
                    if h.handle.owner.buffer_id == "a").handle
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

  def test_anonymous_token_never_fabricates_residency(self):
    cache = DeterministicLRUCache(capacity_bytes=128, line_bytes=64)
    cache.record_hit(None)
    cache.record_miss()
    cache.refill(None)
    snapshot = cache.snapshot()
    assert snapshot["hits"] == 1
    assert snapshot["misses"] == 1
    assert snapshot["refills"] == 1
    assert snapshot["resident_lines"] == 0
    assert snapshot["resident_tokens"] == ()

  def test_reset_clears_metadata_and_statistics(self):
    cache = DeterministicLRUCache(capacity_bytes=64, line_bytes=64)
    cache.record_miss()
    cache.refill("A")
    cache.reset()
    assert cache.snapshot() == {
      "hits": 0,
      "misses": 0,
      "refills": 0,
      "evictions": 0,
      "resident_lines": 0,
      "resident_bytes": 0,
      "capacity_bytes": 64,
      "resident_tokens": (),
    }

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

  def test_reset_clears_entries_callbacks_and_statistics(self):
    table = MshrTable(capacity=1)
    allocation = table.allocate("A")
    assert isinstance(allocation, MshrAllocation)
    table.wait(allocation.token, lambda: None)
    assert isinstance(table.allocate("B"), MshrWait)
    table.reset()
    assert table.snapshot() == {
      "active": 0,
      "merged": 0,
      "stalls": 0,
      "callbacks": 0,
      "capacity": 1,
      "version": 0,
    }


class TestGatherTransferRoutes:
  @staticmethod
  def _transaction(transaction_id, op, *, src=None, dst=None, owner=None):
    from pipeline_validator.memory.transfer import MemoryTransaction

    return MemoryTransaction(
      transaction_id=transaction_id,
      op=op,
      issuer=_task_owner() if owner is None else owner,
      src=src,
      dst=dst,
      bytes_total=64,
      completion_event=transaction_id,
      tile_id=0,
    )

  @staticmethod
  def _view(space: str, owner=None):
    from pipeline_validator.memory.allocator import AllocationHandle, BankSegment
    from pipeline_validator.memory.transfer import ResolvedMemoryView

    actual_owner = _task_owner() if owner is None else owner
    segments = (
      BankSegment(0, 0, 64),
      BankSegment(1, 64, 64),
    )
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
      handle=handle,
      offset_bytes=0,
      size_bytes=128,
      address=0,
      segments=segments,
      permissions="r",
    )

  @pytest.mark.parametrize("full_memory", [False, True])
  def test_six_gather_routes_have_exact_leg_sequences(self, full_memory):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import (
      TransferLegKind,
      TransferManager,
      TransferOp,
    )

    expected = {
      TransferOp.GATHER_L1_HIT: (TransferLegKind.L1_CACHE_LOOKUP,),
      TransferOp.GATHER_L2_HIT: (
        TransferLegKind.L1_CACHE_LOOKUP,
        TransferLegKind.L2_CACHE_LOOKUP,
        TransferLegKind.NOC_RESPONSE,
        TransferLegKind.LOCAL_DMA,
        TransferLegKind.L1_CACHE_FILL,
      ),
      TransferOp.GATHER_MISS_LOOKUP: (
        TransferLegKind.L1_CACHE_LOOKUP,
        TransferLegKind.L2_CACHE_LOOKUP,
      ),
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
    }
    manager = TransferManager(HardwareConfig(), full_memory=full_memory)
    for index, (op, expected_legs) in enumerate(expected.items()):
      transaction = self._transaction(f"route:{index}", op)
      manager.submit(transaction, cycle=0)
      assert tuple(leg.kind for leg in transaction.legs) == expected_legs
      assert TransferLegKind.GLOBAL_DMA not in expected_legs
      assert TransferLegKind.L2_WRITE not in expected_legs

  def test_gather_route_cannot_collapse(self):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.transfer import TransferManager, TransferOp

    manager = TransferManager(HardwareConfig(), full_memory=False)
    transaction = self._transaction("no-collapse", TransferOp.GATHER_L1_HIT)
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
      [
        _req(owner_a, 32, buf_id="a", space="l1"),
        _req(owner_b, 32, buf_id="b", space="l1"),
      ]
    )
    assert not isinstance(initial, AdmissionFailure)
    handle_a, _handle_b = allocator.commit(initial, cycle=0)
    allocator.request_release(handle_a, owner_a, cycle=1)

    destination_owner = _task_owner(buf="destination")
    fragmented = allocator.plan_bundle(
      [_req(destination_owner, 96, buf_id="destination", space="l1")]
    )
    assert not isinstance(fragmented, AdmissionFailure)
    handle = allocator.commit(fragmented, cycle=2)[0]
    assert [
      (segment.bank_id, segment.address, segment.size_bytes)
      for segment in handle.bank_segments
    ] == [(0, 0, 32), (0, 64, 64)]
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
    assert [
      (segment.bank_id, segment.address, segment.size_bytes)
      for segment in sliced.segments
    ] == [(0, 16, 16), (0, 64, 48)]

  def test_cancel_returns_gather_cache_hbm_noc_and_l1_resources(self):
    from pipeline_validator.config import HardwareConfig
    from pipeline_validator.memory.noc import NoCRouter, VCId
    from pipeline_validator.memory.transfer import TransferManager, TransferOp

    config = HardwareConfig().with_overrides(
      hbm_fixed_latency_cycles=1000,
      hbm_outstanding_limit=1,
    )
    noc = NoCRouter(
      vc_depth=config.noc_vc_depth,
      router_latency_cycles=config.noc_router_latency_cycles,
    )
    manager = TransferManager(config, full_memory=True, noc=noc)
    owner = _task_owner()

    lookup = self._transaction("lookup", TransferOp.GATHER_L1_HIT, owner=owner)
    manager.submit(lookup, cycle=0)
    manager.step(cycle=0)
    assert manager._l1_cache_lookup[0]._holders[0] == "lookup"
    manager.cancel_owner(owner, cycle=0)
    assert manager._l1_cache_lookup[0]._holders[0] is None

    hbm = self._transaction(
      "hbm",
      TransferOp.GATHER_HBM_REFILL,
      src=self._view("hbm", owner),
      owner=owner,
    )
    manager.submit(hbm, cycle=1)
    manager.step(cycle=1)
    assert manager._hbm_read._outstanding == 1
    manager.cancel_owner(owner, cycle=1)
    assert manager._hbm_read._outstanding == 0

    refill = self._transaction("refill", TransferOp.GATHER_L2_REFILL, owner=owner)
    manager.submit(refill, cycle=2)
    manager.step(cycle=2)
    traversed = noc.step(cycle=3)
    manager.note_traversed(traversed, cycle=3)
    manager.step(cycle=3)
    vc1 = noc.vcs[VCId.VC1_DMA_READ_RSP.value]
    assert vc1.credit_available == config.noc_vc_depth - 1
    manager.cancel_owner(owner, cycle=3)
    assert vc1.credit_available == config.noc_vc_depth

    destination = self._transaction(
      "destination",
      TransferOp.GATHER_DEST_WRITE,
      dst=self._view("l1", owner),
      owner=owner,
    )
    manager.submit(destination, cycle=4)
    manager.step(cycle=4)
    assert manager._l1_write[0]._holders[0] == "destination"
    manager.cancel_owner(owner, cycle=4)

    snapshot = manager.snapshot()
    assert manager.inflight_count == 0
    assert manager._hbm_read._outstanding == 0
    assert vc1.credit_available == config.noc_vc_depth
    assert all(
      stage["busy_resources"] == 0 and stage["outstanding"] == 0
      for stage in snapshot["stages"].values()
    )
