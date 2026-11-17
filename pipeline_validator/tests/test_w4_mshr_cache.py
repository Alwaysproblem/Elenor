"""C030 oracle observations for the L1 MSHR token lifetime and the aggregate cache.

RTL (`hardware/coral_nexus/rtl/memory/{mshr_table,cache,transfer_outbridge}.sv`) is the only
cosim L1 execution owner. These tests pin the Python REFERENCE behavior the RTL must refine,
and the sourced geometry the RTL consumes. They do not model or replace the RTL, and they add
no second MSHR/cache implementation.

Two things are deliberately observed here instead of assumed:

* `MshrTable` frees a token as soon as `complete()` returns. That is the weaker reference
  baseline; the RTL contract (W4 plan section 4.4, protocol `mshr:`) is stronger: a token is
  only reusable after every waiter has its data AND the terminal reference has closed. The
  tests show what the model does not enforce, so nobody mistakes the model for the contract.
* `DeterministicLRUCache` is fully associative with a global-oldest victim, and the profile
  capacity is `banks * cache_bytes_per_bank` (aggregate across banks, not one bank).
"""

from __future__ import annotations

import pytest

from pipeline_validator.config import HardwareConfig
from pipeline_validator.memory import (
  DeterministicLRUCache,
  MemoryInvariantError,
  MshrAllocation,
  MshrTable,
  MshrWait,
)
from pipeline_validator.memory.cache import CacheRange
from pipeline_validator.profiles import build_registry

LINE = 64


def _line(cache: DeterministicLRUCache, offset: int, fill: int, *, dirty: bool = False):
  identity = cache.identity_for("A", 1, offset)
  from pipeline_validator.memory.cache import CacheProvenance

  cache.refill(
    None,
    identity=identity,
    provenance=CacheProvenance("A", "A", 1, offset, LINE),
    data=bytes([fill]) * LINE,
    validity=bytes([1]) * LINE,
    dirty=dirty,
  )
  return identity


def test_profile_cache_capacity_is_aggregate_across_banks() -> None:
  hw = HardwareConfig()
  registry = build_registry(hw)
  for mode in range(4):
    profile = registry.profile("l1", mode)
    cache = DeterministicLRUCache(profile.cache_bytes, hw.cache_line_bytes, level="l1")
    assert profile.cache_bytes == hw.tile_l1_banks * profile.cache_bytes_per_bank
    capacity_lines = cache.stats.capacity_bytes // hw.cache_line_bytes
    assert capacity_lines == profile.cache_bytes // hw.cache_line_bytes
    if profile.cache_bytes_per_bank:
      # Aggregate across banks: one bank's worth would be 1/banks of this.
      assert capacity_lines == hw.tile_l1_banks * (profile.cache_bytes_per_bank // hw.cache_line_bytes)
      assert capacity_lines > profile.cache_bytes_per_bank // hw.cache_line_bytes
    else:
      assert not cache.enabled, "profile 0 is a disabled cache, not a one-line fallback"


def test_victim_is_the_global_oldest_line_not_a_set_local_one() -> None:
  cache = DeterministicLRUCache(4 * LINE, LINE)
  ids = [_line(cache, i * LINE, i) for i in range(4)]
  cache.read_line(ids[0])  # touch: line 0 becomes the most recent
  _line(cache, 4 * LINE, 9)  # overflow: the global oldest (line 1) must go
  assert cache.contains(ids[0])
  assert not cache.contains(ids[1])
  assert all(cache.contains(ids[i]) for i in (2, 3))
  assert cache.stats.evictions == 1


def test_dirty_line_is_never_silently_evicted() -> None:
  cache = DeterministicLRUCache(2 * LINE, LINE, write_policy="write_back")
  first = _line(cache, 0, 1, dirty=True)
  _line(cache, LINE, 2)
  with pytest.raises(MemoryInvariantError, match="real clean before eviction"):
    _line(cache, 2 * LINE, 3)
  assert cache.contains(first), "refused eviction must not lose the dirty line"
  assert cache.stats.dirty_lines == 1


def test_old_generation_refill_is_rejected_and_nothing_is_written() -> None:
  cache = DeterministicLRUCache(2 * LINE, LINE)
  cache.reconfigure(2 * LINE, 1)
  identity = cache.identity_for("A", 1, 0)
  from pipeline_validator.memory.cache import CacheProvenance

  with pytest.raises(MemoryInvariantError, match="old-generation"):
    cache.refill(
      None,
      identity=identity,
      provenance=CacheProvenance("A", "A", 1, 0, LINE),
      data=bytes(LINE),
      profile_generation=0,
    )
  assert not cache.contains(identity)


def test_reconfigure_needs_a_strictly_newer_generation_and_no_dirty_or_pending_work() -> None:
  cache = DeterministicLRUCache(2 * LINE, LINE, write_policy="write_back")
  with pytest.raises(MemoryInvariantError, match="monotonically"):
    cache.reconfigure(2 * LINE, 0)
  _line(cache, 0, 1, dirty=True)
  with pytest.raises(MemoryInvariantError, match="dirty lines"):
    cache.reconfigure(2 * LINE, 1)
  requests = cache.begin_maintenance(None, clean=True, invalidate=False)
  assert len(requests) == 1
  with pytest.raises(MemoryInvariantError, match="pending clean"):
    cache.reconfigure(2 * LINE, 1)
  cache.complete_clean(requests[0].request_id)
  cache.reconfigure(2 * LINE, 1)
  assert cache.profile_generation == 1
  assert cache.stats.resident_lines == 0, "reconfigure drops every line"


def test_failed_clean_keeps_the_line_dirty_and_range_maintenance_is_exact() -> None:
  cache = DeterministicLRUCache(8 * LINE, LINE, write_policy="write_back")
  for i in range(4):
    _line(cache, i * LINE, i, dirty=True)
  in_range = cache.begin_maintenance((CacheRange("A", 1, 0, 2 * LINE),), clean=True, invalidate=True)
  assert len(in_range) == 2, "one clean request per dirty in-range line"
  assert {r.provenance.source_offset for r in in_range} == {0, LINE}
  cache.complete_clean(in_range[0].request_id, success=False)
  assert cache.stats.dirty_lines == 4, "a failed clean keeps the line dirty"
  cache.complete_clean(in_range[1].request_id, success=True)
  assert cache.stats.dirty_lines == 3
  with pytest.raises(MemoryInvariantError, match="unknown cache clean request"):
    cache.complete_clean("cache-clean:bogus")
  with pytest.raises(MemoryInvariantError, match="without clean"):
    cache.begin_maintenance((CacheRange("A", 1, 2 * LINE, LINE),), clean=False, invalidate=True)


def test_eviction_cannot_destroy_waiter_data_but_the_model_does_not_hold_the_token() -> None:
  """W4 section 4.4: the entry-owned fill buffer survives cache eviction.

  The Python model carries no fill buffer. This test pins the part the model CAN show: the
  cache may lose the line while an MSHR waiter is still registered, so the waiter must be
  served from MSHR-owned data (the RTL contract), never by a cache re-lookup.
  """
  table = MshrTable(capacity=1)
  cache = DeterministicLRUCache(LINE, LINE)
  leader = table.allocate("A@1:0")
  assert isinstance(leader, MshrAllocation) and leader.leader
  waiter = table.allocate("A@1:0")
  assert isinstance(waiter, MshrAllocation) and not waiter.leader
  assert waiter.token == leader.token
  served: list[str] = []
  table.wait(leader.token, lambda: served.append("w0"))
  table.wait(leader.token, lambda: served.append("w1"))
  identity = _line(cache, 0, 7)
  _line(cache, LINE, 8)  # capacity one: evicts the refilled line
  assert not cache.contains(identity), "the cache no longer holds the refilled line"
  assert table.is_active(leader.token), "the MSHR entry outlives the cache line"
  assert table.snapshot()["callbacks"] == 2


def test_model_frees_the_token_at_complete_so_it_is_not_the_reuse_contract() -> None:
  """The reference baseline frees at complete(); the RTL must be stronger (section 4.4).

  Pinned so the gap stays visible: after complete() the model happily reallocates the same
  capacity slot to another group while callbacks are still unread by the caller.
  """
  table = MshrTable(capacity=1)
  first = table.allocate("A")
  assert isinstance(first, MshrAllocation)
  table.wait(first.token, lambda: None)
  ready = table.complete(first.token)
  assert len(ready) == 1, "callbacks are only RETURNED, not yet consumed by the waiter"
  second = table.allocate("B")
  assert isinstance(second, MshrAllocation), (
    "the model reuses capacity before the waiter consumed anything; "
    "RTL tb_mshr_table/tb_l1_miss_path assert the stronger no-reuse-until-terminal rule"
  )
  assert second.token != first.token, "tokens are monotone, which is NOT a reuse bound"


def test_merge_never_crosses_a_profile_generation() -> None:
  table = MshrTable(capacity=4)
  old = table.allocate("A")
  assert isinstance(old, MshrAllocation)
  table.complete(old.token)
  table.reconfigure(1)
  new = table.allocate("A")
  assert isinstance(new, MshrAllocation) and new.leader and new.generation == 1
  with pytest.raises(MemoryInvariantError, match="old profile generation"):
    table.allocate("A", generation=0)
  table.wait(new.token, lambda: None)
  with pytest.raises(MemoryInvariantError, match="active refills"):
    table.reconfigure(2)


def test_cancel_isolates_the_entry_without_running_waiter_callbacks() -> None:
  table = MshrTable(capacity=1)
  entry = table.allocate("A")
  assert isinstance(entry, MshrAllocation)
  ran: list[str] = []
  table.wait(entry.token, lambda: ran.append("data"))
  assert table.cancel(entry.token) == 1
  assert ran == [], "cancel must never deliver success data"
  assert not table.is_active(entry.token)
  with pytest.raises(MemoryInvariantError):
    table.complete(entry.token)
  again = table.allocate("A")
  assert isinstance(again, MshrAllocation) and again.leader


def test_capacity_stall_is_structured_and_does_not_drop_the_request() -> None:
  table = MshrTable(capacity=2)
  held = [table.allocate(f"G{i}") for i in range(2)]
  assert all(isinstance(h, MshrAllocation) for h in held)
  wait = table.allocate("G2")
  assert isinstance(wait, MshrWait) and wait.reason == "mshr_full"
  assert table.snapshot()["stalls"] == 1
  assert table.snapshot()["active"] == 2, "a stalled allocation holds no capacity"
  assert isinstance(held[0], MshrAllocation)
  table.complete(held[0].token)
  retry = table.allocate("G2")
  assert isinstance(retry, MshrAllocation) and retry.leader
