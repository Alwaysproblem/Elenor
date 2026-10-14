"""Plan §Verification 5: host page pool ownership, epochs and failure atomicity."""

import pytest

from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.memory.hbm_region import HBMRegion
from pipeline_validator.memory.page_pool import (
  HostPagePoolSpec,
  PagePoolRegistry,
  PoolPageError,
)

PAGE_BYTES = 4096
POOL_BYTES = PAGE_BYTES * 16


def _make_oracle():
  store = ByteStore()
  store.seed_hbm(0x100000, bytes(POOL_BYTES))
  hbm = HBMRegion(base_iova=0, size_bytes=1 << 24)
  handle = hbm.bind_external(GlobalBinding("POOL", 0x100000, POOL_BYTES, "rw"), 0)
  store.register_hbm_binding(handle, "rw")
  return store, hbm


def _make_registry(store, hbm, *, cached_hook=None, owners=None):
  spec = HostPagePoolSpec(
    name="kv",
    binding="POOL",
    page_bytes=PAGE_BYTES,
    page_count=16,
    initial_owners=owners if owners is not None else {"owner_0": (0, 1), "owner_1": (2,)},
  )
  registry = PagePoolRegistry()
  registry.initialize(
    [spec], {"POOL": object()}, hbm, store, 0, is_binding_cached=cached_hook
  )
  return registry


def test_lowest_free_page_is_allocated_and_reused_with_epoch_growth(tmp_path):
  store, hbm = _make_oracle()
  registry = _make_registry(store, hbm)
  allocation = registry.allocate("kv", "owner_1", 2, cycle=1)
  assert allocation.pages == (3, 4)  # owner_1 already owns page 2
  registry.free("kv", "owner_0", cycle=2)
  regrown = registry.allocate("kv", "owner_0", 1, cycle=3)
  assert regrown.pages == (0,)  # lowest free page
  assert regrown.epochs[0] == 2  # granted at 0, freed -> 1, regranted -> 2


def test_lease_blocks_free_and_alloc_then_releases(tmp_path):
  store, hbm = _make_oracle()
  registry = _make_registry(store, hbm)
  lease = registry.lease("owner_0", "POOL", 0, PAGE_BYTES)
  with pytest.raises(PoolPageError, match="host_pool_busy"):
    registry.free("kv", "owner_0", cycle=1)
  with pytest.raises(PoolPageError, match="host_pool_busy"):
    registry.allocate("kv", "owner_0", 1, cycle=2)
  lease.release("txn:1")
  registry.free("kv", "owner_0", cycle=3)


def test_old_transaction_cannot_write_new_owner(tmp_path):
  store, hbm = _make_oracle()
  registry = _make_registry(store, hbm)
  lease = registry.lease("owner_0", "POOL", PAGE_BYTES, PAGE_BYTES)  # page 1
  lease.acquire("txn:old")
  lease.release("txn:old")  # terminal acknowledgement frees the lease count
  registry.free("kv", "owner_0", cycle=1)
  registry.allocate("kv", "owner_1", 1, cycle=2)  # page 0, the lowest free page
  assert registry._page_owner(0) == "owner_1"
  assert not lease.is_valid()  # the old transaction cannot write the new owner's pages
  with pytest.raises(PoolPageError, match="host_scope_inactive"):
    lease.acquire("txn:newer")


def test_double_free_is_rejected(tmp_path):
  store, hbm = _make_oracle()
  registry = _make_registry(store, hbm)
  registry.free("kv", "owner_1", cycle=1)
  with pytest.raises(PoolPageError, match="host_scope_inactive"):
    registry.free("kv", "owner_1", cycle=2)


def test_out_of_pages_fails_atomically(tmp_path):
  store, hbm = _make_oracle()
  registry = _make_registry(store, hbm)
  before = registry.snapshot()
  with pytest.raises(PoolPageError, match="host_pool_out_of_pages"):
    registry.allocate("kv", "owner_1", 14, cycle=1)  # only 13 pages free
  after = registry.snapshot()
  assert before["pools"]["kv"]["allocated_pages"] == after["pools"]["kv"]["allocated_pages"]
  assert before["pools"]["kv"]["live_pages"] == after["pools"]["kv"]["live_pages"]


def test_cross_owner_access_is_memory_scope_violation(tmp_path):
  store, hbm = _make_oracle()
  registry = _make_registry(store, hbm)
  with pytest.raises(PoolPageError, match="memory_scope_violation"):
    registry.lease("owner_0", "POOL", 2 * PAGE_BYTES, PAGE_BYTES)  # page 2 is owner_1's
  with pytest.raises(PoolPageError, match="memory_scope_violation"):
    registry.lease("owner_1", "POOL", PAGE_BYTES - 1, 2)  # spans out of page 1


def test_pool_with_residual_cache_state_is_rejected(tmp_path):
  from pipeline_validator.memory.cache import DeterministicLRUCache

  store, hbm = _make_oracle()
  cache = DeterministicLRUCache(capacity_bytes=4 * 64, line_bytes=64)
  store.register_cache("l2", 0, cache)
  store.seed_cache_line("l2", 0, "POOL", 0, bytes(64))
  with pytest.raises(PoolPageError, match="managed_pool_has_cached_state"):
    _make_registry(store, hbm, cached_hook=store.binding_has_cached_state)


def test_reallocated_page_clears_hbm_and_cache_validity(tmp_path):
  store, hbm = _make_oracle()
  registry = _make_registry(store, hbm)
  registry.free("kv", "owner_0", cycle=1)
  registry.allocate("kv", "owner_1", 2, cycle=2)  # pages 0..1 reallocated
  view_handle = hbm.get_handle("POOL")
  segments = hbm.resolve(view_handle, 0, PAGE_BYTES)
  from pipeline_validator.memory.transfer import ResolvedMemoryView

  view = ResolvedMemoryView(
    handle=view_handle,
    offset_bytes=0,
    size_bytes=PAGE_BYTES,
    address=segments[0].address,
    segments=segments,
    permissions="rw",
  )
  data, mask = store.read_view_relaxed(view)
  assert data == bytes(PAGE_BYTES)  # oracle data is preserved
  assert set(mask) == {0}  # but validity was cleared for the new owner


def test_root_pin_blocks_free_until_unpinned(tmp_path):
  store, hbm = _make_oracle()
  registry = _make_registry(store, hbm)
  registry.pin_root(101, {"POOL": "owner_0"}, cycle=1)
  with pytest.raises(PoolPageError, match="host_pool_busy"):
    registry.free("kv", "owner_0", cycle=2)
  registry.unpin_root(101, cycle=3)
  registry.free("kv", "owner_0", cycle=4)


def test_scope_binding_mismatch_is_rejected(tmp_path):
  store, hbm = _make_oracle()
  hbm.bind_external(GlobalBinding("OTHER", 0x400000, PAGE_BYTES, "rw"), 1)
  registry = _make_registry(store, hbm)
  with pytest.raises(PoolPageError, match="scope_binding_mismatch"):
    registry.pin_root(7, {"OTHER": "owner_0"}, cycle=1)


def test_seed_hbm_rejects_runtime_cycles(tmp_path):
  store, _hbm = _make_oracle()
  with pytest.raises(Exception, match="cycle-0"):
    store.seed_hbm(0x100000, b"\x01", cycle=5)
