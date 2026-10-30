"""P05 SW-A2: grant-time engine affinity and live-handle accounting.

Allocator-level regressions for the engine pool rules that P05 introduced on
top of the P04 per-Tile manager.  Pure ``SharedAmManager`` unit tests: no
Simulator, no IR, and no mutation of ``bound_engine`` or any other internal
state to fabricate situations -- every scenario is a legal request sequence.

Covered acceptance:
- Grant picks the engine with the fewest currently live handles, ties to the
  lowest engine id; the slot, quota reservation and engine binding are one
  atomic ``am_allocate`` result carrying the live counts at grant.
- The engine handle increments with the grant and decrements exactly once at
  the safe free, strictly before parked requests are woken (the wake grant
  weighs the freed engine's updated count).
- A FINAL commit does not decrement: the handle stays live through the FINAL
  output and any output backpressure until ``release_final``/abort.
- ``bound_engine`` is fixed for the lease lifetime; accounting shifts never
  migrate a live lease (a busy engine keeps its leases).
- Reset/rebind/abort clear the accounting; the snapshot exposes every engine
  id (zero counts included) and the counts always sum to the live leases.
- Legal load distributions on 1/2/4 engines, including more than two live
  handles on one engine: the familiar "at most 2 per engine from an empty
  start" is a least-live distribution outcome, never a per-engine cap
  (execution 总则 §3.1).  Engine ids are an independent domain; physical slot
  groups never influence affinity.

Run:
  conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_boa_affinity.py -q
"""

from __future__ import annotations

import json

import pytest

from pipeline_validator.config import MAX_BOA_ENGINES, HardwareConfig
from pipeline_validator.runtime.shared_am import AmLease, AmOwner, SharedAmManager
from pipeline_validator.trace import Tracer

_CAPACITY = 65536
_DIMS = (64, 256)
_PAYLOAD = b"\x00" * (64 * 256 * 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _never_woken(lease: AmLease, cycle: int) -> None:
  raise AssertionError(f"unexpected AM wake for lease {lease.lease_id} at {cycle}")


def _manager(engine_count: int, *, group_count: int = 4, tracer: Tracer | None = None):
  return SharedAmManager(
    slot_capacity_bytes=_CAPACITY,
    group_count=group_count,
    slots_per_group=2,
    slots_per_context=2,
    engine_count=engine_count,
    tracer=tracer,
  )


def _owner(ctx_id: int, *, epoch: int = 0) -> AmOwner:
  return AmOwner(context_name="ctx", launch_generation=0, ctx_id=ctx_id, tile_id=0, epoch=epoch)


def _acquire(
  manager, ctx_id: int, *, name: str, logical_slot: int, cycle: int, bundle_slots: int = 1, wake=None
):
  return manager.request(
    _owner(ctx_id),
    buffer_name=name,
    logical_slot=logical_slot,
    dims=_DIMS,
    byte_count=_CAPACITY,
    output_base=f"l1:{name}",
    bundle_slots=bundle_slots,
    cycle=cycle,
    wake=wake or _never_woken,
  )


def _commit_final(manager, lease: AmLease, cycle: int) -> None:
  """Drive one granted chain through INIT+FINAL commit; the slot is not freed."""
  manager.begin_step(lease.lease_id, seq=0, init=True, final=True, cycle=cycle)
  manager.commit_step(lease.lease_id, seq=0, data=_PAYLOAD, final=True, cycle=cycle + 1)


def _assert_accounting(manager) -> dict[int, int]:
  """Every engine id is present and the handles sum to the live leases."""
  snapshot = manager.snapshot()
  counts = snapshot["engine_handle_counts"]
  assert list(counts) == list(range(snapshot["engine_count"]))
  assert sum(counts.values()) == snapshot["live_leases"]
  return counts


def _named_events(tracer: Tracer, name: str) -> list[dict]:
  events = json.loads(tracer.to_chrome_json())["traceEvents"]
  return [event for event in events if event["name"] == name and event["ph"] == "i"]


# ---------------------------------------------------------------------------
# Engine pool configuration
# ---------------------------------------------------------------------------


def test_engine_count_is_an_exact_bounded_integer():
  for count in range(1, MAX_BOA_ENGINES + 1):
    assert _manager(count).snapshot()["engine_count"] == count
  for bad in (True, False, 0, -1, MAX_BOA_ENGINES + 1, 2.0, "2"):
    with pytest.raises(ValueError):
      _manager(bad)


def test_engine_ids_are_independent_of_physical_groups():
  tracer = Tracer(HardwareConfig())
  manager = _manager(2, tracer=tracer)
  engines = []
  groups = []
  for index in range(8):
    lease = _acquire(manager, index, name=f"a{index}", logical_slot=index % 2, cycle=index)
    assert lease is not None
    engines.append(lease.bound_engine)
    groups.append(lease.group)
  # Slots fill groups 0..3 in physical order while the least-live rule walks
  # the engine pool: the two id domains cross and never derive each other.
  assert groups == [group for group in range(4) for _ in range(2)]
  assert engines == [0, 1, 0, 1, 0, 1, 0, 1]
  assert _assert_accounting(manager) == {0: 4, 1: 4}
  allocate = _named_events(tracer, "am_allocate")[6]
  # The seventh lease sits in physical group 3 yet binds engine 0: the group
  # never pins the engine.
  assert allocate["args"]["slot"] == "3.0"
  assert allocate["args"]["bound_engine"] == 0


# ---------------------------------------------------------------------------
# Grant-time least-live selection with lowest-id ties
# ---------------------------------------------------------------------------


def test_grant_walks_engines_by_live_count_with_lowest_id_ties():
  manager = _manager(4)
  engines = []
  for index in range(8):
    lease = _acquire(manager, index, name=f"a{index}", logical_slot=index % 2, cycle=index)
    assert lease is not None
    engines.append(lease.bound_engine)
    assert _assert_accounting(manager) == {
      engine: engines.count(engine) for engine in range(4)
    }
  # Four ties at (1,1,1,1) go back to engine 0: lowest id wins every tie.
  assert engines == [0, 1, 2, 3, 0, 1, 2, 3]
  # Freeing the engine-2 lease makes it the unique least-live engine.
  manager.abort_lease(manager.lease(3).lease_id, 9, "test")
  ninth = _acquire(manager, 8, name="a8", logical_slot=0, cycle=10)
  assert ninth is not None
  assert ninth.bound_engine == 2
  assert _assert_accounting(manager) == {0: 2, 1: 2, 2: 2, 3: 2}


def test_grant_trace_carries_atomic_slot_engine_quota_and_counts():
  tracer = Tracer(HardwareConfig())
  manager = _manager(3, tracer=tracer)
  first = _acquire(manager, 0, name="a", logical_slot=0, cycle=7)
  assert first is not None
  second = _acquire(manager, 1, name="b", logical_slot=1, cycle=8)
  assert second is not None
  events = _named_events(tracer, "am_allocate")
  assert [event["args"]["bound_engine"] for event in events] == [0, 1]
  grant = events[1]["args"]
  # One trace event at the grant cycle: slot label, engine binding, the
  # owner's quota/live handles and the pool-wide counts land together.
  assert grant["slot"] == "0.1" and grant["generation"] == 1
  assert grant["quota_slots"] == 2
  assert grant["owner_live_handles"] == 1
  assert grant["engine_live_handles"] == 1
  assert grant["live_leases"] == 2
  assert grant["engine_count"] == 3
  assert _named_events(tracer, "am_request_wake") == []


# ---------------------------------------------------------------------------
# Exactly-once decrement before waking parked requests
# ---------------------------------------------------------------------------


def test_decrement_happens_before_the_parked_request_is_woken():
  manager = _manager(2, group_count=2)
  first = _acquire(manager, 0, name="a", logical_slot=0, cycle=0)
  second = _acquire(manager, 1, name="b", logical_slot=1, cycle=1)
  third = _acquire(manager, 2, name="c", logical_slot=0, cycle=2)
  assert [lease.bound_engine for lease in (first, second, third)] == [0, 1, 0]
  assert _assert_accounting(manager) == {0: 2, 1: 1}
  woken: list[tuple[AmLease, int]] = []
  # Only one physical slot is free, so the fresh bundle-2 request parks.
  parked = _acquire(
    manager, 3, name="d", logical_slot=0, cycle=3, bundle_slots=2,
    wake=lambda lease, cycle: woken.append((lease, cycle)),
  )
  assert parked is None and manager.snapshot()["pending_requests"] == 1
  # Freeing the engine-0 lease drops its count to (1,1) *before* the wake
  # grant, so the parked request ties onto engine 0.  A decrement after the
  # wake would present (2,1) and bind engine 1 instead.
  manager.abort_lease(third.lease_id, 4, "test")
  assert len(woken) == 1
  reborn = woken[0][0]
  assert reborn.bound_engine == 0
  assert _assert_accounting(manager) == {0: 2, 1: 1}


# ---------------------------------------------------------------------------
# Lifetime: FINAL holds the handle; affinity never migrates
# ---------------------------------------------------------------------------


def test_final_commit_holds_handle_through_output_backpressure():
  tracer = Tracer(HardwareConfig())
  manager = _manager(2, group_count=2, tracer=tracer)
  lease = _acquire(manager, 0, name="a", logical_slot=0, cycle=0)
  assert lease is not None and lease.bound_engine == 0
  _commit_final(manager, lease, 1)
  # FINAL consumed the chain's source liveness but the slot, quota and engine
  # handle stay live through the output/backpressure window.
  assert manager.snapshot()["engine_handle_counts"] == {0: 1, 1: 0}
  assert manager.snapshot()["live_leases"] == 1
  manager.release_final(lease.lease_id, 2)
  counts = _assert_accounting(manager)
  assert counts == {0: 0, 1: 0}
  release = _named_events(tracer, "am_release")[0]["args"]
  assert release["bound_engine"] == 0
  assert release["engine_live_handles"] == 0


def test_bound_engine_stays_fixed_while_accounting_shifts():
  tracer = Tracer(HardwareConfig())
  manager = _manager(4, tracer=tracer)
  first = _acquire(manager, 0, name="a", logical_slot=0, cycle=0)
  second = _acquire(manager, 1, name="b", logical_slot=1, cycle=1)
  assert (first.bound_engine, second.bound_engine) == (0, 1)
  _commit_final(manager, first, 2)
  # The FINAL commit holds engine 0's handle, so the pool reads (1,1,0,0) and
  # the next grant legitimately lands on the idle engine 2.  The still-live
  # leases keep their original bindings: no re-evaluation, no API re-pins a
  # live lease, even after engine 0's handle finally dies.
  third = _acquire(manager, 2, name="c", logical_slot=0, cycle=3)
  assert third.bound_engine == 2
  assert (first.bound_engine, second.bound_engine) == (0, 1)
  manager.release_final(first.lease_id, 4)
  assert (first.bound_engine, second.bound_engine, third.bound_engine) == (0, 1, 2)
  for event in _named_events(tracer, "am_step") + _named_events(tracer, "am_commit"):
    assert event["args"]["bound_engine"] == 0
  assert _assert_accounting(manager) == {0: 0, 1: 1, 2: 1, 3: 0}


def test_step_and_commit_trace_carry_the_granted_engine():
  tracer = Tracer(HardwareConfig())
  manager = _manager(2, group_count=2, tracer=tracer)
  lease = _acquire(manager, 0, name="a", logical_slot=0, cycle=0)
  assert lease is not None
  manager.begin_step(lease.lease_id, seq=0, init=True, final=False, cycle=1)
  manager.commit_step(lease.lease_id, seq=0, data=_PAYLOAD, final=False, cycle=2)
  manager.begin_step(lease.lease_id, seq=1, init=False, final=True, cycle=3)
  manager.commit_step(lease.lease_id, seq=1, data=_PAYLOAD, final=True, cycle=4)
  manager.release_final(lease.lease_id, 5)
  engines = {
    name: [event["args"]["bound_engine"] for event in _named_events(tracer, name)]
    for name in ("am_allocate", "am_step", "am_commit", "am_release")
  }
  assert engines == {
    "am_allocate": [0],
    "am_step": [0, 0],
    "am_commit": [0, 0],
    "am_release": [0],
  }


# ---------------------------------------------------------------------------
# Reset / abort / rebind accounting
# ---------------------------------------------------------------------------


def test_abort_and_reset_zero_every_engine_domain():
  manager = _manager(3)
  for index in range(3):
    lease = _acquire(manager, index, name=f"a{index}", logical_slot=index % 2, cycle=index)
    assert lease is not None
  assert _assert_accounting(manager) == {0: 1, 1: 1, 2: 1}
  manager.abort_lease(manager.lease(2).lease_id, 4, "test")
  assert _assert_accounting(manager) == {0: 1, 1: 0, 2: 1}
  manager.reset(cycle=5)
  snapshot = manager.snapshot()
  assert snapshot["engine_handle_counts"] == {0: 0, 1: 0, 2: 0}
  assert snapshot["live_leases"] == 0
  assert snapshot["pending_requests"] == 0
  # The allocator serves a fresh incarnation immediately after reset.
  reborn = _acquire(manager, 0, name="reborn", logical_slot=0, cycle=6)
  assert reborn is not None and reborn.bound_engine == 0
  assert _assert_accounting(manager) == {0: 1, 1: 0, 2: 0}


def test_owner_abort_frees_all_handles_and_rebind_grants_least_live():
  manager = _manager(2)
  owner = _owner(0)
  for index, name in enumerate(("a", "b")):
    lease = manager.request(
      owner,
      buffer_name=name,
      logical_slot=index,
      dims=_DIMS,
      byte_count=_CAPACITY,
      output_base=f"l1:{name}",
      bundle_slots=2,
      cycle=index,
      wake=_never_woken,
    )
    assert lease is not None
  # Least-live spreads even one owner's two chains across the pool: from an
  # empty start the grants tie onto engine 0, then engine 1.
  assert _assert_accounting(manager) == {0: 1, 1: 1}
  # Context rebind: the whole previous incarnation dies at once.
  assert manager.abort_owner(owner, 2, "rebind") == 2
  assert _assert_accounting(manager) == {0: 0, 1: 0}
  reborn = manager.request(
    _owner(0, epoch=1),
    buffer_name="reborn",
    logical_slot=0,
    dims=_DIMS,
    byte_count=_CAPACITY,
    output_base="l1:reborn",
    bundle_slots=2,
    cycle=3,
    wake=_never_woken,
  )
  assert reborn is not None and reborn.bound_engine == 0
  assert _assert_accounting(manager) == {0: 1, 1: 0}


# ---------------------------------------------------------------------------
# Legal load distributions: no per-engine occupancy cap
# ---------------------------------------------------------------------------


def test_one_engine_serves_eight_live_handles():
  # 总则 §3.1: one engine may serve more than two live handles; only the
  # per-context quota (2) and the physical slots bound occupancy.
  manager = _manager(1)
  for index in range(8):
    lease = _acquire(manager, index, name=f"a{index}", logical_slot=index % 2, cycle=index)
    assert lease is not None, "a legal fill must never be rejected by an engine cap"
    assert lease.bound_engine == 0
  counts = _assert_accounting(manager)
  assert counts == {0: 8}
  assert manager.snapshot()["live_leases"] == 8


def test_two_engines_balance_four_then_eight_live_handles():
  manager = _manager(2)
  engines = []
  for index in range(4):
    lease = _acquire(manager, index, name=f"a{index}", logical_slot=index % 2, cycle=index)
    assert lease is not None
    engines.append(lease.bound_engine)
  # From an empty start the least-live rule yields the 2/2 distribution; it
  # is an outcome of balancing, not a hardware two-handle limit (总则 §3.1).
  assert engines == [0, 1, 0, 1]
  assert _assert_accounting(manager) == {0: 2, 1: 2}
  for index in range(4, 8):
    lease = _acquire(manager, index, name=f"a{index}", logical_slot=index % 2, cycle=index)
    assert lease is not None
    engines.append(lease.bound_engine)
  assert engines == [0, 1, 0, 1, 0, 1, 0, 1]
  # Eight handles on two engines: four live handles per engine is legal.
  assert _assert_accounting(manager) == {0: 4, 1: 4}


def test_four_engines_balance_eight_live_handles():
  manager = _manager(4)
  engines = []
  for index in range(8):
    lease = _acquire(manager, index, name=f"a{index}", logical_slot=index % 2, cycle=index)
    assert lease is not None
    engines.append(lease.bound_engine)
  assert engines == [0, 1, 2, 3, 0, 1, 2, 3]
  assert _assert_accounting(manager) == {0: 2, 1: 2, 2: 2, 3: 2}
