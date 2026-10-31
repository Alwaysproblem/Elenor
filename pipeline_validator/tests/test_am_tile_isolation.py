"""Q08a/Q08b + RPT-P04-3: per-Tile AM allocator isolation (plan P04 SW-A1).

Every case assembles the production topology (``TileGroup`` -> ``ComputeTile``
-> the tile's own ``SharedAmManager``) and drives the real allocator/context
lifecycle; no case constructs standalone managers as a fake topology.  Under
the pre-cutover Group-shared single-manager assembly these cases fail (tiles
contend for one 8-slot pool, snapshots lack ``tile_id``/per-Tile rows); after
the per_tile_v1 cutover they must pass unchanged.

Covered acceptance:
- Q08a: 4-Tile assembly, Tile0's 8 slots legally full at per-owner quota 2,
  Tile1 (same context identity apart from tile_id) still grants at the same
  requested cycle; per-Tile snapshots and aggregate sums stay independent.
- Q08b: Tile0 context fault / reset / rebind only touches Tile0's allocator
  while a Tile1 live lease and parked request remain byte-identical; the
  freed Tile0 slot serves Tile0's own FIFO waiter, never a foreign Tile.
- RPT-P04-3: report ``am_slot_zero_leak`` names the offending Tile(s);
  ``am_fault_expected`` requires exact Tile attribution (a wrong-Tile
  expectation fails despite an equal total fault count; global ``True`` is
  rejected); all Tile rows survive report bounding with >16 Tiles.

Run:
  conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_am_tile_isolation.py -q
  conda run -n elenor-validator python -m pipeline_validator.tests.test_am_tile_isolation
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from pipeline_validator.compiled_program import WorkloadInfo
from pipeline_validator.config import HardwareConfig
from pipeline_validator.execution_ir import ExecTileInst, ExecTileOp, ExecTileProgram
from pipeline_validator.report import WorkloadReport, build_report, report_to_json
from pipeline_validator.runtime.shared_am import AmLease, AmOwner
from pipeline_validator.simulator import SimResult
from pipeline_validator.tile_group import TileGroup

_CAPACITY = 65536
_FILL_CYCLE = 100
_FAULT_CYCLE = 200
_RESET_CYCLE = 300
_DIMS = (64, 256)
_PAYLOAD = b"\x00" * (64 * 256 * 4)
_GROUP_COUNT = 4
_SLOTS_PER_GROUP = 2
_SLOTS_PER_CONTEXT = 2


# ---------------------------------------------------------------------------
# Production assembly and allocator helpers
# ---------------------------------------------------------------------------


def _hw(num_tiles: int) -> HardwareConfig:
  return HardwareConfig().with_overrides(
    num_tiles=num_tiles,
    boa_am_slot_capacity_bytes=_CAPACITY,
    boa_am_group_count=_GROUP_COUNT,
    boa_am_slots_per_group=_SLOTS_PER_GROUP,
    boa_am_slots_per_context=_SLOTS_PER_CONTEXT,
  )


def _group(num_tiles: int = 4, context_count: int = 1, **overrides: int) -> TileGroup:
  return TileGroup(_hw(num_tiles).with_overrides(**overrides), fidelity="runtime", context_count=context_count)


def _require_per_tile_allocators(group: TileGroup) -> list:
  """Assert the cutover topology: one dedicated 8-slot allocator per Tile."""
  managers = [tile.shared_am for tile in group.tiles]
  assert len({id(manager) for manager in managers}) == len(group.tiles), (
    "each ComputeTile must own exactly one dedicated SharedAmManager"
  )
  for tile_id, manager in enumerate(managers):
    snapshot = manager.snapshot()
    assert snapshot["tile_id"] == tile_id
    assert snapshot["group_count"] == _GROUP_COUNT
    assert snapshot["slots_per_group"] == _SLOTS_PER_GROUP
    assert snapshot["slots_per_context"] == _SLOTS_PER_CONTEXT
    assert snapshot["total_slots"] == _GROUP_COUNT * _SLOTS_PER_GROUP
  return managers


def _owner(context_name: str, ctx_id: int, tile_id: int, *, epoch: int = 0) -> AmOwner:
  return AmOwner(
    context_name=context_name, launch_generation=0, ctx_id=ctx_id, tile_id=tile_id, epoch=epoch
  )


def _request(
  manager,
  owner: AmOwner,
  name: str,
  logical_slot: int,
  cycle: int,
  *,
  bundle_slots: int = 1,
  wake=None,
):
  return manager.request(
    owner,
    buffer_name=name,
    logical_slot=logical_slot,
    dims=_DIMS,
    byte_count=_CAPACITY,
    output_base=f"l1:{name}",
    bundle_slots=bundle_slots,
    cycle=cycle,
    wake=wake or _never_woken,
  )


def _never_woken(lease: AmLease, cycle: int) -> None:
  raise AssertionError(f"unexpected AM wake for lease {lease.lease_id} at cycle {cycle}")


def _finish_chain(manager, lease: AmLease, cycle: int) -> None:
  """Drive one granted chain through INIT+FINAL commit and slot_free."""
  manager.begin_step(lease.lease_id, seq=0, init=True, final=True, cycle=cycle)
  manager.commit_step(lease.lease_id, seq=0, data=_PAYLOAD, final=True, cycle=cycle + 1)
  manager.release_final(lease.lease_id, cycle + 2)


def _fill_tile_with_owner(manager, tile_id: int, cycle: int, first_owner: AmOwner, name: str) -> list:
  """Legally fill all 8 slots: first_owner holds 1 chain, fillers 2/2/2/1."""
  lease = _request(manager, first_owner, name, 0, cycle)
  assert lease is not None, f"Tile{tile_id} owner lease must grant on its own allocator"
  leases = [lease]
  for index, (bundle, count) in enumerate(((2, 2), (2, 2), (2, 2), (1, 1))):
    owner = _owner(f"fill{index}", index, tile_id)
    for logical in range(count):
      filler = _request(
        manager, owner, f"t{tile_id}f{index}s{logical}", logical, cycle, bundle_slots=bundle
      )
      assert filler is not None, f"Tile{tile_id} legal fill must grant within its own 8 slots"
      leases.append(filler)
  return leases


def _fill_and_park(manager, tile_id: int, cycle: int) -> tuple[dict, list[int]]:
  """Fill all 8 slots, then park one fresh request; returns (snapshot, wake_cycles)."""
  for ctx in range(_GROUP_COUNT):
    owner = _owner("ctx", ctx, tile_id)
    for logical in (0, 1):
      lease = _request(
        manager, owner, f"t{tile_id}c{ctx}s{logical}", logical, cycle, bundle_slots=2
      )
      assert lease is not None, f"Tile{tile_id} fill must grant on its own allocator"
  wakes: list[int] = []
  parked = _request(
    manager,
    _owner("waiter", 9, tile_id),
    f"t{tile_id}wait",
    0,
    cycle,
    wake=lambda lease, grant_cycle: wakes.append(grant_cycle),
  )
  assert parked is None, f"Tile{tile_id} waiter must park while its own allocator is full"
  return manager.snapshot(), wakes


def _run_to_completion(tile, *, max_steps: int = 64) -> None:
  """Step the tile UCE until its single bound context retires (RET at end)."""
  for step in range(max_steps):
    if not tile.uce.has_active_contexts():
      return
    tile.uce.step(1 + step, tile)
  raise AssertionError("UCE context did not retire within the step budget")


def _bind(tile, program_name: str, role_event_id: str, context_id: int) -> None:
  # An empty instruction stream faults (no RET); bind a minimal legal program.
  program = ExecTileProgram(
    name=program_name, insts=(ExecTileInst(op=ExecTileOp.RET, instruction_id=f"{program_name}.ret"),)
  )
  bound = tile.uce.bind_context(
    program, role_id=0, role_event_id=role_event_id, context_id=context_id, cycle=0
  )
  assert bound == context_id, f"Tile{tile.tile_id} must bind hardware context {context_id}"


# ---------------------------------------------------------------------------
# Q08a: Tile isolation capacity
# ---------------------------------------------------------------------------


def _case_q08a_isolation() -> dict:
  group = _group(num_tiles=4)
  managers = _require_per_tile_allocators(group)
  tile0_leases = []
  for ctx in range(_GROUP_COUNT):
    owner = _owner("ctx", ctx, 0)
    for logical in (0, 1):
      lease = _request(
        managers[0], owner, f"t0c{ctx}s{logical}", logical, _FILL_CYCLE, bundle_slots=2
      )
      assert lease is not None, "Tile0 legal fill (quota 2 per owner) must grant in its own 8 slots"
      tile0_leases.append(lease)
  wakes: list[int] = []
  tile1_lease = _request(
    managers[1],
    _owner("ctx", 0, 1),
    "t1c0s0",
    0,
    _FILL_CYCLE,
    wake=lambda lease, grant_cycle: wakes.append(grant_cycle),
  )
  assert tile1_lease is not None, "Tile1 must grant at its own request cycle while Tile0 is full"
  assert wakes == []
  tile0_after = managers[0].snapshot()
  tile1_after = managers[1].snapshot()
  idle_after = managers[2].snapshot()
  group_snapshot = group.snapshot()
  aggregate = group_snapshot["shared_am"]
  per_tile = group_snapshot["shared_am_per_tile"]
  return {
    "tile_ids": [manager.snapshot()["tile_id"] for manager in managers],
    "tile0": {
      "free_slots": tile0_after["free_slots"],
      "live_leases": tile0_after["live_leases"],
      "request_count": tile0_after["request_count"],
      "acquire_count": tile0_after["acquire_count"],
      "parked_count": tile0_after["parked_count"],
      "fault_count": tile0_after["fault_count"],
    },
    "tile1": {
      "free_slots": tile1_after["free_slots"],
      "live_leases": tile1_after["live_leases"],
      "request_count": tile1_after["request_count"],
      "acquire_count": tile1_after["acquire_count"],
      "parked_count": tile1_after["parked_count"],
    },
    "idle_tile2": {"free_slots": idle_after["free_slots"], "total_slots": idle_after["total_slots"]},
    "tile0_physical_slots": sorted([lease.group, lease.slot] for lease in tile0_leases),
    # Engine ids come from the engine pool, not from slot groups: the default
    # single-engine pool binds every lease to engine 0 while its slot may sit
    # in any physical group.
    "engine_affinity_independent_of_groups": {
      "engine_ids": sorted({lease.bound_engine for lease in [*tile0_leases, tile1_lease]}),
      "group_ids": sorted({lease.group for lease in [*tile0_leases, tile1_lease]}),
      "tile0_engine_handle_counts": managers[0].snapshot()["engine_handle_counts"],
      "tile0_live_leases": managers[0].snapshot()["live_leases"],
    },
    "tile1_lease": {
      "owner_tile_id": tile1_lease.owner.tile_id,
      "logical_slot": tile1_lease.logical_slot,
    },
    "per_tile_rows": [row["tile_id"] for row in per_tile],
    "per_tile_free_slots": [row["free_slots"] for row in per_tile],
    "aggregate": {
      "topology": aggregate["topology"],
      "tile_count": aggregate["tile_count"],
      "slot_capacity_bytes": aggregate["slot_capacity_bytes"],
      "groups_per_tile": aggregate["groups_per_tile"],
      "slots_per_group": aggregate["slots_per_group"],
      "slots_per_context": aggregate["slots_per_context"],
      "total_slots": aggregate["total_slots"],
      "free_slots": aggregate["free_slots"],
      "live_leases": aggregate["live_leases"],
      "request_count": aggregate["request_count"],
      "acquire_count": aggregate["acquire_count"],
      "fault_count": aggregate["fault_count"],
      "sum_tile_peak_live": aggregate["sum_tile_peak_live"],
      "sum_tile_peak_pending": aggregate["sum_tile_peak_pending"],
      "has_global_peak_keys": "peak_live" in aggregate or "peak_pending" in aggregate,
    },
  }


# ---------------------------------------------------------------------------
# Q08b: fault / reset / rebind stay inside Tile0
# ---------------------------------------------------------------------------


def _case_q08b_parked_request_fault() -> dict:
  """A WAIT_AM context fault cancels only its own parked request."""
  group = _group(num_tiles=4, context_count=2)
  managers = _require_per_tile_allocators(group)
  tile0 = group.tiles[0]
  _bind(tile0, "q08b_park_ctx0", "role0", 0)
  _bind(tile0, "q08b_park_ctx1", "role1", 1)
  owner0 = tile0.uce._am_owner(tile0.uce.contexts[0])
  owner1 = tile0.uce._am_owner(tile0.uce.contexts[1])
  _fill_tile_with_owner(managers[0], 0, _FILL_CYCLE, owner0, "t0ctx0")
  tile1_before, tile1_wakes = _fill_and_park(managers[1], 1, _FILL_CYCLE)
  tile0_wakes: list[int] = []
  parked = _request(
    managers[0],
    owner1,
    "t0ctx1",
    1,
    _FILL_CYCLE,
    wake=lambda lease, grant_cycle: tile0_wakes.append(grant_cycle),
  )
  assert parked is None, "Tile0 WAIT_AM context must park while Tile0 is full"
  assert managers[0].snapshot()["pending_requests"] == 1
  # TileUCE._fault_context is the production funnel every context fault goes
  # through (instruction faults, engine failures, tile.return leaks).
  tile0.uce._fault_context(tile0.uce.contexts[1], "engine completion failed: boa", _FAULT_CYCLE)
  tile0_after = managers[0].snapshot()
  return {
    "tile0": {
      "pending_requests": tile0_after["pending_requests"],
      "cancelled_request_count": tile0_after["cancelled_request_count"],
      "fault_count": tile0_after["fault_count"],
      "live_leases": tile0_after["live_leases"],
      "wake_count": tile0_after["wake_count"],
      "wake_cycles": tile0_wakes,
      "ctx0_lease_live": managers[0].live_for_owner(owner0),
      "terminal_count": len(tile0_after["terminals"]),
      "uce_faulted": tile0.uce.faulted,
      "uce_fault_reason": tile0.uce.fault_reason,
    },
    "tile1_unchanged": managers[1].snapshot() == tile1_before,
    "tile1_wake_cycles": tile1_wakes,
    "tile1_uce_faulted": group.tiles[1].uce.faulted,
  }


def _case_q08b_live_lease_fault() -> dict:
  """A faulted live lease frees its slot to Tile0's own FIFO waiter only."""
  group = _group(num_tiles=4)
  managers = _require_per_tile_allocators(group)
  tile0 = group.tiles[0]
  _bind(tile0, "q08b_live_ctx0", "ctx", 0)
  owner0 = tile0.uce._am_owner(tile0.uce.contexts[0])
  _fill_tile_with_owner(managers[0], 0, _FILL_CYCLE, owner0, "t0ctx0")
  tile1_before, tile1_wakes = _fill_and_park(managers[1], 1, _FILL_CYCLE)
  tile0_wakes: list[int] = []
  waiter0 = _owner("waiter", 8, 0)
  parked = _request(
    managers[0],
    waiter0,
    "t0wait",
    0,
    _FILL_CYCLE,
    wake=lambda lease, grant_cycle: tile0_wakes.append(grant_cycle),
  )
  assert parked is None, "Tile0 waiter must park while Tile0 is full"
  tile0.uce._fault_context(tile0.uce.contexts[0], "engine completion failed: boa", _FAULT_CYCLE)
  tile0_after = managers[0].snapshot()
  fault_terminals = [entry for entry in tile0_after["terminals"] if entry["kind"] == "fault"]
  uce_terminals = tile0.uce.drain_terminal_events()
  return {
    "tile0": {
      "fault_count": tile0_after["fault_count"],
      "wake_count": tile0_after["wake_count"],
      "wake_cycles": tile0_wakes,
      "pending_requests": tile0_after["pending_requests"],
      "live_leases": tile0_after["live_leases"],
      "faulted_owner_live": managers[0].live_for_owner(owner0),
      "waiter_live": managers[0].live_for_owner(waiter0),
      "fault_terminals": [
        {
          "tile_id": entry["tile_id"],
          "owner": entry["owner"],
          "success": entry["success"],
          "detail": entry["detail"],
        }
        for entry in fault_terminals
      ],
      "uce_fault_reason": tile0.uce.fault_reason,
      "uce_terminals": [{"status": event.status, "ctx_id": event.ctx_id} for event in uce_terminals],
    },
    "tile1_unchanged": managers[1].snapshot() == tile1_before,
    "tile1_wake_cycles": tile1_wakes,
  }


def _case_q08b_tile_reset() -> dict:
  """ComputeTile.reset clears exactly its own allocator; Tile1 is untouched."""
  group = _group(num_tiles=4)
  managers = _require_per_tile_allocators(group)
  _fill_tile_with_owner(managers[0], 0, _FILL_CYCLE, _owner("ctx", 0, 0), "t0c0s0")
  waiter0 = _request(managers[0], _owner("waiter", 8, 0), "t0wait", 0, _FILL_CYCLE)
  assert waiter0 is None, "Tile0 waiter must park while Tile0 is full"
  tile1_before, tile1_wakes = _fill_and_park(managers[1], 1, _FILL_CYCLE)
  tile0 = group.tiles[0]
  tile0.reset(_RESET_CYCLE)
  tile0_after = managers[0].snapshot()
  reborn = _request(managers[0], _owner("ctx", 0, 0, epoch=1), "t0reborn", 0, _RESET_CYCLE)
  assert reborn is not None, "Tile0 allocator must serve a fresh incarnation right after reset"
  reborn_engine_handle_counts = managers[0].snapshot()["engine_handle_counts"]
  return {
    "tile0": {
      "free_slots": tile0_after["free_slots"],
      "live_leases": tile0_after["live_leases"],
      "pending_requests": tile0_after["pending_requests"],
      "reserved_slots": tile0_after["reserved_slots"],
      "terminals": [
        (entry["kind"], entry["success"], entry["detail"], entry["cycle"]) for entry in tile0_after["terminals"]
      ],
      # 8 allocation bumps (one per granted slot) plus exactly one reset bump.
      "slot_generations": sorted({slot["generation"] for slot in tile0_after["slots"]}),
      "request_count": tile0_after["request_count"],
      "uce_active_contexts": tile0.uce.has_active_contexts(),
      "reborn_bound_engine": reborn.bound_engine,
      "reborn_group": reborn.group,
      "reborn_engine_handle_counts": reborn_engine_handle_counts,
    },
    "tile1_unchanged": managers[1].snapshot() == tile1_before,
    "tile1_wake_cycles": tile1_wakes,
  }


def _case_q08b_rebind() -> dict:
  """Re-binding a DONE context aborts only the previous incarnation's lease."""
  # frame_bind_cycles=0 keeps the UCE FSM on the RET-only path; the case under
  # test is AM owner rebind, not the L1 Arena/Frame admission ticket.
  group = _group(num_tiles=4, frame_bind_cycles=0)
  managers = _require_per_tile_allocators(group)
  tile0 = group.tiles[0]
  _bind(tile0, "q08b_rebind_first", "role0", 0)
  _run_to_completion(tile0)
  owner_prev = tile0.uce._am_owner(tile0.uce.contexts[0])
  stale = _request(managers[0], owner_prev, "t0stale", 0, 40)
  assert stale is not None, "previous incarnation lease must grant on Tile0's own allocator"
  tile1_before, tile1_wakes = _fill_and_park(managers[1], 1, 50)
  rebound = tile0.uce.bind_context(
    ExecTileProgram(
      name="q08b_rebind_second",
      insts=(ExecTileInst(op=ExecTileOp.RET, instruction_id="q08b_rebind_second.ret"),),
    ),
    role_id=0,
    role_event_id="role0",
    context_id=0,
    cycle=55,
  )
  assert rebound == 0, "a retired context must be re-bindable"
  tile0_after = managers[0].snapshot()
  rebind_terminals = [
    {"tile_id": entry["tile_id"], "kind": entry["kind"], "owner_epoch": entry["owner_epoch"], "cycle": entry["cycle"]}
    for entry in tile0_after["terminals"]
    if entry["detail"] == "context rebind"
  ]
  owner_next = tile0.uce._am_owner(tile0.uce.contexts[0])
  again = _request(managers[0], owner_next, "t0next", 0, 60)
  assert again is not None, "the re-bound incarnation must reuse the freed logical slot"
  return {
    "tile0": {
      "stale_owner_live": managers[0].live_for_owner(owner_prev),
      "free_slots_at_rebind": tile0_after["free_slots"],
      "rebind_terminals": rebind_terminals,
      "fault_count": tile0_after["fault_count"],
      "prev_epoch": owner_prev.epoch,
      "next_epoch": owner_next.epoch,
      "next_logical_slot": again.logical_slot,
      "uce_faulted": tile0.uce.faulted,
    },
    "tile1_unchanged": managers[1].snapshot() == tile1_before,
    "tile1_wake_cycles": tile1_wakes,
  }


def _case_group_reset_recovery() -> dict:
  """TileGroup.reset reaches every Tile allocator exactly once."""
  group = _group(num_tiles=4)
  managers = _require_per_tile_allocators(group)
  # TileGroup.__init__ only enqueues profile initialization; the group's step
  # loop completes it.  Drive the same controller steps so explicit recovery
  # is legal on this assembly before exercising the reset path.
  cycle = 0
  init_bound = group.cfg.memory_target.profile_command_timeout_cycles
  while not group.profile_controller.initialized:
    group.profile_controller.step(cycle)
    cycle += 1
    assert cycle <= init_bound, "profile initialization did not complete"
  lease0 = _request(managers[0], _owner("ctx", 0, 0), "t0c0s0", 0, _FILL_CYCLE)
  assert lease0 is not None
  _finish_chain(managers[0], lease0, _FILL_CYCLE + 1)
  lease2 = _request(managers[2], _owner("ctx", 2, 2), "t2c2s0", 0, _FILL_CYCLE)
  assert lease2 is not None
  pre_generations = [
    {slot["slot"]: slot["generation"] for slot in manager.snapshot()["slots"]} for manager in managers
  ]
  group.reset()
  tiles = []
  for tile_id, manager in enumerate(managers):
    after = manager.snapshot()
    drifted = [
      slot["slot"]
      for slot in after["slots"]
      if slot["generation"] != pre_generations[tile_id][slot["slot"]] + 1
    ]
    tiles.append(
      {
        "tile_id": tile_id,
        "free_slots": after["free_slots"],
        "live_leases": after["live_leases"],
        "pending_requests": after["pending_requests"],
        "reserved_slots": after["reserved_slots"],
        "terminals": [(entry["kind"], entry["detail"]) for entry in after["terminals"]],
        "slots_reset_exactly_once": not drifted,
        "drifted_slots": drifted,
      }
    )
  return {"tiles": tiles}


# ---------------------------------------------------------------------------
# RPT-P04-3: per-Tile report attribution and bounding
# ---------------------------------------------------------------------------


def _build_report(group: TileGroup, expected: dict) -> WorkloadReport:
  result = SimResult(
    cycles=1000,
    completed=True,
    reason="p04 per-Tile AM isolation",
    group_snapshot=group.snapshot(),
  )
  info = WorkloadInfo(
    name="p04_am_tile_isolation",
    description="per-Tile AM allocator isolation (Q08/RPT-P04-3)",
    expected=expected,
  )
  return build_report(info, result)


def _check(report: WorkloadReport, name: str) -> dict:
  matches = [item for item in report.checks if item.get("check") == name]
  assert len(matches) == 1, f"report must emit the {name} check exactly once"
  return matches[0]


def _as_str_map(mapping: object) -> dict[str, int]:
  assert isinstance(mapping, Mapping), f"expected a mapping, got {type(mapping).__name__}"
  return {str(key): int(value) for key, value in mapping.items()}


def _case_report_zero_leak() -> dict:
  group = _group(num_tiles=4)
  managers = _require_per_tile_allocators(group)
  leaked = _request(managers[1], _owner("ctx", 1, 1), "t1c1s0", 0, _FILL_CYCLE)
  assert leaked is not None
  for tile_id in (0, 2, 3):
    lease = _request(
      managers[tile_id], _owner("ctx", tile_id, tile_id), f"t{tile_id}clean", 0, _FILL_CYCLE
    )
    assert lease is not None
    _finish_chain(managers[tile_id], lease, _FILL_CYCLE + 1)
  report = _build_report(group, {})
  zero = _check(report, "am_slot_zero_leak")
  return {
    "pass": zero["pass"],
    "expected": zero["expected"],
    "actual": zero["actual"],
    "rows": [
      {
        "tile_id": row["tile_id"],
        "pass": row["pass"],
        "free_slots": row["free_slots"],
        "total_slots": row["total_slots"],
        "live_leases": row["live_leases"],
        "pending_requests": row["pending_requests"],
        "reserved_slots": row["reserved_slots"],
      }
      for row in zero["per_tile"]
    ],
    "report_row_count": len(report.shared_am_per_tile),
  }


def _fault_isolation_group() -> TileGroup:
  """Real Tile0 context fault plus a clean Tile1 chain; Tiles 2/3 idle."""
  group = _group(num_tiles=4)
  managers = _require_per_tile_allocators(group)
  tile0 = group.tiles[0]
  _bind(tile0, "rpt_fault_ctx0", "ctx", 0)
  owner0 = tile0.uce._am_owner(tile0.uce.contexts[0])
  lease0 = _request(managers[0], owner0, "t0ctx0", 0, _FILL_CYCLE)
  assert lease0 is not None
  tile0.uce._fault_context(tile0.uce.contexts[0], "engine completion failed: boa", _FAULT_CYCLE)
  lease1 = _request(managers[1], _owner("ctx", 1, 1), "t1c1s0", 0, _FILL_CYCLE)
  assert lease1 is not None
  _finish_chain(managers[1], lease1, _FILL_CYCLE + 1)
  return group


def _case_report_fault_attribution() -> dict:
  group = _fault_isolation_group()
  baseline = _build_report(group, {})
  success = _check(baseline, "am_chain_success")
  zero = _check(baseline, "am_slot_zero_leak")
  named = _build_report(group, {"am_fault_expected": {0: 1}})
  named_check = _check(named, "am_fault_expected")
  blamed = _build_report(group, {"am_fault_expected": {1: 1}})
  blamed_check = _check(blamed, "am_fault_expected")
  rejected = None
  try:
    _build_report(group, {"am_fault_expected": True})
  except ValueError as exc:
    rejected = str(exc)

  def fault_rows(check: dict) -> list[dict]:
    return [
      {
        "tile_id": row["tile_id"],
        "expected": row["expected"],
        "actual": row["actual"],
        "pass": row["pass"],
      }
      for row in check["per_tile"]
    ]

  return {
    "chain_success": {
      "pass": success["pass"],
      "expected": success["expected"],
      "actual": success["actual"],
      "rows": fault_rows(success),
    },
    "zero_leak_pass": zero["pass"],
    "zero_leak_failing_tiles": [row["tile_id"] for row in zero["per_tile"] if not row["pass"]],
    "baseline_emits_fault_expected_check": any(
      item.get("check") == "am_fault_expected" for item in baseline.checks
    ),
    "named": {
      "pass": named_check["pass"],
      "expected": _as_str_map(named_check["expected"]),
      "actual": _as_str_map(named_check["actual"]),
      "rows": fault_rows(named_check),
    },
    "blamed": {
      "pass": blamed_check["pass"],
      "expected": _as_str_map(blamed_check["expected"]),
      "actual": _as_str_map(blamed_check["actual"]),
      "rows": fault_rows(blamed_check),
    },
    "global_true_rejection": rejected,
  }


def _case_report_boundary() -> dict:
  group = _group(num_tiles=17)
  managers = _require_per_tile_allocators(group)
  leaked = _request(managers[2], _owner("ctx", 2, 2), "t2leak", 0, _FILL_CYCLE)
  assert leaked is not None
  group_snapshot = group.snapshot()
  aggregate = group_snapshot["shared_am"]
  report = _build_report(group, {})
  zero = _check(report, "am_slot_zero_leak")
  payload = json.loads(report_to_json(report))
  row2 = report.shared_am_per_tile[2]
  return {
    "aggregate": {
      "topology": aggregate["topology"],
      "tile_count": aggregate["tile_count"],
      "total_slots": aggregate["total_slots"],
      "free_slots": aggregate["free_slots"],
      "live_leases": aggregate["live_leases"],
      "acquire_count": aggregate["acquire_count"],
      "sum_tile_peak_live": aggregate["sum_tile_peak_live"],
      "sum_tile_peak_pending": aggregate["sum_tile_peak_pending"],
      "has_global_peak_keys": "peak_live" in aggregate or "peak_pending" in aggregate,
    },
    "snapshot_rows": [row["tile_id"] for row in group_snapshot["shared_am_per_tile"]],
    "report_rows": [row["tile_id"] for row in report.shared_am_per_tile],
    "json_rows": [row["tile_id"] for row in payload["shared_am_per_tile"]],
    "leaking_report_row": {
      "tile_id": row2["tile_id"],
      "live_leases": row2["live_leases"],
      "free_slots": row2["free_slots"],
      "total_slots": row2["total_slots"],
    },
    "zero_leak_pass": zero["pass"],
    "zero_leak_row_count": len(zero["per_tile"]),
    "zero_leak_failing_tiles": [row["tile_id"] for row in zero["per_tile"] if not row["pass"]],
    "leaked_check_row": next(row for row in zero["per_tile"] if row["tile_id"] == 2),
  }


# ---------------------------------------------------------------------------
# Pytest regressions
# ---------------------------------------------------------------------------


def test_q08a_every_tile_owns_an_independent_eight_slot_allocator():
  case = _case_q08a_isolation()
  assert case["tile_ids"] == [0, 1, 2, 3]
  assert case["idle_tile2"] == {"free_slots": 8, "total_slots": 8}
  assert case["per_tile_rows"] == [0, 1, 2, 3]
  aggregate = case["aggregate"]
  assert aggregate["topology"] == "per_tile_v1"
  assert aggregate["tile_count"] == 4
  assert aggregate["slot_capacity_bytes"] == _CAPACITY
  assert aggregate["groups_per_tile"] == _GROUP_COUNT
  assert aggregate["slots_per_group"] == _SLOTS_PER_GROUP
  assert aggregate["slots_per_context"] == _SLOTS_PER_CONTEXT
  assert aggregate["has_global_peak_keys"] is False
  assert aggregate["sum_tile_peak_pending"] == 0


def test_q08a_tile0_full_grants_tile1_at_same_requested_cycle():
  case = _case_q08a_isolation()
  # Tile0 holds all 8 physical slots through four quota-2 owners.
  assert case["tile0"] == {
    "free_slots": 0,
    "live_leases": 8,
    "request_count": 8,
    "acquire_count": 8,
    "parked_count": 0,
    "fault_count": 0,
  }
  assert case["tile0_physical_slots"] == [[group, slot] for group in range(4) for slot in range(2)]
  affinity = case["engine_affinity_independent_of_groups"]
  # All nine leases bind engine 0 while their physical slots span groups 0..3:
  # engine affinity is an independent id domain, never derived from the group.
  assert affinity["engine_ids"] == [0]
  assert affinity["group_ids"] == [0, 1, 2, 3]
  assert affinity["tile0_engine_handle_counts"] == {0: 8}
  assert affinity["tile0_live_leases"] == 8
  # The identical context identity on Tile1 grants immediately at the same
  # cycle; per-Tile counters stay independent and nothing parks.
  assert case["tile1_lease"] == {"owner_tile_id": 1, "logical_slot": 0}
  assert case["tile1"] == {
    "free_slots": 7,
    "live_leases": 1,
    "request_count": 1,
    "acquire_count": 1,
    "parked_count": 0,
  }
  assert case["per_tile_free_slots"] == [0, 7, 8, 8]
  aggregate = case["aggregate"]
  assert aggregate["total_slots"] == 32
  # The aggregate is the sum over per-Tile snapshots: 0 + 7 + 8 + 8.
  assert aggregate["free_slots"] == 23
  assert aggregate["live_leases"] == 9
  assert aggregate["request_count"] == 9
  assert aggregate["acquire_count"] == 9
  assert aggregate["fault_count"] == 0
  assert aggregate["sum_tile_peak_live"] == 9


def test_q08b_parked_request_fault_spares_tile0_peer_and_tile1():
  case = _case_q08b_parked_request_fault()
  tile0 = case["tile0"]
  # The faulted WAIT_AM context loses its parked request, never woken.
  assert tile0["pending_requests"] == 0
  assert tile0["cancelled_request_count"] == 1
  assert tile0["wake_count"] == 0
  assert tile0["wake_cycles"] == []
  assert tile0["fault_count"] == 0
  assert tile0["terminal_count"] == 0
  # The sibling Tile0 context keeps its lease; Tile1 keeps everything.
  assert tile0["live_leases"] == 8
  assert tile0["ctx0_lease_live"] == 1
  assert tile0["uce_faulted"] is True
  assert "engine completion failed" in tile0["uce_fault_reason"]
  assert case["tile1_unchanged"] is True
  assert case["tile1_wake_cycles"] == []
  assert case["tile1_uce_faulted"] is False


def test_q08b_live_lease_fault_frees_slot_to_tile0_waiter_only():
  case = _case_q08b_live_lease_fault()
  tile0 = case["tile0"]
  assert tile0["fault_count"] == 1
  assert tile0["faulted_owner_live"] == 0
  assert tile0["wake_count"] == 1
  assert tile0["wake_cycles"] == [_FAULT_CYCLE]
  assert tile0["pending_requests"] == 0
  # Tile0's own FIFO waiter took the freed slot; no foreign Tile advanced.
  assert tile0["live_leases"] == 8
  assert tile0["waiter_live"] == 1
  assert tile0["fault_terminals"] == [
    {"tile_id": 0, "owner": "ctx", "success": False, "detail": "engine completion failed: boa"}
  ]
  assert tile0["uce_fault_reason"] == "engine completion failed: boa"
  assert tile0["uce_terminals"] == [{"status": "fault", "ctx_id": 0}]
  assert case["tile1_unchanged"] is True
  assert case["tile1_wake_cycles"] == []


def test_q08b_tile_reset_clears_only_its_own_allocator():
  case = _case_q08b_tile_reset()
  tile0 = case["tile0"]
  assert tile0["free_slots"] == 8
  assert tile0["live_leases"] == 0
  assert tile0["pending_requests"] == 0
  assert tile0["reserved_slots"] == 0
  # Reset quiesces and frees all 8 leases, then publishes one failure terminal
  # per lease; the bounded ring is not cleared, so no failure record is lost.
  assert tile0["terminals"] == [("fault", False, "reset", _RESET_CYCLE)] * 8
  assert tile0["slot_generations"] == [2]
  assert tile0["request_count"] == 9
  assert tile0["uce_active_contexts"] is False
  # The reborn lease binds engine 0 of the engine pool (the only engine in the
  # default single-engine topology); its physical group is a separate domain
  # and the engine handle accounting restarts cleanly after reset.
  assert tile0["reborn_bound_engine"] == 0
  assert tile0["reborn_engine_handle_counts"] == {0: 1}
  assert case["tile1_unchanged"] is True
  assert case["tile1_wake_cycles"] == []


def test_q08b_rebind_aborts_only_previous_incarnation_via_tile_allocator():
  case = _case_q08b_rebind()
  tile0 = case["tile0"]
  assert tile0["prev_epoch"] == 1
  assert tile0["next_epoch"] == 2
  assert tile0["stale_owner_live"] == 0
  assert tile0["free_slots_at_rebind"] == 8
  assert tile0["rebind_terminals"] == [{"tile_id": 0, "kind": "fault", "owner_epoch": 1, "cycle": 55}]
  assert tile0["fault_count"] == 1
  assert tile0["next_logical_slot"] == 0
  assert tile0["uce_faulted"] is False
  assert case["tile1_unchanged"] is True
  assert case["tile1_wake_cycles"] == []


def test_group_reset_reaches_every_tile_allocator_exactly_once():
  case = _case_group_reset_recovery()
  assert len(case["tiles"]) == 4
  for tile in case["tiles"]:
    assert tile["free_slots"] == 8
    assert tile["live_leases"] == 0
    assert tile["pending_requests"] == 0
    assert tile["reserved_slots"] == 0
    # Terminals survive reset: Tile0's finished chain stays chain_done and
    # Tile2's live lease publishes exactly one reset failure.
    expected = {0: [("chain_done", "")], 2: [("fault", "reset")]}.get(tile["tile_id"], [])
    assert tile["terminals"] == expected
    assert tile["slots_reset_exactly_once"] is True, f"tile {tile['tile_id']}: {tile['drifted_slots']}"


def test_rpt_zero_leak_names_the_single_offending_tile():
  case = _case_report_zero_leak()
  assert case["pass"] is False
  assert case["expected"] is True
  assert case["actual"] is False
  assert [row["tile_id"] for row in case["rows"]] == [0, 1, 2, 3]
  assert [row["tile_id"] for row in case["rows"] if not row["pass"]] == [1]
  leaked = case["rows"][1]
  assert leaked["live_leases"] == 1
  assert leaked["free_slots"] == 7
  assert leaked["total_slots"] == 8
  assert leaked["pending_requests"] == 0
  assert leaked["reserved_slots"] == 0
  for tile_id in (0, 2, 3):
    row = case["rows"][tile_id]
    assert row["pass"] is True
    assert row["free_slots"] == row["total_slots"] == 8
    assert row["live_leases"] == 0
  assert case["report_row_count"] == 4


def test_rpt_fault_attribution_exact_per_tile_and_wrong_tile_rejected():
  case = _case_report_fault_attribution()
  # Without an expectation the fault surfaces as am_chain_success with exact
  # per-Tile counts: Tile0 faulted once, every other Tile stayed clean.
  success = case["chain_success"]
  assert success["pass"] is False
  assert success["expected"] is True
  assert success["actual"] is False
  assert success["rows"] == [
    {"tile_id": 0, "expected": 0, "actual": 1, "pass": False},
    {"tile_id": 1, "expected": 0, "actual": 0, "pass": True},
    {"tile_id": 2, "expected": 0, "actual": 0, "pass": True},
    {"tile_id": 3, "expected": 0, "actual": 0, "pass": True},
  ]
  assert case["zero_leak_pass"] is True
  assert case["zero_leak_failing_tiles"] == []
  assert case["baseline_emits_fault_expected_check"] is False
  # Exact attribution: the correct mapping passes.  The report emits dense
  # maps: every Tile's entry with implicit zeros for unlisted Tiles.
  named = case["named"]
  assert named["pass"] is True
  assert named["expected"] == {"0": 1, "1": 0, "2": 0, "3": 0}
  assert named["actual"] == {"0": 1, "1": 0, "2": 0, "3": 0}
  assert named["rows"][0] == {"tile_id": 0, "expected": 1, "actual": 1, "pass": True}
  # A wrong-Tile expectation with the same total fault count must fail.
  blamed = case["blamed"]
  assert blamed["pass"] is False
  assert blamed["expected"] == {"0": 0, "1": 1, "2": 0, "3": 0}
  assert blamed["actual"] == {"0": 1, "1": 0, "2": 0, "3": 0}
  assert {"tile_id": 1, "expected": 1, "actual": 0, "pass": False} in blamed["rows"]
  assert {"tile_id": 0, "expected": 0, "actual": 1, "pass": False} in blamed["rows"]
  # Global True is rejected outright; it must not suppress Tile0's fault.
  assert case["global_true_rejection"] is not None


def test_rpt_seventeen_tiles_survive_report_bounding():
  case = _case_report_boundary()
  aggregate = case["aggregate"]
  assert aggregate["topology"] == "per_tile_v1"
  assert aggregate["tile_count"] == 17
  assert aggregate["total_slots"] == 136
  assert aggregate["free_slots"] == 135
  assert aggregate["live_leases"] == 1
  assert aggregate["acquire_count"] == 1
  assert aggregate["sum_tile_peak_live"] == 1
  assert aggregate["has_global_peak_keys"] is False
  # Every physical Tile row survives the snapshot, the report object and the
  # JSON consumer; the legacy 16-row tail cap would have dropped Tile 0.
  assert case["snapshot_rows"] == list(range(17))
  assert case["report_rows"] == list(range(17))
  assert case["json_rows"] == list(range(17))
  assert case["leaking_report_row"] == {
    "tile_id": 2,
    "live_leases": 1,
    "free_slots": 7,
    "total_slots": 8,
  }
  assert case["zero_leak_pass"] is False
  assert case["zero_leak_row_count"] == 17
  assert case["zero_leak_failing_tiles"] == [2]
  assert case["leaked_check_row"]["pass"] is False


# ---------------------------------------------------------------------------
# Module smoke: reuse the actual cases, emit diagnostics to stdout only
# ---------------------------------------------------------------------------


_SMOKE_CASES = {
  "q08a_tile_isolation_capacity": _case_q08a_isolation,
  "q08b_parked_request_fault_scope": _case_q08b_parked_request_fault,
  "q08b_live_lease_fault_scope": _case_q08b_live_lease_fault,
  "q08b_tile_reset_scope": _case_q08b_tile_reset,
  "q08b_rebind_scope": _case_q08b_rebind,
  "group_reset_recovery": _case_group_reset_recovery,
  "rpt_report_zero_leak_attribution": _case_report_zero_leak,
  "rpt_report_fault_attribution": _case_report_fault_attribution,
  "rpt_report_seventeen_tile_boundary": _case_report_boundary,
}


def main() -> int:  # pragma: no cover
  cases: dict[str, dict] = {}
  ok = True
  for name, case in _SMOKE_CASES.items():
    try:
      cases[name] = {"ok": True, **case()}
    except Exception as exc:  # smoke reports any case failure inline
      ok = False
      cases[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
  print(
    json.dumps(
      {"workload": "p04_am_tile_isolation", "all_ok": ok, "cases": cases},
      indent=2,
      sort_keys=True,
    )
  )
  return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
