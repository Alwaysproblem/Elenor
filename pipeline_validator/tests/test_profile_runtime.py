from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline_validator.compiler import compile_program
from pipeline_validator.compiler.resources import conservative_arena_bytes, layout_buffers
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.device import DeviceControlRequest
from pipeline_validator.execution_ir import ExecL2Buffer, GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.memory import (
  AdmissionFailure,
  HBMRegion,
  MemoryInvariantError,
  MemoryTransaction,
  NoCRouter,
  ResolvedMemoryView,
  TransferManager,
  TransferOp,
  TransferStatus,
)
from pipeline_validator.memory.allocator import ContextBufferOwner
from pipeline_validator.memory.arena import ArenaPool, RootInvocation
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.memory.cache import CacheLineIdentity, CacheProvenance
from pipeline_validator.profiles import (
  MaintenanceRange,
  MemoryMaintenanceDesc,
  ProfileReconfigDesc,
  SourceRef,
  build_registry,
)
from pipeline_validator.runtime.group_port import GroupPortAdapter
from pipeline_validator.simulator import Simulator
from pipeline_validator.tile_group import TileGroup
from pipeline_validator.workload_ir import parse_workload_ir
from pipeline_validator.workloads import PowWorkload

COPY_IR = """builtin.module {
  tile.program @copy_tile(%task : !nest.task, %buffer : !nest.l2_buffer<64xi8>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 1024> {
    %view = tile.subview %buffer offsets = [0] sizes = [64] strides = [1]
        : !nest.l2_view<64xi8>
    %local = tile.alloc shape = [64] dtype = "i8" alignment = 64
        : !tile.l1_buffer<64xi8>
    %loaded = tile.load.async %view into %local : !tile.event<"loaded">
    tile.await %loaded
    tile.signal input_released(%task)
    %stored = tile.store.async %local into %view : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @copy(
      %src : !nest.global_memref<64xi8>,
      %dst : !nest.global_memref<64xi8>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
          logical_tasks = 1, l2_spm_bytes = 1024, requested_contexts_per_tile = 1> {
    %buffer = nest.alloc slot = "buffer" role = "inout" shape = [64] dtype = "i8" alignment = 64
        : !nest.l2_buffer<64xi8>
    %src_view = nest.subview %src offsets = [0] sizes = [64] strides = [1]
        : !nest.global_view<64xi8>
    %dst_view = nest.subview %dst offsets = [0] sizes = [64] strides = [1]
        : !nest.global_view<64xi8>
    %prefetched = nest.dma.prefetch.async %src_view into %buffer : !nest.event<"prefetched">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %read, %ready = nest.dispatch.tasks.async @copy_tile l1_mode = 0
        tasks(%tasks) globals() bindings(%buffer) ins(%buffer) outs(%buffer)
        signal_policy {input_released = #nest.aggregate<all_tasks>,
                       output_ready = #nest.aggregate<all_tasks>}
        depends_on(%prefetched)
        : (!nest.event<"grid">, !nest.event<"read">, !nest.event<"ready">)
    %written = nest.dma.store.async %buffer into %dst_view depends_on(%ready)
        : !nest.event<"written">
    nest.release %buffer depends_on(%read, %prefetched, %written)
    nest.await %grid, %written
    nest.return
  }
}
"""


def _drain_initialization(group: TileGroup) -> int:
  cycle = 0
  while not group.profile_controller.initialized:
    group.profile_controller.step(cycle)
    cycle += 1
    assert cycle < 10000
  return cycle


def _descriptor(group: TileGroup, command_id: str, level: str, target: int):
  controller = group.profile_controller
  current = controller.active_modes[level]
  profile = group.registry.profile(level, target)
  return ProfileReconfigDesc(
    command_id=command_id,
    level=level,
    expected_mode=current,
    target_mode=target,
    registry_hash=group.registry.registry_hash,
    wait_instruction_ids=(),
    frontier=(),
    affected_domains=(level,),
    member_ids=profile.member_ids,
    exclusive_binding_id="",
    source_ref=SourceRef("<test>", "profile", 0, "profile.reconfig"),
  )


def _new_group(*, fidelity="full_memory", byte_store=None):
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  group = TileGroup(hw, fidelity=fidelity, byte_store=byte_store)
  cycle = _drain_initialization(group)
  group.run_generation = 1
  group.profile_controller.begin_run(1)
  group.transfer_manager.begin_run(1)
  return hw, group, cycle


def _step_command(group: TileGroup, command_id: str, cycle: int, limit=10000):
  controller = group.profile_controller
  for current in range(cycle, cycle + limit):
    group.step(current)
    if controller.status(command_id) in ("completed", "faulted", "cancelled"):
      return current
  raise AssertionError(f"profile command {command_id} did not terminate")


class TestMemberProtocol:
  def test_t20_last_prepare_and_commit_ack_each_hold_publication_and_gate(self):
    _hw, group, cycle = _new_group()
    controller = group.profile_controller
    command = _descriptor(group, "delayed", "l1", 1)
    last = command.member_ids[-1]
    controller.queue_member_response("delayed", last, "PREPARE", delay_cycles=40)
    controller.queue_member_response("delayed", last, "COMMIT", delay_cycles=40)
    assert controller.submit(command, "device@1", cycle)
    saw_ready_wait = saw_commit_wait = False
    for current in range(cycle, cycle + 2000):
      controller.step(current)
      step = controller.snapshot()["commands"]["delayed"]["step"]
      if step == "WAIT_READY_ACK":
        saw_ready_wait = True
        assert controller.active_modes["l1"] == 0
        assert controller.issue_gate_closed("l1")
      if step == "WAIT_COMMIT_ACK":
        saw_commit_wait = True
        assert controller.active_modes["l1"] == 0
        assert controller.generations["l1"] == 0
        assert controller.issue_gate_closed("l1")
      if controller.status("delayed") == "completed":
        break
    assert saw_ready_wait and saw_commit_wait
    assert controller.active_modes["l1"] == 1
    assert controller.generations["l1"] == 1
    assert not controller.issue_gate_closed("l1")

  def test_t21_failed_commit_ack_faults_domain_without_mixed_publication(self):
    _hw, group, cycle = _new_group()
    controller = group.profile_controller
    command = _descriptor(group, "failed", "l1", 1)
    failed_member = command.member_ids[len(command.member_ids) // 2]
    controller.queue_member_response(
      "failed", failed_member, "COMMIT", success=False, reason="injected member failure"
    )
    assert controller.submit(command, "device@1", cycle)
    _step_command(group, "failed", cycle)
    assert controller.status("failed") == "faulted"
    assert controller.active_modes["l1"] == 0
    assert controller.generations["l1"] == 0
    assert controller.issue_gate_closed("l1")
    assert "injected member failure" in controller.command_reason("failed")

  def test_t29_single_writer_backpressures_conflicting_device_control(self):
    hw = HardwareConfig()
    sim = SimConfig()
    workload = PowWorkload(hw=hw)
    artifact = compile_program(workload.module, hw, sim, workload_info=workload.info)
    group = TileGroup(hw, fidelity="runtime")
    _drain_initialization(group)
    group.begin_launch(artifact, {"Y": GlobalBinding("Y", 0x100000, 524288, "rw")})
    port = GroupPortAdapter(group, active_context_capacity=1)
    first = _descriptor(group, "first", "l1", 1)
    second = _descriptor(group, "second", "l1", 2)
    assert port.try_submit_control(DeviceControlRequest("first", first), 0)
    assert not port.try_submit_control(DeviceControlRequest("second", second), 0)

    completions = ()
    for cycle in range(2000):
      group.step(cycle)
      completions = port.poll_control_completions(cycle)
      if completions:
        break
    assert len(completions) == 1
    assert completions[0].command_id == "first"
    assert group.profile_controller.active_modes["l1"] == 1

  def test_t30_rv12_cancel_waits_for_issued_ack_isolation_and_never_reopens_gate(self):
    _hw, group, cycle = _new_group()
    controller = group.profile_controller
    command = _descriptor(group, "cancelled", "l1", 1)
    member = command.member_ids[-1]
    controller.queue_member_response("cancelled", member, "PREPARE", delay_cycles=80)
    assert controller.submit(command, "device@1", cycle)

    current = cycle
    while controller.snapshot()["commands"]["cancelled"]["step"] != "WAIT_READY_ACK":
      controller.step(current)
      current += 1
    controller.request_cancel("cancelled", current)
    for _ in range(20):
      controller.step(current)
      current += 1
    assert controller.status("cancelled") != "completed"
    assert controller.issue_gate_closed("l1")
    _step_command(group, "cancelled", current)
    assert controller.status("cancelled") == "cancelled"
    assert controller.active_modes["l1"] == 0
    assert controller.generations["l1"] == 0
    assert controller.issue_gate_closed("l1")

  def test_t31_rv14_l2_only_commit_changes_only_l2_generation(self):
    _hw, group, cycle = _new_group()
    controller = group.profile_controller
    command = _descriptor(group, "l2-only", "l2", 1)
    assert controller.submit(command, "device@1", cycle)
    _step_command(group, "l2-only", cycle)
    assert dict(controller.active_modes) == {"l1": 0, "l2": 1}
    assert dict(controller.generations) == {"l1": 0, "l2": 1}

  def test_rv13_registry_mismatch_and_initialization_proof_reuse_are_rejected(self):
    _hw, group, cycle = _new_group()
    controller = group.profile_controller
    bad = replace(_descriptor(group, "bad-registry", "l1", 1), registry_hash="f" * 64)
    with pytest.raises(MemoryInvariantError):
      controller.submit(bad, "device@1", cycle)
    with pytest.raises(MemoryInvariantError):
      controller.begin_run(1)


class TestProfileScenario:
  def test_rv01_rv04_rv07_rv11_rv14_profile_example_runs_with_parent_l2_persistence(self):
    from pathlib import Path

    from pipeline_validator.execution_ir import ExecGroupActionOp, ExecModel

    root = Path(__file__).resolve().parents[2]
    hw = HardwareConfig()
    sim_config = SimConfig(
      fidelity="full_memory", context_count=1, device_context_count=1, memory_trace=True, max_cycles=200000
    )
    module = parse_workload_ir((root / "examples/scenarios/profile_reconfiguration.mlir").read_text())
    artifact = compile_program(module, hw, sim_config)
    assert isinstance(artifact.entry, ExecModel)
    task_root = next(task for task in artifact.entry.tasks.values() if task.name == "ctx_tasks")
    dispatches = [
      action.args[0] for action in task_root.actions if action.op is ExecGroupActionOp.DISPATCH_ROLE
    ]
    assert [item.requested_l1_mode for item in dispatches] == [0, 2, 0]
    assert [item.resolved_l1_mode for item in dispatches] == [0, 2, 2]
    reconfig_index = next(
      index
      for index, action in enumerate(task_root.actions)
      if action.op is ExecGroupActionOp.PROFILE_RECONFIG
    )
    prior = task_root.actions[reconfig_index - 1]
    assert prior.op is ExecGroupActionOp.WAIT_EVENT
    assert prior.args
    assert task_root.resource_contract.requested_contexts_per_tile == 1

    simulator = Simulator(hw, sim_config, enable_tracer=True)
    result = simulator.run(load_program(artifact, hw, sim_config))
    assert result.completed, result.reason
    profile = result.group_snapshot["profile"]
    assert profile["active_modes"] == {"l1": 2, "l2": 2}
    assert profile["generations"] == {"l1": 1, "l2": 1}
    assert result.group_snapshot["task_leases"]["active"] == 0

    events = simulator.tracer._events
    reserve = next(
      event
      for event in events
      if event["name"] == "arena_reserve"
      and event["args"].get("space") == "l2"
      and event["args"].get("context_name") == "ctx_tasks"
    )
    retire = next(
      event
      for event in events
      if event["name"] == "arena_retire" and event["args"].get("arena_id") == reserve["args"]["arena_id"]
    )
    l1_commands = [
      event for event in events if event["name"] == "profile_command" and event["args"].get("level") == "l1"
    ]
    assert l1_commands
    assert reserve["ts"] <= min(event["ts"] for event in l1_commands)
    assert max(event["ts"] for event in l1_commands) <= retire["ts"]

    leases: dict[tuple[int, int], int] = {}
    for event in sorted(
      (event for event in events if event["name"] in ("task_lease_acquire", "task_lease_release")),
      key=lambda event: event["ts"],
    ):
      key = (event["pid"], event["args"]["launch_generation"])
      leases[key] = leases.get(key, 0) + (1 if event["name"].endswith("acquire") else -1)
      assert leases[key] in (0, 1)
    assert leases and set(leases.values()) == {0}


class TestByteOracle:
  def test_t17_t18_t23_rv08_inflight_refill_progresses_while_profile_gate_waits(self):
    oracle = ByteStore()
    hw, group, cycle = _new_group(byte_store=oracle)
    controller = group.profile_controller
    enable_cache = _descriptor(group, "enable-l1-cache", "l1", 1)
    assert controller.submit(enable_cache, "device@1", cycle)
    cycle = _step_command(group, enable_cache.command_id, cycle) + 1
    binding = GlobalBinding("refill", 0x400000, 64, "r")
    payload = bytes(range(64))
    oracle.seed_hbm(binding.base_iova, payload)
    handle = group.hbm.bind_external(binding, cycle)
    source = ResolvedMemoryView(handle, 0, 64, handle.base_address, handle.bank_segments, "r")
    identity = CacheLineIdentity(handle.allocation_id, handle.generation, 0)
    provenance = CacheProvenance(
      binding.name, handle.allocation_id, handle.generation, 0, 64
    )
    table = group.tiles[0].l1_mshr
    token = table.allocate("late", generation=1)
    refill = MemoryTransaction(
      "real-refill",
      TransferOp.GATHER_DIRECT_L1_REFILL,
      handle.owner,
      source,
      None,
      64,
      "refill-done",
      tile_id=0,
      run_generation=1,
      profile_generations=(("l1", 0, 1), ("l2", 0, 0)),
      bypass_levels=("l2",),
    )
    group.transfer_manager.submit(refill, cycle)
    # Trusted runtime receipt: kernel retirement is not a Fabric/MSHR barrier.
    controller.note_await("device@1", "wait-old-kernel", ("old-kernel-done",), cycle)
    switch = replace(
      _descriptor(group, "disable-l1-cache", "l1", 0),
      wait_instruction_ids=("wait-old-kernel",),
      frontier=("old-kernel-done",),
    )
    assert controller.submit(switch, "device@1", cycle)
    saw_closed_while_progressing = False
    retired = False
    for current in range(cycle, cycle + 10000):
      group.step(current)
      if controller.issue_gate_closed("l1") and refill.status not in (
        TransferStatus.DONE,
        TransferStatus.FAULTED,
      ):
        saw_closed_while_progressing = True
        assert controller.active_modes["l1"] == 1
      if refill.status is TransferStatus.DONE and not retired:
        assert controller.status(switch.command_id) != "completed"
        assert refill.captured_data == payload
        group.tiles[0].l1_cache.refill(
          "late", identity=identity, provenance=provenance, data=refill.captured_data, profile_generation=1
        )
        table.complete(token.token, generation=1)
        group.transfer_manager.acknowledge(refill.transaction_id, current)
        retired = True
      if controller.status(switch.command_id) == "completed":
        break
    assert saw_closed_while_progressing and retired
    assert controller.status(switch.command_id) == "completed"
    assert controller.active_modes["l1"] == 0
    assert controller.generations["l1"] == 2

  def test_t16_t19_dirty_l1_clean_finishes_downstream_before_commit(self):
    base = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    write_back_l1 = replace(base.memory_target.l1, reset_mode=1, cache_write_policy="write_back")
    hw = replace(base, memory_target=replace(base.memory_target, l1=write_back_l1))
    oracle = ByteStore()
    group = TileGroup(hw, fidelity="full_memory", byte_store=oracle)
    cycle = _drain_initialization(group)
    group.run_generation = 1
    group.profile_controller.begin_run(1)
    group.transfer_manager.begin_run(1)
    controller = group.profile_controller

    target = GlobalBinding("dirty-target", 0x500000, 64, "rw")
    old = b"\x00" * 64
    dirty = b"\xd1" * 64
    oracle.seed_hbm(target.base_iova, old)
    group.hbm.bind_external(target, cycle)
    oracle.seed_cache_line("l1", 0, target.name, 0, dirty, dirty=True)
    oracle.validate_seed_coverage()

    disable = _descriptor(group, "clean-disable", "l1", 0)
    assert controller.submit(disable, "device@1", cycle)
    saw_clean = False
    clean_completion = -1
    command_completion = -1
    for current in range(cycle, cycle + 10000):
      group.step(current)
      for transaction in group.transfer_manager._transactions.values():
        if transaction.op is TransferOp.CACHE_CLEAN_L1:
          saw_clean = True
      if clean_completion < 0 and oracle.read_hbm(target.base_iova, 64) == dirty:
        clean_completion = current
      if controller.status(disable.command_id) == "completed":
        command_completion = current
        break
    assert saw_clean
    assert clean_completion >= 0
    assert command_completion >= clean_completion
    assert oracle.read_hbm(target.base_iova, 64) == dirty
    assert controller.generations["l1"] == 1

  def test_t15_source_capture_precedes_destination_visibility_and_final_bytes_match(self, monkeypatch):
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim_config = SimConfig(fidelity="full_memory", memory_trace=True, max_cycles=100000)
    oracle = ByteStore()
    simulator = Simulator(hw, sim_config, enable_tracer=True, byte_store=oracle)
    source = bytes(range(64))
    old = b"\xa5" * 64
    oracle.seed_hbm(0x100000, source)
    oracle.seed_hbm(0x200000, old)
    bindings = {
      "src": GlobalBinding("src", 0x100000, 64, "r"),
      "dst": GlobalBinding("dst", 0x200000, 64, "w"),
    }
    observed_transient = False
    original_step = simulator.group.step

    def observing_step(cycle):
      nonlocal observed_transient
      done = original_step(cycle)
      for transaction in simulator.group.transfer_manager._transactions.values():
        if (
          transaction.op is TransferOp.GLOBAL_STORE
          and transaction.source_captured_cycle >= 0
          and transaction.destination_committed_cycle < 0
        ):
          observed_transient = True
          assert oracle.read_hbm(0x200000, 64) == old
      return done

    monkeypatch.setattr(simulator.group, "step", observing_step)
    artifact = compile_program(parse_workload_ir(COPY_IR), hw, sim_config)
    loaded = load_program(artifact, hw, sim_config, actual_bindings=bindings)
    result = simulator.run(loaded)
    assert result.completed, result.reason
    assert observed_transient
    assert oracle.read_hbm(0x200000, 64) == source

  def test_t22_old_profile_generation_return_faults_without_overwriting_new_destination(self):
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=5)
    oracle = ByteStore()
    hbm = HBMRegion(byte_store=oracle)
    src_handle = hbm.bind_external(GlobalBinding("src", 0x100000, 64, "r"))
    oracle.seed_hbm(0x100000, bytes(range(64)))

    profile = build_registry(hw).profile("l2", 0)
    pool = ArenaPool(profile)
    from pipeline_validator.execution_ir import ExecL2Buffer

    layout = layout_buffers((ExecL2Buffer("dst", (64,), "i8", "out", 1, 64, 64),), profile, 1024)
    plan = pool.plan_arena(RootInvocation("dst", 0), layout)
    assert not isinstance(plan, AdmissionFailure)
    arena = pool.commit_arena(plan, 0)
    dst_handle = pool.bind_view(arena, "dst", 0)
    src_view = ResolvedMemoryView(src_handle, 0, 64, 0x100000, src_handle.bank_segments, "r")
    dst_view = ResolvedMemoryView(dst_handle, 0, 64, dst_handle.base_address, dst_handle.bank_segments, "w")
    old = b"\x5a" * 64
    oracle.write_view(dst_view, old)
    generation = {"value": 0}

    def valid(transaction, phase):
      return generation["value"] == 0

    noc = NoCRouter(vc_depth=hw.noc_vc_depth, router_latency_cycles=hw.noc_router_latency_cycles)

    def acquire(transaction, cycle):
      for view in (transaction.src, transaction.dst):
        if view is not None and view.handle.memory_space == "l2":
          pool.begin_inflight(view.handle, transaction.transaction_id)

    def release(transaction, cycle):
      for view in (transaction.src, transaction.dst):
        if view is not None and view.handle.memory_space == "l2":
          pool.end_inflight(view.handle, transaction.transaction_id, cycle)

    manager = TransferManager(
      hw,
      full_memory=True,
      noc=noc,
      byte_store=oracle,
      generation_validator=valid,
      reference_acquire=acquire,
      reference_release=release,
    )
    manager.configure_profile_generations({("l2", 0): 0})
    manager.begin_run(1)
    transaction = MemoryTransaction(
      "old-return",
      TransferOp.PREFETCH,
      dst_handle.owner,
      src_view,
      dst_view,
      64,
      "done",
      run_generation=1,
      profile_generations=(("l2", 0, 0),),
    )
    manager.submit(transaction, 0)
    for cycle in range(1000):
      traversed = noc.step(cycle)
      manager.note_traversed(traversed, cycle)
      manager.step(cycle)
      if transaction.source_captured_cycle >= 0:
        generation["value"] = 1
      if transaction.status in (TransferStatus.FAULTED, TransferStatus.DONE):
        break
    assert transaction.status is TransferStatus.FAULTED
    assert oracle.read_view(dst_view) == old

    # plan/01 §6.3: a FAULTED-but-unacknowledged transaction keeps the
    # backing referenced; terminal acknowledgement releases the reference
    # and the padded capacity returns.
    committed_free = pool.snapshot()["free_bytes"]
    assert not pool.invalidate_view(dst_handle, dst_handle.owner, 900)
    held = pool.snapshot()
    assert held["live_backings"] == 1
    assert held["free_bytes"] == committed_free
    assert held["pin_count"] == 0
    assert held["inflight_count"] == 1
    manager.acknowledge("old-return", 950)
    after = pool.snapshot()
    assert after["live_backings"] == 0
    assert after["arena_reserved_bytes"] == 0
    assert after["free_bytes"] == after["user_spm_capacity_bytes"]

  def test_t25_range_maintenance_invalidates_stale_cache_bytes_before_consumer(self):
    oracle = ByteStore()
    hw, group, cycle = _new_group(byte_store=oracle)
    controller = group.profile_controller
    enable_cache = _descriptor(group, "enable-l2-cache", "l2", 1)
    assert controller.submit(enable_cache, "device@1", cycle)
    cycle = _step_command(group, enable_cache.command_id, cycle) + 1

    target = GlobalBinding("target", 0x300000, 64, "rw")
    new = bytes(reversed(range(64)))
    old = b"\x11" * 64
    oracle.seed_hbm(target.base_iova, new)
    handle = group.hbm.bind_external(target, 0)
    oracle.seed_cache_line("l2", 0, "target", 0, old)
    oracle.validate_seed_coverage()
    identity = CacheLineIdentity(handle.allocation_id, handle.generation, 0)
    cache = group.l2_cache
    assert cache.read_line(identity) == old

    controller.bind_owner_inputs("device@1", (handle,))
    command = MemoryMaintenanceDesc(
      command_id="invalidate-target",
      levels=("l2",),
      ranges=(MaintenanceRange(0, 0, 64, ("l2",)),),
      dependencies=(),
      source_ref=SourceRef("<test>", "maintenance", 0, "memory.maintenance"),
    )
    assert controller.submit(command, "device@1", cycle)
    _step_command(group, command.command_id, cycle)
    assert controller.status(command.command_id) == "completed"
    assert cache.snapshot()["resident_lines"] == 0
    assert oracle.read_hbm(target.base_iova, 64) == new


class TestProfileAffinity:
  @pytest.mark.parametrize("level", ["l2", "l1"], ids=["root-buckets", "task-buckets"])
  def test_t12_blocked_same_head_allows_compatible_but_not_smaller_same(self, level):
    import json

    from pipeline_validator.loader import load_program
    from pipeline_validator.profiles import ProfileBytes, ProfileLevelSource

    base = HardwareConfig()
    target = replace(
      base.memory_target,
      **{
        level: replace(getattr(base.memory_target, level), reset_mode=2, system_reserved_spm_per_bank=4096)
      },
    )
    source = replace(
      base.profile_source,
      **{level: ProfileLevelSource({0: ProfileBytes(32768, 0), 2: ProfileBytes(24576, 8192)})},
    )
    geometry = (
      {"group_sram_bytes": 131072, "group_sram_banks": 4}
      if level == "l2"
      else {"tile_l1_bytes": 131072, "tile_l1_banks": 4}
    )
    hw = replace(
      base, memory_target=target, profile_source=source, hbm_fixed_latency_cycles=100, **geometry
    )
    cases = (
      ("hold", 2, 49152, (2,)),
      ("large", 2, 40960, (2,)),
      ("small_same", 2, 8192, (2,)),
      ("small_compatible", 0, 16384, (0, 2)),
    )
    definitions = []
    submits = []
    for name, mode, reserve, allowed in cases:
      allowed_text = ", ".join(map(str, allowed))
      if level == "l2":
        definitions.append(f"""
          nest.context @{name} (%x : !nest.global_memref<64xi8>) placement = 1
            resource_contract = #nest.context_resources<l2_mode = {mode},
              allowed_profiles = [{allowed_text}], logical_tasks = 0,
              l2_spm_bytes = {reserve}, requested_contexts_per_tile = 1> {{
            %b = nest.alloc slot = "buf" role = "in" shape = [64] dtype = "i8"
              alignment = 64 : !nest.l2_buffer<64xi8>
            %v = nest.subview %x offsets = [0] sizes = [64] strides = [1] : !nest.global_view<64xi8>
            %p = nest.dma.prefetch.async %v into %b : !nest.event<"p">
            nest.await %p
            nest.release %b depends_on(%p)
            nest.return
          }}
        """)
        submits.append(f'%{name} = nexus.submit_context.async @{name}(%X) : !nexus.event<"{name}">')
      else:
        ops = 320000 if name == "hold" else 32
        definitions.append(f"""
          tile.program @task_{name} (%t : !nest.task)
            resource_contract = #tile.resources<allowed_profiles = [{allowed_text}],
              tile_l1_spm_bytes_per_context = {reserve}> {{
            %b = tile.alloc shape = [{reserve}] dtype = "i8" alignment = 64 : !tile.l1_buffer<{reserve}xi8>
            %e = tile.evu.async "relu" ops = {ops} : !tile.event<"e">
            tile.await %e
            tile.return
          }}
          nest.context @{name} placement = 1
            resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
              logical_tasks = 1, l2_spm_bytes = 0, requested_contexts_per_tile = 1> {{
            %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
            %g, %r, %w = nest.dispatch.tasks.async @task_{name} l1_mode = {mode}
              tasks(%tasks) globals() bindings() ins() outs() signal_policy {{}}
              : (!nest.event<"g">, !nest.event<"">, !nest.event<"">)
            nest.await %g
            nest.return
          }}
        """)
        submits.append(f'%{name} = nexus.submit_context.async @{name} : !nexus.event<"{name}">')
    arguments = "(%X : !nest.global_memref<64xi8>)" if level == "l2" else ""
    text = (
      "builtin.module {"
      + "\n".join(definitions)
      + f"nexus.program @affinity {arguments} {{"
      + "\n".join(submits)
      + "nexus.await %hold, %large, %small_same, %small_compatible\nnexus.return\n}}"
    )
    config = SimConfig(context_count=4, device_context_count=4, memory_trace=True, max_cycles=30000)
    program = compile_program(parse_workload_ir(text), hw, config)
    bindings = {"X": GlobalBinding("X", 0x1000, 64, "r")} if level == "l2" else {}
    simulator = Simulator(hw, config, enable_tracer=True)
    result = simulator.run(load_program(program, hw, config, actual_bindings=bindings))
    assert result.completed, result.reason
    if level == "l2":
      records = {record["context"]: record for record in result.device_snapshot["port"]["request_records"]}
      starts = {name: record["active_cycle"] for name, record in records.items()}
      assert starts["large"] >= records["hold"]["completion_cycle"]
    else:
      events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
      acquisitions = [event for event in events if event.get("name") == "task_lease_acquire"]
      starts = {
        program.call_bindings[event["args"]["binding_id"]].context_name: event["ts"]
        for event in acquisitions
      }
      hold_release = next(
        event["ts"]
        for event in events
        if event.get("name") == "task_lease_release"
        and program.call_bindings[event["args"]["binding_id"]].context_name == "hold"
      )
      assert starts["large"] >= hold_release
    assert starts["hold"] < starts["small_compatible"] < starts["large"]
    assert starts["small_same"] >= starts["large"]
    assert simulator.group.profile_controller.generations == {"l1": 0, "l2": 0}
    assert not simulator.group._grid_routes and not simulator.group._task_leases


COHERENCE_IR = """builtin.module {
  tile.program @lookup(%t : !nest.task, %table : !nest.global_view<64xi8>,
      %idx : !nest.l2_buffer<16xi32>, %output : !nest.l2_buffer<64xi8>)
      resource_contract = #tile.resources<allowed_profiles = [1], tile_l1_spm_bytes_per_context = 2048,
        l1_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 2000000},
        l2_cache = {required = false, access = "read", bypass = "allowed", target_bytes = 2000000}> {
    %iv = tile.subview %idx offsets = [0] sizes = [16] strides = [1] : !nest.l2_view<16xi32>
    %ov = tile.subview %output offsets = [0] sizes = [64] strides = [1] : !nest.l2_view<64xi8>
    %li = tile.alloc shape = [16] dtype = "i32" : !tile.l1_buffer<16xi32>
    %lv = tile.alloc shape = [64] dtype = "i8" : !tile.l1_buffer<64xi8>
    %ld = tile.load.async %iv into %li : !tile.event<"indices">
    tile.await %ld
    tile.signal input_released(%t)
    %g = tile.gather.global.async %table indices(%li) into %lv
        map = #tile.indexed_map<index_scale = 16 offset = 0 task_stride = 0 repeat = 1 stride = 0 segment = 4>
        window_entries = 1 : !tile.event<"gather">
    tile.await %g
    %st = tile.store.async %lv into %ov : !tile.event<"stored">
    tile.await %st
    tile.signal output_ready(%t)
    tile.return
  }
  nest.context @producer(%fresh : !nest.global_memref<64xi8>, %table : !nest.global_memref<64xi8>)
      placement = 1 resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [1],
        logical_tasks = 0, l2_spm_bytes = 1024, requested_contexts_per_tile = 1> {
    %b = nest.alloc slot = "copy" role = "in" shape = [64] dtype = "i8" : !nest.l2_buffer<64xi8>
    %f = nest.subview %fresh offsets = [0] sizes = [64] strides = [1] : !nest.global_view<64xi8>
    %v = nest.subview %table offsets = [0] sizes = [64] strides = [1] : !nest.global_view<64xi8>
    %p = nest.dma.prefetch.async %f into %b : !nest.event<"p">
    %s = nest.dma.store.async %b into %v depends_on(%p) : !nest.event<"s">
    nest.await %s
    nest.release %b depends_on(%p, %s)
    nest.return
  }
  nest.context @consumer(%table : !nest.global_memref<64xi8>, %indices : !nest.global_memref<16xi32>,
      %output : !nest.global_memref<64xi8>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [1],
        logical_tasks = 1, l2_spm_bytes = 2048, requested_contexts_per_tile = 1> {
    %i = nest.alloc slot = "indices" role = "in" shape = [16] dtype = "i32" : !nest.l2_buffer<16xi32>
    %o = nest.alloc slot = "output" role = "out" shape = [64] dtype = "i8" : !nest.l2_buffer<64xi8>
    %tv = nest.subview %table offsets = [0] sizes = [64] strides = [1] : !nest.global_view<64xi8>
    %iv = nest.subview %indices offsets = [0] sizes = [16] strides = [1] : !nest.global_view<16xi32>
    %ov = nest.subview %output offsets = [0] sizes = [64] strides = [1] : !nest.global_view<64xi8>
    %p = nest.dma.prefetch.async %iv into %i : !nest.event<"p">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %g, %r, %w = nest.dispatch.tasks.async @lookup l1_mode = 1
        tasks(%tasks) globals(%tv) bindings(%i, %o) ins(%i) outs(%o)
        signal_policy {input_released = #nest.aggregate<all_tasks>,
                       output_ready = #nest.aggregate<all_tasks>}
        depends_on(%p) : (!nest.event<"g">, !nest.event<"r">, !nest.event<"w">)
    %s = nest.dma.store.async %o into %ov depends_on(%w) : !nest.event<"s">
    nest.await %g, %s
    nest.release %i depends_on(%p, %r)
    nest.release %o depends_on(%s)
    nest.return
  }
  nexus.program @coherence(%fresh : !nest.global_memref<64xi8>, %table : !nest.global_memref<64xi8>,
      %indices : !nest.global_memref<16xi32>, %output : !nest.global_memref<64xi8>) {
    %p = nexus.submit_context.async @producer(%fresh, %table) : !nexus.event<"p">
    %c = nexus.submit_context.async @consumer(%table, %indices, %output)
        depends_on(%p) : !nexus.event<"c">
    nexus.await %c
    nexus.return
  }
}"""


class TestCompiledCoherence:
  def test_t24_t25_real_dma_and_compiled_maintenance_publish_new_gather_bytes(self):
    from pipeline_validator.compiled_program import seal_program
    from pipeline_validator.memory.cache import CacheLineIdentity

    base = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    hw = replace(
      base,
      memory_target=replace(
        base.memory_target,
        l1=replace(base.memory_target.l1, reset_mode=1),
        l2=replace(base.memory_target.l2, reset_mode=1),
      ),
    )
    config = SimConfig(max_cycles=10000, fidelity="full_memory")
    module = parse_workload_ir(COHERENCE_IR)
    program = compile_program(module, hw, config)
    controls = [op for op in program.entry.body if op.op == "memory_maintenance"]
    assert controls
    new, old = bytes(range(64)), b"\x11" * 64
    stale_during_write = []

    class ObservedBytes(ByteStore):
      def write_view(self, view, data):
        super().write_view(view, data)
        if view.handle.memory_space == "hbm" and view.handle.owner.binding_name == "table":
          identity = CacheLineIdentity(view.handle.allocation_id, view.handle.generation, 0)
          stale_during_write.append(
            (
              simulator.group.tiles[0].l1_cache.read_line(identity),
              simulator.group.l2_cache.read_line(identity),
            )
          )

    oracle = ObservedBytes()
    oracle.seed_hbm(0x1000, new)
    oracle.seed_hbm(0x2000, old)
    # 16 index slots reading 4 B each; four distinct rows (0,1,2,3) repeated
    # so the same cache line is merged by the MSHRs.
    oracle.seed_hbm(0x3000, b"".join(int(row).to_bytes(4, "little") for row in [0, 1, 2, 3] * 4))
    oracle.seed_hbm(0x4000, b"\xee" * 64)
    oracle.seed_cache_line("l1", 0, "table", 0, old)
    oracle.seed_cache_line("l2", 0, "table", 0, old)
    bindings = {
      "fresh": GlobalBinding("fresh", 0x1000, 64, "r"),
      "table": GlobalBinding("table", 0x2000, 64, "rw"),
      "indices": GlobalBinding("indices", 0x3000, 64, "r"),
      "output": GlobalBinding("output", 0x4000, 64, "w"),
    }
    simulator = Simulator(hw, config, byte_store=oracle)
    result = simulator.run(load_program(program, hw, config, actual_bindings=bindings))
    assert result.completed, result.reason
    assert stale_during_write == [(old, old)]
    # The DMA refreshed HBM and the maintenance made the new bytes visible.
    assert oracle.read_hbm(0x2000, 64) == new
    # The Gather then read table[index*16 : +4] per index slot, so OUT holds
    # bytes 0..3, 16..19, 32..35, 48..51 repeated four times.
    expected_out = b"".join(new[row * 16 : row * 16 + 4] for row in [0, 1, 2, 3] * 4)
    assert oracle.read_hbm(0x4000, 64) == expected_out
    assert simulator.group.profile_controller.generations == {"l1": 0, "l2": 0}

    removed_ids = {op.instruction_id for op in controls}
    corrupted = seal_program(
      replace(
        program,
        entry=replace(program.entry, body=tuple(op for op in program.entry.body if op not in controls)),
        source_map={key: value for key, value in program.source_map.items() if key not in removed_ids},
      )
    )
    with pytest.raises(ValueError):
      load_program(corrupted, hw, config, actual_bindings=bindings)
    no_maintenance = replace(
      hw,
      memory_target=replace(
        hw.memory_target, l1=replace(hw.memory_target.l1, maintenance_caps=("bypass",))
      ),
    )
    with pytest.raises(ValueError):
      compile_program(module, no_maintenance, config)


class TestBoundedMaintenance:
  @staticmethod
  def dirty_group(*, timeout: int, latency: int):
    base = HardwareConfig().with_overrides(
      hbm_fixed_latency_cycles=latency, hbm_outstanding_limit=1, noc_vc_depth=1
    )
    hw = replace(
      base,
      memory_target=replace(
        base.memory_target,
        l1=replace(base.memory_target.l1, reset_mode=1, cache_write_policy="write_back"),
        profile_command_timeout_cycles=timeout,
      ),
    )
    oracle = ByteStore()
    group = TileGroup(hw, fidelity="full_memory", byte_store=oracle)
    cycle = _drain_initialization(group)
    group.run_generation = 1
    group.profile_controller.begin_run(1)
    group.transfer_manager.begin_run(1)
    binding = GlobalBinding("dirty", 0x600000, 512, "rw")
    oracle.seed_hbm(binding.base_iova, bytes(512))
    group.hbm.bind_external(binding, cycle)
    payload = b"".join(bytes([index + 1]) * 64 for index in range(8))
    for index in range(8):
      oracle.seed_cache_line(
        "l1", 0, binding.name, index * 64, payload[index * 64 : (index + 1) * 64], dirty=True
      )
    oracle.validate_seed_coverage()
    return group, oracle, binding, payload, cycle

  def test_t19_t23_dirty_work_larger_than_control_window_backpressures_and_finishes(self):
    group, oracle, binding, payload, cycle = self.dirty_group(timeout=10000, latency=10)
    controller = group.profile_controller
    command = _descriptor(group, "bounded-clean", "l1", 0)
    assert group.transfer_manager.tombstone_capacity < 8
    assert controller.submit(command, "device@1", cycle)
    peak = 0
    for current in range(cycle, cycle + 10000):
      group.step(current)
      peak = max(peak, group.transfer_manager.snapshot()["retained_transactions"])
      if controller.status(command.command_id) in ("completed", "faulted"):
        break
    assert controller.status(command.command_id) == "completed"
    assert 0 < peak <= group.transfer_manager.tombstone_capacity
    assert oracle.read_hbm(binding.base_iova, binding.size_bytes) == payload
    assert group.transfer_manager.snapshot()["retained_transactions"] == 0

  def test_t30_faulted_clean_drains_before_explicit_destructive_recovery(self):
    group, oracle, binding, _payload, cycle = self.dirty_group(timeout=500, latency=1000)
    controller = group.profile_controller
    command = _descriptor(group, "timed-out-clean", "l1", 0)
    assert controller.submit(command, "device@1", cycle)
    for current in range(cycle, cycle + 1000):
      group.step(current)
      if controller.status(command.command_id) == "faulted":
        break
    state = controller.snapshot()["commands"][command.command_id]
    assert state["completed_cycle"] - state["accepted_cycle"] == 500
    assert controller.cancellation_pending
    with pytest.raises(MemoryInvariantError):
      controller.recover(current)
    with pytest.raises(MemoryInvariantError):
      controller.begin_run(2)
    fault_tick = current
    for current in range(fault_tick + 1, fault_tick + 3000):
      group.step(current)
      assert controller.issue_gate_closed("l1")
      if not controller.cancellation_pending:
        break
    assert not controller.cancellation_pending
    assert group.transfer_manager.snapshot()["retained_transactions"] == 0
    assert controller.status(command.command_id) == "faulted"
    recovery_tick = current + 1
    controller.recover(recovery_tick)
    for current in range(recovery_tick, recovery_tick + 1000):
      group.step(current)
      if controller.initialized:
        break
    assert controller.initialized
    assert not controller.issue_gate_closed("l1")
    assert controller.active_modes == {"l1": 1, "l2": 0}
    assert group.tiles[0].l1_cache.stats.resident_lines == 0
    assert group.tiles[0].l1_cache.stats.discarded_dirty_lines > 0
    # Recovery is not a fabricated writeback or a successful workload result.
    assert oracle.read_hbm(binding.base_iova + 7 * 64, 64) == bytes(64)


_IO_TILE_BODY = """%view = tile.subview %buffer offsets = [0] sizes = [64] strides = [1]
        : !nest.l2_view<64xi8>
    %local = tile.alloc shape = [64] dtype = "i8" alignment = 64
        : !tile.l1_buffer<64xi8>
    %loaded = tile.load.async %view into %local : !tile.event<"loaded">
    tile.await %loaded
    tile.signal input_released(%task)
    %stored = tile.store.async %local into %view : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return"""

RV04_IR = (
  'builtin.module {\n  tile.program @io_tile_lo(%task : !nest.task, %buffer : !nest.l2_buffer<64xi8>)\n'
  '      resource_contract = #tile.resources<allowed_profiles = [0],\n'
  "          tile_l1_spm_bytes_per_context = 1024> {\n    "
  + _IO_TILE_BODY
  + "\n  }\n"
  + '  tile.program @io_tile_hi(%task : !nest.task, %buffer : !nest.l2_buffer<64xi8>)\n'
  '      resource_contract = #tile.resources<allowed_profiles = [2],\n'
  "          tile_l1_spm_bytes_per_context = 1024> {\n    "
  + _IO_TILE_BODY
  + """
  }
  nest.context @rv04(
      %src : !nest.global_memref<64xi8>,
      %dst : !nest.global_memref<64xi8>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 2, l2_spm_bytes = 1024, requested_contexts_per_tile = 1> {
    %buffer = nest.alloc slot = "buffer" role = "inout" shape = [64] dtype = "i8" alignment = 64
        : !nest.l2_buffer<64xi8>
    %src_view = nest.subview %src offsets = [0] sizes = [64] strides = [1]
        : !nest.global_view<64xi8>
    %dst_view = nest.subview %dst offsets = [0] sizes = [64] strides = [1]
        : !nest.global_view<64xi8>
    %prefetched = nest.dma.prefetch.async %src_view into %buffer : !nest.event<"prefetched">
    %range_one = nest.task.range from = 0 to = 1 : !nest.task_range
    %g1, %r1, %w1 = nest.dispatch.tasks.async @io_tile_lo l1_mode = 0
        tasks(%range_one) globals() bindings(%buffer) ins(%buffer) outs(%buffer)
        signal_policy {input_released = #nest.aggregate<all_tasks>,
                       output_ready = #nest.aggregate<all_tasks>}
        depends_on(%prefetched)
        : (!nest.event<"g1">, !nest.event<"r1">, !nest.event<"w1">)
    nest.await %g1
    %range_two = nest.task.range from = 0 to = 1 : !nest.task_range
    %g2, %r2, %w2 = nest.dispatch.tasks.async @io_tile_hi l1_mode = 2
        tasks(%range_two) globals() bindings(%buffer) ins(%buffer) outs(%buffer)
        signal_policy {input_released = #nest.aggregate<all_tasks>,
                       output_ready = #nest.aggregate<all_tasks>}
        : (!nest.event<"g2">, !nest.event<"r2">, !nest.event<"w2">)
    %written = nest.dma.store.async %buffer into %dst_view depends_on(%w1, %w2)
        : !nest.event<"written">
    nest.release %buffer depends_on(%r1, %r2, %prefetched, %written)
    nest.await %g2, %written
    nest.return
  }
}
"""
)


class TestProfileBoundaryBytes:
  def test_rv04_l1_switch_preserves_parent_l2_bytes(self):
    """Parent L2 data survives an in-Context L1 reconfiguration byte-for-byte.

    The source bytes round-trip src -> L2 -> (L1@mode0) -> L2 -> [L1 reconfig]
    -> (L1@mode2) -> L2 -> dst.  Any L1 switch that invalidated the parent L2
    Arena or its contents corrupts the final HBM bytes.
    """
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim_config = SimConfig(fidelity="full_memory", memory_trace=True, max_cycles=200000)
    oracle = ByteStore()
    simulator = Simulator(hw, sim_config, enable_tracer=True, byte_store=oracle)
    payload = bytes(range(64))
    oracle.seed_hbm(0x100000, payload)
    oracle.seed_hbm(0x200000, b"\xa5" * 64)
    bindings = {
      "src": GlobalBinding("src", 0x100000, 64, "r"),
      "dst": GlobalBinding("dst", 0x200000, 64, "w"),
    }
    artifact = compile_program(parse_workload_ir(RV04_IR), hw, sim_config)
    task = artifact.entry
    modes = [
      action.args[0].requested_l1_mode
      for action in task.actions
      if action.op.value == "dispatch.role"
    ]
    assert modes == [0, 2]
    reconfigs = [action for action in task.actions if action.op.value == "profile.reconfig"]
    assert len(reconfigs) == 1 and reconfigs[0].args[0].level == "l1"
    result = simulator.run(load_program(artifact, hw, sim_config, actual_bindings=bindings))
    assert result.completed, result.reason
    assert result.group_snapshot["profile"]["generations"] == {"l1": 1, "l2": 0}
    assert oracle.read_hbm(0x200000, 64) == payload
    l2_arenas = simulator.group.snapshot()["memory"]["l2"]
    assert l2_arenas["live_arenas"] == 0

  def test_rv13_wrong_entry_mode_rejected_until_explicit_recovery(self):
    from pipeline_validator.tests.test_compiler import COMPATIBLE_IR

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim_config = SimConfig(max_cycles=200000)
    artifact = compile_program(parse_workload_ir(COMPATIBLE_IR), hw, sim_config)
    simulator = Simulator(hw, sim_config)
    loaded = load_program(artifact, hw, sim_config)
    result = simulator.run(loaded)
    assert result.completed, result.reason
    assert simulator.group.profile_controller.active_modes["l2"] == 2
    with pytest.raises(ValueError, match="entry profile"):
      simulator.run(loaded)
    simulator.group.reset()
    recovered = simulator.run(loaded)
    assert recovered.completed, recovered.reason

  def test_t31_l2_only_commit_waits_for_inflight_l1_touching_work(self):
    """An L2-only reconfiguration's Commit waits for L1-touching traffic.

    A refill with L1 profile generations participates in the L2 closure; the
    L2 Commit ACK must not publish while it is still in flight, and the L1
    mode/generation must stay untouched.
    """
    oracle = ByteStore()
    _hw, group, cycle = _new_group(byte_store=oracle)
    controller = group.profile_controller
    binding = GlobalBinding("closure", 0x400000, 64, "r")
    payload = bytes(range(64))
    oracle.seed_hbm(binding.base_iova, payload)
    handle = group.hbm.bind_external(binding, cycle)
    source = ResolvedMemoryView(handle, 0, 64, handle.base_address, handle.bank_segments, "r")
    refill = MemoryTransaction(
      "l1-touching-refill",
      TransferOp.GATHER_DIRECT_L1_REFILL,
      handle.owner,
      source,
      None,
      64,
      "closure-done",
      tile_id=0,
      run_generation=1,
      profile_generations=(("l1", 0, 0), ("l2", 0, 0)),
      bypass_levels=("l2",),
    )
    group.transfer_manager.submit(refill, cycle)
    l1_before = (controller.active_modes["l1"], controller.generations["l1"])
    controller.note_await("device@1", "wait-old-work", ("old-work-done",), cycle)
    switch = replace(
      _descriptor(group, "l2-only-switch", "l2", 1),
      wait_instruction_ids=("wait-old-work",),
      frontier=("old-work-done",),
    )
    assert controller.submit(switch, "device@1", cycle)
    saw_blocked_commit = False
    acknowledged = False
    for current in range(cycle, cycle + 10000):
      group.step(current)
      if refill.status not in (TransferStatus.DONE, TransferStatus.FAULTED):
        if controller.status(switch.command_id) != "completed":
          saw_blocked_commit = True
      if refill.status is TransferStatus.DONE and not acknowledged:
        acknowledged = True
        group.transfer_manager.acknowledge(refill.transaction_id, current)
      if controller.status(switch.command_id) == "completed":
        assert refill.status is TransferStatus.DONE
        break
    assert saw_blocked_commit
    assert controller.status(switch.command_id) == "completed"
    assert (controller.active_modes["l1"], controller.generations["l1"]) == l1_before
    assert controller.generations["l2"] == 1


class TestLeaseRetirementOrdering:
  def test_rv11_lease_returns_only_at_retirement_not_at_release(self):
    """R leases are held across releases/signals and returned at retirement.

    The in-Context L1 switch serializes g1 retirement before g2 admission:
    g2's lease acquisition must follow the L1 profile command, which itself
    waits for g1's grid_done — never g1's input_released or buffer release.
    """
    from pathlib import Path

    from pipeline_validator.execution_ir import ExecModel

    root = Path(__file__).resolve().parents[2]
    hw = HardwareConfig()
    sim_config = SimConfig(
      fidelity="full_memory", context_count=1, device_context_count=1, memory_trace=True, max_cycles=200000
    )
    module = parse_workload_ir((root / "examples/scenarios/profile_reconfiguration.mlir").read_text())
    artifact = compile_program(module, hw, sim_config)
    assert isinstance(artifact.entry, ExecModel)
    simulator = Simulator(hw, sim_config, enable_tracer=True)
    result = simulator.run(load_program(artifact, hw, sim_config))
    assert result.completed, result.reason
    events = simulator.tracer._events
    l1_command = next(
      event
      for event in events
      if event["name"] == "profile_command" and event["args"].get("level") == "l1"
    )
    leases = sorted(
      (event for event in events if event["name"] in ("task_lease_acquire", "task_lease_release")),
      key=lambda event: event["ts"],
    )
    assert len(leases) == 6
    acquires = [event for event in leases if event["name"] == "task_lease_acquire"]
    releases = [event for event in leases if event["name"] == "task_lease_release"]
    assert len(acquires) == len(releases) == 3
    # g2 (first L1 mode-2 task) is admitted only after the L1 command, which
    # is ordered after g1's retirement — its lease was not returned early by
    # g1's input_released or any buffer release.
    mode2_acquire = next(event for event in acquires if event["args"].get("resolved_l1_mode") == 2)
    assert mode2_acquire["ts"] >= l1_command["ts"] + l1_command["dur"]
    per_key: dict[tuple, list[str]] = {}
    for event in leases:
      key = (event["pid"], event["args"]["launch_generation"])
      per_key.setdefault(key, []).append(event["name"].rsplit("_", 1)[-1])
    for sequence in per_key.values():
      assert sequence == ["acquire", "release"] * 3 or sequence == sorted(
        sequence, key=lambda name: name != "acquire"
      )
    counts = [sequence.count("acquire") - sequence.count("release") for sequence in per_key.values()]
    assert set(counts) == {0}


class TestControllerDefenses:
  @staticmethod
  def _maintenance(group: TileGroup, command_id: str, levels=("l1",), bytes_=64, dependencies=()):
    return MemoryMaintenanceDesc(
      command_id,
      tuple(levels),
      (MaintenanceRange(0, 0, bytes_, tuple(levels)),),
      tuple(dependencies),
      SourceRef("<test>", "maintenance", 0, "memory.maintenance"),
    )

  def test_cancelled_maintenance_releases_range_gates_and_allows_new_commands(self):
    group, _oracle, binding, _payload, cycle = self._dirty()
    controller = group.profile_controller
    controller.bind_owner_inputs("device@1", (group.hbm.get_handle(binding.name),))
    command = self._maintenance(group, "cancel-me", bytes_=binding.size_bytes)
    assert controller.submit(command, "device@1", cycle)
    gate_seen = False
    for current in range(cycle, cycle + 1000):
      group.step(current)
      if controller._range_gates:
        gate_seen = True
        break
    assert gate_seen
    controller.request_cancel(command.command_id, current)
    for current in range(current, current + 10000):
      group.step(current)
      if controller.status(command.command_id) == "cancelled":
        break
    assert controller.status(command.command_id) == "cancelled"
    assert controller._range_gates == {}
    # A fresh maintenance command is admitted without explicit recovery,
    # which proves the cancelled command holds no recovery latch.
    followup = self._maintenance(group, "followup", bytes_=binding.size_bytes)
    assert controller.submit(followup, "device@1", current)
    for current in range(current, current + 10000):
      group.step(current)
      if controller.status(followup.command_id) in ("completed", "faulted"):
        break
    assert controller.status(followup.command_id) == "completed"

  def test_device_await_proof_consumed_once_for_maintenance_dependencies(self):
    group, _oracle, binding, _payload, cycle = self._dirty(timeout=200)
    controller = group.profile_controller
    controller.bind_owner_inputs("device@1", (group.hbm.get_handle(binding.name),))
    controller.note_await("device@1", "single-proof", ("old-work",), cycle)
    first = self._maintenance(group, "first-clean", dependencies=("old-work",))
    assert controller.submit(first, "device@1", cycle)
    for current in range(cycle, cycle + 10000):
      group.step(current)
      if controller.status(first.command_id) in ("completed", "faulted"):
        break
    assert controller.status(first.command_id) == "completed"
    # The same await proof cannot authorize a second command: the fallback
    # consumer marks it used, so this command stalls in WAIT_DEPENDENCIES
    # until its own timeout faults it instead of completing.
    second = self._maintenance(group, "second-clean", dependencies=("old-work",))
    assert controller.submit(second, "device@1", current)
    for current in range(current, current + 10000):
      group.step(current)
      if controller.status(second.command_id) in ("completed", "faulted"):
        break
    assert controller.status(second.command_id) == "faulted"
    assert "timeout" in controller.snapshot()["commands"][second.command_id]["reason"]

  def test_cancel_requested_command_converges_to_cancelled_past_timeout(self):
    group, _oracle, binding, _payload, cycle = self._dirty(timeout=50, latency=1000)
    controller = group.profile_controller
    controller.bind_owner_inputs("device@1", (group.hbm.get_handle(binding.name),))
    command = self._maintenance(group, "slow-clean", bytes_=binding.size_bytes)
    assert controller.submit(command, "device@1", cycle)
    hbm_leg_active = False
    for current in range(cycle, cycle + 2000):
      group.step(current)
      for transaction in group.transfer_manager._transactions.values():
        if (
          transaction.current_leg < len(transaction.legs)
          and "HBM" in transaction.legs[transaction.current_leg].kind.name
          and transaction.leg_start_cycle >= 0
          and transaction.status is TransferStatus.RUNNING
        ):
          hbm_leg_active = True
          break
      if hbm_leg_active:
        break
    assert hbm_leg_active
    controller.request_cancel(command.command_id, current)
    terminal = ""
    for current in range(current, current + 20000):
      group.step(current)
      terminal = controller.status(command.command_id)
      if terminal in ("cancelled", "faulted"):
        break
    assert terminal == "cancelled"
    state = controller.snapshot()["commands"][command.command_id]
    assert state["completed_cycle"] - state["accepted_cycle"] > 50

  def test_recovery_invariant_failure_latches_fault_without_escaping_step(self):
    group, _oracle, _binding, _payload, cycle = self._dirty()
    controller = group.profile_controller
    controller.recover(cycle)
    # Pending cache cleans are only checked at the commit-phase invariant;
    # the failure must latch a terminal recovery fault instead of escaping
    # group.step as an uncaught invariant error.
    requests = group.tiles[0].l1_cache.begin_maintenance(None, clean=True, invalidate=True)
    assert requests
    stepped = cycle
    for stepped in range(cycle, cycle + 5000):
      group.step(stepped)
      if not controller._recovering:
        break
    assert not controller._recovering
    assert not controller.initialized
    assert controller._recovery_fault
    assert controller.issue_gate_closed("l1") and controller.issue_gate_closed("l2")

  @staticmethod
  def _dirty(timeout=10000, latency=10):
    return TestBoundedMaintenance.dirty_group(timeout=timeout, latency=latency)


class TestPerTileBackfill:
  def test_t03_full_slots_defer_backfill_and_tiles_admit_independently(self):
    import json

    text = """builtin.module {
      tile.program @fill(%t : !nest.task)
        resource_contract = #tile.resources<allowed_profiles = [0],
          tile_l1_spm_bytes_per_context = 1024> {
        %b = tile.alloc shape = [1024] dtype = "i8" alignment = 64 : !tile.l1_buffer<1024xi8>
        %e = tile.evu.async "relu" ops = 320000 : !tile.event<"e">
        tile.await %e
        tile.return
      }
      tile.program @tiny(%t : !nest.task)
        resource_contract = #tile.resources<allowed_profiles = [0],
          tile_l1_spm_bytes_per_context = 1024> {
        %b = tile.alloc shape = [1024] dtype = "i8" alignment = 64 : !tile.l1_buffer<1024xi8>
        %e = tile.evu.async "relu" ops = 32 : !tile.event<"e">
        tile.await %e
        tile.return
      }
      nest.context @filler placement = 15
        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 20, l2_spm_bytes = 0, requested_contexts_per_tile = 4> {
        %t1 = nest.task.range from = 0 to = 4 : !nest.task_range
        %g1, %r1, %w1 = nest.dispatch.tasks.async @fill l1_mode = 0
          tasks(%t1) globals() bindings() ins() outs() signal_policy {}
          : (!nest.event<"g1">, !nest.event<"">, !nest.event<"">)
        %t2 = nest.task.range from = 0 to = 4 : !nest.task_range
        %g2, %r2, %w2 = nest.dispatch.tasks.async @fill l1_mode = 0
          tasks(%t2) globals() bindings() ins() outs() signal_policy {}
          : (!nest.event<"g2">, !nest.event<"">, !nest.event<"">)
        %t3 = nest.task.range from = 0 to = 4 : !nest.task_range
        %g3, %r3, %w3 = nest.dispatch.tasks.async @fill l1_mode = 0
          tasks(%t3) globals() bindings() ins() outs() signal_policy {}
          : (!nest.event<"g3">, !nest.event<"">, !nest.event<"">)
        %t4 = nest.task.range from = 0 to = 4 : !nest.task_range
        %g4, %r4, %w4 = nest.dispatch.tasks.async @fill l1_mode = 0
          tasks(%t4) globals() bindings() ins() outs() signal_policy {}
          : (!nest.event<"g4">, !nest.event<"">, !nest.event<"">)
        %t5 = nest.task.range from = 0 to = 4 : !nest.task_range
        %g5, %r5, %w5 = nest.dispatch.tasks.async @tiny l1_mode = 0
          tasks(%t5) globals() bindings() ins() outs() signal_policy {}
          : (!nest.event<"g5">, !nest.event<"">, !nest.event<"">)
        nest.await %g1, %g2, %g3, %g4, %g5
        nest.return
      }
      nexus.program @fill_tiles {
        %f = nexus.submit_context.async @filler : !nexus.event<"f">
        nexus.await %f
        nexus.return
      }
    }"""
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    config = SimConfig(context_count=4, device_context_count=1, memory_trace=True, max_cycles=200000)
    program = compile_program(parse_workload_ir(text), hw, config)
    simulator = Simulator(hw, config, enable_tracer=True)
    result = simulator.run(load_program(program, hw, config))
    assert result.completed, result.reason
    events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
    lease_events = sorted(
      (
        event
        for event in events
        if event.get("name") in ("task_lease_acquire", "task_lease_release")
      ),
      key=lambda event: event["ts"],
    )
    assert lease_events
    active = 0
    peak = 0
    filler_first_release = None
    backfill_first_acquire = None
    backfill_tiles: set[int] = set()
    for event in lease_events:
      grid_program = event["args"]["program"]
      delta = 1 if event["name"].endswith("acquire") else -1
      active += delta
      peak = max(peak, active)
      if grid_program == "fill" and delta == -1 and filler_first_release is None:
        filler_first_release = event["ts"]
      if grid_program == "tiny" and delta == 1:
        if backfill_first_acquire is None:
          backfill_first_acquire = event["ts"]
        backfill_tiles.add(event["args"]["tile_id"])
      assert active <= 16
    assert peak == 16
    assert filler_first_release is not None and backfill_first_acquire is not None
    # The fifth per-tile Task shares the parent's R=4 budget: it holds no
    # L1/Slot lease until the first filler Task retires on that tile.
    assert backfill_first_acquire >= filler_first_release
    assert backfill_tiles == {0, 1, 2, 3}
    assert not simulator.group._task_leases and not simulator.group._grid_routes


class TestL2TransactionReferences:
  """plan/01 §3/§6.3: owner-side reference acquisition is atomic."""

  def test_partial_acquisition_failure_rolls_back_earlier_references(self):
    from dataclasses import replace as dc_replace

    from pipeline_validator.memory.allocator import MemoryInvariantError
    from pipeline_validator.memory.transfer import MemoryTransaction, ResolvedMemoryView, TransferOp

    _hw, group, cycle = _new_group()
    buffers = (
      ExecL2Buffer("buf_a", (64,), "i8", "inout", 1, 64, 64),
      ExecL2Buffer("buf_b", (64,), "i8", "inout", 1, 64, 64),
    )
    reserved = conservative_arena_bytes([(b.bytes, b.alignment) for b in buffers], group.l2_sram.profile)
    layout = layout_buffers(buffers, group.l2_sram.profile, reserved, lifetimes=None, slot_capacity=2)
    from pipeline_validator.memory.arena import RootInvocation

    plan = group.l2_sram.plan_arena(RootInvocation("ctx", 0), layout)
    assert not isinstance(plan, AdmissionFailure)
    arena = group.l2_sram.commit_arena(plan, cycle)
    view_a = group.l2_sram.bind_view(arena, "buf_a", cycle)
    view_b = group.l2_sram.bind_view(arena, "buf_b", cycle)

    def resolved(handle):
      return ResolvedMemoryView(
        handle=handle,
        offset_bytes=0,
        size_bytes=handle.size_bytes,
        address=handle.base_address,
        segments=handle.bank_segments,
      )

    stale = dc_replace(
      view_b,
      allocation_id="l2:missing:stale",
      generation=view_b.generation + 77,
    )
    # Healthy endpoint first, stale second: the hook must roll back the
    # first registration when the second acquisition raises.
    transaction = MemoryTransaction(
      transaction_id="txn:partial",
      op=TransferOp.PREFETCH,
      issuer=ContextBufferOwner("ctx", 0, "buf_a"),
      src=resolved(view_a),
      dst=resolved(stale),
      bytes_total=view_a.size_bytes,
      completion_event="e",
    )
    before = group.l2_sram.snapshot()
    with pytest.raises(MemoryInvariantError):
      group._acquire_l2_transaction_references(transaction, cycle)
    after = group.l2_sram.snapshot()
    # The first endpoint's registration was rolled back: no inflight
    # reference, no backing freed, no free-map mutation.
    assert group.l2_sram._views[view_a.allocation_id].inflight == set()
    assert after["live_backings"] == before["live_backings"] == 2
    assert after["free_bytes"] == before["free_bytes"]
    assert after["pool_version"] == before["pool_version"]
    # The healthy path still registers both endpoints exactly once.
    healthy = dc_replace(transaction, transaction_id="txn:healthy", dst=resolved(view_b))
    group._acquire_l2_transaction_references(healthy, cycle)
    assert group.l2_sram._views[view_a.allocation_id].inflight == {"txn:healthy"}
    assert group.l2_sram._views[view_b.allocation_id].inflight == {"txn:healthy"}
    group._release_l2_transaction_references(healthy, cycle)
    assert group.l2_sram._views[view_a.allocation_id].inflight == set()
    assert group.l2_sram._views[view_b.allocation_id].inflight == set()


class TestL2TransactionEndpointDuplication:
  """plan/01 §6.3: deduplicated endpoints and cross-issuer references.

  One physical handle referenced by two distinct issuers keeps the backing
  held until BOTH references are released; duplicate endpoints within one
  transaction count once; a stale duplicate identity rolls back cleanly.
  """

  def test_same_handle_as_src_and_dst_registers_and_releases_once(self):
    from dataclasses import replace as dc_replace

    from pipeline_validator.memory.allocator import ContextBufferOwner, MemoryInvariantError
    from pipeline_validator.memory.arena import RootInvocation
    from pipeline_validator.memory.transfer import MemoryTransaction, ResolvedMemoryView, TransferOp

    _hw, group, cycle = _new_group()
    buffer = ExecL2Buffer("buf", (64,), "i8", "inout", 1, 64, 64)
    reserved = conservative_arena_bytes([(buffer.bytes, buffer.alignment)], group.l2_sram.profile)
    layout = layout_buffers((buffer,), group.l2_sram.profile, reserved, lifetimes=None, slot_capacity=1)
    plan = group.l2_sram.plan_arena(RootInvocation("ctx", 0), layout)
    assert not isinstance(plan, AdmissionFailure)
    arena = group.l2_sram.commit_arena(plan, cycle)
    view = group.l2_sram.bind_view(arena, "buf", cycle)

    def resolved(handle):
      return ResolvedMemoryView(
        handle=handle,
        offset_bytes=0,
        size_bytes=handle.size_bytes,
        address=handle.base_address,
        segments=handle.bank_segments,
      )

    owner = ContextBufferOwner("ctx", 0, "buf")
    txn = MemoryTransaction(
      transaction_id="txn:dup",
      op=TransferOp.PREFETCH,
      issuer=owner,
      src=resolved(view),
      dst=resolved(view),
      bytes_total=view.size_bytes,
      completion_event="e",
    )
    group._acquire_l2_transaction_references(txn, cycle)
    assert group.l2_sram._views[view.allocation_id].inflight == {"txn:dup"}
    assert group.l2_sram.snapshot()["inflight_count"] == 1
    group._release_l2_transaction_references(txn, cycle)
    assert group.l2_sram._views[view.allocation_id].inflight == set()

    # A stale duplicate identity after a first reference is rejected and
    # leaves the first registration untouched.
    stale = dc_replace(view, allocation_id="l2:missing:stale", generation=view.generation + 9)
    two_handles = MemoryTransaction(
      transaction_id="txn:two",
      op=TransferOp.PREFETCH,
      issuer=owner,
      src=resolved(view),
      dst=resolved(stale),
      bytes_total=view.size_bytes,
      completion_event="e2",
    )
    group._acquire_l2_transaction_references(txn, cycle)
    with pytest.raises(MemoryInvariantError):
      group._acquire_l2_transaction_references(two_handles, cycle)
    assert group.l2_sram._views[view.allocation_id].inflight == {"txn:dup"}
    group._release_l2_transaction_references(txn, cycle)
    assert group.l2_sram._views[view.allocation_id].inflight == set()

    # Cross-issuer: two distinct issuers share one private handle; the
    # backing stays held until BOTH references are independently released.
    issuer_a = ContextBufferOwner("issuer_a", 0, "buf")
    issuer_b = ContextBufferOwner("issuer_b", 0, "buf")
    txn_a = MemoryTransaction(
      transaction_id="txn:a",
      op=TransferOp.PREFETCH,
      issuer=issuer_a,
      src=resolved(view),
      dst=resolved(view),
      bytes_total=view.size_bytes,
      completion_event="ea",
    )
    txn_b = MemoryTransaction(
      transaction_id="txn:b",
      op=TransferOp.PREFETCH,
      issuer=issuer_b,
      src=resolved(view),
      dst=resolved(view),
      bytes_total=view.size_bytes,
      completion_event="eb",
    )
    group._acquire_l2_transaction_references(txn_a, cycle)
    group._acquire_l2_transaction_references(txn_b, cycle)
    assert group.l2_sram._views[view.allocation_id].inflight == {"txn:a", "txn:b"}
    group._release_l2_transaction_references(txn_a, cycle)
    assert group.l2_sram._views[view.allocation_id].inflight == {"txn:b"}
    assert group.l2_sram.snapshot()["live_backings"] == 1
    group._release_l2_transaction_references(txn_b, cycle)
    assert group.l2_sram._views[view.allocation_id].inflight == set()
    # References only block release; the owner's invalidation is what frees.
    assert group.l2_sram.snapshot()["live_backings"] == 1
    assert group.l2_sram.invalidate_view(view, view.owner, cycle)
    assert group.l2_sram.snapshot()["live_backings"] == 0
