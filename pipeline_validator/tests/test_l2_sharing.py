"""End-to-end in-process sharing contracts, before the batch-III CLI scenarios."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from xdsl.utils.exceptions import VerifyException

from pipeline_validator.compiled_program import parse_compiled_program, serialize_compiled_program
from pipeline_validator.compiler import compile_program
from pipeline_validator.config import GroupSchedulerConfig, HardwareConfig, SimConfig
from pipeline_validator.execution_ir import ExecModel, GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.profiles import ProfileBytes, ProfileLevelSource
from pipeline_validator.simulator import Simulator
from pipeline_validator.tests.test_l2_sharing_source import CONTEXT_LOCAL_IR
from pipeline_validator.workload_ir import parse_workload_ir, print_workload_ir

READER_AND_MODEL = """
  tile.program @read_shared(%task : !nest.task, %weight : !nest.l2_buffer<8192xi8>,
      %result : !nest.l2_buffer<4x8192xi8>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 8192> {
    %src = tile.subview %weight offsets = [0] sizes = [8192] strides = [1]
        : !nest.l2_view<8192xi8>
    %dst = tile.subview %result task = %task task_dim = 0 offsets = [0, 0]
        sizes = [1, 8192] strides = [1, 1] : !nest.l2_view<1x8192xi8>
    %local = tile.alloc shape = [8192] dtype = "i8" alignment = 256
        : !tile.l1_buffer<8192xi8>
    %loaded = tile.load.async %src into %local : !tile.event<"loaded">
    tile.await %loaded
    tile.signal input_released(%task)
    %stored = tile.store.async %local into %dst : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @reader(%OUT : !nest.global_memref<4x8192xi8>,
      %weight : !nest.l2_buffer<8192xi8>) placement = 15
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 4, l2_spm_bytes = 32768, requested_contexts_per_tile = 1> {
    %result = nest.alloc slot = "result" role = "out" shape = [4, 8192] dtype = "i8"
        alignment = 256 : !nest.l2_buffer<4x8192xi8>
    %output = nest.subview %OUT offsets = [0, 0] sizes = [4, 8192] strides = [1, 1]
        : !nest.global_view<4x8192xi8>
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid, %input_done, %ready = nest.dispatch.tasks.async @read_shared l1_mode = 0
        tasks(%tasks) globals() bindings(%weight, %result) ins(%weight) outs(%result)
        signal_policy {input_released = #nest.aggregate<all_tasks>,
                       output_ready = #nest.aggregate<all_tasks>}
        : (!nest.event<"grid">, !nest.event<"input_done">, !nest.event<"ready">)
    nest.release %weight depends_on(%input_done)
    %written = nest.dma.store.async %result into %output depends_on(%ready)
        : !nest.event<"written">
    nest.release %result depends_on(%written)
    nest.await %grid, %written
    nest.return
  }
  nexus.program @share(%SOURCE : !nest.global_memref<8192xi8>,
      %B_OUT : !nest.global_memref<4x8192xi8>, %C_OUT : !nest.global_memref<4x8192xi8>) {
    %produced = nexus.submit_context.async @producer(%SOURCE) : !nexus.event<"produced">
    %shared = nexus.shared.ref %produced slot = "W" : !nest.l2_buffer<8192xi8>
    %b = nexus.submit_context.async @reader(%B_OUT, %shared) : !nexus.event<"b">
    nexus.await %b
    %c = nexus.submit_context.async @reader(%C_OUT, %shared) : !nexus.event<"c">
    nexus.await %c
    nexus.return
  }
"""

WEIGHT_PRODUCER = """
  nest.context @producer(%SOURCE : !nest.global_memref<8192xi8>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 0, l2_spm_bytes = 8192, requested_contexts_per_tile = 1> {
    %w = nest.alloc slot = "W" role = "in" sharing = "readonly" shape = [8192]
        dtype = "i8" alignment = 256 : !nest.l2_buffer<8192xi8>
    %source = nest.subview %SOURCE offsets = [0] sizes = [8192] strides = [1]
        : !nest.global_view<8192xi8>
    %prefetched = nest.dma.prefetch.async %source into %w : !nest.event<"prefetched">
    %published = nest.publish %w depends_on(%prefetched) : !nest.event<"published">
    nest.release %w depends_on(%prefetched, %published)
    nest.return
  }
"""

FANOUT_PRODUCER = """
  tile.program @make_x(%task : !nest.task, %input : !nest.l2_buffer<8192xi8>,
      %output : !nest.l2_buffer<8192xi8>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
          tile_l1_spm_bytes_per_context = 8192> {
    %src = tile.subview %input offsets = [0] sizes = [8192] strides = [1]
        : !nest.l2_view<8192xi8>
    %dst = tile.subview %output offsets = [0] sizes = [8192] strides = [1]
        : !nest.l2_view<8192xi8>
    %local = tile.alloc shape = [8192] dtype = "i8" alignment = 256
        : !tile.l1_buffer<8192xi8>
    %loaded = tile.load.async %src into %local : !tile.event<"loaded">
    tile.await %loaded
    tile.signal input_released(%task)
    %stored = tile.store.async %local into %dst : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @producer(%SOURCE : !nest.global_memref<8192xi8>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 1, l2_spm_bytes = 16384, requested_contexts_per_tile = 1> {
    %input = nest.alloc slot = "input" role = "in" shape = [8192] dtype = "i8"
        alignment = 256 : !nest.l2_buffer<8192xi8>
    %w = nest.alloc slot = "W" role = "out" sharing = "readonly" shape = [8192]
        dtype = "i8" alignment = 256 : !nest.l2_buffer<8192xi8>
    %source = nest.subview %SOURCE offsets = [0] sizes = [8192] strides = [1]
        : !nest.global_view<8192xi8>
    %prefetched = nest.dma.prefetch.async %source into %input : !nest.event<"prefetched">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %input_done, %ready = nest.dispatch.tasks.async @make_x l1_mode = 0
        tasks(%tasks) globals() bindings(%input, %w) ins(%input) outs(%w)
        signal_policy {input_released = #nest.aggregate<all_tasks>,
                       output_ready = #nest.aggregate<all_tasks>} depends_on(%prefetched)
        : (!nest.event<"grid">, !nest.event<"input_done">, !nest.event<"ready">)
    nest.release %input depends_on(%input_done, %prefetched)
    %published = nest.publish %w depends_on(%ready) : !nest.event<"published">
    nest.release %w depends_on(%published)
    nest.await %grid
    nest.return
  }
"""


def sharing_source(*, fanout: bool = False) -> str:
  producer = FANOUT_PRODUCER if fanout else WEIGHT_PRODUCER
  return f"builtin.module {{{producer}{READER_AND_MODEL}}}\n"

PRIVATE_READER_MODEL = """
  nest.context @private_reader(%SOURCE : !nest.global_memref<8192xi8>,
      %OUT : !nest.global_memref<4x8192xi8>) placement = 15
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 4, l2_spm_bytes = 40960, requested_contexts_per_tile = 1> {
    %weight = nest.alloc slot = "weight" role = "in" shape = [8192] dtype = "i8"
        alignment = 256 : !nest.l2_buffer<8192xi8>
    %result = nest.alloc slot = "result" role = "out" shape = [4, 8192] dtype = "i8"
        alignment = 256 : !nest.l2_buffer<4x8192xi8>
    %source = nest.subview %SOURCE offsets = [0] sizes = [8192] strides = [1]
        : !nest.global_view<8192xi8>
    %output = nest.subview %OUT offsets = [0, 0] sizes = [4, 8192] strides = [1, 1]
        : !nest.global_view<4x8192xi8>
    %prefetched = nest.dma.prefetch.async %source into %weight : !nest.event<"prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid, %input_done, %ready = nest.dispatch.tasks.async @read_shared l1_mode = 0
        tasks(%tasks) globals() bindings(%weight, %result) ins(%weight) outs(%result)
        signal_policy {input_released = #nest.aggregate<all_tasks>,
                       output_ready = #nest.aggregate<all_tasks>} depends_on(%prefetched)
        : (!nest.event<"grid">, !nest.event<"input_done">, !nest.event<"ready">)
    nest.release %weight depends_on(%input_done, %prefetched)
    %written = nest.dma.store.async %result into %output depends_on(%ready)
        : !nest.event<"written">
    nest.release %result depends_on(%written)
    nest.await %grid, %written
    nest.return
  }
  nexus.program @private_copy(%SOURCE : !nest.global_memref<8192xi8>,
      %B_OUT : !nest.global_memref<4x8192xi8>, %C_OUT : !nest.global_memref<4x8192xi8>) {
    %b = nexus.submit_context.async @private_reader(%SOURCE, %B_OUT) : !nexus.event<"b">
    nexus.await %b
    %c = nexus.submit_context.async @private_reader(%SOURCE, %C_OUT) : !nexus.event<"c">
    nexus.await %c
    nexus.return
  }
"""


def private_source() -> str:
  shared_tile = READER_AND_MODEL.split("  nest.context @reader(", 1)[0]
  return f"builtin.module {{{shared_tile}{PRIVATE_READER_MODEL}}}\n"



def _run(source: str):
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10, num_dma_channels=2)
  sim = SimConfig(fidelity="full_memory", device_context_count=2, max_cycles=200000, memory_trace=True)
  oracle = ByteStore()
  payload = bytes(range(256)) * 32
  oracle.seed_hbm(0x100000, payload)
  oracle.seed_hbm(0x200000, bytes(32768))
  oracle.seed_hbm(0x300000, bytes(32768))
  bindings = {
    "SOURCE": GlobalBinding("SOURCE", 0x100000, 8192, "r"),
    "B_OUT": GlobalBinding("B_OUT", 0x200000, 32768, "w"),
    "C_OUT": GlobalBinding("C_OUT", 0x300000, 32768, "w"),
  }
  module = parse_workload_ir(source)
  artifact = compile_program(module, hw, sim)
  loaded = load_program(artifact, hw, sim, actual_bindings=bindings)
  simulator = Simulator(hw, sim, enable_tracer=True, byte_store=oracle)
  result = simulator.run(loaded)
  return module, artifact, simulator, result, oracle, payload


@pytest.mark.parametrize("fanout", [False, True], ids=["weight", "intermediate-fanout"])
def test_two_readers_keep_one_physical_backing_through_delayed_consumer(fanout):
  module, artifact, simulator, result, oracle, payload = _run(sharing_source(fanout=fanout))
  assert result.completed, result.reason
  assert oracle.read_hbm(0x200000, 32768) == payload * 4
  assert oracle.read_hbm(0x300000, 32768) == payload * 4
  assert isinstance(artifact.entry, ExecModel)
  assert print_workload_ir(parse_workload_ir(print_workload_ir(module))) == print_workload_ir(module)
  assert len([task for task in artifact.entry.tasks.values() if task.shared_inputs]) == 2
  assert result.device_snapshot["port"]["request_records"][0]["status"] == "success"
  assert result.group_snapshot["arenas"]["l2"]["live_backings"] == 0
  assert result.group_snapshot["arenas"]["l2"]["pending_shared_claims"] == 0
  assert result.group_snapshot["arenas"]["l2"]["active_shared_references"] == 0
  assert result.group_snapshot["arenas"]["l2"]["arena_reserved_bytes"] == 0
  assert result.tracer is not None
  trace = json.loads(result.tracer.to_chrome_json())["traceEvents"]
  releases = [event for event in trace if event["name"] == "l2_extent_release"]
  shared_frees = [event for event in releases if event["args"].get("buffer_id") == "W"]
  assert len(shared_frees) == 1
  producer = next(
    item for item in result.device_snapshot["launch_records"] if item["context"] == "producer"
  )
  reader_records = [
    item for item in result.device_snapshot["launch_records"] if item["context"] == "reader"
  ]
  assert len(reader_records) == 2
  assert producer["completion_cycle"] < reader_records[0]["admission_cycle"]
  assert reader_records[0]["completion_cycle"] < reader_records[1]["admission_cycle"]
  assert reader_records[1]["admission_cycle"] < shared_frees[0]["args"]["release_cycle"]
  assert reader_records[1]["completion_cycle"] >= shared_frees[0]["args"]["release_cycle"]
  assert simulator.group.transfer_manager.snapshot()["inflight"] == 0
  if fanout:
    issued = simulator.group.transfer_manager.snapshot()["issued_by_op"]
    assert issued.get("prefetch", 0) == 1
    assert issued.get("global_store", 0) == 2


def test_reshaped_equal_byte_prefetch_can_publish_full_readonly_buffer():
  source = (
    sharing_source()
    .replace(
      "%SOURCE : !nest.global_memref<8192xi8>",
      "%SOURCE : !nest.global_memref<128x64xi8>",
    )
    .replace(
      "%source = nest.subview %SOURCE offsets = [0] sizes = [8192] strides = [1]\n"
      "        : !nest.global_view<8192xi8>",
      "%source = nest.subview %SOURCE offsets = [0, 0] sizes = [128, 64] strides = [1, 1]\n"
      "        : !nest.global_view<128x64xi8>",
    )
  )
  _module, artifact, simulator, result, oracle, payload = _run(source)
  assert result.completed, result.reason
  assert isinstance(artifact.entry, ExecModel)
  producer = next(task for task in artifact.entry.tasks.values() if task.name == "producer")
  assert producer.global_inputs[0].dims == (128, 64)
  assert producer.l2_buffers[0].dims == (8192,)
  assert oracle.read_hbm(0x200000, 32768) == payload * 4
  assert oracle.read_hbm(0x300000, 32768) == payload * 4
  assert simulator.group.transfer_manager.snapshot()["issued_by_op"]["prefetch"] == 1
  assert result.group_snapshot["arenas"]["l2"]["live_backings"] == 0


def test_shared_weight_eliminates_one_real_hbm_prefetch_without_skipping_tile_reads():
  _module, _artifact, shared, result, shared_bytes, payload = _run(sharing_source())
  assert result.completed, result.reason
  _module, _artifact, private, control, private_bytes, _ = _run(private_source())
  assert control.completed, control.reason
  for oracle in (shared_bytes, private_bytes):
    assert oracle.read_hbm(0x200000, 32768) == payload * 4
    assert oracle.read_hbm(0x300000, 32768) == payload * 4
  shared_ops = shared.group.transfer_manager.snapshot()["issued_by_op"]
  private_ops = private.group.transfer_manager.snapshot()["issued_by_op"]
  assert shared_ops["prefetch"] * 8192 == 8192
  assert private_ops["prefetch"] * 8192 == 16384
  assert shared_ops["tile_load"] == private_ops["tile_load"]
  assert result.group_snapshot["scheduler"]["reserved_l2_bytes_peak"] == 40960


def _small_l2_hardware() -> HardwareConfig:
  """64 KiB user SPM: one W plus one reader output fits; two do not."""
  base = HardwareConfig()
  return base.with_overrides(
    group_sram_bytes=65536, hbm_fixed_latency_cycles=10, num_dma_channels=2,
    memory_target=replace(
      base.memory_target,
      l2=replace(base.memory_target.l2, system_reserved_spm_per_bank=0),
    ),
    profile_source=replace(
      base.profile_source, l2=ProfileLevelSource({0: ProfileBytes(4096, 0)})
    ),
  )


def _concurrent_readers_source() -> str:
  return (
    sharing_source()
    .replace("    nexus.await %b\n    %c =", "    %c =", 1)
    .replace("    nexus.await %c\n", "    nexus.await %b, %c\n", 1)
  )


def test_pending_reader_waits_for_private_capacity_without_losing_shared_weight():
  source = _concurrent_readers_source()
  hw = _small_l2_hardware()
  sim = SimConfig(
    fidelity="full_memory", context_count=2, device_context_count=2,
    group=GroupSchedulerConfig(active_context_capacity=2), max_cycles=200000, memory_trace=True,
  )
  oracle = ByteStore()
  payload = bytes(range(256)) * 32
  oracle.seed_hbm(0x100000, payload)
  oracle.seed_hbm(0x200000, bytes(32768))
  oracle.seed_hbm(0x300000, bytes(32768))
  bindings = {
    "SOURCE": GlobalBinding("SOURCE", 0x100000, 8192, "r"),
    "B_OUT": GlobalBinding("B_OUT", 0x200000, 32768, "w"),
    "C_OUT": GlobalBinding("C_OUT", 0x300000, 32768, "w"),
  }
  simulator = Simulator(hw, sim, enable_tracer=True, byte_store=oracle)
  compiled = compile_program(parse_workload_ir(source), hw, sim)
  result = simulator.run(load_program(compiled, hw, sim, actual_bindings=bindings))
  assert result.completed, result.reason
  assert oracle.read_hbm(0x200000, 32768) == payload * 4
  assert oracle.read_hbm(0x300000, 32768) == payload * 4
  assert result.tracer is not None
  events = json.loads(result.tracer.to_chrome_json())["traceEvents"]
  waited = [
    item for item in events if item["name"] == "context_admission_wait"
    and item["args"]["wait_reason"] == "WAIT_CAPACITY"
  ]
  assert waited
  records = [
    row for row in result.device_snapshot["port"]["request_records"] if row["context"] == "reader"
  ]
  assert len(records) == 2
  first, second = sorted(records, key=lambda row: row["ready_seq"])
  freed_first_output = [
    item["args"]["release_cycle"] for item in events
    if item["name"] == "l2_extent_release" and item["args"].get("buffer_id") == "result"
  ]
  assert freed_first_output
  assert second["active_cycle"] == min(freed_first_output)
  weight_free = [
    item["args"]["release_cycle"] for item in events
    if item["name"] == "l2_extent_release" and item["args"].get("buffer_id") == "W"
  ]
  assert len(weight_free) == 1
  assert weight_free[0] > first["completion_cycle"]
  assert weight_free[0] >= second["active_cycle"]


def test_fault_while_c_waits_for_capacity_cancels_declared_claim_after_drain(monkeypatch):
  hw = _small_l2_hardware()
  sim = SimConfig(
    fidelity="full_memory", context_count=2, device_context_count=2,
    group=GroupSchedulerConfig(active_context_capacity=2), max_cycles=200000, memory_trace=True,
  )
  oracle = ByteStore()
  oracle.seed_hbm(0x100000, bytes(range(256)) * 32)
  oracle.seed_hbm(0x200000, bytes(32768))
  oracle.seed_hbm(0x300000, bytes(32768))
  bindings = {
    "SOURCE": GlobalBinding("SOURCE", 0x100000, 8192, "r"),
    "B_OUT": GlobalBinding("B_OUT", 0x200000, 32768, "w"),
    "C_OUT": GlobalBinding("C_OUT", 0x300000, 32768, "w"),
  }
  simulator = Simulator(hw, sim, enable_tracer=True, byte_store=oracle)
  assert simulator.tracer is not None
  program = compile_program(parse_workload_ir(_concurrent_readers_source()), hw, sim)
  original_step = simulator.group.step
  observed_pending: list[int] = []

  def fault_at_wait(cycle: int) -> bool:
    done = original_step(cycle)
    if not observed_pending and any(
      item["name"] == "context_admission_wait"
      and item["args"].get("wait_reason") == "WAIT_CAPACITY"
      for item in simulator.tracer._events
    ):
      observed_pending.append(simulator.group.l2_sram.snapshot()["pending_shared_claims"])
      simulator.group.trigger_fault(
        simulator.group._fault_code_for_reason("shared capacity fault injection"),
        cycle=cycle, desc_id="shared capacity fault injection",
      )
    return done

  monkeypatch.setattr(simulator.group, "step", fault_at_wait)
  result = simulator.run(load_program(program, hw, sim, actual_bindings=bindings))
  assert observed_pending == [1]
  assert not result.completed
  assert result.device_snapshot["faulted"]
  assert simulator.group.reset_domain.is_done
  assert simulator.group.poisoned_reason is None
  assert result.group_snapshot["arenas"]["l2"]["live_backings"] == 0
  assert result.group_snapshot["arenas"]["l2"]["pending_shared_claims"] == 0
  frees = [
    item["args"] for item in json.loads(simulator.tracer.to_chrome_json())["traceEvents"]
    if item["name"] == "l2_extent_release" and item["args"].get("buffer_id") == "W"
  ]
  assert len(frees) == 1
  assert any(
    claim["state"] == "CANCELLED" and claim["claim_id"][1] == "weight"
    for claim in frees[0]["claims"]
  )


def test_unisolatable_shared_claim_poison_keeps_backing_and_rejects_reuse(monkeypatch):
  from pipeline_validator.memory import MemoryInvariantError

  base = _small_l2_hardware()
  hw = base.with_overrides(
    memory_target=replace(base.memory_target, profile_command_timeout_cycles=512)
  )
  sim = SimConfig(
    fidelity="full_memory", context_count=2, device_context_count=2,
    group=GroupSchedulerConfig(active_context_capacity=2), max_cycles=20000, memory_trace=True,
  )
  oracle = ByteStore()
  oracle.seed_hbm(0x100000, bytes(range(256)) * 32)
  oracle.seed_hbm(0x200000, bytes(32768))
  oracle.seed_hbm(0x300000, bytes(32768))
  bindings = {
    "SOURCE": GlobalBinding("SOURCE", 0x100000, 8192, "r"),
    "B_OUT": GlobalBinding("B_OUT", 0x200000, 32768, "w"),
    "C_OUT": GlobalBinding("C_OUT", 0x300000, 32768, "w"),
  }
  simulator = Simulator(hw, sim, enable_tracer=True, byte_store=oracle)
  assert simulator.tracer is not None
  program = compile_program(parse_workload_ir(_concurrent_readers_source()), hw, sim)
  original_step = simulator.group.step
  fault_injected = False

  def fault_at_wait(cycle: int) -> bool:
    nonlocal fault_injected
    done = original_step(cycle)
    if not fault_injected and any(
      item["name"] == "context_admission_wait"
      and item["args"].get("wait_reason") == "WAIT_CAPACITY"
      for item in simulator.tracer._events
    ):
      fault_injected = True
      simulator.group.trigger_fault(
        simulator.group._fault_code_for_reason("isolation failure injection"),
        cycle=cycle, desc_id="isolation failure injection",
      )
    return done

  monkeypatch.setattr(simulator.group, "step", fault_at_wait)
  # Force only the reset-domain isolation check to remain incomplete; no view
  # or free-map is cleared to fake a safe recovery.
  monkeypatch.setattr(simulator.group.reset_domain, "_outstanding_zero", lambda _group, _cycle: False)
  loaded = load_program(program, hw, sim, actual_bindings=bindings)
  result = simulator.run(loaded)
  assert fault_injected
  assert not result.completed
  assert "poisoned" in result.reason
  assert not simulator.group.reset_domain.is_done
  assert simulator.group.poisoned_reason is not None
  assert "claim=" in simulator.group.poisoned_reason
  backing_ids = simulator.group.l2_sram.live_backing_ids()
  assert backing_ids and any(":W" in backing_id for backing_id in backing_ids)
  assert any(backing_id in result.reason for backing_id in backing_ids)
  assert result.group_snapshot["arenas"]["l2"]["arena_reserved_bytes"] > 0
  with pytest.raises(MemoryInvariantError, match="poisoned"):
    simulator.group.begin_launch(loaded.compiled, bindings)
  with pytest.raises(MemoryInvariantError, match="reset"):
    simulator.group.reset()


@pytest.mark.parametrize("await_head", [False, True], ids=["queued-tail", "awaited-successor"])
def test_fifo_head_faults_when_tail_reader_claim_retains_the_only_required_extent(await_head):
  reader = READER_AND_MODEL.split("  nexus.program @share(", 1)[0]
  head = """
  nest.context @head(%HEAD_IN : !nest.global_memref<65536xi8>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 0, l2_spm_bytes = 65536, requested_contexts_per_tile = 1> {
    %local = nest.alloc slot = "head_data" role = "in" shape = [65536] dtype = "i8"
        alignment = 256 : !nest.l2_buffer<65536xi8>
    %input = nest.subview %HEAD_IN offsets = [0] sizes = [65536] strides = [1]
        : !nest.global_view<65536xi8>
    %loaded = nest.dma.prefetch.async %input into %local : !nest.event<"loaded">
    nest.release %local depends_on(%loaded)
    nest.return
  }
"""
  model = """
  nexus.program @blocked(%SOURCE : !nest.global_memref<8192xi8>,
      %HEAD_IN : !nest.global_memref<65536xi8>, %C_OUT : !nest.global_memref<4x8192xi8>) {
    %produced = nexus.submit_context.async @producer(%SOURCE) : !nexus.event<"produced">
    %shared = nexus.shared.ref %produced slot = "W" : !nest.l2_buffer<8192xi8>
    %blocked = nexus.submit_context.async @head(%HEAD_IN) depends_on(%produced)
        : !nexus.event<"blocked">
    %c = nexus.submit_context.async @reader(%C_OUT, %shared) : !nexus.event<"c">
    nexus.await %blocked, %c
    nexus.return
  }
"""
  if await_head:
    model = model.replace(
      '    %c = nexus.submit_context.async @reader(%C_OUT, %shared)',
      '    nexus.await %blocked\n    %c = nexus.submit_context.async @reader(%C_OUT, %shared)',
    ).replace("    nexus.await %blocked, %c", "    nexus.await %c")
  hw = _small_l2_hardware()
  sim = SimConfig(
    fidelity="full_memory", device_context_count=2, max_cycles=100000, memory_trace=True,
    group=GroupSchedulerConfig(active_context_capacity=2),
  )
  oracle = ByteStore()
  oracle.seed_hbm(0x100000, bytes(range(256)) * 32)
  oracle.seed_hbm(0x400000, bytes(65536))
  oracle.seed_hbm(0x300000, bytes(32768))
  bindings = {
    "SOURCE": GlobalBinding("SOURCE", 0x100000, 8192, "r"),
    "HEAD_IN": GlobalBinding("HEAD_IN", 0x400000, 65536, "r"),
    "C_OUT": GlobalBinding("C_OUT", 0x300000, 32768, "w"),
  }
  simulator = Simulator(hw, sim, enable_tracer=True, byte_store=oracle)
  program = compile_program(
    parse_workload_ir(f"builtin.module {{{WEIGHT_PRODUCER}{reader}{head}{model}}}"),
    hw, sim,
  )
  result = simulator.run(load_program(program, hw, sim, actual_bindings=bindings))
  assert not result.completed
  assert "retained shared extents prevent FIFO-head admission" in result.reason
  assert simulator.group.reset_domain.is_done
  assert simulator.group.poisoned_reason is None
  assert result.group_snapshot["arenas"]["l2"]["live_backings"] == 0
  assert result.group_snapshot["arenas"]["l2"]["arena_reserved_bytes"] == 0
  assert result.tracer is not None
  frees = [
    item["args"] for item in json.loads(result.tracer.to_chrome_json())["traceEvents"]
    if item["name"] == "l2_extent_release" and item["args"].get("buffer_id") == "W"
  ]
  assert len(frees) == 1
  assert any(claim["state"] == "CANCELLED" for claim in frees[0]["claims"])
  tails = [
    item for item in result.device_snapshot["port"]["request_records"] if item["context"] == "reader"
  ]
  if await_head:
    assert not tails
  else:
    assert len(tails) == 1 and tails[0]["active_cycle"] is None


def test_retained_shared_backing_keeps_fragmentation_temporary_when_others_can_release():
  from pipeline_validator.compiler.resources import layout_buffers
  from pipeline_validator.execution_ir import ExecL2Buffer
  from pipeline_validator.memory import AdmissionFailure, AdmissionFailureKind
  from pipeline_validator.memory.arena import ArenaPool, RootInvocation
  from pipeline_validator.profiles import MemoryProfile

  profile = MemoryProfile(
    level="l2", mode=0, bank_bytes=512, banks=2, pools=1,
    spm_bytes_per_bank=512, cache_bytes_per_bank=0, system_reserved_spm_per_bank=0,
    alignment=64, spm_mapping_id="striped_arena_v0", cache_org_id="profiled_lru_v0",
    cache_write_policy="read_only", maintenance_caps=("invalidate_range", "bypass"),
  )
  pool = ArenaPool(profile)
  pool.run_generation = 1
  weight = ExecL2Buffer("W", (128,), "i8", "in", 1, 64, 128, "readonly")
  weight_layout = layout_buffers((weight,), profile, 128, slot_capacity=1)
  producer_plan = pool.plan_arena(RootInvocation("producer", 0), weight_layout)
  assert not isinstance(producer_plan, AdmissionFailure)
  producer_arena = pool.commit_arena(
    producer_plan, 0, claims_by_slot={"W": (("reader", "W"),)}
  )
  producer = pool.bind_view(producer_arena, "W", 1)
  pool.publish_l2(producer, 2)
  assert pool.invalidate_view(producer, producer.owner, 3)
  assert pool.retire_arena(producer_arena, 4)
  assert pool.snapshot()["live_backings"] == 1

  unit_layout = layout_buffers((), profile, 256, slot_capacity=1)
  holders = []
  for index in range(3):
    plan = pool.plan_arena(RootInvocation(f"holder{index}", 0), unit_layout)
    assert not isinstance(plan, AdmissionFailure)
    holders.append(pool.commit_arena(plan, index + 5))
  assert pool.retire_arena(holders[1], 9)
  pending_layout = layout_buffers((), profile, 384, slot_capacity=1)
  before = pool.snapshot()
  outcome = pool.plan_arena(RootInvocation("fragmented", 0), pending_layout)
  assert isinstance(outcome, AdmissionFailure)
  assert outcome.kind is AdmissionFailureKind.TEMPORARY_CAPACITY
  assert outcome.reason.startswith("WAIT_FRAGMENTATION")
  assert pool.can_fit_with_retained(pending_layout, (producer.backing_id,))
  assert pool.snapshot() == before
  assert pool.cancel_l2_claim(producer.backing_id, ("reader", "W"), 10)
  assert pool.retire_arena(holders[0], 11)
  assert pool.retire_arena(holders[2], 12)
  pool.close_l2_claims()
  assert pool.snapshot()["arena_reserved_bytes"] == 0


def test_published_destination_rejects_forged_write_permission_before_transfer_acceptance():
  from pipeline_validator.compiler.resources import layout_buffers
  from pipeline_validator.execution_ir import ExecL2Buffer
  from pipeline_validator.memory import (
    AdmissionFailure,
    MemoryTransaction,
    ResolvedMemoryView,
    TransferOp,
    TransferStatus,
  )
  from pipeline_validator.memory.arena import RootInvocation
  from pipeline_validator.tile_group import TileGroup

  hw = HardwareConfig()
  oracle = ByteStore()
  group = TileGroup(hw, fidelity="full_memory", byte_store=oracle)
  group.run_generation = 1
  group.l2_sram.run_generation = 1
  group.transfer_manager.begin_run(1)
  pool = group.l2_sram
  layout = layout_buffers(
    (ExecL2Buffer("W", (128,), "i8", "in", 1, 64, 128, "readonly"),),
    pool.profile, 1024, slot_capacity=1,
  )
  plan = pool.plan_arena(RootInvocation("producer", 0), layout)
  assert not isinstance(plan, AdmissionFailure)
  arena = pool.commit_arena(plan, 0)
  producer = pool.bind_view(arena, "W", 1)
  oracle.write_view(
    ResolvedMemoryView(producer, 0, 128, producer.base_address, producer.bank_segments, "rw"),
    b"\x5a" * 128,
  )
  pool.publish_l2(producer, 2)
  binding = GlobalBinding("source", 0x100000, 128, "r")
  oracle.seed_hbm(binding.base_iova, bytes(range(128)))
  source = group.hbm.bind_external(binding, 2)
  src_view = ResolvedMemoryView(source, 0, 128, source.base_address, source.bank_segments, "r")
  forged_dst = ResolvedMemoryView(
    producer, 0, 128, producer.base_address, producer.bank_segments, "rw"
  )
  txn = MemoryTransaction(
    "forged-published-destination", TransferOp.PREFETCH, producer.owner,
    src_view, forged_dst, 128, "done", run_generation=1,
    profile_generations=(("l2", 0, 0),),
  )
  before = pool.snapshot()
  assert not group.validate_transaction_generation(txn, "destination_commit")
  group.transfer_manager.submit(txn, 3)
  assert txn.status is TransferStatus.FAULTED
  assert pool.snapshot() == before
  assert oracle.read_view(forged_dst) == b"\x5a" * 128
  group.transfer_manager.acknowledge(txn.transaction_id, 4)
  assert pool.invalidate_view(producer, producer.owner, 5)
  assert pool.retire_arena(arena, 6)
  group.assert_l2_closed()


def test_zero_consumer_export_releases_its_backing_at_producer_release():
  text = f"""builtin.module {{{WEIGHT_PRODUCER}
  nexus.program @single(%SOURCE : !nest.global_memref<8192xi8>) {{
    %done = nexus.submit_context.async @producer(%SOURCE) : !nexus.event<"done">
    nexus.await %done
    nexus.return
  }}
}}
"""
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  sim = SimConfig(fidelity="full_memory", max_cycles=100000, memory_trace=True)
  module = parse_workload_ir(text)
  artifact = compile_program(module, hw, sim)
  bindings = {"SOURCE": GlobalBinding("SOURCE", 0x100000, 8192, "r")}
  oracle = ByteStore()
  oracle.seed_hbm(0x100000, bytes(range(256)) * 32)
  simulator = Simulator(hw, sim, enable_tracer=True, byte_store=oracle)
  result = simulator.run(load_program(artifact, hw, sim, actual_bindings=bindings))
  assert result.completed, result.reason
  assert result.group_snapshot["arenas"]["l2"]["live_backings"] == 0
  assert result.tracer is not None
  releases = [
    event for event in json.loads(result.tracer.to_chrome_json())["traceEvents"]
    if event["name"] == "l2_extent_release" and event["args"].get("buffer_id") == "W"
  ]
  assert len(releases) == 1

def test_two_runs_rebind_shared_weight_without_reusing_prior_backing_or_bytes():
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10, num_dma_channels=2)
  sim = SimConfig(fidelity="full_memory", device_context_count=2, max_cycles=200000, memory_trace=True)
  artifact = compile_program(parse_workload_ir(sharing_source()), hw, sim)
  oracle = ByteStore()
  first = bytes(range(256)) * 32
  second = bytes(reversed(range(256))) * 32
  oracle.seed_hbm(0x100000, first)
  oracle.seed_hbm(0x200000, bytes(32768))
  oracle.seed_hbm(0x300000, bytes(32768))
  simulator = Simulator(hw, sim, enable_tracer=True, byte_store=oracle)
  outputs = {
    "B_OUT": GlobalBinding("B_OUT", 0x200000, 32768, "w"),
    "C_OUT": GlobalBinding("C_OUT", 0x300000, 32768, "w"),
  }
  for address, expected in ((0x100000, first), (0x400000, second)):
    if address == 0x400000:
      oracle.seed_hbm(address, expected)
    bindings = {**outputs, "SOURCE": GlobalBinding("SOURCE", address, 8192, "r")}
    result = simulator.run(load_program(artifact, hw, sim, actual_bindings=bindings))
    assert result.completed, result.reason
    assert oracle.read_hbm(0x200000, 32768) == expected * 4
    assert oracle.read_hbm(0x300000, 32768) == expected * 4
    simulator.group.assert_l2_closed()
  assert simulator.tracer is not None
  frees = [
    event["args"] for event in json.loads(simulator.tracer.to_chrome_json())["traceEvents"]
    if event["name"] == "l2_extent_release" and event["args"].get("buffer_id") == "W"
  ]
  assert len(frees) == 2
  assert frees[0]["backing_id"] != frees[1]["backing_id"]
  assert frees[0]["run_generation"] != frees[1]["run_generation"]



def test_readonly_import_cannot_become_dispatch_destination():
  bad = sharing_source().replace("ins(%weight) outs(%result)", "ins() outs(%weight, %result)")
  with pytest.raises(VerifyException, match=r"(?i)read|writ|import|readonly|binding"):
    parse_workload_ir(bad)


def test_context_local_tasks_mutate_one_backing_and_release_it() -> None:
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10, num_dma_channels=2)
  sim = SimConfig(fidelity="full_memory", context_count=1, max_cycles=200000, memory_trace=True)
  rows = [
    bytes((r * 53 + (i // 4096) * 17 + i % 251) % 256 for i in range(8192))
    for r in range(4)
  ]
  source = b"".join(rows)
  expected = b"".join(row[4096:] + row[:4096] for row in rows)
  assert expected != source
  oracle = ByteStore()
  oracle.seed_hbm(0x100000, source)
  oracle.seed_hbm(0x200000, bytes(32768))
  bindings = {
    "SOURCE": GlobalBinding("SOURCE", 0x100000, 32768, "r"),
    "OUT": GlobalBinding("OUT", 0x200000, 32768, "w"),
  }
  artifact = compile_program(
    parse_workload_ir(print_workload_ir(parse_workload_ir(CONTEXT_LOCAL_IR))), hw, sim
  )
  replayed = parse_compiled_program(serialize_compiled_program(artifact))
  result = Simulator(hw, sim, enable_tracer=True, byte_store=oracle).run(
    load_program(replayed, hw, sim, actual_bindings=bindings)
  )
  assert result.completed, result.reason
  assert oracle.read_hbm(0x200000, 32768) == expected
  l2 = result.group_snapshot["arenas"]["l2"]
  for key in (
    "live_backings",
    "arena_reserved_bytes",
    "pending_shared_claims",
    "active_shared_references",
  ):
    assert l2[key] == 0
