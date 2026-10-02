"""Plan §Verification 5: host runtime sessions, billing and wake ordering."""

import pytest

from pipeline_validator.compiler.api import compile_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.memory.page_pool import HostPagePoolSpec, PageAllocation
from pipeline_validator.runtime.host_session import (
  HostAllocPages,
  HostDelay,
  HostEnvironment,
  HostFreePages,
  HostWrite,
)
from pipeline_validator.simulator import Simulator
from pipeline_validator.workload_ir import parse_workload_ir

OUT_BASE = 0x100000
OUT_BYTES = 64
POOL_BASE = 0x400000
POOL_BYTES = 65536

_MODEL = """
  builtin.module {
    nest.context @worker placement = 1
        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
            logical_tasks = 0, l2_spm_bytes = 0, requested_contexts_per_tile = 1> { nest.return }
    nexus.program @run(%OUT : !nest.global_memref<64xi8>) {
      %prepared = nexus.host.call.async "fill"
          bindings(%OUT) accesses = [{offset = 0, bytes = 16, mode = "write"}]
          : !nexus.event<"prepared">
      %done = nexus.submit_context.async @worker depends_on(%prepared) : !nexus.event<"done">
      nexus.await %done
      nexus.return
    }
  }
"""


def _fill(request):
  yield HostWrite("OUT", 0, b"\x5a" * 16)
  yield HostWrite("OUT", 8, b"\xa5" * 8)  # declared range overlap, ordered by window


def _bindings():

  return {"OUT": GlobalBinding("OUT", OUT_BASE, OUT_BYTES, "w")}


def _compiled():
  module = parse_workload_ir(_MODEL)
  hw = HardwareConfig()
  sim = SimConfig(fidelity="full_memory", max_cycles=200000)
  return compile_program(module, hw, sim, source_name="<host-runtime-test>"), hw, sim


def _run(host):
  """Compile, load and run once.  ``host`` is the HostEnvironment or None."""
  artifact, hw, sim = _compiled()
  store = ByteStore()
  simulator = Simulator(hw, sim, byte_store=store)
  loaded = load_program(artifact, hw, sim, actual_bindings=_bindings())
  result = simulator.run(loaded, host=host)
  return result, store, simulator


def test_host_call_writes_billed_bytes_and_wakes_submit(tmp_path):
  result, store, _simulator = _run(HostEnvironment({"fill": _fill}))
  assert result.completed, result.reason
  # first write [0,16) then second write [8,16) overwrites by program order
  assert store.read_hbm(OUT_BASE, 16) == b"\x5a" * 8 + b"\xa5" * 8
  # the second write ran after the first one, later-issued wins on overlap
  assert store.read_hbm(OUT_BASE + 8, 8) == b"\xa5" * 8

  device = result.device_snapshot
  host = device["host_runtime"]
  assert host["counters"]["commands"] == 2
  assert host["counters"]["write_bytes"] == 24
  counters = device["counters"]
  assert counters["host_submitted"] == 1
  assert counters["host_completed"] == 1
  assert counters["host_failed"] == 0

  # cycle-N completion wakes the dependent submit at N+1
  records = {record["event"]: record for record in device["launch_records"]}
  prepared, done = records["prepared"], records["done"]
  assert prepared["submit_cycle"] + 1 <= done["submit_cycle"]


def test_host_call_without_environment_is_rejected(tmp_path):
  with pytest.raises(ValueError, match="no host environment"):
    _run(None)


def test_missing_handler_fails_at_issue_time(tmp_path):
  result, _store, _simulator = _run(HostEnvironment({"other": _fill}))
  assert not result.completed
  assert "host handler 'fill' is not registered" in result.reason


def test_uncovered_write_fails(tmp_path):
  def overwriter(request):
    yield HostWrite("OUT", 32, b"\x01" * 4)  # beyond the declared [0, 16)

  result, _store, _simulator = _run(HostEnvironment({"fill": overwriter}))
  assert not result.completed
  assert "beyond its declared range" in result.reason


def test_delay_command_between_writes_completes(tmp_path):
  def writer(request):
    yield HostWrite("OUT", 0, b"\x02" * 4)
    yield HostDelay(2)
    yield HostWrite("OUT", 4, b"\x03" * 4)

  result, store, _simulator = _run(HostEnvironment({"fill": writer}))
  assert result.completed, result.reason
  assert store.read_hbm(OUT_BASE, 8) == b"\x02" * 4 + b"\x03" * 4


def test_host_backpressure_does_not_consume_group_outstanding(tmp_path):
  def slow(request):
    yield HostDelay(64)

  result, _store, _simulator = _run(HostEnvironment({"fill": slow}))
  assert result.completed, result.reason
  device = result.device_snapshot
  assert device["counters"]["host_completed"] == 1
  assert device["port"]["active"] == 0


_POOL_MODEL = """
  builtin.module {
    nest.context @worker placement = 1
        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
            logical_tasks = 0, l2_spm_bytes = 0, requested_contexts_per_tile = 1> { nest.return }
    nexus.program @run(%POOL : !nest.global_memref<64xi8>) {
      %prepared = nexus.host.call.async "pages"
          bindings(%POOL) accesses = [{offset = 0, bytes = 8, mode = "write"}]
          scopes = ["owner_0"] : !nexus.event<"prepared">
      %done = nexus.submit_context.async @worker depends_on(%prepared) : !nexus.event<"done">
      nexus.await %done
      nexus.return
    }
  }
"""


def _pages(request):
  """Allocate two pages, write the first, then free them."""
  allocation = yield HostAllocPages("kv", "owner_0", 2)
  assert isinstance(allocation, PageAllocation)
  assert len(allocation.pages) == 2
  yield HostWrite("POOL", 0, bytes(allocation.pages), scope="owner_0")
  yield HostFreePages("kv", "owner_0")


def _pool_environment():
  spec = HostPagePoolSpec(
    name="kv", binding="POOL", page_bytes=4096, page_count=16, initial_owners={"owner_0": (0,)}
  )
  return HostEnvironment({"pages": _pages}, pools=(spec,))


def test_pool_allocate_and_free_complete_and_deliver_pages():
  """Plan §Verification 5: pool commands complete at issue, not a transfer."""
  hw = HardwareConfig()
  sim = SimConfig(fidelity="full_memory", max_cycles=200000)
  artifact = compile_program(
    parse_workload_ir(_POOL_MODEL), hw, sim, source_name="<host-pool-test>"
  )
  store = ByteStore()
  store.seed_hbm(POOL_BASE, b"\x00" * POOL_BYTES)
  bindings = {"POOL": GlobalBinding("POOL", POOL_BASE, POOL_BYTES, "rw")}
  loaded = load_program(artifact, hw, sim, actual_bindings=bindings)
  result = Simulator(hw, sim, byte_store=store).run(loaded, host=_pool_environment())
  assert result.completed, result.reason
  snapshot = result.device_snapshot["page_pools"]["pools"]
  kv = snapshot["kv"]
  assert kv["allocated_pages"] == 2
  # HostFreePages releases every page the scope holds, its initial page
  # included, so the pool ends empty and closed.
  assert kv["freed_pages"] == 3
  assert kv["live_pages"] == 0
  assert kv["peak_live_pages"] == 3
  assert kv["pins"] == 0
  assert kv["leases"] == 0
