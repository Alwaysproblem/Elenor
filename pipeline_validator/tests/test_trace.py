"""Trace layout tests: lane sort metadata, change-only counters, flows,
and end-to-end memory trace lanes / leg flows / peaks.

PR 5: memory-subsystem state must land on deterministic lanes
(``process_sort_index``/``thread_sort_index`` metadata), counters must be
sampled change-only, and every flow must close.  ``Tracer.assert_well_formed``
is the JSON-level contract these tests enforce.
"""

from __future__ import annotations

import json
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest

from pipeline_validator.compiler import compile_program
from pipeline_validator.config import HardwareConfig
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.simulator import SimConfig, Simulator
from pipeline_validator.trace import Tracer
from pipeline_validator.workloads import PowWorkload


def _events_of(sim: Simulator) -> list[dict]:
  assert sim.tracer is not None
  return json.loads(sim.tracer.to_chrome_json())["traceEvents"]


def run_source(
  simulator: Simulator, module, bindings: dict[str, GlobalBinding] | None = None, *, workload_info=None
):
  artifact = compile_program(module, simulator.hw, simulator.sim, workload_info=workload_info)
  return simulator.run(load_program(artifact, simulator.hw, simulator.sim, actual_bindings=bindings))


def _process_meta(events: list[dict]) -> dict[str, int]:
  names = {e["pid"]: e["args"]["name"] for e in events if e["name"] == "process_name"}
  sorts = {e["pid"]: e["args"]["sort_index"] for e in events if e["name"] == "process_sort_index"}
  return {names[pid]: sort for pid, sort in sorts.items()}


def _thread_meta(events: list[dict], pid: int) -> dict[str, int]:
  names = {e["tid"]: e["args"]["name"] for e in events if e["name"] == "thread_name" and e["pid"] == pid}
  sorts = {
    e["tid"]: e["args"]["sort_index"]
    for e in events
    if e["name"] == "thread_sort_index" and e["pid"] == pid
  }
  return {names[tid]: sort for tid, sort in sorts.items()}


POW_BINDINGS = {"Y": GlobalBinding("Y", 0x100000, 524288, "rw")}


class TestRunTimeline:
  def test_workload_trace_excludes_initialization_without_changing_cycles(self):
    from pipeline_validator.tests.test_profile_runtime import COPY_IR
    from pipeline_validator.workload_ir import parse_workload_ir

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10, clock_mhz=800)
    config = SimConfig(fidelity="full_memory", max_cycles=10000, memory_trace=True)
    bindings = {
      "src": GlobalBinding("src", 0x100000, 64, "r"),
      "dst": GlobalBinding("dst", 0x200000, 64, "w"),
    }
    artifact = compile_program(parse_workload_ir(COPY_IR), hw, config)
    loaded = load_program(artifact, hw, config, actual_bindings=bindings)
    simulator = Simulator(hw, config, enable_tracer=True)
    initial_l1_free = simulator.group.tiles[0].l1_allocator.snapshot()["free_bytes"]
    result = simulator.run(loaded)
    untraced = Simulator(hw, config).run(loaded)
    assert result.completed and untraced.completed
    assert result.cycles == untraced.cycles
    events = _events_of(simulator)
    assert not any(
      event["name"] in ("profile_initialize", "profile_initialized")
      or event.get("args", {}).get("stage") == "INITIALIZE"
      for event in events
    )
    capacity_baselines = [
      event for event in events if event["name"] == "l1_free_bytes" and event["ts"] == 0
    ]
    assert len(capacity_baselines) == hw.num_tiles
    assert all(event["args"]["l1_free_bytes"] == initial_l1_free for event in capacity_baselines)
    assert simulator.tracer is not None
    legs = [event for event in events if "accepted_cycle" in event.get("args", {})]
    assert legs
    for event in legs:
      args = event["args"]
      start = simulator.tracer.cycle_to_us(args["accepted_cycle"])
      end = simulator.tracer.cycle_to_us(args["completion_cycle"])
      assert event["ts"] == pytest.approx(start)
      assert event["dur"] == pytest.approx(max(end - start, 0.001))
    simulator.tracer.assert_well_formed()

  def test_warm_run_retains_capacity_baselines_and_persistent_hbm_bindings(self):
    from pipeline_validator.tests.test_profile_runtime import COPY_IR
    from pipeline_validator.workload_ir import parse_workload_ir

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    config = SimConfig(fidelity="full_memory", max_cycles=10000, memory_trace=True)
    bindings = {
      "src": GlobalBinding("src", 0x100000, 64, "r"),
      "dst": GlobalBinding("dst", 0x200000, 64, "w"),
    }
    artifact = compile_program(parse_workload_ir(COPY_IR), hw, config)
    loaded = load_program(artifact, hw, config, actual_bindings=bindings)
    simulator = Simulator(hw, config, enable_tracer=True)
    pools = [("l2", "TileGroup", simulator.group.l2_sram)]
    pools.extend(("l1", f"Tile{tile.tile_id}", tile.l1_allocator) for tile in simulator.group.tiles)
    expected_free = {}
    for space, track, pool in pools:
      snapshot = pool.snapshot()
      expected_free[(track, f"Memory:{space.upper()} State", f"{space}_free_bytes")] = snapshot[
        "free_bytes"
      ]
      for bank in snapshot["per_bank_occupancy"]:
        expected_free[(track, f"{space.upper()} Bank:{bank['bank_id']}", f"{space}_bank_free_bytes")] = (
          bank["free_bytes"]
        )

    cold = simulator.run(loaded)
    assert cold.completed, cold.reason
    hbm = simulator.group.hbm
    persistent_bindings = {name: hbm.get_handle(name) for name in bindings}
    warm = simulator.run(loaded)
    assert warm.completed, warm.reason
    assert {name: hbm.get_handle(name) for name in bindings} == persistent_bindings
    assert hbm.used_bytes() == sum(binding.size_bytes for binding in bindings.values())

    events = _events_of(simulator)
    tracks = {
      event["pid"]: event["args"]["name"] for event in events if event["name"] == "process_name"
    }
    threads = {
      (event["pid"], event["tid"]): event["args"]["name"]
      for event in events
      if event["name"] == "thread_name"
    }
    first_samples = {}
    for event in events:
      if event["ph"] == "C" and event["ts"] == 0:
        key = (tracks[event["pid"]], threads[(event["pid"], event["tid"])], event["name"])
        first_samples.setdefault(key, event["args"][event["name"]])
    assert {key: first_samples[key] for key in expected_free} == expected_free

    recorded_bindings = {
      event["args"]["binding"]: (event["args"]["allocation_id"], event["args"]["generation"])
      for event in events
      if event["name"] == "hbm_bind"
    }
    expected_bindings = {}
    for name, handle in persistent_bindings.items():
      assert handle is not None
      expected_bindings[name] = (handle.allocation_id, handle.generation)
    assert recorded_bindings == expected_bindings
    for name, value in (
      ("hbm_allocated_bytes", hbm.used_bytes()),
      ("hbm_free_bytes", hbm.size_bytes - hbm.used_bytes()),
    ):
      samples = [event for event in events if event["name"] == name]
      assert samples[-1]["ts"] == 0
      assert samples[-1]["args"][name] == value
    assert simulator.tracer is not None
    simulator.tracer.assert_well_formed()

  def test_uce_terminal_events_do_not_create_zero_duration_state_slices(self):
    hw = HardwareConfig()
    sim = Simulator(hw, SimConfig(fidelity="full_memory", max_cycles=200000), enable_tracer=True)
    workload = PowWorkload(hw=hw)
    result = run_source(sim, workload.module, POW_BINDINGS, workload_info=workload.info)
    assert result.completed, result.reason
    warm = run_source(sim, workload.module, POW_BINDINGS, workload_info=workload.info)
    assert warm.completed, warm.reason

    events = _events_of(sim)
    terminal = [event for event in events if event["name"].startswith(("DONE:", "FAULT:"))]
    assert terminal
    assert all(event["ph"] == "i" for event in terminal)
    threads = {
      (event["pid"], event["tid"])
      for event in events
      if event["name"] == "thread_name" and event["args"]["name"].startswith("UCE CTX")
    }
    for pid, tid in threads:
      open_slices: list[tuple[str, float]] = []
      for event in events:
        if (event["pid"], event["tid"]) != (pid, tid):
          continue
        if event["ph"] == "B":
          open_slices.append((event["name"], event["ts"]))
        elif event["ph"] == "E":
          assert open_slices
          name, start = open_slices.pop()
          assert name == event["name"] and event["ts"] > start
      assert not open_slices

  def test_profile_switch_finishes_before_next_input_load(self, tmp_path):
    repo = Path(__file__).resolve().parents[2]
    trace_path = tmp_path / "profile-switch.json"
    proc = subprocess.run(
      [
        "bash",
        str(repo / "examples/run.sh"),
        "l2-admission-profile-switch",
        "--sim-override",
        "fidelity=full_memory",
        "--memory-trace",
        "--trace-json",
        str(trace_path),
      ],
      capture_output=True,
      text=True,
      cwd=repo,
      timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    events = json.loads(trace_path.read_text())["traceEvents"]
    assert not any(event.get("args", {}).get("stage") == "INITIALIZE" for event in events)
    command = next(
      event
      for event in events
      if event["name"] == "profile_command" and event["args"]["level"] == "l2"
    )
    assert command["args"]["status"] == "completed"
    end = command["ts"] + command["dur"]
    assert end == pytest.approx(command["args"]["completed_cycle"] / 1000)
    inputs = sorted(
      (event for event in events if event.get("cat") == "HBM → L2 Input"),
      key=lambda event: event["ts"],
    )
    assert len(inputs) == 3
    assert max(event["ts"] + event["dur"] for event in inputs[:2]) <= command["ts"]
    assert end <= inputs[2]["ts"]
    members = [event for event in events if event.get("cat") == "Profile:Member"]
    assert {event["args"]["stage"] for event in members} == {"PREPARE", "COMMIT"}
    assert all(command["ts"] <= event["ts"] <= end for event in members)


class TestSortMetadata:
  def test_sort_metadata_orders_lanes(self):
    """Every process/thread carries sort metadata; Device < TileGroup <
    Tile{n}; thread tables order lanes within each process."""
    tr = Tracer(HardwareConfig())
    tr.instant("Device", "Slot:0", "a", 0)
    tr.instant("TileGroup", "Scheduler:L2", "b", 0)
    tr.instant("TileGroup", "HBM → L2 Input #0", "in0", 0)
    tr.instant("TileGroup", "HBM → L2 Input #10", "in10", 0)
    tr.instant("TileGroup", "L2 → HBM Output #0", "out0", 0)
    tr.instant("TileGroup", "L2 → HBM Output #10", "out10", 0)
    tr.instant("TileGroup", "Memory:HBM", "hbm", 0)
    tr.instant("TileGroup", "Memory:L2 Read", "l2r", 0)
    tr.instant("TileGroup", "Memory:L2 Write", "l2w", 0)
    tr.instant("TileGroup", "Memory:L2 State", "l2s", 0)
    tr.instant("TileGroup", "StreamQ:2", "d", 0)
    tr.instant("Tile0", "UCE CTX1", "e", 0)
    tr.instant("Tile0", "MFE_LD0", "f", 0)
    tr.instant("Tile0", "MFE_LD10", "f10", 0)
    tr.instant("Tile0", "BOA", "g", 0)
    tr.instant("Tile0", "EVU", "h", 0)
    tr.instant("Tile0", "MFE", "i", 0)
    tr.instant("Tile0", "USE", "j", 0)
    tr.instant("Tile0", "MFE_ST0", "k", 0)
    tr.instant("Tile0", "MFE_ST10", "k10", 0)
    tr.instant("Tile0", "Memory:L1 Read", "l1r", 0)
    tr.instant("Tile0", "Memory:L1 Write", "l1w", 0)
    tr.instant("Tile0", "Memory:L1 State", "l1s", 0)
    tr.instant("Tile1", "Memory:L1 State", "g", 0)
    events = json.loads(tr.to_chrome_json())["traceEvents"]
    proc = _process_meta(events)
    assert proc["Device"] < proc["TileGroup"] < proc["Tile0"] < proc["Tile1"]
    pids = {e["args"]["name"]: e["pid"] for e in events if e["name"] == "process_name"}
    tg = _thread_meta(events, pids["TileGroup"])
    assert (
      tg["Scheduler:L2"]
      < tg["HBM → L2 Input #0"]
      < tg["HBM → L2 Input #10"]
      < tg["L2 → HBM Output #0"]
      < tg["L2 → HBM Output #10"]
      < tg["Memory:HBM"]
      < tg["Memory:L2 Read"]
      < tg["Memory:L2 Write"]
      < tg["Memory:L2 State"]
      < tg["StreamQ:2"]
    )
    t0 = _thread_meta(events, pids["Tile0"])
    assert t0["UCE CTX1"] < t0["MFE_LD0"] < t0["MFE_LD10"]
    assert (
      t0["MFE_LD10"]
      < t0["BOA"]
      < t0["EVU"]
      < t0["MFE"]
      < t0["USE"]
      < t0["MFE_ST0"]
      < t0["MFE_ST10"]
      < t0["Memory:L1 Read"]
      < t0["Memory:L1 Write"]
      < t0["Memory:L1 State"]
    )
    tr.assert_well_formed()

  def test_unknown_process_and_thread_get_fallback_sort(self):
    tr = Tracer(HardwareConfig())
    tr.instant("Mystery", "Weird", "a", 0)
    events = json.loads(tr.to_chrome_json())["traceEvents"]
    proc = _process_meta(events)
    assert proc["Mystery"] == 900_000

  def test_lane_suffix_outside_reserved_band_is_rejected(self):
    tr = Tracer(HardwareConfig())
    with pytest.raises(ValueError, match="exceeds reserved range"):
      tr.instant("Tile0", "MFE_LD1000", "overflow", 0)


class TestChangeOnlyCounters:
  def test_counter_if_changed_dedups(self):
    tr = Tracer(HardwareConfig())
    tr.counter_if_changed("TileGroup", "occupancy", 0, 1, "tokens", thread="StreamQ:0")
    tr.counter_if_changed("TileGroup", "occupancy", 1, 1, "tokens", thread="StreamQ:0")
    tr.counter_if_changed("TileGroup", "occupancy", 2, 2, "tokens", thread="StreamQ:0")
    tr.counter_if_changed("TileGroup", "occupancy", 3, 2, "tokens", thread="StreamQ:0")
    samples = [e for e in json.loads(tr.to_chrome_json())["traceEvents"] if e["name"] == "occupancy"]
    assert [s["args"]["occupancy"] for s in samples] == [1, 2]
    tr.assert_well_formed()

  def test_counter_thread_defaults_to_counter_name(self):
    tr = Tracer(HardwareConfig())
    tr.counter("Tile0", "active_context_count", 5, 3, "contexts")
    events = json.loads(tr.to_chrome_json())["traceEvents"]
    sample = next(e for e in events if e["name"] == "active_context_count")
    threads = {
      e["args"]["name"] for e in events if e["name"] == "thread_name" and e["pid"] == sample["pid"]
    }
    assert "active_context_count" in threads


class TestFlows:
  def test_flow_well_formed_and_unclosed_detected(self):
    tr = Tracer(HardwareConfig())
    tr.flow_start("TileGroup", "HBM Ch:0", "hbm_read", 3, "txn1")
    tr.flow_step("TileGroup", "NoC:VC2", "noc_response", 7, "txn1")
    tr.flow_end("Tile0", "Local DMA Load", "local_dma", 12, "txn1")
    tr.assert_well_formed()

    broken = Tracer(HardwareConfig())
    broken.flow_start("TileGroup", "HBM Ch:0", "hbm_read", 3, "txn2")
    with pytest.raises(AssertionError, match="lacks an end event"):
      broken.assert_well_formed()

  def test_flow_ids_are_dense_and_deterministic(self):
    tr = Tracer(HardwareConfig())
    assert tr.flow_id("b") == 1
    assert tr.flow_id("a") == 2
    assert tr.flow_id("b") == 1
    assert tr.flow_id("c") == 3


class TestMemoryLanes:
  def test_memory_counters_on_correct_lanes(self):
    """Memory counters stay on state lanes; transfer legs split by direction."""
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = Simulator(
      hw, SimConfig(fidelity="full_memory", max_cycles=200000, memory_trace=True), enable_tracer=True
    )
    workload = PowWorkload(hw=hw)
    result = run_source(sim, workload.module, POW_BINDINGS, workload_info=workload.info)
    assert result.completed, result.reason
    assert sim.tracer is not None
    sim.tracer.assert_well_formed()
    events = _events_of(sim)
    pname = {e["pid"]: e["args"]["name"] for e in events if e.get("name") == "process_name"}
    thread_names = {
      (e["pid"], e["tid"]): e["args"]["name"] for e in events if e.get("name") == "thread_name"
    }
    tile_pids = {pid for pid, name in pname.items() if name.startswith("Tile")}
    tg_pid = next(pid for pid, name in pname.items() if name == "TileGroup")
    for e in events:
      if e.get("name") == "l1_allocated_bytes":
        assert e["pid"] in tile_pids, e
        assert thread_names[(e["pid"], e["tid"])] == "Memory:L1 State"
      if e.get("name") == "l2_allocated_bytes":
        assert e["pid"] == tg_pid, e
        assert thread_names[(e["pid"], e["tid"])] == "Memory:L2 State"
      if e.get("name") in ("hbm_outstanding", "noc_occupancy", "noc_credit_available"):
        assert e["pid"] == tg_pid, e

    expected_leg_lanes = {
      "l1_read": "Memory:L1 Read",
      "l1_write": "Memory:L1 Write",
      "l2_read": "Memory:L2 Read",
      "l2_write": "Memory:L2 Write",
    }
    observed_leg_lanes = {
      e["name"]: thread_names[(e["pid"], e["tid"])]
      for e in events
      if e.get("ph") == "X" and e.get("name") in expected_leg_lanes
    }
    assert expected_leg_lanes.keys() <= observed_leg_lanes.keys()
    for leg_name, lane_name in expected_leg_lanes.items():
      assert observed_leg_lanes[leg_name] == lane_name

  def test_no_consecutive_duplicate_counter_samples(self):
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = Simulator(
      hw, SimConfig(fidelity="full_memory", max_cycles=200000, memory_trace=True), enable_tracer=True
    )
    workload = PowWorkload(hw=hw)
    result = run_source(sim, workload.module, POW_BINDINGS, workload_info=workload.info)
    assert result.completed, result.reason
    sim.tracer.assert_well_formed()

  def test_report_peaks_match_trace_counters(self):
    """The report's memory peak values match the trace counter maxima."""
    from pipeline_validator.report import build_report

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = Simulator(
      hw, SimConfig(fidelity="full_memory", max_cycles=200000, memory_trace=True), enable_tracer=True
    )
    wl = PowWorkload(hw=hw)
    result = run_source(sim, wl.module, POW_BINDINGS, workload_info=wl.info)
    assert result.completed, result.reason
    report = build_report(wl.info, result, num_tiles=hw.num_tiles)
    events = _events_of(sim)
    if report.memory.get("l2_peak_allocated_bytes") is not None:
      peak = max(
        (e["args"]["l2_allocated_bytes"] for e in events if e.get("name") == "l2_allocated_bytes"),
        default=0,
      )
      assert peak == report.memory["l2_peak_allocated_bytes"]
    if report.memory.get("hbm_outstanding_peak") is not None:
      peak = max(
        (e["args"]["hbm_outstanding"] for e in events if e.get("name") == "hbm_outstanding"), default=0
      )
      assert peak == report.memory["hbm_outstanding_peak"]


class TestLegSlicesAndFlows:
  def test_leg_slices_carry_identity_and_flows_close(self):
    """Gather leg slices carry the required identity args and every
    flow has exactly one start and one end."""
    from pipeline_validator.tests.test_runtime import GATHER_BINDINGS, make_gather_module

    module = make_gather_module([("r0", "L1_HIT", "line0", None), ("r1", "L2_HIT", "line1", None)])
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = Simulator(
      hw, SimConfig(fidelity="full_memory", max_cycles=10000, memory_trace=True), enable_tracer=True
    )
    result = run_source(sim, module, GATHER_BINDINGS)
    assert result.completed, result.reason
    assert sim.tracer is not None
    sim.tracer.assert_well_formed()
    events = _events_of(sim)
    thread_names = {
      (e["pid"], e["tid"]): e["args"]["name"] for e in events if e.get("name") == "thread_name"
    }
    expected_memory_lanes = {
      "l1_read": "Memory:L1 Read",
      "l1_write": "Memory:L1 Write",
      "l1_cache_lookup": "Memory:L1 Read",
      "l1_cache_fill": "Memory:L1 Write",
      "l2_read": "Memory:L2 Read",
      "l2_write": "Memory:L2 Write",
      "l2_cache_lookup": "Memory:L2 Read",
      "l2_cache_fill": "Memory:L2 Write",
    }
    required = {
      "transaction_id",
      "flow_id",
      "bytes",
      "accepted_cycle",
      "completion_cycle",
      "source_space",
      "destination_space",
    }
    leg_names = {
      "hbm_read",
      "hbm_write",
      "global_dma",
      "noc_request",
      "noc_response",
      "l2_read",
      "l2_write",
      "local_dma",
      "l1_read",
      "l1_write",
      "l1_cache_lookup",
      "l2_cache_lookup",
      "l1_cache_fill",
      "l2_cache_fill",
    }
    legs = [e for e in events if e.get("ph") == "X" and e["name"] in leg_names]
    assert legs, "no leg slices emitted"
    for e in legs:
      assert required <= set(e.get("args", {})), e["name"]
      expected_lane = expected_memory_lanes.get(e["name"])
      if expected_lane is not None:
        assert thread_names[(e["pid"], e["tid"])] == expected_lane
    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    for e in events:
      if e.get("ph") == "s":
        starts[e["id"]] = starts.get(e["id"], 0) + 1
      elif e.get("ph") == "f":
        ends[e["id"]] = ends.get(e["id"], 0) + 1
    for fid in set(starts) | set(ends):
      assert starts.get(fid, 0) == 1, fid
      assert ends.get(fid, 0) == 1, fid
    summary = [e for e in events if e.get("args", {}).get("summary_kind") == "group_transfer"]
    if summary:
      fid = summary[0]["args"]["flow_id"]
      leg_threads = {
        e["tid"]
        for e in events
        if e.get("ph") == "X" and e["name"] in leg_names and e.get("args", {}).get("flow_id") == fid
      }
      assert len(leg_threads) >= 2, leg_threads


class TestDualContextFixtureWellFormed:
  def test_dual_context_fixture_well_formed(self, tmp_path):
    """The dual-context gather example produces a well-formed trace via
    the CLI pipeline (lanes correct, flows closed, no dup counters)."""
    repo = Path(__file__).resolve().parents[2]
    trace_path = tmp_path / "memory.json"
    proc = subprocess.run(
      [
        "bash",
        str(repo / "examples/run.sh"),
        "gather-matmul-4tiles-2contexts",
        "--memory-trace",
        "--trace-json",
        str(trace_path),
      ],
      capture_output=True,
      text=True,
      cwd=repo,
      timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    trace = json.loads(trace_path.read_text())
    events = trace["traceEvents"]
    pids = {e["pid"] for e in events if e.get("name") == "process_name"}
    assert pids, "no process metadata"
    starts: dict = {}
    ends: dict = {}
    for e in events:
      if e.get("ph") == "s":
        starts[e["id"]] = starts.get(e["id"], 0) + 1
      elif e.get("ph") == "f":
        ends[e["id"]] = ends.get(e["id"], 0) + 1
    for fid in set(starts) | set(ends):
      assert starts.get(fid, 0) == 1, f"flow {fid} double start"
      assert ends.get(fid, 0) == 1, f"flow {fid} double end"
    pname = {e["pid"]: e["args"]["name"] for e in events if e.get("name") == "process_name"}
    tile_pids = {pid for pid, n in pname.items() if n.startswith("Tile")}
    tg_pid = next((pid for pid, n in pname.items() if n == "TileGroup"), None)
    for e in events:
      if e.get("name") == "l1_allocated_bytes":
        assert e["pid"] in tile_pids, e
      if e.get("name") == "l2_allocated_bytes" and tg_pid is not None:
        assert e["pid"] == tg_pid, e
    thread_names = {
      (e["pid"], e["tid"]): e["args"]["name"] for e in events if e.get("name") == "thread_name"
    }
    assert "GroupDMA" not in thread_names.values()
    summaries = [
      e for e in events if e.get("ph") == "X" and e.get("args", {}).get("summary_kind") == "group_transfer"
    ]
    assert summaries
    input_lanes: set[str] = set()
    events_by_lane: dict[tuple[int, int], list[dict]] = {}
    for event in summaries:
      args = event["args"]
      lane_key = (event["pid"], event["tid"])
      lane_name = thread_names[lane_key]
      events_by_lane.setdefault(lane_key, []).append(event)
      assert event["name"] == f"{args['context_name']} / {args['buffer_id']}"
      assert args["duration_cycles"] > 0
      assert args["effective_bandwidth_gbs"] > 0
      if args["direction"] == "HBM → L2 Input":
        assert lane_name == f"HBM → L2 Input #{args['visual_slot']}"
        assert event["cat"] == "HBM → L2 Input"
        input_lanes.add(lane_name)
      else:
        assert args["direction"] == "L2 → HBM Output"
        assert lane_name == f"L2 → HBM Output #{args['visual_slot']}"
        assert event["cat"] == "L2 → HBM Output"
    assert len(input_lanes) >= 2
    for lane_events in events_by_lane.values():
      ordered = sorted(lane_events, key=lambda event: event["ts"])
      for previous, current in pairwise(ordered):
        assert previous["ts"] + previous["dur"] <= current["ts"]
    last: dict = {}
    for e in events:
      if e.get("ph") != "C":
        continue
      key = (e["pid"], e["tid"], e["name"])
      val = e["args"][e["name"]]
      if key in last:
        assert last[key] != val, (key, val)
      last[key] = val


class TestAdmissionLanes:
  def test_admission_instants_on_scheduler_lane(self, tmp_path):
    """l2_admission_wait scenario: admission instants on the Scheduler:L2
    lane and phase_aggregate carries expected/seen counts."""
    repo = Path(__file__).resolve().parents[2]
    trace_path = tmp_path / "admission.json"
    proc = subprocess.run(
      [
        "bash",
        str(repo / "examples/run.sh"),
        "l2-admission-wait",
        "--memory-trace",
        "--trace-json",
        str(trace_path),
      ],
      capture_output=True,
      text=True,
      cwd=repo,
      timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    trace = json.loads(trace_path.read_text())
    events = trace["traceEvents"]
    sched = {
      e["tid"]: e["args"]["name"]
      for e in events
      if e.get("name") == "thread_name" and e["args"]["name"] == "Scheduler:L2"
    }
    assert sched, "Scheduler:L2 thread not registered"
    sched_tid = next(iter(sched))
    admission_names = {"context_admission_wait", "context_admission_retry", "context_first_action"}
    admission_events = [
      event for event in events if event.get("ph") == "i" and event.get("name") in admission_names
    ]
    assert admission_events
    assert all(event.get("tid") == sched_tid for event in admission_events)
    on_lane = {event["name"] for event in admission_events}
    assert admission_names <= on_lane, admission_names - on_lane

    waits = [event for event in admission_events if event["name"] == "context_admission_wait"]
    retries = [event for event in admission_events if event["name"] == "context_admission_retry"]
    first_b = next(
      event
      for event in admission_events
      if event["name"] == "context_first_action" and event.get("args", {}).get("context") == "ctx_b"
    )
    assert waits and retries
    assert {event["args"]["wait_reason"] for event in waits} <= {"WAIT_CAPACITY", "WAIT_FRAGMENTATION"}
    assert all(event["args"]["retry_count"] >= 1 for event in retries)
    assert {event["args"]["context"] for event in waits + retries} == {"ctx_b"}
    wait_requests = {event["args"]["request_id"] for event in waits}
    retry_requests = {event["args"]["request_id"] for event in retries}
    assert wait_requests == retry_requests
    assert min(event["ts"] for event in waits) <= min(event["ts"] for event in retries)
    assert max(event["ts"] for event in retries) <= first_b["ts"]
    aggregates = [e for e in events if e.get("ph") == "i" and e.get("name") == "phase_aggregate"]
    assert aggregates, "no phase_aggregate instant"
    for e in aggregates:
      assert "expected" in e["args"] and "seen" in e["args"]


class TestL2ExtentReleaseTrace:
  """plan/01 §5/§6.6: physical final-free events carry post-mutation state."""

  @staticmethod
  def _pool_with_trace():
    from pipeline_validator.memory.arena import ArenaPool
    from pipeline_validator.profiles import MemoryProfile
    from pipeline_validator.trace import MemoryTrace

    profile = MemoryProfile(
      level="l2",
      mode=0,
      bank_bytes=512,
      banks=2,
      pools=1,
      spm_bytes_per_bank=512,
      cache_bytes_per_bank=0,
      system_reserved_spm_per_bank=0,
      alignment=64,
      spm_mapping_id="striped_arena_v0",
      cache_org_id="profiled_lru_v0",
      cache_write_policy="read_only",
      maintenance_caps=("invalidate_range", "bypass"),
    )
    tracer = Tracer(HardwareConfig())
    pool = ArenaPool(profile, trace=MemoryTrace(tracer))
    return pool, tracer

  def test_release_instant_matches_post_mutation_counters(self):
    from pipeline_validator.compiler.resources import conservative_arena_bytes, layout_buffers
    from pipeline_validator.execution_ir import ExecL2Buffer
    from pipeline_validator.memory import AdmissionFailure
    from pipeline_validator.memory.arena import RootInvocation

    pool, tracer = self._pool_with_trace()
    buffer = ExecL2Buffer("buf", (100,), "i8", "inout", 1, 64, 100)
    reserved = conservative_arena_bytes([(buffer.bytes, buffer.alignment)], pool.profile)
    layout = layout_buffers((buffer,), pool.profile, reserved, lifetimes=None, slot_capacity=1)
    plan = pool.plan_arena(RootInvocation("ctx_a", 3), layout)
    assert not isinstance(plan, AdmissionFailure)
    arena = pool.commit_arena(plan, 5)
    view = pool.bind_view(arena, "buf", 6)
    assert pool.invalidate_view(view, view.owner, 10)
    assert pool.retire_arena(arena, 12)

    events = json.loads(tracer.to_chrome_json())["traceEvents"]
    releases = [event for event in events if event.get("name") == "l2_extent_release"]
    assert len(releases) == 1
    args = releases[0]["args"]
    cycle = args["release_cycle"]
    assert cycle == 10
    # Owner/run identity: releases are attributable across invocations.
    assert args["context_name"] == "ctx_a"
    assert args["launch_generation"] == 3
    assert args["allocation_generation"] == 1
    assert args["run_generation"] == 0  # standalone pool: no TileGroup launch

    # The logical invalidation instant carries the backing identity and its
    # pre-free physical state (plan/01 §5).
    invalidations = [
      event
      for event in events
      if event.get("name") == "buffer_view_invalidate"
      and event.get("args", {}).get("buffer_id") == "buf"
    ]
    assert len(invalidations) == 1
    invalidate_args = invalidations[0]["args"]
    assert invalidate_args["backing_id"] == args["backing_id"]
    assert invalidate_args["backing_state"] == "live"

    def counter_at(name: str, at_cycle: int):
      samples = [
        event
        for event in events
        if event.get("ph") == "C"
        and event.get("name") == name
        and round(event["ts"] * 1000.0) == at_cycle
      ]
      assert samples, f"missing counter {name} at cycle {at_cycle}"
      return samples[-1]["args"][name]

    # The instant carries the same-cycle post-mutation counter values.
    assert counter_at("l2_arena_reserved_bytes", cycle) == args["pool_reserved_bytes"]
    assert counter_at("l2_free_bytes", cycle) == args["pool_free_bytes"]
    assert args["pool_live_backings"] == 0
    # Physical conservation holds at the mutation point.
    profile_capacity = pool.profile.user_spm_bytes
    assert args["pool_reserved_bytes"] + args["pool_free_bytes"] == profile_capacity

    # The backing lifetime slice spans exactly commit -> final-free.
    lifetimes = [event for event in events if event.get("name") == "l2_backing_lifetime"]
    assert len(lifetimes) == 1
    lifetime = lifetimes[0]
    assert round(lifetime["ts"] * 1000.0) == 5
    assert round(lifetime["ts"] * 1000.0) + round(lifetime["dur"] * 1000.0) == 10

    # The Arena retirement remains a distinct later event with its own slice.
    retires = [
      event
      for event in events
      if event.get("name") == "arena_retire" and event.get("ph") == "i"
    ]
    assert len(retires) == 1 and round(retires[0]["ts"] * 1000.0) == 12
