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

import pytest

from pipeline_validator.config import HardwareConfig
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.simulator import Simulator, SimConfig
from pipeline_validator.trace import Tracer
from pipeline_validator.workloads import PowWorkload


def _events_of(sim: Simulator) -> list[dict]:
  assert sim.tracer is not None
  return json.loads(sim.tracer.to_chrome_json())["traceEvents"]


def _process_meta(events: list[dict]) -> dict[str, int]:
  names = {
    e["pid"]: e["args"]["name"]
    for e in events if e["name"] == "process_name"
  }
  sorts = {
    e["pid"]: e["args"]["sort_index"]
    for e in events if e["name"] == "process_sort_index"
  }
  return {names[pid]: sort for pid, sort in sorts.items()}


def _thread_meta(events: list[dict], pid: int) -> dict[str, int]:
  names = {
    e["tid"]: e["args"]["name"]
    for e in events
    if e["name"] == "thread_name" and e["pid"] == pid
  }
  sorts = {
    e["tid"]: e["args"]["sort_index"]
    for e in events
    if e["name"] == "thread_sort_index" and e["pid"] == pid
  }
  return {names[tid]: sort for tid, sort in sorts.items()}


POW_BINDINGS = {"Y": GlobalBinding("Y", 0x100000, 524288, "rw")}


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
    pids = {
      e["args"]["name"]: e["pid"]
      for e in events if e["name"] == "process_name"
    }
    tg = _thread_meta(events, pids["TileGroup"])
    assert (tg["Scheduler:L2"] < tg["HBM → L2 Input #0"]
            < tg["HBM → L2 Input #10"] < tg["L2 → HBM Output #0"]
            < tg["L2 → HBM Output #10"] < tg["Memory:HBM"]
            < tg["Memory:L2 Read"] < tg["Memory:L2 Write"]
            < tg["Memory:L2 State"] < tg["StreamQ:2"])
    t0 = _thread_meta(events, pids["Tile0"])
    assert t0["UCE CTX1"] < t0["MFE_LD0"] < t0["MFE_LD10"]
    assert (t0["MFE_LD10"] < t0["BOA"] < t0["EVU"] < t0["MFE"]
            < t0["USE"] < t0["MFE_ST0"] < t0["MFE_ST10"]
            < t0["Memory:L1 Read"] < t0["Memory:L1 Write"]
            < t0["Memory:L1 State"])
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
    tr.counter_if_changed("TileGroup", "occupancy", 0, 1, "tokens",
                          thread="StreamQ:0")
    tr.counter_if_changed("TileGroup", "occupancy", 1, 1, "tokens",
                          thread="StreamQ:0")
    tr.counter_if_changed("TileGroup", "occupancy", 2, 2, "tokens",
                          thread="StreamQ:0")
    tr.counter_if_changed("TileGroup", "occupancy", 3, 2, "tokens",
                          thread="StreamQ:0")
    samples = [e for e in json.loads(tr.to_chrome_json())["traceEvents"]
              if e["name"] == "occupancy"]
    assert [s["args"]["occupancy"] for s in samples] == [1, 2]
    tr.assert_well_formed()

  def test_counter_thread_defaults_to_counter_name(self):
    tr = Tracer(HardwareConfig())
    tr.counter("Tile0", "active_context_count", 5, 3, "contexts")
    events = json.loads(tr.to_chrome_json())["traceEvents"]
    sample = next(e for e in events
                  if e["name"] == "active_context_count")
    threads = {
      e["args"]["name"]
      for e in events
      if e["name"] == "thread_name" and e["pid"] == sample["pid"]
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
      hw, SimConfig(fidelity="full_memory", max_cycles=200000,
                    memory_trace=True),
      enable_tracer=True,
    )
    result = sim.run(PowWorkload().module, input_bindings=POW_BINDINGS)
    assert result.completed, result.reason
    assert sim.tracer is not None
    sim.tracer.assert_well_formed()
    events = _events_of(sim)
    pname = {
      e["pid"]: e["args"]["name"]
      for e in events if e.get("name") == "process_name"
    }
    thread_names = {
      (e["pid"], e["tid"]): e["args"]["name"]
      for e in events if e.get("name") == "thread_name"
    }
    tile_pids = {pid for pid, name in pname.items()
                 if name.startswith("Tile")}
    tg_pid = next(pid for pid, name in pname.items()
                  if name == "TileGroup")
    for e in events:
      if e.get("name") == "l1_allocated_bytes":
        assert e["pid"] in tile_pids, e
        assert thread_names[(e["pid"], e["tid"])] == "Memory:L1 State"
      if e.get("name") == "l2_allocated_bytes":
        assert e["pid"] == tg_pid, e
        assert thread_names[(e["pid"], e["tid"])] == "Memory:L2 State"
      if e.get("name") in ("hbm_outstanding", "noc_occupancy",
                            "noc_credit_available"):
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
      hw, SimConfig(fidelity="full_memory", max_cycles=200000,
                    memory_trace=True),
      enable_tracer=True,
    )
    result = sim.run(PowWorkload().module, input_bindings=POW_BINDINGS)
    assert result.completed, result.reason
    sim.tracer.assert_well_formed()

  def test_report_peaks_match_trace_counters(self):
    """The report's memory peak values match the trace counter maxima."""
    from pipeline_validator.report import build_report

    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = Simulator(
      hw, SimConfig(fidelity="full_memory", max_cycles=200000,
                    memory_trace=True),
      enable_tracer=True,
    )
    wl = PowWorkload()
    result = sim.run(wl.module, input_bindings=POW_BINDINGS)
    assert result.completed, result.reason
    report = build_report(wl, result, num_tiles=hw.num_tiles)
    events = _events_of(sim)
    if report.memory.get("l2_peak_allocated_bytes") is not None:
      peak = max((e["args"]["l2_allocated_bytes"] for e in events
                  if e.get("name") == "l2_allocated_bytes"), default=0)
      assert peak == report.memory["l2_peak_allocated_bytes"]
    if report.memory.get("hbm_outstanding_peak") is not None:
      peak = max((e["args"]["hbm_outstanding"] for e in events
                  if e.get("name") == "hbm_outstanding"), default=0)
      assert peak == report.memory["hbm_outstanding_peak"]


class TestLegSlicesAndFlows:
  def test_leg_slices_carry_identity_and_flows_close(self):
    """Gather leg slices carry the required identity args and every
    flow has exactly one start and one end."""
    from pipeline_validator.tests.test_runtime import (
      GATHER_BINDINGS, make_gather_module,
    )
    module = make_gather_module([
      ("r0", "L1_HIT", "line0", None),
      ("r1", "L2_HIT", "line1", None),
    ])
    hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
    sim = Simulator(
      hw, SimConfig(fidelity="full_memory", max_cycles=10000,
                    memory_trace=True),
      enable_tracer=True,
    )
    result = sim.run(module, input_bindings=GATHER_BINDINGS)
    assert result.completed, result.reason
    assert sim.tracer is not None
    sim.tracer.assert_well_formed()
    events = _events_of(sim)
    thread_names = {
      (e["pid"], e["tid"]): e["args"]["name"]
      for e in events if e.get("name") == "thread_name"
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
      "transaction_id", "flow_id", "bytes", "accepted_cycle",
      "completion_cycle", "source_space", "destination_space",
    }
    leg_names = {"hbm_read", "hbm_write", "global_dma", "noc_request",
                 "noc_response", "l2_read", "l2_write", "local_dma",
                 "l1_read", "l1_write", "l1_cache_lookup", "l2_cache_lookup",
                 "l1_cache_fill", "l2_cache_fill"}
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
    summary = [
      e for e in events
      if e.get("args", {}).get("summary_kind") == "group_transfer"
    ]
    if summary:
      fid = summary[0]["args"]["flow_id"]
      leg_threads = {e["tid"] for e in events
                     if e.get("ph") == "X" and e["name"] in leg_names
                     and e.get("args", {}).get("flow_id") == fid}
      assert len(leg_threads) >= 2, leg_threads


class TestDualContextFixtureWellFormed:
  def test_dual_context_fixture_well_formed(self):
    """The dual-context gather example produces a well-formed trace via
    the CLI pipeline (lanes correct, flows closed, no dup counters)."""
    repo = "/home/yongxiy/Desktop/nexus"
    proc = subprocess.run(
      ["bash", f"{repo}/examples/run.sh",
       "gather-matmul-4tiles-2contexts",
       "--memory-trace",
       "--trace-json", "/tmp/nexus-memory.json"],
      capture_output=True, text=True, cwd=repo, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    with open("/tmp/nexus-memory.json") as f:
      trace = json.load(f)
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
    pname = {e["pid"]: e["args"]["name"]
             for e in events if e.get("name") == "process_name"}
    tile_pids = {pid for pid, n in pname.items() if n.startswith("Tile")}
    tg_pid = next((pid for pid, n in pname.items()
                   if n == "TileGroup"), None)
    for e in events:
      if e.get("name") == "l1_allocated_bytes":
        assert e["pid"] in tile_pids, e
      if e.get("name") == "l2_allocated_bytes" and tg_pid is not None:
        assert e["pid"] == tg_pid, e
    thread_names = {
      (e["pid"], e["tid"]): e["args"]["name"]
      for e in events if e.get("name") == "thread_name"
    }
    assert "GroupDMA" not in thread_names.values()
    summaries = [
      e for e in events
      if e.get("ph") == "X"
      and e.get("args", {}).get("summary_kind") == "group_transfer"
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
      for previous, current in zip(ordered, ordered[1:], strict=False):
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
  def test_admission_instants_on_scheduler_lane(self):
    """l2_admission_wait scenario: admission instants on the Scheduler:L2
    lane and phase_aggregate carries expected/seen counts."""
    repo = "/home/yongxiy/Desktop/nexus"
    proc = subprocess.run(
      ["bash", f"{repo}/examples/run.sh", "l2-admission-wait",
       "--trace-json", "/tmp/nexus-admission.json"],
      capture_output=True, text=True, cwd=repo, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    with open("/tmp/nexus-admission.json") as f:
      trace = json.load(f)
    events = trace["traceEvents"]
    sched = {e["tid"]: e["args"]["name"] for e in events
             if e.get("name") == "thread_name"
             and e["args"]["name"] == "Scheduler:L2"}
    assert sched, "Scheduler:L2 thread not registered"
    sched_tid = next(iter(sched))
    admission_names = {
      "context_admission_wait", "context_admission_retry",
      "context_first_action",
    }
    on_lane = {e["name"] for e in events
               if e.get("ph") == "i" and e.get("tid") == sched_tid}
    assert admission_names <= on_lane, admission_names - on_lane
    aggregates = [e for e in events
                  if e.get("ph") == "i" and e.get("name") == "phase_aggregate"]
    assert aggregates, "no phase_aggregate instant"
    for e in aggregates:
      assert "expected" in e["args"] and "seen" in e["args"]
