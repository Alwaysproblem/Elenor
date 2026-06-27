"""Tests for the ELENOR pipeline validator.

Run with:  python -m pytest pipeline_validator/tests/  (or: pytest)
"""

from __future__ import annotations

import json

import pipeline_validator as pv
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.ir import (
  GroupAction,
  GroupActionOp,
  TileGroupTask,
  TileOp,
  TileRoleBinding,
  make_attention_task,
  make_identity_tile_program,
  make_matmul_task,
  make_matmul_tile_program,
  make_paged_attention_tile_program,
  make_stream_pipeline_tile_program,
  make_tiled_matmul_pipelined_task,
  make_tiled_matmul_task,
  make_tiled_matmul_tile_program,
)
from pipeline_validator.report import build_report, report_to_text
from pipeline_validator.simulator import Simulator
from pipeline_validator.stream_queue import (
  EOSPolicy,
  StreamQueue,
  StreamToken,
)
from pipeline_validator.workloads import (
  AttentionWorkload,
  ConvReLuWorkload,
  MatmulWorkload,
  MoEWorkload,
  PagedAttentionWorkload,
  TiledMatmulPipelinedWorkload,
  TiledMatmulWorkload,
)

# ---------------------------------------------------------------------------
# Stream Queue unit tests
# ---------------------------------------------------------------------------


def make_queue(depth=3, producers=(0,), consumers=(1,), **kw) -> StreamQueue:
  q = StreamQueue(queue_id=0,
                  depth=depth,
                  producers=frozenset(producers),
                  consumers=frozenset(consumers),
                  **kw)
  q.init()
  return q


class TestStreamQueue:

  def test_credit_invariant_initial(self):
    q = make_queue()
    assert q.credit_invariant_holds()
    assert q._credit_available == 3

  def test_acquire_and_push(self):
    q = make_queue()
    assert q.acquire(0) is True
    tok = StreamToken(token_id=0, producer_id=0)
    assert q.push(tok, 1) is True
    assert q.occupancy == 1
    assert q.credit_invariant_holds()

  def test_full_backpressure(self):
    q = make_queue(depth=2)
    # fill both credits
    assert q.acquire(0)
    q.push(StreamToken(token_id=0, producer_id=0), 1)
    assert q.acquire(2)
    q.push(StreamToken(token_id=1, producer_id=0), 3)
    # third acquire must fail (backpressure)
    assert q.acquire(4) is False
    assert q.is_full

  def test_pop_release(self):
    q = make_queue()
    q.acquire(0)
    q.push(StreamToken(token_id=0, producer_id=0), 1)
    tok = q.pop(2)
    assert tok is not None
    assert tok.token_id == 0
    q.release(tok, 3)
    # credit returned
    assert q._credit_available == q.depth
    assert q.credit_invariant_holds()

  def test_empty_consumer_stall(self):
    q = make_queue()
    assert q.is_empty
    tok = q.pop(0)
    assert tok is None
    # PMU recorded stall
    assert q.pmu.stall_cycles.get(0, 0) > 0 or q.pmu.named_cycles.get(
        "queue_empty", 0) > 0

  def test_eos_single_producer(self):
    q = make_queue(depth=2,
                   producers=(0,),
                   consumers=(1,),
                   eos_policy=EOSPolicy.SINGLE_PRODUCER)
    q.push_eos(0, 0)
    assert q.all_eos_seen

  def test_eos_all_producers(self):
    q = make_queue(depth=4,
                   producers=(0, 1),
                   consumers=(2, 3),
                   eos_policy=EOSPolicy.ALL_PRODUCERS)
    q.push_eos(0, 0)
    assert not q.all_eos_seen  # only one of two producers
    q.push_eos(1, 1)
    assert q.all_eos_seen

  def test_sequence_id_monotonic(self):
    q = make_queue(depth=4)
    q.acquire(0)
    q.push(StreamToken(token_id=0, producer_id=0), 1)
    q.acquire(2)
    q.push(StreamToken(token_id=1, producer_id=0), 3)
    t0 = q.pop(4)
    t1 = q.pop(5)
    assert t1.sequence_id > t0.sequence_id

  def test_reset_reconciles_credit(self):
    q = make_queue()
    q.acquire(0)
    q.push(StreamToken(token_id=0, producer_id=0), 1)
    q.pop(2)
    # popped but not released -> credit invariant still holds (popped_unreleased counts)
    assert q.credit_invariant_holds()
    q.reset()
    assert q._credit_available == q.depth
    assert q.occupancy == 0
    assert q.credit_invariant_holds()

  def test_push_eos_enqueues_single_token_and_drains(self):
    # A single push_eos() must create exactly one FIFO token that drains
    # after pop+release, leaving occupancy 0 and the credit invariant intact.
    q = make_queue(depth=1, producers=(0,), consumers=(1,))
    q.push_eos(0, cycle=0)
    assert q.occupancy == 1
    tok = q.pop(cycle=1)
    assert tok is not None
    assert tok.is_eos
    q.release(tok, cycle=2)
    assert q.occupancy == 0
    assert q.credit_invariant_holds()


# ---------------------------------------------------------------------------
# Tile / TileGroupTask IR tests
# ---------------------------------------------------------------------------


class TestIR:

  def test_matmul_tile_program(self):
    p = make_matmul_tile_program()
    assert p.name == "matmul_tile"
    ops = [i.op for i in p.insts]
    assert TileOp.LAUNCH_MFE in ops
    assert TileOp.LAUNCH_BOA in ops
    assert TileOp.WAITALL in ops
    assert TileOp.RET in ops

  def test_stream_pipeline_tile_program(self):
    from pipeline_validator.ir import EngineDesc
    body = [EngineDesc("qk", "BOA", "matmul", {"ops": 1000})]
    p = make_stream_pipeline_tile_program(in_q=0, out_q=1, body_descs=body)
    ops = [i.op for i in p.insts]
    assert TileOp.STREAM_POP in ops
    assert TileOp.STREAM_PUSH in ops
    assert TileOp.STREAM_PUSH_EOS in ops
    # labels resolved
    assert p.label_index("loop") == 0
    assert p.label_index("done") == len(p.insts) - 2

  def test_matmul_task(self):
    t = make_matmul_task()
    ops = [a.op for a in t.actions]
    # Global DMA HBM->L2 prefetch + L2->HBM storeback
    assert ops.count(GroupActionOp.DMA_PREFETCH) == 2  # A + B
    assert ops.count(GroupActionOp.DMA_STORE) == 1  # C storeback
    assert ops.count(GroupActionOp.DISPATCH_ROLE) == 1  # single role
    # No deleted execution-layer contract symbols remain in the module:
    # every public name must avoid the old contract stem.
    import pipeline_validator.ir as ir
    leaked = [n for n in dir(ir) if "region" in n.lower()]
    assert not leaked, f"leaked symbols: {leaked}"

  def test_paged_attention_tile_program(self):
    p = make_paged_attention_tile_program()
    assert p.name == "paged_attention_tile"
    ops = [i.op for i in p.insts]
    # MFE page-stream gather (K and V pages)
    assert ops.count(TileOp.LAUNCH_MFE) >= 3  # gather_K, gather_V, store
    # two BOA matmuls (QK + PV)
    assert ops.count(TileOp.LAUNCH_BOA) == 2
    # two EVU steps (scale/mask + softmax)
    assert ops.count(TileOp.LAUNCH_EVU) == 2
    assert TileOp.WAITALL in ops
    assert TileOp.RET in ops
    # descriptors: page_stream ops for K/V gather
    assert "gather_K_pages" in p.descriptors
    assert "gather_V_pages" in p.descriptors
    assert p.descriptors["gather_K_pages"].op == "page_stream"

  def test_tiled_matmul_tile_program(self):
    num_k_chunks = 4
    p = make_tiled_matmul_tile_program(num_k_chunks=num_k_chunks)
    assert "tiled_matmul" in p.name
    ops = [i.op for i in p.insts]
    # 4 K chunks: each needs load_A + load_B (chunk 0 prefetched before
    # the loop, each chunk i prefetches chunk i+1).
    # Total MFE launches = 2*(4) loads + 4 stores = 12
    assert ops.count(TileOp.LAUNCH_MFE) == 12
    # 4 BOA accumulate launches (one per K chunk)
    assert ops.count(TileOp.LAUNCH_BOA) == 4
    assert ops.count(TileOp.RET) == 1
    # descriptors: per-chunk A/B loads + matmul + store
    assert "load_A_k0" in p.descriptors
    assert "load_A_k3" in p.descriptors
    assert "matmul_k0" in p.descriptors
    assert "matmul_k3" in p.descriptors
    # first chunk is not accumulate, later chunks are
    assert p.descriptors["matmul_k0"].params.get("accumulate") is False
    assert p.descriptors["matmul_k1"].params.get("accumulate") is True

    # Output double-buffer: the store for chunk i is fire-and-forget.
    # For chunks 0..n-2 its wait is deferred so it overlaps a later BOA
    # (the drain sits after ``launch BOA_{i+1}``).  Only the *last* chunk's
    # store is drained in the epilogue, where the wait is necessarily
    # adjacent to its launch (no further BOA to overlap) — that is the
    # expected pipeline epilogue, not a bug.
    store_wait_idx = {ins.args[0]: n
                     for n, ins in enumerate(p.insts)
                     if ins.op == TileOp.WAIT
                     and ins.args[0].startswith("e_store")}
    assert len(store_wait_idx) == 4
    # chunks 0..2 must be deferred: their wait is NOT adjacent to launch
    for i in range(num_k_chunks - 1):
      ev = f"e_store{i}"
      w = store_wait_idx[ev]
      prev = p.insts[w - 1]
      assert not (prev.op == TileOp.LAUNCH_MFE and prev.dst == ev), (
          f"store {ev} waited immediately after launch at inst {w}")
    # last chunk is drained in the epilogue (adjacency is fine there)
    assert f"e_store{num_k_chunks - 1}" in store_wait_idx

  def test_tiled_matmul_task(self):
    t = make_tiled_matmul_task(num_k_chunks=4)
    ops = [a.op for a in t.actions]
    # Global DMA HBM->L2 prefetch + L2->HBM storeback
    assert ops.count(GroupActionOp.DMA_PREFETCH) == 2  # A + B
    assert ops.count(GroupActionOp.DMA_STORE) == 1  # C storeback
    assert ops.count(GroupActionOp.DISPATCH_ROLE) == 1  # single role

  def test_tiled_matmul_pipelined_task(self):
    num_group_chunks = 4
    num_k_chunks = 4
    t = make_tiled_matmul_pipelined_task(
        num_group_chunks=num_group_chunks, num_k_chunks=num_k_chunks)
    ops = [a.op for a in t.actions]
    # Group-level IO pipeline: multiple DMA stages
    assert ops.count(GroupActionOp.DMA_PREFETCH) == num_group_chunks * 2  # A+B per chunk
    assert ops.count(GroupActionOp.DMA_STORE) == num_group_chunks  # C per chunk
    assert ops.count(GroupActionOp.DISPATCH_ROLE) == num_group_chunks  # one dispatch per chunk
    # Verify unique event IDs across chunks (no accidental reuse)
    dsts = [a.dst for a in t.actions if a.dst is not None]
    assert len(dsts) == len(set(dsts)), f"duplicate event ids: {dsts}"
    # Verify the task references the k-chunked tile program
    binding = t.role_bindings[0]
    assert "tiled_matmul" in binding.tile_program.name


  def test_attention_task_has_role_bindings(self):
    t = make_attention_task()
    assert set(t.role_bindings.keys()) == {0, 1}
    r0 = t.role_bindings[0]
    r1 = t.role_bindings[1]
    assert r0.tile_mask == 0x03
    assert r1.tile_mask == 0x0C
    assert r0.out_stream == 0
    assert r1.in_stream == 0
    # producer role pushes, consumer role pops
    p0_ops = [i.op for i in r0.tile_program.insts]
    p1_ops = [i.op for i in r1.tile_program.insts]
    assert TileOp.STREAM_PUSH in p0_ops
    assert TileOp.STREAM_POP in p1_ops
    # no region-style attributes on the task
    assert not hasattr(t, "tile_programs")
    assert not hasattr(t, "insts")

  def test_public_api_has_no_region_surface(self):
    # The deleted execution-layer contract must not leak through the
    # public API: every exported name must avoid the old contract stem
    # ("Region"/"region"), and the new task/role surface must be present.
    stems = ("region", "stage")
    leaked = [n for n in pv.__all__
             if any(s in n.lower() for s in stems)]
    assert not leaked, f"old contract leaked: {leaked}"
    for name in ("TileGroupTask", "TileRoleBinding", "TileGroupSequencer"):
      assert name in pv.__all__, f"{name} missing from public API"


# ---------------------------------------------------------------------------
# End-to-end simulation tests
# ---------------------------------------------------------------------------


class TestSimulation:

  def _run(self, wl, **hw_overrides):
    hw = HardwareConfig().with_overrides(**hw_overrides)
    sim = Simulator(hw, SimConfig(max_cycles=200_000))
    return sim.run(wl.task)

  def test_matmul_completes(self):
    result = self._run(MatmulWorkload())
    assert result.completed, f"matmul did not complete: {result.reason}"
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_tiled_matmul_completes(self):
    result = self._run(TiledMatmulWorkload())
    assert result.completed, (
        f"tiled_matmul did not complete: {result.reason}")
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_tiled_matmul_pipelined_completes(self):
    result = self._run(TiledMatmulPipelinedWorkload())
    assert result.completed, (
        f"tiled_matmul_pipelined did not complete: {result.reason}")
    assert result.cycles > 0
    assert result.credit_invariant_ok


  def test_attention_completes(self):
    result = self._run(AttentionWorkload())
    assert result.completed, f"attention did not complete: {result.reason}"
    assert result.credit_invariant_ok

  def test_moe_completes(self):
    result = self._run(MoEWorkload())
    assert result.completed, f"moe did not complete: {result.reason}"
    assert result.credit_invariant_ok

  def test_conv_relu_completes(self):
    result = self._run(ConvReLuWorkload())
    assert result.completed, f"conv_relu did not complete: {result.reason}"
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_paged_attention_completes(self):
    result = self._run(PagedAttentionWorkload())
    assert result.completed, (
        f"paged_attention did not complete: {result.reason}")
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_matmul_report_has_passing_checks(self):
    wl = MatmulWorkload()
    hw = HardwareConfig()
    sim = Simulator(hw, SimConfig(max_cycles=200_000))
    result = sim.run(wl.task)
    rep = build_report(wl, result)
    # at minimum completion + credit invariant must pass
    completion = next(c for c in rep.checks
                      if c["check"] == "task_completed")
    assert completion["pass"]

  def test_report_text_renderable(self):
    wl = MatmulWorkload()
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=50_000))
    result = sim.run(wl.task)
    rep = build_report(wl, result)
    text = report_to_text(rep)
    assert "Workload: matmul" in text
    assert "Checks:" in text


  def test_tiled_matmul_pipelined_report_has_passing_checks(self):
    wl = TiledMatmulPipelinedWorkload(num_group_chunks=4, num_k_chunks=4)
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000))
    result = sim.run(wl.task)
    rep = build_report(wl, result)
    # completion + credit invariant must pass
    completion = next(c for c in rep.checks
                      if c["check"] == "task_completed")
    assert completion["pass"]
    # multi_stage_group_io check must exist and pass
    gp = next(c for c in rep.checks
              if c["check"] == "multi_stage_group_io")
    assert gp["pass"], f"multi_stage_group_io failed: {gp}"
    assert gp["actual"] is True

  def test_stream_workloads_drain_eos_tokens(self):
    # Attention and MoE use a producer/consumer Stream Queue; after
    # completion every queue must have drained to zero occupancy with no
    # popped-unreleased tokens and an intact credit invariant.
    for wl_cls in (AttentionWorkload, MoEWorkload):
      wl = wl_cls()
      sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000))
      result = sim.run(wl.task)
      assert result.completed, f"{wl.name} did not complete: {result.reason}"
      snaps = result.group_snapshot.get("queues", {})
      assert snaps, f"{wl.name} produced no queue snapshots"
      for qid, snap in snaps.items():
        assert snap["occupancy"] == 0, (
            f"{wl.name} q{qid} occupancy={snap['occupancy']} after done")
        assert snap["popped_unreleased"] == 0, (
            f"{wl.name} q{qid} popped_unreleased={snap['popped_unreleased']}")
        assert snap["credit_invariant_holds"] is True, (
            f"{wl.name} q{qid} credit invariant broken")


# ---------------------------------------------------------------------------
# Tracer tests
# ---------------------------------------------------------------------------


class TestTracer:
  """Tests for the Perfetto/Chrome trace output."""

  def test_trace_has_engine_slices(self):
    wl = MatmulWorkload()
    sim = Simulator(HardwareConfig(),
                    SimConfig(max_cycles=50_000),
                    enable_tracer=True)
    result = sim.run(wl.task)
    assert result.tracer is not None
    data = json.loads(result.tracer.to_chrome_json())
    events = data["traceEvents"]
    # should have metadata, slice (X), and instant (i) events
    phases = {e["ph"] for e in events}
    assert "M" in phases  # metadata (process/thread names)
    assert "X" in phases  # complete slices (engine jobs)
    # engine slices should include BOA
    names = {e["name"] for e in events if e["ph"] == "X"}
    assert any("BOA" in n for n in names)

  def test_trace_html_renderable(self):
    from pipeline_validator.trace import trace_to_html
    wl = PagedAttentionWorkload()
    sim = Simulator(HardwareConfig(),
                    SimConfig(max_cycles=50_000),
                    enable_tracer=True)
    result = sim.run(wl.task)
    html = trace_to_html(result.tracer)
    assert "<html>" in html
    assert "traceEvents" in html or "TRACE" in html
    # should contain engine job data
    assert "BOA" in html or "MFE" in html

  def test_trace_counters_present(self):
    wl = AttentionWorkload()
    sim = Simulator(HardwareConfig(),
                    SimConfig(max_cycles=50_000),
                    enable_tracer=True)
    result = sim.run(wl.task)
    data = json.loads(result.tracer.to_chrome_json())
    events = data["traceEvents"]
    # stream queue counters should be present (ph=C)
    counters = [e for e in events if e["ph"] == "C"]
    assert len(counters) > 0
    counter_names = set()
    for c in counters:
      counter_names.update(c["args"].keys())
    assert "occupancy" in counter_names or "credit_available" in counter_names

  def test_trace_has_tilegroup_runtime_slices(self):
    """TileGroup task/role/Global-DMA/Collective duration bars exist."""
    task = make_group_runtime_trace_task()
    sim = Simulator(HardwareConfig(),
                    SimConfig(max_cycles=50_000),
                    enable_tracer=True)
    result = sim.run(task)
    assert result.completed, f"task did not complete: {result.reason}"
    data = json.loads(result.tracer.to_chrome_json())
    events = data["traceEvents"]
    slices = [e for e in events if e["ph"] == "X"]
    names = {e["name"] for e in slices}
    # task runtime window
    assert "task:group_runtime_trace_task" in names
    # role dispatch runtime window
    assert any(n.startswith("dispatch:role0:ev_role0:run")
               for n in names), names
    # Global DMA runtime windows
    assert "dma.prefetch:dma_prefetch0" in names
    assert "dma.store:dma_store0" in names
    # Collective runtime window
    assert "collective.reduce:coll_reduce0" in names
    # Global DMA slice carries bytes
    gdma = next(e for e in slices
                if e["name"] == "dma.prefetch:dma_prefetch0")
    assert gdma["args"]["bytes"] == 4096
    assert gdma["args"]["l2_slot"] == "l2_in0"
    # Collective slice carries bytes + participant_mask
    coll = next(e for e in slices
                if e["name"] == "collective.reduce:coll_reduce0")
    assert coll["args"]["bytes"] == 2048
    assert coll["args"]["participant_mask"] == 0x01
    # instant markers still present
    instants = {e["name"] for e in events if e["ph"] == "i"}
    assert "tile_role_dispatch" in instants
    assert "tile_role_complete" in instants
    assert "dma_complete" in instants
    assert "collective_complete" in instants
    assert "group_task_done" in instants
    # old execution-layer contract names must be absent: no exported
    # instant may carry the deleted stem.
    old_stems = ("region", "stage")
    leaked = [n for n in instants if any(s in n.lower() for s in old_stems)]
    assert not leaked, f"old instant names leaked: {leaked}"

  def test_matmul_trace_has_global_dma_slices(self):
    """matmul task emits Global DMA HBM->L2 prefetch/store bars on
    the TileGroup timeline, plus MFE load/store bars on each tile."""
    wl = MatmulWorkload()
    sim = Simulator(HardwareConfig(),
                    SimConfig(max_cycles=50_000),
                    enable_tracer=True)
    result = sim.run(wl.task)
    data = json.loads(result.tracer.to_chrome_json())
    events = data["traceEvents"]
    slices = [e for e in events if e["ph"] == "X"]
    names = {e["name"] for e in slices}
    # Global DMA prefetch A + B (HBM->L2)
    assert "dma.prefetch:gdma_prefetch_A" in names
    assert "dma.prefetch:gdma_prefetch_B" in names
    # Global DMA storeback C (L2->HBM)
    assert "dma.store:gdma_store_C" in names
    # task + role runtime windows on TileGroup timeline
    assert "task:matmul_task" in names
    assert any(n.startswith("dispatch:role0:ev_role0:run")
               for n in names), names
    # MFE load/store bars still present on tile tracks (not renamed)
    mfe = [e for e in slices if e["cat"] == "MFE"]
    mfe_names = {e["name"] for e in mfe}
    assert "MFE:load" in mfe_names
    assert "MFE:store" in mfe_names
    # no "Tile DMA" category should exist
    assert not [e for e in slices if e["cat"] == "Tile DMA"], \
        "Tile DMA category should not exist"
    # Global DMA slice carries bytes
    gdma_a = next(e for e in slices
                  if e["name"] == "dma.prefetch:gdma_prefetch_A")
    assert gdma_a["args"]["bytes"] > 0
    assert gdma_a["args"]["l2_slot"] == "l2_buf_A"
    # instant markers include tile_role_dispatch + dma_complete
    instants = {e["name"] for e in events if e["ph"] == "i"}
    assert "tile_role_dispatch" in instants
    assert "dma_complete" in instants
    assert "group_task_done" in instants
    # dma_complete instant must land on a DMA channel thread, not a
    # stale "DMA" or "Global DMA" thread — prevents thread-name regression.
    tg_pid = next(e["pid"] for e in events
                  if e.get("name") == "process_name"
                  and e.get("args", {}).get("name") == "TileGroup")
    thread_names = {
        e["args"]["name"]
        for e in events
        if e.get("name") == "thread_name" and e.get("pid") == tg_pid
    }
    assert "DMA" not in thread_names, \
        "stale 'DMA' thread_name leaked on TileGroup"
    assert "Global DMA" not in thread_names, \
        "stale 'Global DMA' thread_name leaked on TileGroup"
    assert "DMA Ch0" in thread_names
    assert "DMA Ch1" in thread_names
    for e in events:
      if e.get("name") == "dma_complete" and e.get("ph") == "i":
        assert e["cat"] in ("DMA Ch0", "DMA Ch1"), \
            f"dma_complete instant cat={e['cat']}, expected 'DMA Ch0/1'"
        assert "channel" in e["args"], \
            "dma_complete instant must carry channel arg"

  def test_tiled_matmul_trace_has_global_dma_and_mfe(self):
    """tiled_matmul task has Global DMA bars on TileGroup timeline and
    MFE load/store bars on tile tracks (MFE is NOT renamed to Tile DMA)."""
    wl = TiledMatmulWorkload()
    sim = Simulator(HardwareConfig(),
                    SimConfig(max_cycles=50_000),
                    enable_tracer=True)
    result = sim.run(wl.task)
    assert result.completed, (
        f"tiled_matmul did not complete: {result.reason}")
    data = json.loads(result.tracer.to_chrome_json())
    events = data["traceEvents"]
    slices = [e for e in events if e["ph"] == "X"]
    names = {e["name"] for e in slices}
    # Global DMA prefetch A + B (HBM->L2)
    assert "dma.prefetch:gdma_prefetch_A" in names
    assert "dma.prefetch:gdma_prefetch_B" in names
    # Global DMA storeback C (L2->HBM)
    assert "dma.store:gdma_store_C" in names
    # task + role runtime windows
    assert "task:tiled_matmul_task" in names
    assert any(n.startswith("dispatch:role0:ev_role0:run")
               for n in names), names
    # MFE load/store bars present (NOT renamed to Tile DMA)
    mfe = [e for e in slices if e["cat"] == "MFE"]
    mfe_names = {e["name"] for e in mfe}
    assert "MFE:load" in mfe_names
    assert "MFE:store" in mfe_names
    # no "Tile DMA" category should exist
    assert not [e for e in slices if e["cat"] == "Tile DMA"], \
        "Tile DMA category should not exist"

  def test_tiled_matmul_pipelined_trace_has_multi_stage_dma(self):
    """Pipelined tiled matmul task emits multiple Global DMA bars
    (one prefetch/store pair per group chunk) plus multiple role
    dispatch windows, proving the group-level IO pipeline."""
    wl = TiledMatmulPipelinedWorkload(num_group_chunks=4, num_k_chunks=4)
    sim = Simulator(HardwareConfig(),
                    SimConfig(max_cycles=200_000),
                    enable_tracer=True)
    result = sim.run(wl.task)
    assert result.completed, (
        f"tiled_matmul_pipelined did not complete: {result.reason}")
    data = json.loads(result.tracer.to_chrome_json())
    events = data["traceEvents"]
    slices = [e for e in events if e["ph"] == "X"]
    names = {e["name"] for e in slices}
    # Multiple DMA prefetch bars (one A+B pair per group chunk)
    for g in range(4):
        assert f"dma.prefetch:gdma_prefetch_A{g}" in names, (
            f"missing prefetch A{g} in {sorted(names)}")
        assert f"dma.prefetch:gdma_prefetch_B{g}" in names
        assert f"dma.store:gdma_store_C{g}" in names
    # Multiple role dispatch windows (one per group chunk)
    for g in range(4):
        assert any(n.startswith(f"dispatch:role0:ev_role_c{g}:run")
                   for n in names), (
            f"missing role dispatch for chunk {g} in {sorted(names)}")
    # Task runtime window present
    assert "task:tiled_matmul_pipelined_task" in names
    mfe = [e for e in slices if e["cat"] == "MFE"]
    mfe_names = {e["name"] for e in mfe}
    assert "MFE:load" in mfe_names
    assert "MFE:store" in mfe_names
    # ---- Overlap assertion: prove group-level IO pipeline ----
    # DMA_PREFETCH for chunk 1 must start before role dispatch for
    # chunk 0 finishes, proving HBM↔L2 DMA overlaps tile compute.
    by_name: dict[str, dict] = {e["name"]: e for e in slices}
    dma_a1 = by_name.get("dma.prefetch:gdma_prefetch_A1")
    assert dma_a1 is not None, "missing DMA prefetch A1 slice"
    role0 = next((e for e in slices
                  if e["name"].startswith("dispatch:role0:ev_role_c0:run")),
                 None)
    assert role0 is not None, "missing role0 dispatch slice"
    dma_a1_start = dma_a1["ts"]
    role0_end = role0["ts"] + role0["dur"]
    assert dma_a1_start < role0_end, (
        f"DMA prefetch A1 starts at {dma_a1_start} us, "
        f"but role0 ends at {role0_end} us — no overlap")
    # Also verify DMA prefetch B1 overlaps role0
    dma_b1 = by_name.get("dma.prefetch:gdma_prefetch_B1")
    assert dma_b1 is not None, "missing DMA prefetch B1 slice"
    assert dma_b1["ts"] < role0_end, (
        f"DMA prefetch B1 starts at {dma_b1['ts']} us, "
        f"but role0 ends at {role0_end} us — no overlap")



# ---------------------------------------------------------------------------
# Synthetic TileGroupTask for TileGroup runtime trace coverage
# ---------------------------------------------------------------------------


def make_group_runtime_trace_task() -> TileGroupTask:
  """A synthetic task exercising Global DMA, role dispatch, and Collective."""
  t = TileGroupTask(name="group_runtime_trace_task")
  t.role_bindings = {0: TileRoleBinding(role_id=0, tile_mask=0x01,
                                        tile_program=make_identity_tile_program())}
  t.actions = [
      GroupAction(GroupActionOp.DMA_PREFETCH,
                  args=("dma_prefetch0", "l2_in0", 4096),
                  dst="ev_dma0"),
      GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_dma0",)),
      GroupAction(GroupActionOp.DISPATCH_ROLE, args=(0,),
                  dst="ev_role0"),
      GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_role0",)),
      GroupAction(GroupActionOp.COLLECTIVE_RUN,
                  args=("coll_reduce0", "reduce", 2048, 0x01),
                  dst="ev_coll0"),
      GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_coll0",)),
      GroupAction(GroupActionOp.DMA_STORE,
                  args=("dma_store0", "l2_out0", 4096),
                  dst="ev_dma1"),
      GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_dma1",)),
      GroupAction(GroupActionOp.SIGNAL_EVENT,
                  args=("group_task_done",)),
  ]
  return t
