"""Tests for the ELENOR pipeline validator.

Run with:  python -m pytest pipeline_validator/tests/  (or: pytest)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipeline_validator as pv
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.ir import (
  EngineDesc,
  GroupAction,
  GroupActionOp,
  TileGroupTask,
  TileInst,
  TileOp,
  TileProgram,
  TileRoleBinding,
  make_attention_task,
  make_identity_tile_program,
  make_matmul_task,
  make_matmul_tile_program,
  make_paged_attention_tile_program,
  make_pow_tile_program,
  make_stream_pipeline_tile_program,
  make_tiled_matmul_persistent_task,
  make_tiled_matmul_persistent_tile_program,
  make_tiled_matmul_pipelined_pow_task,
  make_tiled_matmul_pipelined_task,
  make_tiled_matmul_pow_nodep_task,
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
  PowWorkload,
  TiledMatmulPipelinedPowWorkload,
  TiledMatmulPipelinedWorkload,
  TiledMatmulPowNodepWorkload,
  TiledMatmulTopWorkload,
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


  def test_conv_relu_tile_program_uses_mfe_im2col_and_boa_matmul(self):
    p = pv.make_conv_relu_tile_program()

    assert "conv" not in p.descriptors
    assert p.descriptors["im2col_window"].kind == "MFE"
    assert p.descriptors["im2col_window"].op == "im2col"
    assert p.descriptors["conv_gemm"].kind == "BOA"
    assert p.descriptors["conv_gemm"].op == "matmul"
    assert (
        p.descriptors["im2col_window"].params["k"]
        == p.descriptors["conv_gemm"].params["k"]
    )

    launches = [(inst.op, inst.dst, inst.args) for inst in p.insts]
    assert (TileOp.LAUNCH_MFE, "e2", ("im2col_window",)) in launches
    assert (TileOp.LAUNCH_BOA, "e3", ("conv_gemm",)) in launches
    assert launches.index((TileOp.LAUNCH_MFE, "e2", ("im2col_window",))) < (
        launches.index((TileOp.LAUNCH_BOA, "e3", ("conv_gemm",)))
    )

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

  def test_tiled_matmul_persistent_task(self):
    num_group_chunks = 4
    num_k_chunks = 4
    t = make_tiled_matmul_persistent_task(
        num_group_chunks=num_group_chunks, num_k_chunks=num_k_chunks)
    ops = [a.op for a in t.actions]
    # Single dispatch (persistent program handles all chunks)
    assert ops.count(GroupActionOp.DISPATCH_ROLE) == 1
    # Multiple prefetches (A+B per chunk) + multiple stores (C per chunk)
    assert ops.count(GroupActionOp.DMA_PREFETCH) == num_group_chunks * 2
    assert ops.count(GroupActionOp.DMA_STORE) == num_group_chunks
    # Verify unique event IDs
    dsts = [a.dst for a in t.actions if a.dst is not None]
    assert len(dsts) == len(set(dsts)), f"duplicate event ids: {dsts}"
    # Verify the task references the persistent tile program
    binding = t.role_bindings[0]
    assert "persistent" in binding.tile_program.name


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

  def test_tiled_matmul_top_completes(self):
    result = self._run(TiledMatmulTopWorkload())
    assert result.completed, (
        f"tiled_matmul_top did not complete: {result.reason}")
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

  def test_pow_completes(self):
    result = self._run(PowWorkload())
    assert result.completed, f"pow did not complete: {result.reason}"
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_pow_report_has_passing_checks(self):
    wl = PowWorkload()
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000))
    result = sim.run(wl.task)
    assert result.completed, result.reason
    rep = build_report(wl, result)
    failed = [c for c in rep.checks if not c["pass"]]
    assert not failed, f"pow failed checks: {failed}"

  def test_tiled_matmul_persistent_is_single_dispatch(self):
    """The persistent task dispatches exactly once (not per-chunk)."""
    t = make_tiled_matmul_persistent_task(
        num_group_chunks=4, num_k_chunks=4)
    ops = [a.op for a in t.actions]
    assert ops.count(GroupActionOp.DISPATCH_ROLE) == 1, (
        "persistent task should dispatch exactly once")
    # multiple prefetches (A+B per chunk) + multiple stores (C per chunk)
    assert ops.count(GroupActionOp.DMA_PREFETCH) == 4 * 2  # A+B per chunk
    assert ops.count(GroupActionOp.DMA_STORE) == 4  # C per chunk

  def test_tiled_matmul_persistent_tile_program_uses_bridged_events(self):
    """The persistent tile program WAITs on ev_dma_* bridged events."""
    p = make_tiled_matmul_persistent_tile_program(
        num_group_chunks=4, num_k_chunks=4)
    wait_events = []
    for ins in p.insts:
      if ins.op == TileOp.WAIT:
        wait_events.append(ins.args[0])
      elif ins.op == TileOp.WAITALL:
        wait_events.extend(ins.args)
    bridged = [e for e in wait_events if e.startswith("ev_dma_")]
    # must wait on ev_dma_A/B for chunks 0..3
    for g in range(4):
      assert f"ev_dma_A{g}" in bridged, (
          f"missing bridged WAIT ev_dma_A{g}")
      assert f"ev_dma_B{g}" in bridged, (
          f"missing bridged WAIT ev_dma_B{g}")


  def test_tiled_matmul_persistent_has_cross_chunk_load_overlap(self):
    """The persistent tile program issues chunk g+1's prologue load
    *inside* chunk g's K-chunk loop (between LAUNCH_BOA and WAIT BOA),
    not at the start of chunk g.  This proves the cross-chunk L2→L1
    load is hidden behind BOA compute, not serialized before it."""
    p = make_tiled_matmul_persistent_tile_program(
        num_group_chunks=4, num_k_chunks=4)
    insts = p.insts
    # Find the cross-chunk overlap: the WAIT ev_dma_A1 + LAUNCH_MFE
    # for g1 must appear between a LAUNCH_BOA and WAIT e_mm for g0.
    # Search for the pattern: LAUNCH_BOA ...mm3_g0... then WAIT ev_dma_A1
    found_overlap = False
    for i, ins in enumerate(insts):
      if (ins.op == TileOp.LAUNCH_BOA
              and ins.dst
              and "_g0" in ins.dst
              and "mm3" in ins.dst):
        # Last BOA of chunk g0 — check that ev_dma_A1 WAIT follows
        # before the WAIT e_mm3_g0
        for j in range(i + 1, min(i + 20, len(insts))):
          if (insts[j].op == TileOp.WAIT
                  and insts[j].args[0] == "ev_dma_A1"):
            found_overlap = True
            break
          if (insts[j].op == TileOp.WAIT
                  and "_g0" in insts[j].args[0]
                  and "mm3" in insts[j].args[0]):
            # WAIT e_mm3_g0 came before ev_dma_A1 — no overlap
            break
        break
    assert found_overlap, (
        "cross-chunk load overlap not found: ev_dma_A1 WAIT should "
        "appear between LAUNCH_BOA mm3_g0 and WAIT e_mm3_g0")

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

  def test_multi_context_trace_has_separate_uce_lanes(self):
    sim = Simulator(HardwareConfig(),
                    SimConfig(context_count=2, max_cycles=5000),
                    enable_tracer=True)
    result = sim.run(make_two_role_same_tile_task())
    assert result.completed, result.reason
    data = json.loads(result.tracer.to_chrome_json())
    events = data["traceEvents"]

    tile0_pid = next(
      e["pid"] for e in events
      if e.get("name") == "process_name"
      and e.get("args", {}).get("name") == "Tile0")
    thread_names = {
      e["tid"]: e["args"]["name"]
      for e in events
      if e.get("name") == "thread_name" and e.get("pid") == tile0_pid
    }
    assert "UCE CTX0" in thread_names.values()
    assert "UCE CTX1" in thread_names.values()

    instants = [e for e in events if e["ph"] == "i"]
    instant_names = {e["name"] for e in instants}
    assert "ctx_switch" in instant_names
    assert "uce_issue" in instant_names

    counter_names = set()
    for e in events:
      if e["ph"] == "C":
        counter_names.update(e["args"].keys())
    assert "active_context_count" in counter_names
    assert "ready_context_count" in counter_names

    ctx0_wait_intervals: list[tuple[float, float]] = []
    wait_starts: list[float] = []
    for e in events:
      if e.get("cat") != "UCE CTX0":
        continue
      if e["ph"] == "B" and e["name"].startswith("WAIT_EVENT"):
        wait_starts.append(e["ts"])
      elif e["ph"] == "E" and wait_starts:
        start = wait_starts.pop()
        ctx0_wait_intervals.append((start, e["ts"]))
    assert ctx0_wait_intervals, "missing CTX0 WAIT_EVENT slice"

    ctx1_issue_ts = [
      e["ts"] for e in instants
      if e["name"] == "uce_issue"
      and thread_names.get(e["tid"]) == "UCE CTX1"
    ]
    assert ctx1_issue_ts, "missing CTX1 uce_issue instant"
    assert any(start <= ts <= end
               for start, end in ctx0_wait_intervals
               for ts in ctx1_issue_ts), (
      "expected a CTX1 uce_issue during a CTX0 WAIT_EVENT interval")

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


  def test_conv_relu_trace_uses_mfe_im2col_before_boa_matmul(self):
    wl = ConvReLuWorkload()
    sim = Simulator(HardwareConfig(),
                    SimConfig(max_cycles=200_000),
                    enable_tracer=True)
    result = sim.run(wl.task)
    assert result.completed, f"conv_relu did not complete: {result.reason}"
    assert result.tracer is not None

    data = json.loads(result.tracer.to_chrome_json())
    slices = [
        e for e in data["traceEvents"]
        if e.get("ph") == "X" and e.get("args", {}).get("program") == "conv_relu_tile"
    ]
    names = {e["name"] for e in slices}
    assert "MFE:im2col" in names
    assert "BOA:matmul" in names
    assert "BOA:conv" not in names

    by_tile: dict[int, dict[str, list[dict]]] = {}
    for event in slices:
      if event["name"] not in {"MFE:im2col", "BOA:matmul"}:
        continue
      by_tile.setdefault(event["pid"], {}).setdefault(event["name"], []).append(event)

    boa_tiles = 0
    for tile_events in by_tile.values():
      for boa in tile_events.get("BOA:matmul", []):
        boa_tiles += 1
        assert any(
            mfe["ts"] + mfe["dur"] <= boa["ts"]
            for mfe in tile_events.get("MFE:im2col", [])
        )
    assert boa_tiles == 4

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


# ---------------------------------------------------------------------------
# UCE multi-context tests
# ---------------------------------------------------------------------------


def make_waiting_mfe_program(name="ctx_wait_mfe") -> TileProgram:
  p = TileProgram(name=name)
  p.descriptors = {
    "mfe_load": EngineDesc("mfe_load", "MFE", "load",
                            {"bytes": 128 * 1024, "ops": 0}),
  }
  p.insts = [
    TileInst(TileOp.LAUNCH_MFE, dst="e_load", args=("mfe_load",)),
    TileInst(TileOp.WAIT, args=("e_load",)),
    TileInst(TileOp.RET),
  ]
  p.resolve_labels()
  return p


def make_short_evu_program(name="ctx_short_evu") -> TileProgram:
  p = TileProgram(name=name)
  p.descriptors = {
    "evu_short": EngineDesc("evu_short", "EVU", "relu", {"ops": 16}),
  }
  p.insts = [
    TileInst(TileOp.LAUNCH_EVU, dst="e_evu", args=("evu_short",)),
    TileInst(TileOp.WAIT, args=("e_evu",)),
    TileInst(TileOp.RET),
  ]
  p.resolve_labels()
  return p


def make_two_role_same_tile_task() -> TileGroupTask:
  t = TileGroupTask(name="two_role_same_tile")
  t.role_bindings = {
    0: TileRoleBinding(role_id=0, tile_mask=0x01,
                       tile_program=make_waiting_mfe_program()),
    1: TileRoleBinding(role_id=1, tile_mask=0x01,
                       tile_program=make_short_evu_program()),
  }
  t.actions = [
    GroupAction(GroupActionOp.DISPATCH_ROLE, args=(0,), dst="ev_role0"),
    GroupAction(GroupActionOp.DISPATCH_ROLE, args=(1,), dst="ev_role1"),
    GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_role0",)),
    GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_role1",)),
    GroupAction(GroupActionOp.SIGNAL_EVENT, args=("group_task_done",)),
  ]
  return t


class TestUceContextMode:

  def test_context_count_two_overlaps_two_roles_on_same_tile(self):
    dual = Simulator(HardwareConfig(),
                     SimConfig(context_count=2, max_cycles=5000))
    dual_result = dual.run(make_two_role_same_tile_task())
    assert dual_result.completed, dual_result.reason
    assert dual_result.pmu.events.get("uce_context_switch", 0) > 0

    single = Simulator(HardwareConfig(),
                       SimConfig(context_count=1, max_cycles=5000))
    single_result = single.run(make_two_role_same_tile_task())
    assert single_result.completed, single_result.reason
    assert dual_result.cycles < single_result.cycles, (
      f"context_count=2 should overlap roles: ctx2={dual_result.cycles}, "
      f"ctx1={single_result.cycles}")

  def test_context_count_one_serializes_same_tile_roles(self):
    sim = Simulator(HardwareConfig(),
                    SimConfig(context_count=1, max_cycles=5000))
    result = sim.run(make_two_role_same_tile_task())
    assert result.completed, result.reason
    assert result.pmu.named_cycles.get("dispatch_wait", 0) > 0

  def test_context_count_validation(self):
    with pytest.raises(ValueError, match="context_count must be 1 or 2"):
      SimConfig(context_count=0)
    with pytest.raises(ValueError, match="context_count must be 1 or 2"):
      SimConfig(context_count=3)

  def test_full_memory_tile_scratchpad_snapshot_active(self):
    sim = Simulator(HardwareConfig(),
                    SimConfig(fidelity="full_memory", max_cycles=200_000))
    result = sim.run(MatmulWorkload().task)
    assert result.completed, result.reason
    tiles = result.group_snapshot["tiles"]
    assert tiles, "no tile snapshots"
    t0 = tiles[0]
    assert "l1_frame" in t0
    assert t0["l1_frame"]["state"] in ("FRAME_ACTIVE", "IDLE")
    assert "uce" in t0
    assert t0["uce"]["context_count"] == 1
    assert "contexts" in t0["uce"]


class TestMfeStreamBuffer:

  def test_mfe_stream_buffer_override_faults_page_prefetch(self):
    def make_page_stream_task(prefetch_depth: int) -> TileGroupTask:
      p = TileProgram(name="page_stream_tile")
      p.descriptors["page_stream"] = EngineDesc(
        "page_stream", "MFE", "page_stream",
        {"bytes": 8192, "num_pages": 4, "page_size": 16,
         "prefetch_depth": prefetch_depth})
      p.insts = [
        TileInst(TileOp.LAUNCH_MFE, dst="e_page", args=("page_stream",)),
        TileInst(TileOp.WAIT, args=("e_page",)),
        TileInst(TileOp.RET),
      ]
      p.resolve_labels()

      t = TileGroupTask(name="page_stream_task")
      t.streams = []
      t.role_bindings = {
        0: TileRoleBinding(role_id=0, tile_mask=0x0F, tile_program=p),
      }
      t.actions = [
        GroupAction(GroupActionOp.DISPATCH_ROLE, args=(0,), dst="ev_role0"),
        GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_role0",)),
        GroupAction(GroupActionOp.SIGNAL_EVENT, args=("group_task_done",)),
      ]
      return t

    hw = HardwareConfig().with_overrides(mfe_stream_buffer_bytes=4096)
    sim = Simulator(hw, SimConfig(max_cycles=1000))
    result = sim.run(make_page_stream_task(3))
    assert not result.completed, (
      f"expected fault for over-capacity prefetch, got completed={result.completed}")
    assert "MFE page_stream prefetch requires 6144 bytes" in result.reason, (
      f"reason should mention 6144 bytes, got: {result.reason}")

    hw = HardwareConfig().with_overrides(mfe_stream_buffer_bytes=4096)
    sim = Simulator(hw, SimConfig(max_cycles=1000))
    result = sim.run(make_page_stream_task(2))
    assert result.completed, (
      f"exact-fit prefetch should complete, got: {result.reason}")

    hw = HardwareConfig()
    sim = Simulator(hw, SimConfig(max_cycles=1000))
    result = sim.run(make_page_stream_task(3))
    assert result.completed, (
      f"default non-enforcing buffer should complete, got: {result.reason}")


# ---------------------------------------------------------------------------
# Tiled-matmul-pipelined-pow tests (two-stage: matmul + EVU pow)
# ---------------------------------------------------------------------------


class TestTiledMatmulPipelinedPow:
  """Tests for the two-stage tiled-matmul-pipelined-pow workload.

  The workload runs a pipelined tiled matmul (role 0) followed by an EVU
  elementwise pow (role 1) with a strict phase fence: the pow phase starts
  only after all matmul stores drain to HBM.  This covers:
    - End-to-end completion of a multi-role two-stage task.
    - PMU fingerprint: BOA active (matmul) + EVU active (pow) + MFE active
      (loads/stores in both phases) + multi-stage group IO.
    - Trace: pow prefetch starts after the last matmul store (phase fence),
      and EVU:pow slices appear on the tile tracks.
    - Sequencer backpressure: repeated DISPATCH_ROLE to the same tile_mask
      (pow phase) must not overwrite a running role.
  """

  def _run(self, **hw_overrides):
    hw = HardwareConfig().with_overrides(**hw_overrides)
    sim = Simulator(hw, SimConfig(max_cycles=200_000))
    return sim.run(TiledMatmulPipelinedPowWorkload().task)

  def test_completes(self):
    result = self._run()
    assert result.completed, (
        f"tiled_matmul_pipelined_pow did not complete: {result.reason}")
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_task_structure_two_roles(self):
    """The task has two role bindings and the pow phase actions."""
    t = make_tiled_matmul_pipelined_pow_task(
        num_group_chunks=4, num_k_chunks=4)
    assert len(t.role_bindings) == 2
    assert 0 in t.role_bindings and 1 in t.role_bindings
    # role 0 = matmul, role 1 = pow
    assert t.role_bindings[0].tile_program.name == "tiled_matmul_4k_tile"
    assert t.role_bindings[1].tile_program.name == "pow_4k_tile"
    # pow phase: 4 prefetches + 4 dispatches(role 1) + 4 stores + 4 drains
    ops = [a.op for a in t.actions]
    # matmul phase: 4 dispatches(role 0) + pow phase: 4 dispatches(role 1)
    assert ops.count(GroupActionOp.DISPATCH_ROLE) == 8
    # 4 matmul C stores + 4 pow output stores
    assert ops.count(GroupActionOp.DMA_STORE) == 8
    # matmul prefetches (4x2 A+B) + pow prefetches (4) = 12
    assert ops.count(GroupActionOp.DMA_PREFETCH) == 12

  def test_pow_tile_program_structure(self):
    """The pow tile program is load -> pow -> store."""
    p = make_pow_tile_program(name="pow_4k_tile",
                               chunk_bytes=128 * 128 * 2)
    assert p.name == "pow_4k_tile"
    ops = [i.op for i in p.insts]
    # launch.mfe, wait, launch.evu, wait, launch.mfe, wait, ret
    assert ops == [
        TileOp.LAUNCH_MFE, TileOp.WAIT,
        TileOp.LAUNCH_EVU, TileOp.WAIT,
        TileOp.LAUNCH_MFE, TileOp.WAIT,
        TileOp.RET,
    ]
    # descriptors
    assert "load_pow" in p.descriptors
    assert "pow_chunk" in p.descriptors
    assert "store_pow" in p.descriptors
    assert p.descriptors["pow_chunk"].kind == "EVU"
    assert p.descriptors["pow_chunk"].op == "pow"
    assert p.descriptors["pow_chunk"].params["exponent"] == 2
    assert p.descriptors["pow_chunk"].params["ops"] == 65536

  def test_report_all_checks_pass(self):
    """The report fingerprint passes: BOA + EVU + MFE + multi-stage IO."""
    wl = TiledMatmulPipelinedPowWorkload(num_group_chunks=4, num_k_chunks=4)
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000))
    result = sim.run(wl.task)
    assert result.completed, (
        f"did not complete: {result.reason}")
    rep = build_report(wl, result)
    # all checks must pass
    failed = [c for c in rep.checks if not c["pass"]]
    assert not failed, f"failed checks: {failed}"
    # verify specific checks exist and pass
    completion = next(c for c in rep.checks
                      if c["check"] == "task_completed")
    assert completion["pass"]
    gp = next(c for c in rep.checks
              if c["check"] == "multi_stage_group_io")
    assert gp["pass"]
    assert gp["actual"] is True
    evu = next(c for c in rep.checks
               if c["check"] == "evu_active_ratio")
    assert evu["pass"], f"EVU should be active (pow phase): {evu}"
    # EVU must have run (pow phase)
    assert rep.engine_active.get("EVU", 0) > 0
    # BOA must have run (matmul phase)
    assert rep.engine_active.get("BOA", 0) > 0

  def test_trace_has_pow_phase_and_fence(self):
    """Trace contains EVU:pow slices, pow role dispatches, pow prefetch/store
    bars, and the phase fence: the first pow prefetch starts after the last
    matmul store ends."""
    wl = TiledMatmulPipelinedPowWorkload(num_group_chunks=4, num_k_chunks=4)
    sim = Simulator(HardwareConfig(),
                    SimConfig(max_cycles=200_000),
                    enable_tracer=True)
    result = sim.run(wl.task)
    assert result.completed, (
        f"did not complete: {result.reason}")
    data = json.loads(result.tracer.to_chrome_json())
    events = data["traceEvents"]
    slices = [e for e in events if e["ph"] == "X"]
    names = {e["name"] for e in slices}

    # ---- matmul phase DMA bars ----
    for g in range(4):
      assert f"dma.prefetch:gdma_prefetch_A{g}" in names, names
      assert f"dma.store:gdma_store_C{g}" in names, names

    # ---- pow phase DMA bars ----
    for g in range(4):
      assert f"dma.prefetch:gdma_prefetch_pow{g}" in names, names
      assert f"dma.store:gdma_store_pow{g}" in names, names

    # ---- pow role dispatch windows (role 1) ----
    for g in range(4):
      assert any(
          n.startswith(f"dispatch:role1:ev_role_pow{g}:run")
          for n in names), (
          f"missing pow role dispatch for chunk {g} in {sorted(names)}")

    # ---- EVU:pow engine slices on tile tracks ----
    evu_slices = [e for e in slices if e["cat"] == "EVU"]
    evu_names = {e["name"] for e in evu_slices}
    assert "EVU:pow" in evu_names, (
        f"EVU:pow slice missing, got: {evu_names}")

    # ---- phase fence: pow prefetch 0 starts after matmul store C3 ends ----
    by_name = {e["name"]: e for e in slices}
    store_c3 = by_name.get("dma.store:gdma_store_C3")
    pow_prefetch0 = by_name.get("dma.prefetch:gdma_prefetch_pow0")
    assert store_c3 is not None, "missing matmul store C3 slice"
    assert pow_prefetch0 is not None, "missing pow prefetch 0 slice"
    store_c3_end = store_c3["ts"] + store_c3["dur"]
    assert pow_prefetch0["ts"] >= store_c3_end, (
        f"phase fence violated: pow prefetch 0 starts at "
        f"{pow_prefetch0['ts']} but matmul store C3 ends at "
        f"{store_c3_end}")

  def test_pow_prefetch_overlap_with_compute(self):
    """The up-front pow prefetches overlap: pow prefetch for chunk g+1
    starts before pow dispatch for chunk g finishes, proving the prefetches
    are issued up front (async DMA) while earlier chunks compute."""
    wl = TiledMatmulPipelinedPowWorkload(num_group_chunks=4, num_k_chunks=4)
    sim = Simulator(HardwareConfig(),
                    SimConfig(max_cycles=200_000),
                    enable_tracer=True)
    result = sim.run(wl.task)
    assert result.completed, f"did not complete: {result.reason}"
    data = json.loads(result.tracer.to_chrome_json())
    slices = [e for e in data["traceEvents"] if e["ph"] == "X"]
    by_name = {e["name"]: e for e in slices}
    # pow prefetch 1 must start before pow dispatch 0 finishes
    pow_prefetch1 = by_name.get("dma.prefetch:gdma_prefetch_pow1")
    assert pow_prefetch1 is not None, "missing pow prefetch 1"
    pow_dispatch0 = next(
        (e for e in slices
         if e["name"].startswith("dispatch:role1:ev_role_pow0:run")),
        None)
    assert pow_dispatch0 is not None, "missing pow dispatch 0"
    dispatch0_end = pow_dispatch0["ts"] + pow_dispatch0["dur"]
    assert pow_prefetch1["ts"] < dispatch0_end, (
        f"pow prefetch 1 starts at {pow_prefetch1['ts']} but pow dispatch 0 "
        f"ends at {dispatch0_end} — prefetches not issued up front")

  def test_sequencer_backpressure_same_mask_dispatch(self):
    """Regression: two DISPATCH_ROLE actions to the same tile_mask with no
    intervening WAIT must complete — the sequencer backpressures the second
    dispatch until the tiles from the first are done, instead of
    overwriting tile state and deadlocking."""
    t = TileGroupTask(name="test_backpressure")
    t.streams = []
    t.role_bindings = {
      0: TileRoleBinding(role_id=0, tile_mask=0x0F,
                         tile_program=make_pow_tile_program(
                             name="pow_tile",
                             chunk_bytes=128 * 128 * 2)),
    }
    t.actions = [
      GroupAction(GroupActionOp.DMA_PREFETCH,
                  args=("pref0", "l2_0", 128 * 128 * 2 * 4),
                  dst="ev_dma0"),
      GroupAction(GroupActionOp.DMA_PREFETCH,
                  args=("pref1", "l2_1", 128 * 128 * 2 * 4),
                  dst="ev_dma1"),
      GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_dma0",)),
      GroupAction(GroupActionOp.DISPATCH_ROLE, args=(0,), dst="ev_role0"),
      GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_dma1",)),
      GroupAction(GroupActionOp.DISPATCH_ROLE, args=(0,), dst="ev_role1"),
      GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_role0",)),
      GroupAction(GroupActionOp.DMA_STORE,
                  args=("store0", "l2_0", 128 * 128 * 2 * 4),
                  dst="ev_store0"),
      GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_role1",)),
      GroupAction(GroupActionOp.DMA_STORE,
                  args=("store1", "l2_1", 128 * 128 * 2 * 4),
                  dst="ev_store1"),
      GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_store0",)),
      GroupAction(GroupActionOp.WAIT_EVENT, args=("ev_store1",)),
      GroupAction(GroupActionOp.SIGNAL_EVENT, args=("group_task_done",)),
    ]
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=50_000))
    result = sim.run(t)
    assert result.completed, (
        f"backpressure test did not complete: {result.reason}")
    # both dispatches and both stores must have run
    assert result.pmu.events.get("tgs_dispatch_role", 0) == 2
    assert result.pmu.events.get("tgs_dma_store", 0) == 2
    assert result.pmu.events.get("tile_done", 0) == 8  # 2 dispatches x 4 tiles


class TestTiledMatmulPowNodep:
  """Tests for the nodep tiled-matmul + pow trace fixture workload."""

  def _run(self, enable_tracer: bool = False):
    sim = Simulator(HardwareConfig(),
                    SimConfig(max_cycles=200_000),
                    enable_tracer=enable_tracer)
    return sim.run(TiledMatmulPowNodepWorkload().task)

  def test_print_ir_matches_fixture_ignoring_whitespace(self):
    fixture = (Path(__file__).resolve().parents[2] /
               "tiled_matmul_pow_nodep.ir")
    actual = " ".join(make_tiled_matmul_pow_nodep_task().pretty_print().split())
    expected = " ".join(fixture.read_text().split())
    assert actual == expected

  def test_task_structure_and_counts(self):
    t = make_tiled_matmul_pow_nodep_task()
    assert set(t.role_bindings.keys()) == {0, 1}
    assert t.role_bindings[0].tile_program.name == "tiled_matmul_4k_tile"
    assert t.role_bindings[1].tile_program.name == "pow_4k_tile"

    ops = [a.op for a in t.actions]
    assert ops.count(GroupActionOp.DMA_PREFETCH) == 12
    assert ops.count(GroupActionOp.DISPATCH_ROLE) == 8
    assert ops.count(GroupActionOp.DMA_STORE) == 8
    assert ops.count(GroupActionOp.WAIT_EVENT) == 28
    assert len(t.actions) == 57

    assert t.actions[0].dst == "ev_dma_pow_in0"
    assert t.actions[3].dst == "ev_dma_pow_in3"
    assert t.actions[11].dst == "ev_role_pow3"
    assert t.actions[12].dst == "ev_dma_A0"
    assert t.actions[19].dst == "ev_dma_B3"
    assert t.actions[20].op is GroupActionOp.WAIT_EVENT
    assert t.actions[20].args == ("ev_dma_A0",)
    assert t.actions[22].dst == "ev_role_c0"
    assert t.actions[31].dst == "ev_role_c3"
    assert t.actions[40].op is GroupActionOp.WAIT_EVENT
    assert t.actions[40].args == ("ev_role_pow0",)
    assert t.actions[41].dst == "ev_dma_pow_out0"
    assert t.actions[-1].op is GroupActionOp.SIGNAL_EVENT

  def test_completes_and_report_checks_pass(self):
    wl = TiledMatmulPowNodepWorkload()
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000))
    result = sim.run(wl.task)
    assert result.completed, (
        f"tiled_matmul_pow_nodep did not complete: {result.reason}")
    assert result.credit_invariant_ok

    rep = build_report(wl, result)
    failed = [c for c in rep.checks if not c["pass"]]
    assert not failed, f"failed checks: {failed}"

  def test_trace_has_nodep_order(self):
    result = self._run(enable_tracer=True)
    assert result.completed, (
        f"tiled_matmul_pow_nodep did not complete: {result.reason}")

    data = json.loads(result.tracer.to_chrome_json())
    slices = [e for e in data["traceEvents"] if e["ph"] == "X"]
    names = {e["name"] for e in slices}

    for g in range(4):
      assert f"dma.prefetch:gdma_prefetch_A{g}" in names, names
      assert f"dma.prefetch:gdma_prefetch_B{g}" in names, names
      assert f"dma.prefetch:gdma_prefetch_pow{g}" in names, names
      assert f"dma.store:gdma_store_C{g}" in names, names
      assert f"dma.store:gdma_store_pow{g}" in names, names

    assert "EVU:pow" in names
    assert "BOA:matmul" in names

    by_name = {e["name"]: e for e in slices}
    pow_prefetch0 = by_name.get("dma.prefetch:gdma_prefetch_pow0")
    store_c3 = by_name.get("dma.store:gdma_store_C3")
    store_pow0 = by_name.get("dma.store:gdma_store_pow0")
    assert pow_prefetch0 is not None, "missing pow prefetch 0 slice"
    assert store_c3 is not None, "missing C3 store slice"
    assert store_pow0 is not None, "missing pow store 0 slice"
    assert pow_prefetch0["ts"] < store_c3["ts"], (
        f"pow prefetch 0 starts at {pow_prefetch0['ts']} but C3 store starts "
        f"at {store_c3['ts']}")
    assert store_c3["ts"] <= store_pow0["ts"], (
        f"C3 store starts at {store_c3['ts']} but pow store 0 starts at "
        f"{store_pow0['ts']}")

    pow_dispatch0 = next(
        (e for e in slices
         if e["name"].startswith("dispatch:role1:ev_role_pow0:run")),
        None)
    matmul_dispatch0 = next(
        (e for e in slices
         if e["name"].startswith("dispatch:role0:ev_role_c0:run")),
        None)
    assert pow_dispatch0 is not None, "missing pow dispatch 0 slice"
    assert matmul_dispatch0 is not None, "missing matmul dispatch 0 slice"
    assert pow_dispatch0["ts"] < matmul_dispatch0["ts"], (
        f"pow dispatch 0 starts at {pow_dispatch0['ts']} but matmul dispatch "
        f"0 starts at {matmul_dispatch0['ts']}")
