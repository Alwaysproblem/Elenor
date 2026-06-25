"""Tests for the ELENOR pipeline validator.

Run with:  python -m pytest pipeline_validator/tests/  (or: pytest)
"""

from __future__ import annotations

from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.ir import (
    RegionOp,
    TileOp,
    make_matmul_region,
    make_matmul_tile_program,
    make_paged_attention_tile_program,
    make_stream_pipeline_tile_program,
    make_tiled_matmul_region,
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
    TiledMatmulWorkload,
)

# ---------------------------------------------------------------------------
# Stream Queue unit tests
# ---------------------------------------------------------------------------


def make_queue(depth=3, producers=(0, ), consumers=(1, ), **kw) -> StreamQueue:
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
                       producers=(0, ),
                       consumers=(1, ),
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


# ---------------------------------------------------------------------------
# Tile / Region Program IR tests
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

    def test_matmul_region(self):
        r = make_matmul_region()
        ops = [i.op for i in r.insts]
        assert RegionOp.REGION_BEGIN in ops
        assert RegionOp.DISPATCH_STAGE in ops
        assert RegionOp.REGION_END in ops

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

    def test_tiled_matmul_region(self):
        r = make_tiled_matmul_region(num_k_chunks=4)
        ops = [i.op for i in r.insts]
        assert RegionOp.REGION_BEGIN in ops
        assert RegionOp.DISPATCH_STAGE in ops
        assert RegionOp.REGION_END in ops


# ---------------------------------------------------------------------------
# End-to-end simulation tests
# ---------------------------------------------------------------------------


class TestSimulation:

    def _run(self, wl, **hw_overrides):
        hw = HardwareConfig().with_overrides(**hw_overrides)
        sim = Simulator(hw, SimConfig(max_cycles=200_000))
        return sim.run(wl.region)

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
        result = sim.run(wl.region)
        rep = build_report(wl, result)
        # at minimum completion + credit invariant must pass
        completion = next(c for c in rep.checks
                          if c["check"] == "region_completed")
        assert completion["pass"]

    def test_report_text_renderable(self):
        wl = MatmulWorkload()
        sim = Simulator(HardwareConfig(), SimConfig(max_cycles=50_000))
        result = sim.run(wl.region)
        rep = build_report(wl, result)
        text = report_to_text(rep)
        assert "Workload: matmul" in text
        assert "Checks:" in text


class TestTracer:
    """Tests for the Perfetto/Chrome trace output."""

    def test_trace_has_engine_slices(self):
        from pipeline_validator.simulator import Simulator
        wl = MatmulWorkload()
        sim = Simulator(HardwareConfig(),
                        SimConfig(max_cycles=50_000),
                        enable_tracer=True)
        result = sim.run(wl.region)
        assert result.tracer is not None
        import json
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
        from pipeline_validator.simulator import Simulator
        from pipeline_validator.trace import trace_to_html
        wl = PagedAttentionWorkload()
        sim = Simulator(HardwareConfig(),
                        SimConfig(max_cycles=50_000),
                        enable_tracer=True)
        result = sim.run(wl.region)
        html = trace_to_html(result.tracer)
        assert "<html>" in html
        assert "traceEvents" in html or "TRACE" in html
        # should contain engine job data
        assert "BOA" in html or "MFE" in html

    def test_trace_counters_present(self):
        from pipeline_validator.simulator import Simulator
        wl = AttentionWorkload()
        sim = Simulator(HardwareConfig(),
                        SimConfig(max_cycles=50_000),
                        enable_tracer=True)
        result = sim.run(wl.region)
        import json
        data = json.loads(result.tracer.to_chrome_json())
        events = data["traceEvents"]
        # stream queue counters should be present (ph=C)
        counters = [e for e in events if e["ph"] == "C"]
        assert len(counters) > 0
        counter_names = set()
        for c in counters:
            counter_names.update(c["args"].keys())
        assert "occupancy" in counter_names or "credit_available" in counter_names
