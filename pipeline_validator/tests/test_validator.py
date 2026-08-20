"""Tests for the ELENOR pipeline validator.

Run with:  python -m pytest pipeline_validator/tests/  (or: pytest)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.exceptions import VerifyException

from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.dialects.elenor import (
  NestAllocOp,
  NestAwaitOp,
  NestBarrierOp,
  NestBuffer,
  NestCollectiveOp,
  NestContextOp,
  NestDispatchOp,
  NestDMAStoreOp,
  NestEvent,
  NestGlobalMemref,
  NestPrefetchOp,
  NestReleaseOp,
  NestReturnOp,
  NestTaskRangeOp,
  NexusAwaitOp,
  NexusProgramOp,
  NexusReturnOp,
  NexusSubmitContextOp,
  TaskRange,
  TileAwaitOp,
  TileBoaOp,
  TileEvent,
  TileEvuOp,
  TileLoadOp,
  TilePowOp,
  TileProgramDefOp,
  TileReturnOp,
  TileSignalOp,
  TileStoreOp,
)
from pipeline_validator.report import build_report, report_to_text
from pipeline_validator.simulator import Simulator
from pipeline_validator.stream_queue import EOSPolicy, StreamQueue, StreamToken
from pipeline_validator.workload_builders import (
  make_identity_tile_program,
  make_pow_task,
  make_pow_tile_program,
)
from pipeline_validator.workload_ir import (
  parse_workload_ir,
  print_workload_ir,
  verify_workload_ir,
)
from pipeline_validator.workloads import ALL_WORKLOADS, PowWorkload

# ---------------------------------------------------------------------------
# xDSL workload IR tests
# ---------------------------------------------------------------------------


class TestXDSLIR:
  def _assert_verify_failure(self, module: ModuleOp, message: str) -> None:
    with pytest.raises(VerifyException, match=message):
      verify_workload_ir(module)

  def test_pow_workload_round_trip_uses_function_calls(self):
    """PowWorkload IR round-trips in the function-call dialect."""
    text = print_workload_ir(PowWorkload().module)
    for fragment in ("nest.alloc", "nest.dispatch.tasks.async", "depends_on", "tile.signal"):
      assert fragment in text

    reparsed = parse_workload_ir(text, source_name="<pow>")
    verify_workload_ir(reparsed)
    assert print_workload_ir(reparsed) == text

  def test_function_call_op_builders_verify(self):
    """Every function-call operation can participate in a verified module."""
    load = TileLoadOp(bytes_total=64, tag="load")
    pow_op = TilePowOp(bytes_total=64, exponent=2, pow_ops=32, tag="pow")
    store = TileStoreOp(bytes_total=64, tag="store")
    evu = TileEvuOp(op_name="relu", evu_ops=16, tag="evu")
    boa = TileBoaOp(op_name="matmul", m=1, n=1, k=1, boa_ops=2, tag="boa")
    program = TileProgramDefOp(
      "all_ops",
      [
        load,
        TileAwaitOp([load.result]),
        TileSignalOp("input_released"),
        pow_op,
        TileAwaitOp([pow_op.result]),
        evu,
        TileAwaitOp([evu.result]),
        boa,
        TileAwaitOp([boa.result]),
        store,
        TileAwaitOp([store.result]),
        TileSignalOp("output_ready"),
        TileReturnOp(),
      ],
    )

    buffer = NestAllocOp(slot="l2_buf", bytes_total=256)
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    prefetch = NestPrefetchOp(buffer=buffer.result, bytes_total=256, tag="prefetch")
    dispatch = NestDispatchOp(
      "all_ops",
      tasks.result,
      buffer.result,
      buffer.result,
      "grid_done",
      "input_released",
      "output_ready",
      depends_on=[prefetch.result],
    )
    collective = NestCollectiveOp("reduce", bytes_total=256, participant_mask=1, tag="collective")
    dma_store = NestDMAStoreOp(
      buffer=buffer.result,
      bytes_total=256,
      tag="store_done",
      depends_on=[dispatch.output_ready],
    )
    release = NestReleaseOp(buffer.result, depends_on=[dma_store.result])
    module = ModuleOp(
      [
        program,
        NestContextOp(
          "all_ops_context",
          [
            buffer,
            tasks,
            prefetch,
            dispatch,
            collective,
            dma_store,
            release,
            NestAwaitOp([dispatch.grid_done, collective.result, dma_store.result]),
            NestBarrierOp(),
            NestReturnOp(),
          ],
          placement=1,
        ),
      ]
    )

    verify_workload_ir(module)
    assert isinstance(buffer.result.type, NestBuffer)
    assert isinstance(tasks.result.type, TaskRange)
    assert isinstance(prefetch.result.type, NestEvent)
    assert isinstance(load.result.type, TileEvent)
    assert isinstance(make_pow_task(), ModuleOp)
    assert isinstance(make_pow_tile_program(), TileProgramDefOp)
    assert isinstance(make_identity_tile_program(), TileProgramDefOp)
    assert ALL_WORKLOADS == [PowWorkload]

  def test_verifier_rejects_unknown_program_symbol(self):
    buffer = NestAllocOp("l2_buf", 256)
    tasks = NestTaskRangeOp(0, 1)
    dispatch = NestDispatchOp(
      "missing_program",
      tasks.result,
      buffer.result,
      buffer.result,
      "grid_done",
      "input_released",
      "output_ready",
    )
    module = ModuleOp(
      [
        NestContextOp(
          "unknown_program",
          [buffer, tasks, dispatch, NestReturnOp()],
          placement=1,
        )
      ]
    )
    self._assert_verify_failure(module, "dispatch references unknown tile program '@missing_program'")

  def test_verifier_rejects_undefined_event_in_await(self):
    buffer = NestAllocOp("l2_buf", 256)
    later_prefetch = NestPrefetchOp(buffer.result, 256, "defined_later")
    module = ModuleOp(
      [
        NestContextOp(
          "undefined_await",
          [buffer, NestAwaitOp([later_prefetch.result]), later_prefetch, NestReturnOp()],
          placement=1,
        )
      ]
    )
    self._assert_verify_failure(module, "nest.await references undefined event 'defined_later'")

  def test_verifier_rejects_duplicate_event_tag(self):
    buffer = NestAllocOp("l2_buf", 256)
    first = NestPrefetchOp(buffer.result, 256, "duplicate")
    second = NestPrefetchOp(buffer.result, 256, "duplicate")
    module = ModuleOp(
      [
        NestContextOp(
          "duplicate_event",
          [buffer, first, second, NestReturnOp()],
          placement=1,
        )
      ]
    )
    self._assert_verify_failure(module, "duplicate event tag 'duplicate'")

  def test_dispatch_context_attribute_round_trip(self):
    """``context = N`` prints, parses, and round-trips; absent when not pinned."""
    prog = make_identity_tile_program()
    buffer = NestAllocOp(slot="l2_buf", bytes_total=256)
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    dispatch = NestDispatchOp(
      prog.sym_name.data,
      tasks.result,
      buffer.result,
      buffer.result,
      "grid_done",
      "input_released",
      "output_ready",
      context_id=1,
    )
    module = ModuleOp([
      prog,
      NestContextOp(
        "pinned_ctx",
        [buffer, tasks, dispatch, NestAwaitOp([dispatch.grid_done]), NestReturnOp()],
        placement=1,
      ),
    ])
    verify_workload_ir(module)
    text = print_workload_ir(module)
    assert "context = 1" in text
    reparsed = parse_workload_ir(text, source_name="<pinned>")
    verify_workload_ir(reparsed)
    found = False
    for op in reparsed.ops:
      if isinstance(op, NestContextOp):
        for body_op in op.body.blocks[0].ops:
          if isinstance(body_op, NestDispatchOp):
            assert body_op.context_id is not None
            assert int(body_op.context_id.value.data) == 1
            found = True
    assert found, "no NestDispatchOp found in reparsed context body"
    # Unpinned dispatch must NOT print the attribute.
    assert "context = " not in print_workload_ir(PowWorkload().module)

  def test_verifier_rejects_negative_dispatch_context(self):
    prog = make_identity_tile_program()
    buffer = NestAllocOp(slot="l2_buf", bytes_total=256)
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    dispatch = NestDispatchOp(
      prog.sym_name.data,
      tasks.result,
      buffer.result,
      buffer.result,
      "grid_done",
      "input_released",
      "output_ready",
      context_id=-1,
    )
    module = ModuleOp([
      prog,
      NestContextOp(
        "neg_ctx",
        [buffer, tasks, dispatch, NestReturnOp()],
        placement=1,
      ),
    ])
    self._assert_verify_failure(module, "dispatch context must be >= 0")

  def test_context_level_context_attribute_round_trip(self):
    """``context = N`` on ``nest.context`` (device slot pin) prints, parses, and round-trips."""
    prog = make_identity_tile_program()
    buffer = NestAllocOp(slot="l2_buf", bytes_total=256)
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    dispatch = NestDispatchOp(
      prog.sym_name.data,
      tasks.result,
      buffer.result,
      buffer.result,
      "grid_done",
      "input_released",
      "output_ready",
    )
    module = ModuleOp([
      prog,
      NestContextOp(
        "ctx_default_pin",
        [buffer, tasks, dispatch, NestAwaitOp([dispatch.grid_done]), NestReturnOp()],
        placement=1,
        context_id=1,
      ),
    ])
    verify_workload_ir(module)
    text = print_workload_ir(module)
    assert "context = 1" in text
    reparsed = parse_workload_ir(text, source_name="<ctx-pin>")
    verify_workload_ir(reparsed)
    ctx_op = next(op for op in reparsed.ops if isinstance(op, NestContextOp))
    assert ctx_op.context_id is not None
    assert int(ctx_op.context_id.value.data) == 1
    # Unpinned context must NOT print the attribute.
    assert "context = " not in print_workload_ir(PowWorkload().module)

  def test_verifier_rejects_negative_context_level_context(self):
    prog = make_identity_tile_program()
    buffer = NestAllocOp(slot="l2_buf", bytes_total=256)
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    dispatch = NestDispatchOp(
      prog.sym_name.data,
      tasks.result,
      buffer.result,
      buffer.result,
      "grid_done",
      "input_released",
      "output_ready",
    )
    module = ModuleOp([
      prog,
      NestContextOp(
        "neg_ctx_pin",
        [buffer, tasks, dispatch, NestReturnOp()],
        placement=1,
        context_id=-1,
      ),
    ])
    self._assert_verify_failure(module, "nest.context context must be >= 0")

  def test_nexus_program_round_trip(self):
    """nexus.program with submit/await/return round-trips in custom assembly."""
    prog = make_identity_tile_program()
    ctxs = []
    for i in range(2):
      buf = NestAllocOp(slot="l2_buf", bytes_total=256)
      tasks = NestTaskRangeOp(from_task=0, to_task=1)
      disp = NestDispatchOp(
        prog.sym_name.data, tasks.result, buf.result, buf.result,
        f"ev_grid_c{i}", "", "",
      )
      ctxs.append(NestContextOp(
        f"ctx{i}",
        [buf, tasks, disp, NestAwaitOp([disp.grid_done]), NestReturnOp()],
        placement=1, context_id=i,
      ))
    sub0 = NexusSubmitContextOp("ctx0", "done0")
    sub1 = NexusSubmitContextOp("ctx1", "done1")
    program = NexusProgramOp(
      "run_model",
      [sub0, sub1, NexusAwaitOp([sub0.result, sub1.result]), NexusReturnOp()],
      arg_types=[NestGlobalMemref.of([4, 128, 128], "bf16")],
    )
    module = ModuleOp([prog, *ctxs, program])
    verify_workload_ir(module)
    text = print_workload_ir(module)
    assert "nexus.program" in text
    assert "nexus.submit_context.async" in text
    assert '!nexus.event<"' in text
    assert "!nest.global_memref<4x128x128xbf16>" in text
    reparsed = parse_workload_ir(text)
    verify_workload_ir(reparsed)
    assert print_workload_ir(reparsed) == text

  def test_verifier_rejects_unknown_submit_context(self):
    """submit_context referencing an undefined nest.context is rejected."""
    prog = make_identity_tile_program()
    buf = NestAllocOp(slot="l2_buf", bytes_total=256)
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    disp = NestDispatchOp(
      prog.sym_name.data, tasks.result, buf.result, buf.result, "ev0", "", "",
    )
    ctx0 = NestContextOp(
      "ctx0", [buf, tasks, disp, NestAwaitOp([disp.grid_done]), NestReturnOp()],
      placement=1, context_id=0,
    )
    program = NexusProgramOp(
      "bad", [NexusSubmitContextOp("missing", "done0"), NexusReturnOp()],
    )
    module = ModuleOp([prog, ctx0, program])
    self._assert_verify_failure(module, "submit_context references unknown nest.context '@missing'")

  def test_verifier_rejects_undefined_nexus_await(self):
    """nexus.await referencing an event not yet submitted is rejected."""
    prog = make_identity_tile_program()
    sub = NexusSubmitContextOp("ctx0", "done0")
    buf = NestAllocOp(slot="l2_buf", bytes_total=256)
    tasks = NestTaskRangeOp(from_task=0, to_task=1)
    disp = NestDispatchOp(
      prog.sym_name.data, tasks.result, buf.result, buf.result, "ev0", "", "",
    )
    ctx0 = NestContextOp(
      "ctx0", [buf, tasks, disp, NestAwaitOp([disp.grid_done]), NestReturnOp()],
      placement=1, context_id=0,
    )
    program = NexusProgramOp(
      "bad",
      [NexusAwaitOp([sub.result]), sub, NexusReturnOp()],
    )
    module = ModuleOp([prog, ctx0, program])
    self._assert_verify_failure(module, "nexus.await references undefined event 'done0'")

class TestExternalIRCLI:
  def _run_cli(self, *args: str):
    return subprocess.run(
      [sys.executable, "-m", "pipeline_validator", *args],
      cwd=Path(__file__).resolve().parents[2],
      capture_output=True,
      text=True,
    )

  def test_ir_file_success_and_print_only_mode(self, tmp_path: Path):
    input_path = tmp_path / "pow.mlir"
    trace_json = tmp_path / "trace.json"
    trace_html = tmp_path / "trace.html"
    report_path = tmp_path / "report.txt"
    input_path.write_text(print_workload_ir(PowWorkload().module), encoding="utf-8")

    run = self._run_cli(
      "--ir-file",
      str(input_path),
      "--trace-json",
      str(trace_json),
      "--trace-html",
      str(trace_html),
      "--report",
      str(report_path),
    )
    assert run.returncode == 0, run.stderr
    data = json.loads(trace_json.read_text(encoding="utf-8"))
    assert data["traceEvents"]
    assert "traceEvents" in trace_html.read_text(encoding="utf-8")
    report = report_path.read_text(encoding="utf-8")
    assert "[PASS] task_completed" in report
    assert "[PASS] credit_invariant" in report

    trace_json.unlink()
    print_only = self._run_cli("--ir-file", str(input_path), "--print-ir", "--trace-json", str(trace_json))
    assert print_only.returncode == 0, print_only.stderr
    assert print_only.stdout == input_path.read_text(encoding="utf-8")
    assert not trace_json.exists()

  def test_ir_file_conflicts_and_load_errors(self, tmp_path: Path):
    input_path = tmp_path / "pow.mlir"
    input_path.write_text(print_workload_ir(PowWorkload().module), encoding="utf-8")

    conflict = self._run_cli("--ir-file", str(input_path), "-w", "pow")
    assert conflict.returncode == 2

    missing_trace = tmp_path / "missing.json"
    missing_report = tmp_path / "missing.txt"
    missing_path = tmp_path / "missing.mlir"
    missing = self._run_cli(
      "--ir-file",
      str(missing_path),
      "--trace-json",
      str(missing_trace),
      "--report",
      str(missing_report),
    )
    assert missing.returncode == 2
    assert "failed to load IR" in missing.stderr
    assert str(missing_path) in missing.stderr
    assert not missing_trace.exists()
    assert not missing_report.exists()

    bad_utf8 = tmp_path / "bad_utf8.mlir"
    bad_utf8.write_bytes(b"\xff\xfe")
    bad_utf8_res = self._run_cli("--ir-file", str(bad_utf8))
    assert bad_utf8_res.returncode == 2
    assert "failed to load IR" in bad_utf8_res.stderr
    assert str(bad_utf8) in bad_utf8_res.stderr

    unknown_op = tmp_path / "unknown_op.mlir"
    unknown_op.write_text(
      input_path.read_text(encoding="utf-8").replace("nest.dma.prefetch", "nest.bad_op", 1),
      encoding="utf-8",
    )
    unknown_res = self._run_cli("--ir-file", str(unknown_op))
    assert unknown_res.returncode == 2
    assert "failed to load IR" in unknown_res.stderr
    assert str(unknown_op) in unknown_res.stderr

    malformed = tmp_path / "malformed.mlir"
    malformed.write_text("not mlir\n", encoding="utf-8")
    malformed_res = self._run_cli("--ir-file", str(malformed))
    assert malformed_res.returncode == 2
    assert "failed to load IR" in malformed_res.stderr
    assert str(malformed) in malformed_res.stderr

  def test_context_mode_cli_bounds(self):
    ok = self._run_cli("-w", "pow", "--context-mode", "3", "--max-cycles", "200000")
    assert ok.returncode == 0, ok.stderr
    for bad in ("0", "9"):
      res = self._run_cli("-w", "pow", "--context-mode", bad)
      assert res.returncode == 2
      assert "--context-mode must be between 1 and 8" in res.stderr

  def test_example_mlir_runs_on_two_device_contexts(self):
    res = self._run_cli(
      "--ir-file", "examples/example.mlir",
      "--device-context-mode", "2",
      "--max-cycles", "200000",
    )
    assert res.returncode == 0, res.stderr
    assert "Completed:" in res.stdout and "True" in res.stdout

  def test_device_context_mode_cli_bounds(self):
    for bad in ("0", "9"):
      res = self._run_cli("-w", "pow", "--device-context-mode", bad)
      assert res.returncode == 2
      assert "--device-context-mode must be between 1 and 8" in res.stderr

# ---------------------------------------------------------------------------
# Stream Queue unit tests
# ---------------------------------------------------------------------------


def make_queue(depth=3, producers=(0,), consumers=(1,), **kw) -> StreamQueue:
  q = StreamQueue(
    queue_id=0, depth=depth, producers=frozenset(producers), consumers=frozenset(consumers), **kw
  )
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
    assert q.pmu.stall_cycles.get(0, 0) > 0 or q.pmu.named_cycles.get("queue_empty", 0) > 0

  def test_eos_single_producer(self):
    q = make_queue(depth=2, producers=(0,), consumers=(1,), eos_policy=EOSPolicy.SINGLE_PRODUCER)
    q.push_eos(0, 0)
    assert q.all_eos_seen

  def test_eos_all_producers(self):
    q = make_queue(depth=4, producers=(0, 1), consumers=(2, 3), eos_policy=EOSPolicy.ALL_PRODUCERS)
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
# Pow simulation tests
# ---------------------------------------------------------------------------


class TestPowSimulation:
  def test_pow_completes(self):
    hw = HardwareConfig()
    sim = Simulator(hw, SimConfig(max_cycles=200_000))
    result = sim.run(PowWorkload().module)
    assert result.completed, f"pow did not complete: {result.reason}"
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_pow_report_has_passing_checks(self):
    wl = PowWorkload()
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000))
    result = sim.run(wl.module)
    assert result.completed, result.reason
    rep = build_report(wl, result)
    failed = [c for c in rep.checks if not c["pass"]]
    assert not failed, f"pow failed checks: {failed}"

  def test_pow_report_text_renderable(self):
    wl = PowWorkload()
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000))
    result = sim.run(wl.module)
    rep = build_report(wl, result)
    text = report_to_text(rep)
    assert "Workload: pow" in text
    assert "Checks:" in text
