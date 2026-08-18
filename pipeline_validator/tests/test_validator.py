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

import pipeline_validator as pv
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.dialects.elenor import (
  AddOp,
  BOADescriptorOp,
  BranchEosOp,
  BranchOp,
  BranchPredicateOp,
  CollectiveRunOp,
  CompareOp,
  DispatchRoleOp,
  DMALoadOp,
  DMAStoreOp,
  EVUDescriptorOp,
  FenceOp,
  GroupActionLike,
  GroupBarrierOp,
  GroupDMAPrefetchOp,
  GroupDMAStoreOp,
  GroupWaitEventOp,
  InitStreamOp,
  LabelOp,
  LaunchBOAOp,
  LaunchEVUOp,
  LaunchMFEOp,
  LaunchUSEOp,
  LoadDescriptorOp,
  MFEDescriptorOp,
  MoveOp,
  NopOp,
  PatchDescriptorOp,
  ProfileBeginOp,
  ProfileEndOp,
  ReturnOp,
  SignalEventOp,
  StoreDescriptorOp,
  StreamAcquireOp,
  StreamDescOp,
  StreamEosOp,
  StreamPopOp,
  StreamPushOp,
  StreamReleaseOp,
  TileGroupTaskOp,
  TileInstructionLike,
  TileProgramOp,
  TileRoleBindingOp,
  TrapOp,
  USEDescriptorOp,
  WaitAllOp,
  WaitOp,
)
from pipeline_validator.ir_lowering import lower_workload_ir
from pipeline_validator.report import build_report, report_to_text
from pipeline_validator.simulator import Simulator
from pipeline_validator.stream_queue import EOSPolicy, StreamQueue, StreamToken
from pipeline_validator.workload_builders import (
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
from pipeline_validator.workload_ir import (
  parse_workload_ir,
  parse_workload_ir_pretty,
  print_workload_ir,
  print_workload_ir_pretty,
  verify_workload_ir,
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
# xDSL region-traversal helpers (public-surface inspection only)
# ---------------------------------------------------------------------------


def _region_ops(region) -> list:
  """Return the ops of the single block in ``region`` in source order."""
  return list(region.blocks[0].ops)


def task_of(module: ModuleOp) -> TileGroupTaskOp:
  """Return the unique TileGroupTaskOp inside a module (verified)."""
  return verify_workload_ir(module)


def task_streams(task: TileGroupTaskOp) -> list[StreamDescOp]:
  return _region_ops(task.streams)


def task_roles(task: TileGroupTaskOp) -> list[TileRoleBindingOp]:
  return _region_ops(task.roles)


def task_actions(task: TileGroupTaskOp) -> list:
  return _region_ops(task.actions)


def role_program(role: TileRoleBindingOp) -> TileProgramOp:
  ops = _region_ops(role.program)
  assert len(ops) == 1 and isinstance(ops[0], TileProgramOp)
  return ops[0]


def program_descriptors(prog: TileProgramOp) -> list:
  return _region_ops(prog.descriptors)


def program_instructions(prog: TileProgramOp) -> list:
  return _region_ops(prog.instructions)


def descriptor_map(prog: TileProgramOp) -> dict[str, tuple[str, str, dict]]:
  """name -> (engine_kind, op_name, params) for each descriptor op."""
  out: dict[str, tuple[str, str, dict]] = {}
  for d in program_descriptors(prog):
    kind = d.ENGINE_KIND
    name = d.descriptor_name.data
    op_name = d.op_name.data
    params = {k: _scalar(v) for k, v in d.params.data.items()}
    out[name] = (kind, op_name, params)
  return out


def _scalar(attr):
  from xdsl.dialects.builtin import FloatAttr, IntegerAttr, StringAttr

  if isinstance(attr, IntegerAttr):
    v = int(attr.value.data)
    # i1 bool encoding
    if str(attr.type) == "i1":
      return bool(v)
    return v
  if isinstance(attr, FloatAttr):
    return float(attr.value.data)
  if isinstance(attr, StringAttr):
    return attr.data
  raise TypeError(f"cannot decode scalar {attr!r}")


def instruction_classes(prog: TileProgramOp) -> list[type]:
  """List of concrete op classes for each (non-Label) instruction."""
  return [type(op) for op in program_instructions(prog) if not isinstance(op, LabelOp)]


def action_classes(task: TileGroupTaskOp) -> list[type]:
  return [type(op) for op in task_actions(task)]


def _role_map(task: TileGroupTaskOp) -> dict[int, TileRoleBindingOp]:
  return {int(r.role_id.value.data): r for r in task_roles(task)}


# ---------------------------------------------------------------------------
# xDSL workload IR tests
# ---------------------------------------------------------------------------


class TestXDSLIR:
  def test_minimal_multi_dot_generic_round_trip(self):
    module = ModuleOp(
      [
        TileGroupTaskOp(
          "round_trip_task",
          [],
          [
            TileRoleBindingOp(
              0,
              0x01,
              TileProgramOp(
                "round_trip_program",
                [BOADescriptorOp("desc0", "matmul", {"flag": False, "count": 0, "ratio": 3.5, "tag": "0"})],
                [ReturnOp()],
              ),
            )
          ],
          [],
        )
      ]
    )

    text = print_workload_ir(module)

    assert text.endswith("\n")
    assert text.count("elenor.runtime.tile_group_task") == 1
    assert "flag = false" in text
    assert "count = 0 : index" in text
    assert "ratio = 3.500000e+00 : f64" in text
    assert 'tag = "0"' in text
    # empty streams/actions single-block regions print as empty blocks
    assert "^bb0:" in text
    assert "^bb1:" in text

    reparsed = parse_workload_ir(text, source_name="<round-trip>")
    assert print_workload_ir(reparsed) == text

  def test_all_registered_operations_lower_exhaustively(self):
    """Build a verifier-legal module exercising every tile instruction and
    group action op, lower it, and assert each Exec opcode appears at least
    once.  LabelOp only enters the label map; DMA synthesizes a descriptor."""
    # A tile program with one of each instruction kind.  Descriptors are
    # referenced by the launch ops; stream ops use queue 0 (declared at task
    # level) and disabled (-1) where allowed.
    descriptors = [
      MFEDescriptorOp("mfe_d", "load", {"bytes": 64, "ops": 0}),
      BOADescriptorOp("boa_d", "matmul", {"m": 8, "n": 8, "k": 8, "ops": 128}),
      EVUDescriptorOp("evu_d", "relu", {"ops": 16}),
      USEDescriptorOp("use_d", "run", {"ops": 4}),
    ]
    instructions = [
      LabelOp("start"),
      NopOp(),
      MoveOp("r0", (1,)),
      AddOp("r0", (1, 1)),
      CompareOp("r0", (0,)),
      StreamPopOp(queue_id=0, destination_token="tok"),
      BranchOp("start"),
      BranchPredicateOp("start"),
      BranchEosOp("tok", "start"),
      LaunchMFEOp(descriptor="mfe_d", event="e_mfe"),
      LaunchBOAOp(descriptor="boa_d", event="e_boa"),
      LaunchEVUOp(descriptor="evu_d", event="e_evu"),
      LaunchUSEOp(descriptor="use_d", event="e_use"),
      DMALoadOp(event="e_dl"),
      DMAStoreOp(event="e_ds"),
      WaitOp(event="e_mfe"),
      WaitAllOp(events=["e_boa", "e_evu"]),
      FenceOp(),
      StreamAcquireOp(queue_id=0, destination_token="tok2"),
      StreamPushOp(queue_id=0, token_register="tok2", producer_id=0),
      StreamReleaseOp(queue_id=0, token_register="tok"),
      StreamEosOp(queue_id=0, producer_id=0),
      PatchDescriptorOp(args=("x",)),
      LoadDescriptorOp(destination="dst", args=("a",)),
      StoreDescriptorOp(args=("s",)),
      ProfileBeginOp(),
      ProfileEndOp(),
      TrapOp(),
      ReturnOp(),
    ]
    prog = TileProgramOp(name="exhaustive", descriptors=descriptors, instructions=instructions)
    role = TileRoleBindingOp(role_id=0, tile_mask=0x01, program=prog)
    stream = StreamDescOp(queue_id=0, depth=2, producer_mask=0x01, consumer_mask=0x01)
    actions = [
      InitStreamOp(queue_id=0, depth=2, producer_mask=0x01, consumer_mask=0x01),
      GroupDMAPrefetchOp(descriptor="gd", l2_slot="l2", event="ev_p"),
      GroupWaitEventOp(event="ev_p"),
      GroupDMAStoreOp(descriptor="gs", l2_slot="l2", event="ev_s"),
      GroupWaitEventOp(event="ev_s"),
      DispatchRoleOp(role_id=0, event="ev_r"),
      GroupWaitEventOp(event="ev_r"),
      CollectiveRunOp(
        descriptor="coll", collective="reduce", bytes_total=128, participant_mask=0x01, event="ev_c"
      ),
      GroupWaitEventOp(event="ev_c"),
      GroupBarrierOp(),
      SignalEventOp(event="group_task_done"),
    ]
    task = TileGroupTaskOp(name="exhaustive_task", streams=[stream], roles=[role], actions=actions)
    module = ModuleOp([task])

    lowered = lower_workload_ir(module)
    tile_ops = {i.op.value for i in lowered.role_bindings[0].tile_program.insts}
    group_ops = {a.op.value for a in lowered.actions}
    # every tile opcode literal must appear at least once
    expected_tile = {
      "nop",
      "mov",
      "add",
      "cmp",
      "br",
      "brp",
      "br_eos",
      "ret",
      "launch.boa",
      "launch.evu",
      "launch.mfe",
      "launch.use",
      "dma_load",
      "dma_store",
      "wait",
      "waitall",
      "fence",
      "stream.pop",
      "stream.push",
      "stream.acquire",
      "stream.release",
      "stream.eos",
      "patch.desc",
      "load.desc",
      "store.desc",
      "prof.begin",
      "prof.end",
      "trap",
    }
    for v in expected_tile:
      assert v in tile_ops, f"missing tile opcode {v}"
    # LabelOp does not become an instruction; it only seeds the label map
    assert "start" in lowered.role_bindings[0].tile_program.labels
    # every group opcode literal must appear at least once
    expected_group = {
      "init.stream",
      "dma.prefetch",
      "dma.store",
      "dispatch.role",
      "wait.event",
      "barrier.group",
      "collective.run",
      "signal.event",
    }
    for v in expected_group:
      assert v in group_ops, f"missing group opcode {v}"
    # DMA synthesized descriptor
    dma_insts = [
      i for i in lowered.role_bindings[0].tile_program.insts if i.op.value in ("dma_load", "dma_store")
    ]
    assert dma_insts
    for di in dma_insts:
      assert di.args[0] == "dma"

  def _basic_program(
    self, instructions: list | None = None, descriptors: list | None = None, name: str = "p"
  ) -> TileProgramOp:
    return TileProgramOp(
      name=name,
      descriptors=descriptors or [BOADescriptorOp("d", "matmul", {"m": 1, "n": 1, "k": 1, "ops": 2})],
      instructions=instructions or [ReturnOp()],
    )

  def _basic_module(
    self,
    *,
    program: TileProgramOp | None = None,
    streams: list[StreamDescOp] | None = None,
    actions: list | None = None,
    tile_mask: int = 0x01,
    role_id: int = 0,
    task_name: str = "t",
  ) -> ModuleOp:
    role = TileRoleBindingOp(role_id=role_id, tile_mask=tile_mask, program=program or self._basic_program())
    return ModuleOp(
      [
        TileGroupTaskOp(
          name=task_name,
          streams=streams or [],
          roles=[role],
          actions=actions
          or [
            DispatchRoleOp(role_id=role_id, event="ev_role0"),
            GroupWaitEventOp(event="ev_role0"),
            SignalEventOp(event="group_task_done"),
          ],
        )
      ]
    )

  def _assert_verify_failure(self, module: ModuleOp, message: str) -> None:
    with pytest.raises(VerifyException) as excinfo:
      verify_workload_ir(module)
    assert message in str(excinfo.value)
    with pytest.raises(VerifyException) as excinfo2:
      Simulator(HardwareConfig(), SimConfig(max_cycles=10)).run(module)
    assert message in str(excinfo2.value)

  def test_all_workloads_round_trip_and_simulate_equivalently(self):
    from pipeline_validator.workloads import ALL_WORKLOADS

    hw = HardwareConfig()
    sim_cfg = SimConfig(max_cycles=200_000)
    for wl_cls in ALL_WORKLOADS:
      wl = wl_cls()
      text = print_workload_ir(wl.module)
      reparsed = parse_workload_ir(text, source_name=f"<{wl.name}>")
      assert print_workload_ir(reparsed) == text

      enable_tracer = wl.name in {"attention", "tiled_matmul_pow_nodep"}
      direct = Simulator(hw, sim_cfg, enable_tracer=enable_tracer).run(wl.module)
      parsed = Simulator(hw, sim_cfg, enable_tracer=enable_tracer).run(reparsed)

      assert direct.completed == parsed.completed == True, wl.name
      assert direct.credit_invariant_ok == parsed.credit_invariant_ok == True, wl.name
      assert direct.cycles == parsed.cycles, wl.name
      assert dict(direct.pmu.events) == dict(parsed.pmu.events), wl.name
      assert dict(direct.pmu.named_cycles) == dict(parsed.pmu.named_cycles), wl.name
      assert dict(direct.pmu.stall_cycles) == dict(parsed.pmu.stall_cycles), wl.name
      if enable_tracer:
        assert direct.tracer is not None and parsed.tracer is not None
        assert direct.tracer.to_chrome_json() == parsed.tracer.to_chrome_json(), wl.name

    persistent = make_tiled_matmul_persistent_task()
    p_text = print_workload_ir(persistent)
    p_reparsed = parse_workload_ir(p_text, source_name="<persistent>")
    p_direct = Simulator(hw, sim_cfg).run(persistent)
    p_parsed = Simulator(hw, sim_cfg).run(p_reparsed)
    assert p_direct.completed and p_parsed.completed
    assert p_direct.credit_invariant_ok and p_parsed.credit_invariant_ok
    assert p_direct.cycles == p_parsed.cycles == 19761
    assert dict(p_direct.pmu.events) == dict(p_parsed.pmu.events)
    assert dict(p_direct.pmu.named_cycles) == dict(p_parsed.pmu.named_cycles)
    assert dict(p_direct.pmu.stall_cycles) == dict(p_parsed.pmu.stall_cycles)

  def test_pretty_round_trip_all_workloads(self):
    from pipeline_validator.workloads import ALL_WORKLOADS

    hw = HardwareConfig()
    sim_cfg = SimConfig(max_cycles=200_000)
    for wl_cls in ALL_WORKLOADS:
      wl = wl_cls()
      text = print_workload_ir_pretty(wl.module)
      reparsed = parse_workload_ir_pretty(text, source_name=f"<{wl.name}>")
      assert print_workload_ir_pretty(reparsed) == text, wl.name

      direct = Simulator(hw, sim_cfg).run(wl.module)
      parsed = Simulator(hw, sim_cfg).run(reparsed)
      assert direct.completed == parsed.completed == True, wl.name
      assert direct.credit_invariant_ok == parsed.credit_invariant_ok == True, wl.name
      assert direct.cycles == parsed.cycles, wl.name
      assert dict(direct.pmu.events) == dict(parsed.pmu.events), wl.name
      assert dict(direct.pmu.named_cycles) == dict(parsed.pmu.named_cycles), wl.name
      assert dict(direct.pmu.stall_cycles) == dict(parsed.pmu.stall_cycles), wl.name

    persistent = make_tiled_matmul_persistent_task()
    p_text = print_workload_ir_pretty(persistent)
    p_reparsed = parse_workload_ir_pretty(p_text, source_name="<persistent>")
    assert print_workload_ir_pretty(p_reparsed) == p_text
    p_direct = Simulator(hw, sim_cfg).run(persistent)
    p_parsed = Simulator(hw, sim_cfg).run(p_reparsed)
    assert p_direct.completed and p_parsed.completed
    assert p_direct.credit_invariant_ok and p_parsed.credit_invariant_ok
    assert p_direct.cycles == p_parsed.cycles == 19761
    assert dict(p_direct.pmu.events) == dict(p_parsed.pmu.events)
    assert dict(p_direct.pmu.named_cycles) == dict(p_parsed.pmu.named_cycles)
    assert dict(p_direct.pmu.stall_cycles) == dict(p_parsed.pmu.stall_cycles)

  def test_pretty_and_generic_produce_same_module(self):
    module = make_matmul_task()
    hw = HardwareConfig()
    sim_cfg = SimConfig(max_cycles=200_000)

    from_pretty = parse_workload_ir_pretty(print_workload_ir_pretty(module), source_name="<pretty>")
    from_generic = parse_workload_ir(print_workload_ir(module), source_name="<generic>")

    pretty_res = Simulator(hw, sim_cfg).run(from_pretty)
    generic_res = Simulator(hw, sim_cfg).run(from_generic)
    assert pretty_res.completed and generic_res.completed
    assert pretty_res.cycles == generic_res.cycles
    assert dict(pretty_res.pmu.events) == dict(generic_res.pmu.events)
    assert dict(pretty_res.pmu.named_cycles) == dict(generic_res.pmu.named_cycles)
    assert dict(pretty_res.pmu.stall_cycles) == dict(generic_res.pmu.stall_cycles)

  def test_pretty_print_is_not_generic(self):
    text = print_workload_ir_pretty(make_matmul_task())
    assert '"elenor.runtime' not in text
    assert 'elenor.runtime.tile_group_task "matmul_task"' in text
    assert 'elenor.runtime.tile.launch.boa "matmul" -> "e2"' in text
    assert "comment=" in text

  def test_verifier_rejects_invalid_structure_and_references(self):
    dup_role_module = ModuleOp(
      [
        TileGroupTaskOp(
          name="dup_role",
          streams=[],
          roles=[
            TileRoleBindingOp(role_id=0, tile_mask=0x01, program=self._basic_program(name="p0")),
            TileRoleBindingOp(role_id=0, tile_mask=0x02, program=self._basic_program(name="p1")),
          ],
          actions=[],
        )
      ]
    )
    self._assert_verify_failure(dup_role_module, "duplicate role_id 0")

    dup_stream_module = ModuleOp(
      [
        TileGroupTaskOp(
          name="dup_stream",
          streams=[
            StreamDescOp(queue_id=0, depth=2, producer_mask=0x01, consumer_mask=0x01),
            StreamDescOp(queue_id=0, depth=2, producer_mask=0x01, consumer_mask=0x01),
          ],
          roles=[TileRoleBindingOp(role_id=0, tile_mask=0x01, program=self._basic_program())],
          actions=[],
        )
      ]
    )
    self._assert_verify_failure(dup_stream_module, "duplicate stream 0")

    dup_event_module = self._basic_module(
      actions=[DispatchRoleOp(role_id=0, event="ev0"), SignalEventOp(event="ev0")]
    )
    self._assert_verify_failure(dup_event_module, "duplicate group event 'ev0'")

    self._assert_verify_failure(self._basic_module(tile_mask=0), "tile_mask must be non-zero")

    unknown_role_module = ModuleOp(
      [
        TileGroupTaskOp(
          name="unknown_role",
          streams=[],
          roles=[TileRoleBindingOp(role_id=0, tile_mask=0x01, program=self._basic_program())],
          actions=[DispatchRoleOp(role_id=1, event="ev_role1")],
        )
      ]
    )
    self._assert_verify_failure(unknown_role_module, "unknown role_id 1")

    wrong_engine_module = self._basic_module(
      program=self._basic_program(
        instructions=[LaunchBOAOp(descriptor="d", event="e0"), WaitOp(event="e0"), ReturnOp()],
        descriptors=[EVUDescriptorOp("d", "relu", {"ops": 1})],
      )
    )
    self._assert_verify_failure(wrong_engine_module, "tile launch references unknown descriptor 'd'")

    missing_label_module = self._basic_module(
      program=self._basic_program(instructions=[BranchOp("missing"), ReturnOp()])
    )
    self._assert_verify_failure(missing_label_module, "branch target 'missing' is not defined")

    undeclared_stream_module = self._basic_module(
      program=self._basic_program(
        instructions=[StreamPopOp(queue_id=0, destination_token="tok"), ReturnOp()]
      )
    )
    self._assert_verify_failure(undeclared_stream_module, "stream 0 is not declared")

    undef_token_module = self._basic_module(
      program=self._basic_program(instructions=[BranchEosOp("tok", "done"), LabelOp("done"), ReturnOp()])
    )
    self._assert_verify_failure(undef_token_module, "token register 'tok' is not defined")

    ev_dma0_module = self._basic_module(
      program=self._basic_program(instructions=[WaitOp(event="ev_dma0"), ReturnOp()])
    )
    self._assert_verify_failure(ev_dma0_module, "wait references unknown event ev_dma0")

    unresolved_dma_module = self._basic_module(
      program=self._basic_program(instructions=[WaitOp(event="ev_dma_missing"), ReturnOp()])
    )
    self._assert_verify_failure(unresolved_dma_module, "wait references unknown event ev_dma_missing")

    wait_before_producer_module = self._basic_module(
      actions=[
        GroupWaitEventOp(event="ev_p"),
        GroupDMAPrefetchOp(descriptor="gd", l2_slot="l2", event="ev_p"),
        SignalEventOp(event="group_task_done"),
      ]
    )
    self._assert_verify_failure(wait_before_producer_module, "group wait references unknown event ev_p")

  def test_verifier_rejects_invalid_scalar_and_stream_token_state(self):
    valid = self._basic_module(
      program=self._basic_program(
        descriptors=[BOADescriptorOp("d", "matmul", {"x": 0, "m": 1, "n": 1, "k": 1, "ops": 2})],
        instructions=[LaunchBOAOp("d", "e0"), WaitOp("e0"), ReturnOp()],
      )
    )
    invalid_text = print_workload_ir(valid).replace("x = 0 : index", "x = 0 : i32")
    with pytest.raises(VerifyException) as excinfo:
      parse_workload_ir(invalid_text, source_name="<invalid-scalar>")
    assert "invalid scalar type" in str(excinfo.value)

    live_redefine_module = ModuleOp(
      [
        TileGroupTaskOp(
          name="live_redefine",
          streams=[StreamDescOp(queue_id=0, depth=2, producer_mask=0x01, consumer_mask=0x01)],
          roles=[
            TileRoleBindingOp(
              role_id=0,
              tile_mask=0x01,
              program=TileProgramOp(
                name="prog",
                descriptors=[],
                instructions=[
                  StreamAcquireOp(queue_id=0, destination_token="tok"),
                  StreamAcquireOp(queue_id=0, destination_token="tok"),
                  ReturnOp(),
                ],
              ),
            )
          ],
          actions=[DispatchRoleOp(role_id=0, event="ev_role0")],
        )
      ]
    )
    self._assert_verify_failure(live_redefine_module, "token register 'tok' is already live")

    invalid_qid_module = ModuleOp(
      [
        TileGroupTaskOp(
          name="invalid_qid",
          streams=[],
          roles=[
            TileRoleBindingOp(
              role_id=0,
              tile_mask=0x01,
              program=TileProgramOp(
                name="prog",
                descriptors=[],
                instructions=[StreamAcquireOp(queue_id=-2, destination_token="tok"), ReturnOp()],
              ),
            )
          ],
          actions=[DispatchRoleOp(role_id=0, event="ev_role0")],
        )
      ]
    )
    self._assert_verify_failure(invalid_qid_module, "stream -2 is not declared")


class TestExternalIRCLI:
  def _run_cli(self, *args: str):
    return subprocess.run(
      [sys.executable, "-m", "pipeline_validator", *args],
      cwd=Path(__file__).resolve().parents[2],
      capture_output=True,
      text=True,
    )

  def test_ir_file_success_and_print_only_mode(self, tmp_path: Path):
    input_path = tmp_path / "input.mlir"
    trace_json = tmp_path / "trace.json"
    trace_html = tmp_path / "trace.html"
    report_path = tmp_path / "report.txt"
    input_path.write_text(print_workload_ir(make_tiled_matmul_pow_nodep_task()), encoding="utf-8")

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
    assert any(e.get("name") == "group_task_done" for e in data["traceEvents"])
    html = trace_html.read_text(encoding="utf-8")
    assert "traceEvents" in html
    report = report_path.read_text(encoding="utf-8")
    assert "Workload: tiled_matmul_pipelined_task" in report
    assert "[PASS] task_completed" in report
    assert "[PASS] credit_invariant" in report

    trace_json.unlink()
    print_only = self._run_cli("--ir-file", str(input_path), "--print-ir", "--trace-json", str(trace_json))
    assert print_only.returncode == 0, print_only.stderr
    assert print_only.stdout == input_path.read_text(encoding="utf-8")
    assert not trace_json.exists()

  def test_ir_file_conflicts_and_load_errors(self, tmp_path: Path):
    ok_input = tmp_path / "ok.mlir"
    ok_input.write_text(print_workload_ir(make_matmul_task()), encoding="utf-8")

    conflict = self._run_cli("--ir-file", str(ok_input), "-w", "matmul")
    assert conflict.returncode == 2

    missing_trace = tmp_path / "missing.json"
    missing_report = tmp_path / "missing.txt"
    missing = self._run_cli(
      "--ir-file",
      str(tmp_path / "missing.mlir"),
      "--trace-json",
      str(missing_trace),
      "--report",
      str(missing_report),
    )
    assert missing.returncode == 2
    assert "failed to load IR" in missing.stderr
    assert str(tmp_path / "missing.mlir") in missing.stderr
    assert not missing_trace.exists()
    assert not missing_report.exists()

    bad_utf8 = tmp_path / "bad_utf8.mlir"
    bad_utf8.write_bytes(b"\\xff\\xfe")
    utf8_trace = tmp_path / "utf8.json"
    utf8_report = tmp_path / "utf8.txt"
    bad_utf8_res = self._run_cli(
      "--ir-file", str(bad_utf8), "--trace-json", str(utf8_trace), "--report", str(utf8_report)
    )
    assert bad_utf8_res.returncode == 2
    assert "failed to load IR" in bad_utf8_res.stderr
    assert str(bad_utf8) in bad_utf8_res.stderr
    assert not utf8_trace.exists()
    assert not utf8_report.exists()

    unknown_op = tmp_path / "unknown_op.mlir"
    unknown_op.write_text(
      print_workload_ir(make_matmul_task()).replace(
        '"elenor.runtime.tile_group_task"', '"elenor.runtime.unknown"', 1
      ),
      encoding="utf-8",
    )
    unknown_res = self._run_cli("--ir-file", str(unknown_op))
    assert unknown_res.returncode == 2
    assert "failed to load IR" in unknown_res.stderr
    assert str(unknown_op) in unknown_res.stderr

    malformed = tmp_path / "malformed.mlir"
    malformed.write_text("not mlir\\n", encoding="utf-8")
    malformed_res = self._run_cli("--ir-file", str(malformed))
    assert malformed_res.returncode == 2
    assert "failed to load IR" in malformed_res.stderr
    assert str(malformed) in malformed_res.stderr

    dup_role = tmp_path / "dup_role.mlir"
    dup_role_module = ModuleOp(
      [
        TileGroupTaskOp(
          name="dup_role",
          streams=[],
          roles=[
            TileRoleBindingOp(
              role_id=0,
              tile_mask=0x01,
              program=TileProgramOp(
                name="a",
                descriptors=[BOADescriptorOp("d0", "matmul", {"m": 1, "n": 1, "k": 1, "ops": 2})],
                instructions=[ReturnOp()],
              ),
            ),
            TileRoleBindingOp(
              role_id=0,
              tile_mask=0x02,
              program=TileProgramOp(
                name="b",
                descriptors=[BOADescriptorOp("d1", "matmul", {"m": 1, "n": 1, "k": 1, "ops": 2})],
                instructions=[ReturnOp()],
              ),
            ),
          ],
          actions=[],
        )
      ]
    )
    dup_role.write_text(print_workload_ir(dup_role_module), encoding="utf-8")
    dup_role_res = self._run_cli("--ir-file", str(dup_role))
    assert dup_role_res.returncode == 2
    assert "failed to load IR" in dup_role_res.stderr
    assert "duplicate role_id 0" in dup_role_res.stderr

  def test_ir_file_pretty_success(self, tmp_path: Path):
    input_path = tmp_path / "input_pretty.mlir"
    report_path = tmp_path / "report_pretty.txt"
    input_path.write_text(print_workload_ir_pretty(make_tiled_matmul_pow_nodep_task()), encoding="utf-8")

    run = self._run_cli("--ir-file-pretty", str(input_path), "--report", str(report_path))
    assert run.returncode == 0, run.stderr
    report = report_path.read_text(encoding="utf-8")
    assert "Workload: tiled_matmul_pipelined_task" in report
    assert "[PASS] task_completed" in report
    assert "[PASS] credit_invariant" in report

    print_only = self._run_cli("--ir-file-pretty", str(input_path), "--print-ir-pretty")
    assert print_only.returncode == 0, print_only.stderr
    assert print_only.stdout == input_path.read_text(encoding="utf-8")

  def test_ir_file_pretty_conflicts_and_errors(self, tmp_path: Path):
    ok_input = tmp_path / "ok_pretty.mlir"
    ok_input.write_text(print_workload_ir_pretty(make_matmul_task()), encoding="utf-8")

    both_ir = self._run_cli("--ir-file-pretty", str(ok_input), "--ir-file", str(ok_input))
    assert both_ir.returncode == 2

    with_workload = self._run_cli("--ir-file-pretty", str(ok_input), "-w", "matmul")
    assert with_workload.returncode == 2

    both_print = self._run_cli("--print-ir", "--print-ir-pretty", "-w", "matmul")
    assert both_print.returncode == 2

    malformed = tmp_path / "malformed_pretty.mlir"
    malformed.write_text('elenor.runtime.tile_group_task "broken" {\n', encoding="utf-8")
    malformed_res = self._run_cli("--ir-file-pretty", str(malformed))
    assert malformed_res.returncode == 2
    assert "failed to load IR" in malformed_res.stderr
    assert str(malformed) in malformed_res.stderr


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
# Tile / TileGroupTask IR tests
# ---------------------------------------------------------------------------


class TestIR:
  def test_matmul_tile_program(self):
    p = make_matmul_tile_program()
    assert p.program_name.data == "matmul_tile"
    classes = instruction_classes(p)
    assert LaunchMFEOp in classes
    assert LaunchBOAOp in classes
    assert WaitAllOp in classes
    assert ReturnOp in classes

  def test_conv_relu_tile_program_uses_mfe_im2col_and_boa_matmul(self):
    p = pv.make_conv_relu_tile_program()
    dmap = descriptor_map(p)
    assert "conv" not in dmap
    assert dmap["im2col_window"][0] == "MFE"
    assert dmap["im2col_window"][1] == "im2col"
    assert dmap["conv_gemm"][0] == "BOA"
    assert dmap["conv_gemm"][1] == "matmul"
    assert dmap["im2col_window"][2]["k"] == dmap["conv_gemm"][2]["k"]

    launches = [
      (type(op), op.event.data, op.descriptor.data)
      for op in program_instructions(p)
      if isinstance(op, (LaunchMFEOp, LaunchBOAOp, LaunchEVUOp))
    ]
    assert (LaunchMFEOp, "e2", "im2col_window") in launches
    assert (LaunchBOAOp, "e3", "conv_gemm") in launches
    mfe_idx = next(
      i for i, (t, e, d) in enumerate(launches) if t is LaunchMFEOp and e == "e2" and d == "im2col_window"
    )
    boa_idx = next(
      i for i, (t, e, d) in enumerate(launches) if t is LaunchBOAOp and e == "e3" and d == "conv_gemm"
    )
    assert mfe_idx < boa_idx

  def test_stream_pipeline_tile_program(self):
    body = [BOADescriptorOp("qk", "matmul", {"ops": 1000})]
    p = make_stream_pipeline_tile_program(in_q=0, out_q=1, body_descs=body)
    classes = instruction_classes(p)
    assert StreamPopOp in classes
    assert StreamPushOp in classes
    assert StreamEosOp in classes
    # labels present and resolved: "loop" at the head, "done" just before
    # the final StreamEosOp + ReturnOp (two trailing non-label ops).
    insts = program_instructions(p)
    label_positions = {op.label.data: i for i, op in enumerate(insts) if isinstance(op, LabelOp)}
    assert label_positions["loop"] == 0
    assert label_positions["done"] == len(insts) - 3  # eos + ret after

  def test_matmul_task(self):
    task = task_of(make_matmul_task())
    classes = action_classes(task)
    # Global DMA HBM->L2 prefetch + L2->HBM storeback
    assert classes.count(GroupDMAPrefetchOp) == 2  # A + B
    assert classes.count(GroupDMAStoreOp) == 1  # C storeback
    assert classes.count(DispatchRoleOp) == 1  # single role
    # No deleted execution-layer contract symbols remain in the module:
    # every public name must avoid the old contract stem.
    leaked = [n for n in dir(pv) if "region" in n.lower()]
    assert not leaked, f"leaked symbols: {leaked}"

  def test_paged_attention_tile_program(self):
    p = make_paged_attention_tile_program()
    assert p.program_name.data == "paged_attention_tile"
    classes = instruction_classes(p)
    # MFE page-stream gather (K and V pages)
    assert classes.count(LaunchMFEOp) >= 3  # gather_K, gather_V, store
    # two BOA matmuls (QK + PV)
    assert classes.count(LaunchBOAOp) == 2
    # two EVU steps (scale/mask + softmax)
    assert classes.count(LaunchEVUOp) == 2
    assert WaitAllOp in classes
    assert ReturnOp in classes
    # descriptors: page_stream ops for K/V gather
    dmap = descriptor_map(p)
    assert "gather_K_pages" in dmap
    assert "gather_V_pages" in dmap
    assert dmap["gather_K_pages"][1] == "page_stream"

  def test_tiled_matmul_tile_program(self):
    num_k_chunks = 4
    p = make_tiled_matmul_tile_program(num_k_chunks=num_k_chunks)
    assert "tiled_matmul" in p.program_name.data
    classes = instruction_classes(p)
    # 4 K chunks: each needs load_A + load_B (chunk 0 prefetched before
    # the loop, each chunk i prefetches chunk i+1).
    # Total MFE launches = 2*(4) loads + 4 stores = 12
    assert classes.count(LaunchMFEOp) == 12
    # 4 BOA accumulate launches (one per K chunk)
    assert classes.count(LaunchBOAOp) == 4
    assert classes.count(ReturnOp) == 1
    # descriptors: per-chunk A/B loads + matmul + store
    dmap = descriptor_map(p)
    assert "load_A_k0" in dmap
    assert "load_A_k3" in dmap
    assert "matmul_k0" in dmap
    assert "matmul_k3" in dmap
    # first chunk is not accumulate, later chunks are
    assert dmap["matmul_k0"][2].get("accumulate") is False
    assert dmap["matmul_k1"][2].get("accumulate") is True

    # Output double-buffer: the store for chunk i is fire-and-forget.
    # For chunks 0..n-2 its wait is deferred so it overlaps a later BOA
    # (the drain sits after ``launch BOA_{i+1}``).  Only the *last* chunk's
    # store is drained in the epilogue, where the wait is necessarily
    # adjacent to its launch (no further BOA to overlap) — that is the
    # expected pipeline epilogue, not a bug.
    insts = program_instructions(p)
    store_wait_idx: dict[str, int] = {}
    for n, op in enumerate(insts):
      if isinstance(op, WaitOp) and op.event.data.startswith("e_store"):
        store_wait_idx[op.event.data] = n
    assert len(store_wait_idx) == 4
    # chunks 0..2 must be deferred: their wait is NOT adjacent to launch
    for i in range(num_k_chunks - 1):
      ev = f"e_store{i}"
      w = store_wait_idx[ev]
      prev = insts[w - 1]
      assert not (isinstance(prev, LaunchMFEOp) and prev.event.data == ev), (
        f"store {ev} waited immediately after launch at inst {w}"
      )
    # last chunk is drained in the epilogue (adjacency is fine there)
    assert f"e_store{num_k_chunks - 1}" in store_wait_idx

  def test_tiled_matmul_task(self):
    task = task_of(make_tiled_matmul_task(num_k_chunks=4))
    classes = action_classes(task)
    # Global DMA HBM->L2 prefetch + L2->HBM storeback
    assert classes.count(GroupDMAPrefetchOp) == 2  # A + B
    assert classes.count(GroupDMAStoreOp) == 1  # C storeback
    assert classes.count(DispatchRoleOp) == 1  # single role

  def test_tiled_matmul_pipelined_task(self):
    num_group_chunks = 4
    num_k_chunks = 4
    task = task_of(
      make_tiled_matmul_pipelined_task(num_group_chunks=num_group_chunks, num_k_chunks=num_k_chunks)
    )
    classes = action_classes(task)
    # Group-level IO pipeline: multiple DMA stages
    assert classes.count(GroupDMAPrefetchOp) == num_group_chunks * 2  # A+B per chunk
    assert classes.count(GroupDMAStoreOp) == num_group_chunks  # C per chunk
    assert classes.count(DispatchRoleOp) == num_group_chunks  # one dispatch per chunk
    # Verify unique event IDs across chunks (no accidental reuse)
    events = [
      op.event.data
      for op in task_actions(task)
      if isinstance(
        op, (GroupDMAPrefetchOp, GroupDMAStoreOp, DispatchRoleOp, CollectiveRunOp, SignalEventOp)
      )
    ]
    assert len(events) == len(set(events)), f"duplicate event ids: {events}"
    # Verify the task references the k-chunked tile program
    rmap = _role_map(task)
    assert "tiled_matmul" in role_program(rmap[0]).program_name.data

  def test_tiled_matmul_persistent_task(self):
    num_group_chunks = 4
    num_k_chunks = 4
    task = task_of(
      make_tiled_matmul_persistent_task(num_group_chunks=num_group_chunks, num_k_chunks=num_k_chunks)
    )
    classes = action_classes(task)
    # Single dispatch (persistent program handles all chunks)
    assert classes.count(DispatchRoleOp) == 1
    # Multiple prefetches (A+B per chunk) + multiple stores (C per chunk)
    assert classes.count(GroupDMAPrefetchOp) == num_group_chunks * 2
    assert classes.count(GroupDMAStoreOp) == num_group_chunks
    # Verify unique event IDs
    events = [
      op.event.data
      for op in task_actions(task)
      if isinstance(
        op, (GroupDMAPrefetchOp, GroupDMAStoreOp, DispatchRoleOp, CollectiveRunOp, SignalEventOp)
      )
    ]
    assert len(events) == len(set(events)), f"duplicate event ids: {events}"
    # Verify the task references the persistent tile program
    rmap = _role_map(task)
    assert "persistent" in role_program(rmap[0]).program_name.data

  def test_attention_task_has_role_bindings(self):
    task = task_of(make_attention_task())
    rmap = _role_map(task)
    assert set(rmap.keys()) == {0, 1}
    r0 = rmap[0]
    r1 = rmap[1]
    assert int(r0.tile_mask.value.data) == 0x03
    assert int(r1.tile_mask.value.data) == 0x0C
    assert int(r0.out_stream.value.data) == 0
    assert int(r1.in_stream.value.data) == 0
    # producer role pushes, consumer role pops
    p0_classes = instruction_classes(role_program(r0))
    p1_classes = instruction_classes(role_program(r1))
    assert StreamPushOp in p0_classes
    assert StreamPopOp in p1_classes
    # no region-style attributes on the task
    assert not hasattr(task, "tile_programs")
    assert not hasattr(task, "insts")

  def test_public_api_has_no_region_surface(self):
    # The deleted execution-layer contract must not leak through the
    # public API: every exported name must avoid the old contract stem
    # ("Region"/"region"), and the new task/role surface must be present.
    stems = ("region", "stage")
    leaked = [n for n in pv.__all__ if any(s in n.lower() for s in stems)]
    assert not leaked, f"old contract leaked: {leaked}"
    for name in ("TileGroupTaskOp", "TileRoleBindingOp", "TileGroupSequencer"):
      assert name in pv.__all__, f"{name} missing from public API"


# ---------------------------------------------------------------------------
# End-to-end simulation tests
# ---------------------------------------------------------------------------


class TestSimulation:
  def _run(self, wl, **hw_overrides):
    hw = HardwareConfig().with_overrides(**hw_overrides)
    sim = Simulator(hw, SimConfig(max_cycles=200_000))
    return sim.run(wl.module)

  def test_matmul_completes(self):
    result = self._run(MatmulWorkload())
    assert result.completed, f"matmul did not complete: {result.reason}"
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_tiled_matmul_completes(self):
    result = self._run(TiledMatmulWorkload())
    assert result.completed, f"tiled_matmul did not complete: {result.reason}"
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_tiled_matmul_pipelined_completes(self):
    result = self._run(TiledMatmulPipelinedWorkload())
    assert result.completed, f"tiled_matmul_pipelined did not complete: {result.reason}"
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_tiled_matmul_top_completes(self):
    result = self._run(TiledMatmulTopWorkload())
    assert result.completed, f"tiled_matmul_top did not complete: {result.reason}"
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
    assert result.completed, f"paged_attention did not complete: {result.reason}"
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_matmul_report_has_passing_checks(self):
    wl = MatmulWorkload()
    hw = HardwareConfig()
    sim = Simulator(hw, SimConfig(max_cycles=200_000))
    result = sim.run(wl.module)
    rep = build_report(wl, result)
    # at minimum completion + credit invariant must pass
    completion = next(c for c in rep.checks if c["check"] == "task_completed")
    assert completion["pass"]

  def test_report_text_renderable(self):
    wl = MatmulWorkload()
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=50_000))
    result = sim.run(wl.module)
    rep = build_report(wl, result)
    text = report_to_text(rep)
    assert "Workload: matmul" in text
    assert "Checks:" in text

  def test_tiled_matmul_pipelined_report_has_passing_checks(self):
    wl = TiledMatmulPipelinedWorkload(num_group_chunks=4, num_k_chunks=4)
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000))
    result = sim.run(wl.module)
    rep = build_report(wl, result)
    # completion + credit invariant must pass
    completion = next(c for c in rep.checks if c["check"] == "task_completed")
    assert completion["pass"]
    # multi_stage_group_io check must exist and pass
    gp = next(c for c in rep.checks if c["check"] == "multi_stage_group_io")
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
    result = sim.run(wl.module)
    assert result.completed, result.reason
    rep = build_report(wl, result)
    failed = [c for c in rep.checks if not c["pass"]]
    assert not failed, f"pow failed checks: {failed}"

  def test_tiled_matmul_persistent_is_single_dispatch(self):
    """The persistent task dispatches exactly once (not per-chunk)."""
    task = task_of(make_tiled_matmul_persistent_task(num_group_chunks=4, num_k_chunks=4))
    classes = action_classes(task)
    assert classes.count(DispatchRoleOp) == 1, "persistent task should dispatch exactly once"
    # multiple prefetches (A+B per chunk) + multiple stores (C per chunk)
    assert classes.count(GroupDMAPrefetchOp) == 4 * 2  # A+B per chunk
    assert classes.count(GroupDMAStoreOp) == 4  # C per chunk

  def test_tiled_matmul_persistent_tile_program_uses_bridged_events(self):
    """The persistent tile program WAITs on ev_dma_* bridged events."""
    p = make_tiled_matmul_persistent_tile_program(num_group_chunks=4, num_k_chunks=4)
    wait_events: list[str] = []
    for op in program_instructions(p):
      if isinstance(op, WaitOp):
        wait_events.append(op.event.data)
      elif isinstance(op, WaitAllOp):
        wait_events.extend(e.data for e in op.events.data)
    bridged = [e for e in wait_events if e.startswith("ev_dma_")]
    # must wait on ev_dma_A/B for chunks 0..3
    for g in range(4):
      assert f"ev_dma_A{g}" in bridged, f"missing bridged WAIT ev_dma_A{g}"
      assert f"ev_dma_B{g}" in bridged, f"missing bridged WAIT ev_dma_B{g}"

  def test_tiled_matmul_persistent_has_cross_chunk_load_overlap(self):
    """The persistent tile program issues chunk g+1's prologue load
    *inside* chunk g's K-chunk loop (between LAUNCH_BOA and WAIT BOA),
    not at the start of chunk g.  This proves the cross-chunk L2→L1
    load is hidden behind BOA compute, not serialized before it."""
    p = make_tiled_matmul_persistent_tile_program(num_group_chunks=4, num_k_chunks=4)
    insts = program_instructions(p)
    # Find the cross-chunk overlap: the WAIT ev_dma_A1 + LAUNCH_MFE
    # for g1 must appear between a LAUNCH_BOA and WAIT e_mm for g0.
    # Search for the pattern: LAUNCH_BOA ...mm3_g0... then WAIT ev_dma_A1
    found_overlap = False
    for i, op in enumerate(insts):
      if (
        isinstance(op, LaunchBOAOp) and op.event.data and "_g0" in op.event.data and "mm3" in op.event.data
      ):
        # Last BOA of chunk g0 — check that ev_dma_A1 WAIT follows
        # before the WAIT e_mm3_g0
        for j in range(i + 1, min(i + 20, len(insts))):
          if isinstance(insts[j], WaitOp) and insts[j].event.data == "ev_dma_A1":
            found_overlap = True
            break
          if isinstance(insts[j], WaitOp) and "_g0" in insts[j].event.data and "mm3" in insts[j].event.data:
            # WAIT e_mm3_g0 came before ev_dma_A1 — no overlap
            break
        break
    assert found_overlap, (
      "cross-chunk load overlap not found: ev_dma_A1 WAIT should "
      "appear between LAUNCH_BOA mm3_g0 and WAIT e_mm3_g0"
    )

  def test_stream_workloads_drain_eos_tokens(self):
    # Attention and MoE use a producer/consumer Stream Queue; after
    # completion every queue must have drained to zero occupancy with no
    # popped-unreleased tokens and an intact credit invariant.
    for wl_cls in (AttentionWorkload, MoEWorkload):
      wl = wl_cls()
      sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000))
      result = sim.run(wl.module)
      assert result.completed, f"{wl.name} did not complete: {result.reason}"
      snaps = result.group_snapshot.get("queues", {})
      assert snaps, f"{wl.name} produced no queue snapshots"
      for qid, snap in snaps.items():
        assert snap["occupancy"] == 0, f"{wl.name} q{qid} occupancy={snap['occupancy']} after done"
        assert snap["popped_unreleased"] == 0, (
          f"{wl.name} q{qid} popped_unreleased={snap['popped_unreleased']}"
        )
        assert snap["credit_invariant_holds"] is True, f"{wl.name} q{qid} credit invariant broken"


# ---------------------------------------------------------------------------
# Tracer tests
# ---------------------------------------------------------------------------


class TestTracer:
  """Tests for the Perfetto/Chrome trace output."""

  def test_trace_has_engine_slices(self):
    wl = MatmulWorkload()
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=50_000), enable_tracer=True)
    result = sim.run(wl.module)
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
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=50_000), enable_tracer=True)
    result = sim.run(wl.module)
    html = trace_to_html(result.tracer)
    assert "<html>" in html
    assert "traceEvents" in html or "TRACE" in html
    # should contain engine job data
    assert "BOA" in html or "MFE" in html

  def test_trace_counters_present(self):
    wl = AttentionWorkload()
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=50_000), enable_tracer=True)
    result = sim.run(wl.module)
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
    sim = Simulator(HardwareConfig(), SimConfig(context_count=2, max_cycles=5000), enable_tracer=True)
    result = sim.run(make_two_role_same_tile_module())
    assert result.completed, result.reason
    data = json.loads(result.tracer.to_chrome_json())
    events = data["traceEvents"]

    tile0_pid = next(
      e["pid"]
      for e in events
      if e.get("name") == "process_name" and e.get("args", {}).get("name") == "Tile0"
    )
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
      e["ts"] for e in instants if e["name"] == "uce_issue" and thread_names.get(e["tid"]) == "UCE CTX1"
    ]
    assert ctx1_issue_ts, "missing CTX1 uce_issue instant"
    assert any(start <= ts <= end for start, end in ctx0_wait_intervals for ts in ctx1_issue_ts), (
      "expected a CTX1 uce_issue during a CTX0 WAIT_EVENT interval"
    )

  def test_trace_has_tilegroup_runtime_slices(self):
    """TileGroup task/role/Global-DMA/Collective duration bars exist."""
    module = make_group_runtime_trace_module()
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=50_000), enable_tracer=True)
    result = sim.run(module)
    assert result.completed, f"task did not complete: {result.reason}"
    data = json.loads(result.tracer.to_chrome_json())
    events = data["traceEvents"]
    slices = [e for e in events if e["ph"] == "X"]
    names = {e["name"] for e in slices}
    # task runtime window
    assert "task:group_runtime_trace_task" in names
    # role dispatch runtime window
    assert any(n.startswith("dispatch:role0:ev_role0:run") for n in names), names
    # Global DMA runtime windows
    assert "dma.prefetch:dma_prefetch0" in names
    assert "dma.store:dma_store0" in names
    # Collective runtime window
    assert "collective.reduce:coll_reduce0" in names
    # Global DMA slice carries bytes
    gdma = next(e for e in slices if e["name"] == "dma.prefetch:dma_prefetch0")
    assert gdma["args"]["bytes"] == 4096
    assert gdma["args"]["l2_slot"] == "l2_in0"
    # Collective slice carries bytes + participant_mask
    coll = next(e for e in slices if e["name"] == "collective.reduce:coll_reduce0")
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
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=50_000), enable_tracer=True)
    result = sim.run(wl.module)
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
    assert any(n.startswith("dispatch:role0:ev_role0:run") for n in names), names
    # MFE load/store bars still present on tile tracks (not renamed)
    mfe = [e for e in slices if e["cat"] == "MFE"]
    mfe_names = {e["name"] for e in mfe}
    assert "MFE:load" in mfe_names
    assert "MFE:store" in mfe_names
    # no "Tile DMA" category should exist
    assert not [e for e in slices if e["cat"] == "Tile DMA"], "Tile DMA category should not exist"
    # Global DMA slice carries bytes
    gdma_a = next(e for e in slices if e["name"] == "dma.prefetch:gdma_prefetch_A")
    assert gdma_a["args"]["bytes"] > 0
    assert gdma_a["args"]["l2_slot"] == "l2_buf_A"
    # instant markers include tile_role_dispatch + dma_complete
    instants = {e["name"] for e in events if e["ph"] == "i"}
    assert "tile_role_dispatch" in instants
    assert "dma_complete" in instants
    assert "group_task_done" in instants
    # dma_complete instant must land on a DMA channel thread, not a
    # stale "DMA" or "Global DMA" thread — prevents thread-name regression.
    tg_pid = next(
      e["pid"]
      for e in events
      if e.get("name") == "process_name" and e.get("args", {}).get("name") == "TileGroup"
    )
    thread_names = {
      e["args"]["name"] for e in events if e.get("name") == "thread_name" and e.get("pid") == tg_pid
    }
    assert "DMA" not in thread_names, "stale 'DMA' thread_name leaked on TileGroup"
    assert "Global DMA" not in thread_names, "stale 'Global DMA' thread_name leaked on TileGroup"
    assert "DMA Ch0" in thread_names
    assert "DMA Ch1" in thread_names
    for e in events:
      if e.get("name") == "dma_complete" and e.get("ph") == "i":
        assert e["cat"] in ("DMA Ch0", "DMA Ch1"), (
          f"dma_complete instant cat={e['cat']}, expected 'DMA Ch0/1'"
        )
        assert "channel" in e["args"], "dma_complete instant must carry channel arg"

  def test_tiled_matmul_trace_has_global_dma_and_mfe(self):
    """tiled_matmul task has Global DMA bars on TileGroup timeline and
    MFE load/store bars on tile tracks (MFE is NOT renamed to Tile DMA)."""
    wl = TiledMatmulWorkload()
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=50_000), enable_tracer=True)
    result = sim.run(wl.module)
    assert result.completed, f"tiled_matmul did not complete: {result.reason}"
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
    assert any(n.startswith("dispatch:role0:ev_role0:run") for n in names), names
    # MFE load/store bars present (NOT renamed to Tile DMA)
    mfe = [e for e in slices if e["cat"] == "MFE"]
    mfe_names = {e["name"] for e in mfe}
    assert "MFE:load" in mfe_names
    assert "MFE:store" in mfe_names
    # no "Tile DMA" category should exist
    assert not [e for e in slices if e["cat"] == "Tile DMA"], "Tile DMA category should not exist"

  def test_conv_relu_trace_uses_mfe_im2col_before_boa_matmul(self):
    wl = ConvReLuWorkload()
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000), enable_tracer=True)
    result = sim.run(wl.module)
    assert result.completed, f"conv_relu did not complete: {result.reason}"
    assert result.tracer is not None

    data = json.loads(result.tracer.to_chrome_json())
    slices = [
      e
      for e in data["traceEvents"]
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
        assert any(mfe["ts"] + mfe["dur"] <= boa["ts"] for mfe in tile_events.get("MFE:im2col", []))
    assert boa_tiles == 4

  def test_tiled_matmul_pipelined_trace_has_multi_stage_dma(self):
    """Pipelined tiled matmul task emits multiple Global DMA bars
    (one prefetch/store pair per group chunk) plus multiple role
    dispatch windows, proving the group-level IO pipeline."""
    wl = TiledMatmulPipelinedWorkload(num_group_chunks=4, num_k_chunks=4)
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000), enable_tracer=True)
    result = sim.run(wl.module)
    assert result.completed, f"tiled_matmul_pipelined did not complete: {result.reason}"
    data = json.loads(result.tracer.to_chrome_json())
    events = data["traceEvents"]
    slices = [e for e in events if e["ph"] == "X"]
    names = {e["name"] for e in slices}
    # Multiple DMA prefetch bars (one A+B pair per group chunk)
    for g in range(4):
      assert f"dma.prefetch:gdma_prefetch_A{g}" in names, f"missing prefetch A{g} in {sorted(names)}"
      assert f"dma.prefetch:gdma_prefetch_B{g}" in names
      assert f"dma.store:gdma_store_C{g}" in names
    # Multiple role dispatch windows (one per group chunk)
    for g in range(4):
      assert any(n.startswith(f"dispatch:role0:ev_role_c{g}:run") for n in names), (
        f"missing role dispatch for chunk {g} in {sorted(names)}"
      )
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
    role0 = next((e for e in slices if e["name"].startswith("dispatch:role0:ev_role_c0:run")), None)
    assert role0 is not None, "missing role0 dispatch slice"
    dma_a1_start = dma_a1["ts"]
    role0_end = role0["ts"] + role0["dur"]
    assert dma_a1_start < role0_end, (
      f"DMA prefetch A1 starts at {dma_a1_start} us, but role0 ends at {role0_end} us — no overlap"
    )
    # Also verify DMA prefetch B1 overlaps role0
    dma_b1 = by_name.get("dma.prefetch:gdma_prefetch_B1")
    assert dma_b1 is not None, "missing DMA prefetch B1 slice"
    assert dma_b1["ts"] < role0_end, (
      f"DMA prefetch B1 starts at {dma_b1['ts']} us, but role0 ends at {role0_end} us — no overlap"
    )


# ---------------------------------------------------------------------------
# Synthetic TileGroupTask for TileGroup runtime trace coverage
# ---------------------------------------------------------------------------


def make_group_runtime_trace_module() -> ModuleOp:
  """A synthetic task exercising Global DMA, role dispatch, and Collective."""
  role = TileRoleBindingOp(role_id=0, tile_mask=0x01, program=make_identity_tile_program())
  actions: list[GroupActionLike] = [
    GroupDMAPrefetchOp(descriptor="dma_prefetch0", l2_slot="l2_in0", event="ev_dma0", bytes_total=4096),
    GroupWaitEventOp(event="ev_dma0"),
    DispatchRoleOp(role_id=0, event="ev_role0"),
    GroupWaitEventOp(event="ev_role0"),
    CollectiveRunOp(
      descriptor="coll_reduce0",
      collective="reduce",
      bytes_total=2048,
      participant_mask=0x01,
      event="ev_coll0",
    ),
    GroupWaitEventOp(event="ev_coll0"),
    GroupDMAStoreOp(descriptor="dma_store0", l2_slot="l2_out0", event="ev_dma1", bytes_total=4096),
    GroupWaitEventOp(event="ev_dma1"),
    SignalEventOp(event="group_task_done"),
  ]
  task = TileGroupTaskOp(name="group_runtime_trace_task", streams=[], roles=[role], actions=actions)
  return ModuleOp([task])


# ---------------------------------------------------------------------------
# UCE multi-context tests
# ---------------------------------------------------------------------------


def make_waiting_mfe_program(name="ctx_wait_mfe") -> TileProgramOp:
  descriptors = [MFEDescriptorOp("mfe_load", "load", {"bytes": 128 * 1024, "ops": 0})]
  instructions: list[TileInstructionLike] = [
    LaunchMFEOp(descriptor="mfe_load", event="e_load"),
    WaitOp(event="e_load"),
    ReturnOp(),
  ]
  return TileProgramOp(name=name, descriptors=descriptors, instructions=instructions)


def make_short_evu_program(name="ctx_short_evu") -> TileProgramOp:
  descriptors = [EVUDescriptorOp("evu_short", "relu", {"ops": 16})]
  instructions: list[TileInstructionLike] = [
    LaunchEVUOp(descriptor="evu_short", event="e_evu"),
    WaitOp(event="e_evu"),
    ReturnOp(),
  ]
  return TileProgramOp(name=name, descriptors=descriptors, instructions=instructions)


def make_two_role_same_tile_module() -> ModuleOp:
  roles = [
    TileRoleBindingOp(role_id=0, tile_mask=0x01, program=make_waiting_mfe_program()),
    TileRoleBindingOp(role_id=1, tile_mask=0x01, program=make_short_evu_program()),
  ]
  actions: list[GroupActionLike] = [
    DispatchRoleOp(role_id=0, event="ev_role0"),
    DispatchRoleOp(role_id=1, event="ev_role1"),
    GroupWaitEventOp(event="ev_role0"),
    GroupWaitEventOp(event="ev_role1"),
    SignalEventOp(event="group_task_done"),
  ]
  task = TileGroupTaskOp(name="two_role_same_tile", streams=[], roles=roles, actions=actions)
  return ModuleOp([task])


class TestUceContextMode:
  def test_context_count_two_overlaps_two_roles_on_same_tile(self):
    dual = Simulator(HardwareConfig(), SimConfig(context_count=2, max_cycles=5000))
    dual_result = dual.run(make_two_role_same_tile_module())
    assert dual_result.completed, dual_result.reason
    assert dual_result.pmu.events.get("uce_context_switch", 0) > 0

    single = Simulator(HardwareConfig(), SimConfig(context_count=1, max_cycles=5000))
    single_result = single.run(make_two_role_same_tile_module())
    assert single_result.completed, single_result.reason
    assert dual_result.cycles < single_result.cycles, (
      f"context_count=2 should overlap roles: ctx2={dual_result.cycles}, ctx1={single_result.cycles}"
    )

  def test_context_count_one_serializes_same_tile_roles(self):
    sim = Simulator(HardwareConfig(), SimConfig(context_count=1, max_cycles=5000))
    result = sim.run(make_two_role_same_tile_module())
    assert result.completed, result.reason
    assert result.pmu.named_cycles.get("dispatch_wait", 0) > 0

  def test_context_count_validation(self):
    with pytest.raises(ValueError, match="context_count must be 1 or 2"):
      SimConfig(context_count=0)
    with pytest.raises(ValueError, match="context_count must be 1 or 2"):
      SimConfig(context_count=3)

  def test_full_memory_tile_scratchpad_snapshot_active(self):
    sim = Simulator(HardwareConfig(), SimConfig(fidelity="full_memory", max_cycles=200_000))
    result = sim.run(MatmulWorkload().module)
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
    def make_page_stream_module(prefetch_depth: int) -> ModuleOp:
      descriptors = [
        MFEDescriptorOp(
          "page_stream",
          "page_stream",
          {"bytes": 8192, "num_pages": 4, "page_size": 16, "prefetch_depth": prefetch_depth},
        )
      ]
      instructions: list[TileInstructionLike] = [
        LaunchMFEOp(descriptor="page_stream", event="e_page"),
        WaitOp(event="e_page"),
        ReturnOp(),
      ]
      prog = TileProgramOp(name="page_stream_tile", descriptors=descriptors, instructions=instructions)
      role = TileRoleBindingOp(role_id=0, tile_mask=0x0F, program=prog)
      actions: list[GroupActionLike] = [
        DispatchRoleOp(role_id=0, event="ev_role0"),
        GroupWaitEventOp(event="ev_role0"),
        SignalEventOp(event="group_task_done"),
      ]
      task = TileGroupTaskOp(name="page_stream_task", streams=[], roles=[role], actions=actions)
      return ModuleOp([task])

    hw = HardwareConfig().with_overrides(mfe_stream_buffer_bytes=4096)
    sim = Simulator(hw, SimConfig(max_cycles=1000))
    result = sim.run(make_page_stream_module(3))
    assert not result.completed, (
      f"expected fault for over-capacity prefetch, got completed={result.completed}"
    )
    assert "MFE page_stream prefetch requires 6144 bytes" in result.reason, (
      f"reason should mention 6144 bytes, got: {result.reason}"
    )

    hw = HardwareConfig().with_overrides(mfe_stream_buffer_bytes=4096)
    sim = Simulator(hw, SimConfig(max_cycles=1000))
    result = sim.run(make_page_stream_module(2))
    assert result.completed, f"exact-fit prefetch should complete, got: {result.reason}"

    hw = HardwareConfig()
    sim = Simulator(hw, SimConfig(max_cycles=1000))
    result = sim.run(make_page_stream_module(3))
    assert result.completed, f"default non-enforcing buffer should complete, got: {result.reason}"


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
    return sim.run(TiledMatmulPipelinedPowWorkload().module)

  def test_completes(self):
    result = self._run()
    assert result.completed, f"tiled_matmul_pipelined_pow did not complete: {result.reason}"
    assert result.cycles > 0
    assert result.credit_invariant_ok

  def test_task_structure_two_roles(self):
    """The task has two role bindings and the pow phase actions."""
    task = task_of(make_tiled_matmul_pipelined_pow_task(num_group_chunks=4, num_k_chunks=4))
    rmap = _role_map(task)
    assert len(rmap) == 2
    assert 0 in rmap and 1 in rmap
    # role 0 = matmul, role 1 = pow
    assert role_program(rmap[0]).program_name.data == "tiled_matmul_4k_tile"
    assert role_program(rmap[1]).program_name.data == "pow_4k_tile"
    # pow phase: 4 prefetches + 4 dispatches(role 1) + 4 stores + 4 drains
    classes = action_classes(task)
    # matmul phase: 4 dispatches(role 0) + pow phase: 4 dispatches(role 1)
    assert classes.count(DispatchRoleOp) == 8
    # 4 matmul C stores + 4 pow output stores
    assert classes.count(GroupDMAStoreOp) == 8
    # matmul prefetches (4x2 A+B) + pow prefetches (4) = 12
    assert classes.count(GroupDMAPrefetchOp) == 12

  def test_pow_tile_program_structure(self):
    """The pow tile program is load -> pow -> store."""
    p = make_pow_tile_program(name="pow_4k_tile", chunk_bytes=128 * 128 * 2)
    assert p.program_name.data == "pow_4k_tile"
    classes = instruction_classes(p)
    # launch.mfe, wait, launch.evu, wait, launch.mfe, wait, ret
    assert classes == [LaunchMFEOp, WaitOp, LaunchEVUOp, WaitOp, LaunchMFEOp, WaitOp, ReturnOp]
    # descriptors
    dmap = descriptor_map(p)
    assert "load_pow" in dmap
    assert "pow_chunk" in dmap
    assert "store_pow" in dmap
    assert dmap["pow_chunk"][0] == "EVU"
    assert dmap["pow_chunk"][1] == "pow"
    assert dmap["pow_chunk"][2]["exponent"] == 2
    assert dmap["pow_chunk"][2]["ops"] == 65536

  def test_report_all_checks_pass(self):
    """The report fingerprint passes: BOA + EVU + MFE + multi-stage IO."""
    wl = TiledMatmulPipelinedPowWorkload(num_group_chunks=4, num_k_chunks=4)
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000))
    result = sim.run(wl.module)
    assert result.completed, f"did not complete: {result.reason}"
    rep = build_report(wl, result)
    # all checks must pass
    failed = [c for c in rep.checks if not c["pass"]]
    assert not failed, f"failed checks: {failed}"
    # verify specific checks exist and pass
    completion = next(c for c in rep.checks if c["check"] == "task_completed")
    assert completion["pass"]
    gp = next(c for c in rep.checks if c["check"] == "multi_stage_group_io")
    assert gp["pass"]
    assert gp["actual"] is True
    evu = next(c for c in rep.checks if c["check"] == "evu_active_ratio")
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
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000), enable_tracer=True)
    result = sim.run(wl.module)
    assert result.completed, f"did not complete: {result.reason}"
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
      assert any(n.startswith(f"dispatch:role1:ev_role_pow{g}:run") for n in names), (
        f"missing pow role dispatch for chunk {g} in {sorted(names)}"
      )

    # ---- EVU:pow engine slices on tile tracks ----
    evu_slices = [e for e in slices if e["cat"] == "EVU"]
    evu_names = {e["name"] for e in evu_slices}
    assert "EVU:pow" in evu_names, f"EVU:pow slice missing, got: {evu_names}"

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
      f"{store_c3_end}"
    )

  def test_pow_prefetch_overlap_with_compute(self):
    """The up-front pow prefetches overlap: pow prefetch for chunk g+1
    starts before pow dispatch for chunk g finishes, proving the prefetches
    are issued up front (async DMA) while earlier chunks compute."""
    wl = TiledMatmulPipelinedPowWorkload(num_group_chunks=4, num_k_chunks=4)
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000), enable_tracer=True)
    result = sim.run(wl.module)
    assert result.completed, f"did not complete: {result.reason}"
    data = json.loads(result.tracer.to_chrome_json())
    slices = [e for e in data["traceEvents"] if e["ph"] == "X"]
    by_name = {e["name"]: e for e in slices}
    # pow prefetch 1 must start before pow dispatch 0 finishes
    pow_prefetch1 = by_name.get("dma.prefetch:gdma_prefetch_pow1")
    assert pow_prefetch1 is not None, "missing pow prefetch 1"
    pow_dispatch0 = next(
      (e for e in slices if e["name"].startswith("dispatch:role1:ev_role_pow0:run")), None
    )
    assert pow_dispatch0 is not None, "missing pow dispatch 0"
    dispatch0_end = pow_dispatch0["ts"] + pow_dispatch0["dur"]
    assert pow_prefetch1["ts"] < dispatch0_end, (
      f"pow prefetch 1 starts at {pow_prefetch1['ts']} but pow dispatch 0 "
      f"ends at {dispatch0_end} — prefetches not issued up front"
    )

  def test_sequencer_backpressure_same_mask_dispatch(self):
    """Regression: two DISPATCH_ROLE actions to the same tile_mask with no
    intervening WAIT must complete — the sequencer backpressures the second
    dispatch until the tiles from the first are done, instead of
    overwriting tile state and deadlocking."""
    chunk_bytes = 128 * 128 * 2
    prog = make_pow_tile_program(name="pow_tile", chunk_bytes=chunk_bytes)
    role = TileRoleBindingOp(role_id=0, tile_mask=0x0F, program=prog)
    actions = [
      GroupDMAPrefetchOp(descriptor="pref0", l2_slot="l2_0", event="ev_dma0", bytes_total=chunk_bytes * 4),
      GroupDMAPrefetchOp(descriptor="pref1", l2_slot="l2_1", event="ev_dma1", bytes_total=chunk_bytes * 4),
      GroupWaitEventOp(event="ev_dma0"),
      DispatchRoleOp(role_id=0, event="ev_role0"),
      GroupWaitEventOp(event="ev_dma1"),
      DispatchRoleOp(role_id=0, event="ev_role1"),
      GroupWaitEventOp(event="ev_role0"),
      GroupDMAStoreOp(descriptor="store0", l2_slot="l2_0", event="ev_store0", bytes_total=chunk_bytes * 4),
      GroupWaitEventOp(event="ev_role1"),
      GroupDMAStoreOp(descriptor="store1", l2_slot="l2_1", event="ev_store1", bytes_total=chunk_bytes * 4),
      GroupWaitEventOp(event="ev_store0"),
      GroupWaitEventOp(event="ev_store1"),
      SignalEventOp(event="group_task_done"),
    ]
    task = TileGroupTaskOp(name="test_backpressure", streams=[], roles=[role], actions=actions)
    module = ModuleOp([task])
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=50_000))
    result = sim.run(module)
    assert result.completed, f"backpressure test did not complete: {result.reason}"
    # both dispatches and both stores must have run
    assert result.pmu.events.get("tgs_dispatch_role", 0) == 2
    assert result.pmu.events.get("tgs_dma_store", 0) == 2
    assert result.pmu.events.get("tile_done", 0) == 8  # 2 dispatches x 4 tiles


class TestTiledMatmulPowNodep:
  """Tests for the nodep tiled-matmul + pow trace fixture workload."""

  def _run(self, enable_tracer: bool = False):
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000), enable_tracer=enable_tracer)
    return sim.run(TiledMatmulPowNodepWorkload().module)

  def test_print_ir_matches_golden(self):
    golden = Path(__file__).resolve().parents[1] / "tests" / "golden" / "tiled_matmul_pow_nodep.mlir"
    actual = print_workload_ir(make_tiled_matmul_pow_nodep_task())
    expected = golden.read_text(encoding="utf-8")
    assert actual == expected

  def test_task_structure_and_counts(self):
    task = task_of(make_tiled_matmul_pow_nodep_task())
    rmap = _role_map(task)
    assert set(rmap.keys()) == {0, 1}
    assert role_program(rmap[0]).program_name.data == "tiled_matmul_4k_tile"
    assert role_program(rmap[1]).program_name.data == "pow_4k_tile"

    classes = action_classes(task)
    assert classes.count(GroupDMAPrefetchOp) == 12
    assert classes.count(DispatchRoleOp) == 8
    assert classes.count(GroupDMAStoreOp) == 8
    assert classes.count(GroupWaitEventOp) == 28
    assert len(classes) == 57

    actions = task_actions(task)
    assert actions[0].event.data == "ev_dma_pow_in0"
    assert actions[3].event.data == "ev_dma_pow_in3"
    assert actions[11].event.data == "ev_role_pow3"
    assert actions[12].event.data == "ev_dma_A0"
    assert actions[19].event.data == "ev_dma_B3"
    assert isinstance(actions[20], GroupWaitEventOp)
    assert actions[20].event.data == "ev_dma_A0"
    assert actions[22].event.data == "ev_role_c0"
    assert actions[31].event.data == "ev_role_c3"
    assert isinstance(actions[40], GroupWaitEventOp)
    assert actions[40].event.data == "ev_role_pow0"
    assert actions[41].event.data == "ev_dma_pow_out0"
    assert isinstance(actions[-1], SignalEventOp)

  def test_completes_and_report_checks_pass(self):
    wl = TiledMatmulPowNodepWorkload()
    sim = Simulator(HardwareConfig(), SimConfig(max_cycles=200_000))
    result = sim.run(wl.module)
    assert result.completed, f"tiled_matmul_pow_nodep did not complete: {result.reason}"
    assert result.credit_invariant_ok

    rep = build_report(wl, result)
    failed = [c for c in rep.checks if not c["pass"]]
    assert not failed, f"failed checks: {failed}"

  def test_trace_has_nodep_order(self):
    result = self._run(enable_tracer=True)
    assert result.completed, f"tiled_matmul_pow_nodep did not complete: {result.reason}"

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
      f"pow prefetch 0 starts at {pow_prefetch0['ts']} but C3 store starts at {store_c3['ts']}"
    )
    assert store_c3["ts"] <= store_pow0["ts"], (
      f"C3 store starts at {store_c3['ts']} but pow store 0 starts at {store_pow0['ts']}"
    )

    pow_dispatch0 = next(
      (e for e in slices if e["name"].startswith("dispatch:role1:ev_role_pow0:run")), None
    )
    matmul_dispatch0 = next(
      (e for e in slices if e["name"].startswith("dispatch:role0:ev_role_c0:run")), None
    )
    assert pow_dispatch0 is not None, "missing pow dispatch 0 slice"
    assert matmul_dispatch0 is not None, "missing matmul dispatch 0 slice"
    assert pow_dispatch0["ts"] < matmul_dispatch0["ts"], (
      f"pow dispatch 0 starts at {pow_dispatch0['ts']} but matmul dispatch "
      f"0 starts at {matmul_dispatch0['ts']}"
    )
