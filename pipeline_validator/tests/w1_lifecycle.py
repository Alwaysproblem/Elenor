"""C035 test drivers: real Group resources, compiled Grids and observed C ABI traffic."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from xdsl.dialects.builtin import ModuleOp

from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.cosim.rtl_backend import RtlTileBackend
from pipeline_validator.cosim.tile_port import BoundaryMessage, TilePort
from pipeline_validator.dialects.elenor import (
  NestAllocOp,
  NestAwaitOp,
  NestContextOp,
  NestDispatchOp,
  NestReleaseOp,
  NestReturnOp,
  NestTask,
  NestTaskRangeOp,
  TileProgramDefOp,
  TileReturnOp,
  TileSignalOp,
)
from pipeline_validator.execution_ir import ExecTileGroupTask
from pipeline_validator.tests.test_cn_abi_phase import BUILD_SCRIPT, IMAGE, REPO
from pipeline_validator.tests.test_runtime import (
  L2_WAIT_BYTES,
  L2_WAIT_DIMS,
  context_resources,
  make_waiting_mfe_program,
  prepare_group_source,
  tile_resources,
)
from pipeline_validator.tile_group import TileGroup


def reader_module(
  participants: int, grids: int, leases: int, reads: bool, pin: int | tuple[int, ...] | None
) -> ModuleOp:
  """Real compiler access summaries: reporting an input phase does not imply an L2 read."""
  if reads:
    program = make_waiting_mfe_program("reader")
  else:
    program = TileProgramDefOp("reader", tile_resources(), arg_types=[NestTask()], arg_names=["task"])
    program.body.block.add_ops([TileSignalOp("input_released", program.body.block.args[0]), TileReturnOp()])
  tasks = NestTaskRangeOp(0, participants)
  buf = NestAllocOp("input", "in", L2_WAIT_DIMS, "bf16") if reads else None
  args = [buf.result] if buf is not None else []
  dispatches = [
    NestDispatchOp(
      "reader",
      tasks.result,
      [],
      args,
      [],
      f"grid{i}",
      f"input{i}",
      "",
      l1_mode=0,
      bindings=args,
      context_id=pin[i] if isinstance(pin, tuple) else pin,
      signal_policy={"input_released": "all_tasks"},
    )
    for i in range(grids)
  ]
  release = [NestReleaseOp(buf.result, depends_on=[d.input_released for d in dispatches])] if buf else []
  context = NestContextOp(
    "c035",
    context_resources(
      participants * grids, L2_WAIT_BYTES if reads else 0, requested_contexts_per_tile=leases
    ),
    [
      *([buf] if buf else []),
      tasks,
      *dispatches,
      *release,
      NestAwaitOp([d.grid_done for d in dispatches]),
      NestReturnOp(),
    ],
    placement=(1 << participants) - 1,
  )
  return ModuleOp([program, context])


@dataclass
class Run:
  group: TileGroup
  cycle: int = 0

  def tick(self) -> bool:
    done = self.group.step(self.cycle)
    self.cycle += 1
    assert self.group.credit_invariants_hold()
    return done

  def until(self, condition: Callable[[], bool], limit: int = 4000) -> None:
    for _ in range(limit):
      if condition():
        return
      self.tick()
    raise AssertionError(
      f"C035 stalled at cycle {self.cycle}; reset={self.group.reset_domain.state.name}, "
      f"tickets={[(t.txn_id, t.state.name) for t in self.group.admission.tickets.values()]}, "
      f"wait={[r.wait_reasons for r in self.group._grid_routes.values()]}"
    )

  @property
  def pins(self) -> int:
    return self.group.l2_sram.snapshot()["pin_count"]

  @property
  def leases(self) -> int:
    return sum(map(len, self.group._task_leases.values()))

  def finish(self, *, fault: bool = False) -> None:
    self.until(lambda: self.group.sequencer.done and (not fault or self.group.reset_domain.is_done))
    assert self.group.sequencer.faulted is fault
    assert not self.group.admission.tickets and not self.group._grid_routes
    assert self.leases == self.pins == 0 and self.group.unclosed_l2_objects() == []
    assert self.group.event_table.reserved == 0 and self.group.transfer_manager.inflight_count == 0


def start(
  *,
  participants: int = 1,
  grids: int = 1,
  leases: int = 1,
  reads: bool = True,
  pin: int | tuple[int, ...] | None = None,
  ports: Mapping[int, TilePort] | None = None,
) -> Run:
  hw = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
  sim = SimConfig(fidelity="runtime", context_count=2)
  group = TileGroup(hw, fidelity=sim.fidelity, context_count=sim.context_count, boundary_ports=ports)
  artifact = prepare_group_source(group, reader_module(participants, grids, leases, reads, pin), sim=sim)
  assert isinstance(artifact.entry, ExecTileGroupTask)
  group.load_task(artifact.entry)
  return Run(group)


class ObservedNative(RtlTileBackend):
  """Observe submitted packets and receipts returned by bp_collect; never synthesize a response."""

  def __init__(self, tile_id: int, **kwargs) -> None:
    super().__init__(tile_id, **kwargs)
    self.sent: list[BoundaryMessage] = []
    self.received: list[BoundaryMessage] = []

  def send(self, msg: BoundaryMessage) -> bytes:
    blob = super().send(msg)
    self.sent.append(BoundaryMessage.from_wire(blob))
    return blob

  def pop_inbound(self) -> BoundaryMessage | None:
    msg = super().pop_inbound()
    if msg is not None:
      self.received.append(msg)
    return msg

  def messages(self, name: str, *, sent: bool = False) -> list[BoundaryMessage]:
    return [m for m in (self.sent if sent else self.received) if m.message_type == name]

  def summary(self) -> dict:
    return {
      "evidence_tier": "stub",
      "abi_hash": self.abi.abi_info(),
      "tile_id": self.tile_id,
      "ready_seed": self._ready_seed,
      "fixture_flags": self._stub_flags,
      "submitted": [[m.message_type, m.fields] for m in self.sent],
      "received": [[m.message_type, m.fields] for m in self.received],
      "posedge_count": int(self.abi.status().posedge_count),
    }


def native_cases(module: str, out: Path) -> dict:
  """Use the locked libc for the C025 .so and the same read-only conda interpreter as the validator."""
  subprocess.run(["bash", str(BUILD_SCRIPT)], check=True, stdout=subprocess.PIPE)
  prefix = Path(sys.prefix).resolve()
  subprocess.run(
    [
      "docker",
      "run",
      "--rm",
      "--network=none",
      "--user",
      f"{os.getuid()}:{os.getgid()}",
      "-e",
      "HOME=/tmp",
      "-e",
      "PYTHONDONTWRITEBYTECODE=1",
      "-v",
      f"{REPO}:/repo:ro",
      "-v",
      f"{prefix}:{prefix}:ro",
      "-v",
      f"{out}:/out",
      "-w",
      "/repo",
      IMAGE,
      str(prefix / "bin/python"),
      "-m",
      f"pipeline_validator.tests.{module}",
      "/out",
    ],
    check=True,
    timeout=180,
  )
  return json.loads((out / "c035.json").read_text())
