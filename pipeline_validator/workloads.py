"""Workload definitions.

Each workload builds one xDSL ModuleOp plus immutable metadata persisted in the
compiled artifact.  Runtime and report code consume ``WorkloadInfo`` without
reconstructing author source IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects.builtin import ModuleOp

from .compiled_program import WorkloadInfo
from .config import HardwareConfig, WorkloadConfig
from .workload_builders import make_pow_task


@dataclass
class Workload:
  """Base workload: author source plus expected observable results."""

  name: str
  module: ModuleOp
  expected: dict = field(default_factory=dict)
  description: str = ""
  config: WorkloadConfig | None = None

  @property
  def info(self) -> WorkloadInfo:
    return WorkloadInfo(name=self.name, description=self.description, expected=self.expected)


class PowWorkload(Workload):
  """Standalone EVU pow task with pipelined group DMA.

  Mirrors ``pow.ir``: one role (role 1) across 4 tiles, four up-front
  HBM->L2 prefetches, per-chunk dispatch once the input is visible in L2,
  per-chunk L2->HBM store once the pow tiles finish, then drain all stores.
  No BOA role participates in this workload.
  """

  def __init__(
    self,
    cfg: WorkloadConfig | None = None,
    num_group_chunks: int = 4,
    *,
    hw: HardwareConfig | None = None,
    context_count: int = 1,
  ):
    cfg = cfg or WorkloadConfig(name="pow")
    hw = hw or HardwareConfig()
    module = make_pow_task(num_group_chunks=num_group_chunks, hw=hw, context_count=context_count)
    super().__init__(
      name="pow",
      module=module,
      description=(
        "Standalone EVU pow(x, 2) workload across 4 tiles. "
        f"{num_group_chunks} group chunks; each chunk prefetches one "
        "pow input tile group to L2, dispatches role 1 "
        "(`pow_4k_tile`) once visible, then stores the output back "
        "to HBM.  Per-tile work: 1 MFE load + 1 EVU pow + 1 MFE store."
      ),
      expected={
        "evu_active_ratio_min": 0.01,
        "mfe_active_ratio_min": 0.03,
        "stream_stall_ratio_max": 0.05,
        "multi_stage_group_io": True,
      },
      config=cfg,
    )


ALL_WORKLOADS: list[type[PowWorkload]] = [PowWorkload]
