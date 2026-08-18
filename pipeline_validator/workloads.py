"""Workload definitions.

Each workload builds one xDSL ModuleOp plus a human-readable description.
The validator runs the module and compares the measured PMU fingerprint
against the architecture's predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects.builtin import ModuleOp

from .config import WorkloadConfig
from .workload_builders import make_pow_task


@dataclass
class Workload:
  """Base workload: a name + a ModuleOp + expected PMU observations."""

  name: str
  module: ModuleOp
  expected: dict = field(default_factory=dict)
  description: str = ""
  config: WorkloadConfig | None = None


class PowWorkload(Workload):
  """Standalone EVU pow task with pipelined group DMA.

  Mirrors ``pow.ir``: one role (role 1) across 4 tiles, four up-front
  HBM->L2 prefetches, per-chunk dispatch once the input is visible in L2,
  per-chunk L2->HBM store once the pow tiles finish, then drain all stores.
  No BOA role participates in this workload.
  """

  def __init__(self, cfg: WorkloadConfig | None = None, num_group_chunks: int = 4):
    cfg = cfg or WorkloadConfig(name="pow")
    module = make_pow_task(num_group_chunks=num_group_chunks)
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
        "mfe_active_ratio_min": 0.05,
        "stream_stall_ratio_max": 0.05,
        "multi_stage_group_io": True,
      },
      config=cfg,
    )


ALL_WORKLOADS: list = [PowWorkload]
