"""NoC router — 4 virtual channels + credit + contention (NoC design 3.2).

Models the NoC router pipeline with per-VC credit tracking.  VC0
(command/event/fault/barrier) has priority and must not be blocked by
VC1/VC2/VC3 (design 3.4 table).  VC2 (DMA write) can be throttled.

Credit-based: each VC tracks downstream credit_available; sends stall
when credit is exhausted.  V1 simplification: one router, no mesh.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum


class VCId(IntEnum):
  """NoC virtual channels (NoC design 3.4)."""
  VC0_COMMAND_EVENT = 0   # command, event, fault, barrier — highest priority
  VC1_DMA_READ_RSP = 1    # DMA read response, memory completion
  VC2_DMA_WRITE = 2       # DMA write, MFE stream fill, bulk data
  VC3_COLLECTIVE = 3      # collective


@dataclass
class Flit:
  """One NoC flit."""
  vc: int
  src: int
  dst: int
  bytes_total: int
  tag: int = 0
  arrived_cycle: int = 0


@dataclass
class VirtualChannel:
  """One virtual channel with credit + a FIFO of pending flits."""
  vc_id: int
  depth: int = 8
  priority: int = 0  # lower = higher priority; VC0 = 0

  def __post_init__(self) -> None:
    self.credit_available: int = self.depth
    self._pending: deque[Flit] = deque()
    self.occupancy_cycles: int = 0
    self.stall_cycles: int = 0
    self._starvation: int = 0  # age-based arbiter (design 3.2 stage 5)

  def can_send(self) -> bool:
    return self.credit_available > 0

  def enqueue(self, flit: Flit, cycle: int) -> None:
    flit.arrived_cycle = cycle
    self._pending.append(flit)

  def try_send(self, cycle: int) -> Flit | None:
    """Attempt to send the head flit.  Returns None if stalled."""
    if not self._pending:
      return None
    if not self.can_send():
      self.stall_cycles += 1
      self._starvation += 1
      return None
    flit = self._pending.popleft()
    self.credit_available -= 1
    self._starvation = 0
    return flit

  def return_credit(self, n: int = 1) -> None:
    self.credit_available = min(self.depth, self.credit_available + n)

  @property
  def occupancy(self) -> int:
    return len(self._pending)

  def tick(self, cycle: int) -> None:
    self.occupancy_cycles += self.occupancy

  def reset(self) -> None:
    self.credit_available = self.depth
    self._pending.clear()
    self.occupancy_cycles = 0
    self.stall_cycles = 0
    self._starvation = 0


@dataclass
class NoCRouter:
  """Single-router NoC model with 4 VCs and priority arbitration.

  V1 simplification: one router (no mesh topology).  Each cycle, the
  router arbitrates across VCs by priority (VC0 first), then sends flits
  that have downstream credit.  VC0 has starvation protection so it is
  never permanently blocked by VC2 bulk traffic.
  """

  vc_depth: int = 8
  router_latency_cycles: int = 4

  def __post_init__(self) -> None:
    self.vcs: dict[int, VirtualChannel] = {
      vc_id.value: VirtualChannel(
        vc_id=vc_id.value,
        depth=self.vc_depth,
        priority=vc_id.value)
      for vc_id in VCId
    }
    self.pmu_switch_contention: int = 0

  def send(self, vc: int, flit: Flit, cycle: int) -> None:
    """Enqueue a flit onto a VC (upstream side)."""
    self.vcs[vc].enqueue(flit, cycle)

  def step(self, cycle: int) -> list[Flit]:
    """Advance one cycle.  Returns the list of flits that traversed.

    Arbitration: VC0 first (priority), then VC1, VC2, VC3.  Starvation
    counter boosts a VC if it has been stalled too long (design 3.2).
    Returns flits after `router_latency_cycles` (modeled as immediate
    return with the latency added by the caller).
    """
    for vc in self.vcs.values():
      vc.tick(cycle)

    sent: list[Flit] = []
    # priority arbitration: VC0 -> VC3
    contention = 0
    for vc_id in sorted(self.vcs.keys()):
      vc = self.vcs[vc_id]
      # starvation boost: if VC0 stalled > 4 cycles, force-send
      if vc.vc_id == VCId.VC0_COMMAND_EVENT and vc._starvation > 4:
        flit = vc._pending[0] if vc._pending else None
        if flit is not None and vc.credit_available > 0:
          vc._pending.popleft()
          vc.credit_available -= 1
          vc._starvation = 0
          sent.append(flit)
          continue
      flit = vc.try_send(cycle)
      if flit is not None:
        sent.append(flit)
      elif vc.occupancy > 0:
        contention += 1
    self.pmu_switch_contention += contention
    return sent

  def return_credit(self, vc: int, n: int = 1) -> None:
    self.vcs[vc].return_credit(n)

  def reset(self) -> None:
    for vc in self.vcs.values():
      vc.reset()
    self.pmu_switch_contention = 0

  def snapshot(self) -> dict:
    return {
      VCId(vc.vc_id).name: {
        "occupancy": vc.occupancy,
        "credit": vc.credit_available,
        "stall_cycles": vc.stall_cycles,
      }
      for vc in self.vcs.values()
    }
