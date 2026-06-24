"""PMU counters and unique stall attribution.

Implements the unique-attribution rule from
design/ELENOR_Architecture_Design_v1.md section 21.6: every stall cycle
has exactly one primary owner.  The hierarchy order resolves overlaps.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum


class StallReason(IntEnum):
    """elenor_stall_reason_t (Architecture 21.6).  Order = priority."""
    NONE = 0
    WAIT_EVENT = 1
    WAIT_OPERAND = 2
    STREAM_CREDIT = 3
    SRAM_BANK = 4
    NOC_VC = 5
    DMA_MEMORY = 6
    UCE_PROGRAM_DESC = 7
    UNKNOWN = 8

    @property
    def label(self) -> str:
        return _STALL_LABELS[self]


_STALL_LABELS = {
    StallReason.NONE: "engine_active",
    StallReason.WAIT_EVENT: "engine_wait_event",
    StallReason.WAIT_OPERAND: "engine_wait_operand",
    StallReason.STREAM_CREDIT: "stream_credit_empty_or_full",
    StallReason.SRAM_BANK: "sram_bank_conflict",
    StallReason.NOC_VC: "noc_backpressure",
    StallReason.DMA_MEMORY: "dma_wait_memory",
    StallReason.UCE_PROGRAM_DESC: "uce_program_or_descriptor_stall",
    StallReason.UNKNOWN: "unknown_or_unclassified",
}


@dataclass
class PMUCounter:
    """A generic PMU counter set with unique stall attribution.

    `stall_cycles` records per-primary-owner stall cycles.
    `named_cycles` records arbitrary cycle counters (occupancy, queue_full, ...)
    `events` records event counts (push, pop, acquire, release, eos, fault).
    """
    stall_cycles: dict = field(default_factory=lambda: defaultdict(int))
    named_cycles: dict = field(default_factory=lambda: defaultdict(int))
    events: dict = field(default_factory=lambda: defaultdict(int))

    def add(self, reason: StallReason, n: int = 1) -> None:
        self.stall_cycles[reason] += n

    def add_cycle(self, name: str, n: int = 1) -> None:
        self.named_cycles[name] += n

    def add_event(self, name: str, n: int = 1) -> None:
        self.events[name] += n

    def active_cycles(self) -> int:
        """Cycles where the engine was *not* stalled (idle-or-busy).

        For engine PMUs this is total - sum(stall).  For a queue PMU the
        occupancy-weighted total is in named_cycles['occupancy'].
        """
        total_stall = sum(self.stall_cycles.values())
        # caller sets total via set_total; we keep a separate field.
        return max(self.named_cycles.get("total", 0) - total_stall, 0)

    def utilization(self, total_cycles: int) -> float:
        """Fraction of tile-cycles where at least one engine was active.

        Computed as total active cycles (StallReason.NONE, aggregated across
        all engines and the UCE) divided by total tile-cycles.  Capped at 1.0.
        """
        if total_cycles <= 0:
            return 0.0
        active = self.stall_cycles.get(StallReason.NONE, 0)
        return min(active / total_cycles, 1.0)

    def stall_breakdown(self) -> dict:
        """Map stall_label -> cycles, dropping zeros."""
        return {
            reason.label: c
            for reason, c in self.stall_cycles.items() if c
        }

    def merge(self, other: PMUCounter) -> None:
        for k, v in other.stall_cycles.items():
            self.stall_cycles[k] += v
        for k, v in other.named_cycles.items():
            self.named_cycles[k] += v
        for k, v in other.events.items():
            self.events[k] += v

    def reset(self) -> None:
        self.stall_cycles.clear()
        self.named_cycles.clear()
        self.events.clear()


def primary_stall_owner(reasons) -> StallReason:
    """Pick the highest-priority (lowest int) non-NONE stall reason.

    Implements the 'each stall cycle has exactly one primary owner' rule.
    """
    active = [r for r in reasons if r != StallReason.NONE]
    if not active:
        return StallReason.NONE
    return min(active)
