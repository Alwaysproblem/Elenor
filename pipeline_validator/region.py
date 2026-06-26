"""Region Sequencer controller.

Executes a Region Program (design/elenor_region_sequencer/ and
Architecture doc 16.3) on the Tile Group.  The sequencer:

  - inits stream queues,
  - prefetches blocks via Group DMA (modelled as latency),
  - dispatches stages (tile_mask -> which tiles run which Tile Program),
  - waits on stage events,
  - pushes EOS, barriers, and ends the region.

Like the Tile UCE it is a one-instruction-per-cycle controller with a
pending-wait mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ir import RegionInst, RegionOp, RegionProgram, StreamDesc
from .pmu import PMUCounter, StallReason

if TYPE_CHECKING:
    from .tile_group import TileGroup


@dataclass
class _RegionWait:
    events: tuple
    started_cycle: int


class RegionSequencer:
    """The Tile Group Region Sequencer."""

    def __init__(self, group: TileGroup):
        self.group = group
        self.cfg = group.cfg
        self.pmu = PMUCounter()
        self.pc = 0
        self.program: RegionProgram | None = None
        self._events_done: set[str] = set()
        self._pending: _RegionWait | None = None
        self._stage_events: dict[int, str] = {}  # stage_id -> event_id
        self._tile_masks: dict[int, int] = {}  # stage_id -> tile_mask
        self.done = False
        # block counter for branch.lt loops
        self._block_reg: dict[str, int] = {}
        # round-robin DMA channel allocation
        self._next_dma_channel: int = 0

    def load(self, program: RegionProgram) -> None:
        self.program = program
        self.pc = 0
        self._events_done.clear()
        self._pending = None
        self._stage_events.clear()
        self._block_reg.clear()
        self._next_dma_channel = 0
        self.done = False

    # ---- per-cycle step -------------------------------------------------

    def step(self, cycle: int) -> tuple[int, str] | None:
        """Execute one cycle.  Returns (stage_id, event_id) if a stage was
        just dispatched (so the TileGroup can start tiles), else None.
        """
        if self.done or self.program is None:
            self.pmu.add_cycle("idle", 1)
            self.pmu.add_cycle("total", 1)
            return None

        if self._pending is not None:
            if all(e in self._events_done for e in self._pending.events):
                self._pending = None
                self.pmu.add_cycle("wait_resolved", 1)
                self.pmu.add_cycle("total", 1)
                return None
            else:
                self.pmu.add(StallReason.WAIT_EVENT, 1)
                self.pmu.add_cycle("wait_event", 1)
                self.pmu.add_cycle("total", 1)
                return None

        if self.pc >= len(self.program.insts):
            self.done = True
            self.pmu.add_cycle("total", 1)
            return None

        ins = self.program.insts[self.pc]
        self.pmu.add_cycle("total", 1)
        return self._issue(ins, cycle)

    # ---- instruction issue ----------------------------------------------

    def _issue(self, ins: RegionInst, cycle: int) -> tuple[int, str] | None:
        op = ins.op
        if op == RegionOp.REGION_BEGIN:
            self.pc += 1
        elif op == RegionOp.REGION_END:
            self.done = True
            self.pc += 1
        elif op == RegionOp.INIT_STREAM:
            qid, depth, pmask, cmask = ins.args
            sdesc = StreamDesc(queue_id=qid,
                               depth=depth,
                               producer_mask=pmask,
                               consumer_mask=cmask)
            self.group.init_stream(sdesc)
            self.pmu.add_event("init_stream")
            self.pc += 1
        elif op == RegionOp.DMA_PREFETCH:
            # Group DMA HBM->L2: model as latency, produces an event.
            if ins.dst is None:
                raise ValueError("DMA_PREFETCH requires dst event id")
            desc_id, dst_l2 = ins.args[0], ins.args[1]
            bytes_total = ins.args[2] if len(ins.args) > 2 else None
            resolved_bytes = bytes_total if bytes_total and bytes_total > 0 else 1024 * 1024
            lat = self._dma_latency(resolved_bytes)
            # round-robin DMA channel allocation
            ch = self._next_dma_channel % self.cfg.num_dma_channels
            self._next_dma_channel += 1
            self.group.schedule_dma(
                ins.dst,
                lat,
                cycle,
                op="dma.prefetch",
                desc_id=desc_id,
                l2_slot=dst_l2,
                bytes_total=resolved_bytes,
                channel=ch,
            )
            self.pmu.add_event("dma_prefetch")
            self.pc += 1
        elif op == RegionOp.DMA_STORE:
            if ins.dst is None:
                raise ValueError("DMA_STORE requires dst event id")
            desc_id, src_l2 = ins.args[0], ins.args[1]
            bytes_total = ins.args[2] if len(ins.args) > 2 else None
            resolved_bytes = bytes_total if bytes_total and bytes_total > 0 else 1024 * 1024
            lat = self._dma_latency(resolved_bytes)
            # round-robin DMA channel allocation
            ch = self._next_dma_channel % self.cfg.num_dma_channels
            self._next_dma_channel += 1
            self.group.schedule_dma(
                ins.dst,
                lat,
                cycle,
                op="dma.store",
                desc_id=desc_id,
                l2_slot=src_l2,
                bytes_total=resolved_bytes,
                channel=ch,
            )
            self.pmu.add_event("dma_store")
            self.pc += 1
        elif op == RegionOp.DISPATCH_STAGE:
            stage_id, tile_mask, prog_idx, *rest = ins.args
            out_stream = rest[0] if len(rest) > 0 else None
            in_stream = rest[1] if len(rest) > 1 else None
            ev = ins.dst or f"ev_stage{stage_id}"
            self._stage_events[stage_id] = ev
            self._tile_masks[stage_id] = tile_mask
            # start the tiles: load program + bind streams
            self.group.dispatch_stage(stage_id, tile_mask, prog_idx,
                                      out_stream, in_stream, cycle, event_id=ev)
            self.pmu.add_event("dispatch_stage")
            self.pc += 1
            return (stage_id, ev)
        elif op == RegionOp.WAIT_EVENT:
            ev = ins.args[0]
            self._pending = _RegionWait(events=(ev, ), started_cycle=cycle)
            self.pc += 1
        elif op == RegionOp.WAIT_STREAM:
            self.pmu.add(StallReason.STREAM_CREDIT, 1)
            self.pc += 1
        elif op == RegionOp.WAIT_CREDIT:
            self.pmu.add(StallReason.STREAM_CREDIT, 1)
            self.pc += 1
        elif op == RegionOp.BARRIER_GROUP:
            self.pmu.add_event("barrier")
            self.pc += 1
        elif op == RegionOp.COLLECTIVE_RUN:
            if ins.dst is None:
                raise ValueError("COLLECTIVE_RUN requires dst event id")
            desc_id, op_name, bytes_total, participant_mask = ins.args
            self.group.schedule_collective(
                desc_id,
                ins.dst,
                op_name,
                bytes_total,
                participant_mask,
                cycle,
            )
            self.pmu.add_event("collective_run")
            self.pc += 1
        elif op == RegionOp.PUSH_EOS:
            qid, producer_id = ins.args
            q = self.group.queues.get(qid)
            if q is not None:
                q.push_eos(producer_id, cycle)
            self.pmu.add_event("push_eos")
            self.pc += 1
        elif op == RegionOp.ADVANCE_BLOCK:
            reg, stride = ins.args
            self._block_reg[reg] = self._block_reg.get(reg, 0) + stride
            self.pc += 1
        elif op == RegionOp.BRANCH_LT:
            lhs_reg, rhs, target = ins.args
            if self._block_reg.get(lhs_reg, 0) < rhs:
                assert self.program is not None
                self.pc = self.program.label_index(target)
            else:
                self.pc += 1
        elif op == RegionOp.SIGNAL_EVENT:
            self._events_done.add(ins.args[0])
            self.pmu.add_event("signal_event")
            self.pc += 1
        else:
            self.pc += 1
        return None

    def _dma_latency(self, bytes_total: int | None = None) -> int:
        """Group DMA latency: bytes / group_dma_bandwidth."""
        # default prefetch = 1MB block
        nbytes = bytes_total if bytes_total and bytes_total > 0 else 1024 * 1024
        bw_bytes_per_cycle = self.cfg.group_dma_bandwidth_gbs * 1e9 / (
            self.cfg.clock_mhz * 1e6)
        return int(max((nbytes + bw_bytes_per_cycle - 1) // bw_bytes_per_cycle, 1))

    def notify_event(self, event_id: str) -> None:
        self._events_done.add(event_id)

    def reset(self) -> None:
        self.pc = 0
        self._events_done.clear()
        self._pending = None
        self._stage_events.clear()
        self._tile_masks.clear()
        self._block_reg.clear()
        self._next_dma_channel = 0
        self.done = False
        self.pmu.reset()
