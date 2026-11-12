"""PythonTileBackend: zero-drift `TilePort` adapter over `ComputeTile` (C022 / W1-02).

Invariants:
- `PythonTileBackend` is the sole execution owner of an un-replaced Python Tile (`backend_kind="python"`).
- `send(msg)` validates pure-value payloads, packs through `wire.py`, reserves G2T channel credit,
  and enqueues into `_outbox`. It NEVER calls `tile.step()` or `uce.step()` and NEVER advances
  simulation cycles or PMU counters.
- `step(cycle, freeze_new_work=...)` executes at most once per integer `cycle`:
  1. Drains `_outbox` through `wire.Rx` and enforces launch/Context generation monotonicity:
     a `TASK_PLAN`, `TASK_COMMIT`, or `TASK_COMMIT_FINAL` carrying a retired `launch_generation`
     or `ctx_generation` is rejected with `stale_plan_rejected += 1`, emits `ACK(outcome="STALE")`,
     and mutates zero Tile state (instruction `op_seq_hint` is never accepted as launch identity).
  2. Advances `self.tile.step(cycle, freeze_new_work=freeze_new_work)` exactly once.
  3. Asserts per-(channel, direction) credit conservation (`available + in_flight + occupancy == cap`).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

from ..tile import ComputeTile
from . import wire
from .tile_port import BackendKind, BoundaryMessage, ChannelCreditLedger


class PythonTileBackend:
  """`TilePort` implementation wrapping a pure-Python `ComputeTile` with zero behavioral drift."""

  def __init__(self, tile: ComputeTile, *, channel_capacities: Mapping[str, int] | None = None) -> None:
    self.tile = tile
    self.credits = ChannelCreditLedger(channel_capacities)
    self._rx = wire.Rx()
    self._tx_seq: dict[tuple[str, str], int] = {}
    self._outbox: deque[tuple[BoundaryMessage, bytes]] = deque()
    self._inbound: deque[BoundaryMessage] = deque()
    self._last_step_cycle: int | None = None
    self.steps_run = 0
    self.live_run_generation = 0
    self.live_launch_generation = 0
    self.stale_plan_rejected = 0
    self.accepted_plans: dict[int, dict[str, int]] = {}

  @property
  def tile_id(self) -> int:
    return self.tile.tile_id

  @property
  def backend_kind(self) -> BackendKind:
    return "python"

  def advance_launch_watermark(self, launch_generation: int) -> None:
    if launch_generation > self.live_launch_generation:
      self.live_launch_generation = launch_generation

  def can_accept_context(self, context_id: int | None = None) -> bool:
    return self.tile.can_accept_context(context_id)

  def outbox_ready(self, channel: str = "CTRL") -> bool:
    return self.credits.pool(channel, "G2T").can_send()

  def send(self, msg: BoundaryMessage) -> bytes:
    """Enqueue one G2T `BoundaryMessage` without stepping `ComputeTile` or `TileUCE`."""
    if not isinstance(msg, BoundaryMessage):
      raise TypeError(f"PythonTileBackend.send requires BoundaryMessage, got {type(msg).__name__}")
    if msg.direction != "G2T":
      raise ValueError(f"PythonTileBackend.send accepts only G2T messages, got {msg.direction}")
    link = (msg.channel, "G2T")
    seq = self._tx_seq.get(link, 0)
    stamped = BoundaryMessage(
      message_type=msg.message_type,
      channel=msg.channel,
      direction="G2T",
      fields=msg.fields,
      message_seq=seq,
      op_seq_hint=msg.op_seq_hint,
    )
    blob = stamped.pack_wire()
    self.credits.pool(msg.channel, "G2T").reserve_on_send()
    self._tx_seq[link] = seq + 1
    self._outbox.append((stamped, blob))
    return blob

  def poll_inbound(self) -> bool:
    return bool(self._inbound)

  def pop_inbound(self) -> BoundaryMessage | None:
    if not self._inbound:
      return None
    msg = self._inbound.popleft()
    self.credits.pool(msg.channel, "T2G").consume_and_return()
    return msg

  def _emit_t2g(self, message_type: str, fields: dict[str, Any]) -> None:
    spec = wire.MESSAGES[wire.MSG_INDEX[message_type]]
    ch = wire.CHANNELS[spec["channel"]]
    link = (ch, "T2G")
    seq = self._tx_seq.get(link, 0)
    msg = BoundaryMessage(
      message_type=message_type, channel=ch, direction="T2G", fields=fields, message_seq=seq
    )
    _ = msg.pack_wire()
    pool = self.credits.pool(ch, "T2G")
    pool.reserve_on_send()
    pool.accept_into_receiver()
    self._tx_seq[link] = seq + 1
    self._inbound.append(msg)

  def _process_outbox(self) -> None:
    while self._outbox:
      msg, blob = self._outbox.popleft()
      g2t_pool = self.credits.pool(msg.channel, "G2T")
      g2t_pool.accept_into_receiver()
      port = (wire.CHANNELS.index(msg.channel), wire.DIRS.index("G2T"))
      decoded = BoundaryMessage.from_wire(blob, expected_port=port)
      code, _words = self._rx.deliver(port, blob)
      g2t_pool.consume_and_return()

      mtype = decoded.message_type
      vals = decoded.fields
      if code == "STALE_GENERATION":
        self.stale_plan_rejected += 1
        if "txn_id" in vals:
          self._emit_t2g(
            "ACK",
            {
              "txn_id": int(vals["txn_id"]),
              "ack_for_message_type": mtype,
              "op_seq": 0,
              "outcome": "STALE",
              "wait_reason": None,
            },
          )
        continue
      if code != "OK":
        if "txn_id" in vals:
          self._emit_t2g(
            "ACK",
            {
              "txn_id": int(vals["txn_id"]),
              "ack_for_message_type": mtype,
              "op_seq": 0,
              "outcome": "FAULT",
              "wait_reason": None,
            },
          )
        continue

      if mtype == "RUN_BEGIN":
        rgen = int(vals["run_generation"])
        if rgen >= self.live_run_generation:
          self.live_run_generation = rgen
      elif mtype in ("TASK_PLAN", "TASK_COMMIT", "TASK_COMMIT_FINAL"):
        lgen = int(vals["launch_generation"])
        # Launch identity is strictly `launch_generation` (and `ctx_generation` via wire.Rx), NEVER
        # an instruction `op_seq_hint`: even a huge op_seq_hint cannot rescue a retired launch.
        if lgen < self.live_launch_generation:
          self.stale_plan_rejected += 1
          self._emit_t2g(
            "ACK",
            {
              "txn_id": int(vals["txn_id"]),
              "ack_for_message_type": mtype,
              "op_seq": 0,
              "outcome": "STALE",
              "wait_reason": None,
            },
          )
          continue
        self.live_launch_generation = lgen
        txn_id = int(vals["txn_id"])
        if mtype == "TASK_PLAN":
          self.accepted_plans[txn_id] = {
            "launch_generation": lgen,
            "dispatch_ordinal": int(vals["dispatch_ordinal"]),
            "task_id": int(vals["task_id"]),
          }
          outcome = "ACCEPTED"
        elif mtype == "TASK_COMMIT":
          outcome = "HOLD_READY"
        else:
          outcome = "ACCEPTED"
        self._emit_t2g(
          "ACK",
          {
            "txn_id": txn_id,
            "ack_for_message_type": mtype,
            "op_seq": 0,
            "outcome": outcome,
            "wait_reason": None,
          },
        )
      elif mtype == "QUERY_CAPS":
        self._emit_t2g(
          "CAPS_RESP",
          {
            "txn_id": int(vals["txn_id"]),
            "context_count": len(self.tile.l1_frames),
            "program_slots": 2,
            "stream_supported": True,
          },
        )

  def step(self, cycle: int, *, freeze_new_work: bool = False) -> None:
    """Advance this Python Tile once in `cycle` (never twice in the same cycle)."""
    if self._last_step_cycle == cycle:
      raise RuntimeError(f"PythonTileBackend(tile={self.tile_id}) stepped twice in cycle {cycle}")
    self._last_step_cycle = cycle
    self.steps_run += 1
    self._process_outbox()
    self.tile.step(cycle, freeze_new_work=freeze_new_work)
    self.credits.assert_conserved()

  def drain_context_terminals(self):
    return self.tile.drain_context_terminals()

  def retire_external_events(self, events: set[str] | tuple[str, ...]) -> None:
    self.tile.retire_external_events(events)

  def request_cancel(self, cycle: int) -> None:
    self.tile.uce._clear_engine_queues(unregister=True)
    self.tile.mfe.begin_reset_drain(cycle)

  def reset(self) -> None:
    self._outbox.clear()
    self._inbound.clear()
    self.credits.reset()
    self._rx = wire.Rx()
    self._tx_seq.clear()
    self._last_step_cycle = None
    self.accepted_plans.clear()
    self.tile.reset()

  def credit_invariant_holds(self) -> bool:
    return self.credits.conserved()

  def snapshot(self) -> dict[str, Any]:
    snap = self.tile.snapshot()
    return {
      **snap,
      "backend_kind": self.backend_kind,
      "port_credits_ok": self.credits.conserved(),
      "stale_plan_rejected": self.stale_plan_rejected,
    }
