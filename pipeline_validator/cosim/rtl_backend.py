"""RTL / stub TilePort backend driving the 5-phase CN C ABI (`bp_drive -> bp_tick -> bp_collect`) (C025).

Responsibilities:
- Implements the full `TilePort` protocol (`send`, `poll_inbound`, `pop_inbound`, `outbox_ready`,
  `step`, `reset`, `credit_invariant_holds`, `snapshot`) with ZERO compute paths on the Python side.
- `send(msg)` packs real `BoundaryMessage` frames through `wire.py` (e.g. 96B `TASK_PLAN`,
  104B `TASK_COMMIT`, 128B `PROGRAM_INSTALL_ACK`, 136B `MEM_REQ`), assigns monotonic per-link
  `message_seq`, reserves `(channel, "G2T")` credit, and queues the frame without advancing ticks.
- `step(cycle)` executes `bp_drive -> bp_tick -> bp_collect` once per integer cycle:
  - Holds any unaccepted (`valid && !ready`) G2T frame stable across ticks until the C ABI reports
    its P1 acceptance receipt;
  - Decodes every T2G frame returned by the DUT's T2G pins via `BoundaryMessage.from_wire`,
    verifies link port/sequence/ranges, and places it in `_inbound`;
  - On `CN_ERR_FATAL_DUT`, freezes further ticks and writes an `INCOMPLETE_FATAL` record with the
    residual pending set (never fabricating a clean drain).
- Unit-stub sessions carry `backend_kind == "stub"` so `assert_cs1_backend_kind` rejects them for
  any CS1 gate claim; C040 switches `is_unit_stub=False` (`backend_kind == "rtl"`) when linked
  against the real oneTile top.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import abi as cn_abi, wire
from .tile_port import BackendKind, BoundaryMessage, ChannelCreditLedger


class RtlTileBackend:
  """Full `TilePort` implementation over the eight-entry CN C ABI."""

  def __init__(
    self,
    tile_id: int,
    *,
    library: Path | str = cn_abi.DEFAULT_STUB,
    is_unit_stub: bool = True,
    ready_seed: int = 1,
    inject_fatal_at_tick: int = -1,
    inject_abort_at_tick: int = -1,
    stub_flags: int = 0,
    incomplete_out_dir: Path | None = None,
    channel_capacities: Mapping[str, int] | None = None,
  ) -> None:
    self._tile_id = tile_id
    self._is_unit_stub = is_unit_stub
    self._ready_seed = ready_seed
    self._inject_fatal_at_tick = inject_fatal_at_tick
    self._inject_abort_at_tick = inject_abort_at_tick
    self._stub_flags = stub_flags
    self._incomplete_out_dir = incomplete_out_dir
    self.credits = ChannelCreditLedger(channel_capacities)
    self.abi = cn_abi.load(library)

    cfg = self._make_cfg()
    self.abi.begin(cfg)
    self.slot_specs = self.abi.layout()
    self._g2t_slot_by_ch: dict[str, int] = {}
    self._ch_by_slot: dict[int, tuple[str, str, int, int]] = {}
    for spec in self.slot_specs:
      ch = str(spec["channel"])
      d = str(spec["dir"])
      sid = int(spec["slot_id"])
      ch_idx = int(spec["channel_index"])
      dir_idx = int(spec["dir_index"])
      self._ch_by_slot[sid] = (ch, d, ch_idx, dir_idx)
      if d == "G2T":
        self._g2t_slot_by_ch[ch] = sid

    self._tx_seq: dict[tuple[str, str], int] = {}
    self._g2t_queues: dict[str, deque[tuple[BoundaryMessage, bytes]]] = {
      ch: deque() for ch in wire.CHANNELS
    }
    self._g2t_inflight: dict[str, tuple[BoundaryMessage, bytes] | None] = dict.fromkeys(
      wire.CHANNELS, None
    )
    self._t2g_ready_override: dict[str, int] = {}
    self._inbound: deque[BoundaryMessage] = deque()
    self._last_step_cycle: int | None = None
    self._last_posedge = int(self.abi.status().posedge_count)
    self.fatal_incomplete: dict[str, Any] | None = None

  def _make_cfg(self) -> cn_abi.CnBeginCfg:
    cfg = cn_abi.CnBeginCfg()
    cfg.abi_version = cn_abi.CN_ABI_VERSION
    cfg.tile_count = 1
    cfg.ready_seed = self._ready_seed & 0xFFFFFFFF
    cfg.inject_fatal_at_tick = self._inject_fatal_at_tick
    cfg.inject_abort_at_tick = self._inject_abort_at_tick
    cfg.trace = 0
    cfg.reserved[0] = self._stub_flags & 0xFF
    return cfg

  @property
  def tile_id(self) -> int:
    return self._tile_id

  @property
  def backend_kind(self) -> BackendKind:
    return "stub" if self._is_unit_stub else "rtl"

  def can_accept_context(self, context_id: int | None = None) -> bool:
    _ = context_id
    return self.fatal_incomplete is None and self.outbox_ready("CTRL")

  def outbox_ready(self, channel: str = "CTRL") -> bool:
    return self.credits.pool(channel, "G2T").can_send()

  def set_t2g_ready(self, channel: str, ready: bool) -> None:
    """Control Group-side readiness on a T2G channel to test T2G backpressure."""
    if channel not in wire.CHANNELS:
      raise ValueError(f"unknown channel {channel!r}")
    self._t2g_ready_override[channel] = 1 if ready else 0

  def send(self, msg: BoundaryMessage) -> bytes:
    """Pack `msg` via `wire.py` and enqueue on its G2T slot without advancing simulation."""
    if self.fatal_incomplete is not None:
      raise cn_abi.CnAbiError("DUT_FATAL_LATCHED", "refusing send after INCOMPLETE_FATAL")
    if not isinstance(msg, BoundaryMessage):
      raise TypeError(f"RtlTileBackend.send requires BoundaryMessage, got {type(msg).__name__}")
    if msg.direction != "G2T":
      raise ValueError(f"RtlTileBackend.send accepts only G2T messages, got {msg.direction}")
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
    self._g2t_queues[msg.channel].append((stamped, blob))
    return blob

  def poll_inbound(self) -> bool:
    return bool(self._inbound)

  def pop_inbound(self) -> BoundaryMessage | None:
    if not self._inbound:
      return None
    msg = self._inbound.popleft()
    self.credits.pool(msg.channel, "T2G").consume_and_return()
    return msg

  def step(self, cycle: int, *, freeze_new_work: bool = False) -> None:
    if self._last_step_cycle == cycle:
      raise RuntimeError(f"RtlTileBackend(tile={self._tile_id}) stepped twice in cycle {cycle}")
    if self.fatal_incomplete is not None:
      raise cn_abi.CnAbiError("DUT_FATAL_LATCHED", "cannot continue stepping after DUT fatal")
    self._last_step_cycle = cycle

    drive_items: list[tuple[int, bytes | memoryview]] = []
    for ch in wire.CHANNELS:
      sid = self._g2t_slot_by_ch[ch]
      if self._g2t_inflight[ch] is None and not freeze_new_work and self._g2t_queues[ch]:
        self._g2t_inflight[ch] = self._g2t_queues[ch].popleft()
      inflight = self._g2t_inflight[ch]
      if inflight is not None:
        drive_items.append((sid, inflight[1]))
      else:
        drive_items.append((sid, b""))

    for spec in self.slot_specs:
      if spec["dir"] == "T2G":
        ch = str(spec["channel"])
        if ch in self._t2g_ready_override:
          drive_items.append((int(spec["slot_id"]), bytes([self._t2g_ready_override[ch]])))

    drc = self.abi.drive(drive_items)
    if drc != cn_abi.CN_OK:
      raise cn_abi.CnAbiError("DRIVE_ERROR", f"rc={drc}")

    trc = self.abi.tick()
    st = self.abi.status()
    if int(st.posedge_count) != self._last_posedge + 1:
      raise cn_abi.CnAbiError(
        "POSEDGE_INVARIANT",
        f"expected posedge {self._last_posedge + 1}, got {int(st.posedge_count)}",
      )
    self._last_posedge = int(st.posedge_count)

    if trc == cn_abi.CN_ERR_FATAL_DUT:
      self._record_incomplete_fatal(cycle, st)
      raise cn_abi.CnAbiError("DUT_FATAL", f"fatal_code={int(st.fatal_code)} at cycle {cycle}")
    if trc != cn_abi.CN_OK:
      raise cn_abi.CnAbiError("TICK_ERROR", f"rc={trc} proto_err={int(st.proto_err)}")

    crc, records = self.abi.collect()
    if crc != cn_abi.CN_OK:
      raise cn_abi.CnAbiError("COLLECT_ERROR", f"rc={crc}")

    for rec in records:
      ch, d, ch_idx, dir_idx = self._ch_by_slot[rec.slot_id]
      if not rec.is_t2g:
        assert d == "G2T"
        g2t_pool = self.credits.pool(ch, "G2T")
        g2t_pool.accept_into_receiver()
        g2t_pool.consume_and_return()
        self._g2t_inflight[ch] = None
      else:
        assert d == "T2G"
        decoded = BoundaryMessage.from_wire(rec.wire_bytes, expected_port=(ch_idx, dir_idx))
        t2g_pool = self.credits.pool(ch, "T2G")
        t2g_pool.reserve_on_send()
        t2g_pool.accept_into_receiver()
        self._inbound.append(decoded)

    self.credits.assert_conserved()

  def _record_incomplete_fatal(self, cycle: int, st: cn_abi.CnStatus) -> None:
    pending: list[dict[str, Any]] = []
    for ch in wire.CHANNELS:
      inflight = self._g2t_inflight[ch]
      if inflight is not None:
        pending.append(
          {
            "channel": ch,
            "state": "in_flight_unaccepted",
            "message_type": inflight[0].message_type,
            "message_seq": inflight[0].message_seq,
          }
        )
      for queued_msg, _blob in self._g2t_queues[ch]:
        pending.append(
          {
            "channel": ch,
            "state": "queued_outbox",
            "message_type": queued_msg.message_type,
            "message_seq": queued_msg.message_seq,
          }
        )
    doc = {
      "schema": "cn_incomplete/v1",
      "verdict": "INCOMPLETE_FATAL",
      "tile_id": self._tile_id,
      "backend_kind": self.backend_kind,
      "cycle": cycle,
      "fatal_code": int(st.fatal_code),
      "posedge_count": int(st.posedge_count),
      "pending_set": pending,
    }
    self.fatal_incomplete = doc
    if self._incomplete_out_dir is not None:
      self._incomplete_out_dir.mkdir(parents=True, exist_ok=True)
      (self._incomplete_out_dir / "incomplete_fatal.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n"
      )

  def retire_external_events(self, events: set[str] | tuple[str, ...]) -> None:
    _ = events

  def request_cancel(self, cycle: int) -> None:
    _ = cycle

  def reset(self) -> None:
    self.abi.destroy()
    self.abi = cn_abi.load(self.abi.path)
    self.abi.begin(self._make_cfg())
    self.slot_specs = self.abi.layout()
    for q in self._g2t_queues.values():
      q.clear()
    for ch in wire.CHANNELS:
      self._g2t_inflight[ch] = None
    self._t2g_ready_override.clear()
    self._inbound.clear()
    self._tx_seq.clear()
    self.credits.reset()
    self._last_step_cycle = None
    self._last_posedge = int(self.abi.status().posedge_count)
    self.fatal_incomplete = None

  def close(self) -> None:
    self.abi.destroy()

  def credit_invariant_holds(self) -> bool:
    return self.credits.conserved()

  def snapshot(self) -> dict[str, Any]:
    st = self.abi.status()
    return {
      "tile_id": self._tile_id,
      "backend_kind": self.backend_kind,
      "tick_count": int(st.tick_count),
      "posedge_count": int(st.posedge_count),
      "fatal_code": int(st.fatal_code),
      "proto_err": int(st.proto_err),
      "staged_pending": int(st.staged_pending),
      "credits": self.credits.snapshot(),
      "fatal_incomplete": self.fatal_incomplete,
    }


RtlStubBackend = RtlTileBackend
