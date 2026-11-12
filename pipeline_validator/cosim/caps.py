"""QUERY_CAPS / CAPS_RESP capability negotiation service (C024 / W1-02).

Serves the CTRL-channel capability query at the Group side using values sourced from the live
TileGroup configuration (context_count, program slots, STREAM support) -- never hard-coded
echoes. Every request/response is a `BoundaryMessage` validated through the generated `wire.py`.
"""

from __future__ import annotations

from .tile_port import BoundaryMessage


class CapsServiceError(RuntimeError):
  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


class CapsService:
  """Answers QUERY_CAPS with CAPS_RESP built from real Group configuration."""

  def __init__(self, *, context_count: int, program_slots: int, stream_supported: bool) -> None:
    for name, val in (("context_count", context_count), ("program_slots", program_slots)):
      if isinstance(val, bool) or not isinstance(val, int) or val < 1:
        raise ValueError(f"CapsService.{name} must be a positive integer")
    if not isinstance(stream_supported, bool):
      raise TypeError("stream_supported must be bool")
    self.context_count = context_count
    self.program_slots = program_slots
    self.stream_supported = stream_supported
    self.stats = {"queries": 0, "responses": 0}

  def handle(self, msg: BoundaryMessage) -> BoundaryMessage:
    if msg.message_type != "QUERY_CAPS" or msg.direction != "G2T":
      raise CapsServiceError("BAD_CAPS_REQUEST", f"{msg.message_type}/{msg.direction}")
    self.stats["queries"] += 1
    resp = BoundaryMessage(
      message_type="CAPS_RESP",
      channel="CTRL",
      direction="T2G",
      fields={
        "txn_id": int(msg.fields["txn_id"]),
        "context_count": self.context_count,
        "program_slots": self.program_slots,
        "stream_supported": self.stream_supported,
      },
      message_seq=0,
    )
    _ = resp.pack_wire()  # encode once: rejects any field-level drift against the wire schema
    self.stats["responses"] += 1
    return resp
