"""STREAM logical sub-queue on the MEMORY channel with independent backpressure (C024 / W1-02).

The parent ruling (W1 plan §13 item 1): STREAM rides the MEMORY channel as an independent
LOGICAL sub-queue with its own bounded budget (`stream_depth_default`, hardware_config.yaml
line 147 = 3); it adds no sixth physical channel. Backpressure here is independent of the
MEMORY main-queue credit: filling the STREAM sub-queue stalls only STREAM traffic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .tile_port import BoundaryMessage

STREAM_OP_WORDS = ("STREAM_ACQUIRE", "STREAM_POP", "STREAM_PUSH", "STREAM_RELEASE")
DEFAULT_STREAM_DEPTH = 3


class StreamSubqueueError(RuntimeError):
  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


@dataclass
class _StreamEntry:
  token_id: int
  seq: int
  kind: str


class StreamSubqueue:
  """One bounded STREAM token queue on the MEMORY channel, independent of the main queue."""

  def __init__(self, *, depth: int = DEFAULT_STREAM_DEPTH) -> None:
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
      raise ValueError("stream sub-queue depth must be >= 1")
    self.depth = depth
    self._entries: deque[_StreamEntry] = deque()
    self._memory_main_queue_backpressured = False
    self.stats = {
      "acquire": 0,
      "pop": 0,
      "push": 0,
      "release": 0,
      "stall_cycles_stream_full": 0,
      "stall_cycles_memory_full": 0,
    }

  def __len__(self) -> int:
    return len(self._entries)

  def _validate(self, msg: BoundaryMessage) -> tuple[int, int]:
    if msg.message_type not in STREAM_OP_WORDS or msg.channel != "MEMORY":
      raise StreamSubqueueError("BAD_STREAM_MESSAGE", msg.message_type)
    token = int(msg.fields["token_id"])
    seq = int(msg.fields["seq"])
    if isinstance(token, bool) or token < 0 or isinstance(seq, bool) or seq < 0:
      raise StreamSubqueueError("BAD_STREAM_IDENTITY", f"token={token} seq={seq}")
    return token, seq

  def handle(self, msg: BoundaryMessage, *, cycle: int = 0) -> str:
    """Execute one STREAM op; STREAM_PUSH on a full sub-queue stalls ONLY stream traffic."""
    _token, seq = self._validate(msg)
    op = msg.message_type
    if op == "STREAM_ACQUIRE":
      self.stats["acquire"] += 1
      return "acquired"
    if op == "STREAM_POP":
      if not self._entries:
        raise StreamSubqueueError("POP_EMPTY", f"seq={seq}")
      entry = self._entries.popleft()
      self.stats["pop"] += 1
      return f"popped:{entry.token_id}"
    if op == "STREAM_PUSH":
      if len(self._entries) >= self.depth:
        # Independent backpressure: the STREAM sub-queue stalls; the MEMORY main queue is
        # untouched (observable via `memory_main_stalled` staying False).
        self.stats["stall_cycles_stream_full"] += 1
        return "stall:stream_full"
      self._entries.append(_StreamEntry(token_id=_token, seq=seq, kind="push"))
      self.stats["push"] += 1
      return "pushed"
    self.stats["release"] += 1
    return "released"

  def note_memory_main_backpressure(self, stalled: bool, *, cycle: int = 0) -> None:
    """Record MEMORY main-queue backpressure WITHOUT stalling the STREAM sub-queue."""
    if stalled == self._memory_main_queue_backpressured:
      return
    self._memory_main_queue_backpressured = stalled
    if stalled:
      self.stats["stall_cycles_memory_full"] += 1

  @property
  def memory_main_stalled(self) -> bool:
    return self._memory_main_queue_backpressured

  def full(self) -> bool:
    return len(self._entries) >= self.depth

  def snapshot(self) -> dict:
    return {
      "depth": self.depth,
      "occupancy": len(self._entries),
      "full": self.full(),
      "memory_main_stalled": self.memory_main_stalled,
      "stats": dict(self.stats),
    }
