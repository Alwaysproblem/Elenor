"""TilePort boundary protocol, pure-value guard, and channel credit ledger (C022 / W1-02).

Invariants (execution_guide.md §6 & W1_Boundary_Cosimulation_Plan.md §8.5):
- No live runtime objects cross the boundary: `validate_boundary_value` accepts only scalar
  primitives (`int`, `bool`, `str`, `bytes`, `memoryview`, `None`), frozen pure-value keys
  (`TaskKey`, `ContextKey`), and read-only tuples/mappings of those values. Passing a mutable or
  runtime object (`ComputeTile`, `TileUCE`, `AllocationHandle`, `ExecTileProgram`, etc.) raises
  `TypeError`.
- `send(msg)` only validates and enqueues outbound G2T messages; it NEVER advances simulation
  state or calls `UCE.step`.
- `step(cycle)` advances a Tile at most once per integer `cycle` (double-step in the same cycle
  raises `RuntimeError`).
- Channel flow-control credits satisfy `available + in_flight + occupancy == capacity` on every
  `(channel, direction)` link at every tick (`CreditConservationError` on underflow, overflow, or
  unmatched return).
- Unit-only stubs carry `backend_kind == "stub"` and are rejected by `assert_cs1_backend_kind`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from . import wire

BackendKind = Literal["python", "rtl", "stub"]
VALID_BACKEND_KINDS: frozenset[str] = frozenset({"python", "rtl", "stub"})
CS1_ELIGIBLE_BACKEND_KINDS: frozenset[str] = frozenset({"python", "rtl"})

# Default per-(channel, dir) boundary FIFO credit capacities derived from C009 channel_budgets.yaml
# transaction_capacity entries (CTRL=16 dispatch_capacity, EVENT=64 bounded window, PROGRAM=8,
# MEMORY=32 inflight_capacity, DEBUG=16).
DEFAULT_CHANNEL_CREDITS: dict[str, int] = {"CTRL": 16, "EVENT": 64, "PROGRAM": 8, "MEMORY": 32, "DEBUG": 16}


class CreditConservationError(RuntimeError):
  """Raised when channel credit conservation (`available + in_flight + occupancy == cap`) breaks."""


class StaleGenerationError(RuntimeError):
  """Raised or recorded when a message carries a retired launch/context generation."""


@dataclass(frozen=True)
class TaskKey:
  """Pure-value Task identity matching `wire_schema_v1.yaml` `task_key`."""

  launch_generation: int
  root_launch_generation: int
  dispatch_ordinal: int
  task_id: int
  context_handle: int = 0

  def __post_init__(self) -> None:
    for name in (
      "launch_generation",
      "root_launch_generation",
      "dispatch_ordinal",
      "task_id",
      "context_handle",
    ):
      val = getattr(self, name)
      if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        raise TypeError(f"TaskKey.{name} must be a non-negative int, got {val!r}")

  def to_wire_fields(self) -> dict[str, int]:
    return {
      "launch_generation": self.launch_generation,
      "root_launch_generation": self.root_launch_generation,
      "dispatch_ordinal": self.dispatch_ordinal,
      "task_id": self.task_id,
      "context_handle": self.context_handle,
    }


@dataclass(frozen=True)
class ContextKey:
  """Pure-value Context identity matching `wire_schema_v1.yaml` `context_key`."""

  tile_id: int
  ctx_slot: int
  ctx_generation: int

  def __post_init__(self) -> None:
    for name in ("tile_id", "ctx_slot", "ctx_generation"):
      val = getattr(self, name)
      if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        raise TypeError(f"ContextKey.{name} must be a non-negative int, got {val!r}")

  def to_wire_fields(self) -> dict[str, int]:
    return {"tile_id": self.tile_id, "ctx_slot": self.ctx_slot, "ctx_generation": self.ctx_generation}


def validate_boundary_value(value: Any, *, path: str = "msg") -> None:
  """Reject any non-value Python runtime object crossing `TilePort`."""
  if value is None or isinstance(value, (bool, int, str, bytes, memoryview, TaskKey, ContextKey)):
    return
  if isinstance(value, tuple):
    for idx, item in enumerate(value):
      validate_boundary_value(item, path=f"{path}[{idx}]")
    return
  if isinstance(value, Mapping):
    for k, v in value.items():
      if not isinstance(k, str):
        raise TypeError(f"{path}: mapping key must be str, got {type(k).__name__}")
      validate_boundary_value(v, path=f"{path}.{k}")
    return
  raise TypeError(f"{path}: runtime object of type {type(value).__name__} may not cross TilePort")


@dataclass(frozen=True)
class BoundaryMessage:
  """Pure-value logical boundary message paired with its C017 wire encoding."""

  message_type: str
  channel: str
  direction: str  # "G2T" | "T2G"
  fields: dict[str, Any] = field(default_factory=dict)
  message_seq: int = 0
  op_seq_hint: int | None = None  # diagnostic only; NEVER used as launch or transport identity

  def __post_init__(self) -> None:
    if self.message_type not in wire.MSG_INDEX:
      raise ValueError(f"unknown boundary message_type {self.message_type!r}")
    spec = wire.MESSAGES[wire.MSG_INDEX[self.message_type]]
    expected_ch = wire.CHANNELS[spec["channel"]]
    expected_dir = wire.DIRS[spec["dir"]]
    if self.channel != expected_ch or self.direction != expected_dir:
      raise ValueError(
        f"{self.message_type} belongs on ({expected_ch}, {expected_dir}), "
        f"got ({self.channel}, {self.direction})"
      )
    if isinstance(self.message_seq, bool) or not isinstance(self.message_seq, int) or self.message_seq < 0:
      raise TypeError("message_seq must be a non-negative int")
    if self.op_seq_hint is not None and (
      isinstance(self.op_seq_hint, bool) or not isinstance(self.op_seq_hint, int)
    ):
      raise TypeError("op_seq_hint must be int or None")
    validate_boundary_value(self.fields, path=f"{self.message_type}.fields")

  def flatten_wire_values(self) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in self.fields.items():
      if isinstance(v, (TaskKey, ContextKey)):
        out.update(v.to_wire_fields())
      elif isinstance(v, memoryview):
        out[k] = v.tobytes()
      else:
        out[k] = v
    return out

  def pack_wire(self) -> bytes:
    return wire.pack(self.message_type, self.flatten_wire_values(), message_seq=self.message_seq)

  @classmethod
  def from_wire(cls, blob: bytes, *, expected_port: tuple[int, int] | None = None) -> BoundaryMessage:
    decoded = wire.unpack(blob, port=expected_port)
    spec = wire.MESSAGES[wire.MSG_INDEX[decoded.message]]
    return cls(
      message_type=decoded.message,
      channel=wire.CHANNELS[spec["channel"]],
      direction=wire.DIRS[spec["dir"]],
      fields=dict(decoded.values),
      message_seq=decoded.message_seq,
    )


@dataclass
class LinkCreditPool:
  """Single `(channel, direction)` credit pool enforcing `available + in_flight + occupancy == capacity`."""

  channel: str
  direction: str
  capacity: int
  available: int = field(init=False)
  in_flight: int = 0
  occupancy: int = 0

  def __post_init__(self) -> None:
    if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity < 1:
      raise ValueError(f"credit capacity for {self.channel}/{self.direction} must be >= 1")
    self.available = self.capacity

  def can_send(self) -> bool:
    return self.available > 0

  def reserve_on_send(self) -> None:
    if self.available <= 0:
      raise CreditConservationError(f"{self.channel}/{self.direction}: send with zero available credit")
    self.available -= 1
    self.in_flight += 1
    self.assert_conserved()

  def accept_into_receiver(self) -> None:
    if self.in_flight <= 0:
      raise CreditConservationError(f"{self.channel}/{self.direction}: accept with zero in_flight")
    self.in_flight -= 1
    self.occupancy += 1
    self.assert_conserved()

  def consume_and_return(self) -> None:
    if self.occupancy <= 0:
      raise CreditConservationError(
        f"{self.channel}/{self.direction}: duplicate or unmatched credit return (occupancy=0)"
      )
    self.occupancy -= 1
    self.available += 1
    self.assert_conserved()

  def reset(self) -> None:
    self.available = self.capacity
    self.in_flight = 0
    self.occupancy = 0

  def conserved(self) -> bool:
    return (
      self.available >= 0
      and self.in_flight >= 0
      and self.occupancy >= 0
      and (self.available + self.in_flight + self.occupancy == self.capacity)
    )

  def assert_conserved(self) -> None:
    if not self.conserved():
      raise CreditConservationError(
        f"{self.channel}/{self.direction} credit invariant broken: "
        f"avail={self.available} + inflight={self.in_flight} + occ={self.occupancy} != cap={self.capacity}"
      )

  def snapshot(self) -> dict[str, int]:
    return {
      "capacity": self.capacity,
      "available": self.available,
      "in_flight": self.in_flight,
      "occupancy": self.occupancy,
    }


class ChannelCreditLedger:
  """Independent per-(channel, direction) credit pools for one TilePort."""

  def __init__(self, capacities: Mapping[str, int] | None = None) -> None:
    caps = dict(DEFAULT_CHANNEL_CREDITS)
    if capacities is not None:
      caps.update(capacities)
    self.pools: dict[tuple[str, str], LinkCreditPool] = {}
    for ch in wire.CHANNELS:
      for d in ("G2T", "T2G"):
        self.pools[(ch, d)] = LinkCreditPool(channel=ch, direction=d, capacity=caps[ch])

  def pool(self, channel: str, direction: str) -> LinkCreditPool:
    try:
      return self.pools[(channel, direction)]
    except KeyError as exc:
      raise ValueError(f"unknown boundary link ({channel!r}, {direction!r})") from exc

  def conserved(self) -> bool:
    return all(p.conserved() for p in self.pools.values())

  def assert_conserved(self) -> None:
    for p in self.pools.values():
      p.assert_conserved()

  def reset(self) -> None:
    for p in self.pools.values():
      p.reset()

  def snapshot(self) -> dict[str, dict[str, int]]:
    return {f"{ch}.{d}": pool.snapshot() for (ch, d), pool in self.pools.items()}


def assert_cs1_backend_kind(kind: str) -> None:
  """Refuse `stub` or unknown backend kinds on any CS1-eligible execution path."""
  if kind == "stub":
    raise ValueError("unit-only stub backend_kind='stub' is forbidden as CS1 Tile evidence")
  if kind not in CS1_ELIGIBLE_BACKEND_KINDS:
    allowed = sorted(CS1_ELIGIBLE_BACKEND_KINDS)
    raise ValueError(f"unsupported backend_kind {kind!r}; expected one of {allowed}")


@runtime_checkable
class TilePort(Protocol):
  """Group-facing boundary port of a single Tile (`python`, `rtl`, or unit-only `stub`)."""

  @property
  def tile_id(self) -> int: ...

  @property
  def backend_kind(self) -> BackendKind: ...

  def can_accept_context(self, context_id: int | None = None) -> bool: ...

  def outbox_ready(self, channel: str = "CTRL") -> bool: ...

  def send(self, msg: BoundaryMessage) -> bytes: ...

  def poll_inbound(self) -> bool: ...

  def pop_inbound(self) -> BoundaryMessage | None: ...

  def step(self, cycle: int, *, freeze_new_work: bool = False) -> None: ...

  def retire_external_events(self, events: set[str] | tuple[str, ...]) -> None: ...

  def request_cancel(self, cycle: int) -> None: ...

  def reset(self) -> None: ...

  def credit_invariant_holds(self) -> bool: ...

  def snapshot(self) -> dict[str, Any]: ...


class TileProtocolStub:
  """Unit-only `TilePort` protocol stub (`backend_kind == 'stub'`, no C ABI, no Verilator).

  Used strictly by unit tests (`test_w1_port.py`) to verify `TilePort` structural typing,
  `wire.py` pack/unpack round-tripping, credit conservation, and stale-generation rejection.
  `assert_cs1_backend_kind` rejects `"stub"` unconditionally.
  """

  def __init__(self, tile_id: int, *, context_count: int = 2) -> None:
    self._tile_id = tile_id
    self._context_count = context_count
    self.credits = ChannelCreditLedger()
    self._rx = wire.Rx()
    self._tx_seq: dict[tuple[str, str], int] = {}
    self._outbox: deque[tuple[BoundaryMessage, bytes]] = deque()
    self._inbound: deque[BoundaryMessage] = deque()
    self._last_step_cycle: int | None = None
    self.steps_run = 0
    self.stale_rejected = 0
    self.live_launch_generation = 0

  @property
  def tile_id(self) -> int:
    return self._tile_id

  @property
  def backend_kind(self) -> BackendKind:
    return "stub"

  def can_accept_context(self, context_id: int | None = None) -> bool:
    if context_id is not None:
      return 0 <= context_id < self._context_count
    return self._context_count > 0

  def outbox_ready(self, channel: str = "CTRL") -> bool:
    return self.credits.pool(channel, "G2T").can_send()

  def send(self, msg: BoundaryMessage) -> bytes:
    if not isinstance(msg, BoundaryMessage):
      raise TypeError(f"TilePort.send requires BoundaryMessage, got {type(msg).__name__}")
    if msg.direction != "G2T":
      raise ValueError(f"TilePort.send accepts only G2T messages, got {msg.direction}")
    link = (msg.channel, "G2T")
    seq = self._tx_seq.get(link, 0)
    stamped = BoundaryMessage(
      message_type=msg.message_type,
      channel=msg.channel,
      direction=msg.direction,
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

  def step(self, cycle: int, *, freeze_new_work: bool = False) -> None:
    if self._last_step_cycle == cycle:
      raise RuntimeError(f"Tile {self._tile_id} stepped twice in cycle {cycle}")
    self._last_step_cycle = cycle
    self.steps_run += 1
    _ = freeze_new_work
    while self._outbox:
      msg, blob = self._outbox.popleft()
      g2t_pool = self.credits.pool(msg.channel, "G2T")
      g2t_pool.accept_into_receiver()
      port = (wire.CHANNELS.index(msg.channel), wire.DIRS.index("G2T"))
      decoded = BoundaryMessage.from_wire(blob, expected_port=port)
      code, _words = self._rx.deliver(port, blob)
      g2t_pool.consume_and_return()
      if code == "STALE_GENERATION":
        self.stale_rejected += 1
        continue
      if decoded.message_type == "TASK_PLAN":
        lgen = int(decoded.fields["launch_generation"])
        if lgen < self.live_launch_generation:
          self.stale_rejected += 1
          self._emit_t2g(
            "ACK",
            {
              "txn_id": int(decoded.fields["txn_id"]),
              "ack_for_message_type": "TASK_PLAN",
              "op_seq": 0,
              "outcome": "STALE",
              "wait_reason": None,
            },
          )
          continue
        self.live_launch_generation = lgen
        self._emit_t2g(
          "ACK",
          {
            "txn_id": int(decoded.fields["txn_id"]),
            "ack_for_message_type": "TASK_PLAN",
            "op_seq": 0,
            "outcome": "ACCEPTED",
            "wait_reason": None,
          },
        )
    self.credits.assert_conserved()

  def retire_external_events(self, events: set[str] | tuple[str, ...]) -> None:
    _ = events

  def request_cancel(self, cycle: int) -> None:
    _ = cycle

  def reset(self) -> None:
    self._outbox.clear()
    self._inbound.clear()
    self.credits.reset()
    self._rx = wire.Rx()
    self._tx_seq.clear()
    self._last_step_cycle = None
    self.live_launch_generation = 0

  def credit_invariant_holds(self) -> bool:
    return self.credits.conserved()

  def snapshot(self) -> dict[str, Any]:
    return {
      "tile_id": self._tile_id,
      "backend_kind": self.backend_kind,
      "steps_run": self.steps_run,
      "stale_rejected": self.stale_rejected,
      "credits": self.credits.snapshot(),
    }
