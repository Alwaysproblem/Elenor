"""C039 co-simulation driver (W5-04): the live Group against the real RTL install endpoint.

Python is the sole time authority. A `TileProgramRtl` drives one compiled `tb_c039_cosim`
(the real `w5/install_service` over the real C030 program port) one bounded command at a
time. A `GroupProgramLink` owns the Group half of the same PROGRAM link: the real
`TileGroup.program_table` residency manager, its attached `ProgramService`, the sealed C017
codec and the sealed receive model. Every G2T frame the Tile accepts is a manager outbox
message; every T2G frame the Tile emits is decoded and fed back to the manager. The driver
never fabricates a Tile message, never rewrites a manager field and never completes an
install on its own.

Scenario steps are declared in the scenario YAML (`steps:`); this module executes them and
records what happened. A failed expectation is reported to the bench (`0a`), which terminates
with `C039_ASSERT tb_c039_cosim peer_reject`, so a mutated Tile dies by the verdict of the
real Group software that observed it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import select
import struct
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline_validator.config import HardwareConfig
from pipeline_validator.cosim import wire
from pipeline_validator.cosim.program_prepare import prepare_programs
from pipeline_validator.cosim.program_service import ProgramImage
from pipeline_validator.cosim.tile_port import BoundaryMessage
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.runtime.program_table import ProgramIdentity, ResidencyError
from pipeline_validator.tile_group import TileGroup

MASK64 = (1 << 64) - 1
MAX_TICKS = 8192  # the bench bound: one command never advances more RTL ticks
LINE_BYTES = 32  # fetchDataBits/8: one C030 fetch line
PROGRAM_G2T = (wire.CHANNELS.index("PROGRAM"), wire.DIRS.index("G2T"))
PROGRAM_T2G = (wire.CHANNELS.index("PROGRAM"), wire.DIRS.index("T2G"))

# Bench command opcodes (tb_c039_cosim.sv header).
(
  OP_DELIVER,
  OP_WAIT,
  OP_PROBE,
  OP_REF,
  OP_T2G_READY,
  OP_FETCH_READY,
  OP_SETUP,
  OP_STATUS,
  OP_RESET,
  OP_VERDICT,
  OP_QUIT,
) = range(1, 12)
ST_OK, ST_EVENT, ST_NOT_ACCEPTED, ST_NO_EVENT = 0, 1, 2, 4
EV_FRAME, EV_FAULT, EV_FETCH_ACCEPT, EV_FETCH_RESPONSE, EV_INSTALL_REJECT, EV_SETUP = range(1, 7)
TAIL_WORDS = 24
# Verdict codes the bench prints with its C039_ASSERT line.
V_EXPECTATION, V_BOUNDED_WAIT, V_PROTOCOL = 1, 2, 3

# Sealed C017 fault-code order (the RTL carries these values, c034_reset_pkg FC_*).
FAULT_CODES = tuple(wire.ENUMS["fault_code"])
# l1_program_port.sv fetch response codes: module localparams, mirrored for the transcript.
FETCH_CODES = ("F_OK", "F_RANGE", "F_ALIGN", "F_ZERO", "F_LEN", "F_UNWRITTEN")
MSG_NAMES = {
  wire.MSG_INDEX[name]: name
  for name in (
    "PROGRAM_INSTALL",
    "PROGRAM_READ_REQ",
    "PROGRAM_READ_RESP",
    "PROGRAM_INSTALL_ACK",
    "PROGRAM_READY",
    "PROGRAM_INVALIDATE",
    "INVALIDATE_ACK",
  )
}


class CosimError(AssertionError):
  """A scenario expectation failed against the live Tile."""

  def __init__(self, verdict: int, message: str, key: str | None = None) -> None:
    super().__init__(message)
    self.verdict = verdict
    self.key = key


class BoundedWait(CosimError):
  def __init__(self, message: str) -> None:
    super().__init__(V_BOUNDED_WAIT, message, "bounded_wait")


class ProtocolError(CosimError):
  def __init__(self, message: str) -> None:
    super().__init__(V_PROTOCOL, message, "protocol")


class ExpectationFailed(CosimError):
  def __init__(self, key: str, expected: Any, observed: Any) -> None:
    super().__init__(V_EXPECTATION, f"{key}: expected {expected!r}, observed {observed!r}", key)
    self.expected = expected
    self.observed = observed


class EndpointGone(RuntimeError):
  """The bench for this label exited, or never built: no further reply can arrive."""


def frame_words(blob: bytes) -> list[int]:
  """Little-endian 64-bit words of one sealed frame."""
  if len(blob) % 8:
    raise ValueError("a wire frame is a whole number of 64-bit words")
  return list(struct.unpack(f"<{len(blob) // 8}Q", blob))


def blob_words(data: bytes) -> list[int]:
  """Blob words of one READ_RESP chunk; the last word is zero-padded like the wire carries it."""
  return frame_words(data + bytes((-len(data)) % 8))


def g2t_message(kind: str, fields: Mapping[str, Any]) -> BoundaryMessage:
  return BoundaryMessage(message_type=kind, channel="PROGRAM", direction="G2T", fields=dict(fields))


def identity(image: ProgramImage, epoch: int, **extra: Any) -> dict[str, Any]:
  return {
    "program_id": image.program_id,
    "version": image.version,
    "package_hash": image.package_hash,
    "program_epoch": epoch,
    **extra,
  }


def _plain(fields: Mapping[str, Any]) -> dict[str, Any]:
  return {
    name: value.hex() if isinstance(value, (bytes, memoryview)) else value for name, value in fields.items()
  }


@dataclass(frozen=True)
class Tail:
  """The bench's fixed observation tail: Tile state after the last edge of a command."""

  state: int
  resident: int
  ready: int
  slot_valid: int
  fault_latched: bool
  fault_code: int
  fault_seen: bool
  t2g_valid: bool
  fetch_allow: bool
  read_req_count: int
  resp_bytes: int
  ins_beats: int
  install_count: int
  warm_hit_count: int
  inv_count: int
  fetch_accepted: int
  rsp_delivered: int
  install_beats: int
  port_busy: int
  bound_hits: int
  invariant_error: bool
  t2g_seq: int
  checks: int
  ticks_total: int


@dataclass(frozen=True)
class Reply:
  status: int
  ticks: int
  g2t_accepted: int
  g2t_pending: bool
  events: tuple[tuple[int, tuple[int, ...]], ...]
  tail: Tail


def parse_reply(words: Sequence[int]) -> Reply:
  if len(words) < 5 + TAIL_WORDS:
    raise ProtocolError(f"truncated bench reply ({len(words)} words)")
  status, ticks, count, accepted, pending = words[:5]
  cursor = 5
  events: list[tuple[int, tuple[int, ...]]] = []
  for _ in range(count):
    if cursor + 2 > len(words):
      raise ProtocolError("truncated bench event header")
    kind, length = words[cursor], words[cursor + 1]
    cursor += 2
    events.append((kind, tuple(words[cursor : cursor + length])))
    cursor += length
  tail = words[cursor:]
  if len(tail) != TAIL_WORDS:
    raise ProtocolError(f"bench reply tail has {len(tail)} words, not {TAIL_WORDS}")
  return Reply(
    status=status,
    ticks=ticks,
    g2t_accepted=accepted,
    g2t_pending=bool(pending),
    events=tuple(events),
    tail=Tail(
      state=tail[0],
      resident=tail[1],
      ready=tail[2],
      slot_valid=tail[3],
      fault_latched=bool(tail[4]),
      fault_code=tail[5],
      fault_seen=bool(tail[6]),
      t2g_valid=bool(tail[7]),
      fetch_allow=bool(tail[8]),
      read_req_count=tail[9],
      resp_bytes=tail[10],
      ins_beats=tail[11],
      install_count=tail[12],
      warm_hit_count=tail[13],
      inv_count=tail[14],
      fetch_accepted=tail[15],
      rsp_delivered=tail[16],
      install_beats=tail[17],
      port_busy=tail[18],
      bound_hits=tail[19],
      invariant_error=bool(tail[20]),
      t2g_seq=tail[21],
      checks=tail[22],
      ticks_total=tail[23],
    ),
  )


class RtlEndpoint:
  """One FIFO pair to one compiled bench. Each call is one command and its one reply.

  The host holds both ends open (`O_RDWR | O_NONBLOCK`): a bench that dies never yields EOF
  and the bench's own open() never blocks, so every read is bounded by a deadline and by the
  harness's own `{label}.rc` files instead of hanging.
  """

  def __init__(self, directory: Path, label: str, *, stall_timeout_s: float = 1200.0) -> None:
    self.directory = directory
    self.label = label
    self.stall_timeout_s = stall_timeout_s
    self.commands = 0
    self._buffer = b""
    self._cmd_fd: int | None = None
    self._evt_fd: int | None = None
    self._lock = threading.Lock()
    self.stopped = threading.Event()

  def open(self) -> None:
    os.mkfifo(self.directory / f"{self.label}.cmd")  # Python -> bench
    os.mkfifo(self.directory / f"{self.label}.evt")  # bench -> Python
    self._cmd_fd = os.open(self.directory / f"{self.label}.cmd", os.O_RDWR | os.O_NONBLOCK)
    self._evt_fd = os.open(self.directory / f"{self.label}.evt", os.O_RDWR | os.O_NONBLOCK)

  def close(self) -> None:
    for name in ("_cmd_fd", "_evt_fd"):
      fd = getattr(self, name)
      if fd is not None:
        os.close(fd)
        setattr(self, name, None)

  @property
  def exit_state(self) -> str | None:
    build = self.directory / f"{self.label}.build.rc"
    run = self.directory / f"{self.label}.rc"
    if build.is_file() and build.read_text().strip() != "0":
      return f"build failed rc={build.read_text().strip()}"
    if run.is_file():
      return f"bench exited rc={run.read_text().strip()}"
    return None

  def _send(self, words: Sequence[int]) -> None:
    assert self._cmd_fd is not None
    line = ("".join(f"{word & MASK64:x} " for word in words) + "\n").encode()
    if os.write(self._cmd_fd, line) != len(line):  # below PIPE_BYTES: never torn
      raise EndpointGone(f"{self.label}: command write was partial")

  def call(self, words: Sequence[int]) -> list[int]:
    assert self._evt_fd is not None
    with self._lock:
      self.commands += 1
      self._send(words)
      raw = self._readline()
    return [int(token, 16) for token in raw.split()]

  def send_only(self, words: Sequence[int]) -> None:
    """A command whose reply is never coming (the bench terminates on it)."""
    self._send(words)

  def _readline(self) -> bytes:
    assert self._evt_fd is not None
    deadline = time.monotonic() + self.stall_timeout_s
    while True:
      if self.stopped.is_set():
        raise EndpointGone(f"{self.label}: container has exited")
      index = self._buffer.find(b"\n")
      if index >= 0:
        line, self._buffer = self._buffer[:index], self._buffer[index + 1 :]
        return line
      readable, _, _ = select.select([self._evt_fd], [], [], 0.5)
      if readable:
        chunk = os.read(self._evt_fd, 1 << 16)
        if chunk:
          self._buffer += chunk
        continue
      gone = self.exit_state
      if gone is not None and self._buffer == b"":
        raise EndpointGone(f"{self.label}: {gone} before replying")
      if time.monotonic() > deadline:
        raise EndpointGone(
          f"{self.label}: no reply within {self.stall_timeout_s:.0f} s ({gone or 'bench alive'})"
        )


class TileProgramRtl:
  """Typed commands for one Tile PROGRAM endpoint. Each call advances a bounded number of ticks."""

  def __init__(self, endpoint: RtlEndpoint, contexts: int) -> None:
    self.endpoint = endpoint
    self.contexts = contexts
    self.ticks = 0
    self.tail = Tail(
      state=0,
      resident=0,
      ready=0,
      slot_valid=0,
      fault_latched=False,
      fault_code=0,
      fault_seen=False,
      t2g_valid=False,
      fetch_allow=False,
      read_req_count=0,
      resp_bytes=0,
      ins_beats=0,
      install_count=0,
      warm_hit_count=0,
      inv_count=0,
      fetch_accepted=0,
      rsp_delivered=0,
      install_beats=0,
      port_busy=0,
      bound_hits=0,
      invariant_error=False,
      t2g_seq=0,
      checks=0,
      ticks_total=0,
    )

  def _take(self, words: Sequence[int]) -> Reply:
    reply = parse_reply(words)
    self.ticks += reply.ticks
    self.tail = reply.tail
    return reply

  def deliver(self, words: Sequence[int]) -> Reply:
    if not 0 < len(words) <= 64:
      raise ProtocolError(f"deliver carries 1..64 words, not {len(words)}")
    return self._take(self.endpoint.call([OP_DELIVER, len(words), *words]))

  def wait(self, max_ticks: int = MAX_TICKS) -> Reply:
    return self._take(self.endpoint.call([OP_WAIT, min(max_ticks, MAX_TICKS)]))

  def status(self) -> Reply:
    return self._take(self.endpoint.call([OP_STATUS]))

  def reset(self) -> Reply:
    return self._take(self.endpoint.call([OP_RESET]))

  def hold_t2g(self, held: bool) -> Reply:
    return self._take(self.endpoint.call([OP_T2G_READY, 0 if held else 1]))

  def hold_fetch(self, held: bool) -> Reply:
    return self._take(self.endpoint.call([OP_FETCH_READY, 0 if held else 1]))

  def probe_arm(self, ctx: int, gen: int, epoch: int, reqid: int, addr: int, length: int) -> Reply:
    return self._take(self.endpoint.call([OP_PROBE, ctx, gen, epoch, reqid, addr, length]))

  def exec_ref(self, inc: bool, prog_id: int, epoch: int) -> Reply:
    return self._take(self.endpoint.call([OP_REF, 1 if inc else 0, prog_id, epoch]))

  def setup(
    self, image: ProgramImage, *, epoch: int, slot: int, text_base: int, text_len: int, text_sha256: bytes
  ) -> Reply:
    digest = int.from_bytes(text_sha256, "little")
    package = int.from_bytes(image.package_hash, "little")
    words = [
      image.program_id,
      image.version,
      *(package >> (64 * i) & MASK64 for i in range(4)),
      epoch,
      slot,
      text_base,
      text_len,
      *(digest >> (64 * i) & MASK64 for i in range(4)),
    ]
    return self._take(self.endpoint.call([OP_SETUP, *words]))

  def verdict(self, code: int) -> None:
    self.endpoint.send_only([OP_VERDICT, code])

  def quit(self) -> None:
    self.endpoint.send_only([OP_QUIT])


@dataclass
class Probe:
  """One fetch probe on the real C030 fetch port (CtxFetchKey: ctx, generation, epoch, reqid)."""

  ctx: int
  addr: int
  key: int
  gate: int | None = None
  fault: int | None = None
  code: int | None = None
  data: bytes | None = None


@dataclass
class GroupProgramLink:
  """Group half of one Tile's PROGRAM link around the live residency manager."""

  group: Any
  manager: Any
  rtl: TileProgramRtl
  images: dict[str, ProgramImage]
  idents: dict[str, ProgramIdentity]
  texts: dict[str, bytes]
  bases: dict[str, int]
  sources: dict[str, int]
  tile_id: int = 0
  g2t_seq: int = 0
  probe_id: int = 1
  install: tuple[str, BoundaryMessage] | None = None
  probe: Probe | None = None
  raw: dict[str, Callable[[BoundaryMessage, int], None]] = field(default_factory=dict)
  corrupt_next: bool = False
  wrong_handle_next: bool = False
  baseline: dict[str, int] = field(default_factory=dict)
  epoch_base: dict[str, int] = field(default_factory=dict)
  log: list[dict[str, Any]] = field(default_factory=list)
  faults: list[str] = field(default_factory=list)
  rejections: list[str] = field(default_factory=list)

  def __post_init__(self) -> None:
    self.resync()

  def resync(self) -> None:
    """A new link session (Tile reset): fresh sequencers, fresh receive models, fresh ledgers."""
    self.g2t_tx = wire.Rx()  # sealed receive model for the stream this driver emits
    self.t2g_rx = wire.Rx()  # sealed receive model for the stream the Tile emits
    self.g2t_seq = 0
    self.probe_id = 1
    self.install = None
    self.probe = None
    self.raw = {}
    self.corrupt_next = False
    self.wrong_handle_next = False
    self.baseline = dict(self.manager.tile_counters(self.tile_id))
    self.faults = []
    self.rejections = []

  def begin_sequence(self) -> None:
    """A new scenario sequence: a new link session and a new origin for the relative epochs."""
    self.epoch_base = {}
    self.log = []
    self.resync()

  @property
  def contexts(self) -> int:
    return self.rtl.contexts

  # ---- Group -> Tile ---------------------------------------------------------------
  def send(
    self, msg: BoundaryMessage, data: bytes = b"", *, raw: bool = False
  ) -> list[tuple[int, tuple[int, ...]]]:
    """Deliver one sealed G2T frame and its follow-on blob words; returns the boundary events."""
    assert msg.direction == "G2T" and msg.channel == "PROGRAM"
    seq = 0 if raw else self.g2t_seq
    frame = BoundaryMessage(
      message_type=msg.message_type,
      channel="PROGRAM",
      direction="G2T",
      fields=dict(msg.fields),
      message_seq=seq,
    )
    blob = frame.pack_wire()
    code, _payload = (wire.Rx() if raw else self.g2t_tx).deliver(PROGRAM_G2T, blob)
    if code != "OK":
      raise ProtocolError(f"the driver emitted a G2T frame the sealed receive model rejects: {code}")
    words = [*frame_words(blob), *blob_words(data)]
    events: list[tuple[int, tuple[int, ...]]] = []
    accepted = 0
    while accepted < len(words):
      reply = self.rtl.deliver(words[accepted:])
      accepted += reply.g2t_accepted
      events.extend(reply.events)
      if reply.status == ST_NOT_ACCEPTED:
        raise BoundedWait(
          f"{msg.message_type}: the Tile stopped accepting G2T words after {reply.ticks} ticks"
        )
      if reply.g2t_pending:
        # A boundary event interrupted the delivery; the held word stays valid until accepted.
        drained = self.rtl.wait()
        events.extend(drained.events)
        if drained.g2t_pending:
          raise BoundedWait(f"{msg.message_type}: held G2T word was never accepted")
    if not raw:
      self.g2t_seq += 1
    self.log.append({"g2t": msg.message_type, "seq": seq, "bytes": len(data), "raw": raw})
    return events

  # ---- Tile -> Group ---------------------------------------------------------------
  def pump(self, done: Callable[[], bool], what: str) -> int:
    """Tick until `done()`, servicing every boundary event in order. Bounded, never polling."""
    ticks = 0
    while not done():
      reply = self.rtl.wait()
      ticks += reply.ticks
      if reply.status == ST_NO_EVENT:
        raise BoundedWait(f"no {what} within {ticks} RTL ticks")
      self.on_events(reply.events)
    return ticks

  def idle(self, ticks: int) -> None:
    """Exactly `ticks` quiet ticks: any boundary event in them is a scenario failure."""
    reply = self.rtl.wait(ticks)
    if reply.status != ST_NO_EVENT:
      kinds = [MSG_NAMES.get(event[1][0], event[0]) for event in reply.events]
      raise ExpectationFailed("idle", "no boundary event", {"ticks": reply.ticks, "events": kinds})
    self.on_events(reply.events)

  def on_events(self, events: Sequence[tuple[int, tuple[int, ...]]]) -> None:
    for kind, words in events:
      if kind == EV_FRAME:
        self._frame(words)
      elif kind == EV_FAULT:
        name = FAULT_CODES[words[0]] if words[0] < len(FAULT_CODES) else f"FC_{words[0]}"
        self.faults.append(name)
        self.log.append({"fault": name})
        if self.install is not None:
          slot, _msg = self.install
          self.install = None
          try:
            self.manager.fail_install(self.idents[slot], self.tile_id, code=name)
          except ResidencyError as exc:
            raise ProtocolError(f"Tile fault during the install of {slot}: {exc.code}") from exc
      elif kind == EV_FETCH_ACCEPT:
        if self.probe is None or words[0] != self.probe.ctx:
          raise ProtocolError("fetch probe accepted on a foreign Context")
        self.probe.gate = words[1]
      elif kind == EV_FETCH_RESPONSE:
        if self.probe is None:
          raise ProtocolError("fetch response without a probe")
        key = words[3] | (words[4] << 64)
        if words[0] != self.probe.ctx or key != self.probe.key:
          raise ProtocolError("fetch response does not echo its request key")
        self.probe.fault, self.probe.code = words[1], words[2]
        self.probe.data = struct.pack("<4Q", *words[5:9])

  def _frame(self, words: tuple[int, ...]) -> None:
    msg_id, seq, *rest = words
    handle, frame = rest[-1], rest[:-1]
    blob = struct.pack(f"<{len(frame)}Q", *frame)
    code, _payload = self.t2g_rx.deliver(PROGRAM_T2G, blob)
    if code != "OK":
      raise ProtocolError(f"the sealed receive model rejects a Tile T2G frame: {code}")
    msg = BoundaryMessage.from_wire(blob, expected_port=PROGRAM_T2G)
    if msg.message_seq != seq or MSG_NAMES.get(msg_id) != msg.message_type:
      raise ProtocolError("bench frame sequence disagrees with the sealed header")
    self.log.append({"t2g": msg.message_type, "seq": seq, "fields": _plain(msg.fields)})
    waiter = self.raw.pop(msg.message_type, None)
    if waiter is not None:
      waiter(msg, handle)
      return
    try:
      if msg.message_type == "PROGRAM_INSTALL_ACK":
        step = self.manager.handle(self.tile_id, msg, local_handle=handle)
        self.install = None  # ACK-verified from now on: only INVALIDATE or a reset can retire it
      else:
        step = self.manager.handle(self.tile_id, msg)
    except ResidencyError as exc:
      self.rejections.append(exc.code)
      raise ProtocolError(f"the residency manager rejected a Tile {msg.message_type}: {exc}") from exc
    for index, out in enumerate(step.outbox):
      data = step.data if index == 0 and out.message_type == "PROGRAM_READ_RESP" else b""
      if self.corrupt_next and data:
        self.corrupt_next = False
        data = bytes([data[0] ^ 0xFF]) + data[1:]  # one byte flipped in transit
        self.log.append({"injection": "corrupt_chunk"})
      elif self.wrong_handle_next and out.message_type == "PROGRAM_READY":
        self.wrong_handle_next = False
        out = g2t_message(
          "PROGRAM_READY", {**out.fields, "local_handle": int(out.fields["local_handle"]) + 1}
        )
        self.log.append({"injection": "wrong_handle"})
      self.on_events(self.send(out, data))

  # ---- observation -----------------------------------------------------------------
  def view(self, slot: str) -> Any:
    return self.manager.view(self.idents[slot], self.tile_id)

  def state(self, slot: str) -> str:
    view = self.view(slot)
    return view.state.name if view is not None else "ABSENT"

  def observe(self) -> dict[str, Any]:
    """Live Tile state (bench tail) beside the live Group ledgers, per sequence since `resync`."""
    tail = self.rtl.status().tail
    tile = self.manager.tile_counters(self.tile_id)
    views = {slot: self.view(slot) for slot in self.images}
    live = {slot: view for slot, view in views.items() if view is not None}
    for slot, view in live.items():
      self.epoch_base.setdefault(slot, view.epoch)
    installed = self.manager.snapshot()["installed"]
    return {
      "states": {slot: view.state.name if view is not None else "ABSENT" for slot, view in views.items()},
      "epochs": {slot: view.epoch - self.epoch_base[slot] for slot, view in live.items()},
      "refs": {slot: view.refs for slot, view in live.items()},
      "bitmap": tail.resident,
      "manager_bitmap": self.manager.resident_bitmap(self.tile_id),
      "ready_bitmap": tail.ready,
      "read_requests": tail.read_req_count,
      "manager_read_requests": tile.get("read_requests", 0) - self.baseline.get("read_requests", 0),
      "external_bytes": tail.resp_bytes,
      "manager_external_bytes": tile.get("external_text_bytes", 0)
      - self.baseline.get("external_text_bytes", 0),
      "warm_hits": tail.warm_hit_count,
      "counters": self.manager.counters(),
      "faults": list(self.faults),
      "rejections": list(self.rejections),
      "t2g_valid": int(tail.t2g_valid),
      "tile_state": tail.state,
      "invariant": not tail.invariant_error,
      "manager_fault": None if installed["fault"] is None else installed["fault"]["code"],
    }

  def check(self, expect: Mapping[str, Any], *, last: Mapping[str, Any] | None) -> None:
    """Evaluate one `check:` mapping against the live Tile and Group."""
    obs = self.observe()
    for key, want in expect.items():
      if key in ("states", "epochs", "refs", "counters"):
        got = {name: obs[key].get(name) for name in want}
      elif key in (
        "tile_state",
        "ready_bitmap",
        "warm_hits",
        "faults",
        "rejections",
        "t2g_valid",
        "invariant",
        "manager_fault",
      ):
        got = obs[key]
      elif key == "bitmap":
        got = obs[key]
        if obs["manager_bitmap"] != want:
          raise ExpectationFailed("bitmap", want, {"tile": got, "manager": obs["manager_bitmap"]})
      elif key in ("read_requests", "external_bytes"):
        got = obs[key]
        if obs[f"manager_{key}"] != want:
          raise ExpectationFailed(key, want, {"tile": got, "manager": obs[f"manager_{key}"]})
      elif key == "lockstep":
        self.manager.assert_lockstep(self.tile_id)
        continue
      elif key == "l2_closed":
        try:
          self.group.assert_l2_closed()
        except Exception as exc:
          raise ExpectationFailed("l2_closed", "no live L2 object", str(exc)) from exc
        residual = self.group.unclosed_l2_objects()
        if residual:
          raise ExpectationFailed("l2_closed", [], residual)
        continue
      elif key == "hbm_intact":
        for name, source in self.sources.items():
          live = self.group.hbm.byte_store.read_hbm(source, len(self.texts[name]))
          if live != self.texts[name]:
            raise ExpectationFailed("hbm_intact", f"{name} source bytes unchanged", "mutated")
        continue
      elif key == "last":
        if last is None:
          raise ExpectationFailed("last", dict(want), "no acquire in this sequence")
        for name, value in want.items():
          if last.get(name) != value:
            raise ExpectationFailed(f"last.{name}", value, last.get(name))
        continue
      else:
        raise ProtocolError(f"unknown expectation {key!r}")
      if got != want:
        raise ExpectationFailed(key, want, got)


# ---- scenario vocabulary -----------------------------------------------------------------
PROBE_GEN, PROBE_EPOCH = 1, 0  # CtxFetchKey fields of every probe: fixed generation, fetch epoch 0
VERBS = frozenset(
  {
    "acquire",
    "settle",
    "release",
    "exec_ref",
    "hold",
    "fetch_hold",
    "idle",
    "probe",
    "probe_settle",
    "invalidate",
    "invalidate_tile",
    "group_reset",
    "patch",
    "inject",
    "await",
    "fixture",
    "check",
  }
)
OPTIONS = frozenset({"holder", "expect", "in_flight", "reason"})
EXPECT_KEYS = frozenset(
  {
    "states",
    "epochs",
    "refs",
    "counters",
    "tile_state",
    "bitmap",
    "ready_bitmap",
    "read_requests",
    "external_bytes",
    "warm_hits",
    "faults",
    "rejections",
    "t2g_valid",
    "invariant",
    "manager_fault",
    "lockstep",
    "l2_closed",
    "hbm_intact",
    "last",
  }
)
INJECTIONS = frozenset(
  {
    "corrupt_chunk",
    "wrong_handle",
    "rewrite",
    "warm_reinstall",
    "stale_epoch",
    "no_setup",
    "unsolicited_resp",
    "oversize",
    "invalidate_with_ref",
  }
)
OBSERVED = (
  "states",
  "epochs",
  "refs",
  "bitmap",
  "ready_bitmap",
  "read_requests",
  "external_bytes",
  "warm_hits",
  "faults",
  "rejections",
  "t2g_valid",
  "tile_state",
  "invariant",
)


def _parse_step(step: Mapping[str, Any], where: str) -> tuple[str, Any, dict[str, Any]]:
  """Exactly one verb plus known options; anything else is a scenario error before any tick."""
  if not isinstance(step, Mapping):
    raise ProtocolError(f"{where}: a step is a mapping with one verb")
  verbs = [name for name in step if name in VERBS]
  unknown = sorted(set(step) - VERBS - OPTIONS)
  if len(verbs) != 1 or unknown:
    raise ProtocolError(f"{where}: expected exactly one verb of {sorted(VERBS)}, got {dict(step)!r}")
  return verbs[0], step[verbs[0]], {name: step[name] for name in step if name in OPTIONS}


def _slot_holder(args: Any) -> tuple[str, Any]:
  return (args, None) if isinstance(args, str) else (args[0], args[1])


def validate_scenario(scenario: Mapping[str, Any], where: str) -> None:
  """Fail closed on the scenario itself: every step and expectation is one this driver knows."""
  if scenario.get("schema") != "cn_program_scenario/v1" or set(scenario["slots"]) != {"A", "B"}:
    raise ProtocolError(f"{where}: expected a two-slot cn_program_scenario/v1")
  steps = scenario.get("steps")
  if not isinstance(steps, Mapping) or not steps:
    raise ProtocolError(f"{where}: `steps:` must name at least one sequence")
  fixtures = scenario.get("fixtures", {})
  if not isinstance(fixtures, Mapping) or not set(fixtures) <= {"C", "oversize"}:
    raise ProtocolError(
      f"{where}: unknown fixture {sorted(fixtures) if isinstance(fixtures, Mapping) else fixtures!r}"
    )
  for name, sequence in steps.items():
    if not isinstance(sequence, list) or not sequence:
      raise ProtocolError(f"{where}: sequence {name!r} is a non-empty list of steps")
    for index, step in enumerate(sequence):
      at = f"{where}: {name}[{index}]"
      verb, args, options = _parse_step(step, at)
      if verb == "check":
        bad = sorted(set(args) - EXPECT_KEYS)
        if bad:
          raise ProtocolError(f"{at}: unknown expectation keys {bad}")
      if verb == "inject":
        names = [args] if isinstance(args, str) else [args[0]] if isinstance(args, list) else []
        if len(names) != 1 or names[0] not in INJECTIONS:
          raise ProtocolError(f"{at}: unknown injection {args!r}")
      if verb in ("probe", "probe_settle", "patch") and not isinstance(args, Mapping):
        raise ProtocolError(f"{at}: {verb} takes a mapping")
      if verb == "fixture" and args not in fixtures:
        raise ProtocolError(f"{at}: fixture {args!r} is not declared under `fixtures:`")
      if verb == "acquire" and options.get("expect") not in (None, "pending", "hit", "wait", "refused"):
        raise ProtocolError(f"{at}: unknown acquire outcome {options.get('expect')!r}")


class SequenceRunner:
  """Executes the `steps:` of one scenario sequence against one live Tile link."""

  def __init__(self, link: GroupProgramLink, group: Any, fixtures: Mapping[str, Any]) -> None:
    self.link = link
    self.group = group
    self.fixtures = fixtures
    self.owed: dict[str, int] = {}  # Tile execution references a pending install still owes
    self.warm_handle = 0

  @property
  def manager(self) -> Any:
    return self.link.manager

  def run(self, steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    last: dict[str, Any] | None = None
    for index, step in enumerate(steps):
      verb, args, options = _parse_step(step, f"step {index}")
      before = self.link.rtl.ticks
      record, last = getattr(self, f"_step_{verb}")(args, options, last)
      record.update({"step": index, "verb": verb, "ticks": self.link.rtl.ticks - before})
      observed = self.link.observe()
      record["observed"] = {key: observed[key] for key in OBSERVED}
      records.append(record)
    return records

  # ---- helpers -----------------------------------------------------------------------
  def _trusted_setup(
    self, image: ProgramImage, *, epoch: int, base: int, text: bytes, text_len: int | None = None
  ) -> None:
    """Write the Tile's trusted setup sideband from the verified sealed image."""
    reply = self.link.rtl.setup(
      image,
      epoch=epoch,
      slot=base,
      text_base=base,
      text_len=len(text) if text_len is None else text_len,
      text_sha256=hashlib.sha256(text).digest(),
    )
    self.link.on_events(reply.events)
    if not any(kind == EV_SETUP for kind, _ in reply.events):
      raise ProtocolError(
        f"trusted setup for program {image.program_id} was not accepted (status {reply.status})"
      )

  def _start(self, slot: str, outbox: Sequence[BoundaryMessage]) -> None:
    install = outbox[0]
    assert install.message_type == "PROGRAM_INSTALL"
    self.link.install = (slot, install)
    # The trusted setup is keyed by the full identity (epoch included) and must precede the header.
    self._trusted_setup(
      self.link.images[slot],
      epoch=int(install.fields["program_epoch"]),
      base=self.link.bases[slot],
      text=self.link.texts[slot],
    )
    for out in outbox:
      self.link.send(out)

  def _settle(self, slot: str) -> bool:
    """Run the in-flight install to its completion point; True when the slot became RESIDENT."""
    self.link.pump(
      lambda: self.link.state(slot) == "RESIDENT" or self.link.install is None, f"install of {slot}"
    )
    self.link.install = None
    owed, self.owed[slot] = self.owed.get(slot, 0), 0
    if self.link.state(slot) != "RESIDENT":
      return False  # the waiters' references died with the aborted install
    self._tile_ref(slot, True, owed)  # each launcher takes its Tile execution reference
    return True

  def _tile_ref(self, slot: str, inc: bool, count: int = 1) -> None:
    view = self.link.view(slot)
    if view is None:
      raise ProtocolError(f"execution reference for {slot} without a residency record")
    for _ in range(count):
      self.link.on_events(self.link.rtl.exec_ref(inc, self.link.images[slot].program_id, view.epoch).events)

  def _await_fault(self, name: str) -> None:
    count = len(self.link.faults)
    self.link.pump(lambda: len(self.link.faults) > count, f"Tile fault {name}")
    if self.link.faults[count:] != [name]:
      raise ExpectationFailed("faults", [name], self.link.faults[count:])

  def _quiet(self, before: Any, what: str) -> None:
    after = self.link.rtl.status().tail
    if (after.read_req_count, after.resident, after.install_count) != (
      before.read_req_count,
      before.resident,
      before.install_count,
    ):
      raise ExpectationFailed(
        what,
        "no read and no residency change",
        {
          "read_requests": [before.read_req_count, after.read_req_count],
          "resident": [before.resident, after.resident],
        },
      )

  # ---- steps -------------------------------------------------------------------------
  def _step_acquire(self, args: Any, options: Mapping[str, Any], last: Any) -> tuple[dict[str, Any], Any]:
    slot, holder = _slot_holder(args)
    holder = options.get("holder", holder)
    before = self.link.rtl.status().tail
    decision = self.manager.acquire(self.link.idents[slot], self.link.tile_id, holder=holder)
    record = {
      "slot": slot,
      "holder": holder,
      "kind": decision.kind,
      "reason": decision.reason,
      "outbox": len(decision.outbox),
      "read_requests": 0,
      "external_bytes": 0,
      "faulted": False,
    }
    if "expect" in options and decision.kind != options["expect"]:
      raise ExpectationFailed("acquire.kind", options["expect"], record)
    if "reason" in options and decision.reason != options["reason"]:
      raise ExpectationFailed("acquire.reason", options["reason"], decision.reason)
    if decision.kind == "pending":
      self.owed[slot] = self.owed.get(slot, 0) + 1
      if decision.outbox:
        self._start(slot, decision.outbox)
      if not options.get("in_flight"):
        if not self._settle(slot):
          record["faulted"] = True
      after = self.link.rtl.status().tail
      record["read_requests"] = after.read_req_count - before.read_req_count
      record["external_bytes"] = after.resp_bytes - before.resp_bytes
    elif decision.kind == "hit":
      self._tile_ref(slot, True)
    return record, record

  def _step_settle(self, args: Any, _options: Mapping[str, Any], last: Any) -> tuple[dict[str, Any], Any]:
    slot, _holder = _slot_holder(args)
    if self.link.install is not None and self.link.install[0] == slot:
      if not self._settle(slot):
        raise ExpectationFailed("settle", f"{slot} RESIDENT", self.link.state(slot))
    elif self.link.state(slot) == "INVALIDATING":
      self.link.pump(lambda: self.link.state(slot) == "ABSENT", f"INVALIDATE_ACK of {slot}")
    else:
      raise ProtocolError(f"settle: nothing in flight for {slot}")
    return {"slot": slot, "state": self.link.state(slot)}, last

  def _step_release(self, args: Any, _options: Mapping[str, Any], last: Any) -> tuple[dict[str, Any], Any]:
    slot, holder = _slot_holder(args)
    holder = _options.get("holder", holder)
    step = self.manager.release(self.link.idents[slot], self.link.tile_id, holder=holder)
    self._tile_ref(slot, False)
    invalidated = False
    for out in step.outbox:
      invalidated |= out.message_type == "PROGRAM_INVALIDATE"
      self.link.send(out)
    if invalidated:
      self.link.pump(lambda: self.link.state(slot) == "ABSENT", f"INVALIDATE_ACK of {slot}")
    return {"slot": slot, "outbox": [out.message_type for out in step.outbox]}, last

  def _step_exec_ref(self, args: Any, _options: Mapping[str, Any], last: Any) -> tuple[dict[str, Any], Any]:
    kind, slot = args
    view = self.link.view(slot)
    self.link.on_events(
      self.link.rtl.exec_ref(
        kind == "inc", self.link.images[slot].program_id, view.epoch if view is not None else 0
      ).events
    )
    return {"slot": slot, "ref": kind}, last

  def _step_hold(self, args: Any, _options: Mapping[str, Any], last: Any) -> tuple[dict[str, Any], Any]:
    self.link.rtl.hold_t2g(bool(args))
    return {"t2g_ready": 0 if args else 1}, last

  def _step_fetch_hold(
    self, args: Any, _options: Mapping[str, Any], last: Any
  ) -> tuple[dict[str, Any], Any]:
    self.link.rtl.hold_fetch(bool(args))
    return {"fetch_ready": 0 if args else 1}, last

  def _step_idle(self, args: Any, _options: Mapping[str, Any], last: Any) -> tuple[dict[str, Any], Any]:
    self.link.idle(int(args))
    return {"quiet_ticks": int(args)}, last

  def _step_probe(
    self, args: Mapping[str, Any], _options: Mapping[str, Any], last: Any
  ) -> tuple[dict[str, Any], Any]:
    slot, offset, ctx = args["slot"], int(args.get("offset", 0)), int(args.get("ctx", 0))
    if ctx >= self.link.contexts or offset % LINE_BYTES:
      raise ProtocolError("probe needs a line-aligned offset on an existing Context")
    reqid = self.link.probe_id
    self.link.probe_id += 1
    addr = self.link.bases[slot] + offset
    self.link.probe = Probe(
      ctx=ctx, addr=addr, key=(ctx << 80) | (PROBE_GEN << 64) | (PROBE_EPOCH << 32) | reqid
    )
    self.link.on_events(
      self.link.rtl.probe_arm(ctx, PROBE_GEN, PROBE_EPOCH, reqid, addr, LINE_BYTES).events
    )
    if args.get("arm_only"):
      probe = self.link.probe
      self.link.pump(lambda: probe.gate is not None, "held fetch accepted")
      return {"slot": slot, "offset": offset, "armed": True}, last
    return self._probe_result(slot, offset, args), last

  def _step_probe_settle(
    self, args: Mapping[str, Any], _options: Mapping[str, Any], last: Any
  ) -> tuple[dict[str, Any], Any]:
    return self._probe_result(args["slot"], int(args.get("offset", 0)), args), last

  def _probe_result(self, slot: str, offset: int, args: Mapping[str, Any]) -> dict[str, Any]:
    probe = self.link.probe
    assert probe is not None and probe.addr == self.link.bases[slot] + offset
    self.link.pump(lambda: probe.gate is not None, "fetch probe accept")
    self.link.pump(lambda: probe.data is not None, "fetch probe response")
    text = self.link.texts[slot]
    padded = text + bytes((-len(text)) % LINE_BYTES)
    unwritten = bool(args.get("unwritten", False))
    got = {
      "fault": probe.fault,
      "code": FETCH_CODES[probe.code] if probe.code is not None else None,
      "gate": probe.gate,
    }
    want = (
      {"fault": 1, "code": "F_UNWRITTEN", "gate": 0}
      if unwritten
      else {"fault": 0, "code": "F_OK", "gate": int(args.get("gate", 1))}
    )
    if got != want:
      raise ExpectationFailed(f"probe {slot}+{offset:#x}", want, got)
    assert probe.data is not None
    if not unwritten and probe.data != padded[offset : offset + LINE_BYTES]:
      raise ExpectationFailed(
        f"probe {slot}+{offset:#x} bytes", padded[offset : offset + LINE_BYTES].hex(), probe.data.hex()
      )
    self.link.probe = None
    return {"slot": slot, "offset": offset, **got, "bytes": probe.data.hex()}

  def _step_invalidate(
    self, args: Any, options: Mapping[str, Any], last: Any
  ) -> tuple[dict[str, Any], Any]:
    slot = args
    step = self.manager.begin_invalidate(self.link.idents[slot], self.link.tile_id)
    for out in step.outbox:
      self.link.send(out)
    if step.outbox and not options.get("in_flight"):
      self.link.pump(lambda: self.link.state(slot) == "ABSENT", f"INVALIDATE_ACK of {slot}")
    return {
      "slot": slot,
      "outbox": [out.message_type for out in step.outbox],
      "state": self.link.state(slot),
    }, last

  def _step_invalidate_tile(
    self, args: Any, _options: Mapping[str, Any], last: Any
  ) -> tuple[dict[str, Any], Any]:
    tile = int(args)
    self.link.rtl.reset()  # a Tile reset voids its residency; the Group record follows
    self.manager.invalidate_tile(tile)
    self.link.resync()
    self.owed = {}
    self.manager.assert_lockstep(tile)
    return {"tile": tile}, last

  def _step_group_reset(
    self, args: Any, _options: Mapping[str, Any], last: Any
  ) -> tuple[dict[str, Any], Any]:
    if args is not True:
      raise ProtocolError("group_reset takes `true`")
    self.link.rtl.reset()
    self.group.reset()
    self.manager.reset()
    self.link.resync()
    self.owed = {}
    self.manager.assert_lockstep()
    return {"reset": True}, last

  def _step_patch(
    self, args: Mapping[str, Any], _options: Mapping[str, Any], last: Any
  ) -> tuple[dict[str, Any], Any]:
    slot = args["slot"]
    try:
      self.manager.patch(
        self.link.idents[slot], self.link.tile_id, field=args["field"], layer=args["layer"]
      )
      refused = None
    except ResidencyError as exc:
      refused = exc.code
      self.link.rejections.append(refused)
    return {"slot": slot, "refused": refused}, last

  def _step_await(self, args: Any, _options: Mapping[str, Any], last: Any) -> tuple[dict[str, Any], Any]:
    if args in FAULT_CODES:
      self.link.pump(lambda: args in self.link.faults, f"Tile fault {args}")
    elif args in MSG_NAMES.values():
      self.link.pump(lambda: args not in self.link.raw, f"raw {args}")
    else:
      raise ProtocolError(f"unknown await operand {args!r}")
    return {"awaited": args}, last

  def _step_check(
    self, args: Mapping[str, Any], _options: Mapping[str, Any], last: Any
  ) -> tuple[dict[str, Any], Any]:
    self.link.check(args, last=last)
    return {"held": sorted(args)}, last

  def _step_fixture(self, args: str, _options: Mapping[str, Any], last: Any) -> tuple[dict[str, Any], Any]:
    """Register a declared negative-only image (never a sealed package) under its fixture name."""
    spec = self.fixtures[args]
    refused = None
    try:
      if "text_from" in spec:
        base = self.link.images[spec["text_from"]]
        text, base_address = self.link.texts[spec["text_from"]], int(spec["text_base"])
        image = ProgramImage(
          program_id=int(spec["program_id"]),
          version=base.version,
          package_hash=hashlib.sha256(b"cn_program_scenario/v1 fixture:" + text).digest(),
          program_epoch=0,
          text=text,
        )
        ident = self.manager.register_image(image, text_base=base_address)
        self.link.idents[args], self.link.images[args] = ident, image
        self.link.texts[args], self.link.bases[args] = text, base_address
      else:
        raise ProtocolError(f"fixture {args!r} is Tile-side only")
    except ResidencyError as exc:
      refused = exc.code
      self.link.rejections.append(refused)
    return {"fixture": args, "refused": refused}, last

  # ---- fault injections: deliberately malformed or hostile Group traffic ---------------
  def _step_inject(self, args: Any, _options: Mapping[str, Any], last: Any) -> tuple[dict[str, Any], Any]:
    name, params = (args, {}) if isinstance(args, str) else (args[0], args[1])
    return getattr(self, f"_inject_{name}")(params), last

  def _inject_corrupt_chunk(self, _params: Mapping[str, Any]) -> dict[str, Any]:
    self.link.corrupt_next = True
    return {"inject": "corrupt_chunk"}

  def _inject_wrong_handle(self, _params: Mapping[str, Any]) -> dict[str, Any]:
    self.link.wrong_handle_next = True
    return {"inject": "wrong_handle"}

  def _inject_rewrite(self, params: Mapping[str, Any]) -> dict[str, Any]:
    """INSTALL of another identity into a slot that is resident: no auto-eviction (CMP-18)."""
    target, source = params["slot"], params["as"]
    base, image = self.link.bases[target], self.link.images[source]
    before = self.link.rtl.status().tail
    self._trusted_setup(image, epoch=0, base=base, text=self.link.texts[source])
    self.link.send(g2t_message("PROGRAM_INSTALL", identity(image, 0, slot=base)), raw=True)
    self._await_fault("INVALID_DESCRIPTOR")
    self._quiet(before, "rewrite")
    return {"inject": "rewrite", "slot": target, "as": source}

  def _inject_warm_reinstall(self, params: Mapping[str, Any]) -> dict[str, Any]:
    """The identical identity installed again while resident: same measurement and handle, zero reads."""
    slot, image, view = params["slot"], self.link.images[params["slot"]], self.link.view(params["slot"])
    text, base = self.link.texts[slot], self.link.bases[slot]
    before = self.link.rtl.status().tail

    def check(msg: BoundaryMessage, handle: int) -> None:
      got = {
        "identity": all(msg.fields[key] == val for key, val in identity(image, view.epoch).items()),
        "installed_len": int(msg.fields["installed_len"]) == len(text),
        "measured_hash": msg.fields["measured_hash"] == hashlib.sha256(text).digest(),
        "handle": handle == view.local_handle,
      }
      if not all(got.values()):
        raise ExpectationFailed("warm INSTALL_ACK", "stored measurement and handle", got)
      self.warm_handle = handle

    self.link.raw["PROGRAM_INSTALL_ACK"] = check
    self.link.send(g2t_message("PROGRAM_INSTALL", identity(image, view.epoch, slot=base)), raw=True)
    self.link.pump(lambda: "PROGRAM_INSTALL_ACK" not in self.link.raw, "warm INSTALL_ACK")
    after = self.link.rtl.status().tail
    if (after.read_req_count, after.warm_hit_count) != (before.read_req_count, before.warm_hit_count + 1):
      raise ExpectationFailed(
        "warm_reinstall",
        {"read_requests": before.read_req_count, "warm_hits": before.warm_hit_count + 1},
        {"read_requests": after.read_req_count, "warm_hits": after.warm_hit_count},
      )
    self.link.send(
      g2t_message("PROGRAM_READY", identity(image, view.epoch, local_handle=self.warm_handle)), raw=True
    )
    return {"inject": "warm_reinstall", "slot": slot}

  def _inject_stale_epoch(self, params: Mapping[str, Any]) -> dict[str, Any]:
    slot, image, base = params["slot"], self.link.images[params["slot"]], self.link.bases[params["slot"]]
    before = self.link.rtl.status().tail
    view = self.link.view(slot)
    epoch = int(params.get("epoch", (view.epoch if view is not None else 0) + 1))
    self.link.send(g2t_message("PROGRAM_INSTALL", identity(image, epoch, slot=base)), raw=True)
    self._await_fault("INVALID_DESCRIPTOR")  # no trusted setup exists for that identity
    self._quiet(before, "stale_epoch")
    return {"inject": "stale_epoch", "slot": slot, "epoch": epoch}

  def _inject_no_setup(self, params: Mapping[str, Any]) -> dict[str, Any]:
    slot, image, base = params["slot"], self.link.images[params["slot"]], self.link.bases[params["slot"]]
    before = self.link.rtl.status().tail
    self.link.send(g2t_message("PROGRAM_INSTALL", identity(image, 0, slot=base)), raw=True)
    self._await_fault("INVALID_DESCRIPTOR")
    self._quiet(before, "no_setup")
    return {"inject": "no_setup", "slot": slot}

  def _inject_unsolicited_resp(self, params: Mapping[str, Any]) -> dict[str, Any]:
    slot, image = params["slot"], self.link.images[params["slot"]]
    view = self.link.view(slot)
    epoch = view.epoch if view is not None else 0
    self.link.send(g2t_message("PROGRAM_READ_RESP", identity(image, epoch, hbm_chunk_len=8)), raw=True)
    self._await_fault("STREAM_PROTOCOL_ERROR")
    return {"inject": "unsolicited_resp", "slot": slot}

  def _inject_oversize(self, params: Mapping[str, Any]) -> dict[str, Any]:
    """A trusted setup whose text cannot fit a slot: the Tile refuses before pulling a byte."""
    spec = self.fixtures["oversize"]
    base, text = int(spec["text_base"]), bytes(int(spec["text_bytes"]))
    image = ProgramImage(int(spec["program_id"]), 1, hashlib.sha256(text).digest(), 0, text)
    before = self.link.rtl.status().tail
    self._trusted_setup(image, epoch=0, base=base, text=text)
    self.link.send(g2t_message("PROGRAM_INSTALL", identity(image, 0, slot=base)), raw=True)
    self._await_fault("L2_CAPACITY_FAULT")
    self._quiet(before, "oversize")
    return {"inject": "oversize", "text_bytes": len(text)}

  def _inject_invalidate_with_ref(self, params: Mapping[str, Any]) -> dict[str, Any]:
    """INVALIDATE behind the Group's back while the Tile still holds an execution reference."""
    slot, image, view = params["slot"], self.link.images[params["slot"]], self.link.view(params["slot"])
    want = identity(image, view.epoch)

    def check(msg: BoundaryMessage, _handle: int) -> None:
      if msg.fields != want:
        raise ExpectationFailed("raw INVALIDATE_ACK", _plain(want), _plain(msg.fields))

    self.link.raw["INVALIDATE_ACK"] = check
    self.link.send(g2t_message("PROGRAM_INVALIDATE", want), raw=True)
    return {"inject": "invalidate_with_ref", "slot": slot}


# ---- one label: a live Group per compiled bench --------------------------------------------
def contexts_of(label: str, default: int) -> int:
  found = re.search(r"-c(\d+)-", label)
  return int(found.group(1)) if found else default


def open_group(
  endpoint: RtlEndpoint,
  label: str,
  rtl: TileProgramRtl,
  *,
  contexts: int,
  scenario: Mapping[str, Any],
  scenario_path: Path,
  package_root: Path,
  config_path: Path,
  texts: Mapping[str, bytes],
  tile_id: int = 0,
) -> tuple[TileGroup, GroupProgramLink]:
  """Build the live Group for one label: its own HBM receives the uploads and its own
  `program_table` drives the link. The Group is really stepped through its profile-controller
  initialization first, so the explicit `TileGroup.reset()` recovery is a legal Group action."""
  group = TileGroup(HardwareConfig(), fidelity="runtime", context_count=contexts, byte_store=ByteStore())
  cycle = 0
  while not group.profile_controller.initialized:
    if cycle > 5000:
      raise ProtocolError(f"{label}: the Group's profile controller never initialized")
    group.step(cycle)
    cycle += 1
  prepared = prepare_programs(scenario_path, package_root, config_path=config_path, hbm=group.hbm)
  for slot in ("A", "B"):
    if prepared.packages[slot].text != texts[slot]:
      raise ProtocolError(f"{label}: the Loader text of {slot} differs from the ELF executable segment")
  manager = group.program_table
  idents = prepared.attach(manager, tile_id)
  link = GroupProgramLink(
    group=group,
    manager=manager,
    rtl=rtl,
    images=dict(prepared.images),
    idents=dict(idents),
    texts={slot: texts[slot] for slot in ("A", "B")},
    bases={
      slot: int(prepared.packages[slot].view["image"]["link_layout"]["text_base"]) for slot in ("A", "B")
    },
    sources={slot: int(scenario["slots"][slot]["hbm_iova"]) for slot in ("A", "B")},
    tile_id=tile_id,
  )
  return group, link


def run_label(
  endpoint: RtlEndpoint,
  label: str,
  *,
  contexts: int,
  scenario: Mapping[str, Any],
  scenario_path: Path,
  package_root: Path,
  config_path: Path,
  texts: Mapping[str, bytes],
) -> dict[str, Any]:
  """Drive each scenario sequence against its live Group and retain the observed verdict."""
  report: dict[str, Any] = {
    "label": label,
    "contexts": contexts,
    "sequences": {},
    "verdict": None,
    "error": None,
  }
  rtl = TileProgramRtl(endpoint, contexts)
  link = None
  try:
    group, link = open_group(
      endpoint,
      label,
      rtl,
      contexts=contexts,
      scenario=scenario,
      scenario_path=scenario_path,
      package_root=package_root,
      config_path=config_path,
      texts=texts,
    )
    runner = SequenceRunner(link, group, scenario.get("fixtures", {}))
    for name, steps in scenario["steps"].items():
      rtl.reset()  # the Tile forgets its residency and its trusted setup table
      group.reset()  # the live Group's explicit recovery (it invalidates residency itself)
      link.manager.reset()  # fresh ledgers for the sequence; images and uploads stay registered
      link.begin_sequence()
      runner.owed = {}
      try:
        report["sequences"][name] = {"ok": True, "steps": runner.run(steps)}
      except CosimError as exc:
        done = [item for item in report["sequences"].values() if item["ok"]]
        report["sequences"][name] = {"ok": False, "steps": None, "detail": str(exc)}
        report["verdict"] = {
          "sequence": name,
          "code": exc.verdict,
          "key": exc.key,
          "detail": str(exc),
          "completed": len(done),
          "ticks": rtl.ticks,
          "faults": list(link.faults),
          "rejections": list(link.rejections),
          "log_tail": link.log[-24:],
        }
        break
  except EndpointGone as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
  except Exception as exc:  # a driver or Group failure is reported, never hidden
    report["error"] = f"{type(exc).__name__}: {exc}"
  finally:
    report["ticks"] = rtl.ticks
    report["checks"] = rtl.tail.checks
    report["commands"] = endpoint.commands
    if link is not None:
      report["faults"], report["rejections"] = list(link.faults), list(link.rejections)
      report["group"] = {
        "l2_objects": link.group.unclosed_l2_objects(),
        "credits_hold": link.group.credit_invariants_hold(),
      }
    if report["verdict"] is not None or report["error"] is not None:
      try:
        rtl.verdict(report["verdict"]["code"] if report["verdict"] else V_PROTOCOL)
      except OSError:
        pass
    else:
      rtl.quit()
  return report


@contextmanager
def drive(
  evidence: Path,
  labels: Sequence[str],
  *,
  scenario: Mapping[str, Any],
  scenario_path: Path,
  package_root: Path,
  config_path: Path,
  texts: Mapping[str, bytes],
  default_contexts: int,
) -> Iterator[None]:
  """`Slice.service`: serve every bench of one run in the harness's own label order."""
  endpoints: dict[str, RtlEndpoint] = {}
  for label in labels:
    endpoints[label] = RtlEndpoint(evidence, label)
    endpoints[label].open()
  report: dict[str, Any] = {}
  failures: list[str] = []

  def worker() -> None:
    for label, endpoint in endpoints.items():
      try:
        report[label] = run_label(
          endpoint,
          label,
          contexts=contexts_of(label, default_contexts),
          scenario=scenario,
          scenario_path=scenario_path,
          package_root=package_root,
          config_path=config_path,
          texts=texts,
        )
      except Exception as exc:  # the driver itself crashed: say so and never leave a bench waiting
        failures.append(f"{label}: {type(exc).__name__}: {exc}")
        try:
          endpoint.send_only([OP_VERDICT, V_PROTOCOL])
        except OSError:
          pass

  thread = threading.Thread(target=worker, name="c039-cosim-driver", daemon=True)
  thread.start()
  try:
    yield
  finally:
    for endpoint in endpoints.values():
      endpoint.stopped.set()
    thread.join(timeout=120)
    for endpoint in endpoints.values():
      endpoint.close()
    (evidence / "residency-cosim.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    if thread.is_alive():
      raise RuntimeError("the C039 co-simulation driver did not stop")
    if failures:
      raise AssertionError("C039 co-simulation driver crashed:\n" + "\n".join(failures))
