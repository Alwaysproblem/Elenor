"""Live C034 boundary peer: real Group reset cycles plus the real C023 service.

The compiled `tb_c034` benches (greens and mutants) reach this peer over one
FIFO pair (`+PEER=/out/<label>`, files `<label>.req`/`<label>.rsp`), newline
ASCII hex; every reply is exactly three words `status a b` (status 0 = OK,
errors reply `ffffffff 0 0` and never hang). The peer is transport only. It
drives the actual pipeline_validator TileGroup/ResetDomain through the C034
production API and the actual C023 MemoryService through the C032
`c032_mfe_peer.Peer`; no software endpoint stub, and the reset sequence is
never enumerated here — every phase request comes from stepping the real
Group until it emits one.

Live Group commands (U3-f/U3-h/U3-i cycles):
  F gen tile ctxslot ctxgen code severity source engine op
     Fresh genuine Simulator/TileGroup, `attach_tile_reset(tile, gen)`, a
     sealed FAULT_REPORT DTO (canonical wire enum names, pack + unpack round
     trip) and the production `accept_tile_fault` -> `0 fault_record_idx 0`.
     Refused while a cycle is still open.
  N   Step the real Group (bounded 64 steps per the C034 contract) until
      `take_tile_reset_request` yields the next request ->
      `0 phase_index run_generation`; real DONE -> `0 ff generation`.
  A phase_idx gen marked mask
      Observe the fired RESET_PHASE_ACK (same sealed codec) through
      `observe_tile_reset_ack` -> `0 0 0`. The Group validates the
      outstanding phase/generation and the DRAIN bitmap / MARK count before
      consuming; a fabricated ACK fails there, not here.
  P gen tile   Production isolation-failure note for THIS run/tile (sticky
      Group poison) -> `0 1 0`.
  C            Actual `TileGroup.reset()` explicit recovery ->
      `0 allowed poisoned`; a mid-drain MemoryInvariantError refusal is
      reported, not forced. Repeatable: every attempt appends to an ordered
      receipt, so a refusal followed by a later success stays visible.
  Z            Require the real DONE observation, the acknowledged phase
      sequence and zero residue, then close the cycle -> `0 0 0`. Z never
      touches an open M..E memory window.
  G depth n    n real FaultRecords into a production FaultRing(slots=depth)
      -> `0 write_count latest_source`; the retained window (length, oldest
      source, last physical index) is checked and recorded.

Real C023 MemoryService commands (U3-e/U3-f windows, delegated verbatim to
the reused C032 peer):
  M            Create the memory peer, refusing to replace an unclosed one.
  R/W/T ...    Forwarded verbatim; the first two reply words are zero-padded
      to three (`0 data 0`, `0 visibility 0`, `0 0 0`). Only 8-byte
      transfers are accepted here.
  B offset     8 destination HBM bytes, little-endian, through the real
      ByteStore -> `0 value 0`.
  E            Assert the MemoryService ledger closed, save the window
      receipt -> `0 accepted_requests terminal_acks`.
"""

from __future__ import annotations

import json
import os
import selectors
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

W6 = Path(__file__).resolve().parent
W4 = W6.parent / "w4"
if str(W4) not in sys.path:
  sys.path.insert(0, str(W4))

from c032_mfe_peer import BASE, SIZE, Peer as MfePeer  # noqa: E402

from pipeline_validator.cosim import wire  # noqa: E402
from pipeline_validator.cosim.tile_port import BoundaryMessage  # noqa: E402
from pipeline_validator.memory.allocator import MemoryInvariantError  # noqa: E402
from pipeline_validator.runtime.fault_ring import FaultRecord, FaultRing  # noqa: E402
from pipeline_validator.tests.test_runtime import TestGridSignalAggregation  # noqa: E402

# C017 frozen wire orders; encoded values are these indices, nothing local.
WIRE_PHASES = tuple(wire.ENUMS["reset_phase"])
FAULT_CODES = tuple(wire.ENUMS["fault_code"])
DRAIN_EVIDENCE = tuple(wire.BITSETS["drain_evidence"])
DONE_PHASE = 0xFF  # N reply sentinel after the real Group reaches DONE
MAX_PHASE_STEPS = 64  # explicit N bound from the C034 contract; the genuine
                      # dispatched fixture reaches each mapped stage in a handful
                      # of steps, so exceeding this means a stuck Group: N must
                      # error, never hang or manufacture progression
TRANSFER_BYTES = 8  # this bench carries 8-byte MFE transfers only
MAX_RING_WRITES = 65536


class PeerError(RuntimeError):
  """Protocol error: replied as a nonzero status, never a hang."""


def _words(rest: list[str], count: int, kind: str) -> list[int]:
  if len(rest) != count:
    raise PeerError(f"{kind} expects {count} hex words, got {len(rest)}")
  try:
    return [int(word, 16) for word in rest]
  except ValueError as exc:
    raise PeerError(f"{kind} carries a non-hex word") from exc


def _reply(a: int, b: int, c: int) -> str:
  return f"{a:x} {b:x} {c:x}\n"


class _Cycle:
  """One live F..Z Group reset cycle owned by this peer."""

  def __init__(self, group: Any, tile: int, generation: int, fault_idx: int,
               domain: str | None) -> None:
    self.group = group
    self.tile = tile
    self.generation = generation
    self.fault_idx = fault_idx
    self.domain = domain
    self.requests: list[dict[str, Any]] = []
    self.acks: list[dict[str, Any]] = []
    self.poison: dict[str, Any] | None = None
    self.recoveries: list[dict[str, Any]] = []  # ordered C receipts, refusals kept
    # Preserved at the first real DONE so an explicit recovery (C) cannot
    # erase the observation Z must still testify about.
    self.done_observed = False
    self.done_observation: dict[str, Any] | None = None
    self.z: dict[str, Any] | None = None
    self.closed = False


class LivePeer:
  """One FIFO endpoint: live Group cycles plus the delegated C023 service."""

  def __init__(self) -> None:
    self.cycles: list[_Cycle] = []
    self.current: _Cycle | None = None
    self.mfe: MfePeer | None = None
    self.windows: list[dict[str, Any]] = []
    self.rings: list[dict[str, Any]] = []
    self._seq = 0

  # -- shared helpers --------------------------------------------------

  def _next_seq(self) -> int:
    self._seq += 1
    return self._seq

  def _message(self, message_type: str, fields: dict[str, Any]) -> BoundaryMessage:
    """Sealed C017 DTO: pack then unpack, so only canonical wire values travel."""
    spec = wire.MESSAGES[wire.MSG_INDEX[message_type]]
    msg = BoundaryMessage(
      message_type=message_type,
      channel=wire.CHANNELS[spec["channel"]],
      direction=wire.DIRS[spec["dir"]],
      fields=fields,
      message_seq=self._next_seq(),
    )
    return BoundaryMessage.from_wire(msg.pack_wire())

  def _cycle(self, command: str) -> _Cycle:
    if self.current is None:
      raise PeerError(f"{command} requires an open live Group cycle (F opens one)")
    return self.current

  def _memory(self, command: str) -> MfePeer:
    if self.mfe is None:
      raise PeerError(f"{command} requires an open MemoryService window (M opens one)")
    return self.mfe

  def _note_done(self, cycle: _Cycle) -> None:
    # Only an actually-DONE domain yields a DONE observation; an active
    # drain must never be recorded as one, or a later recovery refusal
    # would be misread by Z as completed reset evidence.
    if cycle.done_observed or not cycle.group.reset_domain.is_done:
      return
    cycle.done_observed = True
    cycle.done_observation = cycle.group.reset_domain.tile_reset_observation(cycle.tile)

  # -- live Group commands ---------------------------------------------

  def _fault(self, rest: list[str]) -> str:
    generation, tile, ctx_slot, ctx_gen, code, severity, source, engine, op = _words(rest, 9, "F")
    if self.current is not None:
      raise PeerError("live Group cycle already open; Z closes it")
    if code >= len(FAULT_CODES):
      raise PeerError(f"wire fault code {code} outside 0..{len(FAULT_CODES) - 1}")
    group, _sequencer, _grid, _state = TestGridSignalAggregation._dispatch_and_get_grid()
    group.reset_domain.attach_tile_reset(tile, generation)
    report = self._message("FAULT_REPORT", {
      "tile_id": tile, "ctx_slot": ctx_slot, "ctx_generation": ctx_gen,
      "code": FAULT_CODES[code], "severity": severity, "source_id": source,
      "engine_id": engine, "op_seq": op,
    })
    fault_idx = group.accept_tile_fault(report, cycle=group._last_step_cycle)
    if fault_idx < 0:
      # Valid report refused because a reset is already active/done: the
      # Group keeps its ring fact and FSM; nothing opened here.
      raise PeerError("repeated FAULT_REPORT refused by the active/done Group")
    request = group.reset_domain.request
    domain = request.domain.name if request is not None and request.domain is not None else None
    self.current = _Cycle(group, tile, generation, fault_idx, domain)
    return _reply(0, fault_idx, 0)

  def _next_phase(self) -> str:
    cycle = self._cycle("N")
    group, tile = cycle.group, cycle.tile
    cycle_number = group._last_step_cycle + 1  # never step the same cycle twice
    for steps in range(MAX_PHASE_STEPS + 1):
      request = group.reset_domain.take_tile_reset_request(tile)
      if request is not None:
        raw_phase = request.fields["phase"]
        try:  # sealed codec emits canonical names; a normalized index is fine
          phase_index = WIRE_PHASES.index(raw_phase) if isinstance(raw_phase, str) else int(raw_phase)
        except ValueError as exc:
          raise PeerError(f"reset request phase {raw_phase!r} outside the frozen order") from exc
        if not 0 <= phase_index < len(WIRE_PHASES):
          raise PeerError(f"reset request phase index {phase_index} outside the frozen order")
        phase = WIRE_PHASES[phase_index]
        generation = int(request.fields["run_generation"])
        if generation != cycle.generation:
          raise PeerError(f"reset request generation {generation} != attached {cycle.generation}")
        cycle.requests.append({
          "phase": phase, "phase_index": phase_index, "generation": generation,
        })
        return _reply(0, phase_index, generation)
      if group.reset_domain.is_done:
        self._note_done(cycle)
        return _reply(0, DONE_PHASE, cycle.generation)
      if steps == MAX_PHASE_STEPS:
        break  # polled once more after the last permitted step, above
      group.step(cycle_number)
      cycle_number += 1
    raise PeerError(f"no reset request within {MAX_PHASE_STEPS} real Group steps")

  def _ack(self, rest: list[str]) -> str:
    phase_index, generation, marked, mask = _words(rest, 4, "A")
    cycle = self._cycle("A")
    if phase_index >= len(WIRE_PHASES):
      raise PeerError(f"phase index {phase_index} outside the frozen order")
    name = WIRE_PHASES[phase_index]
    ack = self._message("RESET_PHASE_ACK", {
      "phase": name, "run_generation": generation,
      "pending_to_reset": marked, "evidence_mask": mask,
    })
    cycle.group.reset_domain.observe_tile_reset_ack(cycle.tile, ack)
    cycle.acks.append({
      "phase": name, "phase_index": phase_index, "generation": generation,
      "marked": marked, "mask": mask,
    })
    return _reply(0, 0, 0)

  def _poison(self, rest: list[str]) -> str:
    generation, tile = _words(rest, 2, "P")
    cycle = self._cycle("P")
    if generation != cycle.generation or tile != cycle.tile:
      raise PeerError(f"isolation failure {generation}:{tile} is not this attached "
                      f"{cycle.generation}:{cycle.tile} run/tile")
    group = cycle.group
    group.note_tile_isolation_failure(tile, generation)  # TileGroup method only (Section C)
    if group.poisoned_reason is None:
      raise PeerError("isolation failure did not set the sticky Group poison")
    cycle.poison = {"tile": tile, "generation": generation, "observed": True}
    return _reply(0, 1, 0)

  def _recover(self) -> str:
    cycle = self._cycle("C")
    self._note_done(cycle)  # no-op unless the domain truly reached DONE
    allowed, refusal = 1, None
    try:
      cycle.group.reset()
    except MemoryInvariantError as exc:  # the one legitimate refusal class
      allowed, refusal = 0, str(exc)
    poisoned = 1 if cycle.group.poisoned_reason is not None else 0
    cycle.recoveries.append({"allowed": allowed, "poisoned": poisoned, "refusal": refusal})
    return _reply(0, allowed, poisoned)

  def _close(self) -> str:
    cycle = self._cycle("Z")
    group = cycle.group
    if group.reset_domain.is_done:
      self._note_done(cycle)
    if not cycle.done_observed:
      raise PeerError("no real Group DONE observed for this cycle")
    observation = cycle.done_observation
    assert observation is not None
    phases = [str(name) for name in observation.get("phases", [])]
    acked = [str(ack["phase"]) for ack in cycle.acks]
    if phases != acked:
      raise PeerError(f"observed phases {phases} != acknowledged {acked}")
    mark_acks = [ack for ack in cycle.acks if ack["phase"] == "MARK_EVENTS"]
    if mark_acks and observation.get("pending_to_reset") != mark_acks[0]["marked"]:
      raise PeerError(f"pending_to_reset {observation.get('pending_to_reset')!r} != observed "
                      f"MARK count {mark_acks[0]['marked']}")
    drain_acks = [ack for ack in cycle.acks if ack["phase"] == "DRAIN"]
    if drain_acks:
      quiescent = observation.get("is_quiescent_snapshot")
      if not isinstance(quiescent, dict) or set(quiescent) != set(DRAIN_EVIDENCE):
        raise PeerError(f"is_quiescent_snapshot must name the six drain evidence bits, "
                        f"got {quiescent!r}")
      if drain_acks[0]["mask"] == 0x3F and not all(quiescent.values()):
        raise PeerError(f"DRAIN acked full evidence but the snapshot shows {quiescent!r}")
    unclosed = group.unclosed_l2_objects()
    leaks = {
      "active_sequencers": bool(group._active_sequencers),
      "grid_routes": bool(group._grid_routes),
      "task_leases": bool(group._task_leases),
      "l2_arenas": bool(group._l2_arenas),
      "inflight_transfers": int(group.transfer_manager.inflight_count),
      "unclosed_l2_objects": len(unclosed),
      "credit_invariants_broken": int(not group.credit_invariants_hold()),
    }
    if any(leaks.values()):
      raise PeerError(f"live Group cycle left residue: {leaks}; unclosed={unclosed[:8]}")
    cycle.z = {
      "done": True, "phases": phases,
      "pending_to_reset": observation.get("pending_to_reset"),
      "is_quiescent_snapshot": observation.get("is_quiescent_snapshot"),
      "leak_checks": leaks,
    }
    cycle.closed = True
    self.cycles.append(cycle)
    self.current = None
    return _reply(0, 0, 0)

  def _ring(self, rest: list[str]) -> str:
    depth, count = _words(rest, 2, "G")
    if depth <= 0 or count <= 0 or count > MAX_RING_WRITES:
      raise PeerError(f"unreasonable FaultRing geometry depth={depth} count={count}")
    ring = FaultRing(slots=depth)
    for source in range(count):
      ring.write(FaultRecord(source=source))
    latest, oldest = ring.latest(), ring.read(0)
    if latest is None or oldest is None:
      raise PeerError("FaultRing has no records after writes")
    retained = len(ring)
    # Observe the real bounded ring, not just the append-only counters.
    if ring.write_count != count or retained != min(depth, count):
      raise PeerError(f"FaultRing retained {retained} of {count} writes at depth {depth}")
    if oldest.source != count - retained:
      raise PeerError(f"FaultRing oldest retained source {oldest.source} != {count - retained}")
    if latest.fault_record_index != (count - 1) % depth:
      raise PeerError(f"FaultRing last physical index {latest.fault_record_index} != "
                      f"{(count - 1) % depth}")
    self.rings.append({
      "depth": depth, "writes": count, "write_count": ring.write_count, "len": retained,
      "oldest_source": oldest.source, "last_index": latest.fault_record_index,
      "latest_source": latest.source,
    })
    return _reply(0, ring.write_count, latest.source)

  # -- real C023 MemoryService commands --------------------------------

  def _memory_open(self) -> str:
    if self.mfe is not None and not self.mfe.done:
      raise PeerError("unclosed MemoryService window still open")
    self.mfe = MfePeer()
    return _reply(0, 0, 0)

  def _forward(self, kind: str, rest: list[str], line: str) -> str:
    mfe = self._memory(kind)
    if kind in ("R", "W"):
      values = _words(rest, 12 if kind == "W" else 11, kind)
      if values[6] != TRANSFER_BYTES:
        raise PeerError(f"{kind} transfers are {TRANSFER_BYTES}-byte only here, "
                        f"got length {values[6]}")
    answer = mfe.transact(line).split()
    first_two = [*answer, "0", "0"][:2]
    return " ".join([*first_two, "0"]) + "\n"

  def _peek(self, rest: list[str]) -> str:
    (offset,) = _words(rest, 1, "B")
    mfe = self._memory("B")
    store = mfe.service.byte_store
    if store is None:  # the service admits no store; reads are impossible
      raise PeerError("MemoryService has no byte store attached")
    data = store.read_hbm(BASE + SIZE + offset, TRANSFER_BYTES)
    return _reply(0, int.from_bytes(data, "little"), 0)

  def _memory_close(self) -> str:
    mfe = self._memory("E")
    mfe.service.assert_closed()
    mfe.done = True
    self.windows.append({
      "requests": mfe.requests, "writes": mfe.writes,
      "terminals": mfe.terminals, "service": mfe.service.snapshot(),
    })
    self.mfe = None
    return _reply(0, len(mfe.requests), len(mfe.terminals))

  # -- dispatch ----------------------------------------------------------

  def transact(self, line: str) -> str:
    words = line.split()
    kind, rest = words[0], words[1:]
    if kind == "F":
      return self._fault(rest)
    if kind == "N":
      return self._next_phase()
    if kind == "A":
      return self._ack(rest)
    if kind == "P":
      return self._poison(rest)
    if kind == "C":
      return self._recover()
    if kind == "Z":
      return self._close()
    if kind == "G":
      return self._ring(rest)
    if kind == "M":
      return self._memory_open()
    if kind in ("R", "W", "T"):
      return self._forward(kind, rest, line)
    if kind == "B":
      return self._peek(rest)
    if kind == "E":
      return self._memory_close()
    raise PeerError(f"unknown live boundary command {kind!r}")


def _receipt(cycle: _Cycle) -> dict[str, Any]:
  return {
    "fault_idx": cycle.fault_idx,
    "domain": cycle.domain,
    "tile": cycle.tile,
    "generation": cycle.generation,
    "requests": cycle.requests,
    "acks": cycle.acks,
    "poison": cycle.poison,
    "recoveries": cycle.recoveries,
    "z": cycle.z,
  }


@contextmanager
def serve(evidence: Path, labels: Sequence[str]) -> Iterator[None]:
  """Serve one LivePeer per label behind the proven C032 non-blocking FIFO loop.

  The W4 `c032_mfe_peer.serve` loop is parameterized around its own Peer and
  report, so this carries the same O_RDWR|O_NONBLOCK pattern (a bench that
  dies mid-line never hangs the peer) with the C034 command set and receipts.
  """
  selector = selectors.DefaultSelector()
  streams: dict[str, tuple[int, int]] = {}
  peers: dict[str, LivePeer] = {}
  buffers: dict[str, bytes] = {}
  errors: dict[str, list[str]] = {}
  stop = threading.Event()
  for label in labels:
    request_path, response_path = (evidence / f"{label}.{suffix}" for suffix in ("req", "rsp"))
    os.mkfifo(request_path)
    os.mkfifo(response_path)
    request_fd = os.open(request_path, os.O_RDWR | os.O_NONBLOCK)
    response_fd = os.open(response_path, os.O_RDWR | os.O_NONBLOCK)
    streams[label] = (request_fd, response_fd)
    buffers[label] = b""
    selector.register(request_fd, selectors.EVENT_READ, label)

  def run() -> None:
    while not stop.is_set():
      for key, _events in selector.select(0.1):
        label = key.data
        request_fd, response_fd = streams[label]
        buffers[label] += os.read(request_fd, 65536)
        while b"\n" in buffers[label]:
          raw, buffers[label] = buffers[label].split(b"\n", 1)
          if not raw:
            continue
          peer = peers.setdefault(label, LivePeer())
          line = raw.decode()
          try:
            answer = peer.transact(line)
          except Exception as exc:
            errors.setdefault(label, []).append(f"{type(exc).__name__}: {exc}")
            answer = "ffffffff 0 0\n"
          with (evidence / f"{label}.peer.log").open("a") as log:
            log.write(f"> {line}\n< {answer}")
          os.write(response_fd, answer.encode())

  thread = threading.Thread(target=run, name="c034-live-peer", daemon=True)
  thread.start()
  try:
    yield
  finally:
    stop.set()
    thread.join(timeout=5)
    selector.close()
    for request_fd, response_fd in streams.values():
      os.close(request_fd)
      os.close(response_fd)
    report = {}
    for label, peer in peers.items():
      cycles = list(peer.cycles)
      if peer.current is not None:
        cycles.append(peer.current)  # an un-closed cycle is residue, kept raw
      report[label] = {
        "scenarios": [_receipt(cycle) for cycle in cycles],
        "memory_windows": peer.windows,
        "rings": peer.rings,
        "open_scenario": peer.current is not None,
        "open_memory": peer.mfe is not None,
        "errors": errors.get(label, []),
      }
    (evidence / "group-peer.json").write_text(json.dumps(report, indent=2) + "\n")
    if thread.is_alive():
      raise RuntimeError("live C034 peer failed to shut down")
