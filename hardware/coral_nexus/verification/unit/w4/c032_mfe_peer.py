"""Test transport only: RTL MFE FIFOs call the actual C023 MemoryService.

Extends the C031 peer pattern with MEM_OP_WRITE: the write payload travels on
the request line so the bench can prove the written bytes are the bytes the
engine actually read. No L1 execution, simulated completion timers, or
response data model lives here. Each compiled bench/mutant has independent
service state and a durable transcript.
"""

from __future__ import annotations

import json
import os
import selectors
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from pipeline_validator.cosim.memory_service import MEM_OP_READ, MEM_OP_WRITE, MemoryService, MemRequest
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.memory.hbm_region import HBMRegion

BASE = 0x100000
SIZE = 4096
# (binding_id, generation) -> (name, base, permissions, seed salt)
BINDINGS = {
  (17, 1): ("c032_src", BASE, "r", 0),
  (18, 1): ("c032_dst", BASE + SIZE, "rw", 5),
  (17, 2): ("c032_gen2", BASE + 2 * SIZE, "r", 9),
}


def make_service() -> MemoryService:
  store = ByteStore()
  hbm = HBMRegion(byte_store=store)
  service = MemoryService(hbm, store)
  for (_bid, generation), (name, base, permissions, salt) in BINDINGS.items():
    hbm.bind_external(GlobalBinding(name, base, SIZE, permissions))
    store.seed_hbm(base, bytes((i * 17 + (i // 64) * 13 + 3 + salt) & 255 for i in range(SIZE)))
    service.register_binding(name, base, SIZE, permissions, generation=generation)
  return service


class Peer:
  def __init__(self) -> None:
    self.service = make_service()
    self.requests: list[dict[str, int]] = []
    self.writes: list[dict[str, int | str]] = []
    self.terminals: list[dict[str, int]] = []
    self.done = False

  def _submit(self, rest: list[str], kind: str) -> tuple[int, int, bytes | None]:
    expected = 12 if kind == "W" else 11
    if len(rest) != expected:
      raise ValueError(f"{kind} MEM_REQ must carry the exact transport field count")
    numbers = [int(word, 16) for word in rest[:11]]
    txn, sid, count, bid, bgen, off, length, ctx, cgen, arena, scope = numbers
    try:
      name, _base, _permissions, _salt = BINDINGS[(bid, bgen)]
    except KeyError as exc:
      raise ValueError(f"unknown binding {bid}:{bgen}") from exc
    payload = None
    if kind == "W":
      # The bench prints the payload word zero-padded; to_bytes rejects any
      # nonzero byte beyond the length field.
      payload = int(rest[11], 16).to_bytes(length, "little")
    response = self.service.submit_mem_req(
      MemRequest(
        tile_id=0,
        ctx_slot=ctx,
        ctx_generation=cgen,
        txn_id=txn,
        slice_id=sid,
        slice_count=count,
        arena_alloc_gen=arena,
        binding_name=name,
        binding_generation=bgen,
        binding_offset=off,
        scope_epoch=scope,
        mem_op=MEM_OP_READ if kind == "R" else MEM_OP_WRITE,
        length=length,
        payload=payload,
      )
    )
    record = {
      "txn": txn,
      "op": MEM_OP_READ if kind == "R" else MEM_OP_WRITE,
      "offset": off,
      "length": length,
      "ctx": ctx,
      "generation": cgen,
      "binding": bid,
      "binding_generation": bgen,
      "arena": arena,
      "scope": scope,
    }
    self.requests.append(record)
    if kind == "W":
      assert payload is not None
      self.writes.append({"txn": txn, "offset": off, "length": length, "bytes": payload.hex()})
    return response.status, response.visibility, response.data

  def transact(self, line: str) -> str:
    words = line.split()
    kind, rest = words[0], words[1:]
    if kind == "R":
      status, _visibility, data = self._submit(rest, kind)
      if data is None:
        raise ValueError("MemoryService read returned no bytes")
      return f"{status:x} {int.from_bytes(data, 'little'):0128x}\n"
    if kind == "W":
      status, visibility, _data = self._submit(rest, kind)
      return f"{status:x} {visibility:x}\n"
    if kind == "T":
      txn, sid, count, final = [int(word, 16) for word in rest]
      self.service.terminal_ack(txn, sid, count, ("COMPLETED", "ISOLATED", "FAULTED")[final])
      self.terminals.append({"txn": txn, "final": final})
      return "0\n"
    if kind == "D":
      self.service.assert_closed()
      self.done = True
      return "0\n"
    raise ValueError(f"unknown RTL service command {kind!r}")


@contextmanager
def serve(evidence: Path, labels: Sequence[str]) -> Iterator[None]:
  selector = selectors.DefaultSelector()
  streams: dict[str, tuple[int, int]] = {}
  peers: dict[str, Peer] = {}
  buffers: dict[str, bytes] = {}
  errors: dict[str, str] = {}
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
        data = os.read(request_fd, 65536)
        buffers[label] += data
        while b"\n" in buffers[label]:
          raw, buffers[label] = buffers[label].split(b"\n", 1)
          if not raw:
            continue
          if label not in peers:
            peers[label] = Peer()
          peer = peers[label]
          line = raw.decode()
          try:
            answer = peer.transact(line)
          except Exception as exc:
            errors[label] = f"{type(exc).__name__}: {exc}"
            answer = "ffffffff 0\n"
          with (evidence / f"{label}.service.log").open("a") as log:
            log.write(f"> {line}\n< {answer}")
          os.write(response_fd, answer.encode())

  thread = threading.Thread(target=run, name="c032-mfe-service", daemon=True)
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
    report = {
      label: {
        "done": peer.done,
        "requests": peer.requests,
        "writes": peer.writes,
        "terminals": peer.terminals,
        "service": peer.service.snapshot(),
        "error": errors.get(label),
      }
      for label, peer in peers.items()
    }
    (evidence / "memory-service.json").write_text(json.dumps(report, indent=2) + "\n")
    if thread.is_alive():
      raise RuntimeError("MemoryService peer failed to shut down")
