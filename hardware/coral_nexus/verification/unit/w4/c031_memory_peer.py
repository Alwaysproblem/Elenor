"""Test transport only: RTL FIFOs call the actual C023 MemoryService.

No L1 execution, simulated completion timers, or response data model lives here.
Each compiled bench/mutant has independent service state and a durable transcript.
"""

from __future__ import annotations

import json
import os
import selectors
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from pipeline_validator.cosim.memory_service import MEM_OP_READ, MemoryService, MemRequest
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.memory.hbm_region import HBMRegion

BASE = 0x100000
SIZE = 4096
BINDINGS = {(7, 1): ("c031_input", BASE, 0),
            (8, 1): ("c031_peer", BASE + SIZE, 41),
            (7, 2): ("c031_generation2", BASE + 2 * SIZE, 83)}


def make_service() -> MemoryService:
  store = ByteStore()
  hbm = HBMRegion(byte_store=store)
  service = MemoryService(hbm, store)
  for (_bid, generation), (name, base, salt) in BINDINGS.items():
    hbm.bind_external(GlobalBinding(name, base, SIZE, "r"))
    store.seed_hbm(base, bytes((i * 17 + (i // 64) * 13 + 3 + salt) & 255 for i in range(SIZE)))
    service.register_binding(name, base, SIZE, "r", generation=generation)
  return service


class Peer:
  def __init__(self) -> None:
    self.service = make_service()
    self.requests: list[dict[str, int]] = []
    self.terminals: list[dict[str, int]] = []
    self.done = False

  def transact(self, line: str) -> str:
    words = line.split()
    kind, numbers = words[0], [int(word, 16) for word in words[1:]]
    if kind == "R":
      if len(numbers) != 11:
        raise ValueError("MEM_REQ must carry all eleven transport fields")
      txn, sid, count, bid, bgen, off, length, ctx, cgen, arena, scope = numbers
      name, _base, _salt = BINDINGS[(bid, bgen)]
      response = self.service.submit_mem_req(
        MemRequest(
          tile_id=0, ctx_slot=ctx, ctx_generation=cgen, txn_id=txn,
          slice_id=sid, slice_count=count, arena_alloc_gen=arena,
          binding_name=name, binding_generation=bgen, binding_offset=off,
          scope_epoch=scope, mem_op=MEM_OP_READ, length=length,
        )
      )
      if response.data is None:
        raise ValueError("MemoryService read returned no bytes")
      self.requests.append({"txn": txn, "offset": off, "ctx": ctx, "generation": cgen,
                            "binding": bid, "binding_generation": bgen, "arena": arena, "scope": scope})
      return f"{response.status:x} {int.from_bytes(response.data, 'little'):0128x}\n"
    if kind == "T":
      txn, sid, count, final = numbers
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

  thread = threading.Thread(target=run, name="c031-memory-service", daemon=True)
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
      label: {"done": peer.done, "requests": peer.requests, "terminals": peer.terminals,
              "service": peer.service.snapshot(), "error": errors.get(label)}
      for label, peer in peers.items()
    }
    (evidence / "memory-service.json").write_text(json.dumps(report, indent=2) + "\n")
    if thread.is_alive():
      raise RuntimeError("MemoryService peer failed to shut down")
