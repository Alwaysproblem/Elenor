"""ctypes bindings for the eight-entry CN C ABI (C025 / W1-03).

Wraps `libcn_cosim_stub.so` (unit stub) or `libcn_cosim.so` (C040 real Tile top):
  bp_abi_info / bp_begin / bp_layout / bp_drive / bp_tick / bp_collect / bp_status / bp_destroy

Invariants:
- Preallocates fixed-capacity `CnSlot`, `CnSlotLayout`, per-slot input buffers, and `CnCollect`
  output buffer at `begin()`/`layout()` time so steady-state ticks perform zero C buffer
  allocations or whole-capacity copies.
- Verifies `bp_abi_info()` 128-hex identity against `sha256(cn_protocol_v1.yaml) ++
  sha256(cn_layout.json)` and cross-checks `bp_layout` slot capacities against `cn_layout.json`
  (`payload_words` + `header_words`).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STUB = REPO / "out/coral_nexus/cache/cn_abi_stub/libcn_cosim_stub.so"
PROTOCOL_YAML = REPO / "hardware/coral_nexus/protocol/cn_protocol_v1.yaml"
LAYOUT_JSON = REPO / "hardware/coral_nexus/protocol/generated/cn_layout.json"

CN_ABI_VERSION = 1
CN_OK = 0
CN_ERR_ARGS = 1
CN_ERR_STATE = 2
CN_ERR_RANGE = 3
CN_ERR_ABI = 4
CN_ERR_FATAL_DUT = 5
CN_EP_NONE = 0
CN_EP_PAYLOAD_READY_MISMATCH = 1

CN_STUB_FLAG_CORRUPT_T2G = 0x01
CN_STUB_FLAG_EMIT_MEM_REQ = 0x02
CN_STUB_FLAG_GLITCH_PAYLOAD = 0x04


class CnAbiError(RuntimeError):
  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


class CnBeginCfg(ctypes.Structure):
  _fields_ = [
    ("abi_version", ctypes.c_uint32),
    ("tile_count", ctypes.c_uint32),
    ("ready_seed", ctypes.c_uint32),
    ("inject_fatal_at_tick", ctypes.c_int32),
    ("inject_abort_at_tick", ctypes.c_int32),
    ("trace", ctypes.c_uint8),
    ("reserved", ctypes.c_uint8 * 7),
  ]


class CnSlot(ctypes.Structure):
  _fields_ = [
    ("slot_id", ctypes.c_uint32),
    ("bytes", ctypes.POINTER(ctypes.c_uint8)),
    ("len", ctypes.c_uint32),
  ]


class CnIoVec(ctypes.Structure):
  _fields_ = [("slots", ctypes.POINTER(CnSlot)), ("slot_count", ctypes.c_uint32)]


class CnCollect(ctypes.Structure):
  _fields_ = [("buf", ctypes.POINTER(ctypes.c_uint8)), ("buf_len", ctypes.c_uint32)]


class CnSlotLayout(ctypes.Structure):
  _fields_ = [("slot_id", ctypes.c_uint32), ("offset", ctypes.c_uint32), ("len", ctypes.c_uint32)]


class CnStatus(ctypes.Structure):
  _fields_ = [
    ("abi_version", ctypes.c_uint32),
    ("fatal_code", ctypes.c_uint32),
    ("proto_err", ctypes.c_uint32),
    ("message_seq_gap_seen", ctypes.c_uint32),
    ("accepted_but_unreturned_credit", ctypes.c_uint32),
    ("reserved", ctypes.c_uint32),
    ("tick_count", ctypes.c_uint64),
    ("posedge_count", ctypes.c_uint64),
    ("drive_count", ctypes.c_uint64),
    ("collect_count", ctypes.c_uint64),
    ("accepted_beats", ctypes.c_uint64),
    ("returned_beats", ctypes.c_uint64),
    ("staged_pending", ctypes.c_uint64),
  ]


@dataclass(frozen=True)
class CollectedRecord:
  slot_id: int
  is_t2g: bool
  byte_len: int
  wire_bytes: bytes


def expected_abi_hash(
  protocol_path: Path = PROTOCOL_YAML,
  layout_path: Path = LAYOUT_JSON,
) -> str:
  """Compute the 128-hex ABI identity `sha256(cn_protocol_v1.yaml) ++ sha256(cn_layout.json)`."""
  return (
    hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    + hashlib.sha256(layout_path.read_bytes()).hexdigest()
  )


def expected_slot_geometries(layout_path: Path = LAYOUT_JSON) -> list[dict[str, Any]]:
  """Compute expected per-(channel, dir) slot geometries directly from `cn_layout.json`."""
  doc = json.loads(layout_path.read_text())
  word_bytes = int(doc["container"]["word_bytes"])
  header_words = int(doc["header_words"])
  channels = [str(c) for c in doc["channels"]]
  dirs = [str(d) for d in doc["dirs"]]
  out: list[dict[str, Any]] = []
  offset = 0
  slot_id = 0
  for ch_idx, ch_name in enumerate(channels):
    for dir_name in ("G2T", "T2G"):
      dir_idx = dirs.index(dir_name)
      msgs = [
        m
        for m in doc["messages"]
        if int(m["channel_index"]) == ch_idx
        and int(m["dir_index"]) == dir_idx
        and not m.get("host_only", False)
      ]
      max_pwords = max((int(m["payload_words"]) for m in msgs), default=0)
      length = (header_words + max_pwords) * word_bytes
      out.append(
        {
          "slot_id": slot_id,
          "channel": ch_name,
          "channel_index": ch_idx,
          "dir": dir_name,
          "dir_index": dir_idx,
          "offset": offset,
          "len": length,
        }
      )
      offset += length
      slot_id += 1
  return out


def parse_collect_buffer(raw: bytes | memoryview) -> list[CollectedRecord]:
  """Decode the binary `bp_collect` buffer into `CollectedRecord` entries."""
  mv = memoryview(raw)
  if len(mv) < 8:
    raise CnAbiError("COLLECT_CORRUPT", f"header shorter than 8 bytes ({len(mv)})")
  count, total_bytes = struct.unpack_from("<II", mv, 0)
  if total_bytes > len(mv):
    raise CnAbiError("COLLECT_CORRUPT", f"total_bytes {total_bytes} > buffer {len(mv)}")
  cursor = 8
  records: list[CollectedRecord] = []
  for _ in range(count):
    if cursor + 8 > total_bytes:
      raise CnAbiError("COLLECT_CORRUPT", "truncated record header")
    slot_id, tagged_len = struct.unpack_from("<II", mv, cursor)
    cursor += 8
    is_t2g = bool(tagged_len & 0x80000000)
    byte_len = int(tagged_len & 0x7FFFFFFF)
    if is_t2g:
      if cursor + byte_len > total_bytes:
        raise CnAbiError("COLLECT_CORRUPT", "truncated T2G frame bytes")
      payload = bytes(mv[cursor : cursor + byte_len])
      cursor += byte_len
    else:
      payload = b""
    records.append(
      CollectedRecord(slot_id=int(slot_id), is_t2g=is_t2g, byte_len=byte_len, wire_bytes=payload)
    )
  return records


def load(so_path: Path | str = DEFAULT_STUB, *, verify_hash: bool = True) -> CnAbi:
  path = Path(so_path)
  if not path.is_file():
    raise CnAbiError("LIBRARY_MISSING", str(path))
  try:
    lib = ctypes.CDLL(str(path))
  except OSError as exc:
    raise CnAbiError("HOST_GLIBC_TOO_OLD", str(exc)) from exc
  wrapper = CnAbi(lib, path)
  if verify_hash:
    got = wrapper.abi_info()
    exp = expected_abi_hash()
    if got != exp:
      raise CnAbiError("ABI_HASH_MISMATCH", f"{got} != {exp}")
  return wrapper


class CnAbi:
  """Preallocated ctypes wrapper over the 8 authoritative C ABI functions."""

  def __init__(self, lib: ctypes.CDLL, path: Path, *, collect_capacity: int = 16384) -> None:
    self._lib = lib
    self.path = path
    self._sim: ctypes.c_void_p | None = None
    self._collect_capacity = collect_capacity
    self._collect_buf = (ctypes.c_uint8 * collect_capacity)()
    self._collect_struct = CnCollect(
      buf=ctypes.cast(self._collect_buf, ctypes.POINTER(ctypes.c_uint8)),
      buf_len=collect_capacity,
    )
    self._slot_specs: list[dict[str, Any]] = expected_slot_geometries()
    n_slots = len(self._slot_specs)
    self._drive_slots = (CnSlot * n_slots)()
    self._drive_Staging = [
      (ctypes.c_uint8 * int(spec["len"]))() for spec in self._slot_specs
    ]
    self._iov = CnIoVec(slots=self._drive_slots, slot_count=0)

    lib.bp_abi_info.restype = ctypes.c_uint32
    lib.bp_abi_info.argtypes = [ctypes.c_uint8 * 129]
    lib.bp_begin.restype = ctypes.c_void_p
    lib.bp_begin.argtypes = [ctypes.POINTER(CnBeginCfg)]
    lib.bp_layout.restype = ctypes.c_int
    lib.bp_layout.argtypes = [ctypes.c_void_p, ctypes.POINTER(CnSlotLayout), ctypes.c_uint32]
    lib.bp_drive.restype = ctypes.c_int
    lib.bp_drive.argtypes = [ctypes.c_void_p, ctypes.POINTER(CnIoVec)]
    lib.bp_tick.restype = ctypes.c_int
    lib.bp_tick.argtypes = [ctypes.c_void_p]
    lib.bp_collect.restype = ctypes.c_int
    lib.bp_collect.argtypes = [ctypes.c_void_p, ctypes.POINTER(CnCollect)]
    lib.bp_status.restype = ctypes.c_int
    lib.bp_status.argtypes = [ctypes.c_void_p, ctypes.POINTER(CnStatus)]
    lib.bp_destroy.restype = None
    lib.bp_destroy.argtypes = [ctypes.c_void_p]

  def abi_info(self) -> str:
    buf = (ctypes.c_uint8 * 129)()
    ver = self._lib.bp_abi_info(buf)
    if ver != CN_ABI_VERSION:
      raise CnAbiError("ABI_VERSION_MISMATCH", str(ver))
    if buf[128] != 0:
      raise CnAbiError("ABI_HASH_UNTERMINATED")
    return bytes(buf[:128]).decode("ascii")

  def begin(self, cfg: CnBeginCfg) -> None:
    if self._sim is not None:
      raise CnAbiError("ALREADY_BEGUN")
    handle = self._lib.bp_begin(ctypes.byref(cfg))
    if not handle:
      raise CnAbiError("BEGIN_FAILED")
    self._sim = ctypes.c_void_p(handle)

  def layout_raw(self, count: int) -> tuple[int, list[tuple[int, int, int]]]:
    if self._sim is None:
      raise CnAbiError("NOT_BEGUN")
    arr = (CnSlotLayout * max(count, 1))()
    rc = int(self._lib.bp_layout(self._sim, arr, count))
    if rc != CN_OK:
      return rc, []
    return rc, [(int(arr[i].slot_id), int(arr[i].offset), int(arr[i].len)) for i in range(count)]

  def layout(self) -> list[dict[str, Any]]:
    """Query `bp_layout` and verify every slot matches `cn_layout.json`."""
    count = len(self._slot_specs)
    rc, rows = self.layout_raw(count)
    if rc != CN_OK:
      raise CnAbiError("LAYOUT_FAILED", str(rc))
    for spec, (sid, off, length) in zip(self._slot_specs, rows):
      if (sid, off, length) != (spec["slot_id"], spec["offset"], spec["len"]):
        raise CnAbiError(
          "LAYOUT_MISMATCH",
          f"slot {sid}: got ({off}, {length}) != expected ({spec['offset']}, {spec['len']})",
        )
    return list(self._slot_specs)

  def drive(self, items: list[tuple[int, bytes | memoryview]]) -> int:
    """Execute P0 `bp_drive` using preallocated staging buffers (zero per-tick heap allocation)."""
    if self._sim is None:
      raise CnAbiError("NOT_BEGUN")
    if len(items) > len(self._slot_specs):
      return CN_ERR_ARGS
    for idx, (slot_id, data) in enumerate(items):
      mv = memoryview(data)
      n = len(mv)
      if 0 <= slot_id < len(self._drive_Staging) and n <= len(self._drive_Staging[slot_id]):
        stage = self._drive_Staging[slot_id]
        if n > 0:
          ctypes.memmove(stage, mv.tobytes(), n)
        self._drive_slots[idx].slot_id = slot_id
        self._drive_slots[idx].bytes = ctypes.cast(stage, ctypes.POINTER(ctypes.c_uint8))
        self._drive_slots[idx].len = n
      else:
        # Out-of-range slot or oversize frame: pass raw pointer so C ABI returns CN_ERR_ARGS/RANGE.
        raw_bytes = mv.tobytes()
        tmp = (ctypes.c_uint8 * max(n, 1)).from_buffer_copy(raw_bytes or b"\x00")
        self._drive_slots[idx].slot_id = slot_id
        self._drive_slots[idx].bytes = ctypes.cast(tmp, ctypes.POINTER(ctypes.c_uint8))
        self._drive_slots[idx].len = n
    self._iov.slot_count = len(items)
    return int(self._lib.bp_drive(self._sim, ctypes.byref(self._iov)))

  def tick(self) -> int:
    if self._sim is None:
      raise CnAbiError("NOT_BEGUN")
    return int(self._lib.bp_tick(self._sim))

  def collect_raw(self, buf_len: int) -> tuple[int, bytes]:
    if self._sim is None:
      raise CnAbiError("NOT_BEGUN")
    if buf_len <= self._collect_capacity:
      col = CnCollect(
        buf=ctypes.cast(self._collect_buf, ctypes.POINTER(ctypes.c_uint8)),
        buf_len=buf_len,
      )
      rc = int(self._lib.bp_collect(self._sim, ctypes.byref(col)))
      if rc != CN_OK:
        return rc, b""
      total = struct.unpack_from("<I", memoryview(self._collect_buf), 4)[0]
      return rc, bytes(memoryview(self._collect_buf)[:total])
    big = (ctypes.c_uint8 * buf_len)()
    col = CnCollect(buf=ctypes.cast(big, ctypes.POINTER(ctypes.c_uint8)), buf_len=buf_len)
    rc = int(self._lib.bp_collect(self._sim, ctypes.byref(col)))
    if rc != CN_OK:
      return rc, b""
    total = struct.unpack_from("<I", memoryview(big), 4)[0]
    return rc, bytes(memoryview(big)[:total])

  def collect(self) -> tuple[int, list[CollectedRecord]]:
    if self._sim is None:
      raise CnAbiError("NOT_BEGUN")
    rc = int(self._lib.bp_collect(self._sim, ctypes.byref(self._collect_struct)))
    if rc != CN_OK:
      return rc, []
    return rc, parse_collect_buffer(memoryview(self._collect_buf))

  def status(self) -> CnStatus:
    if self._sim is None:
      raise CnAbiError("NOT_BEGUN")
    st = CnStatus()
    rc = int(self._lib.bp_status(self._sim, ctypes.byref(st)))
    if rc != CN_OK:
      raise CnAbiError("STATUS_FAILED", str(rc))
    return st

  def destroy(self) -> None:
    if self._sim is not None:
      self._lib.bp_destroy(self._sim)
      self._sim = None

  def __del__(self) -> None:
    self.destroy()


def build_meta(cache_dir: Path = DEFAULT_STUB.parent) -> dict[str, object]:
  return json.loads((cache_dir / "build_meta.json").read_text())  # type: ignore[no-any-return]
