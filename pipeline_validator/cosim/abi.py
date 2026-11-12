"""ctypes bindings for the eight-entry CN C ABI (C025 / W1-03).

Loads `libcn_cosim_stub.so` (unit stub) or the real `libcn_cosim.so` (C040) and exposes the
`CnAbi` wrapper with the five-phase call discipline enforced on the Python side:
bp_begin -> bp_layout -> (bp_drive -> bp_tick -> bp_collect) x N -> bp_destroy.

When the host glibc is older than the container toolchain (Debian trixie), `load()` raises
`CnAbiError("HOST_GLIBC_TOO_OLD")`; the unit suites then execute the SAME library inside
`coral-nexus-builder:c002` via the in-container selftest driver (`build_cn_abi_stub.sh` cache).
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STUB = REPO / "out/coral_nexus/cache/cn_abi_stub/libcn_cosim_stub.so"

CN_ABI_VERSION = 1
CN_OK = 0
CN_ERR_ARGS = 1
CN_ERR_STATE = 2
CN_ERR_RANGE = 3
CN_ERR_ABI = 4
CN_ERR_FATAL_DUT = 5
CN_EP_PAYLOAD_READY_MISMATCH = 1


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


def load(so_path: Path | str = DEFAULT_STUB) -> CnAbi:
  path = Path(so_path)
  if not path.is_file():
    raise CnAbiError("LIBRARY_MISSING", str(path))
  try:
    lib = ctypes.CDLL(str(path))
  except OSError as exc:
    raise CnAbiError("HOST_GLIBC_TOO_OLD", str(exc)) from exc
  return CnAbi(lib, path)


class CnAbi:
  """Typed wrapper over the eight C entries with Python-side phase enforcement."""

  def __init__(self, lib: ctypes.CDLL, path: Path) -> None:
    self._lib = lib
    self.path = path
    self._sim: ctypes.c_void_p | None = None
    self._layout_done = False
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
    return bytes(buf[:128]).decode("ascii")

  def begin(self, cfg: CnBeginCfg) -> None:
    if self._sim is not None:
      raise CnAbiError("ALREADY_BEGUN")
    handle = self._lib.bp_begin(ctypes.byref(cfg))
    if not handle:
      raise CnAbiError("BEGIN_FAILED")
    self._sim = ctypes.c_void_p(handle)
    self._layout_done = False

  def layout(self, count: int = 16) -> list[CnSlotLayout]:
    if self._sim is None:
      raise CnAbiError("NOT_BEGUN")
    if self._layout_done:
      raise CnAbiError("LAYOUT_TWICE")
    arr = (CnSlotLayout * count)()
    rc = self._lib.bp_layout(self._sim, arr, count)
    if rc == CN_ERR_RANGE:
      raise CnAbiError("LAYOUT_RANGE")
    if rc != CN_OK:
      raise CnAbiError("LAYOUT_FAILED", str(rc))
    self._layout_done = True
    return [arr[i] for i in range(count) if arr[i].len != 0 or arr[i].slot_id < count]

  def drive(self, slots: list[tuple[int, bytes]]) -> int:
    if self._sim is None:
      raise CnAbiError("NOT_BEGUN")
    arr = (CnSlot * len(slots))()
    for i, (slot_id, payload) in enumerate(slots):
      buf = ctypes.create_string_buffer(payload, len(payload))
      arr[i].slot_id = slot_id
      arr[i].bytes = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
      arr[i].len = len(payload)
    vec = CnIoVec(slots=arr, slot_count=len(slots))
    return self._lib.bp_drive(self._sim, ctypes.byref(vec))

  def tick(self) -> int:
    if self._sim is None:
      raise CnAbiError("NOT_BEGUN")
    return self._lib.bp_tick(self._sim)

  def collect(self, buf_len: int = 4096) -> tuple[int, bytes]:
    if self._sim is None:
      raise CnAbiError("NOT_BEGUN")
    buf = ctypes.create_string_buffer(buf_len)
    col = CnCollect(buf=ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)), buf_len=buf_len)
    rc = self._lib.bp_collect(self._sim, ctypes.byref(col))
    return rc, buf.raw[:buf_len]

  def status(self) -> CnStatus:
    if self._sim is None:
      raise CnAbiError("NOT_BEGUN")
    st = CnStatus()
    rc = self._lib.bp_status(self._sim, ctypes.byref(st))
    if rc != CN_OK:
      raise CnAbiError("STATUS_FAILED", str(rc))
    return st

  def destroy(self) -> None:
    if self._sim is not None:
      self._lib.bp_destroy(self._sim)
      self._sim = None


def expected_abi_hash() -> str:
  """Recompute the 128-hex ABI identity from the sealed protocol artifacts."""
  import hashlib

  protocol = (REPO / "hardware/coral_nexus/protocol/cn_protocol_v1.yaml").read_bytes()
  layout = (REPO / "hardware/coral_nexus/protocol/generated/cn_layout.json").read_bytes()
  return hashlib.sha256(protocol).hexdigest() + hashlib.sha256(layout).hexdigest()


def build_meta(cache_dir: Path = DEFAULT_STUB.parent) -> dict:
  return json.loads((cache_dir / "build_meta.json").read_text())
