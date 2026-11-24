"""PROGRAM-channel install/read/invalidate service (C024 / W1-02).

Group-side program service per W1 plan §8.2:
- INSTALL header (G2T) arms one install record for an ALREADY-REGISTERED immutable image
  (C014-schema validated metadata: program_id/version/package_hash/epoch + real text bytes).
- The Tile drives segment fetches with PROGRAM_READ_REQ (T2G); the service answers
  PROGRAM_READ_RESP (G2T) with real chunk bytes on the service data channel and the
  `hbm_chunk_len` on the wire.
- Install completion is ONLY the Tile's PROGRAM_INSTALL_ACK (installed_len + measured_hash);
  the service verifies both against the registered image and never fabricates completion.
- PROGRAM_READY records the Tile-local handle; PROGRAM_INVALIDATE -> INVALIDATE_ACK.

Host prepare validates packages, then registers immutable image identities and uploads their linked
text to HBM. Each ProgramService serves one Tile's ordered PROGRAM channel, independently of data DMA.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..execution_ir import GlobalBinding
from ..memory.allocator import AllocationHandle, MemoryInvariantError
from ..memory.byte_store import ByteStore
from ..memory.hbm_region import HBMRegion
from .tile_port import BoundaryMessage

CHUNK_BYTES = 64


class ProgramServiceError(RuntimeError):
  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


@dataclass(frozen=True)
class ProgramImage:
  """Immutable validated program image metadata (C014 schema fields + real text bytes)."""

  program_id: int
  version: int
  package_hash: bytes  # 32 bytes
  program_epoch: int
  text: bytes

  def __post_init__(self) -> None:
    for name, val in (("program_id", self.program_id), ("version", self.version)):
      if isinstance(val, bool) or not isinstance(val, int) or val < 1:
        raise TypeError(f"ProgramImage.{name} must be a positive int")
    if not isinstance(self.package_hash, bytes) or len(self.package_hash) != 32:
      raise TypeError("ProgramImage.package_hash must be 32 bytes")
    if (
      isinstance(self.program_epoch, bool)
      or not isinstance(self.program_epoch, int)
      or self.program_epoch < 0
    ):
      raise TypeError("ProgramImage.program_epoch must be a non-negative int")
    if not isinstance(self.text, bytes) or not self.text:
      raise TypeError("ProgramImage.text must be non-empty bytes")


@dataclass
class _InstallRecord:
  image: ProgramImage
  slot: int
  acked: bool = False
  ready_handle: int | None = None
  bytes_served: int = 0
  invalidating: bool = False


class ProgramService:
  """Serves PROGRAM channel traffic against registered immutable images."""

  def __init__(self, *, chunk_bytes: int = CHUNK_BYTES, hbm: HBMRegion | None = None) -> None:
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
      raise ValueError("chunk_bytes must be a positive int")
    self.chunk_bytes = chunk_bytes
    self.hbm = hbm if hbm is not None else HBMRegion(byte_store=ByteStore())
    if self.hbm.byte_store is None:
      raise ValueError("PROGRAM reads require byte-backed HBM")
    self._sources: dict[tuple[int, int], AllocationHandle] = {}
    self._next_iova = self.hbm.base_iova
    self._epochs: dict[tuple[int, int], int] = {}
    self._images: dict[tuple[int, int], ProgramImage] = {}
    self._installs: dict[tuple[int, int, int, bytes], _InstallRecord] = {}
    self.stats = {
      "images_registered": 0,
      "installs_armed": 0,
      "chunks_served": 0,
      "text_bytes_served": 0,
      "install_acks": 0,
      "ready": 0,
      "invalidates": 0,
      "rejects": 0,
    }

  def _key3(self, f: dict) -> tuple[int, int, bytes]:
    pkg = f.get("package_hash")
    if isinstance(pkg, memoryview):
      pkg = pkg.tobytes()
    if not isinstance(pkg, bytes) or len(pkg) != 32:
      raise ProgramServiceError("BAD_PACKAGE_HASH", repr(pkg)[:60])
    return (int(f["program_id"]), int(f["version"]), pkg)

  def _key4(self, f: dict) -> tuple[int, int, int, bytes]:
    pid, ver, pkg = self._key3(f)
    return (pid, ver, int(f["program_epoch"]), pkg)

  def _lookup_image(self, program_id: int, version: int, package_hash: bytes) -> ProgramImage | None:
    image = self._images.get((program_id, version))
    return image if image is not None and image.package_hash == package_hash else None

  def register_image(self, image: ProgramImage, *, hbm_iova: int | None = None) -> None:
    key = (image.program_id, image.version)
    existing = self._images.get(key)
    if existing is not None:
      if existing == image and (hbm_iova is None or hbm_iova == self._sources[key].base_address):
        return
      raise ProgramServiceError("IMAGE_IDENTITY_CONFLICT", f"program {image.program_id} v{image.version}")
    base = self._next_iova if hbm_iova is None else hbm_iova
    handle = self.hbm.bind_external(GlobalBinding(f"program:{key[0]}:{key[1]}", base, len(image.text), "r"))
    assert self.hbm.byte_store is not None
    self.hbm.byte_store.seed_hbm(base, image.text, cycle=0)
    self._sources[key] = handle
    self._next_iova = max(self._next_iova, base + len(image.text))
    self._epochs[key] = image.program_epoch
    self._images[key] = image
    self.stats["images_registered"] += 1

  def handle_install(self, msg: BoundaryMessage) -> None:
    """PROGRAM_INSTALL (G2T): arm one install for a registered image. No completion implied."""
    if msg.message_type != "PROGRAM_INSTALL":
      raise ProgramServiceError("BAD_PROGRAM_MESSAGE", msg.message_type)
    image = self._lookup_image(*self._key3(msg.fields))
    if image is None:
      self.stats["rejects"] += 1
      raise ProgramServiceError("UNKNOWN_IMAGE", f"program {msg.fields.get('program_id')} not registered")
    if int(msg.fields["program_epoch"]) != self._epochs[(image.program_id, image.version)]:
      self.stats["rejects"] += 1
      raise ProgramServiceError("EPOCH_MISMATCH", str(msg.fields["program_epoch"]))
    key = self._key4(msg.fields)
    if key in self._installs:
      self.stats["rejects"] += 1
      raise ProgramServiceError("INSTALL_ALREADY_ARMED", f"program {image.program_id}")
    slot = int(msg.fields["slot"])
    if any(rec.slot == slot for rec in self._installs.values()):
      raise ProgramServiceError("SLOT_OCCUPIED", str(slot))
    self._installs[key] = _InstallRecord(image=image, slot=slot)
    self.stats["installs_armed"] += 1

  def handle_read_req(self, msg: BoundaryMessage) -> tuple[BoundaryMessage, bytes]:
    """PROGRAM_READ_REQ (T2G) -> (PROGRAM_READ_RESP, real chunk bytes on the data channel)."""
    if msg.message_type != "PROGRAM_READ_REQ":
      raise ProgramServiceError("BAD_PROGRAM_MESSAGE", msg.message_type)
    image = self._lookup_image(*self._key3(msg.fields))
    if image is None:
      self.stats["rejects"] += 1
      raise ProgramServiceError("UNKNOWN_IMAGE", f"program {msg.fields.get('program_id')} not registered")
    rec = self._installs.get(self._key4(msg.fields))
    if rec is None:
      self.stats["rejects"] += 1
      raise ProgramServiceError("READ_WITHOUT_INSTALL", f"program {image.program_id}")
    if rec.acked or rec.invalidating:
      raise ProgramServiceError("READ_AFTER_INSTALL", f"program {image.program_id}")
    offset = rec.bytes_served  # v5 wire has no offset: the ordered request consumes the next chunk.
    size = min(len(image.text) - offset, self.chunk_bytes)
    if size <= 0:
      raise ProgramServiceError("READ_PAST_IMAGE", f"offset {offset} len {len(image.text)}")
    assert self.hbm.byte_store is not None
    try:
      (segment,) = self.hbm.resolve(self._sources[(image.program_id, image.version)], offset, size, "r")
      chunk = self.hbm.byte_store.read_hbm(segment.address, size)
    except MemoryInvariantError as exc:
      raise ProgramServiceError("PROGRAM_SOURCE_UNAVAILABLE", str(exc)) from exc
    rec.bytes_served += len(chunk)
    self.stats["chunks_served"] += 1
    self.stats["text_bytes_served"] += len(chunk)
    resp = BoundaryMessage(
      message_type="PROGRAM_READ_RESP",
      channel="PROGRAM",
      direction="G2T",
      fields={**msg.fields, "hbm_chunk_len": len(chunk)},
      message_seq=0,
    )
    _ = resp.pack_wire()
    return resp, chunk

  def handle_install_ack(self, msg: BoundaryMessage) -> None:
    """PROGRAM_INSTALL_ACK (T2G): the ONLY install completion; verify len + measured hash."""
    if msg.message_type != "PROGRAM_INSTALL_ACK":
      raise ProgramServiceError("BAD_PROGRAM_MESSAGE", msg.message_type)
    rec = self._installs.get(self._key4(msg.fields))
    if rec is None:
      self.stats["rejects"] += 1
      raise ProgramServiceError("ACK_WITHOUT_INSTALL", str(msg.fields.get("program_id")))
    if rec.acked:
      self.stats["rejects"] += 1
      raise ProgramServiceError("DUPLICATE_INSTALL_ACK", f"program {rec.image.program_id}")
    installed_len = int(msg.fields["installed_len"])
    measured = msg.fields.get("measured_hash")
    if isinstance(measured, memoryview):
      measured = measured.tobytes()
    if installed_len != len(rec.image.text):
      raise ProgramServiceError("INSTALL_LEN_MISMATCH", f"{installed_len} != {len(rec.image.text)}")
    if measured != hashlib.sha256(rec.image.text).digest():
      raise ProgramServiceError("INSTALL_HASH_MISMATCH", f"program {rec.image.program_id}")
    if rec.bytes_served != installed_len or rec.invalidating:
      raise ProgramServiceError("INCOMPLETE_INSTALL", f"served {rec.bytes_served} bytes")
    rec.acked = True
    self.stats["install_acks"] += 1

  def handle_ready(self, msg: BoundaryMessage) -> int:
    """PROGRAM_READY (G2T): authorize the Tile-local handle after the verified ACK."""
    if msg.message_type != "PROGRAM_READY":
      raise ProgramServiceError("BAD_PROGRAM_MESSAGE", msg.message_type)
    rec = self._installs.get(self._key4(msg.fields))
    if rec is None or not rec.acked or rec.invalidating:
      self.stats["rejects"] += 1
      raise ProgramServiceError("READY_WITHOUT_ACK", str(msg.fields.get("program_id")))
    if rec.ready_handle is not None:
      raise ProgramServiceError("DUPLICATE_READY", str(rec.ready_handle))
    rec.ready_handle = int(msg.fields["local_handle"])
    self.stats["ready"] += 1
    return rec.ready_handle

  def handle_invalidate(self, msg: BoundaryMessage) -> None:
    """Serialize invalidation. Keep the occupied record until the real Tile ACK."""
    if msg.message_type != "PROGRAM_INVALIDATE":
      raise ProgramServiceError("BAD_PROGRAM_MESSAGE", msg.message_type)
    rec = self._installs.get(self._key4(msg.fields))
    if rec is None:
      self.stats["rejects"] += 1
      raise ProgramServiceError("INVALIDATE_WITHOUT_INSTALL", str(msg.fields.get("program_id")))
    if rec.invalidating:
      raise ProgramServiceError("DUPLICATE_INVALIDATE", str(rec.slot))
    rec.invalidating = True

  def handle_invalidate_ack(self, msg: BoundaryMessage) -> None:
    key = self._key4(msg.fields)
    rec = self._installs.get(key)
    if msg.message_type != "INVALIDATE_ACK" or rec is None or not rec.invalidating:
      raise ProgramServiceError("UNEXPECTED_INVALIDATE_ACK")
    self._drop_record(key)
    self.stats["invalidates"] += 1

  def _drop_record(self, key: tuple[int, int, int, bytes]) -> None:
    del self._installs[key]
    self._epochs[(key[0], key[1])] = key[2] + 1

  def abort_install(self, msg: BoundaryMessage) -> None:
    key = self._key4(msg.fields)
    rec = self._installs.get(key)
    if rec is None or rec.acked:
      raise ProgramServiceError("NO_PENDING_INSTALL")
    self._drop_record(key)

  def reset(self) -> None:
    """Drop Tile transaction state; preserve immutable host registrations and uploaded HBM bytes."""
    for key in list(self._installs):
      self._drop_record(key)

  def install_complete(self, program_id: int, version: int, package_hash: bytes) -> bool:
    img = self._lookup_image(program_id, version, package_hash)
    if img is None:
      return False
    rec = self._installs.get((program_id, version, self._epochs[(program_id, version)], package_hash))
    return rec is not None and rec.acked and not rec.invalidating

  def snapshot(self) -> dict:
    return {
      "stats": dict(self.stats),
      "installs": {
        f"p{pid}:v{version}:e{epoch}": {
          "slot": rec.slot,
          "epoch": epoch,
          "package_hash": rec.image.package_hash.hex(),
          "invalidating": rec.invalidating,
          "acked": rec.acked,
          "ready_handle": rec.ready_handle,
          "bytes_served": rec.bytes_served,
          "image_bytes": len(rec.image.text),
        }
        for (pid, version, epoch, _hash), rec in self._installs.items()
      },
    }
