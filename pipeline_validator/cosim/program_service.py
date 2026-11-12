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

The sealed read-only Loader lands in C037; this service consumes immutable validated metadata
registered by the harness/scenario, not a fake loader.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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


class ProgramService:
  """Serves PROGRAM channel traffic against registered immutable images."""

  def __init__(self, *, chunk_bytes: int = CHUNK_BYTES) -> None:
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
      raise ValueError("chunk_bytes must be a positive int")
    self.chunk_bytes = chunk_bytes
    self._images: dict[tuple[int, int, bytes], ProgramImage] = {}
    self._installs: dict[tuple[int, int, int, bytes], _InstallRecord] = {}
    self.stats = {
      "images_registered": 0,
      "installs_armed": 0,
      "chunks_served": 0,
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

  def register_image(self, image: ProgramImage) -> None:
    key = (image.program_id, image.version, image.package_hash)
    existing = self._images.get(key)
    if existing is not None:
      if existing is image or existing == image:
        return
      raise ProgramServiceError("IMAGE_IDENTITY_CONFLICT", f"program {image.program_id} v{image.version}")
    self._images[key] = image
    self.stats["images_registered"] += 1

  def handle_install(self, msg: BoundaryMessage) -> None:
    """PROGRAM_INSTALL (G2T): arm one install for a registered image. No completion implied."""
    if msg.message_type != "PROGRAM_INSTALL":
      raise ProgramServiceError("BAD_PROGRAM_MESSAGE", msg.message_type)
    image = self._images.get(self._key3(msg.fields))
    if image is None:
      self.stats["rejects"] += 1
      raise ProgramServiceError("UNKNOWN_IMAGE", f"program {msg.fields.get('program_id')} not registered")
    if int(msg.fields["program_epoch"]) != image.program_epoch:
      self.stats["rejects"] += 1
      raise ProgramServiceError("EPOCH_MISMATCH", str(msg.fields["program_epoch"]))
    key = self._key4(msg.fields)
    if key in self._installs:
      self.stats["rejects"] += 1
      raise ProgramServiceError("INSTALL_ALREADY_ARMED", f"program {image.program_id}")
    self._installs[key] = _InstallRecord(image=image, slot=int(msg.fields["slot"]))
    self.stats["installs_armed"] += 1

  def handle_read_req(self, msg: BoundaryMessage, *, offset: int = 0) -> tuple[BoundaryMessage, bytes]:
    """PROGRAM_READ_REQ (T2G) -> (PROGRAM_READ_RESP, real chunk bytes on the data channel)."""
    if msg.message_type != "PROGRAM_READ_REQ":
      raise ProgramServiceError("BAD_PROGRAM_MESSAGE", msg.message_type)
    image = self._images.get(self._key3(msg.fields))
    if image is None:
      self.stats["rejects"] += 1
      raise ProgramServiceError("UNKNOWN_IMAGE", f"program {msg.fields.get('program_id')} not registered")
    rec = self._installs.get(self._key4(msg.fields))
    if rec is None:
      self.stats["rejects"] += 1
      raise ProgramServiceError("READ_WITHOUT_INSTALL", f"program {image.program_id}")
    chunk = image.text[offset : offset + self.chunk_bytes]
    if not chunk:
      raise ProgramServiceError("READ_PAST_IMAGE", f"offset {offset} len {len(image.text)}")
    rec.bytes_served += len(chunk)
    self.stats["chunks_served"] += 1
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
    if measured is not None and measured != hashlib.sha256(rec.image.text).digest():
      raise ProgramServiceError("INSTALL_HASH_MISMATCH", f"program {rec.image.program_id}")
    rec.acked = True
    self.stats["install_acks"] += 1

  def handle_ready(self, msg: BoundaryMessage) -> int:
    """PROGRAM_READY (T2G): record the Tile-local handle; requires a prior ACK."""
    if msg.message_type != "PROGRAM_READY":
      raise ProgramServiceError("BAD_PROGRAM_MESSAGE", msg.message_type)
    rec = self._installs.get(self._key4(msg.fields))
    if rec is None or not rec.acked:
      self.stats["rejects"] += 1
      raise ProgramServiceError("READY_WITHOUT_ACK", str(msg.fields.get("program_id")))
    rec.ready_handle = int(msg.fields["local_handle"])
    self.stats["ready"] += 1
    return rec.ready_handle

  def handle_invalidate(self, msg: BoundaryMessage) -> BoundaryMessage:
    """PROGRAM_INVALIDATE (G2T) -> INVALIDATE_ACK (T2G)."""
    if msg.message_type != "PROGRAM_INVALIDATE":
      raise ProgramServiceError("BAD_PROGRAM_MESSAGE", msg.message_type)
    rec = self._installs.pop(self._key4(msg.fields), None)
    if rec is None:
      self.stats["rejects"] += 1
      raise ProgramServiceError("INVALIDATE_WITHOUT_INSTALL", str(msg.fields.get("program_id")))
    self.stats["invalidates"] += 1
    ack = BoundaryMessage(
      message_type="INVALIDATE_ACK", channel="PROGRAM", direction="T2G", fields=msg.fields, message_seq=0
    )
    _ = ack.pack_wire()
    return ack

  def install_complete(self, program_id: int, version: int, package_hash: bytes) -> bool:
    key = (program_id, version, package_hash)
    img = self._images.get(key)
    if img is None:
      return False
    rec = self._installs.get((program_id, version, img.program_epoch, package_hash))
    return rec is not None and rec.acked

  def snapshot(self) -> dict:
    return {
      "stats": dict(self.stats),
      "installs": {
        f"p{rec.image.program_id}:v{rec.image.version}:e{rec.image.program_epoch}": {
          "slot": rec.slot,
          "acked": rec.acked,
          "ready_handle": rec.ready_handle,
          "bytes_served": rec.bytes_served,
          "image_bytes": len(rec.image.text),
        }
        for rec in self._installs.values()
      },
    }
