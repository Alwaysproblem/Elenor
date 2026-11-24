"""Program Residency Manager (Architecture doc 15, review P0-2; C039 / W5-04).

Two deliberately distinct faces share this module:

- **Analytic timing model** (legacy Python backend): ``register`` / ``ensure_resident`` charge a
  closed-form cold-launch latency and mark a Tile resident synchronously.  There are no bytes, no
  pending install, no reference ledger and no ACK; it models the ACCEPT cost only, is never
  evidence of residency, and nothing on the real path reads it.
- **ACK-driven residency** (C039): ``register_image`` / ``acquire`` / ``handle`` / ``release`` /
  ``begin_invalidate``.  A Tile is ``RESIDENT`` only after its ``PROGRAM_INSTALL_ACK`` was verified
  against the registered image (installed length + sha256 over the real text); one install per
  ``(identity, Tile)`` is the shared completion point for every waiter; the reference ledger lives
  here and nowhere else; a slot and its bytes return to the ledger only when the INVALIDATE_ACK is
  collected, a pending install faults, or a reset voids the Tile.

Real-path identity is ``(program_id, version, package_hash)`` — the sealed package identity, never
the abstract IR digest.  The wire ``slot`` of ``PROGRAM_INSTALL`` is the pre-linked slot ADDRESS
(``linked_slot.text_base``); the bitmap index reported by ``snapshot()``/``slot_bitmap()`` is the
derived ``(text_base - origin) // text_bytes_per_slot`` — distinct quantities.  Epoch advances per
``(identity, Tile)`` install transaction, mirroring ``ProgramService._drop_record``: a dropped
record (INVALIDATE_ACK, abort, reset) makes the next install on that Tile carry ``epoch + 1``, so a
stale message of a retired transaction can never complete the new one.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from ..package.binding_ir import PackageError, patch_allowed

if TYPE_CHECKING:
  from ..cosim.program_service import ProgramService
  from ..cosim.tile_port import BoundaryMessage


def _cosim():
  """Lazy: ``cosim`` imports this package, so the dependency is taken only at call time."""
  from ..cosim import program_service as _service_mod
  from ..cosim.tile_port import BoundaryMessage

  return BoundaryMessage, _service_mod


class ResidencyState(IntEnum):
  """Per-(identity, Tile) residency phase (design W5 §3.2)."""

  ABSENT = 0  # nothing installed here (the analytic-era HBM_ONLY)
  PENDING_INSTALL = 1  # install armed, bytes not yet ACK-verified (the analytic-era FETCHING)
  RESIDENT = 2  # install ACK verified and PROGRAM_READY published (was TILE_RESIDENT)
  INVALIDATING = 3  # privileged invalidation in flight: no new hits or references


class ResidencyError(RuntimeError):
  """A coded refusal or protocol violation on the real residency path."""

  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


def _positive(name: str, value: Any) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < 1:
    raise ResidencyError("BAD_ARGUMENT", f"{name} must be a positive int, got {value!r}")
  return value


# Wire ``uint`` fields pack one 64-bit little-endian word each (wire.py ``MASK64``).
_UINT64_MAX = (1 << 64) - 1

@dataclass(frozen=True)
class ProgramIdentity:
  """The sealed residency key: ``program_id`` + ``version`` + ``package_hash`` (RES-02 / CMP-18)."""

  program_id: int
  version: int
  package_hash: bytes

  def __post_init__(self) -> None:
    _positive("program_id", self.program_id)
    _positive("version", self.version)
    package_hash = self.package_hash
    if isinstance(package_hash, memoryview):
      package_hash = package_hash.tobytes()
      object.__setattr__(self, "package_hash", package_hash)
    if not isinstance(package_hash, bytes) or len(package_hash) != 32:
      raise ResidencyError("BAD_PACKAGE_HASH", "package_hash must be 32 bytes")

  @property
  def key(self) -> tuple[int, int]:
    return (self.program_id, self.version)

  def wire_fields(self) -> dict[str, Any]:
    return {
      "program_id": self.program_id,
      "version": self.version,
      "package_hash": self.package_hash,
    }

  def __repr__(self) -> str:
    return f"p{self.program_id}:v{self.version}:{self.package_hash[:4].hex()}…"

  @classmethod
  def of(cls, image: Any) -> ProgramIdentity:
    """Accept a normalized identity or a ProgramImage-like object."""
    if isinstance(image, cls):
      return image
    try:
      return cls(int(image.program_id), int(image.version), image.package_hash)
    except AttributeError as exc:
      raise ResidencyError("BAD_IMAGE", "an image needs program_id, version and package_hash") from exc


@dataclass(frozen=True)
class ResidencyStorage:
  """One Tile's program-SRAM geometry: a cumulative byte budget over fixed pre-linked slots.

  ``slot k`` owns ``[origin_bytes + k*slot_bytes, origin_bytes + (k+1)*slot_bytes)``.  The byte
  budget is accounted per installed image and is intentionally independent of
  ``slot_count * slot_bytes``: the cumulative ledger and the slot bitmap answer different
  questions (RES-14, §6 multi fixed pre-linked slot table).
  """

  capacity_bytes: int
  slot_bytes: int
  slot_count: int
  origin_bytes: int = 0

  def __post_init__(self) -> None:
    for name in ("capacity_bytes", "slot_bytes", "slot_count"):
      _positive(name, getattr(self, name))
    if (
      isinstance(self.origin_bytes, bool)
      or not isinstance(self.origin_bytes, int)
      or self.origin_bytes < 0
    ):
      raise ResidencyError("BAD_STORAGE", "origin_bytes must be a non-negative int")

  @classmethod
  def from_slot_freeze(cls, freeze: Mapping[str, Any]) -> ResidencyStorage:
    """Inject the measured ``cn_slot_freeze/v1`` geometry (C038); nothing is hardcoded here."""
    try:
      geometry = freeze["slot_geometry"]
      itcm = int(geometry["itcm_bytes"])
      slot_bytes = int(geometry["text_bytes_per_slot"])
      slot_count = int(geometry["program_slots"])
      origin = int(geometry.get("itcm_origin", 0))
    except (KeyError, TypeError, AttributeError) as exc:
      raise ResidencyError("SLOT_GEOMETRY_MISSING", repr(exc)) from exc
    if slot_count * slot_bytes != itcm:
      raise ResidencyError(
        "SLOT_GEOMETRY_MISMATCH",
        f"program_slots*text_bytes_per_slot != itcm_bytes ({itcm})",
      )
    return cls(
      capacity_bytes=itcm, slot_bytes=slot_bytes, slot_count=slot_count, origin_bytes=origin
    )


@dataclass(frozen=True)
class ResidentHandle:
  """What a warm hit hands to the consumer: the Tile-local handle bound to an epoch."""

  identity: ProgramIdentity
  slot: int  # bitmap index
  slot_address: int  # the pre-linked address carried by the wire `slot` field
  local_handle: int
  epoch: int


@dataclass(frozen=True)
class ResidencyFault:
  """A latched fail-stop fault (VB-11).

  ``affected`` lists the waiter holders whose pending reference died with the aborted install.
  ``identity`` is ``None`` for channel-desync faults that name no single program."""

  tile_id: int
  identity: ProgramIdentity | None
  code: str
  detail: str = ""
  affected: tuple = ()


@dataclass(frozen=True)
class ResidencyDecision:
  """The COMMIT serialization point's answer for one Task.

  * ``kind == "hit"``     warm: a reference was added and ``handle`` is live; zero Tile-external
                          program bytes are implied.
  * ``kind == "pending"`` cold: the Task joined the ONE shared install; ``completion_id`` is the
                          same for every sharer; only the starter carries the
                          ``PROGRAM_INSTALL`` in ``outbox``.
  * ``kind == "wait"``    bounded wait: zero references and zero capacity were taken; ``reason``
                          names the hold and ``blocked_by`` the identities whose explicit
                          invalidation is the exit.
  * ``kind == "refused"`` permanent refusal: ``reason`` names it.
  """

  identity: ProgramIdentity
  tile_id: int
  kind: str
  reason: str | None = None
  blocked_by: tuple[ProgramIdentity, ...] = ()
  completion_id: int | None = None
  handle: ResidentHandle | None = None
  outbox: tuple[BoundaryMessage, ...] = ()


@dataclass(frozen=True)
class ProgramStep:
  """Group-side output produced while processing one Tile message or ledger transition."""

  outbox: tuple[BoundaryMessage, ...] = ()
  data: bytes | None = None


@dataclass(frozen=True)
class ImageInfo:
  """Registered image metadata — the installer's trusted setup values, never invented."""

  identity: ProgramIdentity
  slot: int  # bitmap index
  slot_address: int  # wire slot == pre-linked text_base
  text_base: int
  text_bytes: int
  text_sha256: bytes
  patch_mask: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ResidencyView:
  """A read-only projection of one live ``(identity, Tile)`` record."""

  identity: ProgramIdentity
  tile_id: int
  state: ResidencyState
  epoch: int
  slot: int
  slot_address: int
  installed: bool
  local_handle: int | None
  completion_id: int
  refs: int
  bytes_served: int
  invalidate_sent: bool


@dataclass
class _Image:
  """A registered sealed image and its fixed placement (immutable except the epoch ledger)."""

  source: Any  # the ProgramImage the channel endpoint also serves
  identity: ProgramIdentity
  text: bytes
  text_sha256: bytes
  slot: int
  text_base: int
  patch_mask: tuple[tuple[str, str], ...]
  program_epoch: int  # epoch the image was registered at (the install counter's start)
  epochs: dict[int, int] = field(default_factory=dict)  # tile_id -> next install transaction epoch

  @property
  def info(self) -> ImageInfo:
    return ImageInfo(
      identity=self.identity,
      slot=self.slot,
      slot_address=self.text_base,
      text_base=self.text_base,
      text_bytes=len(self.text),
      text_sha256=self.text_sha256,
      patch_mask=self.patch_mask,
    )


@dataclass
class _Residency:
  """One live residency record; it exists exactly while the Tile channel holds an install record."""

  image: _Image
  tile_id: int
  epoch: int
  state: ResidencyState
  completion_id: int
  installed: bool = False
  local_handle: int | None = None
  holders: set = field(default_factory=set)
  anonymous_refs: int = 0
  bytes_served: int = 0
  invalidate_sent: bool = False
  arm_msg: Any = None
  patches: list = field(default_factory=list)

  @property
  def refs(self) -> int:
    return len(self.holders) + self.anonymous_refs


@dataclass
class ProgramTableEntry:
  """Analytic-model entry (legacy timing state: no bytes, no ACK, no references)."""

  program_id: int
  version: int
  program_hash: int
  epoch: int = 0
  hbm_iova: int = 0
  hbm_bytes: int = 0
  # per-tile residency state
  tile_states: dict[int, ResidencyState] = None  # type: ignore[assignment]
  tile_epochs: dict[int, int] = None  # type: ignore[assignment]

  def __post_init__(self) -> None:
    if self.tile_states is None:
      self.tile_states = {}
    if self.tile_epochs is None:
      self.tile_epochs = {}


class ProgramResidencyManager:
  """Group residency authority.

  Analytic model (unchanged): ``register`` marks identity/size; ``ensure_resident`` synchronously
  marks a Tile resident and returns a closed-form latency; ``cold_load_cycles`` accumulates it.

  Real path (C039): the manager mirrors the Tile channel's install records one-for-one, is the
  single authority for cold/hot judgement, references, the cumulative capacity ledger and the
  slot bitmap, and publishes ``PROGRAM_READY`` only after a verified ``PROGRAM_INSTALL_ACK``.
  """

  def __init__(self, cfg, *, storage: ResidencyStorage | None = None) -> None:
    self.cfg = cfg
    # --- analytic timing model (legacy) ---
    self._entries: dict[int, ProgramTableEntry] = {}
    self.cold_load_cycles: int = 0
    # --- real, ACK-driven residency (C039) ---
    self._storage = storage
    self._images: dict[tuple[int, int], _Image] = {}
    self._services: dict[int, Any] = {}
    self._records: dict[tuple[int, int, int], _Residency] = {}
    self._tile_counters: dict[int, Counter] = {}
    self._counters: Counter = Counter()
    self._waits: Counter = Counter()
    self._fault: ResidencyFault | None = None
    self._completion_seq = 0

  # ------------------------------------------------------------------
  # real path: registration and plumbing
  # ------------------------------------------------------------------
  def configure_storage(self, storage: ResidencyStorage) -> None:
    """Set the slot geometry on a live manager (e.g. ``TileGroup.program_table``); allowed until
    the first ``register_image`` so a built-in manager can be configured before registration."""
    if not isinstance(storage, ResidencyStorage):
      raise ResidencyError("BAD_STORAGE", "a ResidencyStorage is required")
    if self._images:
      raise ResidencyError("STORAGE_LOCKED", "storage is frozen once images are registered")
    self._storage = storage

  def attach(self, tile_id: int, service: ProgramService) -> None:
    """Bind one Tile's PROGRAM-channel endpoint.  The manager is then its only driver so the
    per-(identity, Tile) install epoch stays in lockstep with ``ProgramService._drop_record``."""
    if isinstance(tile_id, bool) or not isinstance(tile_id, int) or tile_id < 0:
      raise ResidencyError("BAD_TILE_ID", repr(tile_id))
    if tile_id in self._services:
      if self._services[tile_id] is service:
        return
      raise ResidencyError("TILE_SERVICE_REBOUND", f"tile {tile_id}")
    if any(existing is service for existing in self._services.values()):
      raise ResidencyError("SERVICE_ALREADY_ATTACHED", f"tile {tile_id}")
    for image in self._images.values():
      service.register_image(image.source)
      image.epochs.setdefault(tile_id, image.program_epoch)
    self._services[tile_id] = service
    self._tile_counters[tile_id] = Counter()

  def register_image(
    self,
    image: Any,
    *,
    text_base: int | None = None,
    slot: int | None = None,
    patch_mask: Any = (),
  ) -> ProgramIdentity:
    """Register the sealed image's residency identity and its fixed pre-linked placement.

    ``slot`` on the wire is the pre-linked slot ADDRESS: it must equal ``text_base`` and be
    aligned to ``slot_bytes``.  Re-registering an identical image is idempotent; the same
    ``(program_id, version)`` with any other hash, bytes, placement or mask is refused as an
    identity ambiguity, never silently re-bound.
    """
    storage = self._require_storage()
    _, service_mod = _cosim()
    if not isinstance(image, service_mod.ProgramImage):
      raise ResidencyError("BAD_IMAGE", "a C024 ProgramImage is required")
    identity = ProgramIdentity.of(image)
    text, epoch0 = image.text, image.program_epoch
    if isinstance(text_base, bool) or (text_base is not None and not isinstance(text_base, int)):
      raise ResidencyError("BAD_TEXT_BASE", repr(text_base))
    if isinstance(slot, bool) or (slot is not None and not isinstance(slot, int)):
      raise ResidencyError("BAD_ARGUMENT", repr(slot))
    if text_base is None:
      if slot is None:
        raise ResidencyError(
          "PLACEMENT_UNDECLARED", "a slot index or a pre-linked text_base is required"
        )
      base = storage.origin_bytes + slot * storage.slot_bytes
    else:
      base = text_base
    rel = base - storage.origin_bytes
    if rel < 0 or rel % storage.slot_bytes != 0:
      raise ResidencyError("TEXT_BASE_UNALIGNED", f"{base:#x} is not slot-aligned")
    slot_index = rel // storage.slot_bytes
    if slot is not None and slot_index != slot:
      raise ResidencyError(
        "SLOT_ADDRESS_MISMATCH",
        f"slot {slot} owns base {storage.origin_bytes + slot * storage.slot_bytes:#x}, not {base:#x}",
      )
    if slot_index >= storage.slot_count:
      raise ResidencyError("SLOT_OUT_OF_RANGE", f"slot {slot_index} >= {storage.slot_count}")
    if len(text) > storage.slot_bytes:
      raise ResidencyError(
        "PROGRAM_EXCEEDS_SLOT", f"{len(text)} B > {storage.slot_bytes} B slot"
      )
    if len(text) > storage.capacity_bytes:
      raise ResidencyError(
        "PROGRAM_EXCEEDS_CAPACITY", f"{len(text)} B > {storage.capacity_bytes} B"
      )
    mask = self._normalize_mask(patch_mask)
    key = identity.key
    existing = self._images.get(key)
    if existing is not None:
      if (
        existing.identity != identity
        or existing.text != text
        or existing.text_base != base
        or existing.patch_mask != mask
        or existing.program_epoch != epoch0
      ):
        raise ResidencyError(
          "IDENTITY_CONFLICT", f"program {identity.program_id} v{identity.version} is already bound"
        )
      return existing.identity
    record = _Image(
      source=image,
      identity=identity,
      text=text,
      text_sha256=hashlib.sha256(text).digest(),
      slot=slot_index,
      text_base=base,
      patch_mask=mask,
      program_epoch=epoch0,
    )
    for service in self._services.values():
      service.register_image(image)
    self._images[key] = record
    for tile_id in self._services:
      record.epochs[tile_id] = epoch0
    return identity

  @staticmethod
  def _normalize_mask(mask: Any) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    for item in mask or ():
      try:
        field_name, layer = item["field"], item["layer"]
      except (TypeError, KeyError) as exc:
        raise ResidencyError("BAD_PATCH_MASK", repr(item)[:80]) from exc
      if not isinstance(field_name, str) or not isinstance(layer, str):
        raise ResidencyError("BAD_PATCH_MASK", repr(item)[:80])
      entries.append((field_name, layer))
    return tuple(sorted(entries))

  def image_info(self, identity: Any) -> ImageInfo:
    """Registered metadata for the sealed image — the installer's trusted setup."""
    return self._image_of(identity).info

  # ------------------------------------------------------------------
  # real path: the COMMIT serialization point
  # ------------------------------------------------------------------
  def acquire(self, identity: Any, tile_id: int, holder: Hashable | None = None) -> ResidencyDecision:
    """Bind one Task's program-space reference (RES-11/14/19).

    A warm hit or a shared pending only adds a reference; the starter of a cold miss returns the
    ``PROGRAM_INSTALL`` header in ``outbox``.  A ``wait`` takes no reference, no slot and no bytes,
    and ``blocked_by`` names the identities whose explicit invalidation is the exit — there is no
    permanent wait: anything that can never fit was refused at ``register_image``.
    """
    img = self._image_of(identity)
    self._tile(tile_id)
    self._check_holder(holder)
    if self._fault is not None:
      self._counters["refusals"] += 1
      return ResidencyDecision(img.identity, tile_id, "refused", reason="RESIDENCY_FAULTED")
    key = (*img.identity.key, tile_id)
    rec = self._records.get(key)
    if rec is not None and rec.state is ResidencyState.RESIDENT:
      service = self._services[tile_id]
      if not service.install_complete(*self._key3(rec.image.identity)):
        raise self._desync(
          tile_id, img.identity, "RESIDENT on the table is not installed on the channel"
        )
      self._reference(rec, holder)
      self._counters["warm_hits"] += 1
      assert rec.local_handle is not None
      handle = ResidentHandle(
        identity=img.identity,
        slot=img.slot,
        slot_address=img.text_base,
        local_handle=rec.local_handle,
        epoch=rec.epoch,
      )
      return ResidencyDecision(
        img.identity, tile_id, "hit", completion_id=rec.completion_id, handle=handle
      )
    if rec is not None and rec.state is ResidencyState.PENDING_INSTALL:
      self._reference(rec, holder)
      self._counters["shared_installs"] += 1
      return ResidencyDecision(img.identity, tile_id, "pending", completion_id=rec.completion_id)
    if rec is not None:  # INVALIDATING: the serialization point already closed the door
      return self._wait(img.identity, tile_id, "WAIT_PROGRAM_INVALIDATING", (img.identity,))
    owner = self._slot_owner(tile_id, img.slot)
    if owner is not None:
      return self._wait(img.identity, tile_id, "WAIT_SLOT_IN_USE", (owner.image.identity,))
    if self._used_bytes(tile_id) + len(img.text) > self._require_storage().capacity_bytes:
      return self._wait(
        img.identity,
        tile_id,
        "WAIT_PROGRAM_CAPACITY",
        tuple(other.image.identity for other in self._tile_records(tile_id)),
      )
    rec = self._arm(img, tile_id)
    self._reference(rec, holder)
    self._counters["cold_misses"] += 1
    return ResidencyDecision(
      img.identity, tile_id, "pending", completion_id=rec.completion_id, outbox=(rec.arm_msg,)
    )

  def release(self, identity: Any, tile_id: int, holder: Hashable | None = None) -> ProgramStep:
    """TASK_COMPLETION bookkeeping: drop one reference only.

    Resident text and capacity are intentionally NOT returned (RES-10/14): the next acquire of the
    same identity is a warm hit.  When an INVALIDATING record's last reference retires this emits
    the ``PROGRAM_INVALIDATE``.
    """
    img = self._image_of(identity)
    if self._fault is not None:
      raise ResidencyError("RESIDENCY_FAULTED", self._fault.code)
    self._tile(tile_id)
    rec = self._records.get((*img.identity.key, tile_id))
    if rec is None:
      raise ResidencyError("REF_NOT_HELD", f"nothing resident for {img.identity}")
    if holder is None:
      if rec.anonymous_refs <= 0:
        raise ResidencyError("REF_UNDERFLOW", "no reference is held")
      rec.anonymous_refs -= 1
    else:
      try:
        known = holder in rec.holders
      except TypeError as exc:
        raise ResidencyError("BAD_HOLDER", f"unhashable holder {holder!r}") from exc
      if not known:
        raise ResidencyError("REF_NOT_HELD", f"holder {holder!r} does not reference the program")
      rec.holders.remove(holder)
    self._counters["references_released"] += 1
    return self._drain_invalidate(rec)

  def begin_invalidate(self, identity: Any, tile_id: int) -> ProgramStep:
    """Privileged invalidation of one ``(identity, Tile)``.

    The record turns INVALIDATING atomically at this serialization point: no new reference, no
    warm hit, even when the install is still pulling (its completion is then consumed silently,
    never published as READY).  The ``PROGRAM_INVALIDATE`` wire is emitted once the record is
    installed and every reference has retired; the slot and bytes return only on INVALIDATE_ACK.
    A single-Tile invalidation leaves other Tiles' residency untouched.
    """
    img = self._image_of(identity)
    self._tile(tile_id)
    if self._fault is not None:
      raise ResidencyError("RESIDENCY_FAULTED", self._fault.code)
    rec = self._records.get((*img.identity.key, tile_id))
    if rec is None:
      return ProgramStep()
    if rec.state is not ResidencyState.INVALIDATING:
      rec.state = ResidencyState.INVALIDATING
      self._counters["invalidates_requested"] += 1
    return self._drain_invalidate(rec)

  # ------------------------------------------------------------------
  # real path: Tile -> Group traffic
  # ------------------------------------------------------------------
  def handle(
    self, tile_id: int, msg: BoundaryMessage, *, local_handle: int | None = None
  ) -> ProgramStep:
    """Feed one T2G message on the Tile's PROGRAM channel.

    ``local_handle`` is the Tile's ACK sideband and is consumed only on ``PROGRAM_INSTALL_ACK``;
    it is passed through verified, never invented.
    """
    self._tile(tile_id)
    if msg.direction != "T2G" or msg.channel != "PROGRAM":
      raise ResidencyError(
        "BAD_PROGRAM_MESSAGE", f"{msg.message_type} is not Tile->Group PROGRAM traffic"
      )
    if self._fault is not None:
      raise ResidencyError("RESIDENCY_FAULTED", self._fault.code)
    if msg.message_type == "PROGRAM_READ_REQ":
      return self._read_req(tile_id, msg)
    if msg.message_type == "PROGRAM_INSTALL_ACK":
      return self._install_ack(tile_id, msg, local_handle)
    if msg.message_type == "INVALIDATE_ACK":
      return self._invalidate_ack(tile_id, msg)
    raise ResidencyError("BAD_PROGRAM_MESSAGE", f"unexpected {msg.message_type}")

  def _read_req(self, tile_id: int, msg: BoundaryMessage) -> ProgramStep:
    rec = self._record_of(tile_id, msg.fields, "STALE_READ_REQUEST")
    if rec.installed:
      # A RESIDENT identity never fetches again: a read here is a wire protocol violation, not a
      # fault that aborts the install — the record stays untouched.
      raise ResidencyError("READ_AFTER_INSTALL", f"program {rec.image.identity.program_id}")
    _, service_mod = _cosim()
    try:
      resp, chunk = self._services[tile_id].handle_read_req(msg)
    except service_mod.ProgramServiceError as exc:
      raise self._fail_install(rec, exc.code, exc.detail) from exc
    except Exception as exc:
      raise self._fail_install(rec, "READ_FAILED", str(exc)) from exc
    rec.bytes_served += len(chunk)
    self._tile_counters[tile_id]["read_requests"] += 1
    self._tile_counters[tile_id]["external_text_bytes"] += len(chunk)
    return ProgramStep(outbox=(resp,), data=chunk)

  def _install_ack(
    self, tile_id: int, msg: BoundaryMessage, local_handle: int | None
  ) -> ProgramStep:
    rec = self._record_of(tile_id, msg.fields, "STALE_INSTALL_ACK")
    if rec.installed:
      raise ResidencyError("DUPLICATE_INSTALL_ACK", f"program {rec.image.identity.program_id}")
    try:
      installed_len = int(msg.fields["installed_len"])
    except (KeyError, TypeError, ValueError) as exc:
      raise self._fail_install(rec, "BAD_PROGRAM_MESSAGE", "installed_len is not an int") from exc
    measured = msg.fields.get("measured_hash")
    if isinstance(measured, memoryview):
      measured = measured.tobytes()
    if installed_len != len(rec.image.text):
      raise self._fail_install(
        rec, "INSTALL_LEN_MISMATCH", f"{installed_len} != {len(rec.image.text)}"
      )
    if measured != rec.image.text_sha256:
      raise self._fail_install(rec, "INSTALL_HASH_MISMATCH", f"program {rec.image.identity}")
    if rec.bytes_served != installed_len:
      raise self._fail_install(
        rec, "INCOMPLETE_INSTALL", f"served {rec.bytes_served} B of {installed_len} B"
      )
    if (
      isinstance(local_handle, bool)
      or not isinstance(local_handle, int)
      or not 0 <= local_handle <= _UINT64_MAX
    ):
      raise self._fail_install(rec, "BAD_LOCAL_HANDLE", repr(local_handle))
    _, service_mod = _cosim()
    try:
      self._services[tile_id].handle_install_ack(msg)
    except service_mod.ProgramServiceError as exc:
      raise self._fail_install(rec, exc.code, exc.detail) from exc
    rec.installed = True
    self._counters["install_acks"] += 1
    if rec.state is ResidencyState.INVALIDATING:
      # Invalidated while the pull was in flight: the completion is consumed silently and never
      # published as PROGRAM_READY — the door was already closed.  The local handle is bound only
      # when the channel accepts READY below, so the manager mirrors ``ready_handle`` exactly.
      return self._drain_invalidate(rec)
    ready = self._g2t(
      "PROGRAM_READY",
      {
        **rec.image.identity.wire_fields(),
        "program_epoch": rec.epoch,
        "local_handle": local_handle,
      },
    )
    try:
      self._services[tile_id].handle_ready(ready)
    except service_mod.ProgramServiceError as exc:
      raise self._fail_install(rec, exc.code, exc.detail) from exc
    rec.local_handle = local_handle
    rec.state = ResidencyState.RESIDENT
    self._counters["ready_published"] += 1
    return ProgramStep(outbox=(ready,))

  def _invalidate_ack(self, tile_id: int, msg: BoundaryMessage) -> ProgramStep:
    rec = self._record_of(tile_id, msg.fields, "UNEXPECTED_INVALIDATE_ACK")
    if rec.state is not ResidencyState.INVALIDATING or not rec.invalidate_sent:
      raise ResidencyError("UNEXPECTED_INVALIDATE_ACK", f"program {rec.image.identity.program_id}")
    _, service_mod = _cosim()
    try:
      self._services[tile_id].handle_invalidate_ack(msg)
    except service_mod.ProgramServiceError as exc:
      self._latch(
        ResidencyFault(
          tile_id, rec.image.identity, "SERVICE_DESYNC",
          f"INVALIDATE_ACK rejected by the channel: {exc.code}",
        )
      )
      raise ResidencyError(exc.code, exc.detail) from exc
    self._drop(rec)
    self._counters["invalidate_acks"] += 1
    return ProgramStep()

  # ------------------------------------------------------------------
  # real path: patches, faults, invalidation lifecycle
  # ------------------------------------------------------------------
  def patch(self, identity: Any, tile_id: int, *, field: str, layer: str) -> None:
    """A runtime descriptor patch at commit (RES-07).

    Only a field the sealed ``patch_mask`` declares, at its declared layer, may change at run
    time; anything else is an ``INVALID_DESCRIPTOR`` fault.  Resident text is never patchable:
    a ``text.*`` field is refused even if a package declared it.
    """
    if not isinstance(field, str) or not field or not isinstance(layer, str) or not layer:
      raise ResidencyError("BAD_PATCH_ARG", "field and layer must be non-empty strings")
    img = self._image_of(identity)
    self._tile(tile_id)
    if self._fault is not None:
      raise ResidencyError("RESIDENCY_FAULTED", self._fault.code)
    rec = self._records.get((*img.identity.key, tile_id))
    if rec is None:
      raise ResidencyError(
        "PROGRAM_NOT_RESIDENT", f"program {img.identity.program_id} on tile {tile_id}"
      )
    if rec.state is ResidencyState.INVALIDATING:
      raise ResidencyError(
        "PROGRAM_INVALIDATING", f"program {img.identity.program_id} on tile {tile_id}"
      )
    code = None
    detail = ""
    if field == "text" or field.startswith("text."):
      code, detail = "INVALID_DESCRIPTOR", f"resident text is never patched: {field}"
    else:
      try:
        patch_allowed(
          {"patch_mask": [{"field": f, "layer": ly} for f, ly in rec.image.patch_mask]},
          field,
          layer,
        )
      except PackageError as exc:
        code, detail = "INVALID_DESCRIPTOR", f"{exc.code}: {field}"
    if code is not None:
      self._latch(ResidencyFault(tile_id, img.identity, code, detail))
      raise ResidencyError(code, detail)
    rec.patches.append((field, layer))
    self._counters["descriptor_patches"] += 1

  def fail_install(self, identity: Any, tile_id: int, code: str = "INSTALL_FAULT") -> None:
    """Report a Tile-side installer fault that produced no valid ACK (e.g. an install FSM or the
    C030 port rejecting): the PENDING install dies with no half-resident record."""
    img = self._image_of(identity)
    self._tile(tile_id)
    rec = self._records.get((*img.identity.key, tile_id))
    if rec is None:
      raise ResidencyError("NO_PENDING_INSTALL", f"program {img.identity.program_id}")
    if rec.installed:
      raise ResidencyError("ALREADY_INSTALLED", "an ACK-verified record cannot be aborted")
    self._fail_install(rec, code, "Tile-side install fault")

  # ------------------------------------------------------------------
  # real path: resets and inspection
  # ------------------------------------------------------------------
  def _drop(self, rec: _Residency) -> None:
    """Remove a record; this Tile's channel epoch for the identity advances drop-for-drop."""
    rec.image.epochs[rec.tile_id] = rec.epoch + 1
    del self._records[(*rec.image.identity.key, rec.tile_id)]

  def _drain_invalidate(self, rec: _Residency | None) -> ProgramStep:
    if (
      rec is not None
      and rec.state is ResidencyState.INVALIDATING
      and rec.installed
      and not rec.invalidate_sent
      and rec.refs == 0
    ):
      msg = self._g2t(
        "PROGRAM_INVALIDATE",
        {**rec.image.identity.wire_fields(), "program_epoch": rec.epoch},
      )
      _, service_mod = _cosim()
      try:
        self._services[rec.tile_id].handle_invalidate(msg)
      except service_mod.ProgramServiceError as exc:
        raise self._desync(
          rec.tile_id,
          rec.image.identity,
          f"INVALIDATE rejected by the channel: {exc.code}",
        ) from exc
      rec.invalidate_sent = True
      self._counters["invalidates_sent"] += 1
      return ProgramStep(outbox=(msg,))
    return ProgramStep()

  def _fail_install(self, rec: _Residency, code: str, detail: str) -> ResidencyError:
    """Abort a PENDING install: drop the channel record, return slot and bytes, release the
    waiters' references and latch the fault — nothing half-resident can ever be hit."""
    _, service_mod = _cosim()
    full_detail = detail
    if rec.arm_msg is not None:
      try:
        self._services[rec.tile_id].abort_install(rec.arm_msg)
      except service_mod.ProgramServiceError as exc:
        full_detail = f"{detail}; channel record not abortable: {exc.code}"
    affected = tuple(sorted(map(repr, rec.holders))) + tuple(
      f"anonymous#{i + 1}" for i in range(rec.anonymous_refs)
    )
    self._drop(rec)
    self._latch(ResidencyFault(rec.tile_id, rec.image.identity, code, full_detail, affected))
    self._counters["installs_faulted"] += 1
    return ResidencyError(code, full_detail)

  def _desync(self, tile_id: int, identity: ProgramIdentity, detail: str) -> ResidencyError:
    self._latch(ResidencyFault(tile_id, identity, "SERVICE_DESYNC", detail))
    return ResidencyError("SERVICE_DESYNC", detail)

  def _latch(self, fault: ResidencyFault) -> None:
    if self._fault is None:
      self._fault = fault

  @property
  def fault(self) -> ResidencyFault | None:
    """The latched fail-stop fault; cleared by ``invalidate_tile``/``invalidate_group``/``reset``."""
    return self._fault

  # ------------------------------------------------------------------
  # invalidation, reset (legacy + real)
  # ------------------------------------------------------------------
  def invalidate_tile(self, tile_id: int) -> None:
    """A Tile reset voids that Tile's residency immediately — the installed state is gone and no
    INVALIDATE handshake exists.  Other Tiles keep theirs; only the held identities' channel
    epochs advance (``ProgramService.reset()`` lockstep)."""
    for e in self._entries.values():
      e.tile_states.pop(tile_id, None)
      e.tile_epochs.pop(tile_id, None)
    service = self._services.get(tile_id)
    if service is None:
      return
    for rec in self._tile_records(tile_id):
      self._drop(rec)
    service.reset()
    if self._fault is not None and self._fault.tile_id == tile_id:
      self._fault = None

  def invalidate_group(self) -> None:
    """Group reset / group invalidation: every residency on every Tile is voided and every
    recorded channel epoch advances — nothing installed before can be hit or ACKed afterwards."""
    for e in self._entries.values():
      e.epoch += 1  # bump epoch so warm hits become cold
      e.tile_states.clear()
      e.tile_epochs.clear()
    for tile_id in list(self._services):
      for rec in self._tile_records(tile_id):
        self._drop(rec)
      self._services[tile_id].reset()
    self._fault = None

  def reset(self) -> None:
    """Full teardown: analytic entries and manager counters cleared; real records dropped through
    each channel's reset — images and their channel epochs stay, exactly like the service keeps
    its registrations, so re-registration stays idempotent.  Per-tile byte counters are cumulative
    channel-traffic accounting and survive, mirroring ``ProgramService.stats``."""
    self._entries.clear()
    self.cold_load_cycles = 0
    for tile_id in list(self._services):
      for rec in self._tile_records(tile_id):
        self._drop(rec)
      self._services[tile_id].reset()
    self._fault = None
    self._counters.clear()
    self._waits.clear()
    self._completion_seq = 0

  # ------------------------------------------------------------------
  # inspection
  # ------------------------------------------------------------------
  def view(self, identity: Any, tile_id: int) -> ResidencyView | None:
    """Read-only projection of the live record; ``None`` when nothing is installed."""
    ident = ProgramIdentity.of(identity)
    img = self._images.get(ident.key)
    if img is None or img.identity != ident:
      return None
    rec = self._records.get((*ident.key, tile_id))
    if rec is None:
      return None
    return ResidencyView(
      identity=rec.image.identity,
      tile_id=tile_id,
      state=rec.state,
      epoch=rec.epoch,
      slot=rec.image.slot,
      slot_address=rec.image.text_base,
      installed=rec.installed,
      local_handle=rec.local_handle,
      completion_id=rec.completion_id,
      refs=rec.refs,
      bytes_served=rec.bytes_served,
      invalidate_sent=rec.invalidate_sent,
    )

  def install_epoch(self, identity: Any, tile_id: int) -> int:
    """The epoch the next ``PROGRAM_INSTALL`` for this ``(identity, Tile)`` must carry — mirrors
    the channel's own install counter: the image's registration epoch, advanced +1 per dropped
    record on that Tile."""
    img = self._image_of(identity)
    self._tile(tile_id)
    return img.epochs.get(tile_id, img.program_epoch)

  def slot_bitmap(self, tile_id: int) -> int:
    """Slots owned by any live record (PENDING_INSTALL / RESIDENT / INVALIDATING)."""
    self._tile(tile_id)
    bits = 0
    for rec in self._tile_records(tile_id):
      bits |= 1 << rec.image.slot
    return bits

  def resident_bitmap(self, tile_id: int) -> int:
    """Slots whose install is ACK-verified (RESIDENT); matches the scenario's resident_bitmap."""
    self._tile(tile_id)
    bits = 0
    for rec in self._tile_records(tile_id):
      if rec.state is ResidencyState.RESIDENT:
        bits |= 1 << rec.image.slot
    return bits

  def used_bytes(self, tile_id: int) -> int:
    """Cumulative installed+installing bytes on the Tile — the RES-14 ledger."""
    self._tile(tile_id)
    return self._used_bytes(tile_id)

  def free_bytes(self, tile_id: int) -> int:
    return self._require_storage().capacity_bytes - self.used_bytes(tile_id)

  def counters(self) -> dict:
    return {**dict(self._counters), "waits": dict(self._waits)}

  def tile_counters(self, tile_id: int) -> dict:
    return dict(self._tile_counters[self._tile(tile_id)])

  def assert_lockstep(self, tile_id: int | None = None) -> None:
    """The manager's live records must equal the channel endpoint's own bookkeeping — the armed
    epoch, the wire slot address, the package hash, the ACK/invalidating flags, the bytes served
    and the endpoint's external-text counter, compared strictly."""
    tiles = self._services.keys() if tile_id is None else (tile_id,)
    for tid in tiles:
      if tid not in self._services:
        raise ResidencyError("TILE_NOT_ATTACHED", f"tile {tid}")
      service = self._services[tid]
      theirs: dict[tuple[int, int, int], dict] = {}
      for key, info in service.snapshot().get("installs", {}).items():
        match = re.fullmatch(r"p(\d+):v(\d+):e(\d+)", key)
        if match is None:
          raise ResidencyError("SERVICE_DESYNC", f"unparseable channel record {key!r}")
        k3 = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        theirs[k3] = info
      mine = {
        (r.image.identity.program_id, r.image.identity.version, r.epoch): r
        for r in self._tile_records(tid)
      }
      if set(theirs) != set(mine):
        raise ResidencyError(
          "SERVICE_DESYNC", f"tile {tid}: channel {sorted(theirs)} != table {sorted(mine)}"
        )
      for key, rec in mine.items():
        info = theirs[key]
        if int(info["epoch"]) != rec.epoch:
          raise ResidencyError(
            "SERVICE_DESYNC", f"tile {tid} program {key}: channel epoch {info['epoch']} != {rec.epoch}"
          )
        if int(info["slot"]) != rec.image.text_base:
          raise ResidencyError(
            "SERVICE_DESYNC",
            f"tile {tid} program {key}: channel slot {info['slot']} != address "
            f"{rec.image.text_base}",
          )
        if info["package_hash"] != rec.image.identity.package_hash.hex():
          raise ResidencyError(
            "SERVICE_DESYNC", f"tile {tid} program {key}: channel package_hash differs"
          )
        if bool(info["invalidating"]) != rec.invalidate_sent:
          raise ResidencyError(
            "SERVICE_DESYNC",
            f"tile {tid} program {key}: channel invalidating={info['invalidating']} "
            f"table invalidate_sent={rec.invalidate_sent}",
          )
        if bool(info["acked"]) != rec.installed:
          raise ResidencyError(
            "SERVICE_DESYNC",
            f"tile {tid} program {key}: channel acked={info['acked']} table installed={rec.installed}",
          )
        if int(info["bytes_served"]) != rec.bytes_served:
          raise ResidencyError(
            "SERVICE_DESYNC",
            f"tile {tid} program {key}: channel served {info['bytes_served']} != {rec.bytes_served}",
          )
        if info.get("ready_handle") != rec.local_handle:
          raise ResidencyError(
            "SERVICE_DESYNC",
            f"tile {tid} program {key}: channel handle {info.get('ready_handle')} != "
            f"{rec.local_handle}",
          )
      served = int(service.stats.get("text_bytes_served", -1))
      if served != self._tile_counters[tid]["external_text_bytes"]:
        raise ResidencyError(
          "SERVICE_DESYNC",
          f"tile {tid}: channel text_bytes_served {served} != "
          f"{self._tile_counters[tid]['external_text_bytes']}",
        )

  def snapshot(self) -> dict:
    tiles: dict[int, dict] = {}
    for tile_id in sorted(self._services):
      records = self._tile_records(tile_id)
      used = self._used_bytes(tile_id)
      slot_bitmap = 0
      resident_bitmap = 0
      slots: dict[int, dict] = {}
      for rec in records:
        slot_bitmap |= 1 << rec.image.slot
        if rec.state is ResidencyState.RESIDENT:
          resident_bitmap |= 1 << rec.image.slot
        slots[rec.image.slot] = {
          "program_id": rec.image.identity.program_id,
          "version": rec.image.identity.version,
          "package_hash": rec.image.identity.package_hash.hex(),
          "state": rec.state.name,
          "epoch": rec.epoch,
          "refs": rec.refs,
          "local_handle": rec.local_handle,
          "completion_id": rec.completion_id,
          "bytes_served": rec.bytes_served,
          "invalidate_sent": rec.invalidate_sent,
          "text_base": rec.image.text_base,
          "text_bytes": len(rec.image.text),
        }
      tiles[tile_id] = {
        "used_bytes": used,
        "free_bytes": (self._storage.capacity_bytes - used) if self._storage else None,
        "slot_bitmap": slot_bitmap,
        "resident_bitmap": resident_bitmap,
        "counters": dict(self._tile_counters[tile_id]),
        "slots": slots,
      }
    images = {
      f"p{image.identity.program_id}:v{image.identity.version}": {
        "package_hash": image.identity.package_hash.hex(),
        "text_bytes": len(image.text),
        "text_sha256": image.text_sha256.hex(),
        "slot": image.slot,
        "text_base": image.text_base,
        "program_epoch": image.program_epoch,
        "epochs": dict(image.epochs),
      }
      for image in self._images.values()
    }
    return {
      # The analytic timing model is NOT residency proof; it exists only for the legacy backend.
      "analytic": {
        pid: {
          "version": e.version,
          "epoch": e.epoch,
          "tile_states": {t: s.name for t, s in e.tile_states.items()},
        }
        for pid, e in self._entries.items()
      },
      "installed": {
        "storage": (
          {
            "capacity_bytes": self._storage.capacity_bytes,
            "slot_bytes": self._storage.slot_bytes,
            "slot_count": self._storage.slot_count,
            "origin_bytes": self._storage.origin_bytes,
          }
          if self._storage is not None
          else None
        ),
        "fault": (
          None
          if self._fault is None
          else {
            "tile_id": self._fault.tile_id,
            "identity": (
              None
              if self._fault.identity is None
              else f"p{self._fault.identity.program_id}:v{self._fault.identity.version}"
            ),
            "code": self._fault.code,
            "detail": self._fault.detail,
            "affected": list(self._fault.affected),
          }
        ),
        "counters": dict(self._counters),
        "waits": dict(self._waits),
        "images": images,
        "tiles": tiles,
      },
    }

  # ------------------------------------------------------------------
  # real path internals
  # ------------------------------------------------------------------
  def _g2t(self, message_type: str, fields: dict) -> BoundaryMessage:
    BoundaryMessage, _ = _cosim()
    msg = BoundaryMessage(
      message_type=message_type,
      channel="PROGRAM",
      direction="G2T",
      fields=dict(fields),
      message_seq=0,
    )
    _ = msg.pack_wire()  # fail early if a field cannot ride the sealed v5 encoding
    return msg

  def _require_storage(self) -> ResidencyStorage:
    if self._storage is None:
      raise ResidencyError("STORAGE_UNCONFIGURED", "configure_storage() the slot geometry first")
    return self._storage

  def _tile(self, tile_id: int) -> int:
    if isinstance(tile_id, bool) or not isinstance(tile_id, int) or tile_id < 0:
      raise ResidencyError("BAD_TILE_ID", repr(tile_id))
    if tile_id not in self._services:
      raise ResidencyError("TILE_NOT_ATTACHED", f"tile {tile_id}")
    return tile_id

  @staticmethod
  def _key3(identity: ProgramIdentity) -> tuple[int, int, bytes]:
    return (identity.program_id, identity.version, identity.package_hash)

  def _image_of(self, identity: Any) -> _Image:
    ident = ProgramIdentity.of(identity)
    img = self._images.get(ident.key)
    if img is None:
      raise ResidencyError("IDENTITY_UNKNOWN", f"program {ident.program_id} v{ident.version}")
    if img.identity != ident:
      raise ResidencyError(
        "IDENTITY_CONFLICT", f"program {ident.program_id} v{ident.version} has a different package"
      )
    return img

  def _record_of(self, tile_id: int, fields: dict, code: str) -> _Residency:
    try:
      program_id, version = int(fields["program_id"]), int(fields["version"])
      epoch = int(fields["program_epoch"])
      package_hash = fields["package_hash"]
    except (KeyError, TypeError, ValueError) as exc:
      raise ResidencyError("BAD_PROGRAM_MESSAGE", repr(exc)) from exc
    if isinstance(package_hash, memoryview):
      package_hash = package_hash.tobytes()
    rec = self._records.get((program_id, version, tile_id))
    if (
      rec is None
      or rec.epoch != epoch
      or rec.image.identity.package_hash != package_hash
    ):
      raise ResidencyError(
        code, f"program {program_id} v{version} epoch {epoch} on tile {tile_id}"
      )
    return rec

  def _tile_records(self, tile_id: int) -> list[_Residency]:
    return [r for r in self._records.values() if r.tile_id == tile_id]

  def _slot_owner(self, tile_id: int, slot: int) -> _Residency | None:
    for rec in self._tile_records(tile_id):
      if rec.image.slot == slot:
        return rec
    return None

  def _used_bytes(self, tile_id: int) -> int:
    return sum(len(rec.image.text) for rec in self._tile_records(tile_id))

  @staticmethod
  def _check_holder(holder: Hashable | None) -> None:
    """Preflight before any mutation: an unhashable holder can never join the ledger."""
    if holder is None:
      return
    try:
      hash(holder)
    except TypeError as exc:
      raise ResidencyError("BAD_HOLDER", f"unhashable holder {holder!r}") from exc

  @staticmethod
  def _reference(rec: _Residency, holder: Hashable | None) -> None:
    if holder is None:
      rec.anonymous_refs += 1
      return
    try:
      known = holder in rec.holders
    except TypeError as exc:
      raise ResidencyError("BAD_HOLDER", f"unhashable holder {holder!r}") from exc
    if known:
      raise ResidencyError("REF_DUPLICATE", f"holder {holder!r} already references the program")
    rec.holders.add(holder)

  def _wait(
    self,
    identity: ProgramIdentity,
    tile_id: int,
    reason: str,
    blocked_by: tuple[ProgramIdentity, ...],
  ) -> ResidencyDecision:
    self._waits[reason] += 1
    return ResidencyDecision(
      identity=identity, tile_id=tile_id, kind="wait", reason=reason, blocked_by=blocked_by
    )

  def _arm(self, img: _Image, tile_id: int) -> _Residency:
    """Arm the ONE shared install for ``(identity, Tile)``: the wire ``slot`` is the pre-linked
    slot ADDRESS, the epoch is this Tile's channel epoch for the identity."""
    img.epochs.setdefault(tile_id, img.program_epoch)
    epoch = img.epochs[tile_id]
    msg = self._g2t(
      "PROGRAM_INSTALL",
      {**img.identity.wire_fields(), "program_epoch": epoch, "slot": img.text_base},
    )
    _, service_mod = _cosim()
    self._completion_seq += 1
    rec = _Residency(
      image=img,
      tile_id=tile_id,
      epoch=epoch,
      state=ResidencyState.PENDING_INSTALL,
      completion_id=self._completion_seq,
      arm_msg=msg,
    )
    try:
      self._services[tile_id].handle_install(msg)
    except service_mod.ProgramServiceError as exc:
      raise ResidencyError(exc.code, exc.detail) from exc
    self._records[(*img.identity.key, tile_id)] = rec
    self._counters["installs_armed"] += 1
    return rec

  # ------------------------------------------------------------------
  # analytic timing model (legacy consumers: admission, reset domain, tests)
  # ------------------------------------------------------------------
  def register(
    self, program_id: int, version: int, program_hash: int, hbm_iova: int, hbm_bytes: int
  ) -> ProgramTableEntry:
    if program_id <= 0 or version <= 0 or program_hash <= 0 or hbm_bytes <= 0:
      raise ValueError("program registration requires complete compiled identity and size")
    prior = self._entries.get(program_id)
    if prior is not None and (prior.version, prior.program_hash, prior.hbm_bytes) == (
      version,
      program_hash,
      hbm_bytes,
    ):
      return prior
    e = ProgramTableEntry(
      program_id=program_id,
      version=version,
      program_hash=program_hash,
      hbm_iova=hbm_iova,
      hbm_bytes=hbm_bytes,
    )
    self._entries[program_id] = e
    return e

  def ensure_resident(self, program_id: int, tile_id: int, cycle: int) -> int:
    """Analytic ONLY: charge the closed-form cold-launch latency and mark the Tile resident
    synchronously (no bytes, no ACK, no references).  Returns the cold-launch latency penalty
    (0 for warm hit, >0 for cold miss).

    Cold path models:
      Group DMA HBM->L2 (program text)  = bytes / group_dma_bandwidth
      Tile DMA L2->L1 program SRAM     = bytes / tile_l1_bandwidth
      install ack                        = 1 cycle
    """
    e = self._entries.get(program_id)
    if e is None:
      raise ValueError(f"program {program_id} was not registered")
    state = e.tile_states.get(tile_id, ResidencyState.ABSENT)
    if state == ResidencyState.RESIDENT and e.tile_epochs.get(tile_id, -1) == e.epoch:
      # warm hit: program resident, epoch matches
      return 0
    # cold miss (or epoch mismatch after reset): install
    lat = self._cold_latency(e.hbm_bytes)
    e.tile_states[tile_id] = ResidencyState.RESIDENT
    e.tile_epochs[tile_id] = e.epoch
    self.cold_load_cycles += lat
    return lat

  def _cold_latency(self, bytes_total: int) -> int:
    """Cold-launch latency for program text install."""
    if bytes_total <= 0:
      return 0
    clock_hz = self.cfg.clock_mhz * 1e6
    hbm_bw = self.cfg.group_dma_bandwidth_gbs * 1e9 / clock_hz
    l1_bw = self.cfg.tile_l1_bandwidth_gbs * 1e9 / clock_hz
    t_hbm = (bytes_total + hbm_bw - 1) // hbm_bw
    t_l1 = (bytes_total + l1_bw - 1) // l1_bw
    return int(t_hbm + t_l1 + 1)  # +1 install ack
