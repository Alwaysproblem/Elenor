"""Program upload is a prepare operation; READ_RESP is sourced from HBM, not the registration copy."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

from pipeline_validator.cosim.program_prepare import prepare_programs
from pipeline_validator.cosim.program_service import ProgramImage, ProgramService, ProgramServiceError
from pipeline_validator.cosim.tile_port import BoundaryMessage
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.memory.hbm_region import HBMRegion
from pipeline_validator.package import PackageError


def frame(kind: str, image: ProgramImage, *, epoch: int = 0, **extra) -> BoundaryMessage:
  direction = "G2T" if kind in ("PROGRAM_INSTALL", "PROGRAM_INVALIDATE") else "T2G"
  fields = {
    "program_id": image.program_id,
    "version": image.version,
    "package_hash": image.package_hash,
    "program_epoch": epoch,
    **extra,
  }
  return BoundaryMessage(
    message_type=kind, channel="PROGRAM", direction=direction, fields=fields, message_seq=0
  )


def test_pull_uses_uploaded_hbm_and_ignores_neither_transfer_count_nor_digest() -> None:
  text = b"0123456789"
  image = ProgramImage(1, 1, bytes(32), 0, text)
  svc = ProgramService(chunk_bytes=8)
  svc.register_image(image, hbm_iova=0x1000)
  assert svc.hbm.byte_store is not None
  assert svc.hbm.byte_store.read_hbm(0x1000, 10) == text
  # Alter the source after registration. A service slicing ProgramImage.text would silently mask this.
  svc.hbm.byte_store.seed_hbm(0x1008, b"XX")
  svc.handle_install(frame("PROGRAM_INSTALL", image, slot=0))
  req = frame("PROGRAM_READ_REQ", image)
  ack = frame("PROGRAM_INSTALL_ACK", image, installed_len=10, measured_hash=hashlib.sha256(text).digest())
  assert svc.handle_read_req(req)[1] == text[:8]
  with pytest.raises(ProgramServiceError, match="INCOMPLETE_INSTALL"):
    svc.handle_install_ack(ack)
  assert svc.handle_read_req(req)[1] == b"XX"
  with pytest.raises(ProgramServiceError, match="INSTALL_HASH_MISMATCH"):
    svc.handle_install_ack(frame("PROGRAM_INSTALL_ACK", image, installed_len=10))
  with pytest.raises(ProgramServiceError, match="INSTALL_HASH_MISMATCH"):
    svc.handle_install_ack(
      frame(
        "PROGRAM_INSTALL_ACK",
        image,
        installed_len=10,
        measured_hash=hashlib.sha256(text[:8] + b"XX").digest(),
      )
    )
  assert not svc.install_complete(1, 1, image.package_hash)
  svc.abort_install(req)
  with pytest.raises(ProgramServiceError, match="ACK_WITHOUT_INSTALL"):
    svc.handle_install_ack(ack)


def test_revoked_hbm_source_is_never_served_from_the_registration_copy() -> None:
  svc = ProgramService()
  image = ProgramImage(1, 1, bytes(32), 0, b"code")
  svc.register_image(image)
  svc.handle_install(frame("PROGRAM_INSTALL", image, slot=0))
  svc.hbm.reset()
  with pytest.raises(ProgramServiceError, match="PROGRAM_SOURCE_UNAVAILABLE"):
    svc.handle_read_req(frame("PROGRAM_READ_REQ", image))
  assert svc.stats["text_bytes_served"] == 0


def test_invalidation_retains_slot_until_real_ack_and_rejects_late_generation() -> None:
  svc = ProgramService()
  a, b = (ProgramImage(i, 1, bytes([i]) * 32, 0, bytes([i]) * 4) for i in (1, 2))
  for img in (a, b):
    svc.register_image(img)
  svc.handle_install(frame("PROGRAM_INSTALL", a, slot=0))
  svc.handle_read_req(frame("PROGRAM_READ_REQ", a))
  svc.handle_install_ack(
    frame("PROGRAM_INSTALL_ACK", a, installed_len=4, measured_hash=hashlib.sha256(a.text).digest())
  )
  svc.handle_invalidate(frame("PROGRAM_INVALIDATE", a))
  with pytest.raises(ProgramServiceError, match="SLOT_OCCUPIED"):
    svc.handle_install(frame("PROGRAM_INSTALL", b, slot=0))
  svc.handle_invalidate_ack(frame("INVALIDATE_ACK", a))
  with pytest.raises(ProgramServiceError, match="EPOCH_MISMATCH"):
    svc.handle_install(frame("PROGRAM_INSTALL", a, slot=0))
  svc.handle_install(frame("PROGRAM_INSTALL", a, epoch=1, slot=0))
  with pytest.raises(ProgramServiceError, match="READ_WITHOUT_INSTALL"):
    svc.handle_read_req(frame("PROGRAM_READ_REQ", a))
  assert svc.handle_read_req(frame("PROGRAM_READ_REQ", a, epoch=1))[1] == a.text


def test_bad_second_package_cannot_partially_upload_to_live_group(tmp_path: Path) -> None:
  h = Path(__file__).resolve().parents[2] / "hardware/coral_nexus"
  config = h / "configs/cs1_c2.yaml"
  root = tmp_path / "package"
  subprocess.run(
    [
      "bash",
      str(h / "scripts/cn.sh"),
      "package",
      "--binding",
      str(h / "kernels/bindings/scalar_ctl_v1.yaml"),
      "--config",
      str(config),
      "--out",
      str(root),
    ],
    check=True,
    timeout=300,
  )
  original = (h / "verification/scenarios/t58_cold_warm.yaml").read_text()
  hbm = HBMRegion(byte_store=ByteStore())
  for edit, error, message in (
    ({"package_hash": "0" * 64}, PackageError, "SCENARIO_HASH_MISMATCH"),
    ({"hbm_iova": 0x300001}, ValueError, "overlapping"),
  ):
    scenario = yaml.safe_load(original)
    scenario["slots"]["B"].update(edit)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(scenario))
    with pytest.raises(error, match=message):
      prepare_programs(path, root, config_path=config, hbm=hbm)
    assert hbm.bound_ranges() == ()
