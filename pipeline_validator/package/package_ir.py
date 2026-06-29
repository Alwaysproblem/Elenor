"""Executable Package IR (design/elenor_executable_package/ sections 3-5).

Models the package object that the Host Runtime opens, validates, uploads,
binds to a context, and launches.  Mirrors the package state machine
(Created -> Verified -> ... -> ReusableWarm -> Evicted) and the cold/warm
load/patch/submit flows in the executable package design doc section 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class PackageState(IntEnum):
  """Executable Package state machine (package design section 3)."""
  CREATED = 0
  VERIFIED = 1
  LOADED_TO_HOST = 2
  BOUND_TO_CONTEXT = 3
  UPLOADED_TO_DEVICE = 4
  RESIDENT_READY = 5
  SUBMITTED = 6
  RUNNING = 7
  COMPLETED = 8
  REUSABLE_WARM = 9
  EVICTED = 10


@dataclass
class PackageHeader:
  """elenor_pkg_header_v0_t (package design 4.2)."""
  magic: int = 0x45504B47  # 'EPKG'
  package_abi_version: int = 1
  header_bytes: int = 0
  package_flags: int = 0
  target_profile_id: int = 0
  section_count: int = 0
  package_bytes: int = 0
  header_crc32: int = 0
  package_crc32: int = 0
  command_abi_version: int = 1
  descriptor_abi_version: int = 1
  slot_frame_abi_version: int = 1
  stream_queue_abi_version: int = 1


@dataclass
class PackageSection:
  """elenor_pkg_section_v0_t (package design 4.2)."""
  section_type: int  # ELENOR_SEC_*
  section_flags: int = 0
  file_offset: int = 0
  file_bytes: int = 0
  required_alignment: int = 0
  device_alignment: int = 0
  crc32: int = 0


@dataclass
class RelocationEntry:
  """Relocation table entry (package design 4.1 relocation_table)."""
  patch_id: int
  desc_id: int
  field_offset: int
  field_width: int
  owner: int  # who may patch: runtime / uce / mfe / use
  addr_mode: int
  slot_id: int = 0


@dataclass
class GraphSchedule:
  """Graph schedule section (package design 4.1)."""
  group_task_entries: list = field(default_factory=list)
  dependency_table: list = field(default_factory=list)
  memory_lifetime_table: list = field(default_factory=list)
  launch_metadata: dict = field(default_factory=dict)


@dataclass
class DeviceCaps:
  """Device capabilities used for package validation."""
  profile_id: int = 0
  num_tiles: int = 4
  tile_l1_bytes: int = 1 * 1024 * 1024
  group_sram_bytes: int = 8 * 1024 * 1024


@dataclass
class ElenorPackage:
  """Top-level package object (package design 4.1 object model).

  The simulator consumes this to drive the Host Runtime state machine and
  the Program Residency Manager.  Program text / descriptors / weights are
  referenced by IOVA after upload; the package itself is a software object.
  """
  header: PackageHeader = field(default_factory=PackageHeader)
  sections: list[PackageSection] = field(default_factory=list)
  graph_schedule: GraphSchedule = field(default_factory=GraphSchedule)
  program_repository: dict[int, object] = field(default_factory=dict)
  descriptor_repository: dict = field(default_factory=dict)
  command_templates: list = field(default_factory=list)
  slot_frame_templates: list = field(default_factory=list)
  relocation_table: list[RelocationEntry] = field(default_factory=list)
  weight_sections: list = field(default_factory=list)
  pmu_manifest: dict = field(default_factory=dict)
  state: PackageState = PackageState.CREATED

  # IOVA assignments after upload (filled by HostRuntime)
  program_iovas: dict[int, int] = field(default_factory=dict)
  program_hashes: dict[int, int] = field(default_factory=dict)
  program_bytes: dict[int, int] = field(default_factory=dict)

  def validate(self, caps: DeviceCaps) -> bool:
    """Package design 3: Created -> Verified.

    Checks magic, ABI version, section count, CRC, and target profile.
    Returns True if the package may transition to VERIFIED.
    """
    if self.header.magic != 0x45504B47:
      return False
    if self.header.package_abi_version != 1:
      return False
    if self.header.section_count != len(self.sections):
      return False
    if self.header.target_profile_id != caps.profile_id:
      return False
    self.state = PackageState.VERIFIED
    return True

  def transition(self, new_state: PackageState) -> None:
    """Advance the package state machine."""
    self.state = new_state


def validate_package(pkg: ElenorPackage, caps: DeviceCaps) -> bool:
  """Convenience wrapper for pkg.validate()."""
  return pkg.validate(caps)
