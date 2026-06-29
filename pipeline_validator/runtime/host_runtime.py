"""Host Runtime — package open/validate/upload/patch/submit (Executable
Package 5.1, Driver-Firmware 5.1).

Models the user-mode runtime's software overhead: package validation,
context binding, descriptor patching, and command buffer building.  Each
operation consumes cycles from `HardwareConfig` (host_validate_cycles,
host_patch_cycles, etc.).  These are the runtime overheads that the
existing validator treats as zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import HardwareConfig
from ..package.package_ir import DeviceCaps, ElenorPackage, PackageState


@dataclass
class HostRuntime:
  """User-mode runtime: package lifecycle + descriptor patch."""
  cfg: HardwareConfig

  def __post_init__(self) -> None:
    self.pmu_validate_cycles: int = 0
    self.pmu_patch_cycles: int = 0
    self.pmu_build_cmd_cycles: int = 0
    self.warm_launch_overhead_cycles: int = 0

  def open_and_validate(self, pkg: ElenorPackage,
                        caps: DeviceCaps) -> int:
    """Package design 3: Created -> Verified.  Returns cycles consumed."""
    if not pkg.validate(caps):
      return 0
    self.pmu_validate_cycles += self.cfg.host_validate_cycles
    return self.cfg.host_validate_cycles

  def bind_context(self, pkg: ElenorPackage, context_id: int,
                   queue_id: int) -> int:
    """Package design 3: Verified -> BoundToContext.  Returns cycles."""
    pkg.transition(PackageState.BOUND_TO_CONTEXT)
    return 1  # lightweight binding

  def upload(self, pkg: ElenorPackage) -> int:
    """Package design 3: BoundToContext -> UploadedToDevice.
    Program/descriptor/weight sections get HBM IOVAs.  Returns cycles."""
    pkg.transition(PackageState.UPLOADED_TO_DEVICE)
    pkg.transition(PackageState.RESIDENT_READY)
    # upload cost modeled as validate + IOVA allocation (lightweight in sim)
    return 5

  def patch_descriptors(self, pkg: ElenorPackage,
                        warm: bool = False) -> int:
    """Descriptor patch (context/shape/buffer binding).

    Warm path: patch + descriptor cache invalidate (Driver-Firmware 5.2).
    Cold path: full relocation patch.
    """
    if warm:
      self.warm_launch_overhead_cycles += self.cfg.host_patch_cycles
    else:
      self.pmu_patch_cycles += self.cfg.host_patch_cycles
    return self.cfg.host_patch_cycles

  def build_launch(self, pkg: ElenorPackage) -> int:
    """Build command buffer from command templates.  Returns cycles."""
    pkg.transition(PackageState.SUBMITTED)
    self.pmu_build_cmd_cycles += 2
    return 2

  def reset(self) -> None:
    self.pmu_validate_cycles = 0
    self.pmu_patch_cycles = 0
    self.pmu_build_cmd_cycles = 0
    self.warm_launch_overhead_cycles = 0

  def snapshot(self) -> dict:
    return {
      "validate_cycles": self.pmu_validate_cycles,
      "patch_cycles": self.pmu_patch_cycles,
      "build_cmd_cycles": self.pmu_build_cmd_cycles,
      "warm_launch_overhead": self.warm_launch_overhead_cycles,
    }
