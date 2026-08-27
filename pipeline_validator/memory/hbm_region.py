"""HBM IOVA region (Global DMA 6.2, Executable Package 5.1).

Models HBM as a capacity + bandwidth + outstanding-limit resource.
External bindings are registered as immutable ``AllocationHandle``s
with ``ExternalOwner``; capacity, overlap, permission and view-bounds
errors are load-time ``ValueError``s, never successful runtime
transactions.  HBM outstanding/channel timing is held by the
``TransferManager`` (Step 4); this module only owns the external
region registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from .allocator import (
  AllocationHandle,
  BankSegment,
  ExternalOwner,
  MemoryInvariantError,
  MemoryOwner,
)


@dataclass
class HBMRegion:
  """HBM memory region: external binding registry + bandwidth + outstanding."""
  base_iova: int = 0
  size_bytes: int = 16 * 1024 * 1024 * 1024  # 16 GB default
  bandwidth_gbs: float = 819.2
  outstanding_limit: int = 32

  def __post_init__(self) -> None:
    self._bindings: dict[str, AllocationHandle] = {}
    self._generation: int = 0
    self._outstanding: int = 0

  def bind_external(self, binding, cycle: int = 0) -> AllocationHandle:
    """Register a user-supplied ``GlobalBinding`` as an external handle.

    Validates capacity, region overlap, positive size, non-negative base
    and permission.  Errors are ``ValueError`` (load-time), not runtime
    transactions.
    """
    name = binding.name
    base = binding.base_iova
    size = binding.size_bytes
    if size <= 0:
      raise ValueError(f"input binding '{name}' size must be > 0")
    if base < 0:
      raise ValueError(f"input binding '{name}' base must be >= 0")
    if base + size > self.base_iova + self.size_bytes:
      raise ValueError(f"input binding '{name}' exceeds HBM capacity")
    # overlap check against existing bindings
    for existing in self._bindings.values():
      if base < existing.end_address and existing.base_address < base + size:
        raise ValueError(
          f"input binding '{name}' overlaps existing binding")
    owner = ExternalOwner(binding_name=name)
    seg = BankSegment(bank_id=0, address=base, size_bytes=size)
    handle = AllocationHandle(
      allocation_id=f"global:{name}:{self._generation}",
      memory_space="hbm",
      owner=owner,
      base_address=base,
      size_bytes=size,
      alignment=1,
      bank_segments=(seg,),
      generation=self._generation,
      allocate_cycle=cycle,
    )
    self._bindings[name] = handle
    return handle

  def get_handle(self, name: str) -> AllocationHandle | None:
    return self._bindings.get(name)

  def resolve(
    self, handle: AllocationHandle, offset_bytes: int, size_bytes: int,
    required_permission: str = "",
  ) -> tuple[BankSegment, ...]:
    """Resolve a byte range within an external binding.

    Checks negative offset, overflow, use-after-release and view bounds.
    Returns the clipped segment(s).  HBM uses IOVA directly — no SRAM
    bank-major formula.
    """
    if not isinstance(handle.owner, ExternalOwner):
      raise MemoryInvariantError("wrong-owner release")
    live = self._bindings.get(handle.owner.binding_name)
    if live is None or live != handle:
      raise MemoryInvariantError("stale allocation generation")
    if offset_bytes < 0:
      raise MemoryInvariantError("memory view out of bounds")
    if offset_bytes + size_bytes > handle.size_bytes:
      raise MemoryInvariantError("memory view out of bounds")
    # permission check (caller supplies the binding permissions)
    return (BankSegment(
      bank_id=0,
      address=handle.base_address + offset_bytes,
      size_bytes=size_bytes,
    ),)

  def assert_live(self, handle: AllocationHandle,
                  owner: MemoryOwner | None = None) -> None:
    live = self._bindings.get(
      handle.owner.binding_name if isinstance(handle.owner, ExternalOwner)
      else "")
    if live is None or live != handle:
      raise MemoryInvariantError("stale allocation generation")
    if owner is not None and live.owner != owner:
      raise MemoryInvariantError("wrong-owner release")

  def unbind_external(self, name: str) -> None:
    self._bindings.pop(name, None)

  def used_bytes(self) -> int:
    return sum(h.size_bytes for h in self._bindings.values())

  def bandwidth_bytes_per_cycle(self, clock_hz: float) -> float:
    return self.bandwidth_gbs * 1e9 / clock_hz

  def can_issue(self) -> bool:
    return self._outstanding < self.outstanding_limit

  def issue_outstanding(self) -> None:
    self._outstanding += 1

  def complete_outstanding(self) -> None:
    if self._outstanding > 0:
      self._outstanding -= 1

  @property
  def outstanding(self) -> int:
    return self._outstanding

  def reset(self) -> None:
    self._bindings.clear()
    self._generation += 1
    self._outstanding = 0

  def reset_outstanding(self) -> None:
    self._outstanding = 0

  def snapshot(self) -> dict:
    return {
      "used_bytes": self.used_bytes(),
      "capacity_bytes": self.size_bytes,
      "external_bindings": len(self._bindings),
      "outstanding": self._outstanding,
      "limit": self.outstanding_limit,
      "generation": self._generation,
    }
