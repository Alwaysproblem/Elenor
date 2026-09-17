"""HBM IOVA region with generation-safe external bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .allocator import AllocationHandle, BankSegment, ExternalOwner, MemoryInvariantError, MemoryOwner

if TYPE_CHECKING:
  from ..trace import MemoryTrace
  from .byte_store import ByteStore


@dataclass
class HBMRegion:
  """External HBM binding registry plus load-time range/permission checks."""

  base_iova: int = 0
  size_bytes: int = 16 * 1024 * 1024 * 1024
  bandwidth_gbs: float = 819.2
  outstanding_limit: int = 32
  trace: MemoryTrace | None = None
  byte_store: ByteStore | None = None

  def __post_init__(self) -> None:
    self._bindings: dict[str, AllocationHandle] = {}
    self._permissions: dict[str, str] = {}
    self._by_allocation_id: dict[str, AllocationHandle] = {}
    self._next_generation: int = 0
    self._outstanding: int = 0

  def bind_external(self, binding, cycle: int = 0) -> AllocationHandle:
    name = binding.name
    base = binding.base_iova
    size = binding.size_bytes
    permissions = binding.permissions
    if not name:
      raise ValueError("input binding name must be non-empty")
    if name in self._bindings:
      raise ValueError(f"input binding '{name}' is already bound")
    if permissions not in ("r", "w", "rw"):
      raise ValueError(f"input binding '{name}' has invalid permissions")
    if size <= 0:
      raise ValueError(f"input binding '{name}' size must be > 0")
    if base < self.base_iova:
      raise ValueError(f"input binding '{name}' base is below HBM region")
    if base + size > self.base_iova + self.size_bytes:
      raise ValueError(f"input binding '{name}' exceeds HBM capacity")
    for existing in self._bindings.values():
      if base < existing.end_address and existing.base_address < base + size:
        raise ValueError(f"input binding '{name}' overlaps existing binding")

    if self._next_generation >= (1 << 64) - 1:
      if self._bindings or self._outstanding:
        raise MemoryInvariantError("HBM binding generation exhausted before quiescence")
      raise MemoryInvariantError("HBM binding generation exhausted; explicit recovery required")
    self._next_generation += 1
    generation = self._next_generation
    owner = ExternalOwner(binding_name=name)
    segment = BankSegment(bank_id=0, address=base, size_bytes=size)
    handle = AllocationHandle(
      allocation_id=f"global:{name}:{generation}",
      memory_space="hbm",
      owner=owner,
      base_address=base,
      size_bytes=size,
      alignment=1,
      bank_segments=(segment,),
      generation=generation,
      allocate_cycle=cycle,
    )
    self._bindings[name] = handle
    self._permissions[name] = permissions
    self._by_allocation_id[handle.allocation_id] = handle
    if self.byte_store is not None:
      self.byte_store.register_hbm_binding(handle, permissions)
    if self.trace is not None:
      self.trace.hbm_bind(binding, handle, cycle)
      self.trace.hbm(self.snapshot(), cycle)
    return handle

  def get_handle(self, name: str) -> AllocationHandle | None:
    return self._bindings.get(name)

  def get_handle_by_allocation_id(self, allocation_id: str) -> AllocationHandle | None:
    return self._by_allocation_id.get(allocation_id)

  def permissions(self, name: str) -> str:
    try:
      return self._permissions[name]
    except KeyError as exc:
      raise MemoryInvariantError("unknown HBM binding") from exc

  def resolve(
    self, handle: AllocationHandle, offset_bytes: int, size_bytes: int, required_permission: str = ""
  ) -> tuple[BankSegment, ...]:
    if not isinstance(handle.owner, ExternalOwner):
      raise MemoryInvariantError("wrong-owner release")
    live = self._bindings.get(handle.owner.binding_name)
    if live is None or live != handle:
      raise MemoryInvariantError("stale allocation generation")
    if type(offset_bytes) is not int or type(size_bytes) is not int:
      raise MemoryInvariantError("memory view out of bounds")
    if offset_bytes < 0 or size_bytes < 0 or offset_bytes + size_bytes > handle.size_bytes:
      raise MemoryInvariantError("memory view out of bounds")
    if required_permission:
      if required_permission not in ("r", "w"):
        raise MemoryInvariantError("invalid HBM permission request")
      if required_permission not in self._permissions[handle.owner.binding_name]:
        raise MemoryInvariantError(
          f"HBM binding '{handle.owner.binding_name}' lacks {required_permission} permission"
        )
    return (BankSegment(bank_id=0, address=handle.base_address + offset_bytes, size_bytes=size_bytes),)

  def assert_live(self, handle: AllocationHandle, owner: MemoryOwner | None = None) -> None:
    live = self._bindings.get(handle.owner.binding_name if isinstance(handle.owner, ExternalOwner) else "")
    if live is None or live != handle:
      raise MemoryInvariantError("stale allocation generation")
    if owner is not None and live.owner != owner:
      raise MemoryInvariantError("wrong-owner release")

  def unbind_external(self, name: str, cycle: int = 0) -> None:
    handle = self._bindings.pop(name, None)
    if handle is None:
      raise MemoryInvariantError("unknown HBM binding")
    self._permissions.pop(name, None)
    self._by_allocation_id.pop(handle.allocation_id, None)
    if self.byte_store is not None:
      self.byte_store.unregister_hbm_binding(name, handle.allocation_id, handle.generation)
    if self.trace is not None:
      self.trace.hbm_unbind(name, cycle)
      self.trace.hbm(self.snapshot(), cycle)

  def used_bytes(self) -> int:
    return sum(handle.size_bytes for handle in self._bindings.values())

  def bandwidth_bytes_per_cycle(self, clock_hz: float) -> float:
    return self.bandwidth_gbs * 1e9 / clock_hz

  def can_issue(self) -> bool:
    return self._outstanding < self.outstanding_limit

  def issue_outstanding(self) -> None:
    if not self.can_issue():
      raise MemoryInvariantError("HBM outstanding credit exhausted")
    self._outstanding += 1

  def complete_outstanding(self) -> None:
    if self._outstanding <= 0:
      raise MemoryInvariantError("HBM outstanding credit underflow")
    self._outstanding -= 1

  @property
  def outstanding(self) -> int:
    return self._outstanding

  def reset(self) -> None:
    if self._outstanding:
      raise MemoryInvariantError("HBM reset requires confirmed transaction drain")
    for name in tuple(self._bindings):
      self.unbind_external(name)

  def reset_outstanding(self) -> None:
    if self._outstanding:
      raise MemoryInvariantError("cannot discard live HBM outstanding credits")

  def snapshot(self) -> dict[str, object]:
    return {
      "used_bytes": self.used_bytes(),
      "capacity_bytes": self.size_bytes,
      "external_bindings": len(self._bindings),
      "bindings": tuple(
        {
          "name": name,
          "allocation_id": handle.allocation_id,
          "generation": handle.generation,
          "base": handle.base_address,
          "bytes": handle.size_bytes,
          "permissions": self._permissions[name],
        }
        for name, handle in sorted(self._bindings.items())
      ),
      "outstanding": self._outstanding,
      "limit": self.outstanding_limit,
      "next_generation": self._next_generation,
    }
