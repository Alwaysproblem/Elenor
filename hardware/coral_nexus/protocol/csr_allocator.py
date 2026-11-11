"""Deterministic CSR address allocation for the declared ``nexus.*`` names (C017).

Names come from ``csr_name_map.yaml``; occupied addresses come from the table parsed
out of the pinned ``Csr.scala`` (``upstream_csr_occupied.json``). Allocation is
first-fit ascending inside ``allocation_window`` in DECLARED order, so appending a
name never moves an earlier one and file iteration order cannot change the result.

This allocates ADDRESSES only. The access level of every register stays whatever
``csr_name_map.yaml`` says (``unspecified`` today); nothing here invents one.
"""

from __future__ import annotations

from typing import Any


class CsrAllocError(Exception):
  """Allocation refused; carries a coded reason so callers and tests can tell classes apart."""

  def __init__(self, code: str, detail: str) -> None:
    super().__init__(f"{code}: {detail}")
    self.code = code
    self.detail = detail


def window_of(csr_map: dict[str, Any]) -> tuple[int, int]:
  window = csr_map.get("allocation_window")
  if not isinstance(window, dict) or not window.get("source"):
    raise CsrAllocError("NO_WINDOW", "csr_name_map.yaml has no sourced allocation_window")
  low, high = window.get("low"), window.get("high")
  if (
    not (isinstance(low, int) and isinstance(high, int)) or isinstance(low, bool) or isinstance(high, bool)
  ):
    raise CsrAllocError("NO_WINDOW", f"allocation_window bounds must be integers, got {low!r}/{high!r}")
  if not 0 <= low <= high <= 0xFFF:
    raise CsrAllocError("NO_WINDOW", f"allocation_window {low:#x}..{high:#x} is not a 12-bit range")
  return low, high


def allocate(csr_map: dict[str, Any], upstream: dict[str, Any]) -> list[dict[str, Any]]:
  """Return one record per declared name, in declared order, each with a unique free address."""
  low, high = window_of(csr_map)
  occupied = {int(e["address"]): str(e["name"]) for e in upstream["occupied"]}
  declared = csr_map["declared"]
  names = [str(item["name"]) for item in declared]
  if len(set(names)) != len(names):
    raise CsrAllocError("DUPLICATE_NAME", "a declared CSR name appears twice")
  for item in declared:
    if "address" in item:
      raise CsrAllocError(
        "ADDRESS_SUPPLIED", f"{item['name']} supplies an address; only the allocator assigns"
      )
  free = [a for a in range(low, high + 1) if a not in occupied]
  if len(declared) > len(free):
    raise CsrAllocError(
      "CAPACITY",
      f"{len(declared)} names need addresses but {low:#x}..{high:#x} has {len(free)} free "
      f"after {len(occupied)} upstream-occupied",
    )
  return [
    {
      "name": str(item["name"]),
      "address": address,
      "owner": item["owner"],
      "scope": item["scope"],
      "access": item.get("access", "unspecified"),
    }
    for item, address in zip(declared, free)
  ]
