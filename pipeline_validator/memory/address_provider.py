"""Pure indexed-address resolution shared by Tile Gather and Tile Scatter.

Plan §2: every indexed access resolves ``remote_element = index[i] * S + O
+ t * P + j * T`` (one segment per repeat ``j``), copying ``L`` elements
into ``local_element = (i * R + j) * L``.  All units are elements of the
remote dtype.  This module is the only addressing implementation; the
MFE engines call it after decoding index bytes.
"""

from __future__ import annotations

from ..execution_ir import ExecIndexedMap
from .transfer import ResolvedMemoryView, slice_resolved_view


class IndexedAddressError(ValueError):
  """Raised for negative indices, overflow, or out-of-view segments."""


def decode_index_i32(data: bytes) -> int:
  """Decode one little-endian signed i32."""
  if len(data) != 4:
    raise IndexedAddressError(f"index read returned {len(data)} bytes, expected 4")
  return int.from_bytes(data, "little", signed=True)


def resolve_indexed_segments(
  address_map: ExecIndexedMap,
  index: int,
  ordinal: int,
  task_id: int,
  remote: ResolvedMemoryView,
  local: ResolvedMemoryView,
  element_bytes: int,
) -> tuple[tuple[ResolvedMemoryView, ResolvedMemoryView], ...]:
  """Resolve one index slot into (remote_slice, local_slice) pairs.

  ``index`` is the decoded i32 value, ``ordinal`` the index slot ``i``
  and ``task_id`` the logical task ``t``.  Negative indices raise
  ``indexed_address_out_of_bounds``; uint64 element-address overflow
  raises ``indexed_address_overflow``; any segment extending past the
  remote view raises ``indexed_address_out_of_bounds``.  Python-style
  negative wraparound is never applied.
  """
  index_scale = address_map.index_scale
  offset = address_map.offset
  task_stride = address_map.task_stride
  repeat = address_map.repeat
  stride = address_map.stride
  segment = address_map.segment

  if index < 0:
    raise IndexedAddressError("indexed_address_out_of_bounds: negative index")
  remote_elements = remote.size_bytes // element_bytes
  base = index * index_scale + offset + task_id * task_stride
  pairs: list[tuple[ResolvedMemoryView, ResolvedMemoryView]] = []
  for j in range(repeat):
    start = base + j * stride
    if start > remote_elements or segment > remote_elements - start:
      raise IndexedAddressError(
        f"indexed_address_out_of_bounds: segment [{start}, +{segment}) exceeds remote view"
        f" of {remote_elements} elements"
      )
    element_addr = start * element_bytes
    if element_addr + segment * element_bytes >= 1 << 64:
      raise IndexedAddressError("indexed_address_overflow")
    local_start = (ordinal * repeat + j) * segment * element_bytes
    remote_slice = slice_resolved_view(remote, element_addr, segment * element_bytes)
    local_slice = slice_resolved_view(local, local_start, segment * element_bytes)
    if remote_slice is None or local_slice is None:
      raise IndexedAddressError("indexed segment slice is not resolvable")
    pairs.append((remote_slice, local_slice))
  return tuple(pairs)
