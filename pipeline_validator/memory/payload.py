"""Payload tracker — byte tracking + layout checking (review P0-3).

Tracks payload metadata (IOVA, bytes, layout, dtype, shape, strides) across
DMA copies and cross-engine handoffs.  Does NOT store or verify numerical
values — that belongs to a separate Python golden / reference kernel layer
(verification_bringup design verification tiers).  This module only
validates the *data layout contract* between engines, which is what P0-3
(paged attention cross-engine ABI) needs to freeze.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Payload:
  """One payload region's metadata."""
  iova: int
  bytes_total: int
  layout: str = ""        # e.g. "ncshd", "paged_kv", "row_major"
  dtype: str = "bf16"
  shape: tuple = ()
  strides: tuple = ()
  head_dim: int = 0       # attention-specific
  producer_kind: str = ""  # "MFE" / "BOA" / "EVU" / "DMA"

  def layout_compat(self, consumer_kind: str,
                    expected_layout: str | None = None,
                    expected_head_dim: int | None = None) -> bool:
    """Check whether this payload's layout matches a consumer's expectation."""
    if expected_layout is not None and self.layout != expected_layout:
      return False
    if expected_head_dim is not None and self.head_dim != expected_head_dim:
      return False
    return True


@dataclass
class PayloadTracker:
  """Tracks payload metadata across DMA copies and engine handoffs."""
  _regions: dict[int, Payload] = field(default_factory=dict)
  layout_fault_count: int = 0
  copy_count: int = 0

  def alloc(self, iova: int, p: Payload) -> None:
    self._regions[iova] = p

  def free(self, iova: int) -> None:
    self._regions.pop(iova, None)

  def get(self, iova: int) -> Payload | None:
    return self._regions.get(iova)

  def copy(self, src_iova: int, dst_iova: int, bytes_total: int,
           layout_transform: str | None = None) -> bool:
    """Record a DMA copy.  Returns False if layout transform is invalid."""
    src = self._regions.get(src_iova)
    if src is None:
      # source not tracked — allow copy but don't create metadata
      self.copy_count += 1
      return True
    new_layout = src.layout
    if layout_transform is not None:
      # V1: layout transforms are named and validated against a small set
      valid_transforms = {"transpose", "reshape", "packed_kv", "none", None}
      if layout_transform not in valid_transforms:
        self.layout_fault_count += 1
        return False
      if layout_transform == "packed_kv":
        new_layout = "paged_kv"
    dst = Payload(
      iova=dst_iova, bytes_total=bytes_total, layout=new_layout,
      dtype=src.dtype, shape=src.shape, strides=src.strides,
      head_dim=src.head_dim, producer_kind="DMA")
    self._regions[dst_iova] = dst
    self.copy_count += 1
    return True

  def check_layout_compat(self, payload_iova: int, consumer_kind: str,
                          expected_layout: str | None = None,
                          expected_head_dim: int | None = None) -> bool:
    """P0-3: check cross-engine layout compatibility before consumption."""
    p = self._regions.get(payload_iova)
    if p is None:
      # untracked payload — allow (timing_only compat)
      return True
    ok = p.layout_compat(consumer_kind, expected_layout, expected_head_dim)
    if not ok:
      self.layout_fault_count += 1
    return ok

  def reset(self) -> None:
    self._regions.clear()
    self.layout_fault_count = 0
    self.copy_count = 0

  def snapshot(self) -> dict:
    return {
      "regions": len(self._regions),
      "copy_count": self.copy_count,
      "layout_faults": self.layout_fault_count,
    }
