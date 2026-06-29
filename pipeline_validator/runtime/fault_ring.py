"""Fault Ring — fault record storage (Fault/Reset design 4.1, review P0-5).

Per the review P0-5 recommendation, this module uses the Fault/Reset
document's definition as the *authoritative* fault record struct.  All
other docs (Runtime ABI) normative-reference it.  The simulator never
decodes the Runtime ABI's incompatible v0 struct.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum


class FaultCode(IntEnum):
  """Fault code examples (Runtime ABI 4.4)."""
  INVALID_DESCRIPTOR = 1
  UNSUPPORTED_ABI = 2
  ADDRESS_FAULT = 3
  DMA_TIMEOUT = 4
  EVENT_DEPENDENCY_TIMEOUT = 5
  STREAM_PROTOCOL_ERROR = 6
  ENGINE_INTERNAL_FAULT = 7
  RESET_DURING_EXECUTION = 8
  L2_CAPACITY_FAULT = 9
  INVALID_LAYOUT_FAULT = 10
  SLOT_PERMISSION_FAULT = 11


class FaultDomain(IntEnum):
  """Reset domain (Compute Tile 6.47-6.48, Driver-Firmware 3.4)."""
  QUEUE = 0
  TILE = 1
  GROUP = 2
  DEVICE = 3


@dataclass
class FaultRecord:
  """elenor_fault_record_v0_t — authoritative per Fault/Reset 4.1.

  This is the full-diagnosis struct (abi_version, code, source, severity,
  fault_record_index, context_id, command_id, event_id, program_id, ...).
  """
  abi_version: int = 1
  code: FaultCode = FaultCode.ENGINE_INTERNAL_FAULT
  source: int = 0
  severity: int = 0
  fault_record_index: int = 0
  context_id: int = 0
  command_id: int = 0
  event_id: int = 0
  program_id: int = 0
  desc_id: int = 0
  group_id: int = 0
  tile_id: int = 0
  queue_id: int = 0
  slot_id: int = 0
  patch_id: int = 0
  engine_id: int = 0
  offending_addr: int = 0
  aux0: int = 0
  aux1: int = 0
  pmu_snapshot_ptr: int = 0


class FaultRing:
  """Fixed-size ring buffer of fault records.

  Producer: any engine / DMA / slot-frame validator that detects a fault.
  Consumer: host runtime reads via elenor_fault_read().
  """

  def __init__(self, slots: int = 16) -> None:
    self._records: deque[FaultRecord] = deque(maxlen=slots)
    self._head: int = 0
    self.write_count: int = 0
    self._slots: int = slots

  def write(self, rec: FaultRecord) -> int:
    """Write a fault record, return its fault_record_index."""
    rec.fault_record_index = self._head
    maxlen = self._records.maxlen or self._slots
    self._head = (self._head + 1) % maxlen
    self._records.append(rec)
    self.write_count += 1
    return rec.fault_record_index

  def read(self, slot: int) -> FaultRecord | None:
    """Read the fault record at a ring slot (oldest-first)."""
    if 0 <= slot < len(self._records):
      return list(self._records)[slot]
    return None

  def latest(self) -> FaultRecord | None:
    if self._records:
      return self._records[-1]
    return None

  def __len__(self) -> int:
    return len(self._records)

  def reset(self) -> None:
    self._records.clear()
    self._head = 0
    self.write_count = 0

  def snapshot(self) -> dict:
    return {
      "count": len(self._records),
      "write_count": self.write_count,
      "latest_code": (self._records[-1].code.name
                      if self._records else None),
    }
