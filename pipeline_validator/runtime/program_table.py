"""Program Residency Manager (Architecture doc 15, review P0-2).

Models the program residency contract: Tile Group Sequencer ensures
program-ready before dispatch; Tile UCE only consumes a resident local
handle.  Cold Launch = residency miss -> implicit fetch/verify/install.
Warm Launch = program resident -> patch descriptor/context/shape only.

Correctness is gated by program_id + version + hash + epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ResidencyState(IntEnum):
  HBM_ONLY = 0       # program in HBM, not installed to any tile
  FETCHING = 1       # Group/Tile DMA in flight to install
  TILE_RESIDENT = 2  # installed to tile-local program SRAM
  EVICTED = 3        # slot reclaimed, needs re-fetch


@dataclass
class ProgramTableEntry:
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
  """Tracks which tiles have which programs resident.

  cold launch latency = Group DMA HBM->L2 (program text)
                      + Tile DMA L2->L1 program SRAM (install)
                      + install ack cycle
  warm launch latency = 0 (program resident, descriptor cache invalidate only)
  """

  def __init__(self, cfg) -> None:
    self.cfg = cfg
    self._entries: dict[int, ProgramTableEntry] = {}
    self.cold_load_cycles: int = 0  # PMU: total cold-load cycles spent

  def register(self, program_id: int, version: int, program_hash: int,
               hbm_iova: int, hbm_bytes: int) -> ProgramTableEntry:
    e = ProgramTableEntry(
      program_id=program_id, version=version, program_hash=program_hash,
      hbm_iova=hbm_iova, hbm_bytes=hbm_bytes)
    self._entries[program_id] = e
    return e

  def ensure_resident(self, program_id: int, tile_id: int,
                      cycle: int) -> int:
    """Ensure a program is resident on a tile.  Returns the cold-launch
    latency penalty (0 for warm hit, >0 for cold miss).

    Cold path models:
      Group DMA HBM->L2 (program text)  = bytes / group_dma_bandwidth
      Tile DMA L2->L1 program SRAM     = bytes / tile_l1_bandwidth
      install ack                        = 1 cycle
    """
    e = self._entries.get(program_id)
    if e is None:
      # unregistered program: treat as cold miss with default size
      return self._cold_latency(64 * 1024)
    state = e.tile_states.get(tile_id, ResidencyState.HBM_ONLY)
    if state == ResidencyState.TILE_RESIDENT and e.tile_epochs.get(
        tile_id, -1) == e.epoch:
      # warm hit: program resident, epoch matches
      return 0
    # cold miss (or epoch mismatch after reset): install
    lat = self._cold_latency(e.hbm_bytes)
    e.tile_states[tile_id] = ResidencyState.TILE_RESIDENT
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

  def invalidate_tile(self, tile_id: int) -> None:
    """Invalidate all program residency for a tile (tile reset).

    Compute Tile 6.48: tile_soft_reset invalidates program_epoch.
    Next dispatch must re-install (cold miss).
    """
    for e in self._entries.values():
      e.tile_states.pop(tile_id, None)
      e.tile_epochs.pop(tile_id, None)

  def invalidate_group(self) -> None:
    """Invalidate all program residency for the group (group reset)."""
    for e in self._entries.values():
      e.epoch += 1  # bump epoch so warm hits become cold
      e.tile_states.clear()
      e.tile_epochs.clear()

  def evict(self, program_id: int, tile_id: int) -> None:
    e = self._entries.get(program_id)
    if e is not None:
      e.tile_states[tile_id] = ResidencyState.EVICTED

  def reset(self) -> None:
    self._entries.clear()
    self.cold_load_cycles = 0

  def snapshot(self) -> dict:
    return {
      pid: {
        "version": e.version,
        "epoch": e.epoch,
        "tile_states": {
          t: s.name for t, s in e.tile_states.items()},
      } for pid, e in self._entries.items()
    }
