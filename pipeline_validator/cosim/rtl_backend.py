"""Stub-mode RTL TilePort backend over the CN C ABI (C025 / W1-03, unit-only).

`backend_kind == "rtl"` in stub configuration (`CnBeginCfg.tile_count` served by
`cn_protocol_stub.sv`). This backend exists to drive the five-phase ABI discipline
(bp_drive -> bp_tick -> bp_collect) from the Python TilePort surface in unit tests; the REAL
Tile top replaces the stub in C040. It has NO compute path: execution stays with the model
behind the ABI.

CS1 rule: a backend whose underlying library is the STUB build may never be used as CS1
evidence (`tile_port.assert_cs1_backend_kind` rejects `backend_kind="stub"`).
"""

from __future__ import annotations

from pathlib import Path

from . import abi as cn_abi
from .tile_port import BackendKind, BoundaryMessage, ChannelCreditLedger


class RtlStubBackend:
  """TilePort surface over one `CnAbi` session driving `cn_protocol_stub.sv` (unit-only)."""

  def __init__(
    self,
    tile_id: int,
    *,
    ready_seed: int = 1,
    inject_fatal_at_tick: int = -1,
    library: Path | str | None = None,
    channel_capacities=None,
  ) -> None:
    self.tile_id_val = tile_id
    self.abi = cn_abi.load(Path(library) if library is not None else cn_abi.DEFAULT_STUB)
    self.credits = ChannelCreditLedger(channel_capacities)
    cfg = cn_abi.CnBeginCfg()
    cfg.abi_version = cn_abi.CN_ABI_VERSION
    cfg.tile_count = 1
    cfg.ready_seed = ready_seed & 0xFFFFFFFF
    cfg.inject_fatal_at_tick = inject_fatal_at_tick
    cfg.inject_abort_at_tick = -1
    self.abi.begin(cfg)
    self.slots = self.abi.layout()
    self._last_tick_posedge = self.status().posedge_count
    self._last_step_cycle: int | None = None

  @property
  def tile_id(self) -> int:
    return self.tile_id_val

  @property
  def backend_kind(self) -> BackendKind:
    return "rtl"

  def status(self) -> cn_abi.CnStatus:
    return self.abi.status()

  def drive_tick_collect(self, payload: bytes) -> tuple[int, bytes]:
    """One full phase cycle: P0 drive, P1-P3 tick, then collect. Returns (tick_rc, collected)."""
    rc = self.abi.drive([(0, payload)])
    if rc != cn_abi.CN_OK:
      return rc, b""
    trc = self.abi.tick()
    crc, data = self.abi.collect()
    if crc != cn_abi.CN_OK:
      return trc, b""
    return trc, data

  def step(self, cycle: int, *, freeze_new_work: bool = False) -> None:
    if self._last_step_cycle == cycle:
      raise RuntimeError(f"RtlStubBackend(tile={self.tile_id}) stepped twice in cycle {cycle}")
    self._last_step_cycle = cycle
    st = self.status()
    if st.posedge_count != self._last_tick_posedge:
      raise RuntimeError("posedge advanced without an explicit tick cycle")
    self._last_tick_posedge = st.posedge_count

  def reset(self) -> None:
    self.abi.destroy()
    self.abi = cn_abi.load(self.abi.path)
    cfg = cn_abi.CnBeginCfg()
    cfg.abi_version = cn_abi.CN_ABI_VERSION
    cfg.tile_count = 1
    cfg.ready_seed = 1
    cfg.inject_fatal_at_tick = -1
    cfg.inject_abort_at_tick = -1
    self.abi.begin(cfg)
    self.slots = self.abi.layout()
    self._last_tick_posedge = self.status().posedge_count

  def close(self) -> None:
    self.abi.destroy()

  def credit_invariant_holds(self) -> bool:
    return self.credits.conserved()

  def snapshot(self) -> dict:
    st = self.status()
    return {
      "tile_id": self.tile_id,
      "backend_kind": self.backend_kind,
      "tick_count": st.tick_count,
      "posedge_count": st.posedge_count,
      "fatal_code": st.fatal_code,
      "proto_err": st.proto_err,
    }

  # TilePort surface pieces not exercised by the stub link are explicit non-operations with the
  # reason recorded (they arrive with the real Tile top in C040).
  def send(self, msg: BoundaryMessage) -> bytes:  # pragma: no cover - C040 wiring
    raise NotImplementedError("stub backend drives the link via drive_tick_collect, not send()")

  def poll_inbound(self) -> bool:  # pragma: no cover - C040 wiring
    return False

  def pop_inbound(self):  # pragma: no cover - C040 wiring
    return None
