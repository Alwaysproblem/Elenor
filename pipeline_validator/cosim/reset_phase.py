"""RESET_PHASE command bridge: ResetState <-> wire reset_phase mapping (C024 / W1-02).

Maps the Group's `ResetState` FSM onto the CTRL-channel `RESET_PHASE_REQ`/`RESET_PHASE_ACK`
wire pair (W6-03 owns the semantics; this file only carries the mapping and the ACK shape).
"""

from __future__ import annotations

from ..runtime.reset_domain import ResetState
from .tile_port import BoundaryMessage

# Wire reset_phase enum order (wire.py ENUMS['reset_phase']).
WIRE_PHASES: tuple[str, ...] = (
  "FREEZE_ISSUE",
  "CANCEL_UNISSUED",
  "DRAIN",
  "MARK_EVENTS",
  "INVALIDATE_RESIDENCY",
  "REBUILD_CREDIT",
  "RESUME_DONE",
)

# ResetState (runtime/reset_domain.py) -> wire phase name. IDLE/FAULT_DETECTED precede the
# phased sequence and are not wire phases; DONE maps to the terminal RESUME_DONE phase.
STATE_TO_PHASE: dict[ResetState, str] = {
  ResetState.STOP_QUEUE: "FREEZE_ISSUE",
  ResetState.FREEZE_DISPATCH: "CANCEL_UNISSUED",
  ResetState.DRAIN_SAFE: "DRAIN",
  ResetState.MARK_EVENTS: "MARK_EVENTS",
  ResetState.RESET_DOMAIN: "INVALIDATE_RESIDENCY",
  ResetState.CLEAR_CREDIT_CACHE: "REBUILD_CREDIT",
  ResetState.RESUME: "RESUME_DONE",
  ResetState.DONE: "RESUME_DONE",
}


class ResetPhaseError(RuntimeError):
  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


class ResetPhaseService:
  """Produces RESET_PHASE_REQ from Group state and consumes RESET_PHASE_ACK."""

  def __init__(self) -> None:
    self.stats = {"requests_emitted": 0, "acks_accepted": 0, "acks_rejected": 0}
    self._pending: set[tuple[str, int]] = set()

  @staticmethod
  def phase_for(state: ResetState) -> str:
    try:
      return STATE_TO_PHASE[state]
    except KeyError as exc:
      raise ResetPhaseError("STATE_HAS_NO_WIRE_PHASE", str(state)) from exc

  def emit_request(self, state: ResetState, run_generation: int) -> BoundaryMessage:
    phase = self.phase_for(state)
    msg = BoundaryMessage(
      message_type="RESET_PHASE_REQ",
      channel="CTRL",
      direction="G2T",
      fields={"phase": phase, "run_generation": run_generation},
      message_seq=0,
    )
    _ = msg.pack_wire()
    self._pending.add((phase, run_generation))
    self.stats["requests_emitted"] += 1
    return msg

  def handle_ack(self, msg: BoundaryMessage) -> None:
    if msg.message_type != "RESET_PHASE_ACK" or msg.direction != "T2G":
      raise ResetPhaseError("BAD_RESET_ACK", f"{msg.message_type}/{msg.direction}")
    key = (str(msg.fields["phase"]), int(msg.fields["run_generation"]))
    if key not in self._pending:
      self.stats["acks_rejected"] += 1
      raise ResetPhaseError("UNSOLICITED_RESET_ACK", f"{key[0]}@{key[1]}")
    self._pending.discard(key)
    self.stats["acks_accepted"] += 1

  def pending(self) -> list[tuple[str, int]]:
    return sorted(self._pending)

  def snapshot(self) -> dict:
    return {"stats": dict(self.stats), "pending": self.pending()}
