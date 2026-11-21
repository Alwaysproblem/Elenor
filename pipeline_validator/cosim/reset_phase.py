"""RESET_PHASE command bridge: ResetState <-> wire reset_phase mapping (C024 / W1-02).

Maps the Group's `ResetState` FSM onto the CTRL-channel `RESET_PHASE_REQ`/`RESET_PHASE_ACK`
wire pair.  C034 owns the phase semantics: the ACK carries the MARK count
(`pending_to_reset`) and the DRAIN quiescence evidence bitmap (`drain_evidence`),
validated here against the sealed C017 schema.
"""

from __future__ import annotations

from ..runtime.reset_domain import ResetState
from . import wire
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

# C034 ACK shape: DRAIN reports every drain_evidence bit (W6-03: the quiescence
# evidence set; PENDING events are deliberately absent from it), every other
# stage reports none, and only MARK_EVENTS carries a pending_to_reset count.
DRAIN_PHASE = "DRAIN"
MARK_PHASE = "MARK_EVENTS"
DRAIN_EVIDENCE_NAMES: tuple[str, ...] = tuple(wire.BITSETS["drain_evidence"])
DRAIN_EVIDENCE_FULL_MASK = (1 << len(DRAIN_EVIDENCE_NAMES)) - 1


class ResetPhaseError(RuntimeError):
  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


def _normalize_phase(value: object) -> str | None:
  """Wire phases normalize from their enum name or encoded int index."""
  if isinstance(value, str):
    return value if value in WIRE_PHASES else None
  if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(WIRE_PHASES):
    return WIRE_PHASES[value]
  return None


def _ack_word(fields: dict, name: str) -> int:
  """One uint64 ACK word: a real int in [0, 2^64), never bool/str coercion."""
  value = fields.get(name)
  if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > wire.MASK64:
    raise ResetPhaseError("BAD_RESET_ACK", f"{name}={value!r}")
  return value


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

  def handle_ack(self, msg: BoundaryMessage) -> dict:
    """Validate and consume one RESET_PHASE_ACK; returns the observed evidence.

    Outstanding first, then bitmap/count, consumption last: an erroneous,
    early or duplicate ACK raises without consuming the owed request.
    """
    if msg.message_type != "RESET_PHASE_ACK" or msg.direction != "T2G":
      raise ResetPhaseError("BAD_RESET_ACK", f"{msg.message_type}/{msg.direction}")
    expected = {f[0] for f in wire.MESSAGES[wire.MSG_INDEX["RESET_PHASE_ACK"]]["fields"]}
    unknown = sorted(set(msg.fields) - expected)
    if unknown:
      # fail closed: unknown/extra fields never ride a sealed ACK
      raise ResetPhaseError("BAD_RESET_ACK", f"unknown fields {unknown}")
    phase = _normalize_phase(msg.fields.get("phase"))
    if phase is None:
      raise ResetPhaseError("BAD_RESET_ACK", f"phase={msg.fields.get('phase')!r}")
    generation = _ack_word(msg.fields, "run_generation")
    key = (phase, generation)
    if key not in self._pending:
      self.stats["acks_rejected"] += 1
      raise ResetPhaseError("UNSOLICITED_RESET_ACK", f"{key[0]}@{key[1]}")
    mask = _ack_word(msg.fields, "evidence_mask")
    marked = _ack_word(msg.fields, "pending_to_reset")
    if phase == DRAIN_PHASE:
      if mask != DRAIN_EVIDENCE_FULL_MASK:
        raise ResetPhaseError(
          "ACK_EVIDENCE_MASK",
          f"DRAIN requires 0x{DRAIN_EVIDENCE_FULL_MASK:x}, got 0x{mask:x}",
        )
    elif mask != 0:
      raise ResetPhaseError("ACK_EVIDENCE_MASK", f"{phase} carries no drain evidence, got 0x{mask:x}")
    if phase != MARK_PHASE and marked != 0:
      raise ResetPhaseError("ACK_MARK_COUNT", f"{phase} carries no pending_to_reset, got {marked}")
    self._pending.discard(key)
    self.stats["acks_accepted"] += 1
    return {
      "phase": phase,
      "run_generation": generation,
      "pending_to_reset": marked,
      "evidence_mask": mask,
      "evidence": {name: bool(mask >> i & 1) for i, name in enumerate(DRAIN_EVIDENCE_NAMES)},
    }

  def begin_run(self) -> None:
    """Drop outstanding requests from a previous reset run (stats keep)."""
    self._pending.clear()

  def pending(self) -> list[tuple[str, int]]:
    return sorted(self._pending)

  def snapshot(self) -> dict:
    return {"stats": dict(self.stats), "pending": self.pending()}
