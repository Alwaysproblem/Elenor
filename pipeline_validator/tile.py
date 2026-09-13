"""Compute Tile and Tile UCE controller.

The Tile UCE is the tile-local controller that interprets Tile Programs,
launches BOA/EVU/MFE/USE work, waits on local/external events, and performs
stream operations against shared Stream Queues.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from .config import MAX_CONTEXT_COUNT, HardwareConfig
from .engines import BOAEngine, Engine, EngineJob, EngineState, EVUEngine, MFEEngine, USEEngine
from .execution_ir import (
  ExecEngineDesc,
  ExecGatherDesc,
  ExecTileInst,
  ExecTileOp,
  ExecTileProgram,
  GridInstanceId,
  PhaseSignal,
  TaskIdentity,
)
from .memory import (
  AdmissionFailure,
  AdmissionFailureKind,
  AllocationHandle,
  AllocationPlan,
  AllocationRequest,
  AllocationState,
  MemoryInvariantError,
  ResolvedMemoryView,
  TaskBufferOwner,
)
from .memory.l1_slot_frame import SlotFrame
from .memory.l2_sram import L2SRAM
from .pmu import PMUCounter, StallReason
from .stream_queue import StreamQueue, StreamToken
from .trace import Tracer

_LAUNCH_OPS: frozenset[ExecTileOp] = frozenset(
  {
    ExecTileOp.LAUNCH_BOA,
    ExecTileOp.LAUNCH_EVU,
    ExecTileOp.LAUNCH_USE,
    ExecTileOp.LAUNCH_MFE,
    ExecTileOp.LAUNCH_GATHER,
  }
)


class _UCEContextState(Enum):
  EMPTY = "empty"
  ACCEPT = "accept"
  FRAME_BIND = "frame_bind"
  READY = "ready"
  WAIT_EVENT = "wait_event"
  WAIT_STREAM = "wait_stream"
  WAIT_ENGINE_QUEUE = "wait_engine_queue"
  DONE = "done"
  FAULT = "fault"


class _UCEHeadStatus(Enum):
  ELIGIBLE = "eligible"
  WAIT_EVENT = "wait_event"
  WAIT_ENGINE_QUEUE = "wait_engine_queue"
  WAIT_STREAM = "wait_stream"
  FAULT = "fault"


class _UCEEventScope(Enum):
  LOCAL = "local"
  EXTERNAL = "external"


@dataclass(frozen=True)
class _UCEEventRef:
  scope: _UCEEventScope
  owner_ctx: int | None
  local_name: str
  runtime_id: str
  expected_sequence: int = 0


@dataclass(frozen=True)
class _UCETerminalEvent:
  tile_id: int
  ctx_id: int
  role_id: int | None
  role_event_id: str | None
  status: str
  reason: str = ""


@dataclass
class _UCEContext:
  ctx_id: int
  state: _UCEContextState = _UCEContextState.EMPTY
  program: ExecTileProgram | None = None
  pc: int = 0
  role_id: int | None = None
  role_event_id: str | None = None
  wait_refs: tuple[_UCEEventRef, ...] = ()
  wait_all: bool = False
  tokens: dict[str, StreamToken] = field(default_factory=dict)
  events_done: set[str] = field(default_factory=set)
  event_records: dict[str, _UCEEventRef] = field(default_factory=dict)
  prepare_remaining: int = 0
  frame_bind_remaining: int = 0
  fault_reason: str = ""
  trace_slice_name: str | None = None
  prepare_total: int = 0
  memory: _TileContextMemory | None = None  # PR 2: per-context memory state
  task_identity: TaskIdentity | None = None  # PR 3: dispatch grid + logical task
  l1_live_names: set[str] = field(default_factory=set)


@dataclass
class _TileContextMemory:
  """Per-context resolved global/L2/L1 memory state.

  Global and L2 dictionaries are keyed by the original tile-program
  formal index, including task/global formals before the L2 prefix.
  ``l1_handles`` is the live name-to-handle map shared with TileGroup
  terminal/reset bookkeeping; explicit free removes the binding once.

  """

  task_identity: TaskIdentity
  l2_formal_handles: dict[int, AllocationHandle] = field(default_factory=dict)
  global_formal_views: dict[int, ResolvedMemoryView] = field(default_factory=dict)
  l1_handles: dict[str, AllocationHandle] = field(default_factory=dict)
  l2_resolver: L2SRAM | None = None


@dataclass
class TileAdmission:
  """One Tile's side-effect-free context and L1 admission plan."""

  tile: ComputeTile
  logical_task_id: int
  context_id: int
  l1_plan: AllocationPlan | None
  prepare_cycles: int = 0
  l1_handles: dict[str, AllocationHandle] = field(default_factory=dict)
  bound: bool = False


@dataclass
class _ResolvedEngineLaunch:
  """Result of ``_build_engine_launch``: desc + optional transaction."""

  desc: ExecEngineDesc
  transaction: object | None = None
  launch_params: dict = field(default_factory=dict)


@dataclass
class _EngineQueueEntry:
  ctx_id: int
  event_ref: _UCEEventRef
  desc_ref: str
  op: ExecTileOp
  engine_kind: str
  launch_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _UCEHeadProbe:
  status: _UCEHeadStatus
  reason: str = ""


_ACTIVE_CONTEXT_STATES = {
  _UCEContextState.ACCEPT,
  _UCEContextState.FRAME_BIND,
  _UCEContextState.READY,
  _UCEContextState.WAIT_EVENT,
  _UCEContextState.WAIT_STREAM,
  _UCEContextState.WAIT_ENGINE_QUEUE,
}


class TileUCE:
  """Tile Unified Control Engine with 1..MAX_CONTEXT_COUNT execution contexts."""

  def __init__(
    self,
    tile_id: int,
    cfg: HardwareConfig,
    tracer: Tracer | None = None,
    context_count: int = 1,
    runtime_enabled: bool = False,
  ):
    if context_count < 1 or context_count > MAX_CONTEXT_COUNT:
      raise ValueError("context_count must be between 1 and 8")
    self.tile_id = tile_id
    self.cfg = cfg
    self.tracer = tracer
    self.runtime_enabled = runtime_enabled
    self.pmu = PMUCounter()
    self.context_count = context_count
    self.contexts = [_UCEContext(ctx_id=i) for i in range(context_count)]
    self._current_ctx: int = 0
    self._rr_next_ctx: int = 0
    self._terminal_events: list[_UCETerminalEvent] = []
    self._local_event_owner: dict[str, _UCEEventRef] = {}
    self._external_events_done: set[str] = set()
    self._completion_queue: deque[_UCEEventRef] = deque()
    self._event_done_callback: Callable[[str], None] | None = None
    self._phase_signal_callback: Callable[[PhaseSignal, int], None] | None = None
    self._engine_queues: dict[str, deque[_EngineQueueEntry]] = {
      "BOA": deque(),
      "EVU": deque(),
      "MFE_LOAD": deque(),
      "MFE_STORE": deque(),
      "MFE_GATHER": deque(),
      "USE": deque(),
    }
    self._engine_queue_depths: dict[str, int] = {
      "BOA": 1,
      "EVU": 1,
      "USE": 1,
      "MFE_LOAD": cfg.mfe_load_queue_depth,
      "MFE_STORE": cfg.mfe_store_queue_depth,
      "MFE_GATHER": cfg.mfe_load_queue_depth,
    }

  def can_accept_context(self, context_id: int | None = None) -> bool:
    if context_id is None:
      return any(self._context_available(ctx) for ctx in self.contexts)
    return self._context_available(self.contexts[context_id])

  def available_context_id(self, preferred: int | None = None) -> int | None:
    """Return the exact context id a later atomic bind may consume."""
    candidates = self.contexts if preferred is None else (self.contexts[preferred],)
    return next((ctx.ctx_id for ctx in candidates if self._context_available(ctx)), None)

  def bind_context(
    self,
    program: ExecTileProgram,
    role_id: int | None,
    role_event_id: str | None,
    prepare_cycles: int = 0,
    context_id: int | None = None,
    memory: _TileContextMemory | None = None,
    task_identity: TaskIdentity | None = None,
  ) -> int | None:
    candidates = self.contexts if context_id is None else (self.contexts[context_id],)
    for ctx in candidates:
      if not self._context_available(ctx):
        continue
      self._unregister_context_events(ctx)
      ctx.program = program
      ctx.pc = 0
      ctx.role_id = role_id
      ctx.role_event_id = role_event_id
      ctx.wait_refs = ()
      ctx.wait_all = False
      ctx.tokens.clear()
      ctx.events_done.clear()
      ctx.event_records.clear()
      ctx.prepare_total = 1 + prepare_cycles if self.runtime_enabled else 0
      ctx.prepare_remaining = ctx.prepare_total
      ctx.frame_bind_remaining = self.cfg.frame_bind_cycles if self.runtime_enabled else 0
      ctx.memory = memory
      ctx.task_identity = task_identity
      ctx.l1_live_names = {buffer.name for buffer in program.l1_buffers}
      if self.runtime_enabled:
        ctx.state = _UCEContextState.ACCEPT
      else:
        ctx.state = _UCEContextState.READY
      return ctx.ctx_id
    return None

  def rollback_context(self, context_id: int) -> None:
    """Undo an atomic-dispatch bind before the context has executed."""
    ctx = self.contexts[context_id]
    self._unregister_context_events(ctx)
    for key, fifo in self._engine_queues.items():
      self._engine_queues[key] = deque(entry for entry in fifo if entry.ctx_id != context_id)
    self._reset_context(ctx)

  def step(self, cycle: int, tile: ComputeTile, freeze_new_work: bool = False) -> None:
    self.pmu.add_cycle("total", 1)
    self._ensure_context_traces(cycle)
    # Running engines may finish while reset drains. Apply those
    # completions, but do not start queued engines or advance the UCE PC.
    self._drain_completion_queue()
    if freeze_new_work:
      if self.has_active_contexts():
        self.pmu.add_cycle("reset_freeze", 1)
        self.pmu.add(StallReason.WAIT_EVENT, 1)
      else:
        self.pmu.add_cycle("idle", 1)
      self._sample_context_counters(cycle)
      return
    self._drain_engine_queues(cycle, tile)
    if self.faulted:
      self._sample_context_counters(cycle)
      return

    for ctx in self.contexts:
      if ctx.state == _UCEContextState.ACCEPT:
        self._advance_accept(ctx, cycle)
      elif ctx.state == _UCEContextState.FRAME_BIND:
        self._advance_frame_bind(ctx, cycle, tile)
    if self.faulted:
      self._sample_context_counters(cycle)
      return

    selected, selected_probe, probes = self._select_context(cycle, tile)
    self._record_blocked_heads(probes, cycle)
    if selected is not None and selected_probe is not None:
      if selected_probe.status == _UCEHeadStatus.FAULT:
        self._fault_context(selected, selected_probe.reason, cycle)
      else:
        self._wake_selected_context(selected, cycle)
        self._issue_context(selected, cycle, tile)
      self._sample_context_counters(cycle)
      return

    statuses = {probe.status for _ctx, probe in probes}
    if _UCEHeadStatus.WAIT_EVENT in statuses:
      self.pmu.add(StallReason.WAIT_EVENT, 1)
      self.pmu.add_cycle("wait_event", 1)
    elif _UCEHeadStatus.WAIT_ENGINE_QUEUE in statuses:
      self.pmu.add(StallReason.WAIT_OPERAND, 1)
      self.pmu.add_cycle("engine_queue_full", 1)
    elif _UCEHeadStatus.WAIT_STREAM in statuses:
      self.pmu.add(StallReason.STREAM_CREDIT, 1)
      self.pmu.add_cycle("wait_stream", 1)
    elif self.has_active_contexts():
      self.pmu.add(StallReason.NONE, 1)
    else:
      self.pmu.add_cycle("idle", 1)
    self._sample_context_counters(cycle)

  def notify_event(self, runtime_id: str) -> None:
    ref = self._local_event_owner.get(runtime_id)
    if ref is not None:
      self._completion_queue.append(ref)
      return
    if "ev_dma_" in runtime_id:
      self._external_events_done.add(runtime_id)
      return
    self.pmu.add_event("uce_unknown_event")

  def retire_external_events(self, events: set[str] | tuple[str, ...]) -> None:
    """Discard completed external events owned by a retired launch."""
    self._external_events_done.difference_update(events)

  def has_active_contexts(self) -> bool:
    return any(ctx.state in _ACTIVE_CONTEXT_STATES for ctx in self.contexts)

  def active_context_count(self) -> int:
    return sum(1 for ctx in self.contexts if ctx.state in _ACTIVE_CONTEXT_STATES)

  def ready_context_count(self) -> int:
    return sum(1 for ctx in self.contexts if ctx.state == _UCEContextState.READY)

  @property
  def faulted(self) -> bool:
    return any(ctx.state == _UCEContextState.FAULT for ctx in self.contexts)

  @property
  def fault_reason(self) -> str:
    for ctx in self.contexts:
      if ctx.fault_reason:
        return ctx.fault_reason
    return ""

  def drain_terminal_events(self) -> list[_UCETerminalEvent]:
    terms = list(self._terminal_events)
    self._terminal_events.clear()
    return terms

  def snapshot(self) -> dict:
    return {
      "context_count": self.context_count,
      "current_ctx": self._current_ctx,
      "rr_next_ctx": self._rr_next_ctx,
      "active_context_count": self.active_context_count(),
      "ready_context_count": self.ready_context_count(),
      "engine_queues": {
        key: [
          {
            "ctx_id": entry.ctx_id,
            "runtime_id": entry.event_ref.runtime_id,
            "local_name": entry.event_ref.local_name,
            "desc_ref": entry.desc_ref,
            "op": entry.op.value,
            "engine_kind": entry.engine_kind,
          }
          for entry in queue
        ]
        for key, queue in self._engine_queues.items()
      },
      "external_events_done": sorted(self._external_events_done),
      "contexts": [
        {
          "ctx_id": ctx.ctx_id,
          "state": ctx.state.value,
          "program": ctx.program.name if ctx.program is not None else None,
          "pc": ctx.pc,
          "role_id": ctx.role_id,
          "role_event_id": ctx.role_event_id,
          "wait_refs": [
            {"scope": ref.scope.value, "runtime_id": ref.runtime_id, "local_name": ref.local_name}
            for ref in ctx.wait_refs
          ],
          "events_done": sorted(ctx.events_done),
          "fault_reason": ctx.fault_reason,
        }
        for ctx in self.contexts
      ],
    }

  def reset(self) -> None:
    for ctx in self.contexts:
      self._close_context_trace(ctx, 0)
      self._reset_context(ctx)
    self._current_ctx = 0
    self._rr_next_ctx = 0
    self._terminal_events.clear()
    self._local_event_owner.clear()
    self._external_events_done.clear()
    self._completion_queue.clear()
    self._event_done_callback = None
    self._phase_signal_callback = None
    self._clear_engine_queues(unregister=False)
    self.pmu.reset()

  def _runtime_event_id(self, ctx: _UCEContext, local_name: str) -> str:
    return f"ctx{ctx.ctx_id}:{local_name}"

  def _make_local_event_ref(self, ctx: _UCEContext, local_name: str) -> _UCEEventRef:
    return _UCEEventRef(
      _UCEEventScope.LOCAL, ctx.ctx_id, local_name, self._runtime_event_id(ctx, local_name)
    )

  def _make_external_event_ref(self, local_name: str) -> _UCEEventRef:
    return _UCEEventRef(_UCEEventScope.EXTERNAL, None, local_name, local_name)

  def _make_wait_ref(self, ctx: _UCEContext, local_name: str, cycle: int) -> _UCEEventRef | None:
    if "ev_dma_" in local_name:
      return self._make_external_event_ref(local_name)
    ref = ctx.event_records.get(local_name)
    if ref is not None:
      return ref
    self._fault_context(ctx, f"wait references unknown event {local_name}", cycle)
    return None

  def _context_available(self, ctx: _UCEContext) -> bool:
    if ctx.state == _UCEContextState.EMPTY:
      return True
    if ctx.state != _UCEContextState.DONE:
      return False
    if any(ref.local_name not in ctx.events_done for ref in ctx.event_records.values()):
      return False
    return not any(entry.ctx_id == ctx.ctx_id for queue in self._engine_queues.values() for entry in queue)

  def _event_ref_done(self, ref: _UCEEventRef) -> bool:
    if ref.scope == _UCEEventScope.EXTERNAL:
      return ref.runtime_id in self._external_events_done
    assert ref.owner_ctx is not None
    owner = self.contexts[ref.owner_ctx]
    return ref.local_name in owner.events_done

  def _drain_completion_queue(self) -> None:
    while self._completion_queue:
      ref = self._completion_queue.popleft()
      self._apply_local_completion(ref)

  def _apply_local_completion(self, ref: _UCEEventRef) -> None:
    if ref.owner_ctx is None or ref.owner_ctx >= len(self.contexts):
      return
    owner = self.contexts[ref.owner_ctx]
    if ref.local_name in owner.events_done:
      return
    owner.events_done.add(ref.local_name)
    if self._event_done_callback is not None:
      self._event_done_callback(ref.runtime_id)

  def _advance_accept(self, ctx: _UCEContext, cycle: int) -> None:
    if ctx.prepare_remaining <= 0:
      next_state = _UCEContextState.FRAME_BIND if ctx.frame_bind_remaining > 0 else _UCEContextState.READY
      self._set_context_state(ctx, next_state, cycle)
      return
    if ctx.prepare_remaining == ctx.prepare_total:
      self.pmu.add_cycle("task_accept", 1)
    else:
      self.pmu.add_cycle("prepared_check", 1)
    ctx.prepare_remaining -= 1
    if ctx.prepare_remaining == 0:
      next_state = _UCEContextState.FRAME_BIND if ctx.frame_bind_remaining > 0 else _UCEContextState.READY
      self._set_context_state(ctx, next_state, cycle)

  def _advance_frame_bind(self, ctx: _UCEContext, cycle: int, tile: ComputeTile) -> None:
    self.pmu.add_cycle("frame_bind", 1)
    if ctx.frame_bind_remaining > 1:
      ctx.frame_bind_remaining -= 1
      return
    ctx.frame_bind_remaining = 0
    ok, _cycles = tile.l1_frames[ctx.ctx_id].bind(cycle, self.cfg.frame_bind_cycles)
    if not ok:
      self.pmu.add_event("l1_frame_fault")
      self._fault_context(ctx, "l1_frame_fault", cycle)
      return
    if tile.memory_trace is not None and self.tracer is not None:
      self.tracer.instant(
        f"Tile{self.tile_id}",
        "Lifecycle",
        "frame_bind",
        cycle,
        {
          "ctx_id": ctx.ctx_id,
          "generation": tile.l1_frames[ctx.ctx_id].generation,
          "tile_id": self.tile_id,
        },
      )
    self._set_context_state(ctx, _UCEContextState.READY, cycle)

  def _retry_wait_stream(self, ctx: _UCEContext, cycle: int, tile: ComputeTile) -> bool:
    if ctx.program is None or ctx.pc >= len(ctx.program.insts):
      return False
    ins = ctx.program.insts[ctx.pc]
    if ins.op == ExecTileOp.STREAM_POP:
      qid = ins.args[0]
      q = tile.get_stream(qid)
      tok = q.pop(cycle)
      if tok is None:
        return False
      assert ins.dst is not None
      ctx.tokens[ins.dst] = tok
      self.pmu.add_event("stream_pop")
      ctx.pc += 1
      return True
    if ins.op == ExecTileOp.STREAM_ACQUIRE:
      qid = ins.args[0]
      if qid < 0:
        ctx.pc += 1
        return True
      q = tile.get_stream(qid)
      if not q.acquire(cycle):
        return False
      assert ins.dst is not None
      ctx.tokens[ins.dst] = StreamToken(token_id=-1, producer_id=tile.tile_id)
      ctx.pc += 1
      return True
    return False

  def _wait_refs_ready(self, ctx: _UCEContext) -> bool:
    if not ctx.wait_refs:
      return True
    if ctx.wait_all:
      return all(self._event_ref_done(ref) for ref in ctx.wait_refs)
    return any(self._event_ref_done(ref) for ref in ctx.wait_refs)

  def _probe_wait_instruction(self, ctx: _UCEContext, ins: ExecTileInst) -> _UCEHeadProbe:
    names = (ins.args[0],) if ins.op == ExecTileOp.WAIT else tuple(ins.args)
    done: list[bool] = []
    for name in names:
      if "ev_dma_" in name:
        done.append(name in self._external_events_done)
        continue
      ref = ctx.event_records.get(name)
      if ref is None:
        return _UCEHeadProbe(_UCEHeadStatus.FAULT, f"wait references unknown event {name}")
      done.append(self._event_ref_done(ref))
    ready = all(done) if ins.op == ExecTileOp.WAITALL else any(done)
    return _UCEHeadProbe(_UCEHeadStatus.ELIGIBLE if ready else _UCEHeadStatus.WAIT_EVENT)

  def _probe_head(self, ctx: _UCEContext, tile: ComputeTile) -> _UCEHeadProbe | None:
    if ctx.state not in (
      _UCEContextState.READY,
      _UCEContextState.WAIT_EVENT,
      _UCEContextState.WAIT_STREAM,
      _UCEContextState.WAIT_ENGINE_QUEUE,
    ):
      return None
    if ctx.program is None:
      return _UCEHeadProbe(_UCEHeadStatus.FAULT, "active context has no tile program")
    if ctx.state == _UCEContextState.WAIT_EVENT and not self._wait_refs_ready(ctx):
      return _UCEHeadProbe(_UCEHeadStatus.WAIT_EVENT)
    if ctx.pc >= len(ctx.program.insts):
      return _UCEHeadProbe(_UCEHeadStatus.ELIGIBLE)

    ins = ctx.program.insts[ctx.pc]
    if ins.op in (ExecTileOp.WAIT, ExecTileOp.WAITALL):
      return self._probe_wait_instruction(ctx, ins)
    if ins.op in _LAUNCH_OPS:
      try:
        self._assert_launch_l1_live(ctx, ins)
        queue_key = self._launch_queue_key(ctx, ins)
      except (KeyError, MemoryInvariantError, ValueError) as exc:
        return _UCEHeadProbe(_UCEHeadStatus.FAULT, str(exc))
      if ins.dst is None:
        return _UCEHeadProbe(_UCEHeadStatus.FAULT, "engine launch has no completion event")
      if ins.dst in ctx.event_records:
        return _UCEHeadProbe(_UCEHeadStatus.FAULT, f"duplicate completion event '{ins.dst}'")
      if len(self._engine_queues[queue_key]) >= self._engine_queue_depths[queue_key]:
        return _UCEHeadProbe(_UCEHeadStatus.WAIT_ENGINE_QUEUE)
      return _UCEHeadProbe(_UCEHeadStatus.ELIGIBLE)
    if ins.op in (ExecTileOp.STREAM_POP, ExecTileOp.STREAM_ACQUIRE):
      qid = ins.args[0]
      if qid < 0 and ins.op == ExecTileOp.STREAM_ACQUIRE:
        return _UCEHeadProbe(_UCEHeadStatus.ELIGIBLE)
      try:
        queue = tile.get_stream(qid)
      except KeyError:
        return _UCEHeadProbe(_UCEHeadStatus.FAULT, f"unknown stream queue {qid}")
      eligible = (
        not queue.is_empty if ins.op == ExecTileOp.STREAM_POP else not queue.is_full and not queue._faulted
      )
      return _UCEHeadProbe(_UCEHeadStatus.ELIGIBLE if eligible else _UCEHeadStatus.WAIT_STREAM)
    return _UCEHeadProbe(_UCEHeadStatus.ELIGIBLE)

  def _select_context(
    self, cycle: int, tile: ComputeTile
  ) -> tuple[_UCEContext | None, _UCEHeadProbe | None, list[tuple[_UCEContext, _UCEHeadProbe]]]:
    probes: list[tuple[_UCEContext, _UCEHeadProbe]] = []
    for offset in range(self.context_count):
      idx = (self._rr_next_ctx + offset) % self.context_count
      ctx = self.contexts[idx]
      probe = self._probe_head(ctx, tile)
      if probe is None:
        continue
      probes.append((ctx, probe))
      if probe.status not in (_UCEHeadStatus.ELIGIBLE, _UCEHeadStatus.FAULT):
        continue
      if idx != self._current_ctx:
        self._switch_context(self._current_ctx, idx, cycle, reason="eligible_head_rr")
      self._current_ctx = idx
      self._rr_next_ctx = (idx + 1) % self.context_count
      return ctx, probe, probes
    return None, None, probes

  def _record_blocked_heads(self, probes: list[tuple[_UCEContext, _UCEHeadProbe]], cycle: int) -> None:
    state_by_status = {
      _UCEHeadStatus.WAIT_ENGINE_QUEUE: _UCEContextState.WAIT_ENGINE_QUEUE,
      _UCEHeadStatus.WAIT_STREAM: _UCEContextState.WAIT_STREAM,
    }
    for ctx, probe in probes:
      if ctx.state == _UCEContextState.WAIT_EVENT and self._wait_refs_ready(ctx):
        ctx.wait_refs = ()
        ctx.wait_all = False
        self._set_context_state(ctx, _UCEContextState.READY, cycle)
      blocked_state = state_by_status.get(probe.status)
      if ctx.state == _UCEContextState.READY and blocked_state is not None:
        self._set_context_state(ctx, blocked_state, cycle)

  def _wake_selected_context(self, ctx: _UCEContext, cycle: int) -> None:
    if ctx.state == _UCEContextState.WAIT_EVENT:
      ctx.wait_refs = ()
      ctx.wait_all = False
    if ctx.state in (
      _UCEContextState.WAIT_EVENT,
      _UCEContextState.WAIT_STREAM,
      _UCEContextState.WAIT_ENGINE_QUEUE,
    ):
      self._set_context_state(ctx, _UCEContextState.READY, cycle)

  def _trace_issue(self, ctx: _UCEContext, ins: ExecTileInst, cycle: int) -> None:
    if self.tracer is None:
      return
    issue_args = {
      "ctx_id": ctx.ctx_id,
      "program": ctx.program.name if ctx.program is not None else None,
      "pc": ctx.pc,
      "op": ins.op.value,
    }
    if ins.dst is not None:
      issue_args["event_id"] = ins.dst
    self.tracer.instant(f"Tile{self.tile_id}", f"UCE CTX{ctx.ctx_id}", "uce_issue", cycle, issue_args)

  def _switch_context(self, from_ctx: int, to_ctx: int, cycle: int, reason: str) -> None:
    self.pmu.add_event("uce_context_switch")
    if self.tracer is not None:
      self.tracer.instant(
        f"Tile{self.tile_id}",
        f"UCE CTX{to_ctx}",
        "ctx_switch",
        cycle,
        {
          "from_ctx": from_ctx,
          "to_ctx": to_ctx,
          "reason": reason,
          "active_context_count": self.active_context_count(),
          "ready_context_count": self.ready_context_count(),
        },
      )

  def _issue_context(self, ctx: _UCEContext, cycle: int, tile: ComputeTile) -> None:
    assert ctx.program is not None
    if ctx.pc >= len(ctx.program.insts):
      self._complete_context(ctx, cycle)
      return

    ins = ctx.program.insts[ctx.pc]
    self._trace_issue(ctx, ins, cycle)

    prog = ctx.program
    op = ins.op
    if op == ExecTileOp.NOP:
      ctx.pc += 1
    elif op in (
      ExecTileOp.MOV,
      ExecTileOp.ADD,
      ExecTileOp.CMP,
      ExecTileOp.LOAD_DESC,
      ExecTileOp.STORE_DESC,
      ExecTileOp.PROF_BEGIN,
      ExecTileOp.PROF_END,
    ):
      ctx.pc += 1
    elif op in (ExecTileOp.BR, ExecTileOp.BRP):
      ctx.pc = prog.label_index(ins.args[0])
    elif op == ExecTileOp.BR_EOS:
      tok = ctx.tokens.get(ins.args[0])
      if tok is not None and tok.is_eos:
        ctx.pc = prog.label_index(ins.args[1])
      else:
        ctx.pc += 1
    elif op == ExecTileOp.WAIT:
      self._issue_wait(ctx, cycle, (ins.args[0],), wait_all=False)
    elif op == ExecTileOp.WAITALL:
      self._issue_wait(ctx, cycle, tuple(ins.args), wait_all=True)
    elif op == ExecTileOp.FENCE:
      self.pmu.add_cycle("fence", 1)
      ctx.pc += 1
    elif op == ExecTileOp.SIGNAL_PHASE:
      phase = ins.args[0]
      if ctx.task_identity is not None:
        signal = PhaseSignal(task=ctx.task_identity, phase=phase)
        if self._phase_signal_callback is not None:
          self._phase_signal_callback(signal, cycle)
        if self.tracer is not None:
          grid = ctx.task_identity.grid
          self.tracer.instant(
            f"Tile{self.tile_id}",
            f"UCE CTX{ctx.ctx_id}",
            "tile_signal",
            cycle,
            {
              "context_name": grid.context_name,
              "device_slot": grid.device_slot,
              "launch_generation": grid.launch_generation,
              "dispatch_ordinal": grid.dispatch_ordinal,
              "task_id": ctx.task_identity.task_id,
              "phase": phase,
              "tile_id": self.tile_id,
              "hardware_context_id": ctx.ctx_id,
            },
          )
      self.pmu.add_event("tile_signal")
      ctx.pc += 1
    elif op == ExecTileOp.FREE_L1:
      try:
        self._free_l1(ctx, ins, cycle, tile)
      except MemoryInvariantError as exc:
        self._fault_context(ctx, f"tile.free: {exc}", cycle)
      else:
        ctx.pc += 1
    elif op in _LAUNCH_OPS:
      try:
        self._assert_launch_l1_live(ctx, ins)
      except MemoryInvariantError as exc:
        self._fault_context(ctx, f"tile.free: {exc}", cycle)
      else:
        self._issue_engine_launch(ctx, self._launch_queue_key(ctx, ins), ins, cycle)
    elif op == ExecTileOp.STREAM_POP:
      if not self._retry_wait_stream(ctx, cycle, tile):
        self.pmu.add(StallReason.STREAM_CREDIT, 1)
        self.pmu.add_cycle("stream_empty", 1)
        self._set_context_state(ctx, _UCEContextState.WAIT_STREAM, cycle)
    elif op == ExecTileOp.STREAM_ACQUIRE:
      if not self._retry_wait_stream(ctx, cycle, tile):
        self.pmu.add(StallReason.STREAM_CREDIT, 1)
        self.pmu.add_cycle("stream_full", 1)
        self._set_context_state(ctx, _UCEContextState.WAIT_STREAM, cycle)
    elif op == ExecTileOp.STREAM_PUSH:
      qid, tok_reg, producer_id = ins.args
      if qid >= 0:
        q = tile.get_stream(qid)
        tok = ctx.tokens.get(tok_reg)
        if tok is None:
          tok = StreamToken(token_id=q._next_token_id, producer_id=producer_id)
        tok.producer_id = producer_id
        tok.token_id = q._next_token_id
        q._next_token_id += 1
        q.push(tok, cycle)
        self.pmu.add_event("stream_push")
      ctx.pc += 1
    elif op == ExecTileOp.STREAM_RELEASE:
      qid, tok_reg = ins.args
      if qid >= 0:
        q = tile.get_stream(qid)
        tok = ctx.tokens.get(tok_reg)
        if tok is not None:
          q.release(tok, cycle)
          del ctx.tokens[tok_reg]
          self.pmu.add_event("stream_release")
      ctx.pc += 1
    elif op == ExecTileOp.STREAM_PUSH_EOS:
      qid, producer_id = ins.args
      if qid >= 0:
        q = tile.get_stream(qid)
        q.push_eos(producer_id, cycle)
        self.pmu.add_event("stream_eos")
      ctx.pc += 1
    elif op == ExecTileOp.PATCH_DESC:
      self.pmu.add_cycle("patch_desc", 1)
      ctx.pc += 1
    elif op == ExecTileOp.TRAP:
      self._fault_context(ctx, "trap", cycle)
    elif op == ExecTileOp.RET:
      self._complete_context(ctx, cycle)
    else:
      ctx.pc += 1

  @staticmethod
  def _descriptor_l1_bases(program: ExecTileProgram, desc_ref: str) -> tuple[frozenset[str], bool]:
    """Return explicit L1 bases and whether a descriptor is opaque."""
    desc = program.descriptors.get(desc_ref)
    if desc is None:
      raise MemoryInvariantError("instruction references an unknown descriptor")
    bases: set[str] = set()
    if desc.transfer is not None:
      if desc.transfer.src.space == "l1":
        bases.add(desc.transfer.src.base)
      if desc.transfer.dst.space == "l1":
        bases.add(desc.transfer.dst.base)
      return frozenset(bases), False
    gather = desc.params.get("gather")
    if isinstance(gather, ExecGatherDesc):
      for view in (gather.source, gather.indices, gather.destination):
        if view.space == "l1":
          bases.add(view.base)
      return frozenset(bases), False
    if desc.op == "gather":
      raise MemoryInvariantError("gather descriptor is missing")
    return frozenset(), True

  def _assert_launch_l1_live(self, ctx: _UCEContext, ins: ExecTileInst) -> None:
    """Reject raw execution paths that access an explicitly freed L1."""
    if ctx.program is None or not ins.args or not isinstance(ins.args[0], str):
      raise MemoryInvariantError("launch has an invalid descriptor reference")
    if len(ctx.l1_live_names) == len(ctx.program.l1_buffers):
      return
    bases, _opaque = self._descriptor_l1_bases(ctx.program, ins.args[0])
    for base in bases:
      if base not in ctx.l1_live_names:
        raise MemoryInvariantError(f"use-after-free of L1 buffer '{base}'")
      if self.runtime_enabled and (ctx.memory is None or base not in ctx.memory.l1_handles):
        raise MemoryInvariantError(f"missing physical L1 binding for '{base}'")

  def _assert_free_dependencies_complete(self, ctx: _UCEContext, buffer_name: str) -> None:
    """Check prior queued/active accesses without adding hot-path state."""
    assert ctx.program is not None
    relevant_events: set[str] = set()
    for prior in ctx.program.insts[: ctx.pc]:
      if prior.op not in _LAUNCH_OPS:
        continue
      if prior.dst is None:
        continue
      event_ref = ctx.event_records.get(prior.dst)
      if event_ref is None:
        # Private Exec control flow may skip a textual predecessor.
        continue
      if not prior.args or not isinstance(prior.args[0], str):
        raise MemoryInvariantError("prior launch has an invalid descriptor reference")
      bases, opaque = self._descriptor_l1_bases(ctx.program, prior.args[0])
      if not opaque and buffer_name not in bases:
        continue
      if prior.dst in relevant_events:
        raise MemoryInvariantError(f"ambiguous duplicate completion event '{prior.dst}'")
      relevant_events.add(prior.dst)
      if (
        event_ref.scope != _UCEEventScope.LOCAL
        or event_ref.owner_ctx != ctx.ctx_id
        or event_ref.runtime_id != self._runtime_event_id(ctx, prior.dst)
      ):
        raise MemoryInvariantError(f"prior access event '{prior.dst}' is not owned by this context")
      if prior.dst not in ctx.events_done:
        raise MemoryInvariantError(f"L1 buffer '{buffer_name}' has pending event '{prior.dst}'")

  def _free_l1(self, ctx: _UCEContext, ins: ExecTileInst, cycle: int, tile: ComputeTile) -> None:
    """Preflight, final-free, then invalidate one L1 program binding."""
    if len(ins.args) != 1 or not isinstance(ins.args[0], str) or ctx.program is None:
      raise MemoryInvariantError("requires one L1 buffer name")
    buffer_name = ins.args[0]
    slot_ids = [
      slot_id for slot_id, buffer in enumerate(ctx.program.l1_buffers) if buffer.name == buffer_name
    ]
    if len(slot_ids) != 1:
      raise MemoryInvariantError(f"L1 buffer '{buffer_name}' is not a unique program allocation")
    if buffer_name not in ctx.l1_live_names:
      raise MemoryInvariantError(f"double free of L1 buffer '{buffer_name}'")

    self._assert_free_dependencies_complete(ctx, buffer_name)
    if not self.runtime_enabled:
      ctx.l1_live_names.remove(buffer_name)
      self.pmu.add_event("l1_free")
      return

    if ctx.memory is None or ctx.task_identity is None:
      raise MemoryInvariantError("physical L1 context identity is missing")
    if ctx.memory.task_identity != ctx.task_identity:
      raise MemoryInvariantError("physical L1 TaskIdentity does not match context")
    if ctx.role_event_id is None:
      raise MemoryInvariantError("physical L1 role event is missing")
    handle = ctx.memory.l1_handles.get(buffer_name)
    if handle is None or handle.memory_space != "l1":
      raise MemoryInvariantError(f"missing physical L1 binding for '{buffer_name}'")

    task = ctx.task_identity
    expected_owner = TaskBufferOwner(
      task.grid.context_name,
      task.grid.launch_generation,
      ctx.role_event_id,
      task.task_id,
      self.tile_id,
      ctx.ctx_id,
      buffer_name,
    )
    tile.l1_allocator.assert_live(handle, expected_owner)
    record = tile.l1_allocator._live.get(handle.allocation_id)
    if record is None or record.handle != handle:
      raise MemoryInvariantError("stale allocation generation")
    if record.state != AllocationState.LIVE:
      raise MemoryInvariantError(f"double free of L1 buffer '{buffer_name}'")
    if record.pins:
      raise MemoryInvariantError(f"L1 buffer '{buffer_name}' still has allocator pins")

    slot_id = slot_ids[0]
    frame = tile.l1_frames[ctx.ctx_id]
    frame.assert_slot_binding(slot_id, handle)
    transfer_manager = tile.mfe.transfer_manager
    if transfer_manager is None:
      raise MemoryInvariantError("physical L1 transfer manager is missing")
    if transfer_manager.has_inflight_access(handle):
      raise MemoryInvariantError(f"L1 buffer '{buffer_name}' has an in-flight transfer")

    if not tile.l1_allocator.request_release(handle, expected_owner, cycle):
      raise MemoryInvariantError(f"L1 buffer '{buffer_name}' did not final-free")
    frame.release_slot(slot_id, handle)
    del ctx.memory.l1_handles[buffer_name]
    ctx.l1_live_names.remove(buffer_name)
    self.pmu.add_event("l1_free")

  def _issue_wait(self, ctx: _UCEContext, cycle: int, event_names: tuple[str, ...], wait_all: bool) -> None:
    refs: list[_UCEEventRef] = []
    for event_name in event_names:
      ref = self._make_wait_ref(ctx, event_name, cycle)
      if ref is None:
        return
      refs.append(ref)
    ctx.pc += 1
    ctx.wait_refs = tuple(refs)
    ctx.wait_all = wait_all
    if self._wait_refs_ready(ctx):
      ctx.wait_refs = ()
      ctx.wait_all = False
      return
    self._set_context_state(ctx, _UCEContextState.WAIT_EVENT, cycle)

  def _launch_queue_key(self, ctx: _UCEContext, ins: ExecTileInst) -> str:
    if ins.op == ExecTileOp.LAUNCH_BOA:
      return "BOA"
    if ins.op == ExecTileOp.LAUNCH_EVU:
      return "EVU"
    if ins.op == ExecTileOp.LAUNCH_USE:
      return "USE"
    if ins.op == ExecTileOp.LAUNCH_GATHER:
      return "MFE_GATHER"
    desc_ref = ins.args[0] if ins.args else ""
    desc = ctx.program.descriptors.get(desc_ref) if ctx.program is not None else None
    if desc is not None and desc.op in ("store", "dma_store"):
      return "MFE_STORE"
    return "MFE_LOAD"

  def _issue_engine_launch(self, ctx: _UCEContext, queue_key: str, ins: ExecTileInst, cycle: int) -> None:
    if self.tracer is not None:
      self.tracer.instant(
        f"Tile{self.tile_id}",
        f"UCE CTX{ctx.ctx_id}",
        "uce_launch_attempt",
        cycle,
        {"ctx_id": ctx.ctx_id, "pc": ctx.pc, "queue": queue_key, "event_id": ins.dst},
      )
    if not self._enqueue_engine_launch(ctx, queue_key, ins, cycle):
      self._set_context_state(ctx, _UCEContextState.WAIT_ENGINE_QUEUE, cycle)

  def _enqueue_engine_launch(self, ctx: _UCEContext, queue_key: str, ins: ExecTileInst, cycle: int) -> bool:
    fifo = self._engine_queues[queue_key]
    if len(fifo) >= self._engine_queue_depths[queue_key]:
      self.pmu.add(StallReason.WAIT_OPERAND, 1)
      self.pmu.add_cycle("engine_queue_full", 1)
      return False
    assert ins.dst is not None
    local_name = ins.dst
    event_ref = self._make_local_event_ref(ctx, local_name)
    ctx.event_records[local_name] = event_ref
    self._local_event_owner[event_ref.runtime_id] = event_ref
    desc_ref = ins.args[0] if ins.args else ""
    launch_params: dict = {}
    engine_kind = "MFE" if queue_key in ("MFE_LOAD", "MFE_GATHER", "MFE_STORE") else queue_key
    fifo.append(
      _EngineQueueEntry(ctx.ctx_id, event_ref, desc_ref, ins.op, engine_kind, launch_params=launch_params)
    )
    if self.tracer is not None:
      self.tracer.instant(
        f"Tile{self.tile_id}",
        f"UCE CTX{ctx.ctx_id}",
        "uce_launch_accepted",
        cycle,
        {
          "ctx_id": ctx.ctx_id,
          "pc": ctx.pc,
          "queue": queue_key,
          "event_id": local_name,
          "runtime_id": event_ref.runtime_id,
        },
      )
    ctx.pc += 1
    return True

  def _drain_engine_queues(self, cycle: int, tile: ComputeTile) -> None:
    from .memory.allocator import MemoryInvariantError

    for queue_key in ("BOA", "EVU", "MFE_LOAD", "MFE_GATHER", "MFE_STORE", "USE"):
      fifo = self._engine_queues[queue_key]
      if not fifo:
        continue
      entry = fifo[0]
      ctx = self.contexts[entry.ctx_id]
      if ctx.program is None:
        fifo.popleft()
        self._drop_event_ref(entry.event_ref)
        continue
      engine = self._engine_for_queue(tile, queue_key)
      try:
        resolved = self._build_engine_launch(ctx, entry, tile)
        job: object | None
        if queue_key == "MFE_GATHER":
          job = tile.mfe.launch_gather(
            resolved.desc, cycle, entry.event_ref.runtime_id, **resolved.launch_params
          )
        else:
          is_mfe = queue_key in ("MFE_LOAD", "MFE_STORE")
          job = engine.launch(
            resolved.desc,
            cycle,
            entry.event_ref.runtime_id,
            transaction=resolved.transaction if is_mfe else None,
          )
      except (ValueError, MemoryInvariantError) as exc:
        fifo.popleft()
        self._drop_event_ref(entry.event_ref)
        self._fault_context(ctx, str(exc), cycle)
        self._clear_engine_queues(unregister=True)
        return
      if job is None:
        continue
      fifo.popleft()

  def _engine_for_queue(self, tile: ComputeTile, queue_key: str) -> Engine:
    if queue_key == "BOA":
      return tile.boa
    if queue_key == "EVU":
      return tile.evu
    if queue_key in ("MFE_LOAD", "MFE_GATHER", "MFE_STORE"):
      return tile.mfe
    return tile.use

  def _build_engine_launch(
    self, ctx: _UCEContext, entry: _EngineQueueEntry, tile: ComputeTile
  ) -> _ResolvedEngineLaunch:
    """Build the ExecEngineDesc + optional MemoryTransaction for this launch.

    For MFE load/store (desc.transfer is not None): build a real
    ``MemoryTransaction`` with resolved src/dst views from the context
    memory.  For timing_only contexts (ctx.memory is None) the
    transaction carries src/dst=None so the manager uses the collapsed
    path.
    """
    assert ctx.program is not None
    base = ctx.program.descriptors[entry.desc_ref]
    desc = ExecEngineDesc(base.name, base.kind, base.op, dict(base.params))
    transaction = None
    launch_params: dict = {}
    if base.transfer is not None:
      desc.params["bytes"] = base.transfer.bytes
      transaction = self._build_tile_transaction(ctx, base.transfer, entry, tile)
    elif base.op == "gather":
      gather = base.params.get("gather")
      if not isinstance(gather, ExecGatherDesc):
        raise ValueError("gather descriptor is missing")
      if gather.cache_target_bytes > self.cfg.l1_cache_capacity_bytes:
        raise ValueError("gather cache_target_bytes exceeds L1 cache capacity")
      if gather.l1_mshr_hint > self.cfg.l1_mshr_entries:
        raise ValueError("gather l1_mshr_hint exceeds L1 MSHR capacity")
      source = None
      indices = None
      destination = None
      if tile.runtime_enabled:
        if ctx.memory is None:
          raise ValueError("gather context memory is missing")
        source = self._resolve_tile_view(gather.source, ctx.memory, tile)
        indices = self._resolve_tile_view(gather.indices, ctx.memory, tile)
        destination = self._resolve_tile_view(gather.destination, ctx.memory, tile)
      from .memory.allocator import TaskBufferOwner

      task = ctx.task_identity
      grid = task.grid if task is not None else None
      owner = (
        destination.handle.owner
        if destination is not None
        else TaskBufferOwner(
          grid.context_name if grid is not None else "ctx",
          grid.launch_generation if grid is not None else 0,
          ctx.role_event_id or "ev",
          task.task_id if task is not None else 0,
          self.tile_id,
          ctx.ctx_id,
          gather.destination.base,
        )
      )
      launch_params = {
        "source": source,
        "indices": indices,
        "destination": destination,
        "issuer": owner,
        "namespace": (
          grid.launch_generation if grid is not None else 0,
          grid.dispatch_ordinal if grid is not None else 0,
          task.task_id if task is not None else 0,
          ctx.ctx_id,
          entry.event_ref.local_name,
        ),
      }
    desc.params = {
      **desc.params,
      "tile_id": self.tile_id,
      "ctx_id": ctx.ctx_id,
      "program": ctx.program.name,
      "local_event_id": entry.event_ref.local_name,
    }
    return _ResolvedEngineLaunch(desc=desc, transaction=transaction, launch_params=launch_params)

  def _build_tile_transaction(
    self, ctx: _UCEContext, transfer, entry: _EngineQueueEntry, tile: ComputeTile
  ):
    """Build a tile-local MemoryTransaction for an MFE load/store."""
    from .memory.transfer import MemoryTransaction, TransferOp

    op = TransferOp.TILE_LOAD if transfer.src.space == "l2" else TransferOp.TILE_STORE
    role_ev = ctx.role_event_id or "ev"
    task = ctx.task_identity
    if ctx.memory is not None:
      if task is not None and ctx.memory.task_identity != task:
        raise MemoryInvariantError("tile transaction TaskIdentity does not match context")
      task = ctx.memory.task_identity
    logical_task = task.task_id if task is not None else 0
    gen = task.grid.launch_generation if task is not None else 0
    txn_id = f"{gen}:{role_ev}:t{logical_task}:{entry.event_ref.local_name}"
    src = None
    dst = None
    if tile.runtime_enabled:
      if ctx.memory is None or task is None:
        raise ValueError("tile context memory identity is missing")
      src = self._resolve_tile_view(transfer.src, ctx.memory, tile)
      dst = self._resolve_tile_view(transfer.dst, ctx.memory, tile)
      l1_endpoint = dst if op == TransferOp.TILE_LOAD else src
      owner = l1_endpoint.handle.owner
      if (
        not isinstance(owner, TaskBufferOwner)
        or owner.context_name != task.grid.context_name
        or owner.context_launch_generation != task.grid.launch_generation
        or owner.role_event_id != role_ev
        or owner.logical_task_id != task.task_id
        or owner.physical_tile_id != self.tile_id
        or owner.hardware_context_id != ctx.ctx_id
      ):
        raise MemoryInvariantError("tile transaction L1 endpoint has an invalid owner")
    else:
      owner = TaskBufferOwner(
        task.grid.context_name if task is not None else "ctx",
        gen,
        role_ev,
        logical_task,
        self.tile_id,
        ctx.ctx_id,
        "task",
      )
    return MemoryTransaction(
      transaction_id=txn_id,
      op=op,
      issuer=owner,
      src=src,
      dst=dst,
      bytes_total=transfer.bytes,
      completion_event=entry.event_ref.runtime_id,
      tile_id=self.tile_id,
    )

  @staticmethod
  def _resolve_tile_view(view, memory, tile):
    """Resolve one lowered formal/L1 view through its live allocator."""
    from .memory.allocator import MemoryInvariantError
    from .memory.transfer import ResolvedMemoryView

    if view.space == "global":
      if not view.base.startswith("formal:"):
        raise MemoryInvariantError("global view has invalid formal base")
      formal_index = int(view.base.removeprefix("formal:"))
      resolved = memory.global_formal_views.get(formal_index)
      if not isinstance(resolved, ResolvedMemoryView):
        raise MemoryInvariantError("missing global actual for tile formal")
      if view.bytes > resolved.size_bytes:
        raise MemoryInvariantError("memory view out of bounds")
      return resolved

    handle = None
    resolver = None
    if view.space == "l2":
      if not view.base.startswith("formal:"):
        raise MemoryInvariantError("l2 view has invalid formal base")
      formal_index = int(view.base.removeprefix("formal:"))
      handle = memory.l2_formal_handles.get(formal_index)
      resolver = memory.l2_resolver
    elif view.space == "l1":
      handle = memory.l1_handles.get(view.base)
      resolver = tile.l1_allocator
    else:
      raise MemoryInvariantError(f"unsupported tile memory space '{view.space}'")
    if handle is None or resolver is None:
      raise MemoryInvariantError("missing or stale tile memory handle")
    offsets = list(view.offsets)
    if view.task_dim is not None and view.task_dim < len(offsets):
      offsets[view.task_dim] += memory.task_identity.task_id
    element_offset = 0
    for i, off in enumerate(offsets):
      stride = 1
      for dimension in view.backing_dims[i + 1 :]:
        stride *= dimension
      element_offset += off * stride
    offset_bytes = element_offset * view.element_bytes
    segments = resolver.resolve_segments(handle, offset_bytes, view.bytes)
    return ResolvedMemoryView(
      handle=handle,
      offset_bytes=offset_bytes,
      size_bytes=view.bytes,
      address=segments[0].address,
      segments=segments,
    )

  def _complete_context(self, ctx: _UCEContext, cycle: int) -> None:
    if ctx.state in (_UCEContextState.DONE, _UCEContextState.FAULT):
      return
    self.pmu.add_event("tile_done")
    self._terminal_events.append(
      _UCETerminalEvent(
        tile_id=self.tile_id,
        ctx_id=ctx.ctx_id,
        role_id=ctx.role_id,
        role_event_id=ctx.role_event_id,
        status="done",
      )
    )
    self._set_context_state(ctx, _UCEContextState.DONE, cycle)
    self._close_context_trace(ctx, cycle)

  def _fault_context(self, ctx: _UCEContext, reason: str, cycle: int) -> None:
    if ctx.state == _UCEContextState.FAULT:
      return
    ctx.wait_refs = ()
    ctx.wait_all = False
    ctx.fault_reason = reason
    self.pmu.add_event("tile_fault")
    self._terminal_events.append(
      _UCETerminalEvent(
        tile_id=self.tile_id,
        ctx_id=ctx.ctx_id,
        role_id=ctx.role_id,
        role_event_id=ctx.role_event_id,
        status="fault",
        reason=reason,
      )
    )
    self._set_context_state(ctx, _UCEContextState.FAULT, cycle)
    self._close_context_trace(ctx, cycle)

  def _drop_event_ref(self, ref: _UCEEventRef) -> None:
    self._local_event_owner.pop(ref.runtime_id, None)
    if ref.owner_ctx is None or ref.owner_ctx >= len(self.contexts):
      return
    owner = self.contexts[ref.owner_ctx]
    owner.event_records.pop(ref.local_name, None)
    owner.events_done.discard(ref.local_name)
    owner.wait_refs = tuple(r for r in owner.wait_refs if r.runtime_id != ref.runtime_id)

  def _clear_engine_queues(self, unregister: bool) -> None:
    for fifo in self._engine_queues.values():
      while fifo:
        entry = fifo.popleft()
        if unregister:
          self._drop_event_ref(entry.event_ref)

  def _unregister_context_events(self, ctx: _UCEContext) -> None:
    runtime_ids = [
      runtime_id for runtime_id, ref in self._local_event_owner.items() if ref.owner_ctx == ctx.ctx_id
    ]
    for runtime_id in runtime_ids:
      self._local_event_owner.pop(runtime_id, None)
    ctx.event_records.clear()
    ctx.events_done.clear()
    ctx.wait_refs = ()
    ctx.wait_all = False

  def _reset_context(self, ctx: _UCEContext) -> None:
    ctx.state = _UCEContextState.EMPTY
    ctx.program = None
    ctx.pc = 0
    ctx.role_id = None
    ctx.role_event_id = None
    ctx.wait_refs = ()
    ctx.wait_all = False
    ctx.tokens.clear()
    ctx.events_done.clear()
    ctx.event_records.clear()
    ctx.prepare_remaining = 0
    ctx.frame_bind_remaining = 0
    ctx.fault_reason = ""
    ctx.trace_slice_name = None
    ctx.prepare_total = 0
    ctx.memory = None
    ctx.task_identity = None
    ctx.l1_live_names.clear()

  def _state_label(self, ctx: _UCEContext) -> str:
    if ctx.state == _UCEContextState.EMPTY:
      return "EMPTY"
    prefix = {
      _UCEContextState.ACCEPT: "ACCEPT",
      _UCEContextState.FRAME_BIND: "FRAME_BIND",
      _UCEContextState.READY: "READY",
      _UCEContextState.WAIT_EVENT: "WAIT_EVENT",
      _UCEContextState.WAIT_STREAM: "WAIT_STREAM",
      _UCEContextState.WAIT_ENGINE_QUEUE: "WAIT_ENGINE_QUEUE",
      _UCEContextState.DONE: "DONE",
      _UCEContextState.FAULT: "FAULT",
    }[ctx.state]
    program = ctx.program.name if ctx.program is not None else "<empty>"
    return f"{prefix}:{program}"

  def _set_context_state(self, ctx: _UCEContext, state: _UCEContextState, cycle: int) -> None:
    ctx.state = state
    self._trace_context_state(ctx, self._state_label(ctx), cycle)

  def _ensure_context_traces(self, cycle: int) -> None:
    if self.tracer is None:
      return
    for ctx in self.contexts:
      # terminal/empty contexts are not traced as state slices; the
      # slice is closed at completion (DONE/FAULT) and never opened for
      # idle EMPTY contexts, so _ensure_context_traces must not re-open.
      if ctx.state in (_UCEContextState.EMPTY, _UCEContextState.DONE, _UCEContextState.FAULT):
        continue
      state_name = self._state_label(ctx)
      if ctx.trace_slice_name != state_name:
        self._trace_context_state(ctx, state_name, cycle)

  def _trace_context_state(
    self, ctx: _UCEContext, new_state_name: str, cycle: int, args: dict | None = None
  ) -> None:
    thread = f"UCE CTX{ctx.ctx_id}"
    if self.tracer is not None and ctx.trace_slice_name is not None:
      self.tracer.end(f"Tile{self.tile_id}", thread, ctx.trace_slice_name, cycle)
    ctx.trace_slice_name = new_state_name
    if self.tracer is not None:
      self.tracer.begin(f"Tile{self.tile_id}", thread, new_state_name, cycle, args=args)

  def _close_context_trace(self, ctx: _UCEContext, cycle: int) -> None:
    if self.tracer is not None and ctx.trace_slice_name is not None:
      self.tracer.end(f"Tile{self.tile_id}", f"UCE CTX{ctx.ctx_id}", ctx.trace_slice_name, cycle)
    ctx.trace_slice_name = None

  def _sample_context_counters(self, cycle: int) -> None:
    if self.tracer is None:
      return
    self.tracer.counter_if_changed(
      f"Tile{self.tile_id}",
      "active_context_count",
      cycle,
      self.active_context_count(),
      "contexts",
      thread="UCE",
    )
    self.tracer.counter_if_changed(
      f"Tile{self.tile_id}",
      "ready_context_count",
      cycle,
      self.ready_context_count(),
      "contexts",
      thread="UCE",
    )


class ComputeTile:
  """One Compute Tile: UCE + BOA/EVU/MFE/USE + L1 SRAM + stream ports.

  PR 2 (§5.1): holds a per-tile L1 ``BankedFreeExtentAllocator`` (shared
  capacity across UCE contexts) and one ``SlotFrame`` per hardware UCE
  context.  The shared ``TransferManager`` is injected so MFE lane heads
  submit local transactions.
  """

  def __init__(
    self,
    tile_id: int,
    cfg: HardwareConfig,
    tracer: Tracer | None = None,
    runtime_enabled: bool = False,
    memory_enabled: bool = False,
    context_count: int = 1,
    transfer_manager=None,
    l2_cache=None,
    l2_mshr=None,
    memory_trace=None,
  ):
    self.tile_id = tile_id
    self.cfg = cfg
    self.tracer = tracer
    self.memory_trace = memory_trace
    self.runtime_enabled = runtime_enabled
    self.memory_enabled = memory_enabled
    self.uce = TileUCE(tile_id, cfg, tracer, context_count=context_count, runtime_enabled=runtime_enabled)
    self.boa = BOAEngine(cfg, tile_id, tracer)
    self.evu = EVUEngine(cfg, tile_id, tracer)
    from .memory import DeterministicLRUCache, MshrTable

    self.l1_cache = DeterministicLRUCache(cfg.l1_cache_capacity_bytes, cfg.cache_line_bytes)
    self.l1_mshr = MshrTable(cfg.l1_mshr_entries)
    self.mfe = MFEEngine(
      cfg,
      tile_id,
      tracer,
      transfer_manager=transfer_manager,
      l1_cache=self.l1_cache,
      l1_mshr=self.l1_mshr,
      l2_cache=l2_cache,
      l2_mshr=l2_mshr,
      memory_trace=memory_trace,
    )
    self.use = USEEngine(cfg, tile_id, tracer)
    self.streams: dict[int, StreamQueue] = {}
    self.pmu = PMUCounter()
    # PR 2: per-tile L1 allocator + one frame per UCE context
    from .memory.allocator import BankedFreeExtentAllocator

    self.l1_allocator = BankedFreeExtentAllocator(
      memory_space="l1",
      capacity_bytes=cfg.tile_l1_bytes,
      banks=cfg.tile_l1_banks,
      trace=memory_trace,
      trace_tile_id=tile_id,
    )
    self.l1_frames: list[SlotFrame] = [
      SlotFrame(frame_id=context_id, l1_bytes=cfg.tile_l1_bytes) for context_id in range(context_count)
    ]

  def bind_stream(self, qid: int, q: StreamQueue) -> None:
    self.streams[qid] = q

  def unbind_stream(self, qid: int) -> None:
    self.streams.pop(qid, None)

  def get_stream(self, qid: int) -> StreamQueue:
    return self.streams[qid]

  def can_accept_context(self, context_id: int | None = None) -> bool:
    return self.available_context_id(context_id) is not None

  def plan_admission(
    self,
    program: ExecTileProgram,
    grid: GridInstanceId,
    role_event_id: str,
    logical_task_id: int,
    context_id: int | None = None,
  ) -> TileAdmission | AdmissionFailure | None:
    """Choose one exact context and plan its whole L1 bundle, without mutation."""
    if context_id is not None and not 0 <= context_id < len(self.l1_frames):
      return AdmissionFailure(
        AdmissionFailureKind.INVALID_REQUEST, f"hardware context id {context_id} is out of range"
      )
    selected = self.available_context_id(context_id)
    if selected is None:
      return None

    l1_plan = None
    if (self.memory_enabled or self.runtime_enabled) and program.l1_buffers:
      task = TaskIdentity(grid=grid, task_id=logical_task_id)
      requests = [
        AllocationRequest(
          memory_space="l1",
          buffer_id=buffer.name,
          owner=TaskBufferOwner(
            task.grid.context_name,
            task.grid.launch_generation,
            role_event_id,
            task.task_id,
            self.tile_id,
            selected,
            buffer.name,
          ),
          size_bytes=buffer.bytes,
          alignment=max(buffer.alignment, 1),
        )
        for buffer in program.l1_buffers
      ]
      candidate = self.l1_allocator.plan_bundle(requests)
      if isinstance(candidate, AdmissionFailure):
        return candidate
      l1_plan = candidate
    return TileAdmission(tile=self, logical_task_id=logical_task_id, context_id=selected, l1_plan=l1_plan)

  def commit_admission(self, admission: TileAdmission, program: ExecTileProgram, cycle: int) -> None:
    """Commit one Tile-local plan and prepare its exact context frame."""
    if admission.tile is not self:
      raise MemoryInvariantError("Tile admission belongs to another tile")
    if admission.bound or admission.l1_handles:
      raise MemoryInvariantError("Tile admission was already committed")
    if self.available_context_id(admission.context_id) != admission.context_id:
      raise MemoryInvariantError(f"UCE context {admission.context_id} is no longer available")
    if (self.memory_enabled or self.runtime_enabled) and program.l1_buffers and admission.l1_plan is None:
      raise MemoryInvariantError("Tile admission is missing its L1 plan")

    handles: tuple[AllocationHandle, ...] = ()
    frame = self.l1_frames[admission.context_id]
    materialize_l1 = self.memory_enabled or self.runtime_enabled
    materialized_specs = tuple(program.l1_buffers) if materialize_l1 else ()
    try:
      if admission.l1_plan is not None:
        expected_names = tuple(buffer.name for buffer in materialized_specs)
        planned_names = tuple(request.buffer_id for request in admission.l1_plan.requests)
        if planned_names != expected_names:
          raise MemoryInvariantError("Tile admission plan does not match the tile program")
        handles = self.l1_allocator.commit(admission.l1_plan, cycle)
      if len(handles) != len(materialized_specs):
        raise MemoryInvariantError("Tile admission committed an incomplete L1 bundle")
      admission.l1_handles = {buffer.name: handle for buffer, handle in zip(materialized_specs, handles)}
      if materialize_l1 and not frame.prepare(list(handles), list(materialized_specs)):
        raise MemoryInvariantError(f"L1 frame prepare failed on tile {self.tile_id}")
    except (MemoryInvariantError, ValueError):
      frame.release()
      for handle in handles:
        if not self.l1_allocator.is_released(handle):
          self.l1_allocator.request_release(handle, handle.owner, cycle)
      admission.l1_handles.clear()
      raise

    if materialize_l1 and self.memory_trace is not None and self.tracer is not None:
      self.tracer.instant(
        f"Tile{self.tile_id}",
        "Lifecycle",
        "frame_prepare",
        cycle,
        {
          "ctx_id": admission.context_id,
          "generation": frame.generation,
          "tile_id": self.tile_id,
          "slots": len(materialized_specs),
        },
      )

  def abort_admission(self, admission: TileAdmission, cycle: int) -> None:
    """Undo a committed/prepared/bound Tile admission without partial residue."""
    if admission.tile is not self:
      raise MemoryInvariantError("Tile admission belongs to another tile")
    frame = self.l1_frames[admission.context_id]
    generation = frame.generation
    if admission.bound:
      self.rollback_program(admission.context_id, cycle)
    else:
      frame.release()
      if self.memory_trace is not None and self.tracer is not None:
        self.tracer.instant(
          f"Tile{self.tile_id}",
          "Lifecycle",
          "frame_release",
          cycle,
          {
            "ctx_id": admission.context_id,
            "generation": generation,
            "tile_id": self.tile_id,
            "reason": "dispatch_rollback",
          },
        )
    for handle in tuple(admission.l1_handles.values()):
      if not self.l1_allocator.is_released(handle):
        self.l1_allocator.request_release(handle, handle.owner, cycle)
    admission.l1_handles.clear()
    admission.l1_plan = None
    admission.bound = False

  def load_program(
    self,
    program: ExecTileProgram,
    role_id: int | None,
    role_event_id: str | None,
    prepare_cycles: int = 0,
    context_id: int | None = None,
    memory: _TileContextMemory | None = None,
    task_identity: TaskIdentity | None = None,
  ) -> int | None:
    return self.uce.bind_context(
      program,
      role_id=role_id,
      role_event_id=role_event_id,
      prepare_cycles=prepare_cycles,
      context_id=context_id,
      memory=memory,
      task_identity=task_identity,
    )

  def available_context_id(self, preferred: int | None = None) -> int | None:
    if preferred is not None and not 0 <= preferred < len(self.l1_frames):
      return None
    candidates = range(len(self.l1_frames)) if preferred is None else (preferred,)
    return next(
      (
        context_id
        for context_id in candidates
        if self.uce.available_context_id(context_id) == context_id
        and self.l1_frames[context_id].is_available
      ),
      None,
    )

  def rollback_program(self, context_id: int, cycle: int = 0) -> None:
    self.uce.rollback_context(context_id)
    frame = self.l1_frames[context_id]
    generation = frame.generation
    frame.release()
    if self.memory_trace is not None and self.tracer is not None:
      self.tracer.instant(
        f"Tile{self.tile_id}",
        "Lifecycle",
        "frame_release",
        cycle,
        {"ctx_id": context_id, "generation": generation, "tile_id": self.tile_id, "reason": "rollback"},
      )

  def drain_context_terminals(self) -> list[_UCETerminalEvent]:
    return self.uce.drain_terminal_events()

  def retire_external_events(self, events: set[str] | tuple[str, ...]) -> None:
    self.uce.retire_external_events(events)

  def step(self, cycle: int, freeze_new_work: bool = False) -> EngineJob | None:
    completed = None
    # Running engines always tick so drain can complete.  The UCE receives
    # completions but freezes queued launches/PC issue when requested.
    for eng in (self.boa, self.evu, self.mfe, self.use):
      jobs = eng.tick(cycle, start_queued=not freeze_new_work) if eng is self.mfe else eng.tick(cycle)
      for job in jobs:
        completed = job
        self.uce.notify_event(job.event_id)
    self.uce.step(cycle, self, freeze_new_work=freeze_new_work)
    self._aggregate_pmu()
    return completed

  def _aggregate_pmu(self) -> None:
    for eng in (self.boa, self.evu, self.mfe, self.use):
      self.pmu.merge(eng.pmu)
      eng.pmu.reset()
    self.pmu.merge(self.uce.pmu)
    self.uce.pmu.reset()

  @property
  def done(self) -> bool:
    return not self.uce.has_active_contexts() and all(
      eng.state in (EngineState.IDLE, EngineState.DONE) for eng in (self.boa, self.evu, self.mfe, self.use)
    )

  def reset(self) -> None:
    self.uce.reset()
    for eng in (self.boa, self.evu, self.mfe, self.use):
      eng.reset()
    self.pmu.reset()
    self.l1_allocator.reset()
    for f in self.l1_frames:
      f.reset()

  def snapshot(self) -> dict:
    return {
      "tile_id": self.tile_id,
      "faulted": self.uce.faulted,
      "fault_reason": self.uce.fault_reason,
      "boa_state": self.boa.state.name,
      "evu_state": self.evu.state.name,
      "mfe_state": self.mfe.state.name,
      "use_state": self.use.state.name,
      "gather_active_jobs": len(self.mfe._gather_jobs),
      "uce": self.uce.snapshot(),
      "l1_frames": [f.snapshot() for f in self.l1_frames],
    }
