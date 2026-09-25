"""Single-writer SRAM profile and cache-maintenance controller.

The controller executes only compiler-authored descriptors.  It never infers a
missing command from the current mode and never treats an empty simulator as an
implicit ACK.  Prepare/Commit acknowledgements travel through a bounded,
one-item-per-cycle member bus and are validated against member identity and
profile generation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..execution_ir import ExecGroupActionOp, ExecTileGroupTask
from ..profiles import (
  MAINTENANCE_STEPS,
  PROFILE_STEPS,
  MemoryMaintenanceDesc,
  MemoryProfile,
  ProfileReconfigDesc,
  ProfileRegistry,
)
from .allocator import AllocationHandle, ExternalOwner, MemoryInvariantError
from .arena import ArenaPool
from .cache import CacheCleanRequest, CacheRange, DeterministicLRUCache
from .mshr import MshrTable
from .transfer import MemoryTransaction, ResolvedMemoryView, TransferManager, TransferOp, TransferStatus

_TERMINAL = frozenset(("completed", "faulted", "cancelled"))


@dataclass(frozen=True)
class _AwaitProof:
  run_generation: int
  owner: str
  instruction_id: str
  events: tuple[str, ...]
  cycle: int


@dataclass
class _DependencyReceipt:
  run_generation: int
  owner: str
  events: tuple[str, ...]
  cycle: int
  consumed: bool = False


@dataclass
class _MemberState:
  member_id: tuple[str, int, int]
  active_mode: int
  generation: int
  shadow_mode: int
  shadow_generation: int
  prepared_runtime_id: str = ""
  committed_runtime_id: str = ""


@dataclass(frozen=True)
class _BusRequest:
  runtime_id: str
  static_command_id: str
  member_id: tuple[str, int, int]
  stage: str
  target_mode: int
  generation: int


@dataclass(frozen=True)
class _BusResponse:
  runtime_id: str
  static_command_id: str
  member_id: tuple[str, int, int]
  stage: str
  generation: int
  success: bool
  reason: str
  ready_cycle: int


@dataclass(frozen=True)
class _ResponseOverride:
  static_command_id: str
  member_id: tuple[str, int, int]
  stage: str
  success: bool
  delay_cycles: int
  reason: str
  generation: int | None


@dataclass(frozen=True)
class _PendingClean:
  cache: DeterministicLRUCache
  request: CacheCleanRequest
  level: str


@dataclass
class _Command:
  runtime_id: str
  owner: str
  descriptor: ProfileReconfigDesc | MemoryMaintenanceDesc
  kind: str
  state: str
  step: str
  accepted_cycle: int
  completed_cycle: int = -1
  reason: str = ""
  cancel_requested: bool = False
  issue_closed: bool = False
  prepared_enqueued: bool = False
  commit_enqueued: bool = False
  ready_acks: set[tuple[str, int, int]] = field(default_factory=set)
  commit_acks: set[tuple[str, int, int]] = field(default_factory=set)
  consumed_awaits: set[str] = field(default_factory=set)
  maintenance_started: bool = False
  pending_clean_requests: deque[_PendingClean] = field(default_factory=deque)
  clean_transactions: dict[str, tuple[DeterministicLRUCache, str]] = field(default_factory=dict)
  maintenance_owner: ExternalOwner | None = None
  resolved_ranges: tuple[CacheRange, ...] = ()

  @property
  def command_id(self) -> str:
    return self.descriptor.command_id

  @property
  def cleanup_pending(self) -> bool:
    return bool(self.pending_clean_requests or self.clean_transactions)

  @property
  def requires_recovery(self) -> bool:
    if self.state == "faulted":
      return True
    # A cancelled maintenance command released its range gates at isolation
    # confirmation, so it does not hold the run hostage for recovery.
    return self.state == "cancelled" and self.issue_closed and self.kind != "memory_maintenance"


class ProfileController:
  """The unique profile/configuration writer for one TileGroup."""

  def __init__(self, group: object, *, trace=None) -> None:
    self.group = group
    self.trace = trace
    self._registry: ProfileRegistry | None = None
    self._run_generation = -1
    self._active_modes_mut = {"l1": -1, "l2": -1}
    self._generations_mut = {"l1": 0, "l2": 0}
    self._active_modes_view = MappingProxyType(self._active_modes_mut)
    self._generations_view = MappingProxyType(self._generations_mut)
    self._members: dict[tuple[str, int, int], _MemberState] = {}
    self._requests: deque[_BusRequest] = deque()
    self._responses: deque[_BusResponse] = deque()
    self._response_overrides: deque[_ResponseOverride] = deque()
    self._ignored_acks = 0
    self._initializing = False
    self._initialized = False
    self._initialization_fault = ""
    self._initialization_acks: set[tuple[str, int, int]] = set()
    self._commands: dict[str, _Command] = {}
    self._active_command: _Command | None = None
    self._awaits: dict[tuple[str, str], _AwaitProof] = {}
    self._used_awaits: set[tuple[str, str, str]] = set()
    self._dependency_receipts: list[_DependencyReceipt] = []
    self._startup_authorized = {"l1": False, "l2": False}
    self._startup_authorization_run = -1
    self._owner_inputs: dict[str, tuple[AllocationHandle, ...]] = {}
    self._profile_gates = {"l1": True, "l2": True}
    self._range_gates: dict[str, tuple[CacheRange, ...]] = {}
    self._recovering = False
    self._recovery_fault = ""
    self._recovery_targets: dict[str, int] = {"l1": -1, "l2": -1}
    self._recovery_generations: dict[str, int] = {"l1": 0, "l2": 0}
    self._recovery_acks: set[tuple[str, tuple[str, int, int], str]] = set()
    self._recovery_phase = "prepare"

  # -- public lifecycle ------------------------------------------------

  @property
  def active_modes(self):
    return self._active_modes_view

  @property
  def generations(self):
    return self._generations_view

  @property
  def initialized(self) -> bool:
    return self._initialized

  def initialize(self, registry: ProfileRegistry, cycle: int) -> None:
    if self._registry is not None:
      raise MemoryInvariantError("profile controller is already initialized")
    self._registry = registry
    for level in ("l1", "l2"):
      target = getattr(registry.target, level)
      profile = registry.profile(level, target.reset_mode)
      self._active_modes_mut[level] = profile.mode
      self._generations_mut[level] = 0
      self._verify_level_components(level, profile, generation=0)
      for member_id in profile.member_ids:
        if member_id in self._members:
          raise MemoryInvariantError("duplicate profile member identity")
        self._members[member_id] = _MemberState(
          member_id=member_id,
          active_mode=profile.mode,
          generation=0,
          shadow_mode=profile.mode,
          shadow_generation=0,
        )
        self._requests.append(
          _BusRequest(
            runtime_id="initialize",
            static_command_id="initialize",
            member_id=member_id,
            stage="INITIALIZE",
            target_mode=profile.mode,
            generation=0,
          )
        )
    self._initializing = True
    self._initialized = False
    self._profile_gates["l1"] = True
    self._profile_gates["l2"] = True
    self._trace_event("profile_initialize", cycle, {"registry_hash": registry.registry_hash})

  def recover(self, cycle: int) -> None:
    """Explicit post-fault recovery: rebuild reset_mode with monotonic generations.

    Never runs inside ``Simulator.run``.  Requires a terminal command (or
    none), proven isolation for both layers, and re-acknowledges every member
    through the real config bus: all Prepare ACKs first, then Commit requests,
    then publish only after every Commit ACK.
    """
    if not self._initialized or self._initializing:
      raise MemoryInvariantError("profile controller is not ready for recovery")
    if self.cancellation_pending:
      raise MemoryInvariantError("cannot recover while command isolation is incomplete")
    if self._recovering:
      raise MemoryInvariantError("profile recovery is already in progress")
    for level in ("l1", "l2"):
      if not self._level_quiescent(level):
        raise MemoryInvariantError(f"{level} domain is not quiescent for recovery")
    assert self._registry is not None
    for level in ("l1", "l2"):
      target = getattr(self._registry.target, level)
      profile = self._registry.profile(level, target.reset_mode)
      next_generation = self._generations_mut[level] + 1
      self._recovery_targets[level] = profile.mode
      self._recovery_generations[level] = next_generation
      for member_id in profile.member_ids:
        self._requests.append(
          _BusRequest("recover", "recover", member_id, "PREPARE", profile.mode, next_generation)
        )
    self._recovery_acks.clear()
    self._recovery_phase = "prepare"
    self._initialized = False
    self._profile_gates["l1"] = True
    self._profile_gates["l2"] = True
    self._recovering = True
    self._trace_event("profile_recover", cycle, {})

  def _recovery_expected(self, stage: str) -> set[tuple[str, tuple[str, int, int], str]]:
    assert self._registry is not None
    expected: set[tuple[str, tuple[str, int, int], str]] = set()
    for level in ("l1", "l2"):
      profile = self._registry.profile(level, self._recovery_targets[level])
      for member_id in profile.member_ids:
        expected.add((level, member_id, stage))
    return expected

  def _step_recovery(self, cycle: int) -> None:
    assert self._registry is not None
    if self._recovery_phase == "prepare":
      if self._recovery_acks != self._recovery_expected("PREPARE"):
        return
      for level in ("l1", "l2"):
        profile = self._registry.profile(level, self._recovery_targets[level])
        generation = self._recovery_generations[level]
        for member_id in profile.member_ids:
          self._requests.append(
            _BusRequest("recover", "recover", member_id, "COMMIT", profile.mode, generation)
          )
      self._recovery_acks.clear()
      self._recovery_phase = "commit"
      return
    if self._recovery_acks != self._recovery_expected("COMMIT"):
      return
    try:
      self._finish_recovery(cycle)
    except MemoryInvariantError as exc:
      # A failed recovery must not escape the simulator step loop; latch a
      # terminal fault with both issue gates closed (already closed while
      # not initialized) and let the caller observe the dead controller.
      self._recovering = False
      self._recovery_fault = str(exc)
      self._profile_gates["l1"] = True
      self._profile_gates["l2"] = True
      self._trace_event("profile_recovery_fault", cycle, {"reason": self._recovery_fault})

  def _finish_recovery(self, cycle: int) -> None:
    assert self._registry is not None
    for level in ("l1", "l2"):
      profile = self._registry.profile(level, self._recovery_targets[level])
      caches = self._caches(level)
      mshrs = self._mshrs(level)
      if len(caches) != profile.pools or len(mshrs) != profile.pools:
        raise MemoryInvariantError(f"{level} recovery component count mismatch")
      if any(mshr.stats.active for mshr in mshrs):
        raise MemoryInvariantError(f"{level} recovery reset has active cache refills")
      if any(cache.stats.pending_cleans for cache in caches):
        raise MemoryInvariantError(f"{level} recovery reset has pending cache cleans")

    discarded_dirty_lines = 0
    for level in ("l1", "l2"):
      profile = self._registry.profile(level, self._recovery_targets[level])
      generation = self._recovery_generations[level]
      caches = self._caches(level)
      mshrs = self._mshrs(level)
      for cache, mshr in zip(caches, mshrs, strict=True):
        discarded_dirty_lines += cache.reset_after_isolation(refills_quiescent=not bool(mshr.stats.active))
      for pool_id in range(profile.pools):
        self._pool(level, pool_id).reconfigure(profile, generation, cycle)
      for cache in caches:
        cache.reconfigure(profile.cache_bytes, generation, write_policy=profile.cache_write_policy)
      for mshr in mshrs:
        mshr.reconfigure(generation)
      for member_id in profile.member_ids:
        member = self._members[member_id]
        member.active_mode = profile.mode
        member.generation = generation
        member.shadow_mode = profile.mode
        member.shadow_generation = generation
        member.prepared_runtime_id = ""
        member.committed_runtime_id = ""
      self._active_modes_mut[level] = profile.mode
      self._generations_mut[level] = generation
      self._transfer_manager().note_profile_commit(level, tuple(range(profile.pools)), generation)
    self._commands.clear()
    self._active_command = None
    self._awaits.clear()
    self._used_awaits.clear()
    self._dependency_receipts.clear()
    self._owner_inputs.clear()
    self._range_gates.clear()
    self._profile_gates["l1"] = False
    self._profile_gates["l2"] = False
    self._initialized = True
    self._recovering = False
    self._startup_authorized["l1"] = True
    self._startup_authorization_run = -1
    self._startup_authorized["l2"] = True
    self._trace_event("profile_recovered", cycle, {"discarded_dirty_lines": discarded_dirty_lines})

  def begin_run(self, run_generation: int) -> None:
    if type(run_generation) is not int or run_generation < 0:
      raise ValueError("run_generation must be a non-negative integer")
    if self.cancellation_pending:
      raise MemoryInvariantError("cannot begin a run while command isolation is incomplete")
    if any(record.requires_recovery for record in self._commands.values()):
      raise MemoryInvariantError("cannot begin a run before explicit profile recovery")
    if run_generation <= self._run_generation:
      raise MemoryInvariantError("run_generation must increase monotonically")
    if not self._initialized:
      raise MemoryInvariantError("profile initialization ACKs are incomplete")
    if any(self._startup_authorized.values()):
      if self._startup_authorization_run < 0:
        self._startup_authorization_run = run_generation
      elif self._startup_authorization_run != run_generation:
        self._startup_authorized["l1"] = False
        self._startup_authorized["l2"] = False
    self._run_generation = run_generation
    self._commands.clear()
    self._active_command = None
    self._awaits.clear()
    self._used_awaits.clear()
    self._dependency_receipts.clear()
    self._owner_inputs.clear()
    self._range_gates.clear()

  def note_user_issue(self, levels: tuple[str, ...]) -> None:
    """Startup authorization expires when real user work enters that domain."""
    for level in levels:
      if level not in self._startup_authorized:
        raise MemoryInvariantError("user issue names an unknown profile domain")
      self._startup_authorized[level] = False

  def bind_owner_inputs(self, owner: str, handles: tuple[AllocationHandle, ...]) -> None:
    """Bind descriptor input indices to actual generation-bearing handles."""
    self._validate_owner(owner)
    if owner in self._owner_inputs:
      raise MemoryInvariantError("profile owner inputs are already bound")
    for handle in handles:
      if not isinstance(handle, AllocationHandle) or handle.memory_space != "hbm":
        raise MemoryInvariantError("maintenance inputs must be HBM handles")
    self._owner_inputs[owner] = tuple(handles)

  def note_await(self, owner: str, instruction_id: str, events: tuple[str, ...], cycle: int) -> None:
    self._validate_owner(owner)
    if not instruction_id or not events or any(not event for event in events):
      raise MemoryInvariantError("ordinary await proof requires an instruction id and events")
    key = (owner, instruction_id)
    if key in self._awaits:
      raise MemoryInvariantError("ordinary await instruction was recorded twice")
    self._awaits[key] = _AwaitProof(self._run_generation, owner, instruction_id, tuple(events), cycle)

  def note_dependencies(self, owner: str, events: tuple[str, ...], cycle: int) -> None:
    """Record exact scheduler-proven Group dependency readiness.

    This is not an ordinary await and creates no edge.  One generated
    maintenance command may consume one same-run exact receipt.
    """
    self._validate_owner(owner)
    normalized = tuple(events)
    if len(normalized) != len(set(normalized)) or any(not event for event in normalized):
      raise MemoryInvariantError("dependency receipt events must be unique and non-empty strings")
    self._dependency_receipts.append(_DependencyReceipt(self._run_generation, owner, normalized, cycle))

  def submit(self, command: ProfileReconfigDesc | MemoryMaintenanceDesc, owner: str, cycle: int) -> bool:
    if not self._initialized:
      raise MemoryInvariantError("profile controller is not initialized")
    self._validate_owner(owner)
    if not isinstance(command, (ProfileReconfigDesc, MemoryMaintenanceDesc)):
      raise TypeError("unsupported profile controller command")
    runtime_id = f"r{self._run_generation}:{owner}:{command.command_id}"
    existing = self._commands.get(command.command_id)
    if existing is not None:
      if existing.runtime_id == runtime_id and existing.owner == owner and existing.descriptor == command:
        return True
      raise MemoryInvariantError("conflicting duplicate profile command id")
    if self.cancellation_pending:
      return False
    if any(record.requires_recovery for record in self._commands.values()):
      raise MemoryInvariantError("profile controller requires explicit recovery")

    if isinstance(command, ProfileReconfigDesc):
      self._validate_profile_command(command, owner)
      kind = "profile_reconfig"
    else:
      self._validate_maintenance_command(command)
      kind = "memory_maintenance"
    record = _Command(
      runtime_id=runtime_id,
      owner=owner,
      descriptor=command,
      kind=kind,
      state="pending",
      step=command.steps[0],
      accepted_cycle=cycle,
    )
    self._commands[command.command_id] = record
    self._active_command = record
    self._trace_command(record, cycle)
    return True

  def status(self, command_id: str) -> str:
    try:
      return self._commands[command_id].state
    except KeyError as exc:
      raise MemoryInvariantError("unknown profile command id") from exc

  def command_reason(self, command_id: str) -> str:
    try:
      return self._commands[command_id].reason
    except KeyError as exc:
      raise MemoryInvariantError("unknown profile command id") from exc

  def issue_gate_closed(self, level: str) -> bool:
    if level not in ("l1", "l2"):
      raise ValueError("profile level must be l1 or l2")
    return self._profile_gates[level] or not self._initialized

  def range_issue_blocked(
    self, level: str, allocation_id: str, allocation_generation: int, offset: int, size: int
  ) -> bool:
    if self.issue_gate_closed(level):
      return True
    end = offset + size
    return any(
      item.allocation_id == allocation_id
      and item.allocation_generation == allocation_generation
      and item.offset < end
      and offset < item.end
      for item in self._range_gates.get(level, ())
    )

  @property
  def cancellation_pending(self) -> bool:
    """Whether command, clean-transfer, or member-bus isolation needs progress."""
    return (
      any(
        record.state in ("pending", "running") or record.cleanup_pending
        for record in self._commands.values()
      )
      or bool(self._requests)
      or bool(self._responses)
    )

  def request_cancel_all(self, cycle: int) -> None:
    """Request isolation for every live command without materializing a snapshot."""
    for command_id, record in self._commands.items():
      if record.state in ("pending", "running"):
        self.request_cancel(command_id, cycle)
      elif record.state == "faulted" and record.cleanup_pending:
        self._advance_fault_cleanup(record, cycle)

  def request_cancel(self, command_id: str, cycle: int) -> None:
    try:
      record = self._commands[command_id]
    except KeyError as exc:
      raise MemoryInvariantError("unknown profile command id") from exc
    if record.state in _TERMINAL or record.cancel_requested:
      return
    record.cancel_requested = True
    self._trace_event(
      "profile_cancel_requested",
      cycle,
      {"command_id": command_id, "runtime_id": record.runtime_id, "stage": record.step},
    )
    if record.state == "pending" and not record.issue_closed:
      self._finish(record, "cancelled", cycle, "cancelled before issue gate closed")

  def queue_member_response(
    self,
    command_id: str,
    member_id: tuple[str, int, int],
    stage: str,
    *,
    success: bool = True,
    delay_cycles: int = 0,
    reason: str = "",
    generation: int | None = None,
  ) -> None:
    """Queue one deterministic member ACK override for fault/delay testing."""
    if stage not in ("INITIALIZE", "PREPARE", "COMMIT"):
      raise ValueError("member response stage must be INITIALIZE, PREPARE or COMMIT")
    if member_id not in self._members:
      raise ValueError("unknown profile member")
    if type(delay_cycles) is not int or delay_cycles < 0:
      raise ValueError("member response delay must be non-negative")
    self._response_overrides.append(
      _ResponseOverride(command_id, member_id, stage, success, delay_cycles, reason, generation)
    )

  def step(self, cycle: int) -> None:
    """Advance the config bus and at most one command FSM transition."""
    self._service_one_bus_item(cycle)
    if self._initializing:
      if self._initialization_fault:
        return
      if len(self._initialization_acks) == len(self._members):
        self._initializing = False
        self._initialized = True
        self._startup_authorized["l1"] = True
        self._startup_authorized["l2"] = True
        self._profile_gates["l1"] = False
        self._profile_gates["l2"] = False
        self._trace_event("profile_initialized", cycle, {})
      return
    if self._recovering:
      self._step_recovery(cycle)
      return

    record = self._active_command
    if record is None:
      return
    if record.state == "faulted":
      self._advance_fault_cleanup(record, cycle)
      return
    if record.state in _TERMINAL:
      return
    assert self._registry is not None
    timeout = self._registry.target.profile_command_timeout_cycles
    # A command already converging to cancellation owns its terminal state;
    # isolation drain is bounded by tombstone capacity, not the command
    # timeout, so it must never be re-faulted as a spurious timeout.
    if record.cancel_requested:
      self._advance_cancel(record, cycle)
      return
    if cycle - record.accepted_cycle >= timeout:
      self._fault(record, cycle, "profile command timeout")
      return
    if record.state == "pending":
      record.state = "running"
    if record.kind == "profile_reconfig":
      self._step_profile(record, cycle)
    else:
      self._step_maintenance(record, cycle)

  # -- descriptor validation -----------------------------------------

  def _validate_owner(self, owner: str) -> None:
    if self._run_generation < 0:
      raise MemoryInvariantError("begin_run must precede runtime controller use")
    prefix, separator, suffix = owner.rpartition("@")
    if not separator or not prefix or not suffix.isdecimal():
      raise MemoryInvariantError("controller owner must be name@launch_generation")
    if prefix == "device" and int(suffix) != self._run_generation:
      raise MemoryInvariantError("device controller owner uses the wrong run generation")

  def _validate_profile_command(self, command: ProfileReconfigDesc, owner: str) -> None:
    assert self._registry is not None
    if tuple(command.steps) != PROFILE_STEPS:
      raise MemoryInvariantError("profile command omits or reorders required steps")
    if command.registry_hash != self._registry.registry_hash:
      raise MemoryInvariantError("profile command registry hash mismatch")
    if command.level not in ("l1", "l2"):
      raise MemoryInvariantError("profile command level must be l1 or l2")
    if command.expected_mode != self._active_modes_mut[command.level]:
      raise MemoryInvariantError("profile command expected mode is not active")
    if command.target_mode == command.expected_mode:
      raise MemoryInvariantError("same-mode profile command is not executable")
    profile = self._registry.profile(command.level, command.target_mode)
    if tuple(command.member_ids) != profile.member_ids:
      raise MemoryInvariantError("profile command member list does not match target")
    if not command.wait_instruction_ids and command.frontier:
      raise MemoryInvariantError("profile command frontier lacks an ordinary await")
    if command.level == "l1" and command.exclusive_binding_id:
      owner_binding = owner.rsplit("@", 1)[0]
      if owner_binding not in ("device", command.exclusive_binding_id):
        raise MemoryInvariantError("L1-exclusive profile command owner mismatch")

  def _validate_maintenance_command(self, command: MemoryMaintenanceDesc) -> None:
    if tuple(command.steps) != MAINTENANCE_STEPS:
      raise MemoryInvariantError("maintenance command omits or reorders required steps")
    if not command.levels or any(level not in ("l1", "l2") for level in command.levels):
      raise MemoryInvariantError("maintenance command has invalid levels")
    if len(command.levels) != len(set(command.levels)):
      raise MemoryInvariantError("maintenance command levels are duplicated")
    if not command.ranges:
      raise MemoryInvariantError("maintenance command must contain a real range")

  # -- command FSM ----------------------------------------------------

  def _move(self, record: _Command, step: str, cycle: int) -> None:
    record.step = step
    self._trace_step(record, cycle)

  def _step_profile(self, record: _Command, cycle: int) -> None:
    command = record.descriptor
    assert isinstance(command, ProfileReconfigDesc)
    step = record.step
    if step == "ACQUIRE":
      self._move(record, "CHECK_FRONTIER", cycle)
    elif step == "CHECK_FRONTIER":
      if self._consume_frontier(record, command.level, command.wait_instruction_ids, command.frontier):
        violation = self._frontier_plan_violation(record, command)
        if violation:
          self._fault(record, cycle, violation)
          return
        self._move(record, "CLOSE_ISSUE", cycle)
    elif step == "CLOSE_ISSUE":
      self._profile_gates[command.level] = True
      record.issue_closed = True
      self._move(record, "DRAIN_REFERENCES", cycle)
    elif step == "DRAIN_REFERENCES":
      if self._level_quiescent(command.level):
        self._move(record, "CLEAN_INVALIDATE", cycle)
    elif step == "CLEAN_INVALIDATE":
      if self._maintain_caches(record, cycle, levels=(command.level,), ranges=None):
        self._move(record, "DRAIN_DOWNSTREAM", cycle)
    elif step == "DRAIN_DOWNSTREAM":
      if self._clean_transactions_done(record, cycle) and self._level_quiescent(command.level):
        self._move(record, "PREPARE", cycle)
    elif step == "PREPARE":
      if not record.prepared_enqueued:
        next_generation = self._generations_mut[command.level] + 1
        for member_id in command.member_ids:
          self._requests.append(
            _BusRequest(
              record.runtime_id,
              command.command_id,
              member_id,
              "PREPARE",
              command.target_mode,
              next_generation,
            )
          )
        record.prepared_enqueued = True
      self._move(record, "WAIT_READY_ACK", cycle)
    elif step == "WAIT_READY_ACK":
      if record.ready_acks == set(command.member_ids):
        self._move(record, "COMMIT", cycle)
    elif step == "COMMIT":
      if not record.commit_enqueued:
        next_generation = self._generations_mut[command.level] + 1
        for member_id in command.member_ids:
          self._requests.append(
            _BusRequest(
              record.runtime_id,
              command.command_id,
              member_id,
              "COMMIT",
              command.target_mode,
              next_generation,
            )
          )
        record.commit_enqueued = True
      self._move(record, "WAIT_COMMIT_ACK", cycle)
    elif step == "WAIT_COMMIT_ACK":
      if record.commit_acks == set(command.member_ids):
        try:
          self._publish_profile(command, cycle)
        except Exception as exc:
          self._fault(record, cycle, f"profile publish failed: {exc}")
          return
        self._move(record, "OPEN_ISSUE", cycle)
    elif step == "OPEN_ISSUE":
      self._profile_gates[command.level] = False
      self._move(record, "RELEASE", cycle)
    elif step == "RELEASE":
      self._finish(record, "completed", cycle, "")

  def _step_maintenance(self, record: _Command, cycle: int) -> None:
    command = record.descriptor
    assert isinstance(command, MemoryMaintenanceDesc)
    step = record.step
    if step == "ACQUIRE":
      self._move(record, "WAIT_DEPENDENCIES", cycle)
    elif step == "WAIT_DEPENDENCIES":
      if self._consume_dependency_events(record, command.dependencies, command.levels):
        self._move(record, "BLOCK_RANGE_ISSUE", cycle)
    elif step == "BLOCK_RANGE_ISSUE":
      record.resolved_ranges = self._resolve_ranges(record.owner, command)
      for level in command.levels:
        self._range_gates[level] = record.resolved_ranges
      record.issue_closed = True
      self._move(record, "DRAIN_RANGE_REFERENCES", cycle)
    elif step == "DRAIN_RANGE_REFERENCES":
      manager = self._transfer_manager()
      if not manager.range_inflight(record.resolved_ranges):
        self._move(record, "CLEAN_INVALIDATE", cycle)
    elif step == "CLEAN_INVALIDATE":
      if self._maintain_caches(record, cycle, levels=command.levels, ranges=record.resolved_ranges):
        self._move(record, "DRAIN_DOWNSTREAM", cycle)
    elif step == "DRAIN_DOWNSTREAM":
      if self._clean_transactions_done(record, cycle):
        self._move(record, "ACK", cycle)
    elif step == "ACK":
      self._move(record, "UNBLOCK_RANGE_ISSUE", cycle)
    elif step == "UNBLOCK_RANGE_ISSUE":
      for level in command.levels:
        self._range_gates.pop(level, None)
      self._move(record, "RELEASE", cycle)
    elif step == "RELEASE":
      self._finish(record, "completed", cycle, "")

  # -- member bus -----------------------------------------------------

  def _service_one_bus_item(self, cycle: int) -> None:
    # One request OR one response total per Tick.  A ready response wins so
    # finite response capacity cannot be starved by a long request sequence.
    response_index = next(
      (index for index, item in enumerate(self._responses) if item.ready_cycle <= cycle), None
    )
    if response_index is not None:
      response = self._responses[response_index]
      del self._responses[response_index]
      self._consume_response(response, cycle)
      return
    if self._requests:
      request = self._requests.popleft()
      self._service_request(request, cycle)

  def _service_request(self, request: _BusRequest, cycle: int) -> None:
    member = self._members.get(request.member_id)
    success = True
    reason = ""
    if member is None:
      success = False
      reason = "unknown member"
    elif request.stage == "INITIALIZE":
      if member.active_mode != request.target_mode or member.generation != 0:
        success = False
        reason = "member reset state mismatch"
    elif request.stage == "PREPARE":
      try:
        self._verify_member_ready(
          request.member_id, request.target_mode, allow_dirty=request.runtime_id == "recover"
        )
      except Exception as exc:
        success = False
        reason = str(exc)
      if success:
        member.shadow_mode = request.target_mode
        member.shadow_generation = request.generation
        member.prepared_runtime_id = request.runtime_id
    elif request.stage == "COMMIT":
      if (
        member.prepared_runtime_id != request.runtime_id
        or member.shadow_mode != request.target_mode
        or member.shadow_generation != request.generation
      ):
        success = False
        reason = "member commit lacks matching prepared shadow"
      else:
        # This is the member-local shadow write.  Global publication waits for
        # every validated Commit ACK.
        member.committed_runtime_id = request.runtime_id
    else:
      success = False
      reason = "unknown member bus stage"

    override = self._pop_response_override(request)
    ready_cycle = cycle
    generation = request.generation
    if override is not None:
      success = override.success
      reason = override.reason
      ready_cycle += override.delay_cycles
      if override.generation is not None:
        generation = override.generation
    self._responses.append(
      _BusResponse(
        request.runtime_id,
        request.static_command_id,
        request.member_id,
        request.stage,
        generation,
        success,
        reason,
        ready_cycle,
      )
    )
    self._trace_member(request, cycle, "request")

  def _consume_response(self, response: _BusResponse, cycle: int) -> None:
    self._trace_member(response, cycle, "ack")
    member = self._members.get(response.member_id)
    if member is None:
      self._ignored_acks += 1
      return
    if response.stage == "INITIALIZE":
      if response.runtime_id != "initialize" or response.generation != 0:
        self._ignored_acks += 1
        return
      if not response.success:
        self._initialization_fault = response.reason or "member initialization failed"
        return
      self._initialization_acks.add(response.member_id)
      return

    if response.runtime_id == "recover":
      if not self._recovering:
        self._ignored_acks += 1
        return
      level = response.member_id[0]
      if level not in ("l1", "l2"):
        self._ignored_acks += 1
        return
      if response.generation != self._recovery_generations[level]:
        self._ignored_acks += 1
        return
      if not response.success:
        self._recovery_acks.clear()
        self._recovering = False
        self._recovery_fault = response.reason or "member recovery ACK failed"
        self._trace_event("profile_recover_failed", cycle, {"reason": self._recovery_fault})
        return
      self._recovery_acks.add((level, response.member_id, response.stage))
      return

    record = self._commands.get(response.static_command_id)
    if record is None or record.runtime_id != response.runtime_id:
      self._ignored_acks += 1
      return
    command = record.descriptor
    if not isinstance(command, ProfileReconfigDesc):
      self._ignored_acks += 1
      return
    expected_generation = self._generations_mut[command.level] + 1
    if response.generation != expected_generation:
      self._ignored_acks += 1
      return
    if not response.success:
      self._fault(record, cycle, response.reason or f"member {response.stage} failed")
      return
    if response.stage == "PREPARE":
      record.ready_acks.add(response.member_id)
    elif response.stage == "COMMIT":
      record.commit_acks.add(response.member_id)
    else:
      self._ignored_acks += 1
      return

  def _pop_response_override(self, request: _BusRequest) -> _ResponseOverride | None:
    for item in tuple(self._response_overrides):
      if (
        item.static_command_id == request.static_command_id
        and item.member_id == request.member_id
        and item.stage == request.stage
      ):
        self._response_overrides.remove(item)
        return item
    return None

  # -- runtime closure ------------------------------------------------

  @staticmethod
  def _require_attr(obj: object, name: str):
    if not hasattr(obj, name):
      raise MemoryInvariantError(f"profile backend integration missing {name}")
    return getattr(obj, name)

  def _tiles(self) -> tuple[object, ...]:
    tiles = self._require_attr(self.group, "tiles")
    return tuple(tiles)

  def _pool(self, level: str, pool_id: int) -> ArenaPool:
    if level == "l2":
      pool = self._require_attr(self.group, "l2_sram")
    else:
      tiles = self._tiles()
      if not 0 <= pool_id < len(tiles):
        raise MemoryInvariantError("L1 profile member names an unknown Tile")
      pool = self._require_attr(tiles[pool_id], "l1_allocator")
    if not isinstance(pool, ArenaPool):
      raise MemoryInvariantError(f"{level} profile backend requires ArenaPool")
    return pool

  def _caches(self, level: str) -> tuple[DeterministicLRUCache, ...]:
    if level == "l2":
      cache = self._require_attr(self.group, "l2_cache")
      if not isinstance(cache, DeterministicLRUCache):
        raise MemoryInvariantError("L2 cache integration is missing")
      return (cache,)
    result: list[DeterministicLRUCache] = []
    for tile in self._tiles():
      mfe = self._require_attr(tile, "mfe")
      cache = self._require_attr(mfe, "l1_cache")
      if not isinstance(cache, DeterministicLRUCache):
        raise MemoryInvariantError("L1 cache integration is missing")
      result.append(cache)
    return tuple(result)

  def _mshrs(self, level: str) -> tuple[MshrTable, ...]:
    if level == "l2":
      raw = (self._require_attr(self.group, "l2_mshr"),)
    else:
      raw = tuple(self._require_attr(self._require_attr(tile, "mfe"), "l1_mshr") for tile in self._tiles())
    result: list[MshrTable] = []
    for mshr in raw:
      if not isinstance(mshr, MshrTable):
        raise MemoryInvariantError(f"{level} MSHR integration is missing")
      result.append(mshr)
    return tuple(result)

  def _transfer_manager(self) -> TransferManager:
    manager = self._require_attr(self.group, "transfer_manager")
    if not isinstance(manager, TransferManager):
      raise MemoryInvariantError("transfer manager integration is missing")
    return manager

  def _verify_level_components(self, level: str, profile: MemoryProfile, generation: int) -> None:
    for pool_id in range(profile.pools):
      pool = self._pool(level, pool_id)
      if pool.profile.mode != profile.mode or pool.profile_generation != generation:
        raise MemoryInvariantError(f"{level} ArenaPool reset profile is not installed")
    for cache in self._caches(level):
      if (
        cache.capacity_bytes != profile.cache_bytes
        or cache.profile_generation != generation
        or cache.write_policy != profile.cache_write_policy
      ):
        raise MemoryInvariantError(f"{level} cache reset profile is not installed")
    for mshr in self._mshrs(level):
      if mshr.generation != generation:
        raise MemoryInvariantError(f"{level} MSHR reset generation is not installed")

  @staticmethod
  def _task_uses_l1(task: object) -> bool:
    if not isinstance(task, ExecTileGroupTask):
      raise MemoryInvariantError("profile backend requires an executable TileGroup task")
    return any(action.op is ExecGroupActionOp.DISPATCH_ROLE for action in task.actions)

  def _frontier_plan_violation(self, record: _Command, command: ProfileReconfigDesc) -> str:
    if command.level != "l1":
      return ""
    group = self.group
    owner_name, _, owner_generation = record.owner.rpartition("@")
    own_generation = int(owner_generation)
    own_binding = command.exclusive_binding_id if owner_name != "device" else ""
    pending = self._require_attr(group, "_pending_root_requests")
    for root_record in pending.values():
      request = self._require_attr(root_record, "request")
      task = self._require_attr(request, "task")
      if self._task_uses_l1(task):
        return f"compiled L1 exclusion omitted a pending L1-producing root {task.binding_id}"
    sequencers = self._require_attr(group, "_active_sequencers")
    for sequencer in sequencers:
      task = self._require_attr(sequencer, "task")
      if not isinstance(task, ExecTileGroupTask):
        raise MemoryInvariantError("profile backend sequencer has no executable task")
      is_own_parent = (
        own_binding
        and task.binding_id == own_binding
        and self._require_attr(sequencer, "context_launch_generation") == own_generation
      )
      if is_own_parent:
        continue
      submission_pc = self._require_attr(sequencer, "submission_pc")
      remaining = task.actions[submission_pc:]
      if any(action.op is ExecGroupActionOp.DISPATCH_ROLE for action in remaining):
        return f"compiled L1 exclusion omitted a foreign root with future Task dispatch: {task.binding_id}"
    return ""

  def _verify_member_ready(
    self, member_id: tuple[str, int, int], target_mode: int, *, allow_dirty: bool = False
  ) -> None:
    level, pool_id, bank_id = member_id
    assert self._registry is not None
    profile = self._registry.profile(level, target_mode)
    if member_id not in profile.member_ids:
      raise MemoryInvariantError("profile target omits member")
    pool = self._pool(level, pool_id)
    snapshot = pool.snapshot()
    if snapshot["live_arenas"] or snapshot["live_views"]:
      raise MemoryInvariantError("member ArenaPool is not quiescent")
    if snapshot["pin_count"] or snapshot["inflight_count"]:
      raise MemoryInvariantError("member ArenaPool retains references")
    if not 0 <= bank_id < pool.banks:
      raise MemoryInvariantError("profile member bank is out of range")
    cache = self._caches(level)[pool_id if level == "l1" else 0]
    cache_stats = cache.stats
    if cache_stats.pending_cleans:
      raise MemoryInvariantError("member cache clean isolation is incomplete")
    if cache_stats.dirty_lines and not allow_dirty:
      raise MemoryInvariantError("member cache maintenance is incomplete")
    mshr = self._mshrs(level)[pool_id if level == "l1" else 0]
    if mshr.stats.active:
      raise MemoryInvariantError("member MSHR refills are not drained")

  def _closure_ledger(self, level: str) -> dict[str, int]:
    """Count live records which actually participate in the affected layer."""
    group = self.group
    pending = self._require_attr(group, "_pending_root_requests")
    routes = self._require_attr(group, "_grid_routes")
    leases = self._require_attr(group, "_task_leases")
    arenas = self._require_attr(group, "_l2_arenas")
    participating_routes: list[object] = []
    for route in routes.values():
      sequencer = self._require_attr(route, "sequencer")
      task = self._require_attr(sequencer, "task")
      if level != "l1" or self._task_uses_l1(task):
        participating_routes.append(route)
    active_routes = 0
    for route in participating_routes:
      expected = self._require_attr(route, "expected")
      retired = self._require_attr(route, "retired")
      if not isinstance(expected, dict) or not isinstance(retired, set):
        raise MemoryInvariantError("profile backend GridRoute ledger is malformed")
      if set(expected) - retired:
        active_routes += 1
    pending_l1 = 0
    for root_record in pending.values():
      request = self._require_attr(root_record, "request")
      task = self._require_attr(request, "task")
      pending_l1 += self._task_uses_l1(task)
    return {
      "pending_roots": pending_l1 if level == "l1" else len(pending),
      "routes": len(participating_routes),
      "active_routes": active_routes,
      # Every TaskIdentity lease denotes a committed L1 Arena/Frame.
      "task_leases": len(leases),
      "l2_arenas": 0 if level == "l1" else len(arenas),
    }

  def _level_quiescent(self, level: str) -> bool:
    ledger = self._closure_ledger(level)
    if level == "l1":
      if ledger["pending_roots"] or ledger["routes"] or ledger["task_leases"]:
        return False
    else:
      if ledger["pending_roots"] or ledger["routes"] or ledger["task_leases"] or ledger["l2_arenas"]:
        return False
    levels = ("l1",) if level == "l1" else ("l1", "l2")
    transfer_closure = self._transfer_manager().closure_snapshot(levels)
    if not transfer_closure["quiescent"]:
      return False
    pools: list[ArenaPool] = []
    if level == "l1":
      pools.extend(self._pool("l1", index) for index, _ in enumerate(self._tiles()))
    else:
      pools.append(self._pool("l2", 0))
      pools.extend(self._pool("l1", index) for index, _ in enumerate(self._tiles()))
    for pool in pools:
      snapshot = pool.snapshot()
      if (
        snapshot["live_arenas"]
        or snapshot["live_views"]
        or snapshot["pin_count"]
        or snapshot["inflight_count"]
      ):
        return False
    for tile in self._tiles():
      uce = self._require_attr(tile, "uce")
      if uce.has_active_contexts():
        return False
      frames = self._require_attr(tile, "l1_frames")
      if any(not frame.is_available for frame in frames):
        return False
      mfe = self._require_attr(tile, "mfe")
      if mfe.state.name not in ("IDLE", "DONE"):
        return False
    for affected_level in levels:
      if any(mshr.stats.active for mshr in self._mshrs(affected_level)):
        return False
    return True

  # -- waits, ranges, cache maintenance -------------------------------

  def _consume_frontier(
    self, record: _Command, level: str, instruction_ids: tuple[str, ...], frontier: tuple[str, ...]
  ) -> bool:
    if not instruction_ids and not frontier:
      if not record.owner.startswith("device@"):
        self._fault(
          record, record.accepted_cycle, "empty frontier requires Device initialization authorization"
        )
        return False
      if not self._startup_authorized[level]:
        self._fault(
          record,
          record.accepted_cycle,
          f"{level} initialization/recovery authorization was already consumed",
        )
        return False
      self._startup_authorized[level] = False
      record.consumed_awaits.add(f"{level}:<startup>")
      return True

    proofs: list[tuple[tuple[str, str], tuple[str, str, str], _AwaitProof]] = []
    for instruction_id in instruction_ids:
      proof_key = (record.owner, instruction_id)
      used_key = (record.owner, instruction_id, level)
      proof = self._awaits.get(proof_key)
      if proof is None or proof.run_generation != self._run_generation:
        self._fault(record, record.accepted_cycle, "missing generation-bound await proof")
        return False
      if used_key in self._used_awaits:
        self._fault(record, record.accepted_cycle, f"await proof was already consumed for {level}")
        return False
      proofs.append((proof_key, used_key, proof))
    events = {event for _, _, proof in proofs for event in proof.events}
    if not set(frontier) <= events:
      self._fault(record, record.accepted_cycle, "await proof does not cover command frontier")
      return False
    for proof_key, used_key, _ in proofs:
      self._used_awaits.add(used_key)
      record.consumed_awaits.add(f"{level}:{proof_key[1]}")
    return True

  def _consume_dependency_events(
    self, record: _Command, dependencies: tuple[str, ...], levels: tuple[str, ...]
  ) -> bool:
    receipt = next(
      (
        item
        for item in self._dependency_receipts
        if not item.consumed
        and item.run_generation == self._run_generation
        and item.owner == record.owner
        and item.events == tuple(dependencies)
      ),
      None,
    )
    if receipt is not None:
      receipt.consumed = True
      return True
    if record.owner.startswith("device@"):
      if not dependencies:
        return True
      candidates = tuple(
        (key, proof)
        for key, proof in self._awaits.items()
        if key[0] == record.owner and proof.run_generation == self._run_generation
      )
      if not candidates:
        return False
      events = {event for _, proof in candidates for event in proof.events}
      if not set(dependencies) <= events:
        return False
      matched = tuple(
        (key, proof) for key, proof in candidates if set(proof.events) & set(dependencies)
      )
      if any((key[0], key[1], "dependency") in self._used_awaits for key, _ in matched):
        return False
      for key, _proof in matched:
        self._used_awaits.add((key[0], key[1], "dependency"))
        record.consumed_awaits.add(f"dependency:{key[1]}")
      return True
    return False

  def _resolve_ranges(self, owner: str, command: MemoryMaintenanceDesc) -> tuple[CacheRange, ...]:
    handles = self._owner_inputs.get(owner)
    if handles is None:
      raise MemoryInvariantError("maintenance owner inputs were not bound")
    result: list[CacheRange] = []
    for item in command.ranges:
      if not 0 <= item.input_index < len(handles):
        raise MemoryInvariantError("maintenance input index is out of range")
      handle = handles[item.input_index]
      if item.offset < 0 or item.bytes <= 0 or item.offset + item.bytes > handle.size_bytes:
        raise MemoryInvariantError("maintenance range exceeds actual binding")
      result.append(CacheRange(handle.allocation_id, handle.generation, item.offset, item.bytes))
    return tuple(result)

  def _maintain_caches(
    self, record: _Command, cycle: int, *, levels: tuple[str, ...], ranges: tuple[CacheRange, ...] | None
  ) -> bool:
    if not record.maintenance_started:
      record.maintenance_started = True
      record.maintenance_owner = ExternalOwner(f"profile:{record.runtime_id}")
      try:
        for level in levels:
          for cache in self._caches(level):
            for request in cache.begin_maintenance(ranges, clean=True, invalidate=True):
              record.pending_clean_requests.append(_PendingClean(cache, request, level))
      except Exception as exc:
        self._fault(record, cycle, f"cache maintenance setup failed: {exc}")
        return False

    self._clean_transactions_done(record, cycle)
    if record.state == "faulted":
      return False

    manager = self._transfer_manager()
    issue_limit = manager.tombstone_capacity
    while (
      record.pending_clean_requests
      and len(record.clean_transactions) < issue_limit
      and manager.can_accept_new
      and manager.available_issue_slots
    ):
      pending = record.pending_clean_requests[0]
      try:
        transaction = self._clean_transaction(record, pending.cache, pending.request, pending.level, cycle)
        manager.submit(transaction, cycle)
      except Exception as exc:
        self._fault(record, cycle, f"cache clean submission failed: {exc}")
        return False
      record.pending_clean_requests.popleft()
      record.clean_transactions[pending.request.request_id] = (pending.cache, transaction.transaction_id)
    return not record.cleanup_pending

  @staticmethod
  def _discard_pending_clean_requests(record: _Command) -> None:
    """Unpin cache lines whose clean transaction was never submitted."""
    while record.pending_clean_requests:
      pending = record.pending_clean_requests[0]
      pending.cache.complete_clean(pending.request.request_id, success=False)
      record.pending_clean_requests.popleft()

  def _clean_transaction(
    self, record: _Command, cache: DeterministicLRUCache, request: CacheCleanRequest, level: str, cycle: int
  ) -> MemoryTransaction:
    hbm = self._require_attr(self.group, "hbm")
    handle = hbm.get_handle_by_allocation_id(request.provenance.allocation_id)
    if handle is None or handle.generation != request.provenance.allocation_generation:
      raise MemoryInvariantError("dirty cache provenance references stale HBM")
    segments = hbm.resolve(
      handle, request.provenance.source_offset, request.provenance.bytes, required_permission="w"
    )
    destination = ResolvedMemoryView(
      handle=handle,
      offset_bytes=request.provenance.source_offset,
      size_bytes=request.provenance.bytes,
      address=segments[0].address,
      segments=segments,
      permissions="w",
    )
    profile_generations = [(level, cache.pool_id, cache.profile_generation)]
    if level == "l1":
      profile_generations.append(("l2", 0, self._generations_mut["l2"]))
    assert record.maintenance_owner is not None
    return MemoryTransaction(
      transaction_id=f"{record.runtime_id}:{request.request_id}",
      op=TransferOp.CACHE_CLEAN_L1 if level == "l1" else TransferOp.CACHE_CLEAN_L2,
      issuer=record.maintenance_owner,
      src=None,
      dst=destination,
      bytes_total=len(request.data),
      completion_event=record.command_id,
      tile_id=cache.pool_id if level == "l1" else None,
      captured_data=request.data,
      run_generation=self._run_generation,
      profile_generations=tuple(profile_generations),
    )

  def _clean_transactions_done(self, record: _Command, cycle: int) -> bool:
    manager = self._transfer_manager()
    for request_id, (cache, transaction_id) in tuple(record.clean_transactions.items()):
      status = manager.status(transaction_id)
      if status is TransferStatus.DONE:
        manager.acknowledge(transaction_id, cycle)
        cache.complete_clean(request_id, success=True)
        del record.clean_transactions[request_id]
      elif status in (TransferStatus.FAULTED, TransferStatus.CANCELLED):
        manager.acknowledge(transaction_id, cycle)
        cache.complete_clean(request_id, success=False)
        del record.clean_transactions[request_id]
        self._fault(record, cycle, f"cache clean transfer {status.value}")
        return False
    return not record.clean_transactions

  # -- publication/cancel/fault --------------------------------------

  def _publish_profile(self, command: ProfileReconfigDesc, cycle: int) -> None:
    assert self._registry is not None
    profile = self._registry.profile(command.level, command.target_mode)
    generation = self._generations_mut[command.level] + 1
    for pool_id in range(profile.pools):
      self._pool(command.level, pool_id).reconfigure(profile, generation, cycle)
    for cache in self._caches(command.level):
      cache.reconfigure(profile.cache_bytes, generation, write_policy=profile.cache_write_policy)
    for mshr in self._mshrs(command.level):
      mshr.reconfigure(generation)
    for member_id in command.member_ids:
      member = self._members[member_id]
      if (
        member.committed_runtime_id != self._commands[command.command_id].runtime_id
        or member.shadow_mode != command.target_mode
        or member.shadow_generation != generation
      ):
        raise MemoryInvariantError("member commit publication lacks confirmed shadow")
      member.active_mode = command.target_mode
      member.generation = generation
      member.prepared_runtime_id = ""
      member.committed_runtime_id = ""
    self._active_modes_mut[command.level] = command.target_mode
    self._generations_mut[command.level] = generation
    self._transfer_manager().note_profile_commit(command.level, tuple(range(profile.pools)), generation)

  def _isolation_complete(self, record: _Command) -> bool:
    queued = any(item.runtime_id == record.runtime_id for item in self._requests) or any(
      item.runtime_id == record.runtime_id for item in self._responses
    )
    return not queued and not record.cleanup_pending

  def _advance_cancel(self, record: _Command, cycle: int) -> None:
    self._discard_pending_clean_requests(record)
    if record.maintenance_owner is not None:
      self._transfer_manager().cancel_owner(record.maintenance_owner, cycle)
    self._clean_transactions_done_for_cancel(record, cycle)
    if not self._isolation_complete(record):
      return
    # A cancelled maintenance command releases its range gates once isolation
    # is confirmed; only faulted commands stay latched for explicit recovery.
    if record.kind == "memory_maintenance":
      assert isinstance(record.descriptor, MemoryMaintenanceDesc)
      for level in record.descriptor.levels:
        self._range_gates.pop(level, None)
    # Once a profile command closed issue, cancellation never opens the gate;
    # explicit recovery/reset owns reopening after isolation.
    self._finish(record, "cancelled", cycle, "cancelled after transaction isolation")

  def _advance_fault_cleanup(self, record: _Command, cycle: int) -> None:
    """Continue isolation without changing the command's terminal fault."""
    self._discard_pending_clean_requests(record)
    if record.maintenance_owner is not None:
      self._transfer_manager().cancel_owner(record.maintenance_owner, cycle)
    self._clean_transactions_done_for_cancel(record, cycle)

  def _clean_transactions_done_for_cancel(self, record: _Command, cycle: int) -> None:
    if not record.clean_transactions:
      return
    manager = self._transfer_manager()
    for request_id, (cache, transaction_id) in tuple(record.clean_transactions.items()):
      status = manager.status(transaction_id)
      if status in (TransferStatus.DONE, TransferStatus.CANCELLED, TransferStatus.FAULTED):
        manager.acknowledge(transaction_id, cycle)
        cache.complete_clean(request_id, success=status is TransferStatus.DONE)
        del record.clean_transactions[request_id]

  def _fault(self, record: _Command, cycle: int, reason: str) -> None:
    first_fault = record.state != "faulted"
    if first_fault:
      record.reason = reason
      record.completed_cycle = cycle
      record.state = "faulted"
    # A timeout, failed ACK or half-commit must leave the issue gate closed.
    # Range gates stay latched until explicit recovery; the profile gate is
    # closed for the affected layer and only reopened by a fresh explicit
    # command or reset recovery.
    if record.kind == "profile_reconfig":
      assert isinstance(record.descriptor, ProfileReconfigDesc)
      self._profile_gates[record.descriptor.level] = True
    self._advance_fault_cleanup(record, cycle)
    if first_fault:
      self._trace_command(record, cycle)

  def _finish(self, record: _Command, state: str, cycle: int, reason: str) -> None:
    record.state = state
    record.reason = reason
    record.completed_cycle = cycle
    self._trace_command(record, cycle)

  # -- deterministic observability -----------------------------------

  def snapshot(self) -> dict[str, object]:
    return {
      "initialized": self._initialized,
      "initializing": self._initializing,
      "initialization_fault": self._initialization_fault,
      "recovering": self._recovering,
      "recovery_phase": self._recovery_phase if self._recovering else "",
      "run_generation": self._run_generation,
      "registry_hash": self._registry.registry_hash if self._registry is not None else "",
      "active_modes": dict(self._active_modes_mut),
      "generations": dict(self._generations_mut),
      "issue_gates": dict(self._profile_gates),
      "startup_authorized": dict(self._startup_authorized),
      "dependency_receipts": {
        "recorded": len(self._dependency_receipts),
        "unconsumed": sum(not item.consumed for item in self._dependency_receipts),
      },
      "range_gates": {
        level: tuple(
          (item.allocation_id, item.allocation_generation, item.offset, item.bytes) for item in ranges
        )
        for level, ranges in self._range_gates.items()
      },
      "member_bus": {
        "requests": len(self._requests),
        "responses": len(self._responses),
        "ignored_acks": self._ignored_acks,
        "initialization_acks": len(self._initialization_acks),
      },
      "members": tuple(
        {
          "member_id": member.member_id,
          "active_mode": member.active_mode,
          "generation": member.generation,
          "shadow_mode": member.shadow_mode,
          "shadow_generation": member.shadow_generation,
          "prepared_runtime_id": member.prepared_runtime_id,
          "committed_runtime_id": member.committed_runtime_id,
        }
        for member in sorted(self._members.values(), key=lambda item: item.member_id)
      ),
      "commands": {
        command_id: {
          "runtime_id": record.runtime_id,
          "owner": record.owner,
          "kind": record.kind,
          "state": record.state,
          "step": record.step,
          "accepted_cycle": record.accepted_cycle,
          "completed_cycle": record.completed_cycle,
          "reason": record.reason,
        }
        for command_id, record in sorted(self._commands.items())
      },
    }

  def _trace_event(self, name: str, cycle: int, args: dict[str, Any]) -> None:
    if self.trace is None:
      return
    method = getattr(self.trace, name, None)
    if callable(method):
      method(cycle, args)

  def _trace_command(self, record: _Command, cycle: int) -> None:
    if self.trace is None:
      return
    method = getattr(self.trace, "profile_command", None)
    if callable(method):
      method(record, cycle)

  def _trace_step(self, record: _Command, cycle: int) -> None:
    if self.trace is None:
      return
    method = getattr(self.trace, "profile_step", None)
    if callable(method):
      method(record, cycle)

  def _trace_member(self, item: object, cycle: int, status: str) -> None:
    if self.trace is None:
      return
    method = getattr(self.trace, "profile_member_ack", None)
    if callable(method):
      method(item, cycle, status)
