"""Host runtime session: ``nexus.host.call.async`` routines (plan §4).

The session is the hardware-facing ``HostPort`` the CPU model talks to.
It owns one generator-backed routine per issued ``host_call``:

- each host_call first pays ``hw.host_patch_cycles`` (software patch cost);
- afterwards at most ``sim.device.issue_width`` commands are issued per
  cycle, round-robin across routines;
- one routine has at most one outstanding command; its result is
  delivered at harvest and the next command issues no earlier than the
  following cycle;
- active routines are capped by ``sim.device.pending_capacity``;
- reads and writes are billed through the same ``TransferManager``/
  ``ByteStore`` pair as device traffic (``TransferOp.HOST_READ`` /
  ``TransferOp.HOST_WRITE``); handlers never touch the ByteStore.

Clock order owned by ``Simulator``: ``controller.step`` → ``host.step`` →
``group.step`` → ``host.harvest`` → ``controller.harvest_completions``.
A command completing in cycle ``N`` therefore delivers its result in the
harvest of ``N`` and the routine's next command issues at ``N + 1`` at
the earliest.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import HardwareConfig, SimConfig
from ..device import DeviceCompletion, DeviceCompletionStatus, DeviceHostRequest, HostPort
from ..memory.allocator import ExternalOwner
from ..memory.page_pool import HostPagePoolSpec, PageAllocation, PagePoolRegistry, PoolPageError
from ..memory.transfer import MemoryTransaction, ResolvedMemoryView, TransferOp, TransferStatus

if TYPE_CHECKING:
  from ..tile_group import TileGroup
  from ..trace import Tracer


class HostCommand:
  """Base class of host routine commands; never instantiated."""


@dataclass(frozen=True)
class HostRead(HostCommand):
  binding: str
  offset: int
  bytes: int
  scope: str | None = None


@dataclass(frozen=True)
class HostWrite(HostCommand):
  binding: str
  offset: int
  data: bytes
  scope: str | None = None


@dataclass(frozen=True)
class HostAllocPages(HostCommand):
  pool: str
  scope: str
  count: int


@dataclass(frozen=True)
class HostFreePages(HostCommand):
  pool: str
  scope: str


@dataclass(frozen=True)
class HostDelay(HostCommand):
  cycles: int


HostResult = bytes | PageAllocation | None
HostRoutineFactory = Callable[[DeviceHostRequest], Generator[HostCommand, HostResult, None]]

_NOT_READY = object()


class HostEnvironment:
  """Static description of the software host: handlers and page pools."""

  def __init__(
    self,
    handlers: Mapping[str, HostRoutineFactory],
    pools: tuple[HostPagePoolSpec, ...] = (),
  ) -> None:
    for name, handler in handlers.items():
      if not name.strip():
        raise ValueError("host handler names must be non-empty strings")
      if not callable(handler):
        raise ValueError(f"host handler '{name}' must be callable")
    self.handlers: Mapping[str, HostRoutineFactory] = dict(handlers)
    self.pools: tuple[HostPagePoolSpec, ...] = tuple(pools)
    for pool in self.pools:
      if not isinstance(pool, HostPagePoolSpec):
        raise ValueError("host pools must be HostPagePoolSpec values")


class _Routine:
  """One live host_call: its generator and outstanding command."""

  __slots__ = (
    "command_ordinal",
    "current",
    "delay_ready_cycle",
    "generator",
    "issued_commands",
    "pool_result",
    "ready_cycle",
    "request",
    "staged",
    "transaction",
    "transaction_id",
  )

  def __init__(self, request: DeviceHostRequest, generator, ready_cycle: int) -> None:
    self.request = request
    self.generator = generator
    self.ready_cycle = ready_cycle
    self.current: HostCommand | None = None
    self.staged: HostCommand | None = None
    self.transaction_id: str | None = None
    self.transaction: MemoryTransaction | None = None
    self.delay_ready_cycle: int | None = None
    self.pool_result: PageAllocation | None = None
    self.command_ordinal = 0
    self.issued_commands = 0

  @property
  def busy(self) -> bool:
    return self.current is not None


def _required_modes(mode: str) -> frozenset[str]:
  return {"read": frozenset({"r"}), "write": frozenset({"w"}), "readwrite": frozenset({"r", "w"})}[mode]


class HostSession(HostPort):
  """Concrete scheduling and command execution for one run generation."""

  def __init__(
    self,
    environment: HostEnvironment,
    registry: PagePoolRegistry,
    *,
    group: TileGroup,
    hw: HardwareConfig,
    sim: SimConfig,
  ) -> None:
    self._environment = environment
    self._registry = registry
    self._group = group
    self._hw = hw
    self._sim = sim
    self._transfer = group.transfer_manager
    self._hbm = group.hbm
    self._pmu = group.pmu
    self._tracer: Tracer | None = group.tracer
    self._owner = ExternalOwner(f"host:{group.run_generation}")
    self._run_generation = group.run_generation
    self._active: dict[int, _Routine] = {}
    self._rotation: deque[int] = deque()
    self._ready_completions: list[DeviceCompletion] = []
    self._next_transaction_seq = 0
    self._counters = {
      "calls": 0,
      "commands": 0,
      "software_cycles": 0,
      "read_bytes": 0,
      "write_bytes": 0,
    }

  # -- HostPort protocol -------------------------------------------------

  def try_submit(self, request: DeviceHostRequest, cycle: int) -> bool:
    """Accept one host_call atomically; invalid routines fail before issue."""
    handler = self._environment.handlers.get(request.name)
    if handler is None:
      raise ValueError(f"host handler '{request.name}' is not registered")
    if len(self._active) >= self._sim.device.pending_capacity:
      return False
    for scope in request.command.scopes:
      try:
        self._registry.pool_for_scope(scope)
      except PoolPageError as exc:
        raise ValueError(f"host_call '{request.name}' names unknown scope '{scope}'") from exc
    if len(request.command.accesses) != len(request.binding_names):
      raise ValueError(f"host_call '{request.name}' access/binding pairing is inconsistent")
    generator = handler(request)
    routine = _Routine(request, generator, cycle + self._hw.host_patch_cycles)
    self._active[request.request_id] = routine
    self._rotation.append(request.request_id)
    self._counters["calls"] += 1
    self._trace("host_call", cycle, {"request_id": request.request_id, "name": request.name})
    self.pmu_note()
    return True

  def step(self, cycle: int) -> None:
    """Issue up to ``issue_width`` host commands, round-robin."""
    width = self._sim.device.issue_width
    issued = 0
    for request_id in tuple(self._rotation):
      routine = self._active.get(request_id)
      if routine is None:
        self._rotation.remove(request_id)
        continue
      if routine.busy or cycle < routine.ready_cycle or issued >= width:
        continue
      if self._fetch_command(routine, cycle):
        issued += 1
    if issued:
      self._counters["software_cycles"] += 1
    self._rotation.rotate(-1)

  def poll_completions(self, cycle: int) -> tuple[DeviceCompletion, ...]:
    """Return host completions made visible by the preceding harvest."""
    completions = tuple(self._ready_completions)
    self._ready_completions.clear()
    return completions

  def abort(self, cycle: int) -> None:
    """Fault path: cancel unissued commands and in-flight host transfers."""
    self._transfer.cancel_owner(self._owner, cycle)
    for request_id, routine in tuple(self._active.items()):
      if routine.transaction_id is None:
        self._retire(routine, request_id, DeviceCompletionStatus.ERROR, "host runtime aborted", cycle)
        continue
      # In-flight transactions report through the harvest loop: the reset
      # drain isolates them terminal before their completion is delivered.

  def close(self) -> bool:
    """True when no routine survives; success paths call assert_closed()."""
    return not self._active

  # -- harvest -----------------------------------------------------------

  def harvest(self, cycle: int) -> None:
    """Deliver results after ``group.step``; completions visible same cycle."""
    for request_id in tuple(self._active):
      routine = self._active[request_id]
      if routine.current is None:
        continue
      if self._is_delay_command(routine.current):
        delay_ready = routine.delay_ready_cycle
        if delay_ready is None or cycle < delay_ready:
          continue
        self._deliver(routine, request_id, None, cycle)
        continue
      if isinstance(routine.current, (HostAllocPages, HostFreePages)):
        # Pool commands carry no transfer: they complete at issue and the
        # retained allocation is delivered here.
        self._deliver(routine, request_id, routine.pool_result, cycle)
        continue
      if routine.transaction_id is None or routine.transaction is None:
        continue
      transaction = routine.transaction
      transaction_id = routine.transaction_id
      status = self._transfer.status(transaction_id)
      if status in (TransferStatus.PENDING, TransferStatus.RUNNING, TransferStatus.CANCEL_REQUESTED):
        continue
      if status is TransferStatus.DONE:
        result = self._result_for(routine.current, transaction)
        self._transfer.acknowledge(transaction_id, cycle)
        self._count_bytes(routine.current)
        self._deliver(routine, request_id, result, cycle)
      else:
        self._transfer.acknowledge(transaction_id, cycle)
        reason = getattr(transaction, "fault_reason", "") or (
          f"host command {type(routine.current).__name__} ended {status.value}"
        )
        self._retire(
          routine,
          request_id,
          DeviceCompletionStatus.ERROR,
          reason,
          cycle,
        )

  # -- internals ---------------------------------------------------------

  @staticmethod
  def _is_delay_command(command: HostCommand) -> bool:
    return isinstance(command, HostDelay)

  def _result_for(self, command: HostCommand, transaction: MemoryTransaction | None) -> HostResult:
    if isinstance(command, HostRead):
      if transaction is None or transaction.captured_data is None:
        raise PoolPageError("host read completed without captured bytes")
      return transaction.captured_data
    return None

  def _count_bytes(self, command: HostCommand) -> None:
    if isinstance(command, HostRead):
      self._counters["read_bytes"] += command.bytes
    elif isinstance(command, HostWrite):
      self._counters["write_bytes"] += len(command.data)

  def _fetch_command(self, routine: _Routine, cycle: int) -> bool:
    """Issue the staged command, starting the generator on the first call."""
    try:
      command = routine.staged
      if command is None:
        command = next(routine.generator)
      if command is None:
        raise ValueError("host routine yielded None instead of a command")
      routine.staged = None
      self._issue(routine, command, cycle)
      return True
    except StopIteration:
      self._retire(routine, routine.request.request_id, DeviceCompletionStatus.SUCCESS, "", cycle)
      return False
    except (PoolPageError, ValueError, RuntimeError) as exc:
      self._retire(routine, routine.request.request_id, DeviceCompletionStatus.ERROR, str(exc), cycle)
      return False

  def _deliver(self, routine: _Routine, request_id: int, result: HostResult, cycle: int) -> None:
    """Send one result into the generator; the next command issues next cycle."""
    routine.transaction_id = None
    routine.transaction = None
    routine.delay_ready_cycle = None
    routine.current = None
    try:
      routine.staged = routine.generator.send(result)
    except StopIteration:
      self._retire(routine, request_id, DeviceCompletionStatus.SUCCESS, "", cycle)
      return
    except (PoolPageError, ValueError, RuntimeError) as exc:
      self._retire(routine, request_id, DeviceCompletionStatus.ERROR, str(exc), cycle)
      return

  def _retire(
    self,
    routine: _Routine,
    request_id: int,
    status: DeviceCompletionStatus,
    reason: str,
    cycle: int,
  ) -> None:
    self._active.pop(request_id, None)
    try:
      self._rotation.remove(request_id)
    except ValueError:
      pass
    try:
      routine.generator.close()
    except Exception:
      pass
    self._ready_completions.append(DeviceCompletion(request_id, status, reason, cycle))

  def _issue(self, routine: _Routine, command: HostCommand, cycle: int) -> None:
    """Validate one command against the host_call contract, then issue it."""
    request = routine.request
    if not isinstance(command, HostCommand):
      raise ValueError(f"host routine '{request.name}' yielded an unsupported command object")
    if isinstance(command, (HostRead, HostWrite)):
      self._validate_memory_command(routine, command)
      transaction = self._submit_memory_transaction(routine, command, cycle)
      routine.current = command
      routine.transaction_id = transaction.transaction_id
      routine.transaction = transaction
    elif isinstance(command, HostAllocPages):
      # Page-pool commands complete synchronously: the result is retained and
      # delivered by the next harvest, so the generator sees the page ids.
      allocation = self._registry.allocate(command.pool, command.scope, command.count, cycle=cycle)
      self._trace(
        "pool_allocate",
        cycle,
        {"pool": command.pool, "scope": command.scope, "pages": allocation.pages},
      )
      routine.current = command
      routine.pool_result = allocation
    elif isinstance(command, HostFreePages):
      self._registry.free(command.pool, command.scope, cycle=cycle)
      self._trace("pool_free", cycle, {"pool": command.pool, "scope": command.scope})
      routine.current = command
      routine.pool_result = None
    elif isinstance(command, HostDelay):
      if type(command.cycles) is not int or command.cycles < 0:
        raise ValueError("host delay must be a non-negative cycle count")
      routine.current = command
      routine.delay_ready_cycle = cycle + command.cycles
    else:
      raise ValueError(f"host routine '{request.name}' yielded an unknown command {type(command).__name__}")
    routine.command_ordinal += 1
    routine.issued_commands += 1
    self._counters["commands"] += 1
    self._trace(
      "host_command",
      cycle,
      {
        "request_id": request.request_id,
        "command": type(command).__name__,
        "ordinal": routine.command_ordinal,
      },
    )

  def _validate_memory_command(self, routine: _Routine, command: HostRead | HostWrite) -> None:
    request = routine.request
    declared: dict[str, list[tuple[int, int, str]]] = {}
    for binding, access in zip(request.binding_names, request.command.accesses):
      declared.setdefault(binding, []).append((access.offset, access.offset + access.bytes, access.mode))
    spans = declared.get(command.binding)
    if spans is None:
      raise ValueError(f"host routine '{request.name}' accesses unbound input '{command.binding}'")
    end = command.offset + (command.bytes if isinstance(command, HostRead) else len(command.data))
    if isinstance(command, HostRead) and command.bytes <= 0:
      raise ValueError("host read must request at least one byte")
    required = "r" if isinstance(command, HostRead) else "w"
    for start, stop, mode in spans:
      if start <= command.offset and end <= stop:
        if required not in _required_modes(mode):
          raise ValueError(
            f"host routine '{request.name}' {required}-accesses '{command.binding}'"
            " beyond its declared mode"
          )
        break
    else:
      raise ValueError(
        f"host routine '{request.name}' accesses '{command.binding}'"
        f" [{command.offset}, {end}) beyond its declared range"
      )
    if self._registry.is_pool_binding(command.binding):
      if command.scope is None:
        raise ValueError(
          f"managed pool binding '{command.binding}' requires a scope on host access"
        )
      try:
        self._registry.pool_for_scope(command.scope)
      except PoolPageError as exc:
        raise ValueError(f"host routine '{request.name}' names unknown scope '{command.scope}'") from exc
      if self._registry.pool_for_scope(command.scope) != self._registry.pool_for_binding(command.binding):
        raise ValueError(
          f"scope '{command.scope}' does not belong to the pool backing '{command.binding}'"
        )
      if not self._registry.scope_owns_range(
        command.scope, command.binding, command.offset, end - command.offset
      ):
        raise PoolPageError(
          f"memory_scope_violation: [{command.offset}, {end}) on '{command.binding}'"
          f" lies outside pages owned by scope '{command.scope}'"
        )

  def _submit_memory_transaction(
    self, routine: _Routine, command: HostRead | HostWrite, cycle: int
  ) -> MemoryTransaction:
    name = command.binding
    handle = self._hbm.get_handle(name)
    if handle is None:
      raise ValueError(f"host command references unbound input '{name}'")
    size = command.bytes if isinstance(command, HostRead) else len(command.data)
    required_permission = "r" if isinstance(command, HostRead) else "w"
    segments = self._hbm.resolve(handle, command.offset, size, required_permission=required_permission)
    view = ResolvedMemoryView(
      handle=handle,
      offset_bytes=command.offset,
      size_bytes=size,
      address=segments[0].address,
      segments=segments,
      permissions=self._hbm.permissions(name),
    )
    op = TransferOp.HOST_READ if isinstance(command, HostRead) else TransferOp.HOST_WRITE
    self._next_transaction_seq += 1
    run_generation, profile_generations = self._transfer.transaction_identity((("l2", 0),))
    transaction = MemoryTransaction(
      transaction_id=f"host:{run_generation}:{routine.request.request_id}:{self._next_transaction_seq}",
      op=op,
      issuer=self._owner,
      src=None if isinstance(command, HostWrite) else view,
      dst=None if isinstance(command, HostRead) else view,
      bytes_total=size,
      completion_event="",
      captured_data=None if isinstance(command, HostRead) else command.data,
      run_generation=run_generation,
      profile_generations=profile_generations,
    )
    self._trace(
      "host_read" if isinstance(command, HostRead) else "host_write",
      cycle,
      {"binding": name, "offset": command.offset, "bytes": size, "scope": command.scope},
    )
    self._transfer.submit(transaction, cycle, self._pmu)
    return transaction

  def _trace(self, event: str, cycle: int, args: dict) -> None:
    if self._tracer is None:
      return
    self._tracer.instant("TileGroup", "Host", event, cycle, args)

  def pmu_note(self) -> None:
    self._pmu.add_event("device_host_call")

  def snapshot(self) -> dict:
    return {
      "active": len(self._active),
      "pending_capacity": self._sim.device.pending_capacity,
      "counters": dict(self._counters),
    }

  def assert_closed(self) -> None:
    if self._active:
      names = sorted(routine.request.name for routine in self._active.values())
      raise PoolPageError(f"host routines still live at exit: {names}")
