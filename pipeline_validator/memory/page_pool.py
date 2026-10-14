"""Generic host-managed page pool with scope ownership (plan §4).

The registry tracks software page ownership for ``nexus.host.call.async``
routines and runtime-enforced scopes.  Pages live inside one backing HBM
binding per pool; a scope owns whole pages between ``HostAllocPages`` and
``HostFreePages``.

Runtime seam for tile gather/scatter enforcement (wired by the MFE slice,
not here): a scoped tile access resolves byte segments against a global
view.  Before a sub-transaction is submitted the engine calls
``lease = registry.lease(scope, binding_name, offset, bytes)``, which
raises ``memory_scope_violation`` unless the byte range lies inside pages
the scope currently owns; then ``lease.acquire(txn_id)`` at submit and
``lease.release(txn_id)`` after the transaction's terminal
acknowledgement.  ``lease.is_valid()`` turns false as soon as any covered
page is freed or re-granted at a new epoch, so
``TileGroup.validate_transaction_generation`` can reject a late return
that would otherwise write a new owner's pages.  This module never calls
the engines itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn

from .allocator import AllocationHandle, MemoryInvariantError

if TYPE_CHECKING:
  from .byte_store import ByteStore
  from .hbm_region import HBMRegion


_POOL_PAGE_ALIGNMENT_BYTES = 64


class PoolPageError(MemoryInvariantError):
  """Page-pool contract violation; messages carry the plan §4 error codes."""


def _raise(code: str, message: str) -> NoReturn:
  raise PoolPageError(f"{code}: {message}")


@dataclass(frozen=True)
class HostPagePoolSpec:
  """Static declaration of one managed page pool (plan §4)."""

  name: str
  binding: str
  page_bytes: int
  page_count: int
  initial_owners: Mapping[str, tuple[int, ...]]

  def __post_init__(self) -> None:
    if not isinstance(self.name, str) or not self.name.strip():
      raise ValueError("host page pool name must be a non-empty string")
    if not isinstance(self.binding, str) or not self.binding.strip():
      raise ValueError("host page pool binding must be a non-empty string")
    if type(self.page_bytes) is not int or self.page_bytes <= 0:
      raise ValueError("host page pool page_bytes must be positive")
    if self.page_bytes % _POOL_PAGE_ALIGNMENT_BYTES:
      raise ValueError("host page pool page_bytes must be 64-byte aligned")
    if type(self.page_count) is not int or self.page_count <= 0:
      raise ValueError("host page pool page_count must be positive")
    owners: dict[str, tuple[int, ...]] = {}
    for scope, pages in self.initial_owners.items():
      if not isinstance(scope, str) or not scope.strip():
        raise ValueError("host page pool initial owner scope must be a non-empty string")
      page_tuple = tuple(pages)
      if not page_tuple:
        raise ValueError("host page pool initial owner must list at least one page")
      for page in page_tuple:
        if type(page) is not int or isinstance(page, bool) or not 0 <= page < self.page_count:
          raise ValueError(f"host page pool initial owner page id {page!r} is out of range")
      if len(page_tuple) != len(set(page_tuple)):
        raise ValueError(f"host page pool initial owner scope '{scope}' repeats a page")
      owners[scope] = page_tuple
    all_pages: list[int] = []
    for pages in owners.values():
      all_pages.extend(pages)
    if len(all_pages) != len(set(all_pages)):
      raise ValueError("host page pool initial owners overlap on one page")
    object.__setattr__(self, "initial_owners", owners)


@dataclass(frozen=True)
class PageAllocation:
  """Result of one successful ``PagePoolRegistry.allocate``."""

  scope: str
  pool: str
  pages: tuple[int, ...]
  epochs: tuple[int, ...]


class AccessLease:
  """Validity token for one resolved scoped byte range (plan §资源表).

  ``is_valid`` stays true while every covered page is still owned by the
  lease's scope at the same epoch.  Engines acquire one transaction id at
  submit and release it at terminal acknowledgement; acquiring through a
  stale lease faults instead of silently writing a new owner.
  """

  def __init__(self, registry: PagePoolRegistry, scope: str, epochs: tuple[tuple[int, int], ...]):
    self._registry = registry
    self._scope = scope
    self._epochs = epochs
    self._acquired: set[str] = set()

  @property
  def scope(self) -> str:
    return self._scope

  @property
  def epochs(self) -> tuple[tuple[int, int], ...]:
    return self._epochs

  @property
  def acquired_ids(self) -> tuple[str, ...]:
    return tuple(sorted(self._acquired))

  def is_valid(self) -> bool:
    registry = self._registry
    return all(
      registry._page_epoch(page) == epoch and registry._page_owner(page) == self._scope
      for page, epoch in self._epochs
    )

  def acquire(self, transaction_id: str) -> None:
    if not self.is_valid():
      _raise(
        "host_scope_inactive",
        f"scope '{self._scope}' lease is stale; transaction '{transaction_id}'"
        " may not touch pages with a new owner",
      )
    self._acquired.add(transaction_id)

  def release(self, transaction_id: str) -> None:
    self._acquired.discard(transaction_id)
    self._registry._release_lease(self._scope)


@dataclass
class _ScopeState:
  """Per-scope page ownership, root pins and live lease count."""

  pages: dict[int, int] = field(default_factory=dict)  # page id -> owning epoch
  pinned_roots: set[int] = field(default_factory=set)
  leases: int = 0


class PagePoolRegistry:
  """Software page-ownership oracle shared by HostSession and scopes."""

  def __init__(self) -> None:
    self._specs: dict[str, HostPagePoolSpec] = {}
    self._scope_pool: dict[str, str] = {}
    self._pool_handles: dict[str, AllocationHandle] = {}
    self._scope_states: dict[str, _ScopeState] = {}
    self._page_epochs: dict[int, int] = {}
    self._root_pins: dict[int, set[str]] = {}  # request id -> pinned scopes
    self._initial_counts: dict[str, int] = {}
    self._allocated_counts: dict[str, int] = {}
    self._freed_counts: dict[str, int] = {}
    self._peaks: dict[str, int] = {}
    self._byte_store: ByteStore | None = None
    self._run_generation = -1
    self._closed = False

  # -- construction ------------------------------------------------------

  def initialize(
    self,
    specs: Sequence[HostPagePoolSpec],
    bindings: Mapping[str, object],
    hbm: HBMRegion,
    byte_store: ByteStore | None,
    run_generation: int,
    *,
    is_binding_cached: Callable[[str], bool] | None = None,
  ) -> None:
    """Register pools against actual launch bindings (plan §4 init order).

    ``is_binding_cached`` is the oracle hook the caller wires to the
    ByteStore: a pool whose binding still carries cache lines is rejected
    with ``managed_pool_has_cached_state`` before the session starts.
    """
    if self._specs:
      _raise("host_pool_registered", "page pool registry is already initialized")
    if type(run_generation) is not int or run_generation < 0:
      raise ValueError("run_generation must be a non-negative integer")
    for spec in specs:
      if not isinstance(spec, HostPagePoolSpec):
        raise ValueError("page pool registry requires HostPagePoolSpec values")
      if spec.name in self._specs:
        _raise("host_pool_registered", f"page pool '{spec.name}' is registered twice")
      bindings_count = sum(1 for other in specs if other.binding == spec.binding)
      if bindings_count != 1:
        _raise("host_pool_registered", f"pool '{spec.name}' binding '{spec.binding}' is not unique")
      if spec.binding not in bindings:
        raise ValueError(
          f"page pool '{spec.name}' names binding '{spec.binding}' with no actual launch binding"
        )
    scope_owners: dict[str, str] = {}
    for spec in specs:
      for scope in spec.initial_owners:
        if scope in scope_owners:
          _raise(
            "host_pool_registered",
            f"scope '{scope}' is declared by pools '{scope_owners[scope]}' and '{spec.name}'",
          )
        scope_owners[scope] = spec.name

    for spec in specs:
      handle = hbm.get_handle(spec.binding)
      if handle is None:
        raise ValueError(
          f"page pool '{spec.name}' references binding '{spec.binding}'"
          " that the Group has not bound"
        )
      if handle.size_bytes < spec.page_count * spec.page_bytes:
        _raise(
          "host_pool_capacity",
          f"page pool '{spec.name}' exceeds binding '{spec.binding}' capacity",
        )
      self._specs[spec.name] = spec
      self._pool_handles[spec.name] = handle
      self._initial_counts[spec.name] = 0
      self._allocated_counts[spec.name] = 0
      self._freed_counts[spec.name] = 0
      self._peaks[spec.name] = 0
      self._scope_pool.update(dict.fromkeys(spec.initial_owners, spec.name))
      self._scope_states.update({scope: _ScopeState() for scope in spec.initial_owners})
    for scope, pool_name in scope_owners.items():
      for page in self._specs[pool_name].initial_owners[scope]:
        self._grant_page(pool_name, scope, page)
        self._initial_counts[pool_name] += 1
    for pool_name in self._specs:
      self._peaks[pool_name] = self._pool_live(pool_name)
    self._byte_store = byte_store
    self._run_generation = run_generation
    if byte_store is not None and is_binding_cached is not None:
      for spec in specs:
        if is_binding_cached(spec.binding):
          _raise(
            "managed_pool_has_cached_state",
            f"binding '{spec.binding}' of pool '{spec.name}' still holds cache lines",
          )

  # -- ownership helpers -------------------------------------------------

  def _require_open(self) -> None:
    if self._closed:
      _raise("host_pool_closed", "page pool registry is closed")

  def _require_scope(self, scope: str, pool: str | None = None) -> _ScopeState:
    state = self._scope_states.get(scope)
    if state is None:
      _raise("host_scope_inactive", f"scope '{scope}' belongs to no registered pool")
    owner = self._scope_pool[scope]
    if pool is not None and owner != pool:
      _raise(
        "host_pool_unknown_scope",
        f"scope '{scope}' belongs to pool '{owner}', not '{pool}'",
      )
    return state

  def _grant_page(self, pool: str, scope: str, page: int) -> None:
    del pool  # pool identity is implicit in the scope's owner
    self._page_epochs.setdefault(page, 0)
    state = self._scope_states[scope]
    state.pages[page] = self._page_epochs[page]

  def _page_epoch(self, page: int) -> int:
    return self._page_epochs.get(page, -1)

  def _page_owner(self, page: int) -> str | None:
    for scope, state in self._scope_states.items():
      if page in state.pages:
        return scope
    return None

  def _pool_scopes(self, pool: str) -> tuple[str, ...]:
    return tuple(scope for scope, name in self._scope_pool.items() if name == pool)

  def _pool_live(self, pool: str) -> int:
    return sum(len(self._scope_states[scope].pages) for scope in self._pool_scopes(pool))

  def _release_lease(self, scope: str) -> None:
    state = self._scope_states.get(scope)
    if state is None or state.leases <= 0:
      _raise("host_scope_inactive", f"scope '{scope}' released a lease it never held")
    state.leases -= 1

  def _invalidate_pages(self, pool: str, pages: Sequence[int]) -> None:
    """Clear HBM validity bits and stale cache copies of freed pages."""
    if self._byte_store is None:
      return
    spec = self._specs[pool]
    handle = self._pool_handles[pool]
    for page in pages:
      self._byte_store.invalidate_binding_range(
        handle, page * spec.page_bytes, spec.page_bytes, binding_name=spec.binding
      )

  # -- public API (plan §4) ----------------------------------------------

  def binding_for_scope(self, scope: str) -> str:
    """Backing binding name of the pool that owns ``scope``."""
    self._require_scope(scope)
    return self._specs[self._scope_pool[scope]].binding

  def pool_for_scope(self, scope: str) -> str:
    self._require_scope(scope)
    return self._scope_pool[scope]

  def is_pool_binding(self, binding: str) -> bool:
    """True when ``binding`` is the backing binding of a registered pool."""
    return any(spec.binding == binding for spec in self._specs.values())

  def pool_for_binding(self, binding: str) -> str:
    """Pool name owning ``binding``; raises when it backs no pool."""
    for pool, spec in self._specs.items():
      if spec.binding == binding:
        return pool
    _raise("host_pool_unknown", f"binding '{binding}' backs no registered pool")

  def scope_owns_range(self, scope: str, binding: str, offset: int, size: int) -> bool:
    """True when [offset, offset+size) lies inside pages the scope owns."""
    if size <= 0:
      return False
    if binding != self.binding_for_scope(scope):
      return False
    spec = self._specs[self._scope_pool[scope]]
    if offset < 0 or offset + size > spec.page_count * spec.page_bytes:
      return False
    first, last = offset // spec.page_bytes, (offset + size - 1) // spec.page_bytes
    state = self._scope_states[scope]
    return all(page in state.pages for page in range(first, last + 1))

  def lease(self, scope: str, binding: str, offset: int, size: int) -> AccessLease:
    """Validate a scoped access range and return its validity token."""
    if not self.scope_owns_range(scope, binding, offset, size):
      _raise(
        "memory_scope_violation",
        f"scoped access [{offset}, {offset + size}) on '{binding}' does not lie"
        f" inside pages currently owned by scope '{scope}'",
      )
    state = self._require_scope(scope)
    spec = self._specs[self._scope_pool[scope]]
    first, last = offset // spec.page_bytes, (offset + size - 1) // spec.page_bytes
    epochs = tuple((page, state.pages[page]) for page in range(first, last + 1))
    state.leases += 1
    return AccessLease(self, scope, epochs)

  def allocate(self, pool: str, scope: str, count: int, *, cycle: int) -> PageAllocation:
    """Lowest-free-page allocation; atomic on failure (plan §4)."""
    del cycle
    self._require_open()
    if pool not in self._specs:
      _raise("host_pool_unknown", f"page pool '{pool}' is not registered")
    state = self._require_scope(scope, pool)
    if type(count) is not int or isinstance(count, bool) or count <= 0:
      raise ValueError("page allocation count must be a positive integer")
    if state.leases or state.pinned_roots:
      _raise(
        "host_pool_busy",
        f"scope '{scope}' has {state.leases} active leases and {len(state.pinned_roots)}"
        " root pins; page allocation requires none",
      )
    spec = self._specs[pool]
    owned: set[int] = set()
    for pool_scope in self._pool_scopes(pool):
      owned.update(self._scope_states[pool_scope].pages)
    free = [page for page in range(spec.page_count) if page not in owned]
    if len(free) < count:
      _raise(
        "host_pool_out_of_pages",
        f"pool '{pool}' cannot allocate {count} pages for scope '{scope}';"
        f" only {len(free)} free pages remain",
      )
    pages = tuple(sorted(free)[:count])
    for page in pages:
      self._page_epochs[page] = self._page_epochs.get(page, 0) + 1
      self._grant_page(pool, scope, page)
    self._invalidate_pages(pool, pages)
    self._allocated_counts[pool] += count
    self._peaks[pool] = max(self._peaks[pool], self._pool_live(pool))
    return PageAllocation(scope, pool, pages, tuple(self._page_epochs[page] for page in pages))

  def free(self, pool: str, scope: str, *, cycle: int) -> None:
    """Release every page owned by a scope; requires no pins or leases."""
    del cycle
    self._require_open()
    if pool not in self._specs:
      _raise("host_pool_unknown", f"page pool '{pool}' is not registered")
    state = self._require_scope(scope, pool)
    if not state.pages:
      _raise("host_scope_inactive", f"scope '{scope}' owns no pages and cannot be freed again")
    if state.leases or state.pinned_roots:
      _raise(
        "host_pool_busy",
        f"scope '{scope}' has {state.leases} active leases and {len(state.pinned_roots)}"
        " root pins; page free requires none",
      )
    released = tuple(state.pages)
    state.pages.clear()
    for page in released:
      self._page_epochs[page] += 1
    self._freed_counts[pool] += len(released)

  def pin_root(self, request_id: int, scope_bindings: Mapping[str, str], *, cycle: int) -> None:
    """Atomically pin every scope backing one root request (plan §4).

    Pinned scopes refuse allocate/free until ``unpin_root``; rejected or
    backpressured requests roll the pin back before returning.
    """
    del cycle
    self._require_open()
    if request_id in self._root_pins:
      raise ValueError(f"root request {request_id} is already pinned")
    scopes: list[str] = []
    for binding, scope in scope_bindings.items():
      self._require_scope(scope)
      owner_binding = self._specs[self._scope_pool[scope]].binding
      if owner_binding != binding:
        _raise(
          "scope_binding_mismatch",
          f"scope '{scope}' is owned by pool binding '{owner_binding}', not '{binding}'",
        )
      scopes.append(scope)
    for scope in scopes:
      self._scope_states[scope].pinned_roots.add(request_id)
    self._root_pins[request_id] = set(scopes)

  def unpin_root(self, request_id: int, *, cycle: int) -> None:
    del cycle
    scopes = self._root_pins.pop(request_id, None)
    if scopes is None:
      raise ValueError(f"root request {request_id} has no pin to release")
    for scope in scopes:
      self._scope_states[scope].pinned_roots.discard(request_id)

  # -- accounting --------------------------------------------------------

  def live_page_count(self) -> int:
    return sum(len(state.pages) for state in self._scope_states.values())

  def abort_after_drain(self, *, cycle: int) -> None:
    """Fault-path teardown after the ResetDomain drain reached DONE."""
    del cycle
    self._require_open()
    for state in self._scope_states.values():
      state.pages.clear()
      state.pinned_roots.clear()
      state.leases = 0
    self._root_pins.clear()
    self._closed = True

  def assert_closed(self) -> None:
    """Success-path closure check: no surviving ownership of any kind."""
    if self._closed:
      return
    survivors: list[str] = []
    for scope, state in self._scope_states.items():
      if state.pages:
        survivors.append(f"scope '{scope}' still owns {len(state.pages)} pages")
      if state.leases:
        survivors.append(f"scope '{scope}' still holds {state.leases} leases")
      if state.pinned_roots:
        survivors.append(f"scope '{scope}' still pins {len(state.pinned_roots)} roots")
    if self._root_pins:
      survivors.append(f"{len(self._root_pins)} root pin records remain")
    if survivors:
      _raise("host_pool_busy", "; ".join(survivors))
    for pool in self._specs:
      granted = self._initial_counts[pool] + self._allocated_counts[pool]
      freed = self._freed_counts[pool]
      if granted != freed:
        _raise(
          "host_pool_busy",
          f"pool '{pool}' conservation violation: {granted - freed} pages leaked",
        )

  def snapshot(self) -> dict[str, object]:
    pools: dict[str, object] = {}
    for pool, spec in self._specs.items():
      scopes = self._pool_scopes(pool)
      pools[pool] = {
        "binding": spec.binding,
        "page_bytes": spec.page_bytes,
        "page_count": spec.page_count,
        "initial_pages": self._initial_counts[pool],
        "allocated_pages": self._allocated_counts[pool],
        "freed_pages": self._freed_counts[pool],
        "live_pages": self._pool_live(pool),
        "peak_live_pages": self._peaks[pool],
        "pins": sum(len(self._scope_states[scope].pinned_roots) for scope in scopes),
        "leases": sum(self._scope_states[scope].leases for scope in scopes),
      }
    return {
      "run_generation": self._run_generation,
      "closed": self._closed,
      "pools": pools,
      "scopes": {
        scope: {
          "pool": self._scope_pool[scope],
          "pages": sorted(state.pages),
          "epochs": {str(page): epoch for page, epoch in sorted(state.pages.items())},
          "pins": len(state.pinned_roots),
          "leases": state.leases,
        }
        for scope, state in self._scope_states.items()
      },
    }
