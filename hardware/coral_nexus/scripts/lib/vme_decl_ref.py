"""Reference model of the VME region declaration ABI (C012): E1 request/park, E2 consume, mode tables.

Everything here is DRIVEN BY `protocol/w3_vme_decl_abi_v1.json`: the state names, the event names,
the fault codes, the legal modes and the legal transitions are read from that file. A test that
deletes a transition from the JSON therefore changes what this model accepts; the model is not a
second copy of the contract.

This is a CONTRACT CHECK of the declaration protocol, not an RTL result. Nothing here is a
claim that W2/W3 hardware does it (that is C044/C053).

Two events, kept apart on purpose (W3 plan 16):
  E1  request/park  the declared target instruction is at the Context's issue head and valid and has
                    NOT entered the shared RVV queue. A free instance is granted at once; otherwise
                    ONE waiter is registered. A waiter never depends on the instruction being accepted.
  E2  consume       the granted instruction is accepted by the front end; the pending declaration is
                    consumed in that single beat. It is the only consume point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class DeclError(Exception):
  """Coded protocol fault. `code` is stable so tests pin the failure kind."""

  def __init__(self, code: str, detail: str) -> None:
    super().__init__(f"{code}: {detail}")
    self.code = code


@dataclass
class Pending:
  kind: str
  chain_id: int | None
  generation: int


@dataclass
class ContextState:
  generation: int = 1
  pending: Pending | None = None
  phase: str = "IDLE"  # IDLE | PARKED | ACTIVE | END_DRAIN | REGION_DONE | FAILED
  region_id: int | None = None
  waiter: bool = False
  granted: bool = False  # a grant exists whose target has not been accepted yet
  chains: dict[int, str] = field(default_factory=dict)  # chain_id -> OPEN | FINAL_SUBMITTED | FINAL_DONE
  end_declared: bool = False
  accepted_unsettled: int = 0  # ops accepted into the shared queue whose result is not settled
  settled_obligations: int = 0


class DeclModel:
  """One model instance serves `contexts` Contexts and `instances` VME instances."""

  def __init__(self, abi: dict[str, Any], contexts: int = 2, instances: int = 1) -> None:
    self.abi = abi
    self.kinds: set[str] = set(abi["declaration"]["kinds"])
    self.faults: dict[str, dict[str, Any]] = {f["code"]: f for f in abi["faults"]}
    self.events: set[str] = set(abi["events"])  # E1 / E2
    self.contexts = [ContextState() for _ in range(contexts)]
    self.instances: list[int | None] = [None] * instances  # instance -> owning ctx
    self.waiters: list[int] = []  # FIFO of ctx with a registered waiter
    self.queue: list[tuple[int, int]] = []  # shared RVV queue: (ctx, region_id)
    self.pmu = {"grant": 0, "park": 0, "wake": 0, "state_free": 0, "region_done": 0, "fault": 0}
    self._next_region = 1
    self.log: list[tuple[str, int, str]] = []

  # ---- helpers ---------------------------------------------------------------------------
  def _ctx(self, ctx: int) -> ContextState:
    if not 0 <= ctx < len(self.contexts):
      raise DeclError("CTX_OUT_OF_RANGE", f"ctx {ctx}")
    return self.contexts[ctx]

  def _fault(self, ctx: int, code: str, detail: str) -> DeclError:
    if code not in self.faults:
      raise DeclError("FAULT_CODE_UNKNOWN", code)
    self.pmu["fault"] += 1
    self._ctx(ctx).phase = "FAILED"
    self.log.append(("fault", ctx, code))
    return DeclError(code, detail)

  def _has_free(self) -> int | None:
    return next((i for i, owner in enumerate(self.instances) if owner is None), None)

  # ---- software side: csrw to the declaration CSR ------------------------------------------
  def csrw(self, ctx: int, kind: str, chain_id: int | None = None) -> None:
    """Arm the pending declaration. Only legal when the record is EMPTY (rule A1)."""
    st = self._ctx(ctx)
    if kind not in self.kinds:
      raise self._fault(ctx, "UNKNOWN_KIND", kind)
    if st.pending is not None:
      raise self._fault(ctx, "ARMED_CONFLICT", f"pending {st.pending.kind} already ARMED")
    if kind == "REGION_START" and st.phase in ("ACTIVE", "END_DRAIN", "PARKED"):
      raise self._fault(ctx, "NESTED_START", f"START while {st.phase}")
    st.pending = Pending(kind, chain_id, st.generation)
    self.log.append(("csrw", ctx, kind))

  # ---- E1: request / park ----------------------------------------------------------------
  def target_at_head(self, ctx: int) -> str:
    """E1. Returns 'GRANTED' or 'PARKED'. The target has NOT entered the shared queue."""
    st = self._ctx(ctx)
    p = st.pending
    if p is None:
      # A touching instruction with no declaration is F1 only when it needs a region.
      return "NOT_DECLARED"
    if p.generation != st.generation:
      raise self._fault(ctx, "STALE_GENERATION", "pending survived a Context generation change")
    if p.kind != "REGION_START":
      return "NOT_A_REGION_REQUEST"  # chain/END declarations need an ACTIVE region, no lease request
    if st.phase == "ACTIVE" or st.granted:
      return "GRANTED"  # idempotent: the head is re-presented while a grant stands
    if st.waiter:
      return "PARKED"  # exactly ONE waiter per Context
    free = self._has_free()
    if free is not None:
      self.instances[free] = ctx
      st.region_id = self._next_region
      self._next_region += 1
      st.granted = True
      st.phase = "ACTIVE"
      self.pmu["grant"] += 1
      self.log.append(("grant", ctx, f"instance{free}"))
      return "GRANTED"
    st.waiter = True
    st.phase = "PARKED"
    self.waiters.append(ctx)
    self.pmu["park"] += 1
    self.log.append(("park", ctx, "waiter"))
    return "PARKED"

  # ---- E2: consume ------------------------------------------------------------------------
  def accepted(self, ctx: int) -> Pending:
    """E2. The target is accepted by the front end; the pending declaration is consumed once."""
    st = self._ctx(ctx)
    if st.pending is None:
      raise self._fault(ctx, "CONSUME_WITHOUT_PENDING", "accepted with nothing ARMED")
    if st.waiter and not st.granted:
      raise self._fault(ctx, "ACCEPTED_WHILE_PARKED", "a parked target must stay at the head")
    p = st.pending
    if p.kind == "REGION_START":
      if not st.granted:
        raise self._fault(ctx, "ACCEPTED_WITHOUT_GRANT", "START accepted before E1 granted a lease")
      st.granted = False
      self.queue.append((ctx, st.region_id or 0))
      st.accepted_unsettled += 1
    elif p.kind in ("CHAIN_FIRST", "CHAIN_CONTINUE", "CHAIN_FINAL"):
      if st.phase != "ACTIVE":
        raise self._fault(ctx, "TOUCH_WITHOUT_BINDING", f"{p.kind} outside an ACTIVE region")
      self._chain_step(ctx, st, p)
      st.accepted_unsettled += 1
    elif p.kind == "REGION_END":
      if st.phase != "ACTIVE":
        raise self._fault(ctx, "TOUCH_WITHOUT_BINDING", "END without an ACTIVE region")
      open_chains = [c for c, s in st.chains.items() if s == "OPEN"]
      if open_chains:
        raise self._fault(ctx, "END_WITH_OPEN_CHAIN", f"chains {open_chains} have no FINAL")
      st.end_declared = True
      st.phase = "END_DRAIN"
    st.pending = None  # consumed exactly once
    self.log.append(("consume", ctx, p.kind))
    return p

  def _chain_step(self, ctx: int, st: ContextState, p: Pending) -> None:
    cid = p.chain_id if p.chain_id is not None else 0
    cur = st.chains.get(cid)
    if p.kind == "CHAIN_FIRST":
      if cur is not None:
        raise self._fault(ctx, "DUPLICATE_FIRST", f"chain {cid} already {cur}")
      st.chains[cid] = "OPEN"
    elif p.kind == "CHAIN_CONTINUE":
      if cur != "OPEN":
        raise self._fault(
          ctx, "CONTINUE_WITHOUT_FIRST" if cur is None else "ACCUMULATE_AFTER_FINAL", f"chain {cid} {cur}"
        )
    else:  # CHAIN_FINAL
      if cur is None:
        raise self._fault(ctx, "FINAL_WITHOUT_FIRST", f"chain {cid}")
      if cur != "OPEN":
        raise self._fault(ctx, "ACCUMULATE_AFTER_FINAL", f"chain {cid} already {cur}")
      st.chains[cid] = "FINAL_SUBMITTED"

  def touch_tile(self, ctx: int) -> None:
    """A tile-touching instruction (rule F1/F3): needs an ACTIVE, not-yet-ended region."""
    st = self._ctx(ctx)
    if st.phase == "END_DRAIN" or st.end_declared or st.phase == "REGION_DONE":
      raise self._fault(ctx, "TOUCH_AFTER_END", f"phase {st.phase}")
    if st.phase != "ACTIVE":
      raise self._fault(ctx, "TOUCH_WITHOUT_BINDING", f"phase {st.phase}")

  # ---- back-end completion ---------------------------------------------------------------
  def settle(self, ctx: int, chain_id: int | None = None) -> None:
    """A previously accepted operation completed for real (not merely accepted)."""
    st = self._ctx(ctx)
    if st.accepted_unsettled <= 0:
      raise self._fault(ctx, "SETTLE_WITHOUT_ACCEPT", "nothing outstanding")
    st.accepted_unsettled -= 1
    st.settled_obligations += 1
    if chain_id is not None and st.chains.get(chain_id) == "FINAL_SUBMITTED":
      st.chains[chain_id] = "FINAL_DONE"
    # an op left the shared queue
    for i, (c, _) in enumerate(self.queue):
      if c == ctx:
        del self.queue[i]
        break

  def state_free(self, ctx: int) -> None:
    """Final output visible and old accesses done: state may be recycled. Does NOT hand over."""
    st = self._ctx(ctx)
    if st.phase != "END_DRAIN":
      raise self._fault(ctx, "STATE_FREE_BEFORE_END", f"phase {st.phase}")
    if st.accepted_unsettled:
      raise self._fault(ctx, "STATE_FREE_WITH_OUTSTANDING", f"{st.accepted_unsettled} unsettled")
    self.pmu["state_free"] += 1
    self.log.append(("state_free", ctx, ""))  # the lease is still held: no wake here

  def region_done(self, ctx: int) -> list[int]:
    """The only success terminal. Only now may a waiter be woken and granted."""
    st = self._ctx(ctx)
    if st.phase != "END_DRAIN":
      raise self._fault(ctx, "REGION_DONE_BEFORE_END", f"phase {st.phase}")
    if st.accepted_unsettled:
      raise self._fault(ctx, "REGION_DONE_WITH_OUTSTANDING", f"{st.accepted_unsettled} unsettled")
    if not any(e[0] == "state_free" and e[1] == ctx for e in self.log):
      raise self._fault(ctx, "REGION_DONE_BEFORE_STATE_FREE", "order is state_free then region_done")
    for i, owner in enumerate(self.instances):
      if owner == ctx:
        self.instances[i] = None
    st.phase = "REGION_DONE"
    st.region_id = None
    st.chains.clear()
    st.end_declared = False
    self.pmu["region_done"] += 1
    self.log.append(("region_done", ctx, ""))
    return self._wake()

  def _wake(self) -> list[int]:
    woken: list[int] = []
    while self.waiters:
      free = self._has_free()
      if free is None:
        break
      ctx = self.waiters.pop(0)
      st = self._ctx(ctx)
      self.instances[free] = ctx
      st.waiter = False
      st.granted = True
      st.phase = "ACTIVE"
      st.region_id = self._next_region
      self._next_region += 1
      self.pmu["wake"] += 1
      self.pmu["grant"] += 1
      self.log.append(("wake", ctx, f"instance{free}"))
      woken.append(ctx)
    return woken

  # ---- cancel / redirect / task exit -------------------------------------------------------
  def redirect(self, ctx: int) -> None:
    """A taken branch / flush between the declaration and its target (rule A4/F5)."""
    st = self._ctx(ctx)
    if st.pending is not None:
      st.pending = None  # cleared; no silent rebind to a later instruction
      self.log.append(("pending_cleared", ctx, "redirect"))
    if st.waiter:
      st.waiter = False
      self.waiters = [c for c in self.waiters if c != ctx]
      st.phase = "IDLE"
    if st.granted:
      # grant already taken, target not accepted: the resource is SETTLED, not assumed free of effect
      st.granted = False
      for i, owner in enumerate(self.instances):
        if owner == ctx:
          self.instances[i] = None
      st.phase = "IDLE"
      st.region_id = None
      self.log.append(("grant_settled", ctx, "redirect"))
      self._wake()

  def cancel(self, ctx: int) -> None:
    """Cancel/fault: stop new ops, keep owner+generation, never publish a success region_done."""
    st = self._ctx(ctx)
    st.pending = None
    if st.waiter:
      st.waiter = False
      self.waiters = [c for c in self.waiters if c != ctx]
    st.granted = False
    st.phase = "FAILED"
    self.log.append(("cancel", ctx, ""))

  def task_exit(self, ctx: int) -> None:
    """A Task leaving with an open region is a protocol fault (F4)."""
    st = self._ctx(ctx)
    if st.phase in ("ACTIVE", "END_DRAIN") and not st.end_declared:
      raise self._fault(ctx, "EXIT_WITHOUT_END", f"phase {st.phase}")
    if st.phase in ("ACTIVE", "END_DRAIN") and st.end_declared:
      raise self._fault(ctx, "EXIT_BEFORE_REGION_DONE", "END declared but region_done not reached")
    st.generation += 1  # a new Task identity: nothing survives
    st.pending = None
    st.phase = "IDLE"


# ----------------------------------------------------------------------------------------------
# Mode / transition tables (per sealed kernel, rule MAT-10)
# ----------------------------------------------------------------------------------------------
def mtype_value(tm: int, tk: int, mtwiden: int) -> int:
  """Upstream encoding, read from vme_test_utils.h MtypeValue: tm[23:10] | tk[7:5] | mtwiden[1:0]."""
  return ((tm & 0x3FFF) << 10) | ((tk & 0x7) << 5) | (mtwiden & 0x3)


class ModeTableError(Exception):
  def __init__(self, code: str, detail: str) -> None:
    super().__init__(f"{code}: {detail}")
    self.code = code


def _val(node: Any) -> Any:
  """A limit is either a bare value or a sourced mapping {value|values|lmul, source}."""
  if isinstance(node, dict):
    for key in ("value", "values", "lmul"):
      if key in node:
        return node[key]
  return node


def validate_mode_table(table: dict[str, Any], abi: dict[str, Any]) -> list[str]:
  """Static (build/load) checks of one kernel's table. Returns coded errors; never raises."""
  errors: list[str] = []
  limits = abi["mode_table_schema"]["limits"]
  tk_max = {k: _val(v) for k, v in limits["tk_max_by_mtwiden"].items()}
  tm_max = _val(limits["tm_max"])
  mtwiden_ok = _val(limits["mtwiden_values"])
  disabled = _val(limits["disabled_mtwiden"])
  lmul_by_sew = {k: _val(v) for k, v in limits["vtype_lmul_by_sew"].items()}
  modes = table.get("modes") or []
  names = [m.get("name") for m in modes]
  if len(names) != len(set(names)):
    errors.append("MODE_DUPLICATE: mode names must be unique")
  enabled = {m["name"] for m in modes if m.get("enabled", True)}
  for m in modes:
    where = f"mode {m.get('name')}"
    for key in ("mtype", "vtype", "mtwiden", "tk", "tm"):
      if key not in m:
        errors.append(f"{where}: MODE_FIELD_MISSING {key}")
    if any(k not in m for k in ("mtype", "mtwiden", "tk", "tm")):
      continue
    if m["mtype"] != mtype_value(m["tm"], m["tk"], m["mtwiden"]):
      want = mtype_value(m["tm"], m["tk"], m["mtwiden"])
      errors.append(
        f"{where}: MTYPE_MISMATCH {m['mtype']} != MtypeValue(tm={m['tm']}, tk={m['tk']}, "
        f"mtwiden={m['mtwiden']}) = {want}"
      )
    if m["mtwiden"] not in mtwiden_ok:
      errors.append(f"{where}: MTWIDEN_OUT_OF_BOUNDS {m['mtwiden']}")
    kmax = tk_max.get(str(m["mtwiden"]))
    if kmax is not None and m["tk"] > kmax:
      errors.append(f"{where}: TK_ABOVE_CLAMP {m['tk']} > {kmax} (the front end clamps it silently)")
    if m["tm"] > tm_max:
      errors.append(f"{where}: TM_ABOVE_CLAMP {m['tm']} > {tm_max}")
    if m.get("enabled", True) and m["mtwiden"] in disabled:
      errors.append(
        f"{where}: DISABLED_MTWIDEN_ENABLED mtwiden {m['mtwiden']} is not allowed on the tensor path"
      )
    if "vtype" in m and m["vtype"] >> 9:
      errors.append(f"{where}: VTYPE_RESERVED_BITS {m['vtype']:#x} sets bits above altfmt[8]")
    if m.get("enabled", True) and m["mtwiden"] != 0:
      derived = lmul_by_sew.get(str(m["vtype"] >> 3 & 7))
      if derived is None or (m["vtype"] & 7) != derived:
        errors.append(f"{where}: VTYPE_LMUL_NOT_DERIVED vtype {m['vtype']:#x}")
  trans = table.get("transitions") or []
  for t in trans:
    for end in ("from", "to"):
      s = t.get(end)
      if s not in enabled and s not in ("NONE", "CLEARED"):
        errors.append(f"transition {t.get('from')}->{t.get('to')}: TRANSITION_UNKNOWN_MODE {s!r}")
  seen = {(t.get("from"), t.get("to")) for t in trans}
  if len(seen) != len(trans):
    errors.append("TRANSITION_DUPLICATE")
  if "CLEARED" in enabled:
    errors.append("CLEARED_IS_NOT_A_MODE: it is the state after an inter-block vset*, not a compute mode")
  for t in trans:
    if t.get("from") == "CLEARED" and t.get("restore") is not True:
      errors.append(
        f"transition CLEARED->{t.get('to')}: RESTORE_PREAMBLE_NOT_REQUIRED (must carry restore: true)"
      )
  reach, frontier = {"NONE"}, ["NONE"]
  while frontier:
    cur = frontier.pop()
    for t in trans:
      if t.get("from") == cur and t.get("to") not in reach:
        reach.add(t["to"])
        frontier.append(t["to"])
  for name in sorted(enabled - reach):
    errors.append(f"mode {name}: UNREACHABLE from NONE")
  return errors


def access_allowed(table: dict[str, Any], current: str, requested: str, *, inside_asm_block: bool) -> None:
  """Runtime check of one configuration change against the SEALED table. Raises ModeTableError."""
  if inside_asm_block:
    raise ModeTableError("VSET_INSIDE_ASM_BLOCK", "a vset* inside an asm block is never legal")
  enabled = {m["name"] for m in table["modes"] if m.get("enabled", True)}
  disabled = {m["name"] for m in table["modes"] if not m.get("enabled", True)}
  if requested in disabled:
    raise ModeTableError("DISABLED_MODE", requested)
  # NONE and CLEARED are STATES (nothing configured / cleared by an inter-block vset*), not compute
  # modes, so they are legal requested targets only through an explicit transition row.
  if requested not in enabled and requested not in ("NONE", "CLEARED"):
    raise ModeTableError("MODE_NOT_IN_TABLE", requested)
  for t in table["transitions"]:
    if t["from"] == current and t["to"] == requested:
      return
  if current == "CLEARED":
    raise ModeTableError(
      "MISSING_RESTORE_PREAMBLE", f"tile use at CLEARED needs an mset* restore to {requested}"
    )
  raise ModeTableError("TRANSITION_NOT_IN_TABLE", f"{current} -> {requested}")
