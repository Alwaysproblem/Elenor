"""Validate the C012 VME declaration ABI and kernel mode tables.

What this checks, none of it a string search:

1. Every `source: {file, line, needle}` in the ABI and in the mode tables is READ BACK from the
   pinned tree (one whole-tree walk, so a nested citation cannot escape).
2. The ABI carries the REQUIRED contract held in THIS file (events, ordering, fault set, cleanup
   paths). A synchronized edit that deletes a fault together with the check that would notice it
   cannot pass, because the required set is not in the JSON it constrains.
3. Mode tables are checked by `vme_decl_ref.validate_mode_table` (encoding recomputed, clamps
   respected, transitions closed, CLEARED restore, reachability) and by running access sequences.
4. A battery of declaration scenarios is executed on the reference model; each must end in the
   outcome the ABI names (E1 with no grant leaves the queue empty, a stalled pending survives a
   grant, one E2 consumes once, a repeated consume is refused, ...).
5. The field request names only things that exist in the ABI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB))

import vme_decl_ref as ref  # noqa: E402

H = LIB.parents[1]
UPSTREAM = H / "upstream" / "coralnpu"
HDL = UPSTREAM / "hdl"
PROTOCOL = H / "protocol"
ABI = PROTOCOL / "w3_vme_decl_abi_v1.json"
TABLES = PROTOCOL / "w3_vme_mode_tables_v1.json"
REQUEST = PROTOCOL / "field_requests" / "w3.yaml"
PROFILE = PROTOCOL / "w3_numeric_profile_v1.yaml"
CSR_NAMES = PROTOCOL / "csr_name_map.yaml"
CATALOG = PROTOCOL / "cn_protocol_v1.yaml"

# ----------------------------------------------------------------------------------------------
# The required contract. Held here so one edit cannot weaken the ABI and its guard together.
# ----------------------------------------------------------------------------------------------
REQUIRED_KINDS = ("REGION_START", "REGION_END", "CHAIN_FIRST", "CHAIN_CONTINUE", "CHAIN_FINAL")
REQUIRED_EVENTS = ("E1", "E2")
REQUIRED_ORDERING = ("state_free", "region_done", "wake")
REQUIRED_PROPERTIES = ("U1", "U2", "U3", "U4", "U5", "U6")
REQUIRED_FAULT_GROUPS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7")
REQUIRED_CLEANUP = (
  "grant_then_redirect",
  "waiter_then_redirect",
  "armed_then_redirect",
  "accepted_then_cancel",
)
# fault code -> group. Removing one from the ABI, or moving it, is rejected.
REQUIRED_FAULTS: dict[str, str] = {
  "ARMED_CONFLICT": "F2",
  "NESTED_START": "F2",
  "UNKNOWN_KIND": "F6",
  "TOUCH_WITHOUT_BINDING": "F1",
  "TOUCH_AFTER_END": "F3",
  "EXIT_WITHOUT_END": "F4",
  "EXIT_BEFORE_REGION_DONE": "F4",
  "STALE_GENERATION": "F5",
  "CONSUME_WITHOUT_PENDING": "F6",
  "ACCEPTED_WHILE_PARKED": "F5",
  "ACCEPTED_WITHOUT_GRANT": "F5",
  "DUPLICATE_FIRST": "F7",
  "CONTINUE_WITHOUT_FIRST": "F7",
  "FINAL_WITHOUT_FIRST": "F7",
  "ACCUMULATE_AFTER_FINAL": "F7",
  "END_WITH_OPEN_CHAIN": "F7",
  "SETTLE_WITHOUT_ACCEPT": "F5",
  "STATE_FREE_BEFORE_END": "F3",
  "STATE_FREE_WITH_OUTSTANDING": "F3",
  "REGION_DONE_BEFORE_END": "F3",
  "REGION_DONE_WITH_OUTSTANDING": "F3",
  "REGION_DONE_BEFORE_STATE_FREE": "F3",
  "CTX_OUT_OF_RANGE": "F6",
  "FAULT_CODE_UNKNOWN": "F6",
}
# Hardware limits the mode-table checker relies on. Held here so widening one in the JSON (which would
# make a lying table pass) is rejected; each is also read back from RvvFrontEnd.sv through its source.
REQUIRED_LIMITS: dict[str, Any] = {
  "tk_max_by_mtwiden": {"1": 1, "2": 2, "3": 4},
  "tm_max": 16,
  "mtwiden_values": [0, 1, 2, 3],
  "disabled_mtwiden": [3],
  "vtype_lmul_by_sew": {"0": 0, "1": 1, "2": 2},
}
# The design-text corrections this card made; deleting one would silently revert to the wrong text.
REQUIRED_CORRECTIONS = ("VTYPE-01", "ANCHOR-01", "ANCHOR-02", "TK-CLAMP-01", "ENC-01", "LIMITS-01")
# The kernel tables the contract must carry and the checks each must keep passing.
REQUIRED_TABLES = ("reference", "tail")
REQUIRED_REFERENCE_MODES = {"FP32-MAC": True, "BF16-MAC-TK1": True, "BF16-MAC-TK2": True, "INT8": False}
# Numeric facts of the reference template, recomputed from MtypeValue rather than trusted.
REFERENCE_MTYPES = {
  "FP32-MAC": (16, 1, 1),
  "BF16-MAC-TK1": (16, 1, 2),
  "BF16-MAC-TK2": (16, 2, 2),
  "INT8": (16, 1, 3),
}
# Faults the whole ABI must keep marked statically rejectable (build/load) vs runtime only.
STATIC_FAULTS = {
  "UNKNOWN_KIND",
  "TOUCH_WITHOUT_BINDING",
  "TOUCH_AFTER_END",
  "DUPLICATE_FIRST",
  "CONTINUE_WITHOUT_FIRST",
  "FINAL_WITHOUT_FIRST",
  "ACCUMULATE_AFTER_FINAL",
  "END_WITH_OPEN_CHAIN",
  "CTX_OUT_OF_RANGE",
  "FAULT_CODE_UNKNOWN",
}
# Required W3 CSR logical names (csr_name_map.yaml owns them; this file only checks they exist).
REQUIRED_CSR_NAMES = {
  "nexus.mregion.start",
  "nexus.mregion.end",
  "nexus.mregion.first",
  "nexus.mregion.continue",
  "nexus.mregion.final",
}


def load(path: Path) -> Any:
  text = path.read_text()
  return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


# ----------------------------------------------------------------------------------------------
# source read-back, whole tree
# ----------------------------------------------------------------------------------------------
def _read_line(file: str, line: int) -> str | None:
  for root in (HDL, UPSTREAM, UPSTREAM / "hdl" / "verilog" / "rvv" / "design" / ".."):
    path = (root / file).resolve()
    if path.is_file():
      lines = path.read_text().splitlines()
      return lines[line - 1] if 0 < line <= len(lines) else None
  return None


def iter_sources(node: Any, where: str = "") -> list[tuple[str, dict]]:
  found: list[tuple[str, dict]] = []
  if isinstance(node, dict):
    for key, value in node.items():
      if key in ("source", "limits_source") and isinstance(value, dict) and "line" in value:
        found.append((f"{where}.{key}", value))
      else:
        found.extend(iter_sources(value, f"{where}.{key}"))
  elif isinstance(node, list):
    for index, value in enumerate(node):
      found.extend(iter_sources(value, f"{where}[{index}]"))
  return found


def check_sources(document: Any, label: str, errors: list[str]) -> int:
  count = 0
  for where, src in iter_sources(document):
    count += 1
    if not {"file", "line", "needle"} <= set(src):
      errors.append(f"{label}{where}: MISSING_SOURCE (need file/line/needle)")
      continue
    text = _read_line(str(src["file"]), int(src["line"]))
    if text is None:
      errors.append(f"{label}{where}: SOURCE_FILE_MISSING {src['file']}:{src['line']}")
    elif str(src["needle"]) not in text:
      errors.append(
        f"{label}{where}: SOURCE_LINE_MISMATCH {src['file']}:{src['line']} lacks {src['needle']!r}"
      )
  return count


# ----------------------------------------------------------------------------------------------
# ABI
# ----------------------------------------------------------------------------------------------
def check_abi(abi: dict[str, Any], errors: list[str]) -> None:
  if abi.get("schema") != "cn_vme_decl_abi/v1" or abi.get("status") != "draft":
    errors.append("abi: schema/status must be cn_vme_decl_abi/v1 + draft (C020 seals, not this card)")
  carrier = abi.get("carrier") or {}
  if carrier.get("bit_positions") != "unresolved":
    errors.append("abi.carrier.bit_positions: must stay 'unresolved' until the W1 allocator assigns them")
  if carrier.get("width_bits") != "unresolved":
    errors.append("abi.carrier.width_bits: must stay 'unresolved' (the plan's 32 is not a sourced bound)")
  if carrier.get("access") != "unspecified":
    errors.append("abi.carrier.access: must stay 'unspecified'; the access level is not frozen here")
  for key in ("bit_positions_owner", "width_reason"):
    if not str(carrier.get(key) or "").strip():
      errors.append(f"abi.carrier.{key}: an unresolved field needs an owner and a reason")
  for forbidden in ("address", "opcode", "bits", "csr_address", "encoding"):
    if forbidden in carrier:
      errors.append(f"abi.carrier.{forbidden}: a number or encoding was invented here (INVENTED_VALUE)")
  names = set(carrier.get("csr_name_map_entries") or [])
  if names != REQUIRED_CSR_NAMES:
    errors.append(f"abi.carrier.csr_name_map_entries: must be exactly {sorted(REQUIRED_CSR_NAMES)}")
  if tuple((abi.get("declaration") or {}).get("kinds") or ()) != REQUIRED_KINDS:
    errors.append(f"abi.declaration.kinds: must be exactly {list(REQUIRED_KINDS)}")
  if (abi.get("declaration") or {}).get("one_pending_per_context") is not True:
    errors.append("abi.declaration.one_pending_per_context: must stay true")
  events = abi.get("events") or {}
  for name in REQUIRED_EVENTS:
    if name not in events:
      errors.append(f"abi.events.{name}: required event is missing")
  e1, e2 = events.get("E1") or {}, events.get("E2") or {}
  if e1.get("requires_accepted") is not False:
    errors.append(
      "abi.events.E1.requires_accepted: the waiter must NOT depend on the target being accepted"
    )
  if e1.get("registers_waiter") is not True:
    errors.append("abi.events.E1.registers_waiter: E1 must register the waiter")
  if e1.get("may_consume_pending") is not False or e1.get("may_read_pending") is not True:
    errors.append("abi.events.E1: E1 may READ pending but never CONSUME it")
  if e2.get("consume_count") != 1 or e2.get("unique_consume_point") is not True:
    errors.append("abi.events.E2: must be the single, unique consume point")
  if tuple(abi.get("ordering") or ()) != REQUIRED_ORDERING:
    errors.append(f"abi.ordering: must be exactly {list(REQUIRED_ORDERING)}")
  props = set(abi.get("properties") or {})
  for u in REQUIRED_PROPERTIES:
    if u not in props:
      errors.append(f"abi.properties.{u}: required property is missing")
  groups = set(abi.get("fault_groups") or {})
  for g in REQUIRED_FAULT_GROUPS:
    if g not in groups:
      errors.append(f"abi.fault_groups.{g}: required group is missing")
  faults = {f.get("code"): f for f in abi.get("faults") or []}
  if len(faults) != len(abi.get("faults") or []):
    errors.append("abi.faults: duplicate fault code")
  for code, group in REQUIRED_FAULTS.items():
    f = faults.get(code)
    if f is None:
      errors.append(f"abi.faults: required fault {code} is missing")
      continue
    if f.get("group") != group:
      errors.append(f"abi.faults.{code}: group {f.get('group')!r} must stay {group!r}")
    if (code in STATIC_FAULTS) != bool(f.get("static")):
      errors.append(f"abi.faults.{code}: static must be {code in STATIC_FAULTS}")
    if not str(f.get("text") or "").strip():
      errors.append(f"abi.faults.{code}: needs a description")
  for code in sorted(set(faults) - set(REQUIRED_FAULTS)):
    errors.append(f"abi.faults.{code}: not in the validator's required set; add it there deliberately")
  cleanup = abi.get("cleanup") or {}
  for key in REQUIRED_CLEANUP:
    if not str(cleanup.get(key) or "").strip():
      errors.append(f"abi.cleanup.{key}: required cleanup path is missing")
  if cleanup.get("final_does_not_drain") is not True or cleanup.get("end_does_not_release") is not True:
    errors.append("abi.cleanup: FINAL must not drain and END must not release")
  if "SETTLED" not in str(cleanup.get("grant_then_redirect", "")):
    errors.append(
      "abi.cleanup.grant_then_redirect: a taken grant must be SETTLED, never assumed effect-free"
    )
  if "never publishes" not in str(cleanup.get("accepted_then_cancel", "")):
    errors.append("abi.cleanup.accepted_then_cancel: must state a success region_done is never published")


def _limit_value(node: Any) -> Any:
  return ref._val(node)


def check_limits(abi: dict[str, Any], errors: list[str]) -> None:
  schema = abi.get("mode_table_schema") or {}
  limits = schema.get("limits") or {}
  enc = schema.get("encoding") or {}
  if "source" not in enc or "formula" not in enc:
    errors.append("abi.mode_table_schema.encoding: needs formula and a read-back source")
  for name, want in REQUIRED_LIMITS.items():
    node = limits.get(name)
    if node is None:
      errors.append(f"abi.mode_table_schema.limits.{name}: required limit is missing")
      continue
    if name == "tk_max_by_mtwiden":
      got: Any = {k: _limit_value(v) for k, v in node.items()}
      for key, entry in node.items():
        if not isinstance(entry, dict) or "source" not in entry:
          errors.append(f"abi.mode_table_schema.limits.{name}.{key}: a hardware clamp needs its own source")
    elif name == "vtype_lmul_by_sew":
      got = {k: _limit_value(v) for k, v in node.items()}
      for key, entry in node.items():
        if not isinstance(entry, dict) or "source" not in entry:
          errors.append(f"abi.mode_table_schema.limits.{name}.{key}: LMUL derivation needs its own source")
    else:
      got = _limit_value(node)
      if name != "disabled_mtwiden" and not (isinstance(node, dict) and "source" in node):
        errors.append(f"abi.mode_table_schema.limits.{name}: a hardware limit needs its own source")
    if got != want:
      errors.append(f"abi.mode_table_schema.limits.{name}: {got!r} != required {want!r} (hardware fact)")
  disabled = limits.get("disabled_mtwiden")
  if not (isinstance(disabled, dict) and str(disabled.get("decided_by") or "").strip()):
    errors.append(
      "abi.mode_table_schema.limits.disabled_mtwiden: a decision (not a hardware fact) needs decided_by"
    )
  have = {c.get("id") for c in abi.get("corrections_to_design_text") or []}
  for cid in REQUIRED_CORRECTIONS:
    if cid not in have:
      errors.append(f"abi.corrections_to_design_text: correction {cid} was removed")
  if not abi.get("mode_table_schema"):
    errors.append("abi.mode_table_schema: required section is missing")


def check_profile_link(abi: dict[str, Any], tables: dict[str, Any], errors: list[str]) -> None:
  if tables.get("abi") != ABI.name:
    errors.append(f"tables.abi: must name {ABI.name}")
  if tables.get("numeric_profile") != PROFILE.name:
    errors.append(f"tables.numeric_profile: must name {PROFILE.name}")
  profile = load(PROFILE)
  rules = {r["id"] for r in profile["rules"]}
  if "FRM-01" not in rules or "NAR-01" not in rules:
    errors.append("profile: C011 rules the mode tables extend are missing")


# ----------------------------------------------------------------------------------------------
# mode tables
# ----------------------------------------------------------------------------------------------
def check_tables(abi: dict[str, Any], tables: dict[str, Any], errors: list[str]) -> None:
  for name in REQUIRED_TABLES:
    table = tables.get(name)
    if table is None:
      errors.append(f"tables.{name}: required kernel table is missing")
      continue
    errors.extend(f"tables.{name}: {e}" for e in ref.validate_mode_table(table, abi))
  reference = tables.get("reference") or {}
  modes = {m.get("name"): m for m in reference.get("modes") or []}
  for name, enabled in REQUIRED_REFERENCE_MODES.items():
    m = modes.get(name)
    if m is None:
      errors.append(f"tables.reference: required mode {name} is missing")
      continue
    if bool(m.get("enabled", True)) is not enabled:
      errors.append(f"tables.reference.{name}: enabled must be {enabled}")
    tm, tk, w = REFERENCE_MTYPES[name]
    if m.get("mtype") != ref.mtype_value(tm, tk, w):
      errors.append(f"tables.reference.{name}: mtype {m.get('mtype')} != MtypeValue{(tm, tk, w)}")
  tail, base = tables.get("tail") or {}, reference
  tail_tm = {m["name"]: m["tm"] for m in tail.get("modes") or []}
  base_tm = {m["name"]: m["tm"] for m in base.get("modes") or []}
  if not any(tail_tm.get(n) != base_tm.get(n) for n in tail_tm if n in base_tm):
    errors.append("tables.tail: must differ from the reference (a different tm), or it proves nothing")
  if tail.get("scope") == base.get("scope"):
    errors.append("tables.tail.scope: must be its own table, not a copy of the reference scope")
  # the reference note must keep saying it is not a global whitelist
  if "not a global whitelist" not in str(reference.get("scope", "")):
    errors.append("tables.reference.scope: must state it is a reference template, not a global whitelist")


# ----------------------------------------------------------------------------------------------
# scenarios on the reference model
# ----------------------------------------------------------------------------------------------
def _fresh(abi: dict[str, Any], contexts: int = 2, instances: int = 1) -> ref.DeclModel:
  return ref.DeclModel(abi, contexts, instances)


def _expect(code: str, fn: Any, *args: Any, **kw: Any) -> str | None:
  try:
    fn(*args, **kw)
  except ref.DeclError as exc:
    return None if exc.code == code else f"raised {exc.code}, expected {code}"
  return f"no fault, expected {code}"


def run_scenarios(abi: dict[str, Any]) -> list[str]:
  """Execute the card's pass-condition scenarios. Returns a list of failures (empty = all pass)."""
  bad: list[str] = []

  def need(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
      bad.append(f"scenario {name}: {detail or 'failed'}")

  # E1 with no free instance: the Context parks, the shared queue stays EMPTY, one waiter.
  m = _fresh(abi)
  m.csrw(0, "REGION_START")
  need("e1_free_grant", m.target_at_head(0) == "GRANTED")
  m.csrw(1, "REGION_START")
  need("e1_park", m.target_at_head(1) == "PARKED")
  need("e1_park_queue_empty", m.queue == [], f"queue {m.queue}")
  need("e1_one_waiter", m.waiters == [1], f"waiters {m.waiters}")
  need(
    "e1_repeat_no_second_waiter",
    m.target_at_head(1) == "PARKED" and m.waiters == [1],
    f"waiters {m.waiters}",
  )
  need("e1_no_consume", m.contexts[1].pending is not None, "E1 consumed pending")
  # the waiter exists although the target was never accepted
  need("waiter_without_accept", m.contexts[1].waiter and m.contexts[1].accepted_unsettled == 0)
  err = _expect("ACCEPTED_WHILE_PARKED", m.accepted, 1)
  need("parked_target_cannot_be_accepted", err is None, err or "")

  # E2: after the grant, a stalled pending survives; one accept consumes once; a second accept faults.
  m = _fresh(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  need("stall_keeps_pending", m.contexts[0].pending is not None, "pending dropped before E2")
  need("stall_grant_survives", m.contexts[0].granted)
  consumed = m.accepted(0)
  need("e2_consumes", consumed.kind == "REGION_START" and m.contexts[0].pending is None)
  err = _expect("CONSUME_WITHOUT_PENDING", m.accepted, 0)
  need("second_consume_refused", err is None, err or "")

  # the card's ordering: state_free then region_done, and ONLY region_done wakes the waiter.
  # A protocol fault is fail-stop (the Context ends FAILED), so each premature call gets a fresh model.
  def ended_region() -> ref.DeclModel:
    mm = _fresh(abi)
    mm.csrw(0, "REGION_START")
    mm.target_at_head(0)
    mm.accepted(0)
    mm.csrw(0, "CHAIN_FIRST", 0)
    mm.accepted(0)
    mm.csrw(0, "CHAIN_FINAL", 0)
    mm.accepted(0)
    mm.csrw(1, "REGION_START")
    mm.target_at_head(1)
    mm.csrw(0, "REGION_END")
    mm.accepted(0)
    return mm

  m = ended_region()
  err = _expect("STATE_FREE_WITH_OUTSTANDING", m.state_free, 0)
  need("state_free_needs_settled", err is None, err or "")
  need("premature_state_free_is_fail_stop", m.contexts[0].phase == "FAILED")
  m = ended_region()
  err = _expect("REGION_DONE_WITH_OUTSTANDING", m.region_done, 0)
  need("region_done_needs_settled", err is None, err or "")
  m = ended_region()
  for _ in range(3):
    m.settle(0, 0)
  need("final_done", m.contexts[0].chains[0] == "FINAL_DONE")
  err = _expect("REGION_DONE_BEFORE_STATE_FREE", m.region_done, 0)
  need("region_done_after_state_free", err is None, err or "")
  m = ended_region()
  for _ in range(3):
    m.settle(0, 0)
  m.state_free(0)
  need("state_free_does_not_wake", m.waiters == [1] and m.pmu["wake"] == 0, f"waiters {m.waiters}")
  need("lease_held_after_state_free", m.instances == [0], f"instances {m.instances}")
  woken = m.region_done(0)
  need("region_done_wakes_waiter", woken == [1] and m.pmu["wake"] == 1, f"woken {woken}")
  need("woken_context_active", m.contexts[1].phase == "ACTIVE" and m.contexts[1].granted)
  need("lease_handed_over", m.instances == [1], f"instances {m.instances}")
  m = _fresh(abi)
  need("state_free_before_end", _expect("STATE_FREE_BEFORE_END", m.state_free, 0) is None)
  m = _fresh(abi)
  need("region_done_before_end", _expect("REGION_DONE_BEFORE_END", m.region_done, 0) is None)

  # END with an open chain, FINAL then accumulate, FIRST twice, CONTINUE without FIRST.
  for name, code, steps in (
    ("end_open_chain", "END_WITH_OPEN_CHAIN", ["F0", "END"]),
    ("accumulate_after_final", "ACCUMULATE_AFTER_FINAL", ["F0", "FIN", "CONT"]),
    ("duplicate_first", "DUPLICATE_FIRST", ["F0", "F0"]),
    ("continue_without_first", "CONTINUE_WITHOUT_FIRST", ["CONT"]),
    ("final_without_first", "FINAL_WITHOUT_FIRST", ["FIN"]),
  ):
    m = _fresh(abi)
    m.csrw(0, "REGION_START")
    m.target_at_head(0)
    m.accepted(0)
    kind = {"F0": "CHAIN_FIRST", "FIN": "CHAIN_FINAL", "CONT": "CHAIN_CONTINUE", "END": "REGION_END"}
    err = None
    for step in steps:
      m.csrw(0, kind[step], 0)
      try:
        m.accepted(0)
      except ref.DeclError as exc:
        err = None if exc.code == code else f"raised {exc.code}"
        break
    else:
      err = f"no fault, expected {code}"
    need(name, err is None, err or "")

  # touch without binding / after END / start while holding / redirect clears / exit.
  m = _fresh(abi)
  need("touch_without_binding", _expect("TOUCH_WITHOUT_BINDING", m.touch_tile, 0) is None)
  m = _fresh(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  m.accepted(0)
  m.csrw(0, "REGION_END")
  m.accepted(0)
  need("touch_after_end", _expect("TOUCH_AFTER_END", m.touch_tile, 0) is None)
  m = _fresh(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  m.accepted(0)
  need("start_while_active", _expect("NESTED_START", m.csrw, 0, "REGION_START") is None)
  m = _fresh(abi)
  m.csrw(0, "REGION_START")
  need("armed_conflict", _expect("ARMED_CONFLICT", m.csrw, 0, "REGION_END") is None)
  m = _fresh(abi)
  need("unknown_kind", _expect("UNKNOWN_KIND", m.csrw, 0, "BOGUS") is None)
  m = _fresh(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  m.accepted(0)
  need("exit_without_end", _expect("EXIT_WITHOUT_END", m.task_exit, 0) is None)
  m = _fresh(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  m.accepted(0)
  m.csrw(0, "REGION_END")
  m.accepted(0)
  need("exit_before_region_done", _expect("EXIT_BEFORE_REGION_DONE", m.task_exit, 0) is None)

  # a grant already taken, then redirect: the lease is SETTLED and the waiter is woken (not 'no effect').
  m = _fresh(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  m.csrw(1, "REGION_START")
  m.target_at_head(1)
  m.redirect(0)
  need(
    "grant_settled_on_redirect",
    m.instances[0] == 1 and m.contexts[1].phase == "ACTIVE",
    f"instances {m.instances}",
  )
  need("redirect_clears_pending", m.contexts[0].pending is None)
  # a parked waiter that is redirected leaves the waiter list and stays eligible.
  m = _fresh(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  m.csrw(1, "REGION_START")
  m.target_at_head(1)
  m.redirect(1)
  need("waiter_removed_on_redirect", m.waiters == [] and not m.contexts[1].waiter)
  # a Context generation change clears pending the same beat: nothing survives into the new Task.
  m = _fresh(abi)
  m.csrw(0, "REGION_START")
  m.task_exit(0)
  need("generation_clears_pending", m.contexts[0].pending is None and m.contexts[0].generation == 2)
  m.contexts[0].pending = ref.Pending("REGION_START", None, 1)  # a stale record from the old generation
  need("stale_pending_rejected", _expect("STALE_GENERATION", m.target_at_head, 0) is None)
  # cancel with accepted ops: the region ends FAILED and never publishes a success region_done.
  m = _fresh(abi)
  m.csrw(0, "REGION_START")
  m.target_at_head(0)
  m.accepted(0)
  m.cancel(0)
  need("cancel_fails_region", m.contexts[0].phase == "FAILED" and m.pmu["region_done"] == 0)
  need("cancel_ops_still_settle", m.contexts[0].accepted_unsettled == 1)
  m.settle(0)
  need("settle_after_cancel", m.contexts[0].accepted_unsettled == 0)
  err = _expect("SETTLE_WITHOUT_ACCEPT", m.settle, 0)
  need("settle_without_accept", err is None, err or "")

  # Two instances: two Contexts hold concurrently and a third parks (N-dimension, N=2 is interface only).
  m = _fresh(abi, contexts=3, instances=2)
  for c in (0, 1, 2):
    m.csrw(c, "REGION_START")
  need("n2_two_grants", [m.target_at_head(0), m.target_at_head(1)] == ["GRANTED", "GRANTED"])
  need("n2_third_parks", m.target_at_head(2) == "PARKED")
  return bad


# ----------------------------------------------------------------------------------------------
# mode-table sequences (card step 3 and its pass condition)
# ----------------------------------------------------------------------------------------------
def run_table_sequences(tables: dict[str, Any]) -> list[str]:
  bad: list[str] = []
  ref_t, tail_t = tables["reference"], tables["tail"]

  def run(table: dict[str, Any], seq: list[tuple[str, str, bool]]) -> str | None:
    for cur, req, in_block in seq:
      ref.access_allowed(table, cur, req, inside_asm_block=in_block)
    return None

  def expect_error(
    name: str, table: dict[str, Any], cur: str, req: str, code: str, in_block: bool = False
  ) -> None:
    try:
      ref.access_allowed(table, cur, req, inside_asm_block=in_block)
    except ref.ModeTableError as exc:
      if exc.code != code:
        bad.append(f"sequence {name}: raised {exc.code}, expected {code}")
      return
    bad.append(f"sequence {name}: no error, expected {code}")

  # legal: seed -> BF16 -> drain, and the CLEARED restore path between blocks.
  for name, table, seq in (
    (
      "reference_legal",
      ref_t,
      [
        ("NONE", "FP32-MAC", False),
        ("FP32-MAC", "BF16-MAC-TK2", False),
        ("BF16-MAC-TK2", "FP32-MAC", False),
      ],
    ),
    (
      "reference_cleared_restore",
      ref_t,
      [("BF16-MAC-TK2", "CLEARED", False), ("CLEARED", "BF16-MAC-TK2", False)],
    ),
    ("tail_legal", tail_t, [("NONE", "FP32-MAC", False), ("FP32-MAC", "BF16-MAC-TK1", False)]),
    ("tail_cleared_restore", tail_t, [("BF16-MAC-TK1", "CLEARED", False), ("CLEARED", "FP32-MAC", False)]),
  ):
    try:
      run(table, seq)
    except ref.ModeTableError as exc:
      bad.append(f"sequence {name}: {exc}")
  expect_error(
    "vset_inside_block", ref_t, "FP32-MAC", "BF16-MAC-TK2", "VSET_INSIDE_ASM_BLOCK", in_block=True
  )
  expect_error("int8_disabled", ref_t, "FP32-MAC", "INT8", "DISABLED_MODE")
  expect_error("cleared_to_cleared", ref_t, "CLEARED", "CLEARED", "MISSING_RESTORE_PREAMBLE")
  expect_error("unknown_mode", ref_t, "FP32-MAC", "BF16-MAC-TK9", "MODE_NOT_IN_TABLE")
  expect_error("not_in_table_transition", ref_t, "NONE", "BF16-MAC-TK2", "TRANSITION_NOT_IN_TABLE")
  # the reference-only mode is NOT legal in the tail template: the table is per-kernel, not global.
  expect_error("tk2_not_in_tail", tail_t, "FP32-MAC", "BF16-MAC-TK2", "MODE_NOT_IN_TABLE")
  # CLEARED with no restore row for that mode (tail has none for TK2)
  return bad


# ----------------------------------------------------------------------------------------------
# field request
# ----------------------------------------------------------------------------------------------
def check_field_request(request: dict[str, Any], abi: dict[str, Any], errors: list[str]) -> None:
  if request.get("schema") != "cn_field_request/v1" or request.get("package") != "W3":
    errors.append("field_request: schema/package must be cn_field_request/v1 + W3")
  catalog = {m["name"] for m in load(CATALOG)["messages"]}
  csr_names = {d["name"] for d in load(CSR_NAMES)["declared"]}
  kinds = set(abi["declaration"]["kinds"])
  for need in sorted(REQUIRED_CSR_NAMES - csr_names):
    errors.append(f"csr_name_map: {need} is not declared there")
  req = request.get("csr_requests") or []
  asked = {r.get("name") for r in req}
  if asked != REQUIRED_CSR_NAMES:
    errors.append(f"field_request.csr_requests: must be exactly {sorted(REQUIRED_CSR_NAMES)}")
  for r in req:
    for banned in ("address", "bits", "opcode", "width_bits", "encoding"):
      if banned in r:
        errors.append(
          f"field_request.csr_requests.{r.get('name')}.{banned}: invented value (INVENTED_VALUE)"
        )
    if r.get("kind") not in kinds:
      errors.append(
        f"field_request.csr_requests.{r.get('name')}: kind {r.get('kind')!r} is not an ABI kind"
      )
  for item in request.get("messages") or []:
    if item.get("message") in catalog:
      if item.get("new") is True:
        errors.append(f"field_request.messages.{item.get('message')}: already in the C009 catalog, not new")
      continue
    if item.get("new") is not True:
      errors.append(
        f"field_request.messages.{item.get('message')}: not in the C009 catalog; mark new: true"
      )
  for item in request.get("fields_needed") or []:
    if item.get("key") not in abi and item.get("key") != "faults":
      errors.append(
        f"field_request.fields_needed.{item.get('name')}: key {item.get('key')!r} not in the ABI"
      )
  # facts W3 owns that the ABI promises downstream
  needs = {i.get("name") for i in request.get("fields_needed") or []}
  for need in ("region_id", "instance_id", "region_done_record"):
    if need not in needs:
      errors.append(f"field_request.fields_needed: {need} is missing")


# ----------------------------------------------------------------------------------------------
# whole contract
# ----------------------------------------------------------------------------------------------
def validate(
  abi_path: Path = ABI, tables_path: Path = TABLES, request_path: Path = REQUEST
) -> dict[str, Any]:
  abi, tables, request = load(abi_path), load(tables_path), load(request_path)
  return validate_docs(abi, tables, request)


def validate_docs(abi: Any, tables: Any, request: Any) -> dict[str, Any]:
  errors: list[str] = []
  anchors = 0
  for name, doc in (("abi", abi), ("tables", tables), ("field request", request)):
    if not isinstance(doc, dict):
      errors.append(f"{name}: root must be a mapping, got {type(doc).__name__}")
  if errors:
    return {
      "schema": "cn_vme_decl_report/v1",
      "valid": False,
      "errors": errors,
      "anchors_read_back": 0,
      "faults": 0,
      "tables": [],
      "open": [],
    }

  def guarded(label: str, fn: Any, *args: Any) -> Any:
    """Run one check; a structurally broken document is reported, never raised."""
    try:
      return fn(*args)
    except (
      KeyError,
      TypeError,
      AttributeError,
      IndexError,
      ValueError,
      ref.DeclError,
      ref.ModeTableError,
    ) as exc:
      errors.append(f"{label}: document is malformed ({type(exc).__name__}: {exc})")
      return None

  def sources() -> int:
    local: list[str] = []
    n = check_sources(abi, "abi", local) + check_sources(tables, "tables", local)
    errors.extend(local)
    return n

  anchors = guarded("sources", sources) or 0
  guarded("abi", check_abi, abi, errors)
  guarded("limits", check_limits, abi, errors)
  guarded("profile link", check_profile_link, abi, tables, errors)
  guarded("tables", check_tables, abi, tables, errors)
  guarded("field request", check_field_request, request, abi, errors)
  scenarios = guarded("scenarios", run_scenarios, abi) or []
  sequences = guarded("table sequences", run_table_sequences, tables) or []
  errors.extend(scenarios)
  errors.extend(sequences)
  return {
    "schema": "cn_vme_decl_report/v1",
    "valid": not errors,
    "errors": errors,
    "anchors_read_back": anchors,
    "faults": len(abi.get("faults") or []),
    "tables": sorted(k for k in tables if k in REQUIRED_TABLES),
    "open": ["carrier.bit_positions", "carrier.width_bits", "carrier.access", "END encoding"],
  }


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  report = validate()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / "vme_decl_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  if not report["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
