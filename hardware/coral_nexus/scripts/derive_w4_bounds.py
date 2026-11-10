"""Static-envelope derivation and lifecycle oracle for the C013/W4-01 memory contract.

Three jobs, all driven by the committed YAML (never by hard-coded copies):

1. ``validate_constraints`` / ``derive``: recompute the bank / Frame / MSHR /
   response / window budgets from ``configs/w4_memory_constraints.yaml`` with a
   real unit algebra (B vs KiB vs entries vs cycles are distinct dimensions;
   adding entries to cycles is rejected, sources are read back from the repo).
2. ``validate_contract``: structural checks on ``protocol/w4_memory.contract.yaml``
   (ownership split, five separate accounting ledgers, terminal states,
   hold->bind->retire conservation, profile-gate scope, MSHR token lifecycle,
   error codes) plus the W4 field request against the C009 catalog.
3. ``MemoryLifecycleLedger``: a small reference state machine built FROM the
   contract YAML. It executes hold -> bind -> retire with both PRO-16 identities,
   refill-waiter survival, external-write settlement with the three terminal
   states, the three distinct L2 moments, early input release and the page-lease
   acquire/release protocol. The page authorization semantics REFERENCE the
   existing AccessLease transaction set (page_pool.py); they do not re-implement
   a page pool.

Fail closed: any capacity without a source stays on the ``unresolved`` list with
``derivable: false`` + reason + needs; putting a value or expression there is an
error, not a default.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

H = Path(__file__).resolve().parents[1]  # hardware/coral_nexus
REPO = Path(__file__).resolve().parents[3]
UPSTREAM = H / "upstream" / "coralnpu"

CONSTRAINTS = H / "configs/w4_memory_constraints.yaml"
CONTRACT = H / "protocol/w4_memory.contract.yaml"
REQUEST = H / "protocol/field_requests/w4.yaml"
PROTOCOL = H / "protocol/cn_protocol_v1.yaml"


class BoundsError(Exception):
  """Coded error; ``code`` is stable so tests can pin it."""

  def __init__(self, code: str, message: str) -> None:
    super().__init__(f"{code}: {message}")
    self.code = code


class LedgerError(Exception):
  """Coded error from the reference lifecycle ledger."""

  def __init__(self, code: str, message: str) -> None:
    super().__init__(f"{code}: {message}")
    self.code = code


# ---------------------------------------------------------------------------
# Unit algebra
# ---------------------------------------------------------------------------
_UNIT_TABLE: dict[str, tuple[dict[str, int], float]] = {
  # Checked dimensions: bytes (B/KiB), refill-or-slot entries, cycles, bits and
  # the time base for rates. banks/jobs are plain counts (documentation only).
  "B": ({"B": 1}, 1.0),
  "KiB": ({"B": 1}, 1024.0),
  "entries": ({"entry": 1}, 1.0),
  "entry": ({"entry": 1}, 1.0),
  "cycles": ({"cycle": 1}, 1.0),
  "cycle": ({"cycle": 1}, 1.0),
  "bits": ({"bit": 1}, 1.0),
  "banks": ({}, 1.0),
  "bank": ({}, 1.0),
  "jobs": ({}, 1.0),
  "job": ({}, 1.0),
  "s": ({"s": 1}, 1.0),
  "GB_per_s": ({"B": 1, "s": -1}, 1e9),
  "MHz": ({"cycle": 1, "s": -1}, 1e6),
}


def _clean(dims: dict[str, int]) -> tuple[tuple[str, int], ...]:
  return tuple(sorted((k, v) for k, v in dims.items() if v != 0))


@dataclass(frozen=True)
class Unit:
  dims: tuple[tuple[str, int], ...]
  scale: float

  @property
  def name(self) -> str:
    if not self.dims:
      return "scalar"
    num = [f"{base}{'^' + str(p) if p != 1 else ''}" for base, p in self.dims if p > 0]
    den = [f"{base}{'^' + str(-p) if p != -1 else ''}" for base, p in self.dims if p < 0]
    out = "*".join(num) if num else "1"
    return out + ("/" + "*".join(den) if den else "")


def parse_unit(text: str) -> Unit:
  parts = text.split("/")
  dims: dict[str, int] = {}
  scale = 1.0
  for i, token in enumerate(parts):
    base = _UNIT_TABLE.get(token.strip())
    if base is None:
      raise BoundsError("UNKNOWN_UNIT", f"unit {text!r} has no dimension for {token!r}")
    for base_name, power in base[0].items():
      dims[base_name] = dims.get(base_name, 0) + (power if i == 0 else -power)
    scale *= base[1] if i == 0 else 1.0 / base[1]
  return Unit(_clean(dims), scale)


@dataclass(frozen=True)
class Quantity:
  value: float
  unit: Unit

  @property
  def canonical(self) -> float:
    return self.value * self.unit.scale


def _unit_for(dims: tuple[tuple[str, int], ...]) -> Unit:
  return Unit(dims, 1.0)


def q_add(left: Quantity, right: Quantity, op: str) -> Quantity:
  if left.unit.dims != right.unit.dims:
    raise BoundsError(
      "UNIT_MISMATCH", f"cannot {op} {left.unit.name} and {right.unit.name}: never add entries to cycles"
    )
  total = left.canonical + right.canonical if op == "add" else left.canonical - right.canonical
  unit = _unit_for(left.unit.dims)
  return Quantity(total / unit.scale, unit)


def q_mul(left: Quantity, right: Quantity) -> Quantity:
  dims: dict[str, int] = dict(left.unit.dims)
  for base, power in right.unit.dims:
    dims[base] = dims.get(base, 0) + power
  unit = _unit_for(_clean(dims))
  return Quantity(left.canonical * right.canonical / unit.scale, unit)


def q_div(left: Quantity, right: Quantity) -> Quantity:
  if right.canonical == 0:
    raise BoundsError("DIV_BY_ZERO", f"division by zero {right.value}{right.unit.name}")
  dims: dict[str, int] = dict(left.unit.dims)
  for base, power in right.unit.dims:
    dims[base] = dims.get(base, 0) - power
  unit = _unit_for(_clean(dims))
  return Quantity(left.canonical / right.canonical / unit.scale, unit)


def _quotient(a: Quantity, b: Quantity) -> tuple[float, Unit]:
  q = q_div(a, b)
  return q.canonical, _unit_for(q.unit.dims)


def _funcs() -> dict[str, Callable[..., Quantity]]:
  def ceil_div(a: Quantity, b: Quantity) -> Quantity:
    value, unit = _quotient(a, b)
    return Quantity(float(math.ceil(value)), unit)

  def floor_div(a: Quantity, b: Quantity) -> Quantity:
    value, unit = _quotient(a, b)
    return Quantity(float(math.floor(value)), unit)

  def mx(a: Quantity, b: Quantity) -> Quantity:
    if a.unit.dims != b.unit.dims:
      raise BoundsError("UNIT_MISMATCH", f"max() over {a.unit.name} and {b.unit.name}")
    return a if a.canonical >= b.canonical else b

  def mn(a: Quantity, b: Quantity) -> Quantity:
    if a.unit.dims != b.unit.dims:
      raise BoundsError("UNIT_MISMATCH", f"min() over {a.unit.name} and {b.unit.name}")
    return a if a.canonical <= b.canonical else b

  return {"ceil_div": ceil_div, "floor_div": floor_div, "max": mx, "min": mn}


_FUNCS = _funcs()


def eval_quantity(expr: str, env: dict[str, Quantity]) -> Quantity:
  """Evaluate a derivation expression over named Quantity values (safe AST)."""

  def rec(node: ast.AST) -> Quantity:
    if isinstance(node, ast.Expression):
      return rec(node.body)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
      left, right = rec(node.left), rec(node.right)
      if isinstance(node.op, ast.Add):
        return q_add(left, right, "add")
      if isinstance(node.op, ast.Sub):
        return q_add(left, right, "sub")
      return q_mul(left, right) if isinstance(node.op, ast.Mult) else q_div(left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
      operand = rec(node.operand)
      sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
      return Quantity(sign * operand.value, operand.unit)
    if isinstance(node, ast.Constant):
      if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Quantity(float(node.value), Unit((), 1.0))
      raise BoundsError("EXPR_SYNTAX", f"constant {node.value!r} is not a number")
    if isinstance(node, ast.Name):
      if node.id not in env:
        raise BoundsError("UNKNOWN_NAME", f"expression names {node.id!r} which no input/derivation defines")
      return env[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
      fn = _FUNCS.get(node.func.id)
      if fn is None:
        raise BoundsError("EXPR_SYNTAX", f"function {node.func.id!r} is not allowed")
      if len(node.args) != 2 or node.keywords:
        raise BoundsError("EXPR_SYNTAX", f"function {node.func.id} needs exactly 2 positional arguments")
      return fn(rec(node.args[0]), rec(node.args[1]))
    raise BoundsError("EXPR_SYNTAX", f"unsupported syntax: {type(node).__name__}")

  try:
    tree = ast.parse(expr, mode="eval")
  except SyntaxError as exc:
    raise BoundsError("EXPR_SYNTAX", f"expression {expr!r} does not parse: {exc}") from exc
  return rec(tree)


def _rounded(value: float) -> float:
  return float(int(value)) if float(value).is_integer() else value


# ---------------------------------------------------------------------------
# Source read-back
# ---------------------------------------------------------------------------
def _source_line(file: str, line: int) -> str | None:
  for root in (UPSTREAM, REPO):
    path = root / file
    if path.is_file():
      rows = path.read_text().splitlines()
      return rows[line - 1] if 1 <= line <= len(rows) else None
  return None


def check_source(errors: list[str], where: str, source: Any, needle: str, value: Any = None) -> None:
  if not isinstance(source, dict) or not {"file", "line", "needle"} <= set(source):
    errors.append(f"{where}: MISSING_SOURCE (need file/line/needle)")
    return
  text = _source_line(str(source["file"]), int(source["line"]))
  if text is None:
    errors.append(
      f"{where}: SOURCE_FILE_MISSING ({source['file']}:{source['line']} missing or out of range)"
    )
    return
  if str(source["needle"]) not in text:
    errors.append(
      f"{where}: SOURCE_LINE_MISMATCH ({source['file']}:{source['line']} lacks {source['needle']!r})"
    )
  if value is not None and str(value) not in text:
    errors.append(
      f"{where}: SOURCE_LINE_MISMATCH ({source['file']}:{source['line']} lacks value {value!r})"
    )


# ---------------------------------------------------------------------------
# Constraints validation + derivation
# ---------------------------------------------------------------------------
_WINDOW_NAMES = (
  "gather_window_entries_per_job_no_mshr_stall",
  "scatter_window_entries_per_job_no_mshr_stall",
)


def _check_unit_list(data: dict, errors: list[str]) -> None:
  allowed = data.get("units", {}).get("allowed")
  if not isinstance(allowed, list) or not allowed:
    errors.append("units.allowed: MISSING_UNIT_LIST")
    return
  for token in allowed:
    try:
      parse_unit(str(token))
    except BoundsError as exc:
      errors.append(f"units.allowed: {exc}")


# Quantities with no sourced bound. Held HERE so one edit to the constraints file cannot both delete an
# entry and its guard. Each stays derivable:false with reason + needs until its owner supplies a source.
REQUIRED_UNRESOLVED = {
  "per_ctx_outstanding_credit_cap": "C018",
  "mshr_token_width_bits": "C019",
  "response_record_budget": "C018",
  "mfe_external_address_bits": "C019",
  "lease_ref_width_bits": "C019",
}
REQUIRED_POW2_INPUTS = {"cache_line_bytes"}
_UNRESOLVED_FIELDS = {"name", "derivable", "reason", "needs", "owner"}


def validate_constraints(data: dict) -> tuple[dict[str, dict[str, Any]], list[str]]:
  """Validate the constraints file and derive every budget; errors collect, never guess."""
  errors: list[str] = []
  derived: dict[str, dict[str, Any]] = {}
  _check_unit_list(data, errors)
  env: dict[str, Quantity] = {}
  names: set[str] = set()

  for entry in data.get("inputs") or []:
    if not isinstance(entry, dict) or "name" not in entry:
      errors.append("inputs[]: MISSING_NAME")
      continue
    name = str(entry["name"])
    where = f"input {name}"
    if name in names:
      errors.append(f"{where}: DUPLICATE_NAME")
    names.add(name)
    raw = entry.get("value")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
      errors.append(f"{where}: INVALID_VALUE (value must be a number)")
      continue
    try:
      unit = parse_unit(str(entry.get("unit")))
    except BoundsError as exc:
      errors.append(f"{where}: {exc}")
      continue
    if not math.isfinite(raw):
      # NaN fails every comparison, so it would slip past each later guard; inf makes budgets unbounded.
      errors.append(f"{where}: NON_FINITE_VALUE ({raw})")
      continue
    if raw < 0:
      errors.append(f"{where}: NEGATIVE_CAPACITY ({raw})")
      continue
    if raw == 0 and not entry.get("allow_zero"):
      errors.append(f"{where}: VALUE_NOT_POSITIVE ({raw}); declare allow_zero or source a real value")
      continue
    if entry.get("requires") not in (None, "power_of_two"):
      errors.append(f"{where}: UNKNOWN_REQUIREMENT ({entry.get('requires')!r})")
    if entry.get("requires") == "power_of_two":
      check_source(
        errors, f"{where}.requires_source", entry.get("requires_source"), "line_bytes & (line_bytes - 1)"
      )
      if raw != int(raw) or int(raw) & (int(raw) - 1):
        errors.append(f"{where}: NOT_POWER_OF_TWO ({raw}); required by {entry.get('requires_source')}")
        continue
    if "key" not in entry:
      errors.append(f"{where}: MISSING_SOURCE (inputs need the config key they read)")
    check_source(errors, where, entry.get("source"), str(entry.get("key", "")), raw)
    env[name] = Quantity(float(raw), unit)

  for item in data.get("derivations") or []:
    if not isinstance(item, dict) or "name" not in item:
      errors.append("derivations[]: MISSING_NAME")
      continue
    name = str(item["name"])
    where = f"derivation {name}"
    if name in names:
      errors.append(f"{where}: DUPLICATE_NAME")
    names.add(name)
    expr = item.get("expr")
    if not isinstance(expr, str) or not expr:
      errors.append(f"{where}: MISSING_EXPR")
      continue
    try:
      declared = parse_unit(str(item.get("unit")))
      result = eval_quantity(expr, dict(env))
    except BoundsError as exc:
      errors.append(f"{where}: {exc}")
      continue
    if declared.dims != result.unit.dims:
      errors.append(
        f"{where}: UNIT_MISMATCH (declared {declared.name} but expression computes {result.unit.name})"
      )
      continue
    if name in _WINDOW_NAMES and result.canonical < 1:
      errors.append(
        f"{where}: OUT_OF_RANGE (envelope {_rounded(result.value)} < 1 entry per job;"
        " the MSHR cannot serve one job)"
      )
      continue
    if result.canonical <= 0:
      errors.append(f"{where}: NEGATIVE_CAPACITY ({_rounded(result.value)}{result.unit.name})")
      continue
    env[name] = result
    derived[name] = {"value": _rounded(result.value), "unit": result.unit.name, "expr": expr}

  for item in data.get("conservation_checks") or []:
    if not isinstance(item, dict) or "name" not in item:
      errors.append("conservation_checks[]: MISSING_NAME")
      continue
    name = str(item["name"])
    where = f"conservation {name}"
    ref = env.get(str(item.get("equals")))
    if ref is None:
      errors.append(f"{where}: UNKNOWN_NAME (equals {item.get('equals')!r})")
      continue
    try:
      left = eval_quantity(str(item.get("expr")), dict(env))
    except BoundsError as exc:
      errors.append(f"{where}: {exc}")
      continue
    if left.unit.dims != ref.unit.dims:
      errors.append(f"{where}: UNIT_MISMATCH ({left.unit.name} vs {ref.unit.name})")
      continue
    if abs(left.canonical - ref.canonical) > 1e-9:
      errors.append(
        f"{where}: PARTITION_NOT_CONSERVED"
        f" ({_rounded(left.value)}{left.unit.name} != {_rounded(ref.value)}{ref.unit.name})"
      )

  for item in data.get("unresolved") or []:
    if not isinstance(item, dict) or "name" not in item:
      errors.append("unresolved[]: MISSING_NAME")
      continue
    name = str(item["name"])
    where = f"unresolved {name}"
    if item.get("derivable") is not False:
      errors.append(f"{where}: NOT_FAIL_CLOSED (derivable must be false)")
    if REQUIRED_UNRESOLVED.get(name) not in (None, item.get("owner")):
      errors.append(f"{where}: OWNER_CHANGED (must stay {REQUIRED_UNRESOLVED[name]})")
    if not str(item.get("reason") or "").strip():
      errors.append(f"{where}: UNRESOLVED_WITHOUT_REASON")
    if not (isinstance(item.get("needs"), list) and item["needs"]):
      errors.append(f"{where}: UNRESOLVED_WITHOUT_NEEDS")
    # Closed field set: an unsourced entry may carry ONLY these keys. Any other key (value, expr, unit,
    # width_bits, width_rule, bits, depth, max_entries, ...) is a number or rule nobody sourced.
    for key in sorted(set(item) - _UNRESOLVED_FIELDS):
      errors.append(f"{where}: INVENTED_VALUE (unsourced field carries {key}={item[key]!r})")
  declared_pow2 = {
    str(e.get("name"))
    for e in data.get("inputs") or []
    if isinstance(e, dict) and e.get("requires") == "power_of_two"
  }
  for name in sorted(REQUIRED_POW2_INPUTS - declared_pow2):
    errors.append(f"input {name}: POWER_OF_TWO_REQUIREMENT_MISSING (cache.py:110 requires it)")
  seen_unresolved = {str(i.get("name")) for i in data.get("unresolved") or [] if isinstance(i, dict)}
  for required in sorted(set(REQUIRED_UNRESOLVED) - seen_unresolved):
    errors.append(
      f"unresolved {required}: REQUIRED_UNRESOLVED_MISSING (it has no sourced bound; deleting the entry "
      "does not make it derivable)"
    )
  for extra in sorted(seen_unresolved - set(REQUIRED_UNRESOLVED)):
    errors.append(f"unresolved {extra}: NOT_IN_REQUIRED_SET (add it to REQUIRED_UNRESOLVED deliberately)")
  check_mapping_shape(errors, data, "constraints")
  return derived, errors


def derive(data: dict) -> dict[str, dict[str, Any]]:
  """Derive every budget or raise; for callers that want fail-fast."""
  derived = validate_constraints(data)[0]
  if not derived:
    raise BoundsError("DERIVATION_ERROR", "no derivation produced a budget")
  return derived


def bounds_env(data: dict) -> dict[str, float]:
  """name -> numeric value for every input and derivation (ledger constructor input)."""
  values: dict[str, float] = {}
  for entry in data.get("inputs") or []:
    if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)):
      values[str(entry["name"])] = float(entry["value"])
  for name, budget in derive(data).items():
    values[name] = float(budget["value"])
  return values


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------
_EXPECTED_LEDGERS = (
  "cancellation",
  "isolation_confirmed",
  "stale_response",
  "credit",
  "inflight_references",
)
_EXPECTED_TERMINALS = {"COMPLETED", "ISOLATED", "FAULTED"}
_EXPECTED_ERRORS = {
  "address_out_of_range",
  "frame_slot_fault",
  "l1_view_double_free",
  "engine_queue_full",
  "l1_cache_maintenance_conflict",
  "stale_response",
  "scatter_segment_write_fault",
  "host_scope_inactive",
}
_EXPECTED_NON_GOALS = {
  "rtl_changes (this card writes contracts only; C026+ touches RTL)",
  "l2_noc_hbm_physical_redesign",
  "second_python_l1_execution_ledger",
  "changing_l2_physical_reclaim_granularity",
}
_EXPECTED_PROVENANCE = {
  "context_key",
  "txn_id",
  "slice_id",
  "slice_count",
  "arena_alloc_gen",
  "binding_generation",
  "scope_epoch",
}
_EXPECTED_PROTOCOL_IDS = {"GAT-19-ATOMIC", "GAT-19-TERMINAL", "GAT-20-REVOKE", "RELEASE-IDEMPOTENT"}
_REQUEST_KEY_FIELDS: dict[str, set[str]] = {
  "provenance.fields": _EXPECTED_PROVENANCE,
  "terminal.states": {"final_state"},
  "identity_domains.task_key": {"task_key"},
}
_REQUIRED_RTL_STATES = (
  "l1_bank_bytes",
  "l1_cache_mshr",
  "l1_arena_ledger",
  "slot_frame",
  "tile_program_memory",
  "cancel_isolation",
)
# name -> value; each is read back from the pinned upstream by check_source below.
_REQUIRED_BASELINES = {"lsuDataBits": 256, "lsuSlotTxidEntries": 8}
_REQUIRED_PY_STATES = ("external_legs", "l2_arena", "page_authority", "r_lease_root_retire")


def iter_sources(node: Any, where: str = "") -> list[tuple[str, dict]]:
  """Every `source: {file, line, ...}` mapping anywhere in the document, with its path.

  The hand-written per-section checks cover only the paths someone remembered. This walks the whole
  tree so a nested citation (for example addressing.indexed.errors[1]) cannot escape the read-back.
  """
  found: list[tuple[str, dict]] = []
  if isinstance(node, dict):
    src = node.get("source")
    if isinstance(src, dict) and ("file" in src or "line" in src):
      found.append((f"{where}.source" if where else "source", src))
    for key, value in node.items():
      found.extend(iter_sources(value, f"{where}.{key}" if where else str(key)))
  elif isinstance(node, list):
    for index, value in enumerate(node):
      found.extend(iter_sources(value, f"{where}[{index}]"))
  return found


def check_mapping_shape(errors: list[str], node: Any, label: str, where: str = "") -> None:
  """A mapping key must be a plain identifier.

  An unquoted comma inside a YAML flow mapping (`{ api: f(a, b), owner: X }`) silently splits the value
  into junk keys such as `binding: None`. No legitimate key contains parentheses, commas or spaces.
  """
  if isinstance(node, dict):
    for key, value in node.items():
      if (isinstance(key, str) and any(ch in key for ch in "(),")) or key is None:
        errors.append(f"{label}{where}: MALFORMED_KEY ({key!r}); quote the value that contains a comma")
      check_mapping_shape(errors, value, label, f"{where}.{key}")
  elif isinstance(node, list):
    for index, value in enumerate(node):
      check_mapping_shape(errors, value, label, f"{where}[{index}]")


def check_all_sources(errors: list[str], document: dict, label: str) -> int:
  """Read back every cited line in `document`; a source without a needle or line is an error."""
  count = 0
  for where, src in iter_sources(document):
    file = src.get("file")
    if not (isinstance(file, str) and file.startswith(("pipeline_validator", "hdl", "hardware"))):
      continue  # a design-doc anchor (components/...) has no line to read back
    count += 1
    check_source(errors, f"{label}{where}", src, str(src.get("needle", "")))
  return count


def _rule_sources(errors: list[str], where: str, items: Any) -> None:
  if not isinstance(items, list):
    return
  for i, item in enumerate(items):
    if isinstance(item, dict) and "source" in item:
      check_source(errors, f"{where}[{i}]", item["source"], str(item["source"].get("needle", "")))


def validate_contract(
  contract: dict, derived: dict[str, dict[str, Any]], request: dict, protocol: dict
) -> list[str]:
  errors: list[str] = []

  own = contract.get("ownership") or {}
  rtl_rows = [r for r in own.get("rtl_executes") or [] if isinstance(r, dict)]
  py_rows = [r for r in own.get("python_executes") or [] if isinstance(r, dict)]
  for row in rtl_rows + py_rows:
    if not {"state", "scope", "python_role"} <= set(row):
      errors.append(f"ownership: row {row.get('state')} missing state/scope/python_role")
  rtl_states = {str(r.get("state")) for r in rtl_rows}
  py_states = {str(r.get("state")) for r in py_rows}
  for required in _REQUIRED_RTL_STATES:
    if required not in rtl_states:
      errors.append(f"ownership.rtl_executes: missing RTL-owned state {required}")
  for required in _REQUIRED_PY_STATES:
    if required not in py_states:
      errors.append(f"ownership.python_executes: missing Python-owned state {required}")
  overlap = rtl_states & py_states
  if overlap:
    errors.append(f"ownership: {sorted(overlap)} claimed by BOTH sides (single-owner rule)")
  goals = set(own.get("non_goals") or [])
  if goals != _EXPECTED_NON_GOALS:
    errors.append(f"ownership.non_goals: expected {_EXPECTED_NON_GOALS - goals} missing -> {sorted(goals)}")

  prov = contract.get("provenance") or {}
  pfields = [f for f in prov.get("fields") or [] if isinstance(f, dict)]
  pnames = {str(f.get("name")) for f in pfields}
  if pnames != _EXPECTED_PROVENANCE:
    errors.append(
      f"provenance.fields: must be exactly {sorted(_EXPECTED_PROVENANCE)} (got {sorted(pnames)})"
    )
  for f in pfields:
    if "owner" not in f:
      errors.append(f"provenance.fields[{f.get('name')}]: missing owner")
    if f.get("source") is not None:
      check_source(errors, f"provenance.{f.get('name')}", f["source"], str(f["source"].get("needle", "")))
  if "provenance tuple" not in str(prov.get("rule")) or "stale" not in str(prov.get("rule")):
    errors.append("provenance.rule: must tie ACKs to the provenance tuple and call mismatches stale")

  addr = contract.get("addressing") or {}
  if (addr.get("mfe_external") or {}).get("address_bits") != "unresolved":
    errors.append("addressing.mfe_external.address_bits: must stay unresolved until capacities freeze")
  formula = addr.get("indexed") or {}
  if "remote_element" not in str(formula.get("formula")):
    errors.append("addressing.indexed.formula: must restate the SegmentResolver formula")
  if isinstance(formula.get("source"), dict):
    check_source(
      errors, "addressing.indexed.formula", formula["source"], str(formula["source"].get("needle", ""))
    )
  wp = formula.get("window_entries_positive")
  if isinstance(wp, dict) and isinstance(wp.get("source"), dict):
    check_source(
      errors,
      "addressing.indexed.window_entries_positive",
      wp["source"],
      str(wp["source"].get("needle", "")),
    )
  else:
    errors.append("addressing.indexed: window_entries_positive source missing")

  page = contract.get("page_authorization") or {}
  apis = {
    str(a.get("api")).split("(")[0].strip(): a
    for a in page.get("existing_transaction_set") or []
    if isinstance(a, dict)
  }
  for api in (
    "PagePoolRegistry.lease",
    "PagePoolRegistry.scope_owns_range",
    "AccessLease.acquire",
    "AccessLease.release",
    "AccessLease.is_valid",
  ):
    api_row = apis.get(api)
    if api_row is None:
      errors.append(
        f"page_authorization.existing_transaction_set: {api} missing (consume, do not re-implement)"
      )
    else:
      src = api_row.get("source")
      if isinstance(src, dict):
        check_source(errors, f"page_authorization.{api}", src, str(src.get("needle", "")))
  ids = {str(p.get("id")) for p in page.get("protocol") or [] if isinstance(p, dict)}
  if ids != _EXPECTED_PROTOCOL_IDS:
    errors.append(
      f"page_authorization.protocol: ids must be {sorted(_EXPECTED_PROTOCOL_IDS)} (got {sorted(ids)})"
    )
  for p in page.get("protocol") or []:
    if isinstance(p, dict) and not str(p.get("text") or "").strip():
      errors.append(f"page_authorization.protocol[{p.get('id')}]: empty text")
  changes = {str(c.get("api")) for c in page.get("changes_required") or [] if isinstance(c, dict)}
  if not any("acquire_range" in c for c in changes) or not any("idempotent" in c for c in changes):
    errors.append(
      "page_authorization.changes_required: acquire_range + per-txn idempotent release must be listed"
    )

  ledgers = [
    str(row.get("name"))
    for row in (contract.get("accounting_ledgers") or {}).get("ledgers") or []
    if isinstance(row, dict)
  ]
  if set(ledgers) != set(_EXPECTED_LEDGERS):
    errors.append(f"accounting_ledgers: must be exactly {list(_EXPECTED_LEDGERS)} (got {ledgers})")
  if "independent" not in str((contract.get("accounting_ledgers") or {}).get("rule")):
    errors.append(
      "accounting_ledgers.rule: must state the ledgers are independent (no cross-ledger netting)"
    )

  term = contract.get("terminal") or {}
  if set(term.get("states") or []) != _EXPECTED_TERMINALS:
    errors.append(f"terminal.states: must be exactly {sorted(_EXPECTED_TERMINALS)}")
  if term.get("cancel_marker") != "CANCEL_REQUESTED":
    errors.append("terminal.cancel_marker: must be CANCEL_REQUESTED (a marker, not a terminal)")
  _rule_sources(errors, "terminal.rules_sources", term.get("rules_sources"))

  life = contract.get("lifecycle") or {}
  if list(life.get("hold_states") or []) != ["free", "held", "bound"]:
    errors.append("lifecycle.hold_states: must be [free, held, bound]")
  conds = {
    str(c.get("id")): str(c.get("identity")) for c in life.get("conservation") or [] if isinstance(c, dict)
  }
  for cid in ("PRO-16-HOLD", "PRO-16-CAPACITY", "PRO-16-ORACLE"):
    if cid not in conds:
      errors.append(f"lifecycle.conservation: {cid} missing")
  if "allocated == held + bound" not in conds.get("PRO-16-HOLD", ""):
    errors.append("lifecycle.conservation PRO-16-HOLD: identity text drift")
  if "allocated + free == user_spm_per_bank" not in conds.get("PRO-16-CAPACITY", ""):
    errors.append("lifecycle.conservation PRO-16-CAPACITY: identity text drift")
  _rule_sources(errors, "lifecycle.conservation", life.get("conservation"))
  phases = [str(p) for p in (life.get("hold_bind_retire") or {}).get("phases") or []]
  expected_order = [p for p in ("HOLD_ACK", "BIND_INSTALL", "RETIRE", "TASK_COMPLETION") if p in phases]
  if expected_order != ["HOLD_ACK", "BIND_INSTALL", "RETIRE", "TASK_COMPLETION"]:
    errors.append(
      "lifecycle.hold_bind_retire.phases: HOLD_ACK < BIND_INSTALL < RETIRE < TASK_COMPLETION broken"
    )
  _rule_sources(
    errors, "lifecycle.hold_bind_retire", (life.get("hold_bind_retire") or {}).get("cosim_baseline")
  )
  _rule_sources(errors, "lifecycle.l1_free_vs_retirement", life.get("l1_free_vs_retirement"))
  l2_rules = life.get("l2_release") or []
  if not isinstance(l2_rules, list) or len(l2_rules) < 3:
    errors.append("lifecycle.l2_release: needs the three distinct-moment rules")
  else:
    if "WHOLE" not in str(l2_rules[1].get("rule")) or "UNCHANGED" not in str(l2_rules[1].get("rule")):
      errors.append("lifecycle.l2_release[1]: whole-span final-free and unchanged granularity must stay")
    if "DISTINCT" not in str(l2_rules[2].get("rule")):
      errors.append("lifecycle.l2_release[2]: three DISTINCT moments rule must stay")
    _rule_sources(errors, "lifecycle.l2_release", l2_rules)
  early = life.get("early_input_release") or []
  early_text = " ".join(str(r.get("rule", r)) if isinstance(r, dict) else str(r) for r in early)
  if not isinstance(early, list) or len(early) < 3:
    errors.append("lifecycle.early_input_release: per-Task / all_tasks / idempotent rules all required")
  elif "idempoten" not in early_text:
    errors.append("lifecycle.early_input_release: idempotent remaining-pin retirement rule missing")
  gate = life.get("profile_gate") or {}
  if list(gate.get("blocked") or []) != ["new_data_issue"]:
    errors.append("lifecycle.profile_gate.blocked: must be exactly [new_data_issue]")
  exempt = set(gate.get("exempt") or [])
  for path in ("completion_ack", "maintenance_clean", "program_install", "cancel_and_drain"):
    if path not in exempt:
      errors.append(f"lifecycle.profile_gate.exempt: {path} must stay unblocked")
  if set(gate.get("blocked") or []) & exempt:
    errors.append("lifecycle.profile_gate: a path is both blocked and exempt")
  gate_rule = str(gate.get("rule"))
  if "ONLY new data issue" not in gate_rule or "never blocks a terminal ACK" not in gate_rule:
    errors.append(
      "lifecycle.profile_gate.rule: gate scope text drift (new data issue only; terminal ACK never)"
    )

  mshr = contract.get("mshr") or {}
  if list(mshr.get("token_states") or []) != ["ALLOCATED", "ISSUED_REFILL", "WAITERS_PENDING", "FREE"]:
    errors.append("mshr.token_states: lifecycle drift")
  if list(mshr.get("merge_key") or []) != ["binding_id", "binding_generation", "line_offset"]:
    errors.append("mshr.merge_key: must stay the provenance triple")
  cap = str(mshr.get("capacity_ref"))
  if cap not in derived:
    errors.append(f"mshr.capacity_ref: {cap!r} does not resolve in the constraints derivations")
  _rule_sources(errors, "mshr.cosim_baseline", mshr.get("cosim_baseline"))

  err_codes = {
    str(e.get("code")) for e in (contract.get("errors") or {}).get("codes") or [] if isinstance(e, dict)
  }
  if err_codes != _EXPECTED_ERRORS:
    errors.append(f"errors.codes: must be exactly {sorted(_EXPECTED_ERRORS)} (got {sorted(err_codes)})")

  # The baselines the contract must keep, with the value each carries. Held here so one YAML edit
  # cannot delete a baseline and the check that would notice.
  present = {
    str(b.get("name")): b.get("value")
    for b in contract.get("upstream_baselines") or []
    if isinstance(b, dict)
  }
  for name, value in _REQUIRED_BASELINES.items():
    if name not in present:
      errors.append(f"upstream_baselines.{name}: missing (the contract must keep this upstream baseline)")
    elif present[name] != value:
      errors.append(f"upstream_baselines.{name}: value {present[name]!r} != required {value!r}")
  for base in contract.get("upstream_baselines") or []:
    if isinstance(base, dict):
      source = base.get("source") or {}
      check_source(
        errors,
        f"upstream_baselines.{base.get('name')}",
        source,
        str(source.get("needle", "")),
        base.get("value"),
      )

  check_all_sources(errors, contract, "contract")
  check_mapping_shape(errors, contract, "contract")
  errors.extend(check_field_request(request, contract, protocol))
  return errors


def _protocol_fields(protocol: dict) -> dict[str, set[str]]:
  out: dict[str, set[str]] = {}
  for domain, body in (protocol.get("identity_domains") or {}).items():
    if isinstance(body, dict):
      out[f"identity_domains.{domain}"] = {
        str(f.get("name")) for f in body.get("fields") or [] if isinstance(f, dict)
      }
  out.update(_REQUEST_KEY_FIELDS)
  return out


def check_field_request(request: dict, contract: dict, protocol: dict) -> list[str]:
  errors: list[str] = []
  if request.get("schema") != "cn_field_request/v1" or request.get("package") != "W4":
    errors.append("field_request: schema/package must be cn_field_request/v1 + W4")
  catalog = {str(m.get("name")): m for m in protocol.get("messages") or [] if isinstance(m, dict)}
  channels = {str(c.get("name")) for c in protocol.get("channels") or [] if isinstance(c, dict)}
  prov_fields = {
    str(f.get("name"))
    for f in (contract.get("provenance") or {}).get("fields") or []
    if isinstance(f, dict)
  }
  key_fields = _protocol_fields(protocol)
  key_fields["provenance.fields"] = prov_fields  # driven by the contract YAML, no fallback
  for req in request.get("requests") or []:
    if not isinstance(req, dict):
      continue
    msg = catalog.get(str(req.get("message")))
    if msg is None:
      errors.append(f"field_request: message {req.get('message')!r} is not in the C009 catalog")
      continue
    if str(msg.get("dir")) != str(req.get("direction")):
      errors.append(
        f"field_request.{req.get('message')}: direction {req.get('direction')} != catalog {msg.get('dir')}"
      )
    for f in req.get("fields_needed") or []:
      if not isinstance(f, dict):
        continue
      name = str(f.get("name"))
      if not (H / str(f.get("from"))).is_file():
        errors.append(
          f"field_request.{req.get('message')}.{name}: from {f.get('from')!r} is not a protocol file"
        )
      allowed = key_fields.get(str(f.get("key")))
      if allowed is None:
        errors.append(f"field_request.{req.get('message')}.{name}: unknown key {f.get('key')!r}")
      elif name not in allowed:
        errors.append(f"field_request.{req.get('message')}.{name}: no such field under {f.get('key')}")
  for new in request.get("new_messages_requested") or []:
    if not isinstance(new, dict):
      continue
    if str(new.get("name")) in catalog:
      errors.append(f"field_request.new_messages_requested.{new.get('name')}: already in the catalog")
    if str(new.get("channel")) not in channels:
      errors.append(
        f"field_request.new_messages_requested.{new.get('name')}: unknown channel {new.get('channel')}"
      )
    if str(new.get("dir")) not in ("G2T", "T2G"):
      errors.append(
        f"field_request.new_messages_requested.{new.get('name')}: bad direction {new.get('dir')}"
      )
  return errors


# ---------------------------------------------------------------------------
# Reference lifecycle ledger (driven by the contract YAML)
# ---------------------------------------------------------------------------
@dataclass
class _BankState:
  free: int
  held: int = 0
  bound: int = 0

  @property
  def allocated(self) -> int:
    return self.held + self.bound


@dataclass
class _Ticket:
  bank: int
  size: int
  state: str  # held | bound | retired
  slots: tuple[int, ...]
  arena_alloc_gen: int


@dataclass
class _View:
  ticket: str
  bank: int
  size: int
  permission: str
  slot: int
  arena_alloc_gen: int
  invalidated: bool = False
  inflight: int = 0


@dataclass
class _MshrEntry:
  state: str
  generation: int
  waiters: int = 0
  fill_bytes: int = 0
  cache_line: bool = False


@dataclass
class _Txn:
  state: str = "REGISTERED"
  credit: int = 0
  credit_returned: int = 0
  inflight: int = 0
  lease: tuple[str, frozenset[int]] | None = None
  ext_pending: int = 0
  ext_confirmed: int = 0
  isolated: bool = False


@dataclass
class _L2Buffer:
  name: str
  producer: str
  readonly_shared: bool
  padded_bytes: int
  claims: set[str] = field(default_factory=set)
  refs: int = 0
  released: bool = False
  freed: bool = False
  origin_retired: bool = False


@dataclass
class _Root:
  name: str
  buffers: tuple[str, ...]
  retired: bool = False
  completion: bool = False
  outstanding: int = 0
  task_leases: int = 0


@dataclass
class _Pin:
  task: str
  buffer: str
  reads: bool
  writes: bool


class MemoryLifecycleLedger:
  """YAML-driven reference machine for hold->bind->retire plus settlement.

  Counter names, terminal states, gate scope, MSHR token states and hold states
  all come from the contract dict, so mutating the YAML changes behaviour. The
  page-lease operations implement the CONTRACT protocol (atomic check+take,
  terminal-ACK release, idempotent release); the page pool itself is the
  existing AccessLease/PagePoolRegistry set the contract cites -- it is not
  re-implemented here.
  """

  def __init__(self, contract: dict, values: dict[str, float]) -> None:
    life = contract["lifecycle"]
    self.hold_states = tuple(life["hold_states"])
    self.terminal_states = tuple(contract["terminal"]["states"])
    self.cancel_marker = str(contract["terminal"]["cancel_marker"])
    self.ledgers = {str(row["name"]): 0 for row in contract["accounting_ledgers"]["ledgers"]}
    self.gate_blocked = set(life["profile_gate"]["blocked"])
    self.gate_exempt = set(life["profile_gate"]["exempt"])
    self.token_states = tuple(contract["mshr"]["token_states"])
    self.gate_closed = False
    self.issued_by_kind: dict[str, int] = {}
    banks = int(values["tile_l1_banks"])
    self.per_bank = int(values["user_spm_per_bank_mode0"])
    self._banks = [_BankState(free=self.per_bank) for _ in range(banks)]
    self.mshr_capacity = int(values["l1_refill_budget"])
    self.frame_slots = int(values["frame_slot_budget"])
    self._tickets: dict[str, _Ticket] = {}
    self._views: dict[str, _View] = {}
    self._txns: dict[str, _Txn] = {}
    self._mshr: list[_MshrEntry | None] = [None] * self.mshr_capacity
    self._slots_used: set[int] = set()
    self._arena_gen = 0
    self._l2: dict[str, _L2Buffer] = {}
    self._roots: dict[str, _Root] = {}
    self._pins: list[_Pin] = []

  # -- conservation ----------------------------------------------------
  def check_conservation(self) -> None:
    for bank_id, bank in enumerate(self._banks):
      if bank.allocated + bank.free != self.per_bank:
        raise LedgerError(
          "LEDGER_CONSERVATION",
          f"bank {bank_id}: allocated + free != user_spm_per_bank"
          f" ({bank.allocated}+{bank.free} != {self.per_bank})",
        )

  def conservation(self) -> list[dict[str, int]]:
    self.check_conservation()
    return [
      {"bank": i, "held": b.held, "bound": b.bound, "allocated": b.allocated, "free": b.free}
      for i, b in enumerate(self._banks)
    ]

  # -- hold -> bind -> retire -------------------------------------------
  def hold(self, ticket: str, bank: int, size: int, slots: tuple[int, ...] = ()) -> None:
    if size <= 0:
      raise LedgerError("HOLD_NEGATIVE_SIZE", f"ticket {ticket}: size {size} must be positive")
    if not 0 <= bank < len(self._banks):
      raise LedgerError("BANK_OUT_OF_RANGE", f"bank {bank} outside 0..{len(self._banks) - 1}")
    if ticket in self._tickets:
      raise LedgerError("HOLD_DUPLICATE_TICKET", f"ticket {ticket} already exists")
    for slot in slots:
      if not 0 <= slot < self.frame_slots:
        raise LedgerError("SLOT_OUT_OF_RANGE", f"slot {slot} outside 0..{self.frame_slots - 1}")
      if slot in self._slots_used:
        raise LedgerError("SLOT_IN_USE", f"slot {slot} already bound")
    state = self._banks[bank]
    if size > state.free:
      raise LedgerError(
        "HOLD_OVERCOMMIT",
        f"ticket {ticket} wants {size}B on bank {bank} but only {state.free}B free (no oversell)",
      )
    state.free -= size
    state.held += size
    self._slots_used.update(slots)
    self._arena_gen += 1
    self._tickets[ticket] = _Ticket(bank, size, "held", tuple(slots), self._arena_gen)
    self.check_conservation()

  def bind(self, ticket: str) -> int:
    record = self._tickets.get(ticket)
    if record is None or record.state == "retired":
      raise LedgerError("BIND_WITHOUT_HOLD", f"ticket {ticket} has no held reservation")
    if record.state == "bound":
      raise LedgerError("BIND_ALREADY", f"ticket {ticket} is already bound")
    state = self._banks[record.bank]
    state.held -= record.size
    state.bound += record.size
    record.state = "bound"
    self.check_conservation()
    return record.arena_alloc_gen

  def compensate(self, ticket: str) -> None:
    record = self._tickets.get(ticket)
    if record is None or record.state != "held":
      raise LedgerError(
        "COMPENSATE_WITHOUT_HOLD", f"ticket {ticket} cannot compensate (pre-authorization only)"
      )
    state = self._banks[record.bank]
    state.held -= record.size
    state.free += record.size
    self._slots_used.difference_update(record.slots)
    record.state = "retired"
    self.check_conservation()

  def alloc_view(self, ticket: str, view: str, size: int, permission: str, slot: int) -> int:
    record = self._tickets.get(ticket)
    if record is None or record.state != "bound":
      raise LedgerError("VIEW_WITHOUT_BOUND_TICKET", f"view {view}: ticket {ticket} is not bound")
    if view in self._views:
      raise LedgerError("VIEW_DOUBLE_FREE", f"view {view} already exists")
    if not 0 <= slot < self.frame_slots or slot not in record.slots:
      raise LedgerError("SLOT_OUT_OF_RANGE", f"slot {slot} not part of ticket {ticket}")
    if permission not in ("r", "rw"):
      raise LedgerError("PERMISSION_DENIED", f"permission {permission!r} must be r or rw")
    self._views[view] = _View(ticket, record.bank, size, permission, slot, record.arena_alloc_gen)
    self.check_conservation()
    return record.arena_alloc_gen

  def free_view(self, view: str) -> None:
    record = self._views.get(view)
    if record is None:
      raise LedgerError("VIEW_UNKNOWN", f"view {view} does not exist")
    if record.invalidated:
      raise LedgerError("VIEW_DOUBLE_FREE", f"view {view} is already invalidated")
    if record.inflight:
      raise LedgerError("VIEW_INFLIGHT", f"view {view} has {record.inflight} in-flight accesses")
    record.invalidated = True  # only the VIEW: bytes stay bound until retirement

  def retire(self, ticket: str, arena_alloc_gen: int | None = None) -> None:
    record = self._tickets.get(ticket)
    if record is None:
      raise LedgerError("RETIRE_UNKNOWN_TICKET", f"ticket {ticket} does not exist")
    if record.state == "retired":
      raise LedgerError("RETIRE_ALREADY", f"ticket {ticket} is already retired")
    if arena_alloc_gen is not None and arena_alloc_gen != record.arena_alloc_gen:
      raise LedgerError(
        "RETIRE_GENERATION_MISMATCH",
        f"retire for gen {arena_alloc_gen} but the live allocation is gen {record.arena_alloc_gen}",
      )
    live = [v for v in self._views.values() if v.ticket == ticket and not v.invalidated]
    if live:
      raise LedgerError("RETIRE_WITH_LIVE_VIEWS", f"ticket {ticket} still has live views ({len(live)})")
    inflight = sum(v.inflight for v in self._views.values() if v.ticket == ticket)
    if inflight:
      raise LedgerError("RETIRE_WITH_INFLIGHT", f"ticket {ticket} has {inflight} in-flight references")
    state = self._banks[record.bank]
    state.bound -= record.size
    state.free += record.size
    self._slots_used.difference_update(record.slots)
    record.state = "retired"
    self.check_conservation()

  # -- gate -------------------------------------------------------------
  def set_gate(self, closed: bool) -> None:
    self.gate_closed = closed

  def data_issue(self, kind: str) -> bool:
    if self.gate_closed and kind in self.gate_blocked:
      raise LedgerError("GATE_BLOCKS_NEW_ISSUE", f"kind {kind!r} is new data issue and the gate is closed")
    self.issued_by_kind[kind] = self.issued_by_kind.get(kind, 0) + 1
    return True

  # -- credit / in-flight ledgers ----------------------------------------
  def _txn(self, txn: str) -> _Txn:
    if txn not in self._txns:
      self._txns[txn] = _Txn()
    return self._txns[txn]

  def issue_credit(self, txn: str, count: int = 1) -> None:
    record = self._txn(txn)
    if record.state in self.terminal_states:
      raise LedgerError("CREDIT_AFTER_TERMINAL", f"txn {txn} is terminal; no new credit")
    record.credit += count
    self.ledgers["credit"] += count

  def return_credit(self, txn: str, count: int = 1) -> None:
    record = self._txn(txn)
    outstanding = record.credit - record.credit_returned
    if count > outstanding:
      code = "CREDIT_UNDERFLOW" if outstanding == 0 else "CREDIT_OVER_RETURN"
      raise LedgerError(code, f"txn {txn}: returning {count} with {outstanding} outstanding")
    record.credit_returned += count
    self.ledgers["credit"] -= count

  def begin_inflight(self, view: str, txn: str) -> None:
    record = self._views.get(view)
    if record is None or record.invalidated:
      raise LedgerError("ISSUE_ON_INVALIDATED_VIEW", f"view {view} is not live")
    record.inflight += 1
    self._txn(txn).inflight += 1
    self.ledgers["inflight_references"] += 1

  def end_inflight(self, view: str, txn: str) -> None:
    record = self._views.get(view)
    if record is None or record.inflight == 0:
      raise LedgerError("INFLOW_MISMATCH", f"view {view} has no in-flight reference to end")
    record.inflight -= 1
    self._txn(txn).inflight -= 1
    self.ledgers["inflight_references"] -= 1

  # -- MSHR --------------------------------------------------------------
  def _mshr_entry(self, slot: int) -> _MshrEntry:
    entry = self._mshr[slot] if 0 <= slot < self.mshr_capacity else None
    if entry is None:
      raise LedgerError("MSHR_UNKNOWN_TOKEN", f"slot {slot} holds no entry")
    return entry

  def mshr_allocate(self, generation: int) -> int:
    for slot, entry in enumerate(self._mshr):
      if entry is None:
        self._mshr[slot] = _MshrEntry(state=self.token_states[0], generation=generation)
        return slot
    raise LedgerError("MSHR_FULL", f"all {self.mshr_capacity} refill entries are live")

  def mshr_merge(self, slot: int, generation: int) -> None:
    entry = self._mshr_entry(slot)
    if entry.generation != generation:
      raise LedgerError(
        "MSHR_GENERATION_MISMATCH", f"merge on gen {generation} but the entry holds gen {entry.generation}"
      )

  def mshr_add_waiter(self, slot: int) -> None:
    self._mshr_entry(slot).waiters += 1

  def mshr_issue_refill(self, slot: int) -> None:
    entry = self._mshr_entry(slot)
    if entry.state != self.token_states[0]:
      raise LedgerError("MSHR_STATE", f"slot {slot} in {entry.state} cannot issue a refill")
    entry.state = self.token_states[1]

  def mshr_refill(self, slot: int, line_bytes: int) -> None:
    entry = self._mshr_entry(slot)
    if entry.state != self.token_states[1]:
      raise LedgerError("MSHR_STATE", f"slot {slot} in {entry.state} cannot accept refill data")
    entry.fill_bytes = line_bytes
    entry.cache_line = True
    entry.state = self.token_states[2]

  def mshr_evict_cache_line(self, slot: int) -> None:
    """Cache eviction touches only the cache line, never the entry's fill buffer."""
    self._mshr_entry(slot).cache_line = False

  def mshr_waiter_consume(self, slot: int) -> int:
    entry = self._mshr_entry(slot)
    if entry.waiters == 0:
      raise LedgerError("MSHR_STATE", f"slot {slot} has no registered waiter")
    if entry.state != self.token_states[2]:
      raise LedgerError("MSHR_STATE", f"slot {slot} in {entry.state} has no refill data yet")
    entry.waiters -= 1
    if entry.waiters == 0:
      self._mshr_free(slot)
    return entry.fill_bytes

  def _mshr_free(self, slot: int) -> None:
    entry = self._mshr_entry(slot)
    if entry.waiters > 0:
      raise LedgerError(
        "MSHR_TOKEN_EARLY_REUSE", f"slot {slot} still has {entry.waiters} waiters without reliable data"
      )
    self._mshr[slot] = None  # slot reusable; the next allocate writes a fresh generation

  def mshr_force_free(self, slot: int) -> None:
    """Negative probe: attempt to free a token regardless of waiters (must refuse)."""
    self._mshr_free(slot)

  # -- external writes / terminals ---------------------------------------
  def accept_external_write(self, txn: str, view: str, nbytes: int) -> None:
    record = self._txn(txn)
    if record.isolated:
      raise LedgerError("ISOLATION_BROKEN", f"txn {txn} is isolated; no external write may follow")
    if record.state in self.terminal_states:
      raise LedgerError("TERMINAL_ALREADY", f"txn {txn} is terminal ({record.state})")
    view_record = self._views.get(view)
    if view_record is None or "w" not in view_record.permission:
      raise LedgerError("PERMISSION_DENIED", f"view {view} is read-only; external write refused")
    record.ext_pending += nbytes

  def external_write_response(self, txn: str, nbytes: int) -> None:
    record = self._txn(txn)
    if record.state in self.terminal_states:
      self.ledgers["stale_response"] += 1  # a late response is never a completion
      return
    record.ext_pending -= nbytes
    record.ext_confirmed += nbytes

  def revoke_external_write(self, txn: str) -> None:
    raise LedgerError(
      "EXTERNAL_WRITE_IRREVOCABLE",
      f"txn {txn}: an accepted external write cannot be revoked by generation or timeout",
    )

  def cancel(self, txn: str) -> None:
    record = self._txn(txn)
    if record.state in self.terminal_states or record.state == self.cancel_marker:
      return
    record.state = self.cancel_marker
    self.ledgers["cancellation"] += 1

  def settle(self, txn: str, state: str) -> None:
    if state not in self.terminal_states:
      raise LedgerError("UNKNOWN_TERMINAL", f"{state!r} is not one of {list(self.terminal_states)}")
    record = self._txn(txn)
    if record.state in self.terminal_states:
      raise LedgerError("DUPLICATE_TERMINAL", f"txn {txn} already settled as {record.state}")
    if state == "ISOLATED" and record.state != self.cancel_marker:
      raise LedgerError(
        "ISOLATION_REQUIRES_MARKER", "isolation confirmation requires the cancel marker first"
      )
    if state == "COMPLETED" and record.ext_pending:
      raise LedgerError(
        "COMPLETION_WITH_UNSETTLED_WRITE",
        f"txn {txn} still has {record.ext_pending}B of unconfirmed external writes",
      )
    if record.inflight:
      raise LedgerError("TERMINAL_WITH_INFLIGHT", f"txn {txn} holds {record.inflight} in-flight references")
    record.state = state
    record.isolated = state == "ISOLATED"
    if state == "ISOLATED":
      self.ledgers["isolation_confirmed"] += 1
    outstanding = record.credit - record.credit_returned
    if outstanding:
      self.return_credit(txn, outstanding)
    if record.lease is not None:
      self.lease_release(txn)

  def late_response(self, txn: str) -> None:
    self._txn(txn)
    self.ledgers["stale_response"] += 1

  # -- page lease (contract protocol over the existing AccessLease set) ---
  def lease_acquire(
    self, txn: str, scope: str, pages: dict[int, tuple[str, int]], epoch: int
  ) -> tuple[str, frozenset[int]]:
    """Atomic check+take: every page must be owned by ``scope`` at ``epoch``."""
    for page, (owner, page_epoch) in pages.items():
      if owner != scope or page_epoch != epoch:
        raise LedgerError(
          "HOST_SCOPE_INACTIVE",
          f"page {page} owner {owner!r}@{page_epoch} != {scope!r}@{epoch}; nothing taken",
        )
    record = self._txn(txn)
    if record.lease is not None:
      raise LedgerError("LEASE_ALREADY_HELD", f"txn {txn} already holds a lease")
    record.lease = (scope, frozenset(pages))
    return record.lease

  def lease_release(self, txn: str) -> None:
    record = self._txn(txn)
    if record.state not in self.terminal_states:
      raise LedgerError(
        "LEASE_NOT_TERMINAL", f"txn {txn} is {record.state}; release waits for the terminal ACK"
      )
    if record.lease is None:
      raise LedgerError("LEASE_DOUBLE_RELEASE", f"txn {txn} has no lease to release")
    record.lease = None

  # -- L2: three distinct moments ----------------------------------------
  def add_l2_buffer(
    self,
    name: str,
    producer: str,
    padded_bytes: int,
    *,
    readonly_shared: bool,
    claims: set[str] | None = None,
  ) -> None:
    self._l2[name] = _L2Buffer(name, producer, readonly_shared, padded_bytes, claims=set(claims or ()))

  def add_root(
    self, name: str, buffers: tuple[str, ...], *, outstanding: int = 0, task_leases: int = 0
  ) -> None:
    self._roots[name] = _Root(name, buffers, outstanding=outstanding, task_leases=task_leases)

  def l2_close_claim(self, buffer: str, claim: str) -> None:
    self._l2[buffer].claims.discard(claim)  # claims close through their view release

  def l2_release(self, buffer: str) -> None:
    record = self._l2[buffer]
    if record.claims:
      raise LedgerError(
        "L2_CLAIMS_OPEN",
        f"buffer {buffer} still has claims {sorted(record.claims)}; forfeiture is all-or-nothing",
      )
    record.released = True  # permanent forfeiture

  def l2_claim(self, buffer: str, claim: str) -> None:
    record = self._l2[buffer]
    if record.released:
      raise LedgerError("L2_FORFEITED", f"buffer {buffer} was released; new claims are impossible")

  def l2_final_free(self, buffer: str) -> int | None:
    record = self._l2[buffer]
    if not record.released or record.freed or record.refs or record.claims:
      return None
    if record.readonly_shared and not record.origin_retired:
      return None  # shared readonly backing frees only after the producer root retired
    record.freed = True
    return record.padded_bytes  # the WHOLE padded span, granularity unchanged

  def root_retire(self, root: str) -> None:
    record = self._roots[root]
    if record.task_leases or record.outstanding:
      raise LedgerError(
        "ROOT_RETIRE_WITH_OBLIGATIONS",
        f"root {root} still has {record.task_leases} task leases / {record.outstanding} outstanding jobs",
      )
    record.retired = True
    for name in record.buffers:
      self._l2[name].origin_retired = True

  def root_completion(self, root: str) -> None:
    record = self._roots[root]
    if not record.retired:
      raise LedgerError("ROOT_COMPLETION_BEFORE_RETIRE", f"root {root} completion precedes its retirement")
    record.completion = True

  # -- early input release (per Task / all_tasks / idempotent cleanup) ----
  def pin_l2(self, task: str, buffer: str, *, reads: bool, writes: bool) -> None:
    self._l2[buffer].refs += 1
    self._pins.append(_Pin(task, buffer, reads, writes))

  def _drop_pins(self, pins: list[_Pin]) -> int:
    for pin in pins:
      self._l2[pin.buffer].refs -= 1
      self._pins.remove(pin)
    return len(pins)

  def signal_input_released(self, task: str) -> int:
    pure_read = [p for p in self._pins if p.task == task and p.reads and not p.writes]
    return self._drop_pins(pure_read)

  def grid_all_tasks_signaled(self, tasks: tuple[str, ...]) -> int:
    scope = set(tasks)
    pure_read = [p for p in self._pins if p.task in scope and p.reads and not p.writes]
    return self._drop_pins(pure_read)

  def retire_cleanup(self, task: str) -> int:
    """Idempotent: cleaning a task twice never unpins twice."""
    remaining = [p for p in self._pins if p.task == task]
    return self._drop_pins(remaining)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> dict:
  data = yaml.safe_load(path.read_text())
  if not isinstance(data, dict):
    raise BoundsError("INVALID_FILE", f"{path} is not a YAML mapping")
  return data


def load_constraints(path: Path = CONSTRAINTS) -> dict:
  return load_yaml(path)


def load_contract(path: Path = CONTRACT) -> dict:
  return load_yaml(path)


def validate_all(
  constraints_path: Path = CONSTRAINTS,
  contract_path: Path = CONTRACT,
  request_path: Path = REQUEST,
  protocol_path: Path = PROTOCOL,
) -> dict[str, Any]:
  constraints = load_yaml(constraints_path)
  contract = load_yaml(contract_path)
  request = load_yaml(request_path)
  protocol = load_yaml(protocol_path)
  derived, errors = validate_constraints(constraints)
  errors.extend(validate_contract(contract, derived, request, protocol))
  return {
    "valid": not errors,
    "errors": errors,
    "derived": derived,
    "unresolved": [str(i["name"]) for i in constraints.get("unresolved") or [] if isinstance(i, dict)],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Derive W4 memory budgets and validate the C013 contract.")
  parser.add_argument("--constraints", type=Path, default=CONSTRAINTS)
  parser.add_argument("--contract", type=Path, default=CONTRACT)
  parser.add_argument("--request", type=Path, default=REQUEST)
  parser.add_argument("--protocol", type=Path, default=PROTOCOL)
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()
  report = validate_all(args.constraints, args.contract, args.request, args.protocol)
  text = json.dumps(report, indent=2, sort_keys=True)
  if args.out is not None:
    args.out.write_text(text + "\n")
  print(text)
  if not report["valid"]:
    sys.exit(1)


if __name__ == "__main__":
  main()
