"""Contract-driven consumer for the C010 native owner / retire contracts.

Nothing here hard-codes a field. Layouts, required fields, widths, and which
upstream signals must be accounted for are all derived from the YAML files in
`rtl/scalar/doc/`. Deleting a required field from the YAML therefore changes
what this consumer accepts; it is not a copy of the contract that could drift.

Pieces:
- `eval_width`      evaluate a `width_rule` over named parameters, safely.
- `RecordSpec`      ordered, width-resolved field layout built from a YAML group.
- `RecordSpec.pack / decode`   bit-exact encode/decode with coded errors for
  missing, unknown, malformed and out-of-range fields.
- `ConfigParams`    the numeric environment (context_count, rob capacity,
  xlen, ...) for one configuration; C1 and C2 are two instances of it.
- `RobConsumer`     a ROB whose tag layout and match rule come from rob_tag.yaml,
  so a result for a reused slot is decoded as stale rather than as the new op.
- `upstream_trace_signals` / `check_trace_coverage`   read the signal set from
  RvviTrace.scala and demand every signal is carried or excluded with a reason.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

UNRESOLVED = "unresolved"


class ContractError(Exception):
  """Coded error; `code` is stable so tests can pin it."""

  def __init__(self, code: str, detail: str) -> None:
    super().__init__(f"{code}: {detail}")
    self.code = code


# --------------------------------------------------------------------------
# width rules
# --------------------------------------------------------------------------
def log2_ceil(n: int) -> int:
  if n < 1:
    raise ContractError("WIDTH_RULE_DOMAIN", f"log2Ceil needs n >= 1, got {n}")
  return (n - 1).bit_length()


_ALLOWED_FUNCS: dict[str, Callable[..., int]] = {"log2Ceil": log2_ceil, "max": max, "min": min}
_ALLOWED_NODES = (
  ast.Expression,
  ast.BinOp,
  ast.UnaryOp,
  ast.Call,
  ast.Name,
  ast.Load,
  ast.Constant,
  ast.Add,
  ast.Sub,
  ast.Mult,
  ast.FloorDiv,
  ast.USub,
)


def eval_width(rule: str, env: dict[str, int]) -> int:
  """Evaluate a width rule such as `log2Ceil(retirementBufferSize)` over `env`.

  Only integer arithmetic, `log2Ceil`, `max`, `min`, and names found in `env`
  are allowed; anything else (and the literal word `unresolved`) is refused.
  """
  rule = str(rule).strip()
  if rule == UNRESOLVED:
    raise ContractError("WIDTH_UNRESOLVED", "no sourced bound; no width may be assumed")
  try:
    tree = ast.parse(rule, mode="eval")
  except SyntaxError as exc:
    raise ContractError("WIDTH_RULE_SYNTAX", f"{rule!r}: {exc.msg}") from exc
  for node in ast.walk(tree):
    if not isinstance(node, _ALLOWED_NODES):
      raise ContractError("WIDTH_RULE_FORBIDDEN", f"{rule!r} uses {type(node).__name__}")

  def ev(node: ast.AST) -> int:
    if isinstance(node, ast.Expression):
      return ev(node.body)
    if isinstance(node, ast.Constant):
      if isinstance(node.value, bool) or not isinstance(node.value, int):
        raise ContractError("WIDTH_RULE_FORBIDDEN", f"{rule!r}: non-integer literal")
      return node.value
    if isinstance(node, ast.Name):
      if node.id not in env:
        raise ContractError("WIDTH_PARAM_UNKNOWN", f"{rule!r}: no parameter {node.id!r}")
      return int(env[node.id])
    if isinstance(node, ast.UnaryOp):
      return -ev(node.operand)
    if isinstance(node, ast.BinOp):
      a, b = ev(node.left), ev(node.right)
      ops = {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b, ast.FloorDiv: a // b if b else 0}
      return ops[type(node.op)]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
      return int(_ALLOWED_FUNCS[node.func.id](*[ev(a) for a in node.args]))
    raise ContractError("WIDTH_RULE_FORBIDDEN", f"{rule!r}: unsupported construct")

  width = ev(tree)
  if width < 1:
    raise ContractError("ZERO_WIDTH", f"{rule!r} evaluates to {width}; a field is never zero-width")
  return width


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfigParams:
  """Numeric environment for one configuration (C1 and C2 are two instances)."""

  context_count: int
  values: dict[str, int]
  fixture_widths: dict[str, int] | None = None  # explicit test-fixture widths; never a contract value

  def env(self) -> dict[str, int]:
    if self.context_count < 1:
      raise ContractError("CONTEXT_COUNT", f"context_count must be >= 1, got {self.context_count}")
    ctx_bits = max(1, log2_ceil(self.context_count))
    return {**self.values, "context_count": self.context_count, "ctx_bits": ctx_bits}

  def width_override(self, name: str) -> int | None:
    return (self.fixture_widths or {}).get(name)


def split_resolvable(fields: list[dict], env: dict[str, int]) -> tuple[list[dict], list[str]]:
  """Separate fields whose width can be derived now from those that are `unresolved`."""
  ok: list[dict] = []
  open_: list[str] = []
  for f in fields:
    if str(f["width_rule"]).strip() == UNRESOLVED:
      open_.append(f["name"])
    else:
      eval_width(f["width_rule"], env)  # raises on a bad rule rather than hiding it
      ok.append(f)
  return ok, open_


# --------------------------------------------------------------------------
# record layout
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
  name: str
  width: int
  required: bool
  offset: int


class RecordSpec:
  """Ordered, width-resolved layout built from a YAML field list."""

  def __init__(self, label: str, fields: list[dict], config: ConfigParams) -> None:
    self.label = label
    env = config.env()
    laid_out: list[FieldSpec] = []
    offset = 0
    seen: set[str] = set()
    for f in fields:
      name = f["name"]
      if name in seen:
        raise ContractError("DUPLICATE_FIELD", f"{label}.{name}")
      seen.add(name)
      override = config.width_override(name)
      width = override if override is not None else eval_width(f["width_rule"], env)
      laid_out.append(FieldSpec(name, width, bool(f.get("required", False)), offset))
      offset += width
    self.fields = laid_out
    self.bits = offset
    self._by_name = {f.name: f for f in laid_out}

  @property
  def names(self) -> list[str]:
    return [f.name for f in self.fields]

  def pack(self, values: dict[str, int]) -> int:
    """Encode a record. Refuses missing required, unknown, and out-of-range fields."""
    unknown = sorted(set(values) - set(self._by_name))
    if unknown:
      raise ContractError("FIELD_UNKNOWN", f"{self.label}: {unknown}")
    missing = [f.name for f in self.fields if f.required and f.name not in values]
    if missing:
      raise ContractError("FIELD_MISSING", f"{self.label}: required {missing}")
    word = 0
    for f in self.fields:
      if f.name not in values:
        continue
      v = values[f.name]
      if isinstance(v, bool) or not isinstance(v, int):
        raise ContractError("FIELD_TYPE", f"{self.label}.{f.name}: {v!r} is not an integer")
      if v < 0 or v >= 1 << f.width:
        raise ContractError("FIELD_OUT_OF_RANGE", f"{self.label}.{f.name}: {v} does not fit {f.width} bits")
      word |= v << f.offset
    return word

  def decode(self, word: int, present: set[str] | None = None) -> dict[str, int]:
    """Decode a packed word. `present` limits which optional fields are returned."""
    if word < 0 or word >> self.bits:
      raise ContractError("RECORD_OUT_OF_BOUNDS", f"{self.label}: word exceeds {self.bits} bits")
    out: dict[str, int] = {}
    for f in self.fields:
      if f.required or present is None or f.name in present:
        out[f.name] = (word >> f.offset) & ((1 << f.width) - 1)
    return out


# --------------------------------------------------------------------------
# consumer: ROB driven by rob_tag.yaml
# --------------------------------------------------------------------------
_MISMATCH_CODES = {
  "ctx_slot": "WRONG_CONTEXT",
  "slot_generation": "STALE_SLOT_GENERATION",
  "ctx_generation": "STALE_CTX_GENERATION",
  "owner_op_seq": "OP_SEQ_MISMATCH",
}


class RobConsumer:
  """Per-Context ROB whose keys, widths and match rule come from rob_tag.yaml.

  A completion carries TWO packed words: the ROB tag and the owner tag. Both
  are decoded with `RecordSpec` built from the YAML, and every field marked
  `required` in either is compared against the live occupant. So removing
  `slot_generation` from the YAML removes the ABA protection from this class,
  and adding a required field makes `allocate` demand it; nothing is copied
  into the class.

  Fields the consumer manages itself (`ctx_slot`, `rob_slot`, `slot_generation`,
  `ctx_generation`) are filled in at allocation; any other required field
  (`owner_op_seq`, `subop_id`, ...) must be supplied by the caller.
  """

  _MANAGED = ("ctx_slot", "rob_slot", "slot_generation", "ctx_generation")

  def __init__(self, rob_yaml: dict, config: ConfigParams) -> None:
    self.config = config
    self.capacity = int(config.env()["retirementBufferSize"])
    self.tag_spec = RecordSpec("rob_tag", rob_yaml["rob_tag"]["fields"], config)
    self.owner_spec = RecordSpec("owner_tag", rob_yaml["owner_tag"]["fields"], config)
    n = config.context_count
    self._live: list[dict[int, dict[str, int]]] = [{} for _ in range(n)]
    self._head = [0] * n
    self._tail = [0] * n
    self._count = [0] * n
    self._slot_gen = [[0] * self.capacity for _ in range(n)]
    self._ctx_gen = [1] * n

  def _check_ctx(self, ctx: int) -> None:
    if not 0 <= ctx < self.config.context_count:
      raise ContractError("CTX_OUT_OF_RANGE", f"ctx_slot {ctx} not in 0..{self.config.context_count - 1}")

  def allocate(self, ctx: int, **owner_fields: int) -> tuple[int, int]:
    """Allocate in `ctx`; returns (rob_tag_word, owner_tag_word)."""
    self._check_ctx(ctx)
    unknown = sorted(set(owner_fields) - set(self.tag_spec.names) - set(self.owner_spec.names))
    if unknown:
      raise ContractError(
        "FIELD_UNKNOWN", f"allocate: {unknown} is in neither the rob tag nor the owner tag"
      )
    managed = sorted(set(owner_fields) & set(self._MANAGED))
    if managed:
      raise ContractError("FIELD_MANAGED", f"allocate: {managed} are assigned by the ROB, not the caller")
    if self._count[ctx] == self.capacity:
      raise ContractError("ROB_FULL", f"ctx {ctx} partition of {self.capacity} is full")
    slot = self._tail[ctx]
    values = {
      "ctx_slot": ctx,
      "rob_slot": slot,
      "slot_generation": self._slot_gen[ctx][slot],
      "ctx_generation": self._ctx_gen[ctx],
      **owner_fields,
    }
    # Pack BEFORE touching state: a generation that no longer fits its field is
    # refused here rather than wrapping and aliasing an old tag.
    tag_word = self.tag_spec.pack({k: v for k, v in values.items() if k in self.tag_spec.names})
    owner_word = self.owner_spec.pack({k: v for k, v in values.items() if k in self.owner_spec.names})
    self._live[ctx][slot] = {**values, "completed": 0}
    self._tail[ctx] = (slot + 1) % self.capacity
    self._count[ctx] += 1
    return tag_word, owner_word

  def complete(self, tag_word: int, owner_word: int) -> None:
    tag = self.tag_spec.decode(tag_word)
    owner = self.owner_spec.decode(owner_word)
    ctx = tag["ctx_slot"]
    self._check_ctx(ctx)
    slot = tag["rob_slot"]
    if not 0 <= slot < self.capacity:
      raise ContractError("SLOT_OUT_OF_RANGE", f"rob_slot {slot} not in 0..{self.capacity - 1}")
    entry = self._live[ctx].get(slot)
    if entry is None:
      raise ContractError("SLOT_NOT_LIVE", f"ctx {ctx} slot {slot}")
    for source in (tag, owner):
      for name, value in source.items():
        field = self.tag_spec if source is tag else self.owner_spec
        if not field._by_name[name].required or name not in entry:
          continue
        if value != entry[name]:
          code = _MISMATCH_CODES.get(name, "TAG_MISMATCH")
          raise ContractError(code, f"{name}: completion {value} != live {entry[name]}")
    if entry["completed"]:
      raise ContractError("DUPLICATE_COMPLETION", f"ctx {ctx} slot {slot}")
    entry["completed"] = 1

  def retire(self, ctx: int) -> int | None:
    """Retire the head in order; returns its owner_op_seq (or slot) when it retires."""
    self._check_ctx(ctx)
    if self._count[ctx] == 0:
      return None
    head = self._head[ctx]
    entry = self._live[ctx].get(head)
    if entry is None or not entry["completed"]:
      return None
    del self._live[ctx][head]
    self._slot_gen[ctx][head] += 1
    self._head[ctx] = (head + 1) % self.capacity
    self._count[ctx] -= 1
    return entry.get("owner_op_seq", head)

  def live(self, ctx: int) -> int:
    self._check_ctx(ctx)
    return self._count[ctx]

  def reuse_context(self, ctx: int, *, completion_consumed: bool, reuse_rule: list[str]) -> int:
    """Hand a slot to a new Task. The preconditions are the YAML `reuse_rule` list."""
    self._check_ctx(ctx)
    missing = []
    if "old obligations closed" in reuse_rule and self._count[ctx] != 0:
      missing.append("old obligations still open")
    if "TASK_COMPLETION reliably consumed" in reuse_rule and not completion_consumed:
      missing.append("TASK_COMPLETION not consumed")
    if missing:
      raise ContractError("REUSE_BLOCKED", "; ".join(missing))
    if "identity isolated (ctx_generation advanced)" in reuse_rule:
      self._ctx_gen[ctx] += 1
    return self._ctx_gen[ctx]


# --------------------------------------------------------------------------
# upstream trace coverage
# --------------------------------------------------------------------------
_SIGNAL = re.compile(r"^\s*val\s+(\w+)\s*=\s*Input\(")


def upstream_trace_signals(trace_scala: Path, suffix: str = "_i") -> list[str]:
  """Signal base names of the RVVI blackbox IO, read from the source file."""
  names: list[str] = []
  in_class = False
  for line in trace_scala.read_text().splitlines():
    if "class RvviTraceBlackBox" in line:
      in_class = True
      continue
    if in_class and line.startswith("class ") and "RvviTraceBlackBox" not in line:
      break
    match = _SIGNAL.match(line) if in_class else None
    if match and match.group(1).endswith(suffix):
      names.append(match.group(1)[: -len(suffix)])
  return names


def check_trace_coverage(retire_yaml: dict, trace_scala: Path) -> list[str]:
  """Every upstream signal must be carried (and present) or excluded with a reason."""
  errors: list[str] = []
  signals = upstream_trace_signals(trace_scala, retire_yaml["upstream_trace"].get("suffix", "_i"))
  if not signals:
    return [f"no RVVI signals could be read from {trace_scala}"]
  carried = {f["upstream_signal"]: f for f in retire_yaml["fields"] if f.get("upstream_signal")}
  excluded = {e["upstream_signal"]: e for e in retire_yaml["excluded_upstream"]}
  for sig in signals:
    in_c, in_e = sig in carried, sig in excluded
    if in_c and in_e:
      errors.append(f"upstream signal {sig!r} is both carried and excluded")
    elif not in_c and not in_e:
      errors.append(f"upstream signal {sig!r} is neither carried nor excluded with a reason")
    elif in_e and not excluded[sig].get("reason"):
      errors.append(f"upstream signal {sig!r} is excluded without a reason")
  for sig in list(carried) + list(excluded):
    if sig not in signals:
      errors.append(f"contract names upstream signal {sig!r} that RvviTrace.scala does not have")
  required_in_trace = {s for s in signals if s in carried and carried[s].get("required")}
  if not required_in_trace:
    errors.append("no required upstream signal is carried; the oracle would have nothing to compare")
  return errors
