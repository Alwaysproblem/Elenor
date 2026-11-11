"""Validate the C015 W6 contracts (event facility + terminal/reset) and their field request.

What it checks:

1. Every `source: {file, line, needle}` in either contract is READ BACK from the pinned tree or the
   validator (one whole-tree walk, so a nested citation cannot escape).
2. The required contract is held HERE, not in the YAML: states, counters and their check order, kinds,
   the reset phase order, the drain evidence set and its excluded item, terminal states, the retirement
   chain, the settlement classes, gating layers and the debug rules. A synchronized edit that weakens the
   YAML and its guard together cannot pass.
3. Wait-for graphs are BUILT from the contracts and must be acyclic: the reset phase graph, the
   retirement chain, and the DRAIN evidence against the effect of MARK_EVENTS (PENDING==0 is a cycle).
4. Unsourced quantities stay `unresolved` with an owner and a reason; a number there is rejected.
5. The field request names only keys that exist in the contracts and only catalog messages (or new ones).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(LIB.parent))  # derive_w4_bounds: the C013 unit algebra

import w6_contract as w6  # noqa: E402

H = LIB.parents[1]
REPO = LIB.parents[3]
UPSTREAM = H / "upstream" / "coralnpu"
REQUEST = H / "protocol" / "field_requests" / "w6.yaml"
CATALOG = H / "protocol" / "cn_protocol_v1.yaml"
OWNER_TAG = H / "rtl" / "scalar" / "doc" / "rob_tag.yaml"

# ---------------------------------------------------------------------------------------------------
# The required contract. Held here so one edit cannot weaken the YAML and its guard together.
# ---------------------------------------------------------------------------------------------------
REQ_STATES = ("PENDING", "DONE", "ERROR", "TIMEOUT", "RESET")
REQ_TERMINAL = ("DONE", "ERROR", "TIMEOUT", "RESET")
REQ_SCOPES = ("task_private", "tile_shared", "group_l2", "external")
REQ_EVENT_KEY = ("tile_id", "ctx_slot", "ctx_generation", "local_event_id", "event_generation", "sequence")
REQ_LOCAL_KINDS = ("compute_done", "transfer_done", "store_done", "region_done")
REQ_NOT_LOCAL = ("INPUT_RELEASED", "OUTPUT_READY", "TASK_COMPLETION")
REQ_COUNTERS = (
  "stale_sequence",
  "wrong_owner",
  "duplicate_signal",
  "capacity_reject",
  "protocol_error",
  "unknown_event_notify",
)
REQ_SIGNAL_ORDER = (
  "unknown_event_notify",
  "wrong_owner",
  "stale_sequence",
  "wrong_producer",
  "duplicate_signal",
)
REQ_RECLAIM = ("terminal", "consumer_refs_zero", "future_uses_zero")
REQ_WRITERS = {
  "local_event_table": "tile",
  "wait_record": "tile",
  "vme_region_wait_record": "W3",
  "group_event_table": "group",
  "first_fault_latch": "tile",
  "poison_state": "group",
  "max_drain_cycles": "group",
  "reset_phase_order": "group",
  "tile_timers": "none",
}
REQ_FAULT_FIELDS = (
  "code",
  "source_id",
  "severity",
  "task",
  "context",
  "command",
  "event",
  "program",
  "descriptor",
  "group",
  "tile",
  "queue",
  "slot",
  "patch",
  "engine_id",
  "offending_addr",
  "aux0",
  "aux1",
  "cycle",
)
REQ_FAULT_CODES = (
  "INVALID_DESCRIPTOR",
  "UNSUPPORTED_ABI",
  "ADDRESS_FAULT",
  "DMA_TIMEOUT",
  "EVENT_DEPENDENCY_TIMEOUT",
  "STREAM_PROTOCOL_ERROR",
  "ENGINE_INTERNAL_FAULT",
  "RESET_DURING_EXECUTION",
  "L2_CAPACITY_FAULT",
  "INVALID_LAYOUT_FAULT",
  "SLOT_PERMISSION_FAULT",
)
REQ_ARBITRATION = ("source_id", "engine_id", "ctx_slot", "op_seq")
REQ_PHASES = (
  ("FREEZE_ISSUE", ()),
  ("CANCEL_UNISSUED", ("FREEZE_ISSUE",)),
  ("DRAIN", ("CANCEL_UNISSUED",)),
  ("MARK_EVENTS", ("DRAIN",)),
  ("INVALIDATE_RESIDENCY", ("MARK_EVENTS",)),
  ("REBUILD_CREDIT", ("INVALIDATE_RESIDENCY",)),
  ("RESUME_DONE", ("REBUILD_CREDIT",)),
)
REQ_DRAIN_EVIDENCE = (
  "engine_accepted_zero",
  "engine_queue_empty",
  "l1_mshr_empty",
  "transfer_quiescent",
  "no_pending_debug_export",
  "no_unaccepted_phase_or_post_event",
)
REQ_DRAIN_EXCLUDED = ("pending_events_zero",)
REQ_TASK_STATES = ("DONE", "ERROR", "TIMEOUT", "RESET")
REQ_CHAIN = (
  ("exec_drained", ()),
  ("retire_ready", ("exec_drained",)),
  ("retire_authorized", ("retire_ready",)),
  ("local_reclaim", ("retire_authorized",)),
  ("task_completion_stored", ("local_reclaim",)),
  ("task_completion_sent", ("task_completion_stored",)),
  ("parent_pin_cleanup", ("task_completion_sent",)),
  ("parent_r_return", ("task_completion_sent",)),
  ("root_arena_retire", ("parent_pin_cleanup", "parent_r_return")),
  ("root_completion", ("root_arena_retire",)),
)
REQ_FORBIDDEN_EDGES = (
  ("task_completion_stored", "parent_r_return"),
  ("retire_authorized", "parent_r_return"),
  ("task_completion_stored", "parent_pin_cleanup"),
)
REQ_SETTLEMENT = (
  "engine_job",
  "external_write",
  "transfer",
  "local_event",
  "wait_record",
  "vme_region",
  "phase_signal",
  "post_event",
  "retire_ready",
  "task_completion",
  "debug_write",
  "program_reference",
)
REQ_SETTLEMENT_ROLES = ("success", "cancelled", "failed")
REQ_WRAP_COVERS = ("in_flight_completions", "late_acks", "event_queues", "protocol_replay_cache")
REQ_GATING_LAYERS = ("context_park", "unit_idle", "tile_quiescence")
REQ_QUIESCENCE = (
  "all_contexts_parked",
  "lsu_in_flight_empty",
  "mfe_in_flight_empty",
  "rvv_in_flight_empty",
  "vme_in_flight_empty",
  "completion_channel_empty",
)
REQ_EXPORT_FORBIDDEN = ("user_data_bytes", "program_text_bytes", "descriptor_payload")
REQ_DEBUG_PRIV_OPS = ("debug_enable", "debug_quota", "debug_capture_config", "debug_export_target")
REQ_UNRESOLVED = {
  ("facility", ("capacity", "frontier_slots_per_context")): "C018",
  ("facility", ("capacity", "tile_global_slots")): "C019",
  ("facility", ("identity", "widths", "local_event_id")): "C017",
  ("facility", ("identity", "widths", "event_generation")): "C017",
  ("facility", ("identity", "widths", "sequence")): "C017",
  ("terminal", ("reset_endpoint", "max_drain_cycles", "value")): "C019",
  ("terminal", ("debug", "fifo", "depth")): "C019",
  ("terminal", ("first_fault_latch", "fault_codes", "code_width")): "C017",
  ("terminal", ("first_fault_latch", "summary_width")): "C017",
}
BANNED_NUMBER_KEYS = ("address", "bits", "opcode", "width_bits", "encoding")


# ---------------------------------------------------------------------------------------------------
def _read_line(file: str, line: int) -> str | None:
  for root in (REPO, UPSTREAM):
    path = root / file
    if path.is_file():
      lines = path.read_text().splitlines()
      return lines[line - 1] if 0 < line <= len(lines) else None
  return None


def iter_sources(node: Any, where: str = "") -> list[tuple[str, dict]]:
  found: list[tuple[str, dict]] = []
  if isinstance(node, dict):
    if isinstance(node.get("file"), str) and isinstance(node.get("line"), int):
      found.append((where, node))
    for key, value in node.items():
      found.extend(iter_sources(value, f"{where}.{key}"))
  elif isinstance(node, list):
    for index, value in enumerate(node):
      found.extend(iter_sources(value, f"{where}[{index}]"))
  return found


def check_sources(label: str, doc: dict[str, Any], errors: list[str]) -> int:
  count = 0
  for where, src in iter_sources(doc):
    count += 1
    if "needle" not in src:
      errors.append(f"{label}{where}: MISSING_SOURCE (need file/line/needle)")
      continue
    text = _read_line(src["file"], src["line"])
    if text is None:
      errors.append(f"{label}{where}: SOURCE_FILE_MISSING {src['file']}:{src['line']}")
    elif str(src["needle"]) not in text:
      errors.append(
        f"{label}{where}: SOURCE_LINE_MISMATCH {src['file']}:{src['line']} lacks {src['needle']!r}"
      )
  return count


def _eq(errors: list[str], where: str, got: Any, want: Any) -> None:
  if got != want:
    errors.append(f"{where}: {got!r} != required {want!r}")


def _tup(value: Any) -> Any:
  return tuple(value) if isinstance(value, list) else value


def _dig(doc: dict[str, Any], path: tuple[str, ...]) -> Any:
  node: Any = doc
  for key in path:
    node = node[key]
  return node


def _quantity(unit: str) -> Any:
  """A unit-carrying placeholder (value 1) built with the C013 unit algebra."""
  import derive_w4_bounds as bounds

  if unit == "scalar":
    return bounds.Quantity(1.0, bounds.Unit((), 1.0))
  return bounds.Quantity(1.0, bounds.parse_unit(unit))


def check_derivation(label: str, node: dict[str, Any], errors: list[str]) -> None:
  """Evaluate the expression over unit placeholders; the result must carry the declared unit."""
  import derive_w4_bounds as bounds

  try:
    env = {name: _quantity(spec["unit"]) for name, spec in node["inputs"].items()}
    got = bounds.eval_quantity(node["expression"], env)
    want = _quantity(node["result_unit"])
  except bounds.BoundsError as exc:
    errors.append(f"{label}: derivation rejected ({exc})")
    return
  except (KeyError, TypeError) as exc:
    errors.append(f"{label}: derivation is malformed ({type(exc).__name__}: {exc})")
    return
  if got.unit.dims != want.unit.dims:
    errors.append(f"{label}: derivation yields {got.unit.name}, declared {want.unit.name}")
  for name, spec in node["inputs"].items():
    if not str(spec.get("owner", "")).strip():
      errors.append(f"{label}.inputs.{name}: an input with no owner is a guess waiting to happen")
    if "value" in spec and spec["value"] != "unresolved":
      errors.append(f"{label}.inputs.{name}: a numeric value here would be invented")


# ---------------------------------------------------------------------------------------------------
def check_facility(f: dict[str, Any], errors: list[str]) -> None:
  ident = f["identity"]
  _eq(errors, "identity.event_key_fields", _tup(ident["event_key_fields"]), REQ_EVENT_KEY)
  _eq(errors, "identity.scopes", _tup(ident["scopes"]), REQ_SCOPES)
  _eq(errors, "identity.generation_domains_distinct", len(set(ident["generation_domains_distinct"])), 3)
  k = f["event_kinds"]
  _eq(errors, "event_kinds.local_waitable", _tup(k["local_waitable"]), REQ_LOCAL_KINDS)
  names = tuple(x["name"] for x in k["not_local_events"])
  _eq(errors, "event_kinds.not_local_events", names, REQ_NOT_LOCAL)
  if (
    k.get("kind_fixed_at_registration") is not True or k.get("producer_unique_at_registration") is not True
  ):
    errors.append("event_kinds: kind and producer must be fixed at registration")
  if "RESOURCE_RELEASED" not in (k.get("not_waitable") or []):
    errors.append("event_kinds.not_waitable: RESOURCE_RELEASED must not be waitable")
  if set(REQ_NOT_LOCAL) & set(k["local_waitable"]):
    errors.append("event_kinds: a PHASE_SIGNAL/terminal kind is listed as a local event")
  if len(set(k["no_substitution"])) != 5:
    errors.append("event_kinds.no_substitution: the five completions must stay distinct")
  s = f["states"]
  _eq(errors, "states.values", _tup(s["values"]), REQ_STATES)
  _eq(errors, "states.terminal", _tup(s["terminal"]), REQ_TERMINAL)
  if s.get("sticky_terminal") is not True or s.get("no_return_to_pending") is not True:
    errors.append("states: terminal must be sticky and never return to PENDING")
  rc = s["reset_conversion"]
  _eq(errors, "states.reset_conversion.only_from", rc["only_from"], "PENDING")
  _eq(errors, "states.reset_conversion.to", rc["to"], "RESET")
  if rc.get("terminal_unchanged") is not True:
    errors.append("states.reset_conversion: terminal states must stay unchanged")
  _eq(errors, "reclaim.predicate", _tup(f["reclaim"]["predicate"]), REQ_RECLAIM)
  if f["reclaim"]["abandon"]["refused_while"] != "consumer_refs_nonzero":
    errors.append("reclaim.abandon: must be refused while consumers remain")
  sig = f["signal"]
  _eq(errors, "signal.check_order", _tup(sig["check_order"]), REQ_SIGNAL_ORDER)
  if (
    sig.get("at_most_one_state_change_per_signal") is not True
    or sig.get("reject_changes_state") is not False
  ):
    errors.append("signal: a signal changes at most one entry and a reject changes counters only")
  _eq(errors, "counters", tuple(f["counters"]), REQ_COUNTERS)
  for name in REQ_COUNTERS:
    if not str(f["counters"][name].get("increment", "")).strip():
      errors.append(f"counters.{name}: needs an exact increment condition")
  if (
    f["saturation"].get("wrap") != "forbidden" or f["saturation"].get("sticky_bit_per_counter") is not True
  ):
    errors.append("saturation: counters saturate with a sticky bit and never wrap")
  cap = f["capacity"]
  _eq(errors, "capacity.unit", cap["unit"], "entries")
  if cap["group_model_default"].get("not_tile_capacity") is not True:
    errors.append("capacity.group_model_default: the Group default is not a Tile capacity")
  if (
    cap["terminal_reserve_per_context"].get("minimum", 0) < 1
    or cap["terminal_reserve_per_context"].get("separate_from_frontier") is not True
  ):
    errors.append("capacity.terminal_reserve_per_context: needs a separate reserve of at least one entry")
  fb = cap["full_table"]
  _eq(errors, "capacity.full_table.behavior", fb["behavior"], "backpressure_the_context")
  for key in ("fault", "drop_terminal", "release_identity_before_send", "completion_and_credit_blocked"):
    if fb.get(key) is not False:
      errors.append(f"capacity.full_table.{key}: must be false")
  _eq(errors, "capacity.load_time_rejection.code", cap["load_time_rejection"]["code"], "PERMANENT_CAPACITY")
  if cap["load_time_rejection"].get("runtime_fault") is not False:
    errors.append(
      "capacity.load_time_rejection: a static overrun is rejected at load, never faulted at run time"
    )
  _eq(errors, "capacity.static_bound_source", cap["static_bound_source"], "static_analysis")
  check_derivation("capacity.budget_derivation", cap["budget_derivation"], errors)
  _eq(errors, "capacity.budget_derivation.result_unit", cap["budget_derivation"]["result_unit"], "entries")
  _eq(
    errors,
    "capacity.budget_derivation.must_not_exceed",
    cap["budget_derivation"]["must_not_exceed"],
    "tile_global_slots",
  )
  needed = {
    "contexts",
    "frontier_slots_per_context",
    "terminal_reserve_per_context",
    "response_inflight_entries",
  }
  if set(cap["budget_derivation"]["inputs"]) != needed:
    errors.append(
      "capacity.budget_derivation.inputs: frontier, terminal reserve, response bound and Context count"
    )
  w = f["wait"]
  _eq(errors, "wait.kinds", _tup(w["kinds"]), ("WAIT", "WAITALL"))
  if w.get("no_or_mode") is not True or w.get("no_wait_any") is not True:
    errors.append("wait: there is no OR / WAIT_ANY mode")
  _eq(errors, "wait.record_per_context", w["record_per_context"], 1)
  _eq(errors, "wait.record_writer", w["record_writer"], "W6")
  _eq(errors, "wait.wait_operands", w["wait_operands"], 1)
  _eq(errors, "wait.waitall_operands.min", w["waitall_operands"]["min"], 2)
  _eq(errors, "wait.waitall_operands.max", w["waitall_operands"]["max"], "unresolved")
  _eq(errors, "wait.waitall_over_max", w["waitall_over_max"], "decode_reject")
  cr = w["check_register"]
  _eq(errors, "wait.check_register.same_cycle_priority", cr["same_cycle_priority"], "completion_first")
  if cr.get("atomic") is not True or cr.get("park_then_wake_same_cycle") != "forbidden":
    errors.append("wait.check_register: must be atomic and must forbid park-then-wake in one cycle")
  v = w["verdict"]
  _eq(errors, "wait.verdict.waitall_success", v["waitall_success"], "all_operands_done")
  _eq(errors, "wait.verdict.any_terminal_failure", v["any_terminal_failure"], "fail_exit_immediately")
  _eq(errors, "wait.verdict.failure_via_success_wake", v["failure_via_success_wake"], "forbidden")
  if w.get("fault_voids_record") is not True:
    errors.append("wait.fault_voids_record: a faulted Context's record must be voided")
  if set(w["semantics_not_a_fence"]) != {"fence", "store_done", "release", "group_barrier"}:
    errors.append("wait.semantics_not_a_fence: WAIT carries no fence/store_done/release/barrier meaning")
  ps = f["phase_signal"]
  if ps.get("one_instruction_one_message") is not True or ps.get("tile_dedup") != "forbidden":
    errors.append("phase_signal: one accepted instruction is one message and the Tile never dedups")
  if ps.get("physical_tile_in_key") is not False:
    errors.append("phase_signal: the physical Tile is routing only, not part of the aggregation key")
  _eq(
    errors,
    "phase_signal.aggregation_key",
    _tup(ps["aggregation_key"]),
    ("root_launch_generation", "GridInstanceId", "phase", "task"),
  )
  if ps.get("early_release_returns_none") is not True:
    errors.append("phase_signal: an early release returns no local resource")
  pe = f["post_event"]
  if (
    pe.get("never_noop") is not True
    or pe.get("authorization") != "sealed_manifest_grant"
    or pe.get("unauthorized") != "protocol_fault"
  ):
    errors.append(
      "post_event: a sealed-manifest grant is required, unauthorized is a protocol fault, never a no-op"
    )
  _eq(errors, "writers", f["writers"], REQ_WRITERS)


def _graph_from_steps(steps: list[dict[str, Any]]) -> dict[str, list[str]]:
  return {s["step"] if "step" in s else s["name"]: list(s["requires"]) for s in steps}


# An evidence item is only available once the phase that PRODUCES it has run. DRAIN waiting on such an
# item is therefore a wait edge DRAIN -> producer phase; added to the phase graph it exposes any cycle.
EVIDENCE_PRODUCED_BY = {"pending_events_zero": "MARK_EVENTS"}


def drain_wait_graph(phase_graph: dict[str, list[str]], evidence: list[str]) -> dict[str, list[str]]:
  graph = {k: list(v) for k, v in phase_graph.items()}
  for item in evidence:
    producer = EVIDENCE_PRODUCED_BY.get(item)
    if producer is not None:
      graph["DRAIN"].append(producer)
  return graph


def check_terminal(t: dict[str, Any], errors: list[str]) -> None:
  fl = t["first_fault_latch"]
  _eq(errors, "first_fault_latch.per_tile", fl["per_tile"], 1)
  _eq(errors, "first_fault_latch.record_fields", _tup(fl["record_fields"]), REQ_FAULT_FIELDS)
  if fl.get("first_fault_wins") is not True or fl.get("later_fault") != "count_only":
    errors.append("first_fault_latch: first fault wins and later faults only count")
  if fl.get("cleared_by_reset") is not False or fl.get("cleared_by_functional_reset") is not False:
    errors.append("first_fault_latch: neither a reset nor a functional reset may clear the latch")
  _eq(
    errors,
    "first_fault_latch.cleared_only_by",
    _tup(fl["cleared_only_by"]),
    ("FAULT_ACK", "privileged_fault_clear"),
  )
  arb = fl["same_cycle_arbitration"]
  _eq(errors, "first_fault_latch.same_cycle_arbitration.order", _tup(arb["order"]), REQ_ARBITRATION)
  if arb.get("hash_allowed") is not False or arb.get("deterministic") is not True:
    errors.append("first_fault_latch.same_cycle_arbitration: direct fields, no hash, deterministic")
  _eq(errors, "first_fault_latch.fault_codes.names", _tup(fl["fault_codes"]["names"]), REQ_FAULT_CODES)
  _eq(errors, "first_fault_latch.fault_codes.count", fl["fault_codes"]["count"], 11)
  if fl["fault_codes"].get("current_production_codes_are_not_a_hardware_limit") is not True:
    errors.append("first_fault_latch.fault_codes: the six production codes are not a hardware limit")
  if fl["validator_difference"].get("ring_overwrites") is not True:
    errors.append(
      "first_fault_latch.validator_difference: the FaultRing overwrite difference must stay recorded"
    )
  _eq(
    errors,
    "first_fault_latch.attribution_ref.fields",
    _tup(fl["attribution_ref"]["fields"]),
    ("ctx_slot", "ctx_generation", "owner_op_seq"),
  )
  fd = t["fault_domain"]
  _eq(errors, "fault_domain.production_domain", fd["production_domain"], "GROUP")
  if (
    fd.get("never_selects_domain") is not True
    or fd.get("fault_in_one_context_other_continues") != "forbidden"
  ):
    errors.append(
      "fault_domain: the Tile id never selects the domain and no other-Context-continues branch exists"
    )
  _eq(errors, "fault_domain.local_isolation_flt_02a", fd["local_isolation_flt_02a"], "not_implemented")
  re_ = t["reset_endpoint"]
  _eq(errors, "reset_endpoint.authority", re_["authority"], "group")
  if re_.get("tile_advances_unprompted") is not False or re_.get("tile_timers") != "none":
    errors.append("reset_endpoint: the Tile never advances unprompted and owns no timer")
  phases = tuple((p["name"], _tup(p["requires"])) for p in re_["phases"])
  _eq(errors, "reset_endpoint.phases", phases, REQ_PHASES)
  drain = re_["drain"]
  _eq(
    errors, "reset_endpoint.drain.evidence_required", _tup(drain["evidence_required"]), REQ_DRAIN_EVIDENCE
  )
  _eq(
    errors, "reset_endpoint.drain.evidence_excluded", _tup(drain["evidence_excluded"]), REQ_DRAIN_EXCLUDED
  )
  _eq(errors, "reset_endpoint.drain.timeout_means_success", drain["timeout_means_success"], False)
  if set(drain["evidence_required"]) & set(REQ_DRAIN_EXCLUDED):
    errors.append("reset_endpoint.drain: the DRAIN evidence must not include PENDING==0")
  md = re_["max_drain_cycles"]
  _eq(errors, "reset_endpoint.max_drain_cycles.owner", md["owner"], "group")
  _eq(errors, "reset_endpoint.max_drain_cycles.tile_copy", md["tile_copy"], "forbidden")
  me = re_["mark_events"]
  _eq(errors, "reset_endpoint.mark_events.writer", me["writer"], "tile")
  _eq(errors, "reset_endpoint.mark_events.pending_to", me["pending_to"], "RESET")
  if (
    me.get("terminal_unchanged") is not True
    or me.get("tile_marks_error") is not False
    or me.get("group_writes_tile_table") is not False
  ):
    errors.append(
      "reset_endpoint.mark_events: the Tile writes its own table, PENDING->RESET only, never ERROR"
    )
  _eq(errors, "reset_endpoint.poison.owner", re_["poison"]["owner"], "group")
  if re_["invalidate_residency"].get("ack_early") != "forbidden":
    errors.append("reset_endpoint.invalidate_residency: an early acknowledgement is forbidden")
  if re_["rebuild_credit"].get("reuse_before_settlement") != "forbidden":
    errors.append("reset_endpoint.rebuild_credit: no reuse before settlement")
  if re_.get("global_wait_for_acyclic") is not True:
    errors.append("reset_endpoint.global_wait_for_acyclic must be true")
  c = t["cancel"]
  _eq(errors, "cancel.scope", c["scope"], "task")
  if c["cancel_task"].get("accepted_work_dropped_silently") != "forbidden":
    errors.append("cancel.cancel_task: accepted work is never dropped silently")
  if c["classification"]["task_known_but_ctx_generation_conflicts"]["class"] != "protocol_error":
    errors.append("cancel.classification: a live Task with a conflicting generation is a protocol error")
  if c["classification"]["no_live_binding_for_task_key"]["class"] != "stale":
    errors.append("cancel.classification: a Task with no live binding is stale")
  if c.get("generation_as_cancellation") != "forbidden" or c.get("timeout_is_success") is not False:
    errors.append("cancel: a generation is not a cancellation and a timeout is not success")
  ew = c["external_write_settlement"]
  _eq(
    errors,
    "cancel.external_write_settlement.states",
    _tup(ew["states"]),
    ("COMPLETED", "ISOLATED", "FAULTED"),
  )
  iso = ew["isolation_failure"]
  if (
    iso.get("returns_isolated") is not False
    or iso.get("releases_resources") is not False
    or iso.get("effect") != "poison"
  ):
    errors.append(
      "cancel.external_write_settlement.isolation_failure: never ISOLATED, never released, ends in poison"
    )
  tt = t["task_terminal"]
  _eq(errors, "task_terminal.states", _tup(tt["states"]), REQ_TASK_STATES)
  _eq(errors, "task_terminal.success_state", tt["success_state"], "none")
  if (
    tt.get("timeout_from_local_timer") != "forbidden" or tt.get("fault_chain_produces_success") is not False
  ):
    errors.append("task_terminal: no local-timer TIMEOUT and no success after a fault chain")
  if (
    tt["retire_ready"].get("is_task_success") is not False
    or tt["retire_ready"].get("cannot_be_skipped") is not True
  ):
    errors.append("task_terminal.retire_ready: not a success and never skipped")
  chain = tuple((s["step"], _tup(s["requires"])) for s in tt["chain"])
  _eq(errors, "task_terminal.chain", chain, REQ_CHAIN)
  edges = _graph_from_steps(tt["chain"])
  cyc = w6.find_cycle(edges)
  if cyc:
    errors.append(f"task_terminal.chain: circular wait {' -> '.join(cyc)}")
  declared_forbidden = tuple((e["from"], e["to"]) for e in tt["forbidden_edges"])
  _eq(errors, "task_terminal.forbidden_edges", declared_forbidden, REQ_FORBIDDEN_EDGES)
  for src, dst in declared_forbidden:
    if w6.reaches(edges, src, dst):
      errors.append(f"task_terminal.chain: {src} depends on {dst}, the banned wait-for edge")
  if (
    tt.get("python_executes_l1_retire") is not False or tt.get("root_arena_before_root_success") is not True
  ):
    errors.append(
      "task_terminal: Python never executes the L1 retire; root Arena retires before root success"
    )
  if tt["record_storage"].get("identity_released_before_send") is not False:
    errors.append("task_terminal.record_storage: identity is never released before the send")
  # reset graph: DRAIN must not wait on the effect of a later phase
  rg = _graph_from_steps(re_["phases"])
  cyc = w6.find_cycle(rg)
  if cyc:
    errors.append(f"reset_endpoint.phases: circular wait {' -> '.join(cyc)}")
  cyc = w6.find_cycle(drain_wait_graph(rg, drain["evidence_required"]))
  if cyc:
    errors.append(f"reset_endpoint.drain: circular wait {' -> '.join(cyc)}")
  st = t["settlement"]
  _eq(errors, "settlement.classes", tuple(x["name"] for x in st["classes"]), REQ_SETTLEMENT)
  for item in st["classes"]:
    for role in REQ_SETTLEMENT_ROLES:
      if not str(item.get(role, "")).strip():
        errors.append(f"settlement.{item['name']}.{role}: every accepted class needs a settlement owner")
  wq = t["wrap_quiesce"]
  _eq(errors, "wrap_quiesce.reuse_authorized_by", wq["reuse_authorized_by"], "group")
  if wq.get("tile_local_quiescence_sufficient") is not False:
    errors.append("wrap_quiesce: Tile-local quiescence never authorizes reuse")
  _eq(errors, "wrap_quiesce.covers", _tup(wq["covers"]), REQ_WRAP_COVERS)
  if (
    wq.get("identical_bit_pattern_old_copy_may_arrive") != "reuse_forbidden"
    or wq.get("tile_timeout") != "forbidden"
  ):
    errors.append(
      "wrap_quiesce: reuse is forbidden while a same-pattern old copy may arrive; the Tile never times out"
    )
  wd = wq["wire_direction"]
  if wd.get("status") != "decided":
    errors.append(
      "wrap_quiesce.wire_direction: the Tile-asks/Group-answers direction must be recorded as decided"
    )
  _eq(
    errors,
    "wrap_quiesce.wire_direction.request",
    wd.get("request"),
    {"message": "GEN_WRAP_QUIESCE", "dir": "T2G"},
  )
  _eq(
    errors,
    "wrap_quiesce.wire_direction.response",
    wd.get("response"),
    {"message": "QUIESCE_RESP", "dir": "G2T"},
  )
  cg = t["clock_gating"]
  _eq(errors, "clock_gating.layers", _tup(cg["layers"]), REQ_GATING_LAYERS)
  _eq(
    errors, "clock_gating.tile_quiescence.requires", _tup(cg["tile_quiescence"]["requires"]), REQ_QUIESCENCE
  )
  if (
    cg["wfi"].get("enters_tile_gate_condition") is not False
    or cg["wfi"].get("meaning") != "context_park_hint"
  ):
    errors.append("clock_gating.wfi: a park hint only, never part of the Tile gate")
  if (
    cg["context_park"].get("affects_other_contexts") is not False
    or cg["context_park"].get("affects_in_flight_work") is not False
  ):
    errors.append("clock_gating.context_park: must not touch other Contexts or in-flight work")
  if set(cg["context_park"]["must_exclude"]) != {
    "rf_writeback_port",
    "vrf_writeback_port",
    "config_writeback",
  }:
    errors.append("clock_gating.context_park.must_exclude: the write-back ports stay clocked")
  if cg["gating_boundary"].get("excludes_same_cycle_arrival") is not True:
    errors.append("clock_gating.gating_boundary: a same-cycle arrival is excluded from the decision")
  if cg.get("gated_cycles_counted") is not True:
    errors.append("clock_gating: gated cycles must be counted")
  d = t["debug"]
  _eq(errors, "debug.default_mode", d["default_mode"], "off")
  _eq(errors, "debug.after_reset", d["after_reset"], "off")
  if (
    d["off_state"].get("on_chip_history_memory") != "none"
    or d["off_state"].get("hbm_log_ring") != "none"
    or d["off_state"].get("hbm_log_dma") != "none"
  ):
    errors.append("debug.off_state: no history memory, no ring, no DMA while off")
  _eq(errors, "debug.privileged_list_ref.ops", _tup(d["privileged_list_ref"]["ops"]), REQ_DEBUG_PRIV_OPS)
  if d.get("ordinary_kernel_may_enable") is not False:
    errors.append("debug: an ordinary kernel may not enable debug")
  fifo = d["fifo"]
  _eq(errors, "debug.fifo.full_policy", fifo["full_policy"], "drop_new_record")
  _eq(
    errors,
    "debug.fifo.counters",
    _tup(fifo["counters"]),
    ("dropped_count", "overflow_sticky", "window_incomplete"),
  )
  if set(fifo["never_blocks"]) != {
    "issue",
    "response",
    "state_free",
    "wakeup",
    "task_terminal",
    "fault_handling",
  }:
    errors.append("debug.fifo.never_blocks: the full set of functional paths must be listed")
  if fifo.get("functional_message_on_this_path") != "forbidden":
    errors.append("debug.fifo: a functional message never travels this lossy path")
  _eq(errors, "debug.fifo.depth.boundary_test_depth", fifo["depth"]["boundary_test_depth"], 1)
  check_derivation("debug.fifo.depth.derivation", fifo["depth"]["derivation"], errors)
  _eq(
    errors, "debug.fifo.depth.derivation.result_unit", fifo["depth"]["derivation"]["result_unit"], "entries"
  )
  _eq(errors, "debug.export.executed_by", d["export"]["executed_by"], "W4")
  if (
    d["export"].get("on_chip_buffer_growth") != "forbidden"
    or d["export"]["target"].get("may_overwrite_program_or_user_data") is not False
  ):
    errors.append("debug.export: no on-chip growth and no overwrite of program or user data")
  _eq(
    errors,
    "debug.restricted_export_fields.forbidden",
    _tup(d["restricted_export_fields"]["forbidden"]),
    REQ_EXPORT_FORBIDDEN,
  )
  if set(d["restricted_export_fields"]["allowed"]) & set(REQ_EXPORT_FORBIDDEN):
    errors.append("debug.restricted_export_fields: allowed overlaps forbidden")
  cl = d["close_or_cancel_or_reset"]
  if (
    cl.get("clearing_fifo_counts_as_complete") is not False
    or cl.get("timeout_clears") is not False
    or cl.get("stop_new_capture_first") is not True
  ):
    errors.append(
      "debug.close_or_cancel_or_reset: stop capture first; a FIFO clear or timeout completes no write"
    )
  if cl.get("ring_reuse_before_settlement") != "forbidden":
    errors.append("debug.close_or_cancel_or_reset: no ring reuse before settlement")
  rch = t["reliable_channel"]
  if rch.get("no_head_of_line_behind_data") is not True or set(rch["independent_of"]) != {
    "data_channel",
    "debug_channel",
  }:
    errors.append("reliable_channel: control and event never wait on data or debug")


def check_unresolved(f: dict[str, Any], t: dict[str, Any], errors: list[str]) -> None:
  docs = {"facility": f, "terminal": t}
  for (which, path), owner in REQ_UNRESOLVED.items():
    try:
      node = _dig(docs[which], path)
    except (KeyError, TypeError):
      errors.append(f"{which}.{'.'.join(path)}: required unresolved quantity is missing")
      continue
    where = f"{which}.{'.'.join(path)}"
    if isinstance(node, dict) and "value" in node:
      node_value, node_owner = node["value"], node.get("owner")
      carrier = node
    else:
      node_value, node_owner, carrier = node, None, {}
    if node_value != "unresolved":
      errors.append(
        f"{where}: must stay 'unresolved' (a number here would be invented), got {node_value!r}"
      )
    if carrier:
      _eq(errors, f"{where}.owner", node_owner, owner)
      if not str(carrier.get("reason", "")).strip():
        errors.append(f"{where}: unresolved needs a reason")
      for banned in BANNED_NUMBER_KEYS:
        if banned in carrier:
          errors.append(f"{where}.{banned}: invented value (INVENTED_VALUE)")
  # the wait operand ceiling is also unresolved
  mx = f["wait"]["waitall_operands"]
  if mx.get("max") != "unresolved" or mx.get("max_owner") != "C018":
    errors.append("wait.waitall_operands.max: must stay unresolved, owner C018")

  # no numeric literal may sit next to a width or depth key anywhere in the two documents
  def walk(node: Any, where: str) -> None:
    if isinstance(node, dict):
      for key, value in node.items():
        if key in ("width_bits", "bit_position", "csr_address", "opcode"):
          errors.append(f"{where}.{key}: invented encoding")
        walk(value, f"{where}.{key}")
    elif isinstance(node, list):
      for i, value in enumerate(node):
        walk(value, f"{where}[{i}]")

  walk(f, "facility")
  walk(t, "terminal")


def check_owner_tag(t: dict[str, Any], errors: list[str]) -> None:
  ref = t["first_fault_latch"]["attribution_ref"]
  tag = yaml.safe_load(OWNER_TAG.read_text())
  fields = {x["name"] for x in tag["owner_tag"]["fields"]}
  for name in ref["fields"]:
    if name not in fields:
      errors.append(f"first_fault_latch.attribution_ref: {name} is not an OwnerTag field in rob_tag.yaml")


def check_field_request(
  req: dict[str, Any], f: dict[str, Any], t: dict[str, Any], errors: list[str]
) -> None:
  if req.get("schema") != "cn_field_request/v1" or req.get("package") != "W6":
    errors.append("field_request: schema/package must be cn_field_request/v1 + W6")
  catalog_doc = yaml.safe_load(CATALOG.read_text())["messages"]
  catalog = {m["name"] for m in catalog_doc}
  wd = t["wrap_quiesce"]["wire_direction"]
  for side in ("request", "response"):
    entry = wd.get(side) or {}
    cat_dir = next((m["dir"] for m in catalog_doc if m["name"] == entry.get("message")), None)
    if cat_dir != entry.get("dir"):
      errors.append(
        f"wrap_quiesce.wire_direction.{side}: the catalog has {entry.get('message')} as {cat_dir}, "
        f"the contract decided {entry.get('dir')}"
      )
  docs = {"protocol/w6_event_facility.contract.yaml": f, "protocol/w6_terminal_reset.contract.yaml": t}
  groups = (
    ("messages", req.get("messages") or []),
    ("package_fields", req.get("package_fields") or []),
    ("parameters", req.get("parameters") or []),
  )
  for label, items in groups:
    for item in items:
      who = item.get("message") or item.get("name")
      doc = docs.get(item.get("from"))
      if doc is None:
        errors.append(f"field_request.{label}.{who}: from {item.get('from')!r} is not a W6 contract")
        continue
      try:
        _dig(doc, tuple(str(item["key"]).split(".")))
      except (KeyError, TypeError):
        errors.append(
          f"field_request.{label}.{who}: key {item.get('key')!r} does not exist in {item.get('from')}"
        )
      if not str(item.get("why", "")).strip():
        errors.append(f"field_request.{label}.{who}: needs a reason")
      for banned in BANNED_NUMBER_KEYS:
        if banned in item:
          errors.append(f"field_request.{label}.{who}.{banned}: invented value (INVENTED_VALUE)")
  for item in req.get("messages") or []:
    known = item["message"] in catalog
    if known and item.get("new") is True:
      errors.append(f"field_request.messages.{item['message']}: already in the C009 catalog, not new")
    if not known and item.get("new") is not True:
      errors.append(f"field_request.messages.{item['message']}: not in the C009 catalog; mark new: true")
  asked = {m["message"] for m in req.get("messages") or []}
  need = {
    "EVENT_NOTIFY",
    "POST_EVENT",
    "PHASE_SIGNAL",
    "RETIRE_READY",
    "TASK_COMPLETION",
    "GEN_WRAP_QUIESCE",
    "QUIESCE_RESP",
    "CANCEL_TASK",
    "FAULT_REPORT",
    "FAULT_ACK",
    "POISON",
    "RESET_PHASE_REQ",
    "RESET_PHASE_ACK",
  }
  if not need <= asked:
    errors.append(f"field_request.messages: missing {sorted(need - asked)}")
  names = {p.get("name") for p in req.get("package_fields") or []}
  if names != {"frontier_slots_per_context", "waitall_operand_max", "post_event_grant"}:
    errors.append("field_request.package_fields: must request the three W5 manifest fields")
  params = {p.get("name") for p in req.get("parameters") or []}
  if params != {"max_drain_cycles", "debug_fifo_depth"}:
    errors.append("field_request.parameters: must request max_drain_cycles and debug_fifo_depth")


# ---------------------------------------------------------------------------------------------------
def validate_docs(f: Any, t: Any, req: Any) -> dict[str, Any]:
  errors: list[str] = []
  for name, doc in (("facility", f), ("terminal", t), ("field request", req)):
    if not isinstance(doc, dict):
      errors.append(f"{name}: root must be a mapping, got {type(doc).__name__}")
  if errors:
    return {"schema": "cn_w6_contract_report/v1", "valid": False, "errors": errors, "anchors_read_back": 0}

  def guarded(label: str, fn: Any, *args: Any) -> Any:
    try:
      return fn(*args)
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
      errors.append(f"{label}: document is malformed ({type(exc).__name__}: {exc})")
      return None

  anchors = (guarded("sources", check_sources, "facility", f, errors) or 0) + (
    guarded("sources", check_sources, "terminal", t, errors) or 0
  )
  if f.get("status") != "draft" or t.get("status") != "draft":
    errors.append("status: must stay draft (C020 seals, not this card)")
  guarded("facility", check_facility, f, errors)
  guarded("terminal", check_terminal, t, errors)
  guarded("unresolved", check_unresolved, f, t, errors)
  guarded("owner tag", check_owner_tag, t, errors)
  guarded("field request", check_field_request, req, f, t, errors)
  return {
    "schema": "cn_w6_contract_report/v1",
    "valid": not errors,
    "errors": errors,
    "anchors_read_back": anchors,
    "open": [
      "event/generation/sequence widths (C017)",
      "per-Context frontier slots (C018)",
      "tile slot budget, max_drain_cycles, debug FIFO depth (C019)",
      "RTL concurrency and backpressure proof (C033/C034)",
    ],
  }


def validate() -> dict[str, Any]:
  f, t = w6.load_contracts()
  return validate_docs(f, t, yaml.safe_load(REQUEST.read_text()))


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  report = validate()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / "w6_contract_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  if not report["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
