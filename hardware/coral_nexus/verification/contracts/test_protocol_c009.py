"""C009 boundary identity / budget / CSR contract tests.

Each C009 acceptance clause has a case that must be ACCEPTED and one that must
be REJECTED with a specific error:

- same-PC two-Context and same-slot different-generation stay distinguishable
- HOLD and FINAL of one txn are distinct legal stages; a repeated stage is a dup
- a stale generation is rejected; ACK must name the request it answers
- CSR conflict is rejected and names the upstream holder
- widths: cycles-plus-entries unit mixing is rejected; a single-Context
  domain never yields a zero-width field
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
sys.path.insert(0, str(H / "scripts/lib"))

import validate_protocol as vp  # noqa: E402

PROTO = H / "protocol"
GOOD_UNITS = {"max_in_flight": "count", "rate": "msg_per_tick", "lifetime": "tick"}


def _units(**override: str) -> dict[str, str]:
  return {**GOOD_UNITS, **override}


@pytest.fixture
def data() -> dict:
  return copy.deepcopy(vp.load_all(PROTO))


def _errors(fn, *args) -> list[str]:
  errors: list[str] = []
  fn(*args, errors)
  return errors


def test_committed_drafts_validate() -> None:
  report = vp.validate(PROTO)
  assert report["valid"], report["errors"]
  assert report["messages"] >= 30


def test_identity_round_trip_distinguishes_context_and_generation(data: dict) -> None:
  """Built from the YAML field list: same PC / same slot are still different keys."""
  proto = data["protocol"]

  def ctx(tile: int, slot: int, gen: int) -> tuple:
    return vp.make_identity(
      proto, "context_key", {"tile_id": tile, "ctx_slot": slot, "ctx_generation": gen}
    )

  assert ctx(0, 0, 1) != ctx(0, 1, 1), "two Contexts running the same PC must not merge"
  assert ctx(0, 0, 1) != ctx(0, 0, 2), "reused slot must differ by ctx_generation"
  assert ctx(0, 0, 1) == ctx(0, 0, 1)
  assert dict(ctx(3, 2, 9)) == {"tile_id": 3, "ctx_slot": 2, "ctx_generation": 9}


def test_task_key_requires_equal_launch_and_root_generation(data: dict) -> None:
  proto = data["protocol"]
  base = {
    "launch_generation": 4,
    "root_launch_generation": 4,
    "dispatch_ordinal": 0,
    "task_id": 1,
    "context_handle": 0,
  }
  assert dict(vp.make_identity(proto, "task_key", base))["task_id"] == 1
  with pytest.raises(vp.ProtocolError) as exc:
    vp.make_identity(proto, "task_key", {**base, "root_launch_generation": 5})
  assert exc.value.code == "IDENTITY_EQUALITY"


def test_identity_rejects_missing_and_unknown_fields(data: dict) -> None:
  proto = data["protocol"]
  with pytest.raises(vp.ProtocolError) as missing:
    vp.make_identity(proto, "context_key", {"tile_id": 0, "ctx_slot": 0})
  assert missing.value.code == "IDENTITY_FIELD_MISSING"
  with pytest.raises(vp.ProtocolError) as unknown:
    vp.make_identity(
      proto, "context_key", {"tile_id": 0, "ctx_slot": 0, "ctx_generation": 0, "pc": 0x100}
    )
  assert unknown.value.code == "IDENTITY_FIELD_UNKNOWN"


def test_pc_in_identity_is_rejected(data: dict) -> None:
  data["protocol"]["identity_domains"]["op"]["fields"].append({"name": "pc", "owner": "W2"})
  errors = _errors(vp.check_identity, data["protocol"])
  assert any("PC field" in e for e in errors)


def test_root_launch_generation_must_equal_launch_generation(data: dict) -> None:
  fields = data["protocol"]["identity_domains"]["task_key"]["fields"]
  next(f for f in fields if f["name"] == "root_launch_generation").pop("equals")
  assert any("must be declared equal" in e for e in _errors(vp.check_identity, data["protocol"]))


def test_missing_ctx_generation_is_rejected(data: dict) -> None:
  fields = data["protocol"]["identity_domains"]["context_key"]["fields"]
  fields[:] = [f for f in fields if f["name"] != "ctx_generation"]
  assert any("ctx_generation" in e for e in _errors(vp.check_identity, data["protocol"]))


def test_op_seq_as_transport_counter_is_rejected(data: dict) -> None:
  transport = data["protocol"]["identity_domains"]["transport"]["fields"]
  transport.append({"name": "op_seq", "owner": "W2"})
  assert any("transport counter" in e for e in _errors(vp.check_identity, data["protocol"]))


def test_hold_and_final_are_distinct_legal_stages(data: dict) -> None:
  assert _errors(vp.check_dedup_rules, data["protocol"]) == []
  by_name = {m["name"]: m for m in data["protocol"]["messages"]}
  # Same txn_id, different message_type: both legal -> keys differ.
  hold = ("txn-7", "TASK_COMMIT")
  final = ("txn-7", "TASK_COMMIT_FINAL")
  assert hold != final
  assert by_name["TASK_COMMIT"]["stage"] == "HOLD"
  assert by_name["TASK_COMMIT_FINAL"]["stage"] == "FINAL"


def test_collapsing_hold_and_final_is_rejected(data: dict) -> None:
  for msg in data["protocol"]["messages"]:
    if msg["name"] == "TASK_COMMIT_FINAL":
      msg["stage"] = "HOLD"
  assert any("HOLD/FINAL" in e for e in _errors(vp.check_dedup_rules, data["protocol"]))


def test_semantic_key_must_include_message_type(data: dict) -> None:
  data["protocol"]["dedup"]["semantic_key"] = ["txn_id"]
  assert any("semantic key" in e for e in _errors(vp.check_dedup_rules, data["protocol"]))


def test_run_begin_must_not_bump_program_epoch(data: dict) -> None:
  for msg in data["protocol"]["messages"]:
    if msg["name"] == "RUN_BEGIN":
      msg["key"] = ["run_generation", "program_epoch"]
  assert any("RUN_BEGIN" in e for e in _errors(vp.check_dedup_rules, data["protocol"]))


def test_stale_generation_is_rejected_and_current_accepted() -> None:
  vp.check_stale_generation(live_generation=3, message_generation=3)
  vp.check_stale_generation(live_generation=3, message_generation=4)
  with pytest.raises(vp.ProtocolError) as exc:
    vp.check_stale_generation(live_generation=3, message_generation=2)
  assert exc.value.code == "STALE_GENERATION"
  assert "ctx_generation 2 < live 3" in str(exc.value)


def test_message_without_source_is_rejected(data: dict) -> None:
  del data["protocol"]["messages"][0]["source"]
  errors = _errors(vp.check_structure, data["protocol"])
  assert any("missing required field 'source'" in e for e in errors)


def test_stream_subqueue_must_live_on_memory(data: dict) -> None:
  for msg in data["protocol"]["messages"]:
    if msg["name"] == "STREAM_POP":
      msg["channel"] = "CTRL"
  assert any("subqueue" in e for e in _errors(vp.check_structure, data["protocol"]))


def test_csr_address_assignment_is_rejected_and_names_holder(data: dict) -> None:
  data["csr"]["declared"][0]["address"] = 0x7C0
  errors = _errors(vp.check_csr, data["csr"], data["upstream_csr"])
  assert any("MCONTEXT0" in e and "csr_allocator" in e for e in errors), errors


def test_csr_name_colliding_with_upstream_is_rejected(data: dict) -> None:
  data["csr"]["declared"].append(
    {"name": "nexus.mpc", "owner": "W2", "scope": "per_context", "access": "unspecified", "source": "x"}
  )
  errors = _errors(vp.check_csr, data["csr"], data["upstream_csr"])
  assert any("collides with an upstream" in e for e in errors)


def test_upstream_table_is_parsed_from_pin_not_handwritten() -> None:
  table = json.loads((PROTO / "upstream_csr_occupied.json").read_text())
  lock = json.loads((H / "env/sources.lock").read_text())
  pinned = next(s["commit"] for s in lock["sources"] if s["name"] == "coralnpu")
  assert table["source"]["commit"] == pinned
  by_name = {e["name"]: e["address"] for e in table["occupied"]}
  assert by_name["MCONTEXT0"] == 0x7C0 and by_name["MCONTEXT7"] == 0x7C7
  assert by_name["MPC"] == 0x7E0 and by_name["MSP"] == 0x7E1 and by_name["MTYPE"] == 0xC23


def test_extractor_refuses_a_head_that_differs_from_the_lock(tmp_path: Path) -> None:
  lock = json.loads((H / "env/sources.lock").read_text())
  for source in lock["sources"]:
    if source["name"] == "coralnpu":
      source["commit"] = "0" * 40
  bad = tmp_path / "sources.lock"
  bad.write_text(json.dumps(lock))
  cmd = [sys.executable, str(PROTO / "extract_upstream_csr.py"), "--lock", str(bad)]
  proc = subprocess.run(
    [*cmd, "--out", str(tmp_path / "o.json")],
    capture_output=True,
    text=True,
  )
  assert proc.returncode != 0 and "locked commit" in (proc.stderr + proc.stdout)


def test_width_unit_mixing_cycles_with_entries_is_rejected() -> None:
  with pytest.raises(ValueError, match="lifetime must be ticks multiplied by the rate"):
    vp.derive_width("bad", 16, 1, 100, _units(lifetime="cycle"))
  with pytest.raises(ValueError, match="rate must be msg_per_tick"):
    vp.derive_width("bad", 16, 1, 100, _units(rate="cycle"))


def test_width_covers_window_and_single_context_is_never_zero_width() -> None:
  wide = vp.derive_width("w", 16, 1, 100, GOOD_UNITS)
  assert 2**wide.bits > wide.window == 116
  single = vp.derive_width("c1", 1, 0, 0, GOOD_UNITS)
  assert single.bits >= 1, "one Context must still get a non-zero-width field"


def test_underivable_domain_must_explain_itself(data: dict) -> None:
  dom = next(d for d in data["lifetimes"]["domains"] if d["name"] == "run_generation")
  assert dom["derivable"] is False
  del dom["reason"]
  _, errors = vp.derive_all(data["budgets"], data["lifetimes"])
  assert any("gives no reason" in e for e in errors)


def test_underivable_domain_must_list_the_evidence_it_needs(data: dict) -> None:
  dom = next(d for d in data["lifetimes"]["domains"] if d["name"] == "txn_id")
  del dom["needs"]
  _, errors = vp.derive_all(data["budgets"], data["lifetimes"])
  assert any("lists no `needs` evidence" in e for e in errors)


def test_budget_without_file_line_source_is_rejected(data: dict) -> None:
  entry = data["budgets"]["ledgers"]["transaction_capacity"]["entries"][0]
  del entry["source"]["line"]
  assert any("missing source file:line" in e for e in _errors(vp.check_budgets, data["budgets"]))


def _msg(seq: int, mtype: str, txn: int = 7, **extra: object) -> dict:
  return {
    "tile": 0,
    "channel": "CTRL",
    "dir": "G2T",
    "message_seq": seq,
    "message_type": mtype,
    "txn_id": txn,
    **extra,
  }


def test_hold_then_final_both_apply_exactly_once(data: dict) -> None:
  tracker = vp.DedupTracker(data["protocol"])
  tracker.deliver(_msg(0, "TASK_COMMIT"))
  tracker.deliver(_msg(1, "TASK_COMMIT_FINAL"))
  assert tracker.effects == 2, "same txn HOLD and FINAL are distinct legal stages"


def test_repeated_stage_is_a_duplicate_not_a_second_effect(data: dict) -> None:
  tracker = vp.DedupTracker(data["protocol"])
  tracker.deliver(_msg(0, "TASK_COMMIT"))
  with pytest.raises(vp.ProtocolError) as exc:
    tracker.deliver(_msg(1, "TASK_COMMIT"))
  assert exc.value.code == "DUPLICATE_STAGE"
  assert tracker.effects == 1


def test_transport_replay_does_not_repeat_the_side_effect(data: dict) -> None:
  tracker = vp.DedupTracker(data["protocol"])
  tracker.deliver(_msg(0, "TASK_COMMIT"))
  tracker.deliver(_msg(1, "TASK_COMMIT_FINAL"))
  before = tracker.effects
  with pytest.raises(vp.ProtocolError) as exc:
    tracker.deliver(_msg(1, "TASK_COMMIT_FINAL"))
  assert exc.value.code == "TRANSPORT_REPLAY"
  assert tracker.effects == before, "a replay must not apply its effect again"


def test_transport_gap_is_incomplete_not_a_tile_verdict(data: dict) -> None:
  tracker = vp.DedupTracker(data["protocol"])
  tracker.deliver(_msg(0, "TASK_COMMIT"))
  with pytest.raises(vp.ProtocolError) as exc:
    tracker.deliver(_msg(5, "TASK_COMMIT_FINAL"))
  assert exc.value.code == "INCOMPLETE"
  assert data["protocol"]["dedup"]["transport_replay_outcome"] == "INCOMPLETE"


def test_ack_must_reference_a_seen_request(data: dict) -> None:
  tracker = vp.DedupTracker(data["protocol"])
  tracker.deliver(_msg(0, "TASK_COMMIT"))
  ack = _msg(0, "ACK", ack_for_message_type="TASK_COMMIT", channel="CTRL", dir="T2G")
  tracker.deliver(ack)
  stray = _msg(1, "ACK", ack_for_message_type="TASK_COMMIT_FINAL", channel="CTRL", dir="T2G")
  with pytest.raises(vp.ProtocolError) as exc:
    tracker.deliver(stray)
  assert exc.value.code == "ACK_MISMATCH"


def test_message_seq_is_per_tile_channel_direction(data: dict) -> None:
  tracker = vp.DedupTracker(data["protocol"])
  tracker.deliver(_msg(0, "TASK_COMMIT", txn=1))
  other_tile = _msg(0, "TASK_COMMIT", txn=2)
  other_tile["tile"] = 1
  tracker.deliver(other_tile)  # independent counter: seq 0 is valid again
  assert tracker.effects == 2


def test_single_context_gets_a_nonzero_width_index() -> None:
  one = vp.derive_index_width("ctx_slot", 1)
  assert one.bits >= 1, "single Context must not produce a zero-width field"
  assert vp.derive_index_width("ctx_slot", 2).bits == 1
  assert vp.derive_index_width("ctx_slot", 8).bits == 3
  with pytest.raises(ValueError, match="positive integer"):
    vp.derive_index_width("ctx_slot", 0)


def test_stream_as_sixth_channel_is_rejected(data: dict) -> None:
  data["protocol"]["channels"].append({"name": "STREAM", "directions": ["T2G"]})
  errors = _errors(vp.check_phases_and_channels, data["protocol"])
  assert any("STREAM must be a MEMORY sub-queue" in e for e in errors)


def test_phase_rule_removal_is_rejected(data: dict) -> None:
  data["protocol"]["phases"]["rules"] = [
    r for r in data["protocol"]["phases"]["rules"] if r["phase"] != "P1"
  ]
  errors = _errors(vp.check_phases_and_channels, data["protocol"])
  assert any("no rule recorded for P1" in e for e in errors)


# --- fail-closed width derivation -------------------------------------------


def _derivable_domain(name: str = "fixture_dom", **override: dict) -> dict:
  dom: dict = {
    "name": name,
    "max_in_flight": {"ref": "channel_budgets.transaction_capacity.dispatch_capacity"},
    "rate": {"value": 2, "unit": "msg_per_tick", "bounds": name, "source": {"file": "t.txt", "line": 1}},
    "lifetime_ticks": {"value": 10, "unit": "tick", "bounds": name, "source": {"file": "t.txt", "line": 2}},
  }
  dom.update(override)
  return dom


def test_committed_lifetimes_emit_no_width_without_evidence(data: dict) -> None:
  """Every committed domain lacks rate/lifetime evidence -> nothing is derived."""
  widths, errors = vp.derive_all(data["budgets"], data["lifetimes"])
  assert errors == []
  assert widths == [], "no width may be emitted for an unevidenced domain"
  assert all(d["derivable"] is False for d in data["lifetimes"]["domains"])


def test_reset_drain_threshold_is_not_accepted_as_a_lifetime(data: dict) -> None:
  """The old 100-tick reset_domain.py:89 value must not derive a txn width."""
  dom = _derivable_domain(
    "txn_id",
    lifetime_ticks={
      "value": 100,
      "unit": "tick",
      "bounds": "reset_drain_cancel_threshold",
      "source": {"file": "pipeline_validator/runtime/reset_domain.py", "line": 89},
    },
  )
  data["lifetimes"]["domains"] = [dom]
  widths, errors = vp.derive_all(data["budgets"], data["lifetimes"])
  assert widths == []
  assert any("bounds 'reset_drain_cancel_threshold', not 'txn_id'" in e for e in errors)


def test_task_commit_rate_is_not_accepted_as_a_message_rate(data: dict) -> None:
  dom = _derivable_domain(
    "message_seq",
    rate={
      "value": 1,
      "unit": "msg_per_tick",
      "bounds": "task_commit_issue",
      "source": {"file": "pipeline_validator/tile_group.py", "line": 1230},
    },
  )
  data["lifetimes"]["domains"] = [dom]
  widths, errors = vp.derive_all(data["budgets"], data["lifetimes"])
  assert widths == []
  assert any("rate bounds 'task_commit_issue', not 'message_seq'" in e for e in errors)


def test_domain_with_matching_evidence_does_derive(data: dict) -> None:
  data["lifetimes"]["domains"] = [_derivable_domain()]
  widths, errors = vp.derive_all(data["budgets"], data["lifetimes"])
  assert errors == []
  assert [(w.name, w.window) for w in widths] == [("fixture_dom", 16 + 2 * 10)]
  assert 2 ** widths[0].bits > widths[0].window


def test_derivable_claim_without_rate_or_lifetime_is_rejected(data: dict) -> None:
  for missing in ("rate", "lifetime_ticks"):
    dom = _derivable_domain()
    del dom[missing]
    data["lifetimes"]["domains"] = [dom]
    _, errors = vp.derive_all(data["budgets"], data["lifetimes"])
    assert any("mark derivable: false" in e for e in errors), missing


def test_evidence_without_sourced_file_is_rejected(data: dict) -> None:
  dom = _derivable_domain()
  del dom["rate"]["source"]
  data["lifetimes"]["domains"] = [dom]
  _, errors = vp.derive_all(data["budgets"], data["lifetimes"])
  assert any("rate has no sourced file" in e for e in errors)


def test_unresolved_max_in_flight_ref_is_rejected(data: dict) -> None:
  dom = _derivable_domain()
  dom["max_in_flight"] = {"ref": "channel_budgets.transaction_capacity.nope"}
  data["lifetimes"]["domains"] = [dom]
  _, errors = vp.derive_all(data["budgets"], data["lifetimes"])
  assert any("unresolved max_in_flight ref" in e for e in errors)


def test_budget_rates_declare_what_they_bound(data: dict) -> None:
  for rate in data["budgets"]["rates"]:
    assert rate.get("bounds"), f"rate {rate['name']} must declare what it bounds"
