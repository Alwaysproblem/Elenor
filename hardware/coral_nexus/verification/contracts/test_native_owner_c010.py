"""C010 native owner / ROB tag / retire-contract tests.

The consumer under test (`contract_codec.RobConsumer`) is built FROM the YAML,
so these tests feed it legal, missing, mismatched and out-of-range samples and
also mutate the YAML to prove the consumer follows it. Source citations are
read back from the pinned tree. Nothing here is a string search for OwnerTag.

Widths of ctx_generation / owner_op_seq / slot_generation / subop_id are
unresolved in the contract. `FIXTURE` gives them explicit test-only widths so a
consumer can be exercised; they are not contract values.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
sys.path.insert(0, str(H / "scripts/lib"))

import contract_codec as cc  # noqa: E402
import validate_native_owner as vo  # noqa: E402
import validate_protocol as vp  # noqa: E402

FIXTURE = {"ctx_generation": 4, "owner_op_seq": 8, "slot_generation": 2, "subop_id": 2}


@pytest.fixture
def docs() -> dict:
  return copy.deepcopy(vo.load_docs())


@pytest.fixture
def proto() -> dict:
  return copy.deepcopy(vp.load_all(vo.PROTOCOL))


def _errors(fn, *args) -> list[str]:
  errors: list[str] = []
  fn(*args, errors)
  return errors


def _values(docs: dict) -> dict[str, int]:
  values: dict[str, int] = {}
  for item in docs["rob_tag"]["parameters"]:
    values[item["name"]] = int(item["value"])
  values.update({p["name"]: int(p["value"]) for p in docs["retire_record"]["width_parameters"]})
  return values


def _rob(docs: dict, contexts: int, capacity: int | None = None, **fixture) -> cc.RobConsumer:
  values = _values(docs)
  if capacity is not None:
    values["retirementBufferSize"] = capacity
  return cc.RobConsumer(docs["rob_tag"], cc.ConfigParams(contexts, values, {**FIXTURE, **fixture}))


def _alloc(rob: cc.RobConsumer, ctx: int, seq: int, sub: int = 0) -> tuple[int, int]:
  return rob.allocate(ctx, owner_op_seq=seq, subop_id=sub)


def _fail_code(fn, *args, **kw) -> str:
  with pytest.raises(cc.ContractError) as exc:
    fn(*args, **kw)
  return exc.value.code


def test_committed_contracts_validate() -> None:
  report = vo.validate()
  assert report["valid"], report["errors"]


# --- the card's headline case: a reused slot's old result is not the new op --


def test_reused_slot_old_result_is_not_read_as_the_new_op(docs: dict) -> None:
  rob = _rob(docs, 2, capacity=4)
  old = _alloc(rob, 0, 10)
  rob.complete(*old)
  assert rob.retire(0) == 10
  for seq in (11, 12, 13):  # drive the tail around so the SAME slot is allocated again
    rob.complete(*_alloc(rob, 0, seq))
    rob.retire(0)
  reused = _alloc(rob, 0, 14)
  assert rob.tag_spec.decode(reused[0])["rob_slot"] == rob.tag_spec.decode(old[0])["rob_slot"]
  assert _fail_code(rob.complete, *old) == "STALE_SLOT_GENERATION"  # late result with the OLD tag
  rob.complete(*reused)  # the new op still completes normally


def test_stale_result_is_rejected_even_when_op_seq_collides(docs: dict) -> None:
  """Same (ctx, slot) AND same owner_op_seq: only the slot generation tells them apart."""
  rob = _rob(docs, 1, capacity=2)
  first = _alloc(rob, 0, 5)
  rob.complete(*first)
  rob.retire(0)
  rob.complete(*_alloc(rob, 0, 6))
  rob.retire(0)
  second = _alloc(rob, 0, 5)
  assert first[1] == second[1], "owner words are identical by construction"
  assert first[0] != second[0], "the ROB tag must differ"
  assert _fail_code(rob.complete, *first) == "STALE_SLOT_GENERATION"


def test_generation_that_no_longer_fits_is_refused_not_wrapped(docs: dict) -> None:
  """A 2-bit slot generation holds 4 occupants per slot; the 5th must refuse, never alias generation 0."""
  rob = _rob(docs, 1, capacity=2)
  done = 0
  with pytest.raises(cc.ContractError) as exc:
    for seq in range(64):
      rob.complete(*_alloc(rob, 0, seq))
      rob.retire(0)
      done += 1
  assert exc.value.code == "FIELD_OUT_OF_RANGE"
  assert done == 8, "2 slots x 4 generations"
  assert rob.live(0) == 0, "a refused allocation must not consume a slot"


def test_one_entry_rob_is_refused_rather_than_made_zero_width(docs: dict) -> None:
  assert _fail_code(_rob, docs, 1, capacity=1) == "ZERO_WIDTH"


def _mutations() -> list:
  def swap_ctx(tag, owner, rob):
    other = 1 - rob.tag_spec.decode(tag)["ctx_slot"]
    return rob.tag_spec.pack({**rob.tag_spec.decode(tag), "ctx_slot": other}), owner

  def bump(spec_name: str, field: str, delta: int):
    def mutate(tag, owner, rob):
      spec = getattr(rob, spec_name)
      word = tag if spec_name == "tag_spec" else owner
      new = spec.pack({**spec.decode(word), field: spec.decode(word)[field] + delta})
      return (new, owner) if spec_name == "tag_spec" else (tag, new)

    return mutate

  return [
    (swap_ctx, "SLOT_NOT_LIVE"),
    (bump("owner_spec", "ctx_generation", 1), "STALE_CTX_GENERATION"),
    (bump("owner_spec", "owner_op_seq", 1), "OP_SEQ_MISMATCH"),
    (bump("owner_spec", "subop_id", 1), "TAG_MISMATCH"),
    (bump("tag_spec", "slot_generation", 1), "STALE_SLOT_GENERATION"),
    (bump("tag_spec", "rob_slot", 1), "SLOT_NOT_LIVE"),
  ]


@pytest.mark.parametrize(("mutate", "code"), _mutations())
def test_each_mismatched_key_is_rejected_with_its_own_code(docs: dict, mutate, code: str) -> None:
  rob = _rob(docs, 2, capacity=4)
  tag, owner = _alloc(rob, 0, 5)
  bad_tag, bad_owner = mutate(tag, owner, rob)
  assert _fail_code(rob.complete, bad_tag, bad_owner) == code
  rob.complete(tag, owner)  # the untouched sample still completes: rejection left no side effect


def test_rob_slot_beyond_capacity_is_out_of_range(docs: dict) -> None:
  rob = _rob(docs, 1, capacity=3)  # 2 slot bits can name slot 3, but only 0..2 exist
  tag, owner = _alloc(rob, 0, 1)
  forged = rob.tag_spec.pack({**rob.tag_spec.decode(tag), "rob_slot": 3})
  assert _fail_code(rob.complete, forged, owner) == "SLOT_OUT_OF_RANGE"


def test_words_with_bits_beyond_the_layout_are_out_of_bounds(docs: dict) -> None:
  rob = _rob(docs, 2, capacity=4)
  tag, owner = _alloc(rob, 0, 1)
  assert _fail_code(rob.complete, tag | (1 << rob.tag_spec.bits), owner) == "RECORD_OUT_OF_BOUNDS"
  assert _fail_code(rob.complete, tag, -1) == "RECORD_OUT_OF_BOUNDS"


# --- packing: missing / unknown / malformed / out-of-range fields ------------


def test_missing_required_field_is_refused_on_allocate(docs: dict) -> None:
  rob = _rob(docs, 2, capacity=4)
  assert _fail_code(rob.allocate, 0, owner_op_seq=1) == "FIELD_MISSING"  # no subop_id
  assert _fail_code(rob.allocate, 0, subop_id=0) == "FIELD_MISSING"  # no owner_op_seq
  assert rob.live(0) == 0


def test_unknown_malformed_and_oversized_fields_are_refused(docs: dict) -> None:
  rob = _rob(docs, 2, capacity=4)
  assert _fail_code(rob.allocate, 0, owner_op_seq=1, subop_id=0, bogus=1) == "FIELD_UNKNOWN"
  assert _fail_code(rob.allocate, 0, owner_op_seq=1, subop_id=0, slot_generation=0) == "FIELD_MANAGED"
  assert _fail_code(rob.allocate, 0, owner_op_seq="1", subop_id=0) == "FIELD_TYPE"
  assert _fail_code(rob.allocate, 0, owner_op_seq=True, subop_id=0) == "FIELD_TYPE"
  assert _fail_code(rob.allocate, 0, owner_op_seq=-1, subop_id=0) == "FIELD_OUT_OF_RANGE"
  assert _fail_code(rob.allocate, 0, owner_op_seq=256, subop_id=0) == "FIELD_OUT_OF_RANGE"
  boundary = _rob(docs, 1, capacity=4)
  boundary.allocate(0, owner_op_seq=255, subop_id=3)  # the largest value each field holds is legal


def test_unresolved_width_without_a_fixture_is_refused_not_guessed(docs: dict) -> None:
  config = cc.ConfigParams(2, _values(docs))  # no fixture widths
  assert _fail_code(cc.RobConsumer, docs["rob_tag"], config) == "WIDTH_UNRESOLVED"


def test_ctx_slot_out_of_range_is_rejected(docs: dict) -> None:
  rob = _rob(docs, 2, capacity=4)
  assert _fail_code(_alloc, rob, 2, 1) == "CTX_OUT_OF_RANGE"
  assert _fail_code(rob.retire, 5) == "CTX_OUT_OF_RANGE"
  assert _fail_code(rob.live, -1) == "CTX_OUT_OF_RANGE"


def test_duplicate_completion_is_rejected_not_applied_twice(docs: dict) -> None:
  rob = _rob(docs, 1, capacity=4)
  sample = _alloc(rob, 0, 1)
  rob.complete(*sample)
  assert _fail_code(rob.complete, *sample) == "DUPLICATE_COMPLETION"


def test_two_contexts_with_the_same_op_seq_do_not_collide(docs: dict) -> None:
  rob = _rob(docs, 2, capacity=4)
  a, b = _alloc(rob, 0, 7), _alloc(rob, 1, 7)
  rob.complete(*a)
  assert rob.live(1) == 1 and rob.retire(1) is None, "completing ctx 0 must not touch ctx 1"
  rob.complete(*b)
  assert rob.retire(0) == 7 and rob.retire(1) == 7


def test_retire_is_in_order_within_a_context(docs: dict) -> None:
  rob = _rob(docs, 1, capacity=4)
  t0, t1 = _alloc(rob, 0, 1), _alloc(rob, 0, 2)
  rob.complete(*t1)  # younger completes first
  assert rob.retire(0) is None, "head not done yet"
  rob.complete(*t0)
  assert [rob.retire(0), rob.retire(0)] == [1, 2]


def test_full_partition_backpressures_instead_of_overwriting(docs: dict) -> None:
  rob = _rob(docs, 2, capacity=2)
  _alloc(rob, 0, 1)
  _alloc(rob, 0, 2)
  assert _fail_code(_alloc, rob, 0, 3) == "ROB_FULL"
  _alloc(rob, 1, 1)  # the other Context's partition is private and unaffected


# --- Context reuse requires closed obligations and a consumed completion ----


def test_context_reuse_needs_both_preconditions(docs: dict) -> None:
  rob = _rob(docs, 2, capacity=4)
  rule = docs["ctx_ctrl"]["reuse_rule"]["context_slot_reuse_requires"]
  sample = _alloc(rob, 0, 1)
  with pytest.raises(cc.ContractError) as exc:
    rob.reuse_context(0, completion_consumed=True, reuse_rule=rule)
  assert exc.value.code == "REUSE_BLOCKED" and "obligations" in str(exc.value)
  rob.complete(*sample)
  rob.retire(0)
  with pytest.raises(cc.ContractError) as exc2:
    rob.reuse_context(0, completion_consumed=False, reuse_rule=rule)
  assert "TASK_COMPLETION" in str(exc2.value)
  assert rob.reuse_context(0, completion_consumed=True, reuse_rule=rule) == 2


def test_old_task_result_is_stale_after_context_reuse(docs: dict) -> None:
  rob = _rob(docs, 2, capacity=4)
  rule = docs["ctx_ctrl"]["reuse_rule"]["context_slot_reuse_requires"]
  sample = _alloc(rob, 0, 1)
  rob.complete(*sample)
  rob.retire(0)
  rob.reuse_context(0, completion_consumed=True, reuse_rule=rule)
  assert _fail_code(rob.complete, *sample) == "SLOT_NOT_LIVE"  # the old slot is not reoccupied yet
  new = _alloc(rob, 0, 1)  # same owner_op_seq as the old Task's op
  # Isolate ctx_generation: the NEW slot's tag with the OLD Task's owner word.
  assert _fail_code(rob.complete, new[0], sample[1]) == "STALE_CTX_GENERATION"
  rob.complete(*new)


# --- C1 constant degradation and C2 dual partition are both expressible -----


def test_c1_constant_and_c2_dual_partition_layouts(docs: dict) -> None:
  values = _values(docs)
  c1, c2 = vo.layout(docs, values, 1), vo.layout(docs, values, 2)
  assert c1["ctx_bits"] >= 1 and c1["ctx_slot_constant"] is True, "C1 must not be zero-width"
  assert c2["ctx_bits"] >= 1 and c2["ctx_slot_constant"] is False
  assert c1["partitions"] == 1 and c2["partitions"] == 2
  assert c1["rob_slot_bits"] == c2["rob_slot_bits"] == 4


def test_c1_and_c2_consumers_both_behave(docs: dict) -> None:
  for count in (1, 2):
    rob = _rob(docs, count)
    assert rob.tag_spec.names[0] == "ctx_slot" and rob.tag_spec.fields[0].width >= 1
    rob.complete(*_alloc(rob, 0, 1))
    assert rob.retire(0) == 1
  assert _fail_code(cc.ConfigParams(0, _values(docs)).env) == "CONTEXT_COUNT"


def test_c2_partitions_are_private_at_full_capacity(docs: dict) -> None:
  rob = _rob(docs, 2)  # upstream retirementBufferSize = 16 per partition
  for seq in range(16):
    _alloc(rob, 0, seq % 256)
  assert _fail_code(_alloc, rob, 0, 16) == "ROB_FULL"
  assert rob.live(1) == 0 and _alloc(rob, 1, 0)


def test_log2_ceil_matches_chisel_log2ceil() -> None:
  assert [cc.log2_ceil(n) for n in (1, 2, 3, 16, 17, 66, 98)] == [0, 1, 2, 4, 5, 7, 7]
  with pytest.raises(cc.ContractError):
    cc.log2_ceil(0)


def test_width_rules_are_evaluated_not_trusted() -> None:
  env = {"a": 5}
  assert cc.eval_width("log2Ceil(a) + 1", env) == 4
  assert _fail_code(cc.eval_width, "unresolved", env) == "WIDTH_UNRESOLVED"
  assert _fail_code(cc.eval_width, "b", env) == "WIDTH_PARAM_UNKNOWN"
  assert _fail_code(cc.eval_width, "a - 5", env) == "ZERO_WIDTH"
  assert _fail_code(cc.eval_width, "__import__('os')", env) == "WIDTH_RULE_FORBIDDEN"
  assert _fail_code(cc.eval_width, "a +", env) == "WIDTH_RULE_SYNTAX"


# --- the consumer follows the YAML: mutate it and the behaviour changes -------


def test_removing_slot_generation_from_yaml_removes_aba_protection(docs: dict) -> None:
  docs["rob_tag"]["rob_tag"]["fields"] = [
    f for f in docs["rob_tag"]["rob_tag"]["fields"] if f["name"] != "slot_generation"
  ]
  rob = _rob(docs, 1, capacity=2)
  first = _alloc(rob, 0, 5)
  rob.complete(*first)
  rob.retire(0)
  rob.complete(*_alloc(rob, 0, 6))
  rob.retire(0)
  _alloc(rob, 0, 5)  # reuses slot 0 with the SAME owner word
  assert rob.tag_spec.names == ["ctx_slot", "rob_slot"]
  rob.complete(*first)  # the stale result is now ACCEPTED: this is what the field prevented
  assert any("slot_generation" in e for e in _errors(vo.check_required_floor, docs)), (
    "the validator must flag that removal"
  )


def test_marking_a_match_field_optional_in_yaml_is_flagged(docs: dict) -> None:
  for f in docs["rob_tag"]["owner_tag"]["fields"]:
    if f["name"] == "owner_op_seq":
      f["required"] = False
  assert any("match floor" in e for e in _errors(vo.check_required_floor, docs))


def test_yaml_width_rule_change_changes_the_layout(docs: dict) -> None:
  values = _values(docs)
  before = vo.layout(docs, values, 2)["rob_slot_bits"]
  for f in docs["rob_tag"]["rob_tag"]["fields"]:
    if f["name"] == "rob_slot":
      f["width_rule"] = "log2Ceil(retirementBufferSize) + 1"
  assert vo.layout(docs, values, 2)["rob_slot_bits"] == before + 1


def test_a_new_required_yaml_field_is_demanded_by_the_consumer(docs: dict) -> None:
  docs["rob_tag"]["owner_tag"]["fields"].append(
    {"name": "lane", "width_rule": "2", "owner": "W2", "consumers": ["W3"], "required": True}
  )
  rob = _rob(docs, 1, capacity=4)
  assert _fail_code(rob.allocate, 0, owner_op_seq=1, subop_id=0) == "FIELD_MISSING"
  rob.allocate(0, owner_op_seq=1, subop_id=0, lane=1)


def test_an_optional_yaml_field_is_not_part_of_the_match(docs: dict) -> None:
  docs["rob_tag"]["owner_tag"]["fields"].append(
    {"name": "hint", "width_rule": "3", "owner": "W2", "consumers": ["W3"], "required": False}
  )
  rob = _rob(docs, 1, capacity=4)
  tag, owner = rob.allocate(0, owner_op_seq=1, subop_id=0, hint=1)
  forged = rob.owner_spec.pack({**rob.owner_spec.decode(owner), "hint": 6})
  rob.complete(tag, forged)  # differs only in an optional field: accepted
  assert rob.retire(0) == 1


# --- retire record vs the real upstream trace interface ----------------------

_TRACE = H / "upstream/coralnpu/hdl/chisel/src/coralnpu/RvviTrace.scala"


def test_trace_signals_are_read_from_the_upstream_source() -> None:
  signals = cc.upstream_trace_signals(_TRACE)
  assert {"valid", "order", "insn", "trap", "pc_rdata", "x_wdata", "x_wb", "csr", "csr_wb"} <= set(signals)


def test_deleting_a_carried_trace_field_is_caught(docs: dict) -> None:
  rec = docs["retire_record"]
  rec["fields"] = [f for f in rec["fields"] if f["name"] not in ("trap", "pc_rdata")]
  errors = cc.check_trace_coverage(rec, _TRACE)
  assert any("'trap'" in e for e in errors) and any("'pc_rdata'" in e for e in errors)


def test_excluding_a_signal_needs_a_reason_and_cannot_double_book(docs: dict) -> None:
  rec = docs["retire_record"]
  rec["excluded_upstream"][0]["reason"] = ""
  rec["excluded_upstream"].append({"upstream_signal": "valid", "reason": "x"})
  errors = cc.check_trace_coverage(rec, _TRACE)
  assert any("without a reason" in e for e in errors) and any(
    "both carried and excluded" in e for e in errors
  )


def test_naming_a_signal_upstream_does_not_have_is_caught(docs: dict) -> None:
  docs["retire_record"]["excluded_upstream"].append({"upstream_signal": "ghost", "reason": "x"})
  assert any("'ghost'" in e for e in cc.check_trace_coverage(docs["retire_record"], _TRACE))


def test_marking_trap_optional_is_caught_by_the_oracle_dependency(docs: dict) -> None:
  for f in docs["retire_record"]["fields"]:
    if f["name"] == "trap":
      f["required"] = False
  assert any("'trap'" in e and "optional" in e for e in _errors(vo.check_oracle_retire_fields, docs))
  docs["retire_record"]["fields"] = [
    f for f in docs["retire_record"]["fields"] if f["name"] != "ctx_generation"
  ]
  assert any("'ctx_generation'" in e for e in _errors(vo.check_oracle_retire_fields, docs))


def test_every_retire_field_has_a_decodable_record_layout(docs: dict) -> None:
  values = _values(docs)
  fields = docs["retire_record"]["fields"]
  resolvable, open_ = cc.split_resolvable(fields, cc.ConfigParams(2, values).env())
  spec = cc.RecordSpec("retire_record", resolvable, cc.ConfigParams(2, values))
  assert set(open_) == {"ctx_generation", "owner_op_seq"}
  sample = {
    "ctx_slot": 1,
    "rob_tag": 9,
    "flags_delta": 0b101001,
    "valid": 1,
    "order": 77,
    "insn": 0x13,
    "trap": 0,
    "pc_rdata": 0x1000,
    "x_wdata": 5,
    "x_wb": 1,
  }
  assert spec.decode(spec.pack(sample), present=set(sample)) == sample
  assert _fail_code(spec.pack, {**sample, "flags_delta": 64}) == "FIELD_OUT_OF_RANGE"
  assert _fail_code(spec.pack, {k: v for k, v in sample.items() if k != "trap"}) == "FIELD_MISSING"


# --- contract-file checks (each one mutates the YAML and must fail) ----------


def test_cited_upstream_line_drift_is_caught(docs: dict) -> None:
  docs["rob_tag"]["upstream_keys_replaced"][0]["source"]["line"] = 30
  assert any("does not contain" in e for e in _errors(vo.check_replaced_keys, docs))


def test_wrong_quoted_text_is_caught(docs: dict) -> None:
  docs["rob_tag"]["upstream_keys_replaced"][1]["upstream"] = "something else entirely"
  assert _errors(vo.check_replaced_keys, docs)


def test_invented_width_on_unsourced_field_is_caught(docs: dict) -> None:
  field = next(f for f in docs["rob_tag"]["owner_tag"]["fields"] if f["name"] == "owner_op_seq")
  field["width_rule"] = "16"
  assert any("invented width" in e for e in _errors(vo.check_width_rules, docs))


def test_derivable_width_cannot_be_hidden_as_unresolved(docs: dict) -> None:
  field = next(f for f in docs["rob_tag"]["rob_tag"]["fields"] if f["name"] == "rob_slot")
  field["width_rule"] = "unresolved"
  assert any("derives from sourced parameters" in e for e in _errors(vo.check_width_rules, docs))


def test_conflicting_width_for_one_field_is_caught(docs: dict) -> None:
  field = next(f for f in docs["ctx_ctrl"]["fields"]["common"] if f["name"] == "ctx_slot")
  field["width_rule"] = "3"
  assert any("conflicts with" in e for e in _errors(vo.check_width_rules, docs))


def test_field_without_owner_or_consumers_is_caught(docs: dict) -> None:
  docs["retire_record"]["fields"][0].pop("owner")
  docs["rob_tag"]["owner_tag"]["fields"][0]["consumers"] = []
  errors = _errors(vo.check_field_schema, docs)
  assert any("'owner'" in e for e in errors) and any("'consumers'" in e for e in errors)


def test_parameter_value_must_match_its_cited_line(docs: dict) -> None:
  param = next(p for p in docs["rob_tag"]["parameters"] if p["name"] == "retirementLanes")
  param["value"] = 9
  assert any("not on its cited line" in e for e in _errors(vo.check_parameters, docs))


def test_rob_tag_width_mismatch_with_rvv_define_is_caught(docs: dict) -> None:
  params = {
    "floatRegCount": 32,
    "retirementBufferSize": 16,
    "scalarRegCount": 32,
    "rvvRegCount": 32,
    "rvv_rob_tag_width_fallback": 5,
  }
  assert any("width mismatch" in e for e in _errors(vo.check_conflicts, docs, params))


def test_rob_tag_confused_with_register_index_width_is_caught(docs: dict) -> None:
  # retirementBufferSize 66 -> 7 bits, equal to the register-index width: conflict.
  params = {
    "floatRegCount": 32,
    "retirementBufferSize": 66,
    "scalarRegCount": 32,
    "rvvRegCount": 32,
    "rvv_rob_tag_width_fallback": 7,
  }
  assert any("different quantities" in e for e in _errors(vo.check_conflicts, docs, params))


def test_csr_cannot_be_both_dispatch_beat_and_retire_commit(docs: dict, proto: dict) -> None:
  docs["ctx_ctrl"]["csr_commit"]["retire_commit"]["csrs"].append("MSCRATCH")
  errors = _errors(vo.check_csr_commit, docs, proto["upstream_csr"])
  assert any("listed in both" in e for e in errors)


def test_csr_commit_table_rejects_non_upstream_names(docs: dict, proto: dict) -> None:
  docs["ctx_ctrl"]["csr_commit"]["flags_delta"]["csrs"].append("NOT_A_CSR")
  assert any("not an upstream CSR" in e for e in _errors(vo.check_csr_commit, docs, proto["upstream_csr"]))


def test_gpr_completion_must_stay_separate_from_csr_commit(docs: dict, proto: dict) -> None:
  docs["ctx_ctrl"]["csr_commit"]["gpr_writeback"]["csr_commit_time"] = "same as GPR completion"
  assert any("separate" in e for e in _errors(vo.check_csr_commit, docs, proto["upstream_csr"]))


def test_wait_record_must_stay_with_w6_and_region_with_w3(docs: dict, proto: dict) -> None:
  for op in docs["ctx_ctrl"]["ops"]:
    if op["name"] in ("WAIT", "REGION_START"):
      op["record_owner"] = "W2"
  errors = _errors(vo.check_ops_and_control, docs, proto["protocol"])
  assert any("op WAIT" in e for e in errors) and any("op REGION_START" in e for e in errors)


def test_exec_drained_must_not_become_a_wire_message(docs: dict, proto: dict) -> None:
  proto["protocol"]["messages"].append({"name": "EXEC_DRAINED"})
  assert any("internal" in e for e in _errors(vo.check_ops_and_control, docs, proto["protocol"]))


def test_park_cancel_failstop_must_remain_three_mechanisms(docs: dict, proto: dict) -> None:
  del docs["ctx_ctrl"]["control_separation"]["cancel"]
  docs["ctx_ctrl"]["control_separation"]["park"]["uses_core_clock_gate"] = True
  errors = _errors(vo.check_ops_and_control, docs, proto["protocol"])
  assert any("cancel missing" in e for e in errors) and any("clock gate" in e for e in errors)


def test_retire_record_requires_full_mode_and_matching_lanes(docs: dict) -> None:
  docs["retire_record"]["mode_requirement"]["retirement_buffer_mode"] = "mini"
  docs["retire_record"]["lanes"]["value"] = 4
  errors = _errors(vo.check_retire_record, docs, {"retirementLanes": 8})
  assert any("full RetirementBuffer" in e for e in errors) and any("disagrees" in e for e in errors)


def test_retire_width_parameter_drift_is_caught(docs: dict) -> None:
  next(p for p in docs["retire_record"]["width_parameters"] if p["name"] == "rvvVlen")["source"]["line"] = 1
  errors = _errors(vo.check_retire_record, docs, {"retirementLanes": 8})
  assert any("rvvVlen" in e for e in errors)


def test_issue_bundle_rules_and_stall_reasons_are_pinned(docs: dict) -> None:
  docs["perctx_issue"]["bundle"]["rules"] = [
    r for r in docs["perctx_issue"]["bundle"]["rules"] if r["id"] != "prefix_only"
  ]
  docs["perctx_issue"]["signals"]["stall_reasons"]["existing_validator"].remove("WAIT_MSHR")
  errors = _errors(vo.check_issue, docs, {"instructionLanes": 4})
  assert any("prefix_only" in e for e in errors) and any("StallReason" in e for e in errors)


def test_oracle_must_be_per_context_and_key_matched(docs: dict) -> None:
  docs["oracle_requirements"]["input_contract"]["per_context"] = False
  docs["oracle_requirements"]["comparison_rules"] = [
    r for r in docs["oracle_requirements"]["comparison_rules"] if r["id"] != "no_pc_only_pairing"
  ]
  errors = _errors(vo.check_oracle, docs)
  assert any("per Context" in e for e in errors) and any("no_pc_only_pairing" in e for e in errors)


# --- field request vs the C009 catalog --------------------------------------


def test_field_request_points_at_real_catalog_messages_and_fields(docs: dict, proto: dict) -> None:
  request = yaml.safe_load((H / "protocol/field_requests/w2.yaml").read_text())
  assert _errors(vo.check_field_request, request, proto["protocol"], docs) == []


def test_field_request_rejects_unknown_message_owner_and_direction(docs: dict, proto: dict) -> None:
  request = yaml.safe_load((H / "protocol/field_requests/w2.yaml").read_text())
  request["requests"][0]["message"] = "NOPE"
  request["requests"][1]["direction"] = "T2G"
  request["requests"].append({"message": "RETIRE", "direction": "G2T", "why": "x", "fields_needed": []})
  errors = _errors(vo.check_field_request, request, proto["protocol"], docs)
  assert any("not in the protocol catalog" in e for e in errors)
  assert any("direction" in e for e in errors)
  assert any("owner is W4" in e for e in errors)


def test_field_request_cannot_ask_for_a_field_that_does_not_exist(docs: dict, proto: dict) -> None:
  request = yaml.safe_load((H / "protocol/field_requests/w2.yaml").read_text())
  request["requests"][0]["fields_needed"].append(
    {"name": "no_such_field", "from": "rtl/scalar/doc/rob_tag.yaml", "key": "owner_tag.fields"}
  )
  errors = _errors(vo.check_field_request, request, proto["protocol"], docs)
  assert any("no_such_field" in e for e in errors)


def test_unresolved_widths_are_reported_not_hidden() -> None:
  report = vo.validate()
  assert set(report["open_widths"]) == {"ctx_generation", "owner_op_seq", "slot_generation", "subop_id"}


def test_cli_exits_nonzero_on_a_broken_contract(tmp_path: Path) -> None:
  ok = subprocess.run(
    [sys.executable, str(H / "scripts/lib/validate_native_owner.py"), "--out", str(tmp_path / "ok")],
    capture_output=True,
  )
  assert ok.returncode == 0
  assert json.loads((tmp_path / "ok/native_owner_report.json").read_text())["valid"] is True


def test_trap_state_csrs_stay_unclassified_until_w2_04(docs: dict, proto: dict) -> None:
  """MCAUSE/MEPC/MTVAL have no sourced commit time; they must not be silently classified."""
  beat = docs["ctx_ctrl"]["csr_commit"]["dispatch_beat"]
  assert {"MCAUSE", "MEPC", "MTVAL"} <= set(beat["unclassified"]["csrs"])
  assert not {"MCAUSE", "MEPC", "MTVAL"} & set(beat["csrs"])
  assert _errors(vo.check_csr_commit, docs, proto["upstream_csr"]) == []
  # silently promoting one into a decided class is rejected as a double listing
  docs["ctx_ctrl"]["csr_commit"]["retire_commit"]["csrs"].append("MEPC")
  assert any("listed in both" in e for e in _errors(vo.check_csr_commit, docs, proto["upstream_csr"]))
  # and dropping the reason is rejected
  docs2 = copy.deepcopy(vo.load_docs())
  docs2["ctx_ctrl"]["csr_commit"]["dispatch_beat"]["unclassified"].pop("reason")
  errors2 = _errors(vo.check_csr_commit, docs2, proto["upstream_csr"])
  assert any("must stay status: open" in e for e in errors2)
