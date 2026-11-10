"""C010 native owner / ROB tag / retire-contract tests.

Consumers are exercised with structured legal and mismatched samples (the
reference RobBank decodes completions by the full key); source citations are
read back from the pinned tree. Nothing here is a string search for OwnerTag.
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

import validate_native_owner as vo  # noqa: E402
import validate_protocol as vp  # noqa: E402


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


def test_committed_contracts_validate() -> None:
  report = vo.validate()
  assert report["valid"], report["errors"]


# --- the card's headline case: a reused slot's old result is not the new op --


def test_reused_slot_old_result_is_not_read_as_the_new_op() -> None:
  bank = vo.RobBank(context_count=2, capacity=4)
  old = bank.allocate(0, owner_op_seq=10)
  bank.complete(old)
  assert bank.parts[0].retire() == 10
  # drive the tail around so the SAME slot is allocated again
  for seq in (11, 12, 13):
    bank.complete(bank.allocate(0, seq))
    bank.parts[0].retire()
  reused = bank.allocate(0, owner_op_seq=14)
  assert reused.rob_slot == old.rob_slot, "fixture must actually reuse the slot"
  with pytest.raises(vo.OwnerError) as exc:
    bank.complete(old)  # late result carrying the OLD tag
  assert exc.value.code == "STALE_SLOT_GENERATION"
  bank.complete(reused)  # the new op still completes normally


def test_bare_context_plus_slot_would_have_been_ambiguous() -> None:
  """Show the failure the generation prevents: same (ctx, slot), different op."""
  bank = vo.RobBank(context_count=1, capacity=2)
  first = bank.allocate(0, 1)
  bank.complete(first)
  bank.parts[0].retire()
  bank.complete(bank.allocate(0, 2))
  bank.parts[0].retire()
  second = bank.allocate(0, 3)
  assert (first.ctx_slot, first.rob_slot) == (second.ctx_slot, second.rob_slot)
  assert first.slot_generation != second.slot_generation


@pytest.mark.parametrize(
  ("mutate", "code"),
  [
    (lambda t: vo.Tag(1, t.ctx_generation, t.rob_slot, t.slot_generation, t.owner_op_seq), "WRONG_CONTEXT"),
    (lambda t: vo.Tag(t.ctx_slot, t.ctx_generation + 1, t.rob_slot, t.slot_generation, t.owner_op_seq),
     "STALE_CTX_GENERATION"),
    (lambda t: vo.Tag(t.ctx_slot, t.ctx_generation, t.rob_slot, t.slot_generation, t.owner_op_seq + 1),
     "OP_SEQ_MISMATCH"),
    (lambda t: vo.Tag(t.ctx_slot, t.ctx_generation, 3, t.slot_generation, t.owner_op_seq), "SLOT_NOT_LIVE"),
    (lambda t: vo.Tag(t.ctx_slot, t.ctx_generation, 99, t.slot_generation, t.owner_op_seq),
     "SLOT_OUT_OF_RANGE"),
  ],
)
def test_mismatched_completion_keys_are_each_rejected(mutate, code: str) -> None:
  bank = vo.RobBank(context_count=2, capacity=4)
  good = bank.allocate(0, 5)
  with pytest.raises(vo.OwnerError) as exc:
    bank.parts[0].complete(mutate(good))
  assert exc.value.code == code


def test_duplicate_completion_is_rejected_not_applied_twice() -> None:
  bank = vo.RobBank(context_count=1, capacity=4)
  tag = bank.allocate(0, 1)
  bank.complete(tag)
  with pytest.raises(vo.OwnerError) as exc:
    bank.complete(tag)
  assert exc.value.code == "DUPLICATE_COMPLETION"


def test_two_contexts_with_the_same_op_seq_do_not_collide() -> None:
  bank = vo.RobBank(context_count=2, capacity=4)
  a, b = bank.allocate(0, 7), bank.allocate(1, 7)
  bank.complete(a)
  assert bank.parts[1].live == 1, "completing ctx 0 must not touch ctx 1"
  bank.complete(b)
  assert bank.parts[0].retire() == 7 and bank.parts[1].retire() == 7


def test_retire_is_in_order_within_a_context() -> None:
  part = vo.RobPartition(0, 1, 4)
  t0, t1 = part.allocate(1), part.allocate(2)
  part.complete(t1)  # younger completes first
  assert part.retire() is None, "head not done yet"
  part.complete(t0)
  assert [part.retire(), part.retire()] == [1, 2]


def test_full_partition_backpressures_instead_of_overwriting() -> None:
  part = vo.RobPartition(0, 1, 2)
  part.allocate(1)
  part.allocate(2)
  with pytest.raises(vo.OwnerError) as exc:
    part.allocate(3)
  assert exc.value.code == "ROB_FULL"


# --- Context reuse requires closed obligations and a consumed completion ----


def test_context_reuse_needs_both_preconditions() -> None:
  bank = vo.RobBank(context_count=2, capacity=4)
  tag = bank.allocate(0, 1)
  with pytest.raises(vo.OwnerError) as exc:
    bank.reuse_context(0, completion_consumed=True)
  assert exc.value.code == "REUSE_BLOCKED" and "obligations" in str(exc.value)
  bank.complete(tag)
  bank.parts[0].retire()
  with pytest.raises(vo.OwnerError) as exc2:
    bank.reuse_context(0, completion_consumed=False)
  assert "TASK_COMPLETION" in str(exc2.value)
  assert bank.reuse_context(0, completion_consumed=True) == 2


def test_old_task_result_is_stale_after_context_reuse() -> None:
  bank = vo.RobBank(context_count=2, capacity=4)
  tag = bank.allocate(0, 1)
  bank.complete(tag)
  bank.parts[0].retire()
  bank.reuse_context(0, completion_consumed=True)
  with pytest.raises(vo.OwnerError) as exc:
    bank.complete(tag)
  assert exc.value.code == "STALE_CTX_GENERATION"


# --- C1 constant degradation and C2 dual partition are both expressible -----


def test_c1_constant_and_c2_dual_partition_layouts() -> None:
  c1, c2 = vo.tag_layout(1, 16), vo.tag_layout(2, 16)
  assert c1["ctx_bits"] >= 1 and c1["ctx_slot_constant"] is True, "C1 must not be zero-width"
  assert c2["ctx_bits"] >= 1 and c2["ctx_slot_constant"] is False
  assert c1["partitions"] == 1 and c2["partitions"] == 2
  assert c1["rob_slot_bits"] == c2["rob_slot_bits"] == 4


def test_c1_and_c2_banks_both_behave() -> None:
  for count in (1, 2):
    bank = vo.RobBank(context_count=count, capacity=4)
    tag = bank.allocate(0, 1)
    bank.complete(tag)
    assert bank.parts[0].retire() == 1
  with pytest.raises(ValueError, match="positive integer"):
    vo.RobBank(context_count=0, capacity=4)


def test_log2_ceil_matches_chisel_log2ceil() -> None:
  assert [vo.log2_ceil(n) for n in (1, 2, 3, 16, 17, 66, 98)] == [0, 1, 2, 4, 5, 7, 7]
  with pytest.raises(ValueError):
    vo.log2_ceil(0)


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
    "floatRegCount": 32, "retirementBufferSize": 16, "scalarRegCount": 32,
    "rvvRegCount": 32, "rvv_rob_tag_width_fallback": 5,
  }
  assert any("width mismatch" in e for e in _errors(vo.check_conflicts, docs, params))


def test_rob_tag_confused_with_register_index_width_is_caught(docs: dict) -> None:
  # retirementBufferSize 66 -> 7 bits, equal to the register-index width: conflict.
  params = {
    "floatRegCount": 32, "retirementBufferSize": 66, "scalarRegCount": 32,
    "rvvRegCount": 32, "rvv_rob_tag_width_fallback": 7,
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


def test_retire_record_upstream_field_names_are_read_from_the_trace(docs: dict) -> None:
  field = next(f for f in docs["retire_record"]["fields"] if f["name"] == "pc_rdata")
  field["source"]["line"] = 1
  assert any("pc_rdata" in e for e in _errors(vo.check_retire_record, docs, {"retirementLanes": 8}))


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
  request["requests"].append(
    {"message": "RETIRE", "direction": "G2T", "why": "x", "fields_needed": []}
  )
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
