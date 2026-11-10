"""C013 memory ownership / page authorization / bounded lifetime contract tests.

The validator (hardware/coral_nexus/scripts/derive_w4_bounds.py) is driven by
the committed YAML; every test below either checks the derived envelope against
an independent hand computation, executes the reference lifecycle ledger, or
mutates an input and asserts the coded check fires.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
sys.path.insert(0, str(H / "scripts"))

import derive_w4_bounds as dw  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def constraints() -> dict:
  return copy.deepcopy(dw.load_constraints())


@pytest.fixture
def contract() -> dict:
  return copy.deepcopy(dw.load_contract())


@pytest.fixture
def field_request() -> dict:
  return copy.deepcopy(dw.load_yaml(dw.REQUEST))


@pytest.fixture
def protocol() -> dict:
  return copy.deepcopy(dw.load_yaml(dw.PROTOCOL))


def _derive(data: dict) -> dict[str, Any]:
  derived, errors = dw.validate_constraints(data)
  assert errors == [], errors
  return derived


def _ledger(contract: dict, constraints: dict) -> dw.MemoryLifecycleLedger:
  return dw.MemoryLifecycleLedger(contract, dw.bounds_env(constraints))


def _code(fn: Any, *args: Any, **kw: Any) -> str:
  with pytest.raises((dw.LedgerError, dw.BoundsError)) as exc:
    fn(*args, **kw)
  err = exc.value
  assert isinstance(err, (dw.LedgerError, dw.BoundsError))
  return err.code


def _contract_errors(contract: dict, constraints: dict, field_request: dict, protocol: dict) -> list[str]:
  derived, _ = dw.validate_constraints(constraints)
  return dw.validate_contract(contract, derived, field_request, protocol)


# ---------------------------------------------------------------------------
# committed files
# ---------------------------------------------------------------------------
def test_committed_files_validate() -> None:
  report = dw.validate_all()
  assert report["valid"], report["errors"]
  assert set(report["unresolved"]) == {
    "per_ctx_outstanding_credit_cap",
    "mshr_token_width_bits",
    "response_record_budget",
    "mfe_external_address_bits",
    "lease_ref_width_bits",
  }


def test_cli_writes_a_report_and_exits_zero(tmp_path: Path) -> None:
  out = tmp_path / "report.json"
  run = subprocess.run(
    [sys.executable, str(H / "scripts/derive_w4_bounds.py"), "--out", str(out)],
    capture_output=True,
    text=True,
    check=False,
  )
  assert run.returncode == 0, run.stderr
  assert json.loads(out.read_text())["valid"] is True


def test_cli_exits_nonzero_on_a_broken_contract(tmp_path: Path) -> None:
  bad = tmp_path / "bad_contract.yaml"
  data = dw.load_contract()
  del data["accounting_ledgers"]
  bad.write_text(yaml.safe_dump(data))
  run = subprocess.run(
    [sys.executable, str(H / "scripts/derive_w4_bounds.py"), "--contract", str(bad)],
    capture_output=True,
    text=True,
    check=False,
  )
  assert run.returncode == 1
  assert "accounting_ledgers" in run.stdout


# ---------------------------------------------------------------------------
# derivation vs an independent hand computation (arithmetic shown)
# ---------------------------------------------------------------------------
def test_derivation_matches_independent_hand_computation(constraints: dict) -> None:
  d = _derive(constraints)

  # bank budget: 1048576 B total / 16 banks = 65536 B per bank
  assert d["l1_per_bank_bytes"]["value"] == 1048576 // 16 == 65536
  assert d["l1_per_bank_bytes"]["unit"] == "B"

  # per-bank bandwidth: 512 GB/s = 512 * 10**9 B/s; per bank / 16 = 32 * 10**9 B/s;
  # clock 1000 MHz = 10**9 cycles/s; 32e9 B/s / 1e9 cycles/s = 32 B/cycle
  assert d["l1_bank_bytes_per_cycle"]["value"] == (512 * 10**9 / 16) / 10**9 == 32
  assert d["l1_bank_bytes_per_cycle"]["unit"] == "B/cycle"
  # full-tile cache stage: 512 GB/s at 1 GHz = 512 B/cycle (NOT divided by banks)
  assert d["l1_cache_bytes_per_cycle"]["value"] == 512 * 10**9 / 10**9 == 512

  # user SPM mode 0: 65536 - 2048 = 63488 B (profiles.py user_spm_per_bank)
  assert d["user_spm_per_bank_mode0"]["value"] == 65536 - 2048 == 63488
  assert d["user_spm_per_bank_mode1"]["value"] == 57344 - 2048 == 55296
  assert d["user_spm_per_bank_mode2"]["value"] == 49152 - 2048 == 47104
  assert d["user_spm_per_bank_mode3"]["value"] == 32768 - 2048 == 30720

  # bank busy for one line burst: 1 cycle latency + ceil(64 B / 32 B/cycle) = 1 + 2 = 3
  assert d["bank_busy_cycles_line_burst"]["value"] == 1 + (64 + 32 - 1) // 32 == 3
  assert d["bank_busy_cycles_line_burst"]["unit"] == "cycle"

  # window envelope: floor(16 refill entries / 4 concurrent jobs) = 4 entries per job
  assert d["gather_window_entries_per_job_no_mshr_stall"]["value"] == 16 // 4 == 4
  assert d["scatter_window_entries_per_job_no_mshr_stall"]["value"] == 16 // 4 == 4

  # identity budgets carry their own dimension, never summed with each other
  assert d["frame_slot_budget"]["value"] == 16 and d["frame_slot_budget"]["unit"] == "entry"
  assert d["frame_bind_latency_budget"]["value"] == 8 and d["frame_bind_latency_budget"]["unit"] == "cycle"
  assert d["l1_refill_budget"]["value"] == 16 and d["l1_refill_budget"]["unit"] == "entry"
  assert (
    d["external_outstanding_budget"]["value"] == 32 and d["external_outstanding_budget"]["unit"] == "entry"
  )


def test_kib_converts_to_bytes_instead_of_confusing(constraints: dict) -> None:
  entry = next(e for e in constraints["inputs"] if e["name"] == "tile_l1_capacity_bytes")
  entry["value"] = 1024  # 1 MiB expressed in KiB
  entry["unit"] = "KiB"
  derived, errors = dw.validate_constraints(constraints)
  assert derived["l1_per_bank_bytes"]["value"] == 1024 * 1024 // 16 == 65536  # converted, not compared
  assert any("lacks value 1024" in e for e in errors)  # the provenance drift is still flagged


# ---------------------------------------------------------------------------
# unit confusion / capacity / range rejections
# ---------------------------------------------------------------------------
def test_adding_entries_to_cycles_is_rejected(constraints: dict) -> None:
  constraints["derivations"].append(
    {"name": "bad_sum", "expr": "l1_mshr_entries + l1_access_latency_cycles", "unit": "cycles"}
  )
  _, errors = dw.validate_constraints(constraints)
  assert any("UNIT_MISMATCH" in e for e in errors)


def test_calling_bytes_entries_breaks_the_derivation_that_uses_them(constraints: dict) -> None:
  entry = next(e for e in constraints["inputs"] if e["name"] == "cache_line_bytes")
  entry["unit"] = "entries"
  _, errors = dw.validate_constraints(constraints)
  assert any("UNIT_MISMATCH" in e for e in errors)


def test_declaring_the_wrong_result_unit_is_rejected(constraints: dict) -> None:
  item = next(d for d in constraints["derivations"] if d["name"] == "l1_per_bank_bytes")
  item["unit"] = "entries"
  _, errors = dw.validate_constraints(constraints)
  assert any("UNIT_MISMATCH" in e for e in errors)


def test_byte_partition_must_conserve_the_bank(constraints: dict) -> None:
  entry = next(e for e in constraints["inputs"] if e["name"] == "cache_bytes_per_bank_mode1")
  entry["value"] = 8192 + 64  # breaks spm + cache == bank bytes
  _, errors = dw.validate_constraints(constraints)
  assert any("PARTITION_NOT_CONSERVED" in e for e in errors)


def test_zero_capacity_without_allow_zero_is_rejected(constraints: dict) -> None:
  entry = next(e for e in constraints["inputs"] if e["name"] == "l1_mshr_entries")
  entry["value"] = 0
  _, errors = dw.validate_constraints(constraints)
  assert any("VALUE_NOT_POSITIVE" in e for e in errors)


def test_window_envelope_below_one_entry_is_out_of_range(constraints: dict) -> None:
  entry = next(e for e in constraints["inputs"] if e["name"] == "mfe_load_depth")
  entry["value"] = 32  # floor(16/32) = 0 entries per job
  _, errors = dw.validate_constraints(constraints)
  assert any("OUT_OF_RANGE" in e for e in errors)


def test_division_by_zero_is_rejected(constraints: dict) -> None:
  constraints["inputs"] = [
    e for e in constraints["inputs"] if e["name"] != "tile_l1_banks"
  ]  # removing the divisor name is caught as UNKNOWN_NAME
  _, errors = dw.validate_constraints(constraints)
  assert any("UNKNOWN_NAME" in e for e in errors)
  data = copy.deepcopy(constraints)
  data["derivations"].append(
    {"name": "bad_div", "expr": "cache_line_bytes / (cache_line_bytes - cache_line_bytes)", "unit": "banks"}
  )
  _, errors = dw.validate_constraints(data)
  assert any("DIV_BY_ZERO" in e for e in errors)


# ---------------------------------------------------------------------------
# unsourced / invented values stay fail-closed
# ---------------------------------------------------------------------------
_INVENTED = ["16", "8+8", "log2Ceil(16)", "max(8, 8)", "l1_mshr_entries"]


@pytest.mark.parametrize("invention", _INVENTED)
def test_unresolved_field_accepts_no_value_or_expression(constraints: dict, invention: str) -> None:
  item = next(u for u in constraints["unresolved"] if u["name"] == "mshr_token_width_bits")
  item["expr"] = invention
  _, errors = dw.validate_constraints(constraints)
  assert any("INVENTED_VALUE" in e for e in errors)


@pytest.mark.parametrize("invention", _INVENTED)
def test_invention_is_rejected_on_every_unresolved_field_at_once(constraints: dict, invention: str) -> None:
  for item in constraints["unresolved"]:
    item["value"] = 16 if invention == "16" else None
    item["expr"] = None if invention == "16" else invention
  _, errors = dw.validate_constraints(constraints)
  assert any("INVENTED_VALUE" in e for e in errors)
  assert dw.validate_all()["valid"] is True  # the committed files stay clean


_INVENTED_KEYS = [
  ("value", 16),
  ("value", "8+8"),
  ("expr", "log2Ceil(16)"),
  ("width_bits", 8),
  ("width_rule", "8+8"),
  ("width_rule", "log2Ceil(16)"),
  ("bits", 8),
  ("depth", 16),
  ("max_entries", 16),
  ("unit", "B"),
]


@pytest.mark.parametrize(("key", "invention"), _INVENTED_KEYS)
def test_no_key_can_smuggle_a_number_into_an_unsourced_entry(
  constraints: dict, key: str, invention: Any
) -> None:
  """Every site at once, so no conflict is visible; an entry may hold only its closed field set."""
  for item in constraints["unresolved"]:
    item[key] = invention
  _, errors = dw.validate_constraints(constraints)
  assert any("INVENTED_VALUE" in e and key in e for e in errors)
  assert len([e for e in errors if "INVENTED_VALUE" in e]) == len(constraints["unresolved"])


def test_one_entry_gaining_a_width_is_rejected_alone(constraints: dict) -> None:
  next(u for u in constraints["unresolved"] if u["name"] == "mfe_external_address_bits")["width_bits"] = 48
  _, errors = dw.validate_constraints(constraints)
  assert [e for e in errors if "INVENTED_VALUE" in e] == [
    "unresolved mfe_external_address_bits: INVENTED_VALUE (unsourced field carries width_bits=48)"
  ]


@pytest.mark.parametrize("name", sorted(dw.REQUIRED_UNRESOLVED))
def test_deleting_an_unsourced_entry_does_not_make_it_derivable(constraints: dict, name: str) -> None:
  constraints["unresolved"] = [u for u in constraints["unresolved"] if u["name"] != name]
  _, errors = dw.validate_constraints(constraints)
  assert any(name in e and "REQUIRED_UNRESOLVED_MISSING" in e for e in errors)


def test_deleting_every_unsourced_entry_is_rejected(constraints: dict) -> None:
  constraints["unresolved"] = []
  _, errors = dw.validate_constraints(constraints)
  assert len([e for e in errors if "REQUIRED_UNRESOLVED_MISSING" in e]) == len(dw.REQUIRED_UNRESOLVED)


def test_an_unlisted_unsourced_entry_or_changed_owner_is_rejected(constraints: dict) -> None:
  constraints["unresolved"].append(
    {"name": "extra", "derivable": False, "reason": "r", "needs": ["x"], "owner": "C018"}
  )
  _, errors = dw.validate_constraints(constraints)
  assert any("extra" in e and "NOT_IN_REQUIRED_SET" in e for e in errors)
  constraints["unresolved"].pop()
  constraints["unresolved"][0]["owner"] = "C999"
  _, errors = dw.validate_constraints(constraints)
  assert any("OWNER_CHANGED" in e for e in errors)


def test_the_required_unresolved_set_lives_in_the_validator_not_the_yaml(constraints: dict) -> None:
  before = dict(dw.REQUIRED_UNRESOLVED)
  constraints["unresolved"] = []
  assert dw.REQUIRED_UNRESOLVED == before and len(before) == 5


def test_unresolved_without_needs_is_rejected(constraints: dict) -> None:
  item = next(u for u in constraints["unresolved"] if u["name"] == "response_record_budget")
  item["needs"] = []
  _, errors = dw.validate_constraints(constraints)
  assert any("UNRESOLVED_WITHOUT_NEEDS" in e for e in errors)


def test_declared_derivable_must_stay_false(constraints: dict) -> None:
  item = next(u for u in constraints["unresolved"] if u["name"] == "lease_ref_width_bits")
  item["derivable"] = True
  _, errors = dw.validate_constraints(constraints)
  assert any("NOT_FAIL_CLOSED" in e for e in errors)


# ---------------------------------------------------------------------------
# mutation sweep: every input and derivation, several mutations each
# ---------------------------------------------------------------------------
def _mutations() -> list[tuple[str, Any]]:
  return [
    ("negative", -1),
    ("zero", 0),
    ("unit-cycles", "cycles"),
    ("unit-entries", "entries"),
    ("drop-source", None),
    ("drift-line", +5),
    ("wrong-value", 12345),
  ]


@pytest.mark.parametrize(("mutate", "payload"), _mutations())
def test_every_input_mutation_fires_a_check(constraints: dict, mutate: str, payload: Any) -> None:
  fired = 0
  total = 0
  for entry in constraints["inputs"]:
    data = copy.deepcopy(constraints)
    target = next(e for e in data["inputs"] if e["name"] == entry["name"])
    if mutate == "negative":
      target["value"] = -abs(int(target["value"])) if target["value"] else -1
    elif mutate == "zero":
      if target.get("allow_zero"):
        continue  # zero is a declared legitimate partition for this entry
      target["value"] = 0
    elif mutate.startswith("unit"):
      if target["unit"] == payload:
        continue
      target["unit"] = payload
    elif mutate == "drop-source":
      target.pop("source", None)
    elif mutate == "drift-line":
      target["source"]["line"] += 5
    elif mutate == "wrong-value":
      if str(12345) in str(target["value"]):
        continue
      target["value"] = 12345 if target["value"] != 12345 else 54321
    total += 1
    _, errors = dw.validate_constraints(data)
    assert errors, f"{entry['name']} / {mutate} survived"
    fired += 1
  assert fired == total and fired > 0


@pytest.mark.parametrize("mutate", ["wrong-unit", "zero-expr", "drop-expr"])
def test_every_derivation_mutation_fires_a_check(constraints: dict, mutate: str) -> None:
  for item in constraints["derivations"]:
    data = copy.deepcopy(constraints)
    target = next(d for d in data["derivations"] if d["name"] == item["name"])
    if mutate == "wrong-unit":
      target["unit"] = "cycles" if target["unit"] != "cycles" else "entries"
    elif mutate == "zero-expr":
      target["expr"] = f"({target['expr']}) * 0"
    else:
      target.pop("expr", None)
    _, errors = dw.validate_constraints(data)
    assert errors, f"derivation {item['name']} / {mutate} survived"


def test_duplicate_names_are_rejected(constraints: dict) -> None:
  constraints["derivations"].append(
    {"name": "l1_mshr_entries", "expr": "l1_mshr_entries", "unit": "entries"}
  )
  _, errors = dw.validate_constraints(constraints)
  assert any("DUPLICATE_NAME" in e for e in errors)


def test_unknown_unit_is_rejected(constraints: dict) -> None:
  entry = next(e for e in constraints["inputs"] if e["name"] == "cache_line_bytes")
  entry["unit"] = "furlongs"
  _, errors = dw.validate_constraints(constraints)
  assert any("UNKNOWN_UNIT" in e for e in errors)


def test_expression_syntax_is_whitelisted(constraints: dict) -> None:
  for bad in ("__import__('os').system('true')", "l1_mshr_entries ** 2", "cache_line_bytes;"):
    constraints["derivations"].append({"name": f"bad_{len(bad)}", "expr": bad, "unit": "entries"})
  _, errors = dw.validate_constraints(constraints)
  assert sum("EXPR_SYNTAX" in e for e in errors) >= 3


# ---------------------------------------------------------------------------
# contract checks: mutate the committed contract and the validator flags it
# ---------------------------------------------------------------------------
def test_ownership_split_is_single_owner(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  assert _contract_errors(contract, constraints, field_request, protocol) == []
  contract["ownership"]["python_executes"].append(
    {"state": "l1_arena_ledger", "scope": "x", "python_role": "y"}
  )
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("BOTH sides" in e for e in errors)


def test_dropping_a_non_goal_is_caught(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  contract["ownership"]["non_goals"].remove("changing_l2_physical_reclaim_granularity")
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("non_goals" in e for e in errors)


def test_removing_arena_alloc_gen_provenance_is_caught(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  contract["provenance"]["fields"] = [
    f for f in contract["provenance"]["fields"] if f["name"] != "arena_alloc_gen"
  ]
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("provenance.fields" in e for e in errors)
  assert any("field_request.MEM_REQ.arena_alloc_gen" in e for e in errors)


def test_terminal_states_and_marker_are_pinned(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  contract["terminal"]["states"] = ["COMPLETED", "ISOLATED"]  # FAULTED dropped
  contract["terminal"]["cancel_marker"] = "CANCELLED"  # a terminal posing as a marker
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("terminal.states" in e for e in errors)
  assert any("cancel_marker" in e for e in errors)


def test_gate_scope_drift_is_caught(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  contract["lifecycle"]["profile_gate"]["blocked"] = ["new_data_issue", "completion_ack"]
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("profile_gate.blocked" in e for e in errors)


def test_gate_blocking_a_terminal_ack_is_caught(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  contract["lifecycle"]["profile_gate"]["exempt"].remove("completion_ack")
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("completion_ack" in e for e in errors)


def test_hold_conservation_identity_drift_is_caught(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  entry = next(c for c in contract["lifecycle"]["conservation"] if c["id"] == "PRO-16-HOLD")
  entry["identity"] = "allocated == bound"  # hold vanished
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("PRO-16-HOLD" in e for e in errors)


def test_citing_a_line_the_validator_does_not_say_is_caught(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  rule = contract["lifecycle"]["l1_free_vs_retirement"][0]
  rule["source"]["line"] = 1
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("SOURCE_LINE_MISMATCH" in e for e in errors)


def test_upstream_baseline_drift_is_caught(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  base = next(b for b in contract["upstream_baselines"] if b["name"] == "lsuSlotTxidEntries")
  base["source"]["line"] = 165  # the other constant's line
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("SOURCE_LINE_MISMATCH" in e for e in errors)


def test_mshr_section_drift_is_caught(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  contract["mshr"]["merge_key"] = ["bare_address"]  # loses provenance
  contract["mshr"]["token_states"] = ["ALLOCATED", "FREE"]  # loses waiter survival
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("merge_key" in e for e in errors)
  assert any("token_states" in e for e in errors)


def test_error_code_set_is_exact(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  contract["errors"]["codes"].append({"code": "invented_fault", "trigger": "x"})
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("errors.codes" in e for e in errors)


def test_page_authorization_must_consume_the_existing_transaction_set(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  contract["page_authorization"]["existing_transaction_set"] = []
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert sum("consume, do not re-implement" in e for e in errors) == 5


def test_field_request_rejects_unknown_message_and_field(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  field_request["requests"][0]["message"] = "NOT_A_MESSAGE"
  field_request["requests"][1]["fields_needed"].append(
    {"name": "no_such_field", "from": "protocol/w4_memory.contract.yaml", "key": "provenance.fields"}
  )
  field_request["new_messages_requested"].append(
    {"name": "MEM_REQ", "channel": "MEMORY", "dir": "T2G", "why": "dup"}
  )
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("not in the C009 catalog" in e for e in errors)
  assert any("no_such_field" in e for e in errors)
  assert any("already in the catalog" in e for e in errors)


# ---------------------------------------------------------------------------
# reference lifecycle: hold -> bind -> retire with the conservation identity
# ---------------------------------------------------------------------------
def test_hold_bind_retire_conserves_capacity_every_step(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  per_bank = led.per_bank
  led.hold("T1", bank=0, size=4096, slots=(0,))
  assert led.conservation() == [
    {
      "bank": i,
      "held": 4096 if i == 0 else 0,
      "bound": 0,
      "allocated": 4096 if i == 0 else 0,
      "free": per_bank - 4096 if i == 0 else per_bank,
    }
    for i in range(len(led._banks))
  ]
  gen = led.bind("T1")
  assert gen == 1  # the arena allocation generation, real provenance
  led.alloc_view("T1", "v0", size=4096, permission="rw", slot=0)
  led.free_view("v0")  # FREE_L1 invalidates ONE view only
  row = led.conservation()[0]
  assert row["allocated"] == 4096, "a freed view must not shrink the bound arena"
  led.retire("T1", arena_alloc_gen=gen)
  assert led.conservation()[0]["free"] == per_bank  # L1 arena returns at safe retirement


def test_hold_never_oversells_to_a_concurrent_ticket(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led.hold("T1", bank=2, size=led.per_bank - 1024, slots=(3,))
  assert _code(led.hold, "T2", 2, led.per_bank) == "HOLD_OVERCOMMIT"
  led.hold("T2", 2, 1024)  # exactly the remaining capacity is fine
  led.compensate("T1")  # pre-authorization failure returns held -> free
  assert led.conservation()[2]["free"] == led.per_bank - 1024  # T2 still holds its share


def test_hold_bind_retire_negative_paths(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  assert _code(led.hold, "T1", 0, 0) == "HOLD_NEGATIVE_SIZE"
  assert _code(led.hold, "T1", 999, 8) == "BANK_OUT_OF_RANGE"
  assert _code(led.hold, "T1", 0, 64, (99,)) == "SLOT_OUT_OF_RANGE"
  assert _code(led.bind, "T1") == "BIND_WITHOUT_HOLD"
  assert _code(led.compensate, "nope") == "COMPENSATE_WITHOUT_HOLD"
  led.hold("T1", 0, 512, (0,))
  led.bind("T1")
  led.alloc_view("T1", "v0", 512, "rw", 0)
  assert _code(led.retire, "T1") == "RETIRE_WITH_LIVE_VIEWS"
  led.free_view("v0")
  assert _code(led.retire, "T1", arena_alloc_gen=99) == "RETIRE_GENERATION_MISMATCH"
  led.retire("T1")
  assert _code(led.retire, "T1") == "RETIRE_ALREADY"


def test_second_ticket_cannot_steal_a_bound_slot(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led.hold("T1", 0, 512, (5,))
  assert _code(led.hold, "T2", 1, 512, (5,)) == "SLOT_IN_USE"
  led.bind("T1")
  assert _code(led.compensate, "T1") == "COMPENSATE_WITHOUT_HOLD"  # post-authorization: no rollback
  led.alloc_view("T1", "v", 512, "rw", 5)
  led.free_view("v")
  led.retire("T1")  # the bound arena (and its slot) returns at safe retirement
  led.hold("T2", 1, 512, (5,))  # now the slot is free again
  assert led.conservation()[1]["held"] == 512


def test_free_view_refuses_double_free_and_inflight(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led.hold("T1", 0, 512, (0,))
  led.bind("T1")
  led.alloc_view("T1", "v0", 512, "rw", 0)
  led.begin_inflight("v0", "txn-1")
  assert _code(led.free_view, "v0") == "VIEW_INFLIGHT"
  led.end_inflight("v0", "txn-1")
  led.free_view("v0")
  assert _code(led.free_view, "v0") == "VIEW_DOUBLE_FREE"
  assert _code(led.begin_inflight, "v0", "txn-2") == "ISSUE_ON_INVALIDATED_VIEW"


def test_corrupted_bank_state_breaks_the_conservation_identity(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led._banks[0].free -= 8  # corrupt the ledger behind the API
  with pytest.raises(dw.LedgerError) as exc:
    led.conservation()
  assert exc.value.code == "LEDGER_CONSERVATION"


# ---------------------------------------------------------------------------
# refill waiter survival and the MSHR token lifecycle
# ---------------------------------------------------------------------------
def test_refill_waiter_survives_cache_eviction(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  slot = led.mshr_allocate(generation=0)
  led.mshr_add_waiter(slot)
  led.mshr_add_waiter(slot)
  led.mshr_issue_refill(slot)
  led.mshr_refill(slot, line_bytes=64)
  led.mshr_evict_cache_line(slot)  # the cache line is gone...
  assert led.mshr_waiter_consume(slot) == 64  # ...but the fill buffer still serves the waiter
  assert _code(led.mshr_force_free, slot) == "MSHR_TOKEN_EARLY_REUSE"  # one waiter remains
  assert led.mshr_waiter_consume(slot) == 64
  reuse = led.mshr_allocate(generation=1)  # only now the slot is reusable
  assert reuse == slot


def test_mshr_merge_is_generation_scoped_and_capacity_bounded(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  slot = led.mshr_allocate(generation=3)
  led.mshr_merge(slot, generation=3)  # same generation merges
  assert _code(led.mshr_merge, slot, 4) == "MSHR_GENERATION_MISMATCH"
  for _ in range(led.mshr_capacity - 1):
    led.mshr_allocate(generation=3)
  assert _code(led.mshr_allocate, 3) == "MSHR_FULL"
  assert _code(led.mshr_add_waiter, 999) == "MSHR_UNKNOWN_TOKEN"


# ---------------------------------------------------------------------------
# external write settlement: three terminals, five separate ledgers
# ---------------------------------------------------------------------------
def test_external_write_settlement_completes_after_confirmation(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led.hold("T1", 0, 512, (0,))
  led.bind("T1")
  led.alloc_view("T1", "out", 512, "rw", 0)
  led.issue_credit("txn-A", 2)
  led.begin_inflight("out", "txn-A")
  led.lease_acquire("txn-A", "scope-1", {7: ("scope-1", 0), 8: ("scope-1", 0)}, epoch=0)
  led.accept_external_write("txn-A", "out", 64)
  led.accept_external_write("txn-A", "out", 64)
  led.external_write_response("txn-A", 64)
  led.end_inflight("out", "txn-A")
  assert _code(led.settle, "txn-A", "COMPLETED") == "COMPLETION_WITH_UNSETTLED_WRITE"
  led.external_write_response("txn-A", 64)
  assert _code(led.lease_release, "txn-A") == "LEASE_NOT_TERMINAL"  # GAT-19: release at terminal ACK only
  led.settle("txn-A", "COMPLETED")
  assert _code(led.settle, "txn-A", "COMPLETED") == "DUPLICATE_TERMINAL"
  # the terminal settled every ledger exactly once
  assert led.ledgers == {
    "cancellation": 0,
    "isolation_confirmed": 0,
    "stale_response": 0,
    "credit": 0,  # 2 issued, 2 returned at the terminal
    "inflight_references": 0,
  }


def test_isolation_requires_the_marker_and_forbids_later_writes(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led.hold("T1", 0, 512, (0,))
  led.bind("T1")
  led.alloc_view("T1", "out", 512, "rw", 0)
  led.accept_external_write("txn-B", "out", 32)
  assert _code(led.settle, "txn-B", "ISOLATED") == "ISOLATION_REQUIRES_MARKER"
  led.cancel("txn-B")  # only a marker, not a terminal
  led.cancel("txn-B")  # repeated markers do not double count
  assert led.ledgers["cancellation"] == 1
  led.settle("txn-B", "ISOLATED")
  assert led.ledgers["isolation_confirmed"] == 1
  assert _code(led.accept_external_write, "txn-B", "out", 8) == "ISOLATION_BROKEN"
  assert _code(led.revoke_external_write, "txn-B") == "EXTERNAL_WRITE_IRREVOCABLE"


def test_read_only_view_rejects_external_writes(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led.hold("T1", 0, 512, (0,))
  led.bind("T1")
  led.alloc_view("T1", "src", 512, "r", 0)
  assert _code(led.accept_external_write, "txn-C", "src", 8) == "PERMISSION_DENIED"
  assert _code(led.alloc_view, "T1", "bad", 8, "rx", 0) == "PERMISSION_DENIED"


def test_late_response_is_stale_and_touches_no_other_ledger(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led.hold("T1", 0, 512, (0,))
  led.bind("T1")
  led.alloc_view("T1", "out", 512, "rw", 0)
  led.issue_credit("txn-D", 1)
  led.accept_external_write("txn-D", "out", 16)
  led.external_write_response("txn-D", 16)
  led.settle("txn-D", "COMPLETED")
  before = dict(led.ledgers)
  led.external_write_response("txn-D", 16)  # late: dropped, not a completion
  led.late_response("txn-D")
  assert led.ledgers["stale_response"] == 2
  for name in ("cancellation", "isolation_confirmed", "credit", "inflight_references"):
    assert led.ledgers[name] == before[name], f"{name} ledger was cross-written"


def test_credit_ledger_rejects_under_and_over_return(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led.issue_credit("txn-E", 2)
  assert _code(led.return_credit, "txn-E", 3) == "CREDIT_OVER_RETURN"
  led.return_credit("txn-E", 2)
  assert _code(led.return_credit, "txn-E", 1) == "CREDIT_UNDERFLOW"
  assert _code(led.end_inflight, "nowhere", "txn-E") == "INFLOW_MISMATCH"


def test_gate_blocks_only_new_data_issue(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led.set_gate(closed=True)
  assert _code(led.data_issue, "new_data_issue") == "GATE_BLOCKS_NEW_ISSUE"
  for path in ("completion_ack", "maintenance_clean", "program_install", "cancel_and_drain"):
    assert led.data_issue(path) is True
  led.set_gate(closed=False)
  assert led.data_issue("new_data_issue") is True


def test_unknown_terminal_is_refused(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  assert _code(led.settle, "txn-F", "CANCELLED") == "UNKNOWN_TERMINAL"


# ---------------------------------------------------------------------------
# page lease: atomic check+take, terminal release, idempotence
# ---------------------------------------------------------------------------
def test_lease_acquire_is_atomic_and_epoch_checked(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  pages = {7: ("scope-1", 0), 8: ("scope-2", 0)}  # page 8 has a new owner
  assert _code(led.lease_acquire, "txn-G", "scope-1", pages, 0) == "HOST_SCOPE_INACTIVE"
  assert led._txn("txn-G").lease is None, "a failed range must take nothing"
  ok = {7: ("scope-1", 0), 8: ("scope-1", 0)}
  scope, held = led.lease_acquire("txn-G", "scope-1", ok, 0)
  assert scope == "scope-1" and held == frozenset({7, 8})
  led.cancel("txn-G")
  led.settle("txn-G", "ISOLATED")  # settle releases the lease at the terminal
  assert _code(led.lease_release, "txn-G") == "LEASE_DOUBLE_RELEASE"


# ---------------------------------------------------------------------------
# early input release and idempotent remaining-pin retirement
# ---------------------------------------------------------------------------
def test_input_release_is_per_task_then_all_tasks_and_retire_is_idempotent(
  contract: dict, constraints: dict
) -> None:
  led = _ledger(contract, constraints)
  led.add_l2_buffer("B_IN", "R1", 131072, readonly_shared=False)
  led.pin_l2("task-0", "B_IN", reads=True, writes=False)
  led.pin_l2("task-1", "B_IN", reads=True, writes=False)
  led.pin_l2("task-1", "B_IN", reads=True, writes=True)  # read-write pin survives input release
  assert led.signal_input_released("task-0") == 1
  assert led._l2["B_IN"].refs == 2
  assert led.grid_all_tasks_signaled(("task-0", "task-1")) == 1  # only the pure-read pin drops
  assert led._l2["B_IN"].refs == 1
  assert led.retire_cleanup("task-1") == 1  # safe retirement cleans the remainder
  assert led.retire_cleanup("task-1") == 0  # ...idempotently
  assert led._l2["B_IN"].refs == 0


# ---------------------------------------------------------------------------
# the three distinct L2 moments
# ---------------------------------------------------------------------------
def test_private_backing_final_free_precedes_root_retirement(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led.add_l2_buffer("priv", "R1", 65536, readonly_shared=False, claims={"c1"})
  led.add_root("R1", ("priv",))
  assert _code(led.l2_release, "priv") == "L2_CLAIMS_OPEN"  # all-or-nothing preflight
  led.l2_close_claim("priv", "c1")  # claims close through their view release
  led.l2_release("priv")
  span = led.l2_final_free("priv")  # BEFORE root retirement: private backing
  assert span == 65536
  led.root_retire("R1")
  led.root_completion("R1")  # published only after retirement
  assert led._roots["R1"].completion is True


def test_shared_readonly_backing_frees_only_after_producer_root_retirement(
  contract: dict, constraints: dict
) -> None:
  led = _ledger(contract, constraints)
  led.add_l2_buffer("shared", "R1", 131072, readonly_shared=True)
  led.add_root("R1", ("shared",))
  led.l2_release("shared")
  assert led.l2_final_free("shared") is None  # producer still owns the origin
  led.root_retire("R1")
  assert led.l2_final_free("shared") == 131072  # AFTER the producer root retired
  led.root_completion("R1")
  assert _code(led.l2_claim, "shared", "c9") == "L2_FORFEITED"  # release is permanent


def test_root_completion_cannot_precede_retirement(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led.add_l2_buffer("b", "R1", 64, readonly_shared=False)
  led.add_root("R1", ("b",), outstanding=1)
  assert _code(led.root_completion, "R1") == "ROOT_COMPLETION_BEFORE_RETIRE"
  assert _code(led.root_retire, "R1") == "ROOT_RETIRE_WITH_OBLIGATIONS"


def test_pinned_shared_backing_does_not_final_free(contract: dict, constraints: dict) -> None:
  led = _ledger(contract, constraints)
  led.add_l2_buffer("shared", "R1", 131072, readonly_shared=True)
  led.add_root("R1", ("shared",))
  led.pin_l2("task-0", "shared", reads=True, writes=False)
  led.l2_release("shared")
  led.root_retire("R1")
  assert led.l2_final_free("shared") is None  # the pure-read pin still holds a reference
  led.signal_input_released("task-0")
  assert led.l2_final_free("shared") == 131072


# ---------------------------------------------------------------------------
# the contract YAML drives the ledger: mutate it and behaviour changes
# ---------------------------------------------------------------------------
def test_removing_isolated_from_the_yaml_removes_the_terminal(contract: dict, constraints: dict) -> None:
  contract["terminal"]["states"].remove("ISOLATED")
  led = dw.MemoryLifecycleLedger(contract, dw.bounds_env(constraints))
  assert _code(led.settle, "txn-H", "ISOLATED") == "UNKNOWN_TERMINAL"
  assert any(
    "terminal.states" in e
    for e in _contract_errors(contract, constraints, dw.load_yaml(dw.REQUEST), dw.load_yaml(dw.PROTOCOL))
  )


def test_removing_the_credit_ledger_from_the_yaml_breaks_the_counters(
  contract: dict, constraints: dict
) -> None:
  contract["accounting_ledgers"]["ledgers"] = [
    row for row in contract["accounting_ledgers"]["ledgers"] if row["name"] != "credit"
  ]
  led = dw.MemoryLifecycleLedger(contract, dw.bounds_env(constraints))
  with pytest.raises(KeyError):  # the counter the contract no longer names is gone
    led.issue_credit("txn-0", 1)
  assert any(
    "accounting_ledgers" in e
    for e in _contract_errors(contract, constraints, dw.load_yaml(dw.REQUEST), dw.load_yaml(dw.PROTOCOL))
  )


def test_widening_the_gate_in_the_yaml_blocks_more_paths(contract: dict, constraints: dict) -> None:
  contract["lifecycle"]["profile_gate"]["blocked"] = ["new_data_issue", "maintenance_clean"]
  led = dw.MemoryLifecycleLedger(contract, dw.bounds_env(constraints))
  led.set_gate(closed=True)
  assert _code(led.data_issue, "maintenance_clean") == "GATE_BLOCKS_NEW_ISSUE"


def test_removing_the_user_spm_derivation_starves_the_ledger(contract: dict, constraints: dict) -> None:
  constraints["derivations"] = [
    d for d in constraints["derivations"] if d["name"] != "user_spm_per_bank_mode0"
  ]
  with pytest.raises(KeyError):
    dw.MemoryLifecycleLedger(contract, dw.bounds_env(constraints))


def test_changing_mshr_capacity_in_the_yaml_changes_the_table(contract: dict, constraints: dict) -> None:
  entry = next(e for e in constraints["inputs"] if e["name"] == "l1_mshr_entries")
  entry["value"] = 2  # no config line backs a 2-entry L1 MSHR: flagged AND honoured
  led = dw.MemoryLifecycleLedger(contract, dw.bounds_env(constraints))
  led.mshr_allocate(generation=0)
  led.mshr_allocate(generation=0)
  assert _code(led.mshr_allocate, 0) == "MSHR_FULL"
  _, errors = dw.validate_constraints(constraints)
  assert any("SOURCE_LINE_MISMATCH" in e for e in errors)


# ---------------------------------------------------------------------------
# whole-section deletion and the field request (the advisory pattern applied to C013)
# ---------------------------------------------------------------------------
_SECTIONS = [
  "ownership",
  "provenance",
  "addressing",
  "page_authorization",
  "accounting_ledgers",
  "terminal",
  "lifecycle",
  "mshr",
  "errors",
]


@pytest.mark.parametrize("section", _SECTIONS)
def test_deleting_a_contract_section_is_rejected(
  contract: dict, constraints: dict, field_request: dict, protocol: dict, section: str
) -> None:
  contract.pop(section)
  assert _contract_errors(contract, constraints, field_request, protocol), section


def test_a_field_request_from_file_is_checked_per_field(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  """`from` lives on each fields_needed entry (as in C010), not on the request."""
  assert _contract_errors(contract, constraints, field_request, protocol) == []
  field_request["requests"][0]["fields_needed"][0]["from"] = "protocol/does_not_exist.yaml"
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("does_not_exist" in e and "not a protocol file" in e for e in errors)


def test_a_field_request_asking_for_a_missing_field_is_rejected(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  field_request["requests"][0]["fields_needed"].append(
    {"name": "no_such_field", "from": "protocol/w4_memory.contract.yaml", "key": "provenance.fields"}
  )
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("no_such_field" in e for e in errors)


def test_the_required_upstream_baselines_cannot_be_deleted_or_altered(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  gone = copy.deepcopy(contract)
  gone.pop("upstream_baselines")
  errors = _contract_errors(gone, constraints, field_request, protocol)
  assert any("lsuDataBits" in e and "missing" in e for e in errors)
  assert any("lsuSlotTxidEntries" in e and "missing" in e for e in errors)
  changed = copy.deepcopy(contract)
  next(b for b in changed["upstream_baselines"] if b["name"] == "lsuSlotTxidEntries")["value"] = 16
  assert any("required 8" in e for e in _contract_errors(changed, constraints, field_request, protocol))


def test_a_nested_citation_that_drifts_is_caught_wherever_it_hides(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  """The walk covers every `source`, including ones no per-section check lists."""
  errors: list[str] = []
  assert dw.check_all_sources(errors, contract, "contract") >= 25
  assert errors == []
  entry = next(
    e for e in contract["addressing"]["indexed"]["errors"] if e["code"] == "indexed_address_overflow"
  )
  entry["source"]["line"] += 1
  found = _contract_errors(contract, constraints, field_request, protocol)
  assert any("indexed.errors" in e and "SOURCE_LINE_MISMATCH" in e for e in found)


def test_every_source_in_the_committed_contract_and_constraints_reads_back() -> None:
  errors: list[str] = []
  contract = dw.load_contract()
  constraints = dw.load_constraints()
  assert dw.check_all_sources(errors, contract, "contract") >= 25
  assert dw.check_all_sources(errors, constraints, "constraints") >= 20
  assert errors == []


# --- numeric guards: non-finite values and sourced requirements --------------------------------


def _set_input(constraints: dict, name: str, value: Any) -> None:
  next(e for e in constraints["inputs"] if e["name"] == name)["value"] = value


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_capacity_is_rejected_before_it_can_defeat_every_comparison(
  constraints: dict, bad: float
) -> None:
  _set_input(constraints, "tile_l1_capacity_bytes", bad)
  _, errors = dw.validate_constraints(constraints)
  assert any("tile_l1_capacity_bytes" in e and "NON_FINITE_VALUE" in e for e in errors)


@pytest.mark.parametrize("bad", [48, 63, 100, 64.5, 96])
def test_cache_line_must_be_a_power_of_two_as_cache_py_requires(constraints: dict, bad: float) -> None:
  _set_input(constraints, "cache_line_bytes", bad)
  _, errors = dw.validate_constraints(constraints)
  assert any("NOT_POWER_OF_TWO" in e for e in errors)


def test_the_power_of_two_requirement_cannot_be_dropped_or_renamed(constraints: dict) -> None:
  entry = next(e for e in constraints["inputs"] if e["name"] == "cache_line_bytes")
  entry["requires"] = "banana"
  assert any("UNKNOWN_REQUIREMENT" in e for e in dw.validate_constraints(constraints)[1])
  entry.pop("requires")
  assert any("POWER_OF_TWO_REQUIREMENT_MISSING" in e for e in dw.validate_constraints(constraints)[1])


def test_the_power_of_two_requirement_is_read_back_from_cache_py(constraints: dict) -> None:
  entry = next(e for e in constraints["inputs"] if e["name"] == "cache_line_bytes")
  entry["requires_source"]["line"] += 1
  errors = dw.validate_constraints(constraints)[1]
  assert any("requires_source" in e and "SOURCE_LINE_MISMATCH" in e for e in errors)


def test_an_unquoted_comma_that_splits_a_flow_mapping_is_caught(
  contract: dict, constraints: dict, field_request: dict, protocol: dict
) -> None:
  """`{ api: f(a, b), owner: X }` parses to a junk key; both real occurrences were found this way."""
  parsed = yaml.safe_load("{ api: acquire_range(scope, binding, offset), owner: W4-04 }")
  assert "binding" in parsed and parsed["api"] == "acquire_range(scope"  # the failure mode itself
  contract["page_authorization"]["changes_required"][0] = parsed
  errors = _contract_errors(contract, constraints, field_request, protocol)
  assert any("MALFORMED_KEY" in e for e in errors) or any("offset)" in e for e in errors)


def test_malformed_keys_in_the_constraints_are_caught(constraints: dict) -> None:
  constraints["inputs"][0]["size)"] = None
  _, errors = dw.validate_constraints(constraints)
  assert any("MALFORMED_KEY" in e for e in errors)


def test_the_committed_w4_yaml_has_no_malformed_keys(contract: dict, constraints: dict) -> None:
  errors: list[str] = []
  dw.check_mapping_shape(errors, contract, "contract")
  dw.check_mapping_shape(errors, constraints, "constraints")
  assert errors == []
