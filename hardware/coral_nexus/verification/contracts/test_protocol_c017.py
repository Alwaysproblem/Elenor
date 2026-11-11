"""C017 protocol generator / codec / CSR allocation tests.

What is proven here, and how:

* the C and SystemVerilog consumers are built from the GENERATED header/offsets only and must print, for the
  same byte streams, the same line as the generated Python receive model (valid, truncated, trailing,
  reserved, wrong port, bad enum/bool/bitset, condition, replay, gap, stale generation, ACK mismatch,
  duplicate stage, capacity). Every error name of the schema is exercised by that corpus;
* HOLD and FINAL (and their two ACKs) of one txn both apply, a repeated stage does not;
* the generator refuses every provenance gap (missing/foreign source, enum drift, address-like field, hidden
  field, unknown message, un-carried catalog key) and a refused run leaves earlier outputs byte-identical;
* the CSR allocator is deterministic, never hands out an upstream-occupied address, and fails on capacity;
  csr_lint and freeze_check detect a hand-edited generated file.

The SystemVerilog leg needs the in-tree Verilator build of C017 and a docker daemon; it is skipped (and the
skip says why) when either is missing. C and Python legs always run when gcc exists.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
PROTO = H / "protocol"
CONSUMERS = H / "verification/contracts/consumers"
sys.path.insert(0, str(H / "scripts/lib"))
sys.path.insert(0, str(PROTO))
sys.path.insert(0, str(REPO_ROOT))

import csr_allocator  # noqa: E402
import csr_lint  # noqa: E402
import derive_widths  # noqa: E402
import freeze_check  # noqa: E402
import gen_protocol as gp  # noqa: E402
import validate_protocol as vp  # noqa: E402

from pipeline_validator.cosim import wire  # noqa: E402

SV_OUT = REPO_ROOT / "out/coral_nexus/C017"
VERILATOR = SV_OUT / "build/verilator-5.052/bin/verilator"
HAVE_GCC = shutil.which("gcc") is not None
HAVE_DOCKER = shutil.which("docker") is not None and (
  subprocess.run(["docker", "info"], capture_output=True).returncode == 0
)
DOCKER_IMAGE = "coral-nexus-builder:c002"
HAVE_SV = (
  HAVE_DOCKER
  and VERILATOR.exists()
  and subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE], capture_output=True).returncode == 0
)


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def layout() -> dict[str, Any]:
  return json.loads((PROTO / "generated/cn_layout.json").read_text())


@pytest.fixture
def pdir(tmp_path: Path) -> Path:
  """A private copy of the protocol inputs, so mutations never touch the repository."""
  target = tmp_path / "protocol"
  shutil.copytree(PROTO, target, ignore=shutil.ignore_patterns("generated", "__pycache__"))
  return target


def _redirect(tmp_path: Path) -> dict[Path, Path]:
  out = tmp_path / "out"
  out.mkdir(exist_ok=True)
  return {
    gp.GENERATED_DIR / "cn_layout.json": out / "cn_layout.json",
    gp.GENERATED_DIR / "cn_messages.h": out / "cn_messages.h",
    gp.GENERATED_DIR / "cn_offsets.svh": out / "cn_offsets.svh",
    gp.WIRE_PY: out / "wire.py",
  }


def _mutate_schema(pdir: Path, fn: Any) -> None:
  path = pdir / gp.SCHEMA_FILE
  doc = yaml.safe_load(path.read_text())
  fn(doc)
  path.write_text(yaml.safe_dump(doc, sort_keys=False, width=1000))


def _fails(pdir: Path, tmp_path: Path, needle: str) -> None:
  with pytest.raises(gp.GenError) as exc:
    gp.generate(pdir, _redirect(tmp_path))
  assert needle in "\n".join(exc.value.errors), exc.value.errors


# ---------------------------------------------------------------------------------------------------
# generation: committed outputs are current, deterministic, and complete
# ---------------------------------------------------------------------------------------------------
def test_committed_outputs_equal_a_fresh_generation_and_csr_bindings_agree() -> None:
  assert freeze_check.verify_generated() == []
  assert csr_lint.lint() == []


def test_generation_is_byte_idempotent(tmp_path: Path) -> None:
  first = gp.generate(PROTO, _redirect(tmp_path))
  second = gp.generate(PROTO, _redirect(tmp_path))
  assert first == second
  for path, text in first.items():
    assert path.read_text() == text, "the committed file is stale"


def test_every_catalog_and_requested_message_has_a_layout(layout: dict[str, Any]) -> None:
  protocol = yaml.safe_load((PROTO / "cn_protocol_v1.yaml").read_text())
  names = {m["name"] for m in layout["messages"]}
  assert {m["name"] for m in protocol["messages"]} <= names
  for fragment in gp.REQUEST_FILES:
    request = yaml.safe_load((PROTO / "field_requests" / fragment).read_text())
    for item in (request.get("messages") or []) + (request.get("new_messages_requested") or []):
      assert (item.get("message") or item.get("name")) in names


def test_wrap_quiesce_direction_follows_the_user_decision(layout: dict[str, Any]) -> None:
  by = {m["name"]: m for m in layout["messages"]}
  assert (by["GEN_WRAP_QUIESCE"]["dir"], by["QUIESCE_RESP"]["dir"]) == ("T2G", "G2T")


def test_open_items_are_reported_not_hidden(layout: dict[str, Any]) -> None:
  items = layout["open_items"]
  assert {t["field"] for t in items["transport_only_fields"]} >= {"mem_op", "length", "page_count"}
  assert items["unconfirmed_aliases"] == ["profile_l1_gen"]
  assert "ctx_generation" in items["unresolved_widths"]
  assert layout["widths"]["derived"] == {}, "no width may be derived without a sourced rate and lifetime"


# ---------------------------------------------------------------------------------------------------
# generator refusals; every one must also leave earlier outputs untouched
# ---------------------------------------------------------------------------------------------------
def test_a_field_without_provenance_is_refused(pdir: Path, tmp_path: Path) -> None:
  def drop(doc: dict[str, Any]) -> None:
    del doc["messages"]["CAPS_RESP"]["fields"][1]["src"]

  _mutate_schema(pdir, drop)
  _fails(pdir, tmp_path, "needs exactly one provenance")


def test_a_source_needle_that_is_not_in_the_file_is_refused(pdir: Path, tmp_path: Path) -> None:
  def bend(doc: dict[str, Any]) -> None:
    doc["messages"]["CAPS_RESP"]["fields"][1]["src"] = (
      "protocol/field_requests/w2.yaml::no such text anywhere"
    )

  _mutate_schema(pdir, bend)
  _fails(pdir, tmp_path, "not found in")


def test_a_source_file_that_does_not_exist_is_refused(pdir: Path, tmp_path: Path) -> None:
  def bend(doc: dict[str, Any]) -> None:
    doc["messages"]["CAPS_RESP"]["fields"][1]["src"] = "protocol/nope.yaml::program_slots"

  _mutate_schema(pdir, bend)
  _fails(pdir, tmp_path, "does not exist")


def test_an_enum_that_drifts_from_the_validator_is_refused(pdir: Path, tmp_path: Path) -> None:
  _mutate_schema(pdir, lambda d: d["enums"]["event_status"]["members"].remove("RESET"))
  _fails(pdir, tmp_path, "event_status")
  _mutate_schema(pdir, lambda d: d["enums"]["fault_code"]["members"].reverse())
  _fails(pdir, tmp_path, "fault_code")


def test_a_member_missing_from_the_design_text_is_refused(pdir: Path, tmp_path: Path) -> None:
  _mutate_schema(pdir, lambda d: d["enums"]["ack_outcome"]["members"].append("INVENTED_OUTCOME"))
  _fails(pdir, tmp_path, "INVENTED_OUTCOME")


def test_address_like_and_hidden_fields_are_refused(pdir: Path, tmp_path: Path) -> None:
  def addr(doc: dict[str, Any]) -> None:
    doc["messages"]["QUERY_CAPS"]["fields"].append(
      {"name": "payload_addr", "kind": "uint", "transport_only": "x"}
    )

  _mutate_schema(pdir, addr)
  _fails(pdir, tmp_path, "address/opcode/PC")

  def hidden(doc: dict[str, Any]) -> None:
    doc["messages"]["RUN_BEGIN"]["fields"].append(
      {"name": "fetch_epoch", "kind": "uint", "ident": "epochs"}
    )

  shutil.copy(PROTO / gp.SCHEMA_FILE, pdir / gp.SCHEMA_FILE)
  _mutate_schema(pdir, hidden)
  _fails(pdir, tmp_path, "field_visibility hidden")


def test_a_message_outside_the_catalog_and_new_list_is_refused(pdir: Path, tmp_path: Path) -> None:
  def extra(doc: dict[str, Any]) -> None:
    doc["messages"]["SECRET_BACKDOOR"] = {
      "fields": [{"name": "run_generation", "kind": "uint", "ident": "epochs"}]
    }

  _mutate_schema(pdir, extra)
  _fails(pdir, tmp_path, "SECRET_BACKDOOR")


def test_a_catalog_message_missing_from_the_schema_is_refused(pdir: Path, tmp_path: Path) -> None:
  _mutate_schema(pdir, lambda d: d["messages"].pop("RETIRE"))
  _fails(pdir, tmp_path, "RETIRE")


def test_a_catalog_key_that_is_not_carried_is_refused(pdir: Path, tmp_path: Path) -> None:
  def cut(doc: dict[str, Any]) -> None:
    doc["messages"]["STREAM_ERROR"]["fields"] = [
      f for f in doc["messages"]["STREAM_ERROR"]["fields"] if f["name"] != "first_fault_index"
    ]

  _mutate_schema(pdir, cut)
  _fails(pdir, tmp_path, "first_fault_index")


def test_a_field_request_for_a_field_the_wire_does_not_carry_is_refused(pdir: Path, tmp_path: Path) -> None:
  def cut(doc: dict[str, Any]) -> None:
    doc["groups"]["context_key"] = [
      f for f in doc["groups"]["context_key"] if f["name"] != "ctx_generation"
    ]

  _mutate_schema(pdir, cut)
  _fails(pdir, tmp_path, "ctx_generation")


def test_an_identity_field_outside_its_domain_is_refused(pdir: Path, tmp_path: Path) -> None:
  def wrong(doc: dict[str, Any]) -> None:
    doc["messages"]["TASK_PLAN"]["fields"][0]["ident"] = "context_key"

  _mutate_schema(pdir, wrong)
  _fails(pdir, tmp_path, "not a member of identity domain")


def test_a_catalog_message_the_schema_does_not_define_is_named(pdir: Path, tmp_path: Path) -> None:
  _mutate_schema(pdir, lambda d: d["messages"].pop("STREAM_EOS"))
  _fails(pdir, tmp_path, "in the C09 catalog but missing from the schema".replace("C09", "C009"))


def test_an_unknown_direction_is_refused(pdir: Path, tmp_path: Path) -> None:
  def bend(doc: dict[str, Any]) -> None:
    for item in doc["new_messages"]:
      if item["name"] == "LEASE_GRANT":
        item["dir"] = "X2Y"

  _mutate_schema(pdir, bend)
  _fails(pdir, tmp_path, "not in ('G2T', 'T2G', 'H2G')")


def test_a_field_request_that_contradicts_the_wire_direction_is_refused(pdir: Path, tmp_path: Path) -> None:
  path = pdir / "field_requests/w2.yaml"
  doc = yaml.safe_load(path.read_text())
  doc["requests"][2]["direction"] = "G2T"  # ACK is T2G on the wire
  path.write_text(yaml.safe_dump(doc, sort_keys=False, width=1000))
  _fails(pdir, tmp_path, "requested direction G2T differs")


def test_a_requested_csr_that_is_not_declared_is_refused(pdir: Path, tmp_path: Path) -> None:
  path = pdir / "field_requests/w3.yaml"
  doc = yaml.safe_load(path.read_text())
  doc["csr_requests"][0]["name"] = "nexus.mregion.undeclared"
  path.write_text(yaml.safe_dump(doc, sort_keys=False, width=1000))
  _fails(pdir, tmp_path, "nexus.mregion.undeclared")


def test_a_wrong_direction_is_refused(pdir: Path, tmp_path: Path) -> None:
  def flip(doc: dict[str, Any]) -> None:
    for item in doc["new_messages"]:
      if item["name"] == "LEASE_GRANT":
        item["dir"] = "T2G"

  _mutate_schema(pdir, flip)
  _fails(pdir, tmp_path, "LEASE_GRANT")


def test_a_refused_run_leaves_earlier_outputs_byte_identical(pdir: Path, tmp_path: Path) -> None:
  redirect = _redirect(tmp_path)
  gp.generate(pdir, redirect)
  before = {p: _sha(p) for p in redirect.values()}
  _mutate_schema(pdir, lambda d: d["enums"]["event_status"]["members"].remove("RESET"))
  with pytest.raises(gp.GenError):
    gp.generate(pdir, redirect)
  assert {p: _sha(p) for p in redirect.values()} == before


def test_a_digest_must_be_four_words(pdir: Path, tmp_path: Path) -> None:
  def short(doc: dict[str, Any]) -> None:
    doc["groups"]["program_identity"][2]["words"] = 2

  _mutate_schema(pdir, short)
  _fails(pdir, tmp_path, "digest")


# ---------------------------------------------------------------------------------------------------
# widths: nothing is invented
# ---------------------------------------------------------------------------------------------------
def test_a_domain_that_is_neither_derived_nor_declared_underivable_is_refused() -> None:
  data = vp.load_all(PROTO)
  lifetimes = copy.deepcopy(data["lifetimes"])
  del lifetimes["domains"][0]["derivable"]
  with pytest.raises(derive_widths.WidthError):
    derive_widths.derive(data["budgets"], lifetimes)


def test_a_width_from_an_unrelated_constant_is_refused() -> None:
  data = vp.load_all(PROTO)
  lifetimes = copy.deepcopy(data["lifetimes"])
  dom = lifetimes["domains"][0]
  dom.pop("derivable")
  dom["max_in_flight"] = dom.pop("max_in_flight_candidate")
  dom["rate"] = {
    "value": 1,
    "unit": "msg_per_tick",
    "bounds": "other_domain",
    "source": {"file": "pipeline_validator/config.py"},
  }
  dom["lifetime_ticks"] = {
    "value": 100,
    "unit": "tick",
    "bounds": dom["name"],
    "source": {"file": "pipeline_validator/runtime/reset_domain.py"},
  }
  with pytest.raises(derive_widths.WidthError) as exc:
    derive_widths.derive(data["budgets"], lifetimes)
  assert "not 'txn_id'" in str(exc.value)


# ---------------------------------------------------------------------------------------------------
# CSR allocation
# ---------------------------------------------------------------------------------------------------
def _csr_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
  data = vp.load_all(PROTO)
  return copy.deepcopy(data["csr"]), copy.deepcopy(data["upstream_csr"])


def test_allocation_is_deterministic_ascending_and_avoids_every_upstream_address() -> None:
  csr, upstream = _csr_inputs()
  rows = csr_allocator.allocate(csr, upstream)
  addresses = [r["address"] for r in rows]
  assert addresses == sorted(addresses) and len(set(addresses)) == len(addresses)
  assert not {e["address"] for e in upstream["occupied"]} & set(addresses)
  assert [r["name"] for r in rows] == [c["name"] for c in csr["declared"]]
  assert rows == csr_allocator.allocate(csr, upstream)


def test_an_occupied_address_inside_the_window_is_skipped_and_named_by_lint() -> None:
  csr, upstream = _csr_inputs()
  low = csr["allocation_window"]["low"]
  upstream["occupied"].append({"address": low, "line": 1, "name": "FAKE_HOLDER", "width_bits": 12})
  rows = csr_allocator.allocate(csr, upstream)
  assert rows[0]["address"] == low + 1
  clash = [{"name": "nexus.x", "address": low}]
  errors = csr_lint.lint_table(clash, upstream, csr_allocator.window_of(csr))
  assert any("FAKE_HOLDER" in e for e in errors)


def test_allocation_fails_on_capacity_and_on_a_supplied_address() -> None:
  csr, upstream = _csr_inputs()
  csr["allocation_window"]["high"] = csr["allocation_window"]["low"] + 1
  with pytest.raises(csr_allocator.CsrAllocError) as exc:
    csr_allocator.allocate(csr, upstream)
  assert exc.value.code == "CAPACITY"
  csr, upstream = _csr_inputs()
  csr["declared"][0]["address"] = 0x7C8
  with pytest.raises(csr_allocator.CsrAllocError) as exc:
    csr_allocator.allocate(csr, upstream)
  assert exc.value.code == "ADDRESS_SUPPLIED"
  csr, upstream = _csr_inputs()
  del csr["allocation_window"]["source"]
  with pytest.raises(csr_allocator.CsrAllocError) as exc:
    csr_allocator.allocate(csr, upstream)
  assert exc.value.code == "NO_WINDOW"


def test_a_declared_name_never_moves_an_earlier_address() -> None:
  csr, upstream = _csr_inputs()
  before = {r["name"]: r["address"] for r in csr_allocator.allocate(csr, upstream)}
  csr["declared"].append(
    {
      "name": "nexus.mnew.reg",
      "owner": "W3",
      "scope": "per_context",
      "access": "unspecified",
      "source": "x",
    }
  )
  after = {r["name"]: r["address"] for r in csr_allocator.allocate(csr, upstream)}
  assert {k: after[k] for k in before} == before


def test_a_hand_edited_csr_address_in_any_generated_file_is_caught(tmp_path: Path) -> None:
  gen = tmp_path / "generated"
  gen.mkdir()
  for name in ("cn_layout.json", "cn_messages.h", "cn_offsets.svh"):
    shutil.copy(PROTO / "generated" / name, gen / name)
  wire_copy = tmp_path / "wire.py"
  shutil.copy(gp.WIRE_PY, wire_copy)
  assert csr_lint.lint(PROTO, gen, wire_copy) == []
  header = gen / "cn_messages.h"
  header.write_text(
    header.read_text().replace("CN_CSR_NEXUS_MREGION_START 0x7ca", "CN_CSR_NEXUS_MREGION_START 0x7c0")
  )
  assert any("cn_messages.h" in e and "MREGION_START" in e for e in csr_lint.lint(PROTO, gen, wire_copy))
  shutil.copy(PROTO / "generated/cn_messages.h", header)
  svh = gen / "cn_offsets.svh"
  svh.write_text(svh.read_text().replace("12'h7ca", "12'h7e0", 1))
  assert any("cn_offsets.svh" in e for e in csr_lint.lint(PROTO, gen, wire_copy))
  shutil.copy(PROTO / "generated/cn_offsets.svh", svh)
  wire_copy.write_text(wire_copy.read_text().replace("2026", "2027", 0).replace(": 1994", ": 1995", 1))
  assert any("wire.py" in e for e in csr_lint.lint(PROTO, gen, wire_copy))


# ---------------------------------------------------------------------------------------------------
# freeze checker
# ---------------------------------------------------------------------------------------------------
def test_freeze_check_detects_a_hand_edited_generated_file(tmp_path: Path) -> None:
  redirect = _redirect(tmp_path)
  gp.generate(PROTO, redirect)
  assert freeze_check.verify_generated(PROTO, redirect) == []
  target = redirect[gp.GENERATED_DIR / "cn_layout.json"]
  original = target.read_text()
  edited = original.replace('"payload_words": 1', '"payload_words": 2', 1)
  assert edited != original
  target.write_text(edited)
  errors = freeze_check.verify_generated(PROTO, redirect)
  assert any("cn_layout.json" in e and "hand-edited" in e for e in errors)
  target.write_text(original)
  header = redirect[gp.GENERATED_DIR / "cn_messages.h"]
  header.write_text(header.read_text() + "\n")
  assert any("cn_messages.h" in e for e in freeze_check.verify_generated(PROTO, redirect))
  header.unlink()
  assert any("missing" in e for e in freeze_check.verify_generated(PROTO, redirect))


def test_freeze_record_hash_mismatch_and_bad_state_are_rejected(tmp_path: Path) -> None:
  record = freeze_check.collect()
  assert record["state"] == "collected-not-frozen"
  assert freeze_check.verify(record) == []
  bad = copy.deepcopy(record)
  key = next(iter(bad["generated"]))
  bad["generated"][key] = "0" * 64
  assert any(key in e for e in freeze_check.verify(bad))
  bad = copy.deepcopy(record)
  bad["state"] = "sealed"
  assert any("state" in e for e in freeze_check.verify(bad))
  bad = copy.deepcopy(record)
  bad["inputs"]["hardware/coral_nexus/protocol/missing.yaml"] = "0" * 64
  assert any("missing" in e for e in freeze_check.verify(bad))


def test_the_required_seal_set_is_held_by_the_checker() -> None:
  # C017 shipped collect/verify only; C020 added the seal. The required input/generated sets live in
  # the checker (never in a YAML it validates); while a required capacity budget is open there is no
  # committed seal at all and both sealing and verifying say so by name.
  assert hasattr(freeze_check, "seal")
  if freeze_check.open_budgets():
    assert freeze_check.verify_seal() == [
      f"{freeze_check._rel(freeze_check.SEAL_PATH)}: the joint seal is missing"
    ]
    with pytest.raises(gp.GenError):
      freeze_check.seal()
    assert not freeze_check.SEAL_PATH.exists()
  else:
    assert freeze_check.verify_seal() == []
    record = json.loads(freeze_check.SEAL_PATH.read_text())
    assert set(record["inputs"]) == set(freeze_check.REQUIRED_INPUTS)
    assert set(record["generated"]) == set(freeze_check.REQUIRED_GENERATED)
    assert "PROTOCOL_FREEZE" not in json.dumps(record["inputs"]) + json.dumps(record["generated"])


# ---------------------------------------------------------------------------------------------------
# dedup contract (C009 reference) now covers the second ACK of a txn
# ---------------------------------------------------------------------------------------------------
def _dmsg(seq: int, mtype: str, txn: str = "t", **extra: Any) -> dict[str, Any]:
  return {
    "tile": 0,
    "channel": "CTRL",
    "dir": "G2T",
    "message_seq": seq,
    "message_type": mtype,
    "txn_id": txn,
    **extra,
  }


def test_hold_and_final_acks_of_one_txn_both_apply_and_a_repeat_does_not() -> None:
  tracker = vp.DedupTracker(vp.load_all(PROTO)["protocol"])
  tracker.deliver(_dmsg(0, "TASK_COMMIT"))
  tracker.deliver(_dmsg(0, "ACK", dir="T2G", ack_for_message_type="TASK_COMMIT"))
  tracker.deliver(_dmsg(1, "TASK_COMMIT_FINAL"))
  tracker.deliver(_dmsg(1, "ACK", dir="T2G", ack_for_message_type="TASK_COMMIT_FINAL"))
  assert tracker.effects == 4
  with pytest.raises(vp.ProtocolError) as exc:
    tracker.deliver(_dmsg(2, "ACK", dir="T2G", ack_for_message_type="TASK_COMMIT_FINAL"))
  assert exc.value.code == "DUPLICATE_STAGE"


def test_removing_the_ack_stage_field_is_a_catalog_error() -> None:
  data = vp.load_all(PROTO)
  data["protocol"]["dedup"]["stage_fields"] = {}
  errors: list[str] = []
  vp.check_dedup_rules(data["protocol"], errors)
  assert any("stage_fields" in e for e in errors)


# ---------------------------------------------------------------------------------------------------
# consumers: Python == C == SystemVerilog on one corpus
# ---------------------------------------------------------------------------------------------------
def _words_blob(header: list[int], payload: list[int], extra: bytes = b"") -> bytes:
  return b"".join(w.to_bytes(8, "little") for w in header + payload) + extra


def _corpus(layout: dict[str, Any]) -> list[tuple[tuple[int, int], bytes]]:
  """Deterministic byte streams: for each message a valid frame, then hostile variants of it."""
  cases: list[tuple[tuple[int, int], bytes]] = []
  next_seq: dict[tuple[int, int], int] = {}

  def valid_payload(spec: dict[str, Any]) -> list[int]:
    words = [0] * spec["payload_words"]
    for f in spec["fields"]:
      if f["kind"] in ("uint", "digest"):
        for k in range(f["words"]):
          words[f["offset"] + k] = 1 + k
    return words

  for spec in layout["messages"]:
    port = (spec["channel_index"], spec["dir_index"])
    seq = next_seq.get(port, 0)
    pl = valid_payload(spec)
    hdr = [spec["id"], spec["payload_words"], seq, 0]
    cases.append((port, _words_blob(hdr, pl)))
    next_seq[port] = seq + 1
    if spec["fields"] or spec["conditions"]:
      cases.append(
        (port, _words_blob([spec["id"], spec["payload_words"], seq + 1, 0], pl[:-1]))
      )  # truncated
    cases.append(
      (port, _words_blob([spec["id"], spec["payload_words"], seq + 1, 0], pl, b"\x00"))
    )  # trailing
    cases.append((port, _words_blob([spec["id"], spec["payload_words"], seq + 1, 7], pl)))  # reserved
    cases.append((port, _words_blob([spec["id"], spec["payload_words"] + 1, seq + 1, 0], pl)))  # length
    other = ((port[0] + 1) % len(layout["channels"]), port[1])
    cases.append((other, _words_blob([spec["id"], spec["payload_words"], seq + 1, 0], pl)))  # wrong port
    cases.append((port, _words_blob([spec["id"], spec["payload_words"], seq, 0], pl)))  # replay
    cases.append((port, _words_blob([spec["id"], spec["payload_words"], seq + 9, 0], pl)))  # gap
    for f in spec["fields"]:
      bad = list(pl)
      if f["kind"] == "enum":
        bad[f["offset"]] = f["limit"]
      elif f["kind"] == "bool":
        bad[f["offset"]] = 2
      elif f["kind"] == "bitset":
        bad[f["offset"]] = (f["limit"] << 1) | 1 | (1 << 63)
      else:
        continue
      cases.append((port, _words_blob([spec["id"], spec["payload_words"], seq + 1, 0], bad)))
    for c in spec["conditions"]:
      bad = list(pl)
      bad[c["a"]] = bad[c["b"]] + 3 if c["kind"] == 0 else 0
      cases.append((port, _words_blob([spec["id"], spec["payload_words"], seq + 1, 0], bad)))
    if spec["ctx_slot"] >= 0:
      bad = list(pl)
      bad[spec["ctx_slot"]] = layout["consumer_model"]["context_slots"]
      cases.append((port, _words_blob([spec["id"], spec["payload_words"], seq + 1, 0], bad)))
    if spec["ctx_gen"] >= 0:
      bad = list(pl)
      bad[spec["ctx_gen"]] = 0
      cases.append((port, _words_blob([spec["id"], spec["payload_words"], seq + 1, 0], bad)))
    if spec["txn"] >= 0:  # same txn+stage again with the right sequence: duplicate stage
      cases.append((port, _words_blob([spec["id"], spec["payload_words"], seq + 1, 0], pl)))
      next_seq[port] = seq + 2
    if spec["name"] == "ACK":
      bad = list(pl)
      bad[spec["txn"]] = 777
      cases.append((port, _words_blob([spec["id"], spec["payload_words"], next_seq[port], 0], bad)))
  cases.append(((0, 0), b""))
  cases.append(((0, 0), b"\x00" * 31))
  cases.append(((0, 0), _words_blob([len(layout["messages"]), 0, 0, 0], [])))  # unknown message
  return cases


def _python_lines(cases: list[tuple[tuple[int, int], bytes]]) -> list[str]:
  rx = wire.Rx()
  return [rx.line(*rx.deliver(port, blob)) for port, blob in cases]


def _stream(cases: list[tuple[tuple[int, int], bytes]]) -> str:
  return "".join(f"{p[0]} {p[1]} {b.hex()}\n" for p, b in cases)


@pytest.fixture(scope="module")
def corpus(layout: dict[str, Any]) -> list[tuple[tuple[int, int], bytes]]:
  return _corpus(layout)


Case = tuple[tuple[int, int], bytes]
TASK_KEY = {
  "launch_generation": 1,
  "root_launch_generation": 1,
  "dispatch_ordinal": 0,
  "task_id": 3,
  "context_handle": 0,
}


def _port(message: str) -> tuple[int, int]:
  spec = wire.MESSAGES[wire.MSG_INDEX[message]]
  return spec["channel"], spec["dir"]


def _scenarios() -> dict[str, list[Case]]:
  """Fresh-receiver scenarios: each list is delivered to a receiver with no history."""

  def _ack(for_type: str, outcome: str, seq: int) -> bytes:
    values = {
      "txn_id": 9,
      "ack_for_message_type": for_type,
      "outcome": outcome,
      "op_seq": 0,
      "wait_reason": None,
    }
    return wire.pack("ACK", values, seq)

  commit = {**TASK_KEY, "txn_id": 9, "ctx_slot": 2, "ctx_generation": 5, "pool_version": 1}

  def _commit(change: dict[str, int], seq: int) -> bytes:
    return wire.pack("TASK_COMMIT", {**commit, **change}, seq)

  final = {**TASK_KEY, "txn_id": 9, "ctx_slot": 2, "ctx_generation": 5}
  handshake: list[Case] = [
    (_port("TASK_COMMIT"), wire.pack("TASK_COMMIT", commit, 0)),
    (_port("ACK"), _ack("TASK_COMMIT", "HOLD_READY", 0)),
    (_port("TASK_COMMIT_FINAL"), wire.pack("TASK_COMMIT_FINAL", final, 1)),
    (_port("ACK"), _ack("TASK_COMMIT_FINAL", "ACCEPTED", 1)),
    (_port("ACK"), _ack("TASK_COMMIT_FINAL", "ACCEPTED", 2)),
    (_port("TASK_COMMIT_FINAL"), wire.pack("TASK_COMMIT_FINAL", final, 2)),
  ]  # fmt: skip
  old_gen = {"ctx_generation": 4}
  retire = {**TASK_KEY, **old_gen, "drain_evidence_seq": 1}
  limit = wire.DEDUP_ENTRIES
  flood: list[Case] = [
    (_port("QUERY_CAPS"), wire.pack("QUERY_CAPS", {"txn_id": n}, n)) for n in range(limit + 2)
  ]
  stale: list[Case] = [
    (_port("TASK_COMMIT"), wire.pack("TASK_COMMIT", commit, 0)),
    (_port("TASK_COMMIT"), _commit({"txn_id": 10, "ctx_generation": 4}, 1)),
    (_port("TASK_COMMIT"), _commit({"txn_id": 11, "ctx_generation": 6}, 2)),
    (_port("TASK_COMMIT"), _commit({"txn_id": 12, "ctx_slot": 3, "ctx_generation": 1}, 3)),
    # every later message of a Context carries (ctx_slot, ctx_generation) and is stale below the live one
    (_port("TASK_COMMIT_FINAL"), wire.pack("TASK_COMMIT_FINAL", {**final, **old_gen, "txn_id": 13}, 4)),
    (_port("CANCEL_TASK"), wire.pack("CANCEL_TASK", {**TASK_KEY, **old_gen, "ctx_slot": 2}, 5)),
    (_port("RETIRE_READY"), wire.pack("RETIRE_READY", {**retire, "ctx_slot": 2}, 0)),
  ]  # fmt: skip
  return {"handshake": handshake, "flood": flood, "stale": stale}


def test_scenario_outcomes_are_the_documented_ones() -> None:
  scenarios = _scenarios()
  assert [line.split()[0] for line in _python_lines(scenarios["handshake"])] == [
    "OK", "OK", "OK", "OK", "DUPLICATE_STAGE", "DUPLICATE_STAGE"
  ]  # fmt: skip
  flood = [line.split()[0] for line in _python_lines(scenarios["flood"])]
  assert flood[: wire.DEDUP_ENTRIES] == ["OK"] * wire.DEDUP_ENTRIES and flood[-2:] == ["CAPACITY"] * 2
  assert [line.split()[0] for line in _python_lines(scenarios["stale"])] == [
    "OK", "STALE_GENERATION", "OK", "OK", "STALE_GENERATION", "STALE_GENERATION", "STALE_GENERATION"
  ]  # fmt: skip


def test_the_corpus_exercises_every_error_name(corpus: list[tuple[tuple[int, int], bytes]]) -> None:
  seen = {line.split()[0] for line in _python_lines(corpus)}
  assert seen == set(wire.ERRORS), f"never exercised: {sorted(set(wire.ERRORS) - seen)}"


def test_python_pack_unpack_roundtrips_every_message(layout: dict[str, Any]) -> None:
  for spec in layout["messages"]:
    values: dict[str, Any] = {}
    for f in spec["fields"]:
      if f["kind"] == "digest":
        values[f["name"]] = (0x1122334455667788).to_bytes(32, "little")
      elif f["kind"] == "enum":
        values[f["name"]] = wire.ENUMS[f["enum"]][0] if not f["optional"] else None
      elif f["kind"] == "bitset":
        values[f["name"]] = 0
      else:
        values[f["name"]] = 5 if f["kind"] == "uint" else 1
    if spec["name"] in ("TASK_PLAN", "TASK_COMMIT", "TASK_COMMIT_FINAL"):
      values["root_launch_generation"] = values["launch_generation"]
    blob = wire.pack(spec["name"], values, 3)
    decoded = wire.unpack(blob)
    assert decoded.message == spec["name"] and decoded.message_seq == 3
    for name, value in values.items():
      assert decoded.values[name] == value or (value == 1 and decoded.values[name] is True), (
        spec["name"],
        name,
      )


def test_pack_rejects_unknown_missing_and_oversized_values() -> None:
  good = {"txn_id": 1}
  assert wire.pack("QUERY_CAPS", good)
  for values, code in (
    ({"txn_id": 1, "bogus": 2}, "FIELD_UNKNOWN"),
    ({}, "FIELD_MISSING"),
    ({"txn_id": 1 << 64}, "OVERFLOW"),
    ({"txn_id": -1}, "OVERFLOW"),
  ):
    with pytest.raises(wire.WireError) as exc:
      wire.pack("QUERY_CAPS", values)
    assert exc.value.code == code
  with pytest.raises(wire.WireError) as exc:
    wire.pack("NO_SUCH_MESSAGE", {})
  assert exc.value.code == "UNKNOWN_MSG"


@pytest.mark.skipif(not HAVE_GCC, reason="gcc not found")
def test_c_consumer_matches_python_on_the_whole_corpus(
  tmp_path: Path, corpus: list[tuple[tuple[int, int], bytes]]
) -> None:
  binary = tmp_path / "cn_rx"
  build = subprocess.run(
    ["gcc", "-std=c11", "-Wall", "-Wextra", "-Werror", f"-I{PROTO / 'generated'}", "-o", str(binary),
     str(CONSUMERS / "cn_rx.c")],
    capture_output=True,
    text=True,
  )  # fmt: skip
  assert build.returncode == 0, build.stderr
  run = subprocess.run([str(binary)], input=_stream(corpus), capture_output=True, text=True)
  assert run.returncode == 0, run.stderr
  c_lines = run.stdout.splitlines()
  py_lines = _python_lines(corpus)
  assert len(c_lines) == len(py_lines)
  mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(py_lines, c_lines)) if a != b]
  assert not mismatches, mismatches[:3]
  for name, cases in _scenarios().items():
    run = subprocess.run([str(binary)], input=_stream(cases), capture_output=True, text=True)
    assert run.stdout.splitlines() == _python_lines(cases), name


@pytest.mark.skipif(not HAVE_GCC, reason="gcc not found")
def test_a_layout_drift_in_the_generated_header_is_seen_by_the_c_consumer(
  tmp_path: Path, corpus: list[tuple[tuple[int, int], bytes]]
) -> None:
  gen = tmp_path / "gen"
  shutil.copytree(PROTO / "generated", gen)
  header = gen / "cn_messages.h"
  text = header.read_text()
  # move the first message to the other direction: a consumer built from this header must disagree with the
  # generated Python on the very first frame (OK vs WRONG_PORT) while the header still compiles
  assert '{"QUERY_CAPS", 0, 0,' in text
  text = text.replace('{"QUERY_CAPS", 0, 0,', '{"QUERY_CAPS", 0, 1,')
  header.write_text(text)
  binary = tmp_path / "cn_rx_bad"
  build = subprocess.run(
    ["gcc", "-std=c11", f"-I{gen}", "-o", str(binary), str(CONSUMERS / "cn_rx.c")],
    capture_output=True,
    text=True,
  )
  assert build.returncode == 0, build.stderr
  run = subprocess.run([str(binary)], input=_stream(corpus), capture_output=True, text=True)
  assert run.stdout.splitlines() != _python_lines(corpus), "the drift went unnoticed"


@pytest.mark.skipif(
  not HAVE_SV,
  reason="needs docker, the coral-nexus-builder:c002 image and the out/coral_nexus/C017 Verilator build",
)
def test_systemverilog_consumer_matches_python_on_the_whole_corpus(
  corpus: list[tuple[tuple[int, int], bytes]],
) -> None:
  work = SV_OUT / "test_sv"
  shutil.rmtree(work, ignore_errors=True)
  work.mkdir(parents=True)
  shutil.copy(CONSUMERS / "cn_rx.sv", work / "cn_rx.sv")
  shutil.copy(PROTO / "generated/cn_offsets.svh", work / "cn_offsets.svh")
  runs: dict[str, list[Case]] = {"corpus": corpus, **_scenarios()}
  for name, cases in runs.items():
    (work / f"{name}.txt").write_text(_stream(cases))
  script = (
    "export VERILATOR_ROOT=/out/build/verilator-5.052; cd /out/test_sv && "
    "/out/build/verilator-5.052/bin/verilator --binary --timing -Wno-fatal --top-module cn_rx "
    "-Mdir obj cn_rx.sv -o cn_rx > build.log 2>&1 && "
    + " && ".join(f"./obj/cn_rx +cases=/out/test_sv/{n}.txt > {n}.out" for n in runs)
  )
  run = subprocess.run(
    ["bash", str(H / "scripts/lib/docker_run.sh"), DOCKER_IMAGE, str(SV_OUT), "--", "bash", "-c", script],
    capture_output=True,
    text=True,
    timeout=600,
  )  # fmt: skip
  assert run.returncode == 0, (run.stdout + run.stderr)[-2000:] + (work / "build.log").read_text()[-2000:]
  for name, cases in runs.items():
    sv_lines = [ln for ln in (work / f"{name}.out").read_text().splitlines() if " eff=" in ln]
    py_lines = _python_lines(cases)
    assert len(sv_lines) == len(py_lines), name
    mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(py_lines, sv_lines)) if a != b]
    assert not mismatches, (name, mismatches[:3])
