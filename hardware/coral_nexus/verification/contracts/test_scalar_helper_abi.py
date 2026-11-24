"""G1v5 scalar-helper ABI tests (cn_scalar_helper_abi/v1).

What is proven here, and how:

* the committed ABI validates clean and every {file, line, needle} anchor is read back;
* removing a helper, a carrier form, an operand rule, a descriptor field or a fault, widening an
  operand domain, mutating a source line or inventing a number each fails with the named class;
* the owner field requests (w2/w4/w5/w6) carry only declared CSRs, the ABI kinds and no invented
  values; a deleted request or an invented address is refused;
* the generated artifacts carry the four helper CSRs exactly once each and the frozen msg_type
  indices 0..47 (v3) plus TICKET_ROLLBACK == 48 (v4) never move -- a shortened or re-fingerprinted
  index_reservation is refused by the generator;
* the seal covers the ABI file, so a post-seal edit invalidates the seal rather than slipping through.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
PROTO = H / "protocol"
sys.path.insert(0, str(H / "scripts/lib"))
sys.path.insert(0, str(PROTO))
sys.path.insert(0, str(REPO_ROOT))

import freeze_check as fc  # noqa: E402
import gen_protocol as gp  # noqa: E402
import validate_scalar_helper as vsh  # noqa: E402

from pipeline_validator.cosim import wire  # noqa: E402

ABI_PATH = PROTO / "scalar_helper_abi_v1.json"
SCALAR_CSRS = ("nexus.mfe.submit", "nexus.event.wait", "nexus.phase.signal", "nexus.task.ret")
SCALAR_HELPERS = ("mfe_submit", "wait", "signal_phase", "task_ret")


@pytest.fixture
def abi() -> dict[str, Any]:
  return json.loads(ABI_PATH.read_text())


def caught(doc: dict[str, Any], mutate: Any, needle: str) -> None:
  copy_doc = copy.deepcopy(doc)
  mutate(copy_doc)
  report = vsh.validate_doc(copy_doc)
  assert not report["valid"]
  assert any(needle in e for e in report["errors"]), report["errors"]


# ---------------------------------------------------------------------------------------------------
# committed tree
# ---------------------------------------------------------------------------------------------------
def test_the_committed_abi_validates_clean() -> None:
  report = vsh.validate()
  assert report["valid"], report["errors"]
  assert report["anchors_read_back"] >= 9
  assert report["helpers"] == sorted(SCALAR_HELPERS)


def test_the_abi_is_a_sealed_input() -> None:
  assert "hardware/coral_nexus/protocol/scalar_helper_abi_v1.json" in fc.REQUIRED_INPUTS


# ---------------------------------------------------------------------------------------------------
# carrier mutations
# ---------------------------------------------------------------------------------------------------
def test_carrier_mutations_are_refused(abi: dict[str, Any]) -> None:
  caught(abi, lambda d: d["carrier"]["forms"].pop(0), "carrier.forms")
  caught(abi, lambda d: d["carrier"].__setitem__("rd_must_be_x0", False), "rd_must_be_x0")
  caught(abi, lambda d: d["carrier"].__setitem__("other_access", "ignored"), "other_access")
  caught(abi, lambda d: d["carrier"].__setitem__("hand_written_address", "allowed"), "hand_written")
  caught(abi, lambda d: d["carrier"].__setitem__("width_bits", 32), "width_bits")
  caught(abi, lambda d: d["carrier"].__setitem__("address", "0x7d0"), "INVENTED_VALUE")
  caught(abi, lambda d: d["carrier"]["csr_name_map_entries"].pop(), "csr_name_map_entries")
  caught(abi, lambda d: d["carrier"].__setitem__("scope", "global"), "scope")


# ---------------------------------------------------------------------------------------------------
# helper mutations
# ---------------------------------------------------------------------------------------------------
def test_helper_mutations_are_refused(abi: dict[str, Any]) -> None:
  caught(abi, lambda d: d["helpers"].pop(0), "required helper")
  caught(abi, lambda d: d["helpers"][0].__setitem__("csr", "nexus.bogus"), "csr")
  caught(abi, lambda d: d["helpers"][0].__setitem__("kind", "WAIT"), "kind")
  caught(abi, lambda d: d["helpers"][0].__setitem__("owner", "W6"), "owner")
  caught(abi, lambda d: d["helpers"][0]["operand"].__setitem__("kind", "address"), "operand.kind")
  caught(
    abi,
    lambda d: d["helpers"][0]["operand"].__setitem__("resolves_to", "global_descriptors[index]"),
    "resolves_to",
  )
  caught(
    abi,
    lambda d: d["helpers"][0]["operand"].__setitem__("out_of_domain", "IGNORE"),
    "OPERAND_OUT_OF_DOMAIN",
  )
  caught(abi, lambda d: d["helpers"][2]["operand"].__setitem__("domain", {"0": "x", "2": "y"}), "domain")
  caught(abi, lambda d: d["helpers"][3]["operand"].__setitem__("domain", {"0": "ok", "1": "x"}), "domain")
  caught(abi, lambda d: d["helpers"][0].__setitem__("width_bits", 32), "INVENTED_VALUE")
  caught(abi, lambda d: d["helpers"][0].__setitem__("ordering", "any order"), "ordering")


# ---------------------------------------------------------------------------------------------------
# descriptor record + faults + reference kernel
# ---------------------------------------------------------------------------------------------------
def test_descriptor_record_mutations_are_refused(abi: dict[str, Any]) -> None:
  caught(abi, lambda d: d["descriptor_record"]["required_fields"].remove("index"), "required_fields")
  caught(abi, lambda d: d["descriptor_record"].__setitem__("immutable", False), "immutable")
  caught(abi, lambda d: d["descriptor_record"].__setitem__("scope", "global"), "slot_local")
  caught(abi, lambda d: d["descriptor_record"].__setitem__("addressing", "absolute"), "index_only")
  caught(abi, lambda d: d["descriptor_record"].__setitem__("mutation", "allowed"), "mutation")
  caught(abi, lambda d: d["descriptor_record"].__setitem__("patch", "any"), "patch_mask")
  caught(abi, lambda d: d["descriptor_record"].pop("coverage"), "coverage")
  caught(
    abi,
    lambda d: d["descriptor_record"]["coverage"].pop("descriptor_use"),
    "coverage.descriptor_use",
  )
  caught(
    abi,
    lambda d: d["descriptor_record"]["coverage"].__setitem__("descriptor_use", "duplicates allowed"),
    "must state 'exactly one mfe_submit'",
  )
  caught(abi, lambda d: d["descriptor_record"].__setitem__("directions", ["in", "out"]), "directions")


def test_fault_mutations_are_refused(abi: dict[str, Any]) -> None:
  caught(abi, lambda d: d["faults"].pop(0), "required fault")
  caught(abi, lambda d: d["faults"][0].__setitem__("static", True), "static")
  caught(abi, lambda d: d["faults"][0].__setitem__("group", "decode"), "group")


def test_reference_kernel_mutations_are_refused(abi: dict[str, Any]) -> None:
  caught(abi, lambda d: d["reference_kernel"].__setitem__("mode_table", "reference"), "mode_table")
  caught(abi, lambda d: d["reference_kernel"].__setitem__("initial_method", "vtzero"), "initial")
  caught(abi, lambda d: d["reference_kernel"]["region"].__setitem__("start", True), "region")
  caught(abi, lambda d: d["reference_kernel"]["region"]["chains"].append("c0"), "region")


# ---------------------------------------------------------------------------------------------------
# sources are read back
# ---------------------------------------------------------------------------------------------------
def test_a_mutated_source_line_is_caught(abi: dict[str, Any]) -> None:
  def drift(doc: dict[str, Any]) -> None:
    doc["helpers"][0]["one_descriptor_one_transaction"]["line"] = 999

  caught(abi, drift, "SOURCE_FILE_MISSING")


def test_a_missing_source_anchor_is_caught(abi: dict[str, Any]) -> None:
  def drop(doc: dict[str, Any]) -> None:
    doc["helpers"][0]["one_descriptor_one_transaction"].pop("needle")

  caught(abi, drop, "MISSING_SOURCE")


# ---------------------------------------------------------------------------------------------------
# field requests: the four owners name only declared CSRs and invent nothing
# ---------------------------------------------------------------------------------------------------
def _mutate_request(owner: str, mutate: Any) -> dict[str, Any]:
  path = PROTO / "field_requests" / f"{owner}.yaml"
  doc = yaml.safe_load(path.read_text())
  mutate(doc)
  return doc


def test_owner_csr_requests_carry_the_new_names() -> None:
  w4 = _mutate_request("w4", lambda d: None)
  w6 = _mutate_request("w6", lambda d: None)
  w2 = _mutate_request("w2", lambda d: None)
  w5 = _mutate_request("w5", lambda d: None)
  csr_req = lambda doc, name: [e for e in doc.get("csr_requests") or [] if e.get("name") == name]
  assert csr_req(w4, "nexus.mfe.submit")[0]["kind"] == "MFE_SUBMIT"
  assert csr_req(w6, "nexus.event.wait")[0]["kind"] == "WAIT"
  assert csr_req(w6, "nexus.phase.signal")[0]["kind"] == "SIGNAL_PHASE"
  assert csr_req(w2, "nexus.task.ret")[0]["kind"] == "TASK_RET"
  helpers = {h["name"]: h for h in w5["helpers"]}
  for name, csr in zip(SCALAR_HELPERS, SCALAR_CSRS):
    assert helpers[name]["csr"] == csr


def test_an_owner_request_for_an_undeclared_csr_fails_generation(tmp_path: Path) -> None:
  import shutil

  pdir = tmp_path / "protocol"
  shutil.copytree(PROTO, pdir, ignore=shutil.ignore_patterns("generated", "__pycache__"))
  doc = yaml.safe_load((pdir / "field_requests/w4.yaml").read_text())
  doc["csr_requests"][0]["name"] = "nexus.mfe.undeclared"
  (pdir / "field_requests/w4.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, width=1000))
  with pytest.raises(gp.GenError) as exc:
    gp.build_model(pdir)
  assert "not declared in csr_name_map.yaml" in "\n".join(exc.value.errors)


def test_an_invented_address_in_a_request_is_refused(abi: dict[str, Any]) -> None:
  request = yaml.safe_load((PROTO / "field_requests/w5.yaml").read_text())
  for helper in request["helpers"]:
    if helper["name"] == "mfe_submit":
      helper["address"] = "0x7d0"
  errors: list[str] = []
  docs = {
    o: yaml.safe_load((PROTO / "field_requests" / f"{o}.yaml").read_text()) for o in ("w2", "w4", "w6")
  }
  docs["w5"] = request
  # exercise the check against the mutated doc set
  orig_requests = dict(vsh.REQUESTS)
  try:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
      tmp = Path(td)
      for owner, doc in docs.items():
        (tmp / f"{owner}.yaml").write_text(yaml.safe_dump(doc))
      vsh.REQUESTS = {o: tmp / f"{o}.yaml" for o in ("w2", "w4", "w5", "w6")}
      vsh.check_field_requests(errors)
  finally:
    vsh.REQUESTS = orig_requests
  assert any("invented value" in e for e in errors), errors


# ---------------------------------------------------------------------------------------------------
# generated artifacts: helper CSRs allocated, frozen indices never move
# ---------------------------------------------------------------------------------------------------
def test_the_four_scalar_csrs_are_allocated_in_declared_order() -> None:
  layout = json.loads((PROTO / "generated/cn_layout.json").read_text())
  by_name = {c["name"]: c["address"] for c in layout["csr"]}
  declared = [d["name"] for d in yaml.safe_load((PROTO / "csr_name_map.yaml").read_text())["declared"]]
  for csr in SCALAR_CSRS:
    assert csr in by_name, f"{csr} missing from generated layout"
  addrs = [by_name[c] for c in SCALAR_CSRS]
  assert len(set(addrs)) == 4
  assert all(0x7C8 <= a <= 0x7DB for a in addrs)
  assert declared[-4:] == list(SCALAR_CSRS)


def test_wire_py_and_header_carry_the_same_bindings() -> None:
  import re

  header = (PROTO / "generated/cn_messages.h").read_text()
  header_addr = {n: int(v, 16) for n, v in re.findall(r"#define CN_CSR_(\w+) 0x([0-9a-f]+)", header)}
  for name in SCALAR_CSRS:
    macro = "NEXUS_" + name.replace("nexus.", "").upper().replace(".", "_")
    assert macro in header_addr, f"{macro} missing from cn_messages.h"
    assert name in wire.CSR, f"{name} missing from wire.py"
    assert header_addr[macro] == wire.CSR[name], (
      f"{name}: header {header_addr[macro]:#x} != wire {wire.CSR[name]:#x}"
    )


def test_frozen_msg_type_indices_never_move() -> None:
  assert wire.MSG_INDEX["PROGRAM_REGISTER"] == 47
  assert wire.MSG_INDEX["TICKET_ROLLBACK"] == 48
  assert len(wire.MSG_INDEX) == 49


def test_the_v4_reservation_rejects_a_reverted_reserved_list(tmp_path: Path) -> None:
  import shutil

  pdir = tmp_path / "protocol"
  shutil.copytree(PROTO, pdir, ignore=shutil.ignore_patterns("generated", "__pycache__"))
  schema = yaml.safe_load((pdir / gp.SCHEMA_FILE).read_text())
  res = schema["index_reservation"]
  res["reserved"].remove("TICKET_ROLLBACK")  # a v4->v3 revert attempt on the reservation alone
  res["reserved_fingerprint"] = "aeeefc19e3a165b0522baf269476516bd6b538f7b813278df048a79ccc80b9c3"
  (pdir / gp.SCHEMA_FILE).write_text(yaml.safe_dump(schema, sort_keys=False, width=1000))
  with pytest.raises(gp.GenError) as exc:
    gp.build_model(pdir)
  assert any("v4" in e or "shorter" in e or "sealed position 48" in e for e in exc.value.errors), (
    exc.value.errors
  )


def test_a_reordered_reserved_list_is_refused(tmp_path: Path) -> None:
  import hashlib
  import shutil

  pdir = tmp_path / "protocol"
  shutil.copytree(PROTO, pdir, ignore=shutil.ignore_patterns("generated", "__pycache__"))
  schema = yaml.safe_load((pdir / gp.SCHEMA_FILE).read_text())
  res = schema["index_reservation"]
  res["reserved"][47], res["reserved"][48] = res["reserved"][48], res["reserved"][47]
  res["reserved_fingerprint"] = hashlib.sha256("\n".join(res["reserved"]).encode()).hexdigest()
  (pdir / gp.SCHEMA_FILE).write_text(yaml.safe_dump(schema, sort_keys=False, width=1000))
  with pytest.raises(gp.GenError) as exc:
    gp.build_model(pdir)
  assert any("v4" in e or "sealed position" in e for e in exc.value.errors), exc.value.errors
