"""Validate the G1v5 scalar-helper control ABI and its owner field requests.

What this checks, none of it a string search:

1. Every ``{file, line, needle}`` anchor in ``protocol/scalar_helper_abi_v1.json`` is READ BACK from the
   tree it names (the same whole-tree walk validate_vme_decl uses).
2. The ABI carries the REQUIRED contract held in THIS file (carrier forms, the four helper
   registrations, operand resolution rules, the descriptor record contract, the fault set, the
   K-SCALAR-ADD7 reference header). A synchronized edit that deletes a rule together with the check
   that would notice it cannot pass, because the required set is not in the JSON it constrains.
3. The owner field requests (w2/w4/w5/w6) name only CSRs that exist in ``csr_name_map.yaml``, use only
   the ABI helper kinds and never invent an address, opcode or width.
4. A carrier reference model is exercised: the two legal write forms pass, every other access shape is
   the declared protocol fault, every operand outside the declared domain is rejected, and a write to
   the sealed descriptor/event tables is refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB))

H = LIB.parents[1]
PROTOCOL = H / "protocol"
ABI = PROTOCOL / "scalar_helper_abi_v1.json"
CSR_NAMES = PROTOCOL / "csr_name_map.yaml"
W5_CONTRACT = PROTOCOL / "w5_package.contract.yaml"
REQUESTS = {o: PROTOCOL / "field_requests" / f"{o}.yaml" for o in ("w2", "w4", "w5", "w6")}

# ----------------------------------------------------------------------------------------------
# The required contract. Held here so one edit cannot weaken the ABI and its guard together.
# ----------------------------------------------------------------------------------------------
REQUIRED_FORMS = ("CSRRW x0,csr,rs1", "CSRRWI x0,csr,imm")
# helper name -> (csr, kind, owner, operand kind, resolves_to prefix)
REQUIRED_HELPERS: dict[str, tuple[str, str, str, str, str]] = {
  "mfe_submit": ("nexus.mfe.submit", "MFE_SUBMIT", "W4", "descriptor_index", "launch_descriptors"),
  "wait": ("nexus.event.wait", "WAIT", "W6", "event_index", "wait_events"),
  "signal_phase": ("nexus.phase.signal", "SIGNAL_PHASE", "W6", "phase_id", "phase enum"),
  "task_ret": ("nexus.task.ret", "TASK_RET", "W2", "code", "literal code"),
}
# helper name -> the field_request file that must carry its csr_requests entry
CSR_REQUEST_OWNER = {"mfe_submit": "w4", "wait": "w6", "signal_phase": "w6", "task_ret": "w2"}
REQUIRED_CSR_NAMES = {csr for csr, *_ in REQUIRED_HELPERS.values()}
REQUIRED_DIRECTIONS = ("l2_to_frame", "frame_to_l2")
REQUIRED_DESCRIPTOR_FIELDS = ("index", "direction", "handle", "bytes", "slot_offset", "frame_offset")
REQUIRED_PHASE_DOMAIN = {"0": "input_released", "1": "output_ready"}
REQUIRED_FAULTS: dict[str, tuple[str, bool]] = {
  "HELPER_ACCESS_FORM": ("carrier", False),
  "UNKNOWN_HELPER": ("decode", True),
  "OPERAND_OUT_OF_DOMAIN": ("operand", False),
  "DESCRIPTOR_MUTATION": ("descriptor", False),
  "HAND_WRITTEN_ADDRESS": ("build", True),
}
REQUIRED_REFERENCE = {
  "kernel_id": "K-SCALAR-ADD7",
  "kernel_version": 1,
  "kernel_class": "scalar",
  "mode_table": "none",
  "initial_method": "none",
}
BANNED_VALUE_KEYS = ("address", "opcode", "bits", "bit_position", "csr_address", "encoding", "width_bits")


def load(path: Path) -> Any:
  text = path.read_text()
  return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


# ----------------------------------------------------------------------------------------------
# source read-back (file, line, needle), hardware tree rooted
# ----------------------------------------------------------------------------------------------
def _read_line(file: str, line: int) -> str | None:
  path = (H / file).resolve()
  if path.is_file():
    lines = path.read_text().splitlines()
    return lines[line - 1] if 0 < line <= len(lines) else None
  return None


def iter_sources(node: Any, where: str = "") -> list[tuple[str, dict]]:
  """Every {file, line, needle} triple anywhere in the document is a read-back anchor."""
  found: list[tuple[str, dict]] = []
  if isinstance(node, dict):
    if {"file", "line"} <= set(node):
      found.append((where or ".", node))
      return found
    for key, value in node.items():
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
# carrier
# ----------------------------------------------------------------------------------------------
def check_carrier(abi: dict[str, Any], errors: list[str]) -> None:
  carrier = abi.get("carrier") or {}
  names = set(carrier.get("csr_name_map_entries") or [])
  if names != REQUIRED_CSR_NAMES:
    errors.append(f"carrier.csr_name_map_entries: must be exactly {sorted(REQUIRED_CSR_NAMES)}")
  if carrier.get("scope") != "per_context":
    errors.append("carrier.scope: must stay per_context")
  forms = tuple(carrier.get("forms") or ())
  if forms != REQUIRED_FORMS:
    errors.append(f"carrier.forms: must be exactly {list(REQUIRED_FORMS)}")
  if carrier.get("rd_must_be_x0") is not True:
    errors.append("carrier.rd_must_be_x0: the read leg must stay discarded")
  if carrier.get("other_access") != "protocol_fault":
    errors.append("carrier.other_access: any other access must be a protocol fault")
  if carrier.get("hand_written_address") != "forbidden":
    errors.append("carrier.hand_written_address: must stay forbidden")
  for key in ("width_bits", "bit_positions"):
    if carrier.get(key) != "unresolved":
      errors.append(f"carrier.{key}: must stay 'unresolved'; the allocator owns numbers")
  for forbidden in ("address", "opcode", "bits", "csr_address", "encoding"):
    if forbidden in carrier:
      errors.append(f"carrier.{forbidden}: a number or encoding was invented here (INVENTED_VALUE)")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def check_helpers(abi: dict[str, Any], errors: list[str]) -> None:
  helpers = abi.get("helpers")
  if not isinstance(helpers, list):
    errors.append("helpers: must be a list")
    return
  seen: dict[str, dict] = {}
  for i, helper in enumerate(helpers):
    if not isinstance(helper, dict):
      errors.append(f"helpers[{i}]: not an object")
      continue
    name = helper.get("name")
    if not isinstance(name, str):
      errors.append(f"helpers[{i}]: name must be a string")
      continue
    if name in seen:
      errors.append(f"helpers[{i}]: {name} registered twice")
    seen[name] = helper
    want = REQUIRED_HELPERS.get(name)
    if want is None:
      errors.append(f"helpers[{i}]: {name!r} is not in the validator's required set")
      continue
    csr, kind, owner, operand_kind, resolves = want
    if helper.get("csr") != csr:
      errors.append(f"helpers[{i}].{name}.csr: {helper.get('csr')!r} != {csr!r}")
    if helper.get("kind") != kind:
      errors.append(f"helpers[{i}].{name}.kind: {helper.get('kind')!r} != {kind!r}")
    if helper.get("owner") != owner:
      errors.append(f"helpers[{i}].{name}.owner: {helper.get('owner')!r} != {owner!r}")
    operand = helper.get("operand") or {}
    if operand.get("kind") != operand_kind:
      errors.append(f"helpers[{i}].{name}.operand.kind: {operand.get('kind')!r} != {operand_kind!r}")
    if not str(operand.get("resolves_to") or "").startswith(resolves):
      errors.append(
        f"helpers[{i}].{name}.operand.resolves_to: {operand.get('resolves_to')!r} does not "
        f"resolve into {resolves}"
      )
    if operand.get("out_of_domain") != "OPERAND_OUT_OF_DOMAIN":
      errors.append(f"helpers[{i}].{name}.operand: out-of-domain must be OPERAND_OUT_OF_DOMAIN")
    if not str(helper.get("side_effect") or "").strip():
      errors.append(f"helpers[{i}].{name}: a helper must state its observable side effect")
    if helper.get("ordering") != "program order within the Context":
      errors.append(f"helpers[{i}].{name}: ordering must be Context program order")
    for banned in BANNED_VALUE_KEYS:
      if banned in helper:
        errors.append(f"helpers[{i}].{name}.{banned}: invented value (INVENTED_VALUE)")
  for name in REQUIRED_HELPERS:
    if name not in seen:
      errors.append(f"helpers: required helper {name} is missing")
  # operand domains that are enumerations
  by_name = seen
  phase_domain = ((by_name.get("signal_phase") or {}).get("operand") or {}).get("domain")
  if phase_domain != REQUIRED_PHASE_DOMAIN:
    errors.append(f"helpers.signal_phase.operand.domain: {phase_domain!r} != {REQUIRED_PHASE_DOMAIN}")
  ret_domain = ((by_name.get("task_ret") or {}).get("operand") or {}).get("domain")
  if ret_domain != {"0": "normal retire / drain"}:
    errors.append(f"helpers.task_ret.operand.domain: {ret_domain!r} must be exactly code 0")


# ----------------------------------------------------------------------------------------------
# operand rules + descriptor record
# ----------------------------------------------------------------------------------------------
def check_operand_rules(abi: dict[str, Any], errors: list[str]) -> None:
  rules = abi.get("operand_rules") or {}
  if rules.get("index_scope") is None or "slot-local" not in str(rules.get("index_scope")):
    errors.append("operand_rules.index_scope: indices must stay slot-local")
  if "never masked or truncated" not in str(rules.get("no_address")):
    errors.append("operand_rules.no_address: a full address must be rejected, never masked")
  if "sealed" not in str(rules.get("immutable_tables")):
    errors.append("operand_rules.immutable_tables: tables must be sealed image state")


def check_descriptor_record(abi: dict[str, Any], errors: list[str]) -> None:
  record = abi.get("descriptor_record") or {}
  fields = tuple(record.get("required_fields") or ())
  if fields != REQUIRED_DESCRIPTOR_FIELDS:
    errors.append(
      f"descriptor_record.required_fields: {list(fields)} != {list(REQUIRED_DESCRIPTOR_FIELDS)}"
    )
  directions = tuple(record.get("directions") or ())
  if directions != REQUIRED_DIRECTIONS:
    errors.append(f"descriptor_record.directions: {list(directions)} != {list(REQUIRED_DIRECTIONS)}")
  if record.get("scope") != "slot_local":
    errors.append("descriptor_record.scope: must stay slot_local")
  if record.get("immutable") is not True:
    errors.append("descriptor_record.immutable: must stay true")
  if record.get("addressing") != "index_only":
    errors.append(f"descriptor_record.addressing: {record.get('addressing')!r} != 'index_only'")
  if record.get("mutation") != "reject":
    errors.append("descriptor_record.mutation: a kernel-side write must be rejected")
  if record.get("patch") != "patch_mask_whitelist_only":
    errors.append("descriptor_record.patch: only patch_mask fields may change")
  if "never written by the kernel" not in str(record.get("no_full_addresses")):
    errors.append("descriptor_record.no_full_addresses: provenance is owner-supplied, not kernel-written")
  coverage = record.get("coverage")
  if not isinstance(coverage, dict):
    errors.append(
      "descriptor_record.coverage: the use invariant (every descriptor submit+wait, every "
      "phase once, exactly one task_ret) is required"
    )
  else:
    # presence is not enough: each key must carry its semantic needle or the invariant is
    # silently weakened; the enforcing copy is scalar.coverage in the w5 package contract
    needles = {
      "authority": ("scalar.coverage",),
      "descriptor_use": ("exactly one mfe_submit", "exactly one wait"),
      "signal_use": ("exactly once",),
      "retire_use": ("exactly one task_ret",),
    }
    for key, wanted in needles.items():
      text = str(coverage.get(key) or "")
      if not text.strip():
        errors.append(f"descriptor_record.coverage.{key}: required")
        continue
      for needle in wanted:
        if needle not in text:
          errors.append(f"descriptor_record.coverage.{key}: must state {needle!r}")

# ----------------------------------------------------------------------------------------------
# faults + reference kernel
# ----------------------------------------------------------------------------------------------
def check_faults(abi: dict[str, Any], errors: list[str]) -> None:
  faults = abi.get("faults")
  if not isinstance(faults, list):
    errors.append("faults: must be a list")
    return
  got: dict[str, dict] = {}
  for i, fault in enumerate(faults):
    code = fault.get("code")
    if code in got:
      errors.append(f"faults[{i}]: {code} appears twice")
    got[code] = fault
    want = REQUIRED_FAULTS.get(code)
    if want is None:
      errors.append(f"faults[{i}]: {code} is not in the validator's required set")
      continue
    group, static = want
    if fault.get("group") != group:
      errors.append(f"faults[{i}].{code}.group: {fault.get('group')!r} != {group!r}")
    if bool(fault.get("static")) != static:
      errors.append(f"faults[{i}].{code}.static: must be {static}")
    if not str(fault.get("text") or "").strip():
      errors.append(f"faults[{i}].{code}: needs a description")
  for code in REQUIRED_FAULTS:
    if code not in got:
      errors.append(f"faults: required fault {code} is missing")


def check_reference_kernel(abi: dict[str, Any], errors: list[str]) -> None:
  ref = abi.get("reference_kernel") or {}
  for key, want in REQUIRED_REFERENCE.items():
    if ref.get(key) != want:
      errors.append(f"reference_kernel.{key}: {ref.get(key)!r} != {want!r}")
  region = ref.get("region") or {}
  if region.get("start") is not False or region.get("end") is not False or region.get("chains") != []:
    errors.append("reference_kernel.region: G2 uses no region helpers (start/end false, chains empty)")


# ----------------------------------------------------------------------------------------------
# owner field requests + csr_name_map
# ----------------------------------------------------------------------------------------------
def check_csr_name_map(errors: list[str]) -> None:
  doc = load(CSR_NAMES)
  declared = {item["name"]: item for item in doc["declared"]}
  for _helper, (csr, _kind, owner, *_rest) in REQUIRED_HELPERS.items():
    item = declared.get(csr)
    if item is None:
      errors.append(f"csr_name_map: {csr} is not declared")
      continue
    if item.get("owner") != owner:
      errors.append(f"csr_name_map.{csr}.owner: {item.get('owner')!r} != {owner!r}")
    if "address" in item:
      errors.append(f"csr_name_map.{csr}: an address was assigned here (INVENTED_VALUE)")


def check_field_requests(errors: list[str]) -> None:
  docs = {owner: load(path) for owner, path in REQUESTS.items()}
  for name, (csr, kind, _owner, *_rest) in REQUIRED_HELPERS.items():
    file_owner = CSR_REQUEST_OWNER[name]
    entries = docs[file_owner].get("csr_requests") or []
    hit = [e for e in entries if e.get("name") == csr]
    if not hit:
      errors.append(f"field_requests/{file_owner}.yaml: csr_requests lacks {csr}")
      continue
    entry = hit[0]
    if entry.get("kind") != kind:
      errors.append(f"field_requests/{file_owner}.yaml: {csr}.kind {entry.get('kind')!r} != {kind!r}")
    for banned in (*BANNED_VALUE_KEYS, "bits"):
      if banned in entry:
        errors.append(f"field_requests/{file_owner}.yaml: {csr}.{banned} is an invented value")
  w5 = docs["w5"]
  helpers = {h.get("name"): h for h in w5.get("helpers") or []}
  for name, (csr, *_rest) in REQUIRED_HELPERS.items():
    helper = helpers.get(name)
    if helper is None:
      errors.append(f"field_requests/w5.yaml: helpers lacks {name}")
      continue
    if helper.get("csr") != csr:
      errors.append(f"field_requests/w5.yaml: helper {name}.csr {helper.get('csr')!r} != {csr!r}")
    for banned in ("address", "bits", "opcode", "width_bits"):
      if banned in helper:
        errors.append(f"field_requests/w5.yaml: helper {name}.{banned} is an invented value")


# ----------------------------------------------------------------------------------------------
# carrier reference model: the access rules the ABI states, exercised not asserted
# ----------------------------------------------------------------------------------------------
class CarrierError(Exception):
  def __init__(self, code: str, detail: str) -> None:
    super().__init__(f"{code}: {detail}")
    self.code = code


class Carrier:
  """Reference of the ABI carrier: CSR accesses, operand domains, sealed tables."""

  def __init__(self, abi: dict[str, Any]) -> None:
    self.abi = abi
    self.helpers = {h["name"]: h for h in abi["helpers"]}
    self.csr_to_helper = {h["csr"]: h["name"] for h in abi["helpers"]}
    self.forms = tuple(abi["carrier"]["forms"])
    # per-kernel operand tables (the binding supplies them; the model uses the K-SCALAR-ADD7 contract)
    self.tables = {"launch_descriptors": 2, "wait_events": 2}

  def _operand_ok(self, name: str, operand: Any) -> None:
    helper = self.helpers[name]
    dom = helper["operand"]
    kind = dom["kind"]
    if kind == "phase_id":
      if str(operand) not in dom["domain"]:
        raise CarrierError("OPERAND_OUT_OF_DOMAIN", f"{name}: {operand!r} not a phase id")
      return
    if kind == "code":
      if str(operand) not in dom["domain"]:
        raise CarrierError("OPERAND_OUT_OF_DOMAIN", f"{name}: {operand!r} is not code 0")
      return
    index = operand if isinstance(operand, int) and not isinstance(operand, bool) else -1
    table = "launch_descriptors" if kind == "descriptor_index" else "wait_events"
    if not 0 <= index < self.tables[table]:
      raise CarrierError("OPERAND_OUT_OF_DOMAIN", f"{name}: index {operand!r} outside {table}")

  def access(self, form: str, csr: str, rd: str, operand: Any) -> str:
    if csr not in self.csr_to_helper:
      raise CarrierError("UNKNOWN_HELPER", f"{csr} is not a helper CSR")
    name = self.csr_to_helper[csr]
    if form not in self.forms:
      raise CarrierError("HELPER_ACCESS_FORM", f"{form} is not a declared carrier form")
    if rd != "x0":
      raise CarrierError("HELPER_ACCESS_FORM", f"rd {rd!r} must be x0")
    if isinstance(operand, str) and operand.startswith(("0x", "0X")):
      raise CarrierError("HAND_WRITTEN_ADDRESS", f"{name}: address literal {operand!r}")
    self._operand_ok(name, operand)
    return name

  def write_table(self, table: str) -> None:
    if table in ("launch_descriptors", "wait_events"):
      raise CarrierError("DESCRIPTOR_MUTATION", f"{table} is sealed image state")
    raise CarrierError("UNKNOWN_HELPER", f"{table} is not a helper table")


def run_scenarios(abi: dict[str, Any]) -> list[str]:
  """Exercise the card's carrier rules on the reference model. Returns a list of failures."""
  bad: list[str] = []
  model = Carrier(abi)

  def need(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
      bad.append(f"scenario {name}: {detail or 'failed'}")

  def expect(code: str, fn: Any, *args: Any) -> str | None:
    try:
      fn(*args)
    except CarrierError as exc:
      return None if exc.code == code else f"raised {exc.code}, expected {code}"
    return f"no fault, expected {code}"

  # legal carrier forms pass and return the helper name; a raised fault here is a broken acceptance,
  # recorded as a scenario failure rather than allowed to escape
  for name, operand in (
    ("mfe_submit", 0),
    ("mfe_submit", 1),
    ("wait", 0),
    ("wait", 1),
    ("signal_phase", "0"),
    ("signal_phase", "1"),
    ("task_ret", "0"),
    ("task_ret", 0),
  ):
    csr = model.helpers[name]["csr"]
    for form in model.forms:
      try:
        got = model.access(form, csr, "x0", operand)
      except CarrierError as exc:
        need(f"{name}_{form.split()[0]}_accepted", False, f"raised {exc.code}")
        continue
      need(f"{name}_{form.split()[0]}_accepted", got == name, got)

  # every other access shape is the declared protocol fault
  csr = "nexus.mfe.submit"
  for form in ("CSRRS x1,csr,rs1", "CSRRC x0,csr,rs1", "CSRRS x0,csr,rs1", "LW x1,0(csr)", "SW x1,0(csr)"):
    err = expect("HELPER_ACCESS_FORM", model.access, form, csr, "x0", 0)
    need(f"other_access_{form.split()[0]}", err is None, err or "")
  need("rd_not_x0", expect("HELPER_ACCESS_FORM", model.access, "CSRRW x1,csr,rs1", csr, "x1", 0) is None)
  need(
    "rd_x1_csrrwi", expect("HELPER_ACCESS_FORM", model.access, "CSRRWI x1,csr,imm", csr, "x1", 0) is None
  )
  need(
    "unknown_csr", expect("UNKNOWN_HELPER", model.access, model.forms[0], "nexus.bogus", "x0", 0) is None
  )
  need(
    "address_operand",
    expect("HAND_WRITTEN_ADDRESS", model.access, model.forms[0], csr, "x0", "0x7cf00000") is None,
  )

  # operand domains: indices beyond the sealed tables and out-of-domain literals are rejected
  for name, operand in (
    ("mfe_submit", 2),
    ("mfe_submit", -1),
    ("mfe_submit", "1"),
    ("wait", 2),
    ("wait", -1),
    ("signal_phase", "2"),
    ("signal_phase", 2),
    ("task_ret", "1"),
    ("task_ret", 1),
  ):
    err = expect(
      "OPERAND_OUT_OF_DOMAIN", model.access, model.forms[0], model.helpers[name]["csr"], "x0", operand
    )
    need(f"{name}_{operand}_domain", err is None, err or "")

  # the sealed tables cannot be written
  for table in ("launch_descriptors", "wait_events"):
    need(f"{table}_sealed", expect("DESCRIPTOR_MUTATION", model.write_table, table) is None)
  return bad


# ----------------------------------------------------------------------------------------------
# whole contract
# ----------------------------------------------------------------------------------------------
def validate() -> dict[str, Any]:
  """Validate the committed ABI against the committed name map and owner field requests.

  Tests that want private copies monkeypatch the module-level ABI / CSR_NAMES / REQUESTS paths or
  call ``validate_doc`` on a mutated document.
  """
  return validate_doc(load(ABI))


def validate_doc(abi: Any) -> dict[str, Any]:
  errors: list[str] = []
  if not isinstance(abi, dict):
    return {
      "schema": "cn_scalar_helper_report/v1",
      "valid": False,
      "errors": ["abi: root must be a mapping"],
      "anchors_read_back": 0,
      "helpers": [],
    }

  def guarded(label: str, fn: Any, *args: Any) -> Any:
    try:
      return fn(*args)
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
      errors.append(f"{label}: document is malformed ({type(exc).__name__}: {exc})")
      return None

  if abi.get("schema") != "cn_scalar_helper_abi/v1":
    errors.append("abi: schema must be cn_scalar_helper_abi/v1")
  if abi.get("status") != "draft":
    errors.append("abi: status must stay draft (sealing is the freeze record's job)")
  anchors = guarded("sources", check_sources, abi, "abi", errors) or 0
  guarded("carrier", check_carrier, abi, errors)
  guarded("helpers", check_helpers, abi, errors)
  guarded("operand rules", check_operand_rules, abi, errors)
  guarded("descriptor record", check_descriptor_record, abi, errors)
  guarded("faults", check_faults, abi, errors)
  guarded("reference kernel", check_reference_kernel, abi, errors)
  guarded("csr_name_map", check_csr_name_map, errors)
  guarded("field requests", check_field_requests, errors)
  scenarios = guarded("scenarios", run_scenarios, abi) or []
  errors.extend(scenarios)
  return {
    "schema": "cn_scalar_helper_report/v1",
    "valid": not errors,
    "errors": errors,
    "anchors_read_back": anchors,
    "helpers": sorted(h.get("name") for h in abi.get("helpers") or [] if isinstance(h, dict)),
  }


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  report = validate()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / "scalar_helper_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  if not report["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
