"""Validate the C011 numeric contract (w3_numeric_profile_v1.yaml) and its vectors.

Checks, none of which is a grep for a constant or a "close enough" float comparison:

1. Every `source: {file, line, anchor}` in the profile is READ BACK from the pinned tree.
2. The generator parameters are RE-DERIVED from the format table by `numeric_ref.derive_parameters`
   and compared to what the profile states.
3. The cvfpu round-mode encodings the profile cites are read from the vendor copy when it is
   present; its sha256 must match the profile's `read_sha256`, otherwise the check is reported
   as skipped (never silently passed).
4. Every rule has at least one directed vector, and every vector belongs to a rule.
5. Every vector is run through the reference model and compared BIT FOR BIT; ablation vectors must
   produce a DIFFERENT result under the named wrong implementation.
6. The committed real-RTL record (C006, byte-identical to the pinned copy) is compared cell by
   cell against the reference model, including undisturbed cells outside the tile.
7. The committed C011 probe record (a real RTL run of the directed vectors) must agree with both
   the model and the hand-derived expected bits, and must have run the baseline's ELF.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB))

import numeric_probe as probe  # noqa: E402
import numeric_ref as nr  # noqa: E402

H = LIB.parents[1]
REPO = LIB.parents[3]
UPSTREAM_HDL = H / "upstream" / "coralnpu" / "hdl"
PROFILE = H / "protocol" / "w3_numeric_profile_v1.yaml"
VECTORS = H / "verification" / "numerics" / "known_vectors.json"
RECORD = H / "verification" / "numerics" / "upstream_probe" / "c006_bf16_kat_record.json"
PROBE_RECORD = H / "verification" / "numerics" / "upstream_probe" / "c011_numeric_probe_record.json"
PIN = H / "verification" / "expected_evidence.json"
RECORD_PIN_KEY = ("bf16_baseline", "bf16_kat_record.json")


def load_profile(path: Path = PROFILE) -> dict[str, Any]:
  return yaml.safe_load(path.read_text())


def load_vectors(path: Path = VECTORS) -> dict[str, Any]:
  return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# source anchors
# ---------------------------------------------------------------------------
def _read_line(file: str, line: int) -> str | None:
  base = REPO / "hardware" / "coral_nexus" / "upstream" / "coralnpu"
  root = base if file.startswith("rules/") else UPSTREAM_HDL
  path = root / file
  if not path.is_file():
    return None
  lines = path.read_text().splitlines()
  return lines[line - 1] if 0 < line <= len(lines) else None


def iter_sources(node: Any, where: str = "") -> list[tuple[str, dict]]:
  found: list[tuple[str, dict]] = []
  if isinstance(node, dict):
    file = node.get("file")
    if (
      isinstance(file, str)
      and isinstance(node.get("line"), int)
      and file.startswith(("verilog/", "chisel/", "rules/"))
    ):
      found.append((where, node))
    for key, value in node.items():
      found.extend(iter_sources(value, f"{where}.{key}"))
  elif isinstance(node, list):
    for index, value in enumerate(node):
      found.extend(iter_sources(value, f"{where}[{index}]"))
  return found


def check_anchors(profile: dict[str, Any], errors: list[str]) -> int:
  count = 0
  for where, src in iter_sources(profile):
    count += 1
    text = _read_line(src["file"], src["line"])
    if text is None:
      errors.append(f"{where}: cannot read {src['file']}:{src['line']}")
    elif not src.get("anchor"):
      errors.append(f"{where}: {src['file']}:{src['line']} has no anchor to check")
    elif src["anchor"] not in text:
      errors.append(f"{where}: {src['file']}:{src['line']} does not contain {src['anchor']!r}")
  return count


# ---------------------------------------------------------------------------
# generator parameters
# ---------------------------------------------------------------------------
def check_generator(profile: dict[str, Any], errors: list[str]) -> None:
  gen = profile["generator"]
  enabled = tuple(gen["fp_fmt_config"]["enabled"])
  if enabled != nr.BLOCK_FP_FMT_CONFIG:
    errors.append(f"generator.fp_fmt_config.enabled {enabled} != block config {nr.BLOCK_FP_FMT_CONFIG}")
  literal = gen["fp_fmt_config"]["literal"]
  bits = literal.split("b")[-1]
  derived_enabled = tuple(nr.FORMAT_ORDER[i] for i, b in enumerate(bits) if b == "1")
  if derived_enabled != enabled:
    errors.append(f"literal {literal} enables {derived_enabled}, profile says {enabled}")
  try:
    derived = nr.derive_parameters(enabled)
  except nr.NumericError as exc:
    errors.append(f"generator: {exc}")
    return
  stated = gen["derived"]
  for key in (
    "super_exp_bits",
    "super_man_bits",
    "width",
    "vec_len",
    "guard_bits",
    "prod_exp_bits",
    "prod_sig_bits",
  ):
    if stated.get(key) != getattr(derived, key):
      errors.append(
        f"generator.derived.{key}: profile {stated.get(key)} != derived {getattr(derived, key)}"
      )
  if stated.get("lanes") != derived.lanes:
    errors.append(f"generator.derived.lanes: profile {stated.get('lanes')} != derived {derived.lanes}")
  if nr.SIG_BITS_ALIGNED != derived.prod_sig_bits:
    errors.append(f"numeric_ref.SIG_BITS_ALIGNED {nr.SIG_BITS_ALIGNED} != derived {derived.prod_sig_bits}")
  table = {k: tuple(v) for k, v in gen["format_table"]["entries"].items()}
  if table != nr.FP_FORMATS:
    errors.append("generator.format_table differs from numeric_ref.FP_FORMATS")


CVFPU_VENDOR = (
  REPO / "out" / "coral_nexus" / "C005" / "work-coralnpu" / "vendor-cache" / "+coralnpu_deps_ext+cvfpu"
)


def check_cvfpu(profile: dict[str, Any], errors: list[str], notes: list[str]) -> bool:
  """Read the cvfpu encodings back from the vendor copy. Returns whether it could be checked."""
  pin = profile["cvfpu_pin"]
  path = CVFPU_VENDOR / "src" / "fpnew_pkg.sv"
  if not path.is_file():
    notes.append("cvfpu vendor copy absent: encodings were NOT re-read (reported, not passed)")
    return False
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  if digest != pin["read_sha256"]:
    errors.append(f"cvfpu fpnew_pkg.sv sha256 {digest} != profile {pin['read_sha256']}")
    return True
  lines = path.read_text().splitlines()
  for name, entry in pin["encodings"].items():
    text = lines[entry["line"] - 1] if 0 < entry["line"] <= len(lines) else ""
    if name not in text or entry["value"] not in text:
      errors.append(f"cvfpu_pin.encodings.{name}: line {entry['line']} is {text.strip()!r}")
  want = {"RNE": nr.RNE, "RTZ": nr.RTZ, "RDN": nr.RDN, "RUP": nr.RUP, "RMM": nr.RMM, "ROD": nr.ROD}
  for name, value in want.items():
    literal = pin["encodings"][name]["value"]
    if int(literal.split("b")[-1], 2) != value:
      errors.append(f"numeric_ref {name}={value:#05b} but cvfpu says {literal}")
  return True


# ---------------------------------------------------------------------------
# the required contract (held HERE, not in the YAML it constrains)
# ---------------------------------------------------------------------------
# A required set stored in the profile could be deleted by the same edit that deletes a rule, so
# it lives in the validator. Each entry is a rule the C011 card names, with the evidence class the
# rule must keep and the vector KIND that must exercise it. Removing a rule, weakening its evidence
# class, or removing the last vector of a kind makes the whole contract invalid.
#
# Card clause (design/coral_nexus_tile/execution/G1_contracts.md, C011 step 1):
#   finite-width tree -> MUL-01; jam -> JAM-01; internal ROD -> ROD-01; multiplier widening overflow
#   to Inf -> OVF-MUL-01; accumulate ROD overflow to max-normal -> OVF-ACC-01; signed zero, NaN,
#   subnormal -> CAN-01 / SPC-01 / SUB-01; mask -> MSK-01; TK grouping -> TKG-01; the final BF16 RNE
#   narrowing as its own boundary -> NAR-01; VME OF/NV only -> FLG-01; tail -> TIL-01; frm -> FRM-01.
REQUIRED_RULES: dict[str, dict[str, Any]] = {
  "MUL-01": {"evidence": "upstream_source", "kinds": {"cell"}},
  "JAM-01": {"evidence": "upstream_source", "kinds": {"cell", "ablation"}},
  "ROD-01": {"evidence": "upstream_source", "kinds": {"cell", "tk_sequence", "ablation"}},
  "OVF-MUL-01": {"evidence": "upstream_source", "kinds": {"cell"}},
  "OVF-ACC-01": {"evidence": "upstream_source", "kinds": {"cell"}},
  "SPC-01": {"evidence": "upstream_source", "kinds": {"cell"}},
  "MSK-01": {"evidence": "upstream_source", "kinds": {"cell"}},
  "TIL-01": {"evidence": "real_rtl_run", "kinds": {"record_case"}},
  "TKG-01": {"evidence": "upstream_source", "kinds": {"cell", "tk_sequence"}},
  "CAN-01": {"evidence": "upstream_source", "kinds": {"cell"}},
  "SUB-01": {"evidence": "upstream_source", "kinds": {"cell"}},
  "FLG-01": {"evidence": "upstream_source", "kinds": {"cell"}},
  "FRM-01": {"evidence": "upstream_source", "kinds": {"frm"}},
  "NAR-01": {"evidence": "decided_by_card", "kinds": {"narrow"}},
}
# Open items and known differences the contract must keep reporting until their owner closes them.
REQUIRED_OPEN_ITEMS = {"OPEN-02", "OPEN-03", "OPEN-04"}
REQUIRED_KNOWN_DIFFERENCES = {"KD-01"}
# Vector kinds that must be present at all, whatever the rules claim.
REQUIRED_VECTOR_KINDS = {"cell", "ablation", "tk_sequence", "narrow", "frm", "record_case"}


def check_required_contract(profile: dict[str, Any], vectors: dict[str, Any], errors: list[str]) -> None:
  rules = {r["id"]: r for r in profile["rules"]}
  by_id = {v["id"]: v for v in vectors["vectors"]}
  for rid, need in REQUIRED_RULES.items():
    rule = rules.get(rid)
    if rule is None:
      errors.append(f"required rule {rid} is missing from the profile")
      continue
    if rule.get("evidence") != need["evidence"]:
      errors.append(
        f"required rule {rid}: evidence class {rule.get('evidence')!r}, must stay {need['evidence']!r}"
      )
    kinds = {by_id[v]["kind"] for v in rule.get("vectors", []) if v in by_id}
    for kind in sorted(need["kinds"] - kinds):
      errors.append(f"required rule {rid}: no {kind!r} vector exercises it")
  for kind in sorted(REQUIRED_VECTOR_KINDS - {v["kind"] for v in vectors["vectors"]}):
    errors.append(f"required vector kind {kind!r} is absent from known_vectors.json")
  present = {i["id"] for i in profile.get("open_items", []) if i.get("status") != "closed"}
  for item in sorted(REQUIRED_OPEN_ITEMS - present):
    errors.append(f"open item {item} was removed or closed without its owner closing it")
  closed = {i["id"]: i for i in profile.get("open_items", []) if i.get("status") == "closed"}
  if "OPEN-01" not in closed:
    errors.append("open item OPEN-01 must stay closed by the committed probe record")
  known = {k["id"] for k in profile.get("known_differences", [])}
  for kd in sorted(REQUIRED_KNOWN_DIFFERENCES - known):
    errors.append(f"known difference {kd} was removed")
  extra = sorted(set(rules) - set(REQUIRED_RULES))
  if extra:
    errors.append(f"rules {extra} are not in the validator's required set; add them there deliberately")


# ---------------------------------------------------------------------------
# rules vs vectors
# ---------------------------------------------------------------------------
def check_rule_vector_coverage(profile: dict[str, Any], vectors: dict[str, Any], errors: list[str]) -> None:
  ids = [v["id"] for v in vectors["vectors"]]
  if len(ids) != len(set(ids)):
    errors.append("known_vectors.json has duplicate vector ids")
  known = set(ids)
  claimed: set[str] = set()
  for rule in profile["rules"]:
    listed = rule.get("vectors") or []
    if not listed:
      errors.append(f"rule {rule['id']}: no directed vector")
    for vid in listed:
      claimed.add(vid)
      if vid not in known:
        errors.append(f"rule {rule['id']}: vector {vid!r} is not in known_vectors.json")
  rule_ids = {r["id"] for r in profile["rules"]}
  by_id = {v["id"]: v for v in vectors["vectors"]}
  for vid in known - claimed:
    # a vector may be claimed through its own `rules` list; the rule's list must agree
    errors.append(f"vector {vid!r} is claimed by no rule")
  for vid, vec in by_id.items():
    for rid in vec.get("rules", []):
      if rid not in rule_ids:
        errors.append(f"vector {vid!r} names unknown rule {rid!r}")
      elif vid not in next(r for r in profile["rules"] if r["id"] == rid).get("vectors", []):
        errors.append(f"vector {vid!r} claims rule {rid} but the rule does not list it")


# ---------------------------------------------------------------------------
# running vectors
# ---------------------------------------------------------------------------
Impl = dict[str, Any]


def run_cell(vec: dict[str, Any]) -> tuple[int, dict[str, int]]:
  pairs = [tuple(p) for p in vec["pairs_bf16_bits"]]
  bits, mul, add = nr.vtfmm_cell(vec["old_fp32_bits"], pairs, vec["tk"])
  return bits, {"nv": mul.nv | add.nv, "of": mul.of | add.of}


_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
  "cell": ("old_fp32_bits", "pairs_bf16_bits", "tk", "expected_fp32_bits", "expected_flags"),
  "ablation": (
    "old_fp32_bits",
    "pairs_bf16_bits",
    "tk",
    "expected_fp32_bits",
    "ablation",
    "ablated_fp32_bits",
  ),
  "tk_sequence": (
    "old_fp32_bits",
    "pairs_bf16_bits",
    "tk",
    "expected_fp32_bits",
    "expected_split_fp32_bits",
  ),
  "narrow": ("input_fp32_bits", "frm", "expected_bf16_bits", "expected_flags"),
  "frm": ("frm", "expected_error"),
  "record_case": ("record", "case"),
}


def run_vector(vec: dict[str, Any]) -> list[str]:
  """Run one vector through the reference model; return the list of mismatches (empty = pass).

  A vector whose fields do not fit its kind is reported, never raised: a malformed vector must not
  turn a validation failure into a traceback.
  """
  kind = str(vec.get("kind"))
  out: list[str] = []
  need = _REQUIRED_FIELDS.get(kind)
  if need is None:
    return [f"unknown vector kind {kind!r}"]
  missing = [f for f in need if f not in vec]
  if missing:
    return [f"kind {kind!r} needs fields {missing}"]
  try:
    return _run_vector_checked(vec, kind, out)
  except (nr.NumericError, TypeError, ValueError, IndexError) as exc:
    return [f"malformed vector: {exc}"]


def _run_vector_checked(vec: dict[str, Any], kind: str, out: list[str]) -> list[str]:
  if kind in ("cell", "ablation", "tk_sequence"):
    bits, flags = run_cell(vec)
    if bits != vec["expected_fp32_bits"]:
      out.append(f"bits {bits:#010x} != expected {vec['expected_fp32_bits']:#010x}")
    if "expected_flags" in vec and flags != vec["expected_flags"]:
      out.append(f"flags {flags} != expected {vec['expected_flags']}")
    if kind == "tk_sequence":
      pairs = [tuple(p) for p in vec["pairs_bf16_bits"]]
      mid = nr.vtfmm_cell(vec["old_fp32_bits"], [pairs[0], (0, 0)], 1)[0]
      split = nr.vtfmm_cell(mid, [pairs[1], (0, 0)], 1)[0]
      if split != vec["expected_split_fp32_bits"]:
        out.append(f"split {split:#010x} != expected {vec['expected_split_fp32_bits']:#010x}")
      if split == bits:
        out.append("tk grouping vector no longer distinguishes grouped from split")
  elif kind == "narrow":
    bits, st = nr.narrow_fp32_to_bf16(vec["input_fp32_bits"], vec["frm"])
    if bits != vec["expected_bf16_bits"]:
      out.append(f"bits {bits:#06x} != expected {vec['expected_bf16_bits']:#06x}")
    got = {"nv": st.nv, "of": st.of, "nx": st.nx}
    if got != vec["expected_flags"]:
      out.append(f"flags {got} != expected {vec['expected_flags']}")
  elif kind == "frm":
    try:
      nr.check_frm(vec["frm"])
      out.append("frm was admitted")
    except nr.NumericError as exc:
      if exc.code != vec["expected_error"]:
        out.append(f"error {exc.code} != expected {vec['expected_error']}")
  elif kind == "record_case":
    out.extend(check_record_case(vec))
  else:
    out.append(f"unknown vector kind {kind!r}")
  return out


def check_record_case(vec: dict[str, Any], record_path: Path | None = None) -> list[str]:
  """A `record_case` vector must point at a real tail case of the committed real-RTL record.

  The case must exist, be a tail tile (tm or tn below the full dimension), and every cell outside
  the tm x tn window must still hold the preloaded accumulator (undisturbed). Without this a vector
  could name any case and the tail rule would have no evidence behind it.
  """
  path = record_path or RECORD
  if not path.is_file():
    return [f"record {path} missing"]
  record = json.loads(path.read_text())
  if vec["record"] != "bf16_kat_record.json":
    return [f"record_case names unknown record {vec['record']!r}"]
  cases = {c["case"]: c for c in record["cases"]}
  case = cases.get(vec["case"])
  if case is None:
    return [f"record has no case {vec['case']}"]
  tm, tn = case["tm"], case["tn"]
  if tm >= nr_dim() and tn >= nr_dim():
    return [f"case {vec['case']} is a full tile (tm={tm}, tn={tn}); it carries no tail evidence"]
  if not case["init_mode"]:
    return [f"case {vec['case']} did not preload the accumulator, so undisturbed cells cannot be shown"]
  outside = [i for i in range(nr_dim() ** 2) if i // nr_dim() >= tm or i % nr_dim() >= tn]
  moved = [i for i in outside if case["actual_u32"][i] != case["c_init_u32"][i]]
  if moved:
    return [f"case {vec['case']}: {len(moved)} cells outside {tm}x{tn} changed (first {moved[0]})"]
  inside = [i for i in range(nr_dim() ** 2) if i not in set(outside)]
  if not inside or all(case["actual_u32"][i] == case["c_init_u32"][i] for i in inside):
    return [f"case {vec['case']}: no cell inside the window was updated"]
  return []


def nr_dim() -> int:
  return 16


def check_vectors(vectors: dict[str, Any], errors: list[str]) -> int:
  count = 0
  for vec in vectors["vectors"]:
    count += 1
    for problem in run_vector(vec):
      errors.append(f"vector {vec['id']}: {problem}")
  return count


# ---- ablations: the contract result must DIFFER from the wrong implementation's result -------
def ablated_cell(vec: dict[str, Any]) -> int:
  """Re-run a cell vector with the named wrong implementation."""
  pairs = [tuple(p) for p in vec["pairs_bf16_bits"]]
  lanes = [nr.mul_lane(a, b, k < vec["tk"]) for k, (a, b) in enumerate(pairs[:2])]
  how = vec["ablation"]
  if how == "truncate":
    prod = _mulbulk_variant(lanes, jam=False, mode=nr.ROD)
  elif how == "rne_internal":
    prod = _mulbulk_variant(lanes, jam=True, mode=nr.RNE)
  else:
    raise nr.NumericError("ABLATION_UNKNOWN", how)
  return nr.accumulate(prod, vec["old_fp32_bits"])[0]


def _mulbulk_variant(lanes: list[Any], *, jam: bool, mode: int) -> int:
  from fractions import Fraction

  live = [lane for lane in lanes if not lane.zero]
  if not live:
    return 0
  em = max(lane.norm_exp for lane in live)
  unit = Fraction(2) ** (em - (nr.SIG_BITS_ALIGNED - 1))
  total = 0
  for lane in live:
    mag = abs(lane.product)
    q = mag / unit
    fl = q.numerator // q.denominator
    if jam and q != fl:
      fl |= 1
    total += -fl if lane.sign else fl
  if total == 0:
    return 0
  return nr.round_exact(Fraction(total) * unit, 23, 8, mode, widen_overflow_inf=True)[0]


def check_ablations(vectors: dict[str, Any], errors: list[str]) -> int:
  count = 0
  for vec in vectors["vectors"]:
    if vec["kind"] != "ablation":
      continue
    count += 1
    contract = vec["expected_fp32_bits"]
    wrong = ablated_cell(vec)
    if wrong == contract:
      errors.append(
        f"ablation {vec['id']}: the wrong implementation ({vec['ablation']}) gives the contract result"
      )
    if wrong != vec["ablated_fp32_bits"]:
      errors.append(
        f"ablation {vec['id']}: wrong implementation gives {wrong:#010x}, "
        f"vector records {vec['ablated_fp32_bits']:#010x}"
      )
  return count


# ---------------------------------------------------------------------------
# the real-RTL record
# ---------------------------------------------------------------------------
def record_pin() -> str:
  entry = json.loads(PIN.read_text())["entries"][RECORD_PIN_KEY[0]]["artifacts"][RECORD_PIN_KEY[1]]
  return str(entry)


def check_record(errors: list[str], record_path: Path = RECORD) -> int:
  if not record_path.is_file():
    errors.append(f"real-RTL record {record_path} missing")
    return 0
  raw = record_path.read_bytes()
  digest = hashlib.sha256(raw).hexdigest()
  if record_path == RECORD and digest != record_pin():
    errors.append(f"record sha256 {digest} != committed pin {record_pin()}")
  rec = json.loads(raw)
  if rec.get("schema") != "cn_p0_bf16_kat_record/v1":
    errors.append(f"record schema {rec.get('schema')!r}")
    return 0
  cells = 0
  for case in rec["cases"]:
    a = [case["a_bf16_bits"][:16], case["a_bf16_bits"][16:]]
    b = [case["b_bf16_bits"][:16], case["b_bf16_bits"][16:]]
    preload = case["c_init_u32"] if case["init_mode"] else [0] * 256
    model = nr.vtfmm_tile(a, b, preload, case["tm"], case["tn"], case["tk"])
    for index, (want, got) in enumerate(zip(model, case["actual_u32"], strict=True)):
      cells += 1
      if want != got:
        errors.append(
          f"record case {case['case']} cell {index}: model {want:#010x} != real RTL {got:#010x}"
        )
        break
  return cells


def check_probe_record(vectors: dict[str, Any], errors: list[str], path: Path = PROBE_RECORD) -> int:
  """The committed real-RTL probe record must agree with the model AND the hand-derived vectors.

  The ELF it ran must be the very ELF the C006 baseline pinned, so the probe and the baseline
  exercise one binary.
  """
  if not path.is_file():
    errors.append(f"real-RTL probe record {path} missing")
    return 0
  rec = json.loads(path.read_text())
  try:
    groups = probe.build_groups(vectors)
  except (KeyError, TypeError, IndexError, probe.ProbeError) as exc:
    errors.append(f"probe record: vectors cannot be packed for the probe: {type(exc).__name__}: {exc}")
    return 0
  errors.extend(f"probe record: {e}" for e in probe.compare(rec, groups))
  errors.extend(f"probe record: {e}" for e in probe.vector_results(rec, groups, vectors))
  if rec.get("replay_identical") is not True:
    errors.append("probe record: the two seeded runs were not identical")
  baseline = json.loads(RECORD.read_text())
  if rec.get("elf_sha256") != baseline.get("elf_sha256"):
    errors.append("probe record: ran a different ELF than the C006 baseline pinned")
  return sum(len(g["placed"]) for g in groups)


# ---------------------------------------------------------------------------
# whole-contract entry
# ---------------------------------------------------------------------------
def validate(
  profile_path: Path = PROFILE, vectors_path: Path = VECTORS, record_path: Path = RECORD
) -> dict[str, Any]:
  profile = load_profile(profile_path)
  vectors = load_vectors(vectors_path)
  errors: list[str] = []
  notes: list[str] = []
  anchors = check_anchors(profile, errors)
  check_generator(profile, errors)
  cvfpu_checked = check_cvfpu(profile, errors, notes)
  check_required_contract(profile, vectors, errors)
  check_rule_vector_coverage(profile, vectors, errors)
  vector_count = check_vectors(vectors, errors)
  ablation_count = check_ablations(vectors, errors)
  cells = check_record(errors, record_path)
  probed = check_probe_record(vectors, errors)
  if profile.get("support_status") != "experimental":
    errors.append("profile must stay experimental; C047 owns any supported claim")
  open_items = sorted(i["id"] for i in profile.get("open_items", []) if i.get("status") != "closed")
  return {
    "schema": "cn_numeric_report/v1",
    "valid": not errors,
    "errors": errors,
    "notes": notes,
    "profile_id": profile["numeric_profile_id"],
    "anchors_read_back": anchors,
    "cvfpu_encodings_checked": cvfpu_checked,
    "rules": len(profile["rules"]),
    "vectors": vector_count,
    "ablations": ablation_count,
    "real_rtl_cells_compared": cells,
    "real_rtl_probe_vectors_confirmed": probed,
    "open_items": open_items,
  }


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  report = validate()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / "numeric_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  if not report["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
