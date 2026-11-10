"""Reference consumer for the W7 trace / replay / verdict contracts (C016).

This is a CONTRACT CONSUMER, not a co-simulation harness and not a checker suite: it implements exactly the
parts C016 must make testable with real structured input, so that a damaged window or an unearned verdict is
refused with a code that names the problem.

  seal_window / check_window ... C1 format and uniqueness, window sealing
  replay_gate ................... C6: is this window completely replayable
  classify_outcome .............. CASE_PASS / DUT_ERROR / NUMERIC_FAIL / ... -> t_verdict
  evaluate_manifest ............. C7: may a set of per-row verdicts be accepted as a gate

C2/C3/C4/C5 (partial order, conservation, numerics, termination) are specified in the contracts and
implemented by later cards; nothing here claims they run.

Every rule below is read from the YAML contracts, so a test that deletes a rule from the YAML changes what
this module accepts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
H = LIB.parents[1]
CONTRACTS = H / "verification" / "contracts"
WINDOW = CONTRACTS / "trace_window_contract_v1.yaml"
RECORD = CONTRACTS / "log_record_contract_v1.yaml"
VERDICT = CONTRACTS / "checker_verdict_contract_v1.yaml"
BUNDLE = CONTRACTS / "replay_bundle_contract_v1.yaml"
PERF = CONTRACTS / "perf_gates_contract_v1.yaml"
MATRIX = H / "verification" / "matrix" / "t_matrix.yaml"
CATALOG = H / "protocol" / "cn_protocol_v1.yaml"


class TraceError(Exception):
  """Coded rejection; `code` is stable so tests can pin it."""

  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(code + (f": {detail}" if detail else ""))
    self.code = code


def load(path: Path) -> dict[str, Any]:
  return yaml.safe_load(path.read_text())  # type: ignore[no-any-return]


class Contracts:
  """The five W7 contracts plus the C009 catalog, loaded once and passed around."""

  def __init__(self) -> None:
    self.window = load(WINDOW)
    self.record = load(RECORD)
    self.verdict = load(VERDICT)
    self.bundle = load(BUNDLE)
    self.perf = load(PERF)
    catalog = load(CATALOG)
    self.phases = list(catalog["phases"]["order"])
    self.channels = [c["name"] for c in catalog["channels"]]
    self.fields = {f["name"]: f for f in self.record["fields"]}


def canonical(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def records_digest(records: list[dict[str, Any]]) -> str:
  h = hashlib.sha256()
  for rec in records:
    h.update(canonical(_jsonable(rec)))
    h.update(b"\n")
  return h.hexdigest()


def _jsonable(rec: dict[str, Any]) -> dict[str, Any]:
  """Bytes payloads are hashed as hex so the canonical form is JSON."""
  out = dict(rec)
  if isinstance(out.get("payload_bytes"), (bytes, bytearray)):
    out["payload_bytes"] = bytes(out["payload_bytes"]).hex()
  return out


def _is_int(v: Any) -> bool:
  return isinstance(v, int) and not isinstance(v, bool)


# ---------------------------------------------------------------------------------------------------
# sealing
# ---------------------------------------------------------------------------------------------------
def seal_window(window: dict[str, Any]) -> dict[str, Any]:
  """Compute the sealing record from the window's own records and header fields."""
  recs = window["records"]
  return {
    "first_seq": recs[0]["seq"] if recs else 0,
    "last_seq": recs[-1]["seq"] if recs else -1,
    "record_count": len(recs),
    "end_boundary": window["end_boundary"],
    "gap_status": window["gap_status"],
    "records_sha256": records_digest(recs),
  }


# ---------------------------------------------------------------------------------------------------
# C1: format and uniqueness
# ---------------------------------------------------------------------------------------------------
def check_window(window: dict[str, Any], c: Contracts) -> list[dict[str, Any]]:
  """C1. Returns findings `{code, seq, detail}`; empty means the records are well formed."""
  out: list[dict[str, Any]] = []

  def add(code: str, seq: int | None, detail: str = "") -> None:
    out.append({"code": code, "seq": seq, "detail": detail})

  header = c.window["header"]
  for name in header["required_fields"]:
    if name not in window and name != "seal":
      add("MISSING_HEADER_FIELD", None, name)
  allowed = set(header["required_fields"]) | {"records"}
  for name in window:
    if name not in allowed:
      add("UNKNOWN_FIELD", None, f"header.{name}")
  if window.get("window_kind") not in header["window_kinds"]:
    add("BAD_WINDOW_KIND", None, str(window.get("window_kind")))
  records = window.get("records")
  if not isinstance(records, list):
    add("BAD_TYPE", None, "records")
    return out

  rc = c.record
  allowed_rec = set(c.fields)
  forbidden = set(rc["raw_log_forbidden_fields"])
  r1_channels = set(rc["reliability"]["R1"]["channels"])
  seen_keys: set[tuple] = set()
  pending_stall: dict[tuple, str] = {}
  last: tuple[int, int] | None = None
  r1_expected = 0
  msg_expected: dict[tuple, int] = {}
  prev_seq: int | None = None
  for rec in records:
    seq = rec.get("seq")
    for name in rec:
      if name in forbidden:
        add("QUANTIZED_FIELD_IN_RAW_LOG", seq, name)
      elif name not in allowed_rec:
        add("UNKNOWN_FIELD", seq, name)
    for name, spec in c.fields.items():
      if spec["required"] and name not in rec:
        add("MISSING_FIELD", seq, name)
    for name in ("seq", "cycle", "len"):
      if name in rec and not _is_int(rec[name]):
        add("BAD_TYPE", seq, f"{name} must be an integer, got {type(rec[name]).__name__}")
    for name in ("valid", "ready", "accepted"):
      if name in rec and not isinstance(rec[name], bool):
        add("BAD_TYPE", seq, f"{name} must be a bool")
    if _is_int(seq):
      if prev_seq is not None and seq != prev_seq + 1:
        add("SEQ_GAP", seq, f"expected {prev_seq + 1}")
      prev_seq = seq
    phase = rec.get("phase")
    if phase not in c.phases:
      add("BAD_PHASE", seq, str(phase))
    protocol = rec.get("protocol")
    if protocol not in c.record["protocol"]["names"]:
      add("BAD_PROTOCOL", seq, str(protocol))
    if rec.get("dir") not in c.record["protocol"]["dir"]:
      add("BAD_DIR", seq, str(rec.get("dir")))
    if rec.get("reliability") not in c.record["reliability"]["classes"]:
      add("BAD_RELIABILITY", seq, str(rec.get("reliability")))
    if _is_int(rec.get("cycle")) and phase in c.phases:
      here = (rec["cycle"], c.phases.index(phase))
      if last is not None and here < last:
        add("CYCLE_REGRESSION", seq, f"{here} after {last}")
      last = here
    ok_shape = all(isinstance(rec.get(n), bool) for n in ("valid", "ready", "accepted"))
    if ok_shape and rec["accepted"] != (rec["valid"] and rec["ready"]):
      add("HANDSHAKE_MISMATCH", seq, "accepted must equal valid and ready")
    key = (window.get("window_id"), protocol, rec.get("port"), rec.get("dir"), rec.get("cycle"), phase)
    if key in seen_keys:
      add("DUPLICATE_KEY", seq, str(key[1:]))
    seen_keys.add(key)
    pkey = (protocol, rec.get("port"), rec.get("dir"))
    if pkey in pending_stall and rec.get("payload_hash") != pending_stall[pkey]:
      add("UNSTABLE_PAYLOAD", seq, "payload changed while valid and not ready")
    if ok_shape and rec["valid"] and not rec["ready"]:
      pending_stall[pkey] = rec.get("payload_hash", "")
    else:
      pending_stall.pop(pkey, None)
    if rec.get("reliability") == "R1":
      if protocol not in r1_channels:
        add("R1_ON_DEBUG", seq, f"R1 record on {protocol}")
      if rec.get("r1_seq") != r1_expected:
        add("R1_MISSING", seq, f"r1_seq {rec.get('r1_seq')} expected {r1_expected}")
      r1_expected = (rec.get("r1_seq") if _is_int(rec.get("r1_seq")) else r1_expected) + 1
    if protocol == "DEBUG" and rec.get("reliability") == "R1":
      pass
    if rec.get("message_seq") is not None:
      ident = rec.get("identity") or {}
      mkey = (ident.get("tile_id"), protocol, rec.get("dir"))
      want = msg_expected.get(mkey)
      if want is not None and rec["message_seq"] != want:
        add("MESSAGE_SEQ_GAP", seq, f"message_seq {rec['message_seq']} expected {want}")
      msg_expected[mkey] = rec["message_seq"] + 1
    pb = rec.get("payload_bytes")
    if pb is not None:
      if rec.get("len") != len(pb):
        add("LEN_MISMATCH", seq, f"len {rec.get('len')} != {len(pb)} bytes")
      if sha256(bytes(pb)) != rec.get("payload_hash"):
        add("HASH_MISMATCH", seq, "payload_hash does not match payload_bytes")
  return out


# ---------------------------------------------------------------------------------------------------
# C6: replay gate
# ---------------------------------------------------------------------------------------------------
def replay_gate(
  window: dict[str, Any], blob_store: dict[str, bytes], declared_end: dict[str, Any], c: Contracts
) -> dict[str, Any]:
  """Is the window COMPLETELY replayable? `declared_end` is held by the scenario, outside the window.

  Replayable is not task-settled: that is C5's question.
  """
  reasons: list[str] = []
  comp = c.window
  findings = check_window(window, c)
  if any(f["code"] in ("SEQ_GAP",) for f in findings):
    reasons.append("seq_gap")
  if any(f["code"] == "HASH_MISMATCH" for f in findings):
    reasons.append("hash_mismatch")
  if any(f["code"] in ("MISSING_HEADER_FIELD", "BAD_TYPE", "UNKNOWN_FIELD") for f in findings):
    reasons.append("malformed_window")
  recs = window.get("records") or []
  seal = window.get("seal")
  if not isinstance(seal, dict):
    reasons.append("missing_seal")
  else:
    need = comp["seal"]["required_fields"]
    if any(n not in seal for n in need):
      reasons.append("incomplete_seal")
    else:
      want = seal_window(window)
      for name in ("first_seq", "last_seq", "record_count", "records_sha256", "end_boundary", "gap_status"):
        if seal.get(name) != want[name]:
          reasons.append(f"seal_mismatch:{name}")
  if window.get("gap_status") != "none":
    reasons.append(f"gap:{window.get('gap_status')}")
  elif isinstance(seal, dict) and seal.get("gap_status") not in (None, "none"):
    reasons.append(f"gap:{seal.get('gap_status')}")
  eb = window.get("end_boundary")
  if not isinstance(eb, dict) or eb.get("type") not in comp["end_boundary"]["types"]:
    reasons.append("bad_end_boundary")
  elif eb != declared_end:
    reasons.append("end_boundary_differs_from_declaration")
  elif not recs or recs[-1].get("cycle") != declared_end.get("cycle"):
    reasons.append("truncated_end")
  elif (
    eb["type"] == "run_terminal"
    and recs[-1].get("kind") != comp["end_boundary"]["run_terminal"]["last_record"]
  ):
    reasons.append("truncated_end")
  manifest = window.get("blob_manifest") or []
  listed: dict[str, dict[str, Any]] = {}
  for entry in manifest:
    if any(n not in entry for n in comp["blob_manifest"]["entry_required_fields"]):
      reasons.append("bad_blob_entry")
      continue
    if entry["blob_kind"] not in comp["blob_manifest"]["blob_kinds"]:
      reasons.append("bad_blob_kind")
    listed[entry["sha256"]] = entry
    data = blob_store.get(entry["sha256"])
    if data is None:
      reasons.append("missing_blob")
    elif sha256(data) != entry["sha256"] or len(data) != entry["len"]:
      reasons.append("blob_hash_mismatch")
  for rec in recs:
    if rec.get("payload_bytes") is None and rec.get("blob_ref") is None and rec.get("len", 0) > 0:
      reasons.append("hash_only_record")
    if rec.get("blob_ref") is not None and rec["blob_ref"] not in listed:
      reasons.append("unlisted_blob_ref")
  kinds = {e.get("blob_kind") for e in manifest if isinstance(e, dict)}
  if "initial_memory" not in kinds:
    reasons.append("missing_initial_snapshot")
  if "image_ref" not in kinds:
    reasons.append("missing_image")
  cfg = window.get("config_refs") or {}
  need_cfg = c.bundle["config"]["required"]
  for name in need_cfg:
    if not cfg.get(name):
      reasons.append(f"missing_config:{name}")
  se = window.get("startup_epoch")
  if not isinstance(se, dict) or any(n not in se for n in comp["startup_epoch"]["required_fields"]):
    reasons.append("missing_startup_epoch")
  reasons = sorted(set(reasons))
  return {
    "verdict": "REPLAYABLE" if not reasons else "NOT_FULLY_REPLAYABLE",
    "reasons": reasons,
    "exit_code": 0 if not reasons else 3,
    "summary": {
      "records": len(recs),
      "first_seq": recs[0]["seq"] if recs else None,
      "findings": len(findings),
    },
  }


# ---------------------------------------------------------------------------------------------------
# outcome classes
# ---------------------------------------------------------------------------------------------------
def classify_outcome(
  c: Contracts,
  *,
  expected_dut_error: bool,
  dut_ended_in_error: bool,
  checker_failed: bool = False,
  numeric_mismatch: bool = False,
  pending_at_end: bool = False,
  replayable: bool = True,
) -> dict[str, str]:
  """One outcome class per run; a legal injected fault that ends as expected is a CASE_PASS."""
  oc = c.verdict["outcome_classes"]
  if not replayable:
    cls = "NOT_FULLY_REPLAYABLE"
  elif numeric_mismatch:
    cls = "NUMERIC_FAIL"
  elif checker_failed:
    cls = "CHECKER_FAIL"
  elif pending_at_end:
    cls = "PENDING_NOT_COMPLETE"
  elif dut_ended_in_error and not expected_dut_error:
    cls = "DUT_ERROR"
  elif expected_dut_error and not dut_ended_in_error:
    cls = "DUT_ERROR"  # the fault the scenario injected did not happen
  else:
    cls = "CASE_PASS"
  return {"class": cls, "t_verdict": oc["maps_to_t_verdict"][cls]}


# ---------------------------------------------------------------------------------------------------
# C7: manifest gate
# ---------------------------------------------------------------------------------------------------
def _check_evidence(
  entries: Any, root: Path, t_id: str, problems: list[dict[str, str]], where: str, c: Contracts
) -> list[str]:
  kinds: list[str] = []
  vc = c.verdict["verdict"]
  if not isinstance(entries, list):
    problems.append({"t_id": t_id, "code": "INSUFFICIENT_EVIDENCE", "detail": f"{where}: no evidence list"})
    return kinds
  for e in entries:
    miss = [n for n in vc["evidence_entry_required_fields"] if n not in e]
    if miss:
      problems.append({"t_id": t_id, "code": "INSUFFICIENT_EVIDENCE", "detail": f"{where}: {miss}"})
      continue
    if e["kind"] not in c.verdict["evidence"]["kinds"]:
      problems.append({"t_id": t_id, "code": "KIND_NOT_ALLOWED", "detail": f"{where}: {e['kind']}"})
      continue
    if e["t_id"] != t_id:
      problems.append(
        {"t_id": t_id, "code": "DANGLING_REF", "detail": f"{where}: evidence belongs to {e['t_id']}"}
      )
      continue
    path = root / e["path"]
    if not path.is_file():
      problems.append(
        {"t_id": t_id, "code": "DANGLING_REF", "detail": f"{where}: {e['path']} does not exist"}
      )
      continue
    if sha256(path.read_bytes()) != e["sha256"]:
      problems.append({"t_id": t_id, "code": "HASH_MISMATCH", "detail": f"{where}: {e['path']}"})
      continue
    kinds.append(e["kind"])
  return kinds


def _checkers_pass(v: dict[str, Any], c: Contracts) -> bool:
  want = list(c.verdict["checkers"].keys())
  want = [k for k in want if k.startswith("C") and len(k) == 2]
  got = {x.get("name"): x.get("status") for x in v.get("checker", []) if isinstance(x, dict)}
  return all(got.get(k) == "PASS" for k in want)


def _row_kinds_ok(
  have: list[str], need: list[str], *, py_alone: bool, c: Contracts, problems: list, t_id: str, where: str
) -> bool:
  ok = True
  if "CS1_PY" in have and "CS1_LOG" not in have and not py_alone and "CS1_PY" not in need:
    problems.append(
      {"t_id": t_id, "code": "KIND_NOT_ALLOWED", "detail": f"{where}: CS1_PY cannot stand in for CS1_LOG"}
    )
    ok = False
  for k in need:
    if k not in have:
      problems.append({"t_id": t_id, "code": "INSUFFICIENT_EVIDENCE", "detail": f"{where}: needs {k}"})
      ok = False
  return ok


def evaluate_manifest(
  matrix: dict[str, Any], verdicts: dict[str, dict[str, Any]], root: Path, c: Contracts
) -> dict[str, Any]:
  """C7. `manifest_valid` and `gate_pass` are different: a valid manifest can still fail the gate."""
  problems: list[dict[str, str]] = []
  accepted: dict[str, str] = {}
  vc = c.verdict["verdict"]
  rows = {r["t_id"]: r for r in matrix["rows"]}
  for t_id, row in rows.items():
    v = verdicts.get(t_id)
    if v is None:
      accepted[t_id] = "MISSING_ROW"
      problems.append({"t_id": t_id, "code": "MISSING_ROW", "detail": "no verdict"})
      continue
    miss = [n for n in vc["required_fields"] if n not in v]
    if miss:
      problems.append({"t_id": t_id, "code": "INSUFFICIENT_EVIDENCE", "detail": f"verdict lacks {miss}"})
      accepted[t_id] = "REJECTED"
      continue
    status = v["t_verdict"]
    if status not in vc["t_verdict_values"]:
      problems.append({"t_id": t_id, "code": "BAD_VERDICT", "detail": str(status)})
      accepted[t_id] = "REJECTED"
      continue
    if v.get("t_scope") == "harness_only" and status == "PASS":
      problems.append(
        {"t_id": t_id, "code": "KIND_NOT_ALLOWED", "detail": "harness_only is not CS1 evidence"}
      )
      accepted[t_id] = "REJECTED"
      continue
    if status == "DEFERRED_NOT_APPLICABLE":
      if row["status"] != "DEFERRED_NOT_APPLICABLE" or not (
        v.get("defer_reason") and v.get("start_condition")
      ):
        problems.append(
          {
            "t_id": t_id,
            "code": "BAD_DEFERRAL",
            "detail": "row is not deferred in the matrix, or reason/start condition missing",
          }
        )
        accepted[t_id] = "REJECTED"
      elif v["defer_reason"] != row["defer_reason"] or v["start_condition"] != row["start_condition"]:
        problems.append(
          {
            "t_id": t_id,
            "code": "BAD_DEFERRAL",
            "detail": "reason or start condition differs from the matrix",
          }
        )
        accepted[t_id] = "REJECTED"
      else:
        accepted[t_id] = "DEFERRED_NOT_APPLICABLE"
      continue
    if status == "BLOCKED":
      b = v.get("blocked_by")
      if (
        not isinstance(b, dict)
        or b.get("kind") not in ("environment_assumption", "unfrozen_spec")
        or not b.get("ref")
      ):
        problems.append(
          {
            "t_id": t_id,
            "code": "BAD_BLOCKED",
            "detail": "BLOCKED must cite a missing assumption or an unfrozen spec",
          }
        )
        accepted[t_id] = "REJECTED"
      else:
        accepted[t_id] = "BLOCKED"
      continue
    if status == "FAIL":
      repro = v.get("repro") or {}
      if not all(repro.get(n) for n in ("window_id", "seed", "inject_plan_ref")):
        problems.append(
          {
            "t_id": t_id,
            "code": "INSUFFICIENT_EVIDENCE",
            "detail": "a FAIL needs window_id, seed and inject_plan_ref",
          }
        )
        accepted[t_id] = "REJECTED"
      else:
        accepted[t_id] = "FAIL"
      continue
    if status in ("NOT_RUN", "REJECTED"):
      accepted[t_id] = status
      continue
    # PASS
    if row["status"] == "DEFERRED_NOT_APPLICABLE":
      problems.append({"t_id": t_id, "code": "BAD_DEFERRAL", "detail": "a deferred row cannot be PASS"})
      accepted[t_id] = "REJECTED"
      continue
    before = len(problems)
    subs = row.get("subitems") or []
    mandatory_subs = [s for s in subs if s["scope"] == "mandatory"]
    if mandatory_subs:
      got = v.get("subitems") or {}
      for s in subs:
        sv = got.get(s["id"])
        where = f"subitem {s['id']}"
        if s["scope"] == "mandatory":
          if not isinstance(sv, dict) or sv.get("status") != "PASS":
            problems.append(
              {"t_id": t_id, "code": "INSUFFICIENT_EVIDENCE", "detail": f"{where} is not PASS"}
            )
            continue
          have = _check_evidence(sv.get("evidence"), root, t_id, problems, where, c)
          _row_kinds_ok(
            have,
            [k for k in s["evidence_kinds"] if k != "CS1_PY"],
            py_alone=bool(s.get("py_alone_allowed")),
            c=c,
            problems=problems,
            t_id=t_id,
            where=where,
          )
          if "CS1_PY" in s["evidence_kinds"] and "CS1_PY" not in have:
            problems.append(
              {"t_id": t_id, "code": "INSUFFICIENT_EVIDENCE", "detail": f"{where}: needs CS1_PY"}
            )
          if "CS1_LOG" in s["evidence_kinds"] and not _checkers_pass(v, c):
            problems.append(
              {"t_id": t_id, "code": "INSUFFICIENT_EVIDENCE", "detail": f"{where}: C1..C7 must all be PASS"}
            )
        elif s["scope"] == "deferred":
          if (
            not isinstance(sv, dict)
            or sv.get("status") != "DEFERRED_NOT_APPLICABLE"
            or sv.get("defer_reason") != s["defer_reason"]
            or sv.get("start_condition") != s["start_condition"]
          ):
            problems.append(
              {
                "t_id": t_id,
                "code": "BAD_DEFERRAL",
                "detail": f"{where} must be deferred with the matrix reason and start condition",
              }
            )
        else:
          if isinstance(sv, dict) and sv.get("status") == "PASS":
            problems.append(
              {
                "t_id": t_id,
                "code": "KIND_NOT_ALLOWED",
                "detail": f"{where} is outside the approved run scope and cannot be PASS",
              }
            )
    else:
      have = _check_evidence(v.get("evidence"), root, t_id, problems, "row", c)
      need = [k for k in row["evidence_kinds"] if k != "CS1_PY"]
      py_ok = row["evidence_kinds"] == ["CS1_PY"]
      _row_kinds_ok(have, need, py_alone=py_ok, c=c, problems=problems, t_id=t_id, where="row")
      if py_ok and "CS1_PY" not in have:
        problems.append({"t_id": t_id, "code": "INSUFFICIENT_EVIDENCE", "detail": "row: needs CS1_PY"})
      if "CS1_LOG" in row["evidence_kinds"] and not _checkers_pass(v, c):
        problems.append(
          {"t_id": t_id, "code": "INSUFFICIENT_EVIDENCE", "detail": "row: C1..C7 must all be PASS"}
        )
    accepted[t_id] = "PASS" if len(problems) == before else "REJECTED"

  fatal = {"HASH_MISMATCH", "DANGLING_REF", "BAD_DEFERRAL", "BAD_BLOCKED", "BAD_VERDICT", "MISSING_ROW"}
  manifest_valid = not any(p["code"] in fatal for p in problems)
  applicable = [t for t, r in rows.items() if r["status"] != "DEFERRED_NOT_APPLICABLE"]
  gate_pass = manifest_valid and all(accepted.get(t) == "PASS" for t in applicable)
  return {
    "manifest_valid": manifest_valid,
    "gate_pass": gate_pass,
    "accepted": accepted,
    "problems": problems,
    "rows_checked": len(rows),
  }
