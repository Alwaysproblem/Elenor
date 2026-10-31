"""Mutated-real-trace regression for the Python trace-audit fallback.

``hardware/scripts/am_gate_cases.py::audit_trace_events`` is the fallback used
only when the in-repo Perfetto ``trace_processor_shell`` is unavailable in the
container.  It must enforce exactly the committed P07 SQL predicates
(``plan/evidence/P07/trace-audit.sql``): tile-qualified joins,
``am_release.ts == am_chain_terminal.ts`` (equality, not just nondecreasing),
and a successful FINAL terminating in a ``chain_done`` terminal.  Each
mutation below edits a REAL full-runtime trace (parser -> compiler -> loader
-> Simulator via ``test_am_terminal_order._run_fixture``) and the audit must
reject it; a pristine trace must pass.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hardware" / "scripts"))

import am_gate_cases  # noqa: E402

from pipeline_validator.tests.test_am_terminal_order import _run_fixture  # noqa: E402


def _trace_events() -> list[dict]:
  simulator, _store, result, _marks = _run_fixture()
  assert result.completed, result.reason
  assert simulator.tracer is not None
  return json.loads(simulator.tracer.to_chrome_json())["traceEvents"]


def _instants(events: list[dict], name: str) -> list[dict]:
  return [e for e in events if e.get("name") == name and e.get("ph") == "i"]


def _chain_done_ready(events: list[dict]) -> list[dict]:
  return [e for e in _instants(events, "boa_result_ready") if e["args"].get("status") == "chain_done"]


class _Ctx:
  """Minimal stand-in for container_check.CaseContext's check/metrics face."""

  def __init__(self):
    self.checks: list[tuple[str, bool]] = []
    self.metrics: dict = {}

  def check(self, name: str, passed: bool, detail: str = "") -> bool:
    self.checks.append((name, bool(passed)))
    return bool(passed)


def _write(tmp_path: Path, events: list[dict], name: str) -> Path:
  path = tmp_path / name
  path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
  return path


def test_fallback_audit_accepts_pristine_real_trace(tmp_path):
  result = am_gate_cases.audit_trace_events(_trace_events())
  assert result["finals"] >= 1 and result["unfinished"] == 0
  assert (
    result["ordered_ts"] == result["ordered_emission"] == result["success_terminals"] == result["finals"]
  ), result
  assert result["terminals"] == result["finals"] and result["fault_terminals"] == 0
  ctx = _Ctx()
  assert am_gate_cases._python_ordering_audit(ctx, _write(tmp_path, _trace_events(), "pristine.json"), "x")
  assert ctx.checks and all(passed for _name, passed in ctx.checks), ctx.checks


def test_fallback_audit_rejects_terminal_shifted_after_release(tmp_path):
  # SQL requires f.ts = t.ts; a nondecreasing-only audit wrongly accepted this.
  shifted = copy.deepcopy(_trace_events())
  (release,) = _instants(shifted, "am_release")
  (terminal,) = _instants(shifted, "am_chain_terminal")
  (ready,) = _chain_done_ready(shifted)
  assert terminal["ts"] == release["ts"]  # the healthy trace publishes at free
  terminal["ts"] = release["ts"] + 1.5
  ready["ts"] = ready["ts"] + 3.0
  result = am_gate_cases.audit_trace_events(shifted)
  assert result["ordered_ts"] < result["finals"], result
  assert result["ordered_emission"] == result["success_terminals"] == result["finals"], result
  ctx = _Ctx()
  assert not am_gate_cases._python_ordering_audit(ctx, _write(tmp_path, shifted, "shifted.json"), "x")
  assert [name for name, passed in ctx.checks if not passed] == [
    "trace-audit-x-python-finals-fully-ordered"
  ], ctx.checks


def test_fallback_audit_joins_are_tile_qualified(tmp_path):
  # The same lease id exists on every Tile; a terminal attributed to another
  # Tile must not satisfy the join for this final.
  cross_tile = copy.deepcopy(_trace_events())
  (terminal,) = _instants(cross_tile, "am_chain_terminal")
  ready = _chain_done_ready(cross_tile)[0]
  assert terminal["args"]["tile_id"] == ready["args"]["tile_id"]
  terminal["args"]["tile_id"] += 1
  result = am_gate_cases.audit_trace_events(cross_tile)
  assert result["ordered_ts"] == result["ordered_emission"] == result["success_terminals"] == 0, result
  assert result["finals"] >= 1
  ctx = _Ctx()
  assert not am_gate_cases._python_ordering_audit(ctx, _write(tmp_path, cross_tile, "cross.json"), "x")
  assert [name for name, passed in ctx.checks if not passed] == [
    "trace-audit-x-python-finals-fully-ordered"
  ], ctx.checks


def test_fallback_audit_requires_chain_done_success_terminals(tmp_path):
  faulted = copy.deepcopy(_trace_events())
  (terminal,) = _instants(faulted, "am_chain_terminal")
  assert terminal["args"]["kind"] == "chain_done"
  terminal["args"]["kind"] = "fault"
  result = am_gate_cases.audit_trace_events(faulted)
  assert result["success_terminals"] == 0 < result["finals"], result
  ctx = _Ctx()
  assert not am_gate_cases._python_ordering_audit(ctx, _write(tmp_path, faulted, "faulted.json"), "x")
  failed = {name for name, passed in ctx.checks if not passed}
  assert failed == {
    "trace-audit-x-python-finals-fully-ordered",
    "trace-audit-x-python-no-fault-terminals",
  }, ctx.checks
