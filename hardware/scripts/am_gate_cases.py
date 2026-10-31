"""P09 shared-AM software protocol/cycle case adapters for the container runner.

Three registered targets, all driven through the real production entry points
(the permanent pytest scenario modules), never a copy of their logic:

- ``shared-am-functional`` (layer F): ``test_am_tile_isolation`` (Q08a/Q08b +
  RPT), ``test_am_slot_reuse`` (Q09 old-event cross-slot) and
  ``test_boa_am``'s T01/Q06 quota/bundle rejections, each via the module's
  ``--case/--out/--seed`` CLI (``pipeline_validator.tests.case_cli``).
- ``two-slot-progress`` (layers F/C): ``test_boa_multi_engine`` /
  ``test_boa_affinity`` CLIs; the request's schema-1 config
  (``hardware/configs/boa-{1,2,4}engine.yaml``) is passed through to the real
  tests, whose ``validator.hw_overrides`` build the actual ``HardwareConfig``.
  ``t06-one-engine-many-handles`` fail-closes unless the active config is the
  1-engine instance; the multi-engine cases consume the matching 2/4 configs.
- ``shared-am-cycle`` (layer C): ``test_am_terminal_order`` plus the committed
  P07 Perfetto audit ``plan/evidence/P07/trace-audit.sql`` executed by the
  in-repo ``examples/artifacts/tools/trace_processor_shell`` against the real
  captured traces.  If that binary is unavailable in the container the
  equivalent ordering audit runs in Python on the same trace JSON (and the
  report says so explicitly); PASS never depends on which path ran.

The fixed-scenario targets pin their hardware configuration: non-empty
``validator.hw_overrides`` is rejected (fail closed, never silently
discarded), mirroring the numeric-profile/soft-conv wiring.
"""

from __future__ import annotations

import json
import sys
from itertools import pairwise
from pathlib import Path

TRACE_AUDIT_SQL = "plan/evidence/P07/trace-audit.sql"
TRACE_PROCESSOR = "examples/artifacts/tools/trace_processor_shell"

# The order of the result sets printed by trace-audit.sql.
_SQL_HEADERS = (
  ("name", "value", "severity"),
  ("slices", "unfinished"),
  ("finals", "ordered_ts", "ordered_emission", "success_terminals"),
  ("terminals", "fault_terminals"),
)


def _request_config(ctx) -> Path:
  """The request's config file, resolved inside the read-only repo mount."""
  rel = ctx.request.get("config_path")
  path = ctx.repo / rel if rel else None
  assert path is not None and path.is_file(), f"request config {rel!r} missing in repo"
  return path


def _fixed_scenario_no_overrides(ctx) -> bool:
  overrides = ctx.config.get("validator", {}).get("hw_overrides", {})
  return ctx.check(
    "fixed-scenario-no-overrides",
    not overrides,
    f"{ctx.request['target']} scenarios pin their hardware configuration; "
    f"validator.hw_overrides must be empty (nothing is silently discarded), got {overrides}",
  )


def _run_module_case(ctx, module: str, case: str, *, config: Path | None = None, label: str = "") -> bool:
  """Run one module CLI case and assert on its real report."""
  prefix = f"{label}:{case}" if label else case
  out = ctx.artifacts_dir()
  argv = [
    sys.executable,
    "-m",
    module,
    "--case",
    case,
    "--out",
    str(out),
    "--seed",
    str(ctx.request["seed"]),
  ]
  if config is not None:
    argv += ["--config", str(config)]
  code = ctx.run(
    argv,
    ctx.repo,
    f"{prefix}-module",
    env={"PYTHONPATH": str(ctx.repo)},
    timeout=3600,
  )
  ok = ctx.check(f"{prefix}-exit", code == 0, f"exit={code}; see command log")

  report_path = out / f"{case}.report.json"
  report = None
  if report_path.is_file():
    try:
      report = json.loads(report_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
      ok &= ctx.check(f"{prefix}-report-parses", False, f"invalid JSON: {exc}")
  else:
    ok &= ctx.check(f"{prefix}-report-written", False, f"{report_path} missing")

  if report is not None:
    ok &= ctx.check(
      f"{prefix}-status",
      report.get("status") == "PASS",
      f"status={report.get('status')!r} counts={report.get('counts')}",
    )
    for item in report.get("checks", []):
      ok &= ctx.check(f"{prefix}:{item['id']}", item["pass"], item.get("detail", ""))
    metrics = report.get("metrics", {})
    progress = metrics.get("progress")
    if isinstance(progress, dict) and isinstance(progress.get("cycles"), int):
      ctx.metrics["cycles"] = progress["cycles"]
    for key in ("cycles", "blocked_cycles", "free_cycles"):
      if key in metrics:
        ctx.metrics[key] = metrics[key]

  ctx.record_artifact(report_path, deterministic=True)
  for trace in sorted(out.glob(f"{case}*.trace.json")):
    ctx.record_artifact(trace, deterministic=True)
  return bool(ok)



def case_q09_old_event_cross_slot(ctx) -> bool:
  """Q09: FINAL(slot0) -> ALLOC(slot0) -> await(old event), no intermediate reap."""
  return _fixed_scenario_no_overrides(ctx) and _run_module_case(
    ctx, "pipeline_validator.tests.test_am_slot_reuse", "q09-old-event-cross-slot"
  )


def case_q08_tile_isolation(ctx) -> bool:
  """Q08a/Q08b per-Tile allocator isolation plus RPT per-Tile report attribution."""
  return _fixed_scenario_no_overrides(ctx) and _run_module_case(
    ctx, "pipeline_validator.tests.test_am_tile_isolation", "q08-tile-isolation"
  )


def case_t01_q06_over_bundle_reject(ctx) -> bool:
  """T01/Q06: a third FINAL-less chain request is a permanent error with no
  side effect; continuation/park/cancel semantics stay intact."""
  return _fixed_scenario_no_overrides(ctx) and _run_module_case(
    ctx, "pipeline_validator.tests.test_boa_am", "t01-q06-over-bundle-reject"
  )


# --- two-slot-progress --------------------------------------------------------


def case_t06_one_engine_many_handles(ctx) -> bool:
  """T06-P05-3: one engine legally serves 4 live handles at quota 2 (real
  execution under boa-1engine.yaml) and 8 handles at the allocator level."""
  config = _request_config(ctx)
  ok = _run_module_case(
    ctx,
    "pipeline_validator.tests.test_boa_multi_engine",
    "t06-one-engine-many-handles",
    config=config,
    label="engine-pool",
  )
  ok &= _run_module_case(
    ctx,
    "pipeline_validator.tests.test_boa_affinity",
    "one-engine-many-handles",
    config=config,
    label="affinity",
  )
  return bool(ok)


def case_engine_pool_progress(ctx) -> bool:
  """Config-driven real execution: 4 live handles under quota 2, deterministic
  estimate selection, busy-affinity no-migration (>=2 engines), 2-engine
  four-context/eight-handle and fault-drain scenarios (2-engine config)."""
  return _run_module_case(
    ctx,
    "pipeline_validator.tests.test_boa_multi_engine",
    "engine-pool-progress",
    config=_request_config(ctx),
  )


def case_engine_affinity_units(ctx) -> bool:
  """Grant-time least-live affinity, live-handle accounting, reset/abort
  accounting and the config-matched legal load distribution."""
  return _run_module_case(
    ctx,
    "pipeline_validator.tests.test_boa_affinity",
    "engine-affinity-units",
    config=_request_config(ctx),
  )


# --- shared-am-cycle ----------------------------------------------------------


def _parse_tp_sections(stdout: str) -> dict[tuple[str, ...], list[list[str]]]:
  """Split ``trace_processor_shell`` CSV-ish output into per-statement rows."""
  sections: dict[tuple[str, ...], list[list[str]]] = {}
  header: tuple[str, ...] | None = None
  for raw in stdout.splitlines():
    line = raw.strip()
    if not line or line.startswith("[") or line.startswith("Loading trace"):
      continue
    fields = [field.strip() for field in line.split(",")]
    if fields and fields[0].startswith('"'):
      header = tuple(field.strip('"') for field in fields)
      if header not in _SQL_HEADERS:
        raise ValueError(f"unexpected trace-audit result header {header}")
      sections.setdefault(header, [])
      continue
    if header is None:
      raise ValueError(f"trace-audit data row before any header: {line!r}")
    sections[header].append(fields)
  missing = [list(head) for head in _SQL_HEADERS if head not in sections]
  if missing:
    raise ValueError(f"trace-audit statements missing from output: {missing}")
  return sections


def audit_trace_events(events: list) -> dict:
  """Python equivalent of trace-audit.sql over raw chrome ``traceEvents``.

  Mirrors the SQL predicates exactly rather than merely checking a
  nondecreasing chain: joins are qualified by ``(tile_id, lease_id)`` /
  ``(tile_id, event_id)`` so a same-lease event from another Tile never
  satisfies a join, ``am_release.ts`` must EQUAL ``am_chain_terminal.ts``
  (shifted-later terminals are rejected, not just early ones), and a
  successful final must terminate in a ``chain_done`` terminal.  Same-cycle
  stages are additionally ordered by emission index.
  """
  slices = [event for event in events if event.get("ph") == "X"]
  marks = [
    dict(event.get("args", {}), name=event["name"], ts=event["ts"], seq=index)
    for index, event in enumerate(events)
    if event.get("ph") == "i" and event.get("cat") in ("SharedAM", "BoaResult")
  ]
  finals = [
    m for m in marks if m["name"] == "boa_result_ready" and m.get("status") == "chain_done"
  ]
  terminals = [m for m in marks if m["name"] == "am_chain_terminal"]
  ordered_ts = ordered_emission = success_terminals = 0
  for record in finals:
    tile, lease, event_id = record.get("tile_id"), record.get("lease_id"), record.get("event_id")

    def _one(name: str, **match: object) -> list:
      return [
        m
        for m in marks
        if m["name"] == name and all(m.get(key) == value for key, value in match.items())
      ]

    commit = _one("am_commit", final=True, lease_id=lease, tile_id=tile)
    output = _one("boa_result_output_written", event_id=event_id, tile_id=tile)
    release = _one("am_release", lease_id=lease, tile_id=tile)
    terminal = [m for m in terminals if m.get("lease_id") == lease and m.get("tile_id") == tile]
    if not (len(commit) == len(output) == len(release) == len(terminal) == 1):
      continue
    stages = [commit[0], output[0], release[0], terminal[0], record]
    if terminal[0].get("kind") == "chain_done":
      success_terminals += 1
    if (
      stages[0]["ts"] <= stages[1]["ts"]
      and stages[1]["ts"] <= stages[2]["ts"]
      and stages[2]["ts"] == stages[3]["ts"]
      and stages[3]["ts"] <= stages[4]["ts"]
    ):
      ordered_ts += 1
    if all(a["seq"] < b["seq"] for a, b in pairwise(stages)):
      ordered_emission += 1
  return {
    "slices": len(slices),
    "unfinished": sum(1 for s in slices if s.get("dur", 0) < 0),
    "finals": len(finals),
    "ordered_ts": ordered_ts,
    "ordered_emission": ordered_emission,
    "success_terminals": success_terminals,
    "terminals": len(terminals),
    "fault_terminals": sum(1 for t in terminals if t.get("kind") == "fault"),
  }


def _python_ordering_audit(ctx, trace_path: Path, label: str) -> bool:
  """Equivalent of trace-audit.sql on the raw chrome JSON (fallback only)."""
  note = "fallback Python audit; trace_processor_shell unavailable"
  events = json.loads(trace_path.read_text(encoding="utf-8"))["traceEvents"]
  result = audit_trace_events(events)
  ok = ctx.check(
    f"trace-audit-{label}-python-no-unfinished-slices",
    result["unfinished"] == 0,
    f"slices={result['slices']} unfinished={result['unfinished']} ({note})",
  )
  ok &= ctx.check(
    f"trace-audit-{label}-python-finals-fully-ordered",
    result["finals"] >= 1
    and result["ordered_ts"] == result["ordered_emission"]
    and result["success_terminals"] == result["finals"],
    f"finals={result['finals']} ordered_ts={result['ordered_ts']} "
    f"ordered_emission={result['ordered_emission']} "
    f"success_terminals={result['success_terminals']} ({note})",
  )
  ok &= ctx.check(
    f"trace-audit-{label}-python-no-fault-terminals",
    result["terminals"] == result["finals"] and result["fault_terminals"] == 0,
    f"terminals={result['terminals']} fault_terminals={result['fault_terminals']} ({note})",
  )
  ctx.metrics[f"{label}_slices"] = result["slices"]
  ctx.metrics[f"{label}_finals"] = result["finals"]
  return bool(ok)


def _trace_audit(ctx, trace_path: Path, label: str) -> bool:
  """Run the committed P07 SQL via the in-repo trace_processor_shell."""
  shell = ctx.repo / TRACE_PROCESSOR
  sql = ctx.repo / TRACE_AUDIT_SQL
  if not shell.is_file():
    # The locked image mounts the repo read-only, so the binary is normally
    # present; if it is not, the equivalent ordering audit runs in Python.
    return _python_ordering_audit(ctx, trace_path, label)
  code, stdout = ctx.run_capture(
    [str(shell), "-q", str(sql), str(trace_path)],
    ctx.repo,
    f"trace-audit-{label}",
  )
  ok = ctx.check(f"trace-audit-{label}-exit", code == 0, f"exit={code}; see command log")
  try:
    sections = _parse_tp_sections(stdout)
  except ValueError as exc:
    return bool(ok & ctx.check(f"trace-audit-{label}-parses", False, str(exc)))
  ok &= ctx.check(
    f"trace-audit-{label}-parses", True, f"{len(sections)} statements parsed"
  )

  stats = sections[_SQL_HEADERS[0]]
  ok &= ctx.check(
    f"trace-audit-{label}-stats-clean",
    not stats,
    f"error/data_loss stats rows: {stats}",
  )
  (slice_row,) = sections[_SQL_HEADERS[1]]
  ok &= ctx.check(
    f"trace-audit-{label}-no-unfinished-slices",
    int(slice_row[1]) == 0,
    f"slices={slice_row[0]} unfinished={slice_row[1]}",
  )
  (final_row,) = sections[_SQL_HEADERS[2]]
  finals, ordered_ts, ordered_emission, success_terminals = (int(value) for value in final_row)
  ok &= ctx.check(
    f"trace-audit-{label}-finals-fully-ordered",
    finals >= 1 and ordered_ts == ordered_emission == success_terminals == finals,
    f"finals={finals} ordered_ts={ordered_ts} ordered_emission={ordered_emission} "
    f"success_terminals={success_terminals}",
  )
  (terminal_row,) = sections[_SQL_HEADERS[3]]
  ok &= ctx.check(
    f"trace-audit-{label}-no-fault-terminals",
    int(terminal_row[0]) == finals and int(terminal_row[1]) == 0,
    f"terminals={terminal_row[0]} fault_terminals={terminal_row[1]}",
  )
  ctx.metrics[f"{label}_slices"] = int(slice_row[0])
  ctx.metrics[f"{label}_finals"] = finals
  return bool(ok)


def case_terminal_order_backpressure(ctx) -> bool:
  """S-P07-1/2/3: full-runtime FINAL ordering by identity (free + 20k-cycle
  backpressure), fault cleanup cycles, plus the independent Perfetto SQL audit
  on both captured traces."""
  ok = _fixed_scenario_no_overrides(ctx)
  ok &= _run_module_case(
    ctx, "pipeline_validator.tests.test_am_terminal_order", "terminal-order-backpressure"
  )
  artifacts = ctx.artifacts_dir()
  for label in ("free", "blocked"):
    trace = artifacts / f"terminal-order-backpressure.{label}.trace.json"
    if not ctx.check(f"trace-audit-{label}-trace-exists", trace.is_file(), str(trace)):
      ok = False
      continue
    ok &= _trace_audit(ctx, trace, label)
  return bool(ok)


def case_manager_terminal_units(ctx) -> bool:
  """S-P07-4/5: non-FINAL commits without terminal/output, inflight-access
  quiesce, single-fault double abort, release refusal, bounded terminal ring."""
  return _fixed_scenario_no_overrides(ctx) and _run_module_case(
    ctx, "pipeline_validator.tests.test_am_terminal_order", "manager-terminal-units"
  )
