"""Command-line interface for the ELENOR pipeline validator."""

from __future__ import annotations

import argparse
import json
import sys

from xdsl.utils.exceptions import ParseError, VerifyException

from .config import MAX_CONTEXT_COUNT, HardwareConfig, SimConfig
from .report import build_report, report_to_json, report_to_text
from .simulator import Simulator
from .trace import trace_to_html
from .workload_ir import (
  load_workload_ir,
  print_workload_ir,
  verify_workload_ir,
)
from .workloads import ALL_WORKLOADS, Workload


def _list_workloads() -> None:
  print("Available workloads:")
  for wl_cls in ALL_WORKLOADS:
    wl = wl_cls()
    print(f"  {wl.name:<12}  {wl.description[:80]}")


def _parse_overrides(items: list[str]) -> dict[str, str | float | int]:
  out: dict[str, str | float | int] = {}
  for item in items or []:
    if "=" not in item:
      raise SystemExit(f"bad override '{item}', expected key=value")
    k, v = item.split("=", 1)
    try:
      out[k] = int(v)
    except ValueError:
      try:
        out[k] = float(v)
      except ValueError:
        out[k] = v
  return out


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(
    prog="pipeline_validator",
    description=(
      "ELENOR runtime pipeline efficiency validator "
      "(1 Tile Group + 4 Compute Tiles, cycle-accurate)."
    ),
  )
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument("-l", "--list", action="store_true", help="list available workloads and exit")
  mode.add_argument("-w", "--workload", default=None, help="workload to run")
  mode.add_argument("-a", "--all", action="store_true", help="run all workloads")
  mode.add_argument("--ir-file", metavar="PATH", help="load and run one external IR module")
  parser.add_argument(
    "--hw-override",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="override a HardwareConfig field, e.g. clock_mhz=2000",
  )
  parser.add_argument(
    "--sim-override",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="override a SimConfig field, e.g. trace=True",
  )
  parser.add_argument(
    "--context-mode",
    type=int,
    default=None,
    metavar="N",
    help="Tile UCE execution context count: 1-8 (default: 1)",
  )
  parser.add_argument(
    "--device-context-mode",
    type=int,
    default=None,
    metavar="N",
    help="device execution context (TileGroup slot) count: 1-8 (default: 1)",
  )
  parser.add_argument("--max-cycles", type=int, default=None, help="cycle cap (default 2_000_000)")
  parser.add_argument("--trace", action="store_true", help="enable per-cycle trace dump")
  parser.add_argument(
    "--trace-json",
    default=None,
    metavar="PATH",
    help="write Perfetto/Chrome trace.json to PATH (enables tracing)",
  )
  parser.add_argument(
    "--trace-html",
    default=None,
    metavar="PATH",
    help="write standalone trace.html to PATH (enables tracing)",
  )
  parser.add_argument(
    "--print-ir",
    action="store_true",
    help="print workload IR (custom assembly) and exit (no simulation)",
  )
  parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
  parser.add_argument("--report", default=None, help="write report to this path (default: stdout)")
  args = parser.parse_args(argv)

  if args.context_mode is not None and not 1 <= args.context_mode <= MAX_CONTEXT_COUNT:
    parser.error("--context-mode must be between 1 and 8")
  if args.device_context_mode is not None and not 1 <= args.device_context_mode <= MAX_CONTEXT_COUNT:
    parser.error("--device-context-mode must be between 1 and 8")
  if args.list:
    _list_workloads()
    return 0

  hw = HardwareConfig().with_overrides(**_parse_overrides(args.hw_override))
  sim_overrides = _parse_overrides(args.sim_override)
  if args.max_cycles is not None:
    sim_overrides["max_cycles"] = args.max_cycles
  if args.context_mode is not None:
    sim_overrides["context_count"] = args.context_mode
  if args.device_context_mode is not None:
    sim_overrides["device_context_count"] = args.device_context_mode
  if args.trace:
    sim_overrides["trace"] = True
  sim_cfg = SimConfig().with_overrides(**sim_overrides)

  workloads: list[Workload] = []
  if args.ir_file is not None:
    ir_path = args.ir_file
    try:
      module = load_workload_ir(args.ir_file)
      task = verify_workload_ir(module)
    except (OSError, UnicodeError, ParseError, VerifyException) as exc:
      print(f"failed to load IR '{ir_path}': {exc}", file=sys.stderr)
      return 2
    workloads.append(
      Workload(
        name=task.sym_name.data,
        module=module,
        expected={},
        description=f"External IR: {ir_path}",
      )
    )
  else:
    names = [args.workload or "pow"]
    if args.all:
      names = [c().name for c in ALL_WORKLOADS]
    for name in names:
      match = next((c for c in ALL_WORKLOADS if c().name == name), None)
      if match is None:
        print(f"unknown workload '{name}'", file=sys.stderr)
        _list_workloads()
        return 2
      workloads.append(match())

  if args.print_ir:
    for idx, wl in enumerate(workloads):
      if idx:
        sys.stdout.write("\n")
      sys.stdout.write(print_workload_ir(wl.module))
    return 0

  outputs = []
  overall_pass = True
  enable_tracer = bool(args.trace_json or args.trace_html)
  for wl in workloads:
    sim = Simulator(hw, sim_cfg, enable_tracer=enable_tracer)
    result = sim.run(wl.module)
    rep = build_report(wl, result)
    outputs.append(rep)
    if not all(ch.get("pass", False) for ch in rep.checks):
      overall_pass = False
    if enable_tracer and result.tracer is not None:
      if args.trace_json:
        path = args.trace_json
        if len(workloads) > 1:
          path = path.replace(".json", f"_{wl.name}.json")
        with open(path, "w") as f:
          f.write(result.tracer.to_chrome_json())
        print(f"trace (perfetto json) written to {path}", file=sys.stderr)
      if args.trace_html:
        path = args.trace_html
        if len(workloads) > 1:
          path = path.replace(".html", f"_{wl.name}.html")
        with open(path, "w") as f:
          f.write(trace_to_html(result.tracer))
        print(f"trace (html) written to {path}", file=sys.stderr)

  text = (
    "\n".join(report_to_text(r) for r in outputs)
    if not args.json
    else json.dumps([json.loads(report_to_json(r)) for r in outputs], indent=2)
  )

  if args.report:
    with open(args.report, "w") as f:
      f.write(text + "\n")
    print(f"report written to {args.report}", file=sys.stderr)
  else:
    print(text)

  return 0 if overall_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
