"""Command-line interface for the ELENOR pipeline validator."""

from __future__ import annotations

import argparse
import json
import sys

from .config import HardwareConfig, SimConfig
from .report import build_report, report_to_json, report_to_text
from .simulator import Simulator
from .trace import trace_to_html
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
        # try int, then float, then str
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
        description="ELENOR runtime pipeline efficiency validator "
        "(1 Tile Group + 4 Compute Tiles, cycle-accurate).",
    )
    parser.add_argument("-l",
                        "--list",
                        action="store_true",
                        help="list available workloads and exit")
    parser.add_argument("-w",
                        "--workload",
                        default="matmul",
                        help="workload to run (default: matmul)")
    parser.add_argument("-a",
                        "--all",
                        action="store_true",
                        help="run all workloads")
    parser.add_argument(
        "--hw-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a HardwareConfig field, e.g. clock_mhz=2000")
    parser.add_argument("--sim-override",
                        action="append",
                        default=[],
                        metavar="KEY=VALUE",
                        help="override a SimConfig field, e.g. trace=True")
    parser.add_argument("--max-cycles",
                        type=int,
                        default=None,
                        help="cycle cap (default 2_000_000)")
    parser.add_argument("--trace",
                        action="store_true",
                        help="enable per-cycle trace dump")
    parser.add_argument(
        "--trace-json",
        default=None,
        metavar="PATH",
        help="write Perfetto/Chrome trace.json to PATH (enables tracing)")
    parser.add_argument(
        "--trace-html",
        default=None,
        metavar="PATH",
        help="write standalone trace.html to PATH (enables tracing)")
    parser.add_argument("--print-ir",
                        action="store_true",
                        help="print the TileGroupTask/Tile IR and exit (no simulation)")
    parser.add_argument("--json",
                        action="store_true",
                        help="emit JSON instead of text")
    parser.add_argument("--report",
                        default=None,
                        help="write report to this path (default: stdout)")
    args = parser.parse_args(argv)

    if args.list:
        _list_workloads()
        return 0

    hw = HardwareConfig().with_overrides(**_parse_overrides(args.hw_override))
    sim_overrides = _parse_overrides(args.sim_override)
    if args.max_cycles is not None:
        sim_overrides["max_cycles"] = args.max_cycles
    if args.trace:
        sim_overrides["trace"] = True
    sim_cfg = SimConfig().with_overrides(**sim_overrides)

    names = [args.workload]
    if args.all:
        names = [c().name for c in ALL_WORKLOADS]

    workloads: list[Workload] = []
    for n in names:
        match = next((c for c in ALL_WORKLOADS if c().name == n), None)
        if match is None:
            print(f"unknown workload '{n}'", file=sys.stderr)
            _list_workloads()
            return 2
        workloads.append(match())

    if args.print_ir:
        for wl in workloads:
            print(wl.task.pretty_print())
        return 0


    outputs = []
    overall_pass = True
    enable_tracer = bool(args.trace_json or args.trace_html)
    for wl in workloads:
        sim = Simulator(hw, sim_cfg, enable_tracer=enable_tracer)
        result = sim.run(wl.task)
        rep = build_report(wl, result)
        outputs.append(rep)
        if not all(ch.get("pass", False) for ch in rep.checks):
            overall_pass = False
        # write trace outputs if requested
        if enable_tracer and result.tracer is not None:
            if args.trace_json:
                path = args.trace_json
                if len(workloads) > 1:
                    path = path.replace(".json", f"_{wl.name}.json")
                with open(path, "w") as f:
                    f.write(result.tracer.to_chrome_json())
                print(f"trace (perfetto json) written to {path}",
                      file=sys.stderr)
            if args.trace_html:
                path = args.trace_html
                if len(workloads) > 1:
                    path = path.replace(".html", f"_{wl.name}.html")
                with open(path, "w") as f:
                    f.write(trace_to_html(result.tracer))
                print(f"trace (html) written to {path}", file=sys.stderr)

    # emit
    text = "\n".join(report_to_text(r) for r in outputs) if not args.json \
        else json.dumps([json.loads(report_to_json(r)) for r in outputs], indent=2)

    if args.report:
        with open(args.report, "w") as f:
            f.write(text + "\n")
        print(f"report written to {args.report}", file=sys.stderr)
    else:
        print(text)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
