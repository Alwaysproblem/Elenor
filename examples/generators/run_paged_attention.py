"""Run one PagedAttention decode variant end to end (plan §6).

Compiles the generated fixture, builds the host environment and ByteStore
oracle from the scenario, then runs ``Simulator.run(host=...)`` and reuses the
validator's report/trace serialization.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paged_attention_common import (
  load_scenario,
  make_host_environment,
  seed_globals,
  workload_path,
)

from pipeline_validator.compiled_program import serialize_compiled_program
from pipeline_validator.compiler.api import compile_program, dump_compiled_source
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.report import build_report, report_to_json, report_to_text
from pipeline_validator.simulator import Simulator
from pipeline_validator.workload_ir import load_workload_ir

# No scenario override is needed: the pipeline's live Grid window is its
# partition depth, which fits group policy s1's shipped capacity.
SCENARIO_SIM_OVERRIDES: dict[str, object] = {}


def _parse_overrides(specs: list[str]) -> dict[str, object]:
  result: dict[str, object] = {}
  for spec in specs:
    if "=" not in spec:
      raise ValueError(f"override must be KEY=VALUE, got {spec!r}")
    key, _, value = spec.partition("=")
    try:
      result[key.strip()] = int(value, 0)
    except ValueError as exc:
      raise ValueError(f"override {key!r} must be an integer") from exc
  return result


def build_bindings(scenario) -> dict[str, GlobalBinding]:
  return {
    name: GlobalBinding(name, base, size, permission)
    for name, (base, size, permission) in scenario.bindings().items()
  }


def build_oracle(scenario) -> ByteStore:
  store = ByteStore()
  for name, payload in seed_globals(scenario).items():
    store.seed_hbm(scenario.bindings()[name][0], payload)
  return store


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scenario", required=True, help="scenario json emitted by the generator")
  parser.add_argument("--variant", choices=("pipeline", "baseline"), required=True)
  parser.add_argument("--hw-config", default=None, help="grouped HardwareConfig yaml")
  parser.add_argument("--hw-override", action="append", default=[], metavar="KEY=VALUE")
  parser.add_argument("--sim-override", action="append", default=[], metavar="KEY=VALUE")
  parser.add_argument("--group-policy", choices=("s0", "s1", "s2"), default=None)
  parser.add_argument("--context-mode", type=int, default=None, metavar="N")
  parser.add_argument("--device-context-mode", type=int, default=None, metavar="N")
  parser.add_argument("--max-cycles", type=int, default=None)
  parser.add_argument("--memory-trace", action="store_true")
  parser.add_argument("--trace-json", default=None, metavar="PATH")
  parser.add_argument("--compiled-file", default=None, metavar="PATH")
  parser.add_argument("--compiled-output", default=None, metavar="PATH")
  parser.add_argument("--json", action="store_true")
  parser.add_argument("--report", default=None, metavar="PATH")
  parser.add_argument("--detailed", action="store_true")
  args = parser.parse_args(argv)

  scenario = load_scenario(Path(args.scenario), args.variant)
  overrides = dict(SCENARIO_SIM_OVERRIDES)
  overrides.update(_parse_overrides(args.sim_override))
  if args.group_policy is not None:
    overrides["group.policy"] = args.group_policy
  if args.context_mode is not None:
    overrides["context_count"] = args.context_mode
  if args.device_context_mode is not None:
    overrides["device_context_count"] = args.device_context_mode
  if args.max_cycles is not None:
    overrides["max_cycles"] = args.max_cycles
  if args.memory_trace:
    overrides["memory_trace"] = True

  hw = HardwareConfig.from_yaml(args.hw_config) if args.hw_config else HardwareConfig()
  # The decode fixtures declare l1_mode/l2_mode = 1, so the engine's caches
  # must actually hold capacity: the bundled default target resets to mode 0
  # (SPM only), which would silently turn every Gather into a bypass.
  target = hw.memory_target
  hw = hw.with_overrides(
    memory_target=replace(
      target,
      l1=replace(target.l1, reset_mode=scenario.l1_mode),
      l2=replace(target.l2, reset_mode=scenario.l2_mode),
    )
  )
  hw = hw.with_overrides(**_parse_overrides(args.hw_override))
  sim = SimConfig(fidelity="full_memory").with_overrides(**overrides)

  bindings = build_bindings(scenario)
  oracle = build_oracle(scenario)
  ir_path = workload_path(Path(args.scenario), args.variant)
  module = load_workload_ir(ir_path)

  if args.compiled_file is not None:
    from pipeline_validator.compiled_program import parse_compiled_program

    artifact = parse_compiled_program(Path(args.compiled_file).read_text(encoding="utf-8"))
  else:
    artifact = compile_program(
      module,
      hw,
      sim,
      binding_assumptions=bindings,
      source_name=str(ir_path),
    )
  if args.compiled_output is not None:
    path = Path(args.compiled_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_compiled_program(artifact), encoding="utf-8")
    path.with_suffix(".executable.mlir").write_text(
      dump_compiled_source(artifact), encoding="utf-8"
    )
    print(f"[compiled] {path}", file=sys.stderr)

  loaded = load_program(artifact, hw, sim, actual_bindings=bindings)
  environment, _state = make_host_environment(scenario)
  simulator = Simulator(
    hw,
    sim,
    enable_tracer=bool(args.trace_json or args.memory_trace),
    byte_store=oracle,
  )
  print(f"[run] {args.variant}", file=sys.stderr)
  result = simulator.run(loaded, host=environment)
  report = build_report(loaded.compiled.workload_info, result, num_tiles=hw.num_tiles)
  text = (
    json.dumps([json.loads(report_to_json(report))], indent=2)
    if args.json
    else report_to_text(report, detailed=args.detailed)
  )
  if args.trace_json and result.tracer is not None:
    path = Path(args.trace_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.tracer.to_chrome_json(), encoding="utf-8")
    print(f"[trace] {path}", file=sys.stderr)
  if args.report is not None:
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"[report] {path}", file=sys.stderr)
  else:
    print(text)
  return 0 if result.completed else 1


if __name__ == "__main__":
  raise SystemExit(main())
