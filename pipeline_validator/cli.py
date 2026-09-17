"""Command-line interface for the ELENOR pipeline validator."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

import yaml
from xdsl.utils.exceptions import ParseError, VerifyException

from .compiled_program import (
  CompiledProgram,
  WorkloadInfo,
  dump_executable_ir,
  parse_compiled_program,
  serialize_compiled_program,
)
from .config import _HW_YAML_PATH_TO_FIELD, MAX_CONTEXT_COUNT, HardwareConfig, SimConfig
from .execution_ir import GlobalBinding
from .loader import load_program
from .profiles import ProfileBytes, ProfileLevelSource
from .report import build_report, report_to_json, report_to_text
from .simulator import Simulator
from .trace import trace_to_html
from .workload_ir import load_workload_ir, print_workload_ir, verify_workload_ir
from .workloads import ALL_WORKLOADS, PowWorkload, Workload

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "examples" / "artifacts" / "compiled"


def _parse_input_binding(spec: str) -> GlobalBinding:
  """Parse ``NAME=BASE:SIZE:PERM`` into a global launch binding."""
  error = f"invalid --input-binding '{spec}': expected NAME=BASE:SIZE:PERM"
  try:
    name, value = spec.split("=", 1)
    base, size, permissions = value.split(":")
    if not name or permissions not in {"r", "w", "rw"}:
      raise ValueError
    return GlobalBinding(name, int(base, 0), int(size, 0), permissions)
  except ValueError:
    raise ValueError(error) from None


def _list_workloads() -> None:
  print("Available workloads:")
  for wl_cls in ALL_WORKLOADS:
    wl = wl_cls()
    print(f"  {wl.name:<12}  {wl.description[:80]}")


def _parse_overrides(items: list[str]) -> dict[str, str | float | int]:
  out: dict[str, str | float | int] = {}
  for item in items or []:
    if "=" not in item:
      raise ValueError(f"bad override '{item}', expected key=value")
    key, value = item.split("=", 1)
    if not key or key in out:
      raise ValueError(f"duplicate or empty override key '{key}'")
    try:
      out[key] = int(value)
    except ValueError:
      try:
        out[key] = float(value)
      except ValueError:
        out[key] = value
  return out


def _parse_profile_bytes(items: list[str]) -> dict[tuple[str, int], ProfileBytes]:
  overrides: dict[tuple[str, int], ProfileBytes] = {}
  for item in items or []:
    error = f"invalid --profile-bytes '{item}': expected LEVEL:MODE=SPM_BYTES:CACHE_BYTES"
    try:
      selector, values = item.split("=", 1)
      level, mode_text = selector.split(":", 1)
      spm_text, cache_text = values.split(":", 1)
      level = level.lower()
      mode = int(mode_text, 0)
      if level not in ("l1", "l2"):
        raise ValueError
      value = ProfileBytes(int(spm_text, 0), int(cache_text, 0))
    except ValueError:
      raise ValueError(error) from None
    key = (level, mode)
    if key in overrides:
      raise ValueError(f"duplicate --profile-bytes override for {level}:{mode}")
    overrides[key] = value
  return overrides


def _apply_profile_bytes(
  hw: HardwareConfig, overrides: Mapping[tuple[str, int], ProfileBytes]
) -> HardwareConfig:
  source = hw.profile_source
  levels: dict[str, ProfileLevelSource] = {}
  for level in ("l1", "l2"):
    modes = dict(getattr(source, level).modes)
    for (override_level, mode), value in overrides.items():
      if override_level != level:
        continue
      if mode not in modes:
        raise ValueError(f"--profile-bytes references unknown {level} mode {mode}")
      modes[mode] = value
    levels[level] = ProfileLevelSource(modes)
  return replace(hw, profile_source=replace(source, l1=levels["l1"], l2=levels["l2"]))


def _plain_yaml_value(value):
  if is_dataclass(value) and not isinstance(value, type):
    return _plain_yaml_value(asdict(value))
  if isinstance(value, Mapping):
    return {key: _plain_yaml_value(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_plain_yaml_value(item) for item in value]
  return value


def _set_nested(root: dict, dotted: str, value) -> None:
  parts = dotted.split(".")
  node = root
  for part in parts[:-1]:
    child = node.setdefault(part, {})
    if not isinstance(child, dict):
      raise ValueError(f"hardware snapshot path collision at '{dotted}'")
    node = child
  node[parts[-1]] = _plain_yaml_value(value)


def _hardware_yaml(hw: HardwareConfig) -> str:
  """Serialize every HardwareConfig field, including typed profile trees."""
  document: dict = {"schema_version": 2}
  for path, field_name in _HW_YAML_PATH_TO_FIELD.items():
    _set_nested(document, path, getattr(hw, field_name))
  return yaml.safe_dump(document, sort_keys=False, allow_unicode=True)


def _artifact_paths(output: Path) -> tuple[Path, Path, Path, Path]:
  stem = output.with_suffix("") if output.suffix else output
  return (output, Path(f"{stem}.exec.txt"), Path(f"{stem}.compiled.mlir.txt"), Path(f"{stem}.target.yaml"))


def _write_content_addressed(files: Mapping[Path, str]) -> None:
  for path, content in files.items():
    if path.exists():
      try:
        current = path.read_text(encoding="utf-8")
      except (OSError, UnicodeError) as exc:
        raise OSError(f"failed to read existing artifact '{path}': {exc}") from exc
      if current != content:
        raise ValueError(f"refusing to overwrite existing artifact with different content: '{path}'")
  for path, content in files.items():
    if path.exists():
      continue
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _persist_program(
  program: CompiledProgram, hw: HardwareConfig, compiled_source: str, requested_output: str | None
) -> Path:
  output = (
    Path(requested_output)
    if requested_output is not None
    else _DEFAULT_ARTIFACT_DIR / f"{program.artifact_hash}.json"
  )
  artifact, executable, compiled_mlir, target = _artifact_paths(output)
  _write_content_addressed(
    {
      artifact: serialize_compiled_program(program),
      executable: dump_executable_ir(program),
      compiled_mlir: compiled_source,
      target: _hardware_yaml(hw),
    }
  )
  return artifact


def _multi_output_path(path: str, workload_name: str, count: int) -> Path:
  result = Path(path)
  if count == 1:
    return result
  return result.with_name(f"{result.stem}_{workload_name}{result.suffix}")


def _load_external_workload(path: str) -> Workload:
  module = load_workload_ir(path)
  entry = verify_workload_ir(module)
  name = entry.sym_name.data
  return Workload(name=name, module=module, expected={}, description=f"External IR: {path}")


def _make_builtin_workload(
  workload_type: type[PowWorkload], hw: HardwareConfig, context_count: int
) -> Workload:
  return workload_type(hw=hw, context_count=context_count)


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(
    prog="pipeline_validator",
    description=(
      "ELENOR runtime pipeline efficiency validator (1 Tile Group + 4 Compute Tiles, cycle-accurate)."
    ),
  )
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument("-l", "--list", action="store_true", help="list available workloads and exit")
  mode.add_argument("-w", "--workload", default=None, help="workload to compile and run")
  mode.add_argument("-a", "--all", action="store_true", help="compile and run all workloads")
  mode.add_argument("--ir-file", metavar="PATH", help="load one external source IR module")
  mode.add_argument("--compiled-file", metavar="PATH", help="load and run one compiled JSON artifact")
  parser.add_argument(
    "--compile-only",
    action="store_true",
    help="compile and persist source input without loading or running it",
  )
  parser.add_argument(
    "--compiled-output", metavar="PATH", help="write the compiled JSON artifact and sibling dumps to PATH"
  )
  parser.add_argument(
    "--profile-bytes",
    action="append",
    default=[],
    metavar="LEVEL:MODE=SPM_BYTES:CACHE_BYTES",
    help="override one existing source-compilation profile using explicit per-bank bytes",
  )
  parser.add_argument(
    "--input-binding",
    action="append",
    default=[],
    metavar="NAME=BASE:SIZE:PERM",
    help="bind a global input: NAME=BASE:SIZE:PERM",
  )
  parser.add_argument(
    "--hw-override",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="override a HardwareConfig field, e.g. clock_mhz=2000",
  )
  parser.add_argument(
    "--hw-config", default=None, metavar="PATH", help="load grouped HardwareConfig values from a YAML file"
  )
  parser.add_argument(
    "--sim-override",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="override a SimConfig field, e.g. group.action_capacity=16",
  )
  parser.add_argument(
    "--group-policy",
    choices=("s0", "s1", "s2"),
    default=None,
    help="select the Group hardware scheduler policy",
  )
  parser.add_argument(
    "--context-mode",
    type=int,
    default=None,
    metavar="N",
    help="exact hardware Tile UCE context count per Tile: 1-8 (default: 1)",
  )
  parser.add_argument(
    "--device-context-mode",
    type=int,
    default=None,
    metavar="N",
    help="CPU outstanding Group-launch limit: 1-8 (default: 1)",
  )
  parser.add_argument("--max-cycles", type=int, default=None, help="cycle cap (default 2_000_000)")
  parser.add_argument("--trace", action="store_true", help="enable per-cycle trace dump")
  parser.add_argument(
    "--memory-trace",
    action="store_true",
    help="emit memory lanes/counters/flows in the trace and memory peaks in the report",
  )
  parser.add_argument(
    "--trace-json", default=None, metavar="PATH", help="write Perfetto/Chrome trace.json to PATH"
  )
  parser.add_argument(
    "--trace-html", default=None, metavar="PATH", help="write standalone trace.html to PATH"
  )
  parser.add_argument(
    "--print-ir", action="store_true", help="print author source IR and exit without compilation"
  )
  parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
  parser.add_argument("--report", default=None, help="write report to this path (default: stdout)")
  args = parser.parse_args(argv)

  if args.compiled_file is not None and (
    args.compile_only or args.compiled_output is not None or args.profile_bytes or args.print_ir
  ):
    parser.error(
      "--compiled-file is mutually exclusive with --compile-only, --compiled-output, "
      "--profile-bytes, and --print-ir"
    )
  if args.print_ir and (args.compile_only or args.compiled_output is not None or args.profile_bytes):
    parser.error("--print-ir is mutually exclusive with compilation output options")
  if args.all and args.compiled_output is not None:
    parser.error("--compiled-output requires one source input and cannot be used with --all")

  try:
    parsed_bindings = [_parse_input_binding(spec) for spec in args.input_binding]
    bindings: dict[str, GlobalBinding] = {}
    for binding in parsed_bindings:
      if binding.name in bindings:
        raise ValueError(f"duplicate --input-binding for '{binding.name}'")
      bindings[binding.name] = binding
    hw_overrides = _parse_overrides(args.hw_override)
    sim_overrides = _parse_overrides(args.sim_override)
    profile_overrides = _parse_profile_bytes(args.profile_bytes)
  except ValueError as exc:
    parser.error(str(exc))

  if args.context_mode is not None and not 1 <= args.context_mode <= MAX_CONTEXT_COUNT:
    parser.error("--context-mode must be between 1 and 8")
  if args.device_context_mode is not None and not 1 <= args.device_context_mode <= MAX_CONTEXT_COUNT:
    parser.error("--device-context-mode must be between 1 and 8")
  if args.list:
    _list_workloads()
    return 0

  try:
    hw = HardwareConfig.from_yaml(args.hw_config) if args.hw_config else HardwareConfig()
    hw = hw.with_overrides(**hw_overrides)
    if profile_overrides:
      hw = _apply_profile_bytes(hw, profile_overrides)
  except (OSError, TypeError, ValueError) as exc:
    print(f"failed to load hardware configuration: {exc}", file=sys.stderr)
    return 2

  if args.group_policy is not None:
    if "group.policy" in sim_overrides:
      parser.error("use either --group-policy or --sim-override group.policy=..., not both")
    sim_overrides["group.policy"] = args.group_policy
  if args.max_cycles is not None:
    sim_overrides["max_cycles"] = args.max_cycles
  if args.context_mode is not None:
    sim_overrides["context_count"] = args.context_mode
  if args.device_context_mode is not None:
    sim_overrides["device_context_count"] = args.device_context_mode
  if args.trace:
    sim_overrides["trace"] = True
  if args.memory_trace:
    sim_overrides["memory_trace"] = True
  try:
    sim_cfg = SimConfig().with_overrides(**sim_overrides)
  except (TypeError, ValueError) as exc:
    print(f"invalid input: {exc}", file=sys.stderr)
    return 2

  if args.compiled_file is not None:
    try:
      print(f"[load] compiled artifact {args.compiled_file}", file=sys.stderr)
      program = parse_compiled_program(Path(args.compiled_file).read_text(encoding="utf-8"))
      loaded_programs = [load_program(program, hw, sim_cfg, actual_bindings=bindings)]
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
      print(f"failed to load compiled artifact '{args.compiled_file}': {exc}", file=sys.stderr)
      return 2
  else:
    try:
      if args.ir_file is not None:
        workloads = [_load_external_workload(args.ir_file)]
      else:
        names = [args.workload or "pow"]
        if args.all:
          names = [workload_type().name for workload_type in ALL_WORKLOADS]
        workloads = []
        for name in names:
          workload_type = next((item for item in ALL_WORKLOADS if item().name == name), None)
          if workload_type is None:
            print(f"unknown workload '{name}'", file=sys.stderr)
            _list_workloads()
            return 2
          workloads.append(_make_builtin_workload(workload_type, hw, sim_cfg.context_count))
    except (OSError, UnicodeError, ParseError, VerifyException, TypeError, ValueError) as exc:
      source = args.ir_file or args.workload or "pow"
      print(f"failed to load source IR '{source}': {exc}", file=sys.stderr)
      return 2

    if args.print_ir:
      for index, workload in enumerate(workloads):
        if index:
          sys.stdout.write("\n")
        sys.stdout.write(print_workload_ir(workload.module))
      return 0

    # Compiler imports are intentionally confined to the source-mode branch.
    # A --compiled-file process never imports pipeline_validator.compiler.
    from .compiler import compile_program
    from .compiler.api import dump_compiled_source

    loaded_programs = []
    for workload in workloads:
      source_name = args.ir_file if args.ir_file is not None else f"<builtin:{workload.name}>"
      try:
        print(f"[compile] {workload.name}", file=sys.stderr)
        program = compile_program(
          workload.module,
          hw,
          sim_cfg,
          binding_assumptions=bindings or None,
          source_name=source_name,
          workload_info=workload.info,
        )
        artifact_path = _persist_program(program, hw, dump_compiled_source(program), args.compiled_output)
        print(f"compiled artifact {program.artifact_hash} written to {artifact_path}", file=sys.stderr)
      except (OSError, UnicodeError, ParseError, VerifyException, TypeError, ValueError) as exc:
        print(f"failed to compile '{workload.name}': {exc}", file=sys.stderr)
        return 2
      if args.compile_only:
        continue
      try:
        print(f"[load] {artifact_path}", file=sys.stderr)
        loaded_programs.append(load_program(program, hw, sim_cfg, actual_bindings=bindings))
      except (TypeError, ValueError) as exc:
        print(f"failed to load compiled artifact '{artifact_path}': {exc}", file=sys.stderr)
        return 2

    if args.compile_only:
      return 0

  outputs = []
  overall_pass = True
  enable_tracer = bool(args.trace or args.memory_trace or args.trace_json or args.trace_html)
  for loaded in loaded_programs:
    info: WorkloadInfo = loaded.compiled.workload_info
    sim = Simulator(hw, sim_cfg, enable_tracer=enable_tracer)
    try:
      print(f"[run] {info.name}", file=sys.stderr)
      result = sim.run(loaded)
    except (RuntimeError, ValueError) as exc:
      print(f"execution failed for '{info.name}': {exc}", file=sys.stderr)
      return 1
    report = build_report(info, result, num_tiles=hw.num_tiles)
    outputs.append(report)
    if not all(check.get("pass", False) for check in report.checks):
      overall_pass = False
    if enable_tracer and result.tracer is not None:
      try:
        if args.trace_json:
          path = _multi_output_path(args.trace_json, info.name, len(loaded_programs))
          path.write_text(result.tracer.to_chrome_json(), encoding="utf-8")
          print(f"trace (perfetto json) written to {path}", file=sys.stderr)
        if args.trace_html:
          path = _multi_output_path(args.trace_html, info.name, len(loaded_programs))
          path.write_text(trace_to_html(result.tracer), encoding="utf-8")
          print(f"trace (html) written to {path}", file=sys.stderr)
      except OSError as exc:
        print(f"failed to write execution trace: {exc}", file=sys.stderr)
        return 1

  text = (
    "\n".join(report_to_text(report) for report in outputs)
    if not args.json
    else json.dumps([json.loads(report_to_json(report)) for report in outputs], indent=2)
  )
  try:
    if args.report:
      Path(args.report).write_text(text + "\n", encoding="utf-8")
      print(f"report written to {args.report}", file=sys.stderr)
    else:
      print(text)
  except OSError as exc:
    print(f"failed to write execution report: {exc}", file=sys.stderr)
    return 1

  return 0 if overall_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
