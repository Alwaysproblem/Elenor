#!/bin/bash

set -e

# Run every runnable workload plus every protocol scenario registered in
# `examples/run.sh list`, writing each sweep into its own artifact folder
# (repo-root relative, gitignored):
#   runtime     -> examples/traces/runtime-traces/   （默认只跑 runtime）
#   full_memory -> examples/traces/full-memory-traces/（--full-memory 时追加）
# Each scenario emits <name>.json (raw Perfetto/Chrome trace) and
# <name>.report.json (CLI JSON report; embeds compiled_artifact_hash).
# NEST subgraphs are excluded: 112 extra entries x 2 fidelities dominate the
# runtime; run them through `bash examples/run.sh nest-<name>` or the
# batch-III corpus runner when needed.
#
# Names are parsed from `run.sh list` itself (no hardcoded drift); override
# for a quick spot check with e.g.:
#   SCENARIOS='l2-shared-weight l2-private-weight' bash examples/run_all_workload.sh
#   bash examples/run_all_workload.sh --scenarios 'l2-shared-weight,l2-private-weight'
# （--scenarios 给出时优先于 SCENARIOS 环境变量；空格或逗号分隔均可。）
#
# Usage:
#   bash examples/run_all_workload.sh                  # 仅 runtime
#   bash examples/run_all_workload.sh --full-memory    # runtime + full_memory
#   bash examples/run_all_workload.sh --scenarios 'a b c' [--full-memory]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_MEMORY_DIR="$ROOT_DIR/examples/traces/full-memory-traces"
RUNTIME_DIR="$ROOT_DIR/examples/traces/runtime-traces"

# 默认只跑 runtime；--full-memory 追加 full_memory 一轮。
# --scenarios 覆盖 SCENARIOS 环境变量（空格或逗号分隔）。
RUN_FULL_MEMORY=0
SCENARIOS_ARG=""
SCENARIOS_SET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -fm|--full-memory) RUN_FULL_MEMORY=1 ;;
    -s|--scenarios)
      [[ $# -ge 2 ]] || { echo "error: --scenarios requires a value" >&2; exit 2; }
      [[ -n "$2" ]] || { echo "error: --scenarios value is empty" >&2; exit 2; }
      case "$2" in
        -*) echo "error: --scenarios value '$2' looks like an option" >&2; exit 2 ;;
      esac
      SCENARIOS_ARG="$2"
      SCENARIOS_SET=1
      shift
      ;;
    -h|--help)
      grep -E '^# (Usage:| *bash)' "$0" | sed 's/^# \{0,2\}//'
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1' (expected --full-memory / --scenarios)" >&2
      exit 2
      ;;
  esac
  shift
done

parse_list() {
  # Emit the runnable workload and protocol scenario names from `run.sh list`.
  # A name line's first token is a bare kebab-case id whose remainder is empty
  # or a path/description; wrapped description lines never match that shape.
  bash "$ROOT_DIR/examples/run.sh" list | awk '
    /^Runnable workloads:/ {section="run"; next}
    /^Protocol scenarios:/ {section="run"; next}
    /^NEST subgraphs/      {section="nest"; next}
    /^[[:space:]]*$/       {next}
    section == "run" {
      line = $0
      sub(/^[[:space:]]+/, "", line)
      first = line
      sub(/[[:space:]].*/, "", first)
      rest = line
      sub(/^[^[:space:]]*[[:space:]]*/, "", rest)
      if (first !~ /^[a-z0-9][a-z0-9-]*$/) next
      if (rest != "" && rest !~ /^(workloads\/|scenarios\/|\()/) next
      print first
    }'
}

if [[ "$SCENARIOS_SET" -eq 1 ]]; then
  IFS=' ,' read -r -a SCENARIO_LIST <<< "$SCENARIOS_ARG"
  # 过滤空 token（尾随逗号、纯逗号等）。
  FILTERED=()
  for s in "${SCENARIO_LIST[@]}"; do
    [[ -n "$s" ]] && FILTERED+=("$s")
  done
  if [[ ${#FILTERED[@]} -eq 0 ]]; then
    echo "error: --scenarios value contains no scenario names" >&2
    exit 2
  fi
  SCENARIO_LIST=("${FILTERED[@]}")
elif [[ -n "${SCENARIOS:-}" ]]; then
  read -r -a SCENARIO_LIST <<< "$SCENARIOS"
else
  mapfile -t SCENARIO_LIST < <(parse_list)
fi
if [[ ${#SCENARIO_LIST[@]} -eq 0 ]]; then
  echo "error: parsed no scenarios from run.sh list" >&2
  exit 2
fi

run_all() {
  local fidelity="$1"
  local out_dir="$2"
  local name
  mkdir -p "$out_dir"
  for name in "${SCENARIO_LIST[@]}"; do
    # A scenario that reads a byte oracle pins full_memory; skip it in a
    # sweep that cannot provide one instead of failing the whole run.
    required="$(bash "$ROOT_DIR/examples/run.sh" fidelity-of "$name" || true)"
    if [[ -n "$required" && "$required" != "$fidelity" ]]; then
      echo "--- SKIP [fidelity=$fidelity] $name (requires $required)"
      continue
    fi
    echo "=== [fidelity=$fidelity] $name ==="
    bash "$ROOT_DIR/examples/run.sh" "$name" \
      --memory-trace \
      --sim-override fidelity="$fidelity" \
      --trace-json "$out_dir/$name.json" \
      --report "$out_dir/$name.log"
  done
}

run_all runtime "$RUNTIME_DIR"
if [[ "$RUN_FULL_MEMORY" -eq 1 ]]; then
  run_all full_memory "$FULL_MEMORY_DIR"
fi
