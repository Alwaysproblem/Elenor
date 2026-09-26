#!/bin/bash

set -e

# Run every runnable workload plus every protocol scenario registered in
# `examples/run.sh list` at two fidelities, writing each sweep into its own
# artifact folder (repo-root relative, gitignored):
#   full_memory -> examples/traces/full-memory-traces/
#   runtime     -> examples/traces/runtime-traces/
# Each scenario emits <name>.json (raw Perfetto/Chrome trace) and
# <name>.report.json (CLI JSON report; embeds compiled_artifact_hash).
# NEST subgraphs are excluded: 112 extra entries x 2 fidelities dominate the
# runtime; run them through `bash examples/run.sh nest-<name>` or the
# batch-III corpus runner when needed.
#
# Names are parsed from `run.sh list` itself (no hardcoded drift); override
# for a quick spot check with e.g.:
#   SCENARIOS='l2-shared-weight l2-private-weight' bash .vscode/run_all_workload.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_MEMORY_DIR="$ROOT_DIR/examples/traces/full-memory-traces"
RUNTIME_DIR="$ROOT_DIR/examples/traces/runtime-traces"

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

if [[ -n "${SCENARIOS:-}" ]]; then
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
    echo "=== [fidelity=$fidelity] $name ==="
    bash "$ROOT_DIR/examples/run.sh" "$name" \
      --memory-trace \
      --sim-override fidelity="$fidelity" \
      --trace-json "$out_dir/$name.json" \
      --report "$out_dir/$name.report.json"
  done
}

run_all full_memory "$FULL_MEMORY_DIR"
run_all runtime "$RUNTIME_DIR"
