#!/usr/bin/env bash
# Runs INSIDE the locked container on the materialized P0 work root.
# Builds the configured bazel targets and executes every registered P0
# workload through the unmodified core simulator, capturing raw evidence.
#
# Mounts provided by handlers/baseline.sh:
#   /work      materialized work root (rw)
#   /out       evidence directory (rw)
#   /config.yaml   p0_coral.yaml
#   /registry.yaml p0_workloads.yaml
#   /cnlib         hardware/coral_nexus/scripts/lib (load_yaml_limited)
set -euo pipefail

OUT=/out
CFG=/config.yaml
REGISTRY=/registry.yaml

cd /work/coralnpu
bazel_root="$(python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, '/cnlib')
from sources_tool import load_yaml_limited
print(load_yaml_limited(Path('${CFG}'))['bazel']['output_user_root'])
")"
cycles_cap="$(python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, '/cnlib')
from sources_tool import load_yaml_limited
print(load_yaml_limited(Path('${CFG}'))['sim']['cycles_cap'])
")"

targets="$(python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, '/cnlib')
from sources_tool import load_yaml_limited
print('\n'.join(load_yaml_limited(Path('${CFG}'))['bazel']['targets']))
")"

mkdir -p "$OUT/build"
bazel --output_user_root="${bazel_root}" build -c opt \
  --disk_cache=/work/disk-cache --repository_cache=/work/repo-cache \
  $targets 2>&1 | tee "$OUT/build/bazel.log"
build_rc=${PIPESTATUS[0]}
echo "$build_rc" > "$OUT/build/bazel_exit_code"
[ "$build_rc" = 0 ]

exec_root="$(bazel --output_user_root="${bazel_root}" info output_base)/execroot/_main"
sim_bin="$(ls "$exec_root"/bazel-out/k8-opt/bin/tests/verilator_sim/core_mini_axi_sim | head -1)"

run_workload() {  # run_workload <name> <elf-relpath>
  local name="$1" elf_rel="$2"
  local elf
  elf="$(ls "$exec_root"/bazel-out/k8-opt-ST-*/bin/"$elf_rel" 2>/dev/null | head -1)"
  if [ -z "$elf" ] || [ ! -f "$elf" ]; then
    echo "127" > "$OUT/$name.exit_code"
    echo "elf not found under bazel-out: $elf_rel" > "$OUT/$name.stderr"
    return 1
  fi
  sha256sum "$elf" | awk '{print $1}' > "$OUT/$name.elf_sha256"
  sha256sum "$sim_bin" | awk '{print $1}' > "$OUT/$name.sim_sha256"
  set +e
  timeout 300 "$sim_bin" --binary "$elf" --cycles "$cycles_cap" \
    > "$OUT/$name.stdout" 2> "$OUT/$name.stderr"
  local rc=$?
  set -e
  echo "$rc" > "$OUT/$name.exit_code"
  [ "$rc" = 0 ]
}

python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, '/cnlib')
from sources_tool import load_yaml_limited
doc = load_yaml_limited(Path('${REGISTRY}'))
for workload in doc['workloads']:
  print(workload['name'] + '\t' + workload['elf'])
" > "$OUT/workload_list.txt"

overall=0
while IFS=$'\t' read -r name elf_rel; do
  [ -n "$name" ] || continue
  if run_workload "$name" "$elf_rel"; then
    echo "PASS $name" | tee -a "$OUT/summary.txt"
  else
    echo "FAIL $name" | tee -a "$OUT/summary.txt"
    overall=1
  fi
done < "$OUT/workload_list.txt"
exit "$overall"
