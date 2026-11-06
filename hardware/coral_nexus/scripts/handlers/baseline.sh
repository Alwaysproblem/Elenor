#!/usr/bin/env bash
# baseline handler (C005): runs the P0 workload set on the UNMODIFIED
# upstream core inside the locked container.
#
# Usage: baseline.sh --config <p0_coral.yaml> [--out DIR]
#   WORK_ROOT=<dir> overrides the materialized work root (default
#   out/coral_nexus/work/p0; materialized fresh from the source lock when
#   missing, with the p0 overlay applied).
#
# Refuses configs whose kind is not p0_upstream (exit 3) — CS1 targets are a
# different evidence domain and must never ride the P0 baseline.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"
LIB="$H/scripts/lib"

CONFIG=""
OUT="${OUT:-out/coral_nexus/C005/baseline}"
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "error: baseline: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$CONFIG" ] || { echo "error: baseline requires --config" >&2; exit 2; }

# Resolve config to an absolute path for the in-container mount while
# keeping a root-relative form for host-side YAML parsing.
if [ -f "$CONFIG" ]; then
  CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
else
  CONFIG_ABS="$ROOT/$CONFIG"
  CONFIG="$CONFIG"
fi

kind="$(python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, '$LIB')
from sources_tool import load_yaml_limited

try:
  print(load_yaml_limited(Path('$CONFIG_ABS')).get('kind', ''))
except Exception as exc:
  print('error: %s' % exc)
")"
case "$kind" in
  p0_upstream) ;;
  "") echo "error: config $CONFIG has no kind field" >&2; exit 3 ;;
  *) echo "error: config kind '$kind' is not p0_upstream; P0 baseline refuses non-P0 configs" >&2; exit 3 ;;
esac
WORK_ROOT="${WORK_ROOT:-$ROOT/out/coral_nexus/work/p0}"
if [ ! -d "$WORK_ROOT/coralnpu" ]; then
  mkdir -p "$(dirname "$WORK_ROOT")"
  bash "$LIB/materialize.sh" \
    --lock "$H/env/sources.lock" \
    --out "$WORK_ROOT" \
    --sources coralnpu >/dev/null
  echo "[baseline] materialized fresh work root: $WORK_ROOT"
fi

image="$(python3 -c "import json;print(json.load(open('$H/env/image.lock'))['tag'])")"
mkdir -p "$OUT"
out_abs="$(cd "$OUT" && pwd)"
work_abs="$(cd "$WORK_ROOT" && pwd)"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$work_abs:/work" \
  -v "$out_abs:/out" \
  -v "$CONFIG_ABS:/config.yaml:ro" \
  -v "$H/configs/p0_workloads.yaml:/registry.yaml:ro" \
  -v "$LIB:/cnlib:ro" \
  "$image" \
  bash /cnlib/baseline_inside.sh
rc=$?

python3 - "$OUT" "$rc" <<'PY'
import json
import sys
from pathlib import Path

out, rc = Path(sys.argv[1]), int(sys.argv[2])
registry = json.dumps(None)
report = {
  "schema": "cn_p0_baseline_report/v1",
  "config_kind": "p0_upstream",
  "container_exit_code": rc,
  "workloads": [],
}
for line in (out / "workload_list.txt").read_text().splitlines():
  name, elf_rel = line.split("\t")
  entry = {"name": name, "elf": elf_rel}
  for field in ("exit_code", "elf_sha256", "sim_sha256"):
    path = out / f"{name}.{field}"
    entry[field] = path.read_text().strip() if path.exists() else None
  stdout = out / f"{name}.stdout"
  entry["stdout_bytes"] = stdout.stat().st_size if stdout.exists() else 0
  entry["pass"] = entry["exit_code"] == "0"
  report["workloads"].append(entry)
report["pass"] = rc == 0 and all(w["pass"] for w in report["workloads"])
report["build_exit_code"] = (
  (out / "build/bazel_exit_code").read_text().strip()
  if (out / "build/bazel_exit_code").exists()
  else None
)
(out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY
[ "$rc" = 0 ]
