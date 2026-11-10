#!/usr/bin/env bash
# numeric-probe handler (C011): run the directed BF16 vectors on the UNMODIFIED upstream
# VmeCoreMiniAxi Verilator model inside the locked container, then compare the recorded actual
# bits to the reference model ON THE HOST. The container only records; it never decides pass.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"
OUT="${OUT:-out/coral_nexus/C011/probe}"
WORK_ROOT="${WORK_ROOT:-$ROOT/out/coral_nexus/work/p0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --config) shift 2 ;;
    *) echo "error: numeric-probe: unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT"
if [ ! -f "$WORK_ROOT/coralnpu/tests/cocotb/vme_test/p0_numeric/p0_numeric_probe_bench.py" ]; then
  # A work root materialized before this overlay existed is stale: it needs a fresh one.
  fresh="$ROOT/out/coral_nexus/work/p0_numeric"
  rm -rf "$fresh"
  bash "$H/scripts/lib/materialize.sh" --lock "$H/env/sources.lock" --out "$fresh" --sources coralnpu >/dev/null \
    || { echo "error: materialize failed" >&2; exit 1; }
  WORK_ROOT="$fresh"
fi
image="$(python3 -c "import json;print(json.load(open('$H/env/image.lock'))['tag'])")"
work_abs="$(cd "$WORK_ROOT" && pwd)"; out_abs="$(cd "$OUT" && pwd)"
BFRUN="$ROOT/out/coral_nexus/C011/numrun"; mkdir -p "$BFRUN"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/w \
  -v "$work_abs:/work" -v "$out_abs:/out" -v "$BFRUN:/w" -v "$ROOT:/workspace:ro" \
  "$image" python3 /workspace/hardware/coral_nexus/scripts/lib/p0_numeric_capture.py /work /out
rc=$?
[ "$rc" -eq 0 ] || { echo "error: numeric probe capture failed (rc=$rc); logs in $OUT" >&2; exit 1; }
# Host-side verdict: compare actual bits to the reference model and to the hand-derived vectors.
python3 - "$H" "$OUT" <<'PY'
import json, sys
from pathlib import Path
H, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(H / "scripts/lib"))
import numeric_probe as probe
vectors = json.loads((H / "verification/numerics/known_vectors.json").read_text())
record = json.loads((out / "numeric_probe_record.json").read_text())
groups = probe.build_groups(vectors)
errors = probe.compare(record, groups) + probe.vector_results(record, groups, vectors)
tamper = probe.tamper_selftest(record, groups)
verdict = {"schema": "cn_numeric_probe_verdict/v1", "errors": errors, "tamper_self_test": tamper,
           "groups": len(groups), "cells_compared": 256 * len(groups),
           "placed_vectors": sum(len(g["placed"]) for g in groups),
           "pass": not errors and tamper["detected"] and record.get("replay_identical") is True}
(out / "numeric_probe_verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
print(json.dumps(verdict, indent=2, sort_keys=True))
sys.exit(0 if verdict["pass"] else 1)
PY
