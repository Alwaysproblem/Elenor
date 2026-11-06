#!/usr/bin/env bash
# BF16 baseline handler (C006): runs the VME matmul test on the unmodified
# upstream core. Uses the vme_test_verilator cocotb target which exercises
# vtfmm.alt (BF16 x BF16 -> FP32 tile accumulate) with known inputs.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"
OUT="${OUT:-out/coral_nexus/C006/baseline}"
WORK_ROOT="${WORK_ROOT:-$ROOT/out/coral_nexus/work/p0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    *) echo "error: bf16-baseline: unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT"
if [ ! -d "$WORK_ROOT/coralnpu" ]; then
  bash "$H/scripts/lib/materialize.sh" --lock "$H/env/sources.lock" \
    --out "$WORK_ROOT" --sources coralnpu >/dev/null
fi
image="$(python3 -c "import json;print(json.load(open('$H/env/image.lock'))['tag'])")"
work_abs="$(cd "$WORK_ROOT" && pwd)"
out_abs="$(cd "$OUT" && pwd)"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$work_abs:/work" -v "$out_abs:/out" \
  "$image" bash -c '
cd /work/coralnpu
bazel --output_user_root=/work/bazel-root test -c opt \
  --disk_cache=/work/disk-cache --repository_cache=/work/repo-cache \
  --test_output=all \
  //tests/cocotb/vme_test:vme_test_verilator > /out/vme_test.log 2>&1
echo $? > /out/vme_test_exit_code
'
rc=$(cat "$OUT/vme_test_exit_code" 2>/dev/null || echo 1)
python3 - "$OUT" "$rc" <<'PY'
import json, sys
from pathlib import Path
out, rc = Path(sys.argv[1]), int(sys.argv[2])
log = (out / "vme_test.log").read_text() if (out / "vme_test.log").exists() else ""
passed = "PASSED" in log and rc == 0
report = {
  "schema": "cn_p0_bf16_baseline/v1",
  "test_target": "//tests/cocotb/vme_test:vme_test_verilator",
  "test_result": "PASSED" if passed else "FAILED",
  "exit_code": rc,
  "bf16_cases": ["vtfmm_bf16_mt0", "vtfmm_bf16_mt8"],
  "evidence_tier": "p0_upstream_rtl",
  "notes": "VME core executes BF16 matmul with known inputs (vme_matmul_test.cc:269). KAT capture, not independent oracle (C047).",
}
(out / "bf16_baseline_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
if not passed: sys.exit(1)
PY
