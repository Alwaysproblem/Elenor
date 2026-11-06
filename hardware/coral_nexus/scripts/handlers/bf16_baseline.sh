#!/usr/bin/env bash
# BF16 baseline handler (C006).
#
# Runs the P0 BF16 KAT capture on the unmodified upstream VmeCoreMiniAxi
# Verilator model: the benchmark file is an overlay ADD (upstream untouched),
# executed TWICE with different cocotb seeds. Report carries input/actual
# bits, accumulator preload, per-case cycle_count, ELF hash and the
# bit-identical replay proof. Tamper/missing-file negatives live in the
# unit suites, not here.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"
OUT="${OUT:-out/coral_nexus/C006/baseline}"
WORK_ROOT="${WORK_ROOT:-$ROOT/out/coral_nexus/work/p0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    *) echo "error: bf16-baseline: unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT"

# Re-materialize when the KAT overlay is missing from the work root (fresh
# checkouts, or a previous materialize predating the overlay entries).
if [ ! -f "$WORK_ROOT/coralnpu/tests/cocotb/vme_test/p0_kat/p0_bf16_kat_bench.py" ]; then
  bash "$H/scripts/lib/materialize.sh" --lock "$H/env/sources.lock" \
    --out "$WORK_ROOT" --sources coralnpu >/dev/null
fi

image="$(python3 -c "import json;print(json.load(open('$H/env/image.lock'))['tag'])")"
work_abs="$(cd "$WORK_ROOT" && pwd)"
out_abs="$(cd "$OUT" && pwd)"
repo_root="$(cd "$ROOT" && pwd)"

BFRUN="$ROOT/out/coral_nexus/C006/bfrun"
mkdir -p "$BFRUN"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/w \
  -v "$work_abs:/work" -v "$out_abs:/out" -v "$BFRUN:/w" \
  -v "$repo_root:/workspace:ro" \
  "$image" \
  python3 /workspace/hardware/coral_nexus/scripts/lib/p0_bf16_capture.py \
  /work /out
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "error: BF16 KAT capture failed (rc=$rc); log: $OUT/kat_run_b.log" >&2
  exit 1
fi
echo "[bf16-baseline] report: $OUT/bf16_kat_report.json"
