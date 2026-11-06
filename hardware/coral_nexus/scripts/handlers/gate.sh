#!/usr/bin/env bash
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"
STAGE=""
OUT="${OUT:-out/coral_nexus/C007/gate}"
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "error: gate: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$STAGE" ] || { echo "error: gate requires --stage" >&2; exit 2; }
mkdir -p "$OUT"
case "$STAGE" in
  G0)
    EVIDENCE_INDEX="$ROOT/out/coral_nexus/C007/evidence_index"
    mkdir -p "$EVIDENCE_INDEX"
    ln -sfn "$ROOT/out/coral_nexus/C001/attempt-001/sources" "$EVIDENCE_INDEX/sources_verify" 2>/dev/null || true
    ln -sfn "$ROOT/out/coral_nexus/C002/verify" "$EVIDENCE_INDEX/environment_verify" 2>/dev/null || true
    ln -sfn "$ROOT/out/coral_nexus/C004/export-001" "$EVIDENCE_INDEX/seed_export" 2>/dev/null || true
    ln -sfn "$ROOT/out/coral_nexus/C005/baseline" "$EVIDENCE_INDEX/baseline_run" 2>/dev/null || true
    exec python3 "$H/verification/gates.py" --evidence-index "$EVIDENCE_INDEX" --out "$OUT" ;;
  *)
    echo "error: gate stage '$STAGE' not implemented yet" >&2; exit 4 ;;
esac
