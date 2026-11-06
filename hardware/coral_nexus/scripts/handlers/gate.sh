#!/usr/bin/env bash
# gate handler (C007): runs a registered stage gate.
# Usage: gate.sh --stage <G0|G1|...> [--out DIR]
#
# For G0: builds the evidence manifest (real sha256 of report files +
# required artifact list), then runs the gate which RE-HASHES and
# compares. A directory containing only {"pass":true} with no real
# artifacts cannot pass.
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
    # Symlink real evidence directories into the index.
    ln -sfn "$ROOT/out/coral_nexus/C001/attempt-001/sources" "$EVIDENCE_INDEX/sources_verify" 2>/dev/null || true
    ln -sfn "$ROOT/out/coral_nexus/C002/verify" "$EVIDENCE_INDEX/environment_verify" 2>/dev/null || true
    ln -sfn "$ROOT/out/coral_nexus/C004/export-001" "$EVIDENCE_INDEX/seed_export" 2>/dev/null || true
    ln -sfn "$ROOT/out/coral_nexus/C005/baseline" "$EVIDENCE_INDEX/baseline_run" 2>/dev/null || true
    ln -sfn "$ROOT/out/coral_nexus/C006/attempt-001" "$EVIDENCE_INDEX/bf16_baseline" 2>/dev/null || true
    # Build the evidence manifest with real hashes + artifact lists.
    python3 "$H/verification/gates.py" --build-manifest --evidence-index "$EVIDENCE_INDEX" 2>/dev/null || \
      python3 -c "
import sys; sys.path.insert(0, '$H/verification')
from gates import build_evidence_manifest
from pathlib import Path
m = build_evidence_manifest(Path('$EVIDENCE_INDEX'))
print('manifest built:', len(m['entries']), 'entries')
"
    # Run the gate: re-hash + compare + check artifacts.
    exec python3 "$H/verification/gates.py" --evidence-index "$EVIDENCE_INDEX" --out "$OUT"
    ;;
  *)
    echo "error: gate stage '$STAGE' not implemented yet" >&2; exit 4
    ;;
esac
