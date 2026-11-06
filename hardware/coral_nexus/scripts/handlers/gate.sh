#!/usr/bin/env bash
# gate handler (C007): runs a registered stage gate.
# Usage: gate.sh --stage <G0|G1|...> [--out DIR] [--evidence-index DIR] [--pin FILE]
#
# For G0: verifies every required evidence report and artifact against a
# COMMITTED pin (verification/expected_evidence.json). Re-hashing against a
# manifest rebuilt from the same directory proves nothing, so the pin is the
# root of trust: a directory containing only {"pass": true} cannot pass.
# A manifest is still written for the record (what was evaluated), but it is
# never the pass criterion.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"

STAGE=""
OUT="${OUT:-out/coral_nexus/C007/gate}"
EVIDENCE_INDEX=""
PIN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --evidence-index) EVIDENCE_INDEX="$2"; shift 2 ;;
    --pin) PIN="$2"; shift 2 ;;
    *) echo "error: gate: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$STAGE" ] || { echo "error: gate requires --stage" >&2; exit 2; }
mkdir -p "$OUT"

case "$STAGE" in
  G0)
    if [ -z "$EVIDENCE_INDEX" ]; then
      EVIDENCE_INDEX="$ROOT/out/coral_nexus/C007/evidence_index"
      mkdir -p "$EVIDENCE_INDEX"
      # Symlink real evidence directories into the index.
      ln -sfn "$ROOT/out/coral_nexus/C001/attempt-001/sources" "$EVIDENCE_INDEX/sources_verify" 2>/dev/null || true
      ln -sfn "$ROOT/out/coral_nexus/C002/verify" "$EVIDENCE_INDEX/environment_verify" 2>/dev/null || true
      ln -sfn "$ROOT/out/coral_nexus/C004/export-005" "$EVIDENCE_INDEX/seed_export" 2>/dev/null || true
      ln -sfn "$ROOT/out/coral_nexus/C005/baseline" "$EVIDENCE_INDEX/baseline_run" 2>/dev/null || true
      ln -sfn "$ROOT/out/coral_nexus/C006/attempt-001" "$EVIDENCE_INDEX/bf16_baseline" 2>/dev/null || true
      # C004 closeout: offline rebuild + devcontainer launch are part of G0.
      ln -sfn "$ROOT/out/coral_nexus/C004/offline/rebuild4" "$EVIDENCE_INDEX/offline_rebuild" 2>/dev/null || true
      ln -sfn "$ROOT/out/coral_nexus/C004/devcontainer" "$EVIDENCE_INDEX/devcontainer_probe" 2>/dev/null || true
    fi
    [ -n "$PIN" ] || PIN="$H/verification/expected_evidence.json"
    # Record what was evaluated (never the pass criterion).
    python3 "$H/verification/gates.py" --build-manifest --evidence-index "$EVIDENCE_INDEX" >/dev/null 2>&1 || true
    # Verify against the committed pin; fail closed if it is absent.
    exec python3 "$H/verification/gates.py" \
      --evidence-index "$EVIDENCE_INDEX" --pin "$PIN" --out "$OUT"
    ;;
  *)
    echo "error: gate stage '$STAGE' not implemented yet" >&2; exit 4
    ;;
esac
