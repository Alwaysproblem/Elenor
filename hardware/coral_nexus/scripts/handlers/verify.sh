#!/usr/bin/env bash
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"
MANIFEST=""
OUT="${OUT:-out/coral_nexus/C007/verify}"
while [ $# -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "error: verify: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$MANIFEST" ] || { echo "error: verify requires --manifest" >&2; exit 2; }
if [ -f "$ROOT/$MANIFEST" ]; then MANIFEST_ABS="$ROOT/$MANIFEST"; else MANIFEST_ABS="$MANIFEST"; fi
mkdir -p "$OUT"
python3 - "$MANIFEST_ABS" "$OUT" <<'PY'
import json, sys
from pathlib import Path
manifest_path, out = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
report = {"schema": "cn_verify_report/v1", "manifest": str(manifest_path), "verified": True, "checks": []}
for entry in manifest.get("evidence", []):
  p = Path(entry["path"])
  ok = p.exists()
  report["checks"].append({"evidence": entry.get("name", str(p)), "pass": ok})
  if not ok: report["verified"] = False
(out / "verify_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
if not report["verified"]: sys.exit(1)
PY
