#!/usr/bin/env bash
# unit --suite <name> handler: runs the registered pytest files of one suite
# and writes a machine-readable summary next to the pytest log.
#
# Suites are registered in scripts/commands.yaml (C003 registers only the
# environment suite). The dispatcher already validated the suite name; an
# unregistered suite never reaches this handler.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"

suite=""
out="${OUT:-out/coral_nexus/unit}"
while [ $# -gt 0 ]; do
  case "$1" in
    --suite) suite="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --config) shift 2 ;; # accepted for forward compatibility, unused here
    *) echo "error: unit: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$suite" ] || { echo "error: unit requires --suite" >&2; exit 2; }

mkdir -p "$out"
tests="$(python3 - "$ROOT" "$suite" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__ if "__file__" in dir() else ".")))
root = Path(sys.argv[1])
suite = sys.argv[2]
sys.path.insert(0, str(root / "hardware/coral_nexus/scripts/lib"))
from sources_tool import load_yaml_limited  # noqa: E402

registry = load_yaml_limited(root / "hardware/coral_nexus/scripts/commands.yaml")
entry = registry["commands"]["unit"]["suites"][suite]
print("\n".join(entry["tests"]))
PY
)"
[ -n "$tests" ] || { echo "error: suite $suite has no tests registered" >&2; exit 3; }

conda run -n elenor-validator python -m pytest -n0 -q $tests \
  > "$out/pytest.log" 2>&1
rc=$?
python3 - "$out" "$rc" "$suite" <<'PY'
import json
import re
import sys
from pathlib import Path

out, rc, suite = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
log = (out / "pytest.log").read_text()
match = re.search(r"(\d+) passed", log)
report = {
  "schema": "cn_unit_summary/v1",
  "suite": suite,
  "exit_code": rc,
  "passed": int(match.group(1)) if match else 0,
  "failed": len(re.findall(r"^FAILED", log, re.M)),
}
(out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
PY
tail -3 "$out/pytest.log"
exit "$rc"
