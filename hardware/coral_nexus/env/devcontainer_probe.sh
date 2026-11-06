#!/usr/bin/env bash
# Launch the devcontainer for real and run the C004 probes inside it.
#
# Usage: devcontainer_probe.sh --repo ROOT --config FILE --image TAG --out DIR
#
# The static JSON check is not proof, so this launches the container through
# the Dev Containers CLI, then probes identity, tool versions, the workspace
# mount and a real compile. When the CLI is unavailable it records a blocker
# and exits 3 — never a silent pass.
set -uo pipefail

repo=""
cfg=""
tag=""
out=""
work=""
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --config) cfg="$2"; shift 2 ;;
    --image) tag="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --work) work="$2"; shift 2 ;;
    *) echo "error: devcontainer-probe: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$repo" ] && [ -n "$cfg" ] && [ -n "$tag" ] && [ -n "$out" ] || {
  echo "error: --repo, --config, --image and --out are required" >&2
  exit 2
}
mkdir -p "$out"

write_report() {  # write_report <status> <pass:true|false> <extra-json>
  python3 - "$out" "$tag" "$1" "$2" "$3" <<'PY'
import json
import sys
from datetime import datetime, timezone

out, tag, status, passed, extra = sys.argv[1:6]
report = {
  "schema": "cn_devcontainer_probe/v1",
  "status": status,
  "image": tag,
  "devcontainer_cli": "npx -y @devcontainers/cli@latest",
  "checks": json.loads(extra) if extra else {},
  "pass": passed == "true",
  "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
with open(f"{out}/devcontainer_report.json", "w") as fh:
  json.dump(report, fh, indent=2, sort_keys=True)
  fh.write("\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY
}

if ! command -v npx >/dev/null 2>&1; then
  write_report blocked false '{"blocker": "npx (Node.js) unavailable"}'
  echo "error: devcontainer CLI unavailable; blocker recorded (not a pass)" >&2
  exit 3
fi

dc() { npx -y @devcontainers/cli@latest "$@"; }

# Always build from the current config: a reused container can carry mounts
# that the JSON no longer declares (a stale root-owned volume broke a probe).
dc up --workspace-folder "$repo" --config "$cfg" --remove-existing-container \
  >"$out/up.log" 2>&1
up_rc=$?

probes="$out/probes.log"
: > "$probes"
probe_rc=0
if [ "$up_rc" -eq 0 ]; then
  for cmd in \
    "id -un; id -u" \
    "bazel --version" \
    "clang --version | head -1" \
    "findmnt -rno TARGET,SOURCE,FSTYPE /workspace"; do
    echo "--- $cmd" >>"$probes"
    dc exec --workspace-folder "$repo" --config "$cfg" -- bash -lc "$cmd" >>"$probes" 2>&1 || probe_rc=1
  done
  # Compile inside the derived work root (upstream stays pristine; //nexus_p0
  # is an overlay target that only exists there). Symlinks and the bazel
  # output root live outside the workspace so nothing is written into it.
  echo "--- bazel build //nexus_p0:p0_scalar_add (in derived work root)" >>"$probes"
  dc exec --workspace-folder "$repo" --config "$cfg" -- \
    bash -lc "set -e; test -d $work || { echo 'missing work root: $work'; exit 4; }; cd $work/coralnpu && bazel --output_user_root=/tmp/bazel-probe-root build -c opt --symlink_prefix=/tmp/bazel-probe- //nexus_p0:p0_scalar_add" \
    >>"$probes" 2>&1 || probe_rc=1
else
  probe_rc=1
fi

checks=$(python3 - "$probes" "$up_rc" "$probe_rc" <<'PY'
import json
import sys

log = open(sys.argv[1]).read()
up_rc, probe_rc = int(sys.argv[2]), int(sys.argv[3])
checks = {
  "up": up_rc == 0,
  "non_root_user": "uid=0(" not in log and "builder" in log,
  "workspace_mount": "/workspace" in log,
  "bazel_version": "bazel " in log,
  "compiler_version": "clang version" in log,
  "compile_probe": "Build completed successfully" in log,
}
print(json.dumps(checks))
sys.exit(0 if all(checks.values()) and probe_rc == 0 else 1)
PY
)
checks_rc=$?

if [ "$checks_rc" -eq 0 ]; then
  write_report passed true "$checks"
  exit 0
fi
write_report failed false "$checks"
exit 1
