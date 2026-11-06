#!/usr/bin/env bash
# Container lifecycle handler for the Coral Nexus locked build environment.
#
# Usage:
#   env.sh build  [--out DIR]   Build the locked image (caller UID/GID).
#   env.sh verify [--out DIR]   Run the in-container verification once.
#
# Registered subcommands only; anything else exits 2 with usage. The script
# never mounts the Docker socket, never uses --privileged, and runs jobs as
# the caller's UID/GID.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
IMAGE_LOCK="$H/env/image.lock"

image_tag() {
  python3 -c "import json;print(json.load(open('$IMAGE_LOCK'))['tag'])"
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker CLI not available" >&2
    exit 3
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "error: docker daemon not reachable" >&2
    exit 3
  fi
}

do_build() {
  require_docker
  local out="${OUT:-out/coral_nexus/C002/build}"
  mkdir -p "$out"
  local tag base_digest
  tag="$(image_tag)"
  base_digest="$(python3 -c "import json;print(json.load(open('$IMAGE_LOCK'))['base']['digest'])")"
  echo "[env] building $tag (base $base_digest)"
  docker build \
    --build-arg _UID="$(id -u)" \
    --build-arg _GID="$(id -g)" \
    -f "$H/env/Dockerfile" \
    -t "$tag" \
    "$H/env" 2>&1 | tee "$out/build.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "error: docker build failed (rc=$rc); log: $out/build.log" >&2
    exit 1
  fi
  local image_id
  image_id="$(docker image inspect "$tag" --format '{{.Id}}')"
  python3 - "$out" "$tag" "$image_id" "$base_digest" <<'PY'
import json
import sys
from datetime import datetime, timezone

out, tag, image_id, base_digest = sys.argv[1:5]
report = {
  "schema": "cn_environment_build/v1",
  "tag": tag,
  "image_id": image_id,
  "locked_base_digest": base_digest,
  "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
with open(f"{out}/build_manifest.json", "w") as handle:
  json.dump(report, handle, indent=2, sort_keys=True)
  handle.write("\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY
}

do_verify() {
  require_docker
  local out="${OUT:-out/coral_nexus/C002/verify}"
  mkdir -p "$out"
  local tag
  tag="$(image_tag)"
  if ! docker image inspect "$tag" >/dev/null 2>&1; then
    echo "error: image $tag not built; run 'env.sh build' first" >&2
    exit 3
  fi
  local repo_root cache_root out_abs
  repo_root="$(cd "$H/../.." && pwd)"
  cache_root="$repo_root/out/coral_nexus/cache"
  mkdir -p "$cache_root/toolchain" "$out"
  out_abs="$(cd "$out" && pwd)"
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    --env REPORT_DIR=/report \
    --env TOOLCHAIN_DIR=/toolchain \
    --env H_ROOT=/workspace/hardware/coral_nexus \
    -v "$out_abs:/report" \
    -v "$cache_root/toolchain:/toolchain" \
    -v "$repo_root:/workspace:ro" \
    "$tag" \
    bash /workspace/hardware/coral_nexus/env/verify_inside.sh
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "error: in-container verification failed (rc=$rc); report: $out" >&2
    exit 1
  fi
  echo "[env] verification report: $out/environment_checks.json"
}

do_export() {
  local out="${OUT:?--out is required for export}"
  python3 "$H/scripts/lib/seed_export.py" \
    --seeds "$H/env/seed_targets.json" \
    --cache "$H/../../out/coral_nexus/cache/seeds" \
    --out "$out"
}


usage() {
  sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
  exit 2
}

[ $# -ge 1 ] || usage
sub="$1"
shift
OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    *) echo "error: env $sub: unknown argument $1" >&2; exit 2 ;;
  esac
done
case "$sub" in
  build) do_build "$@" ;;
  verify) do_verify "$@" ;;
  export) do_export "$@" ;;
  *) usage ;;
esac
