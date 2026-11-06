#!/usr/bin/env bash
# Container lifecycle handler for the Coral Nexus locked build environment.
#
# Usage:
#   env.sh build  [--out DIR]   Build the locked image (caller UID/GID).
#   env.sh verify [--out DIR]   Run the in-container verification once.
#   env.sh export [--out DIR]   Export the offline seed archive.
#   env.sh offline-rebuild --export DIR --out DIR [--work DIR]
#                               Rebuild P0 targets with --network none and
#                               empty build caches, using only the exported
#                               vendor archive.
#   env.sh devcontainer-probe --out DIR
#                               Launch the devcontainer for real and run the
#                               identity/version/mount/compile probes inside
#                               it; records a blocker (exit 3) when the CLI is
#                               unavailable instead of passing on static JSON.
#   env.sh offline-rebuild --export DIR --out DIR [--work DIR]
#                               Rebuild P0 targets with --network none and
#                               empty caches, using only the exported vendor dir.
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


do_offline_rebuild() {
  require_docker
  local export_dir="${EXPORT:?--export is required}"
  local out="${OUT:?--out is required}"
  local work="${WORK:-$out/work}"
  local tag
  tag="$(image_tag)"
  if ! docker image inspect "$tag" >/dev/null 2>&1; then
    echo "error: image $tag not built; run 'env.sh build' first" >&2
    exit 3
  fi
  if [ ! -d "$export_dir/bazel_vendor_cache" ]; then
    echo "error: $export_dir/bazel_vendor_cache missing; run 'env.sh export'" >&2
    exit 1
  fi
  mkdir -p "$out" "$work"
  local repo_root out_abs export_abs work_abs vendor_sha
  repo_root="$(cd "$H/../.." && pwd)"
  out_abs="$(cd "$out" && pwd)"
  export_abs="$(cd "$export_dir" && pwd)"
  work_abs="$(cd "$work" && pwd)"
  vendor_sha="$(python3 -c "
import json
seeds = json.load(open('$H/env/seed_targets.json'))['seeds']
print(next(s['sha256'] for s in seeds if s['name'] == 'bazel_vendor_cache'))
")"
  echo "[env] offline rebuild: network=none, empty caches, vendor from $export_abs"
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    --network none \
    --env WORK_DIR=/work \
    --env VENDOR_DIR=/vendor/bazel_vendor_cache \
    --env VENDOR_SHA256="$vendor_sha" \
    --env REPORT_DIR=/report \
    -v "$work_abs:/work" \
    -v "$export_abs:/vendor:ro" \
    -v "$out_abs:/report" \
    -v "$repo_root:/workspace:ro" \
    "$tag" \
    bash /workspace/hardware/coral_nexus/env/offline_rebuild_inside.sh
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "error: offline rebuild failed (rc=$rc); log: $out/build.log" >&2
    exit 1
  fi
  echo "[env] offline rebuild report: $out/offline_rebuild.json"
}

do_devcontainer_probe() {
  local out="${OUT:?--out is required}"
  local repo_root
  repo_root="$(cd "$H/../.." && pwd)"
  local work="${WORK:-$repo_root/out/coral_nexus/C004/offline/work}"
  bash "$H/env/devcontainer_probe.sh" \
    --repo "$repo_root" \
    --config "$repo_root/.devcontainer/coral_nexus/devcontainer.json" \
    --image "$(image_tag)" \
    --work "/workspace/out/coral_nexus/C004/offline/work" \
    --out "$out"
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
EXPORT=""
WORK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --export) EXPORT="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    *) echo "error: env $sub: unknown argument $1" >&2; exit 2 ;;
  esac
done
case "$sub" in
  build) do_build "$@" ;;
  verify) do_verify "$@" ;;
  export) do_export "$@" ;;
  offline-rebuild) do_offline_rebuild "$@" ;;
  devcontainer-probe) do_devcontainer_probe "$@" ;;
  *) usage ;;
esac
