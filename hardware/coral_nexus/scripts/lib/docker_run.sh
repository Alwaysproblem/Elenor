#!/usr/bin/env bash
# Shared docker run wrapper for container-executor commands.
#
# Invariants (tested by test_dispatch.py):
#   * runs as the caller's UID/GID, never root and never --privileged
#   * never mounts /var/run/docker.sock
#   * mounts the repository read-only at /workspace and exactly one writable
#     output directory at /out
# Usage: docker_run.sh <image> <outdir> -- <command...>
set -uo pipefail

if [ $# -lt 3 ]; then
  echo "usage: docker_run.sh <image> <outdir> -- <command...>" >&2
  exit 2
fi

image="$1"
outdir="$2"
shift 2
[ "${1:-}" = "--" ] || { echo "usage: docker_run.sh <image> <outdir> -- <command...>" >&2; exit 2; }
shift

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "$outdir"

exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  -v "$repo_root:/workspace:ro" \
  -v "$(cd "$outdir" && pwd):/out" \
  -w /workspace \
  "$image" \
  "$@"
