#!/usr/bin/env bash
# cn — unified entry for the Coral Nexus hardware workspace (host side).
#
# Dispatches only commands registered in scripts/commands.yaml. Arguments are
# passed through argv (spaces are safe); unknown commands or suites exit
# non-zero and are never reported as supported. Every invocation records a
# run manifest (command, args, executor, exit code, duration).
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"

exec python3 "$here/lib/cn_dispatch.py" --root "$ROOT" "$@"
