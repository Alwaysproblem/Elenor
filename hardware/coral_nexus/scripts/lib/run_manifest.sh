#!/usr/bin/env bash
# CLI wrapper around run_manifest.py for bash handlers that need to record
# an invocation envelope. Usage:
#   run_manifest.sh <fields-json-file> <output-path>
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$here/run_manifest.py" "$@"
