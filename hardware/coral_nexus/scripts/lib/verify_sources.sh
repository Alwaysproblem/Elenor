#!/usr/bin/env bash
# Verify every source-lock invariant and write a JSON report.
#
# Usage:
#   verify_sources.sh --lock <sources.lock> --out <report-dir>
#
# Checks: gitlink commit recorded in the index, pristine submodule work
# trees, archive content digests, embedded pins (record-only), overlay
# declarations, and patch ownership (single owner per touched file).
# Exits non-zero when any check fails; the report names each failure.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "$here/sources_tool.py" verify "$@"
