#!/usr/bin/env bash
# Materialize the isolated derived work root from the pinned sources.
#
# Usage:
#   materialize.sh --lock <sources.lock> --out <work-root> [--sources a,b]
#
# The work root is created fresh (refuses non-empty directories), sources are
# extracted from `git archive` at the locked commits, overlay entries are
# copied, and the patch series is applied strictly. The pristine submodules
# under hardware/coral_nexus/upstream are never modified.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "$here/sources_tool.py" materialize "$@"
