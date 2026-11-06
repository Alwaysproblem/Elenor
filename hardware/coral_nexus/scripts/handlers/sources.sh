#!/usr/bin/env bash
# sources <subcommand> handler for the cn dispatcher.
#
# Registered subcommands (commands.yaml): verify — delegates to
# scripts/lib/sources_tool.py. --lock and --out default to the repo's locked
# locations so the documented bare command `cn sources verify` works; the
# pass-through form (--lock/--out) stays available and overrides.
# Anything else exits 2.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib="$(cd "$here/../lib" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"

[ $# -ge 1 ] || { echo "error: sources requires a subcommand (verify)" >&2; exit 2; }
sub="$1"
shift
case "$sub" in
  verify)
    lock=""
    out=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --lock) lock="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        *) echo "error: sources verify: unknown argument $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$lock" ] || lock="$H/env/sources.lock"
    [ -n "$out" ] || out="$ROOT/out/coral_nexus/C001/verify-latest"
    mkdir -p "$(dirname "$out")"
    exec python3 "$lib/sources_tool.py" verify --lock "$lock" --out "$out"
    ;;
  *)
    echo "error: unknown sources subcommand '$sub' (verify)" >&2
    exit 2
    ;;
esac
