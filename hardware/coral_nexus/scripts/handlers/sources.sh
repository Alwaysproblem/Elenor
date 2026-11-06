#!/usr/bin/env bash
# sources <subcommand> handler for the cn dispatcher.
#
# Registered subcommands (commands.yaml): verify — delegates to
# scripts/lib/verify_sources.sh with the subcommand stripped. Anything else
# exits 2.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib="$(cd "$here/../lib" && pwd)"

[ $# -ge 1 ] || { echo "error: sources requires a subcommand (verify)" >&2; exit 2; }
sub="$1"
shift
case "$sub" in
  verify) exec bash "$lib/verify_sources.sh" "$@" ;;
  *)
    echo "error: unknown sources subcommand '$sub' (verify)" >&2
    exit 2
    ;;
esac
