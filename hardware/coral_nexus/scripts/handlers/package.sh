#!/usr/bin/env bash
# package handler (C037): build -> resource recheck -> seal -> reload, by really linking the selftest ELF
# with the pinned C004 toolchain. Exit 1 on any build/seal/load refusal, 2 on usage, 3 on a missing
# prerequisite (no toolchain is a refusal, never a fabricated image).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"

binding=""; config=""; out="out/coral_nexus/package"
while [ $# -gt 0 ]; do
  case "$1" in
    --binding) binding="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    *) echo "error: package: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$binding" ] || { echo "error: package requires --binding" >&2; exit 2; }
[ -n "$config" ] || { echo "error: package requires --config" >&2; exit 2; }
[ -f "$binding" ] || { echo "error: binding $binding not found" >&2; exit 3; }
[ -f "$config" ] || { echo "error: config $config not found" >&2; exit 3; }
[ -d "$ROOT/out/coral_nexus/cache/toolchain/extracted/toolchain_coralnpu_v2/bin" ] || {
  echo "error: pinned toolchain not extracted (cn env seed pulls the locked archive)" >&2; exit 3; }

exec conda run --no-capture-output -n elenor-validator python "$H/scripts/lib/package_build.py" \
  --binding "$binding" --config "$config" --out "$out"
