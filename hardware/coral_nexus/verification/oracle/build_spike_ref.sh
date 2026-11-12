#!/usr/bin/env bash
# Build and cache libcn_spike_ref.so inside coral-nexus-builder:c002 from the
# W0-01 pinned riscv-isa-sim@93a10ae6 archive plus CoralNPU's 9 pinned patches
# and sw/coralnpu_sim/spike_simulator.{h,cc} (C021 / W7-02a).
#
# Usage: build_spike_ref.sh [--cache-dir DIR] [--force]
# Prints the host path to build_meta.json on stdout.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"

CACHE_DIR="${CN_SPIKE_CACHE_DIR:-$ROOT/out/coral_nexus/cache/spike_ref}"
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "error: build_spike_ref.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

SEED="$ROOT/out/coral_nexus/cache/seeds/riscv_isa_sim-93a10ae6.tar.gz"
EXPECTED_SEED_SHA="064f4c1f22005899fa5106b822bfdc7034ffc9babc2f5821771589047f028a66"
if [ ! -f "$SEED" ]; then
  echo "error: missing pinned Spike seed $SEED" >&2
  exit 3
fi
actual_seed_sha="$(sha256sum "$SEED" | awk '{print $1}')"
if [ "$actual_seed_sha" != "$EXPECTED_SEED_SHA" ]; then
  echo "error: Spike seed sha256 $actual_seed_sha != $EXPECTED_SEED_SHA" >&2
  exit 1
fi

PATCH_DIR="$H/upstream/coralnpu/third_party/spike"
SIM_CC="$H/upstream/coralnpu/sw/coralnpu_sim/spike_simulator.cc"
SIM_H="$H/upstream/coralnpu/sw/coralnpu_sim/spike_simulator.h"
SHIM_CC="$here/spike_c_shim.cc"
IMAGE_LOCK="$H/env/image.lock"
IMAGE="$(python3 -c "import json; print(json.load(open('$IMAGE_LOCK'))['tag'])")"

RECIPE_DIGEST="$(
  sha256sum "$SEED" "$PATCH_DIR"/000*.patch "$SIM_CC" "$SIM_H" "$SHIM_CC" "$IMAGE_LOCK" "$0" \
    | sha256sum | awk '{print $1}'
)"

mkdir -p "$CACHE_DIR"
META="$CACHE_DIR/build_meta.json"
SO="$CACHE_DIR/libcn_spike_ref.so"

if [ "$FORCE" -eq 0 ] && [ -f "$SO" ] && [ -f "$META" ]; then
  cached_digest="$(python3 -c "import json; print(json.load(open('$META')).get('recipe_sha256',''))" 2>/dev/null || true)"
  if [ "$cached_digest" = "$RECIPE_DIGEST" ]; then
    echo "$META"
    exit 0
  fi
fi

cp -L "$SEED" "$CACHE_DIR/riscv_isa_sim.tar.gz"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  -v "$ROOT:/workspace:ro" \
  -v "$(cd "$CACHE_DIR" && pwd):/out" \
  -w /workspace \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    if [ ! -f /out/build/libriscv.a ] || [ "'"$FORCE"'" = "1" ]; then
      rm -rf /out/src /out/build
      mkdir -p /out/src /out/build
      tar -xf /out/riscv_isa_sim.tar.gz --strip-components=1 -C /out/src
      cd /out/src
      for p in /workspace/hardware/coral_nexus/upstream/coralnpu/third_party/spike/000*.patch; do
        patch -p1 < "$p" >/dev/null
      done
      cd /out/build
      CXXFLAGS="-std=c++20 -fPIC" CFLAGS="-fPIC" /out/src/configure \
        --without-boost --without-boost-asio --without-boost-regex >/out/configure.log 2>&1
      make -j8 libriscv.a libsoftfloat.a libfesvr.a libdisasm.a libfdt.a >/out/make.log 2>&1
    fi
    g++ -std=c++20 -O2 -fPIC -shared \
      -I/workspace/hardware/coral_nexus/upstream/coralnpu \
      -I/out/build -I/out/src \
      /workspace/hardware/coral_nexus/upstream/coralnpu/sw/coralnpu_sim/spike_simulator.cc \
      /workspace/hardware/coral_nexus/verification/oracle/spike_c_shim.cc \
      /out/build/libriscv.a \
      /out/build/libsoftfloat.a \
      /out/build/libfesvr.a \
      /out/build/libdisasm.a \
      /out/build/libfdt.a \
      -ldl -lpthread \
      -o /out/libcn_spike_ref.so.tmp
    mv -f /out/libcn_spike_ref.so.tmp /out/libcn_spike_ref.so
  '

rm -f "$CACHE_DIR/riscv_isa_sim.tar.gz"
so_sha="$(sha256sum "$SO" | awk '{print $1}')"
python3 - "$META" "$RECIPE_DIGEST" "$EXPECTED_SEED_SHA" "$so_sha" "$IMAGE" <<'PY'
import json
import sys
from pathlib import Path

meta, recipe, seed_sha, so_sha, image = sys.argv[1:6]
doc = {
  "schema": "cn_spike_ref_build/v1",
  "image": image,
  "riscv_isa_sim_commit": "93a10ae685ac85bd3d8a62054f8f180a4f76fc82",
  "seed_sha256": seed_sha,
  "recipe_sha256": recipe,
  "library": "libcn_spike_ref.so",
  "library_sha256": so_sha,
}
Path(meta).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
PY
echo "$META"
