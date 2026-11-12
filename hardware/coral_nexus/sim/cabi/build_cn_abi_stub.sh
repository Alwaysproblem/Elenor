#!/usr/bin/env bash
# Build libcn_cosim_stub.so + the ABI selftest driver inside coral-nexus-builder:c002 (C025/W1-03).
#
# Verilator 5.052 from the C004 seed is unpacked under the out dir; the slot table and ABI
# identity hashes are GENERATED from the sealed protocol artifacts (gen_slot_table.py), never
# hand-written. Cached by content digest under out/coral_nexus/cache/cn_abi_stub/.
#
# Usage: build_cn_abi_stub.sh [--force]
# Prints the cache directory (containing libcn_cosim_stub.so and abi_selftest_driver) on stdout.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"

CACHE_DIR="${CN_ABI_STUB_CACHE:-$ROOT/out/coral_nexus/cache/cn_abi_stub}"
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    *) echo "error: build_cn_abi_stub.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

IMAGE="$(python3 -c "import json;print(json.load(open('$H/env/image.lock'))['tag'])")"
VERILATOR_SEED="$ROOT/out/coral_nexus/cache/seeds/verilator-v5.052.tar.gz"
VERILATOR_ROOT_HOST="${CN_VERILATOR_ROOT:-$ROOT/out/coral_nexus/C017/build/verilator-5.052}"

RECIPE_SHA="$(sha256sum "$H/sim/cabi/cn_abi.h" "$H/sim/cabi/cn_abi.cc" "$H/sim/cabi/abi_selftest_driver.c" \
  "$H/sim/unit/cn_protocol_stub.sv" "$H/scripts/lib/gen_slot_table.py" "$H/protocol/generated/cn_layout.json" \
  "$H/protocol/cn_protocol_v1.yaml" "$0" | sha256sum | awk '{print $1}')"

if [ "$FORCE" -eq 0 ] && [ -f "$CACHE_DIR/libcn_cosim_stub.so" ] && [ -f "$CACHE_DIR/abi_selftest_driver" ] \
  && [ -f "$CACHE_DIR/build_meta.json" ] \
  && [ "$(python3 -c "import json;print(json.load(open('$CACHE_DIR/build_meta.json')).get('recipe_sha256',''))" 2>/dev/null)" = "$RECIPE_SHA" ]; then
  echo "$CACHE_DIR"
  exit 0
fi

if [ ! -x "$VERILATOR_ROOT_HOST/bin/verilator_bin" ]; then
  mkdir -p "$(dirname "$VERILATOR_ROOT_HOST")"
  rm -rf "$VERILATOR_ROOT_HOST"
  tar -xzf "$VERILATOR_SEED" -C "$(dirname "$VERILATOR_ROOT_HOST")"
fi

mkdir -p "$CACHE_DIR/work"
rm -rf "$CACHE_DIR/verilator-5.052"; cp -r "$VERILATOR_ROOT_HOST" "$CACHE_DIR/verilator-5.052"
bash "$H/scripts/lib/docker_run.sh" "$IMAGE" "$CACHE_DIR" -- bash -exc '
  set -euo pipefail
  export VERILATOR_ROOT=/out/verilator-5.052
  ls /out/verilator-5.052/bin/verilator >/dev/null
  cd /out/work
  # 1) slot table + ABI hashes from the sealed layout
  python3 /workspace/coral_nexus/scripts/lib/gen_slot_table.py \
    /workspace/coral_nexus/protocol/generated/cn_layout.json \
    /workspace/coral_nexus/protocol/cn_protocol_v1.yaml \
    /out/work/cn_slot_table.h
  # 2) Verilate the stub top (C++ outputs only)
  "$VERILATOR_ROOT/bin/verilator" --cc -Wno-fatal --top-module cn_protocol_stub \
    -Mdir obj /workspace/coral_nexus/sim/unit/cn_protocol_stub.sv >verilate.log 2>&1
  # 3) shared library: model + ABI + verilated runtime
  g++ -std=c++20 -O2 -fPIC -shared \
    -I/out/work -I/workspace/coral_nexus/sim/cabi \
    -I"$VERILATOR_ROOT/include" -I"$VERILATOR_ROOT/include/vltstd" \
    -I/out/work/obj \
    /workspace/coral_nexus/sim/cabi/cn_abi.cc \
    obj/*.cpp \
    "$VERILATOR_ROOT"/include/verilated.cpp \
    "$VERILATOR_ROOT"/include/verilated_cov.cpp \
    "$VERILATOR_ROOT"/include/verilated_threads.cpp \
    -ldl -lpthread \
    -o /out/libcn_cosim_stub.so
  # 4) selftest driver (dlopen the .so, so it exercises the exact loadable artifact)
  gcc -std=c11 -O2 \
    -I/workspace/coral_nexus/sim/cabi -I/out/work \
    /workspace/coral_nexus/sim/cabi/abi_selftest_driver.c \
    -ldl -o /out/abi_selftest_driver
'
rm -rf "$CACHE_DIR/work"
python3 - "$CACHE_DIR/build_meta.json" "$RECIPE_SHA" "$IMAGE" <<'PY'
import json
import sys
from pathlib import Path

meta, recipe, image = sys.argv[1:4]
import hashlib

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

Path(meta).write_text(
    json.dumps(
        {
            "schema": "cn_abi_stub_build/v1",
            "image": image,
            "recipe_sha256": recipe,
            "library_sha256": sha(Path(meta).parent / "libcn_cosim_stub.so"),
            "driver_sha256": sha(Path(meta).parent / "abi_selftest_driver"),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
echo "$CACHE_DIR"
