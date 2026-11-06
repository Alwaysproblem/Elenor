#!/usr/bin/env bash
# In-container offline rebuild probe (C004).
#
# Runs with --network none, an EMPTY output_user_root/disk_cache/repository_cache
# and only the exported vendor archive as a dependency source. Any attempt to
# reach the network or a stale local cache fails the build instead of
# silently succeeding.
#
# The exported archive is mounted read-only and its pinned tree digest is
# re-verified before use. Bazel needs a writable vendor root (it creates one
# `bazel-external` symlink), so a working copy is made inside the report dir;
# the archive itself is never modified.
#
# Env: WORK_DIR (writable materialized work root), VENDOR_DIR (exported
#      bazel_vendor_cache, read-only), VENDOR_SHA256 (pinned digest),
#      REPORT_DIR (writable report dir).
set -uo pipefail

work="${WORK_DIR:?WORK_DIR required}"
vendor="${VENDOR_DIR:?VENDOR_DIR required}"
expected="${VENDOR_SHA256:?VENDOR_SHA256 required}"
report="${REPORT_DIR:?REPORT_DIR required}"

caches="$report/caches"
mkdir -p "$caches/bazel-root" "$caches/disk-cache" "$caches/repo-cache" "$report/home"
export HOME="$report/home"

for d in bazel-root disk-cache repo-cache; do
  n=$(find "$caches/$d" -mindepth 1 | wc -l)
  if [ "$n" -ne 0 ]; then
    echo "error: cache $d is not empty ($n entries)" >&2
    exit 1
  fi
done

digest_cmd='import hashlib, os, sys
from pathlib import Path
root = Path(sys.argv[1])
d = hashlib.sha256()
for p in sorted(root.rglob("*")):
  rel = p.relative_to(root).as_posix()
  if p.is_symlink():
    entry = "symlink:" + os.readlink(p)
  elif p.is_file():
    entry = hashlib.sha256(p.read_bytes()).hexdigest()
  else:
    continue
  d.update(rel.encode()); d.update(b"\0"); d.update(entry.encode()); d.update(b"\0")
print(d.hexdigest())'

echo "[offline] verifying archive digest"
actual="$(python3 -c "$digest_cmd" "$vendor")"
if [ "$actual" != "$expected" ]; then
  echo "error: vendor archive digest $actual != pinned $expected" >&2
  exit 1
fi
echo "[offline] archive digest ok: ${actual:0:16}"

echo "[offline] creating writable vendor working copy"
rm -rf "$report/vendor"
cp -a "$vendor" "$report/vendor"
copy_digest="$(python3 -c "$digest_cmd" "$report/vendor")"
if [ "$copy_digest" != "$expected" ]; then
  echo "error: working copy digest $copy_digest != pinned $expected" >&2
  exit 1
fi

targets="//tests/verilator_sim:core_mini_axi_sim //examples:coralnpu_v2_hello_world_add_floats //nexus_p0:p0_scalar_add"
cd "$work/coralnpu" || { echo "error: $work/coralnpu missing" >&2; exit 1; }

bazel --output_user_root="$caches/bazel-root" build -c opt \
  --disk_cache="$caches/disk-cache" \
  --repository_cache="$caches/repo-cache" \
  --vendor_dir="$report/vendor" \
  $targets 2>&1 | tee "$report/build.log"
rc=${PIPESTATUS[0]}

# ELF targets land in a static-link opt config (k8-opt-ST-*), not in
# bazel-bin (which symlinks k8-opt). Search the real execroot.
exec_root="$(bazel --output_user_root="$caches/bazel-root" info output_base)/execroot/_main"
mkdir -p "$report/artifacts"
for name in coralnpu_v2_hello_world_add_floats.elf p0_scalar_add.elf; do
  found=$(find "$exec_root/bazel-out" -path "*k8-opt-ST-*" -name "$name" -type f 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    cp -f "$found" "$report/artifacts/"
  else
    echo "warning: $name not found under $exec_root/bazel-out" >&2
  fi
done

python3 - "$report" "$work" "$vendor" "$rc" "$expected" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

report = Path(sys.argv[1])
work = Path(sys.argv[2])
vendor = sys.argv[3]
rc = int(sys.argv[4])
pinned = sys.argv[5]

elfs = [
  "coralnpu_v2_hello_world_add_floats.elf",
  "p0_scalar_add.elf",
]
found = {}
for name in elfs:
  p = report / "artifacts" / name
  if p.is_file():
    found[name] = hashlib.sha256(p.read_bytes()).hexdigest()
out = {
  "schema": "cn_offline_rebuild/v1",
  "network": "none",
  "caches": "empty at start (bazel-root, disk-cache, repo-cache)",
  "vendor_dir": vendor,
  "vendor_sha256": pinned,
  "vendor_digest_verified": True,
  "bazel_rc": rc,
  "elfs": found,
  "pass": rc == 0 and len(found) == len(elfs),
  "rebuilt_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
(report / "offline_rebuild.json").write_text(
  json.dumps(out, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(out, indent=2, sort_keys=True))
sys.exit(0 if out["pass"] else 1)
PY
