#!/usr/bin/env bash
# build --prepare-only handler (C019): generate + validate one cs1 target WITHOUT claiming any full Tile
# compile. It (1) re-renders every target from its sources and byte-compares the committed files,
# (2) validates the target through the C008 constraint consumer, (3) compiles a GENERATED C consumer of the
# target constants with the pinned C004 RV32 toolchain -- runnability evidence is a real compile, not field
# equality. A cs1_c* config is required; kind p0_upstream is refused (that is the baseline command's input).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$here/../.." && pwd)"
ROOT="$(cd "$H/../.." && pwd)"

prepare_only=0; config=""; out="out/coral_nexus/build"
while [ $# -gt 0 ]; do
  case "$1" in
    prepare-only|--prepare-only) prepare_only=1; shift ;;
    --config) config="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    *) echo "error: build: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ "$prepare_only" = "1" ] || { echo "error: only --prepare-only is implemented (full build is C040)" >&2; exit 4; }
[ -n "$config" ] || { echo "error: build --prepare-only requires --config" >&2; exit 2; }
mkdir -p "$out"

python3 - "$config" <<'PY'
import sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
if cfg.get("kind") != "cs1_target":
    print(f"error: build --prepare-only needs a cs1_target config, got kind {cfg.get('kind')!r}", file=sys.stderr)
    raise SystemExit(3)
PY
[ $? -eq 0 ] || exit 3

python3 "$H/scripts/lib/gen_targets.py" --check || { echo "error: committed targets are stale or hand-edited" >&2; exit 1; }

python3 - "$config" "$out" "$H" "$ROOT" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
import yaml

cfg_path, out, H, ROOT = (Path(p) for p in sys.argv[1:5])
cfg = yaml.safe_load(cfg_path.read_text())
name = cfg["name"]
c = cfg["constraints"]
toolchain = ROOT / "out/coral_nexus/cache/toolchain/extracted/toolchain_coralnpu_v2/bin"
header = out / f"{name}_params.h"
header.parent.mkdir(parents=True, exist_ok=True)
header.write_text(
    f"/* GENERATED consumer of the {name} target (C019); compiled to prove the config is usable. */\n"
    "#ifndef CN_CS1_PARAMS_H_\n#define CN_CS1_PARAMS_H_\n"
    f"#define CN_{name.upper()}_CONTEXT_COUNT {c['context_count']}\n"
    f"#define CN_{name.upper()}_CTX_SLOT_BITS {cfg['index_widths']['ctx_slot_bits']}\n"
    f"#define CN_{name.upper()}_PROGRAM_SLOTS {c['program_slots']}\n"
    f"#define CN_{name.upper()}_ITCM_BYTES {c['itcm_bytes']}\n"
    f"#define CN_{name.upper()}_DTCM_BYTES {c['dtcm_bytes']}\n"
    f"#define CN_{name.upper()}_TEXT_BYTES {cfg['envelope']['measured']['text_bytes']}\n"
    f"#define CN_{name.upper()}_STACK_BYTES {cfg['envelope']['measured']['stack_bytes']}\n"
    "#endif\n"
)
consumer = out / f"{name}_consumer.c"
consumer.write_text(
    f'#include "{name}_params.h"\n'
    "\n"
    f"_Static_assert(CN_{name.upper()}_CONTEXT_COUNT >= 1, \"a target with no Context is not a target\");\n"
    f"_Static_assert(CN_{name.upper()}_CTX_SLOT_BITS >= 1, \"C1 must not produce a zero-width index\");\n"
    f"_Static_assert(CN_{name.upper()}_PROGRAM_SLOTS * CN_{name.upper()}_TEXT_BYTES <= CN_{name.upper()}_ITCM_BYTES,\n"
    "                \"the measured images do not fit ITCM\");\n"
    f"_Static_assert(CN_{name.upper()}_STACK_BYTES <= CN_{name.upper()}_DTCM_BYTES, \"stack exceeds DTCM\");\n"
    "\nint cn_cs1_target_consumer(void) { return 0; }\n"
)
MARCH = "rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma"
compile_cmd = [
    str(toolchain / "clang"), "--target=riscv32-unknown-elf", f"-march={MARCH}", "-mabi=ilp32",
    "-mcmodel=medany", "-ffreestanding", "-fno-builtin", "-Wall", "-Wextra", "-Werror",
    "-c", str(consumer), "-o", str(out / f"{name}_consumer.o"),
]
proc = subprocess.run(compile_cmd, capture_output=True, text=True)
verdict = {
    "schema": "cn_build_prepare_only/v1",
    "config": str(cfg_path),
    "target": name,
    "constraints_sha256": cfg["fingerprints"]["constraints_sha256"],
    "consumer_compiled": proc.returncode == 0,
    "compiler": "toolchain_coralnpu_v2 (locked)",
    "executable_claimed": False,
    "note": "prepare-only: no full Tile top exists; C040 provides it",
}
(out / "prepare_only_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
if proc.returncode != 0:
    print(proc.stderr, file=sys.stderr)
    raise SystemExit(1)
print(json.dumps(verdict, indent=2))
PY
