"""C030 x C028 native fetch-port integration: mutants, then real generated
PerCtxFetchMux (contextCount 1 and 2, sealed fetchDataBits = 256) driving the
real l1_program_port through tb_c028_port.sv's bridge.

Route (fixed, no owned-file changes): the existing chisel_binary target
//hdl/chisel/src/coralnpu:core_mini_verification_axi_cc_library_emit_verilog_binary
is built and RUN directly with `--contextCount={1,2} --lsuDataBits=128
--moduleName=CoreMiniVerification`, leaving fetchDataBits at its sealed
default 256. The genrule bundle (route a) is also built and probed as
evidence: it carries `module PerCtxFetchMux` but only at contextCount=1 and
fetchDataBits=128 (its baked gen_flags), which is why the binary is run
directly for the real geometry. The mux module closure (PerCtxFetchMux +
ContextIssueArbiter + InstructionBuffer + CircularBufferMulti) is extracted
from each emitted CoreMiniVerification.sv and compiled against the port.
"""

from __future__ import annotations

from typing import Any

import pytest
from w4_rtl_harness import Bench, Slice, check_verdicts, run_slice

TOP = "tb_c028_port"
EXTRACT_FN = """extract_modules() {
  src=$1; out=$2; rootmod=$3
  mods="$rootmod"
  : > "$out"
  for m in $mods; do
    awk -v mod="$m" '
      $0 == "module " mod "(" {f=1}
      f {print}
      f && $0 == "endmodule" {exit}
    ' "$src" >> "$out" || exit 1
    grep -q "^module $m(" "$out" || { echo "missing module $m" >&2; exit 1; }
  done
  while :; do
    new=""
    for dep in $(sed -n 's/^ *\\([A-Z][A-Za-z0-9_]*\\) [a-z_][A-Za-z0-9_]* (.*$/\\1/p' "$out" | sort -u); do
      case " $mods " in *" $dep "*) ;;
        *) if grep -q "^module $dep(" "$src"; then
             mods="$mods $dep"; new=1
             awk -v mod="$dep" '
               $0 == "module " mod "(" {f=1}
               f {print}
               f && $0 == "endmodule" {exit}
             ' "$src" >> "$out" || exit 1
           else echo "unresolved module $dep" >&2; exit 1; fi;;
      esac
    done
    [ -z "$new" ] && break
  done
  printf "%s\\n" "$mods" > "$out.manifest"
}"""

# Real PerCtxFetchMux Verilog is produced inside the container (no owned-file change): the
# genrule bundle is probed as route evidence, then the existing chisel_binary target is RUN for
# both sealed Context counts at the sealed 256-bit fetch width.
EMIT_STEPS = [
  # Route (a) evidence: the existing genrule bundle DOES contain the mux,
  # but only at contextCount=1 / fetchDataBits=128 (baked gen_flags).
  "mkdir -p /out/route-a",
  "rc=0",
  "$BZL build $COMMON //hdl/chisel/src/coralnpu:core_mini_verification_axi_cc_library_emit_verilog "
  "> /out/route-a/build.log 2>&1 || rc=$?",
  'printf "%s\\n" "$rc" > /out/route-a/build.rc',
  'test "$rc" -eq 0',
  "cp /tmp/bazel-bin/hdl/chisel/src/coralnpu/CoreMiniVerificationAxi.sv /out/route-a/",
  "cp /tmp/bazel-bin/hdl/chisel/src/coralnpu/VCoreMiniVerificationAxi_parameters.h /out/route-a/",
  '{ grep -c "^module PerCtxFetchMux(" /out/route-a/CoreMiniVerificationAxi.sv; '
  'grep -E "KP_(contextCount|fetchDataBits|ctxGenerationBits|ownerOpSeqBits)" '
  "/out/route-a/VCoreMiniVerificationAxi_parameters.h; } > /out/route-a/probe.txt",
  # Real geometry: build + RUN the existing emit binary with the sealed
  # default fetchDataBits=256 for both sealed context counts.
  # Verilator itself is already built and on VEXE (the shared harness did that once).
  "$BZL build $COMMON //hdl/chisel/src/coralnpu:core_mini_verification_axi_cc_library_emit_verilog_binary "
  "@coralnpu_hw//third_party/llvm-firtool:firtool "
  "> /out/emit-build.log 2>&1",
  "EMIT=/tmp/bazel-bin/hdl/chisel/src/coralnpu/core_mini_verification_axi_cc_library_emit_verilog_binary",
  "FIRTOOL_DIR=/tmp/bazel-bin/third_party/llvm-firtool",
  "for n in 1 2; do",
  '  CHISEL_FIRTOOL_PATH="$FIRTOOL_DIR" "$EMIT" --target-dir="/out/emit-c$n" '
  "--contextCount=$n --lsuDataBits=128 --moduleName=CoreMiniVerification "
  '> "/out/emit-c$n.log" 2>&1',
  '  grep -E "KP_(contextCount|fetchDataBits|ctxGenerationBits|ownerOpSeqBits)" '
  '"/out/emit-c$n/VCoreMiniVerification_parameters.h" > "/out/emit-c$n.params"',
  "done",
  EXTRACT_FN,
  "extract_modules /out/emit-c1/CoreMiniVerification.sv /out/perctx_mux_c1.sv PerCtxFetchMux",
  "extract_modules /out/emit-c2/CoreMiniVerification.sv /out/perctx_mux_c2.sv PerCtxFetchMux",
  'grep -q "^module PerCtxFetchMux(" /out/perctx_mux_c1.sv',
  'grep -q "^module PerCtxFetchMux(" /out/perctx_mux_c2.sv',
  'grep -q "io_accepted_1" /out/perctx_mux_c2.sv',
]

BENCH = Bench(
  top=TOP,
  sources=("c029_params_pkg.sv", "l1_program_port.sv", "tb/tb_c028_port.sv"),
  contexts=(1, 2),
  extra_sources=("/out/perctx_mux_c{contexts}.sv",),
  args_by_context={1: "+define+C028_C1"},
  timeout_s=300,
)
SPEC = Slice(
  name="c030fetch",
  benches=[BENCH],
  mutant_globs=("fetchint-*.patch",),
  mutant_bench=BENCH,
  min_mutants=6,
  mutant_contexts=2,
  post_toolchain_steps=EMIT_STEPS,
)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  return run_slice(SPEC, tmp_path_factory)


def test_c030_fetch_integration_real_mux_and_mutation_acceptance(evidence: dict[str, Any]) -> None:
  path = evidence["path"]
  # Route evidence: the existing bundle emits the mux, at C=1/128b only.
  probe = (path / "route-a/probe.txt").read_text().split("\n")
  assert probe[0].strip() == "1", "route (a): PerCtxFetchMux module count"
  assert any("KP_contextCount  1" in line for line in probe), probe
  assert any("KP_fetchDataBits  128" in line for line in probe), probe
  # Real geometry emissions for both sealed context counts at 256-bit lines.
  for count in (1, 2):
    params_header = (path / f"emit-c{count}.params").read_text()
    assert f"KP_contextCount  {count}" in params_header, params_header
    assert "KP_fetchDataBits  256" in params_header, params_header
    assert "KP_ctxGenerationBits  16" in params_header, params_header
    assert "KP_ownerOpSeqBits  32" in params_header, params_header
  check_verdicts(evidence, pass_marker="C030_PASS")
  for case in evidence["positive"]:
    log = (path / f"{case['label']}.log").read_text()
    assert f"contexts={case['context_count']}" in log, log[-4000:]
