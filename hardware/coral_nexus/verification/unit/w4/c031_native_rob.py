"""C031 native full-ROB emission and ``c031_rob_adapter`` generation (runner-time helper).

The parent integration (``test_w4_c031.py``) imports :data:`NATIVE_STEPS` and passes
it as ``Slice(post_toolchain_steps=NATIVE_STEPS)``. The steps are fully self-contained
for the locked builder container (only ``/work``, ``/out``, ``/rtl``, ``/mutants`` and
the Bazel caches are mounted): the generator script travels inside the step list as a
quoted heredoc, so nothing under the repository's verification directory is read at
run time.

What the steps do (after the shared harness has the toolchain ready):

1. Build and RUN the pinned chisel emitter binary
   ``//hdl/chisel/src/coralnpu:core_mini_verification_axi_cc_library_emit_verilog_binary``
   with ``--contextCount={1,2} --lsuDataBits=128 --moduleName=CoreMiniVerification
   --enableFloat=True --enableZfbfmin=True --enableVerification=True``. This selects
   the FULL ``RetirementBuffer`` behind the real patched ``RetirementPartitionBank``
   from the materialized Core. The C030 fetch slice emission (``useRetirementBuffer``
   false -> mini ROB) is explicitly NOT reused as proof; the generated parameter
   header must show ``KP_useRetirementBuffer``/``KP_enableVerification``/
   ``KP_enableFloat`` true and the sealed Context count, or generation fails closed
   with no silent mini fallback.
2. Extract the real ``RetirementPartitionBank`` dependency closure from each emission
   (``EXTRACT_FN`` reused from ``test_w4_c030_fetch``) and require exactly the sealed
   Context count of instantiated ``RetirementBuffer`` partitions per emission.
3. Write ``/out/c031_rob_c{1,2}.sv`` = closure + generated ``c031_rob_adapter``.
   The adapter parses the emitted port declarations and connects EVERY port (mapped
   inputs, remaining inputs tied ``'0``, all outputs to named wires) so Verilator sees
   no PINMISSING/PINCONNECTEMPTY, and fails closed listing any required port that the
   emitter pruned.

Parent bench wiring for the memory slice:

    extra_sources=("/out/c031_rob_c{contexts}.sv",)
    args_by_context={1: "+define+C031_C1"}

Bench sources must compile ``c029_params_pkg.sv`` and ``c031_lsu_pkg.sv`` before the
generated file (the adapter only references ``c031_lsu_pkg`` types; Verilator needs
the package earlier on the command line).

Adapter semantics (native hardware, nothing modeled):

- ``enq_valid_i`` performs exactly one dispatch per asserted cycle on instruction
  lane 0 (``io_inst_0_ready`` is producer-owned and tied ready, mirroring the SCore
  dispatch producer that paces on ``nSpace``; the TB keeps per-Context outstanding
  <= ``retirementBufferSize`` and pulses one instruction per cycle).
- Encodings are real RISC-V words with ``imm=0`` and ``rd`` forwarded from the
  request, so both Contexts dispatch identical instruction bits (identity rides
  the OwnerTag): scalar load LW (0000011) / scalar store SW (0100011, ``rd``
  rides the rs2 data field) use ``rs1=x0``; float load FLW (0000111) / float
  store FSW (0100111) use ``rs1=x1`` per the parent FLW proof encodings
  (0x0000a287 = FLW f5,0(x1); 0x0050a027 = FSW f5,0(x1)); all with funct3=010
  (byte_count==4 enforced); ``enq_i.fence`` maps to ``FENCE iorw,iorw``
  (32'h0FF0000F), a non-writing instruction that self-retires.
- Completions with ``final_state==COMPLETED`` only: store -> the actual
  ``storeComplete`` OwnerTag port; scalar load -> ``writeDataScalar`` lane
  ``instructionLanes`` (the first regfile producer port beyond the dispatch lanes,
  the SCore long-latency LSU path); float load -> ``writeDataFloat`` lane 1
  (``FloatCore.write_ports(1)``, the ``lsu_rd`` forwarding port; lane 0 is the
  FloatCore ALU port and stays tied). ``FAULTED`` -> ``fault`` OwnerTag with
  ``decode=0`` (runtime fault, never a decode fault). ``ISOLATED`` (cancel
  isolation) is deliberately not mapped to the ROB.
- ``generation_i`` feeds ``perCtxGen``; the adapter never pulses ``perCtxReset``,
  so stale injection is a completion-generation mismatch counted by the real
  ``staleRejected`` counters of each partition.
- Outputs: ``empty_o[c]`` = ``perCtxEmpty_c``; ``retired_o[c]`` = ``perCtxNRetired_c``
  (the real per-cycle retiring lane count); ``stale_o`` = aggregate ``staleRejected``.
- ``C031_C1`` (the parent's ``+define+`` for Context count 1) is cross-checked
  against the generation at time 0 (``$fatal`` on mismatch).
"""

from __future__ import annotations

from test_w4_c030_fetch import EXTRACT_FN

ROB_MODULE = "RetirementPartitionBank"
ADAPTER_MODULE = "c031_rob_adapter"
GENERATED_SV = "/out/c031_rob_c{contexts}.sv"
EMIT_BINARY = "//hdl/chisel/src/coralnpu:core_mini_verification_axi_cc_library_emit_verilog_binary"
PARAM_KEYS = (
    "KP_(contextCount|ctxGenerationBits|ownerOpSeqBits|instructionLanes|"
    "retirementBufferSize|retirementLanes|xlen|programCounterBits|"
    "enableVerification|useRetirementBuffer|enableFloat|enableZfbfmin|"
    "enableRvv|enableVme)"
)

GENERATOR_PY = '''#!/usr/bin/env python3
"""Generate the C031 native ROB adapter file (runs inside the builder container).

usage: c031_rob_gen.py <contexts> <params-header> <closure.sv> <out.sv>

Fails closed on any missing full-mode parameter proof or any required port the
emitter pruned. The adapter only rewires c031_lsu_pkg requests/completions onto
the real emitted RetirementPartitionBank ports; no ROB behavior is modeled.
"""
import os
import re
import sys

ROB = "RetirementPartitionBank"
ADAPTER = "c031_rob_adapter"

EXPECT_TRUE = (
    "KP_enableVerification",
    "KP_useRetirementBuffer",
    "KP_enableFloat",
    "KP_enableZfbfmin",
)
EXPECT_FALSE = ("KP_enableRvv", "KP_enableVme")
EXPECT_CONST = {
    "KP_ctxGenerationBits": "16",
    "KP_ownerOpSeqBits": "32",
    "KP_xlen": "32",
    "KP_programCounterBits": "32",
}
INT_KEYS = ("KP_instructionLanes", "KP_retirementBufferSize", "KP_retirementLanes")


def die(msg):
    sys.stderr.write("c031_rob_gen: %s\\n" % msg)
    sys.exit(1)


def parse_params(text):
    got = {}
    for line in text.splitlines():
        m = re.match(r"#define\\s+(KP_\\w+)\\s+(\\S+)", line)
        if m:
            got[m.group(1)] = m.group(2)
    return got


def parse_ports(closure):
    m = re.search(r"^module %s\\((.*?)^\\);" % ROB, closure, re.S | re.M)
    if not m:
        die("module %s port list not found in closure" % ROB)
    body = re.sub(r"//[^\\n]*", "", m.group(1))
    ports, order = {}, []
    direction, width = None, 1
    for item in body.split(","):
        toks = item.split()
        name = None
        for t in toks:
            if t in ("input", "output", "inout"):
                direction = t
                # A new declaration statement starts at 1 bit unless its own
                # range follows; continuation items (no direction keyword)
                # inherit both direction and width from the previous item.
                width = 1
            elif t.startswith("["):
                r = re.fullmatch(r"\\[(\\d+):(\\d+)\\]", t)
                if not r:
                    die("unsupported port range %r in %r" % (t, item.strip()))
                width = int(r.group(1)) - int(r.group(2)) + 1
            elif name is None:
                name = t
            else:
                die("unexpected token %r in port item %r" % (t, item.strip()))
        if name is None:
            continue
        if direction is None:
            die("port %s has no direction" % name)
        if name in ports:
            die("duplicate port %s" % name)
        ports[name] = (direction, width)
        order.append((name, direction, width))
    return ports, order


def fit(ports, name, src, src_w):
    """Width-fit a mapped source expression to the emitted input port."""
    direction, width = ports[name]
    if direction != "input":
        die("%s is not an input" % name)
    if width == src_w:
        return src
    if width == 1:
        return "%s[0]" % src
    if width < src_w:
        return "%s[%d:0]" % (src, width - 1)
    return "{%d'd0, %s}" % (width - src_w, src)


def out_ext(ports, name, want):
    direction, width = ports[name]
    wire = "w_" + name
    if width == want:
        return wire
    if width < want:
        return "{%d'd0, %s}" % (want - width, wire)
    die("%s width %d exceeds the agreed adapter port width %d" % (name, width, want))


def main():
    if len(sys.argv) != 5:
        die("usage: c031_rob_gen.py <contexts> <params> <closure> <out>")
    n = int(sys.argv[1])
    if n not in (1, 2):
        die("contexts must be 1 or 2")
    params = parse_params(open(sys.argv[2]).read())
    if params.get("KP_contextCount") != str(n):
        die(
            "KP_contextCount=%r != %d (wrong emission for this file)"
            % (params.get("KP_contextCount"), n)
        )
    for key in EXPECT_TRUE:
        if params.get(key) != "true":
            die(
                "%s=%r (expected true): full-ROB/FP emission not proven; "
                "mini or no-float fallback is forbidden" % (key, params.get(key))
            )
    for key in EXPECT_FALSE:
        if params.get(key) != "false":
            die("%s=%r (expected false for the scalar C031 slice)" % (key, params.get(key)))
    for key, want in EXPECT_CONST.items():
        if params.get(key) != want:
            die("%s=%r != %s" % (key, params.get(key), want))
    ints = {}
    for key in INT_KEYS:
        try:
            ints[key] = int(params[key])
        except (KeyError, ValueError):
            die("%s missing or invalid in params header" % key)
    if ints["KP_instructionLanes"] < 1 or ints["KP_retirementBufferSize"] < 1:
        die("degenerate ROB geometry: %r" % (ints,))
    scalar_lane = ints["KP_instructionLanes"]
    float_lane = 1

    closure = open(sys.argv[3]).read()
    ports, order = parse_ports(closure)

    req = {
        "clock": ("clk", 1),
        "reset": ("~rst_n", 1),
        "io_dispatchOpSeq_0": ("enq_i.owner.owner_op_seq", 32),
        "io_inst_0_valid": ("enq_ok", 1),
        "io_inst_0_ready": ("1'b1", 1),
        "io_inst_0_bits_addr": ("enq_i.pc", 32),
        "io_inst_0_bits_inst": ("inst_word", 32),
        "io_writeAddrScalar_0_valid": ("enq_mark_scalar", 1),
        "io_writeAddrScalar_0_addr": ("enq_i.rd", 5),
        "io_storeComplete_valid": ("cmpl_store_valid", 1),
        "io_storeComplete_bits_ctx_slot": ("completion_i.owner.ctx_slot[0]", 1),
        "io_storeComplete_bits_ctx_generation": ("completion_i.owner.ctx_generation", 16),
        "io_storeComplete_bits_owner_op_seq": ("completion_i.owner.owner_op_seq", 32),
        "io_fault_valid": ("cmpl_fault_valid", 1),
        "io_fault_bits_mepc": ("completion_i.pc", 32),
        "io_fault_bits_owner_ctx_slot": ("completion_i.owner.ctx_slot[0]", 1),
        "io_fault_bits_owner_ctx_generation": ("completion_i.owner.ctx_generation", 16),
        "io_fault_bits_owner_owner_op_seq": ("completion_i.owner.owner_op_seq", 32),
        "io_writeAddrFloat_valid": ("enq_mark_float", 1),
        "io_writeAddrFloat_addr": ("enq_i.rd", 5),
    }
    for c in range(n):
        req["io_perCtxGen_%d" % c] = ("generation_i[%d]" % c, 16)
    if n > 1:
        req["io_activeCtx"] = ("enq_i.owner.ctx_slot[0]", 1)
    sl = "io_writeDataScalar_%d" % scalar_lane
    req[sl + "_valid"] = ("cmpl_scalar_valid", 1)
    req[sl + "_bits_owner_ctx_slot"] = ("completion_i.owner.ctx_slot[0]", 1)
    req[sl + "_bits_owner_ctx_generation"] = ("completion_i.owner.ctx_generation", 16)
    req[sl + "_bits_owner_owner_op_seq"] = ("completion_i.owner.owner_op_seq", 32)
    fl = "io_writeDataFloat_%d" % float_lane
    req[fl + "_valid"] = ("cmpl_float_valid", 1)
    req[fl + "_bits_owner_ctx_slot"] = ("completion_i.owner.ctx_slot[0]", 1)
    req[fl + "_bits_owner_ctx_generation"] = ("completion_i.owner.ctx_generation", 16)
    req[fl + "_bits_owner_owner_op_seq"] = ("completion_i.owner.owner_op_seq", 32)
    # Payload fields the emitter may prune per mode (the ROB matches strictly by
    # OwnerTag; result-data capture is what keeps data ports alive). Mapped only
    # when the emitted module actually carries them.
    opt = {
        sl + "_bits_addr": ("completion_i.rd", 5),
        sl + "_bits_data": ("completion_i.data", 32),
        fl + "_bits_addr": ("completion_i.rd", 5),
        fl + "_bits_data": ("completion_i.data", 32),
    }

    need_out = ["io_staleRejected"]
    for c in range(n):
        need_out.append("io_perCtxEmpty_%d" % c)
        need_out.append("io_perCtxNRetired_%d" % c)

    missing = [k for k in sorted(set(list(req) + need_out)) if k not in ports]
    if missing:
        die("required ports pruned from emitted %s: %s" % (ROB, ", ".join(missing)))
    for k in req:
        if ports[k][0] != "input":
            die("%s is not an input in the emitted module" % k)
    for k in need_out:
        if ports[k][0] != "output":
            die("%s is not an output in the emitted module" % k)
    for c in range(n):
        if ports["io_perCtxEmpty_%d" % c][1] != 1:
            die("io_perCtxEmpty_%d is not 1 bit" % c)

    mapped = dict(req)
    for name, spec in opt.items():
        if name in ports and ports[name][0] == "input":
            mapped[name] = spec

    conn, decl_used, decl_unused = [], [], []
    for name, direction, width in order:
        if direction == "input":
            if name in mapped:
                src, src_w = mapped[name]
                expr = fit(ports, name, src, src_w)
            else:
                expr = "'0"
            conn.append("    .%s(%s)," % (name, expr))
        else:
            rng = "" if width == 1 else "[%d:0] " % (width - 1)
            decl = "  wire %s%s;" % (rng, "w_" + name)
            (decl_used if name in need_out else decl_unused).append(decl)
            conn.append("    .%s(w_%s)," % (name, name))
    conn[-1] = conn[-1].rstrip(",")
    if not conn[-1].endswith(")"):
        die("internal: last connection malformed: %r" % conn[-1])

    lines = []
    lines.append("// %s -- AUTO-GENERATED at runner time (c031_native_rob.NATIVE_STEPS)." % ADAPTER)
    lines.append("// Regenerate by re-running the c031 native emission steps; do not edit.")
    lines.append("//")
    lines.append("// Contexts: %d. Emission flags: --contextCount=%d --lsuDataBits=128" % (n, n))
    lines.append("//   --moduleName=CoreMiniVerification --enableFloat=True")
    lines.append("//   --enableZfbfmin=True --enableVerification=True (full ROB, never mini).")
    lines.append("// Parameter-header proof carried by this generation:")
    for key in sorted(params):
        lines.append("//   %s %s" % (key, params[key]))
    manifest_path = sys.argv[3] + ".manifest"
    if os.path.exists(manifest_path):
        manifest = open(manifest_path).read().split()
        lines.append("// Real closure extracted from the materialized Core: %s" % " ".join(manifest))
    lines.append("//")
    lines.append("// Mapping contract (see c031_native_rob docstring): one dispatch per")
    lines.append("// enq_valid_i cycle on lane 0 as real LW/FLW/SW/FSW words (imm=0, rd")
    lines.append("// forwarded; scalar rs1=x0, float rs1=x1: 0x0000a287=FLW f5,0(x1),")
    lines.append("// 0x0050a027=FSW f5,0(x1); byte_count==4 enforced; fence -> FENCE iorw,iorw);")
    lines.append("// COMPLETED store -> storeComplete OwnerTag; scalar load -> writeDataScalar")
    lines.append("// lane %d (first regfile producer past the dispatch lanes);" % scalar_lane)
    lines.append("// -> writeDataFloat lane %d (FloatCore.write_ports(1), the lsu_rd path);" % float_lane)
    lines.append("// FAULTED -> fault OwnerTag (decode=0); ISOLATED (cancel) not mapped;")
    lines.append("// generation_i -> perCtxGen, perCtxReset tied 0 (stale injection is a")
    lines.append("// completion-generation mismatch counted by the real staleRejected).")
    lines.append("module %s #(" % ADAPTER)
    lines.append("  parameter int CONTEXT_COUNT = %d" % n)
    lines.append(") (")
    lines.append("  input logic clk,")
    lines.append("  input logic rst_n,")
    lines.append("  input logic enq_valid_i,")
    lines.append("  input c031_lsu_pkg::request_t enq_i,")
    lines.append("  input logic completion_valid_i,")
    lines.append("  input c031_lsu_pkg::completion_t completion_i,")
    lines.append("  input logic [15:0] generation_i [CONTEXT_COUNT],")
    lines.append("  output logic [CONTEXT_COUNT-1:0] empty_o,")
    lines.append("  output logic [3:0] retired_o [CONTEXT_COUNT],")
    lines.append("  output logic [15:0] stale_o")
    lines.append(");")
    lines.append("`ifdef C031_C1")
    lines.append("  localparam int GEN_CONTEXTS = 1;")
    lines.append("`else")
    lines.append("  localparam int GEN_CONTEXTS = 2;")
    lines.append("`endif")
    lines.append("  initial begin")
    lines.append("    if (CONTEXT_COUNT != GEN_CONTEXTS)")
    fmt = '%s: CONTEXT_COUNT=%%0d but this generation is %%0d (C031_C1 mismatch)' % ADAPTER
    lines.append('      $fatal(1, "%s", CONTEXT_COUNT, GEN_CONTEXTS);' % fmt)
    lines.append("  end")
    lines.append("")
    lines.append("  wire enq_slot_ok = (enq_i.owner.ctx_slot < CONTEXT_COUNT);")
    lines.append("  wire enq_word_ok = enq_i.fence || (enq_i.byte_count == 3'd4);")
    lines.append("  wire enq_ok = enq_valid_i && enq_slot_ok && enq_word_ok;")
    lines.append("  always_comb begin")
    lines.append("    if (enq_valid_i && !(enq_slot_ok && enq_word_ok))")
    fmt = '%s: enq rejected: byte_count=%%0d ctx_slot=0x%%0h fence=%%0b' % ADAPTER
    lines.append('      $fatal(1, "%s",' % fmt)
    lines.append("             enq_i.byte_count, enq_i.owner.ctx_slot, enq_i.fence);")
    lines.append("  end")
    lines.append("")
    lines.append("  // Real RISC-V word encodings; rd is forwarded from the request so both")
    lines.append("  // Contexts dispatch identical instruction bits (identity rides OwnerTag).")
    lines.append("  // Scalar: LW/SW with rs1=x0, imm=0. Float: FLW/FSW with rs1=x1, imm=0")
    lines.append("  // (0x0000a287 = FLW f5,0(x1); 0x0050a027 = FSW f5,0(x1) at rd=5).")
    lines.append("  // enq_i.fence maps to FENCE iorw,iorw (non-writing, self-retiring).")
    lines.append("  wire [31:0] inst_word = enq_i.fence ? 32'h0FF0000F :")
    lines.append("    (enq_i.float_reg")
    lines.append("      ? (enq_i.write ? {7'd0, enq_i.rd, 3'b010, 5'd1, 5'd0, 7'b0100111}")
    lines.append("                      : {12'd0, 5'd1, 3'b010, enq_i.rd, 7'b0000111})")
    lines.append("      : (enq_i.write ? {7'd0, enq_i.rd, 3'b010, 5'd0, 5'd0, 7'b0100011}")
    lines.append("                      : {12'd0, 5'd0, 3'b010, enq_i.rd, 7'b0000011}));")
    lines.append("")
    lines.append("  import c031_lsu_pkg::COMPLETED, c031_lsu_pkg::FAULTED;")
    lines.append("  wire cmpl_completed = completion_valid_i && (completion_i.final_state == COMPLETED);")
    lines.append("  wire cmpl_store_valid  = cmpl_completed &&  completion_i.write;")
    lines.append("  wire cmpl_scalar_valid = cmpl_completed && !completion_i.write")
    lines.append("                                && !completion_i.float_reg;")
    lines.append("  wire cmpl_float_valid  = cmpl_completed && !completion_i.write")
    lines.append("                                &&  completion_i.float_reg;")
    lines.append("  wire cmpl_fault_valid  = completion_valid_i")
    lines.append("                             && (completion_i.final_state == FAULTED);")
    lines.append("")
    lines.append("  wire enq_mark_scalar = enq_ok && !enq_i.fence && !enq_i.write")
    lines.append("                        && !enq_i.float_reg && (enq_i.rd != 5'd0);")
    lines.append("  wire enq_mark_float  = enq_ok && !enq_i.fence && !enq_i.write &&  enq_i.float_reg;")
    lines.append("")
    lines.extend(decl_used)
    if decl_unused:
        lines.append("  /* verilator lint_off UNUSEDSIGNAL */")
        lines.extend(decl_unused)
        lines.append("  /* verilator lint_on UNUSEDSIGNAL */")
    lines.append("")
    lines.append("  %s u_rob (" % ROB)
    lines.extend(conn)
    lines.append("  );")
    lines.append("")
    empty_expr = ", ".join("w_io_perCtxEmpty_%d" % c for c in reversed(range(n)))
    lines.append("  assign empty_o = {%s};" % empty_expr)
    for c in range(n):
        lines.append(
            "  assign retired_o[%d] = %s;" % (c, out_ext(ports, "io_perCtxNRetired_%d" % c, 4))
        )
    lines.append("  assign stale_o = %s;" % out_ext(ports, "io_staleRejected", 16))
    lines.append("endmodule")

    adapter = "\\n".join(lines) + "\\n"
    with open(sys.argv[4], "w") as f:
        f.write(closure)
        if not closure.endswith("\\n"):
            f.write("\\n")
        f.write(adapter)
    n_in = sum(1 for _, direction, _ in order if direction == "input")
    n_out = len(order) - n_in
    print(
        "c031_rob_gen: contexts=%d ports=%d (in=%d out=%d) mapped_inputs=%d "
        "tied_inputs=%d used_outputs=%d unused_outputs=%d scalar_wb_lane=%d "
        "float_wb_lane=%d"
        % (
            n,
            len(order),
            n_in,
            n_out,
            len(mapped),
            n_in - len(mapped),
            len(need_out),
            n_out - len(need_out),
            scalar_lane,
            float_lane,
        )
    )


if __name__ == "__main__":
    main()
'''

# NATIVE_STEPS: executed by the common harness after the toolchain is ready.
NATIVE_STEPS = [
    # Full-ROB emission: same pinned chisel_binary + firtool targets the C030
    # fetch slice proved, but with --enableVerification=True so the Retirement
    # behind RetirementPartitionBank is the FULL buffer (mini=false), and
    # --enableFloat=True so the scalar+FP owned writeback ports are materialized.
    "$BZL build $COMMON "
    + EMIT_BINARY
    + " @coralnpu_hw//third_party/llvm-firtool:firtool > /out/c031-emit-build.log 2>&1",
    "EMIT=/tmp/bazel-bin/hdl/chisel/src/coralnpu/core_mini_verification_axi_cc_library_emit_verilog_binary",
    "FIRTOOL_DIR=/tmp/bazel-bin/third_party/llvm-firtool",
    "for n in 1 2; do",
    '  CHISEL_FIRTOOL_PATH="$FIRTOOL_DIR" "$EMIT" --target-dir="/out/c031-emit-c$n" '
    "--contextCount=$n --lsuDataBits=128 --moduleName=CoreMiniVerification "
    "--enableFloat=True --enableZfbfmin=True --enableVerification=True "
    '> "/out/c031-emit-c$n.log" 2>&1',
    '  grep -E "' + PARAM_KEYS + '" "/out/c031-emit-c$n/VCoreMiniVerification_parameters.h" '
    '> "/out/c031-emit-c$n.params"',
    "done",
    # Real dependency closure of the patched RetirementPartitionBank per emission.
    EXTRACT_FN,
    "extract_modules /out/c031-emit-c1/CoreMiniVerification.sv /out/c031_rpb_closure_c1.sv "
    + ROB_MODULE,
    "extract_modules /out/c031-emit-c2/CoreMiniVerification.sv /out/c031_rpb_closure_c2.sv "
    + ROB_MODULE,
    'grep -q "^module ' + ROB_MODULE + '(" /out/c031_rpb_closure_c1.sv',
    'grep -q "^module ' + ROB_MODULE + '(" /out/c031_rpb_closure_c2.sv',
    # Sealed Context counts: the real partitioned bank instantiates one full
    # RetirementBuffer partition per Context in each emission.
    'test "$(grep -cE "RetirementBuffer(_[0-9]+)? partitions_" '
    '"/out/c031-emit-c1/CoreMiniVerification.sv")" -eq 1',
    'test "$(grep -cE "RetirementBuffer(_[0-9]+)? partitions_" '
    '"/out/c031-emit-c2/CoreMiniVerification.sv")" -eq 2',
    # The generator script itself travels in the step list (the container never
    # mounts the repository's verification directory).
    "cat > /out/c031_rob_gen.py <<'C031ROBEOF'",
    *GENERATOR_PY.splitlines(),
    "C031ROBEOF",
    "python3 /out/c031_rob_gen.py 1 /out/c031-emit-c1.params /out/c031_rpb_closure_c1.sv "
    "/out/c031_rob_c1.sv",
    "python3 /out/c031_rob_gen.py 2 /out/c031-emit-c2.params /out/c031_rpb_closure_c2.sv "
    "/out/c031_rob_c2.sv",
    "test -s /out/c031_rob_c1.sv",
    "test -s /out/c031_rob_c2.sv",
]
