// c038_scalar_pkg.sv — C038 scalar helper issue-adapter shared types.
//
// The launch environment (install/dispatch owner, C039/C040) expands the
// sealed binding tables declared by protocol/scalar_helper_abi_v1.json
// (cn_scalar_helper_abi/v1) into the records below. The kernel only names an
// INDEX; per the ABI, ctx/arena/scope/binding provenance is supplied by the
// Frame/descriptor owner at submit time and the tables are immutable image
// state — this package defines the RTL-side record shape only. It invents no
// width, operand domain or protocol value beyond the sealed ABI.
//
// CSR addresses are the GENERATED C017 allocator output (cn_offsets.svh,
// regenerated inside the v5 seal); a handwritten numeric address here would
// be a protocol fault, exactly as on the kernel build side. The include is
// resolved through +incdir by the build/test harness, which also pins the
// generated file's hash against the seal.
`timescale 1ns/1ps
package c038_scalar_pkg;

  // Generated C017 allocator constants (sealed file, 32-bit ints over 12-bit
  // CSR literals by design of the allocator output format).
  /* verilator lint_off WIDTHEXPAND */
  `include "cn_offsets.svh"
  /* verilator lint_on WIDTHEXPAND */

  // ---- instruction-side accept result (same edge as instr_ready_o) ----
  // PASS: not a scalar-helper access (the front-end routes it elsewhere —
  //       non-Zicsr encodings and CSRs outside the four helper addresses).
  // OK: accepted; retire the instruction normally (covers accepted
  //     mfe_submit, a direct-DONE WAIT verdict and an accepted
  //     signal_phase).
  // PARK: accepted WAIT whose verdict is "park this Context's next issue";
  //       the instruction itself retires — the C040 front-end holds the
  //       Context's issue front until the W6 wake verdict (EVT-04).
  // RET: accepted task_ret (the CMD_DRAIN leg fired); the retire chain past
  //      it is C033/C035 — this unit emits no TASK_COMPLETION itself.
  // FAULT: protocol fault, instruction consumed, no owner side effect.
  localparam logic [2:0] RSP_PASS = 3'd0, RSP_OK = 3'd1, RSP_PARK = 3'd2,
                        RSP_RET = 3'd3, RSP_FAULT = 3'd4;

  // ---- ABI fault classes (protocol/scalar_helper_abi_v1.json #faults) ----
  // Logical ABI codes for observability; fault_code_o carries the C017 wire
  // fault_code enum value (c034_reset_pkg FC_*) for first_fault_latch.
  localparam logic [3:0] FAB_NONE = 4'd0, FAB_ACCESS_FORM = 4'd1,
                        FAB_OPERAND_DOMAIN = 4'd2, FAB_WAIT_REJECTED = 4'd3,
                        FAB_WAIT_FAILED = 4'd4, FAB_RETIRE_REJECTED = 4'd5;

  // One sealed launch_descriptors[i] entry, environment-expanded.
  //   dir_in=1: external->local (l2_to_frame); dir_in=0: local->external
  //   (frame_to_l2). The external leg names the installed binding by
  //   (binding_id, binding_generation, binding_offset); local_off is the byte
  //   offset into the Tile-local Frame window (the ABI's frame_offset);
  //   bytes is the move length. The matching wait_events[i] entry is carried
  //   on the sibling wait_events_i port (event_key_t, W6 identity).
  typedef struct packed {
    logic        dir_in;
    logic [31:0] binding_id;
    logic [31:0] binding_generation;
    logic [31:0] binding_offset;
    logic [31:0] local_off;
    logic [31:0] bytes;
  } scalar_desc_t;

endpackage
