// c032_mfe_pkg.sv — C032 private MFE transport types.
//
// Geometry/capacity are sealed-input consumers in c029_params_pkg; these
// structs and codes are not a new wire/silicon ABI. The error-code space is
// shared with transfer_cancel_tracker.sv by value; see CODE_* localparams
// there. Wire field widths are transport containers, not frozen silicon
// widths; terminal_state encoding follows wire.py
// [COMPLETED, ISOLATED, FAULTED].
`timescale 1ns/1ps
package c032_mfe_pkg;
  import c029_params_pkg::*;

  localparam logic [1:0] MFE_COMPLETED = 2'd0, MFE_ISOLATED = 2'd1, MFE_FAULTED = 2'd2;

  localparam logic [3:0] ERR_NONE = 4'd0, ERR_RANGE = 4'd1, ERR_PERMISSION = 4'd2,
    ERR_BINDING = 4'd3, ERR_KIND = 4'd4, ERR_MEMORY = 4'd5,
    ERR_DUPLICATE_TERMINAL = 4'd6, ERR_TERMINAL_AFTER_RELEASE = 4'd7,
    ERR_DUPLICATE_RELEASE = 4'd8, ERR_RELEASE_WITHOUT_ACK = 4'd9,
    ERR_RELEASE_EARLY = 4'd10, ERR_PROTOCOL = 4'd11, ERR_RESET = 4'd12;

  localparam logic [1:0] KIND_LOCAL = 2'd0, KIND_EXT = 2'd1;
  localparam int unsigned MFE_INFLIGHT = 4;

  // One direct-move descriptor = one transaction. local_offset is the byte
  // offset into the Tile-local window; the binding tuple addresses the
  // external leg. ctx/arena/scope provenance rides every external request.
  typedef struct packed {
    logic [31:0] txn_id;
    logic [1:0] src_kind, dst_kind;
    logic [31:0] binding_id, binding_generation, binding_offset, local_offset, bytes;
    logic [31:0] ctx_slot, ctx_generation, arena_generation, scope_epoch;
  } mfe_desc_t;

  // Immutable binding metadata installed by the Frame/descriptor owner.
  typedef struct packed {
    logic [31:0] id, generation, size;
    logic readable, writable;
  } mfe_binding_t;

  // C023 external channel field set (C031 outbridge / memory_service), plus
  // `write` (== MEM_OP_WRITE) and the payload that travels on the service
  // channel for writes.
  typedef struct packed {
    logic [31:0] txn_id, slice_id, slice_count;
    logic [31:0] binding_id, binding_generation, binding_offset, length;
    logic [31:0] ctx_slot, ctx_generation, arena_generation, scope_epoch;
    logic write;
    logic [LINE_BYTES*8-1:0] payload;
  } mfe_mem_req_t;
  typedef struct packed {
    logic [31:0] txn_id, slice_id, binding_generation, status;
    logic [LINE_BYTES*8-1:0] data;
  } mfe_mem_rsp_t;

  typedef struct packed {
    logic [31:0] txn_id, slice_id, slice_count;
    logic [1:0] final_state;
  } mfe_term_ack_t;

  typedef struct packed {
    logic [31:0] txn_id;
    logic [1:0] final_state;
    logic [3:0] error;
  } mfe_terminal_t;
endpackage
