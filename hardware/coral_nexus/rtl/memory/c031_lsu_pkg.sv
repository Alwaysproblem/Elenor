// C031 private RTL transport types. Geometry/capacities are sealed-input
// consumers in c029_params_pkg; these structs are not a new wire/silicon ABI.
`timescale 1ns/1ps
package c031_lsu_pkg;
  import c029_params_pkg::*;
  localparam int CTX_W = 32;
  localparam int RECORD_W = $clog2(LSU_CREDIT_CAP);
  localparam int BANK_W = $clog2(BANK_COUNT);
  localparam int ADDR_W = $clog2(BANK_BYTES);
  localparam int BEAT_W = $clog2(BANK_BYTES / DATA_BYTES + 1);
  localparam int TOKEN_W = $clog2(MSHR_ENTRIES);
  localparam int WAITER_W = $clog2(LSU_MSHR_WAITERS);
  localparam logic [1:0] COMPLETED = 2'd0, ISOLATED = 2'd1, FAULTED = 2'd2;
  localparam logic [3:0] ERR_NONE = 4'd0, ERR_OWNER = 4'd1, ERR_RANGE = 4'd2,
    ERR_PERMISSION = 4'd3, ERR_UNINITIALIZED = 4'd4, ERR_MEMORY = 4'd5;

  // C026 scalar identity: generation 16, sequence 32, subop 8. ctx_slot is
  // widened to an explicit transport word and range-checked before indexing.
  typedef struct packed {
    logic [31:0] ctx_slot;
    logic [15:0] ctx_generation;
    logic [31:0] owner_op_seq;
    logic [7:0] subop_id;
  } owner_t;

  // Immutable, trusted binding supplied by the Frame/descriptor owner, not
  // an arbitrary address/binding tuple carried by an untrusted scalar opcode.
  // Each window is installed only while its Context has no old obligations.
  // spm_base is a per-bank byte offset; stripes start at bank 0.
  typedef struct packed {
    logic valid, readable, writable, cached;
    logic [31:0] base, size, spm_base, stripe_bytes;
    logic [31:0] binding_id, binding_generation, binding_offset;
    logic [31:0] arena_generation, scope_epoch;
  } window_t;

  // One scalar memory instruction; byte_count is 1, 2 or 4, including
  // unaligned/cross-stripe accesses. Floating loads/stores use raw 4-byte bits.
  typedef struct packed {
    owner_t owner;
    logic [31:0] pc, address, data;
    logic [4:0] rd;
    logic [2:0] byte_count;
    logic write, unsigned_load, float_reg, fence;
  } request_t;

  typedef struct packed {
    owner_t owner;
    logic [31:0] pc, data;
    logic [4:0] rd;
    logic write, float_reg;
    logic [1:0] final_state;
    logic [3:0] error;
  } completion_t;

  // A bounded LSU record owns each read until the response handshake.
  // Reads may cross a cache line; the miss engine sequences both pieces.
  typedef struct packed {
    owner_t owner;
    logic [RECORD_W-1:0] record_id;
    logic [31:0] binding_id, binding_generation, offset;
    logic [31:0] arena_generation, scope_epoch, profile_generation;
    logic [2:0] byte_count;
  } cache_request_t;

  typedef struct packed {
    owner_t owner;
    logic [RECORD_W-1:0] record_id;
    logic [31:0] data;
    logic [1:0] final_state;
    logic [3:0] error;
  } cache_response_t;

  typedef struct packed {
    logic valid, write;
    logic [BANK_W-1:0] bank;
    logic [ADDR_W-1:0] address;
    logic [BEAT_W-1:0] beats;
    logic wmode, wvalid;
    logic [DATA_BYTES*8-1:0] wdata;
    logic [DATA_BYTES-1:0] wben;
    logic rready;
  } cache_port_request_t;

  typedef struct packed {
    logic ready, reject, wready, rvalid;
    logic [DATA_BYTES*8-1:0] rdata;
    logic [DATA_BYTES-1:0] rben;
    logic done, busy;
  } cache_port_response_t;

  typedef struct packed {
    logic [31:0] txn_id, slice_id, slice_count;
    logic [31:0] binding_id, binding_generation, binding_offset, length;
    logic [31:0] ctx_slot, ctx_generation, arena_generation, scope_epoch;
  } memory_request_t;
  typedef struct packed {
    logic [31:0] txn_id, slice_id, binding_generation, status;
    logic [LINE_BYTES*8-1:0] data;
  } memory_response_t;
  typedef struct packed {
    logic [31:0] txn_id, slice_id, slice_count;
    logic [1:0] final_state;
  } memory_terminal_t;
endpackage
