// l1_miss_path.sv — C030 integration of cache.sv + mshr_table.sv +
// transfer_outbridge.sv into one L1 miss path.
//
// Thin structural glue only; every state machine lives in the three modules.
// The wrapper serialises the two command-shaped clients (cache and MSHR each
// accept one command per cycle) behind a small sequencer driven by the
// testbench/LSU through the `ctl_*` ports, so the integrated bench can issue the
// real command sequences:
//
//   lookup miss -> MSHR alloc (leader/merge) -> ISSUE -> outbridge MEM_REQ ->
//   MEM_RSP -> fill beat into the MSHR entry -> waiter consume (data comes from
//   the entry-owned fill buffer) -> optional cache refill -> waiter release ->
//   terminal ack -> MSHR OP_TERMINAL settle.
//
// MSHR OP_TERMINAL is raised ONLY by the outbridge settle channel after its
// TERMINAL_ACK handshake, never by the bench shortcut.
//
// C031 additions (defaults preserve C030 standalone behaviour): MSHR_KEY_W
// parameterises the full provenance key width; m_rsp_waiter returns the merged
// waiter id; mshr_filled_o/mshr_cancelled_o wake waiters without NOT_FILLED
// polling; mshr_refill_pend_o/mshr_merged_o and the settle pulse
// (settle_valid_o/settle_slot_o/settle_gen_o, matched on slot+entry_gen so a
// reused slot cannot alias) expose the obligation state; fault_valid_o pulses
// when the outbridge fault-cancels an MSHR entry (external fault, distinct from
// user cancellation); the wire MEMORY ctx/arena/scope fields are no longer
// dropped.
`timescale 1ns/1ps
module l1_miss_path
  import c029_params_pkg::*;
#(
  parameter int unsigned WAITERS = 4,
  parameter int unsigned MSHR_KEY_W = 64
) (
  input  logic clk,
  input  logic rst_n,

  // ---- MSHR command port (bench/LSU driven) ----
  input  logic m_valid,
  output logic m_ready,
  input  logic [2:0] m_op,
  input  logic [MSHR_KEY_W-1:0] m_key,
  input  logic [31:0] m_profile_gen,
  input  logic [$clog2(MSHR_ENTRIES)-1:0] m_slot,
  input  logic [31:0] m_entry_gen,
  input  logic [$clog2(WAITERS)-1:0] m_waiter,
  input  logic [LINE_BYTES*8-1:0] m_data,
  input  logic [LINE_BYTES-1:0] m_mask,
  input  logic m_last,
  output logic m_rsp_valid,
  input  logic m_rsp_ready,
  output logic [3:0] m_rsp_code,
  output logic [$clog2(MSHR_ENTRIES)-1:0] m_rsp_slot,
  output logic [31:0] m_rsp_entry_gen,
  output logic [LINE_BYTES*8-1:0] m_rsp_data,
  output logic [LINE_BYTES-1:0] m_rsp_mask,
  output logic [$clog2(WAITERS)-1:0] m_rsp_waiter,

  // ---- cache command port ----
  input  logic c_valid,
  output logic c_ready,
  input  logic [2:0] c_op,
  input  logic [63:0] c_alloc_id,
  input  logic [31:0] c_alloc_gen,
  input  logic [31:0] c_line_offset,
  input  logic [31:0] c_profile_gen,
  input  logic [LINE_BYTES*8-1:0] c_data,
  input  logic [LINE_BYTES-1:0] c_validity,
  input  logic c_dirty,
  input  logic c_maint_all,
  input  logic c_maint_clean,
  input  logic c_maint_invalidate,
  input  logic [31:0] c_range_off,
  input  logic [31:0] c_range_len,
  input  logic [31:0] c_clean_id,
  input  logic c_clean_ok,
  input  logic [$clog2(PROFILE_COUNT)-1:0] c_profile_idx,
  input  logic c_write_back,
  output logic c_rsp_valid,
  input  logic c_rsp_ready,
  output logic [3:0] c_rsp_code,
  output logic [LINE_BYTES*8-1:0] c_rsp_data,
  output logic [LINE_BYTES-1:0] c_rsp_validity,
  output logic clean_valid,
  input  logic clean_ready,
  output logic [31:0] clean_id,
  output logic [63:0] clean_alloc_id,
  output logic [31:0] clean_alloc_gen,
  output logic [31:0] clean_line_offset,
  output logic [LINE_BYTES*8-1:0] clean_data,

  // ---- refill descriptor handed to the outbridge on ISSUE ----
  input  logic iss_valid,
  output logic iss_ready,
  input  logic [31:0] iss_binding_id,
  input  logic [31:0] iss_binding_generation,
  input  logic [31:0] iss_binding_offset,
  input  logic [31:0] iss_ctx_slot,
  input  logic [31:0] iss_ctx_generation,
  input  logic [31:0] iss_arena_alloc_gen,
  input  logic [31:0] iss_scope_epoch,
  input  logic [$clog2(MSHR_ENTRIES)-1:0] iss_mshr_slot,
  input  logic [31:0] iss_mshr_gen,

  input  logic cancel_valid,
  input  logic [$clog2(MSHR_ENTRIES)-1:0] cancel_mshr_slot,
  input  logic [31:0] cancel_mshr_gen,

  // ---- wire MEMORY channel ----
  output logic mem_req_valid,
  input  logic mem_req_ready,
  output logic [31:0] mem_req_txn_id,
  output logic [31:0] mem_req_slice_id,
  output logic [31:0] mem_req_slice_count,
  output logic [31:0] mem_req_binding_id,
  output logic [31:0] mem_req_binding_generation,
  output logic [31:0] mem_req_binding_offset,
  output logic [31:0] mem_req_length,
  output logic [31:0] mem_req_ctx_slot_o,
  output logic [31:0] mem_req_ctx_generation_o,
  output logic [31:0] mem_req_arena_alloc_gen_o,
  output logic [31:0] mem_req_scope_epoch_o,
  input  logic mem_rsp_valid,
  output logic mem_rsp_ready,
  input  logic [31:0] mem_rsp_txn_id,
  input  logic [31:0] mem_rsp_slice_id,
  input  logic [31:0] mem_rsp_binding_generation,
  input  logic [31:0] mem_rsp_status,
  input  logic [LINE_BYTES*8-1:0] mem_rsp_data,
  output logic term_valid,
  input  logic term_ready,
  output logic [31:0] term_txn_id,
  output logic [31:0] term_slice_id,
  output logic [31:0] term_slice_count,
  output logic [1:0] term_final_state,

  // ---- observability ----
  output logic [$clog2(MSHR_ENTRIES+1)-1:0] mshr_active_o,
  output logic [MSHR_ENTRIES-1:0] mshr_busy_o,
  output logic [MSHR_ENTRIES-1:0] mshr_filled_o,
  output logic [MSHR_ENTRIES-1:0] mshr_cancelled_o,
  output logic [MSHR_ENTRIES-1:0] mshr_refill_pend_o,
  output logic [63:0] mshr_merged_o,
  output logic settle_valid_o,
  output logic [$clog2(MSHR_ENTRIES)-1:0] settle_slot_o,
  output logic [31:0] settle_gen_o,
  output logic fault_valid_o,
  output logic [$clog2(MSHR_ENTRIES)-1:0] fault_slot_o,
  output logic [31:0] fault_gen_o,
  output logic [31:0] cache_resident_o,
  output logic [31:0] cache_dirty_o,
  output logic [31:0] cache_pending_cleans_o,
  output logic [31:0] cache_capacity_o,
  output logic [63:0] cache_evictions_o,
  output logic [$clog2(MSHR_ENTRIES+1)-1:0] bridge_outstanding_o,
  output logic [63:0] bridge_sent_o,
  output logic [63:0] bridge_terminals_o,
  output logic [63:0] bridge_absorbed_o,
  output logic [63:0] bridge_stale_rsp_o,
  // ---- cache master side of the l1_bank cache data port (bench/top wires it to l1_bank) ----
  input  logic [$clog2(PROFILE_COUNT)-1:0] bank_profile_i,
  output logic cache_quiescent_o,
  output logic cp_valid_o,
  output logic cp_write_o,
  output logic [(BANK_COUNT <= 1 ? 1 : $clog2(BANK_COUNT))-1:0] cp_bank_o,
  output logic [$clog2(BANK_BYTES)-1:0] cp_addr_o,
  output logic [$clog2(BANK_BYTES / DATA_BYTES + 1)-1:0] cp_beats_o,
  output logic cp_wmode_o,
  output logic cp_wdata_valid_o,
  output logic [DATA_BYTES*8-1:0] cp_wdata_o,
  output logic [DATA_BYTES-1:0] cp_wben_o,
  output logic cp_rdata_ready_o,
  input  logic cp_ready_i,
  input  logic cp_rej_i,
  input  logic cp_wdata_ready_i,
  input  logic cp_rdata_valid_i,
  input  logic [DATA_BYTES*8-1:0] cp_rdata_i,
  input  logic [DATA_BYTES-1:0] cp_rben_i,
  input  logic cp_done_i,
  output logic [31:0] cache_op_cycles_o,
  output logic invariant_error_o
);
  localparam logic [2:0] OP_TERMINAL = 3'd7;
  localparam logic [2:0] MOP_CANCEL = 3'd5;

  // The bridge owns FILL / CANCEL / TERMINAL (fault) commands to the MSHR and
  // wins the single MSHR command port over the bench/LSU command. Responses to
  // bridge-originated commands are consumed internally and never shown to the
  // client; a client response is held until its own rsp_ready.
  logic mc_valid, mc_ready;
  logic [2:0] mc_op;
  logic [$clog2(MSHR_ENTRIES)-1:0] mc_slot;
  logic [31:0] mc_gen;
  logic [LINE_BYTES*8-1:0] mc_data;
  logic ms_valid;
  logic mshr_rsp_valid, mshr_rsp_ready;
  logic [3:0] mshr_rsp_code;
  logic [$clog2(MSHR_ENTRIES)-1:0] mshr_rsp_slot;
  logic [31:0] mshr_rsp_gen;
  logic [LINE_BYTES*8-1:0] mshr_rsp_data;
  logic [LINE_BYTES-1:0] mshr_rsp_mask;
  logic mshr_cmd_ready;
  logic [$clog2(MSHR_ENTRIES)-1:0] ms_cmd_slot;
  logic [31:0] ms_cmd_gen;
  logic [2:0] ms_cmd_op;
  logic [$clog2(WAITERS)-1:0] ms_cmd_waiter;
  logic [LINE_BYTES*8-1:0] ms_cmd_data;
  logic [LINE_BYTES-1:0] ms_cmd_mask;
  logic ms_cmd_last;
  logic [MSHR_KEY_W-1:0] ms_cmd_key;
  logic [31:0] ms_cmd_pgen;

  always_comb begin
    ms_cmd_op = m_op; ms_cmd_slot = m_slot; ms_cmd_gen = m_entry_gen; ms_cmd_waiter = m_waiter;
    ms_cmd_data = m_data; ms_cmd_mask = m_mask; ms_cmd_last = m_last; ms_cmd_key = m_key;
    ms_cmd_pgen = m_profile_gen;
    mc_ready = 1'b0; m_ready = 1'b0;
    if (mc_valid) begin
      ms_valid = 1'b1;
      ms_cmd_op = mc_op; ms_cmd_slot = mc_slot; ms_cmd_gen = mc_gen; ms_cmd_waiter = '0;
      ms_cmd_data = mc_data; ms_cmd_mask = '1; ms_cmd_last = 1'b1;
      mc_ready = mshr_cmd_ready;
    end else begin
      ms_valid = m_valid;
      m_ready = mshr_cmd_ready;
    end
  end

  logic rsp_from_bridge;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) rsp_from_bridge <= 1'b0;
    else if (ms_valid && mshr_cmd_ready) rsp_from_bridge <= mc_valid;
  end
  assign m_rsp_valid = mshr_rsp_valid && !rsp_from_bridge;
  assign m_rsp_code = mshr_rsp_code;
  assign m_rsp_slot = mshr_rsp_slot;
  assign m_rsp_entry_gen = mshr_rsp_gen;
  assign m_rsp_data = mshr_rsp_data;
  assign m_rsp_mask = mshr_rsp_mask;
  assign mshr_rsp_ready = rsp_from_bridge ? 1'b1 : m_rsp_ready;

  // External fault notification: the outbridge cancels an MSHR entry ONLY on a
  // faulted MEM_RSP, never on user cancellation (which this wrapper forwards
  // nowhere). The pulse is the accepting edge of that MSHR CANCEL command.
  assign fault_valid_o = mc_valid && mc_ready && (mc_op == MOP_CANCEL);
  assign fault_slot_o = mc_slot;
  assign fault_gen_o = mc_gen;

  logic mshr_invariant_error, cache_invariant_error, bridge_invariant_error;
  assign invariant_error_o = mshr_invariant_error | cache_invariant_error | bridge_invariant_error;

  mshr_table #(.ENTRIES(MSHR_ENTRIES), .LINE_B(LINE_BYTES), .WAITERS(WAITERS),
               .KEY_W(MSHR_KEY_W)) u_mshr (
    .clk, .rst_n,
    .cmd_valid(ms_valid), .cmd_ready(mshr_cmd_ready), .cmd_op(ms_cmd_op), .cmd_key(ms_cmd_key),
    .cmd_profile_gen(ms_cmd_pgen), .cmd_slot(ms_cmd_slot), .cmd_entry_gen(ms_cmd_gen),
    .cmd_waiter(ms_cmd_waiter), .cmd_data(ms_cmd_data), .cmd_mask(ms_cmd_mask), .cmd_last(ms_cmd_last),
    .rsp_valid(mshr_rsp_valid), .rsp_ready(mshr_rsp_ready), .rsp_code(mshr_rsp_code),
    .rsp_slot(mshr_rsp_slot), .rsp_entry_gen(mshr_rsp_gen), .rsp_waiter(m_rsp_waiter),
    .rsp_data(mshr_rsp_data), .rsp_mask(mshr_rsp_mask),
    .active_o(mshr_active_o), .merged_o(mshr_merged_o), .stalls_o(), .stale_rejects_o(),
    .profile_gen_o(),
    .entry_busy_o(mshr_busy_o),
    .entry_filled_o(mshr_filled_o), .entry_cancelled_o(mshr_cancelled_o),
    .refill_pend_o(mshr_refill_pend_o),
    .settle_valid_o(settle_valid_o), .settle_slot_o(settle_slot_o), .settle_gen_o(settle_gen_o),
    .invariant_error_o(mshr_invariant_error)
  );

  cache #(.LINE_B(LINE_BYTES)) u_cache (
    .clk, .rst_n,
    .cmd_valid(c_valid), .cmd_ready(c_ready), .cmd_op(c_op), .cmd_alloc_id(c_alloc_id),
    .cmd_alloc_gen(c_alloc_gen), .cmd_line_offset(c_line_offset), .cmd_profile_gen(c_profile_gen),
    .cmd_data(c_data), .cmd_validity(c_validity), .cmd_dirty(c_dirty),
    .cmd_maint_all(c_maint_all), .cmd_maint_clean(c_maint_clean),
    .cmd_maint_invalidate(c_maint_invalidate), .cmd_range_off(c_range_off),
    .cmd_range_len(c_range_len), .cmd_clean_id(c_clean_id), .cmd_clean_ok(c_clean_ok),
    .cmd_profile_idx(c_profile_idx), .cmd_write_back(c_write_back),
    .rsp_valid(c_rsp_valid), .rsp_ready(c_rsp_ready), .rsp_code(c_rsp_code),
    .rsp_data(c_rsp_data), .rsp_validity(c_rsp_validity),
    .clean_valid, .clean_ready, .clean_id, .clean_alloc_id, .clean_alloc_gen,
    .clean_line_offset, .clean_data, .clean_profile_gen(),
    .profile_gen_o(), .capacity_lines_o(cache_capacity_o), .write_back_o(),
    .resident_lines_o(cache_resident_o), .dirty_lines_o(cache_dirty_o),
    .pending_cleans_o(cache_pending_cleans_o), .hits_o(), .misses_o(), .refills_o(),
    .evictions_o(cache_evictions_o), .stale_rejects_o(), .op_cycles_o(cache_op_cycles_o),
    .maint_busy_o(),
    .bank_profile_i, .cache_quiescent_o,
    .cp_valid_o, .cp_write_o, .cp_bank_o, .cp_addr_o, .cp_beats_o, .cp_wmode_o,
    .cp_wdata_valid_o, .cp_wdata_o, .cp_wben_o, .cp_rdata_ready_o,
    .cp_ready_i, .cp_rej_i, .cp_wdata_ready_i, .cp_rdata_valid_i, .cp_rdata_i, .cp_rben_i,
    .cp_done_i,
    .invariant_error_o(cache_invariant_error)
  );

  transfer_outbridge #(.RECORDS(MSHR_ENTRIES), .LINE_B(LINE_BYTES)) u_bridge (
    .clk, .rst_n,
    .req_valid(iss_valid), .req_ready(iss_ready), .req_mshr_slot(iss_mshr_slot),
    .req_mshr_gen(iss_mshr_gen), .req_binding_id(iss_binding_id),
    .req_binding_generation(iss_binding_generation), .req_binding_offset(iss_binding_offset),
    .req_ctx_slot(iss_ctx_slot), .req_ctx_generation(iss_ctx_generation),
    .req_arena_alloc_gen(iss_arena_alloc_gen), .req_scope_epoch(iss_scope_epoch),
    .mem_req_valid, .mem_req_ready, .mem_req_txn_id, .mem_req_slice_id, .mem_req_slice_count,
    .mem_req_binding_id, .mem_req_binding_generation, .mem_req_binding_offset,
    .mem_req_ctx_slot(mem_req_ctx_slot_o), .mem_req_ctx_generation(mem_req_ctx_generation_o),
    .mem_req_arena_alloc_gen(mem_req_arena_alloc_gen_o), .mem_req_scope_epoch(mem_req_scope_epoch_o),
    .mem_req_length,
    .mem_rsp_valid, .mem_rsp_ready, .mem_rsp_txn_id, .mem_rsp_slice_id,
    .mem_rsp_binding_generation, .mem_rsp_status, .mem_rsp_data,
    .cancel_valid, .cancel_mshr_slot, .cancel_mshr_gen,
    .mc_valid, .mc_ready, .mc_op, .mc_slot, .mc_gen, .mc_data,
    .term_valid, .term_ready, .term_txn_id, .term_slice_id, .term_slice_count, .term_final_state,
    .outstanding_o(bridge_outstanding_o), .sent_o(bridge_sent_o), .rsp_o(),
    .terminals_o(bridge_terminals_o), .absorbed_o(bridge_absorbed_o),
    .stale_rsp_o(bridge_stale_rsp_o), .invariant_error_o(bridge_invariant_error)
  );
endmodule
