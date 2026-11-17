// lsu_cache_path.sv — C031 cached-scalar miss owner over the real C030
// l1_miss_path (cache.sv + mshr_table.sv + transfer_outbridge.sv).
//
// One bounded record per (Context, record_id < LSU_CREDIT_CAP) runs the real
// miss flow of G2_scalar.md C031:
//
//   cache LOOKUP -> on miss MSHR ALLOC (leader/merge) keyed by the FULL
//   96-bit provenance {binding_id, binding_generation, line_offset} -> leader
//   ISSUE + real outbridge descriptor (full ctx/arena/scope provenance) ->
//   readiness-mask wake -> CONSUME (fill bytes captured per record BEFORE the
//   token RELEASE) -> leader clean cache REFILL -> RELEASE -> settle pulse
//   (external TERMINAL_ACK closed this exact (slot, entry_gen) obligation) ->
//   piece advance / result.
//
//   * Cross-line 1/2/4-byte reads run as two sequential per-record pieces with
//     the piece-0 bytes retained across pieces.
//   * Cancel marks ONLY the local waiter. The shared refill obligation is
//     retained to the real external response/terminal; the cancelled owner then
//     gets ISOLATED and peers keep COMPLETED. The bridge cancel mark is never
//     asserted from here.
//   * An external fault (bridge fault-cancel of the MSHR entry, or the terminal
//     CANCELLED consume it causes) maps to FAULTED, never ISOLATED.
//   * The drain gate is the mshr OP_TERMINAL settle pulse matched on the exact
//     (slot, entry_gen) token, so a reused slot can never satisfy an old
//     obligation (no ABA), and a record may RELEASE before the external
//     terminal closes without blocking peers.
//   * CONSUME wakes come from the mshr filled/cancelled masks, never from
//     repeated NOT_FILLED polling; capacity retries run through a bounded
//     backoff state.
//   * Configuration is accepted only when the whole path is quiescent: bank
//     switches first (outside), then this module reconfigures cache and MSHR.
//   * MSHR/cache/bridge command service is round-robin per channel, so
//     independent misses and merges progress despite one slow refill; a
//     response held under backpressure blocks only its own record.
//
// No private fake cache data, no fake timers: every byte a response carries
// came from the cache bank or a real external MEM_RSP.
`timescale 1ns/1ps
module lsu_cache_path
  import c029_params_pkg::*;
  import c031_lsu_pkg::*;
#(
  parameter int unsigned CONTEXT_COUNT = C2_CONTEXTS,
  parameter int unsigned MSHR_KEY_W = 96
) (
  input  logic clk,
  input  logic rst_n,

  input  logic [CONTEXT_COUNT-1:0] req_valid_i,
  output logic [CONTEXT_COUNT-1:0] req_ready_o,
  input  cache_request_t req_i [CONTEXT_COUNT],

  output logic [CONTEXT_COUNT-1:0] rsp_valid_o,
  input  logic [CONTEXT_COUNT-1:0] rsp_ready_i,
  output cache_response_t rsp_o [CONTEXT_COUNT],

  input  logic [CONTEXT_COUNT-1:0] cancel_i,
  input  logic [15:0] cancel_generation_i [CONTEXT_COUNT],

  input  logic config_valid_i,
  output logic config_ready_o,
  output logic config_done_o,
  output logic config_error_o,
  input  logic [$clog2(PROFILE_COUNT)-1:0] config_profile_i,
  input  logic [31:0] config_generation_i,
  input  logic [$clog2(PROFILE_COUNT)-1:0] bank_profile_i,

  output cache_port_request_t cp_o,
  input  cache_port_response_t cp_i,
  output logic cache_quiescent_o,

  output logic mem_req_valid_o,
  input  logic mem_req_ready_i,
  output memory_request_t mem_req_o,
  input  logic mem_rsp_valid_i,
  output logic mem_rsp_ready_o,
  input  memory_response_t mem_rsp_i,
  output logic term_valid_o,
  input  logic term_ready_i,
  output memory_terminal_t term_o,

  output logic [31:0] outstanding_o,
  output logic [63:0] wait_mshr_o [CONTEXT_COUNT],
  output logic [63:0] merged_o,
  output logic invariant_error_o
);
  localparam int unsigned CAP = LSU_CREDIT_CAP;
  localparam int unsigned NREC = CONTEXT_COUNT * CAP;
  localparam int unsigned IDX_W = (NREC <= 1) ? 1 : $clog2(NREC);
  // TOKEN_W / WAITER_W come from c031_lsu_pkg.

  // mshr_table / cache opcodes and response codes (C030 encodings)
  localparam logic [2:0] OP_ALLOC = 3'd0, OP_ISSUE = 3'd1, OP_CONSUME = 3'd3,
                         OP_RELEASE = 3'd4, OP_CANCEL = 3'd5, OP_RECONFIG = 3'd6;
  localparam logic [3:0] MR_OK = 4'd0, MR_LEADER = 4'd1, MR_MERGED = 4'd2,
                         MR_FULL = 4'd3, MR_WAITERS_FULL = 4'd4, MR_STALE_PROFILE = 4'd5,
                         MR_BAD_TOKEN = 4'd6, MR_BAD_STATE = 4'd7, MR_NOT_FILLED = 4'd8,
                         MR_CANCELLED = 4'd9, MR_GEN_EXHAUSTED = 4'd12;
  localparam logic [2:0] C_LOOKUP = 3'd0, C_REFILL = 3'd1, C_RECONFIG = 3'd5;
  localparam logic [3:0] CR_OK = 4'd0, CR_HIT = 4'd1, CR_MISS = 4'd2, CR_DISABLED = 4'd3;

  typedef enum logic [3:0] {
    R_IDLE, R_LOOKUP, R_BACKOFF, R_ALLOC, R_ISSUE, R_ABORT, R_BRIDGE, R_WAIT,
    R_CONSUME, R_REFILL, R_RELEASE, R_DRAIN, R_RSP
  } rec_state_t;

  typedef enum logic [1:0] { CFG_IDLE, CFG_CACHE, CFG_MSHR } cfg_state_t;

  initial begin
    assert (MSHR_KEY_W >= 96)
      else $fatal(1, "C031_ASSERT lsu_cache_path provenance key truncated to %0d bits", MSHR_KEY_W);
    assert (LSU_MSHR_WAITERS >= NREC)
      else $fatal(1, "C031_ASSERT lsu_cache_path waiter budget below the record pool");
  end

  // ---------------- per-record state ----------------
  rec_state_t st [NREC];
  owner_t     r_owner [NREC];
  logic [RECORD_W-1:0] r_rid [NREC];
  logic [31:0] r_off [NREC];          // binding-relative offset of byte 0
  logic [31:0] r_bid [NREC], r_bgen [NREC], r_arena [NREC], r_scope [NREC], r_pgen [NREC];
  logic [2:0]  r_bc [NREC];
  logic [5:0]  r_off0 [NREC];         // piece-0 byte offset inside line 0
  logic [2:0]  r_n0 [NREC], r_n1 [NREC];
  logic        r_has_p1 [NREC], r_piece [NREC];
  logic [31:0] r_p0d [NREC], r_p1d [NREC];
  logic [TOKEN_W-1:0] r_slot [NREC];
  logic [31:0] r_mgen [NREC];
  logic [WAITER_W-1:0] r_waiter [NREC];
  logic r_leader [NREC], r_cache_on [NREC], r_faulted [NREC], r_cancelled [NREC],
        r_settled [NREC], r_bad [NREC];
  logic [LINE_BYTES*8-1:0] r_line [NREC];   // entry-owned fill bytes (leader refill)
  logic [LINE_BYTES-1:0]   r_lmask [NREC];
  logic [4:0] r_backoff [NREC];

  function automatic logic [31:0] rec_line_off(input int unsigned r);
    return (r_off[r] & ~32'(LINE_BYTES - 1)) + (r_piece[r] ? 32'(LINE_BYTES) : 32'd0);
  endfunction
  function automatic logic [MSHR_KEY_W-1:0] rec_key(input int unsigned r);
    return {r_bid[r], r_bgen[r], rec_line_off(r)};
  endfunction
  function automatic logic [31:0] rec_data(input int unsigned r);
    logic [31:0] shifted;
    shifted = 32'd0;
    if (r_has_p1[r]) shifted = r_p1d[r] << (8 * r_n0[r]);
    return r_p0d[r] | shifted;
  endfunction
  function automatic logic [1:0] rec_final(input int unsigned r);
    return r_faulted[r] ? FAULTED : (r_cancelled[r] ? ISOLATED : COMPLETED);
  endfunction
  function automatic logic [3:0] rec_error(input int unsigned r);
    return (r_faulted[r] || r_bad[r]) ? ERR_MEMORY : ERR_NONE;
  endfunction
  // true while the record holds, or is blocked acquiring, an MSHR token
  function automatic logic rec_waits_mshr(input rec_state_t s);
    return s inside {R_BACKOFF, R_ALLOC, R_ISSUE, R_ABORT, R_BRIDGE, R_WAIT,
                     R_CONSUME, R_REFILL, R_RELEASE, R_DRAIN};
  endfunction

  // ---------------- sequential core declarations ----------------
  logic m_inflight, c_inflight, m_flight_cfg, c_flight_cfg;
  logic [2:0] m_flight_op, c_flight_op;
  logic [IDX_W-1:0] m_owner, c_owner;
  cfg_state_t cfg_st;
  logic [$clog2(PROFILE_COUNT)-1:0] cfg_prof;
  logic [31:0] cfg_gen, cur_profile_gen;
  logic inv;

  // ---------------- channel wiring to l1_miss_path ----------------
  logic m_valid, m_ready;
  logic [2:0] m_op;
  logic [MSHR_KEY_W-1:0] m_key;
  logic [31:0] m_profile_gen, m_entry_gen;
  logic [TOKEN_W-1:0] m_slot;
  logic [WAITER_W-1:0] m_waiter;
  logic m_rsp_valid;
  logic [3:0] m_rsp_code;
  logic [TOKEN_W-1:0] m_rsp_slot;
  logic [31:0] m_rsp_entry_gen;
  logic [WAITER_W-1:0] m_rsp_waiter;
  logic [LINE_BYTES*8-1:0] m_rsp_data;
  logic [LINE_BYTES-1:0] m_rsp_mask;

  logic c_valid, c_ready;
  logic [2:0] c_op;
  logic [63:0] c_alloc_id;
  logic [31:0] c_alloc_gen, c_line_offset, c_profile_gen;
  logic [LINE_BYTES*8-1:0] c_data;
  logic [LINE_BYTES-1:0] c_validity;
  logic [$clog2(PROFILE_COUNT)-1:0] c_profile_idx;
  logic c_rsp_valid;
  logic [3:0] c_rsp_code;
  logic [LINE_BYTES*8-1:0] c_rsp_data;
  logic [LINE_BYTES-1:0] c_rsp_validity;

  logic iss_valid, iss_ready;
  logic [31:0] iss_binding_id, iss_binding_generation, iss_binding_offset;
  logic [31:0] iss_ctx_slot, iss_ctx_generation, iss_arena_alloc_gen, iss_scope_epoch;
  logic [TOKEN_W-1:0] iss_mshr_slot;
  logic [31:0] iss_mshr_gen;

  logic mp_mem_req_valid, mp_mem_req_ready, mp_mem_rsp_valid, mp_mem_rsp_ready;
  logic [31:0] mp_mem_req_txn_id, mp_mem_req_slice_id, mp_mem_req_slice_count,
               mp_mem_req_binding_id, mp_mem_req_binding_generation, mp_mem_req_binding_offset,
               mp_mem_req_length;
  logic [31:0] mp_mem_rsp_txn_id, mp_mem_rsp_slice_id, mp_mem_rsp_binding_generation, mp_mem_rsp_status;
  logic [LINE_BYTES*8-1:0] mp_mem_rsp_data;
  logic mp_term_valid, mp_term_ready;
  logic [31:0] mp_term_txn_id, mp_term_slice_id, mp_term_slice_count;
  logic [1:0] mp_term_final_state;

  logic [$clog2(MSHR_ENTRIES+1)-1:0] mshr_active_o, bridge_outstanding_o;
  logic [MSHR_ENTRIES-1:0] mshr_busy_o, mshr_filled_o, mshr_cancelled_o, mshr_refill_pend_o;
  logic fault_valid_o, settle_valid_o;
  logic [TOKEN_W-1:0] fault_slot_o, settle_slot_o;
  logic [31:0] fault_gen_o, settle_gen_o;
  logic [31:0] mp_req_ctx_slot, mp_req_ctx_generation, mp_req_arena, mp_req_scope;
  logic [63:0] mshr_merged_o;
  logic mp_cache_quiescent, mp_invariant_error;
  logic [31:0] mp_cache_op_cycles;
  logic [63:0] mp_bridge_sent_o, mp_bridge_terminals_o, mp_bridge_absorbed_o, mp_bridge_stale_rsp_o;
  logic [31:0] mp_cache_resident_o, mp_cache_dirty_o, mp_cache_pending_cleans_o, mp_cache_capacity_o;
  logic [63:0] mp_cache_evictions_o;
  logic clean_valid;

  l1_miss_path #(.WAITERS(LSU_MSHR_WAITERS), .MSHR_KEY_W(MSHR_KEY_W)) mp (
    .clk, .rst_n,
    .m_valid, .m_ready, .m_op, .m_key, .m_profile_gen, .m_slot, .m_entry_gen, .m_waiter,
    .m_data('0), .m_mask('1), .m_last(1'b1),
    .m_rsp_valid, .m_rsp_ready(1'b1), .m_rsp_code, .m_rsp_slot, .m_rsp_entry_gen,
    .m_rsp_waiter, .m_rsp_data, .m_rsp_mask,
    .c_valid, .c_ready, .c_op, .c_alloc_id, .c_alloc_gen, .c_line_offset, .c_profile_gen,
    .c_data, .c_validity, .c_dirty(1'b0), .c_maint_all(1'b0), .c_maint_clean(1'b0),
    .c_maint_invalidate(1'b0), .c_range_off('0), .c_range_len('0), .c_clean_id('0),
    .c_clean_ok(1'b0), .c_profile_idx, .c_write_back(1'b0),
    .c_rsp_valid, .c_rsp_ready(1'b1), .c_rsp_code, .c_rsp_data, .c_rsp_validity,
    .clean_valid, .clean_ready(1'b1), .clean_id(), .clean_alloc_id(), .clean_alloc_gen(),
    .clean_line_offset(), .clean_data(),
    .iss_valid, .iss_ready, .iss_binding_id, .iss_binding_generation, .iss_binding_offset,
    .iss_ctx_slot, .iss_ctx_generation, .iss_arena_alloc_gen, .iss_scope_epoch,
    .iss_mshr_slot, .iss_mshr_gen,
    .cancel_valid(1'b0), .cancel_mshr_slot('0), .cancel_mshr_gen('0),
    .mem_req_valid(mp_mem_req_valid), .mem_req_ready(mp_mem_req_ready),
    .mem_req_txn_id(mp_mem_req_txn_id), .mem_req_slice_id(mp_mem_req_slice_id),
    .mem_req_slice_count(mp_mem_req_slice_count), .mem_req_binding_id(mp_mem_req_binding_id),
    .mem_req_binding_generation(mp_mem_req_binding_generation),
    .mem_req_binding_offset(mp_mem_req_binding_offset), .mem_req_length(mp_mem_req_length),
    .mem_req_ctx_slot_o(mp_req_ctx_slot), .mem_req_ctx_generation_o(mp_req_ctx_generation),
    .mem_req_arena_alloc_gen_o(mp_req_arena), .mem_req_scope_epoch_o(mp_req_scope),
    .mem_rsp_valid(mp_mem_rsp_valid), .mem_rsp_ready(mp_mem_rsp_ready),
    .mem_rsp_txn_id(mp_mem_rsp_txn_id), .mem_rsp_slice_id(mp_mem_rsp_slice_id),
    .mem_rsp_binding_generation(mp_mem_rsp_binding_generation),
    .mem_rsp_status(mp_mem_rsp_status), .mem_rsp_data(mp_mem_rsp_data),
    .term_valid(mp_term_valid), .term_ready(mp_term_ready),
    .term_txn_id(mp_term_txn_id), .term_slice_id(mp_term_slice_id),
    .term_slice_count(mp_term_slice_count), .term_final_state(mp_term_final_state),
    .mshr_active_o, .mshr_busy_o, .mshr_merged_o, .mshr_filled_o, .mshr_cancelled_o,
    .mshr_refill_pend_o, .settle_valid_o, .settle_slot_o, .settle_gen_o,
    .fault_valid_o, .fault_slot_o, .fault_gen_o,
    .cache_resident_o(mp_cache_resident_o), .cache_dirty_o(mp_cache_dirty_o),
    .cache_pending_cleans_o(mp_cache_pending_cleans_o), .cache_capacity_o(mp_cache_capacity_o),
    .cache_evictions_o(mp_cache_evictions_o), .bridge_outstanding_o,
    .bridge_sent_o(mp_bridge_sent_o), .bridge_terminals_o(mp_bridge_terminals_o),
    .bridge_absorbed_o(mp_bridge_absorbed_o), .bridge_stale_rsp_o(mp_bridge_stale_rsp_o),
    .bank_profile_i, .cache_quiescent_o(mp_cache_quiescent),
    .cp_valid_o(cp_o.valid), .cp_write_o(cp_o.write), .cp_bank_o(cp_o.bank),
    .cp_addr_o(cp_o.address), .cp_beats_o(cp_o.beats), .cp_wmode_o(cp_o.wmode),
    .cp_wdata_valid_o(cp_o.wvalid), .cp_wdata_o(cp_o.wdata), .cp_wben_o(cp_o.wben),
    .cp_rdata_ready_o(cp_o.rready),
    .cp_ready_i(cp_i.ready), .cp_rej_i(cp_i.reject), .cp_wdata_ready_i(cp_i.wready),
    .cp_rdata_valid_i(cp_i.rvalid), .cp_rdata_i(cp_i.rdata), .cp_rben_i(cp_i.rben),
    .cp_done_i(cp_i.done),
    .cache_op_cycles_o(mp_cache_op_cycles), .invariant_error_o(mp_invariant_error)
  );

  // Wire MEMORY channel passthrough (full provenance).
  assign mem_req_valid_o = mp_mem_req_valid;
  assign mp_mem_req_ready = mem_req_ready_i;
  assign mem_req_o.txn_id = mp_mem_req_txn_id;
  assign mem_req_o.slice_id = mp_mem_req_slice_id;
  assign mem_req_o.slice_count = mp_mem_req_slice_count;
  assign mem_req_o.binding_id = mp_mem_req_binding_id;
  assign mem_req_o.binding_generation = mp_mem_req_binding_generation;
  assign mem_req_o.binding_offset = mp_mem_req_binding_offset;
  assign mem_req_o.length = mp_mem_req_length;
  assign mem_req_o.ctx_slot = mp_req_ctx_slot;
  assign mem_req_o.ctx_generation = mp_req_ctx_generation;
  assign mem_req_o.arena_generation = mp_req_arena;
  assign mem_req_o.scope_epoch = mp_req_scope;
  assign mp_mem_rsp_valid = mem_rsp_valid_i;
  assign mem_rsp_ready_o = mp_mem_rsp_ready;
  assign mp_mem_rsp_txn_id = mem_rsp_i.txn_id;
  assign mp_mem_rsp_slice_id = mem_rsp_i.slice_id;
  assign mp_mem_rsp_binding_generation = mem_rsp_i.binding_generation;
  assign mp_mem_rsp_status = mem_rsp_i.status;
  assign mp_mem_rsp_data = mem_rsp_i.data;
  assign term_valid_o = mp_term_valid;
  assign mp_term_ready = term_ready_i;
  assign term_o.txn_id = mp_term_txn_id;
  assign term_o.slice_id = mp_term_slice_id;
  assign term_o.slice_count = mp_term_slice_count;
  assign term_o.final_state = mp_term_final_state;
  assign merged_o = mshr_merged_o;

  // ---------------- outstanding / wait gauges / quiescence ----------------
  always_comb begin
    int unsigned n;
    n = 0;
    for (int unsigned i = 0; i < NREC; i++)
      if (st[i] != R_IDLE) n++;
    outstanding_o = 32'(n);
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      int unsigned w;
      w = 0;
      for (int unsigned i = 0; i < NREC; i++)
        if (rec_waits_mshr(st[i]) && (i / CAP) == unsigned'(c)) w++;
      wait_mshr_o[c] = 64'(w);
    end
  end

  logic [CONTEXT_COUNT-1:0] rsp_granted;
  logic [IDX_W-1:0] rsp_grant [CONTEXT_COUNT];
  logic [IDX_W-1:0] rsp_ptr [CONTEXT_COUNT];

  assign cache_quiescent_o = (outstanding_o == 0) && !m_inflight && !c_inflight &&
                             (mshr_active_o == 0) && (bridge_outstanding_o == 0) &&
                             mp_cache_quiescent && (cfg_st == CFG_IDLE) && !(|rsp_granted);
  assign config_ready_o = cache_quiescent_o;
  assign invariant_error_o = mp_invariant_error | inv;

  // ---------------- ingress ----------------
  always_comb begin
    // Ready is a level on the per-Context credit: a free slot for this ctx, so
    // the provider may offer any credit it holds. It must NOT read req_i
    // contents: doing so creates a comb loop through the mslot credit table.
    // The accepted (ctx, record_id) is asserted occupied-free at accept time.
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      req_ready_o[c] = 1'b0;
      for (int k = 0; k < CAP; k++)
        if (st[c * CAP + k] == R_IDLE) req_ready_o[c] = 1'b1;
    end
  end

  // ---------------- channel arbiters (round robin per channel) ----------------
  logic m_have, m_cfg;
  logic [IDX_W-1:0] m_pick;
  logic [IDX_W-1:0] pm;
  logic c_have, c_cfg;
  logic [IDX_W-1:0] c_pick;
  logic [IDX_W-1:0] pc;
  logic i_have;
  logic [IDX_W-1:0] i_pick;
  logic [IDX_W-1:0] pi;

  always_comb begin
    int unsigned idx;
    idx = 0;
    m_have = 1'b0; m_cfg = 1'b0; m_pick = '0;
    if (cfg_st == CFG_MSHR) begin
      m_have = 1'b1; m_cfg = 1'b1;
    end else begin
      for (int unsigned k = 0; k < NREC; k++) begin
        idx = (32'(pm) + k) % NREC;
        if (!m_have && (st[idx] inside {R_ALLOC, R_ISSUE, R_CONSUME, R_RELEASE, R_ABORT})) begin
          m_have = 1'b1; m_pick = IDX_W'(idx);
        end
      end
    end
    m_valid = m_have;
    m_op = OP_ALLOC; m_key = '0; m_profile_gen = '0; m_slot = '0; m_entry_gen = '0;
    m_waiter = '0;
    if (m_cfg) begin
      m_op = OP_RECONFIG;
      m_profile_gen = cfg_gen;
    end else if (m_have) begin
      case (st[m_pick])
        R_ALLOC: begin
          m_op = OP_ALLOC;
          m_key = rec_key(32'(m_pick));
          m_profile_gen = r_pgen[m_pick];
        end
        R_ISSUE: begin
          m_op = OP_ISSUE;
          m_slot = r_slot[m_pick];
          m_entry_gen = r_mgen[m_pick];
        end
        R_ABORT: begin
          m_op = OP_CANCEL;
          m_slot = r_slot[m_pick];
          m_entry_gen = r_mgen[m_pick];
        end
        R_CONSUME: begin
          m_op = OP_CONSUME;
          m_slot = r_slot[m_pick];
          m_entry_gen = r_mgen[m_pick];
          m_waiter = r_waiter[m_pick];
        end
        R_RELEASE: begin
          m_op = OP_RELEASE;
          m_slot = r_slot[m_pick];
          m_entry_gen = r_mgen[m_pick];
          m_waiter = r_waiter[m_pick];
        end
        default: m_valid = 1'b0;
      endcase
    end
  end


  always_comb begin
    int unsigned idx;
    idx = 0;
    c_have = 1'b0; c_cfg = 1'b0; c_pick = '0;
    if (cfg_st == CFG_CACHE) begin
      c_have = 1'b1; c_cfg = 1'b1;
    end else begin
      for (int unsigned k = 0; k < NREC; k++) begin
        idx = (32'(pc) + k) % NREC;
        if (!c_have && (st[idx] inside {R_LOOKUP, R_REFILL})) begin
          c_have = 1'b1; c_pick = IDX_W'(idx);
        end
      end
    end
    c_valid = c_have;
    c_op = C_LOOKUP; c_alloc_id = '0; c_alloc_gen = '0; c_line_offset = '0;
    c_profile_gen = cur_profile_gen; c_data = '0; c_validity = '1; c_profile_idx = '0;
    if (c_cfg) begin
      c_op = C_RECONFIG;
      c_profile_gen = cfg_gen;
      c_profile_idx = cfg_prof;
    end else if (c_have) begin
      c_op = (st[c_pick] == R_REFILL) ? C_REFILL : C_LOOKUP;
      c_alloc_id = {r_bid[c_pick], r_bgen[c_pick]};
      c_alloc_gen = r_arena[c_pick];
      c_line_offset = rec_line_off(32'(c_pick));
      if (st[c_pick] == R_REFILL) begin
        c_data = r_line[c_pick];
        c_validity = r_lmask[c_pick];
      end
    end
  end

  always_comb begin
    int unsigned idx;
    idx = 0;
    i_have = 1'b0; i_pick = '0;
    for (int unsigned k = 0; k < NREC; k++) begin
      idx = (32'(pi) + k) % NREC;
      if (!i_have && (st[idx] == R_BRIDGE)) begin
        i_have = 1'b1; i_pick = IDX_W'(idx);
      end
    end
    iss_valid = i_have;
    iss_binding_id = '0; iss_binding_generation = '0; iss_binding_offset = '0;
    iss_ctx_slot = '0; iss_ctx_generation = '0; iss_arena_alloc_gen = '0; iss_scope_epoch = '0;
    iss_mshr_slot = '0; iss_mshr_gen = '0;
    if (i_have) begin
      iss_binding_id = r_bid[i_pick];
      iss_binding_generation = r_bgen[i_pick];
      iss_binding_offset = rec_line_off(32'(i_pick));
      iss_ctx_slot = r_owner[i_pick].ctx_slot;
      iss_ctx_generation = {16'd0, r_owner[i_pick].ctx_generation};
      iss_arena_alloc_gen = r_arena[i_pick];
      iss_scope_epoch = r_scope[i_pick];
      iss_mshr_slot = r_slot[i_pick];
      iss_mshr_gen = r_mgen[i_pick];
    end
  end

  // ---------------- response grants per Context ----------------
  always_comb begin
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      rsp_valid_o[c] = rsp_granted[c] && (st[rsp_grant[c]] == R_RSP);
      rsp_o[c] = '{owner: r_owner[rsp_grant[c]], record_id: r_rid[rsp_grant[c]],
                   data: rec_data(32'(rsp_grant[c])), final_state: rec_final(32'(rsp_grant[c])),
                   error: rec_error(32'(rsp_grant[c]))};
    end
  end

  // ---------------- sequential core ----------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_inflight <= 1'b0; c_inflight <= 1'b0;
      m_flight_cfg <= 1'b0; c_flight_cfg <= 1'b0;
      m_flight_op <= '0; c_flight_op <= '0;
      m_owner <= '0; c_owner <= '0;
      pm <= '0; pc <= '0; pi <= '0;
      rsp_granted <= '0;
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        rsp_grant[c] <= '0;
        rsp_ptr[c] <= '0;
      end
      cfg_st <= CFG_IDLE; cfg_prof <= '0; cfg_gen <= '0; cur_profile_gen <= '0;
      config_done_o <= 1'b0; config_error_o <= 1'b0;
      inv <= 1'b0;
      for (int unsigned i = 0; i < NREC; i++) begin
        st[i] <= R_IDLE;
        r_owner[i] <= '0; r_rid[i] <= '0; r_off[i] <= '0;
        r_bid[i] <= '0; r_bgen[i] <= '0; r_arena[i] <= '0; r_scope[i] <= '0; r_pgen[i] <= '0;
        r_bc[i] <= '0; r_off0[i] <= '0; r_n0[i] <= '0; r_n1[i] <= '0;
        r_has_p1[i] <= 1'b0; r_piece[i] <= 1'b0;
        r_p0d[i] <= '0; r_p1d[i] <= '0;
        r_slot[i] <= '0; r_mgen[i] <= '0; r_waiter[i] <= '0;
        r_leader[i] <= 1'b0; r_cache_on[i] <= 1'b0; r_faulted[i] <= 1'b0;
        r_cancelled[i] <= 1'b0; r_settled[i] <= 1'b0; r_bad[i] <= 1'b0;
        r_line[i] <= '0; r_lmask[i] <= '0; r_backoff[i] <= '0;
      end
    end else begin
      config_done_o <= 1'b0;
      config_error_o <= 1'b0;

      // ---- ingress accepts
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (req_valid_i[c] && req_ready_o[c]) begin
          automatic int unsigned r = unsigned'(c) * CAP + 32'(req_i[c].record_id);
          if (st[r] != R_IDLE) begin
            $error("C031_ASSERT lsu_cache_path ctx %0d accepts occupied record_id %0d", c, req_i[c].record_id);
            inv <= 1'b1;
          end
          r_owner[r] <= req_i[c].owner;
          r_rid[r] <= req_i[c].record_id;
          r_off[r] <= req_i[c].offset;
          r_bid[r] <= req_i[c].binding_id;
          r_bgen[r] <= req_i[c].binding_generation;
          r_arena[r] <= req_i[c].arena_generation;
          r_scope[r] <= req_i[c].scope_epoch;
          r_pgen[r] <= req_i[c].profile_generation;
          r_bc[r] <= req_i[c].byte_count;
          r_off0[r] <= 6'(req_i[c].offset % LINE_BYTES);
          r_has_p1[r] <= ((req_i[c].offset % LINE_BYTES) + 32'(req_i[c].byte_count)) > LINE_BYTES;
          r_n0[r] <= (((req_i[c].offset % LINE_BYTES) + 32'(req_i[c].byte_count)) > LINE_BYTES)
                       ? 3'(LINE_BYTES - (req_i[c].offset % LINE_BYTES))
                       : req_i[c].byte_count;
          r_n1[r] <= 3'(((req_i[c].offset % LINE_BYTES) + 32'(req_i[c].byte_count)) - LINE_BYTES);
          r_piece[r] <= 1'b0;
          r_p0d[r] <= '0; r_p1d[r] <= '0;
          r_slot[r] <= '0; r_mgen[r] <= '0; r_waiter[r] <= '0;
          r_leader[r] <= 1'b0; r_cache_on[r] <= 1'b1;
          r_faulted[r] <= 1'b0; r_cancelled[r] <= 1'b0; r_settled[r] <= 1'b0;
          // A request accepted on the same edge as its Context's cancel is
          // born cancelled: the cancellation snapshot must not miss it.
          if (cancel_i[c] && req_i[c].owner.ctx_generation == cancel_generation_i[c])
            r_cancelled[r] <= 1'b1;
          r_line[r] <= '0; r_lmask[r] <= '0; r_backoff[r] <= '0;
          if (req_i[c].byte_count inside {3'd1, 3'd2, 3'd4}) begin
            r_bad[r] <= 1'b0;
            st[r] <= R_LOOKUP;
          end else begin
            $error("C031_ASSERT lsu_cache_path byte_count %0d not 1/2/4", req_i[c].byte_count);
            r_bad[r] <= 1'b1;
            inv <= 1'b1;
            st[r] <= R_RSP;
          end
        end
      end

      // ---- local cancellation marks (never touches the shared refill)
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (cancel_i[c]) begin
          for (int k = 0; k < CAP; k++) begin
            automatic int unsigned r = unsigned'(c) * CAP + unsigned'(k);
            if (st[r] != R_IDLE && st[r] != R_RSP &&
                r_owner[r].ctx_generation == cancel_generation_i[c])
              r_cancelled[r] <= 1'b1;
          end
        end
      end

      // ---- external fault and settle events, matched on the full token
      //      (slot AND entry_gen), so a reused slot can never alias.
      if (fault_valid_o)
        for (int unsigned i = 0; i < NREC; i++)
          if (st[i] inside {R_WAIT, R_CONSUME, R_REFILL, R_RELEASE, R_DRAIN} &&
              {TOKEN_W'(r_slot[i]), r_mgen[i]} == {fault_slot_o, fault_gen_o})
            r_faulted[i] <= 1'b1;
      if (settle_valid_o)
        for (int unsigned i = 0; i < NREC; i++)
          if (st[i] inside {R_WAIT, R_CONSUME, R_REFILL, R_RELEASE, R_DRAIN} &&
              {TOKEN_W'(r_slot[i]), r_mgen[i]} == {settle_slot_o, settle_gen_o})
            r_settled[i] <= 1'b1;

      // ---- command accepts
      if (m_valid && m_ready) begin
        m_inflight <= 1'b1;
        m_flight_cfg <= m_cfg;
        m_flight_op <= m_op;
        m_owner <= m_pick;
        if (!m_cfg) pm <= IDX_W'(32'(m_pick) + 1);
      end
      if (c_valid && c_ready) begin
        c_inflight <= 1'b1;
        c_flight_cfg <= c_cfg;
        c_flight_op <= c_op;
        c_owner <= c_pick;
        if (!c_cfg) pc <= IDX_W'(32'(c_pick) + 1);
      end
      if (iss_valid && iss_ready) begin
        st[i_pick] <= R_WAIT;
        pi <= IDX_W'(32'(i_pick) + 1);
      end

      // ---- MSHR command response
      if (m_rsp_valid) begin
        if (m_flight_cfg) begin
          m_inflight <= 1'b0;
          if (m_rsp_code == MR_OK) begin
            config_done_o <= 1'b1;
            cfg_st <= CFG_IDLE;
          end else begin
            config_error_o <= 1'b1;
            cfg_st <= CFG_IDLE;
          end
        end else if (!m_inflight) begin
          $error("C031_ASSERT lsu_cache_path stray MSHR response code %0d", m_rsp_code);
          inv <= 1'b1;
        end else begin
          automatic int unsigned r = 32'(m_owner);
          m_inflight <= 1'b0;
          case (m_flight_op)
            OP_ALLOC: begin
              case (m_rsp_code)
                MR_LEADER: begin
                  r_slot[r] <= m_rsp_slot;
                  r_mgen[r] <= m_rsp_entry_gen;
                  r_waiter[r] <= '0;
                  r_leader[r] <= 1'b1;
                  st[r] <= R_ISSUE;
                end
                MR_MERGED: begin
                  r_slot[r] <= m_rsp_slot;
                  r_mgen[r] <= m_rsp_entry_gen;
                  r_waiter[r] <= m_rsp_waiter;
                  r_leader[r] <= 1'b0;
                  // A merge into an entry whose obligation ALREADY settled (mask
                  // view read in this response cycle) missed that settle pulse;
                  // an unissued entry (no fill, no cancel) still owes its pulse.
                  if (!mshr_refill_pend_o[m_rsp_slot] &&
                      (mshr_filled_o[m_rsp_slot] || mshr_cancelled_o[m_rsp_slot]))
                    r_settled[r] <= 1'b1;
                  st[r] <= R_WAIT;
                end
                MR_FULL, MR_WAITERS_FULL: begin
                  r_backoff[r] <= 5'd16;
                  st[r] <= R_BACKOFF;
                end
                default: begin
                  $error("C031_ASSERT lsu_cache_path alloc failed code %0d", m_rsp_code);
                  r_bad[r] <= 1'b1;
                  inv <= 1'b1;
                  st[r] <= R_RSP;
                end
              endcase
            end
            OP_ISSUE: begin
              if (m_rsp_code == MR_OK) st[r] <= R_BRIDGE;
              else begin
                $error("C031_ASSERT lsu_cache_path issue failed code %0d", m_rsp_code);
                inv <= 1'b1;
                st[r] <= R_ABORT;
              end
            end
            OP_CANCEL: begin
              // Defensive abort of a token the MSHR refused to issue: the
              // entry now carries a terminal cancel, so consume/release close it.
              if (m_rsp_code != MR_OK) begin
                $error("C031_ASSERT lsu_cache_path abort cancel code %0d", m_rsp_code);
                inv <= 1'b1;
              end
              r_settled[r] <= 1'b1;
              st[r] <= R_WAIT;
            end
            OP_CONSUME: begin
              if (m_rsp_code == MR_OK) begin
                if (r_piece[r] == 1'b0) begin
                  for (int b = 0; b < 4; b++)
                    if (b < int'(r_n0[r]))
                      r_p0d[r][8*b +: 8] <= m_rsp_data[8*(int'(r_off0[r]) + b) +: 8];
                end else begin
                  for (int b = 0; b < 4; b++)
                    if (b < int'(r_n1[r]))
                      r_p1d[r][8*b +: 8] <= m_rsp_data[8*b +: 8];
                end
                if (r_leader[r] && r_cache_on[r]) begin
                  r_line[r] <= m_rsp_data;
                  r_lmask[r] <= m_rsp_mask;
                  st[r] <= R_REFILL;
                end else begin
                  st[r] <= R_RELEASE;
                end
              end else if (m_rsp_code == MR_CANCELLED) begin
                // entry cancelled by the bridge => external fault, no data
                r_faulted[r] <= 1'b1;
                st[r] <= R_RELEASE;
              end else if (m_rsp_code == MR_NOT_FILLED || m_rsp_code == MR_BAD_TOKEN) begin
                st[r] <= R_WAIT; // readiness mask re-arms; never a busy poll
              end else begin
                $error("C031_ASSERT lsu_cache_path consume failed code %0d", m_rsp_code);
                inv <= 1'b1;
                st[r] <= R_WAIT;
              end
            end
            OP_RELEASE: begin
              if (m_rsp_code != MR_OK) begin
                $error("C031_ASSERT lsu_cache_path release failed code %0d", m_rsp_code);
                inv <= 1'b1;
              end
              st[r] <= R_DRAIN;
            end
            default: begin
              $error("C031_ASSERT lsu_cache_path unexpected MSHR flight op %0d", m_flight_op);
              inv <= 1'b1;
            end
          endcase
        end
      end

      // ---- cache command response
      if (c_rsp_valid) begin
        if (c_flight_cfg) begin
          c_inflight <= 1'b0;
          if (c_rsp_code == CR_OK) begin
            cur_profile_gen <= cfg_gen;
            cfg_st <= CFG_MSHR;
          end else begin
            config_error_o <= 1'b1;
            cfg_st <= CFG_IDLE;
          end
        end else if (!c_inflight) begin
          $error("C031_ASSERT lsu_cache_path stray cache response code %0d", c_rsp_code);
          inv <= 1'b1;
        end else begin
          automatic int unsigned r = 32'(c_owner);
          c_inflight <= 1'b0;
          case (c_flight_op)
            C_LOOKUP: begin
              case (c_rsp_code)
                CR_HIT: begin
                  if (r_piece[r] == 1'b0) begin
                    for (int b = 0; b < 4; b++)
                      if (b < int'(r_n0[r]))
                        r_p0d[r][8*b +: 8] <= c_rsp_data[8*(int'(r_off0[r]) + b) +: 8];
                  end else begin
                    for (int b = 0; b < 4; b++)
                      if (b < int'(r_n1[r]))
                        r_p1d[r][8*b +: 8] <= c_rsp_data[8*b +: 8];
                  end
                  // piece complete with no external obligation
                  if (r_faulted[r] || r_cancelled[r]) st[r] <= R_RSP;
                  else if (!r_piece[r] && r_has_p1[r]) begin
                    r_piece[r] <= 1'b1;
                    st[r] <= R_LOOKUP;
                  end else st[r] <= R_RSP;
                end
                CR_MISS: st[r] <= R_ALLOC;
                CR_DISABLED: begin
                  r_cache_on[r] <= 1'b0;
                  st[r] <= R_ALLOC;
                end
                default: begin
                  $error("C031_ASSERT lsu_cache_path lookup code %0d", c_rsp_code);
                  r_bad[r] <= 1'b1;
                  inv <= 1'b1;
                  st[r] <= R_RSP;
                end
              endcase
            end
            C_REFILL: begin
              if (c_rsp_code != CR_OK && c_rsp_code != CR_DISABLED) begin
                $error("C031_ASSERT lsu_cache_path refill code %0d", c_rsp_code);
                inv <= 1'b1;
              end
              st[r] <= R_RELEASE;
            end
            default: begin
              $error("C031_ASSERT lsu_cache_path unexpected cache flight op %0d", c_flight_op);
              inv <= 1'b1;
            end
          endcase
        end
      end

      // ---- readiness-mask wake and level-driven steps
      for (int unsigned i = 0; i < NREC; i++) begin
        if (st[i] == R_WAIT &&
            (mshr_filled_o[r_slot[i]] || mshr_cancelled_o[r_slot[i]]))
          st[i] <= R_CONSUME;
        if (st[i] == R_BACKOFF) begin
          if (r_backoff[i] != 5'd0) r_backoff[i] <= r_backoff[i] - 5'd1;
          else st[i] <= R_ALLOC;
        end
        if (st[i] == R_DRAIN && r_settled[i]) begin
          // The exact external obligation drained before the result reports.
          if (r_faulted[i] || r_cancelled[i]) st[i] <= R_RSP;
          else if (!r_piece[i] && r_has_p1[i]) begin
            r_piece[i] <= 1'b1;
            st[i] <= R_LOOKUP;
          end else st[i] <= R_RSP;
        end
      end

      // ---- configuration accept (only fully quiescent)
      if (cfg_st == CFG_IDLE && config_valid_i && config_ready_o) begin
        cfg_prof <= config_profile_i;
        cfg_gen <= config_generation_i;
        cfg_st <= CFG_CACHE;
      end

      // ---- response grants: one held record per Context, round robin
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        automatic int unsigned base = unsigned'(c) * CAP;
        if (rsp_granted[c] && rsp_valid_o[c] && rsp_ready_i[c]) begin
          st[rsp_grant[c]] <= R_IDLE;
          rsp_granted[c] <= 1'b0;
          rsp_ptr[c] <= IDX_W'((32'(rsp_grant[c]) - base + 1) % CAP);
        end else if (!rsp_granted[c]) begin
          for (int k = 0; k < CAP; k++) begin
            automatic int unsigned idx = base + (32'(rsp_ptr[c]) + unsigned'(k)) % CAP;
            if (!rsp_granted[c] && st[idx] == R_RSP) begin
              rsp_granted[c] <= 1'b1;
              rsp_grant[c] <= IDX_W'(idx);
            end
          end
        end
      end
    end
  end

  // ---------------- live invariants ----------------
  always @(posedge clk) begin
    if (rst_n) begin
      for (int unsigned i = 0; i < NREC; i++) begin
        if (st[i] inside {R_WAIT, R_CONSUME, R_REFILL} && !mshr_busy_o[r_slot[i]])
          $error("C031_ASSERT lsu_cache_path record %0d holds waiter on free MSHR slot %0d",
                 i, r_slot[i]);
        // OP_TERMINAL frees the slot and pulses settle on the same edge; the
        // draining record samples that pulse one edge later, so a closed slot
        // with the record's own settle pulse in flight is still reachable.
        if (st[i] == R_DRAIN && !r_settled[i] && !mshr_refill_pend_o[r_slot[i]] &&
            !mshr_busy_o[r_slot[i]] &&
            !(settle_valid_o && {TOKEN_W'(r_slot[i]), r_mgen[i]} == {settle_slot_o, settle_gen_o}))
          $error("C031_ASSERT lsu_cache_path record %0d drains a closed slot %0d", i, r_slot[i]);
      end
      if (m_inflight && !m_flight_cfg &&
          !(st[m_owner] inside {R_ALLOC, R_ISSUE, R_CONSUME, R_RELEASE, R_ABORT}))
        $error("C031_ASSERT lsu_cache_path MSHR flight owner %0d state %0d", m_owner, st[m_owner]);
      if (c_inflight && !c_flight_cfg &&
          !(st[c_owner] inside {R_LOOKUP, R_REFILL}))
        $error("C031_ASSERT lsu_cache_path cache flight owner %0d state %0d", c_owner, st[c_owner]);
      for (int c = 0; c < CONTEXT_COUNT; c++)
        if (rsp_granted[c] && rsp_valid_o[c] && st[rsp_grant[c]] != R_RSP)
          $error("C031_ASSERT lsu_cache_path context %0d granted a non-response record", c);
    end
  end
endmodule
