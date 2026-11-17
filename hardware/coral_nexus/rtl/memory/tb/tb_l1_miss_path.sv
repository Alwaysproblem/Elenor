// tb_l1_miss_path.sv — C030 integrated bench: real cache.sv + mshr_table.sv +
// transfer_outbridge.sv behind l1_miss_path.sv.
//
// The wire side is a bench-owned memory-service model with its own terminal-
// reference ledger (mirroring pipeline_validator/cosim/memory_service.py: one
// reference per accepted (txn_id, slice), released only by TERMINAL_ACK). It
// holds responses back on demand so delay/cancel orderings are real.
//
// Scenarios:
//   S1  W4 §4.4: two waiters merge; the line is refilled into the CACHE, the
//       cache line is then EVICTED by capacity pressure, and the delayed waiter
//       still receives the original bytes from the entry-owned fill buffer; the
//       MSHR slot is not reusable until both deliveries AND both reference
//       releases close AND the service-side terminal reference has closed.
//   S2  ISSUE -> CANCEL -> waiters consume/release -> delayed MEM_RSP: the slot
//       stays busy until the TERMINAL_ACK(ISOLATED) is handed to the service
//       and the bridge settles the MSHR; late data is absorbed, never delivered.
//   S3  faulted MEM_RSP: waiters get a terminal no-data response, the service
//       reference closes with FAULTED, the slot frees only after all releases.
//   S4  stale/forged MEM_RSP (wrong txn, wrong binding generation) is counted
//       and ignored; it neither fills an entry nor closes a reference.
//   S5  cache: dirty eviction faults (no silent discard), range maintenance
//       streams one clean per dirty line, failed clean keeps the line dirty,
//       reconfigure refuses while dirty/clean-pending and only advances.
//   S6  closure: after every scenario the service ledger, MSHR and bridge are
//       all empty and no invariant error was raised.
`timescale 1ns/1ps
module tb_l1_miss_path
  import c029_params_pkg::*;
#(
  parameter int CONTEXT_COUNT = C2_CONTEXTS
) ();
  localparam int E = MSHR_ENTRIES;
  localparam int LB = LINE_BYTES;
  localparam int W = 4;
  localparam int SW = $clog2(E);
  localparam int WW = $clog2(W);
  localparam int PROF = 1; // profile 1 cache capacity from the sourced table
  localparam logic [$clog2(PROFILE_COUNT)-1:0] PROF_IDX = $clog2(PROFILE_COUNT)'(PROF);

  localparam logic [2:0] M_ALLOC = 3'd0, M_ISSUE = 3'd1, M_CONSUME = 3'd3, M_RELEASE = 3'd4,
                         M_CANCEL = 3'd5, M_RECONFIG = 3'd6;
  localparam logic [3:0] MR_OK = 4'd0, MR_LEADER = 4'd1, MR_MERGED = 4'd2, MR_FULL = 4'd3,
                         MR_BAD_TOKEN = 4'd6, MR_BAD_STATE = 4'd7, MR_NOT_FILLED = 4'd8,
                         MR_CANCELLED = 4'd9;
  localparam logic [2:0] C_LOOKUP = 3'd0, C_REFILL = 3'd1, C_DIRTY = 3'd2, C_MAINT = 3'd3,
                         C_DONE = 3'd4, C_RECONFIG = 3'd5;
  localparam logic [3:0] CR_OK = 4'd0, CR_HIT = 4'd1, CR_MISS = 4'd2, CR_DISABLED = 4'd3,
                         CR_STALE = 4'd4, CR_PENDING = 4'd5, CR_EVICT_DIRTY = 4'd6,
                         CR_NOT_RESIDENT = 4'd7, CR_READ_ONLY = 4'd8, CR_DIRTY_NO_CLEAN = 4'd9,
                         CR_UNKNOWN_CLEAN = 4'd10, CR_BUSY = 4'd11, CR_PROFILE_SKEW = 4'd14,
                         CR_BANK_FAULT = 4'd15;

  logic clk = 0, rst_n = 0;
  // MSHR client
  logic m_valid = 0, m_ready, m_last = 0;
  logic [2:0] m_op = 0;
  logic [63:0] m_key = 0;
  logic [31:0] m_profile_gen = 0, m_entry_gen = 0;
  logic [SW-1:0] m_slot = 0;
  logic [WW-1:0] m_waiter = 0;
  logic [LB*8-1:0] m_data = 0;
  logic [LB-1:0] m_mask = 0;
  logic m_rsp_valid, m_rsp_ready = 1;
  logic [3:0] m_rsp_code;
  logic [SW-1:0] m_rsp_slot;
  logic [31:0] m_rsp_entry_gen;
  logic [LB*8-1:0] m_rsp_data;
  logic [LB-1:0] m_rsp_mask;
  // cache client
  logic c_valid = 0, c_ready;
  logic [2:0] c_op = 0;
  logic [63:0] c_alloc_id = 0;
  logic [31:0] c_alloc_gen = 0, c_line_offset = 0, c_profile_gen = 0;
  logic [LB*8-1:0] c_data = 0;
  logic [LB-1:0] c_validity = '1;
  logic c_dirty = 0, c_maint_all = 0, c_maint_clean = 0, c_maint_invalidate = 0;
  logic [31:0] c_range_off = 0, c_range_len = 0, c_clean_id = 0;
  logic c_clean_ok = 1;
  logic [$clog2(PROFILE_COUNT)-1:0] c_profile_idx = 0;
  logic c_write_back = 0;
  logic c_rsp_valid, c_rsp_ready = 1;
  logic [3:0] c_rsp_code;
  logic [LB*8-1:0] c_rsp_data;
  logic [LB-1:0] c_rsp_validity;
  logic clean_valid, clean_ready = 1;
  logic [31:0] clean_id, clean_alloc_gen, clean_line_offset;
  logic [63:0] clean_alloc_id;
  logic [LB*8-1:0] clean_data;
  // refill descriptor + cancel
  logic iss_valid = 0, iss_ready;
  logic [31:0] iss_binding_id = 7, iss_binding_generation = 1, iss_binding_offset = 0;
  logic [31:0] iss_ctx_slot = 0, iss_ctx_generation = 1, iss_arena_alloc_gen = 1, iss_scope_epoch = 0;
  logic [SW-1:0] iss_mshr_slot = 0;
  logic [31:0] iss_mshr_gen = 0;
  logic cancel_valid = 0;
  logic [SW-1:0] cancel_mshr_slot = 0;
  logic [31:0] cancel_mshr_gen = 0;
  // wire
  logic mem_req_valid, mem_req_ready = 1;
  logic [31:0] mem_req_txn_id, mem_req_slice_id, mem_req_slice_count, mem_req_binding_id;
  logic [31:0] mem_req_binding_generation, mem_req_binding_offset, mem_req_length;
  logic mem_rsp_valid = 0, mem_rsp_ready;
  logic [31:0] mem_rsp_txn_id = 0, mem_rsp_slice_id = 0, mem_rsp_binding_generation = 0, mem_rsp_status = 0;
  logic [LB*8-1:0] mem_rsp_data = 0;
  logic term_valid, term_ready = 1;
  logic [31:0] term_txn_id, term_slice_id, term_slice_count;
  logic [1:0] term_final_state;
  logic [$clog2(E+1)-1:0] mshr_active_o;
  logic [E-1:0] mshr_busy_o;
  logic [31:0] cache_resident_o, cache_dirty_o, cache_pending_cleans_o, cache_capacity_o;
  logic [63:0] cache_evictions_o;
  logic [$clog2(E+1)-1:0] bridge_outstanding_o;
  logic [63:0] bridge_sent_o, bridge_terminals_o, bridge_absorbed_o, bridge_stale_rsp_o;
  logic invariant_error_o;

  // ---------------- real l1_bank behind the cache's cp_* data port ----------------
  localparam int BANKS = BANK_COUNT;
  localparam int BANK_IDX_W = (BANK_COUNT <= 1) ? 1 : $clog2(BANK_COUNT);
  localparam int BANK_ADDR_W = $clog2(BANK_BYTES);
  localparam int BANK_LEN_W = $clog2(BANK_BYTES + 1);
  localparam int BEAT_W = $clog2(BANK_BYTES / DATA_BYTES + 1);
  localparam int PROFILE_IDX_W = $clog2(PROFILE_COUNT);
  localparam int MAX_SEG = BANK_COUNT;
  localparam int SEG_CNT_W = $clog2(MAX_SEG + 1);
  localparam int TXN_W = 64;

  // cache master <-> bank (names match l1_miss_path's cp_* ports)
  logic [PROFILE_IDX_W-1:0] bank_profile_i;
  logic cache_quiescent_o;
  logic cp_valid_o, cp_write_o, cp_wmode_o, cp_wdata_valid_o, cp_rdata_ready_o;
  logic [BANK_IDX_W-1:0] cp_bank_o;
  logic [BANK_ADDR_W-1:0] cp_addr_o;
  logic [BEAT_W-1:0] cp_beats_o;
  logic [DATA_BYTES*8-1:0] cp_wdata_o;
  logic [DATA_BYTES-1:0] cp_wben_o;
  logic cp_ready_i, cp_rej_i, cp_wdata_ready_i, cp_rdata_valid_i, cp_done_i;
  logic [DATA_BYTES*8-1:0] cp_rdata_i;
  logic [DATA_BYTES-1:0] cp_rben_i;
  logic [31:0] cache_op_cycles_o;
  logic [2:0] cp_rej_code_unused;
  logic cp_busy_unused;
  logic [63:0] bank_profile_gen_unused;

  // SPM-side client of the same bank (bench-driven, Context 0 and optionally Context 1)
  logic [CONTEXT_COUNT-1:0] sp_req_valid = '0, sp_req_ready, sp_req_rej;
  logic [CONTEXT_COUNT-1:0] sp_req_write = '0;
  logic [CONTEXT_COUNT-1:0][TXN_W-1:0] sp_txn = '0, sp_offer_gen, sp_done_txn, sp_done_gen;
  logic [CONTEXT_COUNT-1:0][SEG_CNT_W-1:0] sp_seg_count = '0;
  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_IDX_W-1:0] sp_bank = '0;
  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_ADDR_W-1:0] sp_addr = '0;
  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_LEN_W-1:0] sp_len = '0;
  logic [CONTEXT_COUNT-1:0] sp_req_done, sp_wdata_valid = '0, sp_wdata_ready;
  logic [CONTEXT_COUNT-1:0][DATA_BYTES*8-1:0] sp_wdata = '0;
  logic [CONTEXT_COUNT-1:0] sp_rvalid, sp_rready = '1;
  logic [CONTEXT_COUNT-1:0][TXN_W-1:0] sp_rtxn, sp_rgen;
  logic [CONTEXT_COUNT-1:0][DATA_BYTES*8-1:0] sp_rdata;
  logic [CONTEXT_COUNT-1:0][DATA_BYTES-1:0] sp_rben;
  logic [CONTEXT_COUNT-1:0] sp_fin_valid = '0, sp_fin_ready, sp_fin_ack, sp_fin_rej;
  logic [CONTEXT_COUNT-1:0][TXN_W-1:0] sp_fin_txn = '0, sp_fin_gen = '0;
  logic [CONTEXT_COUNT-1:0] sp_inv_ready, sp_inv_ack, sp_inv_rej;
  logic prof_valid = 1'b0, prof_ack, prof_rej;
  logic [PROFILE_IDX_W-1:0] prof_idx = '0;
  logic [63:0] prof_gen = '0;
  logic [1:0] prof_rej_code;
  logic [PROFILE_IDX_W-1:0] bank_active_profile;

  l1_bank #(.CONTEXT_COUNT(CONTEXT_COUNT)) bank (
    .clk(clk), .rstn(rst_n),
    .req_valid_i(sp_req_valid), .req_ready_o(sp_req_ready), .req_rej_o(sp_req_rej),
    .req_rej_code_o(), .req_txn_i(sp_txn), .req_generation_o(sp_offer_gen),
    .req_write_i(sp_req_write), .req_seg_count_i(sp_seg_count), .req_bank_i(sp_bank),
    .req_addr_i(sp_addr), .req_len_i(sp_len),
    .req_done_o(sp_req_done), .req_done_txn_o(sp_done_txn), .req_done_generation_o(sp_done_gen),
    .wdata_valid_i(sp_wdata_valid), .wdata_ready_o(sp_wdata_ready), .wdata_i(sp_wdata),
    .rdata_valid_o(sp_rvalid), .rdata_ready_i(sp_rready), .rdata_txn_o(sp_rtxn),
    .rdata_generation_o(sp_rgen), .rdata_data_o(sp_rdata), .rdata_ben_o(sp_rben),
    .fin_valid_i(sp_fin_valid), .fin_txn_i(sp_fin_txn), .fin_generation_i(sp_fin_gen),
    .fin_ready_o(sp_fin_ready), .fin_ack_o(sp_fin_ack), .fin_rej_o(sp_fin_rej),
    .fin_rej_code_o(),
    .inv_valid_i('0), .inv_ready_o(sp_inv_ready), .inv_bank_i('0), .inv_addr_i('0),
    .inv_len_i('0), .inv_ack_o(sp_inv_ack), .inv_rej_o(sp_inv_rej), .inv_rej_code_o(),
    .profile_set_valid_i(prof_valid), .profile_index_i(prof_idx), .profile_generation_i(prof_gen),
    .profile_set_ready_o(), .profile_ack_o(prof_ack), .profile_rej_o(prof_rej),
    .profile_rej_code_o(prof_rej_code),
    .cp_valid_i(cp_valid_o), .cp_write_i(cp_write_o), .cp_bank_i(cp_bank_o),
    .cp_addr_i(cp_addr_o), .cp_beats_i(cp_beats_o), .cp_wmode_i(cp_wmode_o),
    .cp_wdata_valid_i(cp_wdata_valid_o), .cp_wdata_i(cp_wdata_o), .cp_wben_i(cp_wben_o),
    .cp_rdata_ready_i(cp_rdata_ready_o), .cache_quiescent_i(cache_quiescent_o),
    .cp_ready_o(cp_ready_i), .cp_rej_o(cp_rej_i), .cp_rej_code_o(cp_rej_code_unused),
    .cp_wdata_ready_o(cp_wdata_ready_i), .cp_rdata_valid_o(cp_rdata_valid_i),
    .cp_rdata_o(cp_rdata_i), .cp_rben_o(cp_rben_i), .cp_done_o(cp_done_i),
    .cp_busy_o(cp_busy_unused), .active_profile_o(bank_active_profile),
    .profile_generation_o(bank_profile_gen_unused)
  );
  assign bank_profile_i = bank_active_profile;

  l1_miss_path #(.WAITERS(W)) dut (.*);

  always #5 clk = ~clk;
  // 2048-line multi-cycle scans: a generous simulated-time bound (a hung bench still dies).
  initial begin #4_000_000_000; $fatal(1, "C030_ASSERT miss_path watchdog"); end

  int checks = 0;
  task automatic require(input logic ok, input string label);
    checks++;
    if (ok !== 1'b1) $fatal(1, "C030_ASSERT tb_l1_miss_path %s", label);
  endtask

  // ---------------- bench-owned memory-service model ----------------
  // One terminal reference per accepted (txn,slice 0), released only by TERMINAL_ACK.
  int svc_refs = 0;
  int svc_acks = 0;
  logic [31:0] svc_open_txn [E];
  logic svc_open [E];
  logic [1:0] svc_last_final;
  logic [31:0] svc_last_ack_txn;
  initial for (int i = 0; i < E; i++) svc_open[i] = 0;
  // Handshake-captured request log, indexed by accepted-request number. The bench
  // asserts the log never overflows, so no entry is ever aliased or overwritten.
  localparam int QDEPTH = 64;
  logic [31:0] q_txn [QDEPTH];
  logic [31:0] q_gen [QDEPTH];
  int q_count = 0;
  // Handshake-captured clean-request queues (clean_valid && clean_ready).
  logic [31:0] clean_ids [$];
  logic [31:0] clean_offs [$];
  logic [63:0] clean_aids [$];
  logic [LB*8-1:0] clean_datas [$];
  logic [SW-1:0] drain_slot [E];
  logic [31:0] drain_gen0;
  logic [SW-1:0] pressure_slot [E];
  logic [31:0] pressure_gen [E];
  logic [LB*8-1:0] stale_probe;

  always @(posedge clk) begin
    if (rst_n && mem_req_valid && mem_req_ready) begin
      require(mem_req_slice_id == 0 && mem_req_slice_count == 1 && mem_req_length == LB,
              "MEM_REQ shape");
      for (int i = 0; i < E; i++)
        if (!svc_open[i]) begin
          svc_open[i] = 1; svc_open_txn[i] = mem_req_txn_id; svc_refs++; break;
        end
      require(q_count < QDEPTH, "request log capacity not exceeded");
      q_txn[q_count] = mem_req_txn_id; q_gen[q_count] = mem_req_binding_generation;
      q_count++;
    end
    if (rst_n && term_valid && term_ready) begin
      automatic logic found = 0;
      for (int i = 0; i < E; i++)
        if (svc_open[i] && svc_open_txn[i] == term_txn_id) begin
          svc_open[i] = 0; svc_refs--; found = 1; break;
        end
      require(found, "TERMINAL_ACK names an open service reference (no forged/duplicate ack)");
      svc_acks++; svc_last_final = term_final_state; svc_last_ack_txn = term_txn_id;
    end
  end

  always @(posedge clk) begin
    if (rst_n && clean_valid && clean_ready) begin
      clean_ids.push_back(clean_id); clean_offs.push_back(clean_line_offset);
      clean_aids.push_back(clean_alloc_id); clean_datas.push_back(clean_data);
    end
  end

  // Issue one maintenance command and wait for its response with a bounded watchdog.
  // The bound is derived from the ACTIVE line count (aggregate across banks), never a fixed
  // cycle count: the check pass and the walk are each at most one slot per cycle (the scan
  // reads SCAN_LANES >= 1 slots per cycle), and every dirty in-range line costs one bank
  // read transaction: 2 beats + the bank's CP_MAX_WAIT arbitration (8) + hand-off margin.
  localparam int CLEAN_LINE_CYCLES = 16;
  task automatic run_maint(input string label);
    automatic int guard = 0;
    automatic int cap = int'(cache_capacity_o);
    automatic int bound = 2 * cap + cap * CLEAN_LINE_CYCLES + 64;
    @(negedge clk); c_op = C_MAINT; c_valid = 1;
    while (!c_ready) @(negedge clk);
    @(negedge clk); c_valid = 0;
    while (!c_rsp_valid) begin
      @(negedge clk); guard++;
      require(guard < bound, {label, ": maintenance completes within the walker bound"});
    end
    require(c_rsp_code == CR_OK, {label, ": maintenance response"});
  endtask

  task automatic wire_rsp(input logic [31:0] txn, input logic [31:0] bgen, input logic [31:0] status,
                          input logic [LB*8-1:0] data);
    @(negedge clk);
    mem_rsp_txn_id = txn; mem_rsp_slice_id = 0; mem_rsp_binding_generation = bgen;
    mem_rsp_status = status; mem_rsp_data = data; mem_rsp_valid = 1;
    @(negedge clk);
    mem_rsp_valid = 0;
  endtask

  task automatic settle_cycles(input int n);
    repeat (n) @(negedge clk);
  endtask

  // Switch the REAL bank profile (strictly increasing bank generation) and wait for its ACK.
  logic [63:0] bank_gen_seq = 64'd0;
  task automatic bank_switch(input logic [PROFILE_IDX_W-1:0] idx);
    automatic int unsigned wd;
    bank_gen_seq = bank_gen_seq + 64'd1;
    @(negedge clk);
    prof_idx = idx; prof_gen = bank_gen_seq; prof_valid = 1'b1;
    @(negedge clk);
    prof_valid = 1'b0;
    wd = 0;
    while (!prof_ack && !prof_rej && wd < 50) begin @(negedge clk); wd++; end
    require(prof_ack && !prof_rej, "bank profile switch acknowledged");
    @(negedge clk);
    require(bank_active_profile == idx, "bank reports the new active profile");
  endtask

  // Drive one SPM request into the bank and report how many cycles it waited before the bank
  // ACCEPTED it. Acceptance is the posedge where valid && ready hold together; after that edge
  // ready legitimately drops (the context is active), so it must not be sampled at the next
  // negedge. spm_hs is set by a monitor that sees the real handshake.
  logic [CONTEXT_COUNT-1:0] spm_hs = '0;
  always @(posedge clk) begin
    for (int c = 0; c < CONTEXT_COUNT; c++)
      if (rst_n && sp_req_valid[c] && sp_req_ready[c]) spm_hs[c] <= 1'b1;
  end
  task automatic spm_hold_bank(input int ctx, input int bank_idx, input logic [BANK_ADDR_W-1:0] addr,
                               input logic [TXN_W-1:0] txn, output int waited);
    automatic int wd;
    waited = 0;
    @(negedge clk);
    spm_hs[ctx] = 1'b0;
    sp_txn[ctx] = txn; sp_req_write[ctx] = 1'b0; sp_seg_count[ctx] = SEG_CNT_W'(1);
    sp_bank[ctx][0] = BANK_IDX_W'(bank_idx); sp_addr[ctx][0] = addr;
    sp_len[ctx][0] = BANK_LEN_W'(DATA_BYTES);
    sp_req_valid[ctx] = 1'b1;
    wd = 0;
    @(negedge clk);  // the first possible accepting edge has now passed
    while (!spm_hs[ctx] && wd < 400) begin @(negedge clk); wd++; end
    sp_req_valid[ctx] = 1'b0;
    waited = wd;
    require(spm_hs[ctx], "SPM request was accepted by the bank");
  endtask

  task automatic spm_release_bank(input int ctx, input logic [TXN_W-1:0] txn,
                                  input logic [TXN_W-1:0] gen);
    automatic int wd;
    wd = 0;
    while (!sp_req_done[ctx] && wd < 200) begin @(negedge clk); wd++; end
    require(sp_req_done[ctx], "SPM read data phase completes");
    sp_fin_txn[ctx] = txn; sp_fin_gen[ctx] = gen; sp_fin_valid[ctx] = 1'b1;
    @(negedge clk);
    sp_fin_valid[ctx] = 1'b0;
  endtask

  function automatic logic [LB*8-1:0] pattern(input logic [7:0] seed);
    logic [LB*8-1:0] p;
    for (int b = 0; b < LB; b++) p[b*8 +: 8] = seed + 8'(b);
    return p;
  endfunction

  // MSHR command with its registered response
  task automatic mcmd(input logic [2:0] op, input logic [63:0] key, input logic [31:0] pg,
                      input logic [SW-1:0] slot, input logic [31:0] eg, input logic [WW-1:0] wt,
                      input logic [3:0] expect_code, input string label);
    @(negedge clk);
    m_op = op; m_key = key; m_profile_gen = pg; m_slot = slot; m_entry_gen = eg; m_waiter = wt;
    m_valid = 1;
    while (!m_ready) @(negedge clk);
    @(negedge clk);
    m_valid = 0;
    while (!m_rsp_valid) @(negedge clk);
    require(m_rsp_code == expect_code, {label, ": code"});
  endtask

  task automatic ccmd(input logic [2:0] op, input logic [3:0] expect_code, input string label);
    @(negedge clk);
    c_op = op; c_valid = 1;
    while (!c_ready) @(negedge clk);
    @(negedge clk);
    c_valid = 0;
    while (!c_rsp_valid) @(negedge clk);
    require(c_rsp_code == expect_code, {label, ": code"});
  endtask

  // Hand the refill obligation to the bridge exactly as the ISSUE step would.
  task automatic issue_refill(input logic [SW-1:0] slot, input logic [31:0] gen);
    mcmd(M_ISSUE, 0, 0, slot, gen, 0, MR_OK, "mshr issue");
    @(negedge clk);
    iss_mshr_slot = slot; iss_mshr_gen = gen; iss_valid = 1;
    while (!iss_ready) @(negedge clk);
    @(negedge clk);
    iss_valid = 0;
  endtask

  logic [SW-1:0] a_slot;
  logic [31:0] a_gen, txn_a;
  logic [LB*8-1:0] gold;
  logic [SW-1:0] fill_slot_q [E];
  logic [31:0] fill_gen_q [E];
  int q_before;

  initial begin
    gold = pattern(8'h40);
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
    require(int'(mshr_active_o) == 0 && int'(bridge_outstanding_o) == 0 && svc_refs == 0 &&
            !invariant_error_o, "reset state");
    // Select cache profile 1 (sourced capacity), write-through (read-only).
    bank_switch(PROF_IDX);
    c_profile_gen = 1; c_profile_idx = PROF_IDX; c_write_back = 0;
    ccmd(C_RECONFIG, CR_OK, "cache profile select");
    // Aggregate capacity: banks * cache_bytes_per_bank / line, NOT one bank's worth
    // (validator profiles.py: cache_bytes = banks * cache_bytes_per_bank).
    require(int'(cache_capacity_o) == BANK_COUNT * (CACHE_BYTES_PER_BANK[PROF] / LB),
            "cache capacity is the aggregate across banks from the sourced table");
    require(int'(cache_capacity_o) > CACHE_BYTES_PER_BANK[PROF] / LB,
            "aggregate capacity exceeds a single bank's worth");

    // ---------------- S1: merge, cache refill, eviction, delayed waiter ------
    q_before = q_count;
    mcmd(M_ALLOC, 64'hA000, 0, 0, 0, 0, MR_LEADER, "leader");
    a_slot = m_rsp_slot; a_gen = m_rsp_entry_gen;
    mcmd(M_ALLOC, 64'hA000, 0, 0, 0, 0, MR_MERGED, "second waiter merges");
    issue_refill(a_slot, a_gen);
    settle_cycles(3);
    require(q_count == q_before + 1 && svc_refs == 1, "one MEM_REQ, one service reference");
    txn_a = q_txn[q_before];
    // Backpressure the terminal ACK: the response is handled and the data lands in
    // the MSHR entry, but the service-side terminal reference stays open.
    term_ready = 0;
    wire_rsp(txn_a, 1, 0, gold);
    settle_cycles(6);
    require(svc_refs == 1 && svc_acks == 0 && term_valid, "terminal ACK is held under backpressure");
    require(bridge_terminals_o == 0, "no terminal counted before the ACK handshake");
    // Install the line into the cache, then evict it with capacity pressure.
    c_alloc_id = 64'h77; c_alloc_gen = 3; c_line_offset = 0; c_profile_gen = 1;
    c_data = gold; c_validity = '1; c_dirty = 0;
    ccmd(C_REFILL, CR_OK, "cache refill of the line");
    ccmd(C_LOOKUP, CR_HIT, "line is resident");
    require(c_rsp_data == gold, "cache hit returns the refilled bytes");
    for (int i = 1; i <= int'(cache_capacity_o); i++) begin
      c_line_offset = 32'(i * LB); c_data = pattern(8'(i));
      ccmd(C_REFILL, CR_OK, "pressure refill");
    end
    require(cache_evictions_o >= 1, "capacity pressure evicted a line");
    c_line_offset = 0;
    ccmd(C_LOOKUP, CR_MISS, "original cache line was evicted");
    // Delivery and every waiter release happen while the terminal ACK is still held.
    // The delayed waiters still read the original bytes from the MSHR entry.
    mcmd(M_CONSUME, 0, 0, a_slot, a_gen, 0, MR_OK, "waiter 0 delivery after eviction");
    require(m_rsp_data == gold && m_rsp_mask == '1, "evicted line: waiter 0 gets entry bytes");
    mcmd(M_RELEASE, 0, 0, a_slot, a_gen, 0, MR_OK, "waiter 0 release");
    require(mshr_busy_o[a_slot], "slot busy with a delayed waiter outstanding");
    mcmd(M_ALLOC, 64'hB000, 0, 0, 0, 0, MR_LEADER, "other entry allocates");
    require(m_rsp_slot != a_slot, "pending slot is not reused");
    drain_slot[0] = m_rsp_slot;
    drain_gen0 = m_rsp_entry_gen;
    mcmd(M_CONSUME, 0, 0, a_slot, a_gen, 1, MR_OK, "delayed waiter 1 delivery");
    require(m_rsp_data == gold && m_rsp_mask == '1, "delayed waiter gets the original bytes");
    require(mshr_busy_o[a_slot], "slot busy until the delayed reference closes");
    mcmd(M_RELEASE, 0, 0, a_slot, a_gen, 1, MR_OK, "waiter 1 release");
    // Both waiters delivered and released, but the terminal is unacknowledged: the
    // token must still be busy and survive allocation pressure.
    require(mshr_busy_o[a_slot], "all waiters closed but terminal unacked: slot busy");
    require(svc_refs == 1, "service reference still open until the ACK handshake");
    for (int i = 0; i < E - 2; i++) begin
      mcmd(M_ALLOC, 64'hB100 + 64'(i), 0, 0, 0, 0, MR_LEADER, "pressure around the unacked token");
      require(m_rsp_slot != a_slot, "unacked token never reallocated");
      pressure_slot[i] = m_rsp_slot; pressure_gen[i] = m_rsp_entry_gen;
    end
    mcmd(M_ALLOC, 64'hB1FF, 0, 0, 0, 0, MR_FULL, "table full while the successful refill's terminal is unacked");
    require(mshr_busy_o[a_slot], "still busy after allocation pressure");
    term_ready = 1;
    settle_cycles(6);
    require(svc_refs == 0 && svc_last_final == 2'd0 && svc_last_ack_txn == txn_a && bridge_terminals_o == 1,
            "COMPLETED terminal ACK handshake closes the reference");
    require(!mshr_busy_o[a_slot], "token freed only after the ACK handshake");
    mcmd(M_ALLOC, 64'hB1FF, 0, 0, 0, 0, MR_LEADER, "slot reusable after the ACK");
    require(m_rsp_slot == a_slot && m_rsp_entry_gen != a_gen, "reuse after the ACK carries a fresh generation");
    pressure_slot[E-2] = m_rsp_slot; pressure_gen[E-2] = m_rsp_entry_gen;
    // Drain the pressure entries, the earlier side entry and the reused entry through
    // the real path, so every ledger closes.
    for (int i = 0; i < E - 1; i++) begin
      automatic logic [SW-1:0] ds = (i < E - 2) ? pressure_slot[i] : pressure_slot[E-2];
      automatic logic [31:0] dg = (i < E - 2) ? pressure_gen[i] : pressure_gen[E-2];
      q_before = q_count;
      issue_refill(ds, dg);
      settle_cycles(3);
      wire_rsp(q_txn[q_before], 1, 0, pattern(8'(8'h50 + i)));
      settle_cycles(8);
      mcmd(M_CONSUME, 0, 0, ds, dg, 0, MR_OK, "pressure entry delivery");
      require(m_rsp_data == pattern(8'(8'h50 + i)), "pressure entry holds its own bytes");
      mcmd(M_RELEASE, 0, 0, ds, dg, 0, MR_OK, "pressure entry release");
      require(!mshr_busy_o[ds], "pressure entry closed after delivery, release and terminal ACK");
    end
    q_before = q_count;
    issue_refill(drain_slot[0], drain_gen0);
    settle_cycles(3);
    wire_rsp(q_txn[q_before], 1, 0, pattern(8'h21));
    settle_cycles(8);
    mcmd(M_CONSUME, 0, 0, drain_slot[0], drain_gen0, 0, MR_OK, "side entry delivery");
    require(m_rsp_data == pattern(8'h21), "side entry holds its own bytes");
    mcmd(M_RELEASE, 0, 0, drain_slot[0], drain_gen0, 0, MR_OK, "side entry release");
    require(!mshr_busy_o[drain_slot[0]], "side entry closed");

    // ---------------- S2: ISSUE -> CANCEL -> consume/release -> delayed MEM_RSP ----
    mcmd(M_ALLOC, 64'hC000, 0, 0, 0, 0, MR_LEADER, "cancel leader");
    a_slot = m_rsp_slot; a_gen = m_rsp_entry_gen;
    mcmd(M_ALLOC, 64'hC000, 0, 0, 0, 0, MR_MERGED, "cancel second waiter");
    q_before = q_count;
    issue_refill(a_slot, a_gen);
    settle_cycles(3);
    require(q_count == q_before + 1, "cancel case sent exactly one MEM_REQ");
    txn_a = q_txn[q_before];
    @(negedge clk); cancel_mshr_slot = a_slot; cancel_mshr_gen = a_gen; cancel_valid = 1;
    @(negedge clk); cancel_valid = 0;
    mcmd(M_CANCEL, 0, 0, a_slot, a_gen, 0, MR_OK, "mshr cancel mark");
    for (int w = 0; w < 2; w++) begin
      mcmd(M_CONSUME, 0, 0, a_slot, a_gen, WW'(w), MR_CANCELLED, "cancelled waiter terminal response");
      require(m_rsp_data == '0 && m_rsp_mask == '0, "cancelled waiter gets no data");
      mcmd(M_RELEASE, 0, 0, a_slot, a_gen, WW'(w), MR_OK, "cancelled waiter release");
    end
    require(mshr_busy_o[a_slot], "all waiters released but the accepted refill is unsettled: slot busy");
    require(svc_refs == 1, "service terminal reference still open before any terminal ack");
    wire_rsp(txn_a, 1, 0, pattern(8'hEE));
    settle_cycles(8);
    require(svc_refs == 0 && svc_last_final == 2'd1 && svc_last_ack_txn == txn_a,
            "ISOLATED terminal ack closes the service reference");
    require(bridge_absorbed_o >= 1, "late data was absorbed, not delivered");
    require(!mshr_busy_o[a_slot], "slot freed only after the terminal ack settled the MSHR");
    require(int'(bridge_outstanding_o) == 0, "bridge record closed");

    // ---------------- S2b: cancel and MEM_RSP land in the same cycle ----------------
    // The response sees rcancel=0 and parks the record in FILL while the cancel mark
    // lands on the same edge. Only the FILL-with-cancel absorb path can stop the data
    // from reaching the MSHR entry; it must be absorbed and the ISOLATED terminal owed.
    mcmd(M_ALLOC, 64'hC800, 0, 0, 0, 0, MR_LEADER, "race leader");
    a_slot = m_rsp_slot; a_gen = m_rsp_entry_gen;
    q_before = q_count;
    issue_refill(a_slot, a_gen);
    settle_cycles(3);
    txn_a = q_txn[q_before];
    @(negedge clk);
    cancel_mshr_slot = a_slot; cancel_mshr_gen = a_gen; cancel_valid = 1;
    mem_rsp_txn_id = txn_a; mem_rsp_slice_id = 0; mem_rsp_binding_generation = 1;
    mem_rsp_status = 0; mem_rsp_data = pattern(8'hCD); mem_rsp_valid = 1;
    @(negedge clk);
    cancel_valid = 0; mem_rsp_valid = 0;
    mcmd(M_CANCEL, 0, 0, a_slot, a_gen, 0, MR_OK, "race: mshr cancel mark");
    settle_cycles(8);
    require(svc_refs == 0 && svc_last_final == 2'd1 && svc_last_ack_txn == txn_a,
            "race: ISOLATED terminal ack, not COMPLETED");
    mcmd(M_CONSUME, 0, 0, a_slot, a_gen, 0, MR_CANCELLED, "race: waiter gets a terminal response");
    require(m_rsp_data == '0 && m_rsp_mask == '0, "race: the data never reached the MSHR entry");
    mcmd(M_RELEASE, 0, 0, a_slot, a_gen, 0, MR_OK, "race: waiter release");
    require(!mshr_busy_o[a_slot] && int'(bridge_outstanding_o) == 0, "race: entry and bridge record closed");
    // ---------------- S3: faulted MEM_RSP ----------------------------------------
    mcmd(M_ALLOC, 64'hD000, 0, 0, 0, 0, MR_LEADER, "fault leader");
    a_slot = m_rsp_slot; a_gen = m_rsp_entry_gen;
    q_before = q_count;
    issue_refill(a_slot, a_gen);
    settle_cycles(3);
    txn_a = q_txn[q_before];
    wire_rsp(txn_a, 1, 32'd1, '0);
    settle_cycles(8);
    require(svc_last_final == 2'd2 && svc_last_ack_txn == txn_a && svc_refs == 0,
            "FAULTED terminal ack closes the service reference");
    mcmd(M_CONSUME, 0, 0, a_slot, a_gen, 0, MR_CANCELLED, "fault: waiter gets a terminal response");
    require(m_rsp_data == '0, "fault response carries no data");
    require(mshr_busy_o[a_slot], "faulted entry busy until its waiter releases");
    mcmd(M_RELEASE, 0, 0, a_slot, a_gen, 0, MR_OK, "fault: waiter release");
    require(!mshr_busy_o[a_slot], "faulted entry freed after release");

    // ---------------- S4: stale / forged MEM_RSP ---------------------------------
    mcmd(M_ALLOC, 64'hE000, 0, 0, 0, 0, MR_LEADER, "stale-probe leader");
    a_slot = m_rsp_slot; a_gen = m_rsp_entry_gen;
    q_before = q_count;
    issue_refill(a_slot, a_gen);
    settle_cycles(3);
    txn_a = q_txn[q_before];
    wire_rsp(txn_a + 32'd9, 1, 0, pattern(8'hAA));
    wire_rsp(txn_a, 32'd5, 0, pattern(8'hBB));
    settle_cycles(4);
    require(bridge_stale_rsp_o == 2, "wrong txn and wrong binding generation are counted as stale");
    require(svc_refs == 1, "stale responses closed no service reference");
    mcmd(M_CONSUME, 0, 0, a_slot, a_gen, 0, MR_NOT_FILLED, "stale response did not fill the entry");
    stale_probe = pattern(8'h5A);
    wire_rsp(txn_a, 1, 0, stale_probe);
    settle_cycles(8);
    require(svc_refs == 0 && svc_last_final == 2'd0, "the real response then completes normally");
    mcmd(M_CONSUME, 0, 0, a_slot, a_gen, 0, MR_OK, "genuine data delivered");
    require(m_rsp_data == stale_probe, "entry holds the genuine response bytes only");
    mcmd(M_RELEASE, 0, 0, a_slot, a_gen, 0, MR_OK, "genuine release");
    require(!mshr_busy_o[a_slot], "stale-probe entry closed");

    // ---------------- S5: cache dirty eviction, maintenance, reconfigure ----------
    c_profile_gen = 2; c_profile_idx = PROF_IDX; c_write_back = 1;
    ccmd(C_RECONFIG, CR_OK, "reconfigure to write-back at a higher generation");
    c_alloc_id = 64'h99; c_alloc_gen = 1; c_profile_gen = 2; c_dirty = 1; c_validity = '1;
    for (int i = 0; i < int'(cache_capacity_o); i++) begin
      c_line_offset = 32'(i * LB); c_data = pattern(8'(8'h10 + i));
      ccmd(C_REFILL, CR_OK, "fill the cache with dirty lines");
    end
    require(int'(cache_dirty_o) == int'(cache_capacity_o), "every resident line is dirty");
    c_line_offset = 32'(int'(cache_capacity_o) * LB); c_data = pattern(8'hCC);
    ccmd(C_REFILL, CR_EVICT_DIRTY, "evicting a dirty line faults, never discards it");
    require(int'(cache_dirty_o) == int'(cache_capacity_o), "no dirty line was lost");
    c_profile_gen = 3; c_profile_idx = 0; c_write_back = 0;
    ccmd(C_RECONFIG, CR_BUSY, "reconfigure refused while lines are dirty");
    c_profile_gen = 1;
    ccmd(C_RECONFIG, CR_STALE, "reconfigure must advance the generation");
    c_profile_gen = 2;
    c_alloc_id = 64'h99; c_alloc_gen = 1;
    c_maint_all = 0; c_maint_clean = 1; c_maint_invalidate = 1;
    c_range_off = 0; c_range_len = 32'(2 * LB);
    clean_ids.delete(); clean_offs.delete(); clean_aids.delete(); clean_datas.delete();
    run_maint("range maintenance");
    begin
      for (int i = 0; i < clean_ids.size(); i++) begin
        require(clean_aids[i] == 64'h99 && clean_offs[i] < 32'(2 * LB),
                "clean request stays inside the maintenance range");
        require(clean_datas[i] == pattern(8'(8'h10 + int'(clean_offs[i] / LB))),
                "clean request carries the dirty line bytes");
      end
      require(clean_ids.size() == 2, "one clean request per dirty in-range line");
      require(int'(cache_pending_cleans_o) == 2, "both cleans pending");
      c_clean_id = clean_ids[0]; c_clean_ok = 0;
      ccmd(C_DONE, CR_OK, "failed clean completes");
      require(int'(cache_dirty_o) == int'(cache_capacity_o), "failed clean keeps the line dirty");
      require(int'(cache_pending_cleans_o) == 1, "failed clean is no longer pending");
      c_clean_id = clean_ids[1]; c_clean_ok = 1;
      ccmd(C_DONE, CR_OK, "successful clean completes");
      require(int'(cache_dirty_o) == int'(cache_capacity_o) - 1, "successful clean clears exactly one dirty line");
      c_clean_id = 32'hDEAD; c_clean_ok = 1;
      ccmd(C_DONE, CR_UNKNOWN_CLEAN, "unknown clean id is refused");
    end

    // ---------------- S5b: remaining cache rules ---------------------------------
    // Maintenance finished, so every line is clean or clean-pending-free except dirty ones;
    // clean all remaining dirty lines to reach a reconfigurable state.
    c_maint_all = 1; c_maint_clean = 1; c_maint_invalidate = 1;
    clean_ids.delete(); clean_offs.delete(); clean_aids.delete(); clean_datas.delete();
    run_maint("full maintenance");
    begin
      // One request per dirty, non-pending line: capacity minus the one cleaned line.
      require(clean_ids.size() == int'(cache_capacity_o) - 1,
              "full maintenance cleans every remaining dirty line, none dropped");
      for (int i = 0; i < clean_ids.size(); i++) begin
        c_clean_id = clean_ids[i]; c_clean_ok = 1;
        ccmd(C_DONE, CR_OK, "complete remaining clean");
      end
      require(int'(cache_dirty_o) == 0 && int'(cache_pending_cleans_o) == 0 && int'(cache_resident_o) == 0,
              "full clean+invalidate leaves an empty cache");
    end
    c_maint_all = 0; c_maint_clean = 0; c_maint_invalidate = 0;
    // Stale-generation refill is refused, counted, and writes nothing.
    c_profile_gen = 1; c_dirty = 0; c_alloc_id = 64'hAB; c_alloc_gen = 1; c_line_offset = 0;
    c_data = pattern(8'h33);
    ccmd(C_REFILL, CR_STALE, "old-generation refill is rejected");
    require(int'(cache_resident_o) == 0, "rejected stale refill wrote no line");
    // LRU victim identity: fill to capacity, touch the oldest line, then overflow.
    c_profile_gen = 2;
    for (int i = 0; i < int'(cache_capacity_o); i++) begin
      c_line_offset = 32'(i * LB); c_data = pattern(8'(8'h60 + i));
      ccmd(C_REFILL, CR_OK, "fill cache with clean lines");
    end
    c_line_offset = 0;
    ccmd(C_LOOKUP, CR_HIT, "touch the oldest line so it becomes most recent");
    c_line_offset = 32'(int'(cache_capacity_o) * LB); c_data = pattern(8'h77);
    ccmd(C_REFILL, CR_OK, "overflow refill evicts the true LRU line");
    c_line_offset = 0;
    ccmd(C_LOOKUP, CR_HIT, "recently touched line survived the eviction");
    c_line_offset = LB;
    ccmd(C_LOOKUP, CR_MISS, "the least recently used line was the victim");
    // Reconfigure must strictly advance the generation, even when the cache is clean.
    c_profile_gen = 2; c_profile_idx = 0; c_write_back = 0;
    ccmd(C_RECONFIG, CR_STALE, "reconfigure with an equal generation is refused");
    // A disabled cache (profile 0, capacity 0) is not a one-line fallback.
    bank_switch('0);
    c_profile_gen = 3; c_profile_idx = 0; c_write_back = 0;
    ccmd(C_RECONFIG, CR_OK, "reconfigure to the disabled-cache profile");
    require(int'(cache_capacity_o) == 0, "profile 0 capacity is zero from the sourced table");
    c_line_offset = 0;
    ccmd(C_LOOKUP, CR_DISABLED, "lookup on a disabled cache is refused, never a hit");
    c_data = pattern(8'h11);
    ccmd(C_REFILL, CR_DISABLED, "refill on a disabled cache is refused");
    require(int'(cache_resident_o) == 0, "disabled cache holds no line");
    // ---------------- S7: multi-lane victim selection (global LRU across lane groups) -------
    // Start from an empty cache, fill every active slot (clean), touch ONE line that sits in
    // the LAST lane group and ONE in the first, then overflow: the victim must be the global
    // oldest line wherever it sits, and a free slot must be the FIRST free one. A scan that
    // lets later lanes overwrite earlier ones of the same cycle picks the wrong line.
    bank_switch(PROF_IDX);
    c_profile_gen = 4; c_profile_idx = PROF_IDX; c_write_back = 0; c_dirty = 0;
    ccmd(C_RECONFIG, CR_OK, "S7: re-enable the cache at profile 1");
    begin
      automatic int cap = int'(cache_capacity_o);
      automatic int victim_line = 3;  // not at a lane-group boundary on purpose
      require(cap == BANK_COUNT * (CACHE_BYTES_PER_BANK[PROF] / LB), "S7: aggregate capacity again");
      c_alloc_id = 64'h5A; c_alloc_gen = 1; c_profile_gen = 4;
      for (int i = 0; i < cap; i++) begin
        c_line_offset = 32'(i * LB); c_data = pattern(8'(i)); c_validity = '1;
        ccmd(C_REFILL, CR_OK, "S7: fill every slot");
      end
      require(int'(cache_resident_o) == cap, "S7: cache is full");
      // Touch everything EXCEPT victim_line, oldest-first, so victim_line is the global LRU.
      for (int i = 0; i < cap; i++) begin
        if (i != victim_line) begin
          c_line_offset = 32'(i * LB);
          ccmd(C_LOOKUP, CR_HIT, "S7: touch a non-victim line");
          require(c_rsp_data == pattern(8'(i)), "S7: hit returns the bytes stored in the bank");
        end
      end
      c_line_offset = 32'(cap * LB); c_data = pattern(8'hD7);
      ccmd(C_REFILL, CR_OK, "S7: overflow refill evicts exactly one line");
      c_line_offset = 32'(victim_line * LB);
      ccmd(C_LOOKUP, CR_MISS, "S7: the global-oldest line was the victim");
      for (int i = 0; i < cap; i += 97) begin
        if (i != victim_line) begin
          c_line_offset = 32'(i * LB);
          ccmd(C_LOOKUP, CR_HIT, "S7: no other line was evicted");
        end
      end
      // Free-slot choice: drop two lines, refill: the FIRST free slot must be reused, so
      // residency returns to capacity without an eviction.
      begin
        automatic logic [63:0] ev0 = cache_evictions_o;
        c_range_off = 32'(5 * LB); c_range_len = 32'(LB);
        c_alloc_id = 64'h5A; c_alloc_gen = 1; c_maint_all = 0; c_maint_clean = 0;
        c_maint_invalidate = 1;
        run_maint("S7: invalidate_range drops a clean line");
        require(int'(cache_resident_o) == cap - 1, "S7: clean in-range line dropped without a clean");
        c_maint_invalidate = 0;
        c_line_offset = 32'(5 * LB); c_data = pattern(8'h5C);
        ccmd(C_REFILL, CR_OK, "S7: refill into the freed slot");
        require(int'(cache_resident_o) == cap && cache_evictions_o == ev0,
                "S7: a free slot was used, no eviction");
      end
    end

    // ---------------- S8: cache <-> SPM arbitration on the real bank ------------------------
    // The cache's bank port and an SPM request contend for the SAME bank. SPM accepted first
    // holds the bank: the cache op must wait, then complete with correct bytes; after the SPM
    // finishes the cache proceeds. Bytes read back through the cache are the stored ones.
    begin
      automatic int waited;
      automatic int bank_of_line = 0;  // line 0 lives in bank 0 (slot % BANK_COUNT)
      automatic logic [TXN_W-1:0] t1 = 64'h77;
      automatic logic [TXN_W-1:0] g1;
      automatic int cyc_before, cyc_after;
      c_line_offset = 0; c_alloc_id = 64'h5A; c_alloc_gen = 1; c_profile_gen = 4;
      // An SPM holder on bank 0 (a user-SPM address: above the reserved bytes, below the cache window).
      spm_hold_bank(0, bank_of_line, BANK_ADDR_W'(SYSTEM_RESERVED_BYTES), t1, waited);
      require(waited == 0, "S8: an idle bank accepts the SPM request at once");
      g1 = sp_offer_gen[0];
      // Issue a cache lookup of a line striped onto the held bank; it must wait, not corrupt.
      cyc_before = cache_op_cycles_o;
      @(negedge clk); c_op = C_LOOKUP; c_valid = 1;
      while (!c_ready) @(negedge clk);
      @(negedge clk); c_valid = 0;
      repeat (20) @(negedge clk);
      require(!c_rsp_valid, "S8: the cache operation waits behind the SPM holder (no response yet)");
      // Release the SPM holder; the cache op then completes with the stored bytes.
      spm_release_bank(0, t1, sp_done_gen[0]);
      begin
        automatic int wd = 0;
        while (!c_rsp_valid && wd < 400) begin @(negedge clk); wd++; end
      end
      require(c_rsp_valid && c_rsp_code == CR_HIT, "S8: the waiting lookup completes as a hit");
      require(c_rsp_data == pattern(8'(0)), "S8: bytes read through the shared bank are the stored ones");
    end

    // ---------------- S9: a bank profile switch the cache has not followed -----------------
    // The bank and the cache must agree on the cache window. A switch of the bank alone leaves
    // the cache addressing a stale window: every cache operation is refused (PROFILE_SKEW) with
    // no side effect, until the cache is reconfigured to the same profile.
    begin
      automatic int res0 = int'(cache_resident_o);
      // The cache is idle and clean, so the bank accepts a switch to another cache profile.
      bank_switch($clog2(PROFILE_COUNT)'(2));
      c_line_offset = 0;
      ccmd(C_LOOKUP, CR_PROFILE_SKEW, "S9: lookup refused while the profiles disagree");
      c_data = pattern(8'h42);
      ccmd(C_REFILL, CR_PROFILE_SKEW, "S9: refill refused while the profiles disagree");
      require(int'(cache_resident_o) == res0, "S9: a refused operation changed no line");
      // Reconfigure the cache to the bank's profile: operations resume at the new capacity.
      c_profile_gen = 5; c_profile_idx = $clog2(PROFILE_COUNT)'(2); c_write_back = 0;
      ccmd(C_RECONFIG, CR_OK, "S9: the cache follows the bank to profile 2");
      require(int'(cache_capacity_o) == BANK_COUNT * (CACHE_BYTES_PER_BANK[2] / LB),
              "S9: capacity is the aggregate of the new profile");
      require(int'(cache_resident_o) == 0, "S9: reconfigure dropped every line");
      c_alloc_id = 64'h6B; c_alloc_gen = 1; c_line_offset = 0; c_profile_gen = 5;
      c_data = pattern(8'h99); c_validity = '1;
      ccmd(C_REFILL, CR_OK, "S9: refill at the new profile");
      ccmd(C_LOOKUP, CR_HIT, "S9: hit at the new profile");
      require(c_rsp_data == pattern(8'h99), "S9: bytes come back from the NEW cache window");
    end
    // ---------------- S6: closure ---------------------------------------------
    require(int'(mshr_active_o) == 0 && int'(bridge_outstanding_o) == 0 && svc_refs == 0,
            "MSHR, bridge and service ledger all closed");
    require(bridge_sent_o == bridge_terminals_o, "every accepted MEM_REQ got exactly one terminal ack");
    require(!invariant_error_o, "no internal invariant error in any module");
    $display("C030_PASS tb_l1_miss_path checks=%0d", checks);
    $finish;
  end
endmodule
