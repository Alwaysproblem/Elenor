// cache.sv — C030 L1 cache control over the shared l1_bank cache partition.
//
// Functional reference: pipeline_validator/memory/cache.py (`profiled_lru_v0`):
// fully associative, global true-LRU victim, line = identity + per-byte validity
// + data + dirty. This module keeps those semantics exactly; it changes only HOW
// they are computed:
//   * line DATA and per-byte VALIDITY live in the bank SRAM (l1_bank cache data
//     port), never in a private array. Physical slot s maps to bank (s mod
//     BANK_COUNT), row (s / BANK_COUNT); the row's bank-local address is
//     SPM_BYTES_PER_BANK[profile] + row * LINE_B. Slots are striped across all
//     banks, so the aggregate capacity is the sourced
//     BANK_COUNT * CACHE_BYTES_PER_BANK[profile] / LINE_B (profile 1: 2048 lines),
//     not one bank's worth.
//   * tag / dirty / pending / LRU-stamp state is a bounded multi-cycle scan over
//     the ACTIVE slots, SCAN_LANES slots per cycle. Tag/free/LRU fan-out is
//     therefore SCAN_LANES wide, not capacity wide. Operation latency is bounded
//     by ceil(active_lines / SCAN_LANES) plus the bank access; the bound is
//     derived from the active-line count, never from a fixed cycle count.
//     SCAN_LANES is an implementation knob (no PPA claim, unsourced).
//   * a line is live iff v[slot] and its fill generation equals the profile
//     generation, so a reconfigure drops every line in O(1) (the model clears
//     them; observable behavior is identical).
//
// Semantics carried from cache.py:
//   * refill/maintenance carry a profile generation; an old-generation refill is
//     rejected and counted, never written;
//   * eviction takes the global LRU line; a dirty or clean-pending victim FAULTS
//     (EVICT_DIRTY) with no side effect - dirty data is never discarded;
//   * maintenance streams one clean request per dirty, non-pending in-range line
//     (data read back from the bank), invalidates clean lines when asked, and a
//     dirty in-range line without `clean` is refused up front with no side effect;
//   * complete_clean(success) clears dirty (and invalidates when requested); a
//     failed clean keeps the line dirty;
//   * reconfigure takes effect only when no line is dirty and no clean is
//     pending, strictly advances the generation, and only to the profile the bank
//     already runs (bank_profile_i); any operation while the cache and bank
//     profiles disagree is refused with PROFILE_SKEW and no side effect.
//
// One command per cycle; the response is registered and held until rsp_ready.
`timescale 1ns/1ps
module cache
  import c029_params_pkg::*;
#(
  parameter int unsigned LINE_B = LINE_BYTES,
  parameter int unsigned SCAN_LANES = 8
) (
  input  logic clk,
  input  logic rst_n,

  input  logic cmd_valid,
  output logic cmd_ready,
  input  logic [2:0] cmd_op,
  input  logic [63:0] cmd_alloc_id,
  input  logic [31:0] cmd_alloc_gen,
  input  logic [31:0] cmd_line_offset,
  input  logic [31:0] cmd_profile_gen,
  input  logic [LINE_B*8-1:0] cmd_data,
  input  logic [LINE_B-1:0] cmd_validity,
  input  logic cmd_dirty,
  input  logic cmd_maint_all,
  input  logic cmd_maint_clean,
  input  logic cmd_maint_invalidate,
  input  logic [31:0] cmd_range_off,
  input  logic [31:0] cmd_range_len,
  input  logic [31:0] cmd_clean_id,
  input  logic cmd_clean_ok,
  input  logic [$clog2(PROFILE_COUNT)-1:0] cmd_profile_idx,
  input  logic cmd_write_back,

  output logic rsp_valid,
  input  logic rsp_ready,
  output logic [3:0] rsp_code,
  output logic [LINE_B*8-1:0] rsp_data,
  output logic [LINE_B-1:0] rsp_validity,

  // streamed writeback work (one request per dirty line)
  output logic clean_valid,
  input  logic clean_ready,
  output logic [31:0] clean_id,
  output logic [63:0] clean_alloc_id,
  output logic [31:0] clean_alloc_gen,
  output logic [31:0] clean_line_offset,
  output logic [LINE_B*8-1:0] clean_data,
  output logic [31:0] clean_profile_gen,

  // ---- l1_bank cache data port (master side; frozen cp_* interface) ----
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

  output logic [31:0] profile_gen_o,
  output logic [31:0] capacity_lines_o,
  output logic write_back_o,
  output logic [31:0] resident_lines_o,
  output logic [31:0] dirty_lines_o,
  output logic [31:0] pending_cleans_o,
  output logic [63:0] hits_o, misses_o, refills_o, evictions_o, stale_rejects_o,
  output logic [31:0] op_cycles_o,
  output logic maint_busy_o,
  output logic invariant_error_o
);
  localparam int unsigned BEATS = LINE_B / DATA_BYTES;
  localparam int unsigned BANK_IDX_W = (BANK_COUNT <= 1) ? 1 : $clog2(BANK_COUNT);
  localparam int unsigned BANK_ADDR_W = $clog2(BANK_BYTES);
  localparam int unsigned BEAT_W = $clog2(BANK_BYTES / DATA_BYTES + 1);

  function automatic int unsigned max_lines();
    int unsigned m;
    m = 0;
    for (int p = 0; p < PROFILE_COUNT; p++)
      if (BANK_COUNT * (CACHE_BYTES_PER_BANK[p] / LINE_B) > m)
        m = BANK_COUNT * (CACHE_BYTES_PER_BANK[p] / LINE_B);
    return m;
  endfunction
  localparam int unsigned PHYS = max_lines();
  localparam int unsigned SLOT_W = (PHYS <= 1) ? 1 : $clog2(PHYS);

  initial begin
    assert (LINE_B % DATA_BYTES == 0 && BEATS >= 1)
      else $fatal(1, "C030_ASSERT cache line is not an integral number of bank beats");
    assert (CACHE_BYTES_PER_BANK[0] == 0 || PHYS > 0)
      else $fatal(1, "C030_ASSERT cache profile table has no capacity");
  end

  localparam logic [2:0] OP_LOOKUP = 3'd0, OP_REFILL = 3'd1, OP_MARK_DIRTY = 3'd2,
                         OP_MAINT = 3'd3, OP_COMPLETE_CLEAN = 3'd4, OP_RECONFIG = 3'd5;
  localparam logic [3:0] R_OK = 4'd0, R_HIT = 4'd1, R_MISS = 4'd2, R_DISABLED = 4'd3,
                         R_STALE_PROFILE = 4'd4, R_PENDING = 4'd5, R_EVICT_DIRTY = 4'd6,
                         R_NOT_RESIDENT = 4'd7, R_READ_ONLY = 4'd8, R_DIRTY_NO_CLEAN = 4'd9,
                         R_UNKNOWN_CLEAN = 4'd10, R_BUSY = 4'd11, R_BAD_OP = 4'd12,
                         R_BAD_CMD = 4'd13, R_PROFILE_SKEW = 4'd14, R_BANK_FAULT = 4'd15;

  // ---- per-slot tag state (data/validity live in the bank) ----
  logic v [PHYS];
  logic [31:0] lgen [PHYS];
  logic [63:0] aid [PHYS];
  logic [31:0] agen [PHYS];
  logic [31:0] aoff [PHYS];
  logic dirty [PHYS];
  logic pend [PHYS];
  logic inv_after [PHYS];
  logic [31:0] pid [PHYS];
  logic [63:0] stamp [PHYS];

  logic [31:0] profile_gen, capacity_lines, next_clean_id;
  logic [$clog2(PROFILE_COUNT)-1:0] prof_idx;
  logic write_back;
  logic [63:0] touch, hits, misses, refills, evictions, stale_rejects;
  logic [31:0] resident_cnt, dirty_cnt, pend_cnt;
  logic inv_err;

  typedef enum logic [4:0] {
    S_IDLE, S_SCAN, S_DISPATCH, S_BANK_REQ, S_BANK_WR, S_BANK_RD, S_BANK_DONE, S_FINISH,
    S_MAINT_CHECK, S_MAINT_STEP, S_MAINT_CLEAN, S_MAINT_DONE
  } state_t;
  state_t state;

  // latched command
  logic [2:0] c_op;
  logic [63:0] c_aid;
  logic [31:0] c_agen, c_off, c_pgen, c_rng_off, c_rng_len, c_cid;
  logic [LINE_B*8-1:0] c_data;
  logic [LINE_B-1:0] c_val;
  logic c_dirty, c_all, c_clean, c_inv, c_cok, c_wb;
  logic [$clog2(PROFILE_COUNT)-1:0] c_pidx;

  // scan accumulators
  logic [31:0] s_idx;
  logic s_found, s_free_found, s_vic_found, s_any_dirty;
  logic [SLOT_W-1:0] s_hit, s_free, s_vic;
  logic [63:0] s_best;
  logic s_early;

  // bank access sequencing
  logic b_write, b_wmode;
  logic [SLOT_W-1:0] b_slot;
  logic [3:0] b_then;      // what to do after the transfer
  logic [BEAT_W-1:0] b_beat;
  logic [LINE_B*8-1:0] b_wdata;
  logic [LINE_B-1:0] b_wben;
  logic [LINE_B*8-1:0] b_rdata;
  logic [LINE_B-1:0] b_rben;
  logic done_seen;

  logic [SLOT_W-1:0] t_slot;
  logic [3:0] t_code;
  logic [LINE_B*8-1:0] t_rdata;
  logic [LINE_B-1:0] t_rval;
  logic t_dirty;
  logic [31:0] cyc;

  localparam logic [3:0] A_RESPOND = 4'd0, A_AFTER_WRITE = 4'd1, A_AFTER_READ_HIT = 4'd2,
                         A_AFTER_READ_CLEAN = 4'd3;

  assign profile_gen_o = profile_gen;
  assign capacity_lines_o = capacity_lines;
  assign write_back_o = write_back;
  assign resident_lines_o = resident_cnt;
  assign dirty_lines_o = dirty_cnt;
  assign pending_cleans_o = pend_cnt;
  assign hits_o = hits;
  assign misses_o = misses;
  assign refills_o = refills;
  assign evictions_o = evictions;
  assign stale_rejects_o = stale_rejects;
  assign maint_busy_o = (state == S_MAINT_CHECK) || (state == S_MAINT_STEP) ||
                        (state == S_MAINT_CLEAN) || (state == S_MAINT_DONE);
  assign invariant_error_o = inv_err;
  assign cmd_ready = rst_n && (state == S_IDLE) && !rsp_valid;
  assign cache_quiescent_o = rst_n && (state == S_IDLE) && !rsp_valid && !clean_valid &&
                             (dirty_cnt == 0) && (pend_cnt == 0);

  function automatic logic slot_live(input logic [SLOT_W-1:0] i);
    return v[i] && (lgen[i] == profile_gen);
  endfunction

  function automatic logic in_range(input logic [SLOT_W-1:0] i, input logic all_lines,
                                    input logic [63:0] ra, input logic [31:0] rg,
                                    input logic [31:0] ro, input logic [31:0] rl);
    return all_lines ||
           (aid[i] == ra && agen[i] == rg && (aoff[i] + 32'(LINE_B)) > ro && aoff[i] < (ro + rl));
  endfunction

  // physical placement of a slot
  function automatic logic [BANK_IDX_W-1:0] slot_bank(input logic [SLOT_W-1:0] s);
    return BANK_IDX_W'(int'(s) % BANK_COUNT);
  endfunction
  function automatic logic [BANK_ADDR_W-1:0] slot_addr(input logic [SLOT_W-1:0] s);
    return BANK_ADDR_W'(SPM_BYTES_PER_BANK[prof_idx] + (int'(s) / BANK_COUNT) * LINE_B);
  endfunction

  always_comb begin
    cp_valid_o = (state == S_BANK_REQ);
    cp_write_o = b_write;
    cp_bank_o = slot_bank(b_slot);
    cp_addr_o = slot_addr(b_slot);
    cp_beats_o = BEAT_W'(BEATS);
    cp_wmode_o = b_wmode;
    cp_wdata_valid_o = (state == S_BANK_WR);
    cp_wdata_o = b_wdata[b_beat*DATA_BYTES*8 +: DATA_BYTES*8];
    cp_wben_o = b_wben[b_beat*DATA_BYTES +: DATA_BYTES];
    cp_rdata_ready_o = (state == S_BANK_RD);
  end


  task automatic go(input state_t s);
    state <= s;
  endtask

  task automatic respond(input logic [3:0] code, input logic [LINE_B*8-1:0] data,
                         input logic [LINE_B-1:0] validity);
    rsp_valid <= 1'b1;
    rsp_code <= code;
    rsp_data <= data;
    rsp_validity <= validity;
    op_cycles_o <= cyc;
    go(S_IDLE);
  endtask

  task automatic start_scan(input logic early);
    s_idx <= '0;
    s_found <= 1'b0; s_free_found <= 1'b0; s_vic_found <= 1'b0; s_any_dirty <= 1'b0;
    s_hit <= '0; s_free <= '0; s_vic <= '0; s_best <= '1;
    s_early <= early;
    go(S_SCAN);
  endtask

  task automatic start_bank(input logic [SLOT_W-1:0] slot, input logic write, input logic wmode,
                            input logic [LINE_B*8-1:0] wdata, input logic [LINE_B-1:0] wben,
                            input logic [3:0] then_a);
    b_slot <= slot; b_write <= write; b_wmode <= wmode; b_wdata <= wdata; b_wben <= wben;
    b_then <= then_a; b_beat <= '0; done_seen <= 1'b0;
    go(S_BANK_REQ);
  endtask

  always_ff @(posedge clk or negedge rst_n) begin
    logic [31:0] base;
    logic [SLOT_W-1:0] ix;
    logic f_found, f_free_found, f_vic_found, f_dirty;
    logic [SLOT_W-1:0] f_hit, f_free, f_vic;
    logic [63:0] f_best;

    if (!rst_n) begin
      rsp_valid <= 1'b0; rsp_code <= '0; rsp_data <= '0; rsp_validity <= '0;
      clean_valid <= 1'b0; clean_id <= '0; clean_alloc_id <= '0; clean_alloc_gen <= '0;
      clean_line_offset <= '0; clean_data <= '0; clean_profile_gen <= '0;
      profile_gen <= '0; capacity_lines <= '0; next_clean_id <= 32'd1; prof_idx <= '0;
      write_back <= 1'b0;
      touch <= 64'd1; hits <= '0; misses <= '0; refills <= '0; evictions <= '0;
      stale_rejects <= '0; inv_err <= 1'b0;
      resident_cnt <= '0; dirty_cnt <= '0; pend_cnt <= '0;
      go(S_IDLE); cyc <= '0; op_cycles_o <= '0;
      c_op <= '0; c_aid <= '0; c_agen <= '0; c_off <= '0; c_pgen <= '0; c_rng_off <= '0;
      c_rng_len <= '0; c_cid <= '0; c_data <= '0; c_val <= '0; c_dirty <= 1'b0; c_all <= 1'b0;
      c_clean <= 1'b0; c_inv <= 1'b0; c_cok <= 1'b0; c_wb <= 1'b0; c_pidx <= '0;
      s_idx <= '0; s_found <= 1'b0; s_free_found <= 1'b0; s_vic_found <= 1'b0; s_any_dirty <= 1'b0;
      s_hit <= '0; s_free <= '0; s_vic <= '0; s_best <= '0; s_early <= 1'b0;
      b_write <= 1'b0; b_wmode <= 1'b0; b_slot <= '0; b_then <= '0; b_beat <= '0;
      b_wdata <= '0; b_wben <= '0; b_rdata <= '0; b_rben <= '0; done_seen <= 1'b0;
      t_slot <= '0; t_code <= '0; t_rdata <= '0; t_rval <= '0; t_dirty <= 1'b0;
      for (int i = 0; i < PHYS; i++) begin
        v[i] <= 1'b0; lgen[i] <= '0; aid[i] <= '0; agen[i] <= '0; aoff[i] <= '0;
        dirty[i] <= 1'b0; pend[i] <= 1'b0; inv_after[i] <= 1'b0; pid[i] <= '0; stamp[i] <= '0;
      end
    end else begin
      if (rsp_valid && rsp_ready) rsp_valid <= 1'b0;
      if (clean_valid && clean_ready) clean_valid <= 1'b0;
      if (state != S_IDLE) cyc <= cyc + 32'd1;
      if (cp_done_i) done_seen <= 1'b1;

      case (state)
        S_IDLE: begin
          if (cmd_valid && cmd_ready) begin
            c_op <= cmd_op; c_aid <= cmd_alloc_id; c_agen <= cmd_alloc_gen; c_off <= cmd_line_offset;
            c_pgen <= cmd_profile_gen; c_data <= cmd_data; c_val <= cmd_validity; c_dirty <= cmd_dirty;
            c_all <= cmd_maint_all; c_clean <= cmd_maint_clean; c_inv <= cmd_maint_invalidate;
            c_rng_off <= cmd_range_off; c_rng_len <= cmd_range_len; c_cid <= cmd_clean_id;
            c_cok <= cmd_clean_ok; c_wb <= cmd_write_back; c_pidx <= cmd_profile_idx;
            cyc <= 32'd1;
            case (cmd_op)
              OP_LOOKUP: begin
                if (capacity_lines == 0) respond(R_DISABLED, '0, '0);
                else if (bank_profile_i != prof_idx) respond(R_PROFILE_SKEW, '0, '0);
                else if (cmd_line_offset % 32'(LINE_B) != 0) respond(R_BAD_CMD, '0, '0);
                else start_scan(1'b1);
              end
              OP_REFILL: begin
                if (capacity_lines == 0) respond(R_DISABLED, '0, '0);
                else if (cmd_profile_gen != profile_gen) begin
                  stale_rejects <= stale_rejects + 64'd1;
                  respond(R_STALE_PROFILE, '0, '0);
                end else if (bank_profile_i != prof_idx) respond(R_PROFILE_SKEW, '0, '0);
                else if (cmd_line_offset % 32'(LINE_B) != 0) respond(R_BAD_CMD, '0, '0);
                else if (cmd_dirty && !write_back) respond(R_READ_ONLY, '0, '0);
                else start_scan(1'b0);
              end
              OP_MARK_DIRTY: begin
                if (!write_back) respond(R_READ_ONLY, '0, '0);
                else if (bank_profile_i != prof_idx) respond(R_PROFILE_SKEW, '0, '0);
                else start_scan(1'b1);
              end
              OP_MAINT: begin
                if (cmd_profile_gen != profile_gen) begin
                  stale_rejects <= stale_rejects + 64'd1;
                  respond(R_STALE_PROFILE, '0, '0);
                end else if (bank_profile_i != prof_idx) respond(R_PROFILE_SKEW, '0, '0);
                else begin
                  s_idx <= '0; s_any_dirty <= 1'b0; go(S_MAINT_CHECK);
                end
              end
              OP_COMPLETE_CLEAN: begin
                // Straight to the id walk: no identity scan is meaningful for a clean ACK.
                s_idx <= '0;
                go(S_MAINT_STEP);
              end
              OP_RECONFIG: begin
                if (cmd_profile_gen <= profile_gen) respond(R_STALE_PROFILE, '0, '0);
                else if (dirty_cnt != 0 || pend_cnt != 0) respond(R_BUSY, '0, '0);
                else if (bank_profile_i != cmd_profile_idx) respond(R_PROFILE_SKEW, '0, '0);
                else begin
                  // Every line dies with the old generation (O(1) drop).
                  profile_gen <= cmd_profile_gen;
                  prof_idx <= cmd_profile_idx;
                  capacity_lines <= 32'(BANK_COUNT * (CACHE_BYTES_PER_BANK[cmd_profile_idx] / LINE_B));
                  write_back <= cmd_write_back;
                  resident_cnt <= '0;
                  respond(R_OK, '0, '0);
                end
              end
              default: respond(R_BAD_OP, '0, '0);
            endcase
          end
        end

        // ---- bounded multi-cycle tag/free/LRU scan, SCAN_LANES slots per cycle ----
        S_SCAN: begin
          // Multi-lane scan: lane results fold through BLOCKING block-locals so a later lane
          // sees the earlier lanes of the SAME cycle (first free slot wins, minimum stamp
          // wins with the lower index on a tie), then commit once with nonblocking writes.
          f_found = s_found; f_hit = s_hit;
          f_free_found = s_free_found; f_free = s_free;
          f_vic_found = s_vic_found; f_vic = s_vic; f_best = s_best;
          for (int l = 0; l < SCAN_LANES; l++) begin
            base = s_idx + 32'(l);
            ix = SLOT_W'(base);
            if (base < capacity_lines) begin
              if (slot_live(ix)) begin
                if (aid[ix] == c_aid && agen[ix] == c_agen && aoff[ix] == c_off && !f_found) begin
                  f_found = 1'b1; f_hit = ix;
                end
                if (!f_vic_found || stamp[ix] < f_best) begin
                  f_vic_found = 1'b1; f_vic = ix; f_best = stamp[ix];
                end
              end else if (!f_free_found) begin
                f_free_found = 1'b1; f_free = ix;
              end
            end
          end
          s_found <= f_found; s_hit <= f_hit;
          s_free_found <= f_free_found; s_free <= f_free;
          s_vic_found <= f_vic_found; s_vic <= f_vic; s_best <= f_best;
          // Early exit once the identity is found; otherwise finish at capacity.
          if (s_idx + 32'(SCAN_LANES) >= capacity_lines) go(S_DISPATCH);
          else if (s_early && f_found) go(S_DISPATCH);
          else s_idx <= s_idx + 32'(SCAN_LANES);
        end

        S_DISPATCH: begin
          case (c_op)
            OP_LOOKUP: begin
              if (!s_found) begin
                misses <= misses + 64'd1;
                respond(R_MISS, '0, '0);
              end else begin
                hits <= hits + 64'd1;
                stamp[s_hit] <= touch; touch <= touch + 64'd1;
                t_slot <= s_hit;
                start_bank(s_hit, 1'b0, 1'b0, '0, '0, A_AFTER_READ_HIT);
              end
            end
            OP_REFILL: begin
              if (s_found) begin
                if (pend[s_hit]) respond(R_PENDING, '0, '0);
                else begin
                  t_slot <= s_hit;
                  start_bank(s_hit, 1'b1, 1'b0, c_data, c_val, A_AFTER_WRITE);
                end
              end else if (resident_cnt >= capacity_lines) begin
                if (!s_vic_found) begin
                  inv_err <= 1'b1;
                  respond(R_BAD_CMD, '0, '0);
                end else if (dirty[s_vic] || pend[s_vic]) respond(R_EVICT_DIRTY, '0, '0);
                else begin
                  t_slot <= s_vic;
                  start_bank(s_vic, 1'b1, 1'b0, c_data, c_val, A_AFTER_WRITE);
                end
              end else if (!s_free_found) begin
                inv_err <= 1'b1;
                respond(R_BAD_CMD, '0, '0);
              end else begin
                t_slot <= s_free;
                start_bank(s_free, 1'b1, 1'b0, c_data, c_val, A_AFTER_WRITE);
              end
            end
            OP_MARK_DIRTY: begin
              if (!s_found) respond(R_NOT_RESIDENT, '0, '0);
              else if (pend[s_hit]) respond(R_PENDING, '0, '0);
              else begin
                t_slot <= s_hit;
                start_bank(s_hit, 1'b1, 1'b1, c_data, '1, A_AFTER_WRITE);
              end
            end
            OP_COMPLETE_CLEAN: begin
              // Not reachable: OP_COMPLETE_CLEAN skips the identity scan (see S_IDLE).
              respond(R_BAD_OP, '0, '0);
            end
            default: respond(R_BAD_OP, '0, '0);
          endcase
        end

        // ---- bank access: present the request, stream beats, wait for the done pulse ----
        S_BANK_REQ: begin
          if (cp_rej_i) begin
            inv_err <= 1'b1;
            respond(R_BANK_FAULT, '0, '0);
          end else if (cp_ready_i) begin
            b_beat <= '0;
            go(b_write ? S_BANK_WR : S_BANK_RD);
          end
        end
        S_BANK_WR: begin
          if (cp_wdata_ready_i) begin
            if (b_beat == BEAT_W'(BEATS - 1)) go(S_BANK_DONE);
            else b_beat <= b_beat + BEAT_W'(1);
          end
        end
        S_BANK_RD: begin
          if (cp_rdata_valid_i) begin
            b_rdata[b_beat*DATA_BYTES*8 +: DATA_BYTES*8] <= cp_rdata_i;
            b_rben[b_beat*DATA_BYTES +: DATA_BYTES] <= cp_rben_i;
            if (b_beat == BEAT_W'(BEATS - 1)) go(S_BANK_DONE);
            else b_beat <= b_beat + BEAT_W'(1);
          end
        end
        S_BANK_DONE: begin
          if (done_seen || cp_done_i) begin
            go(S_FINISH);
          end
        end

        S_FINISH: begin
          case (b_then)
            A_AFTER_READ_HIT: respond(R_HIT, b_rdata, b_rben);
            A_AFTER_READ_CLEAN: begin
              // The dirty line's bytes are in b_rdata: emit the writeback request.
              clean_valid <= 1'b1;
              clean_id <= next_clean_id;
              clean_alloc_id <= aid[t_slot];
              clean_alloc_gen <= agen[t_slot];
              clean_line_offset <= aoff[t_slot];
              clean_data <= b_rdata;
              clean_profile_gen <= profile_gen;
              pend[t_slot] <= 1'b1;
              pid[t_slot] <= next_clean_id;
              inv_after[t_slot] <= c_inv;
              pend_cnt <= pend_cnt + 32'd1;
              next_clean_id <= next_clean_id + 32'd1;
              s_idx <= s_idx + 32'd1;
              go(S_MAINT_CLEAN);
            end
            default: begin // A_AFTER_WRITE: commit the tag now that the bytes are in the bank
              t_dirty = c_dirty;
              if (c_op == OP_MARK_DIRTY) begin
                if (!dirty[t_slot]) dirty_cnt <= dirty_cnt + 32'd1;
                dirty[t_slot] <= 1'b1;
                stamp[t_slot] <= touch; touch <= touch + 64'd1;
                respond(R_OK, '0, '0);
              end else begin
                if (slot_live(t_slot)) begin
                  if (s_found) begin
                    // refill of a resident line: dirty only ever sets
                    if (c_dirty && !dirty[t_slot]) begin
                      dirty[t_slot] <= 1'b1; dirty_cnt <= dirty_cnt + 32'd1;
                    end
                  end else begin
                    // victim replaced: it was clean and not pending
                    v[t_slot] <= 1'b1; lgen[t_slot] <= profile_gen;
                    aid[t_slot] <= c_aid; agen[t_slot] <= c_agen; aoff[t_slot] <= c_off;
                    dirty[t_slot] <= c_dirty; pend[t_slot] <= 1'b0; inv_after[t_slot] <= 1'b0;
                    if (c_dirty) dirty_cnt <= dirty_cnt + 32'd1;
                    evictions <= evictions + 64'd1;
                  end
                end else begin
                  v[t_slot] <= 1'b1; lgen[t_slot] <= profile_gen;
                  aid[t_slot] <= c_aid; agen[t_slot] <= c_agen; aoff[t_slot] <= c_off;
                  dirty[t_slot] <= c_dirty; pend[t_slot] <= 1'b0; inv_after[t_slot] <= 1'b0;
                  resident_cnt <= resident_cnt + 32'd1;
                  if (c_dirty) dirty_cnt <= dirty_cnt + 32'd1;
                end
                stamp[t_slot] <= touch; touch <= touch + 64'd1;
                refills <= refills + 64'd1;
                respond(R_OK, '0, '0);
              end
            end
          endcase
        end

        // ---- maintenance: pass 1 checks, pass 2 walks one slot per step ----
        S_MAINT_CHECK: begin
          f_dirty = s_any_dirty;
          for (int l = 0; l < SCAN_LANES; l++) begin
            base = s_idx + 32'(l);
            ix = SLOT_W'(base);
            if (base < capacity_lines)
              if (slot_live(ix) && dirty[ix] && !pend[ix] &&
                  in_range(ix, c_all, c_aid, c_agen, c_rng_off, c_rng_len))
                f_dirty = 1'b1;
          end
          s_any_dirty <= f_dirty;
          if (s_idx + 32'(SCAN_LANES) >= capacity_lines) begin
            s_idx <= '0;
            go(S_MAINT_DONE);
          end else s_idx <= s_idx + 32'(SCAN_LANES);
        end

        S_MAINT_DONE: begin
          if (c_op == OP_MAINT && s_idx == 0 && s_any_dirty && !c_clean) begin
            respond(R_DIRTY_NO_CLEAN, '0, '0);
          end else begin
            s_idx <= '0;
            go(S_MAINT_STEP);
          end
        end

        S_MAINT_STEP: begin
          if (c_op == OP_COMPLETE_CLEAN) begin
            // locate the pending clean by id, one slot per cycle
            if (s_idx >= capacity_lines) respond(R_UNKNOWN_CLEAN, '0, '0);
            else begin
              ix = SLOT_W'(s_idx);
              if (slot_live(ix) && pend[ix] && pid[ix] == c_cid) begin
                pend[ix] <= 1'b0; pend_cnt <= pend_cnt - 32'd1;
                if (c_cok) begin
                  dirty[ix] <= 1'b0; dirty_cnt <= dirty_cnt - 32'd1;
                  if (inv_after[ix]) begin
                    v[ix] <= 1'b0; resident_cnt <= resident_cnt - 32'd1;
                  end
                end
                inv_after[ix] <= 1'b0;
                respond(R_OK, '0, '0);
              end else s_idx <= s_idx + 32'd1;
            end
          end else if (s_idx >= capacity_lines) begin
            respond(R_OK, '0, '0);
          end else begin
            ix = SLOT_W'(s_idx);
            if (slot_live(ix) && in_range(ix, c_all, c_aid, c_agen, c_rng_off, c_rng_len) &&
                !pend[ix]) begin
              if (dirty[ix]) begin
                if (c_clean && !clean_valid) begin
                  t_slot <= ix;
                  start_bank(ix, 1'b0, 1'b0, '0, '0, A_AFTER_READ_CLEAN);
                end
              end else begin
                // invalidate_range (clean=false, invalidate=true) must still drop CLEAN lines
                // (cache.py:499: `elif invalidate: remove`); only dirty lines need the clean.
                if (c_inv) begin
                  v[ix] <= 1'b0; resident_cnt <= resident_cnt - 32'd1;
                end
                s_idx <= s_idx + 32'd1;
              end
            end else s_idx <= s_idx + 32'd1;
          end
        end

        S_MAINT_CLEAN: begin
          // The clean request was presented in S_FINISH; resume the walk.
          go(S_MAINT_STEP);
        end

        default: go(S_IDLE);
      endcase
    end
  end

  // Live invariants over the counters (updated with every transition), so the check is
  // O(1) per cycle regardless of the aggregate line count.
  always @(posedge clk) begin
    if (rst_n) begin
      if (resident_cnt > capacity_lines)
        $error("C030_ASSERT cache residency %0d exceeds capacity %0d", resident_cnt, capacity_lines);
      if (dirty_cnt > resident_cnt)
        $error("C030_ASSERT cache dirty %0d exceeds resident %0d", dirty_cnt, resident_cnt);
      if (pend_cnt > dirty_cnt)
        $error("C030_ASSERT cache pending cleans %0d exceed dirty lines %0d", pend_cnt, dirty_cnt);
      if (dirty_cnt != 0 && !write_back)
        $error("C030_ASSERT cache dirty lines under a read-only policy");
    end
  end
endmodule
