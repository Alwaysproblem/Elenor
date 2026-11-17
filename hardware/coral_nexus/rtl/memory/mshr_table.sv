// mshr_table.sv — C030 L1 MSHR with entry-owned fill buffers (W4 plan §4.4).
//
// Semantics sources: pipeline_validator/memory/mshr.py (merge by
// (profile generation, key), capacity stall, cancel, reconfigure) and the
// stronger RTL contract of protocol/w4_memory.contract.yaml `mshr:`:
//   * entry states ALLOCATED -> ISSUED_REFILL -> WAITERS_PENDING -> FREE;
//   * the line fill buffer (data + per-byte validity) lives in the ENTRY, so a
//     cache eviction can never destroy waiter data;
//   * an entry becomes FREE only when every registered waiter has received its
//     data/terminal response (CONSUME) AND every token reference was released
//     (RELEASE). A slot is therefore never reused early;
//   * a reused slot gets a fresh per-slot `entry_gen`, so an old token
//     (slot, old gen) is rejected as BAD_TOKEN and never reads new bytes;
//   * entries of different profile generations never merge, and a refill
//     carrying an old profile generation is rejected and counted (MEM-13).
//
// ENTRIES and LINE_BYTES come from the sealed C013/C019 sources via
// c029_params_pkg. WAITERS (waiters per refill entry) has NO sourced bound
// (w4_memory_constraints.yaml response_record_budget is non-derivable): it is a
// bench/integration parameter, and a merge beyond it is refused with
// WAITERS_FULL (never dropped, never silently merged). KEY_W / GEN_W / EGEN_W
// are transport containers, not frozen silicon widths; entry-gen exhaustion
// is refused with GEN_EXHAUSTED rather than wrapped.
//
// One command per cycle; the response is registered and held until
// rsp_ready; cmd_ready is low while a response is pending.
`timescale 1ns/1ps
module mshr_table
  import c029_params_pkg::*;
#(
  parameter int unsigned ENTRIES = MSHR_ENTRIES,
  parameter int unsigned LINE_B = LINE_BYTES,
  parameter int unsigned WAITERS = 4,
  parameter int unsigned KEY_W = 64,
  parameter int unsigned GEN_W = 32,
  parameter int unsigned EGEN_W = 32
) (
  input  logic clk,
  input  logic rst_n,

  input  logic cmd_valid,
  output logic cmd_ready,
  input  logic [2:0] cmd_op,
  input  logic [KEY_W-1:0] cmd_key,
  input  logic [GEN_W-1:0] cmd_profile_gen,
  input  logic [$clog2(ENTRIES)-1:0] cmd_slot,
  input  logic [EGEN_W-1:0] cmd_entry_gen,
  input  logic [$clog2(WAITERS)-1:0] cmd_waiter,
  input  logic [LINE_B*8-1:0] cmd_data,
  input  logic [LINE_B-1:0] cmd_mask,
  input  logic cmd_last,

  output logic rsp_valid,
  input  logic rsp_ready,
  output logic [3:0] rsp_code,
  output logic [$clog2(ENTRIES)-1:0] rsp_slot,
  output logic [EGEN_W-1:0] rsp_entry_gen,
  output logic [$clog2(WAITERS)-1:0] rsp_waiter,
  output logic [LINE_B*8-1:0] rsp_data,
  output logic [LINE_B-1:0] rsp_mask,

  output logic [$clog2(ENTRIES+1)-1:0] active_o,
  output logic [63:0] merged_o,
  output logic [63:0] stalls_o,
  output logic [63:0] stale_rejects_o,
  output logic [GEN_W-1:0] profile_gen_o,
  output logic [ENTRIES-1:0] entry_busy_o,
  // C031 read-only readiness views: a waiter CONSUME succeeds on
  // entry_filled_o (data ready) or entry_cancelled_o (terminal no-data); a
  // slot with refill_pend_o still owes its external TERMINAL_ACK. The settle
  // pulse is the one-cycle OP_TERMINAL accept that closes the obligation of
  // exactly (settle_slot_o, settle_gen_o).
  output logic [ENTRIES-1:0] entry_filled_o,
  output logic [ENTRIES-1:0] entry_cancelled_o,
  output logic [ENTRIES-1:0] refill_pend_o,
  output logic settle_valid_o,
  output logic [$clog2(ENTRIES)-1:0] settle_slot_o,
  output logic [EGEN_W-1:0] settle_gen_o,
  output logic invariant_error_o
);
  localparam int unsigned SLOT_W = $clog2(ENTRIES);
  localparam int unsigned WAIT_W = $clog2(WAITERS);

  localparam logic [2:0] OP_ALLOC = 3'd0, OP_ISSUE = 3'd1, OP_FILL = 3'd2, OP_CONSUME = 3'd3,
                         OP_RELEASE = 3'd4, OP_CANCEL = 3'd5, OP_RECONFIG = 3'd6, OP_TERMINAL = 3'd7;
  localparam logic [3:0] R_OK = 4'd0, R_LEADER = 4'd1, R_MERGED = 4'd2, R_FULL = 4'd3,
                         R_WAITERS_FULL = 4'd4, R_STALE_PROFILE = 4'd5, R_BAD_TOKEN = 4'd6,
                         R_BAD_STATE = 4'd7, R_NOT_FILLED = 4'd8, R_CANCELLED = 4'd9,
                         R_NOT_DELIVERED = 4'd10, R_ALREADY = 4'd11, R_GEN_EXHAUSTED = 4'd12,
                         R_ACTIVE = 4'd13, R_BAD_OP = 4'd14;
  localparam logic [1:0] S_FREE = 2'd0, S_ALLOCATED = 2'd1, S_ISSUED = 2'd2, S_PENDING = 2'd3;
  localparam logic [1:0] W_NONE = 2'd0, W_REG = 2'd1, W_DELIVERED = 2'd2, W_CLOSED = 2'd3;

  logic [1:0] state [ENTRIES];
  logic cancelled [ENTRIES];
  // An ISSUED refill is an accepted external obligation with a service-side
  // terminal reference. It stays outstanding until the explicit terminal ACK
  // settles it (OP_TERMINAL) - for successful refills too. The final FILL beat
  // and CANCEL never settle it (C013 §8 / W4 §4.4 / C030: no token reuse before
  // every waiter, response and terminal has closed).
  logic refill_out [ENTRIES];
  logic [KEY_W-1:0] key [ENTRIES];
  logic [GEN_W-1:0] egen_profile [ENTRIES];
  logic [EGEN_W-1:0] egen [ENTRIES];
  logic [1:0] wstate [ENTRIES][WAITERS];
  logic [LINE_B*8-1:0] fbuf [ENTRIES];
  logic [LINE_B-1:0] fmask [ENTRIES];
  logic [GEN_W-1:0] profile_gen;
  logic [63:0] merged, stalls, stale_rejects;
  logic inv_err;

  assign profile_gen_o = profile_gen;
  assign merged_o = merged;
  assign stalls_o = stalls;
  assign stale_rejects_o = stale_rejects;
  assign invariant_error_o = inv_err;
  assign cmd_ready = rst_n && !rsp_valid;

  // C031 readiness masks (read-only views of the entry FSM).
  always_comb begin
    for (int e = 0; e < ENTRIES; e++) begin
      entry_filled_o[e] = (state[e] == S_PENDING) && !cancelled[e];
      entry_cancelled_o[e] = cancelled[e] && (state[e] != S_FREE);
      refill_pend_o[e] = refill_out[e];
    end
  end

  always_comb begin
    int n;
    n = 0;
    for (int e = 0; e < ENTRIES; e++) begin
      entry_busy_o[e] = state[e] != S_FREE;
      if (state[e] != S_FREE) n++;
    end
    active_o = n[$clog2(ENTRIES+1)-1:0];
  end

  // A slot may return to FREE only when the entry is terminal (filled or
  // cancelled), no accepted refill is still outstanding, and every registered
  // waiter is both delivered and released.
  function automatic logic can_free(input logic [1:0] st, input logic canc, input logic refill,
                                    input logic [1:0] ws [WAITERS]);
    logic ok;
    ok = (canc || st == S_PENDING) && !refill;
    for (int w = 0; w < WAITERS; w++)
      if (ws[w] != W_NONE && ws[w] != W_CLOSED) ok = 1'b0;
    return ok;
  endfunction

  task automatic respond(input logic [3:0] code, input logic [SLOT_W-1:0] slot,
                         input logic [EGEN_W-1:0] eg, input logic [WAIT_W-1:0] wt,
                         input logic [LINE_B*8-1:0] data, input logic [LINE_B-1:0] mask);
    rsp_valid <= 1'b1;
    rsp_code <= code;
    rsp_slot <= slot;
    rsp_entry_gen <= eg;
    rsp_waiter <= wt;
    rsp_data <= data;
    rsp_mask <= mask;
  endtask

  always_ff @(posedge clk or negedge rst_n) begin
    logic found, have_free, w_found;
    logic [SLOT_W-1:0] hit, freeslot;
    logic [WAIT_W-1:0] wfree;
    logic [1:0] ws [WAITERS];
    logic [SLOT_W-1:0] s;
    if (!rst_n) begin
      rsp_valid <= 1'b0;
      rsp_code <= '0; rsp_slot <= '0; rsp_entry_gen <= '0; rsp_waiter <= '0;
      rsp_data <= '0; rsp_mask <= '0;
      settle_valid_o <= 1'b0; settle_slot_o <= '0; settle_gen_o <= '0;
      profile_gen <= '0;
      merged <= '0; stalls <= '0; stale_rejects <= '0;
      inv_err <= 1'b0;
      for (int e = 0; e < ENTRIES; e++) begin
        state[e] <= S_FREE; cancelled[e] <= 1'b0; refill_out[e] <= 1'b0; key[e] <= '0; egen_profile[e] <= '0;
        egen[e] <= '0; fbuf[e] <= '0; fmask[e] <= '0;
        for (int w = 0; w < WAITERS; w++) wstate[e][w] <= W_NONE;
      end
    end else begin
      if (rsp_valid && rsp_ready) rsp_valid <= 1'b0;
      settle_valid_o <= 1'b0;
      if (cmd_valid && cmd_ready) begin
        s = cmd_slot;
        for (int w = 0; w < WAITERS; w++) ws[w] = wstate[s][w];
        case (cmd_op)
          OP_ALLOC: begin
            found = 1'b0; hit = '0; have_free = 1'b0; freeslot = '0;
            for (int e = 0; e < ENTRIES; e++) begin
              if (state[e] != S_FREE && !cancelled[e] && key[e] == cmd_key &&
                  egen_profile[e] == profile_gen && !found) begin
                found = 1'b1; hit = SLOT_W'(e);
              end
              if (state[e] == S_FREE && !have_free) begin
                have_free = 1'b1; freeslot = SLOT_W'(e);
              end
            end
            if (cmd_profile_gen != profile_gen) begin
              stale_rejects <= stale_rejects + 64'd1;
              respond(R_STALE_PROFILE, '0, '0, '0, '0, '0);
            end else if (found) begin
              w_found = 1'b0; wfree = '0;
              for (int w = 0; w < WAITERS; w++)
                if (wstate[hit][w] == W_NONE && !w_found) begin w_found = 1'b1; wfree = WAIT_W'(w); end
              if (!w_found) begin
                stalls <= stalls + 64'd1;
                respond(R_WAITERS_FULL, hit, egen[hit], '0, '0, '0);
              end else begin
                wstate[hit][wfree] <= W_REG;
                merged <= merged + 64'd1;
                respond(R_MERGED, hit, egen[hit], wfree, '0, '0);
              end
            end else if (!have_free) begin
              stalls <= stalls + 64'd1;
              respond(R_FULL, '0, '0, '0, '0, '0);
            end else if (&egen[freeslot]) begin
              respond(R_GEN_EXHAUSTED, freeslot, egen[freeslot], '0, '0, '0);
            end else begin
              // Only a slot that is S_FREE can be chosen above; a violated
              // rule is an RTL bug, so it is also checked here.
              if (state[freeslot] != S_FREE) inv_err <= 1'b1;
              state[freeslot] <= S_ALLOCATED;
              cancelled[freeslot] <= 1'b0;
              refill_out[freeslot] <= 1'b0;
              key[freeslot] <= cmd_key;
              egen_profile[freeslot] <= profile_gen;
              egen[freeslot] <= egen[freeslot] + EGEN_W'(1);
              fmask[freeslot] <= '0;
              fbuf[freeslot] <= '0;
              for (int w = 0; w < WAITERS; w++) wstate[freeslot][w] <= (w == 0) ? W_REG : W_NONE;
              respond(R_LEADER, freeslot, egen[freeslot] + EGEN_W'(1), '0, '0, '0);
            end
          end
          OP_ISSUE: begin
            if (state[s] == S_FREE || egen[s] != cmd_entry_gen) respond(R_BAD_TOKEN, s, cmd_entry_gen, '0, '0, '0);
            else if (cancelled[s] || state[s] != S_ALLOCATED) respond(R_BAD_STATE, s, egen[s], '0, '0, '0);
            else begin
              state[s] <= S_ISSUED;
              refill_out[s] <= 1'b1;
              respond(R_OK, s, egen[s], '0, '0, '0);
            end
          end
          OP_FILL: begin
            if (state[s] == S_FREE || egen[s] != cmd_entry_gen) respond(R_BAD_TOKEN, s, cmd_entry_gen, '0, '0, '0);
            else if (egen_profile[s] != profile_gen) begin
              // Old-generation refill: counted, never written (MEM-13).
              stale_rejects <= stale_rejects + 64'd1;
              respond(R_STALE_PROFILE, s, egen[s], '0, '0, '0);
            end else if (cancelled[s] || state[s] != S_ISSUED) respond(R_BAD_STATE, s, egen[s], '0, '0, '0);
            else begin
              for (int b = 0; b < LINE_B; b++)
                if (cmd_mask[b]) fbuf[s][b*8 +: 8] <= cmd_data[b*8 +: 8];
              fmask[s] <= fmask[s] | cmd_mask;
              if (cmd_last) state[s] <= S_PENDING;
              respond(R_OK, s, egen[s], '0, '0, '0);
            end
          end
          OP_CONSUME: begin
            if (state[s] == S_FREE || egen[s] != cmd_entry_gen || ws[cmd_waiter] == W_NONE)
              respond(R_BAD_TOKEN, s, cmd_entry_gen, cmd_waiter, '0, '0);
            else if (ws[cmd_waiter] != W_REG) respond(R_ALREADY, s, egen[s], cmd_waiter, '0, '0);
            else if (cancelled[s]) begin
              // Terminal cancelled response carries no data.
              wstate[s][cmd_waiter] <= W_DELIVERED;
              respond(R_CANCELLED, s, egen[s], cmd_waiter, '0, '0);
            end else if (state[s] != S_PENDING) respond(R_NOT_FILLED, s, egen[s], cmd_waiter, '0, '0);
            else begin
              wstate[s][cmd_waiter] <= W_DELIVERED;
              respond(R_OK, s, egen[s], cmd_waiter, fbuf[s], fmask[s]);
            end
          end
          OP_RELEASE: begin
            if (state[s] == S_FREE || egen[s] != cmd_entry_gen || ws[cmd_waiter] == W_NONE)
              respond(R_BAD_TOKEN, s, cmd_entry_gen, cmd_waiter, '0, '0);
            else if (ws[cmd_waiter] == W_CLOSED) respond(R_ALREADY, s, egen[s], cmd_waiter, '0, '0);
            else if (ws[cmd_waiter] != W_DELIVERED) respond(R_NOT_DELIVERED, s, egen[s], cmd_waiter, '0, '0);
            else begin
              ws[cmd_waiter] = W_CLOSED;
              wstate[s][cmd_waiter] <= W_CLOSED;
              if (can_free(state[s], cancelled[s], refill_out[s], ws)) begin
                state[s] <= S_FREE;
                for (int w = 0; w < WAITERS; w++) wstate[s][w] <= W_NONE;
              end
              respond(R_OK, s, egen[s], cmd_waiter, '0, '0);
            end
          end
          OP_CANCEL: begin
            if (state[s] == S_FREE || egen[s] != cmd_entry_gen) respond(R_BAD_TOKEN, s, cmd_entry_gen, '0, '0, '0);
            else if (egen_profile[s] != profile_gen) begin
              stale_rejects <= stale_rejects + 64'd1;
              respond(R_STALE_PROFILE, s, egen[s], '0, '0, '0);
            end else begin
              cancelled[s] <= 1'b1;
              respond(R_OK, s, egen[s], '0, '0, '0);
            end
          end
          OP_TERMINAL: begin
            // Outbridge terminal/isolation ACK for an accepted refill.
            if (state[s] == S_FREE || egen[s] != cmd_entry_gen) respond(R_BAD_TOKEN, s, cmd_entry_gen, '0, '0, '0);
            else if (!refill_out[s] || !(cancelled[s] || state[s] == S_PENDING))
              respond(R_BAD_STATE, s, egen[s], '0, '0, '0);
            else begin
              refill_out[s] <= 1'b0;
              settle_valid_o <= 1'b1;
              settle_slot_o <= s;
              settle_gen_o <= egen[s];
              if (can_free(state[s], cancelled[s], 1'b0, ws)) begin
                state[s] <= S_FREE;
                for (int w = 0; w < WAITERS; w++) wstate[s][w] <= W_NONE;
              end
              respond(R_OK, s, egen[s], '0, '0, '0);
            end
          end
          OP_RECONFIG: begin
            if (cmd_profile_gen <= profile_gen) respond(R_STALE_PROFILE, '0, '0, '0, '0, '0);
            else if (active_o != 0) respond(R_ACTIVE, '0, '0, '0, '0, '0);
            else begin
              profile_gen <= cmd_profile_gen;
              respond(R_OK, '0, '0, '0, '0, '0);
            end
          end
          default: respond(R_BAD_OP, '0, '0, '0, '0, '0);
        endcase
      end
    end
  end
  // Live invariant: a FREE slot holds no registered waiter (checked in the
  // clock domain, separately from the command FSM so it cannot mask itself).
  always @(posedge clk) begin
    if (rst_n) begin
      for (int e = 0; e < ENTRIES; e++) begin
        if (state[e] == S_FREE) begin
          for (int w = 0; w < WAITERS; w++)
            if (wstate[e][w] != W_NONE)
              $error("C030_ASSERT mshr_table free slot %0d holds waiter %0d", e, w);
          if (refill_out[e])
            $error("C030_ASSERT mshr_table free slot %0d holds an outstanding refill", e);
        end
      end
    end
  end
endmodule
