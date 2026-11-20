// wait_record.sv — C033 one persistent WAIT/WAITALL record per Context (W6-02).
//
// This module IS the single writer of its Context's wait record (C015: one
// record per Context, writer = W6). It owns the record registers only; the
// owning event_table resolves keys against the real persistent table and
// supplies, per operand, an EFFECTIVE status (committed state overlaid with
// the completion accepted in the same cycle), so check+register is atomic at
// one point and a same-cycle completion always wins over parking
// ("completion first", never park-and-wake-in-one).
//
// Issue semantics (inputs valid-qualified, decided in the presented cycle):
//   - fire requires: not busy (record free) AND every enabled operand
//     resolved OK (issue_res_ok_i, table-side classification) AND no
//     matching context fault on this edge (see fault below).
//   - any enabled operand in an effective terminal-failure state
//     (ERROR/TIMEOUT/RESET) => direct_fail_o, fail exit immediately, no park
//     (WAITALL does not wait for the remaining operands).
//   - all enabled operands terminal DONE => direct_ok_o (no park, no later
//     wake).
//   - otherwise park_o: the record is written in one edge (keys, waiter
//     generation, count) and stays armed until a whole-record verdict; an
//     armed record is never overwritten (busy).
//
// Wake while parked: on a cycle where a completion is accepted (eval_cycle_i)
// the stored keys are probed; any operand terminal-failure => wake_fail_o,
// all operands DONE => wake_ok_o, exactly one pulse, on the accepted
// completion cycle. Wake clears armed; the waiter's pins (one consumer_ref
// per stored operand, taken at park) are released via release_o in the same
// edge — all-or-none per record. WAIT and WAITALL share the verdict rule
// (any failure fails, all DONE succeeds), so no kind bit is stored.
//
// Fault: a fault routed to this record voids it — an armed record whose
// generation matches (armed cleared, ALL pins released atomically, no wake
// of either kind; the faulted Context settles through its own fault path),
// and equally a registration issued on the same edge whose waiter
// generation matches (the issue is voided before it fires: not accepted, no
// park, no pins, no verdict). A fault with a non-matching generation leaves
// a legitimate issue and any armed record untouched. Fault wins over a wake
// on the same edge.
//
// A pinned entry can never be reclaimed under the record (consumer_refs > 0
// blocks the table's reclaim), so a stored probe always finds its entry live.
`timescale 1ns/1ps
module wait_record
  import c033_event_pkg::*;
#(
  parameter int unsigned OPERANDS,  // explicit at instantiation (bench fixture value)
  localparam int unsigned CNT_W = $clog2(OPERANDS + 1)
) (
  input  logic clk_i,
  input  logic rst_ni,

  // ---- issue: atomic check + register, routed by the owning event_table ----
  input  logic                      issue_valid_i,
  input  word_t                     issue_gen_i,
  input  word_t                     issue_count_i,
  input  event_key_t [OPERANDS-1:0] issue_keys_i,
  input  logic [OPERANDS-1:0]       issue_res_ok_i,   // live + full key + future available
  input  logic [OPERANDS-1:0]       issue_term_i,     // effective terminal DONE
  input  logic [OPERANDS-1:0]       issue_fail_i,     // effective terminal failure
  output logic                      busy_o,           // committed armed
  output logic                      issue_fire_o,
  output logic                      direct_ok_o,
  output logic                      direct_fail_o,
  output logic                      park_o,

  // ---- stored keys out, for table probes and pin-release mapping ----
  output logic                      armed_o,
  output event_key_t [OPERANDS-1:0] stored_keys_o,

  // ---- wake evaluation (effective status of the stored keys) ----
  input  logic                      eval_cycle_i,     // a completion is accepted this cycle
  input  logic [OPERANDS-1:0]       probe_term_i,     // stored key resolves live, effective DONE
  input  logic [OPERANDS-1:0]       probe_fail_i,     // stored key resolves live, effective failure
  output logic                      wake_ok_o,
  output logic                      wake_fail_o,
  output logic [OPERANDS-1:0]       release_o,        // pins to drop this edge (wake or fault)

  // ---- context fault void (slot-routed by the table) ----
  input  logic                      fault_valid_i,
  input  word_t                     fault_gen_i
);

  logic                      armed_q;
  word_t                     gen_q;
  logic [CNT_W-1:0]          count_q;
  event_key_t [OPERANDS-1:0] keys_q;

  logic [OPERANDS-1:0] issue_en;
  logic [OPERANDS-1:0] armed_en;
  logic                issue_res_all;
  logic                issue_fail_any;
  logic                issue_all_term;
  logic                stored_fail_any;
  logic                stored_all_term;
  logic                fault_hit;
  logic                fault_kill_issue;

  // Enabled-operand masks (count bits of record state, not word compares).
  always_comb begin
    for (int unsigned i = 0; i < OPERANDS; i++) begin
      issue_en[i] = (word_t'(i) < issue_count_i);
      armed_en[i] = (word_t'(i) < word_t'(count_q));
    end
  end

  // ---- fault: voids a matching armed record, or a matching in-flight issue ----
  assign fault_hit        = fault_valid_i && armed_q && (gen_q == fault_gen_i);
  assign fault_kill_issue = fault_valid_i && !armed_q && (issue_gen_i == fault_gen_i);

  // ---- issue decision (combinational, current issue, valid-qualified) ----
  assign issue_res_all  = ( &(issue_res_ok_i | ~issue_en)) & (|issue_en);
  assign issue_fail_any = |(issue_fail_i & issue_en);
  assign issue_all_term = &(issue_term_i | ~issue_en);

  assign issue_fire_o  = issue_valid_i && !armed_q && !fault_kill_issue && issue_res_all;
  assign direct_ok_o   = issue_fire_o && !issue_fail_any && issue_all_term;
  assign direct_fail_o = issue_fire_o && issue_fail_any;
  assign park_o        = issue_fire_o && !issue_fail_any && !issue_all_term;
  assign busy_o        = armed_q;

  // ---- wake evaluation on an accepted-completion cycle ----
  assign stored_fail_any = |(probe_fail_i & armed_en);
  assign stored_all_term = &(probe_term_i | ~armed_en);

  assign wake_ok_o   = armed_q && eval_cycle_i && !stored_fail_any && stored_all_term && !fault_hit;
  assign wake_fail_o = armed_q && eval_cycle_i && stored_fail_any && !fault_hit;

  always_comb begin
    if (fault_hit || wake_ok_o || wake_fail_o) begin
      release_o = armed_en;
    end else begin
      release_o = '0;
    end
  end

  // ---- record registers: exactly one writer, whole-record writes ----
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      armed_q <= 1'b0;
      gen_q   <= '0;
      count_q <= '0;
      keys_q  <= '0;
    end else if (fault_hit || wake_ok_o || wake_fail_o) begin
      armed_q <= 1'b0;  // verdict reached / record voided: clear as a whole
    end else if (park_o) begin
      armed_q <= 1'b1;
      gen_q   <= issue_gen_i;
      count_q <= CNT_W'(issue_count_i);
      keys_q  <= issue_keys_i;
    end
  end

  assign armed_o       = armed_q;
  assign stored_keys_o = keys_q;

endmodule
