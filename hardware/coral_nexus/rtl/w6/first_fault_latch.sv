// first_fault_latch.sv — C034 per-Tile first-fault latch (W6-03, FLT-10/11).
//
// Source lanes carry DECODED fault records (c034_reset_pkg::fault_t), not
// wire payloads. Guarantees:
//   * first-fault-wins; armed only while no reset is active and none has
//     completed (a clear never reopens a closed arm gate);
//   * same-cycle arbitration: strict unsigned lexicographic minimum on
//     source_id -> engine_id -> owner.ctx.ctx_slot -> op_seq (no hash);
//     lowest lane only on identical ordering keys;
//   * every valid fault that does not capture increments the saturating
//     suppressed counter (also while the arm gate is closed); wrap is
//     forbidden and only the cold reset clears it — a functional clear
//     never does;
//   * the record is retained until an authorized clear (FAULT_ACK with a
//     matching tile, or privileged fault_clear); FAULT_REPORT fires once
//     per latched record and holds the full payload under stall;
//   * a wrong-tile ack, or any clear presented before the record's report
//     fires, is ignored: the owed report is preserved (clear_ready_o says a
//     clear is acceptable now; a clear on the report-fire edge is accepted);
//   * same-edge clear + new fault with the arm gate open: the OLD record
//     retires and the new fault starts the next record — the clear is never
//     reinterpreted as clearing the newly arrived record.
`timescale 1ns/1ps
module first_fault_latch
  import c034_reset_pkg::*;
  import c033_event_pkg::*;
#(
  parameter int unsigned SOURCES,
  parameter int unsigned COUNTER_BITS,
  parameter word_t      TILE_ID = 64'd0
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,

  // reset-domain view gating the arm
  input  logic                 reset_active_i,
  input  logic                 reset_done_i,

  // decoded fault lanes
  input  logic [SOURCES-1:0]   fault_valid_i,
  input  fault_t [SOURCES-1:0] fault_i,

  // committed latch view (retained until authorized clear)
  output logic                 record_valid_o,
  output fault_t               record_o,

  // FAULT_REPORT, reliable path: once per latched record
  output logic                 report_valid_o,
  input  logic                 report_ready_i,

  // authorized clear: FAULT_ACK(tile) or privileged fault_clear
  input  logic                 ack_valid_i,
  input  word_t                ack_tile_i,
  input  logic                 priv_clear_i,
  output logic                 clear_ready_o,

  // suppressed-fault accounting
  output word_t                suppressed_o,
  output logic                 suppressed_seen_o
);

  logic                 record_valid_q;
  fault_t               record_q;
  logic                 report_fired_q;
  logic [COUNTER_BITS-1:0] suppressed_q;

  // Saturation intermediate: wide enough for the counter AND the per-edge
  // increment, so a large SOURCES burst never narrows to zero.
  localparam int unsigned SUP_INC_W = (COUNTER_BITS > $clog2(SOURCES + 1))
                                      ? COUNTER_BITS : $clog2(SOURCES + 1);
  logic [SUP_INC_W:0]        sup_ext;
  logic [COUNTER_BITS-1:0]   sup_next;

  logic                 arm_open;
  logic                 report_fire;
  logic                 clear_req, clear_accept;
  logic                 capture;
  logic                 win_valid;
  int                   win_lane;
  int                   n_sup;

  assign arm_open = !(reset_active_i || reset_done_i);

  // ---- strict lexicographic minimum over the valid lanes ----
  function automatic logic lane_less(input fault_t a, input fault_t b);
    if (a.source_id != b.source_id) begin
      return a.source_id < b.source_id;
    end else if (a.engine_id != b.engine_id) begin
      return a.engine_id < b.engine_id;
    end else if (a.owner.ctx.ctx_slot != b.owner.ctx.ctx_slot) begin
      return a.owner.ctx.ctx_slot < b.owner.ctx.ctx_slot;
    end else begin
      return a.op_seq < b.op_seq;
    end
  endfunction

  always_comb begin
    win_valid = 1'b0;
    win_lane  = 0;
    for (int unsigned s = 0; s < SOURCES; s++) begin
      if (fault_valid_i[s] && (!win_valid || lane_less(fault_i[s], fault_i[win_lane]))) begin
        win_valid = 1'b1;
        win_lane  = int'(s);
      end
    end
  end

  // ---- report and clear handshakes ----
  assign report_valid_o = record_valid_q && !report_fired_q;
  assign report_fire    = report_valid_o && report_ready_i;
  assign clear_req      = (ack_valid_i && (ack_tile_i == TILE_ID)) || priv_clear_i;
  assign clear_ready_o  = !record_valid_q || report_fired_q || report_fire;
  assign clear_accept   = clear_req && clear_ready_o;

  // First-fault-wins capture: empty latch, or one retired by a clear
  // accepted this edge, and the arm gate open.
  assign capture = win_valid && arm_open && (!record_valid_q || clear_accept);

  always_comb begin
    // every valid fault that did not capture on this edge is suppressed
    n_sup = 0;
    for (int unsigned s = 0; s < SOURCES; s++) begin
      if (fault_valid_i[s] && !(capture && (int'(s) == win_lane))) begin
        n_sup++;
      end
    end
    sup_ext  = {{(SUP_INC_W+1-COUNTER_BITS){1'b0}}, suppressed_q} +
               (SUP_INC_W+1)'(n_sup);
    sup_next = (|sup_ext[SUP_INC_W:COUNTER_BITS]) ? '1
                                                       : sup_ext[COUNTER_BITS-1:0];
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      record_valid_q <= 1'b0;
      record_q       <= '0;
      report_fired_q <= 1'b0;
      suppressed_q   <= '0;
    end else begin
      if (capture) begin
        record_q       <= fault_i[win_lane];
        record_valid_q <= 1'b1;
        report_fired_q <= 1'b0;  // the next record owes its own report
      end else if (clear_accept) begin
        record_valid_q <= 1'b0;  // old record retired by the authorized clear
        report_fired_q <= 1'b0;
      end else if (report_fire) begin
        report_fired_q <= 1'b1;  // once per latch
      end
      suppressed_q <= sup_next;
    end
  end

  assign record_valid_o    = record_valid_q;
  assign record_o          = record_q;
  assign suppressed_o      = word_t'(suppressed_q);
  assign suppressed_seen_o = (suppressed_q != '0);

endmodule
