// transfer_outbridge.sv — C030 Tile-side leg of the external memory channel.
//
// Bridges the L1 MSHR's refill obligations to the wire MEMORY channel
// (pipeline_validator/cosim/wire.py MEM_REQ / MEM_RSP / TERMINAL_ACK, served by
// pipeline_validator/cosim/memory_service.py). The service creates one terminal
// REFERENCE per accepted (txn_id, slice_id) and releases it only on
// TERMINAL_ACK; this module is the Tile-side owner of that obligation.
//
// Record life (one record per accepted obligation):
//   REQ    MEM_REQ not yet accepted by the channel
//   SENT   MEM_REQ accepted, response outstanding
//   FILL   good response held; the data is delivered to the MSHR entry
//   FCANCEL faulted response: the MSHR entry is cancelled so its waiters get a
//          terminal no-data response
//   TERM   TERMINAL_ACK owed to the service
//   SETTLE the TERMINAL_ACK handshake happened; OP_TERMINAL settles the MSHR's
//          refill obligation (for successful, cancelled and faulted refills alike)
//
// Rules:
//   * a MEM_RSP is NOT the terminal; only the TERMINAL_ACK the Tile sends after
//     the response is handled closes the service-side reference;
//   * the final state reports what actually happened: FAULTED on a faulted
//     response, COMPLETED when the data reached the MSHR entry, ISOLATED when
//     the obligation was cancelled before delivery (late data is absorbed);
//   * the MSHR is told OP_TERMINAL only AFTER the TERMINAL_ACK handshake, for every
//     record; with term_ready low the MSHR token stays busy no matter how many
//     waiters have already consumed and released;
//   * MEM_RSP is matched on (txn_id, slice 0, binding_generation); anything else
//     is counted as a stale response and ignored;
//   * records are bounded by RECORDS; a request beyond that is held at the input
//     (ready low), never dropped; txn_id is a monotonic per-bridge counter.
//
// Wire field widths are transport containers, not frozen silicon widths.
// terminal_state encoding follows wire.py: [COMPLETED, ISOLATED, FAULTED].
`timescale 1ns/1ps
module transfer_outbridge
  import c029_params_pkg::*;
#(
  parameter int unsigned RECORDS = MSHR_ENTRIES,
  parameter int unsigned LINE_B = LINE_BYTES
) (
  input  logic clk,
  input  logic rst_n,

  // ---- accepted refill obligation from the MSHR ----
  input  logic req_valid,
  output logic req_ready,
  input  logic [$clog2(RECORDS)-1:0] req_mshr_slot,
  input  logic [31:0] req_mshr_gen,
  input  logic [31:0] req_binding_id,
  input  logic [31:0] req_binding_generation,
  input  logic [31:0] req_binding_offset,
  input  logic [31:0] req_ctx_slot,
  input  logic [31:0] req_ctx_generation,
  input  logic [31:0] req_arena_alloc_gen,
  input  logic [31:0] req_scope_epoch,

  // ---- MEM_REQ (T2G) ----
  output logic mem_req_valid,
  input  logic mem_req_ready,
  output logic [31:0] mem_req_txn_id,
  output logic [31:0] mem_req_slice_id,
  output logic [31:0] mem_req_slice_count,
  output logic [31:0] mem_req_binding_id,
  output logic [31:0] mem_req_binding_generation,
  output logic [31:0] mem_req_binding_offset,
  output logic [31:0] mem_req_ctx_slot,
  output logic [31:0] mem_req_ctx_generation,
  output logic [31:0] mem_req_arena_alloc_gen,
  output logic [31:0] mem_req_scope_epoch,
  output logic [31:0] mem_req_length,

  // ---- MEM_RSP (G2T) ----
  input  logic mem_rsp_valid,
  output logic mem_rsp_ready,
  input  logic [31:0] mem_rsp_txn_id,
  input  logic [31:0] mem_rsp_slice_id,
  input  logic [31:0] mem_rsp_binding_generation,
  input  logic [31:0] mem_rsp_status,
  input  logic [LINE_B*8-1:0] mem_rsp_data,

  // ---- cancellation request mark from the MSHR client (never settles) ----
  input  logic cancel_valid,
  input  logic [$clog2(RECORDS)-1:0] cancel_mshr_slot,
  input  logic [31:0] cancel_mshr_gen,

  // ---- MSHR command channel: FILL / CANCEL / TERMINAL (MSHR opcodes) ----
  output logic mc_valid,
  input  logic mc_ready,
  output logic [2:0] mc_op,
  output logic [$clog2(RECORDS)-1:0] mc_slot,
  output logic [31:0] mc_gen,
  output logic [LINE_B*8-1:0] mc_data,

  // ---- TERMINAL_ACK (T2G) ----
  output logic term_valid,
  input  logic term_ready,
  output logic [31:0] term_txn_id,
  output logic [31:0] term_slice_id,
  output logic [31:0] term_slice_count,
  output logic [1:0] term_final_state,

  output logic [$clog2(RECORDS+1)-1:0] outstanding_o,
  output logic [63:0] sent_o, rsp_o, terminals_o, absorbed_o, stale_rsp_o,
  output logic invariant_error_o
);
  localparam int unsigned SW = (RECORDS <= 1) ? 1 : $clog2(RECORDS);
  localparam logic [1:0] F_COMPLETED = 2'd0, F_ISOLATED = 2'd1, F_FAULTED = 2'd2;
  localparam logic [2:0] MOP_FILL = 3'd2, MOP_CANCEL = 3'd5, MOP_TERMINAL = 3'd7;

  localparam logic [2:0] R_FREE = 3'd0, R_REQ = 3'd1, R_SENT = 3'd2, R_FILL = 3'd3,
                         R_FCANCEL = 3'd4, R_TERM = 3'd5, R_SETTLE = 3'd6;

  logic [2:0] rstate [RECORDS];
  logic [31:0] rtxn [RECORDS];
  logic [31:0] rbgen [RECORDS];
  logic [31:0] rmslot [RECORDS];
  logic [31:0] rmgen [RECORDS];
  logic [31:0] rbid [RECORDS];
  logic [31:0] roff [RECORDS];
  logic [31:0] rctx [RECORDS];
  logic [31:0] rcgen [RECORDS];
  logic [31:0] rarena [RECORDS];
  logic [31:0] rscope [RECORDS];
  logic rcancel [RECORDS];
  logic rfault [RECORDS];
  logic rdelivered [RECORDS];
  logic [LINE_B*8-1:0] rdata [RECORDS];

  logic [31:0] next_txn;
  logic [63:0] sent, rsp, terms, absorbed, stale_rsp;
  logic inv_err;

  assign sent_o = sent;
  assign rsp_o = rsp;
  assign terminals_o = terms;
  assign absorbed_o = absorbed;
  assign stale_rsp_o = stale_rsp;
  assign invariant_error_o = inv_err;

  logic [SW-1:0] req_pick, free_pick, term_pick, settle_pick, fcancel_pick, fill_pick, mc_pick;
  logic req_have, free_have, term_have, settle_have, fcancel_have, fill_have;
  always_comb begin
    int n;
    n = 0;
    req_have = 1'b0; free_have = 1'b0; term_have = 1'b0; settle_have = 1'b0;
    fcancel_have = 1'b0; fill_have = 1'b0;
    req_pick = '0; free_pick = '0; term_pick = '0; settle_pick = '0; fcancel_pick = '0; fill_pick = '0;
    for (int i = 0; i < RECORDS; i++) begin
      if (rstate[i] != R_FREE) n++;
      // Oldest-transaction (service FIFO) selection: a MEM_REQ or TERMINAL
      // presented under backpressure keeps its record until the handshake, so
      // a newly allocated lower-index record can never change the presented
      // wire fields while valid is held without acceptance.
      if (rstate[i] == R_REQ && (!req_have || rtxn[i] < rtxn[req_pick])) begin
        req_have = 1'b1; req_pick = SW'(i);
      end
      if (rstate[i] == R_FREE && !free_have) begin free_have = 1'b1; free_pick = SW'(i); end
      if (rstate[i] == R_TERM && (!term_have || rtxn[i] < rtxn[term_pick])) begin
        term_have = 1'b1; term_pick = SW'(i);
      end
      if (rstate[i] == R_SETTLE && !settle_have) begin settle_have = 1'b1; settle_pick = SW'(i); end
      if (rstate[i] == R_FCANCEL && !fcancel_have) begin fcancel_have = 1'b1; fcancel_pick = SW'(i); end
      if (rstate[i] == R_FILL && !rcancel[i] && !fill_have) begin fill_have = 1'b1; fill_pick = SW'(i); end
    end
    outstanding_o = n[$clog2(RECORDS+1)-1:0];
    // MSHR command priority: settle > fault-cancel > fill.
    mc_valid = settle_have || fcancel_have || fill_have;
    mc_pick = settle_have ? settle_pick : (fcancel_have ? fcancel_pick : fill_pick);
    mc_op = settle_have ? MOP_TERMINAL : (fcancel_have ? MOP_CANCEL : MOP_FILL);
  end

  assign mc_slot = SW'(rmslot[mc_pick]);
  assign mc_gen = rmgen[mc_pick];
  assign mc_data = rdata[mc_pick];

  assign req_ready = rst_n && free_have;
  assign mem_req_valid = req_have;
  assign mem_req_txn_id = rtxn[req_pick];
  assign mem_req_slice_id = 32'd0;
  assign mem_req_slice_count = 32'd1;
  assign mem_req_binding_id = rbid[req_pick];
  assign mem_req_binding_generation = rbgen[req_pick];
  assign mem_req_binding_offset = roff[req_pick];
  assign mem_req_ctx_slot = rctx[req_pick];
  assign mem_req_ctx_generation = rcgen[req_pick];
  assign mem_req_arena_alloc_gen = rarena[req_pick];
  assign mem_req_scope_epoch = rscope[req_pick];
  assign mem_req_length = 32'(LINE_B);

  assign term_valid = term_have;
  assign term_txn_id = rtxn[term_pick];
  assign term_slice_id = 32'd0;
  assign term_slice_count = 32'd1;
  assign term_final_state = rfault[term_pick] ? F_FAULTED : (rdelivered[term_pick] ? F_COMPLETED : F_ISOLATED);

  logic rsp_match;
  logic [SW-1:0] rsp_idx;
  always_comb begin
    rsp_match = 1'b0; rsp_idx = '0;
    for (int i = 0; i < RECORDS; i++)
      if (rstate[i] == R_SENT && rtxn[i] == mem_rsp_txn_id && mem_rsp_slice_id == 32'd0 &&
          rbgen[i] == mem_rsp_binding_generation && !rsp_match) begin
        rsp_match = 1'b1; rsp_idx = SW'(i);
      end
  end
  assign mem_rsp_ready = rst_n;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      next_txn <= 32'd1;
      sent <= '0; rsp <= '0; terms <= '0; absorbed <= '0; stale_rsp <= '0; inv_err <= 1'b0;
      for (int i = 0; i < RECORDS; i++) begin
        rstate[i] <= R_FREE; rtxn[i] <= '0; rbgen[i] <= '0; rmslot[i] <= '0; rmgen[i] <= '0;
        rbid[i] <= '0; roff[i] <= '0; rctx[i] <= '0; rcgen[i] <= '0; rarena[i] <= '0;
        rscope[i] <= '0; rcancel[i] <= 1'b0; rfault[i] <= 1'b0; rdelivered[i] <= 1'b0;
        rdata[i] <= '0;
      end
    end else begin
      // A good response whose obligation was cancelled before delivery: the
      // data is absorbed and the MSHR refill obligation still needs its terminal.
      for (int i = 0; i < RECORDS; i++)
        if (rstate[i] == R_FILL && rcancel[i]) begin
          absorbed <= absorbed + 64'd1;
          rstate[i] <= R_TERM;
        end

      if (req_valid && req_ready) begin
        rstate[free_pick] <= R_REQ;
        rtxn[free_pick] <= next_txn;
        next_txn <= next_txn + 32'd1;
        rbid[free_pick] <= req_binding_id;
        rbgen[free_pick] <= req_binding_generation;
        roff[free_pick] <= req_binding_offset;
        rmslot[free_pick] <= 32'(req_mshr_slot);
        rmgen[free_pick] <= req_mshr_gen;
        rctx[free_pick] <= req_ctx_slot;
        rcgen[free_pick] <= req_ctx_generation;
        rarena[free_pick] <= req_arena_alloc_gen;
        rscope[free_pick] <= req_scope_epoch;
        rcancel[free_pick] <= 1'b0;
        rfault[free_pick] <= 1'b0;
        rdelivered[free_pick] <= 1'b0;
      end

      if (mem_req_valid && mem_req_ready) begin
        rstate[req_pick] <= R_SENT;
        sent <= sent + 64'd1;
      end

      // Cancel is a request mark on a not-yet-delivered record; it never settles.
      if (cancel_valid)
        for (int i = 0; i < RECORDS; i++)
          if ((rstate[i] == R_REQ || rstate[i] == R_SENT || rstate[i] == R_FILL) &&
              SW'(rmslot[i]) == cancel_mshr_slot && rmgen[i] == cancel_mshr_gen)
            rcancel[i] <= 1'b1;

      if (mem_rsp_valid && mem_rsp_ready) begin
        if (rsp_match) begin
          rsp <= rsp + 64'd1;
          if (mem_rsp_status != 32'd0) begin
            rfault[rsp_idx] <= 1'b1;
            if (rcancel[rsp_idx]) rstate[rsp_idx] <= R_TERM;
            else begin
              rcancel[rsp_idx] <= 1'b1;
              rstate[rsp_idx] <= R_FCANCEL;
            end
          end else if (rcancel[rsp_idx]) begin
            absorbed <= absorbed + 64'd1;
            rstate[rsp_idx] <= R_TERM;
          end else begin
            rdata[rsp_idx] <= mem_rsp_data;
            rstate[rsp_idx] <= R_FILL;
          end
        end else begin
          stale_rsp <= stale_rsp + 64'd1;
        end
      end

      if (mc_valid && mc_ready) begin
        case (mc_op)
          MOP_FILL: begin rdelivered[mc_pick] <= 1'b1; rstate[mc_pick] <= R_TERM; end
          MOP_CANCEL: rstate[mc_pick] <= R_TERM;
          default: rstate[mc_pick] <= R_FREE;
        endcase
      end

      if (term_valid && term_ready) begin
        rstate[term_pick] <= R_SETTLE;
        terms <= terms + 64'd1;
      end
    end
  end

  always @(posedge clk) begin
    if (rst_n) begin
      for (int i = 0; i < RECORDS; i++)
        for (int j = i + 1; j < RECORDS; j++)
          if (rstate[i] != R_FREE && rstate[j] != R_FREE && rtxn[i] == rtxn[j])
            $error("C030_ASSERT outbridge duplicate live txn_id %0d", rtxn[i]);
    end
  end
endmodule
