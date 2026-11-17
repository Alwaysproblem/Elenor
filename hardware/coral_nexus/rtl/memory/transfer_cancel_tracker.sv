// transfer_cancel_tracker.sv — C032 per-transaction terminal accounting.
//
// One entry per accepted MFE transaction (and usable standalone for injected
// protocol violations). A cancel request only MARKS an entry
// (CANCEL_REQUESTED); settlement happens exactly once per entry as
// COMPLETED / ISOLATED / FAULTED, decided by the engine's real completion
// evidence (all bytes moved, write leg ACKed, authorize fault, transport
// fault) — never by a timeout.
//
// Refused protocol violations, each with a distinct sticky error code, a
// refused count and the sticky invariant flag:
//   4'd6  ERR_DUPLICATE_TERMINAL      settle on an already-settled entry
//   4'd7  ERR_TERMINAL_AFTER_RELEASE  settle on a released/free entry
//   4'd8  ERR_DUPLICATE_RELEASE       release on a free entry
//   4'd9  ERR_RELEASE_WITHOUT_ACK     release while wire obligations are open
//   4'd10 ERR_RELEASE_EARLY           release before any terminal
//   4'd11 ERR_PROTOCOL                obligation/ack on a free entry
//   4'd12 ERR_RESET                   quiescent_reset while not quiescent
// These values mirror c032_mfe_pkg::ERR_* by value so mfe_top and the bench
// compare one space. A refused command changes no entry state and emits no
// terminal. ERR_RESET is a recoverable Group retry: it keeps the poison but
// does not raise the invariant flag.
//
// Poison (Group-owned): isolation_failed_i or any refused command sets a
// sticky poison that refuses new allocations until quiescent_reset_i arrives
// with every entry free and no pending terminal output; cancels and new
// starts never clear it.
//
// Terminals are queued FIFO (depth ENTRIES: one push per entry before any
// pop can be owed) and the head presentation is register-stable under
// backpressure. A released entry may still have its terminal queued; release
// only requires settle + all wire acknowledgements closed.
`timescale 1ns/1ps
module transfer_cancel_tracker #(
  parameter int unsigned ENTRIES = 4,  // >= 2
  parameter int unsigned ACK_W = 16    // wire obligations per entry
) (
  input  logic clk,
  input  logic rst_n,

  output logic alloc_ready_o,
  input  logic alloc_valid_i,
  input  logic [31:0] alloc_txn_i,
  output logic [$clog2(ENTRIES)-1:0] alloc_slot_o,

  input  logic cancel_valid_i,
  input  logic [31:0] cancel_txn_i,

  input  logic obligate_valid_i,
  input  logic [$clog2(ENTRIES)-1:0] obligate_slot_i,
  input  logic ack_valid_i,
  input  logic [$clog2(ENTRIES)-1:0] ack_slot_i,

  input  logic settle_valid_i,
  input  logic [$clog2(ENTRIES)-1:0] settle_slot_i,
  input  logic [1:0] settle_final_i,
  input  logic [3:0] settle_err_i,
  input  logic release_valid_i,
  input  logic [$clog2(ENTRIES)-1:0] release_slot_i,

  output logic term_valid_o,
  input  logic term_ready_i,
  output logic [31:0] term_txn_o,
  output logic [1:0] term_final_o,
  output logic [3:0] term_err_o,

  output logic [ENTRIES-1:0] free_o, live_o, marked_o, settled_o, acks_open_o,

  output logic poisoned_o,
  input  logic isolation_failed_i,
  input  logic quiescent_reset_i,
  output logic [3:0] err_code_o,
  output logic invariant_error_o,
  output logic [63:0] terminals_o, stale_cancel_o, refused_o
);
  localparam int unsigned FW = (ENTRIES <= 2) ? 1 : $clog2(ENTRIES);
  localparam int unsigned CW = $clog2(ENTRIES + 1);

  // Mirrors of c032_mfe_pkg::ERR_* (this module stays package-independent).
  localparam logic [3:0] CODE_DUPLICATE_TERMINAL = 4'd6, CODE_TERMINAL_AFTER_RELEASE = 4'd7,
    CODE_DUPLICATE_RELEASE = 4'd8, CODE_RELEASE_WITHOUT_ACK = 4'd9,
    CODE_RELEASE_EARLY = 4'd10, CODE_PROTOCOL = 4'd11, CODE_RESET = 4'd12;

  localparam logic [1:0] E_FREE = 2'd0, E_RUN = 2'd1, E_SETTLED = 2'd2;

  logic [1:0] estate [ENTRIES];
  logic [31:0] etxn [ENTRIES];
  logic emark [ENTRIES];
  logic [ACK_W-1:0] owed [ENTRIES], done_cnt [ENTRIES];

  logic [31:0] ftxn [ENTRIES];
  logic [1:0] ffinal [ENTRIES];
  logic [3:0] ferr [ENTRIES];
  logic [FW-1:0] fhead, ftail;
  logic [CW-1:0] fcount;

  logic poisoned, inv_err;
  logic [3:0] err_code;
  logic [63:0] terminals, stale_cancel, refused;

  logic free_have, cancel_hit, quiescent, push, pop;
  logic [FW-1:0] cancel_idx;
  logic [3:0] refusal_code;

  always_comb begin
    free_have = 1'b0; alloc_slot_o = '0;
    for (int i = 0; i < ENTRIES; i++)
      if (estate[i] == E_FREE && !free_have) begin
        free_have = 1'b1; alloc_slot_o = FW'(i);
      end
  end
  assign alloc_ready_o = free_have && !poisoned;

  always_comb begin
    cancel_hit = 1'b0; cancel_idx = '0;
    for (int i = 0; i < ENTRIES; i++)
      if (estate[i] == E_RUN && etxn[i] == cancel_txn_i && !cancel_hit) begin
        cancel_hit = 1'b1; cancel_idx = FW'(i);
      end
  end

  always_comb begin
    quiescent = 1'b1;
    for (int i = 0; i < ENTRIES; i++)
      if (estate[i] != E_FREE) quiescent = 1'b0;
    if (fcount != CW'(0)) quiescent = 1'b0;
  end

  assign push = settle_valid_i && (estate[settle_slot_i] == E_RUN);
  assign pop = term_valid_o && term_ready_i;

  // One deterministic code per cycle; a later condition wins the priority.
  always_comb begin
    refusal_code = 4'd0;
    if (settle_valid_i) begin
      if (estate[settle_slot_i] == E_SETTLED) refusal_code = CODE_DUPLICATE_TERMINAL;
      else if (estate[settle_slot_i] == E_FREE) refusal_code = CODE_TERMINAL_AFTER_RELEASE;
    end
    if (release_valid_i) begin
      if (estate[release_slot_i] == E_FREE) refusal_code = CODE_DUPLICATE_RELEASE;
      else if (estate[release_slot_i] == E_RUN) refusal_code = CODE_RELEASE_EARLY;
      else if (owed[release_slot_i] != done_cnt[release_slot_i])
        refusal_code = CODE_RELEASE_WITHOUT_ACK;
    end
    if ((obligate_valid_i && estate[obligate_slot_i] == E_FREE) ||
        (ack_valid_i && estate[ack_slot_i] == E_FREE)) refusal_code = CODE_PROTOCOL;
    if (quiescent_reset_i && poisoned && !quiescent) refusal_code = CODE_RESET;
  end

  always_comb begin
    for (int i = 0; i < ENTRIES; i++) begin
      free_o[i] = (estate[i] == E_FREE);
      live_o[i] = (estate[i] != E_FREE);
      marked_o[i] = emark[i] && (estate[i] == E_RUN);
      settled_o[i] = (estate[i] == E_SETTLED);
      acks_open_o[i] = (estate[i] != E_FREE) && (owed[i] != done_cnt[i]);
    end
  end

  assign term_valid_o = (fcount != CW'(0));
  assign term_txn_o = ftxn[fhead];
  assign term_final_o = ffinal[fhead];
  assign term_err_o = ferr[fhead];

  assign poisoned_o = poisoned;
  assign err_code_o = err_code;
  assign invariant_error_o = inv_err;
  assign terminals_o = terminals;
  assign stale_cancel_o = stale_cancel;
  assign refused_o = refused;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      poisoned <= 1'b0; inv_err <= 1'b0; err_code <= 4'd0;
      terminals <= 64'd0; stale_cancel <= 64'd0; refused <= 64'd0;
      fhead <= '0; ftail <= '0; fcount <= CW'(0);
      for (int i = 0; i < ENTRIES; i++) begin
        estate[i] <= E_FREE; etxn[i] <= 32'd0; emark[i] <= 1'b0;
        owed[i] <= '0; done_cnt[i] <= '0;
        ftxn[i] <= 32'd0; ffinal[i] <= 2'd0; ferr[i] <= 4'd0;
      end
    end else begin
      if (refusal_code != 4'd0) begin
        err_code <= refusal_code;
        // A refused quiescent reset is a recoverable Group retry, not a
        // Tile-side protocol violation: it keeps the poison and the error
        // code but raises neither the invariant flag nor the refused count.
        if (refusal_code != CODE_RESET) begin
          refused <= refused + 64'd1;
          inv_err <= 1'b1;
        end
        poisoned <= 1'b1;
      end else begin
        if (isolation_failed_i) poisoned <= 1'b1;
        if (quiescent_reset_i && poisoned && quiescent) poisoned <= 1'b0;
      end

      if (alloc_valid_i && alloc_ready_o) begin
        estate[alloc_slot_o] <= E_RUN;
        etxn[alloc_slot_o] <= alloc_txn_i;
        emark[alloc_slot_o] <= 1'b0;
        owed[alloc_slot_o] <= '0;
        done_cnt[alloc_slot_o] <= '0;
      end

      // A cancel is only a mark. A mark racing the settle on the same edge
      // loses: the entry settles once and the mark is counted stale below
      // when it can no longer apply.
      if (cancel_valid_i) begin
        if (cancel_hit) emark[cancel_idx] <= 1'b1;
        else stale_cancel <= stale_cancel + 64'd1;
      end

      if (obligate_valid_i && estate[obligate_slot_i] != E_FREE)
        owed[obligate_slot_i] <= owed[obligate_slot_i] + ACK_W'(1);
      if (ack_valid_i && estate[ack_slot_i] != E_FREE)
        done_cnt[ack_slot_i] <= done_cnt[ack_slot_i] + ACK_W'(1);

      if (push) begin
        estate[settle_slot_i] <= E_SETTLED;
        ftxn[ftail] <= etxn[settle_slot_i];
        ffinal[ftail] <= settle_final_i;
        ferr[ftail] <= settle_err_i;
        ftail <= (ftail == FW'(ENTRIES - 1)) ? '0 : FW'(ftail + 32'(1));
        terminals <= terminals + 64'd1;
      end


      if (release_valid_i && estate[release_slot_i] == E_SETTLED &&
          owed[release_slot_i] == done_cnt[release_slot_i])
        estate[release_slot_i] <= E_FREE;

      if (pop) fhead <= (fhead == FW'(ENTRIES - 1)) ? '0 : FW'(fhead + 32'(1));

      case ({push, pop})
        2'b10: fcount <= fcount + CW'(1);
        2'b01: fcount <= fcount - CW'(1);
        default: ;
      endcase
    end
  end

  // Never-happens invariants in green runs; refusal paths themselves are
  // observable state the bench asserts on.
  always @(posedge clk) begin
    if (rst_n) begin
      if (push && fcount == CW'(ENTRIES))
        $error("C032_ASSERT tracker terminal FIFO overflow");
      if (fcount > CW'(ENTRIES))
        $error("C032_ASSERT tracker terminal FIFO count corrupt");
      if (ack_valid_i && estate[ack_slot_i] != E_FREE && done_cnt[ack_slot_i] >= owed[ack_slot_i])
        $error("C032_ASSERT tracker acknowledgment without obligation");
    end
  end
endmodule
