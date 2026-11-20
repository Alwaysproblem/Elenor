// event_table.sv — C033 local persistent event table + WAIT issue head (W6-02).
//
// Per-Context persistent entry banks (ctx_slot x FRONTIER) with the two-count
// frontier (consumer_refs / future_uses), one wait_record per Context
// (single writer) and a fair completion_bus fan-in. All geometry parameters
// are required explicitly at instantiation.
//
// Same-cycle semantics (frozen C033 cycle contract):
//   * A granted completion is always serviced: sticky terminal write into a
//     live entry or counters only, never an allocation, so a full table
//     still services completions and the bus never stalls.
//   * Completion beats issue: an operand resolved to the slot a same-cycle
//     granted completion applies to sees the incoming status ("effective
//     status"), so the wait direct-verdicts instead of parking.
//     Park-then-wake-in-one-cycle never happens.
//   * An allocation is visible only after its edge: a completion or WAIT for
//     a name allocated in the same cycle misses (unknown / unresolved).
//   * Issue ref migration is all-or-none per accepted issue: every operand
//     consumes one future_use (rejected operand => whole issue rejected, no
//     partial pin, future unchanged; future_uses==0 => rejected). A parked
//     record takes one consumer_ref pin per operand; a direct verdict
//     settles in the same edge, so consumer_refs nets to zero while
//     future_uses still drops by one.
//   * Reclaim (evaluated every edge on post-delta values): terminal AND
//     consumer_refs==0 AND future_uses==0. A pinned entry is never
//     reclaimed; a terminal entry with future_uses>0 is never reused.
//   * wake_ok_o/wake_fail_o pulse on the accepted completion cycle for
//     previously parked records only, separate from direct_ok_o/direct_fail_o.
//     Two records (two consumers pinning one shared-scope event) may settle
//     on one edge; releases accumulate on the shared slot.
//   * fault_valid_i voids the armed record whose slot+generation match,
//     releasing all its pins, no wake of either kind; a registration issued
//     on the same edge with the same waiter generation is voided before it
//     fires (not accepted, no pin, no verdict); a stale-generation fault
//     changes nothing.
//   * Scope rule: shared-scope operands are waitable from any Context; a
//     task_private operand only by its own full context identity.
//
// Counters (C015): aggregates and details describe a fired NOTIFY
// (completion), classified in check order with one aggregate + at most one
// detail per rejected notification. Every rejected presented WAIT — shape,
// duplicate operands, unroutable context, unresolved/mismatched operand, no
// remaining future use, busy record — is one registration violation: exactly
// one protocol_error for the issuing Context (route-unattributable events
// use Context 0 as the documented route bucket; a well-formed fault-killed
// issue is not a violation and counts nothing). A malformed (non-terminal
// status) completion is a protocol_error attributed to its routed Context.
// Allocation rejections are protocol_error; a full frontier is backpressure
// and counts capacity_reject. The three sources can reject on one edge: per
// counter they contribute 0..3, added with saturation (wrap is forbidden,
// COUNTER_BITS>=1). counts_seen_o is the sticky nonzero view of each
// counter. consumer_refs are bounded by CONTEXT_COUNT (one pin per live
// record per distinct event) and stored narrow; public counts stay word_t.
//
// Rejected inputs change counters only. All scans are bounded by elaboration
// constants; identities are full 64-bit words. rst_ni is a synchronous
// active-low cold reset, not a protocol reset.
`timescale 1ns/1ps
module event_table
  import c033_event_pkg::*;
#(
  parameter int unsigned CONTEXT_COUNT,
  parameter int unsigned FRONTIER,
  parameter int unsigned OPERANDS,
  parameter int unsigned SOURCES,
  parameter int unsigned COUNTER_BITS,
  parameter word_t      TILE_ID = 64'd0
) (
  input  logic                        clk_i,
  input  logic                        rst_ni,

  // event registration (producer bound here, kind fixed)
  input  logic                        alloc_valid_i,
  input  event_key_t                  alloc_key_i,
  input  logic [1:0]                  alloc_kind_i,
  input  word_t                       alloc_producer_i,
  input  word_t                       alloc_future_i,
  output logic                        alloc_ready_o,
  output logic                        alloc_reject_o,

  // completion fan-in (one granted producer per cycle)
  input  logic [SOURCES-1:0]          comp_valid_i,
  input  completion_t [SOURCES-1:0]   comp_i,
  output logic [SOURCES-1:0]          comp_ready_o,

  // single global WAIT/WAITALL issue head
  input  logic                        wait_valid_i,
  input  logic                        wait_all_i,
  input  context_key_t                wait_ctx_i,
  input  word_t                       wait_count_i,
  input  event_key_t [OPERANDS-1:0]   wait_keys_i,
  output logic                        wait_ready_o,
  output logic                        wait_reject_o,
  output logic                        direct_ok_o,
  output logic                        direct_fail_o,
  output logic                        park_o,

  // committed armed state and parked-record wake pulses
  output logic [CONTEXT_COUNT-1:0]    armed_o,
  output logic [CONTEXT_COUNT-1:0]    wake_ok_o,
  output logic [CONTEXT_COUNT-1:0]    wake_fail_o,

  // context fault: void the matched wait record
  input  logic                        fault_valid_i,
  input  context_key_t                fault_ctx_i,

  // committed-state query (never counts, no preview; absent => all zero)
  input  event_key_t                  query_key_i,
  output event_view_t                 query_o,

  // per-Context PMU counters (saturating) and sticky seen flags
  output counts_t [CONTEXT_COUNT-1:0] counts_o,
  output counter_flags_t [CONTEXT_COUNT-1:0] counts_seen_o
);

  // counts_t field order is frozen by the package; index map kept local.
  localparam int unsigned NCNT    = 10;
  localparam int unsigned C_UNK   = 0;  // unknown_event_notify
  localparam int unsigned C_OWNER = 1;  // wrong_owner
  localparam int unsigned C_STALE = 2;  // stale_sequence
  localparam int unsigned C_DUP   = 3;  // duplicate_signal
  localparam int unsigned C_CAP   = 4;  // capacity_reject
  localparam int unsigned C_PROTO = 5;  // protocol_error
  localparam int unsigned C_WCTX  = 6;  // wrong_context (detail)
  localparam int unsigned C_WSCP  = 7;  // wrong_scope (detail)
  localparam int unsigned C_WSEQ  = 8;  // wrong_sequence (detail)
  localparam int unsigned C_WEVG  = 9;  // wrong_event_generation (detail)

  // Pins are bounded: one per live record per distinct event.
  localparam int unsigned REF_W = $clog2(CONTEXT_COUNT + 1);

  typedef struct packed {
    logic         live;
    event_key_t   key;          // full key, ctx included (never mutates)
    logic [1:0]   kind;
    event_status_t status;
    word_t        producer_id;  // bound at registration
    word_t        error_code;
    logic [REF_W-1:0] consumer_refs;
    word_t        future_uses;
  } entry_t;

  entry_t [CONTEXT_COUNT-1:0][FRONTIER-1:0] tbl_q;

  // Per-slot same-edge control: signed pin delta, one future decrement
  // (a single issue touches each slot at most once), alloc pick.
  int   dref [CONTEXT_COUNT][FRONTIER];
  logic [CONTEXT_COUNT-1:0][FRONTIER-1:0] dec_fut, slot_wr_q, slot_clr_q;

  // ------------------------------------------------------------------
  // Completion fan-in: fair round-robin, exactly one granted producer.
  // ------------------------------------------------------------------
  logic        comp_fire;
  completion_t comp_g;

  completion_bus #(
    .SOURCES(SOURCES)
  ) u_comp_bus (
    .clk_i         (clk_i),
    .rst_ni        (rst_ni),
    .req_valid_i   (comp_valid_i),
    .comp_i        (comp_i),
    .grant_ready_o (comp_ready_o),
    .grant_valid_o (comp_fire),
    .grant_comp_o  (comp_g)
  );

  // ------------------------------------------------------------------
  // Granted-completion classification: C015 check order, one aggregate +
  // at most one detail per rejected notification, no entry mutation.
  // ------------------------------------------------------------------
  logic comp_status_ok;   // payload carries a terminal status encoding
  logic comp_route_ok;    // tile matches and ctx_slot routable
  int   comp_bank;
  logic comp_name_hit;    // live local name in the routed bank
  int   comp_slot;
  logic comp_ctx_ok, comp_scope_ok, comp_evgen_ok, comp_seq_ok;
  logic comp_prod_ok, comp_not_dup;
  logic comp_apply;       // this completion writes a sticky terminal now

  assign comp_status_ok = (comp_g.status == EV_DONE) || (comp_g.status == EV_ERROR) ||
                          (comp_g.status == EV_TIMEOUT) || (comp_g.status == EV_RESET);
  assign comp_route_ok  = (comp_g.key.ctx.tile_id == TILE_ID) &&
                          (comp_g.key.ctx.ctx_slot < word_t'(CONTEXT_COUNT));

  always_comb begin
    comp_bank = (comp_g.key.ctx.ctx_slot < word_t'(CONTEXT_COUNT))
              ? int'(comp_g.key.ctx.ctx_slot) : 0;
  end

  always_comb begin
    comp_name_hit = 1'b0;
    comp_slot     = 0;
    for (int unsigned f = 0; f < FRONTIER; f++) begin
      if (tbl_q[comp_bank][f].live &&
          (tbl_q[comp_bank][f].key.local_event_id == comp_g.key.local_event_id)) begin
        comp_name_hit = 1'b1;
        comp_slot     = int'(f);
      end
    end
  end

  always_comb begin
    if (!comp_name_hit) begin
      comp_ctx_ok = 1'b1; comp_scope_ok = 1'b1; comp_evgen_ok = 1'b1;
      comp_seq_ok = 1'b1; comp_prod_ok = 1'b1; comp_not_dup = 1'b1;
    end else begin
      comp_ctx_ok   = tbl_q[comp_bank][comp_slot].key.ctx.ctx_generation
                      == comp_g.key.ctx.ctx_generation;
      comp_scope_ok = tbl_q[comp_bank][comp_slot].key.scope == comp_g.key.scope;
      comp_evgen_ok = tbl_q[comp_bank][comp_slot].key.event_generation
                      == comp_g.key.event_generation;
      comp_seq_ok   = tbl_q[comp_bank][comp_slot].key.seq == comp_g.key.seq;
      comp_prod_ok  = tbl_q[comp_bank][comp_slot].producer_id == comp_g.producer_id;
      comp_not_dup  = tbl_q[comp_bank][comp_slot].status == EV_PENDING;
    end
  end

  assign comp_apply = comp_fire && comp_status_ok && comp_route_ok && comp_name_hit &&
                      comp_ctx_ok && comp_scope_ok && comp_evgen_ok && comp_seq_ok &&
                      comp_prod_ok && comp_not_dup;

  // Effective status for same-cycle readers: committed state overlaid by the
  // terminal a granted completion writes this edge.
  function automatic event_status_t eff_status(input int b, input int f);
    if (comp_apply && (comp_bank == b) && (comp_slot == f)) begin
      return comp_g.status;
    end
    return tbl_q[b][f].status;
  endfunction

  function automatic logic status_is_fail(input event_status_t s);
    return (s == EV_ERROR) || (s == EV_TIMEOUT) || (s == EV_RESET);
  endfunction

  // ------------------------------------------------------------------
  // Allocation path.
  // ------------------------------------------------------------------
  logic alloc_route_ok;
  logic alloc_dup_live;
  int   alloc_bank;
  int   alloc_free_slot;
  logic alloc_free_ok;
  logic alloc_full;
  logic alloc_fire;

  assign alloc_route_ok = (alloc_key_i.ctx.tile_id == TILE_ID) &&
                          (alloc_key_i.ctx.ctx_slot < word_t'(CONTEXT_COUNT));

  always_comb begin
    alloc_bank = (alloc_key_i.ctx.ctx_slot < word_t'(CONTEXT_COUNT))
               ? int'(alloc_key_i.ctx.ctx_slot) : 0;
  end

  // Same physical name (bank-local local_event_id) while live is forbidden,
  // even under a new event_generation.
  always_comb begin
    alloc_dup_live = 1'b0;
    for (int unsigned f = 0; f < FRONTIER; f++) begin
      if (tbl_q[alloc_bank][f].live &&
          (tbl_q[alloc_bank][f].key.local_event_id == alloc_key_i.local_event_id)) begin
        alloc_dup_live = 1'b1;
      end
    end
  end

  always_comb begin
    alloc_free_ok    = 1'b0;
    alloc_free_slot  = 0;
    for (int unsigned f = 0; f < FRONTIER; f++) begin
      if (!tbl_q[alloc_bank][f].live) begin
        alloc_free_ok   = 1'b1;
        alloc_free_slot = int'(f);
      end
    end
  end

  assign alloc_full     = alloc_route_ok && !alloc_dup_live && !alloc_free_ok;
  assign alloc_fire     = alloc_valid_i && alloc_route_ok && !alloc_dup_live && alloc_free_ok;
  assign alloc_ready_o  = alloc_fire;  // full is a plain stall, never a reject
  assign alloc_reject_o = alloc_valid_i && (!alloc_route_ok || alloc_dup_live);

  // ------------------------------------------------------------------
  // WAIT / WAITALL issue: route, shape, duplicate operands, then per-operand
  // resolution by ONE full-key equality against the live table.
  // ------------------------------------------------------------------
  logic wait_route_ok;
  int   wait_bank;
  logic wait_shape_ok;
  logic wait_dup_ok;
  logic wait_go;

  assign wait_route_ok = (wait_ctx_i.tile_id == TILE_ID) &&
                         (wait_ctx_i.ctx_slot < word_t'(CONTEXT_COUNT));
  always_comb begin
    wait_bank = (wait_ctx_i.ctx_slot < word_t'(CONTEXT_COUNT))
              ? int'(wait_ctx_i.ctx_slot) : 0;
  end

  // Shape: WAIT takes exactly one operand, WAITALL two..OPERANDS.
  assign wait_shape_ok = (wait_count_i >= word_t'(1)) && (wait_count_i <= word_t'(OPERANDS)) &&
                         (wait_all_i ? (wait_count_i >= word_t'(2)) : (wait_count_i == word_t'(1)));

  always_comb begin
    wait_dup_ok = 1'b1;
    for (int unsigned i = 0; i < OPERANDS; i++) begin
      for (int unsigned j = 0; j < OPERANDS; j++) begin
        if ((j > i) && (word_t'(i) < wait_count_i) && (word_t'(j) < wait_count_i) &&
            (wait_keys_i[i] == wait_keys_i[j])) begin
          wait_dup_ok = 1'b0;
        end
      end
    end
  end

  assign wait_go = wait_valid_i && wait_route_ok && wait_shape_ok && wait_dup_ok;

  logic [OPERANDS-1:0] wait_en, wop_route_ok, wop_hit, wop_rule_ok, wop_fut_ok;
  logic [OPERANDS-1:0] wres_ok, wterm, wfail;
  logic wait_ops_ok;
  int wop_bank [OPERANDS];
  int wop_slot [OPERANDS];

  always_comb begin
    for (int unsigned i = 0; i < OPERANDS; i++) begin
      wait_en[i]     = word_t'(i) < wait_count_i;
      wop_route_ok[i] = (wait_keys_i[i].ctx.tile_id == TILE_ID) &&
                        (wait_keys_i[i].ctx.ctx_slot < word_t'(CONTEXT_COUNT));
      wop_bank[i] = (wait_keys_i[i].ctx.ctx_slot < word_t'(CONTEXT_COUNT))
                  ? int'(wait_keys_i[i].ctx.ctx_slot) : 0;
      wop_hit[i]  = 1'b0;
      wop_slot[i] = 0;
      if (wop_route_ok[i]) begin
        for (int unsigned f = 0; f < FRONTIER; f++) begin
          if (tbl_q[wop_bank[i]][f].live && (tbl_q[wop_bank[i]][f].key == wait_keys_i[i])) begin
            wop_hit[i]  = 1'b1;
            wop_slot[i] = int'(f);
          end
        end
      end
      wop_rule_ok[i] = !wop_hit[i] ||
                       (tbl_q[wop_bank[i]][wop_slot[i]].key.scope != EV_SCOPE_TASK_PRIVATE) ||
                       (tbl_q[wop_bank[i]][wop_slot[i]].key.ctx == wait_ctx_i);
      wop_fut_ok[i]  = !wop_hit[i] ||
                       (tbl_q[wop_bank[i]][wop_slot[i]].future_uses != word_t'(0));
      wres_ok[i] = wop_route_ok[i] && wop_hit[i] && wop_rule_ok[i] && wop_fut_ok[i];
      wterm[i]   = wres_ok[i] && (eff_status(wop_bank[i], wop_slot[i]) == EV_DONE);
      wfail[i]   = wres_ok[i] && status_is_fail(eff_status(wop_bank[i], wop_slot[i]));
    end
  end

  assign wait_ops_ok = (&(wres_ok | ~wait_en)) & (|wait_en);

  // Context fault routing: a routable (tile, slot) fault reaches the one
  // record it addresses; generation matching happens inside the record.
  logic fault_route_ok;
  int   fault_bank;

  assign fault_route_ok = fault_valid_i && (fault_ctx_i.tile_id == TILE_ID) &&
                          (fault_ctx_i.ctx_slot < word_t'(CONTEXT_COUNT));
  always_comb begin
    fault_bank = (fault_ctx_i.ctx_slot < word_t'(CONTEXT_COUNT))
               ? int'(fault_ctx_i.ctx_slot) : 0;
  end

  // ------------------------------------------------------------------
  // Per-Context wait records (single writer each) + probe machinery.
  // ------------------------------------------------------------------
  logic [CONTEXT_COUNT-1:0] rec_issue_valid, rec_busy, rec_fire, rec_dok, rec_dfail, rec_park;
  logic [CONTEXT_COUNT-1:0] rec_armed, rec_wake_ok, rec_wake_fail, rec_fault_valid;

  event_key_t [OPERANDS-1:0] rec_keys [CONTEXT_COUNT];
  logic probe_route [CONTEXT_COUNT][OPERANDS];
  logic [OPERANDS-1:0] probe_hit  [CONTEXT_COUNT];
  logic [OPERANDS-1:0] probe_term [CONTEXT_COUNT];
  logic [OPERANDS-1:0] probe_fail [CONTEXT_COUNT];
  logic [OPERANDS-1:0] rec_release [CONTEXT_COUNT];
  int rel_bank [CONTEXT_COUNT][OPERANDS];
  int rel_slot [CONTEXT_COUNT][OPERANDS];

  always_comb begin
    for (int unsigned c = 0; c < CONTEXT_COUNT; c++) begin
      rec_issue_valid[c] = wait_go && (int'(c) == wait_bank);
      rec_fault_valid[c] = fault_route_ok && (int'(c) == fault_bank);
    end
  end

  // Probe each record's stored keys: full-key equality on live slots (a
  // pinned entry can never be reclaimed, so an armed record always hits),
  // effective status for wake verdicts, resolved slot for pin release.
  always_comb begin
    for (int unsigned c = 0; c < CONTEXT_COUNT; c++) begin
      for (int unsigned i = 0; i < OPERANDS; i++) begin
        probe_route[c][i] = (rec_keys[c][i].ctx.tile_id == TILE_ID) &&
                            (rec_keys[c][i].ctx.ctx_slot < word_t'(CONTEXT_COUNT));
        rel_bank[c][i] = (rec_keys[c][i].ctx.ctx_slot < word_t'(CONTEXT_COUNT))
                       ? int'(rec_keys[c][i].ctx.ctx_slot) : 0;
        probe_hit[c][i] = 1'b0;
        rel_slot[c][i]  = 0;
        if (probe_route[c][i]) begin
          for (int unsigned f = 0; f < FRONTIER; f++) begin
            if (tbl_q[rel_bank[c][i]][f].live && (tbl_q[rel_bank[c][i]][f].key == rec_keys[c][i])) begin
              probe_hit[c][i] = 1'b1;
              rel_slot[c][i]  = int'(f);
            end
          end
        end
        probe_term[c][i] = probe_hit[c][i] &&
                           (eff_status(rel_bank[c][i], rel_slot[c][i]) == EV_DONE);
        probe_fail[c][i] = probe_hit[c][i] &&
                           status_is_fail(eff_status(rel_bank[c][i], rel_slot[c][i]));
      end
    end
  end

  for (genvar gc = 0; gc < int'(CONTEXT_COUNT); gc++) begin : g_wait_record
    wait_record #(
      .OPERANDS(OPERANDS)
    ) u_wait_record (
      .clk_i          (clk_i),
      .rst_ni         (rst_ni),
      .issue_valid_i  (rec_issue_valid[gc]),
      .issue_gen_i    (wait_ctx_i.ctx_generation),
      .issue_count_i  (wait_count_i),
      .issue_keys_i   (wait_keys_i),
      .issue_res_ok_i (wres_ok),
      .issue_term_i   (wterm),
      .issue_fail_i   (wfail),
      .busy_o         (rec_busy[gc]),
      .issue_fire_o   (rec_fire[gc]),
      .direct_ok_o    (rec_dok[gc]),
      .direct_fail_o  (rec_dfail[gc]),
      .park_o         (rec_park[gc]),
      .armed_o        (rec_armed[gc]),
      .stored_keys_o  (rec_keys[gc]),
      .eval_cycle_i   (comp_fire),
      .probe_term_i   (probe_term[gc]),
      .probe_fail_i   (probe_fail[gc]),
      .wake_ok_o      (rec_wake_ok[gc]),
      .wake_fail_o    (rec_wake_fail[gc]),
      .release_o      (rec_release[gc]),
      .fault_valid_i  (rec_fault_valid[gc]),
      .fault_gen_i    (fault_ctx_i.ctx_generation)
    );
  end

  assign wait_ready_o  = |rec_fire;
  assign wait_reject_o = wait_valid_i && !(|rec_fire);
  assign direct_ok_o   = |rec_dok;
  assign direct_fail_o = |rec_dfail;
  assign park_o        = |rec_park;
  assign armed_o       = rec_armed;
  assign wake_ok_o     = rec_wake_ok;
  assign wake_fail_o   = rec_wake_fail;

  // ------------------------------------------------------------------
  // Same-edge slot deltas: pins accumulate across records (wake/fault
  // release, park acquire); the single issue head decrements each of its
  // distinct operand slots' future once.
  // ------------------------------------------------------------------
  always_comb begin
    for (int unsigned b = 0; b < CONTEXT_COUNT; b++) begin
      for (int unsigned f = 0; f < FRONTIER; f++) begin
        dref[b][f]      = 0;
        dec_fut[b][f]   = 1'b0;
        slot_wr_q[b][f] = alloc_fire && (alloc_bank == int'(b)) && (alloc_free_slot == int'(f));
      end
    end
    if (|rec_fire) begin
      for (int unsigned i = 0; i < OPERANDS; i++) begin
        if (wait_en[i]) begin
          dec_fut[wop_bank[i]][wop_slot[i]] = 1'b1;
          if (park_o) begin
            dref[wop_bank[i]][wop_slot[i]] = dref[wop_bank[i]][wop_slot[i]] + 1;
          end
        end
      end
    end
    for (int unsigned c = 0; c < CONTEXT_COUNT; c++) begin
      for (int unsigned i = 0; i < OPERANDS; i++) begin
        if (rec_release[c][i] && probe_hit[c][i]) begin
          dref[rel_bank[c][i]][rel_slot[c][i]] = dref[rel_bank[c][i]][rel_slot[c][i]] - 1;
        end
      end
    end
  end

  // Reclaim on post-delta values; a slot an allocation writes this edge is
  // freshly PENDING and never reclaimed.
  always_comb begin
    for (int unsigned b = 0; b < CONTEXT_COUNT; b++) begin
      for (int unsigned f = 0; f < FRONTIER; f++) begin
        slot_clr_q[b][f] = tbl_q[b][f].live && !slot_wr_q[b][f] &&
          (((comp_apply && (comp_bank == int'(b)) && (comp_slot == int'(f))) ||
            (tbl_q[b][f].status != EV_PENDING)) &&
           ((int'(tbl_q[b][f].consumer_refs) + dref[b][f]) == 0) &&
           (!dec_fut[b][f] ? (tbl_q[b][f].future_uses == word_t'(0))
                           : (tbl_q[b][f].future_uses == word_t'(1))));
      end
    end
  end

  // ------------------------------------------------------------------
  // Persistent entry storage update (single writer per slot).
  // ------------------------------------------------------------------
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      for (int unsigned b = 0; b < CONTEXT_COUNT; b++) begin
        for (int unsigned f = 0; f < FRONTIER; f++) begin
          tbl_q[b][f] <= '0;
        end
      end
    end else begin
      for (int unsigned b = 0; b < CONTEXT_COUNT; b++) begin
        for (int unsigned f = 0; f < FRONTIER; f++) begin
          if (slot_wr_q[b][f]) begin
            tbl_q[b][f].live          <= 1'b1;
            tbl_q[b][f].key           <= alloc_key_i;
            tbl_q[b][f].kind          <= alloc_kind_i;
            tbl_q[b][f].status        <= EV_PENDING;
            tbl_q[b][f].producer_id   <= alloc_producer_i;
            tbl_q[b][f].error_code    <= word_t'(0);
            tbl_q[b][f].consumer_refs <= '0;
            tbl_q[b][f].future_uses   <= alloc_future_i;
          end else begin
            if (comp_apply && (comp_bank == int'(b)) && (comp_slot == int'(f))) begin
              tbl_q[b][f].status     <= comp_g.status;  // sticky terminal
              tbl_q[b][f].error_code <= comp_g.error_code;
            end
            if (dref[b][f] != 0) begin
              tbl_q[b][f].consumer_refs <= REF_W'(int'(tbl_q[b][f].consumer_refs) + dref[b][f]);
            end
            if (dec_fut[b][f]) begin
              tbl_q[b][f].future_uses <= tbl_q[b][f].future_uses - word_t'(1);
            end
            if (slot_clr_q[b][f]) begin
              tbl_q[b][f].live <= 1'b0;  // terminal, refs==0, future==0
            end
          end
        end
      end
    end
  end

  // ------------------------------------------------------------------
  // Per-Context counters. Each source (completion / allocation / WAIT)
  // contributes 0 or 1 to any one counter per cycle; the three sources are
  // independent rejects on one edge, so their contributions sum (0..3) and
  // the sum is applied with saturation. Wrap is forbidden (C015).
  // ------------------------------------------------------------------
  logic [NCNT-1:0]         inc_comp  [CONTEXT_COUNT];
  logic [NCNT-1:0]         inc_alloc [CONTEXT_COUNT];
  logic [NCNT-1:0]         inc_wait  [CONTEXT_COUNT];
  int                      inc_n     [CONTEXT_COUNT][NCNT];
  logic [COUNTER_BITS-1:0] cnt_q     [CONTEXT_COUNT][NCNT];
  logic [COUNTER_BITS-1:0] cnt_next  [CONTEXT_COUNT][NCNT];
  logic [COUNTER_BITS+1:0] cnt_ext;
  int                      comp_bucket;

  always_comb begin
    for (int unsigned c = 0; c < CONTEXT_COUNT; c++) begin
      inc_comp[c]  = '0;
      inc_alloc[c] = '0;
      inc_wait[c]  = '0;
    end
    // completion: bucket is the routed Context when routable, else Context 0
    comp_bucket = comp_route_ok ? comp_bank : 0;
    if (comp_fire) begin
      if (!comp_status_ok) begin
        inc_comp[comp_bucket][C_PROTO] = 1'b1;  // non-terminal status payload
      end else if (!comp_route_ok) begin
        inc_comp[0][C_OWNER] = 1'b1;
        inc_comp[0][C_WCTX]  = 1'b1;
      end else if (!comp_name_hit) begin
        inc_comp[comp_bucket][C_UNK] = 1'b1;
      end else if (!comp_ctx_ok) begin
        inc_comp[comp_bucket][C_OWNER] = 1'b1;
        inc_comp[comp_bucket][C_WCTX]  = 1'b1;
      end else if (!comp_scope_ok) begin
        inc_comp[comp_bucket][C_OWNER] = 1'b1;
        inc_comp[comp_bucket][C_WSCP]  = 1'b1;
      end else if (!comp_evgen_ok) begin
        inc_comp[comp_bucket][C_STALE] = 1'b1;
        inc_comp[comp_bucket][C_WEVG]  = 1'b1;
      end else if (!comp_seq_ok) begin
        inc_comp[comp_bucket][C_STALE] = 1'b1;
        inc_comp[comp_bucket][C_WSEQ]  = 1'b1;
      end else if (!comp_prod_ok) begin
        inc_comp[comp_bucket][C_OWNER] = 1'b1;
      end else if (!comp_not_dup) begin
        inc_comp[comp_bucket][C_DUP] = 1'b1;
      end
    end
    // allocation: registration violations are protocol_error; full frontier
    // is backpressure and counts capacity_reject per presented cycle
    if (alloc_valid_i && !alloc_route_ok) begin
      inc_alloc[0][C_PROTO] = 1'b1;
    end
    if (alloc_valid_i && alloc_route_ok && alloc_dup_live) begin
      inc_alloc[alloc_bank][C_PROTO] = 1'b1;
    end
    if (alloc_valid_i && alloc_full) begin
      inc_alloc[alloc_bank][C_CAP] = 1'b1;
    end
    // WAIT: every rejected presented issue is one registration violation
    // (protocol_error only); a well-formed fault-killed issue counts nothing
    if (wait_valid_i && !wait_route_ok) begin
      inc_wait[0][C_PROTO] = 1'b1;
    end
    if (wait_valid_i && wait_route_ok && (!wait_shape_ok || !wait_dup_ok)) begin
      inc_wait[wait_bank][C_PROTO] = 1'b1;
    end
    if (wait_go && (!wait_ops_ok || rec_busy[wait_bank])) begin
      inc_wait[wait_bank][C_PROTO] = 1'b1;
    end
    // sum the independent sources per counter, saturate on any carry
    for (int unsigned c = 0; c < CONTEXT_COUNT; c++) begin
      for (int unsigned k = 0; k < NCNT; k++) begin
        inc_n[c][k] = int'(inc_comp[c][k]) + int'(inc_alloc[c][k]) + int'(inc_wait[c][k]);
        cnt_ext = {2'b00, cnt_q[c][k]} + (COUNTER_BITS+2)'(inc_n[c][k]);
        cnt_next[c][k] = (|cnt_ext[COUNTER_BITS+1:COUNTER_BITS]) ? '1
                                                               : cnt_ext[COUNTER_BITS-1:0];
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      for (int unsigned c = 0; c < CONTEXT_COUNT; c++) begin
        for (int unsigned k = 0; k < NCNT; k++) begin
          cnt_q[c][k] <= '0;
        end
      end
    end else begin
      for (int unsigned c = 0; c < CONTEXT_COUNT; c++) begin
        for (int unsigned k = 0; k < NCNT; k++) begin
          cnt_q[c][k] <= cnt_next[c][k];
        end
      end
    end
  end

  always_comb begin
    for (int unsigned c = 0; c < CONTEXT_COUNT; c++) begin
      counts_o[c].unknown_event_notify   = word_t'(cnt_q[c][C_UNK]);
      counts_o[c].wrong_owner            = word_t'(cnt_q[c][C_OWNER]);
      counts_o[c].stale_sequence         = word_t'(cnt_q[c][C_STALE]);
      counts_o[c].duplicate_signal       = word_t'(cnt_q[c][C_DUP]);
      counts_o[c].capacity_reject        = word_t'(cnt_q[c][C_CAP]);
      counts_o[c].protocol_error         = word_t'(cnt_q[c][C_PROTO]);
      counts_o[c].wrong_context          = word_t'(cnt_q[c][C_WCTX]);
      counts_o[c].wrong_scope            = word_t'(cnt_q[c][C_WSCP]);
      counts_o[c].wrong_sequence         = word_t'(cnt_q[c][C_WSEQ]);
      counts_o[c].wrong_event_generation = word_t'(cnt_q[c][C_WEVG]);
      counts_seen_o[c].unknown_event_notify   = (cnt_q[c][C_UNK]   != '0);
      counts_seen_o[c].wrong_owner            = (cnt_q[c][C_OWNER] != '0);
      counts_seen_o[c].stale_sequence         = (cnt_q[c][C_STALE] != '0);
      counts_seen_o[c].duplicate_signal       = (cnt_q[c][C_DUP]   != '0);
      counts_seen_o[c].capacity_reject        = (cnt_q[c][C_CAP]   != '0);
      counts_seen_o[c].protocol_error         = (cnt_q[c][C_PROTO] != '0);
      counts_seen_o[c].wrong_context          = (cnt_q[c][C_WCTX]  != '0);
      counts_seen_o[c].wrong_scope            = (cnt_q[c][C_WSCP]  != '0);
      counts_seen_o[c].wrong_sequence         = (cnt_q[c][C_WSEQ]  != '0);
      counts_seen_o[c].wrong_event_generation = (cnt_q[c][C_WEVG]  != '0);
    end
  end

  // ------------------------------------------------------------------
  // Committed-state query: full-key lookup, absent => all-zero view.
  // ------------------------------------------------------------------
  logic q_route;
  int   q_bank;
  logic q_hit;
  int   q_slot;

  always_comb begin
    q_route = (query_key_i.ctx.tile_id == TILE_ID) &&
              (query_key_i.ctx.ctx_slot < word_t'(CONTEXT_COUNT));
    q_bank  = (query_key_i.ctx.ctx_slot < word_t'(CONTEXT_COUNT))
            ? int'(query_key_i.ctx.ctx_slot) : 0;
    q_hit   = 1'b0;
    q_slot  = 0;
    if (q_route) begin
      for (int unsigned f = 0; f < FRONTIER; f++) begin
        if (tbl_q[q_bank][f].live && (tbl_q[q_bank][f].key == query_key_i)) begin
          q_hit  = 1'b1;
          q_slot = int'(f);
        end
      end
    end
    query_o = '0;
    if (q_hit) begin
      query_o.live          = 1'b1;
      query_o.status        = tbl_q[q_bank][q_slot].status;
      query_o.kind          = tbl_q[q_bank][q_slot].kind;
      query_o.producer_id   = tbl_q[q_bank][q_slot].producer_id;
      query_o.error_code    = tbl_q[q_bank][q_slot].error_code;
      query_o.consumer_refs = word_t'(tbl_q[q_bank][q_slot].consumer_refs);
      query_o.future_uses   = tbl_q[q_bank][q_slot].future_uses;
    end
  end

endmodule
