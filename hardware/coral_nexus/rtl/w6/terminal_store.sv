// terminal_store.sv — C033 (W6-02) per-Context admitted lifecycle, reliable
// terminal storage and PHASE_SIGNAL emitter.
//
// Scope (G2_scalar.md #c033 items 4-5, W6 plan §5.2 items 4-7): one
// admitted lifecycle per Context slot with an independent terminal reserve.
// The evidence inputs only report facts produced elsewhere: no W2/W4/Group
// resource return, no timer; an abort or elapsed time never publishes or
// drops a terminal (w6_terminal_reset.contract.yaml#task_terminal).
//
// Lifecycle: LS_FREE -START-> LS_RUN -DRAIN-> LS_RRPEND -RR fired->
// LS_RRSENT -RETIRE-> LS_RETIRED -FINISH-> LS_TERMPEND -TASK_COMPLETION
// fired-> LS_FREE. active_o clears ONLY on the TASK_COMPLETION fire; a slot
// is reusable by a new START only after it.
//
// cmd port (routed by cmd_owner_i.ctx): key violations (tile mismatch,
// ctx_slot out of range, full-owner mismatch) and order violations (kind
// illegal in the state: duplicate DRAIN/RETIRE/FINISH, RETIRE before the RR
// fire, FINISH before RETIRE authority, slot reuse) are CONSUMED with a
// combinational cmd_reject_o pulse in the fire cycle (sample it with the
// fire edge), no state change. Absent evidence is BACKPRESSURE
// (cmd_ready_o=0, no reject): CMD_DRAIN needs exec_drained &&
// no_old_inflight && region_ended (and no phase byte still owed), CMD_FINISH
// needs local_reclaimed && w4_closed (real W4 closure, especially TIMEOUT).
// Storing never depends on an unrelated peer or on egress backpressure.
//
// Phase admission cutline (EVT-09/§5.2: no PHASE_SIGNAL may trail a drained
// or cancelled task): accepted only while the slot is LS_RUN, not
// fail-latched, its phase storage free, and no abort of the same owner is
// accepted on that edge. A same-edge phase + DRAIN of one owner never lets
// both through: DRAIN holds until the accepted phase message has fired, so a
// slot stays LS_RUN while it owes a phase byte and the message task_key
// (taken from the slot owner) can never outlive its owner. Unrelated
// Contexts are independent. No dedup, no suppression: every accepted
// instruction emits exactly one PHASE_SIGNAL (duplicates included, EVT-14);
// phase_seq_i is the link-side sequencer's value, only carried through.
//
// Abort (in-band event, no ready): accepted only for the live full owner
// before the terminal is stored and with status != TS_DONE; stale identity,
// free slot, DONE status or a stored terminal is a no-effect event. An
// accepted abort only LATCHES the first failure (first wins; a later or
// same-edge more severe abort never overwrites). CMD_FINISH — including on
// the same edge as an accepted abort — stores the latched failure
// status/ref instead of success; terminal bytes freeze at LS_TERMPEND.
//
// Egress: RETIRE_READY and TASK_COMPLETION share one round-robin-fair port.
// An offer parks the selected SLOT INDEX until fire and drives the payload
// combinationally from that slot's record, which is immutable while offered
// (no writer exists in LS_RRPEND/LS_TERMPEND before the fire itself, and a
// slot owing a phase byte is pinned in LS_RUN): offers are stable under
// stall and can never be stolen by a peer, with no added latency. An RR
// offer carries is_completion=0 with neutral status/ref (never a bogus
// terminal); the drain evidence sequence captured at DRAIN rides both kinds.
//
// rst_ni is the synchronous active-low COLD reset, never a functional
// protocol reset.
`timescale 1ns/1ps

module terminal_store
  import c033_event_pkg::*;
#(
  // CONTEXT_COUNT has no default: the bench supplies the real C1/C2 value.
  parameter int unsigned CONTEXT_COUNT,
  parameter word_t TILE_ID = 64'd0
) (
  input  logic         clk_i,
  input  logic         rst_ni,

  // Lifecycle command port (all kinds share it; routed by cmd_owner_i).
  input  logic           cmd_valid_i,
  input  cmd_kind_t      cmd_kind_i,
  input  owner_t         cmd_owner_i,
  input  logic           exec_drained_i,
  input  logic           no_old_inflight_i,
  input  logic           region_ended_i,
  input  logic           local_reclaimed_i,
  input  logic           w4_closed_i,
  input  task_status_t   finish_status_i,
  input  word_t          drain_evidence_seq_i,
  input  word_t          first_fault_ref_i,
  output logic           cmd_ready_o,
  output logic           cmd_reject_o,

  // Abort / cancel observation event.
  input  logic           abort_valid_i,
  input  owner_t         abort_owner_i,
  input  task_status_t   abort_status_i,
  input  word_t          abort_ref_i,

  // Phase instruction ingress (one accepted instruction = one message).
  input  logic           phase_valid_i,
  input  owner_t         phase_owner_i,
  input  logic           phase_direction_i,
  input  word_t          phase_seq_i,
  output logic           phase_ready_o,

  // PHASE_SIGNAL egress.
  output logic           phase_valid_o,
  output phase_msg_t     phase_o,
  input  logic           phase_ready_i,

  // RETIRE_READY / TASK_COMPLETION egress (one port).
  output logic           msg_valid_o,
  output terminal_msg_t  msg_o,
  input  logic           msg_ready_i,

  // Committed per-Context lifecycle status (packed).
  output logic [CONTEXT_COUNT-1:0] active_o,
  output logic [CONTEXT_COUNT-1:0] ready_pending_o,
  output logic [CONTEXT_COUNT-1:0] retire_sent_o,
  output logic [CONTEXT_COUNT-1:0] completion_pending_o
);

  localparam logic [2:0] LS_FREE = 3'd0, LS_RUN = 3'd1, LS_RRPEND = 3'd2,
    LS_RRSENT = 3'd3, LS_RETIRED = 3'd4, LS_TERMPEND = 3'd5;

  // Index width; >= 1 bit even for CONTEXT_COUNT == 1 (no 0-width selects).
  localparam int unsigned SLOT_W = (CONTEXT_COUNT <= 1) ? 1 : $clog2(CONTEXT_COUNT);
  localparam word_t CC_W = word_t'(CONTEXT_COUNT);

  // Per-slot record. owner/ev_seq/term_* are immutable from their write
  // until the owning egress fire; ph_* are immutable while ph_full.
  typedef struct packed {
    logic [2:0]   state;       // LS_*
    owner_t       owner;       // admitted full owner (valid unless LS_FREE)
    word_t        ev_seq;      // drain evidence sequence captured at DRAIN
    task_status_t term_status; // stored terminal status (frozen in LS_TERMPEND)
    word_t        term_ref;    // stored first-fault reference
    logic         fail_latch;  // first failure latched from an abort
    task_status_t fail_status;
    word_t        fail_ref;
    logic         ph_full;     // phase byte owed (accepted, unfired)
    logic         ph_dir;      // its direction
    word_t        ph_seq;      // its link-side message sequence
  } ctx_rec_t;

  ctx_rec_t r_ctx [CONTEXT_COUNT];

  // Egress offers: parked slot index + valid. No payload copy — the payload
  // is selected from the parked record (immutable while offered). Round-
  // robin grant pointers advance past the fired slot.
  logic              r_msg_valid;
  logic [SLOT_W-1:0] r_msg_sel;
  logic [SLOT_W-1:0] r_ptr_msg;
  logic              r_ph_valid;
  logic [SLOT_W-1:0] r_ph_sel;
  logic [SLOT_W-1:0] r_ptr_ph;

  // ------------------------------------------------------------------
  // Routing and bounds guards: every array read below is gated to a
  // bounds-checked slot, so no out-of-range index can occur.
  // ------------------------------------------------------------------
  function automatic logic route_ok(owner_t o);
    route_ok = (o.ctx.tile_id == TILE_ID) && (o.ctx.ctx_slot < CC_W);
  endfunction

  logic route_cmd, route_abort, route_phase;
  logic [SLOT_W-1:0] sel_cmd, sel_abort, sel_phase;
  assign route_cmd   = route_ok(cmd_owner_i);
  assign route_abort = route_ok(abort_owner_i);
  assign route_phase = route_ok(phase_owner_i);
  assign sel_cmd   = route_cmd   ? cmd_owner_i.ctx.ctx_slot[SLOT_W-1:0]   : '0;
  assign sel_abort = route_abort ? abort_owner_i.ctx.ctx_slot[SLOT_W-1:0] : '0;
  assign sel_phase = route_phase ? phase_owner_i.ctx.ctx_slot[SLOT_W-1:0] : '0;

  // Abort: scoped, live, pre-terminal, non-DONE; anything else no effect.
  logic abort_hit;
  assign abort_hit = abort_valid_i && route_abort
    && (r_ctx[sel_abort].state != LS_FREE)
    && (r_ctx[sel_abort].state != LS_TERMPEND)
    && (r_ctx[sel_abort].owner == abort_owner_i)
    && (abort_status_i != TS_DONE);

  // Phase admission: LS_RUN only, not cancelled, own storage free, and no
  // same-edge accepted abort of this owner. No reject pin: everything else
  // backpressures (a post-cancel producer stays blocked, which is the
  // intended cutline).
  assign phase_ready_o = route_phase
    && (r_ctx[sel_phase].state == LS_RUN)
    && !r_ctx[sel_phase].fail_latch
    && (r_ctx[sel_phase].owner == phase_owner_i)
    && !r_ctx[sel_phase].ph_full
    && !(abort_hit && (sel_abort == sel_phase));

  // Same-owner phase accepted on the DRAIN edge: the DRAIN must hold until
  // that phase byte fired (the two never pass together).
  logic phase_edge_cmd;
  assign phase_edge_cmd = phase_valid_i && phase_ready_o && (sel_phase == sel_cmd);

  // ------------------------------------------------------------------
  // Command decode: cmd_ready_o is the pure consume function; cmd_reject_o
  // pulses only in the cycle a violating command is consumed.
  // ------------------------------------------------------------------
  logic       cmd_owner_ok;
  logic [2:0] st_cmd;
  logic       acc_start, acc_drain, acc_retire, acc_finish;

  assign st_cmd       = r_ctx[sel_cmd].state;
  assign cmd_owner_ok = route_cmd && (st_cmd != LS_FREE)
    && (r_ctx[sel_cmd].owner == cmd_owner_i);

  always_comb begin
    cmd_ready_o  = 1'b0;
    cmd_reject_o = 1'b0;
    acc_start    = 1'b0;
    acc_drain    = 1'b0;
    acc_retire   = 1'b0;
    acc_finish   = 1'b0;
    if (!route_cmd) begin
      cmd_ready_o  = 1'b1;          // key violation: consume and reject
      cmd_reject_o = cmd_valid_i;
    end else begin
      case (cmd_kind_i)
        CMD_START: begin
          if (st_cmd == LS_FREE) begin
            cmd_ready_o = 1'b1;
            acc_start   = cmd_valid_i;
          end else begin
            cmd_ready_o  = 1'b1;    // reuse before the last terminal fire
            cmd_reject_o = cmd_valid_i;
          end
        end
        CMD_DRAIN: begin
          if (!cmd_owner_ok || (st_cmd != LS_RUN)) begin
            cmd_ready_o  = 1'b1;    // wrong owner / order (duplicate)
            cmd_reject_o = cmd_valid_i;
          end else if (!(exec_drained_i && no_old_inflight_i && region_ended_i)
                       || r_ctx[sel_cmd].ph_full || phase_edge_cmd) begin
            cmd_ready_o = 1'b0;     // absent evidence or owed phase byte
          end else begin
            cmd_ready_o = 1'b1;
            acc_drain   = cmd_valid_i;
          end
        end
        CMD_RETIRE: begin
          if (!cmd_owner_ok || (st_cmd != LS_RRSENT)) begin
            cmd_ready_o  = 1'b1;    // RETIRE only after the RR fire
            cmd_reject_o = cmd_valid_i;
          end else begin
            cmd_ready_o = 1'b1;
            acc_retire  = cmd_valid_i;
          end
        end
        CMD_FINISH: begin
          if (!cmd_owner_ok || (st_cmd != LS_RETIRED)) begin
            cmd_ready_o  = 1'b1;    // FINISH only after authority
            cmd_reject_o = cmd_valid_i;
          end else if (!(local_reclaimed_i && w4_closed_i)) begin
            cmd_ready_o = 1'b0;     // absent evidence (real W4 closure)
          end else begin
            cmd_ready_o = 1'b1;
            acc_finish  = cmd_valid_i;
          end
        end
        default: begin
          cmd_ready_o  = 1'b1;
          cmd_reject_o = cmd_valid_i;
        end
      endcase
    end
  end

  // Stored-terminal bytes for an accepting FINISH. First failure wins: the
  // pre-edge latch beats everything; a same-edge accepted abort on the same
  // slot is the first failure when nothing was latched yet and beats the
  // FINISH inputs (no success out of an abort edge).
  task_status_t fin_status;
  word_t        fin_ref;
  always_comb begin
    if (r_ctx[sel_cmd].fail_latch) begin
      fin_status = r_ctx[sel_cmd].fail_status;
      fin_ref    = r_ctx[sel_cmd].fail_ref;
    end else if (abort_hit && (sel_abort == sel_cmd)) begin
      fin_status = abort_status_i;
      fin_ref    = abort_ref_i;
    end else begin
      fin_status = finish_status_i;
      fin_ref    = first_fault_ref_i;
    end
  end

  // ------------------------------------------------------------------
  // Egress arbitration. RR over the pending masks; an offer parks the slot
  // index, so the masks are only scanned while the port is idle and a
  // stalled offer can never be stolen by a peer.
  // ------------------------------------------------------------------
  logic [CONTEXT_COUNT-1:0] msg_mask, ph_mask;
  always_comb begin
    for (int unsigned c = 0; c < CONTEXT_COUNT; c++) begin
      msg_mask[c] = (r_ctx[c].state == LS_RRPEND)
                    || (r_ctx[c].state == LS_TERMPEND);
      ph_mask[c]  = r_ctx[c].ph_full
                    && !(r_ph_valid && (r_ph_sel == SLOT_W'(c)));
    end
  end

  logic        msg_pick_ok, ph_pick_ok;
  int unsigned msg_pick, ph_pick, k, idx;
  always_comb begin
    msg_pick_ok = 1'b0;
    msg_pick    = 0;
    for (k = 0; k < CONTEXT_COUNT; k++) begin
      idx = (int'(r_ptr_msg) + k) % CONTEXT_COUNT;
      if (!msg_pick_ok && msg_mask[idx]) begin
        msg_pick    = idx;
        msg_pick_ok = 1'b1;
      end
    end
    ph_pick_ok = 1'b0;
    ph_pick    = 0;
    for (k = 0; k < CONTEXT_COUNT; k++) begin
      idx = (int'(r_ptr_ph) + k) % CONTEXT_COUNT;
      if (!ph_pick_ok && ph_mask[idx]) begin
        ph_pick    = idx;
        ph_pick_ok = 1'b1;
      end
    end
  end

  assign msg_valid_o   = r_msg_valid;
  assign phase_valid_o = r_ph_valid;

  // Payloads selected from the parked record. Immutability while offered:
  //   msg: the parked slot is LS_RRPEND or LS_TERMPEND; owner/ev_seq/term_*
  //     have no writer there before the fire itself (START writes only a
  //     FREE slot, DRAIN only a RUN slot), so bytes are stable until fire.
  //   phase: ph_full pins the slot in LS_RUN (DRAIN cannot pass while owed),
  //     so owner/ph_dir/ph_seq cannot change until the fire clears ph_full.
  always_comb begin
    msg_o.owner              = r_ctx[r_msg_sel].owner;
    msg_o.drain_evidence_seq = r_ctx[r_msg_sel].ev_seq;
    if (r_ctx[r_msg_sel].state == LS_TERMPEND) begin
      msg_o.is_completion   = 1'b1;
      msg_o.status          = r_ctx[r_msg_sel].term_status;
      msg_o.first_fault_ref = r_ctx[r_msg_sel].term_ref;
    end else begin
      msg_o.is_completion   = 1'b0;  // RETIRE_READY is not a terminal
      msg_o.status          = TS_DONE;
      msg_o.first_fault_ref = 64'd0;
    end
  end

  always_comb begin
    phase_o.task_key    = r_ctx[r_ph_sel].owner.task_key;
    phase_o.direction   = r_ctx[r_ph_sel].ph_dir;
    phase_o.message_seq = r_ctx[r_ph_sel].ph_seq;
  end

  for (genvar g = 0; g < CONTEXT_COUNT; g++) begin : g_status
    assign active_o[g]             = (r_ctx[g].state != LS_FREE);
    assign ready_pending_o[g]      = (r_ctx[g].state == LS_RRPEND);
    assign retire_sent_o[g]        = (r_ctx[g].state == LS_RRSENT)
                                     || (r_ctx[g].state == LS_RETIRED)
                                     || (r_ctx[g].state == LS_TERMPEND);
    assign completion_pending_o[g] = (r_ctx[g].state == LS_TERMPEND);
  end

  // ------------------------------------------------------------------
  // Sequential core. Conflicting writes are impossible:
  //   phase accept vs phase fire: accept needs !ph_full, fire clears a
  //     ph_full slot -> distinct slots.
  //   phase accept vs DRAIN accept: same-slot pairs are excluded by
  //     phase_edge_cmd in the DRAIN gate; other slots are distinct records.
  //   cmd accepts vs msg fire: DRAIN needs LS_RUN / FINISH LS_RETIRED /
  //     START LS_FREE, msg fire touches only LS_RRPEND/LS_TERMPEND slots.
  //   abort latch vs FINISH accept: disjoint fields (fail_* vs term_*/state)
  //     and fin_* above already resolved the bytes pre-edge; the completion
  //     fire slot (LS_TERMPEND) is excluded from abort entirely.
  // ------------------------------------------------------------------
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      // Cold reset: per-element clear (never a whole-array '0).
      for (int j = 0; j < CONTEXT_COUNT; j++) begin
        r_ctx[j].state       <= LS_FREE;
        r_ctx[j].owner       <= owner_t'('0);
        r_ctx[j].ev_seq      <= 64'd0;
        r_ctx[j].term_status <= TS_DONE;
        r_ctx[j].term_ref    <= 64'd0;
        r_ctx[j].fail_latch  <= 1'b0;
        r_ctx[j].fail_status <= TS_DONE;
        r_ctx[j].fail_ref    <= 64'd0;
        r_ctx[j].ph_full     <= 1'b0;
        r_ctx[j].ph_dir      <= 1'b0;
        r_ctx[j].ph_seq      <= 64'd0;
      end
      r_msg_valid <= 1'b0;
      r_msg_sel   <= '0;
      r_ptr_msg   <= '0;
      r_ph_valid  <= 1'b0;
      r_ph_sel    <= '0;
      r_ptr_ph    <= '0;
    end else begin
      // Phase admission.
      if (phase_valid_i && phase_ready_o) begin
        r_ctx[sel_phase].ph_full <= 1'b1;
        r_ctx[sel_phase].ph_dir  <= phase_direction_i;
        r_ctx[sel_phase].ph_seq  <= phase_seq_i;
      end

      // Abort: latch the first failure only.
      if (abort_hit && !r_ctx[sel_abort].fail_latch) begin
        r_ctx[sel_abort].fail_latch  <= 1'b1;
        r_ctx[sel_abort].fail_status <= abort_status_i;
        r_ctx[sel_abort].fail_ref    <= abort_ref_i;
      end

      // Lifecycle commands.
      if (acc_start) begin
        r_ctx[sel_cmd].state       <= LS_RUN;
        r_ctx[sel_cmd].owner       <= cmd_owner_i;
        r_ctx[sel_cmd].ev_seq      <= 64'd0;
        r_ctx[sel_cmd].term_status <= TS_DONE;
        r_ctx[sel_cmd].term_ref    <= 64'd0;
        r_ctx[sel_cmd].fail_latch  <= 1'b0;
        r_ctx[sel_cmd].fail_status <= TS_DONE;
        r_ctx[sel_cmd].fail_ref    <= 64'd0;
      end
      if (acc_drain) begin
        r_ctx[sel_cmd].state  <= LS_RRPEND;
        r_ctx[sel_cmd].ev_seq <= drain_evidence_seq_i;
      end
      if (acc_retire) begin
        r_ctx[sel_cmd].state <= LS_RETIRED;
      end
      if (acc_finish) begin
        r_ctx[sel_cmd].state       <= LS_TERMPEND;
        r_ctx[sel_cmd].term_status <= fin_status;
        r_ctx[sel_cmd].term_ref    <= fin_ref;
      end

      // Message egress: park / hold / fire.
      if (r_msg_valid) begin
        if (msg_ready_i) begin
          r_msg_valid <= 1'b0;
          r_ptr_msg   <= SLOT_W'((int'(r_msg_sel) + 1) % CONTEXT_COUNT);
          if (r_ctx[r_msg_sel].state == LS_TERMPEND) begin
            // TASK_COMPLETION fired: the ONLY clear of active_o.
            r_ctx[r_msg_sel].state       <= LS_FREE;
            r_ctx[r_msg_sel].owner       <= owner_t'('0);
            r_ctx[r_msg_sel].ev_seq      <= 64'd0;
            r_ctx[r_msg_sel].term_status <= TS_DONE;
            r_ctx[r_msg_sel].term_ref    <= 64'd0;
            r_ctx[r_msg_sel].fail_latch  <= 1'b0;
            r_ctx[r_msg_sel].fail_status <= TS_DONE;
            r_ctx[r_msg_sel].fail_ref    <= 64'd0;
          end else begin
            r_ctx[r_msg_sel].state <= LS_RRSENT;
          end
        end
      end else if (msg_pick_ok) begin
        r_msg_valid <= 1'b1;
        r_msg_sel   <= SLOT_W'(msg_pick);
      end

      // Phase egress: park / hold / fire.
      if (r_ph_valid) begin
        if (phase_ready_i) begin
          r_ph_valid <= 1'b0;
          r_ptr_ph   <= SLOT_W'((int'(r_ph_sel) + 1) % CONTEXT_COUNT);
          r_ctx[r_ph_sel].ph_full <= 1'b0;
        end
      end else if (ph_pick_ok) begin
        r_ph_valid <= 1'b1;
        r_ph_sel   <= SLOT_W'(ph_pick);
      end
    end
  end

endmodule
