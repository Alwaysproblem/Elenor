// reset_endpoint.sv — C034 (W6-03) passive Tile-side reset segment endpoint.
//
// Group is the sole authority on reset order, timing and resume
// (w6_terminal_reset.contract.yaml #reset_endpoint): each RESET_PHASE_REQ is
// executed exactly once, no stage advances unprompted, and no timer changes
// functional state. Phases follow the C017 frozen order 0..6 — the encoded
// phase IS its stage number, so ordering uses the packed value and its
// bounded successor only. Drain evidence is the C017 little-endian mask
// 0..5 (accepted_zero, queue_empty, mshr_empty, transfer_quiescent,
// no_pending_debug, no_unaccepted_phase_or_post).
//
// Epoch: an authenticated epoch is one presented on the RUN_BEGIN link.
// The first epoch or a DIFFERENT generation is admitted only with no live
// Task, a truly quiet snapshot, no pending ACK/action, no active reset and
// no Group poison; a replay of the running generation never rearms.
// reset_done clears only when a NEW epoch is admitted; run activation never
// touches the first-fault latch.
//
// Sequencing: one command outstanding; the permission pointer r_stage
// (count of ACK-fired stages, 7 = epoch complete) advances only on the ACK
// fire. A badly encoded phase, wrong epoch or future/out-of-order stage is
// consumed as a protocol reject with no side effect. A replay of the
// outstanding stage is absorbed — the pending ACK is the idempotent answer.
// A duplicate of an acknowledged stage re-offers the original logical ACK
// without re-executing. A duplicate arriving while the ACK port is occupied
// is held (backpressure), never rejected: a legitimate replay is not a
// protocol error. The MARK count is the only per-phase value that must
// persist; there is no second ledger.
//
// ACK payload: r_ack_valid/r_ack_phase are the only ACK registers; the
// generation/mark/mask fields are canonical derivations — epoch generation
// (every path is epoch-checked), the saved MARK count, the canonical DRAIN
// mask 3f, else zero. Each derivation is stable while the ACK is offered
// (a run admission or new MARK cannot pass an offered ACK), so no second
// 64-bit generation or mark register exists.
//
// Freeze: issue_frozen_o rises combinationally on the FREEZE admission
// edge (a same-edge CMD_START is already rejected by the terminal store),
// holds with the active reset until the RESUME ACK fires, and also covers
// Group poison and the window before the first authorized run — work needs
// an epoch. Phase requests are never blocked by poison: poison recovery
// IS an explicit quiescent reset. reset_done never implies self-resume.
//
// Reset failure cutline: reset_failure_o is high for the whole live reset
// INCLUDING the FREEZE admission edge — an evidence-backed FINISH accepted
// anywhere in that window cannot store success in the terminal store (an
// earlier or same-edge scoped failure still wins; otherwise TS_RESET with
// reset_ref_o). The reference is LATCHED at the accepted FREEZE from
// reset_fault_ref_i and used for the rest of the reset (sweep aborts and
// the cutline), never re-read as an unstable live value.
//
// Phase effects (once per epoch):
//   FREEZE      latch freeze (registered) + instant ACK.
//   CANCEL_UNISSUED  one real ingress-clear handshake plus one scoped
//               TS_RESET abort per still-open live Task (live && !aborted
//               && !terminal-committed, Group failure ref; stored terminals
//               immutable); the ACK waits for BOTH. Under freeze the live
//               set only shrinks, so the sweep terminates.
//   DRAIN       complete only on ONE snapshot with all six evidence bits
//               simultaneously true — no cross-cycle OR, no timer, and
//               PENDING events are not drain evidence.
//   MARK        one-edge mark_events_o write; the count is captured on the
//               NEXT edge (the table's committed result — same-edge
//               accepted completions are already excluded by the writer)
//               and is the ACK payload and the saved duplicate-ACK word.
//   INVALIDATE  the offer is precondition-gated (zero references AND the
//               table ready to abandon), so valid&&ready at an edge is
//               always the single authorized fire; abandon_events pulses
//               on it; older program refs / retained consumers hold it.
//   REBUILD     fire only on the local credit owner's acceptance with the
//               live evidence all-true again (no rebuild while obligations
//               reappeared after DRAIN).
//   RESUME      takes effect at its ACK fire (freeze release + done latch);
//               the Group decides when.
//
// CANCEL_TASK classifies against the real live bindings by full TaskKey:
// absent -> stale (saturating count, idempotent); TaskKey present with a
// conflicting context identity -> explicit conflict (cancel_fault_o,
// neither owner mutated); exact live -> exactly one scoped TS_RESET abort;
// a repeat or already failure-latched / terminal-committed owner is a
// duplicate. The functional cancel wins the single abort-writer port over
// the phase sweep, so a first cancel can still bind the failure on a
// same-edge FINISH.
//
// rst_ni is the synchronous active-low COLD reset, never a functional one.
`timescale 1ns/1ps

module reset_endpoint
  import c033_event_pkg::*;
  import c034_reset_pkg::*;
#(
  parameter int unsigned CONTEXT_COUNT,
  parameter word_t TILE_ID = 64'd0
) (
  input  logic         clk_i,
  input  logic         rst_ni,

  // Group epoch (RUN_BEGIN) and Group-owned sticky poison level.
  input  logic         run_begin_i,
  input  word_t        run_generation_i,
  output logic         run_ready_o,
  input  logic         group_poisoned_i,

  // Freeze / reset status.
  output logic         issue_frozen_o,
  output logic         reset_active_o,
  output logic         reset_done_o,
  output logic         reset_failure_o,
  output word_t        reset_ref_o,

  // Passive phase request port.
  input  logic         req_valid_i,
  input  reset_phase_t req_phase_i,
  input  word_t        req_generation_i,
  output logic         req_ready_o,
  output logic         req_reject_o,

  // One registered stable ACK (RESET_PHASE_ACK payload semantics).
  output logic         ack_valid_o,
  output reset_phase_t ack_phase_o,
  output word_t        ack_generation_o,
  output word_t        ack_marked_o,
  output logic [5:0]   ack_mask_o,
  input  logic         ack_ready_i,

  // Committed live-Task view from the terminal store (source records).
  input  logic [CONTEXT_COUNT-1:0]  live_i,
  input  owner_t [CONTEXT_COUNT-1:0] live_owner_i,
  input  logic [CONTEXT_COUNT-1:0]   aborted_i,
  input  logic [CONTEXT_COUNT-1:0]   terminal_committed_i,
  input  word_t        reset_fault_ref_i,  // sampled once at FREEZE accept

  // CANCEL_UNISSUED: unaccepted ingress clear (one real handshake).
  output logic         clear_ingress_valid_o,
  input  logic         clear_ingress_ready_i,

  // Single abort writer (phase sweep + scoped CANCEL_TASK share it).
  output logic         abort_valid_o,
  output owner_t       abort_owner_o,
  output task_status_t abort_status_o,
  output word_t        abort_ref_o,

  // DRAIN evidence (C017 bit order 0..5) and live observable snapshot.
  input  logic [5:0]   drain_evidence_i,
  output logic [5:0]   is_quiescent_snapshot_o,

  // MARK_EVENTS: one-edge local table write + committed count.
  output logic         mark_events_o,
  input  word_t        mark_count_i,

  // INVALIDATE_RESIDENCY: single authorized fire.
  output logic         invalidate_valid_o,
  input  logic         invalidate_ready_i,
  input  logic         refs_zero_i,
  input  logic         abandon_ready_i,
  output logic         abandon_events_o,

  // REBUILD_CREDIT: local credit owner acceptance.
  output logic         rebuild_valid_o,
  input  logic         rebuild_ready_i,

  // Functional CANCEL_TASK.
  input  logic         cancel_valid_i,
  input  owner_t       cancel_owner_i,
  input  word_t        cancel_ref_i,
  output logic         cancel_ready_o,
  output cancel_class_t cancel_class_o,
  output logic         cancel_fault_o,
  output word_t        stale_cancels_o
);

  // ------------------------------------------------------------------
  // State.
  // ------------------------------------------------------------------
  logic         r_run_active;
  word_t        r_run_gen;
  logic [2:0]   r_stage;        // ACK-fired stages this epoch, 7 = complete
  logic         r_reset_active;
  word_t        r_reset_ref;     // Group failure ref latched at FREEZE
  logic         r_reset_done;
  logic         r_cmd_busy;
  reset_phase_t r_cmd_phase;
  logic         r_clr_done;
  logic         r_mark_pulse;
  logic         r_mark_capture;
  word_t        r_mark_saved;   // only persistent per-phase ACK payload
  logic         r_ack_valid;
  reset_phase_t r_ack_phase;
  word_t        r_stale_cancels;

  // ------------------------------------------------------------------
  // Group epoch admission and the per-cycle effective epoch view. A
  // handshake on this port is the authentication: no separate credential
  // exists at this boundary.
  // ------------------------------------------------------------------
  logic drain_all;
  assign drain_all = (drain_evidence_i == 6'h3f);

  assign run_ready_o = !group_poisoned_i
                    && !r_reset_active
                    && !r_cmd_busy
                    && !r_ack_valid
                    && drain_all
                    && !(|live_i)
                    && (!r_run_active || (run_generation_i != r_run_gen));

  logic run_accept;
  assign run_accept = run_begin_i && run_ready_o;

  word_t eff_gen;
  assign eff_gen = run_accept ? run_generation_i : r_run_gen;

  logic [2:0] eff_stage;
  assign eff_stage = run_accept ? 3'd0 : r_stage;

  // ------------------------------------------------------------------
  // Request classification. The encoded phase (packed, 0..6, 7 = invalid)
  // compares directly against the stage counter — no index translation.
  // ------------------------------------------------------------------
  logic [2:0] ph_bits;
  assign ph_bits = 3'(req_phase_i);

  logic req_epoch_ok;
  assign req_epoch_ok = (r_run_active || run_accept) && (req_generation_i == eff_gen);

  logic req_port_free;
  assign req_port_free = !r_cmd_busy && !r_ack_valid;

  logic req_accept, req_dupe;
  always_comb begin
    req_ready_o  = 1'b0;
    req_reject_o = 1'b0;
    req_accept   = 1'b0;
    req_dupe     = 1'b0;
    if (req_valid_i) begin
      if ((ph_bits == 3'd7) || !req_epoch_ok) begin
        req_ready_o  = 1'b1;  // bad encoding / wrong epoch
        req_reject_o = 1'b1;
      end else if (ph_bits > eff_stage) begin
        req_ready_o  = 1'b1;  // future or out-of-order stage
        req_reject_o = 1'b1;
      end else if (ph_bits == eff_stage) begin
        if (req_port_free) begin
          req_ready_o = 1'b1;
          req_accept  = 1'b1;
        end else if (r_cmd_busy) begin
          req_ready_o = 1'b1;  // replay of the outstanding stage is
        end                   // absorbed: the pending ACK answers it
        // else: a duplicate-ACK offer occupies the port — hold the request
        // (backpressure) until it frees; never reject a legitimate replay.
      end else if (req_port_free) begin
        req_ready_o = 1'b1;    // duplicate of an acknowledged stage
        req_dupe    = 1'b1;
      end
    end
  end

  // Freeze covers the FREEZE admission edge itself (a same-edge CMD_START
  // is rejected before the latch closes), the active reset until the
  // RESUME ACK fires, Group poison, and the pre-epoch window.
  logic freeze_accept;
  assign freeze_accept  = req_valid_i && req_ready_o && req_accept
                          && (req_phase_i == RP_FREEZE);
  assign issue_frozen_o = !r_run_active || r_reset_active || group_poisoned_i
                          || freeze_accept;

  // ------------------------------------------------------------------
  // CANCEL_UNISSUED sweep: lowest still-open live Task. live/aborted/
  // committed are the terminal store's committed source records.
  // ------------------------------------------------------------------
  logic       sweep_active, sweep_ok;
  int unsigned sweep_idx;

  assign sweep_active = r_cmd_busy && !r_ack_valid && (r_cmd_phase == RP_CANCEL);

  always_comb begin
    sweep_ok  = 1'b0;
    sweep_idx = 0;
    for (int unsigned c = 0; c < CONTEXT_COUNT; c++) begin
      if (!sweep_ok && live_i[c] && !aborted_i[c] && !terminal_committed_i[c]) begin
        sweep_ok  = 1'b1;
        sweep_idx = c;
      end
    end
  end

  // ------------------------------------------------------------------
  // Functional CANCEL_TASK classification against the real live bindings
  // (full TaskKey first, then full owner; a foreign tile never binds).
  // ------------------------------------------------------------------
  logic cancel_exact, cancel_conflict;
  int unsigned cancel_idx;

  always_comb begin
    cancel_exact    = 1'b0;
    cancel_conflict = 1'b0;
    cancel_idx      = 0;
    for (int unsigned c = 0; c < CONTEXT_COUNT; c++) begin
      if (live_i[c] && (cancel_owner_i.ctx.tile_id == TILE_ID)
          && (live_owner_i[c] == cancel_owner_i)) begin
        cancel_exact = 1'b1;
        cancel_idx   = c;
      end
      if (live_i[c] && (live_owner_i[c].task_key == cancel_owner_i.task_key)
          && !(live_owner_i[c] == cancel_owner_i)) begin
        cancel_conflict = 1'b1;  // TaskKey present, context identity differs
      end
    end
  end

  always_comb begin
    if (cancel_conflict) begin
      cancel_class_o = CC_CONFLICT;
    end else if (cancel_exact && !aborted_i[cancel_idx]
                 && !terminal_committed_i[cancel_idx]) begin
      cancel_class_o = CC_APPLY;
    end else if (cancel_exact) begin
      cancel_class_o = CC_DUPLICATE;  // repeat / already latched or terminal
    end else begin
      cancel_class_o = CC_STALE;
    end
  end

  assign cancel_ready_o = 1'b1;
  assign cancel_fault_o = cancel_valid_i && (cancel_class_o == CC_CONFLICT);

  logic cancel_abort;
  assign cancel_abort = cancel_valid_i && (cancel_class_o == CC_APPLY);

  // Single abort writer: a functional cancel never waits behind the sweep.
  assign abort_valid_o  = cancel_abort || (sweep_active && sweep_ok);
  assign abort_owner_o  = cancel_abort ? cancel_owner_i : live_owner_i[sweep_idx];
  assign abort_status_o = TS_RESET;
  assign abort_ref_o    = cancel_abort ? cancel_ref_i : r_reset_ref;

  // ------------------------------------------------------------------
  // Phase-effect handshakes. INVALIDATE/REBUILD offers are precondition-
  // gated, so valid&&ready at an edge is always an authorized fire.
  // ------------------------------------------------------------------
  assign clear_ingress_valid_o = r_cmd_busy && !r_ack_valid
                                 && (r_cmd_phase == RP_CANCEL) && !r_clr_done;
  logic clear_fire;
  assign clear_fire = clear_ingress_valid_o && clear_ingress_ready_i;

  assign invalidate_valid_o = r_cmd_busy && !r_ack_valid
                              && (r_cmd_phase == RP_INVALIDATE)
                              && refs_zero_i && abandon_ready_i;
  logic invalidate_fire;
  assign invalidate_fire  = invalidate_valid_o && invalidate_ready_i;
  assign abandon_events_o = invalidate_fire;

  assign rebuild_valid_o = r_cmd_busy && !r_ack_valid
                           && (r_cmd_phase == RP_REBUILD) && drain_all;
  logic rebuild_fire;
  assign rebuild_fire = rebuild_valid_o && rebuild_ready_i;

  // Command-completion predicates (each implies busy && no ACK offered).
  logic cancel_done, drain_done;
  assign cancel_done = r_cmd_busy && !r_ack_valid && (r_cmd_phase == RP_CANCEL)
                       && r_clr_done && !sweep_ok;
  assign drain_done  = r_cmd_busy && !r_ack_valid && (r_cmd_phase == RP_DRAIN)
                       && drain_all;

  assign mark_events_o           = r_mark_pulse;
  assign is_quiescent_snapshot_o = drain_evidence_i;

  // ACK port: validity+phase are registered once; the payload fields are
  // canonical derivations, stable while offered (no run admission, no new
  // MARK and no command progress can pass an offered ACK).
  assign ack_valid_o      = r_ack_valid;
  assign ack_phase_o      = r_ack_phase;
  assign ack_generation_o = eff_gen;
  assign ack_marked_o     = (r_ack_phase == RP_MARK) ? r_mark_saved : 64'd0;
  assign ack_mask_o       = (r_ack_phase == RP_DRAIN) ? 6'h3f : 6'h0;
  assign reset_active_o   = r_reset_active;
  assign reset_done_o     = r_reset_done;
  // Same-edge correctness: during the FREEZE accept cycle the latch has not
  // closed yet, so the live (being-latched) value is forwarded; afterwards
  // the register holds it and later input changes have no effect.
  assign reset_failure_o = r_reset_active || freeze_accept;
  assign reset_ref_o     = freeze_accept ? reset_fault_ref_i : r_reset_ref;
  assign stale_cancels_o  = r_stale_cancels;

  // ------------------------------------------------------------------
  // One ACK payload source per edge: the request paths need a free port,
  // the completion paths need the command busy with no ACK offered —
  // pairwise exclusive, so a single priority mux feeds one register.
  // ------------------------------------------------------------------
  logic        ack_load;
  reset_phase_t ack_ld_phase;
  always_comb begin
    ack_load     = 1'b0;
    ack_ld_phase = RP_FREEZE;
    if (req_valid_i && req_ready_o) begin
      if (req_dupe) begin
        ack_load     = 1'b1;  // original logical ACK, no re-execution
        ack_ld_phase = req_phase_i;
      end else if (req_accept && (req_phase_i == RP_FREEZE)) begin
        ack_load     = 1'b1;
        ack_ld_phase = RP_FREEZE;
      end else if (req_accept && (req_phase_i == RP_RESUME)) begin
        ack_load     = 1'b1;  // effect lands at the ACK fire
        ack_ld_phase = RP_RESUME;
      end
    end
    if (r_mark_capture) begin
      ack_load     = 1'b1;
      ack_ld_phase = RP_MARK;
    end
    if (cancel_done) begin
      ack_load     = 1'b1;
      ack_ld_phase = RP_CANCEL;
    end
    if (drain_done) begin
      ack_load     = 1'b1;
      ack_ld_phase = RP_DRAIN;
    end
    if (invalidate_fire) begin
      ack_load     = 1'b1;
      ack_ld_phase = RP_INVALIDATE;
    end
    if (rebuild_fire) begin
      ack_load     = 1'b1;
      ack_ld_phase = RP_REBUILD;
    end
  end

  // ------------------------------------------------------------------
  // Sequential core. Writer-conflict audit: epoch admission needs a free
  // port and no ACK; the request-accept branch needs a free port; the
  // completion predicates need the command busy with no ACK offered; the
  // ACK-fire branch needs an ACK offered — every register below has at
  // most one writer per edge.
  // ------------------------------------------------------------------
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      r_run_active   <= 1'b0;
      r_run_gen      <= 64'd0;
      r_stage        <= 3'd0;
      r_reset_active <= 1'b0;
      r_reset_done   <= 1'b0;
      r_reset_ref    <= 64'd0;
      r_cmd_busy     <= 1'b0;
      r_cmd_phase    <= RP_FREEZE;
      r_clr_done     <= 1'b0;
      r_mark_pulse   <= 1'b0;
      r_mark_capture <= 1'b0;
      r_mark_saved   <= 64'd0;
      r_ack_valid    <= 1'b0;
      r_ack_phase    <= RP_FREEZE;
      r_stale_cancels <= 64'd0;
    end else begin
      // Group epoch admission (single writer of the epoch registers).
      if (run_accept) begin
        r_run_active <= 1'b1;
        r_run_gen    <= run_generation_i;
        r_stage      <= 3'd0;
        r_reset_done <= 1'b0;  // clears on the NEW run only
      end

      // RESET_PHASE_REQ acceptance (rejects/duplicates write nothing here:
      // duplicates feed ack_load above; absorbs are pure consumption).
      if (req_valid_i && req_ready_o && req_accept) begin
        r_cmd_busy  <= 1'b1;
        r_cmd_phase <= req_phase_i;
        r_clr_done  <= 1'b0;
        if (req_phase_i == RP_FREEZE) begin
          r_reset_active <= 1'b1;            // freeze latches at admission
          r_reset_ref    <= reset_fault_ref_i;  // retained for the reset
        end else if (req_phase_i == RP_MARK) begin
          r_mark_pulse <= 1'b1;    // one-edge table write next cycle
        end
      end

      // CANCEL_UNISSUED ingress clear.
      if (clear_fire) begin
        r_clr_done <= 1'b1;
      end

      // MARK: pulse edge, then capture the table's committed count.
      if (r_mark_pulse) begin
        r_mark_pulse   <= 1'b0;
        r_mark_capture <= 1'b1;
      end else if (r_mark_capture) begin
        r_mark_capture <= 1'b0;
        r_mark_saved   <= mark_count_i;
      end

      // One ACK register pair: load or fire, never both.
      if (ack_load) begin
        r_ack_valid <= 1'b1;
        r_ack_phase <= ack_ld_phase;
      end else if (r_ack_valid && ack_ready_i) begin
        r_ack_valid <= 1'b0;
        if (r_cmd_busy) begin
          r_cmd_busy <= 1'b0;
          r_stage    <= 3'(r_ack_phase) + 3'd1;  // RESUME (6) -> 7
          if (r_ack_phase == RP_RESUME) begin
            r_reset_active <= 1'b0;
            r_reset_done   <= 1'b1;
          end
        end
      end

      // Stale CANCEL_TASK bookkeeping (saturating, wrap forbidden).
      if (cancel_valid_i && (cancel_class_o == CC_STALE)
          && (r_stale_cancels != {64{1'b1}})) begin
        r_stale_cancels <= r_stale_cancels + 64'd1;
      end
    end
  end

endmodule
