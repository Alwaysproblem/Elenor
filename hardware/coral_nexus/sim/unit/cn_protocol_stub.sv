/* cn_protocol_stub.sv — unit-only multi-channel wire-frame stub Tile (C025 / W1-03).
 *
 * `backend_kind == "stub"`: NEVER admissible as CS1 evidence (W1 plan §7.6).
 *
 * 5 channels (`CTRL=0, EVENT=1, PROGRAM=2, MEMORY=3, DEBUG=4`) at full wire width
 * (`MAX_WORDS = 17` x 64-bit). Deterministic `g2t_ready[c]` lowering from `cfg_ready_seed`
 * (LFSR + bounded service guarantee; also stalls while the channel's T2G response path is
 * occupied and `t2g_ready[c]` is low). Accepted G2T frames are decoded with `cn_offsets.svh`
 * constants; every T2G frame is staged by a clock edge, so it is a candidate for tick T+1
 * only, never accepted in tick T.
 *
 * Legacy wire-transport fixture (`cfg_lifecycle == 0`, default; unchanged C025 behavior):
 *   QUERY_CAPS->CAPS_RESP; TASK_PLAN/COMMIT/FINAL-> canned ACKs (COMMIT establishes
 *   `live_ctx_gen`); RESET_PHASE_REQ->ACK(pending=0, mask=15); EVENT_NOTIFY->POST_EVENT;
 *   PROGRAM_INSTALL->READ_REQ; PROGRAM_READ_RESP->INSTALL_ACK; PROGRAM_INVALIDATE->
 *   INVALIDATE_ACK; MEM_RSP->TERMINAL_ACK. `cfg_emit_mem_req` emits a 136B MEM_REQ on MEMORY
 *   T2G at reset release; `cfg_corrupt_t2g` flips payload word 0 bit 0; `cfg_glitch_payload`
 *   corrupts the post-edge G2T snapshot (ABI payload-stability check).
 *
 * Explicit unit lifecycle fixture (`cfg_lifecycle == 1`, C035 / W1-04; legacy fault flags
 * are rejected in this mode by the C ABI). True finite state over TWO local context holds
 * (matching `CAPS_RESP.context_count = 2`) and a bounded 4-entry ticket table. The stub
 * NEVER models application numerical work and fabricates no data: a task is drained by
 * definition once FINAL is authorized, but its REAL obligations (input phase signal,
 * retirement authorization, terminal delivery) are enforced exactly. Ticket residency is
 * the single source of truth for outstanding obligations: PLANNED -> HELD -> DRAIN ->
 * COMPLETION, and the ticket (with its slot) is freed ONLY when its TASK_COMPLETION frame
 * actually FIRES — no parallel occupancy counters exist to drift. CTRL/EVENT responses run
 * through depth-4 FIFOs (one request may queue several ordered frames; admission stops one
 * slot before full so nothing is dropped). Stage rules are strict on FULL widths: an old or
 * aliased transaction/slot/generation probe (e.g. ctx_slot 65 vs 1) changes no state.
 *   - TASK_PLAN  {task_key, txn, pool_epoch, pool_version} -> ticket PLANNED,
 *     ACK(ACCEPTED) (WAIT(WAIT_CONTROL_RESOURCE) while frozen or table full; FAULT on
 *     root!=launch or duplicate live task_key under a different txn; same-txn replay re-ACKs).
 *   - TASK_COMMIT validates the PLAN transaction, Context generation and the live Tile
 *     pool version. Real reservations and releases change that version; PLAN never does.
 *     On same-generation version drift, ACK(STALE) and a matching RESOURCE_UPDATE report
 *     the current pool/slot without a reservation. On HOLD_READY the report carries the
 *     post-reservation version and the newly allocated Arena. Only this positive outcome
 *     grants ownership. An occupied slot reports WAIT_SLOT.
 *     `cfg_fail_hold`: the FIRST valid reservation is made FOR REAL (slot, generation and a
 *     consumed arena gen) and then rejected with ACK(FAULT, wait_reason=0). A re-sent
 *     duplicate of the exact failed COMMIT re-ACKs FAULT; NO fresh txn may rebind the hold —
 *     the exact TICKET_ROLLBACK -> ACK(COMPENSATED) is the only release proof.
 *   - TASK_COMMIT_FINAL{task_key, txn, slot, gen} -> strict full-width match against the
 *     live hold (txn==hold_txn, slot, generation). ACK(ACCEPTED); ticket drained-by-
 *     definition. `cfg_input_released`: the input phase signal is emitted only AFTER the
 *     FINAL ACCEPT ACK actually FIRES, and RETIRE_READY waits until that PHASE_SIGNAL fired
 *     (no drain over an owed phase, C033). Without the flag RETIRE_READY is queued at FINAL
 *     acceptance. `cfg_fail_final`: the FIRST valid FINAL is authorized irreversibly, then
 *     the local installation fails: ACK(FAULT, for TASK_COMMIT_FINAL) — never a phase
 *     signal, never success; closure is cancel -> drain -> RETIRE -> terminal.
 *   - TICKET_ROLLBACK{task_key, txn, slot, gen} -> strict full-width match (PLANNED: plan
 *     txn; HELD incl. faulted reservations: hold txn + slot + generation). Local release
 *     commits at the clock edge (slot freed, high-water advanced); only then is
 *     ACK(COMPENSATED) staged. Post-FINAL: ACK(FAULT). Unknown/mismatched: ACK(STALE).
 *   - CANCEL_TASK{task_key, slot, gen} -> no ACK exists; honored ONLY on an authorized
 *     (post-FINAL) ticket with full-width slot/generation match: marks the terminal status
 *     RESET (a failed FINAL installation announces its RETIRE_READY here). A pre-FINAL
 *     cancel finds no live Task (C015): no promotion, no terminal — compensation is the
 *     only close.
 *   - RETIRE{task_key, tile, slot, gen, arena_alloc_gen} -> honored only when the ticket
 *     OBSERVED its RETIRE_READY fire (lt_rr_fired; a forged early RETIRE reclaims nothing)
 *     and the FULL immutable identity matches: tile_id == configured link, slot,
 *     generation and the exact arena_alloc_gen reported in the hold's RESOURCE_UPDATE.
 *     Then the ticket enters COMPLETION: the slot is locally released for accounting
 *     (high-water advances) but the ticket stays resident until its TASK_COMPLETION frame
 *     FIRES (status DONE, or RESET after a cancel, or ERROR with first_fault_ref = the
 *     hold's arena_alloc_gen after a failed FINAL installation). A mismatched RETIRE has
 *     no effect and produces no terminal.
 *   - RUN_BEGIN: accepts a strictly increasing run_generation (first from cold accepted);
 *     establishes the epoch; never replies.
 *   - RESET_PHASE_REQ: answered only for the current epoch. FREEZE_ISSUE latches an
 *     admission freeze (PLAN/COMMIT/FINAL -> ACK(WAIT, WAIT_CONTROL_RESOURCE); tickets are
 *     never dropped reset-side); RESUME_DONE clears it. DRAIN waits for all tickets and
 *     all queued/offered EVENT reports. A withdrawn hold can still owe RESOURCE_UPDATE
 *     after its compensation ACK. DRAIN then answers once with evidence mask 0x3f.
 *     Other phases answer immediately
 *     with mask 0 and pending_to_reset=0 (only MARK_EVENTS may carry a count in the real
 *     C034 service; this fixture has no local event entries). Epoch mismatch: no reply.
 *   - QUERY_CAPS / EVENT_NOTIFY / PROGRAM_* / MEM_RSP keep the legacy transport behaviors.
 */
`timescale 1ns/1ps

module cn_protocol_stub (
  input  wire          clk,
  input  wire          rst_n,

  // 5-channel G2T (Group -> Tile) full-width wire frames (17 x 64b = 1088b per channel)
  input  wire [4:0]    g2t_valid,
  output wire [4:0]    g2t_ready,
  input  wire [24:0]   g2t_words_flat,
  input  wire [5439:0] g2t_frame_flat,

  // 5-channel T2G (Tile -> Group) full-width wire frames
  output reg  [4:0]    t2g_valid,
  input  wire [4:0]    t2g_ready,
  output wire [24:0]   t2g_words_flat,
  output wire [5439:0] t2g_frame_flat,

  // configuration / debug / fault injection
  input  wire [31:0]   cfg_ready_seed,
  input  wire [31:0]   cfg_inject_fatal_at_tick,
  input  wire [31:0]   cfg_inject_abort_at_tick,
  input  wire          cfg_corrupt_t2g,
  input  wire          cfg_emit_mem_req,
  input  wire          cfg_glitch_payload,
  input  wire          cfg_lifecycle,
  input  wire          cfg_fail_hold,
  input  wire          cfg_fail_final,
  input  wire          cfg_input_released,
  input  wire [31:0]   cfg_tile_id,
  output reg  [63:0]   posedge_count,
  output reg  [31:0]   fatal_code,
  output reg  [63:0]   dbg_last_g2t_word0
);

  `include "cn_offsets.svh"

  localparam int MAX_WORDS = 17;

  reg [31:0] lfsr;
  reg [63:0] t2g_seq [0:4];
  reg [63:0] live_ctx_gen [0:CN_CONTEXT_SLOTS-1];
  reg        mem_req_sent;
  wire [1087:0] g2t_frame [0:4];
  reg  [4:0]    t2g_words [0:4];
  reg  [1087:0] t2g_frame [0:4];

  // ---- lifecycle fixture state (C035 / W1-04; active only under cfg_lifecycle) ----
  localparam int LF_FIFO_DEPTH = 4; // CTRL/EVENT T2G response FIFOs
  localparam int LT_ENTRIES    = 4; // bounded admission ticket table
  localparam int LT_SLOTS      = 2; // local ctx holds; matches CAPS_RESP.context_count

  localparam [2:0] LT_FREE       = 3'd0;
  localparam [2:0] LT_PLANNED    = 3'd1;
  localparam [2:0] LT_HELD       = 3'd2; // reservation established (lt_faulted: rejected one)
  localparam [2:0] LT_DRAIN      = 3'd3; // FINAL authorized; awaiting retirement
  localparam [2:0] LT_COMPLETION = 3'd4; // terminal queued; resident until it FIRES
  localparam [2:0] LT_VERSION_STALE = 3'd5; // obsolete PLAN metadata only; no hold

  // Sealed C017 enum values (cn_messages.h; the generated .svh carries layout constants only).
  localparam [63:0] CN_OUT_HOLD_READY  = 64'd0;
  localparam [63:0] CN_OUT_ACCEPTED    = 64'd1;
  localparam [63:0] CN_OUT_WAIT        = 64'd2;
  localparam [63:0] CN_OUT_FAULT       = 64'd3;
  localparam [63:0] CN_OUT_STALE       = 64'd4;
  localparam [63:0] CN_OUT_COMPENSATED = 64'd5;
  localparam [63:0] CN_WHY_SLOT        = 64'd2; // WAIT_SLOT
  localparam [63:0] CN_WHY_CONTROL     = 64'd3; // WAIT_CONTROL_RESOURCE
  localparam [63:0] CN_TASK_DONE       = 64'd0;
  localparam [63:0] CN_TASK_ERROR      = 64'd1;
  localparam [63:0] CN_TASK_RESET      = 64'd3;
  localparam [31:0] CN_PHASE_FREEZE    = 32'd0; // FREEZE_ISSUE
  localparam [31:0] CN_PHASE_DRAIN     = 32'd2; // DRAIN
  localparam [31:0] CN_PHASE_RESUME    = 32'd6; // RESUME_DONE
  localparam [63:0] CN_DRAIN_EVIDENCE_FULL = 64'h3f;

  reg [1087:0] lf_frame [0:1][0:LF_FIFO_DEPTH-1];
  reg [4:0]    lf_words [0:1][0:LF_FIFO_DEPTH-1];
  reg [2:0]    lf_head  [0:4]; // [0..1] used; [0..4] indexable from the ready generate
  reg [2:0]    lf_count [0:4];

  reg [2:0]  lt_state      [0:LT_ENTRIES-1];
  reg [63:0] lt_launch     [0:LT_ENTRIES-1];
  reg [63:0] lt_dispatch   [0:LT_ENTRIES-1];
  reg [63:0] lt_task       [0:LT_ENTRIES-1];
  reg [63:0] lt_handle     [0:LT_ENTRIES-1];
  reg [63:0] lt_plan_txn   [0:LT_ENTRIES-1];
  reg [63:0] lt_hold_txn   [0:LT_ENTRIES-1];
  reg [63:0] lt_final_txn  [0:LT_ENTRIES-1];
  reg [63:0] lt_pool_epoch [0:LT_ENTRIES-1];
  reg [63:0] lt_pool_ver   [0:LT_ENTRIES-1];
  reg [63:0] lt_gen        [0:LT_ENTRIES-1];
  reg [63:0] lt_arena      [0:LT_ENTRIES-1];
  reg [63:0] lt_drain_seq  [0:LT_ENTRIES-1];
  reg [5:0]  lt_slot       [0:LT_ENTRIES-1];
  reg        lt_cancel     [0:LT_ENTRIES-1];
  reg        lt_final_faulted [0:LT_ENTRIES-1]; // FAIL_FINAL: authorized, install failed
  reg        lt_faulted    [0:LT_ENTRIES-1]; // reserved then rejected (cfg_fail_hold)
  reg        lt_phase_owed [0:LT_ENTRIES-1]; // fixture seeds the input-release phase
  reg        lt_phase_armed [0:LT_ENTRIES-1]; // PHASE_SIGNAL queued after FINAL ACK fired
  reg        lt_phase_fired [0:LT_ENTRIES-1]; // PHASE_SIGNAL handshake completed
  reg        lt_rr_pending  [0:LT_ENTRIES-1]; // RETIRE_READY owed, gated on the fired phase
  reg        lt_rr_fired    [0:LT_ENTRIES-1]; // RETIRE_READY handshake completed (RETIRE gate)

  reg [63:0] lf_arena_ctr;      // fresh per reservation; never reused across rollback/release
  reg [63:0] lf_pool_version;   // changes at physical reservation and release, not per request
  reg [63:0] lf_slot_arena_gen [0:LT_SLOTS-1]; // last actual allocation on the physical slot
  reg [63:0] lf_drain_ctr;      // drain evidence sequence source
  reg        lf_fail_armed;     // cfg_fail_hold: fault exactly one reservation
  reg        lf_fail_done;
  reg        lf_fail_final_armed; // cfg_fail_final: fault exactly one FINAL installation
  reg        lf_fail_final_done;
  reg [63:0] lf_epoch;          // current run epoch (RUN_BEGIN)
  reg        lf_epoch_seen;
  reg        lf_frozen;         // FREEZE_ISSUE .. RESUME_DONE
  reg        lf_parked_drain;   // DRAIN request awaiting closed obligations
  reg [63:0] lf_parked_epoch;

  // per-edge blocking temporaries (re-assigned at the top of every posedge section)
  integer  lf_push_n [0:1];
  integer  lf_seq_n  [0:1];
  reg      lf_pop_now [0:1];
  integer  lf_t;
  integer  lf_free;
  reg [63:0] p8;
  reg [63:0] lf_fired_word0;

  genvar ch;
  generate
    for (ch = 0; ch < 5; ch = ch + 1) begin : gen_ready
      assign g2t_frame[ch] = g2t_frame_flat[ch * 1088 +: 1088];
      assign t2g_words_flat[ch * 5 +: 5] = t2g_words[ch];
      assign t2g_frame_flat[ch * 1088 +: 1088] = t2g_frame[ch];
      wire lfsr_open = lfsr[ch] | (posedge_count[2:0] == 3'b111);
      wire local_busy = (ch == 3) & cfg_emit_mem_req & (!mem_req_sent);
      wire out_space = ((!t2g_valid[ch]) | t2g_ready[ch]) & (!local_busy);
      // Lifecycle CTRL/EVENT admission needs FIFO headroom: one accepted frame may push
      // two ordered responses, so acceptance stops one slot before a full FIFO.
      wire lf_space = (!cfg_lifecycle) || (ch >= 2) || (lf_count[ch] <= (LF_FIFO_DEPTH - 2));
      // A single HOLD can emit an EVENT report alongside a phase on a different
      // ticket. Lack of EVENT headroom must not stop an unrelated rollback.
      wire hold_room = !cfg_lifecycle || ch != 0 || g2t_frame[ch][31:0] != CN_MSG_TASK_COMMIT ||
                       lf_count[1] <= (LF_FIFO_DEPTH - 2);
      assign g2t_ready[ch] = rst_n & lfsr_open & out_space & lf_space & hold_room;
    end
  endgenerate

  function automatic [63:0] get_word(input [1087:0] frame, input int idx);
    get_word = frame[idx * 64 +: 64];
  endfunction

  function automatic [1087:0] set_word(
    input [1087:0] frame,
    input int idx,
    input [63:0] val
  );
    reg [1087:0] tmp;
    begin
      tmp = frame;
      tmp[idx * 64 +: 64] = val;
      set_word = tmp;
    end
  endfunction

  function automatic integer lf_find_ticket(
    input [63:0] lg,
    input [63:0] dp,
    input [63:0] tk,
    input [63:0] hd
  );
    integer i;
    begin
      lf_find_ticket = -1;
      for (i = 0; i < LT_ENTRIES; i = i + 1) begin
        if (lt_state[i] != LT_FREE && lt_launch[i] == lg && lt_dispatch[i] == dp &&
            lt_task[i] == tk && lt_handle[i] == hd) begin
          lf_find_ticket = i;
        end
      end
    end
  endfunction

  function automatic integer lf_free_ticket();
    integer i;
    begin
      lf_free_ticket = -1;
      for (i = 0; i < LT_ENTRIES; i = i + 1) begin
        if (lt_state[i] == LT_FREE) begin
          lf_free_ticket = i;
        end
      end
    end
  endfunction

  // A slot is occupied from HOLD establishment until the terminal FIRES (COMPLETION
  // residency), so early reuse is impossible without any parallel occupancy counter.
  function automatic integer lf_slot_held(input [63:0] s);
    integer i;
    begin
      lf_slot_held = 0;
      for (i = 0; i < LT_ENTRIES; i = i + 1) begin
        if ((lt_state[i] == LT_HELD || lt_state[i] == LT_DRAIN || lt_state[i] == LT_COMPLETION) &&
            lt_slot[i] == s[5:0]) begin
          lf_slot_held = 1;
        end
      end
    end
  endfunction

  // A rollback releases the local hold, not its independently stalled EVENT beat.
  function automatic logic lf_quiet();
    logic quiet;
    integer i;
    // Inspect both storage locations: a popped FIFO entry may be parked on the link.
    begin
      quiet = lf_count[1] == 0 && !t2g_valid[1];
      for (i = 0; i < LT_ENTRIES; i = i + 1) begin
        if (lt_state[i] != LT_FREE) begin
          quiet = 1'b0;
        end
      end
      lf_quiet = quiet;
    end
  endfunction

  // Queue one lifecycle T2G frame on channel `ch` (0=CTRL, 1=EVENT). Word 2 (message_seq)
  // is stamped here so several frames queued at the same edge stay gapless per channel.
  // Occupancy accounting: every call writes the RUNNING per-edge total
  // `old_count - pop_this_edge + pushes_so_far`; the final nonblocking assignment of the
  // edge therefore carries one consistent total (pop_now is computed before any emit).
  task automatic lf_emit(
    input integer ch,
    input [31:0] mtype,
    input integer pwords,
    input [1087:0] frm
  );
    integer idx;
    integer eff_head;
    integer eff_count;
    reg [1087:0] f2;
    begin
      f2 = set_word(frm, 2, 64'(t2g_seq[ch] + lf_seq_n[ch]));
      eff_head  = lf_head[ch] + (lf_pop_now[ch] ? 1 : 0);
      eff_count = lf_count[ch] - (lf_pop_now[ch] ? 1 : 0) + lf_push_n[ch];
      idx = (eff_head + eff_count) % LF_FIFO_DEPTH;
      lf_frame[ch][idx] <= f2;
      lf_words[ch][idx] <= 5'(CN_HEADER_WORDS + pwords);
      lf_push_n[ch] = lf_push_n[ch] + 1;
      lf_seq_n[ch]  = lf_seq_n[ch] + 1;
      t2g_seq[ch]   <= t2g_seq[ch] + 64'(lf_seq_n[ch]);
      lf_count[ch]  <= lf_count[ch] - (lf_pop_now[ch] ? 1 : 0) + lf_push_n[ch];
    end
  endtask

  task automatic lf_ack(
    input [63:0] txn,
    input [63:0] ack_for,
    input [63:0] outcome,
    input [63:0] reason
  );
    reg [1087:0] f;
    begin
      f = '0;
      f = set_word(f, 0, CN_MSG_ACK);
      f = set_word(f, 1, CN_WORDS_ACK);
      f = set_word(f, 4 + CN_OFF_ACK_TXN_ID, txn);
      f = set_word(f, 4 + CN_OFF_ACK_ACK_FOR_MESSAGE_TYPE, ack_for);
      f = set_word(f, 4 + CN_OFF_ACK_OP_SEQ, 64'd0);
      f = set_word(f, 4 + CN_OFF_ACK_OUTCOME, outcome);
      f = set_word(f, 4 + CN_OFF_ACK_WAIT_REASON, reason);
      lf_emit(0, CN_MSG_ACK, CN_WORDS_ACK, f);
    end
  endtask

  // The outcome on CTRL determines whether this EVENT reports a new hold or a
  // stale plan. Sample actual owner values at the edge, not old NBA registers.
  task automatic lf_resource_update(
    input [63:0] slot,
    input [63:0] gen,
    input [63:0] version,
    input [63:0] arena
  );
    reg [1087:0] f;
    begin
      f = '0;
      f = set_word(f, 0, CN_MSG_RESOURCE_UPDATE);
      f = set_word(f, 1, CN_WORDS_RESOURCE_UPDATE);
      f = set_word(f, 4 + CN_OFF_RESOURCE_UPDATE_TILE_ID, 64'(cfg_tile_id));
      f = set_word(f, 4 + CN_OFF_RESOURCE_UPDATE_CTX_SLOT, slot);
      f = set_word(f, 4 + CN_OFF_RESOURCE_UPDATE_CTX_GENERATION, gen);
      f = set_word(f, 4 + CN_OFF_RESOURCE_UPDATE_VERSION, version);
      f = set_word(f, 4 + CN_OFF_RESOURCE_UPDATE_ARENA_ALLOC_GEN, arena);
      lf_emit(1, CN_MSG_RESOURCE_UPDATE, CN_WORDS_RESOURCE_UPDATE, f);
    end
  endtask

  task automatic lf_phase_signal(input integer t);
    reg [1087:0] f;
    begin
      f = '0;
      f = set_word(f, 0, CN_MSG_PHASE_SIGNAL);
      f = set_word(f, 1, CN_WORDS_PHASE_SIGNAL);
      f = set_word(f, 4 + CN_OFF_PHASE_SIGNAL_LAUNCH_GENERATION, lt_launch[t]);
      f = set_word(f, 4 + CN_OFF_PHASE_SIGNAL_ROOT_LAUNCH_GENERATION, lt_launch[t]);
      f = set_word(f, 4 + CN_OFF_PHASE_SIGNAL_DISPATCH_ORDINAL, lt_dispatch[t]);
      f = set_word(f, 4 + CN_OFF_PHASE_SIGNAL_TASK_ID, lt_task[t]);
      f = set_word(f, 4 + CN_OFF_PHASE_SIGNAL_CONTEXT_HANDLE, lt_handle[t]);
      f = set_word(f, 4 + CN_OFF_PHASE_SIGNAL_DIRECTION, 64'd0); // input_released
      lf_emit(1, CN_MSG_PHASE_SIGNAL, CN_WORDS_PHASE_SIGNAL, f);
    end
  endtask

  // `drain_seq` is passed by value: at the FINAL edge the ticket register is still old.
  task automatic lf_retire_ready(input integer t, input [63:0] drain_seq);
    reg [1087:0] f;
    begin
      f = '0;
      f = set_word(f, 0, CN_MSG_RETIRE_READY);
      f = set_word(f, 1, CN_WORDS_RETIRE_READY);
      f = set_word(f, 4 + CN_OFF_RETIRE_READY_LAUNCH_GENERATION, lt_launch[t]);
      f = set_word(f, 4 + CN_OFF_RETIRE_READY_ROOT_LAUNCH_GENERATION, lt_launch[t]);
      f = set_word(f, 4 + CN_OFF_RETIRE_READY_DISPATCH_ORDINAL, lt_dispatch[t]);
      f = set_word(f, 4 + CN_OFF_RETIRE_READY_TASK_ID, lt_task[t]);
      f = set_word(f, 4 + CN_OFF_RETIRE_READY_CONTEXT_HANDLE, lt_handle[t]);
      f = set_word(f, 4 + CN_OFF_RETIRE_READY_CTX_SLOT, 64'(lt_slot[t]));
      f = set_word(f, 4 + CN_OFF_RETIRE_READY_CTX_GENERATION, lt_gen[t]);
      f = set_word(f, 4 + CN_OFF_RETIRE_READY_DRAIN_EVIDENCE_SEQ, drain_seq);
      lf_emit(0, CN_MSG_RETIRE_READY, CN_WORDS_RETIRE_READY, f);
    end
  endtask

  task automatic lf_completion(input integer t, input [63:0] status, input [63:0] fault_ref);
    reg [1087:0] f;
    begin
      f = '0;
      f = set_word(f, 0, CN_MSG_TASK_COMPLETION);
      f = set_word(f, 1, CN_WORDS_TASK_COMPLETION);
      f = set_word(f, 4 + CN_OFF_TASK_COMPLETION_LAUNCH_GENERATION, lt_launch[t]);
      f = set_word(f, 4 + CN_OFF_TASK_COMPLETION_ROOT_LAUNCH_GENERATION, lt_launch[t]);
      f = set_word(f, 4 + CN_OFF_TASK_COMPLETION_DISPATCH_ORDINAL, lt_dispatch[t]);
      f = set_word(f, 4 + CN_OFF_TASK_COMPLETION_TASK_ID, lt_task[t]);
      f = set_word(f, 4 + CN_OFF_TASK_COMPLETION_CONTEXT_HANDLE, lt_handle[t]);
      f = set_word(f, 4 + CN_OFF_TASK_COMPLETION_STATUS, status);
      f = set_word(f, 4 + CN_OFF_TASK_COMPLETION_FIRST_FAULT_REF, fault_ref);
      lf_emit(0, CN_MSG_TASK_COMPLETION, CN_WORDS_TASK_COMPLETION, f);
    end
  endtask

  task automatic lf_reset_phase_ack(
    input [63:0] phase,
    input [63:0] run_generation,
    input [63:0] pending,
    input [63:0] mask
  );
    reg [1087:0] f;
    begin
      f = '0;
      f = set_word(f, 0, CN_MSG_RESET_PHASE_ACK);
      f = set_word(f, 1, CN_WORDS_RESET_PHASE_ACK);
      f = set_word(f, 4 + CN_OFF_RESET_PHASE_ACK_PHASE, phase);
      f = set_word(f, 4 + CN_OFF_RESET_PHASE_ACK_RUN_GENERATION, run_generation);
      f = set_word(f, 4 + CN_OFF_RESET_PHASE_ACK_PENDING_TO_RESET, pending);
      f = set_word(f, 4 + CN_OFF_RESET_PHASE_ACK_EVIDENCE_MASK, mask);
      lf_emit(0, CN_MSG_RESET_PHASE_ACK, CN_WORDS_RESET_PHASE_ACK, f);
    end
  endtask

  integer c;
  integer s;
  reg [1087:0] next_frame;
  reg [4:0]    next_words;
  reg          produce;
  reg [63:0]   in_msg_type;
  reg [63:0]   in_pwords;
  reg [63:0]   p0, p1, p2, p3, p4, p5, p6, p7;
  reg [63:0]   corrupt_mask;
  reg [63:0]   ack_outcome;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      posedge_count      <= 64'd0;
      lfsr               <= (cfg_ready_seed == 32'd0) ? 32'h1ACE_B00C : cfg_ready_seed;
      fatal_code         <= 32'd0;
      dbg_last_g2t_word0 <= 64'd0;
      t2g_valid          <= 5'b0;
      mem_req_sent       <= 1'b0;
      for (c = 0; c < 5; c = c + 1) begin
        t2g_words[c] <= 5'd0;
        t2g_frame[c] <= '0;
        t2g_seq[c]   <= 64'd0;
        lf_head[c]   <= 3'd0;
        lf_count[c]  <= 3'd0;
      end
      for (s = 0; s < CN_CONTEXT_SLOTS; s = s + 1) begin
        live_ctx_gen[s] <= 64'd0;
      end
      for (s = 0; s < LT_SLOTS; s = s + 1) lf_slot_arena_gen[s] <= 64'd0;
      for (s = 0; s < LT_ENTRIES; s = s + 1) begin
        lt_state[s]         <= LT_FREE;
        lt_launch[s]        <= 64'd0;
        lt_dispatch[s]      <= 64'd0;
        lt_task[s]          <= 64'd0;
        lt_handle[s]        <= 64'd0;
        lt_plan_txn[s]      <= 64'd0;
        lt_hold_txn[s]      <= 64'd0;
        lt_final_txn[s]     <= 64'd0;
        lt_pool_epoch[s]    <= 64'd0;
        lt_pool_ver[s]      <= 64'd0;
        lt_gen[s]           <= 64'd0;
        lt_arena[s]         <= 64'd0;
        lt_drain_seq[s]     <= 64'd0;
        lt_slot[s]          <= 6'd0;
        lt_cancel[s]        <= 1'b0;
        lt_final_faulted[s] <= 1'b0;
        lt_faulted[s]       <= 1'b0;
        lt_phase_owed[s]    <= 1'b0;
        lt_phase_armed[s]   <= 1'b0;
        lt_phase_fired[s]   <= 1'b0;
        lt_rr_pending[s]    <= 1'b0;
        lt_rr_fired[s]      <= 1'b0;
      end
      lf_arena_ctr        <= 64'd0;
      lf_pool_version     <= 64'd0;
      lf_drain_ctr        <= 64'd0;
      lf_fail_armed       <= cfg_fail_hold;
      lf_fail_done        <= 1'b0;
      lf_fail_final_armed <= cfg_fail_final;
      lf_fail_final_done  <= 1'b0;
      lf_epoch            <= 64'd0;
      lf_epoch_seen       <= 1'b0;
      lf_frozen           <= 1'b0;
      lf_parked_drain     <= 1'b0;
      lf_parked_epoch     <= 64'd0;
    end else begin
      posedge_count <= posedge_count + 64'd1;
      lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
      corrupt_mask = cfg_corrupt_t2g ? 64'h1 : 64'h0;

      if (cfg_inject_abort_at_tick == posedge_count[31:0]) begin
        $fatal(1, "cn_protocol_stub: injected process-level crash");
      end
      if (cfg_inject_fatal_at_tick == posedge_count[31:0]) begin
        fatal_code <= 32'd1;
      end

      // Lifecycle per-edge temporaries FIRST: the fire hooks below may queue responses,
      // and every `lf_emit` consumes these (stale values would misplace FIFO slots and
      // message sequence numbers). pop_now is decided before any emission this edge.
      for (c = 0; c < 2; c = c + 1) begin
        lf_push_n[c]  = 0;
        lf_seq_n[c]   = 0;
        lf_pop_now[c] = cfg_lifecycle && (lf_count[c] != 0) &&
                        ((!t2g_valid[c]) || t2g_ready[c]);
      end

      // Clear T2G beats that completed the pre-edge handshake `(t2g_valid && t2g_ready)`
      // and advance the real ticket state machine on the FIRED frame (never a parallel
      // counter — residency IS the bookkeeping):
      //   - FIRED FINAL ACCEPT ACK  -> arm the owed input phase signal (never earlier);
      //   - FIRED PHASE_SIGNAL      -> release the gated RETIRE_READY (no drain over an
      //                                owed phase);
      //   - FIRED RETIRE_READY      -> open the RETIRE authorization window;
      //   - FIRED TASK_COMPLETION   -> free the ticket (its slot becomes reusable).
      for (c = 0; c < 5; c = c + 1) begin
        if (t2g_valid[c] && t2g_ready[c]) begin
          t2g_valid[c] <= 1'b0;
          if (cfg_lifecycle) begin
            lf_fired_word0 = get_word(t2g_frame[c], 0);
            if (c == 0 && lf_fired_word0[31:0] == CN_MSG_ACK &&
                get_word(t2g_frame[c], 4 + CN_OFF_ACK_ACK_FOR_MESSAGE_TYPE) == CN_MSG_TASK_COMMIT_FINAL &&
                get_word(t2g_frame[c], 4 + CN_OFF_ACK_OUTCOME) == CN_OUT_ACCEPTED) begin
              for (lf_t = 0; lf_t < LT_ENTRIES; lf_t = lf_t + 1) begin
                if (lt_state[lf_t] == LT_DRAIN &&
                    lt_final_txn[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_ACK_TXN_ID) &&
                    lt_phase_owed[lf_t] && !lt_phase_armed[lf_t]) begin
                  lt_phase_armed[lf_t] <= 1'b1;
                  lf_phase_signal(lf_t);
                end
              end
            end
            if (c == 1 && lf_fired_word0[31:0] == CN_MSG_PHASE_SIGNAL) begin
              for (lf_t = 0; lf_t < LT_ENTRIES; lf_t = lf_t + 1) begin
                if (lt_state[lf_t] != LT_FREE &&
                    lt_launch[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_PHASE_SIGNAL_LAUNCH_GENERATION) &&
                    lt_dispatch[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_PHASE_SIGNAL_DISPATCH_ORDINAL) &&
                    lt_task[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_PHASE_SIGNAL_TASK_ID) &&
                    lt_handle[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_PHASE_SIGNAL_CONTEXT_HANDLE) &&
                    lt_phase_armed[lf_t] && !lt_phase_fired[lf_t]) begin
                  lt_phase_fired[lf_t] <= 1'b1;
                  if (lt_state[lf_t] == LT_DRAIN && lt_rr_pending[lf_t]) begin
                    lt_rr_pending[lf_t] <= 1'b0;
                    lf_retire_ready(lf_t, lt_drain_seq[lf_t]);
                  end
                end
              end
            end
            if (c == 0 && lf_fired_word0[31:0] == CN_MSG_RETIRE_READY) begin
              for (lf_t = 0; lf_t < LT_ENTRIES; lf_t = lf_t + 1) begin
                if ((lt_state[lf_t] == LT_DRAIN || lt_state[lf_t] == LT_COMPLETION) &&
                    lt_launch[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_RETIRE_READY_LAUNCH_GENERATION) &&
                    lt_dispatch[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_RETIRE_READY_DISPATCH_ORDINAL) &&
                    lt_task[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_RETIRE_READY_TASK_ID) &&
                    lt_handle[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_RETIRE_READY_CONTEXT_HANDLE)) begin
                  lt_rr_fired[lf_t] <= 1'b1;
                end
              end
            end
            if (c == 0 && lf_fired_word0[31:0] == CN_MSG_TASK_COMPLETION) begin
              for (lf_t = 0; lf_t < LT_ENTRIES; lf_t = lf_t + 1) begin
                if (lt_state[lf_t] == LT_COMPLETION &&
                    lt_launch[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_TASK_COMPLETION_LAUNCH_GENERATION) &&
                    lt_dispatch[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_TASK_COMPLETION_DISPATCH_ORDINAL) &&
                    lt_task[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_TASK_COMPLETION_TASK_ID) &&
                    lt_handle[lf_t] == get_word(t2g_frame[c], 4 + CN_OFF_TASK_COMPLETION_CONTEXT_HANDLE)) begin
                  lt_state[lf_t] <= LT_FREE; // terminal observed: slot identity reusable
                end
              end
            end
          end
        end
      end

      // Lifecycle staged-response pops: fill a freed CTRL/EVENT output register from the
      // FIFO. A popped frame is a next-tick candidate exactly like a legacy staged
      // response. Occupancy is committed on the SAME edge even when nothing else is
      // pushed (pop-only edges must not leave stale entries reachable); the write is
      // skipped when an emit already ran, because every `lf_emit` total includes this
      // pop — exactly one consistent update survives the edge either way.
      for (c = 0; c < 2; c = c + 1) begin
        if (lf_pop_now[c]) begin
          t2g_valid[c] <= 1'b1;
          t2g_words[c] <= lf_words[c][lf_head[c][1:0]];
          t2g_frame[c] <= lf_frame[c][lf_head[c][1:0]];
          lf_head[c]   <= (lf_head[c] == LF_FIFO_DEPTH - 1) ? 3'd0 : lf_head[c] + 3'd1;
          if (lf_push_n[c] == 0) begin
            lf_count[c] <= lf_count[c] - 3'd1;
          end
        end
      end

      // Optional unsolicited 136-byte MEM_REQ (4 header + 13 payload = 17 words) on MEMORY (ch 3).
      if (cfg_emit_mem_req && !mem_req_sent && !t2g_valid[3]) begin
        next_frame = '0;
        next_frame = set_word(next_frame, 0, CN_MSG_MEM_REQ);
        next_frame = set_word(next_frame, 1, CN_WORDS_MEM_REQ);
        next_frame = set_word(next_frame, 2, t2g_seq[3]);
        next_frame = set_word(next_frame, 3, 64'd0);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_TILE_ID, (64'd0 ^ corrupt_mask));
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_CTX_SLOT, 64'd1);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_CTX_GENERATION, 64'd2);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_TXN_ID, 64'd77);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_SLICE_ID, 64'd0);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_SLICE_COUNT, 64'd1);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_ARENA_ALLOC_GEN, 64'd3);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_BINDING_ID, 64'd4);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_BINDING_GENERATION, 64'd5);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_BINDING_OFFSET, 64'd64);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_SCOPE_EPOCH, 64'd1);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_MEM_OP, 64'd0);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_LENGTH, 64'd128);
        t2g_valid[3] <= 1'b1;
        t2g_words[3] <= 5'd17;
        t2g_frame[3] <= next_frame;
        t2g_seq[3]   <= t2g_seq[3] + 64'd1;
        mem_req_sent <= 1'b1;
      end

      // Process accepted G2T frames per channel.
      for (c = 0; c < 5; c = c + 1) begin
        if (g2t_valid[c] && g2t_ready[c]) begin
          in_msg_type = get_word(g2t_frame[c], 0);
          in_pwords   = get_word(g2t_frame[c], 1);
          p0 = get_word(g2t_frame[c], 4);
          p1 = get_word(g2t_frame[c], 5);
          p2 = get_word(g2t_frame[c], 6);
          p3 = get_word(g2t_frame[c], 7);
          p4 = get_word(g2t_frame[c], 8);
          p5 = get_word(g2t_frame[c], 9);
          p6 = get_word(g2t_frame[c], 10);
          p7 = get_word(g2t_frame[c], 11);
          p8 = get_word(g2t_frame[c], 12);

          if (c == 0) begin
            dbg_last_g2t_word0 <= cfg_glitch_payload ? (in_msg_type ^ 64'hDEAD) : in_msg_type;
          end

          produce    = 1'b0;
          next_frame = '0;
          next_words = 5'd0;

          if (cfg_lifecycle && c == 0 &&
              (in_msg_type[31:0] == CN_MSG_TASK_PLAN ||
               in_msg_type[31:0] == CN_MSG_TASK_COMMIT ||
               in_msg_type[31:0] == CN_MSG_TASK_COMMIT_FINAL ||
               in_msg_type[31:0] == CN_MSG_TICKET_ROLLBACK ||
               in_msg_type[31:0] == CN_MSG_RETIRE ||
               in_msg_type[31:0] == CN_MSG_CANCEL_TASK ||
               in_msg_type[31:0] == CN_MSG_RUN_BEGIN ||
               in_msg_type[31:0] == CN_MSG_RESET_PHASE_REQ)) begin
            // ---- explicit lifecycle endpoints (C035): strict stages, no echo frames ----
            case (in_msg_type[31:0])
              CN_MSG_TASK_PLAN: begin
                lf_t = lf_find_ticket(p0, p2, p3, p4);
                if (p1 != p0) begin
                  lf_ack(p5, CN_MSG_TASK_PLAN, CN_OUT_FAULT, 64'd0);
                end else if (lf_frozen) begin
                  lf_ack(p5, CN_MSG_TASK_PLAN, CN_OUT_WAIT, CN_WHY_CONTROL);
                end else if (lf_t >= 0) begin
                  if (lt_plan_txn[lf_t] == p5) begin
                    lf_ack(p5, CN_MSG_TASK_PLAN, CN_OUT_ACCEPTED, 64'd0); // replay, no effect
                  end else begin
                    lf_ack(p5, CN_MSG_TASK_PLAN, CN_OUT_FAULT, 64'd0); // duplicate stage
                  end
                end else begin
                  lf_free = lf_free_ticket();
                  if (lf_free < 0) begin
                    lf_ack(p5, CN_MSG_TASK_PLAN, CN_OUT_WAIT, CN_WHY_CONTROL);
                  end else begin
                    lt_state[lf_free]       <= LT_PLANNED;
                    lt_launch[lf_free]      <= p0;
                    lt_dispatch[lf_free]    <= p2;
                    lt_task[lf_free]        <= p3;
                    lt_handle[lf_free]      <= p4;
                    lt_plan_txn[lf_free]    <= p5;
                    lt_pool_epoch[lf_free]  <= p6;
                    lt_pool_ver[lf_free]    <= p7;
                    lt_cancel[lf_free]      <= 1'b0;
                    lt_faulted[lf_free]     <= 1'b0;
                    lt_final_faulted[lf_free] <= 1'b0;
                    lt_phase_owed[lf_free]  <= cfg_input_released;
                    lt_phase_armed[lf_free] <= 1'b0;
                    lt_phase_fired[lf_free] <= 1'b0;
                    lt_rr_pending[lf_free]  <= 1'b0;
                    lt_rr_fired[lf_free]    <= 1'b0;
                    lf_ack(p5, CN_MSG_TASK_PLAN, CN_OUT_ACCEPTED, 64'd0);
                  end
                end
              end
              CN_MSG_TASK_COMMIT: begin
                lf_t = lf_find_ticket(p0, p2, p3, p4);
                if (p1 != p0) begin
                  lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_FAULT, 64'd0);
                end else if (lf_frozen) begin
                  lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_WAIT, CN_WHY_CONTROL);
                end else if (lf_t < 0) begin
                  lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_FAULT, 64'd0); // commit without plan
                end else if (lt_state[lf_t] == LT_PLANNED) begin
                  if (p5 != lt_plan_txn[lf_t] || p6 >= LT_SLOTS || p8 != lt_pool_ver[lf_t]) begin
                    lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_FAULT, 64'd0);
                  end else if (p7 < live_ctx_gen[p6[5:0]]) begin
                    lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_STALE, 64'd0);
                  end else if (lf_slot_held(p6) != 0) begin
                    lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_WAIT, CN_WHY_SLOT);
                  end else if (p8 != lf_pool_version) begin
                    lt_state[lf_t] <= LT_VERSION_STALE;
                    lt_slot[lf_t] <= p6[5:0];
                    lt_gen[lf_t] <= p7;
                    lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_STALE, 64'd0);
                    lf_resource_update(p6, p7, lf_pool_version, lf_slot_arena_gen[int'(p6)]);
                  end else begin
                    lt_state[lf_t]    <= LT_HELD;
                    lt_hold_txn[lf_t] <= p5;
                    lt_slot[lf_t]     <= p6[5:0];
                    lt_gen[lf_t]      <= p7;
                    lt_arena[lf_t]    <= lf_arena_ctr + 64'd1;
                    lf_slot_arena_gen[int'(p6)] <= lf_arena_ctr + 64'd1;
                    lf_arena_ctr      <= lf_arena_ctr + 64'd1;
                    lf_pool_version   <= lf_pool_version + 64'd1;
                    live_ctx_gen[p6[5:0]] <= p7;
                    if (lf_fail_armed && !lf_fail_done) begin
                      lf_fail_done <= 1'b1;
                      lt_faulted[lf_t] <= 1'b1;
                      lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_FAULT, 64'd0);
                    end else begin
                      lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_HOLD_READY, 64'd0);
                      lf_resource_update(p6, p7, lf_pool_version + 64'd1, lf_arena_ctr + 64'd1);
                    end
                  end
                end else if (lt_state[lf_t] == LT_VERSION_STALE) begin
                  // A replay cannot allocate under the already invalidated plan.
                  lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_STALE, 64'd0);
                end else if (lt_state[lf_t] == LT_HELD && lt_faulted[lf_t]) begin
                  // Rejected hold: the exact duplicate re-ACKs the same FAULT; anything
                  // else (fresh txn, other slot/generation) must NOT rebind — the exact
                  // TICKET_ROLLBACK is the only release proof. No state changes.
                  lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_FAULT, 64'd0);
                end else if (lt_state[lf_t] == LT_HELD) begin
                  if (lt_hold_txn[lf_t] == p5 && p6 == 64'(lt_slot[lf_t]) &&
                      p7 == lt_gen[lf_t] && p8 == lt_pool_ver[lf_t]) begin
                    lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_HOLD_READY, 64'd0); // replay
                  end else begin
                    lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_FAULT, 64'd0);
                  end
                end else begin
                  lf_ack(p5, CN_MSG_TASK_COMMIT, CN_OUT_FAULT, 64'd0);
                end
              end
              CN_MSG_TASK_COMMIT_FINAL: begin
                lf_t = lf_find_ticket(p0, p2, p3, p4);
                if (p1 != p0 || lf_t < 0 || lt_faulted[lf_t]) begin
                  lf_ack(p5, CN_MSG_TASK_COMMIT_FINAL, CN_OUT_FAULT, 64'd0);
                end else if (lf_frozen) begin
                  lf_ack(p5, CN_MSG_TASK_COMMIT_FINAL, CN_OUT_WAIT, CN_WHY_CONTROL);
                end else if (lt_state[lf_t] == LT_HELD) begin
                  if (p5 != lt_hold_txn[lf_t] || p6 != 64'(lt_slot[lf_t]) || p7 != lt_gen[lf_t]) begin
                    // Full-width match: an old, aliased (slot 65 vs 1) or mismatched
                    // transaction must never authorize the hold.
                    lf_ack(p5, CN_MSG_TASK_COMMIT_FINAL, CN_OUT_STALE, 64'd0);
                  end else if (lf_fail_final_armed && !lf_fail_final_done) begin
                    // Valid grant crossed P1; the injected local installation fails AFTER
                    // the irrevocable authorization: same owner, FAULT (never success, no
                    // phase signal). Closure is cancel -> drain -> RETIRE -> terminal.
                    lf_fail_final_done    <= 1'b1;
                    lt_state[lf_t]        <= LT_DRAIN;
                    lt_final_faulted[lf_t] <= 1'b1;
                    lt_final_txn[lf_t]    <= p5;
                    lt_drain_seq[lf_t]    <= lf_drain_ctr + 64'd1;
                    lf_drain_ctr          <= lf_drain_ctr + 64'd1;
                    lf_ack(p5, CN_MSG_TASK_COMMIT_FINAL, CN_OUT_FAULT, 64'd0);
                  end else begin
                    lt_state[lf_t]        <= LT_DRAIN;
                    lt_final_faulted[lf_t] <= 1'b0;
                    lt_final_txn[lf_t]    <= p5;
                    lt_drain_seq[lf_t]    <= lf_drain_ctr + 64'd1;
                    lf_drain_ctr          <= lf_drain_ctr + 64'd1;
                    lf_ack(p5, CN_MSG_TASK_COMMIT_FINAL, CN_OUT_ACCEPTED, 64'd0);
                    if (!lt_phase_owed[lf_t]) begin
                      lf_retire_ready(lf_t, lf_drain_ctr + 64'd1);
                    end else begin
                      lt_rr_pending[lf_t] <= 1'b1; // gated on the fired input phase signal
                    end
                  end
                end else if (lt_state[lf_t] == LT_DRAIN) begin
                  if (lt_final_txn[lf_t] == p5 && p6 == 64'(lt_slot[lf_t]) && p7 == lt_gen[lf_t]) begin
                    if (lt_final_faulted[lf_t]) begin
                      lf_ack(p5, CN_MSG_TASK_COMMIT_FINAL, CN_OUT_FAULT, 64'd0); // failed install
                    end else begin
                      lf_ack(p5, CN_MSG_TASK_COMMIT_FINAL, CN_OUT_ACCEPTED, 64'd0); // replay
                    end
                  end else begin
                    lf_ack(p5, CN_MSG_TASK_COMMIT_FINAL, CN_OUT_FAULT, 64'd0);
                  end
                end else begin
                  lf_ack(p5, CN_MSG_TASK_COMMIT_FINAL, CN_OUT_FAULT, 64'd0); // not held
                end
              end
              CN_MSG_TICKET_ROLLBACK: begin
                lf_t = lf_find_ticket(p0, p2, p3, p4);
                if (p1 != p0) begin
                  lf_ack(p5, CN_MSG_TICKET_ROLLBACK, CN_OUT_FAULT, 64'd0);
                end else if (lf_t < 0) begin
                  lf_ack(p5, CN_MSG_TICKET_ROLLBACK, CN_OUT_STALE, 64'd0);
                end else if (lt_state[lf_t] == LT_PLANNED || lt_state[lf_t] == LT_VERSION_STALE) begin
                  if (lt_plan_txn[lf_t] != p5 || (lt_state[lf_t] == LT_VERSION_STALE &&
                      (p6 != 64'(lt_slot[lf_t]) || p7 != lt_gen[lf_t]))) begin
                    lf_ack(p5, CN_MSG_TICKET_ROLLBACK, CN_OUT_STALE, 64'd0);
                  end else begin
                    lt_state[lf_t] <= LT_FREE; // release commits at this edge...
                    lf_ack(p5, CN_MSG_TICKET_ROLLBACK, CN_OUT_COMPENSATED, 64'd0); // ...staged after
                  end
                end else if (lt_state[lf_t] == LT_HELD) begin
                  if (lt_hold_txn[lf_t] != p5 || p6 != 64'(lt_slot[lf_t]) || p7 != lt_gen[lf_t]) begin
                    // An old or aliased transaction must never release a newer hold.
                    lf_ack(p5, CN_MSG_TICKET_ROLLBACK, CN_OUT_STALE, 64'd0);
                  end else begin
                    live_ctx_gen[lt_slot[lf_t]] <= lt_gen[lf_t] + 64'd1; // reuse needs newer gen
                    lf_pool_version <= lf_pool_version + 64'd1;
                    lt_state[lf_t] <= LT_FREE;
                    lf_ack(p5, CN_MSG_TICKET_ROLLBACK, CN_OUT_COMPENSATED, 64'd0);
                  end
                end else begin
                  // Post-FINAL authorization is irreversible: cancel/drain is the only path.
                  lf_ack(p5, CN_MSG_TICKET_ROLLBACK, CN_OUT_FAULT, 64'd0);
                end
              end
              CN_MSG_CANCEL_TASK: begin
                lf_t = lf_find_ticket(p0, p2, p3, p4);
                // CANCEL_TASK carries NO txn_id: ctx_slot is payload word 5 and
                // ctx_generation word 6 (CN_OFF_CANCEL_TASK_CTX_SLOT/_CTX_GENERATION).
                if (lf_t >= 0 && p1 == p0 && lt_state[lf_t] == LT_DRAIN &&
                    p5 == 64'(lt_slot[lf_t]) && p6 == lt_gen[lf_t] && !lt_cancel[lf_t]) begin
                  // Authorized work only, full-width match: mark the retirement terminal
                  // status. A pre-FINAL cancel finds no live Task (C015): no promotion, no
                  // terminal — compensation via TICKET_ROLLBACK is the only close.
                  lt_cancel[lf_t] <= 1'b1;
                  if (lt_final_faulted[lf_t]) begin
                    // A failed installation owes no phase and never announced readiness:
                    // the matched cancel IS its local drain, so announce RETIRE_READY now.
                    lt_rr_pending[lf_t] <= 1'b0;
                    lt_phase_owed[lf_t] <= 1'b0;
                    lf_retire_ready(lf_t, lt_drain_seq[lf_t]);
                  end
                end
              end
              CN_MSG_RETIRE: begin
                lf_t = lf_find_ticket(p0, p2, p3, p4);
                if (lf_t >= 0 && lt_state[lf_t] == LT_DRAIN && lt_rr_fired[lf_t] &&
                    p5 == 64'(cfg_tile_id) && p6 == 64'(lt_slot[lf_t]) &&
                    p7 == lt_gen[lf_t] && p8 == lt_arena[lf_t]) begin
                  // Observed READY + full immutable identity matched: release the slot for
                  // accounting, but the ticket stays resident (COMPLETION) until its
                  // terminal actually FIRES — no early slot reuse, no silent drop.
                  live_ctx_gen[lt_slot[lf_t]] <= lt_gen[lf_t] + 64'd1;
                  lf_pool_version <= lf_pool_version + 64'd1;
                  lt_state[lf_t] <= LT_COMPLETION;
                  lf_completion(lf_t,
                                lt_final_faulted[lf_t] ? CN_TASK_ERROR :
                                (lt_cancel[lf_t] ? CN_TASK_RESET : CN_TASK_DONE),
                                lt_final_faulted[lf_t] ? lt_arena[lf_t] : 64'd0);
                end
                // Unknown/early (READY never observed)/mismatched RETIRE: no local effect,
                // no fabricated terminal.
              end
              CN_MSG_RUN_BEGIN: begin
                if ((!lf_epoch_seen || p0 > lf_epoch) && lf_quiet() && !lf_frozen && !lf_parked_drain) begin
                  lf_epoch        <= p0;
                  lf_epoch_seen   <= 1'b1;
                  lf_frozen       <= 1'b0;
                  lf_parked_drain <= 1'b0;
                end
              end
              CN_MSG_RESET_PHASE_REQ: begin
                if (lf_epoch_seen && p1 == lf_epoch) begin
                  if (p0[31:0] == CN_PHASE_FREEZE) begin
                    lf_frozen <= 1'b1; // tickets are never dropped reset-side
                  end
                  if (p0[31:0] == CN_PHASE_RESUME) begin
                    lf_frozen <= 1'b0;
                  end
                  if (p0[31:0] == CN_PHASE_DRAIN && !lf_quiet()) begin
                    lf_parked_drain <= 1'b1; // answer once, after every ticket closed
                    lf_parked_epoch <= p1;
                  end else if (p0[31:0] == CN_PHASE_DRAIN) begin
                    lf_reset_phase_ack(p0, p1, 64'd0, CN_DRAIN_EVIDENCE_FULL);
                  end else begin
                    // pending_to_reset counts local event-table entries awaiting reset and
                    // only MARK_EVENTS may carry it (C034 service ACK_MARK_COUNT). This
                    // fixture keeps no local event entries, so every immediate reply
                    // carries the actual value 0; residency still gates DRAIN below.
                    lf_reset_phase_ack(p0, p1, 64'd0, 64'd0);
                  end
                end
                // Stale-epoch or pre-RUN_BEGIN request: no reply (fail closed, C034 rule).
              end
              default: begin
                produce = 1'b0;
              end
            endcase
          end else begin
            case (in_msg_type[31:0])
              CN_MSG_QUERY_CAPS: begin
                produce    = 1'b1;
                next_words = 5'(CN_HEADER_WORDS + CN_WORDS_CAPS_RESP);
                next_frame = set_word(next_frame, 0, CN_MSG_CAPS_RESP);
                next_frame = set_word(next_frame, 1, CN_WORDS_CAPS_RESP);
                next_frame = set_word(next_frame, 2, t2g_seq[c]);
                next_frame = set_word(next_frame, 4 + CN_OFF_CAPS_RESP_TXN_ID, p0 ^ corrupt_mask);
                next_frame = set_word(next_frame, 4 + CN_OFF_CAPS_RESP_CONTEXT_COUNT, 64'd2);
                next_frame = set_word(next_frame, 4 + CN_OFF_CAPS_RESP_PROGRAM_SLOTS, 64'd2);
                next_frame = set_word(next_frame, 4 + CN_OFF_CAPS_RESP_STREAM_SUPPORTED, 64'd1);
              end
              CN_MSG_TASK_PLAN: begin
                produce    = 1'b1;
                next_words = 5'(CN_HEADER_WORDS + CN_WORDS_ACK);
                next_frame = set_word(next_frame, 0, CN_MSG_ACK);
                next_frame = set_word(next_frame, 1, CN_WORDS_ACK);
                next_frame = set_word(next_frame, 2, t2g_seq[c]);
                next_frame = set_word(next_frame, 4 + CN_OFF_ACK_TXN_ID, p5 ^ corrupt_mask);
                next_frame = set_word(next_frame, 4 + CN_OFF_ACK_ACK_FOR_MESSAGE_TYPE, CN_MSG_TASK_PLAN);
                next_frame = set_word(next_frame, 4 + CN_OFF_ACK_OP_SEQ, 64'd0);
                next_frame = set_word(next_frame, 4 + CN_OFF_ACK_OUTCOME, 64'd1); // ACCEPTED
                next_frame = set_word(next_frame, 4 + CN_OFF_ACK_WAIT_REASON, 64'd0);
              end
              CN_MSG_TASK_COMMIT,
              CN_MSG_TASK_COMMIT_FINAL: begin
                produce     = 1'b1;
                next_words  = 5'(CN_HEADER_WORDS + CN_WORDS_ACK);
                if (p6 < CN_CONTEXT_SLOTS && p7 < live_ctx_gen[p6[5:0]]) begin
                  ack_outcome = 64'd4; // STALE
                end else begin
                  if (in_msg_type[31:0] == CN_MSG_TASK_COMMIT && p6 < CN_CONTEXT_SLOTS) begin
                    live_ctx_gen[p6[5:0]] <= p7;
                    ack_outcome = 64'd0; // HOLD_READY
                  end else begin
                    ack_outcome = 64'd1; // ACCEPTED
                  end
                end
                next_frame = set_word(next_frame, 0, CN_MSG_ACK);
                next_frame = set_word(next_frame, 1, CN_WORDS_ACK);
                next_frame = set_word(next_frame, 2, t2g_seq[c]);
                next_frame = set_word(next_frame, 4 + CN_OFF_ACK_TXN_ID, p5 ^ corrupt_mask);
                next_frame = set_word(next_frame, 4 + CN_OFF_ACK_ACK_FOR_MESSAGE_TYPE, in_msg_type);
                next_frame = set_word(next_frame, 4 + CN_OFF_ACK_OP_SEQ, 64'd0);
                next_frame = set_word(next_frame, 4 + CN_OFF_ACK_OUTCOME, ack_outcome);
                next_frame = set_word(next_frame, 4 + CN_OFF_ACK_WAIT_REASON, 64'd0);
              end
              CN_MSG_RESET_PHASE_REQ: begin
                produce    = 1'b1;
                next_words = 5'(CN_HEADER_WORDS + CN_WORDS_RESET_PHASE_ACK);
                next_frame = set_word(next_frame, 0, CN_MSG_RESET_PHASE_ACK);
                next_frame = set_word(next_frame, 1, CN_WORDS_RESET_PHASE_ACK);
                next_frame = set_word(next_frame, 2, t2g_seq[c]);
                next_frame = set_word(next_frame, 4 + CN_OFF_RESET_PHASE_ACK_PHASE, p0 ^ corrupt_mask);
                next_frame = set_word(next_frame, 4 + CN_OFF_RESET_PHASE_ACK_RUN_GENERATION, p1);
                next_frame = set_word(next_frame, 4 + CN_OFF_RESET_PHASE_ACK_PENDING_TO_RESET, 64'd0);
                next_frame = set_word(next_frame, 4 + CN_OFF_RESET_PHASE_ACK_EVIDENCE_MASK, 64'd15);
              end
              CN_MSG_EVENT_NOTIFY: begin
                produce    = 1'b1;
                next_words = 5'(CN_HEADER_WORDS + CN_WORDS_POST_EVENT);
                next_frame = set_word(next_frame, 0, CN_MSG_POST_EVENT);
                next_frame = set_word(next_frame, 1, CN_WORDS_POST_EVENT);
                next_frame = set_word(next_frame, 2, t2g_seq[c]);
                next_frame = set_word(next_frame, 4 + 0, p0 ^ corrupt_mask);
                next_frame = set_word(next_frame, 4 + 1, p1);
                next_frame = set_word(next_frame, 4 + 2, p2);
                next_frame = set_word(next_frame, 4 + 3, p3);
                next_frame = set_word(next_frame, 4 + 4, p4);
                next_frame = set_word(next_frame, 4 + 5, p5);
                next_frame = set_word(next_frame, 4 + 6, p6);
                next_frame = set_word(next_frame, 4 + 7, p7);
              end
              CN_MSG_PROGRAM_INSTALL: begin
                produce    = 1'b1;
                next_words = 5'(CN_HEADER_WORDS + CN_WORDS_PROGRAM_READ_REQ);
                next_frame = set_word(next_frame, 0, CN_MSG_PROGRAM_READ_REQ);
                next_frame = set_word(next_frame, 1, CN_WORDS_PROGRAM_READ_REQ);
                next_frame = set_word(next_frame, 2, t2g_seq[c]);
                next_frame = set_word(next_frame, 4 + 0, p0 ^ corrupt_mask);
                next_frame = set_word(next_frame, 4 + 1, p1);
                next_frame = set_word(next_frame, 4 + 2, p2);
                next_frame = set_word(next_frame, 4 + 3, p3);
                next_frame = set_word(next_frame, 4 + 4, p4);
                next_frame = set_word(next_frame, 4 + 5, p5);
                next_frame = set_word(next_frame, 4 + 6, p6);
              end
              CN_MSG_PROGRAM_READ_RESP: begin
                produce    = 1'b1;
                next_words = 5'(CN_HEADER_WORDS + CN_WORDS_PROGRAM_INSTALL_ACK);
                next_frame = set_word(next_frame, 0, CN_MSG_PROGRAM_INSTALL_ACK);
                next_frame = set_word(next_frame, 1, CN_WORDS_PROGRAM_INSTALL_ACK);
                next_frame = set_word(next_frame, 2, t2g_seq[c]);
                next_frame = set_word(next_frame, 4 + 0, p0 ^ corrupt_mask);
                next_frame = set_word(next_frame, 4 + 1, p1);
                next_frame = set_word(next_frame, 4 + 2, p2);
                next_frame = set_word(next_frame, 4 + 3, p3);
                next_frame = set_word(next_frame, 4 + 4, p4);
                next_frame = set_word(next_frame, 4 + 5, p5);
                next_frame = set_word(next_frame, 4 + 6, p6);
                next_frame = set_word(next_frame, 4 + 7, p7); // installed_len = hbm_chunk_len
                next_frame = set_word(next_frame, 4 + 8, p2);
                next_frame = set_word(next_frame, 4 + 9, p3);
                next_frame = set_word(next_frame, 4 + 10, p4);
                next_frame = set_word(next_frame, 4 + 11, p5);
              end
              CN_MSG_PROGRAM_INVALIDATE: begin
                produce    = 1'b1;
                next_words = 5'(CN_HEADER_WORDS + CN_WORDS_INVALIDATE_ACK);
                next_frame = set_word(next_frame, 0, CN_MSG_INVALIDATE_ACK);
                next_frame = set_word(next_frame, 1, CN_WORDS_INVALIDATE_ACK);
                next_frame = set_word(next_frame, 2, t2g_seq[c]);
                next_frame = set_word(next_frame, 4 + 0, p0 ^ corrupt_mask);
                next_frame = set_word(next_frame, 4 + 1, p1);
                next_frame = set_word(next_frame, 4 + 2, p2);
                next_frame = set_word(next_frame, 4 + 3, p3);
                next_frame = set_word(next_frame, 4 + 4, p4);
                next_frame = set_word(next_frame, 4 + 5, p5);
                next_frame = set_word(next_frame, 4 + 6, p6);
              end
              CN_MSG_MEM_RSP: begin
                produce    = 1'b1;
                next_words = 5'(CN_HEADER_WORDS + CN_WORDS_TERMINAL_ACK);
                next_frame = set_word(next_frame, 0, CN_MSG_TERMINAL_ACK);
                next_frame = set_word(next_frame, 1, CN_WORDS_TERMINAL_ACK);
                next_frame = set_word(next_frame, 2, t2g_seq[c]);
                next_frame = set_word(next_frame, 4 + CN_OFF_TERMINAL_ACK_TXN_ID, p0 ^ corrupt_mask);
                next_frame = set_word(next_frame, 4 + CN_OFF_TERMINAL_ACK_SLICE_ID, p1);
                next_frame = set_word(next_frame, 4 + CN_OFF_TERMINAL_ACK_SLICE_COUNT, 64'd1);
                next_frame = set_word(next_frame, 4 + CN_OFF_TERMINAL_ACK_FINAL_STATE, 64'd0);
              end
              default: begin
                produce = 1'b0;
              end
            endcase
          end

          if (produce) begin
            t2g_valid[c] <= 1'b1;
            t2g_words[c] <= next_words;
            t2g_frame[c] <= next_frame;
            t2g_seq[c]   <= t2g_seq[c] + 64'd1;
          end
        end
      end

      // A parked DRAIN is answered only after the last local owner and EVENT beat close.
      if (cfg_lifecycle && lf_parked_drain && lf_epoch_seen && lf_parked_epoch == lf_epoch) begin
        if (lf_quiet()) begin
          lf_parked_drain <= 1'b0;
          lf_reset_phase_ack(64'(CN_PHASE_DRAIN), lf_parked_epoch, 64'd0, CN_DRAIN_EVIDENCE_FULL);
        end
      end
    end
  end

endmodule
